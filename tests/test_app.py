"""The desktop program, without a screen or a sound card.

The audio path is Session -> StreamProcessor -> backend, and only the backend
knows what a device is.  Swapping it for one that reads an array is what lets
the whole program be exercised here: not a mock of the engine, the engine --
the same callback the microphone drives, on a different clock.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import natvox
from natvox import api
from natvox.app.backend import AudioUnavailable, OfflineBackend, list_devices
from natvox.app.core import MachineReport, Settings, Studio
from natvox.realtime import StreamProcessor


@pytest.fixture
def voice_audio(sample_rate):
    t = np.arange(int(1.5 * sample_rate)) / sample_rate
    return 0.3 * np.sin(2 * np.pi * 130 * t) + 0.1 * np.sin(2 * np.pi * 260 * t)


class TestSettings:
    def test_they_survive_a_round_trip(self, tmp_path):
        path = tmp_path / "settings.json"
        Settings(voice="female", overrides={"pitch_semitones": 6.0},
                 block_size=128, remote_url="ws://x/y").save(path)
        back = Settings.load(path)
        assert back.voice == "female"
        assert back.overrides == {"pitch_semitones": 6.0}
        assert back.block_size == 128 and back.remote_url == "ws://x/y"

    def test_a_missing_file_is_defaults_not_an_error(self, tmp_path):
        assert Settings.load(tmp_path / "nope.json") == Settings()

    @pytest.mark.parametrize("content", ["{", "[]", "null", "not json at all"])
    def test_a_damaged_file_is_defaults_not_a_crash(self, tmp_path, content):
        """A program that will not start because of its own old config is
        worse than one that starts with defaults."""
        path = tmp_path / "settings.json"
        path.write_text(content)
        assert Settings.load(path) == Settings()

    def test_a_key_this_version_does_not_know_is_ignored(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"voice": "female", "from_the_future": 1}))
        assert Settings.load(path).voice == "female"


class TestProfile:
    def test_a_voice_and_its_overrides(self):
        studio = Studio(Settings(voice="female_soft"))
        assert studio.current()["pitch_semitones"] == 4.5
        studio.adjust(pitch_semitones=6.0)
        assert studio.current()["pitch_semitones"] == 6.0
        assert studio.current()["breathiness"] == 0.08      # the rest is the voice

    def test_changing_voice_drops_the_overrides(self):
        """They described the old voice.  Carrying them over silently is how a
        preset comes out sounding like the last one."""
        studio = Studio(Settings(voice="female_soft"))
        studio.adjust(pitch_semitones=6.0)
        studio.set_voice("female_bright")
        assert studio.current()["pitch_semitones"] == 7.5

    def test_an_impossible_setting_is_refused_and_changes_nothing(self):
        studio = Studio(Settings(voice="female"))
        with pytest.raises(api.ParameterError):
            studio.adjust(pitch_semitones=99.0)
        assert studio.current()["pitch_semitones"] == 7.0

    def test_an_unknown_voice_is_refused(self):
        with pytest.raises(api.ParameterError):
            Studio(Settings()).set_voice("sultry")


class TestSelfTest:
    def test_it_measures_this_machine(self):
        report = Studio(Settings(voice="female")).self_test(256, seconds=0.3)
        assert report.block_size == 256
        assert report.deadline_ms == pytest.approx(256 / 48 , abs=0.01)
        assert 0 < report.mean_ms <= report.worst_ms
        assert report.blocks > 10
        assert report.summary()

    @pytest.mark.parametrize("worst,comfortable,usable", [
        (1.0, True, True), (3.0, False, True), (5.0, False, False)])
    def test_the_verdict_is_about_the_worst_block_not_the_average(
            self, worst, comfortable, usable):
        """The average never drops out.  The worst block is what clicks."""
        report = MachineReport(256, 5.33, worst, 0.5, 100)
        assert report.comfortable is comfortable
        assert report.usable is usable


class TestOfflineBackend:
    def test_it_produces_what_the_offline_converter_does(self, voice_audio, sample_rate):
        """The backend must not be a second implementation of anything."""
        session = api.Session(sample_rate, "female")
        processor = StreamProcessor(session)
        backend = OfflineBackend(voice_audio, 256)
        backend.start(processor)
        assert backend.wait(30)
        served = backend.output()[session.latency_samples:]

        fresh = api.Session(sample_rate, "female")
        direct = np.concatenate([fresh.process(voice_audio[i:i + 256])
                                 for i in range(0, voice_audio.size, 256)])
        direct = direct[fresh.latency_samples:]
        n = min(served.size, direct.size)
        assert np.max(np.abs(served[:n] - direct[:n])) < 1e-6

    def test_paced_mode_runs_at_about_real_time(self, voice_audio, sample_rate):
        processor = StreamProcessor(api.Session(sample_rate, "female_soft"))
        backend = OfflineBackend(voice_audio[:sample_rate // 2], 256, realtime=True)
        started = time.perf_counter()
        backend.start(processor)
        assert backend.wait(20)
        elapsed = time.perf_counter() - started
        assert 0.4 < elapsed < 1.2, f"half a second of audio took {elapsed:.2f}s"

    def test_looping_keeps_going_until_stopped(self, voice_audio, sample_rate):
        processor = StreamProcessor(api.Session(sample_rate, "off"))
        backend = OfflineBackend(voice_audio[:4096], 256, realtime=True, loop=True)
        backend.start(processor)
        time.sleep(0.3)
        assert backend.running
        backend.stop()
        assert not backend.running
        assert backend.output().size > 4096

    def test_a_failure_inside_the_callback_is_reported_not_swallowed(self, sample_rate):
        class Broken:
            sample_rate = 48000
            latency_samples = 0
            def process(self, block): raise ValueError("no")
            def reset(self): pass

        backend = OfflineBackend(np.zeros(1024), 256)
        backend.start(StreamProcessor(Broken()))
        assert backend.wait(10)
        assert isinstance(backend.error, ValueError)


class TestDevices:
    def test_listing_works_or_says_what_to_install(self):
        """There is no sound card here, so the list is empty -- but the code
        path that reads it has to run, and its failure has to be a sentence."""
        try:
            devices = list_devices()
        except AudioUnavailable as exc:
            assert "install" in str(exc).lower()
            return
        assert isinstance(devices, list)
        for device in devices:
            assert device.label


class TestRunning:
    def _run(self, studio, audio, block=256, realtime=False, **kwargs):
        backend = OfflineBackend(audio, block, realtime=realtime, **kwargs)
        studio.start(backend)
        return backend

    def test_start_and_stop(self, voice_audio):
        studio = Studio(Settings(voice="female"))
        backend = self._run(studio, voice_audio, realtime=True, loop=True)
        assert studio.running
        studio.stop()
        assert not studio.running
        assert backend.error is None

    def test_the_meters_show_what_went_through(self, voice_audio):
        studio = Studio(Settings(voice="female"))
        backend = self._run(studio, voice_audio)
        backend.wait(30)
        metrics = studio.metrics()
        assert metrics.input_peak > 0.1 and metrics.output_peak > 0.1
        assert metrics.voiced and abs(metrics.f0_hz - 130) < 8
        assert metrics.latency_ms > 0
        assert metrics.dropouts == 0 or metrics.load_peak > 1.0
        studio.stop()

    def test_the_meters_decay_rather_than_resetting_when_read(self, voice_audio,
                                                              sample_rate):
        """Two things paint a meter -- the timer and whatever a test does --
        and a peak that clears on read means the second one sees silence."""
        studio = Studio(Settings(voice="off"))
        backend = self._run(studio, voice_audio)
        backend.wait(30)
        first = studio.metrics().input_peak
        assert first > 0.1
        assert studio.metrics().input_peak == first     # reading changes nothing
        studio.stop()

    def test_the_meters_fall_when_the_sound_stops(self, voice_audio, sample_rate):
        studio = Studio(Settings(voice="off"))
        silence = np.concatenate([voice_audio, np.zeros(sample_rate)])
        backend = self._run(studio, silence)
        backend.wait(30)
        assert studio.metrics().input_peak < 0.01
        studio.stop()

    def test_metrics_before_starting_are_empty_rather_than_an_error(self):
        assert Studio(Settings()).metrics().running is False

    def test_a_setting_changed_while_running_reaches_the_audio(self, voice_audio,
                                                               sample_rate):
        studio = Studio(Settings(voice="female_soft"))
        backend = OfflineBackend(voice_audio, 256, realtime=True, loop=True)
        studio.start(backend)
        time.sleep(0.25)
        studio.adjust(pitch_semitones=8.0)
        time.sleep(0.45)
        studio.stop()
        assert studio.last_error is None
        out = backend.output()
        assert out.size > sample_rate // 4
        assert np.all(np.isfinite(out))

    def test_the_recorder_keeps_the_last_few_seconds(self, voice_audio):
        studio = Studio(Settings(voice="off"))
        backend = self._run(studio, voice_audio)
        backend.wait(30)
        recent = studio.recent_input()
        assert recent.size == voice_audio.size
        assert np.max(np.abs(recent[-2000:] - voice_audio[-2000:])) < 1e-9
        studio.stop()

    def test_saving_a_wav_round_trips(self, voice_audio, tmp_path):
        from natvox.server import read_wav

        studio = Studio(Settings(voice="female_soft"))
        path = studio.save_wav(tmp_path / "out.wav", studio.convert(voice_audio))
        audio, rate = read_wav(Path(path).read_bytes())
        assert rate == studio.settings.sample_rate
        assert audio.size == voice_audio.size


class TestTheABComparison:
    """The control the whole judgement rests on, so it gets its own class."""

    def test_the_original_is_delayed_by_exactly_the_engine_s_latency(self, sample_rate):
        """Not approximately.  A dry signal a few samples out comb-filters
        against the processed one and makes the original sound worse."""
        noise = np.random.default_rng(0).standard_normal(sample_rate) * 0.2
        session = api.Session(sample_rate, "female")
        processor = StreamProcessor(session, dry_wet=0.0)
        out = np.concatenate([processor(noise[i:i + 256], 256)[:, 0]
                              for i in range(0, noise.size, 256)]).astype(np.float64)
        lag = session.latency_samples
        window = slice(20000, 30000)
        expected = noise[window.start - lag:window.stop - lag]
        assert np.max(np.abs(out[window] - expected)) < 1e-6

    def test_pressing_it_does_not_play_silence_first(self, sample_rate):
        """The delay line is fed on every block.  Filling it only while the
        original is being listened to meant the first 60 ms of every A/B were
        the silence it had been holding."""
        t = np.arange(sample_rate) / sample_rate
        tone = 0.3 * np.sin(2 * np.pi * 130 * t)
        processor = StreamProcessor(api.Session(sample_rate, "female"))
        out = []
        switch = 256 * 90
        for i in range(0, tone.size, 256):
            if i == switch:
                processor.dry_wet = 0.0
            out.append(processor(tone[i:i + 256], 256)[:, 0])
        y = np.concatenate(out).astype(np.float64)
        after = y[switch:switch + int(0.05 * sample_rate)]
        assert np.sqrt(np.mean(after ** 2)) > 0.1, "the A/B started with silence"

    def test_it_does_not_click(self, sample_rate):
        t = np.arange(sample_rate) / sample_rate
        tone = 0.3 * np.sin(2 * np.pi * 130 * t)
        processor = StreamProcessor(api.Session(sample_rate, "female"))
        out, switch = [], 256 * 90
        for i in range(0, tone.size, 256):
            if i == switch:
                processor.dry_wet = 0.0
            out.append(processor(tone[i:i + 256], 256)[:, 0])
        y = np.concatenate(out).astype(np.float64)
        at_switch = np.max(np.abs(np.diff(y[switch - 100:switch + 2000])))
        elsewhere = np.max(np.abs(np.diff(y[10000:20000])))
        assert at_switch < elsewhere * 1.5, (at_switch, elsewhere)

    def test_the_studio_exposes_it(self, voice_audio):
        studio = Studio(Settings(voice="female"))
        backend = OfflineBackend(voice_audio, 256, realtime=True, loop=True)
        studio.start(backend)
        assert studio.bypass is False
        studio.bypass = True
        assert studio.bypass is True
        studio.stop()
