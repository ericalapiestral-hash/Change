"""End-to-end checks on the command line."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from natvox import presets
from natvox.cli import main


@pytest.fixture
def wav(tmp_path, utterance, sample_rate):
    audio, _ = utterance
    path = tmp_path / "in.wav"
    sf.write(path, audio, sample_rate)
    return path


class TestProcess:
    def test_writes_audio_of_the_same_length_and_rate(self, wav, tmp_path, capsys):
        out = tmp_path / "out.wav"
        assert main(["process", str(wav), str(out), "-p", "male_to_female"]) == 0
        original, rate = sf.read(wav)
        converted, out_rate = sf.read(out)
        assert out_rate == rate
        assert converted.shape == original.shape
        assert np.max(np.abs(converted)) > 0.01
        assert "latency" in capsys.readouterr().out

    def test_explicit_shifts_override_the_preset(self, wav, tmp_path):
        from evaluate import pitch_track

        out = tmp_path / "out.wav"
        assert main(["process", str(wav), str(out),
                     "-p", "deeper", "--pitch", "4", "--formant", "2"]) == 0
        original, rate = sf.read(wav)
        converted, _ = sf.read(out)
        _, dry = pitch_track(original, rate)
        _, wet = pitch_track(converted, rate)
        n = min(dry.size, wet.size)
        both = (dry[:n] > 0) & (wet[:n] > 0)
        achieved = np.median(wet[:n][both] / dry[:n][both])
        assert achieved > 1.0  # up, not down as "deeper" alone would be

    def test_block_size_does_not_change_the_file(self, wav, tmp_path):
        outputs = []
        for block in (256, 4096):
            out = tmp_path / f"out{block}.wav"
            assert main(["process", str(wav), str(out),
                         "-p", "younger", "--block", str(block)]) == 0
            outputs.append(sf.read(out)[0])
        assert np.max(np.abs(outputs[0] - outputs[1])) < 1e-3

    def test_stereo_survives_the_round_trip(self, tmp_path, sample_rate):
        rng = np.random.default_rng(0)
        stereo = rng.normal(0, 0.1, (sample_rate // 2, 2))
        src = tmp_path / "stereo.wav"
        sf.write(src, stereo, sample_rate)
        out = tmp_path / "out.wav"
        assert main(["process", str(src), str(out), "-p", "brighter"]) == 0
        assert sf.read(out)[0].shape == stereo.shape

    def test_warns_when_settings_will_cost_naturalness(self, wav, tmp_path, capsys):
        out = tmp_path / "out.wav"
        assert main(["process", str(wav), str(out), "--pitch", "11"]) == 0
        assert "naturalness" in capsys.readouterr().err

    def test_unknown_preset_is_rejected(self, wav, tmp_path):
        with pytest.raises(SystemExit):
            main(["process", str(wav), str(tmp_path / "o.wav"), "-p", "nope"])


class TestPresets:
    def test_lists_every_preset(self, capsys):
        assert main(["presets"]) == 0
        out = capsys.readouterr().out
        for name in presets.names():
            assert name in out


class TestDevices:
    @pytest.mark.parametrize("argv", [["devices"], ["app", "--devices"]])
    def test_reports_clearly_when_there_is_nothing_to_list(self, argv, capsys):
        """Headless machines have no PortAudio, or have it and no hardware.

        Both are ordinary on a build runner and neither may look like a crash,
        so whichever one happens has to leave behind a sentence saying what to
        do about it.
        """
        code = main(argv)
        captured = capsys.readouterr()
        if code != 0:
            assert any(word in captured.err
                       for word in ("realtime", "install", "Plug something in"))

    def test_the_listing_puts_the_best_host_api_first(self, capsys, monkeypatch):
        """PortAudio offers one microphone once per host API and its own order
        puts the worst copy first, which is the whole reason this sorts."""
        from natvox.app import backend as backend_module

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from test_app import FakeSoundDevice, fake_device

        fake = FakeSoundDevice(
            [fake_device("Yeti", 0, default_low_input_latency=0.09),
             fake_device("Yeti", 1, default_low_input_latency=0.003)],
            ["MME", "Windows WASAPI"])
        monkeypatch.setattr(backend_module, "_sounddevice", lambda: fake)
        assert main(["devices"]) == 0
        lines = [l for l in capsys.readouterr().out.splitlines() if "Yeti" in l]
        assert "WASAPI" in lines[0] and "MME" in lines[1]
        assert "3.0ms" in lines[0] and "90.0ms" in lines[1]


class TestLoopback:
    def test_it_says_what_went_wrong_rather_than_raising(self, capsys):
        """No sound card here, so this is the real failure path."""
        assert main(["app", "--loopback", "--attempts", "1"]) == 1
        assert capsys.readouterr().err.strip()

    def test_it_reports_the_measurement_and_what_the_converter_adds(
            self, capsys, monkeypatch):
        from natvox.app import loopback

        monkeypatch.setattr(
            loopback, "through_devices",
            lambda *a, **k: loopback.RoundTrip(48000, 256, [100.0], 10.0))
        assert main(["app", "--loopback", "-p", "female"]) == 0
        out = capsys.readouterr().out
        assert "round trip" in out and "100.0 ms" in out
        assert "the converter adds" in out

    def test_hearing_nothing_back_is_a_failure_not_a_success(
            self, capsys, monkeypatch):
        """Zero would be a wonderful round trip and is what silence looks
        like, so it has to be reported as nothing having come back."""
        from natvox.app import loopback

        monkeypatch.setattr(loopback, "through_devices",
                            lambda *a, **k: loopback.RoundTrip(48000, 256, []))
        assert main(["app", "--loopback"]) == 1
        assert "loops round" in capsys.readouterr().out

    def test_exclusive_mode_off_windows_says_why(self, capsys):
        assert main(["live", "--exclusive"]) == 1
        assert "WASAPI" in capsys.readouterr().err
