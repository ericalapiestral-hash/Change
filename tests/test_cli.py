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
        err = capsys.readouterr().err
        assert "note:" in err
        # It warns, and it does not overstate: measured on this implementation
        # the degradation past the threshold is gradual rather than a cliff.
        assert "gradual" in err or "listen" in err

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

    def test_the_listing_shows_what_not_asking_would_cost(self, capsys, monkeypatch):
        """The gap between the two columns is the delay a program pays for
        leaving PortAudio's latency argument out."""
        from natvox.app import backend as backend_module

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from test_app import FakeSoundDevice, fake_device

        fake = FakeSoundDevice([fake_device("Yeti", 0,
                                            default_low_input_latency=0.003)],
                               ["Windows WASAPI"])
        monkeypatch.setattr(backend_module, "_sounddevice", lambda: fake)
        assert main(["devices"]) == 0
        captured = capsys.readouterr()
        row = [l for l in captured.out.splitlines() if "Yeti" in l][0]
        assert "3.0ms" in row and "90.0ms" in row
        assert "low" in captured.out and "high" in captured.out


class TestTheBundledEntryPoint:
    """`natvox-cli.exe` implies `app`, which is a trap for the other names."""

    @pytest.fixture
    def entry(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packaging"))
        import entry as module
        return module

    def test_the_list_of_subcommands_is_the_parser_s_own(self):
        import argparse

        from natvox.cli import SUBCOMMANDS, build_parser

        actions = [a for a in build_parser()._actions
                   if isinstance(a, argparse._SubParsersAction)]
        assert len(actions) == 1
        assert set(actions[0].choices) == set(SUBCOMMANDS)

    def test_a_flag_is_handed_to_app(self, entry, capsys):
        assert entry.main(["--check", "--seconds", "0.05"]) == 0
        assert "blocks:" in capsys.readouterr().out

    @pytest.mark.parametrize("console", [True, False])
    def test_no_arguments_at_all_is_not_an_index_error(self, entry, monkeypatch,
                                                       console):
        """Both builds with nothing after them open the window.

        Parameterised on the build because they take different routes there --
        the windowed one returns before it looks at the arguments and the
        console one goes through the parser -- so a test that only covers the
        first would pass while `natvox-cli.exe` raised IndexError on its own
        empty argument list.
        """
        opened = []
        monkeypatch.setattr(entry, "is_console_build", lambda: console)
        monkeypatch.setattr("natvox.app.gui.main",
                            lambda argv: opened.append(argv) or 0)
        assert entry.main([]) == 0
        assert opened

    def test_a_subcommand_is_not(self, entry, capsys):
        """`natvox-cli.exe devices` is what the README shows for the
        unbundled program, and it has to mean the same thing here."""
        assert entry.main(["presets"]) == 0
        assert "female_soft" in capsys.readouterr().out


class TestTune:
    def test_it_says_what_to_install_when_there_is_no_microphone(self, capsys):
        assert main(["app", "--tune", "--seconds", "0.2"]) == 1
        err = capsys.readouterr().err
        assert "talk normally" in err
        assert "could not record" in err or "install" in err.lower()

    def test_it_prints_the_command_that_uses_what_it_found(self, capsys, monkeypatch):
        import sys as _sys

        from natvox.app import backend as backend_module

        _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        from bench import sustained

        voice = np.concatenate([sustained(120.0, v, 0.5, 48000)
                                for v in ("a", "i", "u", "e")])

        class Fake:
            @staticmethod
            def rec(frames, **kwargs):
                out = np.zeros((frames, 1), dtype="float32")
                take = min(frames, voice.size)
                out[:take, 0] = voice[:take]
                return out

            @staticmethod
            def wait():
                pass

        monkeypatch.setattr(backend_module, "_sounddevice", lambda: Fake)
        monkeypatch.setattr(backend_module, "rate_mismatch", lambda *a, **k: "")
        assert main(["app", "--tune", "--seconds", "2"]) == 0
        out = capsys.readouterr().out
        assert "120 Hz" in out
        assert "natvox live --pitch" in out
        assert "--f0-min" in out, "the floor it found is worth carrying over"


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

    def test_it_quotes_the_engine_the_user_will_actually_run(
            self, capsys, monkeypatch):
        """api.Session budgets extra delay so its settings can be moved live,
        which is right for the window and wrong to quote to somebody about to
        run `natvox live`.  On the default profile the two differ by 16 ms,
        and the larger one was being printed for the command that does not
        pay it."""
        from natvox import api
        from natvox.app import loopback
        from natvox.config import VoiceProfile
        from natvox.engine import VoiceChanger

        monkeypatch.setattr(
            loopback, "through_devices",
            lambda *a, **k: loopback.RoundTrip(48000, 256, [100.0]))
        assert main(["app", "--loopback"]) == 0
        out = capsys.readouterr().out

        live = VoiceChanger(48000, VoiceProfile()).latency_ms
        window = api.Session(48000, VoiceProfile()).latency_ms
        assert window > live + 5.0, "the two must differ, or this proves nothing"
        assert f"{live:.1f} ms" in out
        assert f"{window:.1f} ms" not in out

    def test_what_it_asks_portaudio_for_reaches_the_measurement(
            self, capsys, monkeypatch):
        from natvox.app import loopback

        asked = {}

        def fake(*args, **kwargs):
            asked.update(kwargs)
            return loopback.RoundTrip(48000, 256, [100.0])

        monkeypatch.setattr(loopback, "through_devices", fake)
        assert main(["app", "--loopback"]) == 0
        assert asked["latency"] == "low", "the default must not be PortAudio's"
        assert main(["app", "--loopback", "--latency", "high"]) == 0
        assert asked["latency"] == "high"

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
