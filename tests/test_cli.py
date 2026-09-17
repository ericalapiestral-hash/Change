"""End-to-end checks on the command line."""
from __future__ import annotations

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
    def test_reports_clearly_when_audio_is_unavailable(self, capsys):
        """Headless machines have no PortAudio; that must not look like a crash."""
        code = main(["devices"])
        captured = capsys.readouterr()
        if code != 0:
            assert "realtime" in captured.err
