"""Contract tests for the streaming engine: what callers may rely on."""
from __future__ import annotations

import numpy as np
import pytest

import natvox
from natvox import VoiceChanger, VoiceProfile, presets

BLOCK_SIZES = [64, 128, 333, 512, 1000, 2048, 4096]


def process(audio, sample_rate, profile, block):
    changer = VoiceChanger(sample_rate, profile)
    chunks = [changer.process(audio[i:i + block])
              for i in range(0, audio.size, block)]
    chunks.append(changer.flush())
    return np.concatenate(chunks)[changer.latency_samples:][:audio.size]


class TestStreamingContract:
    def test_output_length_always_matches_input(self, sample_rate):
        changer = VoiceChanger(sample_rate, presets.get("male_to_female"))
        rng = np.random.default_rng(0)
        for size in [1, 7, 64, 333, 4096]:
            assert changer.process(rng.normal(0, 0.1, size)).size == size

    @pytest.mark.parametrize("name", ["off", "younger", "male_to_female",
                                      "female_to_male", "anonymous"])
    def test_result_does_not_depend_on_block_size(self, utterance, sample_rate, name):
        """A real-time caller cannot choose its block size; the result must
        not change when the audio device hands over 64 frames instead of 4096."""
        audio, _ = utterance
        profile = presets.get(name)
        reference = process(audio, sample_rate, profile, 1024)
        for block in BLOCK_SIZES:
            other = process(audio, sample_rate, profile, block)
            # Identical grain placement; the residue is float noise in the
            # FFT-based grain resampler, ~100 dB below the signal.
            assert np.max(np.abs(other - reference)) < 1e-4, f"block={block}"

    def test_latency_is_declared_and_constant(self, sample_rate):
        changer = VoiceChanger(sample_rate, presets.get("male_to_female"))
        latency = changer.latency_samples
        assert 0 < latency < sample_rate * 0.1  # under 100 ms
        for _ in range(20):
            changer.process(np.zeros(512))
        assert changer.latency_samples == latency

    def test_impulse_arrives_at_the_declared_latency(self, sample_rate):
        changer = VoiceChanger(sample_rate, VoiceProfile(pitch_semitones=3.0))
        x = np.zeros(sample_rate // 2)
        x[5000] = 0.8
        out = np.concatenate(
            [changer.process(x[i:i + 512]) for i in range(0, x.size, 512)]
        )
        peak = int(np.argmax(np.abs(out)))
        assert abs(peak - (5000 + changer.latency_samples)) < 0.01 * sample_rate

    def test_reset_restores_a_fresh_engine(self, utterance, sample_rate):
        audio, _ = utterance
        profile = presets.get("younger")
        first = process(audio, sample_rate, profile, 512)
        changer = VoiceChanger(sample_rate, profile)
        for i in range(0, audio.size // 2, 512):
            changer.process(audio[i:i + 512])
        changer.reset()
        chunks = [changer.process(audio[i:i + 512]) for i in range(0, audio.size, 512)]
        chunks.append(changer.flush())
        again = np.concatenate(chunks)[changer.latency_samples:][:audio.size]
        assert np.max(np.abs(again - first)) < 1e-4

    def test_silence_in_silence_out(self, sample_rate):
        changer = VoiceChanger(sample_rate, presets.get("male_to_female"))
        out = np.concatenate([changer.process(np.zeros(1024)) for _ in range(20)])
        assert np.max(np.abs(out)) < 1e-9

    def test_output_never_clips(self, sample_rate):
        """A limiter that lets even one sample past would tick audibly."""
        rng = np.random.default_rng(3)
        loud = np.clip(rng.normal(0, 1.5, sample_rate), -4, 4)
        out = process(loud, sample_rate, presets.get("younger").replace(
            output_gain_db=12.0), 512)
        assert np.max(np.abs(out)) <= 1.0

    def test_handles_dc_and_rumble(self, sample_rate):
        t = np.arange(sample_rate) / sample_rate
        x = 0.5 + 0.3 * np.sin(2 * np.pi * 12 * t)  # offset plus subsonic
        out = process(x, sample_rate, presets.get("deeper"), 512)
        assert abs(np.mean(out[2000:])) < 0.02


class TestIdentity:
    def test_zero_shift_is_a_passthrough(self, utterance, sample_rate):
        audio, _ = utterance
        out = process(audio, sample_rate, VoiceProfile(highpass_hz=0.0), 512)
        guard = slice(3000, -3000)
        error = np.mean((out[guard] - audio[guard]) ** 2) / np.mean(audio[guard] ** 2)
        assert 10 * np.log10(error + 1e-30) < -100

    def test_zero_shift_changes_nothing_but_the_rumble_filter(self, utterance,
                                                              sample_rate):
        from natvox.dsp.util import BiquadHighpass

        audio, _ = utterance
        profile = VoiceProfile(highpass_hz=60.0)
        out = process(audio, sample_rate, profile, 512)
        expected = BiquadHighpass(sample_rate, 60.0)(audio)
        guard = slice(3000, -3000)
        error = (np.mean((out[guard] - expected[guard]) ** 2)
                 / np.mean(expected[guard] ** 2))
        assert 10 * np.log10(error + 1e-30) < -100

    def test_disabling_the_rumble_filter_is_exact(self, sample_rate):
        from natvox.dsp.util import BiquadHighpass

        rng = np.random.default_rng(9)
        x = rng.normal(0, 0.2, 4096)
        assert np.array_equal(BiquadHighpass(sample_rate, 0.0)(x), x)


class TestMultichannel:
    def test_process_array_keeps_shape(self, sample_rate):
        rng = np.random.default_rng(4)
        stereo = rng.normal(0, 0.1, (sample_rate // 2, 2))
        out = natvox.process_array(stereo, sample_rate, presets.get("brighter"))
        assert out.shape == stereo.shape
        mono = natvox.process_array(stereo[:, 0], sample_rate, presets.get("brighter"))
        assert mono.ndim == 1 and mono.size == stereo.shape[0]


class TestProfile:
    def test_semitone_conversion_round_trips(self):
        for st in [-7.0, -1.5, 0.0, 3.0, 12.0]:
            assert abs(natvox.ratio_to_semitones(
                natvox.semitones_to_ratio(st)) - st) < 1e-9

    def test_an_octave_is_a_doubling(self):
        assert abs(VoiceProfile(pitch_semitones=12.0).pitch_ratio - 2.0) < 1e-12

    @pytest.mark.parametrize("kwargs", [
        {"f0_min": 0.0}, {"f0_min": 500.0, "f0_max": 100.0},
        {"breathiness": 1.5}, {"pitch_semitones": 40.0},
    ])
    def test_rejects_impossible_settings(self, kwargs):
        with pytest.raises(ValueError):
            VoiceProfile(**kwargs)

    def test_warns_about_settings_that_cost_naturalness(self):
        assert VoiceProfile(pitch_semitones=10.0).warnings()
        assert VoiceProfile(pitch_semitones=6.0, formant_semitones=0.0).warnings()
        assert not VoiceProfile(pitch_semitones=2.0, formant_semitones=1.0).warnings()

    def test_every_preset_is_usable_and_sane(self, sample_rate):
        for name in presets.names():
            profile = presets.get(name)
            changer = VoiceChanger(sample_rate, profile)
            assert 0 < changer.latency_ms < 100
            assert changer.process(np.zeros(512)).size == 512

    def test_unknown_preset_names_the_alternatives(self):
        with pytest.raises(KeyError, match="male_to_female"):
            presets.get("nope")


class TestRealTimeSafety:
    """What a real audio callback does to an engine that was only ever tested
    offline: hands it garbage, and judges it on its worst block rather than its
    average."""

    @pytest.mark.parametrize("poison", [np.nan, np.inf, -np.inf])
    def test_a_non_finite_sample_does_not_kill_the_stream(self, sample_rate, poison):
        """An xrun or an interface unplugged mid-stream can put a NaN in the
        buffer.  Before this was guarded, four of them raised out of the audio
        callback, and because the exception escaped before the tracker's
        cursor advanced it retried the same poisoned frame forever: the stream
        stopped for good and the buffers grew without bound.  In the browser
        it did the same thing silently.
        """
        rng = np.random.default_rng(0)
        audio = 0.2 * rng.standard_normal(sample_rate * 2)
        audio[sample_rate:sample_rate + 4] = poison

        changer = VoiceChanger(sample_rate, presets.get("male_to_female"))
        out = np.concatenate([changer.process(audio[i:i + 128])
                              for i in range(0, audio.size, 128)])
        assert np.all(np.isfinite(out)), "non-finite samples reached the output"
        tail = out[-sample_rate // 2:]
        assert np.sqrt(np.mean(tail * tail)) > 0.01, "the engine never recovered"

    def test_the_first_block_is_not_the_most_expensive_one(self, sample_rate):
        """Priming used to happen inside the first process() call, which made
        that one callback carry a whole latency of pitch tracking - over the
        deadline at 128 frames and twice it at 64, so the first callback was
        near-certain to drop out.  It now happens at construction.
        """
        import time

        rng = np.random.default_rng(1)
        audio = 0.2 * rng.standard_normal(sample_rate)
        changer = VoiceChanger(sample_rate, presets.get("male_to_female"))
        changer.process(audio[:128])          # warm numpy and scipy paths
        changer = VoiceChanger(sample_rate, presets.get("male_to_female"))

        started = time.perf_counter()
        changer.process(audio[:128])
        first = time.perf_counter() - started
        rest = []
        for i in range(128, 128 * 60, 128):
            started = time.perf_counter()
            changer.process(audio[i:i + 128])
            rest.append(time.perf_counter() - started)
        assert first < 6 * np.median(rest), (
            f"first block {first * 1e6:.0f} us against a median of "
            f"{np.median(rest) * 1e6:.0f} us")

    def test_nothing_grows_without_bound(self, sample_rate):
        """Sixty seconds of speech must not leave the engine holding more than
        it started with."""
        from synth_speech import utterance

        audio, _ = utterance(sample_rate)
        changer = VoiceChanger(sample_rate, presets.get("male_to_female"))
        for _ in range(4):
            changer.process(audio[:128])
        baseline = (changer._in._buf.nbytes + changer._acc._sig.nbytes,
                    len(changer._marks), len(changer._frames))
        long_audio = np.tile(audio, 25)      # ~60 s
        for i in range(0, long_audio.size, 128):
            changer.process(long_audio[i:i + 128])
        after = (changer._in._buf.nbytes + changer._acc._sig.nbytes,
                 len(changer._marks), len(changer._frames))
        assert after[0] == baseline[0], f"buffers grew {baseline[0]} -> {after[0]}"
        assert after[1] < 64 and after[2] < 64, f"lists grew to {after[1:]}"

    @pytest.mark.parametrize("name", ["silence-then-shout", "sweep", "clipping",
                                      "hum", "dc"])
    def test_pathological_inputs_stay_finite_and_bounded(self, sample_rate, name):
        n = sample_rate
        t = np.arange(n) / sample_rate
        if name == "silence-then-shout":
            audio = np.zeros(n)
            audio[n // 2:] = 0.9 * np.sin(2 * np.pi * 130 * t[n // 2:])
        elif name == "sweep":
            audio = 0.5 * np.sin(2 * np.pi * np.cumsum(np.linspace(60, 600, n)) / sample_rate)
        elif name == "clipping":
            audio = np.clip(4.0 * np.sin(2 * np.pi * 150 * t), -1.0, 1.0)
        elif name == "hum":
            audio = 0.4 * np.sin(2 * np.pi * 50 * t) + 0.2 * np.sin(2 * np.pi * 130 * t)
        else:
            audio = 0.6 + 0.2 * np.sin(2 * np.pi * 130 * t)

        changer = VoiceChanger(sample_rate, presets.get("male_to_female"))
        out = np.concatenate([changer.process(audio[i:i + 128])
                              for i in range(0, n, 128)])
        assert np.all(np.isfinite(out))
        assert np.max(np.abs(out)) <= 1.0
