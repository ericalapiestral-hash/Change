"""The streaming machinery around a conversion model.

No model is shipped or trained here, so what is tested is the part that is
model-independent and easy to get subtly wrong: window bookkeeping, seam
cross-fades, F0 conditioning and latency accounting.  An identity model makes
those failures visible -- anything other than a perfect reconstruction is the
wrapper's fault, not the model's.
"""
from __future__ import annotations

import numpy as np
import pytest

import natvox
from natvox.neural import BlockAdapter, Pipeline, StreamingNeuralConverter, build


def identity(audio, sample_rate, f0):
    return audio


def run(converter, audio, block=512):
    return np.concatenate([converter.process(audio[i:i + block])
                           for i in range(0, audio.size, block)])


class TestStreamingWrapper:
    def test_identity_model_reconstructs_exactly(self, sample_rate):
        """Windowing and cross-fading must be transparent on their own."""
        rng = np.random.default_rng(0)
        x = rng.normal(0, 0.2, sample_rate)  # broadband: any seam shows up
        converter = StreamingNeuralConverter(sample_rate, identity)
        y = run(converter, x)
        latency = converter.latency_samples
        usable = slice(latency, latency + sample_rate // 2)
        assert np.max(np.abs(y[usable] - x[:sample_rate // 2])) < 1e-9

    def test_latency_is_one_hop_plus_context(self, sample_rate):
        converter = StreamingNeuralConverter(sample_rate, identity,
                                             hop_seconds=0.05, context_seconds=0.05)
        assert converter.latency_samples == converter.hop + converter.context

    def test_shorter_hop_lowers_latency(self, sample_rate):
        fast = StreamingNeuralConverter(sample_rate, identity, hop_seconds=0.02)
        slow = StreamingNeuralConverter(sample_rate, identity, hop_seconds=0.16)
        assert fast.latency_samples < slow.latency_samples

    def test_model_receives_context_on_both_sides(self, sample_rate):
        seen = []
        converter = StreamingNeuralConverter(
            sample_rate, lambda a, sr, f0: (seen.append(a.size), a)[1])
        run(converter, np.zeros(sample_rate))
        assert seen and all(n == converter.hop + 2 * converter.context for n in seen)

    def test_f0_conditioning_carries_the_pitch_shift(self, sample_rate):
        seen = []

        def spy(audio, sr, f0):
            seen.append(f0)
            return audio

        tone = 0.3 * np.sin(2 * np.pi * 150 * np.arange(sample_rate) / sample_rate)
        converter = StreamingNeuralConverter(sample_rate, spy,
                                             pitch_shift_semitones=5.0)
        run(converter, tone)
        values = np.concatenate([v for v in seen[len(seen) // 2:] if v.size])
        voiced = values[values > 0]
        assert voiced.size > 0
        assert abs(np.median(voiced) - 150 * 2 ** (5 / 12)) < 3.0

    def test_f0_is_zero_where_there_is_no_pitch(self, sample_rate):
        seen = []
        rng = np.random.default_rng(1)
        converter = StreamingNeuralConverter(
            sample_rate, lambda a, sr, f0: (seen.append(f0), a)[1])
        run(converter, rng.normal(0, 0.2, sample_rate))
        values = np.concatenate([v for v in seen if v.size])
        assert np.all(values == 0.0)

    def test_rejects_a_model_that_changes_length(self, sample_rate):
        converter = StreamingNeuralConverter(sample_rate, lambda a, sr, f0: a[:-1])
        with pytest.raises(ValueError, match="samples"):
            run(converter, np.zeros(sample_rate))

    def test_rejects_context_too_short_for_pitch_tracking(self, sample_rate):
        with pytest.raises(ValueError, match="look-ahead"):
            StreamingNeuralConverter(sample_rate, identity,
                                     context_seconds=0.008,
                                     crossfade_seconds=0.005)

    def test_rejects_crossfade_longer_than_the_overlap(self, sample_rate):
        with pytest.raises(ValueError, match="crossfade"):
            StreamingNeuralConverter(sample_rate, identity,
                                     context_seconds=0.05, crossfade_seconds=0.08)

    def test_reset_clears_state(self, sample_rate):
        rng = np.random.default_rng(2)
        x = rng.normal(0, 0.2, sample_rate // 2)
        converter = StreamingNeuralConverter(sample_rate, identity)
        first = run(converter, x)
        converter.reset()
        assert np.max(np.abs(run(converter, x) - first)) < 1e-12


class TestPipeline:
    def test_latencies_add(self, sample_rate):
        neural = StreamingNeuralConverter(sample_rate, identity)
        dsp = natvox.VoiceChanger(sample_rate, natvox.presets.get("brighter"))
        pipeline = Pipeline(neural, dsp)
        assert pipeline.latency_samples == neural.latency_samples + dsp.latency_samples

    def test_rejects_mismatched_sample_rates(self, sample_rate):
        with pytest.raises(ValueError, match="sample rate"):
            Pipeline(StreamingNeuralConverter(sample_rate, identity),
                     natvox.VoiceChanger(24000))

    def test_build_skips_the_dsp_stage_when_it_would_do_nothing(self, sample_rate):
        assert isinstance(build(sample_rate, identity), StreamingNeuralConverter)
        assert isinstance(build(sample_rate, identity,
                                natvox.presets.get("brighter")), Pipeline)

    def test_dsp_polish_still_shifts_pitch_after_the_model(self, sample_rate,
                                                          sustained_vowel):
        from evaluate import pitch_track

        dry, f0 = sustained_vowel
        chain = build(sample_rate, identity, natvox.presets.get("younger"))
        wet = run(chain, dry)[chain.latency_samples:]
        _, tracked = pitch_track(wet, sample_rate, f0_min=50.0, f0_max=800.0)
        voiced = tracked[tracked > 0]
        expected = f0 * natvox.presets.get("younger").pitch_ratio
        assert abs(np.median(voiced) - expected) / expected < 0.01


class TestBlockAdapter:
    def test_bridges_variable_callbacks_to_a_fixed_window(self, sample_rate):
        sizes = []
        adapter = BlockAdapter(sample_rate, 1024,
                               lambda c: (sizes.append(c.size), c)[1])
        rng = np.random.default_rng(3)
        x = rng.normal(0, 0.2, sample_rate // 2)
        out, i = [], 0
        while i < x.size:
            n = int(rng.choice([128, 256, 480, 512]))
            out.append(adapter.process(x[i:i + n]))
            i += n
        y = np.concatenate(out)
        assert all(s == 1024 for s in sizes)
        assert y.size == x.size
        assert np.max(np.abs(y[1024:] - x[:y.size - 1024])) < 1e-12
