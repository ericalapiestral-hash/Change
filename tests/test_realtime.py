"""The audio-callback layer, exercised without a sound card.

PortAudio is not available in every environment (and never in CI), so the
callback logic lives in :class:`StreamProcessor`, which needs no device.  What
matters here is that it survives what a real device does: block sizes that
change from call to call, multichannel buffers, and float32 output.
"""
from __future__ import annotations

import numpy as np
import pytest

import natvox
from natvox.realtime import StreamProcessor


@pytest.fixture
def processor(sample_rate):
    return StreamProcessor(
        natvox.VoiceChanger(sample_rate, natvox.presets.get("male_to_female")),
        channels=2,
    )


def drive(processor, audio, seed=0):
    """Feed audio the way a device would: in blocks of varying size."""
    rng = np.random.default_rng(seed)
    out, i = [], 0
    while i < audio.size:
        frames = int(rng.choice([128, 256, 480, 512]))
        block = audio[i:i + frames]
        if block.size == 0:
            break
        out.append(processor(block, block.size))
        i += block.size
    return np.concatenate(out)


class TestStreamProcessor:
    def test_handles_jittery_block_sizes(self, processor, sample_rate):
        tone = 0.3 * np.sin(2 * np.pi * 140 * np.arange(sample_rate) / sample_rate)
        out = drive(processor, tone)
        assert out.shape[1] == 2
        assert out.dtype == np.float32
        assert np.all(np.isfinite(out))

    def test_mixes_multichannel_input_to_mono(self, sample_rate):
        processor = StreamProcessor(natvox.VoiceChanger(sample_rate), channels=1)
        stereo = np.zeros((512, 2))
        stereo[:, 0] = 0.4
        out = processor(stereo, 512)
        assert out.shape == (512, 1)

    def test_duplicates_mono_across_output_channels(self, processor, sample_rate):
        tone = 0.3 * np.sin(2 * np.pi * 140 * np.arange(sample_rate // 4) / sample_rate)
        out = drive(processor, tone)
        assert np.array_equal(out[:, 0], out[:, 1])

    def test_reports_cpu_load_and_deadline_misses(self, processor, sample_rate):
        tone = 0.3 * np.sin(2 * np.pi * 140 * np.arange(sample_rate) / sample_rate)
        drive(processor, tone)
        stats = processor.stats
        assert stats.blocks > 0 and stats.frames == pytest.approx(sample_rate, rel=0.01)
        assert 0.0 < stats.mean_load < 1.0
        assert stats.peak_load >= stats.mean_load
        assert "CPU load" in stats.summary()

    def test_dry_blend_is_delay_matched(self, sample_rate):
        """An undelayed dry path would comb-filter against the wet one."""
        processor = StreamProcessor(
            natvox.VoiceChanger(sample_rate, natvox.presets.get("brighter")),
            channels=1, dry_wet=0.0,
        )
        rng = np.random.default_rng(4)
        x = rng.normal(0, 0.2, sample_rate // 2)
        out = drive(processor, x)[:, 0]
        delay = processor.converter.latency_samples
        usable = slice(delay + 1000, delay + 20000)
        reference = x[1000:20000]
        assert np.max(np.abs(out[usable] - reference)) < 1e-4

    def test_reset_clears_statistics(self, processor, sample_rate):
        drive(processor, np.zeros(sample_rate // 4))
        processor.reset()
        assert processor.stats.blocks == 0
