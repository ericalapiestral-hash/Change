"""Unit tests for the analysis building blocks."""
from __future__ import annotations

import numpy as np
import pytest

from natvox.dsp.epochs import EpochTracker
from natvox.dsp.f0 import YinF0Tracker
from natvox.dsp.psola import Mark, grain_half_length, nearest_mark
from natvox.dsp.util import (
    OverlapAccumulator, RingBuffer, WINDOWS, resample_grain, soft_clip,
)


def harmonic(f0, seconds, sr, amplitude=1.0):
    t = np.arange(int(sr * seconds)) / sr
    x = sum(np.sin(2 * np.pi * h * f0 * t + h * 0.7) / h ** 1.1
            for h in range(1, 40) if h * f0 < sr / 2)
    return amplitude * x / np.max(np.abs(x))


class TestRingBuffer:
    def test_absolute_indexing_survives_discarding(self):
        buf = RingBuffer(16)
        buf.push(np.arange(10.0))
        assert list(buf.view(2, 6)) == [2, 3, 4, 5]
        buf.discard_to(5)
        buf.push(np.arange(10.0, 20.0))
        assert list(buf.view(8, 12)) == [8, 9, 10, 11]
        assert buf.origin == 5 and buf.end == 20

    def test_out_of_range_reads_are_zero_filled(self):
        buf = RingBuffer(16)
        buf.push(np.arange(1.0, 5.0))
        assert list(buf.view(-2, 2)) == [0, 0, 1, 2]
        assert list(buf.view(2, 6)) == [3, 4, 0, 0]


class TestOverlapAccumulator:
    def test_coherent_overlap_add_reconstructs_exactly(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 4000)
        buf = RingBuffer(8192)
        buf.push(x)
        acc = OverlapAccumulator(8192)
        half = 288
        for mark in range(half, 4000 - half, half):
            window = WINDOWS.get(2 * half)
            acc.add(mark - half, buf.view(mark - half, mark + half) * window, window)
        out = acc.read(600, 3000)
        assert np.max(np.abs(out - x[600:3000])) < 1e-9

    def test_uneven_spacing_still_reconstructs(self):
        """Normalising by the window sum must not assume a regular grid."""
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, 4000)
        buf = RingBuffer(8192)
        buf.push(x)
        acc = OverlapAccumulator(8192)
        half, mark = 288, 288
        while mark < 4000 - half:
            window = WINDOWS.get(2 * half)
            acc.add(mark - half, buf.view(mark - half, mark + half) * window, window)
            mark += int(half * rng.uniform(0.75, 1.25))
        out = acc.read(800, 3000)
        assert np.max(np.abs(out - x[800:3000])) < 1e-9

    def test_incoherent_grains_are_power_normalised(self):
        """Independent grains must not be divided by a coherent window sum."""
        rng = np.random.default_rng(2)
        acc = OverlapAccumulator(8192)
        half = 288
        for mark in range(half, 4000 - half, half):
            window = WINDOWS.get(2 * half)
            grain = rng.normal(0, 1, 2 * half) * window
            acc.add(mark - half, grain, window, coherent=False)
        out = acc.read(1000, 3000)
        # Unit-variance independent grains must come back at unit variance.
        assert 0.9 < np.std(out) < 1.1


class TestResampleGrain:
    @pytest.mark.parametrize("ratio", [0.7, 0.85, 1.0, 1.2, 1.5])
    def test_length_change_scales_frequency(self, ratio):
        sr, f0, n = 48000, 400.0, 2048
        grain = WINDOWS.get(n) * np.sin(2 * np.pi * f0 * np.arange(n) / sr)
        out = resample_grain(grain, int(round(n / ratio)))
        spec = np.abs(np.fft.rfft(out))
        peak = np.argmax(spec) * sr / out.size
        assert abs(peak - f0 * ratio) < 0.03 * f0 * ratio

    def test_fractional_delay_shifts_without_distorting(self):
        n = 1024
        t = np.arange(n)
        grain = WINDOWS.get(n) * np.sin(2 * np.pi * 50 * t / n)
        shifted = resample_grain(grain, n, fractional_delay=0.5)
        reference = WINDOWS.get(n) * np.sin(2 * np.pi * 50 * (t - 0.5) / n)
        interior = slice(100, -100)
        assert np.max(np.abs(shifted[interior] - reference[interior])) < 5e-3

    def test_no_op_returns_input_untouched(self):
        grain = WINDOWS.get(512) * np.random.default_rng(0).normal(0, 1, 512)
        assert np.array_equal(resample_grain(grain, 512, 0.0), grain)


class TestSoftClip:
    def test_transparent_below_the_knee(self):
        x = np.linspace(-0.7, 0.7, 1001)
        assert np.array_equal(soft_clip(x, knee=0.75), x)

    def test_bounded_above_the_knee(self):
        x = np.linspace(-8.0, 8.0, 1001)
        y = soft_clip(x, ceiling=0.98, knee=0.75)
        # tanh saturates in float64, so the ceiling is reached but never passed.
        assert np.max(np.abs(y)) <= 0.98
        assert np.all(np.diff(y) >= 0)  # monotone: no fold-back distortion


class TestPitchTracking:
    @pytest.mark.parametrize("f0", [80, 110, 165, 220, 330, 440])
    def test_accuracy_on_steady_tones(self, f0, sample_rate):
        x = harmonic(f0, 1.0, sample_rate)
        tracker = YinF0Tracker(sample_rate, 70, 500)
        estimates = [
            tracker.estimate(x[p - tracker.half:p - tracker.half + tracker.span], p).f0
            for p in range(tracker.half, x.size - tracker.lookahead, 240)
        ]
        voiced = [f for f in estimates if f > 0]
        assert len(voiced) > 100
        assert abs(np.median(voiced) - f0) / f0 < 0.002

    def test_noise_is_never_called_voiced(self, sample_rate, fricative_noise):
        tracker = YinF0Tracker(sample_rate, 75, 500)
        x = fricative_noise
        voiced = sum(
            tracker.estimate(x[p - tracker.half:p - tracker.half + tracker.span], p).voiced
            for p in range(tracker.half, x.size - tracker.lookahead, 240)
        )
        assert voiced == 0

    def test_voicing_matches_ground_truth(self, utterance, sample_rate):
        x, truth = utterance
        tracker = YinF0Tracker(sample_rate, 75, 500)
        # A frame sees `span` samples around its position, so frames straddling
        # a voiced/unvoiced boundary genuinely contain both; scoring them
        # against a single ground-truth sample would be meaningless.  The
        # margin also covers the synthesiser's own amplitude ramp and the
        # ringing of its formant resonators, which keep real periodic energy
        # in the signal for ~20 ms after the label says the vowel ended.
        margin = tracker.span // 2 + int(0.020 * sample_rate)
        interior = np.convolve(
            truth["voiced"].astype(float), np.ones(2 * margin + 1), mode="same"
        )
        settled = (interior == 0) | (interior == 2 * margin + 1)

        positions = [p for p in range(tracker.half, x.size - tracker.lookahead, 240)
                     if settled[p]]
        decisions = np.array([
            tracker.estimate(x[p - tracker.half:p - tracker.half + tracker.span], p).voiced
            for p in positions
        ])
        actual = truth["voiced"][positions]
        loud = np.array([np.max(np.abs(x[p:p + 240])) > 0.02 for p in positions])
        true_pos = np.sum(decisions & actual & loud)
        false_pos = np.sum(decisions & ~actual & loud)
        recall = true_pos / max(np.sum(actual & loud), 1)
        # A false "voiced" on a fricative is the expensive error -- it gets
        # pitch-shifted and buzzes -- so precision must be perfect.
        assert false_pos == 0
        assert recall > 0.95

    def test_survives_an_octave_trap(self, sample_rate):
        """A strong second harmonic must not pull the estimate up an octave."""
        t = np.arange(sample_rate) / sample_rate
        x = 0.3 * np.sin(2 * np.pi * 100 * t) + 1.0 * np.sin(2 * np.pi * 200 * t)
        tracker = YinF0Tracker(sample_rate, 70, 500)
        estimates = [
            tracker.estimate(x[p - tracker.half:p - tracker.half + tracker.span], p).f0
            for p in range(tracker.half, x.size - tracker.lookahead, 240)
        ]
        voiced = [f for f in estimates if f > 0]
        assert abs(np.median(voiced) - 100.0) < 2.0


class TestEpochs:
    def test_marks_lock_to_a_consistent_phase(self, sample_rate):
        period = 400.0
        x = harmonic(sample_rate / period, 0.5, sample_rate)
        buf = RingBuffer(1 << 16)
        buf.push(x)
        tracker = EpochTracker()
        mark = tracker.bootstrap(buf, 1000, period)
        gaps = []
        for _ in range(40):
            nxt = tracker.locate(buf, mark + int(period), mark, period)
            gaps.append(nxt - mark)
            mark = nxt
        assert np.all(np.abs(np.array(gaps) - period) <= 1)

    def test_recovers_when_the_period_estimate_is_wrong(self, sample_rate):
        period = 400.0
        x = harmonic(sample_rate / period, 0.5, sample_rate)
        buf = RingBuffer(1 << 16)
        buf.push(x)
        tracker = EpochTracker()
        mark = tracker.bootstrap(buf, 1000, period)
        # Predict 15% early; the correlation lock should pull it back.
        nxt = tracker.locate(buf, mark + int(period * 0.85), mark, period)
        assert abs((nxt - mark) - period) <= 2


class TestGrainGeometry:
    def test_widens_only_when_spacing_would_outrun_the_grain(self):
        assert grain_half_length(600, 1.0, 1.0) == 600
        assert grain_half_length(600, 1.5, 1.0) == 600      # pitch up: no need
        assert grain_half_length(600, 0.7, 1.5) > 600       # down + up: needs it
        assert grain_half_length(600, 0.1, 3.0) <= 600 * 1.6  # capped

    def test_nearest_mark_picks_by_time(self):
        marks = [Mark(0, 100, True), Mark(100, 100, True), Mark(200, 100, True)]
        assert nearest_mark(marks, 0) == 0
        assert nearest_mark(marks, 140) == 1
        assert nearest_mark(marks, 190) == 2
        assert nearest_mark(marks, 1000) == 2
