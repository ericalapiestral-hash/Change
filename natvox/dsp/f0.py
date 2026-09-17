"""Fundamental-frequency tracking (YIN) with voicing detection.

Pitch errors are the single loudest source of artifacts in a PSOLA voice
changer: one octave jump on a sustained vowel produces an audible croak, and a
false "voiced" verdict on a fricative turns it into a buzz.  So this tracker
spends its effort on *stability* rather than on chasing the last cent of
accuracy -- an octave-continuity bias, a median guard and hysteresis on the
voicing gate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft

from .util import WINDOWS, EPS

#: Voiced speech puts most of its energy under this frequency (the fundamental
#: plus F1); fricatives put almost none there.  Used as a second, independent
#: voicing cue -- periodicity alone will occasionally latch onto noise.
LOW_BAND_HZ = 1000.0


@dataclass
class F0Frame:
    """One pitch observation, in absolute stream coordinates."""

    position: int          # absolute sample index the estimate refers to
    f0: float              # Hz, 0.0 when unvoiced
    period: float          # samples, 0.0 when unvoiced
    voiced: bool
    periodicity: float     # 0..1, how strongly periodic the frame looked
    low_band: float        # 0..1, share of energy below LOW_BAND_HZ
    rms: float


class YinF0Tracker:
    """Streaming YIN estimator.

    The estimate for a frame uses samples ``[pos - half, pos + half + tau_max)``
    where ``half = window // 2``; :attr:`lookahead` reports how far past the
    analysis point that reaches, which is what sets the engine's latency floor.
    """

    def __init__(
        self,
        sample_rate: int,
        f0_min: float = 70.0,
        f0_max: float = 500.0,
        threshold: float = 0.15,
        voiced_periodicity: float = 0.72,
        unvoiced_periodicity: float = 0.55,
        voiced_low_band: float = 0.30,
        unvoiced_low_band: float = 0.18,
        onset_frames: int = 2,
    ) -> None:
        if not 0 < f0_min < f0_max < sample_rate / 2:
            raise ValueError("require 0 < f0_min < f0_max < nyquist")
        self.sample_rate = int(sample_rate)
        self.f0_min = float(f0_min)
        self.f0_max = float(f0_max)
        self.threshold = float(threshold)
        # Hysteresis: it takes more evidence to declare voicing than to keep it.
        # Both cues must agree to start voicing; either one suffices to sustain
        # it, so a momentary dip mid-vowel does not punch a hole in the output.
        self.voiced_periodicity = float(voiced_periodicity)
        self.unvoiced_periodicity = float(unvoiced_periodicity)
        self.voiced_low_band = float(voiced_low_band)
        self.unvoiced_low_band = float(unvoiced_low_band)
        self.onset_frames = int(onset_frames)

        self.tau_min = max(2, int(np.floor(sample_rate / f0_max)))
        self.tau_max = int(np.ceil(sample_rate / f0_min)) + 1
        self.window = self.tau_max          # YIN integration window
        self.span = self.window + self.tau_max
        self._span = self.span
        self._nfft = next_fast_len(self._span + self.window)

        # Label the frame at the centre of everything it looks at: that is
        # where the estimate is actually valid, and it splits the buffering
        # evenly instead of charging it all to look-ahead.
        self.half = self.span // 2
        self.lookahead = self.span - self.half
        self.history = self.half

        self._band_nfft = next_fast_len(self.window)
        self._band_cut = int(LOW_BAND_HZ * self._band_nfft / sample_rate) + 1

        self._prev_tau = 0.0
        self._prev_voiced = False
        self._pending = 0
        self._f0_hist: list[float] = []
        self._noise_rms = 1e-4

    # ------------------------------------------------------------------ core
    def _cmnd(self, x: np.ndarray) -> np.ndarray:
        """Cumulative mean normalised difference function d'(tau)."""
        n = self._span
        w = self.window
        power = np.concatenate(([0.0], np.cumsum(x * x)))
        f_win = rfft(x[:w], self._nfft)
        f_all = rfft(x, self._nfft)
        corr = irfft(np.conj(f_win) * f_all, self._nfft)[: self.tau_max + 1]

        taus = np.arange(self.tau_max + 1)
        diff = (power[w] - power[0]) + (power[taus + w] - power[taus]) - 2.0 * corr
        np.maximum(diff, 0.0, out=diff)

        cmnd = np.empty_like(diff)
        cmnd[0] = 1.0
        running = np.cumsum(diff[1:])
        cmnd[1:] = diff[1:] * taus[1:] / np.maximum(running, EPS)
        return cmnd

    def _low_band_ratio(self, x: np.ndarray) -> float:
        """Share of the frame's energy below :data:`LOW_BAND_HZ`.

        Costs one short FFT and separates vowels from fricatives far more
        reliably than periodicity does, because a band-limited noise burst can
        look periodic to an autocorrelation but can never look low-pitched.
        """
        w = self.window
        spec = np.abs(rfft(x[:w] * WINDOWS.get(w), self._band_nfft)) ** 2
        total = float(np.sum(spec))
        if total <= EPS:
            return 0.0
        return float(np.sum(spec[:self._band_cut]) / total)

    def _pick_tau(self, cmnd: np.ndarray) -> float:
        """Absolute-threshold search with an octave-continuity preference."""
        band = cmnd[self.tau_min:self.tau_max + 1]
        if band.size == 0:
            return 0.0

        best = int(np.argmin(band)) + self.tau_min
        below = np.flatnonzero(band < self.threshold)
        if below.size:
            # First dip under the threshold, then walk down to its local
            # minimum. Taking the *first* dip rather than the global one is
            # what keeps YIN off the sub-harmonics.
            tau = int(below[0]) + self.tau_min
            while tau + 1 <= self.tau_max and cmnd[tau + 1] < cmnd[tau]:
                tau += 1
            best = tau

        # Octave guard: if the previous frame was voiced and there is a dip
        # near the previous period that is nearly as deep, stay on it.
        if self._prev_voiced and self._prev_tau > 0:
            lo = max(self.tau_min, int(self._prev_tau * 0.80))
            hi = min(self.tau_max, int(self._prev_tau * 1.25))
            if hi > lo:
                local = int(np.argmin(cmnd[lo:hi + 1])) + lo
                if local != best and cmnd[local] <= cmnd[best] * 1.30 + 0.02:
                    best = local

        return self._parabolic(cmnd, best)

    @staticmethod
    def _parabolic(y: np.ndarray, i: int) -> float:
        """Sub-sample minimum from the three points around index ``i``."""
        if i <= 0 or i >= y.size - 1:
            return float(i)
        a, b, c = y[i - 1], y[i], y[i + 1]
        denom = a - 2.0 * b + c
        if abs(denom) < EPS:
            return float(i)
        return float(i) + 0.5 * (a - c) / denom

    # ----------------------------------------------------------------- public
    def estimate(self, segment: np.ndarray, position: int) -> F0Frame:
        """Estimate F0 for ``segment`` (length :attr:`_span`) centred per docstring."""
        if segment.size < self._span:
            segment = np.pad(segment, (0, self._span - segment.size))
        x = segment[: self._span].astype(np.float64, copy=False)

        rms = float(np.sqrt(np.mean(x * x) + EPS))
        # Slow-rising / fast-falling noise floor tracks room tone without
        # latching onto speech.
        if rms < self._noise_rms:
            self._noise_rms = 0.9 * self._noise_rms + 0.1 * rms
        else:
            self._noise_rms = 0.9995 * self._noise_rms + 0.0005 * rms

        cmnd = self._cmnd(x)
        tau = self._pick_tau(cmnd)
        idx = int(round(tau))
        periodicity = 0.0
        if 0 < idx < cmnd.size:
            periodicity = float(np.clip(1.0 - cmnd[idx], 0.0, 1.0))
        low_band = self._low_band_ratio(x)

        loud_enough = rms > max(self._noise_rms * 2.0, 1.5e-4)
        usable = bool(loud_enough and tau >= self.tau_min)
        if self._prev_voiced:
            # Sustain on either cue: vowels dip in periodicity at formant
            # transitions, and low-band energy dips on close vowels.
            voiced = usable and (
                periodicity >= self.unvoiced_periodicity
                and low_band >= self.unvoiced_low_band
            )
            self._pending = self.onset_frames if voiced else 0
        else:
            # Start only when both cues agree, for several frames running.
            candidate = usable and (
                periodicity >= self.voiced_periodicity
                and low_band >= self.voiced_low_band
            )
            self._pending = self._pending + 1 if candidate else 0
            voiced = self._pending >= self.onset_frames

        f0 = self.sample_rate / tau if (voiced and tau > 0) else 0.0
        if voiced:
            f0 = self._median_guard(f0)
            tau = self.sample_rate / f0

        self._prev_voiced = voiced
        self._prev_tau = tau if voiced else self._prev_tau
        return F0Frame(
            position=position,
            f0=f0 if voiced else 0.0,
            period=(self.sample_rate / f0) if voiced and f0 > 0 else 0.0,
            voiced=voiced,
            periodicity=periodicity,
            low_band=low_band,
            rms=rms,
        )

    def _median_guard(self, f0: float) -> float:
        """Median of the last three voiced estimates, but only to veto outliers.

        A plain median would smear real pitch movement; this only replaces the
        current value when it disagrees with both neighbours by over a
        semitone-and-a-half, i.e. when it looks like a tracking slip.
        """
        self._f0_hist.append(f0)
        if len(self._f0_hist) > 3:
            self._f0_hist.pop(0)
        if len(self._f0_hist) < 3:
            return f0
        med = float(np.median(self._f0_hist))
        if med > 0 and abs(np.log2(f0 / med)) > 0.12:
            return med
        return f0

    def reset(self) -> None:
        self._prev_tau = 0.0
        self._prev_voiced = False
        self._pending = 0
        self._f0_hist.clear()
        self._noise_rms = 1e-4
