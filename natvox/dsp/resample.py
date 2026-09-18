"""Polyphase windowed-sinc resampling of one grain, with a sub-sample shift.

This is where a grain's length is changed, which is how formants move without
pitch following them. Two properties are load-bearing:

* The fractional shift.  Grain positions land between samples, and rounding
  them to whole samples jitters the synthesis period enough to put a noise
  floor around -29 dB under the voice.  Carrying the fraction in the kernel
  phase removes it.
* The cutoff scaling.  Shortening a grain stretches its spectrum, so without
  lowering the kernel's cutoff the top of the band folds back as aliasing.

A transform-based resampler would also do this, and an earlier version did.
It was replaced because a transform of the grain's *exact* length is needed,
and grain lengths follow the pitch period -- arbitrary numbers, which need
mixed-radix or Bluestein transforms.  The browser build has to do the same
arithmetic inside an audio callback with no library and no allocation, and a
polyphase kernel is both simpler there and identical here: the two
implementations agree to -76 dB, far below anything the engine produces.
"""
from __future__ import annotations

import numpy as np
from scipy.special import i0

#: Kernel half-width in output samples, before widening for compression.
HALF_TAPS = 16
#: Sub-sample resolution of the kernel table.
PHASES = 512
#: Kaiser shape; ~-80 dB stopband, which is 25 dB below the engine's own floor.
BETA = 9.0


class GrainResampler:
    """Band-limited resampling of one grain.

    The formant ratio is fixed for a given configuration, so the whole kernel
    table is built once and resampling is then a fixed-length dot product per
    output sample.
    """

    def __init__(self, ratio: float, half_taps: int = HALF_TAPS,
                 phases: int = PHASES, beta: float = BETA) -> None:
        self.ratio = float(ratio)
        self.phases = int(phases)
        cutoff = min(1.0, 1.0 / self.ratio)
        # Widen when compressing so the same number of sinc lobes is covered.
        self.half = int(np.ceil(half_taps * max(1.0, self.ratio)))
        self.taps = 2 * self.half

        offsets = np.arange(-self.half + 1, self.half + 1, dtype=np.float64)[None, :]
        mu = np.arange(phases, dtype=np.float64)[:, None] / phases
        x = offsets - mu                                  # distance in input samples

        window = np.zeros_like(x)
        inside = np.abs(x) <= self.half
        scaled = x[inside] / self.half
        window[inside] = i0(beta * np.sqrt(np.maximum(1.0 - scaled * scaled, 0.0))) / i0(beta)

        table = np.sinc(cutoff * x) * window
        # Unity DC gain per phase: truncating the sinc otherwise leaves a
        # ripple that reads as a level wobble across the grain.
        table /= table.sum(axis=1, keepdims=True)
        self.table = np.ascontiguousarray(table)
        self._offsets = np.arange(-self.half + 1, self.half + 1, dtype=np.int64)

    def __call__(self, grain: np.ndarray, out_len: int,
                 fractional_delay: float = 0.0) -> np.ndarray:
        """Resample `grain` to `out_len` samples, delayed by a fraction of one."""
        n, m = grain.size, int(out_len)
        if n < 4 or m < 4:
            return grain.astype(np.float64, copy=False)
        step = n / m
        pos = (np.arange(m) - fractional_delay) * step
        base = np.floor(pos).astype(np.int64)
        phase = np.minimum(np.round((pos - base) * self.phases).astype(np.int64),
                           self.phases - 1)
        # The grain is Hann-windowed and so is ~zero at both ends, which makes
        # zero extension indistinguishable from the periodic extension a
        # transform-based resampler would assume.
        pad = self.half + 1
        padded = np.concatenate([np.zeros(pad), grain, np.zeros(pad + 1)])
        gathered = padded[np.clip(base[:, None] + self._offsets[None, :] + pad,
                                  0, padded.size - 1)]
        return np.einsum('ij,ij->i', gathered, self.table[phase])
