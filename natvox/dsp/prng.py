"""xoshiro128** - a small deterministic generator, matched byte for byte by the
JavaScript port in ``web/dsp/prng.js``.

The engine needs randomness in two places: the jitter that keeps unvoiced grain
spacing from stamping a buzz onto fricatives, and the noise for the breath mix.
Using numpy's generator here and something else in JavaScript would make the two
implementations diverge on any signal containing consonants, which would destroy
the cheapest way to verify the port did not silently degrade quality: feed both
the same audio and diff the samples.

Quality beyond "uncorrelated" is not needed -- this seeds inaudible jitter and a
noise bed, not cryptography -- so a generator that is trivially portable wins
over a better one that is not.
"""
from __future__ import annotations

import math

import numpy as np

MASK32 = 0xFFFFFFFF


def _rotl(x: int, k: int) -> int:
    return ((x << k) | (x >> (32 - k))) & MASK32


class Prng:
    """Deterministic 32-bit generator; see module docstring."""

    def __init__(self, seed: int = 0x5EED) -> None:
        s = seed & MASK32
        state = []
        for _ in range(4):
            s = (s + 0x9E3779B9) & MASK32
            z = s
            z = ((z ^ (z >> 16)) * 0x21F0AAAD) & MASK32
            z = ((z ^ (z >> 15)) * 0x735A2D97) & MASK32
            state.append((z ^ (z >> 15)) & MASK32)
        self._s = state
        self._spare: float | None = None

    def next(self) -> int:
        s = self._s
        result = (_rotl((s[1] * 5) & MASK32, 7) * 9) & MASK32
        t = (s[1] << 9) & MASK32
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = _rotl(s[3], 11)
        return result

    def uniform(self) -> float:
        return (self.next() >> 8) * (1.0 / 16777216.0)

    def range(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.uniform()

    def normal(self) -> float:
        if self._spare is not None:
            value, self._spare = self._spare, None
            return value
        u1 = self.uniform()
        if u1 < 1e-12:
            u1 = 1e-12
        u2 = self.uniform()
        r = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2
        self._spare = r * math.sin(theta)
        return r * math.cos(theta)

    def normals(self, count: int) -> np.ndarray:
        """``count`` standard normals as an array."""
        return np.fromiter((self.normal() for _ in range(count)),
                           dtype=np.float64, count=count)
