"""Pitch-mark (epoch) tracking.

PSOLA cuts the signal into one grain per glottal period.  Where exactly each
cut lands matters far more than people expect: if consecutive marks sit at
different phases of the period, every grain is a slightly different waveform
and the overlap-add sums them incoherently -- which is heard as roughness or a
buzzy "second voice".

Rather than trying to find the true glottal closure instant (fragile, and
unnecessary), this tracker locks each mark to the *same phase* as the previous
one by maximising normalised cross-correlation against the previous period.
Consistency is what PSOLA actually needs.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from .util import EPS, RingBuffer


class EpochTracker:
    """Phase-locked pitch-mark placement.

    Parameters
    ----------
    search_fraction:
        How far either side of the predicted position to look, as a fraction of
        the period.  Wide enough to absorb a mis-estimated period, narrow
        enough that it cannot skip to a neighbouring pulse.
    """

    def __init__(self, search_fraction: float = 0.30) -> None:
        self.search_fraction = float(search_fraction)
        self.last_confidence = 0.0

    def locate(self, buf: RingBuffer, predicted: int, reference: int, period: float) -> int:
        """Place a mark near ``predicted``, phase-locked to the one at ``reference``."""
        half = max(4, int(round(period * 0.5)))
        shift = max(1, int(round(period * self.search_fraction)))

        ref = buf.view(reference - half, reference + half)
        ref_energy = float(np.dot(ref, ref))
        if ref_energy <= EPS:
            self.last_confidence = 0.0
            return predicted

        seg = buf.view(predicted - shift - half, predicted + shift + half)
        width = 2 * half
        num = signal.correlate(seg, ref, mode="valid")
        if num.size != 2 * shift + 1:
            self.last_confidence = 0.0
            return predicted

        cum = np.concatenate(([0.0], np.cumsum(seg * seg)))
        seg_energy = cum[width:width + num.size] - cum[:num.size]
        ncc = num / np.sqrt(np.maximum(seg_energy * ref_energy, EPS))

        best = int(np.argmax(ncc))
        self.last_confidence = float(ncc[best])
        return predicted + (best - shift)

    def bootstrap(self, buf: RingBuffer, start: int, period: float) -> int:
        """First mark of a voiced run: the strongest excitation in one period.

        Any phase would do -- the correlation lock takes over from the next
        mark on -- but starting at the energy peak means the very first grain
        already carries a full pulse.
        """
        span = max(8, int(round(period)))
        x = buf.view(start, start + span)
        if x.size == 0:
            return start
        # Short-term energy; smoothing keeps a single sample of noise from
        # winning over the real excitation peak.
        win = max(3, span // 10)
        energy = np.convolve(x * x, np.ones(win) / win, mode="same")
        self.last_confidence = 1.0
        return start + int(np.argmax(energy))

    def reset(self) -> None:
        self.last_confidence = 0.0
