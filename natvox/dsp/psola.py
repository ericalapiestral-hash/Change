"""Grain extraction and placement -- the part that actually changes the voice.

The whole engine rests on one observation: a windowed grain spanning two pitch
periods is, to a good approximation, the vocal tract's response to a single
glottal pulse.  That gives two independent controls with no filtering at all:

* **Pitch** is set by how far apart grains are *placed*.  Nothing about the
  grain's content changes, so the formants ride along untouched -- no
  chipmunk effect, and no envelope estimation to get wrong.
* **Formants** are set by *resampling the grain*.  Squeezing an impulse
  response in time stretches its spectrum; the pitch is unaffected because
  pitch lives in the placement, not in the grain length.

Because no phase vocoder is involved, the waveform inside each grain is the
original recorded waveform.  That is why the result keeps the speaker's
natural timbre instead of the smeared, metallic quality of FFT-based shifters.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .util import WINDOWS, resample_grain


@dataclass(frozen=True)
class Mark:
    """One analysis pitch mark.

    ``deviation`` is how far the real glottal pulse sat from where the
    smoothed pitch estimate predicted it.  It is the speaker's own
    period-to-period irregularity, measured for free while phase-locking the
    mark, and carrying it through to synthesis is what stops the output being
    more perfectly periodic than the voice that went in.
    """

    position: int
    period: float
    voiced: bool
    deviation: float = 0.0


def grain_half_length(period: float, pitch_ratio: float, formant_ratio: float,
                      max_scale: float = 1.6) -> int:
    """Half-length of the grain window, in samples.

    One period either side of the mark is the textbook choice and is what
    preserves the spectral envelope.  It is widened when the formant ratio
    outruns the pitch ratio -- lowering pitch while raising formants is the
    case that reaches it.

    What the widening buys is not, as it first appears, the avoidance of gaps
    between grains: the spacing only exceeds the grain when
    ``formant_ratio / (2 * pitch_ratio) > 1``, which no shipped preset comes
    near.  What it actually buys is a flatter summed window for overlap-add to
    divide by.  That was established by measurement rather than argument, and
    the measurement contradicted the theory: forcing the scale back to the
    textbook 1.0 makes the output worse on both spectral envelope error
    (3.2 -> 5.0 dB) and inharmonic energy (-57.9 -> -54.7 dB).  The cap is a
    real three-way trade between envelope accuracy, inharmonic energy and
    latency; 1.6 was measured, not assumed.
    """
    scale = min(max(1.0, formant_ratio / max(pitch_ratio, 1e-6)), max_scale)
    return max(8, int(round(period * scale)))


def build_grain(view, mark: int, half: int, formant_ratio: float,
                fractional_delay: float = 0.0, resampler=None):
    """Window a grain around ``mark``, resample it, and shift it sub-sample.

    Returns ``(grain, window)``; the window is handed to the accumulator so
    overlap-add can normalise by how much coverage each output sample got.
    ``fractional_delay`` is the part of the target position that falls between
    samples -- see :mod:`natvox.dsp.resample` for why it matters.  ``resampler``
    is the engine's prebuilt kernel; without one a cached kernel is looked up
    by ratio, which is fine outside the real-time path.
    """
    length = 2 * half
    raw = view(mark - half, mark + half)
    window = WINDOWS.get(length)
    grain = raw * window

    same_length = abs(formant_ratio - 1.0) < 1e-4
    if same_length and abs(fractional_delay) <= 1e-4:
        return grain, window

    # Shorter grain -> spectrum stretched upward -> formants raised.
    out_len = length if same_length else max(8, int(round(length / formant_ratio)))
    if resampler is not None:
        resampled = resampler(grain, out_len, fractional_delay)
    else:
        resampled = resample_grain(grain, out_len, fractional_delay)
    return resampled, WINDOWS.get(out_len)


def nearest_mark(marks, position: float, start_hint: int = 0) -> int:
    """Index of the mark closest in time to ``position``.

    Choosing by time (rather than walking analysis and synthesis marks in
    lockstep) is what keeps duration identical to the input: grains are
    naturally repeated when pitch goes up and skipped when it goes down,
    without any accumulating drift.
    """
    i = start_hint
    n = len(marks)
    while i + 1 < n and marks[i + 1].position <= position:
        i += 1
    if i + 1 < n:
        if abs(marks[i + 1].position - position) < abs(marks[i].position - position):
            return i + 1
    return i
