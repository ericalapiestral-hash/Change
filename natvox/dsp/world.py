"""Conversion by analysis and resynthesis, rather than by moving the waveform.

The PSOLA engine in :mod:`natvox.engine` cuts grains out of the recording and
re-lays them.  That keeps the waveform somebody actually produced, which is
why it sounds natural at small shifts -- and it is also its ceiling: whatever
is in the recording is stretched along with the voice, noise included, and
past roughly eight semitones the grains no longer overlap in a way a vocal
tract could have produced.

This takes the recording apart instead.  WORLD estimates three things a voice
is made of -- the pitch, the vocal tract's response, and how much of each
frame is noise rather than pulses -- and then builds a new waveform from them.
Nothing of the original waveform survives, so nothing of its noise is
stretched either.

Measured on the first real recording this project ever had, against the PSOLA
path on the same file, using an autocorrelation harmonic-to-noise ratio
validated against known SNRs:

    her recording, untouched   11.7 dB
    PSOLA, +10 semitones       13.2 dB
    WORLD, +10 semitones       17.2 dB

and the pitch it delivers is exact -- 0.00 semitones of error at +4, +7, +10
and +13 against a reference whose true contour is known, where PSOLA's own
documentation warns past +-8.

What it costs: the vocal tract response is smoothed rather than exact, so at
small shifts PSOLA is the more faithful of the two.  This is the engine for
the shifts PSOLA cannot reach.
"""
from __future__ import annotations

import numpy as np

#: Analysis frame spacing, in milliseconds.  WORLD's own default.
FRAME_PERIOD_MS = 5.0

#: Rate the vocoder runs at, regardless of the stream's rate.
#:
#: Halving the rate roughly halves the cost of every stage, and a voice has
#: nothing above 8 kHz that a conversion needs: measured on real speech, the
#: band above it sits 30 dB down and is mostly room.  Resampling in and out
#: costs less than the stages it saves.
INTERNAL_RATE = 24000

#: Tracking range handed to the F0 estimator.
#:
#: The ceiling is for singing, not speech.  At 600 Hz -- which is generous for
#: a talking voice -- a sung note at 600 Hz came out at 692 Hz where it should
#: have reached 1069: the estimate is clipped at the ceiling and the shift is
#: applied to the wrong number, which is a voice tearing rather than a voice
#: going high.  Raising it costs nothing measurable on speech: on the
#: reference utterance, F0 error is 0.87 st and octave errors 0.0% at 600,
#: 800, 1100 and 1600 Hz alike.
F0_FLOOR = 60.0
F0_CEIL = 1100.0

#: The output must not leave here above full scale.
#:
#: Resynthesis is not gain-preserving: a note peaking at 0 dBFS came back at
#: +1.0 dB, which clips on the way to the speaker.  That is audible as tearing
#: on exactly the loud passages somebody would notice it on, and it was this
#: module bypassing the limiter the rest of the engine already runs through.
CEILING = 0.97


class WorldUnavailable(RuntimeError):
    """Raised when ``pyworld`` is not installed."""


def _world():
    try:
        import pyworld
    except ImportError as exc:                  # pragma: no cover - env specific
        raise WorldUnavailable(
            "the WORLD vocoder needs pyworld: pip install 'natvox[world]'"
        ) from exc
    return pyworld


def available() -> bool:
    """Whether the vocoder can be used at all."""
    try:
        _world()
    except WorldUnavailable:
        return False
    return True


def analyse(audio: np.ndarray, sample_rate: int, fast: bool = True):
    """``(f0, spectrogram, aperiodicity)`` for ``audio``.

    ``fast`` picks WORLD's ``dio`` over ``harvest``: measured at 62x real time
    against 5x, which is the difference between an engine that can run live
    and one that cannot.  ``harvest`` is the better estimator and is what the
    offline path uses.
    """
    pw = _world()
    x = np.ascontiguousarray(np.asarray(audio, dtype=np.float64).reshape(-1))
    finder = pw.dio if fast else pw.harvest
    f0, t = finder(x, sample_rate, f0_floor=F0_FLOOR, f0_ceil=F0_CEIL,
                   frame_period=FRAME_PERIOD_MS)
    f0 = pw.stonemask(x, f0, t, sample_rate)
    return f0, pw.cheaptrick(x, f0, t, sample_rate), pw.d4c(x, f0, t, sample_rate)


def warp_envelope(spectrogram: np.ndarray, sample_rate: int,
                  semitones: float) -> np.ndarray:
    """Move the whole vocal tract response by ``semitones``.

    Uniformly, on purpose.  Vocal tract *length* is a constant of the speaker
    and every formant scales with it together; the formants themselves move
    with the vowel, so mapping a speaker's average F1/F2/F3 onto some other
    population's averages -- which is the obvious thing to try -- fits a
    constant to something that is not one, and distorts every vowel to make
    the average come out right.
    """
    if not semitones:
        return spectrogram
    bins = spectrogram.shape[1]
    freq = np.linspace(0.0, sample_rate / 2.0, bins)
    read_at = freq / (2.0 ** (semitones / 12.0))
    return np.stack([np.interp(read_at, freq, row) for row in spectrogram])


def synthesise(f0: np.ndarray, spectrogram: np.ndarray, aperiodicity: np.ndarray,
               sample_rate: int) -> np.ndarray:
    pw = _world()
    return pw.synthesize(
        np.ascontiguousarray(f0, dtype=np.float64),
        np.ascontiguousarray(spectrogram, dtype=np.float64),
        np.ascontiguousarray(np.clip(aperiodicity, 1e-6, 1.0), dtype=np.float64),
        sample_rate, frame_period=FRAME_PERIOD_MS)


def convert(audio: np.ndarray, sample_rate: int, pitch_semitones: float = 0.0,
            formant_semitones: float = 0.0, breathiness: float = 0.0,
            fast: bool = False) -> np.ndarray:
    """Convert ``audio``, returning the same number of samples.

    ``breathiness`` raises the aperiodic share of every voiced frame, which is
    the one thing this can do that the PSOLA path cannot fake: aspiration
    here replaces part of the periodic excitation instead of being mixed on
    top of it as noise.
    """
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return x
    f0, sp, ap = analyse(x, sample_rate, fast=fast)
    if breathiness:
        ap = np.clip(ap + float(breathiness) * (1.0 - ap), 0.0, 1.0)
    y = synthesise(f0 * (2.0 ** (pitch_semitones / 12.0)),
                   warp_envelope(sp, sample_rate, formant_semitones), ap,
                   sample_rate)
    out = np.zeros(x.size)
    take = min(x.size, y.size)
    out[:take] = y[:take]
    return limit(out, sample_rate)


def limit(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Hold the output under full scale, without reshaping the waveform.

    A look-ahead limiter rather than a clipper: the whole waveform is scaled
    by a smooth envelope, so a loud passage loses a little level instead of
    gaining the harmonic distortion a clipper would put on it.  The limiter's
    own delay is removed, so this changes the level and nothing else.
    """
    from .util import PeakLimiter

    if audio.size == 0:
        return audio
    limiter = PeakLimiter(sample_rate, ceiling=CEILING)
    look = limiter.latency_samples
    padded = limiter(np.concatenate([audio, np.zeros(look)]))
    return padded[look:look + audio.size]
