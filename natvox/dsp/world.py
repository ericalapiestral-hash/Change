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

#: Shortest context, as a multiple of the lowest tracked period.
#:
#: The vocoder needs several pitch periods to estimate a vocal tract response
#: at all.  Measured at 48 kHz with f0_min at 70 Hz -- a 14.3 ms period --
#: context of 80 ms gives 21.6 dB HNR, 40 ms gives 15.4, and 30 ms gives a
#: signal with no periodicity my measure can find at all: the level is right
#: and the voice is gone.  Three periods is where that starts, so it is a
#: refusal rather than a setting somebody can quietly ruin the sound with.
MIN_CONTEXT_PERIODS = 3.0

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


class LiveConverter:
    """WORLD conversion over a continuous stream, block by block.

    The vocoder wants a whole utterance; a callback hands it 256 samples at a
    time.  The gap between those is a windowing problem, and it is the same
    one every block-based conversion model has, so this leans on the machinery
    already written for those: a hop with context on both sides that is
    analysed and then discarded, and a cross-fade between consecutive outputs,
    because two windows synthesised independently do not agree at their seam
    and butting them together puts a click at every hop.

    Interface-compatible with :class:`~natvox.engine.VoiceChanger`, so the
    live path can hold either.

    **It converts at 24 kHz whatever the stream's rate is.**  Measured on the
    reference utterance at +10 semitones, with the resampling in and out
    counted:

        48000 Hz internal   x1.3 real time   23.5 dB
        24000 Hz internal   x9.4 real time   23.4 dB

    Seven times the speed for a tenth of a decibel.  The window is analysed
    three times over -- hop plus context on each side -- so x1.3 is not
    enough to run live at all, and x9.4 leaves room for a machine that is
    also doing something else.
    """

    def __init__(self, sample_rate: int, pitch_semitones: float = 0.0,
                 formant_semitones: float = 0.0, breathiness: float = 0.0,
                 hop_ms: float = 80.0, context_ms: float = 80.0,
                 crossfade_ms: float = 20.0, internal_rate: int = INTERNAL_RATE,
                 f0_min: float = 70.0, f0_max: float = 800.0) -> None:
        from ..neural.rvc import StreamingNeuralConverter

        _world()                                # fail here, not in the callback
        needed = 1000.0 * MIN_CONTEXT_PERIODS / max(float(f0_min), 1e-6)
        if context_ms < needed:
            raise ValueError(
                f"context_ms must be at least {needed:.0f} ms to track down to "
                f"{f0_min:g} Hz ({MIN_CONTEXT_PERIODS:g} periods); at less "
                "than that the vocoder returns a signal at the right level "
                "with no voice in it")
        self.sample_rate = int(sample_rate)
        self.internal_rate = int(min(internal_rate, sample_rate))
        self.pitch_semitones = float(pitch_semitones)
        self.formant_semitones = float(formant_semitones)
        self.breathiness = float(breathiness)
        self._f0_min, self._f0_max = float(f0_min), float(f0_max)
        self._stream = StreamingNeuralConverter(
            self.sample_rate, self._convert_window,
            hop_seconds=hop_ms / 1000.0, context_seconds=context_ms / 1000.0,
            crossfade_seconds=crossfade_ms / 1000.0,
            # The vocoder does its own pitch work; letting the wrapper shift
            # the F0 it passes as well would apply the interval twice.
            pitch_shift_semitones=0.0, f0_min=f0_min, f0_max=f0_max)

    # -------------------------------------------------------------- interface
    @property
    def latency_samples(self) -> int:
        return self._stream.latency_samples

    @property
    def latency_ms(self) -> float:
        return 1000.0 * self.latency_samples / self.sample_rate

    def process(self, block: np.ndarray) -> np.ndarray:
        return self._stream.process(block)

    def flush(self) -> np.ndarray:
        """Whatever is still inside, so an offline caller loses no tail."""
        return self._stream.process(np.zeros(self.latency_samples))

    def reset(self) -> None:
        self._stream.reset()

    def set(self, profile) -> None:
        """Move the settings under a running stream.

        The window function reads these every hop, so a change lands within
        one of them -- 80 ms at the defaults -- with no rebuild and nothing to
        cross-fade.  What cannot move is the tracked pitch range: it sizes the
        tracker and the context around it, and changing those mid-stream would
        move the delay under whoever is listening.
        """
        if (float(profile.f0_min) != self._f0_min
                or float(profile.f0_max) != self._f0_max):
            raise ValueError(
                "the tracked pitch range cannot change while the stream is "
                "running: it sets the delay, and moving the delay mid-sentence "
                "shifts the audio in time")
        self.pitch_semitones = float(profile.pitch_semitones)
        self.formant_semitones = float(profile.formant_semitones)
        self.breathiness = float(profile.breathiness)

    # ----------------------------------------------------------------- inside
    def _convert_window(self, audio: np.ndarray, sample_rate: int,
                        f0: np.ndarray) -> np.ndarray:
        """One window, in and out at the stream's rate.

        ``f0`` is what the wrapper measured; the vocoder re-estimates it on
        the window it is actually given, which is the one that includes the
        context.
        """
        if self.internal_rate == sample_rate:
            return convert(audio, sample_rate, self.pitch_semitones,
                           self.formant_semitones, self.breathiness, fast=True)
        small = _resample(audio, sample_rate, self.internal_rate)
        done = convert(small, self.internal_rate, self.pitch_semitones,
                       self.formant_semitones, self.breathiness, fast=True)
        back = _resample(done, self.internal_rate, sample_rate)
        out = np.zeros(audio.size)
        take = min(audio.size, back.size)
        out[:take] = back[:take]
        return out


def _resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    from math import gcd

    from scipy.signal import resample_poly

    if from_rate == to_rate:
        return audio
    divisor = gcd(int(from_rate), int(to_rate))
    return resample_poly(audio, to_rate // divisor, from_rate // divisor)
