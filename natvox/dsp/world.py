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

#: How much longer the cross-fade must be than the alignment search.
#:
#: Aligning the seam moves where the next hop is read from, so the fade has
#: to be long enough to slide over that move rather than step across it.
#: Measured at hop 30 ms: a 12 ms fade over a 3 ms search stepped, 20 ms over
#: the same search did not.
MIN_FADE_PER_ALIGN = 6

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
    time.  So the stream is cut into windows -- a hop with context on each
    side that is analysed and then discarded -- and the converted hops are
    spliced back together.

    **The splice is the whole problem.**  ``pw.synthesize`` starts its phase
    accumulator at zero on every call, so two windows covering the same
    instant put their pitch pulses in different places.  Cross-fading between
    them cancels harmonics, and the shorter the hop the more seams there are
    to cancel at.  Measured at +10 semitones, cross-fading straight:

        hop 80 ms   21.6 dB        hop 40 ms   17.0 dB        hop 20 ms   8.2 dB

    Sliding the incoming window against the tail already written, and taking
    the offset where they agree best, removes it entirely:

        hop 80 ms   23.5 dB        hop 40 ms   23.5 dB        hop 20 ms   23.6 dB

    -- flat in the hop, and equal to converting the whole file at once.  That
    is what makes a short hop affordable, and the hop is half the delay.

    Interface-compatible with :class:`~natvox.engine.VoiceChanger`, so the
    live path can hold either.

    **It converts at 24 kHz whatever the stream's rate is**, which costs a
    tenth of a decibel and buys seven times the speed:

        48000 Hz internal   x1.3 real time   23.5 dB
        24000 Hz internal   x9.4 real time   23.4 dB

    The window is analysed ``(hop + 2*context) / hop`` times over, so that
    headroom is what a short hop spends.
    """

    def __init__(self, sample_rate: int, pitch_semitones: float = 0.0,
                 formant_semitones: float = 0.0, breathiness: float = 0.0,
                 hop_ms: float = 30.0, context_ms: float = 45.0,
                 crossfade_ms: float = 20.0, align_ms: float = 3.0,
                 internal_rate: int = INTERNAL_RATE,
                 f0_min: float = 70.0, f0_max: float = 800.0) -> None:
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

        per_ms = self.sample_rate / 1000.0
        self.hop = max(1, int(hop_ms * per_ms))
        self.context = max(1, int(context_ms * per_ms))
        self.crossfade = max(2, int(crossfade_ms * per_ms))
        self.align = max(0, int(align_ms * per_ms))
        if self.crossfade + self.align > self.context:
            raise ValueError("crossfade plus alignment must fit in the context")
        if self.crossfade < MIN_FADE_PER_ALIGN * self.align:
            raise ValueError(
                f"crossfade_ms must be at least {MIN_FADE_PER_ALIGN} times "
                f"align_ms: the alignment moves where the next hop is read "
                f"from, and a fade shorter than that steps rather than slides")
        self._fade = np.hanning(2 * self.crossfade)
        self.reset()

    # -------------------------------------------------------------- interface
    @property
    def latency_samples(self) -> int:
        """One hop to fill the window, plus the context that follows it."""
        return self.hop + self.context

    @property
    def latency_ms(self) -> float:
        return 1000.0 * self.latency_samples / self.sample_rate

    def reset(self) -> None:
        # Primed with one context of silence, so the first window is ready
        # after hop + context samples rather than hop + 2*context.  Without
        # it the output queue runs dry once, early, and the only thing left
        # to do is insert silence -- which shifts everything after it, by a
        # different amount for every block size.
        self._held = np.zeros(self.context)     # input not yet consumed
        self._tail = np.zeros(0)                # overlap awaiting its partner
        self._out = np.zeros(self.latency_samples)
        self._primed = False

    def set(self, profile) -> None:
        """Move the settings under a running stream.

        The window function reads these every hop, so a change lands within
        one of them with no rebuild and nothing to cross-fade.  What cannot
        move is the tracked pitch range: it sizes the context, and changing
        that mid-stream would move the delay under whoever is listening.
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

    def process(self, block: np.ndarray) -> np.ndarray:
        x = np.asarray(block, dtype=np.float64).reshape(-1)
        if not self._primed:
            self._out = np.zeros(self.latency_samples)
            self._primed = True
        self._held = np.concatenate([self._held, x])

        span = self.hop + 2 * self.context
        while self._held.size >= span:
            self._emit(self._held[:span])
            self._held = self._held[self.hop:]

        if self._out.size < x.size:
            self._out = np.concatenate(
                [np.zeros(x.size - self._out.size), self._out])
        out, self._out = self._out[:x.size], self._out[x.size:]
        return out

    def flush(self) -> np.ndarray:
        """Whatever is still inside, so an offline caller loses no tail."""
        return self.process(np.zeros(self.latency_samples))

    # ----------------------------------------------------------------- inside
    def _emit(self, window: np.ndarray) -> None:
        converted = self._convert(window)
        keep = self.hop + self.crossfade
        if self._tail.size != self.crossfade:
            # The first hop follows the silence the stream was primed with, and
            # walking straight out of it is a step. Fade in over the same
            # length every other seam is faded over.
            core = converted[self.context:self.context + keep].copy()
            core[:self.crossfade] *= self._fade[:self.crossfade]
            self._out = np.concatenate([self._out, core[:self.hop]])
            self._tail = core[self.hop:].copy()
            return
        offset = self.context + self._best_offset(converted)
        core = converted[offset:offset + keep]
        if core.size < keep:                    # too near the end to slide
            core = converted[self.context:self.context + keep]
        head = (self._tail * self._fade[self.crossfade:]
                + core[:self.crossfade] * self._fade[:self.crossfade])
        self._out = np.concatenate([self._out, head, core[self.crossfade:self.hop]])
        self._tail = core[self.hop:].copy()

    def _best_offset(self, converted: np.ndarray) -> int:
        """Where this window agrees with what has already been written.

        Normalised, so it picks the offset whose *shape* matches rather than
        whichever one happens to be loudest.
        """
        if not self.align:
            return 0
        best, best_score = 0, -np.inf
        norm = np.linalg.norm(self._tail)
        if norm < 1e-12:
            return 0
        for lag in range(-self.align, self.align + 1):
            start = self.context + lag
            segment = converted[start:start + self.crossfade]
            if segment.size < self.crossfade:
                continue
            score = float(self._tail.dot(segment)
                          / (np.linalg.norm(segment) + 1e-12))
            if score > best_score:
                best_score, best = score, lag
        return best

    def _convert(self, window: np.ndarray) -> np.ndarray:
        if self.internal_rate == self.sample_rate:
            return convert(window, self.sample_rate, self.pitch_semitones,
                           self.formant_semitones, self.breathiness, fast=True)
        small = _resample(window, self.sample_rate, self.internal_rate)
        done = convert(small, self.internal_rate, self.pitch_semitones,
                       self.formant_semitones, self.breathiness, fast=True)
        back = _resample(done, self.internal_rate, self.sample_rate)
        out = np.zeros(window.size)
        take = min(window.size, back.size)
        out[:take] = back[:take]
        return out


def _resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    from math import gcd

    from scipy.signal import resample_poly

    if from_rate == to_rate:
        return audio
    divisor = gcd(int(from_rate), int(to_rate))
    return resample_poly(audio, to_rate // divisor, from_rate // divisor)
