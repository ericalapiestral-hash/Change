"""Streaming voice changer.

Call :meth:`VoiceChanger.process` with successive blocks of audio; it returns
the same number of samples each time, delayed by
:attr:`VoiceChanger.latency_samples`.  The block size is free and may vary,
which is what a real-time audio callback needs.

Pipeline for one block::

    high-pass -> F0 track -> pitch marks -> PSOLA grains -> loudness match -> limiter

Voiced and unvoiced audio take deliberately different routes.  Voiced audio is
rebuilt grain by grain; unvoiced audio is overlap-added at 50% with the same
Hann window, which reconstructs it exactly.  Consonants therefore stay as
crisp as they were recorded instead of acquiring the pitched buzz that gives
most voice changers away.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from .config import INTONATION_LIMIT_ST, TILT_PIVOT_HZ, VoiceProfile
from .dsp.epochs import EpochTracker
from .dsp.f0 import YinF0Tracker
from .dsp.prng import CounterNoise, Prng
from .dsp.psola import Mark, build_grain, grain_half_length, nearest_mark
from .dsp.resample import GrainResampler
from .dsp.util import (
    BiquadHighpass,
    OverlapAccumulator,
    PeakLimiter,
    RingBuffer,
    RmsMatcher,
    TiltFilter,
    round_half_up,
    soft_clip,
)

#: Pitch is re-estimated this often.  Fine enough to follow speech intonation,
#: coarse enough that tracking costs well under 2% of a core.
F0_HOP_SECONDS = 0.005

#: Spacing of the pseudo pitch marks used where there is no pitch to lock to.
UNVOICED_HOP_SECONDS = 0.006

#: Random spread applied to unvoiced grain spacing, as a fraction of the hop.
#: Grains laid at a fixed rate stamp that rate onto the audio as an amplitude
#: modulation -- measured at +17 dB over the noise floor at 167 Hz, i.e. an
#: audible buzz on every fricative, which is exactly the tell this engine
#: exists to avoid.  Spreading the spacing turns that line into broadband
#: noise far below the signal.  Reconstruction stays exact regardless of
#: spacing, because overlap-add divides by the window sum it actually got.
UNVOICED_JITTER = 0.25

#: How hard unvoiced synthesis is pulled back onto the analysis grid each mark.
#: Leaving a voiced run generally leaves synthesis up to half a period out of
#: step, and only exact alignment makes unvoiced audio a true passthrough.
#: Snapping would put a step in the middle of a consonant, so it is eased back
#: instead -- fast enough to converge inside a short fricative, slow enough
#: that the implied time-warp stays far below anything audible.
UNVOICED_RELOCK = 0.45

#: How much of the speaker's own period-to-period irregularity to carry
#: through to grain placement.
#:
#: A real voice jitters by a few tenths of a percent from one glottal pulse to
#: the next.  Placing grains on a smoothed pitch estimate throws that away and
#: hands back a voice measurably *more* periodic than the speaker -- jitter cut
#: to about 0.6x and harmonics-to-noise ratio pushed above the input's own.
#: Nothing in the artifact metrics can see it, because they are measured on a
#: synthetic vowel with no jitter to lose, and yet over-regularity is the
#: oldest robotic tell there is.  Carrying the deviation back at less than
#: unity is deliberate: the tracker's estimate of it is itself noisy, and
#: restoring it fully would add tracking noise on top of the real irregularity.
MICRO_TIMING = 0.35

#: Short-term to average energy ratio above which an unvoiced grain is treated
#: as containing a transient.
#:
#: A stop release is about three milliseconds long -- shorter than one unvoiced
#: grain -- so it straddles two of them, and resampling each about its own
#: centre displaces the burst by a different amount in each.  Overlap-add then
#: sums two copies a millisecond apart and the consonant is heard doubled.
#: Measured: correlation against the input falls from 0.999 to 0.28-0.84 on
#: /t/, /p/ and /k/ when consonant shifting is on.  Grains carrying a transient
#: are therefore passed through unresampled, which costs a small spectral
#: inconsistency between a release and the vowel after it and buys back an
#: intact consonant.  Bursts measure 4.4-4.6 on this statistic against at most
#: 2.3 for steady fricatives, so the threshold sits in open space.
TRANSIENT_CREST = 2.8

#: Averaging window for the short-term term above (0.5 ms).
TRANSIENT_WINDOW_SECONDS = 0.0005

#: Time constant of the running pitch average that intonation expansion works
#: against.
#:
#: This is the one number that decides what "intonation" means here.  Too
#: short and the average tracks the contour it is supposed to be measured
#: against, so nothing is left to expand; too long and it stops being
#: intonation at all and becomes the speaker's habitual pitch, at which point
#: expansion is just a slow, unbounded transpose.  Measured on the reference
#: utterance, the range actually delivered for a setting of 1.22 runs 1.058x
#: at 0.3 s, 1.149x at 1.5 s and 1.191x at 4 s -- the setting is a gain on the
#: deviation, and the time constant decides how much of the contour counts as
#: deviation in the first place.
#:
#: 1.5 s sits above phrase length and below register, which is the band where
#: the gender difference in F0 range is actually measured.  It is also why the
#: setting is documented as a gain rather than as a promised range ratio:
#: turning it into one would mean dividing by a factor that depends on the
#: speaker's own contour, which is not a constant and would be a lie in both
#: directions.
INTONATION_TC = 1.50

#: Output ceiling, and how far ahead the limiter looks to reach it.
#:
#: 1.5 ms is enough for the gain to arrive before any peak a voice can produce
#: and short enough to disappear into a latency budget that is already 60 ms.
#: The release is long enough not to pump on a syllable and short enough that
#: one shout does not duck the sentence after it.
#:
#: The engine used to rely on a static soft clipper here, which distorts by
#: construction: -31 dB of total harmonic distortion at full scale, -18 dB at
#: 1.4x, -13 dB at 2x.  Shouting into a microphone reaches all three, and none
#: of the artifact metrics could see any of it, because waveshaping products
#: land on exact multiples of F0 and get counted as signal.
LIMITER_CEILING = 0.97
LIMITER_LOOKAHEAD_MS = 1.5
LIMITER_RELEASE_MS = 80.0

#: Where the soft clipper now sits: above the limiter's guaranteed ceiling, so
#: it is a backstop against arithmetic surprises rather than part of the sound.
#: It is kept because "the output cannot exceed 1.0" should not depend on a
#: proof holding.
SAFETY_CEILING = 0.995

#: Aspiration noise level at ``breathiness == 1.0``, relative to the local
#: envelope of the signal *within the aspiration band*.  Larger than it looks:
#: the band holds a small fraction of a vowel's energy, so a multiplier of the
#: in-band envelope is a much smaller multiplier of the voice.
ASPIRATION_DEPTH = 4.0




class VoiceChanger:
    """Real-time pitch and formant shifter.

    Parameters
    ----------
    sample_rate:
        Input/output rate in Hz.  44100 or 48000 are typical.
    profile:
        Settings; see :class:`natvox.config.VoiceProfile`.
    """

    def __init__(self, sample_rate: int, profile: VoiceProfile | None = None) -> None:
        self.sample_rate = int(sample_rate)
        self.profile = profile or VoiceProfile()

        p = self.profile
        self._pitch_ratio = p.pitch_ratio
        self._formant_ratio = p.formant_ratio
        self._intonation = float(p.intonation)
        # Intonation expansion makes the pitch ratio vary per glottal pulse,
        # so everything sized from the ratio has to be sized from the end of
        # its range that asks for the most.  Longer grains come from the
        # *lower* end, so that is the one the buffers are budgeted against.
        span = INTONATION_LIMIT_ST if abs(self._intonation - 1.0) > 1e-9 else 0.0
        self._ratio_lo = self._pitch_ratio * 2.0 ** (-span / 12.0)
        self._ratio_hi = self._pitch_ratio * 2.0 ** (span / 12.0)

        self._f0 = YinF0Tracker(sample_rate, p.f0_min, p.f0_max)
        self._epochs = EpochTracker()
        self._highpass = BiquadHighpass(sample_rate, p.highpass_hz)
        self._loudness = RmsMatcher(sample_rate)
        self._f0_hop = max(1, round_half_up((F0_HOP_SECONDS * sample_rate)))
        self._onset_lookahead = max(0, round_half_up((p.onset_lookahead_ms * sample_rate / 1000.0)))
        self._transient_window = max(4, round_half_up((TRANSIENT_WINDOW_SECONDS * sample_rate)))
        self._unvoiced_hop = max(8, round_half_up((UNVOICED_HOP_SECONDS * sample_rate)))

        # Worst-case grain reach decides the delay: emitting a sample needs
        # every grain that overlaps it, and the grain furthest ahead was cut
        # from input one further grain-length beyond that.
        # The tracker clamps periods to its own tau_max, which rounds up past
        # sample_rate/f0_min; budgeting from the nominal value leaves the
        # longest grains a couple of samples short of their buffer.
        longest_period = float(self._f0.tau_max)
        self._max_half = max(
            grain_half_length(longest_period, self._ratio_lo, self._formant_ratio),
            round_half_up((self._unvoiced_hop * (1.0 + UNVOICED_JITTER))),
        )
        # Grains are laid down centred on their target, and lowering formants
        # stretches them, so the furthest a grain reaches past its centre is
        # the grain half-length divided by the formant ratio.
        self._max_window_half = int(
            np.ceil(self._max_half / min(self._formant_ratio, 1.0))
        ) + 1
        # Emitting a sample needs every grain that overlaps it, and the last
        # of those cannot be placed until its mark exists -- which waits on
        # both the buffer reaching the grain's far edge and pitch tracking
        # resolving that far ahead.  One pitch hop of margin keeps the budget
        # off the boundary, where it would otherwise depend on block size.
        # Phase-locking is allowed to pull a mark back by up to
        # search_fraction of a period, so the newest mark can sit that much
        # earlier than predicted -- and grain coverage reaches that much less
        # far ahead than the gate above suggests.
        # How far behind the newest sample the newest mark can sit.  Mark
        # creation is gated on the *next* mark's requirements, so once it
        # declines, the newest existing mark is already a full period behind
        # that gate -- which is why `longest_period` appears here and not only
        # inside the gate itself.  The gate itself waits for whichever comes
        # later: enough buffer for the next grain, or pitch resolved that far.
        epoch_slack = int(np.ceil(self._epochs.search_fraction * longest_period))
        mark_lag = int(longest_period) + max(
            epoch_slack + self._max_half,
            self._f0.lookahead + self._f0_hop + self._onset_lookahead,
        )
        self._latency = self._max_window_half + mark_lag + self._f0_hop
        self._acc_retention = self._max_window_half + self._max_half + self._f0_hop

        capacity = max(1 << 14, 8 * self._latency)
        self._in = RingBuffer(capacity)
        self._dry = RingBuffer(capacity)
        self._acc = OverlapAccumulator(capacity)

        self._gain = 10.0 ** (p.output_gain_db / 20.0)
        # Built once: the formant ratio is fixed for the life of a
        # configuration, so the kernel never has to be rebuilt mid-stream.
        self._resampler = GrainResampler(self._formant_ratio)
        self._unity_resampler = GrainResampler(1.0)
        self._tilt = TiltFilter(sample_rate, p.tilt_db, TILT_PIVOT_HZ)
        self._limiter = PeakLimiter(sample_rate, LIMITER_CEILING,
                                    LIMITER_LOOKAHEAD_MS, LIMITER_RELEASE_MS)
        nyquist = sample_rate * 0.5
        # A high-pass followed by a low-pass rather than a true Butterworth
        # band-pass. The shape barely differs for a noise bed, and the browser
        # build can reproduce a cascade of two biquads exactly -- which is what
        # lets the two implementations be diffed sample for sample instead of
        # only compared statistically.
        self._breath_sos = np.vstack([
            signal.butter(2, min(1500.0, nyquist * 0.9) / nyquist,
                          btype="highpass", output="sos"),
            signal.butter(2, min(7000.0, nyquist * 0.95) / nyquist,
                          btype="lowpass", output="sos"),
        ])
        env_tc = 0.010
        alpha = float(np.exp(-1.0 / (env_tc * sample_rate)))
        self._env_b, self._env_a = [1.0 - alpha], [1.0, -alpha]
        self._reset_state()
        self._prime()

    # ------------------------------------------------------------------ setup
    def _reset_state(self) -> None:
        self._primed = False
        self._frames: list = []
        self._frame_hint = 0
        self._marks: list[Mark] = []
        self._mark_hint = 0
        self._f0_pos = self._f0.half
        self._last_mark = 0
        self._last_voiced = False
        # Running mean of log period, in natural log of samples.  ``None``
        # until the first voiced mark, so the expansion starts from the
        # speaker's actual pitch rather than sliding down from a guess.
        self._log_period_mean: float | None = None
        self._synth_pos: float | None = None
        self._out_pos = 0
        self._breath_zi = np.zeros((self._breath_sos.shape[0], 2))
        self._breath_key_zi = np.zeros((self._breath_sos.shape[0], 2))
        self._breath_env_zi = np.zeros(1)
        # Separate streams: sharing one generator would interleave the two
        # draw sequences differently depending on how many marks a given block
        # happened to produce, making the output block-size dependent.
        # Both are the portable generator rather than numpy's, so that the
        # browser build produces the same samples and the two can be diffed.
        self._jitter_rng = Prng(0x5EED)
        self._noise_rng = CounterNoise(0xB2EA7)

    def _prime(self) -> None:
        """Fill the delay line and run the analysis it implies.

        This used to happen inside the first :meth:`process` call, which made
        that one callback do the pitch tracking for a whole latency of padding
        -- measured at 2.0-2.8 ms, over the deadline at 128 frames and twice it
        at 64, so the very first callback was guaranteed to drop out.  The work
        is identical wherever it runs, since every stage is a catch-up loop
        gated on the buffer, so it is done at construction where overrunning
        costs nothing.
        """
        if self._primed:
            return
        pad = np.zeros(self._latency)
        self._in.push(pad)
        self._dry.push(pad)
        self._primed = True
        self._track_pitch()
        self._extend_marks()
        self._synthesise()

    def reset(self) -> None:
        """Clear all state; use between unrelated streams."""
        self._f0.reset()
        self._epochs.reset()
        self._highpass = BiquadHighpass(self.sample_rate, self.profile.highpass_hz)
        self._loudness = RmsMatcher(self.sample_rate)
        self._tilt = TiltFilter(self.sample_rate, self.profile.tilt_db, TILT_PIVOT_HZ)
        self._limiter.reset()
        self._in = RingBuffer(self._in._buf.size)
        self._dry = RingBuffer(self._dry._buf.size)
        self._acc = OverlapAccumulator(self._acc._sig.size)
        self._reset_state()
        self._prime()

    @property
    def latency_samples(self) -> int:
        """Delay from input to output, in samples.

        The analysis delay plus the limiter's look-ahead.  The two are kept
        apart internally because only the first is padding that has to be fed
        through the buffers at construction; the second is a delay line the
        limiter carries itself.
        """
        return self._latency + self._limiter.latency_samples

    @property
    def latency_ms(self) -> float:
        return 1000.0 * self.latency_samples / self.sample_rate

    @property
    def observed_pitch(self) -> tuple[float, bool]:
        """``(hz, voiced)`` from the newest pitch frame; ``(0.0, False)`` if none.

        Published because a meter that re-runs its own tracker over the output
        would be measuring a different thing from the one the engine acted on,
        and would disagree with it at exactly the moments that matter -- the
        onsets and the creaky phrase ends where voicing is in doubt.
        """
        if not self._frames:
            return 0.0, False
        frame = self._frames[-1]
        return (float(frame.f0), bool(frame.voiced))

    # ---------------------------------------------------------------- process
    def process(self, block: np.ndarray) -> np.ndarray:
        """Transform one block and return the same number of samples."""
        x = np.asarray(block, dtype=np.float64).reshape(-1)
        # One NaN or Inf from the device -- an xrun, an interface unplugged
        # mid-stream, a misbehaving upstream plugin -- otherwise reaches pitch
        # tracking, raises out of the audio callback and stops the stream for
        # good, with the buffers growing without bound behind it.
        if x.size and not np.isfinite(x).all():
            x = np.nan_to_num(x, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
        self._prime()

        filtered = self._highpass(x)
        self._in.push(filtered)
        self._dry.push(filtered)

        self._track_pitch()
        self._extend_marks()
        self._synthesise()

        n = x.size
        dry = self._dry.view(self._out_pos, self._out_pos + n)
        if self.profile.is_identity:
            wet = dry
        else:
            wet = self._acc.read(self._out_pos, self._out_pos + n)
            # Tilt before the loudness match, so the match cancels whatever
            # broadband level the slope implies and leaves only its colour.
            wet = self._tilt(wet)
            wet = self._loudness(dry, wet)
            if self.profile.breathiness > 0.0:
                voiced = self._acc.voiced_share(self._out_pos, self._out_pos + n)
                wet = self._add_aspiration(wet, voiced)
        self._out_pos += n
        self._prune()
        # Limit, then clip.  The limiter is what keeps a shout intact; the
        # clipper is what keeps a bug from reaching the speakers.
        return soft_clip(self._limiter(wet * self._gain),
                         ceiling=SAFETY_CEILING, knee=LIMITER_CEILING)

    def flush(self) -> np.ndarray:
        """Remaining output once the input has ended (``latency_samples`` long)."""
        return self.process(np.zeros(self.latency_samples))

    # --------------------------------------------------------------- internals
    def _track_pitch(self) -> None:
        f0, end = self._f0, self._in.end
        while self._f0_pos + f0.lookahead <= end:
            start = self._f0_pos - f0.half
            segment = self._in.view(start, start + f0.span)
            self._frames.append(f0.estimate(segment, self._f0_pos))
            self._f0_pos += self._f0_hop

    def _frame_at(self, position: float):
        """Most recent pitch observation at or before ``position``.

        Falling back to the newest frame when ``position`` runs ahead is
        deliberate: F0 moves slowly enough that a few milliseconds of staleness
        is harmless, and it keeps pitch tracking from adding to the delay.
        """
        frames = self._frames
        if not frames:
            return None
        i = min(self._frame_hint, len(frames) - 1)
        while i + 1 < len(frames) and frames[i + 1].position <= position:
            i += 1
        while i > 0 and frames[i].position > position:
            i -= 1
        self._frame_hint = i
        return frames[i]

    def _extend_marks(self) -> None:
        """Place analysis pitch marks as far ahead as the buffer allows.

        A mark is created only once pitch tracking has reached the *previous*
        mark, and uses the estimate at that point.  What matters for
        determinism is that the frame chosen is a fixed function of the mark,
        not of how much audio happens to have arrived; waiting for pitch to
        resolve a further period ahead would also be deterministic but would
        hold every mark back by a period, and latency is budgeted from how far
        behind the newest mark can sit.
        """
        end = self._in.end
        f0 = self._f0
        horizon = self._last_mark + self._onset_lookahead
        while True:
            if not self._frames or self._frames[-1].position < horizon:
                return  # pitch is not yet resolved far enough ahead
            frame = self._frame_at(self._last_mark)
            if frame is None:
                return
            if not frame.voiced:
                frame = self._earliest_voiced_within(self._last_mark, horizon) or frame

            if frame.voiced:
                period = float(np.clip(frame.period, f0.tau_min, f0.tau_max))
                # Computed, not committed: this loop may still decline to
                # create the mark below, and the running average must advance
                # once per mark that exists rather than once per attempt.
                # Attempts depend on how often the caller calls us.
                ratio, pending_mean = self._ratio_for(period)
                half = grain_half_length(period, ratio, self._formant_ratio)
                # Phase-locking may place the mark up to search_fraction of a
                # period *later* than predicted, and the grain then needs a
                # half-length beyond that.  Budgeting only to the predicted
                # position leaves the occasional grain with a zero-filled tail
                # -- a real defect, and one that only appears at small block
                # sizes, where the buffer has advanced less by the time the
                # grain is cut.
                search = round_half_up((period * self._epochs.search_fraction))
                reach = search + max(half, round_half_up((period * 0.85)))
                predicted = self._last_mark + round_half_up((period))
                if predicted + reach > end:
                    return
                if self._last_voiced:
                    mark = self._epochs.locate(self._in, predicted, self._last_mark, period)
                else:
                    mark = self._epochs.bootstrap(self._in, self._last_mark, period)
                mark = max(mark, self._last_mark + 1)
                deviation = float(mark - predicted)
            else:
                ratio, pending_mean = self._pitch_ratio, self._log_period_mean
                period = float(self._unvoiced_hop)
                # Test the worst-case gap before drawing, so a draw is never
                # consumed by a mark we then decline to create.
                widest = round_half_up((self._unvoiced_hop * (1.0 + UNVOICED_JITTER)))
                if self._last_mark + widest + self._unvoiced_hop > end:
                    return
                spread = 1.0 + self._jitter_rng.range(-UNVOICED_JITTER, UNVOICED_JITTER)
                mark = self._last_mark + max(8, round_half_up((self._unvoiced_hop * spread)))
                deviation = 0.0

            self._marks.append(Mark(mark, period, frame.voiced, deviation, ratio))
            self._log_period_mean = pending_mean
            self._last_mark = mark
            self._last_voiced = frame.voiced
            horizon = mark + self._onset_lookahead

    def _ratio_for(self, period: float):
        """``(ratio, mean_after)`` for a mark of this period, after expansion.

        The speaker is up or down relative to their own running average by
        ``12 * log2(mean_period / period)`` semitones; expansion multiplies
        that deviation and folds the surplus into the ratio.  Deliberately
        measured against the average *before* this mark updates it, so a mark
        is never partly compared against itself.

        Pure, and the advanced average is handed back rather than stored,
        because the caller may still decline to create this mark -- and how
        many times it tries before succeeding is a function of the audio block
        size, which nothing in the output is allowed to depend on.  Storing it
        here instead cost a measured 0.1 of full scale between a 64-sample
        block and a 1024-sample one.
        """
        if self._intonation == 1.0:
            return self._pitch_ratio, self._log_period_mean
        log_period = float(np.log(period))
        mean = self._log_period_mean
        if mean is None:
            mean = log_period
        # One-pole in the mark domain: a mark covers one period of time, so
        # the coefficient is derived from that period rather than a fixed hop.
        alpha = float(np.exp(-period / (INTONATION_TC * self.sample_rate)))
        advanced = alpha * mean + (1.0 - alpha) * log_period
        deviation_st = 12.0 * (mean - log_period) / np.log(2.0)
        extra = float(np.clip((self._intonation - 1.0) * deviation_st,
                              -INTONATION_LIMIT_ST, INTONATION_LIMIT_ST))
        ratio = float(np.clip(self._pitch_ratio * 2.0 ** (extra / 12.0),
                              self._ratio_lo, self._ratio_hi))
        return ratio, advanced

    def _earliest_voiced_within(self, start: int, stop: int):
        """First voiced pitch frame just after ``start``, if there is one.

        Pitch tracking is inherently retrospective -- it needs a couple of
        periods before it can call a frame voiced -- so at a vowel onset the
        frame *at* the mark still reads unvoiced while the vowel is already
        running.  Looking a few milliseconds ahead lets the mark be placed on
        the voiced path from the start of the syllable instead of a period or
        two into it.  The frames are already computed; the cost is the extra
        buffering needed to guarantee they exist, which is why the horizon is
        a declared setting rather than whatever happens to be available.
        """
        if stop <= start:
            return None
        for frame in self._frames:
            if frame.position <= start:
                continue
            if frame.position > stop:
                break
            if frame.voiced:
                return frame
        return None

    def _synthesise(self) -> None:
        """Lay grains down at the shifted spacing."""
        marks = self._marks
        if not marks:
            return
        if self._synth_pos is None:
            self._synth_pos = float(marks[0].position)

        limit = float(marks[-1].position)
        shift_unvoiced = self.profile.shift_unvoiced
        alpha = self._formant_ratio
        view = self._in.view

        while self._synth_pos <= limit:
            idx = nearest_mark(marks, self._synth_pos, self._mark_hint)
            self._mark_hint = idx
            mark = marks[idx]

            # Put the grain where the speaker's own glottal pulse was, not
            # where a smoothed pitch estimate says it should have been.  The
            # offset is scaled by the pitch ratio so that the *relative*
            # irregularity is preserved rather than its absolute size, and it
            # is applied to this grain only -- it must not accumulate into the
            # synthesis cursor, or the pitch itself would wander.
            target = self._synth_pos
            ratio = mark.ratio if mark.voiced else self._pitch_ratio
            if mark.voiced and MICRO_TIMING:
                # Divided by the pitch ratio only when shifting up.  Shifting
                # up repeats grains, so consecutive synthesis marks often
                # carry the same deviation and the irregularity is diluted;
                # shifting down skips grains, which decorrelates successive
                # deviations and would otherwise amplify it past the
                # speaker's own.
                #
                # The obvious objection is that this throws away the speaker's
                # jitter exactly where the shift is largest and the result is
                # most at risk of sounding mechanical.  Measured on the
                # reference utterance, against the same engine with the
                # division removed:
                #
                #   shift   +0.0  +4.5  +7.0  +9.0  +11.0  +13.0
                #   shipped  0.96  0.86  0.96  1.04   0.95   0.86   (x source)
                #   no div   0.96  0.89  0.97  1.17   0.98   1.04
                #
                # No downward trend with shift in either, and the swing between
                # adjacent shifts is larger than the gap between the columns.
                # So the objection is not visible here and the division stays.
                # If it is ever revisited, beat this table rather than the
                # argument.
                target += MICRO_TIMING * mark.deviation / max(ratio, 1.0)

            # Split the target position into a whole-sample slot and the
            # remainder, which is carried inside the grain as a phase ramp.
            base = int(np.floor(target))
            frac = float(target - base)

            if mark.voiced:
                half = grain_half_length(mark.period, ratio, alpha)
                grain, window = build_grain(view, mark.position, half, alpha, frac,
                                            self._resampler)
                # Voiced grains are pitch-synchronous, so they stay in phase
                # with each other even after resampling.
                coherent = True
                step = mark.period / ratio
            else:
                if idx + 1 >= len(marks):
                    # The gap to the next mark is random, so it cannot be
                    # guessed: waiting keeps synthesis exactly on the analysis
                    # grid instead of drifting and having to be pulled back.
                    break
                # Grain length stays fixed while spacing varies, which keeps
                # neighbouring grains overlapping enough to normalise cleanly.
                half = self._unvoiced_hop
                formant = alpha if shift_unvoiced else 1.0
                if shift_unvoiced and self._carries_transient(mark.position, half):
                    formant = 1.0
                resampler = (self._resampler if abs(formant - alpha) < 1e-9
                             else self._unity_resampler)
                grain, window = build_grain(view, mark.position, half, formant, frac,
                                            resampler)
                coherent = abs(formant - 1.0) < 1e-4
                step = 0.0  # set below, from the true gap to the next mark

            self._acc.add(base - window.size // 2, grain, window, coherent,
                          mark.voiced)

            if mark.voiced:
                self._synth_pos += step
            else:
                # Step by the real gap to the next mark, then ease back onto
                # the analysis grid so an unvoiced stretch stays a
                # sample-accurate copy rather than a slowly sliding one.
                target = float(marks[idx + 1].position)
                self._synth_pos += target - mark.position
                self._synth_pos += UNVOICED_RELOCK * (target - self._synth_pos)

    def _carries_transient(self, centre: int, half: int) -> bool:
        """Whether this unvoiced grain contains a plosive release.

        Peak short-term energy against the grain's mean: a burst concentrates
        almost all of its energy into a fraction of the grain, a fricative
        spreads it evenly.
        """
        segment = self._in.view(centre - half, centre + half)
        power = segment * segment
        mean = float(np.mean(power))
        if mean <= 1e-18:
            return False
        # A trailing window rather than a centred one, purely so the browser
        # port can compute the identical statistic with a running sum.
        window = self._transient_window
        if power.size < window:
            return False
        cumulative = np.concatenate(([0.0], np.cumsum(power)))
        sums = cumulative[window:] - cumulative[:-window]
        return bool(np.sqrt(float(np.max(sums)) / window / mean) > TRANSIENT_CREST)

    def _add_aspiration(self, wet: np.ndarray, voiced: np.ndarray) -> np.ndarray:
        """Mix a little aspiration noise into the voiced audio.

        A large upward pitch shift spreads the harmonics apart and thins the
        spectrum out; real voices fill that region with breath.  The noise is
        band-limited to where aspiration actually lives (roughly 1.5-7 kHz)
        rather than being broadband, so it reads as breath instead of as hiss,
        and it is gated by the signal's own envelope so silence stays silent.

        It is also gated by how voiced each sample is, which is the difference
        between aspiration and hiss.  Aspiration is noise leaking through a
        glottis that is not closing completely -- it happens *during phonation*
        and nowhere else.  Adding it to a fricative instead lands noise on top
        of noise, which does not sound breathy, only noisier: measured at
        +2.8 dB of added noise across /s/ and /f/ for nothing.  The per-sample
        voicing weight comes from the overlap-add windows themselves, so the
        gate follows the same grains the audio did.

        The envelope that sets its level is taken from the signal *in the same
        band as the noise*, not from the signal as a whole.  In a real voice
        the aspiration is generated at the glottis and then filtered by the
        same vocal tract as the voiced source, so it is loud on a bright vowel
        and quiet on a dark one.  Keying it off broadband level instead puts a
        constant hiss over every vowel, including the ones with nothing up
        there -- which measured as 7.2 dB of spectral envelope error against
        1.2 dB without the noise, i.e. the "breath" was reshaping the vowel.
        Keying it off the band's own energy is a first-order stand-in for that
        filtering, costs one more biquad pair, and needs no envelope
        estimation anywhere.
        """
        amount = self.profile.breathiness
        # Drawn one per output sample, so the sequence consumed is the same
        # however the caller chunks its audio.
        noise = self._noise_rng.normals(wet.size)
        shaped, self._breath_zi = signal.sosfilt(self._breath_sos, noise, zi=self._breath_zi)
        key, self._breath_key_zi = signal.sosfilt(self._breath_sos, wet,
                                                  zi=self._breath_key_zi)
        envelope, self._breath_env_zi = signal.lfilter(
            self._env_b, self._env_a, np.abs(key), zi=self._breath_env_zi
        )
        return wet + shaped * envelope * voiced * (amount * ASPIRATION_DEPTH)

    def _prune(self) -> None:
        """Release buffers and list entries that nothing can reach back to.

        The retention point is derived from what is still referenced rather
        than from a fixed margin.  Trimming even slightly too early is not a
        loud failure -- the ring buffer zero-fills what is missing -- but it
        silently corrupts whichever grain straddles the boundary, and since
        trimming happens on block boundaries, *which* grain that is depends on
        the caller's block size.
        """
        # Marks first: synthesis never revisits anything before its cursor, so
        # that cursor is the floor for how much history is still live.
        if len(self._marks) > 8:
            cutoff = self._out_pos - 2 * self._max_half
            drop = 0
            while (drop < self._mark_hint
                   and drop < len(self._marks) - 4
                   and self._marks[drop].position < cutoff):
                drop += 1
            if drop:
                del self._marks[:drop]
                self._mark_hint -= drop
        if len(self._frames) > 8:
            cutoff = self._out_pos - 2 * self._max_half
            drop = 0
            while (drop < self._frame_hint
                   and drop < len(self._frames) - 4
                   and self._frames[drop].position < cutoff):
                drop += 1
            if drop:
                del self._frames[:drop]
                self._frame_hint -= drop

        # Oldest input sample any remaining consumer can still ask for: the
        # earliest surviving grain, the pitch tracker's own window, and the
        # output cursor.
        oldest = self._out_pos - self._max_half
        if self._marks:
            oldest = min(oldest, self._marks[0].position - self._max_half)
        oldest = min(oldest, self._f0_pos - self._f0.half)
        self._in.discard_to(oldest - self._f0_hop)

        self._dry.discard_to(self._out_pos - 1)
        # Keep well over a grain's worth of already-emitted accumulator behind
        # the read point. A voiced onset lengthens grains abruptly -- an
        # unvoiced grain reaches back 6 ms, the first voiced one can reach back
        # 14 -- so the occasional grain lands just behind the cursor. Trimming
        # at the cursor would clip it, and *whether* it got clipped would
        # depend on how far the caller's block size had advanced the cursor,
        # which is exactly the kind of thing that must not vary. Retention
        # costs only memory, not delay, so it is set generously.
        self._acc.discard_to(self._out_pos - self._acc_retention)
