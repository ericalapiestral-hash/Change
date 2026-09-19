"""Measuring the voice that is actually talking, so the shift is not a guess.

Every preset here is a statement about a speaker nobody has heard.  ``female``
raises pitch by 7 semitones because adult male and female F0 differ by about
9.6 and three quarters of that is where PSOLA stays transparent -- but seven
semitones lands a 100 Hz speaker at 150 Hz and a 140 Hz speaker at 210 Hz, and
only one of those is a woman's pitch.  The same preset either overshoots or
falls short depending on who bought it.

The program already tracks pitch, so it does not have to assume.  A few seconds
of ordinary speech gives the speaker's own habitual F0 and their own range, and
from those the shift is arithmetic rather than a guess.

Two decisions here are worth stating, because they are the ones that could
reasonably have gone the other way:

**Median, not mean.**  Connected speech falls at the end of every phrase and
many speakers drop into creak there, an octave or more below their habitual
pitch.  A mean is dragged down by that tail and would ask for too large a
shift.  The median ignores it, which is the whole reason to use one.

**Formant shift is fixed, not a fraction of the pitch shift.**  Vocal folds and
vocal tract do not scale together: adult male and female tract lengths differ
by about 1.17, which is 2.7 semitones, and that is true of a woman with a low
voice as much as one with a high voice.  So the tract shift is a constant and
the pitch shift is whatever the speaker needs.  Scaling the formants with the
pitch -- the "about 40% of the pitch shift" rule of thumb -- happens to give
the right answer for an average male speaker and the wrong one for everybody
else, in the direction that makes a low voice sound like a child.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import (NATURAL_PITCH_LIMIT, VoiceProfile, ratio_to_semitones,
                      semitones_to_ratio)
from ..dsp.f0 import YinF0Tracker

#: Median speaking F0 to aim a voice at, in Hz.
#:
#: Not the average of the target group, deliberately.  Adult female speaking F0
#: averages around 210 Hz, but the point where listeners stop hearing a voice
#: as male sits well below that -- the reported crossover in listening studies
#: is somewhere around 155-180 Hz.  Aiming at the average asks for the whole
#: distance when only the boundary has to be crossed, and for a low voice that
#: difference is large: 110 Hz to 210 Hz is +10.9 semitones, to 185 Hz is +8.9.
#:
#: 185 clears the top of that crossover band with a little margin.  It is a
#: starting point rather than a fact -- the crossover is a range, it is not
#: measured here, and nothing is known about whether it sits in the same place
#: for Korean -- which is why it is settable and why the shift it implies is
#: always reported rather than silently applied.
FEMALE_TARGET_HZ = 185.0
MALE_TARGET_HZ = 115.0

#: Vocal tract length shift between adult male and female, in semitones.
#:
#: From the ~1.17 length ratio.  A constant, deliberately: see the module
#: docstring.  It is what the shipped ``female`` preset already uses (2.6), and
#: that preset's apparent "40% of the pitch shift" is a coincidence of 2.6/7.0.
TRACT_SEMITONES = 2.6

#: How far the tracker looks while measuring, in Hz.
#:
#: Wider than any preset tracks with, at both ends: the point is to *see* creak
#: and falsetto rather than to convert them, and a median is not moved by a
#: tail it can see.  A range that clipped them would move the median instead.
MEASURE_F0_MIN = 55.0
MEASURE_F0_MAX = 900.0

#: How often to take an estimate while measuring, in milliseconds.
HOP_MS = 10.0

#: How much voiced speech is enough to believe the answer.
MIN_VOICED_SECONDS = 1.0

#: How far *below* the median a frame may be and still count, in semitones.
#:
#: Low side only, and that asymmetry is the whole design.  A symmetric gate
#: seems obviously right and is not: a speaker who genuinely swings between
#: 110 and 220 Hz has a median sitting in whichever mode has more frames, and
#: a gate centred there throws the other mode away -- half their range, gone,
#: because they used it.  Measured: 199 frames kept of 396, and the reported
#: range collapsed from twelve semitones to zero.
#:
#: Nothing needs trimming above the median.  A tracker finding *twice* the
#: period is the common failure and lands an octave down; finding half of it
#: is rare.  So the rule is "ignore the low tail", and it is right for the
#: thing the low tail is used for either way: ``f0_min`` should be the lowest
#: pitch worth tracking, and neither an octave error nor genuine creak is
#: that.  Tracking down to creak is most of what the latency buys.
#:
#: The first real measurement made the case.  A 126 Hz speaker read
#: "63-157 Hz, a range of 15.8 semitones", and 63 is exactly half of 126.
#: The median shrugged it off -- an order statistic ignores a tail -- but the
#: tenth percentile *is* the tail, and the converted profile's f0_min comes
#: from there.  Taken at face value it asked for a 53 Hz tracking floor:
#: worse latency than the lowest row in the README's own table, and a floor
#: low enough to invite the very error that produced it.
#:
#: Nine semitones is below any habitual tenth percentile and comfortably
#: above the twelve a halved period lands at, so it separates the two without
#: having to decide which any one frame is.
FLOOR_GATE_ST = 9.0

#: Margin below the measured floor for the converted profile's ``f0_min``.
#:
#: ``f0_min`` sets the latency floor, so a speaker who never goes below 105 Hz
#: should not pay for tracking to 75.  The margin is in semitones rather than
#: Hz so it means the same thing at every pitch, and it is generous because
#: setting it above a speaker's actual range causes octave errors, which sound
#: far worse than the latency it saves.
FLOOR_MARGIN_ST = 3.0


@dataclass
class VoicePrint:
    """What a speaker's own voice measures."""

    sample_rate: int
    #: Habitual pitch: the median of voiced frames.  See the module docstring.
    median_hz: float
    #: 10th and 90th percentiles of voiced frames, which is the range the
    #: speaker actually uses rather than the extremes they can reach.
    low_hz: float
    high_hz: float
    #: Fraction of frames the tracker called voiced.  Low means noise, whisper,
    #: or not enough speech -- all of which make the median untrustworthy.
    voiced_share: float
    voiced_seconds: float
    seconds: float
    #: Frames below the floor gate, which the range excludes.  Almost always
    #: the tracker finding twice the period, sometimes genuine creak, and
    #: neither is a pitch worth tracking down to -- see :data:`FLOOR_GATE_ST`.
    octave_errors: int = 0

    @property
    def range_semitones(self) -> float:
        """Between the 10th and 90th percentile, in semitones."""
        if self.low_hz <= 0.0 or self.high_hz <= 0.0:
            return 0.0
        return ratio_to_semitones(self.high_hz / self.low_hz)

    @property
    def usable(self) -> bool:
        return (self.median_hz > 0.0
                and self.voiced_seconds >= MIN_VOICED_SECONDS)

    def shift_to(self, target_hz: float) -> float:
        """Semitones from this voice to ``target_hz``."""
        if self.median_hz <= 0.0 or target_hz <= 0.0:
            return 0.0
        return ratio_to_semitones(target_hz / self.median_hz)

    def summary(self) -> str:
        if not self.usable:
            return ("not enough voiced speech to measure -- say a couple of "
                    f"sentences in your ordinary voice ({self.voiced_seconds:.1f}s "
                    f"of {self.seconds:.1f}s was voiced)")
        line = (f"your voice   {self.median_hz:.0f} Hz "
                f"({self.low_hz:.0f}-{self.high_hz:.0f} Hz, "
                f"a range of {self.range_semitones:.1f} semitones)\n"
                f"  measured over {self.voiced_seconds:.1f}s of voiced speech")
        if self.octave_errors:
            line += (f", ignoring a low tail of {self.octave_errors} frames "
                     "-- creak, or the tracker an octave out")
        return line


def measure(audio, sample_rate: int, f0_min: float = MEASURE_F0_MIN,
            f0_max: float = MEASURE_F0_MAX, hop_ms: float = HOP_MS) -> VoicePrint:
    """Track pitch across ``audio`` and summarise the voice in it.

    Offline and self-contained: it builds its own tracker rather than reading
    the running engine's, so the measurement does not depend on what the
    engine happens to be configured for -- which is the thing being chosen.
    """
    audio = np.asarray(audio, dtype=np.float64).reshape(-1)
    seconds = audio.size / sample_rate if sample_rate else 0.0
    tracker = YinF0Tracker(sample_rate, f0_min=f0_min, f0_max=f0_max)
    span = tracker._span                        # noqa: SLF001 - the window it needs
    hop = max(1, int(round(hop_ms * sample_rate / 1000.0)))

    pitches = []
    frames = 0
    for start in range(0, max(0, audio.size - span + 1), hop):
        frame = tracker.estimate(audio[start:start + span], start)
        frames += 1
        if frame.voiced and frame.f0 > 0.0:
            pitches.append(frame.f0)

    voiced_seconds = len(pitches) * hop / sample_rate if sample_rate else 0.0
    share = len(pitches) / frames if frames else 0.0
    if not pitches:
        return VoicePrint(sample_rate, 0.0, 0.0, 0.0, share, 0.0, seconds)

    values = np.asarray(pitches)
    # Median first, from everything: it is an order statistic and a tail of
    # octave errors barely moves it.  Then drop the low tail and take the
    # percentiles from what is left -- see FLOOR_GATE_ST for why only the low
    # one, which is not the obvious choice.
    median = float(np.median(values))
    kept = values[values >= median / semitones_to_ratio(FLOOR_GATE_ST)]
    if kept.size:
        median = float(np.median(kept))
    else:                                       # nothing survived; say so
        kept = values
    return VoicePrint(
        sample_rate,
        median,
        float(np.percentile(kept, 10)),
        float(np.percentile(kept, 90)),
        share, voiced_seconds, seconds,
        octave_errors=int(values.size - kept.size),
    )


@dataclass
class Suggestion:
    """A profile fitted to one speaker, and what fitting it costs."""

    profile: VoiceProfile
    voice: VoicePrint
    target_hz: float
    #: Where ``profile`` actually lands this speaker's median pitch.
    lands_hz: float
    #: Where they would land if the shift were held inside the range where
    #: PSOLA stays transparent.  Equal to ``lands_hz`` when it already is.
    transparent_hz: float
    notes: list[str]

    @property
    def is_transparent(self) -> bool:
        return abs(self.profile.pitch_semitones) <= NATURAL_PITCH_LIMIT

    def summary(self) -> str:
        lines = [self.voice.summary()]
        if not self.voice.usable:
            return "\n".join(lines)
        lines.append(
            f"to reach {self.target_hz:.0f} Hz   "
            f"pitch {self.profile.pitch_semitones:+.1f} st, "
            f"formant {self.profile.formant_semitones:+.1f} st "
            f"-> lands at {self.lands_hz:.0f} Hz")
        lines.extend(f"  note: {note}" for note in self.notes)
        return "\n".join(lines)


def suggest(voice: VoicePrint, target_hz: float = FEMALE_TARGET_HZ,
            base: VoiceProfile | None = None) -> Suggestion:
    """Fit ``base`` to this speaker: pitch to the target, tract by physiology.

    Only the two speaker-dependent numbers are computed -- how far this voice
    has to move, and how low it goes.  Everything else in ``base`` (breathiness,
    tilt, intonation, whether consonants shift) is a choice about what kind of
    voice to make and is left alone, because measuring a speaker says nothing
    about it.
    """
    from .. import presets

    base = base if base is not None else presets.get("female")
    if not voice.usable:
        return Suggestion(base, voice, target_hz, 0.0, 0.0,
                          ["say a couple of sentences in your ordinary voice, "
                           "then measure again"])

    wanted = voice.shift_to(target_hz)
    # The direction comes from the voice being made, not from the arithmetic.
    # A speaker who already sits at the target needs no pitch shift and still
    # wants the tract of whoever they are trying to sound like, and the sign of
    # a near-zero number cannot say which that is -- the base preset can.
    toward = base.formant_semitones if base.formant_semitones else wanted
    tract = TRACT_SEMITONES if toward >= 0.0 else -TRACT_SEMITONES
    notes = []

    # The floor the speaker actually uses, not the one the preset guessed.
    # Below their own range by a margin, because a floor set too high causes
    # octave errors, which sound far worse than the latency it would save.
    floor = max(40.0, voice.low_hz / semitones_to_ratio(FLOOR_MARGIN_ST))
    floor = min(floor, voice.median_hz * 0.95)

    profile = base.replace(
        pitch_semitones=round(wanted, 1),
        formant_semitones=round(tract, 1),
        f0_min=round(floor, 1),
    )
    lands = voice.median_hz * semitones_to_ratio(profile.pitch_semitones)

    transparent_st = float(np.clip(wanted, -NATURAL_PITCH_LIMIT, NATURAL_PITCH_LIMIT))
    transparent_hz = voice.median_hz * semitones_to_ratio(transparent_st)

    if abs(wanted) > NATURAL_PITCH_LIMIT:
        notes.append(
            f"{abs(wanted):.1f} st is past the +-{NATURAL_PITCH_LIMIT:.0f} st this "
            f"package warns at. Measured here, the degradation is gradual rather "
            f"than a cliff -- spectral envelope error on connected speech goes "
            f"0.65 dB at +4.5 st to 1.13 dB at +13 -- so it is worth hearing "
            f"before believing. Holding it to {transparent_st:+.1f} st would land "
            f"you at {transparent_hz:.0f} Hz instead of {target_hz:.0f}. Try both.")
    if voice.range_semitones < 4.0:
        notes.append(
            f"your pitch range is {voice.range_semitones:.1f} st, which is narrow; "
            "a uniform shift keeps it exactly, so the 'Pitch range' slider is "
            "doing more work here than usual")
    if profile.f0_min > base.f0_min + 1.0:
        notes.append(
            f"tracking floor raised to {profile.f0_min:.0f} Hz from your own range, "
            "which also cuts latency")
    notes.extend(profile.warnings())
    return Suggestion(profile, voice, target_hz, lands, transparent_hz, notes)
