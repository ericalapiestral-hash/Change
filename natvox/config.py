"""User-facing settings for the DSP voice changer."""
from __future__ import annotations

from dataclasses import dataclass, replace

# Beyond roughly this much shift, *any* pitch/formant method starts to sound
# processed, because the vocal-tract response being stretched no longer
# matches a physically plausible speaker.  We warn rather than forbid.
NATURAL_PITCH_LIMIT = 8.0
NATURAL_FORMANT_LIMIT = 5.0

#: How far the dynamic pitch ratio may stray from the profile's nominal shift
#: when :attr:`VoiceProfile.intonation` is not 1.0.  Intonation expansion works
#: on the *deviation* from a slowly-moving average, so this bounds an excursion,
#: not the shift itself -- and it bounds the engine's latency budget, which has
#: to cover the longest grain any ratio in the range can ask for.
INTONATION_LIMIT_ST = 4.0

#: Frequency at which the tilt filter is unity gain.  Roughly where the first
#: formant of a female voice sits, so tilting about it trades low-frequency
#: body against high-frequency air without moving the vowel's centre.
TILT_PIVOT_HZ = 1000.0


def semitones_to_ratio(semitones: float) -> float:
    return float(2.0 ** (semitones / 12.0))


def ratio_to_semitones(ratio: float) -> float:
    import math
    return 12.0 * math.log2(ratio)


@dataclass(frozen=True)
class VoiceProfile:
    """How the voice should be changed.

    Pitch and formants are independent.  Pitch alone makes a chipmunk or a
    giant; formants alone change apparent body size at the same pitch.  A
    believable speaker change moves both, but by different amounts -- vocal
    folds and vocal tract do not scale together.

    Attributes
    ----------
    pitch_semitones:
        Perceived pitch shift.  Intonation is preserved because the whole F0
        contour is scaled, not flattened onto a target.
    formant_semitones:
        Vocal-tract size shift.  Positive sounds smaller/brighter.
    f0_min, f0_max:
        Search range for pitch tracking.  ``f0_min`` sets the latency floor --
        the engine needs about two periods of the lowest pitch it must handle,
        so raising it is the main way to get faster response.
    shift_unvoiced:
        Whether to also shift fricatives and other unvoiced sounds.  Off by
        default: unvoiced audio is then passed through bit-exact, which is the
        safest possible answer for those sounds.
    breathiness:
        0..1 mix of a shaped noise component, useful to cover the slight
        thinning of a large upward pitch shift.
    intonation:
        Scale applied to the speaker's *pitch range*, as opposed to
        :attr:`pitch_semitones`, which moves the whole contour.  1.0 keeps the
        speaker's own range.  Above 1.0 the deviations from a slowly-moving
        average are stretched, below 1.0 they are flattened.

        This exists because a uniform pitch shift preserves the semitone range
        exactly, and range is itself a gender cue: read speech from adult men
        spans roughly 2-2.5 semitones of standard deviation against 2.5-3.5 for
        women, so a man shifted up an octave still speaks with a man's
        intonation.  Expansion is bounded by
        :data:`INTONATION_LIMIT_ST` and is applied per glottal pulse, so it
        follows the contour rather than re-drawing it.
    tilt_db:
        Spectral tilt from low to high frequency, in dB, pivoting at
        :data:`TILT_PIVOT_HZ`.  Positive is brighter.  Formant shifting scales
        the vocal tract but leaves the *source* spectrum alone; a female
        glottal source has a higher open quotient, which reads as less energy
        low down and more air up top than the same tract would produce on a
        male source.  This is the control for that, and it is deliberately a
        gentle first-order shelf rather than an EQ curve: it colours, it does
        not sculpt.
    output_gain_db:
        Applied after loudness matching, before the limiter.
    """

    pitch_semitones: float = 0.0
    formant_semitones: float = 0.0
    f0_min: float = 75.0
    f0_max: float = 500.0
    shift_unvoiced: bool = False
    breathiness: float = 0.0
    intonation: float = 1.0
    tilt_db: float = 0.0
    output_gain_db: float = 0.0
    #: Rumble filter cutoff.  Zero disables it; values between zero and
    #: :data:`natvox.dsp.util.BiquadHighpass.MIN_CUTOFF_HZ` also disable it,
    #: because a biquad that close to DC is numerically unstable.
    #: How far ahead of a mark pitch tracking may be consulted when deciding
    #: whether that mark is voiced.  Pitch tracking cannot call a frame voiced
    #: until it has seen a couple of periods, so without this the first
    #: 20-30 ms of every syllable leaves on the unvoiced path -- unshifted, at
    #: the speaker's own pitch, which is heard as a scoop into every syllable.
    #: Reading a little way ahead recovers most of it, and costs exactly that
    #: much latency.  Zero disables it.
    onset_lookahead_ms: float = 8.0

    highpass_hz: float = 60.0

    def __post_init__(self) -> None:
        if not 0.0 < self.f0_min < self.f0_max:
            raise ValueError("require 0 < f0_min < f0_max")
        if not 0.0 <= self.breathiness <= 1.0:
            raise ValueError("breathiness must be in [0, 1]")
        if not 0.5 <= self.intonation <= 2.0:
            raise ValueError("intonation must be in [0.5, 2.0]")
        if abs(self.tilt_db) > 12.0:
            raise ValueError("tilt_db must be within +-12 dB")
        if abs(self.pitch_semitones) > 24.0 or abs(self.formant_semitones) > 24.0:
            raise ValueError("shifts beyond +-24 semitones are not supported")
        if not 0.0 <= self.onset_lookahead_ms <= 30.0:
            raise ValueError("onset_lookahead_ms must be in [0, 30]")
        if not 0.0 <= self.highpass_hz <= 500.0:
            raise ValueError("highpass_hz must be in [0, 500]; 0 disables it")

    @property
    def pitch_ratio(self) -> float:
        return semitones_to_ratio(self.pitch_semitones)

    @property
    def formant_ratio(self) -> float:
        return semitones_to_ratio(self.formant_semitones)

    @property
    def is_identity(self) -> bool:
        return (
            abs(self.pitch_semitones) < 1e-6
            and abs(self.formant_semitones) < 1e-6
            and self.breathiness <= 0.0
            and abs(self.intonation - 1.0) < 1e-9
            and abs(self.tilt_db) < 1e-9
        )

    def warnings(self) -> list[str]:
        """Settings that are legal but will cost naturalness."""
        notes = []
        if abs(self.pitch_semitones) > NATURAL_PITCH_LIMIT:
            notes.append(
                f"pitch shift of {self.pitch_semitones:+.1f} st exceeds the "
                f"+-{NATURAL_PITCH_LIMIT:.0f} st range where PSOLA stays transparent; "
                "expect some loss of naturalness"
            )
        if abs(self.formant_semitones) > NATURAL_FORMANT_LIMIT:
            notes.append(
                f"formant shift of {self.formant_semitones:+.1f} st exceeds "
                f"+-{NATURAL_FORMANT_LIMIT:.0f} st; the voice may sound cartoonish"
            )
        if self.pitch_semitones > 4.0 and self.formant_semitones <= 0.0:
            notes.append(
                "raising pitch without raising formants sounds like a sped-up "
                "recording; try formant_semitones around 40% of the pitch shift"
            )
        if self.intonation > 1.0 and self.pitch_semitones <= 0.0:
            notes.append(
                "intonation expansion is a cue for a *higher* voice; on a "
                "downward or neutral shift it mostly reads as unsteadiness"
            )
        if self.pitch_semitones < -4.0 and self.formant_semitones >= 0.0:
            notes.append(
                "lowering pitch without lowering formants sounds hollow; try a "
                "negative formant shift around 40% of the pitch shift"
            )
        return notes

    def replace(self, **kwargs) -> "VoiceProfile":
        return replace(self, **kwargs)
