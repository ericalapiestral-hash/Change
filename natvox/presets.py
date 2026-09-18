"""Ready-made voice profiles.

The numbers come from the physiology rather than from taste.  Adult male F0
averages ~120 Hz against ~210 Hz for female (a ratio of 1.75, or about 9.6
semitones), while vocal tract length differs by only ~1.17 (about 2.7
semitones).  Shifting pitch by the full 9.6 st is past where PSOLA stays
transparent, so the gender presets take roughly three quarters of it: clearly
the other gender, still clearly a human being.
"""
from __future__ import annotations

from .config import VoiceProfile

#: Every preset tracks to 800 Hz.  The ceiling costs nothing -- not latency,
#: not measurable time, not accuracy at low pitch -- and a voice raised to
#: shout goes past a conversational ceiling, where the tracker locks onto twice
#: the period and reports an octave down.  See VoiceProfile.f0_max.
PRESETS: dict[str, VoiceProfile] = {
    "off": VoiceProfile(),
    # 0.02 rather than the 0.03 this carried before aspiration was keyed off
    # the signal's energy in the aspiration band instead of its broadband
    # level: the new keying is louder for the same setting, and this restores
    # the level actually measured here (-45.9 dB of added energy on voiced
    # audio against -44.8 dB before).
    "male_to_female": VoiceProfile(
        pitch_semitones=7.0, formant_semitones=2.6, f0_min=70.0, f0_max=800.0,
        shift_unvoiced=True, breathiness=0.02,
    ),
    "female_to_male": VoiceProfile(
        pitch_semitones=-7.0, formant_semitones=-2.6, f0_min=110.0, f0_max=800.0,
        shift_unvoiced=True,
    ),
    "male_to_female_subtle": VoiceProfile(
        pitch_semitones=4.5, formant_semitones=1.8, f0_min=70.0, f0_max=800.0,
    ),
    # The presets above move pitch and vocal-tract size and nothing else,
    # which is the transparent thing to do and also the reason they still
    # sound like a man an octave up: pitch and tract length are two of the
    # cues, and the ear uses more than two.  The three below add the rest of
    # what the literature actually measures between male and female speech.
    #
    #   intonation  F0 standard deviation in read speech is ~2.0-2.8 st for
    #               men against ~2.4-3.4 st for women, a ratio near 1.2.  A
    #               uniform shift preserves the speaker's range exactly, so
    #               without this the output keeps a man's intonation.
    #   tilt_db     Long-term average spectra differ by a few dB of slope
    #               beyond what tract scaling explains -- a higher glottal
    #               open quotient means less energy low down and more air up
    #               top.  Formant shifting moves the filter; this is the
    #               source.
    #   breathiness Female phonation is measurably breathier (lower HNR,
    #               larger H1-H2).  Gated to voiced audio only, so it is
    #               aspiration rather than hiss.  The three settings below put
    #               a sustained vowel at 27.7, 24.2 and 21.7 dB HNR, which is
    #               the range real modal-to-breathy female phonation measures
    #               in; the same engine without it returns 47 dB, cleaner than
    #               any human being.
    #
    # Every one of them is a cue rather than a transformation: none of them
    # changes *whose* voice it is, and no amount of them will.  That needs a
    # conversion model -- see natvox.neural.
    "female": VoiceProfile(
        pitch_semitones=7.0, formant_semitones=2.6, f0_min=70.0, f0_max=800.0,
        shift_unvoiced=True, breathiness=0.12, intonation=1.22, tilt_db=2.0,
    ),
    "female_soft": VoiceProfile(
        pitch_semitones=4.5, formant_semitones=1.8, f0_min=70.0, f0_max=800.0,
        breathiness=0.08, intonation=1.15, tilt_db=1.2,
    ),
    "female_bright": VoiceProfile(
        pitch_semitones=7.5, formant_semitones=3.2, f0_min=70.0, f0_max=800.0,
        shift_unvoiced=True, breathiness=0.16, intonation=1.28, tilt_db=3.5,
    ),
    "female_to_male_subtle": VoiceProfile(
        pitch_semitones=-4.5, formant_semitones=-1.8, f0_min=110.0, f0_max=800.0,
    ),
    # Same speaker, different apparent age/size -- these stay well inside the
    # transparent range and are the ones that hold up best under scrutiny.
    "deeper": VoiceProfile(pitch_semitones=-2.5, formant_semitones=-1.2, f0_min=65.0),
    "brighter": VoiceProfile(pitch_semitones=1.5, formant_semitones=1.0),
    "younger": VoiceProfile(pitch_semitones=3.0, formant_semitones=2.2, f0_max=800.0),
    # Disguise: enough change to break recognition, no cartoon quality.
    "anonymous": VoiceProfile(
        pitch_semitones=-3.5, formant_semitones=2.0, f0_min=70.0, shift_unvoiced=True,
    ),
}


def get(name: str) -> VoiceProfile:
    try:
        return PRESETS[name]
    except KeyError:
        raise KeyError(
            f"unknown preset {name!r}; available: {', '.join(sorted(PRESETS))}"
        ) from None


def names() -> list[str]:
    return sorted(PRESETS)
