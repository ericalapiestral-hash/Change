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

PRESETS: dict[str, VoiceProfile] = {
    "off": VoiceProfile(),
    "male_to_female": VoiceProfile(
        pitch_semitones=7.0, formant_semitones=2.6, f0_min=70.0, f0_max=400.0,
        shift_unvoiced=True, breathiness=0.03,
    ),
    "female_to_male": VoiceProfile(
        pitch_semitones=-7.0, formant_semitones=-2.6, f0_min=110.0, f0_max=500.0,
        shift_unvoiced=True,
    ),
    "male_to_female_subtle": VoiceProfile(
        pitch_semitones=4.5, formant_semitones=1.8, f0_min=70.0, f0_max=400.0,
    ),
    "female_to_male_subtle": VoiceProfile(
        pitch_semitones=-4.5, formant_semitones=-1.8, f0_min=110.0, f0_max=500.0,
    ),
    # Same speaker, different apparent age/size -- these stay well inside the
    # transparent range and are the ones that hold up best under scrutiny.
    "deeper": VoiceProfile(pitch_semitones=-2.5, formant_semitones=-1.2, f0_min=65.0),
    "brighter": VoiceProfile(pitch_semitones=1.5, formant_semitones=1.0),
    "younger": VoiceProfile(pitch_semitones=3.0, formant_semitones=2.2, f0_max=550.0),
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
