"""Fitting the shift to the speaker instead of guessing at them.

Every preset is a statement about a voice nobody has heard: +7 semitones lands
a 100 Hz speaker at 150 Hz and a 140 Hz speaker at 210 Hz, and only one of
those is a woman's pitch.  These check that measuring changes the answer, and
that the two decisions which could have gone the other way -- median rather
than mean, and a fixed tract shift rather than a fraction of the pitch shift --
are the ones actually implemented.
"""
from __future__ import annotations

import numpy as np
import pytest

from natvox import presets
from natvox.app import voiceprint
from natvox.config import NATURAL_PITCH_LIMIT

SR = 48000


@pytest.fixture(scope="module")
def speech():
    """A few vowels at a given pitch, which is as close to speech as a test
    can get and still know the right answer."""
    from bench import sustained

    def make(hz, seconds=0.5, vowels=("a", "i", "u", "e")):
        return np.concatenate([sustained(hz, v, seconds, SR) for v in vowels])

    return make


class TestMeasuring:
    @pytest.mark.parametrize("hz", [85.0, 110.0, 145.0, 200.0, 260.0])
    def test_it_finds_the_pitch_that_is_there(self, speech, hz):
        voice = voiceprint.measure(speech(hz), SR)
        assert voice.usable
        assert voice.median_hz == pytest.approx(hz, rel=0.02)

    def test_a_creak_tail_does_not_drag_it_down(self, speech):
        """The decision that could have gone the other way.

        Connected speech falls at the end of every phrase and many speakers
        drop into creak there, an octave or more below their habitual pitch.
        A mean is dragged down by that tail and would ask for too large a
        shift; the median ignores it, which is the whole reason to use one.
        """
        body = speech(120.0, seconds=0.5)          # 2.0 s of ordinary voice
        creak = speech(60.0, seconds=0.15)         # 0.6 s of fry at the end
        voice = voiceprint.measure(np.concatenate([body, creak]), SR)
        assert voice.median_hz == pytest.approx(120.0, rel=0.03)

        # What a mean would have given, so the test is about the choice and
        # not about a tolerance that happens to pass.
        assert voice.median_hz > 115.0, "a mean here lands near 106 Hz"

    def test_a_halved_period_does_not_become_the_speaker_s_floor(self, speech):
        """The first real measurement: a 126 Hz speaker read "63-157 Hz, a
        range of 15.8 semitones", and 63 is exactly half of 126.

        The median survived it -- an order statistic ignores a tail -- but the
        tenth percentile is the tail, and the converted profile's f0_min comes
        from there.  Taken at face value it asked for a 53 Hz tracking floor:
        worse latency than the lowest row in the README's table, and a floor
        low enough to invite the very error that produced it.
        """
        voice = voiceprint.measure(
            np.concatenate([speech(126.0), speech(63.0, seconds=0.08)]), SR)
        assert voice.median_hz == pytest.approx(126.0, rel=0.02)
        assert voice.low_hz > 100.0, "63 Hz is not this speaker's floor"
        assert voice.octave_errors > 0
        assert "low tail" in voice.summary()

        floor = voiceprint.suggest(voice).profile.f0_min
        assert floor > voice.median_hz / 2, "never below a halved period"
        assert floor > 90.0

    def test_a_genuinely_wide_voice_is_not_clipped(self, speech):
        """The reason the gate is low-side only.

        A symmetric gate seems obviously right and is not: this speaker's
        median sits in whichever mode has more frames, and a gate centred
        there throws the other mode away.  Measured with one: 199 frames of
        396 kept, and the reported range collapsed from twelve semitones to
        zero -- half their voice, discarded because they used it.
        """
        voice = voiceprint.measure(
            np.concatenate([speech(100.0), speech(200.0)]), SR)
        assert voice.octave_errors == 0
        assert voice.range_semitones == pytest.approx(12.0, abs=1.5)
        assert voice.low_hz == pytest.approx(100.0, rel=0.05)
        assert voice.high_hz == pytest.approx(200.0, rel=0.05)

    def test_it_reports_the_range_the_speaker_uses(self, speech):
        low, high = speech(110.0, seconds=0.5), speech(220.0, seconds=0.5)
        voice = voiceprint.measure(np.concatenate([low, high]), SR)
        assert voice.range_semitones == pytest.approx(12.0, abs=1.5)

    @pytest.mark.parametrize("audio", ["silence", "noise", "too short"])
    def test_it_refuses_rather_than_guessing(self, audio, speech):
        rng = np.random.default_rng(0)
        signal = {
            "silence": np.zeros(3 * SR),
            "noise": rng.standard_normal(3 * SR) * 0.1,
            "too short": speech(120.0, seconds=0.05, vowels=("a",)),
        }[audio]
        voice = voiceprint.measure(signal, SR)
        assert not voice.usable
        assert "ordinary voice" in voice.summary()

    def test_the_shift_arithmetic(self, speech):
        voice = voiceprint.measure(speech(100.0), SR)
        assert voice.shift_to(200.0) == pytest.approx(12.0, abs=0.4)
        assert voice.shift_to(100.0) == pytest.approx(0.0, abs=0.4)
        assert voice.shift_to(50.0) == pytest.approx(-12.0, abs=0.4)


class TestSuggesting:
    def test_it_lands_the_speaker_on_the_target(self, speech):
        for hz in (95.0, 120.0, 150.0, 185.0):
            s = voiceprint.suggest(voiceprint.measure(speech(hz), SR))
            assert s.lands_hz == pytest.approx(voiceprint.FEMALE_TARGET_HZ, rel=0.02)

    def test_the_tract_shift_is_the_same_for_every_speaker(self, speech):
        """The other decision that could have gone the other way.

        Vocal folds and vocal tract do not scale together: male and female
        tract lengths differ by about 1.17 whatever the speaker's pitch, so a
        woman with a low voice still has a woman's tract.  Scaling formants
        with the pitch gives the right answer for an average male speaker and
        the wrong one for everybody else, in the direction that makes a low
        voice sound like a child.
        """
        shifts = {}
        for hz in (95.0, 120.0, 150.0, 185.0):
            s = voiceprint.suggest(voiceprint.measure(speech(hz), SR))
            shifts[hz] = (s.profile.pitch_semitones, s.profile.formant_semitones)
        pitches = {p for p, _ in shifts.values()}
        formants = {f for _, f in shifts.values()}
        assert len(pitches) == 4, "the pitch shift must depend on the speaker"
        assert formants == {voiceprint.TRACT_SEMITONES}, \
            f"the tract shift must not: {shifts}"

    def test_a_low_voice_is_told_it_cannot_get_there_transparently(self, speech):
        s = voiceprint.suggest(voiceprint.measure(speech(100.0), SR))
        assert not s.is_transparent
        note = " ".join(s.notes)
        assert "past" in note
        assert f"{s.transparent_hz:.0f} Hz" in note, \
            "it has to say where staying inside the limit would land them"
        assert s.transparent_hz < voiceprint.FEMALE_TARGET_HZ

    def test_a_voice_already_near_the_target_is_barely_moved(self, speech):
        s = voiceprint.suggest(voiceprint.measure(speech(190.0), SR))
        assert s.is_transparent
        assert abs(s.profile.pitch_semitones) < 1.5

    def test_going_down_flips_the_tract_shift_too(self, speech):
        s = voiceprint.suggest(voiceprint.measure(speech(210.0), SR),
                               target_hz=voiceprint.MALE_TARGET_HZ,
                               base=presets.get("female_to_male"))
        assert s.profile.pitch_semitones < 0
        assert s.profile.formant_semitones == -voiceprint.TRACT_SEMITONES

    def test_a_speaker_already_at_the_target_still_gets_the_tract(self, speech):
        """They need no pitch shift and still want the tract of whoever they
        are trying to sound like.  The sign of a near-zero number cannot say
        which that is, so it comes from the preset."""
        s = voiceprint.suggest(
            voiceprint.measure(speech(voiceprint.FEMALE_TARGET_HZ), SR))
        assert abs(s.profile.pitch_semitones) < 0.5
        assert s.profile.formant_semitones == voiceprint.TRACT_SEMITONES

    def test_the_floor_comes_from_the_speaker_not_the_preset(self, speech):
        """f0_min sets the latency floor, so somebody who never goes below
        150 Hz should not pay for tracking to 75."""
        low = voiceprint.suggest(voiceprint.measure(speech(95.0), SR))
        high = voiceprint.suggest(voiceprint.measure(speech(185.0), SR))
        assert high.profile.f0_min > low.profile.f0_min
        assert low.profile.f0_min < 95.0, "never above the speaker's own pitch"
        assert high.profile.f0_min < 185.0

    def test_measuring_a_speaker_says_nothing_about_what_voice_to_make(self, speech):
        """Only the two speaker-dependent numbers are computed.  Breathiness,
        tilt, intonation and whether consonants shift are choices about the
        result, not facts about the person."""
        base = presets.get("female_bright")
        s = voiceprint.suggest(voiceprint.measure(speech(120.0), SR), base=base)
        assert s.profile.breathiness == base.breathiness
        assert s.profile.tilt_db == base.tilt_db
        assert s.profile.intonation == base.intonation
        assert s.profile.shift_unvoiced == base.shift_unvoiced

    def test_an_unusable_measurement_changes_nothing(self):
        base = presets.get("female")
        voice = voiceprint.measure(np.zeros(SR), SR)
        s = voiceprint.suggest(voice, base=base)
        assert s.profile == base
        assert "ordinary voice" in s.summary()

    def test_the_summary_names_the_numbers_it_used(self, speech):
        s = voiceprint.suggest(voiceprint.measure(speech(120.0), SR))
        text = s.summary()
        assert "120 Hz" in text
        assert f"{voiceprint.FEMALE_TARGET_HZ:.0f} Hz" in text
        assert "pitch +" in text and "formant +" in text


class TestAgainstThePresets:
    """What measuring is worth, stated as the error it removes."""

    @pytest.mark.parametrize("hz,preset_lands", [
        (95.0, 142.0), (110.0, 165.0), (125.0, 187.0), (145.0, 217.0)])
    def test_the_shipped_preset_over_or_undershoots_by_speaker(
            self, speech, hz, preset_lands):
        from natvox.config import semitones_to_ratio

        female = presets.get("female")
        landed = hz * semitones_to_ratio(female.pitch_semitones)
        assert landed == pytest.approx(preset_lands, rel=0.02)

        fitted = voiceprint.suggest(voiceprint.measure(speech(hz), SR))
        assert abs(fitted.lands_hz - voiceprint.FEMALE_TARGET_HZ) \
            < abs(landed - voiceprint.FEMALE_TARGET_HZ), \
            "fitting has to beat the preset, or it is not worth the button"
