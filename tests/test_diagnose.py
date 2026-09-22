"""Measuring a recording that nobody knows the right answer for.

Every other test in this suite compares against ground truth: the synthetic
utterance knows its own pitch contour and formant tracks, so a metric can say
by how much the engine missed.  That is what makes those tests sharp, and it
is also why none of them could be pointed at the recording somebody made of
their own voice and called robotic.

These cover the measurements that need no answer in advance.  They are
calibrated against the same synthetic utterance -- not because it is a
person, but because it is the one recording in this repository that is clean
by construction, so anything the diagnostic complains about on it is the
diagnostic being wrong.
"""
from __future__ import annotations

import numpy as np
import pytest

from natvox.app import diagnose


def report(**over):
    """A Report with the clean reference's numbers, and its complaints."""
    base = dict(sample_rate=48000, seconds=2.4, peak_db=-6.0, speech_db=-17.8,
                quiet_db=-73.8, clipped_samples=0, dc_offset=0.0,
                band_levels=(0.0, -3.0, -14.0, -20.0, -11.0),
                voiced_share=0.71, median_hz=110.0,
                octave_jumps_per_s=0.0, octave_jumps=0,
                voicing_flips_per_s=3.4,
                median_step_st=0.09, p95_step_st=0.26)
    base.update(over)
    r = diagnose.Report(**base)
    r.complaints = diagnose._complaints(r)          # noqa: SLF001
    return r


class TestTheCleanReference:
    """Whatever this says about known-good audio is what it says about noise.

    A diagnostic that cries wolf on the clean case is one nobody finishes
    reading, and two earlier attempts at a band-limit verdict did exactly
    that.  So the first thing it has to do is stay quiet here.
    """

    def test_nothing_looks_wrong(self, utterance, sample_rate):
        audio, _ = utterance
        assert diagnose.look(audio, sample_rate).complaints == []

    def test_it_finds_the_voice(self, utterance, sample_rate):
        audio, truth = utterance
        r = diagnose.look(audio, sample_rate)
        heard = truth["f0"][truth["voiced"]]
        assert r.voiced_share > 0.5
        assert abs(r.median_hz - np.median(heard)) < 6.0

    def test_the_tracker_never_jumps_an_octave(self, utterance, sample_rate):
        audio, _ = utterance
        r = diagnose.look(audio, sample_rate)
        assert r.octave_jumps_per_s == diagnose.CLEAN_OCTAVE_JUMPS_PER_S == 0.0

    def test_adjacent_frames_barely_move(self, utterance, sample_rate):
        audio, _ = utterance
        r = diagnose.look(audio, sample_rate)
        assert r.median_step_st == pytest.approx(diagnose.CLEAN_MEDIAN_STEP_ST,
                                                 abs=0.05)

    def test_voicing_changes_at_the_rate_syllables_do(self, utterance, sample_rate):
        audio, _ = utterance
        r = diagnose.look(audio, sample_rate)
        assert r.voicing_flips_per_s == pytest.approx(diagnose.CLEAN_FLIPS_PER_S,
                                                      abs=0.5)
        assert r.voicing_flips_per_s < diagnose.FLIPS_COMPLAINT_PER_S

    def test_the_speech_stands_well_clear_of_the_quiet(self, utterance, sample_rate):
        audio, _ = utterance
        r = diagnose.look(audio, sample_rate)
        assert r.headroom_db > diagnose.QUIET_HEADROOM_DB + 10

    @pytest.mark.parametrize("snr_db", [40, 20, 10])
    def test_a_noisy_room_does_not_explain_a_misbehaving_tracker(
            self, utterance, sample_rate, snr_db):
        """The claim the module's docstring makes, held to.

        If noise moved these numbers, "your room is loud" would be an
        available excuse for every reading they ever produce, and the
        diagnostic would be useless on exactly the recordings it is for.
        """
        audio, _ = utterance
        rng = np.random.default_rng(3)
        level = np.sqrt(np.mean(audio * audio)) / 10 ** (snr_db / 20)
        r = diagnose.look(audio + rng.normal(0, level, audio.size), sample_rate)
        assert r.octave_jumps_per_s == 0.0
        assert r.median_step_st < diagnose.CLEAN_MEDIAN_STEP_ST + 0.05
        assert r.voiced_share > 0.6


class TestWhatItComplainsAbout:
    def test_clipping(self, utterance, sample_rate):
        audio, _ = utterance
        r = diagnose.look(np.clip(audio * 6, -1.0, 1.0), sample_rate)
        assert r.clipped_samples > 1000
        assert any("at the rail" in note for note in r.complaints)

    def test_a_loud_room(self, utterance, sample_rate):
        audio, _ = utterance
        rng = np.random.default_rng(11)
        level = np.sqrt(np.mean(audio * audio)) / 10 ** (6 / 20)
        r = diagnose.look(audio + rng.normal(0, level, audio.size), sample_rate)
        assert r.headroom_db < diagnose.QUIET_HEADROOM_DB
        assert any("quiet between the words" in note for note in r.complaints)

    def test_a_dc_offset(self, utterance, sample_rate):
        audio, _ = utterance
        r = diagnose.look(audio * 0.5 + 0.05, sample_rate)
        assert r.dc_offset == pytest.approx(0.05, abs=0.002)
        assert any("DC offset" in note for note in r.complaints)

    def test_a_dc_offset_is_not_reported_when_clipping_explains_it(self):
        """Clipping one side of a waveform moves its mean, which is not a fault.

        Two complaints for one cause sends somebody looking for a second
        problem that is not there.
        """
        r = report(dc_offset=0.2, clipped_samples=9000)
        assert any("at the rail" in note for note in r.complaints)
        assert not any("DC offset" in note for note in r.complaints)

    def test_silence(self, sample_rate):
        r = diagnose.look(np.zeros(sample_rate), sample_rate)
        assert r.voiced_share == 0.0
        assert any("reads as voiced" in note for note in r.complaints)

    def test_octave_jumps(self):
        r = report(octave_jumps_per_s=2.0, octave_jumps=20)
        note = next(n for n in r.complaints if "octave" in n)
        assert "robotic" in note

    def test_voicing_flapping(self):
        r = report(voicing_flips_per_s=20.0)
        assert any("flapping" in note for note in r.complaints)

    def test_a_threshold_is_a_threshold(self):
        """Just under each limit says nothing; just over says one thing."""
        assert report(octave_jumps_per_s=diagnose.JUMPS_COMPLAINT_PER_S - 0.01,
                      octave_jumps=20).complaints == []
        assert len(report(octave_jumps_per_s=diagnose.JUMPS_COMPLAINT_PER_S + 0.01,
                          octave_jumps=20).complaints) == 1


class TestWhereTheEnergyIs:
    """A single number here was wrong, and nearly carried a verdict.

    "The frequency below which 99.5% of the energy sits" rests on the last
    half percent of it, which is the part a filter's skirt, a noise floor and
    a resampler each disagree about: one utterance band-limited to 3.4 kHz
    read 1734 Hz kept at 48 kHz and 3734 Hz resampled to 8 kHz.  Band levels
    are sums over wide ranges, so nothing hinges on a tail.
    """

    @staticmethod
    def low_passed(audio, cut, sample_rate):
        from scipy import signal

        sos = signal.butter(8, cut / (sample_rate / 2), btype="low", output="sos")
        return signal.sosfilt(sos, audio)

    def test_the_reference_reads_what_the_table_says(self, utterance, sample_rate):
        audio, _ = utterance
        levels = diagnose.look(audio, sample_rate).band_levels
        assert levels == pytest.approx((0, -3, -14, -20, -11), abs=1.5)

    def test_one_signal_reads_the_same_at_two_sample_rates(self, utterance,
                                                           sample_rate):
        """The failure that retired the old number, held to directly."""
        from scipy import signal

        audio, _ = utterance
        wide = diagnose.look(audio, sample_rate)
        resampled = diagnose.look(signal.resample_poly(audio, 1, 3),
                                  sample_rate // 3)
        # The top band is gone at 16 kHz; everything the rate still carries
        # has to agree, which is what the old number could not manage.
        assert resampled.band_levels[:4] == pytest.approx(wide.band_levels[:4],
                                                          abs=2.0)
        assert resampled.cue_band_db == pytest.approx(wide.cue_band_db, abs=2.0)

    def test_a_telephone_band_still_carries_a_voice(self, utterance, sample_rate):
        audio, _ = utterance
        r = diagnose.look(self.low_passed(audio, 3400, sample_rate), sample_rate)
        assert r.cue_band_db > diagnose.CUE_BAND_FLOOR_DB
        assert r.complaints == [], "a telephone line is narrow, not broken"

    def test_narrower_than_a_telephone_is_complained_about(self, utterance,
                                                           sample_rate):
        """What a real microphone turned out to be doing."""
        audio, _ = utterance
        r = diagnose.look(self.low_passed(audio, 2000, sample_rate), sample_rate)
        assert r.cue_band_db < diagnose.CUE_BAND_FLOOR_DB
        note = next(n for n in r.complaints if "cue" in n or "band" in n)
        assert "sound female" in note
        assert "before this program" in note, "it is not the engine's fault"

    def test_the_cue_band_is_where_the_female_cue_lives(self):
        assert diagnose.CUE_BAND in diagnose.BANDS
        assert diagnose.CUE_BAND == (2000, 4000), "F2, F3 and the fricatives"

    def test_the_summary_prints_every_band(self, utterance, sample_rate):
        text = diagnose.look(audio_of(utterance), sample_rate).summary()
        for band in diagnose.BANDS:
            assert diagnose._band_name(band) in text       # noqa: SLF001


def audio_of(utterance):
    return utterance[0]


class TestItSurvivesWhateverItIsPointedAt:
    """It is pointed at files from other people's machines, by definition."""

    def test_nothing_at_all(self):
        r = diagnose.look(np.zeros(0), 48000)
        assert r.seconds == 0.0
        assert r.median_hz == 0.0
        assert r.summary()

    def test_shorter_than_the_tracker_can_look_at(self, sample_rate):
        r = diagnose.look(np.zeros(64), sample_rate)
        assert r.voiced_share == 0.0
        assert r.octave_jumps_per_s == 0.0

    def test_a_sample_rate_of_zero(self):
        assert diagnose.look(np.zeros(1000), 0).seconds == 0.0

    def test_two_channels_are_averaged_not_interleaved(self, utterance,
                                                       sample_rate):
        """``reshape(-1)`` on a stereo file gives a signal at twice the pitch.

        Which is the exact defect this whole module exists to find, arriving
        by the back door.
        """
        audio, _ = utterance
        stereo = np.stack([audio, audio], axis=1)
        assert (diagnose.look(stereo, sample_rate).median_hz
                == pytest.approx(diagnose.look(audio, sample_rate).median_hz))

    def test_a_file_full_of_nonsense(self, sample_rate):
        audio = np.full(sample_rate, np.nan)
        audio[::2] = np.inf
        r = diagnose.look(audio, sample_rate)
        assert np.isfinite(r.peak_db) and np.isfinite(r.median_step_st)

    def test_a_telephone_rate_still_tracks_the_voice(self, utterance,
                                                     sample_rate):
        from scipy import signal

        audio, _ = utterance
        r = diagnose.look(signal.resample_poly(audio, 1, 6), 8000)
        assert 90 < r.median_hz < 140

    def test_a_rate_with_no_room_for_the_tracking_ceiling(self):
        """The tracker refuses ``f0_max >= nyquist``, and would raise.

        No real file arrives at 1 kHz, but this is pointed at other people's
        files and a diagnostic that raises on one of them tells them nothing.
        """
        assert diagnose.F0_MAX > 1000 / 2             # the case this is about
        r = diagnose.look(np.zeros(4000), 1000)
        assert r.seconds == pytest.approx(4.0)
        assert r.voiced_share == 0.0

    def test_the_hop_is_adjustable_without_changing_the_answer(self, utterance,
                                                              sample_rate):
        audio, _ = utterance
        fine = diagnose.look(audio, sample_rate, hop_ms=5.0)
        coarse = diagnose.look(audio, sample_rate, hop_ms=20.0)
        assert abs(fine.median_hz - coarse.median_hz) < 3.0
        assert fine.octave_jumps_per_s == coarse.octave_jumps_per_s == 0.0


class TestComparingTwoRecordings:
    """Which of the two made it robotic is the only question worth asking."""

    def test_a_recording_against_itself_blames_nobody(self):
        r = report()
        text = diagnose.compare(r, r)
        assert "not adding instability" in text
        assert "doing wrong" not in text

    def test_it_reports_the_shift_the_engine_actually_delivered(self):
        text = diagnose.compare(report(), report(median_hz=220.0))
        assert "+12.0 st" in text

    def test_a_shift_that_never_happened_is_visible(self):
        """Asking for +6.6 and getting +3.3 is the tracker halving the pitch."""
        text = diagnose.compare(report(), report(median_hz=133.0))
        assert "+3.3 st" in text

    def test_octave_jumps_the_engine_invented(self):
        text = diagnose.compare(report(),
                                report(octave_jumps_per_s=2.0, octave_jumps=20))
        assert "the engine is doing wrong" in text
        assert "robotic sound itself" in text

    def test_pitch_wobble_the_engine_invented(self):
        text = diagnose.compare(report(), report(median_step_st=0.6))
        assert "less steady than the voice was" in text

    def test_voicing_flapping_the_engine_invented(self):
        text = diagnose.compare(report(), report(voicing_flips_per_s=12.0))
        assert "mid-word" in text

    def test_resynthesis_alone_is_not_an_accusation(self):
        """The engine rebuilds the pitch track; the two never measure alike."""
        after = report(median_hz=186.0, median_step_st=0.08,
                       voiced_share=0.68,
                       band_levels=(0.0, -3.0, -13.0, -18.0, -10.0))
        assert "not adding instability" in diagnose.compare(report(), after)

    def test_a_bad_recording_passed_through_blames_the_recording(self):
        before = report(clipped_samples=5000)
        assert "passing the recording's own" in diagnose.compare(before, before)

    def test_it_does_not_divide_by_a_pitch_nobody_found(self):
        text = diagnose.compare(report(median_hz=0.0), report(median_hz=0.0))
        assert "st)" not in text


class TestTheSummaryIsReadable:
    def test_a_clean_recording_points_downstream(self, utterance, sample_rate):
        audio, _ = utterance
        text = diagnose.look(audio, sample_rate).summary()
        assert "downstream of the" in text
        assert "what looks wrong" not in text

    def test_the_engines_own_output_is_not_a_microphone(self, utterance,
                                                        sample_rate):
        audio, _ = utterance
        text = diagnose.look(audio, sample_rate).summary(is_recording=False)
        assert "downstream of the" not in text
        assert "on its own" in text

    def test_complaints_are_listed_under_a_heading(self):
        text = report(octave_jumps_per_s=3.0, octave_jumps=30).summary()
        assert "what looks wrong" in text
        assert "* the pitch tracker jumps" in text

    def test_it_prints_what_clean_looks_like_beside_what_this_is(self):
        """A number with nothing to compare it against is not a diagnosis."""
        text = report().summary()
        assert f"clean: {diagnose.CLEAN_FLIPS_PER_S:.1f}" in text
        assert f"clean: {diagnose.CLEAN_MEDIAN_STEP_ST:.2f}" in text

    def test_clipping_is_counted_in_the_level_line(self):
        assert "9000 samples at the rail" in report(clipped_samples=9000).summary()


class TestTheBlindSpotARealRecordingFound:
    """0.12 -> 0.47 octave jumps a second, and this reported no fault.

    Four times as many, one every two seconds, audible -- and the bare
    difference, 0.35, sat under the margin allowed for resynthesis.  A margin
    calibrated on a base of zero says nothing about a base that is not zero.
    """

    def test_the_real_numbers_are_now_a_complaint(self):
        text = diagnose.compare(
            report(octave_jumps_per_s=0.12, octave_jumps=3),
            report(octave_jumps_per_s=0.47, octave_jumps=12, median_hz=174.0))
        assert "the engine is doing wrong" in text
        assert "0.35 octave jumps" in text

    def test_the_bare_difference_alone_would_have_missed_it(self):
        assert 0.47 - 0.12 < diagnose.ADDED_JUMPS_PER_S

    def test_a_ratio_on_a_base_of_nearly_nothing_is_not_enough(self):
        """0.02 -> 0.06 is three times as many and still inaudible."""
        text = diagnose.compare(
            report(octave_jumps_per_s=0.02, octave_jumps=2),
            report(octave_jumps_per_s=0.06, octave_jumps=9))
        assert "doing wrong" not in text

    def test_the_engine_passing_them_through_is_not_the_engine_inventing_them(self):
        text = diagnose.compare(
            report(octave_jumps_per_s=0.40, octave_jumps=10),
            report(octave_jumps_per_s=0.45, octave_jumps=11))
        assert "doing wrong" not in text

    def test_the_engine_removing_them_is_never_a_complaint(self):
        """What the same microphone did on its better day: 0.10 -> 0.00."""
        text = diagnose.compare(
            report(octave_jumps_per_s=0.10, octave_jumps=1),
            report(octave_jumps_per_s=0.0, octave_jumps=0))
        assert "doing wrong" not in text


class TestLandingShortOfFemale:
    """The thing most likely to be the whole answer, and not a defect.

    A preset is a fixed interval rather than a destination: +7 semitones took
    a real 104 Hz speaker to 155 Hz, and a voice is heard as male or
    ambiguous below about 165.  Every artifact metric was clean, the
    conversion did exactly what it was asked, and the result still sounded
    like a man.
    """

    def test_the_real_recording_is_told_it_landed_short(self):
        text = diagnose.compare(report(median_hz=104.0),
                                report(median_hz=155.0))
        assert "155 Hz" in text
        assert "Fit it to my voice" in text

    def test_it_is_not_filed_as_the_engine_doing_something_wrong(self):
        text = diagnose.compare(report(median_hz=104.0),
                                report(median_hz=155.0))
        assert "worth knowing" in text
        assert "the engine is doing wrong" not in text

    def test_clearing_the_crossover_says_nothing(self):
        text = diagnose.compare(report(median_hz=104.0),
                                report(median_hz=186.0))
        assert "heard as male" not in text

    def test_it_stays_quiet_when_nobody_asked_to_go_up(self):
        """Nothing is landing short if the point was to go down."""
        text = diagnose.compare(report(median_hz=200.0),
                                report(median_hz=130.0))
        assert "heard as male" not in text

    def test_a_dull_recording_is_named_as_the_other_half(self):
        after = report(median_hz=186.0,
                       band_levels=(0.0, -4.0, -12.0, -29.0, -30.0))
        text = diagnose.compare(report(median_hz=104.0), after)
        assert "F2, F3 and the consonants" in text
        assert "closer to the microphone" in text
        assert "no setting here can" in text
