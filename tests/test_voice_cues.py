"""The three controls that decide what *kind* of voice comes out.

Pitch and formant shifting change a voice's register and apparent size.  These
change the cues on top of that -- range, source spectrum, aspiration -- which
is what separates "the same speaker transposed" from "a different speaker".
Each one is verified against the same engine with that control neutralised,
because all three run alongside a shift that moves the same measurements.
"""
from __future__ import annotations

import numpy as np
import pytest

from evaluate import added_energy_db, pitch_range_st, spectral_tilt_db

import natvox
from natvox import VoiceChanger, VoiceProfile, presets
from natvox.dsp.util import OverlapAccumulator, TiltFilter, hann

FEMALE = presets.get("female")


def process(audio, sample_rate, profile, block=512):
    changer = VoiceChanger(sample_rate, profile)
    chunks = [changer.process(audio[i:i + block])
              for i in range(0, audio.size, block)]
    chunks.append(changer.flush())
    return np.concatenate(chunks)[changer.latency_samples:][:audio.size]


class TestIntonation:
    def test_a_setting_of_one_changes_nothing(self, utterance, sample_rate):
        """The default has to be free, or nobody can trust the other settings."""
        audio, _ = utterance
        plain = VoiceProfile(pitch_semitones=5.0, formant_semitones=2.0)
        assert np.array_equal(
            process(audio, sample_rate, plain),
            process(audio, sample_rate, plain.replace(intonation=1.0)),
        )

    def test_the_range_widens_and_keeps_widening(self, utterance, sample_rate):
        audio, _ = utterance
        flat = pitch_range_st(process(audio, sample_rate, FEMALE.replace(intonation=1.0)),
                              sample_rate)
        widths = [
            pitch_range_st(process(audio, sample_rate, FEMALE.replace(intonation=v)),
                           sample_rate) / flat
            for v in (1.0, 1.15, 1.30, 1.50)
        ]
        assert widths[0] == pytest.approx(1.0, abs=1e-9)
        assert all(b > a for a, b in zip(widths, widths[1:])), widths
        # The delivered range is a fraction of the setting, because only the
        # part of the contour faster than INTONATION_TC counts as deviation.
        # That fraction is what it is; what must hold is that it is real and
        # bounded well short of a transposition.
        assert 1.05 < widths[2] < 1.35, widths

    def test_flattening_is_available_too(self, utterance, sample_rate):
        audio, _ = utterance
        flat = pitch_range_st(process(audio, sample_rate, FEMALE.replace(intonation=1.0)),
                              sample_rate)
        squashed = pitch_range_st(
            process(audio, sample_rate, FEMALE.replace(intonation=0.7)), sample_rate)
        assert squashed < flat

    def test_the_average_pitch_still_lands_where_it_was_asked_to(
            self, utterance, sample_rate):
        """Expansion works on deviations, so the centre must not move."""
        from evaluate import pitch_error_cents

        audio, _ = utterance
        wet = process(audio, sample_rate, FEMALE)
        # Deviations are deliberately introduced, so the *median* deviation is
        # the intonation itself; what would signal a bug is the centre drifting,
        # which would show up as a median far larger than the range expansion.
        assert pitch_error_cents(audio, wet, sample_rate, FEMALE.pitch_ratio) < 80.0

    def test_the_latency_budget_covers_the_whole_ratio_range(self, sample_rate):
        """Expansion lowers the ratio as often as it raises it, and the lower
        end asks for the longest grains.  Budgeting from the nominal ratio
        instead would leave those grains with zero-filled tails."""
        nominal = VoiceChanger(sample_rate, FEMALE.replace(intonation=1.0))
        expanded = VoiceChanger(sample_rate, FEMALE)
        assert expanded.latency_samples >= nominal.latency_samples
        assert expanded._ratio_lo < expanded._pitch_ratio < expanded._ratio_hi

    def test_a_steady_pitch_has_nothing_to_expand(self, sample_rate):
        """Expansion works on deviation from the speaker's own average, so a
        tone that never deviates must come out at exactly the nominal shift.

        Measured on the pitch rather than the waveform on purpose: any change
        to the ratio, however brief, moves the synthesis cursor permanently, so
        two runs stay a fraction of a period out of step with each other for
        good.  That is a phase offset, not a pitch error, and comparing samples
        would report it as one.
        """
        from evaluate import pitch_error_cents

        n = 2 * sample_rate
        t = np.arange(n) / sample_rate
        steady = 0.4 * np.sin(2 * np.pi * 130.0 * t)
        profile = VoiceProfile(pitch_semitones=7.0, f0_min=70.0)
        flat = process(steady, sample_rate, profile)
        wide = process(steady, sample_rate, profile.replace(intonation=2.0))
        assert pitch_error_cents(steady, wide, sample_rate, profile.pitch_ratio) < 15.0
        # The floor here (~0.20 st) is the engine's own micro-timing, and it is
        # there with the feature switched off.  What matters is that doubling
        # the setting does not amplify it: tracker noise expanded into audible
        # wobble is the obvious way for this control to go wrong.
        assert pitch_range_st(wide, sample_rate) < 1.01 * pitch_range_st(flat, sample_rate)


class TestTilt:
    def test_zero_is_bit_exact(self, utterance, sample_rate):
        audio, _ = utterance
        plain = VoiceProfile(pitch_semitones=5.0)
        assert np.array_equal(
            process(audio, sample_rate, plain),
            process(audio, sample_rate, plain.replace(tilt_db=0.0)),
        )

    @pytest.mark.parametrize("gain", [-6.0, -2.0, 2.0, 6.0])
    def test_the_filter_delivers_the_slope_it_advertises(self, sample_rate, gain):
        noise = np.random.default_rng(1).standard_normal(sample_rate * 2)
        tilted = TiltFilter(sample_rate, gain)(noise.copy())
        delivered = spectral_tilt_db(tilted, sample_rate) - spectral_tilt_db(noise, sample_rate)
        # Measured between band centres rather than at the asymptotes, so the
        # figure is ~0.83 of the setting; the sign and scale are the contract.
        assert delivered == pytest.approx(gain * 0.83, abs=0.25)

    def test_the_pivot_is_where_the_asymptotes_cross(self, sample_rate):
        f = TiltFilter(sample_rate, 8.0, 1000.0)
        n = 1 << 15
        impulse = np.zeros(n)
        impulse[0] = 1.0
        response = 20 * np.log10(np.abs(np.fft.rfft(f(impulse))) + 1e-18)
        freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
        at = lambda hz: response[int(np.argmin(np.abs(freqs - hz)))]
        assert at(20.0) == pytest.approx(-4.0, abs=0.2)
        assert at(sample_rate * 0.4) == pytest.approx(+4.0, abs=0.3)
        assert at(1000.0) > at(500.0) and at(1000.0) < at(2000.0)

    def test_it_survives_the_loudness_matcher(self, utterance, sample_rate):
        """Tilt runs before loudness matching so the match removes only the
        level the slope implies, not the slope."""
        audio, _ = utterance
        level = process(audio, sample_rate, FEMALE.replace(tilt_db=0.0))
        bright = process(audio, sample_rate, FEMALE.replace(tilt_db=4.0))
        delivered = (spectral_tilt_db(bright, sample_rate)
                     - spectral_tilt_db(level, sample_rate))
        assert delivered > 1.5, delivered


class TestAspiration:
    @pytest.mark.parametrize("name", ["female", "female_soft", "female_bright"])
    def test_it_lands_on_voiced_audio_and_nowhere_else(self, utterance, sample_rate, name):
        audio, truth = utterance
        profile = presets.get(name)
        dry_air = process(audio, sample_rate, profile.replace(breathiness=0.0))
        breathy = process(audio, sample_rate, profile)
        voiced = truth["voiced"]
        # 50 ms, not 30: the ground truth stops calling a vowel voiced while it
        # is still sounding, and the engine goes on treating that decaying tail
        # as voiced -- correctly, since it still has a pitch.  A 30 ms margin
        # includes one such tail and reports the breath on it as breath on a
        # consonant, at -41 dB instead of the real -97.
        margin = int(0.05 * sample_rate)
        away = np.convolve(voiced.astype(float), np.ones(2 * margin + 1),
                           mode="same") == 0
        loud = np.convolve(np.abs(audio), np.ones(512) / 512, mode="same") > 5e-3
        on_voiced = added_energy_db(dry_air, breathy, voiced)
        on_unvoiced = added_energy_db(dry_air, breathy, away & loud)
        assert on_voiced > -45.0
        assert on_unvoiced < on_voiced - 55.0, (on_voiced, on_unvoiced)

    def test_it_leaves_untouched_consonants_untouched(self, fricative_noise, sample_rate):
        """Consonant passthrough is bit-exact; breath must not spoil that."""
        profile = FEMALE.replace(shift_unvoiced=False, breathiness=0.3)
        quiet = process(fricative_noise, sample_rate, profile.replace(breathiness=0.0))
        breathy = process(fricative_noise, sample_rate, profile)
        assert added_energy_db(quiet, breathy) < -60.0

    def test_it_follows_the_voice_rather_than_sitting_on_top_of_it(self, sample_rate):
        """Real aspiration is filtered by the same tract as the voice, so it is
        loud on a bright vowel and quiet on a dark one.  Keying it off the
        band's own energy is the cheap stand-in for that; keying it off
        broadband level instead would put the same hiss over both."""
        from scipy import signal

        n = sample_rate
        t = np.arange(n) / sample_rate
        pulse = signal.sawtooth(2 * np.pi * 120.0 * t) * 0.4
        dark = signal.sosfilt(
            signal.butter(4, 700 / (sample_rate / 2), output="sos"), pulse)
        bright = signal.sosfilt(
            signal.butter(2, [700 / (sample_rate / 2), 6000 / (sample_rate / 2)],
                          btype="band", output="sos"), pulse)
        dark *= 0.4 / max(np.max(np.abs(dark)), 1e-9)
        bright *= 0.4 / max(np.max(np.abs(bright)), 1e-9)

        def added(x):
            return added_energy_db(process(x, sample_rate, FEMALE.replace(breathiness=0.0)),
                                   process(x, sample_rate, FEMALE))

        assert added(bright) > added(dark) + 6.0, (added(bright), added(dark))

    def test_silence_stays_silent(self, sample_rate):
        changer = VoiceChanger(sample_rate, FEMALE.replace(breathiness=1.0))
        out = np.concatenate([changer.process(np.zeros(512)) for _ in range(40)])
        assert np.max(np.abs(out)) == 0.0


class TestVoicedShare:
    def test_it_reports_the_fraction_of_coverage_that_was_voiced(self):
        acc = OverlapAccumulator(1024)
        window = hann(64)
        acc.add(0, window.copy(), window, True, True)
        acc.add(32, window.copy(), window, True, False)
        share = acc.voiced_share(0, 128)
        assert share.min() >= 0.0 and share.max() <= 1.0
        assert share[16] == pytest.approx(1.0)   # only the voiced grain reaches
        assert share[80] == pytest.approx(0.0)   # only the unvoiced one does
        assert 0.0 < share[48] < 1.0             # both

    def test_uncovered_samples_read_as_unvoiced(self):
        acc = OverlapAccumulator(1024)
        assert np.all(acc.voiced_share(0, 64) == 0.0)


class TestContract:
    @pytest.mark.parametrize("block", [64, 128, 333, 512, 1000, 4096])
    @pytest.mark.parametrize("name", ["female", "female_bright"])
    def test_the_cues_do_not_break_block_size_independence(
            self, utterance, sample_rate, name, block):
        """Every cue updates on a schedule fixed by the audio, not by the
        caller: intonation once per glottal pulse, tilt and aspiration once per
        sample.  Anything updated per block would land here."""
        audio, _ = utterance
        profile = presets.get(name)
        reference = process(audio, sample_rate, profile, 1024)
        assert np.max(np.abs(process(audio, sample_rate, profile, block)
                             - reference)) < 1e-4

    @pytest.mark.parametrize("kwargs", [
        {"intonation": 0.4}, {"intonation": 2.5}, {"tilt_db": 14.0},
        {"tilt_db": -13.0},
    ])
    def test_impossible_settings_are_refused(self, kwargs):
        with pytest.raises(ValueError):
            VoiceProfile(**kwargs)

    @pytest.mark.parametrize("kwargs", [{"intonation": 1.2}, {"tilt_db": 2.0}])
    def test_a_cue_on_its_own_is_not_an_identity(self, kwargs):
        assert not VoiceProfile(**kwargs).is_identity
        assert VoiceProfile().is_identity

    def test_output_stays_bounded_with_everything_turned_up(self, sample_rate):
        profile = VoiceProfile(pitch_semitones=8.0, formant_semitones=4.0,
                               intonation=2.0, tilt_db=12.0, breathiness=1.0,
                               shift_unvoiced=True, f0_min=70.0)
        rng = np.random.default_rng(7)
        changer = VoiceChanger(sample_rate, profile)
        out = np.concatenate([changer.process(rng.normal(0, 0.3, 512))
                              for _ in range(60)])
        assert np.all(np.isfinite(out)) and np.max(np.abs(out)) < 1.0
