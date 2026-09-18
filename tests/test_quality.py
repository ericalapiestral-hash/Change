"""Artifact thresholds.

These are the tests that encode the actual requirement -- that the output does
not sound processed.  Each threshold below corresponds to a specific way a
voice changer gives itself away, and each was set from measurements on this
engine with headroom, so a regression trips it rather than slipping out in a
release.
"""
from __future__ import annotations

import time

import numpy as np
import pytest
from scipy import signal as sg

import natvox
from natvox import VoiceChanger, presets

from evaluate import (  # noqa: E402  (tools/ is on the path via conftest)
    formant_error_db, harmonic_split_db, hnr_db, pitch_track,
)

SHIFTING_PRESETS = ["brighter", "deeper", "younger", "male_to_female_subtle",
                    "male_to_female", "female_to_male", "anonymous"]


def convert(audio, sample_rate, profile):
    return natvox.process_array(audio, sample_rate, profile, block_size=512)


class TestPitch:
    @pytest.mark.parametrize("name", SHIFTING_PRESETS)
    def test_pitch_lands_where_it_was_asked_to(self, sustained_vowel, sample_rate, name):
        """Better than 20 cents; beyond that a shift reads as out of tune."""
        dry, f0 = sustained_vowel
        profile = presets.get(name)
        wet = convert(dry, sample_rate, profile)
        _, tracked = pitch_track(wet, sample_rate, f0_min=50.0, f0_max=800.0)
        voiced = tracked[tracked > 0]
        assert voiced.size > 50
        achieved = np.median(voiced) / f0
        cents = abs(1200 * np.log2(achieved / profile.pitch_ratio))
        assert cents < 20.0, f"{name}: off by {cents:.1f} cents"

    def test_intonation_is_scaled_not_flattened(self, utterance, sample_rate):
        """A monotone output is the most obvious robot tell of all."""
        audio, truth = utterance
        profile = presets.get("male_to_female")
        wet = convert(audio, sample_rate, profile)
        positions, dry_f0 = pitch_track(audio, sample_rate)
        _, wet_f0 = pitch_track(wet, sample_rate, f0_min=50.0, f0_max=800.0)
        n = min(dry_f0.size, wet_f0.size)
        both = (dry_f0[:n] > 0) & (wet_f0[:n] > 0)
        # Compare the 5th-95th percentile spread rather than peak-to-peak: a
        # single mis-tracked frame in either measurement would otherwise decide
        # the result, and it is the shape of the contour that matters here.
        def spread(values):
            lo, hi = np.percentile(np.log2(values), [5, 95])
            return hi - lo
        dry_range = spread(dry_f0[:n][both])
        wet_range = spread(wet_f0[:n][both])
        assert wet_range > 0.8 * dry_range, f"{wet_range:.3f} vs {dry_range:.3f}"


class TestFormants:
    @pytest.mark.parametrize("name", ["brighter", "deeper", "younger",
                                      "male_to_female_subtle", "female_to_male"])
    def test_spectral_envelope_lands_where_intended(self, utterance, sample_rate, name):
        """Within 2 dB RMS of the input envelope warped by the formant ratio."""
        audio, truth = utterance
        profile = presets.get(name)
        wet = convert(audio, sample_rate, profile)
        error = formant_error_db(audio, wet, sample_rate, profile.formant_ratio,
                                 mask=truth["voiced"])
        assert error < 2.0, f"{name}: envelope off by {error:.2f} dB"

    def test_pitch_shift_alone_leaves_formants_alone(self, utterance, sample_rate):
        """The whole point of PSOLA: pitch moves, timbre does not."""
        audio, truth = utterance
        profile = natvox.VoiceProfile(pitch_semitones=5.0, formant_semitones=0.0)
        wet = convert(audio, sample_rate, profile)
        error = formant_error_db(audio, wet, sample_rate, 1.0, mask=truth["voiced"])
        assert error < 2.0, f"formants moved by {error:.2f} dB"


class TestArtifacts:
    @pytest.mark.parametrize("name", SHIFTING_PRESETS)
    def test_inharmonic_energy_stays_buried(self, sustained_vowel, sample_rate, name):
        """Energy that is not at a harmonic of the output pitch.

        This is the number that tracks "sounds robotic" most directly: grain
        jitter, comb sidebands, subharmonics and resampling aliasing all land
        here.  A natural voice's own jitter sits around -25 dB, so staying
        below -35 dB means the engine adds less than the speaker does.
        """
        dry, f0 = sustained_vowel
        profile = presets.get(name)
        wet = convert(dry, sample_rate, profile)
        guard = slice(int(0.05 * sample_rate), -int(0.05 * sample_rate))
        level = harmonic_split_db(wet[guard], sample_rate, f0 * profile.pitch_ratio)
        assert level < -35.0, f"{name}: inharmonic energy at {level:.1f} dB"

    @pytest.mark.parametrize("name", SHIFTING_PRESETS)
    def test_harmonics_to_noise_stays_healthy(self, sustained_vowel, sample_rate, name):
        """25 dB is already better than a typical human voice (15-25 dB)."""
        dry, f0 = sustained_vowel
        profile = presets.get(name)
        wet = convert(dry, sample_rate, profile)
        guard = slice(int(0.05 * sample_rate), -int(0.05 * sample_rate))
        value = hnr_db(wet[guard], sample_rate, f0 * profile.pitch_ratio)
        assert value > 25.0, f"{name}: HNR down to {value:.1f} dB"

    @pytest.mark.parametrize("name", ["male_to_female", "female_to_male", "anonymous"])
    def test_no_buzz_at_the_grain_rate(self, fricative_noise, sample_rate, name):
        """Grains laid at a fixed rate stamp that rate onto fricatives.

        Before the spacing was randomised and the power normalisation fixed,
        this measured +17 dB -- an audible buzz on every /s/.
        """
        wet = convert(fricative_noise, sample_rate, presets.get(name))
        envelope = np.abs(sg.hilbert(wet[8000:-8000]))
        freqs, power = sg.welch(envelope - envelope.mean(), sample_rate, nperseg=16384)
        band = (freqs > 20) & (freqs < 800)
        excess = 10 * np.log10(power[band].max() / np.median(power[band]))
        assert excess < 8.0, f"{name}: {excess:.1f} dB modulation peak"

    def test_consonants_are_passed_through_untouched(self, fricative_noise, sample_rate):
        """With shift_unvoiced off, unvoiced audio must be bit-exact.

        This is the strongest guarantee available for the sounds listeners are
        most sensitive to -- consonants cannot acquire an artifact they were
        never processed for.
        """
        profile = presets.get("male_to_female_subtle").replace(highpass_hz=0.0)
        assert not profile.shift_unvoiced
        wet = convert(fricative_noise, sample_rate, profile)
        guard = slice(6000, -6000)
        error = (np.mean((wet[guard] - fricative_noise[guard]) ** 2)
                 / np.mean(fricative_noise[guard] ** 2))
        assert 10 * np.log10(error + 1e-30) < -50

    def test_shifting_consonants_moves_them_in_the_right_direction(
            self, fricative_noise, sample_rate):
        def centroid(x):
            spec = np.abs(np.fft.rfft(x[8000:-8000])) ** 2
            freqs = np.fft.rfftfreq(x[8000:-8000].size, 1.0 / sample_rate)
            return float(np.sum(freqs * spec) / np.sum(spec))

        base = centroid(fricative_noise)
        up = presets.get("male_to_female")       # formants up
        down = presets.get("female_to_male")     # formants down
        assert centroid(convert(fricative_noise, sample_rate, up)) > base * 1.02
        assert centroid(convert(fricative_noise, sample_rate, down)) < base * 0.98


class TestPerformance:
    @pytest.mark.parametrize("name", ["male_to_female", "female_to_male"])
    def test_runs_comfortably_faster_than_real_time(self, utterance, sample_rate, name):
        """Real-time needs headroom, not merely keeping up."""
        audio, _ = utterance
        changer = VoiceChanger(sample_rate, presets.get(name))
        blocks = [audio[i:i + 256] for i in range(0, audio.size, 256)]
        started = time.perf_counter()
        for block in blocks:
            changer.process(block)
        elapsed = time.perf_counter() - started
        assert elapsed / (audio.size / sample_rate) < 0.35

    @pytest.mark.parametrize("name", presets.PRESETS)
    def test_latency_is_low_enough_for_conversation(self, sample_rate, name):
        """Engine delay budget.

        Most of this is not arbitrary: PSOLA needs about two periods of the
        lowest pitch it must track, pitch tracking needs a couple more before
        it can call a frame voiced, and onset look-ahead buys back the
        syllable-initial pitch errors that come from that. 70 ms is the point
        past which talking over your own voice stops feeling natural, and
        every preset stays inside it; `f0_min` and `onset_lookahead_ms` are
        the two knobs that move it.
        """
        assert VoiceChanger(sample_rate, presets.get(name)).latency_ms < 70.0

    def test_onset_lookahead_trades_latency_for_syllable_starts(self, sample_rate):
        profile = presets.get("male_to_female")
        fast = VoiceChanger(sample_rate, profile.replace(onset_lookahead_ms=0.0))
        careful = VoiceChanger(sample_rate, profile.replace(onset_lookahead_ms=8.0))
        assert careful.latency_ms > fast.latency_ms

    def test_lower_f0_min_is_the_price_of_tracking_deep_voices(self, sample_rate):
        deep = VoiceChanger(sample_rate, natvox.VoiceProfile(f0_min=65.0))
        shallow = VoiceChanger(sample_rate, natvox.VoiceProfile(f0_min=140.0))
        assert deep.latency_ms > shallow.latency_ms


class TestBoundaries:
    """The handovers between the voiced and unvoiced paths.

    Everything above is measured on steady audio.  An audit found that engine
    variants with visibly different onset and creak behaviour scored
    identically on all of it, because the places the engine misbehaves are
    exactly the places those metrics erode or average away.  These are the
    numbers that were bad and are now good; they exist so they cannot quietly
    go back.
    """

    @pytest.mark.parametrize("name", SHIFTING_PRESETS)
    def test_syllables_start_on_the_voiced_path(self, sample_rate, name):
        """Pitch tracking cannot call a frame voiced until it has seen a
        couple of periods, so without look-ahead the first 20-30 ms of every
        syllable leaves unshifted, at the speaker's own pitch - heard as a
        scoop into every syllable.  Measured at 22-31 ms before the fix.
        """
        from evaluate import onset_lag_ms
        from synth_speech import onset_train

        audio, truth = onset_train(sample_rate)
        mean, worst = onset_lag_ms(audio, sample_rate, presets.get(name), truth["onsets"])
        assert mean < 22.0, f"{name}: mean onset lag {mean:.1f} ms"
        assert worst < 25.0, f"{name}: worst onset lag {worst:.1f} ms"

    @pytest.mark.parametrize("name", SHIFTING_PRESETS)
    def test_creaky_phrase_ends_stay_converted(self, sample_rate, name):
        """Almost every sentence ends in creak.  Its periods are irregular
        enough to defeat a periodicity test, and audio that falls to the
        unvoiced path comes out at the speaker's real pitch - so the voice
        reverts exactly where a listener is most likely to notice.  Measured
        at 46-82% of the creak before the fix, flipping 11-14 times.
        """
        from evaluate import creak_voicing
        from synth_speech import creak_fall

        audio, truth = creak_fall(sample_rate)
        share, flips = creak_voicing(audio, sample_rate, presets.get(name), truth["creak"])
        assert share < 0.15, f"{name}: {share:.0%} of the creak left unconverted"
        assert flips <= 3, f"{name}: path flipped {flips} times"

    @pytest.mark.parametrize("name", ["brighter", "younger", "male_to_female",
                                      "deeper", "anonymous"])
    def test_the_voice_does_not_come_back_more_perfect_than_it_went_in(
            self, sample_rate, name):
        """Over-regularity is the oldest robotic tell there is.

        A real voice varies by a few tenths of a percent in period from one
        glottal pulse to the next.  Placing grains on a smoothed pitch estimate
        throws that away: before the speaker's own micro-timing was carried
        through, jitter came back at 0.58-0.75x of the input and the
        harmonics-to-noise ratio was pushed *above* it.  No steady-vowel metric
        can see this, because the vowel those are measured on has no jitter to
        lose.
        """
        from evaluate import jitter_shimmer
        from synth_speech import human_vowel

        dry = human_vowel(sample_rate)
        region = (int(0.08 * sample_rate), dry.size - int(0.08 * sample_rate))
        profile = presets.get(name)
        wet = convert(dry, sample_rate, profile)

        dry_jitter, _ = jitter_shimmer(dry, sample_rate, 120.0, region)
        wet_jitter, _ = jitter_shimmer(wet, sample_rate, 120.0 * profile.pitch_ratio, region)
        ratio = wet_jitter / dry_jitter
        assert 0.75 < ratio < 1.6, f"{name}: jitter came back at {ratio:.2f}x the input"

    def test_micro_timing_invents_nothing_on_a_perfectly_regular_input(
            self, sustained_vowel, sample_rate):
        """Carrying the speaker's irregularity through must not manufacture
        any: a jitter-free input has to stay jitter-free, or the engine would
        be adding the very roughness it exists to avoid."""
        dry, f0 = sustained_vowel
        profile = presets.get("male_to_female_subtle")
        wet = convert(dry, sample_rate, profile)
        guard = slice(int(0.05 * sample_rate), -int(0.05 * sample_rate))
        level = harmonic_split_db(wet[guard], sample_rate, f0 * profile.pitch_ratio)
        assert level < -50.0, f"inharmonic energy at {level:.1f} dB"

    @pytest.mark.parametrize("f0", [250, 255, 260, 300])
    def test_pitch_above_half_the_ceiling_is_not_read_an_octave_high(
            self, sample_rate, f0):
        """YIN's first-dip rule can take a shallow dip at half the true period
        when the pitch is above f0_max/2.  With female_to_male's own 500 Hz
        ceiling that band is 250-265 Hz - ordinary female speech, i.e. exactly
        what the preset is for - and a vowel tracked an octave out is not
        detuned but destroyed: HNR fell to 2 dB before this was guarded.
        """
        from synth_speech import VOWELS, glottal_source, vocal_tract

        n = int(sample_rate * 0.8)
        source = glottal_source(np.full(n, float(f0)), sample_rate,
                                np.random.default_rng(3), jitter=0.0, shimmer=0.0)
        y = vocal_tract(source, np.stack([np.full(n, f) for f in VOWELS["e"]]), sample_rate)
        y /= max(np.max(np.abs(y)), 1e-9)
        fade = int(0.02 * sample_rate)
        y[:fade] *= np.linspace(0, 1, fade)
        y[-fade:] *= np.linspace(1, 0, fade)
        dry = 0.5 * y

        profile = presets.get("female_to_male")
        wet = convert(dry, sample_rate, profile)
        guard = slice(int(0.06 * sample_rate), -int(0.06 * sample_rate))
        target = f0 * profile.pitch_ratio
        assert hnr_db(wet[guard], sample_rate, target) > 25.0
