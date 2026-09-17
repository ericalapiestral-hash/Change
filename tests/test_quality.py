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
        dry_range = np.ptp(np.log2(dry_f0[:n][both]))
        wet_range = np.ptp(np.log2(wet_f0[:n][both]))
        assert wet_range > 0.8 * dry_range


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
        """Above ~50 ms of engine delay, talking over it starts to feel wrong."""
        assert VoiceChanger(sample_rate, presets.get(name)).latency_ms < 50.0

    def test_lower_f0_min_is_the_price_of_tracking_deep_voices(self, sample_rate):
        deep = VoiceChanger(sample_rate, natvox.VoiceProfile(f0_min=65.0))
        shallow = VoiceChanger(sample_rate, natvox.VoiceProfile(f0_min=140.0))
        assert deep.latency_ms > shallow.latency_ms
