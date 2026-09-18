"""What happens when the voice gets loud, and what happens when it does not.

Shouting into a voice changer is where the cheap parts of the design show.
Three of them failed here, and only one was visible to any metric that
existed:

* the output stage was a static soft clipper, which distorts by construction
  and whose products land on exact multiples of F0 -- so inharmonic energy
  *improved* as the distortion got worse;
* the pitch ceiling was set for conversation, and a true F0 above it is not
  read as "too high" but as an octave down, which is the growl;
* the loudness matcher was suspected of pumping at the onset, and measurement
  said it does not, which is also worth keeping a test for.
"""
from __future__ import annotations

import numpy as np
import pytest

from evaluate import (jitter_shimmer, manufactured_band_db, out_of_band_db,
                      pitch_track)

import natvox
from natvox import VoiceChanger, VoiceProfile, presets
from natvox.dsp.util import PeakLimiter
from natvox.engine import LIMITER_CEILING, SAFETY_CEILING

BAND_EDGE = 5000.0


def band_limited(sample_rate, f0=120.0, seconds=1.0):
    """A vowel with nothing above BAND_EDGE, so anything there afterwards is ours."""
    from bench import band_limited_vowel

    return band_limited_vowel(f0, seconds, sample_rate)


def drive(audio, sample_rate, profile, level, block=512):
    trim = slice(int(0.1 * sample_rate), -int(0.1 * sample_rate))
    return natvox.process_array(audio * level, sample_rate, profile, block)[trim]


def thd_db(y, sample_rate, f0):
    spectrum = np.abs(np.fft.rfft(y * np.hanning(y.size))) ** 2
    freqs = np.fft.rfftfreq(y.size, 1.0 / sample_rate)
    fundamental = spectrum[(freqs > f0 * 0.9) & (freqs < f0 * 1.1)].sum()
    harmonics = sum(spectrum[(freqs > f0 * k * 0.98) & (freqs < f0 * k * 1.02)].sum()
                    for k in range(2, 12))
    return 10.0 * np.log10(max(harmonics, 1e-30) / max(fundamental, 1e-30))


class TestLimiter:
    @pytest.mark.parametrize("level", [0.5, 1.0, 1.5, 4.0, 40.0])
    def test_the_ceiling_is_a_guarantee_not_a_target(self, sample_rate, level):
        """Two smoothing stages sized so that the gain applied to a sample is
        never above the gain that sample needed.  It either holds for every
        drive or the proof is wrong."""
        limiter = PeakLimiter(sample_rate)
        rng = np.random.default_rng(3)
        x = level * rng.standard_normal(sample_rate)
        out = np.concatenate([limiter(x[i:i + 512]) for i in range(0, x.size, 512)])
        assert np.max(np.abs(out)) <= limiter.ceiling + 1e-12

    # (drive, worst acceptable THD).  The static clipper this replaced read
    # -31 dB at full scale, -18 dB at 1.4x and -13 dB at 2x -- so even at
    # twelve times full scale, pulling the gain down by 22 dB, this is 28 dB
    # cleaner than the old stage was at two.
    @pytest.mark.parametrize("level,worst", [(1.0, -70.0), (2.0, -55.0),
                                             (6.0, -44.0), (12.0, -39.0)])
    def test_it_reduces_gain_instead_of_reshaping(self, sample_rate, level, worst):
        t = np.arange(sample_rate) / sample_rate
        limiter = PeakLimiter(sample_rate)
        x = level * np.sin(2 * np.pi * 200 * t)
        out = np.concatenate([limiter(x[i:i + 512]) for i in range(0, x.size, 512)])
        measured = thd_db(out, sample_rate, 200.0)
        assert measured < worst, f"drive {level}x: {measured:.1f} dB"

    def test_it_leaves_quiet_audio_completely_alone(self, sample_rate):
        """Below the ceiling it must be a pure delay, exactly -- a limiter that
        touches audio it was not asked to touch is a compressor."""
        limiter = PeakLimiter(sample_rate)
        rng = np.random.default_rng(4)
        x = rng.standard_normal(4096)
        x *= 0.9 * limiter.ceiling / np.max(np.abs(x))    # peaks just under it
        out = np.concatenate([limiter(x[i:i + 256]) for i in range(0, x.size, 256)])
        delayed = np.concatenate([np.zeros(limiter.latency_samples), x])[:x.size]
        assert np.max(np.abs(out - delayed)) == 0.0

    @pytest.mark.parametrize("block", [1, 64, 333, 4096, 12000])
    def test_the_result_does_not_depend_on_block_size(self, sample_rate, block):
        """Including blocks past the internal chunk size, where the closed-form
        release has to be stitched across passes."""
        t = np.arange(sample_rate) / sample_rate
        x = 2.0 * np.sin(2 * np.pi * 200 * t) * (0.2 + np.abs(np.sin(2 * np.pi * 3 * t)))

        def run(size):
            limiter = PeakLimiter(sample_rate)
            return np.concatenate([limiter(x[i:i + size]) for i in range(0, x.size, size)])

        assert np.max(np.abs(run(block) - run(1024))) < 1e-12

    def test_it_recovers_between_shouts_without_ducking_the_speech_after(
            self, sample_rate):
        t = np.arange(sample_rate) / sample_rate
        tone = np.sin(2 * np.pi * 200 * t)
        level = np.full(t.size, 0.2)
        level[int(0.2 * sample_rate):int(0.3 * sample_rate)] = 3.0
        limiter = PeakLimiter(sample_rate)
        x = tone * level
        out = np.concatenate([limiter(x[i:i + 256]) for i in range(0, x.size, 256)])
        after = out[int(0.5 * sample_rate):int(0.9 * sample_rate)]
        quiet = x[int(0.5 * sample_rate):int(0.9 * sample_rate)]
        recovered = np.max(np.abs(after)) / np.max(np.abs(quiet))
        assert recovered > 0.99, f"still ducked by {20 * np.log10(recovered):.2f} dB"

    def test_reset_clears_it(self, sample_rate):
        limiter = PeakLimiter(sample_rate)
        limiter(np.full(1024, 5.0))
        limiter.reset()
        x = 0.3 * np.ones(1024)
        out = limiter(x)
        assert np.max(np.abs(out[limiter.latency_samples:])) == pytest.approx(0.3)


class TestShouting:
    @pytest.mark.parametrize("name", presets.names())
    def test_driving_it_to_four_times_full_scale_manufactures_nothing(
            self, sample_rate, name):
        """A band-limited vowel in; anything above its band out was made here.

        This is the only view of clipping any metric in this package has, and
        it had to be added: waveshaping products are harmonic, so inharmonic
        energy counts them as signal and reads *better* as the clipping gets
        worse -- -54.9 dB at a fifth of full scale against -57.5 dB at nearly
        twice it, while audible distortion went from nothing to -13 dB.
        """
        profile = presets.get(name).replace(breathiness=0.0)
        audio = band_limited(sample_rate)
        edge = BAND_EDGE * max(profile.formant_ratio, 1.0)
        made = manufactured_band_db(drive(audio, sample_rate, profile, 0.5),
                                    drive(audio, sample_rate, profile, 4.0),
                                    sample_rate, edge)
        assert made < -70.0, f"{name}: {made:.1f} dB of manufactured band"

    @pytest.mark.parametrize("level", [1.0, 3.0, 20.0])
    def test_the_output_ceiling_holds_however_hard_it_is_driven(
            self, sample_rate, level):
        audio = band_limited(sample_rate)
        out = natvox.process_array(audio * level, sample_rate,
                                   presets.get("female"), 512)
        assert np.all(np.isfinite(out))
        assert np.max(np.abs(out)) <= SAFETY_CEILING + 1e-9

    def test_the_input_band_survives_the_drive(self, sample_rate):
        """Not tearing is half of it; the other half is that the voice is still
        there underneath."""
        audio = band_limited(sample_rate)
        profile = presets.get("female").replace(breathiness=0.0)
        quiet = drive(audio, sample_rate, profile, 0.5)
        loud = drive(audio, sample_rate, profile, 4.0)
        edge = BAND_EDGE * profile.formant_ratio
        assert out_of_band_db(loud, sample_rate, edge) < -25.0

        # Same spectrum in band once the level difference is taken out.  Only
        # bins within 60 dB of the peak: below that the vowel has no energy and
        # the comparison is one noise floor against another.
        def in_band(x):
            spectrum = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
            freqs = np.fft.rfftfreq(x.size, 1.0 / sample_rate)
            share = spectrum[(freqs > 100) & (freqs < edge)]
            return 10 * np.log10(share / max(share.sum(), 1e-30) + 1e-30)

        hot, cool = in_band(loud), in_band(quiet)
        live = cool > cool.max() - 60.0
        assert np.sqrt(np.mean((hot[live] - cool[live]) ** 2)) < 2.0

    def test_a_shout_onset_does_not_pump(self, sample_rate):
        """The loudness matcher was the suspect and is not the culprit: over a
        50 ms window its excursion through a +20 dB step is the same as its
        excursion during steady speech."""
        from synth_speech import VOWELS, glottal_source, vocal_tract

        n = int(1.6 * sample_rate)
        source = glottal_source(np.full(n, 130.0), sample_rate,
                                np.random.default_rng(3), jitter=0.005, shimmer=0.03)
        voice = vocal_tract(source, np.stack([np.full(n, f) for f in VOWELS["a"]]),
                            sample_rate)
        voice /= max(np.abs(voice).max(), 1e-9)
        envelope = np.full(n, 0.06)
        start, stop = int(0.6 * sample_rate), int(1.1 * sample_rate)
        ramp = int(0.015 * sample_rate)
        envelope[start:stop] = 0.6
        envelope[start - ramp:start] = np.linspace(0.06, 0.6, ramp)
        envelope[stop:stop + ramp] = np.linspace(0.6, 0.06, ramp)
        audio = voice * envelope

        profile = presets.get("female")
        wet = natvox.process_array(audio, sample_rate, profile, 512)
        kernel = np.ones(int(0.05 * sample_rate))
        kernel /= kernel.sum()
        dry_env = np.sqrt(np.convolve(audio * audio, kernel, "same"))
        wet_env = np.sqrt(np.convolve(wet * wet, kernel, "same"))

        def spread(region):
            ratio = 20 * np.log10(np.maximum(wet_env[region], 1e-9)
                                  / np.maximum(dry_env[region], 1e-9))
            return float(np.ptp(ratio - np.median(ratio)))

        onset = spread(slice(start - ramp, start + int(0.3 * sample_rate)))
        steady = spread(slice(int(0.2 * sample_rate), int(0.5 * sample_rate)))
        assert onset < steady + 0.5, f"onset {onset:.2f} dB against steady {steady:.2f}"


class TestPitchCeiling:
    def test_a_shout_past_the_conversational_range_is_not_read_an_octave_down(
            self, sample_rate):
        """At a 400 Hz ceiling a sweep to 460 Hz produced 27 octave errors in
        388 voiced frames.  The ceiling costs nothing, so it is set for
        shouting rather than for conversation."""
        from synth_speech import VOWELS, glottal_source, vocal_tract

        n = int(2.0 * sample_rate)
        contour = np.geomspace(120.0, 460.0, n)
        source = glottal_source(contour, sample_rate, np.random.default_rng(3),
                                jitter=0.004, shimmer=0.02)
        audio = vocal_tract(source, np.stack([np.full(n, f) for f in VOWELS["a"]]),
                            sample_rate)
        audio = 0.5 * audio / max(np.abs(audio).max(), 1e-9)

        profile = presets.get("female").replace(breathiness=0.0)
        wet = natvox.process_array(audio, sample_rate, profile, 512)
        positions, values = pitch_track(wet, sample_rate, f0_min=60, f0_max=1400)
        voiced = values > 0
        wanted = np.interp(positions, np.arange(n), contour) * profile.pitch_ratio
        cents = 1200 * np.log2(np.maximum(values[voiced], 1e-9) / wanted[voiced])
        octaves = int(np.sum(np.abs(cents) > 550))
        assert octaves <= 3, f"{octaves} octave errors in {voiced.sum()} frames"

    @pytest.mark.parametrize("name", presets.names())
    def test_every_preset_can_be_shouted_into(self, name):
        assert presets.get(name).f0_max >= 600.0

    @pytest.mark.parametrize("f0", [85.0, 110.0, 200.0])
    def test_a_high_ceiling_does_not_confuse_a_low_voice(self, sample_rate, f0):
        """The risk of the other direction: a wider search finding a harmonic
        and reporting double.  Measured at zero across the adult male range."""
        from synth_speech import VOWELS, glottal_source, vocal_tract
        from natvox.dsp.f0 import YinF0Tracker

        n = int(1.5 * sample_rate)
        source = glottal_source(np.full(n, f0), sample_rate,
                                np.random.default_rng(7), jitter=0.006, shimmer=0.04)
        audio = vocal_tract(source, np.stack([np.full(n, f) for f in VOWELS["a"]]),
                            sample_rate)
        audio = 0.5 * audio / max(np.abs(audio).max(), 1e-9)

        tracker = YinF0Tracker(sample_rate, 70.0, 800.0)
        position, seen = tracker.half, []
        while position + tracker.lookahead <= audio.size:
            segment = np.zeros(tracker.span)
            start = position - tracker.half
            lo, hi = max(start, 0), min(start + tracker.span, audio.size)
            segment[lo - start:hi - start] = audio[lo:hi]
            frame = tracker.estimate(segment, position)
            if frame.voiced:
                seen.append(frame.f0)
            position += 240
        assert seen, "nothing was called voiced"
        assert all(f0 / 1.4 < value < f0 * 1.4 for value in seen)


class TestIrregularity:
    """Whether the voice comes back as irregular as it went in.

    Over-regularity is the robotic tell.  The number this package used to quote
    for shimmer -- flattened to 0.55x -- was an artifact of measuring peak
    amplitude per period, which moves whenever anything changes the phase
    inside a period.  A plain passthrough through the rumble filter measured
    0.61x that way.  Period RMS is the measure that survives the comparison.
    """

    @pytest.mark.parametrize("name", ["off", "brighter", "younger",
                                      "male_to_female", "female", "female_to_male"])
    def test_amplitude_irregularity_is_not_flattened(self, sample_rate, name):
        from synth_speech import human_vowel

        dry = human_vowel(sample_rate)
        region = (int(0.08 * sample_rate), dry.size - int(0.08 * sample_rate))
        profile = presets.get(name)
        wet = natvox.process_array(dry, sample_rate, profile, 512)
        _, dry_shimmer = jitter_shimmer(dry, sample_rate, 120.0, region)
        _, wet_shimmer = jitter_shimmer(wet, sample_rate,
                                        120.0 * profile.pitch_ratio, region)
        ratio = wet_shimmer / dry_shimmer
        assert 0.85 < ratio < 2.0, f"{name}: shimmer came back at {ratio:.2f}x"

    def test_a_passthrough_returns_exactly_what_it_was_given(self, sample_rate):
        """The measurement's own control.  Without it the rumble filter's phase
        response reads as the engine destroying a third of the speaker's
        shimmer."""
        from synth_speech import human_vowel

        dry = human_vowel(sample_rate)
        region = (int(0.08 * sample_rate), dry.size - int(0.08 * sample_rate))
        wet = natvox.process_array(dry, sample_rate,
                                   VoiceProfile(highpass_hz=0.0), 512)
        _, dry_shimmer = jitter_shimmer(dry, sample_rate, 120.0, region)
        _, wet_shimmer = jitter_shimmer(wet, sample_rate, 120.0, region)
        assert wet_shimmer / dry_shimmer == pytest.approx(1.0, abs=0.02)

    @pytest.mark.parametrize("name", ["male_to_female_subtle", "male_to_female",
                                      "female", "female_to_male"])
    def test_it_does_not_invent_irregularity(self, sustained_vowel, sample_rate, name):
        """A steady input must not come back wobbling.  Measured at 0.11-0.32%
        of shimmer against 1.7-3.7% for a human voice, and exactly zero
        jitter."""
        dry, f0 = sustained_vowel
        region = (int(0.1 * sample_rate), dry.size - int(0.1 * sample_rate))
        profile = presets.get(name)
        wet = natvox.process_array(dry, sample_rate, profile, 512)
        jitter, shimmer = jitter_shimmer(wet, sample_rate,
                                         f0 * profile.pitch_ratio, region)
        assert jitter < 0.0005, f"{name}: invented {jitter:.4%} of jitter"
        assert shimmer < 0.005, f"{name}: invented {shimmer:.4%} of shimmer"
