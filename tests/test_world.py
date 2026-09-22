"""The analysis-and-resynthesis engine, and why it exists.

The PSOLA path moves the recording's own waveform, so the recording's own
noise moves with it, and past about eight semitones the grains stop
overlapping the way a vocal tract could have produced.  Both limits showed up
on the first real recording this project ever had.

This engine takes the voice apart and builds a new one.  Everything below is
measured against either a signal whose true answer is known, or the PSOLA
path on the same input.
"""
from __future__ import annotations

import numpy as np
import pytest

from natvox.dsp import world

pytestmark = pytest.mark.skipif(not world.available(), reason="pyworld not installed")


def psola(audio, sr, pitch, formant=0.0):
    from natvox.config import VoiceProfile
    from natvox.engine import VoiceChanger

    changer = VoiceChanger(sr, VoiceProfile(
        pitch_semitones=pitch, formant_semitones=formant, f0_min=70.0,
        intonation=1.0, breathiness=0.0, tilt_db=0.0))
    blocks = [changer.process(audio[i:i + 256]) for i in range(0, audio.size, 256)]
    blocks.append(changer.flush())
    return np.concatenate(blocks)[changer.latency_samples:][:audio.size]


def delivered_semitones(dry, wet, sr):
    from evaluate import pitch_track

    _, a = pitch_track(dry, sr)
    _, b = pitch_track(wet, sr)
    n = min(a.size, b.size)
    both = (a[:n] > 0) & (b[:n] > 0)
    return float(12 * np.log2(np.median(b[:n][both] / a[:n][both])))


class TestThePitchItDelivers:
    """The headline claim, and the reason this engine was written."""

    @pytest.mark.parametrize("asked", [4.0, 7.0, 10.0, 13.0])
    def test_it_lands_exactly_where_it_was_asked(self, utterance, sample_rate,
                                                 asked):
        audio, _ = utterance
        got = delivered_semitones(audio, world.convert(audio, sample_rate,
                                                       pitch_semitones=asked),
                                  sample_rate)
        assert got == pytest.approx(asked, abs=0.05)

    def test_there_is_no_ceiling_where_psola_has_one(self, utterance, sample_rate):
        """+-8 st is documented as where the time-domain method stops being
        transparent. This one is as exact at +13 as it is at +4."""
        audio, _ = utterance
        small = abs(delivered_semitones(
            audio, world.convert(audio, sample_rate, pitch_semitones=4.0),
            sample_rate) - 4.0)
        large = abs(delivered_semitones(
            audio, world.convert(audio, sample_rate, pitch_semitones=13.0),
            sample_rate) - 13.0)
        assert large < 0.05 and small < 0.05

    def test_down_works_as_well_as_up(self, utterance, sample_rate):
        audio, _ = utterance
        got = delivered_semitones(
            audio, world.convert(audio, sample_rate, pitch_semitones=-5.0),
            sample_rate)
        assert got == pytest.approx(-5.0, abs=0.05)


class TestItIsQuieterThanTheWaveformMethod:
    """The measurement that started it, on the kind of input that showed it."""

    def test_it_beats_psola_on_a_noisy_recording(self, utterance, sample_rate):
        from evaluate import harmonic_to_noise_db

        audio, _ = utterance
        rng = np.random.default_rng(7)
        level = np.sqrt(np.mean(audio * audio)) / 10 ** (18 / 20)
        noisy = audio + rng.normal(0, level, audio.size)
        by_world = harmonic_to_noise_db(
            world.convert(noisy, sample_rate, pitch_semitones=10.0,
                          formant_semitones=2.6), sample_rate)
        by_psola = harmonic_to_noise_db(psola(noisy, sample_rate, 10.0, 2.6),
                                        sample_rate)
        # 1.0 dB, not the 4.0 the real recording gave (17.2 against 13.2).
        # The synthetic utterance understates the gap and it is worth knowing
        # by how much: its pitch contour is smooth and its noise is white, and
        # both of those are kind to a method that stretches the waveform. A
        # person's pitch moves, and the room is not white.
        assert by_world > by_psola + 1.0, (by_world, by_psola)

    def test_it_does_not_get_worse_as_the_shift_grows(self, utterance, sample_rate):
        """PSOLA does. That asymmetry is the whole argument for this engine."""
        from evaluate import harmonic_to_noise_db

        audio, _ = utterance
        near = harmonic_to_noise_db(
            world.convert(audio, sample_rate, pitch_semitones=7.0), sample_rate)
        far = harmonic_to_noise_db(
            world.convert(audio, sample_rate, pitch_semitones=13.0), sample_rate)
        assert far > near - 3.0, (near, far)


class TestTheVocalTract:
    def test_a_formant_shift_moves_the_envelope_up(self, utterance, sample_rate):
        from evaluate import mean_envelope

        audio, _ = utterance
        flat = mean_envelope(audio, sample_rate)
        moved = mean_envelope(world.convert(audio, sample_rate,
                                            formant_semitones=4.0), sample_rate)
        freq = np.linspace(0, sample_rate / 2, flat.size)
        band = (freq > 300) & (freq < 4000)
        centre = lambda e: float(np.sum(freq[band] * 10 ** (e[band] / 10))
                                 / np.sum(10 ** (e[band] / 10)))
        assert centre(moved) > centre(flat)

    def test_pitch_alone_leaves_the_tract_alone(self, utterance, sample_rate):
        """The two controls are independent, which is what stops a voice
        turning into a chipmunk."""
        from evaluate import mean_envelope

        audio, _ = utterance
        before = mean_envelope(audio, sample_rate)
        after = mean_envelope(world.convert(audio, sample_rate,
                                            pitch_semitones=10.0), sample_rate)
        freq = np.linspace(0, sample_rate / 2, before.size)
        band = (freq > 500) & (freq < 4000)
        assert float(np.mean(np.abs(after[band] - before[band]))) < 6.0

    def test_warping_by_nothing_changes_nothing(self):
        sp = np.random.default_rng(0).random((20, 513)) + 0.1
        assert world.warp_envelope(sp, 48000, 0.0) is sp

    def test_warping_up_reads_from_lower_down(self):
        """A formant at 1000 Hz has to come out at 1000 * 2**(st/12)."""
        bins = 513
        freq = np.linspace(0, 24000, bins)
        sp = np.exp(-((freq - 1000.0) / 60.0) ** 2)[None, :] + 1e-9
        moved = world.warp_envelope(sp, 48000, 4.0)
        assert freq[int(np.argmax(moved[0]))] == pytest.approx(
            1000 * 2 ** (4 / 12), rel=0.05)


class TestBreathiness:
    def test_it_raises_the_aperiodic_share(self, utterance, sample_rate):
        audio, _ = utterance
        _, _, dry = world.analyse(audio, sample_rate)
        breathy = world.convert(audio, sample_rate, breathiness=0.5)
        _, _, wet = world.analyse(breathy, sample_rate)
        assert wet.mean() > dry.mean()

    def test_zero_leaves_it_alone(self, utterance, sample_rate):
        audio, _ = utterance
        a = world.convert(audio, sample_rate, pitch_semitones=2.0)
        b = world.convert(audio, sample_rate, pitch_semitones=2.0, breathiness=0.0)
        assert np.array_equal(a, b)


class TestItSurvivesWhateverItIsGiven:
    def test_nothing_at_all(self):
        assert world.convert(np.zeros(0), 48000).size == 0

    def test_silence(self, sample_rate):
        y = world.convert(np.zeros(sample_rate), sample_rate, pitch_semitones=10.0)
        assert y.size == sample_rate and np.all(np.isfinite(y))

    def test_shorter_than_a_frame(self, sample_rate):
        y = world.convert(np.zeros(64), sample_rate, pitch_semitones=5.0)
        assert y.size == 64

    def test_the_length_is_always_the_input_length(self, utterance, sample_rate):
        audio, _ = utterance
        for st in (-6.0, 0.0, 12.0):
            assert world.convert(audio, sample_rate,
                                 pitch_semitones=st).size == audio.size

    def test_it_never_returns_something_unplayable(self, utterance, sample_rate):
        audio, _ = utterance
        y = world.convert(audio, sample_rate, pitch_semitones=13.0,
                          formant_semitones=4.0, breathiness=0.3)
        assert np.all(np.isfinite(y))
        assert np.max(np.abs(y)) < 10.0


class TestTheFastPath:
    """``dio`` runs at 62x real time against ``harvest``'s 5x, which is the
    difference between an engine that can run live and one that cannot."""

    def test_both_estimators_land_on_the_same_pitch(self, utterance, sample_rate):
        audio, _ = utterance
        slow = world.convert(audio, sample_rate, pitch_semitones=10.0, fast=False)
        quick = world.convert(audio, sample_rate, pitch_semitones=10.0, fast=True)
        assert delivered_semitones(audio, quick, sample_rate) == pytest.approx(
            delivered_semitones(audio, slow, sample_rate), abs=0.3)

    def test_the_fast_one_is_actually_faster(self, utterance, sample_rate):
        import time

        audio, _ = utterance
        def clock(fast):
            t = time.perf_counter()
            world.analyse(audio, sample_rate, fast=fast)
            return time.perf_counter() - t
        assert clock(True) < clock(False)


def sung(sample_rate, f0_hz, seconds=1.5, level=0.3, vibrato_st=0.4, seed=3):
    """A sustained sung note: steady pitch, vibrato, a real vowel."""
    from synth_speech import VOWELS, glottal_source, vocal_tract

    rng = np.random.default_rng(seed)
    n = int(sample_rate * seconds)
    t = np.arange(n) / sample_rate
    f0 = f0_hz * 2 ** (vibrato_st / 12 * np.sin(2 * np.pi * 5.5 * t))
    source = glottal_source(f0, sample_rate, rng, jitter=0.006, shimmer=0.02)
    vowel = VOWELS["a"]
    y = vocal_tract(source, np.stack([np.full(n, vowel[i]) for i in range(3)]),
                    sample_rate)
    return np.ascontiguousarray(y / max(np.max(np.abs(y)), 1e-9) * level)


def fundamental(y, sample_rate, lo=100.0, hi=2500.0):
    """The fundamental of a sustained note, by autocorrelation.

    Three instruments preceded this one and each failed differently. The
    repository's pitch_track tops out below a sung note put up ten semitones
    and returned zero for audio that was correct, which read as the engine
    breaking. "The strongest partial" returned 791 Hz for a 150 Hz note,
    because the strongest partial of a sung /a/ is whichever harmonic sits
    under F1. A harmonic product spectrum then locked onto the subharmonic at
    1069 Hz, where a handful of tiny factors beat the true peak's product.

    Autocorrelation asks the only question that matters for a held note --
    what does this waveform repeat at -- and has no spectrum to be fooled by.
    """
    seg = np.asarray(y, dtype=np.float64)
    seg = seg[len(seg) // 3:len(seg) // 3 + sample_rate // 2]
    if seg.size < 4 * int(sample_rate / lo):
        return 0.0
    seg = seg - seg.mean()
    if seg.dot(seg) < 1e-18:
        return 0.0
    n = 1 << int(np.ceil(np.log2(2 * seg.size)))
    spectrum = np.fft.rfft(seg * np.hanning(seg.size), n)
    acf = np.fft.irfft(np.abs(spectrum) ** 2, n)[:seg.size]
    lo_lag, hi_lag = int(sample_rate / hi), int(sample_rate / lo)
    window = acf[lo_lag:hi_lag + 1]
    if window.size == 0 or acf[0] <= 0:
        return 0.0
    return float(sample_rate / (lo_lag + int(np.argmax(window))))


class TestSingingAndShouting:
    """A voice that is only ever tested on quiet talking breaks on both.

    Everything above this class was measured on connected speech at
    conversational level.  Somebody who wants to sing through it, or raise
    their voice, meets two failures that speech never reaches.
    """

    @pytest.mark.parametrize("hz", [150, 250, 400, 600, 800])
    def test_a_sung_note_is_moved_by_what_was_asked(self, sample_rate, hz):
        """The ratio, measured with one instrument on both ends.

        Not the absolute pitch: at 1069 Hz the harmonic product spectrum
        locks onto the subharmonic, and a test that compared against an
        absolute number would have been testing that failure rather than the
        engine. The same bias on both ends cancels.
        """
        note = sung(sample_rate, float(hz))
        out = world.convert(note, sample_rate, pitch_semitones=10.0,
                            formant_semitones=2.0)
        before = fundamental(note, sample_rate, hi=2500.0)
        after = fundamental(out, sample_rate, hi=2500.0)
        assert before > 0 and after > 0
        assert 12 * np.log2(after / before) == pytest.approx(10.0, abs=0.4)

    def test_the_ceiling_is_set_for_singing_not_speaking(self):
        """At 600 Hz a sung note came out at 692 where it should have reached
        1069: the estimate is clipped at the ceiling and the shift is then
        applied to the wrong number. That is a voice tearing, not a voice
        going high."""
        assert world.F0_CEIL >= 1000.0

    def test_raising_it_costs_speech_nothing(self, utterance, sample_rate):
        """Which is why it could be raised at all."""
        audio, _ = utterance
        f0, _, _ = world.analyse(audio, sample_rate)
        voiced = f0[f0 > 0]
        assert voiced.size
        assert float(np.mean(voiced > 400.0)) < 0.02, "no octave errors upward"

    @pytest.mark.parametrize("level", [0.3, 0.9, 1.0])
    def test_nothing_ever_leaves_above_full_scale(self, sample_rate, level):
        """Resynthesis is not gain-preserving: a note peaking at 0 dBFS came
        back at +1.0 dB, which clips on the way to the speaker."""
        note = np.clip(sung(sample_rate, 250.0, level=1.0) * level, -1.0, 1.0)
        out = world.convert(note, sample_rate, pitch_semitones=10.0,
                            formant_semitones=2.0)
        assert np.max(np.abs(out)) <= 1.0

    def test_it_loses_level_rather_than_gaining_distortion(self, sample_rate):
        """A clipper would hold the ceiling too, and tear the voice doing it."""
        from evaluate import harmonic_to_noise_db

        quiet = world.convert(sung(sample_rate, 250.0, level=0.2), sample_rate,
                              pitch_semitones=10.0)
        loud = world.convert(sung(sample_rate, 250.0, level=1.0), sample_rate,
                             pitch_semitones=10.0)
        assert harmonic_to_noise_db(loud, sample_rate) > \
            harmonic_to_noise_db(quiet, sample_rate) - 3.0

    def test_a_clipped_input_degrades_gently(self, sample_rate):
        """Shouting into a microphone clips before this ever sees it."""
        from evaluate import harmonic_to_noise_db

        note = sung(sample_rate, 250.0, level=1.0)
        gentle = world.convert(np.clip(note * 1.5, -1, 1), sample_rate,
                               pitch_semitones=10.0)
        brutal = world.convert(np.clip(note * 6.0, -1, 1), sample_rate,
                               pitch_semitones=10.0)
        # 4.9 dB, measured. Driving a signal six times into the rail is not
        # free and the test says what it actually costs rather than the round
        # number that would have been convenient.
        assert harmonic_to_noise_db(brutal, sample_rate) > \
            harmonic_to_noise_db(gentle, sample_rate) - 5.5

    def test_the_limiter_changes_the_level_and_nothing_else(self, sample_rate):
        """Its own delay is removed, so a quiet signal comes back untouched."""
        quiet = sung(sample_rate, 250.0, level=0.05)
        assert np.max(np.abs(quiet)) < world.CEILING
        assert world.limit(quiet, sample_rate) == pytest.approx(quiet, abs=1e-9)


class TestConvertingALiveStream:
    """The vocoder wants a whole utterance; a callback hands it 256 samples.

    The gap between those is a windowing problem, and the traps are the ones
    every block-based model meets: windows synthesised independently do not
    agree at their seam, and a window too short to hold a few pitch periods
    has nothing to estimate a vocal tract from.
    """

    def stream(self, audio, sample_rate, **kw):
        live = world.LiveConverter(sample_rate, pitch_semitones=10.0,
                                   formant_semitones=2.0, **kw)
        blocks = [live.process(audio[i:i + 256])
                  for i in range(0, audio.size, 256)]
        blocks.append(live.flush())
        out = np.concatenate(blocks)
        return live, out[live.latency_samples:live.latency_samples + audio.size]

    def test_it_converts_as_well_as_the_offline_path(self, utterance, sample_rate):
        from evaluate import harmonic_to_noise_db

        audio, _ = utterance
        _, streamed = self.stream(audio, sample_rate)
        offline = world.convert(audio, sample_rate, pitch_semitones=10.0,
                                formant_semitones=2.0)
        assert harmonic_to_noise_db(streamed, sample_rate) > \
            harmonic_to_noise_db(offline, sample_rate) - 3.0

    def test_the_seams_add_no_step_the_speech_did_not_have(self, utterance,
                                                           sample_rate):
        """Against the input's own worst transient, not an absolute bar.

        A fixed threshold here read 0 clicks at one context and 98 at the
        next, and the two recordings' worst jumps were 0.2163 and 0.2311 --
        the same event either side of the line. The input itself steps 44x
        its typical sample-to-sample move at a plosive, so the only question
        the splice raises is whether it adds one.
        """
        def worst(signal):
            steps = np.abs(np.diff(signal))
            return float(np.max(steps) / np.median(steps[steps > 0]))

        audio, _ = utterance
        _, out = self.stream(audio, sample_rate)
        assert worst(out) < worst(audio) * 1.5

    def test_it_delivers_the_pitch_it_was_asked_for(self, utterance, sample_rate):
        audio, _ = utterance
        _, out = self.stream(audio, sample_rate)
        assert delivered_semitones(audio, out, sample_rate) == pytest.approx(
            10.0, abs=0.5)

    def test_it_runs_faster_than_real_time(self, utterance, sample_rate):
        """The window is analysed three times over -- hop plus context on each
        side -- so the vocoder has to beat real time by more than three."""
        import time

        audio, _ = utterance
        start = time.perf_counter()
        self.stream(audio, sample_rate)
        speed = (audio.size / sample_rate) / (time.perf_counter() - start)
        assert speed > 1.5, f"only x{speed:.1f} real time"

    def test_it_converts_at_24_khz_whatever_the_stream_is(self, sample_rate):
        live = world.LiveConverter(sample_rate)
        assert live.internal_rate == world.INTERNAL_RATE < sample_rate

    def test_a_low_rate_stream_is_not_upsampled_to_meet_it(self):
        live = world.LiveConverter(16000)
        assert live.internal_rate == 16000

    def test_a_context_too_short_to_hold_a_voice_is_refused(self, sample_rate):
        """30 ms at 70 Hz returned a signal at the right level with no
        periodicity in it at all. Silently."""
        with pytest.raises(ValueError, match="no voice in it"):
            world.LiveConverter(sample_rate, context_ms=30.0, f0_min=70.0)

    def test_the_refusal_moves_with_the_tracking_floor(self, sample_rate):
        """A higher floor is a shorter period, so less context suffices."""
        world.LiveConverter(sample_rate, context_ms=30.0, f0_min=120.0)
        with pytest.raises(ValueError):
            world.LiveConverter(sample_rate, context_ms=30.0, f0_min=60.0)

    def test_it_reports_a_latency_the_caller_can_act_on(self, sample_rate):
        live = world.LiveConverter(sample_rate, hop_ms=80.0, context_ms=80.0)
        assert live.latency_ms == pytest.approx(160.0, abs=1.0)
        assert live.latency_samples == int(0.16 * sample_rate)

    def test_the_default_delay_is_the_one_worth_shipping(self, sample_rate):
        """160 ms is the worst range there is for hearing your own voice --
        it is what delayed-auditory-feedback devices use to disrupt speech.
        Aligning the seam is what brought it down."""
        assert world.LiveConverter(sample_rate).latency_ms <= 80.0

    def test_a_short_hop_is_as_clean_as_a_long_one(self, utterance, sample_rate):
        """Without alignment it is not: 21.6 dB at an 80 ms hop, 17.0 at 40,
        8.2 at 20, because every seam cancels harmonics that two windows
        placed differently."""
        from evaluate import harmonic_to_noise_db

        audio, _ = utterance
        _, short = self.stream(audio, sample_rate, hop_ms=20.0, context_ms=45.0)
        _, long = self.stream(audio, sample_rate, hop_ms=80.0, context_ms=80.0)
        assert harmonic_to_noise_db(short, sample_rate) > \
            harmonic_to_noise_db(long, sample_rate) - 1.5

    def test_turning_the_alignment_off_is_visibly_worse(self, utterance,
                                                        sample_rate):
        """The measurement the default rests on."""
        from evaluate import harmonic_to_noise_db

        audio, _ = utterance
        _, aligned = self.stream(audio, sample_rate, hop_ms=20.0,
                                 context_ms=45.0)
        _, blind = self.stream(audio, sample_rate, hop_ms=20.0, context_ms=45.0,
                               align_ms=0.0)
        assert harmonic_to_noise_db(aligned, sample_rate) > \
            harmonic_to_noise_db(blind, sample_rate) + 5.0

    def test_a_fade_too_short_for_its_own_search_is_refused(self, sample_rate):
        """The alignment moves where the next hop is read from; a fade
        shorter than that steps across the move instead of sliding over it."""
        with pytest.raises(ValueError, match="steps rather than slides"):
            world.LiveConverter(sample_rate, crossfade_ms=6.0, align_ms=3.0)

    def test_reset_clears_it(self, utterance, sample_rate):
        audio, _ = utterance
        live = world.LiveConverter(sample_rate, pitch_semitones=7.0)
        first = np.concatenate([live.process(audio[i:i + 256])
                                for i in range(0, 48000, 256)])
        live.reset()
        again = np.concatenate([live.process(audio[i:i + 256])
                                for i in range(0, 48000, 256)])
        assert np.allclose(first, again)

    def test_block_size_does_not_change_the_answer(self, utterance, sample_rate):
        audio, _ = utterance
        outs = []
        for block in (128, 512):
            live = world.LiveConverter(sample_rate, pitch_semitones=7.0)
            y = np.concatenate([live.process(audio[i:i + block])
                                for i in range(0, audio.size, block)])
            outs.append(y[live.latency_samples:live.latency_samples + 40000])
        assert np.allclose(outs[0], outs[1], atol=1e-9)
