"""Measuring the round trip, against delays whose true value is known.

Nothing here opens a device.  The estimator is fed a signal that was delayed
by an exact number of samples, so every assertion is against the truth rather
than against another estimate -- which is the only way to find out whether the
thing that decides "is a better virtual cable worth building" can be trusted.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from natvox.app import loopback
from natvox.app.backend import AudioUnavailable

SR = 48000


def echo(sent, delay_samples, tail=24000, gain=1.0, noise=0.0, seed=0):
    """``sent`` arriving ``delay_samples`` late in a longer recording."""
    out = np.zeros(int(delay_samples) + sent.size + tail)
    out[int(delay_samples):int(delay_samples) + sent.size] = sent * gain
    if noise:
        out += np.random.default_rng(seed).standard_normal(out.size) * noise
    return out


class TestTheProbe:
    def test_it_sweeps_upwards_over_the_band_it_claims(self):
        sent = loopback.probe(SR)
        assert sent.size == int(round(loopback.PROBE_SECONDS * SR))
        # Instantaneous frequency from the zero crossings of each half.
        half = sent.size // 2
        low = np.count_nonzero(np.diff(np.signbit(sent[:half]))) / (half / SR) / 2
        high = np.count_nonzero(np.diff(np.signbit(sent[half:]))) / (half / SR) / 2
        assert low < high
        assert loopback.PROBE_LOW_HZ * 0.8 < low
        assert high < loopback.PROBE_HIGH_HZ * 1.2

    def test_it_starts_and_stops_at_zero(self):
        """A probe that steps to level is a click with a sweep attached, and
        the click is what the room hears."""
        sent = loopback.probe(SR)
        assert abs(sent[0]) < 1e-9 and abs(sent[-1]) < 1e-9

    def test_it_is_not_played_at_full_scale(self):
        """Whoever runs this is wearing headphones, because the README told
        them to.  The matched filter finds an echo 34 dB below this level, so
        full scale would buy nothing and cost an ear."""
        sent = loopback.probe(SR)
        assert np.max(np.abs(sent)) == pytest.approx(loopback.PROBE_LEVEL, rel=0.02)
        assert loopback.PROBE_LEVEL <= 0.3

    def test_a_silly_short_request_still_produces_a_signal(self):
        assert loopback.probe(SR, seconds=1e-6).size >= 16


class TestTheEstimator:
    @pytest.mark.parametrize("delay", [0, 1, 137, 4801, 48000])
    def test_it_finds_a_whole_sample_delay_exactly(self, delay):
        sent = loopback.probe(SR)
        found = loopback.estimate_delay(echo(sent, delay), sent, SR)
        assert found is not None
        assert abs(found * SR - delay) < 0.05

    @pytest.mark.parametrize("fraction", [0.25, 0.5, 0.75])
    def test_it_reads_between_samples(self, fraction):
        """At 48 kHz one sample is 0.02 ms; the numbers being compared here
        differ by fractions of a millisecond, so the peak alone is too coarse."""
        sent = loopback.probe(SR)
        up = signal.resample_poly(
            np.concatenate([np.zeros(500), sent, np.zeros(24000)]), 4, 1)
        pad = int(round(4 * fraction))
        shifted = np.concatenate([np.zeros(pad), up])[::4]
        found = loopback.estimate_delay(shifted, sent, SR)
        assert found is not None
        assert abs(found * SR - (500 + fraction)) < 0.35

    def test_it_survives_a_quiet_echo(self):
        """Normalised, so the score is how much it looks like the probe rather
        than how loud it is -- a cable set 34 dB down is still a cable."""
        sent = loopback.probe(SR)
        found = loopback.estimate_delay(echo(sent, 900, gain=0.02), sent, SR)
        assert found is not None and abs(found * SR - 900) < 0.5

    def test_it_survives_a_path_that_ate_half_the_band(self):
        """A telephone-band path in the way widens the peak; it must not move
        it, or the answer would depend on what the cable did to the sound.

        The band limit is linear phase so that its own delay is exactly half
        its length, and the truth stays a known number rather than a guess: a
        Butterworth here would add 0.3 ms of group delay of its own and there
        would be no way to tell that from the estimator being wrong.
        """
        sent = loopback.probe(SR)
        taps = signal.firwin(511, [300, 3400], pass_zero=False, fs=SR)
        heard = np.convolve(echo(sent, 1500), taps)
        found = loopback.estimate_delay(heard, sent, SR)
        assert found is not None
        assert abs(found * SR - (1500 + 255)) < 1.0

    def test_it_survives_noise_on_top_of_the_echo(self):
        sent = loopback.probe(SR)
        found = loopback.estimate_delay(echo(sent, 2200, gain=0.3, noise=0.05),
                                        sent, SR)
        assert found is not None and abs(found * SR - 2200) < 1.0


class TestWhatItRefuses:
    """A confident wrong number is worse than no number.

    This is the part that a peak-against-its-neighbours test gets wrong: the
    largest of fifty thousand Gaussian lags does stand out from the median, by
    about sqrt(2 ln N), which depends on how long you listened rather than on
    whether anything came back.  Such a test called white noise a 0.88-second
    delay.
    """

    def _nothing(self, kind, rng):
        n = 48000
        if kind == "silence":
            return np.zeros(n)
        if kind == "white noise":
            return rng.standard_normal(n) * 0.2
        if kind == "a tone":
            return 0.5 * np.sin(2 * np.pi * 440 * np.arange(n) / SR)
        if kind == "speech-shaped noise":
            return signal.sosfilt(
                signal.butter(4, [200, 3500], "bandpass", fs=SR, output="sos"),
                rng.standard_normal(n)) * 0.5
        raise AssertionError(kind)

    @pytest.mark.parametrize("kind", ["silence", "white noise", "a tone",
                                      "speech-shaped noise"])
    @pytest.mark.parametrize("seed", range(6))
    def test_it_says_nothing_came_back(self, kind, seed):
        sent = loopback.probe(SR)
        heard = self._nothing(kind, np.random.default_rng(seed))
        assert loopback.estimate_delay(heard, sent, SR) is None

    def test_a_recording_shorter_than_the_probe_is_refused(self):
        sent = loopback.probe(SR)
        assert loopback.estimate_delay(sent[:100], sent, SR) is None

    def test_an_absurd_probe_is_refused(self):
        assert loopback.estimate_delay(np.zeros(1000), np.zeros(4), SR) is None


class TestMeasure:
    def test_it_takes_the_median_of_several_tries(self):
        delays = iter([1000, 1000, 9000, 1000, 1000])

        def play_and_record(sent):
            return echo(sent, next(delays))

        trip = loopback.measure(play_and_record, SR, attempts=5)
        assert trip.attempts == 5
        assert abs(trip.measured_ms - 1000 / SR * 1000.0) < 0.05

    def test_the_spread_shows_an_unstable_path(self):
        """A path that answers differently each time is resampling or
        rebuffering, and the median alone would hide it."""
        delays = iter([1000, 1480, 1000, 1480, 1000])
        trip = loopback.measure(lambda sent: echo(sent, next(delays)), SR,
                                attempts=5)
        assert trip.spread_ms == pytest.approx(480 / SR * 1000.0, abs=0.05)

    def test_a_try_that_heard_nothing_is_dropped_not_counted(self):
        answers = iter([True, False, True])

        def play_and_record(sent):
            if next(answers):
                return echo(sent, 1200)
            return np.zeros(48000)

        trip = loopback.measure(play_and_record, SR, attempts=3)
        assert trip.attempts == 2
        assert abs(trip.measured_ms - 1200 / SR * 1000.0) < 0.05

    def test_what_is_left_over_is_what_nobody_reported(self):
        trip = loopback.measure(lambda sent: echo(sent, 4800), SR, attempts=1,
                                reported_ms=10.0, engine_ms=60.0)
        assert trip.measured_ms == pytest.approx(100.0, abs=0.05)
        assert trip.unexplained_ms == pytest.approx(30.0, abs=0.05)

    def test_hearing_nothing_at_all_says_what_to_check(self):
        trip = loopback.measure(lambda sent: np.zeros(48000), SR, attempts=2)
        assert trip.attempts == 0
        assert np.isnan(trip.measured_ms)
        assert "loops round" in trip.summary()

    def test_the_summary_names_every_part_it_accounted_for(self):
        trip = loopback.measure(lambda sent: echo(sent, 4800), SR, attempts=3,
                                reported_ms=10.0, engine_ms=60.0)
        text = trip.summary()
        assert "round trip" in text and "100.0 ms" in text
        assert "engine" in text and "buffers" in text and "unexplained" in text
        assert "spread" in text


class TestThroughDevices:
    def test_a_device_that_will_not_open_is_a_sentence_not_a_traceback(self):
        """There is no sound card here, so this is the real failure path.

        PortAudio reports a missing device, a rate the pair cannot agree on and
        a device held by something else as the same exception type with a
        different string, and the string is the useful part.
        """
        try:
            with pytest.raises(AudioUnavailable) as raised:
                loopback.through_devices(999, 999, attempts=1)
        except AudioUnavailable as exc:         # no PortAudio at all
            assert "install" in str(exc).lower()
            return
        assert str(raised.value)
