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


class TestTheWarmUp:
    """The probe used to sit at frame 0 of the array handed to the device, so
    it left in the stream's very first callback.  Whether a driver has settled
    by then is an empirical question, so the lead-in is a knob -- and the knob
    only means anything if the answer does not depend on it."""

    @pytest.mark.parametrize("warmup", [0.0, 0.05, 0.25, 1.0])
    def test_the_answer_does_not_depend_on_it(self, warmup):
        """The lead-in is subtracted again, so changing it is not a change of
        units.  Without that, two warm-ups could not be compared at all."""
        def play_and_record(signal):
            return echo(signal, 1500, tail=0)

        trip = loopback.measure(play_and_record, SR, attempts=1,
                                warmup_seconds=warmup)
        assert trip.measured_ms == pytest.approx(1500 / SR * 1000.0, abs=0.05)

    def test_the_silence_really_is_played(self):
        seen = {}

        def play_and_record(signal):
            seen["length"] = signal.size
            return echo(signal, 100, tail=0)

        loopback.measure(play_and_record, SR, attempts=1, warmup_seconds=0.25,
                         tail_seconds=0.5)
        assert seen["length"] == pytest.approx(
            int(0.25 * SR) + loopback.probe(SR).size + int(0.5 * SR))

    def test_an_echo_from_before_it_was_sent_is_dropped(self):
        """Not an echo.  Keeping it would put a negative round trip into the
        median; dropping it costs an attempt and the count says so."""
        sent = loopback.probe(SR)

        def play_and_record(signal):
            # The probe appears 10 ms EARLIER than the warm-up says it left.
            out = np.zeros(signal.size)
            at = int(0.24 * SR)
            out[at:at + sent.size] = sent
            return out

        trip = loopback.measure(play_and_record, SR, attempts=3,
                                warmup_seconds=0.25)
        assert trip.attempts == 0
        assert "loops round" in trip.summary()

    def test_a_negative_warmup_is_treated_as_none(self):
        trip = loopback.measure(lambda s: echo(s, 480, tail=0), SR, attempts=1,
                                warmup_seconds=-1.0)
        assert trip.measured_ms == pytest.approx(10.0, abs=0.05)


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
                                granted_ms=10.0, claimed_ms=5.0, engine_ms=60.0)
        assert trip.measured_ms == pytest.approx(100.0, abs=0.05)
        # Granted, not claimed: subtracting what PortAudio was asked for from
        # what it delivered is subtracting the request from the delivery.
        assert trip.unexplained_ms == pytest.approx(30.0, abs=0.05)

    def test_hearing_nothing_at_all_says_what_to_check(self):
        trip = loopback.measure(lambda sent: np.zeros(48000), SR, attempts=2)
        assert trip.attempts == 0
        assert np.isnan(trip.measured_ms)
        assert "loops round" in trip.summary()

    def test_the_summary_names_every_part_it_accounted_for(self):
        trip = loopback.measure(lambda sent: echo(sent, 4800), SR, attempts=3,
                                granted_ms=10.0, claimed_ms=5.0, engine_ms=60.0)
        text = trip.summary()
        assert "round trip" in text and "100.0 ms" in text
        assert "engine" in text and "buffers" in text and "unexplained" in text
        assert "spread" in text
        assert "granted" in text and "asked for 5.0" in text, \
            "both numbers, and which one was subtracted"
        assert "audio engine" in text, \
            "the remainder is not a closed list and PortAudio discards the "\
            "engine's own latency"

    def test_a_stream_that_will_not_say_falls_back_and_labels_it(self):
        trip = loopback.measure(lambda sent: echo(sent, 4800), SR, attempts=1,
                                granted_ms=0.0, claimed_ms=5.0, engine_ms=60.0)
        text = trip.summary()
        assert "catalog" in text and "would not say" in text
        # Nothing was granted, so nothing is subtracted for it.
        assert trip.unexplained_ms == pytest.approx(40.0, abs=0.05)


class CableStream:
    """A sd.Stream that behaves like a loopback cable.

    Drives the callback from its own thread with one cursor for both
    directions, exactly as PortAudio does, and returns each output sample
    ``delay`` samples later.  ``delay`` has to be at least one block: a device
    that answered sooner would be returning audio it had not been given yet.
    """

    def __init__(self, delay=1000, granted=(0.004, 0.005), **kwargs):
        import threading

        self.kwargs = kwargs
        self.delay = delay
        self.latency = granted
        self.block = int(kwargs.get("blocksize") or 256)
        assert self.delay >= self.block
        self.callback = kwargs["callback"]
        self._stop = threading.Event()
        self._thread = None
        self.closed = False

    def start(self):
        import threading

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        played = np.zeros(0)
        pos = 0
        while not self._stop.is_set():
            out = np.zeros((self.block, 1), dtype="float32")
            heard = np.zeros((self.block, 1), dtype="float32")
            lo = pos - self.delay
            if lo >= 0:
                take = played[lo:lo + self.block]
                heard[:take.size, 0] = take
            self.callback(heard, out, self.block, None, 0)
            played = np.concatenate([played, out[:, 0].astype(np.float64)])
            pos += self.block

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def close(self):
        self.closed = True


def cable(monkeypatch, delay=1000, granted=(0.004, 0.005), claimed=5.0):
    """Point through_devices at a CableStream and record what it was asked."""
    from natvox.app import backend as backend_module

    seen = {}

    class Fake:
        @staticmethod
        def Stream(**kwargs):                   # noqa: N802 - sounddevice's name
            seen.update(kwargs)
            seen["stream"] = CableStream(delay, granted, **kwargs)
            return seen["stream"]

    def claim(input_device, output_device, setting="low"):
        seen["setting"] = setting
        return claimed

    monkeypatch.setattr(backend_module, "_sounddevice", lambda: Fake)
    monkeypatch.setattr(backend_module, "rate_mismatch", lambda *a, **k: "")
    monkeypatch.setattr(backend_module, "reported_latency_ms", claim)
    return seen


class TestWhatItAsksFor:
    """`through_devices` opens the stream, so it decides what is measured.

    Left to sounddevice's default the stream is opened at `high` latency --
    the setting meant for robust playback, not conversation -- and the
    measurement then reports a path nobody would choose.
    """

    def test_it_asks_for_low_and_measures_the_real_delay(self, monkeypatch):
        seen = cable(monkeypatch, delay=1024)
        trip = loopback.through_devices(1, 2, attempts=2, block_size=256)
        assert seen["latency"] == "low"
        assert seen["setting"] == "low", \
            "the claim shown must describe the stream that was opened"
        assert trip.measured_ms == pytest.approx(1024 / SR * 1000.0, abs=0.1)

    def test_the_setting_is_carried_through_verbatim(self, monkeypatch):
        seen = cable(monkeypatch)
        loopback.through_devices(1, 2, attempts=1, latency="high")
        assert seen["latency"] == "high"
        assert seen["setting"] == "high"

    def test_it_subtracts_what_portaudio_granted_not_what_it_asked_for(
            self, monkeypatch):
        """The defect this replaced.  sounddevice resolves latency="low" by
        reading the same catalog key reported_latency_ms reads, so subtracting
        that was subtracting the request from the delivery."""
        seen = cable(monkeypatch, delay=4800, granted=(0.010, 0.012), claimed=5.0)
        trip = loopback.through_devices(1, 2, attempts=1)
        assert trip.granted_ms == pytest.approx(22.0)
        assert trip.claimed_ms == pytest.approx(5.0)
        assert trip.unexplained_ms == pytest.approx(100.0 - 22.0, abs=0.1)
        assert "granted" in trip.summary() and "asked for 5.0" in trip.summary()

    def test_a_stream_that_will_not_report_falls_back_rather_than_guessing(
            self, monkeypatch):
        seen = cable(monkeypatch, delay=4800, granted=None, claimed=5.0)
        trip = loopback.through_devices(1, 2, attempts=1)
        assert trip.granted_ms == 0.0
        assert "would not say" in trip.summary()

    def test_one_stream_serves_every_probe(self, monkeypatch):
        """playrec opens and closes a stream per attempt, so every probe is
        heard by a pipeline that has not settled -- and the stream is gone
        before anyone can ask what latency it was given."""
        seen = cable(monkeypatch, delay=1024)
        trip = loopback.through_devices(1, 2, attempts=4)
        assert trip.attempts == 4
        assert trip.spread_ms == pytest.approx(0.0, abs=0.05), \
            "one settled stream answers the same way every time"

    def test_the_stream_is_closed_even_when_a_probe_fails(self, monkeypatch):
        seen = cable(monkeypatch, delay=1024)
        monkeypatch.setattr(loopback, "estimate_delay",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
        with pytest.raises(RuntimeError):
            loopback.through_devices(1, 2, attempts=1)
        assert seen["stream"].closed, "a held device is worse than a bad number"


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
