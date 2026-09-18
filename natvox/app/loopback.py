"""Measuring the path the audio actually takes, end to end.

A virtual cable, the device buffers, the Windows audio engine and this
program's own delay all add up, and only the total is audible.  Every part of
it *reports* a number -- PortAudio has an opinion, the driver has an opinion --
and the reported numbers are estimates that leave out whatever sits between
them.  The way to know is to send a sound out and listen for it coming back.

That also settles the question this exists to settle.  Before replacing a
virtual cable with a better one it is worth knowing what the current one
costs, because the answer decides whether writing a kernel driver would buy
milliseconds or nothing at all.

The estimator is separate from anything that opens a device, so it can be
tested against a signal delayed by a known amount rather than against a guess
about what a sound card did.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: The probe is a short logarithmic sweep rather than a click.
#:
#: A click has all its energy in one sample, so it is either loud enough to be
#: unpleasant or too quiet to find.  A sweep spreads the same energy over 40 ms
#: and is then compressed back into a single sharp peak by the matched filter,
#: which is the whole point of using one: the peak is as narrow as a click's
#: would be, at a fraction of the level.  It also survives the band limits of
#: whatever it passes through, since a cable or a codec that removes part of
#: the band only widens the peak a little.
PROBE_SECONDS = 0.040
PROBE_LOW_HZ = 300.0
PROBE_HIGH_HZ = 6000.0

#: Peak level of the probe, -12 dB rather than full scale.
#:
#: Somebody running this is wearing headphones -- the README tells them to, so
#: that the output does not feed back into the microphone -- and a full-scale
#: sweep straight into them is unkind at best.  The matched filter finds an
#: echo 34 dB below this without complaining, so the level buys nothing.
PROBE_LEVEL = 0.25

#: How well the echo has to match the probe to be believed, as a normalised
#: correlation.
#:
#: Normalised, not a ratio against the surrounding correlation, and the
#: difference is not cosmetic: a plain peak-against-its-neighbours test called
#: white noise a 0.88-second delay, confidently, because the largest of fifty
#: thousand Gaussian lags does stand out from the median.  It stands out by
#: about sqrt(2 ln N), which depends on how long you listened rather than on
#: whether anything came back.
#:
#: The normalised version has no such dependence.  A clean echo scores near 1
#: whatever its level; noise scores about sqrt(2 ln N / L), which for fifty
#: thousand lags and a 1920-sample probe is 0.11.  0.3 sits between them with
#: room for a cable that has eaten part of the band.
MATCH_THRESHOLD = 0.3

#: Silence played before the probe, so the stream is not brand new when it goes.
#:
#: Without it the probe sits at frame 0 of the array handed to the device and
#: leaves in the stream's very first callback, microseconds after it opened.
#: Whether that matters is an empirical question -- a driver that settles
#: instantly gives the same answer either way -- so this is a knob rather than
#: an assumption: measure with ``--warmup 0`` and with the default, and the
#: difference is how much of the number was the stream still waking up.
WARMUP_SECONDS = 0.25

#: How far below the loudest moment of the recording the echo may be, in dB,
#: before it stops being believed.
#:
#: Being blind to level is what normalisation is for, and it is also how a
#: normalised score goes wrong: any nearly-silent stretch divides by nearly
#: nothing.  A bandpass in the path leaves a decaying ring after the echo, the
#: ring is narrowband, and the first third of a logarithmic sweep is narrowband
#: too -- so a ring 60 dB down scored 1.0 and moved the answer 49 ms late.
#:
#: Flooring the divisor at this much below the recording's loudest window fixes
#: it without reintroducing a level threshold: quiet content is suppressed in
#: proportion to how quiet it is (a window 60 dB down can score at most 0.03),
#: while an echo that is merely quiet in absolute terms is still the loudest
#: thing in its own recording and still scores 1.
QUIET_FLOOR_DB = -30.0


def probe(sample_rate: int, seconds: float = PROBE_SECONDS,
          low: float = PROBE_LOW_HZ, high: float = PROBE_HIGH_HZ,
          level: float = PROBE_LEVEL) -> np.ndarray:
    """A logarithmic sweep, faded at both ends so it starts and stops cleanly."""
    n = max(16, int(round(seconds * sample_rate)))
    # Re-derived from the clamped length rather than taken as asked for: the
    # phase below raises the sweep ratio to t/seconds, so a request shorter
    # than the clamp would put the exponent in the millions.
    seconds = n / sample_rate
    t = np.arange(n) / sample_rate
    ratio = high / low
    # Constant fractional bandwidth per unit time; the instantaneous frequency
    # goes from `low` to `high` geometrically.
    phase = 2.0 * np.pi * low * seconds / np.log(ratio) * (ratio ** (t / seconds) - 1.0)
    sweep = np.sin(phase)
    fade = max(8, n // 16)
    window = np.ones(n)
    window[:fade] = np.linspace(0.0, 1.0, fade)
    window[-fade:] = np.linspace(1.0, 0.0, fade)
    return sweep * window * level


def estimate_delay(recording: np.ndarray, sent: np.ndarray,
                   sample_rate: int) -> float | None:
    """Delay in seconds between ``sent`` and its echo in ``recording``.

    Matched filter, then a parabola through the peak and its neighbours, which
    puts the answer between samples: at 48 kHz a whole sample is 0.02 ms and
    the quantities being compared here differ by fractions of a millisecond.

    ``None`` when nothing recognisable came back -- an unconnected cable, a
    muted device, the wrong device -- because a confident wrong number is worse
    than no number.
    """
    recording = np.asarray(recording, dtype=np.float64).reshape(-1)
    sent = np.asarray(sent, dtype=np.float64).reshape(-1)
    length = sent.size
    if recording.size < length + 2 or length < 8:
        return None

    raw = np.correlate(recording, sent, mode="valid")
    # Energy of the recording under each position of the probe, so the score
    # is "how much does this look like the probe" rather than "how loud is it"
    # -- floored, so that "nearly nothing here" cannot look like anything.
    cumulative = np.concatenate(([0.0], np.cumsum(recording * recording)))
    window = cumulative[length:] - cumulative[:-length]
    floor = float(np.max(window)) * 10.0 ** (QUIET_FLOOR_DB / 10.0)
    scale = np.sqrt(np.maximum(np.maximum(window, floor)
                               * float(np.dot(sent, sent)), 1e-30))
    match = np.abs(raw) / scale
    if match.size < 3:
        return None

    peak = int(np.argmax(match))
    if float(match[peak]) < MATCH_THRESHOLD:
        return None

    offset = 0.0
    if 0 < peak < match.size - 1:
        before, here, after = match[peak - 1], match[peak], match[peak + 1]
        denominator = before - 2.0 * here + after
        if denominator != 0.0:
            offset = float(np.clip(0.5 * (before - after) / denominator, -0.5, 0.5))
    return (peak + offset) / sample_rate


@dataclass
class RoundTrip:
    """What a device pair actually costs, measured rather than reported."""

    sample_rate: int
    block_size: int
    delays_ms: list[float]
    #: What PortAudio *granted*, read off the open stream.
    #:
    #: Not what it was asked for, and the difference is the whole point.
    #: sounddevice resolves ``latency="low"`` by reading the device catalog's
    #: ``default_low_*_latency`` and passing it as ``suggestedLatency``; the
    #: same key is what :func:`natvox.app.backend.reported_latency_ms` reads.
    #: Subtracting that from a measurement is subtracting the request from the
    #: delivery, and PortAudio's own documentation says the two "may differ
    #: significantly".  ``Pa_GetStreamInfo`` has the real figure and this is
    #: it.
    granted_ms: float = 0.0
    #: What the device catalog claimed, i.e. what was asked for.  Kept only so
    #: the summary can show both and say which one was subtracted.
    claimed_ms: float = 0.0
    engine_ms: float = 0.0
    #: Anything found while setting the measurement up that changes how to
    #: read it -- a device running at a rate nobody asked for, say.  Part of
    #: the result rather than a separate return, because a measurement whose
    #: caveat arrives separately gets quoted without it.
    note: str = ""

    @property
    def measured_ms(self) -> float:
        return float(np.median(self.delays_ms)) if self.delays_ms else float("nan")

    @property
    def spread_ms(self) -> float:
        """Difference between the slowest and fastest attempt.

        A stable path gives the same answer every time.  A spread of more than
        a block or two means something is resampling or rebuffering, and the
        median alone would hide it.
        """
        return (float(np.max(self.delays_ms) - np.min(self.delays_ms))
                if len(self.delays_ms) > 1 else 0.0)

    @property
    def unexplained_ms(self) -> float:
        """Measured, minus what this program and PortAudio account for.

        What is left is everything neither of them can see.  That is not a
        closed list and it is not all the cable: PortAudio's WASAPI backend
        fetches the audio engine's own latency and then throws it away --
        ``pa_win_wasapi.c`` reads ``IAudioClient::GetStreamLatency`` and the
        line that would add it is commented out -- so the engine's path is
        inside this remainder on every Windows machine, cable or no cable.
        """
        return self.measured_ms - self.granted_ms - self.engine_ms

    @property
    def granted_from(self) -> str:
        return "PortAudio granted" if self.granted_ms else "claimed"

    @property
    def attempts(self) -> int:
        return len(self.delays_ms)

    def summary(self) -> str:
        if not self.delays_ms:
            return "\n".join(filter(None, [
                "nothing came back -- check that the output device really "
                "loops round to the input device, and that neither is muted",
                self.note,
            ]))
        lines = [
            f"round trip   {self.measured_ms:.1f} ms"
            + (f" (spread {self.spread_ms:.1f} ms over {self.attempts} tries)"
               if self.attempts > 1 else ""),
        ]
        if self.engine_ms:
            lines.append(f"  engine     {self.engine_ms:.1f} ms")
        if self.granted_ms:
            asked = (f"; it was asked for {self.claimed_ms:.1f}"
                     if self.claimed_ms else "")
            lines.append(f"  buffers    {self.granted_ms:.1f} ms "
                         f"(PortAudio granted{asked})")
        elif self.claimed_ms:
            lines.append(f"  buffers    {self.claimed_ms:.1f} ms (the device "
                         f"catalog's claim; the stream would not say)")
        if self.engine_ms or self.granted_ms or self.claimed_ms:
            lines.append(f"  unexplained {self.unexplained_ms:.1f} ms -- what neither "
                         f"this program nor\n              PortAudio accounts for: "
                         f"the Windows audio engine (PortAudio\n              fetches "
                         f"its latency and discards it), a virtual cable, a resampler")
        if self.note:
            lines.append(self.note)
        return "\n".join(lines)


def measure(play_and_record, sample_rate: int, block_size: int = 256,
            attempts: int = 5, tail_seconds: float = 0.5,
            granted_ms: float = 0.0, claimed_ms: float = 0.0,
            engine_ms: float = 0.0,
            note: str = "", warmup_seconds: float = WARMUP_SECONDS) -> RoundTrip:
    """Send the probe ``attempts`` times and time each echo.

    ``play_and_record(signal) -> recording`` does whatever opens the devices,
    so that everything above it can be tested without any.

    The probe goes out after ``warmup_seconds`` of silence and that offset is
    subtracted again, so the answer means the same thing whatever the warm-up
    is -- which is what makes comparing two warm-ups a measurement rather than
    a change of units.
    """
    sent = probe(sample_rate)
    lead = max(0, int(max(0.0, warmup_seconds) * sample_rate))
    padded = np.concatenate([np.zeros(lead), sent,
                             np.zeros(int(tail_seconds * sample_rate))])
    delays = []
    for _ in range(max(1, attempts)):
        recording = np.asarray(play_and_record(padded), dtype=np.float64).reshape(-1)
        delay = estimate_delay(recording, sent, sample_rate)
        if delay is None:
            continue
        delay -= lead / sample_rate
        # An echo found before the probe was sent is not an echo.  Dropping it
        # costs an attempt and says so in the count; keeping it would put a
        # negative round trip into the median.
        if delay >= 0.0:
            delays.append(delay * 1000.0)
    return RoundTrip(sample_rate, block_size, delays, granted_ms, claimed_ms,
                     engine_ms, note)


def through_devices(input_device=None, output_device=None, sample_rate: int = 48000,
                    block_size: int = 256, attempts: int = 5, exclusive: bool = False,
                    engine_ms: float = 0.0, latency="low",
                    warmup_seconds: float = WARMUP_SECONDS) -> RoundTrip:
    """Measure a real device pair.  Loop the output back to the input first.

    With a virtual cable that means selecting the cable's playback end as the
    output and its recording end as the input, which is exactly the path the
    voice takes on its way to another program.

    One stream, opened once and held across every probe.  ``sounddevice.playrec``
    would have been shorter and opens and closes a fresh stream per call, which
    costs two things: every probe is then heard by a pipeline that has not
    settled, and the stream is gone before anyone can ask what latency it was
    actually given.  That second one matters more -- it is the difference
    between subtracting what PortAudio delivered and subtracting what it was
    asked for.

    ``latency`` is passed to PortAudio and is not optional in practice: left
    out, sounddevice asks for ``"high"``, and the measurement then reports a
    path nobody would choose to use.
    """
    import time

    from .backend import (AudioUnavailable, _sounddevice, exclusive_settings,
                          rate_mismatch, reported_latency_ms)

    sd = _sounddevice()
    settings = (exclusive_settings(input_device, output_device)
                if exclusive else None)

    # Written by the main thread between probes, read by the audio thread
    # during one.  Never both at once: the main thread parks `data` at None
    # (which makes the callback output silence and touch nothing else), sets
    # the rest up, and only then hands the signal over.
    state = {"data": None, "out": None, "frame": 0, "warnings": 0}

    def callback(indata, outdata, frames, time_info, status):
        if status:
            state["warnings"] += 1
        data = state["data"]
        if data is None:
            outdata.fill(0.0)
            return
        out, i = state["out"], state["frame"]
        n = max(0, min(frames, data.size - i))
        if n:
            # One cursor for both directions, so outdata[k] and out[k] are the
            # same callback and the delay recovered later is a real loop delay
            # rather than an artefact of when recording started.
            outdata[:n, 0] = data[i:i + n]
            out[i:i + n] = indata[:n, 0]
        if n < frames:
            outdata[n:].fill(0.0)
        state["frame"] = i + frames

    try:
        stream = sd.Stream(
            samplerate=sample_rate, blocksize=block_size, channels=(1, 1),
            device=(input_device, output_device), dtype="float32",
            extra_settings=settings, latency=latency, callback=callback,
        )
        stream.start()
    except AudioUnavailable:
        raise
    except Exception as exc:                    # noqa: BLE001 - as LiveBackend
        # PortAudio reports a missing device, a rate the pair cannot agree on
        # and a device already held by something else as the same exception
        # type with a different string.  The string is the useful part.
        raise AudioUnavailable(f"could not open those devices: {exc}") from exc

    try:
        granted = _granted_ms(stream)

        def play_and_record(signal: np.ndarray) -> np.ndarray:
            state["data"] = None                # the callback goes quiet
            state["out"] = np.zeros(signal.size)
            state["frame"] = 0
            state["data"] = np.ascontiguousarray(signal, dtype=np.float64)
            deadline = time.monotonic() + 3.0 * signal.size / sample_rate + 2.0
            while state["frame"] < signal.size:
                if time.monotonic() > deadline:
                    raise AudioUnavailable(
                        "the stream stopped feeding us; the device was "
                        "probably taken away mid-measurement")
                time.sleep(0.005)
            state["data"] = None
            return state["out"].copy()

        return measure(play_and_record, sample_rate, block_size, attempts,
                       granted_ms=granted,
                       claimed_ms=reported_latency_ms(input_device, output_device,
                                                      setting=latency),
                       engine_ms=engine_ms, warmup_seconds=warmup_seconds,
                       note=rate_mismatch(input_device, output_device, sample_rate))
    finally:
        try:
            stream.stop()
        finally:
            stream.close()


def _granted_ms(stream) -> float:
    """What PortAudio says it actually gave, in and out, or 0.0.

    Zero rather than a guess when it will not say, because the summary can
    then fall back to the catalog's claim and *label* it as a fallback.  A
    silent substitution here is how the request came to be printed as though
    it were the delivery in the first place.
    """
    try:
        return 1000.0 * float(sum(stream.latency))
    except (TypeError, ValueError, AttributeError):
        return 0.0
