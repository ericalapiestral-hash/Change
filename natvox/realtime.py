"""Live microphone -> speaker (or virtual cable) operation.

The audio callback is the one place where being slow is not a quality issue
but a correctness one: overrun the deadline and the stream drops samples,
which is heard as a click regardless of how good the DSP is.  So the callback
itself is kept trivial -- convert, copy out, record timing -- and everything
that can be decided in advance is decided in the constructor.

:class:`StreamProcessor` holds that logic and has no dependency on any audio
library, so it can be exercised (and its deadline behaviour measured) without
a sound card.  :class:`RealtimeSession` is the thin PortAudio binding on top.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .dsp.util import FixedDelay

#: How long the A/B takes to cross from one path to the other.  Long enough
#: that the switch itself is not a click, short enough that it does not blur
#: the comparison it exists to make.
AB_FADE_SECONDS = 0.012


@dataclass
class StreamStats:
    """What the callback observed, for reporting load and dropouts."""

    blocks: int = 0
    frames: int = 0
    overruns: int = 0          # blocks that missed the real-time deadline
    device_warnings: int = 0   # under/overflows reported by the audio backend
    peak_load: float = 0.0     # worst block's processing time / its duration
    mean_load: float = 0.0
    _load_total: float = field(default=0.0, repr=False)

    def record(self, frames: int, elapsed: float, sample_rate: int) -> None:
        budget = frames / sample_rate
        load = elapsed / budget if budget > 0 else 0.0
        self.blocks += 1
        self.frames += frames
        self._load_total += load
        self.peak_load = max(self.peak_load, load)
        self.mean_load = self._load_total / self.blocks
        if load >= 1.0:
            self.overruns += 1

    def summary(self) -> str:
        seconds = self.frames / max(self.blocks, 1)
        return (
            f"{self.blocks} blocks, {self.frames} frames | "
            f"CPU load mean {self.mean_load:.1%} peak {self.peak_load:.1%} | "
            f"deadline misses {self.overruns} | device warnings {self.device_warnings}"
        )


class StreamProcessor:
    """Device-independent core of the real-time callback.

    Parameters
    ----------
    converter:
        Anything satisfying :class:`natvox.neural.base.VoiceConverter` -- the
        DSP engine, a neural converter, or a pipeline of both.
    channels:
        Output channel count.  Input is mixed to mono first: voice conversion
        is a mono problem, and converting channels separately would let them
        drift apart in pitch.
    """

    def __init__(self, converter, channels: int = 1, dry_wet: float = 1.0,
                 fade_seconds: float = AB_FADE_SECONDS) -> None:
        self.converter = converter
        self.sample_rate = converter.sample_rate
        self.channels = max(1, int(channels))
        self.stats = StreamStats()
        # The delay line is fed on every block, not only while the original is
        # being listened to.  Filling it on demand meant the first
        # `latency_samples` of every A/B were the silence it had been holding
        # -- 60 ms of nothing at the exact moment someone is trying to hear a
        # difference, and the comparison the whole engine is judged by.
        self._dry = FixedDelay(converter.latency_samples)
        self._dry_out = np.zeros(4096)
        self._mix = self._target = float(np.clip(dry_wet, 0.0, 1.0))
        self._step = 1.0 / max(1.0, fade_seconds * self.sample_rate)

    @property
    def latency_ms(self) -> float:
        return 1000.0 * self.converter.latency_samples / self.sample_rate

    @property
    def dry_wet(self) -> float:
        """1.0 is fully converted, 0.0 the delay-matched original."""
        return self._target

    @dry_wet.setter
    def dry_wet(self, value: float) -> None:
        self._target = float(np.clip(value, 0.0, 1.0))

    def __call__(self, indata: np.ndarray, frames: int) -> np.ndarray:
        """Process one device block; returns ``(frames, channels)``."""
        started = time.perf_counter()
        mono = np.asarray(indata if indata.ndim == 1 else np.mean(indata, axis=1),
                          dtype=np.float64)
        wet = self.converter.process(mono)
        n = wet.size
        if self._dry_out.size < n:
            self._dry_out = np.zeros(n)
        # Delayed to match, or the two copies comb-filter against each other
        # and the original sounds worse than the processed path for reasons
        # that have nothing to do with the processing.
        dry = self._dry.process(mono, self._dry_out)[:n]
        wet = self._blend(wet, dry, n)
        self.stats.record(frames, time.perf_counter() - started, self.sample_rate)
        out = wet.astype(np.float32, copy=False)
        return np.repeat(out[:, None], self.channels, axis=1)

    def _blend(self, wet: np.ndarray, dry: np.ndarray, n: int) -> np.ndarray:
        """Cross-fade toward the target mix.  Switching outright is a click,
        and a click is exactly what an A/B must not add."""
        mix, target = self._mix, self._target
        if mix == target:
            if target >= 1.0:
                return wet
            return target * wet + (1.0 - target) * dry
        direction = 1.0 if target > mix else -1.0
        ramp = mix + direction * self._step * np.arange(1, n + 1)
        np.clip(ramp, min(mix, target), max(mix, target), out=ramp)
        self._mix = float(ramp[-1])
        return ramp * wet + (1.0 - ramp) * dry

    def reset(self) -> None:
        self.converter.reset()
        self.stats = StreamStats()
        self._dry.reset()
        self._mix = self._target


def list_devices():
    """Available audio devices, or a clear error if PortAudio is missing."""
    import sounddevice as sd
    return sd.query_devices()


class RealtimeSession:
    """Open a duplex stream and run a :class:`StreamProcessor` on it.

    ``block_size`` is the device buffer.  It adds to the engine's own delay,
    so smaller is better for responsiveness and worse for dropout margin; 256
    at 48 kHz (5.3 ms) is a good starting point on a machine that is not busy.
    """

    def __init__(self, processor: StreamProcessor, input_device=None,
                 output_device=None, block_size: int = 256,
                 extra_settings=None, latency="low") -> None:
        self.processor = processor
        self.input_device = input_device
        self.output_device = output_device
        self.block_size = int(block_size)
        #: Host-API specific settings, e.g. ``sounddevice.WasapiSettings``.
        self.extra_settings = extra_settings
        #: Latency to ask PortAudio for: ``"low"``, ``"high"``, or seconds.
        #:
        #: Passed explicitly because sounddevice's own default is ``"high"``
        #: (``sounddevice._default_latency = 'high', 'high'``), which resolves
        #: to the device's ``default_high_*_latency`` -- the figure meant for
        #: robust non-interactive playback, not for talking to somebody.  A
        #: stream opened without this argument is opened at that setting, and
        #: nothing anywhere says so: every latency number this package
        #: reported came from ``default_low_*_latency`` instead, describing a
        #: stream that had never been opened.
        self.latency = latency
        self._stream = None

    @property
    def device_latency_ms(self) -> float:
        """What the device buffers add, in and out.

        Taken from the open stream, because PortAudio treats the requested
        latency as a suggestion and is free to give something else.  Before
        the stream exists there is nothing to read, so the block size stands
        in -- an estimate, and labelled as one by being replaced the moment
        there is a measurement.
        """
        stream = self._stream
        if stream is not None:
            try:
                return 1000.0 * float(sum(stream.latency))
            except (TypeError, ValueError):     # not a pair; PortAudio is odd
                pass
        return 2000.0 * self.block_size / self.processor.sample_rate

    @property
    def total_latency_ms(self) -> float:
        """Engine delay plus the device buffer, which is what a user hears."""
        return self.processor.latency_ms + self.device_latency_ms

    def _callback(self, indata, outdata, frames, time_info, status):
        if status:
            self.processor.stats.device_warnings += 1
        outdata[:] = self.processor(np.asarray(indata), frames)

    def start(self) -> None:
        import sounddevice as sd

        self._stream = sd.Stream(
            samplerate=self.processor.sample_rate,
            blocksize=self.block_size,
            dtype="float32",
            channels=(1, self.processor.channels),
            device=(self.input_device, self.output_device),
            extra_settings=self.extra_settings,
            latency=self.latency,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False
