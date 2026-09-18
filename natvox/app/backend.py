"""Where the audio comes from, and where it goes.

The callback core in :mod:`natvox.realtime` has no audio library inside it, and
that is what makes this program testable without a sound card: the same
processor a real duplex stream drives can be driven from an array instead,
either as fast as the machine will go or paced against a clock.  The live path
and the test path are then the same code with a different clock, rather than
the real thing and a mock of it.

PortAudio is imported lazily and its absence is reported as a sentence rather
than an ImportError, because "no audio backend installed" is a thing a user
can fix and a stack trace is not.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from ..realtime import RealtimeSession, StreamProcessor


class AudioUnavailable(RuntimeError):
    """No usable audio backend, with an explanation of what to install."""


@dataclass(frozen=True)
class Device:
    """One audio device, as the picker needs it."""

    index: int
    name: str
    host_api: str
    inputs: int
    outputs: int
    default_sample_rate: float

    @property
    def label(self) -> str:
        return f"{self.name} ({self.host_api})"


def _sounddevice():
    try:
        import sounddevice as sd
    except OSError as exc:                      # PortAudio itself is missing
        raise AudioUnavailable(
            "the PortAudio library is not installed, so no audio device can be "
            "opened. Linux: apt install libportaudio2. macOS: brew install "
            "portaudio. Windows: it ships with the sounddevice wheel, so "
            "reinstall it."
        ) from exc
    except ImportError as exc:
        raise AudioUnavailable(
            "the sounddevice package is not installed; pip install "
            "'natvox[app]'"
        ) from exc
    return sd


def list_devices() -> list[Device]:
    """Every device PortAudio can see."""
    sd = _sounddevice()
    out = []
    apis = sd.query_hostapis()
    for index, info in enumerate(sd.query_devices()):
        api = apis[info["hostapi"]]["name"] if info["hostapi"] < len(apis) else "?"
        out.append(Device(index, info["name"], api,
                          int(info["max_input_channels"]),
                          int(info["max_output_channels"]),
                          float(info["default_samplerate"])))
    return out


def default_devices() -> tuple[int | None, int | None]:
    sd = _sounddevice()
    try:
        inp, outp = sd.default.device
    except Exception:                           # noqa: BLE001 - reported as none
        return None, None
    to_index = lambda v: int(v) if isinstance(v, (int, np.integer)) and v >= 0 else None
    return to_index(inp), to_index(outp)


class LiveBackend:
    """A real duplex stream, opened on the chosen devices."""

    kind = "live"

    def __init__(self, input_device=None, output_device=None,
                 block_size: int = 256) -> None:
        self.input_device = input_device
        self.output_device = output_device
        self.block_size = int(block_size)
        self._session: RealtimeSession | None = None

    @property
    def running(self) -> bool:
        return self._session is not None

    def start(self, processor: StreamProcessor) -> None:
        _sounddevice()                          # fail with a sentence, early
        session = RealtimeSession(processor, self.input_device,
                                  self.output_device, self.block_size)
        try:
            session.start()
        except Exception as exc:                # noqa: BLE001 - see below
            # PortAudio reports a wrong sample rate, a device in use, and a
            # device that has been unplugged all as the same exception type
            # with a different string.  The string is the useful part.
            raise AudioUnavailable(f"could not open the audio stream: {exc}") from exc
        self._session = session

    def stop(self) -> None:
        if self._session is not None:
            self._session.stop()
            self._session = None

    @property
    def device_latency_ms(self) -> float:
        """What the device buffers add, in and out."""
        return 2000.0 * self.block_size / 48000.0


class OfflineBackend:
    """Drive the processor from an array instead of a microphone.

    Two speeds, and both are honest uses rather than test scaffolding:

    ``realtime=False`` runs as fast as the machine can, which is how a file is
    converted and how the self-test measures whether this machine can keep up.

    ``realtime=True`` paces against a clock, so the meters move and a dropout
    means what it means.  A test can then exercise the whole program -- start
    it, watch the meters, move a slider, stop it -- with no hardware anywhere.
    """

    kind = "offline"

    def __init__(self, source: np.ndarray, block_size: int = 256,
                 realtime: bool = False, loop: bool = False) -> None:
        self.source = np.asarray(source, dtype=np.float64).reshape(-1)
        self.block_size = int(block_size)
        self.realtime = bool(realtime)
        self.loop = bool(loop)
        self.captured: list[np.ndarray] = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._done = threading.Event()
        self.error: Exception | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def finished(self) -> bool:
        return self._done.is_set()

    @property
    def device_latency_ms(self) -> float:
        return 0.0

    def start(self, processor: StreamProcessor) -> None:
        self._stop.clear()
        self._done.clear()
        self.captured = []
        self.error = None
        self._thread = threading.Thread(target=self._run, args=(processor,),
                                        daemon=True)
        self._thread.start()

    def _run(self, processor: StreamProcessor) -> None:
        try:
            rate = processor.sample_rate
            block = self.block_size
            started = time.perf_counter()
            emitted = 0
            position = 0
            while not self._stop.is_set():
                if position >= self.source.size:
                    if not self.loop:
                        break
                    position = 0
                chunk = self.source[position:position + block]
                position += chunk.size
                if chunk.size == 0:
                    break
                out = processor(chunk, chunk.size)
                self.captured.append(np.asarray(out, dtype=np.float32).copy())
                emitted += chunk.size
                if self.realtime:
                    due = started + emitted / rate
                    remaining = due - time.perf_counter()
                    if remaining > 0:
                        self._stop.wait(remaining)
        except Exception as exc:                # noqa: BLE001 - surfaced below
            self.error = exc
        finally:
            self._done.set()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def wait(self, timeout: float = 30.0) -> bool:
        return self._done.wait(timeout)

    def output(self) -> np.ndarray:
        """Everything produced so far, mono."""
        if not self.captured:
            return np.zeros(0)
        stacked = np.concatenate(self.captured)
        if stacked.ndim == 2:
            stacked = stacked[:, 0]
        return stacked.astype(np.float64)
