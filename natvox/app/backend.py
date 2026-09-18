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


#: PortAudio's name for the modern Windows API, spelled exactly once.
WASAPI = "Windows WASAPI"

#: How much delay each host API tends to add, lowest first.
#:
#: This matters more than any other setting in the program and is the one most
#: likely to be wrong by default.  PortAudio offers the same physical device
#: once per host API it can reach, and on Windows it picks MME unless told
#: otherwise -- an interface from 1991 that goes through the system mixer and
#: several buffers on the way.  WASAPI talks to the audio engine directly, and
#: in exclusive mode it bypasses the mixer as well.
#:
#: The engine's own delay is about 60 ms.  The host API can add more than that
#: without anyone choosing it, which is why the picker sorts on this and why
#: `natvox-cli --devices` prints what each one claims.
HOST_API_RANK = {
    "ASIO": 0,                      # Windows, needs a driver from the vendor
    "Core Audio": 0,                # macOS
    "JACK Audio Connection Kit": 0,
    "ALSA": 1,
    WASAPI: 1,
    "Windows WDM-KS": 1,            # kernel streaming: low, but exclusive only
    "Windows DirectSound": 3,
    "OSS": 3,
    "MME": 4,
}

#: Used for anything not in the table: neither preferred nor penalised.
UNKNOWN_RANK = 2


@dataclass(frozen=True)
class Device:
    """One audio device, as the picker needs it."""

    index: int
    name: str
    host_api: str
    inputs: int
    outputs: int
    default_sample_rate: float
    #: What the driver claims its buffers cost, in milliseconds.  An estimate,
    #: and usually an optimistic one -- it leaves out whatever sits between the
    #: driver and this program.  `natvox.app.loopback` measures the truth.
    low_input_ms: float = 0.0
    low_output_ms: float = 0.0

    @property
    def rank(self) -> int:
        return HOST_API_RANK.get(self.host_api, UNKNOWN_RANK)

    @property
    def claimed_ms(self) -> float:
        """Whichever direction this device is usable in."""
        return self.low_input_ms if self.inputs else self.low_output_ms

    @property
    def label(self) -> str:
        return f"{self.name} ({self.host_api})"

    @property
    def detail(self) -> str:
        return f"{self.label} -- claims {self.claimed_ms:.1f} ms"


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


def list_devices(sort: bool = True) -> list[Device]:
    """Every device PortAudio can see, lowest-latency host API first.

    Sorted because the same microphone appears several times -- once per host
    API -- and which copy is picked decides more of the delay than anything
    else the program does.  A picker that lists them in PortAudio's order
    reliably puts the worst one first.
    """
    sd = _sounddevice()
    out = []
    apis = sd.query_hostapis()
    for index, info in enumerate(sd.query_devices()):
        api = apis[info["hostapi"]]["name"] if info["hostapi"] < len(apis) else "?"
        out.append(Device(
            index, info["name"], api,
            int(info["max_input_channels"]), int(info["max_output_channels"]),
            float(info["default_samplerate"]),
            1000.0 * float(info.get("default_low_input_latency") or 0.0),
            1000.0 * float(info.get("default_low_output_latency") or 0.0),
        ))
    if sort:
        out.sort(key=lambda d: (d.rank, d.claimed_ms, d.name))
    return out


def host_api_of(device) -> str:
    """Host API name for a device index, or '' when it cannot be resolved.

    Every failure is the same answer.  PortAudio has its own exception type
    for a device index that is out of range, so a list of the exceptions worth
    catching here is a list that will be wrong on somebody else's machine --
    and the only caller that matters, :func:`exclusive_settings`, treats "I do
    not know what this is" and "this is not WASAPI" identically anyway.
    """
    if device is None:
        return ""
    try:
        sd = _sounddevice()
        info = sd.query_devices(device)
        apis = sd.query_hostapis()
        index = int(info["hostapi"])
        return apis[index]["name"] if index < len(apis) else ""
    except Exception:                           # noqa: BLE001 - see above
        return ""


def exclusive_settings(input_device=None, output_device=None):
    """WASAPI exclusive-mode settings for a device pair.

    Exclusive mode hands the device to this program alone and skips the
    Windows audio engine's mixer, which is where a good part of the delay on
    Windows lives -- and, on a shared device, a sample-rate conversion nobody
    asked for.  The cost is that nothing else can use that device while this
    runs, which is why it is a choice rather than the default.

    Raises rather than quietly falling back, because a request for exclusive
    mode that is silently ignored looks exactly like a measurement saying
    exclusive mode does not help.
    """
    sd = _sounddevice()
    names = {host_api_of(input_device), host_api_of(output_device)}
    names.discard("")
    if not names or not names <= {WASAPI}:
        raise AudioUnavailable(
            "exclusive mode is a WASAPI feature; pick the WASAPI copy "
            "of both devices, or turn it off"
        )
    return sd.WasapiSettings(exclusive=True)


def rate_mismatch(input_device=None, output_device=None, rate: int = 48000) -> str:
    """A sentence naming any device whose own rate is not ``rate``, or ''.

    A device running at a different rate than the stream does not refuse: the
    audio engine quietly inserts a resampler, which costs delay and a little
    quality, and nothing anywhere says it happened.  It is the most common
    reason a virtual cable measures worse than it should -- cables usually
    default to 44100 while everything else here is at 48000 -- and it is fixed
    in the device's own properties in about ten seconds, which makes it worth
    far more than it costs to detect.
    """
    try:
        sd = _sounddevice()
    except AudioUnavailable:
        return ""
    wrong = []
    for device, kind in ((input_device, "input"), (output_device, "output")):
        try:
            info = sd.query_devices(device, kind)
            native = int(round(float(info["default_samplerate"])))
        except Exception:                       # noqa: BLE001 - best effort
            continue
        if native != int(rate):
            wrong.append(f"the {kind} device is set to {native} Hz")
    if not wrong:
        return ""
    return (f"{' and '.join(wrong)}, not {int(rate)} Hz -- Windows will be "
            f"resampling, which costs delay and quality and says nothing. "
            f"Match them in the device's properties, or pass --rate.")


def reported_latency_ms(input_device=None, output_device=None) -> float:
    """What the two drivers claim their buffers cost, together.

    Best effort on purpose: this number exists to be subtracted from a measured
    round trip, and a device that will not answer should leave the remainder
    unexplained rather than stop the measurement.
    """
    try:
        sd = _sounddevice()
    except AudioUnavailable:
        return 0.0
    total = 0.0
    for device, kind in ((input_device, "input"), (output_device, "output")):
        try:
            info = sd.query_devices(device, kind)
            total += 1000.0 * float(info.get(f"default_low_{kind}_latency") or 0.0)
        except Exception:                        # noqa: BLE001 - best effort
            continue
    return total


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
                 block_size: int = 256, exclusive: bool = False) -> None:
        self.input_device = input_device
        self.output_device = output_device
        self.block_size = int(block_size)
        #: See :func:`exclusive_settings`.
        self.exclusive = bool(exclusive)
        self._session: RealtimeSession | None = None

    @property
    def running(self) -> bool:
        return self._session is not None

    def start(self, processor: StreamProcessor) -> None:
        _sounddevice()                          # fail with a sentence, early
        extra = (exclusive_settings(self.input_device, self.output_device)
                 if self.exclusive else None)
        session = RealtimeSession(processor, self.input_device,
                                  self.output_device, self.block_size,
                                  extra_settings=extra)
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

    @property
    def claimed_latency_ms(self) -> float:
        """What the drivers themselves claim, which is usually less."""
        return reported_latency_ms(self.input_device, self.output_device)


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
