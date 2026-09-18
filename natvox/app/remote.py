"""Running the conversion on another machine, and what that costs.

This exists because "can the work happen somewhere other than my computer" is
a reasonable question with an unreasonable answer, and the useful thing is to
make the answer measurable instead of arguing about it.

The arithmetic is not subtle.  The engine's delay is already 60 ms and the
device buffers add 10; moving the conversion to another machine adds a network
round trip *and* a buffer deep enough to absorb the variation in that round
trip, because audio that arrives late is not audio.  A buffer is the price of
jitter, not of distance, so it is the jitter that decides it -- and a link with
a 5 ms round trip and 10 ms of jitter is worse than one with 20 ms and none.

:func:`probe` measures both on a real link and reports what they would add.
:class:`RemoteConverter` then does the thing, with the network on its own
thread and a declared buffer, so that the audio callback never waits on a
socket and the delay it announces is the delay it has.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

from .wsclient import WebSocketClient, WebSocketError

#: Multiple of the measured jitter to hold, before rounding up to whole blocks.
#: Three standard deviations of a one-sided delay distribution covers about
#: 99% of packets; the other 1% is a click.
JITTER_MULTIPLE = 3.0

#: Never buffer less than this, whatever the link measures.  A probe is a
#: sample of a few seconds and a network is not stationary.
MIN_BUFFER_MS = 10.0


@dataclass
class LinkReport:
    """What a link measured, and what it would cost to use it."""

    url: str
    blocks: int
    block_ms: float
    median_ms: float
    p95_ms: float
    worst_ms: float
    jitter_ms: float
    engine_latency_ms: float

    @property
    def buffer_ms(self) -> float:
        """Jitter buffer this link needs, rounded up to whole blocks."""
        want = max(MIN_BUFFER_MS, JITTER_MULTIPLE * self.jitter_ms)
        blocks = int(np.ceil(want / self.block_ms)) if self.block_ms else 1
        return blocks * self.block_ms

    @property
    def total_ms(self) -> float:
        """Mouth to ear, if the conversion ran over this link."""
        return self.engine_latency_ms + self.median_ms + self.buffer_ms

    @property
    def verdict(self) -> str:
        total = self.total_ms
        if total < 120:
            return "usable for conversation"
        if total < 250:
            return "noticeable in conversation; fine one-way"
        return "one-way only (streaming, or converting files)"

    def summary(self) -> str:
        return (
            f"{self.url}\n"
            f"  round trip   median {self.median_ms:.1f} ms, p95 {self.p95_ms:.1f} ms, "
            f"worst {self.worst_ms:.1f} ms\n"
            f"  jitter       {self.jitter_ms:.1f} ms -> needs {self.buffer_ms:.1f} ms of buffer\n"
            f"  engine       {self.engine_latency_ms:.1f} ms\n"
            f"  total        {self.total_ms:.1f} ms -- {self.verdict}"
        )


def probe(url: str, block_size: int = 256, sample_rate: int = 48000,
          blocks: int = 200, warmup: int = 20) -> LinkReport:
    """Measure a link by sending it silence and timing the answers.

    Silence rather than a signal on purpose: this is measuring the link, and a
    server that spent longer on louder audio would make the number depend on
    what was said into it.
    """
    with WebSocketClient(url) as client:
        message = client.receive()
        if message is None:
            raise WebSocketError("server closed the stream immediately")
        opcode, payload = message
        hello = json.loads(payload) if opcode == 0x1 else {}
        engine_ms = float(hello.get("latency_ms") or 0.0)
        silence = np.zeros(block_size, dtype="<f4").tobytes()
        times = []
        for index in range(warmup + blocks):
            started = time.perf_counter()
            client.send_binary(silence)
            answer = client.receive()
            if answer is None:
                raise WebSocketError("server closed the stream during the probe")
            if index >= warmup:
                times.append((time.perf_counter() - started) * 1000.0)
    spent = np.asarray(times)
    median = float(np.median(spent))
    # Deviation above the median rather than the standard deviation: what a
    # buffer has to absorb is lateness, and a packet that arrives early costs
    # nothing.
    late = np.maximum(spent - median, 0.0)
    return LinkReport(
        url=url, blocks=spent.size, block_ms=1000.0 * block_size / sample_rate,
        median_ms=median, p95_ms=float(np.percentile(spent, 95)),
        worst_ms=float(spent.max()), jitter_ms=float(np.sqrt(np.mean(late ** 2))),
        engine_latency_ms=engine_ms,
    )


class RemoteConverter:
    """A converter whose work happens on another machine.

    Satisfies the same interface as the local engine, so everything above it --
    metering, the A/B, the device callback -- is unchanged.  What differs is
    where the delay comes from and that some of it is not under our control.

    The socket is read and written on a worker thread.  :meth:`process` puts a
    block on a queue and takes one off a buffer, so the audio callback does no
    I/O at all; if the buffer is empty it emits silence and counts it, because
    a callback that waited for the network would turn one late packet into a
    dropout on every subsequent block.
    """

    def __init__(self, url: str, sample_rate: int = 48000, profile=None,
                 block_size: int = 256, buffer_ms: float | None = None,
                 timeout: float = 10.0) -> None:
        from .. import api

        self.sample_rate = int(sample_rate)
        self.block_size = int(block_size)
        self.url = url
        self.underruns = 0
        self.errors: list[str] = []
        self._closed = threading.Event()

        self._client = WebSocketClient(url, timeout=timeout)
        message = self._client.receive()
        if message is None:
            raise WebSocketError("server closed the stream immediately")
        opcode, payload = message
        hello = json.loads(payload) if opcode == 0x1 else {}
        if int(hello.get("rate") or self.sample_rate) != self.sample_rate:
            self._client.close()
            raise WebSocketError(
                f"server is running at {hello.get('rate')} Hz, this program at "
                f"{self.sample_rate}; put ?rate={self.sample_rate} on the URL"
            )
        remote_latency = int(hello.get("latency_samples") or 0)
        if profile is not None:
            self._client.send_json(api.profile_to_dict(profile))
            reply = self._client.receive()
            if reply and reply[0] == 0x1:
                answer = json.loads(reply[1])
                if "error" in answer:
                    self._client.close()
                    raise WebSocketError(f"server refused the settings: {answer['error']}")
                remote_latency = int(round(
                    float(answer.get("latency_ms", 0.0)) * self.sample_rate / 1000.0
                )) or remote_latency

        buffer = MIN_BUFFER_MS if buffer_ms is None else float(buffer_ms)
        self._buffer_samples = max(self.block_size,
                                   int(round(buffer * self.sample_rate / 1000.0)))
        self.latency_samples = remote_latency + self._buffer_samples

        self._outbound: queue.Queue = queue.Queue(maxsize=64)
        self._inbound = np.zeros(0)
        self._ready = queue.Queue()
        # Prime the buffer, which is what the declared delay is made of.
        self._ready.put(np.zeros(self._buffer_samples))
        self._worker = threading.Thread(target=self._pump, daemon=True)
        self._worker.start()

    def _pump(self) -> None:
        sender = threading.Thread(target=self._send_loop, daemon=True)
        sender.start()
        try:
            while not self._closed.is_set():
                message = self._client.receive()
                if message is None:
                    break
                opcode, payload = message
                if opcode == 0x2:
                    self._ready.put(np.frombuffer(payload, dtype="<f4").astype(np.float64))
                elif opcode == 0x1:
                    answer = json.loads(payload or b"{}")
                    if "error" in answer:
                        self.errors.append(str(answer["error"]))
        except (WebSocketError, OSError, ValueError) as exc:
            if not self._closed.is_set():
                self.errors.append(str(exc))
        finally:
            self._closed.set()

    def _send_loop(self) -> None:
        while not self._closed.is_set():
            try:
                block = self._outbound.get(timeout=0.2)
            except queue.Empty:
                continue
            if block is None:
                return
            try:
                self._client.send_binary(np.asarray(block, dtype="<f4").tobytes())
            except (WebSocketError, OSError) as exc:
                if not self._closed.is_set():
                    self.errors.append(str(exc))
                self._closed.set()
                return

    def process(self, block: np.ndarray) -> np.ndarray:
        x = np.asarray(block, dtype=np.float64).reshape(-1)
        n = x.size
        if n == 0:
            return x
        try:
            self._outbound.put_nowait(x.copy())
        except queue.Full:
            # The link has stalled.  Dropping the newest block keeps the queue
            # from becoming a second, unbounded, unannounced latency.
            self.underruns += 1
        while self._inbound.size < n:
            try:
                self._inbound = np.concatenate([self._inbound, self._ready.get_nowait()])
            except queue.Empty:
                self.underruns += 1
                self._inbound = np.concatenate([self._inbound, np.zeros(n)])
        out, self._inbound = self._inbound[:n], self._inbound[n:]
        return out

    @property
    def observed_pitch(self) -> tuple[float, bool]:
        """Not available remotely: the tracker is on the other machine."""
        return 0.0, False

    def reset(self) -> None:
        self._inbound = np.zeros(0)
        self.underruns = 0

    def close(self) -> None:
        self._closed.set()
        try:
            self._outbound.put_nowait(None)
        except queue.Full:
            pass
        self._client.close()
