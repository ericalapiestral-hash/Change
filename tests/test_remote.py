"""Converting on another machine, and what it costs.

The point of these is not that the feature works -- it is that the program can
tell you, on your own link, whether it is worth using.  A number measured on
the connection you actually have beats any claim made here about typical
latency.
"""
from __future__ import annotations

import json
import threading
import time

import numpy as np
import pytest

from natvox import api
from natvox.app.remote import JITTER_MULTIPLE, LinkReport, RemoteConverter, probe
from natvox.app.wsclient import WebSocketClient, WebSocketError
from natvox.server import Server


@pytest.fixture(scope="module")
def service():
    server = Server(("127.0.0.1", 0))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


@pytest.fixture
def stream_url(service, sample_rate):
    return f"ws://{service}/v1/stream?voice=female&rate={sample_rate}"


def paced(converter, audio, sample_rate, block=256):
    """Feed it the way an audio callback would: no faster than real time."""
    out, started, emitted = [], time.perf_counter(), 0
    for i in range(0, audio.size, block):
        chunk = audio[i:i + block]
        out.append(converter.process(chunk))
        emitted += chunk.size
        slack = started + emitted / sample_rate - time.perf_counter()
        if slack > 0:
            time.sleep(slack)
    return np.concatenate(out)


class TestClient:
    def test_it_speaks_to_the_server_in_this_package(self, stream_url):
        with WebSocketClient(stream_url) as client:
            opcode, payload = client.receive()
            assert opcode == 0x1
            assert json.loads(payload)["ready"] is True

    def test_an_unreachable_address_is_a_sentence(self):
        with pytest.raises(WebSocketError, match="could not reach"):
            WebSocketClient("ws://127.0.0.1:1/v1/stream", timeout=2.0)

    def test_https_is_refused_with_the_reason(self):
        with pytest.raises(WebSocketError, match="certificate"):
            WebSocketClient("wss://example.invalid/v1/stream")

    def test_a_route_that_is_not_a_stream_is_refused(self, service):
        with pytest.raises(WebSocketError, match="refused the upgrade"):
            WebSocketClient(f"ws://{service}/v1/voices")


class TestProbe:
    def test_it_reports_the_link_it_measured(self, stream_url, sample_rate):
        report = probe(stream_url, block_size=256, sample_rate=sample_rate,
                       blocks=60, warmup=10)
        assert report.blocks == 60
        assert 0 < report.median_ms <= report.p95_ms <= report.worst_ms
        assert report.jitter_ms >= 0
        assert report.engine_latency_ms > 40          # the server's own delay
        assert report.buffer_ms >= report.block_ms
        assert report.total_ms > report.engine_latency_ms
        assert report.summary().count("\n") == 4

    def test_loopback_is_the_floor_and_it_is_not_free(self, stream_url, sample_rate):
        """Even with no network at all the protocol costs something, and the
        buffer it implies is what a real link adds on top."""
        report = probe(stream_url, blocks=60, warmup=10, sample_rate=sample_rate)
        assert report.median_ms < 15.0
        assert report.total_ms > report.engine_latency_ms + report.block_ms

    @pytest.mark.parametrize("median,verdict", [
        (5.0, "usable for conversation"),
        (80.0, "noticeable"),
        (250.0, "one-way only"),
    ])
    def test_the_verdict_follows_the_total(self, median, verdict):
        report = LinkReport("ws://x", 100, 5.33, median, median, median, 1.0, 60.0)
        assert verdict in report.verdict

    def test_the_buffer_is_sized_from_jitter_not_from_distance(self):
        """A link with a long but steady round trip needs no more buffer than
        a short one; a short but erratic link needs a lot."""
        steady = LinkReport("ws://x", 100, 5.33, 120.0, 121.0, 122.0, 0.3, 60.0)
        erratic = LinkReport("ws://x", 100, 5.33, 8.0, 40.0, 60.0, 14.0, 60.0)
        assert steady.buffer_ms < erratic.buffer_ms
        assert erratic.buffer_ms >= JITTER_MULTIPLE * erratic.jitter_ms


class TestRemoteConverter:
    def test_it_matches_the_same_conversion_done_locally(self, stream_url, sample_rate):
        t = np.arange(2 * sample_rate) / sample_rate
        audio = 0.3 * np.sin(2 * np.pi * 130 * t) + 0.1 * np.sin(2 * np.pi * 260 * t)
        converter = RemoteConverter(stream_url, sample_rate, buffer_ms=20)
        try:
            served = paced(converter, audio, sample_rate)
            assert converter.underruns == 0, converter.errors
            remote = served[converter.latency_samples:]
        finally:
            converter.close()

        session = api.Session(sample_rate, "female")
        local = np.concatenate([session.process(audio[i:i + 256])
                                for i in range(0, audio.size, 256)])
        local = local[session.latency_samples:]
        n = min(remote.size, local.size)
        # float32 on the wire is the only difference there should be.
        assert np.max(np.abs(remote[:n] - local[:n])) < 1e-4

    def test_the_delay_it_declares_is_the_delay_it_has(self, stream_url, sample_rate):
        converter = RemoteConverter(stream_url, sample_rate, buffer_ms=20)
        try:
            server_side = api.Session(sample_rate, "female").latency_samples
            buffer = int(round(0.020 * sample_rate))
            assert converter.latency_samples == server_side + buffer
        finally:
            converter.close()

    def test_a_link_that_cannot_keep_up_reports_it_instead_of_stalling(
            self, stream_url, sample_rate):
        """Driven faster than real time, the far end falls behind.  The
        callback must not wait for it -- one late packet would then become a
        dropout on every block after it."""
        converter = RemoteConverter(stream_url, sample_rate, buffer_ms=10)
        try:
            audio = np.zeros(sample_rate)
            for i in range(0, audio.size, 256):          # flat out, no pacing
                chunk = audio[i:i + 256]
                assert converter.process(chunk).size == chunk.size
            assert converter.underruns > 0
        finally:
            converter.close()

    def test_a_sample_rate_the_server_is_not_running_at_is_refused(self, service):
        with pytest.raises(WebSocketError, match="rate="):
            RemoteConverter(f"ws://{service}/v1/stream?voice=off&rate=44100", 48000)

    def test_settings_the_server_refuses_are_reported_at_connect(self, service,
                                                                 sample_rate):
        from natvox import VoiceProfile

        url = f"ws://{service}/v1/stream?voice=off&rate={sample_rate}"
        # f0_min this low needs more delay than the session budgeted for.
        with pytest.raises(WebSocketError, match="refused the settings"):
            RemoteConverter(url, sample_rate, VoiceProfile(f0_min=42.0))

    def test_closing_it_twice_is_harmless(self, stream_url, sample_rate):
        converter = RemoteConverter(stream_url, sample_rate)
        converter.close()
        converter.close()

    def test_it_reports_no_pitch_because_the_tracker_is_elsewhere(self, stream_url,
                                                                  sample_rate):
        converter = RemoteConverter(stream_url, sample_rate)
        try:
            assert converter.observed_pitch == (0.0, False)
        finally:
            converter.close()


class TestServerManners:
    """A streaming server spends its life being hung up on.

    The first version of these asserted only that no traceback was printed,
    which passed -- and passed for the wrong reason.  A client that resets a
    live stream is handled inside the stream handler, so it never reaches
    `handle_error` and there was never anything to print.  The assertion was
    true and empty.  Both halves are now pinned down separately: what the
    resets actually do, and what `handle_error` does when it is reached.
    """

    def test_a_client_that_vanishes_leaves_the_server_serving(self, capfd):
        import socket as socketlib
        import struct
        import sys

        # SO_LINGER is `struct linger`, which is two ints on Unix and two
        # u_shorts on Windows.  Windows turned out to accept the eight-byte
        # version anyway, so this is not fixing a failure -- it is declining to
        # depend on an undocumented tolerance somewhere being wrong would be
        # silent: a rejected setsockopt means a polite close, and a polite
        # close is the one thing this must not do.
        linger = struct.pack("HH" if sys.platform == "win32" else "ii", 1, 0)

        # Its own server.  "Nothing was printed" is only worth asserting when
        # nothing else could have printed; sharing the module's server made
        # this fail about once in four full runs on a neighbour's output.
        server = Server(("127.0.0.1", 0))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        port = server.server_address[1]
        capfd.readouterr()
        try:
            # Half of them read the upgrade first, half vanish before the
            # server has written a byte.  The second half is the one that
            # mattered: the greeting used to be sent outside the handler's
            # guard, so a client gone by then printed a stack trace -- which
            # only showed up on Windows, where the reset lands while the write
            # is still in flight.
            for read_first in (True, False, True, False, False):
                sock = socketlib.create_connection(("127.0.0.1", port), timeout=5)
                sock.sendall(b"GET /v1/stream?voice=off HTTP/1.1\r\nHost: x\r\n"
                             b"Upgrade: websocket\r\nConnection: Upgrade\r\n"
                             b"Sec-WebSocket-Key: AAAAAAAAAAAAAAAAAAAAAA==\r\n"
                             b"Sec-WebSocket-Version: 13\r\n\r\n")
                if read_first:
                    sock.recv(200)
                sock.setsockopt(socketlib.SOL_SOCKET, socketlib.SO_LINGER, linger)
                sock.close()                     # RST, not a clean close
            time.sleep(0.5)
            # The positive half: it still works afterwards.
            with WebSocketClient(f"ws://127.0.0.1:{port}/v1/stream?voice=off") as ok:
                opcode, payload = ok.receive()
                assert opcode == 0x1 and json.loads(payload)["ready"] is True
                ok.send_binary(np.zeros(128, dtype="<f4").tobytes())
                assert ok.receive()[0] == 0x2
        finally:
            server.shutdown()
            server.server_close()
        assert "Traceback" not in capfd.readouterr().err

    def test_handle_error_counts_a_lost_connection_instead_of_printing(self, capfd):
        """Reached when the socket dies while the handler is finishing with it,
        which is where the tracebacks were actually coming from."""
        server = Server(("127.0.0.1", 0))
        try:
            capfd.readouterr()
            try:
                raise ConnectionResetError(104, "Connection reset by peer")
            except ConnectionResetError:
                server.handle_error(None, ("127.0.0.1", 12345))
            assert server.dropped == 1
            assert "Traceback" not in capfd.readouterr().err
        finally:
            server.server_close()

    def test_a_real_error_is_still_printed(self, capfd):
        """The point is not to be quiet.  It is to be quiet about the one thing
        that is not a problem, so that the log still means something."""
        server = Server(("127.0.0.1", 0))
        try:
            capfd.readouterr()
            try:
                raise ValueError("something actually went wrong")
            except ValueError:
                server.handle_error(None, ("127.0.0.1", 12345))
            assert server.dropped == 0
            assert "something actually went wrong" in capfd.readouterr().err
        finally:
            server.server_close()

    def test_the_stream_survives_its_neighbours_vanishing(self, stream_url,
                                                          sample_rate):
        converter = RemoteConverter(stream_url, sample_rate, buffer_ms=20)
        try:
            out = paced(converter, np.zeros(sample_rate // 4), sample_rate)
            assert out.size == sample_rate // 4
            assert not converter.errors
        finally:
            converter.close()
