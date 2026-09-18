"""The network surface, exercised over a real socket.

There is no mocking here on purpose.  The framing, the handshake and the
upgrade are the parts most likely to be wrong, and a fake client that shares
this repository's idea of the protocol would agree with it whatever it did.
The client below is written from RFC 6455 rather than from
:mod:`natvox.server`, and masks its frames as a browser does.
"""
from __future__ import annotations

import base64
import json
import os
import socket
import struct
import urllib.error
import urllib.request

import numpy as np
import pytest

from natvox import api
from natvox.server import Server, read_wav, write_wav


@pytest.fixture(scope="module")
def service():
    server = Server(("127.0.0.1", 0))
    import threading
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def get(service, path):
    with urllib.request.urlopen(f"http://{service}{path}", timeout=10) as response:
        return response.status, json.loads(response.read())


def post(service, path, body, content_type="application/octet-stream"):
    request = urllib.request.Request(f"http://{service}{path}", data=body,
                                     headers={"Content-Type": content_type},
                                     method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.status, response.read()


class WsClient:
    """A masking WebSocket client, written from the RFC."""

    def __init__(self, service: str, path: str) -> None:
        host, port = service.split(":")
        self.sock = socket.create_connection((host, int(port)), timeout=15)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall(
            f"GET {path} HTTP/1.1\r\nHost: {service}\r\nUpgrade: websocket\r\n"
            f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n\r\n".encode()
        )
        header = b""
        while b"\r\n\r\n" not in header:
            chunk = self.sock.recv(1)
            if not chunk:
                raise AssertionError(f"handshake failed: {header!r}")
            header += chunk
        self.status = header.split(b"\r\n", 1)[0]
        self.headers = header

    def _read(self, n):
        out = b""
        while len(out) < n:
            chunk = self.sock.recv(n - len(out))
            if not chunk:
                raise AssertionError("closed mid-frame")
            out += chunk
        return out

    def send(self, opcode: int, payload: bytes) -> None:
        mask = os.urandom(4)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        n = len(payload)
        if n < 126:
            head = struct.pack("!BB", 0x80 | opcode, 0x80 | n)
        elif n < (1 << 16):
            head = struct.pack("!BBH", 0x80 | opcode, 0x80 | 126, n)
        else:
            head = struct.pack("!BBQ", 0x80 | opcode, 0x80 | 127, n)
        self.sock.sendall(head + mask + masked)

    def receive(self):
        first, second = self._read(2)
        opcode = first & 0x0F
        length = second & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._read(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._read(8))[0]
        assert not second & 0x80, "a server must not mask"
        return opcode, self._read(length) if length else b""

    def audio(self, block: np.ndarray) -> np.ndarray:
        self.send(0x2, np.asarray(block, dtype="<f4").tobytes())
        opcode, payload = self.receive()
        assert opcode == 0x2, (opcode, payload[:200])
        return np.frombuffer(payload, dtype="<f4").astype(np.float64)

    def control(self, obj) -> dict:
        self.send(0x1, json.dumps(obj).encode())
        opcode, payload = self.receive()
        assert opcode == 0x1
        return json.loads(payload)

    def close(self):
        try:
            self.send(0x8, struct.pack("!H", 1000))
        except OSError:
            pass
        self.sock.close()


class TestHttp:
    def test_the_index_lists_the_endpoints(self, service):
        status, body = get(service, "/")
        assert status == 200 and "GET /v1/stream" in body["endpoints"]

    def test_the_schema_is_the_same_one_the_library_publishes(self, service):
        _, body = get(service, "/v1/schema")
        assert body == json.loads(json.dumps(api.describe()))

    def test_voices_are_listed_with_their_settings(self, service):
        _, body = get(service, "/v1/voices")
        female = next(v for v in body["voices"] if v["name"] == "female")
        assert female["settings"]["intonation"] == pytest.approx(1.22)

    def test_unknown_routes_are_404_with_a_reason(self, service):
        with pytest.raises(urllib.error.HTTPError) as caught:
            get(service, "/v1/nope")
        assert caught.value.code == 404
        assert "no route" in json.loads(caught.value.read())["error"]

    def test_a_bad_setting_is_400_and_says_which(self, service):
        with pytest.raises(urllib.error.HTTPError) as caught:
            post(service, "/v1/convert?voice=female&pitch_semitones=99",
                 np.zeros(1024, dtype="<f4").tobytes())
        assert caught.value.code == 400
        assert "pitch_semitones" in json.loads(caught.value.read())["error"]

    def test_an_unknown_voice_is_400(self, service):
        with pytest.raises(urllib.error.HTTPError) as caught:
            get(service, "/v1/stream?voice=nobody")
        assert caught.value.code == 400


class TestConvertEndpoint:
    def test_raw_float32_round_trips(self, service, sample_rate):
        rng = np.random.default_rng(4)
        audio = (rng.normal(0, 0.1, sample_rate // 2)).astype("<f4")
        status, body = post(service, f"/v1/convert?voice=female&rate={sample_rate}",
                            audio.tobytes())
        assert status == 200
        out = np.frombuffer(body, dtype="<f4")
        assert out.size == audio.size and np.all(np.isfinite(out))

    def test_it_agrees_with_the_library(self, service, sample_rate):
        audio = (0.3 * np.sin(2 * np.pi * 140
                              * np.arange(sample_rate // 2) / sample_rate)).astype("<f4")
        _, body = post(service, f"/v1/convert?voice=female_soft&rate={sample_rate}",
                       audio.tobytes())
        served = np.frombuffer(body, dtype="<f4").astype(np.float64)
        local = api.convert(audio.astype(np.float64), sample_rate, "female_soft")
        assert np.max(np.abs(served - local)) < 1e-6

    def test_query_overrides_are_applied(self, service, sample_rate):
        audio = (0.3 * np.sin(2 * np.pi * 140
                              * np.arange(sample_rate // 2) / sample_rate)).astype("<f4")
        _, body = post(
            service,
            f"/v1/convert?voice=female&rate={sample_rate}&pitch_semitones=2&breathiness=0",
            audio.tobytes())
        served = np.frombuffer(body, dtype="<f4").astype(np.float64)
        expected = api.convert(
            audio.astype(np.float64), sample_rate,
            api.profile_from_dict({"pitch_semitones": 2.0, "breathiness": 0.0},
                                  api.get_voice("female").profile))
        assert np.max(np.abs(served - expected)) < 1e-6

    def test_wav_in_wav_out(self, service, sample_rate):
        tone = 0.4 * np.sin(2 * np.pi * 130 * np.arange(sample_rate) / sample_rate)
        status, body = post(service, "/v1/convert?voice=female",
                            write_wav(tone, sample_rate), "audio/wav")
        assert status == 200
        out, rate = read_wav(body)
        assert rate == sample_rate and out.size == tone.size

    def test_a_truncated_wav_is_400_not_500(self, service):
        with pytest.raises(urllib.error.HTTPError) as caught:
            post(service, "/v1/convert?voice=female", b"RIFF\x00\x00\x00\x00WAVE",
                 "audio/wav")
        assert caught.value.code == 400

    def test_an_empty_body_is_400(self, service):
        with pytest.raises(urllib.error.HTTPError) as caught:
            post(service, "/v1/convert?voice=female", b"")
        assert caught.value.code == 400

    def test_a_ragged_float32_body_is_400(self, service):
        with pytest.raises(urllib.error.HTTPError) as caught:
            post(service, "/v1/convert?voice=female", b"\x00\x00\x00")
        assert caught.value.code == 400


class TestStream:
    def test_it_upgrades_and_states_its_latency(self, service):
        client = WsClient(service, "/v1/stream?voice=female&rate=48000")
        assert b"101" in client.status
        opcode, payload = client.receive()
        ready = json.loads(payload)
        assert opcode == 0x1 and ready["ready"] and ready["voice"] == "female"
        assert 0 < ready["latency_ms"] < 200
        assert ready["format"] == "float32le mono"
        client.close()

    def test_blocks_come_back_the_same_length(self, service):
        client = WsClient(service, "/v1/stream?voice=female&rate=48000")
        client.receive()
        for size in (64, 128, 480, 1024):
            out = client.audio(np.zeros(size))
            assert out.size == size
        client.close()

    def test_it_matches_a_local_session(self, service, sample_rate):
        client = WsClient(service, f"/v1/stream?voice=female&rate={sample_rate}")
        client.receive()
        rng = np.random.default_rng(11)
        blocks = [rng.normal(0, 0.1, 512) for _ in range(20)]
        served = np.concatenate([client.audio(b) for b in blocks])
        client.close()
        session = api.Session(sample_rate, "female")
        local = np.concatenate([session.process(b) for b in blocks])
        assert np.max(np.abs(served - local)) < 1e-6

    def test_settings_can_change_while_audio_is_flowing(self, service, sample_rate):
        client = WsClient(service, f"/v1/stream?voice=female_soft&rate={sample_rate}")
        client.receive()
        for _ in range(10):
            client.audio(np.zeros(512))
        reply = client.control({"pitch_semitones": 6.0})
        assert reply["ok"] and reply["settings"]["pitch_semitones"] == 6.0
        assert reply["build_ms"] > 0
        for _ in range(10):
            assert client.audio(np.zeros(512)).size == 512
        client.close()

    def test_switching_to_a_named_voice_works(self, service, sample_rate):
        client = WsClient(service, f"/v1/stream?voice=off&rate={sample_rate}")
        client.receive()
        reply = client.control({"voice": "male_to_female_subtle"})
        assert reply["voice"] == "male_to_female_subtle"
        client.close()

    def test_a_rejected_change_is_reported_and_the_stream_survives(
            self, service, sample_rate):
        client = WsClient(service, f"/v1/stream?voice=female&rate={sample_rate}")
        client.receive()
        assert "error" in client.control({"pitch_semitones": 99.0})
        assert "error" in client.control({"nonsense": 1})
        assert "delay" in client.control({"f0_min": 45.0})["error"]
        assert client.audio(np.zeros(256)).size == 256
        client.close()

    def test_a_ping_is_answered_without_the_caller_doing_anything(self, service):
        client = WsClient(service, "/v1/stream?voice=off")
        client.receive()
        client.send(0x9, b"alive?")
        opcode, payload = client.receive()
        assert opcode == 0xA and payload == b"alive?"
        client.close()

    def test_a_fragmented_message_is_reassembled(self, service):
        client = WsClient(service, "/v1/stream?voice=off&rate=48000")
        client.receive()
        payload = np.zeros(30, dtype="<f4").tobytes()   # halves stay under 126
        half = len(payload) // 2
        mask = os.urandom(4)
        def frame(first_byte, chunk):
            masked = bytes(b ^ mask[i % 4] for i, b in enumerate(chunk))
            return (struct.pack("!BB", first_byte, 0x80 | len(chunk)) + mask + masked)
        client.sock.sendall(frame(0x02, payload[:half]))      # binary, not final
        client.sock.sendall(frame(0x80, payload[half:]))      # continuation, final
        opcode, out = client.receive()
        assert opcode == 0x2 and len(out) == len(payload)
        client.close()

    def test_a_ragged_audio_frame_is_reported_without_dropping_the_stream(self, service):
        client = WsClient(service, "/v1/stream?voice=off")
        client.receive()
        client.send(0x2, b"\x00\x00\x00")
        opcode, payload = client.receive()
        assert opcode == 0x1 and "float32" in json.loads(payload)["error"]
        assert client.audio(np.zeros(128)).size == 128
        client.close()


class TestWav:
    @pytest.mark.parametrize("rate", [16000, 44100, 48000])
    def test_round_trip_keeps_the_audio(self, rate):
        tone = 0.5 * np.sin(2 * np.pi * 200 * np.arange(rate) / rate)
        out, back = read_wav(write_wav(tone, rate))
        assert back == rate
        assert np.max(np.abs(out - tone)) < 2e-4      # 16-bit quantisation
