"""A local HTTP and WebSocket service around the engine.

Three endpoints do the work::

    GET  /v1/schema                  what every parameter means, as data
    GET  /v1/voices                  the voices this server knows
    POST /v1/convert?voice=female    a whole file in, a whole file out
    GET  /v1/stream?voice=female     a WebSocket: PCM in, PCM out, live control

The streaming endpoint is the reason this exists.  Conversion has to happen
somewhere with the audio, and "somewhere" is often not the process holding the
microphone -- a game, a browser tab, a phone.  A socket that takes blocks of
PCM and hands blocks back, at a latency the server states up front, is the
smallest thing that makes the engine usable from all of them.

Deliberately built on the standard library alone, WebSocket framing included.
The engine's only hard dependency is numpy; a service that dragged in a web
framework would make the interesting part harder to deploy than the boring
part, and the protocol is a few hundred lines.

It binds to the loopback interface by default, and that default is the
security model: there is no authentication here, so anything that can reach
the port can use the engine and register voices on it.  Putting it on a
routable address is a decision for whoever does it, and ``--host`` makes them
make it.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import struct
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import api
from .api import ParameterError

#: Magic from RFC 6455; the handshake is a proof that the server understood
#: the protocol rather than a security measure.
_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

#: Refuse bodies larger than this.  A conversion holds the whole array in
#: memory several times over, so an unbounded POST is a way to stop the
#: machine rather than a way to convert a long file.
MAX_BODY_BYTES = 256 * 1024 * 1024

#: Largest single WebSocket message accepted.  Audio arrives in blocks of a
#: few hundred samples; anything near this is not audio.
MAX_FRAME_BYTES = 16 * 1024 * 1024

DEFAULT_RATE = 48000


# ------------------------------------------------------------------ WebSocket

class WebSocketError(Exception):
    pass


class WebSocket:
    """Enough of RFC 6455 to carry audio: binary, text, ping, pong, close.

    No extensions and no fragmentation on the way out.  Fragments *in* are
    reassembled, because browsers send them and refusing would be a bug that
    only appears on large messages.
    """

    TEXT, BINARY, CLOSE, PING, PONG = 0x1, 0x2, 0x8, 0x9, 0xA

    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.closed = False
        self._send_lock = threading.Lock()

    # -- reading
    def _read_exactly(self, n: int) -> bytes:
        chunks = []
        while n > 0:
            chunk = self.sock.recv(n)
            if not chunk:
                raise WebSocketError("connection closed mid-frame")
            chunks.append(chunk)
            n -= len(chunk)
        return b"".join(chunks)

    def _read_frame(self):
        header = self._read_exactly(2)
        final = bool(header[0] & 0x80)
        opcode = header[0] & 0x0F
        masked = bool(header[1] & 0x80)
        length = header[1] & 0x7F
        if length == 126:
            length = struct.unpack(">H", self._read_exactly(2))[0]
        elif length == 127:
            length = struct.unpack(">Q", self._read_exactly(8))[0]
        if length > MAX_FRAME_BYTES:
            raise WebSocketError(f"frame of {length} bytes exceeds the limit")
        mask = self._read_exactly(4) if masked else b""
        payload = self._read_exactly(length) if length else b""
        if masked and payload:
            key = np.frombuffer(mask, dtype=np.uint8)
            data = np.frombuffer(payload, dtype=np.uint8)
            payload = (data ^ np.resize(key, data.size)).tobytes()
        return final, opcode, payload

    def receive(self):
        """Next application message as ``(opcode, payload)``; ``None`` on close.

        Control frames are handled here and never surface: a ping that the
        caller had to remember to answer would be a ping that goes unanswered
        the first time someone writes a loop.
        """
        buffer = bytearray()
        kind = None
        while True:
            final, opcode, payload = self._read_frame()
            if opcode == self.CLOSE:
                self.close()
                return None
            if opcode == self.PING:
                self._send(self.PONG, payload)
                continue
            if opcode == self.PONG:
                continue
            if opcode == 0x0:                       # continuation
                if kind is None:
                    raise WebSocketError("continuation without a start frame")
            else:
                kind = opcode
            buffer += payload
            if len(buffer) > MAX_FRAME_BYTES:
                raise WebSocketError("fragmented message exceeds the limit")
            if final:
                return kind, bytes(buffer)

    # -- writing
    def _send(self, opcode: int, payload: bytes) -> None:
        if self.closed:
            return
        length = len(payload)
        if length < 126:
            header = struct.pack("!BB", 0x80 | opcode, length)
        elif length < (1 << 16):
            header = struct.pack("!BBH", 0x80 | opcode, 126, length)
        else:
            header = struct.pack("!BBQ", 0x80 | opcode, 127, length)
        with self._send_lock:
            try:
                self.sock.sendall(header + payload)
            except OSError as exc:
                self.closed = True
                raise WebSocketError(str(exc)) from None

    def send_binary(self, payload: bytes) -> None:
        self._send(self.BINARY, payload)

    def send_json(self, obj) -> None:
        self._send(self.TEXT, json.dumps(obj).encode())

    def close(self, code: int = 1000) -> None:
        if self.closed:
            return
        try:
            self._send(self.CLOSE, struct.pack("!H", code))
        except WebSocketError:
            pass
        self.closed = True


def _accept_key(key: str) -> str:
    digest = hashlib.sha1((key + _WS_GUID).encode()).digest()
    return base64.b64encode(digest).decode()


# ----------------------------------------------------------------- PCM & WAV

def _to_float(payload: bytes) -> np.ndarray:
    """Interpret a binary frame as little-endian float32 mono."""
    if len(payload) % 4:
        raise ParameterError(
            f"audio frames are float32, so their length must be a multiple of "
            f"4 bytes; got {len(payload)}"
        )
    audio = np.frombuffer(payload, dtype="<f4").astype(np.float64)
    if audio.size and not np.isfinite(audio).all():
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    return audio


def _from_float(audio: np.ndarray) -> bytes:
    return np.asarray(audio, dtype="<f4").tobytes()


def read_wav(data: bytes):
    """``(mono float64, sample_rate)`` from 8/16/32-bit PCM WAV bytes."""
    with wave.open(BytesIO(data), "rb") as handle:
        channels, width, rate = (handle.getnchannels(), handle.getsampwidth(),
                                 handle.getframerate())
        frames = handle.readframes(handle.getnframes())
    if width == 2:
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768.0
    elif width == 4:
        samples = np.frombuffer(frames, dtype="<i4").astype(np.float64) / 2147483648.0
    elif width == 1:
        samples = (np.frombuffer(frames, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    else:
        raise ParameterError(f"unsupported WAV sample width: {width} bytes")
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, rate


def write_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    clipped = np.clip(np.asarray(audio, dtype=np.float64), -1.0, 1.0)
    out = BytesIO()
    with wave.open(out, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(np.round(clipped * 32767.0).astype("<i2").tobytes())
    return out.getvalue()


# ------------------------------------------------------------------- routing

def _voice_from_query(query: dict) -> api.Voice:
    """The voice a request asks for: a named one, with any overrides applied."""
    name = query.get("voice", ["off"])[0]
    voice = api.get_voice(name)
    overrides = {k: v[0] for k, v in query.items()
                 if k not in ("voice", "rate", "block")}
    if not overrides:
        return voice
    typed: dict = {}
    for key, raw in overrides.items():
        if key not in {p.name for p in api.PARAMETERS}:
            raise ParameterError(f"unknown setting {key!r} in the query string")
        typed[key] = raw.lower() in ("1", "true", "yes", "on") \
            if key == "shift_unvoiced" else raw
    return api.Voice(voice.name, api.profile_from_dict(typed, voice.profile),
                     voice.model, voice.summary)


def _rate_from_query(query: dict) -> int:
    raw = query.get("rate", [str(DEFAULT_RATE)])[0]
    try:
        rate = int(raw)
    except ValueError:
        raise ParameterError(f"rate must be an integer, got {raw!r}") from None
    if not 8000 <= rate <= 192000:
        raise ParameterError(f"rate must be between 8000 and 192000, got {rate}")
    return rate


class Handler(BaseHTTPRequestHandler):
    server_version = "natvox"
    protocol_version = "HTTP/1.1"

    # -- helpers
    def _json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def _bytes(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):            # quiet by default
        if self.server.verbose:                   # type: ignore[attr-defined]
            super().log_message(fmt, *args)

    # -- verbs
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path in ("/", "/v1"):
                return self._json({
                    "name": "natvox",
                    "version": api.describe()["version"],
                    "endpoints": {
                        "GET /v1/health": "liveness",
                        "GET /v1/schema": "every parameter, its units and range",
                        "GET /v1/voices": "available voices",
                        "POST /v1/voices": "register a voice from JSON",
                        "POST /v1/convert": "a whole file in, a whole file out",
                        "GET /v1/stream": "WebSocket: PCM in, PCM out",
                    },
                })
            if parsed.path == "/v1/health":
                return self._json({"ok": True})
            if parsed.path == "/v1/schema":
                return self._json(api.describe())
            if parsed.path == "/v1/voices":
                return self._json({"voices": [v.as_dict() for v in api.voices()]})
            if parsed.path == "/v1/stream":
                return self._stream(query)
        except ParameterError as exc:
            return self._error(400, str(exc))
        self._error(404, f"no route for GET {parsed.path}")

    def do_POST(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self._error(400, "Content-Length must be an integer")
        if length > MAX_BODY_BYTES:
            return self._error(413, f"body exceeds {MAX_BODY_BYTES} bytes")
        body = self.rfile.read(length) if length else b""
        try:
            if parsed.path == "/v1/convert":
                return self._convert(query, body)
            if parsed.path == "/v1/voices":
                voice = api.register_voice(api.Voice.from_dict(json.loads(body or b"{}")))
                return self._json(voice.as_dict(), 201)
        except ParameterError as exc:
            return self._error(400, str(exc))
        except (wave.Error, EOFError) as exc:
            return self._error(400, f"could not read the audio: {exc}")
        except json.JSONDecodeError as exc:
            return self._error(400, f"body is not valid JSON: {exc}")
        self._error(404, f"no route for POST {parsed.path}")

    # -- endpoints
    def _convert(self, query: dict, body: bytes) -> None:
        if not body:
            raise ParameterError("no audio in the request body")
        voice = _voice_from_query(query)
        content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip()
        if content_type in ("audio/wav", "audio/x-wav", "audio/wave"):
            audio, rate = read_wav(body)
            converted = api.convert(audio, rate, voice)
            return self._bytes(write_wav(converted, rate), "audio/wav")
        rate = _rate_from_query(query)
        converted = api.convert(_to_float(body), rate, voice)
        return self._bytes(_from_float(converted), "application/octet-stream")

    def _stream(self, query: dict) -> None:
        if (self.headers.get("Upgrade") or "").lower() != "websocket":
            raise ParameterError("/v1/stream is a WebSocket endpoint; send an "
                                 "Upgrade: websocket request")
        key = self.headers.get("Sec-WebSocket-Key")
        if not key:
            raise ParameterError("missing Sec-WebSocket-Key")
        voice = _voice_from_query(query)
        rate = _rate_from_query(query)
        session = api.Session(rate, voice)

        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", _accept_key(key))
        self.end_headers()
        self.wfile.flush()

        ws = WebSocket(self.connection)
        self.close_connection = True
        ws.send_json({
            "ready": True, "voice": session.voice.name, "rate": rate,
            "latency_ms": round(session.latency_ms, 2),
            "latency_samples": session.latency_samples,
            "format": "float32le mono",
            "settings": session.settings(),
        })
        try:
            self._pump(ws, session)
        except (WebSocketError, OSError):
            pass
        finally:
            ws.close()

    @staticmethod
    def _pump(ws: WebSocket, session: api.Session) -> None:
        """Audio in, audio out; text frames change the voice as it runs.

        The reply to a control frame carries the settings that ended up in
        force, not the ones that were asked for.  A caller that sends
        ``{"pitch_semitones": 40}`` needs to know it did not get 40.
        """
        while True:
            message = ws.receive()
            if message is None:
                return
            opcode, payload = message
            if opcode == WebSocket.BINARY:
                try:
                    block = _to_float(payload)
                except ParameterError as exc:
                    ws.send_json({"error": str(exc)})
                    continue
                ws.send_binary(_from_float(session.process(block)))
            elif opcode == WebSocket.TEXT:
                try:
                    request = json.loads(payload or b"{}")
                    if not isinstance(request, dict):
                        raise ParameterError("a control frame must be a JSON object")
                    name = request.pop("voice", None)
                    build_ms = session.set(name, **request)
                except (ParameterError, json.JSONDecodeError) as exc:
                    ws.send_json({"error": str(exc)})
                    continue
                ws.send_json({
                    "ok": True, "voice": session.voice.name,
                    "settings": session.settings(),
                    "build_ms": round(build_ms, 2),
                    "latency_ms": round(session.latency_ms, 2),
                })


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, verbose: bool = False) -> None:
        self.verbose = verbose
        super().__init__(address, Handler)


def serve(host: str = "127.0.0.1", port: int = 8420, verbose: bool = False) -> Server:
    """Start a server on a background thread and return it (call ``shutdown``)."""
    server = Server((host, port), verbose)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main(host: str = "127.0.0.1", port: int = 8420, verbose: bool = False) -> int:
    server = Server((host, port), verbose)
    shown = host if host != "0.0.0.0" else socket.gethostname()
    print(f"natvox serving on http://{shown}:{server.server_address[1]}")
    print(f"  schema   http://{shown}:{server.server_address[1]}/v1/schema")
    print(f"  stream   ws://{shown}:{server.server_address[1]}/v1/stream?voice=female")
    if host not in ("127.0.0.1", "localhost", "::1"):
        print("  note: this port has no authentication and is not on loopback")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(os.environ.get("NATVOX_HOST", "127.0.0.1"),
                          int(os.environ.get("NATVOX_PORT", "8420"))))
