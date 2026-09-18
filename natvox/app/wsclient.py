"""A small WebSocket client, standard library only.

The server side lives in :mod:`natvox.server` and speaks the same protocol
from the other end.  This is separate rather than shared because the two
halves of RFC 6455 differ in the one place that matters -- a client masks its
frames and a server must not -- and a single class with a flag for that is a
class where the flag is eventually wrong.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import struct
import threading
from urllib.parse import urlparse

TEXT, BINARY, CLOSE, PING, PONG = 0x1, 0x2, 0x8, 0x9, 0xA

_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

#: Refuse a frame larger than this rather than allocating whatever a server
#: claims to be sending.
MAX_FRAME_BYTES = 16 * 1024 * 1024


class WebSocketError(RuntimeError):
    pass


class WebSocketClient:
    """Connect, then send and receive whole messages."""

    def __init__(self, url: str, timeout: float = 10.0) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in ("ws", "http", ""):
            raise WebSocketError(
                f"only ws:// is supported here, not {parsed.scheme!r}; wss "
                f"would need a certificate chain this program does not manage"
            )
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 80
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        self.url = url
        try:
            self.sock = socket.create_connection((host, port), timeout=timeout)
        except OSError as exc:
            raise WebSocketError(f"could not reach {host}:{port}: {exc}") from exc
        # Audio is small and frequent, which is the case Nagle's algorithm is
        # worst for: it would hold a block back waiting for company.
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._send_lock = threading.Lock()
        self.closed = False
        self._handshake(host, port, path)

    def _handshake(self, host: str, port: int, path: str) -> None:
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall(
            f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\n"
            f"Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n".encode()
        )
        header = b""
        while b"\r\n\r\n" not in header:
            chunk = self.sock.recv(1)
            if not chunk:
                raise WebSocketError(f"connection closed during the handshake: {header!r}")
            header += chunk
            if len(header) > 65536:
                raise WebSocketError("handshake response is implausibly large")
        status = header.split(b"\r\n", 1)[0]
        if b" 101" not in status:
            body = header.decode("utf-8", "replace")
            raise WebSocketError(f"server refused the upgrade: {status.decode()!r}\n{body}")
        expected = base64.b64encode(hashlib.sha1((key + _GUID).encode()).digest()).decode()
        if expected.lower().encode() not in header.lower():
            raise WebSocketError("server's Sec-WebSocket-Accept does not match")

    # -- reading
    def _exactly(self, n: int) -> bytes:
        parts = []
        while n > 0:
            chunk = self.sock.recv(n)
            if not chunk:
                raise WebSocketError("connection closed mid-frame")
            parts.append(chunk)
            n -= len(chunk)
        return b"".join(parts)

    def receive(self):
        """Next message as ``(opcode, payload)``; ``None`` once closed."""
        buffer = bytearray()
        kind = None
        while True:
            first, second = self._exactly(2)
            final = bool(first & 0x80)
            opcode = first & 0x0F
            if second & 0x80:
                raise WebSocketError("a server must not mask its frames")
            length = second & 0x7F
            if length == 126:
                length = struct.unpack("!H", self._exactly(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", self._exactly(8))[0]
            if length > MAX_FRAME_BYTES:
                raise WebSocketError(f"frame of {length} bytes exceeds the limit")
            payload = self._exactly(length) if length else b""
            if opcode == CLOSE:
                self.close()
                return None
            if opcode == PING:
                self._frame(PONG, payload)
                continue
            if opcode == PONG:
                continue
            if opcode != 0x0:
                kind = opcode
            elif kind is None:
                raise WebSocketError("continuation without a start frame")
            buffer += payload
            if len(buffer) > MAX_FRAME_BYTES:
                raise WebSocketError("fragmented message exceeds the limit")
            if final:
                return kind, bytes(buffer)

    # -- writing
    def _frame(self, opcode: int, payload: bytes) -> None:
        if self.closed:
            raise WebSocketError("socket is closed")
        mask = os.urandom(4)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        n = len(payload)
        if n < 126:
            header = struct.pack("!BB", 0x80 | opcode, 0x80 | n)
        elif n < (1 << 16):
            header = struct.pack("!BBH", 0x80 | opcode, 0x80 | 126, n)
        else:
            header = struct.pack("!BBQ", 0x80 | opcode, 0x80 | 127, n)
        with self._send_lock:
            try:
                self.sock.sendall(header + mask + masked)
            except OSError as exc:
                self.closed = True
                raise WebSocketError(str(exc)) from None

    def send_binary(self, payload: bytes) -> None:
        self._frame(BINARY, payload)

    def send_json(self, obj) -> None:
        self._frame(TEXT, json.dumps(obj).encode())

    def close(self) -> None:
        if self.closed:
            return
        try:
            self._frame(CLOSE, struct.pack("!H", 1000))
        except WebSocketError:
            pass
        self.closed = True
        try:
            self.sock.close()
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
