from __future__ import annotations

import json
import socket
from base64 import b64decode, b64encode
from typing import Any, Dict, Tuple

import numpy as np


class NumpyJsonEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return {
                "__numpy_array__": True,
                "dtype": str(obj.dtype),
                "shape": obj.shape,
                "data": b64encode(obj.tobytes()).decode("ascii"),
            }
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _numpy_object_hook(dct: Dict[str, Any]) -> Any:
    if dct.get("__numpy_array__"):
        raw = b64decode(dct["data"])
        return np.frombuffer(raw, dtype=dct["dtype"]).reshape(dct["shape"])
    return dct


class JsonSocket:
    """Tiny length-prefixed JSON socket helper."""

    def __init__(self, sock: socket.socket):
        self.sock = sock

    def send(self, message: Dict[str, Any]) -> None:
        payload = json.dumps(message, cls=NumpyJsonEncoder).encode("utf-8")
        header = len(payload).to_bytes(4, "big")
        self.sock.sendall(header)
        self.sock.sendall(payload)

    def recv(self) -> Dict[str, Any]:
        header = self._recv_exact(4)
        size = int.from_bytes(header, "big")
        payload = self._recv_exact(size)
        return json.loads(payload.decode("utf-8"), object_hook=_numpy_object_hook)

    def close(self) -> None:
        self.sock.close()

    def _recv_exact(self, size: int) -> bytes:
        chunks = []
        remaining = size
        while remaining > 0:
            chunk = self.sock.recv(remaining)
            if not chunk:
                raise ConnectionError("Socket closed while receiving data.")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)


def connect_json_socket(host: str, port: int, timeout: float | None = None) -> JsonSocket:
    sock = socket.create_connection((host, port), timeout=timeout)
    return JsonSocket(sock)


def listen_tcp(host: str, port: int) -> socket.socket:
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(1)
    return server_socket


def accept_json_socket(server_socket: socket.socket) -> Tuple[JsonSocket, Tuple[str, int]]:
    client_socket, addr = server_socket.accept()
    return JsonSocket(client_socket), addr
