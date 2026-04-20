from __future__ import annotations

import argparse
import traceback
from typing import Any, Dict

from policy_runtime.adapters import PolicyAdapter
from policy_runtime.protocol import (
    MESSAGE_END_EPISODE,
    MESSAGE_GET_ACTION,
    MESSAGE_START_EPISODE,
    make_error_response,
    make_success_response,
)
from policy_runtime.transport import accept_json_socket, listen_tcp
from policy_runtime.utils import create_instance, parse_kwargs_json


def handle_message(adapter: PolicyAdapter, message: Dict[str, Any]) -> Dict[str, Any]:
    message_type = message.get("type")
    data = message.get("data", {})

    if message_type == MESSAGE_START_EPISODE:
        adapter.start_episode(data)
        return make_success_response()

    if message_type == MESSAGE_GET_ACTION:
        observation = data.get("observation", {})
        action = adapter.get_action(observation)
        return make_success_response({"action": action})

    if message_type == MESSAGE_END_EPISODE:
        adapter.end_episode()
        return make_success_response()

    return make_error_response(f"Unsupported message type: {message_type}")


def serve_forever(adapter: PolicyAdapter, host: str, port: int) -> None:
    server_socket = listen_tcp(host=host, port=port)
    try:
        while True:
            conn, addr = accept_json_socket(server_socket)
            try:
                print(f"Client connected from {addr[0]}:{addr[1]}")
                while True:
                    try:
                        message = conn.recv()
                    except ConnectionError:
                        break

                    try:
                        response = handle_message(adapter, message)
                    except Exception as exc:  # pragma: no cover - runtime guard
                        traceback.print_exc()
                        response = make_error_response(str(exc))

                    conn.send(response)
            finally:
                conn.close()
    finally:
        server_socket.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal policy server runtime.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--adapter", type=str, required=True, help="Import path: module.submodule:Class")
    parser.add_argument("--adapter-kwargs", type=str, default=None, help="JSON object passed to adapter constructor")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter_kwargs = parse_kwargs_json(args.adapter_kwargs)
    adapter = create_instance(args.adapter, PolicyAdapter, **adapter_kwargs)
    serve_forever(adapter=adapter, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
