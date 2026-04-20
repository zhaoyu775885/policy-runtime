from __future__ import annotations

import argparse
import uuid
from typing import Any, Dict

from policy_runtime.adapters import EnvAdapter
from policy_runtime.protocol import (
    MESSAGE_END_EPISODE,
    MESSAGE_GET_ACTION,
    MESSAGE_START_EPISODE,
    make_request,
)
from policy_runtime.transport import connect_json_socket
from policy_runtime.utils import create_instance, parse_kwargs_json


def _expect_ok(response: Dict[str, Any]) -> Dict[str, Any]:
    if response.get("ok"):
        return response.get("data", {})
    raise RuntimeError(response.get("error", "Unknown server error."))


def run_client(
    adapter: EnvAdapter,
    host: str,
    port: int,
    num_episodes: int = 1,
    timeout: float | None = None,
) -> None:
    conn = connect_json_socket(host=host, port=port, timeout=timeout)
    try:
        for _ in range(num_episodes):
            episode_id = str(uuid.uuid4())
            episode_info = adapter.reset() or {}

            conn.send(make_request(MESSAGE_START_EPISODE, episode_id, episode_info))
            _expect_ok(conn.recv())

            while not adapter.is_done():
                observation = adapter.get_observation()
                conn.send(
                    make_request(
                        MESSAGE_GET_ACTION,
                        episode_id,
                        {"observation": observation},
                    )
                )
                response_data = _expect_ok(conn.recv())
                action = response_data["action"]
                adapter.apply_action(action)

            result = adapter.get_result() or {}
            conn.send(make_request(MESSAGE_END_EPISODE, episode_id, {"result": result}))
            _expect_ok(conn.recv())
    finally:
        adapter.close()
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal env client runtime.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--adapter", type=str, required=True, help="Import path: module.submodule:Class")
    parser.add_argument("--adapter-kwargs", type=str, default=None, help="JSON object passed to adapter constructor")
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter_kwargs = parse_kwargs_json(args.adapter_kwargs)
    adapter = create_instance(args.adapter, EnvAdapter, **adapter_kwargs)
    run_client(
        adapter=adapter,
        host=args.host,
        port=args.port,
        num_episodes=args.num_episodes,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
