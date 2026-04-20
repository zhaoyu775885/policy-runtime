from __future__ import annotations

from typing import Any, Dict


MESSAGE_START_EPISODE = "start_episode"
MESSAGE_GET_ACTION = "get_action"
MESSAGE_END_EPISODE = "end_episode"


def make_request(message_type: str, episode_id: str, data: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "type": message_type,
        "episode_id": episode_id,
        "data": data or {},
    }


def make_success_response(data: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "ok": True,
        "data": data or {},
    }


def make_error_response(error: str) -> Dict[str, Any]:
    return {
        "ok": False,
        "error": error,
    }

