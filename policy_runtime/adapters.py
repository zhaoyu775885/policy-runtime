from __future__ import annotations

from typing import Any, Dict


class EnvAdapter:
    """Environment-side adapter interface for the minimal v1 runtime."""

    def reset(self) -> Dict[str, Any]:
        raise NotImplementedError

    def get_observation(self) -> Dict[str, Any]:
        raise NotImplementedError

    def apply_action(self, action: Any) -> None:
        raise NotImplementedError

    def is_done(self) -> bool:
        raise NotImplementedError

    def get_result(self) -> Dict[str, Any]:
        return {}

    def close(self) -> None:
        return None


class PolicyAdapter:
    """Policy-side adapter interface for the minimal v1 runtime."""

    def start_episode(self, episode_info: Dict[str, Any]) -> None:
        raise NotImplementedError

    def get_action(self, observation: Dict[str, Any]) -> Any:
        raise NotImplementedError

    def end_episode(self) -> None:
        raise NotImplementedError
