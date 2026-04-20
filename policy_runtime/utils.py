from __future__ import annotations

import importlib
from typing import Any, Type


def import_symbol(path: str) -> Any:
    try:
        module_name, symbol_name = path.split(":", 1)
    except ValueError as exc:
        raise ValueError(f"Invalid import path '{path}'. Expected format 'module.submodule:Symbol'.") from exc

    module = importlib.import_module(module_name)
    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:
        raise AttributeError(f"Module '{module_name}' has no symbol '{symbol_name}'.") from exc


def create_instance(path: str, expected_type: Type[Any], **kwargs: Any) -> Any:
    symbol = import_symbol(path)
    instance = symbol(**kwargs)
    if not isinstance(instance, expected_type):
        raise TypeError(f"Loaded instance from '{path}' is not a '{expected_type.__name__}'.")
    return instance


def parse_kwargs_json(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}

    import json

    data = json.loads(raw)
    if not isinstance(data, dict):
        raise TypeError("Adapter kwargs must decode to a JSON object.")
    return data
