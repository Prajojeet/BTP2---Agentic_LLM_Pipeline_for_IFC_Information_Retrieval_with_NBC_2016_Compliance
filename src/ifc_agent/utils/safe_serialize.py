"""Serialize IFCOpenShell / numpy values into JSON-safe Python primitives.

Tools must return JSON-serialisable values so that LangChain can stuff them
into ToolMessage.content without surprises.
"""
from __future__ import annotations

from typing import Any


def to_jsonable(value: Any, _depth: int = 0, _max_depth: int = 6) -> Any:
    """Best-effort recursive conversion to JSON-safe primitives.

    - IFC entities → {"GlobalId": ..., "type": ..., "Name": ...}
    - numpy scalars/arrays → python ints/floats/lists
    - bytes → utf-8 str (replacement on errors)
    - everything else → fallback to str()
    """
    if _depth > _max_depth:
        return f"<truncated:{type(value).__name__}>"

    # Cheap base cases
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    # numpy
    try:
        import numpy as _np  # local import keeps the cold path light
    except ImportError:
        _np = None  # type: ignore[assignment]
    if _np is not None:
        if isinstance(value, _np.generic):
            return value.item()
        if isinstance(value, _np.ndarray):
            return [to_jsonable(x, _depth + 1, _max_depth) for x in value.tolist()]

    # IFC entities — duck-type on .is_a() to avoid a hard import cycle
    if hasattr(value, "is_a") and callable(value.is_a):
        try:
            entity_type = value.is_a()
        except Exception:
            entity_type = type(value).__name__
        out: dict[str, Any] = {"type": entity_type}
        gid = getattr(value, "GlobalId", None)
        if gid:
            out["GlobalId"] = gid
        name = getattr(value, "Name", None)
        if name:
            out["Name"] = name
        return out

    # Containers
    if isinstance(value, dict):
        return {str(k): to_jsonable(v, _depth + 1, _max_depth) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(v, _depth + 1, _max_depth) for v in value]

    # Fallback
    return str(value)


def truncate_list(items: list, limit: int = 200) -> dict:
    """Trim large lists for safe display in the agent context window."""
    n = len(items)
    if n <= limit:
        return {"count": n, "items": items}
    return {
        "count": n,
        "truncated": True,
        "showing": limit,
        "items": items[:limit],
    }
