"""FindingsStore — session-scoped in-memory store for compliance findings.

A *finding* is the structured result of a single compliance check (one tool call):
verdict (pass/fail/indeterminate), the cited code clause, the parameters that
were checked, and the list of elements that failed (if any).

Findings are accumulated during a session so that follow-up questions like
"show me the rooms that failed travel distance" can be answered without
re-running the check.

Session-only by design — the store is created in ``app.py`` per Streamlit
session and discarded when the session ends.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Optional

Verdict = Literal["pass", "fail", "indeterminate"]


@dataclass
class Finding:
    """One compliance check result."""

    finding_id: str
    check_name: str
    clause: str
    verdict: Verdict
    summary: str
    params_checked: dict[str, Any] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FindingsStore:
    """Thread-safe per-session store for ``Finding`` objects.

    The store is shared between compliance tools (which write to it) and
    retrieval tools / the UI (which read from it).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._findings: list[Finding] = []

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------
    def record(
        self,
        *,
        check_name: str,
        clause: str,
        verdict: Verdict,
        summary: str,
        params_checked: Optional[dict[str, Any]] = None,
        failures: Optional[list[dict[str, Any]]] = None,
        extras: Optional[dict[str, Any]] = None,
    ) -> Finding:
        """Persist a new finding and return it."""
        f = Finding(
            finding_id=f"F-{uuid.uuid4().hex[:8].upper()}",
            check_name=check_name,
            clause=clause,
            verdict=verdict,
            summary=summary,
            params_checked=params_checked or {},
            failures=failures or [],
            extras=extras or {},
        )
        with self._lock:
            self._findings.append(f)
        return f

    def clear(self) -> int:
        """Drop all findings; returns how many were removed."""
        with self._lock:
            n = len(self._findings)
            self._findings.clear()
        return n

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def all(self) -> list[Finding]:
        with self._lock:
            return list(self._findings)

    def get(self, finding_id: str) -> Optional[Finding]:
        with self._lock:
            for f in self._findings:
                if f.finding_id == finding_id:
                    return f
        return None

    def by_check(self, check_name: str) -> list[Finding]:
        with self._lock:
            return [f for f in self._findings if f.check_name == check_name]

    def __len__(self) -> int:  # pragma: no cover
        with self._lock:
            return len(self._findings)


__all__ = ["Finding", "FindingsStore", "Verdict"]
