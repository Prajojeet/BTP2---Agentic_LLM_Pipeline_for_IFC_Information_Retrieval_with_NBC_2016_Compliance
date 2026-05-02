"""Runtime configuration for ifc_agent.

All knobs live here so the rest of the codebase doesn't read os.environ directly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env once, at import time. Safe to call repeatedly.
load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    model: str
    temperature: float
    max_iterations: int

    @classmethod
    def from_env(cls) -> "Settings":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        return cls(
            openai_api_key=api_key,
            model=os.getenv("IFC_AGENT_MODEL", "gpt-5.4").strip(),
            temperature=float(os.getenv("IFC_AGENT_TEMPERATURE", "0")),
            max_iterations=int(os.getenv("IFC_AGENT_MAX_ITERATIONS", "25")),
        )


def get_settings() -> Settings:
    """Cached settings accessor."""
    if not hasattr(get_settings, "_cache"):
        get_settings._cache = Settings.from_env()  # type: ignore[attr-defined]
    return get_settings._cache  # type: ignore[attr-defined]
