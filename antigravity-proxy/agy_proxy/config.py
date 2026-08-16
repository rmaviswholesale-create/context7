"""Runtime configuration for agy-proxy (env-driven, no secrets in code)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_presets_path() -> str:
    # presets.yaml lives at the project root, next to this package.
    return str(Path(__file__).resolve().parent.parent / "presets.yaml")


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8081
    # The Gemini/Antigravity API key handed to the SDK. Your own key.
    api_key: str | None = None
    # Optional bearer token clients must present (defence for a shared LAN bind).
    proxy_token: str | None = None
    presets_path: str = field(default_factory=_default_presets_path)
    default_model: str = "gemini-3-pro"
    # Live-Agent session cache (stateful mode).
    session_ttl_seconds: int = 900
    session_max: int = 64
    log_level: str = "info"

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            host=os.getenv("AGY_HOST", "127.0.0.1"),
            port=int(os.getenv("AGY_PORT", "8081")),
            api_key=os.getenv("AGY_API_KEY") or os.getenv("GEMINI_API_KEY"),
            proxy_token=os.getenv("AGY_PROXY_TOKEN") or None,
            presets_path=os.getenv("AGY_PRESETS", _default_presets_path()),
            default_model=os.getenv("AGY_DEFAULT_MODEL", "gemini-3-pro"),
            session_ttl_seconds=int(os.getenv("AGY_SESSION_TTL", "900")),
            session_max=int(os.getenv("AGY_SESSION_MAX", "64")),
            log_level=os.getenv("AGY_LOG_LEVEL", "info"),
        )
