"""Runtime configuration for agy-proxy (env-driven, no secrets in code).

Settings is the only holder of secrets: the process environment is scrubbed once
at startup (see ``security.scrub_process_env``), so the Antigravity harness and any
shell it spawns never inherit the API key or the proxy token.
"""

from __future__ import annotations

import ipaddress
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path


class ConfigError(ValueError):
    """Invalid configuration; the message names the offending environment variable."""


def _default_presets_path() -> str:
    # presets.yaml lives at the project root, next to this package.
    return str(Path(__file__).resolve().parent.parent / "presets.yaml")


def is_loopback_host(host: str) -> bool:
    h = host.strip().strip("[]").lower()
    if h == "localhost":
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


def _bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None:
        return default
    val = raw.strip().lower()
    if val in _TRUE:
        return True
    if val in _FALSE:
        return False
    raise ConfigError(f"{name}={raw!r} is not a boolean (use 1/0, true/false)")


def _int(env: Mapping[str, str], name: str, default: int, minimum: int = 0) -> int:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        val = int(raw.strip())
    except ValueError as exc:
        raise ConfigError(f"{name}={raw!r} is not an integer") from exc
    if val < minimum:
        raise ConfigError(f"{name}={val} is below the minimum of {minimum}")
    return val


def _workspace(env: Mapping[str, str]) -> str | None:
    raw = (env.get("AGY_WORKSPACE") or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        raise ConfigError(f"AGY_WORKSPACE={raw!r} must be an absolute path")
    if not path.is_dir():
        raise ConfigError(f"AGY_WORKSPACE={raw!r} is not an existing directory")
    return str(path.resolve())


def _hosts(env: Mapping[str, str], bind_host: str) -> tuple[str, ...]:
    raw = env.get("AGY_ALLOWED_HOSTS")
    if raw is not None and raw.strip():
        hosts = [h.strip().lower() for h in raw.split(",") if h.strip()]
        for h in hosts:
            if "/" in h or " " in h:
                raise ConfigError(f"AGY_ALLOWED_HOSTS entry {h!r} is not a bare host name")
        return tuple(hosts)
    hosts = ["127.0.0.1", "localhost", "[::1]"]
    bind = bind_host.strip().lower()
    if bind not in ("0.0.0.0", "::", "") and bind.strip("[]") not in ("127.0.0.1", "::1"):
        hosts.append(f"[{bind.strip('[]')}]" if ":" in bind else bind)
    return tuple(dict.fromkeys(hosts))


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8081
    # Secrets: kept out of repr and out of the process environment.
    api_key: str | None = field(default=None, repr=False)
    proxy_token: str | None = field(default=None, repr=False)
    # True when the token was generated at startup (and must be shown once).
    token_generated: bool = False
    # Absolute directory write-capable presets may operate in (AGY_WORKSPACE).
    workspace: str | None = None
    # Accepted Host header values (DNS-rebinding defence).
    allowed_hosts: tuple[str, ...] = ("127.0.0.1", "localhost", "[::1]")
    # Opt in to the agy-coder preset (AGY_ENABLE_CODER=1).
    enable_coder: bool = False
    # Log tool calls/results the agent makes (never sent to clients).
    enable_tool_trace: bool = True
    presets_path: str = field(default_factory=_default_presets_path)
    default_model: str = "gemini-3-pro"
    # Live-Agent session cache (stateful mode).
    session_ttl_seconds: int = 900
    session_max: int = 64
    log_level: str = "info"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        host = env.get("AGY_HOST", "127.0.0.1")
        token = (env.get("AGY_PROXY_TOKEN") or "").strip() or None
        generated = False
        if token is None and is_loopback_host(host):
            from .security import make_token  # local import: security imports config

            token, generated = make_token(), True
        return cls(
            host=host,
            port=_int(env, "AGY_PORT", 8081, 1),
            api_key=env.get("AGY_API_KEY") or env.get("GEMINI_API_KEY") or None,
            proxy_token=token,
            token_generated=generated,
            workspace=_workspace(env),
            allowed_hosts=_hosts(env, host),
            enable_coder=_bool(env, "AGY_ENABLE_CODER", False),
            enable_tool_trace=_bool(env, "AGY_LOG_TOOLS", True),
            presets_path=env.get("AGY_PRESETS", _default_presets_path()),
            default_model=env.get("AGY_DEFAULT_MODEL", "gemini-3-pro"),
            session_ttl_seconds=_int(env, "AGY_SESSION_TTL", 900, 1),
            session_max=_int(env, "AGY_SESSION_MAX", 64, 1),
            log_level=env.get("AGY_LOG_LEVEL", "info"),
        )
