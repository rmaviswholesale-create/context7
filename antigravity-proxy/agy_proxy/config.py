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


# Large enough for long agentic transcripts (Claude Code resends the whole history).
DEFAULT_MAX_BODY = 32 * 1024 * 1024

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


def _home_dirs(env: Mapping[str, str] | None = None) -> set[Path]:
    """Every plausible 'home' of the proxy user (real paths)."""
    homes = {Path.home()}
    if env and env.get("HOME"):
        homes.add(Path(env["HOME"]))
    try:
        import pwd

        homes.add(Path(pwd.getpwuid(os.getuid()).pw_dir))
    except (ImportError, KeyError):
        pass
    return {h.resolve() for h in homes if str(h)}


def broad_workspace_reason(path: str | Path, env: Mapping[str, str] | None = None) -> str | None:
    """Why ``path`` is too broad to be a workspace (``/``, the home dir or an ancestor)."""
    real = Path(path).resolve()
    if real == Path(real.anchor):
        return "the filesystem root"
    for home in _home_dirs(env):
        if real == home:
            return f"the home directory {str(home)!r}"
        if real in home.parents:
            return f"a parent of the home directory {str(home)!r}"
    return None


def check_workspace_path(raw: str, label: str, env: Mapping[str, str] | None = None) -> str:
    """Validate a workspace path and return its resolved form (symlinks followed)."""
    path = Path(raw)
    if not path.is_absolute():
        raise ConfigError(f"{label}={raw!r} must be an absolute path")
    if not path.is_dir():
        raise ConfigError(f"{label}={raw!r} is not an existing directory")
    broad = broad_workspace_reason(path, env)
    if broad:
        resolved = f" (resolves to {str(path.resolve())!r})" if path.resolve() != path else ""
        raise ConfigError(
            f"{label}={raw!r}{resolved} is {broad}: choose a dedicated project directory"
        )
    return str(path.resolve())


def _workspace(env: Mapping[str, str]) -> str | None:
    raw = (env.get("AGY_WORKSPACE") or "").strip()
    return check_workspace_path(raw, "AGY_WORKSPACE", env) if raw else None


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
    # P2 runtime limits.
    max_concurrency: int = 4  # concurrent completions in flight; the next one gets 429
    turn_timeout: int = 600  # seconds for one request, then 504
    max_body_bytes: int = DEFAULT_MAX_BODY  # request body cap, then 413
    teardown_timeout: int = 10  # seconds a harness gets to exit before SIGTERM/SIGKILL
    # Per-request agent budget (0 disables); long-lived session agents get a scaled budget.
    max_model_calls: int = 25
    max_tool_calls: int = 50
    log_level: str = "info"
    # Extra environment variable names the harness may inherit (AGY_ENV_PASSTHROUGH).
    env_passthrough: tuple[str, ...] = ()
    # --- P4a runtime hardening (appended block) ---
    spawn_timeout: int = 30  # seconds the harness gets to complete startup, then kill + 502
    body_timeout: int = 60  # seconds to receive a request body before 408 (no permit is held)
    keepalive_timeout: int = 5  # uvicorn timeout_keep_alive (idle keep-alive sockets)
    limit_concurrency: int = 0  # uvicorn limit_concurrency; 0 = derived (see property)
    # --- end P4a block ---

    @property
    def uvicorn_limit_concurrency(self) -> int:
        """Open-connection cap for uvicorn (503 beyond it); well above the agent cap so cheap
        endpoints and 429 replies keep working under load."""
        return self.limit_concurrency or max(32, self.max_concurrency * 8)

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
            max_concurrency=_int(env, "AGY_MAX_CONCURRENCY", 4, 1),
            turn_timeout=_int(env, "AGY_TURN_TIMEOUT", 600, 1),
            max_body_bytes=_int(env, "AGY_MAX_BODY", DEFAULT_MAX_BODY, 1024),
            teardown_timeout=_int(env, "AGY_TEARDOWN_TIMEOUT", 10, 1),
            max_model_calls=_int(env, "AGY_MAX_MODEL_CALLS", 25, 0),
            max_tool_calls=_int(env, "AGY_MAX_TOOL_CALLS", 50, 0),
            log_level=env.get("AGY_LOG_LEVEL", "info"),
            env_passthrough=tuple(
                n.strip() for n in env.get("AGY_ENV_PASSTHROUGH", "").split(",") if n.strip()
            ),
            # --- P4a runtime hardening (appended block) ---
            spawn_timeout=_int(env, "AGY_SPAWN_TIMEOUT", 30, 1),
            body_timeout=_int(env, "AGY_BODY_TIMEOUT", 60, 1),
            keepalive_timeout=_int(env, "AGY_KEEPALIVE_TIMEOUT", 5, 1),
            limit_concurrency=_int(env, "AGY_LIMIT_CONCURRENCY", 0, 0),
            # --- end P4a block ---
        )
