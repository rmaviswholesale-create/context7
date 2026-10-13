"""Re-verify the installed SDK surface agy-proxy depends on, and report the security posture.

Run ``python -m agy_proxy.doctor`` after upgrading google-antigravity. It fails
loudly if an expected symbol is missing, so drift is caught before runtime. It then
prints the token / bind / workspace / preset posture implied by the current
environment (the same variables the server reads) and flags risky combinations; those
are warnings (exit 0) unless ``--strict`` is given.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path

from .config import ConfigError, Settings, is_loopback_host

_SENSITIVE_DIRS = (".ssh", ".aws", ".gnupg", ".config/gcloud", ".kube", ".docker")
_BROAD_ROOTS = {"/tmp", "/var", "/usr", "/etc", "/opt", "/srv", "/mnt", "/home", "/root", "/Users"}


def check() -> list[str]:
    problems: list[str] = []
    try:
        import google.antigravity as ag
        from google.antigravity import policy
        from google.antigravity.models import (
            GeminiAPIEndpoint,
            GeminiModelOptions,
            ModelTarget,
            ThinkingLevel,
        )
        from google.antigravity.types import (
            AgentBehavior,
            Text,
            Thought,
            ToolCall,
            ToolResult,
        )
    except Exception as exc:  # noqa: BLE001
        return [f"import failed: {exc!r}"]

    for name in ("Agent", "LocalAgentConfig", "CapabilitiesConfig", "BuiltinTools"):
        if not hasattr(ag, name):
            problems.append(f"missing google.antigravity.{name}")

    import inspect

    if not inspect.iscoroutinefunction(ag.Agent.chat):
        problems.append("Agent.chat is no longer a coroutine")
    if not (hasattr(ag.Agent, "__aenter__") and hasattr(ag.Agent, "__aexit__")):
        problems.append("Agent is no longer an async context manager")

    for pol in ("allow_all", "safe_defaults", "workspace_only", "deny_all", "confirm_run_command"):
        if not hasattr(policy, pol):
            problems.append(f"missing policy.{pol}")

    for t in (Text, Thought, ToolCall, ToolResult):
        if "text" not in getattr(t, "model_fields", {}) and t in (Text, Thought):
            problems.append(f"{t.__name__} lost its .text field")

    # Touch the optional-thinking path symbols.
    _ = (ModelTarget, GeminiAPIEndpoint, GeminiModelOptions, ThinkingLevel, AgentBehavior)
    return problems


def _broad_workspace(path: str) -> str | None:
    """A workspace an agent with write access should not be given, or None."""
    p = Path(path).resolve()
    if str(p) in _BROAD_ROOTS or len(p.parts) <= 2:
        return f"{p} is a top-level directory"
    sensitive = [d for d in _SENSITIVE_DIRS if (p / d).exists()]
    if sensitive:
        return f"{p} contains {', '.join(sensitive)}"
    return None


def posture(env: Mapping[str, str] | None = None) -> tuple[list[str], list[str]]:
    """(report lines, warnings) for the configuration in ``env`` (default: os.environ)."""
    from .presets import PresetRegistry

    env = os.environ if env is None else env
    try:
        s = Settings.from_env(env)
    except ConfigError as exc:
        return [], [f"configuration error: {exc}"]
    info: list[str] = []
    warn: list[str] = []

    loopback = is_loopback_host(s.host)
    info.append(f"bind: {s.host}:{s.port} ({'loopback' if loopback else 'NON-LOOPBACK'})")
    if not loopback:
        warn.append(
            f"listening on {s.host}: the agent acts as this user; use TLS, a firewall and a "
            "strong AGY_PROXY_TOKEN"
        )
    if env.get("AGY_PROXY_TOKEN", "").strip():
        n = len(env["AGY_PROXY_TOKEN"].strip())
        info.append(f"token: set ({n} chars)")
        if n < 24:
            warn.append("AGY_PROXY_TOKEN is short; use 32+ random characters")
    else:
        info.append("token: not set (a random one is generated and printed at each start)")
    info.append(f"allowed hosts: {', '.join(s.allowed_hosts)}")
    info.append(f"workspace: {s.workspace or '(none: read-only presets use an empty scratch dir)'}")
    info.append(
        f"limits: concurrency={s.max_concurrency} turn_timeout={s.turn_timeout}s "
        f"max_body={s.max_body_bytes} bytes"
    )

    try:
        reg = PresetRegistry.load(
            s.presets_path, s.default_model, workspace=s.workspace, enable_coder=s.enable_coder
        )
    except Exception as exc:  # noqa: BLE001
        warn.append(f"presets failed to load: {exc}")
        return info, warn
    info.append(f"presets enabled: {', '.join(reg.names())} (default {reg.default_name})")
    for name, why in reg.disabled.items():
        info.append(f"preset disabled: {name} ({why})")

    risky = [n for n in reg.names() if reg.resolve(n).needs_workspace]
    if risky:
        warn.append(
            f"write-capable presets enabled: {', '.join(risky)} -- they can edit files and run "
            "commands as this user anywhere the OS allows, not only inside the workspace"
        )
        for n in risky:
            preset = reg.resolve(n)
            if preset.policy == "allow_all":
                spaces = preset.workspaces or ([s.workspace] if s.workspace else [])
                for ws in spaces:
                    broad = _broad_workspace(ws)
                    if broad:
                        warn.append(f"preset {n} (allow_all) has a broad workspace: {broad}")
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        warn.append("running as root: run the proxy as an unprivileged user or in a container")
    readers = [n for n in reg.names() if not reg.resolve(n).needs_workspace]
    if any(
        t in (reg.resolve(n).enabled_tools or [])
        for n in readers
        for t in ("VIEW_FILE", "SEARCH_DIR", "FIND_FILE", "LIST_DIR")
    ):
        warn.append(
            "read tools are not path-confined (unproven in SDK 0.1.21): read-only presets such "
            "as agy-researcher may read anything this user can"
        )
    return info, warn


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    problems = check()
    if problems:
        print("SDK verification FAILED:")
        for p in problems:
            print("  -", p)
        return 1
    print("SDK verification OK: google-antigravity surface matches agy-proxy expectations.")
    info, warnings = posture()
    print("\nPosture:")
    for line in info:
        print("  -", line)
    for w in warnings:
        print("  WARNING:", w)
    return 1 if warnings and "--strict" in args else 0


if __name__ == "__main__":
    sys.exit(main())
