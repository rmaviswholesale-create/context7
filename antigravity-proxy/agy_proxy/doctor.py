"""Re-verify the installed SDK surface agy-proxy depends on.

Run ``python -m agy_proxy.doctor`` after upgrading google-antigravity. It fails
loudly if an expected symbol is missing, so drift is caught before runtime.
"""

from __future__ import annotations

import sys


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


def main() -> int:
    problems = check()
    if problems:
        print("SDK verification FAILED:")
        for p in problems:
            print("  -", p)
        return 1
    print("SDK verification OK: google-antigravity surface matches agy-proxy expectations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
