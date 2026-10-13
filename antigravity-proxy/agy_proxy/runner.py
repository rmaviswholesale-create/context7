"""AgentRunner -- the single point of contact with the Antigravity SDK.

Builds a ``LocalAgentConfig`` from a resolved preset, runs ``agent.chat()``, and
normalizes the SDK's ``ChatResponse.chunks`` stream into the vocabulary defined
in ``events.py``. Nothing else in the codebase imports the SDK.

See ``docs/SDK_NOTES.md`` for the verified SDK surface this relies on.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from google.antigravity import (
    Agent,
    BuiltinTools,
    CapabilitiesConfig,
    LocalAgentConfig,
    policy,
)
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

from .events import (
    DoneEvent,
    ErrorEvent,
    Event,
    TextEvent,
    ThoughtEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from .presets import Preset, PresetRegistry
from .scratch import Scratch

logger = logging.getLogger("agy_proxy.runner")

# Policy factories keyed by the friendly name used in presets.yaml.
#
# agy-proxy runs headless (no human to answer confirm prompts), so the SDK's
# interactive ``safe_defaults(handler)`` is unavailable. ``safe_defaults`` therefore maps
# to ``confirm_run_command()``, which -- with no handler -- denies RUN_COMMAND but
# ALLOWS everything else, including file writes. The preset's ``enabled_tools``
# allowlist (plus the explicit deny rules in ``_policies``) is the real barrier.
_POLICY_FACTORIES = {
    "allow_all": lambda: [policy.allow_all()],
    "deny_all": lambda: [policy.deny_all()],
    "safe_defaults": lambda: list(policy.confirm_run_command()),
    "confirm_run_command": lambda: list(policy.confirm_run_command()),
}

_THINKING_LEVELS = {
    "minimal": ThinkingLevel.MINIMAL,
    "low": ThinkingLevel.LOW,
    "medium": ThinkingLevel.MEDIUM,
    "high": ThinkingLevel.HIGH,
    "extra_high": ThinkingLevel.EXTRA_HIGH,
}


def _tool_enum(names: list[str] | None) -> list[BuiltinTools] | None:
    """Map names to BuiltinTools. ``None`` = SDK defaults; ``[]`` = no tools."""
    if names is None:
        return None
    out: list[BuiltinTools] = []
    for n in names:
        try:
            out.append(BuiltinTools[n.upper()])
        except KeyError as exc:
            raise ValueError(f"unknown builtin tool {n!r}") from exc
    return out


_WRITE_TOOLS = ("CREATE_FILE", "EDIT_FILE", "RUN_COMMAND")
_UNTRUSTED_LABEL = "[Untrusted client instructions]"


class AgentRunner:
    """Turns resolved presets + messages into a normalized event stream."""

    def __init__(
        self,
        registry: PresetRegistry,
        api_key: str | None,
        workspace: str | None = None,
    ):
        self._registry = registry
        self._api_key = api_key
        self._workspace = workspace

    # -- configuration -------------------------------------------------------

    def build_config(
        self,
        preset: Preset,
        system_extra: str | None = None,
        scratch: Scratch | None = None,
    ) -> LocalAgentConfig:
        """Construct a LocalAgentConfig from a preset (used for both modes).

        ``scratch`` owns save_dir/app_data_dir/HOME/TMPDIR; the caller must call
        ``scratch.cleanup()`` (a new one is created when omitted).
        """
        scratch = scratch or Scratch()
        system = _join_system(preset, system_extra)
        capabilities = CapabilitiesConfig(
            agent_behavior=AgentBehavior[preset.behavior.upper()],
            enabled_tools=_tool_enum(preset.enabled_tools),
            enable_subagents=preset.enable_subagents,
        )
        return LocalAgentConfig(
            system_instructions=system,
            capabilities=capabilities,
            policies=self._policies(preset),
            model=self._build_model(preset),
            api_key=self._api_key,
            workspaces=self._workspaces(preset, scratch),
            save_dir=scratch.save_dir,
            app_data_dir=scratch.app_data_dir,
            env=self._env(scratch),
        )

    def _workspaces(self, preset: Preset, scratch: Scratch) -> list[str]:
        """Explicit workspaces (the SDK turns ``None`` into the proxy's cwd)."""
        if preset.workspaces:
            return list(preset.workspaces)
        if self._workspace:
            return [self._workspace]
        if preset.needs_workspace:
            raise ValueError(f"preset {preset.name!r} needs AGY_WORKSPACE")
        return [scratch.workspace]

    @staticmethod
    def _policies(preset: Preset) -> list[Any]:
        policies: list[Any] = []
        if not preset.needs_workspace:
            # Defence in depth for read-only presets, on top of the tool allowlist.
            policies += [policy.deny(BuiltinTools[t].value) for t in _WRITE_TOOLS]
        policies += _POLICY_FACTORIES[preset.policy]()
        return policies

    def _env(self, scratch: Scratch) -> dict[str, str]:
        # The SDK MERGES this into os.environ (it cannot remove variables), so secrets
        # are scrubbed from the process at startup; here we only redirect scratch space.
        # With Vertex/ADC (no API key) the real HOME is kept: ADC credentials live there.
        env = {"PATH": os.environ.get("PATH", os.defpath), "TMPDIR": scratch.tmp}
        env["HOME"] = scratch.home if self._api_key else os.environ.get("HOME", scratch.home)
        return env

    def _build_model(self, preset: Preset) -> str | ModelTarget:
        model_name = self._registry.effective_model(preset)
        if preset.thinking_level:
            options = GeminiModelOptions(
                thinking_level=_THINKING_LEVELS[preset.thinking_level]
            )
            endpoint = GeminiAPIEndpoint(api_key=self._api_key, options=options)
            return ModelTarget(name=model_name, endpoint=endpoint)
        return model_name

    def new_agent(self, preset: Preset, system_extra: str | None = None) -> Agent:
        """Create an un-entered Agent (caller manages the async context and cleanup)."""
        scratch = Scratch()
        try:
            agent = Agent(self.build_config(preset, system_extra, scratch))
        except BaseException:
            scratch.cleanup()
            raise
        agent.scratch = scratch  # type: ignore[attr-defined]
        return agent

    # -- streaming -----------------------------------------------------------

    async def stream_stateless(
        self, preset: Preset, messages: list[dict[str, str]]
    ) -> AsyncIterator[Event]:
        """One-shot: build a fresh agent, replay the whole conversation, tear down."""
        system_extra, prompt = flatten_messages(messages)
        agent = self.new_agent(preset, system_extra)
        try:
            async with agent:
                async for ev in self._run(agent, prompt):
                    yield ev
        finally:
            scratch = getattr(agent, "scratch", None)
            if scratch is not None:
                scratch.cleanup()

    async def stream_on_agent(
        self, agent: Agent, messages: list[dict[str, str]]
    ) -> AsyncIterator[Event]:
        """Stateful: send only the latest user turn to an already-open agent."""
        prompt = latest_user_text(messages)
        async for ev in self._run(agent, prompt):
            yield ev

    async def _run(self, agent: Agent, prompt: str) -> AsyncIterator[Event]:
        try:
            resp = await agent.chat(prompt)
            async for chunk in resp.chunks:
                if isinstance(chunk, Text):
                    if chunk.text:
                        yield TextEvent(chunk.text)
                elif isinstance(chunk, Thought):
                    if chunk.text:
                        yield ThoughtEvent(chunk.text)
                elif isinstance(chunk, ToolCall):
                    yield ToolCallEvent(
                        name=_tool_name(chunk.name), args=dict(chunk.args or {}), id=chunk.id
                    )
                elif isinstance(chunk, ToolResult):
                    yield ToolResultEvent(
                        name=_tool_name(chunk.name), id=chunk.id, error=chunk.error
                    )
            usage = getattr(resp, "usage_metadata", None)
            yield DoneEvent(
                stop_reason=_stop_reason(getattr(resp, "stop_reason", None)),
                input_tokens=getattr(usage, "prompt_token_count", None),
                output_tokens=getattr(usage, "candidates_token_count", None),
            )
        except Exception as exc:  # noqa: BLE001 - surface any SDK failure to the client
            yield ErrorEvent(message=str(exc) or repr(exc), type=type(exc).__name__)


def _tool_name(name: Any) -> str:
    return name.name if isinstance(name, BuiltinTools) else str(name)


def _stop_reason(reason: Any) -> str | None:
    if reason is None:
        return None
    return getattr(reason, "name", str(reason))


def _join_system(preset: Preset, client_system: str | None) -> str | None:
    """Preset text, plus client system text only when the preset allows it.

    Client text is untrusted: it is labelled and placed BEFORE the preset's own
    instructions so the preset's rules are the last (most recent) word.
    """
    parts: list[str] = []
    if client_system:
        if preset.allow_client_system:
            parts.append(f"{_UNTRUSTED_LABEL}\n{client_system}")
        else:
            logger.debug("ignoring client system text for preset %r", preset.name)
    if preset.system_instructions:
        parts.append(preset.system_instructions)
    return "\n\n".join(parts) if parts else None


# -- message flattening ------------------------------------------------------


def _msg_text(m: dict[str, str]) -> str:
    return (m.get("content") or "").strip()


def latest_user_text(messages: list[dict[str, str]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return _msg_text(m)
    # Fallback: last message of any role.
    return _msg_text(messages[-1]) if messages else ""


def flatten_messages(messages: list[dict[str, str]]) -> tuple[str | None, str]:
    """Split messages into (extra system text, prompt).

    System messages are folded into the system instructions. When there is a
    single user turn, the prompt is just that text. With prior turns, a labelled
    transcript is built so a fresh (stateless) agent has the full context.
    """
    system_parts = [_msg_text(m) for m in messages if m.get("role") == "system"]
    convo = [m for m in messages if m.get("role") in ("user", "assistant", "tool")]
    system_extra = "\n\n".join(p for p in system_parts if p) or None

    if not convo:
        return system_extra, ""
    if len(convo) == 1 and convo[0].get("role") == "user":
        return system_extra, _msg_text(convo[0])

    lines: list[str] = []
    for m in convo:
        role = m.get("role")
        label = {"user": "User", "assistant": "Assistant", "tool": "Tool"}.get(role, role)
        text = _msg_text(m)
        if text:
            lines.append(f"{label}: {text}")
    transcript = "\n\n".join(lines)
    prompt = (
        "Continue this conversation. Respond only as the assistant's next turn.\n\n"
        + transcript
    )
    return system_extra, prompt
