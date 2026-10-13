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

import anyio
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
from google.antigravity.types import AgentBehavior, BudgetConfig

from .events import Event
from .lifecycle import AgentLifecycle, agent_usable
from .messages import flatten_messages
from .presets import Preset, PresetRegistry
from .runner_events import run_chat
from .scratch import Scratch
from .security import env_allowed
from .sessionkey import select_turn_messages

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
# A session agent lives across turns; its lifetime budget is this many per-request budgets.
_SESSION_BUDGET_TURNS = 20


class AgentRunner:
    """Turns resolved presets + messages into a normalized event stream."""

    def __init__(
        self,
        registry: PresetRegistry,
        api_key: str | None,
        workspace: str | None = None,
        *,
        teardown_timeout: float = 10.0,
        budget: BudgetConfig | None = None,
        env_passthrough: tuple[str, ...] = (),
    ):
        self._env_extra = env_passthrough
        self._registry = registry
        self._api_key = api_key
        self._workspace = workspace
        self._budget = budget
        self.lifecycle = AgentLifecycle(teardown_timeout)

    @staticmethod
    def budget_from(max_model_calls: int, max_tool_calls: int) -> BudgetConfig | None:
        """Per-request agent budget from settings (0 = unlimited)."""
        limits = {"max_model_calls": max_model_calls, "max_tool_calls": max_tool_calls}
        limits = {k: v for k, v in limits.items() if v}
        return BudgetConfig(**limits) if limits else None

    # -- lifecycle (thin delegates; see lifecycle.AgentLifecycle) -------------

    async def open_agent(self, agent: Agent) -> None:
        await self.lifecycle.open(agent)

    async def close_agent(self, agent: Agent) -> None:
        await self.lifecycle.close(agent)

    def is_usable(self, agent: Agent) -> bool:
        return agent_usable(agent)

    async def reap(self) -> int:
        return await self.lifecycle.reap()

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
            budget_config=self._budget,
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
        """Harness environment, built from an ALLOWLIST (see ``security.env_allowed``).

        The SDK still MERGES this over ``os.environ`` (it cannot remove variables), so the
        proxy process itself is scrubbed to the same allowlist at startup; building the dict
        explicitly keeps the intent independent of that and of anything set later.
        With Vertex/ADC (no API key) the real HOME is kept: ADC credentials live there.
        """
        env = {k: v for k, v in os.environ.items() if env_allowed(k, self._env_extra)}
        env.setdefault("PATH", os.defpath)
        env["TMPDIR"] = scratch.tmp
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

    def _session_budget(self) -> BudgetConfig | None:
        """Budgets are lifetime-scoped in the SDK, so a long-lived agent gets a scaled one."""
        if self._budget is None:
            return None
        scaled = {
            name: value * _SESSION_BUDGET_TURNS
            for name in ("max_model_calls", "max_tool_calls")
            if (value := getattr(self._budget, name)) is not None
        }
        return self._budget.model_copy(update=scaled)

    def new_agent(
        self, preset: Preset, system_extra: str | None = None, *, long_lived: bool = False
    ) -> Agent:
        """Create an un-entered Agent. Open it with ``open_agent``, release with ``close_agent``
        (which also removes its scratch tree); never ``async with`` it inside a generator."""
        scratch = Scratch()
        try:
            config = self.build_config(preset, system_extra, scratch)
            if long_lived:
                config.budget_config = self._session_budget()
            agent = Agent(config)
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
            await self.open_agent(agent)
            async for ev in self._run(agent, prompt):
                yield ev
        finally:
            # Runs on normal end, error, GeneratorExit and cancellation alike; the shield keeps
            # anyio's level-triggered cancellation from interrupting the teardown awaits.
            with anyio.CancelScope(shield=True):
                await self.close_agent(agent)

    async def stream_on_agent(
        self, agent: Agent, messages: list[dict[str, str]], is_new: bool = False
    ) -> AsyncIterator[Event]:
        """Stateful turn on an already-open agent.

        ``is_new`` (SessionLease.is_new): a fresh agent gets the client's full history
        replayed; a reused one only the messages after its last assistant message.
        """
        _, prompt = flatten_messages(select_turn_messages(messages, is_new))
        try:
            async for ev in self._run(agent, prompt):
                yield ev
        finally:
            # Consumer stopped early (disconnect/cancel): halt the turn, never leave it
            # running inside a cached agent. The session store also evicts such agents.
            with anyio.CancelScope(shield=True):
                await self.lifecycle.halt_turn(agent)

    def _run(self, agent: Agent, prompt: str) -> AsyncIterator[Event]:
        return run_chat(agent, prompt)


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
