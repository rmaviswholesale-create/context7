"""Maps the SDK's ``ChatResponse.chunks`` stream onto the normalized events.

Split out of ``runner.py`` so the event mapping can be tested (and evolve)
independently of agent spawn/teardown. Tool calls and results are executed by the
agent itself and are never part of the wire protocol: they are only logged, and
only when tool tracing is enabled (``AGY_LOG_TOOLS``).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from google.antigravity import BuiltinTools
from google.antigravity.types import Text, Thought, ToolCall, ToolResult

from .errors import RateLimited
from .events import DoneEvent, ErrorEvent, Event, TextEvent, ThoughtEvent

logger = logging.getLogger("agy_proxy.runner")
tool_logger = logging.getLogger("agy_proxy.tools")

_trace_tools = True
_CUT_SHORT = frozenset({"MAX_MODEL_CALLS_EXCEEDED", "MAX_TOOL_CALLS_EXCEEDED"})


def configure(*, trace_tools: bool) -> None:
    """Enable or disable logging of the agent's tool activity."""
    global _trace_tools
    _trace_tools = trace_tools


async def run_chat(agent: Any, prompt: str) -> AsyncIterator[Event]:
    """Send ``prompt`` to an open agent and yield normalized events.

    Always ends with exactly one terminal event (Done or Error) unless the consumer
    stops early. QUOTA_EXHAUSTED becomes a RateLimited error instead of a Done.
    """
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
                if _trace_tools:
                    tool_logger.info("tool call: %s id=%s", tool_name(chunk.name), chunk.id)
            elif isinstance(chunk, ToolResult):
                if _trace_tools:
                    tool_logger.info(
                        "tool result: %s id=%s error=%s",
                        tool_name(chunk.name),
                        chunk.id,
                        bool(chunk.error),
                    )
        reason = stop_reason_name(getattr(resp, "stop_reason", None))
        if reason == "QUOTA_EXHAUSTED":
            yield ErrorEvent(RateLimited("upstream model quota exhausted"))
            return
        if reason in _CUT_SHORT:
            logger.warning("turn cut short by agent budget: %s", reason)
        usage = getattr(resp, "usage_metadata", None)
        yield DoneEvent(
            stop_reason=reason,
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
        )
    except Exception as exc:  # noqa: BLE001 - surface any SDK failure to the client
        yield ErrorEvent.of(exc)


def tool_name(name: Any) -> str:
    return name.name if isinstance(name, BuiltinTools) else str(name)


def stop_reason_name(reason: Any) -> str | None:
    if reason is None:
        return None
    return getattr(reason, "name", str(reason))
