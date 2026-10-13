"""Maps the SDK's ``ChatResponse.chunks`` stream onto the normalized events.

Split out of ``runner.py`` so the event mapping can be tested (and evolve)
independently of agent spawn/teardown.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from google.antigravity import BuiltinTools
from google.antigravity.types import Text, Thought, ToolCall, ToolResult

from .events import (
    DoneEvent,
    ErrorEvent,
    Event,
    TextEvent,
    ThoughtEvent,
    ToolCallEvent,
    ToolResultEvent,
)


async def run_chat(agent: Any, prompt: str) -> AsyncIterator[Event]:
    """Send ``prompt`` to an open agent and yield normalized events."""
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
                    name=tool_name(chunk.name), args=dict(chunk.args or {}), id=chunk.id
                )
            elif isinstance(chunk, ToolResult):
                yield ToolResultEvent(name=tool_name(chunk.name), id=chunk.id, error=chunk.error)
        usage = getattr(resp, "usage_metadata", None)
        yield DoneEvent(
            stop_reason=stop_reason_name(getattr(resp, "stop_reason", None)),
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
        )
    except Exception as exc:  # noqa: BLE001 - surface any SDK failure to the client
        yield ErrorEvent(message=str(exc) or repr(exc), type=type(exc).__name__)


def tool_name(name: Any) -> str:
    return name.name if isinstance(name, BuiltinTools) else str(name)


def stop_reason_name(reason: Any) -> str | None:
    if reason is None:
        return None
    return getattr(reason, "name", str(reason))
