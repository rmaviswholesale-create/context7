"""Normalized, protocol-agnostic stream events.

Both the OpenAI and Anthropic adapters consume this single event vocabulary,
so the SDK-interaction logic is written exactly once (in ``runner.py``) and the
adapters only translate events into their wire format. Kept free of any SDK
import so adapters and their tests never need the Antigravity binary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TextEvent:
    """A delta of the model's visible text output."""

    text: str


@dataclass
class ThoughtEvent:
    """A delta of the model's internal reasoning / thinking."""

    text: str


@dataclass
class ToolCallEvent:
    """The agent invoked a (SDK-executed) tool. Informational for the client."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)
    id: str | None = None


@dataclass
class ToolResultEvent:
    """Result of an SDK-executed tool. Informational for the client."""

    name: str
    id: str | None = None
    error: str | None = None


@dataclass
class DoneEvent:
    """Terminal event: the turn finished normally."""

    stop_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class ErrorEvent:
    """Terminal event: the turn failed."""

    message: str
    type: str = "proxy_error"


Event = TextEvent | ThoughtEvent | ToolCallEvent | ToolResultEvent | DoneEvent | ErrorEvent
