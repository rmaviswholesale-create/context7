"""Normalized, protocol-agnostic stream events.

Both the OpenAI and Anthropic adapters consume this single event vocabulary,
so the SDK-interaction logic is written exactly once (in ``runner_events.py``) and the
adapters only translate events into their wire format. Kept free of any SDK
import so adapters and their tests never need the Antigravity binary.

There are deliberately no tool events: the agent runs its tools server-side, so the
SDK's tool calls/results are logged (when enabled) and never reach the wire.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import ProxyError, to_proxy_error


@dataclass
class TextEvent:
    """A delta of the model's visible text output."""

    text: str


@dataclass
class ThoughtEvent:
    """A delta of the model's internal reasoning / thinking."""

    text: str


@dataclass
class DoneEvent:
    """Terminal event: the turn finished. ``stop_reason`` is the SDK's StopReason name."""

    stop_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class ErrorEvent:
    """Terminal event: the turn failed. Carries a client-safe :class:`ProxyError`."""

    error: ProxyError

    @classmethod
    def of(cls, exc: BaseException) -> ErrorEvent:
        return cls(to_proxy_error(exc))

    @property
    def message(self) -> str:
        return self.error.message


Event = TextEvent | ThoughtEvent | DoneEvent | ErrorEvent

_LENGTH_REASONS = frozenset(
    {"MAX_INPUT_TOKENS_EXCEEDED", "MAX_OUTPUT_TOKENS_EXCEEDED", "MAX_TOTAL_TOKENS_EXCEEDED"}
)


def finish_kind(stop_reason: str | None) -> str:
    """Collapse an SDK stop reason to ``"stop"`` or ``"length"``.

    UNSPECIFIED/None -> stop; MAX_*_TOKENS_EXCEEDED -> length; MAX_MODEL_CALLS /
    MAX_TOOL_CALLS -> stop (the turn was cut short, which the runner logs). QUOTA_EXHAUSTED
    never reaches here: the runner turns it into a RateLimited ErrorEvent.
    """
    return "length" if stop_reason in _LENGTH_REASONS else "stop"
