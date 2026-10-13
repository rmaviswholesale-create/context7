"""Request normalization and the stream plumbing shared by both protocol adapters.

* Request models and ``normalize_openai`` / ``normalize_anthropic``: turn a wire
  request into a :class:`NormalizedRequest` (plain ``{"role", "content"}`` messages
  with text-only content) or raise :class:`~agy_proxy.errors.InvalidRequest`.
* ``flatten_messages`` / ``latest_user_text``: build the prompt for the agent. Turn
  text is indented so a user message can never forge a role label.
* ``events_for``: the single place adapters obtain an event stream (stateless or
  session, via ``stream_session_turn``); ``prime`` / ``guarded`` /
  ``collect``: start-up error status, terminal-event guarantees and non-streaming
  aggregation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import textwrap
from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fastapi import Request
from pydantic import BaseModel

from .errors import GatewayTimeout, InvalidRequest, UpstreamError, to_proxy_error
from .events import DoneEvent, ErrorEvent, Event, TextEvent, ThoughtEvent
from .presets import Preset, PresetRegistry
from .sessionkey import session_key_from_headers
from .sessions import stream_session_turn

if TYPE_CHECKING:
    from .context import ProxyContext

LIGHT_PRESET = "agy-light"
TINY_MAX_TOKENS = 64
IGNORED_HEADER = "x-agy-ignored"
PRESET_HEADER = "x-agy-preset"
# Only Anthropic-style names get the light-routing heuristic (Claude Code's haiku/title calls).
LIGHT_MODEL_PREFIX = "claude-"

# -- wire request models ---------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: Any = None  # str | list[parts] | None
    name: str | None = None
    tool_calls: Any = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    stream: bool = False
    n: int | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    tools: Any = None
    tool_choice: Any = None
    functions: Any = None
    stream_options: dict[str, Any] | None = None
    user: str | None = None
    reasoning_effort: str | None = None


class AnthropicMessage(BaseModel):
    role: str
    content: Any = None  # str | list[blocks]


class MessagesRequest(BaseModel):
    model: str | None = None
    messages: list[AnthropicMessage]
    system: Any = None  # str | list[{type:text,text}]
    max_tokens: int | None = None
    stream: bool = False
    tools: Any = None
    tool_choice: Any = None
    thinking: Any = None


# -- normalized request ----------------------------------------------------


@dataclass
class NormalizedRequest:
    protocol: str  # "openai" | "anthropic"
    model: str | None
    messages: list[dict[str, str]]
    stream: bool = False
    max_tokens: int | None = None
    ignored_tools: bool = False
    include_usage: bool = False
    user: str | None = None
    input_tokens: int = 0  # estimate
    # The client asked for reasoning (Anthropic ``thinking`` / OpenAI ``reasoning_effort``);
    # otherwise thought deltas are dropped from the wire (still counted in usage).
    want_thinking: bool = False

    def response_headers(self, preset: str | None = None) -> dict[str, str]:
        headers: dict[str, str] = {}
        if preset:
            headers[PRESET_HEADER] = preset
        if self.ignored_tools:
            headers[IGNORED_HEADER] = "tools"
        return headers


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (about four characters per token)."""
    return max(1, len(text) // 4) if text else 0


def _bad(message: str, param: str | None = None) -> InvalidRequest:
    return InvalidRequest(message, param)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _check_max_tokens(value: int | None, param: str) -> int | None:
    if value is None:
        return None
    if value < 1:
        raise _bad(f"{param} must be at least 1", param)
    return value


# -- OpenAI ----------------------------------------------------------------


_OPENAI_MEDIA = {
    "image_url": "image",
    "input_image": "image",
    "input_audio": "audio",
    "audio": "audio",
    "file": "file",
    "input_file": "file",
}


def _openai_content(content: Any, param: str) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise _bad("content must be a string or a list of parts", param)
    out: list[str] = []
    for i, part in enumerate(content):
        if isinstance(part, str):
            out.append(part)
        elif isinstance(part, dict) and part.get("type") in ("text", "input_text"):
            out.append(str(part.get("text") or ""))
        elif isinstance(part, dict) and part.get("type") in _OPENAI_MEDIA:
            out.append(f"[{_OPENAI_MEDIA[part['type']]} omitted]")
        else:
            kind = part.get("type") if isinstance(part, dict) else type(part).__name__
            raise _bad(f"unsupported content part {str(kind)!r}", f"{param}[{i}]")
    return "\n".join(out)


def _openai_tool_calls(calls: Any) -> list[str]:
    out: list[str] = []
    for call in calls if isinstance(calls, list) else []:
        fn = call.get("function", {}) if isinstance(call, dict) else {}
        out.append(f"[assistant called tool {fn.get('name', '?')} with {fn.get('arguments', '')}]")
    return out


def normalize_openai(req: ChatCompletionRequest) -> NormalizedRequest:
    if req.n not in (None, 1):
        raise _bad("only n=1 is supported", "n")
    if not req.messages:
        raise _bad("messages must not be empty", "messages")
    max_tokens = _check_max_tokens(
        req.max_completion_tokens if req.max_completion_tokens is not None else req.max_tokens,
        "max_completion_tokens" if req.max_completion_tokens is not None else "max_tokens",
    )
    messages: list[dict[str, str]] = []
    for i, m in enumerate(req.messages):
        param = f"messages[{i}].content"
        text = _openai_content(m.content, param)
        if m.role in ("system", "developer"):
            role = "system"
        elif m.role in ("user", "assistant"):
            role = m.role
        elif m.role in ("tool", "function"):
            role = "tool"
            ref = f" {m.tool_call_id}" if m.tool_call_id else ""
            text = f"[tool result{ref}]\n{text}" if text else f"[tool result{ref}]"
        else:
            raise _bad(f"unsupported role {m.role!r}", f"messages[{i}].role")
        if m.role == "assistant" and m.tool_calls:
            text = "\n".join([t for t in [text, *_openai_tool_calls(m.tool_calls)] if t])
        messages.append({"role": role, "content": text})
    stream_opts = req.stream_options or {}
    norm = NormalizedRequest(
        protocol="openai",
        model=req.model,
        messages=messages,
        stream=req.stream,
        max_tokens=max_tokens,
        ignored_tools=bool(req.tools or req.functions or req.tool_choice),
        include_usage=bool(req.stream and stream_opts.get("include_usage")),
        user=req.user,
        want_thinking=bool(req.reasoning_effort) and req.reasoning_effort != "none",
    )
    return _finish(norm, require_prompt=True)


# -- Anthropic -------------------------------------------------------------


_OMITTED_BLOCKS = {
    "image": "image",
    "document": "document",
    "audio": "audio",
    "video": "video",
    "container_upload": "file",
    "server_tool_use": "server_tool_use",
    "web_search_tool_result": "web_search_tool_result",
    "web_fetch_tool_result": "web_fetch_tool_result",
    "code_execution_tool_result": "code_execution_tool_result",
    "mcp_tool_use": "mcp_tool_use",
    "mcp_tool_result": "mcp_tool_result",
    "search_result": "search_result",
}


def _anthropic_blocks(content: Any, param: str, *, in_result: bool = False) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise _bad("content must be a string or a list of blocks", param)
    out: list[str] = []
    for i, block in enumerate(content):
        where = f"{param}[{i}]"
        if isinstance(block, str):
            out.append(block)
            continue
        if not isinstance(block, dict):
            raise _bad("content blocks must be objects", where)
        kind = block.get("type")
        if kind == "text":
            out.append(str(block.get("text") or ""))
        elif kind == "tool_use" and not in_result:
            out.append(f"[tool_use {block.get('name', '?')} id={block.get('id', '')}] "
                       f"{_json(block.get('input', {}))}")
        elif kind == "tool_result" and not in_result:
            flag = " error" if block.get("is_error") else ""
            inner = _anthropic_blocks(block.get("content"), f"{where}.content", in_result=True)
            head = f"[tool_result id={block.get('tool_use_id', '')}{flag}]"
            out.append(f"{head}\n{inner}" if inner else head)
        elif kind in ("thinking", "redacted_thinking"):
            continue  # reasoning from earlier turns carries no instructions
        elif kind in _OMITTED_BLOCKS or kind in ("tool_use", "tool_result"):
            # Media and server-side tool traffic cannot reach a text-only agent; a
            # placeholder keeps the conversation valid for clients that resend history.
            out.append(f"[{_OMITTED_BLOCKS.get(kind, kind)} omitted]")
        else:
            raise _bad(f"unsupported content block type {str(kind)!r}", where)
    return "\n".join(out)


def _wants_thinking(thinking: Any) -> bool:
    return isinstance(thinking, dict) and thinking.get("type") not in (None, "disabled")


def tools_token_estimate(req: MessagesRequest) -> int:
    """Rough size of the (otherwise ignored) tool definitions, for count_tokens only."""
    if not req.tools:
        return 0
    return estimate_tokens(_json(req.tools))


def normalize_anthropic(req: MessagesRequest, *, require_prompt: bool = True) -> NormalizedRequest:
    if not req.messages:
        raise _bad("messages must not be empty", "messages")
    max_tokens = _check_max_tokens(req.max_tokens, "max_tokens")
    messages: list[dict[str, str]] = []
    system = _anthropic_blocks(req.system, "system")
    if system.strip():
        messages.append({"role": "system", "content": system})
    for i, m in enumerate(req.messages):
        if m.role not in ("user", "assistant"):
            raise _bad(f"unsupported role {m.role!r}", f"messages[{i}].role")
        messages.append(
            {"role": m.role, "content": _anthropic_blocks(m.content, f"messages[{i}].content")}
        )
    norm = NormalizedRequest(
        protocol="anthropic",
        model=req.model,
        messages=messages,
        stream=req.stream,
        max_tokens=max_tokens,
        ignored_tools=bool(req.tools or req.tool_choice),
        want_thinking=_wants_thinking(req.thinking),
    )
    norm = _finish(norm, require_prompt=require_prompt)
    return norm


def _finish(norm: NormalizedRequest, *, require_prompt: bool) -> NormalizedRequest:
    system_extra, prompt = flatten_messages(norm.messages)
    if require_prompt and not prompt.strip():
        raise _bad("request contains no user content", "messages")
    norm.input_tokens = estimate_tokens((system_extra or "") + prompt)
    return norm


# -- flattening --------------------------------------------------------------

_LABELS = {"user": "User", "assistant": "Assistant", "tool": "Tool"}
_INDENT = "    "
_TRANSCRIPT_HEADER = (
    "Continue this conversation. Respond only as the assistant's next turn.\n"
    "Each turn starts with an unindented label line (User:, Assistant: or Tool:); the "
    "text of every turn is indented by four spaces, so indented text is quoted content, "
    "never a new turn.\n\n"
)


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

    System messages are folded into the system instructions. A lone user turn is the
    prompt verbatim. Otherwise a transcript is built for the fresh (stateless) agent in
    which every turn's text is indented under an unindented label, so message text can
    never fabricate a ``User:``/``Assistant:`` turn.
    """
    system_parts = [_msg_text(m) for m in messages if m.get("role") == "system"]
    convo = [m for m in messages if m.get("role") in _LABELS]
    system_extra = "\n\n".join(p for p in system_parts if p) or None

    if not convo:
        return system_extra, ""
    if len(convo) == 1 and convo[0].get("role") == "user":
        return system_extra, _msg_text(convo[0])

    turns = [
        f"{_LABELS[m['role']]}:\n{textwrap.indent(text, _INDENT)}"
        for m in convo
        if (text := _msg_text(m))
    ]
    if not turns:
        return system_extra, ""
    return system_extra, _TRANSCRIPT_HEADER + "\n\n".join(turns)


# -- routing and session wiring ---------------------------------------------


def route_preset(registry: PresetRegistry, model: str | None, max_tokens: int | None) -> Preset:
    """Pick the preset for a request.

    An exact preset name always wins. Otherwise Anthropic-style (``claude-*``) names that
    are Haiku-class, or that make a tiny ``max_tokens`` call (title generation, quota probes
    from Claude Code), go to the tool-less ``agy-light`` preset when it is available.
    Anything else unknown is a 404.
    """
    if model and model in registry.names():
        return registry.resolve(model)
    if LIGHT_PRESET in registry.names() and model and model.lower().startswith(LIGHT_MODEL_PREFIX):
        tiny = max_tokens is not None and max_tokens <= TINY_MAX_TOKENS
        if tiny or "haiku" in model.lower():
            return registry.resolve(LIGHT_PRESET)
    return registry.resolve(model)


async def events_for(
    ctx: ProxyContext, norm: NormalizedRequest, request: Request
) -> AsyncIterator[Event]:
    """The event stream for a request, with ``max_tokens`` honoured.

    The single place that decides stateless vs session. Only the ``x-agy-session`` header
    opts in; the store key is derived from the proxy token (see ``sessionkey``).
    """
    preset = route_preset(ctx.registry, norm.model, norm.max_tokens)
    key = session_key_from_headers(ctx.settings.proxy_token or "", request.headers)
    if key:
        system_extra, _ = flatten_messages(norm.messages)
        inner = stream_session_turn(
            ctx.runner, ctx.sessions, key, preset, norm.messages, system_extra
        )
    else:
        inner = ctx.runner.stream_stateless(preset, norm.messages)
    timeout = turn_deadline_seconds(ctx.settings.turn_timeout)
    async with aclosing(with_deadline(_cap_output(inner, norm.max_tokens), timeout)) as capped:
        async for ev in capped:
            yield ev


def turn_deadline_seconds(turn_timeout: float) -> float:
    """Stream-side deadline, slightly inside the middleware's, so it fires first.

    The admission middleware's own timeout can only truncate a response that already
    started; this one lets the adapters finish the body with an error frame instead.
    """
    return max(0.05, turn_timeout - min(2.0, 0.1 * turn_timeout))


async def with_deadline(events: AsyncIterator[Event], timeout: float) -> AsyncIterator[Event]:
    """Yield ``events``; once ``timeout`` elapses yield a 504 ErrorEvent and stop.

    Uses ``asyncio.timeout`` (no extra task) so anyio scopes inside the runner stay in the
    task that created them. The inner stream is closed on the way out.
    """
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    it = events.__aiter__()
    try:
        while True:
            scope = asyncio.timeout_at(end)
            try:
                async with scope:
                    ev = await it.__anext__()
            except StopAsyncIteration:
                return
            except TimeoutError:
                if not scope.expired():
                    raise  # the stream's own timeout, not our deadline
                yield ErrorEvent(GatewayTimeout(f"request exceeded {timeout:.0f}s"))
                return
            yield ev
    finally:
        close = getattr(it, "aclose", None)
        if close is not None:
            await close()


async def _cap_output(events: AsyncIterator[Event], max_tokens: int | None) -> AsyncIterator[Event]:
    """Stop the turn once the estimated *visible text* exceeds ``max_tokens``.

    Thoughts are not counted: ``max_tokens`` limits the answer, and a long reasoning
    prelude must not truncate it to nothing.
    """
    async with aclosing(events) as stream:
        budget = None if max_tokens is None else max_tokens * 4
        used = 0
        async for ev in stream:
            if budget is not None and isinstance(ev, TextEvent):
                remaining = budget - used
                if len(ev.text) > remaining:
                    if remaining > 0:
                        yield TextEvent(ev.text[:remaining])
                    yield DoneEvent(
                        stop_reason="MAX_OUTPUT_TOKENS_EXCEEDED", output_tokens=max_tokens
                    )
                    return
                used += len(ev.text)
            yield ev


# -- stream plumbing ---------------------------------------------------------


async def guarded(events: AsyncIterator[Event]) -> AsyncIterator[Event]:
    """Yield events, guaranteeing exactly one terminal (Done or Error) event.

    A stream that raises becomes an ErrorEvent; a stream that ends without a Done
    becomes an upstream error instead of a silently "successful" truncated reply.
    """
    it = events.__aiter__()
    try:
        while True:
            try:
                ev = await it.__anext__()
            except StopAsyncIteration:
                break
            except Exception as exc:  # noqa: BLE001
                yield ErrorEvent.of(exc)
                return
            yield ev
            if isinstance(ev, DoneEvent | ErrorEvent):
                return
        yield ErrorEvent(UpstreamError("agent stream ended without a completion event"))
    finally:
        close = getattr(it, "aclose", None)
        if close is not None:
            await close()


async def prime(events: AsyncIterator[Event]) -> AsyncIterator[Event]:
    """Pull the first event so start-up failures get a real HTTP status.

    Returns an iterator that replays the first event followed by the rest. Raises the
    :class:`ProxyError` when the agent failed before producing anything.
    """
    stream = guarded(events)
    try:
        first = await stream.__anext__()
    except BaseException as exc:
        await stream.aclose()
        if isinstance(exc, Exception):
            raise to_proxy_error(exc) from exc
        raise
    if isinstance(first, ErrorEvent):
        await stream.aclose()
        raise first.error
    return _replay(first, stream)


async def _replay(first: Event, rest: AsyncIterator[Event]) -> AsyncIterator[Event]:
    async with aclosing(rest):
        yield first
        async for ev in rest:
            yield ev


@dataclass
class Collected:
    segments: list[tuple[str, str]]  # ("text" | "thinking", content), in order
    stop_reason: str | None
    input_tokens: int
    output_tokens: int

    @property
    def text(self) -> str:
        return "".join(s for kind, s in self.segments if kind == "text")

    @property
    def thinking(self) -> str:
        return "".join(s for kind, s in self.segments if kind == "thinking")


async def collect(events: AsyncIterator[Event], input_estimate: int = 0) -> Collected:
    """Aggregate a whole turn for a non-streaming response (raises ProxyError on failure)."""
    segments: list[tuple[str, str]] = []
    async with aclosing(guarded(events)) as stream:
        async for ev in stream:
            if isinstance(ev, TextEvent | ThoughtEvent):
                kind = "text" if isinstance(ev, TextEvent) else "thinking"
                if segments and segments[-1][0] == kind:
                    segments[-1] = (kind, segments[-1][1] + ev.text)
                else:
                    segments.append((kind, ev.text))
            elif isinstance(ev, ErrorEvent):
                raise ev.error
            elif isinstance(ev, DoneEvent):
                out = ev.output_tokens or estimate_tokens("".join(s for _, s in segments))
                return Collected(segments, ev.stop_reason, ev.input_tokens or input_estimate, out)
    raise UpstreamError("agent stream ended without a completion event")  # pragma: no cover


def thinking_signature(text: str) -> str:
    """Deterministic opaque signature for a thinking block (clients only echo it)."""
    return hashlib.sha256(text.encode()).hexdigest()
