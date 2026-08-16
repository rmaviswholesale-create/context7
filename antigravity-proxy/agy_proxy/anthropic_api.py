"""Anthropic-compatible adapter: /v1/messages.

This is the endpoint Claude Code points at (``ANTHROPIC_BASE_URL``). It emits the
exact Messages streaming event sequence Anthropic clients expect:

    message_start
      (content_block_start / content_block_delta* / content_block_stop)*
    message_delta
    message_stop

Thoughts become ``thinking`` blocks; SDK tool calls become ``tool_use`` blocks;
visible text becomes ``text`` blocks.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from .context import ProxyContext, get_ctx
from .events import (
    DoneEvent,
    ErrorEvent,
    Event,
    TextEvent,
    ThoughtEvent,
    ToolCallEvent,
)

router = APIRouter()


class AnthropicMessage(BaseModel):
    role: str
    content: Any = None  # str | list[blocks]


class MessagesRequest(BaseModel):
    model: str | None = None
    messages: list[AnthropicMessage]
    system: Any = None  # str | list[{type:text,text}]
    max_tokens: int | None = None
    stream: bool = False


def _text_from_blocks(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for b in content:
            if isinstance(b, str):
                out.append(b)
            elif isinstance(b, dict):
                t = b.get("type")
                if t == "text":
                    out.append(b.get("text", ""))
                elif t == "tool_result":
                    out.append(_text_from_blocks(b.get("content")))
        return "".join(out)
    return str(content)


def _system_text(system: Any) -> str:
    if not system:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return "".join(
            b.get("text", "") for b in system if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(system)


def _normalize(req: MessagesRequest) -> list[dict[str, str]]:
    msgs: list[dict[str, str]] = []
    sys = _system_text(req.system)
    if sys:
        msgs.append({"role": "system", "content": sys})
    for m in req.messages:
        msgs.append({"role": m.role, "content": _text_from_blocks(m.content)})
    return msgs


def _stop_reason(stop: str | None) -> str:
    if stop and ("MAX" in stop or "EXCEEDED" in stop):
        return "max_tokens"
    return "end_turn"


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_anthropic(events: AsyncIterator[Event], model: str) -> AsyncIterator[str]:
    mid = f"msg_{uuid.uuid4().hex}"
    yield _sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": mid,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )

    index = -1
    open_kind: str | None = None  # "text" | "thinking" | "tool_use"

    def close_block() -> str | None:
        nonlocal open_kind
        if open_kind is not None:
            out = _sse("content_block_stop", {"type": "content_block_stop", "index": index})
            open_kind = None
            return out
        return None

    stop_reason = "end_turn"
    out_tokens = None
    async for ev in events:
        if isinstance(ev, TextEvent):
            if open_kind != "text":
                closed = close_block()
                if closed:
                    yield closed
                index += 1
                open_kind = "text"
                yield _sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "text_delta", "text": ev.text},
                },
            )
        elif isinstance(ev, ThoughtEvent):
            if open_kind != "thinking":
                closed = close_block()
                if closed:
                    yield closed
                index += 1
                open_kind = "thinking"
                yield _sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {"type": "thinking", "thinking": ""},
                    },
                )
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "thinking_delta", "thinking": ev.text},
                },
            )
        elif isinstance(ev, ToolCallEvent):
            closed = close_block()
            if closed:
                yield closed
            index += 1
            open_kind = "tool_use"
            yield _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": ev.id or f"toolu_{uuid.uuid4().hex[:12]}",
                        "name": ev.name,
                        "input": {},
                    },
                },
            )
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(ev.args, ensure_ascii=False),
                    },
                },
            )
            closed = close_block()
            if closed:
                yield closed
        elif isinstance(ev, ErrorEvent):
            closed = close_block()
            if closed:
                yield closed
            yield _sse(
                "error",
                {"type": "error", "error": {"type": ev.type, "message": ev.message}},
            )
            return
        elif isinstance(ev, DoneEvent):
            stop_reason = _stop_reason(ev.stop_reason)
            out_tokens = ev.output_tokens

    closed = close_block()
    if closed:
        yield closed
    yield _sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": out_tokens or 0},
        },
    )
    yield _sse("message_stop", {"type": "message_stop"})


async def _collect_anthropic(events: AsyncIterator[Event], model: str) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    text_buf: list[str] = []
    think_buf: list[str] = []
    stop_reason = "end_turn"
    usage_in = usage_out = None

    def flush_text() -> None:
        if text_buf:
            content.append({"type": "text", "text": "".join(text_buf)})
            text_buf.clear()

    def flush_think() -> None:
        if think_buf:
            content.append({"type": "thinking", "thinking": "".join(think_buf)})
            think_buf.clear()

    async for ev in events:
        if isinstance(ev, TextEvent):
            flush_think()
            text_buf.append(ev.text)
        elif isinstance(ev, ThoughtEvent):
            flush_text()
            think_buf.append(ev.text)
        elif isinstance(ev, ToolCallEvent):
            flush_think()
            flush_text()
            content.append(
                {
                    "type": "tool_use",
                    "id": ev.id or f"toolu_{uuid.uuid4().hex[:12]}",
                    "name": ev.name,
                    "input": ev.args,
                }
            )
        elif isinstance(ev, ErrorEvent):
            raise _AdapterError(ev.message, ev.type)
        elif isinstance(ev, DoneEvent):
            stop_reason = _stop_reason(ev.stop_reason)
            usage_in, usage_out = ev.input_tokens, ev.output_tokens
    flush_think()
    flush_text()

    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content or [{"type": "text", "text": ""}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": usage_in or 0, "output_tokens": usage_out or 0},
    }


class _AdapterError(Exception):
    def __init__(self, message: str, type_: str):
        super().__init__(message)
        self.type = type_


def _session_id(request: Request) -> str | None:
    return request.headers.get("x-agy-session") or request.headers.get("x-session-id")


async def _events_for(
    ctx: ProxyContext, req: MessagesRequest, request: Request
) -> AsyncIterator[Event]:
    preset = ctx.registry.resolve(req.model)
    messages = _normalize(req)
    sid = _session_id(request)
    if sid:
        from .runner import flatten_messages

        system_extra, _ = flatten_messages(messages)
        async with ctx.sessions.acquire(sid, preset, system_extra) as agent:
            async for ev in ctx.runner.stream_on_agent(agent, messages):
                yield ev
    else:
        async for ev in ctx.runner.stream_stateless(preset, messages):
            yield ev


@router.post("/v1/messages")
async def messages(req: MessagesRequest, request: Request):
    ctx = get_ctx(request)
    model_label = req.model or ctx.registry.default_name

    if req.stream:
        return StreamingResponse(
            _stream_anthropic(_events_for(ctx, req, request), model_label),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    try:
        body = await _collect_anthropic(_events_for(ctx, req, request), model_label)
    except _AdapterError as exc:
        return JSONResponse(
            status_code=502,
            content={"type": "error", "error": {"type": exc.type, "message": str(exc)}},
        )
    return JSONResponse(body)
