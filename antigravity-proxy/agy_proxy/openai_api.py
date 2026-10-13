"""OpenAI-compatible adapter: /v1/chat/completions and /v1/models.

Translates the normalized event stream into OpenAI Chat Completions wire format
(streaming SSE chunks and non-streaming responses). Thoughts are surfaced as
``delta.reasoning_content``; SDK tool calls are surfaced as ``delta.tool_calls``.
"""

from __future__ import annotations

import json
import time
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
from .sessionkey import session_key_from_headers
from .sessions import stream_session_turn

router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: Any = None  # str | list[parts] | None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    stream: bool = False
    user: str | None = None  # accepted and ignored (sessions use the x-agy-session header)


def _extract_text(content: Any) -> str:
    """Flatten OpenAI content (string or list of parts) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
        return "".join(parts)
    return str(content)


def _normalize(messages: list[ChatMessage]) -> list[dict[str, str]]:
    return [{"role": m.role, "content": _extract_text(m.content)} for m in messages]


def _finish_reason(stop: str | None) -> str:
    # SDK stop reasons that indicate a limit was hit map to "length"; else "stop".
    if stop and ("MAX" in stop or "EXCEEDED" in stop):
        return "length"
    return "stop"


def _sse(obj: dict[str, Any]) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


async def _stream_openai(
    events: AsyncIterator[Event], model: str
) -> AsyncIterator[str]:
    cid = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    base = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model}

    def chunk(delta: dict[str, Any], finish: str | None = None) -> str:
        return _sse({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})

    # First chunk announces the assistant role (OpenAI convention).
    yield chunk({"role": "assistant"})
    tool_index = 0
    async for ev in events:
        if isinstance(ev, TextEvent):
            yield chunk({"content": ev.text})
        elif isinstance(ev, ThoughtEvent):
            yield chunk({"reasoning_content": ev.text})
        elif isinstance(ev, ToolCallEvent):
            yield chunk(
                {
                    "tool_calls": [
                        {
                            "index": tool_index,
                            "id": ev.id or f"call_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": {
                                "name": ev.name,
                                "arguments": json.dumps(ev.args, ensure_ascii=False),
                            },
                        }
                    ]
                }
            )
            tool_index += 1
        elif isinstance(ev, ErrorEvent):
            yield chunk({"content": f"\n\n[agy-proxy error: {ev.message}]"})
            yield chunk({}, finish="stop")
            yield "data: [DONE]\n\n"
            return
        elif isinstance(ev, DoneEvent):
            yield chunk({}, finish=_finish_reason(ev.stop_reason))
            yield "data: [DONE]\n\n"
            return
    # Defensive close if the stream ended without an explicit Done.
    yield chunk({}, finish="stop")
    yield "data: [DONE]\n\n"


async def _collect_openai(events: AsyncIterator[Event], model: str) -> dict[str, Any]:
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    finish = "stop"
    usage_in = usage_out = None
    async for ev in events:
        if isinstance(ev, TextEvent):
            text_parts.append(ev.text)
        elif isinstance(ev, ThoughtEvent):
            reasoning_parts.append(ev.text)
        elif isinstance(ev, ToolCallEvent):
            tool_calls.append(
                {
                    "id": ev.id or f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": ev.name,
                        "arguments": json.dumps(ev.args, ensure_ascii=False),
                    },
                }
            )
        elif isinstance(ev, ErrorEvent):
            raise _AdapterError(ev.message, ev.type)
        elif isinstance(ev, DoneEvent):
            finish = _finish_reason(ev.stop_reason)
            usage_in, usage_out = ev.input_tokens, ev.output_tokens

    message: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts) or None}
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": usage_in or 0,
            "completion_tokens": usage_out or 0,
            "total_tokens": (usage_in or 0) + (usage_out or 0),
        },
    }


class _AdapterError(Exception):
    def __init__(self, message: str, type_: str):
        super().__init__(message)
        self.type = type_


def _session_id(req: ChatCompletionRequest, request: Request) -> str | None:
    return session_key_from_headers(get_ctx(request).settings.proxy_token or "", request.headers)


async def _events_for(
    ctx: ProxyContext, req: ChatCompletionRequest, request: Request
) -> AsyncIterator[Event]:
    preset = ctx.registry.resolve(req.model)
    messages = _normalize(req.messages)
    sid = _session_id(req, request)
    if sid:
        system_extra, _ = _flatten_system(messages)
        async for ev in stream_session_turn(
            ctx.runner, ctx.sessions, sid, preset, messages, system_extra
        ):
            yield ev
    else:
        async for ev in ctx.runner.stream_stateless(preset, messages):
            yield ev


def _flatten_system(messages: list[dict[str, str]]) -> tuple[str | None, str]:
    from .runner import flatten_messages

    return flatten_messages(messages)


@router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    ctx = get_ctx(request)
    ctx.registry.resolve(req.model)  # unknown model -> 404 before any streaming starts
    model_label = req.model or ctx.registry.default_name

    if req.stream:
        return StreamingResponse(
            _stream_openai(_events_for(ctx, req, request), model_label),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    try:
        body = await _collect_openai(_events_for(ctx, req, request), model_label)
    except _AdapterError as exc:
        return JSONResponse(
            status_code=502,
            content={"error": {"message": str(exc), "type": exc.type, "code": None}},
        )
    return JSONResponse(body)


@router.get("/v1/models")
async def list_models(request: Request):
    ctx = get_ctx(request)
    now = int(time.time())
    data = [
        {"id": name, "object": "model", "created": now, "owned_by": "antigravity"}
        for name in ctx.registry.names()
    ]
    return {"object": "list", "data": data}
