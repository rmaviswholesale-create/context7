"""Anthropic-compatible adapter: /v1/messages and /v1/messages/count_tokens.

This is the endpoint Claude Code points at (``ANTHROPIC_BASE_URL``). It emits the
Messages streaming event sequence Anthropic clients expect:

    message_start, ping
      (content_block_start / content_block_delta* / content_block_stop)*
    message_delta
    message_stop

Thoughts become ``thinking`` blocks (with a signature) and visible text becomes
``text`` blocks. The agent runs its tools server-side, so there are never ``tool_use``
blocks; client-supplied ``tools`` are ignored (and flagged via ``x-agy-ignored``).
A failure mid-stream is an ``event: error`` frame, after which the stream closes.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .context import get_ctx
from .errors import sse_error_frame
from .events import DoneEvent, ErrorEvent, Event, TextEvent, ThoughtEvent, finish_kind
from .messages import (
    MessagesRequest,
    NormalizedRequest,
    collect,
    events_for,
    guarded,
    normalize_anthropic,
    prime,
    route_preset,
    thinking_signature,
)

router = APIRouter()

_STOP = {"stop": "end_turn", "length": "max_tokens"}


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _block_start(index: int, block: dict[str, Any]) -> str:
    return _sse(
        "content_block_start",
        {"type": "content_block_start", "index": index, "content_block": block},
    )


def _block_delta(index: int, delta: dict[str, Any]) -> str:
    return _sse(
        "content_block_delta", {"type": "content_block_delta", "index": index, "delta": delta}
    )


def _block_stop(index: int) -> str:
    return _sse("content_block_stop", {"type": "content_block_stop", "index": index})


async def _stream_anthropic(
    events: AsyncIterator[Event], model: str, input_tokens: int = 0
) -> AsyncIterator[str]:
    yield _sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            },
        },
    )
    yield _sse("ping", {"type": "ping"})

    index = -1
    open_kind: str | None = None  # "text" | "thinking"
    thinking: list[str] = []
    emitted_chars = 0

    def close_block() -> str:
        nonlocal open_kind
        if open_kind is None:
            return ""
        out = ""
        if open_kind == "thinking":
            sig = {"type": "signature_delta", "signature": thinking_signature("".join(thinking))}
            out += _block_delta(index, sig)
            thinking.clear()
        out += _block_stop(index)
        open_kind = None
        return out

    async for ev in guarded(events):
        if isinstance(ev, TextEvent | ThoughtEvent):
            kind = "text" if isinstance(ev, TextEvent) else "thinking"
            if open_kind != kind:
                closed = close_block()
                if closed:
                    yield closed
                index += 1
                open_kind = kind
                if kind == "text":
                    yield _block_start(index, {"type": "text", "text": ""})
                else:
                    yield _block_start(index, {"type": "thinking", "thinking": "", "signature": ""})
            emitted_chars += len(ev.text)
            if kind == "text":
                yield _block_delta(index, {"type": "text_delta", "text": ev.text})
            else:
                thinking.append(ev.text)
                yield _block_delta(index, {"type": "thinking_delta", "thinking": ev.text})
        elif isinstance(ev, ErrorEvent):
            closed = close_block()
            if closed:
                yield closed
            yield sse_error_frame(ev.error, anthropic=True)
            return
        elif isinstance(ev, DoneEvent):
            closed = close_block()
            if closed:
                yield closed
            if index < 0:  # empty response: clients expect at least one content block
                yield _block_start(0, {"type": "text", "text": ""})
                yield _block_stop(0)
            usage: dict[str, int] = {
                "output_tokens": ev.output_tokens or max(1, emitted_chars // 4)
            }
            if ev.input_tokens:
                usage["input_tokens"] = ev.input_tokens
            yield _sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": _STOP[finish_kind(ev.stop_reason)],
                        "stop_sequence": None,
                    },
                    "usage": usage,
                },
            )
            yield _sse("message_stop", {"type": "message_stop"})
            return


async def _collect_anthropic(
    events: AsyncIterator[Event], model: str, input_tokens: int = 0
) -> dict[str, Any]:
    done = await collect(events, input_tokens)
    content: list[dict[str, Any]] = []
    for kind, text in done.segments:
        if kind == "thinking":
            content.append(
                {"type": "thinking", "thinking": text, "signature": thinking_signature(text)}
            )
        else:
            content.append({"type": "text", "text": text})
    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content or [{"type": "text", "text": ""}],
        "stop_reason": _STOP[finish_kind(done.stop_reason)],
        "stop_sequence": None,
        "usage": {"input_tokens": done.input_tokens, "output_tokens": done.output_tokens},
    }


def _stream_headers(norm: NormalizedRequest) -> dict[str, str]:
    return {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", **norm.response_headers()}


@router.post("/v1/messages")
async def messages(req: MessagesRequest, request: Request):
    ctx = get_ctx(request)
    norm = normalize_anthropic(req)
    preset = route_preset(ctx.registry, norm.model, norm.max_tokens)  # 404 before streaming
    model_label = req.model or preset.name
    events = events_for(ctx, norm, request)

    if norm.stream:
        primed = await prime(events)  # start-up failures become real HTTP errors
        return StreamingResponse(
            _stream_anthropic(primed, model_label, norm.input_tokens),
            media_type="text/event-stream",
            headers=_stream_headers(norm),
        )
    body = await _collect_anthropic(events, model_label, norm.input_tokens)
    return JSONResponse(body, headers=norm.response_headers())


@router.post("/v1/messages/count_tokens")
async def count_tokens(req: MessagesRequest, request: Request):
    norm = normalize_anthropic(req, require_prompt=False)
    return JSONResponse({"input_tokens": norm.input_tokens}, headers=norm.response_headers())
