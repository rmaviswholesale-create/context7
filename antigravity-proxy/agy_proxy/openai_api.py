"""OpenAI-compatible adapter: /v1/chat/completions and /v1/models.

Translates the normalized event stream into OpenAI Chat Completions wire format
(streaming SSE chunks and non-streaming responses). Thoughts are surfaced as
``delta.reasoning_content``. The agent runs its tools server-side, so responses never
contain ``tool_calls``; client-supplied ``tools`` are ignored (and flagged via the
``x-agy-ignored`` response header). Errors are never written into ``delta.content``:
a failure mid-stream is an ``error`` object frame followed by ``[DONE]``.

``/v1/models`` answers in Anthropic's shape when the request carries an
``anthropic-version`` or ``x-api-key`` header (Anthropic SDKs and Claude Code).
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .context import get_ctx
from .errors import UnknownModelError, sse_error_frame, wants_anthropic
from .events import DoneEvent, ErrorEvent, Event, TextEvent, ThoughtEvent, finish_kind
from .messages import (
    ChatCompletionRequest,
    NormalizedRequest,
    collect,
    events_for,
    guarded,
    normalize_openai,
    prime,
    route_preset,
)

router = APIRouter()

_STARTED = int(time.time())
_DONE = "data: [DONE]\n\n"


def _sse(obj: dict[str, Any]) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _usage(prompt: int, completion: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


async def _stream_openai(
    events: AsyncIterator[Event],
    model: str,
    include_usage: bool = False,
    input_tokens: int = 0,
) -> AsyncIterator[str]:
    cid = f"chatcmpl-{uuid.uuid4().hex}"
    base: dict[str, Any] = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
    }
    if include_usage:
        base["usage"] = None

    def chunk(delta: dict[str, Any], finish: str | None = None) -> str:
        return _sse({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})

    # First chunk announces the assistant role (OpenAI convention).
    yield chunk({"role": "assistant"})
    emitted_chars = 0
    async for ev in guarded(events):
        if isinstance(ev, TextEvent):
            emitted_chars += len(ev.text)
            yield chunk({"content": ev.text})
        elif isinstance(ev, ThoughtEvent):
            emitted_chars += len(ev.text)
            yield chunk({"reasoning_content": ev.text})
        elif isinstance(ev, ErrorEvent):
            yield sse_error_frame(ev.error, anthropic=False)
            yield _DONE
            return
        elif isinstance(ev, DoneEvent):
            yield chunk({}, finish=finish_kind(ev.stop_reason))
            if include_usage:
                completion = ev.output_tokens or max(1, emitted_chars // 4)
                prompt = ev.input_tokens or input_tokens
                yield _sse({**base, "choices": [], "usage": _usage(prompt, completion)})
            yield _DONE
            return


async def _collect_openai(
    events: AsyncIterator[Event], model: str, input_tokens: int = 0
) -> dict[str, Any]:
    done = await collect(events, input_tokens)
    message: dict[str, Any] = {"role": "assistant", "content": done.text}
    if done.thinking:
        message["reasoning_content"] = done.thinking
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {"index": 0, "message": message, "finish_reason": finish_kind(done.stop_reason)}
        ],
        "usage": _usage(done.input_tokens, done.output_tokens),
    }


def _stream_headers(norm: NormalizedRequest) -> dict[str, str]:
    return {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", **norm.response_headers()}


@router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    ctx = get_ctx(request)
    norm = normalize_openai(req)
    preset = route_preset(ctx.registry, norm.model, norm.max_tokens)  # 404 before streaming
    model_label = req.model or preset.name
    events = events_for(ctx, norm, request)

    if norm.stream:
        primed = await prime(events)  # start-up failures become real HTTP errors
        return StreamingResponse(
            _stream_openai(primed, model_label, norm.include_usage, norm.input_tokens),
            media_type="text/event-stream",
            headers=_stream_headers(norm),
        )
    body = await _collect_openai(events, model_label, norm.input_tokens)
    return JSONResponse(body, headers=norm.response_headers())


# -- models ------------------------------------------------------------------


def _model_obj(name: str, anthropic: bool) -> dict[str, Any]:
    if anthropic:
        created = datetime.fromtimestamp(_STARTED, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        return {"type": "model", "id": name, "display_name": name, "created_at": created}
    return {"id": name, "object": "model", "created": _STARTED, "owned_by": "antigravity"}


def _is_anthropic(request: Request) -> bool:
    return wants_anthropic(request.url.path, {k.lower(): v for k, v in request.headers.items()})


@router.get("/v1/models")
async def list_models(request: Request):
    names = get_ctx(request).registry.names()
    if _is_anthropic(request):
        data = [_model_obj(n, True) for n in names]
        return {
            "data": data,
            "has_more": False,
            "first_id": data[0]["id"] if data else None,
            "last_id": data[-1]["id"] if data else None,
        }
    return {"object": "list", "data": [_model_obj(n, False) for n in names]}


@router.get("/v1/models/{model_id:path}")
async def get_model(model_id: str, request: Request):
    if model_id not in get_ctx(request).registry.names():
        raise UnknownModelError(f"model {model_id!r} not found", param="model")
    return _model_obj(model_id, _is_anthropic(request))
