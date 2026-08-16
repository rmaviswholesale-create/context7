"""Shared test helpers."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from agy_proxy.events import Event


async def aiter(events: list[Event]) -> AsyncIterator[Event]:
    for e in events:
        yield e


def parse_sse(raw: str) -> list[dict]:
    """Parse OpenAI-style `data: {json}` lines (ignores [DONE])."""
    out = []
    for block in raw.strip().split("\n\n"):
        line = block.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            continue
        out.append(json.loads(payload))
    return out


def parse_sse_events(raw: str) -> list[tuple[str, dict]]:
    """Parse Anthropic-style `event: x\\ndata: {json}` blocks into (event, data)."""
    out = []
    for block in raw.strip().split("\n\n"):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        event = None
        data = None
        for ln in lines:
            if ln.startswith("event:"):
                event = ln[len("event:") :].strip()
            elif ln.startswith("data:"):
                data = json.loads(ln[len("data:") :].strip())
        if event is not None:
            out.append((event, data))
    return out


async def collect(agen) -> str:
    parts = []
    async for chunk in agen:
        parts.append(chunk)
    return "".join(parts)
