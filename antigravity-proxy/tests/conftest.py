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


# -- SDK stand-ins used to drive the real event mapping (runner_events.run_chat) ---------


class StubResp:
    """Minimal ChatResponse: an async ``chunks`` stream plus stop_reason/usage."""

    def __init__(self, chunks, stop_reason=None, usage=None, fail_after=None):
        self._chunks = list(chunks)
        self.stop_reason = stop_reason
        self.usage_metadata = usage
        self._fail_after = fail_after

    @property
    def chunks(self):
        async def gen():
            for i, c in enumerate(self._chunks):
                if self._fail_after is not None and i == self._fail_after:
                    raise RuntimeError("stream broke")
                yield c

        return gen()


class StubAgent:
    def __init__(self, resp=None, chat_exc=None):
        self.resp = resp
        self.chat_exc = chat_exc
        self.prompts = []

    async def chat(self, prompt):
        self.prompts.append(prompt)
        if self.chat_exc:
            raise self.chat_exc
        return self.resp


class ScriptedRunner:
    """Replaces ``ctx.runner``: yields scripted events, raises, or runs a StubAgent."""

    def __init__(self, events=None, exc=None, agent=None):
        self.events = events if events is not None else []
        self.exc = exc
        self.agent = agent
        self.calls = 0
        self.last = None

    async def stream_stateless(self, preset, messages):
        from agy_proxy.messages import flatten_messages
        from agy_proxy.runner_events import run_chat

        self.calls += 1
        self.last = (preset, messages)
        if self.exc:
            raise self.exc
        if self.agent:
            _, prompt = flatten_messages(messages)
            async for ev in run_chat(self.agent, prompt):
                yield ev
            return
        for ev in self.events:
            yield ev
