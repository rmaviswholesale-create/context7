"""Live-Agent session store for stateful (memory-preserving) conversations.

Used only when a client sends a session id (``x-agy-session`` header or the
OpenAI ``user`` field). Keeps an already-opened Agent per session so follow-up
turns reuse the SDK's internal conversation state and skip binary startup.

Bounded by TTL and an LRU cap; evicted agents are properly closed. Turns on a
given session are serialized by a per-session lock (``agent.chat`` is one turn
at a time).
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from google.antigravity import Agent

from .presets import Preset
from .runner import AgentRunner


@dataclass
class _Entry:
    agent: Agent
    preset_name: str
    last_used: float
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class SessionStore:
    def __init__(self, runner: AgentRunner, ttl_seconds: int, max_size: int):
        self._runner = runner
        self._ttl = ttl_seconds
        self._max = max_size
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._guard = asyncio.Lock()

    @asynccontextmanager
    async def acquire(self, session_id: str, preset: Preset, system_extra: str | None):
        """Yield an open, exclusively-held Agent for this session id."""
        entry = await self._get_or_create(session_id, preset, system_extra)
        async with entry.lock:
            entry.last_used = time.monotonic()
            yield entry.agent

    async def _get_or_create(
        self, session_id: str, preset: Preset, system_extra: str | None
    ) -> _Entry:
        async with self._guard:
            await self._sweep_expired_locked()
            entry = self._entries.get(session_id)
            # Rebuild if the preset changed for this session id.
            if entry is not None and entry.preset_name != preset.name:
                await self._close_locked(session_id)
                entry = None
            if entry is not None:
                self._entries.move_to_end(session_id)
                return entry

            agent = self._runner.new_agent(preset, system_extra)
            await agent.__aenter__()
            entry = _Entry(agent=agent, preset_name=preset.name, last_used=time.monotonic())
            self._entries[session_id] = entry
            self._entries.move_to_end(session_id)
            await self._evict_over_capacity_locked()
            return entry

    async def _sweep_expired_locked(self) -> None:
        if self._ttl <= 0:
            return
        now = time.monotonic()
        expired = [
            sid for sid, e in self._entries.items() if now - e.last_used > self._ttl
        ]
        for sid in expired:
            await self._close_locked(sid)

    async def _evict_over_capacity_locked(self) -> None:
        while len(self._entries) > self._max:
            oldest = next(iter(self._entries))
            await self._close_locked(oldest)

    async def _close_locked(self, session_id: str) -> None:
        entry = self._entries.pop(session_id, None)
        if entry is None:
            return
        try:
            await entry.agent.__aexit__(None, None, None)
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass

    async def close_all(self) -> None:
        async with self._guard:
            for sid in list(self._entries):
                await self._close_locked(sid)

    def size(self) -> int:
        return len(self._entries)
