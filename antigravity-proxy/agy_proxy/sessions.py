"""Live-Agent session store for stateful (memory-preserving) conversations.

A session is selected by the ``x-agy-session`` header (see ``sessionkey``) and keeps one
opened Agent so follow-up turns reuse the SDK's conversation state and skip harness startup.

Concurrency model (single event loop; every mutation of the dicts below happens in a
section with no ``await``, so no lock is needed around them):

* per-key *creation records* -- concurrent first requests for a key share ONE agent build;
  a slow build for one key never blocks other keys; a failed build is never cached;
* an *inflight refcount* per entry -- an entry is never closed under a running or queued
  turn. Evicting a busy entry only *retires* it (removed from the index); whoever drops the
  last reference closes it;
* a per-entry lock serialises turns (``agent.chat`` is one turn at a time); after getting
  the lock a request re-validates that its entry is still THE entry for the key (it may have
  been retired by a preset change, an error or capacity pressure while it waited);
* ``last_used`` is refreshed when a turn *ends*, and busy entries are never expired, so a
  TTL shorter than a turn cannot close an agent mid-turn;
* at capacity the least-recently-used *idle* entry is evicted; when every entry is busy a
  new session is refused with 429 instead of killing someone's running turn;
* any turn that did not complete cleanly (error, disconnect, cancellation), or an agent
  whose harness died, is evicted rather than reused.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import anyio

from .errors import RateLimited
from .events import DoneEvent, Event
from .presets import Preset

logger = logging.getLogger("agy_proxy.sessions")


@dataclass(eq=False)
class _Entry:
    key: str
    agent: Any
    preset_name: str
    last_used: float
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    inflight: int = 0  # requests that hold a reference (running or waiting for the lock)
    turns: int = 0  # turns started on this agent (0 => brand new)
    retiring: bool = False  # removed from the index; closed by the last releaser
    closing: bool = False


@dataclass(eq=False)
class _Creation:
    fut: asyncio.Future[Any] | None = None
    preset_name: str = ""
    waiters: int = 0
    entry: _Entry | None = None


class SessionLease:
    """What a request gets for one turn: the open agent and whether it is brand new."""

    def __init__(self, agent: Any, is_new: bool):
        self.agent = agent
        self.is_new = is_new
        self.completed = False

    def complete(self) -> None:
        """Mark the turn as finished cleanly; otherwise the agent is evicted on release."""
        self.completed = True


class SessionStore:
    def __init__(self, runner: Any, ttl_seconds: float, max_size: int):
        self._runner = runner
        self._ttl = ttl_seconds
        self._max = max_size
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._creating: dict[str, _Creation] = {}
        self._retiring: set[_Entry] = set()
        self._bg: set[asyncio.Task[Any]] = set()
        self._reaper: asyncio.Task[Any] | None = None
        self._usable: Callable[[Any], bool] = getattr(runner, "is_usable", lambda _a: True)

    # -- public API ------------------------------------------------------------------------

    @asynccontextmanager
    async def acquire(
        self, key: str, preset: Preset, system_extra: str | None
    ) -> AsyncIterator[SessionLease]:
        """Hold the session's agent exclusively for one turn."""
        self._expire_idle()
        while True:
            entry = await self._pin(key, preset, system_extra)
            try:
                await entry.lock.acquire()
            except BaseException:
                await self._unpin(entry)
                raise
            try:
                if self._entries.get(key) is not entry:
                    continue  # retired while we waited: start over
                if entry.preset_name != preset.name or not self._usable(entry.agent):
                    # Preset changed / harness dead: rebuild. We hold the lock, so nobody is
                    # using the agent (waiters only re-check the index): close it NOW so its
                    # slot is free for the replacement even with other waiters still pinned.
                    self._retire(entry)
                    await self._close_entry(entry)
                    continue
                self._entries.move_to_end(key)
                lease = SessionLease(entry.agent, is_new=entry.turns == 0)
                entry.turns += 1
                try:
                    yield lease
                finally:
                    entry.last_used = time.monotonic()
                    if not lease.completed or not self._usable(entry.agent):
                        self._retire(entry)
                        await self._close_entry(entry)  # lock held: free the slot for waiters
                return
            finally:
                entry.lock.release()
                await self._unpin(entry)

    def size(self) -> int:
        return len(self._entries)

    def start(self) -> None:
        """Start the background reaper (idempotent). Needs a running loop."""
        if self._reaper is None or self._reaper.done():
            self._reaper = asyncio.get_running_loop().create_task(self._reap_loop())

    async def reap_once(self) -> int:
        """Close every idle entry past its TTL (the reaper's body; callable directly)."""
        victims = self._expire_idle(spawn=False)
        for entry in victims:
            await self._close_entry(entry)
        return len(victims)

    async def close_all(self) -> None:
        """Shutdown: stop the reaper, abort creations, close everything (busy or not)."""
        if self._reaper is not None:
            self._reaper.cancel()
            await asyncio.gather(self._reaper, return_exceptions=True)
            self._reaper = None
        creations = [c.fut for c in self._creating.values() if c.fut is not None]
        for fut in creations:
            fut.cancel()
        await asyncio.gather(*creations, return_exceptions=True)
        with anyio.CancelScope(shield=True):
            entries = list(self._entries.values())
            for entry in entries:
                self._retire(entry)
            for entry in entries + list(self._retiring):
                await self._close_entry(entry)
            if self._bg:
                await asyncio.gather(*self._bg, return_exceptions=True)

    # -- creation --------------------------------------------------------------------------

    async def _pin(self, key: str, preset: Preset, system_extra: str | None) -> _Entry:
        """Return the key's entry (creating it if needed) with our reference counted."""
        entry = self._entries.get(key)
        if entry is not None:
            entry.inflight += 1
            return entry
        rec = self._creating.get(key)
        if rec is None:
            victims = self._reserve_slot()
            rec = _Creation(preset_name=preset.name)
            rec.fut = asyncio.ensure_future(self._build(victims, preset, system_extra))
            rec.fut.add_done_callback(lambda f, k=key, r=rec: self._on_built(k, r, f))
            self._creating[key] = rec
        rec.waiters += 1
        try:
            await asyncio.shield(rec.fut)  # type: ignore[arg-type]
        except BaseException:
            rec.waiters -= 1
            if rec.entry is not None:  # built meanwhile and counted us: hand the pin back
                await self._unpin(rec.entry)
            raise
        assert rec.entry is not None
        return rec.entry

    def _reserve_slot(self) -> list[_Entry]:
        """Make room for one more agent; returns idle entries (already retired) to close."""
        victims: list[_Entry] = []
        # Live harnesses = indexed + being built + retired-but-not-yet-closed. Retiring a
        # victim moves it between sets without changing this total; it frees a slot only once
        # closed, which _build does before spawning the replacement.
        total = len(self._entries) + len(self._creating) + len(self._retiring)
        while total - len(victims) >= self._max:
            idle = next((e for e in self._entries.values() if e.inflight == 0), None)
            if idle is None:
                for victim in victims:  # nothing freed after all: put them back
                    victim.retiring = False
                    self._retiring.discard(victim)
                    self._entries[victim.key] = victim
                raise RateLimited(
                    f"all {self._max} live sessions are busy; retry shortly or end a session"
                )
            self._retire(idle)
            victims.append(idle)
        return victims

    async def _build(
        self, victims: list[_Entry], preset: Preset, system_extra: str | None
    ) -> Any:
        for victim in victims:  # free the process slot before spawning another harness
            await self._close_entry(victim)
        agent = self._runner.new_agent(preset, system_extra, long_lived=True)
        await self._runner.open_agent(agent)  # cleans up after itself on failure/cancel
        return agent

    def _on_built(self, key: str, rec: _Creation, fut: asyncio.Future[Any]) -> None:
        if self._creating.get(key) is rec:
            del self._creating[key]
        if fut.cancelled():
            return
        if fut.exception() is not None:  # failed build: nothing cached (waiters see the error)
            return
        entry = _Entry(
            key=key,
            agent=fut.result(),
            preset_name=rec.preset_name,
            last_used=time.monotonic(),
        )
        entry.inflight = rec.waiters
        rec.entry = entry
        self._entries[key] = entry
        self._entries.move_to_end(key)

    # -- retirement / release --------------------------------------------------------------

    def _retire(self, entry: _Entry) -> None:
        """Take the entry out of the index; the last reference holder closes it."""
        if self._entries.get(entry.key) is entry:
            del self._entries[entry.key]
        if not entry.retiring:
            entry.retiring = True
            self._retiring.add(entry)

    async def _unpin(self, entry: _Entry) -> None:
        with anyio.CancelScope(shield=True):
            entry.inflight -= 1
            if entry.retiring and entry.inflight <= 0:
                await self._close_entry(entry)

    async def _close_entry(self, entry: _Entry) -> None:
        if entry.closing:
            return
        entry.closing = True
        self._retiring.add(entry)
        try:
            await self._runner.close_agent(entry.agent)
        except Exception:  # noqa: BLE001 - best-effort teardown
            logger.exception("closing session agent failed")
        finally:
            self._retiring.discard(entry)

    # -- expiry ----------------------------------------------------------------------------

    def _expire_idle(self, spawn: bool = True) -> list[_Entry]:
        """Retire idle entries older than the TTL. Busy entries are never expired."""
        if self._ttl <= 0:
            return []
        now = time.monotonic()
        expired = [
            e
            for e in self._entries.values()
            if e.inflight == 0 and not e.lock.locked() and now - e.last_used > self._ttl
        ]
        for entry in expired:
            self._retire(entry)
        if spawn:
            for entry in expired:
                task = asyncio.ensure_future(self._close_entry(entry))
                self._bg.add(task)
                task.add_done_callback(self._bg.discard)
        return expired

    async def _reap_loop(self) -> None:
        interval = max(self._ttl / 4, 0.05)
        while True:
            await asyncio.sleep(interval)
            try:
                await self.reap_once()
            except Exception:  # noqa: BLE001
                logger.exception("session reaper failed")


async def stream_session_turn(
    runner: Any,
    store: SessionStore,
    key: str,
    preset: Preset,
    messages: list[Any],
    system_extra: str | None,
) -> AsyncIterator[Event]:
    """One turn on a live session: the single call site adapters use for stateful requests.

    The lease is marked complete when the terminal ``DoneEvent`` is produced; any other way
    out (error event, exception, consumer stops early, cancellation) makes the store evict
    the agent instead of reusing a possibly half-finished conversation.
    """
    async with store.acquire(key, preset, system_extra) as lease:
        async for ev in runner.stream_on_agent(lease.agent, messages, is_new=lease.is_new):
            if isinstance(ev, DoneEvent):
                lease.complete()
            yield ev
