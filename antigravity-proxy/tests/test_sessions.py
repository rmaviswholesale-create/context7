from __future__ import annotations

import asyncio

import pytest

from agy_proxy.errors import RateLimited
from agy_proxy.events import DoneEvent, TextEvent
from agy_proxy.presets import Preset
from agy_proxy.sessions import SessionStore, stream_session_turn


class FakeAgent:
    def __init__(self, preset_name: str):
        self.preset_name = preset_name
        self.opened = False
        self.closed = False
        self.close_calls = 0
        self.dead = False


class FakeRunner:
    def __init__(self):
        self.created: list[FakeAgent] = []
        self.open_gate: asyncio.Event | None = None  # when set, open_agent waits for it
        self.open_gates: dict[str, asyncio.Event] = {}  # per preset-name gates
        self.fail_next_open = False
        self.busy_streams: list[tuple] = []

    def new_agent(self, preset: Preset, system_extra=None, **_kw) -> FakeAgent:
        a = FakeAgent(preset.name)
        self.created.append(a)
        return a

    async def open_agent(self, agent: FakeAgent) -> None:
        gate = self.open_gates.get(agent.preset_name) or self.open_gate
        if gate is not None:
            await gate.wait()
        if self.fail_next_open:
            self.fail_next_open = False
            raise RuntimeError("harness failed to start")
        agent.opened = True

    async def close_agent(self, agent: FakeAgent) -> None:
        agent.close_calls += 1
        agent.closed = True

    def is_usable(self, agent: FakeAgent) -> bool:
        return not agent.dead and not agent.closed

    async def stream_on_agent(self, agent, messages, is_new=False):
        self.busy_streams.append((agent, list(messages), is_new))
        yield TextEvent("hi")
        yield DoneEvent(stop_reason=None, input_tokens=1, output_tokens=1)


def P(name: str) -> Preset:
    return Preset(name=name)


async def use(store, key, preset="p", ok=True):
    async with store.acquire(key, P(preset), None) as lease:
        if ok:
            lease.complete()
        return lease


async def test_reuses_agent_and_is_new_flag():
    store = SessionStore(FakeRunner(), ttl_seconds=1000, max_size=10)
    l1 = await use(store, "s1")
    l2 = await use(store, "s1")
    assert l1.agent is l2.agent and store.size() == 1
    assert (l1.is_new, l2.is_new) == (True, False)


async def test_preset_change_rebuilds_and_closes_old():
    store = SessionStore(FakeRunner(), ttl_seconds=1000, max_size=10)
    l1 = await use(store, "s1", "agy-coder")
    l2 = await use(store, "s1", "agy-fast")
    assert l1.agent is not l2.agent and l1.agent.closed
    assert l2.agent.preset_name == "agy-fast" and l2.is_new


async def test_lru_eviction_closes_oldest_idle():
    store = SessionStore(FakeRunner(), ttl_seconds=1000, max_size=2)
    a1 = (await use(store, "s1")).agent
    a2 = (await use(store, "s2")).agent
    await use(store, "s3")
    assert store.size() == 2 and a1.closed and not a2.closed


async def test_ttl_expiry_closes_idle(monkeypatch):
    import agy_proxy.sessions as s

    t = {"now": 1000.0}
    monkeypatch.setattr(s.time, "monotonic", lambda: t["now"])
    store = SessionStore(FakeRunner(), ttl_seconds=10, max_size=10)
    a1 = (await use(store, "s1")).agent
    t["now"] = 1020.0
    assert await store.reap_once() == 1
    assert a1.closed and store.size() == 0


async def test_close_all():
    store = SessionStore(FakeRunner(), ttl_seconds=1000, max_size=10)
    a1 = (await use(store, "s1")).agent
    await store.close_all()
    assert a1.closed and store.size() == 0


async def test_ttl_shorter_than_turn_never_closes_mid_turn():
    store = SessionStore(FakeRunner(), ttl_seconds=0.05, max_size=10)
    async with store.acquire("s1", P("p"), None) as lease:
        lease.complete()
        await asyncio.sleep(0.2)  # the "turn" outlives the TTL several times over
        await store.reap_once()
        await store.reap_once()
        assert not lease.agent.closed and store.size() == 1
    # last_used was refreshed when the turn ended, so it is not instantly expired either
    await store.reap_once()
    assert not lease.agent.closed
    await asyncio.sleep(0.1)
    await store.reap_once()
    assert lease.agent.closed


async def test_background_reaper_expires_idle_entries():
    store = SessionStore(FakeRunner(), ttl_seconds=0.1, max_size=10)
    store.start()
    try:
        agent = (await use(store, "s1")).agent
        for _ in range(100):
            if agent.closed:
                break
            await asyncio.sleep(0.02)
        assert agent.closed and store.size() == 0
    finally:
        await store.close_all()


async def test_capacity_with_all_sessions_busy_is_429_and_keeps_running_turns():
    store = SessionStore(FakeRunner(), ttl_seconds=1000, max_size=1)
    async with store.acquire("busy", P("p"), None) as lease:
        lease.complete()
        with pytest.raises(RateLimited):
            async with store.acquire("other", P("p"), None):
                pass
        assert not lease.agent.closed
    # idle now: a new session may evict it
    await use(store, "other")
    assert lease.agent.closed


async def test_two_busy_sessions_max_size_1_second_creation_does_not_block():
    store = SessionStore(FakeRunner(), ttl_seconds=1000, max_size=1)
    started = asyncio.Event()
    release = asyncio.Event()

    async def hold():
        async with store.acquire("a", P("p"), None) as lease:
            lease.complete()
            started.set()
            await release.wait()

    t = asyncio.ensure_future(hold())
    await started.wait()
    with pytest.raises(RateLimited):
        await asyncio.wait_for(use(store, "b"), 1)
    release.set()
    await t


async def test_preset_swap_mid_turn_waits_for_turn_then_replaces():
    runner = FakeRunner()
    store = SessionStore(runner, ttl_seconds=1000, max_size=10)
    started, release = asyncio.Event(), asyncio.Event()
    events: list[str] = []

    async def first():
        async with store.acquire("s", P("p1"), None) as lease:
            lease.complete()
            started.set()
            await release.wait()
            events.append(f"first-done closed={lease.agent.closed}")
            return lease.agent

    t1 = asyncio.ensure_future(first())
    await started.wait()
    t2 = asyncio.ensure_future(use(store, "s", "p2"))
    await asyncio.sleep(0.05)
    assert not t2.done() and not runner.created[0].closed  # waits; old agent untouched
    release.set()
    old = await t1
    lease2 = await t2
    assert events == ["first-done closed=False"]  # not closed under the running turn
    assert old.closed and old.close_calls == 1
    assert lease2.agent.preset_name == "p2" and lease2.is_new
    assert len(runner.created) == 2 and store.size() == 1


async def test_same_key_concurrent_creation_builds_one_agent():
    runner = FakeRunner()
    runner.open_gate = asyncio.Event()
    store = SessionStore(runner, ttl_seconds=1000, max_size=10)
    tasks = [asyncio.ensure_future(use(store, "k")) for _ in range(5)]
    await asyncio.sleep(0.05)
    runner.open_gate.set()
    leases = await asyncio.gather(*tasks)
    assert len(runner.created) == 1
    assert {id(entry.agent) for entry in leases} == {id(runner.created[0])}
    assert sorted(entry.is_new for entry in leases) == [False] * 4 + [True]


async def test_slow_creation_for_one_key_does_not_block_another():
    runner = FakeRunner()
    runner.open_gates["slow"] = asyncio.Event()
    store = SessionStore(runner, ttl_seconds=1000, max_size=10)
    slow = asyncio.ensure_future(use(store, "a", "slow"))
    await asyncio.sleep(0.02)
    fast = await asyncio.wait_for(use(store, "b", "fast"), 1)
    assert fast.agent.opened and not slow.done()
    runner.open_gates["slow"].set()
    assert (await slow).agent.opened


async def test_failed_creation_is_not_cached():
    runner = FakeRunner()
    runner.fail_next_open = True
    store = SessionStore(runner, ttl_seconds=1000, max_size=10)
    with pytest.raises(RuntimeError):
        await use(store, "k")
    assert store.size() == 0
    lease = await use(store, "k")  # retried from scratch
    assert lease.is_new and store.size() == 1 and len(runner.created) == 2


async def test_failed_creation_releases_capacity():
    runner = FakeRunner()
    runner.fail_next_open = True
    store = SessionStore(runner, ttl_seconds=1000, max_size=1)
    with pytest.raises(RuntimeError):
        await use(store, "a")
    await use(store, "b")  # would 429 if the failed slot leaked


async def test_waiter_cancelled_during_creation_leaves_valid_idle_entry():
    runner = FakeRunner()
    runner.open_gate = asyncio.Event()
    store = SessionStore(runner, ttl_seconds=1000, max_size=1)
    t = asyncio.ensure_future(use(store, "k"))
    await asyncio.sleep(0.02)
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    runner.open_gate.set()
    await asyncio.sleep(0.02)
    assert store.size() == 1  # the built agent is kept, idle, not leaked
    lease = await use(store, "k")
    assert lease.is_new and len(runner.created) == 1


async def test_incomplete_turn_evicts_and_complete_turn_keeps():
    runner = FakeRunner()
    store = SessionStore(runner, ttl_seconds=1000, max_size=10)
    ok = await use(store, "ok", ok=True)
    bad = await use(store, "bad", ok=False)  # error / disconnect / cancel => never completed
    assert not ok.agent.closed and bad.agent.closed
    assert store.size() == 1
    again = await use(store, "bad")
    assert again.agent is not bad.agent and again.is_new


async def test_dead_harness_is_replaced():
    store = SessionStore(FakeRunner(), ttl_seconds=1000, max_size=10)
    l1 = await use(store, "k")
    l1.agent.dead = True
    l2 = await use(store, "k")
    assert l2.agent is not l1.agent and l1.agent.closed and l2.is_new


async def test_retired_busy_entry_closed_once_by_last_releaser():
    runner = FakeRunner()
    store = SessionStore(runner, ttl_seconds=1000, max_size=10)
    started, release = asyncio.Event(), asyncio.Event()

    async def failing():
        async with store.acquire("k", P("p"), None) as lease:  # never completed => evicted
            started.set()
            await release.wait()
            return lease.agent

    t1 = asyncio.ensure_future(failing())
    await started.wait()
    waiter = asyncio.ensure_future(use(store, "k"))  # queued behind the failing turn
    await asyncio.sleep(0.02)
    release.set()
    old = await t1
    lease = await waiter
    assert old.closed and old.close_calls == 1
    assert lease.agent is not old and lease.is_new


async def test_stream_session_turn_completes_on_done_and_evicts_when_abandoned():
    runner = FakeRunner()
    store = SessionStore(runner, ttl_seconds=1000, max_size=10)
    msgs = [{"role": "user", "content": "hi"}]

    gen = stream_session_turn(runner, store, "k", P("p"), msgs, None)
    seen = [ev async for ev in gen]
    assert isinstance(seen[-1], DoneEvent) and store.size() == 1
    agent, _, is_new = runner.busy_streams[-1]
    assert is_new is True

    gen = stream_session_turn(runner, store, "k", P("p"), msgs, None)
    await gen.__anext__()  # consumer walks away after the first event
    await gen.aclose()
    assert runner.busy_streams[-1][2] is False  # reused agent => trailing messages only
    assert agent.closed and store.size() == 0


async def test_close_all_aborts_pending_creation():
    runner = FakeRunner()
    runner.open_gate = asyncio.Event()
    store = SessionStore(runner, ttl_seconds=1000, max_size=10)
    t = asyncio.ensure_future(use(store, "k"))
    await asyncio.sleep(0.02)
    await store.close_all()
    with pytest.raises(asyncio.CancelledError):
        await t
    assert store.size() == 0
