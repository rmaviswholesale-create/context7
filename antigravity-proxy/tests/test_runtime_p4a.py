"""P4a: orphan harnesses, double-cancel teardown, event-loop stalls, slot accounting, admission.

Everything drives the REAL SDK ``Agent`` against the fake ``localharness`` child process
(see ``fake_localharness.py`` for its modes). No model is ever called.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from test_lifecycle import MSGS, PRESETS, alive, leftovers, until

from agy_proxy.errors import ProxyError, RateLimited
from agy_proxy.limits import Admission, AdmissionMiddleware
from agy_proxy.presets import PresetRegistry
from agy_proxy.runner import AgentRunner
from agy_proxy.sessions import SessionStore


def make_runner(timeout: float = 2.0, spawn_timeout: float = 30.0):
    reg = PresetRegistry.load(PRESETS, "gemini-3-pro", workspace=None, enable_coder=False)
    runner = AgentRunner(
        reg, "invalid-key", None, teardown_timeout=timeout, spawn_timeout=spawn_timeout
    )
    runner.lifecycle.kill_after = 0.5
    return runner, reg.resolve("agy-light")


def pid_of(pidfile: Path) -> int:
    return int(pidfile.read_text())


async def lag_monitor(samples: list[float], stop: asyncio.Event) -> None:
    while not stop.is_set():
        t = time.monotonic()
        await asyncio.sleep(0.01)
        samples.append(time.monotonic() - t - 0.01)


# -- (1) orphan on abandoned / failed / garbage open ------------------------------------------


async def test_cancel_while_harness_never_answers_init_leaves_no_orphan(harness, tmproot):
    pidfile = harness("noinit")
    runner, preset = make_runner()

    async def consume():
        async for _ in runner.stream_stateless(preset, MSGS):
            pass

    task = asyncio.ensure_future(consume())
    await until(pidfile.exists)
    await asyncio.sleep(0.5)  # handshake done, init response never comes
    pid = pid_of(pidfile)
    assert alive(pid)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await until(lambda: not alive(pid), 10)
    assert len(runner.lifecycle.live) == 0
    assert leftovers(tmproot) == []


async def test_garbage_handshake_kills_the_harness(harness, tmproot):
    pidfile = harness("garbage")
    runner, preset = make_runner()
    agent = runner.new_agent(preset)
    with pytest.raises(Exception):  # noqa: B017 - protobuf DecodeError from the SDK
        await asyncio.wait_for(runner.open_agent(agent), 20)
    await until(pidfile.exists)
    assert not alive(pid_of(pidfile))
    assert len(runner.lifecycle.live) == 0
    assert leftovers(tmproot) == []


@pytest.mark.parametrize("mode", ["silent", "noinit", "noinit_stubborn"])
async def test_spawn_watchdog_kills_and_raises_5xx(harness, tmproot, mode):
    """A harness that never completes startup is killed after spawn_timeout -> 5xx."""
    pidfile = harness(mode)
    runner, preset = make_runner(spawn_timeout=1.0)
    agent = runner.new_agent(preset)
    t0 = time.monotonic()
    with pytest.raises(ProxyError) as ei:
        await asyncio.wait_for(runner.open_agent(agent), 20)
    assert 500 <= ei.value.status < 600
    assert time.monotonic() - t0 < 15
    assert not alive(pid_of(pidfile))
    assert len(runner.lifecycle.live) == 0
    assert leftovers(tmproot) == []


async def test_cancel_during_blocking_spawn_leaves_no_orphan(harness, tmproot, monkeypatch):
    """Cancelled while the SDK is still blocked reading the harness's OutputConfig."""
    pidfile = harness("ok")
    monkeypatch.setenv("FAKE_DELAY", "1.0")
    runner, preset = make_runner()

    async def consume():
        async for _ in runner.stream_stateless(preset, MSGS):
            pass

    task = asyncio.ensure_future(consume())
    await until(pidfile.exists)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await until(lambda: not alive(pid_of(pidfile)), 10)
    assert leftovers(tmproot) == []


# -- (2) double cancel / shutdown during close ------------------------------------------------


@pytest.mark.parametrize("mode", ["ok", "term"])
async def test_double_cancel_during_teardown_still_kills(harness, tmproot, mode):
    harness(mode)
    runner, preset = make_runner(timeout=1.0)

    async def consume():
        async for _ in runner.stream_stateless(preset, MSGS):
            pass

    task = asyncio.ensure_future(consume())
    await until(
        lambda: any(
            (c := getattr(a, "_conversation", None)) is not None and not c.is_idle
            for a in runner.lifecycle.live.snapshot()
        )
    )
    pid = next(iter(_pids(runner)))
    task.cancel()
    await asyncio.sleep(0.2)  # inside teardown
    task.cancel()  # second native cancel (timeout then parent cancel)
    with pytest.raises(asyncio.CancelledError):
        await task
    await until(lambda: not alive(pid), 15)
    await until(lambda: len(runner.lifecycle.live) == 0 and leftovers(tmproot) == [], 15)


def _pids(runner):
    from agy_proxy.lifecycle import harness_pid

    return {harness_pid(a) for a in runner.lifecycle.live.snapshot()} - {None}


async def test_repeated_cancel_storm_never_leaks(harness, tmproot):
    harness("ok")
    for delay in (0.0, 0.003, 0.01, 0.03, 0.08):
        runner, preset = make_runner(timeout=1.0)

        async def consume(runner=runner, preset=preset):
            async for _ in runner.stream_stateless(preset, MSGS):
                pass

        task = asyncio.ensure_future(consume())
        await until(
            lambda r=runner: any(
                (c := getattr(a, "_conversation", None)) is not None and not c.is_idle
                for a in r.lifecycle.live.snapshot()
            )
        )
        pids = _pids(runner)
        task.cancel()
        await asyncio.sleep(delay)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await until(lambda p=pids: not any(alive(x) for x in p), 15)
        await until(lambda r=runner: len(r.lifecycle.live) == 0, 15)
    await until(lambda: leftovers(tmproot) == [], 15)


async def test_shutdown_during_reaper_close_completes_teardown(harness, tmproot):
    """close_all cancels the reaper mid-close; the harness must still die and the registry
    entry must survive until it is confirmed dead."""
    harness("hangws")
    runner, preset = make_runner(timeout=5.0)
    store = SessionStore(runner, ttl_seconds=0.2, max_size=4)
    store.start()
    async with store.acquire("k", preset, None) as lease:
        lease.complete()
    pid = next(iter(_pids(runner)))
    await until(lambda: bool(store._retiring))
    await asyncio.sleep(0.1)
    assert alive(pid)
    await store.close_all()
    await runner.reap()
    assert not alive(pid)
    assert len(runner.lifecycle.live) == 0
    assert leftovers(tmproot) == []


async def test_close_all_cancelling_a_creation_leaves_no_orphan(harness, tmproot):
    pidfile = harness("noinit")
    runner, preset = make_runner(timeout=2.0)
    store = SessionStore(runner, ttl_seconds=100, max_size=4)

    async def use():
        async with store.acquire("k", preset, None) as lease:
            lease.complete()

    task = asyncio.ensure_future(use())
    await until(pidfile.exists)
    await asyncio.sleep(0.4)
    await store.close_all()
    await runner.reap()
    with pytest.raises(BaseException):  # noqa: B017 - cancelled / upstream error
        await task
    assert not alive(pid_of(pidfile))
    assert leftovers(tmproot) == []


async def test_registry_entry_kept_until_process_confirmed_dead(harness, tmproot):
    harness("stubborn")
    runner, preset = make_runner(timeout=1.0)
    agent = runner.new_agent(preset)
    await runner.open_agent(agent)
    pid = next(iter(_pids(runner)))
    closer = asyncio.ensure_future(runner.close_agent(agent))
    await asyncio.sleep(0.3)  # harness ignores stdin EOF; teardown is waiting
    assert alive(pid) and len(runner.lifecycle.live) == 1
    closer.cancel()
    closer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closer
    await until(lambda: not alive(pid), 15)
    await until(lambda: len(runner.lifecycle.live) == 0 and leftovers(tmproot) == [], 15)


# -- (3) event loop stays responsive ----------------------------------------------------------


async def test_spawn_does_not_stall_the_event_loop(harness, tmproot, monkeypatch):
    harness("ok")
    monkeypatch.setenv("FAKE_DELAY", "0.8")
    runner, preset = make_runner()
    samples: list[float] = []
    stop = asyncio.Event()
    mon = asyncio.ensure_future(lag_monitor(samples, stop))
    await asyncio.sleep(0.1)
    samples.clear()
    agent = runner.new_agent(preset)
    t0 = time.monotonic()
    await runner.open_agent(agent)
    assert time.monotonic() - t0 >= 0.7  # the delay really was waited for
    stop.set()
    await mon
    assert max(samples) < 0.1, f"event loop stalled {max(samples):.3f}s during spawn"
    await runner.close_agent(agent)


async def test_close_does_not_stall_the_event_loop(harness, tmproot, monkeypatch):
    harness("ok")
    monkeypatch.setenv("FAKE_EXIT_DELAY", "0.8")
    runner, preset = make_runner(timeout=5.0)
    agent = runner.new_agent(preset)
    await runner.open_agent(agent)
    samples: list[float] = []
    stop = asyncio.Event()
    mon = asyncio.ensure_future(lag_monitor(samples, stop))
    await asyncio.sleep(0.1)
    samples.clear()
    t0 = time.monotonic()
    await runner.close_agent(agent)
    assert time.monotonic() - t0 >= 0.7
    stop.set()
    await mon
    assert max(samples) < 0.1, f"event loop stalled {max(samples):.3f}s during close"
    assert leftovers(tmproot) == []


# -- (4) slot accounting ----------------------------------------------------------------------


class _FakeRunner:
    def __init__(self):
        self.opened = 0
        self.closed = 0

    def new_agent(self, preset, system_extra, long_lived=False):
        return NS()

    async def open_agent(self, agent):
        await asyncio.sleep(0.02)
        self.opened += 1

    async def close_agent(self, agent):
        await asyncio.sleep(0.02)
        self.closed += 1


async def test_preset_swap_with_waiters_does_not_429_at_session_max_1():
    runner = _FakeRunner()
    store = SessionStore(runner, 100, 1)

    async def use(preset_name: str, hold: float):
        async with store.acquire("k", NS(name=preset_name), None) as lease:
            lease.complete()
            await asyncio.sleep(hold)

    first = asyncio.ensure_future(use("x", 0.2))
    await asyncio.sleep(0.1)
    rest = [asyncio.ensure_future(use("y", 0.05)) for _ in range(3)]
    results = await asyncio.gather(first, *rest, return_exceptions=True)
    assert not [r for r in results if isinstance(r, RateLimited)], results
    assert all(r is None for r in results), results
    assert runner.opened == 2 and runner.closed == 1
    await store.close_all()
    assert runner.opened == runner.closed


# -- (5) admission takes its permit after the body is read ------------------------------------


async def test_slow_body_clients_do_not_hold_permits():
    adm = Admission(2, turn_timeout=60, body_timeout=30)

    async def reader(scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    mw = AdmissionMiddleware(reader, adm)
    scope = {"type": "http", "method": "POST", "path": "/v1/messages", "headers": []}

    async def never():
        await asyncio.sleep(60)

    slow = [
        asyncio.ensure_future(mw(scope, never, lambda m: asyncio.sleep(0))) for _ in range(5)
    ]
    await asyncio.sleep(0.1)
    assert adm.inflight == 0
    sent: list[dict] = []

    async def send(m):
        sent.append(m)

    feed = [{"type": "http.request", "body": b"{}", "more_body": False}]

    async def receive():
        return feed.pop(0) if feed else {"type": "http.disconnect"}

    await asyncio.wait_for(mw(scope, receive, send), 2)
    assert sent[0]["status"] == 200
    for t in slow:
        t.cancel()
    await asyncio.gather(*slow, return_exceptions=True)
    assert adm.inflight == 0


async def test_body_read_deadline_gives_408_and_frees_nothing_held():
    adm = Admission(1, turn_timeout=60, body_timeout=0.2)
    mw = AdmissionMiddleware(lambda s, r, se: asyncio.sleep(0), adm)
    scope = {"type": "http", "method": "POST", "path": "/v1/messages", "headers": []}
    sent: list[dict] = []

    async def send(m):
        sent.append(m)

    async def receive():
        await asyncio.sleep(60)

    await asyncio.wait_for(mw(scope, receive, send), 3)
    assert sent[0]["status"] == 408
    assert adm.inflight == 0


async def test_body_is_replayed_to_the_app_and_full_house_still_429s():
    adm = Admission(1, turn_timeout=60, body_timeout=5)
    seen: list[bytes] = []
    gate = asyncio.Event()

    async def app(scope, receive, send):
        msg = await receive()
        seen.append(msg["body"])
        await gate.wait()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    mw = AdmissionMiddleware(app, adm)
    scope = {"type": "http", "method": "POST", "path": "/v1/messages", "headers": []}

    def feed(chunks):
        q = list(chunks)

        async def receive():
            if q:
                return q.pop(0)
            await asyncio.sleep(60)

        return receive

    chunks = [
        {"type": "http.request", "body": b"ab", "more_body": True},
        {"type": "http.request", "body": b"cd", "more_body": False},
    ]
    out1: list[dict] = []
    out2: list[dict] = []

    async def send1(m):
        out1.append(m)

    async def send2(m):
        out2.append(m)

    t1 = asyncio.ensure_future(mw(scope, feed(chunks), send1))
    await asyncio.sleep(0.1)
    assert seen == [b"abcd"]
    await mw(scope, feed(chunks), send2)
    assert out2[0]["status"] == 429
    gate.set()
    await t1
    assert out1[0]["status"] == 200


# -- (6) config + uvicorn knobs ---------------------------------------------------------------


def test_new_settings_defaults_and_validation():
    from agy_proxy.config import ConfigError, Settings

    s = Settings.from_env({})
    assert (s.spawn_timeout, s.body_timeout, s.keepalive_timeout) == (30, 60, 5)
    assert s.uvicorn_limit_concurrency > s.max_concurrency
    s = Settings.from_env(
        {
            "AGY_SPAWN_TIMEOUT": "7",
            "AGY_BODY_TIMEOUT": "9",
            "AGY_KEEPALIVE_TIMEOUT": "2",
            "AGY_LIMIT_CONCURRENCY": "11",
        }
    )
    assert (s.spawn_timeout, s.body_timeout, s.keepalive_timeout) == (7, 9, 2)
    assert s.uvicorn_limit_concurrency == 11
    with pytest.raises(ConfigError, match="AGY_SPAWN_TIMEOUT"):
        Settings.from_env({"AGY_SPAWN_TIMEOUT": "0"})


def test_main_passes_uvicorn_limits(monkeypatch):
    import uvicorn

    from agy_proxy import __main__ as entry

    captured: dict = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: captured.update(kw))
    monkeypatch.setenv("AGY_PROXY_TOKEN", "t")
    monkeypatch.setenv("AGY_KEEPALIVE_TIMEOUT", "3")
    monkeypatch.setenv("AGY_LIMIT_CONCURRENCY", "17")
    entry.main()
    assert captured["timeout_keep_alive"] == 3
    assert captured["limit_concurrency"] == 17


# -- fallback when the SDK spawn patch cannot be installed ------------------------------------


@pytest.fixture
def unpatched_sdk():
    from agy_proxy import spawn

    spawn.uninstall()
    yield
    assert spawn.install()


@pytest.mark.parametrize("mode", ["noinit", "garbage"])
async def test_pid_snapshot_fallback_kills_new_children(harness, tmproot, unpatched_sdk, mode):
    pidfile = harness(mode)
    runner, preset = make_runner(spawn_timeout=1.0)
    agent = runner.new_agent(preset)
    with pytest.raises(Exception):  # noqa: B017 - timeout (5xx) or protobuf DecodeError
        await asyncio.wait_for(runner.open_agent(agent), 30)
    await until(pidfile.exists)
    await until(lambda: not alive(pid_of(pidfile)), 10)
    assert len(runner.lifecycle.live) == 0
    assert leftovers(tmproot) == []


async def test_cancel_immediately_after_open_starts_leaves_nothing(harness, tmproot):
    pidfile = harness("noinit")
    runner, preset = make_runner()
    agent = runner.new_agent(preset)
    task = asyncio.ensure_future(runner.open_agent(agent))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.8)  # a late Popen in the worker thread must be killed on arrival
    if pidfile.exists():
        assert not alive(pid_of(pidfile))
    assert len(runner.lifecycle.live) == 0
    assert leftovers(tmproot) == []
