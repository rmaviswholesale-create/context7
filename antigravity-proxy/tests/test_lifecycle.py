"""Lifecycle tests with the REAL SDK Agent driving a fake ``localharness`` child process."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import anyio
import pytest

from agy_proxy.lifecycle import harness_pid
from agy_proxy.presets import PresetRegistry
from agy_proxy.runner import AgentRunner

PRESETS = str(Path(__file__).resolve().parent.parent / "presets.yaml")
MSGS = [{"role": "user", "content": "hello"}]


def make_runner(timeout: float = 2.0) -> tuple[AgentRunner, object]:
    reg = PresetRegistry.load(PRESETS, "gemini-3-pro", workspace=None, enable_coder=False)
    runner = AgentRunner(reg, "invalid-key", None, teardown_timeout=timeout)
    runner.lifecycle.kill_after = 0.5
    return runner, reg.resolve("agy-light")


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie (exited, not yet reaped) still answers kill(0); check /proc state.
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ")[-1][0] != "Z"
    except OSError:
        return False


async def until(pred, timeout: float = 15.0):
    deadline = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < deadline, "condition not reached"
        await asyncio.sleep(0.02)


def leftovers(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir())


async def run_and_cancel(runner, preset) -> int:
    """Start a turn (which never completes), cancel it with anyio, return the harness pid."""

    async def consume():
        async for _ in runner.stream_stateless(preset, MSGS):
            pass

    async with anyio.create_task_group() as tg:
        tg.start_soon(consume)
        await until(
            lambda: any(
                (c := getattr(a, "_conversation", None)) is not None and not c.is_idle
                for a in runner.lifecycle.live.snapshot()
            )
        )
        pid = harness_pid(runner.lifecycle.live.snapshot()[0])
        assert pid and alive(pid)
        tg.cancel_scope.cancel()
    return pid


async def test_cancel_mid_turn_closes_agent_process_and_scratch(harness, tmproot):
    runner, preset = make_runner()
    pid = await run_and_cancel(runner, preset)
    assert not alive(pid)
    assert len(runner.lifecycle.live) == 0
    assert leftovers(tmproot) == []  # no agy-proxy-* scratch and no SDK antigravity_* dirs


@pytest.mark.parametrize("mode", ["term", "stubborn"])
async def test_hung_harness_is_killed_by_watchdog(harness, tmproot, mode):
    """The SDK's disconnect blocks the loop in Popen.wait(180); only the thread watchdog
    (SIGTERM, then SIGKILL for `stubborn`) can end it."""
    harness(mode)
    runner, preset = make_runner(timeout=1.0)
    t0 = time.monotonic()
    pid = await run_and_cancel(runner, preset)
    assert time.monotonic() - t0 < 20  # nowhere near the SDK's 180 s wait
    assert not alive(pid)
    assert leftovers(tmproot) == []


async def test_cancel_during_open_leaves_nothing_behind(harness, tmproot):
    runner, preset = make_runner()
    pidfile = harness("ok")

    async def consume():
        async for _ in runner.stream_stateless(preset, MSGS):
            pass

    task = asyncio.ensure_future(consume())
    await until(pidfile.exists)  # the harness process exists; the open is still in flight
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    pid = int(pidfile.read_text())
    assert not alive(pid)
    assert len(runner.lifecycle.live) == 0
    assert leftovers(tmproot) == []


async def test_normal_close_and_reap_registry(harness, tmproot):
    runner, preset = make_runner()
    a1 = runner.new_agent(preset)
    a2 = runner.new_agent(preset)
    await runner.open_agent(a1)
    await runner.open_agent(a2)
    pids = [harness_pid(a1), harness_pid(a2)]
    assert all(pids) and len(runner.lifecycle.live) == 2
    await runner.close_agent(a1)
    assert not alive(pids[0])
    assert await runner.reap() == 1  # lifespan shutdown reaps what is left
    assert not alive(pids[1])
    assert leftovers(tmproot) == []
    await runner.close_agent(a2)  # idempotent


async def test_failed_open_cleans_scratch(monkeypatch, tmproot):
    monkeypatch.setenv("ANTIGRAVITY_HARNESS_PATH", "/nonexistent/localharness")
    runner, preset = make_runner()
    agent = runner.new_agent(preset)
    with pytest.raises(Exception):  # noqa: B017 - FileNotFoundError from Popen
        await runner.open_agent(agent)
    assert len(runner.lifecycle.live) == 0
    assert leftovers(tmproot) == []
