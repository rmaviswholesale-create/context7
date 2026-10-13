"""Harness process helpers: the only place that touches SDK-private connection attributes.

``Agent.__aexit__`` -> ``LocalConnection.disconnect()`` is not safe to rely on alone:

* it ends with a *synchronous* ``Popen.wait(timeout=180)`` that blocks the whole event
  loop, so an ``asyncio.wait_for`` can never fire while a hung harness is being waited for;
* if the waiting is abandoned (cancellation, timeout) the harness child is orphaned.

``Agent.__aenter__`` is no better: the harness ``Popen`` is created inside the SDK call (see
``spawn.py`` for the off-loop, tracked replacement), so a cancelled or failed open cannot find
the child it started.

The harness ``Popen`` lives at ``agent.conversation.connection._process`` (SDK 0.1.21,
see ``docs/SDK_NOTES.md``; pyproject pins ``google-antigravity>=0.1.21,<0.2``). Everything
here degrades to "no process known" when that path is absent (fakes, SDK changes).
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import threading
import time
from typing import Any

import anyio

from . import spawn
from .errors import UpstreamError
from .limits import LiveAgentRegistry
from .spawn import OpenTracker, kill_tree

logger = logging.getLogger("agy_proxy.lifecycle")

spawn.install()  # off-loop, tracked harness spawn (no-op + pid-snapshot fallback if SDK differs)


def harness_process(agent: Any) -> subprocess.Popen[bytes] | None:
    """The agent's localharness ``Popen``, or None if the agent is not (fully) started."""
    try:
        return agent.conversation.connection._process
    except Exception:  # noqa: BLE001 - not started, or the private path changed
        return None


def harness_pid(agent: Any) -> int | None:
    proc = harness_process(agent)
    return proc.pid if proc is not None else None


def agent_usable(agent: Any) -> bool:
    """True when a cached agent can take another turn.

    False once the harness died, or when a previous turn is still marked running
    (e.g. the client vanished mid-turn and the halt did not complete).
    """
    proc = harness_process(agent)
    if proc is not None and proc.poll() is not None:
        return False
    try:
        return bool(agent.conversation.is_idle)
    except Exception:  # noqa: BLE001 - fakes / not started: nothing to say
        return True


class KillWatchdog:
    """Thread that SIGTERMs, then SIGKILLs, a harness that outlives its teardown budget.

    Only used when the SDK's blocking ``Popen.wait`` could not be neutralised (see
    ``_defer_wait``): then a real thread is the only thing that can still act.
    ``Popen`` methods are used (not raw pids) so a reaped process is never signalled.
    """

    def __init__(self, proc: subprocess.Popen[bytes] | None, grace: float, kill_after: float = 3.0):
        self._proc = proc
        self._grace = grace
        self._kill_after = kill_after
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        self.escalated = False

    def start(self) -> KillWatchdog:
        if self._proc is not None:
            self._thread = threading.Thread(target=self._run, name="agy-kill-watchdog", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._done.set()

    def _run(self) -> None:
        proc = self._proc
        assert proc is not None
        if self._done.wait(self._grace) or proc.poll() is not None:
            return
        self.escalated = True
        logger.warning("harness pid %s outlived teardown budget; sending SIGTERM", proc.pid)
        kill_tree(proc, 15)
        if self._done.wait(self._kill_after) or proc.poll() is not None:
            return
        logger.warning("harness pid %s ignored SIGTERM; sending SIGKILL", proc.pid)
        kill_tree(proc, 9)

    def __enter__(self) -> KillWatchdog:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


class _DeferredWait:
    """Stand-in for the harness ``Popen`` while the SDK's ``disconnect()`` runs.

    ``disconnect()`` ends in a synchronous ``Popen.wait(180)`` on the event-loop thread. With
    this proxy installed ``wait`` returns at once; the harness's exit is awaited by us, off the
    loop and with escalation (see ``_stop_process``). Everything else is the real ``Popen``.
    """

    def __init__(self, real: subprocess.Popen[bytes]):
        self._real = real

    def wait(self, timeout: float | None = None) -> int:
        return self._real.poll() or 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def _defer_wait(agent: Any) -> tuple[Any, subprocess.Popen[bytes]] | None:
    try:
        connection = agent.conversation.connection
        real = connection._process
        if real is None:
            return None
        connection._process = _DeferredWait(real)
        return connection, real
    except Exception:  # noqa: BLE001 - not started / private path changed
        return None


def child_pids() -> set[int]:
    """Direct children of this process (Linux /proc scan; empty set elsewhere)."""
    me = os.getpid()
    out: set[int] = set()
    try:
        names = os.listdir("/proc")
    except OSError:
        return out
    for name in names:
        if not name.isdecimal():
            continue
        try:
            with open(f"/proc/{name}/stat") as f:
                ppid = int(f.read().rsplit(") ", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        if ppid == me:
            out.add(int(name))
    return out


def _kill_pid(pid: int) -> None:
    try:
        if os.getpgid(pid) == pid and pid != os.getpgrp():
            os.killpg(pid, 9)
        else:
            os.kill(pid, 9)
    except OSError:
        pass


async def _wait_dead(proc: subprocess.Popen[bytes], timeout: float) -> bool:
    """Poll (never block the loop) until the process exits or ``timeout`` passes."""
    deadline = time.monotonic() + timeout
    while proc.poll() is None:
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(0.02)
    return True


async def _stop_process(proc: subprocess.Popen[bytes], grace: float, kill_after: float) -> bool:
    """Let the harness exit on its own for ``grace``, then SIGTERM, then SIGKILL (whole group).

    Returns True once the process is confirmed dead (reaped).
    """
    if await _wait_dead(proc, grace):
        return True
    logger.warning("harness pid %s outlived its teardown budget; sending SIGTERM", proc.pid)
    kill_tree(proc, 15)
    if await _wait_dead(proc, kill_after):
        return True
    logger.warning("harness pid %s ignored SIGTERM; sending SIGKILL", proc.pid)
    kill_tree(proc, 9)
    return await _wait_dead(proc, 5.0)


async def await_tolerant(task: asyncio.Future[Any]) -> Any:
    """Await ``task`` to completion no matter how often the caller is cancelled.

    The task keeps running (it is shielded); cancellations are remembered and re-raised as one
    ``CancelledError`` after it finished, so the caller still unwinds but never abandons the work.
    """
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            cancelled = True
    if cancelled:
        raise asyncio.CancelledError
    return result


class AgentLifecycle:
    """Open/halt/close an Agent without ever leaking its harness process or scratch dir.

    * ``open`` runs ``Agent.__aenter__`` as its own task with a spawn deadline. Every harness
      process it starts is tracked (``spawn.OpenTracker``), so a cancelled, timed-out or failed
      open kills the child instead of orphaning it.
    * ``close`` runs the teardown as a separate task awaited through ``asyncio.shield`` in a loop
      that survives repeated cancellation. The registry entry and the scratch tree are removed
      only once every process is confirmed dead; the SDK's blocking ``Popen.wait`` is neutralised
      and the real wait is a non-blocking poll with SIGTERM/SIGKILL escalation.
    """

    def __init__(
        self,
        teardown_timeout: float = 10.0,
        registry: LiveAgentRegistry | None = None,
        kill_after: float = 3.0,
        spawn_timeout: float = 30.0,
    ):
        self.teardown_timeout = teardown_timeout
        self.kill_after = kill_after
        self.spawn_timeout = spawn_timeout
        self.live = registry if registry is not None else LiveAgentRegistry()
        self._closing: dict[int, asyncio.Future[None]] = {}
        self._open_lock = asyncio.Lock()  # only used by the pid-snapshot fallback

    # -- open ---------------------------------------------------------------------------------

    async def open(self, agent: Any) -> None:
        self.live.add(agent)
        tracker = OpenTracker(self.spawn_timeout)
        try:
            agent._agy_tracker = tracker
        except AttributeError:  # slotted fakes
            pass
        if spawn.installed():
            await self._open(agent, tracker, None)
            return
        async with self._open_lock:  # fallback: attribute children by snapshot diff
            await self._open(agent, tracker, child_pids())

    async def _open(self, agent: Any, tracker: OpenTracker, before: set[int] | None) -> None:
        async def enter() -> None:
            spawn.set_tracker(tracker)  # this task has its own context copy
            try:
                await asyncio.wait_for(agent.__aenter__(), self.spawn_timeout)
            except TimeoutError as exc:
                raise UpstreamError(
                    f"harness did not start within {self.spawn_timeout:g}s"
                ) from exc

        task = asyncio.ensure_future(enter())
        try:
            await asyncio.shield(task)
        except BaseException:
            await await_tolerant(
                asyncio.ensure_future(self._abandon_open(agent, task, tracker, before))
            )
            raise

    async def _abandon_open(
        self,
        agent: Any,
        task: asyncio.Future[Any],
        tracker: OpenTracker,
        before: set[int] | None,
    ) -> None:
        if not task.done():
            task.cancel()
        tracker.abandon()  # SIGKILL every child this open started (and any that still appears)
        await asyncio.wait({task}, timeout=self.teardown_timeout)
        if not task.done():
            logger.error("agent __aenter__ ignored cancellation")
        elif not task.cancelled():
            task.exception()  # mark retrieved
        if before is not None:  # fallback: whatever new child nobody else owns is ours
            known = {p.pid for a in self.live.snapshot() if (p := harness_process(a))}
            for pid in child_pids() - before - known:
                _kill_pid(pid)
        await self._teardown(agent)

    # -- halt / close -------------------------------------------------------------------------

    async def halt_turn(self, agent: Any) -> None:
        """Ask the harness to stop an unfinished turn (no-op when idle / not started)."""
        await await_tolerant(asyncio.ensure_future(self._halt(agent)))

    async def _halt(self, agent: Any) -> None:
        try:
            conversation = agent.conversation
            if conversation.is_idle:
                return
            await asyncio.wait_for(conversation.cancel(), 5)
        except Exception as exc:  # noqa: BLE001 - includes wait_for's TimeoutError
            logger.debug("halt_turn: %r", exc)

    async def close(self, agent: Any) -> None:
        """Idempotent teardown that survives cancellation of the caller.

        Concurrent callers share one teardown task. Always removes the process and the scratch
        tree; the registry entry only goes once the process is confirmed dead.
        """
        task = self._closing.get(id(agent))
        if task is None:
            if getattr(agent, "_agy_closed", False):
                return
            try:
                agent._agy_closed = True
            except AttributeError:  # slotted fakes
                pass
            task = asyncio.ensure_future(self._teardown(agent))
            key = id(agent)
            self._closing[key] = task
            task.add_done_callback(lambda _t, k=key: self._closing.pop(k, None))
        await await_tolerant(task)

    async def _teardown(self, agent: Any) -> None:
        tracker: OpenTracker | None = getattr(agent, "_agy_tracker", None)
        scratch = getattr(agent, "scratch", None)
        procs: list[subprocess.Popen[bytes]] = []
        main = harness_process(agent)
        if main is not None:
            procs.append(main)
        for p in tracker.procs if tracker is not None else []:
            if all(p is not q for q in procs):
                procs.append(p)
        dead = False
        try:
            await self._halt(agent)
            await self._disconnect(agent, main)
            results = await asyncio.gather(
                *(_stop_process(p, self.teardown_timeout, self.kill_after) for p in procs)
            )
            dead = all(results)
        finally:
            if not dead:  # cancelled or failed above: make sure nothing survives us
                for p in procs:
                    kill_tree(p)
                    try:
                        p.wait(timeout=1)
                    except (subprocess.TimeoutExpired, OSError):
                        pass
                dead = all(p.poll() is not None for p in procs)
            for p in procs:
                _close_pipes(p)
            if dead:
                self.live.discard(agent)
                if scratch is not None:
                    scratch.cleanup()
            else:
                logger.error(
                    "harness pid(s) %s could not be killed; keeping the agent registered",
                    [p.pid for p in procs if p.poll() is None],
                )
                try:
                    agent._agy_closed = False  # let a later close/reap retry
                except AttributeError:
                    pass

    async def _disconnect(self, agent: Any, main: subprocess.Popen[bytes] | None) -> None:
        """The SDK's ``__aexit__`` with its loop-blocking process wait neutralised."""
        deferred = _defer_wait(agent)
        watchdog = None if deferred else KillWatchdog(main, self.teardown_timeout, self.kill_after)
        if watchdog is not None:
            watchdog.start()
        try:
            await asyncio.wait_for(agent.__aexit__(None, None, None), self.teardown_timeout)
        except TimeoutError:
            logger.warning("agent close exceeded %ss", self.teardown_timeout)
        except Exception:  # noqa: BLE001 - never let teardown raise
            logger.exception("agent close failed")
        finally:
            if watchdog is not None:
                watchdog.stop()
            if deferred is not None:
                deferred[0]._process = deferred[1]

    async def reap(self) -> int:
        """Close every agent still registered (lifespan shutdown)."""
        with anyio.CancelScope(shield=True):
            return await self.live.reap(self.close)


def _close_pipes(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None or proc.poll() is None:
        return
    for stream in (proc.stdin, proc.stdout, proc.stderr):
        try:
            if stream is not None:
                stream.close()
        except Exception:  # noqa: BLE001, S110
            pass
