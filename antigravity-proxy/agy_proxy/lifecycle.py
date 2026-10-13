"""Harness process helpers: the only place that touches SDK-private connection attributes.

``Agent.__aexit__`` -> ``LocalConnection.disconnect()`` is not safe to rely on alone:

* it ends with a *synchronous* ``Popen.wait(timeout=180)`` that blocks the whole event
  loop, so an ``asyncio.wait_for`` can never fire while a hung harness is being waited
  for -- a plain thread watchdog is the only thing that can still act;
* if the waiting is abandoned (cancellation, timeout) the harness child is orphaned.

The harness ``Popen`` lives at ``agent.conversation.connection._process`` (SDK 0.1.21,
see ``docs/SDK_NOTES.md``; pyproject pins ``google-antigravity>=0.1.21,<0.2``). Everything
here degrades to "no process known" when that path is absent (fakes, SDK changes).
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import threading
from typing import Any

import anyio

from .limits import LiveAgentRegistry

logger = logging.getLogger("agy_proxy.lifecycle")


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

    Runs in a real thread because the SDK's ``disconnect()`` blocks the event loop.
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
        try:
            proc.terminate()
            if self._done.wait(self._kill_after) or proc.poll() is not None:
                return
            logger.warning("harness pid %s ignored SIGTERM; sending SIGKILL", proc.pid)
            proc.kill()
        except OSError:  # already gone
            pass

    def __enter__(self) -> KillWatchdog:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


class AgentLifecycle:
    """Open/halt/close an Agent without ever leaking its harness process or scratch dir.

    * ``open`` runs ``Agent.__aenter__`` as a shielded task, so a cancelled request cannot
      abandon a half-started harness (the SDK does not clean up on ``CancelledError``).
    * ``close`` is bounded: a thread watchdog (SIGTERM, then SIGKILL) covers the SDK's
      loop-blocking ``Popen.wait(180)``, ``wait_for`` covers the async part, and the process
      and the Scratch tree are removed no matter what. Callers wrap it in a shielded cancel
      scope so it also runs while a request is being cancelled.
    """

    def __init__(
        self,
        teardown_timeout: float = 10.0,
        registry: LiveAgentRegistry | None = None,
        kill_after: float = 3.0,
    ):
        self.teardown_timeout = teardown_timeout
        self.kill_after = kill_after
        self.live = registry if registry is not None else LiveAgentRegistry()

    async def open(self, agent: Any) -> None:
        self.live.add(agent)
        task = asyncio.ensure_future(agent.__aenter__())
        try:
            await asyncio.shield(task)
        except BaseException:
            with anyio.CancelScope(shield=True):
                await self._abandon_open(agent, task)
            raise

    async def _abandon_open(self, agent: Any, task: asyncio.Future[Any]) -> None:
        if not task.done():
            done, _ = await asyncio.wait({task}, timeout=self.teardown_timeout)
            if not done:
                task.cancel()
                await asyncio.wait({task}, timeout=2.0)
        if task.done() and not task.cancelled():
            task.exception()  # mark retrieved
        await self.close(agent)

    async def halt_turn(self, agent: Any) -> None:
        """Ask the harness to stop an unfinished turn (no-op when idle / not started)."""
        try:
            conversation = agent.conversation
            if conversation.is_idle:
                return
            await asyncio.wait_for(conversation.cancel(), 5)
        except Exception as exc:  # noqa: BLE001 - includes wait_for's TimeoutError
            logger.debug("halt_turn: %r", exc)

    async def close(self, agent: Any) -> None:
        """Idempotent, bounded teardown; always removes the process and the scratch tree."""
        if getattr(agent, "_agy_closed", False):
            return
        try:
            agent._agy_closed = True
        except AttributeError:  # slotted fakes
            pass
        proc = harness_process(agent)
        scratch = getattr(agent, "scratch", None)
        try:
            await self.halt_turn(agent)
            with KillWatchdog(proc, self.teardown_timeout, self.kill_after):
                try:
                    await asyncio.wait_for(agent.__aexit__(None, None, None), self.teardown_timeout)
                except TimeoutError:
                    logger.warning("agent close exceeded %ss", self.teardown_timeout)
                except Exception:  # noqa: BLE001 - never let teardown raise
                    logger.exception("agent close failed")
            await terminate_process(proc)  # no-op when the harness already exited
            _close_pipes(proc)
        finally:
            self.live.discard(agent)
            if scratch is not None:
                scratch.cleanup()

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


async def terminate_process(proc: subprocess.Popen[bytes] | None, wait: float = 2.0) -> None:
    """SIGTERM, then SIGKILL, waiting off-loop; no-op for a missing/finished process."""
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        await asyncio.to_thread(proc.wait, wait)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            await asyncio.to_thread(proc.wait, wait)
        except subprocess.TimeoutExpired:
            logger.error("harness pid %s could not be killed", proc.pid)
    except OSError:
        pass
