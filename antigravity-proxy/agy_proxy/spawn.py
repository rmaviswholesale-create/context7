"""Off-loop, tracked, killable harness spawn (patches one private SDK method).

``LocalConnectionStrategy._spawn_harness_and_connect_ws`` (SDK 0.1.21) does, on the event-loop
thread: ``Popen`` -> write InputConfig -> *blocking* ``stdout.read`` of the OutputConfig. Two
problems follow, both fixed by replacing exactly that step:

* a slow or hung harness freezes the whole proxy (every other request included);
* the ``Popen`` is created inside the SDK call, so when the opening request is cancelled
  (or the SDK raises half-way) nobody holds a handle to the child and it is orphaned.

The replacement runs the blocking part in a worker thread, starts the child in its own session
(so ``killpg`` takes its whole process tree), registers it with the current ``OpenTracker``
the instant it exists, and arms a timer that kills it if the handshake does not finish within
the spawn timeout. Everything after the handshake (websocket connect, init exchange) is the
SDK's own async code, untouched.

If the SDK no longer looks like 0.1.x the patch is not installed and
``lifecycle.AgentLifecycle`` falls back to snapshotting child pids around ``__aenter__``.
"""

from __future__ import annotations

import asyncio
import contextvars
import importlib.metadata
import inspect
import logging
import os
import signal
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

from .errors import UpstreamError

logger = logging.getLogger("agy_proxy.spawn")

_TARGET = "_spawn_harness_and_connect_ws"
_EXECUTOR = ThreadPoolExecutor(max_workers=64, thread_name_prefix="agy-spawn")
_current: contextvars.ContextVar[OpenTracker | None] = contextvars.ContextVar(
    "agy_open_tracker", default=None
)
_original: Any = None


def kill_tree(proc: subprocess.Popen[bytes], sig: int = signal.SIGKILL) -> None:
    """Signal the harness, and its whole process group when we started it as a group leader.

    Safe against a reaped/never-ours process: the group is only signalled while the leader is
    still running, and never when it is the proxy's own group.
    """
    try:
        if proc.poll() is not None:
            return
        try:
            if os.getpgid(proc.pid) == proc.pid and proc.pid != os.getpgrp():
                os.killpg(proc.pid, sig)
                return
        except OSError:
            pass
        proc.send_signal(sig)
    except OSError:  # already gone
        pass


class OpenTracker:
    """Every harness process started on behalf of ONE ``Agent.__aenter__``.

    Thread-safe: processes are added from the spawn worker thread. Once ``abandon`` was called
    any process registered later is killed on arrival (the worker may still be mid-Popen).
    """

    def __init__(self, spawn_timeout: float | None = None):
        self.spawn_timeout = spawn_timeout
        self._lock = threading.Lock()
        self._procs: list[subprocess.Popen[bytes]] = []
        self.abandoned = False

    @property
    def procs(self) -> list[subprocess.Popen[bytes]]:
        with self._lock:
            return list(self._procs)

    def add(self, proc: subprocess.Popen[bytes]) -> None:
        with self._lock:
            self._procs.append(proc)
            doomed = self.abandoned
        if doomed:
            kill_tree(proc)

    def abandon(self) -> None:
        """Kill every tracked process now and any that shows up later."""
        with self._lock:
            self.abandoned = True
            procs = list(self._procs)
        for proc in procs:
            kill_tree(proc)


def set_tracker(tracker: OpenTracker | None) -> contextvars.Token[OpenTracker | None]:
    return _current.set(tracker)


def installed() -> bool:
    return _original is not None


def _spawn_blocking(
    strategy: Any, use_interactions_api: bool, tracker: OpenTracker | None
) -> tuple[subprocess.Popen[bytes], Any]:
    """The SDK's blocking half of the spawn, run in a worker thread."""
    import platform
    import struct

    from google.antigravity.connections.local import local_connection as lc

    pb = lc.localharness_pb2
    env_map = {str(k): str(v) for k, v in strategy._env.items()} if strategy._env else {}
    input_config = pb.InputConfig(
        storage_directory=strategy._save_dir or "",
        client_info=pb.ClientInfo(
            language="python",
            version=lc._get_sdk_version(),
            language_version=platform.python_version(),
            os=platform.system().lower(),
            os_version=platform.release(),
        ),
        env=env_map,
    )
    if use_interactions_api:
        input_config.use_interactions_api = True
    merged_env = {**os.environ, **env_map} if strategy._env is not None else None

    process = subprocess.Popen(  # noqa: S603 - the SDK's own binary path
        [strategy._binary_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=merged_env,
        start_new_session=True,
    )
    if tracker is not None:
        tracker.add(process)
    fired = threading.Event()
    timer: threading.Timer | None = None
    timeout = tracker.spawn_timeout if tracker is not None else None
    if timeout:

        def expire() -> None:
            fired.set()
            kill_tree(process)

        timer = threading.Timer(timeout, expire)
        timer.daemon = True
        timer.start()
    try:
        assert process.stdin and process.stdout and process.stderr
        serialized = input_config.SerializeToString()
        process.stdin.write(struct.pack("<I", len(serialized)) + serialized)
        process.stdin.flush()
        raw_len = process.stdout.read(4)
        if not raw_len:
            if fired.is_set():
                raise UpstreamError(f"harness did not start within {timeout}s")
            raise UpstreamError("harness exited before completing its handshake")
        (length,) = struct.unpack("<I", raw_len)
        output_config = pb.OutputConfig()
        output_config.ParseFromString(process.stdout.read(length))
        if fired.is_set():
            raise UpstreamError(f"harness did not start within {timeout}s")
        return process, output_config
    except BaseException:
        kill_tree(process)
        raise
    finally:
        if timer is not None:
            timer.cancel()


async def _spawn_off_loop(
    self: Any, *, use_interactions_api: bool = False
) -> tuple[subprocess.Popen[bytes], Any, str]:
    tracker = _current.get()
    loop = asyncio.get_running_loop()
    process, output_config = await loop.run_in_executor(
        _EXECUTOR, partial(_spawn_blocking, self, use_interactions_api, tracker)
    )
    ws, ws_url = await self._connect_websocket(
        output_config.port, output_config.api_key, process
    )
    return process, ws, ws_url


def install() -> bool:
    """Patch the SDK's spawn step; returns False (and leaves the SDK alone) if it differs."""
    global _original
    if _original is not None:
        return True
    try:
        version = importlib.metadata.version("google-antigravity")
        from google.antigravity.connections.local import local_connection as lc

        cls = lc.LocalConnectionStrategy
        orig = getattr(cls, _TARGET)
        params = inspect.signature(orig).parameters
        if not version.startswith("0.1.") or "use_interactions_api" not in params:
            raise RuntimeError(f"unexpected SDK shape (version {version})")
        if not hasattr(cls, "_connect_websocket"):
            raise RuntimeError("SDK has no _connect_websocket")
    except Exception as exc:  # noqa: BLE001 - any drift means: do not patch
        logger.warning("harness spawn patch not installed (%s); using pid-snapshot fallback", exc)
        return False
    _original = orig
    setattr(cls, _TARGET, _spawn_off_loop)
    return True


def uninstall() -> None:
    """Restore the SDK's own spawn (tests exercise the fallback path with this)."""
    global _original
    if _original is None:
        return
    from google.antigravity.connections.local import local_connection as lc

    setattr(lc.LocalConnectionStrategy, _TARGET, _original)
    _original = None
