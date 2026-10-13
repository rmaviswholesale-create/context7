"""Stand-in for the Go ``localharness`` binary: speaks just enough of the SDK handshake.

Run through ``ANTIGRAVITY_HARNESS_PATH``. It never calls a model and never runs tools; a user
turn is simply never answered, so a turn stays "running" until the client cancels it.

``FAKE_HARNESS_MODE``:
  ok        exits when its stdin closes (what the real harness does)
  term      ignores stdin EOF (a hung harness) but dies on SIGTERM
  stubborn  ignores stdin EOF AND SIGTERM; only SIGKILL stops it
  noinit    completes the stdio handshake but never answers InitializeConversation
  noinit_stubborn  noinit + ignores SIGTERM
  garbage   writes a corrupt OutputConfig on stdout, then hangs
  silent    reads its InputConfig and then says nothing at all (never writes OutputConfig)
  hangws    answers init, then blocks its websocket server (ws close handshake never completes)
``FAKE_DELAY``: seconds to sleep before writing OutputConfig (a slow harness start).
``FAKE_EXIT_DELAY``: seconds to keep running after stdin EOF before exiting (a slow harness stop).
``FAKE_HARNESS_PIDFILE``: where to write this process's pid.
"""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import struct
import sys
import threading
import time

import websockets
from google.antigravity.proto import localharness_pb2 as pb
from google.protobuf import json_format

MODE = os.environ.get("FAKE_HARNESS_MODE", "ok")
DELAY = float(os.environ.get("FAKE_DELAY", "0"))
EXIT_DELAY = float(os.environ.get("FAKE_EXIT_DELAY", "0"))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _watch_stdin() -> None:
    sys.stdin.buffer.read()  # blocks until EOF
    if MODE in ("ok", "hangws"):
        time.sleep(EXIT_DELAY)
        os._exit(0)
    while True:  # hung harness: ignore the EOF forever
        threading.Event().wait(3600)


async def _handler(ws) -> None:
    initialized = False
    async for _raw in ws:
        if not initialized:  # the first frame is InitializeConversationEvent
            initialized = True
            if MODE.startswith("noinit"):
                continue
            resp = pb.OutputEvent(
                initialize_conversation_response=pb.InitializeConversationResponse()
            )
            await ws.send(json_format.MessageToJson(resp))
            if MODE == "hangws":
                time.sleep(3600)  # blocks this harness's loop: the ws close handshake hangs
        # anything else (user_input, halt_request, ...) is deliberately ignored


async def _serve(port: int) -> None:
    async with websockets.serve(_handler, "127.0.0.1", port):
        await asyncio.Future()


def main() -> None:
    n = struct.unpack("<I", sys.stdin.buffer.read(4))[0]
    sys.stdin.buffer.read(n)  # InputConfig (ignored)
    if MODE in ("stubborn", "noinit_stubborn"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    pidfile = os.environ.get("FAKE_HARNESS_PIDFILE")
    if pidfile:
        with open(pidfile, "w") as f:
            f.write(str(os.getpid()))
    if MODE == "silent":
        threading.Event().wait(3600)
    time.sleep(DELAY)
    if MODE == "garbage":
        sys.stdout.buffer.write(struct.pack("<I", 3) + b"\xff\xff\xff")
        sys.stdout.buffer.flush()
        threading.Event().wait(3600)
    port = _free_port()
    out = pb.OutputConfig(port=port, api_key="fake").SerializeToString()
    sys.stdout.buffer.write(struct.pack("<I", len(out)) + out)
    sys.stdout.buffer.flush()
    threading.Thread(target=_watch_stdin, daemon=True).start()
    asyncio.run(_serve(port))


if __name__ == "__main__":
    main()
