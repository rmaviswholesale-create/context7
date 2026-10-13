"""Stand-in for the Go ``localharness`` binary: speaks just enough of the SDK handshake.

Run through ``ANTIGRAVITY_HARNESS_PATH``. It never calls a model and never runs tools; a user
turn is simply never answered, so a turn stays "running" until the client cancels it.

``FAKE_HARNESS_MODE``:
  ok        exits when its stdin closes (what the real harness does)
  term      ignores stdin EOF (a hung harness) but dies on SIGTERM
  stubborn  ignores stdin EOF AND SIGTERM; only SIGKILL stops it
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

import websockets
from google.antigravity.proto import localharness_pb2 as pb
from google.protobuf import json_format

MODE = os.environ.get("FAKE_HARNESS_MODE", "ok")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _watch_stdin() -> None:
    sys.stdin.buffer.read()  # blocks until EOF
    if MODE == "ok":
        os._exit(0)
    while True:  # hung harness: ignore the EOF forever
        threading.Event().wait(3600)


async def _handler(ws) -> None:
    initialized = False
    async for _raw in ws:
        if not initialized:  # the first frame is InitializeConversationEvent
            initialized = True
            resp = pb.OutputEvent(
                initialize_conversation_response=pb.InitializeConversationResponse()
            )
            await ws.send(json_format.MessageToJson(resp))
        # anything else (user_input, halt_request, ...) is deliberately ignored


async def _serve(port: int) -> None:
    async with websockets.serve(_handler, "127.0.0.1", port):
        await asyncio.Future()


def main() -> None:
    n = struct.unpack("<I", sys.stdin.buffer.read(4))[0]
    sys.stdin.buffer.read(n)  # InputConfig (ignored)
    if MODE == "stubborn":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    pidfile = os.environ.get("FAKE_HARNESS_PIDFILE")
    if pidfile:
        with open(pidfile, "w") as f:
            f.write(str(os.getpid()))
    port = _free_port()
    out = pb.OutputConfig(port=port, api_key="fake").SerializeToString()
    sys.stdout.buffer.write(struct.pack("<I", len(out)) + out)
    sys.stdout.buffer.flush()
    threading.Thread(target=_watch_stdin, daemon=True).start()
    asyncio.run(_serve(port))


if __name__ == "__main__":
    main()
