"""Resource limits: request body cap, concurrency admission, turn timeout, live-agent registry.

All middleware here is pure ASGI so it also covers streaming responses. Registration order
(innermost last) is Admission < BodyLimit < token auth < Host/Origin, so unauthenticated
traffic never consumes a concurrency slot or is even read.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .errors import GatewayTimeout, PayloadTooLarge, ProxyError, RateLimited
from .errors import error_response as render_error

logger = logging.getLogger("agy_proxy.limits")



def error_response(err: ProxyError, scope: Scope, headers: dict[str, str] | None = None):
    """Path-aware envelope (Anthropic vs OpenAI) for an error raised in ASGI middleware."""
    request_headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]}
    return render_error(err, scope.get("path", ""), request_headers, headers)


_BODY_METHODS = frozenset({"POST", "PUT", "PATCH"})
# Endpoints that spawn an agent; everything else (models, health, token counting) is cheap.
COMPLETION_PATHS = frozenset({"/v1/chat/completions", "/v1/messages"})


class LiveAgentRegistry:
    """Every agent currently open (stateless or session), so shutdown can reap leftovers."""

    def __init__(self) -> None:
        self._agents: dict[int, Any] = {}

    def add(self, agent: Any) -> None:
        self._agents[id(agent)] = agent

    def discard(self, agent: Any) -> None:
        self._agents.pop(id(agent), None)

    def __len__(self) -> int:
        return len(self._agents)

    def snapshot(self) -> list[Any]:
        return list(self._agents.values())

    async def reap(self, close: Callable[[Any], Awaitable[None]]) -> int:
        """Close every still-registered agent with ``close``; returns how many."""
        agents = self.snapshot()
        for agent in agents:
            try:
                await close(agent)
            except Exception:  # noqa: BLE001 - best effort at shutdown
                logger.exception("failed to reap agent")
        return len(agents)


class Admission:
    """Bounded concurrency: a full house is refused immediately (429), never queued."""

    def __init__(self, max_concurrency: int, turn_timeout: float):
        self.max_concurrency = max_concurrency
        self.turn_timeout = turn_timeout
        self.inflight = 0

    def try_enter(self) -> bool:
        if self.inflight >= self.max_concurrency:
            return False
        self.inflight += 1
        return True

    def leave(self) -> None:
        self.inflight = max(0, self.inflight - 1)


class AdmissionMiddleware:
    """Concurrency cap + per-request deadline (504) + cancel-on-client-disconnect.

    The app runs as a child task so that (a) the deadline can cancel it, and (b) a client
    that hangs up after sending its body (the non-streaming case, where nobody else is
    listening) stops the agent turn instead of burning the model until the deadline.
    Streaming responses also get Starlette's own disconnect handling; this is additive.
    A deadline that fires after the response started can only truncate the stream.
    """

    def __init__(self, app: ASGIApp, admission: Admission, paths: Iterable[str] = COMPLETION_PATHS):
        self.app = app
        self.admission = admission
        self.paths = frozenset(paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or scope["path"] not in self.paths
        ):
            await self.app(scope, receive, send)
            return
        if not self.admission.try_enter():
            err = RateLimited(
                f"server is at capacity ({self.admission.max_concurrency} concurrent requests); "
                "retry shortly"
            )
            await error_response(err, scope, {"retry-after": "1"})(scope, receive, send)
            return
        try:
            await self._run(scope, receive, send)
        finally:
            self.admission.leave()

    async def _run(self, scope: Scope, receive: Receive, send: Send) -> None:
        started = False
        body_done = asyncio.Event()
        disconnected = False

        async def send_w(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        async def receive_w() -> Message:
            message = await receive()
            if message["type"] == "http.request" and not message.get("more_body", False):
                body_done.set()
            return message

        app_task = asyncio.ensure_future(self.app(scope, receive_w, send_w))

        async def watch_disconnect() -> None:
            nonlocal disconnected
            await body_done.wait()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    disconnected = True
                    app_task.cancel()
                    return

        watcher = asyncio.ensure_future(watch_disconnect())
        me = asyncio.current_task()
        try:
            async with asyncio.timeout(self.admission.turn_timeout):
                await app_task
        except TimeoutError:  # builtin: raised by asyncio.timeout
            logger.warning("request exceeded the %ss turn timeout", self.admission.turn_timeout)
            if not started:
                err = GatewayTimeout(f"request exceeded {self.admission.turn_timeout}s")
                await error_response(err, scope)(scope, receive, send)
        except asyncio.CancelledError:
            if not (disconnected and me is not None and me.cancelling() == 0):
                raise  # a real cancellation of this request: propagate
            logger.info("client disconnected; agent turn cancelled")
        finally:
            watcher.cancel()
            if not app_task.done():
                app_task.cancel()
            await asyncio.gather(watcher, app_task, return_exceptions=True)


class BodyLimitMiddleware:
    """413 for request bodies over ``max_bytes`` -- also when there is no Content-Length.

    A declared oversize length is refused before reading anything. Otherwise bytes are
    counted as the app consumes them; once over the cap the app sees a disconnect and
    whatever response it then produces (FastAPI turns that into a 400) is replaced by 413.
    """

    def __init__(self, app: ASGIApp, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    def _too_large(self) -> PayloadTooLarge:
        return PayloadTooLarge(f"request body exceeds {self.max_bytes} bytes")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] not in _BODY_METHODS:
            await self.app(scope, receive, send)
            return
        declared = dict(scope.get("headers", [])).get(b"content-length")
        if declared is not None and declared.strip().isdigit() and int(declared) > self.max_bytes:
            await error_response(self._too_large(), scope)(scope, receive, send)
            return

        seen = 0
        exceeded = False
        responded = False

        async def reject() -> None:
            nonlocal responded
            if not responded:
                responded = True
                await error_response(self._too_large(), scope)(scope, receive, send)

        async def receive_w() -> Message:
            nonlocal seen, exceeded
            if exceeded:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.max_bytes:
                    exceeded = True
                    return {"type": "http.disconnect"}
            return message

        async def send_w(message: Message) -> None:
            nonlocal responded
            if exceeded:
                await reject()  # replaces whatever the app tried to say; later parts dropped
                return
            if message["type"] == "http.response.start":
                responded = True
            await send(message)

        await self.app(scope, receive_w, send_w)
        if exceeded:
            await reject()
