"""Security layer: token auth, Host/Origin checks, bind guard, env scrubbing.

agy-proxy drives an agent that acts with the proxy user's privileges, so every
request must be authenticated and must not originate from a browser page
(DNS rebinding / CSRF). All checks are pure-ASGI middleware so they also protect
streaming responses and run before any body is parsed.
"""

from __future__ import annotations

import hmac
import os
import re
import secrets
from collections.abc import Iterable, MutableMapping

from starlette.types import ASGIApp, Receive, Scope, Send

from .config import Settings, is_loopback_host
from .errors import (
    AuthError,
    ForbiddenError,
    ProxyError,
    UnsupportedMediaType,
    error_response,
)

# The only unauthenticated path; it reveals nothing but liveness.
OPEN_PATHS = frozenset({"/healthz"})
_BODY_METHODS = frozenset({"POST", "PUT", "PATCH"})

_SECRET_NAME = re.compile(r"(KEY|TOKEN|SECRET)", re.IGNORECASE)
_KEEP_EXACT = frozenset({"HOME", "PATH"})


def make_token() -> str:
    return secrets.token_urlsafe(32)


def check_bind(host: str, token: str | None) -> None:
    """Refuse to listen on a non-loopback address without an explicit token.

    A token alone does NOT make a non-loopback bind safe (it is sent in clear text
    and the agent has host privileges); this only guarantees the operator chose one.
    """
    if not is_loopback_host(host) and not token:
        raise SystemExit(
            f"refusing to bind to non-loopback host {host!r} without AGY_PROXY_TOKEN. "
            "Set AGY_PROXY_TOKEN explicitly (and put the proxy behind TLS and a firewall); "
            "tokens are only auto-generated for loopback binds."
        )


def scrub_process_env(environ: MutableMapping[str, str] | None = None) -> list[str]:
    """Remove secrets from the process environment; returns the names removed.

    The SDK merges ``os.environ`` into the harness environment, so the only way to keep
    the API key and proxy token away from the agent (and any shell it runs) is to
    drop them from this process once Settings has captured them. HOME, PATH and
    GOOGLE_* (ADC / Vertex configuration) are kept.
    """
    environ = os.environ if environ is None else environ
    removed: list[str] = []
    for name in list(environ):
        upper = name.upper()
        if upper in _KEEP_EXACT or upper.startswith("GOOGLE_"):
            continue
        if upper.startswith("AGY_") or upper == "GEMINI_API_KEY" or _SECRET_NAME.search(upper):
            environ.pop(name, None)
            removed.append(name)
    return removed


def _host_only(value: str) -> str:
    """Strip an optional port from a Host header value (IPv6 literals keep brackets)."""
    v = value.strip().lower()
    if v.startswith("["):
        end = v.find("]")
        return v[: end + 1] if end != -1 else v
    return v.rsplit(":", 1)[0] if v.count(":") == 1 else v


def _media_type(value: str) -> str:
    return value.split(";", 1)[0].strip().lower()


def _headers(scope: Scope) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in scope.get("headers", []):
        out.setdefault(k.decode("latin-1").lower(), v.decode("latin-1"))
    return out


async def _reject(scope: Scope, receive: Receive, send: Send, err: ProxyError) -> None:
    if scope["type"] == "websocket":
        await send({"type": "websocket.close", "code": 1008})
        return
    headers = {"www-authenticate": "Bearer"} if isinstance(err, AuthError) else None
    await error_response(err, headers)(scope, receive, send)


class HostOriginMiddleware:
    """Reject foreign Host headers, any Origin header, and non-JSON request bodies."""

    def __init__(self, app: ASGIApp, allowed_hosts: Iterable[str]):
        self.app = app
        self.allowed = frozenset(_host_only(h) for h in allowed_hosts)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        h = _headers(scope)
        if _host_only(h.get("host", "")) not in self.allowed:
            await _reject(scope, receive, send, ForbiddenError("host not allowed"))
            return
        # Browsers always attach Origin to cross-site POSTs; API clients never do.
        if "origin" in h:
            await _reject(scope, receive, send, ForbiddenError("browser requests not allowed"))
            return
        if scope["type"] == "http" and scope["method"] in _BODY_METHODS:
            if _media_type(h.get("content-type", "")) != "application/json":
                await _reject(
                    scope,
                    receive,
                    send,
                    UnsupportedMediaType("content-type must be application/json"),
                )
                return
        await self.app(scope, receive, send)


class TokenAuthMiddleware:
    """Require the proxy token as ``Authorization: Bearer`` or ``x-api-key``."""

    def __init__(self, app: ASGIApp, token: str):
        if not token:
            raise ValueError("TokenAuthMiddleware requires a non-empty token")
        self.app = app
        self._token = token.encode()

    def _matches(self, candidate: str | None) -> bool:
        if not candidate:
            return False
        return hmac.compare_digest(candidate.strip().encode(), self._token)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        if scope["type"] == "http" and scope["path"] in OPEN_PATHS:
            await self.app(scope, receive, send)
            return
        h = _headers(scope)
        bearer = None
        auth = h.get("authorization", "")
        if auth[:7].lower() == "bearer ":
            bearer = auth[7:]
        # Evaluate both so timing does not reveal which header was present.
        ok_bearer = self._matches(bearer)
        ok_key = self._matches(h.get("x-api-key"))
        if not (ok_bearer or ok_key):
            await _reject(scope, receive, send, AuthError("invalid or missing token"))
            return
        await self.app(scope, receive, send)


def ensure_token(settings: Settings) -> Settings:
    """Validate the bind and guarantee a token (generated on loopback only)."""
    from dataclasses import replace

    check_bind(settings.host, settings.proxy_token)
    if settings.proxy_token:
        return settings
    return replace(settings, proxy_token=make_token(), token_generated=True)
