"""Security layer: token auth, Host/Origin checks, bind guard, env scrubbing.

agy-proxy drives an agent that acts with the proxy user's privileges, so every
request must be authenticated and must not originate from a browser page
(DNS rebinding / CSRF). All checks are pure-ASGI middleware so they also protect
streaming responses and run before any body is parsed.
"""

from __future__ import annotations

import hmac
import json
import os
import secrets
import sys
from collections.abc import Iterable, Mapping, MutableMapping

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .config import Settings, is_loopback_host
from .errors import (
    AuthError,
    ForbiddenError,
    ProxyError,
    UnsupportedMediaType,
    error_response,
)
from .redact import new_request_id, request_id_var

# The only unauthenticated path; it reveals nothing but liveness.
OPEN_PATHS = frozenset({"/healthz"})
_BODY_METHODS = frozenset({"POST", "PUT", "PATCH"})

# Variables the harness (and any shell it runs) may inherit. Everything else is removed
# from the proxy process once Settings has captured the configuration.
CHILD_ENV_ALLOWLIST = frozenset(
    {
        "PATH",
        "HOME",
        "LANG",
        "TMPDIR",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
        "GOOGLE_GENAI_USE_VERTEXAI",
    }
)
CHILD_ENV_PREFIXES = ("LC_",)
# Needed so the harness can reach the network through an egress proxy / custom CA.
# (Proxy URLs may embed credentials: drop them with AGY_ENV_PASSTHROUGH unset and
# configure egress another way if that matters.)
NETWORK_ENV = frozenset(
    {
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "ALL_PROXY",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "NODE_EXTRA_CA_CERTS",
    }
)
# Interpreter / process plumbing that is harmless and expected to be present.
_RUNTIME_ENV = frozenset(
    {"USER", "LOGNAME", "SHELL", "TERM", "TZ", "VIRTUAL_ENV", "LD_LIBRARY_PATH"}
)


def env_allowed(name: str, extra: Iterable[str] = ()) -> bool:
    upper = name.upper()
    return (
        upper in CHILD_ENV_ALLOWLIST
        or upper in NETWORK_ENV
        or upper in _RUNTIME_ENV
        or upper.startswith(CHILD_ENV_PREFIXES)
        or upper.startswith("PYTHON")
        or upper in {e.upper() for e in extra}
    )


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


def scrub_process_env(
    environ: MutableMapping[str, str] | None = None, extra: Iterable[str] = ()
) -> list[str]:
    """Remove everything but an allowlist from the process environment; returns the names removed.

    The SDK merges ``os.environ`` into the harness environment, so the only way to keep
    the API key, the proxy token and unrelated secrets (cloud credentials, database URLs,
    ``SSH_AUTH_SOCK``...) away from the agent and any shell it spawns is to drop them from
    this process once Settings has captured what it needs. ``extra`` adds operator-chosen
    names (``AGY_ENV_PASSTHROUGH``).

    Caveat: this edits the live environment only. ``/proc/<pid>/environ`` of this process
    still shows the block it was started with, readable by the same user (so by the agent's
    shell). ``__main__`` therefore re-execs with a clean environment; see :func:`reexec_clean`.
    """
    environ = os.environ if environ is None else environ
    removed: list[str] = []
    for name in list(environ):
        if not env_allowed(name, extra):
            environ.pop(name, None)
            removed.append(name)
    return removed


# -- re-exec with a clean environment ----------------------------------------------------

SECRET_VARS = ("AGY_PROXY_TOKEN", "AGY_API_KEY", "GEMINI_API_KEY")
_CLEAN_MARK = "AGY_ENV_CLEAN"
_FD_VAR = "AGY_SECRETS_FD"


def split_secrets(environ: Mapping[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """(clean environment for the new process, secrets to hand over out-of-band)."""
    secret_vals = {k: environ[k] for k in SECRET_VARS if environ.get(k)}
    extra = [e.strip() for e in environ.get("AGY_ENV_PASSTHROUGH", "").split(",") if e.strip()]
    clean = {
        k: v
        for k, v in environ.items()
        if k not in SECRET_VARS and (env_allowed(k, extra) or k.upper().startswith("AGY_"))
    }
    return clean, secret_vals


def reexec_clean(
    argv: list[str] | None = None, environ: Mapping[str, str] | None = None
) -> None:
    """Replace this process with a copy whose environment holds no secrets.

    The secrets travel through an inherited pipe (never on disk, argv or environ) and are
    read once by :func:`adopt_inherited_secrets`. Only returns when re-exec is disabled or
    already done (``AGY_ENV_CLEAN`` marker / ``AGY_NO_REEXEC=1``).
    """
    environ = os.environ if environ is None else environ
    if environ.get(_CLEAN_MARK) or environ.get("AGY_NO_REEXEC", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        return
    clean, secret_vals = split_secrets(environ)
    clean[_CLEAN_MARK] = "1"
    if secret_vals:
        payload = json.dumps(secret_vals).encode()
        r, w = os.pipe()
        if len(payload) > 32_000:  # must fit the pipe buffer: we write before exec
            raise SystemExit("refusing to re-exec: secrets too large")
        os.write(w, payload)
        os.close(w)
        os.set_inheritable(r, True)
        clean[_FD_VAR] = str(r)
    args = argv or [sys.executable, "-m", "agy_proxy"]
    os.execve(args[0], args, clean)


def adopt_inherited_secrets(environ: MutableMapping[str, str] | None = None) -> bool:
    """Read the secrets pipe left by :func:`reexec_clean` into ``environ`` (then close it)."""
    environ = os.environ if environ is None else environ
    raw_fd = environ.pop(_FD_VAR, None)
    if not raw_fd or not raw_fd.isdigit():
        return False
    fd = int(raw_fd)
    try:
        with os.fdopen(fd, "rb") as fh:
            data = json.loads(fh.read() or b"{}")
    except (OSError, ValueError):
        return False
    for k, v in data.items():
        if k in SECRET_VARS and isinstance(v, str):
            environ[k] = v
    return True


def _host_only(value: str) -> str | None:
    """Host without its port; ``None`` when the value is malformed (non-numeric port...).

    IPv6 literals keep their brackets. A port, when present, must be 1-5 digits.
    """
    v = value.strip().lower()
    if v.startswith("["):
        end = v.find("]")
        if end == -1:
            return None
        rest = v[end + 1 :]
        if rest and not (rest.startswith(":") and _valid_port(rest[1:])):
            return None
        return v[: end + 1]
    if v.count(":") == 1:
        host, port = v.rsplit(":", 1)
        return host if _valid_port(port) else None
    return v


def _valid_port(port: str) -> bool:
    return port.isascii() and port.isdigit() and 0 < int(port) <= 65535


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
    request_headers = _headers(scope)
    await error_response(err, scope.get("path", ""), request_headers, headers)(
        scope, receive, send
    )


class HostOriginMiddleware:
    """Reject foreign Host headers, any Origin header, and non-JSON request bodies."""

    def __init__(self, app: ASGIApp, allowed_hosts: Iterable[str]):
        self.app = app
        self.allowed = frozenset(h for h in (_host_only(x) for x in allowed_hosts) if h)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        h = _headers(scope)
        host = _host_only(h.get("host", ""))
        if host is None or host not in self.allowed:
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


class RequestIdMiddleware:
    """Outermost layer: a fresh correlation id on every response (``x-request-id``).

    Client-supplied ids are never trusted or echoed (log injection). The id is also
    placed in a context variable so error bodies, SSE error frames and server logs
    carry the same value.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        rid = new_request_id()
        token = request_id_var.set(rid)

        async def send_w(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = [
                    (k, v)
                    for k, v in message.get("headers", [])
                    if k.lower() not in (b"x-request-id", b"request-id")
                ]
                headers += [(b"x-request-id", rid.encode()), (b"request-id", rid.encode())]
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_w)
        finally:
            request_id_var.reset(token)
