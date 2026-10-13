"""Proxy error types and the single place that renders them as HTTP responses.

Every failure the client can see is a :class:`ProxyError` subclass with a stable HTTP
status and machine-readable kind. :func:`error_response` renders one in the wire format
of whichever protocol the request spoke:

* Anthropic (``/v1/messages*`` and ``/v1/models*`` when ``anthropic-version`` or
  ``x-api-key`` is present): ``{"type": "error", "error": {"type", "message"}}``
* OpenAI (everything else): ``{"error": {"message", "type", "param", "code"}}``

Python exception class names never appear as the ``type``. Middleware, exception
handlers and the streaming adapters all use the helpers here.

Mapping table (status / Anthropic type / OpenAI type / OpenAI code):

=================  ======  =====================  ===================  ===================
class              status  anthropic type         openai type          openai code
=================  ======  =====================  ===================  ===================
InvalidRequest     400     invalid_request_error  invalid_request_...  None
UnsupportedMedia.. 415     invalid_request_error  invalid_request_...  None
AuthError          401     authentication_error   invalid_request_...  invalid_api_key
ForbiddenError     403     permission_error       permission_error     None
UnknownModelError  404     not_found_error        invalid_request_...  model_not_found
PayloadTooLarge    413     request_too_large      invalid_request_...  request_too_large
RateLimited        429     rate_limit_error       rate_limit_error     rate_limit_exceeded
OverloadedError    429/529 overloaded_error       server_error         None
InternalError      500     api_error              server_error         None
UpstreamError      502     api_error              server_error         None
GatewayTimeout     504     api_error              server_error         None
=================  ======  =====================  ===================  ===================
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("agy_proxy.errors")


class ProxyError(Exception):
    """Base class: an error with an HTTP status and a stable machine-readable kind."""

    status: int = 500
    kind: str = "api_error"  # Anthropic error type
    oai_type: str = "server_error"
    oai_code: str | None = None

    def __init__(
        self,
        message: str,
        param: str | None = None,
        *,
        status: int | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.param = param
        if status is not None:
            self.status = status


class InvalidRequest(ProxyError):
    status = 400
    kind = "invalid_request_error"
    oai_type = "invalid_request_error"


class UnsupportedMediaType(InvalidRequest):
    status = 415


class UnknownModelError(ProxyError):
    status = 404
    kind = "not_found_error"
    oai_type = "invalid_request_error"
    oai_code = "model_not_found"


class AuthError(ProxyError):
    status = 401
    kind = "authentication_error"
    oai_type = "invalid_request_error"
    oai_code = "invalid_api_key"


class ForbiddenError(ProxyError):
    status = 403
    kind = "permission_error"
    oai_type = "permission_error"


class PayloadTooLarge(ProxyError):
    status = 413
    kind = "request_too_large"
    oai_type = "invalid_request_error"
    oai_code = "request_too_large"


class RateLimited(ProxyError):
    status = 429
    kind = "rate_limit_error"
    oai_type = "rate_limit_error"
    oai_code = "rate_limit_exceeded"


class OverloadedError(ProxyError):
    """429 when the proxy itself is full, 529 when the upstream is overloaded."""

    status = 429
    kind = "overloaded_error"
    oai_type = "server_error"

    def __init__(self, message: str, param: str | None = None, *, upstream: bool = False):
        super().__init__(message, param, status=529 if upstream else 429)


class InternalError(ProxyError):
    status = 500


class UpstreamError(ProxyError):
    status = 502


class GatewayTimeout(ProxyError):
    status = 504


# Name used by the design document; kept as an alias (builtin TimeoutError is unrelated).
TimeoutError = GatewayTimeout  # noqa: A001


def to_proxy_error(exc: BaseException) -> ProxyError:
    """Convert any exception into a client-safe :class:`ProxyError`."""
    if isinstance(exc, ProxyError):
        return exc
    logger.warning("upstream failure: %r", exc, exc_info=exc)
    text = " ".join(str(exc).split())[:300]
    return UpstreamError(f"agent failure: {text}" if text else "agent failure")


# -- rendering -------------------------------------------------------------


def wants_anthropic(path: str, request_headers: Mapping[str, str] | None = None) -> bool:
    """True when the request speaks the Anthropic protocol (by path, then headers)."""
    if path.startswith("/v1/messages"):
        return True
    if path.startswith("/v1/models") and request_headers:
        return "anthropic-version" in request_headers or "x-api-key" in request_headers
    return False


def error_payload(err: ProxyError, anthropic: bool) -> dict[str, Any]:
    if anthropic:
        return {"type": "error", "error": {"type": err.kind, "message": err.message}}
    return {
        "error": {
            "message": err.message,
            "type": err.oai_type,
            "param": err.param,
            "code": err.oai_code,
        }
    }


def error_response(
    err: ProxyError,
    path: str = "",
    request_headers: Mapping[str, str] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Render ``err`` as a JSON response in the protocol implied by ``path``/headers."""
    body = error_payload(err, wants_anthropic(path, request_headers))
    return JSONResponse(status_code=err.status, content=body, headers=headers)


def sse_error_frame(err: ProxyError, anthropic: bool) -> str:
    """Mid-stream error frame (Anthropic ``event: error``; OpenAI ``data: {error}``)."""
    payload = json.dumps(error_payload(err, anthropic), ensure_ascii=False)
    return f"event: error\ndata: {payload}\n\n" if anthropic else f"data: {payload}\n\n"


# -- handlers --------------------------------------------------------------

_HTTP_KINDS: dict[int, type[ProxyError]] = {
    400: InvalidRequest,
    401: AuthError,
    403: ForbiddenError,
    404: UnknownModelError,
    413: PayloadTooLarge,
    429: RateLimited,
}


def _lower_headers(request: Request) -> dict[str, str]:
    return {k.lower(): v for k, v in request.headers.items()}


def _validation_message(exc: RequestValidationError) -> tuple[str, str | None]:
    errors = exc.errors()
    if not errors:
        return "invalid request body", None
    first = errors[0]
    loc = [str(p) for p in first.get("loc", ()) if p != "body"]
    param = ".".join(loc) or None
    msg = str(first.get("msg", "invalid value"))
    text = f"{param}: {msg}" if param else msg
    return text, param


def install_error_handlers(app: FastAPI) -> None:
    """Route every exception (including unknown routes and validation) to envelopes."""

    def respond(request: Request, err: ProxyError) -> JSONResponse:
        headers = {"www-authenticate": "Bearer"} if isinstance(err, AuthError) else None
        return error_response(err, request.url.path, _lower_headers(request), headers)

    @app.exception_handler(ProxyError)
    async def _proxy_error(request: Request, exc: ProxyError):
        return respond(request, exc)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        message, param = _validation_message(exc)
        return respond(request, InvalidRequest(message, param))

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        cls = _HTTP_KINDS.get(exc.status_code)
        message = {404: "not found", 405: "method not allowed"}.get(
            exc.status_code, str(exc.detail) or "request failed"
        )
        if exc.status_code in (404, 405):
            err: ProxyError = InvalidRequest(message, status=exc.status_code)
            err.kind = "not_found_error" if exc.status_code == 404 else "invalid_request_error"
            err.oai_code = None
        elif cls is not None:
            err = cls(message)
        else:
            err = InvalidRequest(message, status=exc.status_code)
        return respond(request, err)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        logger.error("unhandled error on %s", request.url.path, exc_info=exc)
        return respond(request, InternalError("internal server error"))
