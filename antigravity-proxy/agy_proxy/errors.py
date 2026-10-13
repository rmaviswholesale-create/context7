"""Proxy error types and the single place that renders them as HTTP responses.

P0 keeps the body a simple JSON object. P1 replaces :func:`error_response` with
protocol-specific (OpenAI / Anthropic) envelopes; every middleware and handler
goes through it so the swap is local.
"""

from __future__ import annotations

from fastapi.responses import JSONResponse


class ProxyError(Exception):
    """Base class: an error with an HTTP status and a stable machine-readable kind."""

    status: int = 500
    kind: str = "api_error"

    def __init__(self, message: str, param: str | None = None):
        super().__init__(message)
        self.message = message
        self.param = param


class InvalidRequest(ProxyError):
    status = 400
    kind = "invalid_request_error"


class UnknownModelError(ProxyError):
    status = 404
    kind = "not_found_error"


class AuthError(ProxyError):
    status = 401
    kind = "authentication_error"


class ForbiddenError(ProxyError):
    status = 403
    kind = "permission_error"


class UnsupportedMediaType(InvalidRequest):
    status = 415


def error_response(err: ProxyError, headers: dict[str, str] | None = None) -> JSONResponse:
    """Render ``err`` as a JSON response (the one place P1 will replace)."""
    body = {"error": {"type": err.kind, "message": err.message, "param": err.param}}
    return JSONResponse(status_code=err.status, content=body, headers=headers)
