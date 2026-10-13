"""Server-side log redaction and per-request correlation ids.

Client-visible errors are generic ("upstream error"); the detail is logged here, tagged
with the request's correlation id, after secrets are removed:

* exact values registered with :func:`register_secrets` (API key, proxy token);
* ``key=...`` / ``token=...`` / ``secret=...`` style assignments and query parameters;
* ``Authorization: Bearer ...`` / ``x-api-key: ...`` headers and Google ``AIza...`` keys.
"""

from __future__ import annotations

import contextvars
import logging
import re
import secrets
import traceback

REDACTED = "[redacted]"

_secrets: set[str] = set()
_MIN_SECRET_LEN = 4  # shorter values would shred ordinary text

_PATTERNS = [
    # key=VALUE, token: VALUE, "api_key": "VALUE", ?apikey=VALUE ...
    re.compile(
        r"""(?ix)
        (\b[\w.-]*(?:api[_-]?key|key|token|secret|password|passwd|credential)s?["']?
        \s*[=:]\s*["']?)
        ([^\s"'&,;)}\]]+)
        """
    ),
    re.compile(r"(?i)(\bbearer\s+)([A-Za-z0-9._~+/=-]+)"),
    re.compile(r"(?i)(\b(?:x-api-key|authorization)\s*:\s*)(\S+(?:\s+\S+)?)"),
    re.compile(r"()(AIza[0-9A-Za-z_-]{20,})"),
]

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agy_request_id", default=None
)


def new_request_id() -> str:
    return "req_" + secrets.token_hex(12)


def current_request_id() -> str | None:
    return request_id_var.get()


def register_secrets(*values: str | None) -> None:
    """Remember exact secret values to strip from every redacted string."""
    for v in values:
        if v and len(v) >= _MIN_SECRET_LEN:
            _secrets.add(v)


def clear_secrets() -> None:  # tests
    _secrets.clear()


def redact(text: str) -> str:
    # Longest first so a secret that contains another is removed whole.
    for s in sorted(_secrets, key=len, reverse=True):
        text = text.replace(s, REDACTED)
    for pat in _PATTERNS:
        text = pat.sub(lambda m: f"{m.group(1)}{REDACTED}", text)
    return text


def format_exception(exc: BaseException) -> str:
    """Redacted traceback text (formatted here so it is safe with any log handler)."""
    return redact("".join(traceback.format_exception(exc)))


def log_failure(
    logger: logging.Logger, what: str, exc: BaseException, request_id: str | None = None
) -> None:
    rid = request_id or current_request_id() or "-"
    logger.warning("%s [request_id=%s]: %s", what, rid, format_exception(exc))


class RedactingFormatter(logging.Formatter):
    """Wraps another formatter and redacts the final text (message + traceback)."""

    def __init__(self, inner: logging.Formatter | None = None):
        super().__init__()
        self._inner = inner or logging.Formatter()

    def format(self, record: logging.LogRecord) -> str:
        return redact(self._inner.format(record))


def install_log_redaction(handlers: list[logging.Handler] | None = None) -> None:
    """Wrap the formatter of the root logger's handlers (or ``handlers``)."""
    for h in handlers if handlers is not None else logging.getLogger().handlers:
        if not isinstance(h.formatter, RedactingFormatter):
            h.setFormatter(RedactingFormatter(h.formatter))
