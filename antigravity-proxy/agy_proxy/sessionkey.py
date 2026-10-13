"""Session identity: how a request opts in to a live, memory-preserving agent.

Only the explicit ``x-agy-session`` header selects a session (the OpenAI ``user`` field and
``x-session-id`` are no longer honoured: they are routinely set by clients to end-user ids
and would silently turn every request stateful). The key that indexes the store is
``sha256(token \\0 explicit-id)``, so two callers holding different proxy tokens can never
reach each other's sessions even if they pick the same id.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .errors import InvalidRequest

SESSION_HEADER = "x-agy-session"
_VALID = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


def derive_session_key(token: str, explicit: str | None) -> str | None:
    """Store key for ``explicit`` under ``token``; None when no session was requested.

    A blank header counts as absent; anything else must match ``^[A-Za-z0-9_.:-]{1,64}$``.
    """
    if explicit is None or not explicit.strip():
        return None
    explicit = explicit.strip()
    if not _VALID.fullmatch(explicit):
        raise InvalidRequest(
            f"{SESSION_HEADER} must match [A-Za-z0-9_.:-]{{1,64}}", param=SESSION_HEADER
        )
    return hashlib.sha256(f"{token}\0{explicit}".encode()).hexdigest()


def session_key_from_headers(token: str, headers: Mapping[str, str]) -> str | None:
    """Convenience for adapters: ``derive_session_key(token, headers.get('x-agy-session'))``."""
    return derive_session_key(token, headers.get(SESSION_HEADER))


def _role(message: Any) -> str | None:
    if isinstance(message, Mapping):
        return message.get("role")
    return getattr(message, "role", None)


def select_turn_messages(messages: Sequence[Any], is_new: bool) -> list[Any]:
    """Which messages to send on this turn of a session.

    A brand-new live agent knows nothing, so the client's whole history is replayed. An
    existing agent already holds the earlier turns, so only the trailing messages after the
    last assistant message (the new user/tool input) are sent. System messages never count:
    they are folded into the agent's instructions when it is created.
    """
    convo = [m for m in messages if _role(m) != "system"]
    if is_new:
        return convo
    for i in range(len(convo) - 1, -1, -1):
        if _role(convo[i]) == "assistant":
            return convo[i + 1 :]
    return convo
