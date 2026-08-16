"""Shared request-time context wired onto the FastAPI app state."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from .config import Settings
from .presets import PresetRegistry
from .runner import AgentRunner
from .sessions import SessionStore


@dataclass
class ProxyContext:
    settings: Settings
    registry: PresetRegistry
    runner: AgentRunner
    sessions: SessionStore


def get_ctx(request: Request) -> ProxyContext:
    return request.app.state.ctx
