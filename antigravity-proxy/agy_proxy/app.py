"""FastAPI application factory: mounts both adapters, health, auth, lifecycle."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import __version__, anthropic_api, openai_api
from .config import Settings
from .context import ProxyContext
from .presets import PresetRegistry
from .runner import AgentRunner
from .sessions import SessionStore

logger = logging.getLogger("agy_proxy")

# Public, unauthenticated paths (everything else requires the bearer if configured).
_OPEN_PATHS = {"/healthz", "/", "/docs", "/openapi.json", "/redoc"}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    registry = PresetRegistry.load(settings.presets_path, settings.default_model)
    runner = AgentRunner(registry, settings.api_key)
    sessions = SessionStore(runner, settings.session_ttl_seconds, settings.session_max)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not settings.api_key:
            logger.warning(
                "No AGY_API_KEY/GEMINI_API_KEY set; the SDK will fall back to ADC. "
                "Set your own key to use the free Gemini tier."
            )
        logger.info(
            "agy-proxy %s ready: presets=%s default=%s",
            __version__,
            registry.names(),
            registry.default_name,
        )
        yield
        await sessions.close_all()

    app = FastAPI(title="agy-proxy", version=__version__, lifespan=lifespan)
    app.state.ctx = ProxyContext(
        settings=settings, registry=registry, runner=runner, sessions=sessions
    )

    @app.middleware("http")
    async def auth_guard(request: Request, call_next):
        token = settings.proxy_token
        if token and request.url.path not in _OPEN_PATHS:
            provided = _bearer(request)
            if provided != token:
                return JSONResponse(status_code=401, content={"error": "unauthorized"})
        return await call_next(request)

    app.include_router(openai_api.router)
    app.include_router(anthropic_api.router)

    @app.get("/healthz")
    async def healthz():
        return {
            "status": "ok",
            "version": __version__,
            "presets": registry.names(),
            "default_preset": registry.default_name,
            "live_sessions": sessions.size(),
            "api_key_configured": bool(settings.api_key),
        }

    @app.get("/")
    async def root():
        return {
            "name": "agy-proxy",
            "version": __version__,
            "endpoints": ["/v1/chat/completions", "/v1/messages", "/v1/models", "/healthz"],
        }

    return app


def _bearer(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    # Anthropic clients send x-api-key.
    return request.headers.get("x-api-key")
