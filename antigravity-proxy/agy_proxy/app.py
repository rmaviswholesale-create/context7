"""FastAPI application factory: mounts both adapters, health, auth, lifecycle."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import __version__, anthropic_api, openai_api
from .config import Settings
from .context import ProxyContext
from .errors import install_error_handlers
from .presets import PresetRegistry
from .runner import AgentRunner
from .runner_events import configure as configure_tool_trace
from .security import (
    HostOriginMiddleware,
    TokenAuthMiddleware,
    ensure_token,
    scrub_process_env,
)
from .sessions import SessionStore

logger = logging.getLogger("agy_proxy")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = ensure_token(settings or Settings.from_env())
    assert settings.proxy_token  # ensure_token guarantees one
    # Secrets now live only in Settings; keep them out of the agent's environment.
    scrub_process_env()
    configure_tool_trace(trace_tools=settings.enable_tool_trace)

    registry = PresetRegistry.load(
        settings.presets_path,
        settings.default_model,
        workspace=settings.workspace,
        enable_coder=settings.enable_coder,
    )
    runner = AgentRunner(registry, settings.api_key, settings.workspace)
    sessions = SessionStore(runner, settings.session_ttl_seconds, settings.session_max)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not settings.api_key:
            logger.warning(
                "No AGY_API_KEY/GEMINI_API_KEY set; the SDK will fall back to ADC. "
                "Set your own key to use the free Gemini tier."
            )
        for name, reason in registry.disabled.items():
            logger.warning("preset %r not available: %s", name, reason)
        logger.info(
            "agy-proxy %s ready: presets=%s default=%s workspace=%s",
            __version__,
            registry.names(),
            registry.default_name,
            settings.workspace,
        )
        yield
        await sessions.close_all()

    # No interactive docs / schema: they are unauthenticated attack surface.
    app = FastAPI(
        title="agy-proxy",
        version=__version__,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.ctx = ProxyContext(
        settings=settings, registry=registry, runner=runner, sessions=sessions
    )

    install_error_handlers(app)

    app.include_router(openai_api.router)
    app.include_router(anthropic_api.router)

    @app.get("/healthz")
    async def healthz():
        # Unauthenticated: liveness only. Details are logged at startup, not served.
        return {"status": "ok"}

    @app.get("/")
    async def root():
        return {
            "name": "agy-proxy",
            "version": __version__,
            "endpoints": ["/v1/chat/completions", "/v1/messages", "/v1/models", "/healthz"],
        }

    # add_middleware stacks outward: Host/Origin runs first, then token auth.
    app.add_middleware(TokenAuthMiddleware, token=settings.proxy_token)
    app.add_middleware(HostOriginMiddleware, allowed_hosts=settings.allowed_hosts)

    return app
