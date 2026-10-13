"""Entrypoint: ``python -m agy_proxy`` / ``agy-proxy``."""

from __future__ import annotations

import logging
import sys

import uvicorn

from .app import create_app
from .config import ConfigError, Settings
from .redact import install_log_redaction, register_secrets
from .security import adopt_inherited_secrets, reexec_clean


def main() -> None:
    # Re-exec once with an environment that holds no secrets (they travel through a pipe),
    # so /proc/<pid>/environ of the serving process cannot leak them to the agent's shell.
    reexec_clean()
    adopt_inherited_secrets()
    try:
        settings = Settings.from_env()
    except ConfigError as exc:
        raise SystemExit(f"configuration error: {exc}") from exc
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    register_secrets(settings.api_key, settings.proxy_token)
    install_log_redaction()
    app = create_app(settings)
    token = app.state.ctx.settings.proxy_token
    if app.state.ctx.settings.token_generated:
        # Shown once, on stderr only (never logged). Clients send it as the API key.
        print(
            f"agy-proxy: AGY_PROXY_TOKEN not set; generated a token for this run:\n  {token}\n"
            "Use it as the API key (Authorization: Bearer / x-api-key). "
            "Set AGY_PROXY_TOKEN to make it stable.",
            file=sys.stderr,
            flush=True,
        )
    # log_config=None: uvicorn's loggers propagate to the (redacting) root handler.
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        log_config=None,
    )


if __name__ == "__main__":
    main()
