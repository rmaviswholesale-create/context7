"""Entrypoint: ``python -m agy_proxy`` / ``agy-proxy``."""

from __future__ import annotations

import logging
import sys

import uvicorn

from .app import create_app
from .config import ConfigError, Settings


def main() -> None:
    try:
        settings = Settings.from_env()
    except ConfigError as exc:
        raise SystemExit(f"configuration error: {exc}") from exc
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
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
    uvicorn.run(app, host=settings.host, port=settings.port, log_level=settings.log_level)


if __name__ == "__main__":
    main()
