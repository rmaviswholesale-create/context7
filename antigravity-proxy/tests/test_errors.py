"""Error mapping table, asserted per row on both protocol routes."""

from __future__ import annotations

import re

import pytest
from conftest import ScriptedRunner
from fastapi.testclient import TestClient

from agy_proxy.app import create_app
from agy_proxy.config import Settings
from agy_proxy.errors import (
    AuthError,
    ForbiddenError,
    GatewayTimeout,
    InternalError,
    InvalidRequest,
    OverloadedError,
    PayloadTooLarge,
    RateLimited,
    UnknownModelError,
    UpstreamError,
)

TOKEN = "tok-err"
H = {"authorization": f"Bearer {TOKEN}", "content-type": "application/json"}
OAI = ("/v1/chat/completions", {"model": "agy-light", "messages": [{"role": "user", "content": "hi"}]})
ANT = (
    "/v1/messages",
    {"model": "agy-light", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]},
)

# class, status, anthropic type, openai type, openai code
ROWS = [
    (InvalidRequest("bad"), 400, "invalid_request_error", "invalid_request_error", None),
    (AuthError("no"), 401, "authentication_error", "invalid_request_error", "invalid_api_key"),
    (ForbiddenError("no"), 403, "permission_error", "permission_error", None),
    (UnknownModelError("no"), 404, "not_found_error", "invalid_request_error", "model_not_found"),
    (PayloadTooLarge("big"), 413, "request_too_large", "invalid_request_error", "request_too_large"),
    (RateLimited("slow"), 429, "rate_limit_error", "rate_limit_error", "rate_limit_exceeded"),
    (OverloadedError("full"), 429, "overloaded_error", "server_error", None),
    (OverloadedError("up", upstream=True), 529, "overloaded_error", "server_error", None),
    (InternalError("oops"), 500, "api_error", "server_error", None),
    (UpstreamError("bad gateway"), 502, "api_error", "server_error", None),
    (GatewayTimeout("slow"), 504, "api_error", "server_error", None),
]
CLASS_NAME = re.compile(r"\b\w+(Error|Exception)\b")


def build(tmp_path, runner):
    app = create_app(Settings(proxy_token=TOKEN, workspace=str(tmp_path), api_key="k"))
    app.state.ctx.runner = runner
    return TestClient(app, base_url="http://127.0.0.1:8081", raise_server_exceptions=False)


def assert_anthropic(r, status, kind, message=None):
    assert r.status_code == status, r.text
    body = r.json()
    assert body["type"] == "error" and set(body["error"]) == {"type", "message"}
    assert body["error"]["type"] == kind
    assert not CLASS_NAME.search(body["error"]["message"])
    if message:
        assert body["error"]["message"] == message


def assert_openai(r, status, kind, code):
    assert r.status_code == status, r.text
    err = r.json()["error"]
    assert set(err) == {"message", "type", "param", "code"}
    assert err["type"] == kind and err["code"] == code
    assert not CLASS_NAME.search(err["message"])


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("err, status, a_type, o_type, o_code", ROWS)
def test_proxy_error_rows_on_both_routes(tmp_path, err, status, a_type, o_type, o_code, stream):
    with build(tmp_path, ScriptedRunner(exc=err)) as c:
        path, body = ANT
        assert_anthropic(c.post(path, json={**body, "stream": stream}, headers=H), status, a_type)
        path, body = OAI
        assert_openai(c.post(path, json={**body, "stream": stream}, headers=H), status, o_type, o_code)


def test_auth_failures_use_route_envelope(tmp_path):
    with build(tmp_path, ScriptedRunner()) as c:
        r = c.post(ANT[0], json=ANT[1], headers={"content-type": "application/json"})
        assert_anthropic(r, 401, "authentication_error")
        assert r.headers["www-authenticate"] == "Bearer"
        r = c.post(OAI[0], json=OAI[1], headers={"content-type": "application/json"})
        assert_openai(r, 401, "invalid_request_error", "invalid_api_key")


def test_host_origin_and_content_type_rejections(tmp_path):
    with build(tmp_path, ScriptedRunner()) as c:
        r = c.post(ANT[0], json=ANT[1], headers={**H, "host": "evil.example"})
        assert_anthropic(r, 403, "permission_error")
        r = c.post(OAI[0], json=OAI[1], headers={**H, "origin": "http://evil.example"})
        assert_openai(r, 403, "permission_error", None)
        r = c.post(OAI[0], content="x", headers={**H, "content-type": "text/plain"})
        assert_openai(r, 415, "invalid_request_error", None)
        r = c.post(ANT[0], content="x", headers={**H, "content-type": "text/plain"})
        assert_anthropic(r, 415, "invalid_request_error")


def test_validation_errors_are_400_not_422(tmp_path):
    with build(tmp_path, ScriptedRunner()) as c:
        r = c.post(ANT[0], json={"model": "agy-light"}, headers=H)
        assert_anthropic(r, 400, "invalid_request_error")
        assert "messages" in r.json()["error"]["message"]
        r = c.post(OAI[0], json={"model": "agy-light"}, headers=H)
        assert_openai(r, 400, "invalid_request_error", None)
        assert r.json()["error"]["param"] == "messages"
        r = c.post(OAI[0], content="{not json", headers=H)
        assert_openai(r, 400, "invalid_request_error", None)


def test_unknown_model_envelopes(tmp_path):
    with build(tmp_path, ScriptedRunner()) as c:
        path, body = ANT
        assert_anthropic(
            c.post(path, json={**body, "model": "gpt-4o", "max_tokens": 500}, headers=H),
            404,
            "not_found_error",
        )
        path, body = OAI
        assert_openai(
            c.post(path, json={**body, "model": "gpt-4o"}, headers=H),
            404,
            "invalid_request_error",
            "model_not_found",
        )


def test_unknown_route_and_method(tmp_path):
    with build(tmp_path, ScriptedRunner()) as c:
        r = c.get("/v1/nope", headers=H)
        assert_openai(r, 404, "invalid_request_error", None)
        r = c.get("/v1/messages", headers=H)
        assert_anthropic(r, 405, "invalid_request_error")


def test_unexpected_exception_is_a_500_envelope_without_leaks(tmp_path):
    class Boom:
        def names(self):
            raise RuntimeError("secret internal detail /home/x")

    with build(tmp_path, ScriptedRunner()) as c:
        c.app.state.ctx.registry = Boom()
        r = c.post(ANT[0], json=ANT[1], headers=H)
        assert_anthropic(r, 500, "api_error", "internal server error")
        r = c.post(OAI[0], json=OAI[1], headers=H)
        assert_openai(r, 500, "server_error", None)
        assert "secret" not in r.text and "RuntimeError" not in r.text


def test_agent_exception_message_is_sanitized_and_typed(tmp_path):
    with build(tmp_path, ScriptedRunner(exc=ValueError("harness exploded"))) as c:
        r = c.post(OAI[0], json=OAI[1], headers=H)
        assert_openai(r, 502, "server_error", None)
        assert "harness exploded" in r.json()["error"]["message"]
        assert "ValueError" not in r.text
