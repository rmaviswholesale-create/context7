from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agy_proxy.app import create_app
from agy_proxy.config import Settings
from agy_proxy.events import DoneEvent, TextEvent
from agy_proxy.security import check_bind, make_token, scrub_process_env

TOKEN = "s3cret-token"
JSON = {"content-type": "application/json"}
OAI = {"model": "agy-light", "messages": [{"role": "user", "content": "hi"}]}
ANT = {"model": "agy-light", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}


class StubRunner:
    def __init__(self):
        self.calls = 0

    async def stream_stateless(self, preset, messages):
        self.calls += 1
        yield TextEvent("ok")
        yield DoneEvent(stop_reason=None, input_tokens=1, output_tokens=1)


@pytest.fixture
def client(tmp_path):
    settings = Settings(proxy_token=TOKEN, workspace=str(tmp_path), api_key="gemini-key")
    app = create_app(settings)
    app.state.ctx.runner = StubRunner()
    with TestClient(app, base_url="http://127.0.0.1:8081") as c:
        yield c


def _auth(**extra):
    return {"authorization": f"Bearer {TOKEN}", **JSON, **extra}


def test_healthz_is_open_and_minimal(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


@pytest.mark.parametrize("path", ["/", "/v1/models"])
def test_get_requires_token(client, path):
    assert client.get(path).status_code == 401
    assert client.get(path, headers={"authorization": "Bearer nope"}).status_code == 401
    assert client.get(path, headers={"authorization": TOKEN}).status_code == 401  # no scheme
    assert client.get(path, headers={"authorization": f"Bearer {TOKEN}"}).status_code == 200
    assert client.get(path, headers={"x-api-key": TOKEN}).status_code == 200
    assert client.get(path, headers={"x-api-key": "nope"}).status_code == 401


@pytest.mark.parametrize(
    "path, body", [("/v1/chat/completions", OAI), ("/v1/messages", ANT)]
)
def test_post_auth_matrix(client, path, body):
    assert client.post(path, json=body).status_code == 401
    assert client.post(path, json=body, headers={"authorization": "Bearer x", **JSON}).status_code == 401
    assert client.post(path, json=body, headers={"x-api-key": "x", **JSON}).status_code == 401
    assert client.post(path, json=body, headers=_auth()).status_code == 200
    assert client.post(path, json=body, headers={"x-api-key": TOKEN, **JSON}).status_code == 200
    assert client.app.state.ctx.runner.calls == 2


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_docs_are_disabled(client, path):
    assert client.get(path, headers=_auth()).status_code == 404


def test_host_allowlist(client):
    h = {"authorization": f"Bearer {TOKEN}"}
    assert client.get("/", headers={**h, "host": "localhost:8081"}).status_code == 200
    assert client.get("/", headers={**h, "host": "[::1]:8081"}).status_code == 200
    for bad in ("evil.example", "evil.example:8081", "127.0.0.1.evil.example", ""):
        assert client.get("/", headers={**h, "host": bad}).status_code == 403, bad
    # Even the open health check enforces Host (DNS rebinding).
    assert client.get("/healthz", headers={"host": "evil.example"}).status_code == 403


def test_any_origin_header_is_rejected(client):
    for origin in ("http://evil.example", "http://127.0.0.1:8081", "null"):
        r = client.post("/v1/chat/completions", json=OAI, headers=_auth(origin=origin))
        assert r.status_code == 403, origin
    assert client.get("/", headers={**_auth(), "origin": "http://x"}).status_code == 403


def test_post_must_be_json(client):
    base = {"authorization": f"Bearer {TOKEN}"}
    for ctype in ("text/plain", "application/x-www-form-urlencoded", "multipart/form-data"):
        r = client.post("/v1/chat/completions", content=b"x=1", headers={**base, "content-type": ctype})
        assert r.status_code == 415, ctype
    r = client.post("/v1/chat/completions", content=b"{}", headers=base)
    assert r.status_code == 415
    ok = {**base, "content-type": "application/json; charset=utf-8"}
    assert client.post("/v1/chat/completions", json=OAI, headers=ok).status_code == 200


@pytest.mark.parametrize("path, body", [("/v1/chat/completions", OAI), ("/v1/messages", ANT)])
@pytest.mark.parametrize("stream", [False, True])
def test_unknown_model_is_404(client, path, body, stream):
    r = client.post(path, json={**body, "model": "gpt-4o", "stream": stream}, headers=_auth())
    assert r.status_code == 404
    assert client.app.state.ctx.runner.calls == 0


def test_disabled_coder_is_404(client):
    r = client.post("/v1/chat/completions", json={**OAI, "model": "agy-coder"}, headers=_auth())
    assert r.status_code == 404


def test_missing_model_uses_default(client):
    body = {"messages": OAI["messages"]}
    r = client.post("/v1/chat/completions", json=body, headers=_auth())
    assert r.status_code == 200


def test_non_loopback_bind_without_token_exits():
    with pytest.raises(SystemExit, match="AGY_PROXY_TOKEN"):
        create_app(Settings(host="0.0.0.0", proxy_token=None))
    with pytest.raises(SystemExit):
        check_bind("192.168.1.5", None)
    check_bind("192.168.1.5", "tok")
    check_bind("127.0.0.1", None)
    check_bind("::1", None)
    check_bind("localhost", None)


def test_loopback_without_token_autogenerates(tmp_path):
    app = create_app(Settings(workspace=str(tmp_path)))
    s = app.state.ctx.settings
    assert s.proxy_token and s.token_generated


def test_make_token_is_random_and_long():
    assert make_token() != make_token() and len(make_token()) >= 32


def test_create_app_scrubs_environment(monkeypatch, tmp_path):
    for k, v in {
        "AGY_PROXY_TOKEN": "t",
        "AGY_API_KEY": "k",
        "GEMINI_API_KEY": "g",
        "MY_SERVICE_SECRET": "s",
        "SOME_AUTH_TOKEN": "x",
        "GOOGLE_CLOUD_PROJECT": "proj",
        "HOME": "/home/x",
    }.items():
        monkeypatch.setenv(k, v)
    import os

    settings = Settings.from_env()
    assert settings.proxy_token == "t" and settings.api_key == "k"
    create_app(settings)
    for gone in ("AGY_PROXY_TOKEN", "AGY_API_KEY", "GEMINI_API_KEY", "MY_SERVICE_SECRET", "SOME_AUTH_TOKEN"):
        assert gone not in os.environ
    assert os.environ["GOOGLE_CLOUD_PROJECT"] == "proj"
    assert os.environ["HOME"] == "/home/x" and "PATH" in os.environ
    assert not any("antigravity_" in v for v in os.environ.values())


def test_scrub_process_env_unit():
    env = {"PATH": "/bin", "HOME": "/h", "GOOGLE_API_KEY": "x", "AGY_X": "1", "FOO_KEY": "2", "OK": "3"}
    removed = scrub_process_env(env)
    assert env == {"PATH": "/bin", "HOME": "/h", "GOOGLE_API_KEY": "x", "OK": "3"}
    assert sorted(removed) == ["AGY_X", "FOO_KEY"]
