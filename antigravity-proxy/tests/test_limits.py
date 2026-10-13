"""Admission (429), turn timeout (504), body cap (413), disconnect handling, session keys."""

from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import asynccontextmanager

import httpx
import pytest

from agy_proxy.app import create_app
from agy_proxy.config import ConfigError, Settings
from agy_proxy.events import DoneEvent, TextEvent
from agy_proxy.limits import Admission, AdmissionMiddleware, BodyLimitMiddleware
from agy_proxy.sessionkey import derive_session_key, select_turn_messages
from agy_proxy.sessions import SessionLease

TOKEN = "s3cret-token"
H = {"authorization": f"Bearer {TOKEN}", "content-type": "application/json"}
OAI = {"model": "agy-light", "messages": [{"role": "user", "content": "hi"}]}
ANT = {"model": "agy-light", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}
PATHS = [("/v1/chat/completions", OAI), ("/v1/messages", ANT)]


class GatedRunner:
    """Stateless runner whose turns wait on an event, recording entry and cancellation."""

    def __init__(self):
        self.gate = asyncio.Event()
        self.entered = 0
        self.cancelled = 0
        self.finished = 0
        self.stateful: list[tuple] = []

    async def stream_stateless(self, preset, messages):
        self.entered += 1
        try:
            await self.gate.wait()
            yield TextEvent("ok")
            yield DoneEvent(stop_reason=None, input_tokens=1, output_tokens=1)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.finished += 1

    async def stream_on_agent(self, agent, messages, is_new=False):
        self.stateful.append((agent, messages, is_new))
        yield TextEvent("ok")
        yield DoneEvent(stop_reason=None, input_tokens=1, output_tokens=1)


class RecordingSessions:
    def __init__(self):
        self.keys: list[str] = []

    @asynccontextmanager
    async def acquire(self, key, preset, system_extra):
        self.keys.append(key)
        lease = SessionLease(object(), is_new=True)
        yield lease
        lease.complete()


def make_app(tmp_path, **kw):
    kw.setdefault("max_body_bytes", 1024 * 1024)  # the shipped default is 32 MiB
    settings = Settings(
        proxy_token=TOKEN, workspace=str(tmp_path), api_key="k", **kw
    )
    app = create_app(settings)
    runner = GatedRunner()
    app.state.ctx.runner = runner
    return app, runner


def client(app):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8081"
    )


# -- body cap ---------------------------------------------------------------------------


@pytest.mark.parametrize("path,body", PATHS)
async def test_oversize_body_413_with_content_length(tmp_path, path, body):
    app, runner = make_app(tmp_path)
    big = {**body, "pad": "x" * (2 * 1024 * 1024)}
    async with client(app) as c:
        r = await c.post(path, headers=H, content=json.dumps(big))
    assert r.status_code == 413 and runner.entered == 0
    assert "error" in r.json()


@pytest.mark.parametrize("path,body", PATHS)
async def test_oversize_chunked_body_413_without_content_length(tmp_path, path, body):
    app, runner = make_app(tmp_path)
    payload = json.dumps({**body, "pad": "x" * (2 * 1024 * 1024)}).encode()

    async def chunks():
        for i in range(0, len(payload), 64 * 1024):
            yield payload[i : i + 64 * 1024]

    async with client(app) as c:
        req = c.build_request("POST", path, headers=H, content=chunks())
        assert "content-length" not in req.headers
        r = await c.send(req)
    assert r.status_code == 413 and runner.entered == 0


async def test_body_limit_counts_streamed_bytes_at_asgi_level():
    sent: list[dict] = []
    reached_app = False

    async def app(scope, receive, send):
        nonlocal reached_app
        reached_app = True
        while True:  # a handler that drains the body
            msg = await receive()
            if msg["type"] != "http.request" or not msg.get("more_body"):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"fine"})

    mw = BodyLimitMiddleware(app, max_bytes=100)
    feed = [
        {"type": "http.request", "body": b"a" * 60, "more_body": True},
        {"type": "http.request", "body": b"b" * 60, "more_body": True},
        {"type": "http.request", "body": b"", "more_body": False},
    ]

    async def receive():
        return feed.pop(0) if feed else {"type": "http.disconnect"}

    async def send(m):
        sent.append(m)

    scope = {"type": "http", "method": "POST", "path": "/v1/messages", "headers": []}
    await mw(scope, receive, send)
    assert reached_app
    assert sent[0]["status"] == 413  # not the app's 200, and not a 400/500


async def test_body_at_limit_passes(tmp_path):
    app, runner = make_app(tmp_path, max_body_bytes=1024 * 1024)
    runner.gate.set()
    async with client(app) as c:
        r = await c.post("/v1/chat/completions", headers=H, json=OAI)
    assert r.status_code == 200


# -- admission --------------------------------------------------------------------------


@pytest.mark.parametrize("path,body", PATHS)
async def test_fifth_concurrent_request_gets_429_then_slots_free(tmp_path, path, body):
    app, runner = make_app(tmp_path, max_concurrency=4)
    async with client(app) as c:
        first = [asyncio.ensure_future(c.post(path, headers=H, json=body)) for _ in range(4)]
        while runner.entered < 4:
            await asyncio.sleep(0.01)
        r5 = await asyncio.wait_for(c.post(path, headers=H, json=body), 2)  # immediate, no queue
        assert r5.status_code == 429 and r5.headers["retry-after"]
        assert runner.entered == 4
        runner.gate.set()
        assert [r.status_code for r in await asyncio.gather(*first)] == [200] * 4
        assert (await c.post(path, headers=H, json=body)).status_code == 200


async def test_unauthenticated_requests_do_not_consume_slots(tmp_path):
    app, runner = make_app(tmp_path, max_concurrency=1)
    runner.gate.set()
    async with client(app) as c:
        for _ in range(3):
            assert (await c.post("/v1/messages", json=ANT, headers={"content-type": "application/json"})).status_code == 401
        assert (await c.post("/v1/messages", headers=H, json=ANT)).status_code == 200


async def test_turn_timeout_returns_504_and_cancels_the_turn(tmp_path):
    app, runner = make_app(tmp_path, turn_timeout=1)
    async with client(app) as c:
        r = await asyncio.wait_for(c.post("/v1/chat/completions", headers=H, json=OAI), 10)
    assert r.status_code == 504 and "error" in r.json()
    assert runner.cancelled == 1 and runner.finished == 1  # generator cleanup really ran
    # the slot was released
    runner.gate.set()
    async with client(app) as c:
        assert (await c.post("/v1/chat/completions", headers=H, json=OAI)).status_code == 200


async def test_client_disconnect_cancels_nonstreaming_turn():
    """After the body is read nobody else listens for a hang-up; Admission must."""
    cancelled = asyncio.Event()

    async def app(scope, receive, send):
        await receive()  # the whole body
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    adm = Admission(4, 60)
    mw = AdmissionMiddleware(app, adm)
    feed = [{"type": "http.request", "body": b"{}", "more_body": False}]
    disconnect = asyncio.Event()

    async def receive():
        if feed:
            return feed.pop(0)
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(m):  # pragma: no cover - nothing is sent
        raise AssertionError(m)

    scope = {"type": "http", "method": "POST", "path": "/v1/messages", "headers": []}
    task = asyncio.ensure_future(mw(scope, receive, send))
    await asyncio.sleep(0.05)
    assert adm.inflight == 1
    disconnect.set()
    await asyncio.wait_for(task, 2)  # returns normally, does not raise CancelledError
    assert cancelled.is_set() and adm.inflight == 0


async def test_cancelling_the_request_task_propagates():
    async def app(scope, receive, send):
        await asyncio.sleep(30)

    adm = Admission(1, 60)
    mw = AdmissionMiddleware(app, adm)

    async def receive():
        await asyncio.sleep(30)

    scope = {"type": "http", "method": "POST", "path": "/v1/messages", "headers": []}
    task = asyncio.ensure_future(mw(scope, receive, lambda m: None))
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert adm.inflight == 0


# -- session keys -----------------------------------------------------------------------


def test_session_key_isolated_per_token_and_hashed():
    a = derive_session_key("token-A", "chat-1")
    b = derive_session_key("token-B", "chat-1")
    assert a != b and a == hashlib.sha256(b"token-A\0chat-1").hexdigest()
    assert "chat-1" not in a
    assert derive_session_key("token-A", None) is None
    assert derive_session_key("token-A", "  ") is None


@pytest.mark.parametrize("bad", ["a b", "x" * 65, "a/b", "é", "a\nb", "../x"])
def test_session_key_rejects_bad_ids(bad):
    from agy_proxy.errors import InvalidRequest

    with pytest.raises(InvalidRequest):
        derive_session_key("t", bad)


def test_select_turn_messages():
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    assert [m["content"] for m in select_turn_messages(msgs, True)] == ["q1", "a1", "q2"]
    assert [m["content"] for m in select_turn_messages(msgs, False)] == ["q2"]
    only = [{"role": "user", "content": "q"}]
    assert select_turn_messages(only, False) == only


async def test_only_x_agy_session_header_selects_a_session(tmp_path):
    app, runner = make_app(tmp_path)
    runner.gate.set()
    sessions = RecordingSessions()
    app.state.ctx.sessions = sessions
    async with client(app) as c:
        # the OpenAI `user` field and the legacy x-session-id no longer create sessions
        r = await c.post("/v1/chat/completions", headers={**H, "x-session-id": "s"}, json={**OAI, "user": "u"})
        assert r.status_code == 200 and sessions.keys == [] and runner.entered == 1
        r = await c.post("/v1/chat/completions", headers={**H, "x-agy-session": "chat-1"}, json=OAI)
        assert r.status_code == 200
        r = await c.post("/v1/messages", headers={**H, "x-agy-session": "chat-1"}, json=ANT)
        assert r.status_code == 200
        bad = await c.post("/v1/messages", headers={**H, "x-agy-session": "no spaces"}, json=ANT)
        assert bad.status_code == 400
    assert sessions.keys == [hashlib.sha256(f"{TOKEN}\0chat-1".encode()).hexdigest()] * 2
    assert runner.entered == 1 and len(runner.stateful) == 2 and runner.stateful[0][2] is True


# -- config -----------------------------------------------------------------------------


def test_p2_config_defaults_and_validation():
    s = Settings.from_env({"AGY_PROXY_TOKEN": "t"})
    assert (s.max_concurrency, s.turn_timeout, s.max_body_bytes, s.teardown_timeout) == (
        4,
        600,
        32 * 1024 * 1024,
        10,
    )
    assert Settings.from_env({"AGY_PROXY_TOKEN": "t", "AGY_MAX_CONCURRENCY": "2"}).max_concurrency == 2
    for name, val in [("AGY_MAX_CONCURRENCY", "0"), ("AGY_TURN_TIMEOUT", "x"), ("AGY_MAX_BODY", "5")]:
        with pytest.raises(ConfigError, match=name):
            Settings.from_env({"AGY_PROXY_TOKEN": "t", name: val})


def test_budget_from_settings_reaches_agent_config(tmp_path):
    from agy_proxy.runner import AgentRunner
    from agy_proxy.scratch import Scratch

    app, _ = make_app(tmp_path, max_model_calls=7, max_tool_calls=0)
    runner = AgentRunner(
        app.state.ctx.registry, "k", None, budget=AgentRunner.budget_from(7, 0)
    )
    scratch = Scratch()
    try:
        cfg = runner.build_config(app.state.ctx.registry.resolve("agy-light"), None, scratch)
        assert cfg.budget_config.max_model_calls == 7 and cfg.budget_config.max_tool_calls is None
        long_lived = runner._session_budget()
        assert long_lived.max_model_calls == 7 * 20
    finally:
        scratch.cleanup()
    assert AgentRunner.budget_from(0, 0) is None


# -- merged with P1: middleware errors use the path-aware envelopes ---------------------


def _assert_envelope(path, r, anthropic_type, oai_type):
    data = r.json()
    if path == "/v1/messages":
        assert data["type"] == "error" and data["error"]["type"] == anthropic_type
    else:
        assert data["error"]["type"] == oai_type and "message" in data["error"]


@pytest.mark.parametrize("path,body", PATHS)
async def test_413_and_429_use_protocol_envelopes(tmp_path, path, body):
    app, runner = make_app(tmp_path, max_concurrency=1, max_body_bytes=1024)
    async with client(app) as c:
        r = await c.post(path, headers=H, content=json.dumps({**body, "pad": "x" * 4096}))
        assert r.status_code == 413
        _assert_envelope(path, r, "request_too_large", "invalid_request_error")
        first = asyncio.ensure_future(c.post(path, headers=H, json=body))
        while runner.entered < 1:
            await asyncio.sleep(0.01)
        r = await c.post(path, headers=H, json=body)
        assert r.status_code == 429
        _assert_envelope(path, r, "rate_limit_error", "rate_limit_error")
        runner.gate.set()
        await first


@pytest.mark.parametrize("path,body", PATHS)
async def test_streaming_turn_timeout_before_first_event_is_504(tmp_path, path, body):
    app, _ = make_app(tmp_path, turn_timeout=1)
    async with client(app) as c:
        r = await asyncio.wait_for(c.post(path, headers=H, json={**body, "stream": True}), 10)
    assert r.status_code == 504
    _assert_envelope(path, r, "api_error", "server_error")
