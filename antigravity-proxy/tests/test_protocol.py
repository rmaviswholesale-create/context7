"""Protocol behaviour of both adapters, end to end through TestClient."""

from __future__ import annotations

import json

import pytest
from conftest import (
    ScriptedRunner,
    StubAgent,
    StubResp,
    aiter,
    collect,
    parse_sse,
    parse_sse_events,
)
from fastapi.testclient import TestClient
from google.antigravity import BuiltinTools
from google.antigravity.types import StopReason, Text, Thought, ToolCall, ToolResult

from agy_proxy.anthropic_api import _stream_anthropic
from agy_proxy.app import create_app
from agy_proxy.config import Settings
from agy_proxy.errors import UpstreamError
from agy_proxy.events import DoneEvent, ErrorEvent, TextEvent, ThoughtEvent
from agy_proxy.openai_api import _stream_openai

TOKEN = "tok-p1"
H = {"authorization": f"Bearer {TOKEN}", "content-type": "application/json"}
OAI = {"model": "agy-researcher", "messages": [{"role": "user", "content": "hi"}]}
ANT = {
    "model": "agy-researcher",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "hi"}],
}
ROUTES = [("/v1/chat/completions", OAI), ("/v1/messages", ANT)]


def make_client(tmp_path, runner):
    app = create_app(Settings(proxy_token=TOKEN, workspace=str(tmp_path), api_key="k"))
    app.state.ctx.runner = runner
    return TestClient(app, base_url="http://127.0.0.1:8081", raise_server_exceptions=False)


@pytest.fixture
def mk(tmp_path):
    clients = []

    def build(runner):
        c = make_client(tmp_path, runner)
        c.__enter__()
        clients.append(c)
        return c

    yield build
    for c in clients:
        c.__exit__(None, None, None)


OK = [TextEvent("Hello"), DoneEvent(stop_reason="UNSPECIFIED", input_tokens=9, output_tokens=3)]


# -- no phantom tool calls -----------------------------------------------------


@pytest.mark.parametrize("path, body", ROUTES)
@pytest.mark.parametrize("stream", [False, True])
def test_sdk_tool_activity_never_reaches_the_wire(mk, path, body, stream):
    chunks = [
        Thought(text="plan", step_index=0),
        ToolCall(name=BuiltinTools.VIEW_FILE, args={"path": "/etc/passwd"}, id="t1"),
        ToolResult(name=BuiltinTools.VIEW_FILE, id="t1", result="root:x"),
        Text(text="done", step_index=0),
    ]
    agent = StubAgent(StubResp(chunks, StopReason.UNSPECIFIED))
    c = mk(ScriptedRunner(agent=agent))
    tools = [{"type": "function", "function": {"name": "f"}}]
    r = c.post(path, json={**body, "stream": stream, "tools": tools}, headers=H)
    assert r.status_code == 200
    raw = r.text
    for banned in ("tool_use", "tool_calls", "input_json_delta", "VIEW_FILE", "/etc/passwd"):
        assert banned not in raw
    assert "done" in raw
    assert r.headers["x-agy-ignored"] == "tools"


@pytest.mark.parametrize("path, body", ROUTES)
def test_no_ignored_header_without_tools(mk, path, body):
    c = mk(ScriptedRunner(events=OK))
    assert "x-agy-ignored" not in c.post(path, json=body, headers=H).headers


# -- stop reasons ----------------------------------------------------------------


@pytest.mark.parametrize(
    "sdk, openai, anthropic",
    [
        (None, "stop", "end_turn"),
        ("UNSPECIFIED", "stop", "end_turn"),
        ("MAX_OUTPUT_TOKENS_EXCEEDED", "length", "max_tokens"),
        ("MAX_TOTAL_TOKENS_EXCEEDED", "length", "max_tokens"),
        ("MAX_INPUT_TOKENS_EXCEEDED", "length", "max_tokens"),
        ("MAX_MODEL_CALLS_EXCEEDED", "stop", "end_turn"),
        ("MAX_TOOL_CALLS_EXCEEDED", "stop", "end_turn"),
    ],
)
async def test_stop_reason_table(sdk, openai, anthropic):
    events = [TextEvent("x"), DoneEvent(stop_reason=sdk)]
    chunks = parse_sse(await collect(_stream_openai(aiter(list(events)), "m")))
    assert chunks[-1]["choices"][0]["finish_reason"] == openai
    parsed = parse_sse_events(await collect(_stream_anthropic(aiter(list(events)), "m")))
    assert next(d for n, d in parsed if n == "message_delta")["delta"]["stop_reason"] == anthropic


# -- anthropic stream framing -----------------------------------------------------


async def test_anthropic_stream_sequence_ping_signature_and_usage():
    events = [ThoughtEvent("why"), TextEvent("Hi "), TextEvent("there"), DoneEvent(output_tokens=5)]
    parsed = parse_sse_events(await collect(_stream_anthropic(aiter(events), "m", 42)))
    names = [n for n, _ in parsed]
    assert names[:2] == ["message_start", "ping"]
    assert names[-2:] == ["message_delta", "message_stop"]
    assert names.count("content_block_start") == names.count("content_block_stop") == 2
    assert parsed[0][1]["message"]["usage"]["input_tokens"] == 42
    sig = [d for n, d in parsed if n == "content_block_delta" and d["delta"]["type"] == "signature_delta"]
    assert len(sig) == 1 and sig[0]["delta"]["signature"]
    # signature precedes the thinking block's stop.
    i_sig = parsed.index(next(p for p in parsed if p[1] is sig[0]))
    assert parsed[i_sig + 1][0] == "content_block_stop"
    delta = next(d for n, d in parsed if n == "message_delta")
    assert delta["usage"]["output_tokens"] == 5


async def test_anthropic_empty_response_has_empty_text_block_and_nonzero_usage():
    parsed = parse_sse_events(await collect(_stream_anthropic(aiter([DoneEvent()]), "m")))
    starts = [d for n, d in parsed if n == "content_block_start"]
    assert len(starts) == 1 and starts[0]["content_block"] == {"type": "text", "text": ""}
    assert next(d for n, d in parsed if n == "message_delta")["usage"]["output_tokens"] >= 1


async def test_anthropic_missing_done_is_an_error_frame():
    parsed = parse_sse_events(await collect(_stream_anthropic(aiter([TextEvent("partial")]), "m")))
    names = [n for n, _ in parsed]
    assert names[-1] == "error" and "message_stop" not in names
    err = parsed[-1][1]["error"]
    assert err["type"] == "api_error" and "Error" not in err["type"].replace("api_error", "")


async def test_anthropic_midstream_error_closes_block_then_error():
    events = [TextEvent("a"), ErrorEvent(UpstreamError("boom"))]
    parsed = parse_sse_events(await collect(_stream_anthropic(aiter(events), "m")))
    names = [n for n, _ in parsed]
    assert names[-2:] == ["content_block_stop", "error"]
    assert parsed[-1][1] == {"type": "error", "error": {"type": "api_error", "message": "boom"}}


# -- openai stream framing --------------------------------------------------------


async def test_openai_error_is_never_content_and_ends_with_done():
    raw = await collect(_stream_openai(aiter([TextEvent("hi"), ErrorEvent(UpstreamError("boom"))]), "m"))
    assert raw.strip().endswith("data: [DONE]")
    frames = parse_sse(raw)
    assert frames[-1] == {
        "error": {"message": "boom", "type": "server_error", "param": None, "code": None}
    }
    content = "".join(
        f["choices"][0]["delta"].get("content", "") for f in frames if "choices" in f
    )
    assert content == "hi"
    assert not any(f.get("choices", [{}])[0].get("finish_reason") for f in frames if "choices" in f)


async def test_openai_missing_done_is_an_error_not_a_clean_stop():
    raw = await collect(_stream_openai(aiter([TextEvent("partial")]), "m"))
    frames = parse_sse(raw)
    assert "error" in frames[-1] and raw.strip().endswith("[DONE]")
    assert all(f["choices"][0]["finish_reason"] is None for f in frames if "choices" in f)


async def test_openai_include_usage_final_chunk():
    events = [TextEvent("abcdefgh"), DoneEvent(input_tokens=10, output_tokens=2)]
    frames = parse_sse(await collect(_stream_openai(aiter(events), "m", include_usage=True)))
    assert frames[-1]["choices"] == []
    assert frames[-1]["usage"] == {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
    assert all(f["usage"] is None for f in frames[:-1])
    plain = parse_sse(await collect(_stream_openai(aiter(list(events)), "m")))
    assert all("usage" not in f for f in plain)


# -- HTTP level: start-up failures, usage, shapes -------------------------------


@pytest.mark.parametrize("path, body", ROUTES)
@pytest.mark.parametrize("stream", [False, True])
def test_startup_failure_gets_a_real_status(mk, path, body, stream):
    c = mk(ScriptedRunner(exc=RuntimeError("harness would not start")))
    r = c.post(path, json={**body, "stream": stream}, headers=H)
    assert r.status_code == 502
    assert "text/event-stream" not in r.headers["content-type"]
    assert "RuntimeError" not in r.text


@pytest.mark.parametrize("path, body", ROUTES)
def test_empty_stream_before_any_event_is_502(mk, path, body):
    c = mk(ScriptedRunner(events=[]))
    assert c.post(path, json={**body, "stream": True}, headers=H).status_code == 502


@pytest.mark.parametrize("path, body", ROUTES)
def test_quota_exhausted_before_output_is_429(mk, path, body):
    agent = StubAgent(StubResp([], StopReason.QUOTA_EXHAUSTED))
    r = mk(ScriptedRunner(agent=agent)).post(path, json={**body, "stream": True}, headers=H)
    assert r.status_code == 429


def test_usage_is_real_when_reported_and_estimated_otherwise(mk):
    c = mk(ScriptedRunner(events=OK))
    j = c.post("/v1/chat/completions", json=OAI, headers=H).json()
    assert j["usage"] == {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12}
    j = c.post("/v1/messages", json=ANT, headers=H).json()
    assert j["usage"] == {"input_tokens": 9, "output_tokens": 3}
    c2 = mk(ScriptedRunner(events=[TextEvent("x" * 400), DoneEvent()]))
    long = {**OAI, "messages": [{"role": "user", "content": "y" * 800}]}
    j = c2.post("/v1/chat/completions", json=long, headers=H).json()
    assert j["usage"]["prompt_tokens"] >= 200 and j["usage"]["completion_tokens"] >= 100
    long_a = {**ANT, "messages": [{"role": "user", "content": "y" * 800}]}
    j = c2.post("/v1/messages", json=long_a, headers=H).json()
    assert j["usage"]["input_tokens"] >= 200 and j["usage"]["output_tokens"] >= 100


def test_anthropic_stream_message_start_has_estimate_and_delta_has_real(mk):
    c = mk(ScriptedRunner(events=OK))
    body = {**ANT, "stream": True, "messages": [{"role": "user", "content": "z" * 400}]}
    parsed = parse_sse_events(c.post("/v1/messages", json=body, headers=H).text)
    assert parsed[0][1]["message"]["usage"]["input_tokens"] >= 100
    assert next(d for n, d in parsed if n == "message_delta")["usage"]["output_tokens"] == 3


def test_openai_empty_response_content_is_empty_string(mk):
    c = mk(ScriptedRunner(events=[DoneEvent()]))
    j = c.post("/v1/chat/completions", json=OAI, headers=H).json()
    assert j["choices"][0]["message"]["content"] == ""
    j = c.post("/v1/messages", json=ANT, headers=H).json()
    assert j["content"] == [{"type": "text", "text": ""}]


def test_non_stream_thinking_block_is_signed(mk):
    c = mk(ScriptedRunner(events=[ThoughtEvent("t"), TextEvent("a"), DoneEvent()]))
    body = {**ANT, "thinking": {"type": "enabled", "budget_tokens": 1024}}
    blocks = c.post("/v1/messages", json=body, headers=H).json()["content"]
    assert [b["type"] for b in blocks] == ["thinking", "text"]
    assert blocks[0]["signature"]


def test_stream_options_include_usage_over_http(mk):
    c = mk(ScriptedRunner(events=OK))
    body = {**OAI, "stream": True, "stream_options": {"include_usage": True}}
    frames = parse_sse(c.post("/v1/chat/completions", json=body, headers=H).text)
    assert frames[-1]["usage"]["total_tokens"] == 12


# -- max_tokens and routing --------------------------------------------------------


def test_max_tokens_is_honoured(mk):
    r = ScriptedRunner(events=[TextEvent("a" * 200), TextEvent("b" * 200), DoneEvent()])
    c = mk(r)
    j = c.post("/v1/messages", json={**ANT, "max_tokens": 70}, headers=H).json()
    assert j["stop_reason"] == "max_tokens"
    assert j["content"][0]["text"] == "a" * 200 + "b" * 80  # 70 tokens ~ 280 chars
    j = c.post("/v1/chat/completions", json={**OAI, "max_tokens": 70}, headers=H).json()
    assert j["choices"][0]["finish_reason"] == "length"


def test_max_tokens_truncates_text(mk):
    c = mk(ScriptedRunner(events=[TextEvent("a" * 500), DoneEvent()]))
    j = c.post("/v1/messages", json={**ANT, "max_tokens": 65}, headers=H).json()
    assert j["stop_reason"] == "max_tokens" and len(j["content"][0]["text"]) == 260


@pytest.mark.parametrize(
    "model, max_tokens, expected",
    [
        ("claude-3-5-haiku-20241022", 4096, "agy-light"),
        ("claude-sonnet-4", 1, "agy-light"),
        ("claude-sonnet-4", 64, "agy-light"),
        (None, 10, "agy-researcher"),  # no model: default preset, tiny or not
        ("agy-researcher", 10, "agy-researcher"),  # explicit preset name wins
        (None, 1000, "agy-researcher"),
    ],
)
def test_preset_routing(mk, model, max_tokens, expected):
    r = ScriptedRunner(events=OK)
    c = mk(r)
    body = {"max_tokens": max_tokens, "messages": ANT["messages"]}
    if model:
        body["model"] = model
    assert c.post("/v1/messages", json=body, headers=H).status_code == 200
    assert r.last[0].name == expected


def test_unknown_model_with_normal_max_tokens_is_404(mk):
    r = ScriptedRunner(events=OK)
    resp = mk(r).post("/v1/messages", json={**ANT, "model": "gpt-4o"}, headers=H)
    assert resp.status_code == 404 and r.calls == 0


# -- request normalization -------------------------------------------------------


def test_n_other_than_one_is_400(mk):
    r = ScriptedRunner(events=OK)
    c = mk(r)
    resp = c.post("/v1/chat/completions", json={**OAI, "n": 2}, headers=H)
    assert resp.status_code == 400 and resp.json()["error"]["param"] == "n"
    assert c.post("/v1/chat/completions", json={**OAI, "n": 1}, headers=H).status_code == 200
    assert r.calls == 1


@pytest.mark.parametrize(
    "path, messages",
    [
        ("/v1/chat/completions", []),
        ("/v1/chat/completions", [{"role": "user", "content": ""}]),
        ("/v1/chat/completions", [{"role": "user", "content": "   \n "}]),
        ("/v1/chat/completions", [{"role": "system", "content": "only system"}]),
        ("/v1/chat/completions", [{"role": "user", "content": None}]),
        ("/v1/messages", []),
        ("/v1/messages", [{"role": "user", "content": []}]),
        ("/v1/messages", [{"role": "user", "content": [{"type": "text", "text": " "}]}]),
    ],
)
def test_empty_prompt_is_400_before_any_spawn(mk, path, messages):
    r = ScriptedRunner(events=OK)
    body = {"model": "agy-researcher", "max_tokens": 100, "messages": messages}
    resp = mk(r).post(path, json=body, headers=H)
    assert resp.status_code == 400
    assert r.calls == 0


@pytest.mark.parametrize(
    "path, messages",
    [
        ("/v1/chat/completions", [{"role": "narrator", "content": "x"}]),
        ("/v1/messages", [{"role": "system", "content": "x"}]),
        ("/v1/messages", [{"role": "user", "content": [{"type": "mystery"}]}]),
    ],
)
def test_unsupported_roles_and_blocks_are_400(mk, path, messages):
    r = ScriptedRunner(events=OK)
    body = {"model": "agy-researcher", "max_tokens": 100, "messages": messages}
    assert mk(r).post(path, json=body, headers=H).status_code == 400
    assert r.calls == 0


def test_bad_max_tokens_is_400(mk):
    c = mk(ScriptedRunner(events=OK))
    assert c.post("/v1/messages", json={**ANT, "max_tokens": 0}, headers=H).status_code == 400
    assert c.post("/v1/chat/completions", json={**OAI, "max_tokens": -1}, headers=H).status_code == 400


def _prompt_for(c, runner, path, body):
    assert c.post(path, json=body, headers=H).status_code == 200
    return runner.last[1]


def test_openai_roles_and_joins(mk):
    r = ScriptedRunner(events=OK)
    c = mk(r)
    body = {
        **OAI,
        "messages": [
            {"role": "developer", "content": "dev rules"},
            {"role": "system", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "result text"},
            {"role": "user", "content": "next"},
        ],
    }
    msgs = _prompt_for(c, r, "/v1/chat/completions", body)
    assert [m["role"] for m in msgs] == ["system", "system", "user", "assistant", "tool", "user"]
    assert msgs[1]["content"] == "a\nb"  # blocks joined with newline, not glued
    assert "lookup" in msgs[3]["content"]
    assert "[tool result c1]" in msgs[4]["content"] and "result text" in msgs[4]["content"]


def test_anthropic_blocks_render_as_text(mk):
    r = ScriptedRunner(events=OK)
    c = mk(r)
    body = {
        **ANT,
        "system": [{"type": "text", "text": "s1"}, {"type": "text", "text": "s2"}],
        "tools": [{"name": "t", "input_schema": {}}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "private", "signature": "s"},
                    {"type": "tool_use", "id": "tu1", "name": "Read", "input": {"p": "x"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu1",
                        "content": [{"type": "text", "text": "file body"}],
                        "is_error": True,
                    }
                ],
            },
        ],
    }
    msgs = _prompt_for(c, r, "/v1/messages", body)
    assert msgs[0] == {"role": "system", "content": "s1\ns2"}
    assert msgs[1]["content"] == "a\nb"
    assert "private" not in msgs[2]["content"]
    assert "[tool_use Read id=tu1]" in msgs[2]["content"] and '"p": "x"' in msgs[2]["content"]
    assert "[tool_result id=tu1 error]" in msgs[3]["content"] and "file body" in msgs[3]["content"]


# -- count_tokens and models --------------------------------------------------------


def test_count_tokens(mk):
    r = ScriptedRunner(events=OK)
    c = mk(r)
    body = {"model": "agy-researcher", "messages": [{"role": "user", "content": "x" * 400}]}
    resp = c.post("/v1/messages/count_tokens", json=body, headers=H)
    assert resp.status_code == 200
    assert 90 <= resp.json()["input_tokens"] <= 110
    with_sys = c.post("/v1/messages/count_tokens", json={**body, "system": "s" * 400}, headers=H)
    assert with_sys.json()["input_tokens"] > resp.json()["input_tokens"] + 90
    assert r.calls == 0  # never spawns an agent
    assert c.post("/v1/messages/count_tokens", json={"messages": []}, headers=H).status_code == 400
    assert c.post("/v1/messages/count_tokens", json=body).status_code == 401


def test_models_openai_and_anthropic_shapes(mk):
    c = mk(ScriptedRunner())
    oai = c.get("/v1/models", headers={"authorization": f"Bearer {TOKEN}"}).json()
    assert oai["object"] == "list" and oai["data"][0]["object"] == "model"
    for hdrs in ({"anthropic-version": "2023-06-01"}, {"x-api-key": TOKEN}):
        auth = hdrs if "x-api-key" in hdrs else {**hdrs, "authorization": f"Bearer {TOKEN}"}
        ant = c.get("/v1/models", headers=auth).json()
        assert ant["has_more"] is False and ant["data"][0]["type"] == "model"
        assert ant["first_id"] == ant["data"][0]["id"] and ant["data"][0]["display_name"]


def test_get_model_by_id(mk):
    c = mk(ScriptedRunner())
    auth = {"authorization": f"Bearer {TOKEN}"}
    one = c.get("/v1/models/agy-light", headers=auth)
    assert one.status_code == 200 and one.json()["id"] == "agy-light"
    assert one.json()["object"] == "model"
    ant = c.get("/v1/models/agy-light", headers={**auth, "anthropic-version": "2023-06-01"})
    assert ant.json()["type"] == "model"
    missing = c.get("/v1/models/nope", headers=auth)
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "model_not_found"
    missing_a = c.get("/v1/models/nope", headers={**auth, "anthropic-version": "2023-06-01"})
    assert missing_a.status_code == 404 and missing_a.json()["type"] == "error"


def test_sse_is_valid_json_lines(mk):
    c = mk(ScriptedRunner(events=OK))
    raw = c.post("/v1/messages", json={**ANT, "stream": True}, headers=H).text
    for _, data in parse_sse_events(raw):
        json.dumps(data)
