from conftest import aiter, collect, parse_sse

from agy_proxy.events import DoneEvent, ErrorEvent, TextEvent, ThoughtEvent, ToolCallEvent
from agy_proxy.openai_api import _collect_openai, _stream_openai


async def test_stream_shape_and_order():
    events = [
        ThoughtEvent("thinking..."),
        TextEvent("Hello"),
        TextEvent(" world"),
        DoneEvent(stop_reason="UNSPECIFIED", input_tokens=5, output_tokens=2),
    ]
    raw = await collect(_stream_openai(aiter(events), "agy-coder"))
    chunks = parse_sse(raw)

    # Every chunk carries required OpenAI fields.
    for c in chunks:
        assert c["object"] == "chat.completion.chunk"
        assert c["id"].startswith("chatcmpl-")
        assert c["model"] == "agy-coder"
        assert "created" in c
        assert c["choices"][0]["index"] == 0

    # First delta announces the role.
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    # Reasoning surfaced separately from content.
    assert any(c["choices"][0]["delta"].get("reasoning_content") == "thinking..." for c in chunks)
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text == "Hello world"
    # Terminates correctly.
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert raw.strip().endswith("data: [DONE]")


async def test_stream_tool_call_delta():
    events = [
        ToolCallEvent(name="VIEW_FILE", args={"path": "a.py"}, id="call_1"),
        DoneEvent(),
    ]
    chunks = parse_sse(await collect(_stream_openai(aiter(events), "m")))
    tc = None
    for c in chunks:
        if c["choices"][0]["delta"].get("tool_calls"):
            tc = c["choices"][0]["delta"]["tool_calls"][0]
    assert tc is not None
    assert tc["index"] == 0
    assert tc["function"]["name"] == "VIEW_FILE"
    assert tc["function"]["arguments"] == '{"path": "a.py"}'


async def test_max_tokens_maps_to_length():
    chunks = parse_sse(
        await collect(_stream_openai(aiter([DoneEvent(stop_reason="MAX_OUTPUT_TOKENS_EXCEEDED")]), "m"))
    )
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"


async def test_error_event_closes_stream():
    raw = await collect(_stream_openai(aiter([TextEvent("hi"), ErrorEvent("boom", "RuntimeError")]), "m"))
    assert "boom" in raw
    assert raw.strip().endswith("data: [DONE]")


async def test_non_stream_collect():
    events = [
        ThoughtEvent("t"),
        TextEvent("answer"),
        ToolCallEvent(name="SEARCH_WEB", args={"q": "x"}),
        DoneEvent(input_tokens=3, output_tokens=4),
    ]
    body = await _collect_openai(aiter(events), "agy-fast")
    assert body["object"] == "chat.completion"
    msg = body["choices"][0]["message"]
    assert msg["content"] == "answer"
    assert msg["reasoning_content"] == "t"
    assert msg["tool_calls"][0]["function"]["name"] == "SEARCH_WEB"
    assert body["usage"]["total_tokens"] == 7
