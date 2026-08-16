from conftest import aiter, collect, parse_sse_events

from agy_proxy.anthropic_api import _collect_anthropic, _stream_anthropic
from agy_proxy.events import DoneEvent, ErrorEvent, TextEvent, ThoughtEvent, ToolCallEvent


async def test_stream_event_sequence():
    events = [
        ThoughtEvent("reason"),
        TextEvent("Hi "),
        TextEvent("there"),
        DoneEvent(stop_reason="UNSPECIFIED", output_tokens=3),
    ]
    parsed = parse_sse_events(await collect(_stream_anthropic(aiter(events), "agy-coder")))
    names = [name for name, _ in parsed]

    # Required framing.
    assert names[0] == "message_start"
    assert names[-1] == "message_stop"
    assert "message_delta" in names
    assert names[-2] == "message_delta"

    # A thinking block then a text block, each opened and closed.
    starts = [d for n, d in parsed if n == "content_block_start"]
    assert starts[0]["content_block"]["type"] == "thinking"
    assert starts[1]["content_block"]["type"] == "text"

    # content_block_start/stop are balanced.
    assert names.count("content_block_start") == names.count("content_block_stop")

    # Text reassembles.
    text = "".join(
        d["delta"]["text"]
        for n, d in parsed
        if n == "content_block_delta" and d["delta"]["type"] == "text_delta"
    )
    assert text == "Hi there"

    # message_start usage present; message_delta carries stop_reason.
    msg_start = next(d for n, d in parsed if n == "message_start")
    assert msg_start["message"]["role"] == "assistant"
    msg_delta = next(d for n, d in parsed if n == "message_delta")
    assert msg_delta["delta"]["stop_reason"] == "end_turn"


async def test_stream_indices_monotonic():
    events = [ThoughtEvent("a"), TextEvent("b"), ToolCallEvent("VIEW_FILE", {"p": 1}), DoneEvent()]
    parsed = parse_sse_events(await collect(_stream_anthropic(aiter(events), "m")))
    start_indices = [d["index"] for n, d in parsed if n == "content_block_start"]
    assert start_indices == [0, 1, 2]
    # tool_use block carries name + input json delta.
    tool_start = next(d for n, d in parsed if n == "content_block_start" and d["content_block"]["type"] == "tool_use")
    assert tool_start["content_block"]["name"] == "VIEW_FILE"


async def test_max_tokens_stop_reason():
    parsed = parse_sse_events(
        await collect(_stream_anthropic(aiter([TextEvent("x"), DoneEvent(stop_reason="MAX_TOTAL_TOKENS_EXCEEDED")]), "m"))
    )
    msg_delta = next(d for n, d in parsed if n == "message_delta")
    assert msg_delta["delta"]["stop_reason"] == "max_tokens"


async def test_error_emits_error_event():
    parsed = parse_sse_events(await collect(_stream_anthropic(aiter([ErrorEvent("boom", "ValueError")]), "m")))
    assert any(n == "error" for n, _ in parsed)
    err = next(d for n, d in parsed if n == "error")
    assert err["error"]["message"] == "boom"


async def test_non_stream_collect_blocks():
    events = [ThoughtEvent("t"), TextEvent("hello"), ToolCallEvent("SEARCH_WEB", {"q": "x"}), DoneEvent()]
    body = await _collect_anthropic(aiter(events), "agy-fast")
    assert body["type"] == "message"
    types = [b["type"] for b in body["content"]]
    assert types == ["thinking", "text", "tool_use"]
    assert body["content"][2]["input"] == {"q": "x"}
    assert body["stop_reason"] == "end_turn"
