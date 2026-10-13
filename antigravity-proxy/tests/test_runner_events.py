"""The real SDK-event mapping (``run_chat``) driven by a stub ChatResponse."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from conftest import StubAgent, StubResp
from google.antigravity import BuiltinTools
from google.antigravity.types import StopReason, Text, Thought, ToolCall, ToolResult

from agy_proxy import runner_events
from agy_proxy.errors import RateLimited
from agy_proxy.events import DoneEvent, ErrorEvent, TextEvent, ThoughtEvent
from agy_proxy.runner_events import run_chat


async def _events(agent, prompt="hi"):
    return [e async for e in run_chat(agent, prompt)]


@pytest.fixture(autouse=True)
def _trace_on():
    runner_events.configure(trace_tools=True)
    yield
    runner_events.configure(trace_tools=True)


async def test_text_thought_and_tools_map_without_tool_events(caplog):
    chunks = [
        Thought(text="hmm", step_index=0),
        ToolCall(name=BuiltinTools.VIEW_FILE, args={"path": "a.py"}, id="t1"),
        ToolResult(name=BuiltinTools.VIEW_FILE, id="t1", result="x"),
        Text(text="Hel", step_index=0),
        Text(text="", step_index=0),
        Text(text="lo", step_index=0),
    ]
    usage = SimpleNamespace(prompt_token_count=11, candidates_token_count=7)
    agent = StubAgent(StubResp(chunks, StopReason.UNSPECIFIED, usage))
    with caplog.at_level(logging.INFO, logger="agy_proxy.tools"):
        events = await _events(agent)
    assert events == [
        ThoughtEvent("hmm"),
        TextEvent("Hel"),
        TextEvent("lo"),
        DoneEvent(stop_reason="UNSPECIFIED", input_tokens=11, output_tokens=7),
    ]
    assert agent.prompts == ["hi"]
    assert any("tool call: VIEW_FILE" in r.getMessage() for r in caplog.records)
    assert any("tool result: VIEW_FILE" in r.getMessage() for r in caplog.records)


async def test_tool_trace_can_be_disabled(caplog):
    runner_events.configure(trace_tools=False)
    chunks = [ToolCall(name="x", args={}, id="1"), ToolResult(name="x", id="1", result=1)]
    with caplog.at_level(logging.DEBUG, logger="agy_proxy.tools"):
        events = await _events(StubAgent(StubResp(chunks)))
    assert events == [DoneEvent(stop_reason=None, input_tokens=None, output_tokens=None)]
    assert not [r for r in caplog.records if r.name == "agy_proxy.tools"]


async def test_empty_response_is_just_done():
    events = await _events(StubAgent(StubResp([], StopReason.UNSPECIFIED)))
    assert len(events) == 1 and isinstance(events[0], DoneEvent)


async def test_quota_exhausted_is_rate_limited_error():
    agent = StubAgent(StubResp([Text(text="part", step_index=0)], StopReason.QUOTA_EXHAUSTED))
    events = await _events(agent)
    assert isinstance(events[0], TextEvent)
    assert isinstance(events[-1], ErrorEvent) and isinstance(events[-1].error, RateLimited)
    assert not any(isinstance(e, DoneEvent) for e in events)


@pytest.mark.parametrize(
    "reason", [StopReason.MAX_MODEL_CALLS_EXCEEDED, StopReason.MAX_TOOL_CALLS_EXCEEDED]
)
async def test_budget_cutoffs_are_done_and_logged(reason, caplog):
    with caplog.at_level(logging.WARNING, logger="agy_proxy.runner"):
        events = await _events(StubAgent(StubResp([Text(text="a", step_index=0)], reason)))
    assert isinstance(events[-1], DoneEvent) and events[-1].stop_reason == reason.name
    assert any("cut short" in r.getMessage() for r in caplog.records)


async def test_exception_before_first_chunk_is_error_event():
    events = await _events(StubAgent(chat_exc=RuntimeError("spawn failed")))
    assert len(events) == 1 and isinstance(events[0], ErrorEvent)
    assert "spawn failed" in events[0].message
    assert events[0].error.status == 502


async def test_exception_mid_stream_is_error_event_after_text():
    events = await _events(StubAgent(StubResp([Text(text="a", step_index=0), Text(text="b", step_index=0)], fail_after=1)))
    assert [type(e) for e in events] == [TextEvent, ErrorEvent]
