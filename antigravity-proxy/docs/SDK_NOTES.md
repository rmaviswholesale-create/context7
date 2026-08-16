# `google-antigravity` SDK — verified API notes

Captured by direct introspection of the installed wheel (**v0.1.x**, `pip install google-antigravity`).
These are the exact surfaces `agy-proxy` depends on. Re-verify with
`python -m agy_proxy.doctor` (or the snippet at the bottom) if the SDK is upgraded.

## Construction

```python
from google.antigravity import Agent, LocalAgentConfig, CapabilitiesConfig, BuiltinTools
from google.antigravity import policy
from google.antigravity.types import AgentBehavior

config = LocalAgentConfig(
    system_instructions="You are ...",        # str | CustomSystemInstructions | TemplatedSystemInstructions | None
    capabilities=CapabilitiesConfig(
        agent_behavior=AgentBehavior.AUTONOMOUS,   # or INTERACTIVE
        enabled_tools=[BuiltinTools.VIEW_FILE],    # None = all
        disabled_tools=None,
        enable_subagents=True,
    ),
    policies=[policy.allow_all()],            # or [policy.safe_defaults()], [policy.workspace_only()], ...
    model="gemini-3-pro",                     # str | ModelTarget | None
    api_key="...",                            # your own key; falls back to GEMINI_API_KEY / ADC
    workspaces=["/path"],                     # optional; restricts file tools
    conversation_id=None,
)
```

- `Agent(config)` is an **async context manager**: `async with Agent(config) as agent: ...`
  (spins up the Go `localharness` binary bundled in the wheel). For long-lived stateful use,
  drive `await agent.__aenter__()` / `await agent.__aexit__(None, None, None)` manually.
- Default policy **denies `run_command`** (confirm-required). Pass `policies=[policy.allow_all()]`
  for fully autonomous shell access.

## Chat + streaming

`chat` is a **coroutine** returning a `ChatResponse`:

```python
resp = await agent.chat(prompt)   # prompt: str | Image | Document | Audio | Video | SlashCommand | Sequence[...]
```

`ChatResponse` is a lazy async multi-cursor over a shared buffer. Cursors:

- `async for delta in resp:` → text tokens (str)
- `resp.chunks` → **ordered** stream of `Text | Thought | ToolCall | ToolResult`  ← we use this
- `resp.thoughts` → `Thought` cursor
- `resp.tool_calls` → `ToolCall` cursor
- `await resp.text()` → full text
- `await resp.resolve()` → `list[StreamChunk | ToolCall | ToolResult]`
- `resp.stop_reason` / `resp.usage_metadata` (available after consumption)
- `resp.cancel()`

### Chunk shapes (all pydantic models)

| Type         | Fields |
|--------------|--------|
| `Text`       | `step_index: int`, `text: str` |
| `Thought`    | `step_index: int`, `text: str`, `signature: bytes \| None` |
| `ToolCall`   | `name: BuiltinTools \| str`, `args: dict`, `id: str \| None`, `canonical_path`, `server_name` |
| `ToolResult` | `name`, `id`, `result: Any`, `error: str \| None`, `exception`, `server_name` |

`BuiltinTools`: LIST_DIR, SEARCH_DIR, FIND_FILE, VIEW_FILE, CREATE_FILE, EDIT_FILE, RUN_COMMAND,
ASK_QUESTION, START_SUBAGENT, GENERATE_IMAGE, SEARCH_WEB, READ_URL_CONTENT, FINISH.

`UsageMetadata`: `prompt_token_count`, `candidates_token_count`, `thoughts_token_count`,
`cached_content_token_count`, `total_token_count`, `service_tier`.

`StopReason`: UNSPECIFIED, MAX_MODEL_CALLS_EXCEEDED, MAX_TOOL_CALLS_EXCEEDED,
MAX_INPUT_TOKENS_EXCEEDED, MAX_OUTPUT_TOKENS_EXCEEDED, MAX_TOTAL_TOKENS_EXCEEDED, QUOTA_EXHAUSTED.

## Thinking level (optional)

```python
from google.antigravity.models import ModelTarget, GeminiAPIEndpoint, GeminiModelOptions, ThinkingLevel
model = ModelTarget(
    name="gemini-3-pro",
    endpoint=GeminiAPIEndpoint(api_key=..., options=GeminiModelOptions(thinking_level=ThinkingLevel.HIGH)),
)
```

## Auth

`api_key` on the config, else env `GEMINI_API_KEY`, else Application Default Credentials
(`gcloud auth application-default login`) when `vertex=True` with `project`/`location`.

## Note on tools

The SDK executes builtin tools **internally** (file I/O, shell, search, subagents). The proxy
surfaces `ToolCall`/`ToolResult` as informational stream events — clients do **not** fulfil them.
