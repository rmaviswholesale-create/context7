# `google-antigravity` SDK — verified API notes

Captured by direct introspection of the installed wheel (**v0.1.21**; pinned `>=0.1.21,<0.2`).
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
    policies=[policy.allow_all()],            # or confirm_run_command(), deny_all() (see "Policies" below)
    model="gemini-3-pro",                     # str | ModelTarget | None
    api_key="...",                            # your own key; falls back to GEMINI_API_KEY / ADC
    workspaces=["/path"],                     # None becomes [os.getcwd()] -- always pass explicitly
    save_dir="/abs", app_data_dir="/abs",     # None -> mkdtemp never removed; proxy owns them (Scratch)
    env={"PATH": "...", "HOME": "..."},       # MERGED into os.environ, cannot remove variables
    conversation_id=None,
)
```

- `Agent(config)` is an **async context manager**: `async with Agent(config) as agent: ...`
  (spins up the Go `localharness` binary bundled in the wheel). For long-lived stateful use,
  drive `await agent.__aenter__()` / `await agent.__aexit__(None, None, None)` manually.
- Default policy **denies `run_command`**. Pass `policies=[policy.allow_all()]` for shell access.

## Chat + streaming

`chat` is a **coroutine** returning a `ChatResponse`:

```python
resp = await agent.chat(prompt)   # prompt: str | Image | Document | Audio | Video | SlashCommand | Sequence[...]
```

`ChatResponse` is a lazy async multi-cursor over a shared buffer. Cursors:

- `async for delta in resp:` → text tokens (str)
- `resp.chunks` → ordered stream of `Text | Thought | ToolCall`  ← we use this. **`ToolResult` is
  declared in the type union but is never produced by `Conversation.receive_chunks`** (it only yields
  Thought/Text/ToolCall); tool results are not available on this stream.
- `resp.thoughts` → `Thought` cursor
- `resp.tool_calls` → `ToolCall` cursor
- `await resp.text()` → full text
- `await resp.resolve()` → `list[StreamChunk | ToolCall | ToolResult]`
- `resp.stop_reason` / `resp.usage_metadata` (available after consumption)
- `resp.cancel()` is a coroutine; no-op when the turn is done; sends a halt request

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

The SDK executes builtin tools **internally** (file I/O, shell, search, subagents). `ToolCall`
chunks are informational: the proxy logs them (`AGY_LOG_TOOLS`) and never sends them to clients,
who cannot fulfil them.

## Policies and tool exposure (verified in `hooks/policy.py`, `types.py`)

- `confirm_run_command()` **without a handler** = `[deny(run_command), allow("*")]`: it denies shell
  but **allows file writes** (`create_file`, `edit_file`) and everything else. It is NOT a read-only
  policy. For read-only presets the `enabled_tools` allowlist is the real barrier (the proxy also
  prepends explicit `deny` rules for the write tools).
- `workspace_only(ws)` does `del workspaces` and returns `deny` for **all** file tools. It does not
  restrict paths to the workspace; the proxy rejects it as a preset policy.
- `enabled_tools=None` means `BuiltinTools.default()` (everything except ASK_QUESTION and the
  LIST_DIR/SEARCH_DIR/FIND_FILE trio), which includes RUN_COMMAND, CREATE/EDIT_FILE, START_SUBAGENT,
  SEARCH_WEB, READ_URL_CONTENT, SCHEDULE, GENERATE_IMAGE. `enabled_tools=[]` resolves to an empty set
  (tool-less agent). `enabled_tools` and `disabled_tools` are mutually exclusive.
- `allow_all()` + `workspaces` is **not** proven to confine anything: nothing in the Python SDK ties
  `run_command` or file tools to `workspaces`. Treat write-capable presets as having the full
  privileges of the proxy user (run in a container / unprivileged user).

## Verification log (P0)

| ID | Question | Result |
|----|----------|--------|
| U1 | `confirm_run_command` headless | Confirmed (static): no handler -> deny `run_command`, allow `*` (incl. writes). |
| U2 | Does `workspaces` confine `allow_all`? | Not provable statically; no Python-side enforcement. README says not to rely on it. **Open** (needs a harness run). |
| U6 | Empty `app_data_dir` effects | Default is `~/.gemini/antigravity`; Python code only forwards it to the harness. With an API key passed in config nothing in Python reads credentials from it. Go-side use (cache/creds) **unverified**. |
| U7 | HOME override vs Vertex ADC | `env` is merged over `os.environ` and forwarded in `InputConfig.env`. ADC reads `$HOME/.config/gcloud`, so the proxy keeps the real HOME when no API key is configured and overrides HOME only with an API key. GOOGLE_* is never scrubbed. Harness behaviour **unverified**. |
| U8 | Termination with allowlist / no subagents | `FINISH` is only the structured-output tool (`get_last_structured_output`); nothing in Python requires it to end a turn, so read-only presets omit it. `CapabilitiesConfig` validators only complain about `max_subagent_depth` / `allowed_subagents` with subagents off. Runtime termination **unverified** (needs a real call). |
| - | `env=` semantics | Merges (`{**os.environ, **env}`), cannot unset: scrub the process env at startup. |
| - | `workspaces=None` | Becomes `[os.getcwd()]`: always pass explicit workspaces. |

## Verification log (P2: runtime / lifecycle)

| ID | Question | Result |
|----|----------|--------|
| U3 | Harness pid path | `agent.conversation.connection._process` is the `subprocess.Popen` (SDK 0.1.21, private). Confined to `agy_proxy/lifecycle.py`; pyproject pins `google-antigravity>=0.1.21,<0.2`. Missing path degrades to "no process known". Verified against the real binary. |
| U4 | `cancel()` | `ChatResponse.cancel()` -> `Conversation.cancel()` -> `halt_request`; it halts the *turn*, not the process, and is a no-op once the response is done. After a halt `_client_cancelled` stays set until the next `send`, and `send` first drains `receive_steps()` (which raises `AntigravityCancelledError`) -> session agents whose turn did not finish are evicted, never reused. The runner halts via `agent.conversation.cancel()` (same call). |
| U5 | Blocking disconnect | Confirmed: `LocalConnection.disconnect()` ends in a synchronous `Popen.wait(180)` (then terminate/kill) on the event-loop thread, so `asyncio.wait_for` cannot interrupt it. Handled by a thread watchdog (SIGTERM at `AGY_TEARDOWN_TIMEOUT`, SIGKILL 3 s later) plus `wait_for` for the async part. Tested with a fake harness that ignores stdin EOF / SIGTERM. A healthy real harness closes in ~20 ms. |
| U5b | Blocking startup | `Agent.__aenter__` spawns the harness with sync `Popen`/`stdout.read`; measured loop stall with the real binary was ~0.16 s of a ~5 s open (the rest is async retry/backoff). Cancelling during open is made safe by running `__aenter__` as a shielded task and closing afterwards. A harness spawned by an `__aenter__` that is *force*-cancelled (open exceeded teardown timeout) cannot be located and could be orphaned: **open**, bounded by the harness exiting on stdin EOF. |
| U9 | `stop_reason` / usage | Both are read lazily from the connection (`last_turn_usage`, `_last_turn_stop_reason`) and are only meaningful after the chunk stream is exhausted; `_run` reads them at that point. Budgets are lifetime-scoped (`BudgetScope.LIFETIME`) -> session agents get 20x the per-request budget. |
