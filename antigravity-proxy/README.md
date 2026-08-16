# agy-proxy — a free Antigravity 2.0 custom-agent proxy

Drive **Google Antigravity's** agent loop (Gemini 3, file I/O, shell, search,
sub-agents — all executed by the official [`google-antigravity`](https://pypi.org/project/google-antigravity/)
SDK) from **any OpenAI- or Anthropic-compatible client**: Claude Code, Cursor,
Zed, aider, Continue, and friends.

Same idea as `claude-code-router` / LiteLLM, pointed the other way: your favourite
client stays the same, and the agent behind it becomes Antigravity, configured
with **your own presets** and authenticated with **your own API key**.

```
Claude Code / Cursor / aider ──▶  agy-proxy  ──▶  google.antigravity.Agent ──▶ Gemini
   (OpenAI or Anthropic API)     (localhost)         (your key, free tier)
```

## Why this design

- **Legitimate & local.** It uses the sanctioned SDK with *your own*
  `GEMINI_API_KEY` (free tier works) or Vertex ADC. No credential theft, no
  reverse-engineering of the IDE's private endpoints.
- **Both wire protocols.** `/v1/chat/completions` (OpenAI) **and** `/v1/messages`
  (Anthropic — what Claude Code speaks).
- **Your "personal preference of agents".** Presets in `presets.yaml` map a
  requested model name to a full agent profile (system prompt, model, tool set,
  autonomy policy, thinking depth).
- **Auto conversation state.** Stateless by default (rebuilds context from the
  messages array, like every OpenAI/Anthropic client expects); switches to a
  live, memory-preserving Agent when a client sends a session id.
- **Faithful streaming.** Reasoning is surfaced (`reasoning_content` / `thinking`
  blocks) and SDK tool calls are streamed, with spec-correct SSE framing.

## Install

Requires Python 3.11+.

```bash
cd antigravity-proxy
python -m venv .venv && source .venv/bin/activate
pip install -e .            # add ".[dev]" for tests
```

## Configure

Get a free key at <https://aistudio.google.com/apikey>, then:

```bash
cp .env.example .env        # put GEMINI_API_KEY=... in it
export $(grep -v '^#' .env | xargs)   # or use any dotenv loader
python -m agy_proxy.doctor  # verify the installed SDK matches expectations
```

## Run

```bash
python -m agy_proxy          # serves http://127.0.0.1:8081
```

Sanity check:

```bash
curl -s localhost:8081/healthz
curl -s localhost:8081/v1/models        # lists your presets
```

## Wire up your client

### Claude Code (Anthropic `/v1/messages`)

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8081
export ANTHROPIC_API_KEY=dummy          # any non-empty string; the proxy uses YOUR Gemini key
claude --model agy-coder                 # the model name selects a preset
```

### OpenAI-compatible tools (Cursor, aider, Zed, …)

```
Base URL:  http://127.0.0.1:8081/v1
API key:   dummy   (unless you set AGY_PROXY_TOKEN — then use that)
Model:     agy-coder | agy-researcher | agy-fast
```

```bash
curl -N localhost:8081/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"agy-coder","stream":true,
       "messages":[{"role":"user","content":"list the files here"}]}'
```

## Presets — your agents

Edit `presets.yaml`. The `model` field a client sends is matched against preset
names; unknown names fall back to `default`.

```yaml
default: agy-coder
presets:
  agy-coder:                      # autonomous, full tools, deep thinking
    policy: allow_all
    thinking_level: high
  agy-researcher:                 # read/search only, no edits/shell
    policy: safe_defaults
    disabled_tools: [CREATE_FILE, EDIT_FILE, RUN_COMMAND]
  agy-fast:                       # snappy, minimal thinking
    thinking_level: minimal
    enable_subagents: false
```

Fields: `system_instructions`, `model`, `policy`
(`allow_all` | `safe_defaults` | `workspace_only` | `deny_all` | `confirm_run_command`),
`behavior` (`autonomous` | `interactive`), `thinking_level`
(`minimal`…`extra_high`), `enabled_tools` / `disabled_tools` (BuiltinTools names),
`enable_subagents`, `workspaces`.

> `agy-proxy` runs headless, so `safe_defaults` maps to "allow all tools but guard
> shell execution" (there is no human to answer confirm prompts).

## Conversation state (auto)

- **No session id →** stateless: the whole `messages` array is replayed into a
  fresh agent each call. Works with every client out of the box.
- **Session id present →** stateful: a live Agent is kept per session (LRU + TTL
  bounded, see `AGY_SESSION_*`), so follow-ups reuse the SDK's own memory and skip
  harness startup. Send it via the `x-agy-session` header, or the OpenAI `user`
  field.

## Configuration reference

| Env var | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` / `AGY_API_KEY` | – | Your key handed to the SDK |
| `AGY_HOST` / `AGY_PORT` | `127.0.0.1` / `8081` | Bind address |
| `AGY_PROXY_TOKEN` | – | If set, clients must send it (Bearer / `x-api-key`) |
| `AGY_DEFAULT_MODEL` | `gemini-3-pro` | Model when a preset pins none |
| `AGY_PRESETS` | `./presets.yaml` | Presets file path |
| `AGY_SESSION_TTL` / `AGY_SESSION_MAX` | `900` / `64` | Live-agent cache bounds |

## Endpoints

- `POST /v1/chat/completions` — OpenAI (stream + non-stream)
- `POST /v1/messages` — Anthropic (stream + non-stream)
- `GET /v1/models` — presets as models
- `GET /healthz` — status

## Notes & limits

- Builtin tools (file I/O, shell, search, sub-agents) run **inside** the SDK.
  The proxy surfaces `ToolCall`/`ToolResult` as informational stream events —
  clients don't fulfil them.
- Text-only for now; multimodal inputs are on the roadmap.
- Keep the bind on `127.0.0.1` unless you set `AGY_PROXY_TOKEN`.
- This is personal tooling using your own account. Respect Google's and any
  model provider's terms of service.

## Tests

```bash
pip install -e ".[dev]"
python -m agy_proxy.doctor   # SDK surface check
pytest -q                    # adapters, presets, sessions, flattening
ruff check .
```
