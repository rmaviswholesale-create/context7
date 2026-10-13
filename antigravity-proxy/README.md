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
  blocks) with spec-correct SSE framing. The agent runs its own tools server-side;
  its tool calls are logged, not sent to the client.

## Install

Requires Python 3.11+.

```bash
cd antigravity-proxy
python -m venv .venv && source .venv/bin/activate
pip install -e .            # add ".[dev]" for tests
```

## Security (read this first)

`agy-proxy` hands requests to an agent that **acts with the privileges of the
user running the proxy, on the machine running the proxy**. Whoever can send it a
request can make that agent read files, and (with write presets) change files and
run commands. Treat it like an SSH login.

- **A token is always required.** If `AGY_PROXY_TOKEN` is unset, a random token is
  generated at startup and printed once to stderr (loopback binds only). Clients
  send it as `Authorization: Bearer <token>` or `x-api-key: <token>`. Only
  `/healthz` (liveness only) is unauthenticated; `/docs` and `/openapi.json` are off.
- **Localhost only.** Binding to a non-loopback address without an explicit
  `AGY_PROXY_TOKEN` exits at startup. A token does **not** make a LAN/Internet bind
  safe: it travels in clear text and the agent has host privileges. If you must
  expose it, put it behind TLS and a firewall/VPN, set `AGY_ALLOWED_HOSTS`, and run it in a
  container as an unprivileged user.
- **DNS rebinding / browsers.** Requests whose `Host` is not in the allowlist
  (`127.0.0.1`, `localhost`, `[::1]`, plus your bind host; override with
  `AGY_ALLOWED_HOSTS`) get 403, as does any request carrying an `Origin` header
  (browsers add it; API clients do not). `POST` bodies must be `application/json`.
- **Default preset is read-only.** `agy-researcher` can list/search/view files in
  `AGY_WORKSPACE` and nothing else (no edits, no shell, no web, no subagents).
  `agy-light` has no tools at all.
- **`agy-coder` is opt-in** (`AGY_ENABLE_CODER=1` or edit `presets.yaml`) and,
  like any preset that can write files or run commands, **requires
  `AGY_WORKSPACE`** (an absolute directory). Without it the preset loads disabled
  and requests for it get 404. The workspace is the agent's working directory, not a
  proven sandbox: `agy-coder` can reach anything the proxy user can. Run it in a
  container or as an unprivileged user with only the data you accept it touching.
- **Secrets stay out of the agent.** After start-up the proxy removes `AGY_*`,
  `GEMINI_API_KEY` and `*KEY*`/`*TOKEN*`/`*SECRET*` variables from its own
  environment (keeping `HOME`, `PATH`, `GOOGLE_*`), so the agent and its shell
  do not inherit them. Per-agent scratch directories are created and deleted by
  the proxy.
- **Unknown model names are a 404**, not a silent fallback; the default preset is
  used only when a request omits `model`. Client `system` messages are ignored
  unless the preset sets `allow_client_system: true`, and are then labelled
  `[Untrusted client instructions]`.

## Configure

Get a free key at <https://aistudio.google.com/apikey>, then:

```bash
cp .env.example .env        # put GEMINI_API_KEY=... in it
export $(grep -v '^#' .env | xargs)   # or use any dotenv loader
python -m agy_proxy.doctor  # verify the installed SDK matches expectations
```

## Run

```bash
python -m agy_proxy          # serves http://127.0.0.1:8081 and prints the token once
```

Sanity check:

```bash
curl -s localhost:8081/healthz                                  # {"status":"ok"}
curl -s localhost:8081/v1/models -H "Authorization: Bearer $AGY_PROXY_TOKEN"
```

## Wire up your client

### Claude Code (Anthropic `/v1/messages`)

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8081
export ANTHROPIC_API_KEY=<your AGY_PROXY_TOKEN>   # the proxy token; the proxy uses YOUR Gemini key upstream
claude --model agy-researcher            # the model name selects a preset
```

### OpenAI-compatible tools (Cursor, aider, Zed, …)

```
Base URL:  http://127.0.0.1:8081/v1
API key:   <your AGY_PROXY_TOKEN>
Model:     agy-researcher | agy-light   (agy-coder if enabled)
```

```bash
curl -N localhost:8081/v1/chat/completions \
  -H 'content-type: application/json' -H "Authorization: Bearer $AGY_PROXY_TOKEN" \
  -d '{"model":"agy-researcher","stream":true,
       "messages":[{"role":"user","content":"list the files here"}]}'
```

## Presets — your agents

Edit `presets.yaml`. The `model` field a client sends is matched against preset
names; an unknown name is a 404 and `default` applies only when `model` is absent.

```yaml
default: agy-researcher
presets:
  agy-researcher:                 # read/search only
    policy: confirm_run_command
    enabled_tools: [LIST_DIR, SEARCH_DIR, FIND_FILE, VIEW_FILE]
  agy-light:                      # no tools
    policy: deny_all
    enabled_tools: []
  agy-coder:                      # opt-in; needs AGY_WORKSPACE
    enabled: false                # or AGY_ENABLE_CODER=1
    policy: allow_all
    enabled_tools: [LIST_DIR, SEARCH_DIR, FIND_FILE, VIEW_FILE, CREATE_FILE, EDIT_FILE, RUN_COMMAND]
```

Fields: `system_instructions`, `model`, `policy`
(`allow_all` | `safe_defaults` | `confirm_run_command` | `deny_all`),
`behavior` (`autonomous` | `interactive`), `thinking_level`
(`minimal`…`extra_high`), `enabled_tools` (required allowlist of BuiltinTools names;
`[]` = no tools; omit only with `allow_all`), `enable_subagents` (default `false`),
`allow_client_system` (default `false`), `workspaces`, `enabled`.
`disabled_tools` was removed (use the allowlist) and `workspace_only` is rejected.

> `agy-proxy` runs headless, so `safe_defaults` / `confirm_run_command` mean "deny
> shell, allow everything else" -- **including file writes**. The `enabled_tools`
> allowlist is what makes a preset read-only.

## Conversation state (auto)

- **No session id →** stateless: the whole `messages` array is replayed into a
  fresh agent each call. Works with every client out of the box.
- **`x-agy-session` header present →** stateful: a live Agent is kept per session
  (LRU + TTL bounded, see `AGY_SESSION_*`), so follow-ups reuse the SDK's own memory
  and skip harness startup. The id must match `[A-Za-z0-9_.:-]{1,64}` and is scoped to
  your proxy token. The first turn replays the client's full history; later turns send
  only the messages after the last assistant message. The OpenAI `user` field and
  `x-session-id` are **ignored**. A session whose turn fails, is cancelled, or whose
  harness died is discarded; when all `AGY_SESSION_MAX` sessions are busy a new one
  gets `429`.

## Configuration reference

| Env var | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` / `AGY_API_KEY` | – | Your key handed to the SDK |
| `AGY_HOST` / `AGY_PORT` | `127.0.0.1` / `8081` | Bind address (non-loopback requires `AGY_PROXY_TOKEN`) |
| `AGY_PROXY_TOKEN` | generated | Token clients must send (Bearer / `x-api-key`); auto-generated and printed once on loopback |
| `AGY_ALLOWED_HOSTS` | loopback names | Comma-separated accepted `Host` values |
| `AGY_WORKSPACE` | – | Absolute directory for write-capable presets (required for them) |
| `AGY_ENABLE_CODER` | `0` | Enable the `agy-coder` preset |
| `AGY_LOG_TOOLS` | `1` | Log the agent's tool calls (never sent to clients) |
| `AGY_DEFAULT_MODEL` | `gemini-3-pro` | Model when a preset pins none |
| `AGY_PRESETS` | `./presets.yaml` | Presets file path |
| `AGY_SESSION_TTL` / `AGY_SESSION_MAX` | `900` / `64` | Live-agent cache bounds (idle seconds / entries) |
| `AGY_MAX_CONCURRENCY` | `4` | Completions in flight; the next one gets `429` immediately |
| `AGY_TURN_TIMEOUT` | `600` | Seconds per request before `504` (a streaming response is truncated) |
| `AGY_MAX_BODY` | `1048576` | Request body cap in bytes (`413`, also for chunked bodies) |
| `AGY_TEARDOWN_TIMEOUT` | `10` | Seconds a harness gets to exit before SIGTERM, then SIGKILL |
| `AGY_MAX_MODEL_CALLS` / `AGY_MAX_TOOL_CALLS` | `25` / `50` | Per-request agent budget (`0` = unlimited; session agents get 20x) |

## Endpoints

- `POST /v1/chat/completions` — OpenAI (stream + non-stream)
- `POST /v1/messages` — Anthropic (stream + non-stream)
- `GET /v1/models` — presets as models
- `GET /healthz` — liveness (the only unauthenticated endpoint)

## Notes & limits

- Builtin tools (file I/O, shell, search, sub-agents) run **inside** the SDK.
  Tool calls are logged and not sent to clients, who cannot fulfil them.
- Text-only for now; multimodal inputs are on the roadmap.
- See **Security** above before changing the bind address or enabling `agy-coder`.
- This is personal tooling using your own account. Respect Google's and any
  model provider's terms of service.

## Tests

```bash
pip install -e ".[dev]"
python -m agy_proxy.doctor   # SDK surface check
pytest -q                    # adapters, presets, sessions, flattening
ruff check .
```
