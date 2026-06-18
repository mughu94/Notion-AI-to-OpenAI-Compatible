# NotionChat

OpenAI-compatible HTTP API that routes chat requests to **Notion AI** (`runInferenceTranscript`) using your Notion browser session.

Use it with [Cursor](https://cursor.com), [9router](https://github.com), Postman, or any client that speaks the OpenAI Chat Completions API.

> **Unofficial project.** Not affiliated with or endorsed by Notion. You are responsible for complying with Notion's terms of service and your workspace plan limits.

## Features

- **OpenAI-compatible endpoints**
  - `POST /v1/chat/completions` (streaming and non-streaming)
  - `GET /v1/models` (dynamic list from Notion `getAvailableModels`, with fallbacks)
  - `GET /healthz`
- **Browser cookie authentication** — no separate Notion API key required
- **Model aliases** — e.g. `opus-4.8`, `gpt-4o`, `sonnet-4.6` mapped to Notion internal model IDs
- **Thread state** for normal chat sessions (optional `user` field for continuity)
- **Tools bridge (experimental)** — prompt-based translation between OpenAI-style tool calls and Notion AI output, aimed at Cursor Agent mode
- **Postman collection** in [`postman/`](postman/)

## How it works

```
Client (Cursor / 9router / Postman)
        │
        ▼
  NotionChat (FastAPI)
        │  OpenAI messages + tools → Notion transcript
        ▼
  Notion API  POST /api/v3/runInferenceTranscript
        │  NDJSON stream
        ▼
  NotionChat parses response → OpenAI chat.completion (+ optional tool_calls)
```

Notion AI is a **chat** product. It does not expose native OpenAI function calling. For IDE agent workflows, NotionChat compiles model text into `tool_calls` (JSON, code fences, shell commands) and applies guardrails (scaffold sequencing, denial handling, etc.).

## Requirements

- Python 3.11+ recommended
- A Notion account with AI access (plan limits apply)
- A valid Notion browser session (`token_v2` cookie)

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/NotionChat.git
cd NotionChat
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

| Variable | Description |
|----------|-------------|
| `NOTIONCHAT_API_KEY` | Bearer token clients must send (`Authorization: Bearer ...`) |
| `NOTIONCHAT_HOST` | Bind host (default `127.0.0.1`) |
| `NOTIONCHAT_PORT` | Port (default `8787`) |
| `NOTIONCHAT_ACCOUNT` | Path to account JSON (default `notion_account.json`) |
| `NOTIONCHAT_THREADS_DIR` | Thread state directory (default `threads/`) |
| `NOTIONCHAT_DEFAULT_MODEL` | Default Notion model ID |
| `NOTION_COOKIE` | Optional — full `document.cookie` for auto-bootstrap on startup |

### 3. Bootstrap account from browser cookie

1. Log in to [Notion](https://www.notion.com) in your browser.
2. Open DevTools → **Application** → **Cookies** → `https://www.notion.com`.
3. Copy the full cookie string (must include `token_v2` and ideally `notion_browser_id`).

```bash
python -m notionchat init --cookie "notion_browser_id=...; token_v2=..."
```

If you have multiple workspaces:

```bash
python -m notionchat init --cookie "..." --space-name "My Workspace"
```

This writes `notion_account.json` (gitignored — **do not commit**).

Alternatively, set `NOTION_COOKIE` in `.env` and NotionChat will bootstrap on first request.

### 4. Run the server

```bash
python -m notionchat serve
```

Server URL: `http://127.0.0.1:8787`

### 5. Test

```bash
curl http://127.0.0.1:8787/healthz

curl http://127.0.0.1:8787/v1/models \
  -H "Authorization: Bearer sk-notionchat"

curl http://127.0.0.1:8787/v1/chat/completions \
  -H "Authorization: Bearer sk-notionchat" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"opus-4.8\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one sentence.\"}]}"
```

Import [`postman/NotionChat.postman_collection.json`](postman/NotionChat.postman_collection.json) for more examples.

## Cursor / 9router setup

1. Run NotionChat locally (`python -m notionchat serve`).
2. In your router or Cursor custom model settings:
   - **Base URL:** `http://127.0.0.1:8787/v1`
   - **API key:** value of `NOTIONCHAT_API_KEY` from `.env`
   - **Model:** e.g. `opus-4.8` (see `/v1/models`)

**Chat mode** generally works well.

**Agent mode** (tools) is experimental — see limitations below.

## API

### `POST /v1/chat/completions`

Supports common OpenAI fields:

- `model`, `messages`, `stream`
- `tools`, `tool_choice`, `parallel_tool_calls`
- `user` (optional session key for thread continuity in non-agent chat)

### `GET /v1/models`

Returns models available to your Notion workspace, cached for 5 minutes.

## Project layout

```
notionchat/
  openai_api.py    # FastAPI routes
  client.py        # Notion inference client
  tools.py         # OpenAI tools ↔ Notion bridge / compiler
  transcript.py    # Notion transcript payloads
  ndjson.py        # Stream parser
  bootstrap.py     # Cookie → account bootstrap
  models.py        # Model aliases and /v1/models
postman/           # Postman collection
.env.example
config.example.json
```

## Known limitations

### Tool calling / Cursor Agent — not fully working

This is the main gap if you expect parity with OpenAI, Anthropic, or native Cursor models.

**Root cause:** Notion AI is not an IDE agent. It runs as Notion's chat assistant (`surface: ai_module`), often refuses filesystem/tool access, and does not reliably emit OpenAI-style `tool_calls`. NotionChat bridges this with prompts, parsing, and heuristics — not true function calling.

**What often goes wrong**

| Symptom | Why |
|---------|-----|
| Model says *"I can't access your filesystem / Cursor tools"* | Notion AI persona rejects IDE tool context as out-of-scope |
| `out=0` tool rounds but no real progress | Compiler emitted `Shell`/`Write` from prose; Notion didn't plan the task |
| `npm create` used a subfolder | Notion suggests `npm create vite@latest my-app`; paths drift from workspace (partially mitigated: forced `.` scaffold) |
| Shell + Write in one turn | Cursor runs tools in parallel; writes happen before scaffold finishes (mitigated: scaffold-only first turn) |
| Glob / search loops | Model keeps exploring instead of building (partially mitigated: loop tool filtering) |
| Complex apps (Vite + React + Tailwind + shadcn) stall | Multi-step scaffolding needs reliable tool planning; Notion output is inconsistent |

**What NotionChat tries to do today**

- Detect Cursor Agent requests (tools + system prompt markers)
- Inject full tool schemas and JSON tool-call instructions
- Compile Notion output into `tool_calls` from:
  - JSON `tool_calls` in text
  - NDJSON `tool_use` events
  - `bash` / `npm` / `npx` lines → `Shell`
  - Path-tagged code fences → `Write`
- Bootstrap `npm create vite@latest .` when Notion refuses on coding tasks
- Run scaffold `Shell` alone first; defer `Write` until shell tool results exist
- Suppress Notion refusal prose from reaching Cursor when possible
- Disable Notion thread reuse in agent mode to avoid polluted state

**Recommendation**

| Use case | Suggested backend |
|----------|-------------------|
| Chat, Q&A, drafting | NotionChat — works well |
| Light edits with tools | NotionChat — may work, expect retries |
| Full Agent builds (scaffold + many files) | OpenAI / Anthropic / native Cursor models |

Contributions welcome on [`notionchat/tools.py`](notionchat/tools.py) — especially a more robust `compile_agent_tool_calls` pipeline and better Notion prompt strategy.

### Other limitations

- **Cookie expiry** — refresh `token_v2` when you get 401/403 errors
- **Unofficial API** — Notion can change endpoints or formats without notice (`patchResponseVersion`, NDJSON events, etc.)
- **Rate limits / credits** — subject to your Notion plan
- **Security** — the cookie grants full account access; run locally and never publish `notion_account.json` or `.env`

## Security

- Never commit `.env`, `notion_account.json`, or `threads/`
- Use a strong `NOTIONCHAT_API_KEY` if the server is reachable on your network
- Default bind is `127.0.0.1` — avoid exposing to the public internet without proper auth and TLS

## Troubleshooting

| Problem | Things to try |
|---------|----------------|
| Empty assistant response | Refresh cookie; check `space_id` in account file; verify AI credits |
| `401` / `403` from Notion | Re-run `python -m notionchat init --cookie "..."` |
| Agent does nothing | New Agent chat; check server logs for `IDE bridge tool_calls=...` |
| Wrong model | Call `GET /v1/models`; use an ID from that list |
| Business / empty stream | Ensure `curl_cffi` is installed (Chrome TLS impersonation) |

Enable server logging:

```bash
# NotionChat logs IDE bridge decisions at INFO level
python -m notionchat serve
```

## Development

```bash
pip install -r requirements.txt
python -m py_compile notionchat/*.py
python -m notionchat serve
```

## Disclaimer

This software is provided as-is. Using browser session cookies to access Notion's internal APIs may violate Notion's Terms of Service. Use at your own risk.

## License

No license file is included yet. Add a `LICENSE` (e.g. MIT) before publishing if you want others to reuse the code.
