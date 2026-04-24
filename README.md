# FRAPPE — Multimodal Agentic RAG Pipeline

Fully local, GPU-accelerated conversational AI system. Combines **Corrective RAG** (CRAG), **multimodal vision**, **ReAct tool-calling**, and **real-time web search** into a single LangGraph pipeline served through a Chainlit web UI.

```
                         ┌────────────────────────────────┐
                         │     Chainlit UI (port 7860)     │
                         │  Text · Image · Voice (STT/TTS) │
                         └───────────────┬────────────────┘
                                         │
                         ┌───────────────▼────────────────┐
                         │      LangGraph StateGraph       │
                         │                                 │
                         │  Router → Rewriter → Retriever  │
                         │    → Grader → Generator         │
                         │    └─(CRAG)→ WebSearch          │
                         │  Direct → ReAct Agent           │
                         │  Vision → Vision-RAG / Search   │
                         └────────┬──────────┬────────────┘
                                  │          │
                    ┌─────────────▼──┐  ┌────▼──────────────┐
                    │ llama.cpp/vLLM  │  │  Qdrant (Docker)  │
                    │ Gemma 4 E4B     │  │  Dense + BM25      │
                    │ port 8080/8000  │  │  port 6333/6334    │
                    └────────────────┘  └───────────────────┘
```

## Tech Stack

| Layer | Technology |
|---|---|
| LLM inference | llama.cpp serving Gemma 4 E4B (GGUF, Q4_K_M or Q5_K_M) |
| Orchestration | LangGraph `StateGraph` with typed `AgentState` |
| Vector store | Qdrant — hybrid dense (BGE-M3, 1024-dim) + sparse (BM25) |
| Reranking | BAAI/bge-reranker-base cross-encoder with TTL cache |
| Embeddings | BAAI/bge-m3 via `sentence-transformers` (CPU) |
| Frontend | Chainlit 2.x — streaming, audio, file upload |
| STT | faster-whisper (CPU int8) |
| TTS | edge-tts (Microsoft Azure Neural, no API key) |
| Web search | Tavily API + DuckDuckGo fallback |
| External tools | MCP servers via `langchain-mcp-adapters` |
| Persistence | SQLite — threads, steps, history resume |
| Semantic cache | Qdrant collection with TTL + context-key filter |

## Features

### Pipeline Modes

| Input | Route | Pipeline |
|---|---|---|
| Text query | `rag` | Rewriter → Retriever (hybrid) → Grader → Generator |
| Text (low relevance) | `rag` + CRAG | ... → WebSearch → Generator |
| General question | `direct` | ReAct agent with tools (web, calc, file, MCP) |
| Image only | `vision` | Gemma-4 multimodal analysis → END |
| Image + uploaded doc | `vision_rag` | Vision → Rewriter → RAG pipeline |
| Image + web query | `vision_search` | Vision → WebSearch → Generator |

### Routing (Zero LLM Cost for Common Cases)

The router resolves most queries through regex patterns before ever calling the LLM:

- **Tier 0** (0 ms): `image_data` present → vision route
- **Tier 1** (0 ms): `source_filter` set → forced RAG
- **Tier 2** (0 ms): `keyword_route()` — 6 pattern sets (document pronouns, RAG signals, direct, web, MCP)
- **Tier 3** (0 ms): `session_uploads` + `is_web_query()` heuristic
- **Tier 4** (~200 ms): LLM fallback (64 tokens max, temperature 0.0)

### CRAG (Corrective RAG)

The grader validates document relevance before generation. It distinguishes three cases:

- `yes` → documents sufficient, proceed to generator
- `no / irrelevant` → topic mismatch; with `source_filter` active: generator says "not in document"; otherwise: web search triggered
- `no / needs_live_data` → document contains relevant structure but requires real-time values (prices, exchange rates); web search always triggered regardless of `source_filter`

Grading is three-tiered to avoid unnecessary LLM calls:
- Confidence ≥ 0.75 → `yes` (no LLM)
- Confidence < 0.15 → `no` (no LLM)
- 0.15–0.75 → LLM grader (temperature 0.0, structured JSON output)

### Vision

Four specialized prompts selected by keyword match (0 ms, no LLM):

| Trigger word | Prompt mode | Output |
|---|---|---|
| `fatura`, `invoice` | Invoice extractor | Structured JSON |
| `tablo`, `table` | Table reconstructor | Markdown table |
| `grafik`, `chart` | Chart analyzer | Data points + trend |
| `şema`, `diagram` | Diagram decomposer | Components + flow |
| (default) | General vision | Prose description |

### Semantic Cache

Before the graph runs, `astream_agent()` checks Qdrant for a semantically similar previous answer:

- Similarity threshold: 0.92 cosine (configurable)
- TTL: 24 hours (configurable)
- Cache key includes `source_filter + sorted(session_uploads) + retrieval_strategy` — same question from a different document context never returns a stale answer
- Images and audio inputs bypass the cache entirely

### Long-Term Memory

After 40 messages the conversation history is summarized by the LLM (max 300 words, Turkish) and compressed to the last 10 messages. The summary is stored in SQLite thread metadata and prepended as a `SystemMessage` on every subsequent turn.

### Streaming TTS

`_TtsStreamer` parallelises synthesis with streaming:

1. After 150 characters accumulate, synthesis of the first sentence group starts in the background
2. The rest of the response continues streaming
3. Once streaming ends, the remaining text is synthesized and both MP3 segments are concatenated into a single seamless audio element

Language detection (Turkish character set / function words) selects the correct voice automatically.

## Prerequisites

- **OS**: Linux (tested on Ubuntu 22.04 / WSL2)
- **GPU**: NVIDIA with ≥ 8 GB VRAM (16 GB recommended)
- **Python**: 3.12
- **Tools**: `uv` package manager, Docker, `llama.cpp` compiled with CUDA

**System packages:**
```bash
sudo apt install -y poppler-utils ffmpeg tesseract-ocr
```

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/uabali/Multimodel-Agentic-Chatbot.git
cd Multimodel-Agentic-Chatbot
make setup          # creates .venv via uv, generates .env template
```

### 2. Configure `.env`

Mandatory fields (no defaults — startup fails without them):

```env
# LLM
LLAMA_SERVER_BIN=/absolute/path/to/llama-server
LLM_MODEL_NAME=gemma-4-e4b

# Auth — must be set to strong values
APP_ADMIN_PASSWORD=<strong-password>
APP_PASSWORD_SALT=<random-hex-32>
CHAINLIT_AUTH_SECRET=<random-hex-64>
```

Optional:
```env
TAVILY_API_KEY=tvly_...          # web search
GITHUB_PERSONAL_ACCESS_TOKEN=ghp_...  # GitHub MCP
```

### 3. Start the stack

```bash
make qdrant    # start Qdrant in Docker
make llm       # start llama-server (downloads model on first run)
make app       # start Chainlit UI
```

Open: `http://localhost:7860`  
Admin: username = `APP_ADMIN_USERNAME` (default: `admin`), password = `APP_ADMIN_PASSWORD`

### 4. Verify

```bash
make check    # health check + LLM probe
```

## VRAM Planning

| GPU | Config | KV cache | Total VRAM |
|---|---|---|---|
| 8 GB | `CTX=4096 PARALLEL=2` | ~1.3 GB | ~6.8 GB |
| 12 GB | `CTX=8192 PARALLEL=4` | ~2.7 GB | ~8.3 GB |
| 16 GB | `CTX=16384 PARALLEL=4` | ~5.4 GB | ~10.9 GB |

> `PARALLEL` = simultaneous users. Per-user context = `CTX / PARALLEL`.

## API

The FastAPI router is mounted inside Chainlit at `/api`. All config endpoints require HTTP Basic auth.

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/api/health` | none | LLM + Qdrant reachability + latency |
| `GET` | `/api/config` | admin | Current LLM configuration |
| `PUT` | `/api/config/llm` | admin | Hot-swap LLM URL + model name |
| `POST` | `/api/llm/probe` | admin | Latency measurement without changing config |

Swagger UI: `http://localhost:7860/docs`

**Hot-swap example:**
```bash
curl -u admin:$APP_ADMIN_PASSWORD -X PUT http://localhost:7860/api/config/llm \
  -H 'Content-Type: application/json' \
  -d '{"url":"http://localhost:8000/v1","model_name":"llama-3-8b"}'
```

## MCP Servers

Configured in `src/mcp/mcp_config.json`. Enable by setting `"disabled": false` and providing the required env var.

| Server | Purpose | Required env var |
|---|---|---|
| `filesystem` | Browse local files (active by default) | `MCP_FILESYSTEM_ROOT` (optional) |
| `brave-search` | Web search via Brave | `BRAVE_API_KEY` |
| `google-calendar` | Meeting/schedule awareness | `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` |
| `gmail` | Email search and summarization | `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` |
| `github` | Repos, issues, PRs | `GITHUB_PERSONAL_ACCESS_TOKEN` |

## Project Structure

```
src/
├── main.py                    # Chainlit entry point, session management, STT/TTS
├── config.py                  # Pydantic settings (single source of truth)
├── tts.py                     # edge-tts synthesis, language detection
├── agent/
│   ├── state.py               # AgentState TypedDict
│   ├── graph.py               # LangGraph DAG, conditional edges, astream_agent()
│   ├── nodes.py               # All node implementations
│   ├── prompts.py             # All system prompts
│   ├── routing.py             # Keyword routing patterns + helpers
│   └── web_search.py          # Tavily service, weather formatter
├── rag/
│   ├── ingest.py              # Document loader/splitter/ingester
│   ├── vectorstore.py         # HybridVectorStore (Qdrant)
│   ├── embeddings.py          # BGE-M3 singleton
│   ├── retriever.py           # Strategy factory, dynamic-k, confidence scoring
│   ├── reranker.py            # Cross-encoder with TTL cache
│   ├── semantic_cache.py      # Qdrant-backed query cache with context key
│   └── llm.py                 # DualLLM (chat/rag/agent profiles)
├── mcp/
│   ├── mcp_client.py          # MultiServerMCPClient loader
│   └── mcp_config.json        # Server definitions
├── tools/
│   ├── search.py              # tavily_search, search_web (DuckDuckGo)
│   ├── calculator.py          # AST-based safe eval
│   ├── file_reader.py         # Upload sandbox reader (path traversal protected)
│   └── mcp_bridge.py          # LangChain ↔ MCP tool adapter
├── api/
│   └── router.py              # FastAPI admin endpoints
├── middleware/
│   └── rate_limiter.py        # Sliding-window per-IP limiter
└── persistence/
    └── sqlite_data_layer.py   # Chainlit BaseDataLayer (threads, steps, resume)

tests/
└── test_security.py           # 14 tests: path traversal, rate limiter, API auth
```

## Supported File Types

| Extension | Handler | RAG indexed |
|---|---|---|
| `.pdf` | PyPDFLoader | yes |
| `.docx` | UnstructuredWordDocumentLoader | yes |
| `.txt`, `.md` | TextLoader | yes |
| `.xlsx`, `.csv` | UnstructuredExcelLoader / CSVLoader | yes |
| `.mp3`, `.wav`, `.ogg`, `.m4a`, `.flac` | faster-whisper STT → TXT | yes |
| `.png`, `.jpg`, `.jpeg`, `.webp` | Gemma-4 Vision | no (vision pipeline) |

Upload limits: 5 files per message, 20 MB per file.

## Security

- Passwords hashed with **PBKDF2-HMAC-SHA256** at 210,000 iterations (OWASP 2024 recommendation)
- Constant-time comparison via `hmac.compare_digest` (timing-attack resistant)
- Admin API requires HTTP Basic authentication on all config-mutating endpoints
- `APP_ADMIN_PASSWORD` and `APP_PASSWORD_SALT` have **no defaults** — startup fails without them in `.env`
- File reader sandbox: `Path.resolve()` + `is_relative_to()` prevents directory traversal
- Rate limiter X-Forwarded-For only trusted from `TRUSTED_PROXY_IPS` (default: `127.0.0.1,::1`)

## Makefile Reference

```bash
make setup    # install deps (uv), generate .env template
make qdrant   # docker compose up + health check
make llm      # start llama-server (downloads model if needed)
make app      # chainlit run src/main.py --port 7860
make dev      # check-qdrant + check-llm + app
make check    # health endpoints + LLM probe
make tunnel   # Cloudflare/ngrok tunnel to localhost:7860
make stop     # stop Docker + kill llama-server
make clean    # remove .venv, caches, __pycache__
```

## Running Tests

```bash
source .venv/bin/activate
pytest tests/test_security.py -v
```

## Configuration Reference

Key `.env` variables (see `src/config.py` for full list):

```env
# LLM Backend
LLM_BACKEND=llama.cpp          # or vllm
LLM_SERVER_URL=http://localhost:8080/v1
LLM_MODEL_NAME=gemma-4-e4b
LLM_CONTEXT_SIZE=16384

# RAG Tuning
CHUNK_SIZE=1200
CHUNK_OVERLAP=200
TOP_K=6
RETRIEVAL_STRATEGY=hybrid      # hybrid | similarity | mmr | threshold
USE_RERANK=true
RAG_MIN_DENSE_SIMILARITY=0.38

# Semantic Cache
SEMANTIC_CACHE_ENABLED=true
SEMANTIC_CACHE_THRESHOLD=0.92
SEMANTIC_CACHE_TTL_HOURS=24

# Auth (required, no defaults)
APP_ADMIN_USERNAME=admin
APP_ADMIN_PASSWORD=<required>
APP_PASSWORD_SALT=<required>
CHAINLIT_AUTH_SECRET=<required>

# Proxy (for rate limiter)
TRUSTED_PROXY_IPS=127.0.0.1,::1
```
