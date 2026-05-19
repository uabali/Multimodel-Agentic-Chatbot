# FRAPPE — Multimodal Agentic RAG Chatbot

FRAPPE is a local-first multimodal RAG chatbot built with Chainlit, LangGraph,
Qdrant, BGE-M3 embeddings, a BGE reranker, llama.cpp, SQLite persistence, and
Tavily web search.

The project was developed as a graduation thesis prototype. Its main goal is to
answer Turkish-heavy document questions with grounded citations, while still
supporting direct chat, image input, voice input, web search, and session resume.

## Architecture

```text
Chainlit UI
  ├─ text / image / audio input
  ├─ streaming answers
  ├─ Web Search steps
  └─ side-panel source elements

LangGraph agent
  ├─ router
  ├─ rewriter
  ├─ retriever
  ├─ grader
  ├─ generator
  ├─ web_search
  ├─ direct_response
  ├─ vision
  ├─ vision_rag
  └─ vision_search

Storage and models
  ├─ Qdrant: dense + sparse hybrid retrieval
  ├─ SQLite: Chainlit thread data
  ├─ SQLite: LangGraph checkpoints at data/checkpoint.db
  ├─ llama.cpp: OpenAI-compatible local LLM endpoint
  └─ Tavily: live web search provider
```

## Stack

| Layer | Technology |
|---|---|
| UI | Chainlit 2.x |
| Agent orchestration | LangGraph |
| Checkpointing | `AsyncSqliteSaver`, `data/checkpoint.db` |
| LLM backend | llama.cpp OpenAI-compatible server |
| Default model | `gemma-4-e4b` |
| Embeddings | `BAAI/bge-m3` |
| Vector database | Qdrant |
| Retrieval | Hybrid dense + BM25 |
| Reranking | `BAAI/bge-reranker-base` |
| Web search | Tavily only |
| Speech-to-text | faster-whisper |
| Text-to-speech | edge-tts |
| Observability | LangSmith, sanitized and optional |
| Persistence | SQLite Chainlit data layer |

## Main Features

- Multimodal Chainlit UI for text, images, files, audio, and TTS.
- RAG over uploaded documents with hybrid retrieval and reranking.
- Three-zone dense gate:
  - `pass`: dense score is strong enough to trust retrieval.
  - `soft`: borderline query; retrieval continues and the grader decides.
  - `weak`: retrieval still runs, but insufficient context can refuse without web search.
- Grader reasons: `sufficient`, `partial`, `insufficient_context`, `needs_live_data`, `irrelevant`.
- Graceful refusal when the uploaded documents do not contain enough context.
- Tavily-only web search with a zero-result quality gate.
- Inline citations such as `[Kaynak 1]` and Chainlit side-panel source previews.
- Token-aware context budgeting using `tiktoken` `cl100k_base`.
- Runtime context guard that checks llama.cpp `/props` and uses the safer context size.
- Thread-scoped memory with rolling summary, pinned facts, and last-topic resume message.
- Semantic answer cache with context-aware cache keys.
- LangSmith traces with sanitized metadata, node tags, retrieval scores, and latency fields.

## Pipeline Routes

| Route | Flow | Use case |
|---|---|---|
| `rag` | router → rewriter → retriever → grader → generator | Questions about uploaded documents |
| `web` | router → web_search → generator | Current or live information |
| `direct` | router → direct_response | General chat, math, coding, simple questions |
| `vision` | router → vision | Image-only analysis |
| `vision_rag` | vision → rewriter → retriever → grader → generator | Image plus uploaded document context |
| `vision_search` | vision → web_search → generator | Image plus current/live data |

## Requirements

| Component | Requirement |
|---|---|
| Python | 3.12 |
| Package manager | `uv` |
| Services | Docker / Docker Compose |
| LLM server | llama.cpp `llama-server` |
| macOS | Apple Silicon with Metal recommended |
| Linux | NVIDIA GPU recommended for llama.cpp CUDA |
| RAM | 16 GB minimum, 24 GB+ recommended for local multimodal use |

System packages:

```bash
# Ubuntu / WSL2
sudo apt install -y poppler-utils ffmpeg tesseract-ocr libmagic1 build-essential

# macOS
brew install git cmake ninja python@3.12 libmagic poppler tesseract ffmpeg node
```

## llama.cpp

Build llama.cpp separately:

```bash
git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp
cd ~/llama.cpp

# Linux CUDA
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release

# macOS Metal
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j$(nproc 2>/dev/null || sysctl -n hw.logicalcpu)
```

Set `LLAMA_SERVER_BIN` in `.env` to the built `llama-server` binary.

`llama.cpp` remains the default local backend. `LLM_BACKEND=mlx` is treated as
a compatibility-gated MLX-LM path: start an MLX server on a separate port, run
`uv run python scripts/benchmark_llm_backends.py`, and switch only if streaming,
vision/tool behavior, and `/v1/chat/completions` compatibility pass for your
exact model.

## Setup

```bash
git clone https://github.com/uabali/Multimodel-Agentic-Chatbot.git
cd Multimodel-Agentic-Chatbot
make setup
```

Create local secrets:

```bash
python3 - <<'PY'
import secrets
print("APP_ADMIN_PASSWORD=" + secrets.token_urlsafe(18))
print("APP_PASSWORD_SALT=" + secrets.token_hex(16))
print("CHAINLIT_AUTH_SECRET=" + secrets.token_hex(32))
PY
```

Edit `.env` before starting the app:

```env
LLAMA_SERVER_BIN=/absolute/path/to/llama-server
LLAMA_HF_REPO=lmstudio-community/gemma-4-E4B-it-GGUF:Q4_K_M
LLAMA_CTX_SIZE=8192
LLAMA_PORT=8080

LLM_BACKEND=llama.cpp
LLM_SERVER_URL=http://localhost:8080/v1
LLM_MODEL_NAME=gemma-4-e4b
VISION_MODEL=gemma-4-e4b
LLM_CONTEXT_SIZE=8192

APP_ADMIN_PASSWORD=<strong-password>
APP_PASSWORD_SALT=<random-hex-32>
CHAINLIT_AUTH_SECRET=<random-hex-64>

TAVILY_API_KEY=tvly_...
```

The app refuses to start if `APP_ADMIN_PASSWORD` or `APP_PASSWORD_SALT` still use
the example placeholder values.

## Run

Use separate terminals:

```bash
make qdrant
make llm
make app
```

Open:

```text
http://localhost:7860
```

Useful commands:

```bash
make check
make test
make index          # bulk-index files from data/ into Qdrant
make index DATA_DIR=data/corpus
make stop
make clean
```

## Configuration

Important `.env` values:

```env
# LLM
LLM_BACKEND=llama.cpp
LLM_SERVER_URL=http://localhost:8080/v1
LLM_MODEL_NAME=gemma-4-e4b
LLM_CONTEXT_SIZE=8192
LLM_ENABLE_THINKING=false

# Generation profiles
CHAT_TEMPERATURE=0.7
CHAT_MAX_TOKENS=512
RAG_TEMPERATURE=0.0
RAG_MAX_TOKENS=768
AGENT_TEMPERATURE=0.1
AGENT_MAX_TOKENS=1024
RAG_CONTEXT_SAFETY_MARGIN_TOKENS=700

# Embeddings
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DEVICE=mps

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=user_documents
QDRANT_AUTO_REINDEX=smart

# RAG
CHUNK_SIZE=500
CHUNK_OVERLAP=80
TOP_K=4
RETRIEVAL_STRATEGY=hybrid
BASE_K=4
USE_RERANK=true
RERANK_TOP_N=8
RERANK_FAST_MODE=true
RERANKER_DEVICE=mps

# Terminal logs
APP_LOG_LEVEL=INFO
APP_LOG_PREVIEW_CHARS=96
APP_LOG_STAGE_TIMINGS=true

# Dense gate
RAG_MIN_DENSE_SIMILARITY=0.45
RAG_DENSE_PASS_SIMILARITY=0.62

# Grader
GRADER_CONF_HIGH=0.75
GRADER_CONF_LOW=0.08
GRADER_MAX_DOCS=5

# Web search
TAVILY_API_KEY=tvly_...
WEB_SEARCH_MAX_RESULTS=5
WEATHER_SPECIALIZATION_ENABLED=true

# Observability
APP_LANGSMITH_ENABLED=false
APP_LANGSMITH_REDACT=true
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=frappe-rag-dev

# Semantic cache
SEMANTIC_CACHE_ENABLED=true
SEMANTIC_CACHE_THRESHOLD=0.92
SEMANTIC_CACHE_TTL_HOURS=24

# PDF visual ingest
PDF_VISUAL_INGEST_MAX_PAGES=0
```

If `LANGSMITH_API_KEY` is set, LangSmith tracing is enabled automatically unless
explicitly disabled in settings. Payloads are sanitized before tracing.

## Web Search

Web search uses Tavily only. There is no DuckDuckGo, Brave, Serper, or Google
fallback in the app pipeline.

The Tavily response is treated as empty when results are too weak to trust, for
example when snippets are very short or the result has an unknown date and a
generic title. In that case the assistant returns a graceful refusal instead of
inventing an answer.

During web search, Chainlit shows a tool step and attaches source previews in the
side panel when sources are available.

## Citations

RAG and web answers use inline citation markers:

```text
... answer sentence [Kaynak 1].
```

The final answer may include a short source list. Chainlit also receives side
panel elements for chunks used in context, including source name, page, URL, and
available retrieval/rerank scores.

## Context and Token Budgeting

The app uses `tiktoken` `cl100k_base` for approximate token counting. Gemma uses
a different tokenizer, so the count is not exact, but it is more reliable than a
plain character-based estimate for Turkish-heavy prompts.

The generator builds context with:

- selected recent history,
- retrieved chunks,
- expected output token budget,
- safety margin,
- runtime context size.

At startup, the LLM client tries to read llama.cpp `/props`. If the server
reports a smaller `n_ctx` than configured, the app uses the smaller value.

Check the runtime manually:

```bash
uv run python scripts/verify_llm_runtime.py
```

The script reports configured and discovered context size when `/props` is
available.

## Persistence

There are two SQLite uses:

- Chainlit thread persistence through the app data layer.
- LangGraph checkpoints in `data/checkpoint.db`.

These files are separate. Checkpoints use the Chainlit thread id as LangGraph
`thread_id`, so an interrupted graph run can resume with the same thread id.

Thread memory is scoped to a single Chainlit thread. It stores:

- rolling summary,
- pinned facts,
- last topic.

On resume, Chainlit shows:

```text
Kaldığımız yer: <last topic>
```

## Project Structure

```text
src/
  main.py                     Chainlit app, session handling, audio, source panel
  config.py                   Pydantic settings
  agent/
    graph.py                  LangGraph routes, cache context, checkpoint config
    nodes.py                  Router, retriever, grader, generator, web, vision
    prompts.py                System prompts
    routing.py                Keyword and heuristic routing
    state.py                  AgentState
    web_search.py             Tavily service and source formatting
  rag/
    llm.py                    LLM clients, token counting, n_ctx negotiation
    retriever.py              Retrieval strategy, confidence, deduplication
    vectorstore.py            Qdrant hybrid vector store
    reranker.py               Cross-encoder reranker
    embeddings.py             BGE-M3 embeddings
    ingest.py                 Document ingest
    semantic_cache.py         Qdrant-backed semantic cache
  memory/
    thread_memory.py          Thread-scoped memory
  observability/
    langsmith.py              Sanitized LangSmith helpers
  tools/
    search.py                 Tavily tool
    calculator.py             Safe calculator
    file_reader.py            Uploaded file reader
    mcp_bridge.py             MCP bridge tool
  persistence/
    sqlite_data_layer.py      Chainlit SQLite data layer

tests/
  test_ingest.py
  test_observability.py
  test_rag_retriever.py
  test_security.py
  test_thread_memory.py
  test_token_counter.py
  test_web_search_policy.py
```

## Tests

Run the full suite:

```bash
uv run pytest
```

Current expected result:

```text
111 passed
```

Runtime probe:

```bash
uv run python scripts/verify_llm_runtime.py
```

This requires the llama.cpp server to be running.

## Manual Smoke Test

1. Start Qdrant, llama-server, and Chainlit.
2. Upload a PDF.
3. Ask a document-specific question.
4. Confirm the answer includes `[Kaynak N]`.
5. Open the Chainlit side panel and check source previews and scores.
6. Ask a borderline document question and confirm retrieval still runs.
7. Ask an unrelated document question and confirm graceful refusal.
8. Ask a live-data question and confirm Tavily web search is used.
9. Resume the thread and confirm the last-topic message appears.

## Notes

- PDF text extraction is always available.
- Visual PDF ingest is disabled by default because it renders pages and calls the
  vision model per page.
- `data/` contains local runtime state and should not be committed.
- Web search requires `TAVILY_API_KEY`.
- LangSmith is optional and should be used with redaction enabled.
