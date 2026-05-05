# FRAPPE — Multimodal Agentic RAG Pipeline

Fully local, GPU-accelerated conversational AI system built as a graduation project. Combines **Corrective RAG (CRAG)**, **multimodal vision**, **ReAct tool-calling**, and **real-time web search** in a single LangGraph pipeline served through a Chainlit web UI.

```
┌─────────────────────────────┐
│    Chainlit UI  (port 7860) │
│  Text · Image · Voice       │
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│     LangGraph StateGraph    │
│                             │
│  Router → Rewriter          │
│    → Retriever → Grader     │
│    → Generator              │
│    └─(CRAG)─→ WebSearch     │
│  Direct → ReAct Agent       │
│  Vision → Vision-RAG        │
└──────┬───────────┬──────────┘
       │           │
┌──────▼──────┐ ┌──▼──────────────┐
│ llama.cpp   │ │ Qdrant (Docker) │
│ Gemma 4 E4B │ │ Dense + BM25    │
│ port 8080   │ │ port 6333       │
└─────────────┘ └────────────────┘
```

## Tech Stack

| Layer | Technology |
|---|---|
| LLM inference | llama.cpp · Gemma 4 E4B (GGUF, Q4_K_M / Q5_K_M) |
| Orchestration | LangGraph `StateGraph` |
| Vector store | Qdrant — hybrid dense (BGE-M3, 1024-dim) + sparse (BM25) |
| Reranking | BAAI/bge-reranker-base cross-encoder |
| Frontend | Chainlit 2.x — streaming, audio, file upload |
| STT | faster-whisper (CPU int8) |
| TTS | edge-tts (Microsoft Azure Neural) |
| Web search | Tavily API + DuckDuckGo fallback |
| External tools | MCP servers via `langchain-mcp-adapters` |
| Persistence | SQLite — thread history, session resume |
| Semantic cache | Qdrant collection with TTL + context-key filter |

## Pipeline Modes

| Input | Route | Flow |
|---|---|---|
| Text query | `rag` | Rewriter → Retriever → Grader → Generator |
| Low-relevance text | `rag` + CRAG | → WebSearch → Generator |
| General question | `direct` | ReAct agent (web, calc, file, MCP tools) |
| Image only | `vision` | Gemma-4 multimodal → END |
| Image + document | `vision_rag` | Vision → RAG pipeline |
| Image + web query | `vision_search` | Vision → WebSearch → Generator |

## Prerequisites

- **OS**: Linux (Ubuntu 22.04 / WSL2)
- **GPU**: NVIDIA ≥ 8 GB VRAM (16 GB recommended)
- **Python**: 3.12
- **Tools**: `uv`, Docker, `llama.cpp` compiled with CUDA

```bash
sudo apt install -y poppler-utils ffmpeg tesseract-ocr
```

## Quick Start

```bash
git clone https://github.com/uabali/Multimodel-Agentic-Chatbot.git
cd Multimodel-Agentic-Chatbot
make setup        # creates .venv, generates .env template
```

Edit `.env` — mandatory fields:

```env
LLAMA_SERVER_BIN=/absolute/path/to/llama-server
LLM_MODEL_NAME=gemma-4-e4b
APP_ADMIN_PASSWORD=<strong-password>
APP_PASSWORD_SALT=<random-hex-32>
CHAINLIT_AUTH_SECRET=<random-hex-64>
```

Start the stack:

```bash
make qdrant   # start Qdrant in Docker
make llm      # start llama-server
make app      # start Chainlit UI at http://localhost:7860
make check    # health check
```

## Project Structure

```
src/
├── main.py                   # Chainlit entry point, session, STT/TTS
├── config.py                 # Pydantic settings
├── tts.py                    # edge-tts synthesis, language detection
├── agent/
│   ├── graph.py              # LangGraph DAG, astream_agent()
│   ├── nodes.py              # Node implementations
│   ├── state.py              # AgentState TypedDict
│   ├── routing.py            # Keyword routing + LLM fallback
│   ├── prompts.py            # System prompts
│   └── web_search.py         # Tavily service
├── rag/
│   ├── ingest.py             # Document loader/splitter
│   ├── vectorstore.py        # HybridVectorStore (Qdrant)
│   ├── retriever.py          # Strategy factory, confidence scoring
│   ├── reranker.py           # Cross-encoder with TTL cache
│   ├── semantic_cache.py     # Query cache with context key
│   ├── embeddings.py         # BGE-M3 singleton
│   └── llm.py                # DualLLM profiles (chat/rag/agent)
├── mcp/
│   ├── mcp_client.py         # MultiServerMCPClient
│   └── mcp_config.json       # Server definitions
├── tools/
│   ├── search.py             # Tavily + DuckDuckGo
│   ├── calculator.py         # AST-based safe eval
│   ├── file_reader.py        # Upload sandbox (path traversal protected)
│   └── mcp_bridge.py         # LangChain ↔ MCP adapter
├── api/router.py             # FastAPI admin endpoints
├── middleware/rate_limiter.py # Sliding-window per-IP limiter
└── persistence/sqlite_data_layer.py

tests/
└── test_security.py          # 14 tests: traversal, rate limiter, API auth
```

## Configuration

Key `.env` variables (see `src/config.py` for full list):

```env
LLM_BACKEND=llama.cpp          # or vllm
LLM_SERVER_URL=http://localhost:8080/v1
LLM_CONTEXT_SIZE=16384

CHUNK_SIZE=1200
CHUNK_OVERLAP=200
TOP_K=6
RETRIEVAL_STRATEGY=hybrid      # hybrid | similarity | mmr | threshold
USE_RERANK=true

SEMANTIC_CACHE_ENABLED=true
SEMANTIC_CACHE_THRESHOLD=0.92
SEMANTIC_CACHE_TTL_HOURS=24

TAVILY_API_KEY=tvly_...
```

## Makefile

```bash
make setup    # install deps, generate .env
make qdrant   # start Qdrant (Docker)
make llm      # start llama-server
make app      # start Chainlit
make check    # health + LLM probe
make stop     # stop all services
make clean    # remove .venv, caches
```

## Tests

```bash
source .venv/bin/activate
pytest tests/test_security.py -v
```
