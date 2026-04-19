"""
Unified configuration — OpenAI-compatible LLM backends (llama.cpp / vLLM),
HuggingFace for embeddings.

The app talks to the LLM via an OpenAI-compatible REST API (Chat Completions).
This allows swapping the backend without touching agent/RAG logic:
  - llama.cpp: `llama-server --port 8080`  → http://localhost:8080/v1
  - vLLM:     `vllm/vllm-openai`           → http://localhost:8000/v1
"""

from pathlib import Path
from typing import Optional

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── LLM backend (OpenAI-compatible) ──
    # Backends:
    #  - llama.cpp: `llama-server` (recommended for local multimodal Gemma)
    #  - vLLM: `vllm/vllm-openai` (great for fast text + tool-calling models)
    llm_backend: str = Field(
        default="llama.cpp",
        description="LLM backend label. Controls backend-specific request knobs.",
        validation_alias=AliasChoices("LLM_BACKEND"),
    )
    llm_server_url: str = Field(
        default="http://localhost:8080/v1",
        validation_alias=AliasChoices("LLM_SERVER_URL", "VLLM_SERVER_URL"),
    )
    llm_model_name: str = Field(
        default="gemma-4-e4b",
        validation_alias=AliasChoices("LLM_MODEL_NAME", "VLLM_MODEL_NAME"),
    )

    # thinking=True → chain-of-thought (daha yavaş ama daha derin)
    llm_enable_thinking: bool = False

    # ── vLLM-only tuning (docker-compose'daki vLLM komutuna yansır) ──
    # gpu_memory_utilization: 0.85   → ~8.5 GB / 10 GB rezervasyon
    # max_model_len: 32768           → 32K context
    # max_num_seqs: 8                → eşzamanlı request sayısı (4B için artırıldı)
    # enable_auto_tool_choice: true  → agentic tool calling

    # ── Dual LLM profile (Qwen3-4B 32K context — liberalleştirildi) ──
    chat_temperature: float = 0.7
    chat_num_predict: int = 1024
    chat_max_tokens: int = 1024
    rag_temperature: float = 0.0
    rag_num_predict: int = 1536
    rag_max_tokens: int = 1536
    router_max_tokens: int = 64

    # ── Agentic RAG profile (tool calls, multi-turn reasoning) ──
    # Qwen3-4B tool calling çok güçlü — yüksek token budget
    agent_temperature: float = 0.1
    agent_max_tokens: int = 2048

    # ── Embedding (HuggingFace) ──
    # vLLM GPU'yu yönettiğinden default cpu. docker-compose'da app'a GPU eklenirse cuda.
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    embedding_vector_size: Optional[int] = None

    # ── Vision (optional — vLLM multimodal endpoint) ──
    vision_model: str = ""

    # ── Audio (STT) ──
    stt_model: str = "small"

    # ── Qdrant ──
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "user_documents"
    qdrant_prefer_grpc: bool = True
    qdrant_auto_reindex: str = "smart"
    qdrant_auto_recreate_on_mismatch: bool = True

    # ── RAG Settings ──
    # Qwen3-4B 32K context destekler — daha büyük chunk'lar kullanılabilir
    chunk_size: int = 1200
    chunk_overlap: int = 200
    top_k: int = 6

    # ── Hybrid Retrieval ──
    retrieval_strategy: str = "hybrid"
    base_k: int = 10
    fetch_k: int = 30
    lambda_mult: float = 0.6
    score_threshold: float = 0.70

    # ── Dense Gate ──
    # bge-m3 cosine: ilgisiz belgeler ~0.3-0.45 aralığında; 0.45 makul minimum
    # source_filter varsa dense gate tamamen atlanır (retriever_node'a bakın)
    rag_min_dense_similarity: float = 0.45
    rag_dense_gate_k: int = 12

    # ── Reranker ──
    use_rerank: bool = True
    reranker_model: str = "BAAI/bge-reranker-base"
    # vLLM GPU'yu yönettiğinden default cpu
    reranker_device: str = "cpu"
    rerank_top_n: int = 10
    rerank_fast_mode: bool = False

    # ── Web Search ──
    tavily_api_key: str = ""
    brave_api_key: str = ""
    web_search_max_results: int = 5

    # ── MCP ──
    mcp_filesystem_root: str = ""
    google_client_id: str = ""
    google_client_secret: str = ""

    # ── Confidence ──
    local_search_conf_threshold: float = 0.35

    # ── Auth ──
    app_admin_username: str = "admin"
    app_admin_password: str = "admin"
    app_password_salt: str = "local-dev-salt"

    # ── Paths ──
    upload_dir: Path = Path("uploads")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    def ensure_dirs(self):
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    @property
    def llm_model(self) -> str:
        """Backward-compat alias: LLM model name used in log messages."""
        return self.llm_model_name

    # Backward-compat aliases for older code/docs that still say "vLLM".
    @property
    def vllm_server_url(self) -> str:  # pragma: no cover
        return self.llm_server_url

    @property
    def vllm_model_name(self) -> str:  # pragma: no cover
        return self.llm_model_name


settings = Settings()
settings.ensure_dirs()
