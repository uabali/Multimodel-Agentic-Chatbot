from pathlib import Path
from typing import Optional

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── LLM backend (OpenAI-compatible) ──
    # Backends:
    #  - llama.cpp: `llama-server` (recommended stable default on Apple Silicon)
    #  - mlx: MLX-LM OpenAI-like server; benchmark/compatibility-gated
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

    # Sunucunun gerçek context penceresi (llama-server -c / vLLM max-model-len).
    # Generator bütçesi bu değerden max_tokens çıkarılarak hesaplanır.
    # Varsayılan 8192 — llama-server LLAMA_CTX_SIZE ile eşleşmeli.
    llm_context_size: int = Field(
        default=8192,
        validation_alias=AliasChoices("LLM_CONTEXT_SIZE", "LLM_N_CTX"),
    )

    # ── vLLM-only tuning (docker-compose'daki vLLM komutuna yansır) ──
    # gpu_memory_utilization: 0.85   → ~8.5 GB / 10 GB rezervasyon
    # max_model_len: 32768           → 32K context
    # max_num_seqs: 8                → eşzamanlı request sayısı (4B için artırıldı)
    # enable_auto_tool_choice: true  → agentic tool calling

    # ── Dual LLM profile (Gemma 4 E4B — liberalleştirildi) ──
    chat_temperature: float = 0.7
    chat_max_tokens: int = 768
    rag_temperature: float = 0.0
    rag_max_tokens: int = 1536
    router_max_tokens: int = 64
    rag_prompt_version: str = "rag-v2"
    rag_context_safety_margin_tokens: int = 700

    # ── Agentic RAG profile (tool calls, multi-turn reasoning) ──
    # Gemma 4 E4B tool calling için yüksek token budget
    agent_temperature: float = 0.1
    agent_max_tokens: int = 1536

    # ── Embedding (HuggingFace) ──
    # Mac single-user fast profile defaults to Apple Metal via MPS.
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "mps"
    embedding_vector_size: Optional[int] = None

    # ── Vision (optional — vLLM multimodal endpoint) ──
    vision_model: str = ""

    # ── Audio (STT) ──
    stt_model: str = "small"

    # ── Qdrant ──
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "user_documents"
    qdrant_prefer_grpc: bool = False
    qdrant_auto_reindex: str = "smart"
    qdrant_auto_recreate_on_mismatch: bool = True

    # ── RAG Settings ──
    # Küçük chunk → daha yüksek retrieval precision, daha az irrelevant context
    chunk_size: int = 500
    chunk_overlap: int = 80
    top_k: int = 6

    # PDF sayfalarını vision ile OCR/analiz etme limiti. Varsayılan 0'dır:
    # text PDF ingest hızlı kalır, pahalı sayfa başı vision çağrıları sadece
    # bilinçli olarak N > 0 verildiğinde çalışır.
    pdf_visual_ingest_max_pages: int = Field(
        default=0,
        validation_alias=AliasChoices("PDF_VISUAL_INGEST_MAX_PAGES"),
    )

    # ── Hybrid Retrieval ──
    retrieval_strategy: str = "hybrid"
    base_k: int = 6
    fetch_k: int = 40
    lambda_mult: float = 0.6
    score_threshold: float = 0.62

    # ── Dense Gate ──
    # bge-m3 cosine: ilgisiz belgeler ~0.3-0.45 aralığında; 0.45 makul minimum
    # source_filter varsa dense gate tamamen atlanır (retriever_node'a bakın)
    rag_min_dense_similarity: float = 0.45
    rag_dense_pass_similarity: float = 0.62
    rag_dense_gate_k: int = 12

    # ── Reranker ──
    use_rerank: bool = True
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_device: str = "mps"
    rerank_top_n: int = 8
    rerank_fast_mode: bool = Field(
        default=True,
        validation_alias=AliasChoices("RERANK_FAST_MODE"),
    )
    retriever_score_lookup: Optional[bool] = Field(
        default=None,
        description="When unset, extra retriever score lookup runs only under detailed DEBUG logging.",
        validation_alias=AliasChoices("RETRIEVER_SCORE_LOOKUP"),
    )

    # ── Web Search ──
    tavily_api_key: str = ""
    brave_api_key: str = ""
    web_search_max_results: int = 5
    weather_specialization_enabled: bool = True

    # ── MCP ──
    mcp_filesystem_root: str = ""
    google_client_id: str = ""
    google_client_secret: str = ""

    # ── Semantic Cache ──
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = 0.92
    semantic_cache_ttl_hours: int = 24

    # ── Thread memory / summarization ──
    summary_trigger_messages: int = Field(
        default=32,
        validation_alias=AliasChoices("SUMMARY_TRIGGER_MESSAGES"),
    )
    summary_trigger_tokens: int = Field(
        default=12000,
        validation_alias=AliasChoices("SUMMARY_TRIGGER_TOKENS"),
    )

    history_max_messages_rag: int = Field(
        default=6,
        validation_alias=AliasChoices("HISTORY_MAX_MESSAGES_RAG"),
    )
    history_token_budget_rag: int = Field(
        default=900,
        validation_alias=AliasChoices("HISTORY_TOKEN_BUDGET_RAG"),
    )
    history_max_messages_chat: int = Field(
        default=8,
        validation_alias=AliasChoices("HISTORY_MAX_MESSAGES_CHAT"),
    )
    history_token_budget_chat: int = Field(
        default=1300,
        validation_alias=AliasChoices("HISTORY_TOKEN_BUDGET_CHAT"),
    )
    answer_hallucination_markers: list[str] = Field(
        default=[
            "ihtiyacım", "yapabilmem için", "kritik bilgi", "hesaplayabilmem",
            "belirtmek isterim", "lütfen", "sunabilmem", "verebilmem"
        ],
        validation_alias=AliasChoices("ANSWER_HALLUCINATION_MARKERS"),
    )

    # ── Qdrant tenant isolation (global retrieval without upload scope) ──
    qdrant_tenant_filter_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("QDRANT_TENANT_FILTER_ENABLED"),
    )
    qdrant_include_shared_corpus: bool = Field(
        default=True,
        description="When true, chunks with empty metadata.user_id remain visible (bulk index).",
        validation_alias=AliasChoices("QDRANT_INCLUDE_SHARED_CORPUS"),
    )

    # ── Confidence ──
    local_search_conf_threshold: float = 0.35
    grader_conf_high: float = 0.75
    grader_conf_low: float = 0.08
    grader_max_docs: int = 5

    # ── Observability (LangSmith, opt-in) ──
    app_env: str = Field(default="local", validation_alias=AliasChoices("APP_ENV"))
    app_langsmith_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("APP_LANGSMITH_ENABLED"),
    )
    langsmith_api_key: str = Field(default="", validation_alias=AliasChoices("LANGSMITH_API_KEY"))
    langsmith_project: str = Field(
        default="frappe-rag-dev",
        validation_alias=AliasChoices("LANGSMITH_PROJECT"),
    )
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        validation_alias=AliasChoices("LANGSMITH_ENDPOINT"),
    )
    langsmith_workspace_id: str = Field(
        default="",
        validation_alias=AliasChoices("LANGSMITH_WORKSPACE_ID"),
    )
    app_langsmith_redact: bool = Field(
        default=True,
        validation_alias=AliasChoices("APP_LANGSMITH_REDACT"),
    )
    app_langsmith_preview_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("APP_LANGSMITH_PREVIEW_ENABLED"),
    )
    app_langsmith_preview_chars: int = Field(
        default=240,
        validation_alias=AliasChoices("APP_LANGSMITH_PREVIEW_CHARS"),
    )
    app_langsmith_doc_preview_chars: int = Field(
        default=320,
        validation_alias=AliasChoices("APP_LANGSMITH_DOC_PREVIEW_CHARS"),
    )
    app_langsmith_max_doc_previews: int = Field(
        default=6,
        validation_alias=AliasChoices("APP_LANGSMITH_MAX_DOC_PREVIEWS"),
    )
    app_log_level: str = Field(default="INFO", validation_alias=AliasChoices("APP_LOG_LEVEL"))
    app_log_preview_chars: int = Field(
        default=96,
        validation_alias=AliasChoices("APP_LOG_PREVIEW_CHARS"),
    )
    app_log_stage_timings: bool = Field(
        default=True,
        validation_alias=AliasChoices("APP_LOG_STAGE_TIMINGS"),
    )

    # ── Auth ──
    app_admin_username: str = "admin"
    # No defaults — startup fails loudly if these are missing from .env
    app_admin_password: str = Field(
        ...,
        validation_alias=AliasChoices("APP_ADMIN_PASSWORD"),
        description="Admin password. Must be set in .env — no insecure fallback.",
    )
    app_password_salt: str = Field(
        ...,
        validation_alias=AliasChoices("APP_PASSWORD_SALT"),
        description="PBKDF2 salt. Must be set in .env — no insecure fallback.",
    )

    # ── Paths ──
    upload_dir: Path = Path("uploads")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    def ensure_dirs(self):
        """Kısa: `ensure_dirs` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    @model_validator(mode="after")
    def reject_placeholder_secrets(self) -> "Settings":
        """Kısa: `reject_placeholder_secrets` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        placeholders = {
            "change-me",
            "change-me-in-production",
            "<strong-password>",
            "<random-hex-32>",
            "REPLACE_WITH_STRONG_ADMIN_PASSWORD",
            "REPLACE_WITH_RANDOM_SALT",
        }
        if self.app_admin_password.strip() in placeholders:
            raise ValueError("APP_ADMIN_PASSWORD must be changed from the example placeholder.")
        if self.app_password_salt.strip() in placeholders:
            raise ValueError("APP_PASSWORD_SALT must be changed from the example placeholder.")
        if self.langsmith_api_key.strip() and not self.app_langsmith_enabled:
            self.app_langsmith_enabled = True
        return self

    @property
    def llm_model(self) -> str:
        """Backward-compat alias: LLM model name used in log messages."""
        return self.llm_model_name

    # Backward-compat aliases for older code/docs that still say "vLLM".
    @property
    def vllm_server_url(self) -> str:  # pragma: no cover
        """Kısa: `vllm_server_url` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        return self.llm_server_url

    @property
    def vllm_model_name(self) -> str:  # pragma: no cover
        """Kısa: `vllm_model_name` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        return self.llm_model_name


settings = Settings()
settings.ensure_dirs()
