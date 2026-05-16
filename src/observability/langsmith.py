"""LangSmith observability helpers with opt-in, sanitized tracing."""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.tracers.langchain import LangChainTracer
from langsmith import Client

from src.config import settings

logger = logging.getLogger(__name__)

_client: Client | None = None
_tracer: LangChainTracer | None = None

_CONTENT_KEYS = {
    "answer",
    "content",
    "documents",
    "generation",
    "image",
    "image_data",
    "messages",
    "original_question",
    "page_content",
    "prompt",
    "question",
    "text",
    "transcript",
    "vision_context",
}
_BINARY_KEYS = {"data", "image_url", "base64", "bytes"}
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
_KEY_VALUE_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password|authorization)\s*[:=]\s*['\"]?([A-Za-z0-9._~+/=-]{8,})"
)
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?90\s*)?(?:0?\s*)?5\d{2}[\s.-]?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}(?!\d)")
_TC_RE = re.compile(r"(?<!\d)\d{11}(?!\d)")
_LONG_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9_-]{32,}(?![A-Za-z0-9])")


def reset_langsmith_cache() -> None:
    """Clear cached LangSmith client/tracer; mainly used by tests."""
    global _client, _tracer
    _client = None
    _tracer = None


def stable_hash(value: Any, length: int = 12) -> str:
    """Return a stable short hash for identifiers we should not send raw."""
    if value is None:
        return ""
    text = str(value)
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def is_langsmith_enabled() -> bool:
    return bool(settings.app_langsmith_enabled and settings.langsmith_api_key.strip())


def _redact_text(text: str) -> str:
    if text.startswith("data:image/") or "base64," in text[:120]:
        return _redacted_summary(text, "binary")
    text = _EMAIL_RE.sub("[redacted-email]", text)
    text = _BEARER_RE.sub("Bearer [redacted-token]", text)
    text = _KEY_VALUE_RE.sub(lambda m: f"{m.group(1)}=[redacted-secret]", text)
    text = _PHONE_RE.sub("[redacted-phone]", text)
    text = _TC_RE.sub("[redacted-id]", text)
    text = _LONG_TOKEN_RE.sub("[redacted-token]", text)
    return text


def _redacted_summary(value: str, label: str) -> str:
    return f"[redacted-{label} chars={len(value)} sha256={stable_hash(value)}]"


def sanitize_payload(value: Any, key: str | None = None) -> Any:
    """Sanitize trace inputs/outputs without mutating application objects."""
    key_l = (key or "").lower()
    if isinstance(value, dict):
        return {str(k): sanitize_payload(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_payload(v, key) for v in value]
    if isinstance(value, bytes):
        return f"[redacted-bytes len={len(value)}]"
    if isinstance(value, str):
        if key_l in _BINARY_KEYS or "base64" in key_l or "image" in key_l:
            return _redacted_summary(value, "binary")
        if key_l in _CONTENT_KEYS:
            return _redacted_summary(value, key_l)
        return _redact_text(value)
    return value


def anonymize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    sanitized = sanitize_payload(payload)
    return sanitized if isinstance(sanitized, dict) else {"payload": sanitized}


def safe_preview(text: Any, max_chars: int | None = None) -> str:
    """Return a short, redacted, single-line preview for LangSmith inspection."""
    if not settings.app_langsmith_preview_enabled:
        return ""
    limit = max(0, int(max_chars or settings.app_langsmith_preview_chars))
    if limit <= 0:
        return ""
    preview = "" if text is None else str(text)
    preview = _redact_text(preview)
    preview = re.sub(r"\s+", " ", preview).strip()
    if len(preview) > limit:
        preview = preview[: max(0, limit - 1)].rstrip() + "…"
    return preview


def _doc_source(meta: dict[str, Any]) -> str:
    return str(meta.get("display_name") or meta.get("source_file") or meta.get("source") or "")


def _doc_chunk_id(doc: Any, meta: dict[str, Any]) -> str:
    source = _doc_source(meta)
    chunk_index = meta.get("chunk_index")
    if source and chunk_index is not None:
        return f"{source}#{chunk_index}"
    if source:
        return source
    return stable_hash(getattr(doc, "page_content", "") or "")


def summarize_documents(
    docs: list[Any] | tuple[Any, ...] | None,
    *,
    max_docs: int | None = None,
    preview_chars: int | None = None,
) -> list[dict[str, Any]]:
    """Summarize LangChain Documents without sending full page_content."""
    if not settings.app_langsmith_preview_enabled:
        return []
    limit = max(0, int(max_docs or settings.app_langsmith_max_doc_previews))
    if limit <= 0:
        return []
    char_limit = int(preview_chars or settings.app_langsmith_doc_preview_chars)
    summaries: list[dict[str, Any]] = []
    for doc in list(docs or [])[:limit]:
        meta = getattr(doc, "metadata", {}) or {}
        if not isinstance(meta, dict):
            meta = {}
        summary: dict[str, Any] = {
            "chunk_id": _doc_chunk_id(doc, meta),
            "source_preview": safe_preview(_doc_source(meta), 120),
            "page": meta.get("page"),
            "chunk_index": meta.get("chunk_index"),
            "chunk_type": meta.get("chunk_type"),
            "retrieval_score": meta.get("retrieval_score"),
            "rerank_score": meta.get("rerank_score"),
            "content_preview": safe_preview(getattr(doc, "page_content", "") or "", char_limit),
        }
        summaries.append({k: v for k, v in summary.items() if v not in (None, "", [])})
    return summaries


def summarize_retrieval_trace(
    trace: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
    *,
    max_items: int | None = None,
) -> dict[str, Any]:
    """Flatten retrieval explainability into CSV-friendly LangSmith fields."""
    entries = [e for e in list(trace or []) if isinstance(e, dict)]
    limit = max(0, int(max_items or settings.app_langsmith_max_doc_previews))
    top = entries[:limit] if limit else []

    def _score(value: Any) -> float | None:
        return round(float(value), 6) if isinstance(value, (int, float)) else None

    top_chunks: list[str] = []
    used_chunks: list[str] = []
    item_summaries: list[dict[str, Any]] = []
    for entry in top:
        chunk_id = str(entry.get("chunk_id") or "")
        hybrid = _score(entry.get("hybrid_score"))
        rerank = _score(entry.get("rerank_score"))
        used = bool(entry.get("used_in_context"))
        if chunk_id:
            top_chunks.append(chunk_id)
            if used:
                used_chunks.append(chunk_id)
        item_summaries.append({
            "chunk_id": chunk_id,
            "hybrid_score": hybrid,
            "rerank_score": rerank,
            "used_in_context": used,
        })
    return {
        "retrieval_trace_count": len(entries),
        "used_context_count": sum(1 for e in entries if e.get("used_in_context")),
        "top_chunks": " | ".join(top_chunks),
        "used_chunks": " | ".join(used_chunks),
        "items": item_summaries,
    }


def summarize_source_distribution(docs: list[Any] | tuple[Any, ...] | None) -> str:
    counts: Counter[str] = Counter()
    for doc in docs or []:
        meta = getattr(doc, "metadata", {}) or {}
        if isinstance(meta, dict):
            src = safe_preview(_doc_source(meta) or "?", 120)
            counts[src] += 1
    return " | ".join(f"{src}×{count}" for src, count in counts.most_common(8))


def get_langsmith_client() -> Client | None:
    """Return a cached LangSmith client, or None when tracing is disabled."""
    global _client
    if not is_langsmith_enabled():
        return None
    if _client is None:
        kwargs: dict[str, Any] = {
            "api_key": settings.langsmith_api_key,
            "api_url": settings.langsmith_endpoint,
        }
        if settings.langsmith_workspace_id:
            kwargs["workspace_id"] = settings.langsmith_workspace_id
        if settings.app_langsmith_redact:
            kwargs["anonymizer"] = anonymize_payload
        _client = Client(**kwargs)
    return _client


def get_langsmith_tracer() -> LangChainTracer | None:
    """Return a cached LangChain callback tracer, or None when disabled."""
    global _tracer
    client = get_langsmith_client()
    if client is None:
        return None
    if _tracer is None:
        _tracer = LangChainTracer(project_name=settings.langsmith_project, client=client)
    return _tracer


def _trace_context_value(trace_context: dict[str, Any] | None, key: str, default: Any = None) -> Any:
    if not trace_context:
        return default
    value = trace_context.get(key, default)
    return default if value is None else value


def build_common_metadata(
    *,
    question: str = "",
    source_filter: str = "",
    session_uploads: list[str] | None = None,
    image_data: list[dict] | None = None,
    input_type: str = "text",
    retrieval_strategy: str | None = None,
    use_rerank: bool | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    trace_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build safe metadata for LangSmith runs; never include raw prompt/files."""
    uploads = [u for u in (session_uploads or []) if u]
    session_id = _trace_context_value(trace_context, "session_id", "")
    user_id = _trace_context_value(trace_context, "user_id", "")
    channel = _trace_context_value(trace_context, "channel", "unknown")
    image_count = int(_trace_context_value(trace_context, "image_count", len(image_data or [])) or 0)
    history_turn_count = int(_trace_context_value(trace_context, "history_turn_count", 0) or 0)
    metadata: dict[str, Any] = {
        "app": "frappe-rag-agent",
        "env": settings.app_env,
        "channel": str(channel),
        "input_type": input_type,
        "input_summary": f"{input_type}; images={image_count}; uploads={len(uploads)}",
        "history_turn_count": history_turn_count,
        "has_image": bool(image_data) or image_count > 0,
        "image_count": image_count,
        "has_source_filter": bool(source_filter),
        "source_filter_hash": stable_hash(source_filter),
        "source_filter_preview": safe_preview(source_filter, 120),
        "session_upload_count": len(uploads),
        "session_upload_hashes": [stable_hash(u) for u in sorted(uploads)[:10]],
        "retrieval_strategy": retrieval_strategy or settings.retrieval_strategy,
        "use_rerank": bool(settings.use_rerank if use_rerank is None else use_rerank),
        "llm_backend": settings.llm_backend,
        "llm_model_name": settings.llm_model_name,
        "temperature": settings.chat_temperature if temperature is None else temperature,
        "max_tokens": settings.chat_max_tokens if max_tokens is None else max_tokens,
        "question_chars": len((question or "").strip()),
        "question_hash": stable_hash(question),
        "question_preview": safe_preview(question),
    }
    if session_id:
        metadata["session_id_hash"] = stable_hash(session_id)
    if user_id:
        metadata["user_id_hash"] = stable_hash(user_id)
    for key in ("attempt", "cache", "operation"):
        value = _trace_context_value(trace_context, key, "")
        if value:
            metadata[key] = str(value)
    return metadata


def build_graph_config(
    *,
    run_name: str,
    question: str,
    source_filter: str = "",
    session_uploads: list[str] | None = None,
    image_data: list[dict] | None = None,
    input_type: str = "text",
    retrieval_strategy: str | None = None,
    use_rerank: bool | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    trace_context: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build RunnableConfig for LangGraph with sanitized LangSmith metadata."""
    tracer = get_langsmith_tracer()
    if tracer is None:
        return None
    metadata = build_common_metadata(
        question=question,
        source_filter=source_filter,
        session_uploads=session_uploads,
        image_data=image_data,
        input_type=input_type,
        retrieval_strategy=retrieval_strategy,
        use_rerank=use_rerank,
        temperature=temperature,
        max_tokens=max_tokens,
        trace_context=trace_context,
    )
    channel = metadata.get("channel", "unknown")
    tags = [
        "frappe",
        "langgraph",
        f"env:{settings.app_env}",
        f"channel:{channel}",
        f"input:{input_type}",
    ]
    if metadata["has_image"]:
        tags.append("vision")
    if metadata["has_source_filter"] or metadata["session_upload_count"]:
        tags.append("rag-context")
    return {
        "run_name": run_name,
        "tags": tags,
        "metadata": metadata,
        "callbacks": [tracer],
    }


def record_observation(
    name: str,
    *,
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    run_type: str = "chain",
    error: str | None = None,
) -> str | None:
    """Create a small manual LangSmith run; never raise into app flow."""
    client = get_langsmith_client()
    if client is None:
        return None
    run_id = uuid.uuid4()
    safe_metadata = sanitize_payload(metadata or {})
    try:
        client.create_run(
            name=name,
            run_type=run_type,
            id=run_id,
            inputs=sanitize_payload(inputs or {}),
            project_name=settings.langsmith_project,
            start_time=datetime.now(timezone.utc),
            extra={"metadata": safe_metadata},
            tags=tags or ["frappe"],
        )
        client.update_run(
            run_id=run_id,
            outputs=sanitize_payload(outputs or {}),
            error=error,
            end_time=datetime.now(timezone.utc),
            extra={"metadata": safe_metadata},
            tags=tags or ["frappe"],
        )
        return str(run_id)
    except Exception as exc:
        logger.warning("LangSmith observation skipped for %s: %s", name, exc)
        return None


def record_semantic_cache_hit(
    *,
    question: str,
    cached_answer: str,
    cache_ctx: str,
    trace_context: dict[str, Any] | None = None,
) -> str | None:
    metadata = build_common_metadata(
        question=question,
        trace_context={**(trace_context or {}), "cache": "hit"},
    )
    return record_observation(
        "frappe.semantic_cache_hit",
        inputs={
            "question_hash": stable_hash(question),
            "question_chars": len((question or "").strip()),
            "cache_ctx_hash": stable_hash(cache_ctx),
        },
        outputs={"cached_answer_chars": len(cached_answer or "")},
        metadata=metadata,
        tags=["frappe", "semantic-cache", "cache-hit"],
    )


def record_ingest_observation(
    *,
    file_path: Path,
    result: dict[str, Any] | None,
    elapsed_s: float,
    error: str | None = None,
) -> str | None:
    suffix = file_path.suffix.lower()
    try:
        size_mb = round(file_path.stat().st_size / (1024 * 1024), 3)
    except OSError:
        size_mb = None
    metadata = {
        "app": "frappe-rag-agent",
        "env": settings.app_env,
        "operation": "ingest_file",
        "file_ext": suffix,
        "file_name_hash": stable_hash(file_path.name),
        "file_size_mb": size_mb,
        "visual_pdf_ingest_enabled": bool(suffix == ".pdf" and settings.pdf_visual_ingest_max_pages > 0),
    }
    outputs = {
        "status": "error" if error else (result or {}).get("status", "success"),
        "chunk_count": (result or {}).get("chunk_count", 0),
        "visual_chunk_count": (result or {}).get("visual_chunk_count", 0),
        "elapsed_ms": round(elapsed_s * 1000, 2),
    }
    return record_observation(
        "frappe.ingest_file",
        inputs={
            "file_ext": suffix,
            "file_name_hash": stable_hash(file_path.name),
            "file_size_mb": size_mb,
        },
        outputs=outputs,
        metadata=metadata,
        tags=["frappe", "ingest", f"file:{suffix or 'unknown'}"],
        error=error,
    )
