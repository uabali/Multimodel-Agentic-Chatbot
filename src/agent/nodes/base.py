"""
Base shared nodes utilities and registries for LangGraph.

Provides centralized model loaders, logging wrappers, and historical
message token management to avoid code duplication across node files.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agent.state import AgentState
from src.config import settings
from src.observability.app_logging import log_event, stage_timings_enabled, detailed_trace_enabled
from src.rag.llm import count_message_tokens
from src.utils.helpers import history_turn_count

logger = logging.getLogger(__name__)

_router_llm_cache = None
_rag_llm_cache: dict[tuple, Any] = {}
_chat_llm_cache: dict[tuple, Any] = {}
_RAG_LLM_CACHE_MAXSIZE = 32

_RERANKER_FAILED = object()


class RerankerRegistry:
    """Lazy load and cache cross-encoder reranking model instance (Thread-Safe)."""

    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> Any:
        """Retrieve the cached CrossEncoder model, initializing it if necessary."""
        if cls._instance is _RERANKER_FAILED:
            return None
        if cls._instance is not None:
            return cls._instance
        if not settings.use_rerank:
            return None
        with cls._lock:
            if cls._instance is _RERANKER_FAILED:
                return None
            if cls._instance is not None:
                return cls._instance
            try:
                from src.rag.reranker import create_reranker

                model_name = "fast" if settings.rerank_fast_mode else settings.reranker_model
                cls._instance = create_reranker(
                    model_name=model_name,
                    device=settings.reranker_device,
                )
            except Exception as exc:
                logger.warning("Reranker yüklenemedi (devre dışı): %s", exc)
                cls._instance = _RERANKER_FAILED
        return cls._instance if cls._instance is not _RERANKER_FAILED else None


def get_router_llm() -> Any:
    """Get the cached router model (0 temperature, minimal tokens)."""
    global _router_llm_cache
    if _router_llm_cache is None:
        from src.rag.llm import create_vllm_llm
        _router_llm_cache = create_vllm_llm(temperature=0.0, max_tokens=settings.router_max_tokens)
    return _router_llm_cache


def get_rag_llm(temperature: float = 0.0, max_tokens: int | None = None) -> Any:
    """Get or create cached factual RAG LLM instances based on temp/tokens."""
    import sys
    nodes_mod = sys.modules.get("src.agent.nodes")
    if nodes_mod is not None:
        current_val = getattr(nodes_mod, "_get_rag_llm", None)
        if current_val is not None and not getattr(current_val, "_is_original", False):
            return current_val(temperature, max_tokens)

    if temperature == 0.0 and max_tokens is None:
        from src.rag.llm import get_rag_llm as get_default_rag_llm
        return get_default_rag_llm()
    
    key = (temperature, max_tokens)
    if key not in _rag_llm_cache:
        if len(_rag_llm_cache) >= _RAG_LLM_CACHE_MAXSIZE:
            _rag_llm_cache.pop(next(iter(_rag_llm_cache)))
        from src.rag.llm import create_vllm_llm
        _rag_llm_cache[key] = create_vllm_llm(
            temperature=temperature,
            max_tokens=max_tokens or settings.rag_max_tokens,
        )
    return _rag_llm_cache[key]


def get_chat_llm(temperature: float | None = None, max_tokens: int | None = None) -> Any:
    """Get or create cached conversational LLM client based on temp/tokens."""
    if temperature is None and max_tokens is None:
        from src.rag.llm import get_chat_llm as get_default_chat_llm
        return get_default_chat_llm()
    
    key = (
        temperature if temperature is not None else settings.chat_temperature,
        max_tokens or settings.chat_max_tokens
    )
    if key not in _chat_llm_cache:
        if len(_chat_llm_cache) >= _RAG_LLM_CACHE_MAXSIZE:
            _chat_llm_cache.pop(next(iter(_chat_llm_cache)))
        from src.rag.llm import create_vllm_llm
        _chat_llm_cache[key] = create_vllm_llm(
            temperature=float(key[0]),
            max_tokens=int(key[1]),
        )
    return _chat_llm_cache[key]


def get_agent_llm() -> Any:
    """Get tool-calling ReAct agent LLM profile."""
    from src.rag.llm import get_agent_llm as get_default_agent_llm
    return get_default_agent_llm()


def reset_nodes_llm_cache() -> None:
    """Reset all cached LLM client singletons (called when settings change)."""
    global _router_llm_cache
    _router_llm_cache = None
    _rag_llm_cache.clear()
    _chat_llm_cache.clear()


def select_recent_history(messages: list, *, mode: str = "rag") -> list:
    """Filter chat history to fit inside LLM context window while preserving System Messages."""
    max_messages = settings.history_max_messages_rag if mode == "rag" else settings.history_max_messages_chat
    token_budget = settings.history_token_budget_rag if mode == "rag" else settings.history_token_budget_chat

    system_messages = [m for m in messages if isinstance(m, SystemMessage)]
    chat_messages = [m for m in messages if not isinstance(m, SystemMessage)]
    selected = list(chat_messages[-max_messages:])

    total = count_message_tokens(selected)
    while total > token_budget and len(selected) > 2:
        drop_count = (
            2
            if len(selected) >= 2
            and isinstance(selected[0], HumanMessage)
            and isinstance(selected[1], AIMessage)
            else 1
        )
        for _ in range(drop_count):
            selected.pop(0)
        total = count_message_tokens(selected)
    while total > token_budget and selected:
        selected.pop(0)
        total = count_message_tokens(selected)
    while selected and isinstance(selected[0], AIMessage):
        selected.pop(0)

    return system_messages[:1] + selected


_OBSERVATION_LOG_EVENTS = {
    "frappe.router_decision": "router",
    "frappe.retriever_result": "retriever",
    "frappe.grader_decision": "grader",
    "frappe.generator_result": "generator",
    "frappe.direct_response_result": "generator",
    "frappe.vision_result": "generator",
}


def _float_or_none(value: Any) -> float | None:
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None


def _sid_for_log(state: AgentState) -> str:
    return str(state.get("thread_id") or "?")[:8] or "?"


def _log_node_stage(name: str, state: AgentState, outputs: dict | None, error: str | None) -> None:
    event = _OBSERVATION_LOG_EVENTS.get(name)
    if not event:
        return
    outputs = outputs or {}
    latency = outputs.get("latency_ms_by_stage")
    if not isinstance(latency, dict):
        latency = {}

    route = str(outputs.get("route") or state.get("route") or "?")
    fields: dict[str, Any] = {
        "sid": _sid_for_log(state),
        "turn_id": state.get("turn_id", ""),
        "route": route,
        "input_type": state.get("input_type", "text"),
        "docs": outputs.get("document_count", len(state.get("documents") or [])),
    }
    if event == "router" and outputs.get("route"):
        fields["route"] = outputs["route"]
    if event == "generator" and outputs.get("used_chunks"):
        fields["used_chunks"] = outputs.get("used_chunks")
        
    # Senior developer logging metrics extraction
    if event == "retriever":
        fields["strategy"] = state.get("retrieval_strategy") or "?"
    elif event == "grader":
        fields["relevance"] = outputs.get("relevance") or state.get("relevance") or "?"
        fields["grader_reason"] = outputs.get("grader_reason") or state.get("grader_reason") or "?"
    elif event == "generator":
        fields["retry_path"] = outputs.get("retry_path") or state.get("retry_path") or "primary"
        
    if error:
        fields["error_type"] = type(error).__name__ if not isinstance(error, str) else error.split(":", 1)[0]

    if stage_timings_enabled():
        total_ms = _float_or_none(latency.get("total") or outputs.get("elapsed_ms"))
        if event == "retriever":
            fields["rag_ms"] = total_ms
        elif event == "generator":
            fields["llm_ms"] = _float_or_none(latency.get("llm")) or total_ms
            fields["total_ms"] = total_ms
        else:
            fields["total_ms"] = total_ms

    log_event(logger, event, **fields)


def observe_node(
    name: str,
    state: AgentState,
    *,
    inputs: dict | None = None,
    outputs: dict | None = None,
    metadata: dict | None = None,
    tags: list[str] | None = None,
    error: str | None = None,
) -> None:
    """Record observation details to LangSmith and output unified log_event lines."""
    import sys
    nodes_mod = sys.modules.get("src.agent.nodes")
    if nodes_mod is not None:
        current_val = getattr(nodes_mod, "_observe_node", None)
        if current_val is not None and not getattr(current_val, "_is_original", False):
            return current_val(name, state, inputs=inputs, outputs=outputs, metadata=metadata, tags=tags, error=error)

    _log_node_stage(name, state, outputs, error)
    try:
        from src.observability.langsmith import record_observation, safe_preview, stable_hash

        question = state.get("original_question") or state.get("question", "")
        docs = state.get("documents") or []
        common = {
            "app": "frappe-rag-agent",
            "env": settings.app_env,
            "question_hash": stable_hash(question),
            "question_preview": safe_preview(question),
            "route": state.get("route", ""),
            "input_type": state.get("input_type", "text"),
            "history_turn_count": history_turn_count(list(state.get("messages") or [])),
            "session_upload_count": len(state.get("session_uploads") or []),
            "has_source_filter": bool(state.get("source_filter")),
            "source_filter_hash": stable_hash(state.get("source_filter", "")),
            "source_filter_preview": safe_preview(state.get("source_filter", ""), 120),
            "document_count": len(docs),
        }
        record_observation(
            name,
            inputs=inputs,
            outputs=outputs,
            metadata={**common, **(metadata or {})},
            tags=tags or ["frappe", "node", name.rsplit(".", 1)[-1]],
            error=error,
        )
    except Exception:
        logger.debug("LangSmith node observation failed for %s", name, exc_info=True)


def coerce_llm_text(response: Any) -> str:
    """OpenAI-compatible backends sometimes place text outside `content`."""
    content = getattr(response, "content", "")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                val = item.get("text") or item.get("content")
                if isinstance(val, str):
                    parts.append(val)
        text = "".join(parts)
    else:
        text = str(content or "")

    if text.strip():
        return text.strip()

    for attr in ("additional_kwargs", "response_metadata"):
        data = getattr(response, attr, None) or {}
        if not isinstance(data, dict):
            continue
        for key in ("reasoning_content", "reasoning", "content", "text"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return ""

