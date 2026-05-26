"""
Grader node module.

Evaluates retrieval document relevance against the user's question.
Implements a fast heuristic confidence checker and calls LLM grading as a fallback.
"""

from __future__ import annotations

import logging
import time
import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.state import AgentState
from src.config import settings
from src.agent.nodes.base import get_rag_llm, observe_node, coerce_llm_text

logger = logging.getLogger(__name__)

_GRADER_REASONS = {"sufficient", "irrelevant", "partial", "insufficient_context", "needs_live_data"}


def _parse_grader_payload(text: str) -> tuple[str, str]:
    """Parse JSON or fallback regex from grader LLM response text."""
    raw = (text or "").strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        relevant = str(payload.get("relevant", "")).lower().strip()
        reason = str(payload.get("reason", "")).lower().strip()
        if relevant not in {"yes", "no"}:
            relevant = "no"
        if relevant == "yes":
            return "yes", "sufficient"
        if reason not in _GRADER_REASONS:
            reason = "insufficient_context"
        return "no", reason

    relevance = _parse_yes_no(raw)
    reason = _parse_grader_reason(raw) if relevance == "no" else "sufficient"
    return relevance, reason


def _parse_yes_no(text: str, default: str = "no") -> str:
    """Extract 'yes' or 'no' from raw response using regex."""
    raw = (text or "").strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        relevant = str(payload.get("relevant", "")).lower().strip()
        if relevant in {"yes", "no"}:
            return relevant

    text_lower = raw.lower()
    if re.search(r'"relevant"\s*:\s*"yes"', text_lower):
        return "yes"
    if re.search(r'"relevant"\s*:\s*"no"', text_lower):
        return "no"
    if re.search(r"\byes\b", text_lower):
        return "yes"
    if re.search(r"\bno\b", text_lower):
        return "no"
    return default


def _parse_grader_reason(text: str) -> str:
    """Extract granular relevance reason from LLM response."""
    raw = (text or "").strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        relevant = str(payload.get("relevant", "")).lower().strip()
        reason = str(payload.get("reason", "")).lower().strip()
        if relevant == "yes":
            return "sufficient"
        return reason if reason in _GRADER_REASONS else "insufficient_context"

    text_lower = raw.lower()
    if "insufficient_context" in text_lower:
        return "insufficient_context"
    if "partial" in text_lower:
        return "partial"
    if "needs_live_data" in text_lower:
        return "needs_live_data"
    if "irrelevant" in text_lower:
        return "irrelevant"
    if "sufficient" in text_lower:
        return "sufficient"
    return "insufficient_context"


async def grader_node(state: AgentState) -> AgentState:
    """Grade retrieved document relevance.

    High confidence (estimate_confidence >= 0.75) immediately passes.
    Low confidence (< 0.08) immediately fails.
    Mid confidence runs the LLM grader to safely verify.
    """
    t0 = time.perf_counter()
    from src.rag.retriever import estimate_confidence

    question = state["question"]
    documents = state.get("documents", [])

    if not documents:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.debug("Grader: no_docs → relevance=no [t=%.3fs]", time.perf_counter() - t0)
        observe_node(
            "frappe.grader_decision",
            state,
            outputs={
                "relevance": "no",
                "grader_reason": "insufficient_context",
                "mode": "no_docs",
                "document_count": 0,
                "latency_ms_by_stage": {"total": elapsed_ms},
            },
            metadata={"grader_mode": "no_docs", "grader_confidence": None},
            tags=["frappe", "grader", "no"],
        )
        return {**state, "relevance": "no", "grader_reason": "insufficient_context", "refusal_mode": True}

    if state.get("source_filter") or state.get("session_uploads"):
        from src.agent.routing import is_web_query

        original_q = state.get("original_question") or question
        if is_web_query(original_q):
            pass  # Web queries on uploads need LLM to check if live data is missing
        else:
            confidence = estimate_confidence(question, documents)
            if confidence >= settings.grader_conf_high:
                elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
                logger.debug(
                    "Grader: relevance=yes [mode=file_high_conf, conf=%.3f>=%.3f, docs=%d, t=%.3fs]",
                    confidence, settings.grader_conf_high, len(documents), time.perf_counter() - t0,
                )
                observe_node(
                    "frappe.grader_decision",
                    state,
                    outputs={
                        "relevance": "yes",
                        "grader_reason": "sufficient",
                        "mode": "file_high_conf",
                        "confidence": confidence,
                        "high_threshold": settings.grader_conf_high,
                        "document_count": len(documents),
                        "latency_ms_by_stage": {"total": elapsed_ms},
                    },
                    metadata={
                        "grader_mode": "file_high_conf",
                        "grader_confidence": confidence,
                        "grader_high_threshold": settings.grader_conf_high,
                    },
                    tags=["frappe", "grader", "yes"],
                )
                return {**state, "relevance": "yes", "grader_reason": "sufficient"}
            logger.debug(
                "Grader: file context requires LLM [conf=%.3f<%.3f, docs=%d]",
                confidence, settings.grader_conf_high, len(documents),
            )

        top_docs = documents[:settings.grader_max_docs]
        doc_texts = "\n---\n".join(doc.page_content for doc in top_docs)
        doc_chars = sum(len(d.page_content) for d in top_docs)
        llm = get_rag_llm(temperature=0.0)
        try:
            t_llm = time.perf_counter()
            from src.agent.prompts import GRADER_SYSTEM_PROMPT
            response = await llm.ainvoke([
                SystemMessage(content=GRADER_SYSTEM_PROMPT),
                HumanMessage(content=f"Question: {question}\n\nDocuments:\n{doc_texts}"),
            ])
            response_text = coerce_llm_text(response)
            relevance, reason = _parse_grader_payload(response_text)
            logger.debug(
                "Grader: relevance=%s reason=%s [mode=file_llm, docs=%d/%d, doc_chars=%d, llm_t=%.3fs, t=%.3fs]",
                relevance, reason or "-", len(top_docs), len(documents), doc_chars,
                time.perf_counter() - t_llm, time.perf_counter() - t0,
            )
            llm_ms = round((time.perf_counter() - t_llm) * 1000, 2)
        except Exception as exc:
            logger.warning("Grader: llm_error → yes [err=%s, t=%.3fs]", exc, time.perf_counter() - t0)
            relevance, reason = "yes", ""
            llm_ms = None
        observe_node(
            "frappe.grader_decision",
            state,
            outputs={
                "relevance": relevance,
                "grader_reason": reason,
                "mode": "file_llm",
                "document_count": len(documents),
                "graded_doc_count": len(top_docs),
                "graded_doc_chars": doc_chars,
                "latency_ms_by_stage": {
                    "llm": llm_ms,
                    "total": round((time.perf_counter() - t0) * 1000, 2),
                },
            },
            metadata={"grader_mode": "file_llm", "grader_confidence": None},
            tags=["frappe", "grader", relevance],
        )
        return {**state, "relevance": relevance, "grader_reason": reason}

    confidence = estimate_confidence(question, documents)
    retrieval_gate = state.get("retrieval_gate") or ""
    scoped = bool(state.get("source_filter") or state.get("session_uploads"))

    if retrieval_gate == "weak" and not scoped:
        if confidence < settings.grader_conf_high:
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            logger.debug(
                "Grader: relevance=no [mode=weak_dense_gate, conf=%.3f<%.3f, docs=%d, t=%.3fs]",
                confidence, settings.grader_conf_high, len(documents), time.perf_counter() - t0,
            )
            observe_node(
                "frappe.grader_decision",
                state,
                outputs={
                    "relevance": "no",
                    "grader_reason": "insufficient_context",
                    "mode": "weak_dense_gate",
                    "confidence": confidence,
                    "high_threshold": settings.grader_conf_high,
                    "document_count": len(documents),
                    "latency_ms_by_stage": {"total": elapsed_ms},
                },
                metadata={
                    "grader_mode": "weak_dense_gate",
                    "grader_confidence": confidence,
                    "retrieval_gate": retrieval_gate,
                },
                tags=["frappe", "grader", "no", "gate:weak"],
            )
            return {
                **state,
                "relevance": "no",
                "grader_reason": "insufficient_context",
                "refusal_mode": True,
            }
    elif confidence >= settings.grader_conf_high:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.debug(
            "Grader: relevance=yes [mode=high_conf, conf=%.3f>=%.3f, docs=%d, t=%.3fs]",
            confidence, settings.grader_conf_high, len(documents), time.perf_counter() - t0,
        )
        observe_node(
            "frappe.grader_decision",
            state,
            outputs={
                "relevance": "yes",
                "grader_reason": "sufficient",
                "mode": "high_conf",
                "confidence": confidence,
                "high_threshold": settings.grader_conf_high,
                "document_count": len(documents),
                "latency_ms_by_stage": {"total": elapsed_ms},
            },
            metadata={
                "grader_mode": "high_conf",
                "grader_confidence": confidence,
                "grader_high_threshold": settings.grader_conf_high,
            },
            tags=["frappe", "grader", "yes"],
        )
        return {**state, "relevance": "yes", "grader_reason": "sufficient"}

    if confidence < settings.grader_conf_low:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.debug(
            "Grader: relevance=no [mode=low_conf, conf=%.3f<%.3f, docs=%d, t=%.3fs]",
            confidence, settings.grader_conf_low, len(documents), time.perf_counter() - t0,
        )
        observe_node(
            "frappe.grader_decision",
            state,
            outputs={
                "relevance": "no",
                "grader_reason": "insufficient_context",
                "mode": "low_conf",
                "confidence": confidence,
                "low_threshold": settings.grader_conf_low,
                "document_count": len(documents),
                "latency_ms_by_stage": {"total": elapsed_ms},
            },
            metadata={
                "grader_mode": "low_conf",
                "grader_confidence": confidence,
                "grader_low_threshold": settings.grader_conf_low,
            },
            tags=["frappe", "grader", "no"],
        )
        return {**state, "relevance": "no", "grader_reason": "insufficient_context", "refusal_mode": True}

    top_docs = documents[:settings.grader_max_docs]
    doc_texts = "\n---\n".join(doc.page_content for doc in top_docs)
    doc_chars = sum(len(d.page_content) for d in top_docs)
    llm = get_rag_llm(temperature=0.0)
    try:
        t_llm = time.perf_counter()
        from src.agent.prompts import GRADER_SYSTEM_PROMPT
        response = await llm.ainvoke([
            SystemMessage(content=GRADER_SYSTEM_PROMPT),
            HumanMessage(content=f"Question: {question}\n\nDocuments:\n{doc_texts}"),
        ])
        response_text = coerce_llm_text(response)
        relevance, reason = _parse_grader_payload(response_text)
        logger.debug(
            "Grader: relevance=%s reason=%s [mode=mid_conf, conf=%.3f, docs=%d/%d, doc_chars=%d, llm_t=%.3fs, t=%.3fs]",
            relevance, reason or "-", confidence, len(top_docs), len(documents), doc_chars,
            time.perf_counter() - t_llm, time.perf_counter() - t0,
        )
        llm_ms = round((time.perf_counter() - t_llm) * 1000, 2)
    except Exception as exc:
        logger.warning("Grader: llm_error → no [err=%s, t=%.3fs]", exc, time.perf_counter() - t0)
        relevance, reason = "no", "insufficient_context"
        llm_ms = None

    observe_node(
        "frappe.grader_decision",
        state,
        outputs={
            "relevance": relevance,
            "grader_reason": reason,
            "mode": "mid_conf",
            "confidence": confidence,
            "document_count": len(documents),
            "graded_doc_count": len(top_docs),
            "graded_doc_chars": doc_chars,
            "latency_ms_by_stage": {
                "llm": llm_ms,
                "total": round((time.perf_counter() - t0) * 1000, 2),
            },
        },
        metadata={"grader_mode": "mid_conf", "grader_confidence": confidence},
        tags=["frappe", "grader", relevance],
    )
    return {**state, "relevance": relevance, "grader_reason": reason}
