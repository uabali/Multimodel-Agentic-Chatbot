"""
Router node module for classification.

Routes queries into the correct downstream graph pathway (RAG vs Web vs Direct vs Vision).
"""

from __future__ import annotations

import logging
import time
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.state import AgentState
from src.agent.routing import keyword_route, is_web_query, needs_mcp_tools
from src.agent.prompts import ROUTER_SYSTEM_PROMPT
from src.config import settings
from src.agent.nodes.base import get_router_llm, observe_node
from src.utils.helpers import chat_history_turns_for_log

logger = logging.getLogger(__name__)


def _parse_route(text: str, default: str = "direct") -> str:
    """Extract 'rag', 'web', 'direct', or 'vision' from LLM router response text."""
    text_lower = text.lower().strip()
    if re.search(r"\brag\b", text_lower):
        return "rag"
    if re.search(r"\bweb\b", text_lower):
        return "web"
    if re.search(r"\bdirect\b", text_lower):
        return "direct"
    if re.search(r"\bvision\b", text_lower):
        return "vision"
    if re.search(r'"route"\s*:\s*"rag"', text_lower):
        return "rag"
    if re.search(r'"route"\s*:\s*"web"', text_lower):
        return "web"
    if re.search(r'"route"\s*:\s*"direct"', text_lower):
        return "direct"
    if re.search(r'"route"\s*:\s*"vision"', text_lower):
        return "vision"
    return default


async def router_node(state: AgentState) -> AgentState:
    """Classify the incoming user query into 'rag', 'direct', 'web', or 'vision'.

    Uses instant rules (images, forced search, source filter) -> keywords -> LLM.
    """
    t0 = time.perf_counter()
    question = state["question"]
    prior_messages = list(state.get("messages", []))
    q_len = len(question)
    session_uploads = state.get("session_uploads") or []

    # Path 0: Multimodal inputs immediately route to vision
    if state.get("image_data"):
        imgs = state["image_data"]
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.debug(
            "Router → vision [images=%d, mimes=%s, q_len=%d, t=0.00s]",
            len(imgs),
            ",".join(img.get("mime", "?") for img in imgs),
            q_len,
        )
        observe_node(
            "frappe.router_decision",
            state,
            outputs={"route": "vision", "route_reason": "image_data", "elapsed_ms": elapsed_ms},
            metadata={
                "route_reason": "image_data",
                "query_chars": q_len,
                "image_count": len(imgs),
                "upload_count": len(state.get("session_uploads") or []),
            },
            tags=["frappe", "router", "vision"],
        )
        return {**state, "route": "vision"}

    # Path 1: User explicitly requested live web search
    if state.get("force_web_search"):
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.debug(
            "Router → web [reason=force_web_search, uploads=%d, q_len=%d, t=%.3fs]",
            len(session_uploads),
            q_len,
            time.perf_counter() - t0,
        )
        observe_node(
            "frappe.router_decision",
            state,
            outputs={"route": "web", "route_reason": "force_web_search", "elapsed_ms": elapsed_ms},
            metadata={
                "route_reason": "force_web_search",
                "query_chars": q_len,
                "image_count": 0,
                "upload_count": len(session_uploads),
            },
            tags=["frappe", "router", "web"],
        )
        return {**state, "route": "web"}

    # Path 2: Direct source filter active -> always RAG
    if state.get("source_filter"):
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.debug(
            "Router → rag [reason=source_filter, file='%s', q_len=%d, t=%.3fs]",
            state["source_filter"],
            q_len,
            time.perf_counter() - t0,
        )
        observe_node(
            "frappe.router_decision",
            state,
            outputs={"route": "rag", "route_reason": "source_filter", "elapsed_ms": elapsed_ms},
            metadata={
                "route_reason": "source_filter",
                "query_chars": q_len,
                "image_count": 0,
                "upload_count": len(session_uploads),
            },
            tags=["frappe", "router", "rag"],
        )
        return {**state, "route": "rag"}

    # Path 3: Deterministic keyword-based bypasses
    fast_route = keyword_route(question, has_uploads=bool(session_uploads))
    if fast_route:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.debug(
            "Router → %s [reason=keyword, uploads=%d, q_len=%d, t=%.3fs]",
            fast_route,
            len(session_uploads),
            q_len,
            time.perf_counter() - t0,
        )
        observe_node(
            "frappe.router_decision",
            state,
            outputs={"route": fast_route, "route_reason": "keyword", "elapsed_ms": elapsed_ms},
            metadata={
                "route_reason": "keyword",
                "query_chars": q_len,
                "image_count": 0,
                "upload_count": len(session_uploads),
            },
            tags=["frappe", "router", fast_route],
        )
        return {**state, "route": fast_route}

    # Path 4: Session uploads routing bias
    if session_uploads:
        if is_web_query(question):
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            logger.debug(
                "Router → web [reason=web_override+uploads, uploads=%d, q_len=%d, t=%.3fs]",
                len(session_uploads),
                q_len,
                time.perf_counter() - t0,
            )
            observe_node(
                "frappe.router_decision",
                state,
                outputs={"route": "web", "route_reason": "web_override+uploads", "elapsed_ms": elapsed_ms},
                metadata={
                    "route_reason": "web_override+uploads",
                    "query_chars": q_len,
                    "image_count": 0,
                    "upload_count": len(session_uploads),
                },
                tags=["frappe", "router", "web"],
            )
            return {**state, "route": "web"}
        
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.debug(
            "Router → rag [reason=uploads_bias, uploads=%d, q_len=%d, t=%.3fs]",
            len(session_uploads),
            q_len,
            time.perf_counter() - t0,
        )
        observe_node(
            "frappe.router_decision",
            state,
            outputs={"route": "rag", "route_reason": "uploads_bias", "elapsed_ms": elapsed_ms},
            metadata={
                "route_reason": "uploads_bias",
                "query_chars": q_len,
                "image_count": 0,
                "upload_count": len(session_uploads),
            },
            tags=["frappe", "router", "rag"],
        )
        return {**state, "route": "rag"}

    # Path 5: LLM-based fallback routing for ambiguous queries
    logger.debug(
        "Router → LLM [prior_msgs=%d, q_len=%d, max_tokens=%d]",
        len(prior_messages),
        q_len,
        settings.router_max_tokens,
    )
    t_llm = time.perf_counter()
    llm = get_router_llm()
    try:
        messages_to_send = [SystemMessage(content=ROUTER_SYSTEM_PROMPT)]
        if prior_messages:
            messages_to_send.extend(prior_messages[-4:])
        messages_to_send.append(HumanMessage(content=question))
        response = await llm.ainvoke(messages_to_send)
        route = _parse_route(response.content)
    except Exception as exc:
        logger.warning("Router LLM başarısız → direct [err=%s]", exc)
        route = "direct"

    llm_elapsed = time.perf_counter() - t_llm
    total_elapsed = time.perf_counter() - t0
    logger.debug(
        "Router → %s [reason=llm, llm_t=%.3fs, total_t=%.3fs]",
        route,
        llm_elapsed,
        total_elapsed,
    )
    observe_node(
        "frappe.router_decision",
        state,
        outputs={
            "route": route,
            "route_reason": "llm",
            "llm_ms": round(llm_elapsed * 1000, 2),
            "elapsed_ms": round(total_elapsed * 1000, 2),
        },
        metadata={
            "route_reason": "llm",
            "query_chars": q_len,
            "image_count": 0,
            "upload_count": len(session_uploads),
            "router_max_tokens": settings.router_max_tokens,
        },
        tags=["frappe", "router", route],
    )
    return {**state, "route": route}
