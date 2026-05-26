"""
Vision node module.

Processes multimodal image inputs using Gemma 4, supporting three pathways:
- vision (pure visual analysis)
- vision_rag (visual analysis combined with RAG documents)
- vision_search (visual analysis combined with live web results)
"""

from __future__ import annotations

import logging
import time
from typing import Any

import chainlit as cl
from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.state import AgentState
from src.agent.prompts import select_vision_prompt
from src.config import settings
from src.agent.nodes.base import get_rag_llm, observe_node, coerce_llm_text

logger = logging.getLogger(__name__)


def _build_vision_content_parts(image_data: list[dict], text: str) -> list[dict]:
    """Compile raw images and text into LangChain multimodal content blocks."""
    parts: list[dict] = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:{img['mime']};base64,{img['base64']}"},
        }
        for img in image_data
    ]
    parts.append({"type": "text", "text": text})
    return parts


async def vision_node(state: AgentState) -> AgentState:
    """Run pure visual analysis on uploaded images using the Gemma 4 multimodal API."""
    t0 = time.perf_counter()
    question = state["question"]
    image_data = state.get("image_data") or []
    
    if not image_data:
        logger.warning("Vision: image_data is empty")
        return {**state, "generation": "Görsel verisi bulunamadı."}

    logger.debug("Vision: processing images [count=%d, q_len=%d]", len(image_data), len(question))
    
    prompt = select_vision_prompt(question, image_data)
    parts = _build_vision_content_parts(image_data, question)

    llm = get_rag_llm(temperature=0.4, max_tokens=settings.rag_max_tokens)
    t_llm = time.perf_counter()
    try:
        response = await llm.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content=parts),
        ])
        generation = coerce_llm_text(response)
        status = "success"
        error_str = None
    except Exception as exc:
        logger.error("Vision LLM failed: %s", exc)
        generation = "Görsel analizi sırasında bir hata oluştu."
        status = "error"
        error_str = str(exc)

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    llm_ms = round((time.perf_counter() - t_llm) * 1000, 2) if status == "success" else None

    observe_node(
        "frappe.vision_result",
        state,
        outputs={
            "generation": generation,
            "status": status,
            "latency_ms_by_stage": {
                "llm": llm_ms,
                "total": elapsed_ms,
            },
        },
        metadata={"vision_mode": "pure_vision", "image_count": len(image_data)},
        tags=["frappe", "vision", status],
        error=error_str,
    )
    return {**state, "generation": generation}


async def vision_rag_node(state: AgentState) -> AgentState:
    """Run visual analysis on images and prepare structural vision context for RAG."""
    t0 = time.perf_counter()
    question = state["question"]
    image_data = state.get("image_data") or []

    if not image_data:
        return state

    logger.debug("Vision RAG: generating visual context for RAG pipeline")
    
    prompt = (
        "Sen gelişmiş bir görsel analizörsün. Bu görseli/görselleri son derece detaylı bir şekilde metne ve "
        "tablosal/yapısal verilere dök. Çıkarabildiğin tüm sayısal verileri, isimleri, tabloları, grafikleri ve "
        "metinleri eksiksiz yaz. Bu metin, daha sonra RAG veritabanı aramasıyla birleştirilerek nihai soruyu "
        "yanıtlamak üzere kullanılacak. Sadece görselden çıkardığın ham bilgileri ver, yorum yapma."
    )
    parts = _build_vision_content_parts(image_data, question)

    llm = get_rag_llm(temperature=0.0, max_tokens=1024)
    try:
        response = await llm.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content=parts),
        ])
        vision_context = coerce_llm_text(response)
        logger.debug("Vision RAG: context prepared [%d chars, t=%.3fs]", len(vision_context), time.perf_counter() - t0)
    except Exception as exc:
        logger.error("Vision RAG context generation failed: %s", exc)
        vision_context = ""

    return {**state, "vision_context": vision_context}


async def vision_search_node(state: AgentState) -> AgentState:
    """Combine multimodal image analysis with live web search results.

    Retrieves web results and merges them with image content before generation.
    """
    t0 = time.perf_counter()
    question = state["question"]
    image_data = state.get("image_data") or []

    if not image_data:
        return state

    logger.debug("Vision Search: executing real-time web search for multimodal request")
    
    # Import the lazily managed web search components to avoid circular references
    from src.agent.nodes.web_search import get_web_search_service, web_docs_from_result

    web_docs = []
    vision_context = ""
    service = get_web_search_service()

    async with cl.Step(name="Görsel + Web Arama", type="tool") as step:
        if service:
            step.output = f"Web araması yapılıyor: {question}"
            result = await service.search(question)
            if result:
                web_docs = web_docs_from_result(result, query=question)
                step.output = f"Başarılı! {len(web_docs)} web kaynağı bulundu."
            else:
                step.output = "Web araması boş sonuç döndürdü."
        else:
            step.output = "Web search service unavailable."

    return {**state, "vision_context": vision_context, "documents": web_docs}
