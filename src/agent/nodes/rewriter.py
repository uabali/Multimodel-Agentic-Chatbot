"""
Rewriter node module.

Optimizes incoming user queries to improve retrieval relevance inside vector stores.
"""

from __future__ import annotations

import asyncio
import logging
import time
import re

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.state import AgentState
from src.agent.prompts import REWRITER_SYSTEM_PROMPT
from src.agent.nodes.base import get_rag_llm
from src.config import settings

logger = logging.getLogger(__name__)

_FOLLOW_UP_MARKERS: frozenset[str] = frozenset({
    "bunu", "buna", "bunda", "bunun", "bunları", "bunlari", "bununla",
    "önceki", "onceki", "bahsettiğin", "bahsettigin",
    "söylediğin", "soyledigin", "yukarıdaki", "yukaridaki",
    "this", "that", "it", "these", "those", "above", "previous",
})

_QUESTION_WORDS: frozenset[str] = frozenset({
    "ne", "nedir", "nasıl", "nasil", "neden", "kim", "hangi",
    "kaç", "kac", "nerede", "ne zaman",
    "what", "how", "why", "who", "which", "when", "where",
})

_TECHNICAL_ENTITY_RE = re.compile(
    r"\b("
    r"PostgreSQL|Milvus|Qdrant|Elasticsearch|MongoDB|Redis|Cassandra|"
    r"BM25|TF-?IDF|HNSW|IVF|PQ|"
    r"BERT|GPT|Gemma|LLaMA|Mistral|"
    r"bge-m3|bge-reranker|e5-large|"
    r"Atlas-\d+|Orion|Aurora|"
    r"re-?rank|embedding|vector|chunk|retrieval|latency|inference|"
    r"\d+\s*(?:boyut|dimension|parametre|parameter|milyon|million|petabyte|TB|GB|MB|ms)"
    r")\b",
    re.IGNORECASE,
)


def _should_skip_rewrite(question: str, prior_messages: list) -> bool:
    """Decide whether to skip LLM rewrite (~6s savings) for simple or entity-rich queries."""
    words = question.split()
    q_lower = question.lower()

    if prior_messages:
        q_words = set(re.findall(r"\b\w+\b", q_lower))
        if q_words & _FOLLOW_UP_MARKERS:
            return False

    if _TECHNICAL_ENTITY_RE.search(question):
        return True

    if len(words) <= 8:
        tokens = set(re.findall(r"[a-zA-ZÜüÖöÇçŞşİıĞğ]+", q_lower))
        if tokens & _QUESTION_WORDS or "?" in question:
            return True

    return False


async def rewriter_node(state: AgentState) -> AgentState:
    """Rewrite ambiguous questions based on history context to optimize semantic retrieval.

    Bypasses LLM rewriting entirely for short, clear queries or technical keyphrases.
    """
    t0 = time.perf_counter()
    question = state["question"]
    prior_messages = list(state.get("messages", []))

    if _should_skip_rewrite(question, prior_messages):
        logger.debug(
            "Rewriter: skip [reason=short_clear, q_len=%d, t=%.3fs]",
            len(question),
            time.perf_counter() - t0,
        )
        return state

    llm = get_rag_llm(temperature=0.0)
    messages_to_send = [SystemMessage(content=REWRITER_SYSTEM_PROMPT)]
    memory_ctx = (state.get("memory_context") or "").strip()
    if memory_ctx:
        messages_to_send.append(
            SystemMessage(content=f"Thread memory (rewrite için bağlam):\n{memory_ctx}")
        )
    if prior_messages:
        messages_to_send.extend(prior_messages[-2:])
    messages_to_send.append(HumanMessage(content=question))

    # Pre-embed the original query to warm vector search cache in parallel with LLM generation
    async def _warm_embed_cache() -> None:
        try:
            from src.rag.vectorstore import _cached_embed_query
            await asyncio.to_thread(_cached_embed_query, question)
        except Exception:
            pass

    embed_task = asyncio.create_task(_warm_embed_cache())
    response = await llm.ainvoke(messages_to_send)
    await embed_task  # Ensure embedding cache is warm
    rewritten = response.content.strip()

    # Hallucination check on rewriter output
    _ANSWER_MARKERS = settings.answer_hallucination_markers
    is_hallucination = (
        len(rewritten) > 250
        or "\n" in rewritten
        or any(m in rewritten.lower() for m in _ANSWER_MARKERS)
    )
    if is_hallucination:
        logger.warning(
            "Rewriter: hallucination → original kept [rewritten_len=%d, t=%.3fs]",
            len(rewritten),
            time.perf_counter() - t0,
        )
        return state

    logger.debug(
        "Rewriter: rewritten [%d→%dch, prior=%d, t=%.3fs]",
        len(question),
        len(rewritten),
        len(prior_messages),
        time.perf_counter() - t0,
    )
    return {**state, "question": rewritten}
