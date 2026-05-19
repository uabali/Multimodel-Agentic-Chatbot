"""
LangGraph agent node'ları — SOLID prensiplerine uygun, temiz mimari.

Mimari kararlar:
─────────────────────────────────────────────────────────────────────────────
SRP  ─ Her node fonksiyonu TEK bir graph adımından sorumludur.
     ─ İş mantığı (routing, web search, formatter) ayrı modüllere taşındı:
         src/agent/routing.py   → keyword tabanlı rota tespiti
         src/agent/web_search.py → web arama provider zinciri + formatter

OCP  ─ Web search provider eklemek nodes.py'ı DEĞİŞTİRMEZ;
       sadece WebSearchService.from_settings() içinde yeni provider eklenir.

LSP  ─ Tüm node fonksiyonları (AgentState) → AgentState imzasına uyar.

ISP  ─ Her node sadece ihtiyaç duyduğu state alanlarına erişir.

DIP  ─ Node'lar doğrudan somut LLM yaratmaz; llm.py fabrika fonksiyonlarına
       bağlıdır. Reranker da `_RerankerRegistry` üzerinden alınır.

Modern LangChain kullanımı:
─────────────────────────────────────────────────────────────────────────────
 ✔  `retriever.invoke(query)` → deprecated get_relevant_documents() kaldırıldı
 ✔  ChatPromptTemplate.from_messages() yerine doğrudan liste mesajlar kullanıldı
    (vLLM endpoint chat_template_kwargs gerektirdiğinden ExtraBody ile uyumlu)
 ✔  Router / Grader LLM yanıtları regex parse ile işleniyor (structured output
    vLLM grammar support gerektirdiğinden kullanılmıyor)
"""

from __future__ import annotations

import asyncio
import ast
import datetime
import json
import operator
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

import chainlit as cl
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agent.state import AgentState
from src.agent.routing import (
    keyword_route,
    is_direct_support_query,
    is_turkish_query,
    is_web_query,
    needs_mcp_tools,
    is_weather_query,
    normalize_web_query,
)
from src.agent.web_search import WebSearchService, WebResultFormatter
from src.agent.prompts import (
    ROUTER_SYSTEM_PROMPT,
    REWRITER_SYSTEM_PROMPT,
    GRADER_SYSTEM_PROMPT,
    RAG_WITH_CONTEXT_SYSTEM_PROMPT,
    RAG_MEMORY_PREFERENCES_BLOCK,
    WEB_WITH_CONTEXT_SYSTEM_PROMPT,
    RAG_NO_CONTEXT_SYSTEM_PROMPT,
    build_generator_prompt,
    select_vision_prompt,
)
from src.config import settings
from src.rag.llm import count_message_tokens, count_tokens
from src.security.url_guard import URLFetchError, fetch_public_url_text

logger = logging.getLogger(__name__)


def _history_turn_count(messages: list) -> int:
    """Kısa: `_history_turn_count` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
    return sum(1 for m in messages if isinstance(m, HumanMessage))


def _observe_node(
    name: str,
    state: AgentState,
    *,
    inputs: dict | None = None,
    outputs: dict | None = None,
    metadata: dict | None = None,
    tags: list[str] | None = None,
    error: str | None = None,
) -> None:
    """Best-effort node observation; disabled LangSmith is a cheap no-op."""
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
            "history_turn_count": _history_turn_count(list(state.get("messages") or [])),
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


def select_recent_history(messages: list, *, mode: str = "rag") -> list:
    """Select bounded recent chat history while preserving memory SystemMessage."""
    max_messages = 6 if mode == "rag" else 8
    token_budget = 900 if mode == "rag" else 1300

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





_router_llm_cache = None
_rag_llm_cache: dict[tuple, object] = {}
_RAG_LLM_CACHE_MAXSIZE = 32  # temperature×max_tokens combinations; prevents unbounded growth


def _get_router_llm():
    """Routing için minimal token-budget LLM — modül-level singleton."""
    global _router_llm_cache
    if _router_llm_cache is None:
        from src.rag.llm import create_vllm_llm
        _router_llm_cache = create_vllm_llm(temperature=0.0, max_tokens=settings.router_max_tokens)
    return _router_llm_cache


def _get_rag_llm(temperature: float = 0.0, max_tokens: int | None = None):
    """RAG üretim / grader / rewriter LLM.

    temperature=0.0 ve max_tokens=None → DualLLM singleton (cached).
    Diğer değerler (temperature, max_tokens) tuple'ı ile önbelleklenir;
    per-session ayar değişikliklerinde TCP bağlantısı yeniden kullanılır.
    Cache dolunca en eski girdi çıkarılır (LRU-like eviction).
    """
    if temperature == 0.0 and max_tokens is None:
        from src.rag.llm import get_rag_llm
        return get_rag_llm()
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


_chat_llm_cache: dict[tuple, object] = {}


def _get_chat_llm(temperature: float | None = None, max_tokens: int | None = None):
    """Basit sohbet için araçsız, küçük prompt'lu LLM."""
    if temperature is None and max_tokens is None:
        from src.rag.llm import get_chat_llm
        return get_chat_llm()
    key = (temperature if temperature is not None else settings.chat_temperature, max_tokens or settings.chat_max_tokens)
    if key not in _chat_llm_cache:
        if len(_chat_llm_cache) >= _RAG_LLM_CACHE_MAXSIZE:
            _chat_llm_cache.pop(next(iter(_chat_llm_cache)))
        from src.rag.llm import create_vllm_llm
        _chat_llm_cache[key] = create_vllm_llm(
            temperature=float(key[0]),
            max_tokens=int(key[1]),
        )
    return _chat_llm_cache[key]


def reset_nodes_llm_cache() -> None:
    """LLM ayarları runtime'da değiştiğinde (api/router.py) çağrılır."""
    global _router_llm_cache
    _router_llm_cache = None
    _rag_llm_cache.clear()
    _chat_llm_cache.clear()


def _get_agent_llm():
    """ReAct agent için tool-call uyumlu LLM (düşük sıcaklık, yüksek bütçe)."""
    from src.rag.llm import get_agent_llm
    return get_agent_llm()


_PLAIN_DIRECT_TOOL_RE = re.compile(
    r"("
    r"hesapla|calculate|kaç eder|yüzde|percent|kdv|vat|"
    r"dosya|belge|pdf|upload|yüklediğim|oku|read_uploaded_file|"
    r"github|gitlab|repo|repository|commit|pull request|branch|issue|gist|"
    r"takvim|calendar|email gönder|send email|toplantı ayarla|schedule meeting"
    r")",
    re.IGNORECASE | re.UNICODE,
)
_PLAIN_DIRECT_ARITH_RE = re.compile(r"^\s*[\d\s+\-*/().,^%]+\s*$")
_DATE_QUERY_RE = re.compile(
    r"(bug[üu]n(ün)?\s*(tarih|g[üu]n|g[üu]nl[üu]k|hangi|ne|kaç[ıi]nc[ıi])|"
    r"tarih\s*(nedir|ne|kaç|bugün)|"
    r"bug[üu]n\s*ne\s*g[üu]n[üu]?|"
    r"hangi\s*g[üu]n[üu]?|"
    r"g[üu]n[üu]n\s*tarihi|"
    r"bu\s*g[üu]n\s*(g[üu]nlerden|ne\s*g[üu]n|hangi))",
    re.IGNORECASE | re.UNICODE,
)
_MATH_WORD_RE = re.compile(
    r"(asal|prime|fibonacci|fakt[öo]riyel|factorial|mutlak\s+fark|"
    r"basamakl[ıi]|toplam[ıi]?|çarp[ıi]m[ıi]?|carp[ıi]m[ıi]?|kaç\s+eder|kac\s+eder)",
    re.IGNORECASE | re.UNICODE,
)
_PLAIN_DIRECT_CHAT_RE = re.compile(
    r"^\s*("
    r"merhaba|selam|hey|hi|hello|"
    r"nas[ıi]ls[ıi]n|naber|g[üu]nayd[ıi]n|iyi\s+(g[üu]nler|ak[şs]amlar)|"
    r"te[şs]ekk[üu]r|sa[ğg]ol|tamam|ok|eyvallah|"
    r"sen\s+kimsin|ad[ıi]n\s+ne|ne\s+yapabilirsin"
    r")\b",
    re.IGNORECASE | re.UNICODE,
)


_SAFE_MATH_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval_math_expr(expression: str) -> str:
    """Saf aritmetik ifadeleri LLM/ReAct'e gitmeden hesapla."""
    def _eval(node):
        """Kısa: `_eval` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp):
            op = type(node.op)
            if op not in _SAFE_MATH_OPS:
                raise ValueError(f"Unsupported operator: {op.__name__}")
            return _SAFE_MATH_OPS[op](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op = type(node.op)
            if op not in _SAFE_MATH_OPS:
                raise ValueError(f"Unsupported operator: {op.__name__}")
            return _SAFE_MATH_OPS[op](_eval(node.operand))
        raise ValueError(f"Unsupported expression: {type(node).__name__}")

    normalized = expression.replace("^", "**").replace(",", ".")
    result = _eval(ast.parse(normalized, mode="eval"))
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return f"{expression.strip()} = {result}"


def _should_use_plain_direct_llm(question: str) -> bool:
    """Kısa sohbetlerde ReAct/tool prompt maliyetini atla."""
    q = question.strip()
    if not q:
        return True
    if is_web_query(q) or needs_mcp_tools(q):
        return False
    if _PLAIN_DIRECT_ARITH_RE.fullmatch(q) or _PLAIN_DIRECT_TOOL_RE.search(q):
        return False
    if _PLAIN_DIRECT_CHAT_RE.search(q):
        return True
    # Kısa, araç gerektirmeyen sohbetler ("Eymen??", "devam", "peki") için hızlı yol.
    return len(q) <= 80 and "\n" not in q


def _should_use_math_direct_llm(question: str) -> bool:
    """Kelime problemi matematikte ReAct/tool şemasını atla; küçük model düz çözer."""
    q = question.strip()
    return bool(_MATH_WORD_RE.search(q)) and not is_web_query(q) and not needs_mcp_tools(q)




_RERANKER_FAILED = object()


class _RerankerRegistry:
    """Reranker instance'ını lazy olarak yükler ve önbellekte tutar."""

    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get(cls):
        """Kısa: `get` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
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



_web_search_service = None
_web_search_service_loaded = False


def _get_web_search_service():
    """Kısa: `_get_web_search_service` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
    global _web_search_service, _web_search_service_loaded
    if not _web_search_service_loaded:
        _web_search_service = WebSearchService.from_settings()
        _web_search_service_loaded = True
    return _web_search_service




def _parse_route(text: str, default: str = "direct") -> str:
    """LLM yanıtından 'rag', 'web', 'direct' veya 'vision' çıkarır (regex tabanlı)."""
    import re
    text_lower = text.lower().strip()
    if re.search(r'\brag\b', text_lower):
        return "rag"
    if re.search(r'\bweb\b', text_lower):
        return "web"
    if re.search(r'\bdirect\b', text_lower):
        return "direct"
    if re.search(r'\bvision\b', text_lower):
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
    """Sorguyu 'rag', 'direct' veya 'vision' olarak sınıflandırır.

    Yol 0 (anlık): image_data doluysa LLM'e sormadan direkt 'vision' döner.
    Yol 1 (hızlı): Keyword eşleşmesi varsa LLM çağrısı yapılmaz.
    Yol 2 (yavaş): Belirsiz sorgular için düşük bütçeli LLM, text parsing ile rota belirlenir.
    """
    t0 = time.perf_counter()
    question = state["question"]
    prior_messages = list(state.get("messages", []))
    q_len = len(question)
    session_uploads = state.get("session_uploads") or []

    if state.get("image_data"):
        imgs = state["image_data"]
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            "Router → vision [images=%d, mimes=%s, q_len=%d, t=0.00s]",
            len(imgs),
            ",".join(img.get("mime", "?") for img in imgs),
            q_len,
        )
        _observe_node(
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

    if state.get("force_web_search"):
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            "Router → web [reason=force_web_search, uploads=%d, q_len=%d, t=%.3fs]",
            len(session_uploads), q_len, time.perf_counter() - t0,
        )
        _observe_node(
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

    # Dosya yüklendiyse: deterministik RAG — keyword/LLM routing atlanır.
    if state.get("source_filter"):
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            "Router → rag [reason=source_filter, file='%s', q_len=%d, t=%.3fs]",
            state["source_filter"], q_len, time.perf_counter() - t0,
        )
        _observe_node(
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

    fast_route = keyword_route(question, has_uploads=bool(session_uploads))
    if fast_route:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            "Router → %s [reason=keyword, uploads=%d, q_len=%d, t=%.3fs]",
            fast_route, len(session_uploads), q_len, time.perf_counter() - t0,
        )
        _observe_node(
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

    if session_uploads:
        if is_web_query(question):
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            logger.info(
                "Router → web [reason=web_override+uploads, uploads=%d, q_len=%d, t=%.3fs]",
                len(session_uploads), q_len, time.perf_counter() - t0,
            )
            _observe_node(
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
        logger.info(
            "Router → rag [reason=uploads_bias, uploads=%d, q_len=%d, t=%.3fs]",
            len(session_uploads), q_len, time.perf_counter() - t0,
        )
        _observe_node(
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

    logger.info(
        "Router → LLM [prior_msgs=%d, q_len=%d, max_tokens=%d]",
        len(prior_messages), q_len, settings.router_max_tokens,
    )
    t_llm = time.perf_counter()
    llm = _get_router_llm()
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
    logger.info(
        "Router → %s [reason=llm, llm_t=%.3fs, total_t=%.3fs]",
        route, llm_elapsed, total_elapsed,
    )
    _observe_node(
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
    """True döndürürse rewriter LLM çağrısı (~6s) atlanır.

    Atla  → kısa (≤8 kelime) + soru kelimesi var + follow-up değil.
    Atla  → teknik named entity içeriyor (rewrite semantic yönü kaydırır).
    Devam → çok-turlu follow-up'lar (referans çözümlemesi gerektirir).
    """
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
    """Soruyu vektör veritabanı araması için optimize eder.

    Kısa/net sorgularda ve tek-turlu sorgularda LLM çağrısını atlar (~6s kazanç).
    """
    t0 = time.perf_counter()
    question = state["question"]
    prior_messages = list(state.get("messages", []))

    if _should_skip_rewrite(question, prior_messages):
        logger.info(
            "Rewriter: skip [reason=short_clear, q_len=%d, t=%.3fs]",
            len(question), time.perf_counter() - t0,
        )
        return state

    llm = _get_rag_llm(temperature=0.0)
    messages_to_send = [SystemMessage(content=REWRITER_SYSTEM_PROMPT)]
    memory_ctx = (state.get("memory_context") or "").strip()
    if memory_ctx:
        messages_to_send.append(
            SystemMessage(content=f"Thread memory (rewrite için bağlam):\n{memory_ctx}")
        )
    if prior_messages:
        messages_to_send.extend(prior_messages[-2:])
    messages_to_send.append(HumanMessage(content=question))

    # Dense gate embedding ve LLM rewrite paralelde — retriever_node LRU cache'ten hızlı alır
    async def _warm_embed_cache():
        """Kısa: `_warm_embed_cache` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        try:
            from src.rag.vectorstore import _cached_embed_query
            await asyncio.to_thread(_cached_embed_query, question)
        except Exception:
            pass

    embed_task = asyncio.create_task(_warm_embed_cache())
    response = await llm.ainvoke(messages_to_send)
    await embed_task  # cache'in doldurulmasını garanti et
    rewritten = response.content.strip()

    _ANSWER_MARKERS = ("ihtiyacım", "yapabilmem için", "kritik bilgi", "hesaplayabilmem",
                       "belirtmek isterim", "lütfen", "sunabilmem", "verebilmem")
    is_hallucination = (
        len(rewritten) > 250
        or "\n" in rewritten
        or any(m in rewritten.lower() for m in _ANSWER_MARKERS)
    )
    if is_hallucination:
        logger.warning(
            "Rewriter: hallucination → original kept [rewritten_len=%d, t=%.3fs]",
            len(rewritten), time.perf_counter() - t0,
        )
        return state

    logger.info(
        "Rewriter: rewritten [%d→%dch, prior=%d, t=%.3fs] '%.80s'",
        len(question), len(rewritten), len(prior_messages),
        time.perf_counter() - t0, rewritten,
    )
    return {**state, "question": rewritten}




def _build_tenant_filter(user_id: str = "", thread_id: str = ""):
    """Upload scope yokken kullanıcı izolasyonu — bulk corpus (boş user_id) opsiyonel."""
    from qdrant_client import models as qmodels

    _ = thread_id  # reserved for future per-thread purge APIs
    if not settings.qdrant_tenant_filter_enabled:
        return None

    uid = (user_id or "").strip()
    if not uid:
        return None

    should: list = [
        qmodels.FieldCondition(
            key="metadata.user_id",
            match=qmodels.MatchValue(value=uid),
        ),
    ]
    if settings.qdrant_include_shared_corpus:
        should.append(
            qmodels.IsEmptyCondition(
                is_empty=qmodels.PayloadField(key="metadata.user_id"),
            )
        )
    return qmodels.Filter(should=should)


def _build_source_filter(
    source_filter: str,
    session_uploads: list[str] | None = None,
    *,
    user_id: str = "",
    thread_id: str = "",
):
    """source_filter veya session_uploads'dan Qdrant metadata filtresi oluşturur.

    source_filter verilmişse (mevcut yüklemenin dosya adı) → tek değer eşleşmesi.
    Yoksa ve session_uploads doluysa → bu dosyaların herhangi biriyle eşleşme.
    İkisi de boşsa tenant filtresi (user_id / shared corpus) uygulanır.
    """
    from qdrant_client import models as qmodels

    must: list = []
    if source_filter:
        must.append(
            qmodels.FieldCondition(
                key="metadata.source_file",
                match=qmodels.MatchValue(value=source_filter),
            )
        )
    else:
        uploads = [s for s in (session_uploads or []) if s]
        if uploads:
            must.append(
                qmodels.FieldCondition(
                    key="metadata.source_file",
                    match=qmodels.MatchAny(any=uploads),
                )
            )
        else:
            tenant = _build_tenant_filter(user_id, thread_id)
            if tenant is not None:
                must.append(tenant)

    if not must:
        return None
    if len(must) == 1:
        return qmodels.Filter(must=must)
    return qmodels.Filter(must=must)


_DOCUMENT_OVERVIEW_RE = re.compile(
    r"\b("
    r"ana\s+konu|konusu|önemli\s+bulgu\w*|onemli\s+bulgu\w*|bulgu\w*|"
    r"yöntem|yontem|metod|method|methodology|"
    r"özet|ozet|summarize|summary|abstract|"
    r"sonuç|sonuc|conclusion|giriş|giris|introduction"
    r")\b",
    re.IGNORECASE,
)

_OVERVIEW_SECTION_RE = re.compile(
    r"\b("
    r"abstract|özet|ozet|giriş|giris|introduction|"
    r"yöntem|yontem|methodology|method|"
    r"bulgular|findings|sonuç|sonuc|conclusion|"
    r"tartışma|tartisma|discussion"
    r")\b",
    re.IGNORECASE,
)


def _is_document_overview_question(state: AgentState) -> bool:
    """Kısa: `_is_document_overview_question` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
    q = " ".join(
        str(part or "")
        for part in (state.get("original_question"), state.get("question"))
    )
    return bool(_DOCUMENT_OVERVIEW_RE.search(q))


def _payload_to_document(payload: dict | None) -> Document | None:
    """Kısa: `_payload_to_document` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
    payload = payload or {}
    content = (
        payload.get("page_content")
        or payload.get("content")
        or payload.get("text")
        or payload.get("document")
        or ""
    )
    if not isinstance(content, str) or not content.strip():
        return None
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    extra_meta = {
        k: v for k, v in payload.items()
        if k not in {"page_content", "content", "text", "document", "metadata"}
    }
    return Document(page_content=content, metadata={**extra_meta, **metadata})


def _chunk_sort_key(doc: Document) -> tuple[int, int, str]:
    """Kısa: `_chunk_sort_key` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
    from src.rag.retriever import chunk_id

    meta = getattr(doc, "metadata", {}) or {}
    raw_idx = meta.get("chunk_index")
    idx = 10**9
    if isinstance(raw_idx, int):
        idx = raw_idx
    elif isinstance(raw_idx, str):
        m = re.search(r"\d+", raw_idx)
        if m:
            idx = int(m.group(0))
    try:
        page = int(meta.get("page") or 10**9)
    except (TypeError, ValueError):
        page = 10**9
    return idx, page, chunk_id(doc)


def _fetch_document_overview_chunks(store, qdrant_filter, *, limit: int = 96, max_docs: int = 4) -> list[Document]:
    """Fetch opening/section/closing chunks for broad document questions."""
    if qdrant_filter is None:
        return []
    try:
        records, _ = store.client.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=qdrant_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as exc:
        logger.debug("overview chunk fetch failed: %s", exc)
        return []

    docs = [doc for rec in records if (doc := _payload_to_document(getattr(rec, "payload", None)))]
    if not docs:
        return []
    docs.sort(key=_chunk_sort_key)

    selected: list[Document] = []
    selected.extend(docs[:2])
    selected.extend(doc for doc in docs if _OVERVIEW_SECTION_RE.search(doc.page_content or ""))
    if len(docs) > 2:
        selected.extend(docs[-2:])

    from src.rag.retriever import deduplicate_documents

    return deduplicate_documents(selected, max_docs=max_docs)


async def retriever_node(state: AgentState) -> AgentState:
    """Hybrid retrieval + dense gate + opsiyonel reranking uygular."""
    t0 = time.perf_counter()
    question = state["question"]
    source_filter = state.get("source_filter", "")
    session_uploads = state.get("session_uploads") or []
    latency_ms: dict[str, float] = {}
    dense_score = None
    retrieval_gate = "skip"
    strategy = state.get("retrieval_strategy") or settings.retrieval_strategy
    use_rerank_val = state.get("use_rerank")
    if use_rerank_val is None:
        use_rerank_val = settings.use_rerank

    try:
        from src.rag.vectorstore import get_hybrid_store
        from src.rag.retriever import create_retriever, deduplicate_documents, run_retriever, chunk_id

        store = get_hybrid_store()
        qdrant_filter = _build_source_filter(
            source_filter,
            session_uploads,
            user_id=state.get("user_id") or "",
            thread_id=state.get("thread_id") or "",
        )

        if source_filter or session_uploads:
            dense_score = 1.0
            retrieval_gate = "skip"
            filter_desc = f"source_filter='{source_filter}'" if source_filter else f"uploads={session_uploads}"
            logger.info("Retriever: dense_gate=skip [%s]", filter_desc)
        else:
            t_gate = time.perf_counter()
            try:
                dense_score = await asyncio.to_thread(
                    store.max_dense_similarity, question, qdrant_filter=qdrant_filter
                )
            except Exception as exc:
                logger.warning("Dense gate failed: %s — skipping gate", exc)
                dense_score = settings.rag_min_dense_similarity
            latency_ms["dense_gate"] = round((time.perf_counter() - t_gate) * 1000, 2)
            logger.info(
                "Retriever: dense_gate=%.3f [weak=%.3f, pass=%.3f, t=%.3fs]",
                dense_score, settings.rag_min_dense_similarity, settings.rag_dense_pass_similarity,
                time.perf_counter() - t_gate,
            )
            if dense_score >= settings.rag_dense_pass_similarity:
                retrieval_gate = "pass"
            elif dense_score >= settings.rag_min_dense_similarity:
                retrieval_gate = "soft"
            else:
                retrieval_gate = "weak"
            logger.info("Retriever: dense_gate=%s [score=%.3f]", retrieval_gate, dense_score)

        retriever = create_retriever(
            vectorstore=store.store,
            question=question,
            strategy=strategy,
            base_k=settings.base_k,
            max_k=settings.top_k,
            fetch_k=settings.fetch_k,
            lambda_mult=settings.lambda_mult,
            score_threshold=settings.score_threshold,
            use_rerank=use_rerank_val,
            reranker=_RerankerRegistry.get(),
            rerank_top_n=settings.rerank_top_n,
            qdrant_filter=qdrant_filter,
        )
        t_fetch = time.perf_counter()
        documents = await asyncio.to_thread(run_retriever, retriever, question)
        documents = deduplicate_documents(documents, max_docs=settings.top_k)
        if _is_document_overview_question(state):
            t_overview = time.perf_counter()
            overview_docs = await asyncio.to_thread(
                _fetch_document_overview_chunks,
                store,
                qdrant_filter,
                max_docs=4,
            )
            if overview_docs:
                documents = deduplicate_documents(
                    [*overview_docs, *documents],
                    max_docs=max(settings.top_k, min(settings.rerank_top_n, settings.top_k + len(overview_docs))),
                )
            latency_ms["overview_fetch"] = round((time.perf_counter() - t_overview) * 1000, 2)
            logger.info(
                "Retriever: overview_boost [overview_docs=%d, final_docs=%d, t=%.3fs]",
                len(overview_docs), len(documents), time.perf_counter() - t_overview,
            )
        t_fetch_elapsed = time.perf_counter() - t_fetch
        latency_ms["fetch"] = round(t_fetch_elapsed * 1000, 2)

        # Hybrid (fused) skorları paralel similarity_search_with_score ile çek
        # ve chunk_id üzerinden eşle. Mevcut retriever score'u yutuyor; bu ekstra
        # sorgu Qdrant'ta hızlıdır (~30-100ms).
        hybrid_scores: dict[str, float] = {}
        try:
            search_k = max(settings.rerank_top_n, settings.top_k * 2)
            t_score = time.perf_counter()
            scored_pairs = await asyncio.to_thread(
                _score_lookup_with_filter, store.store, question, search_k, qdrant_filter,
            )
            latency_ms["score_lookup"] = round((time.perf_counter() - t_score) * 1000, 2)
            for d, s in scored_pairs:
                hybrid_scores[chunk_id(d)] = float(s)
        except Exception as exc:
            logger.debug("hybrid score lookup failed: %s", exc)

        # Trace inşası — her dökümana hybrid skorunu metadata'ya yaz
        retrieval_trace: list[dict] = []
        for doc in documents:
            cid = chunk_id(doc)
            h_score = hybrid_scores.get(cid)
            if h_score is not None:
                doc.metadata["retrieval_score"] = h_score
            retrieval_trace.append({
                "chunk_id": cid,
                "hybrid_score": h_score,
                "rerank_score": doc.metadata.get("rerank_score"),
                "used_in_context": False,
            })

        # Kaynak dağılımını özetle
        sources: dict[str, int] = {}
        for doc in documents:
            meta = getattr(doc, "metadata", {}) or {}
            src = meta.get("display_name") or meta.get("source_file", meta.get("source", "?"))
            sources[src] = sources.get(src, 0) + 1
        src_summary = ", ".join(f"{s}×{n}" for s, n in sources.items())

        logger.info(
            "Retriever: docs=%d [strategy=%s, rerank=%s, dense=%.3f, fetch_t=%.3fs, total_t=%.3fs] {%s}",
            len(documents), strategy, use_rerank_val, dense_score,
            t_fetch_elapsed, time.perf_counter() - t0, src_summary,
        )
        if retrieval_trace:
            trace_parts = [
                f"{t['chunk_id']} hybrid={_fmt_score(t['hybrid_score'])} rerank={_fmt_score(t['rerank_score'])}"
                for t in retrieval_trace
            ]
            logger.info("Retriever: trace [%s]", " | ".join(trace_parts))
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        latency_ms["total"] = elapsed_ms
        from src.observability.langsmith import (
            summarize_documents,
            summarize_retrieval_trace,
            summarize_source_distribution,
        )
        trace_summary = summarize_retrieval_trace(retrieval_trace)
        _observe_node(
            "frappe.retriever_result",
            state,
            outputs={
                "status": "success",
                "document_count": len(documents),
                "top_sources": summarize_source_distribution(documents),
                "top_chunks": trace_summary.get("top_chunks", ""),
                "used_chunks": trace_summary.get("used_chunks", ""),
                "retrieval_trace_summary": trace_summary,
                "document_previews": summarize_documents(documents),
                "latency_ms_by_stage": latency_ms,
            },
            metadata={
                "retrieval_strategy": strategy,
                "use_rerank": bool(use_rerank_val),
                "dense_score": dense_score,
                "dense_threshold": settings.rag_min_dense_similarity,
                "dense_pass_threshold": settings.rag_dense_pass_similarity,
                "retrieval_gate": retrieval_gate,
                "top_sources": summarize_source_distribution(documents),
            },
            tags=["frappe", "retriever", "success", f"gate:{retrieval_gate}"],
        )
    except Exception as exc:
        logger.warning("Retriever: error [%s, t=%.3fs]", exc, time.perf_counter() - t0)
        documents = []
        retrieval_trace = []
        _observe_node(
            "frappe.retriever_result",
            state,
            outputs={
                "status": "error",
                "document_count": 0,
                "latency_ms_by_stage": {"total": round((time.perf_counter() - t0) * 1000, 2)},
            },
            metadata={
                "retrieval_strategy": strategy,
                "use_rerank": bool(use_rerank_val),
            },
            tags=["frappe", "retriever", "error"],
            error=str(exc),
        )

    return {**state, "documents": documents, "retrieval_trace": retrieval_trace, "retrieval_gate": retrieval_gate}


def _fmt_score(s) -> str:
    """Kısa: `_fmt_score` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
    return f"{s:.3f}" if isinstance(s, (int, float)) else "?"


def _score_lookup_with_filter(vectorstore, query: str, k: int, qdrant_filter):
    """vectorstore.similarity_search_with_score wrapper — filter desteğiyle."""
    kwargs = {}
    if qdrant_filter is not None:
        kwargs["filter"] = qdrant_filter
    try:
        return vectorstore.similarity_search_with_score(query, k=k, **kwargs)
    except TypeError:
        return vectorstore.similarity_search_with_score(query, k=k)


def _record_to_web_document(record, *, provider: str, query: str, retrieved_at: str) -> Document:
    """Convert a structured web result record into a RAG document."""
    domain = _web_domain(record.url).replace("www.", "")
    excerpt = re.sub(r"\s+", " ", (record.content or "").strip())[:900]
    return Document(
        page_content=(
            f"Title: {record.title}\n"
            f"Published: {record.published or 'unknown'}\n"
            f"URL: {record.url}\n"
            f"Snippet: {excerpt}"
        )[:2500],
        metadata={
            "display_name": record.title,
            "source": record.url,
            "url": record.url,
            "title": record.title,
            "domain": domain,
            "excerpt": excerpt,
            "published": record.published,
            "provider": provider,
            "result_index": record.index,
            "chunk_index": record.index,
            "retrieved_at": retrieved_at,
            "query": query,
            "type": "web_search",
        },
    )


def _published_sort_key(published: str) -> tuple[int, str]:
    """Best-effort freshness key: explicit ISO-ish dates sort above unknown dates."""
    text = (published or "").strip()
    if not text:
        return (0, "")
    match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    if match:
        return (2, match.group(0))
    match = re.search(r"\d{4}", text)
    if match:
        return (1, match.group(0))
    return (0, text)


def _web_domain(url: str) -> str:
    """Kısa: `_web_domain` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _web_source_quality_score(record) -> int:
    """Prefer official/reference sources without dropping weaker results entirely."""
    title = str(getattr(record, "title", "") or "")
    url = str(getattr(record, "url", "") or "")
    content = str(getattr(record, "content", "") or "")
    haystack = f"{title} {url}".lower()
    domain = _web_domain(url).replace("www.", "")
    score = 0

    government_suffixes = (".gov", ".gov.tr", ".gob", ".gov.uk", ".gc.ca", ".gouv.fr", ".go.jp")
    if domain.endswith(government_suffixes) or ".gov." in domain:
        score += 30
    if domain.endswith((".edu", ".edu.tr", ".ac.uk")):
        score += 16
    if any(mark in haystack for mark in ("official", "agency", "ministry", "statistics office", "census bureau")):
        score += 10
    if any(mark in haystack for mark in ("release notes", "help center", "documentation", "docs.", "/docs/")):
        score += 8
    if any(mark in haystack for mark in ("wikipedia.org", "britannica.com", "encyclopedia.com")):
        score += 12

    if len(re.sub(r"\s+", " ", content).strip()) >= 120:
        score += 4
    if not title.strip() or title.strip().lower() in {"search results", "web results", "result", "untitled"}:
        score -= 8
    if len(content.strip()) < 40:
        score -= 8
    if any(mark in haystack for mark in ("instagram.com", "facebook.com", "x.com/", "twitter.com", "tiktok.com", "linkedin.com")):
        score -= 25
    if any(mark in haystack for mark in ("maps.", "/maps", "map listing", "directory listing", "business listing")):
        score -= 15
    if any(mark in haystack for mark in ("reddit.com", "quora.com", "forum", "blogspot.")):
        score -= 8
    return score


_HTTP_URL_RE = re.compile(r"https?://[^\s<>)\"']+", re.IGNORECASE)


def _extract_public_urls(text: str, *, limit: int = 2) -> list[str]:
    urls: list[str] = []
    for match in _HTTP_URL_RE.finditer(text or ""):
        url = match.group(0).rstrip(".,;:!?]")
        if url not in urls:
            urls.append(url)
        if len(urls) >= limit:
            break
    return urls


def _html_to_visible_text(raw: str) -> str:
    """Extract readable text from fetched HTML without adding a hard dependency."""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        body = soup.get_text(" ", strip=True)
        text = f"{title}\n{body}" if title and title not in body[:200] else body
    except Exception:
        text = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw or "")
        text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text or "").strip()


async def _docs_from_explicit_urls(question: str) -> list[Document]:
    docs: list[Document] = []
    for idx, url in enumerate(_extract_public_urls(question), 1):
        try:
            final_url, raw_text = await fetch_public_url_text(url, timeout=8.0, max_bytes=1_500_000)
        except (URLFetchError, Exception) as exc:
            logger.warning("Explicit URL fetch failed [%s]: %s", url, exc)
            continue
        text = _html_to_visible_text(raw_text)
        if not text:
            continue
        domain = _web_domain(final_url).replace("www.", "")
        title = text[:90].strip() or final_url
        excerpt = text[:900].strip()
        docs.append(Document(
            page_content=(
                f"Title: {title}\n"
                f"Published: unknown\n"
                f"URL: {final_url}\n"
                f"Snippet: {excerpt}"
            )[:2500],
            metadata={
                "display_name": title,
                "source": final_url,
                "url": final_url,
                "title": title,
                "domain": domain,
                "excerpt": excerpt,
                "published": "",
                "provider": "direct_url",
                "result_index": idx,
                "chunk_index": idx,
                "retrieved_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
                "query": question,
                "type": "web_search",
                "direct_url": True,
            },
        ))
    return docs


def _hash_text(text: str) -> str:
    """Kısa: `_hash_text` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
    try:
        from src.observability.langsmith import stable_hash
        return stable_hash(text)
    except Exception:
        return ""


def _web_docs_from_result(result, *, query: str, limit: int | None = None) -> list[Document]:
    """Kısa: `_web_docs_from_result` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
    retrieved_at = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    records = list(getattr(result, "records", None) or [])
    if not records:
        records = WebResultFormatter.extract_source_records(
            result.text,
            limit=limit or settings.web_search_max_results,
        )
    deduped_records = []
    seen_urls: set[str] = set()
    for record in records:
        url_key = (getattr(record, "url", "") or "").strip().lower()
        if url_key and url_key in seen_urls:
            continue
        if url_key:
            seen_urls.add(url_key)
        deduped_records.append(record)
    records = deduped_records
    records = sorted(
        records,
        key=lambda r: (_web_source_quality_score(r), _published_sort_key(r.published), -int(getattr(r, "index", 999))),
        reverse=True,
    )
    if records:
        return [
            _record_to_web_document(
                record,
                provider=result.provider,
                query=query,
                retrieved_at=retrieved_at,
            )
            for record in records
        ]
    return [
        Document(
            page_content=result.text[:8000],
            metadata={
                "source": result.provider,
                "display_name": result.provider,
                "provider": result.provider,
                "retrieved_at": retrieved_at,
                "query": query,
                "type": "web_search",
            },
        )
    ]




def _grader_conf_high() -> float:
    """Kısa: `_grader_conf_high` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
    return float(settings.grader_conf_high)


def _grader_conf_low() -> float:
    """Kısa: `_grader_conf_low` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
    return float(settings.grader_conf_low)


def _grader_max_docs() -> int:
    """Kısa: `_grader_max_docs` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
    return int(settings.grader_max_docs)


_GRADER_REASONS = {"sufficient", "irrelevant", "partial", "insufficient_context", "needs_live_data"}


def _parse_grader_payload(text: str) -> tuple[str, str]:
    """Grader JSON'unu güvenli okur; bozuk çıktıda regex fallback kullanır."""
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
    """LLM yanıtından 'yes' veya 'no' çıkarır (JSON öncelikli, regex fallback)."""
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
    if re.search(r'\byes\b', text_lower):
        return "yes"
    if re.search(r'\bno\b', text_lower):
        return "no"
    return default


def _parse_grader_reason(text: str) -> str:
    """Grader yanıtından reason alanını çıkarır."""
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
    """Belge alaka değerlendirmesi — önce sıfır-maliyetli confidence skoru dener.

    Yüksek güven (≥0.7): LLM atlanır → "yes"  (~3s kazanç, çoğu istek).
    Düşük güven  (<0.3): LLM atlanır → "no".
    Orta  (0.3–0.7):     LLM grader çalışır (borderline durum).
    """
    t0 = time.perf_counter()
    from src.rag.retriever import estimate_confidence

    question = state["question"]
    documents = state.get("documents", [])

    if not documents:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info("Grader: no_docs → relevance=no [t=%.3fs]", time.perf_counter() - t0)
        _observe_node(
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
        from src.rag.retriever import estimate_confidence

        original_q = state.get("original_question") or question
        if is_web_query(original_q):
            pass  # Web sorguları için grader LLM'i çalıştır — needs_live_data döndürebilir
        else:
            confidence = estimate_confidence(question, documents)
            if confidence >= _grader_conf_high():
                elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
                logger.info(
                    "Grader: relevance=yes [mode=file_high_conf, conf=%.3f>=%.3f, docs=%d, t=%.3fs]",
                    confidence, _grader_conf_high(), len(documents), time.perf_counter() - t0,
                )
                _observe_node(
                    "frappe.grader_decision",
                    state,
                    outputs={
                        "relevance": "yes",
                        "grader_reason": "sufficient",
                        "mode": "file_high_conf",
                        "confidence": confidence,
                        "high_threshold": _grader_conf_high(),
                        "document_count": len(documents),
                        "latency_ms_by_stage": {"total": elapsed_ms},
                    },
                    metadata={
                        "grader_mode": "file_high_conf",
                        "grader_confidence": confidence,
                        "grader_high_threshold": _grader_conf_high(),
                    },
                    tags=["frappe", "grader", "yes"],
                )
                return {**state, "relevance": "yes", "grader_reason": "sufficient"}
            logger.info(
                "Grader: file context requires LLM [conf=%.3f<%.3f, docs=%d]",
                confidence, _grader_conf_high(), len(documents),
            )

        top_docs = documents[:_grader_max_docs()]
        doc_texts = "\n---\n".join(doc.page_content for doc in top_docs)
        doc_chars = sum(len(d.page_content) for d in top_docs)
        llm = _get_rag_llm(temperature=0.0)
        try:
            t_llm = time.perf_counter()
            response = await llm.ainvoke([
                SystemMessage(content=GRADER_SYSTEM_PROMPT),
                HumanMessage(content=f"Question: {question}\n\nDocuments:\n{doc_texts}"),
            ])
            response_text = _coerce_llm_text(response)
            relevance, reason = _parse_grader_payload(response_text)
            logger.info(
                "Grader: relevance=%s reason=%s [mode=file_llm, docs=%d/%d, doc_chars=%d, llm_t=%.3fs, t=%.3fs]",
                relevance, reason or "-", len(top_docs), len(documents), doc_chars,
                time.perf_counter() - t_llm, time.perf_counter() - t0,
            )
            llm_ms = round((time.perf_counter() - t_llm) * 1000, 2)
        except Exception as exc:
            logger.warning("Grader: llm_error → yes [err=%s, t=%.3fs]", exc, time.perf_counter() - t0)
            relevance, reason = "yes", ""
            llm_ms = None
        _observe_node(
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
        if confidence < _grader_conf_high():
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            logger.info(
                "Grader: relevance=no [mode=weak_dense_gate, conf=%.3f<%.3f, docs=%d, t=%.3fs]",
                confidence, _grader_conf_high(), len(documents), time.perf_counter() - t0,
            )
            _observe_node(
                "frappe.grader_decision",
                state,
                outputs={
                    "relevance": "no",
                    "grader_reason": "insufficient_context",
                    "mode": "weak_dense_gate",
                    "confidence": confidence,
                    "high_threshold": _grader_conf_high(),
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
    elif confidence >= _grader_conf_high():
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            "Grader: relevance=yes [mode=high_conf, conf=%.3f>=%.3f, docs=%d, t=%.3fs]",
            confidence, _grader_conf_high(), len(documents), time.perf_counter() - t0,
        )
        _observe_node(
            "frappe.grader_decision",
            state,
            outputs={
                "relevance": "yes",
                "grader_reason": "sufficient",
                "mode": "high_conf",
                "confidence": confidence,
                "high_threshold": _grader_conf_high(),
                "document_count": len(documents),
                "latency_ms_by_stage": {"total": elapsed_ms},
            },
            metadata={
                "grader_mode": "high_conf",
                "grader_confidence": confidence,
                "grader_high_threshold": _grader_conf_high(),
            },
            tags=["frappe", "grader", "yes"],
        )
        return {**state, "relevance": "yes", "grader_reason": "sufficient"}

    if confidence < _grader_conf_low():
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            "Grader: relevance=no [mode=low_conf, conf=%.3f<%.3f, docs=%d, t=%.3fs]",
            confidence, _grader_conf_low(), len(documents), time.perf_counter() - t0,
        )
        _observe_node(
            "frappe.grader_decision",
            state,
            outputs={
                "relevance": "no",
                "grader_reason": "insufficient_context",
                "mode": "low_conf",
                "confidence": confidence,
                "low_threshold": _grader_conf_low(),
                "document_count": len(documents),
                "latency_ms_by_stage": {"total": elapsed_ms},
            },
            metadata={
                "grader_mode": "low_conf",
                "grader_confidence": confidence,
                "grader_low_threshold": _grader_conf_low(),
            },
            tags=["frappe", "grader", "no"],
        )
        return {**state, "relevance": "no", "grader_reason": "insufficient_context", "refusal_mode": True}

    top_docs = documents[:_grader_max_docs()]
    doc_texts = "\n---\n".join(doc.page_content for doc in top_docs)
    doc_chars = sum(len(d.page_content) for d in top_docs)
    llm = _get_rag_llm(temperature=0.0)
    try:
        t_llm = time.perf_counter()
        response = await llm.ainvoke([
            SystemMessage(content=GRADER_SYSTEM_PROMPT),
            HumanMessage(content=f"Question: {question}\n\nDocuments:\n{doc_texts}"),
        ])
        response_text = _coerce_llm_text(response)
        relevance, reason = _parse_grader_payload(response_text)
        logger.info(
            "Grader: relevance=%s reason=%s [mode=mid_conf, conf=%.3f, docs=%d/%d, doc_chars=%d, llm_t=%.3fs, t=%.3fs]",
            relevance, reason or "-", confidence, len(top_docs), len(documents), doc_chars,
            time.perf_counter() - t_llm, time.perf_counter() - t0,
        )
        llm_ms = round((time.perf_counter() - t_llm) * 1000, 2)
    except Exception as exc:
        # mid_conf hata: güvensiz belgeyle üretim yerine web fallback'e düş
        logger.warning("Grader: llm_error → no [err=%s, t=%.3fs]", exc, time.perf_counter() - t0)
        relevance, reason = "no", "insufficient_context"
        llm_ms = None

    _observe_node(
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




def _build_vision_content_parts(image_data: list[dict], text: str) -> list[dict]:
    """Görsel ve metin parçalarından LLM content listesi oluşturur."""
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
    """Yüklenen görseli Gemma 4 multimodal API ile analiz eder.

    İçerik tipine göre (fatura, tablo, grafik, şema, genel) otomatik prompt seçimi yapılır.
    """
    question = state["question"]
    image_data = state.get("image_data") or []
    prior_messages = list(state.get("messages", []))

    image_names = [img.get("name", "") for img in image_data]
    system_prompt = select_vision_prompt(question, image_names)
    content_parts = _build_vision_content_parts(
        image_data, question.strip() or "Bu görseli analiz et."
    )

    img_sizes = [len(img.get("base64", "")) * 3 // 4 for img in image_data]
    logger.info(
        "Vision: images=%d [mimes=%s, sizes=%s], prior=%d, temp=0.2",
        len(image_data),
        ",".join(img.get("mime", "?") for img in image_data),
        ",".join(f"{s//1024}KB" for s in img_sizes),
        len(prior_messages),
    )
    t0 = time.perf_counter()
    llm = _get_rag_llm(temperature=0.2)

    messages_to_send = [SystemMessage(content=system_prompt)]
    messages_to_send.extend(prior_messages[-6:])
    messages_to_send.append(HumanMessage(content=content_parts))

    try:
        response = await llm.ainvoke(messages_to_send)
        generation = response.content or ""
        logger.info(
            "Vision: done [ans_len=%dch, t=%.3fs]",
            len(generation), time.perf_counter() - t0,
        )
    except Exception as exc:
        logger.error("Vision: error [%s, t=%.3fs]", exc, time.perf_counter() - t0)
        generation = (
            "Görseli işleyemedim. Lütfen PNG, JPEG veya WEBP formatında "
            "ve makul boyutta (< 5 MB) bir görsel yükleyin."
        )

    new_messages = [
        *prior_messages,
        HumanMessage(content=question),
        AIMessage(content=generation),
    ]
    return {**state, "generation": generation, "messages": new_messages, **_final_answer_fields(state, generation, t0=t0, mode="vision")}




async def vision_rag_node(state: AgentState) -> AgentState:
    """Hibrit mode: görsel analizi yapar, sonucu state'e yazar; RAG pipeline devam eder.

    Akış: vision_rag → rewriter → retriever → grader → generator
    Generator, vision_context'i [Görsel Analizi] kaynağı olarak bağlama dahil eder.
    """
    question = state["question"]
    image_data = state.get("image_data") or []
    prior_messages = list(state.get("messages", []))

    image_names = [img.get("name", "") for img in image_data]
    system_prompt = select_vision_prompt(question, image_names)
    content_parts = _build_vision_content_parts(
        image_data,
        "Bu görseli detaylıca analiz et. "
        "Tüm metinleri, sayıları, tablo verilerini ve yapısal bilgileri eksiksiz çıkar. "
        "Sonuç RAG sistemi için kaynak olarak kullanılacak.",
    )

    logger.info(
        "Vision-RAG: %d görsel analiz ediliyor (prompt=%s)",
        len(image_data), system_prompt[:40].replace("\n", " "),
    )
    llm = _get_rag_llm(temperature=0.1)

    try:
        response = await llm.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=content_parts)]
        )
        vision_context = (response.content or "").strip()
    except Exception as exc:
        logger.warning("Vision-RAG görsel analizi başarısız: %s", exc)
        vision_context = ""

    logger.info("Vision-RAG: analiz tamamlandı (%d karakter)", len(vision_context))
    return {**state, "vision_context": vision_context}




async def vision_search_node(state: AgentState) -> AgentState:
    """Görsel analizi + web araması kombinasyonu → generator.

    Kullanım: image_data var VE soru gerçek zamanlı veri gerektiriyor
    (döviz kuru, fiyat, borsa, hava durumu vb.).

    Adımlar:
      1. Görsel → Gemma vision → vision_context (tarih, tutar, döviz birimi vb.)
      2. Web → orijinal soru ile arama → documents
      3. Generator her ikisini birleştirerek hesaplama + sentez yapar.
    """
    question = state["question"]
    image_data = state.get("image_data") or []
    prior_messages = list(state.get("messages", []))

    # ── Adım 1: Görsel analizi ──────────────────────────────────────────────
    image_names = [img.get("name", "") for img in image_data]
    system_prompt = select_vision_prompt(question, image_names)
    content_parts = _build_vision_content_parts(
        image_data,
        "Bu görseli analiz et. Tarih, tutar, döviz birimi, miktar gibi "
        "tüm yapısal verileri olduğu gibi çıkar. "
        "Sonuç gerçek zamanlı web verileriyle birleştirilecek.",
    )

    logger.info("Vision-Search: %d görsel analiz ediliyor", len(image_data))
    llm = _get_rag_llm(temperature=0.1)

    try:
        vision_response = await llm.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=content_parts)]
        )
        vision_context = (vision_response.content or "").strip()
    except Exception as exc:
        logger.warning("Vision-Search görsel analizi başarısız: %s", exc)
        vision_context = ""

    logger.info("Vision-Search: görsel analizi tamamlandı (%d karakter)", len(vision_context))

    # ── Adım 2: Web araması ─────────────────────────────────────────────────
    original_q = state.get("original_question") or question
    service = _get_web_search_service()
    web_docs = []

    async with cl.Step(name="Web Search", type="tool") as step:
        step.input = original_q
        if service:
            web_result = await service.search(original_q)
            if web_result:
                web_docs = _web_docs_from_result(web_result, query=original_q)
                step.output = f"Found via {web_result.provider} ({len(web_result.text)} chars)."
            else:
                logger.warning("Vision-Search: web araması sonuç döndürmedi")
                step.output = "Web search returned no results."
        else:
            step.output = "Web search service unavailable."

    return {**state, "vision_context": vision_context, "documents": web_docs}




def _coerce_llm_text(response) -> str:
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


def _fallback_context_answer(question: str, documents: list[Document], vision_context: str = "") -> str:
    """Son çare: boş LLM çıktısı yerine belgeden tek ilgili bölüm göster."""
    if vision_context.strip():
        return f"Görsel analizden bulunan bilgi:\n\n{vision_context.strip()[:600]}"
    if documents:
        meta = getattr(documents[0], "metadata", {}) or {}
        src = meta.get("display_name") or meta.get("source_file", "belge")
        body = (documents[0].page_content or "").strip()[:600]
        return f"Belgeden ({src}) ilgili bölüm:\n\n{body}"
    return "Bu soruyu yanıtlayabilecek bir belge bağlamı bulunamadı."


@dataclass
class RAGContextAssembly:
    system_content: str
    context: str
    used_chunk_ids: list[str] = field(default_factory=list)
    docs_included: int = 0
    input_budget_tokens: int = 0
    budget_chars: int = 0
    used_chars: int = 0
    overhead_tokens: int = 0
    truncated: bool = False
    max_input_chars: int = 0


def _estimate_history_tokens(messages: list) -> int:
    """Kısa: `_estimate_history_tokens` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
    return count_message_tokens(messages)


def _source_header(index: int, doc: Document) -> str:
    """Kısa: `_source_header` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
    meta = getattr(doc, "metadata", {}) or {}
    src = meta.get("display_name") or meta.get("source_file", meta.get("source", ""))
    page = meta.get("page", "")
    if meta.get("type") == "web_search":
        url = meta.get("url") or meta.get("source", "")
        published = meta.get("published", "")
        header = f"[Kaynak {index}: {src or url}, Web"
        if published:
            header += f", Tarih: {published}"
        if url:
            header += f", URL: {url}"
        header += "]"
        return header
    return f"[Kaynak {index}: {src}" + (f", Sayfa {page}" if page and str(page) not in {"", "?"} else "") + "]"


def _context_signature(text: str) -> str:
    """Context dedupe için metni küçük ve kararlı bir imzaya indirger."""
    normalized = re.sub(r"\s+", " ", (text or "").lower()).strip()
    return normalized[:700]


def _context_overlap_score(question: str, content: str) -> float:
    """Soru-terim örtüşmesiyle çok zayıf chunk'ları ayıklamak için hafif skor üretir."""
    from src.rag.retriever import _tokenize_for_overlap

    terms = _tokenize_for_overlap(question)
    if not terms:
        return 0.0
    normalized_content = " ".join(_tokenize_for_overlap(content))
    if not normalized_content:
        return 0.0
    hits = sum(1 for term in terms if term in normalized_content)
    return hits / max(len(terms), 1)


def _prepare_context_documents(documents: list[Document], question: str) -> list[Document]:
    """Decompose/recompose öncesi kısa, tekrar veya belirgin zayıf chunk'ları temizler."""
    if not documents:
        return []

    unique_docs: list[Document] = []
    seen: set[str] = set()
    for doc in documents:
        content = (doc.page_content or "").strip()
        if len(content) < 40:
            continue
        signature = _context_signature(content)
        if signature in seen:
            continue
        seen.add(signature)
        unique_docs.append(doc)

    if len(unique_docs) <= 2:
        return unique_docs

    scored = [(doc, _context_overlap_score(question, doc.page_content or "")) for doc in unique_docs]
    has_grounded_matches = sum(1 for _, score in scored if score > 0.0) >= 2
    if not has_grounded_matches:
        return unique_docs

    kept: list[Document] = []
    for doc, score in scored:
        meta = getattr(doc, "metadata", {}) or {}
        if score > 0.0 or meta.get("type") == "web_search":
            kept.append(doc)
    return kept or unique_docs[:2]


def assemble_rag_context(
    *,
    documents: list[Document],
    vision_context: str,
    rag_history: list,
    answer_question: str,
    retrieval_trace: list[dict],
    output_tokens: int,
    memory_preferences: str = "",
) -> RAGContextAssembly:
    """Build bounded RAG context and mark retrieval_trace entries used in prompt."""
    context_parts: list[str] = []
    used_chunk_ids: list[str] = []
    documents = _prepare_context_documents(documents, answer_question)

    if vision_context:
        context_parts.append(f"[Görsel Analizi]\n{vision_context}")

    n_ctx = settings.llm_context_size
    history_tokens = _estimate_history_tokens(rag_history)
    overhead_tokens = settings.rag_context_safety_margin_tokens + history_tokens
    input_budget_tokens = max(256, n_ctx - output_tokens - overhead_tokens)
    budget_chars = int(input_budget_tokens * 2.5)
    used_chars = sum(len(p) for p in context_parts)

    from src.rag.retriever import chunk_id as _chunk_id

    for i, doc in enumerate(documents, 1):
        meta = getattr(doc, "metadata", {}) or {}
        header = _source_header(i, doc)
        remaining = budget_chars - used_chars
        if remaining <= len(header) + 50:
            break
        max_chars = min(2500 if meta.get("type") == "web_search" else 2000, remaining - len(header) - 10)
        content = (doc.page_content or "")[:max_chars]
        if not content.strip():
            continue
        candidate = f"{header}\n{content}"
        candidate_context = "\n\n---\n\n".join([*context_parts, candidate])
        while (
            count_tokens(candidate_context) + history_tokens + output_tokens + settings.rag_context_safety_margin_tokens > n_ctx
            and len(content) > 200
        ):
            content = content[: int(len(content) * 0.75)].rstrip()
            candidate = f"{header}\n{content}"
            candidate_context = "\n\n---\n\n".join([*context_parts, candidate])
        if count_tokens(candidate_context) + history_tokens + output_tokens + settings.rag_context_safety_margin_tokens > n_ctx:
            if context_parts:
                break
            content = content[:200].rstrip()
            candidate = f"{header}\n{content}"
        context_parts.append(candidate)
        used_chars += len(header) + len(content) + 10
        used_chunk_ids.append(_chunk_id(doc))

    docs_included = len(context_parts) - (1 if vision_context else 0)
    context = "\n\n---\n\n".join(context_parts)
    only_web_context = bool(documents) and not vision_context and all(
        (getattr(doc, "metadata", {}) or {}).get("type") == "web_search"
        for doc in documents
    )
    prompt_template = WEB_WITH_CONTEXT_SYSTEM_PROMPT if only_web_context else RAG_WITH_CONTEXT_SYSTEM_PROMPT
    system_content = prompt_template.replace("{context}", context)
    prefs = (memory_preferences or "").strip()
    if prefs and not only_web_context:
        system_content += RAG_MEMORY_PREFERENCES_BLOCK.replace("{memory_preferences}", prefs)

    prior_chars = sum(len(getattr(m, "content", "") or "") for m in rag_history)
    total_prompt_chars = len(system_content) + prior_chars + len(answer_question)
    max_input_chars = int(max(256, n_ctx - output_tokens - 50) * 2.5)
    truncated = False
    if total_prompt_chars > max_input_chars:
        safe_ctx_len = max(500, max_input_chars - prior_chars - len(answer_question) - 800)
        context = context[:safe_ctx_len]
        system_content = prompt_template.replace("{context}", context)
        used_chars = min(used_chars, len(context))
        truncated = True

    used_set = set(used_chunk_ids)
    for entry in retrieval_trace:
        if entry.get("chunk_id") in used_set:
            entry["used_in_context"] = True

    return RAGContextAssembly(
        system_content=system_content,
        context=context,
        used_chunk_ids=used_chunk_ids,
        docs_included=docs_included,
        input_budget_tokens=input_budget_tokens,
        budget_chars=budget_chars,
        used_chars=used_chars,
        overhead_tokens=overhead_tokens,
        truncated=truncated,
        max_input_chars=max_input_chars,
    )


def _source_list_line(index: int, doc: Document) -> str:
    """Kısa: `_source_list_line` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
    meta = getattr(doc, "metadata", {}) or {}
    title = meta.get("display_name") or meta.get("title") or meta.get("source_file") or meta.get("source") or f"Kaynak {index}"
    url = meta.get("url") or (meta.get("source") if meta.get("type") == "web_search" else "")
    published = f" — {meta.get('published')}" if meta.get("published") else ""
    page = meta.get("page")
    page_txt = f", s. {page}" if page and str(page) not in {"", "?"} else ""
    if url:
        return f"- [{index}] [{title}]({url}){published}"
    return f"- [{index}] {title}{page_txt}"


def append_used_sources(answer: str, documents: list[Document], question: str) -> str:
    """Append a compact source list for citations that appear in the answer."""
    if not answer.strip() or not documents:
        return answer.strip()
    if re.search(r"(?im)^\s*(kaynaklar|sources)\s*:", answer):
        return answer.strip()
    cited: list[int] = []
    for raw in re.findall(r"\[(?:Kaynak\s*)?(\d+)\]", answer, re.IGNORECASE):
        try:
            idx = int(raw)
        except ValueError:
            continue
        if 1 <= idx <= len(documents) and idx not in cited:
            cited.append(idx)
    if not cited:
        return answer.strip()
    header = "Kaynaklar:" if is_turkish_query(question) else "Sources:"
    lines = [header] + [_source_list_line(idx, documents[idx - 1]) for idx in cited]
    return f"{answer.strip()}\n\n" + "\n".join(lines)


def _final_answer_fields(
    state: AgentState,
    generation: str,
    *,
    t0: float,
    mode: str,
    extra_latency: dict | None = None,
) -> dict:
    """Kısa: `_final_answer_fields` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
    from src.observability.langsmith import safe_preview

    latency = {"total": round((time.perf_counter() - t0) * 1000, 2), **(extra_latency or {})}
    answer_preview = safe_preview(generation)
    _observe_node(
        f"frappe.{mode}_result",
        state,
        outputs={
            "answer_preview": answer_preview,
            "answer_chars": len(generation),
            "route": state.get("route", mode),
            "retry_path": "primary",
            "latency_ms_by_stage": latency,
        },
        metadata={"answer_chars": len(generation), "response_mode": mode},
        tags=["frappe", mode, state.get("route", mode) or mode],
    )
    return {
        "answer_preview": answer_preview,
        "answer_chars": len(generation),
        "document_count": len(state.get("documents") or []),
        "used_context_count": 0,
        "document_previews": [],
        "retrieval_trace_summary": {},
        "top_sources": "",
        "top_chunks": "",
        "used_chunks": "",
        "retry_summary": {"retry_path": "primary"},
        "retry_path": "primary",
        "latency_ms_by_stage": latency,
    }


async def _retry_generator_with_compact_context(
    question: str,
    documents: list[Document],
    prior_messages: list,
    vision_context: str,
) -> str:
    """Boş üretimde daha küçük ve daha direkt bir RAG promptuyla tekrar dene."""
    compact_parts: list[str] = []
    if vision_context.strip():
        compact_parts.append(f"[Görsel Analizi]\n{vision_context.strip()[:1200]}")

    for i, doc in enumerate(documents[:4], 1):
        meta = getattr(doc, "metadata", {}) or {}
        src = meta.get("display_name") or meta.get("source_file", meta.get("source", ""))
        page = meta.get("page", "")
        if meta.get("type") == "web_search":
            url = meta.get("url") or meta.get("source", "")
            published = meta.get("published", "")
            header = f"[Kaynak {i}: {src or url}, Web"
            if published:
                header += f", Tarih: {published}"
            if url:
                header += f", URL: {url}"
            header += "]"
        else:
            header = f"[Kaynak {i}: {src}" + (f", Sayfa {page}" if page and str(page) not in {"", "?"} else "") + "]"
        content = (doc.page_content or "").strip()
        if content:
            compact_parts.append(f"{header}\n{content[:400]}")

    if not compact_parts:
        return ""

    compact_context = "\n\n---\n\n".join(compact_parts)
    only_web_context = bool(documents) and all(
        (getattr(doc, "metadata", {}) or {}).get("type") == "web_search"
        for doc in documents
    )
    missing_answer = (
        "Web araması bu soruyu yanıtlayacak güvenilir ve doğrudan eşleşen bilgi bulamadı."
        if only_web_context
        else "Bu bilgi yüklenen belgelerde yer almamaktadır."
    )
    source_label = "web kaynakları" if only_web_context else "bağlam"
    system_content = (
        f"Adın Frappe, bir RAG asistanısın. Sadece verilen {source_label}na dayanarak "
        "kullanıcının sorusunu aynı dilde, kısa ve doğrudan yanıtla. "
        f"Cevap yoksa sadece '{missing_answer}' yaz.\n\n"
        f"Bağlam:\n{compact_context}"
    )
    llm = _get_rag_llm(temperature=0.0, max_tokens=min(settings.rag_max_tokens, 512))
    response = await llm.ainvoke([
        SystemMessage(content=system_content),
        *select_recent_history(list(prior_messages), mode="rag"),
        HumanMessage(content=question),
    ])
    return _coerce_llm_text(response)


async def _micro_answer_retry(question: str, documents: list[Document]) -> str:
    """Top 3 belge, minimal prompt, kısa yanıt — son LLM denemesi."""
    if not documents:
        return ""
    chunks = []
    for doc in documents[:3]:
        chunk = (doc.page_content or "").strip()[:300]
        if chunk:
            chunks.append(chunk)
    if not chunks:
        return ""
    combined = "\n---\n".join(chunks)
    llm = _get_rag_llm(temperature=0.0, max_tokens=128)
    response = await llm.ainvoke([
        SystemMessage(content="Bu metinden soruya tek cümleyle yanıt ver. Cevap yoksa 'Bilgi bulunamadı.' yaz."),
        HumanMessage(content=f"Soru: {question}\n\nMetin: {combined}"),
    ])
    return _coerce_llm_text(response)


async def generator_node(state: AgentState) -> AgentState:
    """Belgeler ve/veya görsel bağlam varsa RAG ile, yoksa bağlamsız modda yanıt üretir.

    vision_context mevcutsa [Görsel Analizi] başlığıyla bağlamın başına eklenir.
    """
    t0 = time.perf_counter()
    question = state["question"]
    answer_question = state.get("original_question") or question
    documents = state.get("documents", [])
    prior_messages = list(state.get("messages", []))
    rag_history = select_recent_history(prior_messages, mode="rag")
    vision_context = state.get("vision_context", "")

    retrieval_trace = list(state.get("retrieval_trace") or [])
    used_chunk_ids: list[str] = []
    docs_included = 0
    input_budget_tokens = 0
    budget_chars = 0
    used_chars = 0
    overhead_tokens = 0
    context_truncated = False
    max_input_chars = 0
    retry_summary: dict[str, object] = {
        "empty_response": False,
        "compact_retry_answer_chars": 0,
        "micro_retry_answer_chars": 0,
        "fallback_context_answer_used": False,
        "retry_path": "primary",
    }

    if state.get("refusal_mode") or state.get("grader_reason") in {"insufficient_context", "irrelevant"}:
        generation = (
            "Bu soruyu yanıtlayabilecek yeterli bağlam yüklenen belgelerde bulunamadı. "
            "Bu yüzden bağlam dışı bir yanıt üretmiyorum."
        )
        new_messages = [
            *prior_messages,
            HumanMessage(content=answer_question),
            AIMessage(content=generation),
        ]
        return {
            **state,
            "generation": generation,
            "messages": new_messages,
            **_final_answer_fields(state, generation, t0=t0, mode="generator"),
        }

    if state.get("web_search_error") and not documents and not vision_context:
        generation = str(state.get("web_search_error") or "").strip()
        new_messages = [
            *prior_messages,
            HumanMessage(content=answer_question),
            AIMessage(content=generation),
        ]
        return {
            **state,
            "generation": generation,
            "messages": new_messages,
            **_final_answer_fields(state, generation, t0=t0, mode="generator"),
        }

    if documents or vision_context:
        session_max_tok = state.get("max_tokens") or settings.rag_max_tokens
        output_tokens = min(int(session_max_tok), settings.rag_max_tokens)
        assembly = assemble_rag_context(
            documents=documents,
            vision_context=vision_context,
            rag_history=rag_history,
            answer_question=answer_question,
            retrieval_trace=retrieval_trace,
            output_tokens=output_tokens,
            memory_preferences=state.get("memory_context") or "",
        )
        system_content = assembly.system_content
        used_chunk_ids = assembly.used_chunk_ids
        docs_included = assembly.docs_included
        input_budget_tokens = assembly.input_budget_tokens
        budget_chars = assembly.budget_chars
        used_chars = assembly.used_chars
        overhead_tokens = assembly.overhead_tokens
        context_truncated = assembly.truncated
        max_input_chars = assembly.max_input_chars
        if context_truncated:
            logger.warning(
                "Generator: context truncated to %dch to fit n_ctx=%d",
                len(assembly.context), settings.llm_context_size,
            )

        logger.info(
            "Generator: ctx_budget=%dtok/%dch, used=%dch, docs=%d/%d, vision=%s, prior=%d, "
            "n_ctx=%d, output_max=%dtok, overhead=%dtok",
            input_budget_tokens, budget_chars, used_chars,
            docs_included, len(documents), bool(vision_context),
            len(rag_history), settings.llm_context_size, output_tokens, overhead_tokens,
        )

        if used_chunk_ids:
            id_to_entry = {e["chunk_id"]: e for e in retrieval_trace}
            final_parts = []
            for cid in used_chunk_ids:
                e = id_to_entry.get(cid)
                if e:
                    final_parts.append(
                        f"{cid} (hybrid={_fmt_score(e.get('hybrid_score'))},"
                        f"rerank={_fmt_score(e.get('rerank_score'))})"
                    )
                else:
                    final_parts.append(cid)
            logger.info("Generator: final_used [n=%d] [%s]", len(used_chunk_ids), ", ".join(final_parts))
    else:
        system_content = RAG_NO_CONTEXT_SYSTEM_PROMPT
        logger.info(
            "Generator: no_context [prior=%d, n_ctx=%d, output_max=%dtok]",
            len(rag_history), settings.llm_context_size,
            state.get("max_tokens") or settings.rag_max_tokens,
        )

    session_temp = state.get("temperature") or 0.0
    if documents or vision_context:
        session_temp = min(float(session_temp), 0.2)
    session_max_tok = state.get("max_tokens") or None
    if documents or vision_context:
        session_max_tok = min(int(session_max_tok or settings.rag_max_tokens), settings.rag_max_tokens)
    llm = _get_rag_llm(temperature=session_temp, max_tokens=session_max_tok)

    messages_to_send = [SystemMessage(content=system_content)]
    messages_to_send.extend(rag_history)
    messages_to_send.append(HumanMessage(content=answer_question))

    t_llm = time.perf_counter()
    response = await llm.ainvoke(messages_to_send)
    generation = _coerce_llm_text(response)

    if not generation.strip() and (documents or vision_context):
        retry_summary["empty_response"] = True
        retry_summary["retry_path"] = "compact_retry"
        logger.warning("Generator: empty_response → compact retry")
        t_retry = time.perf_counter()
        try:
            generation = await _retry_generator_with_compact_context(
                question=answer_question,
                documents=documents,
                prior_messages=prior_messages,
                vision_context=vision_context,
            )
            logger.info(
                "Generator: compact_retry done [ans_len=%dch, t=%.3fs]",
                len(generation), time.perf_counter() - t_retry,
            )
            retry_summary["compact_retry_answer_chars"] = len(generation)
        except Exception as exc:
            logger.warning("Generator: compact_retry failed: %s", exc)
            generation = ""

    if not generation.strip() and documents:
        retry_summary["retry_path"] = "compact_retry>micro_retry"
        try:
            generation = await _micro_answer_retry(answer_question, documents)
            logger.info("Generator: micro_retry done [ans_len=%dch]", len(generation))
            retry_summary["micro_retry_answer_chars"] = len(generation)
        except Exception as exc:
            logger.warning("Generator: micro_retry failed: %s", exc)
            generation = ""

    if not generation.strip() and any((getattr(d, "metadata", {}) or {}).get("type") == "web_search" for d in documents):
        generation = _web_fallback_answer(answer_question, documents)
        retry_summary["fallback_context_answer_used"] = True
        retry_summary["retry_path"] = "web_structured_fallback"

    if not generation.strip() and (documents or vision_context):
        generation = _fallback_context_answer(answer_question, documents, vision_context)
        retry_summary["fallback_context_answer_used"] = True
        retry_summary["retry_path"] = "compact_retry>micro_retry>fallback_context"

    if generation.strip() and documents:
        generation = append_used_sources(generation, documents, answer_question)

    llm_elapsed = time.perf_counter() - t_llm
    total_elapsed = time.perf_counter() - t0
    logger.info(
        "Generator: done [ans_len=%dch, temp=%.2f, llm_t=%.3fs, total_t=%.3fs]",
        len(generation), session_temp,
        llm_elapsed, total_elapsed,
    )

    new_messages = [
        *prior_messages,
        HumanMessage(content=answer_question),
        AIMessage(content=generation),
    ]
    from src.observability.langsmith import (
        safe_preview,
        summarize_documents,
        summarize_retrieval_trace,
        summarize_source_distribution,
    )
    trace_summary = summarize_retrieval_trace(retrieval_trace)
    latency_ms = {
        "llm": round(llm_elapsed * 1000, 2),
        "total": round(total_elapsed * 1000, 2),
    }
    answer_preview = safe_preview(generation)
    document_previews = summarize_documents(documents)
    top_sources = summarize_source_distribution(documents)
    _observe_node(
        "frappe.generator_result",
        state,
        outputs={
            "answer_preview": answer_preview,
            "answer_chars": len(generation),
            "route": state.get("route", ""),
            "relevance": state.get("relevance", ""),
            "grader_reason": state.get("grader_reason", ""),
            "document_count": len(documents),
            "used_context_count": trace_summary.get("used_context_count", 0),
            "retrieval_trace_summary": trace_summary,
            "document_previews": document_previews,
            "top_sources": top_sources,
            "top_chunks": trace_summary.get("top_chunks", ""),
            "used_chunks": trace_summary.get("used_chunks", ""),
            "retry_summary": retry_summary,
            "retry_path": retry_summary["retry_path"],
            "context_truncated": context_truncated,
            "latency_ms_by_stage": latency_ms,
        },
        metadata={
            "answer_chars": len(generation),
            "docs_included": docs_included,
            "context_budget_tokens": input_budget_tokens,
            "context_budget_chars": budget_chars,
            "context_used_chars": used_chars,
            "context_overhead_tokens": overhead_tokens,
            "context_max_input_chars": max_input_chars,
            "context_truncated": context_truncated,
            "empty_response": retry_summary["empty_response"],
            "fallback_context_answer_used": retry_summary["fallback_context_answer_used"],
            "retry_path": retry_summary["retry_path"],
            "top_sources": top_sources,
        },
        tags=["frappe", "generator", state.get("route", "unknown") or "unknown"],
    )
    return {
        **state,
        "generation": generation,
        "messages": new_messages,
        "retrieval_trace": retrieval_trace,
        "answer_preview": answer_preview,
        "answer_chars": len(generation),
        "document_count": len(documents),
        "used_context_count": trace_summary.get("used_context_count", 0),
        "document_previews": document_previews,
        "retrieval_trace_summary": trace_summary,
        "top_sources": top_sources,
        "top_chunks": trace_summary.get("top_chunks", ""),
        "used_chunks": trace_summary.get("used_chunks", ""),
        "retry_summary": retry_summary,
        "retry_path": retry_summary["retry_path"],
        "latency_ms_by_stage": latency_ms,
    }




async def web_search_node(state: AgentState) -> AgentState:
    """Belge alaka düşük veya bulunamadığında web araması yapar.

    Sağlayıcı politikası: yalnızca Tavily. Düşük kaliteli/sıfır sonuçta refusal'a düşer.
    Web araması için orijinal soru kullanılır — rewriter çıktısı web için uygun değildir.
    """
    # Orijinal soru vektör DB için yeniden yazılmış olabilir; web için doğal dili tercih et.
    t0 = time.perf_counter()
    question = state.get("original_question") or state["question"]
    existing_docs = state.get("documents", [])
    search_query = _build_contextual_web_query(question, list(state.get("messages", [])))
    explicit_url_docs = await _docs_from_explicit_urls(question)

    async with cl.Step(name="Web Search", type="tool") as step:
        step.input = search_query

        service = _get_web_search_service()
        result = await service.search(search_query) if service else None

        if result is None and not explicit_url_docs:
            logger.warning("Web search: Tavily kullanılamıyor veya sonuç yok")
            step.output = "Web search failed."
            _observe_node(
                "frappe.web_search_result",
                state,
                inputs={"query_preview": search_query[:180]},
                outputs={
                    "status": "unavailable" if service is None else "no_result",
                    "provider": "tavily" if service is not None else "",
                    "result_count": 0,
                    "latency_ms_by_stage": {"total": round((time.perf_counter() - t0) * 1000, 2)},
                },
                metadata={
                    "web_query_preview": search_query[:180],
                    "web_query_hash": _hash_text(search_query),
                    "web_result_count": 0,
                },
                tags=["frappe", "web-search", "no-result"],
            )
            err = (
                "Canlı web araması şu anda devre dışı çünkü `TAVILY_API_KEY` ayarlanmamış. "
                "Web araması için `.env` içine `TAVILY_API_KEY` ekleyip uygulamayı yeniden başlatmalısın."
                if service is None
                else "Web araması sonuç döndürmedi. Lütfen sorguyu biraz daraltıp tekrar deneyin."
            )
            return {**state, "documents": existing_docs, "web_search_error": err}

        web_docs = explicit_url_docs + (_web_docs_from_result(result, query=search_query) if result else [])
        step.output = (
            f"Fetched {len(explicit_url_docs)} explicit URL(s) and found content via {result.provider} ({len(result.text)} chars)."
            if result and explicit_url_docs
            else f"Found content via {result.provider} ({len(result.text)} chars)."
            if result
            else f"Fetched {len(explicit_url_docs)} explicit URL(s)."
        )
        if result:
            step.elements = [
                cl.Text(
                    name=f"{result.provider.title()} Results",
                    content=result.text[:2000] + "...",
                    display="inline",
                )
            ]

    domains = sorted({_web_domain(str((d.metadata or {}).get("url") or (d.metadata or {}).get("source") or "")) for d in web_docs})
    domains = [d for d in domains if d]
    dated = [str((d.metadata or {}).get("published") or "") for d in web_docs if (d.metadata or {}).get("published")]
    provider_name = result.provider if result else "direct_url"
    _observe_node(
        "frappe.web_search_result",
        state,
        inputs={"query_preview": search_query[:180]},
        outputs={
            "status": "success",
            "provider": provider_name,
            "result_count": len(web_docs),
            "top_domains": " | ".join(domains[:8]),
            "freshest_published": max(dated) if dated else "",
            "latency_ms_by_stage": {"total": round((time.perf_counter() - t0) * 1000, 2)},
        },
        metadata={
            "web_provider": provider_name,
            "web_query_preview": search_query[:180],
            "web_query_hash": _hash_text(search_query),
            "web_result_count": len(web_docs),
            "top_domains": " | ".join(domains[:8]),
        },
        tags=["frappe", "web-search", "success"],
    )
    return {**state, "documents": existing_docs + web_docs}




_COMPOUND_QUERY_MARKERS = re.compile(
    r"(etkinlik|konser|festival|fuar|sergi|event|activity|activities"
    r"|haber|news|fiyat|price|skor|score|borsa|kur|exchange"
    r"|nerede|nereye|ne zaman|hangi|what|where|when|which"
    # Çok-günlük tahmin / tablo istekleri — format_weather() yerine _fast_web_summarize()
    r"|\d\s*g[üu]nl[üu]k|haftal[ıi]k|tablo|table|forecast|tahmin|g[üu]nl[üu]k\s*tahmin)",
    re.IGNORECASE,
)


_FOLLOWUP_PRICE_RE = re.compile(
    r"(fiyat[ıi]?|kaç\s+para|ne\s+kadar|hisse|stock|price|de[ğg]eri|value)",
    re.IGNORECASE | re.UNICODE,
)
_EXPLICIT_WEB_ENTITY_RE = re.compile(
    r"\b(tesla|tsla|apple|iphone|samsung|xiaomi|alt[ıi]n|gram|euro|dolar|usd|eur|try|"
    r"btc|bitcoin|ethereum|nasdaq|borsa|nvda|nvidia|aapl|msft)\b|\b\d+\s*gb\b",
    re.IGNORECASE | re.UNICODE,
)
_PRODUCT_SUBJECT_PATTERNS = [
    re.compile(
        r"(?:telefon\s+modeli|modeli|cihaz[ıi]?|ürün[üu]?)[^,\n:]*[:,]\s*"
        r"(.{3,90}?)(?:'?[dt][ıi]r|dir|tir|tır|\(|\.|\n|$)",
        re.IGNORECASE | re.UNICODE,
    ),
    re.compile(
        r"\b((?:APPLE\s+)?IPHONE\s*\d{1,2}(?:\s+(?:PRO|PLUS|MINI|PRO MAX))?"
        r"(?:\s+\d+\s*GB)?(?:\s+[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ\s]{2,30})?)\b",
        re.IGNORECASE | re.UNICODE,
    ),
    re.compile(r"\b(TESLA|TSLA|APPLE|AAPL|NVIDIA|NVDA|MICROSOFT|MSFT)\b", re.IGNORECASE),
]
_WEB_QUERY_MAX_CHARS = 360
_CURRENCY_ENTITY_RE = re.compile(
    r"\b(euro|eur|dolar|dollar|usd|try|tl|sterlin|gbp|alt[ıi]n|gram|bitcoin|btc|ethereum|eth)\b",
    re.IGNORECASE | re.UNICODE,
)


def _message_text(message: object) -> str:
    """Kısa: `_message_text` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
    if isinstance(message, dict):
        return str(message.get("content") or "")
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return str(content or "")


def _clean_subject(subject: str) -> str:
    """Kısa: `_clean_subject` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
    subject = re.split(r"\bKaynaklar?:|\bSources:", subject, maxsplit=1, flags=re.IGNORECASE)[0]
    subject = re.sub(r"\s*\[?Kaynak\s+\d+\]?.*$", "", subject, flags=re.IGNORECASE).strip()
    subject = re.sub(r"\s+", " ", subject).strip(" .,:;\"'`")
    return subject[:90]


def _extract_web_subject_from_history(messages: list) -> str:
    """Follow-up web queries için son konuşmadan ürün/hisse konusunu çıkarır."""
    for message in reversed(messages[-8:]):
        text = _clean_subject(_message_text(message))
        if not text:
            continue
        for pattern in _PRODUCT_SUBJECT_PATTERNS:
            match = pattern.search(text)
            if match:
                subject = _clean_subject(match.group(1))
                if 3 <= len(subject) <= 90:
                    return subject
    return ""


def _build_contextual_web_query(question: str, prior_messages: list) -> str:
    """Arama sorgusunu bağlamla zenginleştirir; özellikle 'şu anki fiyatı?' follow-up'ları."""
    normalized = normalize_web_query(question)
    q = question.strip()
    if not _FOLLOWUP_PRICE_RE.search(q):
        return _compact_web_query(normalized)
    today = datetime.date.today().isoformat()
    q_lower = q.lower()
    if _EXPLICIT_WEB_ENTITY_RE.search(q):
        if re.search(r"\b(hisse|stock|borsa)\b", q_lower):
            return _compact_web_query(f"{normalized} {today} official quote Nasdaq Yahoo Finance MarketWatch")
        if re.search(r"\b(fiyat|price)\b", q_lower) and re.search(r"\b(iphone|telefon|phone|apple|samsung|xiaomi)\b", q_lower):
            return _compact_web_query(f"{normalized} Türkiye {today} resmi satıcı fiyat karşılaştırma")
        return _compact_web_query(normalized)

    subject = _extract_web_subject_from_history(prior_messages)
    if not subject:
        return _compact_web_query(normalized)

    if re.search(r"\b(hisse|stock|borsa)\b", q_lower):
        return _compact_web_query(f"{subject} stock price today {today} official quote market")
    if re.search(r"\b(iphone|apple|samsung|xiaomi|telefon|phone)\b", subject, re.IGNORECASE):
        return _compact_web_query(f"{subject} güncel fiyat Türkiye {today} resmi satıcı teknoloji mağazaları")
    return _compact_web_query(f"{subject} {normalized} {today}")


def _compact_web_query(query: str, *, max_chars: int = _WEB_QUERY_MAX_CHARS) -> str:
    """Keep Tavily queries below its hard limit and strip pasted conversation noise."""
    q = re.sub(r"\s+", " ", (query or "").strip())
    if len(q) <= max_chars:
        return q

    first = re.split(r"[.!?\n]", q, maxsplit=1)[0].strip()
    entities = " ".join(dict.fromkeys(m.group(0).upper() for m in _CURRENCY_ENTITY_RE.finditer(q)))
    today = datetime.date.today().isoformat()
    if entities and re.search(r"\b(fiyat|price|kur|exchange|g[üu]ncel|today|bug[üu]n)\b", q, re.IGNORECASE):
        compact = f"{entities} current exchange rate price {today}"
        return compact[:max_chars].rstrip()
    if len(first) >= 20:
        return first[:max_chars].rstrip()
    return q[:max_chars].rstrip()


def _web_fallback_answer(question: str, documents: list[Document]) -> str:
    """Kısa: `_web_fallback_answer` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
    web_docs = [d for d in documents if (getattr(d, "metadata", {}) or {}).get("type") == "web_search"]
    if not web_docs:
        return ""
    lines: list[str] = []
    header = "Web kaynaklarından bulunanlar:" if is_turkish_query(question) else "Found from web sources:"
    lines.append(header)
    for idx, doc in enumerate(web_docs[:5], 1):
        meta = getattr(doc, "metadata", {}) or {}
        title = meta.get("display_name") or meta.get("title") or meta.get("source") or f"Kaynak {idx}"
        url = meta.get("url") or meta.get("source") or ""
        published = f" ({meta.get('published')})" if meta.get("published") else ""
        snippet = re.sub(r"\s+", " ", (doc.page_content or "").strip())
        snippet = re.sub(r"^(Title|Published|URL|Snippet):\s*", "", snippet)
        lines.append(f"- [Kaynak {idx}] {title}{published}: {snippet[:220]}".rstrip())
        if url:
            lines.append(f"  {url}")
    return "\n".join(lines)


def _is_pure_weather_query(question: str) -> bool:
    """Sorgu yalnızca hava durumu soruyorsa True döner; compound sorgular False."""
    return not bool(_COMPOUND_QUERY_MARKERS.search(question))


async def _fast_web_summarize(
    question: str,
    result_text: str,
    prior_messages: list | None = None,
    *,
    search_query: str = "",
) -> str:
    """Web sonuçlarını LLM çağrısıyla özetler; entity isimleri ve rakamları çıkarır."""
    system = (
        "You answer ONLY from the provided web search results.\n"
        "Rules:\n"
        "- Respond in the same language as the user's question.\n"
        "- Turkish question → fully Turkish answer.\n"
        "- Never say you cannot access live data or the internet.\n"
        "- Never repeat the user's question at the start.\n"
        "- Treat the Search query as the intended entity/topic. Ignore results about a different entity, commodity, product, city, ticker, or date.\n"
        "- Extract SPECIFIC entities: names, prices, percentages, dates, company names.\n"
        "- PRICE/STOCK RULE: Give the latest single value when available, with currency, timestamp/date, market status if present, and one short caveat if sources differ.\n"
        "- RECENCY RULE: Prefer the source/result with the latest explicit date or market timestamp. Do NOT list stale historical values unless needed to explain conflict.\n"
        "- If sources conflict, pick the one with the latest date and note it briefly.\n"
        "- Cite each important value/date with bracket citations like [1] or [2], using the result numbers in the provided web results.\n"
        "- Format: use bullet points for multiple facts; prose for single answers.\n"
        "- Do NOT say 'I don't have real-time data' — use what's in the web results.\n"
        "- If the web results do not contain the requested entity/topic, say that reliable matching results were not found; do not answer a different topic.\n"
        "TABLE RULE: If the user asked for a table (tablo, table) OR a multi-day forecast "
        "(5 günlük, haftalık, X-Y arası), extract each day's data from the web results and "
        "present it as a Markdown table with columns: | Gün | Tarih | Hava | Max °C | Min °C |. "
        "Fill only the columns that appear in the results; omit columns with no data. "
        "Do NOT redirect the user to external sites — build the table yourself from the data.\n"
    )
    llm = _get_rag_llm(temperature=0.0)
    messages_to_send = [SystemMessage(content=system)]
    if prior_messages:
        messages_to_send.extend(select_recent_history(list(prior_messages), mode="direct"))
    messages_to_send.append(
        HumanMessage(content=f"Question: {question}\nSearch query: {search_query or question}\n\nWeb results:\n{result_text[:6500]}")
    )
    response = await llm.ainvoke(messages_to_send)
    text = (response.content or "").strip()
    return WebResultFormatter.append_sources(text, result_text, question)


async def direct_response_node(state: AgentState) -> AgentState:
    """Doğrudan yanıt node'u.

    Web sorguları için hızlı yol: ReAct döngüsüne girmeden önce web sonucunu
    özetler ve döner. Geri kalan sorgular için araçlı ReAct agent çalıştırılır.
    """
    t0 = time.perf_counter()
    question = state["question"]
    prior_messages = list(state.get("messages", []))
    direct_history = select_recent_history(prior_messages, mode="direct")

    # Hızlı yol — gerçek zamanlı web sorguları
    if is_web_query(question):
        service = _get_web_search_service()
        search_query = _build_contextual_web_query(question, prior_messages)
        logger.info(
            "Direct: web_fast [query='%.80s', prior=%d]",
            search_query, len(prior_messages),
        )
        t_search = time.perf_counter()
        web_result = await service.search(search_query) if service else None
        if web_result:
            logger.info(
                "Direct: web_result [provider=%s, chars=%d, search_t=%.3fs]",
                web_result.provider, len(web_result.text),
                time.perf_counter() - t_search,
            )
            t_sum = time.perf_counter()
            if settings.weather_specialization_enabled and is_weather_query(question) and _is_pure_weather_query(question):
                answer = WebResultFormatter.format_weather(question, web_result.text)
                logger.info("Direct: weather_format [ans_len=%dch, t=%.3fs]", len(answer), time.perf_counter() - t_sum)
            else:
                answer = await _fast_web_summarize(
                    question,
                    web_result.text,
                    direct_history,
                    search_query=search_query,
                )
                if not answer.strip():
                    answer = _web_fallback_answer(question, _web_docs_from_result(web_result, query=search_query))
                logger.info(
                    "Direct: web_summarize [ans_len=%dch, llm_t=%.3fs, total_t=%.3fs]",
                    len(answer), time.perf_counter() - t_sum, time.perf_counter() - t0,
                )

            new_messages = [
                *prior_messages,
                HumanMessage(content=question),
                AIMessage(content=answer),
            ]
            return {**state, "generation": answer, "messages": new_messages, **_final_answer_fields(state, answer, t0=t0, mode="direct_response")}
        else:
            logger.warning("Direct: web_no_result [search_t=%.3fs]", time.perf_counter() - t_search)
            if service is None:
                answer = (
                    "Canlı web araması şu anda devre dışı çünkü `TAVILY_API_KEY` ayarlanmamış. "
                    "Bu yüzden güncel hava durumu verisini güvenilir şekilde çekemiyorum. "
                    "Web araması için `.env` içine `TAVILY_API_KEY` ekleyip uygulamayı yeniden başlatmalısın."
                )
            else:
                answer = (
                    "Web araması sonuç döndürmedi. Canlı hava durumu gibi güncel bilgiler için "
                    "web sağlayıcısını kontrol edip tekrar deneyebilirsin."
                )
            new_messages = [
                *prior_messages,
                HumanMessage(content=question),
                AIMessage(content=answer),
            ]
            return {**state, "generation": answer, "messages": new_messages, **_final_answer_fields(state, answer, t0=t0, mode="direct_response")}

    if _DATE_QUERY_RE.search(question):
        _today = datetime.date.today()
        _months_tr = {1:"Ocak",2:"Şubat",3:"Mart",4:"Nisan",5:"Mayıs",6:"Haziran",
                      7:"Temmuz",8:"Ağustos",9:"Eylül",10:"Ekim",11:"Kasım",12:"Aralık"}
        _days_tr = {0:"Pazartesi",1:"Salı",2:"Çarşamba",3:"Perşembe",4:"Cuma",5:"Cumartesi",6:"Pazar"}
        answer = f"{_today.day} {_months_tr[_today.month]} {_today.year}, {_days_tr[_today.weekday()]}."
        logger.info("Direct: date_fast [ans='%s', total_t=%.3fs]", answer, time.perf_counter() - t0)
        new_messages = [*prior_messages, HumanMessage(content=question), AIMessage(content=answer)]
        return {**state, "generation": answer, "messages": new_messages, **_final_answer_fields(state, answer, t0=t0, mode="direct_response")}

    if _PLAIN_DIRECT_ARITH_RE.fullmatch(question.strip()):
        try:
            answer = _safe_eval_math_expr(question)
        except Exception as exc:
            answer = f"Hesaplama hatası: {exc}"
        logger.info("Direct: calc_fast [q_len=%d, total_t=%.3fs]", len(question), time.perf_counter() - t0)
        new_messages = [
            *prior_messages,
            HumanMessage(content=question),
            AIMessage(content=answer),
        ]
        return {**state, "generation": answer, "messages": new_messages, **_final_answer_fields(state, answer, t0=t0, mode="direct_response")}

    if _should_use_math_direct_llm(question):
        logger.info("Direct: math_chat [prior=%d, q_len=%d]", len(prior_messages), len(question))
        system_prompt = (
            "Sen kısa ve doğru matematik çözen bir asistansın.\n"
            "Kullanıcının dilinde yanıt ver. Gereken ara adımları kısa göster.\n"
            "Sonucu net biçimde yaz. Görsel, web veya belge bağlamı yoksa bunlardan bahsetme."
        )
        llm = _get_chat_llm(max_tokens=int(state.get("max_tokens") or settings.chat_max_tokens))
        t_math = time.perf_counter()
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=question),
        ])
        generation = getattr(response, "content", "") or ""
        logger.info(
            "Direct: math_done [ans_len=%dch, llm_t=%.3fs, total_t=%.3fs]",
            len(generation), time.perf_counter() - t_math, time.perf_counter() - t0,
        )
        new_messages = [
            *prior_messages,
            HumanMessage(content=question),
            AIMessage(content=generation),
        ]
        return {**state, "generation": generation, "messages": new_messages, **_final_answer_fields(state, generation, t0=t0, mode="direct_response")}

    # Hızlı yol — araç gerektirmeyen kısa sohbet/genel direct yanıtlar.
    # ReAct agent tool şemalarını prompt'a eklediği için küçük yerel modellerde
    # basit mesajlarda gereksiz gecikme yaratır.
    if _should_use_plain_direct_llm(question):
        logger.info("Direct: plain_chat [prior=%d, q_len=%d]", len(prior_messages), len(question))
        _today = datetime.date.today()
        _months_tr = {1:"Ocak",2:"Şubat",3:"Mart",4:"Nisan",5:"Mayıs",6:"Haziran",
                      7:"Temmuz",8:"Ağustos",9:"Eylül",10:"Ekim",11:"Kasım",12:"Aralık"}
        _days_tr = {0:"Pazartesi",1:"Salı",2:"Çarşamba",3:"Perşembe",4:"Cuma",5:"Cumartesi",6:"Pazar"}
        _date_str = f"{_today.day} {_months_tr[_today.month]} {_today.year}, {_days_tr[_today.weekday()]}"
        system_prompt = (
            f"GÜNCEL TARİH: {_date_str}.\n"
            "Sen bir yapay zeka asistanısın. Adın Frappe'dir (başka ismin yok).\n"
            "İsim sorulursa sadece 'Frappe' de; 'Sen Frappe' veya 'Ben Sen' yazma.\n"
            "Kullanıcının diliyle yanıt ver. Türkçe soru → Türkçe yanıt.\n"
            "Kısa ama samimi ol. Selamlamalara sıcak ve doğal yanıt ver (1-2 cümle).\n"
            "Soruyu başta tekrar etme; emoji kullanma."
        )
        if is_direct_support_query(question):
            system_prompt += (
                "\nKullanıcı devam etmeni, yarım kalan cevabı tamamlamanı veya cevap kesilmesini açıklamanı "
                "istiyorsa web arama yapmadan son asistan cevabından devam et ya da teknik nedeni kısa açıkla."
            )
        llm = _get_chat_llm(max_tokens=int(state.get("max_tokens") or settings.chat_max_tokens))
        messages_to_send = [SystemMessage(content=system_prompt)]
        messages_to_send.extend(direct_history)
        messages_to_send.append(HumanMessage(content=question))
        t_plain = time.perf_counter()
        response = await llm.ainvoke(messages_to_send)
        generation = getattr(response, "content", "") or ""
        logger.info(
            "Direct: plain_done [ans_len=%dch, llm_t=%.3fs, total_t=%.3fs]",
            len(generation), time.perf_counter() - t_plain, time.perf_counter() - t0,
        )
        new_messages = [
            *prior_messages,
            HumanMessage(content=question),
            AIMessage(content=generation),
        ]
        return {**state, "generation": generation, "messages": new_messages, **_final_answer_fields(state, generation, t0=t0, mode="direct_response")}

    # Normal yol — araçlı ReAct agent (backend capability dependent)
    from langgraph.prebuilt import create_react_agent
    from src.tools.search import tavily_search
    from src.tools.file_reader import read_uploaded_file
    from src.tools.calculator import calculator
    from src.tools.mcp_bridge import mcp_call
    from src.mcp.mcp_client import get_mcp_tools

    mcp_tools: list = []
    try:
        cached = cl.user_session.get("mcp_langchain_tools")
        if isinstance(cached, list) and cached:
            mcp_tools = cached
    except Exception:
        pass

    if not mcp_tools and needs_mcp_tools(question):
        try:
            mcp_tools = await get_mcp_tools()
        except Exception as exc:
            logger.warning("MCP araçları yüklenemedi: %s", exc)

    base_tools = [tavily_search, calculator, read_uploaded_file, mcp_call]
    all_tools = _get_deduped_tools_cached(mcp_tools, base_tools)

    system_prompt = build_generator_prompt(all_tools)
    llm = _get_agent_llm()
    backend = (settings.llm_backend or "").lower().strip()

    if backend in {"llama.cpp", "llamacpp", "llama"} and needs_mcp_tools(question):
        logger.info("Direct: react_skip [reason=llamacpp_no_tools, prior=%d]", len(prior_messages))
        messages_to_send = [SystemMessage(content=system_prompt)]
        messages_to_send.extend(direct_history)
        messages_to_send.append(HumanMessage(content=question))
        response = await llm.ainvoke(messages_to_send)
        generation = getattr(response, "content", "") or ""
        new_messages = [
            *prior_messages,
            HumanMessage(content=question),
            AIMessage(content=generation),
        ]
        return {**state, "generation": generation, "messages": new_messages, **_final_answer_fields(state, generation, t0=t0, mode="direct_response")}

    logger.info(
        "Direct: react_agent [tools=%d, prior=%d, backend=%s]",
        len(all_tools), len(prior_messages), backend,
    )
    t_react = time.perf_counter()

    # Cache the compiled ReAct graph per-session (recompiling costs ~50-100ms each time).
    llm_key = f"{type(llm).__name__}_{getattr(llm, 'model_name', getattr(llm, 'model', ''))}"
    agent_cache_key = (tuple(sorted(getattr(t, "name", "") for t in all_tools)), llm_key)
    try:
        _cached_agent = cl.user_session.get("_react_agent")
        _cached_agent_key = cl.user_session.get("_react_agent_key")
        if _cached_agent is not None and _cached_agent_key == agent_cache_key:
            agent = _cached_agent
        else:
            agent = create_react_agent(llm, all_tools, prompt=system_prompt)
            cl.user_session.set("_react_agent", agent)
            cl.user_session.set("_react_agent_key", agent_cache_key)
    except Exception:
        agent = create_react_agent(llm, all_tools, prompt=system_prompt)

    result = await agent.ainvoke({"messages": direct_history + [HumanMessage(content=question)]})

    generation = result["messages"][-1].content
    logger.info(
        "Direct: react_done [ans_len=%dch, react_t=%.3fs, total_t=%.3fs]",
        len(generation), time.perf_counter() - t_react, time.perf_counter() - t0,
    )
    new_messages = [
        *prior_messages,
        HumanMessage(content=question),
        AIMessage(content=generation),
    ]
    return {**state, "generation": generation, "messages": new_messages, **_final_answer_fields(state, generation, t0=t0, mode="direct_response")}




def _dedupe_tools(tools: list) -> list:
    """İsme göre aynı araçların tekrarını kaldırır."""
    seen: set[str] = set()
    result = []
    for tool in tools:
        name = getattr(tool, "name", "") or ""
        if name and name not in seen:
            seen.add(name)
            result.append(tool)
    return result


def _get_deduped_tools_cached(mcp_tools: list, base_tools: list) -> list:
    """Dedup sonucunu user_session'da cache'ler; MCP tool seti değişmezse yeniden hesaplamaz."""
    try:
        mcp_names = tuple(getattr(t, "name", "") for t in mcp_tools)
        cached = cl.user_session.get("_deduped_tools_cache")
        cached_key = cl.user_session.get("_deduped_tools_key")
        if cached is not None and cached_key == mcp_names:
            return cached
        result = _dedupe_tools(mcp_tools + base_tools)
        cl.user_session.set("_deduped_tools_cache", result)
        cl.user_session.set("_deduped_tools_key", mcp_names)
        return result
    except Exception:
        return _dedupe_tools(mcp_tools + base_tools)
