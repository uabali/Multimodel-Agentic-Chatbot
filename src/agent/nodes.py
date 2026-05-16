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
import operator
import logging
import re
import threading
import time

import chainlit as cl
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agent.state import AgentState
from src.agent.routing import keyword_route, is_web_query, needs_mcp_tools, is_weather_query, normalize_web_query
from src.agent.web_search import WebSearchService, WebResultFormatter
from src.agent.prompts import (
    ROUTER_SYSTEM_PROMPT,
    REWRITER_SYSTEM_PROMPT,
    GRADER_SYSTEM_PROMPT,
    RAG_WITH_CONTEXT_SYSTEM_PROMPT,
    RAG_NO_CONTEXT_SYSTEM_PROMPT,
    build_generator_prompt,
    select_vision_prompt,
)
from src.config import settings

logger = logging.getLogger(__name__)


def _history_turn_count(messages: list) -> int:
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
    char_budget = 2500 if mode == "rag" else 3500

    system_messages = [m for m in messages if isinstance(m, SystemMessage)]
    chat_messages = [m for m in messages if not isinstance(m, SystemMessage)]
    selected = list(chat_messages[-max_messages:])

    def _chars(items: list) -> int:
        return sum(len(getattr(m, "content", "") or "") for m in items)

    total = _chars(selected)
    while total > char_budget and len(selected) > 2:
        drop_count = (
            2
            if len(selected) >= 2
            and isinstance(selected[0], HumanMessage)
            and isinstance(selected[1], AIMessage)
            else 1
        )
        for _ in range(drop_count):
            removed = selected.pop(0)
            total -= len(getattr(removed, "content", "") or "")
    while total > char_budget and selected:
        removed = selected.pop(0)
        total -= len(getattr(removed, "content", "") or "")
    while selected and isinstance(selected[0], AIMessage):
        selected.pop(0)

    return system_messages[:1] + selected



# ─────────────────────────────────────────────────────────────────────────────
# LLM fabrika erişimleri — DIP: node'lar doğrudan ChatOpenAI yaratmaz
# ─────────────────────────────────────────────────────────────────────────────


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


def _get_chat_llm():
    """Basit sohbet için araçsız, küçük prompt'lu LLM."""
    from src.rag.llm import get_chat_llm
    return get_chat_llm()


def reset_nodes_llm_cache() -> None:
    """LLM ayarları runtime'da değiştiğinde (api/router.py) çağrılır."""
    global _router_llm_cache
    _router_llm_cache = None
    _rag_llm_cache.clear()


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


# ─────────────────────────────────────────────────────────────────────────────
# Reranker kayıt defteri — modül-level global state'i kapsüller (SRP)
# ─────────────────────────────────────────────────────────────────────────────


class _RerankerRegistry:
    """Reranker instance'ını lazy olarak yükler ve önbellekte tutar."""

    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get(cls):
        if cls._instance is not None:
            return cls._instance
        if not settings.use_rerank:
            return None
        with cls._lock:
            if cls._instance is not None:
                return cls._instance
            try:
                from src.rag.reranker import create_reranker
                cls._instance = create_reranker(
                    model_name=settings.reranker_model,
                    device=settings.reranker_device,
                )
            except Exception as exc:
                logger.warning("Reranker yüklenemedi (devre dışı): %s", exc)
                cls._instance = None
        return cls._instance


# ─────────────────────────────────────────────────────────────────────────────
# WebSearchService singleton — her çağrıda provider listesi yeniden kurulmaz
# ─────────────────────────────────────────────────────────────────────────────

_web_search_service = None
_web_search_service_loaded = False


def _get_web_search_service():
    global _web_search_service, _web_search_service_loaded
    if not _web_search_service_loaded:
        _web_search_service = WebSearchService.from_settings()
        _web_search_service_loaded = True
    return _web_search_service


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — Router
# ─────────────────────────────────────────────────────────────────────────────


def _parse_route(text: str, default: str = "direct") -> str:
    """LLM yanıtından 'rag', 'direct' veya 'vision' çıkarır (regex tabanlı)."""
    import re
    text_lower = text.lower().strip()
    if re.search(r'\brag\b', text_lower):
        return "rag"
    if re.search(r'\bdirect\b', text_lower):
        return "direct"
    if re.search(r'\bvision\b', text_lower):
        return "vision"
    if re.search(r'"route"\s*:\s*"rag"', text_lower):
        return "rag"
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
                "Router → direct [reason=web_override+uploads, uploads=%d, q_len=%d, t=%.3fs]",
                len(session_uploads), q_len, time.perf_counter() - t0,
            )
            _observe_node(
                "frappe.router_decision",
                state,
                outputs={"route": "direct", "route_reason": "web_override+uploads", "elapsed_ms": elapsed_ms},
                metadata={
                    "route_reason": "web_override+uploads",
                    "query_chars": q_len,
                    "image_count": 0,
                    "upload_count": len(session_uploads),
                },
                tags=["frappe", "router", "direct"],
            )
            return {**state, "route": "direct"}
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


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — Query Rewriter
# ─────────────────────────────────────────────────────────────────────────────

# Çok-turlu follow-up sorgularında rewriter gereklidir (referans çözümlemesi).
_FOLLOW_UP_MARKERS: frozenset[str] = frozenset({
    "bunu", "buna", "bunda", "bunun", "bunları", "bunlari", "bununla",
    "önceki", "onceki", "bahsettiğin", "bahsettigin",
    "söylediğin", "soyledigin", "yukarıdaki", "yukaridaki",
    "this", "that", "it", "these", "those", "above", "previous",
})

# Kısa sorgularda soru kelimesi varsa rewriter'a gerek yok.
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
    if prior_messages:
        messages_to_send.extend(prior_messages[-2:])
    messages_to_send.append(HumanMessage(content=question))

    # Dense gate embedding ve LLM rewrite paralelde — retriever_node LRU cache'ten hızlı alır
    async def _warm_embed_cache():
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


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — Retriever (hybrid + dense gate + reranking)
# ─────────────────────────────────────────────────────────────────────────────


def _build_source_filter(source_filter: str, session_uploads: list[str] | None = None):
    """source_filter veya session_uploads'dan Qdrant metadata filtresi oluşturur.

    source_filter verilmişse (mevcut yüklemenin dosya adı) → tek değer eşleşmesi.
    Yoksa ve session_uploads doluysa → bu dosyaların herhangi biriyle eşleşme.
    İkisi de boşsa None döner (filtresiz arama).
    """
    from qdrant_client import models as qmodels
    if source_filter:
        return qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="metadata.source_file",
                    match=qmodels.MatchValue(value=source_filter),
                )
            ]
        )
    uploads = [s for s in (session_uploads or []) if s]
    if uploads:
        return qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="metadata.source_file",
                    match=qmodels.MatchAny(any=uploads),
                )
            ]
        )
    return None


async def retriever_node(state: AgentState) -> AgentState:
    """Hybrid retrieval + dense gate + opsiyonel reranking uygular."""
    t0 = time.perf_counter()
    question = state["question"]
    source_filter = state.get("source_filter", "")
    session_uploads = state.get("session_uploads") or []
    latency_ms: dict[str, float] = {}
    dense_score = None
    strategy = state.get("retrieval_strategy") or settings.retrieval_strategy
    use_rerank_val = state.get("use_rerank")
    if use_rerank_val is None:
        use_rerank_val = settings.use_rerank

    try:
        from src.rag.vectorstore import get_hybrid_store
        from src.rag.retriever import create_retriever, deduplicate_documents, run_retriever, chunk_id

        store = get_hybrid_store()
        qdrant_filter = _build_source_filter(source_filter, session_uploads)

        if source_filter or session_uploads:
            dense_score = 1.0
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
                "Retriever: dense_gate=%.3f [threshold=%.3f, t=%.3fs]",
                dense_score, settings.rag_min_dense_similarity,
                time.perf_counter() - t_gate,
            )
            if dense_score < settings.rag_min_dense_similarity:
                elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
                logger.info(
                    "Retriever: gate_reject [score=%.3f < %.3f, t=%.3fs]",
                    dense_score, settings.rag_min_dense_similarity,
                    time.perf_counter() - t0,
                )
                _observe_node(
                    "frappe.retriever_result",
                    state,
                    outputs={
                        "status": "gate_reject",
                        "document_count": 0,
                        "dense_score": dense_score,
                        "dense_threshold": settings.rag_min_dense_similarity,
                        "latency_ms_by_stage": {**latency_ms, "total": elapsed_ms},
                    },
                    metadata={
                        "retrieval_strategy": strategy,
                        "use_rerank": bool(use_rerank_val),
                        "dense_gate_rejected": True,
                        "dense_score": dense_score,
                        "dense_threshold": settings.rag_min_dense_similarity,
                    },
                    tags=["frappe", "retriever", "gate-reject"],
                )
                return {**state, "documents": [], "retrieval_trace": []}

        retriever = create_retriever(
            vectorstore=store.store,
            question=question,
            strategy=strategy,
            base_k=settings.base_k,
            max_k=settings.top_k,
            use_rerank=use_rerank_val,
            reranker=_RerankerRegistry.get(),
            rerank_top_n=settings.rerank_top_n,
            qdrant_filter=qdrant_filter,
        )
        t_fetch = time.perf_counter()
        documents = await asyncio.to_thread(run_retriever, retriever, question)
        documents = deduplicate_documents(documents, max_docs=settings.top_k)
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
                "top_sources": summarize_source_distribution(documents),
            },
            tags=["frappe", "retriever", "success"],
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

    return {**state, "documents": documents, "retrieval_trace": retrieval_trace}


def _fmt_score(s) -> str:
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


# ─────────────────────────────────────────────────────────────────────────────
# Node 4 — Grader (CRAG-style belge alaka değerlendirmesi)
# ─────────────────────────────────────────────────────────────────────────────


MAX_GRADER_DOCS = 5

# Confidence eşikleri: yüksek/düşük durumda LLM atlanır (~3s kazanç).
_GRADER_CONF_HIGH = 0.75  # Bu eşiğin üstünde → doğrudan "yes" (LLM atlanır)
_GRADER_CONF_LOW  = 0.08  # Bu eşiğin altında  → doğrudan "no"  (LLM atlanır)


def _parse_yes_no(text: str, default: str = "no") -> str:
    """LLM yanıtından 'yes' veya 'no' çıkarır (regex tabanlı, structured output'a bağımlı değil)."""
    text_lower = text.lower().strip()
    if re.search(r'\byes\b', text_lower):
        return "yes"
    if re.search(r'\bno\b', text_lower):
        return "no"
    if re.search(r'"relevant"\s*:\s*"yes"', text_lower):
        return "yes"
    if re.search(r'"relevant"\s*:\s*"no"', text_lower):
        return "no"
    return default


def _parse_grader_reason(text: str) -> str:
    """Grader yanıtından 'reason' alanını çıkarır: 'needs_live_data' | 'irrelevant' | ''."""
    text_lower = text.lower()
    if "needs_live_data" in text_lower:
        return "needs_live_data"
    if "irrelevant" in text_lower:
        return "irrelevant"
    return ""


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
                "grader_reason": "irrelevant",
                "mode": "no_docs",
                "document_count": 0,
                "latency_ms_by_stage": {"total": elapsed_ms},
            },
            metadata={"grader_mode": "no_docs", "grader_confidence": None},
            tags=["frappe", "grader", "no"],
        )
        return {**state, "relevance": "no", "grader_reason": "irrelevant"}

    if state.get("source_filter") or state.get("session_uploads"):
        from src.agent.routing import is_web_query
        from src.rag.retriever import estimate_confidence

        original_q = state.get("original_question") or question
        if is_web_query(original_q):
            pass  # Web sorguları için grader LLM'i çalıştır — needs_live_data döndürebilir
        else:
            confidence = estimate_confidence(question, documents)
            if confidence < _GRADER_CONF_LOW:
                elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
                logger.info(
                    "Grader: relevance=no [mode=file_low_conf, conf=%.3f<%.3f, docs=%d, t=%.3fs]",
                    confidence, _GRADER_CONF_LOW, len(documents), time.perf_counter() - t0,
                )
                _observe_node(
                    "frappe.grader_decision",
                    state,
                    outputs={
                        "relevance": "no",
                        "grader_reason": "irrelevant",
                        "mode": "file_low_conf",
                        "confidence": confidence,
                        "low_threshold": _GRADER_CONF_LOW,
                        "document_count": len(documents),
                        "latency_ms_by_stage": {"total": elapsed_ms},
                    },
                    metadata={
                        "grader_mode": "file_low_conf",
                        "grader_confidence": confidence,
                        "grader_low_threshold": _GRADER_CONF_LOW,
                    },
                    tags=["frappe", "grader", "no"],
                )
                return {**state, "relevance": "no", "grader_reason": "irrelevant"}
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            logger.info(
                "Grader: relevance=yes [mode=file_fast, conf=%.3f, docs=%d, t=%.3fs]",
                confidence, len(documents), time.perf_counter() - t0,
            )
            _observe_node(
                "frappe.grader_decision",
                state,
                outputs={
                    "relevance": "yes",
                    "grader_reason": "",
                    "mode": "file_fast",
                    "confidence": confidence,
                    "document_count": len(documents),
                    "latency_ms_by_stage": {"total": elapsed_ms},
                },
                metadata={"grader_mode": "file_fast", "grader_confidence": confidence},
                tags=["frappe", "grader", "yes"],
            )
            return {**state, "relevance": "yes", "grader_reason": ""}

        top_docs = documents[:MAX_GRADER_DOCS]
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
            relevance = _parse_yes_no(response_text)
            reason = _parse_grader_reason(response_text) if relevance == "no" else ""
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

    if confidence >= _GRADER_CONF_HIGH:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            "Grader: relevance=yes [mode=high_conf, conf=%.3f>=%.3f, docs=%d, t=%.3fs]",
            confidence, _GRADER_CONF_HIGH, len(documents), time.perf_counter() - t0,
        )
        _observe_node(
            "frappe.grader_decision",
            state,
            outputs={
                "relevance": "yes",
                "grader_reason": "",
                "mode": "high_conf",
                "confidence": confidence,
                "high_threshold": _GRADER_CONF_HIGH,
                "document_count": len(documents),
                "latency_ms_by_stage": {"total": elapsed_ms},
            },
            metadata={
                "grader_mode": "high_conf",
                "grader_confidence": confidence,
                "grader_high_threshold": _GRADER_CONF_HIGH,
            },
            tags=["frappe", "grader", "yes"],
        )
        return {**state, "relevance": "yes", "grader_reason": ""}

    if confidence < _GRADER_CONF_LOW:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            "Grader: relevance=no [mode=low_conf, conf=%.3f<%.3f, docs=%d, t=%.3fs]",
            confidence, _GRADER_CONF_LOW, len(documents), time.perf_counter() - t0,
        )
        _observe_node(
            "frappe.grader_decision",
            state,
            outputs={
                "relevance": "no",
                "grader_reason": "irrelevant",
                "mode": "low_conf",
                "confidence": confidence,
                "low_threshold": _GRADER_CONF_LOW,
                "document_count": len(documents),
                "latency_ms_by_stage": {"total": elapsed_ms},
            },
            metadata={
                "grader_mode": "low_conf",
                "grader_confidence": confidence,
                "grader_low_threshold": _GRADER_CONF_LOW,
            },
            tags=["frappe", "grader", "no"],
        )
        return {**state, "relevance": "no", "grader_reason": "irrelevant"}

    top_docs = documents[:MAX_GRADER_DOCS]
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
        relevance = _parse_yes_no(response_text)
        reason = _parse_grader_reason(response_text) if relevance == "no" else ""
        logger.info(
            "Grader: relevance=%s reason=%s [mode=mid_conf, conf=%.3f, docs=%d/%d, doc_chars=%d, llm_t=%.3fs, t=%.3fs]",
            relevance, reason or "-", confidence, len(top_docs), len(documents), doc_chars,
            time.perf_counter() - t_llm, time.perf_counter() - t0,
        )
        llm_ms = round((time.perf_counter() - t_llm) * 1000, 2)
    except Exception as exc:
        # mid_conf hata: güvensiz belgeyle üretim yerine web fallback'e düş
        logger.warning("Grader: llm_error → no [err=%s, t=%.3fs]", exc, time.perf_counter() - t0)
        relevance, reason = "no", "irrelevant"
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


# ─────────────────────────────────────────────────────────────────────────────
# Node 5 — Vision (Gemma 4 multimodal görsel analiz)
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# Node 5b — Vision-RAG (Hibrit: görsel analizi → RAG pipeline'ına ilet)
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# Node 5c — Vision-Search (Görsel + Web Arama kombinasyonu)
# ─────────────────────────────────────────────────────────────────────────────


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
                records = WebResultFormatter.extract_source_records(web_result.text, limit=settings.web_search_max_results)
                if records:
                    for record in records:
                        web_docs.append(Document(
                            page_content=(
                                f"Title: {record.title}\n"
                                f"Published: {record.published or 'unknown'}\n"
                                f"URL: {record.url}\n"
                                f"Snippet: {record.content}"
                            )[:2500],
                            metadata={
                                "display_name": record.title,
                                "source": record.url,
                                "url": record.url,
                                "published": record.published,
                                "chunk_index": record.index,
                                "type": "web_search",
                            },
                        ))
                else:
                    web_docs.append(Document(
                        page_content=web_result.text[:8000],
                        metadata={"source": web_result.provider, "display_name": web_result.provider, "type": "web_search"},
                    ))
                step.output = f"Found via {web_result.provider} ({len(web_result.text)} chars)."
            else:
                logger.warning("Vision-Search: web araması sonuç döndürmedi")
                step.output = "Web search returned no results."
        else:
            step.output = "Web search service unavailable."

    return {**state, "vision_context": vision_context, "documents": web_docs}


# ─────────────────────────────────────────────────────────────────────────────
# Node 6 — Generator (RAG yanıtı üretir)
# ─────────────────────────────────────────────────────────────────────────────


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


def _final_answer_fields(
    state: AgentState,
    generation: str,
    *,
    t0: float,
    mode: str,
    extra_latency: dict | None = None,
) -> dict:
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
    system_content = (
        "Adın Frappe, bir RAG asistanısın. Sadece verilen bağlama dayanarak "
        "kullanıcının sorusunu aynı dilde, kısa ve doğrudan yanıtla. "
        "Bağlamda cevap yoksa sadece 'Bu bilgi yüklenen belgelerde yer almamaktadır.' yaz.\n\n"
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
    retry_summary: dict[str, object] = {
        "empty_response": False,
        "compact_retry_answer_chars": 0,
        "micro_retry_answer_chars": 0,
        "fallback_context_answer_used": False,
        "retry_path": "primary",
    }

    if documents or vision_context:
        context_parts = []

        # Görsel analiz sonucu ilk kaynak olarak eklenir
        if vision_context:
            context_parts.append(f"[Görsel Analizi]\n{vision_context}")

        # Bütçe: n_ctx toplam limittir (giriş+çıkış). max_tokens çıkışa ayrılır;
        # geri kalandan sistem şablonu, soru ve geçmiş için güvenlik payı düşülür.
        # Türkçe için muhafazakâr: 1 token ≈ 2.5 karakter.
        n_ctx = settings.llm_context_size
        output_tokens = settings.rag_max_tokens
        prior_chars = sum(len(getattr(m, "content", "") or "") for m in rag_history)
        prior_tokens_est = max(0, prior_chars // 4)
        overhead_tokens = 600 + prior_tokens_est  # sistem şablonu + soru + geçmiş + pay
        input_budget_tokens = max(256, n_ctx - output_tokens - overhead_tokens)
        budget_chars = int(input_budget_tokens * 2.5)
        used_chars = sum(len(p) for p in context_parts)

        from src.rag.retriever import chunk_id as _chunk_id
        for i, doc in enumerate(documents, 1):
            meta = getattr(doc, "metadata", {}) or {}
            src = meta.get("display_name") or meta.get("source_file", meta.get("source", ""))
            page = meta.get("page", "")
            is_web = meta.get("type") == "web_search"
            if is_web:
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
            remaining = budget_chars - used_chars
            if remaining <= len(header) + 50:
                break
            max_chars = min(2500 if is_web else 2000, remaining - len(header) - 10)
            content = doc.page_content[:max_chars]
            context_parts.append(f"{header}\n{content}")
            used_chars += len(header) + len(content) + 10
            used_chunk_ids.append(_chunk_id(doc))

        docs_included = len(context_parts) - (1 if vision_context else 0)
        context = "\n\n---\n\n".join(context_parts)
        # .replace() yerine .format() kullanılmaz — PDF/kod içindeki { } format() çökertiyor
        system_content = RAG_WITH_CONTEXT_SYSTEM_PROMPT.replace("{context}", context)

        total_prompt_chars = len(system_content) + prior_chars + len(answer_question)
        max_input_chars = int((n_ctx - output_tokens - 50) * 2.5)
        if total_prompt_chars > max_input_chars:
            safe_ctx_len = max(500, max_input_chars - prior_chars - len(answer_question) - 800)
            context = context[:safe_ctx_len]
            system_content = RAG_WITH_CONTEXT_SYSTEM_PROMPT.replace("{context}", context)
            logger.warning(
                "Generator: context truncated to %dch to fit n_ctx=%d",
                len(context), n_ctx,
            )

        logger.info(
            "Generator: ctx_budget=%dtok/%dch, used=%dch, docs=%d/%d, vision=%s, prior=%d, "
            "n_ctx=%d, output_max=%dtok, overhead=%dtok",
            input_budget_tokens, budget_chars, used_chars,
            docs_included, len(documents), bool(vision_context),
            len(rag_history), n_ctx, output_tokens, overhead_tokens,
        )

        # Trace'te used_in_context flag'ini işaretle ve final_used log'unu yaz
        used_set = set(used_chunk_ids)
        for entry in retrieval_trace:
            if entry.get("chunk_id") in used_set:
                entry["used_in_context"] = True
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

    if not generation.strip() and (documents or vision_context):
        generation = _fallback_context_answer(answer_question, documents, vision_context)
        retry_summary["fallback_context_answer_used"] = True
        retry_summary["retry_path"] = "compact_retry>micro_retry>fallback_context"

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
            "latency_ms_by_stage": latency_ms,
        },
        metadata={
            "answer_chars": len(generation),
            "docs_included": docs_included,
            "context_budget_tokens": input_budget_tokens,
            "context_budget_chars": budget_chars,
            "context_used_chars": used_chars,
            "context_overhead_tokens": overhead_tokens,
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


# ─────────────────────────────────────────────────────────────────────────────
# Node 7 — Web Search (çok provider destekli, OCP uyumlu)
# ─────────────────────────────────────────────────────────────────────────────


async def web_search_node(state: AgentState) -> AgentState:
    """Belge alaka düşük veya bulunamadığında web araması yapar.

    Provider zinciri: Brave MCP → Tavily → DuckDuckGo (ayarlara göre).
    Web araması için orijinal soru kullanılır — rewriter çıktısı web için uygun değildir.
    """
    # Orijinal soru vektör DB için yeniden yazılmış olabilir; web için doğal dili tercih et.
    question = state.get("original_question") or state["question"]
    existing_docs = state.get("documents", [])
    search_query = _build_contextual_web_query(question, list(state.get("messages", [])))

    async with cl.Step(name="Web Search", type="tool") as step:
        step.input = search_query

        service = _get_web_search_service()
        result = await service.search(search_query) if service else None

        if result is None:
            logger.warning("Web search: Tavily kullanılamıyor veya sonuç yok")
            step.output = "Web search failed."
            return {**state, "documents": existing_docs}

        records = WebResultFormatter.extract_source_records(result.text, limit=settings.web_search_max_results)
        web_docs = [
            Document(
                page_content=(
                    f"Title: {record.title}\n"
                    f"Published: {record.published or 'unknown'}\n"
                    f"URL: {record.url}\n"
                    f"Snippet: {record.content}"
                )[:2500],
                metadata={
                    "display_name": record.title,
                    "source": record.url,
                    "url": record.url,
                    "published": record.published,
                    "chunk_index": record.index,
                    "type": "web_search",
                },
            )
            for record in records
        ]
        if not web_docs:
            web_docs = [
                Document(
                    page_content=result.text[:8000],
                    metadata={"source": result.provider, "display_name": result.provider, "type": "web_search"},
                )
            ]
        step.output = f"Found content via {result.provider} ({len(result.text)} chars)."
        step.elements = [
            cl.Text(
                name=f"{result.provider.title()} Results",
                content=result.text[:2000] + "...",
                display="inline",
            )
        ]

    return {**state, "documents": existing_docs + web_docs}


# ─────────────────────────────────────────────────────────────────────────────
# Node 8 — Direct Response (ReAct agent — araç destekli hızlı yanıt)
# ─────────────────────────────────────────────────────────────────────────────


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


def _message_text(message: object) -> str:
    if isinstance(message, dict):
        return str(message.get("content") or "")
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return str(content or "")


def _clean_subject(subject: str) -> str:
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
        return normalized
    today = datetime.date.today().isoformat()
    q_lower = q.lower()
    if _EXPLICIT_WEB_ENTITY_RE.search(q):
        if re.search(r"\b(hisse|stock|borsa)\b", q_lower):
            return f"{normalized} {today} official quote Nasdaq Yahoo Finance MarketWatch"
        if re.search(r"\b(fiyat|price)\b", q_lower) and re.search(r"\b(iphone|telefon|phone|apple|samsung|xiaomi)\b", q_lower):
            return f"{normalized} Türkiye {today} resmi satıcı fiyat karşılaştırma"
        return normalized

    subject = _extract_web_subject_from_history(prior_messages)
    if not subject:
        return normalized

    if re.search(r"\b(hisse|stock|borsa)\b", q_lower):
        return f"{subject} stock price today {today} official quote market"
    if re.search(r"\b(iphone|apple|samsung|xiaomi|telefon|phone)\b", subject, re.IGNORECASE):
        return f"{subject} güncel fiyat Türkiye {today} resmi satıcı teknoloji mağazaları"
    return f"{subject} {normalized} {today}"


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
            if is_weather_query(question) and _is_pure_weather_query(question):
                answer = WebResultFormatter.format_weather(question, web_result.text)
                logger.info("Direct: weather_format [ans_len=%dch, t=%.3fs]", len(answer), time.perf_counter() - t_sum)
            else:
                answer = await _fast_web_summarize(
                    question,
                    web_result.text,
                    direct_history,
                    search_query=search_query,
                )
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
        llm = _get_chat_llm()
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
        llm = _get_chat_llm()
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
    from src.tools.search import search_web, tavily_search
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

    base_tools = [tavily_search, search_web, calculator, read_uploaded_file, mcp_call]
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


# ─────────────────────────────────────────────────────────────────────────────
# Yardımcılar
# ─────────────────────────────────────────────────────────────────────────────


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
