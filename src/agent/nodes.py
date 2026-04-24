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
import logging
import re
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



# ─────────────────────────────────────────────────────────────────────────────
# LLM fabrika erişimleri — DIP: node'lar doğrudan ChatOpenAI yaratmaz
# ─────────────────────────────────────────────────────────────────────────────


_router_llm_cache = None
_rag_llm_cache: dict[tuple, object] = {}


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
    """
    if temperature == 0.0 and max_tokens is None:
        from src.rag.llm import get_rag_llm
        return get_rag_llm()
    key = (temperature, max_tokens)
    if key not in _rag_llm_cache:
        from src.rag.llm import create_vllm_llm
        _rag_llm_cache[key] = create_vllm_llm(
            temperature=temperature,
            max_tokens=max_tokens or settings.rag_max_tokens,
        )
    return _rag_llm_cache[key]


def reset_nodes_llm_cache() -> None:
    """LLM ayarları runtime'da değiştiğinde (api/router.py) çağrılır."""
    global _router_llm_cache
    _router_llm_cache = None
    _rag_llm_cache.clear()


def _get_agent_llm():
    """ReAct agent için tool-call uyumlu LLM (düşük sıcaklık, yüksek bütçe)."""
    from src.rag.llm import get_agent_llm
    return get_agent_llm()


# ─────────────────────────────────────────────────────────────────────────────
# Reranker kayıt defteri — modül-level global state'i kapsüller (SRP)
# ─────────────────────────────────────────────────────────────────────────────


class _RerankerRegistry:
    """Reranker instance'ını lazy olarak yükler ve önbellekte tutar."""

    _instance = None
    _loading = False

    @classmethod
    def get(cls):
        if cls._instance is not None:
            return cls._instance
        if cls._loading:
            # Another thread is loading; return None so caller skips rerank this request
            return None
        if not settings.use_rerank:
            return None
        cls._loading = True
        try:
            from src.rag.reranker import create_reranker
            cls._instance = create_reranker(
                model_name=settings.reranker_model,
                device=settings.reranker_device,
            )
        except Exception as exc:
            logger.warning("Reranker yüklenemedi (devre dışı): %s", exc)
            cls._instance = None
        finally:
            cls._loading = False
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

    if state.get("image_data"):
        imgs = state["image_data"]
        logger.info(
            "Router → vision [images=%d, mimes=%s, q_len=%d, t=0.00s]",
            len(imgs),
            ",".join(img.get("mime", "?") for img in imgs),
            q_len,
        )
        return {**state, "route": "vision"}

    # Dosya yüklendiyse: deterministik RAG — keyword/LLM routing atlanır.
    if state.get("source_filter"):
        logger.info(
            "Router → rag [reason=source_filter, file='%s', q_len=%d, t=%.3fs]",
            state["source_filter"], q_len, time.perf_counter() - t0,
        )
        return {**state, "route": "rag"}

    session_uploads = state.get("session_uploads") or []
    fast_route = keyword_route(question, has_uploads=bool(session_uploads))
    if fast_route:
        logger.info(
            "Router → %s [reason=keyword, uploads=%d, q_len=%d, t=%.3fs]",
            fast_route, len(session_uploads), q_len, time.perf_counter() - t0,
        )
        return {**state, "route": fast_route}

    if session_uploads:
        if is_web_query(question):
            logger.info(
                "Router → direct [reason=web_override+uploads, uploads=%d, q_len=%d, t=%.3fs]",
                len(session_uploads), q_len, time.perf_counter() - t0,
            )
            return {**state, "route": "direct"}
        logger.info(
            "Router → rag [reason=uploads_bias, uploads=%d, q_len=%d, t=%.3fs]",
            len(session_uploads), q_len, time.perf_counter() - t0,
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
            messages_to_send.extend(prior_messages[-2:])
        messages_to_send.append(HumanMessage(content=question))
        response = await llm.ainvoke(messages_to_send)
        route = _parse_route(response.content)
    except Exception as exc:
        logger.warning("Router LLM başarısız → direct [err=%s]", exc)
        route = "direct"

    logger.info(
        "Router → %s [reason=llm, llm_t=%.3fs, total_t=%.3fs]",
        route, time.perf_counter() - t_llm, time.perf_counter() - t0,
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


def _should_skip_rewrite(question: str, prior_messages: list) -> bool:
    """True döndürürse rewriter LLM çağrısı (~6s) atlanır.

    Atla  → kısa (≤8 kelime) + soru kelimesi var + follow-up değil.
    Devam → çok-turlu follow-up'lar (referans çözümlemesi gerektirir).
    """
    words = question.split()
    q_lower = question.lower()

    if prior_messages and any(m in q_lower for m in _FOLLOW_UP_MARKERS):
        return False

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

    try:
        from src.rag.vectorstore import get_hybrid_store
        from src.rag.retriever import create_retriever, run_retriever

        store = get_hybrid_store()
        qdrant_filter = _build_source_filter(source_filter, session_uploads)

        if source_filter or session_uploads:
            dense_score = 1.0
            filter_desc = f"source_filter='{source_filter}'" if source_filter else f"uploads={session_uploads}"
            logger.info("Retriever: dense_gate=skip [%s]", filter_desc)
        else:
            t_gate = time.perf_counter()
            dense_score = await asyncio.to_thread(
                store.max_dense_similarity, question, qdrant_filter=qdrant_filter
            )
            logger.info(
                "Retriever: dense_gate=%.3f [threshold=%.3f, t=%.3fs]",
                dense_score, settings.rag_min_dense_similarity,
                time.perf_counter() - t_gate,
            )
            if dense_score < settings.rag_min_dense_similarity:
                logger.info(
                    "Retriever: gate_reject [score=%.3f < %.3f, t=%.3fs]",
                    dense_score, settings.rag_min_dense_similarity,
                    time.perf_counter() - t0,
                )
                return {**state, "documents": []}

        strategy = state.get("retrieval_strategy") or settings.retrieval_strategy
        use_rerank_val = state.get("use_rerank")
        if use_rerank_val is None:
            use_rerank_val = settings.use_rerank

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
        t_fetch_elapsed = time.perf_counter() - t_fetch

        # Kaynak dağılımını özetle
        sources: dict[str, int] = {}
        for doc in documents:
            meta = getattr(doc, "metadata", {}) or {}
            src = meta.get("source_file", meta.get("source", "?"))
            sources[src] = sources.get(src, 0) + 1
        src_summary = ", ".join(f"{s}×{n}" for s, n in sources.items())

        logger.info(
            "Retriever: docs=%d [strategy=%s, rerank=%s, dense=%.3f, fetch_t=%.3fs, total_t=%.3fs] {%s}",
            len(documents), strategy, use_rerank_val, dense_score,
            t_fetch_elapsed, time.perf_counter() - t0, src_summary,
        )
    except Exception as exc:
        logger.warning("Retriever: error [%s, t=%.3fs]", exc, time.perf_counter() - t0)
        documents = []

    return {**state, "documents": documents}


# ─────────────────────────────────────────────────────────────────────────────
# Node 4 — Grader (CRAG-style belge alaka değerlendirmesi)
# ─────────────────────────────────────────────────────────────────────────────


MAX_GRADER_DOCS = 5

# Confidence eşikleri: yüksek/düşük durumda LLM atlanır (~3s kazanç).
_GRADER_CONF_HIGH = 0.75  # Bu eşiğin üstünde → doğrudan "yes" (LLM atlanır)
_GRADER_CONF_LOW  = 0.15  # Bu eşiğin altında  → doğrudan "no"  (LLM atlanır)


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
        logger.info("Grader: no_docs → relevance=no [t=%.3fs]", time.perf_counter() - t0)
        return {**state, "relevance": "no", "grader_reason": "irrelevant"}

    if state.get("source_filter") or state.get("session_uploads"):
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
            relevance = _parse_yes_no(response.content)
            reason = _parse_grader_reason(response.content) if relevance == "no" else ""
            logger.info(
                "Grader: relevance=%s reason=%s [mode=file_llm, docs=%d/%d, doc_chars=%d, llm_t=%.3fs, t=%.3fs]",
                relevance, reason or "-", len(top_docs), len(documents), doc_chars,
                time.perf_counter() - t_llm, time.perf_counter() - t0,
            )
        except Exception as exc:
            logger.warning("Grader: llm_error → yes [err=%s, t=%.3fs]", exc, time.perf_counter() - t0)
            relevance, reason = "yes", ""
        return {**state, "relevance": relevance, "grader_reason": reason}

    confidence = estimate_confidence(question, documents)

    if confidence >= _GRADER_CONF_HIGH:
        logger.info(
            "Grader: relevance=yes [mode=high_conf, conf=%.3f>=%.3f, docs=%d, t=%.3fs]",
            confidence, _GRADER_CONF_HIGH, len(documents), time.perf_counter() - t0,
        )
        return {**state, "relevance": "yes", "grader_reason": ""}

    if confidence < _GRADER_CONF_LOW:
        logger.info(
            "Grader: relevance=no [mode=low_conf, conf=%.3f<%.3f, docs=%d, t=%.3fs]",
            confidence, _GRADER_CONF_LOW, len(documents), time.perf_counter() - t0,
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
        relevance = _parse_yes_no(response.content)
        reason = _parse_grader_reason(response.content) if relevance == "no" else ""
        logger.info(
            "Grader: relevance=%s reason=%s [mode=mid_conf, conf=%.3f, docs=%d/%d, doc_chars=%d, llm_t=%.3fs, t=%.3fs]",
            relevance, reason or "-", confidence, len(top_docs), len(documents), doc_chars,
            time.perf_counter() - t_llm, time.perf_counter() - t0,
        )
    except Exception as exc:
        # mid_conf hata: güvensiz belgeyle üretim yerine web fallback'e düş
        logger.warning("Grader: llm_error → no [err=%s, t=%.3fs]", exc, time.perf_counter() - t0)
        relevance, reason = "no", "irrelevant"

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
    return {**state, "generation": generation, "messages": new_messages}


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
        response = await asyncio.to_thread(
            llm.invoke,
            [SystemMessage(content=system_prompt), HumanMessage(content=content_parts)],
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
        vision_response = await asyncio.to_thread(
            llm.invoke,
            [SystemMessage(content=system_prompt), HumanMessage(content=content_parts)],
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
                web_docs.append(Document(
                    page_content=web_result.text[:8000],
                    metadata={"source": web_result.provider, "type": "web_search"},
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


async def generator_node(state: AgentState) -> AgentState:
    """Belgeler ve/veya görsel bağlam varsa RAG ile, yoksa bağlamsız modda yanıt üretir.

    vision_context mevcutsa [Görsel Analizi] başlığıyla bağlamın başına eklenir.
    """
    t0 = time.perf_counter()
    question = state["question"]
    documents = state.get("documents", [])
    prior_messages = list(state.get("messages", []))
    vision_context = state.get("vision_context", "")

    if documents or vision_context:
        context_parts = []

        # Görsel analiz sonucu ilk kaynak olarak eklenir
        if vision_context:
            context_parts.append(f"[Görsel Analizi]\n{vision_context}")

        # Bütçe: n_ctx toplam limittir (giriş+çıkış). max_tokens çıkışa ayrılır;
        # geri kalandan sistem şablonu, soru ve geçmiş için güvenlik payı düşülür.
        # Türkçe için muhafazakâr: 1 token ≈ 2.5 karakter.
        n_ctx = settings.llm_context_size
        output_tokens = state.get("max_tokens") or settings.rag_max_tokens
        # prior_messages[-2:] yaklaşık token maliyetini hesaba kat
        prior_chars = sum(len(getattr(m, "content", "") or "") for m in prior_messages[-2:])
        prior_tokens_est = max(0, prior_chars // 4)
        overhead_tokens = 800 + prior_tokens_est  # sistem şablonu + soru + geçmiş + pay
        input_budget_tokens = max(256, n_ctx - output_tokens - overhead_tokens)
        budget_chars = int(input_budget_tokens * 2.5)
        used_chars = sum(len(p) for p in context_parts)

        for i, doc in enumerate(documents, 1):
            meta = getattr(doc, "metadata", {}) or {}
            src = meta.get("source_file", meta.get("source", ""))
            page = meta.get("page", "")
            header = f"[Kaynak {i}: {src}" + (f", Sayfa {page}" if page and str(page) not in {"", "?"} else "") + "]"
            is_web = meta.get("type") == "web_search"
            remaining = budget_chars - used_chars
            if remaining <= len(header) + 50:
                break
            max_chars = min(2000 if is_web else 1500, remaining - len(header) - 10)
            content = doc.page_content[:max_chars]
            context_parts.append(f"{header}\n{content}")
            used_chars += len(header) + len(content) + 10

        docs_included = len(context_parts) - (1 if vision_context else 0)
        context = "\n\n---\n\n".join(context_parts)
        # .replace() yerine .format() kullanılmaz — PDF/kod içindeki { } format() çökertiyor
        system_content = RAG_WITH_CONTEXT_SYSTEM_PROMPT.replace("{context}", context)
        logger.info(
            "Generator: ctx_budget=%dtok/%dch, used=%dch, docs=%d/%d, vision=%s, prior=%d, "
            "n_ctx=%d, output_max=%dtok, overhead=%dtok",
            input_budget_tokens, budget_chars, used_chars,
            docs_included, len(documents), bool(vision_context),
            len(prior_messages[-2:]), n_ctx, output_tokens, overhead_tokens,
        )
    else:
        system_content = RAG_NO_CONTEXT_SYSTEM_PROMPT
        logger.info(
            "Generator: no_context [prior=%d, n_ctx=%d, output_max=%dtok]",
            len(prior_messages[-2:]), settings.llm_context_size,
            state.get("max_tokens") or settings.rag_max_tokens,
        )

    session_temp = state.get("temperature") or 0.0
    session_max_tok = state.get("max_tokens") or None
    llm = _get_rag_llm(temperature=session_temp, max_tokens=session_max_tok)

    messages_to_send = [SystemMessage(content=system_content)]
    messages_to_send.extend(prior_messages[-2:])
    messages_to_send.append(HumanMessage(content=question))

    t_llm = time.perf_counter()
    response = await llm.ainvoke(messages_to_send)
    generation = response.content or ""

    logger.info(
        "Generator: done [ans_len=%dch, temp=%.2f, llm_t=%.3fs, total_t=%.3fs]",
        len(generation), session_temp,
        time.perf_counter() - t_llm, time.perf_counter() - t0,
    )

    new_messages = [
        *prior_messages,
        HumanMessage(content=state["question"]),
        AIMessage(content=generation),
    ]
    return {**state, "generation": generation, "messages": new_messages}


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
    search_query = normalize_web_query(question)

    async with cl.Step(name="Web Search", type="tool") as step:
        step.input = search_query

        service = _get_web_search_service()
        result = await service.search(search_query) if service else None

        if result is None:
            logger.warning("Web search: Tavily kullanılamıyor veya sonuç yok")
            step.output = "Web search failed."
            return {**state, "documents": existing_docs}

        web_doc = Document(
            page_content=result.text[:8000],
            metadata={"source": result.provider, "type": "web_search"},
        )
        step.output = f"Found content via {result.provider} ({len(result.text)} chars)."
        step.elements = [
            cl.Text(
                name=f"{result.provider.title()} Results",
                content=result.text[:2000] + "...",
                display="inline",
            )
        ]

    return {**state, "documents": existing_docs + [web_doc]}


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


def _is_pure_weather_query(question: str) -> bool:
    """Sorgu yalnızca hava durumu soruyorsa True döner; compound sorgular False."""
    return not bool(_COMPOUND_QUERY_MARKERS.search(question))


async def _fast_web_summarize(question: str, result_text: str, prior_messages: list | None = None) -> str:
    """Web sonuçlarını LLM çağrısıyla özetler; entity isimleri ve rakamları çıkarır."""
    system = (
        "You answer ONLY from the provided web search results.\n"
        "Rules:\n"
        "- Respond in the same language as the user's question.\n"
        "- Turkish question → fully Turkish answer.\n"
        "- Never say you cannot access live data or the internet.\n"
        "- Never repeat the user's question at the start.\n"
        "- Extract SPECIFIC entities: names, prices, percentages, dates, company names.\n"
        "- RECENCY RULE: Always report only the MOST RECENT value. Do NOT list values from multiple historical dates.\n"
        "- If sources conflict, pick the one with the latest date and note it briefly.\n"
        "- Cite the source name inline when possible (e.g. 'Bloomberg'a göre...').\n"
        "- Format: use bullet points for multiple facts; prose for single answers.\n"
        "- Do NOT say 'I don't have real-time data' — use what's in the web results.\n"
        "TABLE RULE: If the user asked for a table (tablo, table) OR a multi-day forecast "
        "(5 günlük, haftalık, X-Y arası), extract each day's data from the web results and "
        "present it as a Markdown table with columns: | Gün | Tarih | Hava | Max °C | Min °C |. "
        "Fill only the columns that appear in the results; omit columns with no data. "
        "Do NOT redirect the user to external sites — build the table yourself from the data.\n"
    )
    llm = _get_rag_llm(temperature=0.0)
    messages_to_send = [SystemMessage(content=system)]
    if prior_messages:
        messages_to_send.extend(prior_messages[-4:])
    messages_to_send.append(
        HumanMessage(content=f"Question: {question}\n\nWeb results:\n{result_text[:5000]}")
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

    # Hızlı yol — gerçek zamanlı web sorguları
    if is_web_query(question):
        service = _get_web_search_service()
        search_query = normalize_web_query(question)
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
                answer = await _fast_web_summarize(question, web_result.text, prior_messages)
                logger.info(
                    "Direct: web_summarize [ans_len=%dch, llm_t=%.3fs, total_t=%.3fs]",
                    len(answer), time.perf_counter() - t_sum, time.perf_counter() - t0,
                )

            new_messages = [
                *prior_messages,
                HumanMessage(content=question),
                AIMessage(content=answer),
            ]
            return {**state, "generation": answer, "messages": new_messages}
        else:
            logger.warning("Direct: web_no_result [search_t=%.3fs]", time.perf_counter() - t_search)

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
        messages_to_send.extend(prior_messages)
        messages_to_send.append(HumanMessage(content=question))
        response = await llm.ainvoke(messages_to_send)
        generation = getattr(response, "content", "") or ""
        new_messages = [
            *prior_messages,
            HumanMessage(content=question),
            AIMessage(content=generation),
        ]
        return {**state, "generation": generation, "messages": new_messages}

    logger.info(
        "Direct: react_agent [tools=%d, prior=%d, backend=%s]",
        len(all_tools), len(prior_messages), backend,
    )
    t_react = time.perf_counter()
    agent = create_react_agent(llm, all_tools, prompt=system_prompt)
    result = await agent.ainvoke({"messages": prior_messages + [HumanMessage(content=question)]})

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
    return {**state, "generation": generation, "messages": new_messages}


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
