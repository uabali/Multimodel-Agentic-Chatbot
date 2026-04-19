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

import chainlit as cl
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agent.state import AgentState
from src.agent.routing import keyword_route, is_web_query, needs_mcp_tools, is_weather_query
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


def _get_router_llm():
    """Routing için minimal token-budget LLM (tek JSON token üretir)."""
    from src.rag.llm import create_vllm_llm
    return create_vllm_llm(temperature=0.0, max_tokens=settings.router_max_tokens)


def _get_rag_llm(temperature: float = 0.0):
    """RAG üretim / grader / rewriter için deterministik LLM."""
    from src.rag.llm import create_vllm_llm
    return create_vllm_llm(temperature=temperature, max_tokens=settings.rag_max_tokens)


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

    @classmethod
    def get(cls):
        if cls._instance is None:
            if settings.use_rerank:
                try:
                    from src.rag.reranker import create_reranker
                    cls._instance = create_reranker(
                        model_name=settings.reranker_model,
                        device=settings.reranker_device,
                    )
                except Exception as exc:
                    logger.warning("Reranker yüklenemedi (devre dışı): %s", exc)
                    cls._instance = None
            else:
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


def router_node(state: AgentState) -> AgentState:
    """Sorguyu 'rag', 'direct' veya 'vision' olarak sınıflandırır.

    Yol 0 (anlık): image_data doluysa LLM'e sormadan direkt 'vision' döner.
    Yol 1 (hızlı): Keyword eşleşmesi varsa LLM çağrısı yapılmaz.
    Yol 2 (yavaş): Belirsiz sorgular için düşük bütçeli LLM, text parsing ile rota belirlenir.
    """
    question = state["question"]
    prior_messages = list(state.get("messages", []))

    if state.get("image_data"):
        logger.info("Router (vision): görsel içerik tespit edildi → 'vision'")
        return {**state, "route": "vision"}

    if state.get("input_type") == "audio":
        logger.info("Router (audio): ses girdisi → 'direct' (hafif yol)")
        return {**state, "route": "direct"}

    # Dosya yüklendiyse: deterministik RAG — keyword/LLM routing atlanır.
    if state.get("source_filter"):
        logger.info(
            "Router (file): dosya bağlı sorgu → 'rag' (source_filter='%s')",
            state["source_filter"],
        )
        return {**state, "route": "rag"}

    session_uploads = state.get("session_uploads") or []
    fast_route = keyword_route(question, has_uploads=bool(session_uploads))
    if fast_route:
        logger.info(
            "Router (keyword): %s ← '%.60s' (session_uploads=%d)",
            fast_route, question, len(session_uploads),
        )
        return {**state, "route": fast_route}

    # Session'da yüklü belge varsa ve keyword karar veremediyse RAG'ı tercih et
    # (follow-up sorular için kritik — yoksa LLM router belge olduğunu bilmez).
    # İstisna: açıkça web gerektiren sorgular (spor, finans, haber) → 'direct'.
    if session_uploads:
        if is_web_query(question):
            logger.info(
                "Router (session-uploads + web override): web sorgusu → 'direct' (%.60s)",
                question,
            )
            return {**state, "route": "direct"}
        logger.info(
            "Router (session-uploads bias): %d dosya yüklü → 'rag' (LLM atlandı)",
            len(session_uploads),
        )
        return {**state, "route": "rag"}

    logger.info("Router (LLM fallback): belirsiz sorgu işleniyor... (%d önceki mesaj)", len(prior_messages))
    llm = _get_router_llm()
    try:
        # Chat history'yi dahil et - follow-up sorular için context
        messages_to_send = [SystemMessage(content=ROUTER_SYSTEM_PROMPT)]
        if prior_messages:
            # Son 2 mesajı context olarak ekle (minimal token kullanımı)
            messages_to_send.extend(prior_messages[-2:])
        messages_to_send.append(HumanMessage(content=question))
        
        response = llm.invoke(messages_to_send)
        route = _parse_route(response.content)
    except Exception as exc:
        logger.warning("Router başarısız, 'direct' varsayıldı: %s", exc)
        route = "direct"

    logger.info("Router (LLM): %s", route)
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


def rewriter_node(state: AgentState) -> AgentState:
    """Soruyu vektör veritabanı araması için optimize eder.

    Kısa/net sorgularda ve tek-turlu sorgularda LLM çağrısını atlar (~6s kazanç).
    """
    question = state["question"]
    prior_messages = list(state.get("messages", []))

    if _should_skip_rewrite(question, prior_messages):
        logger.info("Rewriter: atlandı (kısa/net sorgu) — '%.60s'", question)
        return state

    logger.info("Rewriter: '%.60s' (%d önceki mesaj)", question, len(prior_messages))
    llm = _get_rag_llm(temperature=0.0)
    messages_to_send = [SystemMessage(content=REWRITER_SYSTEM_PROMPT)]
    if prior_messages:
        # Maksimum 2 mesaj (1 tur) — daha fazlası halüsinasyon riskini artırır.
        messages_to_send.extend(prior_messages[-2:])
    messages_to_send.append(HumanMessage(content=question))
    response = llm.invoke(messages_to_send)
    rewritten = response.content.strip()
    logger.info("Rewriter sonucu: '%.80s'", rewritten)

    # Halüsinasyon koruması: rewriter cevap ürettiyse orijinal soruya dön.
    # Belirtiler: çok uzun metin, satır sonu içeriyor, veya cevap kalıpları var.
    _ANSWER_MARKERS = ("ihtiyacım", "yapabilmem için", "kritik bilgi", "hesaplayabilmem",
                       "belirtmek isterim", "lütfen", "sunabilmem", "verebilmem")
    is_hallucination = (
        len(rewritten) > 250
        or "\n" in rewritten
        or any(m in rewritten.lower() for m in _ANSWER_MARKERS)
    )
    if is_hallucination:
        logger.warning("Rewriter halüsinasyon → orijinal soru korunuyor: '%.80s'", rewritten)
        return state

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


def retriever_node(state: AgentState) -> AgentState:
    """Hybrid retrieval + dense gate + opsiyonel reranking uygular."""
    question = state["question"]
    source_filter = state.get("source_filter", "")
    session_uploads = state.get("session_uploads") or []
    logger.info("Retriever: '%.80s'", question)

    try:
        from src.rag.vectorstore import get_hybrid_store
        from src.rag.retriever import create_retriever, run_retriever

        store = get_hybrid_store()
        qdrant_filter = _build_source_filter(source_filter, session_uploads)
        if qdrant_filter:
            if source_filter:
                logger.info("Retriever: source_filter aktif → '%s'", source_filter)
            else:
                logger.info(
                    "Retriever: session_uploads filtresi aktif → %s",
                    session_uploads,
                )

        # source_filter veya session_uploads varsa kullanıcı dosyası kesin indekslendi
        # → dense gate atla (yabancı içerik dışlama gereksiz, filter zaten kısıtlı).
        if source_filter or session_uploads:
            dense_score = 1.0
            logger.info(
                "Dense gate: atlandı (source_filter='%s', session_uploads=%d)",
                source_filter, len(session_uploads),
            )
        else:
            dense_score = store.max_dense_similarity(question, qdrant_filter=qdrant_filter)
            if dense_score < settings.rag_min_dense_similarity:
                logger.info(
                    "Dense gate: %.3f < eşik %.3f → boş sonuç",
                    dense_score, settings.rag_min_dense_similarity,
                )
                return {**state, "documents": []}

        retriever = create_retriever(
            vectorstore=store.store,
            question=question,
            strategy=settings.retrieval_strategy,
            base_k=settings.base_k,
            use_rerank=settings.use_rerank,
            reranker=_RerankerRegistry.get(),
            rerank_top_n=settings.rerank_top_n,
            qdrant_filter=qdrant_filter,
        )
        documents = run_retriever(retriever, question)
        logger.info("Retriever: %d belge (dense_score=%.3f)", len(documents), dense_score)
    except Exception as exc:
        logger.warning("Retriever hatası: %s", exc)
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


def grader_node(state: AgentState) -> AgentState:
    """Belge alaka değerlendirmesi — önce sıfır-maliyetli confidence skoru dener.

    Yüksek güven (≥0.7): LLM atlanır → "yes"  (~3s kazanç, çoğu istek).
    Düşük güven  (<0.3): LLM atlanır → "no".
    Orta  (0.3–0.7):     LLM grader çalışır (borderline durum).
    """
    from src.rag.retriever import estimate_confidence

    question = state["question"]
    documents = state.get("documents", [])

    if not documents:
        logger.info("Grader: belge yok → 'no'")
        return {**state, "relevance": "no"}

    # Dosya bağlı sorgularda confidence skoru atlanır: retriever zaten filtreyle
    # eşleşmiş belgeleri döndürdüğünden skor her zaman yüksek gelir.
    # LLM grader, belgenin soruyu TAMAMEN yanıtlayıp yanıtlamadığını değerlendirir —
    # gerçek zamanlı veri gerektiren hibrit sorgularda web search'ün devreye girmesini sağlar.
    if state.get("source_filter") or state.get("session_uploads"):
        top_docs = documents[:MAX_GRADER_DOCS]
        doc_texts = "\n---\n".join(doc.page_content for doc in top_docs)
        logger.info(
            "Grader: dosya bağlı sorgu → LLM yeterlilik değerlendirmesi (source_filter='%s', session_uploads=%d)",
            state.get("source_filter", ""),
            len(state.get("session_uploads") or []),
        )
        llm = _get_rag_llm(temperature=0.0)
        try:
            response = llm.invoke([
                SystemMessage(content=GRADER_SYSTEM_PROMPT),
                HumanMessage(content=f"Question: {question}\n\nDocuments:\n{doc_texts}"),
            ])
            relevance = _parse_yes_no(response.content)
        except Exception as exc:
            logger.warning("Grader başarısız, 'yes' varsayıldı: %s", exc)
            relevance = "yes"
        logger.info("Grader (dosya): %s", relevance)
        return {**state, "relevance": relevance}

    confidence = estimate_confidence(question, documents)
    logger.info("Grader: confidence=%.3f (%d belge)", confidence, len(documents))

    if confidence >= _GRADER_CONF_HIGH:
        logger.info("Grader: yüksek güven → 'yes' (LLM atlandı)")
        return {**state, "relevance": "yes"}

    if confidence < _GRADER_CONF_LOW:
        logger.info("Grader: düşük güven → 'no' (LLM atlandı)")
        return {**state, "relevance": "no"}

    # Orta güven: LLM doğrulaması
    top_docs = documents[:MAX_GRADER_DOCS]
    doc_texts = "\n---\n".join(doc.page_content for doc in top_docs)
    logger.info("Grader: orta güven (%.3f) → LLM değerlendirmesi", confidence)

    llm = _get_rag_llm(temperature=0.0)
    try:
        response = llm.invoke([
            SystemMessage(content=GRADER_SYSTEM_PROMPT),
            HumanMessage(content=f"Question: {question}\n\nDocuments:\n{doc_texts}"),
        ])
        relevance = _parse_yes_no(response.content)
    except Exception as exc:
        logger.warning("Grader başarısız, 'yes' varsayıldı: %s", exc)
        relevance = "yes"

    logger.info("Grader: %s", relevance)
    return {**state, "relevance": relevance}


# ─────────────────────────────────────────────────────────────────────────────
# Node 5 — Vision (Gemma 4 multimodal görsel analiz)
# ─────────────────────────────────────────────────────────────────────────────


def vision_node(state: AgentState) -> AgentState:
    """Yüklenen görseli Gemma 4 multimodal API ile analiz eder.

    İçerik tipine göre (fatura, tablo, grafik, şema, genel) otomatik prompt seçimi yapılır.
    """
    question = state["question"]
    image_data = state.get("image_data") or []
    prior_messages = list(state.get("messages", []))

    image_names = [img.get("name", "") for img in image_data]
    system_prompt = select_vision_prompt(question, image_names)

    content_parts: list[dict] = []
    for img in image_data:
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:{img['mime']};base64,{img['base64']}"},
        })
    content_parts.append({
        "type": "text",
        "text": question.strip() or "Bu görseli analiz et.",
    })

    logger.info("Vision node: %d görsel, prompt=%s, %d önceki mesaj",
                len(image_data), system_prompt[:40].replace("\n", " "), len(prior_messages))
    llm = _get_rag_llm(temperature=0.2)

    messages_to_send = [SystemMessage(content=system_prompt)]
    messages_to_send.extend(prior_messages[-6:])
    messages_to_send.append(HumanMessage(content=content_parts))

    response = llm.invoke(messages_to_send)
    generation = response.content or ""

    new_messages = [
        *prior_messages,
        HumanMessage(content=question),
        AIMessage(content=generation),
    ]
    return {**state, "generation": generation, "messages": new_messages}


# ─────────────────────────────────────────────────────────────────────────────
# Node 5b — Vision-RAG (Hibrit: görsel analizi → RAG pipeline'ına ilet)
# ─────────────────────────────────────────────────────────────────────────────


def vision_rag_node(state: AgentState) -> AgentState:
    """Hibrit mode: görsel analizi yapar, sonucu state'e yazar; RAG pipeline devam eder.

    Akış: vision_rag → rewriter → retriever → grader → generator
    Generator, vision_context'i [Görsel Analizi] kaynağı olarak bağlama dahil eder.
    """
    question = state["question"]
    image_data = state.get("image_data") or []
    prior_messages = list(state.get("messages", []))

    image_names = [img.get("name", "") for img in image_data]
    system_prompt = select_vision_prompt(question, image_names)

    content_parts: list[dict] = []
    for img in image_data:
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:{img['mime']};base64,{img['base64']}"},
        })
    content_parts.append({
        "type": "text",
        "text": (
            "Bu görseli detaylıca analiz et. "
            "Tüm metinleri, sayıları, tablo verilerini ve yapısal bilgileri eksiksiz çıkar. "
            "Sonuç RAG sistemi için kaynak olarak kullanılacak."
        ),
    })

    logger.info(
        "Vision-RAG: %d görsel analiz ediliyor (prompt=%s)",
        len(image_data), system_prompt[:40].replace("\n", " "),
    )
    llm = _get_rag_llm(temperature=0.1)

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=content_parts),
        ])
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

    content_parts: list[dict] = []
    for img in image_data:
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:{img['mime']};base64,{img['base64']}"},
        })
    content_parts.append({
        "type": "text",
        "text": (
            "Bu görseli analiz et. Tarih, tutar, döviz birimi, miktar gibi "
            "tüm yapısal verileri olduğu gibi çıkar. "
            "Sonuç gerçek zamanlı web verileriyle birleştirilecek."
        ),
    })

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


def generator_node(state: AgentState) -> AgentState:
    """Belgeler ve/veya görsel bağlam varsa RAG ile, yoksa bağlamsız modda yanıt üretir.

    vision_context mevcutsa [Görsel Analizi] başlığıyla bağlamın başına eklenir.
    """
    question = state["question"]
    documents = state.get("documents", [])
    prior_messages = list(state.get("messages", []))
    vision_context = state.get("vision_context", "")

    logger.info(
        "Generator: %d belge + görsel=%s, %d önceki mesaj",
        len(documents), "var" if vision_context else "yok", len(prior_messages),
    )

    if documents or vision_context:
        context_parts = []

        # Görsel analiz sonucu ilk kaynak olarak eklenir
        if vision_context:
            context_parts.append(f"[Görsel Analizi]\n{vision_context}")

        for i, doc in enumerate(documents, 1):
            meta = getattr(doc, "metadata", {}) or {}
            src = meta.get("source_file", meta.get("source", ""))
            page = meta.get("page", "")
            header = f"[Kaynak {i}: {src}" + (f", Sayfa {page}" if page and str(page) not in {"", "?"} else "") + "]"
            # Web sonuçları daha agresif kırpılır — context window taşmasını önler.
            # Belge chunk'ları daha az kırpılır (kritik formül/veri kaybı riski var).
            is_web = meta.get("type") == "web_search"
            max_chars = 3000 if is_web else 6000
            content = doc.page_content[:max_chars]
            context_parts.append(f"{header}\n{content}")

        context = "\n\n---\n\n".join(context_parts)
        # .replace() yerine .format() kullanılmaz — PDF/kod içindeki { } format() çökertiyor
        system_content = RAG_WITH_CONTEXT_SYSTEM_PROMPT.replace("{context}", context)
    else:
        system_content = RAG_NO_CONTEXT_SYSTEM_PROMPT

    llm = _get_rag_llm(temperature=0.0)

    messages_to_send = [SystemMessage(content=system_content)]
    messages_to_send.extend(prior_messages)
    messages_to_send.append(HumanMessage(content=question))

    response = llm.invoke(messages_to_send)
    generation = response.content

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

    async with cl.Step(name="Web Search", type="tool") as step:
        step.input = question

        service = _get_web_search_service()
        result = await service.search(question) if service else None

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
        "- If multiple sources give different values, mention the range or the most recent.\n"
        "- If a source has a published date, prefer the most recent one.\n"
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
    response = await asyncio.to_thread(llm.invoke, messages_to_send)
    text = (response.content or "").strip()
    return WebResultFormatter.append_sources(text, result_text, question)


async def direct_response_node(state: AgentState) -> AgentState:
    """Doğrudan yanıt node'u.

    Web sorguları için hızlı yol: ReAct döngüsüne girmeden önce web sonucunu
    özetler ve döner. Geri kalan sorgular için araçlı ReAct agent çalıştırılır.
    """
    question = state["question"]
    prior_messages = list(state.get("messages", []))
    logger.info("Direct Response: '%.80s' (%d önceki mesaj)", question, len(prior_messages))

    # Hızlı yol — gerçek zamanlı web sorguları
    if is_web_query(question):
        service = _get_web_search_service()
        web_result = await service.search(question) if service else None
        if web_result:
            if is_weather_query(question) and _is_pure_weather_query(question):
                answer = WebResultFormatter.format_weather(question, web_result.text)
            else:
                answer = await _fast_web_summarize(question, web_result.text, prior_messages)

            new_messages = [
                *prior_messages,
                HumanMessage(content=question),
                AIMessage(content=answer),
            ]
            return {**state, "generation": answer, "messages": new_messages}

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
    all_tools = _dedupe_tools(mcp_tools + base_tools)

    system_prompt = build_generator_prompt(all_tools)
    llm = _get_agent_llm()
    backend = (settings.llm_backend or "").lower().strip()

    if backend in {"llama.cpp", "llamacpp", "llama"} and needs_mcp_tools(question):
        logger.info("Tool calling disabled on llama.cpp backend; returning non-tool response.")
        messages_to_send = [SystemMessage(content=system_prompt)]
        messages_to_send.extend(prior_messages)
        messages_to_send.append(HumanMessage(content=question))
        response = llm.invoke(messages_to_send)
        generation = getattr(response, "content", "") or ""
        new_messages = [
            *prior_messages,
            HumanMessage(content=question),
            AIMessage(content=generation),
        ]
        return {**state, "generation": generation, "messages": new_messages}

    agent = create_react_agent(llm, all_tools, prompt=system_prompt)
    result = await agent.ainvoke({"messages": prior_messages + [HumanMessage(content=question)]})

    generation = result["messages"][-1].content
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
