"""
LangGraph agent graph — dual-path pipeline (RAG + ReAct direct).

Graf topolojisi:
                START
                  │
               Router          ← keyword veya LLM ile rota kararı
              /       \\
        rag /           \\ direct
       Rewriter    Direct Response
          │               │
       Retriever          │
          │               │
        Grader            │
       /      \\           │
 yes  /    no  \\          │
Generator  Web Search     │
    │           │         │
    │       Generator     │
    └─────────────────────┘
                │
               END

SOLID uyumu:
 - SRP: Bu modül yalnızca graph yapısını (düğümler, kenarlar, koşullu geçişler) tanımlar.
 - OCP: Yeni düğüm eklemek için sadece build_graph() içine satır eklemek yeterli.
 - DIP: Node'lar bu modüle bağımlı değil; graph onları çağırır.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging

from langgraph.graph import END, StateGraph

from src.agent.state import AgentState
from src.agent.nodes import (
    router_node,
    rewriter_node,
    retriever_node,
    grader_node,
    vision_node,
    vision_rag_node,
    vision_search_node,
    generator_node,
    web_search_node,
    direct_response_node,
)
from src.memory.thread_memory import is_memory_command
from src.observability.langsmith import build_graph_config, record_semantic_cache_hit

logger = logging.getLogger(__name__)


def build_semantic_cache_context(
    *,
    source_filter: str = "",
    session_uploads: list[str] | None = None,
    retrieval_strategy: str | None = None,
    memory_hash: str = "",
) -> str:
    """Stable cache context; memory changes must invalidate cached answers."""
    payload = json.dumps({
        "sf": source_filter or "",
        "su": sorted(session_uploads or []),
        "rs": retrieval_strategy or "",
        "mh": memory_hash or "",
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────
# Koşullu kenar fonksiyonları — SRP: yalnızca geçiş kararı
# ─────────────────────────────────────────────────────────────────────────────


def _route_decision(state: AgentState) -> str:
    """Router node çıkışından rota alır.

    Öncelik sırası (vision rotası için):
      1. vision + source_filter   → vision_rag    (görsel + BU TURDA yüklenen belge)
      2. vision + is_web_query    → vision_search (görsel + gerçek zamanlı web verisi)
      3. vision                   → vision        (saf görsel analiz)

    NOT: session_uploads (önceki turlardan kalan belgeler) vision_rag'ı TETİKLEMEZ.
    Önceki tur belgeleri alakasız içerik getirebilir — sadece aynı turda yüklenen
    belge (source_filter) görsel analiziyle birleştirilmeli.
    """
    from src.agent.routing import is_web_query

    route = state.get("route", "direct")
    if route == "vision":
        if state.get("source_filter"):
            return "vision_rag"
        if is_web_query(state.get("question", "")):
            return "vision_search"
    return route


def _grader_decision(state: AgentState) -> str:
    """Grader node çıkışından geçiş kararı verir.

    "insufficient" → web_search → generator (doğrudan edge, grader'a geri dönmez).
    Döngü koruması graph yapısı tarafından sağlanır, retry_count burada gereksiz.

    source_filter aktifken iki senaryo:
    - reason="irrelevant": belge soruyla ilgisiz → generator "belgede yok" cevabı verir.
    - reason="needs_live_data": belge formülü içeriyor ama canlı veri eksik → web arama.
    Reason belirsizse/yoksa: güvenli taraf olan "sufficient" seçilir.
    """
    if state.get("relevance") == "yes":
        return "sufficient"
    reason = state.get("grader_reason", "")
    if state.get("source_filter"):
        if reason == "needs_live_data":
            return "insufficient"
        return "sufficient"
    return "insufficient"


# ─────────────────────────────────────────────────────────────────────────────
# Graph yapısı
# ─────────────────────────────────────────────────────────────────────────────


def build_graph():
    """LangGraph workflow'unu derler ve döner."""
    workflow = StateGraph(AgentState)

    # Düğümler
    workflow.add_node("router", router_node)
    workflow.add_node("rewriter", rewriter_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("grader", grader_node)
    workflow.add_node("vision", vision_node)
    workflow.add_node("vision_rag", vision_rag_node)
    workflow.add_node("vision_search", vision_search_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("direct_response", direct_response_node)

    # Başlangıç noktası
    workflow.set_entry_point("router")

    # Koşullu: Router → RAG / Direct / Vision / Vision-RAG
    workflow.add_conditional_edges(
        "router",
        _route_decision,
        {
            "rag": "rewriter",
            "direct": "direct_response",
            "vision": "vision",
            "vision_rag": "vision_rag",
            "vision_search": "vision_search",
        },
    )

    workflow.add_edge("vision", END)

    # Vision-RAG: görsel analizi tamamlayıp RAG pipeline'ına devam eder
    workflow.add_edge("vision_rag", "rewriter")

    # Vision-Search: görsel + web araması → direkt generator (rewrite/retrieve gereksiz)
    workflow.add_edge("vision_search", "generator")

    # RAG yolu
    workflow.add_edge("rewriter", "retriever")
    workflow.add_edge("retriever", "grader")

    # Koşullu: Grader → Generate veya Web Search (CRAG döngüsü)
    workflow.add_conditional_edges(
        "grader",
        _grader_decision,
        {"sufficient": "generator", "insufficient": "web_search"},
    )

    # Web search sonrası direkt generate (grader'a geri dönülmez — sonsuz döngü önlemi)
    workflow.add_edge("web_search", "generator")
    workflow.add_edge("generator", END)
    workflow.add_edge("direct_response", END)

    app = workflow.compile()
    logger.info("LangGraph agent graph derlendi.")
    return app


# ─────────────────────────────────────────────────────────────────────────────
# Singleton erişim — lazy init, thread-safe değil (tek süreç varsayımı)
# ─────────────────────────────────────────────────────────────────────────────

_graph = None


def get_graph():
    """Graph singleton'ını döner (ilk çağrıda derlenir).

    Thread safety: build_graph() senkrondur (await noktası yok). asyncio
    cooperative scheduling'de iki coroutine aynı anda buraya giremez.
    Güvenli, ek lock gereksiz.
    """
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ─────────────────────────────────────────────────────────────────────────────
# Initial state fabrikası
# ─────────────────────────────────────────────────────────────────────────────


def _init_state(
    question: str,
    chat_history: list | None = None,
    source_filter: str = "",
    image_data: list[dict] | None = None,
    input_type: str = "text",
    session_uploads: list[str] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    retrieval_strategy: str | None = None,
    use_rerank: bool | None = None,
) -> AgentState:
    from src.config import settings as _s
    return {
        "messages": chat_history or [],
        "documents": [],
        "original_question": question,
        "question": question,
        "generation": "",
        "route": "",
        "relevance": "",
        "grader_reason": "",
        "source_filter": source_filter,
        "session_uploads": list(session_uploads or []),
        "image_data": image_data or [],
        "input_type": input_type,
        "vision_context": "",
        "temperature": temperature if temperature is not None else _s.chat_temperature,
        "max_tokens": max_tokens if max_tokens is not None else _s.chat_max_tokens,
        "retrieval_strategy": retrieval_strategy or _s.retrieval_strategy,
        "use_rerank": use_rerank if use_rerank is not None else _s.use_rerank,
    }


def _turn_run_name(default: str, trace_context: dict | None = None) -> str:
    if default == "frappe.sync_run":
        return default
    channel = str((trace_context or {}).get("channel") or "")
    if channel == "chainlit_audio":
        return "frappe.audio_turn"
    return "frappe.chat_turn"


def _graph_config(
    *,
    run_name: str,
    question: str,
    source_filter: str,
    image_data: list[dict] | None,
    input_type: str,
    session_uploads: list[str] | None,
    temperature: float | None,
    max_tokens: int | None,
    retrieval_strategy: str | None,
    use_rerank: bool | None,
    trace_context: dict | None,
) -> dict | None:
    return build_graph_config(
        run_name=run_name,
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


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def run_agent(
    question: str,
    chat_history: list | None = None,
    source_filter: str = "",
    image_data: list[dict] | None = None,
    input_type: str = "text",
    session_uploads: list[str] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    retrieval_strategy: str | None = None,
    use_rerank: bool | None = None,
    trace_context: dict | None = None,
    memory_hash: str = "",
) -> str:
    """Senkron agent çalıştırıcı (test veya CLI için)."""
    state = _init_state(
        question, chat_history, source_filter, image_data, input_type,
        session_uploads, temperature, max_tokens, retrieval_strategy, use_rerank,
    )
    config = _graph_config(
        run_name="frappe.sync_run",
        question=question,
        source_filter=source_filter,
        image_data=image_data,
        input_type=input_type,
        session_uploads=session_uploads,
        temperature=temperature,
        max_tokens=max_tokens,
        retrieval_strategy=retrieval_strategy,
        use_rerank=use_rerank,
        trace_context=trace_context,
    )
    result = get_graph().invoke(state, config=config) if config else get_graph().invoke(state)
    return result.get("generation", "Bir hata oluştu, lütfen tekrar deneyin.")


async def arun_agent(
    question: str,
    chat_history: list | None = None,
    source_filter: str = "",
    image_data: list[dict] | None = None,
    input_type: str = "text",
    session_uploads: list[str] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    retrieval_strategy: str | None = None,
    use_rerank: bool | None = None,
    trace_context: dict | None = None,
    memory_hash: str = "",
) -> str:
    """Asenkron agent çalıştırıcı."""
    state = _init_state(
        question, chat_history, source_filter, image_data, input_type,
        session_uploads, temperature, max_tokens, retrieval_strategy, use_rerank,
    )
    config = _graph_config(
        run_name=_turn_run_name("frappe.chat_turn", trace_context),
        question=question,
        source_filter=source_filter,
        image_data=image_data,
        input_type=input_type,
        session_uploads=session_uploads,
        temperature=temperature,
        max_tokens=max_tokens,
        retrieval_strategy=retrieval_strategy,
        use_rerank=use_rerank,
        trace_context=trace_context,
    )
    result = await get_graph().ainvoke(state, config=config) if config else await get_graph().ainvoke(state)
    return result.get("generation", "Bir hata oluştu, lütfen tekrar deneyin.")


async def astream_agent(
    question: str,
    chat_history: list | None = None,
    source_filter: str = "",
    image_data: list[dict] | None = None,
    input_type: str = "text",
    session_uploads: list[str] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    retrieval_strategy: str | None = None,
    use_rerank: bool | None = None,
    trace_context: dict | None = None,
    memory_hash: str = "",
):
    """Agent çalıştırıcı: mesaj ve güncelleme olaylarını stream eder.

    Semantic cache etkinse:
      - Önce benzer soru cache'te aranır; bulunursa pipeline atlanır.
      - Bulunamazsa pipeline çalışır ve yanıt cache'e yazılır.
    """
    from src.config import settings as _s

    # Semantic cache bağlam anahtarı — aynı soru farklı belge setlerinde
    # yanlış cache dönmesini engeller.
    def _build_cache_ctx() -> str:
        return build_semantic_cache_context(
            source_filter=source_filter,
            session_uploads=session_uploads,
            retrieval_strategy=retrieval_strategy,
            memory_hash=memory_hash,
        )

    def _should_use_semantic_cache() -> bool:
        if is_memory_command(question):
            return False
        if not (_s.semantic_cache_enabled and not image_data and input_type == "text"):
            return False
        # Lokal sohbetlerde cache embedding maliyeti cevabın kendisinden pahalı.
        # Cache'i belge/RAG bağlamı veya daha uzun, tekrar etmesi muhtemel sorular için tut.
        if source_filter or session_uploads:
            return True
        return len((question or "").strip()) >= 80

    # Semantic cache kontrolü — görsel/ses sorguları cache'e alınmaz
    use_semantic_cache = _should_use_semantic_cache()
    if use_semantic_cache:
        from src.rag.semantic_cache import SemanticCache
        from langchain_core.messages import AIMessageChunk

        cache_ctx = _build_cache_ctx()
        cached = await SemanticCache.get().lookup(question, cache_ctx=cache_ctx)
        if cached:
            record_semantic_cache_hit(
                question=question,
                cached_answer=cached,
                cache_ctx=cache_ctx,
                trace_context=trace_context,
            )
            yield ("updates", {"generator": {"generation": cached}})
            chunk_size = 20
            for i in range(0, len(cached), chunk_size):
                yield ("messages", (AIMessageChunk(content=cached[i:i + chunk_size]),
                                    {"langgraph_node": "generator"}))
            return
    else:
        cache_ctx = ""

    collected_generation = ""
    state = _init_state(
        question, chat_history, source_filter, image_data, input_type,
        session_uploads, temperature, max_tokens, retrieval_strategy, use_rerank,
    )
    config = _graph_config(
        run_name=_turn_run_name("frappe.chat_turn", trace_context),
        question=question,
        source_filter=source_filter,
        image_data=image_data,
        input_type=input_type,
        session_uploads=session_uploads,
        temperature=temperature,
        max_tokens=max_tokens,
        retrieval_strategy=retrieval_strategy,
        use_rerank=use_rerank,
        trace_context=trace_context,
    )
    kwargs = {"stream_mode": ["messages", "updates"]}
    if config:
        kwargs["config"] = config
    async for event in get_graph().astream(state, **kwargs):
        # Yanıtı cache için topla (generator / direct_response / vision)
        if isinstance(event, tuple) and event[0] == "updates":
            for node_name, delta in (event[1] or {}).items():
                if isinstance(delta, dict):
                    gen = delta.get("generation")
                    if gen and node_name in {"generator", "direct_response", "vision"}:
                        collected_generation = gen
        yield event

    # Cache'e yaz (görsel/ses sorgular hariç). Tanısal fallback cevapları cache'leme;
    # aksi halde geçici bir boş-LLM durumu sonraki iyi denemeleri de kirletir.
    is_diagnostic_fallback = collected_generation.startswith("Model bu turda boş yanıt döndürdü.")
    if use_semantic_cache and collected_generation and not is_diagnostic_fallback:
        from src.rag.semantic_cache import SemanticCache
        await SemanticCache.get().store(question, collected_generation, cache_ctx=cache_ctx)
