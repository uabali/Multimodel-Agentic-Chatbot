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

logger = logging.getLogger(__name__)


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

    Dosya-bağlı sorgularda (source_filter veya session_uploads aktifken) web
    aramasına düşmek tehlikeli: kullanıcı belgede olmayan bir alan sorduğunda
    (ör. CV'de olmayan LinkedIn URL'si) web'den konu-genelinde alakasız
    sonuçlar getirilmesin. Generator dürüstçe "bu bilgi belgede yok" der.
    """
    if state.get("relevance") == "yes":
        return "sufficient"
    if state.get("source_filter") or state.get("session_uploads"):
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
    """Graph singleton'ını döner (ilk çağrıda derlenir)."""
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
) -> str:
    """Senkron agent çalıştırıcı (test veya CLI için)."""
    result = get_graph().invoke(
        _init_state(question, chat_history, source_filter, image_data, input_type,
                    session_uploads, temperature, max_tokens, retrieval_strategy, use_rerank)
    )
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
) -> str:
    """Asenkron agent çalıştırıcı."""
    result = await get_graph().ainvoke(
        _init_state(question, chat_history, source_filter, image_data, input_type,
                    session_uploads, temperature, max_tokens, retrieval_strategy, use_rerank)
    )
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
):
    """Agent çalıştırıcı: mesaj ve güncelleme olaylarını stream eder."""
    async for event in get_graph().astream(
        _init_state(question, chat_history, source_filter, image_data, input_type,
                    session_uploads, temperature, max_tokens, retrieval_strategy, use_rerank),
        stream_mode=["messages", "updates"],
    ):
        yield event
