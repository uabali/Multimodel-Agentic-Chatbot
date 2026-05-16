"""
LangGraph agent state — SOLID'e uygun, tek sorumluluk ilkesiyle tanımlanmış.

Açıklama:
- Her alan net bir anlama sahip; birden fazla işi yok.
- `add_messages` reducer sayesinde mesajlar immutable şekilde birikerek güncellenir.
- Yeni bir alan eklemek graph mantığını bozmaz (OCP).
"""

from typing import Annotated, Sequence

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Graph boyunca taşınan paylaşımlı durum."""

    # Konuşma geçmişi — add_messages her güncellemede mesajları birleştirir
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Retriever tarafından bulunan belgeler
    documents: list[Document]

    # Kullanıcının orijinal sorusu — rewriter değiştirmez, web search bu alanı kullanır
    original_question: str

    # Kullanıcının son sorusu (rewriter değiştirerek güncelleyebilir)
    question: str

    # Üretilen yanıt metni
    generation: str

    # Router kararı: "rag" | "direct" | "web" | "vision"
    route: str

    # Grader kararı: "yes" | "no"
    relevance: str

    # Grader reason: "sufficient" | "partial" | "insufficient_context" | "needs_live_data" | "irrelevant" | ""
    # source_filter aktifken yalnızca "needs_live_data" web fallback'e izin verir.
    grader_reason: str

    # Yüklenen dosyaya göre retrieval filtreleme (boş = tüm belgeler)
    source_filter: str

    # Session boyunca yüklenmiş dosyaların adları (kümülatif).
    # source_filter boşsa retriever/grader bu listeyi kullanır; router da
    # "belge sahibi" sinyali olarak değerlendirir (follow-up sorular için kritik).
    session_uploads: list[str]

    # Multimodal: base64 encode edilmiş görseller
    # Her eleman: {"mime": "image/png", "base64": "...", "name": "foto.png"}
    image_data: list[dict]

    # Input tipi ipucu: "text" | "image" | "audio"
    input_type: str

    # vision_rag_node tarafından doldurulur: görselden çıkarılan metin/yapı
    # Generator, bunu belge bağlamına [Görsel Analizi] kaynağı olarak ekler
    vision_context: str

    # Kullanıcı ayar panelinden gelen per-session geçersiz kılmalar
    temperature: float
    max_tokens: int
    retrieval_strategy: str
    use_rerank: bool

    # Retrieval explainability — her chunk için per-stage skorlar.
    # Her eleman: {chunk_id, hybrid_score, rerank_score, used_in_context}
    # retriever_node doldurur, generator_node used_in_context=True işaretler.
    retrieval_trace: list[dict]
    retrieval_gate: str
    refusal_mode: bool
    retry_count: int

    # LangSmith/CSV okunabilirliği için sanitize edilmiş özet alanları.
    web_search_error: str
    answer_preview: str
    answer_chars: int
    document_count: int
    used_context_count: int
    document_previews: list[dict]
    retrieval_trace_summary: dict
    top_sources: str
    top_chunks: str
    used_chunks: str
    retry_summary: dict
    retry_path: str
    latency_ms_by_stage: dict
