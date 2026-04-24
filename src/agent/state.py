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

    # Router kararı: "rag" | "direct"
    route: str

    # Grader kararı: "yes" | "no"
    relevance: str

    # Grader "no" sebep kodu: "irrelevant" | "needs_live_data" | ""
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
