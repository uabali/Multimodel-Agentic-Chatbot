"""
Unified retriever — SOLID uyumlu, modern LangChain API kullanımı.

Tasarım kararları:
 - SRP: Her fonksiyon tek bir operasyon yapar (k hesaplama, strateji seçimi, güven tahmini…).
 - OCP: Yeni retrieval stratejisi eklemek için sadece `_STRATEGY_MAP`'e giriş eklenir.
 - DIP: `create_retriever` somut LangChain sınıflarına değil, vectorstore duck-type arayüzüne bağlıdır.

Modern LangChain:
 - `retriever.invoke(query)` kullanılıyor (deprecated `get_relevant_documents` kaldırıldı).
 - `as_retriever(search_type, search_kwargs)` standart `VectorStoreRetriever` döner.
"""

from __future__ import annotations

import logging
import re
from typing import Callable

from langchain_core.documents import Document

from src.config import settings

logger = logging.getLogger(__name__)

# Güven hesabında göz ardı edilen yaygın kelimeler
_STOPWORDS: frozenset[str] = frozenset({
    "ve", "ile", "icin", "için", "bu", "şu", "su", "bir", "o", "da", "de", "mi", "mu",
    "ne", "nedir", "ama", "fakat", "ancak", "yani", "olan", "olan",
    "the", "is", "are", "what", "who", "where", "when", "how", "in", "on", "of",
})


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic K — sorunun karmaşıklığına göre kaç belge çekileceğini ayarlar
# ─────────────────────────────────────────────────────────────────────────────

_COMPLEXITY_KEYWORDS: list[str] = [
    "ve", "neden", "nasil", "hangi", "ne zaman", "kim",
    "arasindaki fark", "karsilastir",
    "and", "how", "why", "which",
]


def calculate_dynamic_k(question: str, base_k: int = 8, max_k: int = 15) -> int:
    """Soru karmaşıklığına göre dinamik k değeri döner."""
    q = question.lower()
    score = sum(1 for kw in _COMPLEXITY_KEYWORDS if kw in q)
    if score >= 2:
        return min(base_k + 4, max_k)
    if score == 1:
        return min(base_k + 2, max_k)
    return base_k


# ─────────────────────────────────────────────────────────────────────────────
# Auto Strategy — sorunun türüne göre retrieval stratejisi seçer (OCP)
# ─────────────────────────────────────────────────────────────────────────────

# Yeni strateji eklemek → bu dict'e satır eklemek yeterli; başka yer değişmez.
_STRATEGY_MAP: list[tuple[list[str], str]] = [
    (["kac", "sure", "ne zaman", "dakika"], "hybrid"),
    (["neden", "nasil"], "mmr"),
    (["kullanim alanlari", "hangi projelerde", "nerelerde kullanilir"], "threshold"),
]


def auto_select_strategy(question: str) -> str:
    """Soruya göre en uygun retrieval stratejisini seçer."""
    q = question.lower()
    for keywords, strategy in _STRATEGY_MAP:
        if any(kw in q for kw in keywords):
            return strategy
    return "similarity"


# ─────────────────────────────────────────────────────────────────────────────
# Confidence estimation
# ─────────────────────────────────────────────────────────────────────────────


def estimate_confidence(query: str, docs: list[Document]) -> float:
    """Sorgu ile belgeler arasındaki uyum güvenini 0–1 aralığında tahmin eder."""
    if not docs:
        return 0.0

    # Rerank skoru varsa kullan
    top_scores = [
        doc.metadata.get("rerank_score")
        for doc in docs[:3]
        if doc.metadata.get("rerank_score") is not None
    ]
    if top_scores:
        best = max(top_scores)
        thresh = settings.local_search_conf_threshold
        # best >= thresh → confidence [0.5, 1.0]; best < thresh → confidence [0.0, 0.5)
        if best >= thresh:
            return min(1.0, 0.5 + 0.5 * (best - thresh) / max(1.0 - thresh, 1e-6))
        return max(0.0, 0.5 * best / max(thresh, 1e-6))

    # Fallback: term overlap — re.UNICODE Türkçe/Arapça/vb. karakterleri kapsar
    terms = [
        t for t in re.findall(r"[\w]+", query.lower(), re.UNICODE)
        if len(t) >= 3 and t not in _STOPWORDS
    ]
    if not terms:
        return 0.0

    joined = " ".join((doc.page_content or "").lower() for doc in docs[:3])
    if not joined.strip():
        return 0.0

    overlap = sum(1 for t in terms if t in joined)
    coverage = overlap / max(len(terms), 1)
    return min(1.0, coverage * 1.15) if len(terms) > 4 else coverage


# ─────────────────────────────────────────────────────────────────────────────
# run_retriever — modern LangChain API
# ─────────────────────────────────────────────────────────────────────────────


def run_retriever(retriever, query: str) -> list[Document]:
    """Retriever'ı invoke API'siyle çalıştırır (get_relevant_documents kaldırıldı)."""
    if hasattr(retriever, "invoke"):
        docs = retriever.invoke(query)
    elif callable(retriever):
        # Reranker wrapper fonksiyonları (geriye dönük uyumluluk)
        docs = retriever(query)
    else:
        logger.warning("run_retriever: bilinmeyen retriever tipi: %s", type(retriever))
        return []
    return docs if isinstance(docs, list) else []


# ─────────────────────────────────────────────────────────────────────────────
# create_retriever — ana fabrika fonksiyonu
# ─────────────────────────────────────────────────────────────────────────────


def create_retriever(
    vectorstore,
    question: str,
    strategy: str = "auto",
    base_k: int = 8,
    max_k: int | None = None,
    fetch_k: int = 30,
    lambda_mult: float = 0.6,
    score_threshold: float = 0.70,
    use_rerank: bool = False,
    reranker=None,
    rerank_top_n: int = 20,
    qdrant_filter=None,
):
    """Ayarlara göre uygun LangChain retriever'ı oluşturur ve döner.

    Args:
        vectorstore: LangChain VectorStore (as_retriever'ı olmalı).
        question:    Kullanıcı sorusu (k ve strateji hesabı için).
        strategy:    "auto" | "similarity" | "hybrid" | "mmr" | "threshold".
        use_rerank:  True ise reranker wrapper fonksiyonu döner.
        reranker:    CrossEncoder instance (use_rerank=True olduğunda gerekli).
        qdrant_filter: Qdrant metadata filtresi (belirli dosyayla sınırlamak için).

    Returns:
        BaseRetriever veya callable (rerank wrapper).
    """
    should_rerank = use_rerank and reranker is not None
    k = calculate_dynamic_k(question, base_k, max_k=max_k if max_k is not None else base_k + 4)

    if strategy == "auto":
        strategy = auto_select_strategy(question)

    search_k = max(rerank_top_n, k * 2) if should_rerank else k
    search_kwargs: dict = {"k": search_k}

    if qdrant_filter is not None:
        search_kwargs["filter"] = qdrant_filter

    if strategy in ("similarity", "hybrid"):
        base = vectorstore.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
    elif strategy == "mmr":
        search_kwargs.update({"fetch_k": fetch_k, "lambda_mult": lambda_mult})
        base = vectorstore.as_retriever(search_type="mmr", search_kwargs=search_kwargs)
    elif strategy == "threshold":
        search_kwargs["score_threshold"] = score_threshold
        base = vectorstore.as_retriever(
            search_type="similarity_score_threshold", search_kwargs=search_kwargs
        )
    else:
        base = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": search_k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
        )

    if should_rerank:
        from src.rag.reranker import create_rerank_retriever

        def _rerank_wrapper(q: str) -> list[Document]:
            return create_rerank_retriever(base, q, reranker, top_k=k, rerank_top_n=rerank_top_n)

        return _rerank_wrapper

    return base
