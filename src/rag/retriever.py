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

import hashlib
import logging
import re

from langchain_core.documents import Document

from src.config import settings

logger = logging.getLogger(__name__)

# Güven hesabında göz ardı edilen yaygın kelimeler
_STOPWORDS: frozenset[str] = frozenset({
    "ve", "ile", "icin", "için", "bu", "şu", "su", "bir", "o", "da", "de", "mi", "mu",
    "ne", "nedir", "nerede", "nereden", "nereye", "hangi", "kaç", "kac", "kim",
    "dosya", "belge", "pdf", "içindeki", "icindeki", "bulunan", "yer", "alan",
    "ama", "fakat", "ancak", "yani", "olan", "olarak",
    "the", "is", "are", "what", "who", "where", "when", "how", "in", "on", "of",
    "this", "that", "file", "document", "uploaded",
})

_TR_TRANSLATION = str.maketrans({
    "ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u",
    "Ç": "c", "Ğ": "g", "İ": "i", "I": "i", "Ö": "o", "Ş": "s", "Ü": "u",
})


def normalize_query_text(text: str) -> str:
    """Türkçe karakterleri ve fazla boşlukları arama heuristics için normalize eder."""
    return re.sub(r"\s+", " ", (text or "").translate(_TR_TRANSLATION).lower()).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic K — sorunun karmaşıklığına göre kaç belge çekileceğini ayarlar
# ─────────────────────────────────────────────────────────────────────────────

_COMPLEXITY_KEYWORDS: list[str] = [
    "ve", "neden", "nasil", "hangi", "ne zaman", "kim", "karsilastir",
    "arasindaki fark", "avantaj", "dezavantaj", "ozetle", "acikla",
    "and", "how", "why", "which", "compare", "summarize", "explain",
]


def calculate_dynamic_k(question: str, base_k: int = 8, max_k: int = 15) -> int:
    """Soru karmaşıklığına göre dinamik k değeri döner."""
    q = normalize_query_text(question)
    score = sum(1 for kw in _COMPLEXITY_KEYWORDS if kw in q)
    if len(q.split()) >= 18:
        score += 1
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
    (["nereden", "nereye", "kalkis", "varis", "pnr", "bilet", "ticket"], "hybrid"),
    (["kac", "sure", "ne zaman", "dakika", "tarih", "saat"], "hybrid"),
    (["neden", "nasil", "karsilastir", "fark"], "mmr"),
    (["kullanim alanlari", "hangi projelerde", "nerelerde kullanilir"], "threshold"),
]


def auto_select_strategy(question: str) -> str:
    """Soruya göre en uygun retrieval stratejisini seçer."""
    q = normalize_query_text(question)
    for keywords, strategy in _STRATEGY_MAP:
        if any(kw in q for kw in keywords):
            return strategy
    return "similarity"


def _tokenize_for_overlap(text: str) -> list[str]:
    return [
        t for t in re.findall(r"[\w]+", normalize_query_text(text), re.UNICODE)
        if len(t) >= 3 and t not in _STOPWORDS
    ]


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
    terms = _tokenize_for_overlap(query)
    if not terms:
        return 0.0

    joined = normalize_query_text(" ".join((doc.page_content or "") for doc in docs[:3]))
    if not joined.strip():
        return 0.0

    overlap = sum(1 for t in terms if t in joined)
    coverage = overlap / max(len(terms), 1)
    return min(1.0, coverage * 1.15) if len(terms) > 4 else coverage


def deduplicate_documents(documents: list[Document], max_docs: int | None = None) -> list[Document]:
    """Aynı chunk'ın dense/sparse/rerank yollarından tekrar gelmesini engeller."""
    unique: list[Document] = []
    seen: set[str] = set()

    for doc in documents:
        meta = getattr(doc, "metadata", {}) or {}
        content = (doc.page_content or "").strip()
        if not content:
            continue
        source = meta.get("source_file") or meta.get("source") or ""
        page = str(meta.get("page", ""))
        chunk_index = str(meta.get("chunk_index", ""))
        if source or page or chunk_index:
            key = f"{source}|{page}|{chunk_index}|{hashlib.sha1(content[:240].encode()).hexdigest()}"
        else:
            key = hashlib.sha1(content[:500].encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
        if max_docs is not None and len(unique) >= max_docs:
            break
    return unique


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
