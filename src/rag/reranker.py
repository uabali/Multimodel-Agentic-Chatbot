"""
Cross-encoder reranking with TTL cache.

Source: Frappe/src/reranker.py (full port with improvements).
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

try:
    from cachetools import TTLCache
    _CACHE_OK = True
except ImportError:
    _CACHE_OK = False

try:
    from sentence_transformers import CrossEncoder
    _CE_OK = True
except ImportError:
    _CE_OK = False

RERANKER_DEFAULT = "BAAI/bge-reranker-base"
RERANKER_FAST = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_rerank_cache: Optional[TTLCache] = None


def _get_cache() -> Optional[TTLCache]:
    global _rerank_cache
    if _rerank_cache is None and _CACHE_OK:
        _rerank_cache = TTLCache(
            maxsize=int(os.getenv("RERANK_CACHE_SIZE", "100")),
            ttl=int(os.getenv("RERANK_CACHE_TTL", "600")),
        )
    return _rerank_cache


def _cache_key(query: str, docs: List[Document], top_k: Optional[int]) -> str:
    parts = sorted(d.page_content[:200] for d in docs)
    raw = query + "||" + "||".join(parts) + f"||{top_k}"
    return hashlib.md5(raw.encode()).hexdigest()


def resolve_model_name(name: Optional[str] = None) -> str:
    if name:
        if name.lower() == "fast":
            return RERANKER_FAST
        if name.lower() == "default":
            return RERANKER_DEFAULT
        return name
    env = os.getenv("RERANKER_MODEL", "").strip()
    if env:
        if env.lower() == "fast":
            return RERANKER_FAST
        if env.lower() == "default":
            return RERANKER_DEFAULT
        return env
    return RERANKER_DEFAULT


def create_reranker(model_name: Optional[str] = None, device: str = "cpu"):
    if not _CE_OK:
        raise ImportError("sentence-transformers required for reranking.")
    resolved = resolve_model_name(model_name)
    logger.info("Loading reranker: %s (device=%s)", resolved, device)
    return CrossEncoder(resolved, device=device)


def rerank_documents(
    query: str,
    documents: List[Document],
    reranker,
    top_k: Optional[int] = None,
    batch_size: int = 8,
    use_cache: bool = True,
) -> List[Document]:
    if not documents:
        return []
    if not _CE_OK:
        return documents[:top_k] if top_k else documents

    cache = _get_cache() if use_cache else None
    key = None
    if cache is not None:
        key = _cache_key(query, documents, top_k)
        cached = cache.get(key)
        if cached is not None:
            return cached

    pairs = [[query, doc.page_content] for doc in documents]
    try:
        scores = reranker.predict(pairs, batch_size=batch_size)
    except Exception as exc:
        logger.warning("Reranking failed: %s", exc)
        return documents[:top_k] if top_k else documents

    scored = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    result = []
    for score, doc in (scored[:top_k] if top_k else scored):
        # Yeni metadata dict ile kopyala — orijinal Document'ı mutate etme
        new_meta = {**doc.metadata, "rerank_score": float(score)}
        result.append(Document(page_content=doc.page_content, metadata=new_meta))

    if cache is not None and key is not None:
        cache[key] = result
    return result


def create_rerank_retriever(
    base_retriever,
    query: str,
    reranker,
    top_k: Optional[int] = None,
    rerank_top_n: int = 20,
    batch_size: int = 8,
) -> List[Document]:
    if hasattr(base_retriever, "invoke"):
        docs = base_retriever.invoke(query)
    elif callable(base_retriever):
        docs = base_retriever(query)
    else:
        docs = []

    if len(docs) <= 1:
        return docs[:top_k] if top_k else docs

    return rerank_documents(
        query=query,
        documents=docs[:rerank_top_n],
        reranker=reranker,
        top_k=top_k,
        batch_size=batch_size,
    )
