"""
Unified Qdrant vector store — hybrid dense+sparse, smart reindex, dense gate.

Sources:
  - Frappe/src/vectorstore.py: fingerprint-based smart reindex, wait-for-qdrant
  - Local-llm/optimized_vector_store.py: named dense+sparse vectors, FastEmbedSparse, dense gate
  - Final-Project/src/rag/ingest.py: simple collection ensure
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models

from functools import lru_cache

from src.config import settings
from src.rag.embeddings import get_embeddings, get_embedding_dim


@lru_cache(maxsize=256)
def _cached_embed_query(query: str) -> list:
    """Dense gate için embed sonucunu önbellekler (aynı sorgu retriever'da tekrar embed edilmez)."""
    return get_embeddings().embed_query(query)

logger = logging.getLogger(__name__)

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"




def _fingerprint_docs(docs: list[Document]) -> str:
    """Kısa: `_fingerprint_docs` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
    h = hashlib.sha256()
    h.update(f"n={len(docs)}".encode())
    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}
        source = str(meta.get("source", ""))
        page = str(meta.get("page", ""))
        content = getattr(doc, "page_content", "") or ""
        h.update(source.encode("utf-8", errors="ignore"))
        h.update(page.encode("utf-8", errors="ignore"))
        h.update(content.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def _fingerprint_path(collection_name: str) -> Path:
    """Kısa: `_fingerprint_path` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
    base = Path(".rag_cache")
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{collection_name}.fingerprint"


def _load_fingerprint(collection_name: str) -> Optional[str]:
    """Kısa: `_load_fingerprint` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
    fp = _fingerprint_path(collection_name)
    if not fp.exists():
        return None
    return fp.read_text(encoding="utf-8").strip() or None


def _save_fingerprint(collection_name: str, fingerprint: str) -> None:
    """Kısa: `_save_fingerprint` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
    _fingerprint_path(collection_name).write_text(fingerprint, encoding="utf-8")


def _unlink_fingerprint(collection_name: str) -> None:
    """Kısa: `_unlink_fingerprint` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
    try:
        fp = _fingerprint_path(collection_name)
        if fp.exists():
            fp.unlink()
    except OSError as exc:
        logger.debug("Fingerprint unlink skipped: %s", exc)


def _read_dense_vector_size(
    client: QdrantClient, collection_name: str, vector_name: str
) -> tuple[int | None, str]:
    """Return (vector_size, layout) where layout is 'named', 'unnamed', or 'missing'."""
    info = client.get_collection(collection_name=collection_name)
    vectors = info.config.params.vectors
    if vectors is None:
        return None, "missing"
    if isinstance(vectors, dict):
        if vector_name not in vectors:
            return None, "missing"
        vp = vectors[vector_name]
        size = getattr(vp, "size", None)
        return (int(size) if size is not None else None), "named"
    size = getattr(vectors, "size", None)
    return (int(size) if size is not None else None), "unnamed"


def _collection_has_sparse_vector(
    client: QdrantClient, collection_name: str, sparse_vector_name: str
) -> bool:
    """Kısa: `_collection_has_sparse_vector` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
    info = client.get_collection(collection_name=collection_name)
    sparse = info.config.params.sparse_vectors
    if sparse is None:
        return False
    return sparse_vector_name in sparse




def _wait_for_qdrant(client: QdrantClient, timeout_s: float = 20.0, retry_s: float = 1.0) -> None:
    """Kısa: `_wait_for_qdrant` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
    deadline = time.monotonic() + max(0.0, timeout_s)
    last_exc: Exception | None = None
    while True:
        try:
            client.get_collections()
            return
        except Exception as exc:
            last_exc = exc
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Qdrant connection failed: {settings.qdrant_url}. "
                    "Ensure the service is running and ports are open."
                ) from last_exc
            time.sleep(max(0.2, retry_s))


_qdrant_client: Optional[QdrantClient] = None
_qdrant_lock = threading.Lock()


def get_qdrant_client() -> QdrantClient:
    """Kısa: `get_qdrant_client` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client
    with _qdrant_lock:
        if _qdrant_client is not None:
            return _qdrant_client
        client = QdrantClient(url=settings.qdrant_url, prefer_grpc=settings.qdrant_prefer_grpc, check_compatibility=False)
        _wait_for_qdrant(client)
        _qdrant_client = client
    return _qdrant_client




class HybridVectorStore:
    """
    Qdrant with named dense + sparse vectors, hybrid retrieval,
    dense gate, and smart reindex.
    """

    def __init__(self, client: Optional[QdrantClient] = None) -> None:
        """Kısa: `__init__` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        self.client = client or get_qdrant_client()
        self.embeddings = get_embeddings()
        self._sparse = self._init_sparse()
        self._store: Optional[QdrantVectorStore] = None

    def _init_sparse(self):
        """Kısa: `_init_sparse` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        try:
            from langchain_qdrant import FastEmbedSparse
            return FastEmbedSparse(model_name="Qdrant/bm25", batch_size=64)
        except Exception:
            logger.warning("FastEmbedSparse unavailable; sparse retrieval disabled.")
            return None

    def _collection_rebuild_reason(self, expected_dim: int, use_hybrid: bool) -> str:
        """Return a non-empty string if the existing collection must be recreated."""
        if not self.client.collection_exists(settings.qdrant_collection):
            return ""

        dense_size, layout = _read_dense_vector_size(
            self.client, settings.qdrant_collection, DENSE_VECTOR
        )
        if layout != "named":
            return (
                f"expected named dense vector {DENSE_VECTOR!r}, "
                f"found layout={layout!r} (recreate collection)"
            )
        if dense_size is None or dense_size != expected_dim:
            return (
                f"dense vector size mismatch (collection={dense_size}, "
                f"current_embeddings={expected_dim})"
            )
        if use_hybrid and not _collection_has_sparse_vector(
            self.client, settings.qdrant_collection, SPARSE_VECTOR
        ):
            return "hybrid retrieval requires sparse vector but collection has none"
        return ""

    def _rebuild_collection(self, reason: str) -> None:
        """Kısa: `_rebuild_collection` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        logger.warning(
            "Deleting Qdrant collection %r and fingerprint cache: %s",
            settings.qdrant_collection,
            reason,
        )
        try:
            self.client.delete_collection(settings.qdrant_collection)
        except Exception as exc:
            logger.warning("delete_collection failed (may already be gone): %s", exc)
        self._store = None
        _unlink_fingerprint(settings.qdrant_collection)

    @property
    def store(self) -> QdrantVectorStore:
        """Kısa: `store` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        if self._store is None:
            expected_dim = get_embedding_dim()
            use_hybrid = self._sparse is not None
            reason = self._collection_rebuild_reason(expected_dim, use_hybrid)
            if reason:
                if not settings.qdrant_auto_recreate_on_mismatch:
                    raise RuntimeError(
                        f"Qdrant collection {settings.qdrant_collection!r} is incompatible with "
                        f"the current embedding model: {reason}. "
                        "Fix: set QDRANT_AUTO_RECREATE_ON_MISMATCH=true, switch EMBEDDING_MODEL "
                        "to match the existing index, or delete the collection in Qdrant."
                    )
                self._rebuild_collection(reason)
            self._ensure_collection()
            self._ensure_payload_indexes()
            mode = RetrievalMode.HYBRID if self._sparse else RetrievalMode.DENSE
            self._store = QdrantVectorStore(
                client=self.client,
                collection_name=settings.qdrant_collection,
                embedding=self.embeddings,
                sparse_embedding=self._sparse,
                retrieval_mode=mode,
                vector_name=DENSE_VECTOR,
                sparse_vector_name=SPARSE_VECTOR if self._sparse else None,
            )
        return self._store

    def _ensure_collection(self) -> None:
        """Kısa: `_ensure_collection` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        if self.client.collection_exists(settings.qdrant_collection):
            return

        dim = get_embedding_dim()
        vectors_config = {
            DENSE_VECTOR: models.VectorParams(size=dim, distance=models.Distance.COSINE)
        }
        sparse_config = {}
        if self._sparse:
            sparse_config[SPARSE_VECTOR] = models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False)
            )

        self.client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_config if sparse_config else None,
        )
        logger.info("Created Qdrant collection '%s' (dim=%d)", settings.qdrant_collection, dim)
        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self) -> None:
        """Payload index for metadata filters (idempotent)."""
        if not self.client.collection_exists(settings.qdrant_collection):
            return
        indexed_fields = (
            ("metadata.source_file", models.PayloadSchemaType.KEYWORD),
            ("metadata.user_id", models.PayloadSchemaType.KEYWORD),
            ("metadata.thread_id", models.PayloadSchemaType.KEYWORD),
        )
        for field_name, field_schema in indexed_fields:
            try:
                self.client.create_payload_index(
                    collection_name=settings.qdrant_collection,
                    field_name=field_name,
                    field_schema=field_schema,
                )
            except Exception as exc:
                msg = str(exc).lower()
                if "already exists" not in msg and "already indexed" not in msg:
                    logger.debug("Payload index %s skipped: %s", field_name, exc)

    # ── Ingest ──

    def add_documents(self, documents: list[Document], batch_size: int = 100) -> list[str]:
        """Kısa: `add_documents` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        all_ids: list[str] = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i: i + batch_size]
            ids = [str(uuid.uuid4()) for _ in batch]
            added = self.store.add_documents(documents=batch, ids=ids)
            all_ids.extend(added)
        return all_ids

    def smart_reindex(self, documents: list[Document]) -> bool:
        """Returns True if reindex happened, False if skipped."""
        mode = settings.qdrant_auto_reindex.lower()
        if mode == "false":
            return False

        current_fp = _fingerprint_docs(documents)
        last_fp = _load_fingerprint(settings.qdrant_collection)

        if mode == "smart" and current_fp == last_fp:
            logger.info("Smart reindex: fingerprint unchanged, skipping.")
            return False

        if self.client.collection_exists(settings.qdrant_collection):
            try:
                self.client.delete_collection(settings.qdrant_collection)
            except Exception:
                pass
            self._store = None

        self._ensure_collection()
        self.add_documents(documents)
        _save_fingerprint(settings.qdrant_collection, current_fp)
        logger.info("Reindexed %d documents.", len(documents))
        return True

    # ── Search methods ──

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        mode: Optional[RetrievalMode] = None,
        score_threshold: Optional[float] = None,
    ) -> list[Document]:
        """Kısa: `similarity_search` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        if k is None:
            k = settings.top_k
        effective_mode = mode if mode is not None else (
            RetrievalMode.HYBRID if self._sparse else RetrievalMode.DENSE
        )
        # Create a temporary store with the requested mode to avoid mutating shared state
        tmp_store = QdrantVectorStore(
            client=self.client,
            collection_name=settings.qdrant_collection,
            embedding=self.embeddings,
            sparse_embedding=self._sparse,
            retrieval_mode=effective_mode,
            vector_name=DENSE_VECTOR,
            sparse_vector_name=SPARSE_VECTOR if self._sparse else None,
        )
        return tmp_store.similarity_search(query, k=k, score_threshold=score_threshold)

    def similarity_search_hybrid(self, query: str, k: Optional[int] = None) -> list[Document]:
        """Kısa: `similarity_search_hybrid` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        return self.similarity_search(query, k, RetrievalMode.HYBRID)

    def similarity_search_dense(self, query: str, k: Optional[int] = None) -> list[Document]:
        """Kısa: `similarity_search_dense` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        return self.similarity_search(query, k, RetrievalMode.DENSE)

    def max_dense_similarity(self, query: str, k: Optional[int] = None, qdrant_filter=None) -> float:
        """Best dense cosine score — used for the dense gate decision.

        Uses the Qdrant client directly to avoid mutating QdrantVectorStore.retrieval_mode,
        which would be a race condition under concurrent async requests.
        """
        if k is None:
            k = settings.rag_dense_gate_k
        try:
            query_vector = _cached_embed_query(query)
            query_response = self.client.query_points(
                collection_name=settings.qdrant_collection,
                query=query_vector,
                using=DENSE_VECTOR,
                limit=k,
                query_filter=qdrant_filter,
                with_payload=False,
                with_vectors=False,
            )
            results = query_response.points
        except Exception as exc:
            logger.warning("max_dense_similarity search failed: %s", exc)
            return 0.0
        if not results:
            return 0.0
        return max(r.score for r in results)

    def as_retriever(self, search_type: str = "similarity", search_kwargs: Optional[dict] = None):
        """LangChain-compatible retriever for graph nodes."""
        kw = search_kwargs or {"k": settings.top_k}
        return self.store.as_retriever(search_type=search_type, search_kwargs=kw)

    def delete_by_source(self, source_files: list[str]) -> int:
        """Verilen dosya adlarına ait tüm chunk'ları Qdrant'tan siler.

        Returns:
            Silinen dosya sayısı (işlem başarısız olsa 0).
        """
        if not source_files:
            return 0
        try:
            self.client.delete(
                collection_name=settings.qdrant_collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[models.FieldCondition(
                            key="metadata.source_file",
                            match=models.MatchAny(any=source_files),
                        )]
                    )
                ),
            )
            logger.info("Qdrant: silindi — kaynak(lar): %s", source_files)
            return len(source_files)
        except Exception as exc:
            logger.warning("delete_by_source başarısız: %s", exc)
            return 0

    def get_point_count(self) -> int:
        """Kısa: `get_point_count` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        try:
            info = self.client.get_collection(settings.qdrant_collection)
            return getattr(info, "points_count", 0) or 0
        except Exception:
            return 0

    def optimize_storage(self) -> None:
        """Kısa: `optimize_storage` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        try:
            self.client.update_collection(
                collection_name=settings.qdrant_collection,
                optimizers_config=models.OptimizersConfig(
                    deleted_threshold=0.2,
                    vacuum_min_vector_number=1000,
                    default_segment_number=2,
                    flush_interval_sec=10,
                    max_optimization_threads=2,
                ),
            )
        except Exception as exc:
            logger.warning("optimize_storage skipped: %s", exc)


_hybrid_store: Optional[HybridVectorStore] = None
_hybrid_store_lock = threading.Lock()


def get_hybrid_store() -> HybridVectorStore:
    """Kısa: `get_hybrid_store` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
    global _hybrid_store
    if _hybrid_store is not None:
        return _hybrid_store
    with _hybrid_store_lock:
        if _hybrid_store is not None:
            return _hybrid_store
        _hybrid_store = HybridVectorStore()
    return _hybrid_store
