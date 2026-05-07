"""
Semantic cache — sık sorulan sorular için Qdrant tabanlı yanıt önbelleği.

Akış:
  lookup(question) → embed → Qdrant nearest-neighbor (threshold + TTL) → cached response
  store(question, response) → embed → Qdrant upsert

Mevcut altyapıdan yeniden kullanılanlar:
  - get_qdrant_client(), _cached_embed_query() (vectorstore.py)
  - settings.semantic_cache_* (config.py)
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from typing import Optional

from src.config import settings

logger = logging.getLogger(__name__)

_COLLECTION = "semantic_cache"


def _normalize(question: str) -> str:
    """Küçük harf + fazla boşluk temizleme — cache eşleşme kalitesini artırır."""
    return re.sub(r"\s+", " ", question.lower().strip())


class SemanticCache:
    """Qdrant tabanlı semantic cache singleton."""

    _instance: Optional["SemanticCache"] = None
    _collection_ready = False
    _disabled_until = 0.0
    _warned_unavailable = False
    _cooldown_s = 60.0

    @classmethod
    def get(cls) -> "SemanticCache":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _is_temporarily_disabled(self) -> bool:
        return time.monotonic() < self._disabled_until

    def _mark_unavailable(self, exc: Exception) -> None:
        self._collection_ready = False
        self._disabled_until = time.monotonic() + self._cooldown_s
        if not self._warned_unavailable:
            logger.warning(
                "SemanticCache: Qdrant kullanılamıyor, %.0fs boyunca cache atlanacak: %s",
                self._cooldown_s, exc,
            )
            self._warned_unavailable = True
        else:
            logger.debug("SemanticCache: Qdrant hâlâ kullanılamıyor: %s", exc)

    def _ensure_collection(self) -> None:
        if self._collection_ready:
            return
        if self._is_temporarily_disabled():
            return
        try:
            from qdrant_client import models
            from qdrant_client import QdrantClient
            from src.rag.embeddings import get_embedding_dim

            client = QdrantClient(
                url=settings.qdrant_url,
                prefer_grpc=settings.qdrant_prefer_grpc,
                check_compatibility=False,
                timeout=1,
            )
            client.get_collections()
            if not client.collection_exists(_COLLECTION):
                dim = get_embedding_dim()
                client.create_collection(
                    collection_name=_COLLECTION,
                    vectors_config=models.VectorParams(
                        size=dim,
                        distance=models.Distance.COSINE,
                    ),
                )
                logger.info("SemanticCache: koleksiyon oluşturuldu (dim=%d)", dim)
            self._collection_ready = True
            self._warned_unavailable = False
        except Exception as exc:
            self._mark_unavailable(exc)

    async def lookup(self, question: str, cache_ctx: str = "") -> Optional[str]:
        """Benzer soru cache'te varsa yanıtı döner, yoksa None.

        cache_ctx: source_filter/session_uploads/retrieval_strategy'den türetilen
        bağlam özeti — farklı belge setlerindeki aynı sorunun yanlış cache
        dönmesini engeller.
        """
        if not settings.semantic_cache_enabled:
            return None
        if self._is_temporarily_disabled():
            return None
        try:
            from qdrant_client import models
            from src.rag.vectorstore import get_qdrant_client, _cached_embed_query

            await asyncio.to_thread(self._ensure_collection)
            if not self._collection_ready:
                return None

            normalized = _normalize(question)
            embedding = await asyncio.to_thread(_cached_embed_query, normalized)

            cutoff = time.time() - settings.semantic_cache_ttl_hours * 3600
            client = get_qdrant_client()
            query_response = await asyncio.to_thread(
                client.query_points,
                collection_name=_COLLECTION,
                query=embedding,
                limit=1,
                score_threshold=settings.semantic_cache_threshold,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="timestamp",
                            range=models.Range(gte=cutoff),
                        ),
                        models.FieldCondition(
                            key="context_key",
                            match=models.MatchValue(value=cache_ctx),
                        ),
                    ]
                ),
            )
            results = query_response.points
            if results:
                score = results[0].score
                cached = results[0].payload.get("response", "")
                logger.info(
                    "SemanticCache: hit [score=%.3f, threshold=%.3f, ctx=%.12s, q=%.60s]",
                    score, settings.semantic_cache_threshold, cache_ctx or "none", question,
                )
                return cached
        except Exception as exc:
            logger.warning("SemanticCache lookup hatası: %s", exc)
        return None

    async def store(self, question: str, response: str, cache_ctx: str = "") -> None:
        """Soru–yanıt çiftini cache'e kaydeder."""
        if not settings.semantic_cache_enabled:
            return
        if self._is_temporarily_disabled():
            return
        try:
            from qdrant_client import models
            from src.rag.vectorstore import get_qdrant_client, _cached_embed_query

            await asyncio.to_thread(self._ensure_collection)
            if not self._collection_ready:
                return

            normalized = _normalize(question)
            embedding = await asyncio.to_thread(_cached_embed_query, normalized)
            client = get_qdrant_client()
            await asyncio.to_thread(
                client.upsert,
                collection_name=_COLLECTION,
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding,
                        payload={
                            "question": normalized,
                            "response": response,
                            "timestamp": time.time(),
                            "context_key": cache_ctx,
                        },
                    )
                ],
            )
            logger.info(
                "SemanticCache: kaydedildi [ctx=%.12s, q=%.60s, resp=%dch]",
                cache_ctx or "none", question, len(response),
            )
        except Exception as exc:
            logger.warning("SemanticCache store hatası: %s", exc)
