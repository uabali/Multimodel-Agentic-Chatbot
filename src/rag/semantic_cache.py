"""
Semantic cache — sık sorulan sorular için Qdrant tabanlı yanıt önbelleği.

Akış:
  lookup(question) → embed → Qdrant nearest-neighbor (threshold + TTL) → cached response
  store(question, response) → embed → Qdrant upsert

Mevcut altyapıdan yeniden kullanılanlar:
  - get_qdrant_client() (vectorstore.py)
  - get_embeddings()    (embeddings.py)
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

    @classmethod
    def get(cls) -> "SemanticCache":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_collection(self) -> None:
        if self._collection_ready:
            return
        try:
            from qdrant_client import models
            from src.rag.vectorstore import get_qdrant_client
            from src.rag.embeddings import get_embedding_dim

            client = get_qdrant_client()
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
        except Exception as exc:
            logger.warning("SemanticCache: koleksiyon hazırlama başarısız: %s", exc)

    async def lookup(self, question: str) -> Optional[str]:
        """Benzer soru cache'te varsa yanıtı döner, yoksa None."""
        if not settings.semantic_cache_enabled:
            return None
        try:
            from qdrant_client import models
            from src.rag.vectorstore import get_qdrant_client
            from src.rag.embeddings import get_embeddings

            await asyncio.to_thread(self._ensure_collection)
            if not self._collection_ready:
                return None

            normalized = _normalize(question)
            embedding = await asyncio.to_thread(get_embeddings().embed_query, normalized)

            cutoff = time.time() - settings.semantic_cache_ttl_hours * 3600
            client = get_qdrant_client()
            results = await asyncio.to_thread(
                client.search,
                collection_name=_COLLECTION,
                query_vector=embedding,
                limit=1,
                score_threshold=settings.semantic_cache_threshold,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="timestamp",
                            range=models.Range(gte=cutoff),
                        )
                    ]
                ),
            )
            if results:
                score = results[0].score
                cached = results[0].payload.get("response", "")
                logger.info(
                    "SemanticCache: hit [score=%.3f, threshold=%.3f, q=%.60s]",
                    score, settings.semantic_cache_threshold, question,
                )
                return cached
        except Exception as exc:
            logger.warning("SemanticCache lookup hatası: %s", exc)
        return None

    async def store(self, question: str, response: str) -> None:
        """Soru–yanıt çiftini cache'e kaydeder."""
        if not settings.semantic_cache_enabled:
            return
        try:
            from qdrant_client import models
            from src.rag.vectorstore import get_qdrant_client
            from src.rag.embeddings import get_embeddings

            await asyncio.to_thread(self._ensure_collection)
            if not self._collection_ready:
                return

            normalized = _normalize(question)
            embedding = await asyncio.to_thread(get_embeddings().embed_query, normalized)
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
                        },
                    )
                ],
            )
            logger.info(
                "SemanticCache: kaydedildi [q=%.60s, resp=%dch]",
                question, len(response),
            )
        except Exception as exc:
            logger.warning("SemanticCache store hatası: %s", exc)
