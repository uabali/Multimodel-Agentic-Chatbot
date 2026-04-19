"""
Embedding helpers — HuggingFace GPU embeddings as primary backend.

OllamaEmbeddings is removed; all embedding is done locally on GPU via
sentence-transformers / langchain-huggingface.  This is faster than round-
tripping through an Ollama HTTP server and keeps the Docker stack self-contained.
"""

import logging
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings

logger = logging.getLogger(__name__)

# Known embedding model output dimensions — avoids an extra inference probe call.
_KNOWN_DIMS: dict[str, int] = {
    "bge-m3": 1024,
    "bge-large": 1024,
    "bge-base": 768,
    "bge-small": 512,
    "nomic-embed": 768,
    "mxbai-embed-large": 1024,
    "snowflake-arctic-embed": 1024,
    "all-minilm": 384,
    "e5-large": 1024,
    "e5-base": 768,
}


def infer_embedding_dim(model_name: str) -> int | None:
    """Substring match on known model names to resolve output dimension."""
    n = model_name.lower()
    for key, dim in _KNOWN_DIMS.items():
        if key in n:
            return dim
    return None


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Singleton HuggingFace embedding model.

    lru_cache ensures the model weights are loaded into GPU memory only once
    per process — important because sentence-transformers models are large and
    slow to initialise.
    """
    logger.info(
        "Loading embedding model %r on device %r",
        settings.embedding_model,
        settings.embedding_device,
    )
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_embedding_dim() -> int:
    """
    Resolve embedding vector width:
      1. Explicit env EMBEDDING_VECTOR_SIZE
      2. Known model name substring table
      3. Runtime probe (slow, last resort)
    """
    if settings.embedding_vector_size is not None:
        return settings.embedding_vector_size

    inferred = infer_embedding_dim(settings.embedding_model)
    if inferred is not None:
        logger.info(
            "Embedding dim inferred as %d for model %r",
            inferred,
            settings.embedding_model,
        )
        return inferred

    logger.warning(
        "Unknown embedding model %r — probing dimension via dummy encode.",
        settings.embedding_model,
    )
    vec = get_embeddings().embed_query("probe")
    dim = len(vec)
    logger.info("Probed embedding dim: %d", dim)
    return dim
