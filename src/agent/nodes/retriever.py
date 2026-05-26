"""
Retriever node module.

Handles Qdrant-based hybrid vector retrieval, dense gating, reranking,
and document-overview contextual expansions.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from typing import Any

from langchain_core.documents import Document

from src.agent.state import AgentState
from src.config import settings
from src.agent.nodes.base import RerankerRegistry, observe_node, detailed_trace_enabled

logger = logging.getLogger(__name__)

_DOCUMENT_OVERVIEW_RE = re.compile(
    r"\b("
    r"ana\s+konu|konusu|önemli\s+bulgu\w*|onemli\s+bulgu\w*|bulgu\w*|"
    r"yöntem|yontem|metod|method|methodology|"
    r"özet|ozet|summarize|summary|abstract|"
    r"sonuç|sonuc|conclusion|giriş|giris|introduction"
    r")\b",
    re.IGNORECASE,
)

_OVERVIEW_SECTION_RE = re.compile(
    r"\b("
    r"abstract|özet|ozet|giriş|giris|introduction|"
    r"yöntem|yontem|methodology|method|"
    r"bulgular|findings|sonuç|sonuc|conclusion|"
    r"tartışma|tartisma|discussion"
    r")\b",
    re.IGNORECASE,
)


def _build_source_filter(
    source_filter: str,
    session_uploads: list[str] | None = None,
    *,
    user_id: str = "",
    thread_id: str = "",
) -> Any:
    """Build Qdrant metadata filters from current source filter or session uploads."""
    from qdrant_client import models as qmodels

    must: list[Any] = []
    
    # 1. User Isolation Gate
    if settings.qdrant_tenant_filter_enabled and user_id.strip():
        uid = user_id.strip()
        should = [
            qmodels.FieldCondition(key="metadata.user_id", match=qmodels.MatchValue(value=uid))
        ]
        if settings.qdrant_include_shared_corpus:
            should.append(qmodels.IsEmptyCondition(is_empty=qmodels.PayloadField(key="metadata.user_id")))
        must.append(qmodels.Filter(should=should))

    # 2. File Ingestion Scope
    if source_filter.strip():
        must.append(
            qmodels.FieldCondition(
                key="metadata.source_file",
                match=qmodels.MatchValue(value=source_filter.strip()),
            )
        )
        return qmodels.Filter(must=must)

    if session_uploads:
        uploads = [str(x) for x in session_uploads if str(x).strip()]
        if uploads:
            must.append(
                qmodels.FieldCondition(
                    key="metadata.source_file",
                    match=qmodels.MatchAny(any=uploads),
                )
            )
            return qmodels.Filter(must=must)
            
    return qmodels.Filter(must=must) if must else None


def _is_document_overview_question(state: AgentState) -> bool:
    """Detect whether user wants a bird's-eye summary or overview of the file."""
    q = " ".join(
        str(part or "")
        for part in (state.get("original_question"), state.get("question"))
    )
    return bool(_DOCUMENT_OVERVIEW_RE.search(q))


def _payload_to_document(payload: dict | None) -> Document | None:
    """Convert raw Qdrant point payload into a standard Document."""
    payload = payload or {}
    content = (
        payload.get("page_content")
        or payload.get("content")
        or payload.get("text")
        or payload.get("document")
        or ""
    )
    if not isinstance(content, str) or not content.strip():
        return None
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    extra_meta = {
        k: v for k, v in payload.items()
        if k not in {"page_content", "content", "text", "document", "metadata"}
    }
    return Document(page_content=content, metadata={**extra_meta, **metadata})


def _chunk_sort_key(doc: Document) -> tuple[int, int, str]:
    """Helper to sort document chunks in their logical/physical order."""
    from src.rag.retriever import chunk_id

    meta = getattr(doc, "metadata", {}) or {}
    raw_idx = meta.get("chunk_index")
    idx = 10**9
    if isinstance(raw_idx, int):
        idx = raw_idx
    elif isinstance(raw_idx, str):
        m = re.search(r"\d+", raw_idx)
        if m:
            idx = int(m.group(0))
    try:
        page = int(meta.get("page") or 10**9)
    except (TypeError, ValueError):
        page = 10**9
    return idx, page, chunk_id(doc)


def _fetch_document_overview_chunks(store: Any, qdrant_filter: Any, *, limit: int = 96, max_docs: int = 4) -> list[Document]:
    """Scroll Qdrant for structural opening, concluding, and section header chunks."""
    if qdrant_filter is None:
        return []
    try:
        records, _ = store.client.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=qdrant_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as exc:
        logger.debug("overview chunk fetch failed: %s", exc)
        return []

    docs = [doc for rec in records if (doc := _payload_to_document(getattr(rec, "payload", None)))]
    if not docs:
        return []
    docs.sort(key=_chunk_sort_key)

    selected: list[Document] = []
    selected.extend(docs[:2])
    selected.extend(doc for doc in docs if _OVERVIEW_SECTION_RE.search(doc.page_content or ""))
    if len(docs) > 2:
        selected.extend(docs[-2:])

    from src.rag.retriever import deduplicate_documents

    return deduplicate_documents(selected, max_docs=max_docs)


def _fmt_score(s: Any) -> str:
    """Format scores into floating point strings."""
    return f"{s:.3f}" if isinstance(s, (int, float)) else "?"


def _score_lookup_with_filter(vectorstore: Any, query: str, k: int, qdrant_filter: Any) -> list:
    """Wrapper to search with similarity score incorporating filters safely."""
    kwargs = {}
    if qdrant_filter is not None:
        kwargs["filter"] = qdrant_filter
    try:
        return vectorstore.similarity_search_with_score(query, k=k, **kwargs)
    except TypeError:
        return vectorstore.similarity_search_with_score(query, k=k)


def _retriever_score_lookup_enabled() -> bool:
    """Determine whether to run expensive detailed trace score lookups."""
    override = settings.retriever_score_lookup
    if override is not None:
        return bool(override)
    return detailed_trace_enabled(logger)


async def retriever_node(state: AgentState) -> AgentState:
    """Execute hybrid dense+sparse vector search, dense similarity threshold gating, and reranking."""
    t0 = time.perf_counter()
    question = state["question"]
    source_filter = state.get("source_filter", "")
    session_uploads = state.get("session_uploads") or []
    latency_ms: dict[str, float] = {}
    dense_score = None
    retrieval_gate = "skip"
    strategy = state.get("retrieval_strategy") or settings.retrieval_strategy
    use_rerank_val = state.get("use_rerank")
    if use_rerank_val is None:
        use_rerank_val = settings.use_rerank

    try:
        from src.rag.vectorstore import get_hybrid_store
        from src.rag.retriever import create_retriever, deduplicate_documents, run_retriever, chunk_id

        store = await asyncio.to_thread(get_hybrid_store)
        qdrant_filter = _build_source_filter(
            source_filter,
            session_uploads,
            user_id=state.get("user_id") or "",
            thread_id=state.get("thread_id") or "",
        )

        # Skip dense gate similarity matching if explicitly querying uploaded files
        if source_filter or session_uploads:
            dense_score = 1.0
            retrieval_gate = "skip"
            filter_desc = f"source_filter='{source_filter}'" if source_filter else f"uploads={session_uploads}"
            logger.debug("Retriever: dense_gate=skip [%s]", filter_desc)
        else:
            t_gate = time.perf_counter()
            try:
                dense_score = await asyncio.to_thread(
                    store.max_dense_similarity, question, qdrant_filter=qdrant_filter
                )
            except Exception as exc:
                logger.warning("Dense gate failed: %s — skipping gate", exc)
                dense_score = settings.rag_min_dense_similarity
            latency_ms["dense_gate"] = round((time.perf_counter() - t_gate) * 1000, 2)
            
            if dense_score >= settings.rag_dense_pass_similarity:
                retrieval_gate = "pass"
            elif dense_score >= settings.rag_min_dense_similarity:
                retrieval_gate = "soft"
            else:
                retrieval_gate = "weak"
            logger.debug("Retriever: dense_gate=%s [score=%.3f]", retrieval_gate, dense_score)

        retriever = create_retriever(
            vectorstore=store.store,
            question=question,
            strategy=strategy,
            base_k=settings.base_k,
            max_k=settings.top_k,
            fetch_k=settings.fetch_k,
            lambda_mult=settings.lambda_mult,
            score_threshold=settings.score_threshold,
            use_rerank=use_rerank_val,
            reranker=RerankerRegistry.get(),
            rerank_top_n=settings.rerank_top_n,
            qdrant_filter=qdrant_filter,
        )
        
        t_fetch = time.perf_counter()
        documents = await asyncio.to_thread(run_retriever, retriever, question)
        documents = deduplicate_documents(documents, max_docs=settings.top_k)

        # Overview Boost: Fetch structural segments if summarizing or querying overview
        if _is_document_overview_question(state):
            t_overview = time.perf_counter()
            overview_docs = await asyncio.to_thread(
                _fetch_document_overview_chunks,
                store,
                qdrant_filter,
                max_docs=4,
            )
            if overview_docs:
                documents = deduplicate_documents(
                    [*overview_docs, *documents],
                    max_docs=max(settings.top_k, min(settings.rerank_top_n, settings.top_k + len(overview_docs))),
                )
            latency_ms["overview_fetch"] = round((time.perf_counter() - t_overview) * 1000, 2)
            logger.debug(
                "Retriever: overview_boost [overview_docs=%d, final_docs=%d, t=%.3fs]",
                len(overview_docs), len(documents), time.perf_counter() - t_overview,
            )
            
        t_fetch_elapsed = time.perf_counter() - t_fetch
        latency_ms["fetch"] = round(t_fetch_elapsed * 1000, 2)

        # Optional Detailed Explainability trace lookup
        hybrid_scores: dict[str, float] = {}
        if _retriever_score_lookup_enabled():
            try:
                search_k = max(settings.rerank_top_n, settings.top_k * 2)
                t_score = time.perf_counter()
                scored_pairs = await asyncio.to_thread(
                    _score_lookup_with_filter, store.store, question, search_k, qdrant_filter,
                )
                latency_ms["score_lookup"] = round((time.perf_counter() - t_score) * 1000, 2)
                for d, s in scored_pairs:
                    hybrid_scores[chunk_id(d)] = float(s)
            except Exception as exc:
                logger.debug("hybrid score lookup failed: %s", exc)

        # Record per-chunk retrieval details for LangSmith logs
        retrieval_trace: list[dict] = []
        for doc in documents:
            cid = chunk_id(doc)
            h_score = hybrid_scores.get(cid)
            if h_score is not None:
                doc.metadata["retrieval_score"] = h_score
            retrieval_trace.append({
                "chunk_id": cid,
                "hybrid_score": h_score,
                "rerank_score": doc.metadata.get("rerank_score"),
                "used_in_context": False,
            })

        logger.debug(
            "Retriever: docs=%d [strategy=%s, rerank=%s, dense=%.3f, fetch_t=%.3fs, total_t=%.3fs]",
            len(documents), strategy, use_rerank_val, dense_score,
            t_fetch_elapsed, time.perf_counter() - t0,
        )
        if retrieval_trace and detailed_trace_enabled(logger):
            trace_parts = [
                f"{t['chunk_id']} hybrid={_fmt_score(t['hybrid_score'])} rerank={_fmt_score(t['rerank_score'])}"
                for t in retrieval_trace
            ]
            logger.debug("Retriever: trace [%s]", " | ".join(trace_parts))
            
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        latency_ms["total"] = elapsed_ms
        
        from src.observability.langsmith import (
            summarize_documents,
            summarize_retrieval_trace,
            summarize_source_distribution,
        )
        trace_summary = summarize_retrieval_trace(retrieval_trace)
        observe_node(
            "frappe.retriever_result",
            state,
            outputs={
                "status": "success",
                "document_count": len(documents),
                "top_sources": summarize_source_distribution(documents),
                "top_chunks": trace_summary.get("top_chunks", ""),
                "used_chunks": trace_summary.get("used_chunks", ""),
                "retrieval_trace_summary": trace_summary,
                "document_previews": summarize_documents(documents),
                "latency_ms_by_stage": latency_ms,
            },
            metadata={
                "retrieval_strategy": strategy,
                "use_rerank": bool(use_rerank_val),
                "dense_score": dense_score,
                "dense_threshold": settings.rag_min_dense_similarity,
                "dense_pass_threshold": settings.rag_dense_pass_similarity,
                "retrieval_gate": retrieval_gate,
                "top_sources": summarize_source_distribution(documents),
            },
            tags=["frappe", "retriever", "success", f"gate:{retrieval_gate}"],
        )
    except Exception as exc:
        logger.warning("Retriever: error [%s, t=%.3fs]", exc, time.perf_counter() - t0)
        documents = []
        retrieval_trace = []
        observe_node(
            "frappe.retriever_result",
            state,
            outputs={
                "status": "error",
                "document_count": 0,
                "latency_ms_by_stage": {"total": round((time.perf_counter() - t0) * 1000, 2)},
            },
            metadata={
                "retrieval_strategy": strategy,
                "use_rerank": bool(use_rerank_val),
            },
            tags=["frappe", "retriever", "error"],
            error=str(exc),
        )

    return {**state, "documents": documents, "retrieval_trace": retrieval_trace, "retrieval_gate": retrieval_gate}
