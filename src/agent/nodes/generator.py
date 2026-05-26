"""
Generator node module.

Assembles bounded RAG context, handles prompt templates, triggers factual LLM generation,
and runs multi-level compact fallbacks if primary generation is empty or truncated.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agent.routing import current_date_context, is_turkish_query
from src.agent.prompts import (
    RAG_WITH_CONTEXT_SYSTEM_PROMPT,
    RAG_MEMORY_PREFERENCES_BLOCK,
    WEB_WITH_CONTEXT_SYSTEM_PROMPT,
    RAG_NO_CONTEXT_SYSTEM_PROMPT,
)
from src.agent.state import AgentState
from src.config import settings
from src.agent.nodes.base import get_rag_llm, observe_node, coerce_llm_text, select_recent_history
from src.rag.llm import count_message_tokens, count_tokens

logger = logging.getLogger(__name__)


@dataclass
class RAGContextAssembly:
    system_content: str
    context: str
    used_chunk_ids: list[str] = field(default_factory=list)
    docs_included: int = 0
    input_budget_tokens: int = 0
    budget_chars: int = 0
    used_chars: int = 0
    overhead_tokens: int = 0
    truncated: bool = False
    max_input_chars: int = 0


def _fallback_context_answer(question: str, documents: list[Document], vision_context: str = "") -> str:
    """Fallback to direct snippet if LLM output is entirely empty."""
    if vision_context.strip():
        return f"Görsel analizden bulunan bilgi:\n\n{vision_context.strip()[:600]}"
    if documents:
        meta = getattr(documents[0], "metadata", {}) or {}
        src = meta.get("display_name") or meta.get("source_file", "belge")
        body = (documents[0].page_content or "").strip()[:600]
        return f"Belgeden ({src}) ilgili bölüm:\n\n{body}"
    return "Bu soruyu yanıtlayabilecek bir belge bağlamı bulunamadı."


def _estimate_history_tokens(messages: list) -> int:
    return count_message_tokens(messages)


def _source_header(index: int, doc: Document) -> str:
    meta = getattr(doc, "metadata", {}) or {}
    src = meta.get("display_name") or meta.get("source_file", meta.get("source", ""))
    page = meta.get("page", "")
    if meta.get("type") == "web_search":
        url = meta.get("url") or meta.get("source", "")
        published = meta.get("published", "")
        header = f"[Kaynak {index}: {src or url}, Web"
        if published:
            header += f", Tarih: {published}"
        if url:
            header += f", URL: {url}"
        header += "]"
        return header
    return f"[Kaynak {index}: {src}" + (f", Sayfa {page}" if page and str(page) not in {"", "?"} else "") + "]"


def _context_signature(text: str) -> str:
    normalized = re.sub(r"\s+", " ", (text or "").lower()).strip()
    return normalized[:700]


def _context_overlap_score(question: str, content: str) -> float:
    from src.rag.retriever import _tokenize_for_overlap

    terms = _tokenize_for_overlap(question)
    if not terms:
        return 0.0
    normalized_content = " ".join(_tokenize_for_overlap(content))
    if not normalized_content:
        return 0.0
    hits = sum(1 for term in terms if term in normalized_content)
    return hits / max(len(terms), 1)


def _prepare_context_documents(documents: list[Document], question: str) -> list[Document]:
    """Filter out repeating or highly irrelevant chunks before context injection."""
    if not documents:
        return []

    unique_docs: list[Document] = []
    seen: set[str] = set()
    for doc in documents:
        content = (doc.page_content or "").strip()
        if len(content) < 40:
            continue
        signature = _context_signature(content)
        if signature in seen:
            continue
        seen.add(signature)
        unique_docs.append(doc)

    if len(unique_docs) <= 2:
        return unique_docs

    scored = [(doc, _context_overlap_score(question, doc.page_content or "")) for doc in unique_docs]
    has_grounded_matches = sum(1 for _, score in scored if score > 0.0) >= 2
    if not has_grounded_matches:
        return unique_docs

    kept: list[Document] = []
    for doc, score in scored:
        meta = getattr(doc, "metadata", {}) or {}
        if score > 0.0 or meta.get("type") == "web_search":
            kept.append(doc)
    return kept or unique_docs[:2]


def assemble_rag_context(
    *,
    documents: list[Document],
    vision_context: str,
    rag_history: list,
    answer_question: str,
    retrieval_trace: list[dict],
    output_tokens: int,
    memory_preferences: str = "",
) -> RAGContextAssembly:
    """Build bounded prompt context and flag trace elements active in RAG prompt."""
    context_parts: list[str] = []
    used_chunk_ids: list[str] = []
    documents = _prepare_context_documents(documents, answer_question)

    if vision_context:
        context_parts.append(f"[Görsel Analizi]\n{vision_context}")

    n_ctx = settings.llm_context_size
    history_tokens = _estimate_history_tokens(rag_history)
    overhead_tokens = settings.rag_context_safety_margin_tokens + history_tokens
    input_budget_tokens = max(256, n_ctx - output_tokens - overhead_tokens)
    budget_chars = int(input_budget_tokens * 2.5)
    used_chars = sum(len(p) for p in context_parts)

    from src.rag.retriever import chunk_id as _chunk_id

    for i, doc in enumerate(documents, 1):
        meta = getattr(doc, "metadata", {}) or {}
        header = _source_header(i, doc)
        remaining = budget_chars - used_chars
        if remaining <= len(header) + 50:
            break
        max_chars = min(2500 if meta.get("type") == "web_search" else 2000, remaining - len(header) - 10)
        content = (doc.page_content or "")[:max_chars]
        if not content.strip():
            continue
        candidate = f"{header}\n{content}"
        candidate_context = "\n\n---\n\n".join([*context_parts, candidate])
        while (
            count_tokens(candidate_context) + history_tokens + output_tokens + settings.rag_context_safety_margin_tokens > n_ctx
            and len(content) > 200
        ):
            content = content[: int(len(content) * 0.75)].rstrip()
            candidate = f"{header}\n{content}"
            candidate_context = "\n\n---\n\n".join([*context_parts, candidate])
        if count_tokens(candidate_context) + history_tokens + output_tokens + settings.rag_context_safety_margin_tokens > n_ctx:
            if context_parts:
                break
            content = content[:200].rstrip()
            candidate = f"{header}\n{content}"
        context_parts.append(candidate)
        used_chars += len(header) + len(content) + 10
        used_chunk_ids.append(_chunk_id(doc))

    docs_included = len(context_parts) - (1 if vision_context else 0)
    context = "\n\n---\n\n".join(context_parts)
    only_web_context = bool(documents) and not vision_context and all(
        (getattr(doc, "metadata", {}) or {}).get("type") == "web_search"
        for doc in documents
    )
    prompt_template = WEB_WITH_CONTEXT_SYSTEM_PROMPT if only_web_context else RAG_WITH_CONTEXT_SYSTEM_PROMPT
    system_content = prompt_template.replace("{context}", context)
    if only_web_context:
        system_content = system_content.replace(
            "YANIT KURALLARI:",
            f"{current_date_context()}\n"
            "Göreli tarihleri bu tarihe göre çöz; kullanıcı 'yarın' derse mutlak tarihi cevapta belirt.\n"
            "Namaz vakitlerinde Diyanet/resmi kaynakları öncele; resmi kaynak yoksa bunu açıkça belirt.\n\n"
            "YANIT KURALLARI:",
        )
    prefs = (memory_preferences or "").strip()
    if prefs and not only_web_context:
        system_content += RAG_MEMORY_PREFERENCES_BLOCK.replace("{memory_preferences}", prefs)

    prior_chars = sum(len(getattr(m, "content", "") or "") for m in rag_history)
    total_prompt_chars = len(system_content) + prior_chars + len(answer_question)
    max_input_chars = int(max(256, n_ctx - output_tokens - 50) * 2.5)
    truncated = False
    if total_prompt_chars > max_input_chars:
        safe_ctx_len = max(500, max_input_chars - prior_chars - len(answer_question) - 800)
        context = context[:safe_ctx_len]
        system_content = prompt_template.replace("{context}", context)
        if only_web_context:
            system_content = system_content.replace(
                "YANIT KURALLARI:",
                f"{current_date_context()}\n"
                "Göreli tarihleri bu tarihe göre çöz; kullanıcı 'yarın' derse mutlak tarihi cevapta belirt.\n"
                "Namaz vakitlerinde Diyanet/resmi kaynakları öncele; resmi kaynak yoksa bunu açıkça belirt.\n\n"
                "YANIT KURALLARI:",
            )
        used_chars = min(used_chars, len(context))
        truncated = True

    used_set = set(used_chunk_ids)
    for entry in retrieval_trace:
        if entry.get("chunk_id") in used_set:
            entry["used_in_context"] = True

    return RAGContextAssembly(
        system_content=system_content,
        context=context,
        used_chunk_ids=used_chunk_ids,
        docs_included=docs_included,
        input_budget_tokens=input_budget_tokens,
        budget_chars=budget_chars,
        used_chars=used_chars,
        overhead_tokens=overhead_tokens,
        truncated=truncated,
        max_input_chars=max_input_chars,
    )


def _source_list_line(index: int, doc: Document) -> str:
    meta = getattr(doc, "metadata", {}) or {}
    title = meta.get("display_name") or meta.get("title") or meta.get("source_file") or meta.get("source") or f"Kaynak {index}"
    url = meta.get("url") or (meta.get("source") if meta.get("type") == "web_search" else "")
    published = f" — {meta.get('published')}" if meta.get("published") else ""
    page = meta.get("page")
    page_txt = f", s. {page}" if page and str(page) not in {"", "?"} else ""
    if url:
        return f"- [{index}] [{title}]({url}){published}"
    return f"- [{index}] {title}{page_txt}"


def append_used_sources(answer: str, documents: list[Document], question: str) -> str:
    """Format and append bracket-cited sources to the bottom of the LLM response."""
    if not answer.strip() or not documents:
        return answer.strip()
    if re.search(r"(?im)^\s*(kaynaklar|sources)\s*:", answer):
        return answer.strip()
    cited: list[int] = []
    for raw in re.findall(r"\[(?:Kaynak\s*)?(\d+)\]", answer, re.IGNORECASE):
        try:
            idx = int(raw)
        except ValueError:
            continue
        if 1 <= idx <= len(documents) and idx not in cited:
            cited.append(idx)
    if not cited:
        return answer.strip()
    header = "Kaynaklar:" if is_turkish_query(question) else "Sources:"
    lines = [header] + [_source_list_line(idx, documents[idx - 1]) for idx in cited]
    return f"{answer.strip()}\n\n" + "\n".join(lines)


def _final_answer_fields(
    state: AgentState,
    generation: str,
    *,
    t0: float,
    mode: str,
    extra_latency: dict | None = None,
) -> dict:
    from src.observability.langsmith import safe_preview

    latency = {"total": round((time.perf_counter() - t0) * 1000, 2), **(extra_latency or {})}
    answer_preview = safe_preview(generation)
    observe_node(
        f"frappe.{mode}_result",
        state,
        outputs={
            "answer_preview": answer_preview,
            "answer_chars": len(generation),
            "route": state.get("route", mode),
            "retry_path": "primary",
            "latency_ms_by_stage": latency,
        },
        metadata={"answer_chars": len(generation), "response_mode": mode},
        tags=["frappe", mode, state.get("route", mode) or mode],
    )
    return {
        "answer_preview": answer_preview,
        "answer_chars": len(generation),
        "document_count": len(state.get("documents") or []),
        "used_context_count": 0,
        "document_previews": [],
        "retrieval_trace_summary": {},
        "top_sources": "",
        "top_chunks": "",
        "used_chunks": "",
        "retry_summary": {"retry_path": "primary"},
        "retry_path": "primary",
        "latency_ms_by_stage": latency,
    }


async def _retry_generator_with_compact_context(
    question: str,
    documents: list[Document],
    prior_messages: list,
    vision_context: str,
) -> str:
    """First-level RAG generator fallback utilizing compact query formatting."""
    compact_parts: list[str] = []
    if vision_context.strip():
        compact_parts.append(f"[Görsel Analizi]\n{vision_context.strip()[:1200]}")

    for i, doc in enumerate(documents[:4], 1):
        meta = getattr(doc, "metadata", {}) or {}
        src = meta.get("display_name") or meta.get("source_file", meta.get("source", ""))
        page = meta.get("page", "")
        if meta.get("type") == "web_search":
            url = meta.get("url") or meta.get("source", "")
            published = meta.get("published", "")
            header = f"[Kaynak {i}: {src or url}, Web"
            if published:
                header += f", Tarih: {published}"
            if url:
                header += f", URL: {url}"
            header += "]"
        else:
            header = f"[Kaynak {i}: {src}" + (f", Sayfa {page}" if page and str(page) not in {"", "?"} else "") + "]"
        content = (doc.page_content or "").strip()
        if content:
            compact_parts.append(f"{header}\n{content[:400]}")

    if not compact_parts:
        return ""

    compact_context = "\n\n---\n\n".join(compact_parts)
    only_web_context = bool(documents) and all(
        (getattr(doc, "metadata", {}) or {}).get("type") == "web_search"
        for doc in documents
    )
    missing_answer = (
        "Web araması bu soruyu yanıtlayacak güvenilir ve doğrudan eşleşen bilgi bulamadı."
        if only_web_context
        else "Bu bilgi yüklenen belgelerde yer almamaktadır."
    )
    source_label = "web kaynakları" if only_web_context else "bağlam"
    system_content = (
        f"Adın Frappe, bir RAG asistanısın. Sadece verilen {source_label}na dayanarak "
        "kullanıcının sorusunu aynı dilde, kısa ve doğrudan yanıtla. "
        f"Cevap yoksa sadece '{missing_answer}' yaz.\n\n"
        f"Bağlam:\n{compact_context}"
    )
    llm = get_rag_llm(temperature=0.0, max_tokens=min(settings.rag_max_tokens, 512))
    response = await llm.ainvoke([
        SystemMessage(content=system_content),
        *select_recent_history(list(prior_messages), mode="rag"),
        HumanMessage(content=question),
    ])
    return coerce_llm_text(response)


async def _micro_answer_retry(question: str, documents: list[Document]) -> str:
    """Second-level micro RAG query retry utilizing minimal document snippets."""
    if not documents:
        return ""
    chunks = []
    for doc in documents[:3]:
        chunk = (doc.page_content or "").strip()[:300]
        if chunk:
            chunks.append(chunk)
    if not chunks:
        return ""
    combined = "\n---\n".join(chunks)
    llm = get_rag_llm(temperature=0.0, max_tokens=128)
    response = await llm.ainvoke([
        SystemMessage(content="Bu metinden soruya tek cümleyle yanıt ver. Cevap yoksa 'Bilgi bulunamadı.' yaz."),
        HumanMessage(content=f"Soru: {question}\n\nMetin: {combined}"),
    ])
    return coerce_llm_text(response)


async def generator_node(state: AgentState) -> AgentState:
    """Assemble RAG document context, verify token lengths, generate responses, and trigger fallback layers on empty results."""
    t0 = time.perf_counter()
    question = state["question"]
    documents = state.get("documents", [])
    vision_context = state.get("vision_context", "")
    prior_messages = list(state.get("messages", []))
    trace = list(state.get("retrieval_trace") or [])

    # Check if the döküments are graded as completely irrelevant by the grader (CRAG alignment)
    relevance = state.get("relevance", "")
    grader_reason = state.get("grader_reason", "")
    has_web_results = any(
        (getattr(doc, "metadata", {}) or {}).get("type") == "web_search"
        for doc in documents
    )

    if relevance == "no" and documents and not has_web_results:
        if grader_reason in {"irrelevant", "insufficient_context"}:
            is_tr = is_turkish_query(question)
            generation = (
                "Bu bilgi yüklenen belgelerde yer almamaktadır."
                if is_tr
                else "This information is not present in the uploaded documents."
            )
            logger.info("Grader flagged documents as irrelevant and no web search was executed. Direct refusal generated.")
            fields = _final_answer_fields(
                state,
                generation,
                t0=t0,
                mode="generator",
                extra_latency={"llm": 0.0},
            )
            return {**state, "generation": generation, **fields}

    # Route A: Bind RAG prompts if documents or visual extractions exist
    if documents or vision_context.strip():
        output_tokens = state.get("max_tokens") or settings.rag_max_tokens
        rag_history = select_recent_history(prior_messages, mode="rag")

        assembly = assemble_rag_context(
            documents=documents,
            vision_context=vision_context,
            rag_history=rag_history,
            answer_question=question,
            retrieval_trace=trace,
            output_tokens=output_tokens,
            memory_preferences=state.get("memory_context") or "",
        )

        llm = get_rag_llm(
            temperature=state.get("temperature") or settings.rag_temperature,
            max_tokens=output_tokens,
        )

        t_llm = time.perf_counter()
        try:
            response = await llm.ainvoke([
                SystemMessage(content=assembly.system_content),
                *rag_history,
                HumanMessage(content=question),
            ])
            generation = coerce_llm_text(response)
            llm_ms = round((time.perf_counter() - t_llm) * 1000, 2)
            retry_path = "primary"
        except Exception as exc:
            logger.warning("RAG generator failed: %s — invoking compact fallback", exc)
            generation = ""
            llm_ms = None
            retry_path = "error_fallback"

        # Level 1 Fallback: Invoke compact retry if generated text is empty
        if not generation.strip():
            logger.info("Empty generator output; executing compact RAG retry.")
            t_retry1 = time.perf_counter()
            try:
                generation = await _retry_generator_with_compact_context(
                    question, documents, prior_messages, vision_context
                )
                llm_ms = round((time.perf_counter() - t_retry1) * 1000, 2)
                retry_path = "compact_retry"
            except Exception as e:
                logger.warning("Compact RAG retry failed: %s", e)
                generation = ""

        # Level 2 Fallback: Invoke micro snippet fallback if output remains empty
        if not generation.strip():
            logger.info("Empty compact output; executing micro-RAG retry.")
            t_retry2 = time.perf_counter()
            try:
                generation = await _micro_answer_retry(question, documents)
                llm_ms = round((time.perf_counter() - t_retry2) * 1000, 2)
                retry_path = "micro_retry"
            except Exception as e:
                logger.warning("Micro-RAG retry failed: %s", e)
                generation = ""

        # Level 3 Fallback: Hard fallback from document snippets
        if not generation.strip():
            logger.warning("All generators returned empty; reverting to hard snippet copy.")
            generation = _fallback_context_answer(question, documents, vision_context)
            retry_path = "hard_fallback"

        generation = append_used_sources(generation, documents, question)

        from src.observability.langsmith import (
            summarize_documents,
            summarize_retrieval_trace,
            summarize_source_distribution,
        )

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        latency = {
            "llm": llm_ms,
            "total": elapsed_ms,
        }

        trace_summary = summarize_retrieval_trace(trace)
        observe_node(
            "frappe.generator_result",
            state,
            outputs={
                "generation": generation,
                "status": "success",
                "used_chunks": trace_summary.get("used_chunks", ""),
                "retry_path": retry_path,
                "latency_ms_by_stage": latency,
            },
            metadata={
                "rag_prompt_version": settings.rag_prompt_version,
                "docs_included": assembly.docs_included,
                "rag_input_budget_chars": assembly.budget_chars,
                "rag_used_chars": assembly.used_chars,
                "rag_history_tokens": assembly.overhead_tokens - settings.rag_context_safety_margin_tokens,
                "rag_overhead_tokens": assembly.overhead_tokens,
                "rag_truncated": assembly.truncated,
                "response_mode": "rag",
                "retry_path": retry_path,
            },
            tags=["frappe", "generator", "success", f"retry:{retry_path}"],
        )

        return {
            **state,
            "generation": generation,
            "retrieval_trace": trace,
            "answer_preview": generation[:180],
            "answer_chars": len(generation),
            "document_count": len(documents),
            "used_context_count": assembly.docs_included,
            "document_previews": summarize_documents(documents),
            "retrieval_trace_summary": trace_summary,
            "top_sources": summarize_source_distribution(documents),
            "top_chunks": trace_summary.get("top_chunks", ""),
            "used_chunks": trace_summary.get("used_chunks", ""),
            "retry_summary": {
                "retry_path": retry_path,
                "docs_included": assembly.docs_included,
                "rag_truncated": assembly.truncated,
            },
            "retry_path": retry_path,
            "latency_ms_by_stage": latency,
        }

    # Route B: Bind fallback prompts if no documents are retrieved
    logger.debug("Generator: no context available -> running no-context RAG prompt")
    llm = get_rag_llm(
        temperature=state.get("temperature") or settings.rag_temperature,
        max_tokens=state.get("max_tokens") or settings.rag_max_tokens,
    )
    t_llm = time.perf_counter()
    try:
        response = await llm.ainvoke([
            SystemMessage(content=RAG_NO_CONTEXT_SYSTEM_PROMPT),
            *select_recent_history(prior_messages, mode="rag"),
            HumanMessage(content=question),
        ])
        generation = coerce_llm_text(response)
    except Exception as exc:
        logger.error("No-context RAG LLM invocation failed: %s", exc)
        generation = "Soruyu yanıtlayacak herhangi bir belge veya canlı veri bulunamadı."

    fields = _final_answer_fields(
        state,
        generation,
        t0=t0,
        mode="generator",
        extra_latency={"llm": round((time.perf_counter() - t_llm) * 1000, 2)},
    )
    return {**state, "generation": generation, **fields}
