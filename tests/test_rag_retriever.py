from langchain_core.documents import Document

from src.rag.retriever import (
    auto_select_strategy,
    deduplicate_documents,
    estimate_confidence,
    normalize_query_text,
)


def test_normalize_query_text_handles_turkish_letters():
    assert normalize_query_text("Uçuş Bileti Şehirleri") == "ucus bileti sehirleri"


def test_auto_strategy_handles_ticket_route_queries():
    assert auto_select_strategy("Uçuş bileti nereden nereye?") == "hybrid"


def test_confidence_uses_turkish_normalization():
    docs = [Document(page_content="Ucus bileti kalkis Istanbul ve varis Ankara olarak gorunuyor.")]

    assert estimate_confidence("Uçuş biletinin kalkış ve varış bilgisi nedir?", docs) > 0


def test_deduplicate_documents_removes_repeated_chunks():
    docs = [
        Document(
            page_content="same content",
            metadata={"source_file": "a.pdf", "page": 1, "chunk_index": 0},
        ),
        Document(
            page_content="same content",
            metadata={"source_file": "a.pdf", "page": 1, "chunk_index": 0},
        ),
        Document(
            page_content="other content",
            metadata={"source_file": "a.pdf", "page": 1, "chunk_index": 1},
        ),
    ]

    result = deduplicate_documents(docs)

    assert len(result) == 2
    assert [doc.page_content for doc in result] == ["same content", "other content"]


def test_source_filter_irrelevant_does_not_trigger_web_search():
    from src.agent.graph import _grader_decision

    decision = _grader_decision({
        "relevance": "no",
        "grader_reason": "irrelevant",
        "source_filter": "uploaded.pdf",
    })

    assert decision == "sufficient"


def test_source_filter_live_data_still_triggers_web_search():
    from src.agent.graph import _grader_decision

    decision = _grader_decision({
        "relevance": "no",
        "grader_reason": "needs_live_data",
        "source_filter": "uploaded.pdf",
    })

    assert decision == "insufficient"


def test_insufficient_context_without_source_filter_refuses_instead_of_web():
    from src.agent.graph import _grader_decision

    decision = _grader_decision({
        "relevance": "no",
        "grader_reason": "insufficient_context",
        "source_filter": "",
    })

    assert decision == "refuse"


def test_partial_context_without_source_filter_uses_web_search():
    from src.agent.graph import _grader_decision

    decision = _grader_decision({
        "relevance": "no",
        "grader_reason": "partial",
        "source_filter": "",
    })

    assert decision == "insufficient"


def test_parse_grader_reason_supports_extended_enum():
    from src.agent.nodes import _parse_grader_reason

    assert _parse_grader_reason('{"relevant":"yes","reason":"sufficient"}') == "sufficient"
    assert _parse_grader_reason('{"relevant":"no","reason":"partial"}') == "partial"
    assert _parse_grader_reason('{"relevant":"no","reason":"insufficient_context"}') == "insufficient_context"


def test_rag_context_assembly_truncates_for_small_context(monkeypatch):
    from langchain_core.documents import Document
    from src.agent.nodes import assemble_rag_context
    from src.agent import nodes

    monkeypatch.setattr(nodes.settings, "llm_context_size", 1200)
    monkeypatch.setattr(nodes.settings, "rag_context_safety_margin_tokens", 500)

    docs = [
        Document(
            page_content="çok uzun içerik " * 600,
            metadata={"source_file": "a.pdf", "chunk_index": 0},
        )
    ]
    trace = [{"chunk_id": "a.pdf#0", "hybrid_score": 0.9, "rerank_score": 0.8, "used_in_context": False}]

    assembly = assemble_rag_context(
        documents=docs,
        vision_context="",
        rag_history=[],
        answer_question="Belgedeki bilgiyi açıkla",
        retrieval_trace=trace,
        output_tokens=512,
    )

    assert assembly.truncated is True
    assert assembly.input_budget_tokens >= 256
    assert trace[0]["used_in_context"] is True
