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
