from langchain_core.documents import Document
import pytest

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


def test_tiny_rerank_score_does_not_create_high_confidence():
    docs = [
        Document(
            page_content="Bu parça alakasız kısa bir metindir.",
            metadata={"rerank_score": 0.001},
        )
    ]

    assert estimate_confidence("Belgedeki önemli bulguları özetle", docs) < 0.35


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

    assert decision == "refuse"


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


def test_partial_context_with_source_filter_uses_web_search():
    from src.agent.graph import _grader_decision

    decision = _grader_decision({
        "relevance": "no",
        "grader_reason": "partial",
        "source_filter": "uploaded.pdf",
    })

    assert decision == "insufficient"


def test_parse_grader_reason_supports_extended_enum():
    from src.agent.nodes import _parse_grader_payload, _parse_grader_reason

    assert _parse_grader_reason('{"relevant":"yes","reason":"sufficient"}') == "sufficient"
    assert _parse_grader_reason('{"relevant":"no","reason":"partial"}') == "partial"
    assert _parse_grader_reason('{"relevant":"no","reason":"insufficient_context"}') == "insufficient_context"
    assert _parse_grader_reason('{"relevant":"no","reason":"unknown"}') == "insufficient_context"
    assert _parse_grader_payload('{"relevant":"no","reason":"needs_live_data"}') == ("no", "needs_live_data")
    assert _parse_grader_payload('relevant no because partial') == ("no", "partial")


@pytest.mark.anyio
async def test_source_filter_low_confidence_uses_llm_grader(monkeypatch):
    from src.agent import nodes

    calls = {"llm": 0}

    class FakeLLM:
        async def ainvoke(self, messages):
            calls["llm"] += 1

            class Response:
                content = '{"relevant":"yes","reason":"sufficient"}'

            return Response()

    monkeypatch.setattr(nodes, "_get_rag_llm", lambda temperature=0.0, max_tokens=None: FakeLLM())
    monkeypatch.setattr(nodes, "_observe_node", lambda *args, **kwargs: None)
    monkeypatch.setattr(nodes.settings, "grader_conf_high", 0.75)

    result = await nodes.grader_node({
        "question": "Belgedeki önemli bulguları özetle",
        "original_question": "Belgedeki önemli bulguları özetle",
        "source_filter": "uploaded.pdf",
        "session_uploads": ["uploaded.pdf"],
        "documents": [
            Document(
                page_content="Bu parça alakasız kısa bir metindir.",
                metadata={"source_file": "uploaded.pdf", "chunk_index": 9, "rerank_score": 0.001},
            )
        ],
    })

    assert calls["llm"] == 1
    assert result["relevance"] == "yes"


def test_document_overview_fetch_adds_opening_and_section_chunks():
    from types import SimpleNamespace
    from src.agent.nodes import _fetch_document_overview_chunks

    payloads = [
        {"page_content": "Sonuç bölümünde sistemin başarımı tartışılır.", "metadata": {"source_file": "a.pdf", "chunk_index": 5}},
        {"page_content": "Kapak ve proje adı.", "metadata": {"source_file": "a.pdf", "chunk_index": 0}},
        {"page_content": "Yöntem bölümünde hibrit RAG mimarisi anlatılır.", "metadata": {"source_file": "a.pdf", "chunk_index": 3}},
        {"page_content": "Özet bölümünde belgenin ana konusu verilir.", "metadata": {"source_file": "a.pdf", "chunk_index": 1}},
    ]

    class FakeClient:
        def scroll(self, **kwargs):
            return [SimpleNamespace(payload=p) for p in payloads], None

    class FakeStore:
        client = FakeClient()

    docs = _fetch_document_overview_chunks(FakeStore(), object(), max_docs=4)

    assert [doc.metadata["chunk_index"] for doc in docs] == [0, 1, 3, 5]


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
