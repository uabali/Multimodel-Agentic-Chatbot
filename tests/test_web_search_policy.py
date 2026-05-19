from langchain_core.messages import AIMessage
import pytest

from src.agent.nodes import (
    _build_contextual_web_query,
    _compact_web_query,
    _docs_from_explicit_urls,
    _web_docs_from_result,
    _web_fallback_answer,
    append_used_sources,
    assemble_rag_context,
)
from src.agent.routing import keyword_route, is_web_query
from src.agent.web_search import WebResultFormatter, WebSearchResult
from src.agent.web_search import WebSearchService


def test_realtime_query_routes_to_web():
    assert keyword_route("bugünkü Tesla hisse fiyatı ne kadar?") == "web"


def test_answer_quality_followups_do_not_route_to_web():
    assert keyword_route("neden cevapların tak diye kesiliyor?") == "direct"
    assert keyword_route("devam et eden durdun") == "direct"
    assert keyword_route("RAG yeteneğin var mı?") == "direct"
    pasted = (
        "Bu çok teknik ve güzel bir soru. Cevabı hem evet hem de hayır şeklinde verilebilir. "
        "Güncel sistemlerde RAG farklı şekillerde uygulanır. neden cevapların kesiliyor?"
    )
    assert keyword_route(pasted) == "direct"
    assert not is_web_query(pasted)


def test_core_live_queries_still_route_to_web():
    assert keyword_route("tamam bana güncel olarak euro fiyatını araştır") == "web"
    assert keyword_route("bugün dolar kuru ne kadar?") == "web"


def test_init_state_carries_force_web_search():
    from src.agent import graph

    state = graph._init_state("selam", force_web_search=True)

    assert state["force_web_search"] is True


def test_web_search_command_detection():
    from types import SimpleNamespace
    import src.main as main

    assert main._is_web_search_command(SimpleNamespace(command="Web Search")) is True
    assert main._is_web_search_command(SimpleNamespace(command="")) is False


@pytest.mark.anyio
async def test_force_web_search_routes_normal_question_to_web(monkeypatch):
    from src.agent import graph
    from src.agent import nodes

    monkeypatch.setattr(nodes, "_observe_node", lambda *args, **kwargs: None)
    state = graph._init_state("selam nasılsın?", force_web_search=True)

    routed = await nodes.router_node(state)

    assert routed["route"] == "web"


@pytest.mark.anyio
async def test_force_web_search_disabled_keeps_existing_direct_route(monkeypatch):
    from src.agent import graph
    from src.agent import nodes

    monkeypatch.setattr(nodes, "_observe_node", lambda *args, **kwargs: None)
    state = graph._init_state("selam nasılsın?", force_web_search=False)

    routed = await nodes.router_node(state)

    assert routed["route"] == "direct"


def test_contextual_web_query_uses_previous_product_for_price_followup():
    history = [
        AIMessage(content="Belgelerde geçen telefon modeli, APPLE IPHONE 13 128 GB GECE YARISI'dır (Kaynak 1).")
    ]

    query = _build_contextual_web_query("şuanki güncel fiyatı ne kadar?", history)

    assert "IPHONE 13" in query.upper()
    assert "128 GB" in query.upper()
    assert "Türkiye" in query
    assert "güncel fiyat" in query


def test_contextual_web_query_keeps_explicit_entity():
    query = _build_contextual_web_query("bugünkü Tesla hisse fiyatı ne kadar?", [])

    assert query.startswith("bugünkü Tesla hisse fiyatı ne kadar?")
    assert "official quote" in query
    assert "Yahoo Finance" in query


def test_compact_web_query_stays_under_tavily_limit():
    long_query = "tamam bana güncel olarak euro fiyatını araştır " + ("önceki cevap metni " * 80)

    query = _compact_web_query(long_query)

    assert len(query) <= 360
    assert "EURO" in query.upper()


def test_contextual_web_query_compacts_long_price_followup():
    long_query = "şuanki güncel euro fiyatı ne kadar? " + ("önceki konuşma " * 80)

    query = _build_contextual_web_query(long_query, [])

    assert len(query) <= 360
    assert "EURO" in query.upper()


def test_web_sources_are_numbered_and_dated():
    web_text = """Web search results for: TSLA stock price

[Result 1] Nasdaq Tesla Stock
Published: 2026-05-16
Snippet: TSLA last traded at 440.00 USD.
Source: https://example.com/tsla

[Result 2] MarketWatch TSLA
Published: unknown
Snippet: Previous close was 438.00 USD.
Source: https://example.com/marketwatch
"""

    records = WebResultFormatter.extract_source_records(web_text)
    answer = WebResultFormatter.append_sources("TSLA son fiyatı 440.00 USD [1].", web_text, "Tesla hisse fiyatı")

    assert records[0].title == "Nasdaq Tesla Stock"
    assert records[0].published == "2026-05-16"
    assert "- [1] [Nasdaq Tesla Stock](https://example.com/tsla) — 2026-05-16" in answer
    assert "- [2] [MarketWatch TSLA](https://example.com/marketwatch)" in answer


def test_web_result_documents_keep_structured_metadata_and_sort_freshest_first():
    result = WebSearchResult(
        provider="tavily",
        text="""Web search results for: TSLA stock price

[Result 1] Older TSLA
Published: 2025-01-01
Snippet: Older value.
Source: https://old.example/tsla

[Result 2] Fresh TSLA
Published: 2026-05-16
Snippet: Fresh value.
Source: https://fresh.example/tsla
""",
    )

    docs = _web_docs_from_result(result, query="TSLA price")

    assert docs[0].metadata["title"] == "Fresh TSLA"
    assert docs[0].metadata["provider"] == "tavily"
    assert docs[0].metadata["result_index"] == 2
    assert docs[0].metadata["retrieved_at"]
    assert docs[0].metadata["query"] == "TSLA price"
    assert docs[0].metadata["type"] == "web_search"


def test_web_result_documents_prefer_authoritative_sources_over_low_signal_results():
    result = WebSearchResult(
        provider="tavily",
        text="""Web search results for: city population

[Result 1] City Population Map Listing
Published: 2026-01-01
Snippet: Map result.
Source: https://maps.example.com/place/city-population-office

[Result 2] Official Census Agency Population Table
Published: unknown
Snippet: The city population is reported as 123,456 in the latest official table with district-level methodology notes.
Source: https://data.gov.example/census/city-population

[Result 3] Social Post About Population
Published: 2026-01-02
Snippet: Somebody mentioned a number.
Source: https://social.example.com/posts/123
""",
    )

    docs = _web_docs_from_result(result, query="city population")

    assert docs[0].metadata["domain"] == "data.gov.example"
    assert "123,456" in docs[0].metadata["excerpt"]


def test_web_result_documents_dedupe_duplicate_urls_before_numbering():
    result = WebSearchResult(
        provider="tavily",
        text="""Web search results for: sample

[Result 1] Same A
Published: 2026-01-01
Snippet: First.
Source: https://same.example/page

[Result 2] Same A Copy
Published: 2026-01-02
Snippet: Duplicate.
Source: https://same.example/page

[Result 3] Different
Published: 2026-01-03
Snippet: Other.
Source: https://other.example/page
""",
    )

    docs = _web_docs_from_result(result, query="sample")

    assert [doc.metadata["url"] for doc in docs].count("https://same.example/page") == 1
    assert len(docs) == 2


@pytest.mark.anyio
async def test_explicit_url_fetch_becomes_primary_web_document(monkeypatch):
    import src.agent.nodes as nodes

    async def fake_fetch(url, **_kwargs):
        return url, "<html><title>Model Card</title><body>Gemma model details with context length and Turkish tuning notes.</body></html>"

    monkeypatch.setattr(nodes, "fetch_public_url_text", fake_fetch)

    docs = await _docs_from_explicit_urls("https://model.example/card bu sayfadaki modeli açıkla")

    assert len(docs) == 1
    assert docs[0].metadata["direct_url"] is True
    assert docs[0].metadata["domain"] == "model.example"
    assert "Gemma model details" in docs[0].metadata["excerpt"]


def test_web_only_context_uses_web_prompt_not_uploaded_document_refusal():
    from langchain_core.documents import Document

    docs = [
        Document(
            page_content="Title: Release Notes\nPublished: 2026-05-01\nURL: https://help.example/release\nSnippet: Latest version is Example 5.2.",
            metadata={"type": "web_search", "title": "Release Notes", "url": "https://help.example/release"},
        )
    ]

    assembly = assemble_rag_context(
        documents=docs,
        vision_context="",
        rag_history=[],
        answer_question="en yeni sürüm nedir?",
        retrieval_trace=[],
        output_tokens=512,
    )

    assert "Güncel web arama sonuçlarından bağlam sağlandı" in assembly.system_content
    assert "Bu bilgi yüklenen belgelerde yer almamaktadır" not in assembly.system_content


def test_web_source_panel_hides_transport_labels_for_any_web_document():
    from langchain_core.documents import Document
    import src.main as main

    docs = [
        Document(
            page_content=(
                "Title: Official Census Agency Population Table\n"
                "Published: unknown\n"
                "URL: https://data.gov.example/census/city-population\n"
                "Snippet: The city population is reported as 123,456 in the latest official table."
            ),
            metadata={
                "type": "web_search",
                "display_name": "Official Census Agency Population Table",
                "title": "Official Census Agency Population Table",
                "url": "https://data.gov.example/census/city-population",
                "domain": "data.gov.example",
                "excerpt": "The city population is reported as 123,456 in the latest official table.",
            },
        ),
        Document(
            page_content="duplicate",
            metadata={
                "type": "web_search",
                "display_name": "Duplicate",
                "url": "https://data.gov.example/census/city-population",
            },
        ),
    ]

    label, content = main._format_web_source_panel(1, docs[0], docs[0].metadata)

    assert label.startswith("Kaynak 1")
    assert "data.gov.example" in content
    assert "123,456" in content
    assert "Title:" not in content
    assert "Snippet:" not in content


def test_source_panel_dedupes_and_renumbers_web_results(monkeypatch):
    from langchain_core.documents import Document
    from types import SimpleNamespace
    import src.main as main

    monkeypatch.setattr(
        main.cl,
        "Text",
        lambda name, content, display: SimpleNamespace(name=name, content=content, display=display),
    )
    docs = [
        Document(page_content="A", metadata={"type": "web_search", "title": "A", "url": "https://a.example"}),
        Document(page_content="A duplicate", metadata={"type": "web_search", "title": "A copy", "url": "https://a.example"}),
        Document(page_content="B", metadata={"type": "web_search", "title": "B", "url": "https://b.example"}),
    ]

    elements = main._build_source_elements(docs)

    assert [element.name.split(" · ")[0] for element in elements] == ["Kaynak 1", "Kaynak 2"]
    assert "Kaynak 2: B" in elements[1].content


def test_append_used_sources_only_lists_cited_documents():
    from langchain_core.documents import Document

    docs = [
        Document(page_content="a", metadata={"display_name": "A", "url": "https://a.example", "type": "web_search"}),
        Document(page_content="b", metadata={"display_name": "B", "url": "https://b.example", "type": "web_search"}),
    ]

    answer = append_used_sources("Son değer 10 USD [2].", docs, "fiyat nedir?")

    assert "Kaynaklar:" in answer
    assert "[B](https://b.example)" in answer
    assert "[A](https://a.example)" not in answer


def test_tavily_quality_gate_marks_tiny_unknown_results_low_quality():
    assert WebSearchService._is_low_quality_result({
        "title": "Search results",
        "content": "too short",
        "published_date": "unknown",
        "url": "https://example.com",
    })
    assert not WebSearchService._is_low_quality_result({
        "title": "Central Bank Exchange Rates",
        "content": "USD/TRY exchange rate was published with a timestamp and a detailed market note.",
        "published_date": "2026-05-16",
        "url": "https://example.com",
    })


def test_web_structured_fallback_uses_sources_not_raw_document_label():
    from langchain_core.documents import Document

    docs = [
        Document(
            page_content="Title: Euro kuru\nPublished: 2026-05-16\nURL: https://example.com/eur\nSnippet: Euro 52.87 TL seviyesinde.",
            metadata={"type": "web_search", "display_name": "Euro kuru", "url": "https://example.com/eur", "published": "2026-05-16"},
        )
    ]

    answer = _web_fallback_answer("güncel euro fiyatı", docs)

    assert "Web kaynaklarından" in answer
    assert "Belgeden" not in answer
    assert "https://example.com/eur" in answer
