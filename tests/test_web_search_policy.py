from langchain_core.messages import AIMessage

from src.agent.nodes import _build_contextual_web_query, _web_docs_from_result, append_used_sources
from src.agent.routing import keyword_route
from src.agent.web_search import WebResultFormatter, WebSearchResult
from src.agent.web_search import WebSearchService


def test_realtime_query_routes_to_web():
    assert keyword_route("bugünkü Tesla hisse fiyatı ne kadar?") == "web"


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
