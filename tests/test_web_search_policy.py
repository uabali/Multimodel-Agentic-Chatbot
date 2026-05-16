from langchain_core.messages import AIMessage

from src.agent.nodes import _build_contextual_web_query
from src.agent.web_search import WebResultFormatter


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
