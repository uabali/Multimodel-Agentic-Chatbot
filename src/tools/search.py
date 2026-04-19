"""
Web search tools — LangChain @tool dekoratörü ile tanımlı, SOLID uyumlu.

Tasarım kararları:
 - SRP: Her tool yalnızca kendi arama API'sini çağırır; format / sunum yok.
 - Tool açıklamaları (docstring) LLM'in doğru aracı seçmesi için yeterince açıklayıcı.
 - Senkron `@tool` dekoratörü: LangGraph tool executor ile uyumlu
   (`cl.Step` buraya taşınmaz; web_search_node'da zaten var).

Not: Brave MCP aracı nodes.py üzerinden kullanılır; burada tanımlı değildir.
     Bu dosya yalnızca HTTP tabanlı fallback araçları içerir.
"""

from __future__ import annotations

from langchain_core.tools import tool

from src.config import settings


@tool
def search_web(query: str) -> str:
    """Search the internet using DuckDuckGo for real-time or up-to-date information.

    Use this when:
    - The uploaded documents do not contain the answer.
    - The question requires current, live data (news, weather, prices).

    Args:
        query: Search query text. Be specific for better results.

    Returns:
        Formatted search results or an error message.
    """
    from duckduckgo_search import DDGS

    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=5)
        if not results:
            return "No web results found."

        lines = [
            f"{i}. {r['title']}\n   {r['body']}\n   Source: {r['href']}"
            for i, r in enumerate(results, 1)
        ]
        return "\n\n".join(lines)
    except Exception as exc:
        return f"Web search error: {exc}"


@tool
def tavily_search(query: str) -> str:
    """Search the internet via Tavily API for high-quality real-time information.

    Preferred over `search_web` (DuckDuckGo) when Tavily API key is configured.
    Use for: weather, news, stock prices, current events.

    Args:
        query: Search query text.

    Returns:
        Structured search results with summary and sources, or an error message.
    """
    if not settings.tavily_api_key:
        return "ERROR: TAVILY_API_KEY not set. Tavily search unavailable."

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=settings.tavily_api_key)
        resp = client.search(
            query=query,
            max_results=settings.web_search_max_results,
            include_answer=True,
            search_depth="basic",
        )
        answer = (resp.get("answer") or "").strip()
        results = resp.get("results", [])

        if not results and not answer:
            return "No Tavily results found."

        parts = [f"Web search results for: {query}"]
        if answer:
            parts.append(f"[Summary]: {answer}")
        for idx, r in enumerate(results, 1):
            title = r.get("title", "")
            content = (r.get("content") or "")[:300]
            url = r.get("url", "")
            parts.append(f"[Result {idx}] {title}\n{content}\nSource: {url}")

        return "\n\n".join(parts)
    except Exception as exc:
        return f"Tavily search error: {exc}"
