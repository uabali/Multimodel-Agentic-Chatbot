"""
Web search tools — LangChain @tool dekoratörü ile tanımlı, SOLID uyumlu.

Tasarım kararları:
 - SRP: Her tool yalnızca kendi arama API'sini çağırır; format / sunum yok.
 - Tool açıklamaları (docstring) LLM'in doğru aracı seçmesi için yeterince açıklayıcı.
 - Senkron `@tool` dekoratörü: LangGraph tool executor ile uyumlu
   (`cl.Step` buraya taşınmaz; web_search_node'da zaten var).

Not: Proje web sağlayıcısı olarak yalnızca Tavily kullanır.
"""

from __future__ import annotations

from langchain_core.tools import tool

from src.config import settings


@tool
def tavily_search(query: str) -> str:
    """Search the internet via Tavily API for high-quality real-time information.

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
