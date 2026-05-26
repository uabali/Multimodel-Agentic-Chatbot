"""
Web search node module.

Performs live internet search, processes results, handles explicit URL scraping,
and generates real-time answers with bullet-points and markdown citations.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import re
import time
from urllib.parse import urlparse

import chainlit as cl
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.nodes.base import get_rag_llm, observe_node, coerce_llm_text, select_recent_history
from src.agent.routing import current_date_context, is_turkish_query, is_weather_query, normalize_web_query
from src.agent.state import AgentState
from src.agent.web_search import WebSearchResult, WebSearchService, WebSourceRecord, WebResultFormatter
from src.config import settings
from src.security.url_guard import URLFetchError
from src.security.url_guard import fetch_public_url_text as _orig_fetch_public_url_text

async def fetch_public_url_text(url: str, **kwargs) -> tuple[str, str]:
    import sys
    nodes_mod = sys.modules.get("src.agent.nodes")
    if nodes_mod is not None:
        current_val = getattr(nodes_mod, "fetch_public_url_text", None)
        if current_val is not None and not getattr(current_val, "_is_original", False):
            return await current_val(url, **kwargs)
    return await _orig_fetch_public_url_text(url, **kwargs)

logger = logging.getLogger(__name__)

_COMPOUND_QUERY_MARKERS = re.compile(
    r"(etkinlik|konser|festival|fuar|sergi|event|activity|activities"
    r"|haber|news|fiyat|price|skor|score|borsa|kur|exchange"
    r"|nerede|nereye|ne zaman|hangi|what|where|when|which"
    r"|\d\s*g[üu]nl[üu]k|haftal[ıi]k|tablo|table|forecast|tahmin|g[üu]nl[üu]k\s*tahmin)",
    re.IGNORECASE,
)

_FOLLOWUP_PRICE_RE = re.compile(
    r"(fiyat[ıi]?|kaç\s+para|ne\s+kadar|hisse|stock|price|de[ğg]eri|value)",
    re.IGNORECASE | re.UNICODE,
)
_EXPLICIT_WEB_ENTITY_RE = re.compile(
    r"\b(tesla|tsla|apple|iphone|samsung|xiaomi|alt[ıi]n|gram|euro|dolar|usd|eur|try|"
    r"btc|bitcoin|ethereum|nasdaq|borsa|nvda|nvidia|aapl|msft)\b|\b\d+\s*gb\b",
    re.IGNORECASE | re.UNICODE,
)
_PRODUCT_SUBJECT_PATTERNS = [
    re.compile(
        r"(?:telefon\s+modeli|modeli|cihaz[ıi]?|ürün[üu]?)[^,\n:]*[:,]\s*"
        r"(.{3,90}?)(?:'?[dt][ıi]r|dir|tir|tır|\(|\.|\n|$)",
        re.IGNORECASE | re.UNICODE,
    ),
    re.compile(
        r"\b((?:APPLE\s+)?IPHONE\s*\d{1,2}(?:\s+(?:PRO|PLUS|MINI|PRO MAX))?"
        r"(?:\s+\d+\s*GB)?(?:\s+[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ\s]{2,30})?)\b",
        re.IGNORECASE | re.UNICODE,
    ),
    re.compile(r"\b(TESLA|TSLA|APPLE|AAPL|NVIDIA|NVDA|MICROSOFT|MSFT)\b", re.IGNORECASE),
]
_WEB_QUERY_MAX_CHARS = 360
_WEB_QUERY_MAX_SUBQUERIES = 3
_CURRENCY_ENTITY_RE = re.compile(
    r"\b(euro|eur|dolar|dollar|usd|try|tl|sterlin|gbp|alt[ıi]n|gram|bitcoin|btc|ethereum|eth)\b",
    re.IGNORECASE | re.UNICODE,
)
_PRAYER_QUERY_RE = re.compile(
    r"\b(bayram\s+namaz[ıi]|namaz|ezan|imsak|iftar|sahur|prayer\s*time)\b",
    re.IGNORECASE | re.UNICODE,
)
_TOMORROW_RE = re.compile(r"\b(yar[ıi]n|tomorrow)\b", re.IGNORECASE | re.UNICODE)
_YESTERDAY_RE = re.compile(r"\b(d[üu]n|yesterday)\b", re.IGNORECASE | re.UNICODE)
_HTTP_URL_RE = re.compile(r"https?://[^\s<>)\"']+", re.IGNORECASE)

_web_search_service = None
_web_search_service_loaded = False


def get_web_search_service() -> WebSearchService | None:
    """Centralized getter for the Tavily WebSearchService singleton."""
    global _web_search_service, _web_search_service_loaded
    if not _web_search_service_loaded:
        _web_search_service = WebSearchService.from_settings()
        _web_search_service_loaded = True
    return _web_search_service


def _hash_text(text: str) -> str:
    try:
        from src.observability.langsmith import stable_hash
        return stable_hash(text)
    except Exception:
        return ""


def _web_domain(url: str) -> str:
    """Extract internet domain from a full URL string."""
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _published_sort_key(published: str) -> tuple[int, str]:
    """Sort search result dates so fresh explicit dates bubble up above unknown timestamps."""
    text = (published or "").strip()
    if not text:
        return (0, "")
    match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    if match:
        return (2, match.group(0))
    match = re.search(r"\d{4}", text)
    if match:
        return (1, match.group(0))
    return (0, text)


def _web_source_quality_score(record: Any) -> int:
    """Compute structural trust score for a search domain (.gov, wikipedia, etc.)."""
    title = str(getattr(record, "title", "") or "")
    url = str(getattr(record, "url", "") or "")
    content = str(getattr(record, "content", "") or "")
    haystack = f"{title} {url}".lower()
    domain = _web_domain(url).replace("www.", "")
    score = 0

    government_suffixes = (".gov", ".gov.tr", ".gob", ".gov.uk", ".gc.ca", ".gouv.fr", ".go.jp")
    if domain.endswith(government_suffixes) or ".gov." in domain:
        score += 30
    if domain.endswith((".edu", ".edu.tr", ".ac.uk")):
        score += 16
    if any(mark in haystack for mark in ("official", "agency", "ministry", "statistics office", "census bureau")):
        score += 10
    if "diyanet.gov.tr" in domain or "diyanet" in haystack:
        score += 50
    if "mgm.gov.tr" in domain or "meteoroloji" in haystack or "meteoroloji genel müdürlüğü" in haystack:
        score += 40
    if any(mark in haystack for mark in ("weather.com", "accuweather.com", "ventusky.com", "meteoblue.com")):
        score += 12
    if any(mark in haystack for mark in ("release notes", "help center", "documentation", "docs.", "/docs/")):
        score += 8
    if any(mark in haystack for mark in ("wikipedia.org", "britannica.com", "encyclopedia.com")):
        score += 12

    if len(re.sub(r"\s+", " ", content).strip()) >= 120:
        score += 4
    if not title.strip() or title.strip().lower() in {"search results", "web results", "result", "untitled"}:
        score -= 8
    if len(content.strip()) < 40:
        score -= 8
    if any(mark in haystack for mark in ("instagram.com", "facebook.com", "x.com/", "twitter.com", "tiktok.com", "linkedin.com")):
        score -= 25
    if any(mark in haystack for mark in ("maps.", "/maps", "map listing", "directory listing", "business listing")):
        score -= 15
    if any(mark in haystack for mark in ("reddit.com", "quora.com", "forum", "blogspot.")):
        score -= 8
    return score


def _record_to_web_document(record: Any, *, provider: str, query: str, retrieved_at: str) -> Document:
    """Format single web hit record into standardized LangChain RAG Document."""
    domain = _web_domain(record.url).replace("www.", "")
    excerpt = re.sub(r"\s+", " ", (record.content or "").strip())[:900]
    return Document(
        page_content=(
            f"Title: {record.title}\n"
            f"Published: {record.published or 'unknown'}\n"
            f"URL: {record.url}\n"
            f"Snippet: {excerpt}"
        )[:2500],
        metadata={
            "display_name": record.title,
            "source": record.url,
            "url": record.url,
            "title": record.title,
            "domain": domain,
            "excerpt": excerpt,
            "published": record.published,
            "provider": provider,
            "result_index": record.index,
            "chunk_index": record.index,
            "retrieved_at": retrieved_at,
            "query": query,
            "type": "web_search",
        },
    )


def _extract_public_urls(text: str, *, limit: int = 2) -> list[str]:
    """Search string for raw http/https links."""
    urls: list[str] = []
    for match in _HTTP_URL_RE.finditer(text or ""):
        url = match.group(0).rstrip(".,;:!?]")
        if url not in urls:
            urls.append(url)
        if len(urls) >= limit:
            break
    return urls


def _html_to_visible_text(raw: str) -> str:
    """Strip script/style tags and pull readable text from crawled HTML."""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        body = soup.get_text(" ", strip=True)
        text = f"{title}\n{body}" if title and title not in body[:200] else body
    except Exception:
        text = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw or "")
        text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text or "").strip()


async def _docs_from_explicit_urls(question: str) -> list[Document]:
    """Directly fetch and scrape visible text from explicit links found inside a query."""
    docs: list[Document] = []
    for idx, url in enumerate(_extract_public_urls(question), 1):
        try:
            final_url, raw_text = await fetch_public_url_text(url, timeout=8.0, max_bytes=1_500_000)
        except (URLFetchError, Exception) as exc:
            logger.warning("Explicit URL fetch failed [%s]: %s", url, exc)
            continue
        text = _html_to_visible_text(raw_text)
        if not text:
            continue
        domain = _web_domain(final_url).replace("www.", "")
        title = text[:90].strip() or final_url
        excerpt = text[:900].strip()
        docs.append(Document(
            page_content=(
                f"Title: {title}\n"
                f"Published: unknown\n"
                f"URL: {final_url}\n"
                f"Snippet: {excerpt}"
            )[:2500],
            metadata={
                "display_name": title,
                "source": final_url,
                "url": final_url,
                "title": title,
                "domain": domain,
                "excerpt": excerpt,
                "published": "",
                "provider": "direct_url",
                "result_index": idx,
                "chunk_index": idx,
                "retrieved_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
                "query": question,
                "type": "web_search",
                "direct_url": True,
            },
        ))
    return docs


def web_docs_from_result(result: Any, *, query: str, limit: int | None = None) -> list[Document]:
    """Convert raw WebSearchResult items into clean RAG document chunks sorted by trust."""
    retrieved_at = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    records = list(getattr(result, "records", None) or [])
    if not records:
        records = WebResultFormatter.extract_source_records(
            result.text,
            limit=limit or settings.web_search_max_results,
        )
    deduped_records = []
    seen_urls: set[str] = set()
    for record in records:
        url_key = (getattr(record, "url", "") or "").strip().lower()
        if url_key and url_key in seen_urls:
            continue
        if url_key:
            seen_urls.add(url_key)
        deduped_records.append(record)

    # Sort results using combined freshness and authority scoring
    scored = [
        (
            _web_source_quality_score(r),
            _published_sort_key(getattr(r, "published", "")),
            r,
        )
        for r in deduped_records
    ]
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    deduped_records = [x[2] for x in scored]

    provider_label = getattr(result, "provider", "tavily")
    return [
        _record_to_web_document(r, provider=provider_label, query=query, retrieved_at=retrieved_at)
        for r in deduped_records[:limit or settings.web_search_max_results]
    ]


def _message_text(message: object) -> str:
    if isinstance(message, dict):
        return str(message.get("content") or "")
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return str(content or "")


def _clean_subject(subject: str) -> str:
    subject = re.split(r"\bKaynaklar?:|\bSources:", subject, maxsplit=1, flags=re.IGNORECASE)[0]
    subject = re.sub(r"\s*\[?Kaynak\s+\d+\]?.*$", "", subject, flags=re.IGNORECASE).strip()
    subject = re.sub(r"\s+", " ", subject).strip(" .,:;\"'`")
    return subject[:90]


def _extract_web_subject_from_history(messages: list) -> str:
    """Search conversational logs for a referenced product/company focus."""
    for message in reversed(messages[-8:]):
        text = _clean_subject(_message_text(message))
        if not text:
            continue
        for pattern in _PRODUCT_SUBJECT_PATTERNS:
            match = pattern.search(text)
            if match:
                subject = _clean_subject(match.group(1))
                if 3 <= len(subject) <= 90:
                    return subject
    return ""


def _build_contextual_web_query(question: str, prior_messages: list) -> str:
    """Inject temporal context and focus entity targets to formulate search queries."""
    normalized = normalize_web_query(question)
    q = question.strip()
    if not _FOLLOWUP_PRICE_RE.search(q):
        return _compact_web_query(normalized)
    today = datetime.date.today().isoformat()
    q_lower = q.lower()
    if _EXPLICIT_WEB_ENTITY_RE.search(q):
        if re.search(r"\b(hisse|stock|borsa)\b", q_lower):
            return _compact_web_query(f"{normalized} {today} official quote Nasdaq Yahoo Finance MarketWatch")
        if re.search(r"\b(fiyat|price)\b", q_lower) and re.search(r"\b(iphone|telefon|phone|apple|samsung|xiaomi)\b", q_lower):
            return _compact_web_query(f"{normalized} Türkiye {today} resmi satıcı fiyat karşılaştırma")
        return _compact_web_query(normalized)

    subject = _extract_web_subject_from_history(prior_messages)
    if not subject:
        return _compact_web_query(normalized)

    if re.search(r"\b(hisse|stock|borsa)\b", q_lower):
        return _compact_web_query(f"{subject} stock price today {today} official quote market")
    if re.search(r"\b(iphone|apple|samsung|xiaomi|telefon|phone)\b", subject, re.IGNORECASE):
        return _compact_web_query(f"{subject} güncel fiyat Türkiye {today} resmi satıcı teknoloji mağazaları")
    return _compact_web_query(f"{subject} {normalized} {today}")


def _web_target_date(question: str) -> datetime.date:
    today = datetime.date.today()
    if _TOMORROW_RE.search(question or ""):
        return today + datetime.timedelta(days=1)
    if _YESTERDAY_RE.search(question or ""):
        return today - datetime.timedelta(days=1)
    return today


def _format_web_query_date(value: datetime.date) -> str:
    months_tr = {
        1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan", 5: "Mayıs", 6: "Haziran",
        7: "Temmuz", 8: "Ağustos", 9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık",
    }
    return f"{value.day} {months_tr[value.month]} {value.year}"


def _extract_location_for_web_query(question: str) -> str:
    q = question or ""
    if re.search(r"\bfethiye\b", q, re.IGNORECASE | re.UNICODE):
        return "Fethiye Muğla"
    city = WebResultFormatter._extract_city(q)
    if city:
        return city
    m = re.search(
        r"\b([A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+)?)\s+"
        r"(?:hava\s*durumu|havadurumu|namaz|ezan|imsak|iftar)",
        q,
        re.UNICODE,
    )
    return m.group(1).strip() if m else ""


def _build_web_search_queries(question: str, prior_messages: list) -> list[str]:
    """Formulate secondary and specific sub-queries for composite weather/prayer topics."""
    base_query = _build_contextual_web_query(question, prior_messages)
    asks_weather = is_weather_query(question)
    asks_prayer = bool(_PRAYER_QUERY_RE.search(question or ""))
    if not (asks_weather and asks_prayer):
        return [base_query]

    target_date = _web_target_date(question)
    date_tr = _format_web_query_date(target_date)
    location = _extract_location_for_web_query(question) or ""
    scoped_location = f"{location} " if location else ""
    queries = [
        f"{scoped_location}{date_tr} hava durumu tahmini Meteoroloji",
        f"{scoped_location}{date_tr} Kurban Bayramı namazı saati Diyanet",
    ]
    deduped: list[str] = []
    for query in queries:
        compact = _compact_web_query(query)
        if compact and compact not in deduped:
            deduped.append(compact)
    return deduped[:_WEB_QUERY_MAX_SUBQUERIES] or [base_query]


async def _search_web_queries(service: WebSearchService | None, queries: list[str]) -> WebSearchResult | None:
    """Run parallel Tavily queries in a non-blocking gather routine."""
    if service is None or not queries:
        return None
    results = await asyncio.gather(*(service.search(query) for query in queries), return_exceptions=True)
    valid: list[WebSearchResult] = []
    for result in results:
        if isinstance(result, WebSearchResult):
            valid.append(result)
        elif isinstance(result, Exception):
            logger.warning("Parallel Tavily search failed: %s", result)
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]
    return _merge_web_results(valid, queries)


def _merge_web_results(results: list[WebSearchResult], queries: list[str]) -> WebSearchResult:
    """Join multiple WebSearchResult hits into a single distinct result packet."""
    records: list[WebSourceRecord] = []
    seen_urls: set[str] = set()
    for result in results:
        for record in result.records:
            url_key = (record.url or "").strip().lower()
            if url_key and url_key in seen_urls:
                continue
            if url_key:
                seen_urls.add(url_key)
            records.append(WebSourceRecord(
                index=len(records) + 1,
                title=record.title,
                url=record.url,
                content=record.content,
                published=record.published,
            ))
    merged_query = " | ".join(queries)
    if records:
        text = WebSearchService._format_records(merged_query, records)
    else:
        text = "\n\n".join(result.text for result in results if result.text)
    return WebSearchResult(text=text, provider="tavily", records=records)


def _compact_web_query(query: str, *, max_chars: int = _WEB_QUERY_MAX_CHARS) -> str:
    """Keep search query sizes within standard limits to avoid Tavily validation blocks."""
    q = re.sub(r"\s+", " ", (query or "").strip())
    if len(q) <= max_chars:
        return q

    first = re.split(r"[.!?\n]", q, maxsplit=1)[0].strip()
    entities = " ".join(dict.fromkeys(m.group(0).upper() for m in _CURRENCY_ENTITY_RE.finditer(q)))
    today = datetime.date.today().isoformat()
    if entities and re.search(r"\b(fiyat|price|kur|exchange|g[üu]ncel|today|bug[üu]n)\b", q, re.IGNORECASE):
        compact = f"{entities} current exchange rate price {today}"
        return compact[:max_chars].rstrip()
    if len(first) >= 20:
        return first[:max_chars].rstrip()
    return q[:max_chars].rstrip()


def _web_fallback_answer(question: str, documents: list[Document]) -> str:
    web_docs = [d for d in documents if (getattr(d, "metadata", {}) or {}).get("type") == "web_search"]
    if not web_docs:
        return ""
    lines: list[str] = []
    header = "Web kaynaklarından bulunanlar:" if is_turkish_query(question) else "Found from web sources:"
    lines.append(header)
    for idx, doc in enumerate(web_docs[:5], 1):
        meta = getattr(doc, "metadata", {}) or {}
        title = meta.get("display_name") or meta.get("title") or meta.get("source") or f"Kaynak {idx}"
        url = meta.get("url") or meta.get("source") or ""
        published = f" ({meta.get('published')})" if meta.get("published") else ""
        snippet = re.sub(r"\s+", " ", (doc.page_content or "").strip())
        snippet = re.sub(r"^(Title|Published|URL|Snippet):\s*", "", snippet)
        lines.append(f"- [Kaynak {idx}] {title}{published}: {snippet[:220]}".rstrip())
        if url:
            lines.append(f"  {url}")
    return "\n".join(lines)


def _is_pure_weather_query(question: str) -> bool:
    return not bool(_COMPOUND_QUERY_MARKERS.search(question))


async def _fast_web_summarize(
    question: str,
    result_text: str,
    prior_messages: list | None = None,
    *,
    search_query: str = "",
) -> str:
    """Summarize raw web contents incorporating chronological rules and markdown citations."""
    system = (
        "You answer ONLY from the provided web search results.\n"
        f"{current_date_context()} Use this date to resolve relative words like bugün/today, yarın/tomorrow, and dün/yesterday.\n"
        "Rules:\n"
        "- Respond in the same language as the user's question.\n"
        "- Turkish question → fully Turkish answer.\n"
        "- Never say you cannot access live data or the internet.\n"
        "- Never repeat the user's question at the start.\n"
        "- Treat the Search query as the intended entity/topic. Ignore results about a different entity, commodity, product, city, ticker, or date.\n"
        "- Extract SPECIFIC entities: names, prices, percentages, dates, company names.\n"
        "- PRICE/STOCK RULE: Give the latest single value when available, with currency, timestamp/date, market status if present, and one short caveat if sources differ.\n"
        "- RECENCY RULE: Prefer the source/result with the latest explicit date or market timestamp. Do NOT list stale historical values unless needed to explain conflict.\n"
        "- DATE RULE: Prefer results matching the resolved target date in the Search query; ignore stale pages about other dates unless explaining that no matching result was found.\n"
        "- OFFICIAL SOURCE RULE: For prayer times, prefer Diyanet or official sources. If no official matching source exists, say official verification was not found before using secondary sources.\n"
        "- If sources conflict, pick the one with the latest date and note it briefly.\n"
        "- Cite each important value/date with bracket citations like [1] or [2], using the result numbers in the provided web results.\n"
        "- Format: use bullet points for multiple facts; prose for single answers.\n"
        "- Do NOT say 'I don't have real-time data' — use what's in the web results.\n"
        "- If the web results do not contain the requested entity/topic, say that reliable matching results were not found; do not answer a different topic.\n"
        "TABLE RULE: If the user asked for a table (tablo, table) OR a multi-day forecast "
        "(5 günlük, haftalık, X-Y arası), extract each day's data from the web results and "
        "present it as a Markdown table with columns: | Gün | Tarih | Hava | Max °C | Min °C |. "
        "Fill only the columns that appear in the results; omit columns with no data. "
        "Do NOT redirect the user to external sites — build the table yourself from the data.\n"
    )
    llm = get_rag_llm(temperature=0.0)
    messages_to_send = [SystemMessage(content=system)]
    if prior_messages:
        messages_to_send.extend(select_recent_history(list(prior_messages), mode="direct"))
    messages_to_send.append(
        HumanMessage(content=f"Question: {question}\nSearch query: {search_query or question}\n\nWeb results:\n{result_text[:6500]}")
    )
    response = await llm.ainvoke(messages_to_send)
    text = (response.content or "").strip()
    return WebResultFormatter.append_sources(text, result_text, question)


async def web_search_node(state: AgentState) -> AgentState:
    """Perform live web search when RAG matching quality is poor or empty."""
    t0 = time.perf_counter()
    question = state.get("original_question") or state["question"]
    existing_docs = state.get("documents", [])
    search_queries = _build_web_search_queries(question, list(state.get("messages", [])))
    search_query = " | ".join(search_queries)
    explicit_url_docs = await _docs_from_explicit_urls(question)

    async with cl.Step(name="Web Search", type="tool") as step:
        step.input = search_query

        service = get_web_search_service()
        result = await _search_web_queries(service, search_queries)

        if result is None and not explicit_url_docs:
            logger.warning("Web search: Tavily kullanılamıyor veya sonuç yok")
            step.output = "Web search failed."
            observe_node(
                "frappe.web_search_result",
                state,
                inputs={"query_preview": search_query[:180]},
                outputs={
                    "status": "unavailable" if service is None else "no_result",
                    "provider": "tavily" if service is not None else "",
                    "result_count": 0,
                    "latency_ms_by_stage": {"total": round((time.perf_counter() - t0) * 1000, 2)},
                },
                metadata={
                    "web_query_preview": search_query[:180],
                    "web_query_hash": _hash_text(search_query),
                    "web_result_count": 0,
                },
                tags=["frappe", "web-search", "no-result"],
            )
            err = (
                "Canlı web araması şu anda devre dışı çünkü `TAVILY_API_KEY` ayarlanmamış. "
                "Web araması için `.env` içine `TAVILY_API_KEY` ekleyip uygulamayı yeniden başlatmalısın."
                if service is None
                else "Web araması sonuç döndürmedi. Lütfen sorguyu biraz daraltıp tekrar deneyin."
            )
            return {**state, "documents": existing_docs, "web_search_error": err}

        web_docs = explicit_url_docs + (web_docs_from_result(result, query=search_query) if result else [])
        step.output = (
            f"Fetched {len(explicit_url_docs)} explicit URL(s) and found content via {result.provider} ({len(result.text)} chars)."
            if result and explicit_url_docs
            else f"Found content via {result.provider} ({len(result.text)} chars)."
            if result
            else f"Fetched {len(explicit_url_docs)} explicit URL(s)."
        )
        if result:
            step.elements = [
                cl.Text(
                    name=f"{result.provider.title()} Results",
                    content=result.text[:2000] + "...",
                    display="inline",
                )
            ]

    domains = sorted({_web_domain(str((d.metadata or {}).get("url") or (d.metadata or {}).get("source") or "")) for d in web_docs})
    domains = [d for d in domains if d]
    dated = [str((d.metadata or {}).get("published") or "") for d in web_docs if (d.metadata or {}).get("published")]
    provider_name = result.provider if result else "direct_url"
    observe_node(
        "frappe.web_search_result",
        state,
        inputs={"query_preview": search_query[:180]},
        outputs={
            "status": "success",
            "provider": provider_name,
            "result_count": len(web_docs),
            "top_domains": " | ".join(domains[:8]),
            "freshest_published": max(dated) if dated else "",
            "latency_ms_by_stage": {"total": round((time.perf_counter() - t0) * 1000, 2)},
        },
        metadata={
            "web_provider": provider_name,
            "web_query_preview": search_query[:180],
            "web_query_hash": _hash_text(search_query),
            "web_result_count": len(web_docs),
            "top_domains": " | ".join(domains[:8]),
        },
        tags=["frappe", "web-search", "success"],
    )
    return {**state, "documents": existing_docs + web_docs}
