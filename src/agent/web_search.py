"""
Web search provider — Tavily API.

Kullanım:
    service = WebSearchService.from_settings()
    result = await service.search("istanbul hava durumu bugün")

Tavily API key: TAVILY_API_KEY env değişkeni.
Key yoksa web search devre dışı (None döner).
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import NamedTuple

from src.agent.routing import is_turkish_query, normalize_web_query

logger = logging.getLogger(__name__)




@dataclass(frozen=True)
class WebSearchResult:
    """Ham web arama sonucu."""

    text: str
    provider: str  # "tavily"
    records: list["WebSourceRecord"] = field(default_factory=list)


class WebSourceRecord(NamedTuple):
    """Tekil web kaynağı; Markdown citation ve RAG context için kullanılır."""

    index: int
    title: str
    url: str
    content: str
    published: str = ""




class WebSearchService:
    """Tavily API üzerinden web arama servisi."""

    def __init__(self, api_key: str, max_results: int = 5) -> None:
        """Kısa: `__init__` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        self._api_key = api_key
        self._max_results = max_results
        self._client = None

    @classmethod
    def from_settings(cls) -> "WebSearchService | None":
        """Tavily API key varsa servis döner, yoksa None."""
        from src.config import settings
        key = (settings.tavily_api_key or "").strip()
        if not key:
            logger.warning("TAVILY_API_KEY ayarlanmamış — web search devre dışı.")
            return None
        return cls(api_key=key, max_results=settings.web_search_max_results)

    def _get_client(self):
        """TavilyClient singleton — her aramada yeniden oluşturmaktan kaçınır."""
        if self._client is None:
            from tavily import TavilyClient
            self._client = TavilyClient(api_key=self._api_key)
        return self._client

    @staticmethod
    def _is_low_quality_result(result: dict) -> bool:
        """Kısa: `_is_low_quality_result` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        title = re.sub(r"\s+", " ", str(result.get("title") or "")).strip().lower()
        content = re.sub(r"\s+", " ", str(result.get("content") or "")).strip()
        published = str(result.get("published_date") or "").strip().lower()
        generic_title = (
            not title
            or title in {"search results", "web results", "result", "untitled"}
            or title.startswith("results for")
        )
        return len(content) < 30 or (published in {"", "unknown", "none"} and generic_title)

    @staticmethod
    def _format_records(query: str, records: list[WebSourceRecord]) -> str:
        """Kısa: `_format_records` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        if not records:
            return ""
        parts = [f"Web search results for: {query}"]
        for record in records:
            parts.append(
                f"[Result {record.index}] {record.title}\n"
                f"Published: {record.published or 'unknown'}\n"
                f"Snippet: {record.content[:900]}\n"
                f"Source: {record.url}"
            )
        return "\n\n".join(parts)

    async def search(self, query: str) -> WebSearchResult | None:
        """Tavily API ile arama yapar; başarısız olursa None döner."""
        normalized = normalize_web_query(query)
        try:
            import datetime

            def _call() -> list[WebSourceRecord]:
                """Kısa: `_call` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
                client = self._get_client()
                # Zaman duyarlı sorgular için güncel tarih bilgisi eklenir.
                today = datetime.date.today().isoformat()
                dated_query = f"{normalized} (bugün: {today})" if _is_time_sensitive(query) else normalized
                resp = client.search(
                    query=dated_query,
                    max_results=self._max_results,
                    include_answer=False,
                    search_depth="advanced",
                )
                results = resp.get("results", [])
                if not results:
                    return []
                if all(self._is_low_quality_result(r) for r in results):
                    logger.info("Tavily web search zero-result quality gate triggered")
                    return []
                records: list[WebSourceRecord] = []
                for idx, r in enumerate(results, 1):
                    title = r.get("title", "")
                    published = r.get("published_date", "")
                    content = re.sub(r"\s+", " ", r.get("content") or "").strip()[:900]
                    url = r.get("url", "")
                    if not url or self._is_low_quality_result(r):
                        continue
                    records.append(
                        WebSourceRecord(
                            index=len(records) + 1,
                            title=title or url,
                            url=url,
                            content=content,
                            published=published or "",
                        )
                    )
                return records

            records = await asyncio.to_thread(_call)
            text = self._format_records(normalized, records)
            if not text or "ERROR" in text[:80].upper():
                return None
            logger.info("Tavily web search: %d chars", len(text))
            return WebSearchResult(text=text, provider="tavily", records=records)
        except Exception as exc:
            logger.warning("Tavily search failed: %s", exc)
            return None




class WebResultFormatter:
    """Web arama sonuçlarını kullanıcıya gösterim için formatlar."""

    @staticmethod
    def extract_source_records(web_text: str, limit: int = 5) -> list[WebSourceRecord]:
        """Source records çıkarır; eski `[Result]...Source:` formatıyla uyumludur."""
        records: list[WebSourceRecord] = []
        current: dict[str, str | int] | None = None
        snippet_lines: list[str] = []

        def _flush() -> None:
            """Kısa: `_flush` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
            nonlocal current, snippet_lines
            if not current:
                return
            url = str(current.get("url") or "").strip()
            title = re.sub(r"\s+", " ", str(current.get("title") or url)).strip()
            if url and all(r.url != url for r in records):
                records.append(WebSourceRecord(
                    index=int(current.get("index") or len(records) + 1),
                    title=title or url,
                    url=url,
                    content=re.sub(r"\s+", " ", " ".join(snippet_lines)).strip(),
                    published=str(current.get("published") or "").strip(),
                ))
            current = None
            snippet_lines = []

        current_title = ""
        for raw_line in web_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            m = re.match(r"\[Result\s+(\d+)\]\s+(?:\[(.*?)\]\s+)?(.+)", line, re.IGNORECASE)
            if m:
                _flush()
                current = {
                    "index": int(m.group(1)),
                    "published": (m.group(2) or "").strip(),
                    "title": m.group(3).strip(),
                }
                current_title = m.group(3).strip()
                continue
            if current and line.lower().startswith("published:"):
                published = line.split(":", 1)[1].strip()
                if published and published.lower() != "unknown":
                    current["published"] = published
                continue
            if current and line.lower().startswith("snippet:"):
                snippet_lines.append(line.split(":", 1)[1].strip())
                continue
            if line.lower().startswith("source:"):
                url = line.split(":", 1)[1].strip()
                if current is None:
                    current = {"index": len(records) + 1, "title": current_title or url}
                current["url"] = url
                _flush()
                current_title = ""
            elif current:
                snippet_lines.append(line)
            if len(records) >= limit:
                break
        _flush()
        return records[:limit]

    @staticmethod
    def extract_sources(web_text: str, limit: int = 5) -> list[tuple[str, str]]:
        """`Source:` satırlarından (başlık, URL) çiftlerini çıkarır."""
        return [(r.title, r.url) for r in WebResultFormatter.extract_source_records(web_text, limit)]

    @staticmethod
    def append_sources(answer: str, web_text: str, question: str, limit: int = 4) -> str:
        """Yanıtın altına ChatGPT web-search benzeri numaralı kaynak listesi ekler."""
        records = WebResultFormatter.extract_source_records(web_text, limit)
        if not records:
            return answer.strip()
        header = "Kaynaklar:" if is_turkish_query(question) else "Sources:"
        lines = [header]
        for i, record in enumerate(records, 1):
            published = f" — {record.published}" if record.published else ""
            lines.append(f"- [{i}] [{record.title}]({record.url}){published}")
        return f"{answer.strip()}\n\n" + "\n".join(lines)

    @staticmethod
    def _extract_city(question: str) -> str:
        """Hava durumu sorusundan şehir adını çıkarır; bulunamazsa boş döner."""
        # re.IGNORECASE + re.UNICODE ile Türkçe büyük/küçük harf sorunları (İ → i̇) aşılır.
        known = [
            "istanbul", "ankara", "izmir", "bursa", "antalya", "adana", "konya",
            "london", "paris", "berlin", "new york", "tokyo", "dubai", "moscow",
        ]
        for city in known:
            if re.search(r"\b" + city + r"\b", question, re.IGNORECASE | re.UNICODE):
                return city.title()
        # "X hava durumu" veya "weather in/for X" kalıpları (fallback)
        m = re.search(r"(\S+)\s+hava\s*durumu", question, re.IGNORECASE)
        if m:
            candidate = m.group(1).lower()
            if candidate not in {"bugün", "yarın", "şu", "güncel", "bu"}:
                return m.group(1).title()
        m = re.search(r"weather\s+(?:in|for|at)\s+(\S+)", question, re.IGNORECASE)
        if m:
            return m.group(1).title()
        return ""

    @staticmethod
    def format_weather(question: str, web_text: str) -> str:
        """Hava durumu sorguları için yapılandırılmış kısa yanıt üretir."""
        lower = web_text.lower()
        city = WebResultFormatter._extract_city(question)

        temps_c = [int(x) for x in re.findall(r"(\d{1,2})\s*°\s*c", lower, re.IGNORECASE)]
        if not temps_c:
            temps_f = [int(x) for x in re.findall(r"(\d{1,3})\s*°\s*f", lower, re.IGNORECASE)]
            temps_c = [round((f - 32) * 5 / 9) for f in temps_f]
        temps_c = temps_c[:3]

        condition_map = [
            ("parçalı bulutlu", "parçalı bulutlu"),
            ("partly cloudy", "parçalı bulutlu"),
            ("yağmurlu", "yağmurlu"),
            ("chance of rain", "yağmur ihtimali olan"),
            ("rain", "yağışlı"),
            ("windy", "rüzgarlı"),
            ("rüzgarlı", "rüzgarlı"),
            ("güneşli", "güneşli"),
            ("sunny", "güneşli"),
            ("bulutlu", "bulutlu"),
            ("cloudy", "bulutlu"),
        ]
        conditions: list[str] = []
        for needle, label in condition_map:
            if needle in lower and label not in conditions:
                conditions.append(label)
        conditions = conditions[:3]

        air_warning = any(
            t in lower
            for t in ["air quality is unhealthy", "sağlıksız", "yüksek bir kirlilik", "hassas gruplar"]
        )

        location_tr = f"{city} için " if city else ""
        location_en = f"in {city} " if city else ""

        if is_turkish_query(question):
            parts = [f"Web sonuçlarına göre {location_tr}bugünkü durum:"]
            if conditions:
                joined = f"{', '.join(conditions[:-1])} ve {conditions[-1]}" if len(conditions) > 1 else conditions[0]
                parts.append(f"Hava genel olarak {joined}.")
            if temps_c:
                unique = sorted(set(temps_c))
                parts.append(
                    f"Sıcaklık yaklaşık {unique[0]}°C."
                    if len(unique) == 1
                    else f"Sıcaklık yaklaşık {unique[0]}–{unique[-1]}°C aralığında."
                )
            if air_warning:
                parts.append("Hava kalitesi hassas gruplar için sağlıksız olabilir.")
        else:
            parts = [f"Based on web results, today's weather {location_en}:".strip()]
            if conditions:
                parts.append(f"Conditions look generally {', '.join(conditions)}.")
            if temps_c:
                unique = sorted(set(temps_c))
                parts.append(
                    f"Temperature is around {unique[0]}°C."
                    if len(unique) == 1
                    else f"Sources suggest {unique[0]}–{unique[-1]}°C."
                )
            if air_warning:
                parts.append("Air quality may be unhealthy for sensitive groups.")

        return WebResultFormatter.append_sources(" ".join(parts).strip(), web_text, question)




def _is_time_sensitive(query: str) -> bool:
    """Sorgunun gerçek zamanlı/tarih duyarlı olup olmadığını döner."""
    markers = (
        "bugün", "today", "şu an", "şuanki", "şimdiki", "right now", "son 24", "last 24",
        "yarın", "tomorrow", "dün", "yesterday", "bu akşam", "tonight",
        "bu hafta", "this week", "güncel", "latest", "breaking", "son dakika",
        "haber", "fiyat", "price", "kur", "borsa", "hisse", "stock",
        "bayram", "namaz", "ezan", "imsak", "iftar", "hava durumu", "weather", "tahmin",
    )
    q = query.lower()
    return any(m in q for m in markers)


def _coerce_to_str(raw: object) -> str:
    """MCP / tool çıktısını stringe dönüştürür."""
    import json

    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        for key in ("content", "text", "answer", "message"):
            v = raw.get(key)
            if isinstance(v, str) and v.strip():
                return v
        try:
            return json.dumps(raw, ensure_ascii=False)
        except Exception:
            return str(raw)
    return str(raw)
