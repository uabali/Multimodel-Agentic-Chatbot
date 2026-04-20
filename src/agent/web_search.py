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
from typing import NamedTuple

from src.agent.routing import is_turkish_query, normalize_web_query

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Value objects
# ─────────────────────────────────────────────────────────────────────────────


class WebSearchResult(NamedTuple):
    """Ham web arama sonucu."""

    text: str
    provider: str  # "tavily"


# ─────────────────────────────────────────────────────────────────────────────
# Tavily provider
# ─────────────────────────────────────────────────────────────────────────────


class WebSearchService:
    """Tavily API üzerinden web arama servisi."""

    def __init__(self, api_key: str, max_results: int = 5) -> None:
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

    async def search(self, query: str) -> WebSearchResult | None:
        """Tavily API ile arama yapar; başarısız olursa None döner."""
        normalized = normalize_web_query(query)
        try:
            import datetime

            def _call() -> str:
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
                    return ""
                parts = [f"Web search results for: {normalized}"]
                for idx, r in enumerate(results, 1):
                    title = r.get("title", "")
                    published = r.get("published_date", "")
                    content = (r.get("content") or "")[:700]
                    url = r.get("url", "")
                    date_tag = f" [{published}]" if published else ""
                    parts.append(f"[Result {idx}]{date_tag} {title}\n{content}\nSource: {url}")
                return "\n\n".join(parts)

            text = await asyncio.to_thread(_call)
            if not text or "ERROR" in text[:80].upper():
                return None
            logger.info("Tavily web search: %d chars", len(text))
            return WebSearchResult(text=text, provider="tavily")
        except Exception as exc:
            logger.warning("Tavily search failed: %s", exc)
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Result formatter — SRP: yalnızca sunum mantığı
# ─────────────────────────────────────────────────────────────────────────────


class WebResultFormatter:
    """Web arama sonuçlarını kullanıcıya gösterim için formatlar."""

    @staticmethod
    def extract_sources(web_text: str, limit: int = 2) -> list[tuple[str, str]]:
        """`Source:` satırlarından (başlık, URL) çiftlerini çıkarır."""
        sources: list[tuple[str, str]] = []
        current_title = ""
        for raw_line in web_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            m = re.match(r"\[Result\s+\d+\]\s+(.+)", line, re.IGNORECASE)
            if m:
                current_title = m.group(1).strip()
                continue
            if line.lower().startswith("source:"):
                url = line.split(":", 1)[1].strip()
                if url and all(u != url for _, u in sources):
                    sources.append((current_title or url, url))
                    current_title = ""
            if len(sources) >= limit:
                break
        return sources

    @staticmethod
    def append_sources(answer: str, web_text: str, question: str, limit: int = 2) -> str:
        """Yanıtın altına kaynak listesi ekler."""
        sources = WebResultFormatter.extract_sources(web_text, limit)
        if not sources:
            return answer.strip()
        header = "Kaynaklar:" if is_turkish_query(question) else "Sources:"
        lines = [header] + [f"- {title}: {url}" for title, url in sources]
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


# ─────────────────────────────────────────────────────────────────────────────
# Util
# ─────────────────────────────────────────────────────────────────────────────


def _is_time_sensitive(query: str) -> bool:
    """Sorgunun gerçek zamanlı/tarih duyarlı olup olmadığını döner."""
    markers = (
        "bugün", "today", "şu an", "right now", "son 24", "last 24",
        "bu hafta", "this week", "güncel", "latest", "breaking", "son dakika",
        "haber", "fiyatı", "price", "kur", "borsa",
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
