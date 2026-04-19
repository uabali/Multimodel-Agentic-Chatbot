"""
Query-level routing utilities — SRP: yalnızca sorgu sınıflandırması.

Sorumluluklar:
 - Keyword tabanlı hızlı rota tespiti (LLM çağrısı yapılmaz)
 - Web sorgusu / MCP ihtiyacı tespit yardımcıları
 - Sorgu normaliz & dil tespit yardımcıları

SOLID uyumu:
 - SRP: Bu modülde sadece "yönlendirme mantığı" var; LLM / I/O hiç yok.
 - OCP: Yeni pattern eklemek için ilgili listeye satır eklemek yeterli.
 - DIP: Saf Python; dışa bağımlılık yok.
"""

from __future__ import annotations

import re

# ── Keyword pattern'leri ────────────────────────────────────────────────────

# Belge zamiri kalıpları — "bu/şu cv/dosya/belge …" içeren her sorgu RAG'a gider.
# _GENERAL_KNOWLEDGE_PATTERNS'dan ÖNCE kontrol edilir; "nedir" override'ını engeller.
_DOCUMENT_PRONOUN_PATTERNS: list[str] = [
    # "bu/şu" + belge türü → kesinlikle yüklü dosyayla ilgili
    r"\b(bu|şu)\s+(cv|dosya|belge|pdf|rapor|doküman|döküman)\b",
    r"\b(bu|şu)\s+(dosyanın|belgenin|cv'nin|pdf'in|raporun)\b",
    # "kime ait" — dosya sahipliği sorusu
    r"\bkime\s+ait\b",
    # "bu dosya/belge ne(dir)?" — içerik sorusu
    r"\b(bu|şu)\s+(dosya|belge|rapor|cv)\s+(ne(dir)?|hakkında|ile ilgili|içeriyor)\b",
    # e-posta / telefon gibi kişisel bilgi istekleri (CV üzerinden)
    r"\b(e[- ]?mail|e-posta|eposta|telefon|adres|isim|ad[ıi])\s+(nedir|ne|kaç|var\s+m[ıi])\b",
]

# RAG: kullanıcı açıkça belge İÇERİĞİNE atıfta bulunuyorsa LLM atlanır.
# NOT: "dosya" gibi genel kelimeler burada YOK — sadece belge içeriği sorgulama kalıpları.
_RAG_PATTERNS: list[str] = [
    r"(belgede|belgeden|belgedeki|dokümanda|dosyada|dosyadaki|dosyanın\s+içeriğinde|sözleşmede|raporda|metinde)\s",
    r"(belgeye göre|dokümana göre|dosyaya göre|rapora göre)",
    r"(in the document|in the file|according to the|the contract says|the report says)",
    r"(yüklediğim\s+(belge|dosya|pdf|döküman))",
    r"(uploaded\s+(document|file|pdf))",
    r"(indeksle|indexed|kaç kez geçiyor|hangi sayfada)",
    r"(belgeden|dosyadan|rapordan)\s+(özetle|anlat|bul|çıkar|oku)",
    r"(bu\s+dosyanın|bu\s+belgenin)\s+içeriği",
    r"(dosya|belge)nin?\s+(içeriği|içindeki|hakkında|konusu)",
]

# Belge-bağımsız genel sorular — RAG pattern'lardan önce kontrol edilir (override)
# Bu liste, belgede ilgili içerik olsa bile "direct" rotasına yönlendirir.
# NOT: "X nedir?" kalıbı ^ ile anchor'lanmıştır. Aksi halde uzun çok-parçalı
# sorularda ("... bağlantısı nedir?") false positive verir ve yüklü belgeyle
# ilgili sorular web search'e yönlendirilir.
_GENERAL_KNOWLEDGE_PATTERNS: list[str] = [
    # Genel tanım/açıklama soruları — "X nedir?", "what is X?"
    # ^ anchor: tüm soru kısa bir "X nedir?" formu olmalı; substring match yok.
    r"^\s*\w[\w\s]{0,30}\s+(nedir|ne demek|ne anlama gelir|nasıl [çc]al[ıi][şs][ıi]r)\s*\??\s*$",
    r"^(nedir|ne demek|ne anlama)\b",
    r"^\s*(what is|what are|what does \w+ mean|how does \w+ work)\b",
    # Kim kurdu/oluşturdu/geliştirdi — genel bilgi soruları
    r"\b(kim (tarafından|kurdu|olu[şs]turdu|yaptı|geli[şs]tirdi|[çc]ıkardı|buldu|yarattı))\b",
    r"\b(tarafından (geli[şs]tirildi|olu[şs]turuldu|kuruldu|yapıldı))\b",
    r"\b(who (is|are|made|created|built|founded|developed|invented))\b",
    # Açık override — kullanıcı belgeyi dışladığını belirtiyor
    r"\b(belgeden hariç|belgede de[ğg]il|genel olarak|genel bilgi|d[ıi][şs]ar[ıi]dan)\b",
]

_DIRECT_PATTERNS: list[str] = [
    # Selamlama / kimlik / sohbet
    r"^(merhaba|selam|hey|hi|hello|nas[ıi]ls[ıi]n|naber|iyi g[üu]nler|g[üu]nayd[ıi]n|iyi ak[şs]amlar)",
    r"(sen kimsin|ad[ıi]n ne|ne yapabilirsin|who are you|what can you do|what are you)",
    # Sohbet / hazırlık ifadeleri — RAG'a gitmemeli
    r"(haz[ıi]r m[ıi]s[ıi]n|haz[ıi]r[ıi]m|haz[ıi]r|ba[şs]layal[ıi]m|ready|let.s start|let.s go)",
    r"(verece[ğg]im|g[öo]nderece[ğg]im|y[üu]kleyece[ğg]im|atacak|payla[şs]aca[ğg][ıi]m)",
    r"(sana bir|sana [şs]imdi|birazdan|[şs]imdi sana)",
    r"(te[şs]ekk[üu]r|sa[ğg]ol|tamamd[ıi]r|tamam|ok\b|anlad[ıi]m|eyvallah|g[üu]zel|harika)",
    r"(evet|hay[ıi]r|yes|no)\s*$",
    # Matematik / kod
    r"^[\d\s\+\-\*\/\(\)\^\.]+$",
    r"(hesapla|calculate|asal|prime|fibonacci|factorial|s[ıi]rala|sort|reverse)",
    r"(yaz bir kod|write code|write a function|write a script)",
    # Git / VCS / MCP araç çağrısı komutu (genel MCP tanımı DEĞİL)
    r"(github|gitlab|repo|repository|commit|pull request|branch|issue|gist)",
    r"^\s*mcp\s+(çağır|kullan|listele|call|use|list)",
    # Gerçek zamanlı / web (tarihi hava sorguları dahil)
    r"(hava\s*d[uü]?r[uü]mu|havad[uü]?r[uü]?mu|hava\s+nas[ıi]ld[ıi]|hava\s+nas[ıi]l\b|weather|borsa|d[öo]viz|kur|exchange rate|g[üu]ncel|latest news)",
    r"(bug[üu]n|today|[şs]u an|right now|currently|son dakika|breaking)",
    # Spor sonuçları — Türkçe ek varyasyonları dahil ("maci", "sonuclandi", "dun oynanan")
    r"(skor|ma[çc][ıi]?\s*sonucu?|kim\s+kazandı|who\s+won|puan\s+durumu|standings)",
    r"(ma[çc][ıi]?\s*(nas[ıi]l|kazan|oynan|sonu[çc]land)|d[üu]n\s+oynanan)",
    # Takvim / e-posta
    r"(toplant[ıi] ayarla|schedule meeting|takvim|calendar|email g[öo]nder|send email)",
]

_WEB_PATTERNS: list[str] = [
    # Hava durumu — yazım varyasyonları, ek biçimleri ve tarihi sorgular dahil
    # ("havadurumu", "havadrumu", "hava nasıldı", "hava nasıl", "weather")
    r"(hava\s*d[uü]?r[uü]mu|havad[uü]?r[uü]?mu|weather)",
    r"(hava\s+nas[ıi]ld[ıi]|hava\s+nas[ıi]l\b)",
    # Zaman sinyali — açıkça gerçek zamanlı veri isteği
    r"(güncel|latest|today|bugün|son dakika|current|currently|right now)",
    r"(şu\s*an(ki)?|en\s+son\s+(?:haber|gelişme|durum)|breaking\s+news)",
    # Finans
    r"(borsa|döviz|kur|exchange rate|haber|fiyatı?\s+(?:nedir|ne|kaç)|price\s+of)",
    # Spor sonuçları — Türkçe ek varyasyonları dahil ("maci", "sonuclandi", "oynandığı")
    r"(skor|ma[çc][ıi]?\s*sonucu?|kim\s+kazandı|who\s+won|puan\s+durumu|standings|league\s+table)",
    r"(ma[çc][ıi]?\s*(nas[ıi]l|kazan|oynan|sonu[çc]land)|d[üu]n\s+oynanan|oynand[ıi][ğg][ıi])",
    # Yazılım / ürün sürümleri
    r"(son\s+sürüm|yeni\s+sürüm|en\s+son\s+sürüm|latest\s+version|release\s+notes|changelog)",
    # Duyurular / haberler
    r"\b(duyurdu|açıkladı|released|launched|announced)\b",
]

_MCP_PATTERNS: list[str] = [
    r"(github|gitlab|repo|repository|commit|pull request|branch|issue|gist)",
    r"(mcp).*(çağır|kullan|listele|call|use|list)",
    r"(toplantı ayarla|schedule meeting|takvim|calendar|email gönder|send email)",
]

_TURKISH_PATTERNS: list[str] = [
    r"[çğıöşüİı]",
    r"\b(hava|durumu|nasil|nasıl|bugün|istanbul|nedir|ne)\b",
]

_WEATHER_PATTERN = re.compile(
    r"(hava\s*d[uü]?r[uü]mu|havad[uü]?r[uü]?mu|hava\s+nas[ıi]ld[ıi]|hava\s+nas[ıi]l\b|weather)",
    re.IGNORECASE,
)


# ── Public helpers ──────────────────────────────────────────────────────────


def keyword_route(question: str, *, has_uploads: bool = False) -> str | None:
    """RAG veya direct rotası için keyword eşleşmesi dener.

    Öncelik sırası:
    1. _DOCUMENT_PRONOUN_PATTERNS  → rag     (bu/şu cv/dosya/belge — kesin belge referansı)
    2. _GENERAL_KNOWLEDGE_PATTERNS → direct  (belgeden bağımsız genel sorular)
                                            has_uploads=True iken atlanır: kullanıcı
                                            belge yüklemişse "X nedir?" tarzı
                                            soru büyük ihtimalle belge içeriğiyle
                                            ilgilidir — direct'e kaçırma.
    3. _RAG_PATTERNS               → rag     (belgeye özgü sorgular)
    4. _DIRECT_PATTERNS            → direct  (sohbet, matematik, araç komutları, hava/haber)

    Returns:
        "rag", "direct", ya da eşleşme yoksa None (LLM fallback tetiklenir).
    """
    q = question.strip()
    for pattern in _DOCUMENT_PRONOUN_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE | re.UNICODE):
            return "rag"
    if not has_uploads:
        for pattern in _GENERAL_KNOWLEDGE_PATTERNS:
            if re.search(pattern, q, re.IGNORECASE):
                return "direct"
    for pattern in _RAG_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return "rag"
    for pattern in _DIRECT_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return "direct"
    return None


def is_web_query(question: str) -> bool:
    """Sorunun gerçek zamanlı web araması gerektirip gerektirmediğini döner."""
    q = question.strip()
    return any(re.search(p, q, re.IGNORECASE) for p in _WEB_PATTERNS)


def needs_mcp_tools(question: str) -> bool:
    """Sorunun MCP araçlarına ihtiyaç duyup duymadığını döner."""
    q = question.strip()
    return any(re.search(p, q, re.IGNORECASE) for p in _MCP_PATTERNS)


def is_turkish_query(question: str) -> bool:
    """Sorgunun Türkçe olup olmadığını döner."""
    return any(re.search(p, question, re.IGNORECASE) for p in _TURKISH_PATTERNS)


def is_weather_query(question: str) -> bool:
    """Sorgunun hava durumu hakkında olup olmadığını döner."""
    return bool(_WEATHER_PATTERN.search(question))


def normalize_web_query(question: str) -> str:
    """Web araması için sorguyu normalize eder.

    Hava durumu sorguları:
    - Yazım düzeltmesi: "havadrumu" → "hava durumu"
    - Çok-günlük tahmin (Query Expansion): "5 günlük" ifadesi varsa sorguya
      spesifik tarih aralığı ve "sıcaklık tahmin" anahtar kelimesi eklenir.
      Bu, Tavily'nin belirsiz snippet yerine gerçek tahmin verisi getirmesini sağlar.
    - Tekli sorgu: tarih ifadesi yoksa "bugün" eklenir.
    """
    import datetime

    normalized = question.strip()
    if is_weather_query(normalized):
        normalized = re.sub(r"\bhavadurumu\b", "hava durumu", normalized, flags=re.IGNORECASE)

        # Query Expansion: "X günlük" → tarih aralığı ekle
        multi_day = re.search(r"(\d+)\s*g[üu]nl[üu]k", normalized, re.IGNORECASE)
        if multi_day:
            days = int(multi_day.group(1))
            today = datetime.date.today()
            end_date = today + datetime.timedelta(days=days - 1)
            # Tarih Türkçe biçimde: "19 Nisan - 23 Nisan 2026"
            months_tr = {
                1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan", 5: "Mayıs", 6: "Haziran",
                7: "Temmuz", 8: "Ağustos", 9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık",
            }
            start_str = f"{today.day} {months_tr[today.month]}"
            end_str = f"{end_date.day} {months_tr[end_date.month]} {end_date.year}"
            normalized = f"{normalized} {start_str}-{end_str} günlük sıcaklık tahmin"
        elif not re.search(
            r"(bugün|today|yarın|yarin|şu an|right now|currently|current|\d{4})",
            normalized, re.IGNORECASE
        ):
            normalized = f"{normalized} bugün"
    return normalized
