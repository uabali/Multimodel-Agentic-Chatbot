# Frappe — Multimodal RAG Agent

Belgeler, görseller, sesli sorular ve web araması — hepsi tek arayüzde.

## Hızlı Başlangıç

| Komut | Ne Yapar |
|---|---|
| `/upload` | PDF, DOCX, TXT, XLSX, CSV veya ses dosyası yükle |
| `/url <adres>` | Web sayfasını scrape edip bilgi tabanına ekle |
| `/tts <ses>` | Sesli yanıtı aç / ses değiştir (`AhmetNeural`, `EmelNeural`, `auto`…) |
| `/models` | Aktif model isimlerini listele |
| `/whisper download` | Whisper STT modelini indir |

## Desteklenen Girdiler

- **Metin** — doğrudan yaz
- **Görsel** — PNG / JPG / WEBP ekle, Gemma 4 Vision analiz eder
- **Mikrofon** — mikrofon ikonuna bas, Whisper transkribe eder
- **Ses dosyası** — MP3 / WAV / OGG yükle → otomatik transkribe + indeksle
- **Belge** — PDF / DOCX / TXT / MD / XLSX / CSV → Qdrant'a ingest
- **URL** — `/url` komutuyla web içeriği

## Ayarlar (Sol Panel)

Sohbet boyunca istediğin zaman değiştirebilirsin:

- **Sesli Yanıt (TTS)** — edge-TTS ile MP3 çıktı
- **TTS Sesi** — Türkçe / İngilizce ses seçenekleri
- **Temperature** — düşük = tutarlı, yüksek = yaratıcı
- **Max Token** — yanıt uzunluğu
- **Retrieval Stratejisi** — hybrid / similarity / mmr / threshold
- **Reranker** — cross-encoder ile hassasiyet artırma

---

**Stack:** LangGraph · llama.cpp · Qdrant · Chainlit · edge-TTS · faster-whisper · MCP
