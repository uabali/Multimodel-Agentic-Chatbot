# RAG Agent — Unified Pipeline

Agentic RAG Pipeline with MCP, hybrid retrieval, reranking, and multimodal support.

## Komutlar

| Komut | Açıklama |
|---|---|
| `/upload` | Dosya yükle (PDF, DOCX, TXT, MD, XLSX, CSV, ses) |
| `/url <adres>` | Web sayfasını scrape edip RAG'a ekle |
| `/tts <ses>` | TTS sesini değiştir (AhmetNeural, EmelNeural, auto...) |
| `/models` | Aktif modelleri listele |
| `/whisper download` | Whisper modelini indir |

## Desteklenen Girdiler

- **Metin**: Doğrudan chat mesajı
- **Görsel**: PNG, JPG, JPEG, WEBP (drag & drop veya attach)
- **Ses (mikrofon)**: Chainlit mic butonu → Whisper STT
- **Ses dosyası**: MP3, WAV, OGG (yükle → otomatik transcribe → RAG'a ingest)
- **Belgeler**: PDF, DOCX, TXT, MD → chunk → Qdrant
- **Tablo**: XLSX, CSV → Qdrant
- **URL**: `/url` komutuyla web sayfası içeriği

## Desteklenen Çıktılar

- **Metin (streaming)**: Gerçek zamanlı token akışı
- **Ses (TTS)**: edge-TTS ile MP3 çıktı (ayarlardan açılabilir)
- **RAG kaynakları**: Cevabın altında kaynak alıntıları
- **Web arama sonuçları**: Inline step gösterimi
- **Action butonları**: 🔊 Sesli dinle (TTS açık değilse her yanıtta görünür)

## Özellikler

- **Hybrid Retrieval**: Dense + BM25 ile RRF birleştirme
- **Reranking**: Cross-encoder ile hassasiyet artırma
- **Dense Gate**: İlgisiz sorularda yanlış alıntı engeli
- **Web Search**: Brave MCP → Tavily → DuckDuckGo zinciri
- **Vision**: Gemma 4 multimodal görsel analiz
- **MCP**: GitHub, filesystem entegrasyonu
- **Ayar Paneli**: Temperature, max_tokens, strateji, TTS

---

**Stack**: LangGraph + llama.cpp + Qdrant + Chainlit + edge-TTS + faster-whisper + MCP
