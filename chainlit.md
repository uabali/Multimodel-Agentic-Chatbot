# Frappe — Multimodal RAG Agent

Belgeler, görseller, sesli sorular ve web araması — hepsi tek arayüzde.


## Desteklenen Girdiler

- **Metin** — doğrudan yaz
- **Görsel** — PNG / JPG / WEBP ekle, Gemma 4 Vision analiz eder
- **Mikrofon** — mikrofon ikonuna bas, Whisper transkribe eder
- **Ses dosyası** — MP3 / WAV / OGG yükle → otomatik transkribe + indeksle
- **Belge** — PDF / DOCX / TXT / MD / XLSX / CSV → Qdrant'a ingest
- **URL** — `/url` komutuyla web içeriği

---

**Stack:** LangGraph · llama.cpp · Qdrant · Chainlit · edge-TTS · faster-whisper · MCP
