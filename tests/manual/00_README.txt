━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  FRAPPE — MANUEL TEST REHBERİ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Önce şunları başlat:
  Terminal 1 → make qdrant
  Terminal 2 → make llm
  Terminal 3 → make app
  Tarayıcı   → http://localhost:7860

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TEST DOSYALARI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  01_direct_simple.txt       — Basit doğrudan sorular (routing: direct)
  02_format_conciseness.txt  — Cevap formatı ve kısalık
  03_math_calculator.txt     — Matematik ve hesap makinesi tool
  04_turkish_bilingual.txt   — Türkçe/İngilizce dil geçişleri
  05_rag_document.txt        — Belge yükleme + RAG sorguları
  06_web_search.txt          — Gerçek zamanlı web araması
  07_vision.txt              — Görsel analiz (Gemma 4 Vision)
  08_audio_stt.txt           — Mikrofon + ses dosyası (Whisper)
  09_routing_logic.txt       — Router doğruluğu testi
  10_slash_commands.txt      — /upload /url /tts /models komutları
  11_settings_panel.txt      — Ayar paneli etkileri
  12_streaming_tts.txt       — Streaming + TTS kalitesi
  13_mcp_filesystem.txt      — MCP filesystem tool
  14_edge_cases.txt          — Kenar durumlar ve stres testleri
  15_history_resume.txt      — Sohbet geçmişi ve resume

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TEST NOTASYON REHBERİ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ✅ GEÇTI   — Beklenen davranış tam olarak gerçekleşti
  ❌ BAŞARISIZ — Beklenen davranış gerçekleşmedi
  ⚠️  KISMI   — Kısmen doğru, iyileştirme gerekiyor
  ⏭️  ATLANDI  — Bu özellik aktif değil (örn. TAVILY_API_KEY yok)
