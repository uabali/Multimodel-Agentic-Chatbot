def test_dynamic_budget_keeps_simple_math_small(monkeypatch):
    import src.main as main

    monkeypatch.setattr(main.settings, "llm_context_size", 8192)
    monkeypatch.setattr(main.settings, "rag_context_safety_margin_tokens", 700)

    budget = main._dynamic_answer_token_budget("4 sayısının yarısının 5 katı kaç eder?", cap=1536)

    assert 128 <= budget <= 512


def test_dynamic_budget_general_answer_is_larger_than_old_default(monkeypatch):
    import src.main as main

    monkeypatch.setattr(main.settings, "llm_context_size", 8192)
    monkeypatch.setattr(main.settings, "rag_context_safety_margin_tokens", 700)

    budget = main._dynamic_answer_token_budget("OpenAI ChatGPT hakkında fikrin ne, Gemini'den daha mı iyi?", cap=1536)

    assert budget > 512


def test_force_web_search_short_query_gets_room_to_finish(monkeypatch):
    import src.main as main

    monkeypatch.setattr(main.settings, "llm_context_size", 16384)
    monkeypatch.setattr(main.settings, "rag_context_safety_margin_tokens", 700)

    budget = main._dynamic_answer_token_budget("city population", cap=1536, force_web_search=True)

    assert budget > 768
    assert budget <= main._max_token_ceiling()


def test_dynamic_budget_respects_short_and_long_intent(monkeypatch):
    import src.main as main

    monkeypatch.setattr(main.settings, "llm_context_size", 8192)
    monkeypatch.setattr(main.settings, "rag_context_safety_margin_tokens", 700)

    short_budget = main._dynamic_answer_token_budget("kısa açıkla: RAG nedir?", cap=1536)
    long_budget = main._dynamic_answer_token_budget("detaylı anlat: RAG nedir?", cap=1536)

    assert short_budget <= 512
    assert long_budget == 1536
    assert long_budget <= main._max_token_ceiling()


def test_document_summary_budget_is_not_clamped_as_short_answer(monkeypatch):
    import src.main as main

    monkeypatch.setattr(main.settings, "llm_context_size", 8192)
    monkeypatch.setattr(main.settings, "rag_context_safety_margin_tokens", 700)

    budget = main._dynamic_answer_token_budget("Belgedeki önemli bulguları özetle.", cap=1536)
    topic_budget = main._dynamic_answer_token_budget("Bu belgenin ana konusu nedir?", cap=1536)

    assert budget >= 1408
    assert topic_budget >= 1408


def test_document_summary_budget_handles_pronoun_summary_and_short_floor(monkeypatch):
    import src.main as main

    monkeypatch.setattr(main.settings, "llm_context_size", 8192)
    monkeypatch.setattr(main.settings, "rag_context_safety_margin_tokens", 700)

    summary_budget = main._dynamic_answer_token_budget("bunu özetle", cap=1536)
    short_summary_budget = main._dynamic_answer_token_budget("bunu kısa özetle", cap=1536)

    assert summary_budget >= 1408
    assert short_summary_budget >= 768


def test_dynamic_budget_runtime_context_ceiling(monkeypatch):
    import src.main as main

    monkeypatch.setattr(main.settings, "llm_context_size", 8192)
    monkeypatch.setattr(main.settings, "rag_context_safety_margin_tokens", 700)

    budget = main._dynamic_answer_token_budget("detaylı anlat", cap=9999)

    assert budget <= 1536
    assert budget <= main.settings.llm_context_size - main.settings.rag_context_safety_margin_tokens - 512


def test_truncation_detector_catches_numeric_tail_and_source_header(monkeypatch):
    import src.main as main

    monkeypatch.setattr(main, "count_tokens", lambda _text: 42)

    assert main._looks_truncated("The latest value is listed according to 27", 1536)
    assert main._looks_truncated("Güncel değer resmi verilere göre 27", 1536)
    assert main._looks_truncated("Answer\n\nSources:", 1536)


def test_truncation_detector_does_not_retry_short_colon_answer(monkeypatch):
    import src.main as main

    monkeypatch.setattr(main, "count_tokens", lambda _text: 80)

    assert not main._looks_truncated("Fiyatlar genel olarak şöyle:", 1536)


def test_truncated_stream_detection_does_not_trigger_repair_policy(monkeypatch):
    import src.main as main

    monkeypatch.setattr(main, "count_tokens", lambda _text: 1400)
    answer = "Yanıt devam edecek gibi görünen bir bağlaç ile"

    assert main._stream_truncation_detected(answer, 1536)
    assert main._stream_truncation_detected("Tam cevap.", 1536, hit_length_limit=True)
    assert not main._should_repair_truncated_stream(answer, 1536)
    assert not main._should_repair_truncated_stream("Tam cevap.", 1536, hit_length_limit=True)


def test_finish_reason_length_detection():
    import src.main as main

    class Chunk:
        response_metadata = {"finish_reason": "length"}

    assert main._finish_reason_is_length(Chunk())
    assert main._finish_reason_is_length({"choices": [{"finish_reason": "length"}]})
    assert not main._finish_reason_is_length({"finish_reason": "stop"})


def test_lyrics_translation_request_is_safely_refused():
    import src.main as main

    question = "drake - make them pay lyrics türkçe"

    assert main._is_disallowed_lyrics_translation_request(question)
    answer = main._lyrics_translation_refusal(question)
    assert "tam Türkçe çevirisini paylaşamam" in answer
    assert "tema" in answer
