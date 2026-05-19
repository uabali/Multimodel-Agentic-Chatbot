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
