import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.memory.thread_memory import (
    ThreadMemory,
    extract_memory_pin,
    is_memory_command,
    memory_hash,
)


def test_legacy_summary_metadata_converts_to_thread_memory():
    memory = ThreadMemory.from_metadata({"summary": "Eski konuşma özeti"})

    assert memory.rolling_summary == "Eski konuşma özeti"
    assert memory.pinned_facts == []
    assert memory.version == 1


@pytest.mark.anyio
async def test_rolling_summary_uses_previous_summary(monkeypatch):
    import src.main as main

    captured = {}

    class FakeLLM:
        async def ainvoke(self, messages):
            captured["prompt"] = messages[0].content

            class Response:
                content = "Eski tercih korundu. Yeni karar: RAG cevapları kısa olacak."

            return Response()

    monkeypatch.setattr("src.rag.llm.get_rag_llm", lambda: FakeLLM())

    history = []
    for i in range(18):
        history.append({"role": "user", "content": f"soru {i}"})
        history.append({"role": "assistant", "content": f"cevap {i}"})

    recent, updated = await main._summarize_and_compress_history(
        history,
        thread_id=None,
        memory=ThreadMemory(rolling_summary="Eski tercih: akademik dil.", pinned_facts=[]),
    )

    assert len(recent) == main._SUMMARY_KEEP_RECENT  # noqa: SLF001
    assert "Eski tercih: akademik dil." in captured["prompt"]
    assert "soru 0" in captured["prompt"]
    assert "Yeni karar" in updated.rolling_summary


def test_explicit_memory_pin_is_thread_scoped_and_capped():
    pin = extract_memory_pin("bunu hatırla: bu thread'de rapor dili akademik olsun")
    assert pin == "bu thread'de rapor dili akademik olsun"
    assert is_memory_command("not al: kaynakları kısa göster")

    memory = ThreadMemory.empty()
    for i in range(25):
        memory = memory.with_pin(f"not {i}")

    assert len(memory.pinned_facts) == 20
    assert memory.pinned_facts[0] == "not 5"
    assert memory.pinned_facts[-1] == "not 24"


def test_build_lc_history_includes_summary_pins_and_recent_messages():
    import src.main as main

    memory = ThreadMemory(
        rolling_summary="Kullanıcı tez projesi üstünde çalışıyor.",
        pinned_facts=["Cevaplar kısa olsun."],
    )
    history = [
        {"role": "user", "content": "Benim adım Ahmet."},
        {"role": "assistant", "content": "Memnun oldum Ahmet."},
    ]

    result = main._build_lc_history(history, memory=memory)

    assert isinstance(result[0], SystemMessage)
    assert "tez projesi" in result[0].content
    assert "Cevaplar kısa olsun" in result[0].content
    assert isinstance(result[1], HumanMessage)
    assert isinstance(result[2], AIMessage)


def test_history_selector_keeps_role_order_and_budget():
    from src.agent.nodes import select_recent_history

    messages = [SystemMessage(content="memory")]
    for i in range(8):
        messages.append(HumanMessage(content=f"user {i} " + ("x" * 600)))
        messages.append(AIMessage(content=f"assistant {i} " + ("y" * 600)))

    selected = select_recent_history(messages, mode="rag")

    assert isinstance(selected[0], SystemMessage)
    assert len(selected) <= 7
    chat = selected[1:]
    assert not chat or isinstance(chat[0], HumanMessage)
    for idx in range(0, len(chat) - 1, 2):
        assert isinstance(chat[idx], HumanMessage)
        assert isinstance(chat[idx + 1], AIMessage)


def test_semantic_cache_context_changes_when_memory_changes():
    from src.agent.graph import build_semantic_cache_context

    memory_a = ThreadMemory(rolling_summary="A")
    memory_b = ThreadMemory(rolling_summary="B")

    ctx_a = build_semantic_cache_context(memory_hash=memory_hash(memory_a))
    ctx_b = build_semantic_cache_context(memory_hash=memory_hash(memory_b))

    assert ctx_a != ctx_b


def test_semantic_cache_rejects_short_or_fallback_answers():
    from src.agent.graph import _looks_cacheable_generation

    assert not _looks_cacheable_generation("Bu belge, bir yapay zeka sisteminin")
    assert not _looks_cacheable_generation("Bu bilgi yüklenen belgelerde yer almamaktadır.")
    assert _looks_cacheable_generation(
        "Bu belge FRAPPE sisteminin RAG mimarisini, retrieval stratejisini ve değerlendirme sonuçlarını özetler. [Kaynak 1]\n\nKaynaklar:\n- [1] rapor.pdf"
    )


def test_thread_memory_last_topic_round_trip():
    memory = ThreadMemory.empty().with_summary("Özet").with_last_topic("Token-aware RAG bağlam bütçesi")
    restored = ThreadMemory.from_metadata({"memory": memory.to_metadata()})

    assert restored.last_topic == "Token-aware RAG bağlam bütçesi"
    assert "Son konu" in __import__("src.memory.thread_memory", fromlist=["format_memory_context"]).format_memory_context(restored)


@pytest.mark.anyio
async def test_memory_write_commands_skip_semantic_cache(monkeypatch):
    from src.agent import graph
    from src.rag.semantic_cache import SemanticCache

    class FailingCache:
        async def lookup(self, question, cache_ctx=""):
            raise AssertionError("memory commands must not hit semantic cache")

    class FakeGraph:
        async def astream(self, state, **kwargs):
            yield ("updates", {"direct_response": {"generation": "ok"}})

    monkeypatch.setattr(SemanticCache, "get", classmethod(lambda cls: FailingCache()))
    monkeypatch.setattr(graph, "get_graph", lambda: FakeGraph())
    monkeypatch.setattr(graph, "_graph_config", lambda **kwargs: None)

    events = [
        event
        async for event in graph.astream_agent(
            "bunu hatırla: cevaplar kısa olsun",
            source_filter="doc.pdf",
            memory_hash="abc",
        )
    ]

    assert events == [("updates", {"direct_response": {"generation": "ok"}})]


@pytest.mark.anyio
async def test_force_web_search_skips_semantic_cache(monkeypatch):
    from src.agent import graph
    from src.rag.semantic_cache import SemanticCache

    captured = {}

    class FailingCache:
        async def lookup(self, question, cache_ctx=""):
            raise AssertionError("forced web search must not hit semantic cache")

    class FakeGraph:
        async def astream(self, state, **kwargs):
            captured["state"] = state
            yield ("updates", {"web_search": {"documents": []}})

    async def fake_get_graph_async():
        return FakeGraph()

    monkeypatch.setattr(SemanticCache, "get", classmethod(lambda cls: FailingCache()))
    monkeypatch.setattr(graph, "get_graph_async", fake_get_graph_async)
    monkeypatch.setattr(graph, "_graph_config", lambda **kwargs: None)

    events = [
        event
        async for event in graph.astream_agent(
            "bugün gündemde ne var?",
            force_web_search=True,
            memory_hash="abc",
        )
    ]

    assert events == [("updates", {"web_search": {"documents": []}})]
    assert captured["state"]["force_web_search"] is True


@pytest.mark.anyio
async def test_web_queries_skip_semantic_cache(monkeypatch):
    from src.agent import graph
    from src.rag.semantic_cache import SemanticCache

    class FailingCache:
        async def lookup(self, question, cache_ctx=""):
            raise AssertionError("web queries must not hit semantic cache")

    class FakeGraph:
        async def astream(self, state, **kwargs):
            yield ("updates", {"web_search": {"documents": []}})

    async def fake_get_graph_async():
        return FakeGraph()

    monkeypatch.setattr(SemanticCache, "get", classmethod(lambda cls: FailingCache()))
    monkeypatch.setattr(graph, "get_graph_async", fake_get_graph_async)
    monkeypatch.setattr(graph, "_graph_config", lambda **kwargs: None)

    events = [
        event
        async for event in graph.astream_agent(
            "yarın bayram namazı saat kaçta?",
            memory_hash="abc",
        )
    ]

    assert events == [("updates", {"web_search": {"documents": []}})]


@pytest.mark.anyio
async def test_low_quality_semantic_cache_hit_is_ignored(monkeypatch):
    from src.agent import graph
    from src.rag.semantic_cache import SemanticCache

    stored = {"count": 0}

    class LowQualityCache:
        async def lookup(self, question, cache_ctx=""):
            return "Bu belge, bir yapay zeka sisteminin"

        async def store(self, question, response, cache_ctx=""):
            stored["count"] += 1

    class FakeGraph:
        async def astream(self, state, **kwargs):
            yield ("updates", {"generator": {"generation": "Bu cevap yeterince uzun ve tamamlanmış bir cevaptır. Kaynak bağlamı kullanır ve düzgün biter."}})

    async def fake_get_graph_async():
        return FakeGraph()

    monkeypatch.setattr(SemanticCache, "get", classmethod(lambda cls: LowQualityCache()))
    monkeypatch.setattr(graph, "get_graph_async", fake_get_graph_async)
    monkeypatch.setattr(graph, "_graph_config", lambda **kwargs: None)
    monkeypatch.setattr(graph, "record_semantic_cache_miss", lambda **kwargs: None)

    events = [
        event
        async for event in graph.astream_agent(
            "Belgedeki önemli bulguları özetle",
            source_filter="doc.pdf",
            session_uploads=["doc.pdf"],
            memory_hash="abc",
        )
    ]

    assert events == [("updates", {"generator": {"generation": "Bu cevap yeterince uzun ve tamamlanmış bir cevaptır. Kaynak bağlamı kullanır ve düzgün biter."}})]
    assert stored["count"] == 1


def test_merge_resume_histories_prefers_metadata_tail():
    from src.memory.thread_memory import merge_resume_histories

    steps = [{"role": "user", "content": f"q{i}"} for i in range(20)]
    meta = steps[-6:] + [{"role": "assistant", "content": "fresh answer"}]

    merged = merge_resume_histories(steps, meta)

    assert merged[-1]["content"] == "fresh answer"
    assert len(merged) >= len(meta)


def test_chat_history_metadata_patch_caps_messages():
    from src.memory.thread_memory import chat_history_metadata_patch

    history = [{"role": "user", "content": f"m{i}"} for i in range(150)]
    patch = chat_history_metadata_patch(history)

    assert len(patch["chat_history"]) == 100


def test_should_summarize_by_token_budget(monkeypatch):
    import src.main as main

    monkeypatch.setattr(main.settings, "summary_trigger_messages", 100)
    monkeypatch.setattr(main.settings, "summary_trigger_tokens", 50)
    monkeypatch.setattr(main, "count_tokens", lambda text: len(text) // 10)

    long_history = [{"role": "user", "content": "x" * 400} for _ in range(10)]

    assert main._should_summarize_history(long_history)


def test_format_memory_preferences_excludes_last_topic():
    from src.memory.thread_memory import format_memory_preferences

    mem = ThreadMemory(
        rolling_summary="Özet metin",
        pinned_facts=["Kısa cevap"],
        last_topic="Son konu başlığı",
    )
    prefs = format_memory_preferences(mem)

    assert "Özet metin" in prefs
    assert "Kısa cevap" in prefs
    assert "Son konu" not in prefs
