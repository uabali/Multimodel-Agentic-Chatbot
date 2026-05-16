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

    assert len(recent) == main._SUMMARY_KEEP_RECENT
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
