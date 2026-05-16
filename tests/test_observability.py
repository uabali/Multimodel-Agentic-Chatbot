import pytest


def test_langsmith_disabled_is_noop(monkeypatch):
    from src.observability import langsmith as obs

    obs.reset_langsmith_cache()
    monkeypatch.setattr(obs.settings, "app_langsmith_enabled", False)
    monkeypatch.setattr(obs.settings, "langsmith_api_key", "")

    assert obs.get_langsmith_client() is None
    assert obs.get_langsmith_tracer() is None
    assert obs.build_graph_config(run_name="x", question="hello") is None
    assert obs.record_observation("x") is None


def test_langsmith_metadata_and_payload_are_redacted():
    from src.observability import langsmith as obs

    metadata = obs.build_common_metadata(
        question="Ham soru: ali@example.com 12345678901",
        source_filter="raw-invoice.pdf",
        session_uploads=["raw-invoice.pdf"],
        trace_context={"session_id": "session-123", "channel": "unit"},
    )

    metadata_text = str(metadata)
    assert "Ham soru" not in metadata_text
    assert "raw-invoice.pdf" not in metadata_text
    assert "session-123" not in metadata_text
    assert metadata["question_hash"]
    assert metadata["source_filter_hash"]
    assert metadata["session_id_hash"]

    payload = obs.anonymize_payload({
        "question": "Benim mailim ali@example.com ve TC 12345678901",
        "documents": [{"page_content": "gizli belge chunk metni"}],
        "image_url": {"url": "data:image/png;base64,AAAA1111BBBB2222CCCC3333"},
        "headers": {"Authorization": "Bearer abcdefghijklmnopqrstuvwxyz123456"},
    })
    payload_text = str(payload)
    assert "ali@example.com" not in payload_text
    assert "12345678901" not in payload_text
    assert "gizli belge chunk metni" not in payload_text
    assert "AAAA1111" not in payload_text
    assert "abcdefghijklmnopqrstuvwxyz" not in payload_text


def test_graph_config_contains_run_metadata_without_raw_values(monkeypatch):
    from src.observability import langsmith as obs

    sentinel_tracer = object()
    monkeypatch.setattr(obs, "get_langsmith_tracer", lambda: sentinel_tracer)
    monkeypatch.setattr(obs.settings, "app_env", "test")

    config = obs.build_graph_config(
        run_name="frappe.chat_turn",
        question="raw prompt",
        source_filter="secret.pdf",
        session_uploads=["secret.pdf"],
        input_type="text",
        trace_context={"channel": "chainlit_text", "session_id": "session-raw"},
    )

    assert config is not None
    assert config["run_name"] == "frappe.chat_turn"
    assert config["callbacks"] == [sentinel_tracer]
    assert "frappe" in config["tags"]
    assert "langgraph" in config["tags"]
    assert "rag-context" in config["tags"]
    metadata_text = str(config["metadata"])
    assert "raw prompt" not in metadata_text
    assert "secret.pdf" not in metadata_text
    assert "session-raw" not in metadata_text


def test_run_agent_passes_langsmith_config(monkeypatch):
    from src.agent import graph

    captured = {}

    class FakeGraph:
        def invoke(self, state, config=None):
            captured["state"] = state
            captured["config"] = config
            return {"generation": "ok"}

    monkeypatch.setattr(graph, "get_graph", lambda: FakeGraph())
    monkeypatch.setattr(
        graph,
        "_graph_config",
        lambda **kwargs: {
            "run_name": kwargs["run_name"],
            "metadata": {"channel": (kwargs["trace_context"] or {}).get("channel")},
            "callbacks": ["tracer"],
        },
    )

    result = graph.run_agent("hello", trace_context={"channel": "unit"})

    assert result == "ok"
    assert captured["config"]["run_name"] == "frappe.sync_run"
    assert captured["config"]["callbacks"] == ["tracer"]


@pytest.mark.anyio
async def test_semantic_cache_hit_records_manual_observation(monkeypatch):
    from src.agent import graph
    from src.rag.semantic_cache import SemanticCache

    captured = {}

    class FakeCache:
        async def lookup(self, question, cache_ctx=""):
            return "cached answer"

    monkeypatch.setattr(graph, "get_graph", lambda: None)
    monkeypatch.setattr(graph, "record_semantic_cache_hit", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(graph, "build_graph_config", lambda **kwargs: None)
    monkeypatch.setattr(graph, "_graph_config", lambda **kwargs: None)
    monkeypatch.setattr(SemanticCache, "get", classmethod(lambda cls: FakeCache()))

    events = [
        event
        async for event in graph.astream_agent(
            "belgedeki konu nedir?",
            source_filter="document.pdf",
            trace_context={"channel": "unit"},
        )
    ]

    assert captured["cached_answer"] == "cached answer"
    assert captured["trace_context"]["channel"] == "unit"
    assert events[0] == ("updates", {"generator": {"generation": "cached answer"}})


def test_ingest_observation_omits_raw_filename_and_content(monkeypatch, tmp_path):
    from src.observability import langsmith as obs

    captured = {}

    def fake_record_observation(name, **kwargs):
        captured["name"] = name
        captured.update(kwargs)
        return "run-id"

    monkeypatch.setattr(obs, "record_observation", fake_record_observation)

    file_path = tmp_path / "private-contract.pdf"
    file_path.write_bytes(b"PDF content should not be traced")
    obs.record_ingest_observation(
        file_path=file_path,
        result={"status": "success", "chunk_count": 3, "visual_chunk_count": 0},
        elapsed_s=0.25,
    )

    captured_text = str(captured)
    assert captured["name"] == "frappe.ingest_file"
    assert captured["outputs"]["chunk_count"] == 3
    assert "private-contract.pdf" not in captured_text
    assert "PDF content should not be traced" not in captured_text
