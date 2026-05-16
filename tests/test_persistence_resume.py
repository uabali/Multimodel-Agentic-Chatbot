import pytest

from src.persistence.sqlite_data_layer import SQLiteDataLayer


@pytest.mark.anyio
async def test_get_thread_returns_chainlit_resume_shape(tmp_path):
    db = tmp_path / "chainlit.db"
    dl = SQLiteDataLayer(db)
    thread_id = "thread-1"

    await dl.create_step({
        "id": "u1",
        "threadId": thread_id,
        "name": "admin",
        "type": "user_message",
        "input": None,
        "output": "merhaba",
        "metadata": {},
        "createdAt": "2026-05-16T00:00:00Z",
        "tags": [],
    })
    await dl.create_step({
        "id": "a1",
        "threadId": thread_id,
        "parentId": "u1",
        "name": "Frappe",
        "type": "assistant_message",
        "input": None,
        "output": "Merhaba!",
        "metadata": {},
        "createdAt": "2026-05-16T00:00:01Z",
        "tags": [],
    })

    thread = await dl.get_thread(thread_id)

    assert thread is not None
    assert "elements" in thread
    assert thread["steps"][0]["type"] == "user_message"
    assert thread["steps"][0]["input"] == "merhaba"
    assert thread["steps"][0]["output"] == "merhaba"
    assert thread["steps"][0]["start"] == "2026-05-16T00:00:00Z"
    assert thread["steps"][0]["end"] == "2026-05-16T00:00:00Z"
    assert thread["steps"][0]["streaming"] is False
    assert thread["steps"][1]["output"] == "Merhaba!"


@pytest.mark.anyio
async def test_update_thread_creates_missing_thread_for_sidebar_resume(tmp_path):
    db = tmp_path / "chainlit.db"
    dl = SQLiteDataLayer(db)

    await dl.update_thread("thread-2", name="Saved chat", metadata={"session_uploads": ["a.pdf"]})
    thread = await dl.get_thread("thread-2")

    assert thread is not None
    assert thread["name"] == "Saved chat"
    assert thread["metadata"]["session_uploads"] == ["a.pdf"]
