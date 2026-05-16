from langchain_core.documents import Document

from src.rag.ingest import DocumentIngester


class _FakeLoader:
    def load(self, file_path):
        return [Document(page_content="PDF text content")]


class _FakeSplitter:
    def split(self, documents):
        return documents


class _FakeVectorStore:
    def __init__(self):
        self.added_batches = []
        self.deleted_sources = []

    def delete_by_source(self, sources):
        self.deleted_sources.append(sources)

    def add_documents(self, documents):
        self.added_batches.append(list(documents))


def test_pdf_visual_ingest_disabled_skips_vision_pipeline(monkeypatch, tmp_path):
    import src.rag.ingest as ingest

    monkeypatch.setattr(ingest.settings, "pdf_visual_ingest_max_pages", 0)

    class FailingVisualPageIngester:
        def __init__(self):
            raise AssertionError("VisualPageIngester should not run when PDF visual ingest is disabled")

    monkeypatch.setattr(ingest, "VisualPageIngester", FailingVisualPageIngester)

    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    vectorstore = _FakeVectorStore()

    result = DocumentIngester(
        loader=_FakeLoader(),
        splitter=_FakeSplitter(),
        vectorstore=vectorstore,
    ).ingest_file(pdf_path)

    assert result["status"] == "success"
    assert result["chunk_count"] == 1
    assert "visual_chunk_count" not in result
    assert len(vectorstore.added_batches) == 1
    assert vectorstore.added_batches[0][0].metadata["file_type"] == ".pdf"


def test_pdf_visual_ingest_enabled_indexes_visual_chunks(monkeypatch, tmp_path):
    import src.rag.ingest as ingest

    monkeypatch.setattr(ingest.settings, "pdf_visual_ingest_max_pages", 2)

    class FakeVisualPageIngester:
        def ingest_pdf_visuals(self, pdf_path, file_id, display_name):
            return [
                Document(
                    page_content="visual page content",
                    metadata={
                        "source_file": pdf_path.name,
                        "file_id": file_id,
                        "display_name": display_name,
                        "file_type": ".pdf",
                        "chunk_type": "visual_description",
                    },
                )
            ]

    monkeypatch.setattr(ingest, "VisualPageIngester", FakeVisualPageIngester)

    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    vectorstore = _FakeVectorStore()

    result = DocumentIngester(
        loader=_FakeLoader(),
        splitter=_FakeSplitter(),
        vectorstore=vectorstore,
    ).ingest_file(pdf_path)

    assert result["visual_chunk_count"] == 1
    assert len(vectorstore.added_batches) == 2
    assert vectorstore.added_batches[1][0].metadata["chunk_type"] == "visual_description"


def test_ingest_adds_thread_resume_metadata(tmp_path):
    pdf_path = tmp_path / "sample.txt"
    pdf_path.write_text("hello", encoding="utf-8")
    vectorstore = _FakeVectorStore()

    result = DocumentIngester(
        loader=_FakeLoader(),
        splitter=_FakeSplitter(),
        vectorstore=vectorstore,
    ).ingest_file(
        pdf_path,
        display_name="sample.txt",
        extra_metadata={"thread_id": "thread-1", "uploaded_at": "2026-05-16T00:00:00Z"},
    )

    assert result["status"] == "success"
    meta = vectorstore.added_batches[0][0].metadata
    assert meta["thread_id"] == "thread-1"
    assert meta["uploaded_at"] == "2026-05-16T00:00:00Z"
