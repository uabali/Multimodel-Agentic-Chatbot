"""
Ingest pipeline — belge yükleme, bölme ve Qdrant'a indeksleme.

SOLID uyumu:
 - SRP: `DocumentLoader`, `DocumentSplitter`, `DocumentIngester` ayrı sorumluluklar.
 - OCP: Yeni format eklemek için sadece LOADER_MAP'e satır eklemek yeterli.
 - DIP: `DocumentIngester` somut vectorstore tipine değil, protokol arayüzüne bağlıdır.

Kullanım:
    ingester = DocumentIngester.default()
    result = ingester.ingest_file(path)
"""

from __future__ import annotations

import io
import base64
import logging
import uuid
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings

logger = logging.getLogger(__name__)

# ── Desteklenen format → loader sınıfı eşlemesi (OCP kapısı) ──────────────────

LOADER_MAP: dict[str, type] = {}

try:
    from langchain_community.document_loaders import PyPDFLoader
    LOADER_MAP[".pdf"] = PyPDFLoader
except ImportError:
    logger.warning("PyPDFLoader bulunamadı; PDF desteği devre dışı.")

try:
    from langchain_community.document_loaders import TextLoader
    LOADER_MAP[".txt"] = TextLoader
    LOADER_MAP[".md"] = TextLoader  # UnstructuredMarkdownLoader 'markdown' paketi gerektirir
except ImportError:
    logger.warning("TextLoader bulunamadı; TXT/MD desteği devre dışı.")

try:
    from langchain_community.document_loaders import UnstructuredWordDocumentLoader
    LOADER_MAP[".docx"] = UnstructuredWordDocumentLoader
except ImportError:
    logger.warning("UnstructuredWordDocumentLoader bulunamadı; DOCX desteği devre dışı.")

try:
    from langchain_community.document_loaders import UnstructuredExcelLoader
    LOADER_MAP[".xlsx"] = UnstructuredExcelLoader
    LOADER_MAP[".xls"] = UnstructuredExcelLoader
except ImportError:
    logger.warning("UnstructuredExcelLoader bulunamadı; Excel desteği devre dışı. (pip install unstructured[xlsx])")

try:
    from langchain_community.document_loaders.csv_loader import CSVLoader
    LOADER_MAP[".csv"] = CSVLoader
except ImportError:
    logger.warning("CSVLoader bulunamadı; CSV desteği devre dışı.")

# PDF belgelerinde daha iyi parçalama için özelleştirilmiş ayraçlar
_PDF_SEPARATORS: list[str] = ["\n\n\n", "\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


# ─────────────────────────────────────────────────────────────────────────────
# DocumentLoader — SRP: yalnızca belge yükleme
# ─────────────────────────────────────────────────────────────────────────────


class DocumentLoader:
    """Dosya uzantısına göre doğru loader'ı seçer ve belgeleri yükler."""

    def __init__(self, loader_map: dict[str, type] | None = None) -> None:
        self._loader_map = loader_map or LOADER_MAP

    @property
    def supported_extensions(self) -> list[str]:
        return list(self._loader_map.keys())

    def load(self, file_path: Path) -> list[Document]:
        """Dosyayı yükler ve belge listesi döner."""
        suffix = file_path.suffix.lower()
        loader_cls = self._loader_map.get(suffix)
        if loader_cls is None:
            raise ValueError(
                f"Desteklenmeyen dosya türü: {suffix}. "
                f"Desteklenenler: {self.supported_extensions}"
            )
        logger.info("Yükleniyor: %s (%s)", file_path.name, suffix)
        return loader_cls(str(file_path)).load()


# ─────────────────────────────────────────────────────────────────────────────
# DocumentSplitter — SRP: yalnızca metni parçalara ayırma
# ─────────────────────────────────────────────────────────────────────────────


class DocumentSplitter:
    """Langchain RecursiveCharacterTextSplitter wrapper — yapılandırılabilir."""

    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or _PDF_SEPARATORS,
            strip_whitespace=True,
            add_start_index=True,
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """Belge listesini chunk'lara böler."""
        return self._splitter.split_documents(documents)

    @classmethod
    def from_settings(cls) -> "DocumentSplitter":
        """Uygulama ayarlarından splitter oluşturur."""
        return cls(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)


# ─────────────────────────────────────────────────────────────────────────────
# DocumentIngester — SRP: yalnızca indeksleme koordinasyonu
# ─────────────────────────────────────────────────────────────────────────────


class DocumentIngester:
    """Belge yükleme, bölme ve vektör veritabanına yazma sürecini koordine eder.

    DIP: Somut vectorstore tipine değil; `add_documents` metoduna duck-type bağımlı.
    """

    def __init__(
        self,
        loader: DocumentLoader,
        splitter: DocumentSplitter,
        vectorstore,
    ) -> None:
        self._loader = loader
        self._splitter = splitter
        self._vectorstore = vectorstore

    @classmethod
    def default(cls) -> "DocumentIngester":
        """Varsayılan ayarlarla hazır ingester oluşturur."""
        from src.rag.vectorstore import get_hybrid_store

        return cls(
            loader=DocumentLoader(),
            splitter=DocumentSplitter.from_settings(),
            vectorstore=get_hybrid_store(),
        )

    def ingest_file(self, file_path: str | Path) -> dict:
        """Tek bir dosyayı yükler, böler ve Qdrant'a ekler.

        Aynı dosya daha önce yüklendiyse eski chunk'lar silinir — duplicate önlenir.

        Returns:
            {"file_name": str, "file_id": str, "chunk_count": int, "status": str}
        """
        file_path = Path(file_path)

        # Önceki indekslemeden kalan chunk'ları temizle (idempotent upsert davranışı).
        if hasattr(self._vectorstore, "delete_by_source"):
            self._vectorstore.delete_by_source([file_path.name])

        documents = self._loader.load(file_path)

        file_id = str(uuid.uuid4())
        for doc in documents:
            doc.metadata.update({
                "source_file": file_path.name,
                "file_id": file_id,
                "file_type": file_path.suffix.lower(),
            })

        chunks = self._splitter.split(documents)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
        logger.info("%s → %d chunk oluşturuldu", file_path.name, len(chunks))

        self._vectorstore.add_documents(chunks)
        logger.info("%s → %d chunk indekslendi", file_path.name, len(chunks))

        result: dict = {
            "file_name": file_path.name,
            "file_id": file_id,
            "chunk_count": len(chunks),
            "status": "success",
        }

        # Multimodal ingestion: PDF sayfaları Gemma 4 vision ile de analiz edilir
        if file_path.suffix.lower() == ".pdf":
            visual_ingester = VisualPageIngester()
            visual_docs = visual_ingester.ingest_pdf_visuals(file_path, file_id)
            if visual_docs:
                self._vectorstore.add_documents(visual_docs)
                logger.info(
                    "%s → %d görsel açıklama chunk'ı indekslendi",
                    file_path.name, len(visual_docs),
                )
                result["visual_chunk_count"] = len(visual_docs)

        return result


# ─────────────────────────────────────────────────────────────────────────────
# VisualPageIngester — PDF sayfaları Gemma 4 multimodal ile analiz edilir
# ─────────────────────────────────────────────────────────────────────────────


class VisualPageIngester:
    """PDF sayfalarını PNG'ye render edip Gemma 4 vision ile metin/yapı çıkarır.

    Her sayfa ayrı bir Document chunk olarak indekslenir:
      metadata["chunk_type"] = "visual_description"

    Gereklilik: `pdf2image` paketi + sistem seviyesinde `poppler-utils`.
    """

    _SYSTEM_PROMPT = (
        "Sen bir belge analiz asistanısın. PDF sayfalarındaki tüm içeriği "
        "(metin, tablo, grafik, şema, fatura) doğru ve eksiksiz biçimde metne dönüştürürsün. "
        "Tablolar için Markdown formatı kullan. Sayısal değerleri ve isimleri koru. "
        "Yalnızca içeriği döndür; yorum, başlık veya açıklama ekleme."
    )

    _USER_PROMPT = (
        "Bu PDF sayfasını analiz et. Metin içeriğini, tabloları (Markdown), "
        "grafik veri noktalarını ve şema bileşenlerini çıkar. "
        "Sayısal değerleri ve isimleri olduğu gibi koru."
    )

    def __init__(self, dpi: int = 150) -> None:
        self._dpi = dpi

    @staticmethod
    def available() -> bool:
        """pdf2image kurulu mu kontrol eder."""
        try:
            import pdf2image  # noqa: F401
            return True
        except ImportError:
            return False

    def _render_pages(self, pdf_path: Path) -> list[tuple[int, bytes]]:
        """PDF sayfalarını sırayla PNG baytlarına çevirir."""
        from pdf2image import convert_from_path

        images = convert_from_path(str(pdf_path), dpi=self._dpi, fmt="png")
        result = []
        for i, img in enumerate(images, 1):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            result.append((i, buf.getvalue()))
        return result

    def _analyse_page(self, page_num: int, image_bytes: bytes) -> str:
        """Tek sayfayı Gemma 4 vision ile analiz eder; ham metin döner."""
        from langchain_core.messages import HumanMessage, SystemMessage
        from src.rag.llm import get_rag_llm

        b64 = base64.b64encode(image_bytes).decode()
        llm = get_rag_llm()
        try:
            response = llm.invoke([
                SystemMessage(content=self._SYSTEM_PROMPT),
                HumanMessage(content=[
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": self._USER_PROMPT},
                ]),
            ])
            return (response.content or "").strip()
        except Exception as exc:
            logger.warning("Sayfa %d görsel analizi başarısız: %s", page_num, exc)
            return ""

    def ingest_pdf_visuals(self, pdf_path: Path, file_id: str) -> list[Document]:
        """PDF'in her sayfasını görsel analiz eder; chunk Document listesi döner.

        Sayfalar concurrent.futures.ThreadPoolExecutor ile paralel işlenir;
        sıra korunur, hatalı sayfalar atlanır.
        """
        if not self.available():
            logger.warning(
                "pdf2image bulunamadı — görsel ingestion atlandı. "
                "(pip install pdf2image && apt-get install poppler-utils)"
            )
            return []

        try:
            pages = self._render_pages(pdf_path)
        except Exception as exc:
            logger.warning("PDF render hatası (%s): %s", pdf_path.name, exc)
            return []

        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: dict[int, str] = {}
        max_workers = min(4, len(pages))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_page = {
                executor.submit(self._analyse_page, page_num, image_bytes): page_num
                for page_num, image_bytes in pages
            }
            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    results[page_num] = future.result()
                except Exception as exc:
                    logger.warning("Sayfa %d işlenemedi: %s", page_num, exc)
                    results[page_num] = ""

        docs: list[Document] = []
        for page_num, _image_bytes in sorted(pages, key=lambda x: x[0]):
            text = results.get(page_num, "")
            if not text:
                continue
            docs.append(Document(
                page_content=text,
                metadata={
                    "source_file": pdf_path.name,
                    "file_id": file_id,
                    "file_type": ".pdf",
                    "chunk_type": "visual_description",
                    "page": page_num,
                    "chunk_index": f"visual_p{page_num}",
                },
            ))
            logger.info(
                "Sayfa %d/%d görsel analizi tamamlandı (%d karakter)",
                page_num, len(pages), len(text),
            )

        return docs


# ─────────────────────────────────────────────────────────────────────────────
# Geriye dönük uyumluluk — main.py kullanımı bozulmaz
# ─────────────────────────────────────────────────────────────────────────────


def ingest_file(file_path: str | Path) -> dict:
    """Tek bir dosyayı varsayılan ingester ile indeksler (geriye dönük uyumluluk)."""
    return DocumentIngester.default().ingest_file(file_path)


def load_directory(data_dir: str = "data") -> list[Document]:
    """Dizindeki tüm desteklenen dosyaları yükler (toplu indeksleme için)."""
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.warning("Dizin bulunamadı: %s", data_dir)
        return []

    loader = DocumentLoader()
    all_docs: list[Document] = []

    for f in data_path.iterdir():
        if f.is_file() and f.suffix.lower() in loader.supported_extensions:
            try:
                docs = loader.load(f)
                for doc in docs:
                    doc.metadata["source"] = str(f)
                all_docs.extend(docs)
                logger.info("Yüklendi: %s", f.name)
            except Exception as exc:
                logger.warning("%s yüklenemedi: %s", f.name, exc)

    return all_docs
