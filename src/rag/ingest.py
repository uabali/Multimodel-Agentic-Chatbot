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
import time
import uuid
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings

logger = logging.getLogger(__name__)


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

_PDF_SEPARATORS: list[str] = ["\n\n\n", "\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]




class DocumentLoader:
    """Dosya uzantısına göre doğru loader'ı seçer ve belgeleri yükler."""

    def __init__(self, loader_map: dict[str, type] | None = None) -> None:
        """Kısa: `__init__` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        self._loader_map = loader_map or LOADER_MAP

    @property
    def supported_extensions(self) -> list[str]:
        """Kısa: `supported_extensions` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
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




class SemanticParagraphSplitter:
    """Anlamsal Paragraf ve Başlık Bölücü (Semantic Paragraph Chunker).
    
    Metinleri paragraflara (çift satır atlama) ve başlık satırlarına göre ayırır. Paragrafların
    anlamsal bütünlüğünü bozmadan, ardışık paragrafları chunk_size limitine
    ulaşana kadar birleştirir. Eğer bir paragraf tek başına chunk_size limitini
    aşıyorsa, o paragrafı kendi içinde cümle sınırlarına göre böler.
    """
    
    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 200) -> None:
        """Kısa: `__init__` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        import re
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Cümle bölücü regex: Nokta, soru işareti, ünlem sonrası boşlukları yakalar
        self.sentence_endings = re.compile(r"(?<=[.!?])\s+")

    def split_text(self, text: str) -> list[str]:
        """Tek bir metni paragrafları koruyarak böler."""
        import re
        if not text.strip():
            return []
            
        # Paragrafları böl. \n\n ve \r\n\r\n durumlarını kapsar.
        raw_paragraphs = re.split(r"\n\n+", text)
        paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]
        
        chunks: list[str] = []
        current_chunk_parts: list[str] = []
        current_length = 0
        
        for para in paragraphs:
            para_len = len(para)
            
            # Eğer tek bir paragraf chunk_size'tan büyükse, onu cümle cümle bölmeliyiz
            if para_len > self.chunk_size:
                # Önce birikmiş chunk'ı kaydet
                if current_chunk_parts:
                    chunks.append("\n\n".join(current_chunk_parts))
                    current_chunk_parts = []
                    current_length = 0
                
                # Paragrafı cümlelerine ayır
                sentences = [s.strip() for s in self.sentence_endings.split(para) if s.strip()]
                current_sub_parts: list[str] = []
                sub_length = 0
                
                for sentence in sentences:
                    sent_len = len(sentence)
                    if sub_length + sent_len + 1 > self.chunk_size:
                        if current_sub_parts:
                            chunks.append(" ".join(current_sub_parts))
                        # Overlap (çakışma) ekle
                        overlap_parts = []
                        overlap_len = 0
                        for s in reversed(current_sub_parts):
                            if overlap_len + len(s) + 1 <= self.chunk_overlap:
                                overlap_parts.insert(0, s)
                                overlap_len += len(s) + 1
                            else:
                                break
                        current_sub_parts = overlap_parts
                        sub_length = overlap_len
                        
                    current_sub_parts.append(sentence)
                    sub_length += sent_len + 1
                
                if current_sub_parts:
                    chunks.append(" ".join(current_sub_parts))
                continue
                
            # Normal paragraf birleştirme mantığı
            if current_length + para_len + 2 > self.chunk_size:
                # Yeni paragraf sığmıyor, mevcut chunk'ı ekle
                chunks.append("\n\n".join(current_chunk_parts))
                
                # Overlap (çakışma) için son paragrafları al
                overlap_parts = []
                overlap_len = 0
                for p in reversed(current_chunk_parts):
                    if overlap_len + len(p) + 2 <= self.chunk_overlap:
                        overlap_parts.insert(0, p)
                        overlap_len += len(p) + 2
                    else:
                        break
                current_chunk_parts = overlap_parts
                current_length = overlap_len
                
            current_chunk_parts.append(para)
            current_length += para_len + 2
            
        if current_chunk_parts:
            chunks.append("\n\n".join(current_chunk_parts))
            
        return chunks

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Belge listesini döküman seviyesinde böler ve metadataları korur."""
        split_docs: list[Document] = []
        for doc in documents:
            text = doc.page_content
            chunks = self.split_text(text)
            for i, chunk in enumerate(chunks):
                meta = {**doc.metadata}
                # Başlangıç indeksini (start_index) yaklaşık hesapla
                start_idx = text.find(chunk[:100]) if len(chunk) > 100 else text.find(chunk)
                if start_idx != -1:
                    meta["start_index"] = start_idx
                split_docs.append(Document(page_content=chunk, metadata=meta))
        return split_docs


class DocumentSplitter:
    """Anlamsal Paragraf ve Başlık Bölücü (Semantic Paragraph Chunker) wrapper."""

    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ) -> None:
        """Kısa: `__init__` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        # separators parametresi geriye dönük uyumluluk için tutuldu.
        self._splitter = SemanticParagraphSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """Belge listesini paragrafların anlamsal bütünlüğünü koruyarak böler."""
        return self._splitter.split_documents(documents)

    @classmethod
    def from_settings(cls) -> "DocumentSplitter":
        """Uygulama ayarlarından splitter oluşturur."""
        return cls(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)




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
        """Kısa: `__init__` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
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

    def ingest_file(
        self,
        file_path: str | Path,
        display_name: str | None = None,
        extra_metadata: dict | None = None,
    ) -> dict:
        """Tek bir dosyayı yükler, böler ve Qdrant'a ekler.

        Aynı dosya daha önce yüklendiyse eski chunk'lar silinir — duplicate önlenir.

        Returns:
            {"file_name": str, "file_id": str, "chunk_count": int, "status": str}
        """
        file_path = Path(file_path)
        t0 = time.perf_counter()
        try:
            result = self._ingest_file_impl(
                file_path,
                display_name=display_name,
                extra_metadata=extra_metadata,
            )
        except Exception as exc:
            from src.observability.langsmith import record_ingest_observation

            record_ingest_observation(
                file_path=file_path,
                result=None,
                elapsed_s=time.perf_counter() - t0,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

        from src.observability.langsmith import record_ingest_observation

        record_ingest_observation(
            file_path=file_path,
            result=result,
            elapsed_s=time.perf_counter() - t0,
        )
        return result

    def _ingest_file_impl(
        self,
        file_path: str | Path,
        display_name: str | None = None,
        extra_metadata: dict | None = None,
    ) -> dict:
        """Implementation body for ingest_file; separated to keep observation isolated."""
        file_path = Path(file_path)
        display_name = display_name or file_path.name

        # Önceki indekslemeden kalan chunk'ları temizle (idempotent upsert davranışı).
        if hasattr(self._vectorstore, "delete_by_source"):
            self._vectorstore.delete_by_source([file_path.name])

        documents = self._loader.load(file_path)

        file_id = str(uuid.uuid4())
        safe_extra_metadata = {
            str(k): v for k, v in (extra_metadata or {}).items()
            if v is not None and str(k) not in {"source_file", "display_name", "file_id", "file_type"}
        }
        for doc in documents:
            doc.metadata.update({
                "source_file": file_path.name,
                "display_name": display_name,
                "file_id": file_id,
                "file_type": file_path.suffix.lower(),
                **safe_extra_metadata,
            })

        chunks = self._splitter.split(documents)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
        logger.info("%s → %d chunk oluşturuldu", file_path.name, len(chunks))

        self._vectorstore.add_documents(chunks)
        logger.info("%s → %d chunk indekslendi", file_path.name, len(chunks))

        result: dict = {
            "file_name": file_path.name,
            "display_name": display_name,
            "file_id": file_id,
            "chunk_count": len(chunks),
            "status": "success",
        }

        # Opsiyonel multimodal ingestion: default kapalı, çünkü her PDF sayfası
        # ayrı vision LLM çağrısı yapar ve upload latency'yi ciddi artırır.
        if file_path.suffix.lower() == ".pdf":
            if settings.pdf_visual_ingest_max_pages > 0:
                visual_ingester = VisualPageIngester()
                visual_docs = visual_ingester.ingest_pdf_visuals(file_path, file_id, display_name)
                if visual_docs:
                    for doc in visual_docs:
                        doc.metadata.update(safe_extra_metadata)
                    self._vectorstore.add_documents(visual_docs)
                    logger.info(
                        "%s → %d görsel açıklama chunk'ı indekslendi",
                        file_path.name, len(visual_docs),
                    )
                    result["visual_chunk_count"] = len(visual_docs)
            else:
                logger.info("PDF görsel ingestion kapalı: %s", file_path.name)

        return result




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
        """Kısa: `__init__` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        self._dpi = dpi

    @staticmethod
    def available() -> bool:
        """pdf2image kurulu mu kontrol eder."""
        try:
            import pdf2image  # noqa: F401
            return True
        except ImportError:
            return False

    def _render_pages(self, pdf_path: Path, max_pages: int | None = None) -> tuple[list[tuple[int, bytes]], int | None]:
        """PDF sayfalarını sırayla PNG baytlarına çevirir."""
        from pdf2image import convert_from_path, pdfinfo_from_path

        total_pages = None
        try:
            total_pages = int(pdfinfo_from_path(str(pdf_path)).get("Pages", 0)) or None
        except Exception:
            total_pages = None
        kwargs = {"last_page": max_pages} if max_pages and max_pages > 0 else {}
        images = convert_from_path(str(pdf_path), dpi=self._dpi, fmt="png", **kwargs)
        result = []
        for i, img in enumerate(images, 1):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            result.append((i, buf.getvalue()))
        return result, total_pages

    async def _analyse_page_async(self, page_num: int, image_bytes: bytes, semaphore: "asyncio.Semaphore") -> str:
        """Tek sayfayı Gemma vision ile asenkron olarak analiz eder; ham metin döner."""
        from langchain_core.messages import HumanMessage, SystemMessage
        from src.rag.llm import get_rag_llm

        async with semaphore:
            b64 = base64.b64encode(image_bytes).decode()
            llm = get_rag_llm()
            try:
                response = await llm.ainvoke([
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

    async def _ingest_pdf_visuals_async(
        self,
        pdf_path: Path,
        file_id: str,
        display_name: str | None = None,
    ) -> list[Document]:
        """PDF'in her sayfasını asenkron ve paralel olarak görsel analiz eder."""
        import asyncio
        max_pages = settings.pdf_visual_ingest_max_pages
        if max_pages <= 0:
            logger.info("PDF görsel ingestion kapalı: %s", pdf_path.name)
            return []

        try:
            pages, total_pages = self._render_pages(pdf_path, max_pages=max_pages)
        except Exception as exc:
            logger.warning("PDF render hatası (%s): %s", pdf_path.name, exc)
            return []

        if total_pages and total_pages > len(pages):
            logger.info(
                "PDF görsel ingestion ilk %d/%d sayfayla sınırlandı: %s",
                len(pages), total_pages, pdf_path.name,
            )

        # 3 concurrent requests to protect embedding/LLM server rate limits
        semaphore = asyncio.Semaphore(3)
        tasks = []
        
        # Sıralı okumak ve düzgün zip'lemek için sayfaları sıralayarak task'leri oluşturuyoruz
        sorted_pages = sorted(pages, key=lambda x: x[0])
        for page_num, image_bytes in sorted_pages:
            tasks.append(self._analyse_page_async(page_num, image_bytes, semaphore))

        results = await asyncio.gather(*tasks)

        docs: list[Document] = []
        for (page_num, _), text in zip(sorted_pages, results):
            if not text:
                continue
            docs.append(Document(
                page_content=text,
                metadata={
                    "source_file": pdf_path.name,
                    "display_name": display_name or pdf_path.name,
                    "file_id": file_id,
                    "file_type": ".pdf",
                    "chunk_type": "visual_description",
                    "page": page_num,
                    "chunk_index": f"visual_p{page_num}",
                },
            ))
            logger.info(
                "Sayfa %d/%d görsel analizi asenkron olarak tamamlandı (%d karakter)",
                page_num, len(pages), len(text),
            )

        return docs

    def ingest_pdf_visuals(
        self,
        pdf_path: Path,
        file_id: str,
        display_name: str | None = None,
    ) -> list[Document]:
        """PDF'in her sayfasını paralel/asenkron görsel analiz eder (asyncio.run wrapper)."""
        import asyncio
        if not self.available():
            logger.warning(
                "pdf2image bulunamadı — görsel ingestion atlandı. "
                "(pip install pdf2image && apt-get install poppler-utils)"
            )
            return []

        try:
            return asyncio.run(self._ingest_pdf_visuals_async(pdf_path, file_id, display_name))
        except Exception as exc:
            logger.warning("PDF görsel paralel ingestion sırasında beklenmedik hata: %s", exc)
            return []




def ingest_file(
    file_path: str | Path,
    display_name: str | None = None,
    extra_metadata: dict | None = None,
) -> dict:
    """Tek bir dosyayı varsayılan ingester ile indeksler (geriye dönük uyumluluk)."""
    return DocumentIngester.default().ingest_file(
        file_path,
        display_name=display_name,
        extra_metadata=extra_metadata,
    )


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


def index_directory(
    data_dir: str = "data",
    *,
    smart_reindex: bool = False,
    extra_metadata: dict | None = None,
) -> list[dict]:
    """Dizindeki desteklenen dosyaları Qdrant'a indeksler.

    Varsayılan: her dosya için `ingest_file` (idempotent, dosya bazlı sil-yenile).
    smart_reindex=True: tüm dizini yükleyip `HybridVectorStore.smart_reindex` ile
    koleksiyonu fingerprint'e göre yeniden oluşturur (offline corpus senkronu).
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.warning("Dizin bulunamadı: %s", data_dir)
        return []

    loader = DocumentLoader()
    files = sorted(
        f for f in data_path.iterdir()
        if f.is_file() and f.suffix.lower() in loader.supported_extensions
    )
    if not files:
        logger.warning("İndekslenecek dosya yok: %s", data_dir)
        return []

    if smart_reindex:
        from src.rag.vectorstore import get_hybrid_store

        splitter = DocumentSplitter.from_settings()
        all_chunks: list[Document] = []
        for f in files:
            try:
                docs = loader.load(f)
            except Exception as exc:
                logger.warning("%s yüklenemedi: %s", f.name, exc)
                continue
            for doc in docs:
                doc.metadata.setdefault("source_file", f.name)
                doc.metadata.setdefault("display_name", f.name)
                doc.metadata.setdefault("file_type", f.suffix.lower().lstrip("."))
                if extra_metadata:
                    doc.metadata.update(extra_metadata)
            all_chunks.extend(splitter.split(docs))
            logger.info("Hazırlandı: %s", f.name)

        if not all_chunks:
            return []

        store = get_hybrid_store()
        reindexed = store.smart_reindex(all_chunks)
        status = "reindexed" if reindexed else "skipped"
        return [{
            "file_name": data_path.name,
            "chunk_count": len(all_chunks),
            "status": status,
            "mode": "smart_reindex",
        }]

    results: list[dict] = []
    for f in files:
        try:
            results.append(ingest_file(f, extra_metadata=extra_metadata))
        except Exception as exc:
            logger.warning("%s indekslenemedi: %s", f.name, exc)
            results.append({"file_name": f.name, "status": "error", "error": str(exc)})
    return results
