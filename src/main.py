"""
Chainlit application entry point — unified from all three projects.

Features:
  - Password auth (PBKDF2-HMAC-SHA256)
  - SQLite thread/step persistence
  - MCP lifecycle (connect/disconnect/bridge)
  - File upload + ingest to Qdrant (PDF/DOCX/TXT/MD/XLSX/CSV + audio file transcription + URL ingest)
  - Vision (image upload → Gemma 4 multimodal via agent graph)
  - STT via faster-whisper (mic input → text → agent, streaming)
  - TTS via edge-tts (text → MP3 → cl.Audio, toggled per session)
  - LangGraph streaming with smart node filtering
  - Chat Profile (model adi gosterimi)
  - Chat Settings (temperature, max_tokens, retrieval strategy, reranker, TTS)
  - Action buttons (🔊 Sesli dinle — TTS aktif degilse gorünür)
  - Starters for quick actions
"""

import asyncio
import sys
import logging
import base64
import re
import uuid
import os
import secrets
import hashlib
import hmac
import json
import io
import wave
import tempfile
from pathlib import Path

import chainlit as cl
from chainlit.input_widget import Select, Slider, Switch
from chainlit.server import app as _cl_app
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, SystemMessage

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import settings
from src.agent.graph import arun_agent, astream_agent
from src.mcp.mcp_client import get_mcp_tools, is_mcp_tools_cache_warm
from src.rag.ingest import ingest_file
from src.persistence.sqlite_data_layer import SQLiteDataLayer
from src.api.router import router as _api_router
from src.tts import synthesize as tts_synthesize

# ── FastAPI admin/config router'ı Chainlit'e mount et ──────────────────────
# Swagger UI: http://localhost:8000/docs
# Endpoints : http://localhost:8000/api/*
_cl_app.include_router(_api_router)
logger_bootstrap = logging.getLogger("api.mount")
logger_bootstrap.info("FastAPI admin router /api/* mount edildi. Docs: /docs")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


_VISION_FOLLOWUP_RE = re.compile(
    r"("
    r"resim|g[öo]rsel|foto|foto[ğg]raf|image|picture|"
    r"bu\s+(ki[şs]i|adam|kad[ıi]n|nesne|şey|renk|kıyafet|yüz|arka\s*plan)|"
    r"g[öo]rselde|resimde|foto[ğg]rafta|bunda|bundaki|burada|"
    r"ne\s+var|kim\s+var|adam\s+m[ıi]|kad[ıi]n\s+m[ıi]"
    r")",
    re.IGNORECASE | re.UNICODE,
)


def _is_vision_followup(question: str) -> bool:
    """Önceki görseli yalnızca soru gerçekten görsele atıf yapıyorsa yeniden kullan."""
    return bool(_VISION_FOLLOWUP_RE.search((question or "").strip()))


# ── Auth helpers ──


def _ensure_auth_secret():
    if os.getenv("CHAINLIT_AUTH_SECRET"):
        return
    secret = secrets.token_urlsafe(48)
    os.environ["CHAINLIT_AUTH_SECRET"] = secret
    logger.warning("CHAINLIT_AUTH_SECRET not set; generated a random one for this session.")


def _hash_password(password: str, salt: str) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 210_000)
    return dk.hex()


def _constant_time_eq(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


_ensure_auth_secret()


# ── Data layer — singleton shared between Chainlit and internal summarizer ──

_dl_singleton: "SQLiteDataLayer | None" = None


def _get_shared_data_layer() -> "SQLiteDataLayer":
    global _dl_singleton
    if _dl_singleton is None:
        _dl_singleton = SQLiteDataLayer(Path("data") / "chainlit.db")
    return _dl_singleton


@cl.data_layer
def data_layer():
    return _get_shared_data_layer()



# ── Auth callback ──


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if username != settings.app_admin_username:
        return None
    expected = _hash_password(settings.app_admin_password, settings.app_password_salt)
    given = _hash_password(password, settings.app_password_salt)
    if not _constant_time_eq(given, expected):
        return None
    return cl.User(identifier=username, metadata={"role": "admin", "provider": "credentials"})


# ── Chat resume ──


@cl.on_chat_resume
async def on_chat_resume(thread):
    steps = thread.get("steps", []) if isinstance(thread, dict) else []
    history: list[dict] = []
    def _to_text(v) -> str:
        if v is None:
            return ""
        if isinstance(v, str):
            return v
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return str(v)

    for s in steps:
        s_type = (s.get("type") or "").lower()
        if s_type in {"user_message", "user"}:
            content = _to_text(s.get("input")).strip()
            if content:
                history.append({"role": "user", "content": content})
        elif s_type in {"assistant_message", "assistant"}:
            content = _to_text(s.get("output")).strip()
            if content:
                history.append({"role": "assistant", "content": content})
    trimmed = _trim_chat_history(history)
    cl.user_session.set("chat_history", trimmed)
    if trimmed:
        cl.user_session.set("_resume_msg_count", len(trimmed))

    # Önceki oturumdan özet yükle (uzun süreli bellek)
    meta = thread.get("metadata", {}) if isinstance(thread, dict) else {}
    summary = (meta.get("summary", "") or "") if isinstance(meta, dict) else ""
    if summary:
        cl.user_session.set("session_summary", summary)
        logger.info("on_chat_resume: özet yüklendi (%dch)", len(summary))


# ── Whisper / STT ──


_whisper_model = None
_whisper_loading = False


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    from faster_whisper import WhisperModel
    logger.info("Loading Whisper model '%s' (device=cpu, compute_type=int8)...", settings.stt_model)
    _whisper_model = WhisperModel(settings.stt_model, device="cpu", compute_type="int8")
    logger.info("Whisper model '%s' loaded.", settings.stt_model)
    return _whisper_model


async def _preload_whisper() -> None:
    """Whisper modelini background thread'de yukle — startup bloklama olmaz."""
    global _whisper_loading
    if _whisper_model is not None or _whisper_loading or not settings.stt_model:
        return
    _whisper_loading = True
    try:
        import asyncio
        await asyncio.to_thread(_get_whisper_model)
    except Exception as exc:
        logger.warning("Whisper preload failed (will retry on first use): %s", exc)
    finally:
        _whisper_loading = False


async def _preload_reranker() -> None:
    """Reranker modelini background thread'de yukle — startup bloklama olmaz."""
    if not settings.use_rerank:
        return
    try:
        import asyncio
        from src.agent.nodes import _RerankerRegistry
        await asyncio.to_thread(_RerankerRegistry.get)
        logger.info("Reranker preloaded: %s", settings.reranker_model)
    except Exception as exc:
        logger.warning("Reranker preload failed (will retry on first use): %s", exc)


async def _preload_embeddings() -> None:
    """Embedding modelini background thread'de yukle — ilk RAG sorgusunda gecikme olmaz."""
    try:
        from src.rag.vectorstore import _cached_embed_query
        await asyncio.to_thread(_cached_embed_query, "warmup")
        logger.info("Embedding model preloaded: %s", settings.embedding_model)
    except Exception as exc:
        logger.warning("Embedding preload failed (will load on first use): %s", exc)


def _pcm_to_wav(pcm_data: bytes, sample_rate: int = 24000, channels: int = 1, sample_width: int = 2) -> bytes:
    """Convert raw PCM bytes to WAV format for faster_whisper compatibility."""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    return wav_buffer.getvalue()


# ── TTS helpers ──


async def _write_audio_tmp(audio_bytes: bytes) -> str:
    """Ses baytlarını session dizinine (varsa) geçici MP3 olarak yazar.

    Session dizinine yazılırsa on_chat_end'deki shutil.rmtree otomatik temizler.
    Session yoksa system /tmp kullanılır.
    """
    try:
        session_dir = cl.user_session.get("session_upload_dir")
    except Exception:
        session_dir = None

    def _write():
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False, dir=session_dir or None)
        tmp.write(audio_bytes)
        tmp.flush()
        path = tmp.name
        tmp.close()
        return path
    return await asyncio.to_thread(_write)


async def _send_tts(text: str, parent_msg: cl.Message | None = None) -> None:
    """TTS sentezle ve cl.Audio element olarak gonder."""
    voice_pref = cl.user_session.get("tts_voice", "auto")
    voice = None if voice_pref == "auto" else voice_pref
    audio_bytes = await tts_synthesize(text, voice=voice)
    if not audio_bytes:
        return
    tmp_path = await _write_audio_tmp(audio_bytes)
    audio_el = cl.Audio(path=tmp_path, name="response.mp3", display="inline")
    if parent_msg is not None:
        parent_msg.elements = list(getattr(parent_msg, "elements", None) or []) + [audio_el]
        await parent_msg.update()
    else:
        await cl.Message(content="", elements=[audio_el]).send()


# ── Streaming TTS ──


class _TtsStreamer:
    """LLM streaming sırasında TTS sentezini paralel başlatır.

    LLM hâlâ üretirken ilk cümle grubunu arka planda sentezler;
    streaming bitince kalan kısımla birleştirir → tek MP3 → single audio element.

    Kullanım:
        streamer = _TtsStreamer.make(tts_enabled)
        # streaming loop içinde:
        if streamer: streamer.feed(content)
        # streaming bittikten sonra:
        if streamer: await streamer.send_to(msg)
    """

    _SENTENCE_END = re.compile(r"(?<=[.!?\n])\s")
    _MIN_FIRST_CHARS = 150  # Bu kadar karakter birikince ilk chunk'ı arka planda başlat

    def __init__(self, voice: str | None) -> None:
        self._voice = voice
        self._buf = ""
        self._first_task: asyncio.Task | None = None
        self._split_pos = 0  # _buf'ta kaçıncı char'a kadar first_task kapsamında

    def feed(self, chunk: str) -> None:
        """Her streaming chunk'ını besle; eşik aşılınca arka planda TTS başlat."""
        self._buf += chunk
        if self._first_task is None and len(self._buf) >= self._MIN_FIRST_CHARS:
            m = self._SENTENCE_END.search(self._buf, self._MIN_FIRST_CHARS)
            if m:
                self._split_pos = m.start() + 1
                first_text = self._buf[: self._split_pos].strip()
                self._first_task = asyncio.create_task(
                    tts_synthesize(first_text, voice=self._voice)
                )

    async def send_to(self, parent_msg: cl.Message) -> None:
        """Tüm sentezi tamamla, MP3'leri birleştir ve mesaja ekle."""
        remaining = self._buf[self._split_pos :].strip()

        first_audio = await self._first_task if self._first_task else None
        second_audio = (
            await tts_synthesize(remaining, voice=self._voice) if remaining else None
        )

        # MP3 byte'larını birleştir → tek audio element (seamless playback)
        combined = (first_audio or b"") + (second_audio or b"")
        if not combined:
            return

        tmp_path = await _write_audio_tmp(combined)
        audio_el = cl.Audio(path=tmp_path, name="response.mp3", display="inline")
        parent_msg.elements = list(getattr(parent_msg, "elements", None) or []) + [audio_el]
        await parent_msg.update()

    @classmethod
    def make(cls, enabled: bool) -> "_TtsStreamer | None":
        """TTS aktifse streamer oluştur, değilse None."""
        if not enabled:
            return None
        voice_pref = cl.user_session.get("tts_voice", "auto")
        voice = None if voice_pref == "auto" else voice_pref
        return cls(voice=voice)


# ── Vision helpers ──


def _image_to_data(image_path: Path) -> dict:
    """Görsel dosyayı base64 encode ederek agent state formatına çevirir."""
    suffix = image_path.suffix.lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
    mime = mime_map.get(suffix, "image/png")
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return {"mime": mime, "base64": b64, "name": image_path.name}


# ── Session upload tracking ──


def _track_session_upload(filename: str) -> None:
    """Yüklenen bir dosyayı session'daki kümülatif listeye ekler (sıra korunur, tekrar yok)."""
    if not filename:
        return
    # Path traversal koruması: "../" veya mutlak yol içeren isimler reddedilir
    if "/" in filename or "\\" in filename or filename.startswith("."):
        logger.warning("Güvensiz dosya adı reddedildi: %r", filename)
        return
    uploads = cl.user_session.get("session_uploads") or []
    if filename not in uploads:
        uploads.append(filename)
        cl.user_session.set("session_uploads", uploads)


def _session_scoped_filename(name: str) -> str:
    """Qdrant source_file çakışmalarını önlemek için dosya adını oturuma bağla."""
    safe_name = Path(name).name
    stem = Path(safe_name).stem[:80] or "upload"
    suffix = Path(safe_name).suffix
    sid = str(cl.user_session.get("id") or uuid.uuid4().hex)[:8]
    return f"{sid}_{uuid.uuid4().hex[:8]}_{stem}{suffix}"


# ── URL ingest helper ──


async def _ingest_url(url: str, sess_dir: Path) -> dict | None:
    """Bir web URL'sini scrape edip RAG'a ingest eder.

    Returns:
        ingest_file() sonucu dict veya hata durumunda None.
    """
    try:
        from langchain_community.document_loaders import WebBaseLoader
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", url.split("//")[-1])[:60] + ".txt"
        dest = sess_dir / safe_name
        sess_dir.mkdir(parents=True, exist_ok=True)
        loader = WebBaseLoader(url)
        docs = loader.load()
        if not docs:
            return None
        combined = "\n\n".join(d.page_content for d in docs)
        dest.write_text(combined, encoding="utf-8")
        return await cl.make_async(ingest_file)(dest)
    except Exception as exc:
        logger.warning("URL ingest basarisiz (%s): %s", url, exc)
        return None


# ── Chat start ──


@cl.on_chat_start
async def on_chat_start():
    session_id = cl.user_session.get("id") or uuid.uuid4().hex
    sid = session_id[:8]
    logger.info(
        "[%s] session_start [llm=%s, embed=%s, qdrant=%s, ctx=%dtok, rerank=%s]",
        sid, settings.llm_model, settings.embedding_model,
        settings.qdrant_url, settings.llm_context_size, settings.use_rerank,
    )
    cl.user_session.set("_session_id_short", session_id[:8])
    session_upload_dir = settings.upload_dir / f"session_{session_id}"
    session_upload_dir.mkdir(parents=True, exist_ok=True)
    cl.user_session.set("session_upload_dir", str(session_upload_dir))

    # Temiz session state
    cl.user_session.set("chat_history", cl.user_session.get("chat_history") or [])
    cl.user_session.set("pending_vision_images", [])
    cl.user_session.set("last_vision_images", [])
    # Session boyunca yüklenen dosyaların listesi — follow-up RAG sorguları için
    cl.user_session.set("session_uploads", cl.user_session.get("session_uploads") or [])

    tts_default = False
    cl.user_session.set("tts_enabled", tts_default)
    cl.user_session.set("tts_voice", "auto")
    cl.user_session.set("temperature", settings.chat_temperature)
    cl.user_session.set("max_tokens", settings.chat_max_tokens)
    cl.user_session.set("retrieval_strategy", settings.retrieval_strategy)
    cl.user_session.set("use_rerank", settings.use_rerank)

    # Chat Settings widget'larini gonder
    await cl.ChatSettings(
        [
            Switch(
                id="tts_enabled",
                label="Sesli Yanit (TTS)",
                initial=tts_default,
                description="Aciksa her yanit ses olarak da oynatilir.",
            ),
            Select(
                id="tts_voice",
                label="TTS Sesi",
                values=["auto", "tr-TR-AhmetNeural", "tr-TR-EmelNeural", "en-US-AriaNeural", "en-US-GuyNeural"],
                initial_value="auto",
                description="'auto' dil tespiti yapar.",
            ),
            Slider(
                id="temperature",
                label="Sicaklik (Temperature)",
                initial=settings.chat_temperature,
                min=0.0,
                max=1.5,
                step=0.05,
                description="Dusuk = deterministik, yuksek = yaratici.",
            ),
            Slider(
                id="max_tokens",
                label="Max Token",
                initial=float(settings.chat_max_tokens),
                min=256,
                max=1536,
                step=128,
                description="Yanit uzunlugu limiti. MacBook local profilinde 1536 ustu onerilmez.",
            ),
            Select(
                id="retrieval_strategy",
                label="Retrieval Stratejisi",
                values=["hybrid", "similarity", "mmr", "threshold"],
                initial_value=settings.retrieval_strategy,
                description="Belge arama stratejisi.",
            ),
            Switch(
                id="use_rerank",
                label="Reranker",
                initial=settings.use_rerank,
                description="Cross-encoder reranking (daha dogru, biraz yavas).",
            ),
        ]
    ).send()

    try:
        cache_already_warm = is_mcp_tools_cache_warm()
        mcp_tools = await get_mcp_tools()
        cl.user_session.set("mcp_langchain_tools", mcp_tools)
        tool_names = [getattr(t, "name", "") for t in mcp_tools if getattr(t, "name", "")]
    except Exception as e:
        logger.warning("MCP tool preload failed: %s", e)
        cl.user_session.set("mcp_langchain_tools", [])

    resume_count = cl.user_session.get("_resume_msg_count")
    if resume_count:
        cl.user_session.set("_resume_msg_count", None)
        await cl.Message(
            content=f"💬 Önceki sohbet yüklendi — **{resume_count}** mesaj geri getirildi."
        ).send()

    # Whisper preload — ilk ses girdisinde gecikme olmasin
    asyncio.create_task(_preload_whisper())

    # Reranker preload — ilk RAG sorgusunda HuggingFace gecikme olmasin
    asyncio.create_task(_preload_reranker())

    # Embedding model preload — ilk RAG sorgusunda BGE-M3 yükleme gecikmesi olmasin
    asyncio.create_task(_preload_embeddings())



@cl.set_chat_profiles
async def set_chat_profiles():
    return [
        cl.ChatProfile(
            name=f"Frappe  ·  {settings.llm_model_name}",
            markdown_description=(
                f"**{settings.llm_model_name}** — FRAPPE\n\n"
                "Multimodal RAG Agent"
            ),
            icon="/public/logo.svg",
        ),
    ]


@cl.set_starters
async def set_starters():
    starters = [
        cl.Starter(label="📎 Dosya yükle (PDF/DOCX/XLSX/ses/görsel...)", message="/upload"),
        cl.Starter(label="🌐 URL'den belge ingest et", message="/url https://"),
        cl.Starter(label="Aktif modelleri göster", message="/models"),
    ]
    if settings.tavily_api_key:
        starters.append(cl.Starter(label="☀️ Hava durumu", message="Istanbul hava durumu bugun nasil?"))
    return starters


# ── Audio hooks ──


@cl.on_audio_start
async def on_audio_start():
    cl.user_session.set("audio_buffer", bytearray())
    cl.user_session.set("audio_mime_type", "")
    return True


@cl.on_audio_chunk
async def on_audio_chunk(chunk):
    buf = cl.user_session.get("audio_buffer", bytearray())
    if not isinstance(buf, (bytearray, bytes)):
        buf = bytearray()
    buf.extend(chunk.data)
    cl.user_session.set("audio_buffer", buf)
    cl.user_session.set("audio_mime_type", getattr(chunk, "mime_type", "") or "")


@cl.on_audio_end
async def on_audio_end():
    if not settings.stt_model:
        await cl.Message(content="STT disabled. Set STT_MODEL in .env.").send()
        return
    buf = cl.user_session.get("audio_buffer")
    if not buf:
        await cl.Message(content="No audio data received.").send()
        return

    session_dir = Path(cl.user_session.get("session_upload_dir") or settings.upload_dir)
    audio_dir = session_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / f"recording-{uuid.uuid4().hex}.wav"
    wav_data = _pcm_to_wav(bytes(buf), sample_rate=24000, channels=1, sample_width=2)
    audio_path.write_bytes(wav_data)
    cl.user_session.set("audio_buffer", bytearray())

    text = ""
    chat_history = cl.user_session.get("chat_history", [])
    msg = cl.Message(content="Transcribing audio...")
    await msg.send()
    try:
        model = _get_whisper_model()
        segments, _ = model.transcribe(str(audio_path))
        text = " ".join(s.text.strip() for s in segments).strip()
        if not text:
            msg.content = "Could not transcribe audio (empty result)."
            await msg.update()
            return

        msg.content = f"**Transcript:** {text}\n\n"
        await msg.update()

        _audio_summary = cl.user_session.get("session_summary") or ""
        lc_history = _build_lc_history(chat_history, summary=_audio_summary or None)

        # Pending görselleri kontrol et - sesli soruyla birlikte gönderilecek
        pending_images = cl.user_session.get("pending_vision_images", [])
        audio_image_data = []
        if pending_images:
            for p_str in pending_images:
                p = Path(p_str)
                if p.exists() and p.is_file():
                    audio_image_data.append(_image_to_data(p))
            cl.user_session.set("pending_vision_images", [])
            cl.user_session.set("last_vision_images", pending_images)
            if audio_image_data:
                logger.info("Audio sorusuyla birlikte %d görsel gönderiliyor", len(audio_image_data))

        audio_input_type = "image" if audio_image_data else "audio"

        answer_parts: list[str] = []
        audio_session_uploads: list[str] = list(cl.user_session.get("session_uploads") or [])
        # Ses modunda TTS her zaman aktif — streamer oluştur
        audio_tts_streamer = _TtsStreamer.make(enabled=True)
        _a_temp = float(cl.user_session.get("temperature", settings.chat_temperature))
        _a_max_tok = int(cl.user_session.get("max_tokens", settings.chat_max_tokens))
        _a_strategy = cl.user_session.get("retrieval_strategy", settings.retrieval_strategy)
        _a_rerank = bool(cl.user_session.get("use_rerank", settings.use_rerank))
        try:
            async for ev in astream_agent(
                question=text,
                chat_history=lc_history,
                input_type=audio_input_type,
                image_data=audio_image_data or None,
                session_uploads=audio_session_uploads,
                temperature=_a_temp,
                max_tokens=_a_max_tok,
                retrieval_strategy=_a_strategy,
                use_rerank=_a_rerank,
            ):
                if isinstance(ev, tuple) and len(ev) == 2 and ev[0] == "updates":
                    payload = ev[1]
                    if isinstance(payload, dict):
                        for _n, delta in payload.items():
                            if isinstance(delta, dict):
                                gen = delta.get("generation")
                                if isinstance(gen, str) and gen.strip() and not answer_parts:
                                    answer_parts = [gen]
                    continue
                if isinstance(ev, tuple) and len(ev) == 2 and ev[0] == "messages":
                    payload = ev[1]
                    chunk, meta = (payload if isinstance(payload, tuple) and len(payload) == 2
                                   else (payload, None))
                    content = getattr(chunk, "content", None)
                    if isinstance(content, str) and content and isinstance(chunk, AIMessageChunk):
                        node = ""
                        if meta:
                            node = str(meta.get("langgraph_node") or meta.get("node") or "")
                        if node not in {"router", "rewriter", "grader"}:
                            await msg.stream_token(content)
                            answer_parts.append(content)
                            audio_tts_streamer.feed(content)
        except Exception:
            fallback = await arun_agent(
                question=text,
                chat_history=lc_history,
                input_type=audio_input_type,
                image_data=audio_image_data or None,
                session_uploads=audio_session_uploads,
                temperature=_a_temp,
                max_tokens=_a_max_tok,
                retrieval_strategy=_a_strategy,
                use_rerank=_a_rerank,
            )
            answer_parts = [fallback]

        answer = "".join(answer_parts).strip() or "An error occurred, please try again."
        msg.content = f"**Transcript:** {text}\n\n{answer}"
        await msg.update()

        # Streaming sırasında başlatılan TTS'i tamamla ve gönder
        await audio_tts_streamer.send_to(msg)

        chat_history.append({"role": "user", "content": text})
        chat_history.append({"role": "assistant", "content": answer})
        if len(chat_history) >= _SUMMARY_TRIGGER:
            thread_id = cl.user_session.get("id")
            chat_history, new_summary = await _summarize_and_compress_history(
                chat_history, thread_id
            )
            if new_summary:
                cl.user_session.set("session_summary", new_summary)
        else:
            chat_history = _trim_chat_history(chat_history)
        cl.user_session.set("chat_history", chat_history)
    except Exception as e:
        logger.error("STT error: %s", e, exc_info=True)
        msg.content = f"Audio processing error: {e}"
        await msg.update()
        if text:
            chat_history.append({"role": "user", "content": text})
            chat_history.append({"role": "assistant", "content": f"[Ses işleme hatası: {e}]"})
            cl.user_session.set("chat_history", _trim_chat_history(chat_history))


# ── Helpers ──


# ── Upload limitleri ──
_MAX_FILES_PER_MESSAGE = 5    # Tek mesajda kabul edilecek maksimum dosya sayısı
_MAX_FILE_SIZE_MB = 20        # Tek dosya için maksimum boyut (MB)

MAX_HISTORY_TURNS = 20

# Hard cap on stored chat_history dicts — prevents unbounded memory growth
# in very long sessions. Kept larger than MAX_HISTORY_TURNS so session resume
# has more context, but still bounded.
_MAX_STORED_MESSAGES = 100

# Approximate character budget for LLM context window history.
# Per-user context = LLAMA_CTX_SIZE / LLAMA_PARALLEL (e.g. 16384/4 = 4096 tokens).
# System prompt + RAG chunks already consume ~2000-3000 tokens.
# Remaining ~1000-2000 tokens for history → ~4000-8000 chars (~4 chars/token).
# This guard prevents sending more history than the context window can hold.
_MAX_HISTORY_CHARS = 6000


# Uzun süreli bellek: bu eşiği geçince eski mesajlar LLM ile özetlenir.
# chat_history her turu 2 kayıtla temsil eder (user + assistant);
# 40 mesaj = 20 konuşma turu.
_SUMMARY_TRIGGER = 40       # Mesaj sayısı eşiği (2 × tur sayısı)
_SUMMARY_KEEP_RECENT = 10   # Özetlemeden sonra canlı tutulan son mesaj sayısı (5 tur)



async def _summarize_and_compress_history(
    chat_history: list[dict],
    thread_id: str | None,
) -> tuple[list[dict], str]:
    """Eski mesajları LLM ile özetler, chat_history'yi sıkıştırır.

    Returns (compressed_history, summary_text).
    compressed_history = son _SUMMARY_KEEP_RECENT mesaj.
    """
    from src.rag.llm import get_rag_llm

    old_msgs = chat_history[:-_SUMMARY_KEEP_RECENT]
    recent_msgs = chat_history[-_SUMMARY_KEEP_RECENT:]

    lines = []
    for m in old_msgs:
        role = m.get("role", "")
        content = (m.get("content", "") or "")[:600]
        if role == "user":
            lines.append(f"Kullanıcı: {content}")
        elif role == "assistant":
            lines.append(f"Asistan: {content}")

    summary = ""
    try:
        llm = get_rag_llm()
        prompt = (
            "Aşağıdaki konuşmayı, ileride bağlam olarak kullanılacak şekilde "
            "kısa ve bilgi yoğun biçimde özetle. Önemli kararları, gerçekleri "
            "ve bağlamı koru. Maksimum 300 kelime.\n\n"
            f"Konuşma:\n{chr(10).join(lines)}\n\nÖzet:"
        )
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        summary = response.content.strip()
        logger.info(
            "Konuşma özetlendi: %d→%d mesaj, özet=%dch",
            len(chat_history), len(recent_msgs), len(summary),
        )
    except Exception as exc:
        logger.warning("Özetleme başarısız: %s", exc)

    if summary and thread_id:
        try:
            await _get_shared_data_layer().patch_thread_metadata(thread_id, {"summary": summary})
        except Exception as exc:
            logger.warning("Thread metadata güncellenemedi: %s", exc)

    return recent_msgs, summary


def _trim_chat_history(chat_history: list[dict]) -> list[dict]:
    """Trim in-memory chat_history to _MAX_STORED_MESSAGES.

    Called after every append to prevent unbounded growth. The sliding window
    in _build_lc_history handles LLM token budget; this guards RAM.
    """
    if len(chat_history) > _MAX_STORED_MESSAGES:
        return chat_history[-_MAX_STORED_MESSAGES:]
    return chat_history


def _build_lc_history(chat_history: list[dict], summary: str | None = None) -> list:
    """Son MAX_HISTORY_TURNS cift (user+assistant) mesaji LangChain formatina cevirir.

    Uc katmanli koruma:
    1. Turn limiti: son MAX_HISTORY_TURNS cift mesaj
    2. Per-message truncation: her mesaj MAX_MSG_CHARS ile kesilir
    3. Total char budget: toplam karakter sayisi _MAX_HISTORY_CHARS'i asmaz
       (per-user context window'a sigdirmak icin)

    summary verilirse en başa SystemMessage olarak eklenir (uzun süreli bellek).
    """
    MAX_MSG_CHARS = 1500
    lc = []
    for m in chat_history:
        content = m.get("content", "") if isinstance(m, dict) else ""
        if len(content) > MAX_MSG_CHARS:
            content = content[:MAX_MSG_CHARS] + "…"
        role = m.get("role", "") if isinstance(m, dict) else ""
        if role == "user":
            lc.append(HumanMessage(content=content))
        elif role == "assistant":
            lc.append(AIMessage(content=content))
    max_msgs = MAX_HISTORY_TURNS * 2
    if len(lc) > max_msgs:
        lc = lc[-max_msgs:]

    # Total character budget guard — trim oldest user+assistant PAIR to preserve role order
    total_chars = sum(len(m.content) for m in lc)
    while total_chars > _MAX_HISTORY_CHARS and len(lc) >= 4:
        # Always remove a pair (user + assistant) so history never starts with assistant
        removed_u = lc.pop(0)
        removed_a = lc.pop(0)
        total_chars -= len(removed_u.content) + len(removed_a.content)

    if summary:
        lc.insert(0, SystemMessage(content=f"Önceki konuşma özeti:\n{summary}"))

    return lc


def _build_source_elements(docs) -> list[cl.Text]:
    """RAG kaynaklarından minimal Chainlit Text elementleri oluşturur (side panel)."""
    if not docs:
        return []

    seen: set[str] = set()
    elements: list[cl.Text] = []
    for doc in docs:
        meta = getattr(doc, "metadata", None) or {}
        src = meta.get("display_name") or meta.get("source_file", meta.get("source", ""))
        page = meta.get("page", "")

        src_short = Path(src).name if src and ("/" in src or "\\" in src) else (src or "Bilinmeyen kaynak")
        dedup_key = f"{src_short}:{page}"
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        page_str = f" · Sayfa {page}" if page and str(page) not in {"", "?"} else ""
        num = len(elements) + 1
        label = f"[{num}] {src_short}{page_str}"
        elements.append(cl.Text(name=label, content=src_short + page_str, display="side"))

    return elements


_STREAM_CHUNK_TIMEOUT = 60  # saniye — bu sürede chunk gelmezse fallback'e düşer


async def _timed_stream(gen, timeout: float):
    """Async generator'ı her chunk arasında timeout ile sarar."""
    try:
        while True:
            try:
                item = await asyncio.wait_for(gen.__anext__(), timeout=timeout)
                yield item
            except StopAsyncIteration:
                return
    except asyncio.TimeoutError:
        logger.warning("astream_agent: %ss içinde chunk gelmedi, fallback tetikleniyor", timeout)
        raise


# ── Main message handler ──


@cl.on_message
async def on_message(message: cl.Message):
    import time as _time
    _t0 = _time.perf_counter()
    question = message.content
    sid = cl.user_session.get("_session_id_short", "?")
    chat_history = cl.user_session.get("chat_history", [])
    n_attach = len(getattr(message, "elements", None) or [])
    logger.info(
        "[%s] → msg [len=%dch, history=%d, attachments=%d] '%.80s'",
        sid, len(question), len(chat_history), n_attach, question,
    )
    thinking_msg = cl.Message(content="")
    await thinking_msg.send()

    try:
        cmd = (question or "").strip()

        # ── Slash commands ──

        if cmd.lower().startswith("/models"):
            thinking_msg.content = (
                f"**LLM:** `{settings.llm_model}`\n"
                f"**Vision:** `{settings.vision_model or 'disabled'}`\n"
                f"**Embedding:** `{settings.embedding_model}`\n"
                f"**STT:** `{settings.stt_model or 'disabled'}`\n"
                f"**Reranker:** `{settings.reranker_model if settings.use_rerank else 'disabled'}`\n"
                f"**Strategy:** `{settings.retrieval_strategy}`"
            )
            await thinking_msg.update()
            return

        if cmd.lower().startswith("/tts"):
            parts = cmd.split(None, 1)
            if len(parts) == 2:
                voice_arg = parts[1].strip()
                # Kisaltmadan tam ses adına donustur
                _VOICE_MAP = {
                    "ahmet": "tr-TR-AhmetNeural", "ahmetneural": "tr-TR-AhmetNeural",
                    "emel": "tr-TR-EmelNeural", "emelneural": "tr-TR-EmelNeural",
                    "aria": "en-US-AriaNeural", "arianeural": "en-US-AriaNeural",
                    "guy": "en-US-GuyNeural", "guyneural": "en-US-GuyNeural",
                    "auto": "auto",
                }
                voice_name = _VOICE_MAP.get(voice_arg.lower(), voice_arg)
                cl.user_session.set("tts_voice", voice_name)
                cl.user_session.set("tts_enabled", True)
                thinking_msg.content = f"TTS sesi ayarlandi: `{voice_name}` (TTS aktif)"
            else:
                current = cl.user_session.get("tts_voice", "auto")
                thinking_msg.content = (
                    f"Mevcut TTS sesi: `{current}`\n\n"
                    "Kullanim: `/tts <ses>` — Secenekler:\n"
                    "- `AhmetNeural` (Türkçe erkek)\n"
                    "- `EmelNeural` (Türkçe kadin)\n"
                    "- `AriaNeural` (Ingilizce kadin)\n"
                    "- `GuyNeural` (Ingilizce erkek)\n"
                    "- `auto` (otomatik dil tespiti)"
                )
            await thinking_msg.update()
            return

        if cmd.lower().startswith("/whisper download"):
            dl_msg = cl.Message(content=f"Downloading Whisper: `{settings.stt_model}`...")
            await dl_msg.send()
            try:
                _get_whisper_model()
                dl_msg.content = f"Whisper ready: `{settings.stt_model}`"
            except Exception as e:
                dl_msg.content = f"Whisper error: {e}"
            await dl_msg.update()
            thinking_msg.content = ""
            await thinking_msg.update()
            return

        # ── File attachments ──

        upload_candidates = []
        for attr in ("elements", "files", "attachments"):
            items = getattr(message, attr, None) or []
            if not isinstance(items, list):
                continue
            for el in items:
                el_path = getattr(el, "path", None) or (el.get("path") if isinstance(el, dict) else None)
                if not el_path:
                    continue
                try:
                    p = Path(el_path)
                except TypeError:
                    continue
                if p.exists() and p.is_file():
                    upload_candidates.append(el)

        ingested_filenames: list[str] = []
        image_paths: list[Path] = []

        sess_dir = Path(cl.user_session.get("session_upload_dir") or settings.upload_dir)

        if upload_candidates:
            # Dosya sayısı limiti
            if len(upload_candidates) > _MAX_FILES_PER_MESSAGE:
                thinking_msg.content = (
                    f"❌ Tek seferde en fazla **{_MAX_FILES_PER_MESSAGE}** dosya yüklenebilir. "
                    f"({len(upload_candidates)} dosya gönderildi)"
                )
                await thinking_msg.update()
                return

            status_lines = []
            for f in upload_candidates:
                f_path = getattr(f, "path", None) or (f.get("path") if isinstance(f, dict) else None)
                src = Path(f_path)
                name = getattr(f, "name", None) or (f.get("name") if isinstance(f, dict) else None) or src.name

                # Dosya boyutu limiti
                try:
                    size_mb = src.stat().st_size / (1024 * 1024)
                    if size_mb > _MAX_FILE_SIZE_MB:
                        status_lines.append(
                            f"❌ **{name}** — {size_mb:.1f} MB, limit {_MAX_FILE_SIZE_MB} MB."
                        )
                        continue
                except OSError:
                    pass

                mime_val = getattr(f, "mime", None) or (f.get("mime") if isinstance(f, dict) else None) or ""
                mime_str = str(mime_val)
                original_name = Path(name).name
                dest_name = (
                    original_name
                    if (mime_str.startswith("image/") or Path(original_name).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
                    else _session_scoped_filename(original_name)
                )
                dest = sess_dir / dest_name
                sess_dir.mkdir(parents=True, exist_ok=True)
                await asyncio.to_thread(dest.write_bytes, src.read_bytes())

                suffix_lower = dest.suffix.lower()

                if mime_str.startswith("image/") or suffix_lower in {".png", ".jpg", ".jpeg", ".webp"}:
                    image_paths.append(dest)
                    status_lines.append(f"Gorsel alindi: **{dest.name}**")
                    continue

                # Ses dosyası — Whisper ile transcribe et
                if mime_str.startswith("audio/") or suffix_lower in {".mp3", ".wav", ".ogg", ".m4a", ".flac"}:
                    try:
                        model = _get_whisper_model()
                        segments, _ = model.transcribe(str(dest))
                        transcript = " ".join(s.text.strip() for s in segments).strip()
                        if transcript:
                            txt_path = dest.with_suffix(".txt")
                            txt_path.write_text(transcript, encoding="utf-8")
                            result = await cl.make_async(ingest_file)(txt_path, display_name=f"{original_name}.txt")
                            ingested_filenames.append(txt_path.name)
                            _track_session_upload(txt_path.name)
                            status_lines.append(
                                f"**{original_name}** → transcribe + indekslendi (**{result['chunk_count']}** chunk)."
                            )
                        else:
                            status_lines.append(f"**{dest.name}** → transcribe edilemedi.")
                    except Exception as exc:
                        status_lines.append(f"**{dest.name}** → STT hatasi: {exc}")
                    continue

                result = await cl.make_async(ingest_file)(dest, display_name=original_name)
                ingested_filenames.append(dest.name)
                _track_session_upload(dest.name)
                status_lines.append(f"**{result.get('display_name', original_name)}** indekslendi — **{result['chunk_count']}** chunk.")

        if image_paths:
            elements = [cl.Image(path=str(p), name=p.name, display="inline") for p in image_paths]
            # Görselleri session'da sakla - sesli soru için kullanılabilir
            cl.user_session.set("pending_vision_images", [str(p) for p in image_paths])
            
            # Soru boş veya anlamsızsa, sadece görseli sakla ve dön
            # Sesli soru (on_audio_end) bu görseli kullanacak
            question_text = question.strip().lower()
            is_trivial_question = (
                not question_text or
                question_text in {"", ".", "upload", "/upload"} or
                len(question_text) < 3
            )
            
            if is_trivial_question:
                await cl.Message(
                    content="Image(s) received. Ask me about it using text or voice!",
                    elements=elements
                ).send()
                thinking_msg.content = ""
                await thinking_msg.update()
                logger.info("Görsel pending olarak saklandı, sesli/yazılı soru bekleniyor: %s", 
                           [p.name for p in image_paths])
                return
            else:
                # Soru var, görseli hemen işleyeceğiz - pending'i temizle ama last'da sakla
                await cl.Message(content="Image(s) received.", elements=elements).send()
                cl.user_session.set("pending_vision_images", [])
                cl.user_session.set("last_vision_images", [str(p) for p in image_paths])

        if upload_candidates and status_lines:
            thinking_msg.content = "\n".join(status_lines)
            await thinking_msg.update()

        # ── /url command — URL'den belge ingest ──

        if cmd.lower().startswith("/url "):
            import urllib.parse as _urlparse
            raw_url = cmd[5:].strip()
            _parsed = _urlparse.urlparse(raw_url)
            if _parsed.scheme not in {"http", "https"} or not _parsed.netloc:
                thinking_msg.content = "Gecersiz URL. Ornek: `/url https://example.com/makale`"
                await thinking_msg.update()
                return
            url_msg = cl.Message(content=f"Indiriliyor: `{raw_url}`...")
            await url_msg.send()
            result = await _ingest_url(raw_url, sess_dir)
            if result:
                _track_session_upload(result.get("file_name", ""))
                url_msg.content = f"**{raw_url}** indekslendi — **{result['chunk_count']}** chunk."
            else:
                url_msg.content = f"URL ingest basarisiz: `{raw_url}`"
            await url_msg.update()
            thinking_msg.content = ""
            await thinking_msg.update()
            return

        # ── /upload command ──

        if question.strip().lower() in {"/upload", "upload"}:
            files = await cl.AskFileMessage(
                content="Dosya yukle (PDF, DOCX, TXT, MD, XLSX, CSV veya ses dosyasi MP3/WAV/OGG).",
                accept=[
                    "application/pdf",
                    "text/plain",
                    "text/markdown",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "text/csv",
                    "audio/mpeg", "audio/mp3", "audio/wav", "audio/ogg", "audio/x-wav",
                ],
                max_size_mb=_MAX_FILE_SIZE_MB, timeout=180,
            ).send()
            if not files:
                thinking_msg.content = "Dosya yuklenmedi."
                await thinking_msg.update()
                return
            results_text = []
            for f in files:
                src = Path(getattr(f, "path"))
                original_name = Path(getattr(f, "name", src.name) or src.name).name
                dest = sess_dir / _session_scoped_filename(original_name)
                sess_dir.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(src.read_bytes())
                suffix = dest.suffix.lower()
                # Ses dosyasi → Whisper STT → TXT olarak ingest
                if suffix in {".mp3", ".wav", ".ogg", ".m4a", ".flac"}:
                    try:
                        model = _get_whisper_model()
                        segments, _ = model.transcribe(str(dest))
                        transcript = " ".join(s.text.strip() for s in segments).strip()
                        if transcript:
                            txt_path = dest.with_suffix(".txt")
                            txt_path.write_text(transcript, encoding="utf-8")
                            result = await cl.make_async(ingest_file)(txt_path, display_name=f"{original_name}.txt")
                            _track_session_upload(txt_path.name)
                            results_text.append(
                                f"**{original_name}** → transcribe edildi + indekslendi "
                                f"(**{result['chunk_count']}** chunk, {len(transcript)} karakter)."
                            )
                        else:
                            results_text.append(f"**{dest.name}** → ses transcribe edilemedi.")
                    except Exception as exc:
                        results_text.append(f"**{dest.name}** → STT hatasi: {exc}")
                else:
                    result = await cl.make_async(ingest_file)(dest, display_name=original_name)
                    _track_session_upload(result.get("file_name", dest.name))
                    results_text.append(f"**{result.get('display_name', original_name)}** — **{result['chunk_count']}** chunk.")
            thinking_msg.content = "\n".join(results_text) + "\n\nBelgeler hakkinda soru sorabilirsin."
            await thinking_msg.update()
            return

        # ── Agent streaming ──

        session_summary = cl.user_session.get("session_summary") or ""
        lc_history = _build_lc_history(chat_history, summary=session_summary or None)
        source_filter = ingested_filenames[0] if len(ingested_filenames) == 1 else ""
        session_uploads: list[str] = list(cl.user_session.get("session_uploads") or [])

        # Kullanıcı önce soruyu sordu, sonra dosyayı tek başına yükledi:
        # yüklemeyle gelen mesaj boş/önemsiz ise, en son sorulan soruyu yeniden kullan.
        if ingested_filenames:
            stripped = (question or "").strip()
            is_trivial = (not stripped) or len(stripped) < 3 or stripped.lower() in {
                ".", "upload", "/upload", "ok", "tamam",
            }
            if is_trivial:
                pending_q = (cl.user_session.get("pending_question") or "").strip()
                last_user_q = ""
                for m in reversed(chat_history or []):
                    role = (m.get("role") if isinstance(m, dict) else getattr(m, "role", "")) or ""
                    if role in {"user", "human"}:
                        last_user_q = (m.get("content") if isinstance(m, dict) else getattr(m, "content", "")) or ""
                        break
                reuse_q = pending_q or last_user_q
                if reuse_q:
                    logger.info("Dosya tek başına yüklendi → son soru yeniden kullanılıyor: %s", reuse_q[:80])
                    question = reuse_q
                    cl.user_session.set("pending_question", "")

        # Eğer image_paths varsa zaten yukarıda pending kontrolü yapıldı
        # Buraya geldiyse ya soru vardı (pending temizlendi) ya da görsel yoktu
        agent_image_data = [_image_to_data(p) for p in image_paths] if image_paths else []

        # Yeni görsel yüklendiyse reuse counter sıfırla
        if image_paths:
            cl.user_session.set("_vision_reuse_left", 4)
        elif not agent_image_data:
            # Yeni görsel yok — önceki görseli yalnızca görsele açıkça atıf varsa kullan.
            # Aksi halde alakasız matematik/sohbet soruları vision yoluna sapıyor.
            reuse_left = cl.user_session.get("_vision_reuse_left", 0)
            if reuse_left > 0 and _is_vision_followup(question):
                last_imgs = cl.user_session.get("last_vision_images") or []
                valid_imgs = [Path(p) for p in last_imgs if Path(p).exists()]
                if valid_imgs:
                    agent_image_data = [_image_to_data(p) for p in valid_imgs]
                    cl.user_session.set("_vision_reuse_left", reuse_left - 1)
                else:
                    cl.user_session.set("_vision_reuse_left", 0)
            elif reuse_left > 0:
                logger.info("Vision follow-up değil; önceki görsel bu turda kullanılmadı [q=%.60s]", question)
            else:
                cl.user_session.set("last_vision_images", [])

        agent_input_type = "image" if agent_image_data else "text"
        
        thinking_msg.content = ""
        await thinking_msg.update()

        final_parts: list[str] = []
        latest_full_generation = ""
        last_route: str | None = None
        last_documents: list = []
        tts_streamer = _TtsStreamer.make(cl.user_session.get("tts_enabled", False))

        _sess_temp = float(cl.user_session.get("temperature", settings.chat_temperature))
        _sess_max_tok = int(cl.user_session.get("max_tokens", settings.chat_max_tokens))
        _sess_strategy = cl.user_session.get("retrieval_strategy", settings.retrieval_strategy)
        _sess_rerank = bool(cl.user_session.get("use_rerank", settings.use_rerank))

        try:
            async for ev in _timed_stream(
                astream_agent(
                    question=question,
                    chat_history=lc_history,
                    source_filter=source_filter,
                    image_data=agent_image_data or None,
                    input_type=agent_input_type,
                    session_uploads=session_uploads,
                    temperature=_sess_temp,
                    max_tokens=_sess_max_tok,
                    retrieval_strategy=_sess_strategy,
                    use_rerank=_sess_rerank,
                ),
                _STREAM_CHUNK_TIMEOUT,
            ):
                if isinstance(ev, tuple) and len(ev) == 2 and ev[0] == "updates":
                    payload = ev[1]
                    if isinstance(payload, dict):
                        for _node_name, delta in payload.items():
                            if isinstance(delta, dict):
                                if delta.get("route"):
                                    last_route = str(delta["route"])
                                if delta.get("documents") is not None:
                                    last_documents = list(delta["documents"])
                                gen = delta.get("generation")
                                if isinstance(gen, str) and gen.strip():
                                    latest_full_generation = gen
                                if _node_name == "grader" and delta.get("relevance") == "no":
                                    async with cl.Step(name="Belgeler yetersiz — web araması devreye giriyor", type="tool") as _step:
                                        _step.output = "RAG sonuçları soruyu yanıtlamıyor, web araması başlatılıyor."
                    continue

                def _should_stream(chunk: object, meta: dict | None, content: str) -> bool:
                    if not content:
                        return False
                    if not isinstance(chunk, AIMessageChunk):
                        return False
                    c = content.strip()
                    if c.startswith("{") and '"route"' in c:
                        return False
                    node = ""
                    if meta:
                        node = str(meta.get("langgraph_node") or meta.get("node") or
                                   meta.get("name") or meta.get("runnable_name") or "")
                    if node in {"router", "rewriter", "grader"}:
                        return False
                    return True

                if isinstance(ev, tuple) and len(ev) == 2 and ev[0] == "messages":
                    payload = ev[1]
                    chunk, meta = (payload if isinstance(payload, tuple) and len(payload) == 2
                                   else (payload, None))
                    content = getattr(chunk, "content", None)
                    if isinstance(content, str) and _should_stream(chunk, meta, content):
                        await thinking_msg.stream_token(content)
                        final_parts.append(content)
                        if tts_streamer:
                            tts_streamer.feed(content)
                    continue

                if isinstance(ev, dict):
                    gen = ev.get("generation")
                    if isinstance(gen, str) and gen.strip():
                        latest_full_generation = gen
                    continue
        except Exception:
            answer = await arun_agent(
                question=question,
                chat_history=lc_history,
                source_filter=source_filter,
                image_data=agent_image_data or None,
                input_type=agent_input_type,
                session_uploads=session_uploads,
                temperature=_sess_temp,
                max_tokens=_sess_max_tok,
                retrieval_strategy=_sess_strategy,
                use_rerank=_sess_rerank,
            )
            final_parts = [answer]
            for i in range(0, len(answer), 24):
                await thinking_msg.stream_token(answer[i:i + 24])
            # Feed fallback answer into TTS streamer so audio isn't silently dropped
            if tts_streamer:
                tts_streamer.feed(answer)

        answer = "".join(final_parts).strip() or latest_full_generation.strip()

        # Streaming boş geldiyse ainvoke fallback
        if not answer:
            logger.warning("Stream içerik üretemedi, ainvoke fallback çalışıyor")
            try:
                answer = await arun_agent(
                    question=question,
                    chat_history=lc_history,
                    source_filter=source_filter,
                    image_data=agent_image_data or None,
                    input_type=agent_input_type,
                    session_uploads=session_uploads,
                    temperature=_sess_temp,
                    max_tokens=_sess_max_tok,
                    retrieval_strategy=_sess_strategy,
                    use_rerank=_sess_rerank,
                )
            except Exception as exc:
                logger.error("Fallback ainvoke de başarısız: %s", exc)
            answer = (answer or "").strip() or "Bir hata oluştu, lütfen tekrar deneyin."

        # RAG kaynakları — Chainlit Text elementleri olarak ekle (açılır/kapanır)
        source_elements: list[cl.Text] = []
        if last_route == "rag" and last_documents:
            source_elements = _build_source_elements(last_documents)

        thinking_msg.content = answer
        if source_elements:
            thinking_msg.elements = source_elements
        await thinking_msg.update()

        # TTS — streamer LLM üretirken paralelde sentezlemiş olabilir
        if tts_streamer:
            await tts_streamer.send_to(thinking_msg)

        # Action butonları — yalnızca TTS aktif değilse ses ikonu göster
        if not cl.user_session.get("tts_enabled", False):
            thinking_msg.actions = [
                cl.Action(name="action_tts", payload={"answer": answer}, label="🔊"),
            ]
            await thinking_msg.update()

        logger.info(
            "[%s] ← msg [route=%s, ans_len=%dch, total_t=%.3fs]",
            sid, last_route or "?", len(answer), _time.perf_counter() - _t0,
        )
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": answer})

        if len(chat_history) >= _SUMMARY_TRIGGER:
            thread_id = cl.user_session.get("id")
            chat_history, new_summary = await _summarize_and_compress_history(
                chat_history, thread_id
            )
            if new_summary:
                cl.user_session.set("session_summary", new_summary)
        else:
            chat_history = _trim_chat_history(chat_history)

        cl.user_session.set("chat_history", chat_history)

    except Exception as e:
        logger.error(
            "[%s] ✗ msg [err=%s, total_t=%.3fs]",
            sid, type(e).__name__, _time.perf_counter() - _t0, exc_info=True,
        )
        thinking_msg.content = f"Error: {e}"
        await thinking_msg.update()
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": f"[Hata oluştu: {e}]"})
        cl.user_session.set("chat_history", _trim_chat_history(chat_history))


@cl.action_callback("action_tts")
async def on_action_tts(action: cl.Action):
    """🔊 ikonuna basinca secili yaniti TTS ile seslendir."""
    answer = action.payload.get("answer", "")
    if not answer:
        return
    voice_pref = cl.user_session.get("tts_voice", "auto")
    voice = None if voice_pref == "auto" else voice_pref
    audio_bytes = await tts_synthesize(answer, voice=voice)
    if audio_bytes:
        tmp_path = await _write_audio_tmp(audio_bytes)
        await cl.Message(
            content="",
            elements=[cl.Audio(path=tmp_path, name="response.mp3", display="inline")],
        ).send()
    else:
        await cl.Message(content="TTS kullanılamıyor. `edge-tts` kurulu mu? `uv add edge-tts`").send()


@cl.on_settings_update
async def on_settings_update(settings_dict: dict):
    """Kullanici ayar panelinden bir degistirdiginde session state'i guncelle."""
    tts_now = bool(settings_dict.get("tts_enabled", False))
    tts_was = cl.user_session.get("tts_enabled", False)
    tts_voice = settings_dict.get("tts_voice", "auto")
    temp = float(settings_dict.get("temperature", settings.chat_temperature))
    max_tok = int(settings_dict.get("max_tokens", settings.chat_max_tokens))
    strategy = settings_dict.get("retrieval_strategy", settings.retrieval_strategy)
    rerank = bool(settings_dict.get("use_rerank", settings.use_rerank))

    cl.user_session.set("tts_enabled", tts_now)
    cl.user_session.set("tts_voice", tts_voice)
    cl.user_session.set("temperature", temp)
    cl.user_session.set("max_tokens", max_tok)
    cl.user_session.set("retrieval_strategy", strategy)
    cl.user_session.set("use_rerank", rerank)

    lines: list[str] = []
    if tts_now != tts_was:
        lines.append("🔊 Sesli yanıt **aktif**" if tts_now else "🔇 Sesli yanıt **kapalı**")
    if tts_now and tts_voice != "auto":
        lines.append(f"🎤 TTS sesi: **{tts_voice}**")
    lines.append(f"🌡️ Sıcaklık: **{temp}** · Max token: **{max_tok}**")
    lines.append(f"🔍 Strateji: **{strategy}** · Reranker: **{'açık' if rerank else 'kapalı'}**")
    await cl.Message(content="\n".join(lines)).send()

    sid = cl.user_session.get("_session_id_short", "?")
    logger.info(
        "[%s] settings_update [tts=%s, voice=%s, temp=%.2f, max_tokens=%d, strategy=%s, rerank=%s]",
        sid, tts_now, tts_voice, temp, max_tok, strategy, rerank,
    )


@cl.on_stop
async def on_stop():
    logger.info("User stopped generation.")


@cl.on_chat_end
async def on_chat_end():
    import shutil

    sid = cl.user_session.get("_session_id_short", "?")
    session_uploads = cl.user_session.get("session_uploads") or []
    chat_history = cl.user_session.get("chat_history") or []
    n_turns = len(chat_history) // 2

    if session_uploads:
        try:
            from src.rag.vectorstore import get_hybrid_store
            await asyncio.to_thread(get_hybrid_store().delete_by_source, session_uploads)
            logger.info(
                "[%s] session_cleanup: qdrant [files=%d, %s]",
                sid, len(session_uploads), session_uploads,
            )
        except Exception as exc:
            logger.warning("[%s] session_cleanup: qdrant_error [%s]", sid, exc)

    session_dir = cl.user_session.get("session_upload_dir")
    if session_dir:
        p = Path(session_dir)
        if p.exists() and p.is_dir():
            try:
                shutil.rmtree(p)
                logger.info("[%s] session_cleanup: upload_dir [%s]", sid, p)
            except Exception as exc:
                logger.warning("[%s] session_cleanup: rmtree_error [%s]", sid, exc)

    logger.info("[%s] session_end [turns=%d, uploads=%d]", sid, n_turns, len(session_uploads))
