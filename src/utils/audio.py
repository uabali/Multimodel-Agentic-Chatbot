"""
Audio processing and speech streaming utilities.

Consolidates PCM-to-WAV conversion, temporary audio file caching,
asynchronous Whisper STT preloading, and Edge TTS real-time streaming class.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import re
import tempfile
import wave
from typing import Any

import chainlit as cl

from src.config import settings
from src.tts import synthesize as tts_synthesize

logger = logging.getLogger(__name__)

_whisper_model = None
_whisper_loading = False


def get_whisper_model() -> Any:
    """Lazy load and cache the Whisper STT model instance (Singleton)."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    
    from faster_whisper import WhisperModel
    logger.info("Loading Whisper model '%s' (device=cpu, compute_type=int8)...", settings.stt_model)
    _whisper_model = WhisperModel(settings.stt_model, device="cpu", compute_type="int8")
    logger.info("Whisper model '%s' loaded.", settings.stt_model)
    return _whisper_model


async def preload_whisper() -> None:
    """Preload the Whisper STT model in a background thread to avoid blocking startup."""
    global _whisper_loading
    if _whisper_model is not None or _whisper_loading or not settings.stt_model:
        return
    _whisper_loading = True
    try:
        await asyncio.to_thread(get_whisper_model)
    except Exception as exc:
        logger.warning("Whisper preload failed (will retry on first use): %s", exc)
    finally:
        _whisper_loading = False


def pcm_to_wav(pcm_data: bytes, sample_rate: int = 24000, channels: int = 1, sample_width: int = 2) -> bytes:
    """Convert raw PCM byte array to standard WAV format for faster_whisper compatibility."""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    return wav_buffer.getvalue()


async def write_audio_tmp(audio_bytes: bytes) -> str:
    """Write raw audio bytes to a temporary MP3 file in a session-scoped directory.

    Runs in a non-blocking background thread.
    """
    try:
        session_dir = cl.user_session.get("session_upload_dir")
    except Exception:
        session_dir = None

    def _write() -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False, dir=session_dir or None)
        tmp.write(audio_bytes)
        tmp.flush()
        path = tmp.name
        tmp.close()
        return path

    path = await asyncio.to_thread(_write)
    schedule_audio_tmp_cleanup(path)
    return path


def schedule_audio_tmp_cleanup(path: str, delay_seconds: float = 3600.0) -> None:
    """Best-effort cleanup for temporary audio files after the UI has served them."""
    async def _cleanup() -> None:
        await asyncio.sleep(delay_seconds)
        try:
            await asyncio.to_thread(os.unlink, path)
        except FileNotFoundError:
            return
        except Exception as exc:
            logger.debug("Temporary audio cleanup failed for %s: %s", path, exc)

    try:
        asyncio.create_task(_cleanup())
    except RuntimeError:
        logger.debug("No running event loop; skipping temp audio cleanup for %s", path)


class TtsStreamer:
    """Orchestrates parallel TTS synthesis during LLM streaming.

    Synthesizes the first sentence group in the background while the LLM is
    still generating, combining it with subsequent chunks at the end for single-file playback.
    """

    _SENTENCE_END = re.compile(r"(?<=[.!?\n])\s")
    _MIN_FIRST_CHARS = 150  # Character threshold to start background synthesis

    def __init__(self, voice: str | None) -> None:
        self._voice = voice
        self._buf = ""
        self._first_task: asyncio.Task[bytes] | None = None
        self._split_pos = 0

    def feed(self, chunk: str) -> None:
        """Feed a generated chunk into the buffer; starts background TTS when threshold is met."""
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
        """Finalize all background TTS segments, join them into a single audio file, and attach to message."""
        remaining = self._buf[self._split_pos :].strip()

        first_audio = await self._first_task if self._first_task else None
        second_audio = (
            await tts_synthesize(remaining, voice=self._voice) if remaining else None
        )

        if first_audio and second_audio:
            combined = await tts_synthesize(self._buf.strip(), voice=self._voice)
        else:
            combined = first_audio or second_audio or b""
        if not combined:
            return

        tmp_path = await write_audio_tmp(combined)
        audio_el = cl.Audio(path=tmp_path, name="response.mp3", display="inline")
        parent_msg.elements = list(getattr(parent_msg, "elements", None) or []) + [audio_el]
        await parent_msg.update()

    @classmethod
    def make(cls, enabled: bool) -> TtsStreamer | None:
        """Factory method to instantiate TtsStreamer if TTS setting is enabled."""
        if not enabled:
            return None
        voice_pref = cl.user_session.get("tts_voice", "auto")
        voice = None if voice_pref == "auto" else voice_pref
        return cls(voice=voice)
