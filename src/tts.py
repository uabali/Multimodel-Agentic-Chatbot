"""
TTS (Text-to-Speech) helper — edge-tts backend.

edge-tts uses Microsoft Azure Neural TTS under the hood (free, no API key).
Voices: tr-TR-AhmetNeural (TR male), tr-TR-EmelNeural (TR female),
        en-US-AriaNeural (EN female), en-US-GuyNeural (EN male).

Usage:
    wav_bytes = await synthesize("Merhaba!")
    # Returns None if TTS is disabled or edge-tts is not installed.

Auto language detection:
    synthesize() picks a Turkish or English voice based on a simple heuristic.
    Pass voice= explicitly to override.
"""

from __future__ import annotations

import io
import logging
import re

logger = logging.getLogger(__name__)

# Recognised Turkish characters — used for auto language detection
_TR_RE = re.compile(r"[çğıöşüÇĞİÖŞÜ]|(?:^|\s)(ve|ile|bir|bu|ne|da|de|mi|mu)\s", re.IGNORECASE)

DEFAULT_VOICE_TR = "tr-TR-AhmetNeural"
DEFAULT_VOICE_EN = "en-US-AriaNeural"


def _detect_voice(text: str) -> str:
    """Heuristic: if text looks Turkish, return a Turkish voice."""
    if _TR_RE.search(text):
        return DEFAULT_VOICE_TR
    return DEFAULT_VOICE_EN


def _strip_markdown(text: str) -> str:
    """Remove common markdown so TTS reads clean prose."""
    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`[^`]+`", "", text)
    # Remove headers, bold, italic
    text = re.sub(r"#{1,6}\s", "", text)
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
    text = re.sub(r"_{1,2}([^_]+)_{1,2}", r"\1", text)
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Remove markdown links [text](url) → text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Collapse blank lines
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


async def synthesize(
    text: str,
    voice: str | None = None,
    max_chars: int = 1500,
) -> bytes | None:
    """Convert text to speech using edge-tts.

    Args:
        text:      Input text (may contain markdown — will be stripped).
        voice:     Override voice name. Auto-detected if None.
        max_chars: Truncate text to this many characters to keep latency low.

    Returns:
        Raw MP3 bytes, or None if edge-tts is unavailable or text is empty.
    """
    try:
        import edge_tts
    except ImportError:
        logger.warning("edge-tts not installed — TTS disabled. Run: uv add edge-tts")
        return None

    clean = _strip_markdown(text)
    if not clean:
        return None
    if len(clean) > max_chars:
        clean = clean[:max_chars] + "…"

    selected_voice = voice or _detect_voice(clean)

    try:
        communicate = edge_tts.Communicate(clean, selected_voice)
        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        audio = buf.getvalue()
        if not audio:
            logger.warning("edge-tts returned empty audio for voice=%s", selected_voice)
            return None
        logger.debug("TTS: %d chars → %d bytes MP3 (voice=%s)", len(clean), len(audio), selected_voice)
        return audio
    except Exception as exc:
        logger.warning("TTS synthesis failed: %s", exc)
        return None
