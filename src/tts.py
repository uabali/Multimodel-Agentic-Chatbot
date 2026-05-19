from __future__ import annotations

import io
import logging
import re

logger = logging.getLogger(__name__)

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
    # YAML front matter
    text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL)
    # Code blocks (triple backticks — handles incomplete blocks too)
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"```[\s\S]*$", " ", text)
    # Inline code
    text = re.sub(r"`[^`\n]+`", " ", text)
    # Markdown tables (lines with pipes)
    text = re.sub(r"^\|.*\|$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[-|: ]+$", "", text, flags=re.MULTILINE)
    # Headers, bold, italic
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*{1,2}([^*\n]+?)\*{1,2}", r"\1", text)
    text = re.sub(r"_{1,2}([^_\n]+?)_{1,2}", r"\1", text)
    # Horizontal rules
    text = re.sub(r"^[-*_]{3,}$", "", text, flags=re.MULTILINE)
    # Markdown links [text](url) → text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # URLs
    text = re.sub(r"https?://\S+", "", text)
    # HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Collapse blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
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
