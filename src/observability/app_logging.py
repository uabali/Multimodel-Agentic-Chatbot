"""Small terminal-first logging helpers.

The app already has LangSmith for rich traces. This module is intentionally
stricter: concise key=value console lines, a fixed field allowlist, and no raw
prompts or document contents.
"""

from __future__ import annotations

import logging
import re
import shlex
import uuid
from typing import Any

from src.config import settings


LOG_FIELDS = (
    "event",
    "sid",
    "turn_id",
    "route",
    "input_type",
    "q_chars",
    "history_turns",
    "attachments",
    "model",
    "backend",
    "ctx",
    "max_tokens",
    "ttft_ms",
    "llm_ms",
    "rag_ms",
    "total_ms",
    "docs",
    "used_chunks",
    "cache",
    "fallback",
    "error_type",
)

_LOG_FIELD_SET = set(LOG_FIELDS)
_SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password|authorization)\b\s*[:=]\s*['\"]?([^\s'\"]+)"
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_LONG_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9_-]{32,}(?![A-Za-z0-9])")


def new_turn_id() -> str:
    """Return a short non-guessable turn id for local log correlation."""
    return uuid.uuid4().hex[:8]


def configure_app_logging() -> None:
    """Configure readable console logging once, before app/router logs fire."""
    level_name = str(settings.app_log_level or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)-25s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def stage_timings_enabled() -> bool:
    return bool(settings.app_log_stage_timings)


def detailed_trace_enabled(logger: logging.Logger | None = None) -> bool:
    configured = str(settings.app_log_level or "").upper() == "DEBUG"
    logger_debug = bool(logger and logger.isEnabledFor(logging.DEBUG))
    return configured or logger_debug


def safe_log_preview(value: Any, max_chars: int | None = None) -> str:
    """Redacted single-line preview for exceptional debug-only situations."""
    limit = max(0, int(settings.app_log_preview_chars if max_chars is None else max_chars))
    if limit <= 0 or value is None:
        return ""
    text = _redact_log_text(str(value))
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        return text[: max(0, limit - 3)].rstrip() + "..."
    return text


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """Emit one stable key=value log line with only approved developer fields."""
    if not logger.isEnabledFor(level):
        return
    payload = {"event": event, **fields}
    parts: list[str] = []
    for key in LOG_FIELDS:
        if key not in payload:
            continue
        value = payload[key]
        if value is None or value == "":
            continue
        parts.append(f"{key}={_format_log_value(value)}")
    logger.log(level, " ".join(parts))


def _redact_log_text(text: str) -> str:
    text = _BEARER_RE.sub("Bearer [redacted-token]", text)
    text = _SECRET_RE.sub(lambda m: f"{m.group(1)}=[redacted-secret]", text)
    text = _LONG_TOKEN_RE.sub("[redacted-token]", text)
    return text


def _format_log_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        text = f"{value:.2f}".rstrip("0").rstrip(".")
    else:
        text = str(value)
    text = _redact_log_text(text)
    if not text:
        return "''"
    if re.fullmatch(r"[A-Za-z0-9_./:#@+-]+", text):
        return text
    return shlex.quote(text)


def allowed_log_fields() -> tuple[str, ...]:
    return LOG_FIELDS


def filter_log_fields(fields: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in fields.items() if k in _LOG_FIELD_SET}
