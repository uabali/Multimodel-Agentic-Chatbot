"""Small terminal-first logging helpers.

The app already has LangSmith for rich traces. This module is intentionally
stricter: concise key=value console lines, a fixed field allowlist, and no raw
prompts or document contents.
"""

from __future__ import annotations

import logging
import re
import shlex
import sys
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
    # Dynamic RAG/CRAG metrics added by senior review
    "relevance",
    "grader_reason",
    "dense_score",
    "retry_path",
    "strategy",
)

_LOG_FIELD_SET = set(LOG_FIELDS)
_SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password|authorization)\b\s*[:=]\s*['\"]?([^\s'\"]+)"
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_LONG_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9_-]{32,}(?![A-Za-z0-9])")

# Event icons mapping for visual terminal logging
EVENT_ICONS = {
    "turn_start": "🚀",
    "turn_end": "🏁",
    "stream": "⚡",
    "cache": "💾",
    "fallback": "🔄",
    "web_search_result": "🌐",
    "retriever_result": "🔍",
    "grader_decision": "⚖️",
    "generator_result": "🤖",
    "direct_response_result": "💬",
    "turn_error": "🚨",
    "cache_hit": "⚡",
}


class ColorConsoleFormatter(logging.Formatter):
    """Colorizes console log level names when stdout is a TTY for rich readability."""
    
    COLOR_RESET = "\033[0m"
    COLORS = {
        logging.DEBUG: "\033[36m",     # Cyan
        logging.INFO: "\033[34m",      # Blue
        logging.WARNING: "\033[33m",   # Yellow
        logging.ERROR: "\033[31m",     # Red
        logging.CRITICAL: "\033[1;31m",# Bold Red
    }

    def format(self, record: logging.LogRecord) -> str:
        is_tty = sys.stdout.isatty()
        level_color = self.COLORS.get(record.levelno, "")
        
        orig_levelname = record.levelname
        if is_tty and level_color:
            record.levelname = f"{level_color}{record.levelname:<7}{self.COLOR_RESET}"
            
        result = super().format(record)
        record.levelname = orig_levelname  # Restore to prevent side effects
        return result


def new_turn_id() -> str:
    """Return a short non-guessable turn id for local log correlation."""
    return uuid.uuid4().hex[:8]


def configure_app_logging() -> None:
    """Configure readable console logging once, before app/router logs fire."""
    level_name = str(settings.app_log_level or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    
    # Custom logger setup to use the ColorConsoleFormatter
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Remove any existing basicConfig handlers to avoid duplicate prints
    for h in list(logger.handlers):
        logger.removeHandler(h)
        
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    formatter = ColorConsoleFormatter(
        fmt="%(asctime)s | %(levelname)s | %(name)-25s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logging.getLogger("httpx").setLevel(logging.WARNING)


def stage_timings_enabled() -> bool:
    return bool(settings.app_log_stage_timings)


def detailed_trace_enabled(logging_logger: logging.Logger | None = None) -> bool:
    configured = str(settings.app_log_level or "").upper() == "DEBUG"
    logger_debug = bool(logging_logger and logging_logger.isEnabledFor(logging.DEBUG))
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
    logging_logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """Emit one stable key=value log line with only approved developer fields."""
    if not logging_logger.isEnabledFor(level):
        return
        
    payload = {"event": event, **fields}
    parts: list[str] = []
    is_tty = sys.stdout.isatty()
    
    for key in LOG_FIELDS:
        if key not in payload:
            continue
        value = payload[key]
        if value is None or value == "":
            continue
            
        formatted_val = _format_log_value(value)
        
        # Colorize and add icons when printing directly to a local terminal
        if is_tty:
            if key == "event":
                icon = EVENT_ICONS.get(str(value), "🛠️")
                parts.append(f"event={icon} \033[1m{formatted_val}\033[0m")
            elif key == "cache" and value == "hit":
                parts.append(f"cache=⚡ \033[32m{formatted_val}\033[0m")
            elif key == "cache" and value == "miss":
                parts.append(f"cache=💾 \033[33m{formatted_val}\033[0m")
            elif key == "route":
                color = "\033[35m" if value in ("rag", "vision_rag") else "\033[32m"
                parts.append(f"route={color}{formatted_val}\033[0m")
            elif key == "relevance":
                color = "\033[32m" if value == "yes" else "\033[31m"
                parts.append(f"relevance={color}{formatted_val}\033[0m")
            elif key in ("ttft_ms", "llm_ms", "rag_ms", "total_ms"):
                parts.append(f"{key}=\033[33m{formatted_val}ms\033[0m")
            elif key == "dense_score" or key == "grader_reason":
                parts.append(f"{key}=\033[36m{formatted_val}\033[0m")
            else:
                parts.append(f"{key}={formatted_val}")
        else:
            parts.append(f"{key}={formatted_val}")
            
    logging_logger.log(level, " ".join(parts))


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
