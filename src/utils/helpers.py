"""
Consolidated utility helper functions — DRY-compliant.

Provides common functions for password hashing, attachment counting,
and history turn calculations used across both main.py and agent nodes.
"""

from __future__ import annotations

import hashlib
import hmac
from datetime import datetime, timezone
from typing import Any
import langchain_core.messages as lc_msg


def datetime_now_iso() -> str:
    """Return the current time in ISO format with UTC suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def hash_password(password: str, salt: str) -> str:
    """Securely hash a password using PBKDF2 with SHA-256."""
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 210_000)
    return dk.hex()


def constant_time_eq(a: str, b: str) -> bool:
    """Compare two strings in constant time to prevent timing attacks."""
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def message_attachment_count(message: Any) -> int:
    """Count the total number of attachments in a Chainlit message."""
    count = 0
    for attr in ("elements", "files", "attachments"):
        items = getattr(message, attr, None) or []
        if isinstance(items, list):
            count += len(items)
    return count


def history_turn_count(messages: list) -> int:
    """Count the number of user/human turns in LangChain message format."""
    count = 0
    for m in messages:
        # Check standard LangChain message class or duck-typed roles
        role = getattr(m, "type", None) or getattr(m, "role", None)
        if isinstance(m, dict):
            role = m.get("role") or m.get("type")
        if role is None:
            if isinstance(m, lc_msg.HumanMessage):
                role = "human"
        if str(role).lower() in {"human", "user"}:
            count += 1
    return count


def chat_history_turns_for_log(chat_history: list | None) -> int:
    """Count user/human turns in standard dict-based Chainlit chat history list."""
    count = 0
    for item in chat_history or []:
        role = item.get("role") if isinstance(item, dict) else getattr(item, "role", "")
        if str(role).lower() in {"user", "human"}:
            count += 1
    return count
