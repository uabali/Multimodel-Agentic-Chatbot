"""Thread-scoped conversation memory.

This module intentionally keeps memory local to a Chainlit thread. It does not
create cross-chat user memory, which avoids surprising carry-over between new
conversations.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


MEMORY_VERSION = 1
MAX_PINNED_FACTS = 20
MAX_PIN_CHARS = 400

_PIN_PATTERNS = [
    re.compile(r"^\s*(?:bunu\s+hat[ıi]rla|not\s+al|remember\s+this)\s*[:：-]\s*(.+)\s*$", re.I | re.S),
    re.compile(r"^\s*bu\s+sohbet\s+i[çc]in\s+hat[ıi]rla\s*[:：-]?\s*(.+)\s*$", re.I | re.S),
]
_MEMORY_COMMAND_RE = re.compile(
    r"^\s*(?:bunu\s+hat[ıi]rla|not\s+al|remember\s+this|bu\s+sohbet\s+i[çc]in\s+hat[ıi]rla)\b",
    re.I,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _clean_text(text: str, max_chars: int) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip() + "..."
    return cleaned


@dataclass(slots=True)
class ThreadMemory:
    version: int = MEMORY_VERSION
    rolling_summary: str = ""
    pinned_facts: list[str] = field(default_factory=list)
    updated_at: str = ""

    @classmethod
    def empty(cls) -> "ThreadMemory":
        return cls(updated_at=_now_iso())

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any] | None) -> "ThreadMemory":
        """Parse structured memory, falling back to legacy metadata.summary."""
        metadata = metadata or {}
        raw_memory = metadata.get("memory")
        if isinstance(raw_memory, Mapping):
            facts = raw_memory.get("pinned_facts") or []
            pinned = [
                _clean_text(str(item), MAX_PIN_CHARS)
                for item in facts
                if str(item or "").strip()
            ][:MAX_PINNED_FACTS]
            return cls(
                version=int(raw_memory.get("version") or MEMORY_VERSION),
                rolling_summary=_clean_text(str(raw_memory.get("rolling_summary") or ""), 4000),
                pinned_facts=pinned,
                updated_at=str(raw_memory.get("updated_at") or _now_iso()),
            )

        legacy_summary = _clean_text(str(metadata.get("summary") or ""), 4000)
        if legacy_summary:
            return cls(rolling_summary=legacy_summary, updated_at=_now_iso())
        return cls.empty()

    def to_metadata(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "rolling_summary": self.rolling_summary,
            "pinned_facts": list(self.pinned_facts),
            "updated_at": self.updated_at or _now_iso(),
        }

    def with_summary(self, summary: str) -> "ThreadMemory":
        return ThreadMemory(
            version=self.version,
            rolling_summary=_clean_text(summary, 4000),
            pinned_facts=list(self.pinned_facts),
            updated_at=_now_iso(),
        )

    def with_pin(self, fact: str) -> "ThreadMemory":
        fact = _clean_text(fact, MAX_PIN_CHARS)
        if not fact:
            return self
        existing = [item for item in self.pinned_facts if item.strip()]
        if fact in existing:
            existing.remove(fact)
        updated = (existing + [fact])[-MAX_PINNED_FACTS:]
        return ThreadMemory(
            version=self.version,
            rolling_summary=self.rolling_summary,
            pinned_facts=updated,
            updated_at=_now_iso(),
        )


def metadata_patch(memory: ThreadMemory) -> dict[str, Any]:
    """Patch for threads.metadata; keeps legacy summary for old readers."""
    return {
        "memory": memory.to_metadata(),
        "summary": memory.rolling_summary,
    }


def memory_hash(memory: ThreadMemory | Mapping[str, Any] | None) -> str:
    if not isinstance(memory, ThreadMemory):
        memory = ThreadMemory.from_metadata(memory)
    payload = json.dumps(
        {
            "summary": memory.rolling_summary,
            "pins": memory.pinned_facts,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def extract_memory_pin(text: str) -> str | None:
    text = text or ""
    for pattern in _PIN_PATTERNS:
        match = pattern.match(text)
        if match:
            return _clean_text(match.group(1), MAX_PIN_CHARS) or None
    return None


def is_memory_command(text: str) -> bool:
    return bool(_MEMORY_COMMAND_RE.match(text or ""))


def format_memory_context(memory: ThreadMemory | None) -> str:
    if not memory:
        return ""
    parts: list[str] = []
    if memory.rolling_summary.strip():
        parts.append("Önceki konuşma özeti:\n" + memory.rolling_summary.strip())
    if memory.pinned_facts:
        pins = "\n".join(f"- {fact}" for fact in memory.pinned_facts if fact.strip())
        if pins:
            parts.append("Bu thread için kalıcı notlar:\n" + pins)
    return "\n\n".join(parts).strip()
