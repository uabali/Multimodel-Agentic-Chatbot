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
MAX_SUMMARY_TOKENS = 1000
MAX_TOPIC_CHARS = 200
MAX_PERSISTED_CHAT_MESSAGES = 100

_PIN_PATTERNS = [
    re.compile(r"^\s*(?:bunu\s+hat[ıi]rla|not\s+al|remember\s+this)\s*[:：-]\s*(.+)\s*$", re.I | re.S),
    re.compile(r"^\s*bu\s+sohbet\s+i[çc]in\s+hat[ıi]rla\s*[:：-]?\s*(.+)\s*$", re.I | re.S),
]
_MEMORY_COMMAND_RE = re.compile(
    r"^\s*(?:bunu\s+hat[ıi]rla|not\s+al|remember\s+this|bu\s+sohbet\s+i[çc]in\s+hat[ıi]rla)\b",
    re.I,
)


def _now_iso() -> str:
    """Kısa: `_now_iso` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _clean_text(text: str, max_chars: int) -> str:
    """Kısa: `_clean_text` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip() + "..."
    return cleaned


def _clean_summary(text: str) -> str:
    """Kısa: `_clean_summary` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
    cleaned = _clean_text(text, MAX_SUMMARY_TOKENS * 8)
    try:
        from src.rag.llm import count_tokens
        while count_tokens(cleaned) > MAX_SUMMARY_TOKENS and len(cleaned) > 100:
            cleaned = cleaned[: int(len(cleaned) * 0.9)].rstrip()
        if count_tokens(cleaned) > MAX_SUMMARY_TOKENS:
            cleaned = cleaned[:100].rstrip()
        if cleaned and cleaned != text.strip() and not cleaned.endswith("..."):
            cleaned += "..."
    except Exception:
        cleaned = _clean_text(cleaned, 4000)
    return cleaned


@dataclass(slots=True)
class ThreadMemory:
    version: int = MEMORY_VERSION
    rolling_summary: str = ""
    pinned_facts: list[str] = field(default_factory=list)
    last_topic: str = ""
    updated_at: str = ""

    @classmethod
    def empty(cls) -> "ThreadMemory":
        """Kısa: `empty` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
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
                rolling_summary=_clean_summary(str(raw_memory.get("rolling_summary") or "")),
                pinned_facts=pinned,
                last_topic=_clean_text(str(raw_memory.get("last_topic") or ""), MAX_TOPIC_CHARS),
                updated_at=str(raw_memory.get("updated_at") or _now_iso()),
            )

        legacy_summary = _clean_summary(str(metadata.get("summary") or ""))
        if legacy_summary:
            return cls(rolling_summary=legacy_summary, updated_at=_now_iso())
        return cls.empty()

    def to_metadata(self) -> dict[str, Any]:
        """Kısa: `to_metadata` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        return {
            "version": self.version,
            "rolling_summary": self.rolling_summary,
            "pinned_facts": list(self.pinned_facts),
            "last_topic": self.last_topic,
            "updated_at": self.updated_at or _now_iso(),
        }

    def with_summary(self, summary: str) -> "ThreadMemory":
        """Kısa: `with_summary` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        return ThreadMemory(
            version=self.version,
            rolling_summary=_clean_summary(summary),
            pinned_facts=list(self.pinned_facts),
            last_topic=self.last_topic,
            updated_at=_now_iso(),
        )

    def with_pin(self, fact: str) -> "ThreadMemory":
        """Kısa: `with_pin` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
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
            last_topic=self.last_topic,
            updated_at=_now_iso(),
        )

    def with_last_topic(self, topic: str) -> "ThreadMemory":
        """Kısa: `with_last_topic` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        return ThreadMemory(
            version=self.version,
            rolling_summary=self.rolling_summary,
            pinned_facts=list(self.pinned_facts),
            last_topic=_clean_text(topic, MAX_TOPIC_CHARS),
            updated_at=_now_iso(),
        )


def metadata_patch(memory: ThreadMemory) -> dict[str, Any]:
    """Patch for threads.metadata; keeps legacy summary for old readers."""
    return {
        "memory": memory.to_metadata(),
        "summary": memory.rolling_summary,
    }


def memory_hash(memory: ThreadMemory | Mapping[str, Any] | None) -> str:
    """Kısa: `memory_hash` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
    if not isinstance(memory, ThreadMemory):
        memory = ThreadMemory.from_metadata(memory)
    payload = json.dumps(
        {
            "summary": memory.rolling_summary,
            "pins": memory.pinned_facts,
            "last_topic": memory.last_topic,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def extract_memory_pin(text: str) -> str | None:
    """Kısa: `extract_memory_pin` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
    text = text or ""
    for pattern in _PIN_PATTERNS:
        match = pattern.match(text)
        if match:
            return _clean_text(match.group(1), MAX_PIN_CHARS) or None
    return None


def is_memory_command(text: str) -> bool:
    """Kısa: `is_memory_command` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
    return bool(_MEMORY_COMMAND_RE.match(text or ""))


def format_memory_context(memory: ThreadMemory | None) -> str:
    """Kısa: `format_memory_context` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
    if not memory:
        return ""
    parts: list[str] = []
    if memory.rolling_summary.strip():
        parts.append("Önceki konuşma özeti:\n" + memory.rolling_summary.strip())
    if memory.pinned_facts:
        pins = "\n".join(f"- {fact}" for fact in memory.pinned_facts if fact.strip())
        if pins:
            parts.append("Bu thread için kalıcı notlar:\n" + pins)
    if memory.last_topic.strip():
        parts.append("Son konu:\n" + memory.last_topic.strip())
    return "\n\n".join(parts).strip()


def format_memory_preferences(memory: ThreadMemory | None) -> str:
    """RAG generator prompt'u için özet + pin bloğu (last_topic hariç)."""
    if not memory:
        return ""
    parts: list[str] = []
    if memory.rolling_summary.strip():
        parts.append(f"Özet: {memory.rolling_summary.strip()}")
    if memory.pinned_facts:
        pins = "\n".join(f"- {fact}" for fact in memory.pinned_facts if fact.strip())
        if pins:
            parts.append("Kalıcı notlar:\n" + pins)
    return "\n\n".join(parts).strip()


def serialize_chat_history_for_metadata(chat_history: list[dict] | None) -> list[dict]:
    """SQLite thread metadata için son N mesajı normalize eder."""
    out: list[dict] = []
    for item in chat_history or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        content = str(item.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        out.append({"role": role, "content": content[:4000]})
    if len(out) > MAX_PERSISTED_CHAT_MESSAGES:
        out = out[-MAX_PERSISTED_CHAT_MESSAGES:]
    return out


def chat_history_metadata_patch(chat_history: list[dict] | None) -> dict[str, Any]:
    """Thread metadata'ya yazılacak chat_history slice."""
    return {"chat_history": serialize_chat_history_for_metadata(chat_history)}


def merge_resume_histories(
    step_history: list[dict] | None,
    meta_history: list[dict] | None,
) -> list[dict]:
    """Chainlit steps ile SQLite metadata chat_history'yi birleştirir."""
    steps = serialize_chat_history_for_metadata(step_history)
    meta = serialize_chat_history_for_metadata(meta_history)
    if not meta:
        return steps
    if not steps:
        return meta
    if len(meta) >= len(steps):
        return meta
    # Steps daha uzun: eski kısım steps'ten, kuyruk metadata'dan (daha güncel persist).
    tail_len = min(len(meta), len(steps))
    if tail_len <= 0:
        return steps
    merged = steps[:-tail_len] + meta[-tail_len:]
    if len(merged) > MAX_PERSISTED_CHAT_MESSAGES:
        merged = merged[-MAX_PERSISTED_CHAT_MESSAGES:]
    return merged
