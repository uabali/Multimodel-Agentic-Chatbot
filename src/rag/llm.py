"""
LLM factory — OpenAI-compatible backend via ChatOpenAI.

We keep the dual-profile concept (chat vs RAG vs agent) through separate
temperature / max_tokens settings, independent of the backend.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any
from urllib.parse import urljoin

import httpx
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import tiktoken

from src.config import settings

logger = logging.getLogger(__name__)
_runtime_context_checked = False


@lru_cache(maxsize=1)
def _token_encoding():
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Approximate token count with cl100k_base.

    The local Gemma backend uses its own SentencePiece tokenizer, so this is not
    exact. It is still a much better dependency-light estimate for Turkish text
    than character heuristics such as len(text) / 4, which systematically
    over- or under-budget mixed Turkish/English RAG prompts.
    """
    if not text:
        return 0
    return len(_token_encoding().encode(str(text)))


def count_message_tokens(messages: list) -> int:
    """Approximate token count for LangChain-style messages via .content."""
    total = 0
    for message in messages or []:
        content = getattr(message, "content", "")
        if isinstance(content, list):
            content = " ".join(str(item.get("text") or item) if isinstance(item, dict) else str(item) for item in content)
        total += count_tokens(str(content or ""))
    return total


def negotiate_runtime_context_size(timeout_s: float = 1.0) -> int:
    """Best-effort llama.cpp /props n_ctx negotiation.

    If the server reports a smaller context window than configuration, the
    configured runtime value is reduced to the smaller safe value.
    """
    global _runtime_context_checked
    if _runtime_context_checked:
        return int(settings.llm_context_size)
    _runtime_context_checked = True

    base = settings.llm_server_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    props_url = urljoin(base + "/", "props")
    try:
        resp = httpx.get(props_url, timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json()
        reported = data.get("default_generation_settings", {}).get("n_ctx") or data.get("n_ctx")
        if reported is None:
            return int(settings.llm_context_size)
        reported_i = int(reported)
        configured = int(settings.llm_context_size)
        if reported_i != configured:
            effective = min(reported_i, configured)
            logger.warning(
                "LLM context mismatch: configured=%d, runtime=%d, effective=%d",
                configured, reported_i, effective,
            )
            settings.llm_context_size = effective
        return int(settings.llm_context_size)
    except Exception as exc:
        logger.debug("LLM /props context negotiation skipped: %s", exc)
        return int(settings.llm_context_size)


def _make_openai_compat_client(
    temperature: float,
    max_tokens: int,
    top_p: float = 0.95,
    profile_name: str = "custom",
) -> ChatOpenAI:
    """
    Build a ChatOpenAI client pointed at an OpenAI-compatible endpoint.

    Many OpenAI-compat servers require api_key to be a non-empty string; the
    actual value is often ignored unless the server is configured to enforce it.
    We send "dummy" to satisfy client-side validation.

    extra_body is forwarded verbatim; we only send vLLM-specific knobs when the
    backend is vLLM to avoid surprising other servers (e.g. llama.cpp).
    """
    extra_body: dict[str, Any] | None = None
    if (settings.llm_backend or "").lower() in {"vllm"} and settings.llm_enable_thinking:
        extra_body = {
            "chat_template_kwargs": {
                "enable_thinking": True,
            },
        }

    kwargs: dict[str, Any] = {
        "model": settings.llm_model_name,
        "base_url": settings.llm_server_url,
        "api_key": "dummy",
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "streaming": True,
        "name": f"frappe.llm.{profile_name}",
        "tags": ["frappe", "llm", f"profile:{profile_name}"],
        "metadata": {
            "ls_provider": "openai_compat",
            "ls_model_name": settings.llm_model_name,
            "frappe_llm_profile": profile_name,
            "llm_backend": settings.llm_backend,
        },
    }
    if extra_body:
        kwargs["extra_body"] = extra_body
    return ChatOpenAI(**kwargs)


class DualLLM:
    """
    Lazy-initialised LLM clients with three profiles:
      - chat_model:  higher temperature, moderate token budget (conversational)
      - rag_model:   temperature=0, tighter budget (factual, deterministic)
      - agent_model: low temperature, higher budget (agentic RAG, tool calls)

    All hit the same backend server/model; only sampling params differ.
    Tuned for Q5_K_M (~5.48 GB) on 12GB GPU with ~10GB ceiling.
    """

    def __init__(self) -> None:
        negotiate_runtime_context_size()
        self._chat: ChatOpenAI | None = None
        self._rag: ChatOpenAI | None = None
        self._agent: ChatOpenAI | None = None

    @property
    def chat_model(self) -> ChatOpenAI:
        if self._chat is None:
            self._chat = _make_openai_compat_client(
                temperature=settings.chat_temperature,
                max_tokens=settings.chat_max_tokens,
                profile_name="chat",
            )
        return self._chat

    @property
    def rag_model(self) -> ChatOpenAI:
        if self._rag is None:
            self._rag = _make_openai_compat_client(
                temperature=settings.rag_temperature,
                max_tokens=settings.rag_max_tokens,
                profile_name="rag",
            )
        return self._rag

    @property
    def agent_model(self) -> ChatOpenAI:
        """Agentic RAG profile: low temp for consistent tool calls, higher token budget."""
        if self._agent is None:
            self._agent = _make_openai_compat_client(
                temperature=settings.agent_temperature,
                max_tokens=settings.agent_max_tokens,
                top_p=0.9,
                profile_name="agent",
            )
        return self._agent

    def warm_up(self) -> None:
        msg = HumanMessage(content="ping")
        try:
            self.chat_model.invoke([msg])
            self.rag_model.invoke([msg])
        except Exception as exc:
            logger.warning("LLM warm-up skipped: %s", exc)

    def benchmark_chat(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            r = self.chat_model.invoke([HumanMessage(content="Say OK in one word.")])
            dt = time.perf_counter() - t0
            text = r.content if isinstance(r.content, str) else str(r.content)
            return {"response_time_s": round(dt, 3), "chars": len(text)}
        except Exception as exc:
            return {"error": str(exc)}


_dual_llm: DualLLM | None = None


def get_dual_llm() -> DualLLM:
    global _dual_llm
    if _dual_llm is None:
        _dual_llm = DualLLM()
    return _dual_llm


def reset_llm_cache() -> None:
    """Önbellekteki tüm LLM istemcilerini sıfırlar.

    vLLM URL veya model adı runtime'da değiştiğinde çağrılmalıdır.
    Bir sonraki LLM çağrısında yeni ayarlarla fresh istemciler oluşturulur.
    """
    global _dual_llm
    _dual_llm = None
    logger.info("LLM önbelleği sıfırlandı — bir sonraki çağrıda yeniden başlatılacak.")


def get_chat_llm() -> ChatOpenAI:
    return get_dual_llm().chat_model


def get_rag_llm() -> ChatOpenAI:
    return get_dual_llm().rag_model


def get_agent_llm() -> ChatOpenAI:
    """Get the agentic RAG LLM (low temp, higher token budget for tool use)."""
    return get_dual_llm().agent_model


def create_vllm_llm(
    temperature: float = 0.7,
    max_tokens: int = 768,
    top_p: float = 0.95,
) -> ChatOpenAI:
    """Convenience factory for callers that need a one-off OpenAI-compat client."""
    return _make_openai_compat_client(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        profile_name="custom",
    )
