"""
FastAPI router — LLM bağlantı yönetimi ve sistem durumu API'si.

Endpoint'ler:
  GET  /api/health        — LLM + Qdrant'ın erişilebilirliğini kontrol eder
  GET  /api/config        — Mevcut LLM URL ve model adını döner
  PUT  /api/config/llm    — LLM URL'sini runtime'da günceller (LLM önbelleği sıfırlanır)
  POST /api/llm/probe     — LLM sunucusuna ping atar ve yanıt süresini ölçer

Tasarım kararları:
  - SRP: Bu modül yalnızca admin/config HTTP katmanından sorumludur.
  - DIP: `settings` ve `reset_llm_cache` enjekte edilmiş bağımlılıklar gibi kullanılır.
  - Chainlit'in içindeki FastAPI uygulamasına (`/api` prefix) mount edilir;
    ayrı bir port açılmaz.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import secrets
import time
from typing import Annotated

import httpx
from fastapi import APIRouter, Body, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field, HttpUrl

from src.middleware.rate_limiter import rate_limit_chat, rate_limit_config

_basic_security = HTTPBasic(auto_error=False)


def _hash_pw(password: str, salt: str) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 210_000)
    return dk.hex()


async def require_admin(
    credentials: Annotated[HTTPBasicCredentials | None, Depends(_basic_security)],
) -> None:
    """FastAPI dependency — rejects requests without valid admin credentials."""
    from src.config import settings

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin authentication required.",
            headers={"WWW-Authenticate": "Basic"},
        )
    ok_user = secrets.compare_digest(credentials.username, settings.app_admin_username)
    ok_pass = secrets.compare_digest(
        _hash_pw(credentials.password, settings.app_password_salt),
        _hash_pw(settings.app_admin_password, settings.app_password_salt),
    )
    if not (ok_user and ok_pass):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin credentials.",
            headers={"WWW-Authenticate": "Basic"},
        )

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["LLM Config"])


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response şemaları
# ─────────────────────────────────────────────────────────────────────────────


class LLMUrlUpdate(BaseModel):
    """LLM URL güncelleme isteği."""

    url: str = Field(
        ...,
        description="OpenAI-compat endpoint URL. Örnek: http://localhost:8080/v1",
        examples=["http://localhost:8080/v1", "http://localhost:8000/v1"],
    )
    model_name: str | None = Field(
        default=None,
        description="Opsiyonel: Model adını da güncelle. Boş bırakılırsa mevcut model korunur.",
        examples=["gemma-4-e4b", "Qwen/Qwen3-4B-AWQ"],
    )


class LLMConfigResponse(BaseModel):
    """Mevcut LLM yapılandırması."""

    llm_backend: str
    llm_server_url: str
    llm_model_name: str
    llm_enable_thinking: bool
    rag_max_tokens: int
    chat_max_tokens: int
    agent_max_tokens: int


class HealthResponse(BaseModel):
    """Sistem sağlık durumu."""

    status: str  # "ok" | "degraded" | "error"
    llm: dict
    qdrant: dict
    overall_latency_ms: float


class ProbeResponse(BaseModel):
    """LLM ping/probe sonucu."""

    reachable: bool
    url: str
    response_time_ms: float | None = None
    models: list[str] = []
    error: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Yardımcılar
# ─────────────────────────────────────────────────────────────────────────────


async def _check_vllm(url: str, timeout: float = 5.0) -> dict:
    """OpenAI-compat /models endpoint'ine HTTP isteği atar."""
    base = url.rstrip("/")
    # /v1/models → model listesini döner
    models_url = f"{base}/models"
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(models_url)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        if resp.status_code == 200:
            data = resp.json()
            models = [m.get("id", "") for m in data.get("data", [])]
            return {"reachable": True, "latency_ms": elapsed_ms, "models": models}
        return {"reachable": False, "status_code": resp.status_code, "latency_ms": elapsed_ms}
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {"reachable": False, "error": str(exc), "latency_ms": elapsed_ms}


async def _check_qdrant(url: str, timeout: float = 5.0) -> dict:
    """Qdrant /healthz endpoint'ine HTTP isteği atar."""
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{url}/healthz")
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {"reachable": resp.status_code == 200, "latency_ms": elapsed_ms}
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {"reachable": False, "error": str(exc), "latency_ms": elapsed_ms}


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint'ler
# ─────────────────────────────────────────────────────────────────────────────


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="LLM ve Qdrant sağlık kontrolü",
    description="LLM sunucusu ve Qdrant vektör veritabanının erişilebilirliğini eşzamanlı olarak kontrol eder.",
    dependencies=[Depends(rate_limit_chat)],
)
async def health_check() -> HealthResponse:
    from src.config import settings

    t0 = time.perf_counter()

    # LLM ve Qdrant'ı aynı anda kontrol et
    llm_info, qdrant_info = await asyncio.gather(
        _check_vllm(settings.llm_server_url),
        _check_qdrant(settings.qdrant_url),
        return_exceptions=True,
    )

    if isinstance(llm_info, Exception):
        llm_info = {"reachable": False, "error": str(llm_info), "latency_ms": 0}
    if isinstance(qdrant_info, Exception):
        qdrant_info = {"reachable": False, "error": str(qdrant_info), "latency_ms": 0}

    total_ms = round((time.perf_counter() - t0) * 1000, 1)

    if llm_info["reachable"] and qdrant_info["reachable"]:
        overall = "ok"
    elif llm_info["reachable"] or qdrant_info["reachable"]:
        overall = "degraded"
    else:
        overall = "error"

    return HealthResponse(
        status=overall,
        llm=llm_info,
        qdrant=qdrant_info,
        overall_latency_ms=total_ms,
    )


@router.get(
    "/config",
    response_model=LLMConfigResponse,
    summary="Mevcut LLM yapılandırması",
    description="Şu an aktif olan LLM backend/URL, model adı ve token parametrelerini döner.",
    dependencies=[Depends(require_admin)],
)
async def get_config() -> LLMConfigResponse:
    from src.config import settings

    return LLMConfigResponse(
        llm_backend=settings.llm_backend,
        llm_server_url=settings.llm_server_url,
        llm_model_name=settings.llm_model_name,
        llm_enable_thinking=settings.llm_enable_thinking,
        rag_max_tokens=settings.rag_max_tokens,
        chat_max_tokens=settings.chat_max_tokens,
        agent_max_tokens=settings.agent_max_tokens,
    )


@router.put(
    "/config/llm",
    response_model=LLMConfigResponse,
    summary="LLM URL'sini güncelle",
    dependencies=[Depends(require_admin), Depends(rate_limit_config)],
    description=(
        "LLM sunucu URL'sini ve opsiyonel olarak model adını runtime'da günceller. "
        "Güncelleme sonrası LLM istemci önbelleği temizlenir; bir sonraki sohbet "
        "yeni URL'yi kullanır. Değişiklik yalnızca süreç yeniden başlayana kadar geçerlidir — "
        "kalıcı değişiklik için `.env` dosyasını da güncelleyin."
    ),
)
async def update_llm_config(
    payload: Annotated[LLMUrlUpdate, Body(embed=False)],
) -> LLMConfigResponse:
    from src.config import settings
    from src.rag.llm import reset_llm_cache

    # Yeni URL'nin erişilebilir olup olmadığını önce test et
    probe = await _check_vllm(payload.url, timeout=8.0)
    if not probe["reachable"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "message": f"LLM sunucusuna ulaşılamadı: {payload.url}",
                "error": probe.get("error", f"HTTP {probe.get('status_code')}"),
                "hint": "Sunucunun çalıştığından ve URL'nin doğru olduğundan emin olun.",
            },
        )

    # Ayarları güncelle (Pydantic v2 BaseSettings varsayılan olarak mutable)
    old_url = settings.llm_server_url
    settings.llm_server_url = payload.url.rstrip("/")

    if payload.model_name:
        settings.llm_model_name = payload.model_name

    # Önbelleği temizle — bir sonraki LLM çağrısında yeni URL kullanılır
    reset_llm_cache()
    try:
        from src.agent.nodes import reset_nodes_llm_cache
        reset_nodes_llm_cache()
    except Exception:
        pass

    logger.info(
        "LLM URL güncellendi: %s → %s (model: %s)",
        old_url,
        settings.llm_server_url,
        settings.llm_model_name,
    )

    return LLMConfigResponse(
        llm_backend=settings.llm_backend,
        llm_server_url=settings.llm_server_url,
        llm_model_name=settings.llm_model_name,
        llm_enable_thinking=settings.llm_enable_thinking,
        rag_max_tokens=settings.rag_max_tokens,
        chat_max_tokens=settings.chat_max_tokens,
        agent_max_tokens=settings.agent_max_tokens,
    )


# Backward-compat alias endpoint (older UI/docs).
@router.put("/config/vllm", response_model=LLMConfigResponse, include_in_schema=False, dependencies=[Depends(require_admin)])
async def update_vllm_config(payload: Annotated[LLMUrlUpdate, Body(embed=False)]) -> LLMConfigResponse:
    return await update_llm_config(payload)


@router.post(
    "/llm/probe",
    response_model=ProbeResponse,
    summary="LLM bağlantısını test et",
    dependencies=[Depends(require_admin), Depends(rate_limit_chat)],
    description=(
        "Verilen URL'ye (veya mevcut ayara) bağlanmayı dener ve yanıt süresini ölçer. "
        "Ayarları değiştirmez — yalnızca bağlantı testi için kullanın."
    ),
)
async def probe_vllm(
    url: str | None = None,
) -> ProbeResponse:
    from src.config import settings

    target = (url or settings.llm_server_url).rstrip("/")
    info = await _check_vllm(target, timeout=8.0)

    return ProbeResponse(
        reachable=info["reachable"],
        url=target,
        response_time_ms=info.get("latency_ms"),
        models=info.get("models", []),
        error=info.get("error"),
    )
