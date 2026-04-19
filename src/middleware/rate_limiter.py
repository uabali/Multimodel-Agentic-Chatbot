"""
In-memory sliding-window rate limiter — no external dependencies.

Used as a FastAPI dependency and as a Chainlit session guard.

Design:
- Per-IP sliding window: at most `max_requests` requests within `window_seconds`.
- Implemented with a deque of timestamps — O(1) amortised per check.
- Thread-safe via asyncio lock (single-process Chainlit/uvicorn assumption).
- Separate, stricter limits for /api/config/llm (config mutation endpoint).

Usage (FastAPI dependency):
    @router.put("/api/config/llm")
    async def update(..., _=Depends(rate_limit_config)):
        ...

Usage (Chainlit guard):
    allowed, retry_after = chat_rate_limiter.check(ip)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Deque

from fastapi import Depends, HTTPException, Request, status

logger = logging.getLogger(__name__)

_lock = asyncio.Lock()


class SlidingWindowLimiter:
    """Per-key sliding window rate limiter backed by a deque of timestamps."""

    def __init__(self, max_requests: int, window_seconds: float) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: dict[str, Deque[float]] = {}

    def check(self, key: str) -> tuple[bool, float]:
        """Return (allowed, retry_after_seconds).

        Call this from sync or async context — it is not a coroutine.
        For async contexts wrap with asyncio lock at the call site.
        """
        now = time.monotonic()
        window_start = now - self.window_seconds

        bucket = self._buckets.setdefault(key, deque())

        # Prune timestamps outside the window
        while bucket and bucket[0] < window_start:
            bucket.popleft()

        if len(bucket) >= self.max_requests:
            oldest = bucket[0]
            retry_after = self.window_seconds - (now - oldest)
            return False, max(0.0, retry_after)

        bucket.append(now)
        return True, 0.0


# ── Limiter instances ──────────────────────────────────────────────────────

# Chat messages: 30 requests / 60 seconds per IP (generous for real users)
chat_rate_limiter = SlidingWindowLimiter(max_requests=30, window_seconds=60)

# Config mutation: 10 requests / 60 seconds per IP (admin-only endpoint)
config_rate_limiter = SlidingWindowLimiter(max_requests=10, window_seconds=60)


# ── FastAPI dependency helpers ─────────────────────────────────────────────


def _get_client_ip(request: Request) -> str:
    """Extract real client IP, honouring X-Forwarded-For (Cloudflare tunnel)."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


async def rate_limit_chat(request: Request) -> None:
    """FastAPI dependency — enforce chat rate limit."""
    async with _lock:
        ip = _get_client_ip(request)
        allowed, retry_after = chat_rate_limiter.check(ip)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Retry after {retry_after:.0f}s.",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )


async def rate_limit_config(request: Request) -> None:
    """FastAPI dependency — enforce config-mutation rate limit."""
    async with _lock:
        ip = _get_client_ip(request)
        allowed, retry_after = config_rate_limiter.check(ip)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Config rate limit exceeded. Retry after {retry_after:.0f}s.",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )
