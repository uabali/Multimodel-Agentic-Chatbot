"""
Security regression tests — file_reader path traversal, rate limiter, API auth.

Run: pytest tests/test_security.py -v
"""

import hashlib
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ─── file_reader: path traversal ─────────────────────────────────────────────


def _make_upload_dir(tmp_path: Path) -> Path:
    upload = tmp_path / "uploads"
    upload.mkdir()
    (upload / "safe.txt").write_text("hello")
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET")
    return upload


def test_path_traversal_blocked(tmp_path):
    upload_dir = _make_upload_dir(tmp_path)

    mock_settings = MagicMock()
    mock_settings.upload_dir = upload_dir

    with patch("src.tools.file_reader.settings", mock_settings):
        from src.tools.file_reader import read_uploaded_file

        # ../secret.txt must NOT escape the upload dir
        result = read_uploaded_file.invoke({"filename": "../secret.txt"})
        assert "Access denied" in result, f"Expected access denied, got: {result}"
        assert "TOP SECRET" not in result


def test_path_traversal_absolute_blocked(tmp_path):
    upload_dir = _make_upload_dir(tmp_path)

    mock_settings = MagicMock()
    mock_settings.upload_dir = upload_dir

    with patch("src.tools.file_reader.settings", mock_settings):
        from src.tools.file_reader import read_uploaded_file

        result = read_uploaded_file.invoke({"filename": "/etc/passwd"})
        assert "Access denied" in result or "File not found" in result


def test_safe_file_readable(tmp_path):
    upload_dir = _make_upload_dir(tmp_path)

    mock_settings = MagicMock()
    mock_settings.upload_dir = upload_dir

    with patch("src.tools.file_reader.settings", mock_settings):
        from src.tools.file_reader import read_uploaded_file

        result = read_uploaded_file.invoke({"filename": "safe.txt"})
        assert "hello" in result


# ─── rate_limiter: sliding window ────────────────────────────────────────────


def test_rate_limiter_allows_under_limit():
    from src.middleware.rate_limiter import SlidingWindowLimiter

    lim = SlidingWindowLimiter(max_requests=3, window_seconds=60)
    for _ in range(3):
        allowed, _ = lim.check("test-ip")
        assert allowed


def test_rate_limiter_blocks_over_limit():
    from src.middleware.rate_limiter import SlidingWindowLimiter

    lim = SlidingWindowLimiter(max_requests=2, window_seconds=60)
    lim.check("ip")
    lim.check("ip")
    allowed, retry_after = lim.check("ip")
    assert not allowed
    assert retry_after > 0


def test_rate_limiter_separate_keys():
    from src.middleware.rate_limiter import SlidingWindowLimiter

    lim = SlidingWindowLimiter(max_requests=1, window_seconds=60)
    lim.check("ip-a")
    allowed_a, _ = lim.check("ip-a")
    allowed_b, _ = lim.check("ip-b")
    assert not allowed_a
    assert allowed_b


# ─── rate_limiter: X-Forwarded-For only trusted from known proxies ──────────


def test_xff_ignored_from_untrusted_client():
    """An untrusted client cannot spoof their IP via X-Forwarded-For."""
    from src.middleware.rate_limiter import _get_client_ip

    request = MagicMock()
    request.client.host = "1.2.3.4"  # not in TRUSTED_PROXIES
    request.headers.get = lambda h, d=None: "9.9.9.9" if h == "X-Forwarded-For" else d

    with patch("src.middleware.rate_limiter._TRUSTED_PROXIES", frozenset(["127.0.0.1"])):
        ip = _get_client_ip(request)
    assert ip == "1.2.3.4", f"Should use real client IP, got {ip}"


def test_xff_trusted_from_known_proxy():
    """Trusted reverse proxy's X-Forwarded-For is honoured."""
    from src.middleware.rate_limiter import _get_client_ip

    request = MagicMock()
    request.client.host = "127.0.0.1"
    request.headers.get = lambda h, d=None: "5.6.7.8" if h == "X-Forwarded-For" else d

    with patch("src.middleware.rate_limiter._TRUSTED_PROXIES", frozenset(["127.0.0.1"])):
        ip = _get_client_ip(request)
    assert ip == "5.6.7.8", f"Should use forwarded IP, got {ip}"


# ─── API router: admin auth dependency ───────────────────────────────────────


def _make_settings(username="admin", password="s3cr3t", salt="testsalt"):
    s = MagicMock()
    s.app_admin_username = username
    s.app_admin_password = password
    s.app_password_salt = salt
    # LLMConfigResponse fields
    s.llm_backend = "llama.cpp"
    s.llm_server_url = "http://localhost:8080/v1"
    s.llm_model_name = "test-model"
    s.llm_enable_thinking = False
    s.rag_max_tokens = 1536
    s.chat_max_tokens = 1024
    s.agent_max_tokens = 2048
    return s


def _hash_pw(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 210_000).hex()


@pytest.mark.anyio
async def test_require_admin_no_credentials():
    from fastapi import HTTPException
    from src.api.router import require_admin

    with pytest.raises(HTTPException) as exc_info:
        await require_admin(credentials=None)
    assert exc_info.value.status_code == 401


@pytest.mark.anyio
async def test_require_admin_wrong_password():
    from fastapi import HTTPException
    from fastapi.security import HTTPBasicCredentials
    from src.api.router import require_admin

    mock_settings = _make_settings(password="correct", salt="salt123")
    creds = HTTPBasicCredentials(username="admin", password="wrong")

    with patch("src.config.settings", mock_settings):
        with pytest.raises(HTTPException) as exc_info:
            await require_admin(credentials=creds)
    assert exc_info.value.status_code == 401


@pytest.mark.anyio
async def test_require_admin_correct_credentials():
    from fastapi.security import HTTPBasicCredentials
    from src.api.router import require_admin

    mock_settings = _make_settings(password="correct", salt="salt123")
    creds = HTTPBasicCredentials(username="admin", password="correct")

    with patch("src.config.settings", mock_settings):
        result = await require_admin(credentials=creds)
    assert result is None  # dependency returns None on success


# ─── API router: integration with TestClient ─────────────────────────────────


def _make_app():
    """Build a minimal FastAPI app with only the admin router mounted."""
    import base64

    from fastapi import FastAPI
    from src.api.router import router

    app = FastAPI()
    app.include_router(router)
    return app


def _basic_header(username: str, password: str) -> str:
    import base64

    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return f"Basic {token}"


def test_config_endpoint_rejects_unauthenticated():
    from fastapi.testclient import TestClient

    mock_settings = _make_settings()
    with patch("src.config.settings", mock_settings):
        app = _make_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/config")
    assert resp.status_code == 401


def test_config_endpoint_accepts_admin():
    from fastapi.testclient import TestClient

    mock_settings = _make_settings(username="admin", password="s3cr3t", salt="testsalt")

    with patch("src.config.settings", mock_settings):
        app = _make_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get(
            "/api/config",
            headers={"Authorization": _basic_header("admin", "s3cr3t")},
        )
    assert resp.status_code == 200


def test_health_endpoint_public():
    """Health check must remain publicly accessible (no auth)."""
    from unittest.mock import AsyncMock

    from fastapi.testclient import TestClient

    mock_settings = _make_settings()
    mock_settings.llm_server_url = "http://localhost:8080/v1"
    mock_settings.qdrant_url = "http://localhost:6333"

    with patch("src.config.settings", mock_settings):
        with patch("src.api.router._check_vllm", new=AsyncMock(return_value={"reachable": True, "latency_ms": 1.0, "models": []})):
            with patch("src.api.router._check_qdrant", new=AsyncMock(return_value={"reachable": True, "latency_ms": 1.0})):
                app = _make_app()
                client = TestClient(app, raise_server_exceptions=False)
                resp = client.get("/api/health")
    assert resp.status_code == 200
