"""
Security regression tests — file_reader path traversal, rate limiter, API auth.

Run: pytest tests/test_security.py -v
"""

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest




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


def test_rate_limiter_evicts_expired_keys(monkeypatch):
    from src.middleware import rate_limiter
    from src.middleware.rate_limiter import SlidingWindowLimiter

    now = 1000.0
    monkeypatch.setattr(rate_limiter.time, "monotonic", lambda: now)
    lim = SlidingWindowLimiter(max_requests=1, window_seconds=10)
    lim.check("old-ip")

    now = 1011.0
    lim.check("new-ip")

    assert "old-ip" not in lim._buckets
    assert "new-ip" in lim._buckets


def test_rate_limiter_caps_key_count():
    from src.middleware.rate_limiter import SlidingWindowLimiter

    lim = SlidingWindowLimiter(max_requests=2, window_seconds=60, max_keys=2)
    lim.check("ip-a")
    lim.check("ip-b")
    lim.check("ip-c")

    assert len(lim._buckets) <= 2




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


def test_config_vllm_alias_uses_config_rate_limit():
    from fastapi.routing import APIRoute
    from src.api.router import router
    from src.middleware.rate_limiter import rate_limit_config

    route = next(
        route
        for route in router.routes
        if isinstance(route, APIRoute) and route.path == "/api/config/vllm"
    )

    dependencies = [dep.call for dep in route.dependant.dependencies]
    assert rate_limit_config in dependencies


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



def test_calculator_rejects_huge_exponent():
    from src.tools.calculator import calculator

    result = calculator.invoke({"expression": "2**10000000000"})

    assert "Exponent too large" in result



def test_url_guard_blocks_loopback_ip():
    from src.security.url_guard import URLFetchError, validate_public_http_url

    with pytest.raises(URLFetchError):
        validate_public_http_url("http://127.0.0.1:8080/private")


def test_url_guard_blocks_localhost(monkeypatch):
    from src.security import url_guard
    from src.security.url_guard import URLFetchError

    monkeypatch.setattr(url_guard, "_resolve_host", lambda _host: ["127.0.0.1"])

    with pytest.raises(URLFetchError):
        url_guard.validate_public_http_url("http://localhost/private")


def test_url_guard_blocks_link_local_ip():
    from src.security.url_guard import URLFetchError, validate_public_http_url

    with pytest.raises(URLFetchError):
        validate_public_http_url("http://169.254.169.254/latest/meta-data")


def test_url_guard_blocks_private_lan_ip():
    from src.security.url_guard import URLFetchError, validate_public_http_url

    with pytest.raises(URLFetchError):
        validate_public_http_url("http://192.168.1.10/admin")


def test_url_guard_accepts_public_hostname(monkeypatch):
    from src.security import url_guard

    monkeypatch.setattr(url_guard, "_resolve_host", lambda _host: ["93.184.216.34"])

    assert url_guard.validate_public_http_url("https://example.com/page") == "https://example.com/page"


@pytest.mark.anyio
async def test_url_fetch_connects_to_validated_ip(monkeypatch):
    from src.security import url_guard

    captured = {}

    class FakeWriter:
        def write(self, data):
            captured["request"] = data.decode("ascii")

        async def drain(self):
            return None

        def close(self):
            return None

        async def wait_closed(self):
            return None

    async def fake_open_connection(*, host, port, ssl=None, server_hostname=None):
        captured["host"] = host
        captured["port"] = port
        captured["server_hostname"] = server_hostname
        return url_guard.asyncio.StreamReader(limit=2**16), FakeWriter()

    async def fake_read_headers(reader, *, timeout, max_bytes):
        return b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n", b""

    async def fake_read_body(reader, *, initial, content_length, chunked, timeout, max_bytes):
        return b"ok"

    monkeypatch.setattr(url_guard, "_resolve_host", lambda _host: ["93.184.216.34"])
    monkeypatch.setattr(url_guard.asyncio, "open_connection", fake_open_connection)
    monkeypatch.setattr(url_guard, "_read_headers", fake_read_headers)
    monkeypatch.setattr(url_guard, "_read_body", fake_read_body)

    final_url, text = await url_guard.fetch_public_url_text("https://example.com/page")

    assert final_url == "https://example.com/page"
    assert text == "ok"
    assert captured["host"] == "93.184.216.34"
    assert captured["port"] == 443
    assert captured["server_hostname"] == "example.com"


@pytest.mark.anyio
async def test_url_fetch_blocks_redirect_to_private_ip(monkeypatch):
    import httpx

    from src.security import url_guard
    from src.security.url_guard import URLFetchError

    def fake_resolve(host: str) -> list[str]:
        if host == "example.com":
            return ["93.184.216.34"]
        if host == "127.0.0.1":
            return ["127.0.0.1"]
        return ["93.184.216.34"]

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url):
            return httpx.Response(
                status_code=302,
                headers={"location": "http://127.0.0.1/admin"},
                request=httpx.Request("GET", url),
            )

    monkeypatch.setattr(url_guard, "_resolve_host", fake_resolve)
    monkeypatch.setattr(url_guard.httpx, "AsyncClient", FakeClient)

    with pytest.raises(URLFetchError):
        await url_guard.fetch_public_url_text("https://example.com/start")


def test_settings_rejects_placeholder_admin_password(monkeypatch):
    from pydantic import ValidationError
    from src.config import Settings

    monkeypatch.setenv("APP_ADMIN_PASSWORD", "change-me")
    monkeypatch.setenv("APP_PASSWORD_SALT", "test-password-salt")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_rejects_example_salt_placeholder(monkeypatch):
    from pydantic import ValidationError
    from src.config import Settings

    monkeypatch.setenv("APP_ADMIN_PASSWORD", "strong-test-password")
    monkeypatch.setenv("APP_PASSWORD_SALT", "REPLACE_WITH_RANDOM_SALT")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_accepts_non_placeholder_secrets(monkeypatch):
    from src.config import Settings

    monkeypatch.setenv("APP_ADMIN_PASSWORD", "strong-test-password")
    monkeypatch.setenv("APP_PASSWORD_SALT", "random-test-salt")

    settings = Settings(_env_file=None)

    assert settings.app_admin_password == "strong-test-password"
