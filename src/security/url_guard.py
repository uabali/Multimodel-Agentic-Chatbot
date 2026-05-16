"""SSRF-safe URL validation and fetching for user-supplied ingest URLs."""

from __future__ import annotations

import ipaddress
import re
import socket
from urllib.parse import unquote, urljoin, urlparse

import httpx


class URLFetchError(ValueError):
    """Raised when a URL is unsafe or cannot be fetched within guard rails."""


def _is_public_ip(raw_ip: str) -> bool:
    try:
        ip = ipaddress.ip_address(raw_ip)
    except ValueError:
        return False
    return ip.is_global


def _resolve_host(hostname: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise URLFetchError(f"Hostname could not be resolved: {hostname}") from exc

    ips = sorted({info[4][0] for info in infos if info and info[4]})
    if not ips:
        raise URLFetchError(f"Hostname resolved to no addresses: {hostname}")
    return ips


def validate_public_http_url(url: str) -> str:
    """Return a normalized URL if it is HTTP(S) and resolves only to public IPs."""
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in {"http", "https"}:
        raise URLFetchError("Only http and https URLs are allowed.")
    if not parsed.hostname:
        raise URLFetchError("URL must include a hostname.")
    if parsed.username or parsed.password:
        raise URLFetchError("URLs with embedded credentials are not allowed.")

    host = parsed.hostname.strip()
    try:
        ip_literal = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        ips = _resolve_host(host)
    else:
        ips = [str(ip_literal)]

    blocked = [ip for ip in ips if not _is_public_ip(ip)]
    if blocked:
        raise URLFetchError(f"URL resolves to a non-public address: {blocked[0]}")

    return parsed.geturl()


def safe_filename_from_url(url: str) -> str:
    """Create a bounded, portable TXT filename for fetched URL content."""
    parsed = urlparse(url)
    raw = unquote(f"{parsed.netloc}{parsed.path}".strip("/") or parsed.netloc or "url")
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw).strip("_")[:60] or "url"
    return f"{safe}.txt"


async def fetch_public_url_text(
    url: str,
    *,
    timeout: float = 10.0,
    max_bytes: int = 2_000_000,
    max_redirects: int = 5,
) -> tuple[str, str]:
    """Fetch text from a public HTTP(S) URL, validating every redirect target."""
    current = validate_public_http_url(url)

    async with httpx.AsyncClient(
        follow_redirects=False,
        timeout=httpx.Timeout(timeout),
        headers={"User-Agent": "FRAPPE-RAG-Ingest/1.0"},
    ) as client:
        for _ in range(max_redirects + 1):
            response = await client.get(current)

            if 300 <= response.status_code < 400:
                location = response.headers.get("location")
                if not location:
                    raise URLFetchError("Redirect response did not include a Location header.")
                current = validate_public_http_url(urljoin(current, location))
                continue

            response.raise_for_status()

            content_length = response.headers.get("content-length")
            if content_length:
                try:
                    declared_size = int(content_length)
                except ValueError:
                    declared_size = 0
                if declared_size > max_bytes:
                    raise URLFetchError(f"URL content is too large (limit {max_bytes} bytes).")

            data = response.content
            if len(data) > max_bytes:
                raise URLFetchError(f"URL content is too large (limit {max_bytes} bytes).")

            encoding = response.encoding or "utf-8"
            return response.url.__str__(), data.decode(encoding, errors="replace")

    raise URLFetchError("Too many redirects.")
