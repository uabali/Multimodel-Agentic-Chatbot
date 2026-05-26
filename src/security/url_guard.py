"""SSRF-safe URL validation and fetching for user-supplied ingest URLs."""

from __future__ import annotations

import ipaddress
import asyncio
import re
import socket
import ssl
from urllib.parse import unquote, urljoin, urlparse

import httpx


class URLFetchError(ValueError):
    """Raised when a URL is unsafe or cannot be fetched within guard rails."""


def _is_public_ip(raw_ip: str) -> bool:
    """Kısa: `_is_public_ip` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
    try:
        ip = ipaddress.ip_address(raw_ip)
    except ValueError:
        return False
    return ip.is_global


def _parse_ip_literal(host: str) -> str | None:
    """Kısa: `_parse_ip_literal` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
    try:
        return str(ipaddress.ip_address(host.strip("[]")))
    except ValueError:
        return None


def _resolve_host(hostname: str) -> list[str]:
    """Kısa: `_resolve_host` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
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
    normalized, _ = _validate_public_http_url_with_ip(url)
    return normalized


def _validate_public_http_url_with_ip(url: str) -> tuple[str, str]:
    """Return a normalized URL and one pre-resolved public IP for pinned fetching."""
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in {"http", "https"}:
        raise URLFetchError("Only http and https URLs are allowed.")
    if not parsed.hostname:
        raise URLFetchError("URL must include a hostname.")
    if parsed.username or parsed.password:
        raise URLFetchError("URLs with embedded credentials are not allowed.")

    host = parsed.hostname.strip()
    ip_literal = _parse_ip_literal(host)
    if ip_literal is None:
        ips = _resolve_host(host)
    else:
        ips = [ip_literal]

    blocked = [ip for ip in ips if not _is_public_ip(ip)]
    if blocked:
        raise URLFetchError(f"URL resolves to a non-public address: {blocked[0]}")

    return parsed.geturl(), ips[0]


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
    current = url
    for _ in range(max_redirects + 1):
        current, resolved_ip = _validate_public_http_url_with_ip(current)
        status_code, headers, data = await _fetch_once_pinned(
            current,
            resolved_ip,
            timeout=timeout,
            max_bytes=max_bytes,
        )

        if 300 <= status_code < 400:
            location = headers.get("location")
            if not location:
                raise URLFetchError("Redirect response did not include a Location header.")
            current = urljoin(current, location)
            continue

        if status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {status_code}",
                request=httpx.Request("GET", current),
                response=httpx.Response(status_code=status_code, request=httpx.Request("GET", current)),
            )

        content_type = headers.get("content-type", "")
        encoding = _encoding_from_content_type(content_type) or "utf-8"
        return current, data.decode(encoding, errors="replace")

    raise URLFetchError("Too many redirects.")


def _encoding_from_content_type(content_type: str) -> str | None:
    match = re.search(r"charset=([^;\s]+)", content_type, re.IGNORECASE)
    return match.group(1).strip("\"'") if match else None


async def _fetch_once_pinned(
    url: str,
    resolved_ip: str,
    *,
    timeout: float,
    max_bytes: int,
) -> tuple[int, dict[str, str], bytes]:
    """Fetch one URL by connecting to the already validated IP address."""
    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        raise URLFetchError("URL must include a hostname.")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    target = parsed.path or "/"
    if parsed.query:
        target = f"{target}?{parsed.query}"

    ssl_context = None
    server_hostname = None
    if parsed.scheme == "https":
        ssl_context = ssl.create_default_context()
        server_hostname = host

    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(
                host=resolved_ip,
                port=port,
                ssl=ssl_context,
                server_hostname=server_hostname,
            ),
            timeout=timeout,
        )
    except Exception as exc:
        raise URLFetchError(f"URL could not be fetched: {exc}") from exc

    try:
        host_header = host if parsed.port is None else f"{host}:{parsed.port}"
        request = (
            f"GET {target} HTTP/1.1\r\n"
            f"Host: {host_header}\r\n"
            "User-Agent: FRAPPE-RAG-Ingest/1.0\r\n"
            "Accept: text/*, application/json, application/xml, application/xhtml+xml, */*;q=0.5\r\n"
            "Accept-Encoding: identity\r\n"
            "Connection: close\r\n\r\n"
        )
        writer.write(request.encode("ascii"))
        await asyncio.wait_for(writer.drain(), timeout=timeout)

        header_bytes, body_prefix = await _read_headers(reader, timeout=timeout, max_bytes=max_bytes)
        header_text = header_bytes.decode("iso-8859-1", errors="replace")
        status_line, _, raw_headers = header_text.partition("\r\n")
        parts = status_line.split(" ", 2)
        if len(parts) < 2 or not parts[1].isdigit():
            raise URLFetchError("URL returned an invalid HTTP response.")
        status_code = int(parts[1])
        headers = _parse_headers(raw_headers)

        content_length = headers.get("content-length")
        declared_size = None
        if content_length:
            try:
                declared_size = int(content_length)
            except ValueError:
                declared_size = 0
            if declared_size > max_bytes:
                raise URLFetchError(f"URL content is too large (limit {max_bytes} bytes).")

        data = await _read_body(
            reader,
            initial=body_prefix,
            content_length=declared_size,
            chunked=headers.get("transfer-encoding", "").lower() == "chunked",
            timeout=timeout,
            max_bytes=max_bytes,
        )
        return status_code, headers, data
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


async def _read_headers(reader: asyncio.StreamReader, *, timeout: float, max_bytes: int) -> tuple[bytes, bytes]:
    data = bytearray()
    while b"\r\n\r\n" not in data:
        chunk = await asyncio.wait_for(reader.read(4096), timeout=timeout)
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > min(max_bytes, 64_000):
            raise URLFetchError("URL response headers are too large.")
    header_end = data.find(b"\r\n\r\n")
    if header_end < 0:
        raise URLFetchError("URL response did not include complete headers.")
    return bytes(data[: header_end + 4]), bytes(data[header_end + 4 :])


async def _read_body(
    reader: asyncio.StreamReader,
    *,
    initial: bytes,
    content_length: int | None,
    chunked: bool,
    timeout: float,
    max_bytes: int,
) -> bytes:
    data = bytearray(initial)
    if chunked:
        while True:
            if b"\r\n0\r\n" in data or b"\r\n0\r\n\r\n" in data:
                break
            chunk = await asyncio.wait_for(reader.read(65_536), timeout=timeout)
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > max_bytes + 128_000:
                raise URLFetchError(f"URL content is too large (limit {max_bytes} bytes).")
        return _decode_chunked_body(bytes(data), max_bytes=max_bytes)

    if content_length is not None:
        while len(data) < content_length:
            chunk = await asyncio.wait_for(reader.read(min(65_536, content_length - len(data))), timeout=timeout)
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > max_bytes:
                raise URLFetchError(f"URL content is too large (limit {max_bytes} bytes).")
        return bytes(data[:content_length])

    while True:
        chunk = await asyncio.wait_for(reader.read(65_536), timeout=timeout)
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > max_bytes:
            raise URLFetchError(f"URL content is too large (limit {max_bytes} bytes).")
    return bytes(data)


def _decode_chunked_body(raw: bytes, *, max_bytes: int) -> bytes:
    decoded = bytearray()
    pos = 0
    while True:
        line_end = raw.find(b"\r\n", pos)
        if line_end < 0:
            raise URLFetchError("Chunked response ended before a complete chunk header.")
        size_text = raw[pos:line_end].split(b";", 1)[0].strip()
        try:
            size = int(size_text, 16)
        except ValueError as exc:
            raise URLFetchError("Chunked response included an invalid chunk size.") from exc
        pos = line_end + 2
        if size == 0:
            break
        chunk_end = pos + size
        if chunk_end + 2 > len(raw):
            raise URLFetchError("Chunked response ended before a complete chunk body.")
        decoded.extend(raw[pos:chunk_end])
        if len(decoded) > max_bytes:
            raise URLFetchError(f"URL content is too large (limit {max_bytes} bytes).")
        pos = chunk_end + 2
    return bytes(decoded)


def _parse_headers(raw_headers: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    for line in raw_headers.split("\r\n"):
        if not line or ":" not in line:
            continue
        name, value = line.split(":", 1)
        headers[name.strip().lower()] = value.strip()
    return headers
