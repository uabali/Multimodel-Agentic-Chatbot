"""
Load MCP stdio server definitions and build LangChain tools via langchain-mcp-adapters.

Note: MultiServerMCPClient no longer supports `async with client` lifecycle; each
`get_tools()` call loads tool metadata (and per-call sessions are managed internally).

Process-wide tool list cache: first `get_mcp_tools()` loads all configured stdio servers;
later calls reuse the list until the process restarts or `invalidate_mcp_tools_cache()`.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent / "mcp_config.json"

# Process-level cache (survives new Chainlit threads / on_chat_start)
_mcp_tools_cache: list | None = None


def is_mcp_tools_cache_warm() -> bool:
    return _mcp_tools_cache is not None


def invalidate_mcp_tools_cache() -> None:
    global _mcp_tools_cache
    _mcp_tools_cache = None


def _sync_mcp_env_from_settings() -> None:
    """Expose Settings/.env values to `${VAR}` substitution for MCP subprocess env."""
    try:
        from src.config import settings

        pairs = [
            ("BRAVE_API_KEY", settings.brave_api_key),
            ("GOOGLE_CLIENT_ID", settings.google_client_id),
            ("GOOGLE_CLIENT_SECRET", settings.google_client_secret),
            ("MCP_FILESYSTEM_ROOT", settings.mcp_filesystem_root),
        ]
        for key, val in pairs:
            if val and not os.getenv(key):
                os.environ[key] = str(val).strip()
    except Exception as e:
        logger.debug("MCP env sync skipped: %s", e)


def _substitute_env_in_text(raw: str) -> str:
    for key, val in os.environ.items():
        raw = raw.replace(f"${{{key}}}", val)
    return raw


def _parse_config_dict(raw_text: str) -> dict[str, Any]:
    return json.loads(raw_text)


def _filesystem_root() -> str:
    try:
        from src.config import settings

        root = (settings.mcp_filesystem_root or os.getenv("MCP_FILESYSTEM_ROOT", "")).strip()
        if root:
            return str(Path(root).expanduser().resolve())
        return str(settings.upload_dir.resolve())
    except Exception:
        root = os.getenv("MCP_FILESYSTEM_ROOT", "").strip()
        if root:
            return str(Path(root).expanduser().resolve())
        return str(Path("uploads").resolve())


def load_mcp_config() -> dict[str, Any]:
    """Return parsed JSON with `${VAR}` expanded in the raw file text."""
    _sync_mcp_env_from_settings()
    raw = CONFIG_PATH.read_text(encoding="utf-8")
    raw = _substitute_env_in_text(raw)
    # Must not splice a raw Windows path into JSON text: backslashes break `json.loads`
    # (e.g. `\Users` → invalid `\U` escape). Inject a JSON-encoded string instead.
    root_json = json.dumps(_filesystem_root())
    raw = raw.replace('"__FILESYSTEM_ROOT__"', root_json)
    return _parse_config_dict(raw)


def load_mcp_connections() -> dict[str, Any]:
    """Connections dict for MultiServerMCPClient (unwrap `mcpServers`, ensure `transport`)."""
    data = load_mcp_config()
    servers = data.get("mcpServers", data)
    if not isinstance(servers, dict):
        return {}

    out: dict[str, Any] = {}
    for name, cfg in servers.items():
        if not isinstance(cfg, dict):
            continue
        if cfg.get("disabled"):
            logger.debug("MCP server %r skipped (disabled in config)", name)
            continue
        entry = dict(cfg)
        if entry.get("transport") is None and entry.get("command"):
            entry["transport"] = "stdio"
        out[name] = entry

    try:
        from src.config import settings

        if not (settings.google_client_id or "").strip() or not (
            settings.google_client_secret or ""
        ).strip():
            out.pop("google-calendar", None)
            out.pop("gmail", None)
        if not (settings.brave_api_key or os.getenv("BRAVE_API_KEY", "")).strip():
            out.pop("brave-search", None)
    except Exception:
        pass

    return out


def filter_connections(connections: dict[str, Any], *names: str) -> dict[str, Any]:
    """Subset of servers by name (e.g. brave-only for web search)."""
    return {k: v for k, v in connections.items() if k in names and v is not None}


async def get_mcp_tools(*, server_name: str | None = None, force_refresh: bool = False) -> list:
    """Return LangChain tools from configured MCP servers.

    Loads each server separately so a missing Google OAuth bundle does not
    prevent Brave/filesystem tools from registering.

    When ``server_name`` is None, results are cached for the lifetime of the process
    unless ``force_refresh`` is True.
    """
    global _mcp_tools_cache

    connections = load_mcp_connections()
    if not connections:
        return []

    if (
        server_name is None
        and not force_refresh
        and _mcp_tools_cache is not None
    ):
        return list(_mcp_tools_cache)

    names = [server_name] if server_name else list(connections.keys())
    all_tools: list = []

    for name in names:
        conn = connections.get(name)
        if not conn:
            continue
        try:
            client = MultiServerMCPClient({name: conn}, tool_name_prefix=True)
            tools = await client.get_tools(server_name=name)
            all_tools.extend(tools)
        except Exception as e:
            logger.warning("MCP server %r unavailable: %s", name, e)

    if server_name is None:
        _mcp_tools_cache = all_tools

    return all_tools
