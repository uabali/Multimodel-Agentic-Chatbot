"""
Standalone MCP server (FastMCP stdio) — demo tools for external MCP connection.

Source: Final-Project/src/mcp/server.py (full port).

Run: python -m src.mcp.server
Then connect via Chainlit UI "Add MCP" with stdio transport.
"""

import os
import platform
from pathlib import Path
from typing import Any


def list_uploaded_files() -> list[str]:
    root = Path("uploads")
    if not root.exists():
        return []
    files = [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()]
    files.sort()
    return files


def get_system_info() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cwd": os.getcwd(),
    }


def main():
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("rag-agent-local-mcp")

    @mcp.tool()
    def list_uploaded_files_tool() -> list[str]:
        """List all files in the uploads directory."""
        return list_uploaded_files()

    @mcp.tool()
    def get_system_info_tool() -> dict[str, Any]:
        """Return basic system info (platform, Python version, cwd)."""
        return get_system_info()

    mcp.run()


if __name__ == "__main__":
    main()
