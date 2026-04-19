"""
Chainlit MCP -> LangChain tool bridge (from Final-Project).
"""

import json
from typing import Any, Optional

from langchain_core.tools import tool


@tool
async def mcp_call(tool_name: str, tool_input_json: str, connection_name: Optional[str] = None) -> str:
    """Call a tool on a connected MCP server via Chainlit session.

    Args:
        tool_name: MCP tool name
        tool_input_json: JSON string of tool input arguments
        connection_name: Optional MCP connection name to target
    """
    import chainlit as cl

    try:
        tool_input: Any = json.loads(tool_input_json) if tool_input_json else {}
    except Exception as e:
        return f"Invalid JSON input: {e}"

    ctx_session = getattr(cl.context, "session", None)
    mcp_sessions = getattr(ctx_session, "mcp_sessions", None)
    if not mcp_sessions:
        return "No MCP session found. Connect an MCP server via the UI first."

    chosen = None
    if connection_name:
        chosen = mcp_sessions.get(connection_name)
        if not chosen:
            return f"MCP connection '{connection_name}' not found."
    else:
        chosen = next(iter(mcp_sessions.values()), None)

    if not chosen:
        return "No active MCP connection."

    mcp_session, _ = chosen
    try:
        result = await mcp_session.call_tool(tool_name, tool_input)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return f"MCP tool call error: {e}"
