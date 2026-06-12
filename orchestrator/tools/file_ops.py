"""File read/write tools — scoped to /tmp/agent_workspace for safety."""
import os
from pathlib import Path
from langchain_core.tools import tool

WORKSPACE = Path("/tmp/agent_workspace")
WORKSPACE.mkdir(exist_ok=True)


def _safe_path(filename: str) -> Path:
    path = (WORKSPACE / filename).resolve()
    if not str(path).startswith(str(WORKSPACE)):
        raise ValueError(f"Path traversal attempt: {filename}")
    return path


@tool
def file_read_tool(filename: str) -> str:
    """Read a file from the agent workspace (/tmp/agent_workspace)."""
    try:
        return _safe_path(filename).read_text()
    except FileNotFoundError:
        return f"File '{filename}' not found."
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def file_write_tool(filename: str, content: str) -> str:
    """Write content to a file in the agent workspace (/tmp/agent_workspace)."""
    try:
        path = _safe_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return f"Written {len(content)} bytes to {filename}"
    except Exception as e:
        return f"Error writing file: {e}"
