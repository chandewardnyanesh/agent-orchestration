"""
Tool Registry
-------------
Central registry for all tools available to specialist agents.
Each tool is a LangChain StructuredTool with metadata for rate-limiting
and access control.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from langchain_core.tools import StructuredTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Simple name → StructuredTool registry."""

    def __init__(self):
        self._tools: dict[str, StructuredTool] = {}

    def register(self, name: str, tool: StructuredTool) -> None:
        self._tools[name] = tool
        logger.debug("registered tool: %s", name)

    def get(self, name: str) -> StructuredTool | None:
        tool = self._tools.get(name)
        if not tool:
            logger.warning("tool '%s' not found in registry", name)
        return tool

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())


# ── Global singleton ──────────────────────────────────────────────────────────
tool_registry = ToolRegistry()


# ── Auto-register built-in tools on import ────────────────────────────────────
def _register_builtins():
    from orchestrator.tools.web_search import web_search_tool
    from orchestrator.tools.file_ops import file_read_tool, file_write_tool
    from orchestrator.tools.code_executor import code_executor_tool
    from orchestrator.tools.database import database_query_tool
    from orchestrator.tools.api_client import api_call_tool

    tool_registry.register("web_search", web_search_tool)
    tool_registry.register("file_read", file_read_tool)
    tool_registry.register("file_write", file_write_tool)
    tool_registry.register("code_executor", code_executor_tool)
    tool_registry.register("database", database_query_tool)
    tool_registry.register("api_client", api_call_tool)


_register_builtins()
