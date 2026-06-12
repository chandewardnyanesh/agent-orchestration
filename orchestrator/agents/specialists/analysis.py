"""Analysis Specialist Agent — data processing, extraction, statistics."""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from orchestrator.agents.base import BaseAgent, AgentResult, run_tool_loop
from orchestrator.config import get_settings
from orchestrator.tools.registry import tool_registry

settings = get_settings()

SYSTEM_PROMPT = """You are a Data Analysis Specialist. You receive structured or
unstructured data and produce analysis, summaries, or extracted insights.

Available tools: database, file_ops.

Guidelines:
- Prefer structured output (JSON / tables) where the subtask asks for it.
- Call out assumptions and data limitations.
- Be precise with numbers — no hallucinated statistics."""


class AnalysisAgent(BaseAgent):

    def __init__(self):
        super().__init__(name="analysis", model=settings.specialist_model)
        # "file_read" and "file_write" are the registry keys (not "file_ops")
        self._tools = [
            tool_registry.get("database"),
            tool_registry.get("file_read"),
            tool_registry.get("file_write"),
        ]

    def invoke(self, inputs: dict) -> AgentResult:
        subtask: str = inputs["subtask_description"]
        context: str = inputs.get("context", "")

        active_tools = [t for t in self._tools if t is not None]
        llm_with_tools = self.llm.bind_tools(active_tools)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Prior context:\n{context}\n\nSubtask:\n{subtask}"),
        ]
        response, tool_records, total_tokens = run_tool_loop(
            llm_with_tools, messages, active_tools
        )
        return AgentResult(
            output=response.content,
            confidence=0.85,
            tool_calls=tool_records,
            tokens_used=total_tokens,
        )
