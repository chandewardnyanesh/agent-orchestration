"""Code Specialist Agent — writes, executes, and debugs code."""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from orchestrator.agents.base import BaseAgent, AgentResult, run_tool_loop
from orchestrator.config import get_settings
from orchestrator.tools.registry import tool_registry

settings = get_settings()

SYSTEM_PROMPT = """You are a Code Specialist. You write and execute Python code
to fulfill the assigned subtask. ALWAYS execute code to verify it works before
returning it as output.

Available tools: code_executor (sandboxed).

Guidelines:
- Return the final code AND its execution output.
- Handle exceptions gracefully.
- Never execute code that modifies the host filesystem outside /tmp."""


class CodeAgent(BaseAgent):

    def __init__(self):
        super().__init__(name="code", model=settings.specialist_model)
        self._tools = [tool_registry.get("code_executor")]

    def invoke(self, inputs: dict) -> AgentResult:
        subtask: str = inputs["subtask_description"]
        context: str = inputs.get("context", "")

        active_tools = [t for t in self._tools if t is not None]
        llm_with_tools = self.llm.bind_tools(active_tools)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Context:\n{context}\n\nCoding task:\n{subtask}"),
        ]
        response, tool_records, total_tokens = run_tool_loop(
            llm_with_tools, messages, active_tools
        )
        return AgentResult(
            output=response.content,
            confidence=0.80,
            tool_calls=tool_records,
            tokens_used=total_tokens,
        )
