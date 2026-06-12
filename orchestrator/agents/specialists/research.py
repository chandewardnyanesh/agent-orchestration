"""Research Specialist Agent — web search + API lookups."""
from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from orchestrator.agents.base import BaseAgent, AgentResult, run_tool_loop
from orchestrator.config import get_settings
from orchestrator.tools.registry import tool_registry

logger = logging.getLogger(__name__)
settings = get_settings()

SYSTEM_PROMPT = """You are a Research Specialist. Your job is to find accurate,
up-to-date information to fulfill the assigned subtask.

Available tools: web_search, api_client.

Guidelines:
- Use web_search for general research.
- Cite your sources.
- Be concise — return only what is relevant to the subtask.
- If you cannot find reliable information, say so clearly."""


class ResearchAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="research",
            model=settings.specialist_model,
        )
        self._tools = [
            tool_registry.get("web_search"),
            tool_registry.get("api_client"),
        ]

    def invoke(self, inputs: dict) -> AgentResult:
        """
        inputs:
            subtask_description (str)
            context             (str) - prior subtask outputs available
        """
        subtask: str = inputs["subtask_description"]
        context: str = inputs.get("context", "")

        # Build a tool-augmented LLM chain
        llm_with_tools = self.llm.bind_tools(
            [t for t in self._tools if t is not None]
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Prior context:\n{context}\n\nSubtask:\n{subtask}"),
        ]

        response, tool_records, total_tokens = run_tool_loop(
            llm_with_tools, messages, [t for t in self._tools if t is not None]
        )

        return AgentResult(
            output=response.content,
            confidence=0.85,
            tool_calls=tool_records,
            tokens_used=total_tokens,
        )
