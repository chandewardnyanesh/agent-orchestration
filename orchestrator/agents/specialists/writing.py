"""Writing Specialist Agent — drafting documents, summaries, reports."""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from orchestrator.agents.base import BaseAgent, AgentResult
from orchestrator.config import get_settings

settings = get_settings()

SYSTEM_PROMPT = """You are a Writing Specialist. You produce polished, professional
written content from research and analysis provided in context.

Guidelines:
- Match tone and format to the subtask specification.
- Use headers and bullet points only when appropriate.
- Do not invent facts — draw only from the provided context."""


class WritingAgent(BaseAgent):

    def __init__(self):
        super().__init__(name="writing", model=settings.specialist_model)

    def invoke(self, inputs: dict) -> AgentResult:
        subtask: str = inputs["subtask_description"]
        context: str = inputs.get("context", "")

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Research & analysis context:\n{context}\n\nWriting task:\n{subtask}"),
        ]
        response = self.llm.invoke(messages)
        usage = getattr(response, "usage_metadata", {}) or {}
        return AgentResult(
            output=response.content,
            confidence=0.90,
            tokens_used=usage.get("total_tokens", 0),
        )
