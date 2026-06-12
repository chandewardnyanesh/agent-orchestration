"""
Reviewer Agent
--------------
Evaluates specialist output against the original subtask description.
Returns a quality score (0-1) and either approves or requests a retry
with structured feedback.
"""
from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from orchestrator.agents.base import BaseAgent, AgentResult
from orchestrator.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ReviewDecision(BaseModel):
    approved: bool
    quality_score: float = Field(..., ge=0.0, le=1.0)
    feedback: str = Field(default="", description="Specific feedback if rejected")
    issues: list[str] = Field(default_factory=list)


_SYSTEM_PROMPT_TEMPLATE = """You are a Reviewer Agent. Your job is to evaluate whether a specialist
agent's output adequately fulfills the assigned subtask.

Criteria:
- Completeness: Does the output cover what was asked?
- Accuracy:     Is the information correct and well-reasoned?
- Format:       Does it match the expected output format?
- Quality:      Is it at a professional standard?

Respond with ONLY a valid JSON object with keys: approved (bool), quality_score (float 0-1),
feedback (string, empty if approved), issues (list of strings).

Approve if quality_score >= {threshold}. Reject otherwise."""

# Alias used throughout the class
SYSTEM_PROMPT = _SYSTEM_PROMPT_TEMPLATE


class ReviewerAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="reviewer",
            model=settings.reviewer_model,
        )
        self._threshold = settings.reviewer_quality_threshold

    def invoke(self, inputs: dict) -> AgentResult:
        """
        inputs:
            subtask_description  (str) - what the specialist was asked to do
            specialist_output    (str) - what the specialist produced
            expected_format      (str) - expected output format hint
            attempt_number       (int) - retry count (1-indexed)
        """
        subtask_desc: str = inputs["subtask_description"]
        specialist_output: str = inputs["specialist_output"]
        expected_format: str = inputs.get("expected_format", "text")
        attempt: int = inputs.get("attempt_number", 1)

        system = SYSTEM_PROMPT.format(threshold=self._threshold)
        prompt = f"""Subtask description:
{subtask_desc}

Expected output format: {expected_format}
Attempt number: {attempt}

Specialist output to review:
---
{specialist_output}
---

Evaluate and return JSON."""

        messages = [SystemMessage(content=system), HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()

        decision = ReviewDecision(**json.loads(raw))

        if not decision.approved:
            logger.info(
                "reviewer rejected output score=%.2f attempt=%d feedback=%s",
                decision.quality_score, attempt, decision.feedback,
            )

        usage = getattr(response, "usage_metadata", {}) or {}
        return AgentResult(
            output=decision.model_dump(),
            confidence=decision.quality_score,
            tokens_used=usage.get("total_tokens", 0),
        )
