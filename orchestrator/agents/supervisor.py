"""
Supervisor Agent
----------------
Receives a raw user task → queries long-term memory for context →
produces a structured ExecutionPlan (ordered subtasks with specialist
assignments) → checks confidence → flags for HITL if needed.
"""
from __future__ import annotations

import json
import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from orchestrator.agents.base import BaseAgent, AgentResult
from orchestrator.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Domain models ────────────────────────────────────────────────────────────

SpecialistType = Literal["research", "analysis", "writing", "code"]


class SubTask(BaseModel):
    id: str = Field(..., description="Unique subtask identifier, e.g. 'st-1'")
    description: str
    specialist: SpecialistType
    depends_on: list[str] = Field(default_factory=list, description="IDs of subtasks that must complete first")
    expected_output_format: str = Field(default="text")
    estimated_complexity: Literal["low", "medium", "high"] = "medium"


class ExecutionPlan(BaseModel):
    task_id: str
    original_request: str
    subtasks: list[SubTask]
    confidence: float = Field(..., ge=0.0, le=1.0, description="Supervisor confidence in this plan")
    reasoning: str = Field(default="", description="Why this decomposition was chosen")
    requires_human_approval: bool = False


# ── Supervisor Agent ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Supervisor Agent in a multi-agent orchestration system.
Your sole job is to decompose a complex user task into an ordered list of subtasks,
each assigned to a specialist agent.

Specialist roster:
- research   → web search, API lookups, fact-finding
- analysis   → data processing, statistical work, structured extraction
- writing    → drafting documents, summaries, reports
- code       → writing, running, or debugging code

Rules:
1. Produce a valid JSON object matching the ExecutionPlan schema.
2. Set confidence (0-1) based on how clear the task is and how confident you are in the plan.
3. If confidence < {threshold}, set requires_human_approval = true.
4. Respect dependencies — list subtask IDs in depends_on for sequential work.
5. Be concise in reasoning (≤3 sentences).
"""


class SupervisorAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="supervisor",
            model=settings.supervisor_model,
        )
        self._threshold = settings.supervisor_confidence_threshold

    # ------------------------------------------------------------------
    def invoke(self, inputs: dict) -> AgentResult:
        """
        inputs:
            task_id          (str)  - unique task id
            user_request     (str)  - raw user task
            memory_context   (str)  - retrieved long-term memories (optional)
            rejection_notes  (str)  - human reviewer's rejection reason (Phase 3
                                      re-plan path); absent on first attempt
        """
        task_id: str = inputs["task_id"]
        user_request: str = inputs["user_request"]
        memory_context: str = inputs.get("memory_context", "No prior context.")
        rejection_notes: str = inputs.get("rejection_notes", "")

        system = SYSTEM_PROMPT.format(threshold=self._threshold)

        rejection_block = ""
        if rejection_notes:
            rejection_block = (
                f"\n⚠️  PREVIOUS PLAN WAS REJECTED BY A HUMAN REVIEWER.\n"
                f"Rejection reason: {rejection_notes}\n"
                f"You MUST address these concerns in your revised plan.\n"
            )

        prompt = f"""Memory context from past similar tasks:
{memory_context}
{rejection_block}
User request:
{user_request}

Produce an ExecutionPlan JSON for task_id="{task_id}". Output ONLY valid JSON."""

        messages = [SystemMessage(content=system), HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)

        raw = response.content.strip()
        # Strip markdown fences if the model adds them
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()

        plan_dict = json.loads(raw)
        plan_dict["task_id"] = task_id
        plan_dict["original_request"] = user_request
        plan = ExecutionPlan(**plan_dict)

        # Upgrade to human approval if below threshold
        if plan.confidence < self._threshold:
            plan.requires_human_approval = True
            logger.info("task=%s confidence=%.2f → flagged for HITL", task_id, plan.confidence)

        usage = getattr(response, "usage_metadata", {}) or {}
        tokens = usage.get("total_tokens", 0)

        return AgentResult(
            output=plan.model_dump(),
            confidence=plan.confidence,
            tokens_used=tokens,
        )
