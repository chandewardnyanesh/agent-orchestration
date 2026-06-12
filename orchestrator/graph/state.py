"""
LangGraph shared state schema.
Every node reads from and writes to this TypedDict.
"""
from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict


TaskStatus = Literal[
    "pending",
    "planning",
    "awaiting_human_approval",
    "executing",
    "reviewing",
    "retrying",
    "synthesizing",
    "completed",
    "failed",
]


class SubTaskState(TypedDict):
    id: str
    description: str
    specialist: str
    depends_on: list[str]
    expected_output_format: str
    status: Literal["pending", "in_progress", "completed", "failed"]
    output: Optional[str]
    review_approved: Optional[bool]
    review_feedback: Optional[str]
    attempt: int


class OrchestratorState(TypedDict):
    # Core
    task_id: str
    user_request: str
    status: TaskStatus
    error: Optional[str]

    # Planning
    execution_plan: Optional[dict]        # serialised ExecutionPlan
    supervisor_confidence: float

    # Execution
    subtasks: list[SubTaskState]
    completed_outputs: dict[str, str]     # subtask_id → output text

    # Memory
    memory_context: str                   # retrieved from ChromaDB
    memory_written: bool

    # Human-in-the-loop
    hitl_required: bool
    hitl_level: Optional[Literal["notify", "approve_action", "approve_plan", "take_over"]]
    hitl_decision: Optional[Literal["approved", "rejected", "modified", "take_over"]]
    hitl_notes: Optional[str]

    # Final
    final_output: Optional[str]

    # Observability
    trace_id: str
    total_tokens: int
    total_cost_usd: float
