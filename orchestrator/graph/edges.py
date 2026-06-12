"""
Conditional edge functions for the LangGraph workflow.
Each function receives state and returns the name of the next node.

Phase 3 additions:
- after_plan:       auto-escalate when supervisor confidence < threshold
- after_await_human: 'rejected' → re-plan (loops back to plan node with notes)
                     'take_over' → END
                     'approved' / 'modified' → execute_subtasks
"""
from __future__ import annotations

from orchestrator.graph.state import OrchestratorState


def after_plan(state: OrchestratorState) -> str:
    """
    Route to await_human when:
      (a) the supervisor explicitly flagged HITL, OR
      (b) confidence is below the configured threshold (auto-escalation).
    Otherwise proceed directly to execution.
    """
    from orchestrator.config import get_settings
    cfg = get_settings()

    if state.get("hitl_required"):
        return "await_human"

    # Auto-escalate on low confidence even if user didn't request review
    confidence = state.get("supervisor_confidence", 1.0)
    if confidence < cfg.supervisor_confidence_threshold:
        return "await_human"

    return "execute_subtasks"


def after_await_human(state: OrchestratorState) -> str:
    """
    Route based on human decision:
      - rejected  → send back to plan so the Supervisor can re-plan with
                    the rejection notes injected into its prompt
      - take_over → END (human takes full control)
      - approved / modified → execute_subtasks
    """
    decision = state.get("hitl_decision")
    if decision == "take_over":
        return "END"
    if decision == "rejected":
        return "plan"   # re-plan with hitl_notes in context
    # approved or modified → proceed with current (possibly edited) plan
    return "execute_subtasks"


def after_review(state: OrchestratorState) -> str:
    """Route back to execute if subtasks need retrying, else synthesize."""
    status = state.get("status")
    if status == "retrying":
        has_pending = any(
            st["status"] == "pending"
            for st in state.get("subtasks", [])
        )
        if has_pending:
            return "execute_subtasks"
        # All retries exhausted → escalate to human
        return "await_human"
    return "synthesize"
