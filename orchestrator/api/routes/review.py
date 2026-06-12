"""
Human-in-the-Loop review endpoints.

Phase 3:
- POST /review/{task_id}/decide now RESUMES the suspended LangGraph graph
  by calling compiled_graph.invoke(Command(resume=...), config=...).
- The graph continues from inside await_human() with the human's decision.
- Resume runs in a FastAPI BackgroundTask so the HTTP response is immediate.
"""
from __future__ import annotations

import logging
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)

# Module-level import so tests can patch orchestrator.api.routes.review.compiled_graph
from orchestrator.graph.workflow import compiled_graph  # noqa: E402


class ReviewDecision(BaseModel):
    decision: str   # approved | rejected | modified | take_over
    notes: str = ""


def _resume_graph(task_id: str, decision: str, notes: str) -> None:
    """
    Resume a suspended graph checkpoint.

    LangGraph stores the full graph state in the MemorySaver checkpointer
    under the key `thread_id == task_id`.  Calling invoke() with
    Command(resume=payload) and the same thread config picks up exactly
    where interrupt() paused — inside await_human() — and continues.
    """
    from langgraph.types import Command
    from langgraph.errors import GraphInterrupt
    from orchestrator.api.routes.tasks import update_task, _finish_task, _thread_config

    try:
        final = compiled_graph.invoke(
            Command(resume={"decision": decision, "notes": notes}),
            config=_thread_config(task_id),
        )
        # LangGraph 0.6+ signals another pause via __interrupt__ key in state
        if "__interrupt__" in final:
            logger.info("task=%s re-suspended after resume", task_id)
            update_task(task_id, {"status": "awaiting_human_approval", "error": None})
        else:
            _finish_task(task_id, final)
            logger.info("task=%s resumed and completed status=%s", task_id, final.get("status"))

    except GraphInterrupt:
        # Fallback: stream() mode raises GraphInterrupt — keep for compatibility.
        logger.info("task=%s re-suspended (stream mode) after resume", task_id)
        update_task(task_id, {"status": "awaiting_human_approval", "error": None})

    except Exception as exc:
        logger.exception("task=%s resume failed: %s", task_id, exc)
        update_task(task_id, {"status": "failed", "error": str(exc)})


@router.get("/pending")
def list_pending():
    """List all tasks currently awaiting human review."""
    from orchestrator.hitl.queue import HITLQueue
    return HITLQueue().list_pending()


@router.post("/{task_id}/decide")
def post_decision(task_id: str, body: ReviewDecision, background: BackgroundTasks):
    """
    Record a human decision and resume the suspended graph.

    The HTTP response returns immediately with {'recorded': decision}.
    Graph execution continues in the background thread.
    """
    from orchestrator.hitl.queue import HITLQueue

    # Store decision in Redis (for audit trail / UI polling)
    HITLQueue().post_decision(task_id, body.decision, body.notes)

    # Resume the LangGraph execution in a background thread
    background.add_task(_resume_graph, task_id, body.decision, body.notes)

    return {
        "task_id":   task_id,
        "recorded":  body.decision,
        "resuming":  True,
        "message":   "Graph resuming. Poll GET /tasks/{task_id} for status.",
    }
