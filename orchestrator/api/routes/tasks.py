"""
Task submission and status endpoints.

Phase 3 additions:
- _run_task passes config={"configurable": {"thread_id": task_id}} so
  the MemorySaver checkpointer can save/restore graph state per task.
- GraphInterrupt is caught and marks the task 'awaiting_human_approval'
  rather than 'failed' — the graph is merely suspended, not broken.
- _finish_task() is shared by both _run_task and the review resume path
  so the task store is always updated the same way.

Phase 6 additions:
- Task store backed by PostgreSQL via orchestrator.db.task_store (with
  in-memory dict fallback when the DB is unavailable).
- GET /tasks returns paginated task list.
"""
from __future__ import annotations

import uuid
import logging
from fastapi import APIRouter, BackgroundTasks, Query
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)

# Imported at module level so tests can patch orchestrator.api.routes.tasks.compiled_graph
from orchestrator.graph.workflow import compiled_graph  # noqa: E402

# ── Task store (DB-backed with in-memory fallback) ────────────────────────────
from orchestrator.db.task_store import (
    create_task as _db_create,
    get_task as _db_get,
    update_task as _db_update,
    finish_task as _db_finish,
    list_tasks as _db_list,
)


def _thread_config(task_id: str) -> dict:
    return {"configurable": {"thread_id": task_id}}


# Keep a backward-compat shim so review.py can still call update_task/
# _finish_task/_thread_config without importing from db.task_store directly.
def update_task(task_id: str, patch: dict) -> None:
    _db_update(task_id, patch)


def _finish_task(task_id: str, final: dict) -> None:
    _db_finish(task_id, final)


# ─────────────────────────────────────────────────────────────────────────────

class TaskRequest(BaseModel):
    user_request: str
    require_human_review: bool = False


class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str


def _run_task(task_id: str, user_request: str, require_human_review: bool) -> None:
    """Execute the orchestration graph in a background thread."""
    from langgraph.errors import GraphInterrupt

    initial_state = {
        "task_id":          task_id,
        "user_request":     user_request,
        "status":           "pending",
        "error":            None,
        "execution_plan":   None,
        "supervisor_confidence": 0.0,
        "subtasks":         [],
        "completed_outputs": {},
        "memory_context":   "",
        "memory_written":   False,
        "hitl_required":    require_human_review,
        "hitl_level":       "approve_plan" if require_human_review else None,
        "hitl_decision":    None,
        "hitl_notes":       None,
        "final_output":     None,
        "trace_id":         task_id,
        "total_tokens":     0,
        "total_cost_usd":   0.0,
    }
    _db_update(task_id, {"status": "pending", "user_request": user_request})

    try:
        final = compiled_graph.invoke(
            initial_state,
            config=_thread_config(task_id),
        )
        if "__interrupt__" in final:
            logger.info("task=%s suspended — awaiting human approval", task_id)
            _db_update(task_id, {"status": "awaiting_human_approval", "error": None})
        else:
            _db_finish(task_id, final)

    except GraphInterrupt:
        logger.info("task=%s suspended (stream mode) — awaiting human approval", task_id)
        _db_update(task_id, {"status": "awaiting_human_approval", "error": None})

    except Exception as exc:
        logger.exception("task=%s failed: %s", task_id, exc)
        _db_update(task_id, {"status": "failed", "error": str(exc)})


@router.post("/", response_model=TaskResponse)
def submit_task(req: TaskRequest, background: BackgroundTasks):
    task_id = str(uuid.uuid4())
    _db_create(task_id, req.user_request, status="pending")
    background.add_task(_run_task, task_id, req.user_request, req.require_human_review)
    return TaskResponse(task_id=task_id, status="pending", message="Task queued.")


@router.get("/")
def list_tasks(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """Return a paginated list of tasks, newest first."""
    tasks = _db_list(limit=limit, offset=offset)
    return {"tasks": tasks, "count": len(tasks), "limit": limit, "offset": offset}


@router.get("/{task_id}")
def get_task_status(task_id: str):
    """Return the current state snapshot for a task."""
    task = _db_get(task_id)
    if not task:
        return {
            "task_id": task_id,
            "status":  "not_found",
            "message": "Task not found. It may have been submitted before server restart.",
        }
    return task
