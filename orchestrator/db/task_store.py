"""
Persistent task store backed by PostgreSQL, with an in-memory dict fallback.

All public functions work identically whether Postgres is up or not.
Callers do not need to handle the distinction.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── In-memory fallback ────────────────────────────────────────────────────────
_mem: dict[str, dict] = {}


# ── Public API ────────────────────────────────────────────────────────────────

def create_task(task_id: str, user_request: str, status: str = "pending") -> dict:
    """Insert a new task row (or initialise the in-memory entry)."""
    record = {
        "task_id":               task_id,
        "status":                status,
        "user_request":          user_request,
        "supervisor_confidence": 0.0,
        "total_tokens":          0,
        "total_cost_usd":        0.0,
        "final_output":          None,
        "subtasks":              [],
        "hitl_required":         False,
        "hitl_decision":         None,
        "hitl_notes":            None,
        "memory_written":        False,
        "error":                 None,
    }

    from orchestrator.db.engine import is_pg_available, get_session
    if is_pg_available():
        try:
            from orchestrator.db.models import TaskRecord
            with get_session() as sess:
                row = TaskRecord(task_id=task_id, user_request=user_request,
                                 status=status)
                sess.add(row)
            return record
        except Exception as exc:
            logger.warning("DB create_task failed (%s) — using memory", exc)

    _mem[task_id] = record
    return record


def get_task(task_id: str) -> Optional[dict]:
    """Return the task dict or None if not found."""
    from orchestrator.db.engine import is_pg_available, get_session
    if is_pg_available():
        try:
            from orchestrator.db.models import TaskRecord
            with get_session() as sess:
                row = sess.get(TaskRecord, task_id)
                if row:
                    return row.to_dict()
                return None
        except Exception as exc:
            logger.warning("DB get_task failed (%s) — using memory", exc)

    return _mem.get(task_id)


def update_task(task_id: str, patch: dict) -> None:
    """Merge a partial state update into the persisted task."""
    from orchestrator.db.engine import is_pg_available, get_session
    if is_pg_available():
        try:
            from orchestrator.db.models import TaskRecord
            with get_session() as sess:
                row = sess.get(TaskRecord, task_id)
                if row is None:
                    row = TaskRecord(task_id=task_id,
                                     user_request=patch.get("user_request", ""))
                    sess.add(row)
                for k, v in patch.items():
                    if hasattr(row, k):
                        setattr(row, k, v)
            return
        except Exception as exc:
            logger.warning("DB update_task failed (%s) — using memory", exc)

    if task_id not in _mem:
        _mem[task_id] = {"task_id": task_id}
    _mem[task_id].update(patch)


def finish_task(task_id: str, final_state: dict) -> None:
    """Write the completed graph final state into the task store."""
    subtasks = [
        {
            "id":              st["id"],
            "description":     st["description"],
            "specialist":      st["specialist"],
            "status":          st["status"],
            "review_approved": st.get("review_approved"),
            "attempt":         st.get("attempt", 0),
        }
        for st in final_state.get("subtasks", [])
    ]
    patch = {
        "status":                final_state.get("status", "completed"),
        "supervisor_confidence": final_state.get("supervisor_confidence", 0.0),
        "total_tokens":          final_state.get("total_tokens", 0),
        "total_cost_usd":        final_state.get("total_cost_usd", 0.0),
        "final_output":          final_state.get("final_output"),
        "subtasks":              subtasks,
        "hitl_required":         final_state.get("hitl_required", False),
        "hitl_decision":         final_state.get("hitl_decision"),
        "hitl_notes":            final_state.get("hitl_notes"),
        "memory_written":        final_state.get("memory_written", False),
        "error":                 final_state.get("error"),
    }
    update_task(task_id, patch)


def list_tasks(limit: int = 50, offset: int = 0) -> list[dict]:
    """Return a paginated list of tasks, newest first."""
    from orchestrator.db.engine import is_pg_available, get_session
    if is_pg_available():
        try:
            from orchestrator.db.models import TaskRecord
            from sqlalchemy import desc
            with get_session() as sess:
                rows = (
                    sess.query(TaskRecord)
                    .order_by(desc(TaskRecord.created_at))
                    .offset(offset)
                    .limit(limit)
                    .all()
                )
                return [r.to_dict() for r in rows]
        except Exception as exc:
            logger.warning("DB list_tasks failed (%s) — using memory", exc)

    items = sorted(_mem.values(),
                   key=lambda t: t.get("task_id", ""), reverse=True)
    return items[offset: offset + limit]
