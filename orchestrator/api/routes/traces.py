"""
Observability and trace endpoints.

GET  /traces/{task_id}/summary     — tokens, USD, latency, agent breakdown
GET  /traces/{task_id}/cost        — cost breakdown (Redis + in-process)
GET  /traces/{task_id}/spans       — per-agent span list (latency, tokens, confidence)
GET  /traces/{task_id}/snapshots   — node-by-node state snapshots
GET  /traces/{task_id}/replay/{step} — single snapshot step for step-through replay
POST /traces/{task_id}/replay      — start a replay session (returns all steps)
"""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/{task_id}/summary")
def get_summary(task_id: str):
    """Aggregated observability summary — tokens, cost, latency, agent calls."""
    from orchestrator.observability.snapshot_store import task_summary
    return task_summary(task_id)


@router.get("/{task_id}/cost")
def get_task_cost(task_id: str):
    """Detailed cost breakdown from CostTracker (Redis + in-process)."""
    from orchestrator.observability.cost_tracker import CostTracker
    return CostTracker().total_for_task(task_id)


@router.get("/{task_id}/spans")
def get_spans(task_id: str):
    """Return all agent execution spans recorded for this task."""
    from orchestrator.observability.snapshot_store import get_spans
    spans = get_spans(task_id)
    return {
        "task_id":     task_id,
        "span_count":  len(spans),
        "spans":       spans,
    }


@router.get("/{task_id}/snapshots")
def get_snapshots(task_id: str):
    """Return all node snapshots in execution order."""
    from orchestrator.observability.snapshot_store import get_snapshots
    snapshots = get_snapshots(task_id)
    return {
        "task_id":        task_id,
        "snapshot_count": len(snapshots),
        "snapshots":      snapshots,
    }


@router.get("/{task_id}/replay/{step}")
def get_replay_step(task_id: str, step: int):
    """Return a single snapshot step for incremental step-through replay."""
    from orchestrator.observability.replay import ReplaySession
    session = ReplaySession(task_id)
    return session.jump_to(step)


@router.post("/{task_id}/replay")
def start_replay(task_id: str):
    """Return the full step list for a replay session."""
    from orchestrator.observability.replay import ReplaySession
    session = ReplaySession(task_id)
    return {
        "task_id":    task_id,
        "step_count": len(session),
        "steps":      session.all_steps(),
    }


@router.get("/{task_id}/diff")
def diff_steps(task_id: str, step_a: int = 0, step_b: int = 1):
    """Diff two snapshot steps — shows what changed between graph nodes."""
    from orchestrator.observability.replay import ReplaySession
    session = ReplaySession(task_id)
    return {
        "task_id": task_id,
        "step_a":  step_a,
        "step_b":  step_b,
        "diff":    session.compare(step_a, step_b),
    }
