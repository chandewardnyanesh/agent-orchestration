"""
In-process Snapshot & Span Store
----------------------------------
Phase 4 — stores execution snapshots and agent spans per task_id in memory.

Why in-process?
  We don't want to depend on another Docker container (Jaeger, Zipkin, etc.)
  just to see what the system did.  A plain dict works perfectly for a
  single-process server and lets the trace explorer UI work out of the box.

Phase 6 upgrade path:
  Replace the dicts with SQLAlchemy writes to PostgreSQL.
  The store functions are the only place that needs to change.

Data model:
  Snapshot — full state slice saved at the END of every LangGraph node.
  Span     — timing + token record saved for every agent.timed_invoke() call.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

# ── In-memory stores ──────────────────────────────────────────────────────────
# Both are keyed by task_id and hold an ordered list of records.
_snapshots: Dict[str, List[Dict[str, Any]]] = {}
_spans:     Dict[str, List[Dict[str, Any]]] = {}


# ── Snapshot API ──────────────────────────────────────────────────────────────

def save_snapshot(task_id: str, node_name: str, state: dict) -> None:
    """
    Persist a lightweight state slice after a graph node completes.
    Only the observable fields are stored (not the full state blob)
    to keep memory usage bounded.
    """
    if not task_id:
        return
    if task_id not in _snapshots:
        _snapshots[task_id] = []

    _snapshots[task_id].append({
        "step":        len(_snapshots[task_id]),
        "node":        node_name,
        "recorded_at": time.time(),
        "state": {
            "status":                state.get("status"),
            "supervisor_confidence": state.get("supervisor_confidence", 0.0),
            "total_tokens":          state.get("total_tokens", 0),
            "total_cost_usd":        state.get("total_cost_usd", 0.0),
            "hitl_decision":         state.get("hitl_decision"),
            "hitl_level":            state.get("hitl_level"),
            "memory_written":        state.get("memory_written", False),
            "final_output_preview":  (state.get("final_output") or "")[:200],
            "subtasks": [
                {
                    "id":              st.get("id"),
                    "specialist":      st.get("specialist"),
                    "status":          st.get("status"),
                    "review_approved": st.get("review_approved"),
                    "attempt":         st.get("attempt", 0),
                }
                for st in state.get("subtasks", [])
            ],
        }
    })


def get_snapshots(task_id: str) -> List[Dict[str, Any]]:
    """Return all node snapshots for a task, ordered by step number."""
    return _snapshots.get(task_id, [])


def clear_snapshots(task_id: str) -> None:
    _snapshots.pop(task_id, None)


# ── Span API ──────────────────────────────────────────────────────────────────

def save_span(task_id: str, span: Dict[str, Any]) -> None:
    """
    Record an agent execution span (timing, tokens, model, confidence).
    Called automatically by BaseAgent.timed_invoke().
    """
    if not task_id:
        return
    if task_id not in _spans:
        _spans[task_id] = []
    _spans[task_id].append({**span, "recorded_at": time.time()})


def get_spans(task_id: str) -> List[Dict[str, Any]]:
    """Return all agent spans for a task in recording order."""
    return _spans.get(task_id, [])


def clear_spans(task_id: str) -> None:
    _spans.pop(task_id, None)


# ── Summary helpers ───────────────────────────────────────────────────────────

def task_summary(task_id: str) -> Dict[str, Any]:
    """
    Aggregate spans and snapshots into a single summary dict.
    Used by the trace explorer UI and /traces/{task_id}/summary endpoint.
    """
    spans     = get_spans(task_id)
    snapshots = get_snapshots(task_id)

    total_tokens   = sum(s.get("tokens_used", 0) for s in spans)
    total_latency  = sum(s.get("latency_ms", 0)  for s in spans)
    agent_calls    = len(spans)

    # Cost per agent from spans
    from orchestrator.observability.cost_tracker import tokens_to_usd
    breakdown = [
        {
            "agent":      s.get("agent"),
            "model":      s.get("model", "unknown"),
            "tokens":     s.get("tokens_used", 0),
            "latency_ms": round(s.get("latency_ms", 0), 1),
            "confidence": round(s.get("confidence", 0.0), 3),
            "usd":        tokens_to_usd(s.get("tokens_used", 0), s.get("model", "")),
        }
        for s in spans
    ]
    total_usd = sum(b["usd"] for b in breakdown)

    return {
        "task_id":        task_id,
        "agent_calls":    agent_calls,
        "total_tokens":   total_tokens,
        "total_usd":      round(total_usd, 6),
        "total_latency_ms": round(total_latency, 1),
        "breakdown":      breakdown,
        "node_count":     len(snapshots),
        "nodes_executed": [s["node"] for s in snapshots],
    }
