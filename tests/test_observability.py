"""
Phase 4 — Observability Tests

Tests:
  1.  tokens_to_usd() math per model
  2.  tokens_to_usd() uses fallback rate for unknown model
  3.  CostTracker.record() stores in-process when Redis is down
  4.  CostTracker.total_for_task() aggregates in-process entries
  5.  CostTracker.check_alert() fires when cost exceeds threshold
  6.  CostTracker.check_alert() stays silent below threshold
  7.  save_snapshot() stores and get_snapshots() retrieves
  8.  save_span() stores and get_spans() retrieves
  9.  task_summary() aggregates spans into totals
  10. ReplaySession loads snapshots via _load_snapshots
  11. step_forward() returns snapshots in order
  12. jump_to() returns a specific step without advancing cursor
  13. compare() diffs two steps correctly
  14. Nodes save snapshots (_snap helper)
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


# ═══════════════════════════════════════════════════════════════════════════════
# 1–2. tokens_to_usd()
# ═══════════════════════════════════════════════════════════════════════════════

def test_tokens_to_usd_known_model():
    from orchestrator.observability.cost_tracker import tokens_to_usd
    # 1M tokens of claude-haiku at $0.25 = $0.25
    assert abs(tokens_to_usd(1_000_000, "claude-haiku-4-5-20251001") - 0.25) < 0.001


def test_tokens_to_usd_sonnet():
    from orchestrator.observability.cost_tracker import tokens_to_usd
    # 500k tokens of sonnet at $3 / 1M = $1.50
    assert abs(tokens_to_usd(500_000, "claude-sonnet-4-6") - 1.50) < 0.001


def test_tokens_to_usd_unknown_model_uses_fallback():
    from orchestrator.observability.cost_tracker import tokens_to_usd
    # unknown model falls back to $5 / 1M
    assert abs(tokens_to_usd(1_000_000, "unknown-model-x") - 5.0) < 0.001


def test_tokens_to_usd_zero_tokens():
    from orchestrator.observability.cost_tracker import tokens_to_usd
    assert tokens_to_usd(0, "claude-opus-4-6") == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 3–4. CostTracker with Redis fallback
# ═══════════════════════════════════════════════════════════════════════════════

def _make_tracker_no_redis():
    """Build a CostTracker with Redis disabled."""
    from orchestrator.observability.cost_tracker import CostTracker, _local_store
    tracker = CostTracker.__new__(CostTracker)
    from orchestrator.config import get_settings
    tracker._cfg   = get_settings()
    tracker._redis = None
    return tracker, _local_store


def test_cost_tracker_records_in_process_when_redis_down():
    from orchestrator.observability.cost_tracker import _local_store
    tracker, store = _make_tracker_no_redis()
    task_id = "test-cost-001"
    store.pop(task_id, None)   # clean slate

    cost = tracker.record(task_id, "research", "claude-haiku-4-5-20251001", 1_000)
    assert cost > 0
    assert task_id in store
    assert len(store[task_id]) == 1
    store.pop(task_id, None)   # cleanup


def test_cost_tracker_total_aggregates_in_process():
    from orchestrator.observability.cost_tracker import _local_store
    tracker, store = _make_tracker_no_redis()
    task_id = "test-cost-002"
    store.pop(task_id, None)

    tracker.record(task_id, "research",  "claude-haiku-4-5-20251001", 10_000)
    tracker.record(task_id, "writing",   "claude-haiku-4-5-20251001", 10_000)

    totals = tracker.total_for_task(task_id)
    assert totals["total_tokens"] == 20_000
    assert totals["total_usd"] > 0
    assert len(totals["breakdown"]) == 2
    store.pop(task_id, None)


# ═══════════════════════════════════════════════════════════════════════════════
# 5–6. Cost alert
# ═══════════════════════════════════════════════════════════════════════════════

def test_cost_alert_fires_above_threshold():
    from orchestrator.observability.cost_tracker import CostTracker, _local_store
    tracker, store = _make_tracker_no_redis()
    task_id = "test-alert-001"
    store.pop(task_id, None)

    # Inject a very expensive fake entry directly
    store[task_id] = [{"agent": "x", "model": "x", "tokens": 0, "usd": 999.0}]
    fired = tracker.check_alert(task_id)
    assert fired is True
    store.pop(task_id, None)


def test_cost_alert_silent_below_threshold():
    from orchestrator.observability.cost_tracker import _local_store
    tracker, store = _make_tracker_no_redis()
    task_id = "test-alert-002"
    store.pop(task_id, None)

    store[task_id] = [{"agent": "x", "model": "x", "tokens": 0, "usd": 0.001}]
    fired = tracker.check_alert(task_id)
    assert fired is False
    store.pop(task_id, None)


# ═══════════════════════════════════════════════════════════════════════════════
# 7–8. snapshot_store
# ═══════════════════════════════════════════════════════════════════════════════

def test_save_and_get_snapshots():
    from orchestrator.observability.snapshot_store import (
        save_snapshot, get_snapshots, clear_snapshots
    )
    task_id = "snap-test-001"
    clear_snapshots(task_id)

    state = {
        "status": "executing",
        "supervisor_confidence": 0.9,
        "total_tokens": 500,
        "subtasks": [],
    }
    save_snapshot(task_id, "plan", state)
    save_snapshot(task_id, "execute_subtasks", {**state, "status": "reviewing"})

    snaps = get_snapshots(task_id)
    assert len(snaps) == 2
    assert snaps[0]["node"] == "plan"
    assert snaps[1]["node"] == "execute_subtasks"
    assert snaps[0]["step"] == 0
    assert snaps[1]["step"] == 1
    clear_snapshots(task_id)


def test_save_and_get_spans():
    from orchestrator.observability.snapshot_store import save_span, get_spans, clear_spans
    task_id = "span-test-001"
    clear_spans(task_id)

    save_span(task_id, {"agent": "supervisor", "model": "claude-opus-4-6",
                         "latency_ms": 123.4, "tokens_used": 800, "confidence": 0.85})
    save_span(task_id, {"agent": "research",   "model": "claude-sonnet-4-6",
                         "latency_ms": 456.7, "tokens_used": 1200, "confidence": 0.90})

    spans = get_spans(task_id)
    assert len(spans) == 2
    assert spans[0]["agent"] == "supervisor"
    assert spans[1]["agent"] == "research"
    clear_spans(task_id)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. task_summary()
# ═══════════════════════════════════════════════════════════════════════════════

def test_task_summary_aggregates_spans():
    from orchestrator.observability.snapshot_store import (
        save_span, clear_spans, task_summary
    )
    task_id = "summary-test-001"
    clear_spans(task_id)

    save_span(task_id, {"agent": "supervisor", "model": "claude-haiku-4-5-20251001",
                         "latency_ms": 100, "tokens_used": 1_000_000, "confidence": 0.9})
    save_span(task_id, {"agent": "research",   "model": "claude-haiku-4-5-20251001",
                         "latency_ms": 200, "tokens_used": 1_000_000, "confidence": 0.8})

    summary = task_summary(task_id)
    assert summary["total_tokens"]    == 2_000_000
    assert summary["agent_calls"]     == 2
    assert summary["total_latency_ms"] == 300
    assert abs(summary["total_usd"] - 0.50) < 0.01   # 2M × $0.25
    clear_spans(task_id)


# ═══════════════════════════════════════════════════════════════════════════════
# 10–13. ReplaySession
# ═══════════════════════════════════════════════════════════════════════════════

def _make_session_with_snaps(task_id: str, n: int):
    """Pre-populate snapshot_store and return a ReplaySession."""
    from orchestrator.observability.snapshot_store import (
        save_snapshot, clear_snapshots
    )
    from orchestrator.observability.replay import ReplaySession
    clear_snapshots(task_id)
    nodes = ["retrieve_memory", "plan", "execute_subtasks", "review", "synthesize"]
    for i in range(n):
        save_snapshot(task_id, nodes[i % len(nodes)], {
            "status": "executing", "total_tokens": i * 100,
            "supervisor_confidence": 0.8, "subtasks": [],
        })
    return ReplaySession(task_id)


def test_replay_session_loads_snapshots():
    session = _make_session_with_snaps("replay-001", 3)
    assert len(session) == 3


def test_step_forward_returns_in_order():
    session = _make_session_with_snaps("replay-002", 3)
    nodes = []
    while True:
        step = session.step_forward()
        if step.get("status") == "replay_complete":
            break
        nodes.append(step["node"])
    assert nodes == ["retrieve_memory", "plan", "execute_subtasks"]


def test_jump_to_does_not_advance_cursor():
    session = _make_session_with_snaps("replay-003", 3)
    snap = session.jump_to(2)
    assert snap["step"] == 2
    # cursor should still be at 0
    first = session.step_forward()
    assert first["step"] == 0


def test_compare_diffs_correctly():
    from orchestrator.observability.snapshot_store import (
        save_snapshot, clear_snapshots
    )
    from orchestrator.observability.replay import ReplaySession
    task_id = "replay-004"
    clear_snapshots(task_id)
    save_snapshot(task_id, "plan",            {"status": "executing", "total_tokens": 100,
                                               "supervisor_confidence": 0.7, "subtasks": []})
    save_snapshot(task_id, "execute_subtasks", {"status": "reviewing", "total_tokens": 500,
                                                "supervisor_confidence": 0.7, "subtasks": []})
    session = ReplaySession(task_id)
    diff = session.compare(0, 1)
    assert "status"       in diff
    assert "total_tokens" in diff
    assert diff["status"]["step_a"] == "executing"
    assert diff["status"]["step_b"] == "reviewing"


# ═══════════════════════════════════════════════════════════════════════════════
# 14. _snap() helper in nodes.py records snapshots
# ═══════════════════════════════════════════════════════════════════════════════

def test_snap_helper_returns_result_unchanged():
    from orchestrator.graph.nodes import _snap
    from orchestrator.observability.snapshot_store import clear_snapshots
    task_id = "snap-helper-001"
    clear_snapshots(task_id)

    state  = {"task_id": task_id, "user_request": "test", "status": "planning",
               "subtasks": [], "supervisor_confidence": 0.0, "total_tokens": 0}
    result = {"status": "executing", "total_tokens": 100}

    returned = _snap("plan", state, result)
    assert returned is result   # same object, unchanged


def test_snap_helper_saves_snapshot():
    from orchestrator.graph.nodes import _snap
    from orchestrator.observability.snapshot_store import get_snapshots, clear_snapshots
    task_id = "snap-helper-002"
    clear_snapshots(task_id)

    state  = {"task_id": task_id, "user_request": "test", "status": "planning",
               "subtasks": [], "supervisor_confidence": 0.0, "total_tokens": 0}
    result = {"status": "executing"}

    _snap("plan", state, result)
    snaps = get_snapshots(task_id)
    assert len(snaps) == 1
    assert snaps[0]["node"] == "plan"
    clear_snapshots(task_id)
