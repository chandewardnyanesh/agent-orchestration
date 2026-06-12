"""
Phase 3 — HITL Tests

Tests:
  1.  check_escalation() — all five trigger rules
  2.  after_plan edge auto-escalates on low confidence
  3.  after_plan edge passes through on high confidence
  4.  await_human injects interrupt() and returns hitl_decision
  5.  after_await_human routes 'rejected' → plan (re-plan)
  6.  after_await_human routes 'take_over' → END
  7.  after_await_human routes 'approved' → execute_subtasks
  8.  after_await_human routes 'modified' → execute_subtasks
  9.  _run_task catches GraphInterrupt → 'awaiting_human_approval'
  10. _resume_graph completes the graph after approval
  11. Rejection notes are threaded into Supervisor re-plan prompt
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch


# ═══════════════════════════════════════════════════════════════════════════════
# 1–4. check_escalation() rules
# ═══════════════════════════════════════════════════════════════════════════════

def test_escalation_user_requested():
    from orchestrator.hitl.escalation import check_escalation
    should, level = check_escalation(confidence=0.95, user_requested=True)
    assert should is True
    assert level == "approve_action"


def test_escalation_sensitive_operation():
    from orchestrator.hitl.escalation import check_escalation
    should, level = check_escalation(confidence=0.95, operation="send_email")
    assert should is True
    assert level == "approve_action"


def test_escalation_retries_exhausted():
    from orchestrator.hitl.escalation import check_escalation
    from orchestrator.config import get_settings
    max_r = get_settings().max_specialist_retries
    should, level = check_escalation(confidence=0.95, retry_count=max_r)
    assert should is True
    assert level == "take_over"


def test_escalation_low_quality():
    from orchestrator.hitl.escalation import check_escalation
    from orchestrator.config import get_settings
    threshold = get_settings().reviewer_quality_threshold
    should, level = check_escalation(confidence=0.95, quality_score=threshold - 0.01)
    assert should is True
    assert level == "approve_action"


def test_escalation_low_confidence():
    from orchestrator.hitl.escalation import check_escalation
    from orchestrator.config import get_settings
    threshold = get_settings().supervisor_confidence_threshold
    should, level = check_escalation(confidence=threshold - 0.01)
    assert should is True
    assert level == "approve_plan"


def test_no_escalation_normal():
    from orchestrator.hitl.escalation import check_escalation
    should, level = check_escalation(confidence=0.95, quality_score=1.0)
    assert should is False
    assert level is None


# ═══════════════════════════════════════════════════════════════════════════════
# 5. after_plan auto-escalates on low confidence
# ═══════════════════════════════════════════════════════════════════════════════

def test_after_plan_auto_escalates_low_confidence():
    from orchestrator.graph.edges import after_plan
    from orchestrator.config import get_settings
    threshold = get_settings().supervisor_confidence_threshold
    state = {
        "hitl_required": False,
        "supervisor_confidence": threshold - 0.01,
    }
    assert after_plan(state) == "await_human"


def test_after_plan_skips_hitl_on_high_confidence():
    from orchestrator.graph.edges import after_plan
    state = {
        "hitl_required": False,
        "supervisor_confidence": 0.99,
    }
    assert after_plan(state) == "execute_subtasks"


def test_after_plan_respects_explicit_hitl_required():
    from orchestrator.graph.edges import after_plan
    state = {
        "hitl_required": True,
        "supervisor_confidence": 0.99,  # even high confidence escalates if flagged
    }
    assert after_plan(state) == "await_human"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. after_await_human routing for all four decisions
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("decision,expected", [
    ("approved",  "execute_subtasks"),
    ("modified",  "execute_subtasks"),
    ("rejected",  "plan"),
    ("take_over", "END"),
])
def test_after_await_human_routing(decision, expected):
    from orchestrator.graph.edges import after_await_human
    state = {"hitl_decision": decision}
    assert after_await_human(state) == expected


# ═══════════════════════════════════════════════════════════════════════════════
# 7. await_human node calls interrupt() and maps decision to status
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("decision,expected_status", [
    ("approved",  "executing"),
    ("modified",  "executing"),
    ("rejected",  "failed"),
    ("take_over", "failed"),
])
def test_await_human_node_maps_decision_to_status(decision, expected_status):
    from orchestrator.graph.nodes import await_human

    state = {
        "task_id":              "t-test",
        "user_request":         "test",
        "execution_plan":       {},
        "hitl_level":           "approve_plan",
        "supervisor_confidence": 0.4,
    }

    # HITLQueue is imported inside await_human → patch it at its source module.
    # langgraph.types.interrupt is the real function used at runtime.
    with patch("orchestrator.hitl.queue.HITLQueue") as mock_queue_cls, \
         patch("langgraph.types.interrupt", return_value={"decision": decision, "notes": ""}):
        mock_queue_cls.return_value.push = MagicMock()
        result = await_human(state)

    assert result["hitl_decision"] == decision
    assert result["status"]        == expected_status


# ═══════════════════════════════════════════════════════════════════════════════
# 8. _run_task catches GraphInterrupt → 'awaiting_human_approval'
# ═══════════════════════════════════════════════════════════════════════════════

def test_run_task_catches_graph_interrupt():
    from langgraph.errors import GraphInterrupt
    from orchestrator.api.routes.tasks import _run_task
    from orchestrator.db import task_store as ts

    task_id = "interrupt-test-001"
    ts._mem.pop(task_id, None)

    with patch("orchestrator.api.routes.tasks.compiled_graph") as mock_graph, \
         patch("orchestrator.db.engine._pg_available", False):
        mock_graph.invoke.side_effect = GraphInterrupt(())
        _run_task(task_id, "test request", False)

    record = ts.get_task(task_id)
    assert record is not None
    assert record["status"] == "awaiting_human_approval"
    assert record["error"] is None
    ts._mem.pop(task_id, None)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. _resume_graph calls compiled_graph with Command(resume=...)
# ═══════════════════════════════════════════════════════════════════════════════

def test_resume_graph_calls_compiled_graph_with_command():
    from orchestrator.api.routes.review import _resume_graph
    from orchestrator.db import task_store as ts

    task_id = "resume-test-001"
    ts._mem[task_id] = {"task_id": task_id, "status": "awaiting_human_approval"}

    fake_final = {
        "status": "completed",
        "final_output": "Done!",
        "subtasks": [],
        "hitl_required": False,
        "hitl_decision": "approved",
        "hitl_notes": "",
        "memory_written": True,
        "supervisor_confidence": 0.9,
        "total_tokens": 100,
        "total_cost_usd": 0.01,
        "error": None,
    }

    with patch("orchestrator.api.routes.review.compiled_graph") as mock_graph, \
         patch("orchestrator.db.engine._pg_available", False):
        mock_graph.invoke.return_value = fake_final
        _resume_graph(task_id, "approved", "looks good")

    # Verify Command(resume=...) was the first argument
    call_args = mock_graph.invoke.call_args
    command_arg = call_args[0][0]
    assert hasattr(command_arg, "resume")
    assert command_arg.resume["decision"] == "approved"
    assert command_arg.resume["notes"] == "looks good"

    # Task store should be updated with completed state
    record = ts.get_task(task_id)
    assert record["status"] == "completed"
    ts._mem.pop(task_id, None)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Supervisor injects rejection_notes into re-plan prompt
# ═══════════════════════════════════════════════════════════════════════════════

def test_supervisor_includes_rejection_notes_in_prompt():
    from orchestrator.agents.supervisor import SupervisorAgent

    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent._threshold = 0.6

    fake_plan = {
        "task_id":               "t-1",
        "original_request":      "do something",
        "subtasks":              [{"id": "st-1", "description": "step", "specialist": "research",
                                   "depends_on": [], "expected_output_format": "text",
                                   "estimated_complexity": "low"}],
        "confidence":            0.8,
        "reasoning":             "ok",
        "requires_human_approval": False,
    }

    captured_messages = []

    def fake_invoke(messages):
        captured_messages.extend(messages)
        resp = MagicMock()
        resp.content = json.dumps(fake_plan)
        resp.usage_metadata = {}
        return resp

    agent.llm = MagicMock()
    agent.llm.invoke.side_effect = fake_invoke

    agent.invoke({
        "task_id":         "t-1",
        "user_request":    "do something",
        "rejection_notes": "Too many subtasks — please simplify",
    })

    # The HumanMessage content should include the rejection block
    human_msg = captured_messages[1]
    assert "REJECTED" in human_msg.content.upper() or "rejected" in human_msg.content.lower()
    assert "Too many subtasks" in human_msg.content
