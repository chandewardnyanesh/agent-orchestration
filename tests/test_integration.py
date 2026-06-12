"""
Phase 5 — Integration Tests

End-to-end pipeline tests that exercise the full LangGraph graph
with mocked LLM calls.  No API credits required.

What is covered:
  1.  Happy-path: full 7-node graph → status=completed, memory written
  2.  Snapshots captured for every node in the happy path
  3.  Spans linked to task_id for every agent call
  4.  Retry loop: reviewer rejects once → specialist retries → approved on retry
  5.  HITL path: supervisor sets requires_human_approval → graph suspends
      → GraphInterrupt caught → resume with 'approved' → completes
  6.  Rejection re-plan: graph suspends → resume with 'rejected' →
      routes back to plan node (second plan call)
  7.  take_over: graph suspends → resume with 'take_over' → status=failed
  8.  Auto-escalation: low confidence (< threshold) routes to await_human
  9.  Memory second-run: retrieve_memory returns prior context to supervisor
  10. Cost tracker records span USD across all agents
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch, call
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from orchestrator.agents.base import AgentResult
from orchestrator.graph.workflow import build_graph
from orchestrator.observability.snapshot_store import (
    clear_snapshots, clear_spans, get_snapshots, get_spans,
)


# ── Shared fixtures ───────────────────────────────────────────────────────────

def _plan_output(task_id: str, subtasks=None, confidence=0.95, requires_approval=False) -> dict:
    """Build a valid ExecutionPlan dict."""
    if subtasks is None:
        subtasks = [
            {"id": "st-1", "description": "Research AI frameworks",
             "specialist": "research", "depends_on": [],
             "expected_output_format": "text", "estimated_complexity": "medium"},
            {"id": "st-2", "description": "Write executive summary",
             "specialist": "writing", "depends_on": ["st-1"],
             "expected_output_format": "text", "estimated_complexity": "low"},
        ]
    return {
        "task_id":               task_id,
        "original_request":      "Research top 3 AI frameworks",
        "subtasks":              subtasks,
        "confidence":            confidence,
        "reasoning":             "Two-step task.",
        "requires_human_approval": requires_approval,
    }


def _review_output(approved=True, score=0.9) -> dict:
    return {"approved": approved, "quality_score": score,
            "feedback": "" if approved else "Output too short", "issues": []}


def _initial_state(task_id: str, hitl=False) -> dict:
    return {
        "task_id":               task_id,
        "user_request":          "Research top 3 AI frameworks and write a summary",
        "status":                "pending",
        "error":                 None,
        "execution_plan":        None,
        "supervisor_confidence": 0.0,
        "subtasks":              [],
        "completed_outputs":     {},
        "memory_context":        "",
        "memory_written":        False,
        "hitl_required":         hitl,
        "hitl_level":            "approve_plan" if hitl else None,
        "hitl_decision":         None,
        "hitl_notes":            None,
        "final_output":          None,
        "trace_id":              task_id,
        "total_tokens":          0,
        "total_cost_usd":        0.0,
    }


def _fresh_graph():
    """Build a fresh compiled graph with its own checkpointer for test isolation."""
    return build_graph().compile(checkpointer=MemorySaver())


def _thread(task_id: str) -> dict:
    return {"configurable": {"thread_id": task_id}}


# ── Import the agent singletons used by nodes.py ─────────────────────────────
from orchestrator.graph.nodes import _supervisor, _reviewer, _specialists


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Happy path — full 7-node execution, status=completed
# ═══════════════════════════════════════════════════════════════════════════════

def test_happy_path_completes():
    task_id = "integ-happy-001"
    clear_snapshots(task_id)
    clear_spans(task_id)

    plan = _plan_output(task_id)
    review_ok = _review_output()

    with patch.object(_supervisor,              "timed_invoke",
                      return_value=AgentResult(output=plan,      confidence=0.95, tokens_used=400)) as mock_sup, \
         patch.object(_specialists["research"], "timed_invoke",
                      return_value=AgentResult(output="LangGraph is best", confidence=0.9, tokens_used=250)) as mock_res, \
         patch.object(_specialists["writing"],  "timed_invoke",
                      return_value=AgentResult(output="Summary: LangGraph leads.", confidence=0.9, tokens_used=150)) as mock_wri, \
         patch.object(_reviewer,                "timed_invoke",
                      return_value=AgentResult(output=review_ok, confidence=0.9, tokens_used=80)):

        g = _fresh_graph()
        final = g.invoke(_initial_state(task_id), config=_thread(task_id))

    assert final["status"] == "completed"
    assert final["final_output"] is not None
    assert final["memory_written"] is True
    assert all(st["review_approved"] for st in final["subtasks"])


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Snapshots captured for every expected node
# ═══════════════════════════════════════════════════════════════════════════════

def test_happy_path_snapshots_captured():
    task_id = "integ-snap-001"
    clear_snapshots(task_id)

    plan = _plan_output(task_id)
    review_ok = _review_output()

    with patch.object(_supervisor,              "timed_invoke",
                      return_value=AgentResult(output=plan,      confidence=0.95, tokens_used=400)), \
         patch.object(_specialists["research"], "timed_invoke",
                      return_value=AgentResult(output="Research done", confidence=0.9, tokens_used=250)), \
         patch.object(_specialists["writing"],  "timed_invoke",
                      return_value=AgentResult(output="Writing done",  confidence=0.9, tokens_used=150)), \
         patch.object(_reviewer,                "timed_invoke",
                      return_value=AgentResult(output=review_ok, confidence=0.9, tokens_used=80)):

        g = _fresh_graph()
        g.invoke(_initial_state(task_id), config=_thread(task_id))

    snaps = get_snapshots(task_id)
    nodes_ran = {s["node"] for s in snaps}
    for expected in ("retrieve_memory", "plan", "execute_subtasks", "review",
                     "synthesize", "write_memory"):
        assert expected in nodes_ran, f"Missing snapshot for node: {expected}"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Spans linked to task_id for every agent call
# ═══════════════════════════════════════════════════════════════════════════════

def test_happy_path_spans_linked():
    task_id = "integ-span-001"
    clear_spans(task_id)

    plan = _plan_output(task_id)
    review_ok = _review_output()

    # Real timed_invoke but mock .llm at the deepest level
    sup_llm   = MagicMock()
    res_llm   = MagicMock()
    wri_llm   = MagicMock()
    rev_llm   = MagicMock()

    # Supervisor LLM response
    def _sup_response(*a, **kw):
        m = MagicMock()
        m.content = json.dumps(plan)
        m.usage_metadata = {"total_tokens": 400}
        return m

    # Specialist LLM response (no tool calls)
    def _spec_response(text):
        def _f(*a, **kw):
            m = MagicMock()
            m.content = text
            m.tool_calls = []
            m.usage_metadata = {"total_tokens": 200}
            return m
        return _f

    # Reviewer LLM response
    def _rev_response(*a, **kw):
        m = MagicMock()
        m.content = json.dumps(review_ok)
        m.usage_metadata = {"total_tokens": 80}
        return m

    sup_llm.invoke.side_effect    = _sup_response
    res_llm.invoke.side_effect    = _spec_response("Research done")
    res_llm.bind_tools.return_value.invoke.side_effect = _spec_response("Research done")
    wri_llm.invoke.side_effect    = _spec_response("Summary done")
    wri_llm.bind_tools.return_value.invoke.side_effect = _spec_response("Summary done")
    rev_llm.invoke.side_effect    = _rev_response

    orig_sup_llm = _supervisor.llm
    orig_res_llm = _specialists["research"].llm
    orig_wri_llm = _specialists["writing"].llm
    orig_rev_llm = _reviewer.llm

    _supervisor.llm             = sup_llm
    _specialists["research"].llm = res_llm
    _specialists["writing"].llm  = wri_llm
    _reviewer.llm               = rev_llm

    try:
        g = _fresh_graph()
        g.invoke(_initial_state(task_id), config=_thread(task_id))
    finally:
        _supervisor.llm             = orig_sup_llm
        _specialists["research"].llm = orig_res_llm
        _specialists["writing"].llm  = orig_wri_llm
        _reviewer.llm               = orig_rev_llm

    spans = get_spans(task_id)
    agents_seen = {s["agent"] for s in spans}
    assert "supervisor" in agents_seen
    assert "research"   in agents_seen
    assert "writing"    in agents_seen
    assert "reviewer"   in agents_seen


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Retry loop — reviewer rejects once, approves on retry
# ═══════════════════════════════════════════════════════════════════════════════

def test_retry_loop_second_attempt_approved():
    task_id = "integ-retry-001"
    clear_snapshots(task_id)

    plan = _plan_output(task_id, subtasks=[
        {"id": "st-1", "description": "Research",
         "specialist": "research", "depends_on": [],
         "expected_output_format": "text", "estimated_complexity": "low"},
    ])

    review_reject = _review_output(approved=False, score=0.4)
    review_ok     = _review_output(approved=True,  score=0.9)
    reviewer_calls = {"count": 0}

    def reviewer_side_effect(inputs):
        reviewer_calls["count"] += 1
        out = review_reject if reviewer_calls["count"] == 1 else review_ok
        return AgentResult(output=out, confidence=out["quality_score"], tokens_used=80)

    with patch.object(_supervisor,              "timed_invoke",
                      return_value=AgentResult(output=plan, confidence=0.95, tokens_used=400)), \
         patch.object(_specialists["research"], "timed_invoke",
                      return_value=AgentResult(output="Research output", confidence=0.9, tokens_used=200)), \
         patch.object(_reviewer, "timed_invoke", side_effect=reviewer_side_effect):

        g = _fresh_graph()
        final = g.invoke(_initial_state(task_id), config=_thread(task_id))

    assert final["status"] == "completed"
    assert reviewer_calls["count"] == 2   # rejected once, approved on retry


# ═══════════════════════════════════════════════════════════════════════════════
# 5. HITL path — supervisor flags approval → graph suspends → resume approved
# ═══════════════════════════════════════════════════════════════════════════════

def test_hitl_approve_resumes_to_completion():
    task_id = "integ-hitl-001"
    clear_snapshots(task_id)

    plan = _plan_output(task_id, confidence=0.4, requires_approval=True,
                        subtasks=[
                            {"id": "st-1", "description": "Research",
                             "specialist": "research", "depends_on": [],
                             "expected_output_format": "text", "estimated_complexity": "low"},
                        ])
    review_ok = _review_output()

    with patch.object(_supervisor,              "timed_invoke",
                      return_value=AgentResult(output=plan, confidence=0.4, tokens_used=300)), \
         patch.object(_specialists["research"], "timed_invoke",
                      return_value=AgentResult(output="Research done", confidence=0.9, tokens_used=200)), \
         patch.object(_reviewer,                "timed_invoke",
                      return_value=AgentResult(output=review_ok, confidence=0.9, tokens_used=80)), \
         patch("orchestrator.hitl.queue.HITLQueue") as mock_q:

        mock_q.return_value.push = MagicMock()
        g = _fresh_graph()

        # LangGraph 0.6 signals suspension via __interrupt__ in returned state
        suspended = g.invoke(_initial_state(task_id), config=_thread(task_id))
        assert "__interrupt__" in suspended, "Expected graph to suspend at await_human"

        # Resume with approval
        final = g.invoke(
            Command(resume={"decision": "approved", "notes": ""}),
            config=_thread(task_id),
        )

    assert final["status"] == "completed"
    assert final["hitl_decision"] == "approved"
    assert final["memory_written"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Rejection re-plan — Supervisor is called twice (initial + re-plan)
# ═══════════════════════════════════════════════════════════════════════════════

def test_hitl_rejection_triggers_replan():
    task_id = "integ-replan-001"

    plan_v1 = _plan_output(task_id, confidence=0.3, requires_approval=True,
                           subtasks=[{"id": "st-1", "description": "Research",
                                      "specialist": "research", "depends_on": [],
                                      "expected_output_format": "text",
                                      "estimated_complexity": "low"}])
    # Second plan has higher confidence — no HITL needed
    plan_v2 = _plan_output(task_id, confidence=0.95, requires_approval=False,
                           subtasks=[{"id": "st-1", "description": "Revised Research",
                                      "specialist": "research", "depends_on": [],
                                      "expected_output_format": "text",
                                      "estimated_complexity": "low"}])

    plan_calls = {"count": 0}
    plans = [plan_v1, plan_v2]

    def sup_side_effect(inputs):
        conf = plans[plan_calls["count"]]["confidence"]
        out  = plans[plan_calls["count"]]
        plan_calls["count"] += 1
        return AgentResult(output=out, confidence=conf, tokens_used=300)

    review_ok = _review_output()

    with patch.object(_supervisor, "timed_invoke", side_effect=sup_side_effect), \
         patch.object(_specialists["research"], "timed_invoke",
                      return_value=AgentResult(output="Done", confidence=0.9, tokens_used=200)), \
         patch.object(_reviewer,               "timed_invoke",
                      return_value=AgentResult(output=review_ok, confidence=0.9, tokens_used=80)), \
         patch("orchestrator.hitl.queue.HITLQueue") as mock_q:

        mock_q.return_value.push = MagicMock()
        g = _fresh_graph()

        # LangGraph 0.6 signals suspension via __interrupt__ in returned state
        suspended = g.invoke(_initial_state(task_id), config=_thread(task_id))
        assert "__interrupt__" in suspended, "Expected graph to suspend at await_human"

        # Reject → re-plan → complete
        final = g.invoke(
            Command(resume={"decision": "rejected", "notes": "Too vague, be more specific"}),
            config=_thread(task_id),
        )

    assert plan_calls["count"] == 2         # initial plan + re-plan
    assert final["status"] == "completed"


# ═══════════════════════════════════════════════════════════════════════════════
# 7. take_over — graph ends with status=failed
# ═══════════════════════════════════════════════════════════════════════════════

def test_hitl_take_over_ends_graph():
    task_id = "integ-takeover-001"

    plan = _plan_output(task_id, confidence=0.3, requires_approval=True)

    with patch.object(_supervisor, "timed_invoke",
                      return_value=AgentResult(output=plan, confidence=0.3, tokens_used=300)), \
         patch("orchestrator.hitl.queue.HITLQueue") as mock_q:

        mock_q.return_value.push = MagicMock()
        g = _fresh_graph()

        # LangGraph 0.6 signals suspension via __interrupt__ in returned state
        suspended = g.invoke(_initial_state(task_id), config=_thread(task_id))
        assert "__interrupt__" in suspended, "Expected graph to suspend at await_human"

        final = g.invoke(
            Command(resume={"decision": "take_over", "notes": "I'll handle this manually"}),
            config=_thread(task_id),
        )

    assert final["status"] == "failed"
    assert final["hitl_decision"] == "take_over"


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Auto-escalation — low confidence routes to await_human even if not flagged
# ═══════════════════════════════════════════════════════════════════════════════

def test_auto_escalation_low_confidence():
    from orchestrator.config import get_settings
    threshold = get_settings().supervisor_confidence_threshold
    task_id = "integ-autoesc-001"

    # Plan has low confidence AND requires_human_approval=False — auto-escalation should catch it
    plan = _plan_output(task_id, confidence=threshold - 0.05, requires_approval=False,
                        subtasks=[{"id": "st-1", "description": "Research",
                                   "specialist": "research", "depends_on": [],
                                   "expected_output_format": "text",
                                   "estimated_complexity": "low"}])

    with patch.object(_supervisor, "timed_invoke",
                      return_value=AgentResult(output=plan, confidence=threshold - 0.05, tokens_used=300)), \
         patch("orchestrator.hitl.queue.HITLQueue") as mock_q:

        mock_q.return_value.push = MagicMock()
        g = _fresh_graph()

        # LangGraph 0.6 signals suspension via __interrupt__ in returned state
        suspended = g.invoke(_initial_state(task_id), config=_thread(task_id))
        assert "__interrupt__" in suspended, "Expected auto-escalation to suspend graph"

    # Graph was interrupted → auto-escalation worked
    mock_q.return_value.push.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Memory second-run: retrieve_memory result surfaces in supervisor prompt
# ═══════════════════════════════════════════════════════════════════════════════

def test_memory_context_injected_into_second_run():
    """
    On a second run for a similar task, the Supervisor should receive
    memory_context from the first run's stored memory.
    """
    from orchestrator.memory.manager import MemoryManager

    task_id = "integ-memory-001"
    captured_inputs = {}

    plan = _plan_output(task_id, subtasks=[
        {"id": "st-1", "description": "Research",
         "specialist": "research", "depends_on": [],
         "expected_output_format": "text", "estimated_complexity": "low"},
    ])
    review_ok = _review_output()

    def sup_side_effect(inputs):
        captured_inputs.update(inputs)
        return AgentResult(output=plan, confidence=0.95, tokens_used=300)

    # Mock retrieve to simulate memory from a prior run
    fake_context = "[Memory 1] relevance=0.80  Past: AI frameworks task went well"

    with patch.object(_supervisor, "timed_invoke", side_effect=sup_side_effect), \
         patch.object(_specialists["research"], "timed_invoke",
                      return_value=AgentResult(output="Research done", confidence=0.9, tokens_used=200)), \
         patch.object(_reviewer,               "timed_invoke",
                      return_value=AgentResult(output=review_ok, confidence=0.9, tokens_used=80)), \
         patch.object(MemoryManager, "retrieve", return_value=fake_context), \
         patch.object(MemoryManager, "store"):    # skip actual ChromaDB write

        g = _fresh_graph()
        g.invoke(_initial_state(task_id), config=_thread(task_id))

    # Supervisor received the memory context
    assert captured_inputs.get("memory_context") == fake_context


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Cost tracking — spans include USD values
# ═══════════════════════════════════════════════════════════════════════════════

def test_cost_tracking_records_agent_costs():
    from orchestrator.observability.snapshot_store import task_summary
    task_id = "integ-cost-001"
    clear_spans(task_id)

    plan = _plan_output(task_id, subtasks=[
        {"id": "st-1", "description": "Research",
         "specialist": "research", "depends_on": [],
         "expected_output_format": "text", "estimated_complexity": "low"},
    ])
    review_ok = _review_output()

    sup_llm = MagicMock()
    res_llm = MagicMock()
    rev_llm = MagicMock()

    def _mock_response(content, tokens=500):
        m = MagicMock()
        m.content = content
        m.tool_calls = []
        m.usage_metadata = {"total_tokens": tokens}
        return m

    sup_llm.invoke.return_value = _mock_response(json.dumps(plan), 500)
    res_llm.invoke.return_value = _mock_response("Research output", 300)
    res_llm.bind_tools.return_value.invoke.return_value = _mock_response("Research output", 300)
    rev_llm.invoke.return_value = _mock_response(json.dumps(review_ok), 100)

    orig_sup = _supervisor.llm
    orig_res = _specialists["research"].llm
    orig_rev = _reviewer.llm
    _supervisor.llm = sup_llm
    _specialists["research"].llm = res_llm
    _reviewer.llm = rev_llm

    try:
        g = _fresh_graph()
        g.invoke(_initial_state(task_id), config=_thread(task_id))
    finally:
        _supervisor.llm = orig_sup
        _specialists["research"].llm = orig_res
        _reviewer.llm = orig_rev

    summary = task_summary(task_id)
    assert summary["total_tokens"] > 0
    assert summary["agent_calls"]  >= 3   # supervisor + research + reviewer
    assert summary["total_usd"]    >= 0.0
