"""Unit tests for agent components."""
import pytest
from unittest.mock import patch, MagicMock

from orchestrator.agents.supervisor import SupervisorAgent, ExecutionPlan
from orchestrator.agents.reviewer import ReviewerAgent
from orchestrator.hitl.escalation import check_escalation


# ── Supervisor ────────────────────────────────────────────────────────────────

def test_supervisor_produces_valid_plan():
    mock_response = MagicMock()
    mock_response.content = '''{
        "subtasks": [
            {"id": "st-1", "description": "Research AI trends", "specialist": "research",
             "depends_on": [], "expected_output_format": "text", "estimated_complexity": "medium"}
        ],
        "confidence": 0.9,
        "reasoning": "Straightforward research task.",
        "requires_human_approval": false
    }'''
    mock_response.usage_metadata = {"total_tokens": 150}

    with patch.object(SupervisorAgent, "__init__", lambda self: None):
        agent = SupervisorAgent.__new__(SupervisorAgent)
        agent.name = "supervisor"
        agent.model = "test-model"
        agent.llm = MagicMock(invoke=MagicMock(return_value=mock_response))
        agent._threshold = 0.7
        agent._logger = MagicMock()

        result = agent.invoke({"task_id": "t-1", "user_request": "Research AI trends"})

    assert result.confidence == 0.9
    plan = ExecutionPlan(**result.output)
    assert len(plan.subtasks) == 1
    assert plan.subtasks[0].specialist == "research"


# ── Reviewer ──────────────────────────────────────────────────────────────────

def test_reviewer_approves_good_output():
    mock_response = MagicMock()
    mock_response.content = '{"approved": true, "quality_score": 0.95, "feedback": "", "issues": []}'
    mock_response.usage_metadata = {"total_tokens": 80}

    with patch.object(ReviewerAgent, "__init__", lambda self: None):
        agent = ReviewerAgent.__new__(ReviewerAgent)
        agent.name = "reviewer"
        agent.model = "test-model"
        agent.llm = MagicMock(invoke=MagicMock(return_value=mock_response))
        agent._threshold = 0.75
        agent._logger = MagicMock()

        result = agent.invoke({
            "subtask_description": "Research AI trends",
            "specialist_output": "AI trends include LLMs, multi-agent systems...",
        })

    assert result.output["approved"] is True
    assert result.confidence >= 0.75


# ── Escalation logic ──────────────────────────────────────────────────────────

def test_escalation_triggers_on_low_confidence():
    should_escalate, level = check_escalation(confidence=0.5)
    assert should_escalate is True
    assert level == "approve_plan"


def test_escalation_triggers_on_sensitive_operation():
    should_escalate, level = check_escalation(confidence=0.9, operation="send_email")
    assert should_escalate is True
    assert level == "approve_action"


def test_no_escalation_on_normal_task():
    should_escalate, level = check_escalation(confidence=0.95, quality_score=0.9)
    assert should_escalate is False
    assert level is None
