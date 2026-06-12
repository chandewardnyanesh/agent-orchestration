"""
Escalation triggers — decides WHEN and at what level to involve a human.
"""
from __future__ import annotations

from typing import Literal

from orchestrator.config import get_settings

HITLLevel = Literal["notify", "approve_action", "approve_plan", "take_over"]

# Operations considered sensitive → always escalate to approve_action
SENSITIVE_OPERATIONS = {"send_email", "delete_data", "financial_transaction", "external_api_write"}

settings = get_settings()


def check_escalation(
    confidence: float,
    operation: str | None = None,
    retry_count: int = 0,
    quality_score: float = 1.0,
    user_requested: bool = False,
) -> tuple[bool, HITLLevel | None]:
    """
    Returns (should_escalate: bool, level: HITLLevel | None).

    Escalation rules (ordered by severity):
    1. User explicitly requested review → approve_action
    2. Sensitive operation → approve_action
    3. Specialist failed twice on same subtask → take_over
    4. Reviewer quality score below threshold → approve_action
    5. Supervisor confidence below threshold → approve_plan
    """
    if user_requested:
        return True, "approve_action"

    if operation and operation in SENSITIVE_OPERATIONS:
        return True, "approve_action"

    if retry_count >= settings.max_specialist_retries:
        return True, "take_over"

    if quality_score < settings.reviewer_quality_threshold:
        return True, "approve_action"

    if confidence < settings.supervisor_confidence_threshold:
        return True, "approve_plan"

    return False, None
