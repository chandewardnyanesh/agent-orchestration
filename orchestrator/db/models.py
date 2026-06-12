"""SQLAlchemy ORM models."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, JSON, Text

from orchestrator.db.engine import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class TaskRecord(Base):
    __tablename__ = "tasks"

    task_id             = Column(String(64),  primary_key=True, index=True)
    status              = Column(String(32),  nullable=False, default="pending")
    user_request        = Column(Text,        nullable=False, default="")
    supervisor_confidence = Column(Float,     nullable=False, default=0.0)
    total_tokens        = Column(Integer,     nullable=False, default=0)
    total_cost_usd      = Column(Float,       nullable=False, default=0.0)
    final_output        = Column(Text,        nullable=True)
    subtasks            = Column(JSON,        nullable=False, default=list)
    hitl_required       = Column(Boolean,     nullable=False, default=False)
    hitl_decision       = Column(String(32),  nullable=True)
    hitl_notes          = Column(Text,        nullable=True)
    memory_written      = Column(Boolean,     nullable=False, default=False)
    error               = Column(Text,        nullable=True)
    created_at          = Column(DateTime(timezone=True), nullable=False, default=_now)
    updated_at          = Column(DateTime(timezone=True), nullable=False,
                                 default=_now, onupdate=_now)

    def to_dict(self) -> dict:
        return {
            "task_id":               self.task_id,
            "status":                self.status,
            "user_request":          self.user_request,
            "supervisor_confidence": self.supervisor_confidence,
            "total_tokens":          self.total_tokens,
            "total_cost_usd":        self.total_cost_usd,
            "final_output":          self.final_output,
            "subtasks":              self.subtasks or [],
            "hitl_required":         self.hitl_required,
            "hitl_decision":         self.hitl_decision,
            "hitl_notes":            self.hitl_notes,
            "memory_written":        self.memory_written,
            "error":                 self.error,
            "created_at":            self.created_at.isoformat() if self.created_at else None,
            "updated_at":            self.updated_at.isoformat() if self.updated_at else None,
        }
