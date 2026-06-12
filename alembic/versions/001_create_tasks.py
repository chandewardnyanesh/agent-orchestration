"""Create tasks table

Revision ID: 001
Revises:
Create Date: 2026-06-10
"""
from alembic import op
import sqlalchemy as sa

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tasks",
        sa.Column("task_id",             sa.String(64),  primary_key=True),
        sa.Column("status",              sa.String(32),  nullable=False, server_default="pending"),
        sa.Column("user_request",        sa.Text,        nullable=False, server_default=""),
        sa.Column("supervisor_confidence", sa.Float,     nullable=False, server_default="0.0"),
        sa.Column("total_tokens",        sa.Integer,     nullable=False, server_default="0"),
        sa.Column("total_cost_usd",      sa.Float,       nullable=False, server_default="0.0"),
        sa.Column("final_output",        sa.Text,        nullable=True),
        sa.Column("subtasks",            sa.JSON,        nullable=False, server_default="[]"),
        sa.Column("hitl_required",       sa.Boolean,     nullable=False, server_default="false"),
        sa.Column("hitl_decision",       sa.String(32),  nullable=True),
        sa.Column("hitl_notes",          sa.Text,        nullable=True),
        sa.Column("memory_written",      sa.Boolean,     nullable=False, server_default="false"),
        sa.Column("error",               sa.Text,        nullable=True),
        sa.Column("created_at",          sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.Column("updated_at",          sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index("ix_tasks_status",     "tasks", ["status"])
    op.create_index("ix_tasks_created_at", "tasks", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_tasks_created_at", table_name="tasks")
    op.drop_index("ix_tasks_status",     table_name="tasks")
    op.drop_table("tasks")
