"""
Execution Replay System
------------------------
Phase 4: loads snapshots from the in-process snapshot_store (no PostgreSQL needed).
Phase 6: swap _load_snapshots() for a SQLAlchemy query.

Usage:
    session = ReplaySession("task-id-here")
    while True:
        step = session.step_forward()
        if step["status"] == "replay_complete":
            break
        print(step)
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ReplaySession:
    """
    Step through the node-by-node execution snapshots of a past task.
    Supports optional input overrides so you can answer "what if X had been different?"
    """

    def __init__(self, task_id: str):
        self.task_id = task_id
        self._snapshots: list[dict] = self._load_snapshots(task_id)
        self._cursor = 0
        logger.info("ReplaySession task=%s snapshots=%d", task_id, len(self._snapshots))

    @staticmethod
    def _load_snapshots(task_id: str) -> list[dict]:
        """
        Load execution snapshots from the in-process store.

        Phase 6 upgrade: replace the import with a SQLAlchemy query:
            SELECT * FROM task_snapshots
            WHERE task_id = :task_id
            ORDER BY step_number ASC
        """
        from orchestrator.observability.snapshot_store import get_snapshots
        return get_snapshots(task_id)

    # ── Playback ──────────────────────────────────────────────────────────────

    def step_forward(self, override_inputs: Optional[dict] = None) -> dict:
        """
        Return the next snapshot and advance the cursor.
        Optionally override fields to see how execution would diverge.
        """
        if self._cursor >= len(self._snapshots):
            return {"status": "replay_complete", "total_steps": len(self._snapshots)}

        snapshot = dict(self._snapshots[self._cursor])
        if override_inputs:
            snapshot["state"] = {**snapshot.get("state", {}), **override_inputs}

        self._cursor += 1
        return snapshot

    def jump_to(self, step: int) -> dict:
        """Jump to a specific step without advancing the cursor."""
        if step < 0 or step >= len(self._snapshots):
            return {"error": f"Step {step} out of range (0–{len(self._snapshots)-1})"}
        return self._snapshots[step]

    def compare(self, step_a: int, step_b: int) -> dict:
        """Diff two steps — useful for seeing what changed between nodes."""
        if step_a >= len(self._snapshots) or step_b >= len(self._snapshots):
            return {}
        a = self._snapshots[step_a].get("state", {})
        b = self._snapshots[step_b].get("state", {})
        diff = {}
        for k in set(list(a.keys()) + list(b.keys())):
            if a.get(k) != b.get(k):
                diff[k] = {"step_a": a.get(k), "step_b": b.get(k)}
        return diff

    def reset(self) -> None:
        self._cursor = 0

    def all_steps(self) -> list[dict]:
        return list(self._snapshots)

    def __len__(self) -> int:
        return len(self._snapshots)
