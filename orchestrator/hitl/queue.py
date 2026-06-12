"""
HITL Review Queue — backed by Redis lists.
The review UI polls this queue; the API resumes the LangGraph execution
once a decision is posted.
"""
from __future__ import annotations

import json
import time
import redis

from orchestrator.config import get_settings

QUEUE_KEY = "hitl:pending"
DECISION_PREFIX = "hitl:decision:"


class HITLQueue:

    def __init__(self):
        self._redis = redis.from_url(get_settings().redis_url, decode_responses=True)

    def _available(self) -> bool:
        """Return True if Redis is reachable."""
        try:
            self._redis.ping()
            return True
        except Exception:
            return False

    def push(self, payload: dict) -> None:
        """Add a review request to the pending queue."""
        if not self._available():
            return
        payload["queued_at"] = time.time()
        self._redis.lpush(QUEUE_KEY, json.dumps(payload))

    def pop(self) -> dict | None:
        """Non-blocking pop — returns None if queue is empty or Redis is down."""
        if not self._available():
            return None
        item = self._redis.rpop(QUEUE_KEY)
        return json.loads(item) if item else None

    def list_pending(self) -> list[dict]:
        """Return all pending items; empty list if Redis is unavailable."""
        if not self._available():
            return []
        items = self._redis.lrange(QUEUE_KEY, 0, -1)
        return [json.loads(i) for i in items]

    def post_decision(self, task_id: str, decision: str, notes: str = "") -> None:
        """Record a human decision; no-op if Redis is unavailable."""
        if not self._available():
            return
        key = f"{DECISION_PREFIX}{task_id}"
        self._redis.setex(key, 3600, json.dumps({"decision": decision, "notes": notes}))

    def get_decision(self, task_id: str) -> dict | None:
        """Poll for a decision; returns None if Redis is unavailable."""
        if not self._available():
            return None
        raw = self._redis.get(f"{DECISION_PREFIX}{task_id}")
        return json.loads(raw) if raw else None
