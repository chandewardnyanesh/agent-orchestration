"""
LLM Cost Tracker
-----------------
Phase 4:
- Graceful degradation: falls back to in-process list when Redis is unavailable.
- Cost alert: logs a WARNING when a task exceeds the configured threshold.
- tokens_to_usd() is importable standalone (used by snapshot_store.task_summary).
"""
from __future__ import annotations

import json
import logging

from orchestrator.config import get_settings

logger = logging.getLogger(__name__)

# Approximate prices per 1 M tokens (input + output blended) — update as needed
MODEL_COST_PER_1M: dict[str, float] = {
    "claude-opus-4-6":           15.00,
    "claude-sonnet-4-6":          3.00,
    "claude-haiku-4-5-20251001":  0.25,
    "gpt-4o":                    10.00,
    "gpt-4o-mini":                0.30,
}

# In-process fallback store (keyed by task_id)
_local_store: dict[str, list[dict]] = {}


def tokens_to_usd(tokens: int, model: str) -> float:
    """Convert token count to USD using the configured model rate."""
    rate = MODEL_COST_PER_1M.get(model, 5.0)
    return round((tokens / 1_000_000) * rate, 6)


class CostTracker:

    def __init__(self):
        self._cfg = get_settings()
        self._redis = None
        try:
            import redis
            r = redis.from_url(self._cfg.redis_url, decode_responses=True)
            r.ping()
            self._redis = r
        except Exception:
            pass   # Redis unavailable — use in-process fallback

    def _available(self) -> bool:
        if not self._redis:
            return False
        try:
            self._redis.ping()
            return True
        except Exception:
            return False

    # ── Write ─────────────────────────────────────────────────────────────────

    def record(self, task_id: str, agent_name: str, model: str, tokens: int) -> float:
        """
        Record a single agent invocation cost.
        Returns the USD cost of this call.
        """
        cost = tokens_to_usd(tokens, model)
        entry = {"agent": agent_name, "model": model, "tokens": tokens, "usd": cost}

        if self._available():
            key = f"cost:{task_id}"
            self._redis.lpush(key, json.dumps(entry))
            self._redis.expire(key, 86400 * 7)   # keep 7 days
        else:
            # In-process fallback
            if task_id not in _local_store:
                _local_store[task_id] = []
            _local_store[task_id].append(entry)

        return cost

    # ── Read ──────────────────────────────────────────────────────────────────

    def total_for_task(self, task_id: str) -> dict:
        """
        Aggregate all recorded costs for a task.
        Merges Redis and in-process entries so nothing is lost.
        """
        items: list[dict] = []

        if self._available():
            raw = self._redis.lrange(f"cost:{task_id}", 0, -1)
            items.extend(json.loads(r) for r in raw)

        # Always include in-process entries (may exist alongside Redis)
        items.extend(_local_store.get(task_id, []))

        total_tokens = sum(i["tokens"] for i in items)
        total_usd    = sum(i["usd"]    for i in items)

        return {
            "task_id":      task_id,
            "total_tokens": total_tokens,
            "total_usd":    round(total_usd, 6),
            "breakdown":    items,
        }

    # ── Alert ─────────────────────────────────────────────────────────────────

    def check_alert(self, task_id: str) -> bool:
        """
        Return True and log a WARNING if total cost exceeds the threshold.
        Called automatically at the end of each task.
        """
        threshold = self._cfg.cost_alert_threshold_usd
        totals = self.total_for_task(task_id)
        if totals["total_usd"] > threshold:
            logger.warning(
                "COST ALERT task=%s total_usd=%.4f exceeds threshold=%.2f",
                task_id, totals["total_usd"], threshold,
            )
            return True
        return False
