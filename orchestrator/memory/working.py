"""Short-term working memory backed by Redis (TTL-scoped per task)."""
import json
import redis
from orchestrator.config import get_settings

TTL_SECONDS = 3600  # 1 hour


class WorkingMemory:

    def __init__(self):
        self._redis = redis.from_url(get_settings().redis_url, decode_responses=True)
        self._local: dict = {}  # in-process fallback when Redis is unavailable

    def _available(self) -> bool:
        try:
            self._redis.ping()
            return True
        except Exception:
            return False

    def _key(self, task_id: str, field: str) -> str:
        return f"wm:{task_id}:{field}"

    def set(self, task_id: str, key: str, value) -> None:
        k = self._key(task_id, key)
        if self._available():
            self._redis.setex(k, TTL_SECONDS, json.dumps(value))
        else:
            self._local[k] = value

    def get(self, task_id: str, key: str):
        k = self._key(task_id, key)
        if self._available():
            raw = self._redis.get(k)
            return json.loads(raw) if raw else None
        return self._local.get(k)

    def clear(self, task_id: str) -> None:
        if self._available():
            pattern = f"wm:{task_id}:*"
            keys = self._redis.keys(pattern)
            if keys:
                self._redis.delete(*keys)
        else:
            self._local = {k: v for k, v in self._local.items()
                           if not k.startswith(f"wm:{task_id}:")}
