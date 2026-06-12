from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    # LLM
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    supervisor_model: str = "claude-opus-4-6"
    specialist_model: str = "claude-sonnet-4-6"
    reviewer_model: str = "claude-haiku-4-5-20251001"

    # Agent behaviour
    supervisor_confidence_threshold: float = 0.7
    reviewer_quality_threshold: float = 0.75
    max_specialist_retries: int = 2
    max_task_timeout_seconds: int = 300

    # Infrastructure
    database_url: str = "postgresql://agent:secret@localhost:5432/orchestration"
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection: str = "agent_memory"

    # Observability
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"
    enable_cost_tracking: bool = True
    cost_alert_threshold_usd: float = 1.0   # warn when a task exceeds $1

    # Security
    api_secret_key: str = "change-me"
    code_executor_sandbox: bool = True

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
