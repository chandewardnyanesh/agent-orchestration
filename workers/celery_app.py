"""Celery worker setup — specialists can be dispatched as async tasks."""
from celery import Celery
from orchestrator.config import get_settings

settings = get_settings()

app = Celery(
    "agent_orchestration",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # one task at a time per worker for agent safety
)


@app.task(name="run_specialist", bind=True, max_retries=2)
def run_specialist(self, specialist_type: str, inputs: dict) -> dict:
    """
    Celery task wrapper for specialist agents.
    Allows async, distributed execution across worker nodes.
    """
    from orchestrator.agents.specialists.research import ResearchAgent
    from orchestrator.agents.specialists.analysis import AnalysisAgent
    from orchestrator.agents.specialists.writing import WritingAgent
    from orchestrator.agents.specialists.code import CodeAgent

    agents = {
        "research": ResearchAgent,
        "analysis": AnalysisAgent,
        "writing": WritingAgent,
        "code": CodeAgent,
    }

    AgentClass = agents.get(specialist_type)
    if not AgentClass:
        raise ValueError(f"Unknown specialist: {specialist_type}")

    agent = AgentClass()
    result = agent.timed_invoke(inputs)
    return result.to_dict()
