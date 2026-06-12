"""
OpenTelemetry tracing setup + convenience span context manager.
"""
from __future__ import annotations

import functools
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

from orchestrator.config import get_settings

settings = get_settings()

# ── Provider setup (call once at app startup) ─────────────────────────────────

def setup_tracing(service_name: str = "agent-orchestrator") -> None:
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    try:
        exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
    except Exception:
        pass  # Run without export if OTLP is not available

    trace.set_tracer_provider(provider)


tracer = trace.get_tracer("agent-orchestrator")


# ── Convenience helpers ───────────────────────────────────────────────────────

@contextmanager
def agent_span(name: str, task_id: str = "", agent: str = ""):
    """Context manager that wraps an agent step in an OTel span."""
    with tracer.start_as_current_span(name) as span:
        if task_id:
            span.set_attribute("task.id", task_id)
        if agent:
            span.set_attribute("agent.name", agent)
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            raise


def trace_agent(func):
    """Decorator to auto-span any BaseAgent.invoke() call."""
    @functools.wraps(func)
    def wrapper(self, inputs: dict):
        with agent_span(f"{self.name}.invoke", task_id=inputs.get("task_id", ""), agent=self.name) as span:
            result = func(self, inputs)
            span.set_attribute("agent.confidence", result.confidence)
            span.set_attribute("agent.tokens", result.tokens_used)
            return result
    return wrapper
