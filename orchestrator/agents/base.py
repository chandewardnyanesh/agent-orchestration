"""Base agent class shared by all agents in the hierarchy."""
from __future__ import annotations

import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

from langchain_core.messages import ToolMessage

from orchestrator.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def run_tool_loop(
    llm_with_tools,
    messages: list,
    tools: list,
    max_iterations: int = 10,
) -> Tuple[Any, list, int]:
    """
    Drive the LLM ↔ tool call cycle until the LLM stops requesting tools.

    Returns:
        (final_response, tool_call_records, total_tokens_used)

    Each record in tool_call_records is {"name": str, "result": str (truncated)}.
    Stops early if the model issues no tool calls, or after max_iterations rounds.
    """
    tools_by_name = {t.name: t for t in tools if t is not None}
    all_records: list[dict] = []
    total_tokens = 0

    for _ in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        usage = getattr(response, "usage_metadata", {}) or {}
        total_tokens += usage.get("total_tokens", 0)

        tc_list = getattr(response, "tool_calls", [])
        if not tc_list:
            return response, all_records, total_tokens

        tool_messages: list[ToolMessage] = []
        for tc in tc_list:
            tool = tools_by_name.get(tc["name"])
            if tool:
                try:
                    result = tool.invoke(tc["args"])
                except Exception as exc:
                    result = f"Tool error: {exc}"
            else:
                result = f"Tool '{tc['name']}' is not available."

            all_records.append({"name": tc["name"], "result": str(result)[:500]})
            tool_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        messages = messages + [response] + tool_messages

    # Reached max iterations — one final call to force a text response
    response = llm_with_tools.invoke(messages)
    usage = getattr(response, "usage_metadata", {}) or {}
    total_tokens += usage.get("total_tokens", 0)
    return response, all_records, total_tokens


def _build_llm(model: str):
    """
    Return a LangChain-compatible chat model.
    Supports both Anthropic and OpenAI — whichever key is set.
    """
    if "claude" in model.lower():
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            anthropic_api_key=settings.anthropic_api_key,
            max_tokens=4096,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            openai_api_key=settings.openai_api_key,
        )


class AgentResult:
    """Typed container for every agent invocation result."""

    def __init__(
        self,
        output: Any,
        confidence: float = 1.0,
        tool_calls: Optional[list] = None,
        tokens_used: int = 0,
        latency_ms: float = 0.0,
        error: Optional[str] = None,
    ):
        self.output = output
        self.confidence = confidence           # 0-1; used for HITL escalation
        self.tool_calls = tool_calls or []
        self.tokens_used = tokens_used
        self.latency_ms = latency_ms
        self.error = error

    def to_dict(self) -> dict:
        return {
            "output": self.output,
            "confidence": self.confidence,
            "tool_calls": self.tool_calls,
            "tokens_used": self.tokens_used,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


class BaseAgent(ABC):
    """
    Abstract base for all agents. Subclasses must implement `invoke`.

    Provides:
    - LLM instantiation (Anthropic or OpenAI, keyed by model name)
    - Timing / token counting hooks
    - Structured logging
    """

    def __init__(self, name: str, model: str):
        self.name = name
        self.model = model
        self.llm = _build_llm(model)
        self._logger = logging.getLogger(f"agent.{name}")

    @abstractmethod
    def invoke(self, inputs: dict) -> AgentResult:
        """Execute the agent's core logic and return a structured result."""
        ...

    def timed_invoke(self, inputs: dict) -> AgentResult:
        """
        Wrap invoke() with wall-clock timing, OTel span, and in-process
        span recording for the Phase 4 trace explorer.
        """
        t0 = time.perf_counter()

        # ── OTel span (no-ops gracefully when OTLP endpoint is unreachable) ──
        try:
            from orchestrator.observability.tracer import tracer
            span_ctx = tracer.start_as_current_span(
                f"{self.name}.invoke",
                attributes={"task.id": inputs.get("task_id", ""), "agent.name": self.name},
            )
            span_ctx.__enter__()
        except Exception:
            span_ctx = None

        try:
            result = self.invoke(inputs)
        except Exception:
            if span_ctx:
                try:
                    span_ctx.__exit__(None, None, None)
                except Exception:
                    pass
            raise

        result.latency_ms = (time.perf_counter() - t0) * 1000

        # ── Close OTel span ───────────────────────────────────────────────────
        if span_ctx:
            try:
                from opentelemetry.trace import get_current_span
                sp = get_current_span()
                sp.set_attribute("agent.confidence", result.confidence)
                sp.set_attribute("agent.tokens", result.tokens_used)
                span_ctx.__exit__(None, None, None)
            except Exception:
                pass

        # ── In-process span (always works, powers the trace explorer UI) ─────
        try:
            from orchestrator.observability.snapshot_store import save_span
            save_span(inputs.get("task_id", ""), {
                "agent":       self.name,
                "model":       self.model,
                "latency_ms":  result.latency_ms,
                "tokens_used": result.tokens_used,
                "confidence":  result.confidence,
                "error":       result.error,
            })
        except Exception:
            pass

        # ── Cost recording ────────────────────────────────────────────────────
        try:
            from orchestrator.observability.cost_tracker import CostTracker
            CostTracker().record(
                task_id=inputs.get("task_id", ""),
                agent_name=self.name,
                model=self.model,
                tokens=result.tokens_used,
            )
        except Exception:
            pass

        self._logger.info(
            "agent=%s latency=%.0fms tokens=%d confidence=%.2f",
            self.name, result.latency_ms, result.tokens_used, result.confidence,
        )
        return result
