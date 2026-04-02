"""
Observability module — metrics, tracing, and monitoring.

Provides:
- PerformanceTracker: per-step latency, token usage, episode metrics
- MetricsCollector: aggregated stats across all sessions
- LLMTracer: hooks for LangSmith/LangFuse integration (optional)
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class StepMetric:
    """Metrics for a single environment step."""
    step_number: int
    action_type: str
    latency_ms: float
    reward: float
    done: bool
    error: str | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class EpisodeMetric:
    """Aggregated metrics for a complete episode."""
    session_id: str
    task_id: str
    total_steps: int
    total_reward: float
    final_score: float
    total_latency_ms: float
    avg_step_latency_ms: float
    errors: int
    hallucinations_caught: int = 0
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time if self.end_time > self.start_time else 0


class PerformanceTracker:
    """
    Track performance metrics for a single episode.

    Usage:
        tracker = PerformanceTracker("session-123", "invoice_match")
        tracker.record_step(1, "match_invoice", 3.2, 0.05, False)
        ...
        metrics = tracker.get_episode_metrics()
    """

    def __init__(self, session_id: str, task_id: str):
        self.session_id = session_id
        self.task_id = task_id
        self.steps: list[StepMetric] = []
        self.start_time = time.time()
        self._final_score = 0.0

    def record_step(
        self,
        step_number: int,
        action_type: str,
        latency_ms: float,
        reward: float,
        done: bool,
        error: str | None = None,
    ) -> None:
        """Record metrics for a single step."""
        metric = StepMetric(
            step_number=step_number,
            action_type=action_type,
            latency_ms=latency_ms,
            reward=reward,
            done=done,
            error=error,
        )
        self.steps.append(metric)

        logger.info(
            "step_metric",
            session_id=self.session_id,
            step=step_number,
            action=action_type,
            latency_ms=round(latency_ms, 2),
            reward=reward,
            done=done,
            error=error,
        )

    def set_final_score(self, score: float) -> None:
        self._final_score = score

    def get_episode_metrics(self) -> EpisodeMetric:
        """Compute aggregated episode metrics."""
        total_latency = sum(s.latency_ms for s in self.steps)
        avg_latency = total_latency / max(len(self.steps), 1)
        total_reward = sum(s.reward for s in self.steps)
        errors = sum(1 for s in self.steps if s.error)

        return EpisodeMetric(
            session_id=self.session_id,
            task_id=self.task_id,
            total_steps=len(self.steps),
            total_reward=round(total_reward, 4),
            final_score=self._final_score,
            total_latency_ms=round(total_latency, 2),
            avg_step_latency_ms=round(avg_latency, 2),
            errors=errors,
            start_time=self.start_time,
            end_time=time.time(),
        )


class MetricsCollector:
    """
    Global metrics collector — aggregates stats across all sessions.

    Thread-safe for concurrent sessions. Provides summary stats for
    the `/metrics` endpoint and monitoring dashboards.
    """

    def __init__(self):
        self.episodes: list[EpisodeMetric] = []
        self.task_scores: dict[str, list[float]] = defaultdict(list)
        self.total_steps: int = 0
        self.total_errors: int = 0
        self._action_counts: dict[str, int] = defaultdict(int)
        self._latencies: list[float] = []

    def record_episode(self, metric: EpisodeMetric) -> None:
        """Record a completed episode."""
        self.episodes.append(metric)
        self.task_scores[metric.task_id].append(metric.final_score)
        self.total_steps += metric.total_steps
        self.total_errors += metric.errors

        logger.info(
            "episode_complete",
            session_id=metric.session_id,
            task=metric.task_id,
            score=metric.final_score,
            steps=metric.total_steps,
            duration_s=round(metric.duration_seconds, 2),
            avg_latency_ms=metric.avg_step_latency_ms,
        )

    def record_action(self, action_type: str, latency_ms: float) -> None:
        """Record an individual action for global stats."""
        self._action_counts[action_type] += 1
        self._latencies.append(latency_ms)

    def get_summary(self) -> dict[str, Any]:
        """Get aggregated metrics summary."""
        avg_scores = {
            task: round(sum(scores) / max(len(scores), 1), 4)
            for task, scores in self.task_scores.items()
        }

        p50 = sorted(self._latencies)[len(self._latencies) // 2] if self._latencies else 0
        p99 = sorted(self._latencies)[int(len(self._latencies) * 0.99)] if self._latencies else 0

        return {
            "total_episodes": len(self.episodes),
            "total_steps": self.total_steps,
            "total_errors": self.total_errors,
            "avg_scores_by_task": avg_scores,
            "action_counts": dict(self._action_counts),
            "latency_p50_ms": round(p50, 2),
            "latency_p99_ms": round(p99, 2),
        }


class LLMTracer:
    """
    Optional LLM tracing integration.

    Hooks for LangSmith or LangFuse. Records:
    - Input/output tokens per call
    - LLM latency
    - Hallucination detection rate
    - Cost estimation
    """

    def __init__(self, service: str = "none"):
        self.service = service
        self.traces: list[dict] = []
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cost_usd: float = 0.0

    def trace_llm_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        success: bool = True,
    ) -> None:
        """Record an LLM API call."""
        # Cost estimation (approximate)
        cost = self._estimate_cost(model, input_tokens, output_tokens)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost

        trace = {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
            "success": success,
            "cost_usd": round(cost, 6),
            "timestamp": time.time(),
        }
        self.traces.append(trace)

        logger.info("llm_trace", **trace)

    def _estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Rough cost estimation per model."""
        rates = {
            "gpt-4": (0.03, 0.06),            # per 1K tokens
            "gpt-4-turbo": (0.01, 0.03),
            "gpt-3.5-turbo": (0.0005, 0.0015),
            "gemini-pro": (0.00025, 0.0005),
        }
        in_rate, out_rate = rates.get(model, (0.001, 0.002))
        return (input_tokens * in_rate + output_tokens * out_rate) / 1000

    def get_summary(self) -> dict:
        """Get LLM usage summary."""
        return {
            "total_calls": len(self.traces),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "avg_latency_ms": (
                round(sum(t["latency_ms"] for t in self.traces) / max(len(self.traces), 1), 2)
            ),
        }


# Global singleton
_metrics_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector
