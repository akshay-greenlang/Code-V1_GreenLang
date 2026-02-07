# -*- coding: utf-8 -*-
"""
AgentMetricsCollector - Prometheus metrics for GreenLang Agent Factory.

Registers per-agent counters, histograms, and gauges, collects metric
points asynchronously, and provides query helpers for recent metrics.

Metrics registered:
    - agent_execution_total (Counter): Total executions per agent.
    - agent_execution_errors_total (Counter): Total errors per agent.
    - agent_execution_duration_seconds (Histogram): Execution latency.
    - agent_queue_wait_seconds (Histogram): Time in queue before execution.
    - agent_cost_usd_total (Counter): Cumulative cost in USD.
    - agent_queue_depth (Gauge): Current queue depth per agent.
    - agent_active_count (Gauge): Number of active agent instances.

Example:
    >>> collector = AgentMetricsCollector()
    >>> collector.record_execution("carbon-calc", duration_s=0.142, success=True)
    >>> collector.record_cost("carbon-calc", 0.0004)
    >>> print(collector.get_agent_summary("carbon-calc"))

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class MetricPoint:
    """Single metric observation.

    Attributes:
        metric_name: Prometheus-style metric name.
        value: Numeric observation value.
        labels: Label key-value pairs.
        timestamp: UTC ISO-8601 timestamp.
    """

    metric_name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "labels": self.labels,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Internal aggregation helpers
# ---------------------------------------------------------------------------

@dataclass
class _AgentAgg:
    """Mutable per-agent aggregation state."""

    execution_count: int = 0
    error_count: int = 0
    total_duration_s: float = 0.0
    total_cost_usd: float = 0.0
    total_queue_wait_s: float = 0.0
    queue_depth: int = 0
    active_instances: int = 0
    durations: list[float] = field(default_factory=list)

    @property
    def avg_duration_s(self) -> float:
        if self.execution_count == 0:
            return 0.0
        return self.total_duration_s / self.execution_count

    @property
    def success_rate(self) -> float:
        if self.execution_count == 0:
            return 0.0
        return (self.execution_count - self.error_count) / self.execution_count


# ---------------------------------------------------------------------------
# Prometheus metric handles
# ---------------------------------------------------------------------------

class _PrometheusMetrics:
    """Lazy wrappers around prometheus_client metric objects.

    All metric objects are created at first access so that the module can be
    imported even when prometheus_client is not installed.
    """

    _initialized: bool = False
    execution_total: Any = None
    error_total: Any = None
    duration_histogram: Any = None
    queue_wait_histogram: Any = None
    cost_total: Any = None
    queue_depth_gauge: Any = None
    active_gauge: Any = None

    @classmethod
    def ensure_initialized(cls) -> bool:
        """Create Prometheus metrics if the library is available.

        Returns:
            True if prometheus_client is available and metrics are registered.
        """
        if cls._initialized:
            return cls.execution_total is not None

        cls._initialized = True
        try:
            from prometheus_client import Counter, Histogram, Gauge

            cls.execution_total = Counter(
                "agent_execution_total",
                "Total agent executions",
                ["agent_key", "version"],
            )
            cls.error_total = Counter(
                "agent_execution_errors_total",
                "Total agent execution errors",
                ["agent_key", "version", "error_type"],
            )
            cls.duration_histogram = Histogram(
                "agent_execution_duration_seconds",
                "Agent execution duration in seconds",
                ["agent_key"],
                buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            )
            cls.queue_wait_histogram = Histogram(
                "agent_queue_wait_seconds",
                "Time spent waiting in the execution queue",
                ["agent_key"],
                buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0),
            )
            cls.cost_total = Counter(
                "agent_cost_usd_total",
                "Cumulative execution cost in USD",
                ["agent_key"],
            )
            cls.queue_depth_gauge = Gauge(
                "agent_queue_depth",
                "Current queue depth per agent",
                ["agent_key"],
            )
            cls.active_gauge = Gauge(
                "agent_active_count",
                "Number of active agent instances",
                ["agent_key"],
            )
            logger.info("Prometheus metrics registered successfully")
            return True
        except ImportError:
            logger.info("prometheus_client not installed; metrics are in-memory only")
            return False


# ---------------------------------------------------------------------------
# AgentMetricsCollector
# ---------------------------------------------------------------------------

class AgentMetricsCollector:
    """Collect and expose per-agent metrics for the Agent Factory.

    Works in two modes:
    1. **Prometheus mode**: If prometheus_client is installed, all
       observations are forwarded to registered Prometheus metrics.
    2. **In-memory mode**: Metrics are aggregated in-memory and
       queryable via get_agent_summary / get_all_summaries.

    Thread-safe for concurrent recording from multiple agents.

    Example:
        >>> collector = AgentMetricsCollector()
        >>> collector.record_execution("carbon-calc", duration_s=0.15)
    """

    def __init__(self) -> None:
        """Initialize the metrics collector."""
        self._lock = threading.Lock()
        self._agents: Dict[str, _AgentAgg] = defaultdict(_AgentAgg)
        self._points: List[MetricPoint] = []
        self._max_points = 10_000
        self._prom_available = _PrometheusMetrics.ensure_initialized()

    # ---- Recording methods -----------------------------------------------

    def record_execution(
        self,
        agent_key: str,
        *,
        duration_s: float = 0.0,
        success: bool = True,
        version: str = "",
        error_type: str = "",
    ) -> None:
        """Record a completed agent execution.

        Args:
            agent_key: Agent identifier.
            duration_s: Execution duration in seconds.
            success: Whether the execution succeeded.
            version: Agent version (for labeling).
            error_type: Error classification if not success.
        """
        with self._lock:
            agg = self._agents[agent_key]
            agg.execution_count += 1
            agg.total_duration_s += duration_s
            agg.durations.append(duration_s)
            # Keep bounded history
            if len(agg.durations) > 1000:
                agg.durations = agg.durations[-1000:]

            if not success:
                agg.error_count += 1

        # Prometheus metrics
        if self._prom_available:
            pm = _PrometheusMetrics
            if pm.execution_total is not None:
                pm.execution_total.labels(agent_key=agent_key, version=version).inc()
            if not success and pm.error_total is not None:
                pm.error_total.labels(
                    agent_key=agent_key, version=version, error_type=error_type
                ).inc()
            if pm.duration_histogram is not None:
                pm.duration_histogram.labels(agent_key=agent_key).observe(duration_s)

        self._append_point("agent_execution_total", 1, {"agent_key": agent_key})
        logger.debug("Recorded execution for %s (%.3fs, success=%s)", agent_key, duration_s, success)

    def record_queue_wait(self, agent_key: str, wait_s: float) -> None:
        """Record queue wait time for an agent task.

        Args:
            agent_key: Agent identifier.
            wait_s: Queue wait time in seconds.
        """
        with self._lock:
            self._agents[agent_key].total_queue_wait_s += wait_s

        if self._prom_available and _PrometheusMetrics.queue_wait_histogram is not None:
            _PrometheusMetrics.queue_wait_histogram.labels(agent_key=agent_key).observe(wait_s)

        self._append_point("agent_queue_wait_seconds", wait_s, {"agent_key": agent_key})

    def record_cost(self, agent_key: str, cost_usd: float) -> None:
        """Record execution cost for an agent.

        Args:
            agent_key: Agent identifier.
            cost_usd: Cost in USD.
        """
        with self._lock:
            self._agents[agent_key].total_cost_usd += cost_usd

        if self._prom_available and _PrometheusMetrics.cost_total is not None:
            _PrometheusMetrics.cost_total.labels(agent_key=agent_key).inc(cost_usd)

        self._append_point("agent_cost_usd_total", cost_usd, {"agent_key": agent_key})

    def set_queue_depth(self, agent_key: str, depth: int) -> None:
        """Set the current queue depth for an agent.

        Args:
            agent_key: Agent identifier.
            depth: Current queue depth.
        """
        with self._lock:
            self._agents[agent_key].queue_depth = depth

        if self._prom_available and _PrometheusMetrics.queue_depth_gauge is not None:
            _PrometheusMetrics.queue_depth_gauge.labels(agent_key=agent_key).set(depth)

    def set_active_instances(self, agent_key: str, count: int) -> None:
        """Set the number of active instances for an agent.

        Args:
            agent_key: Agent identifier.
            count: Number of active instances.
        """
        with self._lock:
            self._agents[agent_key].active_instances = count

        if self._prom_available and _PrometheusMetrics.active_gauge is not None:
            _PrometheusMetrics.active_gauge.labels(agent_key=agent_key).set(count)

    # ---- Query methods ---------------------------------------------------

    def get_agent_summary(self, agent_key: str) -> Dict[str, Any]:
        """Return a summary of metrics for a specific agent.

        Args:
            agent_key: Agent identifier.

        Returns:
            Summary dictionary.
        """
        with self._lock:
            agg = self._agents.get(agent_key)
            if agg is None:
                return {"agent_key": agent_key, "status": "no_data"}

            durations_sorted = sorted(agg.durations)
            p50 = self._percentile(durations_sorted, 50)
            p95 = self._percentile(durations_sorted, 95)
            p99 = self._percentile(durations_sorted, 99)

            return {
                "agent_key": agent_key,
                "execution_count": agg.execution_count,
                "error_count": agg.error_count,
                "success_rate": round(agg.success_rate * 100, 2),
                "avg_duration_s": round(agg.avg_duration_s, 4),
                "p50_duration_s": round(p50, 4),
                "p95_duration_s": round(p95, 4),
                "p99_duration_s": round(p99, 4),
                "total_cost_usd": round(agg.total_cost_usd, 4),
                "queue_depth": agg.queue_depth,
                "active_instances": agg.active_instances,
            }

    def get_all_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Return summaries for all tracked agents.

        Returns:
            Mapping of agent_key to summary dict.
        """
        with self._lock:
            keys = list(self._agents.keys())
        return {k: self.get_agent_summary(k) for k in keys}

    def get_recent_points(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Return the most recent metric points.

        Args:
            limit: Maximum number of points to return.

        Returns:
            List of serialized MetricPoint dictionaries.
        """
        with self._lock:
            return [p.to_dict() for p in self._points[-limit:]]

    # ---- Internal --------------------------------------------------------

    def _append_point(self, name: str, value: float, labels: Dict[str, str]) -> None:
        """Append a metric point to the internal buffer."""
        point = MetricPoint(metric_name=name, value=value, labels=labels)
        with self._lock:
            self._points.append(point)
            if len(self._points) > self._max_points:
                self._points = self._points[-self._max_points:]

    @staticmethod
    def _percentile(sorted_data: List[float], pct: float) -> float:
        """Compute a percentile from sorted data."""
        if not sorted_data:
            return 0.0
        idx = int(len(sorted_data) * pct / 100)
        idx = min(idx, len(sorted_data) - 1)
        return sorted_data[idx]


__all__ = ["AgentMetricsCollector", "MetricPoint"]
