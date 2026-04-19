"""
GL-001 ThermalCommand Orchestrator - Prometheus Metrics Module

This module provides Prometheus metrics collection and export for the
ThermalCommand Orchestrator, enabling comprehensive observability.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import logging
import time

logger = logging.getLogger(__name__)


# =============================================================================
# METRIC TYPES
# =============================================================================

class Counter:
    """Prometheus-style counter metric."""

    def __init__(self, name: str, description: str, labels: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.labels = labels or []
        self._values: Dict[tuple, float] = {}

    def inc(self, amount: float = 1.0, **label_values) -> None:
        """Increment counter."""
        key = tuple(label_values.get(l, "") for l in self.labels)
        self._values[key] = self._values.get(key, 0) + amount

    def get(self, **label_values) -> float:
        """Get counter value."""
        key = tuple(label_values.get(l, "") for l in self.labels)
        return self._values.get(key, 0)

    def export(self) -> str:
        """Export in Prometheus format."""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} counter",
        ]
        for key, value in self._values.items():
            if self.labels:
                label_str = ",".join(
                    f'{l}="{v}"' for l, v in zip(self.labels, key)
                )
                lines.append(f"{self.name}{{{label_str}}} {value}")
            else:
                lines.append(f"{self.name} {value}")
        return "\n".join(lines)


class Gauge:
    """Prometheus-style gauge metric."""

    def __init__(self, name: str, description: str, labels: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.labels = labels or []
        self._values: Dict[tuple, float] = {}

    def set(self, value: float, **label_values) -> None:
        """Set gauge value."""
        key = tuple(label_values.get(l, "") for l in self.labels)
        self._values[key] = value

    def inc(self, amount: float = 1.0, **label_values) -> None:
        """Increment gauge."""
        key = tuple(label_values.get(l, "") for l in self.labels)
        self._values[key] = self._values.get(key, 0) + amount

    def dec(self, amount: float = 1.0, **label_values) -> None:
        """Decrement gauge."""
        key = tuple(label_values.get(l, "") for l in self.labels)
        self._values[key] = self._values.get(key, 0) - amount

    def get(self, **label_values) -> float:
        """Get gauge value."""
        key = tuple(label_values.get(l, "") for l in self.labels)
        return self._values.get(key, 0)

    def export(self) -> str:
        """Export in Prometheus format."""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} gauge",
        ]
        for key, value in self._values.items():
            if self.labels:
                label_str = ",".join(
                    f'{l}="{v}"' for l, v in zip(self.labels, key)
                )
                lines.append(f"{self.name}{{{label_str}}} {value}")
            else:
                lines.append(f"{self.name} {value}")
        return "\n".join(lines)


class Histogram:
    """Prometheus-style histogram metric."""

    def __init__(
        self,
        name: str,
        description: str,
        buckets: Optional[List[float]] = None,
        labels: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        self.labels = labels or []
        self._values: Dict[tuple, List[float]] = {}

    def observe(self, value: float, **label_values) -> None:
        """Record an observation."""
        key = tuple(label_values.get(l, "") for l in self.labels)
        if key not in self._values:
            self._values[key] = []
        self._values[key].append(value)

    def export(self) -> str:
        """Export in Prometheus format."""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} histogram",
        ]

        for key, observations in self._values.items():
            label_prefix = ""
            if self.labels:
                label_prefix = ",".join(
                    f'{l}="{v}"' for l, v in zip(self.labels, key)
                ) + ","

            # Calculate bucket counts
            for bucket in self.buckets:
                count = sum(1 for o in observations if o <= bucket)
                lines.append(
                    f'{self.name}_bucket{{{label_prefix}le="{bucket}"}} {count}'
                )
            lines.append(
                f'{self.name}_bucket{{{label_prefix}le="+Inf"}} {len(observations)}'
            )

            # Sum and count
            lines.append(
                f"{self.name}_sum{{{label_prefix[:-1]}}} {sum(observations)}"
            )
            lines.append(
                f"{self.name}_count{{{label_prefix[:-1]}}} {len(observations)}"
            )

        return "\n".join(lines)


class Summary:
    """Prometheus-style summary metric."""

    def __init__(
        self,
        name: str,
        description: str,
        quantiles: Optional[List[float]] = None,
        labels: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.quantiles = quantiles or [0.5, 0.9, 0.99]
        self.labels = labels or []
        self._values: Dict[tuple, List[float]] = {}

    def observe(self, value: float, **label_values) -> None:
        """Record an observation."""
        key = tuple(label_values.get(l, "") for l in self.labels)
        if key not in self._values:
            self._values[key] = []
        self._values[key].append(value)

    def export(self) -> str:
        """Export in Prometheus format."""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} summary",
        ]

        for key, observations in self._values.items():
            if not observations:
                continue

            label_prefix = ""
            if self.labels:
                label_prefix = ",".join(
                    f'{l}="{v}"' for l, v in zip(self.labels, key)
                ) + ","

            sorted_obs = sorted(observations)
            n = len(sorted_obs)

            for quantile in self.quantiles:
                idx = int(quantile * n)
                idx = min(idx, n - 1)
                value = sorted_obs[idx]
                lines.append(
                    f'{self.name}{{quantile="{quantile}",{label_prefix[:-1]}}} {value}'
                )

            lines.append(
                f"{self.name}_sum{{{label_prefix[:-1]}}} {sum(observations)}"
            )
            lines.append(
                f"{self.name}_count{{{label_prefix[:-1]}}} {len(observations)}"
            )

        return "\n".join(lines)


# =============================================================================
# METRICS COLLECTOR
# =============================================================================

class OrchestratorMetrics:
    """
    Prometheus metrics collector for ThermalCommand Orchestrator.

    Collects and exports metrics for monitoring and observability.
    """

    def __init__(self, prefix: str = "greenlang_thermal_command"):
        """
        Initialize the metrics collector.

        Args:
            prefix: Metric name prefix
        """
        self.prefix = prefix

        # Counters
        self.workflows_total = Counter(
            f"{prefix}_workflows_total",
            "Total number of workflows executed",
            labels=["status", "type"],
        )
        self.tasks_total = Counter(
            f"{prefix}_tasks_total",
            "Total number of tasks executed",
            labels=["status", "agent_type"],
        )
        self.safety_events_total = Counter(
            f"{prefix}_safety_events_total",
            "Total number of safety events",
            labels=["severity", "type"],
        )
        self.api_requests_total = Counter(
            f"{prefix}_api_requests_total",
            "Total API requests",
            labels=["method", "endpoint", "status"],
        )

        # Gauges
        self.registered_agents = Gauge(
            f"{prefix}_registered_agents",
            "Number of registered agents",
            labels=["type"],
        )
        self.active_workflows = Gauge(
            f"{prefix}_active_workflows",
            "Number of active workflows",
        )
        self.active_alarms = Gauge(
            f"{prefix}_active_alarms",
            "Number of active alarms",
            labels=["severity"],
        )
        self.safety_state = Gauge(
            f"{prefix}_safety_state",
            "Safety system state (0=normal, 1=warning, 2=emergency)",
        )
        self.agent_health = Gauge(
            f"{prefix}_agent_health",
            "Agent health status (0=offline, 1=unhealthy, 2=degraded, 3=healthy)",
            labels=["agent_id", "agent_type"],
        )

        # Histograms
        self.workflow_duration = Histogram(
            f"{prefix}_workflow_duration_seconds",
            "Workflow execution duration",
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0],
            labels=["type"],
        )
        self.task_duration = Histogram(
            f"{prefix}_task_duration_seconds",
            "Task execution duration",
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0],
            labels=["agent_type"],
        )
        self.api_latency = Histogram(
            f"{prefix}_api_latency_seconds",
            "API request latency",
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
            labels=["endpoint"],
        )

        # Summaries
        self.calculation_accuracy = Summary(
            f"{prefix}_calculation_accuracy",
            "Calculation accuracy (vs. reference)",
            quantiles=[0.5, 0.9, 0.99],
            labels=["calculation_type"],
        )

        logger.info(f"OrchestratorMetrics initialized with prefix: {prefix}")

    def record_workflow_completed(
        self,
        workflow_type: str,
        status: str,
        duration_seconds: float,
    ) -> None:
        """Record a completed workflow."""
        self.workflows_total.inc(status=status, type=workflow_type)
        self.workflow_duration.observe(duration_seconds, type=workflow_type)

    def record_task_completed(
        self,
        agent_type: str,
        status: str,
        duration_seconds: float,
    ) -> None:
        """Record a completed task."""
        self.tasks_total.inc(status=status, agent_type=agent_type)
        self.task_duration.observe(duration_seconds, agent_type=agent_type)

    def record_safety_event(self, severity: str, event_type: str) -> None:
        """Record a safety event."""
        self.safety_events_total.inc(severity=severity, type=event_type)

    def record_api_request(
        self,
        method: str,
        endpoint: str,
        status: str,
        latency_seconds: float,
    ) -> None:
        """Record an API request."""
        self.api_requests_total.inc(method=method, endpoint=endpoint, status=status)
        self.api_latency.observe(latency_seconds, endpoint=endpoint)

    def update_agent_count(self, agent_type: str, count: int) -> None:
        """Update registered agent count."""
        self.registered_agents.set(count, type=agent_type)

    def update_active_workflows(self, count: int) -> None:
        """Update active workflow count."""
        self.active_workflows.set(count)

    def update_active_alarms(self, severity: str, count: int) -> None:
        """Update active alarm count."""
        self.active_alarms.set(count, severity=severity)

    def update_safety_state(self, state: int) -> None:
        """Update safety state (0=normal, 1=warning, 2=emergency)."""
        self.safety_state.set(state)

    def update_agent_health(
        self,
        agent_id: str,
        agent_type: str,
        health: int,
    ) -> None:
        """Update agent health (0=offline, 1=unhealthy, 2=degraded, 3=healthy)."""
        self.agent_health.set(health, agent_id=agent_id, agent_type=agent_type)

    def export(self) -> str:
        """Export all metrics in Prometheus format."""
        metrics = [
            self.workflows_total,
            self.tasks_total,
            self.safety_events_total,
            self.api_requests_total,
            self.registered_agents,
            self.active_workflows,
            self.active_alarms,
            self.safety_state,
            self.agent_health,
            self.workflow_duration,
            self.task_duration,
            self.api_latency,
            self.calculation_accuracy,
        ]

        return "\n\n".join(m.export() for m in metrics if m._values)


# =============================================================================
# METRICS HTTP HANDLER
# =============================================================================

class MetricsHTTPHandler:
    """
    HTTP handler for Prometheus metrics endpoint.

    In production, this would be integrated with an HTTP server.
    """

    def __init__(self, metrics: OrchestratorMetrics, port: int = 9090):
        """
        Initialize the metrics HTTP handler.

        Args:
            metrics: OrchestratorMetrics instance
            port: HTTP port for metrics endpoint
        """
        self.metrics = metrics
        self.port = port
        self._running = False

    async def handle_metrics(self) -> str:
        """Handle /metrics endpoint."""
        return self.metrics.export()

    async def start(self) -> None:
        """Start the metrics HTTP server."""
        # In production, this would start an actual HTTP server
        self._running = True
        logger.info(f"Metrics endpoint available at http://localhost:{self.port}/metrics")

    async def stop(self) -> None:
        """Stop the metrics HTTP server."""
        self._running = False
        logger.info("Metrics endpoint stopped")


# =============================================================================
# TIMING UTILITIES
# =============================================================================

class Timer:
    """Context manager for timing operations."""

    def __init__(self, metrics: OrchestratorMetrics, metric_type: str, **labels):
        """
        Initialize the timer.

        Args:
            metrics: OrchestratorMetrics instance
            metric_type: Type of metric to record
            **labels: Labels for the metric
        """
        self.metrics = metrics
        self.metric_type = metric_type
        self.labels = labels
        self._start_time: Optional[float] = None

    def __enter__(self) -> "Timer":
        """Start timing."""
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop timing and record."""
        if self._start_time is not None:
            duration = time.perf_counter() - self._start_time

            if self.metric_type == "workflow":
                self.metrics.workflow_duration.observe(
                    duration,
                    **self.labels
                )
            elif self.metric_type == "task":
                self.metrics.task_duration.observe(
                    duration,
                    **self.labels
                )
            elif self.metric_type == "api":
                self.metrics.api_latency.observe(
                    duration,
                    **self.labels
                )
