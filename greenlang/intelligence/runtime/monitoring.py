# -*- coding: utf-8 -*-
"""
Intelligence Layer Monitoring and Metrics

Provides comprehensive monitoring for:
- LLM provider calls (success rate, latency, cost)
- Circuit breaker states (open/closed/half-open)
- Budget tracking (spent/remaining/alerts)
- Tool invocations (success/failure/latency)
- JSON retry attempts (success/failure rate)
- Context overflow incidents

Integrates with:
- Prometheus metrics (optional)
- CloudWatch (optional)
- Custom dashboards
- Alert systems

Architecture:
    Metrics Collection → Aggregation → Export → Visualization/Alerts
"""

from __future__ import annotations
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import json

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics collected"""

    COUNTER = "counter"  # Incremental count (requests, errors)
    GAUGE = "gauge"  # Current value (active requests, circuit state)
    HISTOGRAM = "histogram"  # Distribution (latency, token count)
    SUMMARY = "summary"  # Aggregated stats (avg, p50, p95, p99)


class AlertSeverity(str, Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """Single metric data point"""

    name: str
    type: MetricType
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "type": self.type.value,
            "value": self.value,
            "timestamp": self.timestamp,
            "labels": self.labels,
        }


@dataclass
class Alert:
    """Alert notification"""

    name: str
    severity: AlertSeverity
    message: str
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "labels": self.labels,
            "resolved": self.resolved,
        }


class MetricsCollector:
    """
    Central metrics collection and aggregation

    Collects metrics from all intelligence layer components:
    - LLM providers
    - Circuit breakers
    - Tool registry
    - Budget tracking

    Provides:
    - Real-time metrics
    - Historical aggregation
    - Alert generation
    - Export to monitoring systems
    """

    def __init__(self, retention_seconds: int = 3600):
        """
        Initialize metrics collector

        Args:
            retention_seconds: How long to keep metrics in memory (default: 1 hour)
        """
        self.retention_seconds = retention_seconds

        # Metrics storage (keyed by metric name)
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Raw metrics for export
        self._metrics_history: deque = deque(maxlen=10000)

        # Alerts
        self._active_alerts: List[Alert] = []
        self._alert_history: deque = deque(maxlen=1000)

        # Alert thresholds
        self._alert_thresholds = {
            "budget_remaining_pct": 10.0,  # Alert when <10% budget remains
            "error_rate_pct": 5.0,  # Alert when error rate >5%
            "circuit_breaker_open_count": 1,  # Alert on any circuit breaker open
            "json_retry_failure_rate_pct": 10.0,  # Alert when >10% JSON retries fail
            "latency_p95_ms": 5000.0,  # Alert when p95 latency >5s
        }

        logger.info("MetricsCollector initialized")

    def increment_counter(
        self, name: str, value: float = 1.0, labels: Dict[str, str] = None
    ):
        """Increment a counter metric"""
        key = self._make_key(name, labels or {})
        self._counters[key] += value

        # Record metric point
        self._metrics_history.append(
            MetricPoint(
                name=name, type=MetricType.COUNTER, value=value, labels=labels or {}
            )
        )

    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric"""
        key = self._make_key(name, labels or {})
        self._gauges[key] = value

        # Record metric point
        self._metrics_history.append(
            MetricPoint(
                name=name, type=MetricType.GAUGE, value=value, labels=labels or {}
            )
        )

    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a histogram value"""
        key = self._make_key(name, labels or {})
        self._histograms[key].append((time.time(), value))

        # Record metric point
        self._metrics_history.append(
            MetricPoint(
                name=name, type=MetricType.HISTOGRAM, value=value, labels=labels or {}
            )
        )

    def _make_key(self, name: str, labels: Dict[str, str]) -> str:
        """Create unique key for metric with labels"""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def get_counter(self, name: str, labels: Dict[str, str] = None) -> float:
        """Get counter value"""
        key = self._make_key(name, labels or {})
        return self._counters.get(key, 0.0)

    def get_gauge(self, name: str, labels: Dict[str, str] = None) -> Optional[float]:
        """Get gauge value"""
        key = self._make_key(name, labels or {})
        return self._gauges.get(key)

    def get_histogram_stats(
        self, name: str, labels: Dict[str, str] = None
    ) -> Dict[str, float]:
        """Get histogram statistics (avg, p50, p95, p99)"""
        key = self._make_key(name, labels or {})
        values = [
            v
            for t, v in self._histograms.get(key, [])
            if time.time() - t < self.retention_seconds
        ]

        if not values:
            return {
                "count": 0,
                "avg": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        values_sorted = sorted(values)
        return {
            "count": len(values),
            "avg": sum(values) / len(values),
            "p50": values_sorted[int(len(values) * 0.50)],
            "p95": values_sorted[int(len(values) * 0.95)],
            "p99": values_sorted[int(len(values) * 0.99)],
            "min": values_sorted[0],
            "max": values_sorted[-1],
        }

    def create_alert(
        self,
        name: str,
        severity: AlertSeverity,
        message: str,
        labels: Dict[str, str] = None,
    ):
        """Create an alert"""
        alert = Alert(
            name=name, severity=severity, message=message, labels=labels or {}
        )

        # Check if alert already exists
        for existing in self._active_alerts:
            if existing.name == name and not existing.resolved:
                logger.debug(f"Alert {name} already active, not creating duplicate")
                return

        self._active_alerts.append(alert)
        self._alert_history.append(alert)

        logger.warning(f"Alert created: [{severity.value}] {name}: {message}")

    def resolve_alert(self, name: str):
        """Resolve an active alert"""
        for alert in self._active_alerts:
            if alert.name == name and not alert.resolved:
                alert.resolved = True
                logger.info(f"Alert resolved: {name}")

    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts"""
        return [a for a in self._active_alerts if not a.resolved]

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get complete metrics summary"""
        return {
            "counters": {k: v for k, v in self._counters.items()},
            "gauges": {k: v for k, v in self._gauges.items()},
            "histograms": {
                k: self.get_histogram_stats(k) for k in self._histograms.keys()
            },
            "active_alerts": [a.to_dict() for a in self.get_active_alerts()],
            "timestamp": time.time(),
        }

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []

        # Counters
        for key, value in self._counters.items():
            lines.append(f"# TYPE {key} counter")
            lines.append(f"{key} {value}")

        # Gauges
        for key, value in self._gauges.items():
            lines.append(f"# TYPE {key} gauge")
            lines.append(f"{key} {value}")

        # Histograms (as summaries)
        for key in self._histograms.keys():
            stats = self.get_histogram_stats(key)
            lines.append(f"# TYPE {key} summary")
            lines.append(f"{key}_count {stats['count']}")
            lines.append(f"{key}_sum {stats['avg'] * stats['count']}")
            lines.append(f"{key}{{quantile=\"0.5\"}} {stats['p50']}")
            lines.append(f"{key}{{quantile=\"0.95\"}} {stats['p95']}")
            lines.append(f"{key}{{quantile=\"0.99\"}} {stats['p99']}")

        return "\n".join(lines)

    def export_json(self) -> str:
        """Export metrics as JSON"""
        return json.dumps(self.get_metrics_summary(), indent=2)

    def check_alerts(self):
        """Check all alert conditions and create alerts if thresholds exceeded"""
        # Budget alerts
        budget_pct = self.get_gauge("intelligence_budget_remaining_pct")
        if (
            budget_pct is not None
            and budget_pct < self._alert_thresholds["budget_remaining_pct"]
        ):
            self.create_alert(
                name="low_budget",
                severity=AlertSeverity.WARNING,
                message=f"Budget remaining: {budget_pct:.1f}% (threshold: {self._alert_thresholds['budget_remaining_pct']}%)",
            )

        # Error rate alerts
        total_requests = self.get_counter("intelligence_requests_total")
        failed_requests = self.get_counter("intelligence_requests_failed")
        if total_requests > 0:
            error_rate = (failed_requests / total_requests) * 100
            if error_rate > self._alert_thresholds["error_rate_pct"]:
                self.create_alert(
                    name="high_error_rate",
                    severity=AlertSeverity.ERROR,
                    message=f"Error rate: {error_rate:.1f}% (threshold: {self._alert_thresholds['error_rate_pct']}%)",
                )

        # Circuit breaker alerts
        open_circuits = self.get_gauge("intelligence_circuit_breaker_open_count")
        if open_circuits and open_circuits > 0:
            self.create_alert(
                name="circuit_breaker_open",
                severity=AlertSeverity.CRITICAL,
                message=f"{int(open_circuits)} circuit breaker(s) open - provider(s) failing",
            )

        # Latency alerts
        latency_stats = self.get_histogram_stats("intelligence_request_duration_ms")
        if latency_stats["p95"] > self._alert_thresholds["latency_p95_ms"]:
            self.create_alert(
                name="high_latency",
                severity=AlertSeverity.WARNING,
                message=f"p95 latency: {latency_stats['p95']:.0f}ms (threshold: {self._alert_thresholds['latency_p95_ms']}ms)",
            )

    def reset(self):
        """Reset all metrics (useful for testing)"""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()
        self._metrics_history.clear()
        self._active_alerts.clear()
        logger.info("Metrics collector reset")


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector (singleton)"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


# Convenience functions for common metrics


def track_provider_request(
    provider: str,
    model: str,
    success: bool,
    duration_ms: float,
    cost_usd: float,
    tokens: int,
):
    """Track LLM provider request"""
    collector = get_metrics_collector()
    labels = {"provider": provider, "model": model}

    # Increment counters
    collector.increment_counter("intelligence_requests_total", 1.0, labels)
    if not success:
        collector.increment_counter("intelligence_requests_failed", 1.0, labels)

    # Record latency
    collector.record_histogram("intelligence_request_duration_ms", duration_ms, labels)

    # Record cost
    collector.record_histogram("intelligence_cost_usd", cost_usd, labels)

    # Record tokens
    collector.record_histogram("intelligence_tokens_used", tokens, labels)


def track_circuit_breaker_state(provider: str, state: str):
    """Track circuit breaker state"""
    collector = get_metrics_collector()
    labels = {"provider": provider}

    # Set gauge for state
    state_value = {"closed": 0, "half_open": 1, "open": 2}.get(state, 0)
    collector.set_gauge("intelligence_circuit_breaker_state", state_value, labels)

    # Count open circuits
    open_count = sum(
        1
        for k, v in collector._gauges.items()
        if "circuit_breaker_state" in k and v == 2
    )
    collector.set_gauge("intelligence_circuit_breaker_open_count", open_count)


def track_tool_invocation(tool_name: str, success: bool, duration_ms: float):
    """Track tool invocation"""
    collector = get_metrics_collector()
    labels = {"tool": tool_name}

    collector.increment_counter("intelligence_tool_invocations_total", 1.0, labels)
    if not success:
        collector.increment_counter("intelligence_tool_invocations_failed", 1.0, labels)

    collector.record_histogram("intelligence_tool_duration_ms", duration_ms, labels)


def track_json_retry(provider: str, attempts: int, success: bool):
    """Track JSON retry attempts"""
    collector = get_metrics_collector()
    labels = {"provider": provider}

    collector.increment_counter("intelligence_json_retries_total", 1.0, labels)
    collector.record_histogram("intelligence_json_retry_attempts", attempts, labels)

    if not success:
        collector.increment_counter("intelligence_json_retries_failed", 1.0, labels)


def track_budget_usage(spent_usd: float, remaining_usd: float, max_usd: float):
    """Track budget usage"""
    collector = get_metrics_collector()

    collector.set_gauge("intelligence_budget_spent_usd", spent_usd)
    collector.set_gauge("intelligence_budget_remaining_usd", remaining_usd)
    collector.set_gauge("intelligence_budget_max_usd", max_usd)

    remaining_pct = (remaining_usd / max_usd) * 100 if max_usd > 0 else 0
    collector.set_gauge("intelligence_budget_remaining_pct", remaining_pct)


def track_context_overflow(provider: str, model: str, messages_truncated: int):
    """Track context overflow incident"""
    collector = get_metrics_collector()
    labels = {"provider": provider, "model": model}

    collector.increment_counter("intelligence_context_overflows_total", 1.0, labels)
    collector.record_histogram(
        "intelligence_messages_truncated", messages_truncated, labels
    )
