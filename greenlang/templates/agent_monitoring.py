# -*- coding: utf-8 -*-
"""Universal Operational Monitoring Mixin for GreenLang AI Agents.

This module provides a production-ready monitoring mixin that can be added
to any GreenLang AI agent to provide instant operational excellence.

Features:
- Performance tracking (latency, cost, token usage)
- Health checks (liveness, readiness)
- Metrics collection (Prometheus-compatible)
- Structured logging (JSON)
- Error tracking
- Alert generation

Usage:
    class MyAgent(OperationalMonitoringMixin, BaseAgent):
        def __init__(self):
            super().__init__()
            self.setup_monitoring(agent_name="my_agent")

Author: GreenLang Framework Team
Date: October 2025
Version: 1.0.0
"""

from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from contextlib import contextmanager
import logging
import json
import time
import traceback
import uuid
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import deque, defaultdict
import threading
from greenlang.determinism import deterministic_uuid, DeterministicClock


class HealthStatus(Enum):
    """Agent health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetrics:
    """Performance metrics for an agent execution."""
    execution_id: str
    agent_name: str
    timestamp: str
    duration_ms: float
    cost_usd: float
    tokens_used: int
    tool_calls: int
    ai_calls: int
    success: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    cache_hit: bool = False
    input_size_bytes: int = 0
    output_size_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class HealthCheckResult:
    """Health check result."""
    status: HealthStatus
    timestamp: str
    checks: Dict[str, bool]
    metrics: Dict[str, float]
    last_error: Optional[str] = None
    uptime_seconds: float = 0
    degradation_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp,
            "checks": self.checks,
            "metrics": self.metrics,
            "last_error": self.last_error,
            "uptime_seconds": self.uptime_seconds,
            "degradation_reasons": self.degradation_reasons
        }


@dataclass
class Alert:
    """Operational alert."""
    alert_id: str
    severity: AlertSeverity
    timestamp: str
    message: str
    context: Dict[str, Any]
    agent_name: str
    resolved: bool = False
    resolved_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "severity": self.severity.value,
            "timestamp": self.timestamp,
            "message": self.message,
            "context": self.context,
            "agent_name": self.agent_name,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at
        }


class MetricsCollector:
    """Collects and aggregates metrics for Prometheus export."""

    def __init__(self):
        self._metrics: Dict[str, List[float]] = defaultdict(list)
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()

    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a metric value."""
        with self._lock:
            metric_key = self._build_metric_key(name, labels)
            self._metrics[metric_key].append(value)

    def increment_counter(self, name: str, labels: Dict[str, str] = None):
        """Increment a counter."""
        with self._lock:
            metric_key = self._build_metric_key(name, labels)
            self._counters[metric_key] += 1

    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge value."""
        with self._lock:
            metric_key = self._build_metric_key(name, labels)
            self._gauges[metric_key] = value

    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        with self._lock:
            return {
                "metrics": dict(self._metrics),
                "counters": dict(self._counters),
                "gauges": dict(self._gauges)
            }

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        with self._lock:
            # Export counters
            for name, value in self._counters.items():
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name} {value}")

            # Export gauges
            for name, value in self._gauges.items():
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {value}")

            # Export histograms (from metrics)
            for name, values in self._metrics.items():
                if values:
                    lines.append(f"# TYPE {name} histogram")
                    lines.append(f"{name}_sum {sum(values)}")
                    lines.append(f"{name}_count {len(values)}")
                    lines.append(f"{name}_avg {sum(values) / len(values)}")

        return "\n".join(lines)

    @staticmethod
    def _build_metric_key(name: str, labels: Dict[str, str] = None) -> str:
        """Build metric key with labels."""
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"


class OperationalMonitoringMixin:
    """Mixin to add operational monitoring to any GreenLang agent.

    Provides:
    - Performance tracking
    - Health checks
    - Metrics collection
    - Alert generation
    - Structured logging

    Integration:
        class MyAgent(OperationalMonitoringMixin, BaseAgent):
            def __init__(self):
                super().__init__()
                self.setup_monitoring(agent_name="my_agent")

            def execute(self, input_data):
                with self.track_execution(input_data):
                    result = self._do_work(input_data)
                    return result
    """

    def setup_monitoring(
        self,
        agent_name: str,
        enable_metrics: bool = True,
        enable_health_checks: bool = True,
        enable_alerting: bool = True,
        max_history: int = 1000,
        alert_callback: Optional[Callable[[Alert], None]] = None,
    ) -> None:
        """Setup operational monitoring.

        Args:
            agent_name: Name of the agent
            enable_metrics: Enable performance metrics
            enable_health_checks: Enable health monitoring
            enable_alerting: Enable alert generation
            max_history: Maximum number of metrics to keep in history
            alert_callback: Optional callback for alert notifications
        """
        self._monitoring_agent_name = agent_name
        self._monitoring_enabled = True
        self._metrics_enabled = enable_metrics
        self._health_checks_enabled = enable_health_checks
        self._alerting_enabled = enable_alerting
        self._max_history = max_history
        self._alert_callback = alert_callback

        # Initialize metrics storage
        self._execution_history: deque = deque(maxlen=max_history)
        self._metrics_collector = MetricsCollector()
        self._alerts: List[Alert] = []
        self._start_time = DeterministicClock.utcnow()
        self._last_error: Optional[str] = None
        self._error_count = 0
        self._success_count = 0
        self._total_executions = 0

        # Performance thresholds for alerts
        self._latency_threshold_ms = 5000  # 5 seconds
        self._error_rate_threshold = 0.1  # 10%
        self._cost_threshold_usd = 1.0  # $1 per execution

        # Structured logger
        self._monitoring_logger = logging.getLogger(f"{agent_name}.monitoring")

        self._log_structured("info", "Monitoring initialized", {
            "agent": agent_name,
            "metrics_enabled": enable_metrics,
            "health_checks_enabled": enable_health_checks,
            "alerting_enabled": enable_alerting
        })

    @contextmanager
    def track_execution(
        self,
        input_data: Dict[str, Any],
        track_tokens: bool = True,
        track_cost: bool = True
    ):
        """Context manager to track execution metrics.

        Usage:
            with self.track_execution(input_data) as tracker:
                result = self._do_work(input_data)
                tracker.set_tokens(result.get('tokens_used', 0))
                tracker.set_cost(result.get('cost', 0.0))

        Args:
            input_data: Input data being processed
            track_tokens: Whether to track token usage
            track_cost: Whether to track cost

        Yields:
            ExecutionTracker: Tracker object to update metrics
        """
        execution_id = str(deterministic_uuid(__name__, str(DeterministicClock.now())))
        start_time = time.time()

        class ExecutionTracker:
            """Helper to track execution details."""
            def __init__(self):
                self.tokens_used = 0
                self.cost_usd = 0.0
                self.tool_calls = 0
                self.ai_calls = 0
                self.cache_hit = False
                self.error_type = None
                self.error_message = None

            def set_tokens(self, tokens: int):
                self.tokens_used = tokens

            def set_cost(self, cost: float):
                self.cost_usd = cost

            def increment_tool_calls(self, count: int = 1):
                self.tool_calls += count

            def increment_ai_calls(self, count: int = 1):
                self.ai_calls += count

            def set_cache_hit(self, hit: bool):
                self.cache_hit = hit

            def set_error(self, error_type: str, error_message: str):
                self.error_type = error_type
                self.error_message = error_message

        tracker = ExecutionTracker()
        success = True

        try:
            yield tracker

        except Exception as e:
            success = False
            tracker.error_type = type(e).__name__
            tracker.error_message = str(e)
            self._last_error = f"{tracker.error_type}: {tracker.error_message}"
            self._error_count += 1

            self._log_structured("error", "Execution failed", {
                "execution_id": execution_id,
                "error_type": tracker.error_type,
                "error_message": tracker.error_message,
                "traceback": traceback.format_exc()
            })

            if self._alerting_enabled:
                self._generate_alert(
                    AlertSeverity.ERROR,
                    f"Execution failed: {tracker.error_type}",
                    {
                        "execution_id": execution_id,
                        "error_message": tracker.error_message
                    }
                )

            raise

        finally:
            duration_ms = (time.time() - start_time) * 1000

            if success:
                self._success_count += 1

            self._total_executions += 1

            # Calculate input/output sizes
            input_size = len(json.dumps(input_data).encode('utf-8'))

            # Create metrics
            metrics = PerformanceMetrics(
                execution_id=execution_id,
                agent_name=self._monitoring_agent_name,
                timestamp=DeterministicClock.utcnow().isoformat(),
                duration_ms=duration_ms,
                cost_usd=tracker.cost_usd,
                tokens_used=tracker.tokens_used,
                tool_calls=tracker.tool_calls,
                ai_calls=tracker.ai_calls,
                success=success,
                error_type=tracker.error_type,
                error_message=tracker.error_message,
                cache_hit=tracker.cache_hit,
                input_size_bytes=input_size,
                output_size_bytes=0  # Can be set by caller
            )

            # Store metrics
            if self._metrics_enabled:
                self._execution_history.append(metrics)

                # Emit Prometheus metrics
                self._emit_metric("execution_duration_ms", duration_ms)
                self._emit_metric("execution_cost_usd", tracker.cost_usd)
                self._emit_metric("execution_tokens", tracker.tokens_used)
                self._metrics_collector.increment_counter(
                    "executions_total",
                    {"agent": self._monitoring_agent_name, "success": str(success)}
                )

                if tracker.cache_hit:
                    self._metrics_collector.increment_counter(
                        "cache_hits_total",
                        {"agent": self._monitoring_agent_name}
                    )

                # Check thresholds and generate alerts
                if self._alerting_enabled:
                    if duration_ms > self._latency_threshold_ms:
                        self._generate_alert(
                            AlertSeverity.WARNING,
                            f"High latency: {duration_ms:.0f}ms",
                            {"execution_id": execution_id, "duration_ms": duration_ms}
                        )

                    if tracker.cost_usd > self._cost_threshold_usd:
                        self._generate_alert(
                            AlertSeverity.WARNING,
                            f"High cost: ${tracker.cost_usd:.2f}",
                            {"execution_id": execution_id, "cost_usd": tracker.cost_usd}
                        )

                    # Check error rate
                    error_rate = self._error_count / self._total_executions
                    if error_rate > self._error_rate_threshold:
                        self._generate_alert(
                            AlertSeverity.CRITICAL,
                            f"High error rate: {error_rate:.1%}",
                            {
                                "error_rate": error_rate,
                                "error_count": self._error_count,
                                "total_executions": self._total_executions
                            }
                        )

                self._log_structured("info", "Execution completed", {
                    "execution_id": execution_id,
                    "duration_ms": duration_ms,
                    "success": success,
                    "cost_usd": tracker.cost_usd,
                    "tokens_used": tracker.tokens_used
                })

    def health_check(self) -> HealthCheckResult:
        """Perform comprehensive health check.

        Returns:
            HealthCheckResult with status and details
        """
        checks = {}
        metrics = {}
        degradation_reasons = []

        # Check if monitoring is enabled
        checks["monitoring_enabled"] = self._monitoring_enabled

        # Check recent errors
        recent_executions = list(self._execution_history)[-10:] if self._execution_history else []
        recent_errors = sum(1 for m in recent_executions if not m.success)
        checks["recent_errors_acceptable"] = recent_errors < 3

        if recent_errors >= 3:
            degradation_reasons.append(f"High recent error count: {recent_errors}/10")

        # Check last error age
        if self._last_error:
            checks["has_recent_error"] = True
            degradation_reasons.append(f"Recent error: {self._last_error}")
        else:
            checks["has_recent_error"] = False

        # Calculate metrics
        if self._total_executions > 0:
            metrics["success_rate"] = self._success_count / self._total_executions
            metrics["error_rate"] = self._error_count / self._total_executions
        else:
            metrics["success_rate"] = 1.0
            metrics["error_rate"] = 0.0

        # Check success rate
        checks["success_rate_healthy"] = metrics["success_rate"] >= 0.9
        if metrics["success_rate"] < 0.9:
            degradation_reasons.append(f"Low success rate: {metrics['success_rate']:.1%}")

        # Average latency
        if recent_executions:
            avg_latency = sum(m.duration_ms for m in recent_executions) / len(recent_executions)
            metrics["avg_latency_ms"] = avg_latency
            checks["latency_acceptable"] = avg_latency < self._latency_threshold_ms

            if avg_latency >= self._latency_threshold_ms:
                degradation_reasons.append(f"High latency: {avg_latency:.0f}ms")
        else:
            metrics["avg_latency_ms"] = 0
            checks["latency_acceptable"] = True

        # Uptime
        uptime_seconds = (DeterministicClock.utcnow() - self._start_time).total_seconds()
        metrics["uptime_seconds"] = uptime_seconds

        # Total executions
        metrics["total_executions"] = self._total_executions
        metrics["error_count"] = self._error_count
        metrics["success_count"] = self._success_count

        # Determine overall status
        if all(checks.values()):
            status = HealthStatus.HEALTHY
        elif metrics.get("success_rate", 0) > 0.5:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.UNHEALTHY

        result = HealthCheckResult(
            status=status,
            timestamp=DeterministicClock.utcnow().isoformat(),
            checks=checks,
            metrics=metrics,
            last_error=self._last_error,
            uptime_seconds=uptime_seconds,
            degradation_reasons=degradation_reasons
        )

        self._log_structured("info", "Health check performed", result.to_dict())

        return result

    def get_performance_summary(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Get performance metrics summary.

        Args:
            window_minutes: Time window in minutes to analyze

        Returns:
            Dictionary with performance statistics
        """
        cutoff_time = DeterministicClock.utcnow() - timedelta(minutes=window_minutes)

        recent_metrics = [
            m for m in self._execution_history
            if datetime.fromisoformat(m.timestamp) >= cutoff_time
        ]

        if not recent_metrics:
            return {
                "window_minutes": window_minutes,
                "total_executions": 0,
                "message": "No executions in time window"
            }

        # Calculate statistics
        total_executions = len(recent_metrics)
        successful = sum(1 for m in recent_metrics if m.success)
        failed = total_executions - successful

        durations = [m.duration_ms for m in recent_metrics]
        costs = [m.cost_usd for m in recent_metrics]
        tokens = [m.tokens_used for m in recent_metrics]

        # Percentiles
        sorted_durations = sorted(durations)
        p50_idx = len(sorted_durations) // 2
        p95_idx = int(len(sorted_durations) * 0.95)
        p99_idx = int(len(sorted_durations) * 0.99)

        summary = {
            "window_minutes": window_minutes,
            "total_executions": total_executions,
            "successful_executions": successful,
            "failed_executions": failed,
            "success_rate": successful / total_executions if total_executions > 0 else 0,
            "latency": {
                "avg_ms": sum(durations) / len(durations),
                "min_ms": min(durations),
                "max_ms": max(durations),
                "p50_ms": sorted_durations[p50_idx] if p50_idx < len(sorted_durations) else 0,
                "p95_ms": sorted_durations[p95_idx] if p95_idx < len(sorted_durations) else 0,
                "p99_ms": sorted_durations[p99_idx] if p99_idx < len(sorted_durations) else 0,
            },
            "cost": {
                "total_usd": sum(costs),
                "avg_usd": sum(costs) / len(costs),
                "min_usd": min(costs),
                "max_usd": max(costs),
            },
            "tokens": {
                "total": sum(tokens),
                "avg": sum(tokens) / len(tokens),
                "min": min(tokens),
                "max": max(tokens),
            },
            "cache_hit_rate": sum(1 for m in recent_metrics if m.cache_hit) / total_executions,
        }

        return summary

    def get_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        unresolved_only: bool = True
    ) -> List[Alert]:
        """Get alerts filtered by criteria.

        Args:
            severity: Filter by severity level
            unresolved_only: Only return unresolved alerts

        Returns:
            List of Alert objects
        """
        alerts = self._alerts

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        if unresolved_only:
            alerts = [a for a in alerts if not a.resolved]

        return alerts

    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved.

        Args:
            alert_id: ID of the alert to resolve

        Returns:
            True if alert was found and resolved
        """
        for alert in self._alerts:
            if alert.alert_id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = DeterministicClock.utcnow().isoformat()

                self._log_structured("info", "Alert resolved", {
                    "alert_id": alert_id,
                    "severity": alert.severity.value
                })

                return True

        return False

    def export_metrics_prometheus(self) -> str:
        """Export metrics in Prometheus format.

        Returns:
            Metrics in Prometheus text format
        """
        return self._metrics_collector.export_prometheus()

    def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent execution history.

        Args:
            limit: Maximum number of executions to return

        Returns:
            List of execution metrics
        """
        history = list(self._execution_history)[-limit:]
        return [m.to_dict() for m in history]

    def reset_monitoring_state(self):
        """Reset all monitoring state (useful for testing)."""
        self._execution_history.clear()
        self._alerts.clear()
        self._error_count = 0
        self._success_count = 0
        self._total_executions = 0
        self._last_error = None

        self._log_structured("warning", "Monitoring state reset", {
            "agent": self._monitoring_agent_name
        })

    def set_thresholds(
        self,
        latency_ms: Optional[int] = None,
        error_rate: Optional[float] = None,
        cost_usd: Optional[float] = None
    ):
        """Update alerting thresholds.

        Args:
            latency_ms: Latency threshold in milliseconds
            error_rate: Error rate threshold (0.0 to 1.0)
            cost_usd: Cost threshold in USD
        """
        if latency_ms is not None:
            self._latency_threshold_ms = latency_ms

        if error_rate is not None:
            self._error_rate_threshold = error_rate

        if cost_usd is not None:
            self._cost_threshold_usd = cost_usd

        self._log_structured("info", "Thresholds updated", {
            "latency_threshold_ms": self._latency_threshold_ms,
            "error_rate_threshold": self._error_rate_threshold,
            "cost_threshold_usd": self._cost_threshold_usd
        })

    def _emit_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Emit Prometheus-compatible metric.

        Args:
            name: Metric name
            value: Metric value
            labels: Optional metric labels
        """
        if not self._metrics_enabled:
            return

        metric_labels = labels or {}
        metric_labels["agent"] = self._monitoring_agent_name

        self._metrics_collector.record_metric(name, value, metric_labels)

    def _log_structured(self, level: str, message: str, extra: Dict[str, Any] = None):
        """Emit structured JSON log.

        Args:
            level: Log level (debug, info, warning, error, critical)
            message: Log message
            extra: Additional context
        """
        log_entry = {
            "timestamp": DeterministicClock.utcnow().isoformat(),
            "level": level,
            "message": message,
            "agent": self._monitoring_agent_name,
        }

        if extra:
            log_entry["context"] = extra

        # Log as JSON
        log_json = json.dumps(log_entry)

        log_func = getattr(self._monitoring_logger, level.lower(), self._monitoring_logger.info)
        log_func(log_json)

    def _generate_alert(self, severity: AlertSeverity, message: str, context: Dict[str, Any]):
        """Generate operational alert.

        Args:
            severity: Alert severity
            message: Alert message
            context: Alert context
        """
        if not self._alerting_enabled:
            return

        alert = Alert(
            alert_id=str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
            severity=severity,
            timestamp=DeterministicClock.utcnow().isoformat(),
            message=message,
            context=context,
            agent_name=self._monitoring_agent_name
        )

        self._alerts.append(alert)

        self._log_structured("warning", f"Alert generated: {message}", {
            "alert_id": alert.alert_id,
            "severity": severity.value,
            "context": context
        })

        # Call alert callback if configured
        if self._alert_callback:
            try:
                self._alert_callback(alert)
            except Exception as e:
                self._log_structured("error", f"Alert callback failed: {e}", {
                    "alert_id": alert.alert_id,
                    "error": str(e)
                })


# Convenience function for quick integration
def add_monitoring(agent_instance, agent_name: str, **kwargs):
    """Add monitoring to an existing agent instance.

    Args:
        agent_instance: Agent instance to add monitoring to
        agent_name: Name of the agent
        **kwargs: Additional monitoring configuration

    Returns:
        The agent instance with monitoring added

    Example:
        agent = CarbonAgent()
        add_monitoring(agent, "carbon_agent")
    """
    if not hasattr(agent_instance, 'setup_monitoring'):
        raise ValueError("Agent must inherit from OperationalMonitoringMixin")

    agent_instance.setup_monitoring(agent_name=agent_name, **kwargs)
    return agent_instance
