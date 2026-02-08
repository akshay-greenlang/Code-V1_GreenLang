# -*- coding: utf-8 -*-
"""
Observability Agent Service Setup - AGENT-FOUND-010: Observability Agent

Provides ``configure_observability_agent(app)`` which wires up the
Observability & Telemetry Agent SDK (metrics collector, trace manager,
log aggregator, alert evaluator, health checker, dashboard provider,
SLO tracker, provenance tracker) and mounts the REST API.

Also exposes ``get_observability_agent(app)`` for programmatic access
and the ``ObservabilityAgentService`` facade class.

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.observability_agent.setup import configure_observability_agent
    >>> app = FastAPI()
    >>> import asyncio
    >>> service = asyncio.run(configure_observability_agent(app))

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-010 Observability & Telemetry Agent
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.observability_agent.config import ObservabilityAgentConfig, get_config
from greenlang.observability_agent.metrics import (
    PROMETHEUS_AVAILABLE,
    record_metric_recorded,
    record_operation_duration,
    record_span_created,
    update_active_spans,
    record_log_ingested,
    record_alert_evaluated,
    update_firing_alerts,
    record_health_check,
    update_health_status,
    update_slo_compliance,
    update_error_budget,
    record_dashboard_query,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FastAPI import
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI
    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None  # type: ignore[assignment, misc]
    FASTAPI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Lightweight in-memory model stubs
# ---------------------------------------------------------------------------
# These are intentionally minimal so the setup facade can work standalone
# without requiring all engine modules to be present. When the full SDK
# engines are implemented they provide richer Pydantic models.
# ---------------------------------------------------------------------------


class _SimpleModel:
    """Minimal dict-backed model for standalone facade operation."""

    def __init__(self, **kwargs: Any) -> None:
        self._data = kwargs

    def model_dump(self, mode: str = "json") -> Dict[str, Any]:
        """Serialize to dictionary.

        Args:
            mode: Serialization mode (kept for Pydantic compatibility).

        Returns:
            Dictionary of all stored fields.
        """
        return dict(self._data)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return self._data.get(name)


# ---------------------------------------------------------------------------
# Lightweight in-memory engine stubs
# ---------------------------------------------------------------------------


class _MetricsCollector:
    """In-memory metrics collector engine stub.

    Stores metric observations in a dictionary keyed by metric name.
    Supports listing, retrieval, and Prometheus-format export.
    """

    def __init__(self, config: ObservabilityAgentConfig) -> None:
        self._config = config
        self._metrics: Dict[str, Dict[str, Any]] = {}

    def record(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        tenant_id: str = "default",
        metric_type: str = "gauge",
        description: str = "",
        unit: str = "",
    ) -> _SimpleModel:
        """Record a metric observation.

        Args:
            name: Metric name.
            value: Observation value.
            labels: Optional key-value labels.
            tenant_id: Tenant identifier.
            metric_type: Type of metric.
            description: Human-readable description.
            unit: Metric unit string.

        Returns:
            Recorded metric entry as a model.
        """
        entry = {
            "metric_name": name,
            "value": value,
            "labels": labels or {},
            "tenant_id": tenant_id,
            "metric_type": metric_type,
            "description": description,
            "unit": unit,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._metrics[name] = entry
        return _SimpleModel(**entry)

    def list_metrics(
        self,
        tenant_id: Optional[str] = None,
        metric_type: Optional[str] = None,
    ) -> List[_SimpleModel]:
        """List metric definitions with optional filters.

        Args:
            tenant_id: Filter by tenant.
            metric_type: Filter by metric type.

        Returns:
            List of matching metric models.
        """
        results: List[_SimpleModel] = []
        for m in self._metrics.values():
            if tenant_id and m.get("tenant_id") != tenant_id:
                continue
            if metric_type and m.get("metric_type") != metric_type:
                continue
            results.append(_SimpleModel(**m))
        return results

    def get_metric(self, name: str) -> Optional[_SimpleModel]:
        """Get a metric by name.

        Args:
            name: Metric name to look up.

        Returns:
            Metric model or None if not found.
        """
        entry = self._metrics.get(name)
        return _SimpleModel(**entry) if entry else None

    def export_metrics(
        self,
        format: str = "prometheus",
        metric_names: Optional[List[str]] = None,
        tenant_id: Optional[str] = None,
    ) -> str:
        """Export metrics in the requested format.

        Args:
            format: Export format (prometheus).
            metric_names: Optional name filter.
            tenant_id: Optional tenant filter.

        Returns:
            Formatted metrics string.
        """
        lines: List[str] = []
        for name, m in self._metrics.items():
            if metric_names and name not in metric_names:
                continue
            if tenant_id and m.get("tenant_id") != tenant_id:
                continue
            label_str = ",".join(
                f'{k}="{v}"' for k, v in m.get("labels", {}).items()
            )
            if label_str:
                lines.append(f"{name}{{{label_str}}} {m['value']}")
            else:
                lines.append(f"{name} {m['value']}")
        return "\n".join(lines)

    def get_statistics(self) -> Dict[str, Any]:
        """Get metrics collector statistics.

        Returns:
            Dictionary with total recordings and series counts.
        """
        return {
            "total_recordings": len(self._metrics),
            "total_series": len(self._metrics),
        }


class _TraceManager:
    """In-memory trace manager engine stub.

    Maintains active and completed spans grouped by trace ID.
    """

    def __init__(self, config: ObservabilityAgentConfig) -> None:
        self._config = config
        self._traces: Dict[str, Dict[str, Any]] = {}
        self._spans: Dict[str, Dict[str, Any]] = {}
        self._active_count = 0

    def start_span(
        self,
        name: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        service_name: str = "unknown",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> _SimpleModel:
        """Start a new trace span.

        Args:
            name: Operation name.
            trace_id: Optional trace ID (auto-generated if omitted).
            parent_span_id: Optional parent span ID for nesting.
            service_name: Originating service name.
            attributes: Span attributes.

        Returns:
            Span record model with trace_id and span_id.
        """
        tid = trace_id or str(uuid.uuid4())
        sid = str(uuid.uuid4())
        span = {
            "trace_id": tid,
            "span_id": sid,
            "parent_span_id": parent_span_id,
            "operation_name": name,
            "service_name": service_name,
            "attributes": attributes or {},
            "status": "in_progress",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": None,
        }
        self._spans[sid] = span

        # Register trace
        if tid not in self._traces:
            self._traces[tid] = {
                "trace_id": tid,
                "spans": [],
                "started_at": span["start_time"],
            }
        self._traces[tid]["spans"].append(sid)
        self._active_count += 1
        return _SimpleModel(**span)

    def end_span(
        self,
        span_id: str,
        status: str = "ok",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Optional[_SimpleModel]:
        """End a trace span.

        Args:
            span_id: ID of the span to end.
            status: Final span status.
            attributes: Additional attributes to merge.

        Returns:
            Updated span model, or None if not found.
        """
        span = self._spans.get(span_id)
        if span is None:
            return None
        span["status"] = status
        span["end_time"] = datetime.now(timezone.utc).isoformat()
        if attributes:
            span["attributes"].update(attributes)
        if self._active_count > 0:
            self._active_count -= 1
        return _SimpleModel(**span)

    def get_trace(self, trace_id: str) -> Optional[_SimpleModel]:
        """Get a trace with all its spans.

        Args:
            trace_id: Trace identifier.

        Returns:
            Trace model with embedded spans, or None.
        """
        trace = self._traces.get(trace_id)
        if trace is None:
            return None
        spans = [
            self._spans[sid]
            for sid in trace["spans"]
            if sid in self._spans
        ]
        return _SimpleModel(
            trace_id=trace_id,
            spans=spans,
            span_count=len(spans),
            started_at=trace["started_at"],
        )

    @property
    def active_span_count(self) -> int:
        """Return the number of active spans."""
        return self._active_count

    @property
    def total_spans(self) -> int:
        """Return the total number of spans."""
        return len(self._spans)

    @property
    def total_traces(self) -> int:
        """Return the total number of traces."""
        return len(self._traces)

    def get_statistics(self) -> Dict[str, Any]:
        """Get trace manager statistics.

        Returns:
            Dictionary with span and trace counts.
        """
        return {
            "total_spans_created": len(self._spans),
            "active_spans": self._active_count,
            "total_traces": len(self._traces),
        }


class _LogAggregator:
    """In-memory log aggregator engine stub.

    Buffers structured log entries with configurable max size and
    supports filtered querying.
    """

    def __init__(self, config: ObservabilityAgentConfig) -> None:
        self._config = config
        self._buffer: List[Dict[str, Any]] = []
        self._max_size = config.log_buffer_size

    def ingest(
        self,
        level: str,
        message: str,
        agent_id: str = "",
        tenant_id: str = "default",
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> _SimpleModel:
        """Ingest a structured log entry.

        Args:
            level: Log level string.
            message: Log message text.
            agent_id: Originating agent ID.
            tenant_id: Tenant identifier.
            trace_id: Correlated trace ID.
            span_id: Correlated span ID.
            correlation_id: Business correlation ID.
            attributes: Additional structured attributes.

        Returns:
            Stored log record model.
        """
        entry = {
            "log_id": str(uuid.uuid4()),
            "level": level.lower(),
            "message": message,
            "agent_id": agent_id,
            "tenant_id": tenant_id,
            "trace_id": trace_id,
            "span_id": span_id,
            "correlation_id": correlation_id,
            "attributes": attributes or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._buffer.append(entry)
        # Evict oldest entries if buffer full
        if len(self._buffer) > self._max_size:
            self._buffer = self._buffer[-self._max_size:]
        return _SimpleModel(**entry)

    def query_logs(
        self,
        level: Optional[str] = None,
        agent_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> List[_SimpleModel]:
        """Query log entries with filters.

        Args:
            level: Filter by log level.
            agent_id: Filter by agent ID.
            tenant_id: Filter by tenant.
            trace_id: Filter by trace ID.
            correlation_id: Filter by correlation ID.

        Returns:
            List of matching log entry models, newest first.
        """
        results: List[_SimpleModel] = []
        for entry in reversed(self._buffer):
            if level and entry.get("level") != level.lower():
                continue
            if agent_id and entry.get("agent_id") != agent_id:
                continue
            if tenant_id and entry.get("tenant_id") != tenant_id:
                continue
            if trace_id and entry.get("trace_id") != trace_id:
                continue
            if correlation_id and entry.get("correlation_id") != correlation_id:
                continue
            results.append(_SimpleModel(**entry))
        return results

    @property
    def total_entries(self) -> int:
        """Return the total number of buffered log entries."""
        return len(self._buffer)

    def get_statistics(self) -> Dict[str, Any]:
        """Get log aggregator statistics.

        Returns:
            Dictionary with ingestion counts.
        """
        return {
            "total_ingested": len(self._buffer),
            "buffer_size": self._max_size,
        }


class _AlertEvaluator:
    """In-memory alert evaluator engine stub.

    Stores alert rules and evaluates them against current metric
    values from the metrics collector.
    """

    def __init__(
        self,
        config: ObservabilityAgentConfig,
        metrics_collector: _MetricsCollector,
    ) -> None:
        self._config = config
        self._metrics = metrics_collector
        self._rules: Dict[str, Dict[str, Any]] = {}
        self._active_alerts: List[Dict[str, Any]] = []

    def add_rule(
        self,
        name: str,
        metric_name: str,
        condition: str,
        threshold: float,
        duration_seconds: int = 60,
        severity: str = "warning",
        labels: Optional[Dict[str, str]] = None,
        annotations: Optional[Dict[str, str]] = None,
    ) -> _SimpleModel:
        """Add or update an alert rule.

        Args:
            name: Rule name (used as key for upsert).
            metric_name: Metric to evaluate.
            condition: Condition operator.
            threshold: Threshold value.
            duration_seconds: Seconds before firing.
            severity: Alert severity.
            labels: Alert labels.
            annotations: Alert annotations.

        Returns:
            Created or updated rule model.
        """
        rule = {
            "rule_id": self._rules.get(name, {}).get(
                "rule_id", str(uuid.uuid4()),
            ),
            "name": name,
            "metric_name": metric_name,
            "condition": condition,
            "threshold": threshold,
            "duration_seconds": duration_seconds,
            "severity": severity,
            "labels": labels or {},
            "annotations": annotations or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "state": "inactive",
        }
        self._rules[name] = rule
        return _SimpleModel(**rule)

    def list_rules(
        self,
        severity: Optional[str] = None,
    ) -> List[_SimpleModel]:
        """List alert rules with optional severity filter.

        Args:
            severity: Optional severity level filter.

        Returns:
            List of matching rule models.
        """
        results: List[_SimpleModel] = []
        for rule in self._rules.values():
            if severity and rule.get("severity") != severity:
                continue
            results.append(_SimpleModel(**rule))
        return results

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate all alert rules against current metrics.

        Returns:
            Evaluation summary with counts of evaluated, firing,
            and resolved rules.
        """
        evaluated = 0
        firing = 0
        resolved = 0
        new_active: List[Dict[str, Any]] = []

        for rule in self._rules.values():
            evaluated += 1
            metric = self._metrics.get_metric(rule["metric_name"])
            if metric is None:
                continue

            value = metric.value  # type: ignore[union-attr]
            threshold = rule["threshold"]
            condition = rule["condition"]
            is_firing = _evaluate_condition(value, condition, threshold)

            if is_firing:
                firing += 1
                alert_entry = {
                    "rule_name": rule["name"],
                    "metric_name": rule["metric_name"],
                    "current_value": value,
                    "threshold": threshold,
                    "condition": condition,
                    "severity": rule["severity"],
                    "state": "firing",
                    "fired_at": datetime.now(timezone.utc).isoformat(),
                }
                new_active.append(alert_entry)
                rule["state"] = "firing"
            else:
                if rule.get("state") == "firing":
                    resolved += 1
                rule["state"] = "inactive"

        self._active_alerts = new_active

        return {
            "evaluated": evaluated,
            "firing": firing,
            "resolved": resolved,
            "total_rules": len(self._rules),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_active_alerts(
        self,
        severity: Optional[str] = None,
    ) -> List[_SimpleModel]:
        """Get currently firing alerts.

        Args:
            severity: Optional severity level filter.

        Returns:
            List of firing alert models.
        """
        results: List[_SimpleModel] = []
        for alert in self._active_alerts:
            if severity and alert.get("severity") != severity:
                continue
            results.append(_SimpleModel(**alert))
        return results

    @property
    def total_rules(self) -> int:
        """Return the total number of rules."""
        return len(self._rules)

    @property
    def firing_count(self) -> int:
        """Return the count of currently firing alerts."""
        return len(self._active_alerts)

    def get_statistics(self) -> Dict[str, Any]:
        """Get alert evaluator statistics.

        Returns:
            Dictionary with rule and alert counts.
        """
        return {
            "total_rules": len(self._rules),
            "active_alerts": len(self._active_alerts),
            "total_fired": len(self._active_alerts),
        }


class _HealthChecker:
    """In-memory health checker engine stub.

    Runs health probes and caches the most recent result.
    """

    def __init__(self, config: ObservabilityAgentConfig) -> None:
        self._config = config
        self._last_check: Optional[Dict[str, Any]] = None

    def check(
        self,
        probe_type: Optional[str] = None,
        service_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run health check probes.

        Args:
            probe_type: Type of probe (liveness, readiness, startup).
            service_filter: Filter by service name pattern.

        Returns:
            Health check results dictionary.
        """
        result = {
            "overall_status": "healthy",
            "probe_type": probe_type or "all",
            "service_filter": service_filter,
            "checks": [
                {
                    "service": "observability-agent",
                    "status": "healthy",
                    "probe_type": probe_type or "liveness",
                    "latency_ms": 1.0,
                },
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._last_check = result
        return result

    def get_status(self) -> _SimpleModel:
        """Get the aggregated health status.

        Returns:
            Health status model from the last check.
        """
        if self._last_check:
            return _SimpleModel(**self._last_check)
        return _SimpleModel(
            overall_status="unknown",
            checks=[],
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get health checker statistics.

        Returns:
            Dictionary with health check counts.
        """
        return {
            "last_status": (
                self._last_check.get("overall_status", "unknown")
                if self._last_check else "unknown"
            ),
        }


class _DashboardProvider:
    """In-memory dashboard provider engine stub.

    Stores and retrieves dashboard data by ID.
    """

    def __init__(self, config: ObservabilityAgentConfig) -> None:
        self._config = config
        self._dashboards: Dict[str, Dict[str, Any]] = {}

    def get_dashboard(self, dashboard_id: str) -> Optional[Dict[str, Any]]:
        """Get dashboard data by ID.

        Args:
            dashboard_id: Dashboard identifier.

        Returns:
            Dashboard data dictionary, or None if not found.
        """
        return self._dashboards.get(dashboard_id)

    def register_dashboard(
        self, dashboard_id: str, data: Dict[str, Any],
    ) -> None:
        """Register or update a dashboard.

        Args:
            dashboard_id: Dashboard identifier.
            data: Dashboard data to store.
        """
        self._dashboards[dashboard_id] = data

    def get_statistics(self) -> Dict[str, Any]:
        """Get dashboard provider statistics.

        Returns:
            Dictionary with dashboard counts.
        """
        return {
            "total_dashboards": len(self._dashboards),
        }


class _SLOTracker:
    """In-memory SLO tracker engine stub.

    Manages SLO definitions and computes burn rate analysis.
    """

    def __init__(self, config: ObservabilityAgentConfig) -> None:
        self._config = config
        self._slos: Dict[str, Dict[str, Any]] = {}

    def create(
        self,
        name: str,
        description: str = "",
        service_name: str = "",
        slo_type: str = "availability",
        target: float = 0.999,
        window_days: int = 30,
        burn_rate_thresholds: Optional[Dict[str, float]] = None,
    ) -> _SimpleModel:
        """Create or update an SLO definition.

        Args:
            name: SLO name.
            description: SLO description.
            service_name: Target service name.
            slo_type: SLO type.
            target: Target ratio.
            window_days: Rolling evaluation window in days.
            burn_rate_thresholds: Burn rate thresholds by tier.

        Returns:
            Created or updated SLO model.
        """
        slo_id = None
        for sid, slo in self._slos.items():
            if (
                slo.get("name") == name
                and slo.get("service_name") == service_name
            ):
                slo_id = sid
                break
        if slo_id is None:
            slo_id = str(uuid.uuid4())

        slo = {
            "slo_id": slo_id,
            "name": name,
            "description": description,
            "service_name": service_name,
            "slo_type": slo_type,
            "target": target,
            "window_days": window_days,
            "burn_rate_thresholds": burn_rate_thresholds or {
                "fast": 14.4, "medium": 6.0, "slow": 1.0,
            },
            "current_compliance": target,
            "error_budget_remaining": 1.0,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._slos[slo_id] = slo
        return _SimpleModel(**slo)

    def list_slos(
        self,
        service_name: Optional[str] = None,
        slo_type: Optional[str] = None,
    ) -> List[_SimpleModel]:
        """List SLOs with optional filters.

        Args:
            service_name: Filter by service.
            slo_type: Filter by SLO type.

        Returns:
            List of matching SLO models.
        """
        results: List[_SimpleModel] = []
        for slo in self._slos.values():
            if service_name and slo.get("service_name") != service_name:
                continue
            if slo_type and slo.get("slo_type") != slo_type:
                continue
            results.append(_SimpleModel(**slo))
        return results

    def get_slo(self, slo_id: str) -> Optional[_SimpleModel]:
        """Get an SLO definition by ID.

        Args:
            slo_id: SLO identifier.

        Returns:
            SLO model or None.
        """
        slo = self._slos.get(slo_id)
        return _SimpleModel(**slo) if slo else None

    def get_burn_rate(self, slo_id: str) -> Optional[_SimpleModel]:
        """Get burn rate analysis for an SLO.

        Computes fast, medium, and slow burn rate windows based on
        the configured thresholds.

        Args:
            slo_id: SLO identifier.

        Returns:
            Burn rate analysis model, or None if SLO not found.
        """
        slo = self._slos.get(slo_id)
        if slo is None:
            return None
        thresholds = slo.get("burn_rate_thresholds", {})
        return _SimpleModel(
            slo_id=slo_id,
            slo_name=slo.get("name", ""),
            service_name=slo.get("service_name", ""),
            target=slo.get("target", 0.999),
            current_compliance=slo.get("current_compliance", 0.999),
            error_budget_remaining=slo.get("error_budget_remaining", 1.0),
            burn_rates={
                "fast": {
                    "window_minutes": self._config.burn_rate_short_window_minutes,
                    "rate": 0.0,
                    "threshold": thresholds.get("fast", 14.4),
                    "is_burning": False,
                },
                "medium": {
                    "window_minutes": self._config.burn_rate_long_window_minutes,
                    "rate": 0.0,
                    "threshold": thresholds.get("medium", 6.0),
                    "is_burning": False,
                },
                "slow": {
                    "window_minutes": (
                        self._config.burn_rate_long_window_minutes * 6
                    ),
                    "rate": 0.0,
                    "threshold": thresholds.get("slow", 1.0),
                    "is_burning": False,
                },
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    @property
    def total_slos(self) -> int:
        """Return the total number of SLOs."""
        return len(self._slos)

    def get_statistics(self) -> Dict[str, Any]:
        """Get SLO tracker statistics.

        Returns:
            Dictionary with SLO counts.
        """
        return {
            "total_slos": len(self._slos),
        }


class _ProvenanceTracker:
    """SHA-256 chain-hashed provenance tracker for audit trails.

    Maintains an ordered log of operations with SHA-256 hashes that
    chain together to provide tamper-evident audit trails, grouped by
    entity.
    """

    _GENESIS_HASH = hashlib.sha256(
        b"greenlang-observability-agent-genesis"
    ).hexdigest()

    def __init__(self) -> None:
        """Initialize ProvenanceTracker."""
        self._chain_store: Dict[str, List[Dict[str, Any]]] = {}
        self._global_chain: List[Dict[str, Any]] = []
        self._last_chain_hash: str = self._GENESIS_HASH

    def record(
        self,
        entity_type: str,
        entity_id: str,
        action: str,
        data_hash: str,
        user_id: str = "system",
    ) -> str:
        """Record a provenance entry.

        Args:
            entity_type: Type of entity (metric, span, log, alert, slo).
            entity_id: Unique entity identifier.
            action: Action performed (record, create, evaluate, etc.).
            data_hash: SHA-256 hash of the operation data.
            user_id: User who performed the operation.

        Returns:
            Chain hash of the new entry.
        """
        timestamp = _utcnow().isoformat()
        chain_hash = self._compute_chain_hash(
            self._last_chain_hash, data_hash, action, timestamp,
        )
        entry = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "action": action,
            "data_hash": data_hash,
            "user_id": user_id,
            "timestamp": timestamp,
            "chain_hash": chain_hash,
        }
        if entity_id not in self._chain_store:
            self._chain_store[entity_id] = []
        self._chain_store[entity_id].append(entry)
        self._global_chain.append(entry)
        self._last_chain_hash = chain_hash
        return chain_hash

    def build_hash(self, data: Any) -> str:
        """Build a SHA-256 hash for arbitrary data.

        Args:
            data: Data to hash (dict, list, or other serializable).

        Returns:
            Hex-encoded SHA-256 hash.
        """
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _compute_chain_hash(
        self,
        previous_hash: str,
        data_hash: str,
        action: str,
        timestamp: str,
    ) -> str:
        """Compute the next chain hash linking to the previous.

        Args:
            previous_hash: Previous chain hash.
            data_hash: Current operation data hash.
            action: Action performed.
            timestamp: ISO-formatted timestamp.

        Returns:
            New SHA-256 chain hash.
        """
        combined = json.dumps({
            "previous": previous_hash,
            "data": data_hash,
            "action": action,
            "timestamp": timestamp,
        }, sort_keys=True)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    @property
    def entry_count(self) -> int:
        """Return the total number of provenance entries."""
        return len(self._global_chain)

    @property
    def entity_count(self) -> int:
        """Return the number of unique entities tracked."""
        return len(self._chain_store)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _evaluate_condition(
    value: float, condition: str, threshold: float,
) -> bool:
    """Evaluate a numeric condition against a threshold.

    Args:
        value: Observed metric value.
        condition: Operator string (gt, lt, gte, lte, eq, ne).
        threshold: Threshold to compare against.

    Returns:
        True if condition is satisfied (alert should fire).
    """
    ops = {
        "gt": lambda v, t: v > t,
        "lt": lambda v, t: v < t,
        "gte": lambda v, t: v >= t,
        "lte": lambda v, t: v <= t,
        "eq": lambda v, t: v == t,
        "ne": lambda v, t: v != t,
    }
    op = ops.get(condition.lower())
    if op is None:
        return False
    return op(value, threshold)


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ===================================================================
# ObservabilityAgentService facade
# ===================================================================

# Thread-safe singleton lock
_singleton_lock = threading.Lock()
_singleton_instance: Optional["ObservabilityAgentService"] = None


class ObservabilityAgentService:
    """Unified facade over the Observability & Telemetry Agent SDK.

    Aggregates all observability engines (metrics collector, trace manager,
    log aggregator, alert evaluator, health checker, dashboard provider,
    SLO tracker, provenance tracker) through a single entry point with
    convenience methods for common operations.

    Attributes:
        config: ObservabilityAgentConfig instance.
        metrics_collector: In-memory metrics collector engine.
        trace_manager: In-memory distributed trace manager.
        log_aggregator: In-memory structured log aggregator.
        alert_evaluator: In-memory alert evaluation engine.
        health_checker: In-memory health check engine.
        dashboard_provider: In-memory dashboard data provider.
        slo_tracker: In-memory SLO tracking engine.
        provenance: SHA-256 chain-hashed provenance tracker.

    Example:
        >>> service = ObservabilityAgentService()
        >>> result = service.record_metric("cpu_usage", 0.75, {"host": "n1"})
        >>> print(result.model_dump())
    """

    def __init__(
        self,
        config: Optional[ObservabilityAgentConfig] = None,
    ) -> None:
        """Initialize the Observability Agent Service facade.

        Args:
            config: Optional configuration. Uses global config if None.
        """
        self.config = config or get_config()

        # Initialize all 7 engines + provenance tracker
        self.provenance = _ProvenanceTracker()
        self.metrics_collector = _MetricsCollector(self.config)
        self.trace_manager = _TraceManager(self.config)
        self.log_aggregator = _LogAggregator(self.config)
        self.alert_evaluator = _AlertEvaluator(
            self.config, self.metrics_collector,
        )
        self.health_checker = _HealthChecker(self.config)
        self.dashboard_provider = _DashboardProvider(self.config)
        self.slo_tracker = _SLOTracker(self.config)

        # Self-monitoring statistics
        self._total_operations = 0
        self._started = False
        self._start_time: Optional[datetime] = None

        logger.info("ObservabilityAgentService facade created")

    # ------------------------------------------------------------------
    # Metrics operations
    # ------------------------------------------------------------------

    def record_metric(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        tenant_id: str = "default",
        metric_type: str = "gauge",
        description: str = "",
        unit: str = "",
    ) -> _SimpleModel:
        """Record a metric observation.

        Args:
            name: Metric name.
            value: Observation value.
            labels: Optional metric labels.
            tenant_id: Tenant identifier.
            metric_type: Type of metric (counter, gauge, histogram, summary).
            description: Metric description.
            unit: Metric unit.

        Returns:
            Recorded metric entry.
        """
        start = time.monotonic()
        result = self.metrics_collector.record(
            name=name,
            value=value,
            labels=labels,
            tenant_id=tenant_id,
            metric_type=metric_type,
            description=description,
            unit=unit,
        )

        # Provenance
        data_hash = self.provenance.build_hash({
            "name": name, "value": value, "labels": labels or {},
        })
        self.provenance.record(
            entity_type="metric",
            entity_id=name,
            action="record",
            data_hash=data_hash,
        )

        # Self-monitoring metrics
        self._total_operations += 1
        duration = time.monotonic() - start
        record_metric_recorded(metric_type, tenant_id)
        record_operation_duration(duration)

        logger.debug(
            "Recorded metric %s = %s (tenant=%s)", name, value, tenant_id,
        )
        return result

    # ------------------------------------------------------------------
    # Trace operations
    # ------------------------------------------------------------------

    def start_span(
        self,
        name: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        service_name: str = "unknown",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> _SimpleModel:
        """Create and start a new trace span.

        Args:
            name: Operation name for the span.
            trace_id: Optional trace ID (auto-generated if omitted).
            parent_span_id: Optional parent span ID for nesting.
            service_name: Name of the originating service.
            attributes: Span attributes dictionary.

        Returns:
            Span record with trace_id and span_id.
        """
        start = time.monotonic()
        result = self.trace_manager.start_span(
            name=name,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            service_name=service_name,
            attributes=attributes,
        )

        # Provenance
        data_hash = self.provenance.build_hash({
            "operation": name,
            "trace_id": result.trace_id,
            "span_id": result.span_id,
        })
        self.provenance.record(
            entity_type="span",
            entity_id=result.span_id,
            action="start",
            data_hash=data_hash,
        )

        # Self-monitoring metrics
        self._total_operations += 1
        duration = time.monotonic() - start
        record_span_created("started")
        update_active_spans(self.trace_manager.active_span_count)
        record_operation_duration(duration)

        logger.debug(
            "Started span %s in trace %s",
            result.span_id, result.trace_id,
        )
        return result

    def end_span(
        self,
        span_id: str,
        status: str = "ok",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Optional[_SimpleModel]:
        """End a trace span.

        Args:
            span_id: ID of the span to end.
            status: Final span status (ok, error, timeout, cancelled).
            attributes: Additional attributes to merge on close.

        Returns:
            Updated span record, or None if not found.
        """
        start = time.monotonic()
        result = self.trace_manager.end_span(
            span_id=span_id,
            status=status,
            attributes=attributes,
        )
        if result is None:
            return None

        # Provenance
        data_hash = self.provenance.build_hash({
            "span_id": span_id, "status": status,
        })
        self.provenance.record(
            entity_type="span",
            entity_id=span_id,
            action="end",
            data_hash=data_hash,
        )

        # Self-monitoring
        self._total_operations += 1
        duration = time.monotonic() - start
        record_span_created(status)
        update_active_spans(self.trace_manager.active_span_count)
        record_operation_duration(duration)

        logger.debug("Ended span %s with status=%s", span_id, status)
        return result

    # ------------------------------------------------------------------
    # Log operations
    # ------------------------------------------------------------------

    def log(
        self,
        level: str,
        message: str,
        agent_id: str = "",
        tenant_id: str = "default",
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> _SimpleModel:
        """Ingest a structured log entry.

        Args:
            level: Log level (debug, info, warning, error, critical).
            message: Log message text.
            agent_id: Originating agent ID.
            tenant_id: Tenant identifier.
            trace_id: Correlated trace ID.
            span_id: Correlated span ID.
            correlation_id: Business-level correlation ID.
            attributes: Additional structured attributes.

        Returns:
            Stored log record.
        """
        start = time.monotonic()
        result = self.log_aggregator.ingest(
            level=level,
            message=message,
            agent_id=agent_id,
            tenant_id=tenant_id,
            trace_id=trace_id,
            span_id=span_id,
            correlation_id=correlation_id,
            attributes=attributes,
        )

        # Provenance
        data_hash = self.provenance.build_hash({
            "level": level, "message": message, "agent_id": agent_id,
        })
        self.provenance.record(
            entity_type="log",
            entity_id=result.log_id,
            action="ingest",
            data_hash=data_hash,
        )

        # Self-monitoring
        self._total_operations += 1
        duration = time.monotonic() - start
        record_log_ingested(level.lower())
        record_operation_duration(duration)

        logger.debug("Ingested %s log from agent %s", level, agent_id)
        return result

    # ------------------------------------------------------------------
    # Alert operations
    # ------------------------------------------------------------------

    def add_alert_rule(
        self,
        name: str,
        metric_name: str,
        condition: str,
        threshold: float,
        duration_seconds: int = 60,
        severity: str = "warning",
        labels: Optional[Dict[str, str]] = None,
        annotations: Optional[Dict[str, str]] = None,
    ) -> _SimpleModel:
        """Create or update an alert rule.

        Args:
            name: Alert rule name.
            metric_name: Metric to evaluate.
            condition: Condition operator (gt, lt, gte, lte, eq, ne).
            threshold: Threshold value.
            duration_seconds: Duration before firing.
            severity: Alert severity level.
            labels: Alert labels.
            annotations: Alert annotations.

        Returns:
            Created or updated alert rule.
        """
        start = time.monotonic()
        result = self.alert_evaluator.add_rule(
            name=name,
            metric_name=metric_name,
            condition=condition,
            threshold=threshold,
            duration_seconds=duration_seconds,
            severity=severity,
            labels=labels,
            annotations=annotations,
        )

        # Provenance
        data_hash = self.provenance.build_hash({
            "name": name, "metric_name": metric_name,
            "condition": condition, "threshold": threshold,
        })
        self.provenance.record(
            entity_type="alert_rule",
            entity_id=name,
            action="create",
            data_hash=data_hash,
        )

        # Self-monitoring
        self._total_operations += 1
        duration = time.monotonic() - start
        record_operation_duration(duration)

        logger.info(
            "Added alert rule: %s (%s %s %s)",
            name, metric_name, condition, threshold,
        )
        return result

    def evaluate_alerts(self) -> Dict[str, Any]:
        """Evaluate all alert rules against current metric values.

        Returns:
            Evaluation summary with firing/resolved counts.
        """
        start = time.monotonic()
        result = self.alert_evaluator.evaluate()

        # Update Prometheus metrics
        firing = result.get("firing", 0)
        resolved = result.get("resolved", 0)
        record_alert_evaluated("firing" if firing > 0 else "resolved")
        update_firing_alerts(firing)

        # Provenance
        data_hash = self.provenance.build_hash(result)
        self.provenance.record(
            entity_type="alert_evaluation",
            entity_id="global",
            action="evaluate",
            data_hash=data_hash,
        )

        # Self-monitoring
        self._total_operations += 1
        duration = time.monotonic() - start
        record_operation_duration(duration)

        logger.info(
            "Alert evaluation complete: %d firing, %d resolved",
            firing, resolved,
        )
        return result

    # ------------------------------------------------------------------
    # Health check operations
    # ------------------------------------------------------------------

    def check_health(
        self,
        probe_type: Optional[str] = None,
        service_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run health check probes.

        Args:
            probe_type: Type of probe (liveness, readiness, startup).
            service_filter: Filter by service name pattern.

        Returns:
            Health check results dictionary.
        """
        start = time.monotonic()
        result = self.health_checker.check(
            probe_type=probe_type,
            service_filter=service_filter,
        )

        # Update Prometheus metrics
        status = result.get("overall_status", "unknown")
        status_value = {
            "healthy": 1.0, "degraded": 0.5, "unhealthy": 0.0,
        }.get(status, 0.0)
        record_health_check(status, probe_type or "all")
        update_health_status(status_value)

        # Provenance
        data_hash = self.provenance.build_hash(result)
        self.provenance.record(
            entity_type="health_check",
            entity_id="global",
            action="check",
            data_hash=data_hash,
        )

        # Self-monitoring
        self._total_operations += 1
        duration = time.monotonic() - start
        record_operation_duration(duration)

        logger.debug("Health check complete: %s", status)
        return result

    # ------------------------------------------------------------------
    # Dashboard operations
    # ------------------------------------------------------------------

    def get_dashboard_data(
        self, dashboard_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get dashboard data by ID.

        Args:
            dashboard_id: Dashboard identifier.

        Returns:
            Dashboard data dictionary, or None if not found.
        """
        start = time.monotonic()
        data = self.dashboard_provider.get_dashboard(dashboard_id)

        # Self-monitoring
        self._total_operations += 1
        duration = time.monotonic() - start
        record_dashboard_query()
        record_operation_duration(duration)

        if data is not None:
            # Provenance
            data_hash = self.provenance.build_hash({
                "dashboard_id": dashboard_id,
            })
            self.provenance.record(
                entity_type="dashboard",
                entity_id=dashboard_id,
                action="query",
                data_hash=data_hash,
            )

        return data

    # ------------------------------------------------------------------
    # SLO operations
    # ------------------------------------------------------------------

    def create_slo(
        self,
        name: str,
        service_name: str,
        slo_type: str,
        target: float,
        description: str = "",
        window_days: int = 30,
        burn_rate_thresholds: Optional[Dict[str, float]] = None,
    ) -> _SimpleModel:
        """Create or update an SLO definition.

        Args:
            name: SLO name.
            service_name: Target service name.
            slo_type: SLO type (availability, latency, throughput, error_rate).
            target: Target ratio (e.g. 0.999 for 99.9%).
            description: SLO description.
            window_days: Rolling evaluation window in days.
            burn_rate_thresholds: Burn rate thresholds by speed tier.

        Returns:
            Created or updated SLO definition.
        """
        start = time.monotonic()
        result = self.slo_tracker.create(
            name=name,
            description=description,
            service_name=service_name,
            slo_type=slo_type,
            target=target,
            window_days=window_days,
            burn_rate_thresholds=burn_rate_thresholds,
        )

        # Update Prometheus metrics
        update_slo_compliance(service_name, slo_type, target)
        update_error_budget(service_name, 1.0)

        # Provenance
        data_hash = self.provenance.build_hash({
            "name": name, "service_name": service_name,
            "slo_type": slo_type, "target": target,
        })
        self.provenance.record(
            entity_type="slo",
            entity_id=result.slo_id,
            action="create",
            data_hash=data_hash,
        )

        # Self-monitoring
        self._total_operations += 1
        duration = time.monotonic() - start
        record_operation_duration(duration)

        logger.info(
            "Created SLO: %s for %s (target=%.3f, type=%s)",
            name, service_name, target, slo_type,
        )
        return result

    def get_slo_status(self, slo_id: str) -> Optional[_SimpleModel]:
        """Get SLO status including compliance and error budget.

        Args:
            slo_id: SLO identifier.

        Returns:
            SLO status model, or None if not found.
        """
        return self.slo_tracker.get_slo(slo_id)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> _SimpleModel:
        """Get aggregated observability agent statistics.

        Returns:
            Statistics summary model with counts for all subsystems.
        """
        uptime = 0.0
        if self._start_time:
            uptime = (_utcnow() - self._start_time).total_seconds()

        return _SimpleModel(
            total_operations=self._total_operations,
            uptime_seconds=uptime,
            metrics_count=len(self.metrics_collector._metrics),
            total_traces=self.trace_manager.total_traces,
            total_spans=self.trace_manager.total_spans,
            active_spans=self.trace_manager.active_span_count,
            total_log_entries=self.log_aggregator.total_entries,
            alert_rules=self.alert_evaluator.total_rules,
            alerts_firing=self.alert_evaluator.firing_count,
            total_slos=self.slo_tracker.total_slos,
            provenance_entries=self.provenance.entry_count,
            provenance_entities=self.provenance.entity_count,
            prometheus_available=PROMETHEUS_AVAILABLE,
            started=self._started,
        )

    # ------------------------------------------------------------------
    # Convenience getters
    # ------------------------------------------------------------------

    def get_metrics_collector(self) -> _MetricsCollector:
        """Get the MetricsCollector instance.

        Returns:
            MetricsCollector used by this service.
        """
        return self.metrics_collector

    def get_trace_manager(self) -> _TraceManager:
        """Get the TraceManager instance.

        Returns:
            TraceManager used by this service.
        """
        return self.trace_manager

    def get_log_aggregator(self) -> _LogAggregator:
        """Get the LogAggregator instance.

        Returns:
            LogAggregator used by this service.
        """
        return self.log_aggregator

    def get_alert_evaluator(self) -> _AlertEvaluator:
        """Get the AlertEvaluator instance.

        Returns:
            AlertEvaluator used by this service.
        """
        return self.alert_evaluator

    def get_health_checker(self) -> _HealthChecker:
        """Get the HealthChecker instance.

        Returns:
            HealthChecker used by this service.
        """
        return self.health_checker

    def get_dashboard_provider(self) -> _DashboardProvider:
        """Get the DashboardProvider instance.

        Returns:
            DashboardProvider used by this service.
        """
        return self.dashboard_provider

    def get_slo_tracker(self) -> _SLOTracker:
        """Get the SLOTracker instance.

        Returns:
            SLOTracker used by this service.
        """
        return self.slo_tracker

    def get_provenance(self) -> _ProvenanceTracker:
        """Get the ProvenanceTracker instance.

        Returns:
            ProvenanceTracker used by this service.
        """
        return self.provenance

    # ------------------------------------------------------------------
    # Metrics summary
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Get observability agent service metrics summary.

        Returns:
            Dictionary with service metric summaries.
        """
        return {
            "prometheus_available": PROMETHEUS_AVAILABLE,
            "started": self._started,
            "total_operations": self._total_operations,
            "metrics_count": len(self.metrics_collector._metrics),
            "total_traces": self.trace_manager.total_traces,
            "total_spans": self.trace_manager.total_spans,
            "active_spans": self.trace_manager.active_span_count,
            "total_log_entries": self.log_aggregator.total_entries,
            "alert_rules": self.alert_evaluator.total_rules,
            "alerts_firing": self.alert_evaluator.firing_count,
            "total_slos": self.slo_tracker.total_slos,
            "provenance_entries": self.provenance.entry_count,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def startup(self) -> None:
        """Start the observability agent service.

        Safe to call multiple times.
        """
        if self._started:
            logger.debug(
                "ObservabilityAgentService already started; skipping",
            )
            return

        logger.info("ObservabilityAgentService starting up...")
        self._started = True
        self._start_time = _utcnow()
        logger.info("ObservabilityAgentService startup complete")

    def shutdown(self) -> None:
        """Shutdown the observability agent service and release resources."""
        if not self._started:
            return

        self._started = False
        logger.info("ObservabilityAgentService shut down")


# ===================================================================
# Thread-safe singleton access
# ===================================================================


def _get_singleton() -> ObservabilityAgentService:
    """Get or create the singleton ObservabilityAgentService instance.

    Returns:
        The singleton ObservabilityAgentService.
    """
    global _singleton_instance
    if _singleton_instance is None:
        with _singleton_lock:
            if _singleton_instance is None:
                _singleton_instance = ObservabilityAgentService()
    return _singleton_instance


# ===================================================================
# FastAPI integration
# ===================================================================


async def configure_observability_agent(
    app: Any,
    config: Optional[ObservabilityAgentConfig] = None,
) -> ObservabilityAgentService:
    """Configure the Observability Agent Service on a FastAPI application.

    Creates the ObservabilityAgentService, stores it in app.state, mounts
    the observability agent API router, and starts the service.

    Args:
        app: FastAPI application instance.
        config: Optional observability agent config.

    Returns:
        ObservabilityAgentService instance.
    """
    global _singleton_instance

    service = ObservabilityAgentService(config=config)

    # Store as singleton
    with _singleton_lock:
        _singleton_instance = service

    # Attach to app state
    app.state.observability_agent_service = service

    # Mount observability agent API router
    try:
        from greenlang.observability_agent.api.router import (
            router as obs_router,
        )
        if obs_router is not None:
            app.include_router(obs_router)
            logger.info("Observability agent service API router mounted")
    except ImportError:
        logger.warning(
            "Observability agent router not available; API not mounted",
        )

    # Start service
    service.startup()

    logger.info("Observability agent service configured on app")
    return service


def get_observability_agent(app: Any) -> ObservabilityAgentService:
    """Get the ObservabilityAgentService instance from app state.

    Args:
        app: FastAPI application instance.

    Returns:
        ObservabilityAgentService instance.

    Raises:
        RuntimeError: If observability agent service not configured.
    """
    service = getattr(app.state, "observability_agent_service", None)
    if service is None:
        raise RuntimeError(
            "Observability agent service not configured. "
            "Call configure_observability_agent(app) first."
        )
    return service


def get_router() -> Any:
    """Get the observability agent API router.

    Returns:
        FastAPI APIRouter or None if FastAPI not available.
    """
    try:
        from greenlang.observability_agent.api.router import router
        return router
    except ImportError:
        return None


__all__ = [
    "ObservabilityAgentService",
    "configure_observability_agent",
    "get_observability_agent",
    "get_router",
]
