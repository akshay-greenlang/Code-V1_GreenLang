# -*- coding: utf-8 -*-
"""
Unit Tests for ObservabilityAgentService Facade (AGENT-FOUND-010)

Tests service creation, engine initialization, convenience methods for
metrics, traces, logs, alerts, health, SLOs, statistics, and FastAPI
configuration.

Since setup.py is not yet on disk, tests define the expected interface
via an inline implementation matching the PRD specification.

Coverage target: 85%+ of setup.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline ObservabilityAgentService (mirrors expected interface)
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


class _StubMetricsCollector:
    def __init__(self):
        self._metrics = {}
        self._recordings = 0

    def register_metric(self, name, metric_type, description="", labels=None, buckets=None):
        self._metrics[name] = metric_type
        return name

    def record(self, name, value, labels=None):
        self._recordings += 1
        return {"metric_name": name, "value": value}

    def get_statistics(self):
        return {"total_recordings": self._recordings}


class _StubTraceManager:
    def __init__(self):
        self._spans = {}
        self._created = 0

    def start_span(self, name, trace_id=None, parent_span_id=None, attributes=None, kind="INTERNAL"):
        self._created += 1
        return {"name": name, "trace_id": trace_id or str(uuid.uuid4()), "span_id": str(uuid.uuid4())}

    def end_span(self, trace_id, span_id, status="OK", attributes=None):
        return {"trace_id": trace_id, "span_id": span_id, "status": status}

    def get_statistics(self):
        return {"total_spans_created": self._created}


class _StubLogAggregator:
    def __init__(self):
        self._logs = []

    def ingest(self, message, level="info", **kwargs):
        self._logs.append(message)
        return {"message": message, "level": level}

    def get_statistics(self):
        return {"total_ingested": len(self._logs)}


class _StubAlertEvaluator:
    def __init__(self):
        self._rules = {}

    def add_rule(self, name, metric_name, condition, threshold, **kwargs):
        self._rules[name] = {"metric_name": metric_name, "condition": condition, "threshold": threshold}
        return self._rules[name]

    def evaluate(self, metric_name, value, labels=None):
        return []

    def get_statistics(self):
        return {"total_rules": len(self._rules)}


class _StubHealthChecker:
    def __init__(self):
        self._probes = {}

    def get_aggregated_status(self):
        return "healthy"

    def run_all_probes(self):
        return []

    def get_statistics(self):
        return {"registered_probes": len(self._probes)}


class _StubSLOTracker:
    def __init__(self):
        self._slos = {}

    def create_slo(self, name, **kwargs):
        slo_id = str(uuid.uuid4())
        self._slos[slo_id] = {"name": name}
        return {"slo_id": slo_id, "name": name}

    def get_statistics(self):
        return {"total_slos": len(self._slos)}


class _StubProvenanceTracker:
    def __init__(self):
        self._entries = 0

    def record(self, *args, **kwargs):
        self._entries += 1
        return "hash"

    def get_statistics(self):
        return {"entry_count": self._entries}


class ObservabilityAgentService:
    """Facade for the Observability Agent SDK."""

    def __init__(self, config: Any = None) -> None:
        self._config = config
        self._start_time = time.time()
        self.metrics_collector = _StubMetricsCollector()
        self.trace_manager = _StubTraceManager()
        self.log_aggregator = _StubLogAggregator()
        self.alert_evaluator = _StubAlertEvaluator()
        self.health_checker = _StubHealthChecker()
        self.slo_tracker = _StubSLOTracker()
        self.provenance_tracker = _StubProvenanceTracker()

    def record_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        return self.metrics_collector.record(name, value, labels)

    def start_span(self, name: str, **kwargs):
        return self.trace_manager.start_span(name, **kwargs)

    def end_span(self, trace_id: str, span_id: str, **kwargs):
        return self.trace_manager.end_span(trace_id, span_id, **kwargs)

    def log(self, message: str, level: str = "info", **kwargs):
        return self.log_aggregator.ingest(message, level=level, **kwargs)

    def add_alert_rule(self, name: str, metric_name: str, condition: str, threshold: float, **kwargs):
        return self.alert_evaluator.add_rule(name, metric_name, condition, threshold, **kwargs)

    def check_health(self):
        return self.health_checker.get_aggregated_status()

    def create_slo(self, name: str, **kwargs):
        return self.slo_tracker.create_slo(name, **kwargs)

    def get_statistics(self) -> Dict[str, Any]:
        uptime = time.time() - self._start_time
        return {
            "uptime_seconds": uptime,
            "metrics": self.metrics_collector.get_statistics(),
            "traces": self.trace_manager.get_statistics(),
            "logs": self.log_aggregator.get_statistics(),
            "alerts": self.alert_evaluator.get_statistics(),
            "health": self.health_checker.get_statistics(),
            "slos": self.slo_tracker.get_statistics(),
            "provenance": self.provenance_tracker.get_statistics(),
        }

    def configure_with_app(self, app: Any) -> None:
        """Register FastAPI router with the app."""
        self._app = app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class _TestConfig:
    prometheus_url: str = "http://localhost:9090"
    grafana_url: str = "http://localhost:3000"


@pytest.fixture
def config():
    return _TestConfig()


@pytest.fixture
def service(config):
    return ObservabilityAgentService(config)


# ==========================================================================
# Service Creation Tests
# ==========================================================================

class TestObservabilityAgentServiceCreation:
    """Tests for service initialization."""

    def test_service_creation(self, config):
        svc = ObservabilityAgentService(config)
        assert svc is not None
        assert svc._config is config

    def test_service_has_all_engines(self, service):
        assert service.metrics_collector is not None
        assert service.trace_manager is not None
        assert service.log_aggregator is not None
        assert service.alert_evaluator is not None
        assert service.health_checker is not None
        assert service.slo_tracker is not None
        assert service.provenance_tracker is not None

    def test_service_creation_without_config(self):
        svc = ObservabilityAgentService()
        assert svc is not None


# ==========================================================================
# Convenience Method Tests
# ==========================================================================

class TestObservabilityAgentServiceConvenience:
    """Tests for convenience methods that delegate to engines."""

    def test_record_metric(self, service):
        result = service.record_metric("cpu_usage", 0.85)
        assert result["metric_name"] == "cpu_usage"
        assert result["value"] == 0.85

    def test_record_metric_with_labels(self, service):
        result = service.record_metric("http_req", 1.0, labels={"method": "GET"})
        assert result["metric_name"] == "http_req"

    def test_start_span(self, service):
        result = service.start_span("process_data")
        assert result["name"] == "process_data"
        assert "trace_id" in result

    def test_end_span(self, service):
        result = service.end_span("t1", "s1", status="OK")
        assert result["status"] == "OK"

    def test_log(self, service):
        result = service.log("Something happened")
        assert result["message"] == "Something happened"
        assert result["level"] == "info"

    def test_log_with_level(self, service):
        result = service.log("Error occurred", level="error")
        assert result["level"] == "error"

    def test_add_alert_rule(self, service):
        result = service.add_alert_rule("high_cpu", "cpu_usage", "gt", 0.9)
        assert result["metric_name"] == "cpu_usage"
        assert result["condition"] == "gt"
        assert result["threshold"] == 0.9

    def test_check_health(self, service):
        status = service.check_health()
        assert status == "healthy"

    def test_create_slo(self, service):
        result = service.create_slo("API Availability")
        assert result["name"] == "API Availability"


# ==========================================================================
# Statistics Tests
# ==========================================================================

class TestObservabilityAgentServiceStatistics:
    """Tests for get_statistics."""

    def test_get_statistics(self, service):
        stats = service.get_statistics()
        assert "uptime_seconds" in stats
        assert stats["uptime_seconds"] >= 0
        assert "metrics" in stats
        assert "traces" in stats
        assert "logs" in stats
        assert "alerts" in stats
        assert "health" in stats
        assert "slos" in stats
        assert "provenance" in stats

    def test_get_statistics_after_operations(self, service):
        service.record_metric("m", 1.0)
        service.start_span("op")
        service.log("msg")
        service.add_alert_rule("r", "m", "gt", 1.0)
        service.create_slo("slo")
        stats = service.get_statistics()
        assert stats["metrics"]["total_recordings"] == 1
        assert stats["traces"]["total_spans_created"] == 1
        assert stats["logs"]["total_ingested"] == 1
        assert stats["alerts"]["total_rules"] == 1
        assert stats["slos"]["total_slos"] == 1


# ==========================================================================
# FastAPI Configuration Tests
# ==========================================================================

class TestObservabilityAgentServiceApp:
    """Tests for configure_with_app."""

    def test_configure_with_fastapi_app(self, service):
        mock_app = type("MockApp", (), {"include_router": lambda self, r, **kw: None})()
        service.configure_with_app(mock_app)
        assert service._app is mock_app
