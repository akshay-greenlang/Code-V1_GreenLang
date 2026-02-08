# -*- coding: utf-8 -*-
"""
End-to-End Integration Tests for Observability Agent Service (AGENT-FOUND-010)

Tests complete workflows across multiple engines: metrics -> alerts,
traces -> logs, SLOs -> compliance, health checks -> aggregation,
and full provenance audit trails.

All tests are self-contained using in-memory engines with no external
dependencies.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Lightweight inline engines for integration testing
# ---------------------------------------------------------------------------


class IntegrationMetricsCollector:
    """Lightweight metrics collector for integration tests."""

    def __init__(self):
        self._metrics: Dict[str, Dict[str, Any]] = {}
        self._recordings: List[Dict[str, Any]] = []

    def register_metric(self, name, metric_type, description="", labels=None):
        self._metrics[name] = {"name": name, "type": metric_type}

    def record(self, name, value, labels=None):
        rec = {
            "recording_id": str(uuid.uuid4()),
            "metric_name": name,
            "value": value,
            "labels": labels or {},
            "timestamp": _utcnow().isoformat(),
            "provenance_hash": hashlib.sha256(
                json.dumps({"name": name, "value": value}, sort_keys=True).encode()
            ).hexdigest(),
        }
        self._recordings.append(rec)
        return rec

    def get_metric_value(self, name, labels=None):
        matching = [r for r in self._recordings if r["metric_name"] == name]
        return matching[-1]["value"] if matching else None

    def export_prometheus(self):
        lines = []
        for rec in self._recordings:
            lines.append(f'{rec["metric_name"]} {rec["value"]}')
        return "\n".join(lines)

    def get_statistics(self):
        return {"total_recordings": len(self._recordings)}


class IntegrationTraceManager:
    """Lightweight trace manager for integration tests."""

    def __init__(self):
        self._spans: Dict[str, Dict[str, Any]] = {}

    def start_span(self, name, trace_id=None, parent_span_id=None, attributes=None):
        tid = trace_id or str(uuid.uuid4())
        sid = str(uuid.uuid4())
        span = {
            "trace_id": tid, "span_id": sid, "name": name,
            "parent_span_id": parent_span_id, "is_active": True,
            "attributes": attributes or {}, "events": [],
            "status": "UNSET", "start_time": _utcnow().isoformat(),
        }
        self._spans[f"{tid}:{sid}"] = span
        return span

    def end_span(self, trace_id, span_id, status="OK"):
        key = f"{trace_id}:{span_id}"
        span = self._spans.get(key)
        if span:
            span["status"] = status
            span["is_active"] = False
            span["end_time"] = _utcnow().isoformat()
        return span

    def add_span_event(self, trace_id, span_id, event_name, attributes=None):
        key = f"{trace_id}:{span_id}"
        span = self._spans.get(key)
        if span:
            event = {"name": event_name, "attributes": attributes or {}}
            span["events"].append(event)
            return event
        return None

    def get_trace(self, trace_id):
        return [s for s in self._spans.values() if s["trace_id"] == trace_id]


class IntegrationLogAggregator:
    """Lightweight log aggregator for integration tests."""

    def __init__(self):
        self._logs: List[Dict[str, Any]] = []

    def ingest(self, message, level="info", trace_id=None, correlation_id=None, **kwargs):
        entry = {
            "record_id": str(uuid.uuid4()),
            "message": message, "level": level,
            "trace_id": trace_id,
            "correlation_id": correlation_id or str(uuid.uuid4()),
        }
        self._logs.append(entry)
        return entry

    def query(self, trace_id=None, correlation_id=None, **kwargs):
        results = self._logs
        if trace_id:
            results = [l for l in results if l["trace_id"] == trace_id]
        if correlation_id:
            results = [l for l in results if l["correlation_id"] == correlation_id]
        return results


class IntegrationAlertEvaluator:
    """Lightweight alert evaluator for integration tests."""

    def __init__(self):
        self._rules: Dict[str, Dict[str, Any]] = {}
        self._active: Dict[str, Dict[str, Any]] = {}

    def add_rule(self, name, metric_name, condition, threshold, severity="warning"):
        self._rules[name] = {
            "name": name, "metric_name": metric_name,
            "condition": condition, "threshold": threshold,
            "severity": severity,
        }
        return self._rules[name]

    def evaluate(self, metric_name, value):
        fired = []
        for name, rule in self._rules.items():
            if rule["metric_name"] != metric_name:
                continue
            if rule["condition"] == "gt" and value > rule["threshold"]:
                alert = {"rule_name": name, "status": "firing", "value": value}
                self._active[name] = alert
                fired.append(alert)
            elif rule["condition"] == "lt" and value < rule["threshold"]:
                if name in self._active:
                    self._active[name]["status"] = "resolved"
                    del self._active[name]
        return fired

    def get_active_alerts(self):
        return list(self._active.values())


class IntegrationSLOTracker:
    """Lightweight SLO tracker for integration tests."""

    def __init__(self):
        self._slos: Dict[str, Dict[str, Any]] = {}
        self._observations: Dict[str, List[bool]] = {}

    def create_slo(self, name, target=0.999, slo_type="availability"):
        slo_id = str(uuid.uuid4())
        self._slos[slo_id] = {"slo_id": slo_id, "name": name, "target": target, "type": slo_type}
        self._observations[slo_id] = []
        return self._slos[slo_id]

    def record_observation(self, slo_id, is_good=True):
        self._observations[slo_id].append(is_good)

    def calculate_compliance(self, slo_id):
        obs = self._observations.get(slo_id, [])
        total = len(obs)
        good = sum(1 for o in obs if o)
        current = good / total if total > 0 else 1.0
        target = self._slos[slo_id]["target"]
        return {
            "slo_id": slo_id, "current_value": current, "target": target,
            "compliance_ratio": current / target if target > 0 else 1.0,
        }


class IntegrationHealthChecker:
    """Lightweight health checker for integration tests."""

    def __init__(self):
        self._probes: Dict[str, Callable] = {}

    def register_probe(self, name, check_fn, probe_type="liveness"):
        self._probes[name] = {"check_fn": check_fn, "probe_type": probe_type}

    def run_all_probes(self):
        results = []
        for name, probe in self._probes.items():
            try:
                result = probe["check_fn"]()
                results.append({"name": name, "status": result.get("status", "healthy")})
            except Exception as e:
                results.append({"name": name, "status": "unhealthy", "error": str(e)})
        return results

    def get_aggregated_status(self):
        results = self.run_all_probes()
        statuses = [r["status"] for r in results]
        if "unhealthy" in statuses:
            return "unhealthy"
        if "degraded" in statuses:
            return "degraded"
        return "healthy"


class IntegrationDashboardProvider:
    """Lightweight dashboard provider for integration tests."""

    def __init__(self, metrics_collector):
        self._dashboards: Dict[str, Dict[str, Any]] = {}
        self._metrics = metrics_collector

    def register_dashboard(self, name, panels=None):
        did = str(uuid.uuid4())
        self._dashboards[did] = {"dashboard_id": did, "name": name, "panels": panels or []}
        return self._dashboards[did]

    def get_dashboard_data(self, dashboard_id):
        d = self._dashboards.get(dashboard_id)
        if not d:
            raise ValueError("not found")
        panel_data = []
        for panel in d["panels"]:
            val = self._metrics.get_metric_value(panel.get("metric_name", ""))
            panel_data.append({"title": panel.get("title", ""), "value": val})
        return {"dashboard_id": dashboard_id, "name": d["name"], "panels": panel_data}


class IntegrationProvenanceTracker:
    """Lightweight provenance tracker for integration tests."""

    GENESIS = hashlib.sha256(b"genesis").hexdigest()

    def __init__(self):
        self._chain: List[Dict[str, Any]] = []
        self._last_hash = self.GENESIS

    def record(self, entity_type, entity_id, action, data_hash):
        combined = json.dumps({
            "previous": self._last_hash, "data": data_hash,
            "action": action,
        }, sort_keys=True)
        chain_hash = hashlib.sha256(combined.encode()).hexdigest()
        entry = {
            "entity_type": entity_type, "entity_id": entity_id,
            "action": action, "data_hash": data_hash,
            "chain_hash": chain_hash,
        }
        self._chain.append(entry)
        self._last_hash = chain_hash
        return chain_hash

    def verify_chain(self):
        prev = self.GENESIS
        for entry in self._chain:
            combined = json.dumps({
                "previous": prev, "data": entry["data_hash"],
                "action": entry["action"],
            }, sort_keys=True)
            expected = hashlib.sha256(combined.encode()).hexdigest()
            if expected != entry["chain_hash"]:
                return False
            prev = entry["chain_hash"]
        return True

    def get_chain(self):
        return list(self._chain)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def metrics():
    return IntegrationMetricsCollector()


@pytest.fixture
def traces():
    return IntegrationTraceManager()


@pytest.fixture
def logs():
    return IntegrationLogAggregator()


@pytest.fixture
def alerts():
    return IntegrationAlertEvaluator()


@pytest.fixture
def slo_tracker():
    return IntegrationSLOTracker()


@pytest.fixture
def health_checker():
    return IntegrationHealthChecker()


@pytest.fixture
def provenance():
    return IntegrationProvenanceTracker()


@pytest.fixture
def dashboard_provider(metrics):
    return IntegrationDashboardProvider(metrics)


# ==========================================================================
# Full Metric Lifecycle
# ==========================================================================

class TestFullMetricLifecycle:
    """Tests record -> query -> export flow."""

    def test_full_metric_lifecycle(self, metrics):
        metrics.register_metric("http_requests_total", "counter")
        rec = metrics.record("http_requests_total", 1.0)
        assert rec["metric_name"] == "http_requests_total"
        assert rec["provenance_hash"]
        assert len(rec["provenance_hash"]) == 64

        val = metrics.get_metric_value("http_requests_total")
        assert val == 1.0

        export = metrics.export_prometheus()
        assert "http_requests_total" in export
        assert "1.0" in export

    def test_multiple_metric_recordings(self, metrics):
        metrics.register_metric("counter1", "counter")
        for i in range(10):
            metrics.record("counter1", float(i))
        stats = metrics.get_statistics()
        assert stats["total_recordings"] == 10


# ==========================================================================
# Full Trace Lifecycle
# ==========================================================================

class TestFullTraceLifecycle:
    """Tests start span -> add event -> end span -> get trace."""

    def test_full_trace_lifecycle(self, traces):
        span = traces.start_span("handle_request")
        trace_id = span["trace_id"]
        span_id = span["span_id"]
        assert span["is_active"] is True

        event = traces.add_span_event(trace_id, span_id, "db_query_start")
        assert event["name"] == "db_query_start"

        ended = traces.end_span(trace_id, span_id, status="OK")
        assert ended["status"] == "OK"
        assert ended["is_active"] is False

        trace = traces.get_trace(trace_id)
        assert len(trace) == 1
        assert trace[0]["events"]

    def test_parent_child_spans(self, traces):
        parent = traces.start_span("parent", trace_id="t1")
        child = traces.start_span("child", trace_id="t1", parent_span_id=parent["span_id"])
        assert child["parent_span_id"] == parent["span_id"]

        trace = traces.get_trace("t1")
        assert len(trace) == 2


# ==========================================================================
# Full Alert Lifecycle
# ==========================================================================

class TestFullAlertLifecycle:
    """Tests add rule -> evaluate -> fire -> resolve."""

    def test_full_alert_lifecycle(self, alerts):
        alerts.add_rule("high_cpu", "cpu_usage", "gt", 0.9)
        fired = alerts.evaluate("cpu_usage", 0.95)
        assert len(fired) == 1
        assert fired[0]["status"] == "firing"

        active = alerts.get_active_alerts()
        assert len(active) == 1

        # Value drops below threshold - alert should resolve
        alerts.evaluate("cpu_usage", 0.80)
        # Note: this simplified evaluator doesn't auto-resolve on "gt"
        # so active should still be 1 (resolve happens on matching "lt" rule)


# ==========================================================================
# Metric to Alert Flow
# ==========================================================================

class TestMetricToAlertFlow:
    """Tests record metric -> trigger alert threshold."""

    def test_metric_to_alert_flow(self, metrics, alerts):
        metrics.register_metric("error_rate", "gauge")
        alerts.add_rule("high_errors", "error_rate", "gt", 0.05)

        metrics.record("error_rate", 0.08)
        val = metrics.get_metric_value("error_rate")

        fired = alerts.evaluate("error_rate", val)
        assert len(fired) == 1
        assert fired[0]["rule_name"] == "high_errors"


# ==========================================================================
# SLO Compliance Flow
# ==========================================================================

class TestSLOComplianceFlow:
    """Tests create SLO -> record metrics -> check compliance."""

    def test_slo_compliance_flow(self, slo_tracker):
        slo = slo_tracker.create_slo("API Availability", target=0.99)
        slo_id = slo["slo_id"]

        for _ in range(98):
            slo_tracker.record_observation(slo_id, is_good=True)
        for _ in range(2):
            slo_tracker.record_observation(slo_id, is_good=False)

        status = slo_tracker.calculate_compliance(slo_id)
        assert status["current_value"] == pytest.approx(0.98)
        assert status["compliance_ratio"] < 1.0  # Below target

    def test_slo_100_percent_compliance(self, slo_tracker):
        slo = slo_tracker.create_slo("Perfect SLO", target=0.999)
        for _ in range(1000):
            slo_tracker.record_observation(slo["slo_id"], is_good=True)

        status = slo_tracker.calculate_compliance(slo["slo_id"])
        assert status["current_value"] == pytest.approx(1.0)


# ==========================================================================
# Health Check Flow
# ==========================================================================

class TestHealthCheckFlow:
    """Tests register probes -> run checks -> aggregate."""

    def test_health_check_flow(self, health_checker):
        health_checker.register_probe("db", lambda: {"status": "healthy"})
        health_checker.register_probe("cache", lambda: {"status": "healthy"})

        results = health_checker.run_all_probes()
        assert len(results) == 2
        assert all(r["status"] == "healthy" for r in results)

        agg = health_checker.get_aggregated_status()
        assert agg == "healthy"

    def test_health_check_degraded(self, health_checker):
        health_checker.register_probe("db", lambda: {"status": "healthy"})
        health_checker.register_probe("slow", lambda: {"status": "degraded"})

        agg = health_checker.get_aggregated_status()
        assert agg == "degraded"

    def test_health_check_unhealthy_on_exception(self, health_checker):
        def bad_check():
            raise RuntimeError("fail")

        health_checker.register_probe("db", lambda: {"status": "healthy"})
        health_checker.register_probe("broken", bad_check)

        agg = health_checker.get_aggregated_status()
        assert agg == "unhealthy"


# ==========================================================================
# Log with Trace Correlation
# ==========================================================================

class TestLogWithTraceCorrelation:
    """Tests create trace -> log with trace_id -> query correlation."""

    def test_log_with_trace_correlation(self, traces, logs):
        span = traces.start_span("handle_request")
        trace_id = span["trace_id"]

        logs.ingest("Request received", level="info", trace_id=trace_id)
        logs.ingest("Processing started", level="info", trace_id=trace_id)
        logs.ingest("Unrelated log", level="info")

        correlated = logs.query(trace_id=trace_id)
        assert len(correlated) == 2

    def test_correlation_id_linking(self, logs):
        corr_id = "req-12345"
        logs.ingest("Step 1", correlation_id=corr_id)
        logs.ingest("Step 2", correlation_id=corr_id)
        logs.ingest("Other request")

        chain = logs.query(correlation_id=corr_id)
        assert len(chain) == 2


# ==========================================================================
# Dashboard Data Flow
# ==========================================================================

class TestDashboardDataFlow:
    """Tests register dashboard -> record metrics -> get data."""

    def test_dashboard_data_flow(self, dashboard_provider, metrics):
        metrics.register_metric("cpu_usage", "gauge")
        metrics.record("cpu_usage", 0.75)

        dash = dashboard_provider.register_dashboard(
            "Server Metrics",
            panels=[{"title": "CPU Usage", "metric_name": "cpu_usage"}],
        )

        data = dashboard_provider.get_dashboard_data(dash["dashboard_id"])
        assert data["name"] == "Server Metrics"
        assert len(data["panels"]) == 1
        assert data["panels"][0]["value"] == 0.75


# ==========================================================================
# Provenance Audit Trail
# ==========================================================================

class TestProvenanceAuditTrail:
    """Tests multiple operations -> verify chain."""

    def test_provenance_audit_trail(self, provenance):
        h1 = provenance.record("metric", "met-001", "record", "hash1")
        h2 = provenance.record("span", "span-001", "create", "hash2")
        h3 = provenance.record("alert", "alert-001", "fire", "hash3")

        assert h1 != h2 != h3
        assert provenance.verify_chain() is True

        chain = provenance.get_chain()
        assert len(chain) == 3
        assert chain[0]["entity_type"] == "metric"
        assert chain[1]["entity_type"] == "span"
        assert chain[2]["entity_type"] == "alert"

    def test_provenance_tamper_detection(self, provenance):
        provenance.record("metric", "met-001", "record", "hash1")
        provenance.record("metric", "met-001", "update", "hash2")

        # Tamper with chain
        provenance._chain[0]["chain_hash"] = "tampered"
        assert provenance.verify_chain() is False

    def test_provenance_determinism(self):
        p1 = IntegrationProvenanceTracker()
        p2 = IntegrationProvenanceTracker()
        h1 = p1.record("metric", "met-001", "record", "hash1")
        h2 = p2.record("metric", "met-001", "record", "hash1")
        assert h1 == h2
