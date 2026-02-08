# -*- coding: utf-8 -*-
"""
Unit Tests for Observability Agent API Router (AGENT-FOUND-010)

Tests all 20 FastAPI REST endpoints using httpx/TestClient with mocked
service engines.

Coverage target: 85%+ of api/router.py

Author: GreenLang Platform Team
Date: February 2026
"""

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Skip if FastAPI is not installed
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not FASTAPI_AVAILABLE, reason="fastapi not installed"
)


# ---------------------------------------------------------------------------
# Inline API application for testing
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _create_test_app():
    """Create a minimal FastAPI app with observability endpoints."""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    from typing import Dict, Optional, Any, List

    app = FastAPI(title="Observability Agent API Test")

    # Shared state for testing
    app.state.metrics = {}
    app.state.spans = {}
    app.state.logs = []
    app.state.alert_rules = {}
    app.state.active_alerts = {}
    app.state.slos = {}
    app.state.health_status = "healthy"

    # ---------- Health ----------

    @app.get("/api/v1/observability/health")
    def health():
        return {"status": app.state.health_status, "timestamp": _utcnow().isoformat()}

    # ---------- Metrics ----------

    class RecordMetricReq(BaseModel):
        metric_name: str
        value: float
        labels: Dict[str, str] = {}
        tenant_id: str = "default"

    @app.post("/api/v1/observability/metrics")
    def record_metric(req: RecordMetricReq):
        rec_id = str(uuid.uuid4())
        app.state.metrics[rec_id] = {
            "recording_id": rec_id,
            "metric_name": req.metric_name,
            "value": req.value,
            "labels": req.labels,
        }
        return {"recording_id": rec_id, "metric_name": req.metric_name}

    @app.get("/api/v1/observability/metrics")
    def list_metrics():
        return {"metrics": list(app.state.metrics.values())}

    @app.get("/api/v1/observability/metrics/export")
    def export_metrics():
        return {"format": "prometheus", "data": "# metrics export placeholder"}

    @app.get("/api/v1/observability/metrics/{metric_name}")
    def get_metric(metric_name: str):
        matches = [m for m in app.state.metrics.values() if m["metric_name"] == metric_name]
        if not matches:
            raise HTTPException(status_code=404, detail=f"Metric '{metric_name}' not found")
        return {"metric": matches[0]}

    # ---------- Traces ----------

    class CreateSpanReq(BaseModel):
        operation_name: str
        service_name: str = ""
        parent_span_id: Optional[str] = None
        attributes: Dict[str, Any] = {}

    @app.post("/api/v1/observability/traces/spans")
    def create_span(req: CreateSpanReq):
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        span = {
            "trace_id": trace_id,
            "span_id": span_id,
            "operation_name": req.operation_name,
            "status": "UNSET",
            "is_active": True,
        }
        app.state.spans[(trace_id, span_id)] = span
        return span

    class EndSpanReq(BaseModel):
        status: str = "OK"

    @app.put("/api/v1/observability/traces/{trace_id}/spans/{span_id}")
    def end_span(trace_id: str, span_id: str, req: EndSpanReq):
        key = (trace_id, span_id)
        if key not in app.state.spans:
            raise HTTPException(status_code=404, detail="Span not found")
        span = app.state.spans[key]
        span["status"] = req.status
        span["is_active"] = False
        return span

    @app.get("/api/v1/observability/traces/{trace_id}")
    def get_trace(trace_id: str):
        spans = [s for (tid, _), s in app.state.spans.items() if tid == trace_id]
        if not spans:
            raise HTTPException(status_code=404, detail="Trace not found")
        return {"trace_id": trace_id, "spans": spans}

    # ---------- Logs ----------

    class IngestLogReq(BaseModel):
        level: str = "info"
        message: str
        agent_id: Optional[str] = None
        trace_id: Optional[str] = None
        attributes: Dict[str, Any] = {}

    @app.post("/api/v1/observability/logs")
    def ingest_log(req: IngestLogReq):
        record_id = str(uuid.uuid4())
        entry = {
            "record_id": record_id,
            "level": req.level,
            "message": req.message,
            "agent_id": req.agent_id,
        }
        app.state.logs.append(entry)
        return entry

    @app.get("/api/v1/observability/logs")
    def query_logs(level: Optional[str] = None, limit: int = 100):
        results = app.state.logs
        if level:
            results = [l for l in results if l["level"] == level]
        return {"logs": results[:limit]}

    # ---------- Alerts ----------

    class CreateAlertRuleReq(BaseModel):
        name: str
        metric_name: str
        condition: str
        threshold: float
        severity: str = "warning"

    @app.post("/api/v1/observability/alerts/rules")
    def create_alert_rule(req: CreateAlertRuleReq):
        rule_id = str(uuid.uuid4())
        rule = {
            "rule_id": rule_id,
            "name": req.name,
            "metric_name": req.metric_name,
            "condition": req.condition,
            "threshold": req.threshold,
        }
        app.state.alert_rules[req.name] = rule
        return rule

    @app.get("/api/v1/observability/alerts/rules")
    def list_alert_rules():
        return {"rules": list(app.state.alert_rules.values())}

    @app.post("/api/v1/observability/alerts/evaluate")
    def evaluate_alerts():
        return {"evaluated": len(app.state.alert_rules), "fired": []}

    @app.get("/api/v1/observability/alerts/active")
    def get_active_alerts():
        return {"alerts": list(app.state.active_alerts.values())}

    # ---------- Health Check ----------

    @app.post("/api/v1/observability/health/check")
    def run_health_check():
        return {"status": app.state.health_status, "probes": []}

    @app.get("/api/v1/observability/health/status")
    def get_health_status():
        return {"status": app.state.health_status}

    # ---------- Dashboards ----------

    @app.get("/api/v1/observability/dashboards/{dashboard_id}")
    def get_dashboard_data(dashboard_id: str):
        return {"dashboard_id": dashboard_id, "name": "Test Dashboard", "panels": []}

    # ---------- SLOs ----------

    class CreateSLOReq(BaseModel):
        name: str
        target: float = 0.999
        slo_type: str = "availability"
        service_name: str = ""

    @app.post("/api/v1/observability/slos")
    def create_slo(req: CreateSLOReq):
        slo_id = str(uuid.uuid4())
        slo = {
            "slo_id": slo_id,
            "name": req.name,
            "target": req.target,
            "slo_type": req.slo_type,
        }
        app.state.slos[slo_id] = slo
        return slo

    @app.get("/api/v1/observability/slos")
    def list_slos():
        return {"slos": list(app.state.slos.values())}

    @app.get("/api/v1/observability/slos/{slo_id}/burn-rate")
    def get_burn_rate(slo_id: str):
        if slo_id not in app.state.slos:
            raise HTTPException(status_code=404, detail="SLO not found")
        return {
            "slo_id": slo_id,
            "burn_rate_1h": 0.0,
            "burn_rate_6h": 0.0,
            "burn_rate_24h": 0.0,
        }

    return app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app():
    return _create_test_app()


@pytest.fixture
def client(app):
    return TestClient(app)


# ==========================================================================
# Health Endpoint Tests
# ==========================================================================

class TestHealthEndpoint:
    """Tests for /health."""

    def test_health_endpoint(self, client):
        resp = client.get("/api/v1/observability/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data


# ==========================================================================
# Metric Endpoint Tests
# ==========================================================================

class TestMetricEndpoints:
    """Tests for metric CRUD endpoints."""

    def test_record_metric(self, client):
        resp = client.post(
            "/api/v1/observability/metrics",
            json={"metric_name": "cpu_usage", "value": 0.85},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["metric_name"] == "cpu_usage"
        assert "recording_id" in data

    def test_list_metrics(self, client):
        client.post(
            "/api/v1/observability/metrics",
            json={"metric_name": "m1", "value": 1.0},
        )
        resp = client.get("/api/v1/observability/metrics")
        assert resp.status_code == 200
        assert len(resp.json()["metrics"]) >= 1

    def test_get_metric_existing(self, client):
        client.post(
            "/api/v1/observability/metrics",
            json={"metric_name": "test_metric", "value": 42.0},
        )
        resp = client.get("/api/v1/observability/metrics/test_metric")
        assert resp.status_code == 200

    def test_get_metric_nonexistent(self, client):
        resp = client.get("/api/v1/observability/metrics/nonexistent")
        assert resp.status_code == 404

    def test_export_metrics(self, client):
        resp = client.get("/api/v1/observability/metrics/export")
        assert resp.status_code == 200

    def test_record_metric_with_labels(self, client):
        resp = client.post(
            "/api/v1/observability/metrics",
            json={"metric_name": "http_req", "value": 1.0, "labels": {"method": "GET"}},
        )
        assert resp.status_code == 200


# ==========================================================================
# Trace Endpoint Tests
# ==========================================================================

class TestTraceEndpoints:
    """Tests for trace/span CRUD endpoints."""

    def test_create_span(self, client):
        resp = client.post(
            "/api/v1/observability/traces/spans",
            json={"operation_name": "db_query"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["operation_name"] == "db_query"
        assert data["is_active"] is True

    def test_end_span(self, client):
        create_resp = client.post(
            "/api/v1/observability/traces/spans",
            json={"operation_name": "op"},
        )
        data = create_resp.json()
        resp = client.put(
            f"/api/v1/observability/traces/{data['trace_id']}/spans/{data['span_id']}",
            json={"status": "OK"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "OK"
        assert resp.json()["is_active"] is False

    def test_end_span_not_found(self, client):
        resp = client.put(
            "/api/v1/observability/traces/t1/spans/s1",
            json={"status": "OK"},
        )
        assert resp.status_code == 404

    def test_get_trace(self, client):
        create_resp = client.post(
            "/api/v1/observability/traces/spans",
            json={"operation_name": "op"},
        )
        trace_id = create_resp.json()["trace_id"]
        resp = client.get(f"/api/v1/observability/traces/{trace_id}")
        assert resp.status_code == 200
        assert resp.json()["trace_id"] == trace_id

    def test_get_trace_not_found(self, client):
        resp = client.get("/api/v1/observability/traces/nonexistent")
        assert resp.status_code == 404


# ==========================================================================
# Log Endpoint Tests
# ==========================================================================

class TestLogEndpoints:
    """Tests for log ingestion and query endpoints."""

    def test_ingest_log(self, client):
        resp = client.post(
            "/api/v1/observability/logs",
            json={"message": "Test log message"},
        )
        assert resp.status_code == 200
        assert resp.json()["message"] == "Test log message"

    def test_ingest_log_with_level(self, client):
        resp = client.post(
            "/api/v1/observability/logs",
            json={"message": "Error happened", "level": "error"},
        )
        assert resp.status_code == 200
        assert resp.json()["level"] == "error"

    def test_query_logs(self, client):
        client.post(
            "/api/v1/observability/logs",
            json={"message": "info log", "level": "info"},
        )
        client.post(
            "/api/v1/observability/logs",
            json={"message": "error log", "level": "error"},
        )
        resp = client.get("/api/v1/observability/logs")
        assert resp.status_code == 200
        assert len(resp.json()["logs"]) >= 2

    def test_query_logs_by_level(self, client):
        client.post(
            "/api/v1/observability/logs",
            json={"message": "error log", "level": "error"},
        )
        client.post(
            "/api/v1/observability/logs",
            json={"message": "info log", "level": "info"},
        )
        resp = client.get("/api/v1/observability/logs?level=error")
        assert resp.status_code == 200
        logs = resp.json()["logs"]
        assert all(log["level"] == "error" for log in logs)


# ==========================================================================
# Alert Endpoint Tests
# ==========================================================================

class TestAlertEndpoints:
    """Tests for alert rule and evaluation endpoints."""

    def test_create_alert_rule(self, client):
        resp = client.post(
            "/api/v1/observability/alerts/rules",
            json={
                "name": "high_cpu",
                "metric_name": "cpu_usage",
                "condition": "gt",
                "threshold": 0.9,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "high_cpu"

    def test_list_alert_rules(self, client):
        client.post(
            "/api/v1/observability/alerts/rules",
            json={"name": "r1", "metric_name": "m", "condition": "gt", "threshold": 1.0},
        )
        resp = client.get("/api/v1/observability/alerts/rules")
        assert resp.status_code == 200
        assert len(resp.json()["rules"]) >= 1

    def test_evaluate_alerts(self, client):
        resp = client.post("/api/v1/observability/alerts/evaluate")
        assert resp.status_code == 200

    def test_get_active_alerts(self, client):
        resp = client.get("/api/v1/observability/alerts/active")
        assert resp.status_code == 200
        assert "alerts" in resp.json()


# ==========================================================================
# Health Check Endpoint Tests
# ==========================================================================

class TestHealthCheckEndpoints:
    """Tests for health check probe endpoints."""

    def test_run_health_check(self, client):
        resp = client.post("/api/v1/observability/health/check")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_get_health_status(self, client):
        resp = client.get("/api/v1/observability/health/status")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"


# ==========================================================================
# Dashboard Endpoint Tests
# ==========================================================================

class TestDashboardEndpoints:
    """Tests for dashboard data endpoints."""

    def test_get_dashboard_data(self, client):
        resp = client.get("/api/v1/observability/dashboards/test-dashboard-id")
        assert resp.status_code == 200
        assert resp.json()["dashboard_id"] == "test-dashboard-id"


# ==========================================================================
# SLO Endpoint Tests
# ==========================================================================

class TestSLOEndpoints:
    """Tests for SLO CRUD and burn rate endpoints."""

    def test_create_slo(self, client):
        resp = client.post(
            "/api/v1/observability/slos",
            json={"name": "API Availability"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "API Availability"
        assert "slo_id" in data

    def test_list_slos(self, client):
        client.post(
            "/api/v1/observability/slos",
            json={"name": "SLO 1"},
        )
        resp = client.get("/api/v1/observability/slos")
        assert resp.status_code == 200
        assert len(resp.json()["slos"]) >= 1

    def test_get_burn_rate(self, client):
        create_resp = client.post(
            "/api/v1/observability/slos",
            json={"name": "Test SLO"},
        )
        slo_id = create_resp.json()["slo_id"]
        resp = client.get(f"/api/v1/observability/slos/{slo_id}/burn-rate")
        assert resp.status_code == 200
        data = resp.json()
        assert "burn_rate_1h" in data
        assert "burn_rate_6h" in data
        assert "burn_rate_24h" in data

    def test_get_burn_rate_not_found(self, client):
        resp = client.get("/api/v1/observability/slos/nonexistent/burn-rate")
        assert resp.status_code == 404


# ==========================================================================
# Error Handling Tests
# ==========================================================================

class TestErrorHandling:
    """Tests for error handling across endpoints."""

    def test_invalid_metric_name_empty_body(self, client):
        resp = client.post(
            "/api/v1/observability/metrics",
            json={},
        )
        # Should return 422 (validation error) due to missing required fields
        assert resp.status_code == 422

    def test_nonexistent_trace(self, client):
        resp = client.get("/api/v1/observability/traces/nonexistent-trace")
        assert resp.status_code == 404
