# -*- coding: utf-8 -*-
"""Tests for per-tenant SLA tracker (DEP9)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from greenlang.factors.sla import FakePromClient, SLATracker, install_sla_routes


@pytest.fixture()
def prom_good():
    return FakePromClient(
        values={
            "1 -": 0.9995,           # uptime query contains this prefix
            "histogram_quantile(0.95": 0.180,
            "histogram_quantile(0.99": 0.420,
            "code=~\"5..\"": 0.0005,  # error rate
            "sum(rate(factors_http_requests_total{tenant": 250.0,  # rps
        }
    )


def test_report_fills_all_fields(prom_good):
    tr = SLATracker(prom_good)
    to = datetime.now(timezone.utc)
    fr = to - timedelta(days=30)
    report = tr.report("acme", fr, to)
    assert 99.0 <= report.uptime_percent <= 100.0
    assert report.p95_latency_ms > 0
    assert report.p99_latency_ms >= report.p95_latency_ms
    assert "availability" in report.slo_targets
    assert isinstance(report.met["availability"], bool)


def test_uptime_failure_flips_met(prom_good):
    prom_bad = FakePromClient(
        values={
            "1 -": 0.80,
            "histogram_quantile(0.95": 2.0,
            "histogram_quantile(0.99": 5.0,
            "code=~\"5..\"": 0.2,
            "sum(rate(factors_http_requests_total": 100.0,
        }
    )
    tr = SLATracker(prom_bad)
    to = datetime.now(timezone.utc)
    fr = to - timedelta(days=1)
    report = tr.report("acme", fr, to)
    assert report.met["availability"] is False
    assert report.met["error_rate"] is False


def test_invalid_window_raises(prom_good):
    tr = SLATracker(prom_good)
    to = datetime.now(timezone.utc)
    fr = to + timedelta(days=1)
    with pytest.raises(ValueError):
        tr.report("acme", fr, to)


def test_install_routes_mounts_report_endpoint(prom_good):
    fastapi = pytest.importorskip("fastapi")
    tc = pytest.importorskip("fastapi.testclient")
    app = fastapi.FastAPI()
    install_sla_routes(app, tracker=SLATracker(prom_good))
    client = tc.TestClient(app)
    resp = client.get("/v1/sla/report", params={"tenant_id": "acme"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["tenant_id"] == "acme"
    assert "uptime_percent" in body
    assert "burn_rates" in body
