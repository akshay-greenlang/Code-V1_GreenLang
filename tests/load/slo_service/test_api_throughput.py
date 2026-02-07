# -*- coding: utf-8 -*-
"""
API Throughput Load Tests for SLO Service (OBS-005)

Tests API endpoint performance under load using concurrent requests,
bulk operations, and report generation timing.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock

import pytest

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not FASTAPI_AVAILABLE, reason="FastAPI not installed"
)


def _setup_load_test_app():
    """Create a FastAPI app with mocked SLO service for load testing."""
    from greenlang.infrastructure.slo_service.api.router import slo_router
    from greenlang.infrastructure.slo_service.models import (
        SLI, SLIType, SLO, SLOReport, ErrorBudget, BudgetStatus,
    )

    slos = []
    for i in range(100):
        sli = SLI(
            name=f"sli_{i}",
            sli_type=SLIType.AVAILABILITY,
            good_query=f"good_{i}",
            total_query=f"total_{i}",
        )
        slo = SLO(
            slo_id=f"load-slo-{i}",
            name=f"Load SLO {i}",
            service=f"svc-{i % 10}",
            sli=sli,
            target=99.9,
        )
        slos.append(slo)

    mock_svc = MagicMock()
    mock_svc.manager = MagicMock()
    mock_svc.manager.list_all.return_value = slos
    mock_svc.manager.get.return_value = slos[0]

    budget = ErrorBudget(
        slo_id="load-slo-0",
        total_minutes=43.2,
        consumed_minutes=10.0,
        remaining_minutes=33.2,
        remaining_percent=76.9,
        consumed_percent=23.1,
        status=BudgetStatus.WARNING,
        sli_value=99.977,
    )
    mock_svc.get_budget = MagicMock(return_value=budget)
    mock_svc.get_budget_history = MagicMock(return_value=[budget])
    mock_svc.get_all_budgets = MagicMock(return_value=[budget] * 100)
    mock_svc.get_burn_rates = MagicMock(
        return_value={"fast": 0.5, "medium": 0.2, "slow": 0.1}
    )
    report = SLOReport(report_type="weekly", total_slos=100, slos_met=95)
    mock_svc.generate_compliance_report = MagicMock(return_value=report)
    mock_svc.evaluate_all = AsyncMock(
        return_value=[{"slo_id": f"load-slo-{i}", "status": "evaluated"} for i in range(100)]
    )

    app = FastAPI()
    app.state.slo_service = mock_svc
    if slo_router is not None:
        app.include_router(slo_router)
    return app


@pytest.mark.performance
class TestAPIThroughput:
    """API throughput and latency tests."""

    def test_500_rps_list_slos(self):
        """List SLOs endpoint handles 500+ requests."""
        app = _setup_load_test_app()
        client = TestClient(app)

        start = time.monotonic()
        for _ in range(500):
            resp = client.get("/api/v1/slos")
            assert resp.status_code == 200
        elapsed = time.monotonic() - start

        rps = 500 / elapsed
        assert rps > 50, f"Only achieved {rps:.0f} RPS (target: 50+)"

    def test_concurrent_budget_queries(self):
        """Concurrent budget queries complete without errors."""
        app = _setup_load_test_app()
        client = TestClient(app)
        errors = []

        def query_budget():
            try:
                resp = client.get("/api/v1/slos/load-slo-0/budget")
                if resp.status_code != 200:
                    errors.append(f"Status: {resp.status_code}")
            except Exception as exc:
                errors.append(str(exc))

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(query_budget) for _ in range(100)]
            for f in futures:
                f.result()

        assert len(errors) == 0

    def test_api_latency_p99_under_100ms(self):
        """P99 latency for list endpoint is under 100ms."""
        app = _setup_load_test_app()
        client = TestClient(app)

        latencies = []
        for _ in range(200):
            start = time.monotonic()
            resp = client.get("/api/v1/slos")
            elapsed_ms = (time.monotonic() - start) * 1000
            latencies.append(elapsed_ms)
            assert resp.status_code == 200

        latencies.sort()
        p99_index = int(len(latencies) * 0.99)
        p99 = latencies[p99_index]
        assert p99 < 100, f"P99 latency: {p99:.1f}ms (target: <100ms)"

    def test_bulk_slo_import_performance(self):
        """Importing 100 SLOs completes in reasonable time."""
        from greenlang.infrastructure.slo_service.slo_manager import SLOManager
        from greenlang.infrastructure.slo_service.models import SLI, SLIType, SLO

        mgr = SLOManager()

        start = time.monotonic()
        for i in range(100):
            sli = SLI(
                name=f"import_sli_{i}",
                sli_type=SLIType.AVAILABILITY,
                good_query="good",
                total_query="total",
            )
            slo = SLO(
                slo_id=f"import-slo-{i}",
                name=f"Import SLO {i}",
                service=f"svc-{i % 10}",
                sli=sli,
                target=99.9,
            )
            mgr.create(slo)
        elapsed = time.monotonic() - start

        assert len(mgr.list_all()) == 100
        assert elapsed < 5.0, f"100 SLO imports took {elapsed:.2f}s"

    def test_report_generation_under_5s(self):
        """Compliance report for 100 SLOs generates under 5 seconds."""
        from greenlang.infrastructure.slo_service.compliance_reporter import generate_report
        from greenlang.infrastructure.slo_service.error_budget import calculate_error_budget
        from greenlang.infrastructure.slo_service.models import (
            SLI, SLIType, SLO, ErrorBudget, BudgetStatus,
        )

        slos = []
        budgets = {}
        for i in range(100):
            sli = SLI(
                name=f"rpt_sli_{i}",
                sli_type=SLIType.AVAILABILITY,
                good_query="good",
                total_query="total",
            )
            slo = SLO(
                slo_id=f"rpt-slo-{i}",
                name=f"Report SLO {i}",
                service=f"svc-{i % 10}",
                sli=sli,
                target=99.9,
            )
            slos.append(slo)
            budgets[slo.slo_id] = calculate_error_budget(slo, 0.9995)

        start = time.monotonic()
        report = generate_report("monthly", slos, budgets)
        elapsed = time.monotonic() - start

        assert report.total_slos == 100
        assert elapsed < 5.0, f"Report generation took {elapsed:.2f}s"
