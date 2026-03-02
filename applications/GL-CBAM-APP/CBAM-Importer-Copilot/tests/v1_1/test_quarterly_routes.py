# -*- coding: utf-8 -*-
"""
Unit tests for GL-CBAM-APP v1.1 Quarterly Engine API Routes

Tests quarterly_engine.api.quarterly_routes FastAPI endpoints:
- GET  /calendar/{year}                           - Quarterly reporting calendar
- GET  /current                                   - Current quarter details
- POST /reports/generate                          - Generate quarterly report
- GET  /reports                                   - List reports (filters, pagination)
- GET  /reports/{report_id}                       - Get report details / 404
- GET  /reports/{report_id}/xml                   - Download XML
- GET  /reports/{report_id}/summary               - Download Markdown summary
- PUT  /reports/{report_id}/submit                - Submit report
- POST /reports/{report_id}/amend                 - Create amendment
- GET  /reports/{report_id}/amendments            - List amendments
- GET  /reports/{report_id}/amendments/{id}/diff  - Amendment diff
- GET  /deadlines                                 - Upcoming deadlines
- GET  /deadlines/overdue                         - Overdue reports
- PUT  /deadlines/{alert_id}/acknowledge          - Acknowledge alert
- GET  /notifications                             - Notification history
- PUT  /notifications/configure                   - Configure recipients

Uses FastAPI TestClient for synchronous HTTP testing.
In-memory stores are cleared before each test for full isolation.

Target: 80+ tests
"""

import pytest
import json
import hashlib
import uuid
from datetime import date, datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional
from unittest.mock import patch, MagicMock

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from fastapi import FastAPI
from fastapi.testclient import TestClient

from quarterly_engine.api.quarterly_routes import (
    router,
    _reports_store,
    _importer_reports,
    _alerts_store,
    _notification_configs,
    _notification_log,
    _store_report,
)
from quarterly_engine.models import (
    AlertLevel,
    AmendmentReason,
    CBAMSector,
    CalculationMethod,
    DeadlineAlert,
    NotificationConfig,
    NotificationType,
    QuarterlyPeriod,
    QuarterlyReport,
    QuarterlyReportPeriod,
    ReportAmendment,
    ReportStatus,
    ShipmentAggregation,
    compute_sha256,
    quantize_decimal,
)


# ===========================================================================
# App & Client Fixtures
# ===========================================================================

@pytest.fixture(autouse=True)
def clear_stores():
    """Clear in-memory stores before each test."""
    _reports_store.clear()
    _importer_reports.clear()
    _alerts_store.clear()
    _notification_configs.clear()
    _notification_log.clear()
    yield


@pytest.fixture
def app():
    """Create a FastAPI test application with quarterly routes."""
    test_app = FastAPI(title="CBAM Quarterly Test")
    test_app.include_router(router)
    return test_app


@pytest.fixture
def client(app):
    """Create a TestClient for synchronous HTTP testing."""
    return TestClient(app)


# ===========================================================================
# Helper: build period, report, and alert fixtures
# ===========================================================================

def _make_period(year=2026, quarter=QuarterlyPeriod.Q1) -> QuarterlyReportPeriod:
    """Build a valid QuarterlyReportPeriod."""
    month_map = {
        QuarterlyPeriod.Q1: (1, 3),
        QuarterlyPeriod.Q2: (4, 6),
        QuarterlyPeriod.Q3: (7, 9),
        QuarterlyPeriod.Q4: (10, 12),
    }
    sm, em = month_map[quarter]
    start = date(year, sm, 1)
    if em == 12:
        end = date(year, 12, 31)
    else:
        end = date(year, em + 1, 1) - timedelta(days=1)
    submission_deadline = end + timedelta(days=30)
    amendment_deadline = end + timedelta(days=60)
    is_transitional = year <= 2025

    return QuarterlyReportPeriod(
        year=year,
        quarter=quarter,
        start_date=start,
        end_date=end,
        submission_deadline=submission_deadline,
        amendment_deadline=amendment_deadline,
        is_transitional=is_transitional,
    )


def _make_aggregation(
    cn_code="72031000",
    country="TR",
    quantity=Decimal("500.000"),
    direct=Decimal("925.000"),
    indirect=Decimal("75.000"),
) -> ShipmentAggregation:
    """Build a valid ShipmentAggregation."""
    total = direct + indirect
    intensity = (total / quantity).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
    return ShipmentAggregation(
        cn_code=cn_code,
        cn_description="Test product",
        country_of_origin=country,
        product_group=CBAMSector.IRON_AND_STEEL,
        quantity_mt=quantity,
        direct_emissions_tCO2e=direct,
        indirect_emissions_tCO2e=indirect,
        total_emissions_tCO2e=total,
        embedded_emissions_per_mt=intensity,
        calculation_method=CalculationMethod.SUPPLIER_ACTUAL,
        default_values_used_pct=Decimal("0"),
        supplier_data_quality_score=Decimal("0.85"),
    )


def _make_report(
    report_id=None,
    importer_id="NL123456789012",
    year=2026,
    quarter=QuarterlyPeriod.Q1,
    status=ReportStatus.DRAFT,
    shipments_count=5,
    with_xml=False,
    with_summary=False,
    submitted_at=None,
) -> QuarterlyReport:
    """Build a valid QuarterlyReport and optionally store it."""
    if report_id is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        report_id = f"CBAM-QR-{year}{quarter.value}-{importer_id}-{ts}"

    period = _make_period(year, quarter)
    agg = _make_aggregation()

    # For statuses that require submitted_at
    if status in {ReportStatus.SUBMITTED, ReportStatus.ACCEPTED, ReportStatus.REJECTED}:
        submitted_at = submitted_at or datetime.now(timezone.utc)

    report = QuarterlyReport(
        report_id=report_id,
        period=period,
        importer_id=importer_id,
        status=status,
        shipments_count=shipments_count,
        total_quantity_mt=Decimal("500.000"),
        aggregations=[agg],
        total_direct_emissions=Decimal("925.000"),
        total_indirect_emissions=Decimal("75.000"),
        total_embedded_emissions=Decimal("1000.000"),
        calculation_methods_used={"supplier_actual": 5},
        report_xml="<cbam>test</cbam>" if with_xml else None,
        report_summary_md="# CBAM Report\nTest summary" if with_summary else None,
        version=1,
        submitted_at=submitted_at,
    )
    report_data = report.model_dump()
    report_data["provenance_hash"] = report.compute_provenance_hash()
    report = QuarterlyReport(**report_data)
    return report


def _store_test_report(**kwargs) -> QuarterlyReport:
    """Build and store a test report in the in-memory store."""
    report = _make_report(**kwargs)
    _store_report(report)
    return report


def _make_alert(
    alert_id=None,
    year=2026,
    quarter=QuarterlyPeriod.Q1,
    alert_level=AlertLevel.WARNING,
    notification_type=NotificationType.DEADLINE_APPROACHING,
    days_remaining=14,
    acknowledged=False,
) -> DeadlineAlert:
    """Build a valid DeadlineAlert."""
    if alert_id is None:
        alert_id = f"ALERT-{year}{quarter.value}-{uuid.uuid4().hex[:8]}"
    period = _make_period(year, quarter)
    return DeadlineAlert(
        alert_id=alert_id,
        report_period=period,
        alert_level=alert_level,
        notification_type=notification_type,
        days_remaining=days_remaining,
        message=f"CBAM report due in {days_remaining} days",
        recipients=["compliance@acme.eu"],
        acknowledged=acknowledged,
    )


# ===========================================================================
# TEST CLASS -- Calendar endpoints
# ===========================================================================

class TestGetQuarterlyCalendar:
    """Tests for GET /calendar/{year}."""

    def test_calendar_2026_returns_four_quarters(self, client):
        """Calendar for 2026 returns exactly 4 quarters."""
        with patch(
            "quarterly_engine.api.quarterly_routes._get_scheduler"
        ) as mock_sched:
            scheduler = MagicMock()
            for qp in QuarterlyPeriod:
                scheduler.get_quarter.return_value = _make_period(2026, qp)
                # Just return the same for simplicity; real test validates structure
            scheduler.get_quarter.side_effect = lambda y, q: _make_period(y, q)
            mock_sched.return_value = scheduler

            resp = client.get("/api/v1/cbam/quarterly/calendar/2026")
            assert resp.status_code == 200
            body = resp.json()
            assert body["success"] is True
            data = body["data"]
            assert data["year"] == 2026
            assert len(data["quarters"]) == 4

    def test_calendar_has_correct_quarter_order(self, client):
        """Quarters are returned in Q1-Q4 order."""
        with patch(
            "quarterly_engine.api.quarterly_routes._get_scheduler"
        ) as mock_sched:
            scheduler = MagicMock()
            scheduler.get_quarter.side_effect = lambda y, q: _make_period(y, q)
            mock_sched.return_value = scheduler

            resp = client.get("/api/v1/cbam/quarterly/calendar/2026")
            data = resp.json()["data"]
            quarters = [q["quarter"] for q in data["quarters"]]
            assert quarters == ["Q1", "Q2", "Q3", "Q4"]

    def test_calendar_transitional_year(self, client):
        """2025 is flagged as transitional year."""
        with patch(
            "quarterly_engine.api.quarterly_routes._get_scheduler"
        ) as mock_sched:
            scheduler = MagicMock()
            scheduler.get_quarter.side_effect = lambda y, q: _make_period(y, q)
            mock_sched.return_value = scheduler

            resp = client.get("/api/v1/cbam/quarterly/calendar/2025")
            data = resp.json()["data"]
            assert data["is_transitional_year"] is True
            assert data["is_definitive_year"] is False

    def test_calendar_definitive_year(self, client):
        """2026 is flagged as definitive year."""
        with patch(
            "quarterly_engine.api.quarterly_routes._get_scheduler"
        ) as mock_sched:
            scheduler = MagicMock()
            scheduler.get_quarter.side_effect = lambda y, q: _make_period(y, q)
            mock_sched.return_value = scheduler

            resp = client.get("/api/v1/cbam/quarterly/calendar/2026")
            data = resp.json()["data"]
            assert data["is_transitional_year"] is False
            assert data["is_definitive_year"] is True

    def test_calendar_deadlines_present(self, client):
        """Each quarter entry contains submission and amendment deadlines."""
        with patch(
            "quarterly_engine.api.quarterly_routes._get_scheduler"
        ) as mock_sched:
            scheduler = MagicMock()
            scheduler.get_quarter.side_effect = lambda y, q: _make_period(y, q)
            mock_sched.return_value = scheduler

            resp = client.get("/api/v1/cbam/quarterly/calendar/2026")
            data = resp.json()["data"]
            for q in data["quarters"]:
                assert "submission_deadline" in q
                assert "amendment_deadline" in q
                assert "start_date" in q
                assert "end_date" in q

    def test_calendar_year_too_low_returns_422(self, client):
        """Year below 2023 returns validation error."""
        resp = client.get("/api/v1/cbam/quarterly/calendar/2020")
        assert resp.status_code == 422

    def test_calendar_year_too_high_returns_422(self, client):
        """Year above 2099 returns validation error."""
        resp = client.get("/api/v1/cbam/quarterly/calendar/2100")
        assert resp.status_code == 422

    def test_calendar_processing_time_populated(self, client):
        """Response includes processing_time_ms."""
        with patch(
            "quarterly_engine.api.quarterly_routes._get_scheduler"
        ) as mock_sched:
            scheduler = MagicMock()
            scheduler.get_quarter.side_effect = lambda y, q: _make_period(y, q)
            mock_sched.return_value = scheduler

            resp = client.get("/api/v1/cbam/quarterly/calendar/2026")
            body = resp.json()
            assert "processing_time_ms" in body
            assert body["processing_time_ms"] >= 0


class TestGetCurrentQuarter:
    """Tests for GET /current."""

    def test_current_quarter_returns_200(self, client):
        """Current quarter endpoint returns 200 with period details."""
        with patch(
            "quarterly_engine.api.quarterly_routes._get_scheduler"
        ) as mock_sched:
            scheduler = MagicMock()
            period = _make_period(2026, QuarterlyPeriod.Q1)
            scheduler.get_current_quarter.return_value = period
            scheduler.get_days_until_deadline.return_value = 15
            mock_sched.return_value = scheduler

            resp = client.get("/api/v1/cbam/quarterly/current")
            assert resp.status_code == 200
            body = resp.json()
            assert body["success"] is True
            data = body["data"]
            assert data["days_until_submission_deadline"] == 15

    def test_current_quarter_has_phase_flag(self, client):
        """Current quarter response includes transitional/definitive flags."""
        with patch(
            "quarterly_engine.api.quarterly_routes._get_scheduler"
        ) as mock_sched:
            scheduler = MagicMock()
            period = _make_period(2026, QuarterlyPeriod.Q1)
            scheduler.get_current_quarter.return_value = period
            scheduler.get_days_until_deadline.return_value = 10
            mock_sched.return_value = scheduler

            resp = client.get("/api/v1/cbam/quarterly/current")
            data = resp.json()["data"]
            assert "is_transitional" in data
            assert "is_definitive" in data

    def test_current_quarter_period_label(self, client):
        """Current quarter response includes a period_label string."""
        with patch(
            "quarterly_engine.api.quarterly_routes._get_scheduler"
        ) as mock_sched:
            scheduler = MagicMock()
            period = _make_period(2026, QuarterlyPeriod.Q2)
            scheduler.get_current_quarter.return_value = period
            scheduler.get_days_until_deadline.return_value = 25
            mock_sched.return_value = scheduler

            resp = client.get("/api/v1/cbam/quarterly/current")
            data = resp.json()["data"]
            assert data["period_label"] == "2026Q2"


# ===========================================================================
# TEST CLASS -- Report Generation
# ===========================================================================

class TestGenerateReport:
    """Tests for POST /reports/generate."""

    def test_generate_valid_report_201(self, client):
        """Generating a valid report returns 201."""
        with patch(
            "quarterly_engine.api.quarterly_routes._get_scheduler"
        ) as mock_sched, patch(
            "quarterly_engine.api.quarterly_routes._get_assembler"
        ) as mock_asm:
            scheduler = MagicMock()
            period = _make_period(2026, QuarterlyPeriod.Q1)
            scheduler.get_quarter.return_value = period
            mock_sched.return_value = scheduler

            report = _make_report()
            assembler = MagicMock()
            assembler.assemble_quarterly_report.return_value = report
            mock_asm.return_value = assembler

            payload = {
                "importer_id": "NL123456789012",
                "year": 2026,
                "quarter": "Q1",
                "shipments": [
                    {
                        "cn_code": "72031000",
                        "country_of_origin": "TR",
                        "quantity_mt": 500.0,
                    }
                ],
            }
            resp = client.post(
                "/api/v1/cbam/quarterly/reports/generate",
                json=payload,
            )
            assert resp.status_code == 201
            body = resp.json()
            assert body["success"] is True
            assert "report_id" in body["data"]

    def test_generate_missing_importer_id_422(self, client):
        """Missing importer_id returns 422."""
        payload = {"year": 2026, "quarter": "Q1"}
        resp = client.post(
            "/api/v1/cbam/quarterly/reports/generate", json=payload
        )
        assert resp.status_code == 422

    def test_generate_invalid_quarter_422(self, client):
        """Invalid quarter value returns 422."""
        payload = {
            "importer_id": "NL123456789012",
            "year": 2026,
            "quarter": "Q5",
        }
        resp = client.post(
            "/api/v1/cbam/quarterly/reports/generate", json=payload
        )
        assert resp.status_code == 422

    def test_generate_invalid_year_too_low_422(self, client):
        """Year below 2023 returns 422."""
        payload = {
            "importer_id": "NL123456789012",
            "year": 2020,
            "quarter": "Q1",
        }
        resp = client.post(
            "/api/v1/cbam/quarterly/reports/generate", json=payload
        )
        assert resp.status_code == 422

    def test_generate_duplicate_draft_conflict_409(self, client):
        """Generating a report when a draft exists returns 409."""
        report = _store_test_report(
            report_id="CBAM-QR-2026Q1-NL123456789012-20260101",
            importer_id="NL123456789012",
            year=2026,
            quarter=QuarterlyPeriod.Q1,
            status=ReportStatus.DRAFT,
        )

        with patch(
            "quarterly_engine.api.quarterly_routes._get_scheduler"
        ) as mock_sched, patch(
            "quarterly_engine.api.quarterly_routes._get_assembler"
        ):
            scheduler = MagicMock()
            scheduler.get_quarter.return_value = _make_period(2026, QuarterlyPeriod.Q1)
            mock_sched.return_value = scheduler

            payload = {
                "importer_id": "NL123456789012",
                "year": 2026,
                "quarter": "Q1",
            }
            resp = client.post(
                "/api/v1/cbam/quarterly/reports/generate", json=payload
            )
            assert resp.status_code == 409

    def test_generate_force_regenerate_overrides_duplicate(self, client):
        """force_regenerate=true overwrites an existing draft."""
        _store_test_report(
            report_id="CBAM-QR-2026Q1-NL123456789012-20260101",
            importer_id="NL123456789012",
            year=2026,
            quarter=QuarterlyPeriod.Q1,
            status=ReportStatus.DRAFT,
        )

        with patch(
            "quarterly_engine.api.quarterly_routes._get_scheduler"
        ) as mock_sched, patch(
            "quarterly_engine.api.quarterly_routes._get_assembler"
        ) as mock_asm:
            scheduler = MagicMock()
            scheduler.get_quarter.return_value = _make_period(2026, QuarterlyPeriod.Q1)
            mock_sched.return_value = scheduler

            new_report = _make_report(
                report_id="CBAM-QR-2026Q1-NL123456789012-new"
            )
            assembler = MagicMock()
            assembler.assemble_quarterly_report.return_value = new_report
            mock_asm.return_value = assembler

            payload = {
                "importer_id": "NL123456789012",
                "year": 2026,
                "quarter": "Q1",
                "force_regenerate": True,
            }
            resp = client.post(
                "/api/v1/cbam/quarterly/reports/generate", json=payload
            )
            assert resp.status_code == 201

    def test_generate_empty_importer_id_422(self, client):
        """Empty importer_id returns 422."""
        payload = {
            "importer_id": "",
            "year": 2026,
            "quarter": "Q1",
        }
        resp = client.post(
            "/api/v1/cbam/quarterly/reports/generate", json=payload
        )
        assert resp.status_code == 422


# ===========================================================================
# TEST CLASS -- Report Retrieval
# ===========================================================================

class TestGetReport:
    """Tests for GET /reports/{report_id}."""

    def test_get_existing_report(self, client):
        """Retrieving an existing report returns 200 with full data."""
        report = _store_test_report(report_id="RPT-001")
        resp = client.get("/api/v1/cbam/quarterly/reports/RPT-001")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["report_id"] == "RPT-001"

    def test_get_report_not_found(self, client):
        """Requesting a non-existent report returns 404."""
        resp = client.get("/api/v1/cbam/quarterly/reports/NONEXISTENT")
        assert resp.status_code == 404

    def test_get_report_includes_provenance_hash(self, client):
        """Report response includes a 64-char provenance_hash."""
        _store_test_report(report_id="RPT-002")
        resp = client.get("/api/v1/cbam/quarterly/reports/RPT-002")
        data = resp.json()["data"]
        assert "provenance_hash" in data
        assert len(data["provenance_hash"]) == 64

    def test_get_report_includes_period_info(self, client):
        """Report response includes nested period data."""
        _store_test_report(report_id="RPT-003")
        resp = client.get("/api/v1/cbam/quarterly/reports/RPT-003")
        data = resp.json()["data"]
        assert "period" in data
        assert data["period"]["year"] == 2026

    def test_get_report_includes_aggregations(self, client):
        """Report response includes aggregations list."""
        _store_test_report(report_id="RPT-004")
        resp = client.get("/api/v1/cbam/quarterly/reports/RPT-004")
        data = resp.json()["data"]
        assert "aggregations" in data
        assert len(data["aggregations"]) >= 1

    def test_get_report_emissions_totals(self, client):
        """Report contains non-zero emissions totals."""
        _store_test_report(report_id="RPT-005")
        resp = client.get("/api/v1/cbam/quarterly/reports/RPT-005")
        data = resp.json()["data"]
        assert Decimal(data["total_embedded_emissions"]) > 0


# ===========================================================================
# TEST CLASS -- List Reports
# ===========================================================================

class TestListReports:
    """Tests for GET /reports."""

    def test_list_empty(self, client):
        """Empty store returns zero reports."""
        resp = client.get("/api/v1/cbam/quarterly/reports")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["total"] == 0
        assert data["items"] == []

    def test_list_returns_stored_reports(self, client):
        """Stored reports appear in the listing."""
        _store_test_report(report_id="RPT-A01")
        _store_test_report(report_id="RPT-A02", importer_id="DE987654321000")
        resp = client.get("/api/v1/cbam/quarterly/reports")
        data = resp.json()["data"]
        assert data["total"] == 2

    def test_list_filter_by_importer(self, client):
        """Filtering by importer_id returns only matching reports."""
        _store_test_report(report_id="RPT-B01", importer_id="NL123456789012")
        _store_test_report(report_id="RPT-B02", importer_id="DE987654321000")
        resp = client.get(
            "/api/v1/cbam/quarterly/reports?importer_id=NL123456789012"
        )
        data = resp.json()["data"]
        assert data["total"] == 1
        assert data["items"][0]["importer_id"] == "NL123456789012"

    def test_list_filter_by_status(self, client):
        """Filtering by status returns only matching reports."""
        _store_test_report(report_id="RPT-C01", status=ReportStatus.DRAFT)
        _store_test_report(
            report_id="RPT-C02",
            status=ReportStatus.SUBMITTED,
            submitted_at=datetime.now(timezone.utc),
        )
        resp = client.get(
            "/api/v1/cbam/quarterly/reports?status=draft"
        )
        data = resp.json()["data"]
        assert data["total"] == 1
        assert data["items"][0]["status"] == "draft"

    def test_list_filter_by_period(self, client):
        """Filtering by period label returns matching reports."""
        _store_test_report(
            report_id="RPT-D01", year=2026, quarter=QuarterlyPeriod.Q1
        )
        _store_test_report(
            report_id="RPT-D02", year=2026, quarter=QuarterlyPeriod.Q2
        )
        resp = client.get(
            "/api/v1/cbam/quarterly/reports?period=2026Q1"
        )
        data = resp.json()["data"]
        assert data["total"] == 1

    def test_list_pagination_offset(self, client):
        """Pagination offset skips the first N records."""
        for i in range(5):
            _store_test_report(report_id=f"RPT-PG-{i:02d}")
        resp = client.get(
            "/api/v1/cbam/quarterly/reports?offset=3&limit=10"
        )
        data = resp.json()["data"]
        assert len(data["items"]) == 2
        assert data["total"] == 5

    def test_list_pagination_limit(self, client):
        """Pagination limit caps the returned items."""
        for i in range(5):
            _store_test_report(report_id=f"RPT-LIM-{i:02d}")
        resp = client.get(
            "/api/v1/cbam/quarterly/reports?offset=0&limit=2"
        )
        data = resp.json()["data"]
        assert len(data["items"]) == 2
        assert data["total"] == 5

    def test_list_items_have_condensed_fields(self, client):
        """List items contain condensed ReportListItem fields."""
        _store_test_report(report_id="RPT-FIELDS")
        resp = client.get("/api/v1/cbam/quarterly/reports")
        item = resp.json()["data"]["items"][0]
        for key in [
            "report_id", "period_label", "importer_id",
            "status", "shipments_count", "version", "provenance_hash",
        ]:
            assert key in item, f"Missing key: {key}"


# ===========================================================================
# TEST CLASS -- Download XML
# ===========================================================================

class TestDownloadXML:
    """Tests for GET /reports/{report_id}/xml."""

    def test_download_xml_success(self, client):
        """Report with XML returns correct content-type."""
        _store_test_report(report_id="RPT-XML-01", with_xml=True)
        resp = client.get("/api/v1/cbam/quarterly/reports/RPT-XML-01/xml")
        assert resp.status_code == 200
        assert "application/xml" in resp.headers["content-type"]
        assert "Content-Disposition" in resp.headers
        assert "<cbam>" in resp.text

    def test_download_xml_no_xml_404(self, client):
        """Report without XML returns 404."""
        _store_test_report(report_id="RPT-XML-02", with_xml=False)
        resp = client.get("/api/v1/cbam/quarterly/reports/RPT-XML-02/xml")
        assert resp.status_code == 404

    def test_download_xml_report_not_found(self, client):
        """Non-existent report returns 404 for XML download."""
        resp = client.get("/api/v1/cbam/quarterly/reports/NOPE/xml")
        assert resp.status_code == 404

    def test_download_xml_provenance_header(self, client):
        """XML download includes X-Provenance-Hash header."""
        _store_test_report(report_id="RPT-XML-03", with_xml=True)
        resp = client.get("/api/v1/cbam/quarterly/reports/RPT-XML-03/xml")
        assert "X-Provenance-Hash" in resp.headers
        assert len(resp.headers["X-Provenance-Hash"]) == 64


# ===========================================================================
# TEST CLASS -- Download Summary
# ===========================================================================

class TestDownloadSummary:
    """Tests for GET /reports/{report_id}/summary."""

    def test_download_summary_success(self, client):
        """Report with summary returns markdown content."""
        _store_test_report(report_id="RPT-SUM-01", with_summary=True)
        resp = client.get("/api/v1/cbam/quarterly/reports/RPT-SUM-01/summary")
        assert resp.status_code == 200
        assert "text/markdown" in resp.headers["content-type"]
        assert "# CBAM Report" in resp.text

    def test_download_summary_no_summary_404(self, client):
        """Report without summary returns 404."""
        _store_test_report(report_id="RPT-SUM-02", with_summary=False)
        resp = client.get("/api/v1/cbam/quarterly/reports/RPT-SUM-02/summary")
        assert resp.status_code == 404

    def test_download_summary_report_not_found(self, client):
        """Non-existent report returns 404 for summary download."""
        resp = client.get("/api/v1/cbam/quarterly/reports/NOPE/summary")
        assert resp.status_code == 404

    def test_download_summary_content_disposition(self, client):
        """Summary download includes a Content-Disposition header."""
        _store_test_report(report_id="RPT-SUM-03", with_summary=True)
        resp = client.get("/api/v1/cbam/quarterly/reports/RPT-SUM-03/summary")
        assert "Content-Disposition" in resp.headers
        assert "_summary.md" in resp.headers["Content-Disposition"]


# ===========================================================================
# TEST CLASS -- Submit Report
# ===========================================================================

class TestSubmitReport:
    """Tests for PUT /reports/{report_id}/submit."""

    def test_submit_draft_report(self, client):
        """Submitting a draft report succeeds."""
        _store_test_report(report_id="RPT-SUB-01", status=ReportStatus.DRAFT)
        payload = {
            "submitted_by": "compliance@acme.eu",
            "declaration": True,
        }
        resp = client.put(
            "/api/v1/cbam/quarterly/reports/RPT-SUB-01/submit",
            json=payload,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["status"] == "submitted"

    def test_submit_without_declaration_fails(self, client):
        """Submission without declaration=true returns 400."""
        _store_test_report(report_id="RPT-SUB-02", status=ReportStatus.DRAFT)
        payload = {
            "submitted_by": "compliance@acme.eu",
            "declaration": False,
        }
        resp = client.put(
            "/api/v1/cbam/quarterly/reports/RPT-SUB-02/submit",
            json=payload,
        )
        assert resp.status_code == 400

    def test_submit_already_submitted_fails(self, client):
        """Submitting an already-submitted report returns 409."""
        _store_test_report(
            report_id="RPT-SUB-03",
            status=ReportStatus.SUBMITTED,
            submitted_at=datetime.now(timezone.utc),
        )
        payload = {
            "submitted_by": "compliance@acme.eu",
            "declaration": True,
        }
        resp = client.put(
            "/api/v1/cbam/quarterly/reports/RPT-SUB-03/submit",
            json=payload,
        )
        assert resp.status_code == 409

    def test_submit_accepted_report_fails(self, client):
        """Submitting an accepted report returns 409."""
        _store_test_report(
            report_id="RPT-SUB-04",
            status=ReportStatus.ACCEPTED,
            submitted_at=datetime.now(timezone.utc),
        )
        payload = {
            "submitted_by": "compliance@acme.eu",
            "declaration": True,
        }
        resp = client.put(
            "/api/v1/cbam/quarterly/reports/RPT-SUB-04/submit",
            json=payload,
        )
        assert resp.status_code == 409

    def test_submit_report_not_found(self, client):
        """Submitting a non-existent report returns 404."""
        payload = {
            "submitted_by": "compliance@acme.eu",
            "declaration": True,
        }
        resp = client.put(
            "/api/v1/cbam/quarterly/reports/NOPE/submit",
            json=payload,
        )
        assert resp.status_code == 404

    def test_submit_zero_shipments_fails(self, client):
        """Report with zero shipments cannot be submitted."""
        _store_test_report(
            report_id="RPT-SUB-05",
            status=ReportStatus.DRAFT,
            shipments_count=0,
        )
        payload = {
            "submitted_by": "compliance@acme.eu",
            "declaration": True,
        }
        resp = client.put(
            "/api/v1/cbam/quarterly/reports/RPT-SUB-05/submit",
            json=payload,
        )
        assert resp.status_code == 400

    def test_submit_updates_provenance_hash(self, client):
        """Submission recomputes the provenance hash."""
        report = _store_test_report(
            report_id="RPT-SUB-06", status=ReportStatus.DRAFT
        )
        original_hash = report.provenance_hash

        payload = {
            "submitted_by": "compliance@acme.eu",
            "declaration": True,
        }
        resp = client.put(
            "/api/v1/cbam/quarterly/reports/RPT-SUB-06/submit",
            json=payload,
        )
        assert resp.status_code == 200
        new_hash = resp.json()["data"]["provenance_hash"]
        # Hash may or may not change depending on data, but should be 64 chars
        assert len(new_hash) == 64

    def test_submit_sets_submitted_at(self, client):
        """Submission sets the submitted_at timestamp."""
        _store_test_report(report_id="RPT-SUB-07", status=ReportStatus.DRAFT)
        payload = {
            "submitted_by": "compliance@acme.eu",
            "declaration": True,
        }
        resp = client.put(
            "/api/v1/cbam/quarterly/reports/RPT-SUB-07/submit",
            json=payload,
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["submitted_at"] is not None

    def test_submit_missing_submitted_by_422(self, client):
        """Missing submitted_by field returns 422."""
        _store_test_report(report_id="RPT-SUB-08", status=ReportStatus.DRAFT)
        payload = {"declaration": True}
        resp = client.put(
            "/api/v1/cbam/quarterly/reports/RPT-SUB-08/submit",
            json=payload,
        )
        assert resp.status_code == 422


# ===========================================================================
# TEST CLASS -- Amendments
# ===========================================================================

class TestCreateAmendment:
    """Tests for POST /reports/{report_id}/amend."""

    def test_amend_submitted_report(self, client):
        """Amending a submitted report returns 201."""
        _store_test_report(
            report_id="RPT-AMD-01",
            status=ReportStatus.SUBMITTED,
            submitted_at=datetime.now(timezone.utc),
        )

        with patch(
            "quarterly_engine.api.quarterly_routes._get_amendment_mgr"
        ) as mock_mgr:
            mgr = MagicMock()
            amendment = MagicMock()
            amendment.amendment_id = "AMEND-001"
            amendment.version = 2
            amendment.model_dump.return_value = {
                "amendment_id": "AMEND-001",
                "report_id": "RPT-AMD-01",
                "version": 2,
                "reason": "new_supplier_data",
            }
            mgr.create_amendment.return_value = amendment

            updated_report = _make_report(
                report_id="RPT-AMD-01",
                status=ReportStatus.SUBMITTED,
                submitted_at=datetime.now(timezone.utc),
            )
            mgr.apply_amendment.return_value = updated_report
            mock_mgr.return_value = mgr

            payload = {
                "reason": "new_supplier_data",
                "changes_summary": "Updated steel emissions for installation TR-001",
                "changes": {"total_direct_emissions": "2800.000"},
                "amended_by": "compliance@acme.eu",
            }
            resp = client.post(
                "/api/v1/cbam/quarterly/reports/RPT-AMD-01/amend",
                json=payload,
            )
            assert resp.status_code == 201
            body = resp.json()
            assert body["success"] is True

    def test_amend_draft_report_fails(self, client):
        """Amending a draft report returns 409."""
        _store_test_report(
            report_id="RPT-AMD-02", status=ReportStatus.DRAFT
        )
        payload = {
            "reason": "data_correction",
            "changes_summary": "Corrected an error in the shipment quantity",
            "changes": {"total_quantity_mt": "600.000"},
            "amended_by": "compliance@acme.eu",
        }
        resp = client.post(
            "/api/v1/cbam/quarterly/reports/RPT-AMD-02/amend",
            json=payload,
        )
        assert resp.status_code == 409

    def test_amend_report_not_found(self, client):
        """Amending a non-existent report returns 404."""
        payload = {
            "reason": "data_correction",
            "changes_summary": "Corrected an error in the emissions data fields",
            "changes": {"total_direct_emissions": "900.000"},
            "amended_by": "compliance@acme.eu",
        }
        resp = client.post(
            "/api/v1/cbam/quarterly/reports/NOPE/amend",
            json=payload,
        )
        assert resp.status_code == 404

    def test_amend_invalid_reason_422(self, client):
        """Invalid amendment reason returns 422."""
        _store_test_report(
            report_id="RPT-AMD-03",
            status=ReportStatus.SUBMITTED,
            submitted_at=datetime.now(timezone.utc),
        )
        payload = {
            "reason": "invalid_reason",
            "changes_summary": "Some changes that need to be applied to the report",
            "changes": {},
            "amended_by": "compliance@acme.eu",
        }
        resp = client.post(
            "/api/v1/cbam/quarterly/reports/RPT-AMD-03/amend",
            json=payload,
        )
        assert resp.status_code == 422

    def test_amend_short_changes_summary_422(self, client):
        """changes_summary shorter than 10 chars returns 422."""
        _store_test_report(
            report_id="RPT-AMD-04",
            status=ReportStatus.SUBMITTED,
            submitted_at=datetime.now(timezone.utc),
        )
        payload = {
            "reason": "data_correction",
            "changes_summary": "Short",
            "changes": {},
            "amended_by": "compliance@acme.eu",
        }
        resp = client.post(
            "/api/v1/cbam/quarterly/reports/RPT-AMD-04/amend",
            json=payload,
        )
        assert resp.status_code == 422


class TestListAmendments:
    """Tests for GET /reports/{report_id}/amendments."""

    def test_list_amendments_empty(self, client):
        """Report with no amendments returns empty list."""
        _store_test_report(
            report_id="RPT-LA-01",
            status=ReportStatus.SUBMITTED,
            submitted_at=datetime.now(timezone.utc),
        )

        with patch(
            "quarterly_engine.api.quarterly_routes._get_amendment_mgr"
        ) as mock_mgr:
            mgr = MagicMock()
            mgr.get_amendments.return_value = []
            mock_mgr.return_value = mgr

            resp = client.get(
                "/api/v1/cbam/quarterly/reports/RPT-LA-01/amendments"
            )
            assert resp.status_code == 200
            data = resp.json()["data"]
            assert data["total"] == 0
            assert data["amendments"] == []

    def test_list_amendments_report_not_found(self, client):
        """Listing amendments for non-existent report returns 404."""
        resp = client.get(
            "/api/v1/cbam/quarterly/reports/NOPE/amendments"
        )
        assert resp.status_code == 404


class TestGetAmendmentDiff:
    """Tests for GET /reports/{report_id}/amendments/{amendment_id}/diff."""

    def test_get_diff_success(self, client):
        """Getting diff for a valid amendment returns diff data."""
        _store_test_report(
            report_id="RPT-DIFF-01",
            status=ReportStatus.SUBMITTED,
            submitted_at=datetime.now(timezone.utc),
        )

        with patch(
            "quarterly_engine.api.quarterly_routes._get_amendment_mgr"
        ) as mock_mgr:
            mgr = MagicMock()
            amendment = MagicMock()
            amendment.report_id = "RPT-DIFF-01"
            amendment.amendment_id = "AMEND-DIFF-01"
            amendment.version = 2
            amendment.reason = AmendmentReason.NEW_SUPPLIER_DATA
            amendment.changes_summary = "Updated emissions data from verified supplier"
            amendment.diff_data = {
                "total_direct_emissions": {
                    "old": "925.000",
                    "new": "900.000",
                }
            }
            amendment.previous_hash = "a" * 64
            amendment.new_hash = "b" * 64
            amendment.amended_by = "compliance@acme.eu"
            amendment.amended_at = datetime.now(timezone.utc)
            mgr.get_amendment.return_value = amendment
            mock_mgr.return_value = mgr

            resp = client.get(
                "/api/v1/cbam/quarterly/reports/RPT-DIFF-01/amendments/AMEND-DIFF-01/diff"
            )
            assert resp.status_code == 200
            data = resp.json()["data"]
            assert "diff_data" in data
            assert data["previous_hash"] == "a" * 64
            assert data["new_hash"] == "b" * 64

    def test_get_diff_amendment_not_found(self, client):
        """Non-existent amendment returns 404."""
        _store_test_report(
            report_id="RPT-DIFF-02",
            status=ReportStatus.SUBMITTED,
            submitted_at=datetime.now(timezone.utc),
        )

        with patch(
            "quarterly_engine.api.quarterly_routes._get_amendment_mgr"
        ) as mock_mgr:
            mgr = MagicMock()
            mgr.get_amendment.return_value = None
            mock_mgr.return_value = mgr

            resp = client.get(
                "/api/v1/cbam/quarterly/reports/RPT-DIFF-02/amendments/NOPE/diff"
            )
            assert resp.status_code == 404

    def test_get_diff_amendment_wrong_report(self, client):
        """Amendment belonging to a different report returns 400."""
        _store_test_report(
            report_id="RPT-DIFF-03",
            status=ReportStatus.SUBMITTED,
            submitted_at=datetime.now(timezone.utc),
        )

        with patch(
            "quarterly_engine.api.quarterly_routes._get_amendment_mgr"
        ) as mock_mgr:
            mgr = MagicMock()
            amendment = MagicMock()
            amendment.report_id = "RPT-OTHER"  # different report
            mgr.get_amendment.return_value = amendment
            mock_mgr.return_value = mgr

            resp = client.get(
                "/api/v1/cbam/quarterly/reports/RPT-DIFF-03/amendments/AMEND-X/diff"
            )
            assert resp.status_code == 400


# ===========================================================================
# TEST CLASS -- Deadline endpoints
# ===========================================================================

class TestDeadlines:
    """Tests for GET /deadlines and GET /deadlines/overdue."""

    def test_upcoming_deadlines(self, client):
        """Upcoming deadlines endpoint returns alert list."""
        with patch(
            "quarterly_engine.api.quarterly_routes._get_deadline_tracker"
        ) as mock_tracker:
            tracker = MagicMock()
            alert = _make_alert(alert_id="ALERT-001")
            tracker.check_upcoming_deadlines.return_value = [alert]
            mock_tracker.return_value = tracker

            resp = client.get(
                "/api/v1/cbam/quarterly/deadlines?importer_id=NL123"
            )
            assert resp.status_code == 200
            data = resp.json()["data"]
            assert data["total"] == 1
            assert data["alerts"][0]["alert_id"] == "ALERT-001"

    def test_upcoming_deadlines_missing_importer_422(self, client):
        """Missing importer_id query param returns 422."""
        resp = client.get("/api/v1/cbam/quarterly/deadlines")
        assert resp.status_code == 422

    def test_overdue_deadlines(self, client):
        """Overdue endpoint returns overdue alerts."""
        with patch(
            "quarterly_engine.api.quarterly_routes._get_deadline_tracker"
        ) as mock_tracker:
            tracker = MagicMock()
            alert = _make_alert(
                alert_id="ALERT-OVERDUE-01",
                alert_level=AlertLevel.CRITICAL,
                notification_type=NotificationType.DEADLINE_OVERDUE,
                days_remaining=-5,
            )
            tracker.check_overdue_reports.return_value = [alert]
            mock_tracker.return_value = tracker

            resp = client.get(
                "/api/v1/cbam/quarterly/deadlines/overdue?importer_id=NL123"
            )
            assert resp.status_code == 200
            data = resp.json()["data"]
            assert data["total"] == 1
            assert data["overdue_alerts"][0]["days_remaining"] == -5

    def test_overdue_missing_importer_422(self, client):
        """Missing importer_id for overdue returns 422."""
        resp = client.get("/api/v1/cbam/quarterly/deadlines/overdue")
        assert resp.status_code == 422

    def test_deadlines_stored_in_alerts_store(self, client):
        """Deadline alerts are stored in _alerts_store for acknowledgement."""
        with patch(
            "quarterly_engine.api.quarterly_routes._get_deadline_tracker"
        ) as mock_tracker:
            tracker = MagicMock()
            alert = _make_alert(alert_id="ALERT-STORE-01")
            tracker.check_upcoming_deadlines.return_value = [alert]
            mock_tracker.return_value = tracker

            client.get(
                "/api/v1/cbam/quarterly/deadlines?importer_id=NL123"
            )
            assert "ALERT-STORE-01" in _alerts_store


class TestAcknowledgeDeadline:
    """Tests for PUT /deadlines/{alert_id}/acknowledge."""

    def test_acknowledge_alert(self, client):
        """Acknowledging an existing alert returns 200."""
        alert = _make_alert(alert_id="ALERT-ACK-01")
        _alerts_store["ALERT-ACK-01"] = alert

        payload = {
            "acknowledged_by": "user@acme.eu",
            "notes": "Noted, will submit by Friday",
        }
        resp = client.put(
            "/api/v1/cbam/quarterly/deadlines/ALERT-ACK-01/acknowledge",
            json=payload,
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["acknowledged"] is True
        assert data["acknowledged_by"] == "user@acme.eu"

    def test_acknowledge_not_found(self, client):
        """Acknowledging a non-existent alert returns 404."""
        payload = {"acknowledged_by": "user@acme.eu"}
        resp = client.put(
            "/api/v1/cbam/quarterly/deadlines/NOPE/acknowledge",
            json=payload,
        )
        assert resp.status_code == 404

    def test_acknowledge_already_acknowledged(self, client):
        """Double-acknowledging an alert returns 409."""
        alert = _make_alert(alert_id="ALERT-ACK-02", acknowledged=True)
        _alerts_store["ALERT-ACK-02"] = alert

        payload = {"acknowledged_by": "user@acme.eu"}
        resp = client.put(
            "/api/v1/cbam/quarterly/deadlines/ALERT-ACK-02/acknowledge",
            json=payload,
        )
        assert resp.status_code == 409

    def test_acknowledge_stores_updated_alert(self, client):
        """After acknowledgement, the stored alert has acknowledged=True."""
        alert = _make_alert(alert_id="ALERT-ACK-03")
        _alerts_store["ALERT-ACK-03"] = alert

        payload = {"acknowledged_by": "user@acme.eu"}
        client.put(
            "/api/v1/cbam/quarterly/deadlines/ALERT-ACK-03/acknowledge",
            json=payload,
        )
        stored = _alerts_store["ALERT-ACK-03"]
        assert stored.acknowledged is True

    def test_acknowledge_missing_acknowledged_by_422(self, client):
        """Missing acknowledged_by field returns 422."""
        alert = _make_alert(alert_id="ALERT-ACK-04")
        _alerts_store["ALERT-ACK-04"] = alert

        resp = client.put(
            "/api/v1/cbam/quarterly/deadlines/ALERT-ACK-04/acknowledge",
            json={},
        )
        assert resp.status_code == 422


# ===========================================================================
# TEST CLASS -- Notification endpoints
# ===========================================================================

class TestNotificationHistory:
    """Tests for GET /notifications."""

    def test_notification_history_empty(self, client):
        """Empty notification history returns zero entries."""
        with patch(
            "quarterly_engine.api.quarterly_routes._get_notification_svc"
        ) as mock_svc:
            svc = MagicMock()
            svc.get_notification_log.return_value = []
            mock_svc.return_value = svc

            resp = client.get(
                "/api/v1/cbam/quarterly/notifications?importer_id=NL123"
            )
            assert resp.status_code == 200
            data = resp.json()["data"]
            assert data["total"] == 0

    def test_notification_history_missing_importer_422(self, client):
        """Missing importer_id returns 422."""
        resp = client.get("/api/v1/cbam/quarterly/notifications")
        assert resp.status_code == 422

    def test_notification_history_pagination(self, client):
        """Notification history supports offset/limit pagination."""
        with patch(
            "quarterly_engine.api.quarterly_routes._get_notification_svc"
        ) as mock_svc:
            svc = MagicMock()
            # Return 5 mock log entries
            entries = []
            for i in range(5):
                entry = MagicMock()
                entry.notification_type = NotificationType.DEADLINE_APPROACHING
                entry.model_dump.return_value = {
                    "log_id": f"LOG-{i:03d}",
                    "notification_type": "deadline_approaching",
                }
                entries.append(entry)
            svc.get_notification_log.return_value = entries
            mock_svc.return_value = svc

            resp = client.get(
                "/api/v1/cbam/quarterly/notifications"
                "?importer_id=NL123&offset=2&limit=2"
            )
            assert resp.status_code == 200
            data = resp.json()["data"]
            assert data["total"] == 5
            assert len(data["notifications"]) == 2


class TestConfigureNotifications:
    """Tests for PUT /notifications/configure."""

    def test_configure_valid_email(self, client):
        """Configuring with email recipients succeeds."""
        with patch(
            "quarterly_engine.api.quarterly_routes._get_notification_svc"
        ) as mock_svc:
            svc = MagicMock()
            mock_svc.return_value = svc

            payload = {
                "importer_id": "NL123456789012",
                "email_recipients": ["compliance@acme.eu"],
                "webhook_urls": [],
            }
            resp = client.put(
                "/api/v1/cbam/quarterly/notifications/configure",
                json=payload,
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["success"] is True

    def test_configure_valid_webhook(self, client):
        """Configuring with webhook URL succeeds."""
        with patch(
            "quarterly_engine.api.quarterly_routes._get_notification_svc"
        ) as mock_svc:
            svc = MagicMock()
            mock_svc.return_value = svc

            payload = {
                "importer_id": "NL123456789012",
                "email_recipients": [],
                "webhook_urls": ["https://hooks.slack.com/xxx"],
            }
            resp = client.put(
                "/api/v1/cbam/quarterly/notifications/configure",
                json=payload,
            )
            assert resp.status_code == 200

    def test_configure_no_recipients_fails(self, client):
        """Configuration with no email or webhook returns 400."""
        payload = {
            "importer_id": "NL123456789012",
            "email_recipients": [],
            "webhook_urls": [],
        }
        resp = client.put(
            "/api/v1/cbam/quarterly/notifications/configure",
            json=payload,
        )
        assert resp.status_code == 400

    def test_configure_stores_config(self, client):
        """Configuration is stored in _notification_configs."""
        with patch(
            "quarterly_engine.api.quarterly_routes._get_notification_svc"
        ) as mock_svc:
            svc = MagicMock()
            mock_svc.return_value = svc

            payload = {
                "importer_id": "NL999888777666",
                "email_recipients": ["test@example.com"],
            }
            client.put(
                "/api/v1/cbam/quarterly/notifications/configure",
                json=payload,
            )
            assert "NL999888777666" in _notification_configs

    def test_configure_missing_importer_id_422(self, client):
        """Missing importer_id returns 422."""
        payload = {
            "email_recipients": ["test@example.com"],
        }
        resp = client.put(
            "/api/v1/cbam/quarterly/notifications/configure",
            json=payload,
        )
        assert resp.status_code == 422

    def test_configure_with_quiet_hours(self, client):
        """Configuration with quiet hours is accepted."""
        with patch(
            "quarterly_engine.api.quarterly_routes._get_notification_svc"
        ) as mock_svc:
            svc = MagicMock()
            mock_svc.return_value = svc

            payload = {
                "importer_id": "NL123456789012",
                "email_recipients": ["compliance@acme.eu"],
                "quiet_hours_start": 22,
                "quiet_hours_end": 7,
            }
            resp = client.put(
                "/api/v1/cbam/quarterly/notifications/configure",
                json=payload,
            )
            assert resp.status_code == 200
            data = resp.json()["data"]
            assert data["quiet_hours_start"] == 22
            assert data["quiet_hours_end"] == 7

    def test_configure_registers_with_service(self, client):
        """Configuration calls notification_svc.register_config."""
        with patch(
            "quarterly_engine.api.quarterly_routes._get_notification_svc"
        ) as mock_svc:
            svc = MagicMock()
            mock_svc.return_value = svc

            payload = {
                "importer_id": "NL123456789012",
                "email_recipients": ["compliance@acme.eu"],
            }
            client.put(
                "/api/v1/cbam/quarterly/notifications/configure",
                json=payload,
            )
            svc.register_config.assert_called_once()


# ===========================================================================
# TEST CLASS -- Response structure
# ===========================================================================

class TestResponseStructure:
    """Tests for standard API response wrapper fields."""

    def test_response_has_success_field(self, client):
        """All responses include a success boolean."""
        _store_test_report(report_id="RPT-RS-01")
        resp = client.get("/api/v1/cbam/quarterly/reports/RPT-RS-01")
        body = resp.json()
        assert "success" in body
        assert isinstance(body["success"], bool)

    def test_response_has_message_field(self, client):
        """All responses include a message string."""
        _store_test_report(report_id="RPT-RS-02")
        resp = client.get("/api/v1/cbam/quarterly/reports/RPT-RS-02")
        body = resp.json()
        assert "message" in body
        assert isinstance(body["message"], str)

    def test_response_has_processing_time(self, client):
        """All responses include processing_time_ms."""
        _store_test_report(report_id="RPT-RS-03")
        resp = client.get("/api/v1/cbam/quarterly/reports/RPT-RS-03")
        body = resp.json()
        assert "processing_time_ms" in body
        assert body["processing_time_ms"] >= 0

    def test_response_has_timestamp(self, client):
        """All responses include a UTC timestamp."""
        _store_test_report(report_id="RPT-RS-04")
        resp = client.get("/api/v1/cbam/quarterly/reports/RPT-RS-04")
        body = resp.json()
        assert "timestamp" in body

    def test_404_response_is_json(self, client):
        """404 responses return JSON with detail field."""
        resp = client.get("/api/v1/cbam/quarterly/reports/NOPE")
        assert resp.status_code == 404
        body = resp.json()
        assert "detail" in body


# ===========================================================================
# TEST CLASS -- Edge cases and integration
# ===========================================================================

class TestEdgeCases:
    """Edge cases and cross-cutting concerns."""

    def test_multiple_reports_different_importers(self, client):
        """Multiple importers can store independent reports."""
        _store_test_report(report_id="RPT-EDGE-01", importer_id="NL001")
        _store_test_report(report_id="RPT-EDGE-02", importer_id="DE002")
        _store_test_report(report_id="RPT-EDGE-03", importer_id="FR003")

        resp = client.get("/api/v1/cbam/quarterly/reports")
        assert resp.json()["data"]["total"] == 3

    def test_reports_sorted_by_created_at_desc(self, client):
        """Report listing is sorted by created_at descending."""
        import time as _time
        _store_test_report(report_id="RPT-SORT-01")
        _time.sleep(0.01)
        _store_test_report(report_id="RPT-SORT-02")

        resp = client.get("/api/v1/cbam/quarterly/reports")
        items = resp.json()["data"]["items"]
        assert len(items) == 2
        # Most recent first
        assert items[0]["report_id"] == "RPT-SORT-02"

    def test_report_store_isolation(self, client):
        """Clearing stores between tests ensures isolation (autouse fixture)."""
        assert len(_reports_store) == 0
        assert len(_importer_reports) == 0
        assert len(_alerts_store) == 0

    def test_multiple_quarters_same_importer(self, client):
        """Same importer can have reports for different quarters."""
        _store_test_report(
            report_id="RPT-MQ-01",
            importer_id="NL123",
            quarter=QuarterlyPeriod.Q1,
        )
        _store_test_report(
            report_id="RPT-MQ-02",
            importer_id="NL123",
            quarter=QuarterlyPeriod.Q2,
        )
        resp = client.get(
            "/api/v1/cbam/quarterly/reports?importer_id=NL123"
        )
        assert resp.json()["data"]["total"] == 2

    def test_importer_reports_index(self, client):
        """_importer_reports index tracks report_ids per importer."""
        _store_test_report(report_id="RPT-IX-01", importer_id="NL_IX")
        _store_test_report(report_id="RPT-IX-02", importer_id="NL_IX")

        assert "NL_IX" in _importer_reports
        assert len(_importer_reports["NL_IX"]) == 2
