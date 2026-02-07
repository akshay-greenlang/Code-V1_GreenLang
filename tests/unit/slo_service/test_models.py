# -*- coding: utf-8 -*-
"""
Unit tests for SLO Service Models (OBS-005)

Tests all data models: SLI, SLO, ErrorBudget, BurnRateAlert, SLOReport,
SLOReportEntry, and all enumerations.

Coverage target: 85%+ of models.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from greenlang.infrastructure.slo_service.models import (
    BudgetStatus,
    BurnRateAlert,
    BurnRateWindow,
    ErrorBudget,
    SLI,
    SLIType,
    SLO,
    SLOReport,
    SLOReportEntry,
    SLOWindow,
)


# ============================================================================
# Enum tests
# ============================================================================


class TestSLIType:
    """Tests for SLIType enum."""

    def test_sli_type_enum_values(self):
        """All five SLI types are defined."""
        assert SLIType.AVAILABILITY.value == "availability"
        assert SLIType.LATENCY.value == "latency"
        assert SLIType.CORRECTNESS.value == "correctness"
        assert SLIType.THROUGHPUT.value == "throughput"
        assert SLIType.FRESHNESS.value == "freshness"

    def test_sli_type_from_string(self):
        """SLIType can be constructed from string value."""
        assert SLIType("availability") == SLIType.AVAILABILITY
        assert SLIType("latency") == SLIType.LATENCY

    def test_sli_type_count(self):
        """Exactly five SLI types exist."""
        assert len(SLIType) == 5


class TestSLOWindow:
    """Tests for SLOWindow enum."""

    def test_slo_window_enum_values(self):
        """All window values are defined."""
        assert SLOWindow.SEVEN_DAYS.value == "7d"
        assert SLOWindow.TWENTY_EIGHT_DAYS.value == "28d"
        assert SLOWindow.THIRTY_DAYS.value == "30d"
        assert SLOWindow.NINETY_DAYS.value == "90d"
        assert SLOWindow.CALENDAR_MONTH.value == "calendar_month"
        assert SLOWindow.CALENDAR_QUARTER.value == "calendar_quarter"

    @pytest.mark.parametrize("window,expected_minutes", [
        (SLOWindow.SEVEN_DAYS, 7 * 24 * 60),
        (SLOWindow.TWENTY_EIGHT_DAYS, 28 * 24 * 60),
        (SLOWindow.THIRTY_DAYS, 30 * 24 * 60),
        (SLOWindow.NINETY_DAYS, 90 * 24 * 60),
        (SLOWindow.CALENDAR_MONTH, 30 * 24 * 60),
        (SLOWindow.CALENDAR_QUARTER, 90 * 24 * 60),
    ])
    def test_slo_window_minutes(self, window, expected_minutes):
        """Window minutes property returns correct values."""
        assert window.minutes == expected_minutes

    @pytest.mark.parametrize("window,expected_duration", [
        (SLOWindow.SEVEN_DAYS, "7d"),
        (SLOWindow.THIRTY_DAYS, "30d"),
        (SLOWindow.CALENDAR_MONTH, "30d"),
        (SLOWindow.CALENDAR_QUARTER, "90d"),
    ])
    def test_slo_window_prometheus_duration(self, window, expected_duration):
        """Prometheus duration strings are correct."""
        assert window.prometheus_duration == expected_duration


class TestBurnRateWindow:
    """Tests for BurnRateWindow enum."""

    def test_burn_rate_window_enum_values(self):
        """All burn rate windows are defined."""
        assert BurnRateWindow.FAST.value == "fast"
        assert BurnRateWindow.MEDIUM.value == "medium"
        assert BurnRateWindow.SLOW.value == "slow"

    @pytest.mark.parametrize("window,threshold", [
        (BurnRateWindow.FAST, 14.4),
        (BurnRateWindow.MEDIUM, 6.0),
        (BurnRateWindow.SLOW, 1.0),
    ])
    def test_burn_rate_thresholds(self, window, threshold):
        """Burn rate thresholds match Google SRE Book values."""
        assert window.threshold == threshold

    @pytest.mark.parametrize("window,long,short", [
        (BurnRateWindow.FAST, "1h", "5m"),
        (BurnRateWindow.MEDIUM, "6h", "30m"),
        (BurnRateWindow.SLOW, "3d", "6h"),
    ])
    def test_burn_rate_windows(self, window, long, short):
        """Long and short window strings are correct."""
        assert window.long_window == long
        assert window.short_window == short

    @pytest.mark.parametrize("window,severity", [
        (BurnRateWindow.FAST, "critical"),
        (BurnRateWindow.MEDIUM, "warning"),
        (BurnRateWindow.SLOW, "info"),
    ])
    def test_burn_rate_severity(self, window, severity):
        """Severity mapping is correct."""
        assert window.severity == severity

    @pytest.mark.parametrize("window,long_min,short_min", [
        (BurnRateWindow.FAST, 60, 5),
        (BurnRateWindow.MEDIUM, 360, 30),
        (BurnRateWindow.SLOW, 4320, 360),
    ])
    def test_burn_rate_window_minutes(self, window, long_min, short_min):
        """Window duration in minutes is correct."""
        assert window.long_window_minutes == long_min
        assert window.short_window_minutes == short_min


class TestBudgetStatus:
    """Tests for BudgetStatus enum."""

    def test_budget_status_enum_values(self):
        """All budget statuses are defined."""
        assert BudgetStatus.HEALTHY.value == "healthy"
        assert BudgetStatus.WARNING.value == "warning"
        assert BudgetStatus.CRITICAL.value == "critical"
        assert BudgetStatus.EXHAUSTED.value == "exhausted"


# ============================================================================
# SLI tests
# ============================================================================


class TestSLI:
    """Tests for SLI dataclass."""

    def test_sli_dataclass_creation(self):
        """SLI can be created with minimal fields."""
        sli = SLI(
            name="test_sli",
            sli_type=SLIType.AVAILABILITY,
            good_query="good_total",
            total_query="total_total",
        )
        assert sli.name == "test_sli"
        assert sli.sli_type == SLIType.AVAILABILITY
        assert sli.threshold_ms is None
        assert sli.unit == ""

    def test_sli_with_threshold(self):
        """SLI with latency threshold is populated correctly."""
        sli = SLI(
            name="latency_sli",
            sli_type=SLIType.LATENCY,
            good_query='bucket{le="0.5"}',
            total_query="count",
            threshold_ms=500.0,
            unit="ms",
        )
        assert sli.threshold_ms == 500.0
        assert sli.unit == "ms"

    def test_sli_to_dict(self, sample_sli_availability):
        """SLI serializes to dictionary correctly."""
        d = sample_sli_availability.to_dict()
        assert d["name"] == "api_availability"
        assert d["sli_type"] == "availability"
        assert "good_query" in d
        assert "total_query" in d

    def test_sli_from_dict(self):
        """SLI deserializes from dictionary correctly."""
        data = {
            "name": "test",
            "sli_type": "latency",
            "good_query": "good",
            "total_query": "total",
            "threshold_ms": 200.0,
            "unit": "ms",
        }
        sli = SLI.from_dict(data)
        assert sli.sli_type == SLIType.LATENCY
        assert sli.threshold_ms == 200.0

    @pytest.mark.parametrize("sli_type", [
        SLIType.AVAILABILITY,
        SLIType.LATENCY,
        SLIType.CORRECTNESS,
        SLIType.THROUGHPUT,
        SLIType.FRESHNESS,
    ])
    def test_sli_types_all(self, sli_type):
        """All SLI types can be used to create an SLI."""
        sli = SLI(
            name=f"test_{sli_type.value}",
            sli_type=sli_type,
            good_query="good",
            total_query="total",
        )
        assert sli.sli_type == sli_type

    def test_sli_roundtrip(self, sample_sli_availability):
        """SLI survives serialization round-trip."""
        d = sample_sli_availability.to_dict()
        restored = SLI.from_dict(d)
        assert restored.name == sample_sli_availability.name
        assert restored.sli_type == sample_sli_availability.sli_type
        assert restored.good_query == sample_sli_availability.good_query


# ============================================================================
# ErrorBudget tests
# ============================================================================


class TestErrorBudget:
    """Tests for ErrorBudget dataclass."""

    def test_error_budget_calculation(self, sample_error_budget):
        """ErrorBudget has correct field values."""
        eb = sample_error_budget
        assert eb.total_minutes == pytest.approx(43.2)
        assert eb.consumed_percent == pytest.approx(10.0)
        assert eb.remaining_percent == pytest.approx(90.0)

    def test_error_budget_status_healthy(self, sample_error_budget):
        """Healthy budget has HEALTHY status."""
        assert sample_error_budget.status == BudgetStatus.HEALTHY

    def test_error_budget_status_warning(self, sample_error_budget_warning):
        """Warning budget has WARNING status."""
        assert sample_error_budget_warning.status == BudgetStatus.WARNING

    def test_error_budget_status_critical(self, sample_error_budget_critical):
        """Critical budget has CRITICAL status."""
        assert sample_error_budget_critical.status == BudgetStatus.CRITICAL

    def test_error_budget_status_exhausted(self, sample_error_budget_exhausted):
        """Exhausted budget has EXHAUSTED status."""
        assert sample_error_budget_exhausted.status == BudgetStatus.EXHAUSTED

    def test_error_budget_calculated_at_auto(self):
        """calculated_at is auto-set if not provided."""
        eb = ErrorBudget(
            slo_id="test",
            total_minutes=43.2,
            consumed_minutes=0,
            remaining_minutes=43.2,
            remaining_percent=100.0,
            consumed_percent=0.0,
            status=BudgetStatus.HEALTHY,
        )
        assert eb.calculated_at is not None

    def test_error_budget_to_dict(self, sample_error_budget):
        """ErrorBudget serializes to dictionary."""
        d = sample_error_budget.to_dict()
        assert d["slo_id"] == "api-availability-99-9"
        assert d["status"] == "healthy"
        assert "calculated_at" in d

    def test_error_budget_from_dict(self):
        """ErrorBudget deserializes from dictionary."""
        data = {
            "slo_id": "test-slo",
            "total_minutes": 43.2,
            "consumed_minutes": 21.6,
            "remaining_minutes": 21.6,
            "remaining_percent": 50.0,
            "consumed_percent": 50.0,
            "status": "critical",
            "sli_value": 99.95,
            "window": "30d",
        }
        eb = ErrorBudget.from_dict(data)
        assert eb.status == BudgetStatus.CRITICAL
        assert eb.remaining_percent == 50.0

    def test_error_budget_remaining_calculation(self):
        """Remaining minutes equals total minus consumed."""
        eb = ErrorBudget(
            slo_id="test",
            total_minutes=100.0,
            consumed_minutes=30.0,
            remaining_minutes=70.0,
            remaining_percent=70.0,
            consumed_percent=30.0,
            status=BudgetStatus.WARNING,
        )
        assert eb.remaining_minutes == eb.total_minutes - eb.consumed_minutes


# ============================================================================
# SLO tests
# ============================================================================


class TestSLO:
    """Tests for SLO dataclass."""

    def test_slo_creation_minimal(self, sample_sli_availability):
        """SLO can be created with minimal required fields."""
        slo = SLO(
            slo_id="min-slo",
            name="Minimal SLO",
            service="test",
            sli=sample_sli_availability,
        )
        assert slo.target == 99.9
        assert slo.window == SLOWindow.THIRTY_DAYS
        assert slo.enabled is True
        assert slo.deleted is False
        assert slo.version == 1

    def test_slo_creation_full(self, sample_slo):
        """SLO with all fields is populated correctly."""
        assert sample_slo.slo_id == "api-availability-99-9"
        assert sample_slo.name == "API Availability"
        assert sample_slo.service == "api-gateway"
        assert sample_slo.target == 99.9
        assert sample_slo.team == "platform"
        assert "tier" in sample_slo.labels

    def test_slo_error_budget_fraction(self, sample_slo):
        """error_budget_fraction for 99.9% target is 0.001."""
        assert sample_slo.error_budget_fraction == pytest.approx(0.001)

    def test_slo_error_budget_fraction_99(self, sample_sli_availability):
        """error_budget_fraction for 99.0% target is 0.01."""
        slo = SLO(
            slo_id="test", name="Test", service="svc",
            sli=sample_sli_availability, target=99.0,
        )
        assert slo.error_budget_fraction == pytest.approx(0.01)

    def test_slo_error_budget_fraction_99_99(self, sample_sli_availability):
        """error_budget_fraction for 99.99% target is 0.0001."""
        slo = SLO(
            slo_id="test", name="Test", service="svc",
            sli=sample_sli_availability, target=99.99,
        )
        assert slo.error_budget_fraction == pytest.approx(0.0001)

    def test_slo_window_minutes(self, sample_slo):
        """window_minutes for 30d window is correct."""
        assert sample_slo.window_minutes == 30 * 24 * 60

    def test_slo_safe_name(self, sample_slo):
        """safe_name sanitizes for Prometheus metric names."""
        assert sample_slo.safe_name == "api_availability"

    def test_slo_fingerprint(self, sample_slo):
        """fingerprint generates a consistent MD5 hash."""
        fp1 = sample_slo.fingerprint()
        fp2 = sample_slo.fingerprint()
        assert fp1 == fp2
        assert len(fp1) == 32  # MD5 hex length

    def test_slo_timestamps_auto(self, sample_sli_availability):
        """created_at and updated_at auto-set when None."""
        slo = SLO(
            slo_id="test", name="Test", service="svc",
            sli=sample_sli_availability,
        )
        assert slo.created_at is not None
        assert slo.updated_at is not None

    def test_slo_to_dict(self, sample_slo):
        """SLO serializes to dictionary with all fields."""
        d = sample_slo.to_dict()
        assert d["slo_id"] == "api-availability-99-9"
        assert d["target"] == 99.9
        assert d["window"] == "30d"
        assert isinstance(d["sli"], dict)
        assert "created_at" in d

    def test_slo_from_dict(self):
        """SLO deserializes from dictionary."""
        data = {
            "slo_id": "from-dict-slo",
            "name": "From Dict",
            "service": "test-svc",
            "target": 99.5,
            "window": "28d",
            "sli": {
                "name": "test_sli",
                "sli_type": "availability",
                "good_query": "good",
                "total_query": "total",
            },
        }
        slo = SLO.from_dict(data)
        assert slo.slo_id == "from-dict-slo"
        assert slo.target == 99.5
        assert slo.window == SLOWindow.TWENTY_EIGHT_DAYS
        assert isinstance(slo.sli, SLI)

    def test_slo_from_yaml_dict(self, sample_slo_yaml_data):
        """SLOs deserialize correctly from YAML-formatted dict."""
        for slo_data in sample_slo_yaml_data["slos"]:
            slo = SLO.from_dict(slo_data)
            assert slo.slo_id
            assert slo.name
            assert isinstance(slo.sli, SLI)

    def test_slo_window_string_conversion(self, sample_sli_availability):
        """Window string is auto-converted to SLOWindow enum."""
        slo = SLO(
            slo_id="test", name="Test", service="svc",
            sli=sample_sli_availability, window="7d",
        )
        assert slo.window == SLOWindow.SEVEN_DAYS

    def test_slo_serialization_roundtrip(self, sample_slo):
        """SLO survives full serialization round-trip."""
        d = sample_slo.to_dict()
        restored = SLO.from_dict(d)
        assert restored.slo_id == sample_slo.slo_id
        assert restored.target == sample_slo.target
        assert restored.sli.name == sample_slo.sli.name


# ============================================================================
# BurnRateAlert tests
# ============================================================================


class TestBurnRateAlert:
    """Tests for BurnRateAlert dataclass."""

    def test_burn_rate_alert_creation(self, sample_burn_rate_alert):
        """BurnRateAlert is created with correct fields."""
        assert sample_burn_rate_alert.slo_id == "api-availability-99-9"
        assert sample_burn_rate_alert.burn_window == "fast"
        assert sample_burn_rate_alert.severity == "critical"
        assert sample_burn_rate_alert.fired_at is not None

    def test_burn_rate_alert_to_dict(self, sample_burn_rate_alert):
        """BurnRateAlert serializes to dictionary."""
        d = sample_burn_rate_alert.to_dict()
        assert d["slo_id"] == "api-availability-99-9"
        assert d["burn_window"] == "fast"
        assert d["threshold"] == 14.4
        assert "fired_at" in d

    def test_burn_rate_alert_auto_id(self):
        """BurnRateAlert generates UUID if not provided."""
        alert = BurnRateAlert(slo_id="test")
        assert alert.alert_id
        assert len(alert.alert_id) > 0


# ============================================================================
# SLOReport tests
# ============================================================================


class TestSLOReport:
    """Tests for SLOReport and SLOReportEntry."""

    def test_slo_report_creation(self):
        """SLOReport can be created with defaults."""
        report = SLOReport(report_type="weekly")
        assert report.report_type == "weekly"
        assert report.generated_at is not None
        assert report.entries == []
        assert report.total_slos == 0

    def test_slo_report_entry_trend(self):
        """SLOReportEntry trend defaults to 'stable'."""
        entry = SLOReportEntry(
            slo_id="test", slo_name="Test",
            service="svc", target=99.9,
            current_sli=99.95, met=True,
            budget_remaining_percent=90.0,
            budget_status="healthy",
        )
        assert entry.trend == "stable"

    def test_slo_report_entry_to_dict(self):
        """SLOReportEntry serializes to dictionary."""
        entry = SLOReportEntry(
            slo_id="test", slo_name="Test",
            service="svc", target=99.9,
            current_sli=99.95, met=True,
            budget_remaining_percent=90.0,
            budget_status="healthy",
            trend="improving",
            violations_count=2,
        )
        d = entry.to_dict()
        assert d["trend"] == "improving"
        assert d["violations_count"] == 2

    def test_slo_report_to_dict(self):
        """SLOReport serializes to dictionary with entries."""
        entry = SLOReportEntry(
            slo_id="test", slo_name="Test",
            service="svc", target=99.9,
            current_sli=99.95, met=True,
            budget_remaining_percent=90.0,
            budget_status="healthy",
        )
        report = SLOReport(
            report_type="monthly",
            entries=[entry],
            total_slos=1,
            slos_met=1,
            overall_compliance_percent=100.0,
        )
        d = report.to_dict()
        assert d["report_type"] == "monthly"
        assert len(d["entries"]) == 1
        assert d["overall_compliance_percent"] == 100.0

    def test_slo_report_from_dict(self):
        """SLOReport deserializes from dictionary."""
        data = {
            "report_type": "quarterly",
            "total_slos": 5,
            "slos_met": 4,
            "slos_not_met": 1,
            "overall_compliance_percent": 80.0,
            "entries": [
                {
                    "slo_id": "test",
                    "slo_name": "Test",
                    "service": "svc",
                    "target": 99.9,
                    "current_sli": 99.5,
                    "met": False,
                    "budget_remaining_percent": 10.0,
                    "budget_status": "critical",
                    "trend": "degrading",
                    "violations_count": 5,
                }
            ],
        }
        report = SLOReport.from_dict(data)
        assert report.report_type == "quarterly"
        assert report.total_slos == 5
        assert len(report.entries) == 1
        assert report.entries[0].trend == "degrading"
