# -*- coding: utf-8 -*-
"""
Unit tests for GL-CBAM-APP v1.1 De Minimis Threshold Monitor and Exemption Manager

Tests ThresholdMonitorEngine:
- Singleton / initialization
- Register imports (basic, accumulation)
- 50-tonne threshold checks (below, at, above)
- Alert levels (80%, 90%, 95%, 100%)
- Electricity and hydrogen exclusion from threshold
- Cumulative by sector group (cement, steel, aluminium, fertilizers)
- Forecast breach date
- Historical import tracking

Tests ExemptionManagerEngine:
- Initialization
- Determine exemption (below/above threshold)
- Exemption status transitions
- Mid-year exemption loss
- Exemption certificate generation
- Annual reset behavior
- SME simplified path
- Multiple importers independent
- Provenance hash on exemption record

Target: 80+ tests
"""

import pytest
import hashlib
import uuid
import threading
from datetime import date, datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from deminimis_engine.threshold_monitor import (
    ThresholdMonitorEngine,
    ThresholdStatus,
    ImportRecord,
    ThresholdAlert,
    SectorBreakdown,
    DE_MINIMIS_THRESHOLD_MT,
    ALERT_THRESHOLDS_PCT,
)
from deminimis_engine.exemption_manager import (
    ExemptionManagerEngine,
    ExemptionResult,
    ExemptionCertificate,
    ExemptionStatus,
    CertificateStatus,
    SMESimplifiedPath,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(autouse=True)
def reset_monitor_singleton():
    """Reset ThresholdMonitorEngine singleton before each test."""
    ThresholdMonitorEngine.reset_instance()
    yield
    ThresholdMonitorEngine.reset_instance()


@pytest.fixture
def monitor():
    """Create a fresh ThresholdMonitorEngine."""
    return ThresholdMonitorEngine()


@pytest.fixture
def exemption_mgr(monitor):
    """Create an ExemptionManagerEngine backed by the monitor fixture."""
    return ExemptionManagerEngine(monitor)


@pytest.fixture
def monitor_with_imports(monitor):
    """Monitor with some pre-loaded imports for IMP-001 in 2026."""
    monitor.add_import("IMP-001", 2026, "72011000", Decimal("10.0"))
    monitor.add_import("IMP-001", 2026, "72031000", Decimal("5.0"))
    return monitor


# ===========================================================================
# TEST CLASS -- ThresholdMonitorEngine initialization
# ===========================================================================

class TestThresholdMonitorInit:
    """Tests for ThresholdMonitorEngine initialization."""

    def test_init(self, monitor):
        assert monitor is not None
        status = monitor.check_threshold("NEW-IMP", 2026)
        assert status.cumulative_mt == Decimal("0")
        assert status.exempt is True

    def test_singleton_pattern(self):
        a = ThresholdMonitorEngine.get_instance()
        b = ThresholdMonitorEngine.get_instance()
        assert a is b

    def test_singleton_reset(self):
        a = ThresholdMonitorEngine.get_instance()
        ThresholdMonitorEngine.reset_instance()
        b = ThresholdMonitorEngine.get_instance()
        assert a is not b

    def test_threshold_value(self):
        assert DE_MINIMIS_THRESHOLD_MT == Decimal("50")


# ===========================================================================
# TEST CLASS -- Register imports
# ===========================================================================

class TestRegisterImport:
    """Tests for add_import."""

    def test_register_import_basic(self, monitor):
        status = monitor.add_import("IMP-001", 2026, "72011000", Decimal("12.5"))
        assert status.cumulative_mt == Decimal("12.5")
        assert status.exempt is True
        assert status.total_records == 1

    def test_register_import_accumulation(self, monitor):
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("10.0"))
        status = monitor.add_import("IMP-001", 2026, "72031000", Decimal("15.0"))
        assert status.cumulative_mt == Decimal("25.0")
        assert status.total_records == 2

    def test_register_import_zero_quantity(self, monitor):
        status = monitor.add_import("IMP-001", 2026, "72011000", Decimal("0"))
        assert status.cumulative_mt == Decimal("0")

    def test_register_import_negative_raises(self, monitor):
        with pytest.raises(ValueError, match="must be >= 0"):
            monitor.add_import("IMP-001", 2026, "72011000", Decimal("-5"))

    def test_register_import_year_before_2026_raises(self, monitor):
        with pytest.raises(ValueError, match="must be >= 2026"):
            monitor.add_import("IMP-001", 2025, "72011000", Decimal("10"))

    def test_register_import_returns_updated_status(self, monitor):
        status = monitor.add_import("IMP-001", 2026, "72011000", Decimal("30"))
        assert isinstance(status, ThresholdStatus)
        assert status.importer_id == "IMP-001"
        assert status.year == 2026

    def test_import_record_has_provenance(self, monitor):
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("10"))
        records = monitor.get_all_records("IMP-001", 2026)
        assert len(records) == 1
        assert len(records[0].provenance_hash) == 64


# ===========================================================================
# TEST CLASS -- Threshold checks
# ===========================================================================

class TestThresholdChecks:
    """Tests for 50-tonne threshold enforcement."""

    def test_threshold_50_tonnes_below(self, monitor):
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("30"))
        status = monitor.check_threshold("IMP-001", 2026)
        assert status.exempt is True
        assert status.percentage == Decimal("60.00")

    def test_threshold_50_tonnes_at_threshold(self, monitor):
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("50"))
        status = monitor.check_threshold("IMP-001", 2026)
        assert status.exempt is False
        assert status.percentage == Decimal("100.00")

    def test_threshold_50_tonnes_above(self, monitor):
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("60"))
        status = monitor.check_threshold("IMP-001", 2026)
        assert status.exempt is False
        assert status.percentage == Decimal("120.00")

    def test_is_exempt_helper(self, monitor):
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("25"))
        assert monitor.is_exempt("IMP-001", 2026) is True

    def test_is_exempt_at_threshold(self, monitor):
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("50"))
        assert monitor.is_exempt("IMP-001", 2026) is False

    def test_get_threshold_percentage(self, monitor):
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("25"))
        pct = monitor.get_threshold_percentage("IMP-001", 2026)
        assert pct == Decimal("50.00")


# ===========================================================================
# TEST CLASS -- Threshold alerts
# ===========================================================================

class TestThresholdAlerts:
    """Tests for alert generation at 80%, 90%, 95%, 100%."""

    def test_alert_80_percent(self, monitor):
        status = monitor.add_import("IMP-001", 2026, "72011000", Decimal("40"))
        assert status.alert_level == 80

    def test_alert_90_percent(self, monitor):
        status = monitor.add_import("IMP-001", 2026, "72011000", Decimal("45"))
        assert status.alert_level == 90

    def test_alert_95_percent(self, monitor):
        status = monitor.add_import("IMP-001", 2026, "72011000", Decimal("48"))
        assert status.alert_level == 95

    def test_alert_100_percent(self, monitor):
        status = monitor.add_import("IMP-001", 2026, "72011000", Decimal("50"))
        assert status.alert_level == 100

    def test_no_alert_below_80(self, monitor):
        status = monitor.add_import("IMP-001", 2026, "72011000", Decimal("35"))
        assert status.alert_level is None

    def test_alert_listener_called(self, monitor):
        alerts_received = []
        monitor.register_alert_listener(lambda a: alerts_received.append(a))
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("45"))
        assert len(alerts_received) >= 1

    def test_alert_not_fired_twice(self, monitor):
        alerts_received = []
        monitor.register_alert_listener(lambda a: alerts_received.append(a))
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("41"))
        count_after_first = len(alerts_received)
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("1"))
        # Should not fire 80% alert again
        assert len([a for a in alerts_received if a.alert_level == 80]) == 1


# ===========================================================================
# TEST CLASS -- Electricity and hydrogen exclusion
# ===========================================================================

class TestExclusions:
    """Tests for electricity (CN 2716) and hydrogen (CN 2804) exclusion."""

    def test_electricity_excluded(self, monitor):
        status = monitor.add_import("IMP-001", 2026, "27160000", Decimal("100"))
        assert status.cumulative_mt == Decimal("0")  # Excluded
        assert status.exempt is True

    def test_hydrogen_excluded(self, monitor):
        status = monitor.add_import("IMP-001", 2026, "28041000", Decimal("100"))
        assert status.cumulative_mt == Decimal("0")  # Excluded
        assert status.exempt is True

    def test_electricity_with_eligible_imports(self, monitor):
        monitor.add_import("IMP-001", 2026, "27160000", Decimal("200"))
        status = monitor.add_import("IMP-001", 2026, "72011000", Decimal("30"))
        assert status.cumulative_mt == Decimal("30")  # Only steel counts

    def test_hydrogen_recorded_but_not_counted(self, monitor):
        monitor.add_import("IMP-001", 2026, "28041000", Decimal("100"))
        records = monitor.get_all_records("IMP-001", 2026)
        assert len(records) == 1
        assert records[0].eligible_for_threshold is False

    def test_cumulative_imports_show_excluded(self, monitor):
        monitor.add_import("IMP-001", 2026, "27160000", Decimal("50"))
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("20"))
        result = monitor.get_cumulative_imports("IMP-001", 2026)
        assert result["total_mt"] == Decimal("70")
        assert result["eligible_mt"] == Decimal("20")
        assert result["excluded_mt"] == Decimal("50")


# ===========================================================================
# TEST CLASS -- Sector breakdown
# ===========================================================================

class TestSectorBreakdown:
    """Tests for cumulative by sector group."""

    def test_iron_steel_sector(self, monitor):
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("20"))
        breakdown = monitor.get_sector_breakdown("IMP-001", 2026)
        assert "iron_steel" in breakdown
        assert breakdown["iron_steel"].cumulative_mt == Decimal("20")

    def test_cement_sector(self, monitor):
        monitor.add_import("IMP-001", 2026, "25231000", Decimal("15"))
        breakdown = monitor.get_sector_breakdown("IMP-001", 2026)
        assert "cement" in breakdown

    def test_aluminium_sector(self, monitor):
        monitor.add_import("IMP-001", 2026, "76011000", Decimal("10"))
        breakdown = monitor.get_sector_breakdown("IMP-001", 2026)
        assert "aluminium" in breakdown

    def test_fertilisers_sector(self, monitor):
        monitor.add_import("IMP-001", 2026, "31021000", Decimal("8"))
        breakdown = monitor.get_sector_breakdown("IMP-001", 2026)
        assert "fertilisers" in breakdown

    def test_multiple_sectors(self, monitor):
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("20"))
        monitor.add_import("IMP-001", 2026, "25231000", Decimal("10"))
        monitor.add_import("IMP-001", 2026, "76011000", Decimal("5"))
        breakdown = monitor.get_sector_breakdown("IMP-001", 2026)
        assert len(breakdown) == 3

    def test_sector_percentage_of_total(self, monitor):
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("30"))
        monitor.add_import("IMP-001", 2026, "25231000", Decimal("20"))
        breakdown = monitor.get_sector_breakdown("IMP-001", 2026)
        iron_pct = breakdown["iron_steel"].percentage_of_total
        cement_pct = breakdown["cement"].percentage_of_total
        assert iron_pct == Decimal("60.00")
        assert cement_pct == Decimal("40.00")


# ===========================================================================
# TEST CLASS -- Forecast breach date
# ===========================================================================

class TestForecastBreachDate:
    """Tests for forecast_threshold_breach."""

    def test_forecast_with_activity(self, monitor):
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("25"))
        projected = monitor.forecast_threshold_breach("IMP-001", 2026)
        # May or may not be None depending on velocity and remaining year
        if projected is not None:
            assert isinstance(projected, date)
            assert projected.year == 2026

    def test_forecast_no_activity(self, monitor):
        projected = monitor.forecast_threshold_breach("IMP-EMPTY", 2026)
        assert projected is None

    def test_forecast_already_breached(self, monitor):
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("60"))
        projected = monitor.forecast_threshold_breach("IMP-001", 2026)
        assert projected is None  # Already breached


# ===========================================================================
# TEST CLASS -- Historical import tracking
# ===========================================================================

class TestHistoricalTracking:
    """Tests for historical import record retrieval."""

    def test_get_all_records(self, monitor):
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("10"))
        monitor.add_import("IMP-001", 2026, "72031000", Decimal("15"))
        records = monitor.get_all_records("IMP-001", 2026)
        assert len(records) == 2

    def test_records_have_sector(self, monitor):
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("10"))
        records = monitor.get_all_records("IMP-001", 2026)
        assert records[0].sector == "iron_steel"

    def test_records_empty_for_unknown(self, monitor):
        records = monitor.get_all_records("UNKNOWN", 2026)
        assert records == []

    def test_import_velocity(self, monitor):
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("10"))
        velocity = monitor.get_import_velocity("IMP-001", 2026)
        assert velocity >= Decimal("0")


# ===========================================================================
# TEST CLASS -- ExemptionManagerEngine
# ===========================================================================

class TestExemptionManagerInit:
    """Tests for ExemptionManagerEngine initialization."""

    def test_init(self, exemption_mgr):
        assert exemption_mgr is not None

    def test_init_with_custom_monitor(self):
        monitor = ThresholdMonitorEngine()
        mgr = ExemptionManagerEngine(monitor)
        assert mgr is not None


# ===========================================================================
# TEST CLASS -- Determine exemption
# ===========================================================================

class TestDetermineExemption:
    """Tests for determine_exemption."""

    def test_exemption_below_threshold(self, exemption_mgr, monitor):
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("10"))
        result = exemption_mgr.determine_exemption("IMP-001", 2026)
        assert result.status == ExemptionStatus.EXEMPT
        assert result.cumulative_mt == Decimal("10")

    def test_exemption_above_threshold(self, exemption_mgr, monitor):
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("55"))
        result = exemption_mgr.determine_exemption("IMP-001", 2026)
        assert result.status == ExemptionStatus.SUBJECT_TO_CBAM

    def test_exemption_approaching(self, exemption_mgr, monitor):
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("42"))
        result = exemption_mgr.determine_exemption("IMP-001", 2026)
        assert result.status == ExemptionStatus.APPROACHING

    def test_exemption_at_80_pct(self, exemption_mgr, monitor):
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("40"))
        result = exemption_mgr.determine_exemption("IMP-001", 2026)
        assert result.status == ExemptionStatus.APPROACHING

    def test_exemption_recommended_actions(self, exemption_mgr, monitor):
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("10"))
        result = exemption_mgr.determine_exemption("IMP-001", 2026)
        assert len(result.recommended_actions) >= 1

    def test_exemption_retroactive_when_exceeded(self, exemption_mgr, monitor):
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("55"))
        result = exemption_mgr.determine_exemption("IMP-001", 2026)
        assert result.retroactive_reporting_required is True
        assert len(result.retroactive_quarters) >= 1


# ===========================================================================
# TEST CLASS -- Exemption status transitions
# ===========================================================================

class TestExemptionStatusTransitions:
    """Tests for exemption status transitions."""

    def test_exempt_to_approaching(self, exemption_mgr, monitor):
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("10"))
        r1 = exemption_mgr.determine_exemption("IMP-001", 2026)
        assert r1.status == ExemptionStatus.EXEMPT

        monitor.add_import("IMP-001", 2026, "72011000", Decimal("32"))
        r2 = exemption_mgr.determine_exemption("IMP-001", 2026)
        assert r2.status == ExemptionStatus.APPROACHING

    def test_approaching_to_subject(self, exemption_mgr, monitor):
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("42"))
        r1 = exemption_mgr.determine_exemption("IMP-001", 2026)
        assert r1.status == ExemptionStatus.APPROACHING

        monitor.add_import("IMP-001", 2026, "72011000", Decimal("10"))
        r2 = exemption_mgr.determine_exemption("IMP-001", 2026)
        assert r2.status == ExemptionStatus.SUBJECT_TO_CBAM


# ===========================================================================
# TEST CLASS -- Mid-year exemption loss
# ===========================================================================

class TestMidYearExemptionLoss:
    """Tests for mid-year exemption loss (revocation)."""

    def test_revoke_exemption(self, exemption_mgr, monitor):
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("55"))
        result = exemption_mgr.revoke_exemption(
            "IMP-001", 2026, "Threshold breached"
        )
        assert result.status == ExemptionStatus.REVOKED
        assert result.loss_date is not None
        assert result.retroactive_reporting_required is True

    def test_revoke_revokes_certificate(self, exemption_mgr, monitor):
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("10"))
        cert = exemption_mgr.issue_exemption_certificate("IMP-001", 2026)
        assert cert.status == CertificateStatus.ACTIVE

        monitor.add_import("IMP-001", 2026, "72011000", Decimal("45"))
        exemption_mgr.revoke_exemption("IMP-001", 2026, "Breach")

        cert_after = exemption_mgr.get_certificate("IMP-001", 2026)
        assert cert_after.status == CertificateStatus.REVOKED


# ===========================================================================
# TEST CLASS -- Exemption certificate generation
# ===========================================================================

class TestExemptionCertificate:
    """Tests for issue_exemption_certificate."""

    def test_issue_certificate(self, exemption_mgr, monitor):
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("10"))
        cert = exemption_mgr.issue_exemption_certificate("IMP-001", 2026)
        assert cert.status == CertificateStatus.ACTIVE
        assert cert.importer_id == "IMP-001"
        assert cert.year == 2026
        assert len(cert.conditions) >= 1

    def test_issue_certificate_not_exempt_raises(self, exemption_mgr, monitor):
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("55"))
        with pytest.raises(ValueError, match="Cannot issue"):
            exemption_mgr.issue_exemption_certificate("IMP-001", 2026)

    def test_certificate_has_provenance(self, exemption_mgr, monitor):
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("10"))
        cert = exemption_mgr.issue_exemption_certificate("IMP-001", 2026)
        assert len(cert.provenance_hash) == 64

    def test_certificate_validity_period(self, exemption_mgr, monitor):
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("10"))
        cert = exemption_mgr.issue_exemption_certificate("IMP-001", 2026)
        assert cert.valid_from == date(2026, 1, 1)
        assert cert.valid_until == date(2026, 12, 31)

    def test_get_certificate(self, exemption_mgr, monitor):
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("10"))
        exemption_mgr.issue_exemption_certificate("IMP-001", 2026)
        cert = exemption_mgr.get_certificate("IMP-001", 2026)
        assert cert is not None

    def test_get_certificate_not_issued(self, exemption_mgr):
        cert = exemption_mgr.get_certificate("IMP-999", 2026)
        assert cert is None


# ===========================================================================
# TEST CLASS -- Annual reset
# ===========================================================================

class TestAnnualReset:
    """Tests for annual reset behavior (year isolation)."""

    def test_different_years_independent(self, monitor):
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("40"))
        monitor.add_import("IMP-001", 2027, "72011000", Decimal("10"))
        status_2026 = monitor.check_threshold("IMP-001", 2026)
        status_2027 = monitor.check_threshold("IMP-001", 2027)
        assert status_2026.cumulative_mt == Decimal("40")
        assert status_2027.cumulative_mt == Decimal("10")

    def test_exemption_resets_annually(self, exemption_mgr, monitor):
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("55"))
        r1 = exemption_mgr.determine_exemption("IMP-001", 2026)
        assert r1.status == ExemptionStatus.SUBJECT_TO_CBAM

        monitor.add_import("IMP-001", 2027, "72011000", Decimal("5"))
        r2 = exemption_mgr.determine_exemption("IMP-001", 2027)
        assert r2.status == ExemptionStatus.EXEMPT


# ===========================================================================
# TEST CLASS -- SME simplified path
# ===========================================================================

class TestSMESimplifiedPath:
    """Tests for get_sme_simplified_path."""

    def test_sme_low_volume(self, exemption_mgr, monitor):
        path = exemption_mgr.get_sme_simplified_path("IMP-SME")
        assert path.is_sme is True
        assert path.simplified_reporting_eligible is True
        assert path.default_values_allowed is True
        assert path.reduced_frequency == "annual"

    def test_sme_medium_volume(self, exemption_mgr, monitor):
        monitor.add_import("IMP-SME2", date.today().year, "72011000", Decimal("30"))
        path = exemption_mgr.get_sme_simplified_path("IMP-SME2")
        assert path.simplified_reporting_eligible is True
        assert path.reduced_frequency == "quarterly"

    def test_sme_above_threshold(self, exemption_mgr, monitor):
        monitor.add_import("IMP-BIG", date.today().year, "72011000", Decimal("60"))
        path = exemption_mgr.get_sme_simplified_path("IMP-BIG")
        assert path.simplified_reporting_eligible is False

    def test_sme_has_provenance(self, exemption_mgr, monitor):
        path = exemption_mgr.get_sme_simplified_path("IMP-SME")
        assert len(path.provenance_hash) == 64

    def test_sme_has_recommended_tools(self, exemption_mgr, monitor):
        path = exemption_mgr.get_sme_simplified_path("IMP-SME")
        assert len(path.recommended_tools) >= 1


# ===========================================================================
# TEST CLASS -- Multiple importers
# ===========================================================================

class TestMultipleImporters:
    """Tests for independent tracking of multiple importers."""

    def test_importers_independent(self, monitor):
        monitor.add_import("IMP-A", 2026, "72011000", Decimal("45"))
        monitor.add_import("IMP-B", 2026, "72011000", Decimal("10"))
        status_a = monitor.check_threshold("IMP-A", 2026)
        status_b = monitor.check_threshold("IMP-B", 2026)
        assert status_a.cumulative_mt == Decimal("45")
        assert status_b.cumulative_mt == Decimal("10")
        assert status_a.exempt is True
        assert status_b.exempt is True

    def test_one_breached_other_exempt(self, exemption_mgr, monitor):
        monitor.add_import("IMP-A", 2026, "72011000", Decimal("55"))
        monitor.add_import("IMP-B", 2026, "72011000", Decimal("10"))
        r_a = exemption_mgr.determine_exemption("IMP-A", 2026)
        r_b = exemption_mgr.determine_exemption("IMP-B", 2026)
        assert r_a.status == ExemptionStatus.SUBJECT_TO_CBAM
        assert r_b.status == ExemptionStatus.EXEMPT


# ===========================================================================
# TEST CLASS -- Provenance hash on exemption record
# ===========================================================================

class TestExemptionProvenance:
    """Tests for provenance hashing on exemption records."""

    def test_exemption_result_has_provenance(self, exemption_mgr, monitor):
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("10"))
        result = exemption_mgr.determine_exemption("IMP-001", 2026)
        assert len(result.provenance_hash) == 64

    def test_threshold_status_has_provenance(self, monitor):
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("10"))
        status = monitor.check_threshold("IMP-001", 2026)
        assert len(status.provenance_hash) == 64

    def test_exemption_history(self, exemption_mgr, monitor):
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("10"))
        exemption_mgr.determine_exemption("IMP-001", 2026)
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("20"))
        exemption_mgr.determine_exemption("IMP-001", 2026)
        history = exemption_mgr.get_exemption_history("IMP-001")
        assert len(history) == 2


# ===========================================================================
# TEST CLASS -- Retroactive reporting
# ===========================================================================

class TestRetroactiveReporting:
    """Tests for retroactive reporting requirements."""

    def test_handle_retroactive_reporting(self, exemption_mgr, monitor):
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("55"))
        plan = exemption_mgr.handle_retroactive_reporting("IMP-001", 2026)
        assert plan["importer_id"] == "IMP-001"
        assert plan["year"] == 2026
        assert len(plan["retroactive_quarters"]) >= 1
        assert plan["status"] == "subject_to_cbam"
        assert "provenance_hash" in plan

    def test_retroactive_has_guidance(self, exemption_mgr, monitor):
        monitor.add_import("IMP-001", 2026, "72011000", Decimal("55"))
        plan = exemption_mgr.handle_retroactive_reporting("IMP-001", 2026)
        assert len(plan["guidance"]) >= 1
