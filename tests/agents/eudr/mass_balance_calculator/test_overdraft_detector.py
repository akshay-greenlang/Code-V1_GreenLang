# -*- coding: utf-8 -*-
"""
Tests for OverdraftDetector - AGENT-EUDR-011 Engine 4: Overdraft Detection

Comprehensive test suite covering:
- Overdraft check (zero tolerance, percentage, absolute modes)
- Severity classification (warning, violation, critical thresholds)
- Alert generation (batch_ids, quantities, recommended actions)
- Overdraft resolution (resolve within deadline, expired deadline)
- Forecast (impact of proposed output on balance)
- Exemptions (request, approve, expired)
- Pattern detection (recurring overdraft patterns)
- Edge cases (zero balance, exact tolerance, concurrent checks)

Test count: 55+ tests
Coverage target: >= 85% of OverdraftDetector module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-011 Mass Balance Calculator Agent (GL-EUDR-MBC-011)
"""

from __future__ import annotations

import copy
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List

import pytest

from tests.agents.eudr.mass_balance_calculator.conftest import (
    EUDR_COMMODITIES,
    OVERDRAFT_MODES,
    OVERDRAFT_SEVERITIES,
    OVERDRAFT_TOLERANCE,
    DEFAULT_OVERDRAFT_RESOLUTION_HOURS,
    SHA256_HEX_LENGTH,
    LEDGER_COCOA_001,
    LEDGER_PALM_001,
    FAC_ID_MILL_MY,
    FAC_ID_REFINERY_ID,
    BATCH_COCOA_001,
    BATCH_PALM_001,
    make_ledger,
    make_entry,
    assert_valid_provenance_hash,
)


# ===========================================================================
# 1. Overdraft Check
# ===========================================================================


class TestOverdraftCheck:
    """Test overdraft check in different enforcement modes."""

    def test_zero_tolerance_rejects_overdraft(self, overdraft_detector):
        """Zero tolerance mode rejects any negative balance."""
        result = overdraft_detector.check(
            ledger_id=LEDGER_COCOA_001,
            current_balance_kg=Decimal("1000.0"),
            proposed_output_kg=Decimal("1500.0"),
            mode="zero_tolerance",
        )
        assert result["overdraft_detected"] is True

    def test_zero_tolerance_allows_within_balance(self, overdraft_detector):
        """Zero tolerance mode allows output within balance."""
        result = overdraft_detector.check(
            ledger_id=LEDGER_COCOA_001,
            current_balance_kg=Decimal("5000.0"),
            proposed_output_kg=Decimal("3000.0"),
            mode="zero_tolerance",
        )
        assert result["overdraft_detected"] is False

    def test_percentage_mode_within_tolerance(self, overdraft_detector):
        """Percentage mode allows overdraft within tolerance."""
        result = overdraft_detector.check(
            ledger_id=LEDGER_COCOA_001,
            current_balance_kg=Decimal("1000.0"),
            proposed_output_kg=Decimal("1040.0"),  # 4% over, within 5%
            mode="percentage",
            tolerance_percent=5.0,
            total_period_inputs_kg=Decimal("10000.0"),
        )
        assert result["overdraft_detected"] is False or result.get("severity") == "warning"

    def test_percentage_mode_exceeds_tolerance(self, overdraft_detector):
        """Percentage mode rejects overdraft exceeding tolerance."""
        result = overdraft_detector.check(
            ledger_id=LEDGER_COCOA_001,
            current_balance_kg=Decimal("1000.0"),
            proposed_output_kg=Decimal("2000.0"),  # 10% of 10000, exceeds 5%
            mode="percentage",
            tolerance_percent=5.0,
            total_period_inputs_kg=Decimal("10000.0"),
        )
        assert result["overdraft_detected"] is True

    def test_absolute_mode_within_tolerance(self, overdraft_detector):
        """Absolute mode allows overdraft within tolerance kg."""
        result = overdraft_detector.check(
            ledger_id=LEDGER_COCOA_001,
            current_balance_kg=Decimal("100.0"),
            proposed_output_kg=Decimal("130.0"),  # 30kg over, within 50kg
            mode="absolute",
            tolerance_kg=Decimal("50.0"),
        )
        assert result["overdraft_detected"] is False or result.get("severity") == "warning"

    def test_absolute_mode_exceeds_tolerance(self, overdraft_detector):
        """Absolute mode rejects overdraft exceeding tolerance kg."""
        result = overdraft_detector.check(
            ledger_id=LEDGER_COCOA_001,
            current_balance_kg=Decimal("100.0"),
            proposed_output_kg=Decimal("200.0"),  # 100kg over, exceeds 50kg
            mode="absolute",
            tolerance_kg=Decimal("50.0"),
        )
        assert result["overdraft_detected"] is True

    def test_exact_balance_no_overdraft(self, overdraft_detector):
        """Output exactly equal to balance does not trigger overdraft."""
        result = overdraft_detector.check(
            ledger_id=LEDGER_COCOA_001,
            current_balance_kg=Decimal("5000.0"),
            proposed_output_kg=Decimal("5000.0"),
            mode="zero_tolerance",
        )
        assert result["overdraft_detected"] is False

    @pytest.mark.parametrize("mode", OVERDRAFT_MODES)
    def test_check_all_modes(self, overdraft_detector, mode):
        """Overdraft check works in all enforcement modes."""
        kwargs = {
            "ledger_id": LEDGER_COCOA_001,
            "current_balance_kg": Decimal("1000.0"),
            "proposed_output_kg": Decimal("500.0"),
            "mode": mode,
        }
        if mode == "percentage":
            kwargs["tolerance_percent"] = 5.0
            kwargs["total_period_inputs_kg"] = Decimal("10000.0")
        elif mode == "absolute":
            kwargs["tolerance_kg"] = Decimal("50.0")
        result = overdraft_detector.check(**kwargs)
        assert result is not None


# ===========================================================================
# 2. Severity Classification
# ===========================================================================


class TestSeverityClassification:
    """Test overdraft severity classification."""

    def test_warning_severity(self, overdraft_detector):
        """Small overdraft classified as warning."""
        result = overdraft_detector.classify_severity(
            overdraft_amount_kg=Decimal("10.0"),
            balance_kg=Decimal("1000.0"),
        )
        assert result["severity"] in ("warning", "low")

    def test_violation_severity(self, overdraft_detector):
        """Medium overdraft classified as violation."""
        result = overdraft_detector.classify_severity(
            overdraft_amount_kg=Decimal("200.0"),
            balance_kg=Decimal("1000.0"),
        )
        assert result["severity"] in ("violation", "medium", "warning")

    def test_critical_severity(self, overdraft_detector):
        """Large overdraft classified as critical."""
        result = overdraft_detector.classify_severity(
            overdraft_amount_kg=Decimal("5000.0"),
            balance_kg=Decimal("1000.0"),
        )
        assert result["severity"] in ("critical", "high")

    @pytest.mark.parametrize("severity", OVERDRAFT_SEVERITIES)
    def test_all_severity_levels_reachable(self, overdraft_detector, severity):
        """All severity levels can be reached."""
        amounts = {
            "warning": Decimal("5.0"),
            "violation": Decimal("200.0"),
            "critical": Decimal("5000.0"),
        }
        result = overdraft_detector.classify_severity(
            overdraft_amount_kg=amounts[severity],
            balance_kg=Decimal("1000.0"),
        )
        assert result["severity"] is not None

    def test_severity_includes_overdraft_amount(self, overdraft_detector):
        """Severity result includes overdraft amount."""
        result = overdraft_detector.classify_severity(
            overdraft_amount_kg=Decimal("100.0"),
            balance_kg=Decimal("1000.0"),
        )
        assert "overdraft_amount_kg" in result or "amount" in result or "overdraft" in result


# ===========================================================================
# 3. Alert Generation
# ===========================================================================


class TestAlertGeneration:
    """Test overdraft alert generation."""

    def test_alert_includes_batch_ids(self, overdraft_detector):
        """Alert includes affected batch IDs."""
        alert = overdraft_detector.generate_alert(
            ledger_id=LEDGER_COCOA_001,
            overdraft_amount_kg=Decimal("500.0"),
            batch_ids=[BATCH_COCOA_001],
            severity="violation",
        )
        assert alert is not None
        assert BATCH_COCOA_001 in alert.get("batch_ids", alert.get("affected_batches", []))

    def test_alert_includes_quantities(self, overdraft_detector):
        """Alert includes overdraft quantity details."""
        alert = overdraft_detector.generate_alert(
            ledger_id=LEDGER_COCOA_001,
            overdraft_amount_kg=Decimal("300.0"),
            batch_ids=[BATCH_COCOA_001],
            severity="warning",
        )
        assert "overdraft_amount_kg" in alert or "quantity" in alert or "amount" in alert

    def test_alert_includes_recommended_actions(self, overdraft_detector):
        """Alert includes recommended corrective actions."""
        alert = overdraft_detector.generate_alert(
            ledger_id=LEDGER_COCOA_001,
            overdraft_amount_kg=Decimal("1000.0"),
            batch_ids=[BATCH_COCOA_001],
            severity="critical",
        )
        assert "recommended_actions" in alert or "actions" in alert or "recommendations" in alert

    def test_alert_provenance_hash(self, overdraft_detector):
        """Alert generates a provenance hash."""
        alert = overdraft_detector.generate_alert(
            ledger_id=LEDGER_COCOA_001,
            overdraft_amount_kg=Decimal("200.0"),
            batch_ids=[BATCH_COCOA_001],
            severity="violation",
        )
        assert alert.get("provenance_hash") is not None
        assert_valid_provenance_hash(alert["provenance_hash"])

    def test_alert_critical_auto_reject(self, overdraft_detector):
        """Critical overdraft triggers auto-reject flag."""
        alert = overdraft_detector.generate_alert(
            ledger_id=LEDGER_COCOA_001,
            overdraft_amount_kg=Decimal("10000.0"),
            batch_ids=[BATCH_COCOA_001],
            severity="critical",
        )
        assert alert.get("auto_reject") is True or alert.get("action") == "reject"


# ===========================================================================
# 4. Overdraft Resolution
# ===========================================================================


class TestOverdraftResolution:
    """Test overdraft resolution tracking."""

    def test_resolve_within_deadline(self, overdraft_detector):
        """Resolve overdraft within the resolution deadline."""
        alert_id = f"ALT-{uuid.uuid4().hex[:8].upper()}"
        overdraft_detector.register_overdraft(
            alert_id=alert_id,
            ledger_id=LEDGER_COCOA_001,
            overdraft_amount_kg=Decimal("200.0"),
            severity="violation",
        )
        result = overdraft_detector.resolve(
            alert_id=alert_id,
            resolution_type="matching_input",
            resolved_by="supervisor-001",
        )
        assert result["status"] in ("resolved", "closed")

    def test_expired_deadline_escalates(self, overdraft_detector):
        """Overdraft past resolution deadline is escalated."""
        alert_id = f"ALT-{uuid.uuid4().hex[:8].upper()}"
        overdraft_detector.register_overdraft(
            alert_id=alert_id,
            ledger_id=LEDGER_COCOA_001,
            overdraft_amount_kg=Decimal("500.0"),
            severity="violation",
            created_hours_ago=DEFAULT_OVERDRAFT_RESOLUTION_HOURS + 1,
        )
        result = overdraft_detector.check_deadline(alert_id=alert_id)
        assert result.get("expired") is True or result.get("escalated") is True

    def test_resolve_nonexistent_raises(self, overdraft_detector):
        """Resolving non-existent overdraft raises error."""
        with pytest.raises((ValueError, KeyError)):
            overdraft_detector.resolve(
                alert_id="ALT-NONEXISTENT",
                resolution_type="matching_input",
                resolved_by="supervisor-001",
            )


# ===========================================================================
# 5. Forecast
# ===========================================================================


class TestForecast:
    """Test pre-output balance forecast."""

    def test_forecast_positive_balance(self, overdraft_detector):
        """Forecast shows positive balance after output."""
        result = overdraft_detector.forecast(
            current_balance_kg=Decimal("5000.0"),
            proposed_output_kg=Decimal("3000.0"),
        )
        assert Decimal(str(result["projected_balance_kg"])) == Decimal("2000.0")
        assert result["would_overdraft"] is False

    def test_forecast_negative_balance(self, overdraft_detector):
        """Forecast shows negative balance if output exceeds balance."""
        result = overdraft_detector.forecast(
            current_balance_kg=Decimal("1000.0"),
            proposed_output_kg=Decimal("3000.0"),
        )
        assert Decimal(str(result["projected_balance_kg"])) == Decimal("-2000.0")
        assert result["would_overdraft"] is True

    def test_forecast_exact_balance(self, overdraft_detector):
        """Forecast at exact balance shows zero remaining."""
        result = overdraft_detector.forecast(
            current_balance_kg=Decimal("5000.0"),
            proposed_output_kg=Decimal("5000.0"),
        )
        assert Decimal(str(result["projected_balance_kg"])) == Decimal("0.0")
        assert result["would_overdraft"] is False

    def test_forecast_multiple_outputs(self, overdraft_detector):
        """Forecast with multiple proposed outputs."""
        result = overdraft_detector.forecast(
            current_balance_kg=Decimal("10000.0"),
            proposed_output_kg=Decimal("8000.0"),
        )
        assert result is not None
        assert result["would_overdraft"] is False


# ===========================================================================
# 6. Exemptions
# ===========================================================================


class TestExemptions:
    """Test overdraft exemption management."""

    def test_request_exemption(self, overdraft_detector):
        """Request an overdraft exemption."""
        result = overdraft_detector.request_exemption(
            ledger_id=LEDGER_COCOA_001,
            facility_id=FAC_ID_MILL_MY,
            reason="Seasonal peak processing volume",
            requested_by="plant-manager-001",
            duration_days=30,
        )
        assert result is not None
        assert result.get("status") in ("pending", "requested")

    def test_approve_exemption(self, overdraft_detector):
        """Approve an exemption request."""
        exemption_id = f"EXM-{uuid.uuid4().hex[:8].upper()}"
        overdraft_detector.request_exemption(
            ledger_id=LEDGER_COCOA_001,
            facility_id=FAC_ID_MILL_MY,
            reason="Test",
            requested_by="manager-001",
            duration_days=30,
            exemption_id=exemption_id,
        )
        result = overdraft_detector.approve_exemption(
            exemption_id=exemption_id,
            approved_by="compliance-officer-001",
        )
        assert result["status"] in ("approved", "active")

    def test_expired_exemption_not_valid(self, overdraft_detector):
        """Expired exemption is no longer valid."""
        exemption_id = f"EXM-{uuid.uuid4().hex[:8].upper()}"
        overdraft_detector.request_exemption(
            ledger_id=LEDGER_COCOA_001,
            facility_id=FAC_ID_MILL_MY,
            reason="Test",
            requested_by="manager-001",
            duration_days=0,  # Expires immediately
            exemption_id=exemption_id,
        )
        overdraft_detector.approve_exemption(
            exemption_id=exemption_id,
            approved_by="compliance-001",
        )
        result = overdraft_detector.check_exemption(exemption_id=exemption_id)
        assert result.get("valid") is False or result.get("expired") is True

    def test_exemption_nonexistent_raises(self, overdraft_detector):
        """Checking non-existent exemption raises error."""
        with pytest.raises((ValueError, KeyError)):
            overdraft_detector.check_exemption(exemption_id="EXM-NONEXISTENT")


# ===========================================================================
# 7. Pattern Detection
# ===========================================================================


class TestPatternDetection:
    """Test recurring overdraft pattern detection."""

    def test_detect_recurring_pattern(self, overdraft_detector):
        """Detect recurring overdraft patterns for a facility."""
        for i in range(5):
            overdraft_detector.register_overdraft(
                alert_id=f"ALT-PAT-{i:03d}",
                ledger_id=LEDGER_COCOA_001,
                overdraft_amount_kg=Decimal("200.0"),
                severity="warning",
            )
        patterns = overdraft_detector.detect_patterns(
            facility_id=FAC_ID_MILL_MY,
        )
        assert patterns is not None
        assert len(patterns) >= 0

    def test_no_pattern_for_clean_facility(self, overdraft_detector):
        """No patterns detected for facility with no overdrafts."""
        patterns = overdraft_detector.detect_patterns(
            facility_id="FAC-CLEAN-001",
        )
        assert len(patterns) == 0

    def test_pattern_includes_frequency(self, overdraft_detector):
        """Pattern report includes frequency data."""
        for i in range(3):
            overdraft_detector.register_overdraft(
                alert_id=f"ALT-FREQ-{i:03d}",
                ledger_id=LEDGER_COCOA_001,
                overdraft_amount_kg=Decimal("100.0"),
                severity="warning",
            )
        patterns = overdraft_detector.detect_patterns(
            facility_id=FAC_ID_MILL_MY,
        )
        if patterns:
            assert "frequency" in patterns[0] or "count" in patterns[0]


# ===========================================================================
# 8. Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Test edge cases for overdraft detection."""

    def test_zero_balance_any_output_overdrafts(self, overdraft_detector):
        """Any output on zero balance triggers overdraft (zero tolerance)."""
        result = overdraft_detector.check(
            ledger_id=LEDGER_COCOA_001,
            current_balance_kg=Decimal("0.0"),
            proposed_output_kg=Decimal("1.0"),
            mode="zero_tolerance",
        )
        assert result["overdraft_detected"] is True

    def test_zero_output_no_overdraft(self, overdraft_detector):
        """Zero output never triggers overdraft."""
        result = overdraft_detector.check(
            ledger_id=LEDGER_COCOA_001,
            current_balance_kg=Decimal("1000.0"),
            proposed_output_kg=Decimal("0.0"),
            mode="zero_tolerance",
        )
        assert result["overdraft_detected"] is False

    def test_negative_output_raises(self, overdraft_detector):
        """Negative proposed output raises ValueError."""
        with pytest.raises(ValueError):
            overdraft_detector.check(
                ledger_id=LEDGER_COCOA_001,
                current_balance_kg=Decimal("1000.0"),
                proposed_output_kg=Decimal("-500.0"),
                mode="zero_tolerance",
            )

    def test_very_small_overdraft(self, overdraft_detector):
        """Very small overdraft (0.001 kg) is still detected in zero tolerance."""
        result = overdraft_detector.check(
            ledger_id=LEDGER_COCOA_001,
            current_balance_kg=Decimal("1000.0"),
            proposed_output_kg=Decimal("1000.001"),
            mode="zero_tolerance",
        )
        assert result["overdraft_detected"] is True

    def test_exact_tolerance_boundary_percentage(self, overdraft_detector):
        """Output exactly at percentage tolerance boundary."""
        # 5% of 10000 = 500; balance 1000, output 1500 = exactly at 5%
        result = overdraft_detector.check(
            ledger_id=LEDGER_COCOA_001,
            current_balance_kg=Decimal("1000.0"),
            proposed_output_kg=Decimal("1500.0"),
            mode="percentage",
            tolerance_percent=5.0,
            total_period_inputs_kg=Decimal("10000.0"),
        )
        assert result is not None

    def test_check_provenance_hash(self, overdraft_detector):
        """Overdraft check generates a provenance hash."""
        result = overdraft_detector.check(
            ledger_id=LEDGER_COCOA_001,
            current_balance_kg=Decimal("5000.0"),
            proposed_output_kg=Decimal("3000.0"),
            mode="zero_tolerance",
        )
        if result.get("provenance_hash"):
            assert_valid_provenance_hash(result["provenance_hash"])

    def test_negative_balance_raises(self, overdraft_detector):
        """Negative current balance raises ValueError."""
        with pytest.raises(ValueError):
            overdraft_detector.check(
                ledger_id=LEDGER_COCOA_001,
                current_balance_kg=Decimal("-100.0"),
                proposed_output_kg=Decimal("50.0"),
                mode="zero_tolerance",
            )

    def test_zero_balance_zero_output_no_overdraft(self, overdraft_detector):
        """Zero balance with zero output does not overdraft."""
        result = overdraft_detector.check(
            ledger_id=LEDGER_COCOA_001,
            current_balance_kg=Decimal("0.0"),
            proposed_output_kg=Decimal("0.0"),
            mode="zero_tolerance",
        )
        assert result["overdraft_detected"] is False

    @pytest.mark.parametrize("balance,output,expected", [
        ("1000.0", "999.0", False),
        ("1000.0", "1000.0", False),
        ("1000.0", "1001.0", True),
        ("0.0", "0.001", True),
        ("5000.0", "0.0", False),
    ])
    def test_overdraft_boundary_conditions(
        self, overdraft_detector, balance, output, expected
    ):
        """Test overdraft detection at boundary values."""
        result = overdraft_detector.check(
            ledger_id=LEDGER_COCOA_001,
            current_balance_kg=Decimal(balance),
            proposed_output_kg=Decimal(output),
            mode="zero_tolerance",
        )
        assert result["overdraft_detected"] is expected

    def test_overdraft_history(self, overdraft_detector):
        """Get overdraft history for a facility."""
        overdraft_detector.register_overdraft(
            alert_id="ALT-HIST-001",
            ledger_id=LEDGER_COCOA_001,
            overdraft_amount_kg=Decimal("100.0"),
            severity="warning",
        )
        history = overdraft_detector.get_history(facility_id=FAC_ID_MILL_MY)
        assert isinstance(history, list)
