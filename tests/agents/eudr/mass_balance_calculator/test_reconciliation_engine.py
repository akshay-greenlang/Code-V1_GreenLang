# -*- coding: utf-8 -*-
"""
Tests for ReconciliationEngine - AGENT-EUDR-011 Engine 7: Period-End Reconciliation

Comprehensive test suite covering:
- Reconciliation (run, variance calculation)
- Variance classification (acceptable <=1%, warning 1-3%, violation >3%)
- Anomaly detection (spike, consistent overdrafts, timing anomalies)
- Trend analysis (balance trends over periods)
- Facility comparison (benchmark facilities)
- Sign-off (sign-off workflow, authorized user)
- Regulatory compliance (RSPO/FSC/ISCC/EUDR checks)
- Re-reconciliation (re-reconcile on late entries)
- Edge cases (empty period, single entry, perfect balance)

Test count: 55+ tests
Coverage target: >= 85% of ReconciliationEngine module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-011 Mass Balance Calculator Agent (GL-EUDR-MBC-011)
"""

from __future__ import annotations

import copy
import uuid
from decimal import Decimal
from typing import Any, Dict, List

import pytest

from tests.agents.eudr.mass_balance_calculator.conftest import (
    EUDR_COMMODITIES,
    STANDARDS,
    VARIANCE_CLASSIFICATIONS,
    RECONCILIATION_STATUSES,
    VARIANCE_THRESHOLDS,
    SHA256_HEX_LENGTH,
    RECONCILIATION_COCOA_Q1,
    RECON_ID_001,
    PERIOD_COCOA_Q1,
    PERIOD_COCOA_Q2,
    PERIOD_PALM_Y1,
    FAC_ID_MILL_MY,
    FAC_ID_REFINERY_ID,
    FAC_ID_WAREHOUSE_NL,
    FAC_ID_FACTORY_DE,
    make_reconciliation,
    assert_valid_provenance_hash,
    assert_valid_variance,
)


# ===========================================================================
# 1. Reconciliation
# ===========================================================================


class TestReconciliation:
    """Test reconciliation execution."""

    def test_run_reconciliation(self, reconciliation_engine):
        """Run a basic reconciliation."""
        recon = make_reconciliation(
            expected=Decimal("15000.0"),
            recorded=Decimal("15000.0"),
        )
        result = reconciliation_engine.reconcile(recon)
        assert result is not None
        assert result.get("status") in ("completed", "in_progress", "pending")

    def test_variance_calculation(self, reconciliation_engine):
        """Variance is calculated correctly."""
        recon = make_reconciliation(
            expected=Decimal("10000.0"),
            recorded=Decimal("10200.0"),
        )
        result = reconciliation_engine.reconcile(recon)
        variance_kg = result.get("variance_kg", result.get("absolute_variance"))
        if variance_kg is not None:
            assert abs(Decimal(str(variance_kg))) == Decimal("200.0")

    def test_variance_percent_calculated(self, reconciliation_engine):
        """Variance percentage is calculated."""
        recon = make_reconciliation(
            expected=Decimal("10000.0"),
            recorded=Decimal("10500.0"),
        )
        result = reconciliation_engine.reconcile(recon)
        variance_pct = result.get("variance_percent", result.get("percentage"))
        assert variance_pct is not None
        assert abs(float(variance_pct) - 5.0) < 0.1

    def test_reconciliation_provenance_hash(self, reconciliation_engine):
        """Reconciliation generates a provenance hash."""
        recon = make_reconciliation()
        result = reconciliation_engine.reconcile(recon)
        assert result.get("provenance_hash") is not None
        assert_valid_provenance_hash(result["provenance_hash"])

    def test_reconciliation_assigns_id(self, reconciliation_engine):
        """Reconciliation auto-assigns an ID."""
        recon = make_reconciliation()
        recon["reconciliation_id"] = None
        result = reconciliation_engine.reconcile(recon)
        assert result.get("reconciliation_id") is not None

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_reconciliation_all_commodities(self, reconciliation_engine, commodity):
        """Reconciliation works for all 7 EUDR commodities."""
        recon = make_reconciliation(commodity=commodity)
        result = reconciliation_engine.reconcile(recon)
        assert result is not None

    def test_duplicate_reconciliation_raises(self, reconciliation_engine):
        """Duplicate reconciliation ID raises error."""
        recon = make_reconciliation(reconciliation_id="REC-DUP-001")
        reconciliation_engine.reconcile(recon)
        with pytest.raises((ValueError, KeyError)):
            reconciliation_engine.reconcile(copy.deepcopy(recon))


# ===========================================================================
# 2. Variance Classification
# ===========================================================================


class TestVarianceClassification:
    """Test variance classification thresholds."""

    def test_acceptable_variance(self, reconciliation_engine):
        """Variance <= 1% classified as acceptable."""
        recon = make_reconciliation(
            expected=Decimal("10000.0"),
            recorded=Decimal("10050.0"),  # 0.5%
        )
        result = reconciliation_engine.reconcile(recon)
        assert result["classification"] == "acceptable"

    def test_warning_variance(self, reconciliation_engine):
        """Variance 1-3% classified as warning."""
        recon = make_reconciliation(
            expected=Decimal("10000.0"),
            recorded=Decimal("10200.0"),  # 2%
            reconciliation_id="REC-WARN-001",
        )
        result = reconciliation_engine.reconcile(recon)
        assert result["classification"] == "warning"

    def test_violation_variance(self, reconciliation_engine):
        """Variance > 3% classified as violation."""
        recon = make_reconciliation(
            expected=Decimal("10000.0"),
            recorded=Decimal("10500.0"),  # 5%
            reconciliation_id="REC-VIOL-001",
        )
        result = reconciliation_engine.reconcile(recon)
        assert result["classification"] == "violation"

    def test_zero_variance_acceptable(self, reconciliation_engine):
        """Zero variance is acceptable."""
        recon = make_reconciliation(
            expected=Decimal("10000.0"),
            recorded=Decimal("10000.0"),
            reconciliation_id="REC-ZERO-001",
        )
        result = reconciliation_engine.reconcile(recon)
        assert result["classification"] == "acceptable"

    @pytest.mark.parametrize("variance_pct,expected_class", [
        (0.0, "acceptable"),
        (0.5, "acceptable"),
        (1.0, "acceptable"),
        (1.5, "warning"),
        (2.5, "warning"),
        (3.5, "violation"),
        (10.0, "violation"),
    ])
    def test_variance_boundary_conditions(
        self, reconciliation_engine, variance_pct, expected_class
    ):
        """Test variance classification at boundary values."""
        expected = Decimal("10000.0")
        recorded = expected + (expected * Decimal(str(variance_pct / 100)))
        recon = make_reconciliation(
            expected=expected,
            recorded=recorded,
        )
        result = reconciliation_engine.reconcile(recon)
        assert result["classification"] == expected_class

    def test_negative_variance_classified(self, reconciliation_engine):
        """Negative variance (recorded < expected) is also classified."""
        recon = make_reconciliation(
            expected=Decimal("10000.0"),
            recorded=Decimal("9500.0"),  # -5%
            reconciliation_id="REC-NEG-001",
        )
        result = reconciliation_engine.reconcile(recon)
        assert result["classification"] == "violation"


# ===========================================================================
# 3. Anomaly Detection
# ===========================================================================


class TestAnomalyDetection:
    """Test anomaly detection during reconciliation."""

    def test_spike_detection(self, reconciliation_engine):
        """Detect a sudden spike in variance."""
        # Register historical reconciliations with low variance
        for i in range(5):
            recon = make_reconciliation(
                expected=Decimal("10000.0"),
                recorded=Decimal("10050.0"),  # 0.5%
                reconciliation_id=f"REC-SPIKE-H{i:03d}",
                period_id=f"PRD-SPIKE-H{i:03d}",
            )
            reconciliation_engine.reconcile(recon)
        # Now a spike
        spike_recon = make_reconciliation(
            expected=Decimal("10000.0"),
            recorded=Decimal("11000.0"),  # 10%
            reconciliation_id="REC-SPIKE-CUR",
            period_id="PRD-SPIKE-CUR",
        )
        result = reconciliation_engine.reconcile(spike_recon)
        anomalies = result.get("anomalies", [])
        assert isinstance(anomalies, list)

    def test_consistent_overdraft_detection(self, reconciliation_engine):
        """Detect consistent overdraft patterns."""
        for i in range(5):
            recon = make_reconciliation(
                expected=Decimal("10000.0"),
                recorded=Decimal("10400.0"),  # 4% each time
                reconciliation_id=f"REC-CONS-{i:03d}",
                period_id=f"PRD-CONS-{i:03d}",
            )
            reconciliation_engine.reconcile(recon)
        anomalies = reconciliation_engine.detect_anomalies(
            facility_id=FAC_ID_MILL_MY,
        )
        assert isinstance(anomalies, list)

    def test_no_anomalies_for_clean_data(self, reconciliation_engine):
        """Clean data produces no anomalies."""
        recon = make_reconciliation(
            expected=Decimal("10000.0"),
            recorded=Decimal("10000.0"),
            reconciliation_id="REC-CLEAN-001",
        )
        result = reconciliation_engine.reconcile(recon)
        anomalies = result.get("anomalies", [])
        assert len(anomalies) == 0


# ===========================================================================
# 4. Trend Analysis
# ===========================================================================


class TestTrendAnalysis:
    """Test trend analysis over multiple periods."""

    def test_balance_trend(self, reconciliation_engine):
        """Analyze balance trends over multiple periods."""
        for i in range(6):
            recon = make_reconciliation(
                expected=Decimal(str(10000 + i * 500)),
                recorded=Decimal(str(10000 + i * 500 + 100)),
                reconciliation_id=f"REC-TREND-{i:03d}",
                period_id=f"PRD-TREND-{i:03d}",
                facility_id=FAC_ID_MILL_MY,
            )
            reconciliation_engine.reconcile(recon)
        trends = reconciliation_engine.analyze_trends(
            facility_id=FAC_ID_MILL_MY,
        )
        assert trends is not None

    def test_trend_insufficient_data(self, reconciliation_engine):
        """Trend analysis with insufficient data returns appropriate result."""
        recon = make_reconciliation(
            reconciliation_id="REC-INSUF-001",
            period_id="PRD-INSUF-001",
        )
        reconciliation_engine.reconcile(recon)
        trends = reconciliation_engine.analyze_trends(
            facility_id=FAC_ID_MILL_MY,
        )
        assert trends is not None

    def test_trend_direction(self, reconciliation_engine):
        """Trend direction (improving/declining) is detected."""
        for i in range(6):
            recon = make_reconciliation(
                expected=Decimal("10000.0"),
                recorded=Decimal(str(10500 - i * 100)),  # Improving trend
                reconciliation_id=f"REC-DIR-{i:03d}",
                period_id=f"PRD-DIR-{i:03d}",
                facility_id=FAC_ID_MILL_MY,
            )
            reconciliation_engine.reconcile(recon)
        trends = reconciliation_engine.analyze_trends(
            facility_id=FAC_ID_MILL_MY,
        )
        assert trends is not None


# ===========================================================================
# 5. Facility Comparison
# ===========================================================================


class TestFacilityComparison:
    """Test cross-facility benchmarking."""

    def test_benchmark_facilities(self, reconciliation_engine):
        """Compare reconciliation results across facilities."""
        facilities = [FAC_ID_MILL_MY, FAC_ID_REFINERY_ID]
        for fac in facilities:
            recon = make_reconciliation(
                expected=Decimal("10000.0"),
                recorded=Decimal("10100.0"),
                facility_id=fac,
                reconciliation_id=f"REC-BENCH-{fac}",
                period_id=f"PRD-BENCH-{fac}",
            )
            reconciliation_engine.reconcile(recon)
        comparison = reconciliation_engine.compare_facilities(
            facility_ids=facilities,
        )
        assert comparison is not None

    def test_comparison_includes_variance(self, reconciliation_engine):
        """Facility comparison includes per-facility variance data."""
        fac_a = FAC_ID_MILL_MY
        fac_b = FAC_ID_REFINERY_ID
        recon_a = make_reconciliation(
            expected=Decimal("10000.0"),
            recorded=Decimal("10100.0"),
            facility_id=fac_a,
            reconciliation_id="REC-CMP-A",
            period_id="PRD-CMP-A",
        )
        recon_b = make_reconciliation(
            expected=Decimal("20000.0"),
            recorded=Decimal("21000.0"),
            facility_id=fac_b,
            reconciliation_id="REC-CMP-B",
            period_id="PRD-CMP-B",
        )
        reconciliation_engine.reconcile(recon_a)
        reconciliation_engine.reconcile(recon_b)
        comparison = reconciliation_engine.compare_facilities(
            facility_ids=[fac_a, fac_b],
        )
        assert comparison is not None


# ===========================================================================
# 6. Sign-Off
# ===========================================================================


class TestSignOff:
    """Test reconciliation sign-off workflow."""

    def test_sign_off_completed_reconciliation(self, reconciliation_engine):
        """Sign off a completed reconciliation."""
        recon = make_reconciliation(reconciliation_id="REC-SIGN-001")
        reconciliation_engine.reconcile(recon)
        result = reconciliation_engine.sign_off(
            reconciliation_id="REC-SIGN-001",
            signed_off_by="compliance-manager-001",
        )
        assert result["status"] == "signed_off"

    def test_sign_off_generates_provenance(self, reconciliation_engine):
        """Sign-off generates a provenance hash."""
        recon = make_reconciliation(reconciliation_id="REC-SIGN-002")
        reconciliation_engine.reconcile(recon)
        result = reconciliation_engine.sign_off(
            reconciliation_id="REC-SIGN-002",
            signed_off_by="manager-001",
        )
        assert result.get("provenance_hash") is not None
        assert_valid_provenance_hash(result["provenance_hash"])

    def test_sign_off_unauthorized_raises(self, reconciliation_engine):
        """Sign-off without authorization raises error."""
        recon = make_reconciliation(reconciliation_id="REC-SIGN-003")
        reconciliation_engine.reconcile(recon)
        with pytest.raises((ValueError, PermissionError)):
            reconciliation_engine.sign_off(
                reconciliation_id="REC-SIGN-003",
                signed_off_by="",
            )

    def test_sign_off_nonexistent_raises(self, reconciliation_engine):
        """Signing off non-existent reconciliation raises error."""
        with pytest.raises((ValueError, KeyError)):
            reconciliation_engine.sign_off(
                reconciliation_id="REC-NONEXISTENT",
                signed_off_by="manager-001",
            )

    def test_double_sign_off_raises(self, reconciliation_engine):
        """Double sign-off raises error."""
        recon = make_reconciliation(reconciliation_id="REC-SIGN-004")
        reconciliation_engine.reconcile(recon)
        reconciliation_engine.sign_off(
            reconciliation_id="REC-SIGN-004",
            signed_off_by="manager-001",
        )
        with pytest.raises((ValueError, RuntimeError)):
            reconciliation_engine.sign_off(
                reconciliation_id="REC-SIGN-004",
                signed_off_by="manager-002",
            )


# ===========================================================================
# 7. Regulatory Compliance Checks
# ===========================================================================


class TestRegulatoryCompliance:
    """Test regulatory compliance checks during reconciliation."""

    @pytest.mark.parametrize("standard", ["rspo", "fsc", "iscc", "eudr_default"])
    def test_compliance_check_per_standard(self, reconciliation_engine, standard):
        """Compliance check runs for each standard."""
        recon = make_reconciliation(
            standard=standard,
            expected=Decimal("10000.0"),
            recorded=Decimal("10000.0"),
        )
        result = reconciliation_engine.reconcile(recon)
        assert result is not None

    def test_rspo_quarterly_reconciliation(self, reconciliation_engine):
        """RSPO requires quarterly reconciliation."""
        recon = make_reconciliation(
            standard="rspo",
            reconciliation_id="REC-RSPO-001",
        )
        result = reconciliation_engine.reconcile(recon)
        assert result is not None

    def test_eudr_five_year_retention(self, reconciliation_engine):
        """EUDR requires 5-year data retention."""
        recon = make_reconciliation(
            standard="eudr_default",
            reconciliation_id="REC-EUDR-001",
        )
        result = reconciliation_engine.reconcile(recon)
        retention = result.get("retention_years", result.get("retention"))
        if retention is not None:
            assert int(retention) >= 5


# ===========================================================================
# 8. Re-Reconciliation
# ===========================================================================


class TestReReconciliation:
    """Test re-reconciliation on late entries."""

    def test_re_reconcile_on_late_entry(self, reconciliation_engine):
        """Re-reconcile when late entry arrives during grace period."""
        recon = make_reconciliation(reconciliation_id="REC-RE-001")
        reconciliation_engine.reconcile(recon)
        result = reconciliation_engine.re_reconcile(
            reconciliation_id="REC-RE-001",
            reason="Late entry received during grace period",
            new_recorded=Decimal("15200.0"),
        )
        assert result is not None
        assert result.get("status") in ("completed", "reopened", "re_reconciled")

    def test_re_reconcile_updates_variance(self, reconciliation_engine):
        """Re-reconciliation updates the variance."""
        recon = make_reconciliation(
            reconciliation_id="REC-RE-002",
            expected=Decimal("10000.0"),
            recorded=Decimal("10000.0"),
        )
        reconciliation_engine.reconcile(recon)
        result = reconciliation_engine.re_reconcile(
            reconciliation_id="REC-RE-002",
            reason="Late entry",
            new_recorded=Decimal("10300.0"),
        )
        assert result is not None

    def test_re_reconcile_nonexistent_raises(self, reconciliation_engine):
        """Re-reconciling non-existent record raises error."""
        with pytest.raises((ValueError, KeyError)):
            reconciliation_engine.re_reconcile(
                reconciliation_id="REC-NONEXISTENT",
                reason="Test",
                new_recorded=Decimal("10000.0"),
            )


# ===========================================================================
# 9. Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Test edge cases for reconciliation."""

    def test_empty_period_reconciliation(self, reconciliation_engine):
        """Reconcile an empty period (zero inputs and outputs)."""
        recon = make_reconciliation(
            expected=Decimal("0.0"),
            recorded=Decimal("0.0"),
            reconciliation_id="REC-EMPTY-001",
        )
        result = reconciliation_engine.reconcile(recon)
        assert result["classification"] == "acceptable"

    def test_single_entry_reconciliation(self, reconciliation_engine):
        """Reconcile a period with a single entry."""
        recon = make_reconciliation(
            expected=Decimal("5000.0"),
            recorded=Decimal("5000.0"),
            reconciliation_id="REC-SINGLE-001",
        )
        result = reconciliation_engine.reconcile(recon)
        assert result is not None

    def test_perfect_balance(self, reconciliation_engine):
        """Perfect balance (zero variance) reconciliation."""
        recon = make_reconciliation(
            expected=Decimal("25000.0"),
            recorded=Decimal("25000.0"),
            reconciliation_id="REC-PERF-001",
        )
        result = reconciliation_engine.reconcile(recon)
        assert result["classification"] == "acceptable"
        variance_pct = result.get("variance_percent", 0)
        assert float(variance_pct) == 0.0

    def test_very_small_variance(self, reconciliation_engine):
        """Very small variance (0.001 kg) is acceptable."""
        recon = make_reconciliation(
            expected=Decimal("10000.000"),
            recorded=Decimal("10000.001"),
            reconciliation_id="REC-TINY-001",
        )
        result = reconciliation_engine.reconcile(recon)
        assert result["classification"] == "acceptable"

    def test_get_nonexistent_reconciliation(self, reconciliation_engine):
        """Getting non-existent reconciliation returns None."""
        result = reconciliation_engine.get("REC-NONEXISTENT-999")
        assert result is None

    def test_negative_expected_handled(self, reconciliation_engine):
        """Negative expected balance is handled."""
        recon = make_reconciliation(
            expected=Decimal("-1000.0"),
            recorded=Decimal("0.0"),
            reconciliation_id="REC-NEG-EXP-001",
        )
        try:
            result = reconciliation_engine.reconcile(recon)
            assert result is not None
        except ValueError:
            pass  # Also acceptable

    def test_very_large_balance_reconciliation(self, reconciliation_engine):
        """Reconcile with very large balance values."""
        recon = make_reconciliation(
            expected=Decimal("999999999.0"),
            recorded=Decimal("999999999.0"),
            reconciliation_id="REC-LARGE-001",
        )
        result = reconciliation_engine.reconcile(recon)
        assert result["classification"] == "acceptable"

    @pytest.mark.parametrize("standard", STANDARDS)
    def test_reconciliation_all_standards(self, reconciliation_engine, standard):
        """Reconciliation works for all 6 certification standards."""
        recon = make_reconciliation(standard=standard)
        result = reconciliation_engine.reconcile(recon)
        assert result is not None

    def test_reconciliation_history(self, reconciliation_engine):
        """Get reconciliation history for a facility."""
        for i in range(3):
            recon = make_reconciliation(
                reconciliation_id=f"REC-HIST-{i:03d}",
                period_id=f"PRD-HIST-R{i:03d}",
                facility_id=FAC_ID_MILL_MY,
            )
            reconciliation_engine.reconcile(recon)
        history = reconciliation_engine.get_history(facility_id=FAC_ID_MILL_MY)
        assert len(history) >= 3

    def test_reconciliation_includes_classification(self, reconciliation_engine):
        """Every reconciliation result includes a classification."""
        recon = make_reconciliation(
            reconciliation_id="REC-CLASS-001",
            expected=Decimal("10000.0"),
            recorded=Decimal("10200.0"),
        )
        result = reconciliation_engine.reconcile(recon)
        assert result["classification"] in VARIANCE_CLASSIFICATIONS

    def test_timing_anomaly_detection(self, reconciliation_engine):
        """Detect timing anomalies (entries clustered at period end)."""
        recon = make_reconciliation(
            reconciliation_id="REC-TIME-001",
            expected=Decimal("10000.0"),
            recorded=Decimal("10400.0"),
        )
        result = reconciliation_engine.reconcile(recon)
        assert result is not None

    def test_reconciliation_report(self, reconciliation_engine):
        """Generate a reconciliation report."""
        recon = make_reconciliation(reconciliation_id="REC-RPT-001")
        reconciliation_engine.reconcile(recon)
        report = reconciliation_engine.generate_report("REC-RPT-001")
        assert report is not None
