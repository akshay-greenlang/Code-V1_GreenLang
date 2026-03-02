# -*- coding: utf-8 -*-
"""
Unit tests for ComplianceCheckerEngine (AGENT-MRV-019, Engine 6)

50 tests covering 7 regulatory frameworks + double-counting prevention + integration:
- GHG Protocol Scope 3 (7 tests)
- ISO 14064 (5 tests)
- CSRD ESRS E1 (5 tests)
- CDP Climate Change (7 tests)
- SBTi (5 tests)
- SB 253 (4 tests)
- GRI 305 (4 tests)
- Double-Counting Prevention DC-BT-001 through DC-BT-007 (7 tests)
- Integration / Category Boundary (6 tests)

Compliance scoring formula:
    penalty_points = failed_checks + warning_checks x 0.5
    score = (total_checks - penalty_points) / total_checks x 100

Author: GL-TestEngineer
Date: February 2026
"""

import pytest
from decimal import Decimal
from unittest.mock import patch, MagicMock

from greenlang.business_travel.compliance_checker import (
    ComplianceCheckerEngine,
    BoundaryClassification,
)
from greenlang.business_travel.models import (
    ComplianceFramework,
    ComplianceStatus,
    ComplianceCheckResult,
)


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset singleton before and after every test for isolation."""
    ComplianceCheckerEngine.reset_instance()
    yield
    ComplianceCheckerEngine.reset_instance()


def _make_mock_config(
    frameworks: str = "GHG_PROTOCOL_SCOPE3,ISO_14064,CSRD_ESRS_E1,CDP,SBTI,SB_253,GRI",
    strict_mode: bool = False,
    materiality_threshold: Decimal = Decimal("0.01"),
):
    """Create a mock config object."""
    mock_cfg = MagicMock()
    mock_cfg.compliance.get_frameworks.return_value = frameworks.split(",")
    mock_cfg.compliance.strict_mode = strict_mode
    mock_cfg.compliance.materiality_threshold = materiality_threshold
    return mock_cfg


@pytest.fixture
def engine():
    """Create a fresh ComplianceCheckerEngine with mocked config."""
    with patch(
        "greenlang.business_travel.compliance_checker.get_config"
    ) as mock_config:
        mock_config.return_value = _make_mock_config()
        eng = ComplianceCheckerEngine()
        yield eng


def _full_result(**overrides):
    """Build a fully-compliant result dict with all fields present."""
    base = {
        "total_co2e": 1500.0,
        "method": "distance_based",
        "calculation_method": "distance_based",
        "ef_sources": ["DEFRA"],
        "ef_source": "defra",
        "exclusions": "None - all modes included",
        "dqi_score": 3.5,
        "data_quality_score": 3.5,
        "mode_breakdown": {"air": 1000.0, "rail": 300.0, "hotel": 200.0},
        "by_mode": {"air": 1000.0, "rail": 300.0, "hotel": 200.0},
        "total_co2e_with_rf": 1650.0,
        "with_rf": 1650.0,
        "total_co2e_without_rf": 1500.0,
        "without_rf": 1500.0,
        "uncertainty_analysis": {"method": "monte_carlo", "ci_95": [1350, 1650]},
        "uncertainty": {"method": "monte_carlo"},
        "base_year": 2019,
        "methodology": "GHG Protocol Scope 3 Category 6, distance-based",
        "targets": "Reduce business travel emissions 30% by 2030",
        "reduction_targets": "30% by 2030",
        "actions": "Rail-first policy, virtual meeting guidelines",
        "reduction_actions": "Rail-first policy",
        "verification_status": "limited_assurance",
        "verified": True,
        "assurance": "limited",
        "assurance_opinion": "limited_assurance",
        "rf_inclusion": True,
        "rf_in_target": True,
        "target_coverage": "67%",
        "sbti_coverage": "67%",
        "progress_tracking": {"2023": 1800, "2024": 1500},
        "year_over_year_change": -16.7,
        "total_scope3_co2e": 50000.0,
        "gases_included": ["CO2", "CH4", "N2O"],
        "emission_gases": ["CO2", "CH4", "N2O"],
        "standards_used": ["GHG Protocol", "DEFRA 2024"],
        "standards": ["GHG Protocol"],
        "reporting_period": "2024",
        "period": "2024",
    }
    base.update(overrides)
    return base


# ==============================================================================
# GHG PROTOCOL TESTS (7)
# ==============================================================================


class TestGHGProtocol:
    """Tests for GHG Protocol Scope 3 compliance checking."""

    def test_ghg_protocol_full_pass(self, engine):
        """Full result with all fields present passes GHG Protocol."""
        result = engine.check_ghg_protocol(_full_result())
        assert result.status == ComplianceStatus.PASS
        assert result.framework == ComplianceFramework.GHG_PROTOCOL

    def test_ghg_protocol_missing_total(self, engine):
        """Missing total_co2e fails GHG Protocol (GHG-BT-001 CRITICAL)."""
        data = _full_result(total_co2e=None)
        result = engine.check_ghg_protocol(data)
        assert result.status == ComplianceStatus.FAIL
        rule_codes = [f["rule_code"] for f in result.findings]
        assert "GHG-BT-001" in rule_codes

    def test_ghg_protocol_missing_method(self, engine):
        """Missing method produces a FAIL on GHG-BT-002."""
        data = _full_result(method=None, calculation_method=None)
        result = engine.check_ghg_protocol(data)
        assert result.status == ComplianceStatus.FAIL
        rule_codes = [f["rule_code"] for f in result.findings]
        assert "GHG-BT-002" in rule_codes

    def test_ghg_protocol_missing_ef_sources(self, engine):
        """Missing ef_sources produces a FAIL on GHG-BT-003."""
        data = _full_result(ef_sources=None, ef_source=None)
        result = engine.check_ghg_protocol(data)
        assert result.status == ComplianceStatus.FAIL
        rule_codes = [f["rule_code"] for f in result.findings]
        assert "GHG-BT-003" in rule_codes

    def test_ghg_protocol_score_100(self, engine):
        """Perfect data scores 100."""
        result = engine.check_ghg_protocol(_full_result())
        assert result.score == Decimal("100.00")

    def test_ghg_protocol_has_findings(self, engine):
        """Incomplete data produces findings."""
        data = _full_result(
            method=None,
            calculation_method=None,
            ef_sources=None,
            ef_source=None,
        )
        result = engine.check_ghg_protocol(data)
        assert len(result.findings) > 0

    def test_ghg_protocol_has_recommendations(self, engine):
        """Incomplete data produces recommendations."""
        data = _full_result(exclusions=None, dqi_score=None, data_quality_score=None)
        result = engine.check_ghg_protocol(data)
        assert len(result.recommendations) > 0


# ==============================================================================
# ISO 14064 TESTS (5)
# ==============================================================================


class TestISO14064:
    """Tests for ISO 14064-1:2018 compliance checking."""

    def test_iso_full_pass(self, engine):
        """Full result passes ISO 14064."""
        result = engine.check_iso_14064(_full_result())
        assert result.status == ComplianceStatus.PASS
        assert result.framework == ComplianceFramework.ISO_14064

    def test_iso_missing_uncertainty(self, engine):
        """Missing uncertainty analysis fails ISO-BT-002."""
        data = _full_result(
            uncertainty_analysis=None,
            uncertainty=None,
            uncertainty_percentage=None,
        )
        result = engine.check_iso_14064(data)
        assert result.status == ComplianceStatus.FAIL
        rule_codes = [f["rule_code"] for f in result.findings]
        assert "ISO-BT-002" in rule_codes

    def test_iso_missing_base_year(self, engine):
        """Missing base year fails ISO-BT-003."""
        data = _full_result(base_year=None)
        result = engine.check_iso_14064(data)
        assert result.status == ComplianceStatus.FAIL
        rule_codes = [f["rule_code"] for f in result.findings]
        assert "ISO-BT-003" in rule_codes

    def test_iso_score(self, engine):
        """Full ISO-compliant data scores 100."""
        result = engine.check_iso_14064(_full_result())
        assert result.score == Decimal("100.00")

    def test_iso_methodology_required(self, engine):
        """Missing methodology fails ISO-BT-004."""
        data = _full_result(
            methodology=None,
            method=None,
            calculation_method=None,
        )
        result = engine.check_iso_14064(data)
        rule_codes = [f["rule_code"] for f in result.findings]
        assert "ISO-BT-004" in rule_codes


# ==============================================================================
# CSRD ESRS TESTS (5)
# ==============================================================================


class TestCSRDESRS:
    """Tests for CSRD ESRS E1 Climate Change compliance checking."""

    def test_csrd_full_pass(self, engine):
        """Full result passes CSRD/ESRS."""
        result = engine.check_csrd_esrs(_full_result())
        assert result.status == ComplianceStatus.PASS

    def test_csrd_missing_methodology(self, engine):
        """Missing methodology fails CSRD-BT-002."""
        data = _full_result(
            methodology=None,
            method=None,
            calculation_method=None,
        )
        result = engine.check_csrd_esrs(data)
        assert result.status == ComplianceStatus.FAIL
        rule_codes = [f["rule_code"] for f in result.findings]
        assert "CSRD-BT-002" in rule_codes

    def test_csrd_missing_mode_breakdown(self, engine):
        """Missing mode breakdown fails CSRD-BT-004."""
        data = _full_result(mode_breakdown=None, by_mode=None)
        result = engine.check_csrd_esrs(data)
        assert result.status == ComplianceStatus.FAIL
        rule_codes = [f["rule_code"] for f in result.findings]
        assert "CSRD-BT-004" in rule_codes

    def test_csrd_targets_recommended(self, engine):
        """Missing targets produces WARNING (CSRD-BT-003)."""
        data = _full_result(targets=None, reduction_targets=None)
        result = engine.check_csrd_esrs(data)
        # Should be WARNING, not FAIL
        rule_codes = [f["rule_code"] for f in result.findings]
        assert "CSRD-BT-003" in rule_codes
        finding = next(f for f in result.findings if f["rule_code"] == "CSRD-BT-003")
        assert finding["status"] == ComplianceStatus.WARNING.value

    def test_csrd_score(self, engine):
        """Full CSRD-compliant data scores 100."""
        result = engine.check_csrd_esrs(_full_result())
        assert result.score == Decimal("100.00")


# ==============================================================================
# CDP TESTS (7)
# ==============================================================================


class TestCDP:
    """Tests for CDP Climate Change Questionnaire compliance checking."""

    def test_cdp_full_pass_both_rf(self, engine):
        """Both with_rf and without_rf present passes CR-BT-001."""
        result = engine.check_cdp(_full_result())
        assert result.status == ComplianceStatus.PASS

    def test_cdp_warning_no_with_rf(self, engine):
        """Only without_rf present produces WARNING on CR-BT-001."""
        data = _full_result(
            total_co2e_with_rf=None,
            with_rf=None,
        )
        result = engine.check_cdp(data)
        rule_codes = [f["rule_code"] for f in result.findings]
        assert "CR-BT-001" in rule_codes
        finding = next(f for f in result.findings if f["rule_code"] == "CR-BT-001")
        assert finding["status"] == ComplianceStatus.WARNING.value

    def test_cdp_warning_no_without_rf(self, engine):
        """Only with_rf present produces WARNING on CR-BT-001."""
        data = _full_result(
            total_co2e_without_rf=None,
            without_rf=None,
        )
        result = engine.check_cdp(data)
        rule_codes = [f["rule_code"] for f in result.findings]
        assert "CR-BT-001" in rule_codes

    def test_cdp_fail_no_rf_data(self, engine):
        """No RF data at all with air travel included fails CR-BT-001."""
        data = _full_result(
            total_co2e_with_rf=None,
            with_rf=None,
            total_co2e_without_rf=None,
            without_rf=None,
        )
        result = engine.check_cdp(data)
        assert result.status == ComplianceStatus.FAIL
        rule_codes = [f["rule_code"] for f in result.findings]
        assert "CR-BT-001" in rule_codes

    def test_cdp_mode_breakdown_required(self, engine):
        """Missing mode breakdown fails CR-BT-003."""
        data = _full_result(mode_breakdown=None, by_mode=None)
        result = engine.check_cdp(data)
        rule_codes = [f["rule_code"] for f in result.findings]
        assert "CR-BT-003" in rule_codes

    def test_cdp_verification_recommended(self, engine):
        """Missing verification produces WARNING on CDP-BT-004."""
        data = _full_result(
            verification_status=None,
            verified=None,
            assurance=None,
        )
        result = engine.check_cdp(data)
        rule_codes = [f["rule_code"] for f in result.findings]
        assert "CDP-BT-004" in rule_codes
        finding = next(f for f in result.findings if f["rule_code"] == "CDP-BT-004")
        assert finding["status"] == ComplianceStatus.WARNING.value

    def test_cdp_score(self, engine):
        """Full CDP-compliant data scores 100."""
        result = engine.check_cdp(_full_result())
        assert result.score == Decimal("100.00")


# ==============================================================================
# SBTi TESTS (5)
# ==============================================================================


class TestSBTi:
    """Tests for Science Based Targets initiative compliance checking."""

    def test_sbti_pass_rf_in_target(self, engine):
        """RF included in target boundary passes CR-BT-002."""
        result = engine.check_sbti(_full_result())
        assert result.status in (ComplianceStatus.PASS, ComplianceStatus.WARNING)
        # CR-BT-002 should pass
        failed_rules = [
            f["rule_code"] for f in result.findings
            if f["status"] == ComplianceStatus.FAIL.value
        ]
        assert "CR-BT-002" not in failed_rules

    def test_sbti_warning_rf_excluded(self, engine):
        """No RF data produces WARNING on CR-BT-002."""
        data = _full_result(
            total_co2e_with_rf=None,
            with_rf=None,
            rf_inclusion=None,
            rf_in_target=None,
        )
        result = engine.check_sbti(data)
        rule_codes = [f["rule_code"] for f in result.findings]
        assert "CR-BT-002" in rule_codes

    def test_sbti_target_coverage(self, engine):
        """Missing target coverage produces WARNING on SBTI-BT-003."""
        data = _full_result(target_coverage=None, sbti_coverage=None)
        result = engine.check_sbti(data)
        rule_codes = [f["rule_code"] for f in result.findings]
        assert "SBTI-BT-003" in rule_codes

    def test_sbti_progress_tracking(self, engine):
        """Missing progress tracking produces WARNING on SBTI-BT-004."""
        data = _full_result(
            progress_tracking=None,
            year_over_year_change=None,
            trend=None,
        )
        result = engine.check_sbti(data)
        rule_codes = [f["rule_code"] for f in result.findings]
        assert "SBTI-BT-004" in rule_codes

    def test_sbti_score(self, engine):
        """Full SBTi-compliant data scores high (>= 80)."""
        result = engine.check_sbti(_full_result())
        assert result.score >= Decimal("80.00")


# ==============================================================================
# SB 253 TESTS (4)
# ==============================================================================


class TestSB253:
    """Tests for California SB 253 compliance checking."""

    def test_sb253_pass_full(self, engine):
        """Full result passes SB 253."""
        result = engine.check_sb_253(_full_result())
        # Check no FAIL findings
        failed = [
            f for f in result.findings
            if f["status"] == ComplianceStatus.FAIL.value
        ]
        assert len(failed) == 0

    def test_sb253_missing_assurance(self, engine):
        """Missing assurance produces WARNING on CR-BT-009."""
        data = _full_result(
            assurance_opinion=None,
            assurance=None,
            verification_status=None,
        )
        result = engine.check_sb_253(data)
        rule_codes = [f["rule_code"] for f in result.findings]
        assert "CR-BT-009" in rule_codes

    def test_sb253_below_materiality(self, engine):
        """Cat 6 below 1% of Scope 3 produces WARNING on CR-BT-004."""
        data = _full_result(
            total_co2e=100.0,  # 0.2% of 50000
            total_scope3_co2e=50000.0,
        )
        result = engine.check_sb_253(data)
        rule_codes = [f["rule_code"] for f in result.findings]
        assert "CR-BT-004" in rule_codes

    def test_sb253_methodology(self, engine):
        """Missing methodology fails SB253-BT-002."""
        data = _full_result(
            methodology=None,
            method=None,
            calculation_method=None,
        )
        result = engine.check_sb_253(data)
        rule_codes = [f["rule_code"] for f in result.findings]
        assert "SB253-BT-002" in rule_codes


# ==============================================================================
# GRI TESTS (4)
# ==============================================================================


class TestGRI:
    """Tests for GRI 305 Emissions Standard compliance checking."""

    def test_gri_pass(self, engine):
        """Full result passes GRI."""
        result = engine.check_gri(_full_result())
        assert result.status == ComplianceStatus.PASS

    def test_gri_missing_gases(self, engine):
        """Missing gases_included produces WARNING on GRI-BT-002."""
        data = _full_result(
            gases_included=None,
            emission_gases=None,
            gases=None,
        )
        result = engine.check_gri(data)
        rule_codes = [f["rule_code"] for f in result.findings]
        assert "GRI-BT-002" in rule_codes

    def test_gri_base_year(self, engine):
        """Missing base year produces WARNING on GRI-BT-003."""
        data = _full_result(base_year=None)
        result = engine.check_gri(data)
        rule_codes = [f["rule_code"] for f in result.findings]
        assert "GRI-BT-003" in rule_codes

    def test_gri_score(self, engine):
        """Full GRI-compliant data scores 100."""
        result = engine.check_gri(_full_result())
        assert result.score == Decimal("100.00")


# ==============================================================================
# DOUBLE-COUNTING PREVENTION TESTS (7)
# ==============================================================================


class TestDoubleCounting:
    """Tests for 7 double-counting prevention rules (DC-BT-001 to DC-BT-007)."""

    def test_dc_001_company_owned_rejected(self, engine):
        """DC-BT-001: Company-owned vehicle flagged."""
        trips = [{"vehicle_ownership": "company_owned", "mode": "road"}]
        findings = engine.check_double_counting(trips)
        codes = [f["rule_code"] for f in findings]
        assert "DC-BT-001" in codes

    def test_dc_002_commuting_rejected(self, engine):
        """DC-BT-002: Commuting trip flagged."""
        trips = [{"purpose": "commuting", "mode": "road"}]
        findings = engine.check_double_counting(trips)
        codes = [f["rule_code"] for f in findings]
        assert "DC-BT-002" in codes

    def test_dc_003_freight_rejected(self, engine):
        """DC-BT-003: Freight trip flagged."""
        trips = [{"trip_type": "freight", "mode": "road"}]
        findings = engine.check_double_counting(trips)
        codes = [f["rule_code"] for f in findings]
        assert "DC-BT-003" in codes

    def test_dc_004_cat4_overlap_flagged(self, engine):
        """DC-BT-004: Trip also reported in Category 4."""
        trips = [{"reported_in_cat4": True, "mode": "air"}]
        findings = engine.check_double_counting(trips)
        codes = [f["rule_code"] for f in findings]
        assert "DC-BT-004" in codes

    def test_dc_005_scope1_fleet_flagged(self, engine):
        """DC-BT-005: Trip also reported in Scope 1 fleet."""
        trips = [{"reported_in_scope1": True, "mode": "road"}]
        findings = engine.check_double_counting(trips)
        codes = [f["rule_code"] for f in findings]
        assert "DC-BT-005" in codes

    def test_dc_006_hotel_food_flagged(self, engine):
        """DC-BT-006: Hotel emissions include meals."""
        trips = [{"mode": "hotel", "includes_meals": True}]
        findings = engine.check_double_counting(trips)
        codes = [f["rule_code"] for f in findings]
        assert "DC-BT-006" in codes

    def test_dc_007_wtt_cat3_flagged(self, engine):
        """DC-BT-007: WTT reported in both Cat 6 and Cat 3."""
        trips = [
            {
                "mode": "air",
                "wtt_separately_reported": True,
                "wtt_reported_in_cat3": True,
            }
        ]
        findings = engine.check_double_counting(trips)
        codes = [f["rule_code"] for f in findings]
        assert "DC-BT-007" in codes


# ==============================================================================
# INTEGRATION / SUMMARY / BOUNDARY TESTS (6)
# ==============================================================================


class TestIntegration:
    """Integration tests for check_all_frameworks, summary, and category boundary."""

    def test_check_all_frameworks_returns_all(self, engine):
        """check_all_frameworks returns results for all enabled frameworks."""
        results = engine.check_all_frameworks(_full_result())
        assert isinstance(results, dict)
        assert len(results) >= 7

    def test_compliance_summary_overall_score(self, engine):
        """get_compliance_summary returns weighted overall score."""
        results = engine.check_all_frameworks(_full_result())
        summary = engine.get_compliance_summary(results)
        assert "overall_score" in summary
        assert summary["overall_score"] > 0
        assert "overall_status" in summary
        assert "frameworks_checked" in summary
        assert summary["frameworks_checked"] >= 7

    def test_category_boundary_business_cat6(self, engine):
        """Business travel classifies as CATEGORY_6."""
        trip = {
            "mode": "air",
            "purpose": "client_visit",
            "vehicle_ownership": "third_party",
        }
        result = engine.check_category_boundary(trip)
        assert result["classification"] == BoundaryClassification.CATEGORY_6.value

    def test_category_boundary_commuting_cat7(self, engine):
        """Commuting classifies as CATEGORY_7."""
        trip = {
            "mode": "road",
            "purpose": "commuting",
            "vehicle_ownership": "personal",
        }
        result = engine.check_category_boundary(trip)
        assert result["classification"] == BoundaryClassification.CATEGORY_7.value

    def test_category_boundary_company_car_scope1(self, engine):
        """Company-owned car classifies as SCOPE_1."""
        trip = {
            "mode": "road",
            "purpose": "business",
            "vehicle_ownership": "company_owned",
        }
        result = engine.check_category_boundary(trip)
        assert result["classification"] == BoundaryClassification.SCOPE_1.value

    def test_singleton_pattern(self):
        """get_instance returns the same object on repeated calls."""
        with patch(
            "greenlang.business_travel.compliance_checker.get_config"
        ) as mock_config:
            mock_config.return_value = _make_mock_config()
            inst1 = ComplianceCheckerEngine.get_instance()
            inst2 = ComplianceCheckerEngine.get_instance()
            assert inst1 is inst2
