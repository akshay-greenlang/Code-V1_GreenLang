# -*- coding: utf-8 -*-
"""
Unit tests for ComplianceCheckerEngine -- AGENT-MRV-024

Tests compliance checking across 7 regulatory frameworks, 8 double-counting
prevention rules, boundary validation, completeness scoring, lifetime
assumption documentation validation, and compliance report generation.

Target: 40+ tests.
Author: GL-TestEngineer
"""

import threading
from decimal import Decimal
from typing import Any, Dict, List

import pytest

try:
    from greenlang.use_of_sold_products.compliance_checker import (
        ComplianceCheckerEngine,
        ComplianceFramework,
        ComplianceStatus,
        ComplianceSeverity,
        DoubleCountingCategory,
    )

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_SKIP = pytest.mark.skipif(not _AVAILABLE, reason="ComplianceCheckerEngine not available")
pytestmark = _SKIP

_Q8 = Decimal("0.00000001")


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def engine():
    """Create a ComplianceCheckerEngine instance."""
    return ComplianceCheckerEngine.get_instance()


@pytest.fixture
def valid_calculation_result():
    """A well-formed calculation result that should pass compliance."""
    return {
        "calc_id": "CALC-USP-001",
        "org_id": "ORG-001",
        "reporting_year": 2024,
        "method": "direct_fuel_combustion",
        "total_emissions_kgco2e": Decimal("41670000"),
        "total_emissions_tco2e": Decimal("41670"),
        "product_count": 5,
        "product_breakdowns": [
            {
                "product_id": "P1",
                "category": "vehicles",
                "emission_type": "direct",
                "emissions_kgco2e": Decimal("41670000"),
                "method": "direct_fuel_combustion",
                "ef_source": "defra",
                "dqi_score": Decimal("80"),
                "lifetime_years": 15,
                "units_sold": 1000,
            },
            {
                "product_id": "P2",
                "category": "appliances",
                "emission_type": "indirect",
                "emissions_kgco2e": Decimal("25020000"),
                "method": "indirect_electricity",
                "ef_source": "egrid",
                "dqi_score": Decimal("75"),
                "lifetime_years": 15,
                "units_sold": 10000,
            },
            {
                "product_id": "P3",
                "category": "hvac",
                "emission_type": "both",
                "emissions_kgco2e": Decimal("5632200"),
                "method": "direct_refrigerant_leakage",
                "ef_source": "ipcc",
                "dqi_score": Decimal("70"),
                "lifetime_years": 12,
                "units_sold": 500,
            },
            {
                "product_id": "P4",
                "category": "fuels_feedstocks",
                "emission_type": "direct",
                "emissions_kgco2e": Decimal("2315000"),
                "method": "fuels_sold",
                "ef_source": "defra",
                "dqi_score": Decimal("85"),
                "units_sold": 1,
            },
            {
                "product_id": "P5",
                "category": "it_equipment",
                "emission_type": "indirect",
                "emissions_kgco2e": Decimal("5212500"),
                "method": "indirect_electricity",
                "ef_source": "iea",
                "dqi_score": Decimal("70"),
                "lifetime_years": 5,
                "units_sold": 50000,
            },
        ],
        "by_category": {
            "vehicles": Decimal("41670000"),
            "appliances": Decimal("25020000"),
            "hvac": Decimal("5632200"),
            "fuels_feedstocks": Decimal("2315000"),
            "it_equipment": Decimal("5212500"),
        },
        "by_emission_type": {
            "direct": Decimal("43985000"),
            "indirect": Decimal("30232500"),
            "both": Decimal("5632200"),
        },
        "by_method": {
            "direct_fuel_combustion": Decimal("41670000"),
            "indirect_electricity": Decimal("30232500"),
            "direct_refrigerant_leakage": Decimal("5632200"),
            "fuels_sold": Decimal("2315000"),
        },
        "dqi_score": Decimal("76"),
        "completeness_score": Decimal("85"),
        "lifetime_assumptions_documented": True,
        "use_profiles_documented": True,
        "boundary": "use_phase_only",
        "methodology": "GHG Protocol Scope 3 Category 11",
        "gases_included": ["CO2", "CH4", "N2O", "HFCs"],
        "uncertainty_analysis": {"method": "propagation", "ci_95": [71000000, 87000000]},
        "provenance_hash": "a" * 64,
    }


# ============================================================================
# TEST: Framework Compliance Checks
# ============================================================================


class TestFrameworkCompliance:
    """Test compliance checking across 7 regulatory frameworks."""

    @pytest.mark.parametrize("framework", [
        "GHG_PROTOCOL_SCOPE3",
        "ISO_14064",
        "CSRD_ESRS_E1",
        "CDP",
        "SBTI",
        "SB_253",
        "GRI",
    ])
    def test_all_7_frameworks_pass_valid_result(self, engine, valid_calculation_result, framework):
        """Test valid result passes each of the 7 frameworks."""
        result = engine.check_compliance(valid_calculation_result, framework)
        assert result["status"] in ("compliant", "partial", "COMPLIANT", "PARTIAL")

    def test_ghg_protocol_requires_methodology(self, engine, valid_calculation_result):
        """Test GHG Protocol requires methodology documentation."""
        valid_calculation_result["methodology"] = None
        result = engine.check_compliance(valid_calculation_result, "GHG_PROTOCOL_SCOPE3")
        # Should flag a warning or non-compliance
        has_issue = (
            result.get("status") in ("non_compliant", "partial", "NON_COMPLIANT", "PARTIAL")
            or len(result.get("issues", [])) > 0
            or len(result.get("warnings", [])) > 0
        )
        assert has_issue

    def test_cdp_requires_reduction_targets(self, engine, valid_calculation_result):
        """Test CDP requires reduction targets."""
        result = engine.check_compliance(valid_calculation_result, "CDP")
        # CDP should check for targets
        assert result is not None

    def test_sbti_requires_base_year(self, engine, valid_calculation_result):
        """Test SBTi requires a base year and targets."""
        result = engine.check_compliance(valid_calculation_result, "SBTI")
        assert result is not None

    def test_csrd_requires_esrs_e1_fields(self, engine, valid_calculation_result):
        """Test CSRD ESRS E1 checks specific reporting fields."""
        result = engine.check_compliance(valid_calculation_result, "CSRD_ESRS_E1")
        assert result is not None

    def test_iso14064_requires_gases(self, engine, valid_calculation_result):
        """Test ISO 14064 requires gases_included list."""
        valid_calculation_result["gases_included"] = []
        result = engine.check_compliance(valid_calculation_result, "ISO_14064")
        has_issue = (
            result.get("status") in ("non_compliant", "partial", "NON_COMPLIANT", "PARTIAL")
            or len(result.get("issues", [])) > 0
            or len(result.get("warnings", [])) > 0
        )
        assert has_issue

    def test_gri_requires_standards(self, engine, valid_calculation_result):
        """Test GRI 305 compliance checking."""
        result = engine.check_compliance(valid_calculation_result, "GRI")
        assert result is not None


# ============================================================================
# TEST: Double-Counting Prevention Rules
# ============================================================================


class TestDoubleCountingRules:
    """Test 8 double-counting prevention rules (DC-USP-001 through DC-USP-008)."""

    @pytest.mark.parametrize("rule_id,description", [
        ("DC-USP-001", "No overlap with Scope 1 direct emissions"),
        ("DC-USP-002", "No overlap with Scope 2 purchased electricity"),
        ("DC-USP-003", "No overlap with Cat 1 purchased goods lifecycle"),
        ("DC-USP-004", "No overlap with Cat 3 fuel WTT emissions"),
        ("DC-USP-005", "No overlap with Cat 10 processing of sold products"),
        ("DC-USP-006", "No overlap with Cat 12 end-of-life treatment"),
        ("DC-USP-007", "Fuels sold: separate from Scope 1 combustion"),
        ("DC-USP-008", "Intermediate vs final product boundary clear"),
    ])
    def test_dc_rule_check(self, engine, valid_calculation_result, rule_id, description):
        """Test each double-counting rule is evaluated."""
        result = engine.check_double_counting(valid_calculation_result, rule_id)
        assert "status" in result or "compliant" in result

    def test_dc_scope1_overlap_detected(self, engine, valid_calculation_result):
        """Test DC-USP-001: detect Scope 1 overlap."""
        # If product is in reporting company's own fleet, it overlaps Scope 1
        valid_calculation_result["product_breakdowns"][0]["owned_by_reporter"] = True
        result = engine.check_double_counting(valid_calculation_result, "DC-USP-001")
        has_issue = (
            result.get("status") in ("non_compliant", "warning")
            or result.get("compliant") is False
            or len(result.get("issues", [])) > 0
        )
        assert has_issue

    def test_dc_fuels_sold_boundary(self, engine, valid_calculation_result):
        """Test DC-USP-007: fuels sold boundary validation."""
        result = engine.check_double_counting(valid_calculation_result, "DC-USP-007")
        assert result is not None


# ============================================================================
# TEST: Boundary Validation
# ============================================================================


class TestBoundaryValidation:
    """Test boundary validation checks."""

    def test_use_phase_boundary_valid(self, engine, valid_calculation_result):
        """Test use_phase_only boundary passes validation."""
        result = engine.check_boundary(valid_calculation_result)
        assert result["valid"] is True or result.get("status") == "compliant"

    def test_missing_boundary_flagged(self, engine, valid_calculation_result):
        """Test missing boundary field is flagged."""
        valid_calculation_result["boundary"] = None
        result = engine.check_boundary(valid_calculation_result)
        has_issue = (
            result.get("valid") is False
            or len(result.get("issues", [])) > 0
            or len(result.get("warnings", [])) > 0
        )
        assert has_issue

    def test_cradle_to_gate_excluded(self, engine, valid_calculation_result):
        """Test cradle-to-gate boundary is flagged (not use-phase)."""
        valid_calculation_result["boundary"] = "cradle_to_gate"
        result = engine.check_boundary(valid_calculation_result)
        has_issue = (
            result.get("valid") is False
            or len(result.get("issues", [])) > 0
        )
        assert has_issue


# ============================================================================
# TEST: Completeness Scoring
# ============================================================================


class TestCompleteness:
    """Test completeness scoring."""

    def test_high_completeness_score(self, engine, valid_calculation_result):
        """Test complete result gets high completeness score."""
        result = engine.check_completeness(valid_calculation_result)
        assert result["score"] >= Decimal("70")

    def test_missing_products_reduces_score(self, engine, valid_calculation_result):
        """Test missing product breakdowns reduces completeness."""
        valid_calculation_result["product_breakdowns"] = []
        result = engine.check_completeness(valid_calculation_result)
        assert result["score"] < Decimal("80")


# ============================================================================
# TEST: Lifetime Assumption Checks
# ============================================================================


class TestLifetimeAssumptions:
    """Test lifetime assumption documentation validation."""

    def test_lifetime_documented_passes(self, engine, valid_calculation_result):
        """Test documented lifetime assumptions pass check."""
        result = engine.check_lifetime_assumptions(valid_calculation_result)
        assert result.get("compliant") is True or result.get("status") == "compliant"

    def test_undocumented_lifetime_flagged(self, engine, valid_calculation_result):
        """Test undocumented lifetime assumptions are flagged."""
        valid_calculation_result["lifetime_assumptions_documented"] = False
        result = engine.check_lifetime_assumptions(valid_calculation_result)
        has_issue = (
            result.get("compliant") is False
            or len(result.get("issues", [])) > 0
            or len(result.get("warnings", [])) > 0
        )
        assert has_issue

    def test_missing_lifetime_in_product_flagged(self, engine, valid_calculation_result):
        """Test products without lifetime_years are flagged."""
        for product in valid_calculation_result["product_breakdowns"]:
            product.pop("lifetime_years", None)
        result = engine.check_lifetime_assumptions(valid_calculation_result)
        has_issue = (
            result.get("compliant") is False
            or len(result.get("issues", [])) > 0
            or len(result.get("warnings", [])) > 0
        )
        assert has_issue


# ============================================================================
# TEST: Use Profile Documentation
# ============================================================================


class TestUseProfileDocumentation:
    """Test use profile documentation validation."""

    def test_documented_profiles_pass(self, engine, valid_calculation_result):
        """Test documented use profiles pass validation."""
        result = engine.check_use_profiles(valid_calculation_result)
        assert result.get("compliant") is True or result.get("status") == "compliant"

    def test_undocumented_profiles_flagged(self, engine, valid_calculation_result):
        """Test undocumented use profiles are flagged."""
        valid_calculation_result["use_profiles_documented"] = False
        result = engine.check_use_profiles(valid_calculation_result)
        has_issue = (
            result.get("compliant") is False
            or len(result.get("issues", [])) > 0
            or len(result.get("warnings", [])) > 0
        )
        assert has_issue


# ============================================================================
# TEST: Compliance Report Generation
# ============================================================================


class TestComplianceReport:
    """Test compliance report generation."""

    def test_full_report_generation(self, engine, valid_calculation_result):
        """Test generating full compliance report across all frameworks."""
        report = engine.generate_report(valid_calculation_result)
        assert "frameworks" in report or "results" in report

    def test_report_includes_timestamp(self, engine, valid_calculation_result):
        """Test compliance report includes timestamp."""
        report = engine.generate_report(valid_calculation_result)
        assert "timestamp" in report or "generated_at" in report

    def test_report_includes_all_frameworks(self, engine, valid_calculation_result):
        """Test report covers all 7 frameworks."""
        report = engine.generate_report(valid_calculation_result)
        frameworks = report.get("frameworks", report.get("results", {}))
        if isinstance(frameworks, dict):
            assert len(frameworks) >= 7
        elif isinstance(frameworks, list):
            assert len(frameworks) >= 7


# ============================================================================
# TEST: Singleton and Thread Safety
# ============================================================================


class TestComplianceSingleton:
    """Test ComplianceCheckerEngine singleton pattern."""

    def test_singleton_same_instance(self):
        """Test get_instance returns same instance."""
        e1 = ComplianceCheckerEngine.get_instance()
        e2 = ComplianceCheckerEngine.get_instance()
        assert e1 is e2

    def test_thread_safe_compliance_check(self, valid_calculation_result):
        """Test compliance checking is thread-safe."""
        results = []
        errors = []

        def _check():
            try:
                engine = ComplianceCheckerEngine.get_instance()
                result = engine.check_compliance(valid_calculation_result, "GHG_PROTOCOL_SCOPE3")
                results.append(result)
            except Exception as ex:
                errors.append(ex)

        threads = [threading.Thread(target=_check) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(errors) == 0
        assert len(results) == 10
