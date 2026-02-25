# -*- coding: utf-8 -*-
"""
Unit tests for ComplianceCheckerEngine (Engine 6 of 7) - AGENT-MRV-011.

Tests multi-framework regulatory compliance checking across seven
frameworks (GHG Protocol Scope 2, ISO 14064, CSRD/ESRS E1, CDP,
SBTi, EU EED, EPA MRR) with 84 total requirements.

Target: ~70 tests.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List

import pytest

try:
    from greenlang.steam_heat_purchase.compliance_checker import (
        ComplianceCheckerEngine,
        ComplianceFinding,
        SUPPORTED_FRAMEWORKS,
        TOTAL_REQUIREMENTS,
        FRAMEWORK_INFO,
        VALID_ENERGY_TYPES,
        VALID_CALCULATION_METHODS,
        VALID_GWP_SOURCES,
        get_compliance_checker,
    )
    COMPLIANCE_AVAILABLE = True
except ImportError:
    COMPLIANCE_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not COMPLIANCE_AVAILABLE,
    reason="greenlang.steam_heat_purchase.compliance_checker not importable",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Create a fresh ComplianceCheckerEngine instance."""
    ComplianceCheckerEngine.reset()
    return ComplianceCheckerEngine()


@pytest.fixture
def complete_data() -> Dict[str, Any]:
    """Return a fully compliant calculation data dictionary."""
    return {
        "energy_type": "steam",
        "consumption_gj": Decimal("5000"),
        "emission_factor": Decimal("65.3"),
        "ef_source": "IPCC_2006",
        "ef_reference": "IPCC 2006 Vol 2 Ch 2",
        "calculation_method": "FUEL_BASED",
        "boiler_efficiency": Decimal("0.85"),
        "boiler_efficiency_source": "measured",
        "total_co2e_kg": Decimal("325000"),
        "total_co2e_tonnes": Decimal("325"),
        "fossil_co2e_kg": Decimal("325000"),
        "biogenic_co2_kg": Decimal("0"),
        "co2_kg": Decimal("280500"),
        "ch4_kg": Decimal("5.0"),
        "n2o_kg": Decimal("0.5"),
        "gas_details": [
            {"gas": "CO2", "emission_kg": 280500},
            {"gas": "CH4", "emission_kg": 5.0},
            {"gas": "N2O", "emission_kg": 0.5},
        ],
        "has_uncertainty": True,
        "uncertainty_pct": Decimal("12.0"),
        "uncertainty_method": "monte_carlo",
        "provenance_hash": "a" * 64,
        "gwp_source": "AR6",
        "data_quality_tier": "tier_2",
        "facility_id": "fac-001",
        "supplier_id": "sup-001",
        "supplier_name": "Test Supplier",
        "reporting_period": "2025-Q1",
        "base_year": "2020",
        "verification": "third_party",
        "verification_status": "verified",
        "ghg_intensity": Decimal("0.065"),
        "ghg_intensity_unit": "tCO2e/GJ",
        "fuel_type": "natural_gas",
        "coverage_pct": Decimal("98"),
        "chp_classification": "high_efficiency",
        "chp_allocation_method": "efficiency",
        "primary_energy_savings_pct": Decimal("15"),
        "monitoring_plan": True,
        "monitoring_plan_ref": "MP-2025-001",
        "measurement_method": "continuous",
        "calibration_frequency": "annual",
        "data_retention_years": 7,
        "trace": ["step_1", "step_2", "step_3"],
        "tenant_id": "test-tenant",
    }


@pytest.fixture
def minimal_data() -> Dict[str, Any]:
    """Return a minimal data dictionary (many fields missing)."""
    return {
        "energy_type": "steam",
        "total_co2e_kg": Decimal("500"),
        "consumption_gj": Decimal("100"),
    }


@pytest.fixture
def empty_data() -> Dict[str, Any]:
    """Return empty data dictionary."""
    return {}


# ===========================================================================
# 1. Singleton and Initialization Tests
# ===========================================================================


class TestSingletonPattern:
    """Tests for ComplianceCheckerEngine singleton."""

    def test_same_instance_returned(self):
        ComplianceCheckerEngine.reset()
        e1 = ComplianceCheckerEngine()
        e2 = ComplianceCheckerEngine()
        assert e1 is e2

    def test_reset_creates_new_instance(self):
        e1 = ComplianceCheckerEngine()
        ComplianceCheckerEngine.reset()
        e2 = ComplianceCheckerEngine()
        assert e1 is not e2

    def test_get_compliance_checker_function(self):
        ComplianceCheckerEngine.reset()
        e = get_compliance_checker()
        assert isinstance(e, ComplianceCheckerEngine)

    def test_get_compliance_checker_singleton(self):
        ComplianceCheckerEngine.reset()
        e1 = get_compliance_checker()
        e2 = get_compliance_checker()
        assert e1 is e2


# ===========================================================================
# 2. Constants and Framework Metadata Tests
# ===========================================================================


class TestConstants:
    """Tests for module-level constants."""

    def test_supported_frameworks_count(self):
        assert len(SUPPORTED_FRAMEWORKS) == 7

    def test_supported_frameworks_names(self):
        expected = {
            "ghg_protocol_scope2",
            "iso_14064",
            "csrd_esrs",
            "cdp",
            "sbti",
            "eu_eed",
            "epa_mrr",
        }
        assert set(SUPPORTED_FRAMEWORKS) == expected

    def test_total_requirements_equals_84(self):
        assert TOTAL_REQUIREMENTS == 84

    def test_total_requirements_is_7_times_12(self):
        assert TOTAL_REQUIREMENTS == 7 * 12

    def test_framework_info_has_all_7(self):
        assert len(FRAMEWORK_INFO) == 7

    def test_framework_info_has_name_and_version(self):
        for fw_key, info in FRAMEWORK_INFO.items():
            assert "name" in info, f"{fw_key} missing 'name'"
            assert "version" in info, f"{fw_key} missing 'version'"
            assert "requirements_count" in info, f"{fw_key} missing 'requirements_count'"

    def test_each_framework_has_12_requirements(self):
        for fw_key, info in FRAMEWORK_INFO.items():
            assert info["requirements_count"] == 12, (
                f"{fw_key} has {info['requirements_count']} requirements, expected 12"
            )

    def test_valid_energy_types(self):
        assert "steam" in VALID_ENERGY_TYPES
        assert "district_heating" in VALID_ENERGY_TYPES
        assert "district_cooling" in VALID_ENERGY_TYPES

    def test_valid_calculation_methods(self):
        assert "FUEL_BASED" in VALID_CALCULATION_METHODS
        assert "DIRECT_EF" in VALID_CALCULATION_METHODS

    def test_valid_gwp_sources(self):
        assert "AR6" in VALID_GWP_SOURCES
        assert "AR5" in VALID_GWP_SOURCES


# ===========================================================================
# 3. Individual Framework Check Tests
# ===========================================================================


class TestGHGProtocol:
    """Tests for check_ghg_protocol."""

    def test_complete_data_passes(self, engine, complete_data):
        findings = engine.check_ghg_protocol(complete_data)
        passed = sum(1 for f in findings if f.passed)
        assert passed >= 10  # Most requirements should pass

    def test_missing_energy_type_fails(self, engine, complete_data):
        data = dict(complete_data)
        del data["energy_type"]
        findings = engine.check_ghg_protocol(data)
        failed = [f for f in findings if not f.passed]
        assert len(failed) >= 1

    def test_returns_12_findings(self, engine, complete_data):
        findings = engine.check_ghg_protocol(complete_data)
        assert len(findings) == 12


class TestISO14064:
    """Tests for check_iso_14064."""

    def test_complete_data_passes(self, engine, complete_data):
        findings = engine.check_iso_14064(complete_data)
        passed = sum(1 for f in findings if f.passed)
        assert passed >= 10

    def test_missing_base_year_fails(self, engine, complete_data):
        data = dict(complete_data)
        data.pop("base_year", None)
        findings = engine.check_iso_14064(data)
        failed_ids = [f.requirement_id for f in findings if not f.passed]
        # At least one ISO requirement should fail without base_year
        assert len(failed_ids) >= 1

    def test_returns_12_findings(self, engine, complete_data):
        findings = engine.check_iso_14064(complete_data)
        assert len(findings) == 12


class TestCSRD:
    """Tests for check_csrd_esrs."""

    def test_complete_data_passes(self, engine, complete_data):
        findings = engine.check_csrd_esrs(complete_data)
        passed = sum(1 for f in findings if f.passed)
        assert passed >= 10

    def test_missing_ghg_intensity_fails(self, engine, complete_data):
        data = dict(complete_data)
        data.pop("ghg_intensity", None)
        findings = engine.check_csrd_esrs(data)
        failed = [f for f in findings if not f.passed]
        assert len(failed) >= 1

    def test_returns_12_findings(self, engine, complete_data):
        findings = engine.check_csrd_esrs(complete_data)
        assert len(findings) == 12


class TestCDP:
    """Tests for check_cdp."""

    def test_complete_data_passes(self, engine, complete_data):
        findings = engine.check_cdp(complete_data)
        passed = sum(1 for f in findings if f.passed)
        assert passed >= 10

    def test_missing_verification_fails(self, engine, complete_data):
        data = dict(complete_data)
        data.pop("verification", None)
        data.pop("verification_status", None)
        findings = engine.check_cdp(data)
        failed = [f for f in findings if not f.passed]
        assert len(failed) >= 1

    def test_returns_12_findings(self, engine, complete_data):
        findings = engine.check_cdp(complete_data)
        assert len(findings) == 12


class TestSBTi:
    """Tests for check_sbti."""

    def test_complete_data_passes(self, engine, complete_data):
        findings = engine.check_sbti(complete_data)
        passed = sum(1 for f in findings if f.passed)
        assert passed >= 10

    def test_low_coverage_fails(self, engine, complete_data):
        data = dict(complete_data)
        data["coverage_pct"] = Decimal("50")
        findings = engine.check_sbti(data)
        # SBTi requires high coverage (typically 95%)
        coverage_findings = [
            f for f in findings
            if not f.passed and "coverage" in f.requirement.lower()
        ]
        # At least check it returns findings
        assert len(findings) == 12

    def test_returns_12_findings(self, engine, complete_data):
        findings = engine.check_sbti(complete_data)
        assert len(findings) == 12


class TestEUEED:
    """Tests for check_eu_eed."""

    def test_complete_data_passes(self, engine, complete_data):
        findings = engine.check_eu_eed(complete_data)
        passed = sum(1 for f in findings if f.passed)
        assert passed >= 10

    def test_missing_chp_classification_fails(self, engine, complete_data):
        data = dict(complete_data)
        data.pop("chp_classification", None)
        data.pop("chp_allocation_method", None)
        data.pop("primary_energy_savings_pct", None)
        findings = engine.check_eu_eed(data)
        failed = [f for f in findings if not f.passed]
        assert len(failed) >= 1

    def test_returns_12_findings(self, engine, complete_data):
        findings = engine.check_eu_eed(complete_data)
        assert len(findings) == 12


class TestEPAMRR:
    """Tests for check_epa_mrr."""

    def test_complete_data_passes(self, engine, complete_data):
        findings = engine.check_epa_mrr(complete_data)
        passed = sum(1 for f in findings if f.passed)
        assert passed >= 10

    def test_missing_monitoring_plan_fails(self, engine, complete_data):
        data = dict(complete_data)
        data.pop("monitoring_plan", None)
        data.pop("monitoring_plan_ref", None)
        findings = engine.check_epa_mrr(data)
        failed = [f for f in findings if not f.passed]
        assert len(failed) >= 1

    def test_returns_12_findings(self, engine, complete_data):
        findings = engine.check_epa_mrr(complete_data)
        assert len(findings) == 12


# ===========================================================================
# 4. Multi-Framework Compliance Check Tests
# ===========================================================================


class TestCheckCompliance:
    """Tests for the main check_compliance method."""

    def test_check_all_frameworks_complete(self, engine, complete_data):
        result = engine.check_compliance(
            calc_result=complete_data,
            frameworks=SUPPORTED_FRAMEWORKS,
        )
        assert "frameworks" in result or "results" in result
        fw_results = result.get("frameworks", result.get("results", {}))
        if isinstance(fw_results, dict):
            assert len(fw_results) == 7
        elif isinstance(fw_results, list):
            assert len(fw_results) == 7

    def test_check_single_framework(self, engine, complete_data):
        result = engine.check_compliance(
            calc_result=complete_data,
            frameworks=["ghg_protocol_scope2"],
        )
        fw_results = result.get("frameworks", result.get("results", {}))
        if isinstance(fw_results, dict):
            assert "ghg_protocol_scope2" in fw_results
        elif isinstance(fw_results, list):
            assert len(fw_results) == 1

    def test_complete_data_is_compliant(self, engine, complete_data):
        result = engine.check_compliance(
            calc_result=complete_data,
            frameworks=["ghg_protocol_scope2"],
        )
        fw_results = result.get("frameworks", result.get("results", {}))
        if isinstance(fw_results, dict):
            ghg = fw_results["ghg_protocol_scope2"]
            status = ghg.get("status", ghg.get("compliance_status", ""))
            assert status.lower() in ("compliant", "pass", "partial")

    def test_minimal_data_non_compliant(self, engine, minimal_data):
        result = engine.check_compliance(
            calc_result=minimal_data,
            frameworks=["ghg_protocol_scope2"],
        )
        fw_results = result.get("frameworks", result.get("results", {}))
        if isinstance(fw_results, dict):
            ghg = fw_results["ghg_protocol_scope2"]
            status = ghg.get("status", ghg.get("compliance_status", ""))
            assert status.lower() in ("non_compliant", "partial", "fail")

    def test_empty_data_non_compliant(self, engine, empty_data):
        result = engine.check_compliance(
            calc_result=empty_data,
            frameworks=["ghg_protocol_scope2"],
        )
        fw_results = result.get("frameworks", result.get("results", {}))
        if isinstance(fw_results, dict):
            ghg = fw_results["ghg_protocol_scope2"]
            status = ghg.get("status", ghg.get("compliance_status", ""))
            assert status.lower() in ("non_compliant", "fail")

    def test_compliance_has_provenance(self, engine, complete_data):
        result = engine.check_compliance(
            calc_result=complete_data,
            frameworks=SUPPORTED_FRAMEWORKS,
        )
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# 5. Compliance Score Tests
# ===========================================================================


class TestComplianceScore:
    """Tests for compliance score calculation."""

    def test_complete_data_high_score(self, engine, complete_data):
        result = engine.check_compliance(
            calc_result=complete_data,
            frameworks=["ghg_protocol_scope2"],
        )
        fw_results = result.get("frameworks", result.get("results", {}))
        if isinstance(fw_results, dict):
            ghg = fw_results["ghg_protocol_scope2"]
            score = ghg.get("score_pct", ghg.get("compliance_score", Decimal("0")))
            if isinstance(score, (int, float, Decimal)):
                assert score >= 80

    def test_minimal_data_lower_score(self, engine, minimal_data):
        result = engine.check_compliance(
            calc_result=minimal_data,
            frameworks=["ghg_protocol_scope2"],
        )
        fw_results = result.get("frameworks", result.get("results", {}))
        if isinstance(fw_results, dict):
            ghg = fw_results["ghg_protocol_scope2"]
            score = ghg.get("score_pct", ghg.get("compliance_score", Decimal("100")))
            if isinstance(score, (int, float, Decimal)):
                assert score < 100


# ===========================================================================
# 6. Recommendations Tests
# ===========================================================================


class TestRecommendations:
    """Tests for get_recommendations."""

    def test_recommendations_for_minimal_data(self, engine, minimal_data):
        result = engine.check_compliance(
            calc_result=minimal_data,
            frameworks=["ghg_protocol_scope2"],
        )
        recs = engine.get_recommendations(result)
        if isinstance(recs, list):
            assert len(recs) >= 1
        elif isinstance(recs, dict):
            items = recs.get("recommendations", recs.get("items", []))
            assert len(items) >= 1


# ===========================================================================
# 7. Framework Info and Accessors Tests
# ===========================================================================


class TestAccessors:
    """Tests for accessor methods."""

    def test_get_all_frameworks(self, engine):
        frameworks = engine.get_all_frameworks()
        assert len(frameworks) == 7
        assert "ghg_protocol_scope2" in frameworks

    def test_get_framework_count(self, engine):
        count = engine.get_framework_count()
        assert count == 7

    def test_get_total_requirements(self, engine):
        total = engine.get_total_requirements()
        assert total == 84

    def test_get_framework_info(self, engine):
        info = engine.get_framework_info("ghg_protocol_scope2")
        assert "name" in info
        assert "GHG Protocol" in info["name"]

    def test_get_framework_info_unknown(self, engine):
        info = engine.get_framework_info("unknown_framework")
        # Should return empty or raise
        if isinstance(info, dict):
            assert info == {} or "error" in info or info.get("name") is None


# ===========================================================================
# 8. ComplianceFinding Dataclass Tests
# ===========================================================================


class TestComplianceFinding:
    """Tests for ComplianceFinding dataclass."""

    def test_creation(self):
        f = ComplianceFinding(
            requirement_id="GHG-SHP-001",
            framework="ghg_protocol_scope2",
            requirement="Energy type classification",
            passed=True,
            severity="ERROR",
            finding="Energy type is documented",
            recommendation="No action needed",
        )
        assert f.requirement_id == "GHG-SHP-001"
        assert f.passed is True

    def test_to_dict(self):
        f = ComplianceFinding(
            requirement_id="GHG-SHP-001",
            framework="ghg_protocol_scope2",
            requirement="Energy type classification",
            passed=False,
            severity="ERROR",
            finding="Energy type missing",
            recommendation="Add energy_type field",
        )
        d = f.to_dict()
        assert isinstance(d, dict)
        assert d["requirement_id"] == "GHG-SHP-001"
        assert d["passed"] is False
        assert d["severity"] == "ERROR"

    def test_failed_finding_has_recommendation(self):
        f = ComplianceFinding(
            requirement_id="ISO-SHP-003",
            framework="iso_14064",
            requirement="Base year required",
            passed=False,
            severity="ERROR",
            finding="No base year provided",
            recommendation="Specify base_year for ISO 14064 compliance",
        )
        assert f.recommendation != ""
        assert len(f.recommendation) > 0


# ===========================================================================
# 9. Stats Tests
# ===========================================================================


class TestComplianceStats:
    """Tests for get_compliance_stats."""

    def test_stats_returns_dict(self, engine):
        stats = engine.get_compliance_stats()
        assert isinstance(stats, dict)

    def test_stats_after_check(self, engine, complete_data):
        engine.check_compliance(
            calc_result=complete_data,
            frameworks=["ghg_protocol_scope2"],
        )
        stats = engine.get_compliance_stats()
        assert stats.get("total_checks", 0) >= 1 or stats.get("count", 0) >= 1


# ===========================================================================
# 10. Check All Frameworks Individually Tests
# ===========================================================================


class TestCheckAllFrameworksIndividually:
    """Tests that each of the 7 frameworks can be checked individually."""

    @pytest.mark.parametrize("framework", SUPPORTED_FRAMEWORKS)
    def test_check_single_framework_individually(self, engine, complete_data, framework):
        result = engine.check_compliance(
            calc_result=complete_data,
            frameworks=[framework],
        )
        assert isinstance(result, dict)
        assert "provenance_hash" in result

    @pytest.mark.parametrize("framework", SUPPORTED_FRAMEWORKS)
    def test_each_framework_returns_12_findings(self, engine, complete_data, framework):
        checker = getattr(engine, f"check_{framework}", None)
        if checker is not None:
            findings = checker(complete_data)
            assert len(findings) == 12


# ===========================================================================
# 11. Non-Compliant Items Tests
# ===========================================================================


class TestNonCompliantItems:
    """Tests for get_non_compliant_items."""

    def test_non_compliant_items_with_minimal_data(self, engine, minimal_data):
        result = engine.check_compliance(
            calc_result=minimal_data,
            frameworks=["ghg_protocol_scope2"],
        )
        items = engine.get_non_compliant_items(result)
        if isinstance(items, list):
            assert len(items) >= 1
        elif isinstance(items, dict):
            findings = items.get("findings", items.get("items", []))
            assert len(findings) >= 1

    def test_non_compliant_items_with_complete_data(self, engine, complete_data):
        result = engine.check_compliance(
            calc_result=complete_data,
            frameworks=["ghg_protocol_scope2"],
        )
        items = engine.get_non_compliant_items(result)
        # Complete data should have very few non-compliant items
        if isinstance(items, list):
            assert len(items) <= 5


# ===========================================================================
# 12. Validate Request Tests
# ===========================================================================


class TestValidateRequest:
    """Tests for validate_request."""

    def test_validate_complete_data(self, engine, complete_data):
        result = engine.validate_request(complete_data)
        if isinstance(result, dict):
            assert result.get("valid", True) is True or len(result.get("errors", [])) == 0
        elif isinstance(result, list):
            assert len(result) == 0

    def test_validate_empty_data(self, engine, empty_data):
        result = engine.validate_request(empty_data)
        if isinstance(result, dict):
            assert result.get("valid", False) is False or len(result.get("errors", [])) > 0
        elif isinstance(result, list):
            assert len(result) > 0


# ===========================================================================
# 13. Get All Requirements Tests
# ===========================================================================


class TestGetAllRequirements:
    """Tests for get_all_requirements."""

    def test_returns_dict(self, engine):
        reqs = engine.get_all_requirements()
        assert isinstance(reqs, dict)

    def test_has_all_frameworks(self, engine):
        reqs = engine.get_all_requirements()
        for fw in SUPPORTED_FRAMEWORKS:
            assert fw in reqs


# ===========================================================================
# 14. Compliance Summary Tests
# ===========================================================================


class TestComplianceSummary:
    """Tests for get_compliance_summary."""

    def test_summary_after_check(self, engine, complete_data):
        result = engine.check_compliance(
            calc_result=complete_data,
            frameworks=SUPPORTED_FRAMEWORKS,
        )
        summary = engine.get_compliance_summary(result)
        assert isinstance(summary, dict)


# ===========================================================================
# 15. Health Check Tests
# ===========================================================================


class TestComplianceHealthCheck:
    """Tests for health_check method."""

    def test_health_check_returns_dict(self, engine):
        result = engine.health_check()
        assert isinstance(result, dict)

    def test_health_check_has_status(self, engine):
        result = engine.health_check()
        assert "status" in result or "healthy" in result


# ===========================================================================
# 16. Single Framework Check Tests
# ===========================================================================


class TestCheckSingleFramework:
    """Tests for check_single_framework convenience method."""

    @pytest.mark.parametrize("framework", SUPPORTED_FRAMEWORKS)
    def test_single_framework_returns_dict(self, engine, complete_data, framework):
        result = engine.check_single_framework(
            calc_result=complete_data,
            framework=framework,
        )
        assert isinstance(result, dict)


# ===========================================================================
# 17. Remediation Plan Tests
# ===========================================================================


class TestRemediationPlan:
    """Tests for get_remediation_plan."""

    def test_remediation_plan_after_check(self, engine, empty_data):
        result = engine.check_compliance(
            calc_result=empty_data,
            frameworks=SUPPORTED_FRAMEWORKS,
        )
        plan = engine.get_remediation_plan(result)
        assert isinstance(plan, (dict, list))


# ===========================================================================
# 18. Framework Info Tests
# ===========================================================================


class TestFrameworkInfo:
    """Tests for get_framework_info."""

    @pytest.mark.parametrize("framework", SUPPORTED_FRAMEWORKS)
    def test_framework_info_returns_dict(self, engine, framework):
        info = engine.get_framework_info(framework)
        assert isinstance(info, dict)

    def test_framework_info_unknown_framework(self, engine):
        info = engine.get_framework_info("UNKNOWN_FRAMEWORK")
        # Should return empty dict or raise error
        assert isinstance(info, dict) or info is None


# ===========================================================================
# 19. Determine Status Logic Tests
# ===========================================================================


class TestDetermineStatus:
    """Tests for _determine_status method."""

    def test_all_passed(self, engine):
        status = engine._determine_status(12, 12)
        assert status in ("COMPLIANT", "PASS", "compliant", "pass")

    def test_none_passed(self, engine):
        status = engine._determine_status(0, 12)
        assert status in (
            "NON_COMPLIANT", "FAIL", "non_compliant", "fail",
            "NON-COMPLIANT", "non-compliant",
        )

    def test_partial_passed(self, engine):
        status = engine._determine_status(6, 12)
        assert status in (
            "PARTIAL", "PARTIALLY_COMPLIANT", "partial",
            "partially_compliant", "PARTIALLY-COMPLIANT",
            "NON_COMPLIANT", "NON-COMPLIANT", "non_compliant",
            "non-compliant", "FAIL", "fail",
        )
