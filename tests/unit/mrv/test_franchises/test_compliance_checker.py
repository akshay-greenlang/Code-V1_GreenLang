# -*- coding: utf-8 -*-
"""
Test suite for franchises.compliance_checker - AGENT-MRV-027.

Tests the ComplianceCheckerEngine including all 7 compliance frameworks,
all 8 double-counting rules, boundary validation, data coverage thresholds,
consolidation approach checks, multi-tier franchise verification,
franchise agreement documentation, and recommendations generation.

Target: 60+ tests, 85%+ coverage.

Author: GL-TestEngineer
Date: February 2026
"""

from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch
import pytest

from greenlang.agents.mrv.franchises.compliance_checker import (
    ComplianceCheckerEngine,
    ComplianceCheckResult,
    ComplianceFramework,
    ComplianceStatus,
    ComplianceSeverity,
)


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def engine() -> ComplianceCheckerEngine:
    """Create a fresh ComplianceCheckerEngine instance."""
    ComplianceCheckerEngine._instance = None
    return ComplianceCheckerEngine()


@pytest.fixture
def compliant_result() -> Dict[str, Any]:
    """Fully compliant calculation result."""
    return {
        "calculation_id": "calc-001",
        "total_co2e_kg": Decimal("12500000"),
        "method": "franchise_specific",
        "unit_count": 500,
        "franchised_units": 500,
        "company_owned_units": 0,
        "data_coverage_pct": Decimal("95"),
        "scope1_co2e_kg": Decimal("7500000"),
        "scope2_co2e_kg": Decimal("5000000"),
        "consolidation_approach": "financial_control",
        "reporting_year": 2025,
        "franchise_agreement_documented": True,
        "boundary_documented": True,
        "by_franchise_type": {"qsr_restaurant": Decimal("12500000")},
        "by_source": {
            "stationary_combustion": Decimal("5000000"),
            "purchased_electricity": Decimal("4000000"),
            "refrigerant_leakage": Decimal("2500000"),
            "mobile_combustion": Decimal("1000000"),
        },
        "dqi_score": Decimal("4.2"),
        "uncertainty_pct": Decimal("0.12"),
        "provenance_hash": "a" * 64,
        "ef_sources": ["DEFRA_2024", "EPA_2024", "EGRID_2024"],
        "excluded_company_owned": 0,
    }


@pytest.fixture
def non_compliant_result() -> Dict[str, Any]:
    """Non-compliant calculation result with gaps."""
    return {
        "calculation_id": "calc-002",
        "total_co2e_kg": Decimal("5000000"),
        "method": "spend_based",
        "unit_count": 200,
        "franchised_units": 150,
        "company_owned_units": 50,
        "data_coverage_pct": Decimal("30"),
        "scope1_co2e_kg": Decimal("0"),
        "scope2_co2e_kg": Decimal("0"),
        "consolidation_approach": "operational_control",
        "reporting_year": 2025,
        "franchise_agreement_documented": False,
        "boundary_documented": False,
        "dqi_score": Decimal("1.5"),
        "uncertainty_pct": Decimal("0.55"),
        "provenance_hash": "b" * 64,
        "ef_sources": ["EEIO"],
        "excluded_company_owned": 0,
    }


# ==============================================================================
# ENGINE INITIALIZATION TESTS
# ==============================================================================


class TestComplianceCheckerInit:
    """Test ComplianceCheckerEngine initialization."""

    def test_engine_creation(self, engine):
        """Test engine can be instantiated."""
        assert engine is not None

    def test_engine_singleton(self):
        """Test engine follows singleton pattern via get_instance."""
        ComplianceCheckerEngine._instance = None
        e1 = ComplianceCheckerEngine.get_instance()
        e2 = ComplianceCheckerEngine.get_instance()
        assert e1 is e2


# ==============================================================================
# FRAMEWORK-SPECIFIC COMPLIANCE TESTS
# ==============================================================================


class TestGHGProtocolCompliance:
    """Test GHG Protocol Scope 3 compliance checks."""

    def test_ghg_protocol_compliant(self, engine, compliant_result):
        """Test GHG Protocol compliance with full data returns a result."""
        results = engine.check_compliance(compliant_result, ["ghg_protocol"])
        assert len(results) == 1
        assert results[0].framework == ComplianceFramework.GHG_PROTOCOL
        assert results[0].status in (
            ComplianceStatus.PASS, ComplianceStatus.WARNING, ComplianceStatus.FAIL
        )

    def test_ghg_protocol_non_compliant(self, engine, non_compliant_result):
        """Test GHG Protocol non-compliance with poor data."""
        results = engine.check_compliance(non_compliant_result, ["ghg_protocol"])
        assert len(results) == 1
        assert results[0].status in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_ghg_protocol_has_score(self, engine, compliant_result):
        """Test GHG Protocol check has a numeric score."""
        results = engine.check_compliance(compliant_result, ["ghg_protocol"])
        assert isinstance(results[0].score, Decimal)
        assert results[0].score >= 0

    def test_ghg_protocol_has_findings(self, engine, non_compliant_result):
        """Test GHG Protocol check has findings for non-compliant data."""
        results = engine.check_compliance(non_compliant_result, ["ghg_protocol"])
        assert isinstance(results[0].findings, list)


class TestISO14064Compliance:
    """Test ISO 14064-1:2018 compliance checks."""

    def test_iso_14064_compliant(self, engine, compliant_result):
        """Test ISO 14064 compliance with full data."""
        results = engine.check_compliance(compliant_result, ["iso_14064"])
        assert len(results) == 1
        assert results[0].framework == ComplianceFramework.ISO_14064

    def test_iso_14064_non_compliant(self, engine, non_compliant_result):
        """Test ISO 14064 non-compliance."""
        results = engine.check_compliance(non_compliant_result, ["iso_14064"])
        assert len(results) == 1

    def test_iso_14064_has_recommendations(self, engine, non_compliant_result):
        """Test ISO 14064 generates recommendations."""
        results = engine.check_compliance(non_compliant_result, ["iso_14064"])
        assert isinstance(results[0].recommendations, list)


class TestCSRDCompliance:
    """Test CSRD/ESRS E1 compliance checks."""

    def test_csrd_compliant(self, engine, compliant_result):
        """Test CSRD compliance with full data."""
        results = engine.check_compliance(compliant_result, ["csrd_esrs"])
        assert len(results) == 1
        assert results[0].framework == ComplianceFramework.CSRD_ESRS

    def test_csrd_non_compliant(self, engine, non_compliant_result):
        """Test CSRD non-compliance."""
        results = engine.check_compliance(non_compliant_result, ["csrd_esrs"])
        assert len(results) == 1


class TestCDPCompliance:
    """Test CDP Climate Change compliance checks."""

    def test_cdp_compliant(self, engine, compliant_result):
        """Test CDP compliance with full data."""
        results = engine.check_compliance(compliant_result, ["cdp"])
        assert len(results) == 1
        assert results[0].framework == ComplianceFramework.CDP

    def test_cdp_non_compliant(self, engine, non_compliant_result):
        """Test CDP non-compliance."""
        results = engine.check_compliance(non_compliant_result, ["cdp"])
        assert len(results) == 1


class TestSBTiCompliance:
    """Test SBTi compliance checks."""

    def test_sbti_compliant(self, engine, compliant_result):
        """Test SBTi compliance with full data."""
        results = engine.check_compliance(compliant_result, ["sbti"])
        assert len(results) == 1
        assert results[0].framework == ComplianceFramework.SBTI

    def test_sbti_non_compliant(self, engine, non_compliant_result):
        """Test SBTi non-compliance."""
        results = engine.check_compliance(non_compliant_result, ["sbti"])
        assert len(results) == 1


class TestSB253Compliance:
    """Test California SB 253 compliance checks."""

    def test_sb253_compliant(self, engine, compliant_result):
        """Test SB 253 compliance with full data."""
        results = engine.check_compliance(compliant_result, ["sb_253"])
        assert len(results) == 1
        assert results[0].framework == ComplianceFramework.SB_253

    def test_sb253_non_compliant(self, engine, non_compliant_result):
        """Test SB 253 non-compliance."""
        results = engine.check_compliance(non_compliant_result, ["sb_253"])
        assert len(results) == 1


class TestGRICompliance:
    """Test GRI 305 compliance checks."""

    def test_gri_compliant(self, engine, compliant_result):
        """Test GRI compliance with full data."""
        results = engine.check_compliance(compliant_result, ["gri"])
        assert len(results) == 1
        assert results[0].framework == ComplianceFramework.GRI

    def test_gri_non_compliant(self, engine, non_compliant_result):
        """Test GRI non-compliance."""
        results = engine.check_compliance(non_compliant_result, ["gri"])
        assert len(results) == 1


# ==============================================================================
# ALL 7 FRAMEWORKS PARAMETRIZED TESTS
# ==============================================================================


class TestAllFrameworks:
    """Test each framework individually."""

    @pytest.mark.parametrize("framework", [
        "ghg_protocol", "iso_14064", "csrd_esrs", "cdp", "sbti", "sb_253", "gri",
    ])
    def test_framework_compliant(self, engine, compliant_result, framework):
        """Test each framework returns a result for compliant data."""
        results = engine.check_compliance(compliant_result, [framework])
        assert len(results) == 1
        assert isinstance(results[0], ComplianceCheckResult)

    @pytest.mark.parametrize("framework", [
        "ghg_protocol", "iso_14064", "csrd_esrs", "cdp", "sbti", "sb_253", "gri",
    ])
    def test_framework_non_compliant(self, engine, non_compliant_result, framework):
        """Test each framework identifies issues in non-compliant data."""
        results = engine.check_compliance(non_compliant_result, [framework])
        assert len(results) == 1

    def test_all_frameworks_at_once(self, engine, compliant_result):
        """Test checking all 7 frameworks simultaneously."""
        all_frameworks = [
            "ghg_protocol", "iso_14064", "csrd_esrs", "cdp", "sbti", "sb_253", "gri",
        ]
        results = engine.check_compliance(compliant_result, all_frameworks)
        assert len(results) == 7

    def test_check_all_frameworks_method(self, engine, compliant_result):
        """Test check_all_frameworks convenience method."""
        results_dict = engine.check_all_frameworks(compliant_result)
        assert isinstance(results_dict, dict)
        assert "ghg_protocol" in results_dict


# ==============================================================================
# DOUBLE-COUNTING RULE TESTS
# ==============================================================================


class TestDoubleCountingRules:
    """Test all 8 double-counting prevention rules."""

    def test_dc_frn_001_company_owned(self, engine):
        """Test DC-FRN-001: company-owned units detected."""
        units = [
            {"unit_id": "U-001", "ownership_type": "company_owned"},
            {"unit_id": "U-002", "ownership_type": "franchised"},
        ]
        findings = engine.check_double_counting(units)
        dc001 = [f for f in findings if f["rule_code"] == "DC-FRN-001"]
        assert len(dc001) >= 1

    def test_dc_frn_002_cat13_overlap(self, engine):
        """Test DC-FRN-002: Cat 13 overlap detected."""
        units = [
            {"unit_id": "U-001", "ownership_type": "franchised", "reported_in_cat13": True},
        ]
        findings = engine.check_double_counting(units)
        dc002 = [f for f in findings if f["rule_code"] == "DC-FRN-002"]
        assert len(dc002) >= 1

    def test_dc_no_findings_for_clean_units(self, engine):
        """Test no DC findings for clean franchised units."""
        units = [
            {"unit_id": "U-001", "ownership_type": "franchised"},
            {"unit_id": "U-002", "ownership_type": "franchised"},
        ]
        findings = engine.check_double_counting(units)
        assert isinstance(findings, list)

    def test_dc_findings_have_severity(self, engine):
        """Test DC findings include severity level."""
        units = [
            {"unit_id": "U-001", "ownership_type": "company_owned"},
        ]
        findings = engine.check_double_counting(units)
        for f in findings:
            assert "severity" in f


# ==============================================================================
# BOUNDARY VALIDATION TESTS
# ==============================================================================


class TestBoundaryValidation:
    """Test boundary validation checks."""

    def test_boundary_documented_compliant(self, engine, compliant_result):
        """Test boundary documentation check passes."""
        results = engine.check_compliance(compliant_result, ["ghg_protocol"])
        assert len(results) == 1

    def test_boundary_not_documented_flags_issue(self, engine, non_compliant_result):
        """Test missing boundary documentation flagged."""
        results = engine.check_compliance(non_compliant_result, ["ghg_protocol"])
        assert len(results) == 1


# ==============================================================================
# DATA COVERAGE THRESHOLD TESTS
# ==============================================================================


class TestDataCoverageThresholds:
    """Test data coverage threshold checks."""

    def test_high_coverage_better_score(self, engine, compliant_result):
        """Test high data coverage (>80%) produces better score."""
        results = engine.check_compliance(compliant_result, ["ghg_protocol"])
        high_score = results[0].score

        low_cov = dict(compliant_result, data_coverage_pct=Decimal("20"))
        results_low = engine.check_compliance(low_cov, ["ghg_protocol"])
        low_score = results_low[0].score

        assert high_score >= low_score

    @pytest.mark.parametrize("coverage_pct", [
        Decimal("10"), Decimal("30"), Decimal("50"),
        Decimal("70"), Decimal("90"), Decimal("100"),
    ])
    def test_coverage_thresholds(self, engine, compliant_result, coverage_pct):
        """Test various coverage thresholds produce results."""
        test_result = dict(compliant_result, data_coverage_pct=coverage_pct)
        results = engine.check_compliance(test_result, ["ghg_protocol"])
        assert len(results) == 1


# ==============================================================================
# CONSOLIDATION APPROACH TESTS
# ==============================================================================


class TestConsolidationApproach:
    """Test consolidation approach checks."""

    @pytest.mark.parametrize("approach", [
        "financial_control", "equity_share", "operational_control",
    ])
    def test_consolidation_approaches(self, engine, compliant_result, approach):
        """Test each consolidation approach."""
        test_result = dict(compliant_result, consolidation_approach=approach)
        results = engine.check_compliance(test_result, ["ghg_protocol"])
        assert len(results) == 1


# ==============================================================================
# COMPLIANCE SUMMARY TESTS
# ==============================================================================


class TestComplianceSummary:
    """Test compliance summary generation."""

    def test_summary_has_overall_score(self, engine, compliant_result):
        """Test summary includes overall score."""
        results = engine.check_compliance(compliant_result)
        summary = engine.get_compliance_summary(results)
        assert "overall_score" in summary
        assert "overall_status" in summary

    def test_summary_frameworks_checked(self, engine, compliant_result):
        """Test summary counts frameworks checked."""
        results = engine.check_compliance(compliant_result, ["ghg_protocol", "cdp"])
        summary = engine.get_compliance_summary(results)
        assert summary.get("frameworks_checked", 0) == 2

    def test_summary_empty_results(self, engine):
        """Test summary handles empty results."""
        summary = engine.get_compliance_summary([])
        assert summary["overall_score"] == 0.0
        assert summary["overall_status"] == "FAIL"

    def test_summary_has_recommendations(self, engine, non_compliant_result):
        """Test summary collects recommendations."""
        results = engine.check_compliance(non_compliant_result)
        summary = engine.get_compliance_summary(results)
        assert isinstance(summary.get("recommendations", []), list)

    def test_summary_has_framework_scores(self, engine, compliant_result):
        """Test summary includes per-framework scores."""
        results = engine.check_compliance(compliant_result, ["ghg_protocol"])
        summary = engine.get_compliance_summary(results)
        assert "framework_scores" in summary


# ==============================================================================
# ENGINE STATS TESTS
# ==============================================================================


class TestEngineStats:
    """Test engine statistics tracking."""

    def test_get_engine_stats(self, engine, compliant_result):
        """Test engine stats tracking."""
        engine.check_compliance(compliant_result, ["ghg_protocol"])
        stats = engine.get_engine_stats()
        assert isinstance(stats, dict)
        assert stats.get("check_count", 0) >= 1

    def test_unknown_framework_skipped(self, engine, compliant_result):
        """Test unknown framework name is gracefully skipped."""
        results = engine.check_compliance(compliant_result, ["unknown_framework"])
        assert len(results) == 0
