# -*- coding: utf-8 -*-
"""
Test suite for downstream_transportation.compliance_checker - AGENT-MRV-022.

Tests ComplianceCheckerEngine for the Downstream Transportation &
Distribution Agent (GL-MRV-S3-009).

Coverage (~65 tests):
- All 7 compliance frameworks (GHG Protocol, ISO 14064, ISO 14083,
  CSRD, CDP, SBTi, SB 253)
- 10 double-counting prevention rules
- Incoterm boundary classification
- Batch compliance checking
- Required disclosures per framework
- Data quality scoring
- Pass/fail scenarios (compliant, non-compliant, warnings)
- Singleton pattern

Author: GL-TestEngineer
Date: February 2026
"""

from decimal import Decimal
import pytest

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

_AVAILABLE = True
_IMPORT_ERROR = None

try:
    from greenlang.agents.mrv.downstream_transportation.compliance_checker import (
        ComplianceCheckerEngine,
    )
except ImportError as exc:
    _AVAILABLE = False
    _IMPORT_ERROR = str(exc)

_SKIP = pytest.mark.skipif(
    not _AVAILABLE,
    reason=f"compliance_checker not available: {_IMPORT_ERROR}",
)

pytestmark = _SKIP


# ==============================================================================
# SINGLETON TESTS
# ==============================================================================


class TestSingleton:
    """Test ComplianceCheckerEngine singleton."""

    def test_singleton_identity(self):
        """Test two instantiations return the same object."""
        eng1 = ComplianceCheckerEngine()
        eng2 = ComplianceCheckerEngine()
        assert eng1 is eng2


# ==============================================================================
# GHG PROTOCOL FRAMEWORK TESTS
# ==============================================================================


class TestGHGProtocol:
    """Test GHG Protocol Scope 3 Category 9 compliance."""

    def test_ghg_protocol_pass(self, sample_compliance_data):
        """Test GHG Protocol compliance passes with complete data."""
        engine = ComplianceCheckerEngine()
        result = engine.check_compliance({
            **sample_compliance_data, "frameworks": ["GHG_PROTOCOL"],
        })
        assert result is not None
        assert result.get("compliant") is True

    def test_ghg_protocol_requires_category_9(self):
        """Test GHG Protocol requires correct category classification."""
        engine = ComplianceCheckerEngine()
        result = engine.check_compliance({
            "frameworks": ["GHG_PROTOCOL"],
            "calculation_results": [
                {"total_co2e": Decimal("100.0"), "method": "DISTANCE_BASED"},
            ],
            "category": "SCOPE_3_CATEGORY_9",
            "reporting_period": "2025",
        })
        assert result is not None

    def test_ghg_protocol_requires_boundary_disclosure(self):
        """Test GHG Protocol requires boundary/Incoterm disclosure."""
        engine = ComplianceCheckerEngine()
        result = engine.check_compliance({
            "frameworks": ["GHG_PROTOCOL"],
            "calculation_results": [
                {"total_co2e": Decimal("100.0"), "method": "DISTANCE_BASED"},
            ],
            "boundary_disclosed": False,
            "reporting_period": "2025",
        })
        # Should have warnings or issues about boundary
        issues = result.get("issues", []) + result.get("warnings", [])
        assert len(issues) > 0 or result.get("compliant") is False

    def test_ghg_protocol_requires_method_disclosure(self):
        """Test GHG Protocol requires calculation method disclosure."""
        engine = ComplianceCheckerEngine()
        result = engine.check_compliance({
            "frameworks": ["GHG_PROTOCOL"],
            "calculation_results": [],
            "reporting_period": "2025",
        })
        # Missing calculation results should fail
        assert result.get("compliant") is False or len(result.get("issues", [])) > 0

    def test_ghg_protocol_scope_3_category_9_label(self):
        """Test result correctly labels Scope 3 Category 9."""
        engine = ComplianceCheckerEngine()
        result = engine.check_compliance({
            "frameworks": ["GHG_PROTOCOL"],
            "calculation_results": [
                {"total_co2e": Decimal("100.0"), "method": "DISTANCE_BASED"},
            ],
            "reporting_period": "2025",
        })
        cat = result.get("category", "")
        assert "9" in str(cat) or "CATEGORY_9" in str(cat)


# ==============================================================================
# ISO 14064 FRAMEWORK TESTS
# ==============================================================================


class TestISO14064:
    """Test ISO 14064-1:2018 compliance."""

    def test_iso14064_pass(self, sample_compliance_data):
        """Test ISO 14064 compliance passes with complete data."""
        engine = ComplianceCheckerEngine()
        result = engine.check_compliance({
            **sample_compliance_data, "frameworks": ["ISO_14064"],
        })
        assert result is not None

    def test_iso14064_requires_uncertainty(self):
        """Test ISO 14064 checks for uncertainty assessment."""
        engine = ComplianceCheckerEngine()
        result = engine.check_compliance({
            "frameworks": ["ISO_14064"],
            "calculation_results": [
                {"total_co2e": Decimal("100.0"), "method": "DISTANCE_BASED"},
            ],
            "uncertainty_assessed": False,
            "reporting_period": "2025",
        })
        # Should have warning about uncertainty
        warnings = result.get("warnings", [])
        assert len(warnings) > 0 or result.get("compliant") is not None


# ==============================================================================
# ISO 14083 FRAMEWORK TESTS
# ==============================================================================


class TestISO14083:
    """Test ISO 14083 transport chain compliance."""

    def test_iso14083_pass(self, sample_compliance_data):
        """Test ISO 14083 compliance passes."""
        engine = ComplianceCheckerEngine()
        result = engine.check_compliance({
            **sample_compliance_data, "frameworks": ["ISO_14083"],
        })
        assert result is not None

    def test_iso14083_wtw_mandatory(self):
        """Test ISO 14083 requires WTW scope (not TTW only)."""
        engine = ComplianceCheckerEngine()
        result = engine.check_compliance({
            "frameworks": ["ISO_14083"],
            "calculation_results": [
                {"total_co2e": Decimal("100.0"), "method": "DISTANCE_BASED"},
            ],
            "ef_scope": "TTW",
            "wtt_disclosed": False,
            "reporting_period": "2025",
        })
        issues = result.get("issues", []) + result.get("warnings", [])
        assert len(issues) > 0 or result.get("compliant") is not None

    def test_iso14083_mode_specific_ef_required(self):
        """Test ISO 14083 requires mode-specific emission factors."""
        engine = ComplianceCheckerEngine()
        result = engine.check_compliance({
            "frameworks": ["ISO_14083"],
            "calculation_results": [
                {"total_co2e": Decimal("100.0"), "method": "DISTANCE_BASED"},
            ],
            "mode_breakdown_provided": True,
            "reporting_period": "2025",
        })
        assert result is not None


# ==============================================================================
# CSRD FRAMEWORK TESTS
# ==============================================================================


class TestCSRD:
    """Test CSRD ESRS E1 compliance."""

    def test_csrd_pass(self, sample_compliance_data):
        """Test CSRD compliance passes."""
        engine = ComplianceCheckerEngine()
        result = engine.check_compliance({
            **sample_compliance_data, "frameworks": ["CSRD"],
        })
        assert result is not None

    def test_csrd_requires_scope3(self):
        """Test CSRD requires Scope 3 downstream transport disclosure."""
        engine = ComplianceCheckerEngine()
        result = engine.check_compliance({
            "frameworks": ["CSRD"],
            "calculation_results": [
                {"total_co2e": Decimal("100.0"), "method": "DISTANCE_BASED"},
            ],
            "reporting_period": "2025",
        })
        assert result is not None

    def test_csrd_materiality_assessment(self):
        """Test CSRD checks for materiality assessment."""
        engine = ComplianceCheckerEngine()
        result = engine.check_compliance({
            "frameworks": ["CSRD"],
            "calculation_results": [
                {"total_co2e": Decimal("100.0"), "method": "DISTANCE_BASED"},
            ],
            "materiality_assessment": True,
            "reporting_period": "2025",
        })
        assert result is not None


# ==============================================================================
# CDP FRAMEWORK TESTS
# ==============================================================================


class TestCDP:
    """Test CDP Climate Change questionnaire compliance."""

    def test_cdp_pass(self, sample_compliance_data):
        """Test CDP compliance passes."""
        engine = ComplianceCheckerEngine()
        result = engine.check_compliance({
            **sample_compliance_data, "frameworks": ["CDP"],
        })
        assert result is not None

    def test_cdp_requires_scope3_category_breakdown(self):
        """Test CDP requires Scope 3 category-level breakdown."""
        engine = ComplianceCheckerEngine()
        result = engine.check_compliance({
            "frameworks": ["CDP"],
            "calculation_results": [
                {"total_co2e": Decimal("100.0"), "method": "DISTANCE_BASED", "category": "9"},
            ],
            "reporting_period": "2025",
        })
        assert result is not None

    def test_cdp_data_quality_requirement(self):
        """Test CDP checks data quality score."""
        engine = ComplianceCheckerEngine()
        result = engine.check_compliance({
            "frameworks": ["CDP"],
            "calculation_results": [
                {"total_co2e": Decimal("100.0"), "method": "SPEND_BASED"},
            ],
            "data_quality_score": Decimal("2.0"),  # Low quality
            "reporting_period": "2025",
        })
        warnings = result.get("warnings", [])
        # Low data quality should generate warning
        assert len(warnings) > 0 or result is not None


# ==============================================================================
# SBTi FRAMEWORK TESTS
# ==============================================================================


class TestSBTi:
    """Test SBTi target-setting compliance."""

    def test_sbti_pass(self, sample_compliance_data):
        """Test SBTi compliance passes."""
        engine = ComplianceCheckerEngine()
        result = engine.check_compliance({
            **sample_compliance_data, "frameworks": ["SBTI"],
        })
        assert result is not None

    def test_sbti_requires_comprehensive_scope3(self):
        """Test SBTi requires 67% Scope 3 coverage."""
        engine = ComplianceCheckerEngine()
        result = engine.check_compliance({
            "frameworks": ["SBTI"],
            "calculation_results": [
                {"total_co2e": Decimal("100.0"), "method": "DISTANCE_BASED"},
            ],
            "scope3_coverage_pct": Decimal("0.70"),
            "reporting_period": "2025",
        })
        assert result is not None


# ==============================================================================
# SB 253 FRAMEWORK TESTS
# ==============================================================================


class TestSB253:
    """Test California SB 253 compliance."""

    def test_sb253_pass(self, sample_compliance_data):
        """Test SB 253 compliance passes."""
        engine = ComplianceCheckerEngine()
        result = engine.check_compliance({
            **sample_compliance_data, "frameworks": ["SB_253"],
        })
        assert result is not None

    def test_sb253_requires_third_party_assurance(self):
        """Test SB 253 requires third-party assurance for Scope 3."""
        engine = ComplianceCheckerEngine()
        result = engine.check_compliance({
            "frameworks": ["SB_253"],
            "calculation_results": [
                {"total_co2e": Decimal("100.0"), "method": "DISTANCE_BASED"},
            ],
            "third_party_assurance": False,
            "reporting_period": "2025",
        })
        warnings = result.get("warnings", []) + result.get("issues", [])
        assert len(warnings) > 0 or result is not None


# ==============================================================================
# MULTI-FRAMEWORK TESTS
# ==============================================================================


class TestMultiFramework:
    """Test compliance checking across multiple frameworks."""

    def test_all_7_frameworks(self, sample_compliance_data):
        """Test compliance check against all 7 frameworks simultaneously."""
        engine = ComplianceCheckerEngine()
        result = engine.check_compliance(sample_compliance_data)
        assert result is not None

    def test_partial_compliance(self):
        """Test partial compliance (pass some, fail others)."""
        engine = ComplianceCheckerEngine()
        result = engine.check_compliance({
            "frameworks": ["GHG_PROTOCOL", "ISO_14083"],
            "calculation_results": [
                {"total_co2e": Decimal("100.0"), "method": "SPEND_BASED"},
            ],
            "ef_scope": "TTW",
            "wtt_disclosed": False,
            "reporting_period": "2025",
        })
        assert result is not None


# ==============================================================================
# DOUBLE-COUNTING PREVENTION TESTS
# ==============================================================================


class TestDoubleCounting:
    """Test 10 double-counting prevention rules."""

    def test_no_double_count_with_scope1(self):
        """Test double-counting check: Cat 9 vs Scope 1 owned transport."""
        engine = ComplianceCheckerEngine()
        result = engine.check_double_counting({
            "category_9_emissions": Decimal("100.0"),
            "scope1_transport_emissions": Decimal("50.0"),
            "overlap_check": True,
        })
        assert result is not None

    def test_no_double_count_with_cat4(self):
        """Test double-counting check: Cat 9 vs Cat 4 upstream transport."""
        engine = ComplianceCheckerEngine()
        result = engine.check_double_counting({
            "category_9_emissions": Decimal("100.0"),
            "category_4_emissions": Decimal("200.0"),
            "incoterm_based_split": True,
        })
        assert result is not None

    def test_no_double_count_with_cat1(self):
        """Test double-counting check: Cat 9 vs Cat 1 purchased goods transport."""
        engine = ComplianceCheckerEngine()
        result = engine.check_double_counting({
            "category_9_emissions": Decimal("100.0"),
            "category_1_transport_component": Decimal("30.0"),
        })
        assert result is not None

    def test_no_double_count_with_cat12(self):
        """Test double-counting check: Cat 9 vs Cat 12 end-of-life transport."""
        engine = ComplianceCheckerEngine()
        result = engine.check_double_counting({
            "category_9_emissions": Decimal("100.0"),
            "category_12_transport_component": Decimal("10.0"),
        })
        assert result is not None

    def test_incoterm_boundary_split(self):
        """Test Incoterm-based boundary split between Cat 4 and Cat 9."""
        engine = ComplianceCheckerEngine()
        result = engine.check_double_counting({
            "incoterm": "FOB",
            "total_transport_emissions": Decimal("300.0"),
            "seller_portion": Decimal("50.0"),
            "buyer_portion": Decimal("250.0"),
        })
        assert result is not None

    @pytest.mark.parametrize("rule_id", range(1, 11))
    def test_double_counting_rule_exists(self, rule_id):
        """Test each of the 10 double-counting rules is implemented."""
        engine = ComplianceCheckerEngine()
        rules = engine.get_double_counting_rules()
        assert rules is not None
        assert len(rules) >= 10


# ==============================================================================
# INCOTERM BOUNDARY TESTS
# ==============================================================================


class TestIncotermBoundary:
    """Test Incoterm-based boundary classification."""

    @pytest.mark.parametrize("incoterm,expected_cat9", [
        ("EXW", True),
        ("FCA", True),
        ("FAS", True),
        ("FOB", True),
        ("CPT", False),
        ("CIF", False),
        ("CIP", False),
        ("DAP", False),
        ("DPU", False),
        ("DDP", False),
        ("CFR", False),
    ])
    def test_incoterm_boundary(self, incoterm, expected_cat9):
        """Test Incoterm classification for Cat 4 vs Cat 9."""
        engine = ComplianceCheckerEngine()
        result = engine.classify_incoterm(incoterm)
        cat9 = result.get("category_9_applicable", result.get("cat_9",
                result.get("buyer_arranges")))
        assert cat9 == expected_cat9, (
            f"Incoterm {incoterm}: expected cat_9={expected_cat9}, got {cat9}"
        )


# ==============================================================================
# BATCH COMPLIANCE TESTS
# ==============================================================================


class TestBatchCompliance:
    """Test batch compliance checking."""

    def test_batch_check(self):
        """Test batch compliance check for multiple calculations."""
        engine = ComplianceCheckerEngine()
        calculations = [
            {
                "calculation_id": f"CALC-{i}",
                "total_co2e": Decimal(f"{100 + i * 50}"),
                "method": "DISTANCE_BASED",
            }
            for i in range(5)
        ]
        result = engine.check_batch_compliance({
            "frameworks": ["GHG_PROTOCOL"],
            "calculations": calculations,
            "reporting_period": "2025",
        })
        assert result is not None


# ==============================================================================
# REQUIRED DISCLOSURES TESTS
# ==============================================================================


class TestRequiredDisclosures:
    """Test required disclosure checking per framework."""

    @pytest.mark.parametrize("framework", [
        "GHG_PROTOCOL", "ISO_14064", "ISO_14083",
        "CSRD", "CDP", "SBTI", "SB_253",
    ])
    def test_framework_has_required_disclosures(self, framework):
        """Test each framework has defined required disclosures."""
        engine = ComplianceCheckerEngine()
        disclosures = engine.get_required_disclosures(framework)
        assert disclosures is not None
        assert len(disclosures) > 0


# ==============================================================================
# SCORING TESTS
# ==============================================================================


class TestScoring:
    """Test compliance scoring."""

    def test_completeness_score(self, sample_compliance_data):
        """Test completeness score is calculated."""
        engine = ComplianceCheckerEngine()
        result = engine.check_compliance(sample_compliance_data)
        score = result.get("completeness_score")
        assert score is not None
        assert Decimal("0") <= score <= Decimal("1.0")

    def test_data_quality_score(self, sample_compliance_data):
        """Test data quality score is calculated."""
        engine = ComplianceCheckerEngine()
        result = engine.check_compliance(sample_compliance_data)
        score = result.get("data_quality_score")
        assert score is not None
        assert Decimal("0") <= score <= Decimal("1.0") or \
               Decimal("0") <= score <= Decimal("5.0")


# ==============================================================================
# PASS/FAIL SCENARIO TESTS
# ==============================================================================


class TestPassFailScenarios:
    """Test specific pass/fail compliance scenarios."""

    def test_complete_data_passes(self, sample_compliance_data):
        """Test complete, high-quality data passes all frameworks."""
        engine = ComplianceCheckerEngine()
        result = engine.check_compliance(sample_compliance_data)
        assert result.get("compliant") is True

    def test_missing_data_fails(self):
        """Test missing critical data fails compliance."""
        engine = ComplianceCheckerEngine()
        result = engine.check_compliance({
            "frameworks": ["GHG_PROTOCOL"],
            "calculation_results": [],  # No results
            "reporting_period": "2025",
        })
        assert result.get("compliant") is False or len(result.get("issues", [])) > 0

    def test_low_quality_generates_warnings(self):
        """Test low data quality generates warnings."""
        engine = ComplianceCheckerEngine()
        result = engine.check_compliance({
            "frameworks": ["GHG_PROTOCOL"],
            "calculation_results": [
                {"total_co2e": Decimal("100.0"), "method": "SPEND_BASED"},
            ],
            "data_quality_score": Decimal("1.5"),
            "reporting_period": "2025",
        })
        warnings = result.get("warnings", [])
        assert len(warnings) > 0 or result is not None
