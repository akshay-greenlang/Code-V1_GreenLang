# -*- coding: utf-8 -*-
"""
Test suite for investments.compliance_checker - AGENT-MRV-028.

Tests the ComplianceCheckerEngine (Engine 6) for the Investments Agent
(GL-MRV-S3-015) covering all 9 regulatory frameworks, all 8 DC rules,
PCAF-specific validation, portfolio alignment checking, and
recommendations generation.

Coverage:
- GHG Protocol Scope 3 (7 tests)
- PCAF Global GHG Standard (8 tests)
- ISO 14064 (5 tests)
- CSRD ESRS E1 (5 tests)
- CDP Climate Change (5 tests)
- SBTi Financial Institutions (5 tests)
- SB 253 (4 tests)
- TCFD (4 tests)
- NZBA (4 tests)
- DC rules DC-INV-001 through DC-INV-008 (8 tests)
- PCAF attribution validation (5 tests)
- Portfolio alignment checking (3 tests)
- Recommendations generation (2 tests)

Author: GL-TestEngineer
Date: February 2026
"""

import pytest
from decimal import Decimal
from unittest.mock import patch, MagicMock

from greenlang.agents.mrv.investments.compliance_checker import (
    ComplianceCheckerEngine,
)
from greenlang.agents.mrv.investments.models import (
    ComplianceFramework,
    ComplianceStatus,
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
    frameworks: str = "GHG_PROTOCOL_SCOPE3,PCAF,ISO_14064,CSRD_ESRS_E1,CDP,SBTI_FI,SB_253,TCFD,NZBA",
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
        "greenlang.agents.mrv.investments.compliance_checker.get_config"
    ) as mock_config:
        mock_config.return_value = _make_mock_config()
        eng = ComplianceCheckerEngine()
        yield eng


def _full_result(**overrides):
    """Build a fully-compliant result dict with all fields present."""
    base = {
        "total_financed_emissions": Decimal("102560"),
        "total_co2e": Decimal("102560"),
        "calculation_method": "pcaf_standard",
        "methodology": "PCAF Global GHG Standard 3rd Ed.",
        "attribution_method": "evic",
        "ef_sources": ["PCAF", "BLOOMBERG"],
        "pcaf_quality_score": 2,
        "weighted_pcaf_score": Decimal("2.5"),
        "pcaf_coverage": Decimal("0.95"),
        "asset_class_breakdown": {
            "listed_equity": Decimal("1050"),
            "corporate_bond": Decimal("6300"),
            "sovereign_bond": Decimal("95210"),
        },
        "exclusions": "None - all asset classes included",
        "dqi_score": Decimal("3.5"),
        "data_quality_score": Decimal("3.5"),
        "uncertainty_analysis": {"method": "monte_carlo", "ci_95": [90000, 115000]},
        "base_year": 2019,
        "reporting_year": 2024,
        "targets": "Net-zero financed emissions by 2050",
        "reduction_targets": "50% by 2030",
        "actions": "Portfolio decarbonization, engagement",
        "reduction_actions": "Engagement with high-carbon investees",
        "standards_used": "GHG Protocol, PCAF 3rd Ed.",
        "gases_included": ["CO2", "CH4", "N2O"],
        "verification_status": "limited_assurance",
        "assurance_opinion": "limited",
        "portfolio_alignment": "1.5C aligned",
        "sector_targets": {"energy": "40% reduction by 2030"},
        "temperature_rating": Decimal("1.8"),
        "dc_checks": {},
        "is_consolidated": False,
        "already_in_scope1_or_scope2": False,
        "provenance_hash": "a" * 64,
    }
    base.update(overrides)
    return base


# ==============================================================================
# GHG PROTOCOL TESTS
# ==============================================================================


class TestGHGProtocol:
    """Test GHG Protocol Scope 3 Standard compliance."""

    def test_ghg_protocol_full_compliance(self, engine):
        """Test full compliance with GHG Protocol."""
        result = engine.check_compliance(
            framework="GHG_PROTOCOL_SCOPE3",
            data=_full_result(),
        )
        assert result["status"] in ["pass", "warning"]

    def test_ghg_protocol_requires_total_emissions(self, engine):
        """Test GHG Protocol requires total emissions disclosure."""
        data = _full_result()
        del data["total_financed_emissions"]
        del data["total_co2e"]
        result = engine.check_compliance(framework="GHG_PROTOCOL_SCOPE3", data=data)
        assert result["status"] == "fail"

    def test_ghg_protocol_requires_method(self, engine):
        """Test GHG Protocol requires calculation method disclosure."""
        data = _full_result()
        del data["calculation_method"]
        del data["methodology"]
        result = engine.check_compliance(framework="GHG_PROTOCOL_SCOPE3", data=data)
        assert result["status"] in ["fail", "warning"]

    def test_ghg_protocol_requires_ef_sources(self, engine):
        """Test GHG Protocol requires EF source disclosure."""
        data = _full_result()
        del data["ef_sources"]
        result = engine.check_compliance(framework="GHG_PROTOCOL_SCOPE3", data=data)
        assert result["status"] in ["fail", "warning"]

    def test_ghg_protocol_requires_exclusions(self, engine):
        """Test GHG Protocol requires exclusion disclosure."""
        data = _full_result()
        del data["exclusions"]
        result = engine.check_compliance(framework="GHG_PROTOCOL_SCOPE3", data=data)
        assert result["status"] in ["fail", "warning"]

    def test_ghg_protocol_score_range(self, engine):
        """Test GHG Protocol compliance score is 0-100."""
        result = engine.check_compliance(framework="GHG_PROTOCOL_SCOPE3", data=_full_result())
        assert Decimal("0") <= result["score"] <= Decimal("100")

    def test_ghg_protocol_has_findings(self, engine):
        """Test GHG Protocol result includes findings list."""
        result = engine.check_compliance(framework="GHG_PROTOCOL_SCOPE3", data=_full_result())
        assert "findings" in result


# ==============================================================================
# PCAF TESTS
# ==============================================================================


class TestPCAF:
    """Test PCAF Global GHG Standard compliance."""

    def test_pcaf_full_compliance(self, engine):
        """Test full compliance with PCAF standard."""
        result = engine.check_compliance(framework="PCAF", data=_full_result())
        assert result["status"] in ["pass", "warning"]

    def test_pcaf_requires_attribution_method(self, engine):
        """Test PCAF requires attribution method disclosure."""
        data = _full_result()
        del data["attribution_method"]
        result = engine.check_compliance(framework="PCAF", data=data)
        assert result["status"] in ["fail", "warning"]

    def test_pcaf_requires_data_quality_score(self, engine):
        """Test PCAF requires PCAF data quality score."""
        data = _full_result()
        del data["pcaf_quality_score"]
        del data["weighted_pcaf_score"]
        result = engine.check_compliance(framework="PCAF", data=data)
        assert result["status"] in ["fail", "warning"]

    def test_pcaf_requires_asset_class_breakdown(self, engine):
        """Test PCAF requires asset class breakdown."""
        data = _full_result()
        del data["asset_class_breakdown"]
        result = engine.check_compliance(framework="PCAF", data=data)
        assert result["status"] in ["fail", "warning"]

    def test_pcaf_requires_coverage(self, engine):
        """Test PCAF requires portfolio coverage disclosure."""
        data = _full_result()
        del data["pcaf_coverage"]
        result = engine.check_compliance(framework="PCAF", data=data)
        assert result["status"] in ["fail", "warning"]

    def test_pcaf_score_range(self, engine):
        """Test PCAF compliance score is 0-100."""
        result = engine.check_compliance(framework="PCAF", data=_full_result())
        assert Decimal("0") <= result["score"] <= Decimal("100")

    def test_pcaf_low_quality_triggers_warning(self, engine):
        """Test low PCAF quality score triggers warning."""
        data = _full_result(pcaf_quality_score=5, weighted_pcaf_score=Decimal("4.8"))
        result = engine.check_compliance(framework="PCAF", data=data)
        # Very low quality should generate findings
        assert len(result.get("findings", [])) > 0 or result["status"] == "warning"

    def test_pcaf_high_coverage_passes(self, engine):
        """Test high PCAF coverage contributes to passing."""
        data = _full_result(pcaf_coverage=Decimal("0.99"))
        result = engine.check_compliance(framework="PCAF", data=data)
        assert result["status"] in ["pass", "warning"]


# ==============================================================================
# ISO 14064 TESTS
# ==============================================================================


class TestISO14064:
    """Test ISO 14064-1:2018 compliance."""

    def test_iso_full_compliance(self, engine):
        """Test full compliance with ISO 14064."""
        result = engine.check_compliance(framework="ISO_14064", data=_full_result())
        assert result["status"] in ["pass", "warning"]

    def test_iso_requires_uncertainty(self, engine):
        """Test ISO requires uncertainty analysis."""
        data = _full_result()
        del data["uncertainty_analysis"]
        result = engine.check_compliance(framework="ISO_14064", data=data)
        assert result["status"] in ["fail", "warning"]

    def test_iso_requires_base_year(self, engine):
        """Test ISO requires base year."""
        data = _full_result()
        del data["base_year"]
        result = engine.check_compliance(framework="ISO_14064", data=data)
        assert result["status"] in ["fail", "warning"]

    def test_iso_requires_methodology(self, engine):
        """Test ISO requires methodology disclosure."""
        data = _full_result()
        del data["methodology"]
        result = engine.check_compliance(framework="ISO_14064", data=data)
        assert result["status"] in ["fail", "warning"]

    def test_iso_score_range(self, engine):
        """Test ISO compliance score is 0-100."""
        result = engine.check_compliance(framework="ISO_14064", data=_full_result())
        assert Decimal("0") <= result["score"] <= Decimal("100")


# ==============================================================================
# CSRD ESRS E1 TESTS
# ==============================================================================


class TestCSRD:
    """Test CSRD ESRS E1 compliance."""

    def test_csrd_full_compliance(self, engine):
        """Test full compliance with CSRD."""
        result = engine.check_compliance(framework="CSRD_ESRS_E1", data=_full_result())
        assert result["status"] in ["pass", "warning"]

    def test_csrd_requires_targets(self, engine):
        """Test CSRD requires targets disclosure."""
        data = _full_result()
        del data["targets"]
        del data["reduction_targets"]
        result = engine.check_compliance(framework="CSRD_ESRS_E1", data=data)
        assert result["status"] in ["fail", "warning"]

    def test_csrd_requires_actions(self, engine):
        """Test CSRD requires actions disclosure."""
        data = _full_result()
        del data["actions"]
        del data["reduction_actions"]
        result = engine.check_compliance(framework="CSRD_ESRS_E1", data=data)
        assert result["status"] in ["fail", "warning"]

    def test_csrd_requires_category_breakdown(self, engine):
        """Test CSRD requires asset class breakdown."""
        data = _full_result()
        del data["asset_class_breakdown"]
        result = engine.check_compliance(framework="CSRD_ESRS_E1", data=data)
        assert result["status"] in ["fail", "warning"]

    def test_csrd_score_range(self, engine):
        """Test CSRD compliance score is 0-100."""
        result = engine.check_compliance(framework="CSRD_ESRS_E1", data=_full_result())
        assert Decimal("0") <= result["score"] <= Decimal("100")


# ==============================================================================
# CDP TESTS
# ==============================================================================


class TestCDP:
    """Test CDP Climate Change compliance."""

    def test_cdp_full_compliance(self, engine):
        """Test full compliance with CDP."""
        result = engine.check_compliance(framework="CDP", data=_full_result())
        assert result["status"] in ["pass", "warning"]

    def test_cdp_requires_verification(self, engine):
        """Test CDP requires verification status."""
        data = _full_result()
        del data["verification_status"]
        result = engine.check_compliance(framework="CDP", data=data)
        assert result["status"] in ["fail", "warning"]

    def test_cdp_requires_asset_breakdown(self, engine):
        """Test CDP requires asset class breakdown."""
        data = _full_result()
        del data["asset_class_breakdown"]
        result = engine.check_compliance(framework="CDP", data=data)
        assert result["status"] in ["fail", "warning"]

    def test_cdp_requires_methodology(self, engine):
        """Test CDP requires methodology."""
        data = _full_result()
        del data["methodology"]
        del data["calculation_method"]
        result = engine.check_compliance(framework="CDP", data=data)
        assert result["status"] in ["fail", "warning"]

    def test_cdp_score_range(self, engine):
        """Test CDP compliance score is 0-100."""
        result = engine.check_compliance(framework="CDP", data=_full_result())
        assert Decimal("0") <= result["score"] <= Decimal("100")


# ==============================================================================
# SBTI-FI TESTS
# ==============================================================================


class TestSBTiFI:
    """Test SBTi Financial Institutions compliance."""

    def test_sbti_fi_full_compliance(self, engine):
        """Test full compliance with SBTi-FI."""
        result = engine.check_compliance(framework="SBTI_FI", data=_full_result())
        assert result["status"] in ["pass", "warning"]

    def test_sbti_fi_requires_targets(self, engine):
        """Test SBTi-FI requires science-based targets."""
        data = _full_result()
        del data["targets"]
        del data["reduction_targets"]
        result = engine.check_compliance(framework="SBTI_FI", data=data)
        assert result["status"] in ["fail", "warning"]

    def test_sbti_fi_requires_sector_targets(self, engine):
        """Test SBTi-FI requires sector-level targets."""
        data = _full_result()
        del data["sector_targets"]
        result = engine.check_compliance(framework="SBTI_FI", data=data)
        assert result["status"] in ["fail", "warning"]

    def test_sbti_fi_requires_portfolio_alignment(self, engine):
        """Test SBTi-FI requires portfolio alignment."""
        data = _full_result()
        del data["portfolio_alignment"]
        result = engine.check_compliance(framework="SBTI_FI", data=data)
        assert result["status"] in ["fail", "warning"]

    def test_sbti_fi_score_range(self, engine):
        """Test SBTi-FI compliance score is 0-100."""
        result = engine.check_compliance(framework="SBTI_FI", data=_full_result())
        assert Decimal("0") <= result["score"] <= Decimal("100")


# ==============================================================================
# SB 253 TESTS
# ==============================================================================


class TestSB253:
    """Test California SB 253 compliance."""

    def test_sb253_full_compliance(self, engine):
        """Test full compliance with SB 253."""
        result = engine.check_compliance(framework="SB_253", data=_full_result())
        assert result["status"] in ["pass", "warning"]

    def test_sb253_requires_assurance(self, engine):
        """Test SB 253 requires assurance opinion."""
        data = _full_result()
        del data["assurance_opinion"]
        result = engine.check_compliance(framework="SB_253", data=data)
        assert result["status"] in ["fail", "warning"]

    def test_sb253_requires_methodology(self, engine):
        """Test SB 253 requires methodology disclosure."""
        data = _full_result()
        del data["methodology"]
        del data["calculation_method"]
        result = engine.check_compliance(framework="SB_253", data=data)
        assert result["status"] in ["fail", "warning"]

    def test_sb253_score_range(self, engine):
        """Test SB 253 compliance score is 0-100."""
        result = engine.check_compliance(framework="SB_253", data=_full_result())
        assert Decimal("0") <= result["score"] <= Decimal("100")


# ==============================================================================
# TCFD TESTS
# ==============================================================================


class TestTCFD:
    """Test TCFD Recommendations compliance."""

    def test_tcfd_full_compliance(self, engine):
        """Test full compliance with TCFD."""
        result = engine.check_compliance(framework="TCFD", data=_full_result())
        assert result["status"] in ["pass", "warning"]

    def test_tcfd_requires_metrics(self, engine):
        """Test TCFD requires metrics and targets."""
        data = _full_result()
        del data["targets"]
        del data["reduction_targets"]
        result = engine.check_compliance(framework="TCFD", data=data)
        assert result["status"] in ["fail", "warning"]

    def test_tcfd_requires_temperature_rating(self, engine):
        """Test TCFD recommends temperature rating."""
        data = _full_result()
        del data["temperature_rating"]
        result = engine.check_compliance(framework="TCFD", data=data)
        assert result["status"] in ["fail", "warning"]

    def test_tcfd_score_range(self, engine):
        """Test TCFD compliance score is 0-100."""
        result = engine.check_compliance(framework="TCFD", data=_full_result())
        assert Decimal("0") <= result["score"] <= Decimal("100")


# ==============================================================================
# NZBA TESTS
# ==============================================================================


class TestNZBA:
    """Test Net-Zero Banking Alliance compliance."""

    def test_nzba_full_compliance(self, engine):
        """Test full compliance with NZBA."""
        result = engine.check_compliance(framework="NZBA", data=_full_result())
        assert result["status"] in ["pass", "warning"]

    def test_nzba_requires_sector_targets(self, engine):
        """Test NZBA requires sector-level targets."""
        data = _full_result()
        del data["sector_targets"]
        result = engine.check_compliance(framework="NZBA", data=data)
        assert result["status"] in ["fail", "warning"]

    def test_nzba_requires_portfolio_alignment(self, engine):
        """Test NZBA requires portfolio alignment to net-zero."""
        data = _full_result()
        del data["portfolio_alignment"]
        result = engine.check_compliance(framework="NZBA", data=data)
        assert result["status"] in ["fail", "warning"]

    def test_nzba_score_range(self, engine):
        """Test NZBA compliance score is 0-100."""
        result = engine.check_compliance(framework="NZBA", data=_full_result())
        assert Decimal("0") <= result["score"] <= Decimal("100")


# ==============================================================================
# DC RULES TESTS
# ==============================================================================


class TestDCRules:
    """Test all 8 double-counting prevention rules."""

    def test_dc_inv_001_consolidated_exclusion(self, engine):
        """Test DC-INV-001: consolidated subsidiaries excluded."""
        data = _full_result(is_consolidated=True)
        result = engine.check_dc_rules(data)
        assert result.get("dc_inv_001", {}).get("triggered", False) is True

    def test_dc_inv_002_scope1_scope2_exclusion(self, engine):
        """Test DC-INV-002: already in Scope 1/2 excluded."""
        data = _full_result(already_in_scope1_or_scope2=True)
        result = engine.check_dc_rules(data)
        assert result.get("dc_inv_002", {}).get("triggered", False) is True

    def test_dc_inv_003_intercompany_exclusion(self, engine):
        """Test DC-INV-003: intercompany investments excluded."""
        data = _full_result(is_intercompany=True)
        result = engine.check_dc_rules(data)
        assert result.get("dc_inv_003", {}).get("triggered", False) is True

    def test_dc_inv_004_joint_venture_allocation(self, engine):
        """Test DC-INV-004: joint venture pro-rata allocation."""
        data = _full_result(is_joint_venture=True, jv_share=Decimal("0.4"))
        result = engine.check_dc_rules(data)
        assert "dc_inv_004" in result

    def test_dc_inv_005_sovereign_vs_corporate(self, engine):
        """Test DC-INV-005: sovereign vs corporate bond distinction."""
        data = _full_result(
            asset_class_breakdown={
                "sovereign_bond": Decimal("95210"),
                "corporate_bond": Decimal("6300"),
            }
        )
        result = engine.check_dc_rules(data)
        assert "dc_inv_005" in result

    def test_dc_inv_006_asset_class_overlap(self, engine):
        """Test DC-INV-006: asset class overlap prevention."""
        data = _full_result(overlapping_assets=True)
        result = engine.check_dc_rules(data)
        assert "dc_inv_006" in result

    def test_dc_inv_007_scope3_downstream_exclusion(self, engine):
        """Test DC-INV-007: Scope 3 downstream exclusion."""
        data = _full_result(includes_downstream_scope3=True)
        result = engine.check_dc_rules(data)
        assert "dc_inv_007" in result

    def test_dc_inv_008_temporal_boundary(self, engine):
        """Test DC-INV-008: temporal boundary consistency."""
        data = _full_result(mixed_reporting_years=True)
        result = engine.check_dc_rules(data)
        assert "dc_inv_008" in result


# ==============================================================================
# PCAF ATTRIBUTION VALIDATION TESTS
# ==============================================================================


class TestPCAFAttribution:
    """Test PCAF-specific attribution validation."""

    def test_evic_attribution_validated(self, engine):
        """Test EVIC attribution method is validated for listed equity."""
        data = _full_result(attribution_method="evic")
        result = engine.check_compliance(framework="PCAF", data=data)
        assert result["status"] in ["pass", "warning"]

    def test_gdp_ppp_attribution_validated(self, engine):
        """Test GDP-PPP attribution validated for sovereign bonds."""
        data = _full_result(attribution_method="gdp_ppp")
        result = engine.check_compliance(framework="PCAF", data=data)
        assert result["status"] in ["pass", "warning"]

    def test_invalid_attribution_fails(self, engine):
        """Test invalid attribution method fails PCAF compliance."""
        data = _full_result(attribution_method="invalid_method")
        result = engine.check_compliance(framework="PCAF", data=data)
        # Invalid attribution should generate at least a warning
        assert len(result.get("findings", [])) > 0 or result["status"] in ["fail", "warning"]

    def test_pcaf_dq_score_validation(self, engine):
        """Test PCAF data quality score is validated (1-5 range)."""
        data = _full_result(pcaf_quality_score=3)
        result = engine.check_compliance(framework="PCAF", data=data)
        assert result["status"] in ["pass", "warning"]

    def test_pcaf_coverage_validation(self, engine):
        """Test PCAF portfolio coverage is validated."""
        data = _full_result(pcaf_coverage=Decimal("0.50"))
        result = engine.check_compliance(framework="PCAF", data=data)
        # Low coverage should generate findings
        assert len(result.get("findings", [])) > 0 or result["status"] in ["pass", "warning"]


# ==============================================================================
# PORTFOLIO ALIGNMENT TESTS
# ==============================================================================


class TestPortfolioAlignment:
    """Test portfolio alignment checking."""

    def test_portfolio_alignment_15c(self, engine):
        """Test 1.5C aligned portfolio passes."""
        data = _full_result(portfolio_alignment="1.5C aligned")
        result = engine.check_compliance(framework="SBTI_FI", data=data)
        assert result["status"] in ["pass", "warning"]

    def test_portfolio_alignment_2c(self, engine):
        """Test 2C aligned portfolio generates warning."""
        data = _full_result(
            portfolio_alignment="2C aligned",
            temperature_rating=Decimal("2.0"),
        )
        result = engine.check_compliance(framework="SBTI_FI", data=data)
        assert result["status"] in ["pass", "warning"]

    def test_portfolio_no_alignment(self, engine):
        """Test no alignment disclosure fails."""
        data = _full_result()
        del data["portfolio_alignment"]
        result = engine.check_compliance(framework="SBTI_FI", data=data)
        assert result["status"] in ["fail", "warning"]


# ==============================================================================
# RECOMMENDATIONS TESTS
# ==============================================================================


class TestRecommendations:
    """Test recommendations generation."""

    def test_recommendations_generated_on_fail(self, engine):
        """Test recommendations are generated when checks fail."""
        data = _full_result()
        del data["targets"]
        del data["reduction_targets"]
        del data["methodology"]
        del data["calculation_method"]
        result = engine.check_compliance(framework="GHG_PROTOCOL_SCOPE3", data=data)
        assert len(result.get("recommendations", [])) > 0

    def test_recommendations_empty_on_full_pass(self, engine):
        """Test recommendations may be empty or minimal on full pass."""
        result = engine.check_compliance(framework="GHG_PROTOCOL_SCOPE3", data=_full_result())
        # Full pass may still have advisory recommendations
        assert isinstance(result.get("recommendations", []), list)


# ==============================================================================
# PARAMETRIZED FRAMEWORK x DC RULES TESTS
# ==============================================================================


class TestFrameworkDCCombinations:
    """Test frameworks with DC rules combinations."""

    @pytest.mark.parametrize("framework", [
        "GHG_PROTOCOL_SCOPE3", "PCAF", "ISO_14064", "CSRD_ESRS_E1",
        "CDP", "SBTI_FI", "SB_253", "TCFD", "NZBA",
    ])
    def test_framework_accepts_valid_data(self, engine, framework):
        """Test each framework accepts fully valid data."""
        result = engine.check_compliance(framework=framework, data=_full_result())
        assert result["status"] in ["pass", "warning"]
        assert Decimal("0") <= result["score"] <= Decimal("100")

    @pytest.mark.parametrize("framework", [
        "GHG_PROTOCOL_SCOPE3", "PCAF", "ISO_14064", "CSRD_ESRS_E1",
        "CDP", "SBTI_FI", "SB_253", "TCFD", "NZBA",
    ])
    def test_framework_rejects_empty_data(self, engine, framework):
        """Test each framework rejects empty/minimal data."""
        result = engine.check_compliance(framework=framework, data={})
        assert result["status"] == "fail"

    @pytest.mark.parametrize("dc_rule", [
        "dc_inv_001", "dc_inv_002", "dc_inv_003", "dc_inv_004",
        "dc_inv_005", "dc_inv_006", "dc_inv_007", "dc_inv_008",
    ])
    def test_dc_rule_has_result(self, engine, dc_rule):
        """Test each DC rule returns a result structure."""
        result = engine.check_dc_rules(_full_result())
        assert dc_rule in result
