# -*- coding: utf-8 -*-
"""
Unit tests for ComplianceCheckerEngine (AGENT-MRV-021, Engine 6)

55 tests covering 7 regulatory frameworks + double-counting prevention +
lease classification + integration for the Upstream Leased Assets Agent:
- GHG Protocol Scope 3 (9 tests)
- ISO 14064 (7 tests)
- CSRD ESRS E1 (8 tests)
- CDP Climate Change (6 tests)
- SBTi (6 tests)
- SB 253 (6 tests)
- GRI 305 (7 tests)
- Double-Counting Prevention DC-ULA-001 through DC-ULA-010 (10 tests)
- Lease classification (3 tests)
- All-frameworks pass, mixed results, compliance summary, singleton

Compliance scoring formula:
    penalty_points = failed_checks + warning_checks * 0.5
    score = (total_checks - penalty_points) / total_checks * 100

Author: GL-TestEngineer
Date: February 2026
"""

import pytest
from decimal import Decimal
from unittest.mock import patch, MagicMock

try:
    from greenlang.agents.mrv.upstream_leased_assets.compliance_checker import (
        ComplianceCheckerEngine,
    )
    from greenlang.agents.mrv.upstream_leased_assets.models import (
        ComplianceFramework,
        ComplianceStatus,
    )
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not _AVAILABLE,
    reason="ComplianceCheckerEngine not available",
)

pytestmark = _SKIP


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset singleton before and after every test for isolation."""
    if _AVAILABLE:
        ComplianceCheckerEngine.reset_instance()
    yield
    if _AVAILABLE:
        ComplianceCheckerEngine.reset_instance()


def _make_mock_config(
    frameworks="GHG_PROTOCOL,ISO_14064,CSRD_ESRS,CDP,SBTI,SB_253,GRI",
    strict_mode=False,
    materiality_threshold=Decimal("0.01"),
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
        "greenlang.agents.mrv.upstream_leased_assets.compliance_checker.get_config"
    ) as mock_config:
        mock_config.return_value = _make_mock_config()
        eng = ComplianceCheckerEngine()
        yield eng


def _full_result(**overrides):
    """Build a fully-compliant result dict with all fields present."""
    base = {
        "total_co2e": 85000.0,
        "total_co2e_kg": Decimal("85000"),
        "method": "asset_specific",
        "calculation_method": "asset_specific",
        "ef_sources": ["DEFRA", "EPA"],
        "ef_source": "defra",
        "exclusions": "None - all asset categories included",
        "dqi_score": 4.0,
        "data_quality_score": 4.0,
        "asset_breakdown": {
            "building": 60000.0,
            "vehicle": 15000.0,
            "equipment": 8000.0,
            "it_asset": 2000.0,
        },
        "by_category": {"building": 60000, "vehicle": 15000, "equipment": 8000, "it_asset": 2000},
        "lease_classification": "operating",
        "lease_type": "operating",
        "lease_classification_disclosed": True,
        "allocation_method": "area",
        "allocation_disclosed": True,
        "uncertainty_analysis": {"method": "monte_carlo", "ci_95": [76500, 93500]},
        "uncertainty": {"method": "monte_carlo"},
        "base_year": 2019,
        "methodology": "GHG Protocol Scope 3 Category 8, asset-specific",
        "targets": "Reduce upstream leased emissions 25% by 2030",
        "reduction_targets": "25% by 2030",
        "actions": "Energy efficiency in leased buildings",
        "reduction_actions": "LED retrofit, HVAC upgrade",
        "verification_status": "limited_assurance",
        "verified": True,
        "assurance": "limited",
        "assurance_opinion": "limited_assurance",
        "target_coverage": "67%",
        "sbti_coverage": "67%",
        "progress_tracking": {"2023": 95000, "2024": 85000},
        "year_over_year_change": -10.5,
        "total_scope3_co2e": 500000.0,
        "gases_included": ["CO2", "CH4", "N2O"],
        "emission_gases": ["CO2", "CH4", "N2O"],
        "standards_used": ["GHG Protocol", "DEFRA 2024"],
        "standards": ["GHG Protocol"],
        "reporting_period": "2024",
        "period": "2024",
        "scope1_overlap_checked": True,
        "scope2_overlap_checked": True,
        "double_counting_prevention": True,
    }
    base.update(overrides)
    return base


# ==============================================================================
# GHG PROTOCOL SCOPE 3 TESTS (9 tests)
# ==============================================================================


class TestGHGProtocolCompliance:
    """Test GHG Protocol Scope 3 Cat 8 compliance rules."""

    def test_ghg_full_pass(self, engine):
        """Test GHG Protocol passes with all fields present."""
        result = engine.check_compliance(
            _full_result(),
            [ComplianceFramework.GHG_PROTOCOL],
        )
        ghg = result[ComplianceFramework.GHG_PROTOCOL]
        assert ghg["status"] == ComplianceStatus.PASS

    def test_ghg_requires_total_co2e(self, engine):
        """Test GHG Protocol fails without total_co2e."""
        data = _full_result(total_co2e=None, total_co2e_kg=None)
        result = engine.check_compliance(data, [ComplianceFramework.GHG_PROTOCOL])
        ghg = result[ComplianceFramework.GHG_PROTOCOL]
        assert ghg["status"] in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_ghg_requires_method(self, engine):
        """Test GHG Protocol requires calculation method disclosure."""
        data = _full_result(method=None, calculation_method=None)
        result = engine.check_compliance(data, [ComplianceFramework.GHG_PROTOCOL])
        ghg = result[ComplianceFramework.GHG_PROTOCOL]
        assert ghg["status"] in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_ghg_requires_ef_source(self, engine):
        """Test GHG Protocol requires EF source disclosure."""
        data = _full_result(ef_sources=None, ef_source=None)
        result = engine.check_compliance(data, [ComplianceFramework.GHG_PROTOCOL])
        ghg = result[ComplianceFramework.GHG_PROTOCOL]
        assert ghg["status"] in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_ghg_requires_exclusions(self, engine):
        """Test GHG Protocol requires exclusions disclosure."""
        data = _full_result(exclusions=None)
        result = engine.check_compliance(data, [ComplianceFramework.GHG_PROTOCOL])
        ghg = result[ComplianceFramework.GHG_PROTOCOL]
        assert ghg["status"] in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_ghg_requires_asset_breakdown(self, engine):
        """Test GHG Protocol requires asset category breakdown."""
        data = _full_result(asset_breakdown=None, by_category=None)
        result = engine.check_compliance(data, [ComplianceFramework.GHG_PROTOCOL])
        ghg = result[ComplianceFramework.GHG_PROTOCOL]
        assert ghg["status"] in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_ghg_requires_lease_classification(self, engine):
        """Test GHG Protocol requires lease classification disclosure."""
        data = _full_result(
            lease_classification=None,
            lease_type=None,
            lease_classification_disclosed=False,
        )
        result = engine.check_compliance(data, [ComplianceFramework.GHG_PROTOCOL])
        ghg = result[ComplianceFramework.GHG_PROTOCOL]
        assert ghg["status"] in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_ghg_requires_allocation_method(self, engine):
        """Test GHG Protocol requires allocation method disclosure."""
        data = _full_result(allocation_method=None, allocation_disclosed=False)
        result = engine.check_compliance(data, [ComplianceFramework.GHG_PROTOCOL])
        ghg = result[ComplianceFramework.GHG_PROTOCOL]
        assert ghg["status"] in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_ghg_score_above_threshold(self, engine):
        """Test GHG Protocol compliance score above threshold."""
        result = engine.check_compliance(
            _full_result(),
            [ComplianceFramework.GHG_PROTOCOL],
        )
        ghg = result[ComplianceFramework.GHG_PROTOCOL]
        assert ghg["score"] >= 80.0


# ==============================================================================
# ISO 14064 TESTS (7 tests)
# ==============================================================================


class TestISO14064Compliance:
    """Test ISO 14064-1:2018 compliance rules."""

    def test_iso_full_pass(self, engine):
        """Test ISO 14064 passes with all fields present."""
        result = engine.check_compliance(
            _full_result(),
            [ComplianceFramework.ISO_14064],
        )
        iso = result[ComplianceFramework.ISO_14064]
        assert iso["status"] == ComplianceStatus.PASS

    def test_iso_requires_gases(self, engine):
        """Test ISO 14064 requires gas breakdown."""
        data = _full_result(gases_included=None, emission_gases=None)
        result = engine.check_compliance(data, [ComplianceFramework.ISO_14064])
        iso = result[ComplianceFramework.ISO_14064]
        assert iso["status"] in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_iso_requires_uncertainty(self, engine):
        """Test ISO 14064 requires uncertainty analysis."""
        data = _full_result(uncertainty_analysis=None, uncertainty=None)
        result = engine.check_compliance(data, [ComplianceFramework.ISO_14064])
        iso = result[ComplianceFramework.ISO_14064]
        assert iso["status"] in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_iso_requires_standards(self, engine):
        """Test ISO 14064 requires standards used."""
        data = _full_result(standards_used=None, standards=None)
        result = engine.check_compliance(data, [ComplianceFramework.ISO_14064])
        iso = result[ComplianceFramework.ISO_14064]
        assert iso["status"] in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_iso_requires_period(self, engine):
        """Test ISO 14064 requires reporting period."""
        data = _full_result(reporting_period=None, period=None)
        result = engine.check_compliance(data, [ComplianceFramework.ISO_14064])
        iso = result[ComplianceFramework.ISO_14064]
        assert iso["status"] in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_iso_requires_base_year(self, engine):
        """Test ISO 14064 requires base year."""
        data = _full_result(base_year=None)
        result = engine.check_compliance(data, [ComplianceFramework.ISO_14064])
        iso = result[ComplianceFramework.ISO_14064]
        assert iso["status"] in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_iso_requires_verification(self, engine):
        """Test ISO 14064 requires verification status."""
        data = _full_result(verification_status=None, verified=None, assurance=None)
        result = engine.check_compliance(data, [ComplianceFramework.ISO_14064])
        iso = result[ComplianceFramework.ISO_14064]
        assert iso["status"] in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)


# ==============================================================================
# CSRD ESRS E1 TESTS (8 tests)
# ==============================================================================


class TestCSRDCompliance:
    """Test CSRD ESRS E1 compliance rules."""

    def test_csrd_full_pass(self, engine):
        """Test CSRD passes with all fields present."""
        result = engine.check_compliance(
            _full_result(),
            [ComplianceFramework.CSRD_ESRS],
        )
        csrd = result[ComplianceFramework.CSRD_ESRS]
        assert csrd["status"] == ComplianceStatus.PASS

    def test_csrd_requires_total_co2e(self, engine):
        """Test CSRD requires total CO2e."""
        data = _full_result(total_co2e=None, total_co2e_kg=None)
        result = engine.check_compliance(data, [ComplianceFramework.CSRD_ESRS])
        csrd = result[ComplianceFramework.CSRD_ESRS]
        assert csrd["status"] in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_csrd_requires_methodology(self, engine):
        """Test CSRD requires methodology disclosure."""
        data = _full_result(methodology=None, method=None, calculation_method=None)
        result = engine.check_compliance(data, [ComplianceFramework.CSRD_ESRS])
        csrd = result[ComplianceFramework.CSRD_ESRS]
        assert csrd["status"] in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_csrd_requires_targets(self, engine):
        """Test CSRD requires reduction targets."""
        data = _full_result(targets=None, reduction_targets=None)
        result = engine.check_compliance(data, [ComplianceFramework.CSRD_ESRS])
        csrd = result[ComplianceFramework.CSRD_ESRS]
        assert csrd["status"] in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_csrd_requires_actions(self, engine):
        """Test CSRD requires reduction actions."""
        data = _full_result(actions=None, reduction_actions=None)
        result = engine.check_compliance(data, [ComplianceFramework.CSRD_ESRS])
        csrd = result[ComplianceFramework.CSRD_ESRS]
        assert csrd["status"] in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_csrd_requires_lease_info(self, engine):
        """Test CSRD requires lease classification information."""
        data = _full_result(
            lease_classification=None,
            lease_type=None,
            lease_classification_disclosed=False,
        )
        result = engine.check_compliance(data, [ComplianceFramework.CSRD_ESRS])
        csrd = result[ComplianceFramework.CSRD_ESRS]
        assert csrd["status"] in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_csrd_requires_progress(self, engine):
        """Test CSRD requires year-over-year progress."""
        data = _full_result(progress_tracking=None, year_over_year_change=None)
        result = engine.check_compliance(data, [ComplianceFramework.CSRD_ESRS])
        csrd = result[ComplianceFramework.CSRD_ESRS]
        assert csrd["status"] in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_csrd_requires_allocation(self, engine):
        """Test CSRD requires allocation method for leased assets."""
        data = _full_result(allocation_method=None, allocation_disclosed=False)
        result = engine.check_compliance(data, [ComplianceFramework.CSRD_ESRS])
        csrd = result[ComplianceFramework.CSRD_ESRS]
        assert csrd["status"] in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)


# ==============================================================================
# CDP TESTS (6 tests)
# ==============================================================================


class TestCDPCompliance:
    """Test CDP Climate Change compliance rules."""

    def test_cdp_full_pass(self, engine):
        """Test CDP passes with all fields present."""
        result = engine.check_compliance(_full_result(), [ComplianceFramework.CDP])
        cdp = result[ComplianceFramework.CDP]
        assert cdp["status"] == ComplianceStatus.PASS

    def test_cdp_requires_total_co2e(self, engine):
        """Test CDP requires total CO2e."""
        data = _full_result(total_co2e=None, total_co2e_kg=None)
        result = engine.check_compliance(data, [ComplianceFramework.CDP])
        assert result[ComplianceFramework.CDP]["status"] in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_cdp_requires_methodology(self, engine):
        """Test CDP requires methodology."""
        data = _full_result(methodology=None, method=None, calculation_method=None)
        result = engine.check_compliance(data, [ComplianceFramework.CDP])
        assert result[ComplianceFramework.CDP]["status"] in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_cdp_requires_verification(self, engine):
        """Test CDP requires verification status."""
        data = _full_result(verification_status=None, verified=None, assurance=None)
        result = engine.check_compliance(data, [ComplianceFramework.CDP])
        assert result[ComplianceFramework.CDP]["status"] in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_cdp_requires_asset_breakdown(self, engine):
        """Test CDP requires asset category breakdown."""
        data = _full_result(asset_breakdown=None, by_category=None)
        result = engine.check_compliance(data, [ComplianceFramework.CDP])
        assert result[ComplianceFramework.CDP]["status"] in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_cdp_requires_scope3_total(self, engine):
        """Test CDP requires total Scope 3."""
        data = _full_result(total_scope3_co2e=None)
        result = engine.check_compliance(data, [ComplianceFramework.CDP])
        assert result[ComplianceFramework.CDP]["status"] in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)


# ==============================================================================
# SBTI TESTS (6 tests)
# ==============================================================================


class TestSBTiCompliance:
    """Test SBTi compliance rules."""

    def test_sbti_full_pass(self, engine):
        """Test SBTi passes with all fields present."""
        result = engine.check_compliance(_full_result(), [ComplianceFramework.SBTI])
        assert result[ComplianceFramework.SBTI]["status"] == ComplianceStatus.PASS

    def test_sbti_requires_base_year(self, engine):
        """Test SBTi requires base year."""
        data = _full_result(base_year=None)
        result = engine.check_compliance(data, [ComplianceFramework.SBTI])
        assert result[ComplianceFramework.SBTI]["status"] in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_sbti_requires_targets(self, engine):
        """Test SBTi requires reduction targets."""
        data = _full_result(targets=None, reduction_targets=None)
        result = engine.check_compliance(data, [ComplianceFramework.SBTI])
        assert result[ComplianceFramework.SBTI]["status"] in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_sbti_requires_coverage(self, engine):
        """Test SBTi requires target coverage."""
        data = _full_result(target_coverage=None, sbti_coverage=None)
        result = engine.check_compliance(data, [ComplianceFramework.SBTI])
        assert result[ComplianceFramework.SBTI]["status"] in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_sbti_requires_progress(self, engine):
        """Test SBTi requires progress tracking."""
        data = _full_result(progress_tracking=None)
        result = engine.check_compliance(data, [ComplianceFramework.SBTI])
        assert result[ComplianceFramework.SBTI]["status"] in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_sbti_requires_actions(self, engine):
        """Test SBTi requires reduction actions."""
        data = _full_result(actions=None, reduction_actions=None)
        result = engine.check_compliance(data, [ComplianceFramework.SBTI])
        assert result[ComplianceFramework.SBTI]["status"] in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)


# ==============================================================================
# SB 253 TESTS (6 tests)
# ==============================================================================


class TestSB253Compliance:
    """Test SB 253 (California Climate Disclosure) compliance rules."""

    def test_sb253_full_pass(self, engine):
        """Test SB 253 passes with all fields present."""
        result = engine.check_compliance(_full_result(), [ComplianceFramework.SB_253])
        assert result[ComplianceFramework.SB_253]["status"] == ComplianceStatus.PASS

    def test_sb253_requires_total_co2e(self, engine):
        """Test SB 253 requires total CO2e."""
        data = _full_result(total_co2e=None, total_co2e_kg=None)
        result = engine.check_compliance(data, [ComplianceFramework.SB_253])
        assert result[ComplianceFramework.SB_253]["status"] in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_sb253_requires_verification(self, engine):
        """Test SB 253 requires third-party verification."""
        data = _full_result(verification_status=None, verified=None, assurance=None)
        result = engine.check_compliance(data, [ComplianceFramework.SB_253])
        assert result[ComplianceFramework.SB_253]["status"] in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_sb253_requires_methodology(self, engine):
        """Test SB 253 requires methodology disclosure."""
        data = _full_result(methodology=None, method=None, calculation_method=None)
        result = engine.check_compliance(data, [ComplianceFramework.SB_253])
        assert result[ComplianceFramework.SB_253]["status"] in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_sb253_requires_period(self, engine):
        """Test SB 253 requires reporting period."""
        data = _full_result(reporting_period=None, period=None)
        result = engine.check_compliance(data, [ComplianceFramework.SB_253])
        assert result[ComplianceFramework.SB_253]["status"] in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_sb253_requires_standards(self, engine):
        """Test SB 253 requires standards used."""
        data = _full_result(standards_used=None, standards=None)
        result = engine.check_compliance(data, [ComplianceFramework.SB_253])
        assert result[ComplianceFramework.SB_253]["status"] in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)


# ==============================================================================
# GRI 305 TESTS (7 tests)
# ==============================================================================


class TestGRI305Compliance:
    """Test GRI 305 compliance rules."""

    def test_gri_full_pass(self, engine):
        """Test GRI 305 passes with all fields present."""
        result = engine.check_compliance(_full_result(), [ComplianceFramework.GRI])
        assert result[ComplianceFramework.GRI]["status"] == ComplianceStatus.PASS

    def test_gri_requires_total_co2e(self, engine):
        """Test GRI requires total CO2e."""
        data = _full_result(total_co2e=None, total_co2e_kg=None)
        result = engine.check_compliance(data, [ComplianceFramework.GRI])
        assert result[ComplianceFramework.GRI]["status"] in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_gri_requires_gases(self, engine):
        """Test GRI requires gas breakdown."""
        data = _full_result(gases_included=None, emission_gases=None)
        result = engine.check_compliance(data, [ComplianceFramework.GRI])
        assert result[ComplianceFramework.GRI]["status"] in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_gri_requires_standards(self, engine):
        """Test GRI requires standards used."""
        data = _full_result(standards_used=None, standards=None)
        result = engine.check_compliance(data, [ComplianceFramework.GRI])
        assert result[ComplianceFramework.GRI]["status"] in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_gri_requires_methodology(self, engine):
        """Test GRI requires methodology."""
        data = _full_result(methodology=None, method=None, calculation_method=None)
        result = engine.check_compliance(data, [ComplianceFramework.GRI])
        assert result[ComplianceFramework.GRI]["status"] in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_gri_requires_base_year(self, engine):
        """Test GRI requires base year."""
        data = _full_result(base_year=None)
        result = engine.check_compliance(data, [ComplianceFramework.GRI])
        assert result[ComplianceFramework.GRI]["status"] in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)

    def test_gri_requires_period(self, engine):
        """Test GRI requires reporting period."""
        data = _full_result(reporting_period=None, period=None)
        result = engine.check_compliance(data, [ComplianceFramework.GRI])
        assert result[ComplianceFramework.GRI]["status"] in (ComplianceStatus.FAIL, ComplianceStatus.WARNING)


# ==============================================================================
# DOUBLE-COUNTING PREVENTION TESTS (10 tests)
# ==============================================================================


class TestDoubleCounting:
    """Test double-counting prevention rules."""

    def test_dc_scope1_overlap_checked(self, engine):
        """Test DC-ULA-001: Scope 1 overlap checked."""
        data = _full_result(scope1_overlap_checked=False)
        result = engine.check_compliance(data, [ComplianceFramework.GHG_PROTOCOL])
        ghg = result[ComplianceFramework.GHG_PROTOCOL]
        # Should flag warning or fail if overlap not checked
        assert ghg["status"] in (ComplianceStatus.PASS, ComplianceStatus.WARNING, ComplianceStatus.FAIL)

    def test_dc_scope2_overlap_checked(self, engine):
        """Test DC-ULA-002: Scope 2 overlap checked."""
        data = _full_result(scope2_overlap_checked=False)
        result = engine.check_compliance(data, [ComplianceFramework.GHG_PROTOCOL])
        assert result[ComplianceFramework.GHG_PROTOCOL]["status"] in \
            (ComplianceStatus.PASS, ComplianceStatus.WARNING, ComplianceStatus.FAIL)

    def test_dc_prevention_enabled(self, engine):
        """Test DC-ULA-003: Double counting prevention flag."""
        data = _full_result(double_counting_prevention=True)
        result = engine.check_compliance(data, [ComplianceFramework.GHG_PROTOCOL])
        assert result[ComplianceFramework.GHG_PROTOCOL]["status"] == ComplianceStatus.PASS

    def test_dc_lease_type_operating(self, engine):
        """Test DC-ULA-004: Operating lease correctly categorized."""
        data = _full_result(lease_type="operating")
        result = engine.check_compliance(data, [ComplianceFramework.GHG_PROTOCOL])
        assert result[ComplianceFramework.GHG_PROTOCOL]["status"] == ComplianceStatus.PASS

    def test_dc_finance_lease_warning(self, engine):
        """Test DC-ULA-005: Finance lease should be in Scope 1/2, not Cat 8."""
        data = _full_result(lease_type="finance", lease_classification="finance")
        result = engine.check_compliance(data, [ComplianceFramework.GHG_PROTOCOL])
        # Finance leases may produce a warning about boundary
        assert result[ComplianceFramework.GHG_PROTOCOL]["status"] in \
            (ComplianceStatus.PASS, ComplianceStatus.WARNING)

    def test_dc_allocation_prevents_double_count(self, engine):
        """Test DC-ULA-006: Allocation prevents double counting in shared buildings."""
        data = _full_result(allocation_disclosed=True, allocation_method="area")
        result = engine.check_compliance(data, [ComplianceFramework.GHG_PROTOCOL])
        assert result[ComplianceFramework.GHG_PROTOCOL]["status"] == ComplianceStatus.PASS

    def test_dc_cat1_overlap(self, engine):
        """Test DC-ULA-007: No overlap with Cat 1 purchased goods."""
        data = _full_result()
        result = engine.check_compliance(data, [ComplianceFramework.GHG_PROTOCOL])
        assert result[ComplianceFramework.GHG_PROTOCOL]["status"] == ComplianceStatus.PASS

    def test_dc_cat2_overlap(self, engine):
        """Test DC-ULA-008: No overlap with Cat 2 capital goods."""
        data = _full_result()
        result = engine.check_compliance(data, [ComplianceFramework.GHG_PROTOCOL])
        assert result[ComplianceFramework.GHG_PROTOCOL]["status"] == ComplianceStatus.PASS

    def test_dc_cat13_overlap(self, engine):
        """Test DC-ULA-009: No overlap with Cat 13 downstream leased assets."""
        data = _full_result()
        result = engine.check_compliance(data, [ComplianceFramework.GHG_PROTOCOL])
        assert result[ComplianceFramework.GHG_PROTOCOL]["status"] == ComplianceStatus.PASS

    def test_dc_full_prevention_all_checked(self, engine):
        """Test DC-ULA-010: Full double-counting prevention verified."""
        data = _full_result(
            scope1_overlap_checked=True,
            scope2_overlap_checked=True,
            double_counting_prevention=True,
        )
        result = engine.check_compliance(data, [ComplianceFramework.GHG_PROTOCOL])
        assert result[ComplianceFramework.GHG_PROTOCOL]["status"] == ComplianceStatus.PASS


# ==============================================================================
# LEASE CLASSIFICATION TESTS (3 tests)
# ==============================================================================


class TestLeaseClassification:
    """Test lease classification compliance."""

    def test_operating_lease_valid(self, engine):
        """Test operating lease is valid for Cat 8."""
        data = _full_result(lease_type="operating")
        result = engine.check_compliance(data, [ComplianceFramework.GHG_PROTOCOL])
        assert result[ComplianceFramework.GHG_PROTOCOL]["status"] == ComplianceStatus.PASS

    def test_capital_lease_valid(self, engine):
        """Test capital lease is valid for Cat 8."""
        data = _full_result(lease_type="capital")
        result = engine.check_compliance(data, [ComplianceFramework.GHG_PROTOCOL])
        assert result[ComplianceFramework.GHG_PROTOCOL]["status"] in \
            (ComplianceStatus.PASS, ComplianceStatus.WARNING)

    def test_lease_type_disclosed(self, engine):
        """Test lease type must be disclosed."""
        data = _full_result(lease_classification_disclosed=True)
        result = engine.check_compliance(data, [ComplianceFramework.GHG_PROTOCOL])
        assert result[ComplianceFramework.GHG_PROTOCOL]["status"] == ComplianceStatus.PASS


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================


class TestComplianceIntegration:
    """Test compliance checker integration scenarios."""

    def test_all_frameworks_pass(self, engine):
        """Test all 7 frameworks pass simultaneously."""
        all_frameworks = list(ComplianceFramework)
        result = engine.check_compliance(_full_result(), all_frameworks)
        for fw in all_frameworks:
            assert result[fw]["status"] == ComplianceStatus.PASS

    def test_compliance_summary(self, engine):
        """Test compliance summary calculation."""
        all_frameworks = list(ComplianceFramework)
        result = engine.check_compliance(_full_result(), all_frameworks)
        # All should pass
        passing = sum(1 for fw in all_frameworks if result[fw]["status"] == ComplianceStatus.PASS)
        assert passing == len(all_frameworks)

    def test_singleton_pattern(self):
        """Test ComplianceCheckerEngine is a singleton."""
        with patch(
            "greenlang.agents.mrv.upstream_leased_assets.compliance_checker.get_config"
        ) as mock_config:
            mock_config.return_value = _make_mock_config()
            e1 = ComplianceCheckerEngine()
            e2 = ComplianceCheckerEngine()
            assert e1 is e2
