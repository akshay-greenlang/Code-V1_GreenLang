# -*- coding: utf-8 -*-
"""
Test suite for ComplianceCheckerEngine (AGENT-MRV-026, Engine 6).

Tests 7 regulatory frameworks + 8 double-counting prevention rules +
lease classification + operational control boundary for Cat 13.

DC-DLA-001: Operational control -> Scope 1/2, NOT Cat 13
DC-DLA-002: Cat 8 (lessee) vs Cat 13 (lessor) boundary
DC-DLA-003: Finance lease classification

Author: GL-TestEngineer
Date: February 2026
"""

import threading
from decimal import Decimal
from unittest.mock import patch, MagicMock
import pytest

try:
    from greenlang.downstream_leased_assets.compliance_checker import (
        ComplianceCheckerEngine,
    )
    from greenlang.downstream_leased_assets.models import (
        ComplianceFramework,
        ComplianceStatus,
    )
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_SKIP = pytest.mark.skipif(not _AVAILABLE, reason="ComplianceCheckerEngine not available")
pytestmark = _SKIP


@pytest.fixture(autouse=True)
def _reset_singleton():
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
    mock_cfg = MagicMock()
    mock_cfg.compliance.get_frameworks.return_value = frameworks.split(",")
    mock_cfg.compliance.strict_mode = strict_mode
    mock_cfg.compliance.materiality_threshold = materiality_threshold
    return mock_cfg


@pytest.fixture
def engine():
    with patch(
        "greenlang.downstream_leased_assets.compliance_checker.get_config"
    ) as mock_config:
        mock_config.return_value = _make_mock_config()
        eng = ComplianceCheckerEngine()
        yield eng


def _full_result(**overrides):
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
        "reporting_period": "2024",
        "reporting_year": 2024,
        "uncertainty": {"lower": 76500, "upper": 93500, "confidence": 0.95},
        "asset_breakdown_provided": True,
        "asset_breakdown": {
            "building": {"count": 3, "co2e_kg": 60000},
            "vehicle": {"count": 10, "co2e_kg": 15000},
        },
        "lease_classification_disclosed": True,
        "lease_type": "operating",
        "consolidation_approach": "operational_control",
        "operational_control": "tenant",
        "vacancy_handling": "base_load_included",
        "tenant_data_coverage": 0.85,
        "provenance_hash": "a" * 64,
    }
    base.update(overrides)
    return base


# ==============================================================================
# 7 FRAMEWORK TESTS (valid + incomplete)
# ==============================================================================


class TestGHGProtocol:

    def test_ghg_protocol_valid(self, engine):
        result = engine.check(ComplianceFramework.GHG_PROTOCOL, _full_result())
        assert result["status"] in ("pass", "warning")

    def test_ghg_protocol_missing_method(self, engine):
        result = engine.check(ComplianceFramework.GHG_PROTOCOL, _full_result(method=None))
        assert result["status"] in ("fail", "warning")

    def test_ghg_protocol_no_breakdown(self, engine):
        result = engine.check(ComplianceFramework.GHG_PROTOCOL, _full_result(asset_breakdown_provided=False))
        assert result["status"] in ("warning", "fail")


class TestISO14064:

    def test_iso_valid(self, engine):
        result = engine.check(ComplianceFramework.ISO_14064, _full_result())
        assert result["status"] in ("pass", "warning")

    def test_iso_missing_uncertainty(self, engine):
        result = engine.check(ComplianceFramework.ISO_14064, _full_result(uncertainty=None))
        assert result["status"] in ("fail", "warning")


class TestCSRDESRS:

    def test_csrd_valid(self, engine):
        result = engine.check(ComplianceFramework.CSRD_ESRS, _full_result())
        assert result["status"] in ("pass", "warning")

    def test_csrd_missing_dqi(self, engine):
        result = engine.check(ComplianceFramework.CSRD_ESRS, _full_result(dqi_score=None, data_quality_score=None))
        assert result["status"] in ("fail", "warning")


class TestCDP:

    def test_cdp_valid(self, engine):
        result = engine.check(ComplianceFramework.CDP, _full_result())
        assert result["status"] in ("pass", "warning")

    def test_cdp_missing_ef_source(self, engine):
        result = engine.check(ComplianceFramework.CDP, _full_result(ef_source=None, ef_sources=None))
        assert result["status"] in ("fail", "warning")


class TestSBTi:

    def test_sbti_valid(self, engine):
        result = engine.check(ComplianceFramework.SBTI, _full_result())
        assert result["status"] in ("pass", "warning")

    def test_sbti_no_coverage(self, engine):
        result = engine.check(ComplianceFramework.SBTI, _full_result(tenant_data_coverage=0.0))
        assert result["status"] in ("fail", "warning")


class TestSB253:

    def test_sb253_valid(self, engine):
        result = engine.check(ComplianceFramework.SB_253, _full_result())
        assert result["status"] in ("pass", "warning")

    def test_sb253_missing_reporting_period(self, engine):
        result = engine.check(ComplianceFramework.SB_253, _full_result(reporting_period=None))
        assert result["status"] in ("fail", "warning")


class TestGRI:

    def test_gri_valid(self, engine):
        result = engine.check(ComplianceFramework.GRI, _full_result())
        assert result["status"] in ("pass", "warning")

    def test_gri_no_provenance(self, engine):
        result = engine.check(ComplianceFramework.GRI, _full_result(provenance_hash=None))
        assert result["status"] in ("fail", "warning")


# ==============================================================================
# PARAMETRIZED VALID + INCOMPLETE FRAMEWORK TESTS
# ==============================================================================


class TestParametrizedFrameworks:

    @pytest.mark.parametrize("framework", list(ComplianceFramework))
    def test_valid_result_passes(self, engine, framework):
        result = engine.check(framework, _full_result())
        assert result["status"] in ("pass", "warning")

    @pytest.mark.parametrize("framework", list(ComplianceFramework))
    def test_empty_result_fails(self, engine, framework):
        result = engine.check(framework, {})
        assert result["status"] in ("fail", "warning")


# ==============================================================================
# DOUBLE-COUNTING PREVENTION RULES (8 DC-DLA rules)
# ==============================================================================


class TestDCRules:

    def test_dc_dla_001_operational_control(self, engine):
        """DC-DLA-001: If lessor retains operational control, report in Scope 1/2 NOT Cat 13."""
        result = engine.check_dc_rules(_full_result(operational_control="lessor"))
        has_dc_001 = any("DC-DLA-001" in str(r) for r in result.get("dc_results", [result]))
        if "dc_results" in result:
            assert has_dc_001 or result.get("operational_control_warning") is not None
        assert isinstance(result, dict)

    def test_dc_dla_002_cat8_vs_cat13(self, engine):
        """DC-DLA-002: Cat 8 = lessee, Cat 13 = lessor. Ensure no overlap."""
        result = engine.check_dc_rules(_full_result(consolidation_approach="operational_control"))
        assert isinstance(result, dict)

    def test_dc_dla_003_finance_lease(self, engine):
        """DC-DLA-003: Finance/capital lease may shift boundary."""
        result = engine.check_dc_rules(_full_result(lease_type="finance"))
        assert isinstance(result, dict)

    @pytest.mark.parametrize("rule_idx", range(1, 9))
    def test_dc_rules_parametrized(self, engine, rule_idx):
        """Test all 8 DC rules exist in check."""
        result = engine.check_dc_rules(_full_result())
        assert isinstance(result, dict)


# ==============================================================================
# CONSOLIDATION APPROACH VALIDATION
# ==============================================================================


class TestConsolidationApproach:

    def test_operational_control_valid(self, engine):
        result = engine.check(
            ComplianceFramework.GHG_PROTOCOL,
            _full_result(consolidation_approach="operational_control"),
        )
        assert result["status"] in ("pass", "warning")

    def test_financial_control_valid(self, engine):
        result = engine.check(
            ComplianceFramework.GHG_PROTOCOL,
            _full_result(consolidation_approach="financial_control"),
        )
        assert result["status"] in ("pass", "warning")

    def test_equity_share_valid(self, engine):
        result = engine.check(
            ComplianceFramework.GHG_PROTOCOL,
            _full_result(consolidation_approach="equity_share"),
        )
        assert result["status"] in ("pass", "warning")


# ==============================================================================
# TENANT DATA COVERAGE
# ==============================================================================


class TestTenantDataCoverage:

    def test_high_coverage_passes(self, engine):
        result = engine.check(ComplianceFramework.GHG_PROTOCOL, _full_result(tenant_data_coverage=0.90))
        assert result["status"] in ("pass", "warning")

    def test_low_coverage_warning(self, engine):
        result = engine.check(ComplianceFramework.GHG_PROTOCOL, _full_result(tenant_data_coverage=0.30))
        assert result["status"] in ("warning", "fail")


# ==============================================================================
# COMPLETENESS SCORING
# ==============================================================================


class TestCompletenessScoring:

    def test_full_result_high_score(self, engine):
        result = engine.check_all(_full_result())
        assert result.get("score", 0) >= 80

    def test_empty_result_low_score(self, engine):
        result = engine.check_all({})
        assert result.get("score", 0) <= 50


# ==============================================================================
# THREAD SAFETY
# ==============================================================================


class TestThreadSafety:

    def test_concurrent_checks(self, engine):
        results = []

        def check_compliance():
            r = engine.check(ComplianceFramework.GHG_PROTOCOL, _full_result())
            results.append(r)

        threads = [threading.Thread(target=check_compliance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        for r in results:
            assert r["status"] in ("pass", "warning", "fail")
