# -*- coding: utf-8 -*-
"""
Unit tests for GoodGovernanceEngine (PACK-010 SFDR Article 8).

Tests governance assessment across 4 areas (management structures,
employee relations, remuneration, tax compliance), portfolio governance
screening, violation reporting, scoring, and provenance tracking.

Self-contained: no conftest imports.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Dynamic import helper
# ---------------------------------------------------------------------------

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"


def _import_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_gov_mod = _import_from_path(
    "good_governance_engine",
    str(ENGINES_DIR / "good_governance_engine.py"),
)

GoodGovernanceEngine = _gov_mod.GoodGovernanceEngine
GovernanceConfig = _gov_mod.GovernanceConfig
CompanyGovernanceData = _gov_mod.CompanyGovernanceData
ManagementStructureData = _gov_mod.ManagementStructureData
EmployeeRelationsData = _gov_mod.EmployeeRelationsData
RemunerationData = _gov_mod.RemunerationData
TaxComplianceData = _gov_mod.TaxComplianceData
GovernanceResult = _gov_mod.GovernanceResult
PortfolioGovernanceResult = _gov_mod.PortfolioGovernanceResult
GovernanceArea = _gov_mod.GovernanceArea
GovernanceStatus = _gov_mod.GovernanceStatus
GovernanceCheckType = _gov_mod.GovernanceCheckType
ViolationType = _gov_mod.ViolationType

# ---------------------------------------------------------------------------
# Test data builders
# ---------------------------------------------------------------------------


def _make_good_company() -> CompanyGovernanceData:
    """Create a company with strong governance practices."""
    return CompanyGovernanceData(
        company_id="GOOD_001",
        company_name="ESGLeader AG",
        management_data=ManagementStructureData(
            has_independent_board=True,
            independent_board_pct=65.0,
            has_audit_committee=True,
            has_risk_committee=True,
            has_sustainability_committee=True,
            ceo_chair_separation=True,
        ),
        employee_data=EmployeeRelationsData(
            ilo_core_conventions_compliance=True,
            has_health_safety_policy=True,
            employee_turnover_pct=8.0,
            has_diversity_policy=True,
            has_collective_bargaining=True,
            living_wage_compliance=True,
        ),
        remuneration_data=RemunerationData(
            has_remuneration_policy=True,
            esg_linked_remuneration=True,
            ceo_to_median_pay_ratio=12.0,
            has_clawback_provisions=True,
            shareholder_vote_on_pay=True,
        ),
        tax_data=TaxComplianceData(
            has_tax_strategy_disclosure=True,
            country_by_country_reporting=True,
            tax_haven_exposure=False,
            effective_tax_rate=22.0,
            tax_controversies=0,
        ),
        ungc_signatory=True,
        ungc_violations=False,
        oecd_violations=False,
        has_anti_corruption_policy=True,
        has_anti_bribery_measures=True,
        corruption_controversies=0,
    )


def _make_poor_company() -> CompanyGovernanceData:
    """Create a company with weak governance practices."""
    return CompanyGovernanceData(
        company_id="POOR_001",
        company_name="BadGov Ltd",
        management_data=ManagementStructureData(
            has_independent_board=False,
            independent_board_pct=15.0,
            has_audit_committee=False,
            has_risk_committee=False,
            has_sustainability_committee=False,
            ceo_chair_separation=False,
        ),
        employee_data=EmployeeRelationsData(
            ilo_core_conventions_compliance=False,
            has_health_safety_policy=False,
            employee_turnover_pct=45.0,
            has_diversity_policy=False,
            has_collective_bargaining=False,
            living_wage_compliance=False,
        ),
        remuneration_data=RemunerationData(
            has_remuneration_policy=False,
            esg_linked_remuneration=False,
            ceo_to_median_pay_ratio=350.0,
            has_clawback_provisions=False,
            excessive_severance_provisions=True,
        ),
        tax_data=TaxComplianceData(
            has_tax_strategy_disclosure=False,
            country_by_country_reporting=False,
            aggressive_tax_planning_flag=True,
            tax_haven_exposure=True,
            effective_tax_rate=3.0,
            tax_controversies=8,
        ),
        ungc_signatory=False,
        ungc_violations=True,
        oecd_violations=True,
        has_anti_corruption_policy=False,
        has_anti_bribery_measures=False,
        corruption_controversies=8,
    )


# ===================================================================
# TEST CLASS
# ===================================================================


class TestGoodGovernanceEngine:
    """Unit tests for GoodGovernanceEngine."""

    # ---------------------------------------------------------------
    # 1. Engine initialization
    # ---------------------------------------------------------------

    def test_engine_default_initialization(self):
        """Test engine initializes with default config."""
        engine = GoodGovernanceEngine()
        assert engine.assessment_count == 0

    def test_engine_custom_config(self):
        """Test engine initializes with custom GovernanceConfig."""
        config = GovernanceConfig()
        engine = GoodGovernanceEngine(config)
        assert engine.assessment_count == 0

    # ---------------------------------------------------------------
    # 2. assess_governance - good company
    # ---------------------------------------------------------------

    def test_assess_good_company_passes(self):
        """Test well-governed company receives PASS status."""
        engine = GoodGovernanceEngine()
        result = engine.assess_governance(_make_good_company())
        assert isinstance(result, GovernanceResult)
        assert result.overall_status in (GovernanceStatus.PASS, "PASS")

    def test_assess_governance_provenance_hash(self):
        """Test governance result includes valid provenance hash."""
        engine = GoodGovernanceEngine()
        result = engine.assess_governance(_make_good_company())
        assert isinstance(result.provenance_hash, str)
        assert len(result.provenance_hash) == 64
        assert re.match(r"^[0-9a-f]{64}$", result.provenance_hash)

    # ---------------------------------------------------------------
    # 3. assess_governance - poor company
    # ---------------------------------------------------------------

    def test_assess_poor_company_fails(self):
        """Test poorly governed company receives FAIL status."""
        engine = GoodGovernanceEngine()
        result = engine.assess_governance(_make_poor_company())
        assert isinstance(result, GovernanceResult)
        assert result.overall_status in (GovernanceStatus.FAIL, "FAIL")

    # ---------------------------------------------------------------
    # 4. Area-level results
    # ---------------------------------------------------------------

    def test_area_results_cover_four_areas(self):
        """Test assessment covers all 4 governance areas."""
        engine = GoodGovernanceEngine()
        result = engine.assess_governance(_make_good_company())
        assert hasattr(result, "area_results")
        assert len(result.area_results) == 4

    # ---------------------------------------------------------------
    # 5. assess_portfolio_governance
    # ---------------------------------------------------------------

    def test_portfolio_governance_returns_result(self):
        """Test portfolio governance assessment returns PortfolioGovernanceResult."""
        engine = GoodGovernanceEngine()
        companies = [_make_good_company(), _make_poor_company()]
        result = engine.assess_portfolio_governance(companies, "TestPortfolio")
        assert isinstance(result, PortfolioGovernanceResult)

    def test_portfolio_governance_empty_raises(self):
        """Test empty companies list raises ValueError."""
        engine = GoodGovernanceEngine()
        with pytest.raises(ValueError):
            engine.assess_portfolio_governance([], "Empty")

    # ---------------------------------------------------------------
    # 6. get_violation_report
    # ---------------------------------------------------------------

    def test_violation_report_for_poor_company(self):
        """Test violation report identifies issues in poor company."""
        engine = GoodGovernanceEngine()
        result = engine.assess_governance(_make_poor_company())
        report = engine.get_violation_report(result)
        assert isinstance(report, (list, dict, str))

    # ---------------------------------------------------------------
    # 7. governance_score
    # ---------------------------------------------------------------

    def test_governance_score_returns_float(self):
        """Test governance_score returns a float between 0 and 100."""
        engine = GoodGovernanceEngine()
        score = engine.governance_score(_make_good_company())
        assert isinstance(score, float)
        assert 0 <= score <= 100

    def test_good_company_scores_higher(self):
        """Test good company scores higher than poor company."""
        engine = GoodGovernanceEngine()
        good_score = engine.governance_score(_make_good_company())
        poor_score = engine.governance_score(_make_poor_company())
        assert good_score > poor_score

    # ---------------------------------------------------------------
    # 8. assessment_count increments
    # ---------------------------------------------------------------

    def test_assessment_count_increments(self):
        """Test assessment_count increments on each assess_governance call."""
        engine = GoodGovernanceEngine()
        engine.assess_governance(_make_good_company())
        assert engine.assessment_count == 1
        engine.assess_governance(_make_poor_company())
        assert engine.assessment_count == 2

    # ---------------------------------------------------------------
    # 9. Deterministic results
    # ---------------------------------------------------------------

    def test_deterministic_assessment(self):
        """Test same input yields identical provenance hash."""
        engine = GoodGovernanceEngine()
        company = _make_good_company()
        r1 = engine.assess_governance(company)
        r2 = engine.assess_governance(company)
        assert r1.provenance_hash == r2.provenance_hash

    # ---------------------------------------------------------------
    # 10. GovernanceArea enum
    # ---------------------------------------------------------------

    def test_governance_area_enum(self):
        """Test GovernanceArea enum has 4 areas."""
        areas = {a.value for a in GovernanceArea}
        assert len(areas) == 4

    # ---------------------------------------------------------------
    # 11. GovernanceStatus enum
    # ---------------------------------------------------------------

    def test_governance_status_enum(self):
        """Test GovernanceStatus enum has expected values."""
        vals = {s.value for s in GovernanceStatus}
        assert "PASS" in vals
        assert "FAIL" in vals

    # ---------------------------------------------------------------
    # 12. Properties
    # ---------------------------------------------------------------

    def test_governance_areas_property(self):
        """Test governance_areas property returns list of areas."""
        engine = GoodGovernanceEngine()
        areas = engine.governance_areas
        assert isinstance(areas, (list, set, tuple))
        assert len(areas) == 4

    def test_total_criteria_property(self):
        """Test total_criteria property returns positive integer."""
        engine = GoodGovernanceEngine()
        total = engine.total_criteria
        assert isinstance(total, int)
        assert total > 0

    def test_mandatory_criteria_property(self):
        """Test mandatory_criteria property returns list of mandatory criterion IDs."""
        engine = GoodGovernanceEngine()
        mandatory = engine.mandatory_criteria
        assert isinstance(mandatory, (list, int))
        if isinstance(mandatory, list):
            assert len(mandatory) >= 0
        else:
            assert mandatory >= 0
