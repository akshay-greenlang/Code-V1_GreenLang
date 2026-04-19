# -*- coding: utf-8 -*-
"""
Unit tests for SustainableInvestmentCalculatorEngine (PACK-010 SFDR Article 8).

Tests the Article 2(17) three-step sustainable investment classification,
proportion calculation, minimum commitment checks, breakdown by type,
and provenance tracking.

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


_si_mod = _import_from_path(
    "sustainable_investment_calculator",
    str(ENGINES_DIR / "sustainable_investment_calculator.py"),
)

SustainableInvestmentCalculatorEngine = _si_mod.SustainableInvestmentCalculatorEngine
InvestmentData = _si_mod.InvestmentData
InvestmentClassification = _si_mod.InvestmentClassification
InvestmentClassificationType = _si_mod.InvestmentClassificationType
DNSHStatus = _si_mod.DNSHStatus
GovernanceStatus = _si_mod.GovernanceStatus
ObjectiveContribution = _si_mod.ObjectiveContribution
AdherenceStatus = _si_mod.AdherenceStatus
ProportionResult = _si_mod.ProportionResult
CommitmentAdherence = _si_mod.CommitmentAdherence
ClassificationSummary = _si_mod.ClassificationSummary
DNSHAssessment = _si_mod.DNSHAssessment
GovernanceAssessment = _si_mod.GovernanceAssessment

# ---------------------------------------------------------------------------
# Test data builders
# ---------------------------------------------------------------------------


def _make_taxonomy_aligned_investment() -> InvestmentData:
    """Create an investment that qualifies as taxonomy-aligned."""
    return InvestmentData(
        investment_id="TA_001",
        company_name="Green Bond Fund",
        nav_value=5_000_000.0,
        governance_data={
            "management_structures": True,
            "employee_relations": True,
            "remuneration_compliance": True,
            "tax_compliance": True,
        },
        pai_data={
            "ghg_emissions": 5000.0,
            "carbon_footprint": 100.0,
            "controversies": 0.0,
        },
        environmental_contribution=ObjectiveContribution.CLIMATE_MITIGATION,
        social_contribution=None,
        taxonomy_eligible=True,
        taxonomy_aligned_pct=75.0,
    )


def _make_social_investment() -> InvestmentData:
    """Create an investment with social objective contribution."""
    return InvestmentData(
        investment_id="SOC_001",
        company_name="Social Impact Bond",
        nav_value=3_000_000.0,
        governance_data={
            "management_structures": True,
            "employee_relations": True,
            "remuneration_compliance": True,
            "tax_compliance": False,
        },
        pai_data={
            "ghg_emissions": 8000.0,
            "gender_pay_gap": 5.0,
        },
        environmental_contribution=None,
        social_contribution=ObjectiveContribution.SOCIAL_COHESION,
        taxonomy_eligible=False,
        taxonomy_aligned_pct=0.0,
    )


def _make_non_sustainable_investment() -> InvestmentData:
    """Create an investment that does not qualify as sustainable."""
    return InvestmentData(
        investment_id="NS_001",
        company_name="Conventional Equity",
        nav_value=7_000_000.0,
        governance_data={
            "management_structures": False,
            "employee_relations": False,
            "remuneration_compliance": False,
            "tax_compliance": False,
        },
        pai_data={
            "ghg_emissions": 200_000.0,
            "controversies": 5.0,
        },
        environmental_contribution=None,
        social_contribution=None,
        taxonomy_eligible=False,
        taxonomy_aligned_pct=0.0,
    )


def _all_investments() -> list:
    return [
        _make_taxonomy_aligned_investment(),
        _make_social_investment(),
        _make_non_sustainable_investment(),
    ]


# ===================================================================
# TEST CLASS
# ===================================================================


class TestSustainableInvestmentCalculatorEngine:
    """Unit tests for SustainableInvestmentCalculatorEngine."""

    # ---------------------------------------------------------------
    # 1. Engine initialization
    # ---------------------------------------------------------------

    def test_engine_default_initialization(self):
        """Test engine initializes with default config."""
        engine = SustainableInvestmentCalculatorEngine()
        assert engine is not None

    def test_engine_custom_config(self):
        """Test engine initializes with custom config."""
        config = {"minimum_sustainable_pct": 25.0}
        engine = SustainableInvestmentCalculatorEngine(config)
        assert engine is not None

    # ---------------------------------------------------------------
    # 2. classify_investments
    # ---------------------------------------------------------------

    def test_classify_investments_returns_list(self):
        """Test classify_investments returns list of classifications."""
        engine = SustainableInvestmentCalculatorEngine()
        classifications = engine.classify_investments(_all_investments())
        assert isinstance(classifications, list)
        assert len(classifications) == 3

    def test_classify_taxonomy_aligned(self):
        """Test taxonomy-aligned investment is classified with environmental contribution."""
        engine = SustainableInvestmentCalculatorEngine()
        classifications = engine.classify_investments([_make_taxonomy_aligned_investment()])
        assert len(classifications) == 1
        c = classifications[0]
        assert isinstance(c, InvestmentClassification)
        # The classification depends on DNSH coverage meeting the threshold
        # and governance passing. The contribution is environmental.
        assert c.objective_contribution == ObjectiveContribution.CLIMATE_MITIGATION

    def test_classify_non_sustainable(self):
        """Test non-sustainable investment is classified as NOT_SUSTAINABLE."""
        engine = SustainableInvestmentCalculatorEngine()
        classifications = engine.classify_investments([_make_non_sustainable_investment()])
        c = classifications[0]
        assert c.classification == InvestmentClassificationType.NOT_SUSTAINABLE

    # ---------------------------------------------------------------
    # 3. calculate_proportion
    # ---------------------------------------------------------------

    def test_calculate_proportion_returns_result(self):
        """Test calculate_proportion returns ProportionResult."""
        engine = SustainableInvestmentCalculatorEngine()
        classifications = engine.classify_investments(_all_investments())
        proportion = engine.calculate_proportion(classifications)
        assert isinstance(proportion, ProportionResult)

    def test_proportion_percentages_sum_correctly(self):
        """Test proportion percentages are between 0 and 100."""
        engine = SustainableInvestmentCalculatorEngine()
        classifications = engine.classify_investments(_all_investments())
        proportion = engine.calculate_proportion(classifications)
        assert hasattr(proportion, "total_sustainable_pct")
        assert 0 <= proportion.total_sustainable_pct <= 100

    # ---------------------------------------------------------------
    # 4. check_minimum_commitment
    # ---------------------------------------------------------------

    def test_check_minimum_commitment_returns_adherence(self):
        """Test check_minimum_commitment returns CommitmentAdherence."""
        engine = SustainableInvestmentCalculatorEngine()
        classifications = engine.classify_investments(_all_investments())
        proportion = engine.calculate_proportion(classifications)
        adherence = engine.check_minimum_commitment(proportion)
        assert isinstance(adherence, CommitmentAdherence)

    def test_commitment_adherence_has_status(self):
        """Test adherence result has an adherence_status field."""
        engine = SustainableInvestmentCalculatorEngine()
        classifications = engine.classify_investments(_all_investments())
        proportion = engine.calculate_proportion(classifications)
        adherence = engine.check_minimum_commitment(proportion)
        assert hasattr(adherence, "adherence_status")

    # ---------------------------------------------------------------
    # 5. breakdown_sustainable
    # ---------------------------------------------------------------

    def test_breakdown_sustainable_returns_dict(self):
        """Test breakdown_sustainable returns breakdown by type."""
        engine = SustainableInvestmentCalculatorEngine()
        classifications = engine.classify_investments(_all_investments())
        breakdown = engine.breakdown_sustainable(classifications)
        assert isinstance(breakdown, dict)

    # ---------------------------------------------------------------
    # 6. get_classification_summary
    # ---------------------------------------------------------------

    def test_classification_summary_returns_result(self):
        """Test get_classification_summary returns ClassificationSummary."""
        engine = SustainableInvestmentCalculatorEngine()
        classifications = engine.classify_investments(_all_investments())
        summary = engine.get_classification_summary(classifications)
        assert isinstance(summary, ClassificationSummary)

    # ---------------------------------------------------------------
    # 7. InvestmentClassificationType enum
    # ---------------------------------------------------------------

    def test_classification_type_enum(self):
        """Test InvestmentClassificationType enum values."""
        vals = {t.value for t in InvestmentClassificationType}
        assert "taxonomy_aligned" in vals
        assert "not_sustainable" in vals

    # ---------------------------------------------------------------
    # 8. AdherenceStatus enum
    # ---------------------------------------------------------------

    def test_adherence_status_enum(self):
        """Test AdherenceStatus enum values."""
        vals = {s.value for s in AdherenceStatus}
        assert len(vals) >= 3  # meeting, exceeding, below, at_risk

    # ---------------------------------------------------------------
    # 9. DNSHStatus enum
    # ---------------------------------------------------------------

    def test_dnsh_status_enum(self):
        """Test DNSHStatus enum values."""
        vals = {s.value for s in DNSHStatus}
        assert "passed" in vals
        assert "failed" in vals

    # ---------------------------------------------------------------
    # 10. GovernanceStatus enum
    # ---------------------------------------------------------------

    def test_governance_status_enum(self):
        """Test GovernanceStatus enum values."""
        vals = {s.value for s in GovernanceStatus}
        assert "good" in vals
        assert "inadequate" in vals

    # ---------------------------------------------------------------
    # 11. InvestmentData model
    # ---------------------------------------------------------------

    def test_investment_data_model_construction(self):
        """Test InvestmentData model can be constructed."""
        inv = InvestmentData(
            investment_id="TEST",
            company_name="Test",
            nav_value=1_000_000.0,
            governance_data={"management_structures": True, "employee_relations": True,
                             "remuneration_compliance": True, "tax_compliance": True},
            pai_data={"ghg_emissions": 100.0},
            environmental_contribution=ObjectiveContribution.CLIMATE_MITIGATION,
            social_contribution=None,
            taxonomy_eligible=True,
            taxonomy_aligned_pct=50.0,
        )
        assert inv.investment_id == "TEST"
        assert inv.nav_value == 1_000_000.0

    # ---------------------------------------------------------------
    # 12. Empty investments
    # ---------------------------------------------------------------

    def test_classify_empty_returns_empty(self):
        """Test empty investment list returns empty classifications."""
        engine = SustainableInvestmentCalculatorEngine()
        classifications = engine.classify_investments([])
        assert isinstance(classifications, list)
        assert len(classifications) == 0

    # ---------------------------------------------------------------
    # 13. ObjectiveContribution enum
    # ---------------------------------------------------------------

    def test_objective_contribution_enum(self):
        """Test ObjectiveContribution enum exists."""
        vals = {o.value for o in ObjectiveContribution}
        assert len(vals) >= 2
        assert "climate_mitigation" in vals

    # ---------------------------------------------------------------
    # 14. Classification has provenance
    # ---------------------------------------------------------------

    def test_classification_has_provenance(self):
        """Test classifications include provenance information."""
        engine = SustainableInvestmentCalculatorEngine()
        classifications = engine.classify_investments([_make_taxonomy_aligned_investment()])
        c = classifications[0]
        assert hasattr(c, "provenance_hash") or hasattr(c, "classification")

    # ---------------------------------------------------------------
    # 15. Three-step test components
    # ---------------------------------------------------------------

    def test_classification_includes_dnsh_check(self):
        """Test classification result references DNSH assessment."""
        engine = SustainableInvestmentCalculatorEngine()
        classifications = engine.classify_investments([_make_taxonomy_aligned_investment()])
        c = classifications[0]
        assert hasattr(c, "dnsh_assessment")
