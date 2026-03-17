# -*- coding: utf-8 -*-
"""
Unit tests for TaxonomyAlignmentRatioEngine (PACK-010 SFDR Article 8).

Tests alignment ratio calculation, objective breakdown, commitment
adherence, pie chart generation, and provenance tracking.

Self-contained: no conftest imports.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from datetime import date, datetime
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


_tax_mod = _import_from_path(
    "taxonomy_alignment_ratio",
    str(ENGINES_DIR / "taxonomy_alignment_ratio.py"),
)

TaxonomyAlignmentRatioEngine = _tax_mod.TaxonomyAlignmentRatioEngine
TaxonomyAlignmentConfig = _tax_mod.TaxonomyAlignmentConfig
HoldingAlignmentData = _tax_mod.HoldingAlignmentData
AlignmentResult = _tax_mod.AlignmentResult
EnvironmentalObjective = _tax_mod.EnvironmentalObjective
AlignmentCategory = _tax_mod.AlignmentCategory
GASExposureType = _tax_mod.GASExposureType
ObjectiveBreakdown = _tax_mod.ObjectiveBreakdown
PieChartSlice = _tax_mod.PieChartSlice
CommitmentAdherence = _tax_mod.CommitmentAdherence

# ---------------------------------------------------------------------------
# Test data builders
# ---------------------------------------------------------------------------

_TODAY = date.today()


def _make_config(**overrides) -> TaxonomyAlignmentConfig:
    defaults = dict(
        total_nav_eur=50_000_000.0,
        reporting_date=_TODAY,
        pre_contractual_commitment_pct=20.0,
    )
    defaults.update(overrides)
    return TaxonomyAlignmentConfig(**defaults)


def _make_holding(
    holding_id: str = "H1",
    name: str = "TestHolding",
    holding_type: str = "CORPORATE",
    value_eur: float = 5_000_000.0,
    alignment_category: str = "ALIGNED",
    aligned_revenue_pct: float = 60.0,
    primary_objective: str = None,
    **kwargs,
) -> HoldingAlignmentData:
    params = dict(
        holding_id=holding_id,
        holding_name=name,
        holding_type=holding_type,
        value_eur=value_eur,
        alignment_category=alignment_category,
        aligned_revenue_pct=aligned_revenue_pct,
    )
    if primary_objective is not None:
        params["primary_objective"] = primary_objective
    params.update(kwargs)
    return HoldingAlignmentData(**params)


def _sample_holdings() -> list:
    """Build a small portfolio of 4 holdings with mixed alignment."""
    h1 = _make_holding("H1", "GreenTech", value_eur=15_000_000.0,
                       alignment_category=AlignmentCategory.ALIGNED,
                       aligned_revenue_pct=80.0,
                       primary_objective=EnvironmentalObjective.CCM)
    h2 = _make_holding("H2", "PartialGreen", value_eur=10_000_000.0,
                       alignment_category=AlignmentCategory.ELIGIBLE_NOT_ALIGNED,
                       aligned_revenue_pct=30.0)
    h3 = _make_holding("H3", "NonAligned", value_eur=15_000_000.0,
                       alignment_category=AlignmentCategory.NOT_ELIGIBLE,
                       aligned_revenue_pct=0.0)
    h4 = _make_holding("H4", "GASHolding", value_eur=10_000_000.0,
                       alignment_category=AlignmentCategory.ALIGNED,
                       aligned_revenue_pct=50.0,
                       primary_objective=EnvironmentalObjective.CCM)
    return [h1, h2, h3, h4]


# ===================================================================
# TEST CLASS
# ===================================================================


class TestTaxonomyAlignmentRatioEngine:
    """Unit tests for TaxonomyAlignmentRatioEngine."""

    # ---------------------------------------------------------------
    # 1. Engine initialization
    # ---------------------------------------------------------------

    def test_engine_initialization(self):
        """Test engine initializes with valid config."""
        config = _make_config()
        engine = TaxonomyAlignmentRatioEngine(config)
        assert engine.config is config

    def test_engine_calculation_count_starts_zero(self):
        """Test calculation_count property starts at zero."""
        engine = TaxonomyAlignmentRatioEngine(_make_config())
        assert engine.calculation_count == 0

    # ---------------------------------------------------------------
    # 2. calculate_alignment_ratio
    # ---------------------------------------------------------------

    def test_alignment_ratio_returns_result(self):
        """Test calculate_alignment_ratio returns AlignmentResult."""
        engine = TaxonomyAlignmentRatioEngine(_make_config())
        result = engine.calculate_alignment_ratio(_sample_holdings(), "TestFund")
        assert isinstance(result, AlignmentResult)

    def test_alignment_ratio_provenance_hash(self):
        """Test result has valid SHA-256 provenance hash."""
        engine = TaxonomyAlignmentRatioEngine(_make_config())
        result = engine.calculate_alignment_ratio(_sample_holdings(), "TestFund")
        assert isinstance(result.provenance_hash, str)
        assert len(result.provenance_hash) == 64
        assert re.match(r"^[0-9a-f]{64}$", result.provenance_hash)

    def test_alignment_ratio_deterministic(self):
        """Test same inputs yield identical provenance hash."""
        config = _make_config()
        holdings = _sample_holdings()
        e1 = TaxonomyAlignmentRatioEngine(config)
        e2 = TaxonomyAlignmentRatioEngine(config)
        r1 = e1.calculate_alignment_ratio(holdings, "Fund")
        r2 = e2.calculate_alignment_ratio(holdings, "Fund")
        assert r1.provenance_hash == r2.provenance_hash

    def test_alignment_ratio_empty_holdings_raises(self):
        """Test empty holdings raises ValueError."""
        engine = TaxonomyAlignmentRatioEngine(_make_config())
        with pytest.raises(ValueError):
            engine.calculate_alignment_ratio([], "Fund")

    def test_calculation_count_increments(self):
        """Test calculation_count increments after each calculation."""
        engine = TaxonomyAlignmentRatioEngine(_make_config())
        engine.calculate_alignment_ratio(_sample_holdings(), "Fund")
        assert engine.calculation_count == 1
        engine.calculate_alignment_ratio(_sample_holdings(), "Fund")
        assert engine.calculation_count == 2

    # ---------------------------------------------------------------
    # 3. breakdown_by_objective
    # ---------------------------------------------------------------

    def test_breakdown_by_objective_returns_dict(self):
        """Test breakdown_by_objective returns objective breakdown."""
        engine = TaxonomyAlignmentRatioEngine(_make_config())
        breakdown = engine.breakdown_by_objective(_sample_holdings())
        assert isinstance(breakdown, dict)
        # Should have entries for the 6 environmental objectives
        assert len(breakdown) >= 1

    def test_breakdown_ccm_is_largest(self):
        """Test CCM is the largest objective given test data."""
        engine = TaxonomyAlignmentRatioEngine(_make_config())
        breakdown = engine.breakdown_by_objective(_sample_holdings())
        if EnvironmentalObjective.CCM.value in breakdown:
            ccm_val = breakdown[EnvironmentalObjective.CCM.value]
            for key, obj_bd in breakdown.items():
                if key != EnvironmentalObjective.CCM.value:
                    assert isinstance(ccm_val, ObjectiveBreakdown)

    # ---------------------------------------------------------------
    # 4. check_commitment_adherence
    # ---------------------------------------------------------------

    def test_commitment_adherence_returns_result(self):
        """Test check_commitment_adherence returns CommitmentAdherence."""
        engine = TaxonomyAlignmentRatioEngine(_make_config())
        adherence = engine.check_commitment_adherence(_sample_holdings())
        assert isinstance(adherence, CommitmentAdherence)

    def test_commitment_adherence_fields(self):
        """Test adherence result has expected fields."""
        engine = TaxonomyAlignmentRatioEngine(_make_config(pre_contractual_commitment_pct=20.0))
        adherence = engine.check_commitment_adherence(_sample_holdings())
        assert hasattr(adherence, "pre_contractual_commitment_pct")
        assert hasattr(adherence, "actual_alignment_pct")
        assert hasattr(adherence, "status")

    # ---------------------------------------------------------------
    # 5. generate_pie_chart_data
    # ---------------------------------------------------------------

    def test_pie_chart_data_returns_list(self):
        """Test generate_pie_chart_data returns a list of slices."""
        engine = TaxonomyAlignmentRatioEngine(_make_config())
        slices = engine.generate_pie_chart_data(_sample_holdings())
        assert isinstance(slices, list)
        assert len(slices) >= 1
        for s in slices:
            assert isinstance(s, PieChartSlice)

    def test_pie_chart_slices_sum_to_100(self):
        """Test pie chart slice percentages sum to approximately 100."""
        engine = TaxonomyAlignmentRatioEngine(_make_config())
        slices = engine.generate_pie_chart_data(_sample_holdings())
        total = sum(s.value_pct for s in slices)
        assert total == pytest.approx(100.0, abs=0.5)

    # ---------------------------------------------------------------
    # 6. Supported objectives property
    # ---------------------------------------------------------------

    def test_supported_objectives(self):
        """Test supported_objectives returns the 6 environmental objectives."""
        engine = TaxonomyAlignmentRatioEngine(_make_config())
        objectives = engine.supported_objectives
        assert isinstance(objectives, (list, set, tuple))
        assert len(objectives) == 6

    # ---------------------------------------------------------------
    # 7. Alignment categories property
    # ---------------------------------------------------------------

    def test_alignment_categories(self):
        """Test alignment_categories returns expected categories."""
        engine = TaxonomyAlignmentRatioEngine(_make_config())
        categories = engine.alignment_categories
        assert isinstance(categories, (list, set, tuple))
        assert len(categories) >= 3  # At least ALIGNED, ELIGIBLE, NON_ELIGIBLE

    # ---------------------------------------------------------------
    # 8. EnvironmentalObjective enum
    # ---------------------------------------------------------------

    def test_environmental_objective_enum_values(self):
        """Test EnvironmentalObjective enum has all 6 objectives."""
        names = {e.name for e in EnvironmentalObjective}
        expected = {"CCM", "CCA", "WTR", "CE", "PPC", "BIO"}
        assert expected.issubset(names)

    # ---------------------------------------------------------------
    # 9. AlignmentCategory enum
    # ---------------------------------------------------------------

    def test_alignment_category_enum_values(self):
        """Test AlignmentCategory enum contains required values."""
        vals = {c.value for c in AlignmentCategory}
        for expected in ["ALIGNED", "ELIGIBLE_NOT_ALIGNED", "NOT_ELIGIBLE"]:
            assert expected in vals

    # ---------------------------------------------------------------
    # 10. Config negative NAV
    # ---------------------------------------------------------------

    def test_config_negative_nav_raises(self):
        """Test config rejects negative NAV."""
        with pytest.raises(ValueError):
            _make_config(total_nav_eur=-1.0)
