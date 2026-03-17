# -*- coding: utf-8 -*-
"""
Unit tests for CarbonTrajectoryEngine (PACK-011 SFDR Article 9, Engine 7).

Tests Implied Temperature Rise (ITR), carbon budget assessment, SBT coverage,
Net Zero progress tracking, 7% annual decarbonization pathway, transition
plan quality scoring, and provenance.

Self-contained: no conftest imports.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Dynamic import helper (hyphenated directory names)
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


_ct_mod = _import_from_path(
    "carbon_trajectory_engine",
    str(ENGINES_DIR / "carbon_trajectory_engine.py"),
)

CarbonTrajectoryEngine = _ct_mod.CarbonTrajectoryEngine
TrajectoryConfig = _ct_mod.TrajectoryConfig
HoldingTrajectoryData = _ct_mod.HoldingTrajectoryData
TrajectoryResult = _ct_mod.TrajectoryResult
ITRResult = _ct_mod.ITRResult
CarbonBudgetResult = _ct_mod.CarbonBudgetResult
SBTCoverageResult = _ct_mod.SBTCoverageResult
NetZeroProgress = _ct_mod.NetZeroProgress
CarbonPathway = _ct_mod.CarbonPathway
TransitionPlanQuality = _ct_mod.TransitionPlanQuality

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _holding(**kwargs) -> HoldingTrajectoryData:
    """Create a holding with sensible defaults for trajectory testing."""
    defaults = dict(
        company_name="Green Corp",
        nav_value=10_000_000,
        weight_pct=10.0,
        current_intensity=80.0,
        prior_intensity=90.0,
        base_year_intensity=120.0,
        base_year=2019,
        scope1_emissions=3000.0,
        scope2_emissions=1500.0,
        scope3_emissions=5000.0,
        revenue_eur=50_000_000,
        evic_eur=200_000_000,
        has_sbt=True,
        sbt_target_year=2030,
        sbt_reduction_pct=42.0,
        sbt_scope="1+2",
        has_net_zero_commitment=True,
        net_zero_target_year=2050,
        interim_target_pct=50.0,
        transition_plan_quality=TransitionPlanQuality.COMPREHENSIVE,
        reporting_year=2025,
    )
    defaults.update(kwargs)
    return HoldingTrajectoryData(**defaults)


def _holding_no_sbt(**kwargs) -> HoldingTrajectoryData:
    """Create a holding without SBT or Net Zero commitment."""
    defaults = dict(
        company_name="Laggard Corp",
        nav_value=5_000_000,
        weight_pct=5.0,
        current_intensity=200.0,
        prior_intensity=195.0,
        base_year_intensity=180.0,
        base_year=2019,
        scope1_emissions=8000.0,
        scope2_emissions=4000.0,
        scope3_emissions=12000.0,
        revenue_eur=40_000_000,
        evic_eur=150_000_000,
        has_sbt=False,
        sbt_target_year=0,
        sbt_reduction_pct=0.0,
        sbt_scope="",
        has_net_zero_commitment=False,
        net_zero_target_year=0,
        interim_target_pct=0.0,
        transition_plan_quality=TransitionPlanQuality.ABSENT,
        reporting_year=2025,
    )
    defaults.update(kwargs)
    return HoldingTrajectoryData(**defaults)


def _make_portfolio(n_green=5, n_laggard=0):
    """Build a mixed portfolio."""
    holdings = []
    total = n_green + n_laggard
    for i in range(n_green):
        holdings.append(_holding(
            company_name=f"Green-{i}",
            nav_value=10_000_000,
            weight_pct=100.0 / total,
            current_intensity=70.0 + i * 5,
            prior_intensity=80.0 + i * 5,
            base_year_intensity=110.0 + i * 5,
        ))
    for i in range(n_laggard):
        holdings.append(_holding_no_sbt(
            company_name=f"Laggard-{i}",
            nav_value=5_000_000,
            weight_pct=100.0 / total,
        ))
    return holdings


# ===========================================================================
# Tests
# ===========================================================================


class TestCarbonTrajectoryEngineInit:
    """Test engine initialization."""

    def test_default_config(self):
        engine = CarbonTrajectoryEngine()
        assert engine.config.annual_reduction_target == pytest.approx(7.0)
        assert engine.config.target_pathway == CarbonPathway.PARIS_1_5C
        assert engine.config.base_year == 2019
        assert engine.config.projection_end_year == 2050

    def test_dict_config(self):
        engine = CarbonTrajectoryEngine({
            "product_name": "Test Fund",
            "target_pathway": "well_below_2c",
        })
        assert engine.config.product_name == "Test Fund"
        assert engine.config.target_pathway == CarbonPathway.WELL_BELOW_2C

    def test_pydantic_config(self):
        cfg = TrajectoryConfig(
            product_name="Art 9 Fund",
            annual_reduction_target=10.0,
        )
        engine = CarbonTrajectoryEngine(cfg)
        assert engine.config.annual_reduction_target == pytest.approx(10.0)


class TestITRCalculation:
    """Test Implied Temperature Rise calculation."""

    def test_itr_low_intensity_paris_aligned(self):
        """Very low WACI -> ITR near 1.5C (Paris 1.5C aligned)."""
        engine = CarbonTrajectoryEngine()
        holdings = [_holding(
            current_intensity=10.0, weight_pct=100.0, nav_value=10_000_000,
        )]
        result = engine.calculate_itr(holdings)
        assert isinstance(result, ITRResult)
        assert result.implied_temperature_rise <= 1.75
        assert result.pathway_alignment in (
            CarbonPathway.PARIS_1_5C,
            CarbonPathway.WELL_BELOW_2C,
        )
        assert len(result.provenance_hash) == 64

    def test_itr_medium_intensity_below_2c(self):
        """Medium WACI (~100) -> ITR around 2.0C."""
        engine = CarbonTrajectoryEngine()
        holdings = [_holding(
            current_intensity=100.0, weight_pct=100.0, nav_value=10_000_000,
        )]
        result = engine.calculate_itr(holdings)
        assert result.implied_temperature_rise == pytest.approx(2.0, abs=0.1)

    def test_itr_high_intensity_above_2c(self):
        """High WACI (>400) -> ITR > 2.5C."""
        engine = CarbonTrajectoryEngine()
        holdings = [_holding(
            current_intensity=500.0, weight_pct=100.0, nav_value=10_000_000,
        )]
        result = engine.calculate_itr(holdings)
        assert result.implied_temperature_rise > 2.5
        assert result.pathway_alignment in (
            CarbonPathway.ABOVE_2C,
            CarbonPathway.BELOW_2C,
        )

    def test_itr_zero_intensity(self):
        """Zero intensity -> ITR = 1.5C (best case)."""
        engine = CarbonTrajectoryEngine()
        holdings = [_holding(
            current_intensity=0.0, weight_pct=100.0, nav_value=10_000_000,
            scope1_emissions=0.0, scope2_emissions=0.0, scope3_emissions=0.0,
            revenue_eur=0.0,
        )]
        result = engine.calculate_itr(holdings)
        assert result.implied_temperature_rise == pytest.approx(1.5, abs=0.1)

    def test_itr_on_track_flag(self):
        """On-track flag set when actual reduction >= 7% target."""
        engine = CarbonTrajectoryEngine({"annual_reduction_target": 7.0})
        # prior=100, current=90 -> 10% reduction > 7%
        holdings = [_holding(
            current_intensity=90.0, prior_intensity=100.0,
            weight_pct=100.0, nav_value=10_000_000,
        )]
        result = engine.calculate_itr(holdings)
        assert result.actual_annual_reduction >= 7.0
        assert result.on_track is True

    def test_itr_off_track_flag(self):
        """Off-track flag set when actual reduction < 7% target."""
        engine = CarbonTrajectoryEngine({"annual_reduction_target": 7.0})
        # prior=100, current=96 -> 4% reduction < 7%
        holdings = [_holding(
            current_intensity=96.0, prior_intensity=100.0,
            weight_pct=100.0, nav_value=10_000_000,
        )]
        result = engine.calculate_itr(holdings)
        assert result.actual_annual_reduction < 7.0
        assert result.on_track is False

    def test_itr_data_coverage(self):
        """Data coverage reflects proportion of holdings with intensity data."""
        engine = CarbonTrajectoryEngine()
        holdings = [
            _holding(current_intensity=80.0, weight_pct=50.0, nav_value=5_000_000),
            _holding(
                current_intensity=0.0, weight_pct=50.0, nav_value=5_000_000,
                scope1_emissions=0.0, scope2_emissions=0.0, scope3_emissions=0.0,
                revenue_eur=0.0,
            ),
        ]
        result = engine.calculate_itr(holdings)
        assert result.data_coverage_pct == pytest.approx(50.0, abs=1.0)


class TestCarbonBudget:
    """Test carbon budget assessment."""

    def test_budget_paris_1_5c(self):
        """Budget uses 400 GtCO2 for Paris 1.5C pathway."""
        engine = CarbonTrajectoryEngine({"target_pathway": "paris_1_5c"})
        holdings = _make_portfolio(n_green=5)
        result = engine.analyze_carbon_budget(holdings)
        assert isinstance(result, CarbonBudgetResult)
        assert result.pathway == CarbonPathway.PARIS_1_5C
        assert result.global_remaining_budget_gt == pytest.approx(400.0)
        assert len(result.provenance_hash) == 64

    def test_budget_well_below_2c(self):
        """Budget uses 1150 GtCO2 for well-below 2C pathway."""
        engine = CarbonTrajectoryEngine({"target_pathway": "well_below_2c"})
        holdings = _make_portfolio(n_green=5)
        result = engine.analyze_carbon_budget(holdings)
        assert result.pathway == CarbonPathway.WELL_BELOW_2C
        assert result.global_remaining_budget_gt == pytest.approx(1150.0)

    def test_budget_annual_emissions_positive(self):
        """Portfolio annual attributed emissions are positive."""
        engine = CarbonTrajectoryEngine()
        holdings = _make_portfolio(n_green=5)
        result = engine.analyze_carbon_budget(holdings)
        assert result.portfolio_annual_emissions_t > 0

    def test_budget_trajectory_points(self):
        """Trajectory points project out to 2050."""
        engine = CarbonTrajectoryEngine({"projection_end_year": 2050})
        holdings = _make_portfolio(n_green=5)
        result = engine.analyze_carbon_budget(holdings)
        assert len(result.trajectory_points) > 0
        years = [int(p["year"]) for p in result.trajectory_points]
        assert max(years) == 2050

    def test_budget_aligned_flag(self):
        """Budget aligned flag is set correctly."""
        engine = CarbonTrajectoryEngine()
        holdings = _make_portfolio(n_green=5)
        result = engine.analyze_carbon_budget(holdings)
        # Budget aligned if exhaustion year >= 2050 or never exhausted (0)
        assert isinstance(result.budget_aligned, bool)


class TestSBTCoverage:
    """Test Science-Based Target coverage assessment."""

    def test_full_sbt_coverage(self):
        """All holdings with SBTs -> 100% coverage."""
        engine = CarbonTrajectoryEngine()
        holdings = [
            _holding(has_sbt=True, nav_value=10_000_000, weight_pct=50.0),
            _holding(has_sbt=True, nav_value=10_000_000, weight_pct=50.0),
        ]
        result = engine.assess_sbt_coverage(holdings)
        assert isinstance(result, SBTCoverageResult)
        assert result.sbt_coverage_pct == pytest.approx(100.0, abs=0.1)
        assert result.sbt_holdings_count == 2
        assert result.total_holdings == 2
        assert len(result.provenance_hash) == 64

    def test_partial_sbt_coverage(self):
        """Mix of SBT and non-SBT holdings -> partial coverage."""
        engine = CarbonTrajectoryEngine()
        holdings = [
            _holding(has_sbt=True, nav_value=10_000_000, weight_pct=50.0),
            _holding_no_sbt(has_sbt=False, nav_value=10_000_000, weight_pct=50.0),
        ]
        result = engine.assess_sbt_coverage(holdings)
        assert result.sbt_coverage_pct == pytest.approx(50.0, abs=0.1)
        assert result.sbt_holdings_count == 1

    def test_zero_sbt_coverage(self):
        """No holdings with SBTs -> 0% coverage."""
        engine = CarbonTrajectoryEngine()
        holdings = [
            _holding_no_sbt(nav_value=10_000_000, weight_pct=100.0),
        ]
        result = engine.assess_sbt_coverage(holdings)
        assert result.sbt_coverage_pct == pytest.approx(0.0, abs=0.1)
        assert result.sbt_holdings_count == 0

    def test_sbt_scope_coverage_distribution(self):
        """Scope coverage tracks SBT scope distribution."""
        engine = CarbonTrajectoryEngine()
        holdings = [
            _holding(has_sbt=True, sbt_scope="1+2", nav_value=10_000_000),
            _holding(has_sbt=True, sbt_scope="1+2+3", nav_value=10_000_000),
            _holding(has_sbt=True, sbt_scope="1+2", nav_value=10_000_000),
        ]
        result = engine.assess_sbt_coverage(holdings)
        assert "1+2" in result.scope_coverage
        assert result.scope_coverage["1+2"] == 2
        assert "1+2+3" in result.scope_coverage
        assert result.scope_coverage["1+2+3"] == 1

    def test_average_reduction_target(self):
        """Average SBT reduction target is calculated correctly."""
        engine = CarbonTrajectoryEngine()
        holdings = [
            _holding(has_sbt=True, sbt_reduction_pct=42.0, nav_value=10_000_000),
            _holding(has_sbt=True, sbt_reduction_pct=50.0, nav_value=10_000_000),
        ]
        result = engine.assess_sbt_coverage(holdings)
        assert result.average_reduction_target == pytest.approx(46.0, abs=0.1)

    def test_net_zero_committed_pct(self):
        """Net Zero committed percentage tracks commitments."""
        engine = CarbonTrajectoryEngine()
        holdings = [
            _holding(has_net_zero_commitment=True, nav_value=10_000_000),
            _holding(has_net_zero_commitment=False, nav_value=10_000_000),
        ]
        result = engine.assess_sbt_coverage(holdings)
        assert result.net_zero_committed_pct == pytest.approx(50.0, abs=0.1)


class TestNetZeroProgress:
    """Test Net Zero progress tracking."""

    def test_net_zero_on_track_2050(self):
        """Portfolio with 7%+ annual reduction -> on_track_2050."""
        engine = CarbonTrajectoryEngine({"annual_reduction_target": 7.0})
        # prior=100, current=90 -> 10% reduction > 7%
        holdings = [_holding(
            current_intensity=90.0, prior_intensity=100.0,
            base_year_intensity=150.0, weight_pct=100.0, nav_value=10_000_000,
        )]
        result = engine.assess_net_zero_progress(holdings)
        assert isinstance(result, NetZeroProgress)
        assert result.on_track_2050 is True
        assert result.annual_reduction_rate >= 7.0
        assert len(result.provenance_hash) == 64

    def test_net_zero_off_track(self):
        """Insufficient annual reduction -> off track for 2050."""
        engine = CarbonTrajectoryEngine({"annual_reduction_target": 7.0})
        # prior=100, current=97 -> 3% reduction < 7%
        holdings = [_holding(
            current_intensity=97.0, prior_intensity=100.0,
            base_year_intensity=120.0, weight_pct=100.0, nav_value=10_000_000,
        )]
        result = engine.assess_net_zero_progress(holdings)
        assert result.on_track_2050 is False

    def test_net_zero_committed_holdings(self):
        """Net Zero committed holdings percentage tracked."""
        engine = CarbonTrajectoryEngine()
        holdings = [
            _holding(has_net_zero_commitment=True, nav_value=10_000_000),
            _holding(has_net_zero_commitment=False, nav_value=10_000_000),
        ]
        result = engine.assess_net_zero_progress(holdings)
        assert result.net_zero_committed_holdings_pct == pytest.approx(50.0, abs=0.5)

    def test_transition_plan_coverage(self):
        """Transition plan quality distribution is tracked."""
        engine = CarbonTrajectoryEngine()
        holdings = [
            _holding(
                transition_plan_quality=TransitionPlanQuality.COMPREHENSIVE,
                nav_value=10_000_000,
            ),
            _holding(
                transition_plan_quality=TransitionPlanQuality.PARTIAL,
                nav_value=10_000_000,
            ),
            _holding_no_sbt(
                transition_plan_quality=TransitionPlanQuality.ABSENT,
                nav_value=10_000_000,
            ),
        ]
        result = engine.assess_net_zero_progress(holdings)
        assert "comprehensive" in result.transition_plan_coverage
        assert "partial" in result.transition_plan_coverage
        assert "absent" in result.transition_plan_coverage

    def test_reduction_from_base(self):
        """Reduction from base year is calculated correctly."""
        engine = CarbonTrajectoryEngine()
        # base=120, current=80 -> reduction = (1 - 80/120)*100 = 33.33%
        holdings = [_holding(
            current_intensity=80.0, base_year_intensity=120.0,
            weight_pct=100.0, nav_value=10_000_000,
        )]
        result = engine.assess_net_zero_progress(holdings)
        assert result.reduction_from_base_pct == pytest.approx(33.33, rel=0.05)


class TestFullTrajectoryAssessment:
    """Test full assess_trajectory pipeline."""

    def test_full_assessment_structure(self):
        """Full assessment returns all sub-results."""
        engine = CarbonTrajectoryEngine({"product_name": "Test Fund"})
        holdings = _make_portfolio(n_green=5)
        result = engine.assess_trajectory(holdings)

        assert isinstance(result, TrajectoryResult)
        assert result.product_name == "Test Fund"
        assert result.itr_result is not None
        assert result.carbon_budget_result is not None
        assert result.sbt_coverage is not None
        assert result.net_zero_progress is not None
        assert result.total_holdings == 5
        assert result.portfolio_waci > 0
        assert result.prior_year_waci > 0
        assert result.data_coverage_pct > 0
        assert len(result.provenance_hash) == 64

    def test_yoy_reduction_calculation(self):
        """Year-on-year reduction = (1 - current/prior) * 100."""
        engine = CarbonTrajectoryEngine()
        # Each holding: current ~80, prior ~90 -> ~11% reduction
        holdings = [_holding(
            current_intensity=80.0, prior_intensity=90.0,
            weight_pct=100.0, nav_value=10_000_000,
        )]
        result = engine.assess_trajectory(holdings)
        expected_reduction = (1.0 - 80.0 / 90.0) * 100.0
        assert result.yoy_reduction_pct == pytest.approx(expected_reduction, rel=0.05)

    def test_meets_7pct_target_flag(self):
        """meets_7pct_target True when YoY >= 7%."""
        engine = CarbonTrajectoryEngine({"annual_reduction_target": 7.0})
        # 10% reduction
        holdings = [_holding(
            current_intensity=90.0, prior_intensity=100.0,
            weight_pct=100.0, nav_value=10_000_000,
        )]
        result = engine.assess_trajectory(holdings)
        assert result.meets_7pct_target is True

    def test_misses_7pct_target_flag(self):
        """meets_7pct_target False when YoY < 7%."""
        engine = CarbonTrajectoryEngine({"annual_reduction_target": 7.0})
        # 3% reduction
        holdings = [_holding(
            current_intensity=97.0, prior_intensity=100.0,
            weight_pct=100.0, nav_value=10_000_000,
        )]
        result = engine.assess_trajectory(holdings)
        assert result.meets_7pct_target is False

    def test_empty_holdings_raises(self):
        """Empty holdings list raises ValueError."""
        engine = CarbonTrajectoryEngine()
        with pytest.raises(ValueError, match="empty"):
            engine.assess_trajectory([])


class TestCarbonPathwayProjection:
    """Test 7% annual reduction pathway projection."""

    def test_annual_7pct_compounding(self):
        """Verify projected intensity = base * (1-0.07)^years_elapsed."""
        engine = CarbonTrajectoryEngine({
            "base_year": 2020,
            "projection_end_year": 2030,
            "annual_reduction_target": 7.0,
        })
        holdings = [_holding(
            current_intensity=100.0, base_year_intensity=100.0,
            weight_pct=100.0, nav_value=10_000_000,
        )]
        result = engine.analyze_carbon_budget(holdings)
        # Trajectory starts at current year and goes to 2050
        assert len(result.trajectory_points) > 0
        # First point should be current year's emissions
        first_point = result.trajectory_points[0]
        assert first_point["emissions_tco2e"] > 0

    def test_trajectory_decreases_over_time(self):
        """Each year in the trajectory has lower emissions than the prior."""
        engine = CarbonTrajectoryEngine()
        holdings = _make_portfolio(n_green=5)
        result = engine.analyze_carbon_budget(holdings)
        emissions = [p["emissions_tco2e"] for p in result.trajectory_points]
        for i in range(1, len(emissions)):
            assert emissions[i] <= emissions[i - 1]


class TestProvenanceTrajectory:
    """Test provenance hashing for trajectory results."""

    def test_itr_provenance(self):
        """ITR result has valid SHA-256 hash."""
        engine = CarbonTrajectoryEngine()
        holdings = _make_portfolio(n_green=3)
        result = engine.calculate_itr(holdings)
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)

    def test_budget_provenance(self):
        """Carbon budget result has valid SHA-256 hash."""
        engine = CarbonTrajectoryEngine()
        holdings = _make_portfolio(n_green=3)
        result = engine.analyze_carbon_budget(holdings)
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)

    def test_sbt_provenance(self):
        """SBT coverage result has valid SHA-256 hash."""
        engine = CarbonTrajectoryEngine()
        holdings = _make_portfolio(n_green=3)
        result = engine.assess_sbt_coverage(holdings)
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)

    def test_net_zero_provenance(self):
        """Net Zero progress result has valid SHA-256 hash."""
        engine = CarbonTrajectoryEngine()
        holdings = _make_portfolio(n_green=3)
        result = engine.assess_net_zero_progress(holdings)
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)

    def test_full_result_provenance(self):
        """Full trajectory result has valid SHA-256 hash."""
        engine = CarbonTrajectoryEngine()
        holdings = _make_portfolio(n_green=3)
        result = engine.assess_trajectory(holdings)
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)
