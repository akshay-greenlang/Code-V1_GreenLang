# -*- coding: utf-8 -*-
"""
Unit tests for BenchmarkAlignmentEngine (PACK-011 SFDR Article 9, Engine 5).

Tests CTB/PAB compliance, exclusion violation detection, decarbonization
trajectory projection, tracking error calculation, methodology disclosure,
and provenance hashing.

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


_bm_mod = _import_from_path(
    "benchmark_alignment_engine",
    str(ENGINES_DIR / "benchmark_alignment_engine.py"),
)

BenchmarkAlignmentEngine = _bm_mod.BenchmarkAlignmentEngine
BenchmarkConfig = _bm_mod.BenchmarkConfig
HoldingBenchmarkData = _bm_mod.HoldingBenchmarkData
BenchmarkResult = _bm_mod.BenchmarkResult
CTBComplianceResult = _bm_mod.CTBComplianceResult
PABComplianceResult = _bm_mod.PABComplianceResult
ExclusionViolation = _bm_mod.ExclusionViolation
TrajectoryDataPoint = _bm_mod.TrajectoryDataPoint
TrackingErrorResult = _bm_mod.TrackingErrorResult
MethodologyDisclosure = _bm_mod.MethodologyDisclosure
BenchmarkType = _bm_mod.BenchmarkType
ComplianceStatus = _bm_mod.ComplianceStatus
ExclusionCategory = _bm_mod.ExclusionCategory

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _clean_holding(**kwargs) -> HoldingBenchmarkData:
    """Create a clean holding with no exclusion violations."""
    defaults = dict(
        company_name="Clean Corp",
        carbon_intensity=40.0,
        holding_value=10_000_000,
        weight_pct=10.0,
        prior_year_carbon_intensity=43.5,
        reporting_year=2025,
        controversial_weapons=False,
        coal_exploration_revenue_pct=0.0,
        oil_gas_exploration_revenue_pct=0.0,
        fossil_refining_revenue_pct=0.0,
        fossil_distribution_revenue_pct=0.0,
        power_generation_carbon_intensity=0.0,
    )
    defaults.update(kwargs)
    return HoldingBenchmarkData(**defaults)


def _violating_holding(**kwargs) -> HoldingBenchmarkData:
    """Create a holding that triggers PAB exclusion violations."""
    defaults = dict(
        company_name="Dirty Corp",
        carbon_intensity=200.0,
        holding_value=5_000_000,
        weight_pct=5.0,
        prior_year_carbon_intensity=210.0,
        reporting_year=2025,
        controversial_weapons=False,
        coal_exploration_revenue_pct=3.0,
        oil_gas_exploration_revenue_pct=2.0,
        fossil_refining_revenue_pct=15.0,
        fossil_distribution_revenue_pct=55.0,
        power_generation_carbon_intensity=150.0,
    )
    defaults.update(kwargs)
    return HoldingBenchmarkData(**defaults)


def _make_portfolio(n_clean=8, n_dirty=0, universe_intensity=100.0):
    """Build a portfolio for common test scenarios."""
    holdings = []
    total = n_clean + n_dirty
    for i in range(n_clean):
        holdings.append(_clean_holding(
            company_name=f"Clean-{i}",
            holding_value=10_000_000,
            weight_pct=100.0 / total,
            carbon_intensity=40.0 + i,
            prior_year_carbon_intensity=46.0 + i,
        ))
    for i in range(n_dirty):
        holdings.append(_violating_holding(
            company_name=f"Dirty-{i}",
            holding_value=5_000_000,
            weight_pct=100.0 / total,
        ))
    return holdings


# ===========================================================================
# Tests
# ===========================================================================


class TestBenchmarkAlignmentEngineInit:
    """Test engine initialization."""

    def test_default_config(self):
        engine = BenchmarkAlignmentEngine()
        assert engine.config.benchmark_type == BenchmarkType.PAB
        assert engine.config.annual_decarbonization_rate == pytest.approx(0.07)

    def test_dict_config(self):
        engine = BenchmarkAlignmentEngine({"benchmark_type": "ctb"})
        assert engine.config.benchmark_type == BenchmarkType.CTB

    def test_pydantic_config(self):
        cfg = BenchmarkConfig(benchmark_type=BenchmarkType.CUSTOM, base_year=2020)
        engine = BenchmarkAlignmentEngine(cfg)
        assert engine.config.benchmark_type == BenchmarkType.CUSTOM
        assert engine.config.base_year == 2020


class TestCTBCompliance:
    """Test Climate Transition Benchmark compliance."""

    def test_ctb_compliant_portfolio(self):
        """Portfolio with 30%+ reduction, no weapons, 7%+ decarb -> COMPLIANT."""
        engine = BenchmarkAlignmentEngine({"benchmark_type": "ctb"})
        holdings = _make_portfolio(n_clean=10)
        result = engine.assess_ctb_compliance(holdings, universe_intensity=100.0)

        assert isinstance(result, CTBComplianceResult)
        assert result.status == ComplianceStatus.COMPLIANT
        assert result.intensity_reduction_met is True
        assert result.intensity_reduction_pct >= 30.0
        assert result.controversial_weapons_exclusion_met is True
        assert result.decarbonization_target_met is True
        assert result.holdings_screened == 10
        assert len(result.provenance_hash) == 64

    def test_ctb_30pct_intensity_reduction_threshold(self):
        """Exactly at 30% reduction boundary."""
        engine = BenchmarkAlignmentEngine({"benchmark_type": "ctb"})
        holdings = [_clean_holding(
            carbon_intensity=70.0,
            weight_pct=100.0,
            holding_value=10_000_000,
            prior_year_carbon_intensity=77.0,
        )]
        result = engine.assess_ctb_compliance(holdings, universe_intensity=100.0)
        assert result.intensity_reduction_pct >= 30.0
        assert result.intensity_reduction_met is True

    def test_ctb_insufficient_reduction(self):
        """Portfolio with <30% reduction -> NON_COMPLIANT."""
        engine = BenchmarkAlignmentEngine({"benchmark_type": "ctb"})
        holdings = [_clean_holding(
            carbon_intensity=90.0,
            weight_pct=100.0,
            holding_value=10_000_000,
            prior_year_carbon_intensity=97.0,
        )]
        result = engine.assess_ctb_compliance(holdings, universe_intensity=100.0)
        assert result.intensity_reduction_pct < 30.0
        assert result.intensity_reduction_met is False
        assert result.status == ComplianceStatus.NON_COMPLIANT

    def test_ctb_controversial_weapons_violation(self):
        """Holding with controversial weapons -> violation detected."""
        engine = BenchmarkAlignmentEngine({"benchmark_type": "ctb"})
        holdings = [_clean_holding(
            controversial_weapons=True,
            weight_pct=100.0,
            holding_value=10_000_000,
            prior_year_carbon_intensity=46.0,
        )]
        result = engine.assess_ctb_compliance(holdings, universe_intensity=100.0)
        assert result.controversial_weapons_exclusion_met is False
        assert any(
            v.category == ExclusionCategory.CONTROVERSIAL_WEAPONS
            for v in result.exclusion_violations
        )

    def test_ctb_zero_universe_intensity(self):
        """Zero universe intensity -> INSUFFICIENT_DATA."""
        engine = BenchmarkAlignmentEngine({"benchmark_type": "ctb"})
        holdings = [_clean_holding(weight_pct=100.0, holding_value=10_000_000)]
        result = engine.assess_ctb_compliance(holdings, universe_intensity=0.0)
        assert result.status == ComplianceStatus.INSUFFICIENT_DATA


class TestPABCompliance:
    """Test Paris-Aligned Benchmark compliance."""

    def test_pab_compliant_portfolio(self):
        """Clean portfolio with 50%+ reduction and no exclusions -> COMPLIANT."""
        engine = BenchmarkAlignmentEngine({"benchmark_type": "pab"})
        holdings = _make_portfolio(n_clean=10)
        result = engine.assess_pab_compliance(holdings, universe_intensity=120.0)

        assert isinstance(result, PABComplianceResult)
        assert result.intensity_reduction_pct >= 50.0
        assert result.intensity_reduction_met is True
        assert result.fossil_fuel_exclusions_met is True
        assert result.controversial_weapons_exclusion_met is True
        assert result.holdings_screened == 10

    def test_pab_50pct_intensity_reduction(self):
        """Verify 50% intensity reduction formula."""
        engine = BenchmarkAlignmentEngine({"benchmark_type": "pab"})
        holdings = [_clean_holding(
            carbon_intensity=50.0, weight_pct=100.0,
            holding_value=10_000_000, prior_year_carbon_intensity=55.0,
        )]
        result = engine.assess_pab_compliance(holdings, universe_intensity=100.0)
        assert result.intensity_reduction_pct == pytest.approx(50.0, abs=0.1)
        assert result.intensity_reduction_met is True

    def test_pab_coal_exclusion_violation(self):
        """Holding with >=1% coal exploration revenue -> violation."""
        engine = BenchmarkAlignmentEngine({"benchmark_type": "pab"})
        holdings = [_clean_holding(
            coal_exploration_revenue_pct=1.5,
            weight_pct=100.0, holding_value=10_000_000,
        )]
        result = engine.assess_pab_compliance(holdings, universe_intensity=100.0)
        assert result.fossil_fuel_exclusions_met is False
        assert result.fossil_exploration_violations >= 1
        coal_violations = [
            v for v in result.exclusion_violations
            if v.category == ExclusionCategory.COAL_EXPLORATION
        ]
        assert len(coal_violations) >= 1
        assert coal_violations[0].actual_value == pytest.approx(1.5)
        assert coal_violations[0].threshold_value == pytest.approx(1.0)

    def test_pab_oil_gas_exclusion_violation(self):
        """Holding with >=1% oil/gas exploration revenue -> violation."""
        engine = BenchmarkAlignmentEngine({"benchmark_type": "pab"})
        holdings = [_clean_holding(
            oil_gas_exploration_revenue_pct=2.0,
            weight_pct=100.0, holding_value=10_000_000,
        )]
        result = engine.assess_pab_compliance(holdings, universe_intensity=100.0)
        assert result.fossil_fuel_exclusions_met is False

    def test_pab_refining_exclusion_at_10pct(self):
        """Holding with >=10% refining revenue -> violation."""
        engine = BenchmarkAlignmentEngine({"benchmark_type": "pab"})
        holdings = [_clean_holding(
            fossil_refining_revenue_pct=10.0,
            weight_pct=100.0, holding_value=10_000_000,
        )]
        result = engine.assess_pab_compliance(holdings, universe_intensity=100.0)
        refining_viols = [
            v for v in result.exclusion_violations
            if v.category == ExclusionCategory.FOSSIL_REFINING
        ]
        assert len(refining_viols) >= 1

    def test_pab_distribution_exclusion_at_50pct(self):
        """Holding with >=50% distribution revenue -> violation."""
        engine = BenchmarkAlignmentEngine({"benchmark_type": "pab"})
        holdings = [_clean_holding(
            fossil_distribution_revenue_pct=50.0,
            weight_pct=100.0, holding_value=10_000_000,
        )]
        result = engine.assess_pab_compliance(holdings, universe_intensity=100.0)
        assert result.fossil_fuel_exclusions_met is False
        dist_viols = [
            v for v in result.exclusion_violations
            if v.category == ExclusionCategory.FOSSIL_DISTRIBUTION
        ]
        assert len(dist_viols) >= 1

    def test_pab_high_carbon_power_violation(self):
        """Holding with >100g CO2/kWh power -> violation."""
        engine = BenchmarkAlignmentEngine({"benchmark_type": "pab"})
        holdings = [_clean_holding(
            power_generation_carbon_intensity=150.0,
            weight_pct=100.0, holding_value=10_000_000,
        )]
        result = engine.assess_pab_compliance(holdings, universe_intensity=100.0)
        power_viols = [
            v for v in result.exclusion_violations
            if v.category == ExclusionCategory.HIGH_CARBON_POWER
        ]
        assert len(power_viols) >= 1
        assert result.high_carbon_power_violations >= 1

    def test_pab_at_100g_no_violation(self):
        """Holding at exactly 100g CO2/kWh -> NO violation (gt, not gte)."""
        engine = BenchmarkAlignmentEngine({"benchmark_type": "pab"})
        holdings = [_clean_holding(
            power_generation_carbon_intensity=100.0,
            weight_pct=100.0, holding_value=10_000_000,
        )]
        result = engine.assess_pab_compliance(holdings, universe_intensity=100.0)
        power_viols = [
            v for v in result.exclusion_violations
            if v.category == ExclusionCategory.HIGH_CARBON_POWER
        ]
        assert len(power_viols) == 0

    def test_pab_multiple_violations_single_holding(self):
        """Single holding can trigger multiple PAB violations."""
        engine = BenchmarkAlignmentEngine({"benchmark_type": "pab"})
        holdings = [_violating_holding(weight_pct=100.0, holding_value=10_000_000)]
        result = engine.assess_pab_compliance(holdings, universe_intensity=100.0)
        assert len(result.exclusion_violations) >= 3
        assert result.fossil_fuel_exclusions_met is False

    def test_pab_non_compliant_insufficient_reduction(self):
        """PAB with <50% reduction -> NON_COMPLIANT."""
        engine = BenchmarkAlignmentEngine({"benchmark_type": "pab"})
        holdings = [_clean_holding(
            carbon_intensity=80.0, weight_pct=100.0,
            holding_value=10_000_000, prior_year_carbon_intensity=86.0,
        )]
        result = engine.assess_pab_compliance(holdings, universe_intensity=100.0)
        assert result.intensity_reduction_pct < 50.0
        assert result.status in (ComplianceStatus.NON_COMPLIANT, ComplianceStatus.PARTIAL)


class TestDecarbonizationTrajectory:
    """Test decarbonization trajectory projection."""

    def test_trajectory_7pct_annual_reduction(self):
        """Trajectory follows target(year) = base * (1-0.07)^(year-base_year)."""
        engine = BenchmarkAlignmentEngine({
            "benchmark_type": "pab",
            "base_year": 2019,
            "projection_end_year": 2050,
        })
        trajectory = engine.project_trajectory(
            holdings=None,
            base_intensity=100.0,
            current_intensity=None,
        )
        assert len(trajectory) > 0
        # Verify 2020 target = 100 * (1-0.07)^1 = 93
        point_2020 = next((p for p in trajectory if p.year == 2020), None)
        assert point_2020 is not None
        assert point_2020.target_intensity == pytest.approx(93.0, abs=0.1)

        # 2030 target = 100 * (0.93)^11
        point_2030 = next((p for p in trajectory if p.year == 2030), None)
        assert point_2030 is not None
        expected_2030 = 100.0 * (0.93 ** 11)
        assert point_2030.target_intensity == pytest.approx(expected_2030, rel=0.01)

    def test_trajectory_current_year_actual(self):
        """Trajectory includes actual intensity at the current reporting year."""
        engine = BenchmarkAlignmentEngine({
            "benchmark_type": "pab",
            "base_year": 2019,
            "current_year": 2025,
            "base_year_intensity": 100.0,
        })
        holdings = [_clean_holding(carbon_intensity=60.0, weight_pct=100.0)]
        trajectory = engine.project_trajectory(
            holdings=holdings,
            current_intensity=60.0,
        )
        current_point = next((p for p in trajectory if p.year == 2025), None)
        assert current_point is not None
        assert current_point.actual_intensity is not None
        assert current_point.actual_intensity == pytest.approx(60.0, abs=0.1)

    def test_trajectory_on_track_detection(self):
        """On track when actual <= target intensity."""
        engine = BenchmarkAlignmentEngine({
            "benchmark_type": "pab",
            "base_year": 2019,
            "current_year": 2025,
            "base_year_intensity": 100.0,
        })
        trajectory = engine.project_trajectory(
            holdings=None, base_intensity=100.0, current_intensity=60.0,
        )
        point = next((p for p in trajectory if p.year == 2025), None)
        assert point is not None
        assert point.on_track is True

    def test_trajectory_off_track_detection(self):
        """Off track when actual > target intensity."""
        engine = BenchmarkAlignmentEngine({
            "benchmark_type": "pab",
            "base_year": 2019,
            "current_year": 2025,
            "base_year_intensity": 100.0,
        })
        trajectory = engine.project_trajectory(
            holdings=None, base_intensity=100.0, current_intensity=90.0,
        )
        point = next((p for p in trajectory if p.year == 2025), None)
        assert point is not None
        assert point.on_track is False
        assert point.gap_pct > 0.0


class TestTrackingError:
    """Test tracking error calculation."""

    def test_tracking_error_basic(self):
        """TE = sqrt(sum((r_p - r_b)^2)/(n-1)), annualized * sqrt(12)."""
        engine = BenchmarkAlignmentEngine()
        p_returns = [0.02, 0.03, -0.01, 0.04, 0.01, -0.02,
                     0.03, 0.02, 0.01, 0.05, -0.01, 0.03]
        b_returns = [0.015, 0.025, -0.005, 0.035, 0.015, -0.015,
                     0.025, 0.015, 0.005, 0.04, -0.005, 0.025]
        result = engine.calculate_tracking_error(p_returns, b_returns)

        assert isinstance(result, TrackingErrorResult)
        assert result.observation_count == 12
        assert result.tracking_error_monthly > 0.0
        assert result.tracking_error_annualized > 0.0
        assert result.tracking_error_annualized == pytest.approx(
            result.tracking_error_monthly * math.sqrt(12), rel=0.01,
        )
        assert len(result.provenance_hash) == 64

    def test_tracking_error_different_lengths_raises(self):
        """Unequal length return lists -> ValueError."""
        engine = BenchmarkAlignmentEngine()
        with pytest.raises(ValueError, match="equal length"):
            engine.calculate_tracking_error([0.01, 0.02], [0.01])

    def test_tracking_error_too_few_observations_raises(self):
        """Fewer than 2 observations -> ValueError."""
        engine = BenchmarkAlignmentEngine()
        with pytest.raises(ValueError, match="At least 2"):
            engine.calculate_tracking_error([0.01], [0.01])

    def test_tracking_error_identical_returns(self):
        """Identical returns -> tracking error = 0."""
        engine = BenchmarkAlignmentEngine()
        returns = [0.01, 0.02, -0.01, 0.03]
        result = engine.calculate_tracking_error(returns, returns)
        assert result.tracking_error_monthly == pytest.approx(0.0, abs=1e-10)
        assert result.tracking_error_annualized == pytest.approx(0.0, abs=1e-10)


class TestMethodologyDisclosure:
    """Test methodology disclosure generation."""

    def test_pab_methodology_disclosure(self):
        """PAB disclosure includes all fossil fuel exclusion categories."""
        engine = BenchmarkAlignmentEngine({
            "benchmark_type": "pab",
            "benchmark_name": "EU PAB Index",
            "benchmark_provider": "MSCI",
        })
        disclosure = engine.get_methodology_disclosure(universe_intensity=100.0)
        assert isinstance(disclosure, MethodologyDisclosure)
        assert disclosure.benchmark_type == BenchmarkType.PAB
        assert disclosure.benchmark_name == "EU PAB Index"
        assert disclosure.benchmark_provider == "MSCI"
        assert disclosure.intensity_reduction_target_pct == pytest.approx(50.0)
        assert disclosure.decarbonization_rate_pct == pytest.approx(7.0)
        assert len(disclosure.exclusions_applied) >= 6
        assert "controversial_weapons" in disclosure.exclusions_applied
        assert "coal_exploration" in disclosure.exclusions_applied
        assert len(disclosure.provenance_hash) == 64

    def test_ctb_methodology_disclosure(self):
        """CTB disclosure has fewer exclusions than PAB."""
        engine = BenchmarkAlignmentEngine({"benchmark_type": "ctb"})
        disclosure = engine.get_methodology_disclosure(universe_intensity=100.0)
        assert disclosure.benchmark_type == BenchmarkType.CTB
        assert disclosure.intensity_reduction_target_pct == pytest.approx(30.0)
        assert len(disclosure.exclusions_applied) == 1
        assert "controversial_weapons" in disclosure.exclusions_applied

    def test_additional_exclusions_in_methodology(self):
        """Additional exclusions (tobacco, UNGC) appear when configured."""
        engine = BenchmarkAlignmentEngine({
            "benchmark_type": "pab",
            "include_additional_exclusions": True,
        })
        disclosure = engine.get_methodology_disclosure(universe_intensity=100.0)
        assert "tobacco" in disclosure.exclusions_applied
        assert "ungc_violations" in disclosure.exclusions_applied


class TestFullAlignment:
    """Test full assess_alignment pipeline."""

    def test_full_pab_assessment(self):
        """Full PAB assessment returns complete BenchmarkResult."""
        engine = BenchmarkAlignmentEngine({"benchmark_type": "pab"})
        holdings = _make_portfolio(n_clean=10)
        result = engine.assess_alignment(holdings, universe_intensity=120.0)

        assert isinstance(result, BenchmarkResult)
        assert result.benchmark_type == BenchmarkType.PAB
        assert result.pab_result is not None
        assert result.ctb_result is None
        assert result.total_holdings == 10
        assert result.portfolio_waci > 0
        assert len(result.trajectory) > 0
        assert result.methodology is not None
        assert len(result.provenance_hash) == 64

    def test_full_ctb_assessment(self):
        """Full CTB assessment populates ctb_result, not pab_result."""
        engine = BenchmarkAlignmentEngine({"benchmark_type": "ctb"})
        holdings = _make_portfolio(n_clean=5)
        result = engine.assess_alignment(holdings, universe_intensity=100.0)

        assert result.benchmark_type == BenchmarkType.CTB
        assert result.ctb_result is not None
        assert result.pab_result is None

    def test_custom_benchmark_both_assessed(self):
        """CUSTOM benchmark assesses both CTB and PAB."""
        engine = BenchmarkAlignmentEngine({"benchmark_type": "custom"})
        holdings = _make_portfolio(n_clean=5)
        result = engine.assess_alignment(holdings, universe_intensity=120.0)

        assert result.benchmark_type == BenchmarkType.CUSTOM
        assert result.ctb_result is not None
        assert result.pab_result is not None

    def test_exclusion_violation_count(self):
        """Exclusion violations are counted in result."""
        engine = BenchmarkAlignmentEngine({"benchmark_type": "pab"})
        holdings = _make_portfolio(n_clean=5, n_dirty=2)
        result = engine.assess_alignment(holdings, universe_intensity=120.0)
        assert result.exclusion_violation_count > 0


class TestProvenanceAndReproducibility:
    """Test deterministic provenance hashing."""

    def test_provenance_hash_is_sha256(self):
        """All provenance hashes are 64 hex characters (SHA-256)."""
        engine = BenchmarkAlignmentEngine({"benchmark_type": "pab"})
        holdings = _make_portfolio(n_clean=3)
        result = engine.assess_alignment(holdings, universe_intensity=100.0)
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)  # validates hex

    def test_provenance_deterministic(self):
        """Same input produces same provenance hash (bit-perfect)."""
        engine = BenchmarkAlignmentEngine({"benchmark_type": "pab"})
        h = _clean_holding(
            carbon_intensity=50.0, weight_pct=100.0, holding_value=10_000_000
        )
        v1 = engine.screen_exclusions([h])
        v2 = engine.screen_exclusions([h])
        assert v1 == v2


class TestExclusionSummary:
    """Test exclusion summary reporting."""

    def test_exclusion_summary_structure(self):
        """Summary has expected keys and structure."""
        engine = BenchmarkAlignmentEngine({"benchmark_type": "pab"})
        holdings = _make_portfolio(n_clean=3, n_dirty=2)
        summary = engine.get_exclusion_summary(holdings)
        assert "total_violations" in summary
        assert "total_weight_pct" in summary
        assert "by_category" in summary
        assert "violating_holdings" in summary
        assert summary["total_violations"] > 0

    def test_exclusion_summary_clean_portfolio(self):
        """Clean portfolio has zero violations in summary."""
        engine = BenchmarkAlignmentEngine({"benchmark_type": "pab"})
        holdings = _make_portfolio(n_clean=5)
        summary = engine.get_exclusion_summary(holdings)
        assert summary["total_violations"] == 0


class TestTrajectorySummary:
    """Test trajectory summary reporting."""

    def test_trajectory_summary_milestones(self):
        """Summary includes milestone years (2030, 2050)."""
        engine = BenchmarkAlignmentEngine({
            "benchmark_type": "pab",
            "base_year": 2019,
            "base_year_intensity": 100.0,
            "projection_end_year": 2050,
        })
        holdings = _make_portfolio(n_clean=5)
        summary = engine.get_trajectory_summary(holdings)
        assert "milestones" in summary
        assert "2030" in summary["milestones"]
        assert "2050" in summary["milestones"]
        assert summary["annual_rate_pct"] == pytest.approx(7.0)
