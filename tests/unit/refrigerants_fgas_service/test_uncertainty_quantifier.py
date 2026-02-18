# -*- coding: utf-8 -*-
"""
Unit tests for UncertaintyQuantifierEngine - AGENT-MRV-002 Engine 5

Tests Monte Carlo simulation, analytical error propagation, data quality
scoring, contribution analysis, confidence intervals, and reproducibility
guarantees across all five calculation methods.

Target: 65+ tests, 700+ lines.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, Optional, Tuple

import pytest

from greenlang.refrigerants_fgas.uncertainty_quantifier import (
    UncertaintyQuantifierEngine,
    UncertaintyMethod,
    UncertaintyResult,
    DataQualityLevel,
    _METHOD_UNCERTAINTY_RANGES,
    _METHOD_DEFAULT_UNCERTAINTY,
    _CONFIDENCE_Z_SCORES,
    _DQI_MULTIPLIERS,
    _DEFAULT_ITERATIONS,
    _MIN_ITERATIONS,
    _MAX_ITERATIONS,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def engine() -> UncertaintyQuantifierEngine:
    """Create a fresh UncertaintyQuantifierEngine."""
    eng = UncertaintyQuantifierEngine()
    yield eng
    eng.clear()


@pytest.fixture
def equipment_params() -> Dict[str, Any]:
    """Standard equipment-based calculation parameters."""
    return {
        "charge_kg": 25.0,
        "leak_rate": 0.10,
        "gwp": 2088.0,
    }


@pytest.fixture
def mass_balance_params() -> Dict[str, Any]:
    """Standard mass balance calculation parameters."""
    return {
        "beginning_inventory_kg": 500.0,
        "ending_inventory_kg": 420.0,
        "purchases_kg": 100.0,
        "sales_kg": 20.0,
        "acquisitions_kg": 10.0,
        "divestitures_kg": 5.0,
        "capacity_change_kg": 30.0,
        "gwp": 1530.0,
    }


@pytest.fixture
def screening_params() -> Dict[str, Any]:
    """Standard screening calculation parameters."""
    return {
        "total_charge_kg": 100.0,
        "leak_rate": 0.15,
        "gwp": 2088.0,
    }


@pytest.fixture
def direct_measurement_params() -> Dict[str, Any]:
    """Standard direct measurement parameters."""
    return {
        "measured_loss_kg": 8.0,
        "gwp": 2088.0,
    }


@pytest.fixture
def top_down_params() -> Dict[str, Any]:
    """Standard top-down calculation parameters."""
    return {
        "total_purchased_kg": 200.0,
        "total_recovered_kg": 150.0,
        "gwp": 2088.0,
    }


# ===========================================================================
# Test: Initialization
# ===========================================================================


class TestUncertaintyQuantifierInit:
    """Tests for engine initialization."""

    def test_creation(self, engine: UncertaintyQuantifierEngine):
        """Engine initializes successfully."""
        assert engine is not None

    def test_repr(self, engine: UncertaintyQuantifierEngine):
        """Engine has a human-readable repr."""
        r = repr(engine)
        assert "UncertaintyQuantifierEngine" in r
        assert "methods=" in r
        assert "assessments=" in r

    def test_len_initially_zero(self, engine: UncertaintyQuantifierEngine):
        """Engine starts with zero assessments."""
        assert len(engine) == 0

    def test_stats_initial(self, engine: UncertaintyQuantifierEngine):
        """get_stats returns correct initial state."""
        stats = engine.get_stats()
        assert stats["total_assessments"] == 0
        assert stats["methods_supported"] == 5
        assert stats["dqi_levels"] == 5
        assert stats["default_iterations"] == _DEFAULT_ITERATIONS

    def test_clear(self, engine: UncertaintyQuantifierEngine):
        """clear() resets assessment history."""
        engine.quantify(
            emissions_tco2e=Decimal("10"),
            method="EQUIPMENT_BASED",
            parameters={"charge_kg": 5.0, "leak_rate": 0.1, "gwp": 2088},
            iterations=100,
            seed=42,
        )
        assert len(engine) == 1
        engine.clear()
        assert len(engine) == 0


# ===========================================================================
# Test: Monte Carlo Simulation
# ===========================================================================


class TestMonteCarloSimulation:
    """Tests for Monte Carlo uncertainty simulation."""

    def test_monte_carlo_equipment_based(
        self, engine: UncertaintyQuantifierEngine, equipment_params: Dict
    ):
        """MC for equipment-based returns valid structure with mean > 0."""
        result = engine.monte_carlo(
            emissions_tco2e=Decimal("5.220"),
            method="EQUIPMENT_BASED",
            parameters=equipment_params,
            iterations=1000,
            seed=42,
        )
        assert result["mean"] is not None
        assert result["mean"] > 0
        assert result["std"] is not None
        assert result["std"] > 0
        assert result["median"] is not None
        assert result["median"] > 0
        assert result["sample_count"] == 1000
        assert "percentiles" in result
        assert "confidence_intervals" in result
        assert "95" in result["confidence_intervals"]

    def test_monte_carlo_mass_balance(
        self, engine: UncertaintyQuantifierEngine, mass_balance_params: Dict
    ):
        """MC for mass balance produces valid results."""
        result = engine.monte_carlo(
            emissions_tco2e=Decimal("206.550"),
            method="MASS_BALANCE",
            parameters=mass_balance_params,
            iterations=1000,
            seed=42,
        )
        assert result["mean"] is not None
        assert result["mean"] > 0
        assert result["std"] is not None
        assert result["std"] > 0
        assert result["sample_count"] == 1000

    def test_monte_carlo_screening(
        self, engine: UncertaintyQuantifierEngine, screening_params: Dict
    ):
        """MC for screening returns higher uncertainty range than equipment-based."""
        result_screening = engine.monte_carlo(
            emissions_tco2e=Decimal("31.320"),
            method="SCREENING",
            parameters=screening_params,
            iterations=2000,
            seed=42,
        )
        result_equip = engine.monte_carlo(
            emissions_tco2e=Decimal("5.220"),
            method="EQUIPMENT_BASED",
            parameters={
                "charge_kg": 25.0,
                "leak_rate": 0.10,
                "gwp": 2088.0,
            },
            iterations=2000,
            seed=42,
        )
        assert result_screening["mean"] > 0
        assert result_screening["std"] > 0
        # Screening should have higher relative uncertainty
        rel_screening = result_screening["std"] / result_screening["mean"]
        rel_equip = result_equip["std"] / result_equip["mean"]
        assert rel_screening > rel_equip

    def test_monte_carlo_direct_measurement(
        self, engine: UncertaintyQuantifierEngine, direct_measurement_params: Dict
    ):
        """MC for direct measurement returns lower uncertainty range."""
        result = engine.monte_carlo(
            emissions_tco2e=Decimal("16.704"),
            method="DIRECT_MEASUREMENT",
            parameters=direct_measurement_params,
            iterations=1000,
            seed=42,
        )
        assert result["mean"] > 0
        assert result["std"] > 0
        # Direct measurement has lowest relative std
        rel_std = result["std"] / result["mean"]
        assert rel_std < 0.20  # Should be well under 20%

    def test_monte_carlo_top_down(
        self, engine: UncertaintyQuantifierEngine, top_down_params: Dict
    ):
        """MC for top-down returns valid results."""
        result = engine.monte_carlo(
            emissions_tco2e=Decimal("104.400"),
            method="TOP_DOWN",
            parameters=top_down_params,
            iterations=1000,
            seed=42,
        )
        assert result["mean"] > 0
        assert result["std"] > 0
        assert result["sample_count"] == 1000

    def test_monte_carlo_reproducibility(
        self, engine: UncertaintyQuantifierEngine, equipment_params: Dict
    ):
        """Same seed produces identical MC results."""
        r1 = engine.monte_carlo(
            emissions_tco2e=Decimal("5.220"),
            method="EQUIPMENT_BASED",
            parameters=equipment_params,
            iterations=500,
            seed=42,
        )
        r2 = engine.monte_carlo(
            emissions_tco2e=Decimal("5.220"),
            method="EQUIPMENT_BASED",
            parameters=equipment_params,
            iterations=500,
            seed=42,
        )
        assert r1["mean"] == r2["mean"]
        assert r1["std"] == r2["std"]
        assert r1["median"] == r2["median"]

    def test_monte_carlo_different_seeds(
        self, engine: UncertaintyQuantifierEngine, equipment_params: Dict
    ):
        """Different seeds produce different MC results."""
        r1 = engine.monte_carlo(
            emissions_tco2e=Decimal("5.220"),
            method="EQUIPMENT_BASED",
            parameters=equipment_params,
            iterations=500,
            seed=42,
        )
        r2 = engine.monte_carlo(
            emissions_tco2e=Decimal("5.220"),
            method="EQUIPMENT_BASED",
            parameters=equipment_params,
            iterations=500,
            seed=99,
        )
        assert r1["mean"] != r2["mean"]

    def test_monte_carlo_custom_iterations(
        self, engine: UncertaintyQuantifierEngine, equipment_params: Dict
    ):
        """Custom iteration counts are respected."""
        r_low = engine.monte_carlo(
            emissions_tco2e=Decimal("5.220"),
            method="EQUIPMENT_BASED",
            parameters=equipment_params,
            iterations=200,
            seed=42,
        )
        r_high = engine.monte_carlo(
            emissions_tco2e=Decimal("5.220"),
            method="EQUIPMENT_BASED",
            parameters=equipment_params,
            iterations=5000,
            seed=42,
        )
        assert r_low["sample_count"] == 200
        assert r_high["sample_count"] == 5000

    def test_monte_carlo_default_seed(
        self, engine: UncertaintyQuantifierEngine, equipment_params: Dict
    ):
        """None seed defaults to 12345 for reproducibility."""
        r1 = engine.monte_carlo(
            emissions_tco2e=Decimal("5.220"),
            method="EQUIPMENT_BASED",
            parameters=equipment_params,
            iterations=200,
            seed=None,
        )
        r2 = engine.monte_carlo(
            emissions_tco2e=Decimal("5.220"),
            method="EQUIPMENT_BASED",
            parameters=equipment_params,
            iterations=200,
            seed=12345,
        )
        assert r1["mean"] == r2["mean"]

    def test_monte_carlo_fallback_no_params(
        self, engine: UncertaintyQuantifierEngine
    ):
        """MC falls back to parametric sampling when no valid params provided."""
        result = engine.monte_carlo(
            emissions_tco2e=Decimal("10.0"),
            method="EQUIPMENT_BASED",
            parameters={},
            iterations=500,
            seed=42,
        )
        # Should still produce samples via fallback
        assert result["sample_count"] == 500
        assert result["mean"] > 0

    def test_monte_carlo_percentiles_structure(
        self, engine: UncertaintyQuantifierEngine, equipment_params: Dict
    ):
        """MC result percentiles contain expected keys."""
        result = engine.monte_carlo(
            emissions_tco2e=Decimal("5.220"),
            method="EQUIPMENT_BASED",
            parameters=equipment_params,
            iterations=1000,
            seed=42,
        )
        expected_keys = {"1", "2.5", "5", "10", "25", "50", "75", "90", "95", "97.5", "99"}
        assert set(result["percentiles"].keys()) == expected_keys

    def test_monte_carlo_percentiles_ordered(
        self, engine: UncertaintyQuantifierEngine, equipment_params: Dict
    ):
        """Percentile values are monotonically non-decreasing."""
        result = engine.monte_carlo(
            emissions_tco2e=Decimal("5.220"),
            method="EQUIPMENT_BASED",
            parameters=equipment_params,
            iterations=2000,
            seed=42,
        )
        p = result["percentiles"]
        keys_sorted = [1, 2.5, 5, 10, 25, 50, 75, 90, 95, 97.5, 99]
        values = [p[str(k)] for k in keys_sorted]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1]


# ===========================================================================
# Test: Analytical Error Propagation
# ===========================================================================


class TestAnalyticalPropagation:
    """Tests for IPCC Approach 1 analytical error propagation."""

    def test_analytical_propagation_equipment_based(
        self, engine: UncertaintyQuantifierEngine, equipment_params: Dict
    ):
        """Analytical propagation for equipment-based returns combined uncertainty."""
        result = engine.analytical_propagation(
            emissions_tco2e=Decimal("5.220"),
            method="EQUIPMENT_BASED",
            parameters=equipment_params,
        )
        assert "combined_uncertainty_pct" in result
        assert result["combined_uncertainty_pct"] > 0
        assert "component_uncertainties" in result
        assert "charge_kg" in result["component_uncertainties"]
        assert "leak_rate" in result["component_uncertainties"]
        assert "gwp" in result["component_uncertainties"]

    def test_analytical_propagation_mass_balance(
        self, engine: UncertaintyQuantifierEngine, mass_balance_params: Dict
    ):
        """Analytical propagation for mass balance returns valid results."""
        result = engine.analytical_propagation(
            emissions_tco2e=Decimal("206.550"),
            method="MASS_BALANCE",
            parameters=mass_balance_params,
        )
        assert result["combined_uncertainty_pct"] > 0
        comps = result["component_uncertainties"]
        assert "beginning_inventory_kg" in comps
        assert "ending_inventory_kg" in comps
        assert "purchases_kg" in comps

    def test_analytical_propagation_screening(
        self, engine: UncertaintyQuantifierEngine, screening_params: Dict
    ):
        """Analytical propagation for screening produces higher uncertainty."""
        result_screening = engine.analytical_propagation(
            emissions_tco2e=Decimal("31.320"),
            method="SCREENING",
            parameters=screening_params,
        )
        result_equip = engine.analytical_propagation(
            emissions_tco2e=Decimal("5.220"),
            method="EQUIPMENT_BASED",
            parameters={"charge_kg": 25.0, "leak_rate": 0.1, "gwp": 2088.0},
        )
        assert result_screening["combined_uncertainty_pct"] > result_equip["combined_uncertainty_pct"]

    def test_analytical_propagation_direct_measurement(
        self, engine: UncertaintyQuantifierEngine, direct_measurement_params: Dict
    ):
        """Analytical for direct measurement returns lowest combined uncertainty."""
        result = engine.analytical_propagation(
            emissions_tco2e=Decimal("16.704"),
            method="DIRECT_MEASUREMENT",
            parameters=direct_measurement_params,
        )
        assert result["combined_uncertainty_pct"] > 0
        # Direct measurement should have lowest analytical uncertainty
        assert result["combined_uncertainty_pct"] < 15.0

    def test_analytical_propagation_top_down(
        self, engine: UncertaintyQuantifierEngine, top_down_params: Dict
    ):
        """Analytical for top-down returns valid results."""
        result = engine.analytical_propagation(
            emissions_tco2e=Decimal("104.400"),
            method="TOP_DOWN",
            parameters=top_down_params,
        )
        assert result["combined_uncertainty_pct"] > 0
        comps = result["component_uncertainties"]
        assert "total_purchased_kg" in comps
        assert "total_recovered_kg" in comps
        assert "gwp" in comps

    def test_analytical_dqi_multiplier_effect(
        self, engine: UncertaintyQuantifierEngine, equipment_params: Dict
    ):
        """DQI multiplier > 1.0 increases analytical uncertainty."""
        r_base = engine.analytical_propagation(
            emissions_tco2e=Decimal("5.220"),
            method="EQUIPMENT_BASED",
            parameters=equipment_params,
            dqi_multiplier=Decimal("1.0"),
        )
        r_high = engine.analytical_propagation(
            emissions_tco2e=Decimal("5.220"),
            method="EQUIPMENT_BASED",
            parameters=equipment_params,
            dqi_multiplier=Decimal("2.0"),
        )
        assert r_high["combined_uncertainty_pct"] > r_base["combined_uncertainty_pct"]


# ===========================================================================
# Test: Confidence Intervals
# ===========================================================================


class TestConfidenceIntervals:
    """Tests for confidence interval calculations."""

    def test_confidence_intervals_90(
        self, engine: UncertaintyQuantifierEngine, equipment_params: Dict
    ):
        """90% CI has lower < mean < upper."""
        result = engine.quantify(
            emissions_tco2e=Decimal("5.220"),
            method="EQUIPMENT_BASED",
            parameters=equipment_params,
            iterations=1000,
            seed=42,
        )
        ci_90 = result.confidence_intervals["90"]
        assert ci_90[0] < result.emissions_tco2e
        assert result.emissions_tco2e < ci_90[1]

    def test_confidence_intervals_95(
        self, engine: UncertaintyQuantifierEngine, equipment_params: Dict
    ):
        """95% CI has lower < mean < upper."""
        result = engine.quantify(
            emissions_tco2e=Decimal("5.220"),
            method="EQUIPMENT_BASED",
            parameters=equipment_params,
            iterations=1000,
            seed=42,
        )
        ci_95 = result.confidence_intervals["95"]
        assert ci_95[0] < result.emissions_tco2e
        assert result.emissions_tco2e < ci_95[1]

    def test_confidence_intervals_99(
        self, engine: UncertaintyQuantifierEngine, equipment_params: Dict
    ):
        """99% CI has lower < mean < upper."""
        result = engine.quantify(
            emissions_tco2e=Decimal("5.220"),
            method="EQUIPMENT_BASED",
            parameters=equipment_params,
            iterations=1000,
            seed=42,
        )
        ci_99 = result.confidence_intervals["99"]
        assert ci_99[0] < result.emissions_tco2e
        assert result.emissions_tco2e < ci_99[1]

    def test_ci_ordering(
        self, engine: UncertaintyQuantifierEngine, equipment_params: Dict
    ):
        """90% CI is narrower than 95% CI, which is narrower than 99% CI."""
        result = engine.quantify(
            emissions_tco2e=Decimal("5.220"),
            method="EQUIPMENT_BASED",
            parameters=equipment_params,
            iterations=2000,
            seed=42,
        )
        ci_90 = result.confidence_intervals["90"]
        ci_95 = result.confidence_intervals["95"]
        ci_99 = result.confidence_intervals["99"]

        width_90 = ci_90[1] - ci_90[0]
        width_95 = ci_95[1] - ci_95[0]
        width_99 = ci_99[1] - ci_99[0]

        assert width_90 <= width_95
        assert width_95 <= width_99

    def test_all_ci_levels_present(
        self, engine: UncertaintyQuantifierEngine, equipment_params: Dict
    ):
        """Result contains all three CI levels."""
        result = engine.quantify(
            emissions_tco2e=Decimal("5.220"),
            method="EQUIPMENT_BASED",
            parameters=equipment_params,
            iterations=500,
            seed=42,
        )
        assert "90" in result.confidence_intervals
        assert "95" in result.confidence_intervals
        assert "99" in result.confidence_intervals


# ===========================================================================
# Test: Data Quality Scoring
# ===========================================================================


class TestDataQualityScoring:
    """Tests for DQI scoring logic."""

    def test_data_quality_scoring_tier5(
        self, engine: UncertaintyQuantifierEngine
    ):
        """Direct measurement data scores DQI = 5."""
        score = engine.score_data_quality(
            method="DIRECT_MEASUREMENT",
            data_sources={
                "data_source": "direct_measurement",
                "measurement_method": "weighed",
                "data_age": "current_year",
                "completeness": "complete",
            },
        )
        assert score == 5.0

    def test_data_quality_scoring_tier3(
        self, engine: UncertaintyQuantifierEngine
    ):
        """Published defaults score DQI = 3."""
        score = engine.score_data_quality(
            method="EQUIPMENT_BASED",
            data_sources={
                "data_source": "published_default",
                "measurement_method": "visual_inspection",
                "data_age": "within_5_years",
                "completeness": "above_70_pct",
            },
        )
        assert score == 3.0

    def test_data_quality_scoring_tier1(
        self, engine: UncertaintyQuantifierEngine
    ):
        """Expert judgment data scores DQI = 1."""
        score = engine.score_data_quality(
            method="SCREENING",
            data_sources={
                "data_source": "expert_judgment",
                "measurement_method": "not_measured",
                "data_age": "older_than_10_years",
                "completeness": "below_50_pct",
            },
        )
        assert score == 1.0

    def test_data_quality_scoring_tier4(
        self, engine: UncertaintyQuantifierEngine
    ):
        """Supplier data and recent records score DQI = 4."""
        score = engine.score_data_quality(
            method="EQUIPMENT_BASED",
            data_sources={
                "data_source": "supplier_data",
                "measurement_method": "ultrasonic",
                "data_age": "within_2_years",
                "completeness": "above_90_pct",
            },
        )
        assert score == 4.0

    def test_data_quality_scoring_tier2(
        self, engine: UncertaintyQuantifierEngine
    ):
        """Estimated/proxy data scores DQI = 2."""
        score = engine.score_data_quality(
            method="TOP_DOWN",
            data_sources={
                "data_source": "estimated",
                "measurement_method": "estimated_from_recharge",
                "data_age": "within_10_years",
                "completeness": "above_50_pct",
            },
        )
        assert score == 2.0

    def test_data_quality_default_equipment_based(
        self, engine: UncertaintyQuantifierEngine
    ):
        """Default DQI for EQUIPMENT_BASED is 3.0."""
        score = engine.score_data_quality("EQUIPMENT_BASED")
        assert score == 3.0

    def test_data_quality_default_mass_balance(
        self, engine: UncertaintyQuantifierEngine
    ):
        """Default DQI for MASS_BALANCE is 4.0."""
        score = engine.score_data_quality("MASS_BALANCE")
        assert score == 4.0

    def test_data_quality_default_screening(
        self, engine: UncertaintyQuantifierEngine
    ):
        """Default DQI for SCREENING is 2.0."""
        score = engine.score_data_quality("SCREENING")
        assert score == 2.0

    def test_data_quality_default_direct_measurement(
        self, engine: UncertaintyQuantifierEngine
    ):
        """Default DQI for DIRECT_MEASUREMENT is 5.0."""
        score = engine.score_data_quality("DIRECT_MEASUREMENT")
        assert score == 5.0

    def test_data_quality_default_top_down(
        self, engine: UncertaintyQuantifierEngine
    ):
        """Default DQI for TOP_DOWN is 3.5."""
        score = engine.score_data_quality("TOP_DOWN")
        assert score == 3.5

    def test_data_quality_empty_sources(
        self, engine: UncertaintyQuantifierEngine
    ):
        """Empty data_sources dict falls back to method default."""
        score = engine.score_data_quality("EQUIPMENT_BASED", data_sources={})
        assert score == 3.0


# ===========================================================================
# Test: Method Uncertainty Ranges
# ===========================================================================


class TestMethodUncertaintyRanges:
    """Tests for method-specific uncertainty ranges."""

    @pytest.mark.parametrize(
        "method,expected_low,expected_high",
        [
            ("EQUIPMENT_BASED", 20.0, 30.0),
            ("MASS_BALANCE", 5.0, 15.0),
            ("SCREENING", 40.0, 60.0),
            ("DIRECT_MEASUREMENT", 5.0, 10.0),
            ("TOP_DOWN", 15.0, 25.0),
        ],
    )
    def test_method_uncertainty_ranges(
        self,
        engine: UncertaintyQuantifierEngine,
        method: str,
        expected_low: float,
        expected_high: float,
    ):
        """Each method returns correct uncertainty range as percentages."""
        low, high = engine.get_method_uncertainty_range(method)
        assert low == pytest.approx(expected_low, rel=1e-6)
        assert high == pytest.approx(expected_high, rel=1e-6)

    def test_invalid_method_raises(self, engine: UncertaintyQuantifierEngine):
        """Unknown method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown uncertainty method"):
            engine.get_method_uncertainty_range("INVALID_METHOD")


# ===========================================================================
# Test: Contribution Analysis
# ===========================================================================


class TestContributionAnalysis:
    """Tests for parameter contribution analysis."""

    def test_contribution_analysis_equipment_based(
        self, engine: UncertaintyQuantifierEngine, equipment_params: Dict
    ):
        """Equipment-based contribution analysis returns parameter fractions summing to 1."""
        contributions = engine.contribution_analysis(
            parameters=equipment_params,
            method="EQUIPMENT_BASED",
        )
        assert "charge_kg" in contributions
        assert "leak_rate" in contributions
        assert "gwp" in contributions
        total = sum(contributions.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_contribution_analysis_leak_rate_dominates(
        self, engine: UncertaintyQuantifierEngine, equipment_params: Dict
    ):
        """For equipment-based, leak_rate (20% base unc) should dominate over charge_kg and gwp (5% each)."""
        contributions = engine.contribution_analysis(
            parameters=equipment_params,
            method="EQUIPMENT_BASED",
        )
        # leak_rate has 0.20 base unc vs 0.05 for charge and gwp
        # contribution = (0.20)^2 / (0.05^2 + 0.20^2 + 0.05^2) = 0.04 / 0.045 = 0.8889
        assert contributions["leak_rate"] > contributions["charge_kg"]
        assert contributions["leak_rate"] > contributions["gwp"]

    def test_contribution_analysis_mass_balance(
        self, engine: UncertaintyQuantifierEngine, mass_balance_params: Dict
    ):
        """Mass balance contribution analysis returns valid fractions."""
        contributions = engine.contribution_analysis(
            parameters=mass_balance_params,
            method="MASS_BALANCE",
        )
        assert len(contributions) > 0
        total = sum(contributions.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_contribution_analysis_screening(
        self, engine: UncertaintyQuantifierEngine, screening_params: Dict
    ):
        """Screening contribution analysis sums to 1."""
        contributions = engine.contribution_analysis(
            parameters=screening_params,
            method="SCREENING",
        )
        assert "leak_rate" in contributions
        total = sum(contributions.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_contribution_analysis_with_dqi_multiplier(
        self, engine: UncertaintyQuantifierEngine, equipment_params: Dict
    ):
        """DQI multiplier does not change the relative contribution fractions."""
        c1 = engine.contribution_analysis(
            parameters=equipment_params,
            method="EQUIPMENT_BASED",
            dqi_multiplier=Decimal("1.0"),
        )
        c2 = engine.contribution_analysis(
            parameters=equipment_params,
            method="EQUIPMENT_BASED",
            dqi_multiplier=Decimal("2.0"),
        )
        # Since multiplier scales all equally, fractions stay the same
        for key in c1:
            assert c1[key] == pytest.approx(c2[key], abs=0.001)


# ===========================================================================
# Test: Full Quantify (Integration of MC + Analytical + DQI)
# ===========================================================================


class TestQuantify:
    """Tests for the main quantify() entry point."""

    def test_quantify_returns_uncertainty_result(
        self, engine: UncertaintyQuantifierEngine, equipment_params: Dict
    ):
        """quantify() returns an UncertaintyResult dataclass."""
        result = engine.quantify(
            emissions_tco2e=Decimal("5.220"),
            method="EQUIPMENT_BASED",
            parameters=equipment_params,
            iterations=500,
            seed=42,
        )
        assert isinstance(result, UncertaintyResult)
        assert result.result_id.startswith("uq_")
        assert result.method == "EQUIPMENT_BASED"

    def test_std_dev_positive(
        self, engine: UncertaintyQuantifierEngine, equipment_params: Dict
    ):
        """Monte Carlo std dev is positive for valid input."""
        result = engine.quantify(
            emissions_tco2e=Decimal("5.220"),
            method="EQUIPMENT_BASED",
            parameters=equipment_params,
            iterations=500,
            seed=42,
        )
        assert result.monte_carlo_std is not None
        assert result.monte_carlo_std > Decimal("0")

    def test_quantify_combined_uncertainty_positive(
        self, engine: UncertaintyQuantifierEngine, equipment_params: Dict
    ):
        """Combined uncertainty percentage is positive."""
        result = engine.quantify(
            emissions_tco2e=Decimal("5.220"),
            method="EQUIPMENT_BASED",
            parameters=equipment_params,
            iterations=500,
            seed=42,
        )
        assert result.combined_uncertainty_pct > Decimal("0")

    def test_quantify_provenance_hash(
        self, engine: UncertaintyQuantifierEngine, equipment_params: Dict
    ):
        """Result includes a 64-char SHA-256 provenance hash."""
        result = engine.quantify(
            emissions_tco2e=Decimal("5.220"),
            method="EQUIPMENT_BASED",
            parameters=equipment_params,
            iterations=500,
            seed=42,
        )
        assert len(result.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in result.provenance_hash)

    def test_quantify_timestamp_present(
        self, engine: UncertaintyQuantifierEngine, equipment_params: Dict
    ):
        """Result includes a timestamp string."""
        result = engine.quantify(
            emissions_tco2e=Decimal("5.220"),
            method="EQUIPMENT_BASED",
            parameters=equipment_params,
            iterations=500,
            seed=42,
        )
        assert result.timestamp is not None
        assert len(result.timestamp) > 0

    def test_quantify_metadata_has_method_range(
        self, engine: UncertaintyQuantifierEngine, equipment_params: Dict
    ):
        """Metadata contains method_range_low and method_range_high."""
        result = engine.quantify(
            emissions_tco2e=Decimal("5.220"),
            method="EQUIPMENT_BASED",
            parameters=equipment_params,
            iterations=500,
            seed=42,
        )
        assert "method_range_low" in result.metadata
        assert "method_range_high" in result.metadata

    def test_quantify_records_history(
        self, engine: UncertaintyQuantifierEngine, equipment_params: Dict
    ):
        """quantify() appends to assessment history."""
        assert len(engine) == 0
        engine.quantify(
            emissions_tco2e=Decimal("5.220"),
            method="EQUIPMENT_BASED",
            parameters=equipment_params,
            iterations=200,
            seed=42,
        )
        assert len(engine) == 1
        engine.quantify(
            emissions_tco2e=Decimal("10.0"),
            method="MASS_BALANCE",
            parameters={"gwp": 1530.0, "beginning_inventory_kg": 100, "ending_inventory_kg": 80},
            iterations=200,
            seed=42,
        )
        assert len(engine) == 2

    def test_quantify_to_dict(
        self, engine: UncertaintyQuantifierEngine, equipment_params: Dict
    ):
        """UncertaintyResult.to_dict() produces a serializable dict."""
        result = engine.quantify(
            emissions_tco2e=Decimal("5.220"),
            method="EQUIPMENT_BASED",
            parameters=equipment_params,
            iterations=200,
            seed=42,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["method"] == "EQUIPMENT_BASED"
        assert d["result_id"].startswith("uq_")
        assert d["provenance_hash"] is not None

    def test_quantify_dqi_score_recorded(
        self, engine: UncertaintyQuantifierEngine, equipment_params: Dict
    ):
        """data_quality_score is set in the result."""
        result = engine.quantify(
            emissions_tco2e=Decimal("5.220"),
            method="EQUIPMENT_BASED",
            parameters=equipment_params,
            iterations=200,
            seed=42,
        )
        assert result.data_quality_score >= Decimal("1")
        assert result.data_quality_score <= Decimal("5")

    def test_quantify_iterations_recorded(
        self, engine: UncertaintyQuantifierEngine, equipment_params: Dict
    ):
        """Monte Carlo iteration count is captured in result."""
        result = engine.quantify(
            emissions_tco2e=Decimal("5.220"),
            method="EQUIPMENT_BASED",
            parameters=equipment_params,
            iterations=750,
            seed=42,
        )
        assert result.monte_carlo_iterations == 750

    def test_quantify_seed_recorded(
        self, engine: UncertaintyQuantifierEngine, equipment_params: Dict
    ):
        """Monte Carlo seed is captured in result."""
        result = engine.quantify(
            emissions_tco2e=Decimal("5.220"),
            method="EQUIPMENT_BASED",
            parameters=equipment_params,
            iterations=200,
            seed=9999,
        )
        assert result.monte_carlo_seed == 9999


# ===========================================================================
# Test: Edge Cases and Error Handling
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_emissions_input(
        self, engine: UncertaintyQuantifierEngine
    ):
        """Zero emissions produces a valid result without crashing."""
        result = engine.quantify(
            emissions_tco2e=Decimal("0"),
            method="EQUIPMENT_BASED",
            parameters={"charge_kg": 0, "leak_rate": 0, "gwp": 0},
            iterations=200,
            seed=42,
        )
        assert isinstance(result, UncertaintyResult)
        assert result.emissions_tco2e == Decimal("0")

    def test_negative_emissions_handled(
        self, engine: UncertaintyQuantifierEngine
    ):
        """Negative emissions_tco2e raises ValueError."""
        with pytest.raises(ValueError, match="emissions_tco2e must be >= 0"):
            engine.quantify(
                emissions_tco2e=Decimal("-1"),
                method="EQUIPMENT_BASED",
                parameters={},
                iterations=200,
                seed=42,
            )

    def test_invalid_method_raises_quantify(
        self, engine: UncertaintyQuantifierEngine
    ):
        """Invalid method string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown uncertainty method"):
            engine.quantify(
                emissions_tco2e=Decimal("10"),
                method="BOGUS_METHOD",
                parameters={},
                iterations=200,
                seed=42,
            )

    def test_iterations_below_minimum(
        self, engine: UncertaintyQuantifierEngine
    ):
        """Iterations below 100 raises ValueError."""
        with pytest.raises(ValueError, match="iterations must be in"):
            engine.quantify(
                emissions_tco2e=Decimal("10"),
                method="EQUIPMENT_BASED",
                parameters={},
                iterations=50,
                seed=42,
            )

    def test_iterations_above_maximum(
        self, engine: UncertaintyQuantifierEngine
    ):
        """Iterations above 100000 raises ValueError."""
        with pytest.raises(ValueError, match="iterations must be in"):
            engine.quantify(
                emissions_tco2e=Decimal("10"),
                method="EQUIPMENT_BASED",
                parameters={},
                iterations=200000,
                seed=42,
            )

    def test_quantify_accepts_float_emissions(
        self, engine: UncertaintyQuantifierEngine, equipment_params: Dict
    ):
        """quantify() auto-converts non-Decimal emissions to Decimal."""
        result = engine.quantify(
            emissions_tco2e=5.220,  # type: ignore[arg-type]
            method="EQUIPMENT_BASED",
            parameters=equipment_params,
            iterations=200,
            seed=42,
        )
        assert isinstance(result, UncertaintyResult)

    def test_quantify_none_parameters(
        self, engine: UncertaintyQuantifierEngine
    ):
        """None parameters defaults to empty dict without error."""
        result = engine.quantify(
            emissions_tco2e=Decimal("10"),
            method="EQUIPMENT_BASED",
            parameters=None,
            iterations=200,
            seed=42,
        )
        assert isinstance(result, UncertaintyResult)


# ===========================================================================
# Test: History
# ===========================================================================


class TestHistory:
    """Tests for assessment history management."""

    def test_get_history_all(
        self, engine: UncertaintyQuantifierEngine, equipment_params: Dict
    ):
        """get_history returns all entries when no filter applied."""
        for _ in range(3):
            engine.quantify(
                emissions_tco2e=Decimal("5.220"),
                method="EQUIPMENT_BASED",
                parameters=equipment_params,
                iterations=200,
                seed=42,
            )
        history = engine.get_history()
        assert len(history) == 3

    def test_get_history_by_method(
        self, engine: UncertaintyQuantifierEngine
    ):
        """get_history can filter by method."""
        engine.quantify(
            emissions_tco2e=Decimal("5"), method="EQUIPMENT_BASED",
            parameters={"charge_kg": 5, "leak_rate": 0.1, "gwp": 2088},
            iterations=200, seed=42,
        )
        engine.quantify(
            emissions_tco2e=Decimal("10"), method="SCREENING",
            parameters={"total_charge_kg": 50, "leak_rate": 0.15, "gwp": 2088},
            iterations=200, seed=42,
        )
        equip_only = engine.get_history(method="EQUIPMENT_BASED")
        assert len(equip_only) == 1
        assert equip_only[0].method == "EQUIPMENT_BASED"

    def test_get_history_with_limit(
        self, engine: UncertaintyQuantifierEngine, equipment_params: Dict
    ):
        """get_history respects limit parameter."""
        for _ in range(5):
            engine.quantify(
                emissions_tco2e=Decimal("5"), method="EQUIPMENT_BASED",
                parameters=equipment_params, iterations=200, seed=42,
            )
        limited = engine.get_history(limit=2)
        assert len(limited) == 2


# ===========================================================================
# Test: Parametrized across methods
# ===========================================================================


class TestParametrizedMethods:
    """Parametrized tests that run across all five methods."""

    @pytest.mark.parametrize(
        "method",
        [
            "EQUIPMENT_BASED",
            "MASS_BALANCE",
            "SCREENING",
            "DIRECT_MEASUREMENT",
            "TOP_DOWN",
        ],
    )
    def test_quantify_all_methods(
        self, engine: UncertaintyQuantifierEngine, method: str
    ):
        """quantify() succeeds for all five supported methods."""
        result = engine.quantify(
            emissions_tco2e=Decimal("10.0"),
            method=method,
            parameters={},
            iterations=200,
            seed=42,
        )
        assert isinstance(result, UncertaintyResult)
        assert result.method == method

    @pytest.mark.parametrize(
        "method",
        [
            "EQUIPMENT_BASED",
            "MASS_BALANCE",
            "SCREENING",
            "DIRECT_MEASUREMENT",
            "TOP_DOWN",
        ],
    )
    def test_analytical_all_methods(
        self, engine: UncertaintyQuantifierEngine, method: str
    ):
        """analytical_propagation() succeeds for all methods."""
        result = engine.analytical_propagation(
            emissions_tco2e=Decimal("10.0"),
            method=method,
            parameters={},
        )
        assert "combined_uncertainty_pct" in result

    @pytest.mark.parametrize(
        "method",
        [
            "EQUIPMENT_BASED",
            "MASS_BALANCE",
            "SCREENING",
            "DIRECT_MEASUREMENT",
            "TOP_DOWN",
        ],
    )
    def test_score_data_quality_all_methods(
        self, engine: UncertaintyQuantifierEngine, method: str
    ):
        """score_data_quality() succeeds for all methods."""
        score = engine.score_data_quality(method)
        assert 1.0 <= score <= 5.0
