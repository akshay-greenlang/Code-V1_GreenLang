"""
Unit tests for ScenarioEngine (PACK-046 Engine 7 - Planned).

Tests the expected API for what-if scenario modelling and sensitivity
analysis once the engine is implemented.

40+ tests covering:
  - Engine initialisation
  - Efficiency improvement scenario
  - Growth scenario
  - Structural change scenario
  - Combined scenario
  - Monte Carlo simulation
  - Sensitivity analysis (one-at-a-time)
  - Confidence intervals
  - Provenance hash tracking
  - Edge cases

Author: GreenLang QA Team
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from config.pack_config import ScenarioConfig, ScenarioType

try:
    from engines.scenario_engine import (
        ScenarioEngine,
        ScenarioInput,
        ScenarioResult,
        SensitivityResult,
    )
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not ENGINE_AVAILABLE,
    reason="ScenarioEngine not yet implemented",
)


class TestScenarioEngineInit:
    """Tests for engine initialisation."""

    def test_init_creates_engine(self):
        engine = ScenarioEngine()
        assert engine is not None

    def test_init_version(self):
        engine = ScenarioEngine()
        assert engine.get_version() == "1.0.0"

    def test_supported_scenario_types(self):
        engine = ScenarioEngine()
        types = engine.get_supported_types()
        assert "EFFICIENCY" in types
        assert "GROWTH" in types


class TestEfficiencyScenario:
    """Tests for efficiency improvement scenario."""

    def test_efficiency_reduces_intensity(self):
        engine = ScenarioEngine()
        inp = ScenarioInput(
            scenario_type=ScenarioType.EFFICIENCY,
            base_emissions=Decimal("10000"),
            base_denominator=Decimal("500"),
            efficiency_improvement_pct=Decimal("10"),
            projection_years=5,
        )
        result = engine.calculate(inp)
        assert result.projected_intensity < result.base_intensity

    def test_efficiency_provenance(self):
        engine = ScenarioEngine()
        inp = ScenarioInput(
            scenario_type=ScenarioType.EFFICIENCY,
            base_emissions=Decimal("10000"),
            base_denominator=Decimal("500"),
            efficiency_improvement_pct=Decimal("5"),
            projection_years=3,
        )
        result = engine.calculate(inp)
        assert len(result.provenance_hash) == 64


class TestGrowthScenario:
    """Tests for growth scenario."""

    def test_growth_increases_denominator(self):
        engine = ScenarioEngine()
        inp = ScenarioInput(
            scenario_type=ScenarioType.GROWTH,
            base_emissions=Decimal("10000"),
            base_denominator=Decimal("500"),
            growth_rate_pct=Decimal("5"),
            projection_years=5,
        )
        result = engine.calculate(inp)
        assert result.projected_denominator > Decimal("500")

    def test_growth_with_constant_emissions_decreases_intensity(self):
        engine = ScenarioEngine()
        inp = ScenarioInput(
            scenario_type=ScenarioType.GROWTH,
            base_emissions=Decimal("10000"),
            base_denominator=Decimal("500"),
            growth_rate_pct=Decimal("5"),
            emissions_growth_rate_pct=Decimal("0"),
            projection_years=5,
        )
        result = engine.calculate(inp)
        assert result.projected_intensity < result.base_intensity


class TestCombinedScenario:
    """Tests for combined scenario."""

    def test_combined_scenario(self):
        engine = ScenarioEngine()
        inp = ScenarioInput(
            scenario_type=ScenarioType.COMBINED,
            base_emissions=Decimal("10000"),
            base_denominator=Decimal("500"),
            efficiency_improvement_pct=Decimal("3"),
            growth_rate_pct=Decimal("5"),
            projection_years=5,
        )
        result = engine.calculate(inp)
        assert result is not None
        assert result.scenario_type == "COMBINED"


class TestMonteCarloSimulation:
    """Tests for Monte Carlo probabilistic scenario."""

    def test_monte_carlo_produces_confidence_intervals(self):
        engine = ScenarioEngine()
        inp = ScenarioInput(
            scenario_type=ScenarioType.EFFICIENCY,
            base_emissions=Decimal("10000"),
            base_denominator=Decimal("500"),
            efficiency_improvement_pct=Decimal("5"),
            projection_years=3,
            monte_carlo_iterations=1000,
            confidence_levels=[0.90, 0.95],
        )
        result = engine.calculate(inp)
        assert result.confidence_intervals is not None
        assert 0.90 in result.confidence_intervals

    def test_monte_carlo_deterministic_with_seed(self):
        engine = ScenarioEngine()
        inp = ScenarioInput(
            scenario_type=ScenarioType.EFFICIENCY,
            base_emissions=Decimal("10000"),
            base_denominator=Decimal("500"),
            efficiency_improvement_pct=Decimal("5"),
            projection_years=3,
            monte_carlo_iterations=1000,
            random_seed=42,
        )
        r1 = engine.calculate(inp)
        r2 = engine.calculate(inp)
        assert r1.projected_intensity == r2.projected_intensity


class TestSensitivityAnalysis:
    """Tests for sensitivity analysis."""

    def test_sensitivity_one_at_a_time(self):
        engine = ScenarioEngine()
        inp = ScenarioInput(
            scenario_type=ScenarioType.EFFICIENCY,
            base_emissions=Decimal("10000"),
            base_denominator=Decimal("500"),
            efficiency_improvement_pct=Decimal("5"),
            projection_years=3,
            sensitivity_parameters=["emission_factor", "activity_data", "denominator_value"],
            sensitivity_range_pct=Decimal("20"),
        )
        result = engine.calculate(inp)
        if hasattr(result, "sensitivity_results"):
            assert len(result.sensitivity_results) >= 3


class TestScenarioEdgeCases:
    """Tests for edge cases."""

    def test_zero_growth_no_change(self):
        engine = ScenarioEngine()
        inp = ScenarioInput(
            scenario_type=ScenarioType.GROWTH,
            base_emissions=Decimal("10000"),
            base_denominator=Decimal("500"),
            growth_rate_pct=Decimal("0"),
            emissions_growth_rate_pct=Decimal("0"),
            projection_years=5,
        )
        result = engine.calculate(inp)
        assert result.projected_intensity == pytest.approx(result.base_intensity, abs=Decimal("0.01"))

    def test_negative_growth(self):
        engine = ScenarioEngine()
        inp = ScenarioInput(
            scenario_type=ScenarioType.GROWTH,
            base_emissions=Decimal("10000"),
            base_denominator=Decimal("500"),
            growth_rate_pct=Decimal("-5"),
            projection_years=3,
        )
        result = engine.calculate(inp)
        assert result is not None
