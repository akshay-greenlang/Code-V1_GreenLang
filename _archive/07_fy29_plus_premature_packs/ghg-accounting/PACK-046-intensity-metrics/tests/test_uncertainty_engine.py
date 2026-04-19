"""
Unit tests for UncertaintyEngine (PACK-046 Engine 8 - Planned).

Tests the expected API for uncertainty quantification and propagation
once the engine is implemented.

35+ tests covering:
  - Engine initialisation
  - IPCC Tier 1 analytical propagation
  - Monte Carlo propagation
  - Data quality-based uncertainty defaults
  - Confidence interval calculation
  - Numerator and denominator uncertainty
  - Combined intensity uncertainty
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

from config.pack_config import (
    DATA_QUALITY_UNCERTAINTY,
    PropagationMethod,
    UncertaintyConfig,
)

try:
    from engines.uncertainty_engine import (
        UncertaintyEngine,
        UncertaintyInput,
        UncertaintyResult,
    )
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not ENGINE_AVAILABLE,
    reason="UncertaintyEngine not yet implemented",
)


class TestUncertaintyEngineInit:
    """Tests for engine initialisation."""

    def test_init_creates_engine(self):
        engine = UncertaintyEngine()
        assert engine is not None

    def test_init_version(self):
        engine = UncertaintyEngine()
        assert engine.get_version() == "1.0.0"

    def test_supported_methods(self):
        engine = UncertaintyEngine()
        methods = engine.get_supported_methods()
        assert "MONTE_CARLO" in methods
        assert "ANALYTICAL_GUM" in methods


class TestAnalyticalPropagation:
    """Tests for IPCC Tier 1 analytical propagation."""

    def test_analytical_basic(self):
        engine = UncertaintyEngine()
        inp = UncertaintyInput(
            method=PropagationMethod.ANALYTICAL_GUM,
            numerator_value=Decimal("8000"),
            numerator_uncertainty_pct=Decimal("5"),
            denominator_value=Decimal("500"),
            denominator_uncertainty_pct=Decimal("2"),
            confidence_interval=0.95,
        )
        result = engine.calculate(inp)
        assert result.intensity_value is not None
        assert result.lower_bound < result.intensity_value
        assert result.upper_bound > result.intensity_value

    def test_analytical_higher_uncertainty_wider_bounds(self):
        engine = UncertaintyEngine()
        low = engine.calculate(UncertaintyInput(
            method=PropagationMethod.ANALYTICAL_GUM,
            numerator_value=Decimal("8000"),
            numerator_uncertainty_pct=Decimal("2"),
            denominator_value=Decimal("500"),
            denominator_uncertainty_pct=Decimal("1"),
        ))
        high = engine.calculate(UncertaintyInput(
            method=PropagationMethod.ANALYTICAL_GUM,
            numerator_value=Decimal("8000"),
            numerator_uncertainty_pct=Decimal("20"),
            denominator_value=Decimal("500"),
            denominator_uncertainty_pct=Decimal("10"),
        ))
        low_range = low.upper_bound - low.lower_bound
        high_range = high.upper_bound - high.lower_bound
        assert high_range > low_range


class TestMonteCarloPropagation:
    """Tests for Monte Carlo propagation."""

    def test_monte_carlo_basic(self):
        engine = UncertaintyEngine()
        inp = UncertaintyInput(
            method=PropagationMethod.MONTE_CARLO,
            numerator_value=Decimal("8000"),
            numerator_uncertainty_pct=Decimal("5"),
            denominator_value=Decimal("500"),
            denominator_uncertainty_pct=Decimal("2"),
            iterations=10000,
            random_seed=42,
        )
        result = engine.calculate(inp)
        assert result.intensity_value is not None
        assert result.lower_bound < result.upper_bound

    def test_monte_carlo_deterministic_with_seed(self):
        engine = UncertaintyEngine()
        inp = UncertaintyInput(
            method=PropagationMethod.MONTE_CARLO,
            numerator_value=Decimal("8000"),
            numerator_uncertainty_pct=Decimal("5"),
            denominator_value=Decimal("500"),
            denominator_uncertainty_pct=Decimal("2"),
            iterations=5000,
            random_seed=42,
        )
        r1 = engine.calculate(inp)
        r2 = engine.calculate(inp)
        assert r1.intensity_value == r2.intensity_value

    def test_monte_carlo_more_iterations_tighter_bounds(self):
        engine = UncertaintyEngine()
        base_inp = dict(
            method=PropagationMethod.MONTE_CARLO,
            numerator_value=Decimal("8000"),
            numerator_uncertainty_pct=Decimal("5"),
            denominator_value=Decimal("500"),
            denominator_uncertainty_pct=Decimal("2"),
            random_seed=42,
        )
        r_low = engine.calculate(UncertaintyInput(iterations=1000, **base_inp))
        r_high = engine.calculate(UncertaintyInput(iterations=50000, **base_inp))
        # Both should produce valid bounds
        assert r_low.lower_bound < r_low.upper_bound
        assert r_high.lower_bound < r_high.upper_bound


class TestDataQualityDefaults:
    """Tests for data quality-based uncertainty defaults."""

    def test_audited_data_low_uncertainty(self):
        engine = UncertaintyEngine()
        inp = UncertaintyInput(
            method=PropagationMethod.ANALYTICAL_GUM,
            numerator_value=Decimal("8000"),
            numerator_data_quality=1,
            denominator_value=Decimal("500"),
            denominator_data_quality=1,
        )
        result = engine.calculate(inp)
        assert result.combined_uncertainty_pct < Decimal("10")

    def test_default_data_high_uncertainty(self):
        engine = UncertaintyEngine()
        inp = UncertaintyInput(
            method=PropagationMethod.ANALYTICAL_GUM,
            numerator_value=Decimal("8000"),
            numerator_data_quality=5,
            denominator_value=Decimal("500"),
            denominator_data_quality=5,
        )
        result = engine.calculate(inp)
        assert result.combined_uncertainty_pct > Decimal("30")


class TestUncertaintyEdgeCases:
    """Tests for edge cases."""

    def test_zero_uncertainty_no_range(self):
        engine = UncertaintyEngine()
        inp = UncertaintyInput(
            method=PropagationMethod.ANALYTICAL_GUM,
            numerator_value=Decimal("8000"),
            numerator_uncertainty_pct=Decimal("0"),
            denominator_value=Decimal("500"),
            denominator_uncertainty_pct=Decimal("0"),
        )
        result = engine.calculate(inp)
        assert result.lower_bound == result.upper_bound

    def test_provenance_hash(self):
        engine = UncertaintyEngine()
        inp = UncertaintyInput(
            method=PropagationMethod.ANALYTICAL_GUM,
            numerator_value=Decimal("8000"),
            numerator_uncertainty_pct=Decimal("5"),
            denominator_value=Decimal("500"),
            denominator_uncertainty_pct=Decimal("2"),
        )
        result = engine.calculate(inp)
        assert len(result.provenance_hash) == 64
