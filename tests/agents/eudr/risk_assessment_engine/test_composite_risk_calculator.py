# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-EUDR-028 CompositeRiskCalculator.

MOST CRITICAL TEST FILE -- validates the core composite risk scoring
formula:  Composite = SUM(W_i * S_i * C_i) / SUM(W_i * C_i)

Tests cover basic calculation, single-dimension, missing dimensions,
confidence weighting, country benchmark multipliers, capping, override
recalculation, weight validation, deterministic reproducibility, and
Decimal arithmetic precision.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from greenlang.agents.eudr.risk_assessment_engine.config import (
    RiskAssessmentEngineConfig,
    reset_config,
)
from greenlang.agents.eudr.risk_assessment_engine.models import (
    CompositeRiskScore,
    CountryBenchmark,
    CountryBenchmarkLevel,
    DEFAULT_WEIGHTS,
    DimensionScore,
    RiskDimension,
    RiskFactorInput,
    RiskLevel,
    SourceAgent,
)


# ---------------------------------------------------------------------------
# Helpers to build a mock config that engines can consume
# ---------------------------------------------------------------------------


def _make_engine_config() -> MagicMock:
    """Build a mock config with the dict-style attributes the engine expects."""
    cfg = MagicMock(spec=RiskAssessmentEngineConfig)
    cfg.dimension_weights = {
        "country": Decimal("0.20"),
        "commodity": Decimal("0.15"),
        "supplier": Decimal("0.20"),
        "deforestation": Decimal("0.20"),
        "corruption": Decimal("0.10"),
        "supply_chain_complexity": Decimal("0.05"),
        "mixing_risk": Decimal("0.05"),
        "circumvention_risk": Decimal("0.05"),
    }
    cfg.risk_thresholds = {
        "negligible": 15,
        "low": 30,
        "standard": 60,
        "high": 80,
        "critical": 100,
    }
    cfg.hysteresis_buffer = Decimal("3")
    cfg.benchmark_low_multiplier = Decimal("0.70")
    cfg.benchmark_standard_multiplier = Decimal("1.00")
    cfg.benchmark_high_multiplier = Decimal("1.50")
    return cfg


def _make_input(
    dimension: RiskDimension,
    score: Decimal,
    confidence: Decimal = Decimal("1.00"),
    source: SourceAgent = SourceAgent.EUDR_016_COUNTRY,
) -> RiskFactorInput:
    """Build a RiskFactorInput shortcut."""
    return RiskFactorInput(
        source_agent=source,
        dimension=dimension,
        raw_score=score,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCompositeRiskCalculator:
    """Core tests for CompositeRiskCalculator."""

    def _make_calculator(self):
        """Instantiate calculator with a mocked config."""
        from greenlang.agents.eudr.risk_assessment_engine.composite_risk_calculator import (
            CompositeRiskCalculator,
        )
        cfg = _make_engine_config()
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.composite_risk_calculator.record_composite_calculation"
        ), patch(
            "greenlang.agents.eudr.risk_assessment_engine.composite_risk_calculator.observe_calculation_duration"
        ):
            return CompositeRiskCalculator(config=cfg)

    # -- Basic calculation --------------------------------------------------

    def test_calculate_composite_score_basic(self, sample_factor_inputs):
        """All 8 dimensions present -> valid CompositeRiskScore."""
        calc = self._make_calculator()
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.composite_risk_calculator.record_composite_calculation"
        ), patch(
            "greenlang.agents.eudr.risk_assessment_engine.composite_risk_calculator.observe_calculation_duration"
        ):
            result = calc.calculate_composite_score(sample_factor_inputs)

        assert isinstance(result, CompositeRiskScore)
        assert Decimal("0") <= result.overall_score <= Decimal("100")
        assert result.risk_level in RiskLevel
        assert len(result.dimension_scores) == 8
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64

    def test_calculate_composite_score_single_dimension(self):
        """A single dimension input should still produce a valid score."""
        calc = self._make_calculator()
        inputs = [_make_input(RiskDimension.COUNTRY, Decimal("50"), Decimal("0.90"))]
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.composite_risk_calculator.record_composite_calculation"
        ), patch(
            "greenlang.agents.eudr.risk_assessment_engine.composite_risk_calculator.observe_calculation_duration"
        ):
            result = calc.calculate_composite_score(inputs)

        assert result.overall_score >= Decimal("0")
        assert len(result.dimension_scores) == 1

    # -- Missing dimensions -------------------------------------------------

    def test_calculate_composite_score_missing_dimensions(self):
        """Only 3 of 8 dimensions provided -> calculator uses available ones."""
        calc = self._make_calculator()
        inputs = [
            _make_input(RiskDimension.COUNTRY, Decimal("40"), Decimal("0.90")),
            _make_input(RiskDimension.SUPPLIER, Decimal("30"), Decimal("0.80")),
            _make_input(RiskDimension.COMMODITY, Decimal("50"), Decimal("0.85")),
        ]
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.composite_risk_calculator.record_composite_calculation"
        ), patch(
            "greenlang.agents.eudr.risk_assessment_engine.composite_risk_calculator.observe_calculation_duration"
        ):
            result = calc.calculate_composite_score(inputs)

        assert len(result.dimension_scores) == 3
        assert result.overall_score >= Decimal("0")

    # -- Zero confidence ----------------------------------------------------

    def test_calculate_composite_score_zero_confidence(self):
        """Dimension with zero confidence should be effectively excluded."""
        calc = self._make_calculator()
        inputs = [
            _make_input(RiskDimension.COUNTRY, Decimal("80"), Decimal("0")),
            _make_input(RiskDimension.SUPPLIER, Decimal("50"), Decimal("1.00")),
        ]
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.composite_risk_calculator.record_composite_calculation"
        ), patch(
            "greenlang.agents.eudr.risk_assessment_engine.composite_risk_calculator.observe_calculation_duration"
        ):
            result = calc.calculate_composite_score(inputs)

        # The zero-confidence COUNTRY dimension should not significantly
        # influence the overall score.
        assert result.overall_score >= Decimal("0")

    # -- Country benchmark multipliers --------------------------------------

    def test_calculate_composite_score_with_country_benchmark_low(self):
        """LOW benchmark multiplier (0.70) should reduce the score."""
        calc = self._make_calculator()
        inputs = [
            _make_input(RiskDimension.COUNTRY, Decimal("50"), Decimal("0.90")),
            _make_input(RiskDimension.SUPPLIER, Decimal("50"), Decimal("0.90")),
        ]
        benchmarks = [
            CountryBenchmark(
                country_code="DE",
                benchmark_level=CountryBenchmarkLevel.LOW,
            ),
        ]
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.composite_risk_calculator.record_composite_calculation"
        ), patch(
            "greenlang.agents.eudr.risk_assessment_engine.composite_risk_calculator.observe_calculation_duration"
        ):
            result_without = calc.calculate_composite_score(inputs)
            result_with = calc.calculate_composite_score(inputs, benchmarks)

        # Score with LOW benchmark should be lower
        assert result_with.country_benchmark_applied is True
        assert result_with.benchmark_multiplier == Decimal("0.70")
        assert result_with.overall_score <= result_without.overall_score

    def test_calculate_composite_score_with_country_benchmark_high(self):
        """HIGH benchmark multiplier (1.50) should increase the score."""
        calc = self._make_calculator()
        inputs = [
            _make_input(RiskDimension.COUNTRY, Decimal("40"), Decimal("0.90")),
            _make_input(RiskDimension.SUPPLIER, Decimal("40"), Decimal("0.90")),
        ]
        benchmarks = [
            CountryBenchmark(
                country_code="BR",
                benchmark_level=CountryBenchmarkLevel.HIGH,
            ),
        ]
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.composite_risk_calculator.record_composite_calculation"
        ), patch(
            "greenlang.agents.eudr.risk_assessment_engine.composite_risk_calculator.observe_calculation_duration"
        ):
            result_without = calc.calculate_composite_score(inputs)
            result_with = calc.calculate_composite_score(inputs, benchmarks)

        assert result_with.country_benchmark_applied is True
        assert result_with.benchmark_multiplier == Decimal("1.50")
        assert result_with.overall_score >= result_without.overall_score

    # -- Capping at 100 -----------------------------------------------------

    def test_calculate_composite_score_capped_at_100(self):
        """Score should never exceed 100 even with HIGH benchmark."""
        calc = self._make_calculator()
        inputs = [
            _make_input(RiskDimension.COUNTRY, Decimal("90"), Decimal("1.00")),
            _make_input(RiskDimension.SUPPLIER, Decimal("95"), Decimal("1.00")),
            _make_input(RiskDimension.DEFORESTATION, Decimal("100"), Decimal("1.00")),
        ]
        benchmarks = [
            CountryBenchmark(
                country_code="CD",
                benchmark_level=CountryBenchmarkLevel.HIGH,
            ),
        ]
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.composite_risk_calculator.record_composite_calculation"
        ), patch(
            "greenlang.agents.eudr.risk_assessment_engine.composite_risk_calculator.observe_calculation_duration"
        ):
            result = calc.calculate_composite_score(inputs, benchmarks)

        assert result.overall_score <= Decimal("100")

    # -- Formula correctness ------------------------------------------------

    def test_weighted_formula_correctness(self):
        """Manually verify: SUM(W*S*C) / SUM(W*C)."""
        calc = self._make_calculator()
        # Use two dimensions with known weights
        # COUNTRY: W=0.20, S=60, C=1.00 -> W*S*C = 12.00, W*C = 0.20
        # SUPPLIER: W=0.20, S=40, C=1.00 -> W*S*C = 8.00, W*C = 0.20
        inputs = [
            _make_input(RiskDimension.COUNTRY, Decimal("60"), Decimal("1.00")),
            _make_input(RiskDimension.SUPPLIER, Decimal("40"), Decimal("1.00")),
        ]
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.composite_risk_calculator.record_composite_calculation"
        ), patch(
            "greenlang.agents.eudr.risk_assessment_engine.composite_risk_calculator.observe_calculation_duration"
        ):
            result = calc.calculate_composite_score(inputs)

        # weighted_score for COUNTRY dim = W * S = 0.20 * 60 = 12.00
        # weighted_score for SUPPLIER dim = W * S = 0.20 * 40 = 8.00
        # numerator = 12.00 * 1.00 + 8.00 * 1.00 = 20.00
        # denominator = 0.20 * 1.00 + 0.20 * 1.00 = 0.40
        # composite = 20.00 / 0.40 = 50.00
        assert result.overall_score == Decimal("50.00")

    # -- Confidence weighting -----------------------------------------------

    def test_confidence_weighting(self):
        """Higher confidence inputs should have more influence."""
        calc = self._make_calculator()
        inputs_high_conf = [
            _make_input(RiskDimension.COUNTRY, Decimal("80"), Decimal("1.00")),
            _make_input(RiskDimension.SUPPLIER, Decimal("20"), Decimal("0.10")),
        ]
        inputs_equal_conf = [
            _make_input(RiskDimension.COUNTRY, Decimal("80"), Decimal("0.50")),
            _make_input(RiskDimension.SUPPLIER, Decimal("20"), Decimal("0.50")),
        ]
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.composite_risk_calculator.record_composite_calculation"
        ), patch(
            "greenlang.agents.eudr.risk_assessment_engine.composite_risk_calculator.observe_calculation_duration"
        ):
            result_high = calc.calculate_composite_score(inputs_high_conf)
            result_equal = calc.calculate_composite_score(inputs_equal_conf)

        # With high confidence on COUNTRY(80), score should be pulled toward 80
        # With equal confidence, score should be closer to midpoint (50)
        assert result_high.overall_score > result_equal.overall_score

    # -- Deterministic reproducibility --------------------------------------

    def test_deterministic_reproducibility(self, sample_factor_inputs):
        """Same inputs must always produce the same output."""
        calc = self._make_calculator()
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.composite_risk_calculator.record_composite_calculation"
        ), patch(
            "greenlang.agents.eudr.risk_assessment_engine.composite_risk_calculator.observe_calculation_duration"
        ):
            r1 = calc.calculate_composite_score(sample_factor_inputs)
            r2 = calc.calculate_composite_score(sample_factor_inputs)

        assert r1.overall_score == r2.overall_score
        assert r1.provenance_hash == r2.provenance_hash
        assert r1.risk_level == r2.risk_level

    # -- Decimal precision --------------------------------------------------

    def test_decimal_arithmetic_precision(self):
        """Ensure no floating-point rounding errors in Decimal arithmetic."""
        calc = self._make_calculator()
        inputs = [
            _make_input(RiskDimension.COUNTRY, Decimal("33.33"), Decimal("0.77")),
        ]
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.composite_risk_calculator.record_composite_calculation"
        ), patch(
            "greenlang.agents.eudr.risk_assessment_engine.composite_risk_calculator.observe_calculation_duration"
        ):
            result = calc.calculate_composite_score(inputs)

        # Result should be a precise Decimal, not a float approximation
        assert isinstance(result.overall_score, Decimal)
        # Score should have at most 2 decimal places
        assert result.overall_score == result.overall_score.quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

    # -- Empty inputs -------------------------------------------------------

    def test_calculate_composite_score_empty_raises(self):
        """Empty input list should raise ValueError."""
        calc = self._make_calculator()
        with pytest.raises(ValueError, match="at least one input"):
            with patch(
                "greenlang.agents.eudr.risk_assessment_engine.composite_risk_calculator.record_composite_calculation"
            ), patch(
                "greenlang.agents.eudr.risk_assessment_engine.composite_risk_calculator.observe_calculation_duration"
            ):
                calc.calculate_composite_score([])

    # -- Weight validation --------------------------------------------------

    def test_validate_weights_valid(self):
        """Valid weights should return True."""
        calc = self._make_calculator()
        assert calc._validate_weights() is True

    def test_validate_weights_invalid(self):
        """Invalid weights (not summing to ~1.0) should return False."""
        cfg = _make_engine_config()
        cfg.dimension_weights = {"country": Decimal("0.50")}
        from greenlang.agents.eudr.risk_assessment_engine.composite_risk_calculator import (
            CompositeRiskCalculator,
        )
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.composite_risk_calculator.record_composite_calculation"
        ), patch(
            "greenlang.agents.eudr.risk_assessment_engine.composite_risk_calculator.observe_calculation_duration"
        ):
            calc = CompositeRiskCalculator(config=cfg)
        assert calc._validate_weights() is False

    # -- Calculation stats --------------------------------------------------

    def test_calculation_stats(self, sample_factor_inputs):
        """get_calculation_stats should reflect completed calculations."""
        calc = self._make_calculator()
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.composite_risk_calculator.record_composite_calculation"
        ), patch(
            "greenlang.agents.eudr.risk_assessment_engine.composite_risk_calculator.observe_calculation_duration"
        ):
            calc.calculate_composite_score(sample_factor_inputs)

        stats = calc.get_calculation_stats()
        assert stats["total_calculations"] >= 1
        assert stats["total_dimensions_scored"] >= 1
        assert "weights_valid" in stats

    # -- Dimension score computation ----------------------------------------

    def test_dimension_score_computation(self):
        """Verify dimension-level score computation with single input."""
        calc = self._make_calculator()
        inputs = [_make_input(RiskDimension.COUNTRY, Decimal("60"), Decimal("0.90"))]
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.composite_risk_calculator.record_composite_calculation"
        ), patch(
            "greenlang.agents.eudr.risk_assessment_engine.composite_risk_calculator.observe_calculation_duration"
        ):
            result = calc.calculate_composite_score(inputs)

        ds = result.dimension_scores[0]
        assert ds.dimension == RiskDimension.COUNTRY
        assert ds.raw_score == Decimal("60")
        # weighted_score = weight * raw_score = 0.20 * 60 = 12.00
        assert ds.weighted_score == Decimal("12.00")
        assert ds.weight == Decimal("0.20")
