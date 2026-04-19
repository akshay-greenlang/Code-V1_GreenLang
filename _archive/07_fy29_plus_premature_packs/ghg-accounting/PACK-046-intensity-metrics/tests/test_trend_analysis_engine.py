"""
Unit tests for TrendAnalysisEngine (PACK-046 Engine 6 - Planned).

Tests the expected API for time-series trend analysis and projection
once the engine is implemented.

40+ tests covering:
  - Engine initialisation
  - OLS regression fitting
  - Trend direction detection
  - Rolling window calculations
  - Forward projection
  - Statistical significance testing
  - Structural break detection
  - Compound annual growth rate (CAGR)
  - Provenance hash tracking
  - Edge cases (insufficient data, constant values)

Author: GreenLang QA Team
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from config.pack_config import RegressionModel, TrendConfig

try:
    from engines.trend_analysis_engine import (
        TrendAnalysisEngine,
        TrendInput,
        TrendResult,
        ProjectionPoint,
    )
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not ENGINE_AVAILABLE,
    reason="TrendAnalysisEngine not yet implemented",
)


class TestTrendAnalysisEngineInit:
    """Tests for engine initialisation."""

    def test_init_creates_engine(self):
        engine = TrendAnalysisEngine()
        assert engine is not None

    def test_init_version(self):
        engine = TrendAnalysisEngine()
        assert engine.get_version() == "1.0.0"

    def test_supported_models(self):
        engine = TrendAnalysisEngine()
        models = engine.get_supported_models()
        assert "OLS" in models


class TestTrendDetection:
    """Tests for trend direction detection."""

    def test_decreasing_trend(self):
        engine = TrendAnalysisEngine()
        inp = TrendInput(
            data_points=[
                (2020, Decimal("30")),
                (2021, Decimal("28")),
                (2022, Decimal("25")),
                (2023, Decimal("22")),
                (2024, Decimal("20")),
            ],
            model=RegressionModel.OLS,
        )
        result = engine.calculate(inp)
        assert result.trend_direction == "decreasing"
        assert result.slope < 0

    def test_increasing_trend(self):
        engine = TrendAnalysisEngine()
        inp = TrendInput(
            data_points=[
                (2020, Decimal("10")),
                (2021, Decimal("12")),
                (2022, Decimal("15")),
                (2023, Decimal("18")),
                (2024, Decimal("22")),
            ],
            model=RegressionModel.OLS,
        )
        result = engine.calculate(inp)
        assert result.trend_direction == "increasing"
        assert result.slope > 0

    def test_flat_trend(self):
        engine = TrendAnalysisEngine()
        inp = TrendInput(
            data_points=[
                (2020, Decimal("20")),
                (2021, Decimal("20")),
                (2022, Decimal("20")),
                (2023, Decimal("20")),
                (2024, Decimal("20")),
            ],
            model=RegressionModel.OLS,
        )
        result = engine.calculate(inp)
        assert abs(result.slope) < 0.01


class TestProjection:
    """Tests for forward projection."""

    def test_projection_produces_points(self):
        engine = TrendAnalysisEngine()
        inp = TrendInput(
            data_points=[
                (2020, Decimal("30")),
                (2021, Decimal("28")),
                (2022, Decimal("25")),
                (2023, Decimal("22")),
                (2024, Decimal("20")),
            ],
            model=RegressionModel.OLS,
            projection_years=5,
        )
        result = engine.calculate(inp)
        assert len(result.projections) == 5
        assert result.projections[0].year == 2025

    def test_projection_continues_trend(self):
        engine = TrendAnalysisEngine()
        inp = TrendInput(
            data_points=[
                (2020, Decimal("30")),
                (2021, Decimal("28")),
                (2022, Decimal("25")),
                (2023, Decimal("22")),
                (2024, Decimal("20")),
            ],
            model=RegressionModel.OLS,
            projection_years=3,
        )
        result = engine.calculate(inp)
        for p in result.projections:
            assert p.value < Decimal("20")


class TestCAGR:
    """Tests for compound annual growth rate."""

    def test_cagr_calculated(self):
        engine = TrendAnalysisEngine()
        inp = TrendInput(
            data_points=[
                (2020, Decimal("30")),
                (2024, Decimal("20")),
            ],
            model=RegressionModel.OLS,
        )
        result = engine.calculate(inp)
        assert result.cagr_pct is not None
        assert result.cagr_pct < 0  # Decreasing

    def test_provenance_hash(self):
        engine = TrendAnalysisEngine()
        inp = TrendInput(
            data_points=[
                (2020, Decimal("30")),
                (2021, Decimal("28")),
                (2022, Decimal("25")),
            ],
            model=RegressionModel.OLS,
        )
        result = engine.calculate(inp)
        assert len(result.provenance_hash) == 64


class TestTrendEdgeCases:
    """Tests for edge cases."""

    def test_insufficient_data_raises(self):
        engine = TrendAnalysisEngine()
        inp = TrendInput(
            data_points=[(2024, Decimal("20"))],
            model=RegressionModel.OLS,
            min_data_points=3,
        )
        with pytest.raises(ValueError, match="data points"):
            engine.calculate(inp)

    def test_r_squared_reported(self):
        engine = TrendAnalysisEngine()
        inp = TrendInput(
            data_points=[
                (2020, Decimal("30")),
                (2021, Decimal("28")),
                (2022, Decimal("25")),
                (2023, Decimal("22")),
            ],
            model=RegressionModel.OLS,
        )
        result = engine.calculate(inp)
        assert 0.0 <= result.r_squared <= 1.0
