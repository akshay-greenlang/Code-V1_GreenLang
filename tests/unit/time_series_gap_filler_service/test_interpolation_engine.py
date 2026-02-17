# -*- coding: utf-8 -*-
"""
Unit tests for InterpolationEngine - AGENT-DATA-014 Time Series Gap Filler

Tests all six interpolation methods and the fill_gaps dispatcher:
    - interpolate_linear
    - interpolate_cubic_spline
    - interpolate_polynomial
    - interpolate_akima
    - interpolate_nearest
    - interpolate_pchip
    - fill_gaps (dispatcher)

Also tests confidence scoring, provenance hashing, edge-gap handling,
empty series, all-missing series, and single-value series.

Target: 60+ tests.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-014 Time Series Gap Filler (GL-DATA-X-017)
"""

from __future__ import annotations

import math
from typing import List, Optional

import pytest

from greenlang.time_series_gap_filler.config import TimeSeriesGapFillerConfig
from greenlang.time_series_gap_filler.interpolation_engine import (
    FillResult,
    InterpolationEngine,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine(fresh_config) -> InterpolationEngine:
    """Create an InterpolationEngine with default test config."""
    return InterpolationEngine(config=fresh_config)


@pytest.fixture
def simple_gap() -> List[Optional[float]]:
    """[1, None, 3] -- single interior gap of length 1."""
    return [1.0, None, 3.0]


@pytest.fixture
def double_gap() -> List[Optional[float]]:
    """[1, None, None, 4] -- interior gap of length 2."""
    return [1.0, None, None, 4.0]


@pytest.fixture
def no_gap() -> List[float]:
    """[1, 2, 3, 4, 5] -- no gaps."""
    return [1.0, 2.0, 3.0, 4.0, 5.0]


@pytest.fixture
def leading_gap() -> List[Optional[float]]:
    """[None, None, 3, 4, 5] -- gap at the start."""
    return [None, None, 3.0, 4.0, 5.0]


@pytest.fixture
def trailing_gap() -> List[Optional[float]]:
    """[1, 2, 3, None, None] -- gap at the end."""
    return [1.0, 2.0, 3.0, None, None]


@pytest.fixture
def all_missing() -> List[Optional[float]]:
    """[None, None, None, None, None] -- entirely missing."""
    return [None, None, None, None, None]


@pytest.fixture
def single_known() -> List[Optional[float]]:
    """[None, None, 5.0, None, None] -- single known value."""
    return [None, None, 5.0, None, None]


@pytest.fixture
def linear_data() -> List[Optional[float]]:
    """[0, 1, 2, None, 4, 5, 6, None, None, 9] -- linear trend."""
    return [0.0, 1.0, 2.0, None, 4.0, 5.0, 6.0, None, None, 9.0]


@pytest.fixture
def quadratic_data() -> List[Optional[float]]:
    """Quadratic values y=x^2 with a gap at x=3."""
    return [0.0, 1.0, 4.0, None, 16.0, 25.0]


@pytest.fixture
def step_data() -> List[Optional[float]]:
    """Step-like data: [0, 0, 0, None, 10, 10, 10]."""
    return [0.0, 0.0, 0.0, None, 10.0, 10.0, 10.0]


@pytest.fixture
def monotone_data() -> List[Optional[float]]:
    """Strictly monotone increasing with a gap: [1, 2, None, 4, 5, None, 7]."""
    return [1.0, 2.0, None, 4.0, 5.0, None, 7.0]


# ===========================================================================
# Tests: interpolate_linear -- basic
# ===========================================================================


class TestInterpolateLinearBasic:
    """Basic linear interpolation tests."""

    def test_single_gap_midpoint(self, engine, simple_gap):
        result = engine.interpolate_linear(simple_gap)
        assert result.filled_values[1] == pytest.approx(2.0, abs=1e-9)

    def test_double_gap_values(self, engine, double_gap):
        result = engine.interpolate_linear(double_gap)
        assert result.filled_values[1] == pytest.approx(2.0, abs=1e-9)
        assert result.filled_values[2] == pytest.approx(3.0, abs=1e-9)

    def test_no_gaps_unchanged(self, engine, no_gap):
        result = engine.interpolate_linear(no_gap)
        assert result.filled_values == no_gap
        assert result.gaps_found == 0
        assert result.total_missing == 0

    def test_preserves_known_values(self, engine, linear_data):
        original_known = {
            i: v for i, v in enumerate(linear_data) if v is not None
        }
        result = engine.interpolate_linear(linear_data)
        for i, v in original_known.items():
            assert result.filled_values[i] == pytest.approx(v, abs=1e-9)

    def test_linear_series_fills_exactly(self, engine, linear_data):
        """For perfectly linear data, fill should be exact."""
        result = engine.interpolate_linear(linear_data)
        assert result.filled_values[3] == pytest.approx(3.0, abs=1e-6)
        assert result.filled_values[7] == pytest.approx(7.0, abs=1e-6)
        assert result.filled_values[8] == pytest.approx(8.0, abs=1e-6)


# ===========================================================================
# Tests: interpolate_linear -- edge gaps
# ===========================================================================


class TestInterpolateLinearEdgeGaps:
    """Linear interpolation for leading and trailing gaps."""

    def test_leading_gap_filled(self, engine, leading_gap):
        result = engine.interpolate_linear(leading_gap)
        assert result.filled_values[0] is not None
        assert result.filled_values[1] is not None
        # With extrapolation from [3,4,5], values should decrease
        assert isinstance(result.filled_values[0], float)

    def test_trailing_gap_filled(self, engine, trailing_gap):
        result = engine.interpolate_linear(trailing_gap)
        assert result.filled_values[3] is not None
        assert result.filled_values[4] is not None
        assert isinstance(result.filled_values[3], float)

    def test_all_missing_filled_to_zero(self, engine, all_missing):
        result = engine.interpolate_linear(all_missing)
        for v in result.filled_values:
            assert v == pytest.approx(0.0, abs=1e-9)


# ===========================================================================
# Tests: interpolate_cubic_spline
# ===========================================================================


class TestInterpolateCubicSpline:
    """Cubic spline interpolation tests."""

    def test_smooth_data_produces_smooth_fill(self, engine):
        """Smooth sinusoidal data should be filled smoothly."""
        n = 20
        values: List[Optional[float]] = [
            math.sin(2 * math.pi * i / n) for i in range(n)
        ]
        values[5] = None
        values[10] = None
        result = engine.interpolate_cubic_spline(values)
        # Filled values should be close to the original sine
        expected_5 = math.sin(2 * math.pi * 5 / n)
        expected_10 = math.sin(2 * math.pi * 10 / n)
        assert result.filled_values[5] == pytest.approx(expected_5, abs=0.15)
        assert result.filled_values[10] == pytest.approx(expected_10, abs=0.15)

    def test_linear_data_matches_linear(self, engine, linear_data):
        """For linear data, cubic spline should match linear interpolation."""
        result = engine.interpolate_cubic_spline(linear_data)
        assert result.filled_values[3] == pytest.approx(3.0, abs=0.5)

    def test_single_gap(self, engine, simple_gap):
        result = engine.interpolate_cubic_spline(simple_gap)
        assert result.filled_values[1] == pytest.approx(2.0, abs=0.5)

    def test_method_name_correct(self, engine, simple_gap):
        result = engine.interpolate_cubic_spline(simple_gap)
        assert result.method == "cubic_spline"

    def test_gaps_found_count(self, engine, simple_gap):
        result = engine.interpolate_cubic_spline(simple_gap)
        assert result.gaps_found == 1


# ===========================================================================
# Tests: interpolate_polynomial
# ===========================================================================


class TestInterpolatePolynomial:
    """Polynomial interpolation tests."""

    def test_degree_1_matches_linear(self, engine, simple_gap):
        """Degree 1 polynomial is equivalent to linear interpolation."""
        result = engine.interpolate_polynomial(simple_gap, degree=1)
        assert result.filled_values[1] == pytest.approx(2.0, abs=0.5)

    def test_degree_2_for_quadratic(self, engine, quadratic_data):
        """Degree 2 should fit quadratic data closely."""
        result = engine.interpolate_polynomial(quadratic_data, degree=2)
        # x=3 -> y=9 for y=x^2
        assert result.filled_values[3] == pytest.approx(9.0, abs=2.0)

    def test_polynomial_preserves_known(self, engine, linear_data):
        original_known = {
            i: v for i, v in enumerate(linear_data) if v is not None
        }
        result = engine.interpolate_polynomial(linear_data, degree=2)
        for i, v in original_known.items():
            assert result.filled_values[i] == pytest.approx(v, abs=1e-6)

    def test_method_name(self, engine, simple_gap):
        result = engine.interpolate_polynomial(simple_gap, degree=2)
        assert result.method == "polynomial"

    def test_high_degree_capped(self, engine, simple_gap):
        """Degree > MAX_POLYNOMIAL_DEGREE is capped safely."""
        result = engine.interpolate_polynomial(simple_gap, degree=100)
        assert isinstance(result, FillResult)
        assert result.filled_values[1] is not None


# ===========================================================================
# Tests: interpolate_akima
# ===========================================================================


class TestInterpolateAkima:
    """Akima interpolation tests."""

    def test_step_data_no_overshoot(self, engine, step_data):
        """Akima should not overshoot for step-like data."""
        result = engine.interpolate_akima(step_data)
        filled_val = result.filled_values[3]
        # Should be between 0 and 10 (the two plateau levels)
        assert 0.0 <= filled_val <= 10.0

    def test_smooth_data(self, engine):
        n = 20
        values: List[Optional[float]] = [
            math.sin(2 * math.pi * i / n) for i in range(n)
        ]
        values[7] = None
        result = engine.interpolate_akima(values)
        expected = math.sin(2 * math.pi * 7 / n)
        assert result.filled_values[7] == pytest.approx(expected, abs=0.2)

    def test_method_name(self, engine, simple_gap):
        result = engine.interpolate_akima(simple_gap)
        assert result.method == "akima"

    def test_single_gap_fill(self, engine, simple_gap):
        result = engine.interpolate_akima(simple_gap)
        assert result.filled_values[1] is not None
        assert result.gaps_found == 1


# ===========================================================================
# Tests: interpolate_nearest
# ===========================================================================


class TestInterpolateNearest:
    """Nearest-neighbour interpolation tests."""

    def test_nearest_fills_with_closest(self, engine, simple_gap):
        result = engine.interpolate_nearest(simple_gap)
        filled_val = result.filled_values[1]
        # Should be either 1.0 or 3.0 (nearest on either side)
        assert filled_val in [1.0, 3.0]

    def test_nearest_left_preferred_equidistant(self, engine):
        """When equidistant, left value is preferred."""
        series: List[Optional[float]] = [10.0, None, 20.0]
        result = engine.interpolate_nearest(series)
        assert result.filled_values[1] == 10.0

    def test_nearest_leading_gap(self, engine, leading_gap):
        result = engine.interpolate_nearest(leading_gap)
        # Leading gaps should be filled with nearest right value (3.0)
        assert result.filled_values[0] == 3.0

    def test_nearest_trailing_gap(self, engine, trailing_gap):
        result = engine.interpolate_nearest(trailing_gap)
        # Trailing gaps filled with nearest left value (3.0)
        assert result.filled_values[3] == 3.0

    def test_method_name(self, engine, simple_gap):
        result = engine.interpolate_nearest(simple_gap)
        assert result.method == "nearest"


# ===========================================================================
# Tests: interpolate_pchip
# ===========================================================================


class TestInterpolatePchip:
    """PCHIP (monotonicity-preserving) interpolation tests."""

    def test_monotone_preservation(self, engine, monotone_data):
        """PCHIP should preserve monotonicity for monotone data."""
        result = engine.interpolate_pchip(monotone_data)
        filled = result.filled_values
        # The series should be non-decreasing
        for i in range(1, len(filled)):
            assert filled[i] >= filled[i - 1] - 1e-6

    def test_single_gap_reasonable(self, engine, simple_gap):
        result = engine.interpolate_pchip(simple_gap)
        assert result.filled_values[1] == pytest.approx(2.0, abs=0.5)

    def test_linear_data_exact(self, engine, linear_data):
        """For linear data, PCHIP should be close to exact."""
        result = engine.interpolate_pchip(linear_data)
        assert result.filled_values[3] == pytest.approx(3.0, abs=0.5)

    def test_method_name(self, engine, simple_gap):
        result = engine.interpolate_pchip(simple_gap)
        assert result.method == "pchip"

    def test_gaps_found(self, engine, monotone_data):
        result = engine.interpolate_pchip(monotone_data)
        assert result.gaps_found == 2


# ===========================================================================
# Tests: fill_gaps dispatcher
# ===========================================================================


class TestFillGapsDispatcher:
    """Tests for the fill_gaps method that dispatches to specific strategies."""

    def test_dispatch_linear(self, engine, simple_gap):
        result = engine.fill_gaps(simple_gap, method="linear")
        assert result.method == "linear"
        assert result.filled_values[1] == pytest.approx(2.0, abs=1e-9)

    def test_dispatch_cubic_spline(self, engine, simple_gap):
        result = engine.fill_gaps(simple_gap, method="cubic_spline")
        assert result.method == "cubic_spline"

    def test_dispatch_polynomial(self, engine, simple_gap):
        result = engine.fill_gaps(simple_gap, method="polynomial")
        assert result.method == "polynomial"

    def test_dispatch_akima(self, engine, simple_gap):
        result = engine.fill_gaps(simple_gap, method="akima")
        assert result.method == "akima"

    def test_dispatch_nearest(self, engine, simple_gap):
        result = engine.fill_gaps(simple_gap, method="nearest")
        assert result.method == "nearest"

    def test_dispatch_pchip(self, engine, simple_gap):
        result = engine.fill_gaps(simple_gap, method="pchip")
        assert result.method == "pchip"

    def test_dispatch_unknown_raises(self, engine, simple_gap):
        with pytest.raises(ValueError, match="Unrecognised interpolation method"):
            engine.fill_gaps(simple_gap, method="fourier_magic")

    def test_dispatch_case_insensitive(self, engine, simple_gap):
        result = engine.fill_gaps(simple_gap, method="LINEAR")
        assert result.method == "linear"


# ===========================================================================
# Tests: confidence scoring
# ===========================================================================


class TestConfidenceScoring:
    """Tests for fill confidence scoring."""

    def test_short_gap_higher_confidence(self, engine):
        """A gap of length 1 should have higher confidence than length 10."""
        short_series: List[Optional[float]] = [1.0, None, 3.0]
        long_series: List[Optional[float]] = (
            [1.0] + [None] * 10 + [12.0]
        )
        short_result = engine.interpolate_linear(short_series)
        long_result = engine.interpolate_linear(long_series)
        short_conf = short_result.mean_confidence
        long_conf = long_result.mean_confidence
        assert short_conf >= long_conf

    def test_confidence_bounded(self, engine, linear_data):
        result = engine.interpolate_linear(linear_data)
        assert 0.0 <= result.mean_confidence <= 1.0
        assert 0.0 <= result.min_confidence <= 1.0

    def test_no_gap_confidence_is_one(self, engine, no_gap):
        result = engine.interpolate_linear(no_gap)
        assert result.mean_confidence == pytest.approx(1.0)

    def test_filled_points_have_confidence(self, engine, simple_gap):
        result = engine.interpolate_linear(simple_gap)
        for fp in result.filled_points:
            assert 0.0 <= fp.confidence <= 1.0
            assert fp.was_missing is True
            assert fp.method == "linear"


# ===========================================================================
# Tests: empty and all-missing series
# ===========================================================================


class TestEmptyAndAllMissing:
    """Tests for degenerate series inputs."""

    def test_empty_series_linear(self, engine):
        result = engine.interpolate_linear([])
        assert result.filled_values == []
        assert result.gaps_found == 0

    def test_all_missing_linear(self, engine, all_missing):
        result = engine.interpolate_linear(all_missing)
        assert len(result.filled_values) == 5
        # All values replaced with 0.0 (no neighbours)
        for v in result.filled_values:
            assert v == pytest.approx(0.0, abs=1e-9)

    def test_single_value_linear(self, engine):
        result = engine.interpolate_linear([42.0])
        assert result.filled_values == [42.0]
        assert result.gaps_found == 0

    def test_single_none_linear(self, engine):
        result = engine.interpolate_linear([None])
        assert len(result.filled_values) == 1
        assert result.filled_values[0] == pytest.approx(0.0, abs=1e-9)

    def test_empty_series_cubic_spline(self, engine):
        result = engine.interpolate_cubic_spline([])
        assert result.filled_values == []

    def test_single_known_value_surrounded(self, engine, single_known):
        """Only one known value at index 2 -- gaps on both sides."""
        result = engine.interpolate_linear(single_known)
        assert len(result.filled_values) == 5


# ===========================================================================
# Tests: provenance
# ===========================================================================


class TestInterpolationProvenance:
    """Provenance tracking tests for InterpolationEngine."""

    def test_result_has_provenance_hash(self, engine, simple_gap):
        result = engine.interpolate_linear(simple_gap)
        assert isinstance(result.provenance_hash, str)
        assert len(result.provenance_hash) == 64

    def test_filled_points_have_provenance(self, engine, simple_gap):
        result = engine.interpolate_linear(simple_gap)
        for fp in result.filled_points:
            assert isinstance(fp.provenance_hash, str)
            assert len(fp.provenance_hash) == 64

    def test_result_has_processing_time(self, engine, simple_gap):
        result = engine.interpolate_linear(simple_gap)
        assert result.processing_time_ms >= 0.0

    def test_result_has_created_at(self, engine, simple_gap):
        result = engine.interpolate_linear(simple_gap)
        assert isinstance(result.created_at, str)
        assert len(result.created_at) > 0

    def test_result_id_is_uuid(self, engine, simple_gap):
        result = engine.interpolate_linear(simple_gap)
        assert isinstance(result.result_id, str)
        assert len(result.result_id) > 0


# ===========================================================================
# Tests: FillResult structure
# ===========================================================================


class TestFillResultStructure:
    """Tests verifying the FillResult object structure."""

    def test_fill_result_type(self, engine, simple_gap):
        result = engine.interpolate_linear(simple_gap)
        assert isinstance(result, FillResult)

    def test_filled_values_length_matches_input(self, engine, linear_data):
        result = engine.interpolate_linear(linear_data)
        assert len(result.filled_values) == len(linear_data)

    def test_gaps_filled_equals_gaps_found(self, engine, linear_data):
        result = engine.interpolate_linear(linear_data)
        assert result.gaps_filled == result.gaps_found

    def test_total_missing_count(self, engine, linear_data):
        result = engine.interpolate_linear(linear_data)
        expected_missing = sum(1 for v in linear_data if v is None)
        assert result.total_missing == expected_missing

    def test_metadata_present(self, engine, simple_gap):
        result = engine.interpolate_linear(simple_gap)
        assert "engine" in result.metadata
        assert result.metadata["engine"] == "InterpolationEngine"
