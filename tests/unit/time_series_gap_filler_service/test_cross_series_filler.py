# -*- coding: utf-8 -*-
"""
Unit tests for CrossSeriesFillerEngine - AGENT-DATA-014

Tests compute_correlation, find_best_donor, fill_regression, fill_ratio,
fill_donor_matching, compute_similarity_matrix, register_reference_series,
threshold filtering, missing-value handling in both series, provenance
tracking, confidence scoring, and edge cases.
Target: 55+ tests.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-014 Time Series Gap Filler (GL-DATA-X-017)
"""

import math
import os
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from greenlang.time_series_gap_filler.config import (
    TimeSeriesGapFillerConfig,
    reset_config,
)
from greenlang.time_series_gap_filler.cross_series_filler import (
    CrossSeriesFillerEngine,
    ReferenceSeries,
    FillResult,
    CrossSeriesResult,
    _pearson_r,
    _ols_fit,
    _is_missing,
    _overlap_indices,
    _compute_confidence,
)
from greenlang.time_series_gap_filler.provenance import get_provenance_tracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Remove GL_TSGF_ env vars and reset config between tests."""
    keys = [k for k in os.environ if k.startswith("GL_TSGF_")]
    for k in keys:
        monkeypatch.delenv(k, raising=False)
    reset_config()
    # Reset provenance tracker to genesis so chain state doesn't bleed
    tracker = get_provenance_tracker()
    tracker.reset()
    yield
    reset_config()


@pytest.fixture
def config():
    """Default test configuration."""
    return TimeSeriesGapFillerConfig(
        correlation_threshold=0.7,
        enable_cross_series=True,
        enable_provenance=True,
        min_data_points=10,
    )


@pytest.fixture
def engine(config):
    """Create a CrossSeriesFillerEngine with test config."""
    return CrossSeriesFillerEngine(config)


@pytest.fixture
def perfect_positive_pair():
    """Two series with perfect positive correlation (r=1.0)."""
    ref = [float(i) for i in range(30)]
    target = [float(i) for i in range(30)]
    return target, ref


@pytest.fixture
def perfect_negative_pair():
    """Two series with perfect negative correlation (r=-1.0)."""
    ref = [float(i) for i in range(30)]
    target = [float(29 - i) for i in range(30)]
    return target, ref


@pytest.fixture
def uncorrelated_pair():
    """Two series with near-zero correlation (sin vs cos)."""
    ref = [math.sin(2 * math.pi * i / 30) for i in range(30)]
    target = [math.cos(2 * math.pi * i / 30) for i in range(30)]
    return target, ref


@pytest.fixture
def moderate_correlation_pair():
    """Two series with moderate positive correlation (~0.6-0.8)."""
    ref = [float(i) for i in range(30)]
    target = [float(i) + (5.0 * math.sin(i * 0.7)) for i in range(30)]
    return target, ref


@pytest.fixture
def target_with_gaps():
    """Target series with gaps at indices 5, 15, 25."""
    values = [float(i) * 2.0 + 10.0 for i in range(30)]
    values[5] = None
    values[15] = None
    values[25] = None
    return values


@pytest.fixture
def reference_series_values():
    """Reference series perfectly correlated with target (target = 2*ref + 10)."""
    return [float(i) for i in range(30)]


@pytest.fixture
def donor_ref(reference_series_values, engine):
    """A registered ReferenceSeries donor."""
    return engine.register_reference_series(
        "donor_a", reference_series_values, name="Donor A",
    )


@pytest.fixture
def multiple_donors(engine):
    """Register three donors with varying correlation to a linear target."""
    ref_high = [float(i) * 2.0 + 5.0 for i in range(30)]
    ref_medium = [float(i) + 10.0 * math.sin(i * 0.3) for i in range(30)]
    ref_low = [math.sin(i * 0.5) * 100.0 for i in range(30)]
    d1 = engine.register_reference_series("ref_high", ref_high)
    d2 = engine.register_reference_series("ref_medium", ref_medium)
    d3 = engine.register_reference_series("ref_low", ref_low)
    return [d1, d2, d3]


# =========================================================================
# Initialization
# =========================================================================


class TestCrossSeriesFillerInit:
    """Tests for CrossSeriesFillerEngine initialization."""

    def test_creates_instance(self, config):
        engine = CrossSeriesFillerEngine(config)
        assert engine is not None

    def test_config_stored(self, engine, config):
        assert engine._config is config

    def test_default_config_used_when_none(self):
        engine = CrossSeriesFillerEngine()
        assert engine._config is not None

    def test_provenance_tracker_created(self, engine):
        assert engine._provenance is not None

    def test_registry_starts_empty(self, engine):
        assert len(engine.get_registry()) == 0


# =========================================================================
# register_reference_series
# =========================================================================


class TestRegisterReferenceSeries:
    """Tests for register_reference_series."""

    def test_register_basic(self, engine):
        """Register a new reference series and retrieve it."""
        ref = engine.register_reference_series("s1", [1.0, 2.0, 3.0])
        assert isinstance(ref, ReferenceSeries)
        assert ref.series_id == "s1"
        assert ref.values == [1.0, 2.0, 3.0]

    def test_register_appears_in_registry(self, engine):
        """Registered series should appear in get_registry()."""
        engine.register_reference_series("s1", [1.0, 2.0, 3.0])
        registry = engine.get_registry()
        assert "s1" in registry

    def test_register_overwrites(self, engine):
        """Re-registering the same ID overwrites the previous."""
        engine.register_reference_series("s1", [1.0, 2.0])
        engine.register_reference_series("s1", [10.0, 20.0, 30.0])
        reg = engine.get_registry()
        assert len(reg["s1"].values) == 3

    def test_register_with_name(self, engine):
        """Name parameter is stored correctly."""
        ref = engine.register_reference_series("s1", [1.0], name="Temperature")
        assert ref.name == "Temperature"

    def test_register_empty_id_raises(self, engine):
        """Empty series_id raises ValueError."""
        with pytest.raises(ValueError):
            engine.register_reference_series("", [1.0, 2.0])

    def test_register_empty_values_raises(self, engine):
        """Empty values list raises ValueError."""
        with pytest.raises(ValueError):
            engine.register_reference_series("s1", [])

    def test_register_provenance_recorded(self, engine):
        """Registration records provenance entry."""
        ref = engine.register_reference_series("s1", [1.0, 2.0, 3.0])
        assert ref.registered_at != ""


# =========================================================================
# compute_correlation
# =========================================================================


class TestComputeCorrelation:
    """Tests for compute_correlation (Pearson r between two series)."""

    def test_perfect_positive(self, engine, perfect_positive_pair):
        """Identical series yield r near 1.0."""
        target, ref = perfect_positive_pair
        r = engine.compute_correlation(target, ref)
        assert r == pytest.approx(1.0, abs=0.01)

    def test_perfect_negative(self, engine, perfect_negative_pair):
        """Perfectly negatively correlated series yield r near -1.0."""
        target, ref = perfect_negative_pair
        r = engine.compute_correlation(target, ref)
        assert r == pytest.approx(-1.0, abs=0.01)

    def test_uncorrelated(self, engine, uncorrelated_pair):
        """Orthogonal sin/cos series yield |r| near 0."""
        target, ref = uncorrelated_pair
        r = engine.compute_correlation(target, ref)
        assert abs(r) < 0.2

    def test_moderate_correlation(self, engine, moderate_correlation_pair):
        """Noisy linear data yields 0.5 < |r| < 1.0."""
        target, ref = moderate_correlation_pair
        r = engine.compute_correlation(target, ref)
        assert 0.5 < abs(r) < 1.0

    def test_constant_series_returns_zero(self, engine):
        """Constant series has undefined correlation; returns 0.0."""
        target = [float(i) for i in range(30)]
        ref = [5.0] * 30
        r = engine.compute_correlation(target, ref)
        assert abs(r) < 0.01

    def test_insufficient_overlap_returns_zero(self, engine):
        """Fewer than 3 overlapping points returns 0.0."""
        target = [1.0, None, None, None]
        ref = [None, None, None, 4.0]
        r = engine.compute_correlation(target, ref)
        assert r == 0.0

    def test_handles_none_in_both_series(self, engine):
        """Gaps in both series are excluded from correlation."""
        target = [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        ref = [1.0, None, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        r = engine.compute_correlation(target, ref)
        assert abs(r) > 0.9


# =========================================================================
# find_best_donor
# =========================================================================


class TestFindBestDonor:
    """Tests for find_best_donor: identifies highest-correlation donor."""

    def test_finds_best_among_multiple(self, engine, multiple_donors):
        """Selects the donor with highest |r| above threshold."""
        target = [float(i) * 2.0 + 10.0 for i in range(30)]
        best = engine.find_best_donor(target, multiple_donors)
        assert best is not None
        assert best.series_id == "ref_high"

    def test_returns_none_when_all_below_threshold(self, engine):
        """Returns None when no donor exceeds correlation_threshold."""
        engine._config.correlation_threshold = 0.99
        target = [float(i) for i in range(30)]
        donors = [
            ReferenceSeries(
                series_id="weak",
                values=[math.sin(i * 0.5) * 100 for i in range(30)],
            ),
        ]
        best = engine.find_best_donor(target, donors)
        assert best is None

    def test_returns_none_for_empty_list(self, engine):
        """Empty donor list returns None."""
        best = engine.find_best_donor([1.0, 2.0, 3.0], [])
        assert best is None

    def test_single_donor_above_threshold(self, engine, donor_ref):
        """Single qualifying donor is returned."""
        target = [float(i) for i in range(30)]
        best = engine.find_best_donor(target, [donor_ref])
        assert best is not None
        assert best.series_id == "donor_a"


# =========================================================================
# fill_regression
# =========================================================================


class TestFillRegression:
    """Tests for fill_regression: OLS regression-based gap filling."""

    def test_fills_all_gaps(self, engine, target_with_gaps, donor_ref):
        """All 3 gaps should be filled."""
        result = engine.fill_regression(target_with_gaps, donor_ref)
        assert isinstance(result, FillResult)
        assert result.gaps_filled == 3
        assert result.values[5] is not None
        assert result.values[15] is not None
        assert result.values[25] is not None

    def test_preserves_known_values(self, engine, target_with_gaps, donor_ref):
        """Original non-gap values must not change."""
        original = [float(i) * 2.0 + 10.0 for i in range(30)]
        result = engine.fill_regression(target_with_gaps, donor_ref)
        for i, v in enumerate(original):
            if target_with_gaps[i] is not None:
                assert result.values[i] == pytest.approx(v, abs=1e-6)

    def test_regression_accuracy(self, engine, donor_ref):
        """Linear regression fills close to true value for y = 2x + 10."""
        target = [float(i) * 2.0 + 10.0 for i in range(30)]
        target[10] = None
        result = engine.fill_regression(target, donor_ref)
        # Expected: 2*10 + 10 = 30.0
        assert abs(result.values[10] - 30.0) < 2.0

    def test_regression_method_name(self, engine, target_with_gaps, donor_ref):
        """Result method should be 'regression'."""
        result = engine.fill_regression(target_with_gaps, donor_ref)
        assert result.method == "regression"

    def test_regression_slope_intercept(self, engine, target_with_gaps, donor_ref):
        """Result should contain slope and intercept."""
        result = engine.fill_regression(target_with_gaps, donor_ref)
        assert abs(result.slope - 2.0) < 0.5
        assert abs(result.intercept - 10.0) < 2.0

    def test_regression_r_squared(self, engine, target_with_gaps, donor_ref):
        """R-squared should be near 1.0 for perfectly correlated data."""
        result = engine.fill_regression(target_with_gaps, donor_ref)
        assert result.r_squared > 0.95

    def test_regression_provenance(self, engine, target_with_gaps, donor_ref):
        """Result includes provenance hash."""
        result = engine.fill_regression(target_with_gaps, donor_ref)
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64

    def test_regression_processing_time(self, engine, target_with_gaps, donor_ref):
        """Result includes processing_time_ms."""
        result = engine.fill_regression(target_with_gaps, donor_ref)
        assert result.processing_time_ms >= 0.0

    def test_insufficient_overlap(self, engine):
        """Insufficient overlap returns unfilled result."""
        target = [1.0, None, None, None]
        donor = ReferenceSeries(
            series_id="d", values=[None, None, None, 4.0],
        )
        result = engine.fill_regression(target, donor)
        assert result.gaps_filled == 0
        assert result.gaps_remaining == 3


# =========================================================================
# fill_ratio
# =========================================================================


class TestFillRatio:
    """Tests for fill_ratio: proportional-scaling gap filling."""

    def test_fills_all_gaps(self, engine, target_with_gaps, donor_ref):
        """All 3 gaps should be filled using ratio method."""
        result = engine.fill_ratio(target_with_gaps, donor_ref)
        assert isinstance(result, FillResult)
        assert result.gaps_filled == 3
        assert result.values[5] is not None
        assert result.values[15] is not None

    def test_ratio_method_name(self, engine, target_with_gaps, donor_ref):
        """Result method should be 'ratio'."""
        result = engine.fill_ratio(target_with_gaps, donor_ref)
        assert result.method == "ratio"

    def test_ratio_same_scale(self, engine):
        """When target and ref have same values, ratio ~1.0."""
        ref_vals = [10.0 + i for i in range(20)]
        target = [10.0 + i for i in range(20)]
        target[8] = None
        donor = engine.register_reference_series("same_scale", ref_vals)
        result = engine.fill_ratio(target, donor)
        assert abs(result.values[8] - 18.0) < 3.0

    def test_ratio_double_scale(self, engine):
        """When target is 2x reference, filled values reflect the ratio."""
        ref_vals = [float(i) + 1.0 for i in range(20)]
        target = [2.0 * (float(i) + 1.0) for i in range(20)]
        target[10] = None
        donor = engine.register_reference_series("scale2", ref_vals)
        result = engine.fill_ratio(target, donor)
        # Expected: 2 * 11 = 22.0
        assert abs(result.values[10] - 22.0) < 5.0

    def test_ratio_preserves_known(self, engine, target_with_gaps, donor_ref):
        """Known values must not change."""
        original = [float(i) * 2.0 + 10.0 for i in range(30)]
        result = engine.fill_ratio(target_with_gaps, donor_ref)
        for i, v in enumerate(original):
            if target_with_gaps[i] is not None:
                assert result.values[i] == pytest.approx(v, abs=1e-6)

    def test_ratio_provenance(self, engine, target_with_gaps, donor_ref):
        """Result includes provenance hash."""
        result = engine.fill_ratio(target_with_gaps, donor_ref)
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64


# =========================================================================
# fill_donor_matching (multi-donor consensus)
# =========================================================================


class TestFillDonorMatching:
    """Tests for fill_donor_matching: weighted multi-donor consensus."""

    def test_fills_gaps_with_multiple_donors(self, engine, target_with_gaps,
                                              multiple_donors):
        """Fills gaps using consensus from multiple donors."""
        result = engine.fill_donor_matching(target_with_gaps, multiple_donors)
        assert isinstance(result, CrossSeriesResult)
        assert result.gaps_filled >= 1

    def test_donor_matching_returns_donor_ids(self, engine, target_with_gaps,
                                               multiple_donors):
        """Result lists which donors contributed."""
        result = engine.fill_donor_matching(target_with_gaps, multiple_donors)
        assert isinstance(result.donor_ids, list)
        assert result.donors_used >= 0

    def test_donor_matching_avg_confidence(self, engine, target_with_gaps,
                                            multiple_donors):
        """Average confidence is between 0 and 1."""
        result = engine.fill_donor_matching(target_with_gaps, multiple_donors)
        if result.gaps_filled > 0:
            assert 0.0 <= result.avg_confidence <= 1.0

    def test_donor_matching_preserves_known(self, engine, target_with_gaps,
                                              multiple_donors):
        """Known values must not change."""
        original = [float(i) * 2.0 + 10.0 for i in range(30)]
        result = engine.fill_donor_matching(target_with_gaps, multiple_donors)
        for i, v in enumerate(original):
            if target_with_gaps[i] is not None:
                assert result.values[i] == pytest.approx(v, abs=1e-6)

    def test_donor_matching_no_viable_donors(self, engine):
        """When no donor exceeds threshold, returns unfilled."""
        engine._config.correlation_threshold = 0.99
        target = [float(i) for i in range(20)]
        target[5] = None
        donors = [
            ReferenceSeries(
                series_id="weak",
                values=[math.sin(i * 0.5) * 100 for i in range(20)],
            ),
        ]
        result = engine.fill_donor_matching(target, donors)
        assert result.gaps_filled == 0

    def test_donor_matching_provenance(self, engine, target_with_gaps,
                                        multiple_donors):
        """Result includes provenance hash."""
        result = engine.fill_donor_matching(target_with_gaps, multiple_donors)
        if result.gaps_filled > 0:
            assert result.provenance_hash != ""
            assert len(result.provenance_hash) == 64


# =========================================================================
# compute_similarity_matrix
# =========================================================================


class TestComputeSimilarityMatrix:
    """Tests for compute_similarity_matrix: pairwise correlation matrix."""

    def test_diagonal_is_one(self, engine, multiple_donors):
        """Diagonal entries are always 1.0."""
        matrix = engine.compute_similarity_matrix(multiple_donors)
        for ref in multiple_donors:
            assert matrix[ref.series_id][ref.series_id] == pytest.approx(1.0)

    def test_symmetric(self, engine, multiple_donors):
        """Matrix is symmetric: matrix[a][b] == matrix[b][a]."""
        matrix = engine.compute_similarity_matrix(multiple_donors)
        ids = [r.series_id for r in multiple_donors]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                assert matrix[ids[i]][ids[j]] == pytest.approx(
                    matrix[ids[j]][ids[i]], abs=1e-10,
                )

    def test_correct_dimensions(self, engine, multiple_donors):
        """Matrix has N x N entries for N series."""
        matrix = engine.compute_similarity_matrix(multiple_donors)
        n = len(multiple_donors)
        assert len(matrix) == n
        for row in matrix.values():
            assert len(row) == n

    def test_empty_list_returns_empty(self, engine):
        """Empty series list returns empty matrix."""
        matrix = engine.compute_similarity_matrix([])
        assert matrix == {}

    def test_single_series(self, engine):
        """Single series returns 1x1 matrix with diagonal 1.0."""
        ref = engine.register_reference_series("solo", [1.0, 2.0, 3.0])
        matrix = engine.compute_similarity_matrix([ref])
        assert matrix["solo"]["solo"] == pytest.approx(1.0)

    def test_values_in_range(self, engine, multiple_donors):
        """All correlation values are in [-1.0, 1.0]."""
        matrix = engine.compute_similarity_matrix(multiple_donors)
        for row in matrix.values():
            for val in row.values():
                assert -1.0 <= val <= 1.0


# =========================================================================
# Threshold filtering
# =========================================================================


class TestThresholdFiltering:
    """Tests for correlation threshold enforcement."""

    def test_high_threshold_filters_weak(self, engine):
        """Setting threshold=0.95 excludes moderately correlated donors."""
        engine._config.correlation_threshold = 0.95
        target = [float(i) for i in range(30)]
        target[10] = None
        # Moderately correlated donor
        donor = engine.register_reference_series(
            "mod",
            [float(i) + 5.0 * math.sin(i * 0.3) for i in range(30)],
        )
        best = engine.find_best_donor(target, [donor])
        assert best is None

    def test_low_threshold_accepts_weak(self, engine):
        """Setting threshold=0.3 accepts weakly correlated donors."""
        engine._config.correlation_threshold = 0.3
        target = [float(i) for i in range(30)]
        target[10] = None
        donor = engine.register_reference_series(
            "mod",
            [float(i) + 5.0 * math.sin(i * 0.3) for i in range(30)],
        )
        best = engine.find_best_donor(target, [donor])
        assert best is not None


# =========================================================================
# Missing values in both series
# =========================================================================


class TestMissingInBothSeries:
    """Tests for handling gaps in both target and donor series."""

    def test_donor_gaps_skip_unfillable_positions(self, engine):
        """Positions where both target and donor are missing cannot be filled."""
        target = [1.0, 2.0, None, 4.0, 5.0, 6.0, None, 8.0, 9.0, 10.0,
                  11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
        donor_vals = list(target)
        donor_vals[2] = 3.0  # Donor has value where target has gap
        donor_vals[6] = None  # Donor also missing at position 6
        donor = engine.register_reference_series("d1", donor_vals)
        result = engine.fill_regression(target, donor)
        # Position 2 should be filled; position 6 cannot be filled
        assert result.values[2] is not None
        assert 6 not in result.filled_indices

    def test_correlation_ignores_mutual_gaps(self, engine):
        """compute_correlation only uses positions where both have data."""
        a = [1.0, None, 3.0, None, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        b = [None, 2.0, 3.0, None, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        r = engine.compute_correlation(a, b)
        # Overlap at indices 2,4,5,6,7,8,9 (7 points) - should compute
        assert abs(r) > 0.9


# =========================================================================
# Edge cases
# =========================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_target(self, engine, donor_ref):
        """Empty target should raise or return empty result."""
        try:
            result = engine.fill_regression([], donor_ref)
            assert result.gaps_filled == 0
        except (ValueError, IndexError):
            pass

    def test_all_missing_target(self, engine, donor_ref):
        """All-None target should handle gracefully."""
        target = [None] * 20
        result = engine.fill_regression(target, donor_ref)
        # Insufficient overlap (all target positions missing) -
        # the gaps_filled should be 0 due to regression needing overlap
        assert result.gaps_filled == 0

    def test_constant_donor(self, engine):
        """Constant donor series produces 0 correlation."""
        target = [float(i) for i in range(20)]
        target[5] = None
        donor = engine.register_reference_series("const", [10.0] * 20)
        r = engine.compute_correlation(target, donor.values)
        assert abs(r) < 0.01

    def test_nan_values_treated_as_missing(self, engine):
        """float('nan') should be treated as missing."""
        target = [1.0, 2.0, float('nan'), 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        ref = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        r = engine.compute_correlation(target, ref)
        # NaN at index 2 excluded, 9 overlapping points
        assert abs(r) > 0.99


# =========================================================================
# Provenance and determinism
# =========================================================================


class TestProvenanceAndDeterminism:
    """Tests for provenance recording and deterministic outputs."""

    def test_regression_provenance_deterministic(self, engine, target_with_gaps,
                                                   donor_ref):
        """Same input produces same provenance hash when time is frozen.

        Provenance chain hashing includes timestamps and parent hash
        which accumulate between calls. We freeze time and reset the
        chain to ensure identical provenance output.
        """
        from datetime import datetime, timezone
        from unittest.mock import patch

        frozen = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        frozen_time_val = 1735689600.0

        with patch(
            "greenlang.time_series_gap_filler.provenance._utcnow",
            return_value=frozen,
        ), patch(
            "greenlang.time_series_gap_filler.cross_series_filler._utcnow",
            return_value=frozen,
        ), patch(
            "greenlang.time_series_gap_filler.cross_series_filler.time.time",
            return_value=frozen_time_val,
        ):
            engine._provenance.reset()
            r1 = engine.fill_regression(list(target_with_gaps), donor_ref)

        with patch(
            "greenlang.time_series_gap_filler.provenance._utcnow",
            return_value=frozen,
        ), patch(
            "greenlang.time_series_gap_filler.cross_series_filler._utcnow",
            return_value=frozen,
        ), patch(
            "greenlang.time_series_gap_filler.cross_series_filler.time.time",
            return_value=frozen_time_val,
        ):
            engine._provenance.reset()
            r2 = engine.fill_regression(list(target_with_gaps), donor_ref)

        assert r1.provenance_hash == r2.provenance_hash

    def test_ratio_provenance_deterministic(self, engine, target_with_gaps,
                                              donor_ref):
        """Same input produces same provenance hash for ratio when time is frozen."""
        from datetime import datetime, timezone
        from unittest.mock import patch

        frozen = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        frozen_time_val = 1735689600.0

        with patch(
            "greenlang.time_series_gap_filler.provenance._utcnow",
            return_value=frozen,
        ), patch(
            "greenlang.time_series_gap_filler.cross_series_filler._utcnow",
            return_value=frozen,
        ), patch(
            "greenlang.time_series_gap_filler.cross_series_filler.time.time",
            return_value=frozen_time_val,
        ):
            engine._provenance.reset()
            r1 = engine.fill_ratio(list(target_with_gaps), donor_ref)

        with patch(
            "greenlang.time_series_gap_filler.provenance._utcnow",
            return_value=frozen,
        ), patch(
            "greenlang.time_series_gap_filler.cross_series_filler._utcnow",
            return_value=frozen,
        ), patch(
            "greenlang.time_series_gap_filler.cross_series_filler.time.time",
            return_value=frozen_time_val,
        ):
            engine._provenance.reset()
            r2 = engine.fill_ratio(list(target_with_gaps), donor_ref)

        assert r1.provenance_hash == r2.provenance_hash

    def test_different_donors_different_provenance(self, engine, target_with_gaps):
        """Different donors produce different provenance hashes."""
        d1 = engine.register_reference_series("d1", [float(i) for i in range(30)])
        d2 = engine.register_reference_series(
            "d2", [float(i) * 3.0 for i in range(30)],
        )
        r1 = engine.fill_regression(target_with_gaps, d1)
        r2 = engine.fill_regression(target_with_gaps, d2)
        assert r1.provenance_hash != r2.provenance_hash


# =========================================================================
# Confidence scoring
# =========================================================================


class TestConfidenceScoring:
    """Tests for confidence scores in cross-series filling."""

    def test_per_point_confidence_present(self, engine, target_with_gaps, donor_ref):
        """Regression result includes per-point confidence."""
        result = engine.fill_regression(target_with_gaps, donor_ref)
        assert len(result.per_point_confidence) > 0

    def test_per_point_confidence_in_range(self, engine, target_with_gaps, donor_ref):
        """All per-point confidences are in [0, 1]."""
        result = engine.fill_regression(target_with_gaps, donor_ref)
        for conf in result.per_point_confidence.values():
            assert 0.0 <= conf <= 1.0

    def test_overall_confidence_in_range(self, engine, target_with_gaps, donor_ref):
        """Overall confidence is in [0, 1]."""
        result = engine.fill_regression(target_with_gaps, donor_ref)
        assert 0.0 <= result.confidence <= 1.0

    def test_high_corr_higher_confidence(self, engine):
        """Higher correlation donor produces higher confidence."""
        target = [2.0 * i + 10.0 for i in range(20)]
        target[10] = None

        d_high = engine.register_reference_series(
            "hi", [float(i) for i in range(20)],
        )
        d_low = engine.register_reference_series(
            "lo", [math.sin(i * 0.5) * 100 for i in range(20)],
        )

        r_high = engine.fill_regression(list(target), d_high)
        r_low = engine.fill_regression(list(target), d_low)

        assert r_high.confidence >= r_low.confidence


# =========================================================================
# Auto-fill convenience methods
# =========================================================================


class TestAutoFill:
    """Tests for auto_fill and auto_fill_consensus."""

    def test_auto_fill_regression(self, engine, target_with_gaps, donor_ref):
        """auto_fill with method='regression' uses best donor."""
        result = engine.auto_fill(target_with_gaps, method="regression")
        assert isinstance(result, FillResult)
        assert result.gaps_filled == 3

    def test_auto_fill_ratio(self, engine, target_with_gaps, donor_ref):
        """auto_fill with method='ratio' uses best donor."""
        result = engine.auto_fill(target_with_gaps, method="ratio")
        assert isinstance(result, FillResult)
        assert result.gaps_filled == 3

    def test_auto_fill_invalid_method(self, engine, target_with_gaps, donor_ref):
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError):
            engine.auto_fill(target_with_gaps, method="invalid")

    def test_auto_fill_consensus(self, engine, target_with_gaps, multiple_donors):
        """auto_fill_consensus uses all registered donors."""
        result = engine.auto_fill_consensus(target_with_gaps)
        assert isinstance(result, CrossSeriesResult)


# =========================================================================
# Diagnostics
# =========================================================================


class TestDiagnostics:
    """Tests for diagnose_donor, rank_donors, get_engine_summary."""

    def test_diagnose_donor(self, engine, donor_ref):
        """diagnose_donor returns diagnostic statistics."""
        target = [float(i) * 2.0 + 10.0 for i in range(30)]
        diag = engine.diagnose_donor(target, donor_ref)
        assert "correlation" in diag
        assert "overlap_count" in diag
        assert "fillable_count" in diag
        assert diag["exceeds_threshold"] is True

    def test_rank_donors(self, engine, multiple_donors):
        """rank_donors returns sorted list by |r|."""
        target = [float(i) * 2.0 + 10.0 for i in range(30)]
        ranked = engine.rank_donors(target, multiple_donors)
        assert len(ranked) == 3
        # First should have highest abs_correlation
        assert ranked[0]["abs_correlation"] >= ranked[1]["abs_correlation"]

    def test_engine_summary(self, engine, multiple_donors):
        """get_engine_summary reports registry size."""
        summary = engine.get_engine_summary()
        assert summary["registered_series"] == 3
        assert summary["engine"] == "CrossSeriesFillerEngine"


# =========================================================================
# Pure-Python helper function tests
# =========================================================================


class TestHelperFunctions:
    """Tests for module-level pure-Python helpers."""

    def test_pearson_r_identical(self):
        """_pearson_r on identical lists returns 1.0."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _pearson_r(x, x) == pytest.approx(1.0, abs=0.01)

    def test_pearson_r_different_lengths_raises(self):
        """_pearson_r raises ValueError for mismatched lengths."""
        with pytest.raises(ValueError):
            _pearson_r([1.0, 2.0], [1.0])

    def test_ols_fit_perfect_line(self):
        """_ols_fit recovers slope and intercept from y = 2x + 3."""
        x = [float(i) for i in range(10)]
        y = [2.0 * xi + 3.0 for xi in x]
        slope, intercept, r2 = _ols_fit(x, y)
        assert slope == pytest.approx(2.0, abs=0.01)
        assert intercept == pytest.approx(3.0, abs=0.01)
        assert r2 > 0.99

    def test_is_missing_none(self):
        assert _is_missing(None) is True

    def test_is_missing_nan(self):
        assert _is_missing(float('nan')) is True

    def test_is_missing_valid(self):
        assert _is_missing(42.0) is False

    def test_overlap_indices(self):
        """_overlap_indices excludes positions with any None."""
        a = [1.0, None, 3.0, 4.0]
        b = [1.0, 2.0, None, 4.0]
        indices = _overlap_indices(a, b)
        assert indices == [0, 3]

    def test_compute_confidence_regression(self):
        """_compute_confidence returns value in [0, 1]."""
        conf = _compute_confidence(r_squared=0.9, n_donors=1, method="regression")
        assert 0.0 <= conf <= 1.0

    def test_compute_confidence_multi_donor_bonus(self):
        """Multiple donors increase confidence."""
        c1 = _compute_confidence(r_squared=0.8, n_donors=1, method="regression")
        c3 = _compute_confidence(r_squared=0.8, n_donors=3, method="regression")
        assert c3 > c1
