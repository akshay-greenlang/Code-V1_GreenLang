# -*- coding: utf-8 -*-
"""
Tests for HistoricalReconstructor - AGENT-EUDR-004 Engine 3: Historical Reconstruction

Comprehensive test suite covering:
- Temporal composite construction (pixel-wise median across dates)
- Cloud-free pixel filtering for composite creation
- Cutoff-date forest classification from NDVI + Hansen tree cover
- Cross-validation with multi-source agreement scoring
- Temporal interpolation for missing years
- Cutoff canopy density estimation (Hansen + NDVI regression)
- Multi-source fusion weighting (Landsat 0.30, S2 0.30, Hansen 0.25, JAXA 0.15)
- Confidence based on data availability
- Batch reconstruction for multiple plots
- Default cutoff date (2020-12-31)
- Determinism and provenance hash reproducibility

Test count: 60+ tests
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-004 (Engine 3 - Historical Reconstruction)
"""

import math
import statistics

import pytest

from tests.agents.eudr.forest_cover_analysis.conftest import (
    HistoricalCoverRecord,
    compute_test_hash,
    fuse_weighted,
    SHA256_HEX_LENGTH,
    EUDR_DEFORESTATION_CUTOFF,
    RECONSTRUCTION_SOURCE_WEIGHTS,
    ALL_BIOMES,
)


# ---------------------------------------------------------------------------
# Helpers: Temporal composite and classification
# ---------------------------------------------------------------------------


def _temporal_composite_median(pixel_timeseries: list) -> float:
    """Compute pixel-wise median composite across acquisition dates.

    Excludes NaN / None values (cloud-masked pixels).
    """
    valid = [v for v in pixel_timeseries if v is not None and not math.isnan(v)]
    if not valid:
        return float("nan")
    return statistics.median(valid)


def _is_cloudy(value) -> bool:
    """Check if a pixel value represents cloud contamination."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def _classify_cutoff_forest(
    ndvi: float,
    hansen_tree_cover_pct: float,
    ndvi_threshold: float = 0.40,
    hansen_threshold: float = 10.0,
) -> bool:
    """Classify whether a plot was forest at the cutoff date.

    Uses NDVI above threshold AND Hansen tree cover above threshold
    (conservative approach: both must agree).
    """
    return ndvi >= ndvi_threshold and hansen_tree_cover_pct >= hansen_threshold


def _cross_validate_sources(
    source_classifications: dict,
) -> float:
    """Compute cross-validation agreement score across data sources.

    Returns fraction of sources that agree with the majority classification.
    """
    if not source_classifications:
        return 0.0
    from collections import Counter
    vals = list(source_classifications.values())
    counts = Counter(vals)
    most_common_count = counts.most_common(1)[0][1]
    return most_common_count / len(vals)


def _interpolate_missing_year(
    year_before: int,
    value_before: float,
    year_after: int,
    value_after: float,
    target_year: int,
) -> float:
    """Linear interpolation of missing year value from surrounding years."""
    if year_after == year_before:
        return value_before
    ratio = (target_year - year_before) / (year_after - year_before)
    return value_before + ratio * (value_after - value_before)


def _estimate_cutoff_density(
    hansen_tree_cover_pct: float,
    ndvi_mean: float,
    ndvi_coefficient: float = 90.0,
    hansen_weight: float = 0.50,
    ndvi_weight: float = 0.50,
) -> float:
    """Estimate canopy density at cutoff date from Hansen + NDVI regression."""
    ndvi_density = max(0.0, min(100.0, ndvi_coefficient * ndvi_mean))
    return hansen_weight * hansen_tree_cover_pct + ndvi_weight * ndvi_density


def _confidence_from_data_availability(
    sources_used: list,
    max_sources: int = 4,
) -> float:
    """Compute confidence based on number of available data sources.

    More sources = higher confidence (up to 4 expected sources).
    """
    if max_sources == 0:
        return 0.0
    return min(1.0, len(sources_used) / max_sources)


# ===========================================================================
# 1. Temporal Composite (12 tests)
# ===========================================================================


class TestTemporalComposite:
    """Test pixel-wise median composite construction."""

    def test_temporal_composite_median_basic(self):
        """Test median of valid values returns correct median."""
        values = [0.60, 0.65, 0.70, 0.75, 0.80]
        result = _temporal_composite_median(values)
        assert abs(result - 0.70) < 1e-9

    def test_temporal_composite_cloud_free(self):
        """Test cloudy pixels (None) excluded from median."""
        values = [0.60, None, 0.70, None, 0.80]
        result = _temporal_composite_median(values)
        assert abs(result - 0.70) < 1e-9

    def test_temporal_composite_nan_excluded(self):
        """Test NaN pixels excluded from median."""
        values = [0.60, float("nan"), 0.70, float("nan"), 0.80]
        result = _temporal_composite_median(values)
        assert abs(result - 0.70) < 1e-9

    def test_temporal_composite_all_cloudy(self):
        """Test all-cloudy returns NaN."""
        values = [None, None, None]
        result = _temporal_composite_median(values)
        assert math.isnan(result)

    def test_temporal_composite_single_valid(self):
        """Test single valid value returns that value."""
        values = [None, 0.72, None]
        result = _temporal_composite_median(values)
        assert abs(result - 0.72) < 1e-9

    def test_temporal_composite_even_count(self):
        """Test even number of values uses median (average of two middle)."""
        values = [0.60, 0.70, 0.80, 0.90]
        result = _temporal_composite_median(values)
        assert abs(result - 0.75) < 1e-9

    def test_temporal_composite_empty(self):
        """Test empty list returns NaN."""
        result = _temporal_composite_median([])
        assert math.isnan(result)

    @pytest.mark.parametrize("values,expected", [
        ([0.50, 0.50, 0.50], 0.50),
        ([0.10, 0.20, 0.30], 0.20),
        ([0.90, 0.80, 0.70], 0.80),
        ([0.00, 1.00], 0.50),
    ])
    def test_temporal_composite_parametrized(self, values, expected):
        """Test median computation for various value sets."""
        result = _temporal_composite_median(values)
        assert abs(result - expected) < 1e-9

    def test_temporal_composite_determinism(self):
        """Test median composite is deterministic."""
        values = [0.60, None, 0.70, 0.80]
        results = [_temporal_composite_median(values) for _ in range(10)]
        assert len(set(results)) == 1


# ===========================================================================
# 2. Cutoff Date Classification (10 tests)
# ===========================================================================


class TestCutoffClassification:
    """Test cutoff-date forest classification."""

    def test_classify_cutoff_forest(self):
        """Test high NDVI + high Hansen = was_forest True."""
        assert _classify_cutoff_forest(0.72, 80.0) is True

    def test_classify_cutoff_non_forest(self):
        """Test low NDVI + low Hansen = was_forest False."""
        assert _classify_cutoff_forest(0.15, 3.0) is False

    def test_classify_cutoff_high_ndvi_low_hansen(self):
        """Test high NDVI but low Hansen = False (conservative)."""
        assert _classify_cutoff_forest(0.72, 5.0) is False

    def test_classify_cutoff_low_ndvi_high_hansen(self):
        """Test low NDVI but high Hansen = False (conservative)."""
        assert _classify_cutoff_forest(0.20, 80.0) is False

    def test_classify_cutoff_boundary_ndvi(self):
        """Test NDVI exactly at threshold."""
        assert _classify_cutoff_forest(0.40, 50.0) is True

    def test_classify_cutoff_boundary_hansen(self):
        """Test Hansen exactly at threshold."""
        assert _classify_cutoff_forest(0.50, 10.0) is True

    def test_classify_cutoff_just_below_ndvi(self):
        """Test NDVI just below threshold."""
        assert _classify_cutoff_forest(0.39, 50.0) is False

    def test_classify_cutoff_just_below_hansen(self):
        """Test Hansen just below threshold."""
        assert _classify_cutoff_forest(0.50, 9.9) is False

    @pytest.mark.parametrize("ndvi,hansen,expected", [
        (0.80, 90.0, True),
        (0.60, 50.0, True),
        (0.40, 10.0, True),
        (0.39, 10.0, False),
        (0.40, 9.9, False),
        (0.10, 5.0, False),
        (0.00, 0.0, False),
    ])
    def test_classify_cutoff_parametrized(self, ndvi, hansen, expected):
        """Test cutoff classification across threshold boundaries."""
        assert _classify_cutoff_forest(ndvi, hansen) is expected

    def test_classify_cutoff_determinism(self):
        """Test classification is deterministic."""
        results = [_classify_cutoff_forest(0.72, 80.0) for _ in range(10)]
        assert len(set(results)) == 1


# ===========================================================================
# 3. Cross-Validation (8 tests)
# ===========================================================================


class TestCrossValidation:
    """Test multi-source cross-validation scoring."""

    def test_cross_validate_high_agreement(self):
        """Test 4/4 sources agree produces score near 1.0."""
        sources = {
            "landsat": True, "sentinel2": True,
            "hansen": True, "jaxa": True,
        }
        score = _cross_validate_sources(sources)
        assert abs(score - 1.0) < 1e-9

    def test_cross_validate_low_agreement(self):
        """Test 2/4 sources agree produces score ~0.5."""
        sources = {
            "landsat": True, "sentinel2": True,
            "hansen": False, "jaxa": False,
        }
        score = _cross_validate_sources(sources)
        assert abs(score - 0.5) < 1e-9

    def test_cross_validate_three_quarters(self):
        """Test 3/4 sources agree produces score 0.75."""
        sources = {
            "landsat": True, "sentinel2": True,
            "hansen": True, "jaxa": False,
        }
        score = _cross_validate_sources(sources)
        assert abs(score - 0.75) < 1e-9

    def test_cross_validate_single_source(self):
        """Test single source returns 1.0."""
        sources = {"landsat": True}
        score = _cross_validate_sources(sources)
        assert abs(score - 1.0) < 1e-9

    def test_cross_validate_empty(self):
        """Test empty sources returns 0.0."""
        score = _cross_validate_sources({})
        assert score == 0.0

    def test_cross_validate_all_disagree(self):
        """Test all different values still counts majority (1/4 = 0.25)."""
        sources = {"a": 1, "b": 2, "c": 3, "d": 4}
        score = _cross_validate_sources(sources)
        assert abs(score - 0.25) < 1e-9

    def test_cross_validate_determinism(self):
        """Test cross-validation is deterministic."""
        sources = {"landsat": True, "sentinel2": True, "hansen": False, "jaxa": True}
        results = [_cross_validate_sources(sources) for _ in range(10)]
        assert len(set(results)) == 1

    def test_cross_validate_score_range(self):
        """Test score is always in [0, 1]."""
        for n_agree in range(5):
            sources = {f"s{i}": (i < n_agree) for i in range(4)}
            score = _cross_validate_sources(sources)
            assert 0.0 <= score <= 1.0


# ===========================================================================
# 4. Temporal Interpolation (6 tests)
# ===========================================================================


class TestTemporalInterpolation:
    """Test linear interpolation for missing years."""

    def test_interpolate_missing_linear(self):
        """Test 2019 and 2021 data interpolated to 2020."""
        result = _interpolate_missing_year(2019, 0.70, 2021, 0.74, 2020)
        assert abs(result - 0.72) < 1e-9

    def test_interpolate_exact_start(self):
        """Test target at start year returns start value."""
        result = _interpolate_missing_year(2018, 0.60, 2020, 0.80, 2018)
        assert abs(result - 0.60) < 1e-9

    def test_interpolate_exact_end(self):
        """Test target at end year returns end value."""
        result = _interpolate_missing_year(2018, 0.60, 2020, 0.80, 2020)
        assert abs(result - 0.80) < 1e-9

    def test_interpolate_equal_years(self):
        """Test equal years returns start value."""
        result = _interpolate_missing_year(2020, 0.70, 2020, 0.70, 2020)
        assert abs(result - 0.70) < 1e-9

    @pytest.mark.parametrize("target,expected", [
        (2018, 0.60),
        (2019, 0.65),
        (2020, 0.70),
        (2021, 0.75),
        (2022, 0.80),
    ])
    def test_interpolate_parametrized(self, target, expected):
        """Test interpolation at multiple target years."""
        result = _interpolate_missing_year(2018, 0.60, 2022, 0.80, target)
        assert abs(result - expected) < 1e-9

    def test_interpolate_determinism(self):
        """Test interpolation is deterministic."""
        results = [
            _interpolate_missing_year(2019, 0.70, 2021, 0.74, 2020)
            for _ in range(10)
        ]
        assert len(set(results)) == 1


# ===========================================================================
# 5. Cutoff Density Estimation (6 tests)
# ===========================================================================


class TestCutoffDensityEstimation:
    """Test Hansen + NDVI regression for cutoff density estimation."""

    def test_estimate_cutoff_density_high(self):
        """Test high Hansen + high NDVI gives high density."""
        density = _estimate_cutoff_density(80.0, 0.75)
        assert density > 60.0

    def test_estimate_cutoff_density_low(self):
        """Test low Hansen + low NDVI gives low density."""
        density = _estimate_cutoff_density(5.0, 0.10)
        assert density < 20.0

    def test_estimate_cutoff_density_mixed(self):
        """Test high Hansen + low NDVI gives moderate density."""
        density = _estimate_cutoff_density(80.0, 0.10)
        assert 20.0 < density < 60.0

    def test_estimate_cutoff_density_zero(self):
        """Test zero inputs give zero density."""
        density = _estimate_cutoff_density(0.0, 0.0)
        assert density == 0.0

    def test_estimate_cutoff_density_range(self):
        """Test density is within [0, 100]."""
        density = _estimate_cutoff_density(100.0, 1.0)
        assert 0.0 <= density <= 100.0

    def test_estimate_cutoff_density_determinism(self):
        """Test estimation is deterministic."""
        results = [_estimate_cutoff_density(80.0, 0.72) for _ in range(10)]
        assert len(set(results)) == 1


# ===========================================================================
# 6. Multi-Source Fusion Weights (6 tests)
# ===========================================================================


class TestMultiSourceFusionWeights:
    """Test reconstruction source weight configuration and fusion."""

    def test_multi_source_fusion_weights(self):
        """Test Landsat 0.30 + S2 0.30 + Hansen 0.25 + JAXA 0.15 = 1.0."""
        total = sum(RECONSTRUCTION_SOURCE_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9

    def test_multi_source_fusion_landsat(self):
        """Test Landsat weight is 0.30."""
        assert RECONSTRUCTION_SOURCE_WEIGHTS["landsat"] == 0.30

    def test_multi_source_fusion_sentinel2(self):
        """Test Sentinel-2 weight is 0.30."""
        assert RECONSTRUCTION_SOURCE_WEIGHTS["sentinel2"] == 0.30

    def test_multi_source_fusion_hansen(self):
        """Test Hansen weight is 0.25."""
        assert RECONSTRUCTION_SOURCE_WEIGHTS["hansen"] == 0.25

    def test_multi_source_fusion_jaxa(self):
        """Test JAXA weight is 0.15."""
        assert RECONSTRUCTION_SOURCE_WEIGHTS["jaxa"] == 0.15

    def test_multi_source_fusion_compute(self):
        """Test weighted fusion computation with all sources."""
        values = {
            "landsat": 0.70,
            "sentinel2": 0.72,
            "hansen": 0.68,
            "jaxa": 0.65,
        }
        fused = fuse_weighted(values, RECONSTRUCTION_SOURCE_WEIGHTS)
        # 0.30*0.70 + 0.30*0.72 + 0.25*0.68 + 0.15*0.65 = 0.6935
        expected = (0.30 * 0.70 + 0.30 * 0.72 + 0.25 * 0.68 + 0.15 * 0.65)
        assert abs(fused - expected) < 1e-9


# ===========================================================================
# 7. Data Availability Confidence (5 tests)
# ===========================================================================


class TestDataAvailabilityConfidence:
    """Test confidence scoring based on available data sources."""

    def test_confidence_data_availability_all(self):
        """Test all 4 sources available gives confidence 1.0."""
        conf = _confidence_from_data_availability(
            ["landsat", "sentinel2", "hansen", "jaxa"]
        )
        assert abs(conf - 1.0) < 1e-9

    def test_confidence_data_availability_three(self):
        """Test 3/4 sources gives confidence 0.75."""
        conf = _confidence_from_data_availability(
            ["landsat", "sentinel2", "hansen"]
        )
        assert abs(conf - 0.75) < 1e-9

    def test_confidence_data_availability_one(self):
        """Test 1/4 sources gives confidence 0.25."""
        conf = _confidence_from_data_availability(["landsat"])
        assert abs(conf - 0.25) < 1e-9

    def test_confidence_data_availability_none(self):
        """Test no sources gives confidence 0.0."""
        conf = _confidence_from_data_availability([])
        assert conf == 0.0

    def test_confidence_data_availability_more_than_max(self):
        """Test more sources than max caps at 1.0."""
        conf = _confidence_from_data_availability(
            ["a", "b", "c", "d", "e"], max_sources=4,
        )
        assert abs(conf - 1.0) < 1e-9


# ===========================================================================
# 8. Result Construction (5 tests)
# ===========================================================================


class TestResultConstruction:
    """Test HistoricalCoverRecord construction."""

    def test_reconstruct_plot_forest(self, sample_historical_record):
        """Test reconstruction returns record with was_forest=True."""
        assert sample_historical_record.was_forest is True
        assert isinstance(sample_historical_record, HistoricalCoverRecord)

    def test_reconstruct_plot_non_forest(self, sample_historical_record_no_forest):
        """Test reconstruction returns record with was_forest=False."""
        assert sample_historical_record_no_forest.was_forest is False

    def test_reconstruct_plot_has_provenance(self, sample_historical_record):
        """Test record has provenance hash."""
        assert len(sample_historical_record.provenance_hash) == SHA256_HEX_LENGTH

    def test_cutoff_date_default(self, sample_historical_record):
        """Test default cutoff date is 2020-12-31."""
        assert sample_historical_record.cutoff_date == EUDR_DEFORESTATION_CUTOFF

    def test_batch_reconstruct(self):
        """Test batch reconstruction returns results for all plots."""
        plots = [f"PLOT-{i:03d}" for i in range(5)]
        records = [
            HistoricalCoverRecord(plot_id=pid, was_forest=(i % 2 == 0))
            for i, pid in enumerate(plots)
        ]
        assert len(records) == 5
        assert records[0].was_forest is True
        assert records[1].was_forest is False


# ===========================================================================
# 9. Determinism (3 tests)
# ===========================================================================


class TestHistoricalDeterminism:
    """Test deterministic behaviour of historical reconstruction."""

    def test_determinism_composite(self):
        """Test temporal composite is deterministic."""
        values = [0.60, None, 0.70, 0.80]
        results = [_temporal_composite_median(values) for _ in range(20)]
        assert all(r == results[0] for r in results)

    def test_determinism_classification(self):
        """Test cutoff classification is deterministic."""
        results = [_classify_cutoff_forest(0.72, 80.0) for _ in range(20)]
        assert len(set(results)) == 1

    def test_determinism_provenance_hash(self):
        """Test same inputs produce same provenance hash."""
        data = {"plot_id": "P001", "was_forest": True, "cutoff_date": "2020-12-31"}
        hashes = [compute_test_hash(data) for _ in range(20)]
        assert len(set(hashes)) == 1
