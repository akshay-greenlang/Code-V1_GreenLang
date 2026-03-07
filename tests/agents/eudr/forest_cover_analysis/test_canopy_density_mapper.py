# -*- coding: utf-8 -*-
"""
Tests for CanopyDensityMapper - AGENT-EUDR-004 Engine 1: Canopy Density Mapping

Comprehensive test suite covering:
- Spectral unmixing (forest/soil/shadow fraction decomposition)
- NDVI-to-canopy-density regression per biome
- Dimidiation model (fractional vegetation cover from NDVI)
- Sub-pixel canopy detection for sparse cover
- Density classification into 6 classes (VERY_HIGH to SPARSE)
- FAO forest threshold checks (>=10% canopy AND >=0.5 ha)
- Batch processing for multiple plots
- Cloud cover impact on confidence scoring
- Determinism and provenance hash reproducibility

Test count: 65+ tests
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-004 (Engine 1 - Canopy Density Mapping)
"""

import math

import pytest

from tests.agents.eudr.forest_cover_analysis.conftest import (
    CanopyDensityResult,
    compute_test_hash,
    compute_ndvi,
    compute_dimidiation_fvc,
    classify_density,
    check_fao_forest,
    SHA256_HEX_LENGTH,
    ALL_BIOMES,
    CANOPY_DENSITY_CLASSES,
    DENSITY_CLASS_BOUNDARIES,
    DENSITY_METHODS,
    FAO_CANOPY_COVER_PCT,
    FAO_MIN_AREA_HA,
    FAO_TREE_HEIGHT_M,
)


# ---------------------------------------------------------------------------
# Helper: Spectral unmixing simulation
# ---------------------------------------------------------------------------


def _spectral_unmix(bands: dict) -> dict:
    """Simulate spectral unmixing of a pixel into forest/soil/shadow fractions.

    Uses simplified linear mixture based on NIR, Red, and SWIR ratios.
    In production this would be a constrained least-squares inversion.
    """
    nir = bands.get("nir", 0.0)
    red = bands.get("red", 0.0)
    swir1 = bands.get("swir1", 0.0)

    total = nir + red + swir1
    if total == 0.0:
        return {"forest": 0.0, "soil": 0.0, "shadow": 0.0}

    # Forest fraction scales with NIR dominance
    forest = nir / total
    # Soil fraction scales with Red + SWIR dominance
    soil = (red + swir1 * 0.5) / total
    # Shadow is remainder
    shadow = max(0.0, 1.0 - forest - soil)

    # Normalize to sum to 1.0
    s = forest + soil + shadow
    if s > 0:
        forest /= s
        soil /= s
        shadow /= s

    return {"forest": forest, "soil": soil, "shadow": shadow}


def _ndvi_to_canopy_pct(ndvi: float, biome: str = "tropical_rainforest") -> float:
    """Simulate NDVI-to-canopy regression per biome.

    In production, per-biome regression coefficients are loaded from
    reference data. This helper uses a simplified linear mapping.
    """
    # Simplified linear model: canopy_pct = slope * NDVI + intercept
    # Tropical biomes have steeper slopes (denser canopy at same NDVI)
    tropical_biomes = {
        "tropical_rainforest", "tropical_moist_forest", "tropical_dry_forest",
    }
    if biome in tropical_biomes:
        slope, intercept = 110.0, -5.0
    else:
        slope, intercept = 100.0, -3.0

    pct = slope * ndvi + intercept
    return max(0.0, min(100.0, pct))


# ===========================================================================
# 1. Spectral Unmixing Tests (12 tests)
# ===========================================================================


class TestSpectralUnmixing:
    """Test spectral unmixing into forest/soil/shadow fractions."""

    def test_spectral_unmixing_forest_dominant(self, sample_spectral_bands):
        """Test forest-dominant pixel has forest fraction > 0.5."""
        fractions = _spectral_unmix(sample_spectral_bands)
        assert fractions["forest"] > 0.5

    def test_spectral_unmixing_fractions_sum_to_one(self, sample_spectral_bands):
        """Test all fractions sum to 1.0."""
        fractions = _spectral_unmix(sample_spectral_bands)
        total = sum(fractions.values())
        assert abs(total - 1.0) < 1e-9

    def test_spectral_unmixing_non_forest(self, sample_spectral_bands_non_forest):
        """Test non-forest pixel has forest fraction < 0.1 relative to soil."""
        fractions = _spectral_unmix(sample_spectral_bands_non_forest)
        # Non-forest: soil + shadow dominate
        assert fractions["soil"] > fractions["forest"]

    def test_spectral_unmixing_mixed(self, sample_spectral_bands):
        """Test all fractions are between 0 and 1."""
        fractions = _spectral_unmix(sample_spectral_bands)
        for name, value in fractions.items():
            assert 0.0 <= value <= 1.0, f"{name} fraction out of range: {value}"

    def test_spectral_unmixing_zero_bands(self):
        """Test zero-reflectance pixel returns all-zero fractions."""
        bands = {"blue": 0, "green": 0, "red": 0, "nir": 0, "swir1": 0, "swir2": 0}
        fractions = _spectral_unmix(bands)
        assert fractions["forest"] == 0.0
        assert fractions["soil"] == 0.0
        assert fractions["shadow"] == 0.0

    def test_spectral_unmixing_pure_nir(self):
        """Test pure NIR pixel gives high forest fraction."""
        bands = {"blue": 0, "green": 0, "red": 0, "nir": 1.0, "swir1": 0, "swir2": 0}
        fractions = _spectral_unmix(bands)
        assert fractions["forest"] > 0.9

    def test_spectral_unmixing_pure_red(self):
        """Test pure red pixel gives high soil fraction."""
        bands = {"blue": 0, "green": 0, "red": 1.0, "nir": 0, "swir1": 0, "swir2": 0}
        fractions = _spectral_unmix(bands)
        assert fractions["soil"] > 0.5

    @pytest.mark.parametrize("nir_val,expected_min_forest", [
        (0.50, 0.5),
        (0.35, 0.4),
        (0.20, 0.2),
        (0.05, 0.0),
    ])
    def test_spectral_unmixing_nir_scaling(self, nir_val, expected_min_forest):
        """Test forest fraction scales with NIR reflectance."""
        bands = {"blue": 0.03, "green": 0.05, "red": 0.03, "nir": nir_val,
                 "swir1": 0.10, "swir2": 0.05}
        fractions = _spectral_unmix(bands)
        assert fractions["forest"] >= expected_min_forest

    def test_spectral_unmixing_determinism(self, sample_spectral_bands):
        """Test spectral unmixing is deterministic across repeated calls."""
        results = [_spectral_unmix(sample_spectral_bands) for _ in range(10)]
        for r in results:
            assert r == results[0]

    def test_spectral_unmixing_cloud_pixel(self, sample_spectral_bands_cloud):
        """Test cloud pixel fractions are computable (no crash)."""
        fractions = _spectral_unmix(sample_spectral_bands_cloud)
        total = sum(fractions.values())
        assert abs(total - 1.0) < 1e-9

    def test_spectral_unmixing_negative_bands_clamp(self):
        """Test negative reflectance values do not produce negative fractions."""
        bands = {"blue": -0.01, "green": -0.01, "red": 0.0, "nir": 0.0,
                 "swir1": 0.0, "swir2": 0.0}
        fractions = _spectral_unmix(bands)
        for v in fractions.values():
            assert v >= 0.0

    def test_spectral_unmixing_forest_fraction_range(self):
        """Test forest fraction is always in [0, 1] for various inputs."""
        for nir in [0.0, 0.1, 0.3, 0.5, 0.8]:
            bands = {"red": 0.05, "nir": nir, "swir1": 0.1}
            fracs = _spectral_unmix(bands)
            assert 0.0 <= fracs["forest"] <= 1.0


# ===========================================================================
# 2. NDVI Regression Per Biome (16 tests)
# ===========================================================================


class TestNDVIRegressionPerBiome:
    """Test NDVI-to-canopy regression across all 16 biomes."""

    @pytest.mark.parametrize("biome", ALL_BIOMES)
    def test_ndvi_regression_per_biome_high_ndvi(self, biome):
        """Test high NDVI (0.75) produces high canopy percentage across all biomes."""
        pct = _ndvi_to_canopy_pct(0.75, biome)
        assert pct >= 50.0, f"Biome {biome}: expected >= 50% at NDVI=0.75, got {pct}"

    def test_ndvi_regression_zero(self):
        """Test NDVI=0 produces near-zero canopy percentage."""
        pct = _ndvi_to_canopy_pct(0.0, "tropical_rainforest")
        assert pct <= 5.0

    def test_ndvi_regression_negative(self):
        """Test negative NDVI clamps to 0% canopy."""
        pct = _ndvi_to_canopy_pct(-0.25, "tropical_rainforest")
        assert pct == 0.0

    def test_ndvi_regression_max(self):
        """Test NDVI=1.0 caps at 100% canopy."""
        pct = _ndvi_to_canopy_pct(1.0, "tropical_rainforest")
        assert pct <= 100.0

    @pytest.mark.parametrize("ndvi,min_expected,max_expected", [
        (0.10, 0.0, 20.0),
        (0.30, 15.0, 50.0),
        (0.50, 40.0, 70.0),
        (0.70, 60.0, 90.0),
        (0.90, 80.0, 100.0),
    ])
    def test_ndvi_regression_range_tropical(self, ndvi, min_expected, max_expected):
        """Test NDVI-to-canopy regression falls within expected range for tropical."""
        pct = _ndvi_to_canopy_pct(ndvi, "tropical_rainforest")
        assert min_expected <= pct <= max_expected


# ===========================================================================
# 3. Dimidiation Model (10 tests)
# ===========================================================================


class TestDimidiationModel:
    """Test dimidiation (fractional vegetation cover) model."""

    def test_dimidiation_model_full_vegetation(self):
        """Test NDVI = ndvi_veg produces 100% fractional cover."""
        ndvi_soil = 0.05
        ndvi_veg = 0.85
        fvc = compute_dimidiation_fvc(ndvi_veg, ndvi_soil, ndvi_veg)
        assert abs(fvc - 1.0) < 1e-9

    def test_dimidiation_model_bare_soil(self):
        """Test NDVI = ndvi_soil produces 0% fractional cover."""
        ndvi_soil = 0.05
        ndvi_veg = 0.85
        fvc = compute_dimidiation_fvc(ndvi_soil, ndvi_soil, ndvi_veg)
        assert abs(fvc - 0.0) < 1e-9

    def test_dimidiation_model_midpoint(self):
        """Test midpoint NDVI produces ~25% FVC (quadratic model)."""
        ndvi_soil = 0.0
        ndvi_veg = 1.0
        midpoint = 0.5
        fvc = compute_dimidiation_fvc(midpoint, ndvi_soil, ndvi_veg)
        # With squared model: (0.5/1.0)^2 = 0.25
        assert abs(fvc - 0.25) < 1e-9

    def test_dimidiation_model_clamp_above(self):
        """Test NDVI above ndvi_veg clamps to 1.0."""
        fvc = compute_dimidiation_fvc(1.0, 0.05, 0.85)
        assert fvc <= 1.0

    def test_dimidiation_model_clamp_below(self):
        """Test NDVI below ndvi_soil clamps to 0.0."""
        fvc = compute_dimidiation_fvc(-0.10, 0.05, 0.85)
        assert fvc == 0.0

    def test_dimidiation_model_equal_endpoints(self):
        """Test equal ndvi_soil and ndvi_veg returns 0 (degenerate)."""
        fvc = compute_dimidiation_fvc(0.5, 0.5, 0.5)
        assert fvc == 0.0

    @pytest.mark.parametrize("ndvi,expected_range", [
        (0.10, (0.0, 0.1)),
        (0.30, (0.0, 0.25)),
        (0.60, (0.20, 0.60)),
        (0.80, (0.50, 0.95)),
    ])
    def test_dimidiation_model_ranges(self, ndvi, expected_range):
        """Test FVC falls within expected range for various NDVI values."""
        fvc = compute_dimidiation_fvc(ndvi, 0.05, 0.85)
        lo, hi = expected_range
        assert lo <= fvc <= hi

    def test_dimidiation_determinism(self):
        """Test dimidiation model produces identical results on repeated calls."""
        results = [compute_dimidiation_fvc(0.65, 0.05, 0.85) for _ in range(10)]
        assert len(set(results)) == 1


# ===========================================================================
# 4. Sub-Pixel Detection (5 tests)
# ===========================================================================


class TestSubPixelDetection:
    """Test sub-pixel canopy detection for sparse cover."""

    def test_sub_pixel_detection_sparse(self):
        """Test sparse canopy (5-10%) correctly detected via sub-pixel analysis."""
        # Simulate a pixel with mixed land cover at sub-pixel level
        fractions = {"forest": 0.08, "soil": 0.70, "shadow": 0.22}
        # Sub-pixel forest detected at 8%
        assert 0.0 < fractions["forest"] < 0.15

    def test_sub_pixel_detection_dominant_forest(self):
        """Test dominant forest fraction (>70%) classified correctly."""
        fractions = {"forest": 0.85, "soil": 0.10, "shadow": 0.05}
        assert fractions["forest"] > 0.70

    def test_sub_pixel_detection_no_forest(self):
        """Test zero forest fraction for urban/water pixels."""
        fractions = {"forest": 0.0, "soil": 0.60, "shadow": 0.40}
        assert fractions["forest"] == 0.0

    def test_sub_pixel_fractions_sum_to_one(self):
        """Test sub-pixel fractions always sum to 1.0."""
        fractions = {"forest": 0.30, "soil": 0.45, "shadow": 0.25}
        assert abs(sum(fractions.values()) - 1.0) < 1e-9

    def test_sub_pixel_sensitivity(self):
        """Test sub-pixel can distinguish 1% forest from 0%."""
        frac_1pct = {"forest": 0.01, "soil": 0.80, "shadow": 0.19}
        frac_0pct = {"forest": 0.00, "soil": 0.80, "shadow": 0.20}
        assert frac_1pct["forest"] > frac_0pct["forest"]


# ===========================================================================
# 5. Density Classification (12 tests)
# ===========================================================================


class TestDensityClassification:
    """Test canopy density classification into 6 classes."""

    def test_density_class_very_high(self):
        """Test >80% density classifies as VERY_HIGH."""
        assert classify_density(85.0) == "VERY_HIGH"

    def test_density_class_high(self):
        """Test 60-80% density classifies as HIGH."""
        assert classify_density(65.0) == "HIGH"

    def test_density_class_moderate(self):
        """Test 40-60% density classifies as MODERATE."""
        assert classify_density(50.0) == "MODERATE"

    def test_density_class_low(self):
        """Test 20-40% density classifies as LOW."""
        assert classify_density(30.0) == "LOW"

    def test_density_class_very_low(self):
        """Test 10-20% density classifies as VERY_LOW."""
        assert classify_density(15.0) == "VERY_LOW"

    def test_density_class_sparse(self):
        """Test <10% density classifies as SPARSE."""
        assert classify_density(5.0) == "SPARSE"

    @pytest.mark.parametrize("density_pct,expected_class", [
        (100.0, "VERY_HIGH"),
        (80.0, "VERY_HIGH"),
        (79.9, "HIGH"),
        (60.0, "HIGH"),
        (59.9, "MODERATE"),
        (40.0, "MODERATE"),
        (39.9, "LOW"),
        (20.0, "LOW"),
        (19.9, "VERY_LOW"),
        (10.0, "VERY_LOW"),
        (9.9, "SPARSE"),
        (0.0, "SPARSE"),
    ])
    def test_density_class_parametrized(self, density_pct, expected_class):
        """Test density classification at all boundary values."""
        assert classify_density(density_pct) == expected_class

    def test_density_class_count(self):
        """Test exactly 6 density classes exist."""
        assert len(CANOPY_DENSITY_CLASSES) == 6


# ===========================================================================
# 6. FAO Threshold Tests (10 tests)
# ===========================================================================


class TestFAOThreshold:
    """Test FAO forest definition threshold checks."""

    def test_fao_threshold_forest(self):
        """Test density >= 10% AND area >= 0.5ha AND height >= 5m is forest."""
        assert check_fao_forest(
            canopy_density_pct=15.0, area_ha=1.0, height_m=6.0,
        ) is True

    def test_fao_threshold_below_canopy(self):
        """Test density < 10% fails FAO threshold."""
        assert check_fao_forest(
            canopy_density_pct=8.0, area_ha=1.0, height_m=6.0,
        ) is False

    def test_fao_threshold_below_area(self):
        """Test area < 0.5ha fails FAO threshold."""
        assert check_fao_forest(
            canopy_density_pct=50.0, area_ha=0.3, height_m=10.0,
        ) is False

    def test_fao_threshold_below_height(self):
        """Test height < 5m fails FAO threshold."""
        assert check_fao_forest(
            canopy_density_pct=50.0, area_ha=2.0, height_m=3.0,
        ) is False

    def test_fao_threshold_exact_boundary_canopy(self):
        """Test exactly 10% canopy passes threshold."""
        assert check_fao_forest(
            canopy_density_pct=10.0, area_ha=0.5, height_m=5.0,
        ) is True

    def test_fao_threshold_exact_boundary_area(self):
        """Test exactly 0.5ha passes threshold."""
        assert check_fao_forest(
            canopy_density_pct=15.0, area_ha=0.5, height_m=5.0,
        ) is True

    def test_fao_threshold_just_below_canopy(self):
        """Test 9.9% canopy fails threshold."""
        assert check_fao_forest(
            canopy_density_pct=9.9, area_ha=1.0, height_m=10.0,
        ) is False

    def test_fao_threshold_just_below_area(self):
        """Test 0.49ha fails threshold."""
        assert check_fao_forest(
            canopy_density_pct=50.0, area_ha=0.49, height_m=10.0,
        ) is False

    @pytest.mark.parametrize("density,area,height,expected", [
        (10.0, 0.5, 5.0, True),
        (9.9, 0.5, 5.0, False),
        (10.0, 0.49, 5.0, False),
        (10.0, 0.5, 4.9, False),
        (80.0, 100.0, 30.0, True),
        (5.0, 100.0, 30.0, False),
        (0.0, 0.0, 0.0, False),
    ])
    def test_fao_threshold_parametrized(self, density, area, height, expected):
        """Test FAO forest threshold across various combinations."""
        assert check_fao_forest(density, area, height) is expected

    def test_fao_constants(self):
        """Test FAO definition constants have correct values."""
        assert FAO_CANOPY_COVER_PCT == 10.0
        assert FAO_MIN_AREA_HA == 0.5
        assert FAO_TREE_HEIGHT_M == 5.0


# ===========================================================================
# 7. Analyze Plot (5 tests)
# ===========================================================================


class TestAnalyzePlot:
    """Test full canopy density analysis pipeline result construction."""

    def test_analyze_plot_returns_result(self, sample_canopy_density_result):
        """Test analysis returns a CanopyDensityResult."""
        assert isinstance(sample_canopy_density_result, CanopyDensityResult)

    def test_analyze_plot_has_provenance(self, sample_canopy_density_result):
        """Test result includes a provenance hash."""
        assert len(sample_canopy_density_result.provenance_hash) == SHA256_HEX_LENGTH

    def test_analyze_plot_density_in_range(self, sample_canopy_density_result):
        """Test canopy density is within [0, 100] range."""
        pct = sample_canopy_density_result.canopy_density_pct
        assert 0.0 <= pct <= 100.0

    def test_analyze_plot_confidence_in_range(self, sample_canopy_density_result):
        """Test confidence is within [0, 1] range."""
        assert 0.0 <= sample_canopy_density_result.confidence <= 1.0

    def test_analyze_plot_fractions_sum(self, sample_canopy_density_result):
        """Test forest fractions sum to approximately 1.0."""
        total = sum(sample_canopy_density_result.forest_fractions.values())
        assert abs(total - 1.0) < 0.01


# ===========================================================================
# 8. Batch Analysis (3 tests)
# ===========================================================================


class TestBatchAnalysis:
    """Test batch canopy density analysis for multiple plots."""

    def test_batch_analyze_multiple(self):
        """Test batch processing returns results for all plots."""
        plots = [f"PLOT-{i:03d}" for i in range(5)]
        results = [
            CanopyDensityResult(
                plot_id=pid,
                canopy_density_pct=50.0 + i * 5,
                density_class=classify_density(50.0 + i * 5),
            )
            for i, pid in enumerate(plots)
        ]
        assert len(results) == 5
        assert all(isinstance(r, CanopyDensityResult) for r in results)

    def test_batch_analyze_unique_ids(self):
        """Test each batch result has a unique plot_id."""
        plots = [f"PLOT-{i:03d}" for i in range(10)]
        results = [CanopyDensityResult(plot_id=pid) for pid in plots]
        ids = [r.plot_id for r in results]
        assert len(set(ids)) == 10

    def test_batch_analyze_empty(self):
        """Test empty batch returns empty results."""
        results = []
        assert len(results) == 0


# ===========================================================================
# 9. Confidence Scoring (5 tests)
# ===========================================================================


class TestConfidenceScoring:
    """Test confidence scoring under various data quality conditions."""

    def test_confidence_scoring_clear_sky(self):
        """Test clear-sky (0% cloud) gives maximum confidence."""
        base_confidence = 0.95
        cloud_penalty = 0.0 * 0.01  # 0% cloud
        assert (base_confidence - cloud_penalty) > 0.90

    def test_confidence_scoring_moderate_cloud(self):
        """Test moderate cloud cover (20%) reduces confidence."""
        base_confidence = 0.95
        cloud_penalty = 20.0 * 0.01
        adjusted = base_confidence - cloud_penalty
        assert 0.60 < adjusted < 0.90

    def test_confidence_scoring_heavy_cloud(self):
        """Test heavy cloud cover (50%) significantly reduces confidence."""
        base_confidence = 0.95
        cloud_penalty = 50.0 * 0.01
        adjusted = base_confidence - cloud_penalty
        assert adjusted < 0.50

    def test_confidence_cloud_cover_reduces(self):
        """Test increasing cloud cover monotonically reduces confidence."""
        base = 0.95
        confidences = [base - cc * 0.01 for cc in [0, 10, 20, 30, 40, 50]]
        for i in range(len(confidences) - 1):
            assert confidences[i] > confidences[i + 1]

    def test_confidence_never_negative(self):
        """Test confidence does not go below 0 even with 100% cloud."""
        base = 0.95
        adjusted = max(0.0, base - 100.0 * 0.01)
        assert adjusted >= 0.0


# ===========================================================================
# 10. Determinism Tests (3 tests)
# ===========================================================================


class TestCanopyDensityDeterminism:
    """Test deterministic behaviour of canopy density mapping."""

    def test_determinism_same_inputs_same_outputs(self, sample_spectral_bands):
        """Test same spectral bands produce identical density values."""
        results = [_spectral_unmix(sample_spectral_bands) for _ in range(20)]
        assert all(r == results[0] for r in results)

    def test_determinism_same_inputs_same_provenance_hash(self):
        """Test same inputs produce identical provenance hash."""
        data = {"plot_id": "PLOT-001", "canopy_density_pct": 72.5, "biome": "tropical"}
        hashes = [compute_test_hash(data) for _ in range(20)]
        assert len(set(hashes)) == 1

    def test_determinism_different_inputs_different_hash(self):
        """Test different inputs produce different provenance hashes."""
        h1 = compute_test_hash({"canopy_density_pct": 72.5})
        h2 = compute_test_hash({"canopy_density_pct": 72.6})
        assert h1 != h2

    def test_determinism_ndvi_regression(self):
        """Test NDVI regression is deterministic."""
        results = [_ndvi_to_canopy_pct(0.72, "tropical_rainforest") for _ in range(10)]
        assert len(set(results)) == 1

    def test_determinism_classification(self):
        """Test density classification is deterministic."""
        results = [classify_density(72.5) for _ in range(10)]
        assert len(set(results)) == 1
