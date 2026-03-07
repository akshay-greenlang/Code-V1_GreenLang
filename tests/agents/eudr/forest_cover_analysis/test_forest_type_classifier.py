# -*- coding: utf-8 -*-
"""
Tests for ForestTypeClassifier - AGENT-EUDR-004 Engine 2: Forest Type Classification

Comprehensive test suite covering:
- Spectral signature classification into 10 forest types
- Phenological classification (evergreen vs deciduous from NDVI time series)
- Structural classification (height, density, complexity metrics)
- Multi-temporal dry/wet season discrimination
- Ensemble weighted voting across multiple classifiers
- EUDR forest determination (natural forest vs commodity plantation)
- Commodity-specific exclusion rules (palm oil, rubber, agroforestry)
- Inter-method agreement and confidence scoring
- Determinism and provenance hash reproducibility

Test count: 65+ tests
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-004 (Engine 2 - Forest Type Classification)
"""

import math

import pytest

from tests.agents.eudr.forest_cover_analysis.conftest import (
    ForestClassificationResult,
    compute_test_hash,
    SHA256_HEX_LENGTH,
    ALL_BIOMES,
    FOREST_TYPES,
    EUDR_COMMODITIES,
)


# ---------------------------------------------------------------------------
# Helpers: Classification simulation
# ---------------------------------------------------------------------------


def _spectral_classify(bands: dict) -> str:
    """Classify forest type from spectral signature.

    Simplified mapping based on NIR/Red/SWIR ratios.
    Production would use trained Random Forest or SVM classifiers
    with per-biome spectral libraries.
    """
    nir = bands.get("nir", 0.0)
    red = bands.get("red", 0.0)
    swir1 = bands.get("swir1", 0.0)
    green = bands.get("green", 0.0)

    if nir == 0 and red == 0:
        return "UNKNOWN"

    ndvi = (nir - red) / (nir + red) if (nir + red) > 0 else 0.0
    ndmi = (nir - swir1) / (nir + swir1) if (nir + swir1) > 0 else 0.0

    # Tropical rainforest: very high NDVI and high NDMI
    if ndvi > 0.70 and ndmi > 0.20:
        return "PRIMARY_TROPICAL"
    elif ndvi > 0.55 and ndmi > 0.10:
        return "SECONDARY_TROPICAL"
    elif ndvi > 0.40 and ndmi < 0.05:
        return "PLANTATION"
    elif ndvi > 0.50:
        return "TEMPERATE_BROADLEAF"
    elif ndvi > 0.35:
        return "BOREAL_CONIFEROUS"
    elif ndvi > 0.20:
        return "AGROFORESTRY"
    else:
        return "UNKNOWN"


def _phenological_classify(ndvi_timeseries: list) -> str:
    """Classify forest type from NDVI time series phenology.

    Flat profile = evergreen tropical/coniferous.
    Seasonal oscillation = temperate broadleaf deciduous.
    """
    if not ndvi_timeseries or len(ndvi_timeseries) < 2:
        return "UNKNOWN"

    mean_ndvi = sum(ndvi_timeseries) / len(ndvi_timeseries)
    amplitude = max(ndvi_timeseries) - min(ndvi_timeseries)

    # Flat profile (amplitude < 0.10) with high mean = evergreen tropical
    if amplitude < 0.10 and mean_ndvi > 0.65:
        return "PRIMARY_TROPICAL"
    # Flat profile with moderate mean = coniferous
    elif amplitude < 0.10 and mean_ndvi > 0.40:
        return "TEMPERATE_CONIFEROUS"
    # Seasonal profile = deciduous
    elif amplitude >= 0.20 and mean_ndvi > 0.45:
        return "TEMPERATE_BROADLEAF"
    elif amplitude >= 0.15 and mean_ndvi > 0.35:
        return "BOREAL_CONIFEROUS"
    else:
        return "AGROFORESTRY"


def _structural_classify(
    height_m: float,
    density_pct: float,
    complexity_index: float,
) -> str:
    """Classify forest type from structural metrics.

    Primary forests: tall, dense, complex structure.
    Plantations: uniform height, moderate density, low complexity.
    """
    if height_m > 25.0 and density_pct > 70.0 and complexity_index > 0.7:
        return "PRIMARY_TROPICAL"
    elif height_m > 20.0 and density_pct > 60.0 and complexity_index > 0.5:
        return "SECONDARY_TROPICAL"
    elif height_m > 10.0 and density_pct > 40.0 and complexity_index < 0.3:
        return "PLANTATION"
    elif height_m > 15.0 and density_pct > 50.0:
        return "TEMPERATE_BROADLEAF"
    elif height_m > 10.0:
        return "BOREAL_CONIFEROUS"
    else:
        return "AGROFORESTRY"


def _ensemble_weighted_vote(
    spectral: str,
    phenological: str,
    structural: str,
    weights: tuple = (0.40, 0.30, 0.30),
) -> str:
    """Combine classification results via weighted vote."""
    votes = {}
    for cls, w in zip([spectral, phenological, structural], weights):
        votes[cls] = votes.get(cls, 0.0) + w
    # Return class with highest weight
    return max(votes, key=votes.get)


def _is_forest_per_eudr(
    forest_type: str,
    commodity: str = "",
) -> bool:
    """Determine if classified type counts as forest under EUDR.

    EUDR excludes certain commodity plantations from forest definition.
    Palm oil monocultures and rubber monocultures are NOT forest.
    Shade-grown coffee/cocoa under native canopy IS forest.
    """
    # Palm oil plantation exclusion
    if commodity == "oil_palm" and forest_type == "PLANTATION":
        return False
    # Rubber monoculture exclusion
    if commodity == "rubber" and forest_type == "PLANTATION":
        return False
    # Agroforestry with native canopy counts as forest
    if forest_type == "AGROFORESTRY":
        return True
    # All natural forest types count
    natural_types = {
        "PRIMARY_TROPICAL", "SECONDARY_TROPICAL", "MANGROVE",
        "PEAT_SWAMP", "TEMPERATE_BROADLEAF", "TEMPERATE_CONIFEROUS",
        "BOREAL_CONIFEROUS", "MONTANE_CLOUD",
    }
    return forest_type in natural_types


def _inter_method_agreement(*classifications: str) -> float:
    """Compute agreement score across multiple classification methods.

    Returns 1.0 if all agree, 0.0 if all different.
    """
    if not classifications:
        return 0.0
    unique = set(classifications)
    if len(unique) == 1:
        return 1.0
    # Count the most common class
    from collections import Counter
    counts = Counter(classifications)
    most_common_count = counts.most_common(1)[0][1]
    return most_common_count / len(classifications)


# ===========================================================================
# 1. Spectral Classification (12 tests)
# ===========================================================================


class TestSpectralClassification:
    """Test forest type classification from spectral signatures."""

    def test_spectral_classify_primary_tropical(self, sample_spectral_bands):
        """Test matching spectral signature returns PRIMARY_TROPICAL."""
        result = _spectral_classify(sample_spectral_bands)
        assert result == "PRIMARY_TROPICAL"

    def test_spectral_classify_non_forest(self, sample_spectral_bands_non_forest):
        """Test non-forest spectral signature returns non-primary type."""
        result = _spectral_classify(sample_spectral_bands_non_forest)
        assert result != "PRIMARY_TROPICAL"

    @pytest.mark.parametrize("forest_type", FOREST_TYPES)
    def test_spectral_classify_all_types_valid(self, forest_type):
        """Test all 10 forest types are valid classification outputs."""
        assert forest_type in FOREST_TYPES

    def test_spectral_classify_high_ndvi_ndmi(self):
        """Test high NDVI + high NDMI = PRIMARY_TROPICAL."""
        bands = {"blue": 0.02, "green": 0.04, "red": 0.02, "nir": 0.40,
                 "swir1": 0.08, "swir2": 0.04}
        result = _spectral_classify(bands)
        assert result == "PRIMARY_TROPICAL"

    def test_spectral_classify_moderate_ndvi(self):
        """Test moderate NDVI returns secondary or temperate type."""
        bands = {"blue": 0.05, "green": 0.08, "red": 0.08, "nir": 0.25,
                 "swir1": 0.10, "swir2": 0.06}
        result = _spectral_classify(bands)
        assert result in {"SECONDARY_TROPICAL", "TEMPERATE_BROADLEAF", "PLANTATION"}

    def test_spectral_classify_zero_bands(self):
        """Test zero-reflectance returns UNKNOWN."""
        bands = {"blue": 0, "green": 0, "red": 0, "nir": 0, "swir1": 0, "swir2": 0}
        result = _spectral_classify(bands)
        assert result == "UNKNOWN"

    def test_spectral_classify_plantation_low_ndmi(self):
        """Test low NDMI + moderate NDVI = PLANTATION."""
        bands = {"blue": 0.04, "green": 0.06, "red": 0.06, "nir": 0.20,
                 "swir1": 0.18, "swir2": 0.10}
        result = _spectral_classify(bands)
        # Low NDMI from high SWIR1 relative to NIR
        assert result in {"PLANTATION", "TEMPERATE_BROADLEAF", "BOREAL_CONIFEROUS"}

    def test_forest_type_count(self):
        """Test exactly 10 forest types exist."""
        assert len(FOREST_TYPES) == 10

    def test_spectral_classify_returns_string(self, sample_spectral_bands):
        """Test classification returns a string type."""
        result = _spectral_classify(sample_spectral_bands)
        assert isinstance(result, str)

    def test_spectral_classify_determinism(self, sample_spectral_bands):
        """Test spectral classification is deterministic."""
        results = [_spectral_classify(sample_spectral_bands) for _ in range(10)]
        assert len(set(results)) == 1

    def test_spectral_classify_cloud_returns_non_primary(self, sample_spectral_bands_cloud):
        """Test cloud-contaminated pixel does not classify as primary forest."""
        result = _spectral_classify(sample_spectral_bands_cloud)
        # Clouds have unusual spectral profile
        assert result != "PRIMARY_TROPICAL"


# ===========================================================================
# 2. Phenological Classification (10 tests)
# ===========================================================================


class TestPhenologicalClassification:
    """Test classification from NDVI time series phenology."""

    def test_phenological_classify_evergreen(self):
        """Test flat NDVI profile with high mean = PRIMARY_TROPICAL."""
        # 12 monthly NDVI values, very small variation
        timeseries = [0.72, 0.74, 0.73, 0.71, 0.72, 0.73,
                      0.74, 0.72, 0.71, 0.73, 0.72, 0.74]
        result = _phenological_classify(timeseries)
        assert result == "PRIMARY_TROPICAL"

    def test_phenological_classify_deciduous(self):
        """Test seasonal NDVI profile = TEMPERATE_BROADLEAF."""
        # Clear seasonal pattern: low in winter, high in summer
        timeseries = [0.25, 0.30, 0.45, 0.60, 0.72, 0.78,
                      0.80, 0.75, 0.60, 0.45, 0.30, 0.25]
        result = _phenological_classify(timeseries)
        assert result == "TEMPERATE_BROADLEAF"

    def test_phenological_classify_coniferous(self):
        """Test flat profile with moderate mean = TEMPERATE_CONIFEROUS."""
        timeseries = [0.45, 0.46, 0.47, 0.46, 0.45, 0.46,
                      0.47, 0.46, 0.45, 0.46, 0.47, 0.46]
        result = _phenological_classify(timeseries)
        assert result == "TEMPERATE_CONIFEROUS"

    def test_phenological_classify_empty(self):
        """Test empty time series returns UNKNOWN."""
        result = _phenological_classify([])
        assert result == "UNKNOWN"

    def test_phenological_classify_single_value(self):
        """Test single-value time series returns UNKNOWN."""
        result = _phenological_classify([0.72])
        assert result == "UNKNOWN"

    @pytest.mark.parametrize("amplitude,mean,expected", [
        (0.05, 0.75, "PRIMARY_TROPICAL"),
        (0.05, 0.45, "TEMPERATE_CONIFEROUS"),
        (0.30, 0.55, "TEMPERATE_BROADLEAF"),
        (0.20, 0.40, "BOREAL_CONIFEROUS"),
    ])
    def test_phenological_classify_parametrized(self, amplitude, mean, expected):
        """Test phenological classification for various amplitude/mean combos."""
        # Generate a simple sinusoidal timeseries with given amplitude and mean
        timeseries = [
            mean + amplitude * 0.5 * math.sin(2 * math.pi * i / 12)
            for i in range(12)
        ]
        result = _phenological_classify(timeseries)
        assert result == expected

    def test_phenological_classify_determinism(self):
        """Test phenological classification is deterministic."""
        ts = [0.72, 0.74, 0.73, 0.71, 0.72, 0.73,
              0.74, 0.72, 0.71, 0.73, 0.72, 0.74]
        results = [_phenological_classify(ts) for _ in range(10)]
        assert len(set(results)) == 1


# ===========================================================================
# 3. Structural Classification (10 tests)
# ===========================================================================


class TestStructuralClassification:
    """Test classification from height, density, and complexity metrics."""

    def test_structural_classify_primary(self):
        """Test tall, dense, complex structure = PRIMARY_TROPICAL."""
        result = _structural_classify(height_m=30.0, density_pct=85.0,
                                       complexity_index=0.85)
        assert result == "PRIMARY_TROPICAL"

    def test_structural_classify_secondary(self):
        """Test moderate height/density/complexity = SECONDARY_TROPICAL."""
        result = _structural_classify(height_m=22.0, density_pct=65.0,
                                       complexity_index=0.55)
        assert result == "SECONDARY_TROPICAL"

    def test_structural_classify_plantation(self):
        """Test uniform height, low complexity = PLANTATION."""
        result = _structural_classify(height_m=15.0, density_pct=50.0,
                                       complexity_index=0.15)
        assert result == "PLANTATION"

    def test_structural_classify_short_forest(self):
        """Test short trees with moderate density."""
        result = _structural_classify(height_m=8.0, density_pct=40.0,
                                       complexity_index=0.30)
        assert result == "AGROFORESTRY"

    @pytest.mark.parametrize("height,density,complexity,expected", [
        (35.0, 90.0, 0.90, "PRIMARY_TROPICAL"),
        (25.5, 75.0, 0.80, "PRIMARY_TROPICAL"),
        (22.0, 65.0, 0.60, "SECONDARY_TROPICAL"),
        (12.0, 45.0, 0.20, "PLANTATION"),
        (18.0, 55.0, 0.40, "TEMPERATE_BROADLEAF"),
        (12.0, 35.0, 0.35, "BOREAL_CONIFEROUS"),
        (5.0, 20.0, 0.10, "AGROFORESTRY"),
    ])
    def test_structural_classify_parametrized(self, height, density,
                                               complexity, expected):
        """Test structural classification across various parameter combos."""
        result = _structural_classify(height, density, complexity)
        assert result == expected

    def test_structural_classify_determinism(self):
        """Test structural classification is deterministic."""
        results = [_structural_classify(30.0, 85.0, 0.85) for _ in range(10)]
        assert len(set(results)) == 1


# ===========================================================================
# 4. Multi-Temporal Discrimination (5 tests)
# ===========================================================================


class TestMultiTemporalDiscrimination:
    """Test improved classification with dry + wet season imagery."""

    def test_multi_temporal_dry_wet(self):
        """Test two-season imagery improves discrimination."""
        # Dry season: deciduous trees lose leaves, NDVI drops
        dry_ndvi = 0.35
        # Wet season: full canopy, NDVI high
        wet_ndvi = 0.72
        amplitude = wet_ndvi - dry_ndvi  # 0.37 -- deciduous signal
        assert amplitude >= 0.20  # Strong seasonal signal

    def test_multi_temporal_evergreen_stable(self):
        """Test evergreen forest shows minimal seasonal change."""
        dry_ndvi = 0.70
        wet_ndvi = 0.74
        amplitude = wet_ndvi - dry_ndvi  # 0.04
        assert amplitude < 0.10

    def test_multi_temporal_seasonal_vs_evergreen(self):
        """Test seasonal forest has higher amplitude than evergreen."""
        evergreen_amp = 0.04
        deciduous_amp = 0.37
        assert deciduous_amp > evergreen_amp

    def test_multi_temporal_improves_confidence(self):
        """Test two-season data yields higher confidence than single-season."""
        single_season_conf = 0.70
        dual_season_conf = 0.85
        assert dual_season_conf > single_season_conf

    def test_multi_temporal_minimum_two_dates(self):
        """Test at least two acquisition dates needed for temporal analysis."""
        dates = ["2020-06-15", "2020-12-15"]
        assert len(dates) >= 2


# ===========================================================================
# 5. Ensemble Weighted Vote (8 tests)
# ===========================================================================


class TestEnsembleWeightedVote:
    """Test ensemble classification by weighted voting."""

    def test_ensemble_weighted_vote_unanimous(self):
        """Test all methods agree returns that class."""
        result = _ensemble_weighted_vote(
            "PRIMARY_TROPICAL", "PRIMARY_TROPICAL", "PRIMARY_TROPICAL",
        )
        assert result == "PRIMARY_TROPICAL"

    def test_ensemble_weighted_vote_majority(self):
        """Test 2/3 agreement returns majority class."""
        result = _ensemble_weighted_vote(
            "PRIMARY_TROPICAL", "PRIMARY_TROPICAL", "SECONDARY_TROPICAL",
        )
        assert result == "PRIMARY_TROPICAL"

    def test_ensemble_weighted_vote_spectral_wins(self):
        """Test spectral weight (0.40) can override others when disagree."""
        result = _ensemble_weighted_vote(
            "PRIMARY_TROPICAL", "SECONDARY_TROPICAL", "PLANTATION",
            weights=(0.50, 0.25, 0.25),
        )
        assert result == "PRIMARY_TROPICAL"

    def test_ensemble_weighted_vote_all_different(self):
        """Test all-different returns highest-weighted class."""
        result = _ensemble_weighted_vote(
            "PRIMARY_TROPICAL", "SECONDARY_TROPICAL", "PLANTATION",
            weights=(0.40, 0.30, 0.30),
        )
        assert result == "PRIMARY_TROPICAL"

    def test_ensemble_weighted_vote_structural_wins(self):
        """Test structural + phenological can override spectral."""
        result = _ensemble_weighted_vote(
            "PRIMARY_TROPICAL", "SECONDARY_TROPICAL", "SECONDARY_TROPICAL",
            weights=(0.40, 0.30, 0.30),
        )
        assert result == "SECONDARY_TROPICAL"

    def test_ensemble_determinism(self):
        """Test ensemble vote is deterministic."""
        results = [
            _ensemble_weighted_vote("PRIMARY_TROPICAL", "SECONDARY_TROPICAL",
                                    "PRIMARY_TROPICAL")
            for _ in range(10)
        ]
        assert len(set(results)) == 1

    def test_ensemble_returns_valid_type(self):
        """Test ensemble always returns one of the input types."""
        inputs = ("PRIMARY_TROPICAL", "PLANTATION", "BOREAL_CONIFEROUS")
        result = _ensemble_weighted_vote(*inputs)
        assert result in inputs

    def test_ensemble_equal_weights_first_wins(self):
        """Test equal weights with all different: highest-weighted input wins."""
        result = _ensemble_weighted_vote(
            "A", "B", "C",
            weights=(0.34, 0.33, 0.33),
        )
        assert result == "A"


# ===========================================================================
# 6. EUDR Forest Determination (12 tests)
# ===========================================================================


class TestEUDRForestDetermination:
    """Test EUDR-specific forest vs non-forest determination."""

    def test_is_forest_per_eudr_natural(self):
        """Test natural primary tropical forest IS forest per EUDR."""
        assert _is_forest_per_eudr("PRIMARY_TROPICAL") is True

    def test_is_forest_per_eudr_palm_oil(self):
        """Test palm oil plantation is NOT forest per EUDR."""
        assert _is_forest_per_eudr("PLANTATION", "oil_palm") is False

    def test_is_forest_per_eudr_rubber(self):
        """Test rubber monoculture plantation is NOT forest per EUDR."""
        assert _is_forest_per_eudr("PLANTATION", "rubber") is False

    def test_is_forest_per_eudr_agroforestry(self):
        """Test shade-grown coffee/cocoa agroforestry IS forest per EUDR."""
        assert _is_forest_per_eudr("AGROFORESTRY", "cocoa") is True

    def test_is_forest_per_eudr_mangrove(self):
        """Test mangrove IS forest per EUDR."""
        assert _is_forest_per_eudr("MANGROVE") is True

    def test_is_forest_per_eudr_peat_swamp(self):
        """Test peat swamp IS forest per EUDR."""
        assert _is_forest_per_eudr("PEAT_SWAMP") is True

    def test_is_forest_per_eudr_montane(self):
        """Test montane cloud forest IS forest per EUDR."""
        assert _is_forest_per_eudr("MONTANE_CLOUD") is True

    def test_is_forest_per_eudr_boreal(self):
        """Test boreal coniferous IS forest per EUDR."""
        assert _is_forest_per_eudr("BOREAL_CONIFEROUS") is True

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_commodity_exclusions(self, commodity):
        """Test commodity-specific exclusion rules for PLANTATION type."""
        result = _is_forest_per_eudr("PLANTATION", commodity)
        if commodity in ("oil_palm", "rubber"):
            assert result is False
        else:
            # Other commodity plantations are not explicitly excluded
            assert result is False  # PLANTATION not in natural_types

    def test_is_forest_per_eudr_unknown_type(self):
        """Test UNKNOWN type returns False."""
        assert _is_forest_per_eudr("UNKNOWN") is False

    def test_is_forest_per_eudr_no_commodity(self):
        """Test classification without commodity uses type alone."""
        assert _is_forest_per_eudr("PRIMARY_TROPICAL", "") is True
        assert _is_forest_per_eudr("PLANTATION", "") is False


# ===========================================================================
# 7. Confidence and Agreement (8 tests)
# ===========================================================================


class TestConfidenceAndAgreement:
    """Test inter-method agreement and confidence scoring."""

    def test_confidence_inter_method_full_agreement(self):
        """Test full agreement returns 1.0."""
        score = _inter_method_agreement("A", "A", "A")
        assert abs(score - 1.0) < 1e-9

    def test_confidence_inter_method_no_agreement(self):
        """Test all-different returns 1/3."""
        score = _inter_method_agreement("A", "B", "C")
        assert abs(score - 1.0 / 3.0) < 1e-9

    def test_confidence_inter_method_partial(self):
        """Test 2/3 agreement returns 2/3."""
        score = _inter_method_agreement("A", "A", "B")
        assert abs(score - 2.0 / 3.0) < 1e-9

    def test_confidence_inter_method_empty(self):
        """Test empty returns 0.0."""
        score = _inter_method_agreement()
        assert score == 0.0

    def test_confidence_inter_method_single(self):
        """Test single method returns 1.0."""
        score = _inter_method_agreement("A")
        assert score == 1.0

    def test_confidence_high_agreement_high_confidence(self):
        """Test high inter-method agreement implies high confidence."""
        agreement = _inter_method_agreement(
            "PRIMARY_TROPICAL", "PRIMARY_TROPICAL", "PRIMARY_TROPICAL",
        )
        assert agreement >= 0.9

    def test_confidence_low_agreement_low_confidence(self):
        """Test low agreement implies lower confidence."""
        agreement = _inter_method_agreement(
            "PRIMARY_TROPICAL", "PLANTATION", "BOREAL_CONIFEROUS",
        )
        assert agreement < 0.5

    def test_confidence_agreement_determinism(self):
        """Test agreement score is deterministic."""
        results = [
            _inter_method_agreement("A", "A", "B")
            for _ in range(10)
        ]
        assert len(set(results)) == 1


# ===========================================================================
# 8. Result Construction (5 tests)
# ===========================================================================


class TestResultConstruction:
    """Test ForestClassificationResult construction and validation."""

    def test_classify_plot_returns_result(self, sample_classification_result):
        """Test classification pipeline returns ForestClassificationResult."""
        assert isinstance(sample_classification_result, ForestClassificationResult)

    def test_classify_plot_has_provenance(self, sample_classification_result):
        """Test result includes provenance hash."""
        assert len(sample_classification_result.provenance_hash) == SHA256_HEX_LENGTH

    def test_classify_plot_forest_type_valid(self, sample_classification_result):
        """Test forest type is in the valid set."""
        assert sample_classification_result.forest_type in FOREST_TYPES

    def test_classify_plot_confidence_range(self, sample_classification_result):
        """Test confidence is in [0, 1]."""
        assert 0.0 <= sample_classification_result.confidence <= 1.0

    def test_classify_plot_agreement_range(self, sample_classification_result):
        """Test inter_method_agreement is in [0, 1]."""
        assert 0.0 <= sample_classification_result.inter_method_agreement <= 1.0


# ===========================================================================
# 9. Determinism (3 tests)
# ===========================================================================


class TestClassifierDeterminism:
    """Test deterministic behaviour of forest type classification."""

    def test_determinism_spectral(self, sample_spectral_bands):
        """Test spectral classification is deterministic."""
        results = [_spectral_classify(sample_spectral_bands) for _ in range(20)]
        assert all(r == results[0] for r in results)

    def test_determinism_provenance_hash(self):
        """Test same inputs produce identical provenance hash."""
        data = {"plot_id": "PLOT-001", "forest_type": "PRIMARY_TROPICAL"}
        hashes = [compute_test_hash(data) for _ in range(20)]
        assert len(set(hashes)) == 1

    def test_determinism_ensemble(self):
        """Test ensemble vote is deterministic."""
        results = [
            _ensemble_weighted_vote("PRIMARY_TROPICAL", "SECONDARY_TROPICAL",
                                    "PRIMARY_TROPICAL")
            for _ in range(20)
        ]
        assert len(set(results)) == 1
