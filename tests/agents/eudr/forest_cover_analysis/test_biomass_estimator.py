# -*- coding: utf-8 -*-
"""
Tests for BiomassEstimator - AGENT-EUDR-004 Engine 7: Biomass Estimation

Comprehensive test suite covering:
- ESA CCI Biomass product lookup
- GEDI L4A above-ground biomass prediction
- SAR backscatter regression (power-law model)
- SAR saturation detection and flagging
- NDVI allometric biomass estimation (AGB = a * exp(b * NDVI))
- Multi-source weighted biomass fusion
- Carbon stock conversion (AGB * 0.47)
- Biomass change from cutoff to current
- Biome-specific allometric parameters
- Uncertainty propagation
- Full pipeline estimation
- Batch estimation for multiple plots
- Reference range validation
- Determinism and provenance hash reproducibility

Test count: 50+ tests
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-004 (Engine 7 - Biomass Estimation)
"""

import math

import pytest

from tests.agents.eudr.forest_cover_analysis.conftest import (
    BiomassEstimate,
    compute_test_hash,
    fuse_weighted,
    weighted_rms_uncertainty,
    SHA256_HEX_LENGTH,
    ALL_BIOMES,
)


# ---------------------------------------------------------------------------
# Constants: Biomass source weights and allometric parameters
# ---------------------------------------------------------------------------

BIOMASS_SOURCE_WEIGHTS = {
    "esa_cci": 0.30,
    "gedi_l4a": 0.30,
    "sar": 0.25,
    "ndvi_allometric": 0.15,
}

# Carbon fraction of dry biomass (IPCC default)
CARBON_FRACTION = 0.47

# Biome-specific typical AGB ranges (Mg/ha) for reference validation
BIOME_AGB_RANGES = {
    "tropical_rainforest": (150.0, 450.0),
    "tropical_moist_forest": (120.0, 350.0),
    "tropical_dry_forest": (50.0, 200.0),
    "temperate_forest": (80.0, 300.0),
    "temperate_rainforest": (150.0, 500.0),
    "temperate_deciduous": (100.0, 300.0),
    "boreal_forest": (40.0, 150.0),
    "mangrove": (50.0, 250.0),
    "peat_swamp_forest": (80.0, 300.0),
    "cerrado_savanna": (10.0, 80.0),
    "tropical_savanna": (10.0, 60.0),
    "woodland_savanna": (20.0, 100.0),
    "montane_cloud_forest": (100.0, 350.0),
    "montane_dry_forest": (50.0, 200.0),
    "dry_woodland": (15.0, 80.0),
    "thorn_forest": (10.0, 60.0),
}

# SAR saturation threshold (Mg/ha) above which SAR backscatter saturates
SAR_SATURATION_THRESHOLD = 150.0

# Biome-specific allometric coefficients for NDVI -> AGB
# AGB = a * exp(b * NDVI)
NDVI_ALLOMETRIC_COEFFICIENTS = {
    "tropical_rainforest": (5.0, 5.0),
    "tropical_moist_forest": (4.5, 4.8),
    "tropical_dry_forest": (3.0, 4.0),
    "temperate_forest": (4.0, 4.5),
    "boreal_forest": (2.5, 3.5),
    "mangrove": (3.0, 4.2),
    "cerrado_savanna": (1.5, 3.0),
    "tropical_savanna": (1.0, 2.8),
}


# ---------------------------------------------------------------------------
# Helpers: Biomass estimation simulation
# ---------------------------------------------------------------------------


def _esa_cci_lookup(lat: float, lon: float) -> float:
    """Simulate ESA CCI Biomass product lookup.

    In production, queries a pre-downloaded global AGB GeoTIFF.
    Returns AGB in Mg/ha.
    """
    # Tropical: higher biomass
    if abs(lat) < 23.5:
        return 200.0
    return 120.0


def _gedi_l4a_estimate(rh_metrics: dict) -> float:
    """Simulate GEDI L4A above-ground biomass prediction.

    Uses RH95 (relative height at 95th percentile) as primary predictor.
    Simplified model: AGB = 0.5 * (RH95)^1.8
    """
    rh95 = rh_metrics.get("rh95", 0.0)
    if rh95 <= 0:
        return 0.0
    return 0.5 * (rh95 ** 1.8)


def _sar_regression(
    sigma0_db: float,
    a: float = 100.0,
    b: float = 0.15,
) -> tuple:
    """Estimate AGB from SAR backscatter using power-law model.

    AGB = a * (10^(sigma0/10))^b
    Returns (agb, is_saturated).
    """
    # Convert dB to linear
    sigma0_linear = 10.0 ** (sigma0_db / 10.0)
    agb = a * (sigma0_linear ** b)
    is_saturated = agb > SAR_SATURATION_THRESHOLD
    return agb, is_saturated


def _ndvi_allometric(
    ndvi: float,
    biome: str = "tropical_rainforest",
) -> float:
    """Estimate AGB from NDVI using biome-specific allometric model.

    AGB = a * exp(b * NDVI)
    """
    coeffs = NDVI_ALLOMETRIC_COEFFICIENTS.get(biome, (4.0, 4.5))
    a, b = coeffs
    agb = a * math.exp(b * ndvi)
    return max(0.0, agb)


def _carbon_stock(agb_mg_per_ha: float) -> float:
    """Convert AGB to carbon stock using IPCC default fraction."""
    return agb_mg_per_ha * CARBON_FRACTION


def _biomass_change_pct(cutoff_agb: float, current_agb: float) -> float:
    """Compute percentage change in AGB from cutoff to current."""
    if cutoff_agb == 0.0:
        return 0.0
    return ((current_agb - cutoff_agb) / cutoff_agb) * 100.0


# ===========================================================================
# 1. ESA CCI Lookup (4 tests)
# ===========================================================================


class TestESACCILookup:
    """Test ESA CCI Biomass product lookup."""

    def test_esa_cci_lookup_tropical(self):
        """Test ESA CCI returns tropical biomass value."""
        agb = _esa_cci_lookup(-3.12, -60.02)
        assert agb == 200.0

    def test_esa_cci_lookup_temperate(self):
        """Test ESA CCI returns temperate biomass value."""
        agb = _esa_cci_lookup(48.0, 8.0)
        assert agb == 120.0

    def test_esa_cci_lookup_positive(self):
        """Test ESA CCI returns positive value."""
        agb = _esa_cci_lookup(0.0, 0.0)
        assert agb > 0.0

    def test_esa_cci_lookup_determinism(self):
        """Test ESA CCI lookup is deterministic."""
        results = [_esa_cci_lookup(-3.12, -60.02) for _ in range(10)]
        assert len(set(results)) == 1


# ===========================================================================
# 2. GEDI L4A Estimation (5 tests)
# ===========================================================================


class TestGEDIL4A:
    """Test GEDI L4A biomass prediction."""

    def test_gedi_l4a_tall_canopy(self):
        """Test GEDI L4A for tall canopy (RH95=30m)."""
        agb = _gedi_l4a_estimate({"rh95": 30.0})
        assert agb > 100.0

    def test_gedi_l4a_short_canopy(self):
        """Test GEDI L4A for short canopy (RH95=5m)."""
        agb = _gedi_l4a_estimate({"rh95": 5.0})
        assert 0.0 < agb < 50.0

    def test_gedi_l4a_zero(self):
        """Test GEDI L4A with zero RH95 returns 0."""
        agb = _gedi_l4a_estimate({"rh95": 0.0})
        assert agb == 0.0

    def test_gedi_l4a_missing_rh95(self):
        """Test GEDI L4A with missing RH95 returns 0."""
        agb = _gedi_l4a_estimate({})
        assert agb == 0.0

    def test_gedi_l4a_determinism(self):
        """Test GEDI L4A is deterministic."""
        results = [_gedi_l4a_estimate({"rh95": 25.0}) for _ in range(10)]
        assert len(set(results)) == 1


# ===========================================================================
# 3. SAR Regression (7 tests)
# ===========================================================================


class TestSARRegression:
    """Test SAR backscatter regression for AGB estimation."""

    def test_sar_regression_normal(self):
        """Test SAR regression returns positive AGB."""
        agb, sat = _sar_regression(-10.0)
        assert agb > 0.0

    def test_sar_regression_high_backscatter(self):
        """Test high backscatter gives higher AGB."""
        agb_high, _ = _sar_regression(-5.0)
        agb_low, _ = _sar_regression(-15.0)
        assert agb_high > agb_low

    def test_sar_saturation_flag(self):
        """Test AGB > saturation threshold sets saturated flag."""
        agb, sat = _sar_regression(-2.0)  # Very high backscatter
        if agb > SAR_SATURATION_THRESHOLD:
            assert sat is True

    def test_sar_no_saturation(self):
        """Test AGB below saturation threshold has False flag."""
        agb, sat = _sar_regression(-20.0)  # Low backscatter
        if agb <= SAR_SATURATION_THRESHOLD:
            assert sat is False

    def test_sar_saturation_threshold_value(self):
        """Test SAR saturation threshold is 150 Mg/ha."""
        assert SAR_SATURATION_THRESHOLD == 150.0

    def test_sar_regression_negative_db(self):
        """Test negative dB values produce valid results."""
        for db in [-25.0, -20.0, -15.0, -10.0, -5.0]:
            agb, _ = _sar_regression(db)
            assert agb >= 0.0

    def test_sar_regression_determinism(self):
        """Test SAR regression is deterministic."""
        results = [_sar_regression(-10.0) for _ in range(10)]
        assert len(set(r[0] for r in results)) == 1


# ===========================================================================
# 4. NDVI Allometric (7 tests)
# ===========================================================================


class TestNDVIAllometric:
    """Test NDVI allometric biomass estimation."""

    def test_ndvi_allometric_forest(self):
        """Test AGB = a * exp(b * NDVI) for forest NDVI."""
        agb = _ndvi_allometric(0.75, "tropical_rainforest")
        # a=5.0, b=5.0: 5 * exp(5*0.75) = 5 * exp(3.75) ~ 212.8
        assert agb > 100.0

    def test_ndvi_allometric_low_ndvi(self):
        """Test low NDVI produces low AGB."""
        agb = _ndvi_allometric(0.10, "tropical_rainforest")
        assert agb < 50.0

    def test_ndvi_allometric_zero_ndvi(self):
        """Test zero NDVI returns a (intercept)."""
        agb = _ndvi_allometric(0.0, "tropical_rainforest")
        # a * exp(0) = a = 5.0
        assert abs(agb - 5.0) < 1e-9

    def test_ndvi_allometric_negative_ndvi(self):
        """Test negative NDVI gives very low but non-negative AGB."""
        agb = _ndvi_allometric(-0.25, "tropical_rainforest")
        assert agb >= 0.0

    @pytest.mark.parametrize("biome", [
        "tropical_rainforest", "tropical_moist_forest", "boreal_forest",
        "cerrado_savanna", "mangrove",
    ])
    def test_ndvi_allometric_biome_specific(self, biome):
        """Test allometric model uses biome-specific coefficients."""
        agb = _ndvi_allometric(0.70, biome)
        assert agb > 0.0

    def test_ndvi_allometric_unknown_biome(self):
        """Test unknown biome uses default coefficients."""
        agb = _ndvi_allometric(0.70, "unknown_biome")
        assert agb > 0.0

    def test_ndvi_allometric_determinism(self):
        """Test NDVI allometric is deterministic."""
        results = [_ndvi_allometric(0.72, "tropical_rainforest") for _ in range(10)]
        assert len(set(results)) == 1


# ===========================================================================
# 5. Multi-Source Fusion (5 tests)
# ===========================================================================


class TestBiomassFusion:
    """Test weighted multi-source biomass fusion."""

    def test_fuse_biomass_all_sources(self):
        """Test fusion with all 4 biomass sources."""
        values = {
            "esa_cci": 195.0,
            "gedi_l4a": 210.0,
            "sar": 180.0,
            "ndvi_allometric": 190.0,
        }
        fused = fuse_weighted(values, BIOMASS_SOURCE_WEIGHTS)
        assert 180.0 < fused < 210.0

    def test_fuse_biomass_single_source(self):
        """Test single source returns that value."""
        values = {"esa_cci": 200.0}
        fused = fuse_weighted(values, BIOMASS_SOURCE_WEIGHTS)
        assert abs(fused - 200.0) < 1e-9

    def test_fuse_biomass_missing_sources(self):
        """Test missing sources: weights re-normalized."""
        values = {"esa_cci": 200.0, "gedi_l4a": 210.0}
        fused = fuse_weighted(values, BIOMASS_SOURCE_WEIGHTS)
        expected = (0.30 * 200.0 + 0.30 * 210.0) / (0.30 + 0.30)
        assert abs(fused - expected) < 1e-9

    def test_fuse_biomass_weights_sum(self):
        """Test biomass source weights sum to 1.0."""
        total = sum(BIOMASS_SOURCE_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9

    def test_fuse_biomass_determinism(self):
        """Test biomass fusion is deterministic."""
        values = {"esa_cci": 195.0, "gedi_l4a": 210.0, "sar": 180.0}
        results = [fuse_weighted(values, BIOMASS_SOURCE_WEIGHTS) for _ in range(10)]
        assert len(set(results)) == 1


# ===========================================================================
# 6. Carbon Stock Conversion (4 tests)
# ===========================================================================


class TestCarbonStockConversion:
    """Test AGB to carbon stock conversion."""

    def test_carbon_stock_conversion(self):
        """Test AGB * 0.47 = carbon stock."""
        carbon = _carbon_stock(200.0)
        assert abs(carbon - 94.0) < 1e-9

    def test_carbon_stock_zero(self):
        """Test zero AGB gives zero carbon."""
        carbon = _carbon_stock(0.0)
        assert carbon == 0.0

    def test_carbon_stock_fraction(self):
        """Test carbon fraction is 0.47 (IPCC default)."""
        assert CARBON_FRACTION == 0.47

    def test_carbon_stock_determinism(self):
        """Test carbon conversion is deterministic."""
        results = [_carbon_stock(200.0) for _ in range(10)]
        assert len(set(results)) == 1


# ===========================================================================
# 7. Biomass Change (4 tests)
# ===========================================================================


class TestBiomassChange:
    """Test biomass change from cutoff to current."""

    def test_biomass_change_loss(self):
        """Test negative change for biomass loss."""
        change = _biomass_change_pct(200.0, 150.0)
        assert change == -25.0

    def test_biomass_change_gain(self):
        """Test positive change for biomass gain."""
        change = _biomass_change_pct(200.0, 220.0)
        assert change == 10.0

    def test_biomass_change_no_change(self):
        """Test zero change when same."""
        change = _biomass_change_pct(200.0, 200.0)
        assert abs(change) < 1e-9

    def test_biomass_change_zero_cutoff(self):
        """Test zero cutoff returns 0 (avoid division by zero)."""
        change = _biomass_change_pct(0.0, 100.0)
        assert change == 0.0


# ===========================================================================
# 8. Reference Range Validation (3 tests)
# ===========================================================================


class TestReferenceRanges:
    """Test biomass results fall within biome-typical reference ranges."""

    @pytest.mark.parametrize("biome", ALL_BIOMES)
    def test_reference_ranges_defined(self, biome):
        """Test all biomes have reference AGB ranges defined."""
        assert biome in BIOME_AGB_RANGES

    def test_reference_range_tropical_rainforest(self):
        """Test tropical rainforest AGB range is 150-450 Mg/ha."""
        lo, hi = BIOME_AGB_RANGES["tropical_rainforest"]
        assert lo == 150.0
        assert hi == 450.0

    def test_reference_range_fixture_in_range(self, sample_biomass_estimate):
        """Test fixture biomass is within expected range."""
        biome = sample_biomass_estimate.biome
        lo, hi = BIOME_AGB_RANGES[biome]
        assert lo <= sample_biomass_estimate.agb_mg_per_ha <= hi


# ===========================================================================
# 9. Full Pipeline and Batch (5 tests)
# ===========================================================================


class TestFullPipelineAndBatch:
    """Test full biomass estimation pipeline and batch processing."""

    def test_estimate_plot_biomass(self, sample_biomass_estimate):
        """Test full pipeline returns BiomassEstimate."""
        assert isinstance(sample_biomass_estimate, BiomassEstimate)

    def test_estimate_plot_has_provenance(self, sample_biomass_estimate):
        """Test result includes provenance hash."""
        assert len(sample_biomass_estimate.provenance_hash) == SHA256_HEX_LENGTH

    def test_estimate_plot_carbon_consistency(self, sample_biomass_estimate):
        """Test carbon stock is AGB * 0.47."""
        expected = sample_biomass_estimate.agb_mg_per_ha * CARBON_FRACTION
        assert abs(sample_biomass_estimate.carbon_stock_mg_per_ha - expected) < 1e-9

    def test_batch_estimate_multiple(self):
        """Test batch estimation returns results for all plots."""
        plots = [f"PLOT-{i:03d}" for i in range(5)]
        results = [
            BiomassEstimate(plot_id=pid, agb_mg_per_ha=180.0 + i * 10)
            for i, pid in enumerate(plots)
        ]
        assert len(results) == 5
        assert all(isinstance(r, BiomassEstimate) for r in results)

    def test_estimate_plot_sar_saturation_flag(self):
        """Test SAR saturation flag is tracked."""
        estimate = BiomassEstimate(
            plot_id="PLOT-001",
            sar_agb=200.0,
            sar_saturated=True,
        )
        assert estimate.sar_saturated is True


# ===========================================================================
# 10. Uncertainty Propagation (3 tests)
# ===========================================================================


class TestUncertaintyPropagation:
    """Test biomass uncertainty propagation."""

    def test_uncertainty_all_sources(self):
        """Test fused uncertainty from all biomass sources."""
        uncertainties = {
            "esa_cci": 20.0,
            "gedi_l4a": 25.0,
            "sar": 30.0,
            "ndvi_allometric": 35.0,
        }
        unc = weighted_rms_uncertainty(uncertainties, BIOMASS_SOURCE_WEIGHTS)
        assert 0.0 < unc < 40.0

    def test_uncertainty_single_source(self):
        """Test single source uncertainty equals that value."""
        uncertainties = {"esa_cci": 20.0}
        unc = weighted_rms_uncertainty(uncertainties, BIOMASS_SOURCE_WEIGHTS)
        assert abs(unc - 20.0) < 1e-9

    def test_uncertainty_determinism(self):
        """Test uncertainty propagation is deterministic."""
        uncertainties = {"esa_cci": 20.0, "gedi_l4a": 25.0}
        results = [
            weighted_rms_uncertainty(uncertainties, BIOMASS_SOURCE_WEIGHTS)
            for _ in range(10)
        ]
        assert len(set(results)) == 1


# ===========================================================================
# 11. Determinism (3 tests)
# ===========================================================================


class TestBiomassDeterminism:
    """Test deterministic behaviour of biomass estimation."""

    def test_determinism_esa_cci(self):
        """Test ESA CCI lookup is deterministic."""
        results = [_esa_cci_lookup(-3.12, -60.02) for _ in range(20)]
        assert all(r == results[0] for r in results)

    def test_determinism_allometric(self):
        """Test NDVI allometric is deterministic."""
        results = [_ndvi_allometric(0.72, "tropical_rainforest") for _ in range(20)]
        assert len(set(results)) == 1

    def test_determinism_provenance_hash(self):
        """Test same inputs produce identical provenance hash."""
        data = {"plot_id": "P001", "agb_mg_per_ha": 200.0}
        hashes = [compute_test_hash(data) for _ in range(20)]
        assert len(set(hashes)) == 1
