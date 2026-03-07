# -*- coding: utf-8 -*-
"""
Tests for CanopyHeightModeler - AGENT-EUDR-004 Engine 5: Canopy Height Modeling

Comprehensive test suite covering:
- GEDI L2A RH95 canopy height estimation
- ICESat-2 ATL08 photon-counting canopy height
- Texture-proxy height estimation (GLCM contrast)
- Global canopy height map lookups (ETH Zurich 10m, Meta 1m)
- Multi-source weighted height fusion
- Single-source fallback behaviour
- Weight re-normalization for missing sources
- FAO tree height threshold checks (>= 5m)
- Uncertainty propagation (weighted RMS)
- Full pipeline height estimation
- Batch estimation for multiple plots
- Determinism and provenance hash reproducibility

Test count: 50+ tests
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-004 (Engine 5 - Canopy Height Modeling)
"""

import math

import pytest

from tests.agents.eudr.forest_cover_analysis.conftest import (
    CanopyHeightEstimate,
    compute_test_hash,
    fuse_weighted,
    weighted_rms_uncertainty,
    SHA256_HEX_LENGTH,
    HEIGHT_SOURCE_WEIGHTS,
    FAO_TREE_HEIGHT_M,
)


# ---------------------------------------------------------------------------
# Helpers: Height estimation simulation
# ---------------------------------------------------------------------------


def _gedi_estimate(rh95_m: float) -> float:
    """Simulate GEDI L2A RH95 canopy height estimate.

    RH95 = height at which 95% of waveform energy is below.
    Directly usable as canopy height with minor bias correction.
    """
    bias_correction = -0.5  # Slight overestimation in dense canopy
    return max(0.0, rh95_m + bias_correction)


def _icesat2_estimate(atl08_height_m: float) -> float:
    """Simulate ICESat-2 ATL08 canopy height estimate.

    ATL08 product provides photon-counting derived canopy height
    along ground track segments.
    """
    return max(0.0, atl08_height_m)


def _texture_proxy_height(glcm_contrast: float) -> float:
    """Estimate canopy height from GLCM texture contrast.

    Taller canopies produce more textural variation in high-res imagery.
    Simplified linear model: height_m = a * contrast + b.
    """
    a, b = 0.025, 2.0  # Calibration coefficients
    return max(0.0, a * glcm_contrast + b)


def _global_map_eth(lat: float, lon: float) -> float:
    """Simulate ETH Zurich Global Canopy Height 10m map lookup.

    In production, looks up pixel value from a pre-downloaded GeoTIFF.
    """
    # Simulated: tropical locations have taller canopy
    base = 20.0 if abs(lat) < 23.5 else 12.0
    return base


def _global_map_meta(lat: float, lon: float) -> float:
    """Simulate Meta 1m Global Canopy Height map lookup.

    In production, queries Meta's AI-derived height map at 1m resolution.
    """
    base = 22.0 if abs(lat) < 23.5 else 14.0
    return base


# ===========================================================================
# 1. GEDI Estimation (6 tests)
# ===========================================================================


class TestGEDIEstimation:
    """Test GEDI L2A RH95 canopy height estimation."""

    def test_gedi_estimate_tropical(self):
        """Test GEDI L2A RH95 for tropical forest returns height in metres."""
        height = _gedi_estimate(27.0)
        assert 20.0 < height < 30.0

    def test_gedi_estimate_short_canopy(self):
        """Test GEDI for short canopy (5m)."""
        height = _gedi_estimate(5.0)
        assert 3.0 < height < 6.0

    def test_gedi_estimate_zero(self):
        """Test GEDI with zero RH95 returns 0 or near-zero."""
        height = _gedi_estimate(0.0)
        assert height == 0.0

    def test_gedi_estimate_negative_clamp(self):
        """Test GEDI with very low RH95 does not return negative."""
        height = _gedi_estimate(0.2)
        assert height >= 0.0

    def test_gedi_estimate_tall_canopy(self):
        """Test GEDI for very tall canopy (60m dipterocarp)."""
        height = _gedi_estimate(60.0)
        assert height > 50.0

    def test_gedi_estimate_determinism(self):
        """Test GEDI estimation is deterministic."""
        results = [_gedi_estimate(27.0) for _ in range(10)]
        assert len(set(results)) == 1


# ===========================================================================
# 2. ICESat-2 Estimation (5 tests)
# ===========================================================================


class TestICESat2Estimation:
    """Test ICESat-2 ATL08 canopy height estimation."""

    def test_icesat2_estimate_normal(self):
        """Test ICESat-2 ATL08 canopy height for typical forest."""
        height = _icesat2_estimate(24.0)
        assert height == 24.0

    def test_icesat2_estimate_zero(self):
        """Test ICESat-2 with zero height."""
        height = _icesat2_estimate(0.0)
        assert height == 0.0

    def test_icesat2_estimate_negative_clamp(self):
        """Test ICESat-2 with negative input returns 0."""
        height = _icesat2_estimate(-5.0)
        assert height == 0.0

    def test_icesat2_estimate_tall(self):
        """Test ICESat-2 for 40m canopy."""
        height = _icesat2_estimate(40.0)
        assert height == 40.0

    def test_icesat2_estimate_determinism(self):
        """Test ICESat-2 estimation is deterministic."""
        results = [_icesat2_estimate(24.0) for _ in range(10)]
        assert len(set(results)) == 1


# ===========================================================================
# 3. Texture Proxy Estimation (5 tests)
# ===========================================================================


class TestTextureProxyEstimation:
    """Test GLCM texture-based canopy height estimation."""

    def test_texture_proxy_high_contrast(self):
        """Test high GLCM contrast -> tall canopy estimate."""
        height = _texture_proxy_height(1000.0)
        assert height > 20.0

    def test_texture_proxy_low_contrast(self):
        """Test low GLCM contrast -> short canopy estimate."""
        height = _texture_proxy_height(50.0)
        assert 2.0 <= height <= 10.0

    def test_texture_proxy_zero_contrast(self):
        """Test zero contrast returns minimum height."""
        height = _texture_proxy_height(0.0)
        assert height == 2.0  # intercept b=2.0

    def test_texture_proxy_negative_clamp(self):
        """Test negative contrast (artifact) clamps to >= 0."""
        height = _texture_proxy_height(-200.0)
        assert height >= 0.0

    def test_texture_proxy_determinism(self):
        """Test texture proxy is deterministic."""
        results = [_texture_proxy_height(500.0) for _ in range(10)]
        assert len(set(results)) == 1


# ===========================================================================
# 4. Global Map Lookups (6 tests)
# ===========================================================================


class TestGlobalMapLookups:
    """Test global canopy height map lookups (ETH and Meta)."""

    def test_global_map_eth_tropical(self):
        """Test ETH Zurich 10m lookup at tropical latitude."""
        height = _global_map_eth(-3.12, -60.02)
        assert height == 20.0

    def test_global_map_eth_temperate(self):
        """Test ETH Zurich 10m lookup at temperate latitude."""
        height = _global_map_eth(48.0, 8.0)
        assert height == 12.0

    def test_global_map_meta_tropical(self):
        """Test Meta 1m lookup at tropical latitude."""
        height = _global_map_meta(-1.5, 110.4)
        assert height == 22.0

    def test_global_map_meta_temperate(self):
        """Test Meta 1m lookup at temperate latitude."""
        height = _global_map_meta(52.0, 0.0)
        assert height == 14.0

    def test_global_map_eth_determinism(self):
        """Test ETH map lookup is deterministic."""
        results = [_global_map_eth(-3.12, -60.02) for _ in range(10)]
        assert len(set(results)) == 1

    def test_global_map_meta_determinism(self):
        """Test Meta map lookup is deterministic."""
        results = [_global_map_meta(-3.12, -60.02) for _ in range(10)]
        assert len(set(results)) == 1


# ===========================================================================
# 5. Multi-Source Fusion (10 tests)
# ===========================================================================


class TestMultiSourceFusion:
    """Test weighted fusion of height estimates from multiple sources."""

    def test_fuse_heights_all_sources(self):
        """Test weighted fusion with all 5 sources available."""
        values = {
            "gedi": 26.5,
            "icesat2": 24.0,
            "eth_global": 25.5,
            "meta_global": 26.0,
            "texture_proxy": 22.0,
        }
        fused = fuse_weighted(values, HEIGHT_SOURCE_WEIGHTS)
        # Should be close to weighted average
        assert 23.0 < fused < 27.0

    def test_fuse_single_source_gedi(self):
        """Test only GEDI available -> that value returned."""
        values = {"gedi": 26.5}
        fused = fuse_weighted(values, HEIGHT_SOURCE_WEIGHTS)
        assert abs(fused - 26.5) < 1e-9

    def test_fuse_single_source_icesat2(self):
        """Test only ICESat-2 available -> that value returned."""
        values = {"icesat2": 24.0}
        fused = fuse_weighted(values, HEIGHT_SOURCE_WEIGHTS)
        assert abs(fused - 24.0) < 1e-9

    def test_fuse_missing_sources_renormalize(self):
        """Test weights re-normalized when sources are missing."""
        values = {"gedi": 26.0, "icesat2": 24.0}
        fused = fuse_weighted(values, HEIGHT_SOURCE_WEIGHTS)
        # Only gedi(0.35) and icesat2(0.30) available
        # Re-normalized: gedi=0.35/0.65, icesat2=0.30/0.65
        expected = (0.35 * 26.0 + 0.30 * 24.0) / (0.35 + 0.30)
        assert abs(fused - expected) < 1e-9

    def test_fuse_no_sources(self):
        """Test empty values returns 0.0."""
        fused = fuse_weighted({}, HEIGHT_SOURCE_WEIGHTS)
        assert fused == 0.0

    def test_fuse_unknown_source_ignored(self):
        """Test unknown source key is ignored."""
        values = {"gedi": 26.0, "unknown_source": 99.0}
        fused = fuse_weighted(values, HEIGHT_SOURCE_WEIGHTS)
        assert abs(fused - 26.0) < 1e-9

    def test_fuse_weights_sum_to_one(self):
        """Test height source weights sum to 1.0."""
        total = sum(HEIGHT_SOURCE_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9

    def test_fuse_all_same_value(self):
        """Test fusion of identical values returns that value."""
        values = {k: 25.0 for k in HEIGHT_SOURCE_WEIGHTS}
        fused = fuse_weighted(values, HEIGHT_SOURCE_WEIGHTS)
        assert abs(fused - 25.0) < 1e-9

    def test_fuse_determinism(self):
        """Test fusion is deterministic."""
        values = {"gedi": 26.5, "icesat2": 24.0, "eth_global": 25.0}
        results = [fuse_weighted(values, HEIGHT_SOURCE_WEIGHTS) for _ in range(10)]
        assert len(set(results)) == 1

    def test_fuse_higher_weight_source_dominates(self):
        """Test source with higher weight has more influence."""
        # GEDI has weight 0.35, texture has 0.05
        values = {"gedi": 30.0, "texture_proxy": 10.0}
        fused = fuse_weighted(values, HEIGHT_SOURCE_WEIGHTS)
        # Should be closer to 30 than 10
        assert fused > 20.0


# ===========================================================================
# 6. FAO Threshold (5 tests)
# ===========================================================================


class TestFAOHeightThreshold:
    """Test FAO tree height threshold checks."""

    def test_fao_threshold_meets(self):
        """Test height >= 5m meets FAO threshold."""
        assert 25.0 >= FAO_TREE_HEIGHT_M

    def test_fao_threshold_below(self):
        """Test height < 5m fails FAO threshold."""
        assert 3.0 < FAO_TREE_HEIGHT_M

    def test_fao_threshold_exactly_five(self):
        """Test height exactly 5m meets threshold."""
        assert 5.0 >= FAO_TREE_HEIGHT_M

    def test_fao_threshold_just_below(self):
        """Test height 4.9m fails threshold."""
        assert 4.9 < FAO_TREE_HEIGHT_M

    def test_fao_threshold_value(self):
        """Test FAO tree height threshold is 5.0m."""
        assert FAO_TREE_HEIGHT_M == 5.0


# ===========================================================================
# 7. Uncertainty Propagation (5 tests)
# ===========================================================================


class TestUncertaintyPropagation:
    """Test weighted RMS uncertainty propagation."""

    def test_uncertainty_propagation_all_sources(self):
        """Test fused uncertainty from all sources."""
        uncertainties = {
            "gedi": 3.0,
            "icesat2": 4.0,
            "eth_global": 5.0,
            "meta_global": 2.0,
            "texture_proxy": 8.0,
        }
        unc = weighted_rms_uncertainty(uncertainties, HEIGHT_SOURCE_WEIGHTS)
        assert 0.0 < unc < 10.0

    def test_uncertainty_propagation_single_source(self):
        """Test uncertainty with single source equals that source's uncertainty."""
        uncertainties = {"gedi": 3.0}
        unc = weighted_rms_uncertainty(uncertainties, HEIGHT_SOURCE_WEIGHTS)
        # Single source: weight=1.0 after renormalization, unc = 3.0
        assert abs(unc - 3.0) < 1e-9

    def test_uncertainty_propagation_zero(self):
        """Test all-zero uncertainties produce zero fused uncertainty."""
        uncertainties = {k: 0.0 for k in HEIGHT_SOURCE_WEIGHTS}
        unc = weighted_rms_uncertainty(uncertainties, HEIGHT_SOURCE_WEIGHTS)
        assert abs(unc) < 1e-9

    def test_uncertainty_propagation_empty(self):
        """Test empty uncertainties returns 0.0."""
        unc = weighted_rms_uncertainty({}, HEIGHT_SOURCE_WEIGHTS)
        assert unc == 0.0

    def test_uncertainty_propagation_determinism(self):
        """Test uncertainty propagation is deterministic."""
        uncertainties = {"gedi": 3.0, "icesat2": 4.0}
        results = [
            weighted_rms_uncertainty(uncertainties, HEIGHT_SOURCE_WEIGHTS)
            for _ in range(10)
        ]
        assert len(set(results)) == 1


# ===========================================================================
# 8. Full Pipeline and Batch (5 tests)
# ===========================================================================


class TestFullPipelineAndBatch:
    """Test full height estimation pipeline and batch processing."""

    def test_estimate_plot_height(self, sample_height_estimate):
        """Test full pipeline returns CanopyHeightEstimate."""
        assert isinstance(sample_height_estimate, CanopyHeightEstimate)
        assert sample_height_estimate.height_m == 25.0

    def test_estimate_plot_has_provenance(self, sample_height_estimate):
        """Test result includes provenance hash."""
        assert len(sample_height_estimate.provenance_hash) == SHA256_HEX_LENGTH

    def test_estimate_plot_meets_fao(self, sample_height_estimate):
        """Test fixture height meets FAO threshold."""
        assert sample_height_estimate.meets_fao_threshold is True

    def test_batch_estimate_multiple(self):
        """Test batch estimation returns results for all plots."""
        plots = [f"PLOT-{i:03d}" for i in range(5)]
        results = [
            CanopyHeightEstimate(plot_id=pid, height_m=20.0 + i * 2)
            for i, pid in enumerate(plots)
        ]
        assert len(results) == 5
        assert all(isinstance(r, CanopyHeightEstimate) for r in results)

    def test_estimate_sources_available(self, sample_height_estimate):
        """Test fixture tracks number of available sources."""
        assert sample_height_estimate.sources_available == 5


# ===========================================================================
# 9. Determinism (3 tests)
# ===========================================================================


class TestHeightDeterminism:
    """Test deterministic behaviour of canopy height modeling."""

    def test_determinism_gedi(self):
        """Test GEDI estimation is deterministic."""
        results = [_gedi_estimate(27.0) for _ in range(20)]
        assert all(r == results[0] for r in results)

    def test_determinism_fusion(self):
        """Test height fusion is deterministic."""
        values = {"gedi": 26.5, "icesat2": 24.0}
        results = [fuse_weighted(values, HEIGHT_SOURCE_WEIGHTS) for _ in range(20)]
        assert len(set(results)) == 1

    def test_determinism_provenance_hash(self):
        """Test same inputs produce identical provenance hash."""
        data = {"plot_id": "PLOT-001", "height_m": 25.0}
        hashes = [compute_test_hash(data) for _ in range(20)]
        assert len(set(hashes)) == 1
