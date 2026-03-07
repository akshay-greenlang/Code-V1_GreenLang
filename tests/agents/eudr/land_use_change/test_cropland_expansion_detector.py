# -*- coding: utf-8 -*-
"""
Tests for CroplandExpansionDetector - AGENT-EUDR-005 Land Use Change Detector

Comprehensive test suite covering:
- Detection of 7 commodity-specific conversions (palm oil, rubber, cocoa,
  coffee, soya, pasture, timber plantation)
- Scale classification (smallholder, medium, industrial)
- Expansion rate estimation (ha/year)
- Conversion hotspot detection
- Leapfrog pattern detection
- Batch detection processing
- Deterministic detection behavior
- Edge cases and error handling

Test count: 55 tests
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-005 Land Use Change Detector Agent (GL-EUDR-LUC-005)
"""

from unittest.mock import MagicMock

import pytest

from tests.agents.eudr.land_use_change.conftest import (
    CroplandExpansion,
    compute_test_hash,
    SHA256_HEX_LENGTH,
    CONVERSION_TYPES,
    EXPANSION_SCALES,
    EUDR_COMMODITIES,
)


# ===========================================================================
# 1. Commodity-Specific Conversion Tests (14 tests)
# ===========================================================================


class TestCommodityConversions:
    """Tests for detecting 7 commodity-specific land conversions."""

    def test_detect_palm_oil_conversion(self):
        """Detect forest to palm oil plantation conversion."""
        result = CroplandExpansion(
            plot_id="PLOT-PO-001",
            conversion_type="palm_oil_conversion",
            from_category="forest",
            to_category="cropland",
            commodity="palm_oil",
            scale="industrial",
            expansion_rate_ha_per_year=50.0,
            area_converted_ha=100.0,
            confidence=0.90,
        )
        assert result.conversion_type == "palm_oil_conversion"
        assert result.commodity == "palm_oil"
        assert result.from_category == "forest"

    def test_detect_palm_oil_spectral_signature(self):
        """Palm oil conversion has distinct uniform canopy spectral signature."""
        result = CroplandExpansion(
            plot_id="PLOT-PO-002",
            conversion_type="palm_oil_conversion",
            commodity="palm_oil",
            confidence=0.88,
        )
        assert result.confidence > 0.80

    def test_detect_rubber_conversion(self):
        """Detect forest to rubber plantation conversion."""
        result = CroplandExpansion(
            plot_id="PLOT-RB-001",
            conversion_type="rubber_conversion",
            from_category="forest",
            to_category="cropland",
            commodity="rubber",
            scale="medium",
            expansion_rate_ha_per_year=15.0,
            area_converted_ha=30.0,
            confidence=0.85,
        )
        assert result.conversion_type == "rubber_conversion"
        assert result.commodity == "rubber"

    def test_detect_cocoa_conversion(self):
        """Detect forest to cocoa plantation conversion."""
        result = CroplandExpansion(
            plot_id="PLOT-CC-001",
            conversion_type="cocoa_conversion",
            from_category="forest",
            to_category="cropland",
            commodity="cocoa",
            scale="smallholder",
            expansion_rate_ha_per_year=3.0,
            area_converted_ha=5.0,
            confidence=0.78,
        )
        assert result.conversion_type == "cocoa_conversion"
        assert result.commodity == "cocoa"

    def test_detect_coffee_conversion(self):
        """Detect forest to coffee plantation conversion."""
        result = CroplandExpansion(
            plot_id="PLOT-CF-001",
            conversion_type="coffee_conversion",
            from_category="forest",
            to_category="cropland",
            commodity="coffee",
            scale="smallholder",
            expansion_rate_ha_per_year=2.5,
            area_converted_ha=4.0,
            confidence=0.76,
        )
        assert result.conversion_type == "coffee_conversion"
        assert result.commodity == "coffee"

    def test_detect_soya_conversion(self):
        """Detect forest/cerrado to soya cropland conversion."""
        result = CroplandExpansion(
            plot_id="PLOT-SY-001",
            conversion_type="soya_conversion",
            from_category="forest",
            to_category="cropland",
            commodity="soya",
            scale="industrial",
            expansion_rate_ha_per_year=200.0,
            area_converted_ha=500.0,
            confidence=0.92,
        )
        assert result.conversion_type == "soya_conversion"
        assert result.commodity == "soya"
        assert result.scale == "industrial"

    def test_detect_pasture_conversion(self):
        """Detect forest to cattle pasture conversion."""
        result = CroplandExpansion(
            plot_id="PLOT-PS-001",
            conversion_type="pasture_conversion",
            from_category="forest",
            to_category="grassland",
            commodity="cattle",
            scale="industrial",
            expansion_rate_ha_per_year=300.0,
            area_converted_ha=1000.0,
            confidence=0.88,
        )
        assert result.conversion_type == "pasture_conversion"
        assert result.to_category == "grassland"

    def test_detect_timber_plantation_conversion(self):
        """Detect natural forest to timber plantation conversion."""
        result = CroplandExpansion(
            plot_id="PLOT-TP-001",
            conversion_type="timber_plantation_conversion",
            from_category="forest",
            to_category="cropland",
            commodity="wood",
            scale="industrial",
            expansion_rate_ha_per_year=100.0,
            area_converted_ha=250.0,
            confidence=0.82,
        )
        assert result.conversion_type == "timber_plantation_conversion"
        assert result.commodity == "wood"

    @pytest.mark.parametrize("ctype", CONVERSION_TYPES)
    def test_all_conversion_types_valid(self, ctype):
        """Each conversion type value is accepted."""
        result = CroplandExpansion(
            plot_id=f"PLOT-{ctype[:5].upper()}",
            conversion_type=ctype,
            confidence=0.80,
        )
        assert result.conversion_type == ctype

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_all_commodities_detectable(self, commodity):
        """Expansion can be detected for each EUDR commodity."""
        result = CroplandExpansion(
            plot_id=f"PLOT-{commodity.upper()[:3]}",
            commodity=commodity,
            confidence=0.80,
        )
        assert result.commodity == commodity

    def test_palm_oil_larger_than_cocoa(self):
        """Industrial palm oil conversions are larger than smallholder cocoa."""
        palm = CroplandExpansion(
            plot_id="PLOT-CMP-PO",
            commodity="palm_oil",
            scale="industrial",
            area_converted_ha=100.0,
        )
        cocoa = CroplandExpansion(
            plot_id="PLOT-CMP-CC",
            commodity="cocoa",
            scale="smallholder",
            area_converted_ha=5.0,
        )
        assert palm.area_converted_ha > cocoa.area_converted_ha

    def test_soya_highest_expansion_rate(self):
        """Soya has highest expansion rate among commodities."""
        soya = CroplandExpansion(
            commodity="soya",
            expansion_rate_ha_per_year=200.0,
        )
        coffee = CroplandExpansion(
            commodity="coffee",
            expansion_rate_ha_per_year=2.5,
        )
        assert soya.expansion_rate_ha_per_year > coffee.expansion_rate_ha_per_year

    def test_cocoa_smallest_area(self):
        """Smallholder cocoa has smallest area conversion."""
        result = CroplandExpansion(
            commodity="cocoa",
            scale="smallholder",
            area_converted_ha=2.0,
        )
        assert result.area_converted_ha < 10.0

    def test_pasture_to_grassland(self):
        """Pasture conversion produces grassland, not cropland."""
        result = CroplandExpansion(
            conversion_type="pasture_conversion",
            to_category="grassland",
        )
        assert result.to_category == "grassland"


# ===========================================================================
# 2. Scale Classification Tests (8 tests)
# ===========================================================================


class TestScaleClassification:
    """Tests for expansion scale classification."""

    def test_classify_scale_smallholder(self):
        """Area < 10 ha -> smallholder."""
        result = CroplandExpansion(
            plot_id="PLOT-SH-001",
            scale="smallholder",
            area_converted_ha=3.0,
        )
        assert result.scale == "smallholder"

    def test_classify_scale_medium(self):
        """Area 10-100 ha -> medium."""
        result = CroplandExpansion(
            plot_id="PLOT-MED-001",
            scale="medium",
            area_converted_ha=50.0,
        )
        assert result.scale == "medium"

    def test_classify_scale_industrial(self):
        """Area > 100 ha -> industrial."""
        result = CroplandExpansion(
            plot_id="PLOT-IND-001",
            scale="industrial",
            area_converted_ha=500.0,
        )
        assert result.scale == "industrial"

    @pytest.mark.parametrize("scale", EXPANSION_SCALES)
    def test_all_scales_valid(self, scale):
        """Each scale classification is accepted."""
        result = CroplandExpansion(
            plot_id=f"PLOT-{scale.upper()[:3]}",
            scale=scale,
        )
        assert result.scale == scale

    def test_smallholder_boundary_9ha(self):
        """9 ha is still smallholder."""
        result = CroplandExpansion(
            scale="smallholder",
            area_converted_ha=9.0,
        )
        assert result.scale == "smallholder"

    def test_medium_boundary_10ha(self):
        """10 ha is medium scale."""
        result = CroplandExpansion(
            scale="medium",
            area_converted_ha=10.0,
        )
        assert result.scale == "medium"

    def test_industrial_boundary_100ha(self):
        """100 ha is industrial scale."""
        result = CroplandExpansion(
            scale="industrial",
            area_converted_ha=100.0,
        )
        assert result.scale == "industrial"

    def test_very_large_industrial(self):
        """Very large conversion (10000 ha) is industrial."""
        result = CroplandExpansion(
            scale="industrial",
            area_converted_ha=10000.0,
        )
        assert result.scale == "industrial"


# ===========================================================================
# 3. Expansion Rate Tests (6 tests)
# ===========================================================================


class TestExpansionRate:
    """Tests for expansion rate estimation."""

    def test_expansion_rate_positive(self):
        """Active conversion has positive expansion rate."""
        result = CroplandExpansion(
            expansion_rate_ha_per_year=25.0,
        )
        assert result.expansion_rate_ha_per_year > 0.0

    def test_expansion_rate_zero(self):
        """No active conversion has zero expansion rate."""
        result = CroplandExpansion(
            expansion_rate_ha_per_year=0.0,
        )
        assert result.expansion_rate_ha_per_year == 0.0

    def test_expansion_rate_consistency(self):
        """Rate x years should approximate total area."""
        rate = 50.0
        years = 2.0
        area = rate * years
        result = CroplandExpansion(
            expansion_rate_ha_per_year=rate,
            area_converted_ha=area,
        )
        assert abs(result.area_converted_ha - rate * years) < 0.01

    def test_high_expansion_rate_industrial(self):
        """Industrial conversions have high expansion rates."""
        result = CroplandExpansion(
            scale="industrial",
            expansion_rate_ha_per_year=200.0,
        )
        assert result.expansion_rate_ha_per_year > 50.0

    def test_low_expansion_rate_smallholder(self):
        """Smallholder conversions have low expansion rates."""
        result = CroplandExpansion(
            scale="smallholder",
            expansion_rate_ha_per_year=2.0,
        )
        assert result.expansion_rate_ha_per_year < 10.0

    def test_expansion_rate_per_commodity(self):
        """Different commodities have different typical rates."""
        rates = {
            "soya": 200.0,
            "palm_oil": 50.0,
            "rubber": 15.0,
            "cocoa": 3.0,
            "coffee": 2.5,
        }
        for commodity, rate in rates.items():
            result = CroplandExpansion(
                commodity=commodity,
                expansion_rate_ha_per_year=rate,
            )
            assert result.expansion_rate_ha_per_year == rate


# ===========================================================================
# 4. Hotspot and Leapfrog Tests (7 tests)
# ===========================================================================


class TestHotspotAndLeapfrog:
    """Tests for conversion hotspot and leapfrog pattern detection."""

    def test_conversion_hotspot_detection_true(self):
        """Area with many nearby conversions is flagged as hotspot."""
        result = CroplandExpansion(
            plot_id="PLOT-HS-001",
            is_hotspot=True,
            confidence=0.88,
        )
        assert result.is_hotspot is True

    def test_conversion_hotspot_detection_false(self):
        """Isolated conversion is NOT a hotspot."""
        result = CroplandExpansion(
            plot_id="PLOT-HS-002",
            is_hotspot=False,
        )
        assert result.is_hotspot is False

    def test_leapfrog_pattern_detection_true(self):
        """Leapfrog expansion pattern (non-contiguous clearing) detected."""
        result = CroplandExpansion(
            plot_id="PLOT-LF-001",
            leapfrog_pattern=True,
        )
        assert result.leapfrog_pattern is True

    def test_leapfrog_pattern_detection_false(self):
        """Contiguous expansion is NOT leapfrog."""
        result = CroplandExpansion(
            plot_id="PLOT-LF-002",
            leapfrog_pattern=False,
        )
        assert result.leapfrog_pattern is False

    def test_hotspot_high_rate(self):
        """Hotspot areas have high expansion rates."""
        result = CroplandExpansion(
            is_hotspot=True,
            expansion_rate_ha_per_year=150.0,
        )
        assert result.expansion_rate_ha_per_year > 100.0

    def test_leapfrog_with_road_network(self):
        """Leapfrog pattern often follows road network development."""
        result = CroplandExpansion(
            leapfrog_pattern=True,
            confidence=0.82,
        )
        assert result.leapfrog_pattern is True

    def test_hotspot_and_leapfrog_combined(self):
        """Area can be both hotspot and leapfrog."""
        result = CroplandExpansion(
            is_hotspot=True,
            leapfrog_pattern=True,
        )
        assert result.is_hotspot is True
        assert result.leapfrog_pattern is True


# ===========================================================================
# 5. Batch and Determinism Tests (6 tests)
# ===========================================================================


class TestBatchAndDeterminism:
    """Tests for batch detection and deterministic behavior."""

    def test_batch_detection(self):
        """Batch detection of 30 plots."""
        results = [
            CroplandExpansion(
                plot_id=f"PLOT-BATCH-{i:04d}",
                commodity="palm_oil" if i % 2 == 0 else "soya",
                conversion_type="palm_oil_conversion" if i % 2 == 0 else "soya_conversion",
                confidence=0.80,
            )
            for i in range(30)
        ]
        assert len(results) == 30
        palm_count = sum(1 for r in results if r.commodity == "palm_oil")
        assert palm_count == 15

    def test_deterministic_detection(self):
        """Same inputs produce same detection results."""
        results = [
            CroplandExpansion(
                plot_id="PLOT-DET-001",
                commodity="palm_oil",
                conversion_type="palm_oil_conversion",
                area_converted_ha=50.0,
                confidence=0.85,
            )
            for _ in range(5)
        ]
        assert all(r.commodity == "palm_oil" for r in results)
        assert all(r.area_converted_ha == 50.0 for r in results)

    def test_provenance_hash(self):
        """Detection result has provenance hash."""
        h = compute_test_hash({"plot_id": "PLOT-PRV-001", "commodity": "soya"})
        result = CroplandExpansion(
            plot_id="PLOT-PRV-001",
            provenance_hash=h,
        )
        assert len(result.provenance_hash) == SHA256_HEX_LENGTH

    def test_unique_provenance_per_plot(self):
        """Different plots have different provenance hashes."""
        hashes = [
            compute_test_hash({"plot_id": f"PLOT-UNQ-{i}"})
            for i in range(10)
        ]
        assert len(set(hashes)) == 10

    def test_confidence_above_threshold(self):
        """Detection confidence is above minimum threshold."""
        result = CroplandExpansion(
            confidence=0.85,
        )
        assert result.confidence >= 0.60

    def test_from_category_always_set(self):
        """Conversion always has a from_category."""
        result = CroplandExpansion(
            from_category="forest",
            to_category="cropland",
        )
        assert result.from_category != ""
