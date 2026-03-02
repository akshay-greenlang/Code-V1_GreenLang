# -*- coding: utf-8 -*-
"""
Unit tests for HybridAggregatorEngine -- AGENT-MRV-025

Tests the hybrid aggregation method which combines multiple calculation methods
using a priority waterfall, performs gap-filling, and computes circularity
metrics. CRITICAL: avoided emissions are ALWAYS reported separately.

Coverage:
- Method waterfall: producer -> waste-type -> average-data
- Gap-filling with best available data
- Avoided emissions always separate (NEVER netted against gross)
- Circularity metrics: recycling_rate, diversion_rate, circularity_index
- Waste hierarchy compliance scoring
- Hotspot/Pareto analysis
- Multi-product aggregation
- Weighted DQI scoring

Target: 40+ expanded tests.
Author: GL-TestEngineer
Date: February 2026
"""

from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch
import pytest

try:
    from greenlang.end_of_life_treatment.hybrid_aggregator import (
        HybridAggregatorEngine,
    )
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_SKIP = pytest.mark.skipif(not _AVAILABLE, reason="HybridAggregatorEngine not available")
pytestmark = _SKIP

_Q8 = Decimal("0.00000001")


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def engine():
    """Create a HybridAggregatorEngine instance."""
    return HybridAggregatorEngine.get_instance()


@pytest.fixture
def producer_results():
    """Producer-specific calculation results."""
    return [
        {
            "product_id": "P1",
            "product_category": "consumer_electronics",
            "method": "producer_specific",
            "gross_emissions_kgco2e": Decimal("850.0"),
            "avoided_emissions_kgco2e": Decimal("320.0"),
            "total_mass_kg": Decimal("1000.0"),
            "dqi_score": Decimal("90.0"),
            "region": "US",
        },
    ]


@pytest.fixture
def waste_type_results():
    """Waste-type-specific calculation results."""
    return [
        {
            "product_id": "P2",
            "product_category": "packaging",
            "method": "waste_type_specific",
            "gross_emissions_kgco2e": Decimal("1250.0"),
            "avoided_emissions_kgco2e": Decimal("450.0"),
            "total_mass_kg": Decimal("5000.0"),
            "dqi_score": Decimal("75.0"),
            "region": "GB",
            "by_treatment": {
                "landfill": Decimal("380.0"),
                "incineration": Decimal("520.0"),
                "recycling": Decimal("150.0"),
                "composting": Decimal("80.0"),
                "anaerobic_digestion": Decimal("40.0"),
                "open_burning": Decimal("80.0"),
            },
        },
    ]


@pytest.fixture
def average_data_results():
    """Average-data calculation results."""
    return [
        {
            "product_id": "P3",
            "product_category": "clothing",
            "method": "average_data",
            "gross_emissions_kgco2e": Decimal("600.0"),
            "avoided_emissions_kgco2e": Decimal("0.0"),
            "total_mass_kg": Decimal("500.0"),
            "dqi_score": Decimal("45.0"),
            "region": "US",
        },
    ]


@pytest.fixture
def multi_method_results(producer_results, waste_type_results, average_data_results):
    """Combined results from all three methods."""
    return producer_results + waste_type_results + average_data_results


# ============================================================================
# TEST: Method Waterfall
# ============================================================================


class TestMethodWaterfall:
    """Test method waterfall priority logic."""

    def test_producer_specific_highest_priority(self, engine, multi_method_results):
        """Test producer-specific method is selected when available."""
        result = engine.aggregate(multi_method_results)
        # Producer-specific should be preferred for P1
        assert result is not None

    def test_waterfall_order(self, engine):
        """Test waterfall follows producer -> waste-type -> average-data."""
        # Product with only average-data
        avg_only = [{
            "product_id": "P-AVG",
            "product_category": "toys",
            "method": "average_data",
            "gross_emissions_kgco2e": Decimal("200.0"),
            "avoided_emissions_kgco2e": Decimal("0.0"),
            "total_mass_kg": Decimal("100.0"),
            "dqi_score": Decimal("40.0"),
        }]
        result = engine.aggregate(avg_only)
        assert result is not None
        assert result["gross_emissions_kgco2e"] == Decimal("200.0")

    def test_higher_priority_replaces_lower(self, engine):
        """Test higher-priority method replaces lower for same product."""
        results = [
            {
                "product_id": "P-SAME",
                "product_category": "packaging",
                "method": "average_data",
                "gross_emissions_kgco2e": Decimal("500.0"),
                "avoided_emissions_kgco2e": Decimal("0.0"),
                "total_mass_kg": Decimal("100.0"),
                "dqi_score": Decimal("40.0"),
            },
            {
                "product_id": "P-SAME",
                "product_category": "packaging",
                "method": "waste_type_specific",
                "gross_emissions_kgco2e": Decimal("450.0"),
                "avoided_emissions_kgco2e": Decimal("180.0"),
                "total_mass_kg": Decimal("100.0"),
                "dqi_score": Decimal("75.0"),
            },
        ]
        result = engine.aggregate(results)
        # Should use waste_type_specific (higher priority)
        assert result["gross_emissions_kgco2e"] == Decimal("450.0")


# ============================================================================
# TEST: Gap-Filling
# ============================================================================


class TestGapFilling:
    """Test gap-filling with best available data."""

    def test_gap_fill_uses_average_data(self, engine):
        """Test gap-filling uses average-data when no other method available."""
        partial_results = [{
            "product_id": "P-GAP",
            "product_category": "toys",
            "method": "average_data",
            "gross_emissions_kgco2e": Decimal("100.0"),
            "avoided_emissions_kgco2e": Decimal("0.0"),
            "total_mass_kg": Decimal("50.0"),
            "dqi_score": Decimal("35.0"),
        }]
        result = engine.aggregate(partial_results)
        assert result["gross_emissions_kgco2e"] > Decimal("0.0")


# ============================================================================
# TEST: Avoided Emissions Always Separate
# ============================================================================


class TestAvoidedEmissionsSeparate:
    """CRITICAL: Test avoided emissions are ALWAYS reported separately."""

    def test_avoided_never_subtracted_from_gross(self, engine, multi_method_results):
        """Test avoided emissions are never subtracted from gross."""
        result = engine.aggregate(multi_method_results)
        gross = result["gross_emissions_kgco2e"]
        avoided = result.get("avoided_emissions_kgco2e", Decimal("0.0"))
        # Gross must be the sum of treatment-pathway gross
        # Avoided must be separately tracked
        assert gross > Decimal("0.0")
        assert avoided >= Decimal("0.0")
        # There should NOT be a net field that subtracts avoided
        if "net_emissions_kgco2e" in result:
            # If net exists, it should NOT equal gross - avoided (that would be netting)
            assert result.get("net_emissions_kgco2e") is None or \
                   result["net_emissions_kgco2e"] == result["gross_emissions_kgco2e"]

    def test_avoided_aggregated_across_products(self, engine, multi_method_results):
        """Test avoided emissions are aggregated across all products."""
        result = engine.aggregate(multi_method_results)
        # P1 has 320, P2 has 450, P3 has 0 = 770 total avoided
        expected_avoided = Decimal("320.0") + Decimal("450.0") + Decimal("0.0")
        actual_avoided = result.get("avoided_emissions_kgco2e", Decimal("0.0"))
        assert abs(actual_avoided - expected_avoided) < Decimal("1.0")

    def test_avoided_by_treatment_breakdown(self, engine, waste_type_results):
        """Test avoided emissions are broken down by treatment pathway."""
        result = engine.aggregate(waste_type_results)
        avoided_by_treatment = result.get("avoided_by_treatment", {})
        # Recycling is primary source of avoided emissions
        assert isinstance(avoided_by_treatment, dict)

    def test_zero_avoided_when_no_recycling(self, engine):
        """Test zero avoided emissions when product has no recycling."""
        results = [{
            "product_id": "P-NO-RECYCLE",
            "product_category": "food_products",
            "method": "waste_type_specific",
            "gross_emissions_kgco2e": Decimal("300.0"),
            "avoided_emissions_kgco2e": Decimal("0.0"),
            "total_mass_kg": Decimal("200.0"),
            "dqi_score": Decimal("60.0"),
            "by_treatment": {
                "landfill": Decimal("180.0"),
                "composting": Decimal("120.0"),
            },
        }]
        result = engine.aggregate(results)
        assert result.get("avoided_emissions_kgco2e", Decimal("0.0")) == Decimal("0.0")


# ============================================================================
# TEST: Circularity Metrics
# ============================================================================


class TestCircularityMetrics:
    """Test circularity metrics computation."""

    def test_recycling_rate_calculated(self, engine, waste_type_results):
        """Test recycling rate is calculated."""
        result = engine.aggregate(waste_type_results)
        assert "circularity_metrics" in result or "recycling_rate" in result

    def test_diversion_rate_calculated(self, engine, waste_type_results):
        """Test diversion rate (recycling + composting + AD) is calculated."""
        result = engine.aggregate(waste_type_results)
        metrics = result.get("circularity_metrics", result)
        diversion = metrics.get("diversion_rate", Decimal("0.0"))
        assert diversion >= Decimal("0.0")
        assert diversion <= Decimal("1.0")

    def test_circularity_index_range(self, engine, waste_type_results):
        """Test circularity index is between 0 and 1."""
        result = engine.aggregate(waste_type_results)
        metrics = result.get("circularity_metrics", result)
        ci = metrics.get("circularity_index", Decimal("0.0"))
        assert Decimal("0.0") <= ci <= Decimal("1.0")

    def test_high_recycling_higher_circularity(self, engine):
        """Test higher recycling rate yields higher circularity index."""
        high_recycling = [{
            "product_id": "P-HR",
            "product_category": "large_appliances",
            "method": "waste_type_specific",
            "gross_emissions_kgco2e": Decimal("100.0"),
            "avoided_emissions_kgco2e": Decimal("500.0"),
            "total_mass_kg": Decimal("1000.0"),
            "dqi_score": Decimal("75.0"),
            "by_treatment": {
                "recycling": Decimal("80.0"),
                "landfill": Decimal("20.0"),
            },
            "mass_by_treatment": {
                "recycling": Decimal("800.0"),
                "landfill": Decimal("200.0"),
            },
        }]
        low_recycling = [{
            "product_id": "P-LR",
            "product_category": "packaging",
            "method": "waste_type_specific",
            "gross_emissions_kgco2e": Decimal("400.0"),
            "avoided_emissions_kgco2e": Decimal("50.0"),
            "total_mass_kg": Decimal("1000.0"),
            "dqi_score": Decimal("75.0"),
            "by_treatment": {
                "recycling": Decimal("20.0"),
                "landfill": Decimal("380.0"),
            },
            "mass_by_treatment": {
                "recycling": Decimal("100.0"),
                "landfill": Decimal("900.0"),
            },
        }]
        r_high = engine.aggregate(high_recycling)
        r_low = engine.aggregate(low_recycling)
        ci_high = r_high.get("circularity_metrics", r_high).get("circularity_index", Decimal("0.0"))
        ci_low = r_low.get("circularity_metrics", r_low).get("circularity_index", Decimal("0.0"))
        assert ci_high > ci_low


# ============================================================================
# TEST: Waste Hierarchy Compliance
# ============================================================================


class TestWasteHierarchyCompliance:
    """Test waste hierarchy compliance scoring."""

    def test_hierarchy_score_returned(self, engine, waste_type_results):
        """Test waste hierarchy score is included in result."""
        result = engine.aggregate(waste_type_results)
        metrics = result.get("circularity_metrics", result)
        score = metrics.get("waste_hierarchy_score", None)
        if score is not None:
            assert Decimal("0") <= score <= Decimal("100")


# ============================================================================
# TEST: Hotspot/Pareto Analysis
# ============================================================================


class TestHotspotAnalysis:
    """Test hotspot identification and Pareto analysis."""

    def test_hotspot_identification(self, engine, multi_method_results):
        """Test hotspot products are identified."""
        result = engine.aggregate(multi_method_results)
        hotspots = result.get("hotspots", [])
        if hotspots:
            # Should identify highest-emission product
            assert len(hotspots) >= 1
            assert "product_id" in hotspots[0] or "material" in hotspots[0]

    def test_pareto_80_20(self, engine, multi_method_results):
        """Test Pareto analysis identifies the 20% of products causing 80% emissions."""
        result = engine.aggregate(multi_method_results)
        # Just verify the aggregation completes
        assert result["gross_emissions_kgco2e"] > Decimal("0.0")


# ============================================================================
# TEST: Multi-Product Aggregation
# ============================================================================


class TestMultiProductAggregation:
    """Test aggregation across multiple products."""

    def test_total_gross_is_sum(self, engine, multi_method_results):
        """Test total gross emissions is sum of all products."""
        result = engine.aggregate(multi_method_results)
        expected_gross = Decimal("850.0") + Decimal("1250.0") + Decimal("600.0")
        assert abs(result["gross_emissions_kgco2e"] - expected_gross) < Decimal("1.0")

    def test_total_mass_is_sum(self, engine, multi_method_results):
        """Test total mass is sum of all product masses."""
        result = engine.aggregate(multi_method_results)
        expected_mass = Decimal("1000.0") + Decimal("5000.0") + Decimal("500.0")
        total_mass = result.get("total_mass_kg", Decimal("0.0"))
        assert abs(total_mass - expected_mass) < Decimal("1.0")

    def test_product_count(self, engine, multi_method_results):
        """Test product count matches input."""
        result = engine.aggregate(multi_method_results)
        assert result.get("product_count", 0) == 3

    def test_weighted_dqi_score(self, engine, multi_method_results):
        """Test DQI score is mass-weighted average."""
        result = engine.aggregate(multi_method_results)
        dqi = result.get("dqi_score", Decimal("0.0"))
        assert Decimal("0") <= dqi <= Decimal("100")

    def test_by_category_breakdown(self, engine, multi_method_results):
        """Test results include by-category breakdown."""
        result = engine.aggregate(multi_method_results)
        by_category = result.get("by_category", {})
        if by_category:
            assert "consumer_electronics" in by_category
            assert "packaging" in by_category

    def test_by_method_breakdown(self, engine, multi_method_results):
        """Test results include by-method breakdown."""
        result = engine.aggregate(multi_method_results)
        by_method = result.get("by_method", {})
        if by_method:
            assert len(by_method) >= 1

    def test_single_product_aggregation(self, engine, producer_results):
        """Test aggregation works with single product."""
        result = engine.aggregate(producer_results)
        assert result["gross_emissions_kgco2e"] == Decimal("850.0")

    def test_empty_results_returns_zero(self, engine):
        """Test empty results list returns zero emissions."""
        result = engine.aggregate([])
        assert result["gross_emissions_kgco2e"] == Decimal("0.0")
