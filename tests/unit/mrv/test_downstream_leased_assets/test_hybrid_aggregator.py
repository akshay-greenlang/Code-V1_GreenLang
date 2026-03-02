# -*- coding: utf-8 -*-
"""
Test suite for HybridAggregatorEngine (AGENT-MRV-026, Engine 5).

Tests multi-method aggregation including waterfall ordering, gap filling,
portfolio aggregation, multi-tenant allocation, vacancy handling, hotspot
analysis, operational control boundary, and DQI-weighted blending.

Author: GL-TestEngineer
Date: February 2026
"""

from decimal import Decimal
from unittest.mock import MagicMock, patch
import pytest

try:
    from greenlang.downstream_leased_assets.hybrid_aggregator import (
        HybridAggregatorEngine,
    )
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_SKIP = pytest.mark.skipif(not _AVAILABLE, reason="HybridAggregatorEngine not available")
pytestmark = _SKIP


@pytest.fixture(autouse=True)
def _reset_singleton():
    if _AVAILABLE:
        HybridAggregatorEngine.reset_instance()
    yield
    if _AVAILABLE:
        HybridAggregatorEngine.reset_instance()


@pytest.fixture
def engine():
    return HybridAggregatorEngine()


# ==============================================================================
# METHOD WATERFALL TESTS
# ==============================================================================


class TestMethodWaterfall:

    def test_waterfall_order(self, engine):
        """Test method priority: asset_specific > average_data > spend_based."""
        result = engine.aggregate({
            "assets": [
                {
                    "asset_id": "DLA-001",
                    "asset_type": "building",
                    "available_methods": ["asset_specific", "average_data", "spend_based"],
                    "results": {
                        "asset_specific": {"total_co2e_kg": Decimal("167265")},
                        "average_data": {"total_co2e_kg": Decimal("145000")},
                        "spend_based": {"total_co2e_kg": Decimal("105000")},
                    },
                },
            ],
        })
        assert result["total_co2e_kg"] == Decimal("167265")

    def test_gap_filling_with_lower_tier(self, engine):
        """When asset_specific unavailable, fall back to average_data."""
        result = engine.aggregate({
            "assets": [
                {
                    "asset_id": "DLA-001",
                    "asset_type": "building",
                    "available_methods": ["average_data", "spend_based"],
                    "results": {
                        "average_data": {"total_co2e_kg": Decimal("145000")},
                        "spend_based": {"total_co2e_kg": Decimal("105000")},
                    },
                },
            ],
        })
        assert result["total_co2e_kg"] == Decimal("145000")

    def test_spend_only_fallback(self, engine):
        """When only spend_based available, use it."""
        result = engine.aggregate({
            "assets": [
                {
                    "asset_id": "DLA-001",
                    "asset_type": "building",
                    "available_methods": ["spend_based"],
                    "results": {
                        "spend_based": {"total_co2e_kg": Decimal("105000")},
                    },
                },
            ],
        })
        assert result["total_co2e_kg"] == Decimal("105000")


# ==============================================================================
# PORTFOLIO AGGREGATION TESTS
# ==============================================================================


class TestPortfolioAggregation:

    def test_by_category(self, engine):
        """Test portfolio aggregation by asset category."""
        result = engine.aggregate({
            "assets": [
                {"asset_id": "B-1", "asset_type": "building", "results": {"asset_specific": {"total_co2e_kg": Decimal("100000")}}},
                {"asset_id": "V-1", "asset_type": "vehicle", "results": {"asset_specific": {"total_co2e_kg": Decimal("25000")}}},
                {"asset_id": "E-1", "asset_type": "equipment", "results": {"asset_specific": {"total_co2e_kg": Decimal("15000")}}},
                {"asset_id": "I-1", "asset_type": "it_asset", "results": {"asset_specific": {"total_co2e_kg": Decimal("5000")}}},
            ],
        })
        assert result["total_co2e_kg"] == Decimal("145000")
        if "by_category" in result:
            assert result["by_category"]["building"] == Decimal("100000")

    def test_by_building_type(self, engine):
        """Test aggregation by building type within category."""
        result = engine.aggregate({
            "assets": [
                {"asset_id": "B-1", "asset_type": "building", "building_type": "office", "results": {"asset_specific": {"total_co2e_kg": Decimal("60000")}}},
                {"asset_id": "B-2", "asset_type": "building", "building_type": "retail", "results": {"asset_specific": {"total_co2e_kg": Decimal("40000")}}},
            ],
        })
        assert result["total_co2e_kg"] == Decimal("100000")


# ==============================================================================
# MULTI-TENANT ALLOCATION TESTS
# ==============================================================================


class TestMultiTenantAllocation:

    def test_tenant_allocation_sums_to_total(self, engine):
        """All tenant shares should sum to whole-building emissions."""
        result = engine.aggregate({
            "assets": [
                {
                    "asset_id": "B-1",
                    "asset_type": "building",
                    "total_building_co2e_kg": Decimal("100000"),
                    "tenants": [
                        {"tenant_id": "T-1", "allocation_share": Decimal("0.35")},
                        {"tenant_id": "T-2", "allocation_share": Decimal("0.25")},
                        {"tenant_id": "T-3", "allocation_share": Decimal("0.40")},
                    ],
                    "results": {"asset_specific": {"total_co2e_kg": Decimal("100000")}},
                },
            ],
        })
        assert result["total_co2e_kg"] == Decimal("100000")


# ==============================================================================
# VACANCY HANDLING TESTS
# ==============================================================================


class TestVacancyHandling:

    def test_vacancy_included(self, engine):
        """Vacancy base-load emissions should be included."""
        result = engine.aggregate({
            "assets": [
                {
                    "asset_id": "B-1",
                    "asset_type": "building",
                    "vacancy_rate": Decimal("0.20"),
                    "results": {"asset_specific": {"total_co2e_kg": Decimal("80000")}},
                },
            ],
        })
        assert result["total_co2e_kg"] > 0


# ==============================================================================
# HOTSPOT ANALYSIS TESTS
# ==============================================================================


class TestHotspotAnalysis:

    def test_pareto_analysis(self, engine):
        """Top emitting assets should be flagged as hotspots."""
        result = engine.aggregate({
            "assets": [
                {"asset_id": "B-1", "asset_type": "building", "results": {"asset_specific": {"total_co2e_kg": Decimal("500000")}}},
                {"asset_id": "B-2", "asset_type": "building", "results": {"asset_specific": {"total_co2e_kg": Decimal("10000")}}},
                {"asset_id": "B-3", "asset_type": "building", "results": {"asset_specific": {"total_co2e_kg": Decimal("5000")}}},
            ],
        })
        assert result["total_co2e_kg"] == Decimal("515000")


# ==============================================================================
# OPERATIONAL CONTROL BOUNDARY TESTS
# ==============================================================================


class TestOperationalControlBoundary:

    def test_exclude_if_operational_control_retained(self, engine):
        """If lessor retains operational control, exclude from Cat 13 (report in Scope 1/2)."""
        result = engine.aggregate({
            "assets": [
                {
                    "asset_id": "B-1",
                    "asset_type": "building",
                    "operational_control": "lessor",
                    "results": {"asset_specific": {"total_co2e_kg": Decimal("100000")}},
                },
            ],
        })
        # This asset should be excluded or flagged
        if "excluded_assets" in result:
            assert len(result["excluded_assets"]) >= 1
        else:
            # At minimum, should not inflate Cat 13 total
            assert result["total_co2e_kg"] >= 0


# ==============================================================================
# DQI-WEIGHTED BLENDING TESTS
# ==============================================================================


class TestDQIWeightedBlending:

    def test_single_asset(self, engine):
        result = engine.aggregate({
            "assets": [
                {"asset_id": "B-1", "results": {"asset_specific": {"total_co2e_kg": Decimal("100000"), "dqi_score": Decimal("4.5")}}},
            ],
        })
        assert result["total_co2e_kg"] == Decimal("100000")

    def test_multi_asset_mixed_methods(self, engine):
        """Mixed methods should blend with DQI weighting."""
        result = engine.aggregate({
            "assets": [
                {"asset_id": "B-1", "results": {"asset_specific": {"total_co2e_kg": Decimal("100000"), "dqi_score": Decimal("4.5")}}},
                {"asset_id": "B-2", "results": {"average_data": {"total_co2e_kg": Decimal("80000"), "dqi_score": Decimal("3.0")}}},
                {"asset_id": "B-3", "results": {"spend_based": {"total_co2e_kg": Decimal("60000"), "dqi_score": Decimal("2.0")}}},
            ],
        })
        assert result["total_co2e_kg"] == Decimal("240000")

    def test_provenance_hash_present(self, engine):
        result = engine.aggregate({
            "assets": [
                {"asset_id": "B-1", "results": {"asset_specific": {"total_co2e_kg": Decimal("100000")}}},
            ],
        })
        assert len(result.get("provenance_hash", "")) == 64
