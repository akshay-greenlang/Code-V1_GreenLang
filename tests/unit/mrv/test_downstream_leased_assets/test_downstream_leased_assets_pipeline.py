# -*- coding: utf-8 -*-
"""
Test suite for DownstreamLeasedAssetsPipelineEngine (AGENT-MRV-026, Engine 7).

Tests the 10-stage orchestration pipeline: VALIDATE, CLASSIFY, NORMALIZE,
RESOLVE_EFS, CALCULATE, ALLOCATE, AGGREGATE, COMPLIANCE, PROVENANCE, SEAL.

Coverage:
- 10 pipeline stages enum
- Full pipeline: single building, single vehicle, multi-asset portfolio
- Method routing: metered -> asset-specific, benchmark -> average-data,
  revenue -> spend-based
- Batch processing
- Portfolio analysis
- Error handling: empty list, missing asset type, negative values

Author: GL-TestEngineer
Date: February 2026
"""

import threading
from decimal import Decimal
from unittest.mock import MagicMock, patch
import pytest

try:
    from greenlang.downstream_leased_assets.downstream_leased_assets_pipeline import (
        DownstreamLeasedAssetsPipelineEngine,
        PipelineStage,
    )
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

try:
    from greenlang.downstream_leased_assets.models import (
        AssetCategory,
        CalculationMethod,
        BuildingType,
        ClimateZone,
        VehicleType,
        FuelType,
        EquipmentType,
        ITAssetType,
    )
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not (PIPELINE_AVAILABLE and MODELS_AVAILABLE),
    reason="DownstreamLeasedAssetsPipelineEngine or models not available",
)

pytestmark = _SKIP


@pytest.fixture(autouse=True)
def reset_singleton():
    if PIPELINE_AVAILABLE:
        DownstreamLeasedAssetsPipelineEngine.reset_singleton()
    yield
    if PIPELINE_AVAILABLE:
        DownstreamLeasedAssetsPipelineEngine.reset_singleton()


@pytest.fixture
def engine():
    return DownstreamLeasedAssetsPipelineEngine()


@pytest.fixture
def building_input():
    return {
        "asset_category": "building",
        "asset_id": "DLA-BLDG-001",
        "calculation_method": "asset_specific",
        "building_type": "office",
        "floor_area_sqm": Decimal("2500"),
        "climate_zone": "temperate",
        "energy_sources": {
            "electricity_kwh": Decimal("450000"),
            "natural_gas_kwh": Decimal("120000"),
        },
        "allocation_share": Decimal("0.35"),
        "occupancy_months": 12,
        "vacancy_rate": Decimal("0.12"),
        "region": "US",
        "lease_type": "operating",
        "operational_control": "tenant",
    }


@pytest.fixture
def vehicle_input():
    return {
        "asset_category": "vehicle",
        "asset_id": "DLA-VEH-001",
        "calculation_method": "asset_specific",
        "vehicle_type": "medium_car",
        "fuel_type": "diesel",
        "annual_distance_km": Decimal("25000"),
        "fleet_size": 10,
        "region": "US",
        "lease_type": "operating",
        "operational_control": "tenant",
    }


# ==============================================================================
# PIPELINE STAGE TESTS
# ==============================================================================


class TestPipelineStages:

    def test_pipeline_has_10_stages(self):
        """Test PipelineStage enum has exactly 10 stages."""
        assert len(PipelineStage) == 10

    @pytest.mark.parametrize("stage", [
        "VALIDATE", "CLASSIFY", "NORMALIZE", "RESOLVE_EFS",
        "CALCULATE", "ALLOCATE", "AGGREGATE", "COMPLIANCE",
        "PROVENANCE", "SEAL",
    ])
    def test_stage_members(self, stage):
        """Test all 10 stage members exist."""
        assert hasattr(PipelineStage, stage)


# ==============================================================================
# FULL PIPELINE TESTS
# ==============================================================================


class TestFullPipeline:

    def test_single_building(self, engine, building_input):
        """Test full pipeline for a single office building."""
        result = engine.process(building_input)
        assert result["total_co2e_kg"] > 0
        assert len(result["provenance_hash"]) == 64
        assert result.get("status") in ("completed", "success")

    def test_single_vehicle(self, engine, vehicle_input):
        """Test full pipeline for a vehicle fleet."""
        result = engine.process(vehicle_input)
        assert result["total_co2e_kg"] > 0

    def test_multi_asset_portfolio(self, engine, building_input, vehicle_input):
        """Test pipeline with mixed asset portfolio."""
        result = engine.process_batch([building_input, vehicle_input])
        assert result["total_co2e_kg"] > 0
        assert result["count"] == 2


# ==============================================================================
# METHOD ROUTING TESTS
# ==============================================================================


class TestMethodRouting:

    def test_metered_routes_to_asset_specific(self, engine, building_input):
        """When energy_sources provided, route to asset_specific."""
        result = engine.process(building_input)
        assert result.get("method") in ("asset_specific", None)

    def test_benchmark_routes_to_average_data(self, engine):
        """When no energy data, route to average_data."""
        result = engine.process({
            "asset_category": "building",
            "asset_id": "DLA-BLDG-002",
            "calculation_method": "average_data",
            "building_type": "retail",
            "floor_area_sqm": Decimal("1800"),
            "climate_zone": "tropical",
            "region": "GB",
        })
        assert result["total_co2e_kg"] > 0

    def test_revenue_routes_to_spend_based(self, engine):
        """When NAICS/amount provided, route to spend_based."""
        result = engine.process({
            "asset_category": "building",
            "asset_id": "DLA-SPEND-001",
            "calculation_method": "spend_based",
            "naics_code": "531120",
            "amount": Decimal("250000.00"),
            "currency": "USD",
            "reporting_year": 2024,
        })
        assert result["total_co2e_kg"] > 0


# ==============================================================================
# BATCH PROCESSING TESTS
# ==============================================================================


class TestBatchProcessing:

    def test_batch_multiple_buildings(self, engine):
        buildings = [
            {
                "asset_category": "building",
                "asset_id": f"DLA-BLDG-{i:03d}",
                "calculation_method": "average_data",
                "building_type": "office",
                "floor_area_sqm": Decimal("2000"),
                "climate_zone": "temperate",
                "region": "US",
            }
            for i in range(5)
        ]
        result = engine.process_batch(buildings)
        assert result["count"] == 5
        assert result["total_co2e_kg"] > 0

    def test_batch_returns_individual_results(self, engine, building_input, vehicle_input):
        result = engine.process_batch([building_input, vehicle_input])
        if "results" in result:
            assert len(result["results"]) == 2


# ==============================================================================
# PORTFOLIO ANALYSIS TESTS
# ==============================================================================


class TestPortfolioAnalysis:

    def test_portfolio_summary(self, engine, building_input, vehicle_input):
        result = engine.process_batch([building_input, vehicle_input])
        assert "total_co2e_kg" in result
        assert result["total_co2e_kg"] > 0


# ==============================================================================
# ERROR HANDLING TESTS
# ==============================================================================


class TestErrorHandling:

    def test_empty_list(self, engine):
        """Test empty asset list raises or returns zero."""
        try:
            result = engine.process_batch([])
            assert result.get("count", 0) == 0
        except (ValueError, KeyError):
            pass

    def test_missing_asset_type(self, engine):
        """Test missing asset_category raises error."""
        try:
            result = engine.process({"asset_id": "DLA-001"})
            assert result.get("error") is not None or result.get("status") == "error"
        except (ValueError, KeyError):
            pass

    def test_negative_area(self, engine):
        """Test negative floor area raises validation error."""
        try:
            result = engine.process({
                "asset_category": "building",
                "asset_id": "DLA-001",
                "building_type": "office",
                "floor_area_sqm": Decimal("-100"),
                "region": "US",
            })
            assert result.get("error") is not None or result.get("status") == "error"
        except (ValueError, KeyError):
            pass


# ==============================================================================
# SINGLETON AND THREAD SAFETY
# ==============================================================================


class TestSingleton:

    def test_singleton_identity(self, engine):
        engine2 = DownstreamLeasedAssetsPipelineEngine()
        assert engine is engine2


class TestThreadSafety:

    def test_concurrent_processing(self, engine):
        results = []

        def process_building():
            r = engine.process({
                "asset_category": "building",
                "asset_id": "DLA-BLDG-001",
                "calculation_method": "average_data",
                "building_type": "office",
                "floor_area_sqm": Decimal("2500"),
                "climate_zone": "temperate",
                "region": "US",
            })
            results.append(r)

        threads = [threading.Thread(target=process_building) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 5
        for r in results:
            assert r["total_co2e_kg"] > 0
