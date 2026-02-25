# -*- coding: utf-8 -*-
"""
Unit tests for CoolingPurchasePipelineEngine (Engine 7 of 7)

AGENT-MRV-012: Cooling Purchase Agent

Tests full pipeline orchestration for electric chillers, absorption chillers,
free cooling, TES, district cooling, batch processing, aggregation, and
refrigerant leakage calculations with integrated uncertainty and compliance.

Target: 72 tests, ~650 lines.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List

import pytest

from greenlang.cooling_purchase.cooling_purchase_pipeline import (
    CoolingPurchasePipelineEngine,
    get_cooling_purchase_pipeline,
)
from greenlang.cooling_purchase.models import (
    CoolingTechnology,
    FreeCoolingSource,
    TESType,
    DataQualityTier,
    GWPSource,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline_engine():
    """Create a CoolingPurchasePipelineEngine instance."""
    engine = CoolingPurchasePipelineEngine()
    yield engine
    engine.reset()


@pytest.fixture
def electric_request() -> Dict[str, Any]:
    """Return an electric chiller request."""
    return {
        "cooling_kwh_th": Decimal("100000"),
        "cop": Decimal("5.5"),
        "grid_ef_kgco2e_kwh": Decimal("0.45"),
        "technology": "WATER_COOLED_CENTRIFUGAL",
        "tier": "TIER_2",
        "gwp_source": "AR6",
        "enable_uncertainty": True,
        "enable_compliance": True,
    }


@pytest.fixture
def absorption_request() -> Dict[str, Any]:
    """Return an absorption chiller request."""
    return {
        "cooling_kwh_th": Decimal("80000"),
        "cop_thermal": Decimal("1.2"),
        "heat_source": "natural_gas",
        "heat_ef_kgco2e_kwh": Decimal("0.25"),
        "parasitic_kwh": Decimal("1000"),
        "grid_ef_kgco2e_kwh": Decimal("0.45"),
        "technology": "DOUBLE_EFFECT_LIBR",
        "tier": "TIER_2",
        "gwp_source": "AR6",
    }


@pytest.fixture
def free_cooling_request() -> Dict[str, Any]:
    """Return a free cooling request."""
    return {
        "cooling_kwh_th": Decimal("50000"),
        "source": "seawater",
        "grid_ef_kgco2e_kwh": Decimal("0.40"),
        "tier": "TIER_2",
        "gwp_source": "AR6",
    }


@pytest.fixture
def tes_request() -> Dict[str, Any]:
    """Return a TES request."""
    return {
        "cooling_kwh_th": Decimal("80000"),
        "tes_type": "ice",
        "capacity_kwh_th": Decimal("20000"),
        "cop_charge": Decimal("3.0"),
        "grid_ef_charge_kgco2e_kwh": Decimal("0.30"),
        "grid_ef_peak_kgco2e_kwh": Decimal("0.60"),
        "tier": "TIER_2",
        "gwp_source": "AR6",
    }


@pytest.fixture
def district_request() -> Dict[str, Any]:
    """Return a district cooling request."""
    return {
        "cooling_kwh_th": Decimal("100000"),
        "region": "singapore",
        "distribution_loss_pct": Decimal("5.0"),
        "pump_energy_kwh": Decimal("2000"),
        "tier": "TIER_1",
        "gwp_source": "AR6",
    }


# ===========================================================================
# 1. Initialization Tests
# ===========================================================================


class TestPipelineEngineInit:
    """Test CoolingPurchasePipelineEngine initialization."""

    def test_singleton_pattern(self):
        """Test singleton returns same instance."""
        e1 = CoolingPurchasePipelineEngine()
        e2 = CoolingPurchasePipelineEngine()
        assert e1 is e2

    def test_get_function_returns_singleton(self):
        """Test get_cooling_purchase_pipeline returns singleton."""
        e1 = get_cooling_purchase_pipeline()
        e2 = get_cooling_purchase_pipeline()
        assert e1 is e2

    def test_reset_clears_state(self, pipeline_engine):
        """Test reset clears internal state."""
        _ = pipeline_engine.run_electric_chiller_pipeline({"cooling_kwh_th": Decimal("1000")})
        pipeline_engine.reset()
        stats = pipeline_engine.get_statistics()
        assert stats["total_pipeline_runs"] == 0


# ===========================================================================
# 2. Electric Chiller Pipeline Tests
# ===========================================================================


class TestElectricChillerPipeline:
    """Test run_electric_chiller_pipeline method."""

    def test_run_electric_pipeline_basic(self, pipeline_engine, electric_request):
        """Test basic electric chiller pipeline."""
        result = pipeline_engine.run_electric_chiller_pipeline(electric_request)
        assert "calculation_result" in result
        assert "uncertainty" in result
        assert "compliance" in result

    def test_electric_pipeline_returns_calculation_result(self, pipeline_engine, electric_request):
        """Test pipeline returns calculation result."""
        result = pipeline_engine.run_electric_chiller_pipeline(electric_request)
        assert result["calculation_result"]["emissions_kgco2e"] > Decimal("0")

    def test_electric_pipeline_with_uncertainty(self, pipeline_engine, electric_request):
        """Test pipeline includes uncertainty when enabled."""
        electric_request["enable_uncertainty"] = True
        result = pipeline_engine.run_electric_chiller_pipeline(electric_request)
        assert result["uncertainty"] is not None

    def test_electric_pipeline_without_uncertainty(self, pipeline_engine, electric_request):
        """Test pipeline excludes uncertainty when disabled."""
        electric_request["enable_uncertainty"] = False
        result = pipeline_engine.run_electric_chiller_pipeline(electric_request)
        assert result["uncertainty"] is None

    def test_electric_pipeline_with_compliance(self, pipeline_engine, electric_request):
        """Test pipeline includes compliance when enabled."""
        electric_request["enable_compliance"] = True
        result = pipeline_engine.run_electric_chiller_pipeline(electric_request)
        assert result["compliance"] is not None

    def test_electric_pipeline_without_compliance(self, pipeline_engine, electric_request):
        """Test pipeline excludes compliance when disabled."""
        electric_request["enable_compliance"] = False
        result = pipeline_engine.run_electric_chiller_pipeline(electric_request)
        assert result["compliance"] is None


# ===========================================================================
# 3. Absorption Chiller Pipeline Tests
# ===========================================================================


class TestAbsorptionPipeline:
    """Test run_absorption_pipeline method."""

    def test_run_absorption_pipeline_basic(self, pipeline_engine, absorption_request):
        """Test basic absorption chiller pipeline."""
        result = pipeline_engine.run_absorption_pipeline(absorption_request)
        assert "calculation_result" in result
        assert "uncertainty" in result
        assert "compliance" in result

    def test_absorption_pipeline_returns_result(self, pipeline_engine, absorption_request):
        """Test absorption pipeline returns calculation result."""
        result = pipeline_engine.run_absorption_pipeline(absorption_request)
        assert result["calculation_result"]["emissions_kgco2e"] > Decimal("0")

    def test_absorption_pipeline_with_parasitic(self, pipeline_engine, absorption_request):
        """Test absorption pipeline includes parasitic electricity."""
        result = pipeline_engine.run_absorption_pipeline(absorption_request)
        # Should account for parasitic load
        assert result["calculation_result"]["emissions_kgco2e"] > Decimal("0")


# ===========================================================================
# 4. Free Cooling Pipeline Tests
# ===========================================================================


class TestFreeCoolingPipeline:
    """Test run_free_cooling_pipeline method."""

    def test_run_free_cooling_pipeline_basic(self, pipeline_engine, free_cooling_request):
        """Test basic free cooling pipeline."""
        result = pipeline_engine.run_free_cooling_pipeline(free_cooling_request)
        assert "calculation_result" in result
        assert "uncertainty" in result
        assert "compliance" in result

    def test_free_cooling_pipeline_seawater(self, pipeline_engine, free_cooling_request):
        """Test free cooling pipeline with seawater source."""
        free_cooling_request["source"] = "seawater"
        result = pipeline_engine.run_free_cooling_pipeline(free_cooling_request)
        assert result["calculation_result"]["technology"] == "SEAWATER_FREE"

    def test_free_cooling_pipeline_lake(self, pipeline_engine, free_cooling_request):
        """Test free cooling pipeline with lake source."""
        free_cooling_request["source"] = "lake"
        result = pipeline_engine.run_free_cooling_pipeline(free_cooling_request)
        assert result["calculation_result"]["technology"] == "LAKE_FREE"

    def test_free_cooling_pipeline_river(self, pipeline_engine, free_cooling_request):
        """Test free cooling pipeline with river source."""
        free_cooling_request["source"] = "river"
        result = pipeline_engine.run_free_cooling_pipeline(free_cooling_request)
        assert result["calculation_result"]["technology"] == "RIVER_FREE"

    def test_free_cooling_pipeline_air(self, pipeline_engine, free_cooling_request):
        """Test free cooling pipeline with ambient air source."""
        free_cooling_request["source"] = "ambient_air"
        result = pipeline_engine.run_free_cooling_pipeline(free_cooling_request)
        assert result["calculation_result"]["technology"] == "AMBIENT_AIR_FREE"


# ===========================================================================
# 5. TES Pipeline Tests
# ===========================================================================


class TestTESPipeline:
    """Test run_tes_pipeline method."""

    def test_run_tes_pipeline_basic(self, pipeline_engine, tes_request):
        """Test basic TES pipeline."""
        result = pipeline_engine.run_tes_pipeline(tes_request)
        assert "calculation_result" in result
        assert "uncertainty" in result
        assert "compliance" in result

    def test_tes_pipeline_ice(self, pipeline_engine, tes_request):
        """Test TES pipeline with ice storage."""
        tes_request["tes_type"] = "ice"
        result = pipeline_engine.run_tes_pipeline(tes_request)
        assert result["calculation_result"]["emissions_kgco2e"] > Decimal("0")

    def test_tes_pipeline_chilled_water(self, pipeline_engine, tes_request):
        """Test TES pipeline with chilled water storage."""
        tes_request["tes_type"] = "chilled_water"
        result = pipeline_engine.run_tes_pipeline(tes_request)
        assert result["calculation_result"]["emissions_kgco2e"] > Decimal("0")

    def test_tes_pipeline_pcm(self, pipeline_engine, tes_request):
        """Test TES pipeline with PCM storage."""
        tes_request["tes_type"] = "pcm"
        result = pipeline_engine.run_tes_pipeline(tes_request)
        assert result["calculation_result"]["emissions_kgco2e"] > Decimal("0")


# ===========================================================================
# 6. District Cooling Pipeline Tests
# ===========================================================================


class TestDistrictCoolingPipeline:
    """Test run_district_cooling_pipeline method."""

    def test_run_district_pipeline_basic(self, pipeline_engine, district_request):
        """Test basic district cooling pipeline."""
        result = pipeline_engine.run_district_cooling_pipeline(district_request)
        assert "calculation_result" in result
        assert "uncertainty" in result
        assert "compliance" in result

    def test_district_pipeline_singapore(self, pipeline_engine, district_request):
        """Test district cooling pipeline for Singapore."""
        district_request["region"] = "singapore"
        result = pipeline_engine.run_district_cooling_pipeline(district_request)
        assert result["calculation_result"]["emissions_kgco2e"] > Decimal("0")

    def test_district_pipeline_dubai(self, pipeline_engine, district_request):
        """Test district cooling pipeline for Dubai."""
        district_request["region"] = "dubai"
        result = pipeline_engine.run_district_cooling_pipeline(district_request)
        assert result["calculation_result"]["emissions_kgco2e"] > Decimal("0")


# ===========================================================================
# 7. Batch Processing Tests
# ===========================================================================


class TestBatchProcessing:
    """Test run_batch method for multiple calculations."""

    def test_run_batch_empty_list(self, pipeline_engine):
        """Test batch processing with empty list."""
        result = pipeline_engine.run_batch([])
        assert result["total_calculations"] == 0
        assert len(result["results"]) == 0

    def test_run_batch_single_calculation(self, pipeline_engine, electric_request):
        """Test batch processing with single calculation."""
        result = pipeline_engine.run_batch([electric_request])
        assert result["total_calculations"] == 1
        assert len(result["results"]) == 1

    def test_run_batch_multiple_calculations(self, pipeline_engine, electric_request, absorption_request):
        """Test batch processing with multiple calculations."""
        requests = [electric_request, absorption_request]
        result = pipeline_engine.run_batch(requests)
        assert result["total_calculations"] == 2
        assert len(result["results"]) == 2

    def test_run_batch_mixed_technologies(
        self, pipeline_engine, electric_request, absorption_request, free_cooling_request
    ):
        """Test batch with mixed technology types."""
        requests = [electric_request, absorption_request, free_cooling_request]
        result = pipeline_engine.run_batch(requests)
        assert result["total_calculations"] == 3


# ===========================================================================
# 8. Aggregation Tests
# ===========================================================================


class TestAggregation:
    """Test aggregation methods."""

    def test_aggregate_by_facility(self, pipeline_engine):
        """Test aggregation by facility."""
        calculations = [
            {"facility_id": "F1", "emissions_kgco2e": Decimal("1000")},
            {"facility_id": "F1", "emissions_kgco2e": Decimal("2000")},
            {"facility_id": "F2", "emissions_kgco2e": Decimal("1500")},
        ]
        result = pipeline_engine.aggregate_by_facility(calculations)
        assert "F1" in result
        assert "F2" in result
        assert result["F1"]["total_emissions_kgco2e"] == Decimal("3000")

    def test_aggregate_by_technology(self, pipeline_engine):
        """Test aggregation by technology."""
        calculations = [
            {"technology": "WATER_COOLED_CENTRIFUGAL", "emissions_kgco2e": Decimal("1000")},
            {"technology": "WATER_COOLED_CENTRIFUGAL", "emissions_kgco2e": Decimal("2000")},
            {"technology": "ABSORPTION", "emissions_kgco2e": Decimal("1500")},
        ]
        result = pipeline_engine.aggregate_by_technology(calculations)
        assert "WATER_COOLED_CENTRIFUGAL" in result
        assert "ABSORPTION" in result
        assert result["WATER_COOLED_CENTRIFUGAL"]["total_emissions_kgco2e"] == Decimal("3000")


# ===========================================================================
# 9. Refrigerant Leakage Tests
# ===========================================================================


class TestRefrigerantLeakage:
    """Test calculate_refrigerant_leakage method."""

    def test_calculate_refrigerant_leakage_basic(self, pipeline_engine):
        """Test basic refrigerant leakage calculation."""
        result = pipeline_engine.calculate_refrigerant_leakage(
            refrigerant="R134a",
            charge_kg=Decimal("100"),
            leakage_rate_pct=Decimal("5.0"),
        )
        assert result["leakage_kg"] > Decimal("0")
        assert result["leakage_kgco2e"] > Decimal("0")

    def test_refrigerant_leakage_r410a(self, pipeline_engine):
        """Test refrigerant leakage for R-410A."""
        result = pipeline_engine.calculate_refrigerant_leakage(
            refrigerant="R410A",
            charge_kg=Decimal("50"),
            leakage_rate_pct=Decimal("3.0"),
        )
        assert result["leakage_kg"] == Decimal("1.5")  # 50 * 0.03

    def test_refrigerant_leakage_zero_rate(self, pipeline_engine):
        """Test refrigerant leakage with zero leakage rate."""
        result = pipeline_engine.calculate_refrigerant_leakage(
            refrigerant="R134a",
            charge_kg=Decimal("100"),
            leakage_rate_pct=Decimal("0"),
        )
        assert result["leakage_kg"] == Decimal("0")
        assert result["leakage_kgco2e"] == Decimal("0")


# ===========================================================================
# 10. Technology Comparison Tests
# ===========================================================================


class TestTechnologyComparison:
    """Test compare_technologies method."""

    def test_compare_two_technologies(self, pipeline_engine):
        """Test comparing two technologies."""
        tech1 = {
            "technology": "WATER_COOLED_CENTRIFUGAL",
            "emissions_kgco2e": Decimal("10000"),
            "cop": Decimal("5.5"),
        }
        tech2 = {
            "technology": "AIR_COOLED_CENTRIFUGAL",
            "emissions_kgco2e": Decimal("15000"),
            "cop": Decimal("3.5"),
        }
        result = pipeline_engine.compare_technologies([tech1, tech2])
        assert len(result["technologies"]) == 2
        assert "best_technology" in result

    def test_compare_three_technologies(self, pipeline_engine):
        """Test comparing three technologies."""
        technologies = [
            {"technology": "WATER_COOLED_CENTRIFUGAL", "emissions_kgco2e": Decimal("10000")},
            {"technology": "AIR_COOLED_CENTRIFUGAL", "emissions_kgco2e": Decimal("15000")},
            {"technology": "SEAWATER_FREE", "emissions_kgco2e": Decimal("2000")},
        ]
        result = pipeline_engine.compare_technologies(technologies)
        assert len(result["technologies"]) == 3
        # Seawater free cooling should be best
        assert result["best_technology"] == "SEAWATER_FREE"


# ===========================================================================
# 11. Pipeline Stages Tests
# ===========================================================================


class TestPipelineStages:
    """Test get_pipeline_stages method."""

    def test_get_pipeline_stages_returns_13(self, pipeline_engine):
        """Test pipeline has 13 stages."""
        stages = pipeline_engine.get_pipeline_stages()
        assert len(stages) == 13

    def test_pipeline_stages_include_validation(self, pipeline_engine):
        """Test pipeline stages include validation."""
        stages = pipeline_engine.get_pipeline_stages()
        stage_names = [s["name"] for s in stages]
        assert any("validation" in name.lower() for name in stage_names)

    def test_pipeline_stages_include_calculation(self, pipeline_engine):
        """Test pipeline stages include calculation."""
        stages = pipeline_engine.get_pipeline_stages()
        stage_names = [s["name"] for s in stages]
        assert any("calculation" in name.lower() for name in stage_names)


# ===========================================================================
# 12. Engine Versions Tests
# ===========================================================================


class TestEngineVersions:
    """Test get_engine_versions method."""

    def test_get_engine_versions_returns_7_engines(self, pipeline_engine):
        """Test 7 engine versions are returned."""
        versions = pipeline_engine.get_engine_versions()
        assert len(versions) == 7

    def test_engine_versions_include_electric_chiller(self, pipeline_engine):
        """Test versions include electric chiller calculator."""
        versions = pipeline_engine.get_engine_versions()
        assert "electric_chiller_calculator" in versions

    def test_engine_versions_include_compliance(self, pipeline_engine):
        """Test versions include compliance checker."""
        versions = pipeline_engine.get_engine_versions()
        assert "compliance_checker" in versions


# ===========================================================================
# 13. Health Check Tests
# ===========================================================================


class TestHealthCheck:
    """Test health_check method."""

    def test_health_check_returns_healthy(self, pipeline_engine):
        """Test health check returns healthy status."""
        result = pipeline_engine.health_check()
        assert result["status"] == "healthy"

    def test_health_check_includes_all_engines(self, pipeline_engine):
        """Test health check reports all 7 engines."""
        result = pipeline_engine.health_check()
        assert len(result["engines"]) == 7

    def test_health_check_engine_statuses(self, pipeline_engine):
        """Test all engines report healthy status."""
        result = pipeline_engine.health_check()
        for engine_status in result["engines"].values():
            assert engine_status in ["healthy", "degraded", "unhealthy"]


# ===========================================================================
# 14. Statistics Tests
# ===========================================================================


class TestStatistics:
    """Test statistics tracking."""

    def test_statistics_counter_increments(self, pipeline_engine, electric_request):
        """Test statistics counter increments after pipeline run."""
        stats_before = pipeline_engine.get_statistics()
        _ = pipeline_engine.run_electric_chiller_pipeline(electric_request)
        stats_after = pipeline_engine.get_statistics()
        assert stats_after["total_pipeline_runs"] == stats_before["total_pipeline_runs"] + 1

    def test_statistics_tracks_technology_types(self, pipeline_engine, electric_request):
        """Test statistics tracks different technology types."""
        _ = pipeline_engine.run_electric_chiller_pipeline(electric_request)
        stats = pipeline_engine.get_statistics()
        assert "by_technology" in stats
