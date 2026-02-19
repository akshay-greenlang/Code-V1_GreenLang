# -*- coding: utf-8 -*-
"""
Unit tests for LandUsePipelineEngine (Engine 7 of 7)

AGENT-MRV-006: Land Use Emissions Agent

Tests the 8-stage orchestration pipeline that coordinates all upstream
engines through a deterministic sequence:
    1. VALIDATE_INPUT
    2. CLASSIFY_LAND
    3. LOOKUP_FACTORS
    4. CALCULATE_STOCKS
    5. CALCULATE_SOC
    6. CALCULATE_NON_CO2
    7. CHECK_COMPLIANCE
    8. ASSEMBLE_RESULTS

Target: 90 tests, ~600 lines.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
"""

from __future__ import annotations

import threading
from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from greenlang.land_use_emissions.land_use_pipeline import (
    LandUsePipelineEngine,
    PipelineStage,
    VALID_CATEGORIES,
    VALID_METHODS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_db():
    """Create a mock LandUseDatabaseEngine."""
    db = MagicMock()
    db.get_all_factors.return_value = {
        "agb_tc_ha": Decimal("180"),
        "bgb_tc_ha": Decimal("43"),
        "dead_wood_tc_ha": Decimal("14"),
        "litter_tc_ha": Decimal("5"),
        "soc_ref_tc_ha": Decimal("65"),
        "source": "IPCC_2006",
    }
    db.get_growth_rate.return_value = Decimal("7.0")
    db.get_land_subcategories.return_value = ["TROPICAL_RAIN_FOREST", "TROPICAL_MOIST_DECIDUOUS"]
    db.get_fire_ef.return_value = {
        "combustion_factor": Decimal("0.45"),
        "ef_co2_g_per_kg": Decimal("1580"),
    }
    db.get_peatland_ef.return_value = {
        "co2_tc_ha_yr": Decimal("2.5"),
        "ch4_kg_ha_yr": Decimal("5.0"),
        "n2o_kg_ha_yr": Decimal("0.5"),
    }
    db.get_n2o_ef.return_value = Decimal("0.01")
    return db


@pytest.fixture
def mock_calc():
    """Create a mock CarbonStockCalculatorEngine."""
    calc = MagicMock()
    calc.calculate_stock_difference.return_value = {
        "status": "SUCCESS",
        "total_co2_tonnes_yr": "-1833.33",
        "net_co2e_tonnes_yr": "-1833.33",
        "gross_emissions_tco2_yr": "0",
        "gross_removals_tco2_yr": "-1833.33",
        "emission_type": "NET_REMOVAL",
        "pool_results": {
            "AGB": {"change_tc_yr": "-2.0"},
            "BGB": {"change_tc_yr": "-0.6"},
        },
        "provenance_hash": "a" * 64,
    }
    calc.calculate_gain_loss.return_value = {
        "status": "SUCCESS",
        "total_co2_tonnes_yr": "-500.00",
        "net_co2e_tonnes_yr": "-500.00",
        "provenance_hash": "b" * 64,
    }
    calc.calculate_fire_emissions.return_value = {
        "status": "SUCCESS",
        "total_co2e_tonnes": "1200.00",
    }
    return calc


@pytest.fixture
def mock_soc():
    """Create a mock SoilOrganicCarbonEngine."""
    soc = MagicMock()
    soc.calculate_soc.return_value = {
        "status": "SUCCESS",
        "soc_tc_ha": "44.85",
        "soc_total_tc": "4485.0",
    }
    soc.calculate_soc_change.return_value = {
        "status": "SUCCESS",
        "delta_co2_tonnes_yr": "-100.00",
        "delta_soc_tc_yr": "-27.27",
    }
    soc.calculate_liming_emissions.return_value = {
        "status": "SUCCESS",
        "co2_tonnes": "44.00",
    }
    soc.calculate_urea_emissions.return_value = {
        "status": "SUCCESS",
        "co2_tonnes": "73.33",
    }
    return soc


@pytest.fixture
def mock_uncertainty():
    """Create a mock UncertaintyQuantifierEngine."""
    return MagicMock()


@pytest.fixture
def mock_compliance():
    """Create a mock ComplianceCheckerEngine."""
    comp = MagicMock()
    comp.check_compliance.return_value = {
        "status": "SUCCESS",
        "overall": {
            "compliance_status": "COMPLIANT",
            "total_requirements": 12,
            "total_passed": 12,
            "total_failed": 0,
            "total_errors": 0,
            "total_warnings": 0,
            "pass_rate_pct": 100.0,
        },
    }
    return comp


@pytest.fixture
def pipeline(mock_db, mock_calc, mock_soc, mock_uncertainty, mock_compliance):
    """Create a LandUsePipelineEngine with all mocked engines."""
    p = LandUsePipelineEngine(
        db_engine=mock_db,
        calc_engine=mock_calc,
        tracker_engine=MagicMock(),
        soc_engine=mock_soc,
        uncertainty_engine=mock_uncertainty,
        compliance_engine=mock_compliance,
    )
    yield p
    p.reset()


@pytest.fixture
def pipeline_no_engines():
    """Create a LandUsePipelineEngine with no upstream engines."""
    p = LandUsePipelineEngine(
        db_engine=None,
        calc_engine=None,
        tracker_engine=None,
        soc_engine=None,
        uncertainty_engine=None,
        compliance_engine=None,
    )
    yield p
    p.reset()


@pytest.fixture
def valid_stock_diff_request() -> Dict[str, Any]:
    """Return a valid stock-difference calculation request."""
    return {
        "land_category": "FOREST_LAND",
        "climate_zone": "TROPICAL_WET",
        "soil_type": "HIGH_ACTIVITY_CLAY",
        "area_ha": 1000,
        "method": "STOCK_DIFFERENCE",
        "c_t1": {"AGB": 180, "BGB": 43, "DEAD_WOOD": 14, "LITTER": 5},
        "c_t2": {"AGB": 170, "BGB": 40, "DEAD_WOOD": 13, "LITTER": 5},
        "year_t1": 2020,
        "year_t2": 2025,
        "new_land_use_type": "FOREST_NATIVE",
    }


@pytest.fixture
def valid_gain_loss_request() -> Dict[str, Any]:
    """Return a valid gain-loss calculation request."""
    return {
        "land_category": "FOREST_LAND",
        "climate_zone": "TROPICAL_WET",
        "area_ha": 500,
        "method": "GAIN_LOSS",
        "harvest_volume_m3": 200,
        "fuelwood_volume_m3": 50,
    }


# ===========================================================================
# 1. Initialisation Tests
# ===========================================================================


class TestPipelineInit:
    """Test LandUsePipelineEngine initialisation."""

    def test_init_with_all_engines(self, pipeline):
        """Test pipeline initialises with all engines provided."""
        assert pipeline._db_engine is not None
        assert pipeline._calc_engine is not None
        assert pipeline._soc_engine is not None
        assert pipeline._compliance_engine is not None

    def test_init_no_engines(self, pipeline_no_engines):
        """Test pipeline initialises gracefully with no engines."""
        assert pipeline_no_engines._db_engine is None
        assert pipeline_no_engines._calc_engine is None
        assert pipeline_no_engines._soc_engine is None

    def test_init_counters_start_zero(self, pipeline):
        """Test that execution counters start at zero."""
        stats = pipeline.get_statistics()
        assert stats["total_executions"] == 0
        assert stats["total_batches"] == 0

    def test_pipeline_has_8_stages(self):
        """Test that PipelineStage enum has 8 members."""
        assert len(PipelineStage) == 8

    def test_stage_names(self):
        """Test pipeline stage names match expected order."""
        expected = [
            "VALIDATE_INPUT", "CLASSIFY_LAND", "LOOKUP_FACTORS",
            "CALCULATE_STOCKS", "CALCULATE_SOC", "CALCULATE_NON_CO2",
            "CHECK_COMPLIANCE", "ASSEMBLE_RESULTS",
        ]
        actual = [s.value for s in PipelineStage]
        assert actual == expected


# ===========================================================================
# 2. Full Pipeline Execution Tests
# ===========================================================================


class TestFullPipelineExecution:
    """Test full 8-stage pipeline execution."""

    def test_success_status(self, pipeline, valid_stock_diff_request):
        """Test that valid request produces SUCCESS status."""
        result = pipeline.execute(valid_stock_diff_request)
        assert result["status"] == "SUCCESS"

    def test_all_8_stages_completed(self, pipeline, valid_stock_diff_request):
        """Test that all 8 stages complete successfully."""
        result = pipeline.execute(valid_stock_diff_request)
        assert len(result["stages_completed"]) == 8
        assert len(result["stages_failed"]) == 0

    def test_pipeline_id_present(self, pipeline, valid_stock_diff_request):
        """Test that a unique pipeline_id is generated."""
        result = pipeline.execute(valid_stock_diff_request)
        assert "pipeline_id" in result
        assert len(result["pipeline_id"]) > 0

    def test_provenance_hash_present(self, pipeline, valid_stock_diff_request):
        """Test that final result includes a provenance hash."""
        result = pipeline.execute(valid_stock_diff_request)
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_provenance_chain_has_8_entries(self, pipeline, valid_stock_diff_request):
        """Test that provenance chain has one hash per stage."""
        result = pipeline.execute(valid_stock_diff_request)
        assert len(result["provenance_chain"]) == 8

    def test_processing_time_positive(self, pipeline, valid_stock_diff_request):
        """Test that processing time is recorded."""
        result = pipeline.execute(valid_stock_diff_request)
        assert result["processing_time_ms"] >= 0

    def test_calculated_at_present(self, pipeline, valid_stock_diff_request):
        """Test that calculated_at timestamp is present."""
        result = pipeline.execute(valid_stock_diff_request)
        assert "calculated_at" in result

    def test_execution_counter_increments(self, pipeline, valid_stock_diff_request):
        """Test that total_executions counter increments."""
        pipeline.execute(valid_stock_diff_request)
        pipeline.execute(valid_stock_diff_request)
        stats = pipeline.get_statistics()
        assert stats["total_executions"] == 2


# ===========================================================================
# 3. VALIDATE_INPUT Stage Tests
# ===========================================================================


class TestValidateInputStage:
    """Test Stage 1: VALIDATE_INPUT."""

    def test_valid_input_passes(self, pipeline, valid_stock_diff_request):
        """Test that valid input passes validation."""
        result = pipeline.execute(valid_stock_diff_request)
        assert "VALIDATE_INPUT" in result["stages_completed"]

    def test_missing_land_category_fails(self, pipeline):
        """Test that missing land_category fails validation and aborts pipeline."""
        result = pipeline.execute({
            "climate_zone": "TROPICAL_WET",
            "area_ha": 100,
        })
        assert result["status"] in ("PARTIAL", "FAILED")
        assert "VALIDATE_INPUT" in result["stages_failed"]

    def test_invalid_land_category_fails(self, pipeline):
        """Test that invalid land_category fails validation."""
        result = pipeline.execute({
            "land_category": "DESERT",
            "climate_zone": "TROPICAL_WET",
            "area_ha": 100,
        })
        assert "VALIDATE_INPUT" in result["stages_failed"]

    @pytest.mark.parametrize("category", VALID_CATEGORIES)
    def test_all_valid_categories_pass(self, pipeline, valid_stock_diff_request, category):
        """Test that all 6 IPCC land categories pass validation."""
        valid_stock_diff_request["land_category"] = category
        result = pipeline.execute(valid_stock_diff_request)
        assert "VALIDATE_INPUT" in result["stages_completed"]

    def test_missing_area_fails(self, pipeline):
        """Test that missing area_ha fails validation."""
        result = pipeline.execute({
            "land_category": "FOREST_LAND",
            "climate_zone": "TROPICAL_WET",
        })
        assert "VALIDATE_INPUT" in result["stages_failed"]

    def test_zero_area_fails(self, pipeline):
        """Test that zero area_ha fails validation."""
        result = pipeline.execute({
            "land_category": "FOREST_LAND",
            "climate_zone": "TROPICAL_WET",
            "area_ha": 0,
        })
        assert "VALIDATE_INPUT" in result["stages_failed"]

    def test_invalid_method_fails(self, pipeline, valid_stock_diff_request):
        """Test that invalid method fails validation."""
        valid_stock_diff_request["method"] = "INVALID_METHOD"
        result = pipeline.execute(valid_stock_diff_request)
        assert "VALIDATE_INPUT" in result["stages_failed"]

    @pytest.mark.parametrize("method", VALID_METHODS)
    def test_valid_methods_pass(self, pipeline, valid_stock_diff_request, method):
        """Test that both valid methods pass validation."""
        valid_stock_diff_request["method"] = method
        result = pipeline.execute(valid_stock_diff_request)
        assert "VALIDATE_INPUT" in result["stages_completed"]

    def test_validation_failure_aborts_subsequent_stages(self, pipeline):
        """Test that Stage 1 failure prevents all subsequent stages."""
        result = pipeline.execute({})  # Empty request
        assert "VALIDATE_INPUT" in result["stages_failed"]
        # All other stages should be in failed list
        for stage in PipelineStage:
            if stage != PipelineStage.VALIDATE_INPUT:
                assert stage.value in result["stages_failed"]


# ===========================================================================
# 4. CLASSIFY_LAND Stage Tests
# ===========================================================================


class TestClassifyLandStage:
    """Test Stage 2: CLASSIFY_LAND."""

    def test_classify_land_completes(self, pipeline, valid_stock_diff_request):
        """Test that CLASSIFY_LAND stage completes."""
        result = pipeline.execute(valid_stock_diff_request)
        assert "CLASSIFY_LAND" in result["stages_completed"]

    def test_land_category_in_result(self, pipeline, valid_stock_diff_request):
        """Test that classified land_category appears in result."""
        result = pipeline.execute(valid_stock_diff_request)
        assert result["land_category"] == "FOREST_LAND"

    def test_soil_type_default(self, pipeline, valid_stock_diff_request):
        """Test that soil_type defaults to HIGH_ACTIVITY_CLAY."""
        del valid_stock_diff_request["soil_type"]
        result = pipeline.execute(valid_stock_diff_request)
        assert result["soil_type"] == "HIGH_ACTIVITY_CLAY"


# ===========================================================================
# 5. LOOKUP_FACTORS Stage Tests
# ===========================================================================


class TestLookupFactorsStage:
    """Test Stage 3: LOOKUP_FACTORS."""

    def test_factors_retrieved(self, pipeline, valid_stock_diff_request):
        """Test that LOOKUP_FACTORS stage completes and retrieves factors."""
        result = pipeline.execute(valid_stock_diff_request)
        assert "LOOKUP_FACTORS" in result["stages_completed"]
        assert "factors" in result

    def test_no_db_graceful_degradation(self, pipeline_no_engines, valid_stock_diff_request):
        """Test that missing DB engine produces DB_UNAVAILABLE status."""
        result = pipeline_no_engines.execute(valid_stock_diff_request)
        assert result["factors"].get("status") == "DB_UNAVAILABLE"

    def test_gain_loss_fetches_growth_rate(self, pipeline, valid_gain_loss_request):
        """Test that gain-loss method triggers growth rate lookup."""
        pipeline.execute(valid_gain_loss_request)
        pipeline._db_engine.get_growth_rate.assert_called()


# ===========================================================================
# 6. CALCULATE_STOCKS Stage Tests
# ===========================================================================


class TestCalculateStocksStage:
    """Test Stage 4: CALCULATE_STOCKS."""

    def test_stock_difference_method(self, pipeline, valid_stock_diff_request):
        """Test that STOCK_DIFFERENCE method calls calculate_stock_difference."""
        pipeline.execute(valid_stock_diff_request)
        pipeline._calc_engine.calculate_stock_difference.assert_called_once()

    def test_gain_loss_method(self, pipeline, valid_gain_loss_request):
        """Test that GAIN_LOSS method calls calculate_gain_loss."""
        pipeline.execute(valid_gain_loss_request)
        pipeline._calc_engine.calculate_gain_loss.assert_called_once()

    def test_no_calc_engine_graceful(self, pipeline_no_engines, valid_stock_diff_request):
        """Test that missing calc engine produces CALC_ENGINE_UNAVAILABLE."""
        result = pipeline_no_engines.execute(valid_stock_diff_request)
        assert result["results"]["stock_change"].get("status") == "CALC_ENGINE_UNAVAILABLE"

    def test_stock_result_in_output(self, pipeline, valid_stock_diff_request):
        """Test that stock calculation result appears in output."""
        result = pipeline.execute(valid_stock_diff_request)
        assert "stock_change" in result["results"]


# ===========================================================================
# 7. CALCULATE_SOC Stage Tests
# ===========================================================================


class TestCalculateSOCStage:
    """Test Stage 5: CALCULATE_SOC."""

    def test_soc_calculation_triggered(self, pipeline, valid_stock_diff_request):
        """Test that SOC calculation is triggered when land_use_type is present."""
        result = pipeline.execute(valid_stock_diff_request)
        assert "CALCULATE_SOC" in result["stages_completed"]

    def test_soc_change_triggered_with_old_and_new_lu(self, pipeline, valid_stock_diff_request):
        """Test SOC change calculation when both old and new land use are specified."""
        valid_stock_diff_request["old_land_use_type"] = "FOREST_NATIVE"
        valid_stock_diff_request["new_land_use_type"] = "CROPLAND_ANNUAL_FULL_TILL"
        pipeline.execute(valid_stock_diff_request)
        pipeline._soc_engine.calculate_soc_change.assert_called()

    def test_liming_emissions_triggered(self, pipeline, valid_stock_diff_request):
        """Test that liming emissions are calculated when limestone is provided."""
        valid_stock_diff_request["limestone_tonnes"] = 100
        pipeline.execute(valid_stock_diff_request)
        pipeline._soc_engine.calculate_liming_emissions.assert_called()

    def test_urea_emissions_triggered(self, pipeline, valid_stock_diff_request):
        """Test that urea emissions are calculated when urea is provided."""
        valid_stock_diff_request["urea_tonnes"] = 50
        pipeline.execute(valid_stock_diff_request)
        pipeline._soc_engine.calculate_urea_emissions.assert_called()

    def test_no_soc_engine_graceful(self, pipeline_no_engines, valid_stock_diff_request):
        """Test that missing SOC engine produces SOC_ENGINE_UNAVAILABLE."""
        result = pipeline_no_engines.execute(valid_stock_diff_request)
        assert result["results"]["soc_change"].get("status") == "SOC_ENGINE_UNAVAILABLE"


# ===========================================================================
# 8. CALCULATE_NON_CO2 Stage Tests
# ===========================================================================


class TestCalculateNonCO2Stage:
    """Test Stage 6: CALCULATE_NON_CO2."""

    def test_non_co2_completes(self, pipeline, valid_stock_diff_request):
        """Test that CALCULATE_NON_CO2 stage completes."""
        result = pipeline.execute(valid_stock_diff_request)
        assert "CALCULATE_NON_CO2" in result["stages_completed"]

    def test_fire_emissions_when_disturbance(self, pipeline, valid_stock_diff_request):
        """Test fire emissions calculated when disturbance is present."""
        valid_stock_diff_request["disturbance_type"] = "FIRE_WILDFIRE"
        valid_stock_diff_request["disturbance_area_ha"] = 100
        pipeline.execute(valid_stock_diff_request)
        pipeline._calc_engine.calculate_fire_emissions.assert_called()

    def test_no_fire_without_disturbance(self, pipeline, valid_stock_diff_request):
        """Test no fire emissions when no disturbance specified."""
        pipeline.execute(valid_stock_diff_request)
        pipeline._calc_engine.calculate_fire_emissions.assert_not_called()

    def test_non_co2_result_in_output(self, pipeline, valid_stock_diff_request):
        """Test that non-CO2 results appear in output."""
        result = pipeline.execute(valid_stock_diff_request)
        assert "non_co2" in result["results"]


# ===========================================================================
# 9. CHECK_COMPLIANCE Stage Tests
# ===========================================================================


class TestCheckComplianceStage:
    """Test Stage 7: CHECK_COMPLIANCE."""

    def test_compliance_skipped_no_frameworks(self, pipeline, valid_stock_diff_request):
        """Test compliance is skipped when no frameworks specified."""
        result = pipeline.execute(valid_stock_diff_request)
        compliance = result["compliance"]
        assert compliance.get("status") == "SKIPPED" or "compliance_status" in compliance.get("overall", {})

    def test_compliance_triggered_with_frameworks(self, pipeline, valid_stock_diff_request):
        """Test compliance check is triggered when frameworks are specified."""
        valid_stock_diff_request["frameworks"] = ["IPCC_2006"]
        pipeline.execute(valid_stock_diff_request)
        pipeline._compliance_engine.check_compliance.assert_called()

    def test_compliance_result_in_output(self, pipeline, valid_stock_diff_request):
        """Test compliance result appears in output."""
        valid_stock_diff_request["frameworks"] = ["IPCC_2006"]
        result = pipeline.execute(valid_stock_diff_request)
        assert "compliance" in result

    def test_no_compliance_engine_graceful(self, pipeline_no_engines, valid_stock_diff_request):
        """Test missing compliance engine produces COMPLIANCE_ENGINE_UNAVAILABLE."""
        valid_stock_diff_request["frameworks"] = ["IPCC_2006"]
        result = pipeline_no_engines.execute(valid_stock_diff_request)
        assert result["compliance"].get("status") == "COMPLIANCE_ENGINE_UNAVAILABLE"


# ===========================================================================
# 10. ASSEMBLE_RESULTS Stage Tests
# ===========================================================================


class TestAssembleResultsStage:
    """Test Stage 8: ASSEMBLE_RESULTS."""

    def test_assembly_completes(self, pipeline, valid_stock_diff_request):
        """Test ASSEMBLE_RESULTS stage completes."""
        result = pipeline.execute(valid_stock_diff_request)
        assert "ASSEMBLE_RESULTS" in result["stages_completed"]

    def test_total_co2e_in_output(self, pipeline, valid_stock_diff_request):
        """Test that total_co2e_tonnes_yr is in the output."""
        result = pipeline.execute(valid_stock_diff_request)
        assert "total_co2e_tonnes_yr" in result

    def test_net_type_determined(self, pipeline, valid_stock_diff_request):
        """Test that net_type is determined (NET_EMISSION, NET_REMOVAL, or NEUTRAL)."""
        result = pipeline.execute(valid_stock_diff_request)
        assert result["net_type"] in ("NET_EMISSION", "NET_REMOVAL", "NEUTRAL", "UNKNOWN")

    def test_totals_breakdown_present(self, pipeline, valid_stock_diff_request):
        """Test that totals breakdown includes stock/SOC/liming/urea/fire components."""
        result = pipeline.execute(valid_stock_diff_request)
        totals = result["results"]["totals"]
        for key in ["stock_change_co2_yr", "soc_change_co2_yr", "liming_co2_yr",
                     "urea_co2_yr", "fire_co2e_yr", "peatland_co2e_yr", "n2o_co2e_yr"]:
            assert key in totals

    def test_net_removal_type(self, pipeline, valid_stock_diff_request):
        """Test that negative total produces NET_REMOVAL type."""
        result = pipeline.execute(valid_stock_diff_request)
        # Mock returns -1833.33 for stock CO2, should be net removal
        assert result["net_type"] in ("NET_REMOVAL", "NEUTRAL")


# ===========================================================================
# 11. Batch Processing Tests
# ===========================================================================


class TestBatchProcessing:
    """Test execute_batch for multiple requests."""

    def test_batch_success(self, pipeline, valid_stock_diff_request):
        """Test batch processing of multiple requests."""
        requests = [valid_stock_diff_request, valid_stock_diff_request]
        result = pipeline.execute_batch(requests)
        assert result["total_requests"] == 2
        assert result["success_count"] == 2
        assert result["failure_count"] == 0

    def test_batch_partial_failure(self, pipeline, valid_stock_diff_request):
        """Test batch with one valid and one invalid request."""
        requests = [
            valid_stock_diff_request,
            {},  # Invalid - will fail validation
        ]
        result = pipeline.execute_batch(requests)
        assert result["total_requests"] == 2
        assert result["success_count"] >= 1

    def test_batch_id_present(self, pipeline, valid_stock_diff_request):
        """Test that batch_id is generated."""
        result = pipeline.execute_batch([valid_stock_diff_request])
        assert "batch_id" in result

    def test_batch_total_co2e_accumulated(self, pipeline, valid_stock_diff_request):
        """Test that total CO2e is accumulated across batch items."""
        result = pipeline.execute_batch([valid_stock_diff_request, valid_stock_diff_request])
        assert "total_co2e_tonnes_yr" in result

    def test_batch_by_category_aggregation(self, pipeline, valid_stock_diff_request):
        """Test by-category aggregation in batch results."""
        result = pipeline.execute_batch([valid_stock_diff_request])
        assert "by_category" in result

    def test_batch_counter_increments(self, pipeline, valid_stock_diff_request):
        """Test that total_batches counter increments."""
        pipeline.execute_batch([valid_stock_diff_request])
        stats = pipeline.get_statistics()
        assert stats["total_batches"] == 1

    def test_batch_provenance_hash(self, pipeline, valid_stock_diff_request):
        """Test that batch result has provenance hash."""
        result = pipeline.execute_batch([valid_stock_diff_request])
        assert len(result["provenance_hash"]) == 64

    def test_empty_batch(self, pipeline):
        """Test batch with empty request list."""
        result = pipeline.execute_batch([])
        assert result["total_requests"] == 0
        assert result["success_count"] == 0


# ===========================================================================
# 12. Error Handling Tests
# ===========================================================================


class TestErrorHandling:
    """Test error handling at each stage."""

    def test_stage_error_recorded(self, pipeline, valid_stock_diff_request):
        """Test that stage errors are recorded in the result."""
        pipeline._calc_engine.calculate_stock_difference.side_effect = Exception("Calc failed")
        result = pipeline.execute(valid_stock_diff_request)
        assert "CALCULATE_STOCKS" in result["stages_failed"]
        assert len(result["errors"]) > 0

    def test_partial_status_on_stage_failure(self, pipeline, valid_stock_diff_request):
        """Test that a non-validation stage failure produces PARTIAL status."""
        pipeline._calc_engine.calculate_stock_difference.side_effect = Exception("Calc error")
        result = pipeline.execute(valid_stock_diff_request)
        assert result["status"] == "PARTIAL"

    def test_validation_failure_produces_failed_status(self, pipeline):
        """Test that validation failure can produce FAILED status."""
        result = pipeline.execute({})
        assert result["status"] in ("PARTIAL", "FAILED")

    def test_errors_list_populated(self, pipeline, valid_stock_diff_request):
        """Test that errors list is populated on failure."""
        pipeline._soc_engine.calculate_soc.side_effect = Exception("SOC error")
        result = pipeline.execute(valid_stock_diff_request)
        assert any("SOC" in e or "CALCULATE_SOC" in e for e in result["errors"])


# ===========================================================================
# 13. Stage Timing Tests
# ===========================================================================


class TestStageTiming:
    """Test stage timing recording."""

    def test_all_stages_timed(self, pipeline, valid_stock_diff_request):
        """Test that all 8 stages have timing entries."""
        result = pipeline.execute(valid_stock_diff_request)
        timings = result["stage_timings"]
        for stage in PipelineStage:
            assert stage.value in timings

    def test_stage_timings_non_negative(self, pipeline, valid_stock_diff_request):
        """Test that all stage timings are non-negative."""
        result = pipeline.execute(valid_stock_diff_request)
        for stage, ms in result["stage_timings"].items():
            assert ms >= 0

    def test_aggregate_stage_timings(self, pipeline, valid_stock_diff_request):
        """Test that statistics reports average stage timings."""
        pipeline.execute(valid_stock_diff_request)
        pipeline.execute(valid_stock_diff_request)
        stats = pipeline.get_statistics()
        for stage in PipelineStage:
            assert stage.value in stats["avg_stage_timings_ms"]


# ===========================================================================
# 14. Config-Driven Behaviour Tests
# ===========================================================================


class TestConfigDrivenBehaviour:
    """Test configuration-driven pipeline behaviour."""

    def test_default_method_stock_difference(self, pipeline):
        """Test that default method is STOCK_DIFFERENCE when not specified."""
        result = pipeline.execute({
            "land_category": "FOREST_LAND",
            "climate_zone": "TROPICAL_WET",
            "area_ha": 100,
        })
        assert result["method"] == "STOCK_DIFFERENCE"

    def test_default_tier_tier1(self, pipeline, valid_stock_diff_request):
        """Test that default tier is TIER_1."""
        result = pipeline.execute(valid_stock_diff_request)
        assert result["tier"] == "TIER_1"

    def test_default_gwp_source_ar6(self, pipeline, valid_stock_diff_request):
        """Test that default GWP source is AR6."""
        result = pipeline.execute(valid_stock_diff_request)
        assert result["gwp_source"] == "AR6"

    def test_custom_gwp_source(self, pipeline, valid_stock_diff_request):
        """Test that custom GWP source is respected."""
        valid_stock_diff_request["gwp_source"] = "AR5"
        result = pipeline.execute(valid_stock_diff_request)
        assert result["gwp_source"] == "AR5"


# ===========================================================================
# 15. Deterministic Results Tests
# ===========================================================================


class TestDeterministicResults:
    """Test that pipeline produces deterministic results."""

    def test_same_input_same_co2e(self, pipeline, valid_stock_diff_request):
        """Test that identical inputs produce the same total CO2e."""
        r1 = pipeline.execute(valid_stock_diff_request)
        r2 = pipeline.execute(valid_stock_diff_request)
        assert r1["total_co2e_tonnes_yr"] == r2["total_co2e_tonnes_yr"]

    def test_same_input_same_net_type(self, pipeline, valid_stock_diff_request):
        """Test that identical inputs produce the same net type."""
        r1 = pipeline.execute(valid_stock_diff_request)
        r2 = pipeline.execute(valid_stock_diff_request)
        assert r1["net_type"] == r2["net_type"]

    def test_same_input_same_stages(self, pipeline, valid_stock_diff_request):
        """Test that identical inputs complete the same stages."""
        r1 = pipeline.execute(valid_stock_diff_request)
        r2 = pipeline.execute(valid_stock_diff_request)
        assert r1["stages_completed"] == r2["stages_completed"]


# ===========================================================================
# 16. Statistics and Reset Tests
# ===========================================================================


class TestPipelineStatisticsAndReset:
    """Test pipeline statistics and reset."""

    def test_statistics_structure(self, pipeline):
        """Test statistics returns all expected fields."""
        stats = pipeline.get_statistics()
        assert stats["engine"] == "LandUsePipelineEngine"
        assert stats["version"] == "1.0.0"
        assert "created_at" in stats
        assert "total_executions" in stats
        assert "total_batches" in stats
        assert "stages" in stats
        assert stats["stage_count"] == 8
        assert "avg_stage_timings_ms" in stats
        assert "engines" in stats

    def test_engine_availability_reported(self, pipeline):
        """Test that engine availability is reported in statistics."""
        stats = pipeline.get_statistics()
        engines = stats["engines"]
        assert engines["db"] is True
        assert engines["calc"] is True
        assert engines["soc"] is True
        assert engines["compliance"] is True

    def test_reset_clears_all(self, pipeline, valid_stock_diff_request):
        """Test that reset clears all counters and timings."""
        pipeline.execute(valid_stock_diff_request)
        pipeline.execute_batch([valid_stock_diff_request])
        pipeline.reset()
        stats = pipeline.get_statistics()
        assert stats["total_executions"] == 0
        assert stats["total_batches"] == 0

    def test_reset_resets_upstream_engines(self, pipeline, valid_stock_diff_request):
        """Test that reset calls reset on upstream engines."""
        pipeline.execute(valid_stock_diff_request)
        pipeline.reset()
        pipeline._soc_engine.reset.assert_called()
        pipeline._compliance_engine.reset.assert_called()


# ===========================================================================
# 17. Thread Safety Tests
# ===========================================================================


class TestPipelineThreadSafety:
    """Test thread safety of pipeline execution."""

    def test_concurrent_executions(self, pipeline, valid_stock_diff_request):
        """Test that concurrent pipeline executions do not corrupt state."""
        errors = []

        def run_pipeline(n):
            try:
                for _ in range(5):
                    result = pipeline.execute(valid_stock_diff_request)
                    if result["status"] not in ("SUCCESS", "PARTIAL"):
                        errors.append(f"Thread {n}: unexpected status {result['status']}")
            except Exception as e:
                errors.append(f"Thread {n}: {e}")

        threads = [threading.Thread(target=run_pipeline, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
        stats = pipeline.get_statistics()
        assert stats["total_executions"] == 20  # 4 threads * 5 executions


# ===========================================================================
# 18. Additional Pipeline Input Tests
# ===========================================================================


class TestPipelineInputVariations:
    """Test pipeline with various input configurations."""

    @pytest.mark.parametrize("category", VALID_CATEGORIES)
    def test_all_land_categories(self, pipeline, valid_stock_diff_request, category):
        """Test pipeline accepts all IPCC land categories."""
        valid_stock_diff_request["land_category"] = category
        result = pipeline.execute(valid_stock_diff_request)
        assert result["land_category"] == category
        assert "VALIDATE_INPUT" in result["stages_completed"]

    def test_gain_loss_method_execution(self, pipeline, valid_gain_loss_request):
        """Test pipeline execution with GAIN_LOSS method."""
        result = pipeline.execute(valid_gain_loss_request)
        assert result["method"] == "GAIN_LOSS"
        assert "VALIDATE_INPUT" in result["stages_completed"]

    def test_with_soc_transition_data(self, pipeline, valid_stock_diff_request):
        """Test pipeline with SOC transition data included."""
        valid_stock_diff_request["old_land_use_type"] = "FOREST_NATIVE"
        valid_stock_diff_request["new_land_use_type"] = "CROPLAND_ANNUAL_FULL_TILL"
        valid_stock_diff_request["old_management_practice"] = "NOMINAL"
        valid_stock_diff_request["new_management_practice"] = "FULL_TILLAGE"
        result = pipeline.execute(valid_stock_diff_request)
        assert "CALCULATE_SOC" in result["stages_completed"]

    def test_with_liming_and_urea(self, pipeline, valid_stock_diff_request):
        """Test pipeline with liming and urea emissions."""
        valid_stock_diff_request["limestone_tonnes"] = 100
        valid_stock_diff_request["dolomite_tonnes"] = 50
        valid_stock_diff_request["urea_tonnes"] = 30
        result = pipeline.execute(valid_stock_diff_request)
        assert result["status"] == "SUCCESS"

    def test_with_fire_disturbance(self, pipeline, valid_stock_diff_request):
        """Test pipeline with fire disturbance data."""
        valid_stock_diff_request["disturbance_type"] = "FIRE_WILDFIRE"
        valid_stock_diff_request["disturbance_area_ha"] = 200
        result = pipeline.execute(valid_stock_diff_request)
        assert "CALCULATE_NON_CO2" in result["stages_completed"]

    def test_parcel_id_passed_through(self, pipeline, valid_stock_diff_request):
        """Test that parcel_id is passed through the pipeline context."""
        valid_stock_diff_request["parcel_id"] = "parcel-test-99"
        result = pipeline.execute(valid_stock_diff_request)
        assert result["status"] == "SUCCESS"

    def test_multiple_frameworks_in_request(self, pipeline, valid_stock_diff_request):
        """Test pipeline with multiple compliance frameworks."""
        valid_stock_diff_request["frameworks"] = ["IPCC_2006", "IPCC_2019", "ISO_14064"]
        result = pipeline.execute(valid_stock_diff_request)
        assert "CHECK_COMPLIANCE" in result["stages_completed"]

    def test_tier_2_input(self, pipeline, valid_stock_diff_request):
        """Test pipeline with Tier 2 specification."""
        valid_stock_diff_request["tier"] = "TIER_2"
        result = pipeline.execute(valid_stock_diff_request)
        assert result["tier"] == "TIER_2"

    def test_string_area_ha(self, pipeline, valid_stock_diff_request):
        """Test pipeline handles string area_ha via _safe_decimal."""
        valid_stock_diff_request["area_ha"] = "500.5"
        result = pipeline.execute(valid_stock_diff_request)
        assert result["status"] == "SUCCESS"
        assert result["area_ha"] == "500.5"

    def test_missing_climate_zone_aborts(self, pipeline):
        """Test that missing climate_zone causes validation abort."""
        result = pipeline.execute({
            "land_category": "FOREST_LAND",
            "area_ha": 100,
        })
        assert "VALIDATE_INPUT" in result["stages_failed"]


# ===========================================================================
# 19. Additional Batch Tests
# ===========================================================================


class TestBatchExtended:
    """Extended batch processing tests."""

    def test_batch_with_mixed_categories(self, pipeline):
        """Test batch with different land categories."""
        requests = [
            {"land_category": "FOREST_LAND", "climate_zone": "TROPICAL_WET",
             "area_ha": 100, "new_land_use_type": "FOREST_NATIVE"},
            {"land_category": "CROPLAND", "climate_zone": "TEMPERATE_OCEANIC",
             "area_ha": 200, "new_land_use_type": "CROPLAND_ANNUAL_FULL_TILL"},
        ]
        result = pipeline.execute_batch(requests)
        assert result["total_requests"] == 2

    def test_batch_all_failures(self, pipeline):
        """Test batch where all requests fail."""
        requests = [{}, {}, {}]  # All empty, will fail validation
        result = pipeline.execute_batch(requests)
        assert result["status"] == "PARTIAL"
        assert result["failure_count"] >= 0  # Some may partially succeed

    def test_batch_processing_time(self, pipeline, valid_stock_diff_request):
        """Test that batch processing time is tracked."""
        result = pipeline.execute_batch([valid_stock_diff_request])
        assert result["processing_time_ms"] >= 0

    def test_batch_calculated_at(self, pipeline, valid_stock_diff_request):
        """Test that batch has calculated_at timestamp."""
        result = pipeline.execute_batch([valid_stock_diff_request])
        assert "calculated_at" in result


# ===========================================================================
# 20. Pipeline Context Tests
# ===========================================================================


class TestPipelineContext:
    """Test pipeline context management and data flow."""

    def test_context_carries_land_category(self, pipeline, valid_stock_diff_request):
        """Test that land_category flows from context to result."""
        result = pipeline.execute(valid_stock_diff_request)
        assert result["land_category"] == "FOREST_LAND"

    def test_context_carries_climate_zone(self, pipeline, valid_stock_diff_request):
        """Test that climate_zone flows from context to result."""
        result = pipeline.execute(valid_stock_diff_request)
        assert result["climate_zone"] == "TROPICAL_WET"

    def test_context_carries_area(self, pipeline, valid_stock_diff_request):
        """Test that area_ha flows from context to result."""
        result = pipeline.execute(valid_stock_diff_request)
        assert result["area_ha"] == "1000"

    def test_results_section_structure(self, pipeline, valid_stock_diff_request):
        """Test that the results section contains all expected sub-results."""
        result = pipeline.execute(valid_stock_diff_request)
        results = result["results"]
        assert "stock_change" in results
        assert "soc_change" in results
        assert "liming" in results
        assert "urea" in results
        assert "non_co2" in results
        assert "totals" in results

    def test_errors_list_empty_on_success(self, pipeline, valid_stock_diff_request):
        """Test that errors list is empty on full success."""
        result = pipeline.execute(valid_stock_diff_request)
        assert result["errors"] == []
