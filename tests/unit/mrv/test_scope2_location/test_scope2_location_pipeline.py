# -*- coding: utf-8 -*-
"""
Unit tests for Scope2LocationPipelineEngine (Engine 7 of 7)

AGENT-MRV-009: Scope 2 Location-Based Emissions Agent

Tests the 8-stage orchestrated pipeline: input validation, factor
resolution, T&D losses, electricity calculation, non-electric
calculation, GWP conversion, compliance checks, and result assembly.
Also covers batch pipeline, facility pipeline, total Scope 2,
uncertainty integration, and pipeline control.

Target: 40 tests, ~500 lines.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

try:
    from greenlang.scope2_location.scope2_location_pipeline import (
        Scope2LocationPipelineEngine,
        PIPELINE_STAGES,
        GWP_TABLE,
        VALID_ENERGY_TYPES,
        VALID_GWP_SOURCES,
    )
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not PIPELINE_AVAILABLE, reason="Pipeline engine not available"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline():
    """Create a Scope2LocationPipelineEngine with no upstream engines."""
    return Scope2LocationPipelineEngine()


@pytest.fixture
def electricity_request() -> Dict[str, Any]:
    """Build a valid electricity calculation request."""
    return {
        "facility_id": "FAC-001",
        "tenant_id": "tenant-001",
        "energy_type": "electricity",
        "consumption_value": Decimal("5000"),
        "consumption_unit": "mwh",
        "country_code": "US",
        "gwp_source": "AR5",
        "include_td_losses": True,
        "include_compliance": False,
    }


@pytest.fixture
def steam_request() -> Dict[str, Any]:
    """Build a valid steam calculation request."""
    return {
        "facility_id": "FAC-002",
        "tenant_id": "tenant-001",
        "energy_type": "steam",
        "consumption_value": Decimal("1200"),
        "consumption_unit": "gj",
        "country_code": "US",
        "gwp_source": "AR5",
        "include_td_losses": False,
        "steam_type": "natural_gas",
    }


# ===========================================================================
# 1. TestPipelineInit
# ===========================================================================


@_SKIP
class TestPipelineInit:
    """Tests for pipeline initialization."""

    def test_init_no_engines(self):
        """Pipeline can be created without upstream engines."""
        p = Scope2LocationPipelineEngine()
        assert p is not None

    def test_init_with_mock_engines(self):
        """Pipeline accepts mock engine arguments."""
        mock_grid = MagicMock()
        mock_elec = MagicMock()
        p = Scope2LocationPipelineEngine(
            grid_factor_db=mock_grid,
            electricity_engine=mock_elec,
        )
        assert p._grid_factor_db is mock_grid
        assert p._electricity is mock_elec

    def test_pipeline_stages_constant(self):
        """PIPELINE_STAGES has 8 stages."""
        assert len(PIPELINE_STAGES) == 8

    def test_gwp_table_ar5(self):
        """GWP_TABLE AR5 has correct CO2 GWP of 1."""
        assert GWP_TABLE["AR5"]["co2"] == Decimal("1")
        assert GWP_TABLE["AR5"]["ch4"] == Decimal("28")
        assert GWP_TABLE["AR5"]["n2o"] == Decimal("265")


# ===========================================================================
# 2. TestValidateInput
# ===========================================================================


@_SKIP
class TestValidateInput:
    """Tests for stage_validate_input."""

    def test_valid_input(self, pipeline, electricity_request):
        """Valid input passes validation."""
        result = pipeline.stage_validate_input(electricity_request)
        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_missing_energy_type(self, pipeline):
        """Missing energy_type fails validation."""
        result = pipeline.stage_validate_input({
            "consumption_value": Decimal("100"),
            "consumption_unit": "mwh",
        })
        assert result["valid"] is False
        assert any("energy_type" in e for e in result["errors"])

    def test_invalid_energy_type(self, pipeline):
        """Invalid energy_type fails validation."""
        result = pipeline.stage_validate_input({
            "energy_type": "nuclear",
            "consumption_value": Decimal("100"),
        })
        assert result["valid"] is False

    def test_negative_consumption(self, pipeline):
        """Negative consumption fails validation."""
        result = pipeline.stage_validate_input({
            "energy_type": "electricity",
            "consumption_value": -100,
        })
        assert result["valid"] is False

    def test_invalid_gwp_source(self, pipeline):
        """Invalid gwp_source fails validation."""
        result = pipeline.stage_validate_input({
            "energy_type": "electricity",
            "consumption_value": Decimal("100"),
            "gwp_source": "INVALID",
        })
        assert result["valid"] is False

    def test_zero_consumption_warning(self, pipeline):
        """Zero consumption triggers warning but passes validation."""
        result = pipeline.stage_validate_input({
            "energy_type": "electricity",
            "consumption_value": 0,
        })
        assert result["valid"] is True
        assert any("zero" in w.lower() for w in result.get("warnings", []))


# ===========================================================================
# 3. TestResolveFactors
# ===========================================================================


@_SKIP
class TestResolveFactors:
    """Tests for stage_resolve_factors."""

    def test_resolve_factors_fallback(self, pipeline):
        """Without grid_factor_db, fallback factors are returned."""
        factors = pipeline.stage_resolve_factors("US")
        assert "co2_kg_per_mwh" in factors
        assert "ch4_kg_per_mwh" in factors
        assert "n2o_kg_per_mwh" in factors
        assert factors["co2_kg_per_mwh"] > Decimal("0")

    def test_resolve_factors_gb(self, pipeline):
        """GB country code returns valid fallback factors."""
        factors = pipeline.stage_resolve_factors("GB")
        assert factors["country_code"] == "GB"

    def test_resolve_factors_eu(self, pipeline):
        """EU country code returns valid fallback factors."""
        factors = pipeline.stage_resolve_factors("DE")
        assert "co2_kg_per_mwh" in factors


# ===========================================================================
# 4. TestTDLosses
# ===========================================================================


@_SKIP
class TestTDLosses:
    """Tests for stage_apply_td_losses."""

    def test_td_losses_default(self, pipeline):
        """T&D losses use world average fallback when no engine."""
        result = pipeline.stage_apply_td_losses(
            Decimal("1000"), "US"
        )
        assert "td_loss_pct" in result
        assert result["gross_consumption_mwh"] > Decimal("1000")

    def test_td_losses_custom(self, pipeline):
        """Custom T&D loss percentage is applied."""
        result = pipeline.stage_apply_td_losses(
            Decimal("1000"), "US", custom_td=Decimal("0.10")
        )
        assert result["td_loss_pct"] == Decimal("0.10")
        expected_gross = Decimal("1000") * Decimal("1.10")
        assert abs(result["gross_consumption_mwh"] - expected_gross) < Decimal("0.01")


# ===========================================================================
# 5. TestElectricityCalc
# ===========================================================================


@_SKIP
class TestElectricityCalc:
    """Tests for stage_calculate_electricity."""

    def test_electricity_calculation(self, pipeline):
        """Electricity calculation produces valid results."""
        result = pipeline.stage_calculate_electricity(
            consumption_mwh=Decimal("5000"),
            co2_ef=Decimal("436.0"),
            ch4_ef=Decimal("0.040"),
            n2o_ef=Decimal("0.006"),
            gwp_source="AR5",
            td_loss_pct=Decimal("0"),
        )
        assert result["energy_type"] == "electricity"
        assert result["total_co2e_kg"] > Decimal("0")
        assert "gas_breakdown" in result
        assert len(result["gas_breakdown"]) == 3

    def test_electricity_with_td_losses(self, pipeline):
        """T&D losses increase gross consumption."""
        result_no_td = pipeline.stage_calculate_electricity(
            Decimal("1000"), Decimal("400"), Decimal("0.03"),
            Decimal("0.005"), "AR5", Decimal("0"),
        )
        result_with_td = pipeline.stage_calculate_electricity(
            Decimal("1000"), Decimal("400"), Decimal("0.03"),
            Decimal("0.005"), "AR5", Decimal("0.05"),
        )
        assert result_with_td["total_co2e_kg"] > result_no_td["total_co2e_kg"]

    def test_electricity_per_gas_breakdown(self, pipeline):
        """Gas breakdown includes CO2, CH4, N2O entries."""
        result = pipeline.stage_calculate_electricity(
            Decimal("1000"), Decimal("400"), Decimal("0.03"),
            Decimal("0.005"), "AR5", Decimal("0"),
        )
        gases = {g["gas"] for g in result["gas_breakdown"]}
        assert "co2" in gases
        assert "ch4" in gases
        assert "n2o" in gases

    def test_electricity_gwp_ar6(self, pipeline):
        """AR6 GWP produces different result than AR5."""
        result_ar5 = pipeline.stage_calculate_electricity(
            Decimal("1000"), Decimal("400"), Decimal("0.03"),
            Decimal("0.005"), "AR5", Decimal("0"),
        )
        result_ar6 = pipeline.stage_calculate_electricity(
            Decimal("1000"), Decimal("400"), Decimal("0.03"),
            Decimal("0.005"), "AR6", Decimal("0"),
        )
        # CH4 and N2O GWPs differ between AR5 and AR6
        assert result_ar5["total_co2e_kg"] != result_ar6["total_co2e_kg"]


# ===========================================================================
# 6. TestNonElectric
# ===========================================================================


@_SKIP
class TestNonElectric:
    """Tests for stage_calculate_non_electric (steam, heating, cooling)."""

    def test_steam_calculation(self, pipeline):
        """Steam calculation returns valid CO2e."""
        result = pipeline.stage_calculate_non_electric([{
            "energy_type": "steam",
            "consumption_gj": Decimal("1200"),
            "sub_type": "natural_gas",
            "country_code": "US",
        }])
        assert result["combined_co2e_kg"] > Decimal("0")

    def test_heating_calculation(self, pipeline):
        """Heating calculation returns valid CO2e."""
        result = pipeline.stage_calculate_non_electric([{
            "energy_type": "heating",
            "consumption_gj": Decimal("500"),
            "sub_type": "district",
            "country_code": "US",
        }])
        assert result["combined_co2e_kg"] > Decimal("0")

    def test_cooling_calculation(self, pipeline):
        """Cooling calculation returns valid CO2e."""
        result = pipeline.stage_calculate_non_electric([{
            "energy_type": "cooling",
            "consumption_gj": Decimal("300"),
            "sub_type": "absorption",
            "country_code": "US",
        }])
        assert result["combined_co2e_kg"] > Decimal("0")


# ===========================================================================
# 7. TestFullPipeline
# ===========================================================================


@_SKIP
class TestFullPipeline:
    """Tests for run_pipeline end-to-end."""

    def test_run_pipeline_electricity(self, pipeline, electricity_request):
        """Full pipeline for electricity returns expected fields."""
        result = pipeline.run_pipeline(electricity_request)
        assert "calculation_id" in result
        assert result["energy_type"] == "electricity"
        assert result["total_co2e_kg"] > Decimal("0")
        assert result["total_co2e_tonnes"] > Decimal("0")
        assert "provenance_hash" in result
        assert "gas_breakdown" in result

    def test_run_pipeline_steam(self, pipeline, steam_request):
        """Full pipeline for steam returns expected fields."""
        result = pipeline.run_pipeline(steam_request)
        assert result["energy_type"] == "steam"
        assert result["total_co2e_kg"] > Decimal("0")

    def test_run_pipeline_invalid_raises(self, pipeline):
        """Invalid input raises ValueError from pipeline."""
        with pytest.raises(ValueError):
            pipeline.run_pipeline({
                "energy_type": "invalid_type",
                "consumption_value": Decimal("100"),
            })


# ===========================================================================
# 8. TestBatchPipeline
# ===========================================================================


@_SKIP
class TestBatchPipeline:
    """Tests for run_batch_pipeline."""

    def test_batch_pipeline_success(self, pipeline, electricity_request):
        """Batch pipeline processes multiple requests."""
        batch = {
            "batch_id": "batch-001",
            "requests": [electricity_request, electricity_request],
        }
        result = pipeline.run_batch_pipeline(batch)
        assert result["total_requests"] == 2
        assert result["successful"] == 2
        assert result["failed"] == 0
        assert len(result["results"]) == 2

    def test_batch_pipeline_with_errors(self, pipeline, electricity_request):
        """Batch pipeline captures errors for invalid requests."""
        bad_request = {"energy_type": "invalid"}
        batch = {
            "batch_id": "batch-002",
            "requests": [electricity_request, bad_request],
        }
        result = pipeline.run_batch_pipeline(batch)
        assert result["successful"] >= 1
        assert result["failed"] >= 1


# ===========================================================================
# 9. TestFacilityPipeline
# ===========================================================================


@_SKIP
class TestFacilityPipeline:
    """Tests for run_facility_pipeline."""

    def test_facility_pipeline(self, pipeline):
        """Facility pipeline calculates for one energy type."""
        energy_data = [{
            "energy_type": "electricity",
            "consumption_value": Decimal("3000"),
            "consumption_unit": "mwh",
            "country_code": "US",
        }]
        result = pipeline.run_facility_pipeline("FAC-001", energy_data)
        assert result["facility_id"] == "FAC-001"
        assert result["energy_types_calculated"] == 1
        assert result["total_co2e_tonnes"] > Decimal("0")


# ===========================================================================
# 10. TestTotalScope2
# ===========================================================================


@_SKIP
class TestTotalScope2:
    """Tests for calculate_total_scope2."""

    def test_total_scope2_electricity_only(self, pipeline):
        """Total Scope 2 with electricity only."""
        result = pipeline.calculate_total_scope2(
            facility_id="FAC-001",
            electricity_mwh=Decimal("5000"),
            country_code="US",
        )
        assert "electricity" in result["results"]
        assert result["total_scope2_co2e_tonnes"] > Decimal("0")

    def test_total_scope2_multiple_types(self, pipeline):
        """Total Scope 2 with electricity and steam."""
        result = pipeline.calculate_total_scope2(
            facility_id="FAC-001",
            electricity_mwh=Decimal("5000"),
            steam_gj=Decimal("1000"),
            country_code="US",
        )
        assert "electricity" in result["results"]
        assert "steam" in result["results"]
        assert len(result["energy_types_calculated"]) == 2


# ===========================================================================
# 11. TestUncertainty
# ===========================================================================


@_SKIP
class TestUncertainty:
    """Tests for run_with_uncertainty."""

    def test_run_with_uncertainty(self, pipeline, electricity_request):
        """Pipeline with uncertainty returns result and uncertainty dict."""
        result = pipeline.run_with_uncertainty(
            electricity_request, mc_iterations=500,
        )
        assert "result" in result
        assert result["result"]["total_co2e_kg"] > Decimal("0")
        # uncertainty may be None if no engine is available
        assert "uncertainty" in result


# ===========================================================================
# 12. TestPipelineControl
# ===========================================================================


@_SKIP
class TestPipelineControl:
    """Tests for get_pipeline_stages, get_pipeline_status, reset_pipeline."""

    def test_get_pipeline_stages(self, pipeline):
        """get_pipeline_stages returns list of 8 stage names."""
        stages = pipeline.get_pipeline_stages()
        assert isinstance(stages, list)
        assert len(stages) == 8
        assert "validate_input" in stages
        assert "assemble_results" in stages

    def test_get_pipeline_status(self, pipeline):
        """get_pipeline_status returns dict with pipeline_runs."""
        status = pipeline.get_pipeline_status()
        assert "pipeline_runs" in status
        assert "engines" in status

    def test_reset_pipeline(self, pipeline, electricity_request):
        """reset_pipeline zeroes counters."""
        pipeline.run_pipeline(electricity_request)
        assert pipeline._pipeline_runs > 0
        pipeline.reset_pipeline()
        assert pipeline._pipeline_runs == 0


# ===========================================================================
# 13. TestStatistics
# ===========================================================================


@_SKIP
class TestStatistics:
    """Tests for get_statistics."""

    def test_get_statistics(self, pipeline):
        """get_statistics returns dict with expected keys."""
        stats = pipeline.get_statistics()
        assert "pipeline_runs" in stats
        assert "total_co2e_processed_tonnes" in stats
        assert "engines_available" in stats
        assert "stages_count" in stats
        assert stats["stages_count"] == 8

    def test_statistics_after_run(self, pipeline, electricity_request):
        """Statistics update after a pipeline run."""
        pipeline.run_pipeline(electricity_request)
        stats = pipeline.get_statistics()
        assert stats["pipeline_runs"] >= 1
