# -*- coding: utf-8 -*-
"""
Unit tests for FugitiveEmissionsPipelineEngine (Engine 7 of 7) - AGENT-MRV-005

Tests the 8-stage orchestration pipeline including validation, source
resolution, component counting, emission calculation, recovery adjustment,
uncertainty quantification, compliance checking, and audit trail generation.

Target: 64 tests, ~1050 lines.

Test classes:
    TestPipelineExecution (15)
    TestPipelineStages (16)
    TestBatchPipeline (8)
    TestWithLDAR (5)
    TestWithUncertainty (5)
    TestWithCompliance (5)
    TestErrors (5)
    TestMetrics (5)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-005 Fugitive Emissions (GL-MRV-SCOPE1-005)
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from greenlang.fugitive_emissions.fugitive_emissions_pipeline import (
    FugitiveEmissionsPipelineEngine,
    PipelineStage,
    GWP_VALUES,
    DEFAULT_EMISSION_FACTORS,
    RECOVERY_DEFAULTS,
    SOURCE_TYPES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Pipeline engine with no external dependencies."""
    return FugitiveEmissionsPipelineEngine()


@pytest.fixture
def equipment_leak_request():
    """Canonical equipment leak calculation request."""
    return {
        "source_type": "EQUIPMENT_LEAK",
        "facility_id": "FAC-001",
        "calculation_method": "AVERAGE_EMISSION_FACTOR",
        "component_count": 100,
        "component_type": "valve",
        "service_type": "gas",
        "operating_hours": 8760,
    }


@pytest.fixture
def coal_mine_request():
    """Coal mine methane calculation request."""
    return {
        "source_type": "COAL_MINE_METHANE",
        "facility_id": "FAC-002",
        "calculation_method": "AVERAGE_EMISSION_FACTOR",
        "component_count": 1,
        "activity_data": 10000,
    }


@pytest.fixture
def wastewater_request():
    """Wastewater treatment calculation request."""
    return {
        "source_type": "WASTEWATER",
        "facility_id": "FAC-003",
        "calculation_method": "AVERAGE_EMISSION_FACTOR",
        "component_count": 1,
        "activity_data": 5000,
    }


@pytest.fixture
def direct_measurement_request():
    """Direct measurement calculation request."""
    return {
        "source_type": "EQUIPMENT_LEAK",
        "facility_id": "FAC-004",
        "calculation_method": "DIRECT_MEASUREMENT",
        "measured_emissions": {"CH4": 500.0, "VOC": 25.0},
    }


@pytest.fixture
def recovery_request():
    """Equipment leak request with gas recovery."""
    return {
        "source_type": "EQUIPMENT_LEAK",
        "facility_id": "FAC-005",
        "calculation_method": "AVERAGE_EMISSION_FACTOR",
        "component_count": 100,
        "component_type": "valve",
        "service_type": "gas",
        "recovery_technology": "vapor_recovery_unit",
    }


@pytest.fixture
def mock_uncertainty_engine():
    """Mock UncertaintyQuantifierEngine."""
    mock = MagicMock()
    mock.quantify_uncertainty.return_value = {
        "method": "monte_carlo",
        "mean_co2e_kg": 5000.0,
        "std_dev_kg": 500.0,
        "confidence_intervals": {"95": {"lower": 4020.0, "upper": 5980.0}},
    }
    return mock


@pytest.fixture
def mock_compliance_engine():
    """Mock ComplianceCheckerEngine."""
    mock = MagicMock()
    mock.check_compliance.return_value = {
        "frameworks_checked": 7,
        "compliant": 5,
        "partial": 2,
        "non_compliant": 0,
        "results": {},
    }
    return mock


# ===========================================================================
# TestPipelineExecution (15 tests)
# ===========================================================================


class TestPipelineExecution:
    """Tests for execute_pipeline end-to-end."""

    def test_basic_execution(self, engine, equipment_leak_request):
        result = engine.execute_pipeline(equipment_leak_request)
        assert result["success"] is True
        assert "pipeline_id" in result

    def test_pipeline_id_format(self, engine, equipment_leak_request):
        result = engine.execute_pipeline(equipment_leak_request)
        assert result["pipeline_id"].startswith("fe_pipe_")

    def test_source_type_echoed(self, engine, equipment_leak_request):
        result = engine.execute_pipeline(equipment_leak_request)
        assert result["source_type"] == "EQUIPMENT_LEAK"

    def test_facility_id_echoed(self, engine, equipment_leak_request):
        result = engine.execute_pipeline(equipment_leak_request)
        assert result["facility_id"] == "FAC-001"

    def test_calculation_method_echoed(self, engine, equipment_leak_request):
        result = engine.execute_pipeline(equipment_leak_request)
        assert result["calculation_method"] == "AVERAGE_EMISSION_FACTOR"

    def test_gwp_source_default_ar6(self, engine, equipment_leak_request):
        result = engine.execute_pipeline(equipment_leak_request)
        assert result["gwp_source"] == "AR6"

    def test_gwp_source_custom(self, engine, equipment_leak_request):
        result = engine.execute_pipeline(
            equipment_leak_request, gwp_source="AR5",
        )
        assert result["gwp_source"] == "AR5"

    def test_calculation_data_present(self, engine, equipment_leak_request):
        result = engine.execute_pipeline(equipment_leak_request)
        assert "calculation_data" in result
        calc_data = result["calculation_data"]
        assert "total_co2e_kg" in calc_data

    def test_audit_trail_present(self, engine, equipment_leak_request):
        result = engine.execute_pipeline(equipment_leak_request)
        assert "audit_trail" in result
        assert len(result["audit_trail"]) >= 1

    def test_provenance_hash_present(self, engine, equipment_leak_request):
        result = engine.execute_pipeline(equipment_leak_request)
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_processing_time(self, engine, equipment_leak_request):
        result = engine.execute_pipeline(equipment_leak_request)
        assert result["processing_time_ms"] > 0

    def test_timestamp_present(self, engine, equipment_leak_request):
        result = engine.execute_pipeline(equipment_leak_request)
        assert "timestamp" in result

    def test_last_completed_stage(self, engine, equipment_leak_request):
        result = engine.execute_pipeline(equipment_leak_request)
        assert result["last_completed_stage"] == "GENERATE_AUDIT"

    def test_stage_results_present(self, engine, equipment_leak_request):
        result = engine.execute_pipeline(equipment_leak_request)
        assert "stage_results" in result
        assert len(result["stage_results"]) >= 1

    def test_coal_mine_execution(self, engine, coal_mine_request):
        result = engine.execute_pipeline(coal_mine_request)
        assert result["success"] is True
        assert result["source_type"] == "COAL_MINE_METHANE"


# ===========================================================================
# TestPipelineStages (16 tests)
# ===========================================================================


class TestPipelineStages:
    """Tests for individual pipeline stages."""

    def test_validate_stage_present(self, engine, equipment_leak_request):
        result = engine.execute_pipeline(equipment_leak_request)
        assert "VALIDATE" in result["stage_results"]

    def test_resolve_source_stage_present(self, engine, equipment_leak_request):
        result = engine.execute_pipeline(equipment_leak_request)
        assert "RESOLVE_SOURCE" in result["stage_results"]

    def test_count_components_stage_present(self, engine, equipment_leak_request):
        result = engine.execute_pipeline(equipment_leak_request)
        assert "COUNT_COMPONENTS" in result["stage_results"]

    def test_calculate_emissions_stage_present(self, engine, equipment_leak_request):
        result = engine.execute_pipeline(equipment_leak_request)
        assert "CALCULATE_EMISSIONS" in result["stage_results"]

    def test_apply_recovery_stage_present(self, engine, equipment_leak_request):
        result = engine.execute_pipeline(equipment_leak_request)
        assert "APPLY_RECOVERY" in result["stage_results"]

    def test_quantify_uncertainty_stage_present(self, engine, equipment_leak_request):
        result = engine.execute_pipeline(equipment_leak_request)
        assert "QUANTIFY_UNCERTAINTY" in result["stage_results"]

    def test_check_compliance_stage_present(self, engine, equipment_leak_request):
        result = engine.execute_pipeline(equipment_leak_request)
        assert "CHECK_COMPLIANCE" in result["stage_results"]

    def test_generate_audit_stage_present(self, engine, equipment_leak_request):
        result = engine.execute_pipeline(equipment_leak_request)
        assert "GENERATE_AUDIT" in result["stage_results"]

    def test_validate_valid_result(self, engine, equipment_leak_request):
        result = engine.execute_pipeline(equipment_leak_request)
        validate = result["stage_results"]["VALIDATE"]
        assert validate["status"] == "valid"

    def test_resolve_source_has_gases(self, engine, equipment_leak_request):
        result = engine.execute_pipeline(equipment_leak_request)
        resolve = result["stage_results"]["RESOLVE_SOURCE"]
        assert "applicable_gases" in resolve

    def test_calculate_has_total(self, engine, equipment_leak_request):
        result = engine.execute_pipeline(equipment_leak_request)
        calc = result["stage_results"]["CALCULATE_EMISSIONS"]
        assert "total_co2e_kg" in calc

    def test_recovery_has_technology(self, engine, equipment_leak_request):
        result = engine.execute_pipeline(equipment_leak_request)
        recovery = result["stage_results"]["APPLY_RECOVERY"]
        assert "recovery_technology" in recovery

    def test_audit_has_provenance(self, engine, equipment_leak_request):
        result = engine.execute_pipeline(equipment_leak_request)
        audit = result["stage_results"]["GENERATE_AUDIT"]
        assert "provenance_hash" in audit

    def test_direct_measurement_method(self, engine, direct_measurement_request):
        result = engine.execute_pipeline(direct_measurement_request)
        calc = result["stage_results"]["CALCULATE_EMISSIONS"]
        assert calc["method_used"] == "DIRECT_MEASUREMENT"

    def test_pipeline_stage_enum_values(self):
        assert PipelineStage.VALIDATE.value == "VALIDATE"
        assert PipelineStage.GENERATE_AUDIT.value == "GENERATE_AUDIT"
        assert len(PipelineStage) == 8

    def test_resolve_source_ef_source(self, engine, equipment_leak_request):
        result = engine.execute_pipeline(equipment_leak_request)
        resolve = result["stage_results"]["RESOLVE_SOURCE"]
        assert "ef_source" in resolve


# ===========================================================================
# TestBatchPipeline (8 tests)
# ===========================================================================


class TestBatchPipeline:
    """Tests for execute_batch_pipeline."""

    def test_basic_batch(self, engine, equipment_leak_request, coal_mine_request):
        result = engine.execute_batch_pipeline(
            [equipment_leak_request, coal_mine_request],
        )
        assert result["total_calculations"] == 2

    def test_batch_all_successful(self, engine, equipment_leak_request):
        result = engine.execute_batch_pipeline(
            [equipment_leak_request, equipment_leak_request],
        )
        assert result["successful"] == 2
        assert result["failed"] == 0

    def test_batch_id_format(self, engine, equipment_leak_request):
        result = engine.execute_batch_pipeline([equipment_leak_request])
        assert result["batch_id"].startswith("fe_batch_")

    def test_batch_total_co2e(self, engine, equipment_leak_request):
        result = engine.execute_batch_pipeline([equipment_leak_request])
        assert result["total_co2e_kg"] >= 0

    def test_batch_total_co2e_tonnes(self, engine, equipment_leak_request):
        result = engine.execute_batch_pipeline([equipment_leak_request])
        assert result["total_co2e_tonnes"] == pytest.approx(
            result["total_co2e_kg"] / 1000.0, rel=0.01,
        )

    def test_batch_results_list(self, engine, equipment_leak_request):
        result = engine.execute_batch_pipeline([equipment_leak_request])
        assert "results" in result
        assert len(result["results"]) == 1

    def test_batch_provenance_hash(self, engine, equipment_leak_request):
        result = engine.execute_batch_pipeline([equipment_leak_request])
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_batch_processing_time(self, engine, equipment_leak_request):
        result = engine.execute_batch_pipeline([equipment_leak_request])
        assert result["processing_time_ms"] > 0


# ===========================================================================
# TestWithLDAR (5 tests)
# ===========================================================================


class TestWithLDAR:
    """Tests for LDAR-related pipeline behavior."""

    def test_screening_method_request(self, engine):
        req = {
            "source_type": "EQUIPMENT_LEAK",
            "facility_id": "FAC-LDAR",
            "calculation_method": "SCREENING_RANGES",
            "component_count": 50,
            "component_type": "valve",
            "service_type": "gas",
        }
        result = engine.execute_pipeline(req)
        assert result["success"] is True

    def test_epa_correlation_method(self, engine):
        req = {
            "source_type": "EQUIPMENT_LEAK",
            "facility_id": "FAC-LDAR",
            "calculation_method": "EPA_CORRELATION",
            "component_count": 50,
            "component_type": "connector",
            "service_type": "gas",
        }
        result = engine.execute_pipeline(req)
        assert result["success"] is True

    def test_unit_specific_correlation(self, engine):
        req = {
            "source_type": "EQUIPMENT_LEAK",
            "facility_id": "FAC-LDAR",
            "calculation_method": "UNIT_SPECIFIC_CORRELATION",
            "component_count": 30,
            "component_type": "pump",
            "service_type": "light_liquid",
        }
        result = engine.execute_pipeline(req)
        assert result["success"] is True

    def test_ldar_with_component_counts_dict(self, engine):
        req = {
            "source_type": "EQUIPMENT_LEAK",
            "facility_id": "FAC-LDAR",
            "calculation_method": "AVERAGE_EMISSION_FACTOR",
            "component_counts": {
                "by_type_and_service": {"valve|gas": 100, "connector|gas": 200},
                "total_active": 300,
            },
        }
        result = engine.execute_pipeline(req)
        assert result["success"] is True

    def test_ldar_zero_components(self, engine):
        req = {
            "source_type": "EQUIPMENT_LEAK",
            "facility_id": "FAC-LDAR",
            "calculation_method": "AVERAGE_EMISSION_FACTOR",
            "component_count": 0,
        }
        result = engine.execute_pipeline(req)
        assert result["success"] is True
        calc_data = result["calculation_data"]
        assert calc_data["total_co2e_kg"] == pytest.approx(0.0, abs=0.01)


# ===========================================================================
# TestWithUncertainty (5 tests)
# ===========================================================================


class TestWithUncertainty:
    """Tests for uncertainty quantification integration."""

    def test_uncertainty_enabled_default(self, engine, equipment_leak_request):
        result = engine.execute_pipeline(equipment_leak_request)
        assert "QUANTIFY_UNCERTAINTY" in result["stage_results"]

    def test_uncertainty_disabled(self, engine, equipment_leak_request):
        result = engine.execute_pipeline(
            equipment_leak_request, enable_uncertainty=False,
        )
        assert "QUANTIFY_UNCERTAINTY" not in result["stage_results"]

    def test_uncertainty_with_mock_engine(
        self, mock_uncertainty_engine, equipment_leak_request,
    ):
        engine = FugitiveEmissionsPipelineEngine(
            uncertainty_engine=mock_uncertainty_engine,
        )
        result = engine.execute_pipeline(equipment_leak_request)
        assert result["success"] is True
        mock_uncertainty_engine.quantify_uncertainty.assert_called_once()

    def test_uncertainty_fallback_analytical(self, engine, equipment_leak_request):
        result = engine.execute_pipeline(equipment_leak_request)
        unc = result["stage_results"]["QUANTIFY_UNCERTAINTY"]
        assert "method" in unc

    def test_uncertainty_confidence_intervals(self, engine, equipment_leak_request):
        result = engine.execute_pipeline(equipment_leak_request)
        unc = result["stage_results"]["QUANTIFY_UNCERTAINTY"]
        assert "confidence_intervals" in unc


# ===========================================================================
# TestWithCompliance (5 tests)
# ===========================================================================


class TestWithCompliance:
    """Tests for compliance checking integration."""

    def test_compliance_enabled_default(self, engine, equipment_leak_request):
        result = engine.execute_pipeline(equipment_leak_request)
        assert "CHECK_COMPLIANCE" in result["stage_results"]

    def test_compliance_disabled(self, engine, equipment_leak_request):
        result = engine.execute_pipeline(
            equipment_leak_request, enable_compliance=False,
        )
        assert "CHECK_COMPLIANCE" not in result["stage_results"]

    def test_compliance_with_mock_engine(
        self, mock_compliance_engine, equipment_leak_request,
    ):
        engine = FugitiveEmissionsPipelineEngine(
            compliance_checker=mock_compliance_engine,
        )
        result = engine.execute_pipeline(equipment_leak_request)
        assert result["success"] is True
        mock_compliance_engine.check_compliance.assert_called_once()

    def test_compliance_fallback(self, engine, equipment_leak_request):
        result = engine.execute_pipeline(equipment_leak_request)
        comp = result["stage_results"]["CHECK_COMPLIANCE"]
        assert "frameworks_checked" in comp or "note" in comp

    def test_compliance_frameworks_param(
        self, mock_compliance_engine, equipment_leak_request,
    ):
        engine = FugitiveEmissionsPipelineEngine(
            compliance_checker=mock_compliance_engine,
        )
        engine.execute_pipeline(
            equipment_leak_request,
            compliance_frameworks=["GHG_PROTOCOL"],
        )
        call_args = mock_compliance_engine.check_compliance.call_args
        assert call_args[1].get("frameworks") == ["GHG_PROTOCOL"] or \
            call_args.kwargs.get("frameworks") == ["GHG_PROTOCOL"]


# ===========================================================================
# TestErrors (5 tests)
# ===========================================================================


class TestErrors:
    """Tests for error handling and validation failures."""

    def test_invalid_source_type(self, engine):
        req = {
            "source_type": "INVALID_SOURCE",
            "facility_id": "FAC-ERR",
        }
        result = engine.execute_pipeline(req)
        assert result["success"] is False

    def test_empty_source_type(self, engine):
        req = {
            "source_type": "",
            "facility_id": "FAC-ERR",
        }
        result = engine.execute_pipeline(req)
        assert result["success"] is False

    def test_invalid_gwp_source(self, engine):
        req = {
            "source_type": "EQUIPMENT_LEAK",
            "facility_id": "FAC-ERR",
        }
        result = engine.execute_pipeline(req, gwp_source="INVALID_GWP")
        assert result["success"] is False

    def test_error_audit_trail(self, engine):
        req = {"source_type": "", "facility_id": "FAC-ERR"}
        result = engine.execute_pipeline(req)
        assert len(result["audit_trail"]) >= 1
        last_entry = result["audit_trail"][-1]
        assert last_entry["status"] == "failed"

    def test_missing_request_fields(self, engine):
        result = engine.execute_pipeline({})
        # Should either fail validation or succeed with defaults
        assert "pipeline_id" in result


# ===========================================================================
# TestMetrics (5 tests)
# ===========================================================================


class TestMetrics:
    """Tests for pipeline statistics and metrics."""

    def test_initial_statistics(self):
        engine = FugitiveEmissionsPipelineEngine()
        stats = engine.get_statistics()
        assert stats["total_pipelines"] == 0
        assert stats["successful_pipelines"] == 0
        assert stats["failed_pipelines"] == 0

    def test_successful_pipeline_increments(self, engine, equipment_leak_request):
        engine.execute_pipeline(equipment_leak_request)
        stats = engine.get_statistics()
        assert stats["total_pipelines"] >= 1
        assert stats["successful_pipelines"] >= 1

    def test_failed_pipeline_increments(self, engine):
        engine.execute_pipeline({"source_type": ""})
        stats = engine.get_statistics()
        assert stats["failed_pipelines"] >= 1

    def test_batch_counter_increments(self, engine, equipment_leak_request):
        engine.execute_batch_pipeline([equipment_leak_request])
        stats = engine.get_statistics()
        assert stats["total_batches"] >= 1

    def test_stage_error_counter(self, engine):
        engine.execute_pipeline({"source_type": "INVALID_SRC"})
        stats = engine.get_statistics()
        assert stats["total_stage_errors"] >= 1
