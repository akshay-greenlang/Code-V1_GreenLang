"""
Unit tests for CapitalGoodsPipelineEngine.

Tests full 10-stage pipeline orchestration, stage execution,
error handling, and result export functionality.
"""

import pytest
from datetime import date
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
import json
import csv
import io

from greenlang.mrv.capital_goods.engines.capital_goods_pipeline import (
    CapitalGoodsPipelineEngine,
    PipelineRequest,
    PipelineResult,
    PipelineStage,
    StageStatus,
    ExportFormat,
)


class TestCapitalGoodsPipelineEngineSingleton:
    """Test singleton pattern."""

    def test_singleton_same_instance(self):
        """Test that multiple calls return same instance."""
        engine1 = CapitalGoodsPipelineEngine()
        engine2 = CapitalGoodsPipelineEngine()
        assert engine1 is engine2

    def test_singleton_with_reset(self):
        """Test singleton reset for testing."""
        engine1 = CapitalGoodsPipelineEngine()
        CapitalGoodsPipelineEngine._instance = None
        engine2 = CapitalGoodsPipelineEngine()
        assert engine1 is not engine2


class TestExecute:
    """Test execute() full pipeline method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        CapitalGoodsPipelineEngine._instance = None
        return CapitalGoodsPipelineEngine()

    @pytest.fixture
    def sample_request(self):
        """Create sample pipeline request."""
        return PipelineRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[
                {
                    "asset_id": f"A{i:03d}",
                    "asset_type": "Server",
                    "purchase_value": 10000.00,
                    "purchase_date": "2024-06-01",
                }
                for i in range(5)
            ],
            calculation_methods=["spend_based", "supplier_specific"],
            frameworks=["ghg_protocol"],
        )

    def test_execute_full_pipeline_success(self, engine, sample_request):
        """Test successful execution of full 10-stage pipeline."""
        result = engine.execute(sample_request)

        assert isinstance(result, PipelineResult)
        assert result.status in [StageStatus.COMPLETED, StageStatus.COMPLETED_WITH_WARNINGS]
        assert len(result.stage_results) == 10  # All 10 stages
        assert result.final_hash is not None

    def test_execute_all_stages_executed(self, engine, sample_request):
        """Test all 10 stages are executed in order."""
        result = engine.execute(sample_request)

        expected_stages = [
            PipelineStage.VALIDATION,
            PipelineStage.DATA_ENRICHMENT,
            PipelineStage.SPEND_BASED_CALC,
            PipelineStage.AVERAGE_DATA_CALC,
            PipelineStage.SUPPLIER_SPECIFIC_CALC,
            PipelineStage.HYBRID_AGGREGATION,
            PipelineStage.DOUBLE_COUNT_PREVENTION,
            PipelineStage.UNCERTAINTY_QUANTIFICATION,
            PipelineStage.COMPLIANCE_CHECK,
            PipelineStage.REPORTING,
        ]

        executed_stages = [stage_result["stage"] for stage_result in result.stage_results]

        assert executed_stages == expected_stages

    def test_execute_stage_timing_recorded(self, engine, sample_request):
        """Test stage execution timing is recorded."""
        result = engine.execute(sample_request)

        for stage_result in result.stage_results:
            assert "start_time" in stage_result
            assert "end_time" in stage_result
            assert "duration_ms" in stage_result
            assert stage_result["duration_ms"] >= 0

    def test_execute_with_errors_continues(self, engine, sample_request):
        """Test pipeline continues on non-critical errors."""
        # Mock a stage to fail non-critically
        with patch.object(engine, '_execute_validation_stage') as mock_validation:
            mock_validation.return_value = {
                "stage": PipelineStage.VALIDATION,
                "status": StageStatus.COMPLETED_WITH_WARNINGS,
                "warnings": ["Non-critical validation warning"],
                "data": {},
            }

            result = engine.execute(sample_request)

            # Pipeline should continue and complete
            assert result.status in [StageStatus.COMPLETED, StageStatus.COMPLETED_WITH_WARNINGS]
            assert len(result.stage_results) == 10


class TestExecuteBatch:
    """Test execute_batch() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        CapitalGoodsPipelineEngine._instance = None
        return CapitalGoodsPipelineEngine()

    def test_execute_batch_multiple_requests(self, engine):
        """Test batch execution of multiple pipeline requests."""
        requests = [
            PipelineRequest(
                organization_id=f"ORG{i:03d}",
                reporting_period_start=date(2024, 1, 1),
                reporting_period_end=date(2024, 12, 31),
                assets=[
                    {
                        "asset_id": "A001",
                        "asset_type": "Equipment",
                        "purchase_value": 5000.00,
                    }
                ],
                calculation_methods=["spend_based"],
                frameworks=["ghg_protocol"],
            )
            for i in range(3)
        ]

        results = engine.execute_batch(requests)

        assert len(results) == 3
        assert all(isinstance(r, PipelineResult) for r in results)

    def test_execute_batch_empty_list(self, engine):
        """Test batch execution with empty list."""
        results = engine.execute_batch([])
        assert results == []

    def test_execute_batch_parallel_processing(self, engine):
        """Test batch uses parallel processing."""
        requests = [
            PipelineRequest(
                organization_id=f"ORG{i:03d}",
                reporting_period_start=date(2024, 1, 1),
                reporting_period_end=date(2024, 12, 31),
                assets=[{"asset_id": "A001", "asset_type": "Equipment", "purchase_value": 5000.00}],
                calculation_methods=["spend_based"],
                frameworks=["ghg_protocol"],
            )
            for i in range(10)
        ]

        with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
            mock_executor.return_value.__enter__.return_value.map.return_value = [
                Mock(status=StageStatus.COMPLETED) for _ in range(10)
            ]
            results = engine.execute_batch(requests)
            assert len(results) == 10


class TestExecuteStage:
    """Test execute_stage() individual stage execution."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        CapitalGoodsPipelineEngine._instance = None
        return CapitalGoodsPipelineEngine()

    def test_execute_validation_stage(self, engine):
        """Test validation stage execution."""
        request = PipelineRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[{"asset_id": "A001", "asset_type": "Server", "purchase_value": 10000.00}],
            calculation_methods=["spend_based"],
            frameworks=["ghg_protocol"],
        )

        stage_result = engine.execute_stage(PipelineStage.VALIDATION, request)

        assert stage_result["stage"] == PipelineStage.VALIDATION
        assert stage_result["status"] in [StageStatus.COMPLETED, StageStatus.COMPLETED_WITH_WARNINGS]

    def test_execute_data_enrichment_stage(self, engine):
        """Test data enrichment stage execution."""
        request = PipelineRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[{"asset_id": "A001", "asset_type": "Server", "purchase_value": 10000.00}],
            calculation_methods=["spend_based"],
            frameworks=["ghg_protocol"],
        )

        stage_result = engine.execute_stage(PipelineStage.DATA_ENRICHMENT, request)

        assert stage_result["stage"] == PipelineStage.DATA_ENRICHMENT
        assert "data" in stage_result

    def test_execute_spend_based_calc_stage(self, engine):
        """Test spend-based calculation stage execution."""
        request = PipelineRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[{"asset_id": "A001", "asset_type": "Server", "purchase_value": 10000.00}],
            calculation_methods=["spend_based"],
            frameworks=["ghg_protocol"],
        )

        stage_result = engine.execute_stage(PipelineStage.SPEND_BASED_CALC, request)

        assert stage_result["stage"] == PipelineStage.SPEND_BASED_CALC
        assert "data" in stage_result

    def test_execute_average_data_calc_stage(self, engine):
        """Test average-data calculation stage execution."""
        request = PipelineRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[{"asset_id": "A001", "asset_type": "Server", "purchase_value": 10000.00}],
            calculation_methods=["average_data"],
            frameworks=["ghg_protocol"],
        )

        stage_result = engine.execute_stage(PipelineStage.AVERAGE_DATA_CALC, request)

        assert stage_result["stage"] == PipelineStage.AVERAGE_DATA_CALC

    def test_execute_supplier_specific_calc_stage(self, engine):
        """Test supplier-specific calculation stage execution."""
        request = PipelineRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[
                {
                    "asset_id": "A001",
                    "asset_type": "Server",
                    "purchase_value": 10000.00,
                    "supplier_id": "SUP001",
                }
            ],
            calculation_methods=["supplier_specific"],
            frameworks=["ghg_protocol"],
        )

        stage_result = engine.execute_stage(PipelineStage.SUPPLIER_SPECIFIC_CALC, request)

        assert stage_result["stage"] == PipelineStage.SUPPLIER_SPECIFIC_CALC

    def test_execute_hybrid_aggregation_stage(self, engine):
        """Test hybrid aggregation stage execution."""
        request = PipelineRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[{"asset_id": "A001", "asset_type": "Server", "purchase_value": 10000.00}],
            calculation_methods=["spend_based", "average_data"],
            frameworks=["ghg_protocol"],
        )

        # Execute prior stages to populate data
        engine.execute_stage(PipelineStage.SPEND_BASED_CALC, request)
        stage_result = engine.execute_stage(PipelineStage.HYBRID_AGGREGATION, request)

        assert stage_result["stage"] == PipelineStage.HYBRID_AGGREGATION

    def test_execute_double_count_prevention_stage(self, engine):
        """Test double-counting prevention stage execution."""
        request = PipelineRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[{"asset_id": "A001", "asset_type": "Server", "purchase_value": 10000.00}],
            calculation_methods=["spend_based"],
            frameworks=["ghg_protocol"],
        )

        stage_result = engine.execute_stage(PipelineStage.DOUBLE_COUNT_PREVENTION, request)

        assert stage_result["stage"] == PipelineStage.DOUBLE_COUNT_PREVENTION

    def test_execute_uncertainty_quantification_stage(self, engine):
        """Test uncertainty quantification stage execution."""
        request = PipelineRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[{"asset_id": "A001", "asset_type": "Server", "purchase_value": 10000.00}],
            calculation_methods=["spend_based"],
            frameworks=["ghg_protocol"],
        )

        stage_result = engine.execute_stage(PipelineStage.UNCERTAINTY_QUANTIFICATION, request)

        assert stage_result["stage"] == PipelineStage.UNCERTAINTY_QUANTIFICATION

    def test_execute_compliance_check_stage(self, engine):
        """Test compliance check stage execution."""
        request = PipelineRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[{"asset_id": "A001", "asset_type": "Server", "purchase_value": 10000.00}],
            calculation_methods=["spend_based"],
            frameworks=["ghg_protocol"],
        )

        stage_result = engine.execute_stage(PipelineStage.COMPLIANCE_CHECK, request)

        assert stage_result["stage"] == PipelineStage.COMPLIANCE_CHECK

    def test_execute_reporting_stage(self, engine):
        """Test reporting stage execution."""
        request = PipelineRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[{"asset_id": "A001", "asset_type": "Server", "purchase_value": 10000.00}],
            calculation_methods=["spend_based"],
            frameworks=["ghg_protocol"],
        )

        stage_result = engine.execute_stage(PipelineStage.REPORTING, request)

        assert stage_result["stage"] == PipelineStage.REPORTING


class TestValidateRequest:
    """Test validate_request() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        CapitalGoodsPipelineEngine._instance = None
        return CapitalGoodsPipelineEngine()

    def test_validate_valid_request(self, engine):
        """Test validation passes for valid request."""
        request = PipelineRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[{"asset_id": "A001", "asset_type": "Server", "purchase_value": 10000.00}],
            calculation_methods=["spend_based"],
            frameworks=["ghg_protocol"],
        )

        is_valid, errors = engine.validate_request(request)

        assert is_valid is True
        assert errors == []

    def test_validate_missing_organization_id(self, engine):
        """Test validation fails for missing organization_id."""
        request = PipelineRequest(
            organization_id=None,
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[{"asset_id": "A001", "asset_type": "Server", "purchase_value": 10000.00}],
            calculation_methods=["spend_based"],
            frameworks=["ghg_protocol"],
        )

        is_valid, errors = engine.validate_request(request)

        assert is_valid is False
        assert any("organization_id" in e for e in errors)

    def test_validate_invalid_date_range(self, engine):
        """Test validation fails for invalid date range."""
        request = PipelineRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 12, 31),
            reporting_period_end=date(2024, 1, 1),  # End before start
            assets=[{"asset_id": "A001", "asset_type": "Server", "purchase_value": 10000.00}],
            calculation_methods=["spend_based"],
            frameworks=["ghg_protocol"],
        )

        is_valid, errors = engine.validate_request(request)

        assert is_valid is False
        assert any("date range" in e.lower() for e in errors)

    def test_validate_empty_assets(self, engine):
        """Test validation fails for empty assets list."""
        request = PipelineRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[],
            calculation_methods=["spend_based"],
            frameworks=["ghg_protocol"],
        )

        is_valid, errors = engine.validate_request(request)

        assert is_valid is False
        assert any("assets" in e.lower() for e in errors)

    def test_validate_invalid_calculation_method(self, engine):
        """Test validation fails for invalid calculation method."""
        request = PipelineRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[{"asset_id": "A001", "asset_type": "Server", "purchase_value": 10000.00}],
            calculation_methods=["invalid_method"],
            frameworks=["ghg_protocol"],
        )

        is_valid, errors = engine.validate_request(request)

        assert is_valid is False
        assert any("calculation method" in e.lower() for e in errors)


class TestGetPipelineStatus:
    """Test get_pipeline_status() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        CapitalGoodsPipelineEngine._instance = None
        return CapitalGoodsPipelineEngine()

    def test_get_pipeline_status(self, engine):
        """Test retrieving pipeline status."""
        status = engine.get_pipeline_status()

        assert "available_stages" in status
        assert "engine_availability" in status
        assert len(status["available_stages"]) == 10

    def test_get_pipeline_status_engine_health(self, engine):
        """Test pipeline status includes engine health."""
        status = engine.get_pipeline_status()

        assert "spend_based_engine" in status["engine_availability"]
        assert "average_data_engine" in status["engine_availability"]
        assert "supplier_specific_engine" in status["engine_availability"]
        assert "hybrid_aggregator_engine" in status["engine_availability"]
        assert "compliance_checker_engine" in status["engine_availability"]


class TestGetStageTiming:
    """Test get_stage_timing() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        CapitalGoodsPipelineEngine._instance = None
        return CapitalGoodsPipelineEngine()

    def test_get_stage_timing_from_result(self, engine):
        """Test extracting stage timing from pipeline result."""
        request = PipelineRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[{"asset_id": "A001", "asset_type": "Server", "purchase_value": 10000.00}],
            calculation_methods=["spend_based"],
            frameworks=["ghg_protocol"],
        )

        result = engine.execute(request)
        timing = engine.get_stage_timing(result)

        assert isinstance(timing, dict)
        assert len(timing) == 10  # All stages
        for stage, duration in timing.items():
            assert duration >= 0

    def test_get_stage_timing_slowest_stage(self, engine):
        """Test identifying slowest stage."""
        request = PipelineRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[{"asset_id": "A001", "asset_type": "Server", "purchase_value": 10000.00}],
            calculation_methods=["spend_based"],
            frameworks=["ghg_protocol"],
        )

        result = engine.execute(request)
        timing = engine.get_stage_timing(result)

        slowest_stage = max(timing, key=timing.get)
        assert slowest_stage in [stage.value for stage in PipelineStage]


class TestExportResults:
    """Test export_results() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        CapitalGoodsPipelineEngine._instance = None
        return CapitalGoodsPipelineEngine()

    @pytest.fixture
    def sample_result(self, engine):
        """Create sample pipeline result."""
        request = PipelineRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[{"asset_id": "A001", "asset_type": "Server", "purchase_value": 10000.00}],
            calculation_methods=["spend_based"],
            frameworks=["ghg_protocol"],
        )
        return engine.execute(request)

    def test_export_results_json(self, engine, sample_result):
        """Test exporting results to JSON format."""
        exported = engine.export_results(sample_result, format=ExportFormat.JSON)

        assert isinstance(exported, str)
        parsed = json.loads(exported)
        assert "organization_id" in parsed
        assert "status" in parsed
        assert "stage_results" in parsed

    def test_export_results_csv(self, engine, sample_result):
        """Test exporting results to CSV format."""
        exported = engine.export_results(sample_result, format=ExportFormat.CSV)

        assert isinstance(exported, str)
        # Parse CSV
        csv_reader = csv.DictReader(io.StringIO(exported))
        rows = list(csv_reader)
        assert len(rows) > 0
        assert "asset_id" in rows[0] or "stage" in rows[0]

    def test_export_results_includes_all_data(self, engine, sample_result):
        """Test export includes all relevant data."""
        exported_json = engine.export_results(sample_result, format=ExportFormat.JSON)
        parsed = json.loads(exported_json)

        assert parsed["organization_id"] == "ORG001"
        assert len(parsed["stage_results"]) == 10
        assert "final_hash" in parsed


class TestGetExecutionSummary:
    """Test get_execution_summary() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        CapitalGoodsPipelineEngine._instance = None
        return CapitalGoodsPipelineEngine()

    def test_get_execution_summary(self, engine):
        """Test generating execution summary."""
        request = PipelineRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[{"asset_id": f"A{i:03d}", "asset_type": "Server", "purchase_value": 10000.00} for i in range(5)],
            calculation_methods=["spend_based"],
            frameworks=["ghg_protocol"],
        )

        result = engine.execute(request)
        summary = engine.get_execution_summary(result)

        assert "organization_id" in summary
        assert "total_assets" in summary
        assert "total_emissions" in summary
        assert "total_duration_ms" in summary
        assert "stages_completed" in summary
        assert "stages_with_warnings" in summary
        assert "stages_failed" in summary

    def test_get_execution_summary_calculates_totals(self, engine):
        """Test summary calculates correct totals."""
        request = PipelineRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[{"asset_id": f"A{i:03d}", "asset_type": "Server", "purchase_value": 10000.00} for i in range(10)],
            calculation_methods=["spend_based"],
            frameworks=["ghg_protocol"],
        )

        result = engine.execute(request)
        summary = engine.get_execution_summary(result)

        assert summary["total_assets"] == 10
        assert summary["stages_completed"] == 10


class TestComparePeriods:
    """Test compare_periods() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        CapitalGoodsPipelineEngine._instance = None
        return CapitalGoodsPipelineEngine()

    def test_compare_periods_two_periods(self, engine):
        """Test comparing two reporting periods."""
        request1 = PipelineRequest(
            organization_id="ORG001",
            reporting_period_start=date(2023, 1, 1),
            reporting_period_end=date(2023, 12, 31),
            assets=[{"asset_id": "A001", "asset_type": "Server", "purchase_value": 8000.00}],
            calculation_methods=["spend_based"],
            frameworks=["ghg_protocol"],
        )

        request2 = PipelineRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[{"asset_id": "A002", "asset_type": "Server", "purchase_value": 10000.00}],
            calculation_methods=["spend_based"],
            frameworks=["ghg_protocol"],
        )

        result1 = engine.execute(request1)
        result2 = engine.execute(request2)

        comparison = engine.compare_periods(result1, result2)

        assert "period1" in comparison
        assert "period2" in comparison
        assert "emissions_change" in comparison
        assert "emissions_change_pct" in comparison
        assert "capex_change" in comparison

    def test_compare_periods_calculates_change(self, engine):
        """Test period comparison calculates correct change percentages."""
        request1 = PipelineRequest(
            organization_id="ORG001",
            reporting_period_start=date(2023, 1, 1),
            reporting_period_end=date(2023, 12, 31),
            assets=[{"asset_id": "A001", "asset_type": "Server", "purchase_value": 10000.00}],
            calculation_methods=["spend_based"],
            frameworks=["ghg_protocol"],
        )

        request2 = PipelineRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[{"asset_id": "A002", "asset_type": "Server", "purchase_value": 15000.00}],  # 50% increase
            calculation_methods=["spend_based"],
            frameworks=["ghg_protocol"],
        )

        result1 = engine.execute(request1)
        result2 = engine.execute(request2)

        comparison = engine.compare_periods(result1, result2)

        # CAPEX increased 50%
        assert abs(comparison["capex_change_pct"] - 50.0) < 1.0


class TestErrorHandling:
    """Test error handling throughout pipeline."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        CapitalGoodsPipelineEngine._instance = None
        return CapitalGoodsPipelineEngine()

    def test_error_handling_invalid_request(self, engine):
        """Test error handling for invalid request."""
        invalid_request = PipelineRequest(
            organization_id=None,  # Invalid
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[],  # Empty
            calculation_methods=["spend_based"],
            frameworks=["ghg_protocol"],
        )

        with pytest.raises(ValueError):
            engine.execute(invalid_request)

    def test_error_handling_stage_failure_continues(self, engine):
        """Test pipeline continues after non-critical stage failure."""
        request = PipelineRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[{"asset_id": "A001", "asset_type": "Server", "purchase_value": 10000.00}],
            calculation_methods=["spend_based"],
            frameworks=["ghg_protocol"],
        )

        # Mock a stage to return warnings
        with patch.object(engine, '_execute_validation_stage') as mock_stage:
            mock_stage.return_value = {
                "stage": PipelineStage.VALIDATION,
                "status": StageStatus.COMPLETED_WITH_WARNINGS,
                "warnings": ["Minor validation issue"],
                "data": {},
            }

            result = engine.execute(request)

            # Should complete despite warnings
            assert result.status in [StageStatus.COMPLETED, StageStatus.COMPLETED_WITH_WARNINGS]

    def test_error_handling_critical_failure_stops(self, engine):
        """Test pipeline stops on critical failure."""
        request = PipelineRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[{"asset_id": "A001", "asset_type": "Server", "purchase_value": 10000.00}],
            calculation_methods=["spend_based"],
            frameworks=["ghg_protocol"],
        )

        # Mock a stage to fail critically
        with patch.object(engine, '_execute_validation_stage') as mock_stage:
            mock_stage.side_effect = Exception("Critical validation failure")

            with pytest.raises(Exception):
                engine.execute(request)


class TestComputeFinalHash:
    """Test compute_final_hash() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        CapitalGoodsPipelineEngine._instance = None
        return CapitalGoodsPipelineEngine()

    def test_compute_final_hash_deterministic(self, engine):
        """Test final hash is deterministic."""
        request = PipelineRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[{"asset_id": "A001", "asset_type": "Server", "purchase_value": 10000.00}],
            calculation_methods=["spend_based"],
            frameworks=["ghg_protocol"],
        )

        result1 = engine.execute(request)
        result2 = engine.execute(request)

        # Same request should produce same hash
        assert result1.final_hash == result2.final_hash

    def test_compute_final_hash_different_requests(self, engine):
        """Test different requests produce different hashes."""
        request1 = PipelineRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[{"asset_id": "A001", "asset_type": "Server", "purchase_value": 10000.00}],
            calculation_methods=["spend_based"],
            frameworks=["ghg_protocol"],
        )

        request2 = PipelineRequest(
            organization_id="ORG002",  # Different org
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[{"asset_id": "A001", "asset_type": "Server", "purchase_value": 10000.00}],
            calculation_methods=["spend_based"],
            frameworks=["ghg_protocol"],
        )

        result1 = engine.execute(request1)
        result2 = engine.execute(request2)

        assert result1.final_hash != result2.final_hash

    def test_compute_final_hash_format(self, engine):
        """Test final hash is SHA-256 format."""
        request = PipelineRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[{"asset_id": "A001", "asset_type": "Server", "purchase_value": 10000.00}],
            calculation_methods=["spend_based"],
            frameworks=["ghg_protocol"],
        )

        result = engine.execute(request)

        assert len(result.final_hash) == 64  # SHA-256 hex length
        assert all(c in '0123456789abcdef' for c in result.final_hash)
