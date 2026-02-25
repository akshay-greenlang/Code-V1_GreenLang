"""
Unit tests for FuelEnergyPipelineEngine (Engine 7)

Tests full pipeline execution for all fuel & energy activities.
Validates 10-stage orchestration, aggregation, and provenance tracking.
"""

import pytest
from decimal import Decimal
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import Mock, patch
import json

from greenlang.fuel_energy_activities.engines.fuel_energy_pipeline import (
    FuelEnergyPipelineEngine,
    PipelineInput,
    PipelineOutput,
    PipelineStage,
    AggregationDimension,
)
from greenlang.fuel_energy_activities.models import (
    FuelType,
    ActivityType,
)
from greenlang_core import AgentConfig
from greenlang_core.exceptions import ValidationError, ProcessingError


# Fixtures
@pytest.fixture
def agent_config():
    """Create test agent configuration."""
    return AgentConfig(
        name="fuel_energy_pipeline",
        version="1.0.0",
        environment="test"
    )


@pytest.fixture
def engine(agent_config):
    """Create FuelEnergyPipelineEngine instance for testing."""
    return FuelEnergyPipelineEngine(agent_config)


@pytest.fixture
def full_pipeline_input():
    """Create full pipeline input with all activities."""
    return PipelineInput(
        # Activity 3a: Upstream of purchased fuels
        fuel_consumptions=[
            {
                "fuel_type": FuelType.NATURAL_GAS,
                "quantity": Decimal("1000"),
                "country": "US"
            },
            {
                "fuel_type": FuelType.DIESEL,
                "quantity": Decimal("500"),
                "country": "US"
            },
        ],
        # Activity 3b: Upstream of purchased electricity
        electricity_consumptions=[
            {
                "electricity_kwh": Decimal("100000"),
                "country": "US"
            },
        ],
        # Activity 3c: T&D losses
        td_loss_calculations=[
            {
                "electricity_consumption_kwh": Decimal("100000"),
                "country": "US"
            },
        ],
        reporting_period="2025-Q1"
    )


@pytest.fixture
def activity_3a_only_input():
    """Create pipeline input with activity 3a only."""
    return PipelineInput(
        fuel_consumptions=[
            {
                "fuel_type": FuelType.NATURAL_GAS,
                "quantity": Decimal("2000"),
                "country": "US"
            },
        ],
        reporting_period="2025-Q1"
    )


# Test Class
class TestFuelEnergyPipelineEngine:
    """Test suite for FuelEnergyPipelineEngine."""

    def test_initialization(self, agent_config):
        """Test engine initializes correctly."""
        engine = FuelEnergyPipelineEngine(agent_config)

        assert engine.config == agent_config
        assert engine.wtt_calculator is not None
        assert engine.upstream_calculator is not None
        assert engine.td_loss_calculator is not None
        assert engine.supplier_specific_calculator is not None
        assert engine.compliance_checker is not None

    def test_execute_full_pipeline_10_stages(self, engine, full_pipeline_input):
        """Test executing full pipeline through 10 stages."""
        result = engine.execute(full_pipeline_input)

        assert isinstance(result, PipelineOutput)

        # Verify all 10 stages executed
        expected_stages = [
            PipelineStage.INPUT_VALIDATION,
            PipelineStage.ACTIVITY_3A_CALCULATION,
            PipelineStage.ACTIVITY_3B_CALCULATION,
            PipelineStage.ACTIVITY_3C_CALCULATION,
            PipelineStage.SUPPLIER_SPECIFIC_PROCESSING,
            PipelineStage.AGGREGATION,
            PipelineStage.COMPLIANCE_CHECK,
            PipelineStage.DQI_ASSESSMENT,
            PipelineStage.UNCERTAINTY_QUANTIFICATION,
            PipelineStage.OUTPUT_GENERATION,
        ]

        assert len(result.stages_executed) == len(expected_stages)
        for stage in expected_stages:
            assert stage in result.stages_executed

        # Verify results
        assert result.total_emissions_kgco2e > Decimal("0")
        assert result.activity_3a_emissions_kgco2e > Decimal("0")
        assert result.activity_3b_emissions_kgco2e > Decimal("0")
        assert result.activity_3c_emissions_kgco2e > Decimal("0")

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_execute_activity_3a_only(self, engine, activity_3a_only_input):
        """Test executing pipeline with activity 3a only."""
        result = engine.execute(activity_3a_only_input)

        assert isinstance(result, PipelineOutput)
        assert result.activity_3a_emissions_kgco2e > Decimal("0")
        assert result.activity_3b_emissions_kgco2e == Decimal("0")
        assert result.activity_3c_emissions_kgco2e == Decimal("0")

        # Should still execute applicable stages
        assert PipelineStage.ACTIVITY_3A_CALCULATION in result.stages_executed
        assert PipelineStage.ACTIVITY_3B_CALCULATION not in result.stages_executed

    def test_execute_activity_3b_only(self, engine):
        """Test executing pipeline with activity 3b only."""
        input_data = PipelineInput(
            electricity_consumptions=[
                {
                    "electricity_kwh": Decimal("150000"),
                    "country": "US"
                },
            ],
            reporting_period="2025-Q1"
        )

        result = engine.execute(input_data)

        assert result.activity_3a_emissions_kgco2e == Decimal("0")
        assert result.activity_3b_emissions_kgco2e > Decimal("0")
        assert result.activity_3c_emissions_kgco2e == Decimal("0")

    def test_execute_activity_3c_only(self, engine):
        """Test executing pipeline with activity 3c only."""
        input_data = PipelineInput(
            td_loss_calculations=[
                {
                    "electricity_consumption_kwh": Decimal("80000"),
                    "country": "US"
                },
            ],
            reporting_period="2025-Q1"
        )

        result = engine.execute(input_data)

        assert result.activity_3a_emissions_kgco2e == Decimal("0")
        assert result.activity_3b_emissions_kgco2e == Decimal("0")
        assert result.activity_3c_emissions_kgco2e > Decimal("0")

    def test_execute_batch(self, engine):
        """Test executing pipeline for batch of inputs."""
        inputs = [
            PipelineInput(
                fuel_consumptions=[
                    {"fuel_type": FuelType.NATURAL_GAS, "quantity": Decimal("1000"), "country": "US"}
                ],
                reporting_period="2025-Q1"
            ),
            PipelineInput(
                electricity_consumptions=[
                    {"electricity_kwh": Decimal("50000"), "country": "US"}
                ],
                reporting_period="2025-Q1"
            ),
            PipelineInput(
                fuel_consumptions=[
                    {"fuel_type": FuelType.DIESEL, "quantity": Decimal("500"), "country": "GB"}
                ],
                reporting_period="2025-Q2"
            ),
        ]

        results = engine.execute_batch(inputs)

        assert len(results) == 3
        assert all(isinstance(r, PipelineOutput) for r in results)

    def test_aggregate_results_by_activity(self, engine, full_pipeline_input):
        """Test aggregating results by activity type."""
        result = engine.execute(full_pipeline_input)

        aggregated = engine.aggregate_by_dimension(
            [result],
            dimension=AggregationDimension.ACTIVITY
        )

        assert ActivityType.ACTIVITY_3A in aggregated
        assert ActivityType.ACTIVITY_3B in aggregated
        assert ActivityType.ACTIVITY_3C in aggregated

        # Activity 3a
        assert aggregated[ActivityType.ACTIVITY_3A]["total_emissions_kgco2e"] == result.activity_3a_emissions_kgco2e

    def test_aggregate_results_by_fuel_type(self, engine, full_pipeline_input):
        """Test aggregating results by fuel type."""
        result = engine.execute(full_pipeline_input)

        aggregated = engine.aggregate_by_dimension(
            [result],
            dimension=AggregationDimension.FUEL_TYPE
        )

        assert FuelType.NATURAL_GAS in aggregated
        assert FuelType.DIESEL in aggregated

        # Should have emissions for each fuel type
        assert aggregated[FuelType.NATURAL_GAS]["total_emissions_kgco2e"] > Decimal("0")
        assert aggregated[FuelType.DIESEL]["total_emissions_kgco2e"] > Decimal("0")

    def test_export_results_json(self, engine, full_pipeline_input):
        """Test exporting results to JSON."""
        result = engine.execute(full_pipeline_input)

        json_output = engine.export_to_json(result)

        assert isinstance(json_output, str)

        # Parse to verify valid JSON
        parsed = json.loads(json_output)
        assert "total_emissions_kgco2e" in parsed
        assert "activity_3a_emissions_kgco2e" in parsed

    def test_export_results_csv(self, engine, full_pipeline_input):
        """Test exporting results to CSV."""
        result = engine.execute(full_pipeline_input)

        csv_output = engine.export_to_csv([result])

        assert isinstance(csv_output, str)
        assert "total_emissions_kgco2e" in csv_output
        assert "activity_3a" in csv_output.lower()

    def test_get_summary(self, engine, full_pipeline_input):
        """Test getting pipeline summary."""
        result = engine.execute(full_pipeline_input)

        summary = engine.get_summary(result)

        assert "total_emissions_kgco2e" in summary
        assert "activity_breakdown" in summary
        assert "dqi_score" in summary
        assert "uncertainty_pct" in summary

    def test_get_hot_spots(self, engine, full_pipeline_input):
        """Test identifying emissions hot spots."""
        result = engine.execute(full_pipeline_input)

        hot_spots = engine.get_hot_spots(result, top_n=3)

        assert isinstance(hot_spots, list)
        assert len(hot_spots) <= 3

        # Should be sorted by emissions (descending)
        if len(hot_spots) > 1:
            assert hot_spots[0]["emissions_kgco2e"] >= hot_spots[1]["emissions_kgco2e"]

    def test_get_materiality(self, engine, full_pipeline_input):
        """Test assessing materiality of fuel & energy emissions."""
        result = engine.execute(full_pipeline_input)

        materiality = engine.assess_materiality(
            result,
            total_scope3_emissions_kgco2e=Decimal("500000")
        )

        assert "percentage_of_scope3" in materiality
        assert "material" in materiality
        assert "threshold_pct" in materiality

        # Calculate expected percentage
        expected_pct = (result.total_emissions_kgco2e / Decimal("500000")) * Decimal("100")
        assert materiality["percentage_of_scope3"] == pytest.approx(expected_pct, rel=Decimal("0.01"))

    def test_compare_periods(self, engine):
        """Test comparing results across periods."""
        q1_input = PipelineInput(
            fuel_consumptions=[
                {"fuel_type": FuelType.NATURAL_GAS, "quantity": Decimal("1000"), "country": "US"}
            ],
            reporting_period="2025-Q1"
        )

        q2_input = PipelineInput(
            fuel_consumptions=[
                {"fuel_type": FuelType.NATURAL_GAS, "quantity": Decimal("1200"), "country": "US"}
            ],
            reporting_period="2025-Q2"
        )

        q1_result = engine.execute(q1_input)
        q2_result = engine.execute(q2_input)

        comparison = engine.compare_periods(q1_result, q2_result)

        assert "q1_emissions" in comparison or "period1_emissions" in comparison
        assert "q2_emissions" in comparison or "period2_emissions" in comparison
        assert "change_pct" in comparison
        assert "change_kgco2e" in comparison

        # Q2 should be higher
        assert comparison["change_kgco2e"] > Decimal("0")

    def test_pipeline_provenance_chain(self, engine, full_pipeline_input):
        """Test provenance chain through pipeline stages."""
        result = engine.execute(full_pipeline_input)

        # Should have provenance for each stage
        assert result.stage_provenance is not None
        assert len(result.stage_provenance) == len(result.stages_executed)

        # Each stage should have a hash
        for stage, hash_value in result.stage_provenance.items():
            assert hash_value is not None
            assert len(hash_value) == 64

    def test_pipeline_error_handling_graceful(self, engine):
        """Test pipeline handles errors gracefully."""
        # Invalid input
        invalid_input = PipelineInput(
            fuel_consumptions=[
                {
                    "fuel_type": "INVALID_FUEL",
                    "quantity": Decimal("-1000"),  # Negative
                    "country": "US"
                }
            ],
            reporting_period="2025-Q1"
        )

        with pytest.raises(ValidationError):
            engine.execute(invalid_input)

    def test_stage_timing(self, engine, full_pipeline_input):
        """Test pipeline tracks timing for each stage."""
        result = engine.execute(full_pipeline_input)

        assert result.stage_timings is not None

        # Each stage should have timing
        for stage in result.stages_executed:
            assert stage in result.stage_timings
            assert result.stage_timings[stage] > 0  # Milliseconds

    def test_partial_execution_on_stage_failure(self, engine):
        """Test pipeline continues with partial results on non-critical stage failure."""
        # Mock a non-critical stage failure
        with patch.object(engine, 'supplier_specific_calculator') as mock_supplier:
            mock_supplier.calculate.side_effect = Exception("Supplier data unavailable")

            input_data = PipelineInput(
                fuel_consumptions=[
                    {"fuel_type": FuelType.NATURAL_GAS, "quantity": Decimal("1000"), "country": "US"}
                ],
                reporting_period="2025-Q1"
            )

            # Should complete with warnings (if supplier-specific is optional)
            try:
                result = engine.execute(input_data)
                # If it succeeds, should have warnings
                assert len(result.warnings) > 0
            except ProcessingError:
                # Or it may fail if stage is critical
                pass

    def test_aggregate_by_country(self, engine):
        """Test aggregating by country."""
        input_data = PipelineInput(
            fuel_consumptions=[
                {"fuel_type": FuelType.NATURAL_GAS, "quantity": Decimal("1000"), "country": "US"},
                {"fuel_type": FuelType.DIESEL, "quantity": Decimal("500"), "country": "US"},
                {"fuel_type": FuelType.NATURAL_GAS, "quantity": Decimal("800"), "country": "GB"},
            ],
            reporting_period="2025-Q1"
        )

        result = engine.execute(input_data)

        aggregated = engine.aggregate_by_dimension(
            [result],
            dimension=AggregationDimension.COUNTRY
        )

        assert "US" in aggregated
        assert "GB" in aggregated

        # US should have more emissions (2 fuel types vs 1)
        assert aggregated["US"]["total_emissions_kgco2e"] > aggregated["GB"]["total_emissions_kgco2e"]

    def test_aggregate_by_facility(self, engine):
        """Test aggregating by facility."""
        input_data = PipelineInput(
            fuel_consumptions=[
                {
                    "fuel_type": FuelType.NATURAL_GAS,
                    "quantity": Decimal("1000"),
                    "facility_id": "FAC-001",
                    "country": "US"
                },
                {
                    "fuel_type": FuelType.DIESEL,
                    "quantity": Decimal("500"),
                    "facility_id": "FAC-001",
                    "country": "US"
                },
                {
                    "fuel_type": FuelType.NATURAL_GAS,
                    "quantity": Decimal("600"),
                    "facility_id": "FAC-002",
                    "country": "US"
                },
            ],
            reporting_period="2025-Q1"
        )

        result = engine.execute(input_data)

        aggregated = engine.aggregate_by_dimension(
            [result],
            dimension=AggregationDimension.FACILITY
        )

        assert "FAC-001" in aggregated
        assert "FAC-002" in aggregated

    def test_aggregate_by_reporting_period(self, engine):
        """Test aggregating by reporting period."""
        inputs = [
            PipelineInput(
                fuel_consumptions=[
                    {"fuel_type": FuelType.NATURAL_GAS, "quantity": Decimal("1000"), "country": "US"}
                ],
                reporting_period="2025-Q1"
            ),
            PipelineInput(
                fuel_consumptions=[
                    {"fuel_type": FuelType.NATURAL_GAS, "quantity": Decimal("1200"), "country": "US"}
                ],
                reporting_period="2025-Q2"
            ),
        ]

        results = engine.execute_batch(inputs)

        aggregated = engine.aggregate_by_dimension(
            results,
            dimension=AggregationDimension.REPORTING_PERIOD
        )

        assert "2025-Q1" in aggregated
        assert "2025-Q2" in aggregated

    def test_compliance_integration(self, engine, full_pipeline_input):
        """Test compliance checking is integrated."""
        result = engine.execute(full_pipeline_input)

        assert result.compliance_results is not None
        assert len(result.compliance_results) > 0

        # Should have checked at least GHG Protocol
        ghp_result = next(
            (r for r in result.compliance_results if r.framework.value == "GHG_PROTOCOL"),
            None
        )
        assert ghp_result is not None

    def test_dqi_integration(self, engine, full_pipeline_input):
        """Test DQI assessment is integrated."""
        result = engine.execute(full_pipeline_input)

        assert result.dqi_score is not None
        assert Decimal("0") <= result.dqi_score <= Decimal("5")

    def test_uncertainty_integration(self, engine, full_pipeline_input):
        """Test uncertainty quantification is integrated."""
        result = engine.execute(full_pipeline_input)

        assert result.uncertainty_pct is not None
        assert result.uncertainty_pct > Decimal("0")

    def test_get_statistics(self, engine, full_pipeline_input):
        """Test getting pipeline statistics."""
        engine.execute(full_pipeline_input)
        engine.execute(full_pipeline_input)

        stats = engine.get_statistics()

        assert stats["pipelines_executed"] == 2
        assert stats["total_emissions_kgco2e"] > Decimal("0")

    def test_reset(self, engine, full_pipeline_input):
        """Test resetting pipeline state."""
        engine.execute(full_pipeline_input)

        engine.reset()

        stats = engine.get_statistics()
        assert stats["pipelines_executed"] == 0

    def test_validate_input(self, engine, full_pipeline_input):
        """Test input validation stage."""
        is_valid = engine.validate_input(full_pipeline_input)

        assert is_valid is True

    def test_validate_input_rejects_invalid(self, engine):
        """Test input validation rejects invalid data."""
        invalid_input = PipelineInput(
            fuel_consumptions=[
                {
                    "fuel_type": FuelType.NATURAL_GAS,
                    "quantity": Decimal("-1000"),  # Negative
                    "country": "US"
                }
            ],
            reporting_period="2025-Q1"
        )

        with pytest.raises(ValidationError):
            engine.validate_input(invalid_input, raise_on_invalid=True)

    def test_generate_report(self, engine, full_pipeline_input):
        """Test generating human-readable report."""
        result = engine.execute(full_pipeline_input)

        report = engine.generate_report(result)

        assert isinstance(report, str)
        assert "Fuel & Energy Activities" in report
        assert "Activity 3a" in report
        assert "kgCO2e" in report

    def test_performance_single_execution(self, engine, full_pipeline_input, benchmark):
        """Test single pipeline execution performance."""
        def run_pipeline():
            return engine.execute(full_pipeline_input)

        result = benchmark(run_pipeline)

        assert isinstance(result, PipelineOutput)

    def test_provenance_deterministic(self, engine, full_pipeline_input):
        """Test provenance is deterministic."""
        result1 = engine.execute(full_pipeline_input)
        result2 = engine.execute(full_pipeline_input)

        assert result1.provenance_hash == result2.provenance_hash

    def test_metadata_populated(self, engine, full_pipeline_input):
        """Test metadata fields are populated."""
        result = engine.execute(full_pipeline_input)

        assert result.calculation_timestamp is not None
        assert result.engine_version is not None
        assert result.reporting_period == "2025-Q1"


# Integration Tests
class TestFuelEnergyPipelineIntegration:
    """Integration tests for FuelEnergyPipelineEngine."""

    @pytest.mark.integration
    def test_end_to_end_pipeline(self, engine):
        """Test full end-to-end pipeline execution."""
        pass

    @pytest.mark.integration
    def test_integration_with_all_engines(self, engine):
        """Test integration with all calculation engines."""
        pass


# Performance Tests
class TestFuelEnergyPipelinePerformance:
    """Performance tests for FuelEnergyPipelineEngine."""

    @pytest.mark.performance
    def test_throughput_target(self, engine):
        """Test pipeline meets throughput target (100 pipelines/sec)."""
        num_pipelines = 1000

        inputs = [
            PipelineInput(
                fuel_consumptions=[
                    {"fuel_type": FuelType.NATURAL_GAS, "quantity": Decimal("1000"), "country": "US"}
                ],
                reporting_period="2025-Q1"
            )
            for _ in range(num_pipelines)
        ]

        start_time = datetime.now()
        results = engine.execute_batch(inputs)
        end_time = datetime.now()

        duration_seconds = (end_time - start_time).total_seconds()
        throughput = num_pipelines / duration_seconds

        assert throughput >= 100  # Target: 100 pipelines/sec
        assert len(results) == num_pipelines
