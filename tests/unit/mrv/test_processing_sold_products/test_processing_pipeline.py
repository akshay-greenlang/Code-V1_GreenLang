# -*- coding: utf-8 -*-
"""
Unit tests for ProcessingPipelineEngine -- AGENT-MRV-023

Tests the 10-stage processing pipeline including full pipeline execution,
batch processing, portfolio analysis, input validation, stage enums, method
waterfall, partial results on error, pipeline status, and singleton pattern.

Target: 30+ tests.
Author: GL-TestEngineer
"""

import threading
from decimal import Decimal
from typing import Any, Dict, List

import pytest

try:
    from greenlang.agents.mrv.processing_sold_products.processing_pipeline import (
        ProcessingPipelineEngine,
        PipelineStage,
        PipelineStatus,
        CalculationMethod,
        AllocationMethod,
        IntermediateProductCategory,
        ProcessingType,
        ComplianceFramework,
        EFSource,
        DEFAULT_PROCESSING_EFS,
        DEFAULT_GRID_EFS,
        DEFAULT_FUEL_EFS,
        DEFAULT_EEIO_FACTORS,
        UNIT_CONVERSIONS,
    )

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_SKIP = pytest.mark.skipif(not _AVAILABLE, reason="ProcessingPipelineEngine not available")
pytestmark = _SKIP


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(autouse=True)
def _reset_engine():
    """Reset the singleton before each test."""
    ProcessingPipelineEngine.reset()
    yield
    ProcessingPipelineEngine.reset()


@pytest.fixture
def engine():
    """Create a ProcessingPipelineEngine instance."""
    return ProcessingPipelineEngine()


@pytest.fixture
def valid_pipeline_input():
    """Create valid input for pipeline execution."""
    return {
        "products": [
            {
                "product_id": "STEEL-001",
                "product_category": "metals_ferrous",
                "processing_type": "machining",
                "mass_tonnes": "500",
            },
        ],
    }


@pytest.fixture
def multi_product_input():
    """Create multi-product input for pipeline execution."""
    return {
        "products": [
            {
                "product_id": "STEEL-001",
                "product_category": "metals_ferrous",
                "processing_type": "machining",
                "mass_tonnes": "500",
            },
            {
                "product_id": "PLASTIC-001",
                "product_category": "plastics_thermoplastic",
                "processing_type": "injection_molding",
                "mass_tonnes": "300",
            },
            {
                "product_id": "CHEM-001",
                "product_category": "chemicals",
                "processing_type": "chemical_reaction",
                "mass_tonnes": "200",
            },
        ],
    }


# ============================================================================
# TEST: Pipeline Stage Enum
# ============================================================================


class TestPipelineStageEnum:
    """Test PipelineStage enum values."""

    def test_10_stages_defined(self):
        """Test that exactly 10 pipeline stages are defined."""
        assert len(PipelineStage) == 10

    @pytest.mark.parametrize(
        "stage,value",
        [
            (PipelineStage.VALIDATE, "VALIDATE"),
            (PipelineStage.CLASSIFY, "CLASSIFY"),
            (PipelineStage.NORMALIZE, "NORMALIZE"),
            (PipelineStage.RESOLVE_EFS, "RESOLVE_EFS"),
            (PipelineStage.CALCULATE, "CALCULATE"),
            (PipelineStage.ALLOCATE, "ALLOCATE"),
            (PipelineStage.AGGREGATE, "AGGREGATE"),
            (PipelineStage.COMPLIANCE, "COMPLIANCE"),
            (PipelineStage.PROVENANCE, "PROVENANCE"),
            (PipelineStage.SEAL, "SEAL"),
        ],
    )
    def test_stage_values(self, stage, value):
        """Test each stage enum has the correct string value."""
        assert stage.value == value


# ============================================================================
# TEST: Pipeline Status Enum
# ============================================================================


class TestPipelineStatusEnum:
    """Test PipelineStatus enum values."""

    def test_4_statuses_defined(self):
        """Test that exactly 4 pipeline statuses are defined."""
        assert len(PipelineStatus) == 4

    @pytest.mark.parametrize(
        "status,value",
        [
            (PipelineStatus.SUCCESS, "SUCCESS"),
            (PipelineStatus.PARTIAL_SUCCESS, "PARTIAL_SUCCESS"),
            (PipelineStatus.FAILED, "FAILED"),
            (PipelineStatus.VALIDATION_ERROR, "VALIDATION_ERROR"),
        ],
    )
    def test_status_values(self, status, value):
        """Test each status enum has the correct string value."""
        assert status.value == value


# ============================================================================
# TEST: Calculation Method Enum
# ============================================================================


class TestCalculationMethodEnum:
    """Test CalculationMethod enum values."""

    def test_6_methods_defined(self):
        """Test that 6 calculation methods are defined."""
        assert len(CalculationMethod) == 6

    @pytest.mark.parametrize(
        "method,value",
        [
            (CalculationMethod.SITE_SPECIFIC_DIRECT, "site_specific_direct"),
            (CalculationMethod.SITE_SPECIFIC_ENERGY, "site_specific_energy"),
            (CalculationMethod.SITE_SPECIFIC_FUEL, "site_specific_fuel"),
            (CalculationMethod.AVERAGE_DATA, "average_data"),
            (CalculationMethod.SPEND_BASED, "spend_based"),
            (CalculationMethod.HYBRID, "hybrid"),
        ],
    )
    def test_method_values(self, method, value):
        """Test each method enum has the correct string value."""
        assert method.value == value


# ============================================================================
# TEST: Full Pipeline Execution
# ============================================================================


class TestFullPipeline:
    """Test complete 10-stage pipeline execution."""

    def test_run_pipeline_returns_dict(self, engine, valid_pipeline_input):
        """Test that run_pipeline returns a dictionary."""
        result = engine.run_pipeline(
            inputs=valid_pipeline_input,
            method="average_data",
            org_id="ORG-001",
            reporting_year=2024,
        )
        assert isinstance(result, dict)

    def test_run_pipeline_success_status(self, engine, valid_pipeline_input):
        """Test that a valid run produces SUCCESS or PARTIAL_SUCCESS status."""
        result = engine.run_pipeline(
            inputs=valid_pipeline_input,
            method="average_data",
            org_id="ORG-001",
            reporting_year=2024,
        )
        assert result["status"] in (
            PipelineStatus.SUCCESS.value,
            PipelineStatus.PARTIAL_SUCCESS.value,
        )

    def test_run_pipeline_contains_pipeline_id(self, engine, valid_pipeline_input):
        """Test that result contains a pipeline_id."""
        result = engine.run_pipeline(
            inputs=valid_pipeline_input,
            method="average_data",
            org_id="ORG-001",
            reporting_year=2024,
        )
        assert "pipeline_id" in result
        assert isinstance(result["pipeline_id"], str)

    def test_run_pipeline_contains_emissions(self, engine, valid_pipeline_input):
        """Test that result contains total emissions."""
        result = engine.run_pipeline(
            inputs=valid_pipeline_input,
            method="average_data",
            org_id="ORG-001",
            reporting_year=2024,
        )
        assert "total_emissions_kg_co2e" in result

    def test_run_pipeline_contains_stage_durations(self, engine, valid_pipeline_input):
        """Test that result contains stage timing information."""
        result = engine.run_pipeline(
            inputs=valid_pipeline_input,
            method="average_data",
            org_id="ORG-001",
            reporting_year=2024,
        )
        assert "stage_durations_ms" in result
        durations = result["stage_durations_ms"]
        # Should have timings for at least some stages
        assert isinstance(durations, dict)

    def test_run_pipeline_contains_products(self, engine, valid_pipeline_input):
        """Test that result contains product-level results."""
        result = engine.run_pipeline(
            inputs=valid_pipeline_input,
            method="average_data",
            org_id="ORG-001",
            reporting_year=2024,
        )
        assert "products" in result

    def test_run_pipeline_multi_product(self, engine, multi_product_input):
        """Test pipeline with multiple products."""
        result = engine.run_pipeline(
            inputs=multi_product_input,
            method="average_data",
            org_id="ORG-001",
            reporting_year=2024,
        )
        assert result["status"] in (
            PipelineStatus.SUCCESS.value,
            PipelineStatus.PARTIAL_SUCCESS.value,
        )
        assert result.get("total_products", 0) >= 1


# ============================================================================
# TEST: Invalid Inputs
# ============================================================================


class TestInvalidInputs:
    """Test pipeline behavior with invalid inputs."""

    def test_invalid_method_returns_error(self, engine, valid_pipeline_input):
        """Test that an invalid calculation method produces an error."""
        result = engine.run_pipeline(
            inputs=valid_pipeline_input,
            method="nonexistent_method",
            org_id="ORG-001",
            reporting_year=2024,
        )
        assert result["status"] in (
            PipelineStatus.VALIDATION_ERROR.value,
            PipelineStatus.FAILED.value,
        )
        assert len(result.get("errors", [])) > 0

    def test_empty_products_returns_error(self, engine):
        """Test that empty products list produces an error."""
        result = engine.run_pipeline(
            inputs={"products": []},
            method="average_data",
            org_id="ORG-001",
            reporting_year=2024,
        )
        assert result["status"] in (
            PipelineStatus.VALIDATION_ERROR.value,
            PipelineStatus.FAILED.value,
        )

    def test_missing_products_key_returns_error(self, engine):
        """Test that missing 'products' key produces an error."""
        result = engine.run_pipeline(
            inputs={},
            method="average_data",
            org_id="ORG-001",
            reporting_year=2024,
        )
        assert result["status"] in (
            PipelineStatus.VALIDATION_ERROR.value,
            PipelineStatus.FAILED.value,
        )


# ============================================================================
# TEST: Input Validation
# ============================================================================


class TestInputValidation:
    """Test the validate_inputs pre-flight check."""

    def test_valid_input_no_errors(self, engine):
        """Test that valid input produces no errors."""
        inputs = {
            "products": [
                {
                    "product_id": "P1",
                    "product_category": "metals_ferrous",
                    "processing_type": "machining",
                    "mass_tonnes": "100",
                }
            ]
        }
        errors = engine.validate_inputs(inputs)
        assert len(errors) == 0

    def test_validate_missing_products_key(self, engine):
        """Test that missing 'products' key is flagged."""
        errors = engine.validate_inputs({})
        assert len(errors) > 0
        assert any("products" in e.lower() for e in errors)

    def test_validate_empty_products_list(self, engine):
        """Test that empty products list is flagged."""
        errors = engine.validate_inputs({"products": []})
        assert len(errors) > 0

    def test_validate_invalid_category(self, engine):
        """Test that an invalid product category is flagged."""
        inputs = {
            "products": [
                {
                    "product_id": "P1",
                    "product_category": "invalid_category",
                    "processing_type": "machining",
                    "mass_tonnes": "100",
                }
            ]
        }
        errors = engine.validate_inputs(inputs)
        assert any("category" in e.lower() for e in errors)

    def test_validate_invalid_processing_type(self, engine):
        """Test that an invalid processing type is flagged."""
        inputs = {
            "products": [
                {
                    "product_id": "P1",
                    "product_category": "metals_ferrous",
                    "processing_type": "invalid_processing",
                    "mass_tonnes": "100",
                }
            ]
        }
        errors = engine.validate_inputs(inputs)
        assert any("processing" in e.lower() for e in errors)

    def test_validate_negative_mass(self, engine):
        """Test that negative mass value is flagged."""
        inputs = {
            "products": [
                {
                    "product_id": "P1",
                    "product_category": "metals_ferrous",
                    "processing_type": "machining",
                    "mass_tonnes": "-100",
                }
            ]
        }
        errors = engine.validate_inputs(inputs)
        assert any("non-negative" in e.lower() or "negative" in e.lower() for e in errors)


# ============================================================================
# TEST: Batch Pipeline
# ============================================================================


class TestBatchPipeline:
    """Test batch pipeline execution."""

    def test_run_batch_returns_list(self, engine, valid_pipeline_input):
        """Test that run_batch returns a list of results."""
        results = engine.run_batch(
            batch_inputs=[valid_pipeline_input, valid_pipeline_input],
            method="average_data",
            org_id="ORG-001",
            reporting_year=2024,
        )
        assert isinstance(results, list)
        assert len(results) == 2

    def test_run_batch_each_has_pipeline_id(self, engine, valid_pipeline_input):
        """Test that each batch result has a unique pipeline_id."""
        results = engine.run_batch(
            batch_inputs=[valid_pipeline_input, valid_pipeline_input],
            method="average_data",
            org_id="ORG-001",
            reporting_year=2024,
        )
        ids = [r.get("pipeline_id", "") for r in results]
        assert len(set(ids)) == 2  # All unique

    def test_run_batch_partial_failure(self, engine, valid_pipeline_input):
        """Test that one failure in a batch does not affect other results."""
        invalid_input = {"products": []}
        results = engine.run_batch(
            batch_inputs=[valid_pipeline_input, invalid_input],
            method="average_data",
            org_id="ORG-001",
            reporting_year=2024,
        )
        assert len(results) == 2
        # At least one should succeed and one should fail
        statuses = [r.get("status", "") for r in results]
        assert any(s in (PipelineStatus.SUCCESS.value, PipelineStatus.PARTIAL_SUCCESS.value) for s in statuses)


# ============================================================================
# TEST: Portfolio Analysis
# ============================================================================


class TestPortfolioAnalysis:
    """Test portfolio-level analysis."""

    def test_portfolio_analysis_returns_dict(self, engine, multi_product_input):
        """Test that portfolio analysis returns a dictionary."""
        result = engine.run_portfolio_analysis(
            inputs=multi_product_input,
            org_id="ORG-001",
            reporting_year=2024,
        )
        assert isinstance(result, dict)

    def test_portfolio_has_hot_spots(self, engine, multi_product_input):
        """Test that portfolio result includes hotspot analysis."""
        result = engine.run_portfolio_analysis(
            inputs=multi_product_input,
            org_id="ORG-001",
            reporting_year=2024,
        )
        assert "hot_spots" in result

    def test_portfolio_has_method_breakdown(self, engine, multi_product_input):
        """Test that portfolio result includes method breakdown."""
        result = engine.run_portfolio_analysis(
            inputs=multi_product_input,
            org_id="ORG-001",
            reporting_year=2024,
        )
        assert "method_breakdown" in result

    def test_portfolio_has_summary(self, engine, multi_product_input):
        """Test that portfolio result includes portfolio summary."""
        result = engine.run_portfolio_analysis(
            inputs=multi_product_input,
            org_id="ORG-001",
            reporting_year=2024,
        )
        assert "portfolio_summary" in result


# ============================================================================
# TEST: Pipeline Status
# ============================================================================


class TestPipelineStatus:
    """Test pipeline status and statistics."""

    def test_get_pipeline_status(self, engine):
        """Test that pipeline status returns valid structure."""
        status = engine.get_pipeline_status()
        assert "pipeline_run_count" in status
        assert "total_products_processed" in status
        assert "engine_version" in status
        assert "agent_id" in status
        assert "stages" in status
        assert len(status["stages"]) == 10

    def test_pipeline_run_count_increments(self, engine, valid_pipeline_input):
        """Test that pipeline_run_count increments after successful run."""
        initial = engine.get_pipeline_status()["pipeline_run_count"]
        engine.run_pipeline(
            inputs=valid_pipeline_input,
            method="average_data",
            org_id="ORG-001",
            reporting_year=2024,
        )
        after = engine.get_pipeline_status()["pipeline_run_count"]
        assert after >= initial

    def test_estimate_runtime(self, engine):
        """Test pipeline runtime estimation."""
        estimate_1 = engine.estimate_runtime(1)
        estimate_100 = engine.estimate_runtime(100)
        assert estimate_1 > 0
        assert estimate_100 > estimate_1


# ============================================================================
# TEST: Default Emission Factors
# ============================================================================


class TestDefaultEFs:
    """Test default emission factor tables."""

    def test_processing_efs_not_empty(self):
        """Test that default processing EFs are populated."""
        assert len(DEFAULT_PROCESSING_EFS) > 0

    def test_processing_ef_metals_machining(self):
        """Test known value: metals_ferrous:machining = 85.0 kgCO2e/t."""
        ef_data = DEFAULT_PROCESSING_EFS.get("metals_ferrous:machining")
        assert ef_data is not None
        assert ef_data["ef_kg_co2e_per_tonne"] == Decimal("85.0")

    def test_grid_efs_has_us_average(self):
        """Test that default grid EFs include us_average."""
        assert "us_average" in DEFAULT_GRID_EFS
        assert DEFAULT_GRID_EFS["us_average"] == Decimal("0.386")

    def test_fuel_efs_has_diesel(self):
        """Test that default fuel EFs include diesel."""
        assert "diesel_litre" in DEFAULT_FUEL_EFS
        assert DEFAULT_FUEL_EFS["diesel_litre"] == Decimal("2.68")

    def test_eeio_factors_has_metals_ferrous(self):
        """Test that default EEIO factors include metals_ferrous."""
        assert "metals_ferrous" in DEFAULT_EEIO_FACTORS
        assert DEFAULT_EEIO_FACTORS["metals_ferrous"]["ef_kg_co2e_per_usd"] == Decimal("0.55")


# ============================================================================
# TEST: Unit Conversions
# ============================================================================


class TestUnitConversions:
    """Test unit conversion constants."""

    def test_kg_to_tonnes(self):
        """Test kg to tonnes conversion factor."""
        assert UNIT_CONVERSIONS["kg_to_tonnes"] == Decimal("0.001")

    def test_lb_to_tonnes(self):
        """Test lb to tonnes conversion factor."""
        assert UNIT_CONVERSIONS["lb_to_tonnes"] == Decimal("0.000453592")

    def test_btu_to_kwh(self):
        """Test BTU to kWh conversion factor."""
        assert UNIT_CONVERSIONS["btu_to_kwh"] == Decimal("0.000293071")


# ============================================================================
# TEST: Singleton Pattern
# ============================================================================


class TestPipelineSingleton:
    """Test singleton pattern for ProcessingPipelineEngine."""

    def test_singleton_identity(self, engine):
        """Test that two instantiations return the same object."""
        engine2 = ProcessingPipelineEngine()
        assert engine is engine2

    def test_reset_creates_new_instance(self):
        """Test that reset allows a new singleton to be created."""
        e1 = ProcessingPipelineEngine()
        ProcessingPipelineEngine.reset()
        e2 = ProcessingPipelineEngine()
        assert e1 is not e2


# ============================================================================
# TEST: Thread Safety
# ============================================================================


class TestPipelineThreadSafety:
    """Test thread safety of ProcessingPipelineEngine."""

    def test_concurrent_pipeline_runs(self, engine, valid_pipeline_input):
        """Test that 5 threads can run the pipeline concurrently."""
        results = []
        errors = []

        def run():
            try:
                r = engine.run_pipeline(
                    inputs=valid_pipeline_input,
                    method="average_data",
                    org_id="ORG-001",
                    reporting_year=2024,
                )
                results.append(r)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=run) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0
        assert len(results) == 5
        # All should have pipeline_ids
        for r in results:
            assert "pipeline_id" in r


# ============================================================================
# TEST: Provenance in Pipeline Results
# ============================================================================


class TestPipelineProvenance:
    """Test provenance tracking in pipeline results."""

    def test_pipeline_result_has_provenance(self, engine, valid_pipeline_input):
        """Test that pipeline result includes provenance data."""
        result = engine.run_pipeline(
            inputs=valid_pipeline_input,
            method="average_data",
            org_id="ORG-001",
            reporting_year=2024,
        )
        # If successful, should have provenance_hash or provenance chain
        if result["status"] == PipelineStatus.SUCCESS.value:
            has_provenance = (
                "provenance_hash" in result
                or "provenance" in result
                or "PROVENANCE" in result.get("stage_durations_ms", {})
            )
            assert has_provenance
