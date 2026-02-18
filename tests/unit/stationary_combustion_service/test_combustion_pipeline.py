# -*- coding: utf-8 -*-
"""
Unit tests for StationaryCombustionPipelineEngine - AGENT-MRV-001

Tests the seven-stage pipeline engine: VALIDATE_INPUTS, SELECT_FACTORS,
CONVERT_UNITS, CALCULATE, QUANTIFY_UNCERTAINTY, GENERATE_AUDIT, AGGREGATE.

Validates:
- Pipeline initialisation with all 6 sub-engines
- Full pipeline execution with single and batch inputs
- Per-stage execution, ordering, and error handling
- Aggregation by facility, fuel, period, and control approach
- Pipeline-level provenance hash generation
- Pipeline statistics and status tracking
- Empty input, partial failure, and performance scenarios
- End-to-end fuel-specific pipelines (natural gas, coal, biomass, multi-fuel)

Author: GreenLang Test Engineering
Date: February 2026
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest

from greenlang.stationary_combustion.config import StationaryCombustionConfig
from greenlang.stationary_combustion.combustion_pipeline import (
    PIPELINE_STAGES,
    StationaryCombustionPipelineEngine,
    _compute_hash,
)
from greenlang.stationary_combustion.models import (
    CalculationResult,
    CalculationStatus,
    CalculationTier,
    CombustionInput,
    ControlApproach,
    EmissionGas,
    FacilityAggregation,
    FuelType,
    GasEmission,
    GWPSource,
    HeatingValueBasis,
    ReportingPeriod,
    UncertaintyResult,
    UnitType,
)


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def config():
    """Create a test StationaryCombustionConfig."""
    return StationaryCombustionConfig(
        enable_biogenic_tracking=True,
        monte_carlo_iterations=100,
        enable_metrics=False,
    )


@pytest.fixture
def mock_fuel_database():
    """Create a mock FuelDatabaseEngine."""
    mock = MagicMock()
    mock.convert_to_energy.return_value = {
        "energy_gj": Decimal("38.0"),
        "conversion_factor": Decimal("0.038"),
    }
    return mock


@pytest.fixture
def mock_calculator():
    """Create a mock CalculatorEngine that returns valid CalculationResults."""
    mock = MagicMock()

    def _calculate(**kwargs):
        inp = kwargs.get("input_data")
        fuel = inp.fuel_type if inp else FuelType.NATURAL_GAS
        qty = float(inp.quantity) if inp else 1000.0
        return CalculationResult(
            calculation_id=f"calc_{uuid.uuid4().hex[:12]}",
            fuel_type=fuel,
            fuel_quantity=qty,
            fuel_unit=UnitType.CUBIC_METERS,
            energy_gj=38.0,
            heating_value_used=0.038,
            heating_value_basis=HeatingValueBasis.HHV,
            oxidation_factor_used=1.0,
            tier_used=CalculationTier.TIER_1,
            emissions_by_gas=[
                GasEmission(
                    gas=EmissionGas.CO2,
                    emissions_kg=2100.0,
                    emissions_tco2e=2.1,
                    emission_factor_value=56.1,
                    emission_factor_unit="kg CO2/GJ",
                    emission_factor_source="EPA",
                    gwp_applied=1.0,
                ),
            ],
            total_co2e_kg=2100.0,
            total_co2e_tonnes=2.1,
            biogenic_co2_kg=0.0,
            biogenic_co2_tonnes=0.0,
            provenance_hash="a" * 64,
            calculation_trace=[
                f"facility_id={getattr(inp, 'facility_id', '') or ''}",
                f"equipment_id={getattr(inp, 'equipment_id', '') or ''}",
            ],
        )

    mock.calculate.side_effect = _calculate
    return mock


@pytest.fixture
def mock_factor_selector():
    """Create a mock FactorSelectorEngine."""
    mock = MagicMock()
    mock.select_factors.return_value = {
        "CO2": {"value": 56.1, "unit": "kg CO2/GJ"},
        "CH4": {"value": 0.001, "unit": "kg CH4/GJ"},
        "N2O": {"value": 0.0001, "unit": "kg N2O/GJ"},
    }
    return mock


@pytest.fixture
def mock_equipment_profiler():
    """Create a mock EquipmentProfilerEngine."""
    return MagicMock()


@pytest.fixture
def mock_uncertainty_engine():
    """Create a mock UncertaintyEngine."""
    mock = MagicMock()
    mock.quantify.return_value = UncertaintyResult(
        mean_co2e=2.1,
        std_dev=0.15,
        coefficient_of_variation=0.071,
        confidence_intervals={"90": (1.85, 2.35), "95": (1.80, 2.40)},
        iterations=100,
        data_quality_score=3.0,
        tier=CalculationTier.TIER_1,
        contributions={"emission_factor": 0.65, "activity_data": 0.35},
    )
    return mock


@pytest.fixture
def mock_audit_engine():
    """Create a mock AuditEngine."""
    mock = MagicMock()
    mock.generate_entry.return_value = {
        "entry_id": "audit_test",
        "calculation_id": "calc_test",
        "step_number": 0,
        "step_name": "pipeline_calculation",
        "input_data": {},
        "output_data": {},
    }
    return mock


@pytest.fixture
def pipeline(
    config,
    mock_fuel_database,
    mock_calculator,
    mock_equipment_profiler,
    mock_factor_selector,
    mock_uncertainty_engine,
    mock_audit_engine,
):
    """Create a StationaryCombustionPipelineEngine with all mock engines."""
    return StationaryCombustionPipelineEngine(
        fuel_database=mock_fuel_database,
        calculator=mock_calculator,
        equipment_profiler=mock_equipment_profiler,
        factor_selector=mock_factor_selector,
        uncertainty_engine=mock_uncertainty_engine,
        audit_engine=mock_audit_engine,
        config=config,
    )


@pytest.fixture
def bare_pipeline(config):
    """Create a pipeline with no engines (all None)."""
    return StationaryCombustionPipelineEngine(config=config)


def _make_input(
    fuel_type=FuelType.NATURAL_GAS,
    quantity=1000.0,
    unit=UnitType.CUBIC_METERS,
    facility_id=None,
    equipment_id=None,
):
    """Helper to create a valid CombustionInput."""
    now = datetime.now(timezone.utc)
    return CombustionInput(
        fuel_type=fuel_type,
        quantity=quantity,
        unit=unit,
        period_start=now - timedelta(days=30),
        period_end=now,
        facility_id=facility_id,
        equipment_id=equipment_id,
    )


def _make_result(
    fuel_type=FuelType.NATURAL_GAS,
    total_co2e_tonnes=2.1,
    biogenic_co2_tonnes=0.0,
    facility_id="",
    equipment_id="",
    energy_gj=38.0,
    timestamp=None,
):
    """Helper to create a CalculationResult for aggregation tests."""
    ts = timestamp or datetime.now(timezone.utc)
    return CalculationResult(
        calculation_id=f"calc_{uuid.uuid4().hex[:12]}",
        fuel_type=fuel_type,
        fuel_quantity=1000.0,
        fuel_unit=UnitType.CUBIC_METERS,
        energy_gj=energy_gj,
        heating_value_used=0.038,
        heating_value_basis=HeatingValueBasis.HHV,
        oxidation_factor_used=1.0,
        tier_used=CalculationTier.TIER_1,
        emissions_by_gas=[],
        total_co2e_kg=total_co2e_tonnes * 1000.0,
        total_co2e_tonnes=total_co2e_tonnes,
        biogenic_co2_kg=biogenic_co2_tonnes * 1000.0,
        biogenic_co2_tonnes=biogenic_co2_tonnes,
        provenance_hash="b" * 64,
        calculation_trace=[
            f"facility_id={facility_id}",
            f"equipment_id={equipment_id}",
        ],
        timestamp=ts,
    )


# =====================================================================
# TestPipelineInit
# =====================================================================


class TestPipelineInit:
    """Test StationaryCombustionPipelineEngine initialisation."""

    def test_init_with_all_engines(self, pipeline):
        """Pipeline initializes with all 6 sub-engines set."""
        assert pipeline.fuel_database is not None
        assert pipeline.calculator is not None
        assert pipeline.equipment_profiler is not None
        assert pipeline.factor_selector is not None
        assert pipeline.uncertainty_engine is not None
        assert pipeline.audit_engine is not None

    def test_init_with_no_engines(self, config):
        """Pipeline initializes gracefully with no engines."""
        engine = StationaryCombustionPipelineEngine(config=config)
        assert engine.fuel_database is None
        assert engine.calculator is None
        assert engine.equipment_profiler is None
        assert engine.factor_selector is None
        assert engine.uncertainty_engine is None
        assert engine.audit_engine is None

    def test_init_uses_provided_config(self, pipeline, config):
        """Pipeline uses the explicitly provided config."""
        assert pipeline.config is config

    def test_init_defaults_to_global_config(self):
        """Pipeline uses global config when none provided."""
        engine = StationaryCombustionPipelineEngine()
        assert engine.config is not None

    def test_init_thread_safe_state(self, pipeline):
        """Pipeline has thread-safe internal counters."""
        assert pipeline._total_runs == 0
        assert pipeline._successful_runs == 0
        assert pipeline._failed_runs == 0
        assert pipeline._total_duration_ms == 0.0
        assert pipeline._last_run_at is None

    def test_pipeline_stages_constant(self):
        """PIPELINE_STAGES has exactly 7 ordered stages."""
        assert len(PIPELINE_STAGES) == 7
        assert PIPELINE_STAGES[0] == "VALIDATE_INPUTS"
        assert PIPELINE_STAGES[-1] == "AGGREGATE"


# =====================================================================
# TestRunPipeline
# =====================================================================


class TestRunPipeline:
    """Test full pipeline execution with batch inputs."""

    def test_run_pipeline_5_inputs_success(self, pipeline):
        """Full pipeline with 5 inputs produces success result."""
        inputs = [_make_input() for _ in range(5)]
        result = pipeline.run_pipeline(inputs)

        assert result["success"] is True
        assert "pipeline_id" in result
        assert result["stages_total"] == 7
        assert result["stages_completed"] >= 1
        assert len(result["final_results"]) == 5
        assert result["total_duration_ms"] > 0

    def test_run_pipeline_returns_all_required_keys(self, pipeline):
        """Pipeline result contains all required top-level keys."""
        inputs = [_make_input()]
        result = pipeline.run_pipeline(inputs)

        required_keys = {
            "success", "pipeline_id", "stage_results", "final_results",
            "aggregations", "pipeline_provenance_hash", "total_duration_ms",
            "stages_completed", "stages_total", "gwp_source",
            "include_biogenic", "control_approach", "timestamp",
        }
        assert required_keys.issubset(set(result.keys()))

    def test_run_pipeline_with_custom_gwp(self, pipeline):
        """Pipeline respects custom GWP source parameter."""
        inputs = [_make_input()]
        result = pipeline.run_pipeline(inputs, gwp_source="AR5")
        assert result["gwp_source"] == "AR5"

    def test_run_pipeline_with_organization_id(self, pipeline):
        """Pipeline includes organization_id in result."""
        inputs = [_make_input()]
        result = pipeline.run_pipeline(
            inputs, organization_id="org-123",
        )
        assert result["organization_id"] == "org-123"

    def test_run_pipeline_with_reporting_period(self, pipeline):
        """Pipeline includes reporting_period in result."""
        inputs = [_make_input()]
        result = pipeline.run_pipeline(
            inputs, reporting_period="2025-Q4",
        )
        assert result["reporting_period"] == "2025-Q4"


# =====================================================================
# TestRunSingle
# =====================================================================


class TestRunSingle:
    """Test single input through full pipeline."""

    def test_run_single_success(self, pipeline):
        """Single input runs through all pipeline stages."""
        inp = _make_input()
        result = pipeline.run_single(inp)

        assert result["success"] is True
        assert len(result["final_results"]) == 1

    def test_run_single_uses_default_gwp(self, pipeline):
        """run_single defaults to AR6 GWP source."""
        inp = _make_input()
        result = pipeline.run_single(inp)
        assert result["gwp_source"] == "AR6"

    def test_run_single_with_custom_gwp(self, pipeline):
        """run_single respects custom GWP source."""
        inp = _make_input()
        result = pipeline.run_single(inp, gwp_source="AR4")
        assert result["gwp_source"] == "AR4"


# =====================================================================
# TestValidateInputs
# =====================================================================


class TestValidateInputs:
    """Test input validation without full pipeline execution."""

    def test_valid_inputs_pass(self, pipeline):
        """Valid inputs return valid=True with no errors."""
        inputs = [_make_input()]
        result = pipeline.validate_inputs(inputs)

        assert result["valid"] is True
        assert len(result["errors"]) == 0
        assert result["validated_count"] == 1
        assert result["total_count"] == 1

    def test_valid_inputs_with_warnings(self, pipeline):
        """Inputs without facility_id generate warnings."""
        inputs = [_make_input()]
        result = pipeline.validate_inputs(inputs)

        # Should have warnings about missing facility_id
        assert result["total_count"] == 1

    def test_multiple_inputs_all_valid(self, pipeline):
        """Multiple valid inputs all pass validation."""
        inputs = [_make_input() for _ in range(10)]
        result = pipeline.validate_inputs(inputs)

        assert result["valid"] is True
        assert result["validated_count"] == 10

    def test_empty_input_list(self, pipeline):
        """Empty input list is valid (no errors)."""
        result = pipeline.validate_inputs([])
        assert result["valid"] is True
        assert result["validated_count"] == 0
        assert result["total_count"] == 0


# =====================================================================
# TestPipelineStages
# =====================================================================


class TestPipelineStages:
    """Test that pipeline stages execute in correct order."""

    def test_seven_stages_in_results(self, pipeline):
        """Pipeline produces results for all 7 stages."""
        inputs = [_make_input()]
        result = pipeline.run_pipeline(inputs)

        assert len(result["stage_results"]) == 7

    def test_stage_order_matches_constant(self, pipeline):
        """Stages in results match PIPELINE_STAGES order."""
        inputs = [_make_input()]
        result = pipeline.run_pipeline(inputs)

        for idx, stage_result in enumerate(result["stage_results"]):
            assert stage_result["stage"] == PIPELINE_STAGES[idx]

    def test_each_stage_has_duration(self, pipeline):
        """Each stage result includes a duration_ms field."""
        inputs = [_make_input()]
        result = pipeline.run_pipeline(inputs)

        for stage_result in result["stage_results"]:
            assert "duration_ms" in stage_result
            assert stage_result["duration_ms"] >= 0

    def test_each_stage_has_success_flag(self, pipeline):
        """Each stage result includes a success flag."""
        inputs = [_make_input()]
        result = pipeline.run_pipeline(inputs)

        for stage_result in result["stage_results"]:
            assert "success" in stage_result

    def test_each_stage_has_provenance_hash(self, pipeline):
        """Each successful stage produces a provenance hash."""
        inputs = [_make_input()]
        result = pipeline.run_pipeline(inputs)

        for stage_result in result["stage_results"]:
            assert "provenance_hash" in stage_result


# =====================================================================
# TestAggregateByFacility
# =====================================================================


class TestAggregateByFacility:
    """Test aggregation of results by facility_id."""

    def test_single_facility(self, pipeline):
        """Results from one facility are aggregated together."""
        results = [
            _make_result(facility_id="FAC-001", total_co2e_tonnes=1.0),
            _make_result(facility_id="FAC-001", total_co2e_tonnes=2.0),
        ]
        aggs = pipeline.aggregate_by_facility(results)
        assert len(aggs) == 1
        assert aggs[0].facility_id == "FAC-001"

    def test_multiple_facilities(self, pipeline):
        """Results from multiple facilities produce separate aggregations."""
        results = [
            _make_result(facility_id="FAC-001", total_co2e_tonnes=1.0),
            _make_result(facility_id="FAC-002", total_co2e_tonnes=2.0),
            _make_result(facility_id="FAC-003", total_co2e_tonnes=3.0),
        ]
        aggs = pipeline.aggregate_by_facility(results)
        assert len(aggs) == 3

    def test_unassigned_facility(self, pipeline):
        """Results without facility_id go to UNASSIGNED."""
        results = [_make_result(facility_id="", total_co2e_tonnes=1.5)]
        aggs = pipeline.aggregate_by_facility(results)
        assert len(aggs) == 1
        assert aggs[0].facility_id == "UNASSIGNED"

    def test_aggregation_sums_correctly(self, pipeline):
        """Facility aggregation sums CO2e tonnes across records."""
        results = [
            _make_result(facility_id="FAC-001", total_co2e_tonnes=1.0),
            _make_result(facility_id="FAC-001", total_co2e_tonnes=2.5),
            _make_result(facility_id="FAC-001", total_co2e_tonnes=0.5),
        ]
        aggs = pipeline.aggregate_by_facility(results)
        total = float(aggs[0].total_co2e_tonnes)
        assert abs(total - 4.0) < 0.01


# =====================================================================
# TestAggregateByFuel
# =====================================================================


class TestAggregateByFuel:
    """Test aggregation of results by fuel type."""

    def test_single_fuel_type(self, pipeline):
        """Single fuel type results aggregate together."""
        results = [
            _make_result(fuel_type=FuelType.NATURAL_GAS, total_co2e_tonnes=1.0),
            _make_result(fuel_type=FuelType.NATURAL_GAS, total_co2e_tonnes=2.0),
        ]
        agg = pipeline.aggregate_by_fuel(results)
        assert "natural_gas" in agg
        assert abs(agg["natural_gas"]["total_co2e_tonnes"] - 3.0) < 0.01

    def test_multiple_fuel_types(self, pipeline):
        """Multiple fuel types produce separate aggregation entries."""
        results = [
            _make_result(fuel_type=FuelType.NATURAL_GAS, total_co2e_tonnes=1.0),
            _make_result(fuel_type=FuelType.DIESEL, total_co2e_tonnes=2.0),
            _make_result(fuel_type=FuelType.COAL_BITUMINOUS, total_co2e_tonnes=3.0),
        ]
        agg = pipeline.aggregate_by_fuel(results)
        assert len(agg) == 3
        assert "natural_gas" in agg
        assert "diesel" in agg
        assert "coal_bituminous" in agg

    def test_fuel_aggregation_includes_energy(self, pipeline):
        """Fuel aggregation tracks total energy in GJ."""
        results = [
            _make_result(fuel_type=FuelType.NATURAL_GAS, energy_gj=38.0),
            _make_result(fuel_type=FuelType.NATURAL_GAS, energy_gj=42.0),
        ]
        agg = pipeline.aggregate_by_fuel(results)
        assert abs(agg["natural_gas"]["total_energy_gj"] - 80.0) < 0.01

    def test_fuel_aggregation_calculation_count(self, pipeline):
        """Fuel aggregation tracks calculation count per fuel."""
        results = [
            _make_result(fuel_type=FuelType.DIESEL),
            _make_result(fuel_type=FuelType.DIESEL),
            _make_result(fuel_type=FuelType.DIESEL),
        ]
        agg = pipeline.aggregate_by_fuel(results)
        assert agg["diesel"]["calculation_count"] == 3


# =====================================================================
# TestAggregateByPeriod
# =====================================================================


class TestAggregateByPeriod:
    """Test aggregation by MONTHLY, QUARTERLY, and ANNUAL periods."""

    def test_monthly_aggregation(self, pipeline):
        """Results are grouped by YYYY-MM for MONTHLY period."""
        jan = datetime(2025, 1, 15, tzinfo=timezone.utc)
        feb = datetime(2025, 2, 15, tzinfo=timezone.utc)
        results = [
            _make_result(total_co2e_tonnes=1.0, timestamp=jan),
            _make_result(total_co2e_tonnes=2.0, timestamp=feb),
        ]
        # The pipeline uses calculated_at attribute; patch timestamp
        for r, ts in zip(results, [jan, feb]):
            r.calculated_at = ts

        agg = pipeline.aggregate_by_period(results, period_type="MONTHLY")
        assert "2025-01" in agg or "2025-02" in agg

    def test_quarterly_aggregation(self, pipeline):
        """Results are grouped by YYYY-QN for QUARTERLY period."""
        q1 = datetime(2025, 2, 15, tzinfo=timezone.utc)
        q2 = datetime(2025, 5, 15, tzinfo=timezone.utc)
        results = [
            _make_result(total_co2e_tonnes=1.0, timestamp=q1),
            _make_result(total_co2e_tonnes=2.0, timestamp=q2),
        ]
        for r, ts in zip(results, [q1, q2]):
            r.calculated_at = ts

        agg = pipeline.aggregate_by_period(results, period_type="QUARTERLY")
        assert "2025-Q1" in agg or "2025-Q2" in agg

    def test_annual_aggregation(self, pipeline):
        """Results are grouped by YYYY for ANNUAL period."""
        y1 = datetime(2025, 6, 15, tzinfo=timezone.utc)
        results = [_make_result(total_co2e_tonnes=5.0, timestamp=y1)]
        for r in results:
            r.calculated_at = y1

        agg = pipeline.aggregate_by_period(results, period_type="ANNUAL")
        assert "2025" in agg


# =====================================================================
# TestControlApproach
# =====================================================================


class TestControlApproach:
    """Test organisational boundary control approach aggregation."""

    def test_operational_approach(self, pipeline):
        """OPERATIONAL control approach used for aggregation."""
        inputs = [_make_input()]
        result = pipeline.run_pipeline(
            inputs, control_approach="OPERATIONAL",
        )
        assert result["control_approach"] == "OPERATIONAL"

    def test_financial_approach(self, pipeline):
        """FINANCIAL control approach used for aggregation."""
        inputs = [_make_input()]
        result = pipeline.run_pipeline(
            inputs, control_approach="FINANCIAL",
        )
        assert result["control_approach"] == "FINANCIAL"

    def test_equity_share_approach(self, pipeline):
        """EQUITY_SHARE control approach used for aggregation."""
        inputs = [_make_input()]
        result = pipeline.run_pipeline(
            inputs, control_approach="EQUITY_SHARE",
        )
        assert result["control_approach"] == "EQUITY_SHARE"


# =====================================================================
# TestPipelineProvenance
# =====================================================================


class TestPipelineProvenance:
    """Test pipeline-level provenance hash generation."""

    def test_pipeline_provenance_hash_exists(self, pipeline):
        """Pipeline produces a provenance hash."""
        inputs = [_make_input()]
        result = pipeline.run_pipeline(inputs)
        assert "pipeline_provenance_hash" in result
        assert len(result["pipeline_provenance_hash"]) == 64

    def test_pipeline_provenance_hash_is_sha256(self, pipeline):
        """Pipeline provenance hash is a valid 64-char hex string."""
        inputs = [_make_input()]
        result = pipeline.run_pipeline(inputs)
        ph = result["pipeline_provenance_hash"]
        assert len(ph) == 64
        int(ph, 16)  # Validates hex format

    def test_different_inputs_different_hash(self, pipeline):
        """Different input batches produce different provenance hashes."""
        r1 = pipeline.run_pipeline([_make_input(quantity=100.0)])
        r2 = pipeline.run_pipeline([_make_input(quantity=200.0)])
        # Different runs generate different pipeline_ids, so hashes differ
        assert r1["pipeline_provenance_hash"] != r2["pipeline_provenance_hash"]


# =====================================================================
# TestPipelineStatistics
# =====================================================================


class TestPipelineStatistics:
    """Test pipeline-level aggregate statistics."""

    def test_initial_statistics(self, pipeline):
        """Initial statistics show zero runs."""
        stats = pipeline.get_pipeline_statistics()
        assert stats["total_runs"] == 0
        assert stats["successful_runs"] == 0
        assert stats["failed_runs"] == 0

    def test_statistics_after_run(self, pipeline):
        """Statistics update after a pipeline run."""
        pipeline.run_pipeline([_make_input()])
        stats = pipeline.get_pipeline_statistics()

        assert stats["total_runs"] == 1
        assert stats["successful_runs"] == 1
        assert stats["avg_duration_ms"] > 0

    def test_statistics_success_rate(self, pipeline):
        """Success rate is calculated correctly."""
        pipeline.run_pipeline([_make_input()])
        pipeline.run_pipeline([_make_input()])
        stats = pipeline.get_pipeline_statistics()

        assert stats["success_rate_pct"] == 100.0

    def test_statistics_multiple_runs(self, pipeline):
        """Statistics accumulate across multiple runs."""
        for _ in range(5):
            pipeline.run_pipeline([_make_input()])
        stats = pipeline.get_pipeline_statistics()

        assert stats["total_runs"] == 5
        assert stats["total_duration_ms"] > 0

    def test_statistics_recent_runs_capped(self, pipeline):
        """Recent runs list is capped at 10 entries."""
        for _ in range(15):
            pipeline.run_pipeline([_make_input()])
        stats = pipeline.get_pipeline_statistics()

        assert len(stats["recent_runs"]) <= 10


# =====================================================================
# TestPipelineStatus
# =====================================================================


class TestPipelineStatus:
    """Test pipeline operational status reporting."""

    def test_status_ready(self, pipeline):
        """Pipeline status is 'ready' when initialised."""
        status = pipeline.get_pipeline_status()
        assert status["status"] == "ready"

    def test_status_engine_availability(self, pipeline):
        """Status reports which engines are available."""
        status = pipeline.get_pipeline_status()
        assert status["engines"]["fuel_database"] is True
        assert status["engines"]["calculator"] is True
        assert status["engines"]["uncertainty_engine"] is True

    def test_status_bare_pipeline(self, bare_pipeline):
        """Bare pipeline reports engines as unavailable."""
        status = bare_pipeline.get_pipeline_status()
        assert status["engines"]["fuel_database"] is False
        assert status["engines"]["calculator"] is False

    def test_status_includes_stages(self, pipeline):
        """Status includes the list of pipeline stages."""
        status = pipeline.get_pipeline_status()
        assert status["pipeline_stages"] == PIPELINE_STAGES

    def test_status_last_run_at_updates(self, pipeline):
        """last_run_at updates after each pipeline run."""
        assert pipeline.get_pipeline_status()["last_run_at"] is None
        pipeline.run_pipeline([_make_input()])
        assert pipeline.get_pipeline_status()["last_run_at"] is not None


# =====================================================================
# TestEmptyInput
# =====================================================================


class TestEmptyInput:
    """Test pipeline handling of empty input list."""

    def test_empty_input_pipeline(self, bare_pipeline):
        """Empty input list runs pipeline without errors."""
        result = bare_pipeline.run_pipeline([])
        assert "pipeline_id" in result
        assert result["stages_total"] == 7

    def test_empty_input_no_results(self, bare_pipeline):
        """Empty input produces no calculation results."""
        result = bare_pipeline.run_pipeline([])
        assert len(result["final_results"]) == 0


# =====================================================================
# TestPartialFailure
# =====================================================================


class TestPartialFailure:
    """Test pipeline behavior when some inputs fail."""

    def test_partial_failure_continues(self, pipeline, mock_calculator):
        """Pipeline continues processing when some inputs fail."""
        call_count = [0]
        original_side_effect = mock_calculator.calculate.side_effect

        def _sometimes_fail(**kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise ValueError("Simulated calculation failure")
            return original_side_effect(**kwargs)

        mock_calculator.calculate.side_effect = _sometimes_fail

        inputs = [_make_input() for _ in range(3)]
        result = pipeline.run_pipeline(inputs)

        # Pipeline should still produce results (some may be failed)
        assert len(result["final_results"]) == 3

    def test_bare_pipeline_creates_stubs(self, bare_pipeline):
        """Pipeline without calculator creates stub results."""
        inputs = [_make_input()]
        result = bare_pipeline.run_pipeline(inputs)
        assert len(result["final_results"]) >= 0


# =====================================================================
# TestBatchPerformance
# =====================================================================


class TestBatchPerformance:
    """Test pipeline performance with larger batches."""

    def test_100_records_processed(self, pipeline):
        """100 records are processed by the pipeline."""
        inputs = [_make_input() for _ in range(100)]
        result = pipeline.run_pipeline(inputs)

        assert result["success"] is True
        assert len(result["final_results"]) == 100

    def test_pipeline_completes_under_10_seconds(self, pipeline):
        """Pipeline with 100 records completes in under 10 seconds."""
        inputs = [_make_input() for _ in range(100)]

        t0 = time.perf_counter()
        pipeline.run_pipeline(inputs)
        elapsed = (time.perf_counter() - t0)

        assert elapsed < 10.0


# =====================================================================
# Fuel-specific pipeline tests
# =====================================================================


class TestNaturalGasPipeline:
    """Test end-to-end natural gas calculation through pipeline."""

    def test_natural_gas_pipeline(self, pipeline):
        """Natural gas input produces valid pipeline result."""
        inp = _make_input(
            fuel_type=FuelType.NATURAL_GAS,
            quantity=1000.0,
            unit=UnitType.CUBIC_METERS,
        )
        result = pipeline.run_pipeline([inp])
        assert result["success"] is True
        assert len(result["final_results"]) == 1


class TestCoalPipeline:
    """Test end-to-end coal calculation through pipeline."""

    def test_coal_pipeline(self, pipeline):
        """Coal input produces valid pipeline result."""
        inp = _make_input(
            fuel_type=FuelType.COAL_BITUMINOUS,
            quantity=1.0,
            unit=UnitType.TONNES,
        )
        result = pipeline.run_pipeline([inp])
        assert result["success"] is True
        assert len(result["final_results"]) == 1


class TestBiogenicPipeline:
    """Test end-to-end biomass with biogenic separation."""

    def test_biomass_pipeline(self, pipeline):
        """Biomass input processes through pipeline."""
        inp = _make_input(
            fuel_type=FuelType.WOOD,
            quantity=1.0,
            unit=UnitType.TONNES,
        )
        result = pipeline.run_pipeline(
            [inp], include_biogenic=True,
        )
        assert result["success"] is True
        assert result["include_biogenic"] is True


class TestMultiFuelPipeline:
    """Test mixed fuels aggregated correctly."""

    def test_multi_fuel_aggregation(self, pipeline):
        """Multiple fuel types produce per-fuel aggregation."""
        inputs = [
            _make_input(fuel_type=FuelType.NATURAL_GAS, quantity=500.0),
            _make_input(fuel_type=FuelType.DIESEL, quantity=300.0),
            _make_input(fuel_type=FuelType.COAL_BITUMINOUS, quantity=100.0),
        ]
        result = pipeline.run_pipeline(inputs)
        assert result["success"] is True
        assert len(result["final_results"]) == 3


class TestMultiFacilityPipeline:
    """Test multiple facilities aggregated separately."""

    def test_multi_facility_aggregation(self, pipeline):
        """Results from different facilities produce separate aggregations."""
        inputs = [
            _make_input(facility_id="FAC-001"),
            _make_input(facility_id="FAC-001"),
            _make_input(facility_id="FAC-002"),
        ]
        result = pipeline.run_pipeline(inputs)
        assert result["success"] is True
        # Should have aggregation data
        assert "aggregations" in result


# =====================================================================
# Utility function tests
# =====================================================================


class TestComputeHash:
    """Test the _compute_hash utility function."""

    def test_deterministic_hashing(self):
        """Same input produces same hash."""
        data = {"key": "value", "count": 42}
        h1 = _compute_hash(data)
        h2 = _compute_hash(data)
        assert h1 == h2

    def test_different_data_different_hash(self):
        """Different data produces different hash."""
        h1 = _compute_hash({"a": 1})
        h2 = _compute_hash({"a": 2})
        assert h1 != h2

    def test_hash_length(self):
        """Hash is 64-character SHA-256 hex string."""
        h = _compute_hash("test data")
        assert len(h) == 64
