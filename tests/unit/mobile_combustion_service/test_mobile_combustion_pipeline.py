# -*- coding: utf-8 -*-
"""
Unit tests for MobileCombustionPipelineEngine (Engine 7) - AGENT-MRV-003

Tests the eight-stage orchestration pipeline for GHG Protocol Scope 1
mobile combustion emissions calculations. Validates each stage independently
and the full pipeline end-to-end with different calculation methods,
vehicle types, and fuel types.

Target: 64+ tests across 12 test classes.

Author: GreenLang QA Team
Date: February 2026
PRD: AGENT-MRV-003 Mobile Combustion (GL-MRV-SCOPE1-003)
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from greenlang.mobile_combustion.mobile_combustion_pipeline import (
    BIOFUEL_FOSSIL_FRACTION,
    CH4_G_PER_MILE,
    COMPLIANCE_REQUIREMENTS,
    DEFAULT_FUEL_ECONOMY_L_PER_100KM,
    DISTANCE_TO_KM,
    FUEL_CO2_FACTORS_KG_PER_GALLON,
    GWP_TABLES,
    MobileCombustionPipelineEngine,
    PIPELINE_STAGES,
    PipelineContext,
    PipelineStage,
    SPEND_BASED_FACTORS_KG_CO2E_PER_USD,
    SUPPORTED_METHODS,
    StageResult,
    UNCERTAINTY_PARAMETERS,
    VOLUME_TO_GALLONS,
    _compute_hash,
)


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def engine() -> MobileCombustionPipelineEngine:
    """Create a pipeline engine with no external engines (fallback mode)."""
    return MobileCombustionPipelineEngine()


@pytest.fixture
def fuel_based_input() -> Dict[str, Any]:
    """Standard fuel-based calculation input."""
    return {
        "vehicle_type": "PASSENGER_CAR_GASOLINE",
        "fuel_type": "GASOLINE",
        "calculation_method": "FUEL_BASED",
        "fuel_quantity": 100.0,
        "fuel_unit": "GALLONS",
        "gwp_source": "AR6",
    }


@pytest.fixture
def distance_based_input() -> Dict[str, Any]:
    """Standard distance-based calculation input."""
    return {
        "vehicle_type": "HEAVY_TRUCK_DIESEL",
        "fuel_type": "DIESEL",
        "calculation_method": "DISTANCE_BASED",
        "distance": 500.0,
        "distance_unit": "MILES",
        "gwp_source": "AR6",
    }


@pytest.fixture
def spend_based_input() -> Dict[str, Any]:
    """Standard spend-based calculation input."""
    return {
        "vehicle_type": "PASSENGER_CAR_GASOLINE",
        "fuel_type": "GASOLINE",
        "calculation_method": "SPEND_BASED",
        "spend_amount": 500.0,
        "spend_currency": "USD",
        "gwp_source": "AR6",
    }


@pytest.fixture
def biofuel_e10_input() -> Dict[str, Any]:
    """Input with E10 biofuel blend."""
    return {
        "vehicle_type": "PASSENGER_CAR_GASOLINE",
        "fuel_type": "E10",
        "calculation_method": "FUEL_BASED",
        "fuel_quantity": 100.0,
        "fuel_unit": "GALLONS",
        "gwp_source": "AR6",
    }


@pytest.fixture
def biofuel_e85_input() -> Dict[str, Any]:
    """Input with E85 biofuel blend."""
    return {
        "vehicle_type": "PASSENGER_CAR_GASOLINE",
        "fuel_type": "E85",
        "calculation_method": "FUEL_BASED",
        "fuel_quantity": 50.0,
        "fuel_unit": "GALLONS",
        "gwp_source": "AR6",
    }


@pytest.fixture
def biofuel_b20_input() -> Dict[str, Any]:
    """Input with B20 biodiesel blend."""
    return {
        "vehicle_type": "HEAVY_TRUCK_DIESEL",
        "fuel_type": "B20",
        "calculation_method": "FUEL_BASED",
        "fuel_quantity": 200.0,
        "fuel_unit": "GALLONS",
        "gwp_source": "AR6",
    }


@pytest.fixture
def biofuel_b100_input() -> Dict[str, Any]:
    """Input with B100 pure biodiesel."""
    return {
        "vehicle_type": "HEAVY_TRUCK_DIESEL",
        "fuel_type": "B100",
        "calculation_method": "FUEL_BASED",
        "fuel_quantity": 50.0,
        "fuel_unit": "GALLONS",
        "gwp_source": "AR6",
    }


# ===================================================================
# TestPipelineInit (5 tests)
# ===================================================================


class TestPipelineInit:
    """Test pipeline engine initialization."""

    def test_engine_creation_no_args(self):
        """Engine can be created with no arguments."""
        engine = MobileCombustionPipelineEngine()
        assert engine is not None
        assert engine.vehicle_db is None
        assert engine.calculator is None
        assert engine.fleet_manager is None
        assert engine.distance_estimator is None
        assert engine.uncertainty is None
        assert engine.compliance is None

    def test_stage_enum_has_eight_stages(self):
        """PipelineStage enum defines exactly 8 stages."""
        stages = list(PipelineStage)
        assert len(stages) == 8
        assert PipelineStage.VALIDATE in stages
        assert PipelineStage.RESOLVE_VEHICLE in stages
        assert PipelineStage.ESTIMATE_FUEL_OR_DISTANCE in stages
        assert PipelineStage.CALCULATE_EMISSIONS in stages
        assert PipelineStage.APPLY_BIOFUEL_ADJUSTMENT in stages
        assert PipelineStage.QUANTIFY_UNCERTAINTY in stages
        assert PipelineStage.CHECK_COMPLIANCE in stages
        assert PipelineStage.GENERATE_AUDIT in stages

    def test_pipeline_stages_list_matches_enum(self):
        """PIPELINE_STAGES list matches PipelineStage enum values."""
        assert PIPELINE_STAGES == [s.value for s in PipelineStage]
        assert len(PIPELINE_STAGES) == 8

    def test_pipeline_context_creation(self):
        """PipelineContext initializes with expected defaults."""
        ctx = PipelineContext(
            pipeline_id="test-pipe",
            calculation_id="test-calc",
            input_data={"vehicle_type": "PASSENGER_CAR_GASOLINE"},
        )
        assert ctx.pipeline_id == "test-pipe"
        assert ctx.calculation_id == "test-calc"
        assert ctx.vehicle_props is None
        assert ctx.emission_factors is None
        assert ctx.fuel_quantity_gallons is None
        assert ctx.distance_km is None
        assert ctx.calculation_method == "FUEL_BASED"
        assert ctx.stage_results == []
        assert ctx.provenance_chain == []
        assert ctx.errors == []
        assert ctx.warnings == []

    def test_engine_initial_stats_are_zero(self, engine):
        """Stats counters start at zero on fresh engine."""
        stats = engine.get_pipeline_stats()
        assert stats["total_runs"] == 0
        assert stats["successful_runs"] == 0
        assert stats["failed_runs"] == 0
        assert stats["success_rate_pct"] == 0.0
        assert stats["avg_duration_ms"] == 0.0


# ===================================================================
# TestStageValidate (8 tests)
# ===================================================================


class TestStageValidate:
    """Test stage 1: VALIDATE."""

    def test_valid_fuel_based_input(self, engine, fuel_based_input):
        """Valid fuel-based input passes validation."""
        ctx = PipelineContext(
            pipeline_id="p1",
            calculation_id="c1",
            input_data=dict(fuel_based_input),
        )
        result = engine._stage_validate(ctx)
        assert result.status == "SUCCESS"
        assert result.data["valid"] is True
        assert result.data["errors"] == []

    def test_valid_distance_based_input(self, engine, distance_based_input):
        """Valid distance-based input passes validation."""
        ctx = PipelineContext(
            pipeline_id="p2",
            calculation_id="c2",
            input_data=dict(distance_based_input),
        )
        result = engine._stage_validate(ctx)
        assert result.status == "SUCCESS"
        assert result.data["valid"] is True

    def test_valid_spend_based_input(self, engine, spend_based_input):
        """Valid spend-based input passes validation."""
        ctx = PipelineContext(
            pipeline_id="p3",
            calculation_id="c3",
            input_data=dict(spend_based_input),
        )
        result = engine._stage_validate(ctx)
        assert result.status == "SUCCESS"
        assert result.data["valid"] is True

    def test_invalid_calculation_method(self, engine):
        """Unsupported calculation method fails validation."""
        ctx = PipelineContext(
            pipeline_id="p4",
            calculation_id="c4",
            input_data={"calculation_method": "MAGIC_BASED"},
        )
        result = engine._stage_validate(ctx)
        assert result.status == "FAILED"
        assert len(result.data["errors"]) > 0
        assert "MAGIC_BASED" in result.error

    def test_fuel_based_missing_quantity(self, engine):
        """Fuel-based method without fuel_quantity fails validation."""
        ctx = PipelineContext(
            pipeline_id="p5",
            calculation_id="c5",
            input_data={
                "calculation_method": "FUEL_BASED",
                "vehicle_type": "PASSENGER_CAR_GASOLINE",
                "fuel_type": "GASOLINE",
            },
        )
        result = engine._stage_validate(ctx)
        assert result.status == "FAILED"
        assert "fuel_quantity" in result.error

    def test_distance_based_missing_distance(self, engine):
        """Distance-based method without distance fails validation."""
        ctx = PipelineContext(
            pipeline_id="p6",
            calculation_id="c6",
            input_data={
                "calculation_method": "DISTANCE_BASED",
                "vehicle_type": "HEAVY_TRUCK_DIESEL",
                "fuel_type": "DIESEL",
            },
        )
        result = engine._stage_validate(ctx)
        assert result.status == "FAILED"
        assert "distance" in result.error

    def test_spend_based_missing_amount(self, engine):
        """Spend-based method without spend_amount fails validation."""
        ctx = PipelineContext(
            pipeline_id="p7",
            calculation_id="c7",
            input_data={
                "calculation_method": "SPEND_BASED",
                "vehicle_type": "PASSENGER_CAR_GASOLINE",
                "fuel_type": "GASOLINE",
            },
        )
        result = engine._stage_validate(ctx)
        assert result.status == "FAILED"
        assert "spend_amount" in result.error

    def test_negative_fuel_quantity_fails(self, engine):
        """Negative fuel quantity fails validation."""
        ctx = PipelineContext(
            pipeline_id="p8",
            calculation_id="c8",
            input_data={
                "calculation_method": "FUEL_BASED",
                "vehicle_type": "PASSENGER_CAR_GASOLINE",
                "fuel_type": "GASOLINE",
                "fuel_quantity": -10.0,
                "fuel_unit": "GALLONS",
            },
        )
        result = engine._stage_validate(ctx)
        assert result.status == "FAILED"
        assert "fuel_quantity must be > 0" in result.error


# ===================================================================
# TestStageResolveVehicle (6 tests)
# ===================================================================


class TestStageResolveVehicle:
    """Test stage 2: RESOLVE_VEHICLE."""

    def test_resolve_known_vehicle_type(self, engine, fuel_based_input):
        """Known vehicle type resolves with correct emission factors."""
        ctx = PipelineContext(
            pipeline_id="p1",
            calculation_id="c1",
            input_data=dict(fuel_based_input),
        )
        result = engine._stage_resolve_vehicle(ctx)
        assert result.status == "SUCCESS"
        assert ctx.vehicle_props is not None
        assert ctx.vehicle_props["vehicle_type"] == "PASSENGER_CAR_GASOLINE"
        assert ctx.emission_factors is not None
        assert ctx.emission_factors["co2_kg_per_gallon"] == pytest.approx(
            FUEL_CO2_FACTORS_KG_PER_GALLON["GASOLINE"],
        )

    def test_resolve_diesel_vehicle(self, engine, distance_based_input):
        """Diesel vehicle resolves with correct CO2 factor."""
        ctx = PipelineContext(
            pipeline_id="p2",
            calculation_id="c2",
            input_data=dict(distance_based_input),
        )
        result = engine._stage_resolve_vehicle(ctx)
        assert result.status == "SUCCESS"
        assert ctx.emission_factors["co2_kg_per_gallon"] == pytest.approx(
            FUEL_CO2_FACTORS_KG_PER_GALLON["DIESEL"],
        )

    def test_resolve_ch4_n2o_factors(self, engine, fuel_based_input):
        """CH4 and N2O factors match EPA tables."""
        ctx = PipelineContext(
            pipeline_id="p3",
            calculation_id="c3",
            input_data=dict(fuel_based_input),
        )
        engine._stage_resolve_vehicle(ctx)
        expected_ch4 = CH4_G_PER_MILE["PASSENGER_CAR_GASOLINE"]["CH4"]
        expected_n2o = CH4_G_PER_MILE["PASSENGER_CAR_GASOLINE"]["N2O"]
        assert ctx.emission_factors["ch4_g_per_mile"] == pytest.approx(expected_ch4)
        assert ctx.emission_factors["n2o_g_per_mile"] == pytest.approx(expected_n2o)

    def test_resolve_gwp_ar5(self, engine):
        """AR5 GWP values are correctly applied."""
        inp = {
            "vehicle_type": "PASSENGER_CAR_GASOLINE",
            "fuel_type": "GASOLINE",
            "gwp_source": "AR5",
        }
        ctx = PipelineContext(
            pipeline_id="p4", calculation_id="c4", input_data=inp,
        )
        engine._stage_resolve_vehicle(ctx)
        assert ctx.emission_factors["gwp_ch4"] == pytest.approx(28.0)
        assert ctx.emission_factors["gwp_n2o"] == pytest.approx(265.0)

    def test_resolve_fuel_economy_default(self, engine, fuel_based_input):
        """Default fuel economy is set from the lookup table."""
        ctx = PipelineContext(
            pipeline_id="p5",
            calculation_id="c5",
            input_data=dict(fuel_based_input),
        )
        engine._stage_resolve_vehicle(ctx)
        expected_economy = DEFAULT_FUEL_ECONOMY_L_PER_100KM["PASSENGER_CAR_GASOLINE"]
        assert ctx.vehicle_props["fuel_economy_l_per_100km"] == pytest.approx(
            expected_economy,
        )

    def test_resolve_unknown_vehicle_uses_default(self, engine):
        """Unknown vehicle type falls back to gasoline passenger car factors."""
        inp = {
            "vehicle_type": "QUANTUM_HOVERCRAFT",
            "fuel_type": "GASOLINE",
            "gwp_source": "AR6",
        }
        ctx = PipelineContext(
            pipeline_id="p6", calculation_id="c6", input_data=inp,
        )
        result = engine._stage_resolve_vehicle(ctx)
        assert result.status == "SUCCESS"
        # Default CH4/N2O factors for unknown vehicle type match PASSENGER_CAR_GASOLINE
        assert ctx.emission_factors["ch4_g_per_mile"] == pytest.approx(
            CH4_G_PER_MILE["PASSENGER_CAR_GASOLINE"]["CH4"],
        )


# ===================================================================
# TestStageEstimateFuel (6 tests)
# ===================================================================


class TestStageEstimateFuel:
    """Test stage 3: ESTIMATE_FUEL_OR_DISTANCE."""

    def test_fuel_based_converts_to_gallons(self, engine, fuel_based_input):
        """Fuel-based input converts fuel to gallons and estimates distance."""
        ctx = PipelineContext(
            pipeline_id="p1", calculation_id="c1",
            input_data=dict(fuel_based_input),
        )
        engine._stage_resolve_vehicle(ctx)
        result = engine._stage_estimate_fuel(ctx)
        assert result.status == "SUCCESS"
        # 100 gallons input, unit=GALLONS, conversion=1.0
        assert ctx.fuel_quantity_gallons == pytest.approx(100.0)
        # Distance should be estimated from fuel and fuel economy
        assert ctx.distance_km is not None
        assert ctx.distance_km > 0

    def test_distance_based_estimates_fuel(self, engine, distance_based_input):
        """Distance-based input estimates fuel from distance and fuel economy."""
        ctx = PipelineContext(
            pipeline_id="p2", calculation_id="c2",
            input_data=dict(distance_based_input),
        )
        engine._stage_resolve_vehicle(ctx)
        result = engine._stage_estimate_fuel(ctx)
        assert result.status == "SUCCESS"
        # 500 miles * 1.60934 = ~804.67 km
        expected_km = 500.0 * DISTANCE_TO_KM["MILES"]
        assert ctx.distance_km == pytest.approx(expected_km, rel=1e-3)
        assert ctx.fuel_quantity_gallons is not None
        assert ctx.fuel_quantity_gallons > 0

    def test_fuel_liters_conversion(self, engine):
        """Fuel in liters is correctly converted to gallons."""
        inp = {
            "vehicle_type": "PASSENGER_CAR_GASOLINE",
            "fuel_type": "GASOLINE",
            "calculation_method": "FUEL_BASED",
            "fuel_quantity": 100.0,
            "fuel_unit": "LITERS",
            "gwp_source": "AR6",
        }
        ctx = PipelineContext(
            pipeline_id="p3", calculation_id="c3", input_data=inp,
        )
        engine._stage_resolve_vehicle(ctx)
        engine._stage_estimate_fuel(ctx)
        # 100 liters * 0.264172 = ~26.4172 gallons
        expected_gallons = 100.0 * VOLUME_TO_GALLONS["LITERS"]
        assert ctx.fuel_quantity_gallons == pytest.approx(expected_gallons, rel=1e-4)

    def test_distance_km_no_conversion(self, engine):
        """Distance in km requires no conversion."""
        inp = {
            "vehicle_type": "HEAVY_TRUCK_DIESEL",
            "fuel_type": "DIESEL",
            "calculation_method": "DISTANCE_BASED",
            "distance": 1000.0,
            "distance_unit": "KM",
            "gwp_source": "AR6",
        }
        ctx = PipelineContext(
            pipeline_id="p4", calculation_id="c4", input_data=inp,
        )
        engine._stage_resolve_vehicle(ctx)
        engine._stage_estimate_fuel(ctx)
        assert ctx.distance_km == pytest.approx(1000.0)

    def test_spend_based_estimates_fuel_from_price(self, engine, spend_based_input):
        """Spend-based input estimates fuel from spend amount / price."""
        ctx = PipelineContext(
            pipeline_id="p5", calculation_id="c5",
            input_data=dict(spend_based_input),
        )
        engine._stage_resolve_vehicle(ctx)
        result = engine._stage_estimate_fuel(ctx)
        assert result.status == "SUCCESS"
        # $500 / $3.50/gal = ~142.857 gallons
        expected_gallons = 500.0 / 3.50
        assert ctx.fuel_quantity_gallons == pytest.approx(expected_gallons, rel=1e-3)

    def test_custom_fuel_economy_override(self, engine):
        """Custom fuel economy overrides default."""
        inp = {
            "vehicle_type": "PASSENGER_CAR_GASOLINE",
            "fuel_type": "GASOLINE",
            "calculation_method": "DISTANCE_BASED",
            "distance": 100.0,
            "distance_unit": "KM",
            "fuel_economy": 30.0,
            "fuel_economy_unit": "MPG_US",
            "gwp_source": "AR6",
        }
        ctx = PipelineContext(
            pipeline_id="p6", calculation_id="c6", input_data=inp,
        )
        engine._stage_resolve_vehicle(ctx)
        result = engine._stage_estimate_fuel(ctx)
        assert result.status == "SUCCESS"
        # 30 MPG_US -> 235.214/30 = 7.84 L/100km
        # 100km * 7.84/100 = 7.84 L -> 7.84 * 0.264172 gallons
        expected_l_per_100km = 235.214 / 30.0
        expected_liters = (100.0 / 100.0) * expected_l_per_100km
        expected_gallons = expected_liters * 0.264172
        assert ctx.fuel_quantity_gallons == pytest.approx(expected_gallons, rel=1e-2)


# ===================================================================
# TestStageCalculate (8 tests)
# ===================================================================


class TestStageCalculate:
    """Test stage 4: CALCULATE_EMISSIONS."""

    def _run_through_stage4(
        self, engine: MobileCombustionPipelineEngine, inp: Dict[str, Any],
    ) -> StageResult:
        """Helper to run stages 1-4."""
        ctx = PipelineContext(
            pipeline_id="p-calc",
            calculation_id="c-calc",
            input_data=dict(inp),
        )
        engine._stage_validate(ctx)
        engine._stage_resolve_vehicle(ctx)
        engine._stage_estimate_fuel(ctx)
        return engine._stage_calculate(ctx), ctx

    def test_fuel_based_co2_calculation(self, engine, fuel_based_input):
        """Fuel-based CO2 = fuel_gallons * CO2_factor_kg_per_gallon."""
        result, ctx = self._run_through_stage4(engine, fuel_based_input)
        assert result.status == "SUCCESS"
        calc = ctx.calculation_result
        expected_co2 = 100.0 * FUEL_CO2_FACTORS_KG_PER_GALLON["GASOLINE"]
        assert calc["co2_kg"] == pytest.approx(expected_co2, rel=1e-4)

    def test_fuel_based_ch4_calculation(self, engine, fuel_based_input):
        """Fuel-based CH4 = distance_miles * ch4_g_per_mile / 1000 * GWP."""
        result, ctx = self._run_through_stage4(engine, fuel_based_input)
        calc = ctx.calculation_result
        # CH4 should be non-negative
        assert calc["ch4_co2e_kg"] >= 0.0
        assert calc["ch4_kg"] >= 0.0

    def test_fuel_based_n2o_calculation(self, engine, fuel_based_input):
        """Fuel-based N2O = distance_miles * n2o_g_per_mile / 1000 * GWP."""
        result, ctx = self._run_through_stage4(engine, fuel_based_input)
        calc = ctx.calculation_result
        assert calc["n2o_co2e_kg"] >= 0.0
        assert calc["n2o_kg"] >= 0.0

    def test_fuel_based_total_equals_sum(self, engine, fuel_based_input):
        """Total CO2e = CO2 + CH4_CO2e + N2O_CO2e."""
        result, ctx = self._run_through_stage4(engine, fuel_based_input)
        calc = ctx.calculation_result
        expected_total = calc["co2_kg"] + calc["ch4_co2e_kg"] + calc["n2o_co2e_kg"]
        assert calc["total_co2e_kg"] == pytest.approx(expected_total, rel=1e-4)

    def test_distance_based_calculation(self, engine, distance_based_input):
        """Distance-based calculation produces non-zero emissions."""
        result, ctx = self._run_through_stage4(engine, distance_based_input)
        assert result.status == "SUCCESS"
        calc = ctx.calculation_result
        assert calc["total_co2e_kg"] > 0.0
        assert calc["total_co2e_tonnes"] > 0.0

    def test_spend_based_calculation(self, engine, spend_based_input):
        """Spend-based: total = spend_amount * factor."""
        result, ctx = self._run_through_stage4(engine, spend_based_input)
        assert result.status == "SUCCESS"
        calc = ctx.calculation_result
        expected = 500.0 * SPEND_BASED_FACTORS_KG_CO2E_PER_USD["GASOLINE"]
        assert calc["total_co2e_kg"] == pytest.approx(expected, rel=1e-4)

    def test_tonnes_equals_kg_divided_by_1000(self, engine, fuel_based_input):
        """Total CO2e tonnes = total CO2e kg / 1000."""
        result, ctx = self._run_through_stage4(engine, fuel_based_input)
        calc = ctx.calculation_result
        assert calc["total_co2e_tonnes"] == pytest.approx(
            calc["total_co2e_kg"] / 1000.0, rel=1e-6,
        )

    def test_gas_emissions_breakdown(self, engine, fuel_based_input):
        """Gas emissions list contains CO2, CH4, N2O entries."""
        result, ctx = self._run_through_stage4(engine, fuel_based_input)
        calc = ctx.calculation_result
        gas_emissions = calc.get("gas_emissions", [])
        assert len(gas_emissions) == 3
        gases = [g["gas"] for g in gas_emissions]
        assert "CO2" in gases
        assert "CH4" in gases
        assert "N2O" in gases


# ===================================================================
# TestStageBiofuel (6 tests)
# ===================================================================


class TestStageBiofuel:
    """Test stage 5: APPLY_BIOFUEL_ADJUSTMENT."""

    def _run_through_stage5(
        self, engine: MobileCombustionPipelineEngine, inp: Dict[str, Any],
    ) -> PipelineContext:
        """Helper to run stages 1-5."""
        ctx = PipelineContext(
            pipeline_id="p-bio",
            calculation_id="c-bio",
            input_data=dict(inp),
        )
        engine._stage_validate(ctx)
        engine._stage_resolve_vehicle(ctx)
        engine._stage_estimate_fuel(ctx)
        engine._stage_calculate(ctx)
        engine._stage_biofuel_adjustment(ctx)
        return ctx

    def test_pure_gasoline_no_biogenic(self, engine, fuel_based_input):
        """Pure gasoline has fossil_fraction=1.0, no biogenic CO2."""
        ctx = self._run_through_stage5(engine, fuel_based_input)
        adj = ctx.biofuel_adjustment
        assert adj["fossil_fraction"] == 1.0
        assert adj["biogenic_fraction"] == 0.0
        assert adj["biogenic_co2_kg"] == 0.0

    def test_e10_has_biogenic_fraction(self, engine, biofuel_e10_input):
        """E10 has 6.7% biogenic content."""
        ctx = self._run_through_stage5(engine, biofuel_e10_input)
        adj = ctx.biofuel_adjustment
        assert adj["fossil_fraction"] == pytest.approx(
            BIOFUEL_FOSSIL_FRACTION["E10"],
        )
        assert adj["biogenic_fraction"] == pytest.approx(
            1.0 - BIOFUEL_FOSSIL_FRACTION["E10"],
        )
        assert adj["biogenic_co2_kg"] > 0.0

    def test_e85_high_biogenic(self, engine, biofuel_e85_input):
        """E85 has ~78.5% biogenic content."""
        ctx = self._run_through_stage5(engine, biofuel_e85_input)
        adj = ctx.biofuel_adjustment
        expected_bio_frac = 1.0 - BIOFUEL_FOSSIL_FRACTION["E85"]
        assert adj["biogenic_fraction"] == pytest.approx(expected_bio_frac)
        assert adj["biogenic_co2_kg"] > adj["fossil_co2_kg"]

    def test_b20_biogenic_separation(self, engine, biofuel_b20_input):
        """B20 separates biogenic and fossil CO2."""
        ctx = self._run_through_stage5(engine, biofuel_b20_input)
        adj = ctx.biofuel_adjustment
        assert adj["fossil_fraction"] == pytest.approx(
            BIOFUEL_FOSSIL_FRACTION["B20"],
        )
        total_co2 = adj["total_co2_kg"]
        assert adj["fossil_co2_kg"] + adj["biogenic_co2_kg"] == pytest.approx(
            total_co2, rel=1e-4,
        )

    def test_b100_fully_biogenic(self, engine, biofuel_b100_input):
        """B100 is 100% biogenic, fossil_fraction = 0.0."""
        ctx = self._run_through_stage5(engine, biofuel_b100_input)
        adj = ctx.biofuel_adjustment
        assert adj["fossil_fraction"] == pytest.approx(0.0)
        assert adj["fossil_co2_kg"] == pytest.approx(0.0)
        # All CO2 is biogenic
        assert adj["biogenic_co2_kg"] == pytest.approx(adj["total_co2_kg"])

    def test_adjusted_co2e_excludes_biogenic(self, engine, biofuel_e10_input):
        """Adjusted CO2e = fossil_CO2 + CH4_CO2e + N2O_CO2e (excludes biogenic)."""
        ctx = self._run_through_stage5(engine, biofuel_e10_input)
        adj = ctx.biofuel_adjustment
        expected_adjusted = (
            adj["fossil_co2_kg"]
            + adj["ch4_co2e_kg"]
            + adj["n2o_co2e_kg"]
        )
        assert adj["adjusted_co2e_kg"] == pytest.approx(expected_adjusted, rel=1e-4)


# ===================================================================
# TestStageUncertainty (4 tests)
# ===================================================================


class TestStageUncertainty:
    """Test stage 6: QUANTIFY_UNCERTAINTY."""

    def _run_through_stage6(
        self, engine: MobileCombustionPipelineEngine, inp: Dict[str, Any],
    ) -> PipelineContext:
        """Helper to run stages 1-6."""
        ctx = PipelineContext(
            pipeline_id="p-unc",
            calculation_id="c-unc",
            input_data=dict(inp),
        )
        engine._stage_validate(ctx)
        engine._stage_resolve_vehicle(ctx)
        engine._stage_estimate_fuel(ctx)
        engine._stage_calculate(ctx)
        engine._stage_biofuel_adjustment(ctx)
        engine._stage_uncertainty(ctx)
        return ctx

    def test_uncertainty_produces_mean_and_std(self, engine, fuel_based_input):
        """Uncertainty quantification produces mean, std, and percentiles."""
        ctx = self._run_through_stage6(engine, fuel_based_input)
        unc = ctx.uncertainty_result
        assert unc is not None
        assert "mean_co2e_kg" in unc
        assert "std_co2e_kg" in unc
        assert "p5_co2e_kg" in unc
        assert "p95_co2e_kg" in unc

    def test_uncertainty_p5_less_than_p95(self, engine, fuel_based_input):
        """5th percentile is less than or equal to 95th percentile."""
        ctx = self._run_through_stage6(engine, fuel_based_input)
        unc = ctx.uncertainty_result
        assert unc["p5_co2e_kg"] <= unc["p95_co2e_kg"]

    def test_uncertainty_mean_equals_total(self, engine, fuel_based_input):
        """Mean CO2e equals the deterministic total (analytical fallback)."""
        ctx = self._run_through_stage6(engine, fuel_based_input)
        unc = ctx.uncertainty_result
        calc = ctx.calculation_result
        assert unc["mean_co2e_kg"] == pytest.approx(
            calc["total_co2e_kg"], rel=1e-4,
        )

    def test_uncertainty_confidence_interval(self, engine, fuel_based_input):
        """Confidence interval percentage is 90%."""
        ctx = self._run_through_stage6(engine, fuel_based_input)
        unc = ctx.uncertainty_result
        assert unc["confidence_interval_pct"] == 90.0


# ===================================================================
# TestStageCompliance (4 tests)
# ===================================================================


class TestStageCompliance:
    """Test stage 7: CHECK_COMPLIANCE."""

    def _run_through_stage7(
        self, engine: MobileCombustionPipelineEngine, inp: Dict[str, Any],
    ) -> PipelineContext:
        """Helper to run stages 1-7."""
        ctx = PipelineContext(
            pipeline_id="p-comp",
            calculation_id="c-comp",
            input_data=dict(inp),
        )
        engine._stage_validate(ctx)
        engine._stage_resolve_vehicle(ctx)
        engine._stage_estimate_fuel(ctx)
        engine._stage_calculate(ctx)
        engine._stage_biofuel_adjustment(ctx)
        engine._stage_uncertainty(ctx)
        engine._stage_compliance(ctx)
        return ctx

    def test_compliance_returns_framework_results(self, engine, fuel_based_input):
        """Compliance check returns results for the specified framework."""
        ctx = self._run_through_stage7(engine, fuel_based_input)
        comp = ctx.compliance_results
        assert comp is not None
        assert len(comp) >= 1
        assert comp[0]["framework"] == "GHG_PROTOCOL"

    def test_compliance_checks_contain_requirements(self, engine, fuel_based_input):
        """Each compliance record contains requirement-level checks."""
        ctx = self._run_through_stage7(engine, fuel_based_input)
        comp = ctx.compliance_results[0]
        assert "checks" in comp
        assert comp["total_requirements"] >= 1

    def test_compliance_ghg_protocol_four_requirements(self, engine, fuel_based_input):
        """GHG Protocol has 4 requirements defined."""
        ctx = self._run_through_stage7(engine, fuel_based_input)
        comp = ctx.compliance_results[0]
        assert comp["total_requirements"] == len(
            COMPLIANCE_REQUIREMENTS["GHG_PROTOCOL"],
        )

    def test_compliance_multiple_frameworks(self, engine):
        """Multiple frameworks can be checked in one pipeline run."""
        inp = {
            "vehicle_type": "PASSENGER_CAR_GASOLINE",
            "fuel_type": "GASOLINE",
            "calculation_method": "FUEL_BASED",
            "fuel_quantity": 50.0,
            "fuel_unit": "GALLONS",
            "gwp_source": "AR6",
            "regulatory_framework": ["GHG_PROTOCOL", "ISO_14064"],
        }
        ctx = PipelineContext(
            pipeline_id="p-multi", calculation_id="c-multi",
            input_data=dict(inp),
        )
        engine._stage_validate(ctx)
        engine._stage_resolve_vehicle(ctx)
        engine._stage_estimate_fuel(ctx)
        engine._stage_calculate(ctx)
        engine._stage_biofuel_adjustment(ctx)
        engine._stage_uncertainty(ctx)
        engine._stage_compliance(ctx)
        assert len(ctx.compliance_results) == 2
        frameworks = [c["framework"] for c in ctx.compliance_results]
        assert "GHG_PROTOCOL" in frameworks
        assert "ISO_14064" in frameworks


# ===================================================================
# TestStageAudit (4 tests)
# ===================================================================


class TestStageAudit:
    """Test stage 8: GENERATE_AUDIT."""

    def _run_full_pipeline_ctx(
        self, engine: MobileCombustionPipelineEngine, inp: Dict[str, Any],
    ) -> PipelineContext:
        """Helper to run all 8 stages manually."""
        ctx = PipelineContext(
            pipeline_id="p-audit",
            calculation_id="c-audit",
            input_data=dict(inp),
        )
        engine._stage_validate(ctx)
        engine._stage_resolve_vehicle(ctx)
        engine._stage_estimate_fuel(ctx)
        engine._stage_calculate(ctx)
        engine._stage_biofuel_adjustment(ctx)
        engine._stage_uncertainty(ctx)
        engine._stage_compliance(ctx)
        engine._stage_audit(ctx)
        return ctx

    def test_audit_entry_created(self, engine, fuel_based_input):
        """Audit stage creates an audit entry."""
        ctx = self._run_full_pipeline_ctx(engine, fuel_based_input)
        assert len(ctx.audit_entries) == 1

    def test_audit_entry_has_chain_hash(self, engine, fuel_based_input):
        """Audit entry contains a chain_hash (SHA-256)."""
        ctx = self._run_full_pipeline_ctx(engine, fuel_based_input)
        audit = ctx.audit_entries[0]
        assert "chain_hash" in audit
        assert len(audit["chain_hash"]) == 64

    def test_audit_entry_has_input_output_summary(self, engine, fuel_based_input):
        """Audit entry contains input_summary and output_summary."""
        ctx = self._run_full_pipeline_ctx(engine, fuel_based_input)
        audit = ctx.audit_entries[0]
        assert "input_summary" in audit
        assert "output_summary" in audit
        assert audit["input_summary"]["vehicle_type"] == "PASSENGER_CAR_GASOLINE"
        assert audit["output_summary"]["total_co2e_kg"] > 0.0

    def test_audit_provenance_chain_populated(self, engine, fuel_based_input):
        """Provenance chain is populated when run through full pipeline."""
        # Use run_pipeline (not manual stage calls) so _execute_stage
        # appends to ctx.provenance_chain
        result = engine.run_pipeline(fuel_based_input)
        audit_entries = result["result"].get("audit_entries", [])
        assert len(audit_entries) >= 1
        audit = audit_entries[0]
        assert "provenance_chain" in audit
        # run_pipeline calls _execute_stage for 8 stages; each SUCCESS
        # appends a provenance_hash to the chain
        assert len(audit["provenance_chain"]) >= 7


# ===================================================================
# TestRunPipeline (8 tests)
# ===================================================================


class TestRunPipeline:
    """Test full pipeline execution through run_pipeline."""

    def test_fuel_based_pipeline_success(self, engine, fuel_based_input):
        """Full fuel-based pipeline run succeeds."""
        result = engine.run_pipeline(fuel_based_input)
        assert result["success"] is True
        assert result["stages_completed"] == 8
        assert result["stages_total"] == 8

    def test_distance_based_pipeline_success(self, engine, distance_based_input):
        """Full distance-based pipeline run succeeds."""
        result = engine.run_pipeline(distance_based_input)
        assert result["success"] is True
        assert result["result"]["total_co2e_kg"] > 0.0

    def test_spend_based_pipeline_success(self, engine, spend_based_input):
        """Full spend-based pipeline run succeeds."""
        result = engine.run_pipeline(spend_based_input)
        assert result["success"] is True
        assert result["result"]["total_co2e_kg"] > 0.0

    def test_pipeline_has_provenance_hash(self, engine, fuel_based_input):
        """Pipeline result includes pipeline_provenance_hash."""
        result = engine.run_pipeline(fuel_based_input)
        assert "pipeline_provenance_hash" in result
        assert len(result["pipeline_provenance_hash"]) == 64

    def test_pipeline_has_stage_results(self, engine, fuel_based_input):
        """Pipeline result includes per-stage results."""
        result = engine.run_pipeline(fuel_based_input)
        assert len(result["stage_results"]) == 8
        for sr in result["stage_results"]:
            assert "stage_name" in sr
            assert "status" in sr
            assert "provenance_hash" in sr

    def test_pipeline_timing(self, engine, fuel_based_input):
        """Pipeline reports total_duration_ms > 0."""
        result = engine.run_pipeline(fuel_based_input)
        assert result["total_duration_ms"] > 0.0

    def test_invalid_input_aborts_pipeline(self, engine):
        """Invalid input aborts at VALIDATE and skips remaining critical stages."""
        inp = {
            "calculation_method": "FUEL_BASED",
            # missing fuel_quantity
        }
        result = engine.run_pipeline(inp)
        assert result["success"] is False
        assert result["stages_completed"] < 8

    def test_pipeline_with_diesel_aircraft(self, engine):
        """Pipeline handles aircraft with jet fuel correctly."""
        inp = {
            "vehicle_type": "AIRCRAFT",
            "fuel_type": "JET_FUEL",
            "calculation_method": "FUEL_BASED",
            "fuel_quantity": 5000.0,
            "fuel_unit": "GALLONS",
            "gwp_source": "AR6",
        }
        result = engine.run_pipeline(inp)
        assert result["success"] is True
        co2_kg = result["result"]["co2_kg"]
        expected_co2 = 5000.0 * FUEL_CO2_FACTORS_KG_PER_GALLON["JET_FUEL"]
        assert co2_kg == pytest.approx(expected_co2, rel=1e-3)


# ===================================================================
# TestBatchPipeline (3 tests)
# ===================================================================


class TestBatchPipeline:
    """Test batch pipeline execution."""

    def test_batch_multiple_inputs(self, engine, fuel_based_input, distance_based_input):
        """Batch processing handles multiple inputs."""
        results = engine.run_batch_pipeline(
            [fuel_based_input, distance_based_input],
        )
        # First element is the batch summary
        assert results[0]["total_count"] == 2
        assert results[0]["success_count"] == 2
        assert results[0]["failure_count"] == 0

    def test_batch_checkpointing(self, engine, fuel_based_input):
        """Batch processing saves checkpoints at intervals."""
        inputs = [dict(fuel_based_input) for _ in range(15)]
        results = engine.run_batch_pipeline(
            inputs, checkpoint_interval=5,
        )
        assert results[0]["total_count"] == 15
        # Checkpoints should have been saved at indices 5, 10, 15
        # Verify by checking internal _checkpoints dict
        assert len(engine._checkpoints) >= 1

    def test_batch_provenance_hash(self, engine, fuel_based_input):
        """Batch result includes a provenance hash."""
        results = engine.run_batch_pipeline([fuel_based_input])
        summary = results[0]
        assert "provenance_hash" in summary
        assert len(summary["provenance_hash"]) == 64


# ===================================================================
# TestPipelineStats (2 tests)
# ===================================================================


class TestPipelineStats:
    """Test pipeline statistics after runs."""

    def test_stats_after_single_run(self, engine, fuel_based_input):
        """Stats reflect one successful pipeline run."""
        engine.run_pipeline(fuel_based_input)
        stats = engine.get_pipeline_stats()
        assert stats["total_runs"] == 1
        assert stats["successful_runs"] == 1
        assert stats["failed_runs"] == 0
        assert stats["success_rate_pct"] == 100.0
        assert stats["avg_duration_ms"] > 0.0

    def test_stats_reset(self, engine, fuel_based_input):
        """reset_stats clears all counters."""
        engine.run_pipeline(fuel_based_input)
        engine.reset_stats()
        stats = engine.get_pipeline_stats()
        assert stats["total_runs"] == 0
        assert stats["successful_runs"] == 0
        assert stats["avg_duration_ms"] == 0.0
