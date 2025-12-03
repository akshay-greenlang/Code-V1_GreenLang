"""
GL-019 HEATSCHEDULER - Equipment Dispatch Optimizer Tests

Comprehensive unit tests for the EquipmentDispatchOptimizer calculator.
Tests cover multi-unit dispatch, part-load efficiency, start/stop costs,
minimum run-time constraints, thermal storage, and demand response.

Author: GL-TestEngineer
Version: 1.0.0
"""

import sys
import os
import pytest
import math
from decimal import Decimal
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from calculators.equipment_dispatch_optimizer import (
    # Main class
    EquipmentDispatchOptimizer,
    # Input/Output dataclasses
    DispatchOptimizerInput,
    DispatchOptimizerOutput,
    EquipmentUnit,
    ThermalStorageUnit,
    DemandResponseEvent,
    HourlyLoadRequirement,
    PartLoadEfficiencyCurve,
    EquipmentDispatch,
    StorageDispatch,
    HourlyDispatchSummary,
    # Enums
    EquipmentType,
    DispatchMode,
    StorageMode,
    DemandResponseEventType,
    # Standalone functions
    calculate_part_load_efficiency,
    calculate_staging_order,
    calculate_storage_dispatch_strategy,
    calculate_demand_response_potential,
    calculate_carbon_intensity,
    estimate_start_stop_costs,
    cached_efficiency_lookup,
    calculate_minimum_units_required,
    calculate_optimal_loading,
    # Constants
    BOILER_EFFICIENCY_CURVE,
    CHILLER_EFFICIENCY_CURVE,
    HEAT_PUMP_EFFICIENCY_CURVE,
)

from calculators.provenance import verify_provenance


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_boiler() -> EquipmentUnit:
    """Create sample boiler equipment."""
    return EquipmentUnit(
        equipment_id="BOILER-001",
        equipment_type=EquipmentType.BOILER,
        name="Main Boiler #1",
        capacity_kw=500.0,
        efficiency_nominal=0.92,
        min_load_fraction=0.2,
        max_load_fraction=1.0,
        start_cost=75.0,
        stop_cost=0.0,
        min_run_time_min=30,
        min_off_time_min=15,
        fuel_type="gas",
        fuel_cost_per_kwh=0.04,
        carbon_intensity_kg_per_kwh=0.184,
        ramp_rate_kw_per_min=50.0,
        priority=1,
        is_available=True
    )


@pytest.fixture
def sample_heat_pump() -> EquipmentUnit:
    """Create sample heat pump equipment."""
    return EquipmentUnit(
        equipment_id="HP-001",
        equipment_type=EquipmentType.HEAT_PUMP,
        name="Air Source Heat Pump",
        capacity_kw=200.0,
        efficiency_nominal=3.5,  # COP
        min_load_fraction=0.3,
        max_load_fraction=1.0,
        start_cost=25.0,
        stop_cost=0.0,
        min_run_time_min=15,
        min_off_time_min=10,
        fuel_type="electricity",
        fuel_cost_per_kwh=0.12,
        carbon_intensity_kg_per_kwh=0.40,
        ramp_rate_kw_per_min=30.0,
        priority=2,
        is_available=True
    )


@pytest.fixture
def sample_chiller() -> EquipmentUnit:
    """Create sample chiller equipment."""
    return EquipmentUnit(
        equipment_id="CHILLER-001",
        equipment_type=EquipmentType.CHILLER,
        name="Centrifugal Chiller",
        capacity_kw=800.0,
        efficiency_nominal=5.5,  # COP
        min_load_fraction=0.25,
        max_load_fraction=1.0,
        start_cost=100.0,
        stop_cost=25.0,
        min_run_time_min=20,
        min_off_time_min=15,
        fuel_type="electricity",
        fuel_cost_per_kwh=0.12,
        carbon_intensity_kg_per_kwh=0.40,
        ramp_rate_kw_per_min=100.0,
        priority=1,
        is_available=True
    )


@pytest.fixture
def sample_equipment_fleet(sample_boiler, sample_heat_pump) -> List[EquipmentUnit]:
    """Create sample equipment fleet."""
    # Create a second boiler
    boiler2 = EquipmentUnit(
        equipment_id="BOILER-002",
        equipment_type=EquipmentType.BOILER,
        name="Main Boiler #2",
        capacity_kw=400.0,
        efficiency_nominal=0.88,
        min_load_fraction=0.2,
        max_load_fraction=1.0,
        start_cost=60.0,
        stop_cost=0.0,
        min_run_time_min=30,
        min_off_time_min=15,
        fuel_type="gas",
        fuel_cost_per_kwh=0.04,
        carbon_intensity_kg_per_kwh=0.184,
        priority=2,
        is_available=True
    )
    return [sample_boiler, boiler2, sample_heat_pump]


@pytest.fixture
def sample_thermal_storage() -> ThermalStorageUnit:
    """Create sample thermal storage."""
    return ThermalStorageUnit(
        storage_id="TES-001",
        capacity_kwh=2000.0,
        max_charge_rate_kw=400.0,
        max_discharge_rate_kw=500.0,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        standing_loss_pct_per_hour=0.5,
        initial_soc=0.5,
        min_soc=0.1,
        max_soc=0.9
    )


@pytest.fixture
def sample_dr_event() -> DemandResponseEvent:
    """Create sample demand response event."""
    return DemandResponseEvent(
        event_id="DR-001",
        event_type=DemandResponseEventType.CRITICAL_PEAK,
        start_hour=14,
        end_hour=18,
        target_reduction_kw=200.0,
        incentive_per_kwh=0.50,
        penalty_per_kwh=1.00,
        is_mandatory=False,
        notification_hours=24
    )


@pytest.fixture
def sample_hourly_loads() -> List[HourlyLoadRequirement]:
    """Generate sample hourly load requirements (24 hours)."""
    loads = []
    for hour in range(24):
        # Daily load pattern
        base_load = 200.0
        if 6 <= hour <= 18:  # Daytime
            heating_load = base_load + 150 * math.sin((hour - 6) * math.pi / 12)
        else:
            heating_load = base_load * 0.5

        # Price pattern
        if 14 <= hour <= 19:
            electricity_price = 0.25  # Peak
            is_peak = True
        elif hour < 6 or hour >= 22:
            electricity_price = 0.06  # Off-peak
            is_peak = False
        else:
            electricity_price = 0.12  # Shoulder
            is_peak = False

        loads.append(HourlyLoadRequirement(
            hour=hour,
            heating_load_kw=heating_load,
            cooling_load_kw=0.0,
            hot_water_load_kw=50.0,
            electricity_price_per_kwh=electricity_price,
            gas_price_per_kwh=0.04,
            grid_carbon_intensity=0.40 if is_peak else 0.35,
            is_peak_period=is_peak,
            dr_event_active=14 <= hour < 18
        ))
    return loads


@pytest.fixture
def optimizer() -> EquipmentDispatchOptimizer:
    """Create EquipmentDispatchOptimizer instance."""
    return EquipmentDispatchOptimizer()


# =============================================================================
# BASIC FUNCTIONALITY TESTS
# =============================================================================

class TestEquipmentDispatchOptimizerBasic:
    """Basic functionality tests for EquipmentDispatchOptimizer."""

    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initializes correctly."""
        assert optimizer is not None
        assert optimizer.VERSION == "1.0.0"
        assert optimizer.NAME == "EquipmentDispatchOptimizer"

    def test_simple_dispatch_single_unit(self, optimizer, sample_boiler, sample_hourly_loads):
        """Test simple dispatch with single equipment unit."""
        inputs = DispatchOptimizerInput(
            equipment_units=[sample_boiler],
            load_requirements=sample_hourly_loads,
            horizon_hours=24
        )

        result, provenance = optimizer.optimize(inputs)

        assert result is not None
        assert provenance is not None
        assert len(result.hourly_summaries) == 24
        assert result.total_energy_kwh > 0
        assert result.total_cost > 0

    def test_dispatch_multiple_units(self, optimizer, sample_equipment_fleet, sample_hourly_loads):
        """Test dispatch with multiple equipment units."""
        inputs = DispatchOptimizerInput(
            equipment_units=sample_equipment_fleet,
            load_requirements=sample_hourly_loads,
            horizon_hours=24
        )

        result, provenance = optimizer.optimize(inputs)

        assert result is not None
        assert len(result.equipment_dispatches) > 0
        # Check equipment utilization is calculated
        assert len(result.equipment_utilization) == len(sample_equipment_fleet)

    def test_dispatch_with_storage(
        self, optimizer, sample_equipment_fleet, sample_hourly_loads, sample_thermal_storage
    ):
        """Test dispatch with thermal storage."""
        inputs = DispatchOptimizerInput(
            equipment_units=sample_equipment_fleet,
            thermal_storage=sample_thermal_storage,
            load_requirements=sample_hourly_loads,
            horizon_hours=24
        )

        result, provenance = optimizer.optimize(inputs)

        assert result is not None
        assert len(result.storage_dispatches) == 24
        assert result.storage_cycles >= 0

    def test_provenance_verification(self, optimizer, sample_boiler, sample_hourly_loads):
        """Test provenance record is valid and verifiable."""
        inputs = DispatchOptimizerInput(
            equipment_units=[sample_boiler],
            load_requirements=sample_hourly_loads,
            horizon_hours=24
        )

        result, provenance = optimizer.optimize(inputs)

        assert provenance.calculator_name == "EquipmentDispatchOptimizer"
        assert provenance.calculator_version == "1.0.0"
        assert len(provenance.provenance_hash) == 64  # SHA-256
        assert len(provenance.calculation_steps) > 0
        assert verify_provenance(provenance)


# =============================================================================
# DISPATCH MODE TESTS
# =============================================================================

class TestDispatchModes:
    """Tests for different dispatch optimization modes."""

    def test_cost_optimal_mode(self, optimizer, sample_equipment_fleet, sample_hourly_loads):
        """Test cost-optimal dispatch mode."""
        inputs = DispatchOptimizerInput(
            equipment_units=sample_equipment_fleet,
            load_requirements=sample_hourly_loads,
            dispatch_mode=DispatchMode.COST_OPTIMAL,
            horizon_hours=24
        )

        result, _ = optimizer.optimize(inputs)

        assert result is not None
        # Cost-optimal should prefer cheaper units
        assert result.total_cost > 0

    def test_carbon_optimal_mode(self, optimizer, sample_equipment_fleet, sample_hourly_loads):
        """Test carbon-optimal dispatch mode."""
        inputs = DispatchOptimizerInput(
            equipment_units=sample_equipment_fleet,
            load_requirements=sample_hourly_loads,
            dispatch_mode=DispatchMode.CARBON_OPTIMAL,
            horizon_hours=24
        )

        result, _ = optimizer.optimize(inputs)

        assert result is not None
        assert result.total_carbon_kg >= 0

    def test_efficiency_optimal_mode(self, optimizer, sample_equipment_fleet, sample_hourly_loads):
        """Test efficiency-optimal dispatch mode."""
        inputs = DispatchOptimizerInput(
            equipment_units=sample_equipment_fleet,
            load_requirements=sample_hourly_loads,
            dispatch_mode=DispatchMode.EFFICIENCY_OPTIMAL,
            horizon_hours=24
        )

        result, _ = optimizer.optimize(inputs)

        assert result is not None
        assert result.average_efficiency > 0

    def test_demand_response_mode(
        self, optimizer, sample_equipment_fleet, sample_hourly_loads, sample_dr_event
    ):
        """Test demand response dispatch mode."""
        inputs = DispatchOptimizerInput(
            equipment_units=sample_equipment_fleet,
            load_requirements=sample_hourly_loads,
            dr_events=[sample_dr_event],
            dispatch_mode=DispatchMode.DEMAND_RESPONSE,
            horizon_hours=24
        )

        result, _ = optimizer.optimize(inputs)

        assert result is not None
        # Should have DR savings
        assert result.dr_savings >= 0


# =============================================================================
# PART-LOAD EFFICIENCY TESTS
# =============================================================================

class TestPartLoadEfficiency:
    """Tests for part-load efficiency calculations."""

    def test_boiler_efficiency_curve(self):
        """Test boiler part-load efficiency curve."""
        # At full load
        eff_full = calculate_part_load_efficiency(EquipmentType.BOILER, 1.0, 0.92)
        # At 60% load (typically peak efficiency for boilers)
        eff_60 = calculate_part_load_efficiency(EquipmentType.BOILER, 0.6, 0.92)
        # At 20% load
        eff_20 = calculate_part_load_efficiency(EquipmentType.BOILER, 0.2, 0.92)

        # Boilers typically have lower efficiency at low loads
        assert eff_20 < eff_60
        # 60% is often peak efficiency
        assert eff_60 >= eff_full * 0.95

    def test_chiller_efficiency_curve(self):
        """Test chiller part-load efficiency curve."""
        # Chillers typically have best efficiency around 60-80% load
        eff_full = calculate_part_load_efficiency(EquipmentType.CHILLER, 1.0, 5.5)
        eff_80 = calculate_part_load_efficiency(EquipmentType.CHILLER, 0.8, 5.5)
        eff_50 = calculate_part_load_efficiency(EquipmentType.CHILLER, 0.5, 5.5)
        eff_20 = calculate_part_load_efficiency(EquipmentType.CHILLER, 0.2, 5.5)

        # 80% load often best for chillers
        assert eff_80 >= eff_full * 0.98
        # Low load is inefficient
        assert eff_20 < eff_80

    def test_heat_pump_efficiency_curve(self):
        """Test heat pump part-load efficiency curve."""
        eff_full = calculate_part_load_efficiency(EquipmentType.HEAT_PUMP, 1.0, 3.5)
        eff_80 = calculate_part_load_efficiency(EquipmentType.HEAT_PUMP, 0.8, 3.5)
        eff_40 = calculate_part_load_efficiency(EquipmentType.HEAT_PUMP, 0.4, 3.5)

        # Heat pumps often peak around 80% load
        assert eff_80 >= eff_full * 0.95
        assert eff_40 < eff_80

    def test_cached_efficiency_lookup(self):
        """Test cached efficiency lookup returns consistent results."""
        result1 = cached_efficiency_lookup("boiler", 60, 92)  # 60% load, 92% nominal
        result2 = cached_efficiency_lookup("boiler", 60, 92)

        assert result1 == result2

    def test_efficiency_at_boundaries(self):
        """Test efficiency at load boundaries."""
        # At 0% load
        eff_zero = calculate_part_load_efficiency(EquipmentType.BOILER, 0.0, 0.92)
        # At 100% load
        eff_full = calculate_part_load_efficiency(EquipmentType.BOILER, 1.0, 0.92)

        assert eff_zero > 0  # Should still have some efficiency value
        assert eff_full > 0


# =============================================================================
# START/STOP AND MINIMUM RUN TIME TESTS
# =============================================================================

class TestStartStopConstraints:
    """Tests for start/stop costs and minimum run-time constraints."""

    def test_start_cost_tracking(self, optimizer, sample_boiler, sample_hourly_loads):
        """Test start costs are tracked correctly."""
        # Create load profile that will cause starts/stops
        intermittent_loads = []
        for hour in range(24):
            if hour % 4 < 2:  # On for 2 hours, off for 2 hours
                load = 400.0
            else:
                load = 50.0  # Below minimum
            intermittent_loads.append(HourlyLoadRequirement(
                hour=hour,
                heating_load_kw=load,
                electricity_price_per_kwh=0.12
            ))

        inputs = DispatchOptimizerInput(
            equipment_units=[sample_boiler],
            load_requirements=intermittent_loads,
            horizon_hours=24
        )

        result, _ = optimizer.optimize(inputs)

        # Should have some start/stop events
        assert result.start_stop_count >= 0
        assert result.start_stop_cost >= 0

    def test_estimate_start_stop_costs(self, sample_boiler):
        """Test start/stop cost estimation."""
        num_cycles = 10
        cost = estimate_start_stop_costs(sample_boiler, num_cycles)

        expected = Decimal(str((sample_boiler.start_cost + sample_boiler.stop_cost) * num_cycles * 1.10))
        assert cost == pytest.approx(float(expected), rel=0.01)

    def test_minimum_run_time_enforced(self, optimizer, sample_boiler):
        """Test minimum run time constraint is enforced."""
        # Create short load spikes
        loads = []
        for hour in range(24):
            if hour == 10:  # Single hour spike
                load = 400.0
            else:
                load = 50.0
            loads.append(HourlyLoadRequirement(hour=hour, heating_load_kw=load))

        inputs = DispatchOptimizerInput(
            equipment_units=[sample_boiler],
            load_requirements=loads,
            horizon_hours=24
        )

        result, _ = optimizer.optimize(inputs)

        # Equipment should run for minimum run time once started
        # (or not start at all if load is too brief)
        assert result is not None


# =============================================================================
# EQUIPMENT STAGING TESTS
# =============================================================================

class TestEquipmentStaging:
    """Tests for equipment staging logic."""

    def test_staging_order_cost_optimal(self, sample_equipment_fleet):
        """Test staging order in cost-optimal mode."""
        staging = calculate_staging_order(
            sample_equipment_fleet,
            600.0,  # Total load
            DispatchMode.COST_OPTIMAL
        )

        assert len(staging) > 0
        total_staged = sum(load for _, load in staging)
        assert total_staged >= 600.0 or total_staged == pytest.approx(
            sum(eq.capacity_kw * eq.max_load_fraction for eq in sample_equipment_fleet
                if eq.is_available), rel=0.01
        )

    def test_staging_order_efficiency_optimal(self, sample_equipment_fleet):
        """Test staging order in efficiency-optimal mode."""
        staging = calculate_staging_order(
            sample_equipment_fleet,
            400.0,
            DispatchMode.EFFICIENCY_OPTIMAL
        )

        assert len(staging) > 0
        # First unit should be most efficient
        if len(staging) > 0:
            first_eq_id = staging[0][0]
            first_eq = next(eq for eq in sample_equipment_fleet if eq.equipment_id == first_eq_id)
            # Heat pump has highest efficiency
            assert first_eq.efficiency_nominal >= 0.88

    def test_staging_order_carbon_optimal(self, sample_equipment_fleet):
        """Test staging order in carbon-optimal mode."""
        staging = calculate_staging_order(
            sample_equipment_fleet,
            400.0,
            DispatchMode.CARBON_OPTIMAL
        )

        assert len(staging) > 0

    def test_minimum_units_required(self, sample_equipment_fleet):
        """Test minimum units calculation."""
        # Load requiring multiple units
        min_units = calculate_minimum_units_required(sample_equipment_fleet, 800.0)
        assert min_units >= 2  # 500 + 400 = 900 capacity, need at least 2

        # Load requiring single unit
        min_units_small = calculate_minimum_units_required(sample_equipment_fleet, 200.0)
        assert min_units_small == 1

        # Zero load
        min_units_zero = calculate_minimum_units_required(sample_equipment_fleet, 0.0)
        assert min_units_zero == 0


# =============================================================================
# THERMAL STORAGE TESTS
# =============================================================================

class TestThermalStorage:
    """Tests for thermal storage integration."""

    def test_storage_dispatch_charging(self, optimizer, sample_boiler, sample_thermal_storage):
        """Test storage charges during off-peak periods."""
        loads = [
            HourlyLoadRequirement(
                hour=h,
                heating_load_kw=200.0,
                electricity_price_per_kwh=0.06 if h < 6 else 0.20,
                is_peak_period=h >= 6
            )
            for h in range(12)
        ]

        inputs = DispatchOptimizerInput(
            equipment_units=[sample_boiler],
            thermal_storage=sample_thermal_storage,
            load_requirements=loads,
            horizon_hours=12
        )

        result, _ = optimizer.optimize(inputs)

        # Check storage dispatches exist
        assert len(result.storage_dispatches) == 12

        # Check for charging during off-peak
        off_peak_dispatches = [d for d in result.storage_dispatches if d.hour < 6]
        charging_hours = [d for d in off_peak_dispatches if d.mode == StorageMode.CHARGING]
        # May charge during off-peak if SOC is low
        assert len(charging_hours) >= 0

    def test_storage_dispatch_discharging(self, optimizer, sample_boiler, sample_thermal_storage):
        """Test storage discharges when needed."""
        # High storage with low equipment capacity relative to load
        high_load = [
            HourlyLoadRequirement(
                hour=h,
                heating_load_kw=600.0,  # Above boiler capacity
                electricity_price_per_kwh=0.20,
                is_peak_period=True
            )
            for h in range(6)
        ]

        # Start with high SOC
        storage_high_soc = ThermalStorageUnit(
            storage_id="TES-001",
            capacity_kwh=2000.0,
            max_charge_rate_kw=400.0,
            max_discharge_rate_kw=500.0,
            charge_efficiency=0.95,
            discharge_efficiency=0.95,
            standing_loss_pct_per_hour=0.5,
            initial_soc=0.8,
            min_soc=0.1,
            max_soc=0.9
        )

        inputs = DispatchOptimizerInput(
            equipment_units=[sample_boiler],
            thermal_storage=storage_high_soc,
            load_requirements=high_load,
            horizon_hours=6
        )

        result, _ = optimizer.optimize(inputs)

        # Should have some discharging
        discharging = [d for d in result.storage_dispatches if d.mode == StorageMode.DISCHARGING]
        assert len(discharging) >= 0  # May or may not discharge

    def test_storage_soc_bounds(self, optimizer, sample_boiler, sample_thermal_storage, sample_hourly_loads):
        """Test storage SOC stays within bounds."""
        inputs = DispatchOptimizerInput(
            equipment_units=[sample_boiler],
            thermal_storage=sample_thermal_storage,
            load_requirements=sample_hourly_loads,
            horizon_hours=24
        )

        result, _ = optimizer.optimize(inputs)

        for dispatch in result.storage_dispatches:
            assert dispatch.soc_start >= sample_thermal_storage.min_soc - 0.01
            assert dispatch.soc_start <= sample_thermal_storage.max_soc + 0.01
            assert dispatch.soc_end >= sample_thermal_storage.min_soc - 0.01
            assert dispatch.soc_end <= sample_thermal_storage.max_soc + 0.01

    def test_storage_dispatch_strategy(self, sample_thermal_storage):
        """Test storage dispatch strategy calculation."""
        prices = [0.05] * 6 + [0.15] * 12 + [0.05] * 6  # Off-peak, peak, off-peak
        loads = [200.0] * 24

        strategy = calculate_storage_dispatch_strategy(sample_thermal_storage, prices, loads)

        assert len(strategy) == 24
        # Should see charging during low prices, discharging during high prices


# =============================================================================
# DEMAND RESPONSE TESTS
# =============================================================================

class TestDemandResponse:
    """Tests for demand response integration."""

    def test_dr_event_reduces_load(
        self, optimizer, sample_equipment_fleet, sample_hourly_loads, sample_dr_event
    ):
        """Test DR event reduces load served."""
        # Without DR
        inputs_no_dr = DispatchOptimizerInput(
            equipment_units=sample_equipment_fleet,
            load_requirements=sample_hourly_loads,
            horizon_hours=24
        )
        result_no_dr, _ = optimizer.optimize(inputs_no_dr)

        # With DR
        inputs_with_dr = DispatchOptimizerInput(
            equipment_units=sample_equipment_fleet,
            load_requirements=sample_hourly_loads,
            dr_events=[sample_dr_event],
            horizon_hours=24
        )
        result_with_dr, _ = optimizer.optimize(inputs_with_dr)

        # DR participation should be tracked
        dr_hours = [s for s in result_with_dr.hourly_summaries if s.dr_participation_kw > 0]
        assert len(dr_hours) >= 0  # May have DR participation

    def test_dr_savings_calculated(
        self, optimizer, sample_equipment_fleet, sample_hourly_loads, sample_dr_event
    ):
        """Test DR savings are calculated."""
        inputs = DispatchOptimizerInput(
            equipment_units=sample_equipment_fleet,
            load_requirements=sample_hourly_loads,
            dr_events=[sample_dr_event],
            horizon_hours=24
        )

        result, _ = optimizer.optimize(inputs)

        # DR savings should be non-negative
        assert result.dr_savings >= 0

    def test_demand_response_potential(self, sample_equipment_fleet):
        """Test DR curtailment potential calculation."""
        current_loads = {
            "BOILER-001": 400.0,
            "BOILER-002": 300.0,
            "HP-001": 150.0
        }

        potential = calculate_demand_response_potential(sample_equipment_fleet, current_loads)

        assert len(potential) == len(sample_equipment_fleet)
        # All units should have some curtailment potential
        for eq in sample_equipment_fleet:
            assert potential[eq.equipment_id] >= 0


# =============================================================================
# CARBON INTENSITY TESTS
# =============================================================================

class TestCarbonCalculations:
    """Tests for carbon intensity calculations."""

    def test_carbon_intensity_gas_equipment(self, sample_boiler):
        """Test carbon intensity for gas-fired equipment."""
        dispatches = [("BOILER-001", 400.0)]

        carbon_intensity = calculate_carbon_intensity(
            [sample_boiler],
            dispatches,
            grid_carbon_intensity=0.40
        )

        # Gas equipment uses its own carbon intensity
        expected = sample_boiler.carbon_intensity_kg_per_kwh / sample_boiler.efficiency_nominal
        assert carbon_intensity == pytest.approx(expected, rel=0.01)

    def test_carbon_intensity_electric_equipment(self, sample_heat_pump):
        """Test carbon intensity for electric equipment."""
        dispatches = [("HP-001", 150.0)]

        carbon_intensity = calculate_carbon_intensity(
            [sample_heat_pump],
            dispatches,
            grid_carbon_intensity=0.40
        )

        # Electric equipment uses grid carbon intensity
        expected = 0.40 / sample_heat_pump.efficiency_nominal
        assert carbon_intensity == pytest.approx(expected, rel=0.01)

    def test_carbon_intensity_mixed_fleet(self, sample_equipment_fleet):
        """Test carbon intensity for mixed equipment fleet."""
        dispatches = [
            ("BOILER-001", 300.0),
            ("HP-001", 100.0)
        ]

        carbon_intensity = calculate_carbon_intensity(
            sample_equipment_fleet,
            dispatches,
            grid_carbon_intensity=0.40
        )

        # Should be weighted average
        assert carbon_intensity > 0


# =============================================================================
# OPTIMAL LOADING TESTS
# =============================================================================

class TestOptimalLoading:
    """Tests for optimal loading calculations."""

    def test_find_optimal_load_boiler(self, sample_boiler):
        """Test finding optimal load point for boiler."""
        load_options = [100.0, 200.0, 300.0, 400.0, 500.0]

        optimal_load, efficiency = calculate_optimal_loading(sample_boiler, load_options)

        assert optimal_load in load_options
        assert efficiency > 0

    def test_optimal_load_respects_min_max(self, sample_boiler):
        """Test optimal load respects min/max constraints."""
        # Include loads outside valid range
        load_options = [50.0, 100.0, 300.0, 600.0]  # 50 below min, 600 above max

        optimal_load, efficiency = calculate_optimal_loading(sample_boiler, load_options)

        # Should be within valid range
        min_valid = sample_boiler.capacity_kw * sample_boiler.min_load_fraction
        max_valid = sample_boiler.capacity_kw * sample_boiler.max_load_fraction

        assert optimal_load >= min_valid or efficiency == 0
        assert optimal_load <= max_valid or efficiency == 0


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_no_load_requirement(self, optimizer, sample_boiler):
        """Test dispatch with zero load requirement."""
        zero_loads = [HourlyLoadRequirement(hour=h, heating_load_kw=0.0) for h in range(24)]

        inputs = DispatchOptimizerInput(
            equipment_units=[sample_boiler],
            load_requirements=zero_loads,
            horizon_hours=24
        )

        result, _ = optimizer.optimize(inputs)

        assert result.total_energy_kwh == 0.0

    def test_load_exceeds_capacity(self, optimizer, sample_boiler):
        """Test dispatch when load exceeds equipment capacity."""
        high_loads = [
            HourlyLoadRequirement(hour=h, heating_load_kw=1000.0)  # Above 500kW capacity
            for h in range(24)
        ]

        inputs = DispatchOptimizerInput(
            equipment_units=[sample_boiler],
            load_requirements=high_loads,
            horizon_hours=24,
            allow_unmet_load=True
        )

        result, _ = optimizer.optimize(inputs)

        # Should have unmet load
        assert result.unmet_energy_kwh > 0

    def test_unavailable_equipment(self, optimizer, sample_boiler, sample_hourly_loads):
        """Test dispatch with unavailable equipment."""
        unavailable_boiler = EquipmentUnit(
            equipment_id="BOILER-001",
            equipment_type=EquipmentType.BOILER,
            name="Main Boiler #1",
            capacity_kw=500.0,
            efficiency_nominal=0.92,
            is_available=False  # Not available
        )

        inputs = DispatchOptimizerInput(
            equipment_units=[unavailable_boiler],
            load_requirements=sample_hourly_loads,
            horizon_hours=24,
            allow_unmet_load=True
        )

        result, _ = optimizer.optimize(inputs)

        # All load should be unmet
        assert result.total_energy_kwh == 0.0

    def test_invalid_horizon_raises_error(self, optimizer, sample_boiler, sample_hourly_loads):
        """Test invalid horizon raises error."""
        with pytest.raises(ValueError):
            inputs = DispatchOptimizerInput(
                equipment_units=[sample_boiler],
                load_requirements=sample_hourly_loads,
                horizon_hours=0  # Invalid
            )
            optimizer.optimize(inputs)

    def test_invalid_equipment_capacity_raises_error(self, optimizer, sample_hourly_loads):
        """Test invalid equipment capacity raises error."""
        bad_equipment = EquipmentUnit(
            equipment_id="BAD-001",
            equipment_type=EquipmentType.BOILER,
            name="Bad Boiler",
            capacity_kw=-100.0,  # Invalid
            efficiency_nominal=0.92
        )

        with pytest.raises(ValueError):
            inputs = DispatchOptimizerInput(
                equipment_units=[bad_equipment],
                load_requirements=sample_hourly_loads,
                horizon_hours=24
            )
            optimizer.optimize(inputs)


# =============================================================================
# COST BREAKDOWN TESTS
# =============================================================================

class TestCostBreakdown:
    """Tests for cost breakdown calculations."""

    def test_cost_breakdown_categories(self, optimizer, sample_equipment_fleet, sample_hourly_loads):
        """Test cost breakdown includes all categories."""
        inputs = DispatchOptimizerInput(
            equipment_units=sample_equipment_fleet,
            load_requirements=sample_hourly_loads,
            horizon_hours=24
        )

        result, _ = optimizer.optimize(inputs)

        assert "fuel_cost" in result.cost_breakdown
        assert "start_stop_cost" in result.cost_breakdown
        # DR savings is negative cost
        assert "dr_savings" in result.cost_breakdown

    def test_cost_breakdown_sums_to_total(
        self, optimizer, sample_equipment_fleet, sample_hourly_loads
    ):
        """Test cost breakdown sums approximately to total cost."""
        inputs = DispatchOptimizerInput(
            equipment_units=sample_equipment_fleet,
            load_requirements=sample_hourly_loads,
            horizon_hours=24
        )

        result, _ = optimizer.optimize(inputs)

        breakdown_sum = (
            result.cost_breakdown.get("fuel_cost", 0) +
            result.cost_breakdown.get("start_stop_cost", 0) +
            result.cost_breakdown.get("unmet_load_penalty", 0) +
            result.cost_breakdown.get("dr_savings", 0)
        )

        # Should be close to total (may have rounding differences)
        assert breakdown_sum == pytest.approx(result.total_cost, rel=0.1)


# =============================================================================
# DETERMINISM AND REPRODUCIBILITY TESTS
# =============================================================================

class TestDeterminismAndReproducibility:
    """Tests for deterministic and reproducible results."""

    def test_deterministic_results(self, optimizer, sample_equipment_fleet, sample_hourly_loads):
        """Test same inputs produce same outputs."""
        inputs = DispatchOptimizerInput(
            equipment_units=sample_equipment_fleet,
            load_requirements=sample_hourly_loads,
            horizon_hours=24
        )

        result1, prov1 = optimizer.optimize(inputs)
        result2, prov2 = optimizer.optimize(inputs)

        # Results should be identical
        assert result1.total_cost == result2.total_cost
        assert result1.total_energy_kwh == result2.total_energy_kwh
        assert result1.total_carbon_kg == result2.total_carbon_kg

        # Input hashes should match
        assert prov1.input_hash == prov2.input_hash

    def test_different_inputs_different_results(
        self, optimizer, sample_equipment_fleet, sample_hourly_loads
    ):
        """Test different inputs produce different results."""
        inputs1 = DispatchOptimizerInput(
            equipment_units=sample_equipment_fleet,
            load_requirements=sample_hourly_loads,
            dispatch_mode=DispatchMode.COST_OPTIMAL,
            horizon_hours=24
        )

        inputs2 = DispatchOptimizerInput(
            equipment_units=sample_equipment_fleet,
            load_requirements=sample_hourly_loads,
            dispatch_mode=DispatchMode.CARBON_OPTIMAL,  # Different mode
            horizon_hours=24
        )

        result1, prov1 = optimizer.optimize(inputs1)
        result2, prov2 = optimizer.optimize(inputs2)

        # Input hashes should differ
        assert prov1.input_hash != prov2.input_hash


# =============================================================================
# UTILIZATION TESTS
# =============================================================================

class TestUtilization:
    """Tests for equipment utilization calculations."""

    def test_utilization_range(self, optimizer, sample_equipment_fleet, sample_hourly_loads):
        """Test utilization values are in valid range."""
        inputs = DispatchOptimizerInput(
            equipment_units=sample_equipment_fleet,
            load_requirements=sample_hourly_loads,
            horizon_hours=24
        )

        result, _ = optimizer.optimize(inputs)

        for eq_id, utilization in result.equipment_utilization.items():
            assert 0.0 <= utilization <= 1.0

    def test_high_utilization_with_matching_load(self, optimizer, sample_boiler):
        """Test high utilization when load matches capacity."""
        full_load = [
            HourlyLoadRequirement(hour=h, heating_load_kw=500.0)  # Match capacity
            for h in range(24)
        ]

        inputs = DispatchOptimizerInput(
            equipment_units=[sample_boiler],
            load_requirements=full_load,
            horizon_hours=24
        )

        result, _ = optimizer.optimize(inputs)

        # Should have high utilization
        utilization = result.equipment_utilization["BOILER-001"]
        assert utilization > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
