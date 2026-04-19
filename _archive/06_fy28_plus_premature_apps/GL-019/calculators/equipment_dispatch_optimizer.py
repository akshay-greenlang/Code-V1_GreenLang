"""
GL-019 HEATSCHEDULER - Equipment Dispatch Optimizer

Multi-unit equipment dispatch optimization for boilers, chillers, heat pumps,
and thermal storage with cost-optimal and carbon-optimal dispatch modes.

This module provides:
- Multi-unit dispatch optimization
- Part-load efficiency curves
- Start/stop cost penalties
- Minimum run-time constraints
- Equipment staging logic
- Demand response integration
- Cost-optimal vs carbon-optimal dispatch modes
- Thermal storage integration

Standards Reference:
- ISO 50001 - Energy Management Systems
- ASHRAE Handbook - HVAC Systems and Equipment
- ASHRAE Guideline 14 - Measurement of Energy and Demand Savings
- DOE Better Buildings - Load Management

Author: GL-CalculatorEngineer
Version: 1.0.0
"""

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, FrozenSet
from functools import lru_cache
from threading import Lock
import hashlib
import json
import math
import statistics

from .provenance import ProvenanceTracker, ProvenanceRecord


# =============================================================================
# CONSTANTS AND REFERENCE DATA
# =============================================================================

# Typical part-load efficiency curves (load fraction -> efficiency multiplier)
BOILER_EFFICIENCY_CURVE = {
    0.0: 0.70,
    0.2: 0.78,
    0.4: 0.85,
    0.6: 0.90,
    0.8: 0.93,
    1.0: 0.92  # Slight drop at full load
}

CHILLER_EFFICIENCY_CURVE = {
    0.0: 0.50,
    0.2: 0.65,
    0.4: 0.80,
    0.5: 0.90,
    0.6: 0.95,
    0.8: 0.98,
    1.0: 0.95  # Slight drop at full load
}

HEAT_PUMP_EFFICIENCY_CURVE = {
    0.0: 0.60,
    0.2: 0.75,
    0.4: 0.88,
    0.6: 0.95,
    0.8: 1.00,
    1.0: 0.97
}

# Carbon intensity defaults (kg CO2/kWh)
GRID_CARBON_INTENSITY = {
    "low": 0.15,      # Low-carbon grid (nuclear, hydro, renewables)
    "medium": 0.40,   # Mixed grid
    "high": 0.70,     # Coal-heavy grid
}

# Natural gas carbon intensity (kg CO2/kWh thermal)
GAS_CARBON_INTENSITY = 0.184

# Default minimum run times (minutes)
DEFAULT_MIN_RUN_TIME = {
    "boiler": 30,
    "chiller": 20,
    "heat_pump": 15,
    "thermal_storage": 0
}

# Default start costs (currency units)
DEFAULT_START_COST = {
    "boiler": 50.0,
    "chiller": 30.0,
    "heat_pump": 20.0,
    "thermal_storage": 0.0
}


class EquipmentType(str, Enum):
    """Types of HVAC equipment."""
    BOILER = "boiler"
    CHILLER = "chiller"
    HEAT_PUMP = "heat_pump"
    THERMAL_STORAGE = "thermal_storage"
    ELECTRIC_HEATER = "electric_heater"
    CHP = "chp"  # Combined Heat and Power


class DispatchMode(str, Enum):
    """Dispatch optimization modes."""
    COST_OPTIMAL = "cost_optimal"
    CARBON_OPTIMAL = "carbon_optimal"
    EFFICIENCY_OPTIMAL = "efficiency_optimal"
    DEMAND_RESPONSE = "demand_response"
    HYBRID = "hybrid"  # Weighted combination


class StorageMode(str, Enum):
    """Thermal storage operating mode."""
    CHARGING = "charging"
    DISCHARGING = "discharging"
    IDLE = "idle"


class DemandResponseEventType(str, Enum):
    """Demand response event types."""
    CRITICAL_PEAK = "critical_peak"
    LOAD_SHED = "load_shed"
    LOAD_SHIFT = "load_shift"
    REGULATION = "regulation"


# =============================================================================
# INPUT/OUTPUT DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class PartLoadEfficiencyCurve:
    """
    Part-load efficiency curve for equipment.

    Attributes:
        equipment_id: Equipment identifier
        load_points: Tuple of load fractions (0-1)
        efficiency_points: Corresponding efficiency values (0-1)
        min_turn_down: Minimum turn-down ratio (0-1)
    """
    equipment_id: str
    load_points: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    efficiency_points: Tuple[float, ...] = (0.70, 0.78, 0.85, 0.90, 0.93, 0.92)
    min_turn_down: float = 0.2


@dataclass(frozen=True)
class EquipmentUnit:
    """
    Single equipment unit specification.

    Attributes:
        equipment_id: Unique equipment identifier
        equipment_type: Type of equipment
        name: Human-readable name
        capacity_kw: Rated capacity (kW thermal output)
        efficiency_nominal: Nominal efficiency (COP for heat pumps)
        efficiency_curve: Part-load efficiency curve
        min_load_fraction: Minimum operating load (0-1)
        max_load_fraction: Maximum operating load (0-1)
        start_cost: Cost to start equipment (currency)
        stop_cost: Cost to stop equipment (currency)
        min_run_time_min: Minimum run time (minutes)
        min_off_time_min: Minimum off time (minutes)
        fuel_type: Fuel type (electricity, gas, etc.)
        fuel_cost_per_kwh: Fuel cost (currency/kWh)
        carbon_intensity_kg_per_kwh: CO2 emissions (kg/kWh fuel)
        ramp_rate_kw_per_min: Ramp rate (kW/min)
        priority: Dispatch priority (1 = highest)
        is_available: Availability flag
    """
    equipment_id: str
    equipment_type: EquipmentType
    name: str
    capacity_kw: float
    efficiency_nominal: float = 0.90
    efficiency_curve: Optional[PartLoadEfficiencyCurve] = None
    min_load_fraction: float = 0.2
    max_load_fraction: float = 1.0
    start_cost: float = 50.0
    stop_cost: float = 0.0
    min_run_time_min: int = 30
    min_off_time_min: int = 15
    fuel_type: str = "electricity"
    fuel_cost_per_kwh: float = 0.10
    carbon_intensity_kg_per_kwh: float = 0.40
    ramp_rate_kw_per_min: float = 100.0
    priority: int = 1
    is_available: bool = True


@dataclass(frozen=True)
class ThermalStorageUnit:
    """
    Thermal storage unit specification.

    Attributes:
        storage_id: Unique identifier
        capacity_kwh: Storage capacity (kWh thermal)
        max_charge_rate_kw: Maximum charging rate (kW)
        max_discharge_rate_kw: Maximum discharge rate (kW)
        charge_efficiency: Charging efficiency (0-1)
        discharge_efficiency: Discharging efficiency (0-1)
        standing_loss_pct_per_hour: Standing losses (%/hour)
        initial_soc: Initial state of charge (0-1)
        min_soc: Minimum state of charge (0-1)
        max_soc: Maximum state of charge (0-1)
    """
    storage_id: str
    capacity_kwh: float
    max_charge_rate_kw: float
    max_discharge_rate_kw: float
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    standing_loss_pct_per_hour: float = 0.5
    initial_soc: float = 0.5
    min_soc: float = 0.1
    max_soc: float = 0.9


@dataclass(frozen=True)
class DemandResponseEvent:
    """
    Demand response event specification.

    Attributes:
        event_id: Unique event identifier
        event_type: Type of DR event
        start_hour: Start hour (0-23)
        end_hour: End hour (0-23)
        target_reduction_kw: Target load reduction (kW)
        incentive_per_kwh: Incentive payment (currency/kWh reduced)
        penalty_per_kwh: Penalty for non-compliance (currency/kWh)
        is_mandatory: Mandatory participation flag
        notification_hours: Hours of advance notice
    """
    event_id: str
    event_type: DemandResponseEventType
    start_hour: int
    end_hour: int
    target_reduction_kw: float
    incentive_per_kwh: float = 0.50
    penalty_per_kwh: float = 1.00
    is_mandatory: bool = False
    notification_hours: int = 24


@dataclass(frozen=True)
class HourlyLoadRequirement:
    """
    Hourly thermal load requirement.

    Attributes:
        hour: Hour index
        heating_load_kw: Heating demand (kW)
        cooling_load_kw: Cooling demand (kW)
        hot_water_load_kw: Hot water demand (kW)
        electricity_price_per_kwh: Electricity price (currency/kWh)
        gas_price_per_kwh: Gas price (currency/kWh thermal)
        grid_carbon_intensity: Grid carbon intensity (kg CO2/kWh)
        is_peak_period: Peak period flag
        dr_event_active: Demand response event active
    """
    hour: int
    heating_load_kw: float = 0.0
    cooling_load_kw: float = 0.0
    hot_water_load_kw: float = 0.0
    electricity_price_per_kwh: float = 0.10
    gas_price_per_kwh: float = 0.04
    grid_carbon_intensity: float = 0.40
    is_peak_period: bool = False
    dr_event_active: bool = False


@dataclass(frozen=True)
class DispatchOptimizerInput:
    """
    Input parameters for equipment dispatch optimization.

    Attributes:
        equipment_units: List of available equipment
        thermal_storage: Thermal storage unit (optional)
        load_requirements: Hourly load requirements
        dr_events: Demand response events
        dispatch_mode: Optimization mode
        carbon_weight: Weight for carbon objective (0-1)
        cost_weight: Weight for cost objective (0-1)
        horizon_hours: Optimization horizon (hours)
        time_step_minutes: Time step (minutes)
        allow_unmet_load: Allow unmet load flag
        unmet_load_penalty: Penalty for unmet load (currency/kWh)
    """
    equipment_units: List[EquipmentUnit]
    thermal_storage: Optional[ThermalStorageUnit] = None
    load_requirements: List[HourlyLoadRequirement] = field(default_factory=list)
    dr_events: List[DemandResponseEvent] = field(default_factory=list)
    dispatch_mode: DispatchMode = DispatchMode.COST_OPTIMAL
    carbon_weight: float = 0.0
    cost_weight: float = 1.0
    horizon_hours: int = 24
    time_step_minutes: int = 60
    allow_unmet_load: bool = False
    unmet_load_penalty: float = 1000.0


@dataclass(frozen=True)
class EquipmentDispatch:
    """
    Single equipment dispatch decision.

    Attributes:
        equipment_id: Equipment identifier
        hour: Hour index
        load_kw: Dispatch load (kW output)
        load_fraction: Load fraction (0-1)
        efficiency: Operating efficiency
        fuel_input_kw: Fuel input (kW)
        operating_cost: Operating cost (currency)
        carbon_emissions_kg: CO2 emissions (kg)
        is_starting: Starting this hour
        is_stopping: Stopping this hour
    """
    equipment_id: str
    hour: int
    load_kw: float
    load_fraction: float
    efficiency: float
    fuel_input_kw: float
    operating_cost: float
    carbon_emissions_kg: float
    is_starting: bool
    is_stopping: bool


@dataclass(frozen=True)
class StorageDispatch:
    """
    Thermal storage dispatch decision.

    Attributes:
        storage_id: Storage identifier
        hour: Hour index
        mode: Operating mode (charging/discharging/idle)
        power_kw: Power flow (positive = charging)
        soc_start: State of charge at start
        soc_end: State of charge at end
        energy_stored_kwh: Energy stored this hour
        energy_released_kwh: Energy released this hour
        losses_kwh: Standing losses this hour
    """
    storage_id: str
    hour: int
    mode: StorageMode
    power_kw: float
    soc_start: float
    soc_end: float
    energy_stored_kwh: float
    energy_released_kwh: float
    losses_kwh: float


@dataclass(frozen=True)
class HourlyDispatchSummary:
    """
    Summary of dispatch for one hour.

    Attributes:
        hour: Hour index
        total_load_kw: Total load served (kW)
        total_heating_kw: Heating provided (kW)
        total_cooling_kw: Cooling provided (kW)
        unmet_load_kw: Unmet load (kW)
        total_fuel_input_kw: Total fuel input (kW)
        total_cost: Total cost (currency)
        total_carbon_kg: Total emissions (kg CO2)
        equipment_count_running: Number of units running
        storage_contribution_kw: Storage contribution (kW)
        dr_participation_kw: DR participation (kW)
    """
    hour: int
    total_load_kw: float
    total_heating_kw: float
    total_cooling_kw: float
    unmet_load_kw: float
    total_fuel_input_kw: float
    total_cost: float
    total_carbon_kg: float
    equipment_count_running: int
    storage_contribution_kw: float
    dr_participation_kw: float


@dataclass(frozen=True)
class DispatchOptimizerOutput:
    """
    Complete output from equipment dispatch optimization.

    Attributes:
        equipment_dispatches: All equipment dispatch decisions
        storage_dispatches: Storage dispatch decisions
        hourly_summaries: Hourly dispatch summaries
        total_energy_kwh: Total energy served (kWh)
        total_fuel_kwh: Total fuel consumed (kWh)
        total_cost: Total operating cost (currency)
        total_carbon_kg: Total CO2 emissions (kg)
        average_efficiency: Average system efficiency
        peak_demand_kw: Peak power demand (kW)
        unmet_energy_kwh: Total unmet energy (kWh)
        start_stop_count: Total start/stop events
        start_stop_cost: Total start/stop costs (currency)
        dr_savings: Demand response savings (currency)
        cost_breakdown: Cost breakdown by category
        equipment_utilization: Utilization by equipment
        storage_cycles: Storage charge/discharge cycles
    """
    equipment_dispatches: List[EquipmentDispatch]
    storage_dispatches: List[StorageDispatch]
    hourly_summaries: List[HourlyDispatchSummary]
    total_energy_kwh: float
    total_fuel_kwh: float
    total_cost: float
    total_carbon_kg: float
    average_efficiency: float
    peak_demand_kw: float
    unmet_energy_kwh: float
    start_stop_count: int
    start_stop_cost: float
    dr_savings: float
    cost_breakdown: Dict[str, float]
    equipment_utilization: Dict[str, float]
    storage_cycles: float


# =============================================================================
# EQUIPMENT DISPATCH OPTIMIZER CLASS
# =============================================================================

class EquipmentDispatchOptimizer:
    """
    Multi-unit equipment dispatch optimizer.

    Implements zero-hallucination, deterministic dispatch optimization for:
    - Multi-unit equipment (boilers, chillers, heat pumps)
    - Part-load efficiency curves
    - Start/stop cost penalties
    - Minimum run-time constraints
    - Equipment staging logic
    - Demand response integration
    - Cost-optimal vs carbon-optimal dispatch
    - Thermal storage integration

    Guarantees:
    - DETERMINISTIC: Same input always produces same output
    - REPRODUCIBLE: SHA-256 verified calculation chain
    - AUDITABLE: Complete step-by-step provenance trail
    - ZERO HALLUCINATION: No LLM in calculation path

    Example:
        >>> optimizer = EquipmentDispatchOptimizer()
        >>> equipment = [EquipmentUnit(...), ...]
        >>> loads = [HourlyLoadRequirement(...), ...]
        >>> inputs = DispatchOptimizerInput(
        ...     equipment_units=equipment,
        ...     load_requirements=loads
        ... )
        >>> result, provenance = optimizer.optimize(inputs)
        >>> print(f"Total Cost: ${result.total_cost:.2f}")
    """

    VERSION = "1.0.0"
    NAME = "EquipmentDispatchOptimizer"

    # Thread-safe cache lock
    _cache_lock = Lock()

    def __init__(self):
        """Initialize the equipment dispatch optimizer."""
        self._tracker: Optional[ProvenanceTracker] = None
        self._step_counter = 0

    def optimize(
        self,
        inputs: DispatchOptimizerInput
    ) -> Tuple[DispatchOptimizerOutput, ProvenanceRecord]:
        """
        Optimize equipment dispatch over the planning horizon.

        Args:
            inputs: DispatchOptimizerInput with equipment and load data

        Returns:
            Tuple of (DispatchOptimizerOutput, ProvenanceRecord)

        Raises:
            ValueError: If inputs are invalid
        """
        # Initialize provenance tracking
        self._tracker = ProvenanceTracker(
            calculator_name=self.NAME,
            calculator_version=self.VERSION,
            metadata={
                "standards": [
                    "ISO 50001",
                    "ASHRAE Handbook",
                    "ASHRAE Guideline 14"
                ],
                "domain": "Equipment Dispatch Optimization"
            }
        )
        self._step_counter = 0

        # Prepare inputs for provenance
        input_dict = {
            "num_equipment_units": len(inputs.equipment_units),
            "has_storage": inputs.thermal_storage is not None,
            "num_load_hours": len(inputs.load_requirements),
            "num_dr_events": len(inputs.dr_events),
            "dispatch_mode": inputs.dispatch_mode.value,
            "carbon_weight": inputs.carbon_weight,
            "cost_weight": inputs.cost_weight,
            "horizon_hours": inputs.horizon_hours
        }
        self._tracker.set_inputs(input_dict)

        # Validate inputs
        self._validate_inputs(inputs)

        # Initialize tracking variables
        equipment_dispatches: List[EquipmentDispatch] = []
        storage_dispatches: List[StorageDispatch] = []
        hourly_summaries: List[HourlyDispatchSummary] = []

        # Track equipment state
        equipment_state = {
            eq.equipment_id: {
                "is_running": False,
                "run_time_min": 0,
                "off_time_min": inputs.time_step_minutes,  # Start available
                "starts": 0,
                "stops": 0
            }
            for eq in inputs.equipment_units
        }

        # Track storage state
        storage_soc = inputs.thermal_storage.initial_soc if inputs.thermal_storage else 0.0

        # Optimize each hour
        for hour_idx in range(inputs.horizon_hours):
            # Get load requirement for this hour
            if hour_idx < len(inputs.load_requirements):
                load_req = inputs.load_requirements[hour_idx]
            else:
                # Use last known requirement
                load_req = inputs.load_requirements[-1] if inputs.load_requirements else HourlyLoadRequirement(hour=hour_idx)

            # Check for DR events
            dr_reduction = 0.0
            for dr_event in inputs.dr_events:
                if dr_event.start_hour <= hour_idx < dr_event.end_hour:
                    dr_reduction = dr_event.target_reduction_kw

            # Calculate total load requirement
            total_load = (
                load_req.heating_load_kw +
                load_req.cooling_load_kw +
                load_req.hot_water_load_kw
            )

            # Adjust for DR
            adjusted_load = max(0.0, total_load - dr_reduction)

            # Dispatch equipment for this hour
            hour_dispatches, remaining_load = self._dispatch_hour(
                inputs=inputs,
                hour=hour_idx,
                load_kw=adjusted_load,
                load_req=load_req,
                equipment_state=equipment_state
            )
            equipment_dispatches.extend(hour_dispatches)

            # Handle storage if available
            storage_dispatch = None
            storage_contribution = 0.0
            if inputs.thermal_storage:
                storage_dispatch, storage_soc, storage_contribution = self._dispatch_storage(
                    inputs=inputs,
                    hour=hour_idx,
                    remaining_load=remaining_load,
                    current_soc=storage_soc,
                    load_req=load_req
                )
                storage_dispatches.append(storage_dispatch)
                remaining_load = max(0.0, remaining_load - storage_contribution)

            # Calculate hourly summary
            summary = self._calculate_hourly_summary(
                hour=hour_idx,
                dispatches=hour_dispatches,
                storage_dispatch=storage_dispatch,
                load_req=load_req,
                remaining_load=remaining_load,
                dr_reduction=dr_reduction
            )
            hourly_summaries.append(summary)

        # Calculate overall metrics
        total_energy = sum(s.total_load_kw for s in hourly_summaries)
        total_fuel = sum(s.total_fuel_input_kw for s in hourly_summaries)
        total_cost = sum(s.total_cost for s in hourly_summaries)
        total_carbon = sum(s.total_carbon_kg for s in hourly_summaries)
        unmet_energy = sum(s.unmet_load_kw for s in hourly_summaries)
        peak_demand = max((s.total_fuel_input_kw for s in hourly_summaries), default=0.0)

        # Calculate start/stop metrics
        start_stop_count = sum(
            state["starts"] + state["stops"]
            for state in equipment_state.values()
        )
        start_stop_cost = self._calculate_start_stop_cost(inputs, equipment_state)

        # Calculate efficiency
        avg_efficiency = total_energy / total_fuel if total_fuel > 0 else 0.0

        # Calculate DR savings
        dr_savings = self._calculate_dr_savings(inputs, hourly_summaries)

        # Calculate utilization
        utilization = self._calculate_utilization(inputs, equipment_dispatches)

        # Calculate storage cycles
        storage_cycles = 0.0
        if inputs.thermal_storage and storage_dispatches:
            total_discharged = sum(
                d.energy_released_kwh for d in storage_dispatches
            )
            storage_cycles = total_discharged / inputs.thermal_storage.capacity_kwh

        # Cost breakdown
        cost_breakdown = {
            "fuel_cost": sum(d.operating_cost for d in equipment_dispatches),
            "start_stop_cost": start_stop_cost,
            "unmet_load_penalty": unmet_energy * inputs.unmet_load_penalty if inputs.allow_unmet_load else 0.0,
            "dr_savings": -dr_savings  # Negative because it's savings
        }

        self._add_step(
            "Calculate optimization totals",
            "aggregate",
            {
                "num_hours": inputs.horizon_hours,
                "num_equipment": len(inputs.equipment_units)
            },
            {
                "total_energy_kwh": total_energy,
                "total_cost": total_cost,
                "total_carbon_kg": total_carbon
            },
            "optimization_totals",
            "Aggregated dispatch results"
        )

        # Create output
        output = DispatchOptimizerOutput(
            equipment_dispatches=equipment_dispatches,
            storage_dispatches=storage_dispatches,
            hourly_summaries=hourly_summaries,
            total_energy_kwh=round(total_energy, 2),
            total_fuel_kwh=round(total_fuel, 2),
            total_cost=round(total_cost + start_stop_cost, 2),
            total_carbon_kg=round(total_carbon, 2),
            average_efficiency=round(avg_efficiency, 4),
            peak_demand_kw=round(peak_demand, 2),
            unmet_energy_kwh=round(unmet_energy, 2),
            start_stop_count=start_stop_count,
            start_stop_cost=round(start_stop_cost, 2),
            dr_savings=round(dr_savings, 2),
            cost_breakdown={k: round(v, 2) for k, v in cost_breakdown.items()},
            equipment_utilization={k: round(v, 4) for k, v in utilization.items()},
            storage_cycles=round(storage_cycles, 2)
        )

        # Set outputs and finalize provenance
        self._tracker.set_outputs({
            "total_energy_kwh": output.total_energy_kwh,
            "total_fuel_kwh": output.total_fuel_kwh,
            "total_cost": output.total_cost,
            "total_carbon_kg": output.total_carbon_kg,
            "average_efficiency": output.average_efficiency,
            "peak_demand_kw": output.peak_demand_kw,
            "unmet_energy_kwh": output.unmet_energy_kwh,
            "start_stop_count": output.start_stop_count,
            "dr_savings": output.dr_savings
        })

        provenance = self._tracker.finalize()
        return output, provenance

    def _add_step(
        self,
        description: str,
        operation: str,
        inputs: Dict[str, Any],
        output_value: Union[float, str, Dict, List],
        output_name: str,
        formula: str = ""
    ) -> None:
        """Add a calculation step to provenance tracking."""
        self._step_counter += 1
        self._tracker.add_step(
            step_number=self._step_counter,
            description=description,
            operation=operation,
            inputs=inputs,
            output_value=output_value,
            output_name=output_name,
            formula=formula
        )

    def _validate_inputs(self, inputs: DispatchOptimizerInput) -> None:
        """Validate input parameters."""
        if not inputs.equipment_units:
            raise ValueError("At least one equipment unit required")

        if inputs.horizon_hours <= 0:
            raise ValueError("Horizon must be positive")

        if inputs.horizon_hours > 8760:
            raise ValueError("Horizon cannot exceed 8760 hours (1 year)")

        for eq in inputs.equipment_units:
            if eq.capacity_kw <= 0:
                raise ValueError(f"Equipment {eq.equipment_id} capacity must be positive")
            if not 0 <= eq.min_load_fraction <= eq.max_load_fraction <= 1:
                raise ValueError(f"Equipment {eq.equipment_id} load fractions invalid")
            if eq.efficiency_nominal <= 0 or eq.efficiency_nominal > 10:
                raise ValueError(f"Equipment {eq.equipment_id} efficiency invalid")

        if inputs.thermal_storage:
            if inputs.thermal_storage.capacity_kwh <= 0:
                raise ValueError("Storage capacity must be positive")
            if not 0 <= inputs.thermal_storage.initial_soc <= 1:
                raise ValueError("Storage initial SOC must be 0-1")

    def _dispatch_hour(
        self,
        inputs: DispatchOptimizerInput,
        hour: int,
        load_kw: float,
        load_req: HourlyLoadRequirement,
        equipment_state: Dict[str, Dict]
    ) -> Tuple[List[EquipmentDispatch], float]:
        """
        Dispatch equipment for a single hour.

        Uses a greedy algorithm based on dispatch mode:
        - COST_OPTIMAL: Lowest marginal cost first
        - CARBON_OPTIMAL: Lowest carbon intensity first
        - EFFICIENCY_OPTIMAL: Highest efficiency first

        Args:
            inputs: Optimizer inputs
            hour: Hour index
            load_kw: Load to serve (kW)
            load_req: Full load requirement data
            equipment_state: Current equipment states

        Returns:
            Tuple of (list of dispatches, remaining unmet load)
        """
        dispatches = []
        remaining_load = load_kw

        # Filter available equipment
        available_equipment = [
            eq for eq in inputs.equipment_units
            if eq.is_available and self._can_start_or_run(eq, equipment_state[eq.equipment_id], inputs.time_step_minutes)
        ]

        # Sort equipment by dispatch priority based on mode
        if inputs.dispatch_mode == DispatchMode.COST_OPTIMAL:
            sorted_equipment = sorted(
                available_equipment,
                key=lambda eq: self._calculate_marginal_cost(eq, load_req)
            )
        elif inputs.dispatch_mode == DispatchMode.CARBON_OPTIMAL:
            sorted_equipment = sorted(
                available_equipment,
                key=lambda eq: eq.carbon_intensity_kg_per_kwh / max(eq.efficiency_nominal, 0.01)
            )
        elif inputs.dispatch_mode == DispatchMode.EFFICIENCY_OPTIMAL:
            sorted_equipment = sorted(
                available_equipment,
                key=lambda eq: -eq.efficiency_nominal
            )
        else:
            # Default: by priority
            sorted_equipment = sorted(
                available_equipment,
                key=lambda eq: eq.priority
            )

        # Dispatch equipment in order
        for eq in sorted_equipment:
            if remaining_load <= 0:
                break

            state = equipment_state[eq.equipment_id]
            was_running = state["is_running"]

            # Calculate dispatch for this unit
            min_output = eq.capacity_kw * eq.min_load_fraction
            max_output = eq.capacity_kw * eq.max_load_fraction

            # Determine output level
            if remaining_load >= min_output:
                output_kw = min(remaining_load, max_output)
                load_fraction = output_kw / eq.capacity_kw

                # Calculate efficiency at this load
                efficiency = self._get_efficiency_at_load(eq, load_fraction)

                # Calculate fuel input
                fuel_input = output_kw / efficiency if efficiency > 0 else 0.0

                # Calculate operating cost
                if eq.fuel_type == "electricity":
                    operating_cost = fuel_input * load_req.electricity_price_per_kwh
                else:
                    operating_cost = fuel_input * load_req.gas_price_per_kwh

                # Calculate carbon
                if eq.fuel_type == "electricity":
                    carbon_kg = fuel_input * load_req.grid_carbon_intensity
                else:
                    carbon_kg = fuel_input * eq.carbon_intensity_kg_per_kwh

                # Check for starting
                is_starting = not was_running

                dispatch = EquipmentDispatch(
                    equipment_id=eq.equipment_id,
                    hour=hour,
                    load_kw=round(output_kw, 4),
                    load_fraction=round(load_fraction, 4),
                    efficiency=round(efficiency, 4),
                    fuel_input_kw=round(fuel_input, 4),
                    operating_cost=round(operating_cost, 4),
                    carbon_emissions_kg=round(carbon_kg, 4),
                    is_starting=is_starting,
                    is_stopping=False
                )
                dispatches.append(dispatch)

                # Update state
                state["is_running"] = True
                state["run_time_min"] += inputs.time_step_minutes
                state["off_time_min"] = 0
                if is_starting:
                    state["starts"] += 1

                remaining_load -= output_kw

            else:
                # Load too low for this unit, skip and check if stopping
                if was_running:
                    # Check minimum run time
                    if state["run_time_min"] >= eq.min_run_time_min:
                        state["is_running"] = False
                        state["run_time_min"] = 0
                        state["stops"] += 1

                        # Add zero dispatch to track stopping
                        dispatch = EquipmentDispatch(
                            equipment_id=eq.equipment_id,
                            hour=hour,
                            load_kw=0.0,
                            load_fraction=0.0,
                            efficiency=0.0,
                            fuel_input_kw=0.0,
                            operating_cost=0.0,
                            carbon_emissions_kg=0.0,
                            is_starting=False,
                            is_stopping=True
                        )
                        dispatches.append(dispatch)
                    else:
                        # Must continue running at minimum load
                        output_kw = min_output
                        load_fraction = eq.min_load_fraction
                        efficiency = self._get_efficiency_at_load(eq, load_fraction)
                        fuel_input = output_kw / efficiency if efficiency > 0 else 0.0

                        if eq.fuel_type == "electricity":
                            operating_cost = fuel_input * load_req.electricity_price_per_kwh
                            carbon_kg = fuel_input * load_req.grid_carbon_intensity
                        else:
                            operating_cost = fuel_input * load_req.gas_price_per_kwh
                            carbon_kg = fuel_input * eq.carbon_intensity_kg_per_kwh

                        dispatch = EquipmentDispatch(
                            equipment_id=eq.equipment_id,
                            hour=hour,
                            load_kw=round(output_kw, 4),
                            load_fraction=round(load_fraction, 4),
                            efficiency=round(efficiency, 4),
                            fuel_input_kw=round(fuel_input, 4),
                            operating_cost=round(operating_cost, 4),
                            carbon_emissions_kg=round(carbon_kg, 4),
                            is_starting=False,
                            is_stopping=False
                        )
                        dispatches.append(dispatch)

                        state["run_time_min"] += inputs.time_step_minutes
                        remaining_load -= output_kw
                else:
                    state["off_time_min"] += inputs.time_step_minutes

        return dispatches, max(0.0, remaining_load)

    def _can_start_or_run(
        self,
        equipment: EquipmentUnit,
        state: Dict,
        time_step_min: int
    ) -> bool:
        """Check if equipment can start or continue running."""
        if state["is_running"]:
            return True
        # Check minimum off time
        return state["off_time_min"] >= equipment.min_off_time_min

    def _calculate_marginal_cost(
        self,
        equipment: EquipmentUnit,
        load_req: HourlyLoadRequirement
    ) -> float:
        """Calculate marginal cost per kWh output for equipment."""
        efficiency = equipment.efficiency_nominal

        if equipment.fuel_type == "electricity":
            fuel_price = load_req.electricity_price_per_kwh
        else:
            fuel_price = load_req.gas_price_per_kwh

        # Marginal cost = fuel price / efficiency
        if efficiency > 0:
            return fuel_price / efficiency
        return float('inf')

    def _get_efficiency_at_load(
        self,
        equipment: EquipmentUnit,
        load_fraction: float
    ) -> float:
        """
        Get equipment efficiency at given load fraction.

        Uses part-load efficiency curve if available.

        Args:
            equipment: Equipment unit
            load_fraction: Load fraction (0-1)

        Returns:
            Efficiency at load
        """
        # Use efficiency curve if available
        if equipment.efficiency_curve:
            curve = equipment.efficiency_curve
            return self._interpolate_efficiency(
                load_fraction,
                list(curve.load_points),
                list(curve.efficiency_points)
            ) * equipment.efficiency_nominal

        # Use default curve based on equipment type
        if equipment.equipment_type == EquipmentType.BOILER:
            curve = BOILER_EFFICIENCY_CURVE
        elif equipment.equipment_type == EquipmentType.CHILLER:
            curve = CHILLER_EFFICIENCY_CURVE
        elif equipment.equipment_type == EquipmentType.HEAT_PUMP:
            curve = HEAT_PUMP_EFFICIENCY_CURVE
        else:
            return equipment.efficiency_nominal

        # Interpolate from default curve
        loads = sorted(curve.keys())
        efficiencies = [curve[l] for l in loads]
        multiplier = self._interpolate_efficiency(load_fraction, loads, efficiencies)

        return multiplier * equipment.efficiency_nominal

    def _interpolate_efficiency(
        self,
        load: float,
        loads: List[float],
        efficiencies: List[float]
    ) -> float:
        """Linear interpolation of efficiency curve."""
        if load <= loads[0]:
            return efficiencies[0]
        if load >= loads[-1]:
            return efficiencies[-1]

        for i in range(len(loads) - 1):
            if loads[i] <= load <= loads[i + 1]:
                frac = (load - loads[i]) / (loads[i + 1] - loads[i])
                return efficiencies[i] + frac * (efficiencies[i + 1] - efficiencies[i])

        return efficiencies[-1]

    def _dispatch_storage(
        self,
        inputs: DispatchOptimizerInput,
        hour: int,
        remaining_load: float,
        current_soc: float,
        load_req: HourlyLoadRequirement
    ) -> Tuple[StorageDispatch, float, float]:
        """
        Dispatch thermal storage for an hour.

        Strategy:
        - Discharge when prices are high or load exceeds equipment capacity
        - Charge when prices are low and equipment has spare capacity

        Args:
            inputs: Optimizer inputs
            hour: Hour index
            remaining_load: Unmet load after equipment dispatch
            current_soc: Current state of charge
            load_req: Load requirement

        Returns:
            Tuple of (StorageDispatch, new_soc, contribution_kw)
        """
        storage = inputs.thermal_storage
        if not storage:
            return None, current_soc, 0.0

        # Calculate standing losses
        losses = current_soc * storage.capacity_kwh * (storage.standing_loss_pct_per_hour / 100)
        losses_soc = losses / storage.capacity_kwh

        # Determine mode
        if remaining_load > 0 and current_soc > storage.min_soc:
            # Discharge to meet load
            mode = StorageMode.DISCHARGING
            available_energy = (current_soc - storage.min_soc) * storage.capacity_kwh
            max_discharge = min(storage.max_discharge_rate_kw, available_energy)
            discharge = min(remaining_load, max_discharge)
            contribution = discharge * storage.discharge_efficiency

            energy_released = discharge
            energy_stored = 0.0
            soc_change = -discharge / storage.capacity_kwh
            power = -discharge

        elif load_req.is_peak_period == False and current_soc < storage.max_soc:
            # Charge during off-peak
            mode = StorageMode.CHARGING
            available_capacity = (storage.max_soc - current_soc) * storage.capacity_kwh
            charge = min(storage.max_charge_rate_kw, available_capacity)

            energy_stored = charge * storage.charge_efficiency
            energy_released = 0.0
            contribution = 0.0
            soc_change = energy_stored / storage.capacity_kwh
            power = charge

        else:
            # Idle
            mode = StorageMode.IDLE
            energy_stored = 0.0
            energy_released = 0.0
            contribution = 0.0
            soc_change = 0.0
            power = 0.0

        # Apply losses and calculate new SOC
        new_soc = max(
            storage.min_soc,
            min(storage.max_soc, current_soc + soc_change - losses_soc)
        )

        dispatch = StorageDispatch(
            storage_id=storage.storage_id,
            hour=hour,
            mode=mode,
            power_kw=round(power, 4),
            soc_start=round(current_soc, 4),
            soc_end=round(new_soc, 4),
            energy_stored_kwh=round(energy_stored, 4),
            energy_released_kwh=round(energy_released, 4),
            losses_kwh=round(losses, 4)
        )

        return dispatch, new_soc, contribution

    def _calculate_hourly_summary(
        self,
        hour: int,
        dispatches: List[EquipmentDispatch],
        storage_dispatch: Optional[StorageDispatch],
        load_req: HourlyLoadRequirement,
        remaining_load: float,
        dr_reduction: float
    ) -> HourlyDispatchSummary:
        """Calculate summary for one hour."""
        total_load = sum(d.load_kw for d in dispatches)
        total_fuel = sum(d.fuel_input_kw for d in dispatches)
        total_cost = sum(d.operating_cost for d in dispatches)
        total_carbon = sum(d.carbon_emissions_kg for d in dispatches)
        running_count = sum(1 for d in dispatches if d.load_kw > 0)

        storage_contribution = 0.0
        if storage_dispatch and storage_dispatch.mode == StorageMode.DISCHARGING:
            storage_contribution = storage_dispatch.energy_released_kwh

        return HourlyDispatchSummary(
            hour=hour,
            total_load_kw=round(total_load + storage_contribution, 4),
            total_heating_kw=round(load_req.heating_load_kw, 4),
            total_cooling_kw=round(load_req.cooling_load_kw, 4),
            unmet_load_kw=round(remaining_load, 4),
            total_fuel_input_kw=round(total_fuel, 4),
            total_cost=round(total_cost, 4),
            total_carbon_kg=round(total_carbon, 4),
            equipment_count_running=running_count,
            storage_contribution_kw=round(storage_contribution, 4),
            dr_participation_kw=round(dr_reduction, 4)
        )

    def _calculate_start_stop_cost(
        self,
        inputs: DispatchOptimizerInput,
        equipment_state: Dict[str, Dict]
    ) -> float:
        """Calculate total start/stop costs."""
        total_cost = 0.0
        for eq in inputs.equipment_units:
            state = equipment_state[eq.equipment_id]
            total_cost += state["starts"] * eq.start_cost
            total_cost += state["stops"] * eq.stop_cost
        return total_cost

    def _calculate_dr_savings(
        self,
        inputs: DispatchOptimizerInput,
        summaries: List[HourlyDispatchSummary]
    ) -> float:
        """Calculate demand response savings."""
        savings = 0.0
        for dr_event in inputs.dr_events:
            for hour in range(dr_event.start_hour, min(dr_event.end_hour, len(summaries))):
                summary = summaries[hour]
                if summary.dr_participation_kw > 0:
                    savings += summary.dr_participation_kw * dr_event.incentive_per_kwh
        return savings

    def _calculate_utilization(
        self,
        inputs: DispatchOptimizerInput,
        dispatches: List[EquipmentDispatch]
    ) -> Dict[str, float]:
        """Calculate equipment utilization."""
        utilization = {}
        for eq in inputs.equipment_units:
            eq_dispatches = [d for d in dispatches if d.equipment_id == eq.equipment_id]
            if eq_dispatches:
                total_output = sum(d.load_kw for d in eq_dispatches)
                max_possible = eq.capacity_kw * inputs.horizon_hours
                utilization[eq.equipment_id] = total_output / max_possible if max_possible > 0 else 0.0
            else:
                utilization[eq.equipment_id] = 0.0
        return utilization


# =============================================================================
# STANDALONE CALCULATION FUNCTIONS
# =============================================================================

def calculate_part_load_efficiency(
    equipment_type: EquipmentType,
    load_fraction: float,
    nominal_efficiency: float
) -> float:
    """
    Calculate equipment efficiency at part load.

    Uses standard part-load curves based on equipment type.

    Args:
        equipment_type: Type of equipment
        load_fraction: Load as fraction of capacity (0-1)
        nominal_efficiency: Rated efficiency at full load

    Returns:
        Efficiency at specified load

    Example:
        >>> eff = calculate_part_load_efficiency(
        ...     EquipmentType.BOILER, 0.5, 0.92
        ... )
        >>> print(f"Efficiency at 50% load: {eff:.2%}")
    """
    if equipment_type == EquipmentType.BOILER:
        curve = BOILER_EFFICIENCY_CURVE
    elif equipment_type == EquipmentType.CHILLER:
        curve = CHILLER_EFFICIENCY_CURVE
    elif equipment_type == EquipmentType.HEAT_PUMP:
        curve = HEAT_PUMP_EFFICIENCY_CURVE
    else:
        return nominal_efficiency

    # Find surrounding points
    loads = sorted(curve.keys())
    if load_fraction <= loads[0]:
        multiplier = curve[loads[0]]
    elif load_fraction >= loads[-1]:
        multiplier = curve[loads[-1]]
    else:
        for i in range(len(loads) - 1):
            if loads[i] <= load_fraction <= loads[i + 1]:
                frac = (load_fraction - loads[i]) / (loads[i + 1] - loads[i])
                multiplier = curve[loads[i]] + frac * (curve[loads[i + 1]] - curve[loads[i]])
                break

    return multiplier * nominal_efficiency


def calculate_staging_order(
    equipment_list: List[EquipmentUnit],
    load_kw: float,
    mode: DispatchMode = DispatchMode.COST_OPTIMAL
) -> List[Tuple[str, float]]:
    """
    Calculate optimal equipment staging order and loads.

    Determines which equipment to run and at what load
    to efficiently meet the total load requirement.

    Args:
        equipment_list: Available equipment units
        load_kw: Total load to meet (kW)
        mode: Dispatch optimization mode

    Returns:
        List of (equipment_id, load_kw) tuples

    Example:
        >>> equipment = [EquipmentUnit(...), ...]
        >>> staging = calculate_staging_order(equipment, 500.0)
        >>> for eq_id, load in staging:
        ...     print(f"{eq_id}: {load:.1f} kW")
    """
    # Sort by efficiency or cost
    if mode == DispatchMode.EFFICIENCY_OPTIMAL:
        sorted_eq = sorted(equipment_list, key=lambda e: -e.efficiency_nominal)
    elif mode == DispatchMode.CARBON_OPTIMAL:
        sorted_eq = sorted(
            equipment_list,
            key=lambda e: e.carbon_intensity_kg_per_kwh / e.efficiency_nominal
        )
    else:
        sorted_eq = sorted(equipment_list, key=lambda e: e.priority)

    staging = []
    remaining_load = load_kw

    for eq in sorted_eq:
        if remaining_load <= 0:
            break

        if not eq.is_available:
            continue

        min_load = eq.capacity_kw * eq.min_load_fraction
        max_load = eq.capacity_kw * eq.max_load_fraction

        if remaining_load >= min_load:
            assigned_load = min(remaining_load, max_load)
            staging.append((eq.equipment_id, assigned_load))
            remaining_load -= assigned_load

    return staging


def calculate_storage_dispatch_strategy(
    storage: ThermalStorageUnit,
    hourly_prices: List[float],
    hourly_loads: List[float]
) -> List[Tuple[int, StorageMode, float]]:
    """
    Calculate optimal storage dispatch strategy.

    Uses price arbitrage: charge during low prices, discharge during high prices.

    Args:
        storage: Thermal storage unit
        hourly_prices: Electricity prices by hour
        hourly_loads: Load requirements by hour

    Returns:
        List of (hour, mode, power_kw) tuples

    Example:
        >>> storage = ThermalStorageUnit(...)
        >>> prices = [0.05, 0.04, 0.03, ..., 0.15, 0.20]
        >>> loads = [100, 120, 150, ..., 200, 180]
        >>> strategy = calculate_storage_dispatch_strategy(storage, prices, loads)
    """
    if not hourly_prices:
        return []

    # Calculate price threshold (median)
    median_price = sorted(hourly_prices)[len(hourly_prices) // 2]

    strategy = []
    current_soc = storage.initial_soc

    for hour, (price, load) in enumerate(zip(hourly_prices, hourly_loads)):
        if price > median_price * 1.2 and current_soc > storage.min_soc:
            # Discharge during high prices
            mode = StorageMode.DISCHARGING
            available = (current_soc - storage.min_soc) * storage.capacity_kwh
            power = -min(storage.max_discharge_rate_kw, available, load)
            current_soc += power / storage.capacity_kwh
        elif price < median_price * 0.8 and current_soc < storage.max_soc:
            # Charge during low prices
            mode = StorageMode.CHARGING
            available = (storage.max_soc - current_soc) * storage.capacity_kwh
            power = min(storage.max_charge_rate_kw, available)
            current_soc += power * storage.charge_efficiency / storage.capacity_kwh
        else:
            mode = StorageMode.IDLE
            power = 0.0

        # Apply standing losses
        losses = current_soc * (storage.standing_loss_pct_per_hour / 100)
        current_soc = max(storage.min_soc, current_soc - losses)

        strategy.append((hour, mode, power))

    return strategy


def calculate_demand_response_potential(
    equipment_list: List[EquipmentUnit],
    current_loads: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate demand response curtailment potential.

    Determines how much each equipment unit can reduce load
    while respecting minimum operating constraints.

    Args:
        equipment_list: Equipment units
        current_loads: Current load by equipment ID

    Returns:
        Dictionary of equipment_id -> curtailment_potential (kW)

    Example:
        >>> equipment = [EquipmentUnit(...), ...]
        >>> current = {"BOILER-1": 400.0, "BOILER-2": 300.0}
        >>> potential = calculate_demand_response_potential(equipment, current)
        >>> total_potential = sum(potential.values())
    """
    potential = {}

    for eq in equipment_list:
        current_load = current_loads.get(eq.equipment_id, 0.0)
        min_load = eq.capacity_kw * eq.min_load_fraction

        if current_load > min_load:
            # Can reduce to minimum
            potential[eq.equipment_id] = current_load - min_load
        elif current_load > 0:
            # Could shut down if minimum run time met
            potential[eq.equipment_id] = current_load
        else:
            potential[eq.equipment_id] = 0.0

    return potential


def calculate_carbon_intensity(
    equipment_list: List[EquipmentUnit],
    dispatches: List[Tuple[str, float]],
    grid_carbon_intensity: float = 0.40
) -> float:
    """
    Calculate weighted average carbon intensity of dispatch.

    Args:
        equipment_list: Equipment units
        dispatches: List of (equipment_id, load_kw) tuples
        grid_carbon_intensity: Grid electricity carbon (kg CO2/kWh)

    Returns:
        Average carbon intensity (kg CO2/kWh output)

    Example:
        >>> carbon = calculate_carbon_intensity(
        ...     equipment, dispatches, 0.45
        ... )
        >>> print(f"Carbon intensity: {carbon:.3f} kg CO2/kWh")
    """
    total_output = 0.0
    total_carbon = 0.0

    eq_lookup = {eq.equipment_id: eq for eq in equipment_list}

    for eq_id, load in dispatches:
        if eq_id in eq_lookup:
            eq = eq_lookup[eq_id]
            total_output += load

            # Calculate fuel input
            fuel_input = load / eq.efficiency_nominal

            # Get carbon intensity
            if eq.fuel_type == "electricity":
                carbon = fuel_input * grid_carbon_intensity
            else:
                carbon = fuel_input * eq.carbon_intensity_kg_per_kwh

            total_carbon += carbon

    if total_output > 0:
        return total_carbon / total_output
    return 0.0


def estimate_start_stop_costs(
    equipment: EquipmentUnit,
    num_cycles: int
) -> Decimal:
    """
    Estimate start/stop costs for equipment.

    Includes direct costs and wear-and-tear estimates.

    Args:
        equipment: Equipment unit
        num_cycles: Number of start/stop cycles

    Returns:
        Total start/stop cost as Decimal

    Example:
        >>> cost = estimate_start_stop_costs(boiler, 10)
        >>> print(f"Start/stop costs: ${cost:.2f}")
    """
    direct_cost = Decimal(str(equipment.start_cost + equipment.stop_cost))
    cycles = Decimal(str(num_cycles))

    # Add wear factor (10% of direct cost)
    wear_factor = Decimal("1.10")

    total = direct_cost * cycles * wear_factor
    return total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


@lru_cache(maxsize=1000)
def cached_efficiency_lookup(
    equipment_type: str,
    load_fraction_pct: int,
    nominal_efficiency_pct: int
) -> float:
    """
    Thread-safe cached efficiency lookup.

    Uses integer percentages for cache keys.

    Args:
        equipment_type: Equipment type string
        load_fraction_pct: Load fraction as percentage (0-100)
        nominal_efficiency_pct: Nominal efficiency as percentage

    Returns:
        Efficiency value
    """
    load_fraction = load_fraction_pct / 100.0
    nominal_efficiency = nominal_efficiency_pct / 100.0

    try:
        eq_type = EquipmentType(equipment_type)
    except ValueError:
        return nominal_efficiency

    return calculate_part_load_efficiency(eq_type, load_fraction, nominal_efficiency)


def calculate_minimum_units_required(
    equipment_list: List[EquipmentUnit],
    total_load_kw: float
) -> int:
    """
    Calculate minimum number of equipment units needed.

    Args:
        equipment_list: Available equipment units
        total_load_kw: Total load to serve (kW)

    Returns:
        Minimum number of units

    Example:
        >>> min_units = calculate_minimum_units_required(equipment, 1000.0)
        >>> print(f"Need at least {min_units} units")
    """
    if total_load_kw <= 0:
        return 0

    # Sort by capacity descending
    sorted_eq = sorted(
        [eq for eq in equipment_list if eq.is_available],
        key=lambda e: -e.capacity_kw
    )

    count = 0
    remaining = total_load_kw

    for eq in sorted_eq:
        max_output = eq.capacity_kw * eq.max_load_fraction
        remaining -= max_output
        count += 1

        if remaining <= 0:
            break

    return count


def calculate_optimal_loading(
    equipment: EquipmentUnit,
    load_options: List[float]
) -> Tuple[float, float]:
    """
    Find optimal loading point for maximum efficiency.

    Args:
        equipment: Equipment unit
        load_options: List of possible load levels (kW)

    Returns:
        Tuple of (optimal_load_kw, efficiency)

    Example:
        >>> loads = [100, 200, 300, 400, 500]
        >>> optimal, eff = calculate_optimal_loading(boiler, loads)
        >>> print(f"Optimal load: {optimal} kW at {eff:.1%} efficiency")
    """
    best_load = 0.0
    best_efficiency = 0.0

    for load in load_options:
        load_fraction = load / equipment.capacity_kw
        if equipment.min_load_fraction <= load_fraction <= equipment.max_load_fraction:
            efficiency = calculate_part_load_efficiency(
                equipment.equipment_type,
                load_fraction,
                equipment.efficiency_nominal
            )
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_load = load

    return best_load, best_efficiency
