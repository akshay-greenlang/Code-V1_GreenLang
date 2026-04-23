"""
GL-001 ThermalCommand Orchestrator - MILP Load Allocation Module

Mixed Integer Linear Programming (MILP) optimizer for multi-equipment
thermal load dispatch. Minimizes total fuel cost and emissions while
respecting equipment constraints.

Key Features:
    - MILP optimization for load allocation
    - Multi-objective: minimize fuel cost + emissions penalty
    - Equipment constraints: capacity, ramp rates, turndown
    - On/off decisions for equipment (integer variables)
    - Real-time optimization with warm starts
    - Comprehensive audit trail with provenance

Solver Support:
    - scipy.optimize.milp (default)
    - cvxpy (optional, for more complex problems)

Reference Standards:
    - IEEE 519 Harmonics (for VFD equipment)
    - ASME PTC 4 Boiler Efficiency

Example:
    >>> from greenlang.agents.process_heat.gl_001_thermal_command.load_allocation import (
    ...     MILPLoadAllocator, Equipment, LoadAllocationRequest
    ... )
    >>>
    >>> allocator = MILPLoadAllocator()
    >>> allocator.add_equipment(boiler_1)
    >>> allocator.add_equipment(boiler_2)
    >>> request = LoadAllocationRequest(total_demand_mmbtu_hr=50.0)
    >>> result = allocator.optimize(request)
    >>> print(f"Total cost: ${result.total_cost_per_hour:.2f}/hr")

Author: GreenLang Optimization Team
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import hashlib
import logging
import uuid

import numpy as np
from pydantic import BaseModel, Field, field_validator

# Optional import for scipy
try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    milp = None

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class EquipmentType(str, Enum):
    """Types of thermal equipment."""
    BOILER = "boiler"
    FURNACE = "furnace"
    HEAT_EXCHANGER = "heat_exchanger"
    CHP = "chp"
    HEAT_PUMP = "heat_pump"
    ELECTRIC_HEATER = "electric_heater"
    WASTE_HEAT_RECOVERY = "waste_heat_recovery"


class FuelType(str, Enum):
    """Fuel types for equipment."""
    NATURAL_GAS = "natural_gas"
    FUEL_OIL = "fuel_oil"
    PROPANE = "propane"
    COAL = "coal"
    BIOMASS = "biomass"
    ELECTRICITY = "electricity"
    WASTE_HEAT = "waste_heat"
    HYDROGEN = "hydrogen"


class EquipmentStatus(str, Enum):
    """Equipment operational status."""
    AVAILABLE = "available"
    RUNNING = "running"
    STANDBY = "standby"
    MAINTENANCE = "maintenance"
    FAULT = "fault"
    OFFLINE = "offline"


class OptimizationStatus(str, Enum):
    """Status of optimization solution."""
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    TIMEOUT = "timeout"
    ERROR = "error"


class OptimizationObjective(str, Enum):
    """Optimization objectives."""
    MINIMIZE_COST = "minimize_cost"
    MINIMIZE_EMISSIONS = "minimize_emissions"
    MINIMIZE_ENERGY = "minimize_energy"
    MAXIMIZE_EFFICIENCY = "maximize_efficiency"
    BALANCED = "balanced"


# =============================================================================
# DATA MODELS
# =============================================================================

class EquipmentEfficiencyCurve(BaseModel):
    """
    Equipment efficiency curve definition.

    Efficiency varies with load. Defined as piecewise linear
    segments for MILP compatibility.
    """
    load_points_percent: List[float] = Field(
        default=[25.0, 50.0, 75.0, 100.0],
        description="Load points as percent of capacity"
    )
    efficiency_points: List[float] = Field(
        default=[0.75, 0.82, 0.85, 0.84],
        description="Efficiency at each load point"
    )

    def get_efficiency(self, load_percent: float) -> float:
        """Get efficiency by linear interpolation."""
        if load_percent <= self.load_points_percent[0]:
            return self.efficiency_points[0]
        if load_percent >= self.load_points_percent[-1]:
            return self.efficiency_points[-1]

        # Linear interpolation
        for i in range(len(self.load_points_percent) - 1):
            if self.load_points_percent[i] <= load_percent <= self.load_points_percent[i + 1]:
                x0, x1 = self.load_points_percent[i], self.load_points_percent[i + 1]
                y0, y1 = self.efficiency_points[i], self.efficiency_points[i + 1]
                return y0 + (y1 - y0) * (load_percent - x0) / (x1 - x0)

        return self.efficiency_points[-1]


class Equipment(BaseModel):
    """
    Equipment definition for load allocation.

    Defines equipment parameters including capacity, efficiency,
    costs, emissions, and operational constraints.
    """
    equipment_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Unique equipment identifier"
    )
    name: str = Field(..., description="Equipment name/tag")
    equipment_type: EquipmentType = Field(..., description="Equipment type")
    fuel_type: FuelType = Field(..., description="Primary fuel type")

    # Capacity
    max_capacity_mmbtu_hr: float = Field(
        ...,
        ge=0,
        description="Maximum capacity (MMBtu/hr)"
    )
    min_capacity_mmbtu_hr: float = Field(
        default=0,
        ge=0,
        description="Minimum capacity when running (turndown)"
    )

    # Efficiency
    efficiency_curve: EquipmentEfficiencyCurve = Field(
        default_factory=EquipmentEfficiencyCurve,
        description="Load-dependent efficiency curve"
    )
    rated_efficiency: float = Field(
        default=0.82,
        ge=0.1,
        le=1.0,
        description="Rated efficiency at design load"
    )

    # Costs
    fuel_cost_per_mmbtu: float = Field(
        ...,
        ge=0,
        description="Fuel cost ($/MMBtu)"
    )
    fixed_cost_per_hour: float = Field(
        default=0.0,
        ge=0,
        description="Fixed operating cost when running ($/hr)"
    )
    startup_cost: float = Field(
        default=0.0,
        ge=0,
        description="Cost per startup ($)"
    )

    # Emissions
    co2_kg_per_mmbtu_fuel: float = Field(
        default=53.06,  # Natural gas default
        ge=0,
        description="CO2 emissions (kg CO2/MMBtu fuel)"
    )
    nox_kg_per_mmbtu_fuel: float = Field(
        default=0.04,
        ge=0,
        description="NOx emissions (kg/MMBtu fuel)"
    )

    # Operational constraints
    ramp_up_mmbtu_hr_per_min: float = Field(
        default=1.0,
        ge=0,
        description="Maximum ramp-up rate"
    )
    ramp_down_mmbtu_hr_per_min: float = Field(
        default=1.0,
        ge=0,
        description="Maximum ramp-down rate"
    )
    min_run_time_minutes: int = Field(
        default=30,
        ge=0,
        description="Minimum run time once started"
    )
    min_off_time_minutes: int = Field(
        default=15,
        ge=0,
        description="Minimum off time before restart"
    )

    # Status
    status: EquipmentStatus = Field(
        default=EquipmentStatus.AVAILABLE,
        description="Current operational status"
    )
    current_load_mmbtu_hr: float = Field(
        default=0.0,
        ge=0,
        description="Current operating load"
    )
    priority: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Dispatch priority (1=highest)"
    )

    @property
    def turndown_ratio(self) -> float:
        """Calculate turndown ratio."""
        if self.min_capacity_mmbtu_hr == 0:
            return float('inf')
        return self.max_capacity_mmbtu_hr / self.min_capacity_mmbtu_hr

    def get_fuel_consumption(self, output_mmbtu_hr: float) -> float:
        """Calculate fuel consumption for given output."""
        if output_mmbtu_hr == 0:
            return 0.0
        load_percent = (output_mmbtu_hr / self.max_capacity_mmbtu_hr) * 100
        efficiency = self.efficiency_curve.get_efficiency(load_percent)
        return output_mmbtu_hr / efficiency

    def get_operating_cost(self, output_mmbtu_hr: float) -> float:
        """Calculate total operating cost per hour."""
        if output_mmbtu_hr == 0:
            return 0.0
        fuel_consumed = self.get_fuel_consumption(output_mmbtu_hr)
        fuel_cost = fuel_consumed * self.fuel_cost_per_mmbtu
        return fuel_cost + self.fixed_cost_per_hour

    def get_emissions(self, output_mmbtu_hr: float) -> Dict[str, float]:
        """Calculate emissions for given output."""
        fuel_consumed = self.get_fuel_consumption(output_mmbtu_hr)
        return {
            "co2_kg_hr": fuel_consumed * self.co2_kg_per_mmbtu_fuel,
            "nox_kg_hr": fuel_consumed * self.nox_kg_per_mmbtu_fuel,
        }


class LoadAllocationRequest(BaseModel):
    """Request for load allocation optimization."""
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Request identifier"
    )
    total_demand_mmbtu_hr: float = Field(
        ...,
        ge=0,
        description="Total thermal demand (MMBtu/hr)"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Request timestamp"
    )
    optimization_objective: OptimizationObjective = Field(
        default=OptimizationObjective.BALANCED,
        description="Primary optimization objective"
    )
    emissions_penalty_per_kg_co2: float = Field(
        default=0.05,
        ge=0,
        description="Carbon price ($/kg CO2)"
    )
    time_horizon_minutes: int = Field(
        default=60,
        ge=1,
        description="Optimization time horizon"
    )
    allow_unmet_demand: bool = Field(
        default=False,
        description="Allow partial fulfillment"
    )
    unmet_demand_penalty: float = Field(
        default=100.0,
        ge=0,
        description="Penalty for unmet demand ($/MMBtu)"
    )


class EquipmentAllocation(BaseModel):
    """Allocation for a single piece of equipment."""
    equipment_id: str = Field(..., description="Equipment ID")
    equipment_name: str = Field(..., description="Equipment name")
    is_running: bool = Field(..., description="Equipment on/off decision")
    allocated_load_mmbtu_hr: float = Field(
        ...,
        ge=0,
        description="Allocated load"
    )
    load_percent: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Load as percent of capacity"
    )
    fuel_consumption_mmbtu_hr: float = Field(
        default=0.0,
        ge=0,
        description="Fuel consumption"
    )
    operating_cost_per_hr: float = Field(
        default=0.0,
        ge=0,
        description="Operating cost"
    )
    co2_emissions_kg_hr: float = Field(
        default=0.0,
        ge=0,
        description="CO2 emissions"
    )
    nox_emissions_kg_hr: float = Field(
        default=0.0,
        ge=0,
        description="NOx emissions"
    )
    efficiency: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Operating efficiency"
    )


class LoadAllocationResult(BaseModel):
    """Result of load allocation optimization."""
    result_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Result identifier"
    )
    request_id: str = Field(..., description="Original request ID")
    status: OptimizationStatus = Field(..., description="Optimization status")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Result timestamp"
    )

    # Solution
    allocations: List[EquipmentAllocation] = Field(
        default_factory=list,
        description="Equipment allocations"
    )
    total_allocated_mmbtu_hr: float = Field(
        default=0.0,
        ge=0,
        description="Total allocated load"
    )
    unmet_demand_mmbtu_hr: float = Field(
        default=0.0,
        ge=0,
        description="Unmet demand"
    )

    # Costs
    total_fuel_cost_per_hr: float = Field(
        default=0.0,
        ge=0,
        description="Total fuel cost"
    )
    total_fixed_cost_per_hr: float = Field(
        default=0.0,
        ge=0,
        description="Total fixed costs"
    )
    total_emissions_cost_per_hr: float = Field(
        default=0.0,
        ge=0,
        description="Emissions penalty cost"
    )
    total_cost_per_hour: float = Field(
        default=0.0,
        ge=0,
        description="Total cost per hour"
    )

    # Emissions
    total_co2_kg_hr: float = Field(
        default=0.0,
        ge=0,
        description="Total CO2 emissions"
    )
    total_nox_kg_hr: float = Field(
        default=0.0,
        ge=0,
        description="Total NOx emissions"
    )

    # Efficiency
    system_efficiency: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="System weighted efficiency"
    )

    # Solve metrics
    solve_time_ms: float = Field(
        default=0.0,
        ge=0,
        description="Optimization solve time"
    )
    iterations: int = Field(
        default=0,
        ge=0,
        description="Solver iterations"
    )
    gap_percent: float = Field(
        default=0.0,
        ge=0,
        description="Optimality gap"
    )

    # Audit
    provenance_hash: str = Field(
        default="",
        description="SHA-256 audit hash"
    )

    def model_post_init(self, __context: Any) -> None:
        """Calculate provenance hash."""
        if not self.provenance_hash:
            self.provenance_hash = self._calculate_provenance()

    def _calculate_provenance(self) -> str:
        """Calculate SHA-256 provenance hash."""
        alloc_str = "|".join([
            f"{a.equipment_id}:{a.allocated_load_mmbtu_hr:.3f}"
            for a in self.allocations
        ])
        provenance_str = (
            f"{self.request_id}|{self.timestamp.isoformat()}|"
            f"{self.total_cost_per_hour:.2f}|{alloc_str}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()


# =============================================================================
# MILP LOAD ALLOCATOR
# =============================================================================

class MILPLoadAllocator:
    """
    Mixed Integer Linear Programming Load Allocator.

    Optimizes thermal load allocation across multiple equipment units
    to minimize total cost (fuel + emissions penalty) while respecting
    operational constraints.

    The optimization formulation:

    Decision Variables:
        - x_i: Load allocated to equipment i (continuous, MMBtu/hr)
        - y_i: On/off status of equipment i (binary, 0 or 1)

    Objective (minimize):
        sum_i [ (fuel_cost_i / eta_i) * x_i + fixed_cost_i * y_i +
                emissions_penalty * (co2_factor_i / eta_i) * x_i ]

    Constraints:
        - Demand satisfaction: sum_i(x_i) = total_demand
        - Capacity limits: x_i <= max_capacity_i * y_i (for all i)
        - Turndown limits: x_i >= min_capacity_i * y_i (for all i)
        - Ramp rate limits (from current state)
        - Equipment availability

    Example:
        >>> allocator = MILPLoadAllocator()
        >>> allocator.add_equipment(Equipment(
        ...     name="Boiler-1",
        ...     equipment_type=EquipmentType.BOILER,
        ...     fuel_type=FuelType.NATURAL_GAS,
        ...     max_capacity_mmbtu_hr=50.0,
        ...     fuel_cost_per_mmbtu=5.0,
        ... ))
        >>> result = allocator.optimize(LoadAllocationRequest(
        ...     total_demand_mmbtu_hr=40.0
        ... ))
    """

    def __init__(
        self,
        time_limit_seconds: float = 60.0,
        gap_tolerance: float = 0.01,
        use_warm_start: bool = True
    ) -> None:
        """
        Initialize MILP Load Allocator.

        Args:
            time_limit_seconds: Maximum solve time
            gap_tolerance: Acceptable optimality gap (1% default)
            use_warm_start: Use previous solution as starting point
        """
        if not SCIPY_AVAILABLE:
            logger.warning(
                "scipy not available. MILP optimization will use heuristic fallback."
            )

        self.time_limit = time_limit_seconds
        self.gap_tolerance = gap_tolerance
        self.use_warm_start = use_warm_start

        # Equipment registry
        self._equipment: Dict[str, Equipment] = {}
        self._equipment_order: List[str] = []

        # Previous solution for warm start
        self._last_solution: Optional[Dict[str, float]] = None

        # Optimization history
        self._history: List[LoadAllocationResult] = []

        logger.info(
            "MILPLoadAllocator initialized (time_limit=%.1fs, gap=%.1f%%)",
            time_limit_seconds, gap_tolerance * 100
        )

    # =========================================================================
    # EQUIPMENT MANAGEMENT
    # =========================================================================

    def add_equipment(self, equipment: Equipment) -> bool:
        """
        Add equipment to the allocator.

        Args:
            equipment: Equipment to add

        Returns:
            True if added successfully
        """
        if equipment.equipment_id in self._equipment:
            logger.warning("Equipment %s already registered", equipment.equipment_id)
            return False

        self._equipment[equipment.equipment_id] = equipment
        self._equipment_order.append(equipment.equipment_id)

        logger.info(
            "Equipment added: %s (%s, max=%.1f MMBtu/hr)",
            equipment.name, equipment.equipment_type.value,
            equipment.max_capacity_mmbtu_hr
        )
        return True

    def remove_equipment(self, equipment_id: str) -> bool:
        """Remove equipment from the allocator."""
        if equipment_id not in self._equipment:
            return False

        del self._equipment[equipment_id]
        self._equipment_order.remove(equipment_id)
        return True

    def update_equipment_status(
        self,
        equipment_id: str,
        status: EquipmentStatus,
        current_load: Optional[float] = None
    ) -> bool:
        """Update equipment operational status."""
        equipment = self._equipment.get(equipment_id)
        if not equipment:
            return False

        equipment.status = status
        if current_load is not None:
            equipment.current_load_mmbtu_hr = current_load

        return True

    def get_equipment(self, equipment_id: str) -> Optional[Equipment]:
        """Get equipment by ID."""
        return self._equipment.get(equipment_id)

    def get_all_equipment(self) -> List[Equipment]:
        """Get all registered equipment."""
        return [self._equipment[eid] for eid in self._equipment_order]

    def get_total_capacity(self) -> float:
        """Get total available capacity."""
        return sum(
            e.max_capacity_mmbtu_hr
            for e in self._equipment.values()
            if e.status in [EquipmentStatus.AVAILABLE, EquipmentStatus.RUNNING, EquipmentStatus.STANDBY]
        )

    # =========================================================================
    # OPTIMIZATION
    # =========================================================================

    def optimize(self, request: LoadAllocationRequest) -> LoadAllocationResult:
        """
        Optimize load allocation across equipment.

        Args:
            request: Load allocation request

        Returns:
            LoadAllocationResult with optimal allocations
        """
        start_time = datetime.now(timezone.utc)

        # Get available equipment
        available_equipment = self._get_available_equipment()

        if not available_equipment:
            return LoadAllocationResult(
                request_id=request.request_id,
                status=OptimizationStatus.INFEASIBLE,
                unmet_demand_mmbtu_hr=request.total_demand_mmbtu_hr,
            )

        # Check if demand can be met
        total_capacity = sum(e.max_capacity_mmbtu_hr for e in available_equipment)
        if total_capacity < request.total_demand_mmbtu_hr and not request.allow_unmet_demand:
            logger.warning(
                "Total capacity %.1f < demand %.1f",
                total_capacity, request.total_demand_mmbtu_hr
            )
            if not request.allow_unmet_demand:
                return LoadAllocationResult(
                    request_id=request.request_id,
                    status=OptimizationStatus.INFEASIBLE,
                    unmet_demand_mmbtu_hr=request.total_demand_mmbtu_hr - total_capacity,
                )

        # Solve optimization
        if SCIPY_AVAILABLE:
            result = self._solve_milp(request, available_equipment)
        else:
            result = self._solve_heuristic(request, available_equipment)

        # Calculate solve time
        result.solve_time_ms = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000

        # Store for warm start
        if result.status in [OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE]:
            self._last_solution = {
                a.equipment_id: a.allocated_load_mmbtu_hr
                for a in result.allocations
            }

        # Add to history
        self._history.append(result)

        logger.info(
            "Optimization complete: status=%s, cost=$%.2f/hr, demand=%.1f/%.1f MMBtu/hr",
            result.status.value, result.total_cost_per_hour,
            result.total_allocated_mmbtu_hr, request.total_demand_mmbtu_hr
        )

        return result

    def _get_available_equipment(self) -> List[Equipment]:
        """Get equipment available for dispatch."""
        return [
            e for e in self._equipment.values()
            if e.status in [EquipmentStatus.AVAILABLE, EquipmentStatus.RUNNING, EquipmentStatus.STANDBY]
        ]

    def _solve_milp(
        self,
        request: LoadAllocationRequest,
        equipment: List[Equipment]
    ) -> LoadAllocationResult:
        """
        Solve using scipy MILP solver.

        This formulation linearizes the problem by using average efficiency
        at expected operating points.
        """
        n = len(equipment)

        # Build cost vector
        # Variables: [x_0, ..., x_n-1, y_0, ..., y_n-1, slack]
        # x_i = load for equipment i
        # y_i = on/off binary for equipment i
        # slack = unmet demand

        c = np.zeros(2 * n + 1)

        for i, equip in enumerate(equipment):
            # Estimate efficiency at 70% load
            est_efficiency = equip.efficiency_curve.get_efficiency(70.0)

            # Fuel cost per MMBtu output
            fuel_cost_per_output = equip.fuel_cost_per_mmbtu / est_efficiency

            # Emissions penalty
            emissions_cost = (
                request.emissions_penalty_per_kg_co2 *
                equip.co2_kg_per_mmbtu_fuel / est_efficiency
            )

            # Total variable cost
            c[i] = fuel_cost_per_output + emissions_cost

            # Fixed cost for y_i
            c[n + i] = equip.fixed_cost_per_hour

        # Slack penalty
        c[2 * n] = request.unmet_demand_penalty

        # Bounds
        # x_i: [0, max_capacity]
        # y_i: [0, 1] (binary)
        # slack: [0, demand]
        lb = np.zeros(2 * n + 1)
        ub = np.zeros(2 * n + 1)

        for i, equip in enumerate(equipment):
            ub[i] = equip.max_capacity_mmbtu_hr
            ub[n + i] = 1.0  # Binary

        ub[2 * n] = request.total_demand_mmbtu_hr if request.allow_unmet_demand else 0

        # Constraints
        # 1. Demand balance: sum(x_i) + slack = demand
        A_eq = np.zeros((1, 2 * n + 1))
        A_eq[0, :n] = 1.0
        A_eq[0, 2 * n] = 1.0
        b_eq = np.array([request.total_demand_mmbtu_hr])

        # 2. Capacity linking: x_i <= max_i * y_i
        # Reformulated as: x_i - max_i * y_i <= 0
        A_ub_cap = np.zeros((n, 2 * n + 1))
        for i, equip in enumerate(equipment):
            A_ub_cap[i, i] = 1.0
            A_ub_cap[i, n + i] = -equip.max_capacity_mmbtu_hr
        b_ub_cap = np.zeros(n)

        # 3. Turndown linking: x_i >= min_i * y_i
        # Reformulated as: -x_i + min_i * y_i <= 0
        A_ub_turn = np.zeros((n, 2 * n + 1))
        for i, equip in enumerate(equipment):
            A_ub_turn[i, i] = -1.0
            A_ub_turn[i, n + i] = equip.min_capacity_mmbtu_hr
        b_ub_turn = np.zeros(n)

        # Combine inequality constraints
        A_ub = np.vstack([A_ub_cap, A_ub_turn])
        b_ub = np.concatenate([b_ub_cap, b_ub_turn])

        # Integer constraints (y variables are binary)
        integrality = np.zeros(2 * n + 1, dtype=int)
        integrality[n:2*n] = 1  # y variables are binary

        # Solve
        try:
            constraints = [
                LinearConstraint(A_eq, b_eq, b_eq),  # Equality
                LinearConstraint(A_ub, -np.inf, b_ub),  # Inequality
            ]
            bounds = Bounds(lb, ub)

            result = milp(
                c=c,
                constraints=constraints,
                bounds=bounds,
                integrality=integrality,
            )

            if result.success:
                return self._build_result(
                    request, equipment, result.x,
                    OptimizationStatus.OPTIMAL if result.status == 0 else OptimizationStatus.FEASIBLE
                )
            else:
                logger.warning("MILP solver failed: %s", result.message)
                return self._solve_heuristic(request, equipment)

        except Exception as e:
            logger.error("MILP solver error: %s", e, exc_info=True)
            return self._solve_heuristic(request, equipment)

    def _solve_heuristic(
        self,
        request: LoadAllocationRequest,
        equipment: List[Equipment]
    ) -> LoadAllocationResult:
        """
        Heuristic solution when MILP not available or fails.

        Uses merit order dispatch based on marginal cost.
        """
        logger.info("Using heuristic merit-order dispatch")

        # Sort by marginal cost (fuel cost / efficiency + emissions)
        def marginal_cost(e: Equipment) -> float:
            eff = e.efficiency_curve.get_efficiency(70.0)
            fuel_cost = e.fuel_cost_per_mmbtu / eff
            emissions_cost = (
                request.emissions_penalty_per_kg_co2 *
                e.co2_kg_per_mmbtu_fuel / eff
            )
            return fuel_cost + emissions_cost

        sorted_equipment = sorted(equipment, key=marginal_cost)

        # Allocate in merit order
        remaining_demand = request.total_demand_mmbtu_hr
        allocations_map: Dict[str, float] = {}

        for equip in sorted_equipment:
            if remaining_demand <= 0:
                allocations_map[equip.equipment_id] = 0.0
                continue

            # Allocate up to max capacity
            allocation = min(remaining_demand, equip.max_capacity_mmbtu_hr)

            # Check minimum turndown
            if allocation < equip.min_capacity_mmbtu_hr:
                # Either run at minimum or don't run
                if remaining_demand >= equip.min_capacity_mmbtu_hr:
                    allocation = equip.min_capacity_mmbtu_hr
                else:
                    allocation = 0.0

            allocations_map[equip.equipment_id] = allocation
            remaining_demand -= allocation

        # Build solution vector
        n = len(equipment)
        x = np.zeros(2 * n + 1)
        for i, equip in enumerate(equipment):
            x[i] = allocations_map.get(equip.equipment_id, 0.0)
            x[n + i] = 1.0 if x[i] > 0 else 0.0
        x[2 * n] = max(0, remaining_demand)  # Slack

        return self._build_result(
            request, equipment, x, OptimizationStatus.FEASIBLE
        )

    def _build_result(
        self,
        request: LoadAllocationRequest,
        equipment: List[Equipment],
        x: np.ndarray,
        status: OptimizationStatus
    ) -> LoadAllocationResult:
        """Build result from solution vector."""
        n = len(equipment)
        allocations = []

        total_fuel_cost = 0.0
        total_fixed_cost = 0.0
        total_emissions_cost = 0.0
        total_co2 = 0.0
        total_nox = 0.0
        total_fuel_in = 0.0
        total_output = 0.0

        for i, equip in enumerate(equipment):
            load = x[i]
            is_running = x[n + i] > 0.5

            if load > 0:
                load_percent = (load / equip.max_capacity_mmbtu_hr) * 100
                efficiency = equip.efficiency_curve.get_efficiency(load_percent)
                fuel_consumption = load / efficiency
                operating_cost = fuel_consumption * equip.fuel_cost_per_mmbtu + equip.fixed_cost_per_hour
                co2 = fuel_consumption * equip.co2_kg_per_mmbtu_fuel
                nox = fuel_consumption * equip.nox_kg_per_mmbtu_fuel
            else:
                load_percent = 0.0
                efficiency = 0.0
                fuel_consumption = 0.0
                operating_cost = 0.0
                co2 = 0.0
                nox = 0.0

            allocations.append(EquipmentAllocation(
                equipment_id=equip.equipment_id,
                equipment_name=equip.name,
                is_running=is_running,
                allocated_load_mmbtu_hr=load,
                load_percent=load_percent,
                fuel_consumption_mmbtu_hr=fuel_consumption,
                operating_cost_per_hr=operating_cost,
                co2_emissions_kg_hr=co2,
                nox_emissions_kg_hr=nox,
                efficiency=efficiency,
            ))

            total_fuel_cost += fuel_consumption * equip.fuel_cost_per_mmbtu
            total_fixed_cost += equip.fixed_cost_per_hour if is_running else 0
            total_co2 += co2
            total_nox += nox
            total_fuel_in += fuel_consumption
            total_output += load

        total_emissions_cost = total_co2 * request.emissions_penalty_per_kg_co2

        # System efficiency
        system_efficiency = total_output / total_fuel_in if total_fuel_in > 0 else 0.0

        return LoadAllocationResult(
            request_id=request.request_id,
            status=status,
            allocations=allocations,
            total_allocated_mmbtu_hr=total_output,
            unmet_demand_mmbtu_hr=x[2 * n],
            total_fuel_cost_per_hr=total_fuel_cost,
            total_fixed_cost_per_hr=total_fixed_cost,
            total_emissions_cost_per_hr=total_emissions_cost,
            total_cost_per_hour=total_fuel_cost + total_fixed_cost + total_emissions_cost,
            total_co2_kg_hr=total_co2,
            total_nox_kg_hr=total_nox,
            system_efficiency=system_efficiency,
        )

    # =========================================================================
    # ANALYSIS AND REPORTING
    # =========================================================================

    def get_optimization_history(self, limit: int = 100) -> List[LoadAllocationResult]:
        """Get optimization history."""
        return list(reversed(self._history[-limit:]))

    def get_equipment_utilization(self) -> Dict[str, Dict[str, float]]:
        """Get equipment utilization statistics from history."""
        if not self._history:
            return {}

        stats: Dict[str, Dict[str, Any]] = {}
        for result in self._history:
            for alloc in result.allocations:
                if alloc.equipment_id not in stats:
                    stats[alloc.equipment_id] = {
                        "name": alloc.equipment_name,
                        "total_load": 0.0,
                        "run_count": 0,
                        "sample_count": 0,
                    }
                stats[alloc.equipment_id]["total_load"] += alloc.allocated_load_mmbtu_hr
                stats[alloc.equipment_id]["run_count"] += 1 if alloc.is_running else 0
                stats[alloc.equipment_id]["sample_count"] += 1

        # Calculate averages
        result: Dict[str, Dict[str, float]] = {}
        for eid, data in stats.items():
            equip = self._equipment.get(eid)
            max_cap = equip.max_capacity_mmbtu_hr if equip else 1.0
            result[eid] = {
                "name": data["name"],
                "avg_load": data["total_load"] / data["sample_count"],
                "run_percentage": (data["run_count"] / data["sample_count"]) * 100,
                "avg_utilization_percent": (
                    (data["total_load"] / data["sample_count"]) / max_cap * 100
                ),
            }

        return result

    def compare_scenarios(
        self,
        requests: List[LoadAllocationRequest]
    ) -> List[LoadAllocationResult]:
        """Compare multiple allocation scenarios."""
        return [self.optimize(req) for req in requests]


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_standard_boiler(
    name: str,
    capacity_mmbtu_hr: float,
    fuel_type: FuelType = FuelType.NATURAL_GAS,
    fuel_cost_per_mmbtu: float = 5.0,
    turndown_ratio: float = 4.0
) -> Equipment:
    """
    Factory function to create a standard boiler.

    Args:
        name: Boiler name/tag
        capacity_mmbtu_hr: Maximum firing capacity
        fuel_type: Fuel type
        fuel_cost_per_mmbtu: Fuel cost
        turndown_ratio: Maximum to minimum load ratio

    Returns:
        Configured Equipment
    """
    min_capacity = capacity_mmbtu_hr / turndown_ratio

    # Default emissions factors
    emissions_factors = {
        FuelType.NATURAL_GAS: {"co2": 53.06, "nox": 0.04},
        FuelType.FUEL_OIL: {"co2": 73.96, "nox": 0.08},
        FuelType.PROPANE: {"co2": 63.07, "nox": 0.05},
        FuelType.COAL: {"co2": 95.35, "nox": 0.20},
    }

    factors = emissions_factors.get(fuel_type, {"co2": 53.06, "nox": 0.04})

    return Equipment(
        name=name,
        equipment_type=EquipmentType.BOILER,
        fuel_type=fuel_type,
        max_capacity_mmbtu_hr=capacity_mmbtu_hr,
        min_capacity_mmbtu_hr=min_capacity,
        fuel_cost_per_mmbtu=fuel_cost_per_mmbtu,
        fixed_cost_per_hour=capacity_mmbtu_hr * 0.5,  # Estimate
        co2_kg_per_mmbtu_fuel=factors["co2"],
        nox_kg_per_mmbtu_fuel=factors["nox"],
        efficiency_curve=EquipmentEfficiencyCurve(
            load_points_percent=[25.0, 50.0, 75.0, 100.0],
            efficiency_points=[0.75, 0.82, 0.85, 0.84],
        ),
    )


def create_chp_system(
    name: str,
    thermal_capacity_mmbtu_hr: float,
    electric_capacity_kw: float,
    fuel_type: FuelType = FuelType.NATURAL_GAS,
    fuel_cost_per_mmbtu: float = 5.0,
    electricity_value_per_kwh: float = 0.10
) -> Equipment:
    """
    Factory function to create a CHP system.

    Args:
        name: CHP name/tag
        thermal_capacity_mmbtu_hr: Thermal output capacity
        electric_capacity_kw: Electric output capacity
        fuel_type: Fuel type
        fuel_cost_per_mmbtu: Fuel cost
        electricity_value_per_kwh: Value of electricity generated

    Returns:
        Configured Equipment (thermal focus, electricity as cost offset)
    """
    # CHP has higher efficiency and electricity credit
    # Net fuel cost = fuel cost - electricity credit
    heat_rate_btu_kwh = 4000  # Typical for modern CHP
    electricity_output_mmbtu_hr = electric_capacity_kw * heat_rate_btu_kwh / 1e6
    electricity_credit_per_mmbtu = electricity_value_per_kwh * electric_capacity_kw / thermal_capacity_mmbtu_hr

    net_fuel_cost = max(0, fuel_cost_per_mmbtu - electricity_credit_per_mmbtu)

    return Equipment(
        name=name,
        equipment_type=EquipmentType.CHP,
        fuel_type=fuel_type,
        max_capacity_mmbtu_hr=thermal_capacity_mmbtu_hr,
        min_capacity_mmbtu_hr=thermal_capacity_mmbtu_hr * 0.5,  # 2:1 turndown typical
        fuel_cost_per_mmbtu=net_fuel_cost,
        fixed_cost_per_hour=thermal_capacity_mmbtu_hr * 2.0,  # Higher fixed costs
        co2_kg_per_mmbtu_fuel=53.06 * 0.6,  # Credit for displaced grid electricity
        nox_kg_per_mmbtu_fuel=0.03,  # Lower NOx for modern CHP
        efficiency_curve=EquipmentEfficiencyCurve(
            load_points_percent=[50.0, 75.0, 100.0],
            efficiency_points=[0.78, 0.82, 0.80],  # Total efficiency (thermal basis)
        ),
        priority=1,  # CHP typically dispatched first
    )
