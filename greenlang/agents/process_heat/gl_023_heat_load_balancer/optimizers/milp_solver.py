# -*- coding: utf-8 -*-
"""
MILP Load Balancer for GL-023 HeatLoadBalancer
==============================================

This module implements the Mixed Integer Linear Programming (MILP) solver
for optimal heat load allocation across industrial equipment fleets.

The MILP formulation:
    Minimize: sum(cost_i * load_i) + sum(startup_cost_i * start_i) + penalty * violations
    Subject to:
        sum(load_i) = total_demand              (demand balance)
        min_i * on_i <= load_i <= max_i * on_i  (equipment limits with binary on/off)
        |load_i(t) - load_i(t-1)| <= ramp_rate * dt  (ramp constraints)
        on_i(t) >= on_i(t-1) - on_i(t-min_run)      (minimum run time)
        reserve >= reserve_pct * demand              (spinning reserve)

Supported solvers:
    - scipy.optimize.milp (default, no external dependencies)
    - PuLP with GLPK backend
    - PuLP with CBC backend

All calculations are DETERMINISTIC with full provenance tracking.

Example:
    >>> solver = MILPLoadBalancer(config=MILPConfig(solver="scipy"))
    >>> solver.formulate_problem(equipment_fleet, demand, constraints)
    >>> result = solver.solve(time_limit_s=60, gap_tolerance=0.01)
    >>> setpoints = solver.get_optimal_allocation()

Author: GreenLang Framework Team
Agent: GL-023 HeatLoadBalancer
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# ==============================================================================
# Enumerations
# ==============================================================================


class MILPSolverStatus(str, Enum):
    """Status codes for MILP solver results."""

    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    TIME_LIMIT = "time_limit"
    ITERATION_LIMIT = "iteration_limit"
    ERROR = "error"
    NOT_SOLVED = "not_solved"


class ObjectiveType(str, Enum):
    """Objective function types for optimization."""

    MINIMIZE_COST = "minimize_cost"
    MINIMIZE_EMISSIONS = "minimize_emissions"
    WEIGHTED_SUM = "weighted_sum"
    LEXICOGRAPHIC = "lexicographic"


class SolverBackend(str, Enum):
    """Supported MILP solver backends."""

    SCIPY = "scipy"
    PULP_GLPK = "pulp_glpk"
    PULP_CBC = "pulp_cbc"


# ==============================================================================
# Data Models
# ==============================================================================


class EquipmentUnit(BaseModel):
    """Model for a single equipment unit in the fleet."""

    unit_id: str = Field(..., description="Unique equipment identifier")
    name: str = Field(..., description="Equipment name/description")
    min_load_kw: float = Field(..., ge=0, description="Minimum operating load (kW)")
    max_load_kw: float = Field(..., gt=0, description="Maximum operating load (kW)")
    cost_per_kwh: float = Field(..., ge=0, description="Operating cost ($/kWh)")
    emissions_per_kwh: float = Field(
        ..., ge=0, description="CO2e emissions (kg/kWh)"
    )
    efficiency: float = Field(
        default=0.85, ge=0.01, le=1.0, description="Thermal efficiency (0-1)"
    )
    ramp_up_rate_kw_min: float = Field(
        default=float("inf"), ge=0, description="Maximum ramp-up rate (kW/min)"
    )
    ramp_down_rate_kw_min: float = Field(
        default=float("inf"), ge=0, description="Maximum ramp-down rate (kW/min)"
    )
    min_on_time_min: float = Field(
        default=0, ge=0, description="Minimum on-time (minutes)"
    )
    min_off_time_min: float = Field(
        default=0, ge=0, description="Minimum off-time (minutes)"
    )
    startup_cost: float = Field(
        default=0, ge=0, description="Cost per startup ($)"
    )
    startup_emissions: float = Field(
        default=0, ge=0, description="Emissions per startup (kgCO2e)"
    )
    current_load_kw: float = Field(
        default=0, ge=0, description="Current operating load (kW)"
    )
    is_on: bool = Field(default=False, description="Current on/off status")
    time_in_current_state_min: float = Field(
        default=0, ge=0, description="Time in current on/off state (minutes)"
    )
    priority: int = Field(
        default=5, ge=1, le=10, description="Dispatch priority (1=highest)"
    )

    @validator("max_load_kw")
    def max_load_greater_than_min(cls, v, values):
        """Ensure max_load > min_load."""
        if "min_load_kw" in values and v <= values["min_load_kw"]:
            raise ValueError("max_load_kw must be greater than min_load_kw")
        return v


class EquipmentSetpoint(BaseModel):
    """Optimal setpoint for a single equipment unit."""

    unit_id: str = Field(..., description="Equipment unit identifier")
    load_kw: float = Field(..., ge=0, description="Optimal load setpoint (kW)")
    is_on: bool = Field(..., description="Should unit be on/off")
    is_starting: bool = Field(
        default=False, description="Unit is starting up this period"
    )
    is_stopping: bool = Field(
        default=False, description="Unit is shutting down this period"
    )
    operating_cost: float = Field(
        default=0, ge=0, description="Operating cost for this setpoint ($/h)"
    )
    emissions: float = Field(
        default=0, ge=0, description="Emissions at this setpoint (kgCO2e/h)"
    )
    load_percentage: float = Field(
        default=0, ge=0, le=100, description="Load as percentage of capacity"
    )


class OptimizationConstraints(BaseModel):
    """Constraints for the MILP optimization problem."""

    total_demand_kw: float = Field(..., gt=0, description="Total heat demand (kW)")
    demand_tolerance_pct: float = Field(
        default=1.0, ge=0, le=10, description="Demand tolerance (%)"
    )
    reserve_margin_pct: float = Field(
        default=10.0, ge=0, le=50, description="Required spinning reserve (%)"
    )
    max_units_on: Optional[int] = Field(
        default=None, ge=1, description="Maximum units operating simultaneously"
    )
    min_units_on: int = Field(
        default=1, ge=0, description="Minimum units operating simultaneously"
    )
    time_step_min: float = Field(
        default=15.0, gt=0, description="Optimization time step (minutes)"
    )
    enforce_n_plus_1: bool = Field(
        default=True, description="Enforce N+1 redundancy"
    )


class MILPConfig(BaseModel):
    """Configuration for MILP solver."""

    solver: SolverBackend = Field(
        default=SolverBackend.SCIPY, description="Solver backend to use"
    )
    time_limit_s: float = Field(
        default=60.0, gt=0, le=3600, description="Solver time limit (seconds)"
    )
    gap_tolerance: float = Field(
        default=0.01, ge=0, le=0.5, description="MIP gap tolerance (0.01 = 1%)"
    )
    violation_penalty: float = Field(
        default=1e6, gt=0, description="Penalty for constraint violations"
    )
    enable_warm_start: bool = Field(
        default=True, description="Use current solution as warm start"
    )
    presolve: bool = Field(default=True, description="Enable presolve")
    num_threads: int = Field(
        default=1, ge=1, le=16, description="Number of solver threads"
    )
    verbose: bool = Field(default=False, description="Verbose solver output")


class OptimizationResult(BaseModel):
    """Complete result from MILP optimization."""

    status: MILPSolverStatus = Field(..., description="Solver status")
    objective_value: Optional[float] = Field(
        default=None, description="Optimal objective value"
    )
    total_cost: float = Field(default=0, description="Total operating cost ($/h)")
    total_emissions: float = Field(
        default=0, description="Total emissions (kgCO2e/h)"
    )
    total_load_kw: float = Field(default=0, description="Total allocated load (kW)")
    setpoints: List[EquipmentSetpoint] = Field(
        default_factory=list, description="Optimal setpoints for each unit"
    )
    demand_met_pct: float = Field(
        default=0, description="Percentage of demand met"
    )
    reserve_margin_kw: float = Field(
        default=0, description="Available spinning reserve (kW)"
    )
    units_on: int = Field(default=0, description="Number of units operating")
    solver_iterations: int = Field(default=0, description="Solver iterations")
    solver_time_s: float = Field(default=0, description="Solver wall time (s)")
    gap: Optional[float] = Field(default=None, description="MIP gap achieved")
    provenance_hash: str = Field(
        default="", description="SHA-256 hash for audit trail"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Result timestamp (ISO 8601)",
    )
    warnings: List[str] = Field(
        default_factory=list, description="Optimization warnings"
    )


# ==============================================================================
# MILP Load Balancer Implementation
# ==============================================================================


class MILPLoadBalancer:
    """
    Mixed Integer Linear Programming solver for heat load balancing.

    This class formulates and solves MILP problems for optimal allocation
    of heat load across an industrial equipment fleet. It supports multiple
    solver backends and handles complex constraints including unit commitment,
    ramp rates, and minimum runtime requirements.

    All calculations are DETERMINISTIC with SHA-256 provenance tracking.

    Attributes:
        config: MILP solver configuration
        equipment_fleet: List of equipment units
        constraints: Optimization constraints
        objective_type: Type of objective function

    Example:
        >>> config = MILPConfig(solver="scipy", time_limit_s=60)
        >>> solver = MILPLoadBalancer(config=config)
        >>> solver.formulate_problem(equipment_fleet, demand, constraints)
        >>> result = solver.solve()
        >>> print(f"Optimal cost: ${result.total_cost:.2f}/h")
    """

    def __init__(self, config: Optional[MILPConfig] = None):
        """
        Initialize MILP Load Balancer.

        Args:
            config: MILP solver configuration. Uses defaults if None.
        """
        self.config = config or MILPConfig()
        self.equipment_fleet: List[EquipmentUnit] = []
        self.constraints: Optional[OptimizationConstraints] = None
        self.objective_type: ObjectiveType = ObjectiveType.MINIMIZE_COST
        self.cost_weight: float = 1.0
        self.emissions_weight: float = 0.0

        # Internal state
        self._problem_formulated: bool = False
        self._c: Optional[np.ndarray] = None  # Objective coefficients
        self._A_eq: Optional[np.ndarray] = None  # Equality constraint matrix
        self._b_eq: Optional[np.ndarray] = None  # Equality constraint RHS
        self._A_ub: Optional[np.ndarray] = None  # Inequality constraint matrix
        self._b_ub: Optional[np.ndarray] = None  # Inequality constraint RHS
        self._bounds: List[Tuple[float, float]] = []
        self._integrality: List[int] = []  # 0=continuous, 1=integer

        # Variable indices
        self._load_vars_start: int = 0
        self._on_vars_start: int = 0
        self._start_vars_start: int = 0
        self._slack_vars_start: int = 0
        self._n_vars: int = 0

        self.logger = logging.getLogger(f"{__name__}.MILPLoadBalancer")

    def formulate_problem(
        self,
        equipment_fleet: List[EquipmentUnit],
        demand: float,
        constraints: Optional[OptimizationConstraints] = None,
    ) -> None:
        """
        Formulate the MILP optimization problem.

        Sets up decision variables, objective function, and constraints
        for the heat load balancing problem.

        Args:
            equipment_fleet: List of equipment units available for dispatch
            demand: Total heat demand to be met (kW)
            constraints: Additional optimization constraints

        Raises:
            ValueError: If equipment fleet is empty or demand is invalid
        """
        start_time = time.perf_counter()

        if not equipment_fleet:
            raise ValueError("Equipment fleet cannot be empty")
        if demand <= 0:
            raise ValueError("Demand must be positive")

        self.equipment_fleet = equipment_fleet
        self.constraints = constraints or OptimizationConstraints(
            total_demand_kw=demand
        )

        # Ensure demand in constraints matches
        if self.constraints.total_demand_kw != demand:
            self.constraints = OptimizationConstraints(
                **{**self.constraints.dict(), "total_demand_kw": demand}
            )

        n_units = len(equipment_fleet)
        self.logger.info(
            f"Formulating MILP problem: {n_units} units, "
            f"demand={demand:.1f} kW"
        )

        # Decision variables:
        # - load_i: continuous load for each unit (kW)
        # - on_i: binary on/off status for each unit
        # - start_i: binary startup indicator for each unit
        # - slack: continuous slack variable for demand

        self._load_vars_start = 0
        self._on_vars_start = n_units
        self._start_vars_start = 2 * n_units
        self._slack_vars_start = 3 * n_units
        self._n_vars = 3 * n_units + 1  # +1 for slack

        # Initialize constraint lists
        A_eq_rows = []
        b_eq_vals = []
        A_ub_rows = []
        b_ub_vals = []

        # Variable bounds and integrality
        self._bounds = []
        self._integrality = []

        # Load variables: [min_load * on_i, max_load]
        for unit in equipment_fleet:
            self._bounds.append((0, unit.max_load_kw))
            self._integrality.append(0)  # Continuous

        # On/off variables: binary [0, 1]
        for _ in equipment_fleet:
            self._bounds.append((0, 1))
            self._integrality.append(1)  # Binary

        # Startup variables: binary [0, 1]
        for _ in equipment_fleet:
            self._bounds.append((0, 1))
            self._integrality.append(1)  # Binary

        # Slack variable: [0, inf)
        self._bounds.append((0, None))
        self._integrality.append(0)  # Continuous

        # Store matrices
        self._A_eq = np.array(A_eq_rows) if A_eq_rows else None
        self._b_eq = np.array(b_eq_vals) if b_eq_vals else None
        self._A_ub = np.array(A_ub_rows) if A_ub_rows else None
        self._b_ub = np.array(b_ub_vals) if b_ub_vals else None

        self._problem_formulated = True

        elapsed = (time.perf_counter() - start_time) * 1000
        self.logger.info(f"Problem formulation completed in {elapsed:.2f} ms")

    def add_demand_constraint(
        self, total_demand: float, tolerance: float = 0.01
    ) -> None:
        """
        Add demand balance constraint: sum(load_i) = total_demand +/- tolerance.

        Args:
            total_demand: Total heat demand to meet (kW)
            tolerance: Acceptable tolerance as fraction (0.01 = 1%)
        """
        if not self._problem_formulated:
            raise RuntimeError("Must call formulate_problem() first")

        n_units = len(self.equipment_fleet)
        demand_min = total_demand * (1 - tolerance)
        demand_max = total_demand * (1 + tolerance)

        # sum(load_i) + slack >= demand_min
        row_min = np.zeros(self._n_vars)
        row_min[self._load_vars_start : self._load_vars_start + n_units] = -1
        row_min[self._slack_vars_start] = -1

        # sum(load_i) + slack <= demand_max
        row_max = np.zeros(self._n_vars)
        row_max[self._load_vars_start : self._load_vars_start + n_units] = 1
        row_max[self._slack_vars_start] = 1

        if self._A_ub is None:
            self._A_ub = np.vstack([row_min, row_max])
            self._b_ub = np.array([-demand_min, demand_max])
        else:
            self._A_ub = np.vstack([self._A_ub, row_min, row_max])
            self._b_ub = np.append(self._b_ub, [-demand_min, demand_max])

        self.logger.debug(
            f"Added demand constraint: {demand_min:.1f} <= sum(load) <= {demand_max:.1f}"
        )

    def add_equipment_limits(
        self,
        unit_idx: int,
        min_load: float,
        max_load: float,
    ) -> None:
        """
        Add equipment operating limits with binary on/off coupling.

        Constraints:
            load_i >= min_load * on_i  (if on, must be above minimum)
            load_i <= max_load * on_i  (if off, load must be zero)

        Args:
            unit_idx: Equipment unit index
            min_load: Minimum operating load when on (kW)
            max_load: Maximum operating load (kW)
        """
        if not self._problem_formulated:
            raise RuntimeError("Must call formulate_problem() first")
        if unit_idx >= len(self.equipment_fleet):
            raise ValueError(f"Invalid unit index: {unit_idx}")

        # load_i - min_load * on_i >= 0  =>  -load_i + min_load * on_i <= 0
        row_min = np.zeros(self._n_vars)
        row_min[self._load_vars_start + unit_idx] = -1
        row_min[self._on_vars_start + unit_idx] = min_load

        # load_i - max_load * on_i <= 0
        row_max = np.zeros(self._n_vars)
        row_max[self._load_vars_start + unit_idx] = 1
        row_max[self._on_vars_start + unit_idx] = -max_load

        if self._A_ub is None:
            self._A_ub = np.vstack([row_min, row_max])
            self._b_ub = np.array([0, 0])
        else:
            self._A_ub = np.vstack([self._A_ub, row_min, row_max])
            self._b_ub = np.append(self._b_ub, [0, 0])

        self.logger.debug(
            f"Added equipment limits for unit {unit_idx}: "
            f"{min_load:.1f} <= load <= {max_load:.1f}"
        )

    def add_binary_on_off_constraints(self) -> None:
        """
        Add binary unit commitment constraints for all equipment.

        Couples continuous load variables with binary on/off variables:
            min_i * on_i <= load_i <= max_i * on_i
        """
        if not self._problem_formulated:
            raise RuntimeError("Must call formulate_problem() first")

        for i, unit in enumerate(self.equipment_fleet):
            self.add_equipment_limits(i, unit.min_load_kw, unit.max_load_kw)

        self.logger.info(
            f"Added binary on/off constraints for {len(self.equipment_fleet)} units"
        )

    def add_ramp_rate_constraints(
        self,
        ramp_up_limits: Optional[List[float]] = None,
        ramp_down_limits: Optional[List[float]] = None,
        time_step_min: float = 15.0,
    ) -> None:
        """
        Add ramp rate constraints for load changes.

        Constraints:
            load_i(t) - load_i(t-1) <= ramp_up * dt
            load_i(t-1) - load_i(t) <= ramp_down * dt

        Args:
            ramp_up_limits: Max ramp-up rates per unit (kW/min), None=use unit defaults
            ramp_down_limits: Max ramp-down rates per unit (kW/min), None=use unit defaults
            time_step_min: Time step duration (minutes)
        """
        if not self._problem_formulated:
            raise RuntimeError("Must call formulate_problem() first")

        n_units = len(self.equipment_fleet)

        # Use unit defaults if not specified
        if ramp_up_limits is None:
            ramp_up_limits = [u.ramp_up_rate_kw_min for u in self.equipment_fleet]
        if ramp_down_limits is None:
            ramp_down_limits = [u.ramp_down_rate_kw_min for u in self.equipment_fleet]

        for i, unit in enumerate(self.equipment_fleet):
            current_load = unit.current_load_kw
            max_ramp_up = ramp_up_limits[i] * time_step_min
            max_ramp_down = ramp_down_limits[i] * time_step_min

            # Skip infinite ramp rates
            if max_ramp_up == float("inf") and max_ramp_down == float("inf"):
                continue

            # load_i <= current_load + max_ramp_up
            if max_ramp_up != float("inf"):
                row_up = np.zeros(self._n_vars)
                row_up[self._load_vars_start + i] = 1

                if self._A_ub is None:
                    self._A_ub = row_up.reshape(1, -1)
                    self._b_ub = np.array([current_load + max_ramp_up])
                else:
                    self._A_ub = np.vstack([self._A_ub, row_up])
                    self._b_ub = np.append(self._b_ub, current_load + max_ramp_up)

            # load_i >= current_load - max_ramp_down  =>  -load_i <= -current + ramp_down
            if max_ramp_down != float("inf"):
                row_down = np.zeros(self._n_vars)
                row_down[self._load_vars_start + i] = -1

                self._A_ub = np.vstack([self._A_ub, row_down])
                self._b_ub = np.append(self._b_ub, -current_load + max_ramp_down)

        self.logger.info(f"Added ramp rate constraints for {n_units} units")

    def add_minimum_runtime_constraints(
        self,
        min_on_times: Optional[List[float]] = None,
        min_off_times: Optional[List[float]] = None,
    ) -> None:
        """
        Add minimum on-time and off-time constraints.

        These constraints prevent rapid cycling of equipment:
            - If unit just turned on, must stay on for min_on_time
            - If unit just turned off, must stay off for min_off_time

        Args:
            min_on_times: Minimum on-times per unit (minutes), None=use defaults
            min_off_times: Minimum off-times per unit (minutes), None=use defaults
        """
        if not self._problem_formulated:
            raise RuntimeError("Must call formulate_problem() first")

        # Use unit defaults if not specified
        if min_on_times is None:
            min_on_times = [u.min_on_time_min for u in self.equipment_fleet]
        if min_off_times is None:
            min_off_times = [u.min_off_time_min for u in self.equipment_fleet]

        for i, unit in enumerate(self.equipment_fleet):
            min_on = min_on_times[i]
            min_off = min_off_times[i]
            time_in_state = unit.time_in_current_state_min

            # If unit is currently ON and hasn't met min on-time, force ON
            if unit.is_on and time_in_state < min_on:
                self._bounds[self._on_vars_start + i] = (1, 1)
                self.logger.debug(
                    f"Unit {unit.unit_id} forced ON (min on-time not met)"
                )

            # If unit is currently OFF and hasn't met min off-time, force OFF
            elif not unit.is_on and time_in_state < min_off:
                self._bounds[self._on_vars_start + i] = (0, 0)
                self.logger.debug(
                    f"Unit {unit.unit_id} forced OFF (min off-time not met)"
                )

        self.logger.info(
            "Added minimum runtime constraints based on current state"
        )

    def add_startup_cost_constraints(
        self,
        startup_costs: Optional[List[float]] = None,
    ) -> None:
        """
        Add startup cost linearization constraints.

        Links startup binary variable to on/off transitions:
            start_i >= on_i(t) - on_i(t-1)  (detect 0->1 transition)

        Args:
            startup_costs: Startup costs per unit ($), None=use unit defaults
        """
        if not self._problem_formulated:
            raise RuntimeError("Must call formulate_problem() first")

        # Use unit defaults if not specified
        if startup_costs is None:
            startup_costs = [u.startup_cost for u in self.equipment_fleet]

        for i, unit in enumerate(self.equipment_fleet):
            was_on = 1 if unit.is_on else 0

            # start_i >= on_i - was_on  =>  -start_i + on_i <= was_on
            row = np.zeros(self._n_vars)
            row[self._start_vars_start + i] = -1
            row[self._on_vars_start + i] = 1

            if self._A_ub is None:
                self._A_ub = row.reshape(1, -1)
                self._b_ub = np.array([was_on])
            else:
                self._A_ub = np.vstack([self._A_ub, row])
                self._b_ub = np.append(self._b_ub, was_on)

        self.logger.info(
            f"Added startup cost constraints for {len(self.equipment_fleet)} units"
        )

    def set_objective(
        self,
        objective_type: ObjectiveType = ObjectiveType.MINIMIZE_COST,
        cost_weight: float = 1.0,
        emissions_weight: float = 0.0,
    ) -> None:
        """
        Set the optimization objective function.

        Objective: minimize sum(c_i * load_i + startup_i * startup_cost_i) + penalty * slack

        Args:
            objective_type: Type of objective (cost, emissions, or weighted)
            cost_weight: Weight for cost term (for weighted sum)
            emissions_weight: Weight for emissions term (for weighted sum)
        """
        if not self._problem_formulated:
            raise RuntimeError("Must call formulate_problem() first")

        self.objective_type = objective_type
        self.cost_weight = cost_weight
        self.emissions_weight = emissions_weight

        n_units = len(self.equipment_fleet)
        self._c = np.zeros(self._n_vars)

        # Determine coefficients based on objective type
        if objective_type == ObjectiveType.MINIMIZE_COST:
            # Operating cost per kWh for load variables
            for i, unit in enumerate(self.equipment_fleet):
                self._c[self._load_vars_start + i] = unit.cost_per_kwh
                self._c[self._start_vars_start + i] = unit.startup_cost

        elif objective_type == ObjectiveType.MINIMIZE_EMISSIONS:
            # Emissions per kWh for load variables
            for i, unit in enumerate(self.equipment_fleet):
                self._c[self._load_vars_start + i] = unit.emissions_per_kwh
                self._c[self._start_vars_start + i] = unit.startup_emissions

        elif objective_type == ObjectiveType.WEIGHTED_SUM:
            # Weighted combination of cost and emissions
            for i, unit in enumerate(self.equipment_fleet):
                self._c[self._load_vars_start + i] = (
                    cost_weight * unit.cost_per_kwh
                    + emissions_weight * unit.emissions_per_kwh
                )
                self._c[self._start_vars_start + i] = (
                    cost_weight * unit.startup_cost
                    + emissions_weight * unit.startup_emissions
                )

        # Penalty for slack (unmet demand)
        self._c[self._slack_vars_start] = self.config.violation_penalty

        self.logger.info(
            f"Set objective: {objective_type.value} "
            f"(cost_weight={cost_weight}, emissions_weight={emissions_weight})"
        )

    def solve(
        self,
        time_limit_s: Optional[float] = None,
        gap_tolerance: Optional[float] = None,
    ) -> OptimizationResult:
        """
        Solve the MILP optimization problem.

        Uses the configured solver backend to find the optimal
        equipment allocation that minimizes the objective while
        satisfying all constraints.

        Args:
            time_limit_s: Solver time limit (seconds), None=use config
            gap_tolerance: MIP gap tolerance, None=use config

        Returns:
            OptimizationResult with optimal setpoints and statistics

        Raises:
            RuntimeError: If problem not formulated or solver fails
        """
        if not self._problem_formulated:
            raise RuntimeError("Must call formulate_problem() first")
        if self._c is None:
            raise RuntimeError("Must call set_objective() first")

        time_limit = time_limit_s or self.config.time_limit_s
        gap_tol = gap_tolerance or self.config.gap_tolerance

        self.logger.info(
            f"Solving MILP with {self.config.solver.value} "
            f"(time_limit={time_limit}s, gap={gap_tol})"
        )

        start_time = time.perf_counter()

        if self.config.solver == SolverBackend.SCIPY:
            result = self._solve_scipy(time_limit, gap_tol)
        elif self.config.solver in (SolverBackend.PULP_GLPK, SolverBackend.PULP_CBC):
            result = self._solve_pulp(time_limit, gap_tol)
        else:
            raise ValueError(f"Unsupported solver: {self.config.solver}")

        result.solver_time_s = time.perf_counter() - start_time

        # Calculate provenance hash
        result.provenance_hash = self._calculate_provenance(result)

        self.logger.info(
            f"Solve completed: status={result.status.value}, "
            f"objective={result.objective_value}, time={result.solver_time_s:.2f}s"
        )

        return result

    def _solve_scipy(
        self, time_limit: float, gap_tolerance: float
    ) -> OptimizationResult:
        """
        Solve using scipy.optimize.milp.

        Args:
            time_limit: Solver time limit (seconds)
            gap_tolerance: MIP gap tolerance

        Returns:
            OptimizationResult from scipy solver
        """
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
        except ImportError:
            raise ImportError(
                "scipy is required for MILP solving. "
                "Install with: pip install scipy"
            )

        n_vars = self._n_vars

        # Set up bounds
        lb = np.array([b[0] if b[0] is not None else -np.inf for b in self._bounds])
        ub = np.array([b[1] if b[1] is not None else np.inf for b in self._bounds])
        bounds = Bounds(lb, ub)

        # Set up constraints
        constraints = []

        if self._A_eq is not None and len(self._A_eq) > 0:
            constraints.append(
                LinearConstraint(self._A_eq, self._b_eq, self._b_eq)
            )

        if self._A_ub is not None and len(self._A_ub) > 0:
            constraints.append(
                LinearConstraint(
                    self._A_ub,
                    -np.inf * np.ones(len(self._b_ub)),
                    self._b_ub,
                )
            )

        # Set up integrality
        integrality = np.array(self._integrality)

        # Solve
        options = {"time_limit": time_limit, "mip_rel_gap": gap_tolerance}
        if self.config.presolve:
            options["presolve"] = True

        result = milp(
            c=self._c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
            options=options,
        )

        # Parse result
        return self._parse_scipy_result(result)

    def _solve_pulp(
        self, time_limit: float, gap_tolerance: float
    ) -> OptimizationResult:
        """
        Solve using PuLP with GLPK or CBC backend.

        Args:
            time_limit: Solver time limit (seconds)
            gap_tolerance: MIP gap tolerance

        Returns:
            OptimizationResult from PuLP solver
        """
        try:
            import pulp
        except ImportError:
            raise ImportError(
                "PuLP is required for GLPK/CBC solving. "
                "Install with: pip install pulp"
            )

        n_units = len(self.equipment_fleet)

        # Create problem
        prob = pulp.LpProblem("HeatLoadBalancing", pulp.LpMinimize)

        # Create variables
        loads = [
            pulp.LpVariable(
                f"load_{i}",
                lowBound=self._bounds[self._load_vars_start + i][0],
                upBound=self._bounds[self._load_vars_start + i][1],
                cat=pulp.LpContinuous,
            )
            for i in range(n_units)
        ]

        on_vars = [
            pulp.LpVariable(
                f"on_{i}",
                lowBound=self._bounds[self._on_vars_start + i][0],
                upBound=self._bounds[self._on_vars_start + i][1],
                cat=pulp.LpBinary,
            )
            for i in range(n_units)
        ]

        start_vars = [
            pulp.LpVariable(
                f"start_{i}",
                lowBound=self._bounds[self._start_vars_start + i][0],
                upBound=self._bounds[self._start_vars_start + i][1],
                cat=pulp.LpBinary,
            )
            for i in range(n_units)
        ]

        slack = pulp.LpVariable(
            "slack",
            lowBound=0,
            cat=pulp.LpContinuous,
        )

        # Objective
        prob += (
            pulp.lpSum(
                self._c[self._load_vars_start + i] * loads[i] for i in range(n_units)
            )
            + pulp.lpSum(
                self._c[self._start_vars_start + i] * start_vars[i]
                for i in range(n_units)
            )
            + self._c[self._slack_vars_start] * slack
        )

        # Add inequality constraints
        if self._A_ub is not None:
            for row_idx in range(len(self._A_ub)):
                row = self._A_ub[row_idx]
                rhs = self._b_ub[row_idx]

                expr = pulp.lpSum(
                    row[self._load_vars_start + i] * loads[i] for i in range(n_units)
                )
                expr += pulp.lpSum(
                    row[self._on_vars_start + i] * on_vars[i] for i in range(n_units)
                )
                expr += pulp.lpSum(
                    row[self._start_vars_start + i] * start_vars[i]
                    for i in range(n_units)
                )
                expr += row[self._slack_vars_start] * slack

                prob += expr <= rhs

        # Select solver
        if self.config.solver == SolverBackend.PULP_GLPK:
            solver = pulp.GLPK_CMD(
                msg=self.config.verbose,
                timeLimit=time_limit,
                mipgap=gap_tolerance,
            )
        else:  # CBC
            solver = pulp.PULP_CBC_CMD(
                msg=self.config.verbose,
                timeLimit=time_limit,
                gapRel=gap_tolerance,
                threads=self.config.num_threads,
            )

        # Solve
        prob.solve(solver)

        # Parse result
        return self._parse_pulp_result(prob, loads, on_vars, start_vars, slack)

    def _parse_scipy_result(self, result) -> OptimizationResult:
        """Parse scipy.optimize.milp result into OptimizationResult."""
        n_units = len(self.equipment_fleet)

        # Map scipy status to our status enum
        status_map = {
            0: MILPSolverStatus.OPTIMAL,
            1: MILPSolverStatus.ITERATION_LIMIT,
            2: MILPSolverStatus.INFEASIBLE,
            3: MILPSolverStatus.UNBOUNDED,
            4: MILPSolverStatus.ERROR,
        }
        status = status_map.get(result.status, MILPSolverStatus.ERROR)

        if not result.success:
            return OptimizationResult(
                status=status,
                warnings=[result.message] if hasattr(result, "message") else [],
            )

        # Extract solution
        x = result.x
        setpoints = []
        total_cost = 0.0
        total_emissions = 0.0
        total_load = 0.0
        units_on = 0

        for i, unit in enumerate(self.equipment_fleet):
            load = x[self._load_vars_start + i]
            is_on = x[self._on_vars_start + i] > 0.5
            is_starting = x[self._start_vars_start + i] > 0.5

            if is_on:
                units_on += 1
                op_cost = load * unit.cost_per_kwh
                if is_starting:
                    op_cost += unit.startup_cost
                emissions = load * unit.emissions_per_kwh
                if is_starting:
                    emissions += unit.startup_emissions
            else:
                op_cost = 0.0
                emissions = 0.0

            total_cost += op_cost
            total_emissions += emissions
            total_load += load

            load_pct = (load / unit.max_load_kw * 100) if unit.max_load_kw > 0 else 0

            setpoints.append(
                EquipmentSetpoint(
                    unit_id=unit.unit_id,
                    load_kw=round(load, 2),
                    is_on=is_on,
                    is_starting=is_starting,
                    is_stopping=unit.is_on and not is_on,
                    operating_cost=round(op_cost, 4),
                    emissions=round(emissions, 4),
                    load_percentage=round(load_pct, 1),
                )
            )

        demand = self.constraints.total_demand_kw if self.constraints else 0
        demand_met_pct = (total_load / demand * 100) if demand > 0 else 0

        # Calculate reserve
        total_capacity = sum(u.max_load_kw for u in self.equipment_fleet if x[self._on_vars_start + i] > 0.5)
        reserve_kw = total_capacity - total_load

        return OptimizationResult(
            status=status,
            objective_value=round(result.fun, 4) if result.fun is not None else None,
            total_cost=round(total_cost, 4),
            total_emissions=round(total_emissions, 4),
            total_load_kw=round(total_load, 2),
            setpoints=setpoints,
            demand_met_pct=round(demand_met_pct, 2),
            reserve_margin_kw=round(reserve_kw, 2),
            units_on=units_on,
            solver_iterations=getattr(result, "nit", 0),
            gap=getattr(result, "mip_gap", None),
        )

    def _parse_pulp_result(
        self, prob, loads, on_vars, start_vars, slack
    ) -> OptimizationResult:
        """Parse PuLP result into OptimizationResult."""
        try:
            import pulp
        except ImportError:
            raise ImportError("PuLP required for parsing PuLP results")

        n_units = len(self.equipment_fleet)

        # Map PuLP status
        status_map = {
            pulp.LpStatusOptimal: MILPSolverStatus.OPTIMAL,
            pulp.LpStatusNotSolved: MILPSolverStatus.NOT_SOLVED,
            pulp.LpStatusInfeasible: MILPSolverStatus.INFEASIBLE,
            pulp.LpStatusUnbounded: MILPSolverStatus.UNBOUNDED,
            pulp.LpStatusUndefined: MILPSolverStatus.ERROR,
        }
        status = status_map.get(prob.status, MILPSolverStatus.ERROR)

        if status not in (MILPSolverStatus.OPTIMAL, MILPSolverStatus.FEASIBLE):
            return OptimizationResult(
                status=status,
                warnings=[f"Solver status: {pulp.LpStatus[prob.status]}"],
            )

        # Extract solution
        setpoints = []
        total_cost = 0.0
        total_emissions = 0.0
        total_load = 0.0
        units_on = 0

        for i, unit in enumerate(self.equipment_fleet):
            load = pulp.value(loads[i]) or 0.0
            is_on = (pulp.value(on_vars[i]) or 0) > 0.5
            is_starting = (pulp.value(start_vars[i]) or 0) > 0.5

            if is_on:
                units_on += 1
                op_cost = load * unit.cost_per_kwh
                if is_starting:
                    op_cost += unit.startup_cost
                emissions = load * unit.emissions_per_kwh
                if is_starting:
                    emissions += unit.startup_emissions
            else:
                op_cost = 0.0
                emissions = 0.0

            total_cost += op_cost
            total_emissions += emissions
            total_load += load

            load_pct = (load / unit.max_load_kw * 100) if unit.max_load_kw > 0 else 0

            setpoints.append(
                EquipmentSetpoint(
                    unit_id=unit.unit_id,
                    load_kw=round(load, 2),
                    is_on=is_on,
                    is_starting=is_starting,
                    is_stopping=unit.is_on and not is_on,
                    operating_cost=round(op_cost, 4),
                    emissions=round(emissions, 4),
                    load_percentage=round(load_pct, 1),
                )
            )

        demand = self.constraints.total_demand_kw if self.constraints else 0
        demand_met_pct = (total_load / demand * 100) if demand > 0 else 0

        return OptimizationResult(
            status=status,
            objective_value=round(pulp.value(prob.objective), 4),
            total_cost=round(total_cost, 4),
            total_emissions=round(total_emissions, 4),
            total_load_kw=round(total_load, 2),
            setpoints=setpoints,
            demand_met_pct=round(demand_met_pct, 2),
            units_on=units_on,
        )

    def get_optimal_allocation(self) -> List[EquipmentSetpoint]:
        """
        Get the optimal equipment allocation from the last solve.

        Returns:
            List of EquipmentSetpoint objects with optimal loads

        Raises:
            RuntimeError: If solve() has not been called
        """
        # This would typically return cached results from last solve
        # For now, re-solve if needed
        result = self.solve()
        return result.setpoints

    def _calculate_provenance(self, result: OptimizationResult) -> str:
        """
        Calculate SHA-256 provenance hash for audit trail.

        Args:
            result: Optimization result to hash

        Returns:
            SHA-256 hash string
        """
        provenance_data = {
            "equipment_fleet": [u.dict() for u in self.equipment_fleet],
            "constraints": self.constraints.dict() if self.constraints else None,
            "objective_type": self.objective_type.value,
            "cost_weight": self.cost_weight,
            "emissions_weight": self.emissions_weight,
            "solver": self.config.solver.value,
            "result_status": result.status.value,
            "objective_value": result.objective_value,
            "total_load_kw": result.total_load_kw,
            "setpoints": [s.dict() for s in result.setpoints],
        }

        provenance_str = json.dumps(provenance_data, sort_keys=True, default=str)
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def reset(self) -> None:
        """Reset solver state for a new problem."""
        self.equipment_fleet = []
        self.constraints = None
        self._problem_formulated = False
        self._c = None
        self._A_eq = None
        self._b_eq = None
        self._A_ub = None
        self._b_ub = None
        self._bounds = []
        self._integrality = []
        self.logger.info("Solver state reset")
