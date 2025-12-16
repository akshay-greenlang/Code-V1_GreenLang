# -*- coding: utf-8 -*-
"""
Constraint Handler for GL-023 HeatLoadBalancer
==============================================

This module provides constraint validation, feasibility checking, and
constraint relaxation for the heat load balancing MILP optimizer.

Key Features:
- Validate equipment setpoints against operational constraints
- Check ramp rate feasibility for load transitions
- Verify reserve margin and N+1 redundancy requirements
- Calculate constraint violations with severity levels
- Relax constraints intelligently when infeasible

All operations are DETERMINISTIC with full provenance tracking.

Example:
    >>> manager = ConstraintManager()
    >>> violations = manager.validate_equipment_constraints(setpoints)
    >>> if violations:
    ...     relaxed = manager.relax_constraints_if_infeasible(priority_order)

Author: GreenLang Framework Team
Agent: GL-023 HeatLoadBalancer
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, validator

from .milp_solver import EquipmentUnit, EquipmentSetpoint, OptimizationConstraints

logger = logging.getLogger(__name__)


# ==============================================================================
# Enumerations
# ==============================================================================


class ConstraintType(str, Enum):
    """Types of constraints in the optimization problem."""

    DEMAND_BALANCE = "demand_balance"
    EQUIPMENT_MIN_LOAD = "equipment_min_load"
    EQUIPMENT_MAX_LOAD = "equipment_max_load"
    RAMP_UP_RATE = "ramp_up_rate"
    RAMP_DOWN_RATE = "ramp_down_rate"
    MINIMUM_ON_TIME = "minimum_on_time"
    MINIMUM_OFF_TIME = "minimum_off_time"
    RESERVE_MARGIN = "reserve_margin"
    N_PLUS_1_REDUNDANCY = "n_plus_1_redundancy"
    MAX_UNITS_ON = "max_units_on"
    MIN_UNITS_ON = "min_units_on"
    STARTUP_LIMIT = "startup_limit"
    EMISSION_LIMIT = "emission_limit"


class ViolationSeverity(str, Enum):
    """Severity levels for constraint violations."""

    CRITICAL = "critical"  # Must be fixed, operation unsafe
    MAJOR = "major"  # Should be fixed, efficiency impact
    MINOR = "minor"  # Acceptable with warning
    WARNING = "warning"  # Informational only


class RelaxationStrategy(str, Enum):
    """Strategies for constraint relaxation."""

    PROPORTIONAL = "proportional"  # Relax by percentage
    FIXED_AMOUNT = "fixed_amount"  # Relax by fixed value
    PRIORITY_BASED = "priority_based"  # Relax lowest priority first
    ELASTIC = "elastic"  # Add elastic penalties


# ==============================================================================
# Data Models
# ==============================================================================


class ConstraintViolation(BaseModel):
    """Details of a single constraint violation."""

    constraint_type: ConstraintType = Field(
        ..., description="Type of violated constraint"
    )
    unit_id: Optional[str] = Field(
        default=None, description="Equipment unit ID if applicable"
    )
    severity: ViolationSeverity = Field(..., description="Violation severity")
    violation_amount: float = Field(..., description="Amount of violation")
    violation_percentage: float = Field(
        ..., description="Violation as percentage of limit"
    )
    limit_value: float = Field(..., description="Constraint limit value")
    actual_value: float = Field(..., description="Actual value observed")
    message: str = Field(..., description="Human-readable description")
    can_relax: bool = Field(
        default=True, description="Whether constraint can be relaxed"
    )
    relaxation_cost: float = Field(
        default=0.0, description="Cost of relaxing this constraint"
    )

    class Config:
        use_enum_values = True


class FeasibilityResult(BaseModel):
    """Result of feasibility check."""

    is_feasible: bool = Field(..., description="Overall feasibility status")
    violations: List[ConstraintViolation] = Field(
        default_factory=list, description="List of violations"
    )
    critical_count: int = Field(default=0, description="Number of critical violations")
    major_count: int = Field(default=0, description="Number of major violations")
    minor_count: int = Field(default=0, description="Number of minor violations")
    total_violation_cost: float = Field(
        default=0.0, description="Total cost of all violations"
    )
    feasibility_score: float = Field(
        default=100.0,
        ge=0,
        le=100,
        description="Feasibility score (100=fully feasible)",
    )
    provenance_hash: str = Field(default="", description="SHA-256 audit hash")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Check timestamp",
    )


class RelaxationResult(BaseModel):
    """Result of constraint relaxation."""

    success: bool = Field(..., description="Whether relaxation succeeded")
    relaxed_constraints: List[ConstraintType] = Field(
        default_factory=list, description="Constraints that were relaxed"
    )
    original_bounds: Dict[str, Tuple[float, float]] = Field(
        default_factory=dict, description="Original constraint bounds"
    )
    relaxed_bounds: Dict[str, Tuple[float, float]] = Field(
        default_factory=dict, description="Relaxed constraint bounds"
    )
    relaxation_cost: float = Field(
        default=0.0, description="Total relaxation penalty cost"
    )
    remaining_violations: List[ConstraintViolation] = Field(
        default_factory=list, description="Violations that could not be relaxed"
    )
    message: str = Field(default="", description="Summary message")
    provenance_hash: str = Field(default="", description="SHA-256 audit hash")


# ==============================================================================
# Constraint Manager Implementation
# ==============================================================================


class ConstraintManager:
    """
    Manages constraint validation, feasibility checking, and relaxation.

    This class provides comprehensive constraint handling for the MILP
    heat load balancer, including validation of equipment setpoints,
    feasibility analysis, and intelligent constraint relaxation.

    All operations are DETERMINISTIC for reproducible results.

    Attributes:
        equipment_fleet: List of equipment units
        constraints: Optimization constraints
        violation_penalties: Penalty costs for each constraint type

    Example:
        >>> manager = ConstraintManager(equipment_fleet, constraints)
        >>> result = manager.validate_equipment_constraints(setpoints)
        >>> if not result.is_feasible:
        ...     relaxed = manager.relax_constraints_if_infeasible(["reserve_margin"])
    """

    # Default penalty costs for constraint violations ($/violation)
    DEFAULT_PENALTIES = {
        ConstraintType.DEMAND_BALANCE: 1000.0,
        ConstraintType.EQUIPMENT_MIN_LOAD: 100.0,
        ConstraintType.EQUIPMENT_MAX_LOAD: 500.0,
        ConstraintType.RAMP_UP_RATE: 50.0,
        ConstraintType.RAMP_DOWN_RATE: 50.0,
        ConstraintType.MINIMUM_ON_TIME: 200.0,
        ConstraintType.MINIMUM_OFF_TIME: 200.0,
        ConstraintType.RESERVE_MARGIN: 300.0,
        ConstraintType.N_PLUS_1_REDUNDANCY: 500.0,
        ConstraintType.MAX_UNITS_ON: 100.0,
        ConstraintType.MIN_UNITS_ON: 100.0,
        ConstraintType.STARTUP_LIMIT: 150.0,
        ConstraintType.EMISSION_LIMIT: 400.0,
    }

    # Constraints that cannot be relaxed (safety critical)
    NON_RELAXABLE = {
        ConstraintType.EQUIPMENT_MAX_LOAD,  # Physical limit
    }

    def __init__(
        self,
        equipment_fleet: Optional[List[EquipmentUnit]] = None,
        constraints: Optional[OptimizationConstraints] = None,
        violation_penalties: Optional[Dict[ConstraintType, float]] = None,
    ):
        """
        Initialize ConstraintManager.

        Args:
            equipment_fleet: List of equipment units
            constraints: Optimization constraints
            violation_penalties: Custom penalty costs per constraint type
        """
        self.equipment_fleet = equipment_fleet or []
        self.constraints = constraints
        self.violation_penalties = {
            **self.DEFAULT_PENALTIES,
            **(violation_penalties or {}),
        }
        self.logger = logging.getLogger(f"{__name__}.ConstraintManager")

    def validate_equipment_constraints(
        self, setpoints: List[EquipmentSetpoint]
    ) -> FeasibilityResult:
        """
        Validate equipment setpoints against all operational constraints.

        Checks:
        - Equipment operating within min/max load bounds
        - Units are either fully on or fully off (no partial states)
        - Load percentages are consistent

        Args:
            setpoints: List of equipment setpoints to validate

        Returns:
            FeasibilityResult with any violations found
        """
        violations = []

        if not self.equipment_fleet:
            self.logger.warning("No equipment fleet configured for validation")
            return FeasibilityResult(is_feasible=True)

        # Create lookup for equipment specs
        equipment_map = {u.unit_id: u for u in self.equipment_fleet}

        for sp in setpoints:
            unit = equipment_map.get(sp.unit_id)
            if unit is None:
                self.logger.warning(f"Unknown equipment unit: {sp.unit_id}")
                continue

            # Check minimum load constraint (only if unit is on)
            if sp.is_on and sp.load_kw < unit.min_load_kw:
                violation_amt = unit.min_load_kw - sp.load_kw
                violation_pct = (
                    (violation_amt / unit.min_load_kw * 100)
                    if unit.min_load_kw > 0
                    else 0
                )
                violations.append(
                    ConstraintViolation(
                        constraint_type=ConstraintType.EQUIPMENT_MIN_LOAD,
                        unit_id=sp.unit_id,
                        severity=ViolationSeverity.MAJOR,
                        violation_amount=violation_amt,
                        violation_percentage=violation_pct,
                        limit_value=unit.min_load_kw,
                        actual_value=sp.load_kw,
                        message=(
                            f"Unit {sp.unit_id} load {sp.load_kw:.1f} kW "
                            f"below minimum {unit.min_load_kw:.1f} kW"
                        ),
                        relaxation_cost=self.violation_penalties[
                            ConstraintType.EQUIPMENT_MIN_LOAD
                        ],
                    )
                )

            # Check maximum load constraint
            if sp.load_kw > unit.max_load_kw:
                violation_amt = sp.load_kw - unit.max_load_kw
                violation_pct = (
                    (violation_amt / unit.max_load_kw * 100)
                    if unit.max_load_kw > 0
                    else 0
                )
                violations.append(
                    ConstraintViolation(
                        constraint_type=ConstraintType.EQUIPMENT_MAX_LOAD,
                        unit_id=sp.unit_id,
                        severity=ViolationSeverity.CRITICAL,
                        violation_amount=violation_amt,
                        violation_percentage=violation_pct,
                        limit_value=unit.max_load_kw,
                        actual_value=sp.load_kw,
                        message=(
                            f"Unit {sp.unit_id} load {sp.load_kw:.1f} kW "
                            f"exceeds maximum {unit.max_load_kw:.1f} kW"
                        ),
                        can_relax=False,  # Physical limit, cannot relax
                        relaxation_cost=float("inf"),
                    )
                )

            # Check unit is off if load is zero
            if sp.load_kw == 0 and sp.is_on:
                violations.append(
                    ConstraintViolation(
                        constraint_type=ConstraintType.EQUIPMENT_MIN_LOAD,
                        unit_id=sp.unit_id,
                        severity=ViolationSeverity.MINOR,
                        violation_amount=0,
                        violation_percentage=0,
                        limit_value=0,
                        actual_value=0,
                        message=(
                            f"Unit {sp.unit_id} is marked ON but has zero load"
                        ),
                        relaxation_cost=self.violation_penalties[
                            ConstraintType.EQUIPMENT_MIN_LOAD
                        ],
                    )
                )

        return self._build_feasibility_result(violations, setpoints)

    def check_ramp_feasibility(
        self,
        current_loads: Dict[str, float],
        target_loads: Dict[str, float],
        dt_min: float = 15.0,
    ) -> FeasibilityResult:
        """
        Check if load transitions are feasible within ramp rate limits.

        Args:
            current_loads: Current load for each unit {unit_id: load_kw}
            target_loads: Target load for each unit {unit_id: load_kw}
            dt_min: Time step duration (minutes)

        Returns:
            FeasibilityResult with any ramp rate violations
        """
        violations = []

        if not self.equipment_fleet:
            return FeasibilityResult(is_feasible=True)

        equipment_map = {u.unit_id: u for u in self.equipment_fleet}

        for unit_id, target_load in target_loads.items():
            unit = equipment_map.get(unit_id)
            if unit is None:
                continue

            current_load = current_loads.get(unit_id, 0.0)
            load_change = target_load - current_load

            # Check ramp-up limit
            if load_change > 0:
                max_ramp_up = unit.ramp_up_rate_kw_min * dt_min
                if max_ramp_up != float("inf") and load_change > max_ramp_up:
                    violation_amt = load_change - max_ramp_up
                    violation_pct = (
                        (violation_amt / max_ramp_up * 100) if max_ramp_up > 0 else 100
                    )
                    violations.append(
                        ConstraintViolation(
                            constraint_type=ConstraintType.RAMP_UP_RATE,
                            unit_id=unit_id,
                            severity=ViolationSeverity.MAJOR,
                            violation_amount=violation_amt,
                            violation_percentage=violation_pct,
                            limit_value=max_ramp_up,
                            actual_value=load_change,
                            message=(
                                f"Unit {unit_id} ramp-up {load_change:.1f} kW "
                                f"exceeds limit {max_ramp_up:.1f} kW/{dt_min:.0f}min"
                            ),
                            relaxation_cost=self.violation_penalties[
                                ConstraintType.RAMP_UP_RATE
                            ],
                        )
                    )

            # Check ramp-down limit
            elif load_change < 0:
                max_ramp_down = unit.ramp_down_rate_kw_min * dt_min
                abs_change = abs(load_change)
                if max_ramp_down != float("inf") and abs_change > max_ramp_down:
                    violation_amt = abs_change - max_ramp_down
                    violation_pct = (
                        (violation_amt / max_ramp_down * 100)
                        if max_ramp_down > 0
                        else 100
                    )
                    violations.append(
                        ConstraintViolation(
                            constraint_type=ConstraintType.RAMP_DOWN_RATE,
                            unit_id=unit_id,
                            severity=ViolationSeverity.MAJOR,
                            violation_amount=violation_amt,
                            violation_percentage=violation_pct,
                            limit_value=max_ramp_down,
                            actual_value=abs_change,
                            message=(
                                f"Unit {unit_id} ramp-down {abs_change:.1f} kW "
                                f"exceeds limit {max_ramp_down:.1f} kW/{dt_min:.0f}min"
                            ),
                            relaxation_cost=self.violation_penalties[
                                ConstraintType.RAMP_DOWN_RATE
                            ],
                        )
                    )

        return self._build_feasibility_result(violations, None)

    def check_reserve_margin(
        self,
        total_capacity_kw: float,
        demand_kw: float,
        reserve_pct: float = 10.0,
    ) -> FeasibilityResult:
        """
        Check if spinning reserve margin requirement is met.

        Reserve margin = (Available Capacity - Demand) / Demand * 100%

        Args:
            total_capacity_kw: Total available capacity from online units (kW)
            demand_kw: Current heat demand (kW)
            reserve_pct: Required reserve margin (%)

        Returns:
            FeasibilityResult with reserve margin status
        """
        violations = []

        if demand_kw <= 0:
            return FeasibilityResult(is_feasible=True)

        reserve_kw = total_capacity_kw - demand_kw
        actual_reserve_pct = (reserve_kw / demand_kw) * 100 if demand_kw > 0 else 0
        required_reserve_kw = demand_kw * (reserve_pct / 100)

        if actual_reserve_pct < reserve_pct:
            violation_amt = required_reserve_kw - reserve_kw
            violation_pct = (
                (reserve_pct - actual_reserve_pct) / reserve_pct * 100
                if reserve_pct > 0
                else 0
            )
            violations.append(
                ConstraintViolation(
                    constraint_type=ConstraintType.RESERVE_MARGIN,
                    unit_id=None,
                    severity=ViolationSeverity.MAJOR,
                    violation_amount=violation_amt,
                    violation_percentage=violation_pct,
                    limit_value=reserve_pct,
                    actual_value=actual_reserve_pct,
                    message=(
                        f"Reserve margin {actual_reserve_pct:.1f}% "
                        f"below required {reserve_pct:.1f}% "
                        f"(shortfall: {violation_amt:.1f} kW)"
                    ),
                    relaxation_cost=self.violation_penalties[
                        ConstraintType.RESERVE_MARGIN
                    ],
                )
            )

        return self._build_feasibility_result(violations, None)

    def check_n_plus_1_redundancy(
        self,
        equipment_status: Dict[str, bool],
        total_demand_kw: float,
    ) -> FeasibilityResult:
        """
        Check N+1 redundancy requirement.

        N+1 means the system can still meet demand if the largest
        single unit fails.

        Args:
            equipment_status: Current on/off status {unit_id: is_on}
            total_demand_kw: Total heat demand to meet (kW)

        Returns:
            FeasibilityResult with N+1 redundancy status
        """
        violations = []

        if not self.equipment_fleet:
            return FeasibilityResult(is_feasible=True)

        equipment_map = {u.unit_id: u for u in self.equipment_fleet}

        # Find online units and their capacities
        online_units = []
        for unit_id, is_on in equipment_status.items():
            if is_on and unit_id in equipment_map:
                online_units.append(equipment_map[unit_id])

        if not online_units:
            if total_demand_kw > 0:
                violations.append(
                    ConstraintViolation(
                        constraint_type=ConstraintType.N_PLUS_1_REDUNDANCY,
                        unit_id=None,
                        severity=ViolationSeverity.CRITICAL,
                        violation_amount=total_demand_kw,
                        violation_percentage=100.0,
                        limit_value=total_demand_kw,
                        actual_value=0,
                        message="No units online to meet demand",
                        can_relax=False,
                    )
                )
            return self._build_feasibility_result(violations, None)

        # Calculate total capacity and largest unit
        total_capacity = sum(u.max_load_kw for u in online_units)
        largest_unit = max(online_units, key=lambda u: u.max_load_kw)
        largest_capacity = largest_unit.max_load_kw

        # N+1 capacity = total minus largest
        n_plus_1_capacity = total_capacity - largest_capacity

        if n_plus_1_capacity < total_demand_kw:
            shortfall = total_demand_kw - n_plus_1_capacity
            violation_pct = (
                (shortfall / total_demand_kw * 100) if total_demand_kw > 0 else 0
            )
            violations.append(
                ConstraintViolation(
                    constraint_type=ConstraintType.N_PLUS_1_REDUNDANCY,
                    unit_id=largest_unit.unit_id,
                    severity=ViolationSeverity.MAJOR,
                    violation_amount=shortfall,
                    violation_percentage=violation_pct,
                    limit_value=total_demand_kw,
                    actual_value=n_plus_1_capacity,
                    message=(
                        f"N+1 redundancy not met: losing largest unit "
                        f"({largest_unit.unit_id}, {largest_capacity:.1f} kW) "
                        f"leaves only {n_plus_1_capacity:.1f} kW capacity "
                        f"vs {total_demand_kw:.1f} kW demand"
                    ),
                    relaxation_cost=self.violation_penalties[
                        ConstraintType.N_PLUS_1_REDUNDANCY
                    ],
                )
            )

        return self._build_feasibility_result(violations, None)

    def calculate_constraint_violations(
        self,
        setpoints: List[EquipmentSetpoint],
        demand_kw: Optional[float] = None,
    ) -> FeasibilityResult:
        """
        Calculate all constraint violations for a given solution.

        Performs comprehensive validation including:
        - Equipment operating limits
        - Demand balance
        - Reserve margin (if constraints specified)
        - N+1 redundancy (if enabled)

        Args:
            setpoints: Equipment setpoints to validate
            demand_kw: Heat demand (kW), uses constraints if not specified

        Returns:
            FeasibilityResult with complete violation analysis
        """
        all_violations = []

        # Equipment constraints
        equipment_result = self.validate_equipment_constraints(setpoints)
        all_violations.extend(equipment_result.violations)

        # Demand balance
        demand = demand_kw or (
            self.constraints.total_demand_kw if self.constraints else None
        )
        if demand is not None and demand > 0:
            total_load = sum(sp.load_kw for sp in setpoints if sp.is_on)
            tolerance = (
                self.constraints.demand_tolerance_pct / 100
                if self.constraints
                else 0.01
            )
            demand_min = demand * (1 - tolerance)
            demand_max = demand * (1 + tolerance)

            if total_load < demand_min:
                shortfall = demand_min - total_load
                all_violations.append(
                    ConstraintViolation(
                        constraint_type=ConstraintType.DEMAND_BALANCE,
                        unit_id=None,
                        severity=ViolationSeverity.CRITICAL,
                        violation_amount=shortfall,
                        violation_percentage=(shortfall / demand * 100),
                        limit_value=demand_min,
                        actual_value=total_load,
                        message=(
                            f"Total load {total_load:.1f} kW below "
                            f"minimum demand {demand_min:.1f} kW"
                        ),
                        relaxation_cost=self.violation_penalties[
                            ConstraintType.DEMAND_BALANCE
                        ],
                    )
                )
            elif total_load > demand_max:
                excess = total_load - demand_max
                all_violations.append(
                    ConstraintViolation(
                        constraint_type=ConstraintType.DEMAND_BALANCE,
                        unit_id=None,
                        severity=ViolationSeverity.MINOR,
                        violation_amount=excess,
                        violation_percentage=(excess / demand * 100),
                        limit_value=demand_max,
                        actual_value=total_load,
                        message=(
                            f"Total load {total_load:.1f} kW exceeds "
                            f"maximum demand {demand_max:.1f} kW"
                        ),
                        relaxation_cost=self.violation_penalties[
                            ConstraintType.DEMAND_BALANCE
                        ]
                        * 0.5,  # Lower penalty for over-production
                    )
                )

        # Reserve margin
        if self.constraints and self.equipment_fleet:
            equipment_map = {u.unit_id: u for u in self.equipment_fleet}
            online_capacity = sum(
                equipment_map[sp.unit_id].max_load_kw
                for sp in setpoints
                if sp.is_on and sp.unit_id in equipment_map
            )
            reserve_result = self.check_reserve_margin(
                online_capacity,
                demand or 0,
                self.constraints.reserve_margin_pct,
            )
            all_violations.extend(reserve_result.violations)

        # N+1 redundancy
        if self.constraints and self.constraints.enforce_n_plus_1:
            status = {sp.unit_id: sp.is_on for sp in setpoints}
            n1_result = self.check_n_plus_1_redundancy(status, demand or 0)
            all_violations.extend(n1_result.violations)

        return self._build_feasibility_result(all_violations, setpoints)

    def relax_constraints_if_infeasible(
        self,
        priority_order: Optional[List[ConstraintType]] = None,
        max_relaxation_pct: float = 20.0,
        strategy: RelaxationStrategy = RelaxationStrategy.PRIORITY_BASED,
    ) -> RelaxationResult:
        """
        Relax constraints in priority order to achieve feasibility.

        Constraints are relaxed from lowest to highest priority until
        the problem becomes feasible or all relaxable constraints
        have been modified.

        Args:
            priority_order: Order to relax constraints (first = relax first).
                           Default: [reserve_margin, ramp_rates, min_runtime, demand]
            max_relaxation_pct: Maximum relaxation as percentage of original
            strategy: Relaxation strategy to use

        Returns:
            RelaxationResult with relaxed bounds and costs
        """
        if priority_order is None:
            # Default priority: relax "softer" constraints first
            priority_order = [
                ConstraintType.RESERVE_MARGIN,
                ConstraintType.RAMP_UP_RATE,
                ConstraintType.RAMP_DOWN_RATE,
                ConstraintType.MINIMUM_ON_TIME,
                ConstraintType.MINIMUM_OFF_TIME,
                ConstraintType.N_PLUS_1_REDUNDANCY,
                ConstraintType.DEMAND_BALANCE,
            ]

        relaxed_constraints = []
        original_bounds: Dict[str, Tuple[float, float]] = {}
        relaxed_bounds: Dict[str, Tuple[float, float]] = {}
        total_cost = 0.0
        remaining_violations = []

        # Get current constraint values
        if self.constraints:
            original_bounds["reserve_margin_pct"] = (
                self.constraints.reserve_margin_pct,
                self.constraints.reserve_margin_pct,
            )
            original_bounds["demand_tolerance_pct"] = (
                self.constraints.demand_tolerance_pct,
                self.constraints.demand_tolerance_pct,
            )

        for constraint_type in priority_order:
            # Skip non-relaxable constraints
            if constraint_type in self.NON_RELAXABLE:
                self.logger.warning(
                    f"Skipping non-relaxable constraint: {constraint_type.value}"
                )
                continue

            # Calculate relaxation based on strategy
            if strategy == RelaxationStrategy.PROPORTIONAL:
                relaxation_factor = 1 + (max_relaxation_pct / 100)
            elif strategy == RelaxationStrategy.FIXED_AMOUNT:
                relaxation_factor = 1.0  # Will add fixed amount instead
            else:  # PRIORITY_BASED or ELASTIC
                relaxation_factor = 1 + (max_relaxation_pct / 100)

            # Apply relaxation based on constraint type
            if constraint_type == ConstraintType.RESERVE_MARGIN and self.constraints:
                new_reserve = max(
                    0, self.constraints.reserve_margin_pct / relaxation_factor
                )
                relaxed_bounds["reserve_margin_pct"] = (new_reserve, new_reserve)
                relaxed_constraints.append(constraint_type)
                total_cost += self.violation_penalties[constraint_type] * (
                    self.constraints.reserve_margin_pct - new_reserve
                )

            elif constraint_type == ConstraintType.DEMAND_BALANCE and self.constraints:
                new_tolerance = min(
                    10.0, self.constraints.demand_tolerance_pct * relaxation_factor
                )
                relaxed_bounds["demand_tolerance_pct"] = (new_tolerance, new_tolerance)
                relaxed_constraints.append(constraint_type)
                total_cost += self.violation_penalties[constraint_type] * (
                    new_tolerance - self.constraints.demand_tolerance_pct
                )

            elif constraint_type in (
                ConstraintType.RAMP_UP_RATE,
                ConstraintType.RAMP_DOWN_RATE,
            ):
                # Mark for equipment-level relaxation
                relaxed_constraints.append(constraint_type)
                total_cost += self.violation_penalties[constraint_type]

            elif constraint_type in (
                ConstraintType.MINIMUM_ON_TIME,
                ConstraintType.MINIMUM_OFF_TIME,
            ):
                # Mark for equipment-level relaxation
                relaxed_constraints.append(constraint_type)
                total_cost += self.violation_penalties[constraint_type]

            elif constraint_type == ConstraintType.N_PLUS_1_REDUNDANCY:
                if self.constraints:
                    relaxed_bounds["enforce_n_plus_1"] = (False, False)
                    relaxed_constraints.append(constraint_type)
                    total_cost += self.violation_penalties[constraint_type]

        # Calculate provenance
        provenance_data = {
            "priority_order": [c.value for c in priority_order],
            "max_relaxation_pct": max_relaxation_pct,
            "strategy": strategy.value,
            "relaxed_constraints": [c.value for c in relaxed_constraints],
            "original_bounds": original_bounds,
            "relaxed_bounds": relaxed_bounds,
            "total_cost": total_cost,
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        return RelaxationResult(
            success=len(relaxed_constraints) > 0,
            relaxed_constraints=relaxed_constraints,
            original_bounds=original_bounds,
            relaxed_bounds=relaxed_bounds,
            relaxation_cost=total_cost,
            remaining_violations=remaining_violations,
            message=(
                f"Relaxed {len(relaxed_constraints)} constraints "
                f"with total penalty cost ${total_cost:.2f}"
            ),
            provenance_hash=provenance_hash,
        )

    def _build_feasibility_result(
        self,
        violations: List[ConstraintViolation],
        setpoints: Optional[List[EquipmentSetpoint]],
    ) -> FeasibilityResult:
        """Build FeasibilityResult from violations list."""
        critical_count = sum(
            1 for v in violations if v.severity == ViolationSeverity.CRITICAL
        )
        major_count = sum(
            1 for v in violations if v.severity == ViolationSeverity.MAJOR
        )
        minor_count = sum(
            1 for v in violations if v.severity == ViolationSeverity.MINOR
        )

        # Calculate total violation cost
        total_cost = sum(v.relaxation_cost for v in violations if v.can_relax)

        # Calculate feasibility score (100 = perfect, 0 = many violations)
        if not violations:
            score = 100.0
        else:
            # Deduct points based on severity
            score = 100.0
            score -= critical_count * 30
            score -= major_count * 15
            score -= minor_count * 5
            score = max(0, score)

        # Calculate provenance hash
        provenance_data = {
            "violations": [v.dict() for v in violations],
            "setpoints": [s.dict() for s in setpoints] if setpoints else None,
            "constraints": self.constraints.dict() if self.constraints else None,
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True, default=str).encode()
        ).hexdigest()

        return FeasibilityResult(
            is_feasible=(critical_count == 0 and major_count == 0),
            violations=violations,
            critical_count=critical_count,
            major_count=major_count,
            minor_count=minor_count,
            total_violation_cost=total_cost,
            feasibility_score=score,
            provenance_hash=provenance_hash,
        )

    def update_equipment_fleet(self, equipment_fleet: List[EquipmentUnit]) -> None:
        """Update the equipment fleet configuration."""
        self.equipment_fleet = equipment_fleet
        self.logger.info(f"Updated equipment fleet: {len(equipment_fleet)} units")

    def update_constraints(self, constraints: OptimizationConstraints) -> None:
        """Update the optimization constraints."""
        self.constraints = constraints
        self.logger.info(f"Updated constraints: demand={constraints.total_demand_kw} kW")
