# -*- coding: utf-8 -*-
"""
Heuristic Fallback Methods for GL-023 HeatLoadBalancer
======================================================

This module provides fast heuristic dispatch algorithms for heat load
balancing when MILP optimization times out or for quick initialization.

Key Methods:
- Merit Order Dispatch: Priority-based loading (cheapest/cleanest first)
- Equal Percentage Loading: Distribute load proportionally across units
- Efficiency Weighted Dispatch: Load most efficient units first

These methods provide approximate solutions quickly but may not be
globally optimal. They serve as:
1. Warm-start initial solutions for MILP
2. Fallback when MILP times out
3. Quick "what-if" scenario analysis

All algorithms are DETERMINISTIC with full provenance tracking.

Example:
    >>> heuristic = HeuristicLoadBalancer(equipment_fleet, constraints)
    >>> result = heuristic.merit_order_dispatch(demand_kw=5000)
    >>> print(f"Load allocated: {result.total_load_kw} kW")

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
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, validator

from .milp_solver import (
    EquipmentUnit,
    EquipmentSetpoint,
    OptimizationConstraints,
    MILPSolverStatus,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Enumerations
# ==============================================================================


class HeuristicMethod(str, Enum):
    """Available heuristic dispatch methods."""

    MERIT_ORDER_COST = "merit_order_cost"
    MERIT_ORDER_EMISSIONS = "merit_order_emissions"
    EQUAL_PERCENTAGE = "equal_percentage"
    EFFICIENCY_WEIGHTED = "efficiency_weighted"
    MAX_EFFICIENCY_FIRST = "max_efficiency_first"
    PROPORTIONAL_CAPACITY = "proportional_capacity"
    ROUND_ROBIN = "round_robin"


class DispatchPriority(str, Enum):
    """Priority criteria for merit order dispatch."""

    LOWEST_COST = "lowest_cost"
    LOWEST_EMISSIONS = "lowest_emissions"
    HIGHEST_EFFICIENCY = "highest_efficiency"
    USER_DEFINED = "user_defined"


# ==============================================================================
# Data Models
# ==============================================================================


class MeritOrderConfig(BaseModel):
    """Configuration for merit order dispatch."""

    priority: DispatchPriority = Field(
        default=DispatchPriority.LOWEST_COST,
        description="Priority criterion for ordering",
    )
    load_at_baseload_first: bool = Field(
        default=True,
        description="Load units to baseload (max efficiency point) first",
    )
    min_units_on: int = Field(
        default=1, ge=0, description="Minimum units to keep online"
    )
    max_units_on: Optional[int] = Field(
        default=None, ge=1, description="Maximum units to run simultaneously"
    )
    reserve_margin_pct: float = Field(
        default=10.0, ge=0, description="Reserve margin requirement (%)"
    )
    avoid_partial_load: bool = Field(
        default=False,
        description="Prefer full load over partial load on units",
    )
    baseload_percentage: float = Field(
        default=70.0,
        ge=0,
        le=100,
        description="Baseload operating point as % of capacity",
    )

    class Config:
        use_enum_values = True


class HeuristicResult(BaseModel):
    """Result from heuristic dispatch."""

    method: HeuristicMethod = Field(..., description="Method used")
    setpoints: List[EquipmentSetpoint] = Field(
        default_factory=list, description="Equipment setpoints"
    )
    total_load_kw: float = Field(default=0, description="Total allocated load (kW)")
    total_cost: float = Field(default=0, description="Total operating cost ($/h)")
    total_emissions: float = Field(
        default=0, description="Total emissions (kgCO2e/h)"
    )
    demand_met_pct: float = Field(
        default=0, description="Percentage of demand met"
    )
    units_on: int = Field(default=0, description="Number of units operating")
    reserve_margin_kw: float = Field(default=0, description="Available reserve (kW)")
    computation_time_ms: float = Field(
        default=0, description="Computation time (ms)"
    )
    is_feasible: bool = Field(
        default=True, description="Whether solution satisfies basic constraints"
    )
    warnings: List[str] = Field(
        default_factory=list, description="Any warnings"
    )
    dispatch_order: List[str] = Field(
        default_factory=list, description="Order units were dispatched"
    )
    provenance_hash: str = Field(default="", description="SHA-256 audit hash")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Dispatch timestamp",
    )

    class Config:
        use_enum_values = True


# ==============================================================================
# Heuristic Load Balancer Implementation
# ==============================================================================


class HeuristicLoadBalancer:
    """
    Fast heuristic dispatch algorithms for heat load balancing.

    Provides quick approximate solutions for cases where:
    - MILP optimization times out
    - Quick scenario analysis is needed
    - Initial solution for warm-starting MILP

    All algorithms are DETERMINISTIC - same inputs produce same outputs.

    Attributes:
        equipment_fleet: List of available equipment units
        constraints: Optimization constraints
        config: Merit order configuration

    Example:
        >>> balancer = HeuristicLoadBalancer(equipment_fleet, constraints)
        >>> result = balancer.merit_order_dispatch(demand_kw=5000)
        >>> result = balancer.efficiency_weighted_dispatch(demand_kw=5000)
    """

    def __init__(
        self,
        equipment_fleet: Optional[List[EquipmentUnit]] = None,
        constraints: Optional[OptimizationConstraints] = None,
        config: Optional[MeritOrderConfig] = None,
    ):
        """
        Initialize HeuristicLoadBalancer.

        Args:
            equipment_fleet: List of equipment units
            constraints: Optimization constraints
            config: Merit order configuration
        """
        self.equipment_fleet = equipment_fleet or []
        self.constraints = constraints
        self.config = config or MeritOrderConfig()
        self.logger = logging.getLogger(f"{__name__}.HeuristicLoadBalancer")

    def merit_order_dispatch(
        self,
        demand_kw: Optional[float] = None,
        priority: Optional[DispatchPriority] = None,
    ) -> HeuristicResult:
        """
        Dispatch equipment in merit order based on priority criterion.

        Units are ordered by the priority criterion and loaded
        sequentially until demand is met. This is a greedy algorithm
        that finds a feasible solution quickly.

        Args:
            demand_kw: Heat demand to meet (kW), uses constraints if not provided
            priority: Override priority criterion

        Returns:
            HeuristicResult with dispatch solution
        """
        start_time = time.perf_counter()

        demand = demand_kw or (
            self.constraints.total_demand_kw if self.constraints else 0
        )
        if demand <= 0:
            return self._empty_result(HeuristicMethod.MERIT_ORDER_COST)

        prio = priority or self.config.priority

        self.logger.info(
            f"Merit order dispatch: demand={demand:.1f} kW, priority={prio.value}"
        )

        # Sort equipment by priority criterion
        if prio == DispatchPriority.LOWEST_COST:
            sorted_fleet = sorted(
                self.equipment_fleet, key=lambda u: u.cost_per_kwh
            )
            method = HeuristicMethod.MERIT_ORDER_COST
        elif prio == DispatchPriority.LOWEST_EMISSIONS:
            sorted_fleet = sorted(
                self.equipment_fleet, key=lambda u: u.emissions_per_kwh
            )
            method = HeuristicMethod.MERIT_ORDER_EMISSIONS
        elif prio == DispatchPriority.HIGHEST_EFFICIENCY:
            sorted_fleet = sorted(
                self.equipment_fleet, key=lambda u: -u.efficiency
            )
            method = HeuristicMethod.MAX_EFFICIENCY_FIRST
        else:
            sorted_fleet = sorted(
                self.equipment_fleet, key=lambda u: u.priority
            )
            method = HeuristicMethod.MERIT_ORDER_COST

        # Dispatch sequentially
        setpoints = []
        remaining_demand = demand
        dispatch_order = []
        total_cost = 0.0
        total_emissions = 0.0
        units_on = 0

        for unit in sorted_fleet:
            if remaining_demand <= 0:
                # Demand met, turn off remaining units
                setpoints.append(
                    EquipmentSetpoint(
                        unit_id=unit.unit_id,
                        load_kw=0,
                        is_on=False,
                    )
                )
                continue

            # Check if we should skip this unit (max units constraint)
            if (
                self.config.max_units_on is not None
                and units_on >= self.config.max_units_on
            ):
                setpoints.append(
                    EquipmentSetpoint(
                        unit_id=unit.unit_id,
                        load_kw=0,
                        is_on=False,
                    )
                )
                continue

            # Calculate load for this unit
            if self.config.avoid_partial_load and remaining_demand >= unit.max_load_kw:
                # Load to maximum
                load = unit.max_load_kw
            elif self.config.load_at_baseload_first:
                # Try baseload first
                baseload = unit.max_load_kw * (self.config.baseload_percentage / 100)
                baseload = max(unit.min_load_kw, min(baseload, unit.max_load_kw))
                if remaining_demand >= baseload:
                    load = min(remaining_demand, unit.max_load_kw)
                elif remaining_demand >= unit.min_load_kw:
                    load = remaining_demand
                else:
                    # Can't meet minimum, skip
                    setpoints.append(
                        EquipmentSetpoint(
                            unit_id=unit.unit_id,
                            load_kw=0,
                            is_on=False,
                        )
                    )
                    continue
            else:
                # Load as much as possible
                if remaining_demand >= unit.min_load_kw:
                    load = min(remaining_demand, unit.max_load_kw)
                else:
                    setpoints.append(
                        EquipmentSetpoint(
                            unit_id=unit.unit_id,
                            load_kw=0,
                            is_on=False,
                        )
                    )
                    continue

            # Add setpoint
            op_cost = load * unit.cost_per_kwh
            emissions = load * unit.emissions_per_kwh
            load_pct = (load / unit.max_load_kw * 100) if unit.max_load_kw > 0 else 0

            setpoints.append(
                EquipmentSetpoint(
                    unit_id=unit.unit_id,
                    load_kw=round(load, 2),
                    is_on=True,
                    is_starting=not unit.is_on,
                    operating_cost=round(op_cost, 4),
                    emissions=round(emissions, 4),
                    load_percentage=round(load_pct, 1),
                )
            )

            remaining_demand -= load
            total_cost += op_cost
            total_emissions += emissions
            units_on += 1
            dispatch_order.append(unit.unit_id)

        # Calculate totals
        total_load = demand - remaining_demand
        demand_met_pct = (total_load / demand * 100) if demand > 0 else 0

        # Calculate reserve
        online_capacity = sum(
            self.equipment_fleet[i].max_load_kw
            for i, sp in enumerate(setpoints)
            if sp.is_on
        )
        reserve_kw = online_capacity - total_load

        # Check feasibility
        is_feasible = True
        warnings = []

        if remaining_demand > 0:
            is_feasible = False
            warnings.append(
                f"Unable to meet full demand: shortfall {remaining_demand:.1f} kW"
            )

        if units_on < self.config.min_units_on:
            warnings.append(
                f"Fewer than minimum units online: {units_on} < {self.config.min_units_on}"
            )

        computation_time = (time.perf_counter() - start_time) * 1000

        result = HeuristicResult(
            method=method,
            setpoints=setpoints,
            total_load_kw=round(total_load, 2),
            total_cost=round(total_cost, 4),
            total_emissions=round(total_emissions, 4),
            demand_met_pct=round(demand_met_pct, 2),
            units_on=units_on,
            reserve_margin_kw=round(reserve_kw, 2),
            computation_time_ms=round(computation_time, 2),
            is_feasible=is_feasible,
            warnings=warnings,
            dispatch_order=dispatch_order,
        )

        result.provenance_hash = self._calculate_provenance(result, demand, prio.value)

        self.logger.info(
            f"Merit order dispatch complete: {units_on} units, "
            f"{total_load:.1f} kW, ${total_cost:.2f}/h"
        )

        return result

    def equal_percentage_loading(
        self,
        demand_kw: Optional[float] = None,
    ) -> HeuristicResult:
        """
        Distribute load equally as percentage of capacity across all units.

        Each unit operates at the same percentage of its maximum capacity.
        This approach balances wear across equipment but may not be
        cost or emissions optimal.

        Args:
            demand_kw: Heat demand to meet (kW)

        Returns:
            HeuristicResult with proportionally loaded solution
        """
        start_time = time.perf_counter()

        demand = demand_kw or (
            self.constraints.total_demand_kw if self.constraints else 0
        )
        if demand <= 0:
            return self._empty_result(HeuristicMethod.EQUAL_PERCENTAGE)

        self.logger.info(f"Equal percentage loading: demand={demand:.1f} kW")

        # Calculate total capacity
        total_capacity = sum(u.max_load_kw for u in self.equipment_fleet)

        if total_capacity <= 0:
            return self._empty_result(HeuristicMethod.EQUAL_PERCENTAGE)

        # Target load percentage
        target_pct = min(100, (demand / total_capacity) * 100)

        setpoints = []
        total_load = 0.0
        total_cost = 0.0
        total_emissions = 0.0
        units_on = 0
        dispatch_order = []
        warnings = []

        for unit in self.equipment_fleet:
            # Calculate target load
            target_load = unit.max_load_kw * (target_pct / 100)

            # Adjust to respect min/max limits
            if target_load < unit.min_load_kw:
                if target_pct > 5:  # Non-trivial demand
                    # Turn on at minimum load
                    load = unit.min_load_kw
                    is_on = True
                else:
                    # Very low demand, keep off
                    load = 0
                    is_on = False
            else:
                load = min(target_load, unit.max_load_kw)
                is_on = True

            if is_on:
                op_cost = load * unit.cost_per_kwh
                emissions = load * unit.emissions_per_kwh
                load_pct = (load / unit.max_load_kw * 100) if unit.max_load_kw > 0 else 0
                units_on += 1
                dispatch_order.append(unit.unit_id)
            else:
                op_cost = 0
                emissions = 0
                load_pct = 0

            setpoints.append(
                EquipmentSetpoint(
                    unit_id=unit.unit_id,
                    load_kw=round(load, 2),
                    is_on=is_on,
                    is_starting=is_on and not unit.is_on,
                    operating_cost=round(op_cost, 4),
                    emissions=round(emissions, 4),
                    load_percentage=round(load_pct, 1),
                )
            )

            total_load += load
            total_cost += op_cost
            total_emissions += emissions

        # Check if demand is met
        demand_met_pct = (total_load / demand * 100) if demand > 0 else 0
        is_feasible = total_load >= demand * 0.99  # 1% tolerance

        if not is_feasible:
            warnings.append(
                f"Equal distribution unable to meet demand: "
                f"{total_load:.1f}/{demand:.1f} kW"
            )

        # Calculate reserve
        online_capacity = sum(
            u.max_load_kw
            for u, sp in zip(self.equipment_fleet, setpoints)
            if sp.is_on
        )
        reserve_kw = online_capacity - total_load

        computation_time = (time.perf_counter() - start_time) * 1000

        result = HeuristicResult(
            method=HeuristicMethod.EQUAL_PERCENTAGE,
            setpoints=setpoints,
            total_load_kw=round(total_load, 2),
            total_cost=round(total_cost, 4),
            total_emissions=round(total_emissions, 4),
            demand_met_pct=round(demand_met_pct, 2),
            units_on=units_on,
            reserve_margin_kw=round(reserve_kw, 2),
            computation_time_ms=round(computation_time, 2),
            is_feasible=is_feasible,
            warnings=warnings,
            dispatch_order=dispatch_order,
        )

        result.provenance_hash = self._calculate_provenance(
            result, demand, "equal_percentage"
        )

        self.logger.info(
            f"Equal percentage loading complete: {target_pct:.1f}% load factor, "
            f"{units_on} units"
        )

        return result

    def efficiency_weighted_dispatch(
        self,
        demand_kw: Optional[float] = None,
    ) -> HeuristicResult:
        """
        Load most efficient units first, weighted by efficiency.

        Units are loaded proportionally to their efficiency ratings,
        with more efficient units receiving higher loads.

        Args:
            demand_kw: Heat demand to meet (kW)

        Returns:
            HeuristicResult with efficiency-weighted solution
        """
        start_time = time.perf_counter()

        demand = demand_kw or (
            self.constraints.total_demand_kw if self.constraints else 0
        )
        if demand <= 0:
            return self._empty_result(HeuristicMethod.EFFICIENCY_WEIGHTED)

        self.logger.info(f"Efficiency weighted dispatch: demand={demand:.1f} kW")

        # Calculate efficiency weights
        total_efficiency = sum(u.efficiency for u in self.equipment_fleet)
        if total_efficiency <= 0:
            # Fallback to equal percentage
            return self.equal_percentage_loading(demand)

        # Calculate weighted capacities
        weighted_capacities = []
        for unit in self.equipment_fleet:
            weight = unit.efficiency / total_efficiency
            weighted_cap = unit.max_load_kw * weight * len(self.equipment_fleet)
            weighted_capacities.append(
                (unit, weight, min(weighted_cap, unit.max_load_kw))
            )

        # Sort by efficiency (highest first)
        weighted_capacities.sort(key=lambda x: x[0].efficiency, reverse=True)

        setpoints = []
        remaining_demand = demand
        total_cost = 0.0
        total_emissions = 0.0
        units_on = 0
        dispatch_order = []
        warnings = []

        for unit, weight, target_cap in weighted_capacities:
            if remaining_demand <= 0:
                setpoints.append(
                    EquipmentSetpoint(
                        unit_id=unit.unit_id,
                        load_kw=0,
                        is_on=False,
                    )
                )
                continue

            # Calculate weighted target load
            proportion = weight * len(self.equipment_fleet)
            target_load = min(demand * proportion, target_cap, remaining_demand)

            # Adjust for min/max
            if target_load < unit.min_load_kw:
                if remaining_demand >= unit.min_load_kw:
                    load = unit.min_load_kw
                else:
                    load = 0
            else:
                load = min(target_load, unit.max_load_kw)

            if load > 0:
                op_cost = load * unit.cost_per_kwh
                emissions = load * unit.emissions_per_kwh
                load_pct = (load / unit.max_load_kw * 100) if unit.max_load_kw > 0 else 0
                units_on += 1
                dispatch_order.append(unit.unit_id)
                remaining_demand -= load
            else:
                op_cost = 0
                emissions = 0
                load_pct = 0

            setpoints.append(
                EquipmentSetpoint(
                    unit_id=unit.unit_id,
                    load_kw=round(load, 2),
                    is_on=load > 0,
                    is_starting=load > 0 and not unit.is_on,
                    operating_cost=round(op_cost, 4),
                    emissions=round(emissions, 4),
                    load_percentage=round(load_pct, 1),
                )
            )

            total_cost += op_cost
            total_emissions += emissions

        # Handle unmet demand by loading more on online units
        if remaining_demand > 0 and units_on > 0:
            # Distribute remaining to online units
            for i, sp in enumerate(setpoints):
                if sp.is_on and remaining_demand > 0:
                    unit = next(
                        u for u in self.equipment_fleet if u.unit_id == sp.unit_id
                    )
                    additional = min(
                        remaining_demand, unit.max_load_kw - sp.load_kw
                    )
                    if additional > 0:
                        new_load = sp.load_kw + additional
                        new_cost = new_load * unit.cost_per_kwh
                        new_emissions = new_load * unit.emissions_per_kwh

                        total_cost += additional * unit.cost_per_kwh
                        total_emissions += additional * unit.emissions_per_kwh

                        setpoints[i] = EquipmentSetpoint(
                            unit_id=sp.unit_id,
                            load_kw=round(new_load, 2),
                            is_on=True,
                            is_starting=sp.is_starting,
                            operating_cost=round(new_cost, 4),
                            emissions=round(new_emissions, 4),
                            load_percentage=round(
                                new_load / unit.max_load_kw * 100, 1
                            ),
                        )
                        remaining_demand -= additional

        total_load = demand - remaining_demand
        demand_met_pct = (total_load / demand * 100) if demand > 0 else 0
        is_feasible = remaining_demand <= demand * 0.01

        if not is_feasible:
            warnings.append(
                f"Unable to meet full demand: shortfall {remaining_demand:.1f} kW"
            )

        # Calculate reserve
        online_capacity = sum(
            u.max_load_kw
            for u, sp in zip(self.equipment_fleet, setpoints)
            if sp.is_on
        )
        reserve_kw = online_capacity - total_load

        computation_time = (time.perf_counter() - start_time) * 1000

        result = HeuristicResult(
            method=HeuristicMethod.EFFICIENCY_WEIGHTED,
            setpoints=setpoints,
            total_load_kw=round(total_load, 2),
            total_cost=round(total_cost, 4),
            total_emissions=round(total_emissions, 4),
            demand_met_pct=round(demand_met_pct, 2),
            units_on=units_on,
            reserve_margin_kw=round(reserve_kw, 2),
            computation_time_ms=round(computation_time, 2),
            is_feasible=is_feasible,
            warnings=warnings,
            dispatch_order=dispatch_order,
        )

        result.provenance_hash = self._calculate_provenance(
            result, demand, "efficiency_weighted"
        )

        self.logger.info(
            f"Efficiency weighted dispatch complete: {units_on} units, "
            f"{total_load:.1f} kW"
        )

        return result

    def proportional_capacity_dispatch(
        self,
        demand_kw: Optional[float] = None,
    ) -> HeuristicResult:
        """
        Distribute load proportionally to each unit's capacity.

        Larger units receive proportionally more load. This minimizes
        the number of units needed while balancing large units.

        Args:
            demand_kw: Heat demand to meet (kW)

        Returns:
            HeuristicResult with capacity-proportional solution
        """
        start_time = time.perf_counter()

        demand = demand_kw or (
            self.constraints.total_demand_kw if self.constraints else 0
        )
        if demand <= 0:
            return self._empty_result(HeuristicMethod.PROPORTIONAL_CAPACITY)

        self.logger.info(f"Proportional capacity dispatch: demand={demand:.1f} kW")

        # Sort by capacity (largest first)
        sorted_fleet = sorted(
            self.equipment_fleet, key=lambda u: -u.max_load_kw
        )

        # Calculate total capacity
        total_capacity = sum(u.max_load_kw for u in sorted_fleet)
        if total_capacity <= 0:
            return self._empty_result(HeuristicMethod.PROPORTIONAL_CAPACITY)

        setpoints = []
        remaining_demand = demand
        total_cost = 0.0
        total_emissions = 0.0
        units_on = 0
        dispatch_order = []
        warnings = []

        for unit in sorted_fleet:
            if remaining_demand <= 0:
                setpoints.append(
                    EquipmentSetpoint(
                        unit_id=unit.unit_id,
                        load_kw=0,
                        is_on=False,
                    )
                )
                continue

            # Proportional target
            proportion = unit.max_load_kw / total_capacity
            target_load = demand * proportion

            # Adjust for limits
            if target_load < unit.min_load_kw:
                if remaining_demand >= unit.min_load_kw:
                    load = unit.min_load_kw
                else:
                    load = 0
            else:
                load = min(target_load, unit.max_load_kw, remaining_demand)

            if load > 0:
                op_cost = load * unit.cost_per_kwh
                emissions = load * unit.emissions_per_kwh
                load_pct = (load / unit.max_load_kw * 100) if unit.max_load_kw > 0 else 0
                units_on += 1
                dispatch_order.append(unit.unit_id)
                remaining_demand -= load
            else:
                op_cost = 0
                emissions = 0
                load_pct = 0

            setpoints.append(
                EquipmentSetpoint(
                    unit_id=unit.unit_id,
                    load_kw=round(load, 2),
                    is_on=load > 0,
                    is_starting=load > 0 and not unit.is_on,
                    operating_cost=round(op_cost, 4),
                    emissions=round(emissions, 4),
                    load_percentage=round(load_pct, 1),
                )
            )

            total_cost += op_cost
            total_emissions += emissions

        total_load = demand - remaining_demand
        demand_met_pct = (total_load / demand * 100) if demand > 0 else 0
        is_feasible = remaining_demand <= demand * 0.01

        if not is_feasible:
            warnings.append(
                f"Unable to meet full demand: shortfall {remaining_demand:.1f} kW"
            )

        online_capacity = sum(
            u.max_load_kw
            for u, sp in zip(sorted_fleet, setpoints)
            if sp.is_on
        )
        reserve_kw = online_capacity - total_load

        computation_time = (time.perf_counter() - start_time) * 1000

        result = HeuristicResult(
            method=HeuristicMethod.PROPORTIONAL_CAPACITY,
            setpoints=setpoints,
            total_load_kw=round(total_load, 2),
            total_cost=round(total_cost, 4),
            total_emissions=round(total_emissions, 4),
            demand_met_pct=round(demand_met_pct, 2),
            units_on=units_on,
            reserve_margin_kw=round(reserve_kw, 2),
            computation_time_ms=round(computation_time, 2),
            is_feasible=is_feasible,
            warnings=warnings,
            dispatch_order=dispatch_order,
        )

        result.provenance_hash = self._calculate_provenance(
            result, demand, "proportional_capacity"
        )

        return result

    def dispatch(
        self,
        method: HeuristicMethod,
        demand_kw: Optional[float] = None,
        **kwargs,
    ) -> HeuristicResult:
        """
        Generic dispatch method that routes to specific algorithms.

        Args:
            method: Heuristic method to use
            demand_kw: Heat demand (kW)
            **kwargs: Additional method-specific arguments

        Returns:
            HeuristicResult from the selected method

        Raises:
            ValueError: If method is not supported
        """
        method_map = {
            HeuristicMethod.MERIT_ORDER_COST: lambda: self.merit_order_dispatch(
                demand_kw, DispatchPriority.LOWEST_COST
            ),
            HeuristicMethod.MERIT_ORDER_EMISSIONS: lambda: self.merit_order_dispatch(
                demand_kw, DispatchPriority.LOWEST_EMISSIONS
            ),
            HeuristicMethod.EQUAL_PERCENTAGE: lambda: self.equal_percentage_loading(
                demand_kw
            ),
            HeuristicMethod.EFFICIENCY_WEIGHTED: lambda: self.efficiency_weighted_dispatch(
                demand_kw
            ),
            HeuristicMethod.MAX_EFFICIENCY_FIRST: lambda: self.merit_order_dispatch(
                demand_kw, DispatchPriority.HIGHEST_EFFICIENCY
            ),
            HeuristicMethod.PROPORTIONAL_CAPACITY: lambda: self.proportional_capacity_dispatch(
                demand_kw
            ),
        }

        if method not in method_map:
            raise ValueError(f"Unsupported heuristic method: {method}")

        return method_map[method]()

    def compare_methods(
        self,
        demand_kw: Optional[float] = None,
        methods: Optional[List[HeuristicMethod]] = None,
    ) -> Dict[HeuristicMethod, HeuristicResult]:
        """
        Compare multiple heuristic methods for the same demand.

        Useful for quick benchmarking of different approaches.

        Args:
            demand_kw: Heat demand (kW)
            methods: Methods to compare, defaults to all available

        Returns:
            Dictionary mapping method to result
        """
        if methods is None:
            methods = [
                HeuristicMethod.MERIT_ORDER_COST,
                HeuristicMethod.MERIT_ORDER_EMISSIONS,
                HeuristicMethod.EQUAL_PERCENTAGE,
                HeuristicMethod.EFFICIENCY_WEIGHTED,
                HeuristicMethod.PROPORTIONAL_CAPACITY,
            ]

        results = {}
        for method in methods:
            try:
                results[method] = self.dispatch(method, demand_kw)
            except Exception as e:
                self.logger.error(f"Method {method.value} failed: {e}")

        return results

    def _empty_result(self, method: HeuristicMethod) -> HeuristicResult:
        """Create empty result for zero demand or error cases."""
        return HeuristicResult(
            method=method,
            setpoints=[
                EquipmentSetpoint(unit_id=u.unit_id, load_kw=0, is_on=False)
                for u in self.equipment_fleet
            ],
            total_load_kw=0,
            total_cost=0,
            total_emissions=0,
            demand_met_pct=100 if not self.equipment_fleet else 0,
            units_on=0,
            is_feasible=True,
            provenance_hash=hashlib.sha256(
                f"empty_{method.value}".encode()
            ).hexdigest(),
        )

    def _calculate_provenance(
        self,
        result: HeuristicResult,
        demand: float,
        method_detail: str,
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        # Handle both enum and string values (due to use_enum_values in Config)
        method_str = (
            result.method.value
            if hasattr(result.method, "value")
            else result.method
        )
        provenance_data = {
            "method": method_str,
            "method_detail": method_detail,
            "demand_kw": demand,
            "equipment_ids": [u.unit_id for u in self.equipment_fleet],
            "total_load_kw": result.total_load_kw,
            "total_cost": result.total_cost,
            "total_emissions": result.total_emissions,
            "units_on": result.units_on,
            "dispatch_order": result.dispatch_order,
        }
        return hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

    def update_equipment_fleet(
        self, equipment_fleet: List[EquipmentUnit]
    ) -> None:
        """Update the equipment fleet configuration."""
        self.equipment_fleet = equipment_fleet
        self.logger.info(f"Updated equipment fleet: {len(equipment_fleet)} units")

    def update_constraints(
        self, constraints: OptimizationConstraints
    ) -> None:
        """Update the optimization constraints."""
        self.constraints = constraints
        self.logger.info(
            f"Updated constraints: demand={constraints.total_demand_kw} kW"
        )

    def update_config(self, config: MeritOrderConfig) -> None:
        """Update the merit order configuration."""
        self.config = config
        self.logger.info(f"Updated config: priority={config.priority}")
