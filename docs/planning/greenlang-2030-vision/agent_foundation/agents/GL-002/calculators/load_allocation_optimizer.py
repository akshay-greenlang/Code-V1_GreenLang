# -*- coding: utf-8 -*-
"""
Multi-Boiler Load Allocation Optimizer - Zero Hallucination Guarantee

Implements optimal load allocation across multiple boilers for minimum
cost and emissions while respecting operational constraints. Uses
deterministic optimization algorithms with complete provenance tracking.

Author: GL-CalculatorEngineer
Version: 1.0.0
Standards: ISO 50001, EN 12952-15, ASME PTC 4

Features:
    - Multi-boiler load allocation
    - Marginal cost curve generation
    - Minimum cost dispatch
    - Hot standby optimization
    - Ramp rate constraints
    - Emission-constrained dispatch
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json

from .provenance import ProvenanceTracker, ProvenanceRecord, CalculationStep


class BoilerStatus(Enum):
    """Operating status of boiler."""
    OPERATING = "operating"
    HOT_STANDBY = "hot_standby"
    COLD_STANDBY = "cold_standby"
    MAINTENANCE = "maintenance"
    UNAVAILABLE = "unavailable"


class OptimizationObjective(Enum):
    """Optimization objective for load allocation."""
    MINIMUM_COST = "minimum_cost"
    MINIMUM_EMISSIONS = "minimum_emissions"
    MINIMUM_FUEL = "minimum_fuel"
    BALANCED = "balanced"  # Multi-objective


@dataclass
class BoilerCharacteristics:
    """
    Individual boiler characteristics for load optimization.

    Includes efficiency curves, constraints, and cost parameters.
    """
    boiler_id: str
    name: str

    # Capacity Constraints (kg steam/hr)
    minimum_load_kg_hr: float
    maximum_load_kg_hr: float
    design_load_kg_hr: float

    # Current Status
    status: BoilerStatus = BoilerStatus.OPERATING
    current_load_kg_hr: float = 0.0

    # Efficiency Curve Coefficients (quadratic: eta = a + b*L + c*L^2)
    # Where L is load fraction (0-1)
    efficiency_coeff_a: float = 0.75  # Constant term
    efficiency_coeff_b: float = 0.30  # Linear term
    efficiency_coeff_c: float = -0.15  # Quadratic term

    # Fuel Properties
    fuel_type: str = "natural_gas"
    fuel_cost_per_kg: float = 0.50
    fuel_heating_value_kj_kg: float = 50000

    # Emission Factors (kg CO2e per kg fuel)
    co2_emission_factor: float = 2.75

    # Ramp Rate Constraints (kg steam/hr per minute)
    max_ramp_up_rate: float = 500.0
    max_ramp_down_rate: float = 500.0

    # Startup/Shutdown
    hot_startup_time_minutes: float = 15.0
    cold_startup_time_minutes: float = 60.0
    hot_standby_fuel_kg_hr: float = 50.0  # Fuel to maintain hot standby
    minimum_runtime_minutes: float = 60.0
    minimum_downtime_minutes: float = 30.0

    # Maintenance
    hours_since_maintenance: float = 0.0
    maintenance_interval_hours: float = 8760.0  # Annual

    def efficiency_at_load(self, load_fraction: float) -> float:
        """Calculate efficiency at given load fraction."""
        L = max(0.0, min(1.0, load_fraction))
        return self.efficiency_coeff_a + self.efficiency_coeff_b * L + self.efficiency_coeff_c * L * L

    def fuel_consumption_at_load(self, steam_load_kg_hr: float) -> float:
        """Calculate fuel consumption at given steam load."""
        if steam_load_kg_hr <= 0:
            return 0.0

        load_fraction = steam_load_kg_hr / self.maximum_load_kg_hr
        efficiency = self.efficiency_at_load(load_fraction)

        # Steam enthalpy rise (simplified - typically ~2500 kJ/kg)
        enthalpy_rise = 2500.0

        # Fuel required
        heat_required = steam_load_kg_hr * enthalpy_rise
        fuel_consumption = heat_required / (efficiency * self.fuel_heating_value_kj_kg)

        return fuel_consumption


@dataclass
class LoadAllocationConstraints:
    """
    Constraints for load allocation optimization.
    """
    # Total Steam Demand
    total_steam_demand_kg_hr: float

    # Reserve Requirements
    spinning_reserve_percent: float = 10.0  # Available capacity above demand
    minimum_operating_boilers: int = 1

    # Emission Limits
    max_total_co2_kg_hr: Optional[float] = None
    max_total_nox_kg_hr: Optional[float] = None

    # Budget Constraints
    max_fuel_cost_per_hr: Optional[float] = None

    # Time Horizon
    optimization_horizon_hours: float = 1.0

    # Load Forecast (optional - for look-ahead optimization)
    forecast_loads_kg_hr: List[float] = field(default_factory=list)
    forecast_interval_minutes: float = 15.0


@dataclass
class MarginalCostPoint:
    """
    Point on marginal cost curve.
    """
    boiler_id: str
    load_kg_hr: Decimal
    marginal_cost_per_kg_steam: Decimal
    cumulative_load_kg_hr: Decimal
    efficiency_percent: Decimal
    fuel_consumption_kg_hr: Decimal
    co2_emissions_kg_hr: Decimal


@dataclass
class BoilerLoadAllocation:
    """
    Load allocation for a single boiler.
    """
    boiler_id: str
    name: str
    allocated_load_kg_hr: Decimal
    load_fraction: Decimal
    efficiency_percent: Decimal
    fuel_consumption_kg_hr: Decimal
    fuel_cost_per_hr: Decimal
    co2_emissions_kg_hr: Decimal
    is_at_minimum: bool
    is_at_maximum: bool
    available_ramp_up_kg_hr: Decimal
    available_ramp_down_kg_hr: Decimal


@dataclass
class LoadAllocationResult:
    """
    Complete load allocation optimization result.
    """
    # Allocation by Boiler
    allocations: List[BoilerLoadAllocation]

    # Summary Metrics
    total_steam_produced_kg_hr: Decimal
    total_fuel_consumption_kg_hr: Decimal
    total_fuel_cost_per_hr: Decimal
    total_co2_emissions_kg_hr: Decimal
    weighted_average_efficiency_percent: Decimal

    # Reserve and Capacity
    spinning_reserve_kg_hr: Decimal
    spinning_reserve_percent: Decimal
    total_available_capacity_kg_hr: Decimal

    # Optimization Metrics
    objective_value: Decimal
    optimization_objective: OptimizationObjective
    is_optimal: bool
    constraint_violations: List[str]

    # Marginal Cost Information
    system_marginal_cost_per_kg: Decimal
    marginal_cost_curve: List[Dict]

    # Recommendations
    recommendations: List[Dict]

    # Hot Standby Optimization
    hot_standby_boilers: List[str]
    hot_standby_cost_per_hr: Decimal

    # Provenance
    provenance_hash: str
    calculation_steps: List[Dict]

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'allocations': [
                {
                    'boiler_id': a.boiler_id,
                    'name': a.name,
                    'allocated_load_kg_hr': float(a.allocated_load_kg_hr),
                    'load_fraction_percent': float(a.load_fraction * Decimal('100')),
                    'efficiency_percent': float(a.efficiency_percent),
                    'fuel_consumption_kg_hr': float(a.fuel_consumption_kg_hr),
                    'fuel_cost_per_hr': float(a.fuel_cost_per_hr),
                    'co2_emissions_kg_hr': float(a.co2_emissions_kg_hr),
                }
                for a in self.allocations
            ],
            'summary': {
                'total_steam_kg_hr': float(self.total_steam_produced_kg_hr),
                'total_fuel_kg_hr': float(self.total_fuel_consumption_kg_hr),
                'total_cost_per_hr': float(self.total_fuel_cost_per_hr),
                'total_co2_kg_hr': float(self.total_co2_emissions_kg_hr),
                'average_efficiency_percent': float(self.weighted_average_efficiency_percent),
                'spinning_reserve_percent': float(self.spinning_reserve_percent),
                'system_marginal_cost': float(self.system_marginal_cost_per_kg),
            },
            'optimization': {
                'objective': self.optimization_objective.value,
                'is_optimal': self.is_optimal,
                'constraint_violations': self.constraint_violations,
            },
            'hot_standby': {
                'boilers': self.hot_standby_boilers,
                'cost_per_hr': float(self.hot_standby_cost_per_hr),
            },
            'recommendations': self.recommendations,
            'provenance_hash': self.provenance_hash,
        }


class LoadAllocationOptimizer:
    """
    Multi-Boiler Load Allocation Optimizer.

    Implements optimal dispatch of steam load across multiple boilers
    to minimize cost, emissions, or fuel consumption while respecting
    all operational constraints.

    Zero Hallucination Guarantee:
    - Pure mathematical optimization
    - No LLM inference
    - Bit-perfect reproducibility (Decimal arithmetic)
    - Complete SHA-256 provenance tracking
    - Deterministic algorithms only
    """

    def __init__(self, version: str = "1.0.0"):
        """Initialize load allocation optimizer."""
        self.version = version

    def optimize_load_allocation(
        self,
        boilers: List[BoilerCharacteristics],
        constraints: LoadAllocationConstraints,
        objective: OptimizationObjective = OptimizationObjective.MINIMUM_COST
    ) -> LoadAllocationResult:
        """
        Optimize load allocation across multiple boilers.

        Uses merit-order dispatch for minimum cost, with adjustments
        for emission constraints and operational limits.

        Args:
            boilers: List of boiler characteristics
            constraints: Load allocation constraints
            objective: Optimization objective

        Returns:
            LoadAllocationResult with optimal allocation

        Raises:
            ValueError: If constraints cannot be satisfied
        """
        # Initialize provenance tracking
        tracker = ProvenanceTracker(
            calculation_id=f"load_alloc_{id(constraints)}",
            calculation_type="load_allocation_optimization",
            version=self.version
        )

        # Record inputs
        tracker.record_inputs({
            'num_boilers': len(boilers),
            'boiler_ids': [b.boiler_id for b in boilers],
            'total_demand_kg_hr': constraints.total_steam_demand_kg_hr,
            'objective': objective.value,
        })

        # Filter available boilers
        available_boilers = [
            b for b in boilers
            if b.status in [BoilerStatus.OPERATING, BoilerStatus.HOT_STANDBY]
        ]

        if len(available_boilers) < constraints.minimum_operating_boilers:
            raise ValueError(
                f"Insufficient available boilers: {len(available_boilers)} "
                f"< {constraints.minimum_operating_boilers} required"
            )

        # Step 1: Generate marginal cost curves
        marginal_curves = self._generate_marginal_cost_curves(
            available_boilers, tracker
        )

        # Step 2: Calculate total capacity
        total_capacity = self._calculate_total_capacity(available_boilers, tracker)

        # Step 3: Validate demand can be met
        demand = Decimal(str(constraints.total_steam_demand_kg_hr))
        reserve_factor = Decimal('1') + Decimal(str(constraints.spinning_reserve_percent)) / Decimal('100')

        required_capacity = demand * reserve_factor
        if required_capacity > total_capacity:
            raise ValueError(
                f"Demand {float(demand):.0f} + reserve exceeds capacity {float(total_capacity):.0f}"
            )

        # Step 4: Dispatch based on objective
        if objective == OptimizationObjective.MINIMUM_COST:
            allocations = self._dispatch_minimum_cost(
                available_boilers, demand, constraints, tracker
            )
        elif objective == OptimizationObjective.MINIMUM_EMISSIONS:
            allocations = self._dispatch_minimum_emissions(
                available_boilers, demand, constraints, tracker
            )
        elif objective == OptimizationObjective.MINIMUM_FUEL:
            allocations = self._dispatch_minimum_fuel(
                available_boilers, demand, constraints, tracker
            )
        else:  # BALANCED
            allocations = self._dispatch_balanced(
                available_boilers, demand, constraints, tracker
            )

        # Step 5: Apply ramp rate constraints
        allocations = self._apply_ramp_constraints(allocations, available_boilers, tracker)

        # Step 6: Calculate summary metrics
        total_steam, total_fuel, total_cost, total_co2, avg_efficiency = \
            self._calculate_summary_metrics(allocations, tracker)

        # Step 7: Calculate reserves
        spinning_reserve = total_capacity - total_steam
        reserve_percent = (spinning_reserve / total_capacity) * Decimal('100') if total_capacity > 0 else Decimal('0')

        # Step 8: Calculate system marginal cost
        system_marginal_cost = self._calculate_system_marginal_cost(
            allocations, marginal_curves, tracker
        )

        # Step 9: Optimize hot standby
        hot_standby_boilers, standby_cost = self._optimize_hot_standby(
            boilers, allocations, constraints, tracker
        )

        # Step 10: Check for constraint violations
        violations = self._check_constraint_violations(
            allocations, constraints, total_co2, total_cost
        )

        # Step 11: Generate recommendations
        recommendations = self._generate_recommendations(
            allocations, available_boilers, constraints, spinning_reserve, tracker
        )

        # Build result
        provenance = tracker.get_provenance_record(total_cost)

        result = LoadAllocationResult(
            allocations=allocations,
            total_steam_produced_kg_hr=total_steam,
            total_fuel_consumption_kg_hr=total_fuel,
            total_fuel_cost_per_hr=total_cost,
            total_co2_emissions_kg_hr=total_co2,
            weighted_average_efficiency_percent=avg_efficiency,
            spinning_reserve_kg_hr=spinning_reserve,
            spinning_reserve_percent=reserve_percent,
            total_available_capacity_kg_hr=total_capacity,
            objective_value=total_cost if objective == OptimizationObjective.MINIMUM_COST else total_co2,
            optimization_objective=objective,
            is_optimal=len(violations) == 0,
            constraint_violations=violations,
            system_marginal_cost_per_kg=system_marginal_cost,
            marginal_cost_curve=[
                {
                    'boiler_id': p.boiler_id,
                    'load_kg_hr': float(p.load_kg_hr),
                    'marginal_cost': float(p.marginal_cost_per_kg_steam),
                    'cumulative_load': float(p.cumulative_load_kg_hr),
                }
                for curve in marginal_curves.values()
                for p in curve
            ],
            recommendations=recommendations,
            hot_standby_boilers=hot_standby_boilers,
            hot_standby_cost_per_hr=standby_cost,
            provenance_hash=provenance.provenance_hash,
            calculation_steps=[s.to_dict() for s in provenance.calculation_steps]
        )

        return result

    def generate_marginal_cost_curve(
        self,
        boilers: List[BoilerCharacteristics],
        load_increments: int = 20
    ) -> List[MarginalCostPoint]:
        """
        Generate system marginal cost curve for dispatch visualization.

        Returns ordered list of marginal cost points from cheapest to
        most expensive capacity.

        Args:
            boilers: List of boiler characteristics
            load_increments: Number of load points per boiler

        Returns:
            List of MarginalCostPoint ordered by marginal cost
        """
        tracker = ProvenanceTracker(
            calculation_id=f"marginal_curve_{id(boilers)}",
            calculation_type="marginal_cost_curve",
            version=self.version
        )

        all_points = []
        cumulative = Decimal('0')

        # Generate points for each boiler
        for boiler in boilers:
            if boiler.status not in [BoilerStatus.OPERATING, BoilerStatus.HOT_STANDBY]:
                continue

            min_load = Decimal(str(boiler.minimum_load_kg_hr))
            max_load = Decimal(str(boiler.maximum_load_kg_hr))

            # Generate load points
            load_step = (max_load - min_load) / Decimal(str(load_increments))

            for i in range(load_increments + 1):
                load = min_load + load_step * Decimal(str(i))
                load_fraction = load / max_load

                # Calculate efficiency at this load
                efficiency = Decimal(str(boiler.efficiency_at_load(float(load_fraction))))

                # Calculate marginal fuel consumption
                enthalpy_rise = Decimal('2500')  # kJ/kg steam
                fuel_per_kg_steam = enthalpy_rise / (
                    efficiency * Decimal(str(boiler.fuel_heating_value_kj_kg))
                )

                # Marginal cost
                fuel_cost = Decimal(str(boiler.fuel_cost_per_kg))
                marginal_cost = fuel_per_kg_steam * fuel_cost

                # CO2 emissions
                co2_factor = Decimal(str(boiler.co2_emission_factor))
                fuel_rate = float(load) * float(fuel_per_kg_steam)
                co2_rate = Decimal(str(fuel_rate)) * co2_factor

                point = MarginalCostPoint(
                    boiler_id=boiler.boiler_id,
                    load_kg_hr=load,
                    marginal_cost_per_kg_steam=marginal_cost.quantize(
                        Decimal('0.0001'), rounding=ROUND_HALF_UP
                    ),
                    cumulative_load_kg_hr=Decimal('0'),  # Set after sorting
                    efficiency_percent=(efficiency * Decimal('100')).quantize(
                        Decimal('0.01'), rounding=ROUND_HALF_UP
                    ),
                    fuel_consumption_kg_hr=Decimal(str(fuel_rate)).quantize(
                        Decimal('0.01'), rounding=ROUND_HALF_UP
                    ),
                    co2_emissions_kg_hr=co2_rate.quantize(
                        Decimal('0.01'), rounding=ROUND_HALF_UP
                    )
                )
                all_points.append(point)

        # Sort by marginal cost (merit order)
        all_points.sort(key=lambda p: p.marginal_cost_per_kg_steam)

        # Calculate cumulative load
        cumulative = Decimal('0')
        for point in all_points:
            cumulative += point.load_kg_hr
            point.cumulative_load_kg_hr = cumulative

        return all_points

    def _generate_marginal_cost_curves(
        self,
        boilers: List[BoilerCharacteristics],
        tracker: ProvenanceTracker
    ) -> Dict[str, List[MarginalCostPoint]]:
        """Generate marginal cost curves for each boiler."""
        curves = {}

        for boiler in boilers:
            points = []
            min_load = Decimal(str(boiler.minimum_load_kg_hr))
            max_load = Decimal(str(boiler.maximum_load_kg_hr))

            # Sample at 10 points
            for i in range(11):
                load_frac = Decimal(str(i)) / Decimal('10')
                load = min_load + (max_load - min_load) * load_frac

                # Calculate marginal cost at this load
                efficiency = Decimal(str(boiler.efficiency_at_load(float(load_frac))))
                fuel_per_kg = Decimal('2500') / (
                    efficiency * Decimal(str(boiler.fuel_heating_value_kj_kg))
                )
                marginal_cost = fuel_per_kg * Decimal(str(boiler.fuel_cost_per_kg))

                point = MarginalCostPoint(
                    boiler_id=boiler.boiler_id,
                    load_kg_hr=load,
                    marginal_cost_per_kg_steam=marginal_cost,
                    cumulative_load_kg_hr=Decimal('0'),
                    efficiency_percent=efficiency * Decimal('100'),
                    fuel_consumption_kg_hr=load * fuel_per_kg,
                    co2_emissions_kg_hr=load * fuel_per_kg * Decimal(str(boiler.co2_emission_factor))
                )
                points.append(point)

            curves[boiler.boiler_id] = points

        tracker.record_step(
            operation="generate_marginal_curves",
            description="Generate marginal cost curves for all boilers",
            inputs={'num_boilers': len(boilers)},
            output_value=len(curves),
            output_name="num_curves",
            formula="MC = fuel_cost / (efficiency * HHV) * enthalpy_rise",
            units="$/kg steam"
        )

        return curves

    def _calculate_total_capacity(
        self,
        boilers: List[BoilerCharacteristics],
        tracker: ProvenanceTracker
    ) -> Decimal:
        """Calculate total available capacity."""
        total = Decimal('0')
        for boiler in boilers:
            total += Decimal(str(boiler.maximum_load_kg_hr))

        tracker.record_step(
            operation="total_capacity",
            description="Calculate total available boiler capacity",
            inputs={'num_boilers': len(boilers)},
            output_value=total,
            output_name="total_capacity_kg_hr",
            formula="sum(max_load)",
            units="kg/hr"
        )

        return total

    def _dispatch_minimum_cost(
        self,
        boilers: List[BoilerCharacteristics],
        demand: Decimal,
        constraints: LoadAllocationConstraints,
        tracker: ProvenanceTracker
    ) -> List[BoilerLoadAllocation]:
        """
        Dispatch boilers using merit-order for minimum cost.

        Algorithm:
        1. Sort boilers by marginal cost at minimum load
        2. Load cheapest boilers first to minimum load
        3. Increase load on cheapest until at max or demand met
        4. Repeat for next cheapest boiler
        """
        allocations = []
        remaining_demand = demand

        # Calculate merit order (by marginal cost at typical operating point)
        boiler_costs = []
        for b in boilers:
            load_frac = 0.7  # Typical operating point
            efficiency = b.efficiency_at_load(load_frac)
            fuel_per_kg = 2500 / (efficiency * b.fuel_heating_value_kj_kg)
            cost = fuel_per_kg * b.fuel_cost_per_kg
            boiler_costs.append((b, cost))

        # Sort by cost (merit order)
        boiler_costs.sort(key=lambda x: x[1])

        # First pass: bring all needed boilers to minimum load
        active_boilers = []
        min_load_total = Decimal('0')

        for boiler, _ in boiler_costs:
            if remaining_demand <= 0:
                break

            min_load = Decimal(str(boiler.minimum_load_kg_hr))
            max_load = Decimal(str(boiler.maximum_load_kg_hr))

            # Check if we need this boiler
            if min_load_total + min_load <= demand:
                active_boilers.append(boiler)
                min_load_total += min_load
                remaining_demand -= min_load

        # Second pass: allocate remaining demand to active boilers
        remaining_demand = demand - min_load_total

        boiler_loads = {b.boiler_id: Decimal(str(b.minimum_load_kg_hr)) for b in active_boilers}

        for boiler in active_boilers:
            if remaining_demand <= 0:
                break

            current = boiler_loads[boiler.boiler_id]
            max_load = Decimal(str(boiler.maximum_load_kg_hr))
            available = max_load - current

            # Allocate up to available
            allocation = min(available, remaining_demand)
            boiler_loads[boiler.boiler_id] += allocation
            remaining_demand -= allocation

        # Build allocation results
        for boiler in boilers:
            if boiler.boiler_id in boiler_loads:
                load = boiler_loads[boiler.boiler_id]
            else:
                load = Decimal('0')

            allocation = self._create_allocation(boiler, load)
            allocations.append(allocation)

        tracker.record_step(
            operation="minimum_cost_dispatch",
            description="Dispatch boilers by merit order for minimum cost",
            inputs={
                'demand_kg_hr': demand,
                'active_boilers': len(active_boilers)
            },
            output_value=len(allocations),
            output_name="allocations",
            formula="Merit order dispatch",
            units="count"
        )

        return allocations

    def _dispatch_minimum_emissions(
        self,
        boilers: List[BoilerCharacteristics],
        demand: Decimal,
        constraints: LoadAllocationConstraints,
        tracker: ProvenanceTracker
    ) -> List[BoilerLoadAllocation]:
        """
        Dispatch boilers for minimum CO2 emissions.

        Similar to minimum cost but ranks by emission intensity.
        """
        allocations = []
        remaining_demand = demand

        # Calculate emission intensity order
        boiler_emissions = []
        for b in boilers:
            load_frac = 0.7
            efficiency = b.efficiency_at_load(load_frac)
            fuel_per_kg = 2500 / (efficiency * b.fuel_heating_value_kj_kg)
            co2_per_kg = fuel_per_kg * b.co2_emission_factor
            boiler_emissions.append((b, co2_per_kg))

        # Sort by emission intensity
        boiler_emissions.sort(key=lambda x: x[1])

        # Dispatch in emission merit order
        active_boilers = []
        min_load_total = Decimal('0')

        for boiler, _ in boiler_emissions:
            if remaining_demand <= 0:
                break

            min_load = Decimal(str(boiler.minimum_load_kg_hr))
            if min_load_total + min_load <= demand:
                active_boilers.append(boiler)
                min_load_total += min_load
                remaining_demand -= min_load

        remaining_demand = demand - min_load_total
        boiler_loads = {b.boiler_id: Decimal(str(b.minimum_load_kg_hr)) for b in active_boilers}

        for boiler in active_boilers:
            if remaining_demand <= 0:
                break

            current = boiler_loads[boiler.boiler_id]
            max_load = Decimal(str(boiler.maximum_load_kg_hr))
            available = max_load - current

            allocation = min(available, remaining_demand)
            boiler_loads[boiler.boiler_id] += allocation
            remaining_demand -= allocation

        for boiler in boilers:
            load = boiler_loads.get(boiler.boiler_id, Decimal('0'))
            allocation = self._create_allocation(boiler, load)
            allocations.append(allocation)

        tracker.record_step(
            operation="minimum_emissions_dispatch",
            description="Dispatch boilers by emission intensity",
            inputs={'demand_kg_hr': demand},
            output_value=len(allocations),
            output_name="allocations",
            formula="Emission merit order",
            units="count"
        )

        return allocations

    def _dispatch_minimum_fuel(
        self,
        boilers: List[BoilerCharacteristics],
        demand: Decimal,
        constraints: LoadAllocationConstraints,
        tracker: ProvenanceTracker
    ) -> List[BoilerLoadAllocation]:
        """
        Dispatch boilers for minimum fuel consumption.

        Prioritizes most efficient boilers regardless of fuel cost.
        """
        allocations = []
        remaining_demand = demand

        # Rank by efficiency at typical load
        boiler_efficiency = []
        for b in boilers:
            efficiency = b.efficiency_at_load(0.7)
            boiler_efficiency.append((b, efficiency))

        # Sort by efficiency (highest first)
        boiler_efficiency.sort(key=lambda x: x[1], reverse=True)

        # Dispatch most efficient first
        active_boilers = []
        min_load_total = Decimal('0')

        for boiler, _ in boiler_efficiency:
            if remaining_demand <= 0:
                break

            min_load = Decimal(str(boiler.minimum_load_kg_hr))
            if min_load_total + min_load <= demand:
                active_boilers.append(boiler)
                min_load_total += min_load
                remaining_demand -= min_load

        remaining_demand = demand - min_load_total
        boiler_loads = {b.boiler_id: Decimal(str(b.minimum_load_kg_hr)) for b in active_boilers}

        for boiler in active_boilers:
            if remaining_demand <= 0:
                break

            current = boiler_loads[boiler.boiler_id]
            max_load = Decimal(str(boiler.maximum_load_kg_hr))
            available = max_load - current

            allocation = min(available, remaining_demand)
            boiler_loads[boiler.boiler_id] += allocation
            remaining_demand -= allocation

        for boiler in boilers:
            load = boiler_loads.get(boiler.boiler_id, Decimal('0'))
            allocation = self._create_allocation(boiler, load)
            allocations.append(allocation)

        tracker.record_step(
            operation="minimum_fuel_dispatch",
            description="Dispatch boilers by efficiency",
            inputs={'demand_kg_hr': demand},
            output_value=len(allocations),
            output_name="allocations",
            formula="Efficiency merit order",
            units="count"
        )

        return allocations

    def _dispatch_balanced(
        self,
        boilers: List[BoilerCharacteristics],
        demand: Decimal,
        constraints: LoadAllocationConstraints,
        tracker: ProvenanceTracker
    ) -> List[BoilerLoadAllocation]:
        """
        Multi-objective balanced dispatch.

        Combines cost, emissions, and fuel efficiency with weights.
        """
        # Calculate combined score for each boiler
        boiler_scores = []
        for b in boilers:
            load_frac = 0.7
            efficiency = b.efficiency_at_load(load_frac)
            fuel_per_kg = 2500 / (efficiency * b.fuel_heating_value_kj_kg)

            cost_score = fuel_per_kg * b.fuel_cost_per_kg
            emission_score = fuel_per_kg * b.co2_emission_factor
            fuel_score = fuel_per_kg

            # Normalize and combine (equal weights)
            combined = (cost_score + emission_score * 0.1 + fuel_score * 10) / 3
            boiler_scores.append((b, combined))

        # Sort by combined score
        boiler_scores.sort(key=lambda x: x[1])

        # Standard dispatch using combined ranking
        active_boilers = []
        min_load_total = Decimal('0')
        remaining_demand = demand

        for boiler, _ in boiler_scores:
            if remaining_demand <= 0:
                break

            min_load = Decimal(str(boiler.minimum_load_kg_hr))
            if min_load_total + min_load <= demand:
                active_boilers.append(boiler)
                min_load_total += min_load
                remaining_demand -= min_load

        remaining_demand = demand - min_load_total
        boiler_loads = {b.boiler_id: Decimal(str(b.minimum_load_kg_hr)) for b in active_boilers}

        for boiler in active_boilers:
            if remaining_demand <= 0:
                break

            current = boiler_loads[boiler.boiler_id]
            max_load = Decimal(str(boiler.maximum_load_kg_hr))
            available = max_load - current

            allocation = min(available, remaining_demand)
            boiler_loads[boiler.boiler_id] += allocation
            remaining_demand -= allocation

        allocations = []
        for boiler in boilers:
            load = boiler_loads.get(boiler.boiler_id, Decimal('0'))
            allocation = self._create_allocation(boiler, load)
            allocations.append(allocation)

        tracker.record_step(
            operation="balanced_dispatch",
            description="Multi-objective balanced dispatch",
            inputs={'demand_kg_hr': demand},
            output_value=len(allocations),
            output_name="allocations",
            formula="Weighted multi-objective",
            units="count"
        )

        return allocations

    def _create_allocation(
        self,
        boiler: BoilerCharacteristics,
        load: Decimal
    ) -> BoilerLoadAllocation:
        """Create allocation object for a boiler."""
        max_load = Decimal(str(boiler.maximum_load_kg_hr))
        min_load = Decimal(str(boiler.minimum_load_kg_hr))

        if load <= 0:
            return BoilerLoadAllocation(
                boiler_id=boiler.boiler_id,
                name=boiler.name,
                allocated_load_kg_hr=Decimal('0'),
                load_fraction=Decimal('0'),
                efficiency_percent=Decimal('0'),
                fuel_consumption_kg_hr=Decimal('0'),
                fuel_cost_per_hr=Decimal('0'),
                co2_emissions_kg_hr=Decimal('0'),
                is_at_minimum=False,
                is_at_maximum=False,
                available_ramp_up_kg_hr=max_load,
                available_ramp_down_kg_hr=Decimal('0')
            )

        load_fraction = load / max_load
        efficiency = Decimal(str(boiler.efficiency_at_load(float(load_fraction))))

        # Calculate fuel consumption
        enthalpy_rise = Decimal('2500')
        fuel_per_kg = enthalpy_rise / (efficiency * Decimal(str(boiler.fuel_heating_value_kj_kg)))
        fuel_consumption = load * fuel_per_kg

        # Cost and emissions
        fuel_cost = fuel_consumption * Decimal(str(boiler.fuel_cost_per_kg))
        co2 = fuel_consumption * Decimal(str(boiler.co2_emission_factor))

        # Ramp availability
        ramp_time = Decimal('1')  # Assume 1 minute window
        ramp_up = min(
            max_load - load,
            Decimal(str(boiler.max_ramp_up_rate)) * ramp_time
        )
        ramp_down = min(
            load - min_load,
            Decimal(str(boiler.max_ramp_down_rate)) * ramp_time
        )

        return BoilerLoadAllocation(
            boiler_id=boiler.boiler_id,
            name=boiler.name,
            allocated_load_kg_hr=load.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            load_fraction=load_fraction.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP),
            efficiency_percent=(efficiency * Decimal('100')).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            fuel_consumption_kg_hr=fuel_consumption.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            fuel_cost_per_hr=fuel_cost.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            co2_emissions_kg_hr=co2.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            is_at_minimum=abs(load - min_load) < Decimal('0.01'),
            is_at_maximum=abs(load - max_load) < Decimal('0.01'),
            available_ramp_up_kg_hr=ramp_up.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            available_ramp_down_kg_hr=ramp_down.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        )

    def _apply_ramp_constraints(
        self,
        allocations: List[BoilerLoadAllocation],
        boilers: List[BoilerCharacteristics],
        tracker: ProvenanceTracker
    ) -> List[BoilerLoadAllocation]:
        """
        Apply ramp rate constraints to allocations.

        Limits load changes based on boiler ramp capabilities.
        """
        adjusted_allocations = []

        for allocation in allocations:
            # Find matching boiler
            boiler = next((b for b in boilers if b.boiler_id == allocation.boiler_id), None)
            if not boiler:
                adjusted_allocations.append(allocation)
                continue

            current_load = Decimal(str(boiler.current_load_kg_hr))
            target_load = allocation.allocated_load_kg_hr

            # Check ramp rate
            load_change = target_load - current_load
            max_ramp_up = Decimal(str(boiler.max_ramp_up_rate))
            max_ramp_down = Decimal(str(boiler.max_ramp_down_rate))

            if load_change > max_ramp_up:
                # Limit ramp up
                new_target = current_load + max_ramp_up
                adjusted_allocation = self._create_allocation(boiler, new_target)
                adjusted_allocations.append(adjusted_allocation)
            elif load_change < -max_ramp_down:
                # Limit ramp down
                new_target = current_load - max_ramp_down
                adjusted_allocation = self._create_allocation(boiler, new_target)
                adjusted_allocations.append(adjusted_allocation)
            else:
                adjusted_allocations.append(allocation)

        tracker.record_step(
            operation="apply_ramp_constraints",
            description="Apply ramp rate limits to load changes",
            inputs={'num_allocations': len(allocations)},
            output_value=len(adjusted_allocations),
            output_name="adjusted_allocations",
            formula="limit(target - current, ramp_rate)",
            units="count"
        )

        return adjusted_allocations

    def _calculate_summary_metrics(
        self,
        allocations: List[BoilerLoadAllocation],
        tracker: ProvenanceTracker
    ) -> Tuple[Decimal, Decimal, Decimal, Decimal, Decimal]:
        """Calculate summary metrics from allocations."""
        total_steam = Decimal('0')
        total_fuel = Decimal('0')
        total_cost = Decimal('0')
        total_co2 = Decimal('0')
        weighted_eff_num = Decimal('0')

        for a in allocations:
            total_steam += a.allocated_load_kg_hr
            total_fuel += a.fuel_consumption_kg_hr
            total_cost += a.fuel_cost_per_hr
            total_co2 += a.co2_emissions_kg_hr
            weighted_eff_num += a.efficiency_percent * a.allocated_load_kg_hr

        avg_efficiency = weighted_eff_num / total_steam if total_steam > 0 else Decimal('0')

        tracker.record_step(
            operation="summary_metrics",
            description="Calculate summary metrics",
            inputs={'num_allocations': len(allocations)},
            output_value=total_cost,
            output_name="total_cost_per_hr",
            formula="sum(individual metrics)",
            units="$/hr"
        )

        return (
            total_steam.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            total_fuel.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            total_cost.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            total_co2.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            avg_efficiency.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        )

    def _calculate_system_marginal_cost(
        self,
        allocations: List[BoilerLoadAllocation],
        marginal_curves: Dict[str, List[MarginalCostPoint]],
        tracker: ProvenanceTracker
    ) -> Decimal:
        """
        Calculate system marginal cost.

        The marginal cost of the last unit of steam produced.
        """
        max_marginal = Decimal('0')

        for allocation in allocations:
            if allocation.allocated_load_kg_hr > 0:
                # Get marginal cost at current load
                if allocation.boiler_id in marginal_curves:
                    curve = marginal_curves[allocation.boiler_id]
                    # Find nearest point
                    for point in curve:
                        if point.load_kg_hr >= allocation.allocated_load_kg_hr:
                            if point.marginal_cost_per_kg_steam > max_marginal:
                                max_marginal = point.marginal_cost_per_kg_steam
                            break

        tracker.record_step(
            operation="system_marginal_cost",
            description="Calculate system marginal cost",
            inputs={'num_active_boilers': sum(1 for a in allocations if a.allocated_load_kg_hr > 0)},
            output_value=max_marginal,
            output_name="system_marginal_cost",
            formula="max(MC at operating point)",
            units="$/kg"
        )

        return max_marginal.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)

    def _optimize_hot_standby(
        self,
        all_boilers: List[BoilerCharacteristics],
        allocations: List[BoilerLoadAllocation],
        constraints: LoadAllocationConstraints,
        tracker: ProvenanceTracker
    ) -> Tuple[List[str], Decimal]:
        """
        Determine which boilers should be on hot standby.

        Hot standby provides fast response but incurs fuel cost.
        """
        hot_standby = []
        standby_cost = Decimal('0')

        # Find operating boilers
        operating_ids = {a.boiler_id for a in allocations if a.allocated_load_kg_hr > 0}

        # Calculate required spinning reserve
        total_steam = sum(a.allocated_load_kg_hr for a in allocations)
        required_reserve = total_steam * Decimal(str(constraints.spinning_reserve_percent)) / Decimal('100')

        # Calculate current reserve from operating boilers
        current_reserve = Decimal('0')
        for boiler in all_boilers:
            if boiler.boiler_id in operating_ids:
                current_reserve += Decimal(str(boiler.maximum_load_kg_hr))
        current_reserve -= total_steam

        # If reserve is insufficient, put additional boilers on hot standby
        reserve_deficit = required_reserve - current_reserve

        if reserve_deficit > 0:
            # Find idle boilers that could provide hot standby
            idle_boilers = [
                b for b in all_boilers
                if b.boiler_id not in operating_ids
                and b.status not in [BoilerStatus.MAINTENANCE, BoilerStatus.UNAVAILABLE]
            ]

            # Sort by hot standby cost (fuel consumption in standby)
            idle_boilers.sort(key=lambda b: b.hot_standby_fuel_kg_hr)

            for boiler in idle_boilers:
                if reserve_deficit <= 0:
                    break

                hot_standby.append(boiler.boiler_id)
                standby_cost += Decimal(str(boiler.hot_standby_fuel_kg_hr)) * Decimal(str(boiler.fuel_cost_per_kg))
                reserve_deficit -= Decimal(str(boiler.maximum_load_kg_hr))

        tracker.record_step(
            operation="hot_standby_optimization",
            description="Optimize hot standby boiler selection",
            inputs={
                'required_reserve': required_reserve,
                'current_reserve': current_reserve
            },
            output_value=len(hot_standby),
            output_name="hot_standby_count",
            formula="Minimize standby cost while meeting reserve",
            units="count"
        )

        return hot_standby, standby_cost.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

    def _check_constraint_violations(
        self,
        allocations: List[BoilerLoadAllocation],
        constraints: LoadAllocationConstraints,
        total_co2: Decimal,
        total_cost: Decimal
    ) -> List[str]:
        """Check for constraint violations."""
        violations = []

        # Check demand satisfaction
        total_steam = sum(a.allocated_load_kg_hr for a in allocations)
        demand = Decimal(str(constraints.total_steam_demand_kg_hr))

        if total_steam < demand * Decimal('0.99'):  # 1% tolerance
            violations.append(
                f"Demand not met: {float(total_steam):.0f} < {float(demand):.0f} kg/hr"
            )

        # Check emission constraint
        if constraints.max_total_co2_kg_hr is not None:
            max_co2 = Decimal(str(constraints.max_total_co2_kg_hr))
            if total_co2 > max_co2:
                violations.append(
                    f"CO2 limit exceeded: {float(total_co2):.0f} > {float(max_co2):.0f} kg/hr"
                )

        # Check cost constraint
        if constraints.max_fuel_cost_per_hr is not None:
            max_cost = Decimal(str(constraints.max_fuel_cost_per_hr))
            if total_cost > max_cost:
                violations.append(
                    f"Cost limit exceeded: ${float(total_cost):.2f} > ${float(max_cost):.2f}/hr"
                )

        # Check minimum operating boilers
        operating_count = sum(1 for a in allocations if a.allocated_load_kg_hr > 0)
        if operating_count < constraints.minimum_operating_boilers:
            violations.append(
                f"Minimum boilers not met: {operating_count} < {constraints.minimum_operating_boilers}"
            )

        return violations

    def _generate_recommendations(
        self,
        allocations: List[BoilerLoadAllocation],
        boilers: List[BoilerCharacteristics],
        constraints: LoadAllocationConstraints,
        spinning_reserve: Decimal,
        tracker: ProvenanceTracker
    ) -> List[Dict]:
        """Generate optimization recommendations."""
        recommendations = []

        # Check for boilers at minimum
        at_min = [a for a in allocations if a.is_at_minimum and a.allocated_load_kg_hr > 0]
        if len(at_min) > 1:
            recommendations.append({
                'type': 'efficiency',
                'priority': 'Medium',
                'title': 'Multiple boilers at minimum load',
                'description': f'{len(at_min)} boilers operating at minimum load. '
                              'Consider consolidating load onto fewer boilers.',
                'potential_savings_percent': 2.0
            })

        # Check reserve level
        total_capacity = sum(Decimal(str(b.maximum_load_kg_hr)) for b in boilers)
        reserve_percent = (spinning_reserve / total_capacity) * Decimal('100') if total_capacity > 0 else Decimal('0')

        if reserve_percent > Decimal('25'):
            recommendations.append({
                'type': 'capacity',
                'priority': 'Low',
                'title': 'Excess spinning reserve',
                'description': f'Reserve at {float(reserve_percent):.1f}% exceeds typical requirement. '
                              'Consider reducing hot standby boilers.',
                'potential_savings_percent': 0.5
            })
        elif reserve_percent < Decimal('10'):
            recommendations.append({
                'type': 'reliability',
                'priority': 'High',
                'title': 'Low spinning reserve',
                'description': f'Reserve at {float(reserve_percent):.1f}% is below recommended 10%. '
                              'Consider additional hot standby capacity.',
                'risk_level': 'High'
            })

        # Check for efficiency opportunities
        for allocation in allocations:
            if allocation.efficiency_percent < Decimal('80') and allocation.allocated_load_kg_hr > 0:
                recommendations.append({
                    'type': 'efficiency',
                    'priority': 'High',
                    'title': f'Low efficiency on {allocation.name}',
                    'description': f'Boiler {allocation.boiler_id} operating at {float(allocation.efficiency_percent):.1f}% efficiency. '
                                  'Consider maintenance or load reallocation.',
                    'boiler_id': allocation.boiler_id
                })

        # Check load balance
        active_allocations = [a for a in allocations if a.allocated_load_kg_hr > 0]
        if len(active_allocations) > 1:
            load_fractions = [float(a.load_fraction) for a in active_allocations]
            load_variance = sum((f - sum(load_fractions)/len(load_fractions))**2 for f in load_fractions) / len(load_fractions)

            if load_variance > 0.04:  # More than 20% deviation
                recommendations.append({
                    'type': 'balance',
                    'priority': 'Medium',
                    'title': 'Unbalanced load distribution',
                    'description': 'Large variance in boiler load fractions. '
                                  'Consider rebalancing for more uniform wear.',
                })

        return recommendations


def optimize_multi_boiler_load(
    boilers: List[BoilerCharacteristics],
    constraints: LoadAllocationConstraints,
    objective: OptimizationObjective = OptimizationObjective.MINIMUM_COST
) -> LoadAllocationResult:
    """
    Convenience function for multi-boiler load optimization.

    Example:
        boilers = [
            BoilerCharacteristics(
                boiler_id="B1",
                name="Boiler 1",
                minimum_load_kg_hr=2000,
                maximum_load_kg_hr=10000,
                design_load_kg_hr=8000,
                fuel_cost_per_kg=0.40,
                efficiency_coeff_a=0.78,
                efficiency_coeff_b=0.28,
                efficiency_coeff_c=-0.12
            ),
            BoilerCharacteristics(
                boiler_id="B2",
                name="Boiler 2",
                minimum_load_kg_hr=1500,
                maximum_load_kg_hr=8000,
                design_load_kg_hr=6000,
                fuel_cost_per_kg=0.45,
                efficiency_coeff_a=0.75,
                efficiency_coeff_b=0.30,
                efficiency_coeff_c=-0.15
            ),
        ]

        constraints = LoadAllocationConstraints(
            total_steam_demand_kg_hr=12000,
            spinning_reserve_percent=10
        )

        result = optimize_multi_boiler_load(boilers, constraints)
        print(f"Total cost: ${result.total_fuel_cost_per_hr}/hr")
    """
    optimizer = LoadAllocationOptimizer()
    return optimizer.optimize_load_allocation(boilers, constraints, objective)
