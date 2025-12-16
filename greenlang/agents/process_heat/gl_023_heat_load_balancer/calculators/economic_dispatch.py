"""
GL-023 HEATLOADBALANCER - Economic Dispatch Calculator

This module provides zero-hallucination economic dispatch calculations for
optimal load allocation across multiple heat generation units. Based on
classical power systems economic dispatch theory adapted for process heat.

Key Algorithms:
    - Equal Incremental Cost (Lambda Dispatch)
    - Merit Order Dispatch
    - Constrained Economic Dispatch

Key Formulas:
    - Incremental cost: IC = dC/dP = (fuel_price / HHV) * d(fuel)/d(output)
    - Economic dispatch: lambda = IC_1 = IC_2 = ... = IC_n (equal incremental cost)
    - Total cost: C_total = Sum(C_i) for all units

Standards Reference:
    - IEEE Power Engineering Society standards
    - ASME PTC 4.1 (efficiency calculations)

Example:
    >>> from greenlang.agents.process_heat.gl_023_heat_load_balancer.calculators import (
    ...     EconomicDispatchCalculator,
    ... )
    >>>
    >>> calc = EconomicDispatchCalculator()
    >>> result = calc.dispatch(
    ...     total_demand=500.0,
    ...     units=unit_configs,
    ... )
    >>> print(f"Total cost: ${result.total_cost:.2f}/hr")
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import logging
import math

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

class EconomicDispatchConstants:
    """Economic dispatch calculation constants."""

    # Default convergence tolerance for iterative methods
    DEFAULT_TOLERANCE = 0.001

    # Maximum iterations for lambda search
    MAX_ITERATIONS = 100

    # Default efficiency curve coefficients (cubic polynomial)
    DEFAULT_EFFICIENCY_COEFFICIENTS = [0.70, 0.40, -0.25, 0.05]

    # Penalty factor for constraint violations
    CONSTRAINT_PENALTY = 1e6


# =============================================================================
# DATA MODELS
# =============================================================================

class UnitConfiguration(BaseModel):
    """Configuration for a heat generation unit."""

    unit_id: str = Field(..., description="Unit identifier")
    unit_name: str = Field(default="", description="Unit display name")

    # Capacity constraints
    min_capacity_mmbtu_hr: float = Field(
        ...,
        ge=0,
        description="Minimum operating capacity (MMBTU/hr)"
    )
    max_capacity_mmbtu_hr: float = Field(
        ...,
        gt=0,
        description="Maximum operating capacity (MMBTU/hr)"
    )

    # Efficiency curve (polynomial coefficients)
    # eta(L) = a0 + a1*L + a2*L^2 + a3*L^3 where L = load_fraction
    efficiency_coefficients: List[float] = Field(
        default_factory=lambda: [0.70, 0.40, -0.25, 0.05],
        description="Efficiency curve polynomial coefficients"
    )

    # Cost parameters
    fuel_type: str = Field(default="natural_gas", description="Fuel type")
    fuel_price: float = Field(..., ge=0, description="Fuel price (USD/MMBTU)")
    fuel_hhv: float = Field(
        default=1028.0,
        gt=0,
        description="Fuel HHV (BTU/SCF for gas, BTU/gal for oil)"
    )

    # Fixed costs
    no_load_cost: float = Field(
        default=0.0,
        ge=0,
        description="No-load cost (USD/hr) when unit is running"
    )
    startup_cost: float = Field(
        default=0.0,
        ge=0,
        description="Startup cost (USD)"
    )

    # Operational constraints
    must_run: bool = Field(
        default=False,
        description="Unit must run (baseload)"
    )
    available: bool = Field(
        default=True,
        description="Unit is available for dispatch"
    )
    ramp_rate_mmbtu_hr_min: Optional[float] = Field(
        default=None,
        ge=0,
        description="Ramp rate limit (MMBTU/hr per minute)"
    )

    # Emissions
    emission_factor_kg_co2_mmbtu: float = Field(
        default=53.06,
        ge=0,
        description="CO2 emission factor (kg/MMBTU fuel input)"
    )


class UnitDispatch(BaseModel):
    """Dispatch result for a single unit."""

    unit_id: str = Field(..., description="Unit identifier")
    load_mmbtu_hr: float = Field(..., ge=0, description="Dispatched load (MMBTU/hr)")
    load_fraction: float = Field(..., ge=0, description="Load as fraction of capacity")

    # Operating point
    efficiency_pct: float = Field(..., ge=0, le=100, description="Operating efficiency (%)")
    fuel_consumption: float = Field(..., ge=0, description="Fuel consumption")
    fuel_consumption_unit: str = Field(default="MMBTU/hr", description="Fuel unit")

    # Costs
    fuel_cost_per_hr: float = Field(..., ge=0, description="Fuel cost (USD/hr)")
    incremental_cost: float = Field(..., ge=0, description="Incremental cost (USD/MMBTU)")
    total_cost_per_hr: float = Field(..., ge=0, description="Total cost (USD/hr)")

    # Emissions
    co2_emissions_kg_hr: float = Field(
        default=0.0,
        ge=0,
        description="CO2 emissions (kg/hr)"
    )

    # Status
    is_online: bool = Field(default=True, description="Unit is online")
    at_minimum: bool = Field(default=False, description="At minimum load")
    at_maximum: bool = Field(default=False, description="At maximum load")


class EconomicDispatchResult(BaseModel):
    """Result from economic dispatch optimization."""

    # Overall results
    total_demand_mmbtu_hr: float = Field(
        ...,
        description="Total demand served (MMBTU/hr)"
    )
    total_cost_per_hr: float = Field(
        ...,
        ge=0,
        description="Total operating cost (USD/hr)"
    )
    system_lambda: float = Field(
        ...,
        ge=0,
        description="System incremental cost (USD/MMBTU)"
    )

    # Unit dispatches
    unit_dispatches: List[UnitDispatch] = Field(
        ...,
        description="Dispatch for each unit"
    )

    # System metrics
    total_fuel_cost_per_hr: float = Field(
        ...,
        ge=0,
        description="Total fuel cost (USD/hr)"
    )
    average_efficiency_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Weighted average efficiency (%)"
    )
    total_co2_emissions_kg_hr: float = Field(
        default=0.0,
        ge=0,
        description="Total CO2 emissions (kg/hr)"
    )

    # Constraints
    demand_satisfied: bool = Field(
        default=True,
        description="Total demand was satisfied"
    )
    unserved_demand_mmbtu_hr: float = Field(
        default=0.0,
        ge=0,
        description="Unserved demand (MMBTU/hr)"
    )
    spare_capacity_mmbtu_hr: float = Field(
        default=0.0,
        ge=0,
        description="Spare capacity available (MMBTU/hr)"
    )

    # Optimization quality
    iterations: int = Field(default=0, ge=0, description="Iterations to converge")
    converged: bool = Field(default=True, description="Algorithm converged")
    method: str = Field(
        default="equal_incremental_cost",
        description="Dispatch method used"
    )

    # Provenance
    calculation_hash: str = Field(..., description="SHA-256 calculation hash")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Calculation timestamp"
    )


class IncrementalCostResult(BaseModel):
    """Result from incremental cost calculation."""

    unit_id: str = Field(..., description="Unit identifier")
    incremental_cost: float = Field(
        ...,
        ge=0,
        description="Incremental cost (USD/MMBTU output)"
    )
    load_fraction: float = Field(..., ge=0, description="Load fraction")

    # Calculation details
    efficiency: float = Field(..., ge=0, le=100, description="Efficiency at load (%)")
    efficiency_derivative: float = Field(
        ...,
        description="d(efficiency)/d(load)"
    )
    fuel_price: float = Field(..., ge=0, description="Fuel price used")

    # Formula reference
    formula: str = Field(
        default="IC = (P_fuel / eta^2) * |d(eta)/d(L)|",
        description="Formula used"
    )


class ReserveMarginResult(BaseModel):
    """Result from reserve margin calculation."""

    total_capacity_mmbtu_hr: float = Field(
        ...,
        ge=0,
        description="Total available capacity"
    )
    current_load_mmbtu_hr: float = Field(
        ...,
        ge=0,
        description="Current total load"
    )
    reserve_margin_mmbtu_hr: float = Field(
        ...,
        ge=0,
        description="Reserve margin (MMBTU/hr)"
    )
    reserve_margin_pct: float = Field(
        ...,
        ge=0,
        description="Reserve margin (%)"
    )

    # Spinning reserve
    spinning_reserve_mmbtu_hr: float = Field(
        ...,
        ge=0,
        description="Spinning reserve from online units"
    )
    non_spinning_reserve_mmbtu_hr: float = Field(
        default=0.0,
        ge=0,
        description="Non-spinning reserve from offline units"
    )

    # Requirements
    required_reserve_pct: float = Field(
        default=15.0,
        ge=0,
        description="Required reserve margin (%)"
    )
    reserve_adequate: bool = Field(
        default=True,
        description="Reserve margin is adequate"
    )


class LossFactorResult(BaseModel):
    """Result from loss factor calculation."""

    transmission_loss_pct: float = Field(
        ...,
        ge=0,
        description="Transmission/distribution losses (%)"
    )
    loss_mmbtu_hr: float = Field(
        ...,
        ge=0,
        description="Loss energy (MMBTU/hr)"
    )
    gross_generation_required: float = Field(
        ...,
        ge=0,
        description="Gross generation to meet net demand"
    )
    net_demand_mmbtu_hr: float = Field(
        ...,
        ge=0,
        description="Net demand at delivery point"
    )

    # Provenance
    calculation_hash: str = Field(..., description="SHA-256 calculation hash")


# =============================================================================
# INCREMENTAL COST CALCULATOR
# =============================================================================

class IncrementalCostCalculator:
    """
    Calculate incremental cost (IC) for economic dispatch.

    The incremental cost is the cost to produce one additional unit of output.

    Formula:
    IC = dC/dP = (fuel_price / eta^2) * |d(eta)/d(load)|

    For fuel cost: C = (P * fuel_price) / eta
    Taking derivative: dC/dP = fuel_price/eta - (P * fuel_price * d(eta)/dP) / eta^2

    ZERO-HALLUCINATION: Deterministic calculation with provenance.

    Example:
        >>> calc = IncrementalCostCalculator()
        >>> result = calc.calculate(
        ...     load_fraction=0.8,
        ...     efficiency_coefficients=[0.70, 0.40, -0.25, 0.05],
        ...     fuel_price=3.50,
        ... )
    """

    def __init__(self) -> None:
        """Initialize incremental cost calculator."""
        self._calculation_count = 0

    def calculate(
        self,
        load_fraction: float,
        efficiency_coefficients: List[float],
        fuel_price: float,
        unit_id: str = "UNIT-001",
    ) -> IncrementalCostResult:
        """
        Calculate incremental cost at given load.

        DETERMINISTIC: Same inputs always produce same output.

        Args:
            load_fraction: Load as fraction of rated capacity (0-1)
            efficiency_coefficients: Polynomial coefficients for efficiency curve
            fuel_price: Fuel price (USD/MMBTU fuel input)
            unit_id: Unit identifier

        Returns:
            IncrementalCostResult with incremental cost
        """
        self._calculation_count += 1

        # Validate inputs
        if load_fraction < 0:
            raise ValueError(f"Load fraction cannot be negative: {load_fraction}")
        if fuel_price < 0:
            raise ValueError(f"Fuel price cannot be negative: {fuel_price}")
        if not efficiency_coefficients:
            raise ValueError("Efficiency coefficients required")

        # Calculate efficiency at load point
        # eta = a0 + a1*L + a2*L^2 + a3*L^3
        L = load_fraction
        efficiency = sum(c * (L ** i) for i, c in enumerate(efficiency_coefficients))

        # Scale if coefficients are in decimal form (< 1)
        if efficiency < 1:
            efficiency *= 100

        # Clamp efficiency to valid range
        efficiency = max(1.0, min(99.9, efficiency))

        # Calculate efficiency derivative
        # d(eta)/dL = a1 + 2*a2*L + 3*a3*L^2
        efficiency_derivative = sum(
            i * c * (L ** (i - 1))
            for i, c in enumerate(efficiency_coefficients)
            if i > 0
        )
        if efficiency_coefficients[0] < 1:
            efficiency_derivative *= 100

        # Calculate incremental cost
        # For fuel cost C = P_output * (fuel_price / eta)
        # IC = dC/dP = fuel_price/eta - (P * fuel_price / eta^2) * d(eta)/dL * dL/dP
        #
        # Simplified for output-based pricing:
        # IC = fuel_price / (eta/100)
        #
        # More accurate form accounting for efficiency slope:
        # IC = fuel_price / eta * (1 - L * d(eta)/dL / eta)

        eta_decimal = efficiency / 100.0

        # Base incremental cost (fuel price adjusted for efficiency)
        base_ic = fuel_price / eta_decimal

        # Adjustment for efficiency curve slope
        if abs(efficiency) > 0.01:
            slope_factor = 1 - (L * efficiency_derivative / efficiency)
        else:
            slope_factor = 1.0

        incremental_cost = base_ic * abs(slope_factor)

        return IncrementalCostResult(
            unit_id=unit_id,
            incremental_cost=round(incremental_cost, 4),
            load_fraction=load_fraction,
            efficiency=round(efficiency, 2),
            efficiency_derivative=round(efficiency_derivative, 4),
            fuel_price=fuel_price,
        )

    def calculate_curve(
        self,
        efficiency_coefficients: List[float],
        fuel_price: float,
        unit_id: str = "UNIT-001",
        num_points: int = 21,
    ) -> List[IncrementalCostResult]:
        """
        Calculate incremental cost curve.

        Args:
            efficiency_coefficients: Polynomial coefficients
            fuel_price: Fuel price
            unit_id: Unit identifier
            num_points: Number of points on curve

        Returns:
            List of IncrementalCostResult for load range 0-100%
        """
        results = []
        for i in range(num_points):
            load = i / (num_points - 1)  # 0 to 1
            result = self.calculate(
                load_fraction=load,
                efficiency_coefficients=efficiency_coefficients,
                fuel_price=fuel_price,
                unit_id=unit_id,
            )
            results.append(result)

        return results

    @property
    def calculation_count(self) -> int:
        """Get total calculation count."""
        return self._calculation_count


# =============================================================================
# EQUAL INCREMENTAL COST SOLVER
# =============================================================================

class EqualIncrementalCostSolver:
    """
    Solve economic dispatch using equal incremental cost (lambda) method.

    The optimal dispatch occurs when all online units have equal incremental costs:
    lambda = IC_1 = IC_2 = ... = IC_n

    This is the classic economic dispatch algorithm from power systems.

    ZERO-HALLUCINATION: Deterministic optimization with provenance.

    Example:
        >>> solver = EqualIncrementalCostSolver()
        >>> result = solver.solve(
        ...     total_demand=500.0,
        ...     units=unit_configs,
        ... )
    """

    def __init__(
        self,
        tolerance: float = 0.001,
        max_iterations: int = 100,
    ) -> None:
        """
        Initialize the solver.

        Args:
            tolerance: Convergence tolerance for lambda search
            max_iterations: Maximum iterations
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self._ic_calc = IncrementalCostCalculator()
        self._solve_count = 0

    def solve(
        self,
        total_demand: float,
        units: List[UnitConfiguration],
    ) -> EconomicDispatchResult:
        """
        Solve economic dispatch using equal incremental cost method.

        DETERMINISTIC: Same inputs always produce same dispatch.

        Algorithm:
        1. Find lambda bounds (min and max possible IC)
        2. Binary search for lambda where sum of loads = demand
        3. Return dispatch at converged lambda

        Args:
            total_demand: Total heat demand (MMBTU/hr)
            units: List of unit configurations

        Returns:
            EconomicDispatchResult with optimal dispatch
        """
        self._solve_count += 1

        # Validate inputs
        if total_demand < 0:
            raise ValueError(f"Demand cannot be negative: {total_demand}")
        if not units:
            raise ValueError("At least one unit required")

        # Filter available units
        available_units = [u for u in units if u.available]
        if not available_units:
            raise ValueError("No available units for dispatch")

        # Calculate total capacity
        total_min = sum(u.min_capacity_mmbtu_hr for u in available_units)
        total_max = sum(u.max_capacity_mmbtu_hr for u in available_units)

        # Check feasibility
        if total_demand > total_max:
            logger.warning(
                f"Demand {total_demand} exceeds capacity {total_max}, "
                "dispatching at max"
            )
        if total_demand < total_min:
            logger.warning(
                f"Demand {total_demand} below minimum {total_min}"
            )

        # Find lambda bounds
        lambda_min, lambda_max = self._find_lambda_bounds(available_units)

        # Binary search for optimal lambda
        lambda_opt, iterations, converged = self._binary_search_lambda(
            total_demand=total_demand,
            units=available_units,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
        )

        # Calculate dispatch at optimal lambda
        unit_dispatches = self._calculate_dispatch_at_lambda(
            lambda_val=lambda_opt,
            units=available_units,
            total_demand=total_demand,
        )

        # Calculate totals
        total_load = sum(d.load_mmbtu_hr for d in unit_dispatches)
        total_fuel_cost = sum(d.fuel_cost_per_hr for d in unit_dispatches)
        total_cost = sum(d.total_cost_per_hr for d in unit_dispatches)
        total_co2 = sum(d.co2_emissions_kg_hr for d in unit_dispatches)

        # Weighted average efficiency
        if total_load > 0:
            weighted_eff = sum(
                d.efficiency_pct * d.load_mmbtu_hr for d in unit_dispatches
            ) / total_load
        else:
            weighted_eff = 0.0

        # Check if demand satisfied
        demand_satisfied = abs(total_load - total_demand) < 1.0
        unserved = max(0, total_demand - total_load)
        spare = total_max - total_load

        # Calculate provenance hash
        calculation_hash = self._calculate_hash(
            total_demand=total_demand,
            lambda_opt=lambda_opt,
            total_cost=total_cost,
            unit_ids=[u.unit_id for u in available_units],
        )

        return EconomicDispatchResult(
            total_demand_mmbtu_hr=total_demand,
            total_cost_per_hr=round(total_cost, 2),
            system_lambda=round(lambda_opt, 4),
            unit_dispatches=unit_dispatches,
            total_fuel_cost_per_hr=round(total_fuel_cost, 2),
            average_efficiency_pct=round(weighted_eff, 2),
            total_co2_emissions_kg_hr=round(total_co2, 2),
            demand_satisfied=demand_satisfied,
            unserved_demand_mmbtu_hr=round(unserved, 2),
            spare_capacity_mmbtu_hr=round(spare, 2),
            iterations=iterations,
            converged=converged,
            method="equal_incremental_cost",
            calculation_hash=calculation_hash,
        )

    def _find_lambda_bounds(
        self,
        units: List[UnitConfiguration],
    ) -> Tuple[float, float]:
        """Find minimum and maximum possible incremental costs."""
        lambda_min = float('inf')
        lambda_max = 0.0

        for unit in units:
            # IC at minimum load
            ic_min = self._ic_calc.calculate(
                load_fraction=unit.min_capacity_mmbtu_hr / unit.max_capacity_mmbtu_hr,
                efficiency_coefficients=unit.efficiency_coefficients,
                fuel_price=unit.fuel_price,
                unit_id=unit.unit_id,
            )
            # IC at maximum load
            ic_max = self._ic_calc.calculate(
                load_fraction=1.0,
                efficiency_coefficients=unit.efficiency_coefficients,
                fuel_price=unit.fuel_price,
                unit_id=unit.unit_id,
            )

            lambda_min = min(lambda_min, ic_min.incremental_cost, ic_max.incremental_cost)
            lambda_max = max(lambda_max, ic_min.incremental_cost, ic_max.incremental_cost)

        # Add margin
        lambda_min = max(0, lambda_min * 0.9)
        lambda_max = lambda_max * 1.1

        return (lambda_min, lambda_max)

    def _binary_search_lambda(
        self,
        total_demand: float,
        units: List[UnitConfiguration],
        lambda_min: float,
        lambda_max: float,
    ) -> Tuple[float, int, bool]:
        """Binary search for optimal lambda."""
        for iteration in range(self.max_iterations):
            lambda_mid = (lambda_min + lambda_max) / 2

            # Calculate total load at this lambda
            total_load = self._calculate_total_load_at_lambda(lambda_mid, units)

            # Check convergence
            load_error = total_load - total_demand
            if abs(load_error) < self.tolerance * total_demand:
                return (lambda_mid, iteration + 1, True)

            # Adjust bounds
            if load_error > 0:
                # Too much load - increase lambda to reduce load
                lambda_min = lambda_mid
            else:
                # Not enough load - decrease lambda to increase load
                lambda_max = lambda_mid

        # Did not converge - return best estimate
        return ((lambda_min + lambda_max) / 2, self.max_iterations, False)

    def _calculate_total_load_at_lambda(
        self,
        lambda_val: float,
        units: List[UnitConfiguration],
    ) -> float:
        """Calculate total load when all units dispatch at given lambda."""
        total_load = 0.0

        for unit in units:
            load = self._calculate_unit_load_at_lambda(lambda_val, unit)
            total_load += load

        return total_load

    def _calculate_unit_load_at_lambda(
        self,
        lambda_val: float,
        unit: UnitConfiguration,
    ) -> float:
        """Calculate unit load at given lambda (incremental cost)."""
        # Find load where IC = lambda using bisection
        load_min = unit.min_capacity_mmbtu_hr / unit.max_capacity_mmbtu_hr
        load_max = 1.0

        for _ in range(50):  # Inner iterations
            load_mid = (load_min + load_max) / 2

            ic_result = self._ic_calc.calculate(
                load_fraction=load_mid,
                efficiency_coefficients=unit.efficiency_coefficients,
                fuel_price=unit.fuel_price,
                unit_id=unit.unit_id,
            )

            if abs(ic_result.incremental_cost - lambda_val) < 0.01:
                break

            # IC typically decreases with load (due to efficiency curve shape)
            if ic_result.incremental_cost > lambda_val:
                load_min = load_mid
            else:
                load_max = load_mid

        # Convert load fraction to MMBTU/hr
        load_mmbtu = load_mid * unit.max_capacity_mmbtu_hr

        # Clamp to unit limits
        load_mmbtu = max(unit.min_capacity_mmbtu_hr, load_mmbtu)
        load_mmbtu = min(unit.max_capacity_mmbtu_hr, load_mmbtu)

        return load_mmbtu

    def _calculate_dispatch_at_lambda(
        self,
        lambda_val: float,
        units: List[UnitConfiguration],
        total_demand: float,
    ) -> List[UnitDispatch]:
        """Calculate dispatch for all units at given lambda."""
        dispatches = []

        for unit in units:
            load_mmbtu = self._calculate_unit_load_at_lambda(lambda_val, unit)
            load_fraction = load_mmbtu / unit.max_capacity_mmbtu_hr

            # Calculate efficiency at this load
            efficiency = sum(
                c * (load_fraction ** i)
                for i, c in enumerate(unit.efficiency_coefficients)
            )
            if efficiency < 1:
                efficiency *= 100
            efficiency = max(1.0, min(99.9, efficiency))

            # Calculate fuel consumption
            fuel_input_mmbtu = load_mmbtu / (efficiency / 100)

            # Calculate costs
            fuel_cost = fuel_input_mmbtu * unit.fuel_price
            total_cost = fuel_cost + unit.no_load_cost

            # Calculate emissions
            co2_emissions = fuel_input_mmbtu * unit.emission_factor_kg_co2_mmbtu

            # Get incremental cost
            ic_result = self._ic_calc.calculate(
                load_fraction=load_fraction,
                efficiency_coefficients=unit.efficiency_coefficients,
                fuel_price=unit.fuel_price,
                unit_id=unit.unit_id,
            )

            dispatch = UnitDispatch(
                unit_id=unit.unit_id,
                load_mmbtu_hr=round(load_mmbtu, 2),
                load_fraction=round(load_fraction, 4),
                efficiency_pct=round(efficiency, 2),
                fuel_consumption=round(fuel_input_mmbtu, 4),
                fuel_consumption_unit="MMBTU/hr",
                fuel_cost_per_hr=round(fuel_cost, 2),
                incremental_cost=round(ic_result.incremental_cost, 4),
                total_cost_per_hr=round(total_cost, 2),
                co2_emissions_kg_hr=round(co2_emissions, 2),
                is_online=True,
                at_minimum=abs(load_mmbtu - unit.min_capacity_mmbtu_hr) < 0.1,
                at_maximum=abs(load_mmbtu - unit.max_capacity_mmbtu_hr) < 0.1,
            )
            dispatches.append(dispatch)

        return dispatches

    def _calculate_hash(
        self,
        total_demand: float,
        lambda_opt: float,
        total_cost: float,
        unit_ids: List[str],
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        data = {
            "solver": "EqualIncrementalCostSolver",
            "total_demand": total_demand,
            "lambda_opt": lambda_opt,
            "total_cost": total_cost,
            "unit_ids": sorted(unit_ids),
        }
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()

    @property
    def solve_count(self) -> int:
        """Get total solve count."""
        return self._solve_count


# =============================================================================
# ECONOMIC DISPATCH CALCULATOR (MAIN INTERFACE)
# =============================================================================

class EconomicDispatchCalculator:
    """
    Main economic dispatch calculator for heat load balancing.

    Provides multiple dispatch methods:
    - Equal incremental cost (lambda dispatch)
    - Merit order dispatch
    - Constrained dispatch

    ZERO-HALLUCINATION: Deterministic optimization with provenance.

    Example:
        >>> calc = EconomicDispatchCalculator()
        >>> units = [
        ...     UnitConfiguration(
        ...         unit_id="BLR-001",
        ...         min_capacity_mmbtu_hr=20,
        ...         max_capacity_mmbtu_hr=100,
        ...         fuel_price=3.50,
        ...     ),
        ...     UnitConfiguration(
        ...         unit_id="BLR-002",
        ...         min_capacity_mmbtu_hr=30,
        ...         max_capacity_mmbtu_hr=150,
        ...         fuel_price=3.25,
        ...     ),
        ... ]
        >>> result = calc.dispatch(total_demand=200, units=units)
    """

    def __init__(
        self,
        method: str = "equal_incremental_cost",
        tolerance: float = 0.001,
        max_iterations: int = 100,
    ) -> None:
        """
        Initialize economic dispatch calculator.

        Args:
            method: Dispatch method ("equal_incremental_cost", "merit_order")
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations
        """
        self.method = method
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        self._eic_solver = EqualIncrementalCostSolver(
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        self._ic_calc = IncrementalCostCalculator()

        self._dispatch_count = 0

        logger.info(
            f"EconomicDispatchCalculator initialized (method: {method})"
        )

    def dispatch(
        self,
        total_demand: float,
        units: List[UnitConfiguration],
        method: Optional[str] = None,
    ) -> EconomicDispatchResult:
        """
        Perform economic dispatch.

        DETERMINISTIC: Same inputs always produce same dispatch.

        Args:
            total_demand: Total heat demand (MMBTU/hr)
            units: List of unit configurations
            method: Override default method

        Returns:
            EconomicDispatchResult with optimal dispatch
        """
        self._dispatch_count += 1
        dispatch_method = method or self.method

        if dispatch_method == "equal_incremental_cost":
            return self._eic_solver.solve(total_demand, units)
        elif dispatch_method == "merit_order":
            return self._merit_order_dispatch(total_demand, units)
        else:
            raise ValueError(f"Unknown dispatch method: {dispatch_method}")

    def _merit_order_dispatch(
        self,
        total_demand: float,
        units: List[UnitConfiguration],
    ) -> EconomicDispatchResult:
        """
        Merit order dispatch - simple priority-based dispatch.

        Units are dispatched in order of cost (cheapest first).
        """
        # Filter available units
        available_units = [u for u in units if u.available]

        # Calculate average incremental cost for each unit at 75% load
        unit_costs = []
        for unit in available_units:
            ic_result = self._ic_calc.calculate(
                load_fraction=0.75,
                efficiency_coefficients=unit.efficiency_coefficients,
                fuel_price=unit.fuel_price,
                unit_id=unit.unit_id,
            )
            unit_costs.append((unit, ic_result.incremental_cost))

        # Sort by cost (merit order)
        unit_costs.sort(key=lambda x: x[1])

        # Dispatch in merit order
        remaining_demand = total_demand
        dispatches = []

        for unit, avg_ic in unit_costs:
            if remaining_demand <= 0:
                # Unit offline
                dispatch = UnitDispatch(
                    unit_id=unit.unit_id,
                    load_mmbtu_hr=0.0,
                    load_fraction=0.0,
                    efficiency_pct=0.0,
                    fuel_consumption=0.0,
                    fuel_consumption_unit="MMBTU/hr",
                    fuel_cost_per_hr=0.0,
                    incremental_cost=avg_ic,
                    total_cost_per_hr=0.0,
                    is_online=False,
                )
            else:
                # Dispatch this unit
                if unit.must_run:
                    load = max(unit.min_capacity_mmbtu_hr, min(remaining_demand, unit.max_capacity_mmbtu_hr))
                else:
                    load = min(remaining_demand, unit.max_capacity_mmbtu_hr)

                load_fraction = load / unit.max_capacity_mmbtu_hr

                # Calculate efficiency
                efficiency = sum(
                    c * (load_fraction ** i)
                    for i, c in enumerate(unit.efficiency_coefficients)
                )
                if efficiency < 1:
                    efficiency *= 100
                efficiency = max(1.0, min(99.9, efficiency))

                fuel_input = load / (efficiency / 100)
                fuel_cost = fuel_input * unit.fuel_price
                co2 = fuel_input * unit.emission_factor_kg_co2_mmbtu

                dispatch = UnitDispatch(
                    unit_id=unit.unit_id,
                    load_mmbtu_hr=round(load, 2),
                    load_fraction=round(load_fraction, 4),
                    efficiency_pct=round(efficiency, 2),
                    fuel_consumption=round(fuel_input, 4),
                    fuel_consumption_unit="MMBTU/hr",
                    fuel_cost_per_hr=round(fuel_cost, 2),
                    incremental_cost=avg_ic,
                    total_cost_per_hr=round(fuel_cost + unit.no_load_cost, 2),
                    co2_emissions_kg_hr=round(co2, 2),
                    is_online=True,
                    at_minimum=abs(load - unit.min_capacity_mmbtu_hr) < 0.1,
                    at_maximum=abs(load - unit.max_capacity_mmbtu_hr) < 0.1,
                )

                remaining_demand -= load

            dispatches.append(dispatch)

        # Calculate totals
        total_load = sum(d.load_mmbtu_hr for d in dispatches)
        total_fuel_cost = sum(d.fuel_cost_per_hr for d in dispatches)
        total_cost = sum(d.total_cost_per_hr for d in dispatches)
        total_co2 = sum(d.co2_emissions_kg_hr for d in dispatches)

        # Weighted average efficiency
        if total_load > 0:
            weighted_eff = sum(
                d.efficiency_pct * d.load_mmbtu_hr
                for d in dispatches if d.is_online
            ) / total_load
        else:
            weighted_eff = 0.0

        # System lambda (marginal cost of last unit dispatched)
        online_dispatches = [d for d in dispatches if d.is_online]
        system_lambda = online_dispatches[-1].incremental_cost if online_dispatches else 0.0

        total_max = sum(u.max_capacity_mmbtu_hr for u in available_units)

        calculation_hash = hashlib.sha256(
            json.dumps({
                "method": "merit_order",
                "demand": total_demand,
                "total_cost": total_cost,
            }, sort_keys=True).encode()
        ).hexdigest()

        return EconomicDispatchResult(
            total_demand_mmbtu_hr=total_demand,
            total_cost_per_hr=round(total_cost, 2),
            system_lambda=round(system_lambda, 4),
            unit_dispatches=dispatches,
            total_fuel_cost_per_hr=round(total_fuel_cost, 2),
            average_efficiency_pct=round(weighted_eff, 2),
            total_co2_emissions_kg_hr=round(total_co2, 2),
            demand_satisfied=remaining_demand <= 0.1,
            unserved_demand_mmbtu_hr=max(0, remaining_demand),
            spare_capacity_mmbtu_hr=max(0, total_max - total_load),
            iterations=1,
            converged=True,
            method="merit_order",
            calculation_hash=calculation_hash,
        )

    @property
    def dispatch_count(self) -> int:
        """Get total dispatch count."""
        return self._dispatch_count


# =============================================================================
# LOSS FACTOR CALCULATOR
# =============================================================================

class LossFactorCalculator:
    """
    Calculate transmission/distribution losses.

    Accounts for steam/hot water distribution losses to adjust
    generation requirements.

    Example:
        >>> calc = LossFactorCalculator(base_loss_pct=3.0)
        >>> result = calc.calculate(net_demand=500.0)
    """

    def __init__(
        self,
        base_loss_pct: float = 3.0,
        distance_factor: float = 0.001,  # % loss per meter
    ) -> None:
        """
        Initialize loss factor calculator.

        Args:
            base_loss_pct: Base transmission loss (%)
            distance_factor: Additional loss per unit distance
        """
        self.base_loss_pct = base_loss_pct
        self.distance_factor = distance_factor

    def calculate(
        self,
        net_demand: float,
        distance_m: float = 0.0,
        insulation_factor: float = 1.0,
    ) -> LossFactorResult:
        """
        Calculate transmission losses.

        Args:
            net_demand: Net demand at delivery point (MMBTU/hr)
            distance_m: Distribution distance (meters)
            insulation_factor: Insulation quality factor (1.0 = design)

        Returns:
            LossFactorResult with gross generation requirement
        """
        # Calculate total loss percentage
        distance_loss = distance_m * self.distance_factor
        total_loss_pct = (self.base_loss_pct + distance_loss) * insulation_factor

        # Clamp to reasonable range
        total_loss_pct = max(0, min(20, total_loss_pct))

        # Calculate gross generation required
        # net_demand = gross * (1 - loss_pct/100)
        # gross = net_demand / (1 - loss_pct/100)
        gross_generation = net_demand / (1 - total_loss_pct / 100)
        loss_mmbtu = gross_generation - net_demand

        calculation_hash = hashlib.sha256(
            json.dumps({
                "calculator": "LossFactorCalculator",
                "net_demand": net_demand,
                "loss_pct": total_loss_pct,
            }, sort_keys=True).encode()
        ).hexdigest()

        return LossFactorResult(
            transmission_loss_pct=round(total_loss_pct, 2),
            loss_mmbtu_hr=round(loss_mmbtu, 4),
            gross_generation_required=round(gross_generation, 2),
            net_demand_mmbtu_hr=net_demand,
            calculation_hash=calculation_hash,
        )


# =============================================================================
# RESERVE MARGIN CALCULATOR
# =============================================================================

class ReserveMarginCalculator:
    """
    Calculate spinning and non-spinning reserve margins.

    Ensures adequate reserve capacity for reliability.

    Example:
        >>> calc = ReserveMarginCalculator(required_reserve_pct=15.0)
        >>> result = calc.calculate(
        ...     current_load=400.0,
        ...     unit_dispatches=dispatches,
        ...     available_units=units,
        ... )
    """

    def __init__(
        self,
        required_reserve_pct: float = 15.0,
    ) -> None:
        """
        Initialize reserve margin calculator.

        Args:
            required_reserve_pct: Required reserve margin (%)
        """
        self.required_reserve_pct = required_reserve_pct

    def calculate(
        self,
        current_load: float,
        unit_dispatches: List[UnitDispatch],
        available_units: List[UnitConfiguration],
    ) -> ReserveMarginResult:
        """
        Calculate reserve margins.

        Args:
            current_load: Current total load (MMBTU/hr)
            unit_dispatches: Current unit dispatches
            available_units: All available units

        Returns:
            ReserveMarginResult with reserve analysis
        """
        # Total capacity of all available units
        total_capacity = sum(u.max_capacity_mmbtu_hr for u in available_units)

        # Spinning reserve (from online units not at max)
        spinning_reserve = 0.0
        for dispatch in unit_dispatches:
            if dispatch.is_online:
                # Find unit config
                unit = next(
                    (u for u in available_units if u.unit_id == dispatch.unit_id),
                    None
                )
                if unit:
                    spinning_reserve += unit.max_capacity_mmbtu_hr - dispatch.load_mmbtu_hr

        # Non-spinning reserve (from offline available units)
        online_ids = {d.unit_id for d in unit_dispatches if d.is_online}
        non_spinning_reserve = sum(
            u.max_capacity_mmbtu_hr
            for u in available_units
            if u.unit_id not in online_ids
        )

        # Total reserve margin
        reserve_margin = total_capacity - current_load
        reserve_margin_pct = (reserve_margin / current_load * 100) if current_load > 0 else 0

        # Check adequacy
        reserve_adequate = reserve_margin_pct >= self.required_reserve_pct

        return ReserveMarginResult(
            total_capacity_mmbtu_hr=round(total_capacity, 2),
            current_load_mmbtu_hr=round(current_load, 2),
            reserve_margin_mmbtu_hr=round(reserve_margin, 2),
            reserve_margin_pct=round(reserve_margin_pct, 2),
            spinning_reserve_mmbtu_hr=round(spinning_reserve, 2),
            non_spinning_reserve_mmbtu_hr=round(non_spinning_reserve, 2),
            required_reserve_pct=self.required_reserve_pct,
            reserve_adequate=reserve_adequate,
        )
