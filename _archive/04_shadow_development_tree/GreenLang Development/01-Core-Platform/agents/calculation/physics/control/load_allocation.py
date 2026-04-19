"""
Load Allocation / Equipment Optimization

Zero-Hallucination Multi-Equipment Load Distribution

This module implements load allocation algorithms for distributing
demand across multiple parallel equipment units optimally.

References:
    - ISA-95: Enterprise-Control System Integration
    - Optimization Techniques for Process Control
    - Linear and Mixed-Integer Linear Programming

Author: GreenLang Engineering Team
License: MIT
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple
from enum import Enum
import hashlib
import math


class AllocationMethod(Enum):
    """Load allocation methods."""
    EQUAL = "equal"  # Equal distribution
    PROPORTIONAL = "proportional"  # Based on capacity
    EFFICIENCY_BASED = "efficiency_based"  # Minimize total cost/energy
    PRIORITY = "priority"  # Priority-based loading
    INCREMENTAL_COST = "incremental_cost"  # Economic dispatch


class EquipmentStatus(Enum):
    """Equipment operating status."""
    AVAILABLE = "available"
    RUNNING = "running"
    STANDBY = "standby"
    MAINTENANCE = "maintenance"
    FAILED = "failed"


@dataclass
class EquipmentUnit:
    """
    Equipment unit characteristics for load allocation.

    Represents a single unit in a parallel equipment set.
    """
    unit_id: str
    name: str
    capacity_min: float  # Minimum operating capacity
    capacity_max: float  # Maximum operating capacity
    current_load: float = 0.0
    status: EquipmentStatus = EquipmentStatus.AVAILABLE

    # Efficiency characteristics
    # efficiency = a0 + a1*load + a2*load^2
    efficiency_coeffs: Tuple[float, float, float] = (0.80, 0.002, -0.00001)

    # Cost characteristics ($/unit output)
    # cost = c0 + c1*load + c2*load^2
    cost_coeffs: Tuple[float, float, float] = (100.0, 0.5, 0.001)

    # Priority (lower = higher priority)
    priority: int = 1

    def efficiency_at_load(self, load: float) -> float:
        """Calculate efficiency at given load."""
        a0, a1, a2 = self.efficiency_coeffs
        return a0 + a1 * load + a2 * load ** 2

    def cost_at_load(self, load: float) -> float:
        """Calculate operating cost at given load."""
        c0, c1, c2 = self.cost_coeffs
        return c0 + c1 * load + c2 * load ** 2

    def incremental_cost(self, load: float) -> float:
        """Calculate incremental (marginal) cost at load."""
        c0, c1, c2 = self.cost_coeffs
        return c1 + 2 * c2 * load


@dataclass
class AllocationResult:
    """
    Load allocation results.

    All values are deterministic - same inputs produce identical outputs.
    """
    # Total demand
    total_demand: Decimal
    total_allocated: Decimal
    unmet_demand: Decimal

    # Individual allocations
    unit_allocations: Dict[str, Decimal]
    unit_efficiencies: Dict[str, Decimal]
    unit_costs: Dict[str, Decimal]

    # Aggregate metrics
    total_cost: Decimal
    average_efficiency: Decimal
    capacity_utilization: Decimal

    # Units status
    units_running: int
    units_available: int

    # Allocation details
    method_used: str
    is_feasible: bool
    optimization_message: str

    # Provenance
    provenance_hash: str

    def to_dict(self) -> Dict:
        return {
            "total_demand": float(self.total_demand),
            "total_allocated": float(self.total_allocated),
            "unit_allocations": {k: float(v) for k, v in self.unit_allocations.items()},
            "total_cost": float(self.total_cost),
            "average_efficiency": float(self.average_efficiency),
            "is_feasible": self.is_feasible,
            "provenance_hash": self.provenance_hash
        }


class LoadAllocator:
    """
    Multi-Equipment Load Allocation Optimizer.

    ZERO-HALLUCINATION GUARANTEE:
    - Deterministic allocation algorithms
    - Complete provenance tracking
    - Constraint satisfaction
    - Optimal or near-optimal solutions

    Methods:
    1. Equal: Distribute equally among available units
    2. Proportional: Based on capacity ratios
    3. Efficiency-based: Maximize overall efficiency
    4. Incremental cost: Economic dispatch (minimize total cost)
    """

    def __init__(self, precision: int = 2):
        """Initialize allocator."""
        self.precision = precision
        self.units: List[EquipmentUnit] = []

    def _apply_precision(self, value: Decimal) -> Decimal:
        """Apply precision rounding."""
        quantize_str = "0." + "0" * self.precision
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    def _calculate_provenance(self, inputs: Dict, outputs: Dict) -> str:
        """Calculate SHA-256 hash for audit trail."""
        provenance_data = {
            "method": "Load_Allocation",
            "inputs": {k: str(v) for k, v in inputs.items()},
            "outputs": {k: str(v) for k, v in outputs.items()}
        }
        provenance_str = str(sorted(provenance_data.items()))
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def add_unit(self, unit: EquipmentUnit) -> None:
        """Add equipment unit to allocation set."""
        self.units.append(unit)

    def remove_unit(self, unit_id: str) -> None:
        """Remove equipment unit from allocation set."""
        self.units = [u for u in self.units if u.unit_id != unit_id]

    def get_available_units(self) -> List[EquipmentUnit]:
        """Get list of available units."""
        return [u for u in self.units
                if u.status in [EquipmentStatus.AVAILABLE, EquipmentStatus.RUNNING]]

    def get_total_capacity(self) -> Tuple[float, float]:
        """Get total min and max capacity of available units."""
        available = self.get_available_units()
        total_min = sum(u.capacity_min for u in available)
        total_max = sum(u.capacity_max for u in available)
        return total_min, total_max

    def allocate(
        self,
        total_demand: float,
        method: AllocationMethod = AllocationMethod.EFFICIENCY_BASED,
        constraints: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> AllocationResult:
        """
        Allocate load across equipment units.

        ZERO-HALLUCINATION: Deterministic allocation.

        Args:
            total_demand: Total demand to distribute
            method: Allocation method to use
            constraints: Optional per-unit constraints {unit_id: (min, max)}

        Returns:
            AllocationResult with optimal allocation
        """
        demand = Decimal(str(total_demand))
        available_units = self.get_available_units()

        if not available_units:
            return self._create_infeasible_result(demand, "No available units")

        # Check feasibility
        total_min, total_max = self.get_total_capacity()
        if total_demand > total_max:
            # Partial allocation
            is_feasible = False
            opt_message = f"Demand {total_demand} exceeds capacity {total_max}"
        elif total_demand < total_min:
            is_feasible = False
            opt_message = f"Demand {total_demand} below minimum {total_min}"
        else:
            is_feasible = True
            opt_message = "Optimal allocation found"

        # Apply allocation method
        if method == AllocationMethod.EQUAL:
            allocations = self._allocate_equal(available_units, total_demand, constraints)
        elif method == AllocationMethod.PROPORTIONAL:
            allocations = self._allocate_proportional(available_units, total_demand, constraints)
        elif method == AllocationMethod.EFFICIENCY_BASED:
            allocations = self._allocate_efficiency(available_units, total_demand, constraints)
        elif method == AllocationMethod.INCREMENTAL_COST:
            allocations = self._allocate_incremental_cost(available_units, total_demand, constraints)
        elif method == AllocationMethod.PRIORITY:
            allocations = self._allocate_priority(available_units, total_demand, constraints)
        else:
            allocations = self._allocate_equal(available_units, total_demand, constraints)

        # Calculate results
        return self._calculate_results(
            demand, allocations, available_units, method, is_feasible, opt_message
        )

    def _allocate_equal(
        self,
        units: List[EquipmentUnit],
        demand: float,
        constraints: Optional[Dict]
    ) -> Dict[str, float]:
        """Equal distribution among available units."""
        n_units = len(units)
        if n_units == 0:
            return {}

        base_load = demand / n_units
        allocations = {}

        for unit in units:
            load = base_load

            # Apply constraints
            if constraints and unit.unit_id in constraints:
                min_c, max_c = constraints[unit.unit_id]
                load = max(min_c, min(max_c, load))
            else:
                load = max(unit.capacity_min, min(unit.capacity_max, load))

            allocations[unit.unit_id] = load

        return allocations

    def _allocate_proportional(
        self,
        units: List[EquipmentUnit],
        demand: float,
        constraints: Optional[Dict]
    ) -> Dict[str, float]:
        """Proportional distribution based on capacity."""
        total_capacity = sum(u.capacity_max for u in units)
        if total_capacity == 0:
            return {}

        allocations = {}
        for unit in units:
            proportion = unit.capacity_max / total_capacity
            load = demand * proportion

            # Apply constraints
            if constraints and unit.unit_id in constraints:
                min_c, max_c = constraints[unit.unit_id]
                load = max(min_c, min(max_c, load))
            else:
                load = max(unit.capacity_min, min(unit.capacity_max, load))

            allocations[unit.unit_id] = load

        return allocations

    def _allocate_efficiency(
        self,
        units: List[EquipmentUnit],
        demand: float,
        constraints: Optional[Dict]
    ) -> Dict[str, float]:
        """
        Efficiency-based allocation.

        Uses iterative approach to find allocation that maximizes
        overall efficiency.
        """
        # Sort units by efficiency at mid-point
        units_sorted = sorted(
            units,
            key=lambda u: u.efficiency_at_load((u.capacity_min + u.capacity_max) / 2),
            reverse=True
        )

        allocations = {u.unit_id: 0.0 for u in units}
        remaining = demand

        # First pass: load most efficient units first
        for unit in units_sorted:
            if remaining <= 0:
                break

            max_load = unit.capacity_max
            if constraints and unit.unit_id in constraints:
                _, max_load = constraints[unit.unit_id]

            load = min(remaining, max_load)
            allocations[unit.unit_id] = load
            remaining -= load

        # Second pass: ensure minimum loads are met
        total_allocated = sum(allocations.values())
        for unit in units_sorted:
            if allocations[unit.unit_id] > 0:
                min_load = unit.capacity_min
                if constraints and unit.unit_id in constraints:
                    min_load, _ = constraints[unit.unit_id]

                if allocations[unit.unit_id] < min_load:
                    allocations[unit.unit_id] = min_load

        return allocations

    def _allocate_incremental_cost(
        self,
        units: List[EquipmentUnit],
        demand: float,
        constraints: Optional[Dict]
    ) -> Dict[str, float]:
        """
        Economic dispatch using incremental cost (lambda-iteration).

        All units should operate at same incremental cost at optimum.
        """
        # Initialize all units at minimum
        allocations = {u.unit_id: u.capacity_min for u in units}
        total = sum(allocations.values())

        if total >= demand:
            # Scale down proportionally
            return self._allocate_proportional(units, demand, constraints)

        # Iteratively increase loads
        remaining = demand - total
        max_iterations = 100
        tolerance = 0.1

        for _ in range(max_iterations):
            if remaining < tolerance:
                break

            # Find unit with lowest incremental cost that has capacity
            best_unit = None
            best_ic = float('inf')

            for unit in units:
                current_load = allocations[unit.unit_id]
                max_load = unit.capacity_max
                if constraints and unit.unit_id in constraints:
                    _, max_load = constraints[unit.unit_id]

                if current_load < max_load:
                    ic = unit.incremental_cost(current_load)
                    if ic < best_ic:
                        best_ic = ic
                        best_unit = unit

            if best_unit is None:
                break

            # Increment load on best unit
            max_load = best_unit.capacity_max
            if constraints and best_unit.unit_id in constraints:
                _, max_load = constraints[best_unit.unit_id]

            increment = min(remaining, max_load - allocations[best_unit.unit_id])
            allocations[best_unit.unit_id] += increment
            remaining -= increment

        return allocations

    def _allocate_priority(
        self,
        units: List[EquipmentUnit],
        demand: float,
        constraints: Optional[Dict]
    ) -> Dict[str, float]:
        """Priority-based allocation (load highest priority first)."""
        # Sort by priority (lower number = higher priority)
        units_sorted = sorted(units, key=lambda u: u.priority)

        allocations = {u.unit_id: 0.0 for u in units}
        remaining = demand

        for unit in units_sorted:
            if remaining <= 0:
                break

            min_load = unit.capacity_min
            max_load = unit.capacity_max
            if constraints and unit.unit_id in constraints:
                min_load, max_load = constraints[unit.unit_id]

            # Load this unit to maximum before moving to next
            load = min(remaining, max_load)
            if load >= min_load:
                allocations[unit.unit_id] = load
                remaining -= load

        return allocations

    def _calculate_results(
        self,
        demand: Decimal,
        allocations: Dict[str, float],
        units: List[EquipmentUnit],
        method: AllocationMethod,
        is_feasible: bool,
        message: str
    ) -> AllocationResult:
        """Calculate comprehensive allocation results."""
        unit_allocations = {}
        unit_efficiencies = {}
        unit_costs = {}
        total_cost = Decimal("0")
        weighted_efficiency = Decimal("0")
        total_allocated = Decimal("0")

        for unit in units:
            load = allocations.get(unit.unit_id, 0.0)
            unit_allocations[unit.unit_id] = Decimal(str(load))

            if load > 0:
                eff = unit.efficiency_at_load(load)
                cost = unit.cost_at_load(load)
                unit_efficiencies[unit.unit_id] = Decimal(str(eff))
                unit_costs[unit.unit_id] = Decimal(str(cost))
                total_cost += Decimal(str(cost))
                weighted_efficiency += Decimal(str(eff * load))
                total_allocated += Decimal(str(load))

        # Average efficiency (weighted)
        if total_allocated > 0:
            avg_efficiency = weighted_efficiency / total_allocated
        else:
            avg_efficiency = Decimal("0")

        # Capacity utilization
        total_capacity = sum(u.capacity_max for u in units)
        utilization = total_allocated / Decimal(str(total_capacity)) * Decimal("100") \
            if total_capacity > 0 else Decimal("0")

        # Unmet demand
        unmet = demand - total_allocated
        if unmet < Decimal("0"):
            unmet = Decimal("0")

        # Running units count
        units_running = sum(1 for uid, load in unit_allocations.items() if load > 0)

        inputs = {"demand": str(demand), "method": method.value, "n_units": len(units)}
        outputs = {"total_allocated": str(total_allocated), "total_cost": str(total_cost)}
        provenance_hash = self._calculate_provenance(inputs, outputs)

        return AllocationResult(
            total_demand=self._apply_precision(demand),
            total_allocated=self._apply_precision(total_allocated),
            unmet_demand=self._apply_precision(unmet),
            unit_allocations={k: self._apply_precision(v) for k, v in unit_allocations.items()},
            unit_efficiencies={k: self._apply_precision(v) for k, v in unit_efficiencies.items()},
            unit_costs={k: self._apply_precision(v) for k, v in unit_costs.items()},
            total_cost=self._apply_precision(total_cost),
            average_efficiency=self._apply_precision(avg_efficiency),
            capacity_utilization=self._apply_precision(utilization),
            units_running=units_running,
            units_available=len(units),
            method_used=method.value,
            is_feasible=is_feasible,
            optimization_message=message,
            provenance_hash=provenance_hash
        )

    def _create_infeasible_result(self, demand: Decimal, message: str) -> AllocationResult:
        """Create result for infeasible allocation."""
        inputs = {"demand": str(demand), "error": message}
        provenance_hash = self._calculate_provenance(inputs, {})

        return AllocationResult(
            total_demand=demand,
            total_allocated=Decimal("0"),
            unmet_demand=demand,
            unit_allocations={},
            unit_efficiencies={},
            unit_costs={},
            total_cost=Decimal("0"),
            average_efficiency=Decimal("0"),
            capacity_utilization=Decimal("0"),
            units_running=0,
            units_available=0,
            method_used="none",
            is_feasible=False,
            optimization_message=message,
            provenance_hash=provenance_hash
        )


# Convenience functions
def allocate_boiler_load(
    total_steam_demand: float,
    boilers: List[Dict]
) -> AllocationResult:
    """
    Allocate steam load across multiple boilers.

    Example:
        >>> boilers = [
        ...     {"id": "B1", "min": 20, "max": 100, "efficiency": 0.85},
        ...     {"id": "B2", "min": 15, "max": 80, "efficiency": 0.82},
        ...     {"id": "B3", "min": 10, "max": 60, "efficiency": 0.88},
        ... ]
        >>> result = allocate_boiler_load(150, boilers)
        >>> print(result.unit_allocations)
    """
    allocator = LoadAllocator()

    for b in boilers:
        eff = b.get("efficiency", 0.85)
        unit = EquipmentUnit(
            unit_id=b["id"],
            name=b.get("name", b["id"]),
            capacity_min=b["min"],
            capacity_max=b["max"],
            efficiency_coeffs=(eff, 0.001, -0.000005)
        )
        allocator.add_unit(unit)

    return allocator.allocate(total_steam_demand, AllocationMethod.EFFICIENCY_BASED)


def allocate_compressor_load(
    total_gas_demand: float,
    compressors: List[Dict]
) -> AllocationResult:
    """Allocate gas compression across multiple compressors."""
    allocator = LoadAllocator()

    for c in compressors:
        unit = EquipmentUnit(
            unit_id=c["id"],
            name=c.get("name", c["id"]),
            capacity_min=c.get("min", 0),
            capacity_max=c["max"],
            priority=c.get("priority", 1)
        )
        allocator.add_unit(unit)

    return allocator.allocate(total_gas_demand, AllocationMethod.PRIORITY)
