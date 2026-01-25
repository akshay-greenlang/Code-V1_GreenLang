"""Zero-hallucination formulas for GL-023 Heat Load Balancer.

Implements optimal load allocation using economic dispatch principles.
All calculations are deterministic with SHA-256 provenance tracking.
"""
import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class UnitLoadPoint:
    """Operating point for a unit."""
    unit_id: str
    load_mw: float
    efficiency_pct: float
    incremental_cost: float
    fuel_consumption_mw: float
    emissions_kg_hr: float


def calculate_efficiency_at_load(
    load_mw: float,
    min_load_mw: float,
    max_load_mw: float,
    curve_a: float,
    curve_b: float,
    curve_c: float,
    base_efficiency: float = 80.0
) -> float:
    """
    Calculate equipment efficiency at a given load.

    Efficiency curve: η(L) = a + b*(L/L_max) + c*(L/L_max)²

    Typical boiler efficiency peaks at 70-90% load.
    """
    if load_mw <= 0 or max_load_mw <= 0:
        return 0.0

    if load_mw < min_load_mw:
        return 0.0  # Below minimum stable combustion

    # Normalize load (0 to 1)
    load_fraction = min(load_mw / max_load_mw, 1.0)

    # If no curve coefficients, use typical boiler curve
    if curve_a == 0 and curve_b == 0 and curve_c == 0:
        # Typical curve: peaks around 75% load
        # η = η_base * (0.9 + 0.2*L - 0.1*L²)
        curve_a = base_efficiency * 0.9
        curve_b = base_efficiency * 0.2
        curve_c = -base_efficiency * 0.1

    efficiency = curve_a + curve_b * load_fraction + curve_c * load_fraction**2
    return max(0.0, min(100.0, round(efficiency, 2)))


def calculate_fuel_consumption(
    thermal_output_mw: float,
    efficiency_pct: float
) -> float:
    """
    Calculate fuel consumption for given thermal output.

    Fuel input = Thermal output / (Efficiency/100)

    Returns fuel consumption in MW (thermal input).
    """
    if efficiency_pct <= 0:
        return 0.0

    fuel_mw = thermal_output_mw / (efficiency_pct / 100.0)
    return round(fuel_mw, 4)


def calculate_hourly_cost(
    fuel_consumption_mw: float,
    fuel_cost_per_mwh: float,
    maintenance_cost_per_mwh: float,
    thermal_output_mw: float
) -> float:
    """
    Calculate hourly operating cost.

    Cost = (Fuel_MW × Fuel_price) + (Output_MW × Maint_cost)
    """
    fuel_cost = fuel_consumption_mw * fuel_cost_per_mwh
    maint_cost = thermal_output_mw * maintenance_cost_per_mwh
    return round(fuel_cost + maint_cost, 2)


def calculate_incremental_cost(
    load_mw: float,
    delta_mw: float,
    min_load: float,
    max_load: float,
    curve_a: float,
    curve_b: float,
    curve_c: float,
    fuel_cost_per_mwh: float,
    base_efficiency: float = 80.0
) -> float:
    """
    Calculate incremental cost (marginal cost) at current load.

    IC = d(Cost)/d(Load) = Fuel_price / η(L) × (1 - L × dη/dL / η)

    This is used for economic dispatch - units with lowest IC
    should be loaded first.
    """
    if load_mw < min_load or load_mw > max_load:
        return float('inf')

    eta = calculate_efficiency_at_load(
        load_mw, min_load, max_load,
        curve_a, curve_b, curve_c, base_efficiency
    )

    if eta <= 0:
        return float('inf')

    # For small delta, incremental cost ≈ fuel_cost / efficiency
    ic = fuel_cost_per_mwh / (eta / 100.0)
    return round(ic, 4)


def calculate_emissions(
    fuel_consumption_mw: float,
    emissions_factor_kg_mwh: float
) -> float:
    """
    Calculate CO2 emissions per hour.

    Emissions = Fuel_consumption × Emission_factor
    """
    return round(fuel_consumption_mw * emissions_factor_kg_mwh, 2)


def economic_dispatch_merit_order(
    units: List[Dict],
    total_demand_mw: float,
    carbon_price: float = 0.0
) -> List[Tuple[str, float]]:
    """
    Perform economic dispatch using merit order method.

    Algorithm:
    1. Calculate incremental cost for each unit at midpoint load
    2. Sort units by incremental cost (lowest first)
    3. Load units in order until demand is met
    4. Respect min/max load constraints

    Returns list of (unit_id, allocated_load) tuples.
    """
    allocations = []
    remaining_demand = total_demand_mw

    # Calculate merit order (incremental cost at 75% load)
    unit_costs = []
    for unit in units:
        if not unit.get('is_available', True):
            continue

        mid_load = (unit['min_load_mw'] + unit['max_load_mw']) / 2
        ic = calculate_incremental_cost(
            load_mw=mid_load,
            delta_mw=0.1,
            min_load=unit['min_load_mw'],
            max_load=unit['max_load_mw'],
            curve_a=unit.get('efficiency_curve_a', 0),
            curve_b=unit.get('efficiency_curve_b', 0),
            curve_c=unit.get('efficiency_curve_c', 0),
            fuel_cost_per_mwh=unit['fuel_cost_per_mwh'],
            base_efficiency=unit.get('current_efficiency_pct', 80)
        )

        # Add carbon cost
        carbon_cost = (unit.get('emissions_factor_kg_co2_mwh', 200) / 1000) * carbon_price
        total_ic = ic + carbon_cost

        unit_costs.append((unit['unit_id'], total_ic, unit))

    # Sort by incremental cost
    unit_costs.sort(key=lambda x: x[1])

    # Allocate load in merit order
    for unit_id, ic, unit in unit_costs:
        if remaining_demand <= 0:
            # Turn off unit if not needed
            allocations.append((unit_id, 0.0))
            continue

        min_load = unit['min_load_mw']
        max_load = unit['max_load_mw']

        if remaining_demand >= min_load:
            # Load this unit
            load = min(remaining_demand, max_load)
            allocations.append((unit_id, load))
            remaining_demand -= load
        else:
            # Demand too low for minimum load - skip or min load
            allocations.append((unit_id, 0.0))

    return allocations


def calculate_fleet_efficiency(
    allocations: List[Tuple[str, float]],
    units: List[Dict]
) -> float:
    """
    Calculate weighted average efficiency of the fleet.

    Fleet_η = Σ(Load_i × η_i) / Σ(Load_i)
    """
    total_load = 0.0
    weighted_efficiency = 0.0

    unit_lookup = {u['unit_id']: u for u in units}

    for unit_id, load in allocations:
        if load <= 0 or unit_id not in unit_lookup:
            continue

        unit = unit_lookup[unit_id]
        eta = calculate_efficiency_at_load(
            load,
            unit['min_load_mw'],
            unit['max_load_mw'],
            unit.get('efficiency_curve_a', 0),
            unit.get('efficiency_curve_b', 0),
            unit.get('efficiency_curve_c', 0),
            unit.get('current_efficiency_pct', 80)
        )

        total_load += load
        weighted_efficiency += load * eta

    if total_load <= 0:
        return 0.0

    return round(weighted_efficiency / total_load, 2)


def calculate_equal_loading(
    units: List[Dict],
    total_demand_mw: float
) -> Tuple[float, float]:
    """
    Calculate efficiency and cost with equal loading (baseline comparison).

    Equal loading distributes demand evenly across available units.
    This is the naive approach that optimized dispatch improves upon.
    """
    available_units = [u for u in units if u.get('is_available', True)]
    n_units = len(available_units)

    if n_units == 0:
        return 0.0, 0.0

    load_per_unit = total_demand_mw / n_units

    total_efficiency_weighted = 0.0
    total_cost = 0.0
    total_load = 0.0

    for unit in available_units:
        # Constrain to min/max
        actual_load = max(unit['min_load_mw'], min(load_per_unit, unit['max_load_mw']))

        eta = calculate_efficiency_at_load(
            actual_load,
            unit['min_load_mw'],
            unit['max_load_mw'],
            unit.get('efficiency_curve_a', 0),
            unit.get('efficiency_curve_b', 0),
            unit.get('efficiency_curve_c', 0),
            unit.get('current_efficiency_pct', 80)
        )

        fuel = calculate_fuel_consumption(actual_load, eta)
        cost = calculate_hourly_cost(
            fuel,
            unit['fuel_cost_per_mwh'],
            unit.get('maintenance_cost_per_mwh', 0),
            actual_load
        )

        total_efficiency_weighted += actual_load * eta
        total_cost += cost
        total_load += actual_load

    avg_efficiency = total_efficiency_weighted / total_load if total_load > 0 else 0
    return round(avg_efficiency, 2), round(total_cost, 2)


def generate_calculation_hash(inputs: Dict, outputs: Dict) -> str:
    """Generate SHA-256 hash for calculation provenance."""
    data = {
        "inputs": inputs,
        "outputs": outputs,
        "formula_version": "1.0.0",
        "method": "economic_dispatch_merit_order"
    }
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()
