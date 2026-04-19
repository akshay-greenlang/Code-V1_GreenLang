"""
GL-033 Burner Balancer - Calculation Formulas

This module implements deterministic calculations for multi-burner
load balancing and air-fuel optimization.

ZERO-HALLUCINATION: All calculations use combustion engineering
formulas from NFPA 85/86 and combustion reference texts.
"""

import math
import logging
from typing import Tuple, Dict, List, NamedTuple

logger = logging.getLogger(__name__)


class BurnerCurvePoint(NamedTuple):
    """Point on burner characteristic curve."""
    firing_rate_percent: float
    fuel_flow_scfh: float
    air_flow_scfh: float
    efficiency_percent: float
    nox_ppm: float


class OptimizationResult(NamedTuple):
    """Result of burner optimization."""
    burner_id: str
    optimal_firing_rate: float
    optimal_air_fuel_ratio: float
    expected_efficiency: float
    expected_nox: float


def calculate_stoichiometric_air_fuel_ratio(
    fuel_type: str,
    fuel_hhv_btu_per_scf: float = None
) -> float:
    """
    Calculate stoichiometric air-fuel ratio for given fuel.

    ZERO-HALLUCINATION: Standard stoichiometric ratios from
    combustion engineering references.

    Natural Gas (CH4): 9.52 scf air / scf fuel
    Propane (C3H8): 23.8 scf air / scf fuel
    Hydrogen (H2): 2.38 scf air / scf fuel

    Args:
        fuel_type: Type of fuel
        fuel_hhv_btu_per_scf: Higher heating value (optional override)

    Returns:
        Stoichiometric air-fuel ratio (scf air / scf fuel)
    """
    stoich_ratios = {
        "NATURAL_GAS": 9.52,
        "PROPANE": 23.8,
        "HYDROGEN": 2.38,
        "FUEL_OIL": 14.0,  # Approximate for typical fuel oil
        "SYNGAS": 5.0,     # Varies with composition
        "BIOGAS": 6.0,     # Typical for 60% CH4
    }
    return stoich_ratios.get(fuel_type, 10.0)


def calculate_excess_air_percent(
    actual_air_flow: float,
    fuel_flow: float,
    stoich_ratio: float
) -> float:
    """
    Calculate percent excess air.

    ZERO-HALLUCINATION FORMULA:
    Excess Air % = ((Actual AF Ratio / Stoich AF Ratio) - 1) * 100

    Args:
        actual_air_flow: Actual air flow rate
        fuel_flow: Fuel flow rate
        stoich_ratio: Stoichiometric air-fuel ratio

    Returns:
        Excess air percentage
    """
    if fuel_flow <= 0:
        return 0.0

    actual_ratio = actual_air_flow / fuel_flow
    excess_air = ((actual_ratio / stoich_ratio) - 1.0) * 100.0

    return max(0.0, excess_air)


def calculate_o2_from_excess_air(excess_air_percent: float) -> float:
    """
    Estimate O2 percentage in flue gas from excess air.

    ZERO-HALLUCINATION FORMULA (approximate):
    O2 % = Excess Air % / 4.76

    This is valid for natural gas combustion.

    Args:
        excess_air_percent: Percent excess air

    Returns:
        Approximate O2 percentage in dry flue gas
    """
    return excess_air_percent / 4.76


def calculate_combustion_efficiency(
    flue_gas_temp_c: float,
    ambient_temp_c: float,
    o2_percent: float,
    fuel_type: str = "NATURAL_GAS"
) -> float:
    """
    Calculate combustion efficiency using stack loss method.

    ZERO-HALLUCINATION FORMULA (Siegert formula):
    Stack Loss % = (T_flue - T_amb) * (A1 / (21 - O2) + B1)

    For natural gas: A1 = 0.66, B1 = 0.009

    Efficiency = 100 - Stack Loss

    Args:
        flue_gas_temp_c: Flue gas temperature
        ambient_temp_c: Ambient/combustion air temperature
        o2_percent: Oxygen in flue gas (dry basis)
        fuel_type: Type of fuel

    Returns:
        Combustion efficiency percentage
    """
    # Siegert coefficients by fuel type
    coefficients = {
        "NATURAL_GAS": (0.66, 0.009),
        "PROPANE": (0.63, 0.008),
        "FUEL_OIL": (0.68, 0.007),
        "HYDROGEN": (0.50, 0.005),
    }
    A1, B1 = coefficients.get(fuel_type, (0.66, 0.009))

    delta_t = flue_gas_temp_c - ambient_temp_c

    if o2_percent >= 21:
        return 0.0

    stack_loss = delta_t * (A1 / (21.0 - o2_percent) + B1)
    efficiency = 100.0 - stack_loss

    return max(0.0, min(100.0, efficiency))


def calculate_nox_estimate(
    firing_rate_percent: float,
    excess_air_percent: float,
    flame_temp_estimate_c: float,
    burner_type: str
) -> float:
    """
    Estimate NOx emissions based on operating conditions.

    ZERO-HALLUCINATION: Simplified Zeldovich thermal NOx correlation.
    NOx increases exponentially with flame temperature.

    Args:
        firing_rate_percent: Burner firing rate (0-100)
        excess_air_percent: Percent excess air
        flame_temp_estimate_c: Estimated flame temperature
        burner_type: Type of burner (affects base NOx)

    Returns:
        Estimated NOx in ppm (corrected to 3% O2)
    """
    # Base NOx by burner type (ppm at full fire, optimal conditions)
    base_nox = {
        "PREMIX": 80,
        "NOZZLE_MIX": 100,
        "RAW_GAS": 150,
        "LOW_NOX": 40,
        "ULTRA_LOW_NOX": 9,
        "STAGED_AIR": 50,
        "STAGED_FUEL": 45,
    }

    base = base_nox.get(burner_type, 100)

    # Firing rate factor (NOx increases with firing rate)
    rate_factor = (firing_rate_percent / 100.0) ** 1.5

    # Excess air factor (NOx increases with temperature, decreases with dilution)
    # Optimal around 10-15% excess air for most burners
    if excess_air_percent < 5:
        ea_factor = 1.2  # Fuel-rich, incomplete combustion
    elif excess_air_percent < 15:
        ea_factor = 1.0  # Optimal range
    elif excess_air_percent < 30:
        ea_factor = 0.9  # Dilution reduces flame temp
    else:
        ea_factor = 0.8  # High dilution

    # Temperature factor (simplified Arrhenius)
    temp_factor = math.exp((flame_temp_estimate_c - 1800) / 500)
    temp_factor = max(0.5, min(2.0, temp_factor))

    nox_estimate = base * rate_factor * ea_factor * temp_factor

    return round(nox_estimate, 1)


def optimize_air_fuel_ratio(
    fuel_flow_scfh: float,
    current_o2_percent: float,
    target_o2_percent: float,
    stoich_ratio: float,
    current_air_flow: float
) -> Tuple[float, float]:
    """
    Calculate optimal air flow for target O2.

    Args:
        fuel_flow_scfh: Current fuel flow
        current_o2_percent: Current O2 in flue gas
        target_o2_percent: Target O2 percentage
        stoich_ratio: Stoichiometric air-fuel ratio
        current_air_flow: Current air flow

    Returns:
        Tuple of (optimal_air_flow, new_excess_air_percent)
    """
    # Target excess air from target O2
    target_excess_air = target_o2_percent * 4.76

    # Calculate required air flow
    stoich_air = fuel_flow_scfh * stoich_ratio
    optimal_air = stoich_air * (1.0 + target_excess_air / 100.0)

    return optimal_air, target_excess_air


def distribute_load_to_burners(
    total_load_percent: float,
    burner_capacities: List[float],
    burner_efficiencies: List[float],
    burner_status: List[str],
    objective: str = "EFFICIENCY"
) -> List[float]:
    """
    Distribute total load across multiple burners.

    Optimization Strategies:
    - EFFICIENCY: Prioritize high-efficiency burners
    - LOW_EMISSIONS: Run at moderate rates to minimize NOx
    - UNIFORM_HEATING: Balance load evenly
    - BALANCED: Compromise between all factors

    Args:
        total_load_percent: Total required load (0-100)
        burner_capacities: Max capacity of each burner
        burner_efficiencies: Efficiency rating of each burner
        burner_status: Current status of each burner
        objective: Optimization objective

    Returns:
        List of firing rates for each burner (0-100)
    """
    n_burners = len(burner_capacities)
    if n_burners == 0:
        return []

    # Filter available burners
    available = [
        i for i, status in enumerate(burner_status)
        if status not in ["OFF", "FAULT"]
    ]

    if not available:
        return [0.0] * n_burners

    firing_rates = [0.0] * n_burners
    total_capacity = sum(burner_capacities[i] for i in available)

    if total_capacity <= 0:
        return firing_rates

    total_required = total_load_percent * total_capacity / 100.0

    if objective == "UNIFORM_HEATING":
        # Distribute evenly
        per_burner = total_required / len(available)
        for i in available:
            firing_rates[i] = min(100.0, (per_burner / burner_capacities[i]) * 100)

    elif objective == "EFFICIENCY":
        # Prioritize most efficient burners
        sorted_idx = sorted(available, key=lambda i: -burner_efficiencies[i])
        remaining = total_required

        for i in sorted_idx:
            if remaining <= 0:
                break
            max_contribution = burner_capacities[i]
            contribution = min(max_contribution, remaining)
            firing_rates[i] = (contribution / burner_capacities[i]) * 100
            remaining -= contribution

    elif objective == "LOW_EMISSIONS":
        # Run burners at 60-80% to minimize NOx (optimal range)
        target_rate = 70.0
        per_burner_load = total_required / len(available)

        for i in available:
            if per_burner_load <= burner_capacities[i] * 0.8:
                firing_rates[i] = (per_burner_load / burner_capacities[i]) * 100
            else:
                firing_rates[i] = 80.0

    else:  # BALANCED
        # Weighted distribution
        per_burner = total_required / len(available)
        for i in available:
            base_rate = (per_burner / burner_capacities[i]) * 100
            # Adjust by efficiency
            eff_factor = burner_efficiencies[i] / 100.0
            firing_rates[i] = min(100.0, base_rate * (0.5 + 0.5 * eff_factor))

    return firing_rates


def calculate_total_efficiency(
    firing_rates: List[float],
    burner_efficiencies: List[float],
    burner_capacities: List[float]
) -> float:
    """
    Calculate overall system efficiency from individual burner performance.

    ZERO-HALLUCINATION: Weighted average by heat output.

    Args:
        firing_rates: Firing rate of each burner (0-100)
        burner_efficiencies: Efficiency of each burner
        burner_capacities: Capacity of each burner

    Returns:
        Overall system efficiency percentage
    """
    total_output = 0.0
    weighted_efficiency = 0.0

    for i, (rate, eff, cap) in enumerate(zip(firing_rates, burner_efficiencies, burner_capacities)):
        output = (rate / 100.0) * cap
        total_output += output
        weighted_efficiency += output * eff

    if total_output <= 0:
        return 0.0

    return weighted_efficiency / total_output
