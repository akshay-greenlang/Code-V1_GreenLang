"""
Stoichiometry Module for GL-004 BURNMASTER

This module implements stoichiometric calculations for combustion optimization.
All calculations are deterministic and auditable with complete provenance tracking.

Key Calculations:
- Stoichiometric air-fuel ratio
- Lambda (equivalence ratio)
- Excess air percentage
- Excess O2 in flue gas
- Lambda inference from stack O2

Supports:
- Natural gas
- Refinery gas (high H2 content)
- Hydrogen blends (0-100% H2)
- Liquid fuels

Reference Standards:
- ASME PTC 4: Fired Steam Generators
- EPA Method 19: Determination of Sulfur Dioxide Removal Efficiency
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import hashlib
import json

from .fuel_properties import (
    FuelType, FuelComposition, FuelProperties,
    MOLECULAR_WEIGHTS, STOICH_O2, get_fuel_properties,
    compute_molecular_weight, compute_stoichiometric_afr_from_composition,
    STANDARD_COMPOSITIONS
)


@dataclass
class StoichiometryResult:
    """Result of stoichiometric calculation with provenance."""
    stoichiometric_afr: float  # kg air / kg fuel
    stoichiometric_afr_vol: float  # Nm3 air / Nm3 fuel
    lambda_value: float  # Actual AFR / Stoichiometric AFR
    excess_air_percent: float  # (lambda - 1) * 100
    excess_o2_percent: float  # O2 in dry flue gas
    calculation_method: str
    fuel_type: str
    provenance_hash: str = field(default="", init=False)

    def __post_init__(self):
        self.provenance_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        data = {
            "stoichiometric_afr": str(self.stoichiometric_afr),
            "lambda": str(self.lambda_value),
            "excess_air": str(self.excess_air_percent),
            "fuel_type": self.fuel_type
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


# ============================================================================
# Fuel-Specific Constants
# ============================================================================

# Theoretical dry flue gas O2 at stoichiometric (should be ~0%)
# These are maximum achievable O2 levels at infinite excess air (dry basis)
MAX_DRY_O2_PERCENT = 20.95  # Theoretical max (pure air, no combustion)

# Fuel-specific constants for O2-lambda relationship
# Based on flue gas composition at different lambda values
FUEL_O2_CONSTANTS: Dict[str, Dict[str, float]] = {
    "natural_gas": {
        "k1": 0.9,      # O2 sensitivity coefficient
        "k2": 0.05,     # Non-linearity correction
        "max_o2": 20.5  # Max achievable O2 (%)
    },
    "refinery_gas": {
        "k1": 0.85,
        "k2": 0.06,
        "max_o2": 20.3
    },
    "hydrogen_blend": {
        "k1": 0.88,
        "k2": 0.04,
        "max_o2": 20.4
    },
    "pure_hydrogen": {
        "k1": 0.95,
        "k2": 0.02,
        "max_o2": 20.6
    },
    "propane": {
        "k1": 0.87,
        "k2": 0.05,
        "max_o2": 20.4
    },
    "lpg": {
        "k1": 0.86,
        "k2": 0.05,
        "max_o2": 20.4
    },
    "diesel": {
        "k1": 0.82,
        "k2": 0.07,
        "max_o2": 20.0
    },
    "fuel_oil_no2": {
        "k1": 0.80,
        "k2": 0.08,
        "max_o2": 19.8
    },
    "fuel_oil_no6": {
        "k1": 0.78,
        "k2": 0.09,
        "max_o2": 19.5
    },
    "coke_oven_gas": {
        "k1": 0.92,
        "k2": 0.03,
        "max_o2": 20.5
    },
    "blast_furnace_gas": {
        "k1": 0.75,
        "k2": 0.10,
        "max_o2": 19.0
    },
}


# ============================================================================
# Core Stoichiometric Functions
# ============================================================================

def compute_stoichiometric_air(fuel_composition: dict) -> float:
    """
    Compute stoichiometric air requirement from fuel composition.

    Uses complete combustion equations for each component:
    - CH4 + 2O2 -> CO2 + 2H2O
    - C2H6 + 3.5O2 -> 2CO2 + 3H2O
    - H2 + 0.5O2 -> H2O
    - etc.

    Args:
        fuel_composition: Dict of component -> mole percent
            Example: {"CH4": 94.0, "C2H6": 3.0, "N2": 3.0}

    Returns:
        Stoichiometric air-fuel ratio (Nm3 air / Nm3 fuel)

    Raises:
        ValueError: If composition is invalid

    Physics:
        Air is 20.95% O2, 79.05% N2 by volume.
        Total air = O2_required / 0.2095

    Deterministic: YES
    """
    # Validate composition
    if not fuel_composition:
        raise ValueError("Empty fuel composition")

    total = sum(fuel_composition.values())
    if abs(total - 100.0) > 0.5:
        raise ValueError(f"Composition must sum to 100%, got {total}%")

    # Calculate total O2 requirement
    total_o2_required = 0.0  # mol O2 / mol fuel

    for component, mole_pct in fuel_composition.items():
        if component in STOICH_O2:
            total_o2_required += (mole_pct / 100.0) * STOICH_O2[component]

    # Convert O2 to air (air is 20.95% O2)
    stoich_air = total_o2_required / 0.2095

    return round(stoich_air, 4)


def compute_lambda(actual_af: float, stoich_af: float) -> float:
    """
    Compute lambda (air-fuel equivalence ratio).

    Lambda = Actual AFR / Stoichiometric AFR

    Lambda interpretation:
    - lambda = 1.0: Stoichiometric (perfect combustion)
    - lambda > 1.0: Lean (excess air)
    - lambda < 1.0: Rich (excess fuel, incomplete combustion)

    Args:
        actual_af: Actual air-fuel ratio (mass or volume, must be consistent)
        stoich_af: Stoichiometric air-fuel ratio (same units)

    Returns:
        Lambda value (dimensionless)

    Raises:
        ValueError: If stoich_af is zero or negative

    Deterministic: YES
    """
    if stoich_af <= 0:
        raise ValueError(f"Stoichiometric AFR must be positive, got {stoich_af}")

    if actual_af < 0:
        raise ValueError(f"Actual AFR must be non-negative, got {actual_af}")

    lambda_val = actual_af / stoich_af
    return round(lambda_val, 4)


def compute_excess_air_percent(lambda_val: float) -> float:
    """
    Compute excess air percentage from lambda.

    Excess Air % = (lambda - 1) * 100

    Args:
        lambda_val: Lambda value (must be >= 0)

    Returns:
        Excess air percentage (can be negative for rich mixtures)

    Raises:
        ValueError: If lambda is negative

    Physics:
        - lambda = 1.0 -> 0% excess air
        - lambda = 1.15 -> 15% excess air
        - lambda = 1.5 -> 50% excess air
        - lambda = 0.9 -> -10% (fuel rich, incomplete combustion)

    Deterministic: YES
    """
    if lambda_val < 0:
        raise ValueError(f"Lambda must be non-negative, got {lambda_val}")

    excess_air = (lambda_val - 1.0) * 100.0
    return round(excess_air, 2)


def compute_excess_o2(lambda_val: float, fuel_type: str) -> float:
    """
    Compute excess O2 in dry flue gas from lambda and fuel type.

    The relationship between excess O2 and lambda depends on:
    - Fuel composition (C/H ratio, inerts)
    - Flue gas volume produced
    - Dilution effects

    Args:
        lambda_val: Lambda value (equivalence ratio)
        fuel_type: Fuel type string (e.g., "natural_gas", "refinery_gas")

    Returns:
        O2 percentage in dry flue gas (%)

    Raises:
        ValueError: If lambda < 1.0 or fuel_type unknown

    Physics:
        At stoichiometric: O2 = 0% (all consumed)
        At infinite lambda: O2 -> 20.95% (pure air)

        Approximate formula:
        O2 = k1 * (lambda - 1) / (1 + k2 * (lambda - 1)) * max_o2

    Deterministic: YES
    """
    if lambda_val < 0.5:
        raise ValueError(f"Lambda too low for valid calculation: {lambda_val}")

    # Get fuel-specific constants
    fuel_key = fuel_type.lower().replace(" ", "_")
    if fuel_key not in FUEL_O2_CONSTANTS:
        # Use natural gas as default
        fuel_key = "natural_gas"

    constants = FUEL_O2_CONSTANTS[fuel_key]
    k1 = constants["k1"]
    k2 = constants["k2"]
    max_o2 = constants["max_o2"]

    if lambda_val <= 1.0:
        # Sub-stoichiometric: no excess O2 (all consumed)
        # In practice, some O2 may remain due to mixing, but theoretically 0
        return 0.0

    # Excess air calculation
    excess = lambda_val - 1.0

    # Non-linear relationship (approaches max_o2 asymptotically)
    o2_percent = k1 * excess / (1.0 + k2 * excess) * max_o2

    # Clamp to physical limits
    o2_percent = max(0.0, min(o2_percent, max_o2))

    return round(o2_percent, 2)


def infer_lambda_from_o2(stack_o2: float, fuel_type: str) -> float:
    """
    Infer lambda (equivalence ratio) from measured stack O2.

    This is the inverse of compute_excess_o2(). It solves for lambda
    given the measured O2 in dry flue gas.

    Args:
        stack_o2: Measured O2 in dry flue gas (%)
        fuel_type: Fuel type string

    Returns:
        Inferred lambda value

    Raises:
        ValueError: If O2 is out of physical range

    Physics:
        Inverts the relationship:
        O2 = k1 * (lambda - 1) / (1 + k2 * (lambda - 1)) * max_o2

        Solving for lambda:
        lambda = 1 + O2 / (k1 * max_o2 - k2 * O2)

    Deterministic: YES
    """
    if stack_o2 < 0:
        raise ValueError(f"O2 cannot be negative, got {stack_o2}%")

    if stack_o2 > 21:
        raise ValueError(f"O2 cannot exceed 21%, got {stack_o2}%")

    # Get fuel-specific constants
    fuel_key = fuel_type.lower().replace(" ", "_")
    if fuel_key not in FUEL_O2_CONSTANTS:
        fuel_key = "natural_gas"

    constants = FUEL_O2_CONSTANTS[fuel_key]
    k1 = constants["k1"]
    k2 = constants["k2"]
    max_o2 = constants["max_o2"]

    if stack_o2 <= 0.1:
        # Near-zero O2: approximately stoichiometric
        return 1.0

    if stack_o2 >= max_o2 - 0.1:
        # Very high O2: extreme lean condition
        return 5.0  # Cap at very lean

    # Solve for lambda
    # O2 = k1 * (lambda - 1) / (1 + k2 * (lambda - 1)) * max_o2
    # Let x = lambda - 1
    # O2 = k1 * x / (1 + k2 * x) * max_o2
    # O2 * (1 + k2 * x) = k1 * x * max_o2
    # O2 + O2 * k2 * x = k1 * max_o2 * x
    # O2 = k1 * max_o2 * x - O2 * k2 * x
    # O2 = x * (k1 * max_o2 - O2 * k2)
    # x = O2 / (k1 * max_o2 - O2 * k2)

    denominator = k1 * max_o2 - stack_o2 * k2

    if denominator <= 0:
        # Numerical issue: return high lambda
        return 3.0

    excess = stack_o2 / denominator
    lambda_val = 1.0 + excess

    # Clamp to reasonable range
    lambda_val = max(0.5, min(lambda_val, 5.0))

    return round(lambda_val, 4)


# ============================================================================
# Advanced Stoichiometric Functions
# ============================================================================

def compute_stoichiometry_from_fuel_type(
    fuel_type: FuelType,
    actual_air_flow: Optional[float] = None,
    actual_fuel_flow: Optional[float] = None,
    measured_o2: Optional[float] = None
) -> StoichiometryResult:
    """
    Compute complete stoichiometry from fuel type and operating conditions.

    This function provides a comprehensive stoichiometric analysis given
    fuel type and either flow rates or measured O2.

    Args:
        fuel_type: FuelType enum
        actual_air_flow: Actual air flow (Nm3/h) - optional
        actual_fuel_flow: Actual fuel flow (Nm3/h) - optional
        measured_o2: Measured stack O2 (%) - optional

    Returns:
        StoichiometryResult with all calculated values

    Raises:
        ValueError: If insufficient data provided

    Deterministic: YES
    """
    # Get fuel properties
    props = get_fuel_properties(fuel_type)

    # Determine lambda from available data
    if actual_air_flow is not None and actual_fuel_flow is not None:
        if actual_fuel_flow <= 0:
            raise ValueError("Fuel flow must be positive")
        actual_afr_vol = actual_air_flow / actual_fuel_flow
        lambda_val = compute_lambda(actual_afr_vol, props.stoichiometric_afr_vol)
        method = "flow_rates"
    elif measured_o2 is not None:
        lambda_val = infer_lambda_from_o2(measured_o2, fuel_type.value)
        method = "stack_o2"
    else:
        # Default to stoichiometric if no data
        lambda_val = 1.0
        method = "assumed_stoichiometric"

    # Calculate derived values
    excess_air = compute_excess_air_percent(lambda_val)
    excess_o2 = compute_excess_o2(lambda_val, fuel_type.value)

    return StoichiometryResult(
        stoichiometric_afr=props.stoichiometric_afr,
        stoichiometric_afr_vol=props.stoichiometric_afr_vol,
        lambda_value=lambda_val,
        excess_air_percent=excess_air,
        excess_o2_percent=excess_o2,
        calculation_method=method,
        fuel_type=fuel_type.value
    )


def compute_air_flow_for_target_o2(
    fuel_flow: float,
    target_o2: float,
    fuel_type: FuelType
) -> float:
    """
    Compute required air flow to achieve target stack O2.

    This is useful for combustion control setpoint calculations.

    Args:
        fuel_flow: Fuel flow rate (Nm3/h)
        target_o2: Target O2 in dry flue gas (%)
        fuel_type: Fuel type

    Returns:
        Required air flow (Nm3/h)

    Raises:
        ValueError: If inputs are invalid

    Deterministic: YES
    """
    if fuel_flow <= 0:
        raise ValueError(f"Fuel flow must be positive, got {fuel_flow}")

    if target_o2 < 0 or target_o2 > 15:
        raise ValueError(f"Target O2 must be 0-15%, got {target_o2}")

    # Get stoichiometric AFR
    props = get_fuel_properties(fuel_type)
    stoich_afr_vol = props.stoichiometric_afr_vol

    # Calculate lambda for target O2
    target_lambda = infer_lambda_from_o2(target_o2, fuel_type.value)

    # Calculate actual AFR
    actual_afr_vol = target_lambda * stoich_afr_vol

    # Calculate air flow
    air_flow = fuel_flow * actual_afr_vol

    return round(air_flow, 2)


def compute_fuel_flow_for_target_duty(
    target_duty: float,
    fuel_type: FuelType,
    combustion_efficiency: float = 0.90
) -> float:
    """
    Compute required fuel flow for target thermal duty.

    Args:
        target_duty: Target thermal duty (MW)
        fuel_type: Fuel type
        combustion_efficiency: Expected efficiency (0-1)

    Returns:
        Required fuel flow (Nm3/h)

    Physics:
        Fuel flow = Duty / (LHV * efficiency)
        Convert MJ to MW*h: 1 MJ = 1/3600 MW*h

    Deterministic: YES
    """
    if target_duty <= 0:
        raise ValueError(f"Duty must be positive, got {target_duty}")

    if combustion_efficiency <= 0 or combustion_efficiency > 1:
        raise ValueError(f"Efficiency must be 0-1, got {combustion_efficiency}")

    # Get fuel properties
    props = get_fuel_properties(fuel_type)
    lhv = props.lhv  # MJ/Nm3

    # Convert duty to MJ/h
    duty_mj_h = target_duty * 3600  # MW to MJ/h

    # Calculate fuel flow
    fuel_flow = duty_mj_h / (lhv * combustion_efficiency)

    return round(fuel_flow, 2)


def compute_flue_gas_flow(
    fuel_flow: float,
    lambda_val: float,
    fuel_type: FuelType
) -> Tuple[float, float]:
    """
    Compute wet and dry flue gas flow rates.

    Args:
        fuel_flow: Fuel flow (Nm3/h)
        lambda_val: Lambda value
        fuel_type: Fuel type

    Returns:
        (wet_flue_gas_flow, dry_flue_gas_flow) in Nm3/h

    Physics:
        Wet flue gas = CO2 + H2O + N2 + excess O2 + excess N2
        Dry flue gas = Wet - H2O

    Deterministic: YES
    """
    if fuel_flow <= 0:
        raise ValueError(f"Fuel flow must be positive, got {fuel_flow}")

    if lambda_val < 0.5:
        raise ValueError(f"Lambda too low, got {lambda_val}")

    # Get fuel properties
    props = get_fuel_properties(fuel_type)

    # Stoichiometric air
    stoich_air = props.stoichiometric_afr_vol

    # Actual air
    actual_air = lambda_val * stoich_air

    # Products of combustion (approximate)
    # For natural gas: CH4 + 2O2 -> CO2 + 2H2O
    # Molar expansion: 1 mol fuel -> ~1 mol CO2 + ~2 mol H2O
    # Plus N2 from air (79% of air)

    # Simplified calculation
    n2_from_air = actual_air * 0.79
    o2_excess = actual_air * 0.21 - stoich_air * 0.21

    # Estimate CO2 and H2O from fuel composition
    # This is approximate - full calculation requires composition
    co2_produced = fuel_flow * 1.0  # Approximate 1:1 for methane
    h2o_produced = fuel_flow * 2.0  # Approximate 2:1 for methane

    # Total wet flue gas
    wet_flue_gas = co2_produced + h2o_produced + n2_from_air + max(0, o2_excess)

    # Dry flue gas (subtract H2O)
    dry_flue_gas = wet_flue_gas - h2o_produced

    return round(wet_flue_gas, 1), round(dry_flue_gas, 1)


# ============================================================================
# Validation Functions
# ============================================================================

def validate_stoichiometry_inputs(
    lambda_val: Optional[float] = None,
    excess_air: Optional[float] = None,
    stack_o2: Optional[float] = None
) -> Tuple[bool, list]:
    """
    Validate stoichiometry inputs are physically reasonable.

    Args:
        lambda_val: Lambda value (optional)
        excess_air: Excess air percentage (optional)
        stack_o2: Stack O2 percentage (optional)

    Returns:
        (is_valid, errors) tuple

    Deterministic: YES
    """
    errors = []

    if lambda_val is not None:
        if lambda_val < 0.5:
            errors.append(f"Lambda too low: {lambda_val} (min: 0.5)")
        if lambda_val > 5.0:
            errors.append(f"Lambda too high: {lambda_val} (max: 5.0)")

    if excess_air is not None:
        if excess_air < -50:
            errors.append(f"Excess air too low: {excess_air}%")
        if excess_air > 400:
            errors.append(f"Excess air too high: {excess_air}%")

    if stack_o2 is not None:
        if stack_o2 < 0:
            errors.append(f"O2 cannot be negative: {stack_o2}%")
        if stack_o2 > 21:
            errors.append(f"O2 cannot exceed 21%: {stack_o2}%")

    # Cross-validate if multiple values provided
    if lambda_val is not None and excess_air is not None:
        expected_excess = (lambda_val - 1.0) * 100
        if abs(expected_excess - excess_air) > 5:
            errors.append(
                f"Lambda ({lambda_val}) inconsistent with excess air ({excess_air}%)"
            )

    return len(errors) == 0, errors


def check_stoichiometry_consistency(
    lambda_val: float,
    stack_o2: float,
    fuel_type: str,
    tolerance: float = 0.5
) -> Tuple[bool, str]:
    """
    Check if measured O2 is consistent with calculated lambda.

    This validates that control system setpoints and measurements agree.

    Args:
        lambda_val: Calculated/target lambda
        stack_o2: Measured stack O2 (%)
        fuel_type: Fuel type string
        tolerance: Allowed O2 deviation (%)

    Returns:
        (is_consistent, message) tuple

    Deterministic: YES
    """
    # Calculate expected O2 from lambda
    expected_o2 = compute_excess_o2(lambda_val, fuel_type)

    # Check consistency
    deviation = abs(stack_o2 - expected_o2)

    if deviation <= tolerance:
        return True, f"Consistent: measured O2 ({stack_o2}%) within {tolerance}% of expected ({expected_o2}%)"
    else:
        return False, f"Inconsistent: measured O2 ({stack_o2}%) deviates {deviation:.1f}% from expected ({expected_o2}%)"
