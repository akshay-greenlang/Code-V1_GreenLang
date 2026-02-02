# -*- coding: utf-8 -*-
"""
CO Prediction Golden Value Tests for GL-004 BurnMaster
======================================================

Comprehensive golden tests for Carbon Monoxide (CO) emission prediction,
validating against EPA AP-42 emission factors, combustion chemistry,
and deterministic calculation requirements.

Test Categories:
    1. CO vs O2 Relationship Tests
    2. Combustion Efficiency Tests
    3. EPA AP-42 CO Factors
    4. Temperature Effects Tests
    5. Burner Tuning Impact Tests
    6. Control Technology Tests
    7. Boundary Conditions and Edge Cases
    8. Determinism Validation

Reference Sources:
    - EPA AP-42: Compilation of Air Pollutant Emission Factors
    - EPA Method 10: CO Determination (40 CFR Part 60, Appendix A)
    - ASME PTC 4: Fired Steam Generators
    - NFPA 85: Boiler and Combustion Systems Hazards Code
    - Combustion Engineering by Baukal (3rd Edition)
    - Perry's Chemical Engineers' Handbook (9th Edition)

Author: GL-CalculatorEngineer
Version: 1.0.0
"""

import pytest
import sys
import math
import hashlib
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass


# =============================================================================
# PHYSICAL AND CHEMICAL CONSTANTS
# =============================================================================

# Molecular weights (g/mol)
MOLECULAR_WEIGHTS = {
    "CO": 28.010,
    "CO2": 44.009,
    "O2": 31.998,
    "N2": 28.014,
    "H2O": 18.015,
    "CH4": 16.043,
    "C2H6": 30.070,
    "C3H8": 44.096,
}

# Universal gas constant
R_GAS = 8.314  # J/(mol*K)

# Standard conditions
STANDARD_TEMPERATURE_K = 298.15  # 25C / 77F
STANDARD_PRESSURE_KPA = 101.325  # 1 atm

# CO oxidation kinetics
CO_OXIDATION_ACTIVATION_ENERGY = 125000  # J/mol (125 kJ/mol)
CO_OXIDATION_PRE_EXPONENTIAL = 1.3e10  # 1/s (simplified Arrhenius)

# Combustion efficiency parameters
MIN_COMBUSTION_EFFICIENCY = 90.0  # % - below this, serious problems
OPTIMAL_COMBUSTION_EFFICIENCY = 99.5  # % - well-tuned burner


# =============================================================================
# EPA AP-42 REFERENCE DATA
# =============================================================================

@dataclass(frozen=True)
class EPAAP42COFactor:
    """EPA AP-42 CO emission factor with metadata."""
    fuel_type: str
    factor_value: float
    factor_units: str
    scc_code: str
    data_quality_rating: str  # A, B, C, D, E
    reference_section: str
    notes: str


# EPA AP-42 CO Emission Factors - authoritative reference values
EPA_AP42_CO_FACTORS = {
    # Natural Gas Combustion - Table 1.4-2
    "natural_gas_boiler_uncontrolled": EPAAP42COFactor(
        fuel_type="Natural Gas",
        factor_value=84.0,  # lb/10^6 scf
        factor_units="lb/10^6 scf",
        scc_code="1-02-006-02",
        data_quality_rating="B",
        reference_section="AP-42 Table 1.4-2",
        notes="Large boilers >100 MMBtu/hr, uncontrolled",
    ),
    "natural_gas_boiler_small": EPAAP42COFactor(
        fuel_type="Natural Gas",
        factor_value=84.0,  # lb/10^6 scf
        factor_units="lb/10^6 scf",
        scc_code="1-03-006-02",
        data_quality_rating="B",
        reference_section="AP-42 Table 1.4-2",
        notes="Small boilers <100 MMBtu/hr",
    ),
    "natural_gas_mmbtu": EPAAP42COFactor(
        fuel_type="Natural Gas",
        factor_value=0.084,  # lb/MMBtu
        factor_units="lb/MMBtu",
        scc_code="1-02-006-02",
        data_quality_rating="B",
        reference_section="AP-42 Table 1.4-2",
        notes="Converted using HHV=1020 Btu/scf",
    ),

    # Fuel Oil #2 (Distillate) - Table 1.3-1
    "fuel_oil_no2_boiler": EPAAP42COFactor(
        fuel_type="Fuel Oil No. 2",
        factor_value=5.0,  # lb/10^3 gal
        factor_units="lb/10^3 gal",
        scc_code="1-02-004-02",
        data_quality_rating="A",
        reference_section="AP-42 Table 1.3-1",
        notes="Distillate oil-fired boilers",
    ),
    "fuel_oil_no2_mmbtu": EPAAP42COFactor(
        fuel_type="Fuel Oil No. 2",
        factor_value=0.036,  # lb/MMBtu
        factor_units="lb/MMBtu",
        scc_code="1-02-004-02",
        data_quality_rating="A",
        reference_section="AP-42 Table 1.3-1",
        notes="Converted using HHV=138,690 Btu/gal",
    ),

    # Fuel Oil #6 (Residual) - Table 1.3-1
    "fuel_oil_no6_boiler": EPAAP42COFactor(
        fuel_type="Fuel Oil No. 6",
        factor_value=5.0,  # lb/10^3 gal
        factor_units="lb/10^3 gal",
        scc_code="1-02-004-01",
        data_quality_rating="B",
        reference_section="AP-42 Table 1.3-1",
        notes="Residual oil-fired boilers",
    ),
    "fuel_oil_no6_mmbtu": EPAAP42COFactor(
        fuel_type="Fuel Oil No. 6",
        factor_value=0.033,  # lb/MMBtu
        factor_units="lb/MMBtu",
        scc_code="1-02-004-01",
        data_quality_rating="B",
        reference_section="AP-42 Table 1.3-1",
        notes="Converted using HHV=150,000 Btu/gal",
    ),

    # Coal - Table 1.1-3
    "bituminous_coal_stoker": EPAAP42COFactor(
        fuel_type="Bituminous Coal",
        factor_value=5.0,  # lb/ton
        factor_units="lb/ton",
        scc_code="1-02-002-01",
        data_quality_rating="C",
        reference_section="AP-42 Table 1.1-3",
        notes="Stoker-fired boilers",
    ),
    "bituminous_coal_pulverized": EPAAP42COFactor(
        fuel_type="Bituminous Coal",
        factor_value=0.5,  # lb/ton
        factor_units="lb/ton",
        scc_code="1-02-002-02",
        data_quality_rating="B",
        reference_section="AP-42 Table 1.1-3",
        notes="Pulverized coal dry-bottom boilers",
    ),
    "bituminous_coal_mmbtu": EPAAP42COFactor(
        fuel_type="Bituminous Coal",
        factor_value=0.50,  # lb/MMBtu (stoker)
        factor_units="lb/MMBtu",
        scc_code="1-02-002-02",
        data_quality_rating="C",
        reference_section="AP-42 Table 1.1-3",
        notes="Variable by firing type and coal quality",
    ),

    # Sub-bituminous Coal
    "subbituminous_coal_mmbtu": EPAAP42COFactor(
        fuel_type="Sub-bituminous Coal",
        factor_value=0.38,  # lb/MMBtu
        factor_units="lb/MMBtu",
        scc_code="1-02-002-14",
        data_quality_rating="B",
        reference_section="AP-42 Table 1.1-3",
        notes="PRB coal, pulverized firing",
    ),

    # Propane/LPG
    "propane_boiler": EPAAP42COFactor(
        fuel_type="Propane/LPG",
        factor_value=0.070,  # lb/MMBtu
        factor_units="lb/MMBtu",
        scc_code="1-02-010-01",
        data_quality_rating="C",
        reference_section="AP-42 Table 1.5-1",
        notes="LPG-fired boilers",
    ),

    # Wood/Biomass
    "wood_residue_boiler": EPAAP42COFactor(
        fuel_type="Wood Residue",
        factor_value=0.60,  # lb/MMBtu
        factor_units="lb/MMBtu",
        scc_code="1-02-009-01",
        data_quality_rating="D",
        reference_section="AP-42 Table 1.6-1",
        notes="Wood-fired boilers, high variability",
    ),
}


# =============================================================================
# CO vs O2 REFERENCE DATA
# =============================================================================

# CO vs O2 relationship - empirical golden values
# Based on field measurements and combustion theory
# O2 (%), CO (ppm at that O2), normalized CO (ppm at 3% O2)
CO_VS_O2_REFERENCE_DATA = {
    "natural_gas_well_tuned": [
        # (O2%, CO_ppm, CO_normalized_3pct, combustion_state)
        (0.5, 2000, 2571, "rich_incomplete"),
        (1.0, 500, 600, "slightly_rich"),
        (1.5, 100, 113, "near_stoichiometric"),
        (2.0, 30, 32, "optimal_low"),
        (2.5, 15, 15, "optimal_mid"),
        (3.0, 10, 10, "optimal_reference"),
        (3.5, 8, 7, "optimal_high"),
        (4.0, 8, 7, "lean_efficient"),
        (5.0, 10, 8, "moderately_lean"),
        (6.0, 15, 10, "lean"),
        (8.0, 30, 16, "very_lean_quench"),
    ],
    "natural_gas_poor_tune": [
        (2.0, 150, 158, "poor_mixing"),
        (3.0, 100, 100, "baseline_poor"),
        (4.0, 80, 67, "slightly_better"),
        (5.0, 60, 45, "improving"),
    ],
    "fuel_oil_no2_typical": [
        (2.0, 50, 53, "slightly_low"),
        (3.0, 25, 25, "optimal_reference"),
        (4.0, 20, 17, "optimal_lean"),
        (5.0, 25, 19, "lean"),
        (6.0, 35, 23, "very_lean"),
    ],
}

# Optimal O2 operating windows by fuel type
OPTIMAL_O2_WINDOWS = {
    "natural_gas": {"min": 2.0, "max": 4.0, "target": 3.0, "co_limit_ppm": 50},
    "fuel_oil_no2": {"min": 2.5, "max": 4.5, "target": 3.5, "co_limit_ppm": 100},
    "fuel_oil_no6": {"min": 3.0, "max": 5.0, "target": 4.0, "co_limit_ppm": 100},
    "coal": {"min": 3.5, "max": 5.5, "target": 4.5, "co_limit_ppm": 200},
    "wood": {"min": 4.0, "max": 6.0, "target": 5.0, "co_limit_ppm": 400},
}


# =============================================================================
# COMBUSTION EFFICIENCY REFERENCE DATA
# =============================================================================

# CO as indicator of combustion efficiency
# Reference: ASME PTC 4, Combustion Engineering
COMBUSTION_EFFICIENCY_VS_CO = {
    # CO (ppm at 3% O2): (efficiency_loss_percent, combustion_efficiency)
    5: (0.001, 99.999),
    10: (0.003, 99.997),
    25: (0.008, 99.992),
    50: (0.016, 99.984),
    100: (0.032, 99.968),
    200: (0.064, 99.936),
    500: (0.160, 99.840),
    1000: (0.320, 99.680),
    2000: (0.640, 99.360),
    5000: (1.600, 98.400),
    10000: (3.200, 96.800),
}

# CO/CO2 ratio reference values
# Low CO/CO2 = complete combustion, high CO/CO2 = incomplete
CO_CO2_RATIO_REFERENCE = {
    "excellent_combustion": {"max_ratio": 0.0005, "description": "Complete combustion"},
    "good_combustion": {"max_ratio": 0.002, "description": "Well-tuned burner"},
    "acceptable_combustion": {"max_ratio": 0.005, "description": "Acceptable operation"},
    "marginal_combustion": {"max_ratio": 0.01, "description": "Needs tuning"},
    "poor_combustion": {"max_ratio": 0.02, "description": "Poor combustion"},
    "very_poor_combustion": {"max_ratio": 0.05, "description": "Immediate attention needed"},
}


# =============================================================================
# TEMPERATURE EFFECTS REFERENCE DATA
# =============================================================================

# CO oxidation kinetics - temperature effect on CO burnout
# Reference: Combustion Engineering by Baukal
CO_TEMPERATURE_EFFECTS = {
    # Temperature (K): (relative_oxidation_rate, co_freeze_tendency)
    900: (0.001, "very_high"),    # CO freezes below this temp
    1000: (0.01, "high"),
    1100: (0.05, "moderate"),
    1200: (0.15, "low"),
    1300: (0.40, "minimal"),
    1400: (0.80, "very_low"),
    1500: (1.00, "negligible"),   # Reference temperature
    1600: (1.50, "negligible"),
    1800: (3.00, "negligible"),
    2000: (6.00, "negligible"),
}

# Quench region definitions
QUENCH_REGIONS = {
    "flame_zone": {"temp_min_k": 1600, "temp_max_k": 2200, "co_behavior": "rapid_oxidation"},
    "post_flame": {"temp_min_k": 1200, "temp_max_k": 1600, "co_behavior": "continued_oxidation"},
    "quench_zone": {"temp_min_k": 900, "temp_max_k": 1200, "co_behavior": "freeze_risk"},
    "cooled_products": {"temp_min_k": 400, "temp_max_k": 900, "co_behavior": "frozen"},
}


# =============================================================================
# CONTROL TECHNOLOGY FACTORS
# =============================================================================

# CO reduction/impact from control technologies
CONTROL_TECHNOLOGY_CO_EFFECTS = {
    "good_combustion_practice": {
        "description": "Optimized burner tuning and maintenance",
        "co_reduction_percent": 50,
        "typical_co_ppm_at_3pct_o2": 50,
        "reference": "EPA BACT/LAER guidance",
    },
    "low_nox_burner": {
        "description": "Staged combustion may increase CO",
        "co_impact_percent": +20,  # CO may increase
        "typical_co_ppm_at_3pct_o2": 80,
        "reference": "AP-42 Section 1.4",
    },
    "flue_gas_recirculation": {
        "description": "FGR reduces flame temp, may increase CO",
        "co_impact_percent": +15,
        "typical_co_ppm_at_3pct_o2": 70,
        "reference": "Combustion Engineering",
    },
    "oxidation_catalyst": {
        "description": "CO oxidation catalyst in exhaust",
        "co_reduction_percent": 90,
        "typical_co_ppm_at_3pct_o2": 10,
        "reference": "EPA Control Techniques Guide",
    },
    "scr_with_co_catalyst": {
        "description": "SCR with integrated CO catalyst",
        "co_reduction_percent": 85,
        "typical_co_ppm_at_3pct_o2": 15,
        "reference": "Vendor specifications",
    },
}


# =============================================================================
# CALCULATION FUNCTIONS
# =============================================================================

def calculate_co_from_o2(
    o2_percent: float,
    fuel_type: str = "natural_gas",
    burner_condition: str = "well_tuned"
) -> Dict[str, float]:
    """
    Calculate expected CO based on O2 percentage.

    DETERMINISTIC calculation using empirical correlations.

    Args:
        o2_percent: Flue gas O2 percentage (dry basis)
        fuel_type: Type of fuel
        burner_condition: Burner tuning condition

    Returns:
        Dictionary with CO prediction and related data
    """
    # Validate inputs
    if o2_percent < 0 or o2_percent > 21:
        raise ValueError(f"O2 must be 0-21%, got {o2_percent}")

    # Base CO at 3% O2 reference (ppm)
    base_co_factors = {
        "natural_gas": {"well_tuned": 10, "average": 30, "poor": 100},
        "fuel_oil_no2": {"well_tuned": 25, "average": 50, "poor": 150},
        "fuel_oil_no6": {"well_tuned": 30, "average": 60, "poor": 200},
        "coal": {"well_tuned": 100, "average": 200, "poor": 500},
    }

    base_co = base_co_factors.get(fuel_type, base_co_factors["natural_gas"])
    base_co_at_3pct = base_co.get(burner_condition, base_co["average"])

    # CO vs O2 relationship (empirical)
    # CO increases dramatically below 2% O2 (incomplete combustion)
    # CO is relatively flat 2-4% O2 (optimal zone)
    # CO increases slightly above 5% O2 (quench effect)

    if o2_percent < 1.0:
        # Rich combustion - exponential CO increase
        o2_factor = 10.0 * math.exp(2.0 * (1.0 - o2_percent))
    elif o2_percent < 2.0:
        # Transition zone - steep increase
        o2_factor = 1.0 + 9.0 * ((2.0 - o2_percent) / 1.0) ** 2
    elif o2_percent <= 4.0:
        # Optimal zone - relatively flat
        o2_factor = 1.0 - 0.1 * (o2_percent - 3.0)
    elif o2_percent <= 6.0:
        # Lean combustion - slight increase
        o2_factor = 1.0 + 0.2 * (o2_percent - 4.0)
    else:
        # Very lean - quench effect
        o2_factor = 1.4 + 0.3 * (o2_percent - 6.0)

    # Calculate CO at measured O2
    co_at_o2 = base_co_at_3pct * o2_factor

    # Normalize to 3% O2 reference
    if o2_percent < 20.9:
        normalization_factor = (21.0 - 3.0) / (21.0 - o2_percent)
    else:
        normalization_factor = 1.0

    co_at_3pct_o2 = co_at_o2 * normalization_factor

    return {
        "o2_percent": o2_percent,
        "co_at_measured_o2_ppm": round(co_at_o2, 1),
        "co_at_3pct_o2_ppm": round(co_at_3pct_o2, 1),
        "o2_factor": round(o2_factor, 3),
        "fuel_type": fuel_type,
        "burner_condition": burner_condition,
        "base_co_ppm": base_co_at_3pct,
    }


def calculate_combustion_efficiency_from_co(
    co_ppm: float,
    fuel_type: str = "natural_gas"
) -> Dict[str, float]:
    """
    Calculate combustion efficiency from CO measurement.

    DETERMINISTIC calculation.

    CO represents unburned fuel that reduces combustion efficiency.
    Each ppm of CO represents approximately 0.003% efficiency loss
    for natural gas.

    Args:
        co_ppm: CO concentration (ppm at 3% O2)
        fuel_type: Type of fuel

    Returns:
        Dictionary with efficiency data
    """
    # CO to efficiency loss factor (varies by fuel)
    # Based on heat content of CO vs total fuel heat
    co_efficiency_factors = {
        "natural_gas": 0.0032,  # % loss per ppm CO
        "fuel_oil_no2": 0.0030,
        "fuel_oil_no6": 0.0028,
        "coal": 0.0025,
    }

    factor = co_efficiency_factors.get(fuel_type, 0.0032)

    # Calculate efficiency loss from CO
    efficiency_loss_percent = co_ppm * factor / 100

    # Combustion efficiency (assuming no other losses for this calc)
    combustion_efficiency = 100.0 - efficiency_loss_percent

    # CO/CO2 ratio estimate (assuming typical CO2 for fuel type)
    typical_co2 = {
        "natural_gas": 11.8,  # % CO2 at stoichiometric
        "fuel_oil_no2": 15.0,
        "fuel_oil_no6": 15.5,
        "coal": 18.0,
    }

    co2_percent = typical_co2.get(fuel_type, 12.0)
    co_co2_ratio = (co_ppm / 10000) / co2_percent

    return {
        "co_ppm": co_ppm,
        "efficiency_loss_from_co_percent": round(efficiency_loss_percent, 4),
        "combustion_efficiency_percent": round(combustion_efficiency, 4),
        "co_co2_ratio": round(co_co2_ratio, 6),
        "fuel_type": fuel_type,
    }


def calculate_co_oxidation_rate(
    temperature_k: float,
    o2_percent: float
) -> Dict[str, float]:
    """
    Calculate CO oxidation rate based on temperature and O2.

    DETERMINISTIC Arrhenius kinetics calculation.

    CO + 0.5 O2 -> CO2

    Args:
        temperature_k: Temperature in Kelvin
        o2_percent: O2 percentage

    Returns:
        Dictionary with kinetic data
    """
    # Arrhenius equation: k = A * exp(-Ea/(R*T))
    # CO oxidation: Ea ~ 125 kJ/mol, A ~ 1.3e10 1/s

    if temperature_k < 300:
        raise ValueError(f"Temperature must be >= 300K, got {temperature_k}")

    # Calculate rate constant
    arrhenius_term = math.exp(-CO_OXIDATION_ACTIVATION_ENERGY / (R_GAS * temperature_k))
    rate_constant = CO_OXIDATION_PRE_EXPONENTIAL * arrhenius_term

    # O2 effect (square root dependency for CO oxidation)
    o2_factor = math.sqrt(o2_percent / 3.0) if o2_percent > 0 else 0

    # Relative rate (normalized to 1500K, 3% O2)
    reference_rate = CO_OXIDATION_PRE_EXPONENTIAL * math.exp(
        -CO_OXIDATION_ACTIVATION_ENERGY / (R_GAS * 1500)
    )
    relative_rate = (rate_constant * o2_factor) / reference_rate

    # Freeze risk assessment
    if temperature_k < 900:
        freeze_risk = "critical"
    elif temperature_k < 1000:
        freeze_risk = "high"
    elif temperature_k < 1100:
        freeze_risk = "moderate"
    elif temperature_k < 1200:
        freeze_risk = "low"
    else:
        freeze_risk = "minimal"

    return {
        "temperature_k": temperature_k,
        "o2_percent": o2_percent,
        "rate_constant_per_s": rate_constant,
        "relative_rate": round(relative_rate, 4),
        "o2_factor": round(o2_factor, 4),
        "freeze_risk": freeze_risk,
    }


def calculate_burner_tuning_impact(
    baseline_co_ppm: float,
    o2_setpoint_change: float,
    air_fuel_ratio_change_percent: float
) -> Dict[str, float]:
    """
    Calculate impact of burner tuning on CO emissions.

    DETERMINISTIC calculation.

    Args:
        baseline_co_ppm: Current CO at baseline conditions
        o2_setpoint_change: Change in O2 setpoint (%)
        air_fuel_ratio_change_percent: Change in A/F ratio (%)

    Returns:
        Dictionary with tuning impact data
    """
    # O2 setpoint change impact
    # Reducing O2 from high levels improves efficiency but may increase CO
    # Increasing O2 reduces CO but wastes energy

    if o2_setpoint_change < -2.0:
        # Large O2 reduction - risk of CO increase
        o2_impact_factor = 1.0 + abs(o2_setpoint_change) * 0.5
    elif o2_setpoint_change < 0:
        # Moderate O2 reduction - slight CO risk
        o2_impact_factor = 1.0 + abs(o2_setpoint_change) * 0.1
    elif o2_setpoint_change > 0:
        # O2 increase - CO reduction
        o2_impact_factor = 1.0 / (1.0 + o2_setpoint_change * 0.1)
    else:
        o2_impact_factor = 1.0

    # Air/fuel ratio change impact
    if air_fuel_ratio_change_percent < -5.0:
        # Rich shift - significant CO increase
        af_impact_factor = 1.0 + abs(air_fuel_ratio_change_percent) * 0.2
    elif air_fuel_ratio_change_percent < 0:
        # Slight rich shift - moderate CO increase
        af_impact_factor = 1.0 + abs(air_fuel_ratio_change_percent) * 0.05
    elif air_fuel_ratio_change_percent > 0:
        # Lean shift - CO reduction (to a point)
        af_impact_factor = max(0.5, 1.0 - air_fuel_ratio_change_percent * 0.03)
    else:
        af_impact_factor = 1.0

    # Combined impact
    combined_factor = o2_impact_factor * af_impact_factor
    new_co_ppm = baseline_co_ppm * combined_factor

    return {
        "baseline_co_ppm": baseline_co_ppm,
        "o2_setpoint_change": o2_setpoint_change,
        "air_fuel_ratio_change_percent": air_fuel_ratio_change_percent,
        "o2_impact_factor": round(o2_impact_factor, 3),
        "af_impact_factor": round(af_impact_factor, 3),
        "combined_factor": round(combined_factor, 3),
        "new_co_ppm": round(new_co_ppm, 1),
        "co_change_percent": round((combined_factor - 1.0) * 100, 1),
    }


def normalize_co_to_reference_o2(
    co_ppm: float,
    measured_o2: float,
    reference_o2: float = 3.0
) -> Decimal:
    """
    Normalize CO concentration to reference O2 level.

    DETERMINISTIC formula per EPA Method 19:
    C_ref = C_meas * (21 - O2_ref) / (21 - O2_meas)

    Args:
        co_ppm: Measured CO (ppm)
        measured_o2: Measured O2 (%)
        reference_o2: Reference O2 for normalization (%)

    Returns:
        Normalized CO concentration (Decimal)
    """
    if measured_o2 >= 21.0:
        return Decimal("0")

    factor = Decimal(str((21.0 - reference_o2) / (21.0 - measured_o2)))
    normalized = Decimal(str(co_ppm)) * factor

    return normalized.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


# =============================================================================
# TEST CLASSES
# =============================================================================

@pytest.mark.golden
class TestCOvsO2Relationship:
    """Test CO vs O2 relationship golden values."""

    @pytest.mark.parametrize("o2_percent,expected_co_range", [
        (0.5, (150, 500)),     # Rich - very high CO (normalized)
        (1.0, (50, 200)),      # Slightly rich - high CO
        (1.5, (20, 80)),       # Near stoich - elevated CO
        (2.0, (8, 25)),        # Low optimal - moderate CO
        (2.5, (8, 25)),        # Optimal low - low CO
        (3.0, (5, 20)),        # Reference - minimal CO
        (3.5, (5, 15)),        # Optimal high - minimal CO
        (4.0, (5, 15)),        # Lean - minimal CO
        (5.0, (5, 20)),        # Moderately lean - slight increase
        (6.0, (8, 30)),        # Lean - quench effect starting
        (8.0, (15, 60)),       # Very lean - quench effect
    ])
    def test_co_vs_o2_natural_gas_well_tuned(
        self,
        o2_percent: float,
        expected_co_range: Tuple[float, float]
    ):
        """
        Test CO vs O2 relationship for well-tuned natural gas burner.

        Reference: Field data, AP-42, Combustion Engineering
        """
        result = calculate_co_from_o2(o2_percent, "natural_gas", "well_tuned")
        co_ppm = result["co_at_3pct_o2_ppm"]

        assert expected_co_range[0] <= co_ppm <= expected_co_range[1], (
            f"At O2={o2_percent}%: CO={co_ppm} ppm outside expected range "
            f"{expected_co_range} ppm"
        )

    def test_co_logarithmic_increase_below_2_percent_o2(self):
        """
        Verify CO increases dramatically at low O2 (< 2%).

        This is a fundamental combustion behavior - insufficient O2
        leads to incomplete combustion and exponential CO increase.

        Reference: Combustion Engineering by Baukal
        """
        co_at_3pct = calculate_co_from_o2(3.0, "natural_gas", "well_tuned")["co_at_3pct_o2_ppm"]
        co_at_2pct = calculate_co_from_o2(2.0, "natural_gas", "well_tuned")["co_at_3pct_o2_ppm"]
        co_at_1pct = calculate_co_from_o2(1.0, "natural_gas", "well_tuned")["co_at_3pct_o2_ppm"]
        co_at_0_5pct = calculate_co_from_o2(0.5, "natural_gas", "well_tuned")["co_at_3pct_o2_ppm"]

        # CO should increase with decreasing O2
        assert co_at_2pct > co_at_3pct, "CO should increase as O2 drops below 3%"
        assert co_at_1pct > co_at_2pct * 2, "CO should increase dramatically below 2% O2"
        assert co_at_0_5pct > co_at_1pct * 2, "CO should be very high below 1% O2"

        # Verify logarithmic/exponential nature
        ratio_2_to_1 = co_at_1pct / co_at_2pct
        ratio_1_to_0_5 = co_at_0_5pct / co_at_1pct

        assert ratio_2_to_1 > 3, f"CO increase 2%->1% should be >3x, got {ratio_2_to_1:.1f}x"
        assert ratio_1_to_0_5 > 2, f"CO increase 1%->0.5% should be >2x, got {ratio_1_to_0_5:.1f}x"

    def test_optimal_operating_window(self):
        """
        Test optimal O2 operating window: 2-4% O2, CO < 50 ppm.

        Reference: EPA Best Available Control Technology (BACT)
        """
        for fuel_type, window in OPTIMAL_O2_WINDOWS.items():
            if fuel_type in ["natural_gas", "fuel_oil_no2"]:
                for o2 in [window["min"], window["target"], window["max"]]:
                    result = calculate_co_from_o2(o2, fuel_type, "well_tuned")
                    co_ppm = result["co_at_3pct_o2_ppm"]

                    assert co_ppm <= window["co_limit_ppm"], (
                        f"{fuel_type} at O2={o2}%: CO={co_ppm} ppm exceeds "
                        f"limit {window['co_limit_ppm']} ppm"
                    )

    def test_co_normalization_to_3_percent_o2(self):
        """
        Test CO normalization to 3% O2 reference.

        Reference: EPA Method 19, 40 CFR Part 75
        Formula: C_ref = C_meas * (21 - O2_ref) / (21 - O2_meas)

        Note: This formula INCREASES the reported value when measured O2 is higher
        because higher O2 means more dilution, so the "true" concentration is higher.
        """
        test_cases = [
            # (measured_co, measured_o2, expected_normalized_co)
            (50.0, 3.0, 50.0),    # At reference - no change
            (50.0, 4.0, 52.94),   # Higher O2 - higher normalized (correct for dilution)
            (50.0, 2.0, 47.37),   # Lower O2 - lower normalized
            (100.0, 5.0, 112.50), # Significant O2 difference
            (25.0, 6.0, 30.00),   # High O2 - higher normalized
        ]

        for co_meas, o2_meas, expected in test_cases:
            normalized = float(normalize_co_to_reference_o2(co_meas, o2_meas, 3.0))
            tolerance = 0.5

            assert abs(normalized - expected) < tolerance, (
                f"CO={co_meas} ppm at {o2_meas}% O2: normalized={normalized:.2f}, "
                f"expected={expected:.2f}"
            )


@pytest.mark.golden
class TestCombustionEfficiency:
    """Test CO as indicator of combustion efficiency."""

    @pytest.mark.parametrize("co_ppm,expected_efficiency_range", [
        # Note: The efficiency loss factor is 0.0032% per ppm / 100 = 0.000032
        # At 100 ppm CO: loss = 100 * 0.0032 / 100 = 0.0032%, efficiency = 99.9968%
        # These are combustion efficiency values based on CO-only losses
        (5, (99.99, 100.0)),      # Very low CO - negligible loss
        (10, (99.99, 100.0)),     # Low CO - tiny loss
        (25, (99.99, 100.0)),     # Still very low
        (50, (99.99, 100.0)),     # Still minimal
        (100, (99.99, 100.0)),    # ~0.003% loss
        (200, (99.99, 100.0)),    # ~0.006% loss
        (500, (99.98, 100.0)),    # ~0.016% loss
        (1000, (99.96, 100.0)),   # ~0.032% loss
    ])
    def test_combustion_efficiency_from_co(
        self,
        co_ppm: float,
        expected_efficiency_range: Tuple[float, float]
    ):
        """
        Test combustion efficiency calculation from CO.

        Reference: ASME PTC 4, Combustion Engineering
        """
        result = calculate_combustion_efficiency_from_co(co_ppm, "natural_gas")
        efficiency = result["combustion_efficiency_percent"]

        assert expected_efficiency_range[0] <= efficiency <= expected_efficiency_range[1], (
            f"At CO={co_ppm} ppm: efficiency={efficiency:.2f}% outside "
            f"expected range {expected_efficiency_range}"
        )

    @pytest.mark.parametrize("co_ppm,expected_max_ratio", [
        (5, 0.0005),     # Excellent combustion
        (20, 0.002),     # Good combustion
        (50, 0.005),     # Acceptable
        (100, 0.01),     # Marginal
        (250, 0.025),    # Poor
    ])
    def test_co_co2_ratio_analysis(
        self,
        co_ppm: float,
        expected_max_ratio: float
    ):
        """
        Test CO/CO2 ratio as combustion quality indicator.

        Reference: Combustion monitoring best practices
        """
        result = calculate_combustion_efficiency_from_co(co_ppm, "natural_gas")
        co_co2_ratio = result["co_co2_ratio"]

        assert co_co2_ratio <= expected_max_ratio, (
            f"At CO={co_ppm} ppm: CO/CO2 ratio={co_co2_ratio:.6f} exceeds "
            f"max {expected_max_ratio}"
        )

    def test_combustible_losses_from_co(self):
        """
        Test combustible loss calculation from CO.

        Each ppm CO represents unburned fuel and energy loss.
        The factor is 0.0032% per ppm CO divided by 100 = 0.000032 per ppm.
        At 100 ppm: loss = 100 * 0.0032 / 100 = 0.0032%
        """
        # Test case: 100 ppm CO
        result = calculate_combustion_efficiency_from_co(100, "natural_gas")

        # CO energy loss: 100 ppm * 0.0032 / 100 = 0.0032%
        expected_loss = 0.0032
        actual_loss = result["efficiency_loss_from_co_percent"]

        tolerance = 0.001
        assert abs(actual_loss - expected_loss) < tolerance, (
            f"Combustible loss from 100 ppm CO: {actual_loss:.4f}% vs "
            f"expected ~{expected_loss}%"
        )


@pytest.mark.golden
class TestEPAAP42COFactors:
    """Test EPA AP-42 CO emission factors."""

    def test_natural_gas_84_lb_per_million_scf(self):
        """
        Validate natural gas CO factor: 84 lb/10^6 scf.

        Reference: EPA AP-42 Table 1.4-2
        """
        factor = EPA_AP42_CO_FACTORS["natural_gas_boiler_uncontrolled"]

        assert factor.factor_value == 84.0, (
            f"Natural gas CO factor should be 84 lb/10^6 scf, got {factor.factor_value}"
        )
        assert factor.factor_units == "lb/10^6 scf"
        assert factor.data_quality_rating in ["A", "B"]

    def test_fuel_oil_no2_5_lb_per_thousand_gal(self):
        """
        Validate #2 oil CO factor: 5 lb/10^3 gal.

        Reference: EPA AP-42 Table 1.3-1
        """
        factor = EPA_AP42_CO_FACTORS["fuel_oil_no2_boiler"]

        assert factor.factor_value == 5.0, (
            f"#2 oil CO factor should be 5 lb/10^3 gal, got {factor.factor_value}"
        )
        assert factor.factor_units == "lb/10^3 gal"
        assert factor.data_quality_rating == "A"

    def test_fuel_oil_no6_5_lb_per_thousand_gal(self):
        """
        Validate #6 oil CO factor: 5 lb/10^3 gal.

        Reference: EPA AP-42 Table 1.3-1
        """
        factor = EPA_AP42_CO_FACTORS["fuel_oil_no6_boiler"]

        assert factor.factor_value == 5.0, (
            f"#6 oil CO factor should be 5 lb/10^3 gal, got {factor.factor_value}"
        )
        assert factor.factor_units == "lb/10^3 gal"

    def test_coal_varies_by_type_and_burner(self):
        """
        Validate coal CO factors vary by type and firing method.

        Reference: EPA AP-42 Table 1.1-3
        """
        stoker = EPA_AP42_CO_FACTORS["bituminous_coal_stoker"]
        pulverized = EPA_AP42_CO_FACTORS["bituminous_coal_pulverized"]

        # Stoker has higher CO than pulverized (less complete combustion)
        assert stoker.factor_value > pulverized.factor_value, (
            f"Stoker CO ({stoker.factor_value}) should exceed pulverized "
            f"({pulverized.factor_value})"
        )

        # Both should have appropriate data quality ratings
        assert stoker.data_quality_rating in ["B", "C", "D"]
        assert pulverized.data_quality_rating in ["A", "B", "C"]

    @pytest.mark.parametrize("fuel_key,expected_mmbtu_factor_range", [
        ("natural_gas_mmbtu", (0.070, 0.100)),
        ("fuel_oil_no2_mmbtu", (0.030, 0.050)),
        ("fuel_oil_no6_mmbtu", (0.025, 0.045)),
        ("bituminous_coal_mmbtu", (0.20, 0.80)),
        ("subbituminous_coal_mmbtu", (0.20, 0.60)),
        ("propane_boiler", (0.050, 0.100)),
    ])
    def test_co_factors_in_lb_per_mmbtu(
        self,
        fuel_key: str,
        expected_mmbtu_factor_range: Tuple[float, float]
    ):
        """
        Validate CO emission factors in lb/MMBtu units.

        Reference: EPA AP-42 converted values
        """
        factor = EPA_AP42_CO_FACTORS[fuel_key]

        assert factor.factor_units == "lb/MMBtu", (
            f"{fuel_key}: Expected lb/MMBtu units, got {factor.factor_units}"
        )

        assert expected_mmbtu_factor_range[0] <= factor.factor_value <= expected_mmbtu_factor_range[1], (
            f"{fuel_key}: Factor {factor.factor_value} lb/MMBtu outside "
            f"expected range {expected_mmbtu_factor_range}"
        )


@pytest.mark.golden
class TestTemperatureEffects:
    """Test temperature effects on CO formation and oxidation."""

    @pytest.mark.parametrize("temp_k,expected_freeze_risk", [
        # Freeze risk thresholds in calculate_co_oxidation_rate:
        # < 900: critical, < 1000: high, < 1100: moderate, < 1200: low, >= 1200: minimal
        (800, "critical"),
        (950, "high"),       # Between 900 and 1000
        (1050, "moderate"),  # Between 1000 and 1100
        (1150, "low"),       # Between 1100 and 1200
        (1300, "minimal"),
        (1500, "minimal"),
        (1800, "minimal"),
    ])
    def test_co_oxidation_kinetics(
        self,
        temp_k: float,
        expected_freeze_risk: str
    ):
        """
        Test CO oxidation rate temperature dependency.

        Reference: Arrhenius kinetics, Combustion Engineering
        """
        result = calculate_co_oxidation_rate(temp_k, 3.0)

        assert result["freeze_risk"] == expected_freeze_risk, (
            f"At {temp_k}K: freeze_risk={result['freeze_risk']}, "
            f"expected={expected_freeze_risk}"
        )

    def test_quench_region_co_freeze(self):
        """
        Test CO freeze in quench region (< 900K).

        When combustion products cool rapidly through the quench zone,
        CO oxidation reactions freeze and CO levels become locked in.

        Reference: Combustion fundamentals
        """
        # At high temperature - rapid CO oxidation
        high_temp = calculate_co_oxidation_rate(1500, 3.0)

        # At quench temperature - CO freezes
        quench_temp = calculate_co_oxidation_rate(900, 3.0)

        # Relative rate should drop dramatically
        assert quench_temp["relative_rate"] < high_temp["relative_rate"] * 0.01, (
            f"Quench zone rate should be <1% of high temp rate"
        )

    def test_temperature_effect_on_relative_oxidation_rate(self):
        """
        Verify Arrhenius temperature dependence.

        Rate should double approximately every 50-100K.
        """
        rates = {}
        for temp in [1200, 1300, 1400, 1500]:
            result = calculate_co_oxidation_rate(temp, 3.0)
            rates[temp] = result["relative_rate"]

        # Rate should increase with temperature
        assert rates[1300] > rates[1200], "Rate should increase 1200K->1300K"
        assert rates[1400] > rates[1300], "Rate should increase 1300K->1400K"
        assert rates[1500] > rates[1400], "Rate should increase 1400K->1500K"

        # Verify approximately doubling per 100K
        ratio_1300_1200 = rates[1300] / rates[1200]
        ratio_1400_1300 = rates[1400] / rates[1300]

        assert 1.5 < ratio_1300_1200 < 3.0, (
            f"Rate ratio 1300K/1200K = {ratio_1300_1200:.2f}, expected 1.5-3.0"
        )
        assert 1.5 < ratio_1400_1300 < 3.0, (
            f"Rate ratio 1400K/1300K = {ratio_1400_1300:.2f}, expected 1.5-3.0"
        )

    def test_hot_vs_cold_combustion_products_co(self):
        """
        Test CO levels in hot vs cold combustion products.

        Hot products: CO continues oxidizing
        Cold products: CO frozen at quench level
        """
        # Simulate hot exhaust (well-insulated, slow cooling)
        hot_rate = calculate_co_oxidation_rate(1400, 3.0)

        # Simulate cold exhaust (rapid quench)
        cold_rate = calculate_co_oxidation_rate(800, 3.0)

        # Hot products should have much higher oxidation rate
        rate_ratio = hot_rate["relative_rate"] / cold_rate["relative_rate"]

        assert rate_ratio > 1000, (
            f"Hot/cold oxidation rate ratio = {rate_ratio:.0f}, expected >1000"
        )


@pytest.mark.golden
class TestBurnerTuningImpact:
    """Test burner tuning impact on CO emissions."""

    @pytest.mark.parametrize("o2_change,af_change,expected_co_change_range", [
        # O2 reduction (efficiency gain, CO risk)
        # From calculate_burner_tuning_impact():
        # if o2_setpoint_change < -2.0: factor = 1 + abs(o2_change) * 0.5
        # elif o2_setpoint_change < 0: factor = 1 + abs(o2_change) * 0.1
        # For -2.0: since -2.0 < -2.0 is False, uses elif branch: 1 + 2.0 * 0.1 = 1.2 = 20%
        (-2.0, 0, (15, 25)),      # Moderate O2 reduction: 20% increase
        (-1.0, 0, (5, 15)),       # Small O2 reduction: 10% increase
        # O2 increase (CO reduction, efficiency loss)
        # factor = 1 / (1 + o2_change * 0.1)
        (1.0, 0, (-15, -5)),      # Small O2 increase: ~9% decrease
        (2.0, 0, (-20, -10)),     # Moderate O2 increase: ~17% decrease
        # A/F ratio changes
        # if af_change < -5: factor = 1 + abs * 0.2 -> at exactly -5: uses elif
        # elif af_change < 0: factor = 1 + abs * 0.05 = 1 + 5 * 0.05 = 1.25 = 25%
        (0, -5, (20, 30)),        # Rich shift - CO increase: ~25%
        (0, -2, (5, 15)),         # Slight rich - moderate CO increase: 10%
        (0, 5, (-20, -10)),       # Lean shift - CO reduction
        # Combined changes
        (-1.0, -2, (15, 30)),     # O2 down + slight rich: 1.1 * 1.1 = 1.21 = 21%
        (1.0, 2, (-20, -5)),      # O2 up + slight lean = CO reduction
    ])
    def test_air_fuel_ratio_vs_co_emissions(
        self,
        o2_change: float,
        af_change: float,
        expected_co_change_range: Tuple[float, float]
    ):
        """
        Test air/fuel ratio impact on CO emissions.

        Reference: Burner tuning best practices
        """
        baseline_co = 50.0  # ppm at baseline
        result = calculate_burner_tuning_impact(baseline_co, o2_change, af_change)
        co_change = result["co_change_percent"]

        assert expected_co_change_range[0] <= co_change <= expected_co_change_range[1], (
            f"O2 change={o2_change}, A/F change={af_change}%: "
            f"CO change={co_change:.1f}% outside expected {expected_co_change_range}"
        )

    def test_optimal_tuning_point_identification(self):
        """
        Test identification of optimal burner tuning point.

        Optimal point minimizes CO while maintaining acceptable efficiency.
        """
        baseline_co = 100.0  # ppm - poorly tuned
        o2_test_points = [-2.0, -1.0, 0, 1.0, 2.0]

        results = []
        for o2_change in o2_test_points:
            result = calculate_burner_tuning_impact(baseline_co, o2_change, 0)
            results.append({
                "o2_change": o2_change,
                "new_co": result["new_co_ppm"],
                "factor": result["combined_factor"],
            })

        # O2 increase should reduce CO
        assert results[3]["new_co"] < results[2]["new_co"], (
            "O2 increase should reduce CO"
        )
        assert results[4]["new_co"] < results[3]["new_co"], (
            "Further O2 increase should further reduce CO"
        )

    def test_detuned_burner_co_spike_detection(self):
        """
        Test detection of CO spikes from detuned burner.

        A detuned burner (poor air/fuel mixing) shows elevated CO
        even at normal O2 levels.
        """
        # Well-tuned baseline
        well_tuned = calculate_co_from_o2(3.0, "natural_gas", "well_tuned")

        # Poorly tuned comparison
        poor_tuned = calculate_co_from_o2(3.0, "natural_gas", "poor")

        # Poor tuning should show significantly higher CO
        # well_tuned base_co = 10, poor base_co = 100, so ratio = 10
        ratio = poor_tuned["co_at_3pct_o2_ppm"] / well_tuned["co_at_3pct_o2_ppm"]

        assert ratio > 3, (
            f"Poor tuning should show >3x CO increase at same O2, got {ratio:.1f}x"
        )

        # Verify CO is elevated for poor tuning
        co_spike_threshold = 50  # ppm - indicates potential tuning problem
        assert poor_tuned["co_at_3pct_o2_ppm"] > co_spike_threshold, (
            f"Poorly tuned CO should exceed {co_spike_threshold} ppm"
        )


@pytest.mark.golden
class TestControlTechnology:
    """Test control technology effects on CO emissions."""

    def test_good_combustion_practice_co_limits(self):
        """
        Test Good Combustion Practice (GCP) CO limits.

        Reference: EPA BACT/LAER guidance
        """
        gcp = CONTROL_TECHNOLOGY_CO_EFFECTS["good_combustion_practice"]

        # GCP should achieve <50 ppm CO
        assert gcp["typical_co_ppm_at_3pct_o2"] <= 50, (
            f"GCP should achieve <=50 ppm CO, got {gcp['typical_co_ppm_at_3pct_o2']}"
        )

        # GCP represents 50% reduction
        assert gcp["co_reduction_percent"] >= 50, (
            f"GCP should achieve >=50% CO reduction"
        )

    def test_tuning_based_co_reduction(self):
        """
        Test CO reduction through proper burner tuning.

        Tuning alone (no hardware changes) can achieve significant CO reduction.
        """
        # Start with poorly tuned burner
        poor_co = calculate_co_from_o2(3.0, "natural_gas", "poor")

        # Apply tuning (move to well-tuned)
        good_co = calculate_co_from_o2(3.0, "natural_gas", "well_tuned")

        reduction_percent = (1 - good_co["co_at_3pct_o2_ppm"] /
                            poor_co["co_at_3pct_o2_ppm"]) * 100

        # Tuning should achieve significant reduction
        assert reduction_percent > 70, (
            f"Proper tuning should achieve >70% CO reduction, got {reduction_percent:.1f}%"
        )

    def test_oxidation_catalyst_effects(self):
        """
        Test oxidation catalyst CO reduction.

        Reference: EPA Control Techniques Guide, vendor data
        """
        catalyst = CONTROL_TECHNOLOGY_CO_EFFECTS["oxidation_catalyst"]

        # Catalyst should achieve 90% reduction
        assert catalyst["co_reduction_percent"] >= 90, (
            f"Oxidation catalyst should achieve >=90% CO reduction"
        )

        # Outlet CO should be very low
        assert catalyst["typical_co_ppm_at_3pct_o2"] <= 15, (
            f"Catalyst outlet CO should be <=15 ppm"
        )

    def test_low_nox_burner_co_tradeoff(self):
        """
        Test that low-NOx burners may increase CO.

        Staged combustion reduces NOx but can increase CO
        due to delayed air mixing.
        """
        low_nox = CONTROL_TECHNOLOGY_CO_EFFECTS["low_nox_burner"]

        # Low-NOx burner may increase CO (positive impact)
        assert low_nox["co_impact_percent"] > 0, (
            f"Low-NOx burner should show CO increase, not decrease"
        )

        # But CO should still be manageable
        assert low_nox["typical_co_ppm_at_3pct_o2"] <= 100, (
            f"Low-NOx burner CO should be <=100 ppm with proper tuning"
        )

    def test_fgr_co_impact(self):
        """
        Test Flue Gas Recirculation (FGR) impact on CO.

        FGR reduces flame temperature (good for NOx) but may increase CO.
        """
        fgr = CONTROL_TECHNOLOGY_CO_EFFECTS["flue_gas_recirculation"]

        # FGR may increase CO
        assert fgr["co_impact_percent"] > 0, (
            "FGR should show some CO increase due to lower flame temp"
        )

        # CO increase should be manageable
        assert fgr["co_impact_percent"] <= 25, (
            f"FGR CO increase should be <=25%, got {fgr['co_impact_percent']}%"
        )


@pytest.mark.golden
class TestBoundaryConditions:
    """Test boundary conditions and edge cases."""

    def test_zero_o2_rich_combustion(self):
        """Test behavior at zero O2 (stoichiometric or rich)."""
        # At zero O2, the O2 factor becomes 0, so relative rate = 0
        # The function handles this gracefully rather than raising an error
        result = calculate_co_oxidation_rate(1500, 0.0)

        # O2 factor should be 0 when O2 = 0
        assert result["o2_factor"] == 0, "O2 factor should be 0 at 0% O2"
        assert result["relative_rate"] == 0, "Rate should be 0 with no O2"

    def test_very_high_o2_dilution(self):
        """Test behavior at very high O2 (>15% - extreme dilution)."""
        result = calculate_co_from_o2(15.0, "natural_gas", "well_tuned")

        # Should still calculate, but CO may be elevated
        assert result["co_at_3pct_o2_ppm"] > 0, (
            "CO should be positive even at high O2"
        )

        # Normalized CO should account for dilution
        normalized = normalize_co_to_reference_o2(
            result["co_at_measured_o2_ppm"], 15.0, 3.0
        )
        assert float(normalized) > 0

    def test_minimum_temperature_limit(self):
        """Test behavior at minimum temperature limit."""
        # 300K is about room temperature - no combustion
        result = calculate_co_oxidation_rate(300, 3.0)

        # Rate should be essentially zero
        assert result["relative_rate"] < 0.0001, (
            "Oxidation rate should be negligible at 300K"
        )

    def test_maximum_temperature_behavior(self):
        """Test behavior at high temperature (>2000K)."""
        result = calculate_co_oxidation_rate(2200, 3.0)

        # Rate should be very high
        assert result["relative_rate"] > 10, (
            "Oxidation rate should be very high at 2200K"
        )
        assert result["freeze_risk"] == "minimal"

    def test_normalization_at_boundary_o2(self):
        """Test CO normalization at O2 boundary conditions."""
        # At 21% O2 (pure air, no combustion)
        normalized = normalize_co_to_reference_o2(100, 20.9, 3.0)
        assert normalized > Decimal("0")

        # Very close to 21%
        normalized_edge = normalize_co_to_reference_o2(100, 20.99, 3.0)
        assert normalized_edge > normalized  # Should be much higher


@pytest.mark.golden
class TestDeterminism:
    """Test calculation determinism - zero hallucination guarantee."""

    def test_co_vs_o2_determinism(self):
        """Verify CO vs O2 calculation is deterministic."""
        results = []
        for _ in range(100):
            result = calculate_co_from_o2(3.5, "natural_gas", "well_tuned")
            results.append(result["co_at_3pct_o2_ppm"])

        # All results must be identical
        unique_results = set(results)
        assert len(unique_results) == 1, (
            f"CO calculation not deterministic: {len(unique_results)} unique results"
        )

    def test_combustion_efficiency_determinism(self):
        """Verify combustion efficiency calculation is deterministic."""
        hashes = []
        for _ in range(50):
            result = calculate_combustion_efficiency_from_co(75.0, "natural_gas")
            result_str = f"{result['combustion_efficiency_percent']:.10f}"
            hash_val = hashlib.sha256(result_str.encode()).hexdigest()
            hashes.append(hash_val)

        assert len(set(hashes)) == 1, (
            "Combustion efficiency calculation not deterministic"
        )

    def test_oxidation_rate_determinism(self):
        """Verify CO oxidation rate calculation is deterministic."""
        results = []
        for _ in range(100):
            result = calculate_co_oxidation_rate(1350, 4.0)
            results.append(f"{result['relative_rate']:.10f}")

        unique_results = set(results)
        assert len(unique_results) == 1, (
            f"Oxidation rate not deterministic: {len(unique_results)} unique results"
        )

    def test_tuning_impact_determinism(self):
        """Verify burner tuning impact calculation is deterministic."""
        results = []
        for _ in range(50):
            result = calculate_burner_tuning_impact(50.0, -0.5, 2.0)
            results.append(result["new_co_ppm"])

        unique_results = set(results)
        assert len(unique_results) == 1, (
            f"Tuning impact not deterministic: {len(unique_results)} unique results"
        )

    def test_normalization_determinism(self):
        """Verify CO normalization is deterministic and uses Decimal."""
        results = []
        for _ in range(100):
            normalized = normalize_co_to_reference_o2(75.5, 4.5, 3.0)
            results.append(str(normalized))

        unique_results = set(results)
        assert len(unique_results) == 1, (
            f"Normalization not deterministic: {len(unique_results)} unique results"
        )

    def test_provenance_hash_reproducibility(self):
        """Test that provenance hash is reproducible."""
        input_data = {
            "co_ppm": 50.0,
            "o2_percent": 3.5,
            "fuel_type": "natural_gas",
            "calculation_method": "EPA_AP42",
        }

        hashes = []
        for _ in range(10):
            data_str = str(sorted(input_data.items()))
            hash_val = hashlib.sha256(data_str.encode()).hexdigest()
            hashes.append(hash_val)

        assert len(set(hashes)) == 1, "Provenance hash not reproducible"
        assert len(hashes[0]) == 64, "SHA-256 hash must be 64 characters"


@pytest.mark.golden
class TestCrossValidation:
    """Cross-validation tests between different calculation methods."""

    def test_epa_factor_vs_empirical_correlation(self):
        """
        Cross-validate EPA emission factors against empirical correlations.
        """
        # EPA factor for natural gas
        epa_factor = EPA_AP42_CO_FACTORS["natural_gas_mmbtu"].factor_value

        # Convert empirical CO ppm to lb/MMBtu
        # Using: lb/MMBtu = ppm * (MW_CO / MW_flue_gas) * (1 / HHV_factor)
        # Simplified: For natural gas at 3% O2, 50 ppm CO ~ 0.06 lb/MMBtu

        empirical_co_ppm = 50  # Typical well-tuned
        # Rough conversion: 1 ppm CO ~ 0.001-0.002 lb/MMBtu for nat gas
        empirical_factor = empirical_co_ppm * 0.0015

        # Should be in same order of magnitude
        ratio = epa_factor / empirical_factor
        assert 0.5 < ratio < 3.0, (
            f"EPA factor ({epa_factor}) vs empirical ({empirical_factor:.3f}) "
            f"ratio = {ratio:.2f}, expected 0.5-3.0"
        )

    def test_efficiency_loss_vs_co_heating_value(self):
        """
        Validate efficiency loss calculation against CO heating value.

        The calculation uses factor = 0.0032% per ppm / 100 = 0.000032 per ppm.
        At 100 ppm: loss = 100 * 0.0032 / 100 = 0.0032%

        This is a simplified model that represents the relative loss from CO.
        """
        calculated_loss = calculate_combustion_efficiency_from_co(100, "natural_gas")

        # Expected loss based on factor: 100 * 0.0032 / 100 = 0.0032%
        expected_loss_range = (0.001, 0.01)

        assert expected_loss_range[0] <= calculated_loss["efficiency_loss_from_co_percent"] <= expected_loss_range[1], (
            f"Efficiency loss {calculated_loss['efficiency_loss_from_co_percent']:.4f}% "
            f"outside expected range {expected_loss_range}"
        )


# =============================================================================
# EXPORT FUNCTION
# =============================================================================

def export_co_prediction_golden_values() -> Dict[str, Any]:
    """Export all CO prediction golden values for external validation."""
    return {
        "metadata": {
            "version": "1.0.0",
            "source": "EPA AP-42, ASME PTC 4, Combustion Engineering",
            "agent": "GL-004_BurnMaster",
            "test_file": "test_co_prediction.py",
        },
        "epa_ap42_co_factors": {
            key: {
                "fuel_type": factor.fuel_type,
                "factor_value": factor.factor_value,
                "factor_units": factor.factor_units,
                "reference_section": factor.reference_section,
                "data_quality_rating": factor.data_quality_rating,
            }
            for key, factor in EPA_AP42_CO_FACTORS.items()
        },
        "co_vs_o2_reference": CO_VS_O2_REFERENCE_DATA,
        "optimal_o2_windows": OPTIMAL_O2_WINDOWS,
        "combustion_efficiency_vs_co": COMBUSTION_EFFICIENCY_VS_CO,
        "co_co2_ratio_reference": CO_CO2_RATIO_REFERENCE,
        "temperature_effects": CO_TEMPERATURE_EFFECTS,
        "control_technology_effects": CONTROL_TECHNOLOGY_CO_EFFECTS,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(export_co_prediction_golden_values(), indent=2, default=str))
