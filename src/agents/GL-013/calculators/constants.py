"""
GL-013 PREDICTMAINT - Physical Constants and Parameters

This module defines all physical constants, material parameters, and
equipment-specific values used in predictive maintenance calculations.

All values are sourced from authoritative standards and references:
- NIST Standard Reference Database
- IEEE Standards (C57.91, C57.96)
- ISO Standards (10816, 13373, 13374)
- ASME Standards
- IEC Standards

Author: GL-CalculatorEngineer
Version: 1.0.0
License: Proprietary - GreenLang
"""

from decimal import Decimal
from dataclasses import dataclass, field
from typing import Dict, Tuple, Final
from enum import Enum, auto


# =============================================================================
# FUNDAMENTAL PHYSICAL CONSTANTS
# Source: NIST CODATA 2018
# =============================================================================

# Boltzmann constant (J/K)
BOLTZMANN_CONSTANT: Final[Decimal] = Decimal("1.380649e-23")

# Boltzmann constant (eV/K) - commonly used in Arrhenius calculations
BOLTZMANN_CONSTANT_EV: Final[Decimal] = Decimal("8.617333262e-5")

# Absolute zero temperature offset (Celsius to Kelvin)
KELVIN_OFFSET: Final[Decimal] = Decimal("273.15")

# Standard reference temperature (K) - typically 298.15 K (25 C)
STANDARD_TEMP_K: Final[Decimal] = Decimal("298.15")

# Standard gravity (m/s^2)
STANDARD_GRAVITY: Final[Decimal] = Decimal("9.80665")

# Pi to 50 decimal places for precision calculations
PI: Final[Decimal] = Decimal(
    "3.14159265358979323846264338327950288419716939937510"
)

# Euler's number to 50 decimal places
E: Final[Decimal] = Decimal(
    "2.71828182845904523536028747135266249775724709369995"
)


# =============================================================================
# WEIBULL DISTRIBUTION PARAMETERS BY EQUIPMENT TYPE
# Source: IEEE 493-2007, Reliability Data for Equipment
# =============================================================================

@dataclass(frozen=True)
class WeibullParameters:
    """
    Weibull distribution parameters for reliability modeling.

    The Weibull distribution is defined by:
    R(t) = exp(-(t/eta)^beta)

    Where:
        beta (shape): Failure mode indicator
            - beta < 1: Infant mortality (decreasing failure rate)
            - beta = 1: Random failures (constant failure rate)
            - beta > 1: Wear-out failures (increasing failure rate)
        eta (scale): Characteristic life (time at which 63.2% have failed)
        gamma (location): Failure-free period (minimum life)

    Reference: Abernethy, R.B., "The New Weibull Handbook", 5th Ed.
    """
    beta: Decimal  # Shape parameter (dimensionless)
    eta: Decimal   # Scale parameter (hours)
    gamma: Decimal = Decimal("0")  # Location parameter (hours)
    description: str = ""
    source: str = ""


# Equipment-specific Weibull parameters
# Source: IEEE 493-2007 Gold Book, OREDA Handbook 2015
WEIBULL_PARAMETERS: Dict[str, WeibullParameters] = {
    # Rotating Equipment
    "motor_ac_induction_small": WeibullParameters(
        beta=Decimal("2.5"),
        eta=Decimal("87600"),  # ~10 years
        gamma=Decimal("0"),
        description="AC induction motor < 200 HP",
        source="IEEE 493-2007"
    ),
    "motor_ac_induction_large": WeibullParameters(
        beta=Decimal("2.2"),
        eta=Decimal("131400"),  # ~15 years
        gamma=Decimal("8760"),  # 1 year burn-in
        description="AC induction motor >= 200 HP",
        source="IEEE 493-2007"
    ),
    "motor_dc": WeibullParameters(
        beta=Decimal("1.8"),
        eta=Decimal("70080"),  # ~8 years
        gamma=Decimal("0"),
        description="DC motor all sizes",
        source="IEEE 493-2007"
    ),
    "pump_centrifugal": WeibullParameters(
        beta=Decimal("1.9"),
        eta=Decimal("52560"),  # ~6 years
        gamma=Decimal("0"),
        description="Centrifugal pump",
        source="OREDA 2015"
    ),
    "pump_reciprocating": WeibullParameters(
        beta=Decimal("1.6"),
        eta=Decimal("35040"),  # ~4 years
        gamma=Decimal("0"),
        description="Reciprocating pump",
        source="OREDA 2015"
    ),
    "compressor_centrifugal": WeibullParameters(
        beta=Decimal("2.0"),
        eta=Decimal("61320"),  # ~7 years
        gamma=Decimal("0"),
        description="Centrifugal compressor",
        source="OREDA 2015"
    ),
    "compressor_reciprocating": WeibullParameters(
        beta=Decimal("1.7"),
        eta=Decimal("43800"),  # ~5 years
        gamma=Decimal("0"),
        description="Reciprocating compressor",
        source="OREDA 2015"
    ),
    "fan_axial": WeibullParameters(
        beta=Decimal("2.3"),
        eta=Decimal("96360"),  # ~11 years
        gamma=Decimal("0"),
        description="Axial fan",
        source="IEEE 493-2007"
    ),
    "fan_centrifugal": WeibullParameters(
        beta=Decimal("2.4"),
        eta=Decimal("105120"),  # ~12 years
        gamma=Decimal("0"),
        description="Centrifugal fan",
        source="IEEE 493-2007"
    ),

    # Electrical Equipment
    "transformer_power_dry": WeibullParameters(
        beta=Decimal("3.0"),
        eta=Decimal("262800"),  # ~30 years
        gamma=Decimal("43800"),  # 5 year minimum
        description="Dry-type power transformer",
        source="IEEE C57.91"
    ),
    "transformer_power_oil": WeibullParameters(
        beta=Decimal("3.5"),
        eta=Decimal("350400"),  # ~40 years
        gamma=Decimal("87600"),  # 10 year minimum
        description="Oil-filled power transformer",
        source="IEEE C57.91"
    ),
    "circuit_breaker_lv": WeibullParameters(
        beta=Decimal("1.5"),
        eta=Decimal("175200"),  # ~20 years
        gamma=Decimal("0"),
        description="Low voltage circuit breaker",
        source="IEEE 493-2007"
    ),
    "circuit_breaker_mv": WeibullParameters(
        beta=Decimal("1.8"),
        eta=Decimal("219000"),  # ~25 years
        gamma=Decimal("0"),
        description="Medium voltage circuit breaker",
        source="IEEE 493-2007"
    ),
    "cable_xlpe": WeibullParameters(
        beta=Decimal("4.0"),
        eta=Decimal("350400"),  # ~40 years
        gamma=Decimal("87600"),
        description="XLPE insulated cable",
        source="IEEE 400.2"
    ),

    # Bearings - Based on L10 life
    "bearing_ball": WeibullParameters(
        beta=Decimal("1.1"),  # Typical for bearings
        eta=Decimal("43800"),  # ~5 years nominal
        gamma=Decimal("0"),
        description="Ball bearing (L10 life basis)",
        source="ISO 281:2007"
    ),
    "bearing_roller_cylindrical": WeibullParameters(
        beta=Decimal("1.1"),
        eta=Decimal("52560"),  # ~6 years nominal
        gamma=Decimal("0"),
        description="Cylindrical roller bearing",
        source="ISO 281:2007"
    ),
    "bearing_roller_spherical": WeibullParameters(
        beta=Decimal("1.1"),
        eta=Decimal("61320"),  # ~7 years nominal
        gamma=Decimal("0"),
        description="Spherical roller bearing",
        source="ISO 281:2007"
    ),

    # Valves
    "valve_control": WeibullParameters(
        beta=Decimal("1.4"),
        eta=Decimal("70080"),  # ~8 years
        gamma=Decimal("0"),
        description="Control valve",
        source="OREDA 2015"
    ),
    "valve_safety": WeibullParameters(
        beta=Decimal("1.3"),
        eta=Decimal("87600"),  # ~10 years
        gamma=Decimal("0"),
        description="Safety relief valve",
        source="OREDA 2015"
    ),

    # Heat Exchangers
    "heat_exchanger_shell_tube": WeibullParameters(
        beta=Decimal("2.5"),
        eta=Decimal("175200"),  # ~20 years
        gamma=Decimal("17520"),
        description="Shell and tube heat exchanger",
        source="OREDA 2015"
    ),
    "heat_exchanger_plate": WeibullParameters(
        beta=Decimal("2.0"),
        eta=Decimal("131400"),  # ~15 years
        gamma=Decimal("8760"),
        description="Plate heat exchanger",
        source="OREDA 2015"
    ),
}


# =============================================================================
# EXPONENTIAL FAILURE RATE DATA (LAMBDA)
# Source: MIL-HDBK-217F, IEEE 493-2007
# =============================================================================

# Failure rates in failures per million hours (FPMH)
FAILURE_RATES_FPMH: Dict[str, Decimal] = {
    # Electronics
    "capacitor_electrolytic": Decimal("1.5"),
    "capacitor_ceramic": Decimal("0.01"),
    "capacitor_film": Decimal("0.02"),
    "resistor_carbon": Decimal("0.003"),
    "resistor_wirewound": Decimal("0.02"),
    "diode_general": Decimal("0.02"),
    "diode_power": Decimal("0.1"),
    "transistor_bjt": Decimal("0.02"),
    "transistor_mosfet": Decimal("0.05"),
    "ic_digital_simple": Decimal("0.05"),
    "ic_digital_complex": Decimal("0.5"),
    "ic_analog": Decimal("0.1"),
    "ic_microprocessor": Decimal("1.0"),
    "connector_general": Decimal("0.02"),
    "connector_power": Decimal("0.05"),
    "relay_general": Decimal("0.2"),
    "relay_solid_state": Decimal("0.1"),
    "switch_toggle": Decimal("0.1"),
    "switch_pushbutton": Decimal("0.2"),

    # Sensors
    "sensor_temperature_rtd": Decimal("0.5"),
    "sensor_temperature_tc": Decimal("0.3"),
    "sensor_pressure_strain": Decimal("1.0"),
    "sensor_pressure_piezo": Decimal("0.5"),
    "sensor_flow_magnetic": Decimal("2.0"),
    "sensor_flow_ultrasonic": Decimal("1.5"),
    "sensor_vibration_piezo": Decimal("1.0"),
    "sensor_vibration_mems": Decimal("2.0"),
    "sensor_level_float": Decimal("3.0"),
    "sensor_level_radar": Decimal("1.0"),

    # Mechanical Components
    "seal_lip": Decimal("50.0"),
    "seal_mechanical": Decimal("100.0"),
    "gasket_elastomer": Decimal("20.0"),
    "coupling_flexible": Decimal("5.0"),
    "coupling_rigid": Decimal("2.0"),
    "belt_v": Decimal("200.0"),
    "belt_timing": Decimal("100.0"),
    "chain_roller": Decimal("50.0"),
    "gearbox_helical": Decimal("10.0"),
    "gearbox_planetary": Decimal("15.0"),
}


# =============================================================================
# ARRHENIUS ACTIVATION ENERGIES
# Source: IEEE C57.91, IEEE C57.96, Various Material Handbooks
# =============================================================================

@dataclass(frozen=True)
class ArrheniusParameters:
    """
    Arrhenius equation parameters for thermal aging.

    The Arrhenius equation: k = A * exp(-Ea / (k_B * T))

    Where:
        Ea: Activation energy (eV)
        A: Pre-exponential factor
        T: Absolute temperature (K)
        k_B: Boltzmann constant (eV/K)

    Reference: IEEE C57.91-2011, Clause 7
    """
    activation_energy_ev: Decimal  # Activation energy in eV
    reference_temp_k: Decimal      # Reference temperature in Kelvin
    reference_life_hours: Decimal  # Life at reference temperature
    description: str = ""
    source: str = ""


# Insulation and material aging parameters
ARRHENIUS_PARAMETERS: Dict[str, ArrheniusParameters] = {
    # Transformer Insulation Classes (IEEE C57.91)
    "insulation_class_a": ArrheniusParameters(
        activation_energy_ev=Decimal("0.87"),
        reference_temp_k=Decimal("378.15"),  # 105 C
        reference_life_hours=Decimal("175200"),  # 20 years
        description="Class A insulation (105 C rating)",
        source="IEEE C57.91-2011"
    ),
    "insulation_class_b": ArrheniusParameters(
        activation_energy_ev=Decimal("0.90"),
        reference_temp_k=Decimal("403.15"),  # 130 C
        reference_life_hours=Decimal("175200"),
        description="Class B insulation (130 C rating)",
        source="IEEE C57.91-2011"
    ),
    "insulation_class_f": ArrheniusParameters(
        activation_energy_ev=Decimal("0.93"),
        reference_temp_k=Decimal("428.15"),  # 155 C
        reference_life_hours=Decimal("175200"),
        description="Class F insulation (155 C rating)",
        source="IEEE C57.91-2011"
    ),
    "insulation_class_h": ArrheniusParameters(
        activation_energy_ev=Decimal("0.95"),
        reference_temp_k=Decimal("453.15"),  # 180 C
        reference_life_hours=Decimal("175200"),
        description="Class H insulation (180 C rating)",
        source="IEEE C57.91-2011"
    ),

    # Oil Degradation
    "transformer_oil_mineral": ArrheniusParameters(
        activation_energy_ev=Decimal("1.10"),
        reference_temp_k=Decimal("373.15"),  # 100 C
        reference_life_hours=Decimal("262800"),  # 30 years
        description="Mineral transformer oil",
        source="IEEE C57.106"
    ),

    # Polymer Degradation
    "xlpe_cable_insulation": ArrheniusParameters(
        activation_energy_ev=Decimal("1.05"),
        reference_temp_k=Decimal("363.15"),  # 90 C
        reference_life_hours=Decimal("350400"),  # 40 years
        description="XLPE cable insulation",
        source="IEEE 400.2"
    ),
    "epdm_rubber": ArrheniusParameters(
        activation_energy_ev=Decimal("0.85"),
        reference_temp_k=Decimal("373.15"),
        reference_life_hours=Decimal("87600"),  # 10 years
        description="EPDM rubber seals",
        source="SAE J2578"
    ),

    # Lubricants
    "lubricant_mineral_oil": ArrheniusParameters(
        activation_energy_ev=Decimal("0.60"),
        reference_temp_k=Decimal("353.15"),  # 80 C
        reference_life_hours=Decimal("8760"),  # 1 year
        description="Mineral lubricating oil",
        source="ASTM D2270"
    ),
    "lubricant_synthetic": ArrheniusParameters(
        activation_energy_ev=Decimal("0.70"),
        reference_temp_k=Decimal("373.15"),  # 100 C
        reference_life_hours=Decimal("17520"),  # 2 years
        description="Synthetic lubricant",
        source="ASTM D2270"
    ),
    "grease_lithium": ArrheniusParameters(
        activation_energy_ev=Decimal("0.55"),
        reference_temp_k=Decimal("343.15"),  # 70 C
        reference_life_hours=Decimal("8760"),  # 1 year
        description="Lithium-based grease",
        source="NLGI"
    ),
}


# =============================================================================
# ISO 10816 VIBRATION SEVERITY STANDARDS
# Source: ISO 10816-1:1995, ISO 10816-3:2009
# =============================================================================

class MachineClass(Enum):
    """
    ISO 10816-3 Machine Classification.

    Class I: Small machines (< 15 kW)
    Class II: Medium machines (15-75 kW) or large machines on flexible foundations
    Class III: Large machines (> 75 kW) on rigid foundations
    Class IV: Large machines on flexible foundations (turbo-machinery)
    """
    CLASS_I = auto()
    CLASS_II = auto()
    CLASS_III = auto()
    CLASS_IV = auto()


class VibrationZone(Enum):
    """
    ISO 10816 Vibration Severity Zones.

    Zone A: Good - Newly commissioned machines
    Zone B: Acceptable - Unrestricted long-term operation
    Zone C: Alert - Short-term operation only
    Zone D: Danger - May cause damage, immediate action required
    """
    ZONE_A = auto()
    ZONE_B = auto()
    ZONE_C = auto()
    ZONE_D = auto()


@dataclass(frozen=True)
class VibrationLimits:
    """
    Vibration velocity limits (mm/s RMS) per ISO 10816.

    Zones are defined by upper boundary values:
    - Below zone_a_upper: Zone A (Good)
    - zone_a_upper to zone_b_upper: Zone B (Acceptable)
    - zone_b_upper to zone_c_upper: Zone C (Alert)
    - Above zone_c_upper: Zone D (Danger)
    """
    zone_a_upper: Decimal  # mm/s RMS
    zone_b_upper: Decimal  # mm/s RMS
    zone_c_upper: Decimal  # mm/s RMS
    description: str = ""


# ISO 10816-3:2009 Table 1 - Evaluation zone boundary values
# Vibration velocity (mm/s RMS) in frequency range 10-1000 Hz
ISO_10816_VIBRATION_LIMITS: Dict[MachineClass, VibrationLimits] = {
    MachineClass.CLASS_I: VibrationLimits(
        zone_a_upper=Decimal("0.71"),
        zone_b_upper=Decimal("1.8"),
        zone_c_upper=Decimal("4.5"),
        description="Small machines < 15 kW"
    ),
    MachineClass.CLASS_II: VibrationLimits(
        zone_a_upper=Decimal("1.12"),
        zone_b_upper=Decimal("2.8"),
        zone_c_upper=Decimal("7.1"),
        description="Medium machines 15-75 kW or large on flexible mounts"
    ),
    MachineClass.CLASS_III: VibrationLimits(
        zone_a_upper=Decimal("1.8"),
        zone_b_upper=Decimal("4.5"),
        zone_c_upper=Decimal("11.2"),
        description="Large machines > 75 kW on rigid foundations"
    ),
    MachineClass.CLASS_IV: VibrationLimits(
        zone_a_upper=Decimal("2.8"),
        zone_b_upper=Decimal("7.1"),
        zone_c_upper=Decimal("18.0"),
        description="Large machines on flexible foundations (turbo)"
    ),
}


# =============================================================================
# BEARING FAULT FREQUENCIES
# Source: ISO 15243:2017, SKF Bearing Calculation
# =============================================================================

@dataclass(frozen=True)
class BearingGeometry:
    """
    Bearing geometry parameters for fault frequency calculation.

    n = Number of rolling elements
    Bd = Ball/roller diameter
    Pd = Pitch diameter
    phi = Contact angle (degrees)

    Fault Frequencies (multiples of shaft speed):
    - BPFO: Ball Pass Frequency Outer = (n/2) * (1 - Bd/Pd * cos(phi))
    - BPFI: Ball Pass Frequency Inner = (n/2) * (1 + Bd/Pd * cos(phi))
    - BSF: Ball Spin Frequency = (Pd/Bd) * (1 - (Bd/Pd * cos(phi))^2) / 2
    - FTF: Fundamental Train Frequency = (1/2) * (1 - Bd/Pd * cos(phi))

    Reference: Harris, T.A., "Rolling Bearing Analysis", 5th Ed.
    """
    num_rolling_elements: int
    ball_diameter_mm: Decimal
    pitch_diameter_mm: Decimal
    contact_angle_deg: Decimal
    bearing_type: str = ""


# Common bearing geometries
# Source: SKF Bearing Catalog
BEARING_GEOMETRIES: Dict[str, BearingGeometry] = {
    "6205": BearingGeometry(
        num_rolling_elements=9,
        ball_diameter_mm=Decimal("7.938"),
        pitch_diameter_mm=Decimal("38.5"),
        contact_angle_deg=Decimal("0"),
        bearing_type="Deep groove ball bearing"
    ),
    "6206": BearingGeometry(
        num_rolling_elements=9,
        ball_diameter_mm=Decimal("9.525"),
        pitch_diameter_mm=Decimal("46.0"),
        contact_angle_deg=Decimal("0"),
        bearing_type="Deep groove ball bearing"
    ),
    "6207": BearingGeometry(
        num_rolling_elements=9,
        ball_diameter_mm=Decimal("11.112"),
        pitch_diameter_mm=Decimal("53.5"),
        contact_angle_deg=Decimal("0"),
        bearing_type="Deep groove ball bearing"
    ),
    "6208": BearingGeometry(
        num_rolling_elements=9,
        ball_diameter_mm=Decimal("12.7"),
        pitch_diameter_mm=Decimal("60.0"),
        contact_angle_deg=Decimal("0"),
        bearing_type="Deep groove ball bearing"
    ),
    "6209": BearingGeometry(
        num_rolling_elements=10,
        ball_diameter_mm=Decimal("12.7"),
        pitch_diameter_mm=Decimal("65.0"),
        contact_angle_deg=Decimal("0"),
        bearing_type="Deep groove ball bearing"
    ),
    "6210": BearingGeometry(
        num_rolling_elements=10,
        ball_diameter_mm=Decimal("14.288"),
        pitch_diameter_mm=Decimal("72.5"),
        contact_angle_deg=Decimal("0"),
        bearing_type="Deep groove ball bearing"
    ),
    "7205_ac": BearingGeometry(
        num_rolling_elements=12,
        ball_diameter_mm=Decimal("6.35"),
        pitch_diameter_mm=Decimal("38.5"),
        contact_angle_deg=Decimal("25"),
        bearing_type="Angular contact ball bearing"
    ),
    "7206_ac": BearingGeometry(
        num_rolling_elements=12,
        ball_diameter_mm=Decimal("7.938"),
        pitch_diameter_mm=Decimal("46.0"),
        contact_angle_deg=Decimal("25"),
        bearing_type="Angular contact ball bearing"
    ),
    "22210_e": BearingGeometry(
        num_rolling_elements=17,
        ball_diameter_mm=Decimal("12.0"),
        pitch_diameter_mm=Decimal("70.0"),
        contact_angle_deg=Decimal("10"),
        bearing_type="Spherical roller bearing"
    ),
    "nu210": BearingGeometry(
        num_rolling_elements=14,
        ball_diameter_mm=Decimal("10.0"),
        pitch_diameter_mm=Decimal("72.5"),
        contact_angle_deg=Decimal("0"),
        bearing_type="Cylindrical roller bearing"
    ),
}


# =============================================================================
# MAINTENANCE COST PARAMETERS
# Source: Industry benchmarks, SMRP Best Practices
# =============================================================================

@dataclass(frozen=True)
class MaintenanceCostParameters:
    """
    Maintenance cost parameters for optimization calculations.

    Cp: Preventive maintenance cost
    Cf: Corrective (failure) maintenance cost
    Ci: Inspection cost
    Cd: Downtime cost per hour

    Reference: SMRP Best Practice 5.1
    """
    preventive_cost: Decimal
    corrective_cost: Decimal
    inspection_cost: Decimal
    downtime_cost_per_hour: Decimal
    currency: str = "USD"


# Typical cost ratios by equipment type
# Source: Plant Engineering surveys, industry benchmarks
MAINTENANCE_COST_RATIOS: Dict[str, MaintenanceCostParameters] = {
    "pump_centrifugal": MaintenanceCostParameters(
        preventive_cost=Decimal("2500"),
        corrective_cost=Decimal("15000"),
        inspection_cost=Decimal("500"),
        downtime_cost_per_hour=Decimal("5000"),
    ),
    "motor_large": MaintenanceCostParameters(
        preventive_cost=Decimal("5000"),
        corrective_cost=Decimal("50000"),
        inspection_cost=Decimal("1000"),
        downtime_cost_per_hour=Decimal("10000"),
    ),
    "compressor": MaintenanceCostParameters(
        preventive_cost=Decimal("10000"),
        corrective_cost=Decimal("100000"),
        inspection_cost=Decimal("2000"),
        downtime_cost_per_hour=Decimal("25000"),
    ),
    "transformer": MaintenanceCostParameters(
        preventive_cost=Decimal("15000"),
        corrective_cost=Decimal("500000"),
        inspection_cost=Decimal("5000"),
        downtime_cost_per_hour=Decimal("50000"),
    ),
    "heat_exchanger": MaintenanceCostParameters(
        preventive_cost=Decimal("8000"),
        corrective_cost=Decimal("40000"),
        inspection_cost=Decimal("2000"),
        downtime_cost_per_hour=Decimal("15000"),
    ),
}


# =============================================================================
# STATISTICAL CONSTANTS
# =============================================================================

# Z-scores for confidence intervals (two-tailed)
Z_SCORES: Dict[str, Decimal] = {
    "80%": Decimal("1.282"),
    "85%": Decimal("1.440"),
    "90%": Decimal("1.645"),
    "95%": Decimal("1.960"),
    "99%": Decimal("2.576"),
    "99.9%": Decimal("3.291"),
}

# Chi-square critical values (commonly used degrees of freedom)
# Format: {df: {confidence: value}}
CHI_SQUARE_CRITICAL: Dict[int, Dict[str, Decimal]] = {
    1: {"90%": Decimal("2.706"), "95%": Decimal("3.841"), "99%": Decimal("6.635")},
    2: {"90%": Decimal("4.605"), "95%": Decimal("5.991"), "99%": Decimal("9.210")},
    5: {"90%": Decimal("9.236"), "95%": Decimal("11.070"), "99%": Decimal("15.086")},
    10: {"90%": Decimal("15.987"), "95%": Decimal("18.307"), "99%": Decimal("23.209")},
    20: {"90%": Decimal("28.412"), "95%": Decimal("31.410"), "99%": Decimal("37.566")},
}


# =============================================================================
# UNIT CONVERSION FACTORS
# =============================================================================

# Time conversions to hours
TIME_TO_HOURS: Dict[str, Decimal] = {
    "seconds": Decimal("1") / Decimal("3600"),
    "minutes": Decimal("1") / Decimal("60"),
    "hours": Decimal("1"),
    "days": Decimal("24"),
    "weeks": Decimal("168"),
    "months": Decimal("730"),  # Average month
    "years": Decimal("8760"),  # 365 days
}

# Temperature conversions (to Kelvin)
def celsius_to_kelvin(celsius: Decimal) -> Decimal:
    """Convert Celsius to Kelvin."""
    return celsius + KELVIN_OFFSET


def fahrenheit_to_kelvin(fahrenheit: Decimal) -> Decimal:
    """Convert Fahrenheit to Kelvin."""
    return (fahrenheit - Decimal("32")) * Decimal("5") / Decimal("9") + KELVIN_OFFSET


# Vibration unit conversions
VIBRATION_CONVERSIONS: Dict[str, Decimal] = {
    "mm_s_to_in_s": Decimal("0.03937"),
    "in_s_to_mm_s": Decimal("25.4"),
    "mm_s_to_ips": Decimal("0.03937"),
    "ips_to_mm_s": Decimal("25.4"),
    "g_to_m_s2": Decimal("9.80665"),
    "m_s2_to_g": Decimal("0.10197"),
    "mils_to_mm": Decimal("0.0254"),
    "mm_to_mils": Decimal("39.37"),
}


# =============================================================================
# DEFAULT CALCULATION PARAMETERS
# =============================================================================

# Default confidence level for uncertainty calculations
DEFAULT_CONFIDENCE_LEVEL: Final[str] = "95%"

# Default precision for Decimal calculations
DEFAULT_DECIMAL_PRECISION: Final[int] = 10

# Default number of Monte Carlo simulations
DEFAULT_MONTE_CARLO_ITERATIONS: Final[int] = 10000

# Default time horizon for predictions (hours)
DEFAULT_PREDICTION_HORIZON_HOURS: Final[Decimal] = Decimal("8760")  # 1 year

# Minimum probability threshold (avoid division by zero)
MIN_PROBABILITY_THRESHOLD: Final[Decimal] = Decimal("1e-10")

# Maximum probability value
MAX_PROBABILITY: Final[Decimal] = Decimal("0.999999999")


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Physical constants
    "BOLTZMANN_CONSTANT",
    "BOLTZMANN_CONSTANT_EV",
    "KELVIN_OFFSET",
    "STANDARD_TEMP_K",
    "STANDARD_GRAVITY",
    "PI",
    "E",

    # Weibull parameters
    "WeibullParameters",
    "WEIBULL_PARAMETERS",

    # Failure rates
    "FAILURE_RATES_FPMH",

    # Arrhenius parameters
    "ArrheniusParameters",
    "ARRHENIUS_PARAMETERS",

    # Vibration standards
    "MachineClass",
    "VibrationZone",
    "VibrationLimits",
    "ISO_10816_VIBRATION_LIMITS",

    # Bearing data
    "BearingGeometry",
    "BEARING_GEOMETRIES",

    # Maintenance costs
    "MaintenanceCostParameters",
    "MAINTENANCE_COST_RATIOS",

    # Statistical constants
    "Z_SCORES",
    "CHI_SQUARE_CRITICAL",

    # Conversions
    "TIME_TO_HOURS",
    "VIBRATION_CONVERSIONS",
    "celsius_to_kelvin",
    "fahrenheit_to_kelvin",

    # Defaults
    "DEFAULT_CONFIDENCE_LEVEL",
    "DEFAULT_DECIMAL_PRECISION",
    "DEFAULT_MONTE_CARLO_ITERATIONS",
    "DEFAULT_PREDICTION_HORIZON_HOURS",
    "MIN_PROBABILITY_THRESHOLD",
    "MAX_PROBABILITY",
]
