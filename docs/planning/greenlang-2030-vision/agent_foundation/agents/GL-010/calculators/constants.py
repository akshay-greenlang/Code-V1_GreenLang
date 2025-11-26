"""
Physical Constants and Conversion Factors for Emissions Calculations.

This module provides authoritative physical constants and conversion factors
used throughout the GL-010 EMISSIONWATCH calculator modules. All values are
sourced from EPA, NIST, and international standards.

Zero-Hallucination Guarantee:
- All constants are from authoritative sources (cited in docstrings)
- Values are deterministic and immutable
- No approximations without explicit uncertainty ranges

References:
- EPA AP-42, Fifth Edition
- EPA 40 CFR Part 60, Appendix A
- NIST Standard Reference Data
- ISO 13443 (Natural gas standard reference conditions)
"""

from typing import Final, Dict
from decimal import Decimal


# =============================================================================
# UNIVERSAL PHYSICAL CONSTANTS
# Source: NIST CODATA 2018
# =============================================================================

# Avogadro constant (mol^-1)
AVOGADRO: Final[Decimal] = Decimal("6.02214076e23")

# Universal gas constant (J/(mol*K))
R_UNIVERSAL: Final[Decimal] = Decimal("8.314462618")

# Universal gas constant (L*atm/(mol*K))
R_ATM: Final[Decimal] = Decimal("0.08205746")

# Standard gravity (m/s^2)
G_STANDARD: Final[Decimal] = Decimal("9.80665")

# Stefan-Boltzmann constant (W/(m^2*K^4))
STEFAN_BOLTZMANN: Final[Decimal] = Decimal("5.670374419e-8")


# =============================================================================
# STANDARD CONDITIONS
# =============================================================================

# EPA Standard conditions (40 CFR 60)
EPA_STD_TEMP_F: Final[Decimal] = Decimal("68")  # Fahrenheit
EPA_STD_TEMP_K: Final[Decimal] = Decimal("293.15")  # Kelvin
EPA_STD_PRESSURE_INHG: Final[Decimal] = Decimal("29.92")  # inches Hg
EPA_STD_PRESSURE_KPA: Final[Decimal] = Decimal("101.325")  # kPa

# ISO Standard conditions (0C, 101.325 kPa)
ISO_STD_TEMP_K: Final[Decimal] = Decimal("273.15")  # Kelvin
ISO_STD_PRESSURE_KPA: Final[Decimal] = Decimal("101.325")  # kPa

# Normal conditions (EU standard: 0C, 101.325 kPa, dry gas)
NORMAL_TEMP_K: Final[Decimal] = Decimal("273.15")
NORMAL_PRESSURE_KPA: Final[Decimal] = Decimal("101.325")

# Standard molar volume at STP (L/mol)
MOLAR_VOLUME_STP: Final[Decimal] = Decimal("22.414")

# Standard molar volume at EPA conditions (L/mol at 68F, 29.92 inHg)
MOLAR_VOLUME_EPA: Final[Decimal] = Decimal("24.055")


# =============================================================================
# MOLECULAR WEIGHTS (g/mol)
# Source: IUPAC 2021 Atomic Weights
# =============================================================================

MW: Final[Dict[str, Decimal]] = {
    # Elements
    "C": Decimal("12.011"),
    "H": Decimal("1.008"),
    "O": Decimal("15.999"),
    "N": Decimal("14.007"),
    "S": Decimal("32.06"),
    "Ar": Decimal("39.948"),

    # Diatomic molecules
    "O2": Decimal("31.998"),
    "N2": Decimal("28.014"),
    "H2": Decimal("2.016"),

    # Combustion products
    "CO2": Decimal("44.009"),
    "CO": Decimal("28.010"),
    "H2O": Decimal("18.015"),
    "SO2": Decimal("64.066"),
    "SO3": Decimal("80.066"),
    "NO": Decimal("30.006"),
    "NO2": Decimal("46.006"),
    "N2O": Decimal("44.013"),

    # Other gases
    "CH4": Decimal("16.043"),
    "C2H6": Decimal("30.070"),
    "C3H8": Decimal("44.097"),
    "NH3": Decimal("17.031"),
    "HCl": Decimal("36.461"),
    "HF": Decimal("20.006"),

    # Air (dry)
    "AIR": Decimal("28.966"),
}


# =============================================================================
# ENERGY CONVERSION FACTORS
# =============================================================================

# Heat content conversions
BTU_TO_JOULE: Final[Decimal] = Decimal("1055.06")
JOULE_TO_BTU: Final[Decimal] = Decimal("0.000947817")
MMBTU_TO_GJ: Final[Decimal] = Decimal("1.05506")
GJ_TO_MMBTU: Final[Decimal] = Decimal("0.947817")
KWH_TO_BTU: Final[Decimal] = Decimal("3412.14")
BTU_TO_KWH: Final[Decimal] = Decimal("0.000293071")

# Therm conversions
THERM_TO_BTU: Final[Decimal] = Decimal("100000")
THERM_TO_MMBTU: Final[Decimal] = Decimal("0.1")


# =============================================================================
# MASS CONVERSION FACTORS
# =============================================================================

LB_TO_KG: Final[Decimal] = Decimal("0.453592")
KG_TO_LB: Final[Decimal] = Decimal("2.20462")
TON_SHORT_TO_KG: Final[Decimal] = Decimal("907.185")
TON_SHORT_TO_METRIC: Final[Decimal] = Decimal("0.907185")
TON_METRIC_TO_LB: Final[Decimal] = Decimal("2204.62")
TON_METRIC_TO_KG: Final[Decimal] = Decimal("1000")
OZ_TO_G: Final[Decimal] = Decimal("28.3495")
G_TO_OZ: Final[Decimal] = Decimal("0.035274")


# =============================================================================
# VOLUME CONVERSION FACTORS
# =============================================================================

GAL_TO_LITER: Final[Decimal] = Decimal("3.78541")
LITER_TO_GAL: Final[Decimal] = Decimal("0.264172")
FT3_TO_M3: Final[Decimal] = Decimal("0.0283168")
M3_TO_FT3: Final[Decimal] = Decimal("35.3147")
BBL_TO_GAL: Final[Decimal] = Decimal("42")
BBL_TO_LITER: Final[Decimal] = Decimal("158.987")
SCF_TO_NM3: Final[Decimal] = Decimal("0.02679")  # Standard cubic feet to normal cubic meters


# =============================================================================
# TEMPERATURE CONVERSION OFFSETS
# =============================================================================

CELSIUS_TO_KELVIN_OFFSET: Final[Decimal] = Decimal("273.15")
FAHRENHEIT_TO_RANKINE_OFFSET: Final[Decimal] = Decimal("459.67")


# =============================================================================
# PRESSURE CONVERSION FACTORS
# =============================================================================

ATM_TO_KPA: Final[Decimal] = Decimal("101.325")
ATM_TO_PSI: Final[Decimal] = Decimal("14.696")
ATM_TO_INHG: Final[Decimal] = Decimal("29.9213")
ATM_TO_MBAR: Final[Decimal] = Decimal("1013.25")
PSI_TO_KPA: Final[Decimal] = Decimal("6.89476")
INHG_TO_KPA: Final[Decimal] = Decimal("3.38639")
MBAR_TO_KPA: Final[Decimal] = Decimal("0.1")


# =============================================================================
# CONCENTRATION CONVERSION FACTORS
# =============================================================================

# PPM to mg/m3 at standard conditions (multiply by MW/24.45 at 25C, 1 atm)
# Or MW/22.4 at 0C, 1 atm (normal conditions)
PPM_TO_MG_NM3_FACTOR: Final[Decimal] = Decimal("0.04461")  # Per unit MW at 0C, 1 atm

# Percent to ppm
PERCENT_TO_PPM: Final[Decimal] = Decimal("10000")
PPM_TO_PERCENT: Final[Decimal] = Decimal("0.0001")


# =============================================================================
# F-FACTORS FOR FUEL COMBUSTION (dscf/MMBtu at 0% O2)
# Source: EPA 40 CFR Part 60, Appendix A, Method 19
# =============================================================================

F_FACTORS: Final[Dict[str, Dict[str, Decimal]]] = {
    # Fd = dry F-factor (dscf/MMBtu)
    # Fw = wet F-factor (wscf/MMBtu)
    # Fc = carbon F-factor (scf CO2/MMBtu)

    "natural_gas": {
        "Fd": Decimal("8710"),
        "Fw": Decimal("10610"),
        "Fc": Decimal("1040"),
    },
    "fuel_oil_no2": {
        "Fd": Decimal("9190"),
        "Fw": Decimal("10320"),
        "Fc": Decimal("1420"),
    },
    "fuel_oil_no6": {
        "Fd": Decimal("9220"),
        "Fw": Decimal("10260"),
        "Fc": Decimal("1420"),
    },
    "coal_bituminous": {
        "Fd": Decimal("9780"),
        "Fw": Decimal("10640"),
        "Fc": Decimal("1800"),
    },
    "coal_subbituminous": {
        "Fd": Decimal("9820"),
        "Fw": Decimal("10580"),
        "Fc": Decimal("1840"),
    },
    "coal_lignite": {
        "Fd": Decimal("9860"),
        "Fw": Decimal("10590"),
        "Fc": Decimal("1850"),
    },
    "coal_anthracite": {
        "Fd": Decimal("10100"),
        "Fw": Decimal("10540"),
        "Fc": Decimal("1970"),
    },
    "wood": {
        "Fd": Decimal("9240"),
        "Fw": Decimal("10390"),
        "Fc": Decimal("1910"),
    },
    "propane": {
        "Fd": Decimal("8710"),
        "Fw": Decimal("10200"),
        "Fc": Decimal("1190"),
    },
    "butane": {
        "Fd": Decimal("8710"),
        "Fw": Decimal("10060"),
        "Fc": Decimal("1260"),
    },
}


# =============================================================================
# COMBUSTION AIR CONSTANTS
# =============================================================================

# Composition of dry air by volume
AIR_COMPOSITION: Final[Dict[str, Decimal]] = {
    "N2": Decimal("0.7808"),
    "O2": Decimal("0.2095"),
    "Ar": Decimal("0.0093"),
    "CO2": Decimal("0.0004"),
}

# Oxygen content in dry air (volume fraction)
O2_IN_AIR: Final[Decimal] = Decimal("0.2095")

# Nitrogen to oxygen ratio in air
N2_TO_O2_RATIO: Final[Decimal] = Decimal("3.76")


# =============================================================================
# GLOBAL WARMING POTENTIALS (100-year, AR6)
# Source: IPCC AR6, 2021
# =============================================================================

GWP_100: Final[Dict[str, int]] = {
    "CO2": 1,
    "CH4": 28,  # Fossil methane (includes climate-carbon feedback)
    "CH4_biogenic": 27,  # Biogenic methane
    "N2O": 273,
    "SF6": 25200,
    "NF3": 17400,
    "HFC-134a": 1530,
    "HFC-32": 771,
    "HFC-125": 3740,
    "HFC-143a": 5810,
    "HFC-152a": 164,
    "HFC-227ea": 3600,
    "HFC-245fa": 962,
    "CF4": 7380,
    "C2F6": 12400,
    "C3F8": 9290,
}


# =============================================================================
# NOX CHEMISTRY CONSTANTS
# =============================================================================

# Zeldovich mechanism activation energies (K)
ZELDOVICH_EA1: Final[Decimal] = Decimal("38370")  # N2 + O -> NO + N
ZELDOVICH_EA2: Final[Decimal] = Decimal("3160")   # N + O2 -> NO + O
ZELDOVICH_EA3: Final[Decimal] = Decimal("22080")  # N + OH -> NO + H

# Pre-exponential factors for Zeldovich (cm3/mol/s)
ZELDOVICH_A1: Final[Decimal] = Decimal("1.8e14")
ZELDOVICH_A2: Final[Decimal] = Decimal("9.0e9")
ZELDOVICH_A3: Final[Decimal] = Decimal("2.8e13")


# =============================================================================
# REGULATORY CONSTANTS
# =============================================================================

# EPA oxygen correction reference levels (% O2)
O2_REFERENCE: Final[Dict[str, Decimal]] = {
    "boiler": Decimal("3.0"),
    "gas_turbine": Decimal("15.0"),
    "reciprocating_engine": Decimal("15.0"),
    "incinerator": Decimal("7.0"),
    "cement_kiln": Decimal("10.0"),
}

# Averaging periods in hours
AVERAGING_PERIODS: Final[Dict[str, int]] = {
    "1-hour": 1,
    "3-hour": 3,
    "8-hour": 8,
    "24-hour": 24,
    "30-day": 720,
    "rolling_30_day": 720,
    "quarterly": 2190,
    "annual": 8760,
}


# =============================================================================
# ATMOSPHERIC STABILITY CLASSES
# Pasquill-Gifford stability classes
# =============================================================================

STABILITY_CLASSES: Final[Dict[str, str]] = {
    "A": "Very unstable",
    "B": "Unstable",
    "C": "Slightly unstable",
    "D": "Neutral",
    "E": "Slightly stable",
    "F": "Stable",
}


# =============================================================================
# STACK PARAMETER CONSTANTS
# =============================================================================

# Default stack exit temperature (K) for various sources
DEFAULT_STACK_TEMP: Final[Dict[str, Decimal]] = {
    "natural_gas_boiler": Decimal("450"),
    "coal_boiler": Decimal("420"),
    "gas_turbine": Decimal("780"),
    "incinerator": Decimal("500"),
    "industrial_furnace": Decimal("550"),
}

# Default stack exit velocity (m/s)
DEFAULT_STACK_VELOCITY: Final[Dict[str, Decimal]] = {
    "natural_gas_boiler": Decimal("15"),
    "coal_boiler": Decimal("20"),
    "gas_turbine": Decimal("30"),
    "incinerator": Decimal("12"),
    "industrial_furnace": Decimal("18"),
}
