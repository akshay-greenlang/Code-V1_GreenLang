"""
Dimensional Constants for GL-FOUND-X-003 Unit & Reference Normalizer.

This module defines the canonical dimensional constants used throughout the
GreenLang Normalizer for dimensional analysis. It provides:

- BASE_DIMENSION_SYMBOLS: The seven SI base dimensions plus GreenLang extensions
- GL_CANONICAL_DIMENSIONS: Mapping of canonical units to their dimensions
- DERIVED_DIMENSION_DEFINITIONS: Definitions of derived dimensions in terms of base dimensions
- DIMENSION_SYMBOLS: Human-readable symbols for dimension display

All mappings are immutable (frozensets/MappingProxyTypes) to ensure deterministic
behavior and prevent accidental modification.

Example:
    >>> from gl_normalizer_core.dimensions.constants import (
    ...     BASE_DIMENSION_SYMBOLS,
    ...     GL_CANONICAL_DIMENSIONS,
    ... )
    >>> assert "length" in BASE_DIMENSION_SYMBOLS
    >>> assert GL_CANONICAL_DIMENSIONS["kg"] == {"mass": 1}
"""

from types import MappingProxyType
from typing import Dict, FrozenSet


# =============================================================================
# Base Dimension Symbols
# =============================================================================
# The seven SI base dimensions plus GreenLang-specific extensions for
# sustainability reporting. These form the fundamental building blocks
# for all dimensional analysis.

BASE_DIMENSION_SYMBOLS: FrozenSet[str] = frozenset({
    # SI Base Dimensions
    "length",           # L - meter (m)
    "mass",             # M - kilogram (kg)
    "time",             # T - second (s)
    "temperature",      # Theta - kelvin (K)
    "amount",           # N - mole (mol)
    "current",          # I - ampere (A)
    "luminosity",       # J - candela (cd)
})

# Human-readable symbols for each base dimension
DIMENSION_SYMBOLS: Dict[str, str] = MappingProxyType({
    "length": "L",
    "mass": "M",
    "time": "T",
    "temperature": "Theta",
    "amount": "N",
    "current": "I",
    "luminosity": "J",
})


# =============================================================================
# Derived Dimension Definitions
# =============================================================================
# Definitions of derived dimensions in terms of base dimensions.
# Each derived dimension is represented as a dictionary mapping
# base dimension names to their exponents.

DERIVED_DIMENSION_DEFINITIONS: Dict[str, Dict[str, int]] = MappingProxyType({
    # Mechanical derived dimensions
    "area": {"length": 2},
    "volume": {"length": 3},
    "velocity": {"length": 1, "time": -1},
    "acceleration": {"length": 1, "time": -2},
    "force": {"mass": 1, "length": 1, "time": -2},
    "pressure": {"mass": 1, "length": -1, "time": -2},
    "density": {"mass": 1, "length": -3},

    # Energy and power dimensions
    "energy": {"mass": 1, "length": 2, "time": -2},
    "power": {"mass": 1, "length": 2, "time": -3},

    # Electromagnetic dimensions
    "charge": {"current": 1, "time": 1},
    "voltage": {"mass": 1, "length": 2, "time": -3, "current": -1},
    "resistance": {"mass": 1, "length": 2, "time": -3, "current": -2},
    "capacitance": {"mass": -1, "length": -2, "time": 4, "current": 2},

    # Flow dimensions
    "volume_flow": {"length": 3, "time": -1},
    "mass_flow": {"mass": 1, "time": -1},

    # GreenLang-specific derived dimensions for sustainability
    "emissions": {"mass": 1},  # Emissions are mass-based (kgCO2e)
    "emissions_intensity": {"time": -2},  # kgCO2e/kWh -> mass / energy
    "energy_intensity": {"length": 2, "time": -2},  # Energy per unit (MJ/kg)
    "carbon_intensity": {"time": 2, "length": -2},  # kgCO2e/MJ

    # Specific heat and thermal dimensions
    "specific_heat": {"length": 2, "time": -2, "temperature": -1},
    "thermal_conductivity": {"mass": 1, "length": 1, "time": -3, "temperature": -1},

    # Concentration and molar dimensions
    "molarity": {"amount": 1, "length": -3},
    "molar_mass": {"mass": 1, "amount": -1},

    # Dimensionless
    "dimensionless": {},
})


# =============================================================================
# GreenLang Canonical Unit to Dimension Mapping
# =============================================================================
# Maps canonical GreenLang units to their dimension signatures.
# This is the authoritative source for unit-to-dimension resolution.

GL_CANONICAL_DIMENSIONS: Dict[str, Dict[str, int]] = MappingProxyType({
    # ==========================================================================
    # Mass Units
    # ==========================================================================
    "kg": {"mass": 1},
    "g": {"mass": 1},
    "mg": {"mass": 1},
    "t": {"mass": 1},  # metric ton
    "tonne": {"mass": 1},
    "lb": {"mass": 1},
    "oz": {"mass": 1},

    # ==========================================================================
    # Length Units
    # ==========================================================================
    "m": {"length": 1},
    "km": {"length": 1},
    "cm": {"length": 1},
    "mm": {"length": 1},
    "mi": {"length": 1},
    "ft": {"length": 1},
    "in": {"length": 1},
    "yd": {"length": 1},
    "nmi": {"length": 1},  # nautical mile

    # ==========================================================================
    # Area Units
    # ==========================================================================
    "m2": {"length": 2},
    "m^2": {"length": 2},
    "km2": {"length": 2},
    "km^2": {"length": 2},
    "ha": {"length": 2},  # hectare
    "acre": {"length": 2},
    "ft2": {"length": 2},
    "ft^2": {"length": 2},

    # ==========================================================================
    # Volume Units
    # ==========================================================================
    "m3": {"length": 3},
    "m^3": {"length": 3},
    "L": {"length": 3},
    "l": {"length": 3},
    "mL": {"length": 3},
    "ml": {"length": 3},
    "gal": {"length": 3},
    "bbl": {"length": 3},  # barrel
    "ft3": {"length": 3},
    "ft^3": {"length": 3},
    "Nm3": {"length": 3},  # normal cubic meter
    "Nm^3": {"length": 3},
    "scf": {"length": 3},  # standard cubic feet

    # ==========================================================================
    # Time Units
    # ==========================================================================
    "s": {"time": 1},
    "min": {"time": 1},
    "h": {"time": 1},
    "hr": {"time": 1},
    "d": {"time": 1},
    "day": {"time": 1},
    "week": {"time": 1},
    "month": {"time": 1},
    "year": {"time": 1},
    "yr": {"time": 1},
    "a": {"time": 1},  # annum

    # ==========================================================================
    # Temperature Units
    # ==========================================================================
    "K": {"temperature": 1},
    "degC": {"temperature": 1},
    "degF": {"temperature": 1},
    "celsius": {"temperature": 1},
    "fahrenheit": {"temperature": 1},

    # ==========================================================================
    # Energy Units
    # ==========================================================================
    "J": {"mass": 1, "length": 2, "time": -2},
    "kJ": {"mass": 1, "length": 2, "time": -2},
    "MJ": {"mass": 1, "length": 2, "time": -2},
    "GJ": {"mass": 1, "length": 2, "time": -2},
    "TJ": {"mass": 1, "length": 2, "time": -2},
    "kWh": {"mass": 1, "length": 2, "time": -2},
    "MWh": {"mass": 1, "length": 2, "time": -2},
    "GWh": {"mass": 1, "length": 2, "time": -2},
    "Wh": {"mass": 1, "length": 2, "time": -2},
    "BTU": {"mass": 1, "length": 2, "time": -2},
    "btu": {"mass": 1, "length": 2, "time": -2},
    "therm": {"mass": 1, "length": 2, "time": -2},
    "MMBtu": {"mass": 1, "length": 2, "time": -2},
    "cal": {"mass": 1, "length": 2, "time": -2},
    "kcal": {"mass": 1, "length": 2, "time": -2},

    # ==========================================================================
    # Power Units
    # ==========================================================================
    "W": {"mass": 1, "length": 2, "time": -3},
    "kW": {"mass": 1, "length": 2, "time": -3},
    "MW": {"mass": 1, "length": 2, "time": -3},
    "GW": {"mass": 1, "length": 2, "time": -3},
    "hp": {"mass": 1, "length": 2, "time": -3},  # horsepower

    # ==========================================================================
    # Pressure Units
    # ==========================================================================
    "Pa": {"mass": 1, "length": -1, "time": -2},
    "kPa": {"mass": 1, "length": -1, "time": -2},
    "MPa": {"mass": 1, "length": -1, "time": -2},
    "bar": {"mass": 1, "length": -1, "time": -2},
    "mbar": {"mass": 1, "length": -1, "time": -2},
    "atm": {"mass": 1, "length": -1, "time": -2},
    "psi": {"mass": 1, "length": -1, "time": -2},

    # ==========================================================================
    # GreenLang Emissions Units
    # ==========================================================================
    # Emissions are fundamentally mass-based
    "kgCO2e": {"mass": 1},
    "kgCO2eq": {"mass": 1},
    "tCO2e": {"mass": 1},
    "tCO2eq": {"mass": 1},
    "gCO2e": {"mass": 1},
    "lbCO2e": {"mass": 1},
    "kgCO2": {"mass": 1},
    "tCO2": {"mass": 1},
    "kgCH4": {"mass": 1},
    "kgN2O": {"mass": 1},

    # ==========================================================================
    # Flow Rate Units
    # ==========================================================================
    "m3/s": {"length": 3, "time": -1},
    "m3/h": {"length": 3, "time": -1},
    "L/s": {"length": 3, "time": -1},
    "L/min": {"length": 3, "time": -1},
    "gal/min": {"length": 3, "time": -1},
    "gpm": {"length": 3, "time": -1},
    "kg/s": {"mass": 1, "time": -1},
    "kg/h": {"mass": 1, "time": -1},
    "t/h": {"mass": 1, "time": -1},

    # ==========================================================================
    # Density Units
    # ==========================================================================
    "kg/m3": {"mass": 1, "length": -3},
    "kg/m^3": {"mass": 1, "length": -3},
    "g/cm3": {"mass": 1, "length": -3},
    "g/cm^3": {"mass": 1, "length": -3},
    "lb/ft3": {"mass": 1, "length": -3},
    "lb/ft^3": {"mass": 1, "length": -3},

    # ==========================================================================
    # Intensity and Ratio Units
    # ==========================================================================
    "kgCO2e/kWh": {"time": 2, "length": -2},  # Emissions intensity
    "kgCO2e/MJ": {"time": 2, "length": -2},
    "kgCO2e/MWh": {"time": 2, "length": -2},
    "tCO2e/MWh": {"time": 2, "length": -2},
    "gCO2e/kWh": {"time": 2, "length": -2},
    "MJ/kg": {"length": 2, "time": -2},  # Specific energy / heating value
    "kJ/kg": {"length": 2, "time": -2},
    "BTU/lb": {"length": 2, "time": -2},
    "kWh/kg": {"length": 2, "time": -2},

    # ==========================================================================
    # Amount of Substance Units
    # ==========================================================================
    "mol": {"amount": 1},
    "kmol": {"amount": 1},
    "mmol": {"amount": 1},

    # ==========================================================================
    # Electrical Units
    # ==========================================================================
    "A": {"current": 1},
    "mA": {"current": 1},
    "V": {"mass": 1, "length": 2, "time": -3, "current": -1},
    "kV": {"mass": 1, "length": 2, "time": -3, "current": -1},
    "ohm": {"mass": 1, "length": 2, "time": -3, "current": -2},
    "F": {"mass": -1, "length": -2, "time": 4, "current": 2},

    # ==========================================================================
    # Dimensionless Units
    # ==========================================================================
    "%": {},
    "ppm": {},
    "ppb": {},
    "ratio": {},
    "fraction": {},
})


# =============================================================================
# Dimension Name Aliases
# =============================================================================
# Maps alternative dimension names to their canonical form

DIMENSION_ALIASES: Dict[str, str] = MappingProxyType({
    # Energy aliases
    "work": "energy",
    "heat": "energy",

    # Mass aliases
    "weight": "mass",
    "emissions_mass": "emissions",

    # Volume aliases
    "capacity": "volume",

    # Flow aliases
    "volumetric_flow": "volume_flow",
    "volumetric_flow_rate": "volume_flow",
    "mass_flow_rate": "mass_flow",

    # Pressure aliases
    "stress": "pressure",

    # Temperature aliases
    "thermodynamic_temperature": "temperature",
})


# =============================================================================
# GreenLang Canonical Unit per Dimension
# =============================================================================
# The default canonical unit for each dimension in GreenLang.
# These are used when converting to canonical form.

CANONICAL_UNIT_PER_DIMENSION: Dict[str, str] = MappingProxyType({
    "length": "m",
    "mass": "kg",
    "time": "s",
    "temperature": "K",
    "amount": "mol",
    "current": "A",
    "luminosity": "cd",
    "area": "m2",
    "volume": "m3",
    "energy": "MJ",
    "power": "kW",
    "pressure": "Pa",
    "density": "kg/m3",
    "volume_flow": "m3/s",
    "mass_flow": "kg/s",
    "emissions": "kgCO2e",
    "emissions_intensity": "kgCO2e/kWh",
    "energy_intensity": "MJ/kg",
    "dimensionless": "",
})


# =============================================================================
# Unit Categories for GreenLang
# =============================================================================
# Categorization of units for reporting and UI purposes

UNIT_CATEGORIES: Dict[str, FrozenSet[str]] = MappingProxyType({
    "base_si": frozenset({"m", "kg", "s", "K", "mol", "A", "cd"}),
    "energy": frozenset({
        "J", "kJ", "MJ", "GJ", "TJ",
        "Wh", "kWh", "MWh", "GWh",
        "BTU", "btu", "MMBtu", "therm",
        "cal", "kcal",
    }),
    "emissions": frozenset({
        "kgCO2e", "kgCO2eq", "tCO2e", "tCO2eq", "gCO2e", "lbCO2e",
        "kgCO2", "tCO2", "kgCH4", "kgN2O",
    }),
    "power": frozenset({"W", "kW", "MW", "GW", "hp"}),
    "volume": frozenset({
        "m3", "m^3", "L", "l", "mL", "ml",
        "gal", "bbl", "ft3", "ft^3",
        "Nm3", "Nm^3", "scf",
    }),
    "mass": frozenset({"kg", "g", "mg", "t", "tonne", "lb", "oz"}),
    "intensity": frozenset({
        "kgCO2e/kWh", "kgCO2e/MJ", "kgCO2e/MWh", "tCO2e/MWh", "gCO2e/kWh",
        "MJ/kg", "kJ/kg", "BTU/lb", "kWh/kg",
    }),
})


# =============================================================================
# Context-Dependent Conversion Units
# =============================================================================
# Units that require additional context (reference conditions, GWP version, etc.)

CONTEXT_DEPENDENT_UNITS: Dict[str, str] = MappingProxyType({
    # Reference condition dependent (temperature/pressure)
    "Nm3": "reference_conditions",
    "Nm^3": "reference_conditions",
    "scf": "reference_conditions",

    # GWP version dependent
    "kgCO2e": "gwp_version",
    "kgCO2eq": "gwp_version",
    "tCO2e": "gwp_version",
    "tCO2eq": "gwp_version",
    "gCO2e": "gwp_version",
    "lbCO2e": "gwp_version",

    # Energy basis dependent (HHV/LHV)
    "MJ/kg": "energy_basis",
    "kJ/kg": "energy_basis",
    "BTU/lb": "energy_basis",
})
