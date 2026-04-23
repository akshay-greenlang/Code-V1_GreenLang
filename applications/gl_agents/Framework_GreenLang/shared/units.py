"""
GreenLang Framework - Unit Conversion

Standardized unit conversion for industrial calculations.
Ensures consistency across all GreenLang agents.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple, Union
import math


class UnitSystem(Enum):
    """Supported unit systems."""
    SI = "SI"          # International System (metric)
    IMPERIAL = "Imperial"  # US Customary
    CGS = "CGS"        # Centimeter-gram-second


class TemperatureUnit(Enum):
    """Temperature units."""
    CELSIUS = "C"
    KELVIN = "K"
    FAHRENHEIT = "F"
    RANKINE = "R"


class EnergyUnit(Enum):
    """Energy units."""
    JOULE = "J"
    KILOJOULE = "kJ"
    MEGAJOULE = "MJ"
    GIGAJOULE = "GJ"
    KILOWATT_HOUR = "kWh"
    MEGAWATT_HOUR = "MWh"
    BTU = "BTU"
    THERM = "therm"
    CALORIE = "cal"
    KILOCALORIE = "kcal"


class PowerUnit(Enum):
    """Power units."""
    WATT = "W"
    KILOWATT = "kW"
    MEGAWATT = "MW"
    GIGAWATT = "GW"
    BTU_PER_HOUR = "BTU/h"
    HORSEPOWER = "hp"
    TON_REFRIGERATION = "TR"


class MassUnit(Enum):
    """Mass units."""
    KILOGRAM = "kg"
    GRAM = "g"
    TONNE = "t"
    POUND = "lb"
    SHORT_TON = "short_ton"
    LONG_TON = "long_ton"


class FlowRateUnit(Enum):
    """Mass/volume flow rate units."""
    KG_PER_SECOND = "kg/s"
    KG_PER_HOUR = "kg/h"
    TONNE_PER_HOUR = "t/h"
    LB_PER_HOUR = "lb/h"
    M3_PER_HOUR = "m3/h"
    GAL_PER_MIN = "gpm"


class PressureUnit(Enum):
    """Pressure units."""
    PASCAL = "Pa"
    KILOPASCAL = "kPa"
    MEGAPASCAL = "MPa"
    BAR = "bar"
    ATMOSPHERE = "atm"
    PSI = "psi"
    MM_HG = "mmHg"
    IN_HG = "inHg"


class AreaUnit(Enum):
    """Area units."""
    SQUARE_METER = "m2"
    SQUARE_CENTIMETER = "cm2"
    SQUARE_FOOT = "ft2"
    SQUARE_INCH = "in2"


@dataclass
class ConversionFactor:
    """Conversion factor with metadata."""
    factor: float
    offset: float = 0.0
    source: str = "Standard"


class UnitConverter:
    """
    Centralized unit conversion utility.

    Provides conversions between common industrial units
    with full traceability and validation.

    Usage:
        >>> conv = UnitConverter()
        >>> result = conv.convert_temperature(100, "C", "F")
        >>> print(result)  # 212.0
    """

    # Temperature conversions (to Kelvin as base)
    TEMP_TO_KELVIN: Dict[str, Tuple[float, float]] = {
        "C": (1.0, 273.15),      # K = C + 273.15
        "K": (1.0, 0.0),         # K = K
        "F": (5/9, 255.372),     # K = (F + 459.67) * 5/9
        "R": (5/9, 0.0),         # K = R * 5/9
    }

    # Energy conversions (to Joules as base)
    ENERGY_TO_JOULES: Dict[str, float] = {
        "J": 1.0,
        "kJ": 1000.0,
        "MJ": 1e6,
        "GJ": 1e9,
        "kWh": 3.6e6,
        "MWh": 3.6e9,
        "BTU": 1055.06,
        "therm": 1.055e8,
        "cal": 4.184,
        "kcal": 4184.0,
    }

    # Power conversions (to Watts as base)
    POWER_TO_WATTS: Dict[str, float] = {
        "W": 1.0,
        "kW": 1000.0,
        "MW": 1e6,
        "GW": 1e9,
        "BTU/h": 0.293071,
        "hp": 745.7,
        "TR": 3516.85,
    }

    # Mass conversions (to kg as base)
    MASS_TO_KG: Dict[str, float] = {
        "kg": 1.0,
        "g": 0.001,
        "t": 1000.0,
        "lb": 0.453592,
        "short_ton": 907.185,
        "long_ton": 1016.05,
    }

    # Pressure conversions (to Pa as base)
    PRESSURE_TO_PA: Dict[str, float] = {
        "Pa": 1.0,
        "kPa": 1000.0,
        "MPa": 1e6,
        "bar": 100000.0,
        "atm": 101325.0,
        "psi": 6894.76,
        "mmHg": 133.322,
        "inHg": 3386.39,
    }

    # Area conversions (to m² as base)
    AREA_TO_M2: Dict[str, float] = {
        "m2": 1.0,
        "cm2": 0.0001,
        "ft2": 0.092903,
        "in2": 0.00064516,
    }

    # Flow rate conversions (to kg/s as base)
    FLOW_TO_KG_S: Dict[str, float] = {
        "kg/s": 1.0,
        "kg/h": 1/3600,
        "t/h": 1000/3600,
        "lb/h": 0.453592/3600,
    }

    def __init__(self, default_system: UnitSystem = UnitSystem.SI):
        """Initialize converter with default unit system."""
        self.default_system = default_system

    def convert_temperature(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
    ) -> float:
        """
        Convert temperature between units.

        Args:
            value: Temperature value
            from_unit: Source unit (C, K, F, R)
            to_unit: Target unit (C, K, F, R)

        Returns:
            Converted temperature
        """
        if from_unit == to_unit:
            return value

        # Convert to Kelvin
        mult, offset = self.TEMP_TO_KELVIN[from_unit]
        kelvin = value * mult + offset

        # Convert from Kelvin
        mult, offset = self.TEMP_TO_KELVIN[to_unit]
        return (kelvin - offset) / mult

    def convert_energy(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
    ) -> float:
        """Convert energy between units."""
        if from_unit == to_unit:
            return value

        # Convert to Joules then to target
        joules = value * self.ENERGY_TO_JOULES[from_unit]
        return joules / self.ENERGY_TO_JOULES[to_unit]

    def convert_power(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
    ) -> float:
        """Convert power between units."""
        if from_unit == to_unit:
            return value

        watts = value * self.POWER_TO_WATTS[from_unit]
        return watts / self.POWER_TO_WATTS[to_unit]

    def convert_mass(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
    ) -> float:
        """Convert mass between units."""
        if from_unit == to_unit:
            return value

        kg = value * self.MASS_TO_KG[from_unit]
        return kg / self.MASS_TO_KG[to_unit]

    def convert_pressure(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
    ) -> float:
        """Convert pressure between units."""
        if from_unit == to_unit:
            return value

        pa = value * self.PRESSURE_TO_PA[from_unit]
        return pa / self.PRESSURE_TO_PA[to_unit]

    def convert_area(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
    ) -> float:
        """Convert area between units."""
        if from_unit == to_unit:
            return value

        m2 = value * self.AREA_TO_M2[from_unit]
        return m2 / self.AREA_TO_M2[to_unit]

    def convert_flow_rate(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
    ) -> float:
        """Convert mass flow rate between units."""
        if from_unit == to_unit:
            return value

        kg_s = value * self.FLOW_TO_KG_S[from_unit]
        return kg_s / self.FLOW_TO_KG_S[to_unit]

    def convert(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
        quantity_type: str = "auto",
    ) -> float:
        """
        Generic conversion with auto-detection.

        Args:
            value: Value to convert
            from_unit: Source unit
            to_unit: Target unit
            quantity_type: Type of quantity (temperature, energy, etc.)

        Returns:
            Converted value
        """
        if quantity_type == "auto":
            quantity_type = self._detect_quantity_type(from_unit)

        converters = {
            "temperature": self.convert_temperature,
            "energy": self.convert_energy,
            "power": self.convert_power,
            "mass": self.convert_mass,
            "pressure": self.convert_pressure,
            "area": self.convert_area,
            "flow_rate": self.convert_flow_rate,
        }

        if quantity_type in converters:
            return converters[quantity_type](value, from_unit, to_unit)
        else:
            raise ValueError(f"Unknown quantity type: {quantity_type}")

    def _detect_quantity_type(self, unit: str) -> str:
        """Detect quantity type from unit string."""
        if unit in self.TEMP_TO_KELVIN:
            return "temperature"
        elif unit in self.ENERGY_TO_JOULES:
            return "energy"
        elif unit in self.POWER_TO_WATTS:
            return "power"
        elif unit in self.MASS_TO_KG:
            return "mass"
        elif unit in self.PRESSURE_TO_PA:
            return "pressure"
        elif unit in self.AREA_TO_M2:
            return "area"
        elif unit in self.FLOW_TO_KG_S:
            return "flow_rate"
        else:
            raise ValueError(f"Unknown unit: {unit}")

    @staticmethod
    def normalize_to_si(
        value: float,
        unit: str,
    ) -> Tuple[float, str]:
        """
        Normalize value to SI base units.

        Returns:
            Tuple of (normalized value, SI unit)
        """
        conv = UnitConverter()

        si_units = {
            "temperature": ("K", "K"),
            "energy": ("J", "J"),
            "power": ("W", "W"),
            "mass": ("kg", "kg"),
            "pressure": ("Pa", "Pa"),
            "area": ("m2", "m2"),
            "flow_rate": ("kg/s", "kg/s"),
        }

        try:
            qty_type = conv._detect_quantity_type(unit)
            target_unit = si_units[qty_type][1]
            normalized = conv.convert(value, unit, target_unit, qty_type)
            return normalized, target_unit
        except ValueError:
            return value, unit


# Singleton instance
UNIT_CONVERTER = UnitConverter()
