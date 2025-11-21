# -*- coding: utf-8 -*-
"""
greenlang/utils/unit_conversion.py

Comprehensive Unit Conversion System for FuelAgentAI v2

Supports:
- Volume conversions (gallons, liters, cubic meters)
- Energy conversions (therms, kWh, MJ, GJ, BTU)
- Mass conversions (tons, tonnes, kg, lbs)
- Temperature conversions (F, C, K)
- Pressure conversions (psi, bar, kPa)

Sources:
- NIST SP 811 (US standard conversions)
- ISO 80000-1 (International standard)
- IEC 80000-13 (Energy conversions)

Author: GreenLang Framework Team
Date: October 2025
"""

from typing import Dict, Tuple, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class UnitType(str, Enum):
    """Unit type categories."""
    VOLUME = "volume"
    ENERGY = "energy"
    MASS = "mass"
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"


class VolumeUnit(str, Enum):
    """Volume units."""
    GALLONS_US = "gallons"
    GALLONS_UK = "gallons_uk"
    LITERS = "liters"
    CUBIC_METERS = "m3"
    CUBIC_FEET = "ft3"
    BARRELS = "barrels"


class EnergyUnit(str, Enum):
    """Energy units."""
    THERMS = "therms"
    KWH = "kWh"
    MWH = "MWh"
    MJ = "MJ"
    GJ = "GJ"
    BTU = "BTU"
    MMBTU = "MMBTU"


class MassUnit(str, Enum):
    """Mass units."""
    TONS_US = "tons"
    TONNES = "tonnes"
    KG = "kg"
    LBS = "lbs"


class TemperatureUnit(str, Enum):
    """Temperature units."""
    FAHRENHEIT = "F"
    CELSIUS = "C"
    KELVIN = "K"


class PressureUnit(str, Enum):
    """Pressure units."""
    PSI = "psi"
    BAR = "bar"
    KPA = "kPa"
    ATM = "atm"


# ==================== CONVERSION FACTORS ====================

# Volume conversions (all to liters)
VOLUME_TO_LITERS = {
    VolumeUnit.GALLONS_US: 3.78541,
    VolumeUnit.GALLONS_UK: 4.54609,
    VolumeUnit.LITERS: 1.0,
    VolumeUnit.CUBIC_METERS: 1000.0,
    VolumeUnit.CUBIC_FEET: 28.3168,
    VolumeUnit.BARRELS: 158.987,
}

# Energy conversions (all to kWh)
ENERGY_TO_KWH = {
    EnergyUnit.THERMS: 29.3001,
    EnergyUnit.KWH: 1.0,
    EnergyUnit.MWH: 1000.0,
    EnergyUnit.MJ: 0.277778,
    EnergyUnit.GJ: 277.778,
    EnergyUnit.BTU: 0.000293071,
    EnergyUnit.MMBTU: 293.071,
}

# Mass conversions (all to kg)
MASS_TO_KG = {
    MassUnit.TONS_US: 907.185,
    MassUnit.TONNES: 1000.0,
    MassUnit.KG: 1.0,
    MassUnit.LBS: 0.453592,
}

# Pressure conversions (all to kPa)
PRESSURE_TO_KPA = {
    PressureUnit.PSI: 6.89476,
    PressureUnit.BAR: 100.0,
    PressureUnit.KPA: 1.0,
    PressureUnit.ATM: 101.325,
}


class UnitConverter:
    """
    Comprehensive unit conversion system.

    Supports conversions between imperial and metric units
    for volume, energy, mass, temperature, and pressure.
    """

    @staticmethod
    def detect_unit_type(unit: str) -> Optional[UnitType]:
        """
        Detect the type of unit.

        Args:
            unit: Unit string (e.g., "gallons", "kWh", "tonnes")

        Returns:
            UnitType enum or None if not recognized
        """
        unit_lower = unit.lower()

        # Volume
        if unit_lower in [u.value for u in VolumeUnit]:
            return UnitType.VOLUME

        # Energy
        if unit_lower in [u.value.lower() for u in EnergyUnit]:
            return UnitType.ENERGY

        # Mass
        if unit_lower in [u.value for u in MassUnit]:
            return UnitType.MASS

        # Temperature
        if unit_lower in [u.value.lower() for u in TemperatureUnit]:
            return UnitType.TEMPERATURE

        # Pressure
        if unit_lower in [u.value.lower() for u in PressureUnit]:
            return UnitType.PRESSURE

        return None

    @staticmethod
    def convert_volume(
        value: float,
        from_unit: str,
        to_unit: str
    ) -> float:
        """
        Convert between volume units.

        Args:
            value: Value to convert
            from_unit: Source unit (gallons, liters, m3, etc.)
            to_unit: Target unit

        Returns:
            Converted value

        Raises:
            ValueError: If units not recognized
        """
        from_unit_enum = VolumeUnit(from_unit.lower())
        to_unit_enum = VolumeUnit(to_unit.lower())

        # Convert to liters (base unit)
        value_liters = value * VOLUME_TO_LITERS[from_unit_enum]

        # Convert from liters to target unit
        result = value_liters / VOLUME_TO_LITERS[to_unit_enum]

        return result

    @staticmethod
    def convert_energy(
        value: float,
        from_unit: str,
        to_unit: str
    ) -> float:
        """
        Convert between energy units.

        Args:
            value: Value to convert
            from_unit: Source unit (therms, kWh, MJ, etc.)
            to_unit: Target unit

        Returns:
            Converted value

        Raises:
            ValueError: If units not recognized
        """
        from_unit_enum = EnergyUnit(from_unit)
        to_unit_enum = EnergyUnit(to_unit)

        # Convert to kWh (base unit)
        value_kwh = value * ENERGY_TO_KWH[from_unit_enum]

        # Convert from kWh to target unit
        result = value_kwh / ENERGY_TO_KWH[to_unit_enum]

        return result

    @staticmethod
    def convert_mass(
        value: float,
        from_unit: str,
        to_unit: str
    ) -> float:
        """
        Convert between mass units.

        Args:
            value: Value to convert
            from_unit: Source unit (tons, tonnes, kg, lbs)
            to_unit: Target unit

        Returns:
            Converted value

        Raises:
            ValueError: If units not recognized
        """
        from_unit_enum = MassUnit(from_unit.lower())
        to_unit_enum = MassUnit(to_unit.lower())

        # Convert to kg (base unit)
        value_kg = value * MASS_TO_KG[from_unit_enum]

        # Convert from kg to target unit
        result = value_kg / MASS_TO_KG[to_unit_enum]

        return result

    @staticmethod
    def convert_temperature(
        value: float,
        from_unit: str,
        to_unit: str
    ) -> float:
        """
        Convert between temperature units.

        Args:
            value: Value to convert
            from_unit: Source unit (F, C, K)
            to_unit: Target unit

        Returns:
            Converted value

        Raises:
            ValueError: If units not recognized
        """
        from_unit_enum = TemperatureUnit(from_unit.upper())
        to_unit_enum = TemperatureUnit(to_unit.upper())

        # Convert to Celsius (base unit)
        if from_unit_enum == TemperatureUnit.CELSIUS:
            value_c = value
        elif from_unit_enum == TemperatureUnit.FAHRENHEIT:
            value_c = (value - 32) * 5 / 9
        else:  # Kelvin
            value_c = value - 273.15

        # Convert from Celsius to target unit
        if to_unit_enum == TemperatureUnit.CELSIUS:
            result = value_c
        elif to_unit_enum == TemperatureUnit.FAHRENHEIT:
            result = value_c * 9 / 5 + 32
        else:  # Kelvin
            result = value_c + 273.15

        return result

    @staticmethod
    def convert_pressure(
        value: float,
        from_unit: str,
        to_unit: str
    ) -> float:
        """
        Convert between pressure units.

        Args:
            value: Value to convert
            from_unit: Source unit (psi, bar, kPa, atm)
            to_unit: Target unit

        Returns:
            Converted value

        Raises:
            ValueError: If units not recognized
        """
        from_unit_enum = PressureUnit(from_unit.lower())
        to_unit_enum = PressureUnit(to_unit.lower())

        # Convert to kPa (base unit)
        value_kpa = value * PRESSURE_TO_KPA[from_unit_enum]

        # Convert from kPa to target unit
        result = value_kpa / PRESSURE_TO_KPA[to_unit_enum]

        return result

    @staticmethod
    def convert(
        value: float,
        from_unit: str,
        to_unit: str
    ) -> float:
        """
        Auto-detect unit type and convert.

        Args:
            value: Value to convert
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted value

        Raises:
            ValueError: If units not recognized or incompatible
        """
        converter = UnitConverter()

        from_type = converter.detect_unit_type(from_unit)
        to_type = converter.detect_unit_type(to_unit)

        if from_type != to_type:
            raise ValueError(
                f"Incompatible unit types: {from_unit} ({from_type}) "
                f"and {to_unit} ({to_type})"
            )

        if from_type == UnitType.VOLUME:
            return converter.convert_volume(value, from_unit, to_unit)
        elif from_type == UnitType.ENERGY:
            return converter.convert_energy(value, from_unit, to_unit)
        elif from_type == UnitType.MASS:
            return converter.convert_mass(value, from_unit, to_unit)
        elif from_type == UnitType.TEMPERATURE:
            return converter.convert_temperature(value, from_unit, to_unit)
        elif from_type == UnitType.PRESSURE:
            return converter.convert_pressure(value, from_unit, to_unit)
        else:
            raise ValueError(f"Unknown unit type: {from_unit}")


# ==================== REGIONAL DEFAULTS ====================

REGIONAL_DEFAULTS = {
    "US": {
        "volume_unit": VolumeUnit.GALLONS_US,
        "energy_unit": EnergyUnit.THERMS,
        "mass_unit": MassUnit.TONS_US,
        "temperature_unit": TemperatureUnit.FAHRENHEIT,
        "pressure_unit": PressureUnit.PSI,
        "currency": "USD",
        "date_format": "MM/DD/YYYY",
        "decimal_separator": ".",
        "thousands_separator": ",",
    },
    "UK": {
        "volume_unit": VolumeUnit.LITERS,
        "energy_unit": EnergyUnit.KWH,
        "mass_unit": MassUnit.TONNES,
        "temperature_unit": TemperatureUnit.CELSIUS,
        "pressure_unit": PressureUnit.BAR,
        "currency": "GBP",
        "date_format": "DD/MM/YYYY",
        "decimal_separator": ".",
        "thousands_separator": ",",
    },
    "EU": {
        "volume_unit": VolumeUnit.LITERS,
        "energy_unit": EnergyUnit.KWH,
        "mass_unit": MassUnit.TONNES,
        "temperature_unit": TemperatureUnit.CELSIUS,
        "pressure_unit": PressureUnit.BAR,
        "currency": "EUR",
        "date_format": "DD/MM/YYYY",
        "decimal_separator": ",",
        "thousands_separator": ".",
    },
    "CA": {  # Canada
        "volume_unit": VolumeUnit.LITERS,
        "energy_unit": EnergyUnit.KWH,
        "mass_unit": MassUnit.TONNES,
        "temperature_unit": TemperatureUnit.CELSIUS,
        "pressure_unit": PressureUnit.KPA,
        "currency": "CAD",
        "date_format": "YYYY-MM-DD",
        "decimal_separator": ".",
        "thousands_separator": ",",
    },
    "AU": {  # Australia
        "volume_unit": VolumeUnit.LITERS,
        "energy_unit": EnergyUnit.KWH,
        "mass_unit": MassUnit.TONNES,
        "temperature_unit": TemperatureUnit.CELSIUS,
        "pressure_unit": PressureUnit.KPA,
        "currency": "AUD",
        "date_format": "DD/MM/YYYY",
        "decimal_separator": ".",
        "thousands_separator": ",",
    },
    "IN": {  # India
        "volume_unit": VolumeUnit.LITERS,
        "energy_unit": EnergyUnit.KWH,
        "mass_unit": MassUnit.TONNES,
        "temperature_unit": TemperatureUnit.CELSIUS,
        "pressure_unit": PressureUnit.KPA,
        "currency": "INR",
        "date_format": "DD/MM/YYYY",
        "decimal_separator": ".",
        "thousands_separator": ",",
    },
    "CN": {  # China
        "volume_unit": VolumeUnit.LITERS,
        "energy_unit": EnergyUnit.KWH,
        "mass_unit": MassUnit.TONNES,
        "temperature_unit": TemperatureUnit.CELSIUS,
        "pressure_unit": PressureUnit.KPA,
        "currency": "CNY",
        "date_format": "YYYY-MM-DD",
        "decimal_separator": ".",
        "thousands_separator": ",",
    },
    "JP": {  # Japan
        "volume_unit": VolumeUnit.LITERS,
        "energy_unit": EnergyUnit.KWH,
        "mass_unit": MassUnit.TONNES,
        "temperature_unit": TemperatureUnit.CELSIUS,
        "pressure_unit": PressureUnit.KPA,
        "currency": "JPY",
        "date_format": "YYYY/MM/DD",
        "decimal_separator": ".",
        "thousands_separator": ",",
    },
    "BR": {  # Brazil
        "volume_unit": VolumeUnit.LITERS,
        "energy_unit": EnergyUnit.KWH,
        "mass_unit": MassUnit.TONNES,
        "temperature_unit": TemperatureUnit.CELSIUS,
        "pressure_unit": PressureUnit.KPA,
        "currency": "BRL",
        "date_format": "DD/MM/YYYY",
        "decimal_separator": ",",
        "thousands_separator": ".",
    },
    "MX": {  # Mexico
        "volume_unit": VolumeUnit.LITERS,
        "energy_unit": EnergyUnit.KWH,
        "mass_unit": MassUnit.TONNES,
        "temperature_unit": TemperatureUnit.CELSIUS,
        "pressure_unit": PressureUnit.KPA,
        "currency": "MXN",
        "date_format": "DD/MM/YYYY",
        "decimal_separator": ".",
        "thousands_separator": ",",
    },
}


def get_regional_defaults(region: str) -> Dict:
    """
    Get regional defaults for units, formatting, etc.

    Args:
        region: ISO country code (US, UK, EU, CA, etc.)

    Returns:
        Dict with regional defaults

    Raises:
        ValueError: If region not recognized
    """
    if region.upper() not in REGIONAL_DEFAULTS:
        logger.warning(f"Unknown region {region}, using US defaults")
        return REGIONAL_DEFAULTS["US"]

    return REGIONAL_DEFAULTS[region.upper()]


def format_number(
    value: float,
    region: str = "US",
    decimals: int = 2
) -> str:
    """
    Format number according to regional conventions.

    Args:
        value: Number to format
        region: ISO country code
        decimals: Number of decimal places

    Returns:
        Formatted number string
    """
    defaults = get_regional_defaults(region)

    # Round to specified decimals
    rounded = round(value, decimals)

    # Format with decimals
    formatted = f"{rounded:.{decimals}f}"

    # Apply regional separators
    decimal_sep = defaults["decimal_separator"]
    thousands_sep = defaults["thousands_separator"]

    # Replace decimal separator
    parts = formatted.split(".")
    if len(parts) == 2:
        integer_part, decimal_part = parts
    else:
        integer_part = parts[0]
        decimal_part = ""

    # Add thousands separator
    integer_with_sep = ""
    for i, digit in enumerate(reversed(integer_part)):
        if i > 0 and i % 3 == 0:
            integer_with_sep = thousands_sep + integer_with_sep
        integer_with_sep = digit + integer_with_sep

    # Combine
    if decimal_part:
        result = integer_with_sep + decimal_sep + decimal_part
    else:
        result = integer_with_sep

    return result
