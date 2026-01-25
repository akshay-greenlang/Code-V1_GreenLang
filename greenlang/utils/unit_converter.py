# -*- coding: utf-8 -*-
"""
Unit Converter Utility

Centralized unit conversion library for GreenLang agents.
Handles energy, fuel, area, and emission unit conversions.

ZERO-HALLUCINATION: All conversions are deterministic mathematical operations.
NO LLM involvement. Fail loudly on unknown units.
"""

from decimal import Decimal
from typing import Dict, Union
import logging

from greenlang.exceptions import ValidationError


class UnitConversionError(Exception):
    """Raised when unit conversion fails"""
    pass


class UnitConverter:
    """Centralized unit conversion utility for all GreenLang agents."""

    # Energy conversion factors to MMBtu
    ENERGY_TO_MMBTU = {
        "MMBtu": 1.0,
        "MBtu": 0.001,
        "Btu": 1e-6,
        "kWh": 0.003412,
        "MWh": 3.412,
        "GWh": 3412.0,
        "therms": 0.1,
        "MJ": 0.000948,
        "GJ": 0.948,
        "kcal": 3.968e-6,
        "Mcal": 0.003968,
        "Gcal": 3.968,
        "kBtu": 0.001,
    }

    # Area conversion factors to square feet
    AREA_TO_SQFT = {
        "sqft": 1.0,
        "sqm": 10.764,
        "m2": 10.764,
        "ft2": 1.0,
        "sqyd": 9.0,
        "acre": 43560.0,
        "hectare": 107639.0,
    }

    # Mass conversion factors to kg
    MASS_TO_KG = {
        "kg": 1.0,
        "g": 0.001,
        "mg": 1e-6,
        "ton": 1000.0,
        "metric_ton": 1000.0,
        "tonne": 1000.0,
        "lb": 0.453592,
        "lbs": 0.453592,
        "pound": 0.453592,
        "short_ton": 907.185,
        "long_ton": 1016.05,
        "oz": 0.0283495,
    }

    # Volume conversion factors to liters
    VOLUME_TO_LITERS = {
        "liter": 1.0,
        "L": 1.0,
        "ml": 0.001,
        "mL": 0.001,
        "gallon": 3.78541,
        "gallons": 3.78541,
        "gal": 3.78541,
        "quart": 0.946353,
        "pint": 0.473176,
        "cup": 0.236588,
        "fl_oz": 0.0295735,
        "m3": 1000.0,
        "cubic_meter": 1000.0,
        "ft3": 28.3168,
        "cubic_feet": 28.3168,
        "barrel": 158.987,
        "bbl": 158.987,
    }

    # Fuel-specific energy content (to MMBtu)
    FUEL_ENERGY_CONTENT = {
        "natural_gas": {
            "therms": 0.1,
            "ccf": 0.103,  # hundred cubic feet
            "mcf": 1.03,  # thousand cubic feet
            "m3": 0.0353,
            "MMBtu": 1.0,
            "GJ": 0.948,
        },
        "diesel": {
            "gallon": 0.138,
            "gallons": 0.138,
            "liter": 0.0365,
            "L": 0.0365,
            "barrel": 5.825,
            "bbl": 5.825,
        },
        "gasoline": {"gallon": 0.125, "gallons": 0.125, "liter": 0.033, "L": 0.033},
        "propane": {
            "gallon": 0.0915,
            "gallons": 0.0915,
            "liter": 0.0242,
            "L": 0.0242,
            "lb": 0.02165,
            "lbs": 0.02165,
            "kg": 0.04774,
        },
        "fuel_oil": {
            "gallon": 0.140,
            "gallons": 0.140,
            "liter": 0.037,
            "L": 0.037,
            "barrel": 5.88,
            "bbl": 5.88,
        },
        "coal": {
            "ton": 20.0,
            "short_ton": 20.0,
            "metric_ton": 22.0,
            "tonne": 22.0,
            "kg": 0.022,
            "lb": 0.01,
            "lbs": 0.01,
        },
        "biomass": {
            "ton": 15.0,  # Average for wood pellets
            "metric_ton": 16.5,
            "kg": 0.0165,
            "lb": 0.0075,
            "lbs": 0.0075,
        },
        "electricity": {"kWh": 0.003412, "MWh": 3.412, "GWh": 3412.0},
    }

    # Time conversions (to hours as base unit)
    TIME_TO_HOURS: Dict[str, Decimal] = {
        'hour': Decimal('1.0'),
        'hours': Decimal('1.0'),
        'hr': Decimal('1.0'),
        'h': Decimal('1.0'),
        'day': Decimal('24.0'),
        'days': Decimal('24.0'),
        'week': Decimal('168.0'),
        'weeks': Decimal('168.0'),
        'month': Decimal('730.0'),  # Average 30.42 days
        'months': Decimal('730.0'),
        'year': Decimal('8760.0'),
        'years': Decimal('8760.0'),
    }

    # Distance conversions (to km as base unit) - from calculation version
    DISTANCE_TO_KM: Dict[str, Decimal] = {
        'km': Decimal('1.0'),
        'kilometer': Decimal('1.0'),
        'kilometers': Decimal('1.0'),
        'mile': Decimal('1.60934'),
        'miles': Decimal('1.60934'),
        'mi': Decimal('1.60934'),
        'm': Decimal('0.001'),
        'meter': Decimal('0.001'),
        'meters': Decimal('0.001'),
        'ft': Decimal('0.0003048'),
        'feet': Decimal('0.0003048'),
        'foot': Decimal('0.0003048'),
    }

    def __init__(self) -> None:
        """Initialize the unit converter with logging."""
        self.logger = logging.getLogger(__name__)

        # Build comprehensive conversion tables for generic convert() method
        self.conversion_tables = {
            'energy': self._build_energy_to_kwh(),
            'volume': self._build_volume_to_liters(),
            'mass': self._build_mass_to_kg(),
            'distance': self.DISTANCE_TO_KM,
            'area': self._build_area_to_m2(),
            'time': self.TIME_TO_HOURS,
        }

    def _build_energy_to_kwh(self) -> Dict[str, Decimal]:
        """Build energy conversion table to kWh as base unit."""
        return {
            'kwh': Decimal('1.0'),
            'mwh': Decimal('1000.0'),
            'gwh': Decimal('1000000.0'),
            'mmbtu': Decimal('293.071'),
            'therm': Decimal('29.3071'),
            'therms': Decimal('29.3071'),
            'gj': Decimal('277.778'),
            'mj': Decimal('0.277778'),
            'btu': Decimal('0.000293071'),
            'mbtu': Decimal('0.293071'),
            'kbtu': Decimal('0.293071'),
        }

    def _build_volume_to_liters(self) -> Dict[str, Decimal]:
        """Build volume conversion table to liters as base unit."""
        return {
            'liter': Decimal('1.0'),
            'liters': Decimal('1.0'),
            'l': Decimal('1.0'),
            'gallon': Decimal('3.78541'),
            'gallons': Decimal('3.78541'),
            'gal': Decimal('3.78541'),
            'm3': Decimal('1000.0'),
            'cubic_meter': Decimal('1000.0'),
            'ccf': Decimal('2831.68'),
            'mcf': Decimal('28316.8'),
            'scf': Decimal('28.3168'),
            'ft3': Decimal('28.3168'),
            'cubic_feet': Decimal('28.3168'),
            'barrel': Decimal('158.987'),
            'bbl': Decimal('158.987'),
            'ml': Decimal('0.001'),
            'quart': Decimal('0.946353'),
            'pint': Decimal('0.473176'),
            'cup': Decimal('0.236588'),
            'fl_oz': Decimal('0.0295735'),
        }

    def _build_mass_to_kg(self) -> Dict[str, Decimal]:
        """Build mass conversion table to kg as base unit."""
        return {
            'kg': Decimal('1.0'),
            'kilogram': Decimal('1.0'),
            'kilograms': Decimal('1.0'),
            'tonne': Decimal('1000.0'),
            'tonnes': Decimal('1000.0'),
            'metric_ton': Decimal('1000.0'),
            'ton': Decimal('907.185'),
            'tons': Decimal('907.185'),
            'short_ton': Decimal('907.185'),
            'long_ton': Decimal('1016.05'),
            'lb': Decimal('0.453592'),
            'lbs': Decimal('0.453592'),
            'pound': Decimal('0.453592'),
            'pounds': Decimal('0.453592'),
            'g': Decimal('0.001'),
            'gram': Decimal('0.001'),
            'grams': Decimal('0.001'),
            'mg': Decimal('0.000001'),
            'oz': Decimal('0.0283495'),
        }

    def _build_area_to_m2(self) -> Dict[str, Decimal]:
        """Build area conversion table to m2 as base unit."""
        return {
            'm2': Decimal('1.0'),
            'square_meter': Decimal('1.0'),
            'sqm': Decimal('1.0'),
            'ft2': Decimal('0.092903'),
            'square_foot': Decimal('0.092903'),
            'sqft': Decimal('0.092903'),
            'sqyd': Decimal('0.836127'),
            'acre': Decimal('4046.86'),
            'acres': Decimal('4046.86'),
            'hectare': Decimal('10000.0'),
            'hectares': Decimal('10000.0'),
            'ha': Decimal('10000.0'),
        }

    def convert_energy(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert energy between different units.

        Args:
            value: The value to convert
            from_unit: The unit to convert from
            to_unit: The unit to convert to

        Returns:
            float: The converted value

        Raises:
            ValueError: If unit is not recognized
        """
        if from_unit == to_unit:
            return value

        # Convert to MMBtu first
        if from_unit not in self.ENERGY_TO_MMBTU:
            raise ValidationError(
                message=f"Unknown energy unit: {from_unit}",
                context={
                    "from_unit": from_unit,
                    "to_unit": to_unit,
                    "supported_units": list(self.ENERGY_TO_MMBTU.keys())
                },
                invalid_fields={"from_unit": f"Unit '{from_unit}' not recognized"}
            )

        mmbtu_value = value * self.ENERGY_TO_MMBTU[from_unit]

        # Convert from MMBtu to target unit
        if to_unit not in self.ENERGY_TO_MMBTU:
            raise ValidationError(
                message=f"Unknown energy unit: {to_unit}",
                context={
                    "from_unit": from_unit,
                    "to_unit": to_unit,
                    "supported_units": list(self.ENERGY_TO_MMBTU.keys())
                },
                invalid_fields={"to_unit": f"Unit '{to_unit}' not recognized"}
            )

        return mmbtu_value / self.ENERGY_TO_MMBTU[to_unit]

    def convert_area(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert area between different units.

        Args:
            value: The value to convert
            from_unit: The unit to convert from
            to_unit: The unit to convert to

        Returns:
            float: The converted value

        Raises:
            ValueError: If unit is not recognized
        """
        if from_unit == to_unit:
            return value

        # Convert to sqft first
        if from_unit not in self.AREA_TO_SQFT:
            raise ValidationError(
                message=f"Unknown area unit: {from_unit}",
                context={
                    "from_unit": from_unit,
                    "to_unit": to_unit,
                    "supported_units": list(self.AREA_TO_SQFT.keys())
                },
                invalid_fields={"from_unit": f"Unit '{from_unit}' not recognized"}
            )

        sqft_value = value * self.AREA_TO_SQFT[from_unit]

        # Convert from sqft to target unit
        if to_unit not in self.AREA_TO_SQFT:
            raise ValidationError(
                message=f"Unknown area unit: {to_unit}",
                context={
                    "from_unit": from_unit,
                    "to_unit": to_unit,
                    "supported_units": list(self.AREA_TO_SQFT.keys())
                },
                invalid_fields={"to_unit": f"Unit '{to_unit}' not recognized"}
            )

        return sqft_value / self.AREA_TO_SQFT[to_unit]

    def convert_mass(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert mass between different units.

        Args:
            value: The value to convert
            from_unit: The unit to convert from
            to_unit: The unit to convert to

        Returns:
            float: The converted value

        Raises:
            ValueError: If unit is not recognized
        """
        if from_unit == to_unit:
            return value

        # Convert to kg first
        if from_unit not in self.MASS_TO_KG:
            raise ValidationError(
                message=f"Unknown mass unit: {from_unit}",
                context={
                    "from_unit": from_unit,
                    "to_unit": to_unit,
                    "supported_units": list(self.MASS_TO_KG.keys())
                },
                invalid_fields={"from_unit": f"Unit '{from_unit}' not recognized"}
            )

        kg_value = value * self.MASS_TO_KG[from_unit]

        # Convert from kg to target unit
        if to_unit not in self.MASS_TO_KG:
            raise ValidationError(
                message=f"Unknown mass unit: {to_unit}",
                context={
                    "from_unit": from_unit,
                    "to_unit": to_unit,
                    "supported_units": list(self.MASS_TO_KG.keys())
                },
                invalid_fields={"to_unit": f"Unit '{to_unit}' not recognized"}
            )

        return kg_value / self.MASS_TO_KG[to_unit]

    def convert_volume(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert volume between different units.

        Args:
            value: The value to convert
            from_unit: The unit to convert from
            to_unit: The unit to convert to

        Returns:
            float: The converted value

        Raises:
            ValueError: If unit is not recognized
        """
        if from_unit == to_unit:
            return value

        # Convert to liters first
        if from_unit not in self.VOLUME_TO_LITERS:
            raise ValidationError(
                message=f"Unknown volume unit: {from_unit}",
                context={
                    "from_unit": from_unit,
                    "to_unit": to_unit,
                    "supported_units": list(self.VOLUME_TO_LITERS.keys())
                },
                invalid_fields={"from_unit": f"Unit '{from_unit}' not recognized"}
            )

        liter_value = value * self.VOLUME_TO_LITERS[from_unit]

        # Convert from liters to target unit
        if to_unit not in self.VOLUME_TO_LITERS:
            raise ValidationError(
                message=f"Unknown volume unit: {to_unit}",
                context={
                    "from_unit": from_unit,
                    "to_unit": to_unit,
                    "supported_units": list(self.VOLUME_TO_LITERS.keys())
                },
                invalid_fields={"to_unit": f"Unit '{to_unit}' not recognized"}
            )

        return liter_value / self.VOLUME_TO_LITERS[to_unit]

    def convert_fuel_to_energy(
        self, value: float, fuel_unit: str, fuel_type: str, energy_unit: str = "MMBtu"
    ) -> float:
        """Convert fuel consumption to energy content.

        Args:
            value: The fuel consumption value
            fuel_unit: The unit of fuel consumption
            fuel_type: The type of fuel
            energy_unit: The target energy unit (default: MMBtu)

        Returns:
            float: The energy content in the target unit

        Raises:
            ValueError: If fuel type or unit is not recognized
        """
        if fuel_type not in self.FUEL_ENERGY_CONTENT:
            # Try generic energy conversion
            return self.convert_energy(value, fuel_unit, energy_unit)

        fuel_factors = self.FUEL_ENERGY_CONTENT[fuel_type]

        if fuel_unit not in fuel_factors:
            raise ValidationError(
                message=f"Unknown unit '{fuel_unit}' for fuel type '{fuel_type}'",
                context={
                    "fuel_type": fuel_type,
                    "fuel_unit": fuel_unit,
                    "energy_unit": energy_unit,
                    "supported_units": list(fuel_factors.keys())
                },
                invalid_fields={"fuel_unit": f"Unit '{fuel_unit}' not supported for fuel type '{fuel_type}'"}
            )

        # Convert to MMBtu
        mmbtu_value = value * fuel_factors[fuel_unit]

        # Convert to target energy unit if not MMBtu
        if energy_unit != "MMBtu":
            return self.convert_energy(mmbtu_value, "MMBtu", energy_unit)

        return mmbtu_value

    def convert_emissions(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert emissions between different units.

        Args:
            value: The emissions value
            from_unit: The unit to convert from (e.g., kgCO2e, tCO2e)
            to_unit: The unit to convert to

        Returns:
            float: The converted value
        """
        # Common emission unit conversions
        emission_conversions = {
            "kgCO2e": 1.0,
            "kg": 1.0,
            "tCO2e": 1000.0,
            "tons": 1000.0,
            "metric_tons": 1000.0,
            "tonnes": 1000.0,
            "lbCO2e": 0.453592,
            "lbs": 0.453592,
            "short_tons": 907.185,
            "MTCO2e": 1000.0,  # Metric tons CO2e
            "gCO2e": 0.001,
            "g": 0.001,
        }

        if from_unit == to_unit:
            return value

        # Convert to kg first
        if from_unit not in emission_conversions:
            raise ValidationError(
                message=f"Unknown emission unit: {from_unit}",
                context={
                    "from_unit": from_unit,
                    "to_unit": to_unit,
                    "supported_units": list(emission_conversions.keys())
                },
                invalid_fields={"from_unit": f"Unit '{from_unit}' not recognized"}
            )

        kg_value = value * emission_conversions[from_unit]

        # Convert from kg to target unit
        if to_unit not in emission_conversions:
            raise ValidationError(
                message=f"Unknown emission unit: {to_unit}",
                context={
                    "from_unit": from_unit,
                    "to_unit": to_unit,
                    "supported_units": list(emission_conversions.keys())
                },
                invalid_fields={"to_unit": f"Unit '{to_unit}' not recognized"}
            )

        return kg_value / emission_conversions[to_unit]

    def normalize_unit_name(self, unit: str) -> str:
        """Normalize unit names to standard format.

        Args:
            unit: The unit name to normalize

        Returns:
            str: The normalized unit name
        """
        # Common unit name variations
        unit_aliases = {
            "square_feet": "sqft",
            "square_foot": "sqft",
            "sq_ft": "sqft",
            "square_meters": "sqm",
            "square_meter": "sqm",
            "sq_m": "sqm",
            "kilowatt_hour": "kWh",
            "kilowatt_hours": "kWh",
            "megawatt_hour": "MWh",
            "megawatt_hours": "MWh",
            "million_btu": "MMBtu",
            "mmbtu": "MMBtu",
            "thousand_cubic_feet": "mcf",
            "hundred_cubic_feet": "ccf",
            "cubic_meters": "m3",
            "cubic_meter": "m3",
            "liters": "liter",
            "litres": "liter",
            "litre": "liter",
        }

        lower_unit = unit.lower().replace("-", "_").replace(" ", "_")
        return unit_aliases.get(lower_unit, unit)

    def get_conversion_factor(
        self, from_unit: str, to_unit: str, conversion_type: str = "energy"
    ) -> float:
        """Get the conversion factor between two units.

        Args:
            from_unit: The unit to convert from
            to_unit: The unit to convert to
            conversion_type: Type of conversion (energy, area, mass, volume)

        Returns:
            float: The conversion factor

        Raises:
            ValidationError: If conversion type is not recognized
        """
        if conversion_type == "energy":
            return self.convert_energy(1.0, from_unit, to_unit)
        elif conversion_type == "area":
            return self.convert_area(1.0, from_unit, to_unit)
        elif conversion_type == "mass":
            return self.convert_mass(1.0, from_unit, to_unit)
        elif conversion_type == "volume":
            return self.convert_volume(1.0, from_unit, to_unit)
        else:
            raise ValidationError(
                message=f"Unknown conversion type: {conversion_type}",
                context={
                    "conversion_type": conversion_type,
                    "from_unit": from_unit,
                    "to_unit": to_unit,
                    "supported_types": ["energy", "area", "mass", "volume"]
                },
                invalid_fields={"conversion_type": f"Type '{conversion_type}' not recognized"}
            )

    def convert(
        self,
        value: Union[float, Decimal],
        from_unit: str,
        to_unit: str,
    ) -> float:
        """
        Generic convert method - automatically detects unit category and performs conversion.

        This is the universal conversion method from the calculation engine.
        Supports: energy, volume, mass, distance, area, time.

        Args:
            value: Numerical value to convert
            from_unit: Source unit (e.g., 'gallon', 'kwh')
            to_unit: Target unit (e.g., 'liter', 'mwh')

        Returns:
            Converted value as float

        Raises:
            UnitConversionError: If units unknown or incompatible
        """
        # Normalize units (lowercase, strip whitespace)
        from_unit = from_unit.lower().strip()
        to_unit = to_unit.lower().strip()

        # If units are the same, no conversion needed
        if from_unit == to_unit:
            return float(value)

        # Convert to Decimal for precision
        if not isinstance(value, Decimal):
            value = Decimal(str(value))

        # Determine unit category
        from_category = self._get_unit_category(from_unit)
        to_category = self._get_unit_category(to_unit)

        if from_category is None:
            raise UnitConversionError(f"Unknown unit: {from_unit}")

        if to_category is None:
            raise UnitConversionError(f"Unknown unit: {to_unit}")

        if from_category != to_category:
            raise UnitConversionError(
                f"Cannot convert between different unit types: {from_unit} ({from_category}) → {to_unit} ({to_category})"
            )

        # Get conversion table for this category
        conversion_table = self.conversion_tables[from_category]

        # Convert: from_unit → base_unit → to_unit
        base_value = value * conversion_table[from_unit]
        converted_value = base_value / conversion_table[to_unit]

        return float(converted_value)

    def _get_unit_category(self, unit: str) -> Union[str, None]:
        """
        Determine which category a unit belongs to.

        Args:
            unit: Unit string (e.g., 'kwh', 'gallon')

        Returns:
            Category name ('energy', 'volume', etc.) or None if unknown
        """
        for category, conversion_table in self.conversion_tables.items():
            if unit in conversion_table:
                return category
        return None

    def is_compatible(self, unit1: str, unit2: str) -> bool:
        """
        Check if two units are compatible (same category).

        Args:
            unit1: First unit
            unit2: Second unit

        Returns:
            True if compatible, False otherwise
        """
        unit1 = unit1.lower().strip()
        unit2 = unit2.lower().strip()

        category1 = self._get_unit_category(unit1)
        category2 = self._get_unit_category(unit2)

        return category1 is not None and category1 == category2

    def get_unit_category(self, unit: str) -> str:
        """
        Get category for a unit.

        Args:
            unit: Unit string

        Returns:
            Category name

        Raises:
            UnitConversionError: If unit unknown
        """
        category = self._get_unit_category(unit.lower().strip())
        if category is None:
            raise UnitConversionError(f"Unknown unit: {unit}")
        return category

    def list_supported_units(self, category: Union[str, None] = None) -> Dict[str, list]:
        """
        List all supported units.

        Args:
            category: Optional category filter ('energy', 'volume', etc.)

        Returns:
            Dictionary mapping categories to unit lists
        """
        if category:
            if category not in self.conversion_tables:
                raise ValueError(f"Unknown category: {category}")
            return {category: list(self.conversion_tables[category].keys())}

        return {
            cat: list(table.keys())
            for cat, table in self.conversion_tables.items()
        }
