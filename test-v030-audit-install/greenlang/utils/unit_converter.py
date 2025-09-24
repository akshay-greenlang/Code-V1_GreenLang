"""
Unit Converter Utility

Centralized unit conversion library for GreenLang agents.
Handles energy, fuel, area, and emission unit conversions.
"""

import logging


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

    def __init__(self):
        """Initialize the unit converter with logging."""
        self.logger = logging.getLogger(__name__)

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
            raise ValueError(f"Unknown energy unit: {from_unit}")

        mmbtu_value = value * self.ENERGY_TO_MMBTU[from_unit]

        # Convert from MMBtu to target unit
        if to_unit not in self.ENERGY_TO_MMBTU:
            raise ValueError(f"Unknown energy unit: {to_unit}")

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
            raise ValueError(f"Unknown area unit: {from_unit}")

        sqft_value = value * self.AREA_TO_SQFT[from_unit]

        # Convert from sqft to target unit
        if to_unit not in self.AREA_TO_SQFT:
            raise ValueError(f"Unknown area unit: {to_unit}")

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
            raise ValueError(f"Unknown mass unit: {from_unit}")

        kg_value = value * self.MASS_TO_KG[from_unit]

        # Convert from kg to target unit
        if to_unit not in self.MASS_TO_KG:
            raise ValueError(f"Unknown mass unit: {to_unit}")

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
            raise ValueError(f"Unknown volume unit: {from_unit}")

        liter_value = value * self.VOLUME_TO_LITERS[from_unit]

        # Convert from liters to target unit
        if to_unit not in self.VOLUME_TO_LITERS:
            raise ValueError(f"Unknown volume unit: {to_unit}")

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
            raise ValueError(f"Unknown unit '{fuel_unit}' for fuel type '{fuel_type}'")

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
            raise ValueError(f"Unknown emission unit: {from_unit}")

        kg_value = value * emission_conversions[from_unit]

        # Convert from kg to target unit
        if to_unit not in emission_conversions:
            raise ValueError(f"Unknown emission unit: {to_unit}")

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
            raise ValueError(f"Unknown conversion type: {conversion_type}")
