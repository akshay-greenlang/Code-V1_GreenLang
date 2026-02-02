"""
Unit Converter for GHG Calculations

This module provides comprehensive unit conversion capabilities for
emission factor calculations. It supports energy, mass, volume, and
distance conversions with high precision.

Supported Unit Categories:
- Energy: kWh, MWh, GJ, MJ, therm, mmBtu, BTU
- Mass: kg, t, g, lb, short_ton, long_ton, oz
- Volume: L, gal (US), gal (UK), m3, cf, barrel
- Distance: km, mi, m, ft, nautical_mile
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class UnitCategory(Enum):
    """Categories of units for conversion."""
    ENERGY = "energy"
    MASS = "mass"
    VOLUME = "volume"
    DISTANCE = "distance"
    AREA = "area"
    POWER = "power"
    EMISSION_INTENSITY = "emission_intensity"


@dataclass
class ConversionResult:
    """Result of a unit conversion operation."""
    value: Decimal
    from_unit: str
    to_unit: str
    conversion_factor: Decimal
    success: bool
    error_message: Optional[str] = None


class UnitConverter:
    """
    Comprehensive unit converter for GHG calculations.

    Supports conversion between all common units used in
    emission factor calculations and GHG reporting.
    """

    # Energy conversion factors (all relative to kWh)
    ENERGY_FACTORS: Dict[str, Decimal] = {
        # Base unit: kWh
        "kwh": Decimal("1"),
        "kWh": Decimal("1"),
        "mwh": Decimal("1000"),
        "MWh": Decimal("1000"),
        "gwh": Decimal("1000000"),
        "GWh": Decimal("1000000"),
        "twh": Decimal("1000000000"),
        "TWh": Decimal("1000000000"),
        # Joules
        "j": Decimal("0.000000277778"),
        "J": Decimal("0.000000277778"),
        "kj": Decimal("0.000277778"),
        "kJ": Decimal("0.000277778"),
        "mj": Decimal("0.277778"),
        "MJ": Decimal("0.277778"),
        "gj": Decimal("277.778"),
        "GJ": Decimal("277.778"),
        "tj": Decimal("277778"),
        "TJ": Decimal("277778"),
        # BTU
        "btu": Decimal("0.000293071"),
        "BTU": Decimal("0.000293071"),
        "mmbtu": Decimal("293.071"),
        "mmBtu": Decimal("293.071"),
        "MMBTU": Decimal("293.071"),
        # Therms
        "therm": Decimal("29.3071"),
        "therms": Decimal("29.3071"),
        "thm": Decimal("29.3071"),
        "dth": Decimal("29.3071"),  # dekatherm
        # Other
        "wh": Decimal("0.001"),
        "Wh": Decimal("0.001"),
        "toe": Decimal("11630"),  # tonne oil equivalent
        "TOE": Decimal("11630"),
        "mtoe": Decimal("11630000"),  # million toe
        "MTOE": Decimal("11630000"),
        "kcal": Decimal("0.001163"),
        "cal": Decimal("0.00000116"),
    }

    # Mass conversion factors (all relative to kg)
    MASS_FACTORS: Dict[str, Decimal] = {
        # Base unit: kg
        "kg": Decimal("1"),
        "g": Decimal("0.001"),
        "mg": Decimal("0.000001"),
        "t": Decimal("1000"),
        "tonne": Decimal("1000"),
        "tonnes": Decimal("1000"),
        "metric_ton": Decimal("1000"),
        "mt": Decimal("1000"),
        "Mt": Decimal("1000000"),  # megatonne
        "kt": Decimal("1000000"),  # kilotonne
        "Gt": Decimal("1000000000"),  # gigatonne
        # Imperial
        "lb": Decimal("0.453592"),
        "lbs": Decimal("0.453592"),
        "pound": Decimal("0.453592"),
        "pounds": Decimal("0.453592"),
        "oz": Decimal("0.0283495"),
        "ounce": Decimal("0.0283495"),
        "ounces": Decimal("0.0283495"),
        "short_ton": Decimal("907.185"),  # US ton
        "us_ton": Decimal("907.185"),
        "long_ton": Decimal("1016.05"),  # UK ton
        "uk_ton": Decimal("1016.05"),
        "stone": Decimal("6.35029"),
        # CO2 specific
        "tco2": Decimal("1000"),  # tonne CO2
        "tCO2": Decimal("1000"),
        "tco2e": Decimal("1000"),  # tonne CO2e
        "tCO2e": Decimal("1000"),
        "mtco2": Decimal("1000000"),  # million tonnes CO2
        "MtCO2": Decimal("1000000"),
        "mtco2e": Decimal("1000000"),
        "MtCO2e": Decimal("1000000"),
        "gtco2": Decimal("1000000000"),  # gigatonne CO2
        "GtCO2": Decimal("1000000000"),
        "kgco2": Decimal("1"),
        "kgCO2": Decimal("1"),
        "kgco2e": Decimal("1"),
        "kgCO2e": Decimal("1"),
    }

    # Volume conversion factors (all relative to liters)
    VOLUME_FACTORS: Dict[str, Decimal] = {
        # Base unit: liter
        "l": Decimal("1"),
        "L": Decimal("1"),
        "litre": Decimal("1"),
        "liter": Decimal("1"),
        "litres": Decimal("1"),
        "liters": Decimal("1"),
        "ml": Decimal("0.001"),
        "mL": Decimal("0.001"),
        # Cubic meters
        "m3": Decimal("1000"),
        "m^3": Decimal("1000"),
        "cubic_metre": Decimal("1000"),
        "cubic_meter": Decimal("1000"),
        "scm": Decimal("1000"),  # standard cubic meter
        "Nm3": Decimal("1000"),  # normal cubic meter
        # Gallons
        "gal": Decimal("3.78541"),  # US gallon
        "gallon": Decimal("3.78541"),
        "us_gal": Decimal("3.78541"),
        "us_gallon": Decimal("3.78541"),
        "uk_gal": Decimal("4.54609"),  # Imperial gallon
        "imp_gal": Decimal("4.54609"),
        "imp_gallon": Decimal("4.54609"),
        # Cubic feet
        "cf": Decimal("28.3168"),
        "ft3": Decimal("28.3168"),
        "cubic_foot": Decimal("28.3168"),
        "cubic_feet": Decimal("28.3168"),
        "scf": Decimal("28.3168"),  # standard cubic foot
        "mcf": Decimal("28316.8"),  # thousand cubic feet
        "Mcf": Decimal("28316.8"),
        "mmcf": Decimal("28316800"),  # million cubic feet
        "MMcf": Decimal("28316800"),
        # Barrels
        "bbl": Decimal("158.987"),  # oil barrel
        "barrel": Decimal("158.987"),
        "barrels": Decimal("158.987"),
        "boe": Decimal("158.987"),  # barrel oil equivalent
        "mboe": Decimal("158987"),  # thousand boe
        # Other
        "pt": Decimal("0.473176"),  # US pint
        "qt": Decimal("0.946353"),  # US quart
    }

    # Distance conversion factors (all relative to km)
    DISTANCE_FACTORS: Dict[str, Decimal] = {
        # Base unit: km
        "km": Decimal("1"),
        "m": Decimal("0.001"),
        "cm": Decimal("0.00001"),
        "mm": Decimal("0.000001"),
        # Imperial
        "mi": Decimal("1.60934"),
        "mile": Decimal("1.60934"),
        "miles": Decimal("1.60934"),
        "ft": Decimal("0.0003048"),
        "foot": Decimal("0.0003048"),
        "feet": Decimal("0.0003048"),
        "yd": Decimal("0.0009144"),
        "yard": Decimal("0.0009144"),
        "yards": Decimal("0.0009144"),
        "in": Decimal("0.0000254"),
        "inch": Decimal("0.0000254"),
        "inches": Decimal("0.0000254"),
        # Nautical
        "nmi": Decimal("1.852"),  # nautical mile
        "nautical_mile": Decimal("1.852"),
        "nautical_miles": Decimal("1.852"),
        # Transport specific
        "pkm": Decimal("1"),  # passenger-km
        "passenger_km": Decimal("1"),
        "tkm": Decimal("1"),  # tonne-km
        "tonne_km": Decimal("1"),
        "vkm": Decimal("1"),  # vehicle-km
        "vehicle_km": Decimal("1"),
    }

    # Area conversion factors (all relative to m2)
    AREA_FACTORS: Dict[str, Decimal] = {
        # Base unit: m2
        "m2": Decimal("1"),
        "m^2": Decimal("1"),
        "sqm": Decimal("1"),
        "sq_m": Decimal("1"),
        "square_metre": Decimal("1"),
        "square_meter": Decimal("1"),
        # Metric
        "km2": Decimal("1000000"),
        "sq_km": Decimal("1000000"),
        "ha": Decimal("10000"),
        "hectare": Decimal("10000"),
        "hectares": Decimal("10000"),
        "cm2": Decimal("0.0001"),
        # Imperial
        "ft2": Decimal("0.092903"),
        "sq_ft": Decimal("0.092903"),
        "square_foot": Decimal("0.092903"),
        "square_feet": Decimal("0.092903"),
        "acre": Decimal("4046.86"),
        "acres": Decimal("4046.86"),
        "mi2": Decimal("2589988"),
        "sq_mi": Decimal("2589988"),
        "square_mile": Decimal("2589988"),
    }

    # Power conversion factors (all relative to kW)
    POWER_FACTORS: Dict[str, Decimal] = {
        # Base unit: kW
        "kw": Decimal("1"),
        "kW": Decimal("1"),
        "w": Decimal("0.001"),
        "W": Decimal("0.001"),
        "mw": Decimal("1000"),
        "MW": Decimal("1000"),
        "gw": Decimal("1000000"),
        "GW": Decimal("1000000"),
        "tw": Decimal("1000000000"),
        "TW": Decimal("1000000000"),
        # Other
        "hp": Decimal("0.745700"),
        "horsepower": Decimal("0.745700"),
        "btu_hr": Decimal("0.000293071"),
        "BTU/hr": Decimal("0.000293071"),
    }

    # All conversion factor dictionaries by category
    CONVERSION_FACTORS: Dict[UnitCategory, Dict[str, Decimal]] = {
        UnitCategory.ENERGY: ENERGY_FACTORS,
        UnitCategory.MASS: MASS_FACTORS,
        UnitCategory.VOLUME: VOLUME_FACTORS,
        UnitCategory.DISTANCE: DISTANCE_FACTORS,
        UnitCategory.AREA: AREA_FACTORS,
        UnitCategory.POWER: POWER_FACTORS,
    }

    def __init__(self, precision: int = 10):
        """
        Initialize the unit converter.

        Args:
            precision: Number of decimal places for results
        """
        self.precision = precision

    def convert(
        self,
        value: Union[float, Decimal, str],
        from_unit: str,
        to_unit: str,
        category: Optional[UnitCategory] = None
    ) -> ConversionResult:
        """
        Convert a value from one unit to another.

        Args:
            value: The value to convert
            from_unit: Source unit
            to_unit: Target unit
            category: Unit category (auto-detected if not provided)

        Returns:
            ConversionResult with converted value
        """
        try:
            # Convert value to Decimal
            decimal_value = Decimal(str(value))

            # Auto-detect category if not provided
            if category is None:
                category = self._detect_category(from_unit, to_unit)

            if category is None:
                return ConversionResult(
                    value=Decimal("0"),
                    from_unit=from_unit,
                    to_unit=to_unit,
                    conversion_factor=Decimal("0"),
                    success=False,
                    error_message=f"Could not determine unit category for {from_unit} to {to_unit}"
                )

            # Get conversion factors
            factors = self.CONVERSION_FACTORS.get(category, {})

            # Normalize unit names
            from_unit_norm = self._normalize_unit(from_unit)
            to_unit_norm = self._normalize_unit(to_unit)

            from_factor = factors.get(from_unit_norm) or factors.get(from_unit)
            to_factor = factors.get(to_unit_norm) or factors.get(to_unit)

            if from_factor is None:
                return ConversionResult(
                    value=Decimal("0"),
                    from_unit=from_unit,
                    to_unit=to_unit,
                    conversion_factor=Decimal("0"),
                    success=False,
                    error_message=f"Unknown source unit: {from_unit}"
                )

            if to_factor is None:
                return ConversionResult(
                    value=Decimal("0"),
                    from_unit=from_unit,
                    to_unit=to_unit,
                    conversion_factor=Decimal("0"),
                    success=False,
                    error_message=f"Unknown target unit: {to_unit}"
                )

            # Calculate conversion factor
            conversion_factor = from_factor / to_factor

            # Convert value
            result = decimal_value * conversion_factor

            # Round to precision
            result = result.quantize(
                Decimal(10) ** -self.precision,
                rounding=ROUND_HALF_UP
            )

            return ConversionResult(
                value=result,
                from_unit=from_unit,
                to_unit=to_unit,
                conversion_factor=conversion_factor,
                success=True
            )

        except Exception as e:
            logger.error(f"Conversion error: {e}")
            return ConversionResult(
                value=Decimal("0"),
                from_unit=from_unit,
                to_unit=to_unit,
                conversion_factor=Decimal("0"),
                success=False,
                error_message=str(e)
            )

    def _detect_category(self, from_unit: str, to_unit: str) -> Optional[UnitCategory]:
        """Detect the unit category based on unit names."""
        from_norm = self._normalize_unit(from_unit)
        to_norm = self._normalize_unit(to_unit)

        for category, factors in self.CONVERSION_FACTORS.items():
            if from_norm in factors or from_unit in factors:
                if to_norm in factors or to_unit in factors:
                    return category

        return None

    def _normalize_unit(self, unit: str) -> str:
        """Normalize unit name for lookup."""
        # Handle common variations
        unit = unit.strip()

        # Common normalizations
        normalizations = {
            "kilowatt-hour": "kWh",
            "kilowatt hour": "kWh",
            "kilowatt_hour": "kWh",
            "megawatt-hour": "MWh",
            "megawatt hour": "MWh",
            "megawatt_hour": "MWh",
            "kilogram": "kg",
            "kilograms": "kg",
            "metric ton": "t",
            "metric_ton": "t",
            "metric tons": "t",
            "liter": "L",
            "liters": "L",
            "litre": "L",
            "litres": "L",
            "gallon": "gal",
            "gallons": "gal",
            "kilometer": "km",
            "kilometers": "km",
            "kilometre": "km",
            "kilometres": "km",
        }

        return normalizations.get(unit.lower(), unit)

    # Convenience methods for common conversions

    def kwh_to_gj(self, kwh: Union[float, Decimal]) -> Decimal:
        """Convert kWh to GJ."""
        result = self.convert(kwh, "kWh", "GJ", UnitCategory.ENERGY)
        return result.value if result.success else Decimal("0")

    def gj_to_kwh(self, gj: Union[float, Decimal]) -> Decimal:
        """Convert GJ to kWh."""
        result = self.convert(gj, "GJ", "kWh", UnitCategory.ENERGY)
        return result.value if result.success else Decimal("0")

    def mmbtu_to_kwh(self, mmbtu: Union[float, Decimal]) -> Decimal:
        """Convert mmBtu to kWh."""
        result = self.convert(mmbtu, "mmBtu", "kWh", UnitCategory.ENERGY)
        return result.value if result.success else Decimal("0")

    def therms_to_kwh(self, therms: Union[float, Decimal]) -> Decimal:
        """Convert therms to kWh."""
        result = self.convert(therms, "therm", "kWh", UnitCategory.ENERGY)
        return result.value if result.success else Decimal("0")

    def kg_to_tonnes(self, kg: Union[float, Decimal]) -> Decimal:
        """Convert kg to tonnes."""
        result = self.convert(kg, "kg", "t", UnitCategory.MASS)
        return result.value if result.success else Decimal("0")

    def tonnes_to_kg(self, tonnes: Union[float, Decimal]) -> Decimal:
        """Convert tonnes to kg."""
        result = self.convert(tonnes, "t", "kg", UnitCategory.MASS)
        return result.value if result.success else Decimal("0")

    def lb_to_kg(self, lb: Union[float, Decimal]) -> Decimal:
        """Convert pounds to kg."""
        result = self.convert(lb, "lb", "kg", UnitCategory.MASS)
        return result.value if result.success else Decimal("0")

    def gallons_to_liters(self, gallons: Union[float, Decimal]) -> Decimal:
        """Convert US gallons to liters."""
        result = self.convert(gallons, "gal", "L", UnitCategory.VOLUME)
        return result.value if result.success else Decimal("0")

    def liters_to_gallons(self, liters: Union[float, Decimal]) -> Decimal:
        """Convert liters to US gallons."""
        result = self.convert(liters, "L", "gal", UnitCategory.VOLUME)
        return result.value if result.success else Decimal("0")

    def m3_to_liters(self, m3: Union[float, Decimal]) -> Decimal:
        """Convert cubic meters to liters."""
        result = self.convert(m3, "m3", "L", UnitCategory.VOLUME)
        return result.value if result.success else Decimal("0")

    def cf_to_m3(self, cf: Union[float, Decimal]) -> Decimal:
        """Convert cubic feet to cubic meters."""
        result = self.convert(cf, "cf", "m3", UnitCategory.VOLUME)
        return result.value if result.success else Decimal("0")

    def mcf_to_m3(self, mcf: Union[float, Decimal]) -> Decimal:
        """Convert thousand cubic feet (Mcf) to cubic meters."""
        result = self.convert(mcf, "Mcf", "m3", UnitCategory.VOLUME)
        return result.value if result.success else Decimal("0")

    def miles_to_km(self, miles: Union[float, Decimal]) -> Decimal:
        """Convert miles to kilometers."""
        result = self.convert(miles, "mi", "km", UnitCategory.DISTANCE)
        return result.value if result.success else Decimal("0")

    def km_to_miles(self, km: Union[float, Decimal]) -> Decimal:
        """Convert kilometers to miles."""
        result = self.convert(km, "km", "mi", UnitCategory.DISTANCE)
        return result.value if result.success else Decimal("0")

    def get_supported_units(self, category: UnitCategory) -> list:
        """Get list of supported units for a category."""
        factors = self.CONVERSION_FACTORS.get(category, {})
        return list(factors.keys())

    def get_all_categories(self) -> list:
        """Get all supported unit categories."""
        return list(UnitCategory)

    def convert_emission_factor(
        self,
        factor_value: Decimal,
        from_mass_unit: str,
        from_activity_unit: str,
        to_mass_unit: str,
        to_activity_unit: str,
        activity_category: UnitCategory
    ) -> Tuple[Decimal, bool]:
        """
        Convert an emission factor from one unit basis to another.

        Example: Convert kg CO2/gallon to kg CO2/liter

        Args:
            factor_value: The emission factor value
            from_mass_unit: Source mass unit (e.g., "kg")
            from_activity_unit: Source activity unit (e.g., "gallon")
            to_mass_unit: Target mass unit (e.g., "kg")
            to_activity_unit: Target activity unit (e.g., "L")
            activity_category: Category of activity unit (ENERGY, VOLUME, etc.)

        Returns:
            Tuple of (converted_value, success)
        """
        try:
            # Convert mass unit
            mass_result = self.convert(1, from_mass_unit, to_mass_unit, UnitCategory.MASS)
            if not mass_result.success:
                return (Decimal("0"), False)

            # Convert activity unit
            activity_result = self.convert(1, from_activity_unit, to_activity_unit, activity_category)
            if not activity_result.success:
                return (Decimal("0"), False)

            # Emission factor conversion:
            # If converting from kg/gal to kg/L, we multiply by (mass_factor) and divide by (activity_factor)
            # Because: kg_new/activity_new = (kg_old * mass_factor) / (activity_old * activity_factor)
            # Since we want per activity_new, we need to account for the ratio
            converted = factor_value * mass_result.conversion_factor / activity_result.conversion_factor

            return (converted.quantize(Decimal(10) ** -self.precision), True)

        except Exception as e:
            logger.error(f"Emission factor conversion error: {e}")
            return (Decimal("0"), False)


# Singleton instance
_converter_instance: Optional[UnitConverter] = None


def get_unit_converter(precision: int = 10) -> UnitConverter:
    """
    Get or create the UnitConverter singleton.

    Args:
        precision: Decimal precision for conversions

    Returns:
        UnitConverter instance
    """
    global _converter_instance

    if _converter_instance is None:
        _converter_instance = UnitConverter(precision=precision)

    return _converter_instance
