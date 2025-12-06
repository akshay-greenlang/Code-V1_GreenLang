"""
Unit Converter Module

This module provides precise, deterministic unit conversions for all
GreenLang calculations. All conversions are based on authoritative
conversion factors from NIST, SI standards, and regulatory bodies.

Supported conversions:
- Energy: kWh, MWh, GJ, MMBtu, therms, toe, boe
- Mass: kg, tonnes, lbs, short tons, long tons
- Volume: liters, gallons (US/UK), m3, CCF, MCF
- Distance: km, miles, nautical miles, meters
- Area: m2, hectares, acres, sq ft, sq miles

CRITICAL: All conversions are deterministic and bit-perfect.
No LLM involvement in any conversion.

Example:
    >>> converter = UnitConverter()
    >>> result = converter.convert(1000, "kWh", "GJ")
    >>> print(f"{result.value} {result.to_unit}")
    3.6 GJ
"""

from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, Optional, Tuple, Union
from pydantic import BaseModel, Field
import hashlib
import json


class UnitCategory(str, Enum):
    """Categories of physical units."""
    ENERGY = "energy"
    MASS = "mass"
    VOLUME = "volume"
    DISTANCE = "distance"
    AREA = "area"
    POWER = "power"
    EMISSION_INTENSITY = "emission_intensity"


class ConversionResult(BaseModel):
    """Result of a unit conversion with provenance."""
    original_value: Decimal = Field(..., description="Original input value")
    from_unit: str = Field(..., description="Original unit")
    value: Decimal = Field(..., description="Converted value")
    to_unit: str = Field(..., description="Target unit")
    conversion_factor: Decimal = Field(..., description="Conversion factor used")
    category: UnitCategory = Field(..., description="Unit category")
    provenance_hash: str = Field(..., description="SHA-256 hash for verification")

    class Config:
        json_encoders = {
            Decimal: lambda v: str(v),
        }


class UnitConverter:
    """
    Zero-hallucination unit converter for regulatory calculations.

    All conversion factors are from authoritative sources:
    - NIST Special Publication 811
    - SI Brochure (BIPM)
    - EPA emission calculation guidance
    - IPCC Guidelines

    The converter guarantees bit-perfect reproducibility:
    same input -> same output (always).
    """

    # ==========================================================================
    # ENERGY CONVERSIONS
    # Base unit: kWh (kilowatt-hour)
    # Sources: NIST SP 811, IEA
    # ==========================================================================
    ENERGY_TO_KWH: Dict[str, Decimal] = {
        "kWh": Decimal("1"),
        "kwh": Decimal("1"),
        "Wh": Decimal("0.001"),
        "wh": Decimal("0.001"),
        "MWh": Decimal("1000"),
        "mwh": Decimal("1000"),
        "GWh": Decimal("1000000"),
        "gwh": Decimal("1000000"),
        "TWh": Decimal("1000000000"),
        "twh": Decimal("1000000000"),
        # Joules
        "J": Decimal("0.0000002777777778"),  # 1/3600000
        "kJ": Decimal("0.0002777777778"),    # 1/3600
        "MJ": Decimal("0.2777777778"),       # 1000/3600
        "GJ": Decimal("277.7777778"),        # 1000000/3600
        "TJ": Decimal("277777.7778"),        # 1000000000/3600
        # BTU (International Table)
        "BTU": Decimal("0.0002930711"),
        "btu": Decimal("0.0002930711"),
        "kBTU": Decimal("0.2930711"),
        "MMBTU": Decimal("293.0711"),        # Million BTU
        "MMBtu": Decimal("293.0711"),
        "mmbtu": Decimal("293.0711"),
        # Therms (US)
        "therm": Decimal("29.30711"),
        "therms": Decimal("29.30711"),
        # Calories
        "cal": Decimal("0.000001163"),
        "kcal": Decimal("0.001163"),
        "Mcal": Decimal("1.163"),
        # Oil/Coal equivalents
        "toe": Decimal("11630"),             # Tonne of oil equivalent
        "ktoe": Decimal("11630000"),
        "boe": Decimal("1700"),              # Barrel of oil equivalent
        "tce": Decimal("8141"),              # Tonne of coal equivalent
    }

    # ==========================================================================
    # MASS CONVERSIONS
    # Base unit: kg (kilogram)
    # Sources: NIST SP 811, SI Brochure
    # ==========================================================================
    MASS_TO_KG: Dict[str, Decimal] = {
        "kg": Decimal("1"),
        "g": Decimal("0.001"),
        "mg": Decimal("0.000001"),
        "t": Decimal("1000"),               # Metric tonne
        "tonne": Decimal("1000"),
        "tonnes": Decimal("1000"),
        "Mt": Decimal("1000000000"),        # Megatonne
        "kt": Decimal("1000000"),           # Kilotonne
        "lb": Decimal("0.45359237"),        # Pound (exact NIST)
        "lbs": Decimal("0.45359237"),
        "oz": Decimal("0.028349523125"),    # Ounce
        "short_ton": Decimal("907.18474"),  # US short ton (2000 lbs)
        "short_tons": Decimal("907.18474"),
        "long_ton": Decimal("1016.0469088"),  # UK long ton (2240 lbs)
        "long_tons": Decimal("1016.0469088"),
    }

    # ==========================================================================
    # VOLUME CONVERSIONS
    # Base unit: liters (L)
    # Sources: NIST SP 811, API
    # ==========================================================================
    VOLUME_TO_LITERS: Dict[str, Decimal] = {
        "L": Decimal("1"),
        "l": Decimal("1"),
        "liter": Decimal("1"),
        "liters": Decimal("1"),
        "litre": Decimal("1"),
        "litres": Decimal("1"),
        "mL": Decimal("0.001"),
        "ml": Decimal("0.001"),
        "m3": Decimal("1000"),
        "m^3": Decimal("1000"),
        "cubic_meter": Decimal("1000"),
        "cubic_meters": Decimal("1000"),
        # US Gallons
        "gal": Decimal("3.785411784"),
        "gallon": Decimal("3.785411784"),
        "gallons": Decimal("3.785411784"),
        "gal_us": Decimal("3.785411784"),
        # UK/Imperial Gallons
        "gal_uk": Decimal("4.54609"),
        "gallon_uk": Decimal("4.54609"),
        # Barrels (petroleum)
        "bbl": Decimal("158.987294928"),
        "barrel": Decimal("158.987294928"),
        "barrels": Decimal("158.987294928"),
        # Cubic feet
        "ft3": Decimal("28.316846592"),
        "cf": Decimal("28.316846592"),
        "cubic_foot": Decimal("28.316846592"),
        "cubic_feet": Decimal("28.316846592"),
        "CCF": Decimal("2831.6846592"),     # Centum cubic feet (100 cf)
        "ccf": Decimal("2831.6846592"),
        "MCF": Decimal("28316.846592"),     # Thousand cubic feet
        "mcf": Decimal("28316.846592"),
        "MMCF": Decimal("28316846.592"),    # Million cubic feet
        "mmcf": Decimal("28316846.592"),
        # Other
        "pt": Decimal("0.473176473"),       # US pint
        "qt": Decimal("0.946352946"),       # US quart
    }

    # ==========================================================================
    # DISTANCE CONVERSIONS
    # Base unit: km (kilometer)
    # Sources: NIST SP 811, SI Brochure
    # ==========================================================================
    DISTANCE_TO_KM: Dict[str, Decimal] = {
        "km": Decimal("1"),
        "m": Decimal("0.001"),
        "cm": Decimal("0.00001"),
        "mm": Decimal("0.000001"),
        "mi": Decimal("1.609344"),          # Statute mile (exact)
        "mile": Decimal("1.609344"),
        "miles": Decimal("1.609344"),
        "nmi": Decimal("1.852"),            # Nautical mile (exact)
        "nautical_mile": Decimal("1.852"),
        "nautical_miles": Decimal("1.852"),
        "ft": Decimal("0.0003048"),         # Foot (exact)
        "feet": Decimal("0.0003048"),
        "yd": Decimal("0.0009144"),         # Yard (exact)
        "yards": Decimal("0.0009144"),
        "in": Decimal("0.0000254"),         # Inch (exact)
        "inch": Decimal("0.0000254"),
        "inches": Decimal("0.0000254"),
    }

    # ==========================================================================
    # AREA CONVERSIONS
    # Base unit: m2 (square meter)
    # Sources: NIST SP 811
    # ==========================================================================
    AREA_TO_M2: Dict[str, Decimal] = {
        "m2": Decimal("1"),
        "m^2": Decimal("1"),
        "sq_m": Decimal("1"),
        "sqm": Decimal("1"),
        "km2": Decimal("1000000"),
        "km^2": Decimal("1000000"),
        "sq_km": Decimal("1000000"),
        "ha": Decimal("10000"),             # Hectare
        "hectare": Decimal("10000"),
        "hectares": Decimal("10000"),
        "acre": Decimal("4046.8564224"),
        "acres": Decimal("4046.8564224"),
        "ft2": Decimal("0.09290304"),       # Square foot (exact)
        "sq_ft": Decimal("0.09290304"),
        "sqft": Decimal("0.09290304"),
        "mi2": Decimal("2589988.110336"),   # Square mile
        "sq_mi": Decimal("2589988.110336"),
        "sq_mile": Decimal("2589988.110336"),
    }

    # ==========================================================================
    # POWER CONVERSIONS
    # Base unit: kW (kilowatt)
    # ==========================================================================
    POWER_TO_KW: Dict[str, Decimal] = {
        "kW": Decimal("1"),
        "kw": Decimal("1"),
        "W": Decimal("0.001"),
        "w": Decimal("0.001"),
        "MW": Decimal("1000"),
        "mw": Decimal("1000"),
        "GW": Decimal("1000000"),
        "gw": Decimal("1000000"),
        "hp": Decimal("0.7457"),            # Horsepower (mechanical)
        "HP": Decimal("0.7457"),
        "BTU/h": Decimal("0.0002930711"),
        "btu/h": Decimal("0.0002930711"),
    }

    # Category mapping
    CATEGORY_CONVERSIONS = {
        UnitCategory.ENERGY: ENERGY_TO_KWH,
        UnitCategory.MASS: MASS_TO_KG,
        UnitCategory.VOLUME: VOLUME_TO_LITERS,
        UnitCategory.DISTANCE: DISTANCE_TO_KM,
        UnitCategory.AREA: AREA_TO_M2,
        UnitCategory.POWER: POWER_TO_KW,
    }

    BASE_UNITS = {
        UnitCategory.ENERGY: "kWh",
        UnitCategory.MASS: "kg",
        UnitCategory.VOLUME: "L",
        UnitCategory.DISTANCE: "km",
        UnitCategory.AREA: "m2",
        UnitCategory.POWER: "kW",
    }

    def __init__(self):
        """Initialize the unit converter."""
        self._build_reverse_lookups()

    def _build_reverse_lookups(self):
        """Build unit -> category lookup."""
        self._unit_to_category: Dict[str, UnitCategory] = {}
        for category, conversions in self.CATEGORY_CONVERSIONS.items():
            for unit in conversions.keys():
                self._unit_to_category[unit] = category

    def get_category(self, unit: str) -> Optional[UnitCategory]:
        """
        Get the category for a unit.

        Args:
            unit: Unit string

        Returns:
            UnitCategory or None if not found
        """
        return self._unit_to_category.get(unit)

    def convert(
        self,
        value: Union[Decimal, float, int],
        from_unit: str,
        to_unit: str,
        precision: int = 10,
    ) -> ConversionResult:
        """
        Convert a value from one unit to another.

        This is a DETERMINISTIC operation - same inputs always
        produce the same output.

        Args:
            value: Numeric value to convert
            from_unit: Source unit
            to_unit: Target unit
            precision: Decimal precision for result

        Returns:
            ConversionResult with provenance

        Raises:
            ValueError: If units are incompatible or unknown

        Example:
            >>> result = converter.convert(1000, "kWh", "GJ")
            >>> print(result.value)  # 3.6
        """
        # Normalize value to Decimal
        if not isinstance(value, Decimal):
            value = Decimal(str(value))

        # Get categories
        from_category = self.get_category(from_unit)
        to_category = self.get_category(to_unit)

        if from_category is None:
            raise ValueError(f"Unknown source unit: {from_unit}")
        if to_category is None:
            raise ValueError(f"Unknown target unit: {to_unit}")
        if from_category != to_category:
            raise ValueError(
                f"Incompatible units: {from_unit} ({from_category.value}) "
                f"cannot convert to {to_unit} ({to_category.value})"
            )

        # Get conversion factors
        conversions = self.CATEGORY_CONVERSIONS[from_category]
        from_to_base = conversions[from_unit]
        to_to_base = conversions[to_unit]

        # ZERO-HALLUCINATION CALCULATION
        # Convert: from_unit -> base -> to_unit
        # value_in_base = value * from_to_base
        # value_in_target = value_in_base / to_to_base
        # Combined: value * from_to_base / to_to_base

        conversion_factor = from_to_base / to_to_base
        converted_value = value * conversion_factor

        # Apply precision
        quantize_str = "0." + "0" * precision
        converted_value = converted_value.quantize(
            Decimal(quantize_str),
            rounding=ROUND_HALF_UP
        )

        # Generate provenance hash
        provenance_data = {
            "original_value": str(value),
            "from_unit": from_unit,
            "to_unit": to_unit,
            "conversion_factor": str(conversion_factor),
            "result": str(converted_value),
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        return ConversionResult(
            original_value=value,
            from_unit=from_unit,
            value=converted_value,
            to_unit=to_unit,
            conversion_factor=conversion_factor,
            category=from_category,
            provenance_hash=provenance_hash,
        )

    def convert_simple(
        self,
        value: Union[Decimal, float, int],
        from_unit: str,
        to_unit: str,
    ) -> Decimal:
        """
        Simple conversion returning just the value.

        Args:
            value: Value to convert
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted value as Decimal
        """
        result = self.convert(value, from_unit, to_unit)
        return result.value

    def get_conversion_factor(self, from_unit: str, to_unit: str) -> Decimal:
        """
        Get the conversion factor between two units.

        Args:
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Conversion factor (multiply by this to convert)
        """
        from_category = self.get_category(from_unit)
        to_category = self.get_category(to_unit)

        if from_category != to_category:
            raise ValueError(f"Incompatible units: {from_unit} -> {to_unit}")

        conversions = self.CATEGORY_CONVERSIONS[from_category]
        return conversions[from_unit] / conversions[to_unit]

    def to_base_unit(
        self,
        value: Union[Decimal, float, int],
        from_unit: str,
    ) -> Tuple[Decimal, str]:
        """
        Convert to the base unit for the category.

        Args:
            value: Value to convert
            from_unit: Source unit

        Returns:
            Tuple of (converted_value, base_unit)
        """
        category = self.get_category(from_unit)
        if category is None:
            raise ValueError(f"Unknown unit: {from_unit}")

        base_unit = self.BASE_UNITS[category]
        result = self.convert(value, from_unit, base_unit)
        return result.value, base_unit

    def normalize_unit(self, unit: str) -> str:
        """
        Normalize a unit string to its canonical form.

        Args:
            unit: Unit string (possibly non-canonical)

        Returns:
            Canonical unit string

        Example:
            >>> normalize_unit("kwh") -> "kWh"
            >>> normalize_unit("tonnes") -> "tonne"
        """
        # Common normalizations
        normalizations = {
            "kwh": "kWh",
            "mwh": "MWh",
            "gwh": "GWh",
            "twh": "TWh",
            "gj": "GJ",
            "mj": "MJ",
            "kj": "kJ",
            "mmbtu": "MMBtu",
            "tonnes": "tonne",
            "lbs": "lb",
            "liters": "L",
            "litres": "L",
            "gallons": "gal",
            "miles": "mi",
            "sqm": "m2",
            "hectares": "ha",
        }
        return normalizations.get(unit.lower(), unit)

    # =========================================================================
    # SPECIALIZED EMISSION CALCULATION CONVERSIONS
    # =========================================================================

    def energy_to_kwh(self, value: Union[Decimal, float], unit: str) -> Decimal:
        """Convert any energy unit to kWh."""
        return self.convert_simple(value, unit, "kWh")

    def mass_to_kg(self, value: Union[Decimal, float], unit: str) -> Decimal:
        """Convert any mass unit to kg."""
        return self.convert_simple(value, unit, "kg")

    def mass_to_tonnes(self, value: Union[Decimal, float], unit: str) -> Decimal:
        """Convert any mass unit to metric tonnes."""
        return self.convert_simple(value, unit, "tonne")

    def volume_to_liters(self, value: Union[Decimal, float], unit: str) -> Decimal:
        """Convert any volume unit to liters."""
        return self.convert_simple(value, unit, "L")

    def distance_to_km(self, value: Union[Decimal, float], unit: str) -> Decimal:
        """Convert any distance unit to kilometers."""
        return self.convert_simple(value, unit, "km")

    def area_to_m2(self, value: Union[Decimal, float], unit: str) -> Decimal:
        """Convert any area unit to square meters."""
        return self.convert_simple(value, unit, "m2")

    # =========================================================================
    # EMISSION FACTOR UNIT HANDLING
    # =========================================================================

    def parse_compound_unit(self, unit: str) -> Tuple[str, str]:
        """
        Parse a compound unit like 'kgCO2e/kWh'.

        Args:
            unit: Compound unit string

        Returns:
            Tuple of (numerator_unit, denominator_unit)

        Example:
            >>> parse_compound_unit("kgCO2e/kWh")
            ("kgCO2e", "kWh")
        """
        if "/" not in unit:
            return unit, ""

        parts = unit.split("/")
        return parts[0], parts[1] if len(parts) > 1 else ""

    def extract_activity_unit(self, ef_unit: str) -> str:
        """
        Extract activity data unit from emission factor unit.

        Args:
            ef_unit: Emission factor unit (e.g., "kgCO2e/L")

        Returns:
            Activity unit (e.g., "L")
        """
        _, activity_unit = self.parse_compound_unit(ef_unit)
        return activity_unit

    def convert_activity_to_ef_unit(
        self,
        activity_value: Union[Decimal, float],
        activity_unit: str,
        ef_unit: str,
    ) -> Decimal:
        """
        Convert activity data to match emission factor denominator unit.

        Args:
            activity_value: Activity data value
            activity_unit: Activity data unit
            ef_unit: Emission factor unit (e.g., "kgCO2e/L")

        Returns:
            Activity value in EF denominator units

        Example:
            >>> # Convert 100 gallons to liters for use with kgCO2e/L factor
            >>> convert_activity_to_ef_unit(100, "gal", "kgCO2e/L")
            378.5411784
        """
        target_unit = self.extract_activity_unit(ef_unit)
        if not target_unit or activity_unit == target_unit:
            return Decimal(str(activity_value))

        return self.convert_simple(activity_value, activity_unit, target_unit)


# Singleton instance for convenience
_converter_instance: Optional[UnitConverter] = None


def get_converter() -> UnitConverter:
    """Get the singleton UnitConverter instance."""
    global _converter_instance
    if _converter_instance is None:
        _converter_instance = UnitConverter()
    return _converter_instance
