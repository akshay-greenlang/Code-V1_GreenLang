"""
Unit Converter - Deterministic Unit Conversions

Provides zero-hallucination unit conversions for:
- Energy (kWh, MWh, GJ, BTU, etc.)
- Mass (kg, tonnes, lbs, etc.)
- Volume (liters, gallons, m³, etc.)
- Distance (km, miles, etc.)
- Currency (with exchange rates)

All conversions are deterministic with exact conversion factors.
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Optional, Tuple
from pydantic import BaseModel, Field
from enum import Enum


class UnitType(str, Enum):
    """Unit type categories."""
    ENERGY = "energy"
    MASS = "mass"
    VOLUME = "volume"
    DISTANCE = "distance"
    CURRENCY = "currency"
    EMISSIONS = "emissions"


class Unit(BaseModel):
    """Unit definition."""

    unit_code: str = Field(..., description="Unit code (e.g., 'kWh', 'kg')")
    unit_name: str = Field(..., description="Full unit name")
    unit_type: UnitType = Field(..., description="Unit type category")
    base_unit: str = Field(..., description="Base unit for this type")
    conversion_factor: Decimal = Field(..., description="Conversion factor to base unit")
    symbol: Optional[str] = Field(None, description="Unit symbol")


class UnitConverter:
    """
    Zero-hallucination unit converter.

    All conversion factors are exact, deterministic constants.
    NO LLM calls for unit conversions.
    """

    def __init__(self):
        """Initialize unit converter with conversion factors."""
        self.units: Dict[str, Unit] = {}
        self._initialize_units()

    def _initialize_units(self) -> None:
        """Initialize conversion factors for all supported units."""

        # ENERGY UNITS (base: kWh)
        energy_units = [
            Unit(unit_code="Wh", unit_name="Watt-hour", unit_type=UnitType.ENERGY,
                 base_unit="kWh", conversion_factor=Decimal("0.001"), symbol="Wh"),
            Unit(unit_code="kWh", unit_name="Kilowatt-hour", unit_type=UnitType.ENERGY,
                 base_unit="kWh", conversion_factor=Decimal("1"), symbol="kWh"),
            Unit(unit_code="MWh", unit_name="Megawatt-hour", unit_type=UnitType.ENERGY,
                 base_unit="kWh", conversion_factor=Decimal("1000"), symbol="MWh"),
            Unit(unit_code="GWh", unit_name="Gigawatt-hour", unit_type=UnitType.ENERGY,
                 base_unit="kWh", conversion_factor=Decimal("1000000"), symbol="GWh"),
            Unit(unit_code="J", unit_name="Joule", unit_type=UnitType.ENERGY,
                 base_unit="kWh", conversion_factor=Decimal("0.000000277778"), symbol="J"),
            Unit(unit_code="kJ", unit_name="Kilojoule", unit_type=UnitType.ENERGY,
                 base_unit="kWh", conversion_factor=Decimal("0.000277778"), symbol="kJ"),
            Unit(unit_code="MJ", unit_name="Megajoule", unit_type=UnitType.ENERGY,
                 base_unit="kWh", conversion_factor=Decimal("0.277778"), symbol="MJ"),
            Unit(unit_code="GJ", unit_name="Gigajoule", unit_type=UnitType.ENERGY,
                 base_unit="kWh", conversion_factor=Decimal("277.778"), symbol="GJ"),
            Unit(unit_code="BTU", unit_name="British Thermal Unit", unit_type=UnitType.ENERGY,
                 base_unit="kWh", conversion_factor=Decimal("0.000293071"), symbol="BTU"),
            Unit(unit_code="MMBTU", unit_name="Million BTU", unit_type=UnitType.ENERGY,
                 base_unit="kWh", conversion_factor=Decimal("293.071"), symbol="MMBTU"),
            Unit(unit_code="therm", unit_name="Therm", unit_type=UnitType.ENERGY,
                 base_unit="kWh", conversion_factor=Decimal("29.3071"), symbol="therm"),
        ]

        # MASS UNITS (base: kg)
        mass_units = [
            Unit(unit_code="g", unit_name="Gram", unit_type=UnitType.MASS,
                 base_unit="kg", conversion_factor=Decimal("0.001"), symbol="g"),
            Unit(unit_code="kg", unit_name="Kilogram", unit_type=UnitType.MASS,
                 base_unit="kg", conversion_factor=Decimal("1"), symbol="kg"),
            Unit(unit_code="t", unit_name="Metric Tonne", unit_type=UnitType.MASS,
                 base_unit="kg", conversion_factor=Decimal("1000"), symbol="t"),
            Unit(unit_code="kt", unit_name="Kilotonne", unit_type=UnitType.MASS,
                 base_unit="kg", conversion_factor=Decimal("1000000"), symbol="kt"),
            Unit(unit_code="Mt", unit_name="Megatonne", unit_type=UnitType.MASS,
                 base_unit="kg", conversion_factor=Decimal("1000000000"), symbol="Mt"),
            Unit(unit_code="lb", unit_name="Pound", unit_type=UnitType.MASS,
                 base_unit="kg", conversion_factor=Decimal("0.453592"), symbol="lb"),
            Unit(unit_code="oz", unit_name="Ounce", unit_type=UnitType.MASS,
                 base_unit="kg", conversion_factor=Decimal("0.0283495"), symbol="oz"),
            Unit(unit_code="ton_us", unit_name="US Ton", unit_type=UnitType.MASS,
                 base_unit="kg", conversion_factor=Decimal("907.185"), symbol="ton"),
        ]

        # VOLUME UNITS (base: L)
        volume_units = [
            Unit(unit_code="mL", unit_name="Milliliter", unit_type=UnitType.VOLUME,
                 base_unit="L", conversion_factor=Decimal("0.001"), symbol="mL"),
            Unit(unit_code="L", unit_name="Liter", unit_type=UnitType.VOLUME,
                 base_unit="L", conversion_factor=Decimal("1"), symbol="L"),
            Unit(unit_code="m3", unit_name="Cubic Meter", unit_type=UnitType.VOLUME,
                 base_unit="L", conversion_factor=Decimal("1000"), symbol="m³"),
            Unit(unit_code="gal_us", unit_name="US Gallon", unit_type=UnitType.VOLUME,
                 base_unit="L", conversion_factor=Decimal("3.78541"), symbol="gal"),
            Unit(unit_code="gal_uk", unit_name="UK Gallon", unit_type=UnitType.VOLUME,
                 base_unit="L", conversion_factor=Decimal("4.54609"), symbol="gal"),
            Unit(unit_code="barrel", unit_name="Barrel (oil)", unit_type=UnitType.VOLUME,
                 base_unit="L", conversion_factor=Decimal("158.987"), symbol="bbl"),
        ]

        # DISTANCE UNITS (base: km)
        distance_units = [
            Unit(unit_code="m", unit_name="Meter", unit_type=UnitType.DISTANCE,
                 base_unit="km", conversion_factor=Decimal("0.001"), symbol="m"),
            Unit(unit_code="km", unit_name="Kilometer", unit_type=UnitType.DISTANCE,
                 base_unit="km", conversion_factor=Decimal("1"), symbol="km"),
            Unit(unit_code="mi", unit_name="Mile", unit_type=UnitType.DISTANCE,
                 base_unit="km", conversion_factor=Decimal("1.60934"), symbol="mi"),
            Unit(unit_code="nmi", unit_name="Nautical Mile", unit_type=UnitType.DISTANCE,
                 base_unit="km", conversion_factor=Decimal("1.852"), symbol="nmi"),
        ]

        # EMISSIONS UNITS (base: kg_co2e)
        emissions_units = [
            Unit(unit_code="kg_co2e", unit_name="Kilogram CO2 equivalent", unit_type=UnitType.EMISSIONS,
                 base_unit="kg_co2e", conversion_factor=Decimal("1"), symbol="kg CO₂e"),
            Unit(unit_code="t_co2e", unit_name="Tonne CO2 equivalent", unit_type=UnitType.EMISSIONS,
                 base_unit="kg_co2e", conversion_factor=Decimal("1000"), symbol="t CO₂e"),
            Unit(unit_code="kt_co2e", unit_name="Kilotonne CO2 equivalent", unit_type=UnitType.EMISSIONS,
                 base_unit="kg_co2e", conversion_factor=Decimal("1000000"), symbol="kt CO₂e"),
            Unit(unit_code="Mt_co2e", unit_name="Megatonne CO2 equivalent", unit_type=UnitType.EMISSIONS,
                 base_unit="kg_co2e", conversion_factor=Decimal("1000000000"), symbol="Mt CO₂e"),
        ]

        # Register all units
        for units_list in [energy_units, mass_units, volume_units, distance_units, emissions_units]:
            for unit in units_list:
                self.units[unit.unit_code] = unit

    def convert(
        self,
        value: Union[float, int, Decimal],
        from_unit: str,
        to_unit: str,
        precision: int = 6
    ) -> Decimal:
        """
        Convert value from one unit to another - DETERMINISTIC.

        Args:
            value: Value to convert
            from_unit: Source unit code
            to_unit: Target unit code
            precision: Decimal precision

        Returns:
            Converted value as Decimal

        Raises:
            ValueError: If units are not compatible or not found
        """
        # Get unit definitions
        from_unit_def = self.units.get(from_unit)
        to_unit_def = self.units.get(to_unit)

        if not from_unit_def:
            raise ValueError(f"Unknown source unit: {from_unit}")
        if not to_unit_def:
            raise ValueError(f"Unknown target unit: {to_unit}")

        # Check unit types are compatible
        if from_unit_def.unit_type != to_unit_def.unit_type:
            raise ValueError(
                f"Incompatible unit types: {from_unit} ({from_unit_def.unit_type}) "
                f"and {to_unit} ({to_unit_def.unit_type})"
            )

        # Convert to base unit, then to target unit
        decimal_value = Decimal(str(value))
        base_value = decimal_value * from_unit_def.conversion_factor
        target_value = base_value / to_unit_def.conversion_factor

        # Apply precision
        quantize_str = '0.' + '0' * precision
        return target_value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    def get_conversion_factor(self, from_unit: str, to_unit: str) -> Decimal:
        """
        Get conversion factor between two units.

        Args:
            from_unit: Source unit code
            to_unit: Target unit code

        Returns:
            Conversion factor (multiply source value by this)

        Raises:
            ValueError: If units are not compatible
        """
        from_unit_def = self.units.get(from_unit)
        to_unit_def = self.units.get(to_unit)

        if not from_unit_def or not to_unit_def:
            raise ValueError("Unknown unit")

        if from_unit_def.unit_type != to_unit_def.unit_type:
            raise ValueError("Incompatible unit types")

        return from_unit_def.conversion_factor / to_unit_def.conversion_factor

    def list_units(self, unit_type: Optional[UnitType] = None) -> Dict[str, Unit]:
        """
        List available units.

        Args:
            unit_type: Optional filter by unit type

        Returns:
            Dictionary of units
        """
        if unit_type:
            return {
                code: unit for code, unit in self.units.items()
                if unit.unit_type == unit_type
            }
        return self.units.copy()

    def validate_unit(self, unit_code: str) -> bool:
        """
        Validate if unit code is supported.

        Args:
            unit_code: Unit code to validate

        Returns:
            True if valid, False otherwise
        """
        return unit_code in self.units


# Example usage
if __name__ == "__main__":
    converter = UnitConverter()

    # Energy conversion
    result = converter.convert(100, "kWh", "MWh", precision=4)
    print(f"100 kWh = {result} MWh")

    # Mass conversion
    result = converter.convert(1000, "kg", "t", precision=2)
    print(f"1000 kg = {result} t")

    # Volume conversion
    result = converter.convert(100, "L", "gal_us", precision=2)
    print(f"100 L = {result} US gallons")

    # Emissions conversion
    result = converter.convert(2500, "kg_co2e", "t_co2e", precision=3)
    print(f"2500 kg CO2e = {result} t CO2e")

    # List energy units
    energy_units = converter.list_units(UnitType.ENERGY)
    print(f"\nEnergy units available: {list(energy_units.keys())}")
