"""
QUDT Ontology Unit Conversion Service

This module implements unit conversions based on the QUDT (Quantities, Units,
Dimensions, and Types) ontology. QUDT provides a standardized vocabulary for
quantities and units of measure.

Features:
- QUDT-compliant unit definitions
- Automatic unit conversion with dimensional analysis
- Support for emission-related units (mass, energy, volume, area)
- SI and Imperial unit systems
- Compound unit support (e.g., kg CO2e/kWh)
- Precision handling with Decimal arithmetic

Usage:
    from services.qudt_units import UnitConverter, get_unit_converter

    converter = get_unit_converter()

    # Simple conversion
    result = converter.convert(100, "kWh", "MJ")  # 360 MJ

    # With uncertainty
    result = converter.convert_with_uncertainty(100, "kWh", "MJ", uncertainty=0.05)

    # Check compatibility
    compatible = converter.are_compatible("kg", "lb")  # True
"""
import logging
import re
import threading
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Unit System Definitions
# =============================================================================


class UnitSystem(Enum):
    """Unit system identifiers."""

    SI = "SI"
    IMPERIAL = "Imperial"
    US = "US"
    CGS = "CGS"


class QuantityKind(Enum):
    """QUDT Quantity Kinds relevant to emissions calculations."""

    # Basic quantities
    MASS = "Mass"
    LENGTH = "Length"
    TIME = "Time"
    AREA = "Area"
    VOLUME = "Volume"
    TEMPERATURE = "Temperature"

    # Energy quantities
    ENERGY = "Energy"
    POWER = "Power"

    # Emission-specific
    MASS_PER_ENERGY = "MassPerEnergy"
    MASS_PER_VOLUME = "MassPerVolume"
    MASS_PER_MASS = "MassPerMass"
    MASS_PER_AREA = "MassPerArea"
    MASS_PER_DISTANCE = "MassPerDistance"
    ENERGY_PER_MASS = "EnergyPerMass"
    ENERGY_PER_VOLUME = "EnergyPerVolume"

    # Other
    DIMENSIONLESS = "Dimensionless"
    CURRENCY = "Currency"
    CURRENCY_PER_MASS = "CurrencyPerMass"
    CURRENCY_PER_ENERGY = "CurrencyPerEnergy"


@dataclass
class Dimension:
    """Represents SI base dimension exponents."""

    mass: int = 0  # M (kilogram)
    length: int = 0  # L (meter)
    time: int = 0  # T (second)
    temperature: int = 0  # Θ (kelvin)
    amount: int = 0  # N (mole)
    current: int = 0  # I (ampere)
    luminosity: int = 0  # J (candela)

    def __eq__(self, other: "Dimension") -> bool:
        if not isinstance(other, Dimension):
            return False
        return (
            self.mass == other.mass
            and self.length == other.length
            and self.time == other.time
            and self.temperature == other.temperature
            and self.amount == other.amount
            and self.current == other.current
            and self.luminosity == other.luminosity
        )

    def __mul__(self, other: "Dimension") -> "Dimension":
        """Multiply dimensions (add exponents)."""
        return Dimension(
            mass=self.mass + other.mass,
            length=self.length + other.length,
            time=self.time + other.time,
            temperature=self.temperature + other.temperature,
            amount=self.amount + other.amount,
            current=self.current + other.current,
            luminosity=self.luminosity + other.luminosity,
        )

    def __truediv__(self, other: "Dimension") -> "Dimension":
        """Divide dimensions (subtract exponents)."""
        return Dimension(
            mass=self.mass - other.mass,
            length=self.length - other.length,
            time=self.time - other.time,
            temperature=self.temperature - other.temperature,
            amount=self.amount - other.amount,
            current=self.current - other.current,
            luminosity=self.luminosity - other.luminosity,
        )

    def is_dimensionless(self) -> bool:
        """Check if dimension is dimensionless."""
        return (
            self.mass == 0
            and self.length == 0
            and self.time == 0
            and self.temperature == 0
            and self.amount == 0
            and self.current == 0
            and self.luminosity == 0
        )


@dataclass
class Unit:
    """Represents a unit of measure."""

    symbol: str
    name: str
    quantity_kind: QuantityKind
    dimension: Dimension
    conversion_factor: Decimal  # Factor to convert to SI base unit
    conversion_offset: Decimal = Decimal("0")  # Offset for temperature conversions
    system: UnitSystem = UnitSystem.SI
    qudt_uri: Optional[str] = None
    aliases: Set[str] = field(default_factory=set)

    def __post_init__(self):
        # Ensure conversion factor is Decimal
        if not isinstance(self.conversion_factor, Decimal):
            self.conversion_factor = Decimal(str(self.conversion_factor))
        if not isinstance(self.conversion_offset, Decimal):
            self.conversion_offset = Decimal(str(self.conversion_offset))


@dataclass
class ConversionResult:
    """Result of a unit conversion."""

    value: Decimal
    from_unit: str
    to_unit: str
    from_value: Decimal
    conversion_factor: Decimal
    uncertainty: Optional[Decimal] = None
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# Standard Unit Definitions
# =============================================================================


# Dimension constants
DIM_MASS = Dimension(mass=1)
DIM_LENGTH = Dimension(length=1)
DIM_TIME = Dimension(time=1)
DIM_AREA = Dimension(length=2)
DIM_VOLUME = Dimension(length=3)
DIM_ENERGY = Dimension(mass=1, length=2, time=-2)
DIM_POWER = Dimension(mass=1, length=2, time=-3)
DIM_TEMPERATURE = Dimension(temperature=1)
DIM_DIMENSIONLESS = Dimension()


# Standard units database
STANDARD_UNITS: Dict[str, Unit] = {}


def _register_unit(unit: Unit) -> None:
    """Register a unit in the standard database."""
    STANDARD_UNITS[unit.symbol.lower()] = unit
    for alias in unit.aliases:
        STANDARD_UNITS[alias.lower()] = unit


def _init_standard_units() -> None:
    """Initialize standard unit definitions."""

    # Mass units
    _register_unit(Unit(
        symbol="kg",
        name="kilogram",
        quantity_kind=QuantityKind.MASS,
        dimension=DIM_MASS,
        conversion_factor=Decimal("1"),
        qudt_uri="http://qudt.org/vocab/unit/KiloGM",
        aliases={"kilogram", "kilograms"},
    ))
    _register_unit(Unit(
        symbol="g",
        name="gram",
        quantity_kind=QuantityKind.MASS,
        dimension=DIM_MASS,
        conversion_factor=Decimal("0.001"),
        qudt_uri="http://qudt.org/vocab/unit/GM",
        aliases={"gram", "grams"},
    ))
    _register_unit(Unit(
        symbol="mg",
        name="milligram",
        quantity_kind=QuantityKind.MASS,
        dimension=DIM_MASS,
        conversion_factor=Decimal("0.000001"),
        aliases={"milligram", "milligrams"},
    ))
    _register_unit(Unit(
        symbol="t",
        name="metric ton",
        quantity_kind=QuantityKind.MASS,
        dimension=DIM_MASS,
        conversion_factor=Decimal("1000"),
        qudt_uri="http://qudt.org/vocab/unit/TON_M",
        aliases={"tonne", "metric_ton", "mt"},
    ))
    _register_unit(Unit(
        symbol="kt",
        name="kiloton",
        quantity_kind=QuantityKind.MASS,
        dimension=DIM_MASS,
        conversion_factor=Decimal("1000000"),
        aliases={"kiloton", "kilotons"},
    ))
    _register_unit(Unit(
        symbol="lb",
        name="pound",
        quantity_kind=QuantityKind.MASS,
        dimension=DIM_MASS,
        conversion_factor=Decimal("0.45359237"),
        system=UnitSystem.IMPERIAL,
        qudt_uri="http://qudt.org/vocab/unit/LB",
        aliases={"pound", "pounds", "lbs"},
    ))
    _register_unit(Unit(
        symbol="oz",
        name="ounce",
        quantity_kind=QuantityKind.MASS,
        dimension=DIM_MASS,
        conversion_factor=Decimal("0.028349523125"),
        system=UnitSystem.IMPERIAL,
        aliases={"ounce", "ounces"},
    ))
    _register_unit(Unit(
        symbol="ton_us",
        name="US short ton",
        quantity_kind=QuantityKind.MASS,
        dimension=DIM_MASS,
        conversion_factor=Decimal("907.18474"),
        system=UnitSystem.US,
        aliases={"short_ton", "us_ton"},
    ))
    _register_unit(Unit(
        symbol="ton_uk",
        name="UK long ton",
        quantity_kind=QuantityKind.MASS,
        dimension=DIM_MASS,
        conversion_factor=Decimal("1016.0469088"),
        system=UnitSystem.IMPERIAL,
        aliases={"long_ton", "uk_ton"},
    ))

    # Energy units
    _register_unit(Unit(
        symbol="J",
        name="joule",
        quantity_kind=QuantityKind.ENERGY,
        dimension=DIM_ENERGY,
        conversion_factor=Decimal("1"),
        qudt_uri="http://qudt.org/vocab/unit/J",
        aliases={"joule", "joules"},
    ))
    _register_unit(Unit(
        symbol="kJ",
        name="kilojoule",
        quantity_kind=QuantityKind.ENERGY,
        dimension=DIM_ENERGY,
        conversion_factor=Decimal("1000"),
        qudt_uri="http://qudt.org/vocab/unit/KiloJ",
        aliases={"kilojoule", "kilojoules"},
    ))
    _register_unit(Unit(
        symbol="MJ",
        name="megajoule",
        quantity_kind=QuantityKind.ENERGY,
        dimension=DIM_ENERGY,
        conversion_factor=Decimal("1000000"),
        qudt_uri="http://qudt.org/vocab/unit/MegaJ",
        aliases={"megajoule", "megajoules"},
    ))
    _register_unit(Unit(
        symbol="GJ",
        name="gigajoule",
        quantity_kind=QuantityKind.ENERGY,
        dimension=DIM_ENERGY,
        conversion_factor=Decimal("1000000000"),
        qudt_uri="http://qudt.org/vocab/unit/GigaJ",
        aliases={"gigajoule", "gigajoules"},
    ))
    _register_unit(Unit(
        symbol="TJ",
        name="terajoule",
        quantity_kind=QuantityKind.ENERGY,
        dimension=DIM_ENERGY,
        conversion_factor=Decimal("1000000000000"),
        aliases={"terajoule", "terajoules"},
    ))
    _register_unit(Unit(
        symbol="Wh",
        name="watt-hour",
        quantity_kind=QuantityKind.ENERGY,
        dimension=DIM_ENERGY,
        conversion_factor=Decimal("3600"),
        qudt_uri="http://qudt.org/vocab/unit/W-HR",
        aliases={"watt_hour", "watt-hour"},
    ))
    _register_unit(Unit(
        symbol="kWh",
        name="kilowatt-hour",
        quantity_kind=QuantityKind.ENERGY,
        dimension=DIM_ENERGY,
        conversion_factor=Decimal("3600000"),
        qudt_uri="http://qudt.org/vocab/unit/KiloW-HR",
        aliases={"kilowatt_hour", "kilowatt-hour", "kwh"},
    ))
    _register_unit(Unit(
        symbol="MWh",
        name="megawatt-hour",
        quantity_kind=QuantityKind.ENERGY,
        dimension=DIM_ENERGY,
        conversion_factor=Decimal("3600000000"),
        qudt_uri="http://qudt.org/vocab/unit/MegaW-HR",
        aliases={"megawatt_hour", "megawatt-hour", "mwh"},
    ))
    _register_unit(Unit(
        symbol="GWh",
        name="gigawatt-hour",
        quantity_kind=QuantityKind.ENERGY,
        dimension=DIM_ENERGY,
        conversion_factor=Decimal("3600000000000"),
        aliases={"gigawatt_hour", "gigawatt-hour", "gwh"},
    ))
    _register_unit(Unit(
        symbol="BTU",
        name="British thermal unit",
        quantity_kind=QuantityKind.ENERGY,
        dimension=DIM_ENERGY,
        conversion_factor=Decimal("1055.06"),
        system=UnitSystem.IMPERIAL,
        qudt_uri="http://qudt.org/vocab/unit/BTU_IT",
        aliases={"btu", "british_thermal_unit"},
    ))
    _register_unit(Unit(
        symbol="MMBTU",
        name="million BTU",
        quantity_kind=QuantityKind.ENERGY,
        dimension=DIM_ENERGY,
        conversion_factor=Decimal("1055060000"),
        system=UnitSystem.IMPERIAL,
        aliases={"mmbtu", "million_btu", "mmBtu"},
    ))
    _register_unit(Unit(
        symbol="therm",
        name="therm",
        quantity_kind=QuantityKind.ENERGY,
        dimension=DIM_ENERGY,
        conversion_factor=Decimal("105506000"),
        system=UnitSystem.US,
        aliases={"therms"},
    ))
    _register_unit(Unit(
        symbol="cal",
        name="calorie",
        quantity_kind=QuantityKind.ENERGY,
        dimension=DIM_ENERGY,
        conversion_factor=Decimal("4.184"),
        aliases={"calorie", "calories"},
    ))
    _register_unit(Unit(
        symbol="kcal",
        name="kilocalorie",
        quantity_kind=QuantityKind.ENERGY,
        dimension=DIM_ENERGY,
        conversion_factor=Decimal("4184"),
        aliases={"kilocalorie", "kilocalories"},
    ))
    _register_unit(Unit(
        symbol="toe",
        name="tonne of oil equivalent",
        quantity_kind=QuantityKind.ENERGY,
        dimension=DIM_ENERGY,
        conversion_factor=Decimal("41868000000"),
        aliases={"tonne_oil_equivalent"},
    ))
    _register_unit(Unit(
        symbol="tce",
        name="tonne of coal equivalent",
        quantity_kind=QuantityKind.ENERGY,
        dimension=DIM_ENERGY,
        conversion_factor=Decimal("29307600000"),
        aliases={"tonne_coal_equivalent"},
    ))

    # Volume units
    _register_unit(Unit(
        symbol="m3",
        name="cubic meter",
        quantity_kind=QuantityKind.VOLUME,
        dimension=DIM_VOLUME,
        conversion_factor=Decimal("1"),
        qudt_uri="http://qudt.org/vocab/unit/M3",
        aliases={"cubic_meter", "m³"},
    ))
    _register_unit(Unit(
        symbol="L",
        name="liter",
        quantity_kind=QuantityKind.VOLUME,
        dimension=DIM_VOLUME,
        conversion_factor=Decimal("0.001"),
        qudt_uri="http://qudt.org/vocab/unit/L",
        aliases={"liter", "liters", "litre", "litres", "l"},
    ))
    _register_unit(Unit(
        symbol="mL",
        name="milliliter",
        quantity_kind=QuantityKind.VOLUME,
        dimension=DIM_VOLUME,
        conversion_factor=Decimal("0.000001"),
        aliases={"milliliter", "milliliters", "ml"},
    ))
    _register_unit(Unit(
        symbol="gal",
        name="US gallon",
        quantity_kind=QuantityKind.VOLUME,
        dimension=DIM_VOLUME,
        conversion_factor=Decimal("0.003785411784"),
        system=UnitSystem.US,
        qudt_uri="http://qudt.org/vocab/unit/GAL_US",
        aliases={"gallon", "gallons", "us_gallon"},
    ))
    _register_unit(Unit(
        symbol="gal_uk",
        name="UK gallon",
        quantity_kind=QuantityKind.VOLUME,
        dimension=DIM_VOLUME,
        conversion_factor=Decimal("0.00454609"),
        system=UnitSystem.IMPERIAL,
        aliases={"imperial_gallon", "uk_gallon"},
    ))
    _register_unit(Unit(
        symbol="bbl",
        name="barrel (oil)",
        quantity_kind=QuantityKind.VOLUME,
        dimension=DIM_VOLUME,
        conversion_factor=Decimal("0.158987294928"),
        system=UnitSystem.US,
        aliases={"barrel", "barrels", "oil_barrel"},
    ))
    _register_unit(Unit(
        symbol="scf",
        name="standard cubic foot",
        quantity_kind=QuantityKind.VOLUME,
        dimension=DIM_VOLUME,
        conversion_factor=Decimal("0.028316846592"),
        system=UnitSystem.US,
        aliases={"standard_cubic_foot", "ft3"},
    ))
    _register_unit(Unit(
        symbol="mcf",
        name="thousand cubic feet",
        quantity_kind=QuantityKind.VOLUME,
        dimension=DIM_VOLUME,
        conversion_factor=Decimal("28.316846592"),
        system=UnitSystem.US,
        aliases={"thousand_cubic_feet", "kcf"},
    ))
    _register_unit(Unit(
        symbol="mmcf",
        name="million cubic feet",
        quantity_kind=QuantityKind.VOLUME,
        dimension=DIM_VOLUME,
        conversion_factor=Decimal("28316.846592"),
        system=UnitSystem.US,
        aliases={"million_cubic_feet"},
    ))

    # Length units
    _register_unit(Unit(
        symbol="m",
        name="meter",
        quantity_kind=QuantityKind.LENGTH,
        dimension=DIM_LENGTH,
        conversion_factor=Decimal("1"),
        qudt_uri="http://qudt.org/vocab/unit/M",
        aliases={"meter", "meters", "metre", "metres"},
    ))
    _register_unit(Unit(
        symbol="km",
        name="kilometer",
        quantity_kind=QuantityKind.LENGTH,
        dimension=DIM_LENGTH,
        conversion_factor=Decimal("1000"),
        qudt_uri="http://qudt.org/vocab/unit/KiloM",
        aliases={"kilometer", "kilometers", "kilometre"},
    ))
    _register_unit(Unit(
        symbol="mi",
        name="mile",
        quantity_kind=QuantityKind.LENGTH,
        dimension=DIM_LENGTH,
        conversion_factor=Decimal("1609.344"),
        system=UnitSystem.IMPERIAL,
        qudt_uri="http://qudt.org/vocab/unit/MI",
        aliases={"mile", "miles"},
    ))
    _register_unit(Unit(
        symbol="nmi",
        name="nautical mile",
        quantity_kind=QuantityKind.LENGTH,
        dimension=DIM_LENGTH,
        conversion_factor=Decimal("1852"),
        aliases={"nautical_mile", "nautical_miles"},
    ))

    # Area units
    _register_unit(Unit(
        symbol="m2",
        name="square meter",
        quantity_kind=QuantityKind.AREA,
        dimension=DIM_AREA,
        conversion_factor=Decimal("1"),
        qudt_uri="http://qudt.org/vocab/unit/M2",
        aliases={"square_meter", "m²", "sq_m"},
    ))
    _register_unit(Unit(
        symbol="ha",
        name="hectare",
        quantity_kind=QuantityKind.AREA,
        dimension=DIM_AREA,
        conversion_factor=Decimal("10000"),
        qudt_uri="http://qudt.org/vocab/unit/HA",
        aliases={"hectare", "hectares"},
    ))
    _register_unit(Unit(
        symbol="km2",
        name="square kilometer",
        quantity_kind=QuantityKind.AREA,
        dimension=DIM_AREA,
        conversion_factor=Decimal("1000000"),
        aliases={"square_kilometer", "km²", "sq_km"},
    ))
    _register_unit(Unit(
        symbol="acre",
        name="acre",
        quantity_kind=QuantityKind.AREA,
        dimension=DIM_AREA,
        conversion_factor=Decimal("4046.8564224"),
        system=UnitSystem.IMPERIAL,
        aliases={"acres"},
    ))

    # Power units
    _register_unit(Unit(
        symbol="W",
        name="watt",
        quantity_kind=QuantityKind.POWER,
        dimension=DIM_POWER,
        conversion_factor=Decimal("1"),
        qudt_uri="http://qudt.org/vocab/unit/W",
        aliases={"watt", "watts"},
    ))
    _register_unit(Unit(
        symbol="kW",
        name="kilowatt",
        quantity_kind=QuantityKind.POWER,
        dimension=DIM_POWER,
        conversion_factor=Decimal("1000"),
        qudt_uri="http://qudt.org/vocab/unit/KiloW",
        aliases={"kilowatt", "kilowatts"},
    ))
    _register_unit(Unit(
        symbol="MW",
        name="megawatt",
        quantity_kind=QuantityKind.POWER,
        dimension=DIM_POWER,
        conversion_factor=Decimal("1000000"),
        qudt_uri="http://qudt.org/vocab/unit/MegaW",
        aliases={"megawatt", "megawatts"},
    ))
    _register_unit(Unit(
        symbol="GW",
        name="gigawatt",
        quantity_kind=QuantityKind.POWER,
        dimension=DIM_POWER,
        conversion_factor=Decimal("1000000000"),
        aliases={"gigawatt", "gigawatts"},
    ))
    _register_unit(Unit(
        symbol="hp",
        name="horsepower",
        quantity_kind=QuantityKind.POWER,
        dimension=DIM_POWER,
        conversion_factor=Decimal("745.7"),
        system=UnitSystem.IMPERIAL,
        aliases={"horsepower"},
    ))

    # CO2 equivalent units (special handling)
    _register_unit(Unit(
        symbol="tCO2e",
        name="tonne CO2 equivalent",
        quantity_kind=QuantityKind.MASS,
        dimension=DIM_MASS,
        conversion_factor=Decimal("1000"),
        aliases={"tco2e", "t_co2e", "tonne_co2e", "tonnes_co2e"},
    ))
    _register_unit(Unit(
        symbol="kgCO2e",
        name="kilogram CO2 equivalent",
        quantity_kind=QuantityKind.MASS,
        dimension=DIM_MASS,
        conversion_factor=Decimal("1"),
        aliases={"kgco2e", "kg_co2e"},
    ))
    _register_unit(Unit(
        symbol="mtCO2e",
        name="million tonnes CO2 equivalent",
        quantity_kind=QuantityKind.MASS,
        dimension=DIM_MASS,
        conversion_factor=Decimal("1000000000"),
        aliases={"mtco2e", "mt_co2e"},
    ))


# Initialize units on module load
_init_standard_units()


# =============================================================================
# Unit Converter
# =============================================================================


class UnitConverter:
    """
    QUDT-compliant unit conversion service.

    Provides dimensional analysis and conversion between compatible units.
    """

    def __init__(self, precision: int = 10):
        """
        Initialize the unit converter.

        Args:
            precision: Decimal precision for calculations
        """
        self.precision = precision
        self._custom_units: Dict[str, Unit] = {}
        self._lock = threading.RLock()

    def get_unit(self, symbol: str) -> Optional[Unit]:
        """Get unit by symbol or alias."""
        symbol_lower = symbol.lower()

        # Check custom units first
        with self._lock:
            if symbol_lower in self._custom_units:
                return self._custom_units[symbol_lower]

        # Check standard units
        return STANDARD_UNITS.get(symbol_lower)

    def register_unit(self, unit: Unit) -> None:
        """Register a custom unit."""
        with self._lock:
            self._custom_units[unit.symbol.lower()] = unit
            for alias in unit.aliases:
                self._custom_units[alias.lower()] = unit

    def are_compatible(self, from_unit: str, to_unit: str) -> bool:
        """Check if two units are dimensionally compatible."""
        from_u = self.get_unit(from_unit)
        to_u = self.get_unit(to_unit)

        if from_u is None or to_u is None:
            return False

        return from_u.dimension == to_u.dimension

    def convert(
        self,
        value: Union[float, Decimal, int],
        from_unit: str,
        to_unit: str,
    ) -> Decimal:
        """
        Convert a value from one unit to another.

        Args:
            value: The value to convert
            from_unit: Source unit symbol
            to_unit: Target unit symbol

        Returns:
            Converted value as Decimal

        Raises:
            ValueError: If units are incompatible or not found
        """
        result = self.convert_with_details(value, from_unit, to_unit)
        return result.value

    def convert_with_details(
        self,
        value: Union[float, Decimal, int],
        from_unit: str,
        to_unit: str,
    ) -> ConversionResult:
        """
        Convert with full details.

        Args:
            value: The value to convert
            from_unit: Source unit symbol
            to_unit: Target unit symbol

        Returns:
            ConversionResult with full details
        """
        # Get units
        from_u = self.get_unit(from_unit)
        to_u = self.get_unit(to_unit)

        if from_u is None:
            raise ValueError(f"Unknown unit: {from_unit}")
        if to_u is None:
            raise ValueError(f"Unknown unit: {to_unit}")

        # Check compatibility
        if from_u.dimension != to_u.dimension:
            raise ValueError(
                f"Incompatible units: {from_unit} ({from_u.quantity_kind.value}) "
                f"and {to_unit} ({to_u.quantity_kind.value})"
            )

        # Convert to Decimal
        if not isinstance(value, Decimal):
            value = Decimal(str(value))

        # Calculate conversion
        # value * from_factor / to_factor
        from_value = value

        if from_u.conversion_offset != 0 or to_u.conversion_offset != 0:
            # Temperature conversion with offset
            si_value = (value + from_u.conversion_offset) * from_u.conversion_factor
            result_value = (si_value / to_u.conversion_factor) - to_u.conversion_offset
        else:
            # Standard conversion
            conversion_factor = from_u.conversion_factor / to_u.conversion_factor
            result_value = value * conversion_factor

        # Round to precision
        result_value = result_value.quantize(
            Decimal(10) ** -self.precision, rounding=ROUND_HALF_UP
        )

        return ConversionResult(
            value=result_value,
            from_unit=from_unit,
            to_unit=to_unit,
            from_value=from_value,
            conversion_factor=from_u.conversion_factor / to_u.conversion_factor,
        )

    def convert_with_uncertainty(
        self,
        value: Union[float, Decimal],
        from_unit: str,
        to_unit: str,
        uncertainty: Union[float, Decimal],
    ) -> ConversionResult:
        """
        Convert with uncertainty propagation.

        Args:
            value: The value to convert
            from_unit: Source unit symbol
            to_unit: Target unit symbol
            uncertainty: Relative uncertainty (e.g., 0.05 for 5%)

        Returns:
            ConversionResult with uncertainty
        """
        result = self.convert_with_details(value, from_unit, to_unit)

        if not isinstance(uncertainty, Decimal):
            uncertainty = Decimal(str(uncertainty))

        result.uncertainty = result.value * uncertainty
        return result

    def list_units(
        self, quantity_kind: Optional[QuantityKind] = None
    ) -> List[Unit]:
        """List available units, optionally filtered by quantity kind."""
        units = list(STANDARD_UNITS.values())

        with self._lock:
            units.extend(self._custom_units.values())

        # Remove duplicates (same unit referenced by multiple aliases)
        seen = set()
        unique_units = []
        for unit in units:
            if unit.symbol not in seen:
                seen.add(unit.symbol)
                unique_units.append(unit)

        if quantity_kind:
            unique_units = [u for u in unique_units if u.quantity_kind == quantity_kind]

        return sorted(unique_units, key=lambda u: u.symbol)

    def get_conversion_factor(self, from_unit: str, to_unit: str) -> Decimal:
        """Get the conversion factor between two units."""
        from_u = self.get_unit(from_unit)
        to_u = self.get_unit(to_unit)

        if from_u is None or to_u is None:
            raise ValueError(f"Unknown unit: {from_unit} or {to_unit}")

        if from_u.dimension != to_u.dimension:
            raise ValueError(f"Incompatible units: {from_unit} and {to_unit}")

        return from_u.conversion_factor / to_u.conversion_factor


# =============================================================================
# Singleton Instance
# =============================================================================

_converter: Optional[UnitConverter] = None
_converter_lock = threading.Lock()


def get_unit_converter() -> UnitConverter:
    """Get the singleton unit converter instance."""
    global _converter
    if _converter is None:
        with _converter_lock:
            if _converter is None:
                _converter = UnitConverter()
    return _converter


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "UnitConverter",
    "Unit",
    "Dimension",
    "QuantityKind",
    "UnitSystem",
    "ConversionResult",
    "get_unit_converter",
    "STANDARD_UNITS",
]
