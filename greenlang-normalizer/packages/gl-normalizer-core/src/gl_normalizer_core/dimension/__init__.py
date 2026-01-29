"""
Dimension Analysis module for the GreenLang Normalizer.

This module provides dimensional analysis capabilities to verify
unit compatibility before conversion operations.

Example:
    >>> from gl_normalizer_core.dimension import DimensionAnalyzer
    >>> analyzer = DimensionAnalyzer()
    >>> compat = analyzer.are_compatible("kilogram", "metric_ton")
    >>> print(compat)
    True
"""

from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
from dataclasses import dataclass
import hashlib

from pydantic import BaseModel, Field
import structlog

from gl_normalizer_core.errors import DimensionMismatchError

logger = structlog.get_logger(__name__)


class BaseDimension(str, Enum):
    """SI base dimensions."""

    MASS = "M"  # Mass [kg]
    LENGTH = "L"  # Length [m]
    TIME = "T"  # Time [s]
    TEMPERATURE = "Θ"  # Temperature [K]
    AMOUNT = "N"  # Amount of substance [mol]
    CURRENT = "I"  # Electric current [A]
    LUMINOSITY = "J"  # Luminous intensity [cd]
    DIMENSIONLESS = "1"  # Dimensionless quantity


@dataclass(frozen=True)
class Dimension:
    """
    Represents the dimension of a physical quantity.

    Dimensions are expressed as a combination of base SI dimensions.
    For example, velocity has dimension L/T (length per time).

    Attributes:
        mass: Exponent of mass dimension
        length: Exponent of length dimension
        time: Exponent of time dimension
        temperature: Exponent of temperature dimension
        amount: Exponent of amount dimension
        current: Exponent of current dimension
        luminosity: Exponent of luminosity dimension
    """

    mass: int = 0
    length: int = 0
    time: int = 0
    temperature: int = 0
    amount: int = 0
    current: int = 0
    luminosity: int = 0

    def __str__(self) -> str:
        """Return string representation of dimension."""
        parts = []
        if self.mass != 0:
            parts.append(f"M^{self.mass}" if self.mass != 1 else "M")
        if self.length != 0:
            parts.append(f"L^{self.length}" if self.length != 1 else "L")
        if self.time != 0:
            parts.append(f"T^{self.time}" if self.time != 1 else "T")
        if self.temperature != 0:
            parts.append(f"Θ^{self.temperature}" if self.temperature != 1 else "Θ")
        if self.amount != 0:
            parts.append(f"N^{self.amount}" if self.amount != 1 else "N")
        if self.current != 0:
            parts.append(f"I^{self.current}" if self.current != 1 else "I")
        if self.luminosity != 0:
            parts.append(f"J^{self.luminosity}" if self.luminosity != 1 else "J")

        return " ".join(parts) if parts else "1"

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

    def __pow__(self, power: int) -> "Dimension":
        """Raise dimension to a power."""
        return Dimension(
            mass=self.mass * power,
            length=self.length * power,
            time=self.time * power,
            temperature=self.temperature * power,
            amount=self.amount * power,
            current=self.current * power,
            luminosity=self.luminosity * power,
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


# Common dimension constants
DIMENSIONLESS = Dimension()
MASS = Dimension(mass=1)
LENGTH = Dimension(length=1)
TIME = Dimension(time=1)
TEMPERATURE = Dimension(temperature=1)
AREA = Dimension(length=2)
VOLUME = Dimension(length=3)
VELOCITY = Dimension(length=1, time=-1)
ACCELERATION = Dimension(length=1, time=-2)
FORCE = Dimension(mass=1, length=1, time=-2)
ENERGY = Dimension(mass=1, length=2, time=-2)
POWER = Dimension(mass=1, length=2, time=-3)
PRESSURE = Dimension(mass=1, length=-1, time=-2)


class DimensionInfo(BaseModel):
    """Information about a unit's dimension."""

    unit: str = Field(..., description="Unit string")
    dimension: str = Field(..., description="Dimension string")
    base_units: Dict[str, int] = Field(
        default_factory=dict,
        description="Base unit exponents"
    )
    is_dimensionless: bool = Field(default=False, description="Whether dimensionless")


class DimensionAnalyzer:
    """
    Analyzer for unit dimensions.

    This class provides dimensional analysis capabilities for verifying
    unit compatibility and deriving dimensions from unit strings.

    Attributes:
        unit_dimensions: Mapping of units to dimensions

    Example:
        >>> analyzer = DimensionAnalyzer()
        >>> dim = analyzer.get_dimension("kilogram")
        >>> print(dim)
        M
    """

    # Unit to dimension mappings
    UNIT_DIMENSIONS: Dict[str, Dimension] = {
        # Mass units
        "kilogram": MASS,
        "gram": MASS,
        "milligram": MASS,
        "metric_ton": MASS,
        "tonne": MASS,
        "pound": MASS,
        "ounce": MASS,
        # Length units
        "meter": LENGTH,
        "kilometer": LENGTH,
        "centimeter": LENGTH,
        "millimeter": LENGTH,
        "mile": LENGTH,
        "foot": LENGTH,
        "inch": LENGTH,
        # Time units
        "second": TIME,
        "minute": TIME,
        "hour": TIME,
        "day": TIME,
        "year": TIME,
        # Area units
        "square_meter": AREA,
        "hectare": AREA,
        "acre": AREA,
        # Volume units
        "cubic_meter": VOLUME,
        "liter": VOLUME,
        "milliliter": VOLUME,
        "gallon": VOLUME,
        "barrel": VOLUME,
        # Energy units
        "joule": ENERGY,
        "kilojoule": ENERGY,
        "megajoule": ENERGY,
        "gigajoule": ENERGY,
        "terajoule": ENERGY,
        "kilowatt_hour": ENERGY,
        "megawatt_hour": ENERGY,
        "gigawatt_hour": ENERGY,
        "british_thermal_unit": ENERGY,
        "therm": ENERGY,
        # Power units
        "watt": POWER,
        "kilowatt": POWER,
        "megawatt": POWER,
        "gigawatt": POWER,
        # Temperature units
        "kelvin": TEMPERATURE,
        "celsius": TEMPERATURE,
        "fahrenheit": TEMPERATURE,
        # Emissions units (treated as mass for conversion)
        "CO2_equivalent": MASS,
        "kg_CO2e": MASS,
        "t_CO2e": MASS,
        "kilogram_CO2_equivalent": MASS,
        "metric_ton_CO2_equivalent": MASS,
    }

    # Compatible dimension groups (can be converted between each other)
    DIMENSION_GROUPS: Dict[str, Set[str]] = {
        "mass": {"kilogram", "gram", "milligram", "metric_ton", "tonne", "pound", "ounce"},
        "length": {"meter", "kilometer", "centimeter", "millimeter", "mile", "foot", "inch"},
        "time": {"second", "minute", "hour", "day", "year"},
        "area": {"square_meter", "hectare", "acre"},
        "volume": {"cubic_meter", "liter", "milliliter", "gallon", "barrel"},
        "energy": {
            "joule", "kilojoule", "megajoule", "gigajoule", "terajoule",
            "kilowatt_hour", "megawatt_hour", "gigawatt_hour",
            "british_thermal_unit", "therm"
        },
        "power": {"watt", "kilowatt", "megawatt", "gigawatt"},
        "temperature": {"kelvin", "celsius", "fahrenheit"},
        "emissions": {
            "CO2_equivalent", "kg_CO2e", "t_CO2e",
            "kilogram_CO2_equivalent", "metric_ton_CO2_equivalent"
        },
    }

    def __init__(self) -> None:
        """Initialize DimensionAnalyzer."""
        self.unit_dimensions = dict(self.UNIT_DIMENSIONS)
        logger.info("DimensionAnalyzer initialized", unit_count=len(self.unit_dimensions))

    def get_dimension(self, unit: str) -> Dimension:
        """
        Get the dimension of a unit.

        Args:
            unit: Unit string

        Returns:
            Dimension of the unit

        Raises:
            DimensionMismatchError: If unit dimension is unknown
        """
        normalized = self._normalize_unit(unit)

        if normalized in self.unit_dimensions:
            return self.unit_dimensions[normalized]

        # Check for compound units with emissions suffix
        for suffix in ["_CO2e", "_CO2_equivalent"]:
            if normalized.endswith(suffix):
                base_unit = normalized[:-len(suffix)]
                if base_unit in self.unit_dimensions:
                    return self.unit_dimensions[base_unit]

        # Unknown dimension
        logger.warning("Unknown unit dimension", unit=unit)
        raise DimensionMismatchError(
            f"Unknown dimension for unit '{unit}'",
            hint=f"Add unit '{unit}' to the dimension registry or check spelling.",
        )

    def get_dimension_info(self, unit: str) -> DimensionInfo:
        """
        Get detailed dimension information for a unit.

        Args:
            unit: Unit string

        Returns:
            DimensionInfo with full details
        """
        try:
            dim = self.get_dimension(unit)
            return DimensionInfo(
                unit=unit,
                dimension=str(dim),
                base_units={
                    "mass": dim.mass,
                    "length": dim.length,
                    "time": dim.time,
                    "temperature": dim.temperature,
                },
                is_dimensionless=dim.is_dimensionless(),
            )
        except DimensionMismatchError:
            return DimensionInfo(
                unit=unit,
                dimension="unknown",
                is_dimensionless=False,
            )

    def are_compatible(self, unit1: str, unit2: str) -> bool:
        """
        Check if two units have compatible dimensions.

        Args:
            unit1: First unit
            unit2: Second unit

        Returns:
            True if units are dimensionally compatible
        """
        norm1 = self._normalize_unit(unit1)
        norm2 = self._normalize_unit(unit2)

        # Same unit is always compatible
        if norm1 == norm2:
            return True

        # Check dimension groups
        for group_units in self.DIMENSION_GROUPS.values():
            if norm1 in group_units and norm2 in group_units:
                return True

        # Check actual dimensions
        try:
            dim1 = self.get_dimension(unit1)
            dim2 = self.get_dimension(unit2)
            return dim1 == dim2
        except DimensionMismatchError:
            return False

    def get_compatible_units(self, unit: str) -> List[str]:
        """
        Get all units compatible with the given unit.

        Args:
            unit: Unit to find compatible units for

        Returns:
            List of compatible unit strings
        """
        normalized = self._normalize_unit(unit)
        compatible = set()

        # Check dimension groups
        for group_units in self.DIMENSION_GROUPS.values():
            if normalized in group_units:
                compatible.update(group_units)

        # Remove the input unit itself
        compatible.discard(normalized)

        return sorted(compatible)

    def validate_conversion(
        self,
        source_unit: str,
        target_unit: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that a conversion is dimensionally valid.

        Args:
            source_unit: Source unit
            target_unit: Target unit

        Returns:
            Tuple of (is_valid, error_message)
        """
        if self.are_compatible(source_unit, target_unit):
            return (True, None)

        try:
            source_dim = self.get_dimension(source_unit)
            target_dim = self.get_dimension(target_unit)
            return (
                False,
                f"Dimension mismatch: {source_unit} has dimension {source_dim}, "
                f"but {target_unit} has dimension {target_dim}",
            )
        except DimensionMismatchError as e:
            return (False, str(e))

    def add_unit_dimension(self, unit: str, dimension: Dimension) -> None:
        """
        Add a unit to dimension mapping.

        Args:
            unit: Unit string
            dimension: Dimension of the unit
        """
        self.unit_dimensions[unit] = dimension
        logger.debug("Added unit dimension", unit=unit, dimension=str(dimension))

    def _normalize_unit(self, unit: str) -> str:
        """Normalize a unit string for lookup."""
        return unit.strip().replace(" ", "_").replace("-", "_")


__all__ = [
    "DimensionAnalyzer",
    "Dimension",
    "DimensionInfo",
    "BaseDimension",
    "DIMENSIONLESS",
    "MASS",
    "LENGTH",
    "TIME",
    "TEMPERATURE",
    "AREA",
    "VOLUME",
    "VELOCITY",
    "ACCELERATION",
    "FORCE",
    "ENERGY",
    "POWER",
    "PRESSURE",
]
