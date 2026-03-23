# -*- coding: utf-8 -*-
"""
Unit Catalog for GL-FOUND-X-002.

This module provides a comprehensive catalog of physical units and their
dimensional relationships. It supports unit registration, alias resolution,
compatibility checking, and unit conversion.

Key Features:
    - Unit registration with dimension tracking
    - Unit alias support (e.g., "kWh" and "kilowatt-hour")
    - Dimensional compatibility checking
    - Unit conversion using SI factors
    - Pre-loaded SI and common derived units

Design Principles:
    - Zero-hallucination: All conversions are deterministic mathematical operations
    - Complete provenance: All conversions can be traced back to SI factors
    - Fail loudly: Unknown units raise errors rather than guessing

Example:
    >>> catalog = UnitCatalog()
    >>> catalog.is_compatible("kWh", "MWh")
    True
    >>> catalog.convert(100, "kWh", "MWh")
    0.1
    >>> catalog.get_dimension("kg")
    'mass'

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator - Task 2.3
"""

from __future__ import annotations

import logging
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, FrozenSet, List, Optional, Set

from pydantic import BaseModel, ConfigDict, Field


logger = logging.getLogger(__name__)


# =============================================================================
# UNIT DEFINITION MODELS
# =============================================================================


class UnitDefinition(BaseModel):
    """
    Definition of a physical unit.

    Represents a single unit with its dimensional classification and
    conversion factor to the SI base unit.

    Attributes:
        symbol: Short symbol for the unit (e.g., "kWh", "kg")
        name: Full name of the unit (e.g., "kilowatt-hour", "kilogram")
        dimension: Physical dimension (e.g., "energy", "mass", "volume")
        si_factor: Conversion factor to SI base unit (multiply to convert)
        si_offset: Offset for temperature conversions (add after multiplying)
        is_si_base: Whether this is the SI base unit for its dimension
        is_canonical: Whether this is the canonical unit for normalization

    Example:
        >>> unit = UnitDefinition(
        ...     symbol="kWh",
        ...     name="kilowatt-hour",
        ...     dimension="energy",
        ...     si_factor=3600000.0,  # 1 kWh = 3,600,000 J
        ...     is_canonical=True
        ... )
    """

    symbol: str = Field(
        ...,
        min_length=1,
        max_length=32,
        description="Short symbol for the unit"
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Full name of the unit"
    )
    dimension: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Physical dimension (e.g., 'energy', 'mass')"
    )
    si_factor: float = Field(
        ...,
        gt=0,
        description="Conversion factor to SI base unit"
    )
    si_offset: float = Field(
        default=0.0,
        description="Offset for temperature conversions"
    )
    is_si_base: bool = Field(
        default=False,
        description="Whether this is the SI base unit"
    )
    is_canonical: bool = Field(
        default=False,
        description="Whether this is the canonical unit for normalization"
    )

    model_config = ConfigDict(frozen=True, extra="forbid")

    def to_si(self, value: float) -> float:
        """
        Convert a value from this unit to SI base unit.

        Args:
            value: Value in this unit

        Returns:
            Value in SI base unit
        """
        return (value * self.si_factor) + self.si_offset

    def from_si(self, value: float) -> float:
        """
        Convert a value from SI base unit to this unit.

        Args:
            value: Value in SI base unit

        Returns:
            Value in this unit
        """
        return (value - self.si_offset) / self.si_factor


class DimensionDefinition(BaseModel):
    """
    Definition of a physical dimension.

    Represents a physical dimension (e.g., energy, mass) with its
    SI base unit and list of common units.

    Attributes:
        name: Name of the dimension (e.g., "energy", "mass")
        si_unit: Symbol of the SI base unit (e.g., "J", "kg")
        canonical_unit: Symbol of the canonical unit for normalization
        common_units: List of common unit symbols
        description: Human-readable description

    Example:
        >>> dimension = DimensionDefinition(
        ...     name="energy",
        ...     si_unit="J",
        ...     canonical_unit="kWh",
        ...     common_units=["kWh", "MWh", "GWh", "J", "kJ", "MJ", "GJ"]
        ... )
    """

    name: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Name of the dimension"
    )
    si_unit: str = Field(
        ...,
        min_length=1,
        max_length=32,
        description="Symbol of the SI base unit"
    )
    canonical_unit: str = Field(
        ...,
        min_length=1,
        max_length=32,
        description="Symbol of the canonical unit for normalization"
    )
    common_units: List[str] = Field(
        default_factory=list,
        description="List of common unit symbols"
    )
    description: Optional[str] = Field(
        default=None,
        max_length=512,
        description="Human-readable description"
    )

    model_config = ConfigDict(frozen=True, extra="forbid")


# =============================================================================
# UNIT CATALOG
# =============================================================================


class UnitCatalog:
    """
    Catalog of physical units and dimensions.

    Provides a centralized registry for unit definitions with support for:
    - Unit registration and lookup
    - Alias resolution
    - Dimensional compatibility checking
    - Unit conversion

    The catalog is pre-loaded with SI units and common derived units
    relevant to sustainability/emissions reporting.

    Example:
        >>> catalog = UnitCatalog()
        >>> catalog.get_unit("kWh")
        UnitDefinition(symbol='kWh', ...)
        >>> catalog.is_compatible("kWh", "MWh")
        True
        >>> catalog.convert(100, "kWh", "MWh")
        0.1
    """

    def __init__(self):
        """Initialize the unit catalog with SI units and common derived units."""
        self._units: Dict[str, UnitDefinition] = {}
        self._dimensions: Dict[str, DimensionDefinition] = {}
        self._aliases: Dict[str, str] = {}  # alias -> canonical symbol
        self._dimension_units: Dict[str, Set[str]] = {}  # dimension -> set of unit symbols

        # Load core SI and common units
        self._load_si_units()

        logger.debug(
            f"UnitCatalog initialized with {len(self._units)} units "
            f"and {len(self._dimensions)} dimensions"
        )

    # -------------------------------------------------------------------------
    # Registration Methods
    # -------------------------------------------------------------------------

    def register_unit(self, unit: UnitDefinition) -> None:
        """
        Register a unit in the catalog.

        Args:
            unit: UnitDefinition to register

        Raises:
            ValueError: If unit symbol is already registered with different definition
        """
        symbol = unit.symbol

        if symbol in self._units:
            existing = self._units[symbol]
            if existing != unit:
                raise ValueError(
                    f"Unit '{symbol}' already registered with different definition"
                )
            return  # Already registered with same definition

        self._units[symbol] = unit

        # Update dimension index
        if unit.dimension not in self._dimension_units:
            self._dimension_units[unit.dimension] = set()
        self._dimension_units[unit.dimension].add(symbol)

        logger.debug(f"Registered unit: {symbol} ({unit.dimension})")

    def register_alias(self, alias: str, canonical: str) -> None:
        """
        Register a unit alias.

        Args:
            alias: The alias symbol (e.g., "kilowatt-hour")
            canonical: The canonical symbol (e.g., "kWh")

        Raises:
            ValueError: If canonical symbol is not registered
        """
        if canonical not in self._units and canonical not in self._aliases:
            raise ValueError(
                f"Cannot register alias '{alias}': canonical unit '{canonical}' not found"
            )

        # Resolve canonical if it's itself an alias
        resolved = self._resolve_alias(canonical)
        self._aliases[alias] = resolved

        logger.debug(f"Registered alias: {alias} -> {resolved}")

    def register_dimension(self, dimension: DimensionDefinition) -> None:
        """
        Register a dimension definition.

        Args:
            dimension: DimensionDefinition to register
        """
        self._dimensions[dimension.name] = dimension

        # Initialize dimension units set if needed
        if dimension.name not in self._dimension_units:
            self._dimension_units[dimension.name] = set()

        logger.debug(f"Registered dimension: {dimension.name}")

    # -------------------------------------------------------------------------
    # Lookup Methods
    # -------------------------------------------------------------------------

    def get_unit(self, symbol: str) -> Optional[UnitDefinition]:
        """
        Get unit by symbol (handles aliases).

        Args:
            symbol: Unit symbol or alias

        Returns:
            UnitDefinition if found, None otherwise
        """
        resolved = self._resolve_alias(symbol)
        return self._units.get(resolved)

    def get_dimension(self, name: str) -> Optional[DimensionDefinition]:
        """
        Get dimension definition by name.

        Args:
            name: Dimension name

        Returns:
            DimensionDefinition if found, None otherwise
        """
        return self._dimensions.get(name)

    def get_unit_dimension(self, symbol: str) -> Optional[str]:
        """
        Get the dimension of a unit.

        Args:
            symbol: Unit symbol or alias

        Returns:
            Dimension name if found, None otherwise
        """
        unit = self.get_unit(symbol)
        return unit.dimension if unit else None

    def get_canonical_unit(self, dimension: str) -> Optional[str]:
        """
        Get the canonical SI unit for a dimension.

        Args:
            dimension: Dimension name

        Returns:
            Canonical unit symbol if found, None otherwise
        """
        dim_def = self._dimensions.get(dimension)
        return dim_def.canonical_unit if dim_def else None

    def get_si_unit(self, dimension: str) -> Optional[str]:
        """
        Get the SI base unit for a dimension.

        Args:
            dimension: Dimension name

        Returns:
            SI base unit symbol if found, None otherwise
        """
        dim_def = self._dimensions.get(dimension)
        return dim_def.si_unit if dim_def else None

    def list_units_for_dimension(self, dimension: str) -> List[str]:
        """
        List all units for a dimension.

        Args:
            dimension: Dimension name

        Returns:
            List of unit symbols for the dimension
        """
        return list(self._dimension_units.get(dimension, set()))

    def list_all_dimensions(self) -> List[str]:
        """
        List all registered dimensions.

        Returns:
            List of dimension names
        """
        return list(self._dimensions.keys())

    def list_all_units(self) -> List[str]:
        """
        List all registered unit symbols.

        Returns:
            List of unit symbols
        """
        return list(self._units.keys())

    # -------------------------------------------------------------------------
    # Validation Methods
    # -------------------------------------------------------------------------

    def is_known_unit(self, symbol: str) -> bool:
        """
        Check if a unit is known (registered or aliased).

        Args:
            symbol: Unit symbol to check

        Returns:
            True if unit is known
        """
        return self.get_unit(symbol) is not None

    def is_compatible(self, unit1: str, unit2: str) -> bool:
        """
        Check if two units have the same dimension.

        Args:
            unit1: First unit symbol
            unit2: Second unit symbol

        Returns:
            True if units are dimensionally compatible
        """
        dim1 = self.get_unit_dimension(unit1)
        dim2 = self.get_unit_dimension(unit2)

        if dim1 is None or dim2 is None:
            return False

        return dim1 == dim2

    def is_canonical(self, symbol: str) -> bool:
        """
        Check if a unit is the canonical unit for its dimension.

        Args:
            symbol: Unit symbol

        Returns:
            True if unit is canonical
        """
        unit = self.get_unit(symbol)
        return unit.is_canonical if unit else False

    # -------------------------------------------------------------------------
    # Conversion Methods
    # -------------------------------------------------------------------------

    def convert(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
        precision: int = 10
    ) -> float:
        """
        Convert value between compatible units.

        Args:
            value: Value to convert
            from_unit: Source unit symbol
            to_unit: Target unit symbol
            precision: Decimal precision for intermediate calculations

        Returns:
            Converted value

        Raises:
            ValueError: If units are unknown or incompatible
        """
        # Handle same unit case
        if from_unit == to_unit:
            return value

        # Resolve aliases
        from_resolved = self._resolve_alias(from_unit)
        to_resolved = self._resolve_alias(to_unit)

        if from_resolved == to_resolved:
            return value

        # Get unit definitions
        from_def = self._units.get(from_resolved)
        to_def = self._units.get(to_resolved)

        if from_def is None:
            raise ValueError(f"Unknown source unit: {from_unit}")
        if to_def is None:
            raise ValueError(f"Unknown target unit: {to_unit}")

        # Check dimensional compatibility
        if from_def.dimension != to_def.dimension:
            raise ValueError(
                f"Cannot convert between incompatible dimensions: "
                f"{from_unit} ({from_def.dimension}) -> {to_unit} ({to_def.dimension})"
            )

        # Convert using high-precision decimal arithmetic
        # from_unit -> SI -> to_unit
        decimal_value = Decimal(str(value))
        decimal_from_factor = Decimal(str(from_def.si_factor))
        decimal_from_offset = Decimal(str(from_def.si_offset))
        decimal_to_factor = Decimal(str(to_def.si_factor))
        decimal_to_offset = Decimal(str(to_def.si_offset))

        # To SI: (value * factor) + offset
        si_value = (decimal_value * decimal_from_factor) + decimal_from_offset

        # From SI: (si_value - offset) / factor
        result = (si_value - decimal_to_offset) / decimal_to_factor

        # Round to precision and convert to float
        quantize_str = "1." + "0" * precision
        result = result.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

        return float(result)

    def get_conversion_factor(self, from_unit: str, to_unit: str) -> float:
        """
        Get the conversion factor between two units.

        Note: This does not account for offsets (relevant for temperature).
        Use convert() for accurate conversions.

        Args:
            from_unit: Source unit symbol
            to_unit: Target unit symbol

        Returns:
            Conversion factor (multiply source by this to get target)

        Raises:
            ValueError: If units are unknown or incompatible
        """
        from_def = self.get_unit(from_unit)
        to_def = self.get_unit(to_unit)

        if from_def is None:
            raise ValueError(f"Unknown source unit: {from_unit}")
        if to_def is None:
            raise ValueError(f"Unknown target unit: {to_unit}")

        if from_def.dimension != to_def.dimension:
            raise ValueError(
                f"Cannot get conversion factor between incompatible dimensions: "
                f"{from_unit} ({from_def.dimension}) -> {to_unit} ({to_def.dimension})"
            )

        return from_def.si_factor / to_def.si_factor

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    def _resolve_alias(self, symbol: str) -> str:
        """
        Resolve a unit alias to its canonical symbol.

        Args:
            symbol: Unit symbol or alias

        Returns:
            Canonical symbol
        """
        # Normalize by stripping whitespace
        symbol = symbol.strip()

        # Check if it's an alias
        if symbol in self._aliases:
            return self._aliases[symbol]

        # Check case-insensitive alias lookup
        symbol_lower = symbol.lower()
        for alias, canonical in self._aliases.items():
            if alias.lower() == symbol_lower:
                return canonical

        # Check if the lowercase version is a registered unit
        for unit_symbol in self._units:
            if unit_symbol.lower() == symbol_lower:
                return unit_symbol

        return symbol

    def _load_si_units(self) -> None:
        """
        Load core SI units and common derived units.

        Covers dimensions relevant to sustainability/emissions reporting:
        - Energy: J, kJ, MJ, GJ, Wh, kWh, MWh, GWh, BTU, therm
        - Mass: g, kg, t, lb, oz, ton
        - Volume: L, mL, m3, gallon, barrel
        - Area: m2, km2, ha, acre, ft2
        - Length: m, km, cm, mm, mi, ft
        - Time: s, min, h, day
        - Temperature: K, C, F
        - Emissions: tCO2e, kgCO2e, MTCO2e
        """
        # ---------------------------------------------------------------------
        # Energy Dimension
        # ---------------------------------------------------------------------
        self.register_dimension(DimensionDefinition(
            name="energy",
            si_unit="J",
            canonical_unit="kWh",
            common_units=["J", "kJ", "MJ", "GJ", "Wh", "kWh", "MWh", "GWh", "BTU", "therm", "MMBtu"],
            description="Energy and work"
        ))

        # Joule family (SI base unit for energy is Joule)
        self.register_unit(UnitDefinition(
            symbol="J",
            name="joule",
            dimension="energy",
            si_factor=1.0,
            is_si_base=True
        ))
        self.register_unit(UnitDefinition(
            symbol="kJ",
            name="kilojoule",
            dimension="energy",
            si_factor=1000.0
        ))
        self.register_unit(UnitDefinition(
            symbol="MJ",
            name="megajoule",
            dimension="energy",
            si_factor=1_000_000.0
        ))
        self.register_unit(UnitDefinition(
            symbol="GJ",
            name="gigajoule",
            dimension="energy",
            si_factor=1_000_000_000.0
        ))

        # Watt-hour family (canonical for sustainability)
        self.register_unit(UnitDefinition(
            symbol="Wh",
            name="watt-hour",
            dimension="energy",
            si_factor=3600.0
        ))
        self.register_unit(UnitDefinition(
            symbol="kWh",
            name="kilowatt-hour",
            dimension="energy",
            si_factor=3_600_000.0,
            is_canonical=True
        ))
        self.register_unit(UnitDefinition(
            symbol="MWh",
            name="megawatt-hour",
            dimension="energy",
            si_factor=3_600_000_000.0
        ))
        self.register_unit(UnitDefinition(
            symbol="GWh",
            name="gigawatt-hour",
            dimension="energy",
            si_factor=3_600_000_000_000.0
        ))

        # BTU family (Imperial)
        self.register_unit(UnitDefinition(
            symbol="BTU",
            name="British thermal unit",
            dimension="energy",
            si_factor=1055.06
        ))
        self.register_unit(UnitDefinition(
            symbol="kBTU",
            name="thousand BTU",
            dimension="energy",
            si_factor=1_055_060.0
        ))
        self.register_unit(UnitDefinition(
            symbol="MMBtu",
            name="million BTU",
            dimension="energy",
            si_factor=1_055_060_000.0
        ))
        self.register_unit(UnitDefinition(
            symbol="therm",
            name="therm",
            dimension="energy",
            si_factor=105_506_000.0
        ))

        # Energy aliases
        self.register_alias("joule", "J")
        self.register_alias("joules", "J")
        self.register_alias("kilojoule", "kJ")
        self.register_alias("kilojoules", "kJ")
        self.register_alias("megajoule", "MJ")
        self.register_alias("megajoules", "MJ")
        self.register_alias("gigajoule", "GJ")
        self.register_alias("gigajoules", "GJ")
        self.register_alias("watt-hour", "Wh")
        self.register_alias("watt-hours", "Wh")
        self.register_alias("kilowatt-hour", "kWh")
        self.register_alias("kilowatt-hours", "kWh")
        self.register_alias("megawatt-hour", "MWh")
        self.register_alias("megawatt-hours", "MWh")
        self.register_alias("gigawatt-hour", "GWh")
        self.register_alias("gigawatt-hours", "GWh")
        self.register_alias("btu", "BTU")
        self.register_alias("Btu", "BTU")
        self.register_alias("therms", "therm")
        self.register_alias("MMBTU", "MMBtu")
        self.register_alias("mmbtu", "MMBtu")

        # ---------------------------------------------------------------------
        # Mass Dimension
        # ---------------------------------------------------------------------
        self.register_dimension(DimensionDefinition(
            name="mass",
            si_unit="kg",
            canonical_unit="kg",
            common_units=["g", "kg", "t", "tonne", "lb", "oz", "ton"],
            description="Mass and weight"
        ))

        # Metric mass
        self.register_unit(UnitDefinition(
            symbol="g",
            name="gram",
            dimension="mass",
            si_factor=0.001
        ))
        self.register_unit(UnitDefinition(
            symbol="kg",
            name="kilogram",
            dimension="mass",
            si_factor=1.0,
            is_si_base=True,
            is_canonical=True
        ))
        self.register_unit(UnitDefinition(
            symbol="mg",
            name="milligram",
            dimension="mass",
            si_factor=0.000001
        ))
        self.register_unit(UnitDefinition(
            symbol="t",
            name="metric ton",
            dimension="mass",
            si_factor=1000.0
        ))
        self.register_unit(UnitDefinition(
            symbol="tonne",
            name="tonne",
            dimension="mass",
            si_factor=1000.0
        ))

        # Imperial mass
        self.register_unit(UnitDefinition(
            symbol="lb",
            name="pound",
            dimension="mass",
            si_factor=0.453592
        ))
        self.register_unit(UnitDefinition(
            symbol="oz",
            name="ounce",
            dimension="mass",
            si_factor=0.0283495
        ))
        self.register_unit(UnitDefinition(
            symbol="ton",
            name="short ton",
            dimension="mass",
            si_factor=907.185
        ))
        self.register_unit(UnitDefinition(
            symbol="long_ton",
            name="long ton",
            dimension="mass",
            si_factor=1016.05
        ))

        # Mass aliases
        self.register_alias("gram", "g")
        self.register_alias("grams", "g")
        self.register_alias("kilogram", "kg")
        self.register_alias("kilograms", "kg")
        self.register_alias("milligram", "mg")
        self.register_alias("milligrams", "mg")
        self.register_alias("metric_ton", "t")
        self.register_alias("metric_tons", "t")
        self.register_alias("tonnes", "tonne")
        self.register_alias("pound", "lb")
        self.register_alias("pounds", "lb")
        self.register_alias("lbs", "lb")
        self.register_alias("ounce", "oz")
        self.register_alias("ounces", "oz")
        self.register_alias("short_ton", "ton")
        self.register_alias("short_tons", "ton")
        self.register_alias("tons", "ton")

        # ---------------------------------------------------------------------
        # Volume Dimension
        # ---------------------------------------------------------------------
        self.register_dimension(DimensionDefinition(
            name="volume",
            si_unit="m3",
            canonical_unit="L",
            common_units=["L", "mL", "m3", "gallon", "barrel", "ft3"],
            description="Volume and capacity"
        ))

        # Metric volume (SI base is m3)
        self.register_unit(UnitDefinition(
            symbol="m3",
            name="cubic meter",
            dimension="volume",
            si_factor=1.0,
            is_si_base=True
        ))
        self.register_unit(UnitDefinition(
            symbol="L",
            name="liter",
            dimension="volume",
            si_factor=0.001,
            is_canonical=True
        ))
        self.register_unit(UnitDefinition(
            symbol="mL",
            name="milliliter",
            dimension="volume",
            si_factor=0.000001
        ))

        # Imperial volume
        self.register_unit(UnitDefinition(
            symbol="gallon",
            name="US gallon",
            dimension="volume",
            si_factor=0.00378541
        ))
        self.register_unit(UnitDefinition(
            symbol="barrel",
            name="barrel (petroleum)",
            dimension="volume",
            si_factor=0.158987
        ))
        self.register_unit(UnitDefinition(
            symbol="ft3",
            name="cubic foot",
            dimension="volume",
            si_factor=0.0283168
        ))

        # Volume aliases
        self.register_alias("liter", "L")
        self.register_alias("liters", "L")
        self.register_alias("litre", "L")
        self.register_alias("litres", "L")
        self.register_alias("milliliter", "mL")
        self.register_alias("milliliters", "mL")
        self.register_alias("cubic_meter", "m3")
        self.register_alias("cubic_meters", "m3")
        self.register_alias("cubic meter", "m3")
        self.register_alias("gallons", "gallon")
        self.register_alias("gal", "gallon")
        self.register_alias("barrels", "barrel")
        self.register_alias("bbl", "barrel")
        self.register_alias("cubic_foot", "ft3")
        self.register_alias("cubic_feet", "ft3")
        self.register_alias("cubic feet", "ft3")

        # ---------------------------------------------------------------------
        # Area Dimension
        # ---------------------------------------------------------------------
        self.register_dimension(DimensionDefinition(
            name="area",
            si_unit="m2",
            canonical_unit="m2",
            common_units=["m2", "km2", "ha", "acre", "ft2"],
            description="Area and surface"
        ))

        # Metric area
        self.register_unit(UnitDefinition(
            symbol="m2",
            name="square meter",
            dimension="area",
            si_factor=1.0,
            is_si_base=True,
            is_canonical=True
        ))
        self.register_unit(UnitDefinition(
            symbol="km2",
            name="square kilometer",
            dimension="area",
            si_factor=1_000_000.0
        ))
        self.register_unit(UnitDefinition(
            symbol="ha",
            name="hectare",
            dimension="area",
            si_factor=10_000.0
        ))
        self.register_unit(UnitDefinition(
            symbol="cm2",
            name="square centimeter",
            dimension="area",
            si_factor=0.0001
        ))

        # Imperial area
        self.register_unit(UnitDefinition(
            symbol="ft2",
            name="square foot",
            dimension="area",
            si_factor=0.092903
        ))
        self.register_unit(UnitDefinition(
            symbol="acre",
            name="acre",
            dimension="area",
            si_factor=4046.86
        ))
        self.register_unit(UnitDefinition(
            symbol="sqyd",
            name="square yard",
            dimension="area",
            si_factor=0.836127
        ))

        # Area aliases
        self.register_alias("square_meter", "m2")
        self.register_alias("square_meters", "m2")
        self.register_alias("sqm", "m2")
        self.register_alias("square_kilometer", "km2")
        self.register_alias("square_kilometers", "km2")
        self.register_alias("hectare", "ha")
        self.register_alias("hectares", "ha")
        self.register_alias("square_foot", "ft2")
        self.register_alias("square_feet", "ft2")
        self.register_alias("sqft", "ft2")
        self.register_alias("acres", "acre")
        self.register_alias("square_yard", "sqyd")
        self.register_alias("square_yards", "sqyd")

        # ---------------------------------------------------------------------
        # Length Dimension
        # ---------------------------------------------------------------------
        self.register_dimension(DimensionDefinition(
            name="length",
            si_unit="m",
            canonical_unit="m",
            common_units=["m", "km", "cm", "mm", "mi", "ft", "yd"],
            description="Length and distance"
        ))

        # Metric length
        self.register_unit(UnitDefinition(
            symbol="m",
            name="meter",
            dimension="length",
            si_factor=1.0,
            is_si_base=True,
            is_canonical=True
        ))
        self.register_unit(UnitDefinition(
            symbol="km",
            name="kilometer",
            dimension="length",
            si_factor=1000.0
        ))
        self.register_unit(UnitDefinition(
            symbol="cm",
            name="centimeter",
            dimension="length",
            si_factor=0.01
        ))
        self.register_unit(UnitDefinition(
            symbol="mm",
            name="millimeter",
            dimension="length",
            si_factor=0.001
        ))

        # Imperial length
        self.register_unit(UnitDefinition(
            symbol="mi",
            name="mile",
            dimension="length",
            si_factor=1609.34
        ))
        self.register_unit(UnitDefinition(
            symbol="ft",
            name="foot",
            dimension="length",
            si_factor=0.3048
        ))
        self.register_unit(UnitDefinition(
            symbol="yd",
            name="yard",
            dimension="length",
            si_factor=0.9144
        ))
        self.register_unit(UnitDefinition(
            symbol="in",
            name="inch",
            dimension="length",
            si_factor=0.0254
        ))

        # Length aliases
        self.register_alias("meter", "m")
        self.register_alias("meters", "m")
        self.register_alias("metre", "m")
        self.register_alias("metres", "m")
        self.register_alias("kilometer", "km")
        self.register_alias("kilometers", "km")
        self.register_alias("kilometre", "km")
        self.register_alias("kilometres", "km")
        self.register_alias("centimeter", "cm")
        self.register_alias("centimeters", "cm")
        self.register_alias("millimeter", "mm")
        self.register_alias("millimeters", "mm")
        self.register_alias("mile", "mi")
        self.register_alias("miles", "mi")
        self.register_alias("foot", "ft")
        self.register_alias("feet", "ft")
        self.register_alias("yard", "yd")
        self.register_alias("yards", "yd")
        self.register_alias("inch", "in")
        self.register_alias("inches", "in")

        # ---------------------------------------------------------------------
        # Time Dimension
        # ---------------------------------------------------------------------
        self.register_dimension(DimensionDefinition(
            name="time",
            si_unit="s",
            canonical_unit="h",
            common_units=["s", "min", "h", "d"],
            description="Time duration"
        ))

        self.register_unit(UnitDefinition(
            symbol="s",
            name="second",
            dimension="time",
            si_factor=1.0,
            is_si_base=True
        ))
        self.register_unit(UnitDefinition(
            symbol="min",
            name="minute",
            dimension="time",
            si_factor=60.0
        ))
        self.register_unit(UnitDefinition(
            symbol="h",
            name="hour",
            dimension="time",
            si_factor=3600.0,
            is_canonical=True
        ))
        self.register_unit(UnitDefinition(
            symbol="d",
            name="day",
            dimension="time",
            si_factor=86400.0
        ))

        # Time aliases
        self.register_alias("second", "s")
        self.register_alias("seconds", "s")
        self.register_alias("sec", "s")
        self.register_alias("minute", "min")
        self.register_alias("minutes", "min")
        self.register_alias("hour", "h")
        self.register_alias("hours", "h")
        self.register_alias("hr", "h")
        self.register_alias("day", "d")
        self.register_alias("days", "d")

        # ---------------------------------------------------------------------
        # Temperature Dimension
        # ---------------------------------------------------------------------
        self.register_dimension(DimensionDefinition(
            name="temperature",
            si_unit="K",
            canonical_unit="K",
            common_units=["K", "C", "F"],
            description="Temperature"
        ))

        self.register_unit(UnitDefinition(
            symbol="K",
            name="kelvin",
            dimension="temperature",
            si_factor=1.0,
            si_offset=0.0,
            is_si_base=True,
            is_canonical=True
        ))
        self.register_unit(UnitDefinition(
            symbol="C",
            name="degree Celsius",
            dimension="temperature",
            si_factor=1.0,
            si_offset=273.15
        ))
        self.register_unit(UnitDefinition(
            symbol="F",
            name="degree Fahrenheit",
            dimension="temperature",
            si_factor=5.0 / 9.0,
            si_offset=273.15 - (32.0 * 5.0 / 9.0)
        ))

        # Temperature aliases
        self.register_alias("kelvin", "K")
        self.register_alias("celsius", "C")
        self.register_alias("fahrenheit", "F")
        self.register_alias("degC", "C")
        self.register_alias("degF", "F")

        # ---------------------------------------------------------------------
        # Emissions Dimension (for GHG emissions)
        # ---------------------------------------------------------------------
        self.register_dimension(DimensionDefinition(
            name="emissions",
            si_unit="kgCO2e",
            canonical_unit="tCO2e",
            common_units=["gCO2e", "kgCO2e", "tCO2e", "MTCO2e"],
            description="Greenhouse gas emissions in CO2 equivalent"
        ))

        self.register_unit(UnitDefinition(
            symbol="gCO2e",
            name="gram CO2 equivalent",
            dimension="emissions",
            si_factor=0.001
        ))
        self.register_unit(UnitDefinition(
            symbol="kgCO2e",
            name="kilogram CO2 equivalent",
            dimension="emissions",
            si_factor=1.0,
            is_si_base=True
        ))
        self.register_unit(UnitDefinition(
            symbol="tCO2e",
            name="metric ton CO2 equivalent",
            dimension="emissions",
            si_factor=1000.0,
            is_canonical=True
        ))
        self.register_unit(UnitDefinition(
            symbol="MTCO2e",
            name="metric ton CO2 equivalent",
            dimension="emissions",
            si_factor=1000.0
        ))

        # Emissions aliases
        self.register_alias("kg CO2e", "kgCO2e")
        self.register_alias("kg CO2 equivalent", "kgCO2e")
        self.register_alias("t CO2e", "tCO2e")
        self.register_alias("tonne CO2e", "tCO2e")
        self.register_alias("tonnes CO2e", "tCO2e")
        self.register_alias("metric ton CO2e", "tCO2e")


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "UnitDefinition",
    "DimensionDefinition",
    "UnitCatalog",
]
