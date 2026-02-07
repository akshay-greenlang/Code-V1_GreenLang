# -*- coding: utf-8 -*-
"""
Dimensional Analyzer - AGENT-FOUND-003: Unit & Reference Normalizer

Provides unit compatibility checking, dimension detection, and
base unit lookup for the Normalizer SDK. Used by the converter
and API layers to validate conversions before execution.

Example:
    >>> from greenlang.normalizer.dimensional import DimensionalAnalyzer
    >>> analyzer = DimensionalAnalyzer()
    >>> analyzer.check_compatibility("kWh", "MWh")  # True
    >>> analyzer.check_compatibility("kWh", "kg")    # False

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-003 Unit & Reference Normalizer
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import List, Optional

from greenlang.normalizer.converter import (
    BASE_UNITS,
    DIMENSION_UNITS,
    UnitConverter,
)
from greenlang.normalizer.models import DimensionInfo, UnitDimension

logger = logging.getLogger(__name__)


class DimensionalAnalyzer:
    """Dimensional analysis engine for unit compatibility.

    Provides methods to check whether two units share the same
    physical dimension, look up the dimension for a unit, and
    enumerate all known dimensions and their units.

    All lookups are O(1) hash-table checks against the static
    conversion tables from the converter module.

    Example:
        >>> analyzer = DimensionalAnalyzer()
        >>> analyzer.check_compatibility("kWh", "MWh")
        True
        >>> analyzer.get_dimension("kg")
        <UnitDimension.MASS: 'mass'>
    """

    def __init__(self) -> None:
        """Initialize DimensionalAnalyzer."""
        self._converter = UnitConverter()
        logger.info("DimensionalAnalyzer initialized")

    def check_compatibility(self, from_unit: str, to_unit: str) -> bool:
        """Check if two units are in the same physical dimension.

        Args:
            from_unit: Source unit name.
            to_unit: Target unit name.

        Returns:
            True if both units share the same dimension.
        """
        from_dim = self.get_dimension(from_unit)
        to_dim = self.get_dimension(to_unit)
        return from_dim is not None and from_dim == to_dim

    def get_dimension(self, unit_str: str) -> Optional[UnitDimension]:
        """Get the physical dimension for a unit.

        Args:
            unit_str: Unit name (case-insensitive).

        Returns:
            UnitDimension enum value or None if unknown.
        """
        normalized = self._converter.normalize_unit_name(unit_str)
        for dimension, units_table in DIMENSION_UNITS.items():
            if normalized in units_table:
                return dimension
        return None

    def get_base_unit(self, dimension: UnitDimension) -> str:
        """Get the base unit symbol for a dimension.

        Args:
            dimension: Physical dimension.

        Returns:
            Base unit symbol string.

        Raises:
            ValueError: If dimension has no base unit.
        """
        base = BASE_UNITS.get(dimension)
        if base is None:
            raise ValueError(f"No base unit for dimension: {dimension.value}")
        return base

    def list_dimensions(self) -> List[DimensionInfo]:
        """List all supported dimensions with their units.

        Returns:
            List of DimensionInfo objects.
        """
        result: List[DimensionInfo] = []
        for dim, units_table in DIMENSION_UNITS.items():
            result.append(DimensionInfo(
                dimension=dim,
                base_unit=BASE_UNITS.get(dim, ""),
                supported_units=list(units_table.keys()),
            ))
        return result

    def is_valid_unit(self, unit_str: str) -> bool:
        """Check if a unit string is recognised.

        Args:
            unit_str: Unit name (case-insensitive).

        Returns:
            True if the unit is in any dimension table.
        """
        return self.get_dimension(unit_str) is not None


__all__ = [
    "DimensionalAnalyzer",
]
