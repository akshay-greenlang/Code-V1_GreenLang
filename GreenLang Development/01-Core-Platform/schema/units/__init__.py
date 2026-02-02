"""
Units Module for GL-FOUND-X-002.

This module provides unit catalog and conversion capabilities for
GreenLang schema validation and normalization.

Components:
    - catalog: Unit catalog with dimension tracking
    - dimensions: Physical dimension definitions
    - conversions: Unit conversion logic
    - packs: Domain-specific unit packs (SI, climate, finance)

Example:
    >>> from greenlang.schema.units import UnitCatalog, UnitDefinition
    >>> catalog = UnitCatalog()
    >>> catalog.is_compatible("kWh", "MWh")
    True
    >>> catalog.convert(100, "kWh", "MWh")
    0.1
    >>> catalog.get_unit_dimension("kg")
    'mass'
"""

from greenlang.schema.units.catalog import (
    UnitDefinition,
    DimensionDefinition,
    UnitCatalog,
)


__all__ = [
    # Core classes
    "UnitDefinition",
    "DimensionDefinition",
    "UnitCatalog",
]
