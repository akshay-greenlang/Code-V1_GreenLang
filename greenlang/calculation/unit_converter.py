# -*- coding: utf-8 -*-
"""
DEPRECATED: Unit Converter Module

This module has been consolidated into greenlang.utils.unit_converter.
This file now provides backward-compatible re-exports with deprecation warnings.

Please update your imports:
    OLD: from greenlang.calculation.unit_converter import UnitConverter
    NEW: from greenlang.utils.unit_converter import UnitConverter
    OR:  from greenlang.utils import UnitConverter

The consolidated version includes all features from both modules:
- ZERO-HALLUCINATION deterministic conversions
- Decimal precision
- Energy, mass, volume, area, distance, time conversions
- Fuel-specific energy content calculations
- Emission unit conversions

This re-export will be removed in version 2.0.0.
"""

import warnings
from greenlang.utils.unit_converter import (
    UnitConverter,
    UnitConversionError,
)

# Issue deprecation warning on import
warnings.warn(
    "greenlang.calculation.unit_converter is deprecated. "
    "Import from greenlang.utils.unit_converter instead. "
    "This compatibility layer will be removed in version 2.0.0.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    "UnitConverter",
    "UnitConversionError",
]
