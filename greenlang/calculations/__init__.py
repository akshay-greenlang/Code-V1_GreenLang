"""
GreenLang Physics Calculations - Backward Compatibility Module

DEPRECATED: This module has been moved to greenlang.calculation.physics
Please update your imports:

  Old: from greenlang.calculations import steam_tables
  New: from greenlang.calculation.physics import steam_tables

This file provides backward-compatible re-exports.

Author: GreenLang Team
Date: 2025-11-21
"""

import warnings

warnings.warn(
    "The greenlang.calculations module has been moved to greenlang.calculation.physics. "
    "Please update your imports to use the new location.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = []
