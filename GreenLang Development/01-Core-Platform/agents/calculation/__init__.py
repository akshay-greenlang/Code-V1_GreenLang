"""
GreenLang Calculation - Backward Compatibility Module

DEPRECATED: Calculation modules have been reorganized.
Please update your imports:

  Old: from greenlang.calculation.core_calculator import EmissionCalculator
  New: from greenlang.agents.calculation.emissions.core_calculator import EmissionCalculator
  Or:  from greenlang.agents.calculation.emissions import EmissionCalculator

This file provides backward-compatible re-exports.

Author: GreenLang Team
Date: 2025-11-21
"""

import warnings

warnings.warn(
    "The greenlang.calculation module has been moved to greenlang.agents.calculation. "
    "Please update your imports to use the new location.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export all emission calculation modules for backward compatibility
from greenlang.agents.calculation.emissions import (
    EmissionCalculator,
    CalculationResult,
    CalculationRequest,
    FactorResolution,
    CalculationStatus,
    FallbackLevel,
    Scope1Calculator,
    Scope1Result,
    Scope2Calculator,
    Scope2Result,
    Scope3Calculator,
    Scope3Result,
    BatchCalculator,
    BatchResult,
    AuditTrail,
    AuditTrailGenerator,
    CalculationStep,
    MultiGasCalculator,
    GasBreakdown,
    GWP_AR6_100YR,
    UncertaintyCalculator,
    UncertaintyResult,
    CalculationValidator,
    ValidationResult,
    ValidationError,
    UnitConverter,
    UnitConversionError,
)

__all__ = [
    # Core
    'EmissionCalculator',
    'CalculationResult',
    'CalculationRequest',
    'FactorResolution',
    'CalculationStatus',
    'FallbackLevel',
    # Scope calculators
    'Scope1Calculator',
    'Scope1Result',
    'Scope2Calculator',
    'Scope2Result',
    'Scope3Calculator',
    'Scope3Result',
    # Batch
    'BatchCalculator',
    'BatchResult',
    # Audit
    'AuditTrail',
    'AuditTrailGenerator',
    'CalculationStep',
    # Gas
    'MultiGasCalculator',
    'GasBreakdown',
    'GWP_AR6_100YR',
    # Uncertainty
    'UncertaintyCalculator',
    'UncertaintyResult',
    # Validation
    'CalculationValidator',
    'ValidationResult',
    'ValidationError',
    # Utilities
    'UnitConverter',
    'UnitConversionError',
]
