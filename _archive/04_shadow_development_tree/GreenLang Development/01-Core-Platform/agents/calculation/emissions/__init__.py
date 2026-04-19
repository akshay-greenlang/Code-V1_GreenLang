"""
GreenLang Emission Calculations - Core Deterministic Emission Calculation Engine

This module provides core emission calculation functionality for Scope 1, 2, and 3.

Features:
- Deterministic core calculator
- Scope-specific calculators (Scope 1, 2, 3)
- Batch calculation support
- Audit trail tracking
- Gas decomposition
- Uncertainty quantification
- Input validation
- Unit conversion

Author: GreenLang Team
Date: 2025-11-21
"""

from .core_calculator import (
    EmissionCalculator,
    CalculationResult,
    CalculationRequest,
    FactorResolution,
    CalculationStatus,
    FallbackLevel,
)
from .scope1_calculator import Scope1Calculator, Scope1Result
from .scope2_calculator import Scope2Calculator, Scope2Result
from .scope3_calculator import Scope3Calculator, Scope3Result
from .batch_calculator import BatchCalculator, BatchResult
from .audit_trail import (
    AuditTrail,
    AuditTrailGenerator,
    CalculationStep,
)
from .gas_decomposition import (
    MultiGasCalculator,
    GasBreakdown,
    GWP_AR6_100YR,
)
from .uncertainty import UncertaintyCalculator, UncertaintyResult
from .validator import (
    CalculationValidator,
    ValidationResult,
    ValidationError,
)
from .unit_converter import UnitConverter, UnitConversionError


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
