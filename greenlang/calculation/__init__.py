"""
GreenLang Zero-Hallucination Calculation Engine

This module provides deterministic, bit-perfect, reproducible calculations
for greenhouse gas emissions across Scopes 1, 2, and 3.

Key Guarantees:
- ZERO HALLUCINATION: No LLM in calculation path
- 100% DETERMINISTIC: Same input → Same output (bit-perfect)
- FULL PROVENANCE: SHA-256 hash audit trail for every calculation
- REGULATORY COMPLIANT: GHG Protocol, IPCC, EPA standards
- PERFORMANCE: <100ms per calculation, 10,000+ calculations/batch

Components:
- EmissionCalculator: Core calculation engine
- Scope1Calculator: Direct emissions (combustion, fugitives, process)
- Scope2Calculator: Indirect energy emissions (location & market-based)
- Scope3Calculator: Value chain emissions (15 categories)
- MultiGasCalculator: CO2e decomposition into CO2, CH4, N2O
- UnitConverter: Deterministic unit conversions
- UncertaintyCalculator: Monte Carlo uncertainty quantification
- AuditTrailGenerator: Complete provenance tracking
- BatchCalculator: High-performance batch processing
- CalculationValidator: Input/output validation

Version: 1.0.0
Last Updated: 2025-01-15
"""

from greenlang.calculation.core_calculator import (
    EmissionCalculator,
    CalculationResult,
    CalculationRequest,
    FactorResolution,
)

from greenlang.calculation.scope1_calculator import Scope1Calculator, Scope1Result
from greenlang.calculation.scope2_calculator import Scope2Calculator, Scope2Result
from greenlang.calculation.scope3_calculator import Scope3Calculator, Scope3Result

from greenlang.calculation.gas_decomposition import (
    MultiGasCalculator,
    GasBreakdown,
    GWP_AR6_100YR,
)

from greenlang.calculation.unit_converter import UnitConverter, UnitConversionError

from greenlang.calculation.uncertainty import (
    UncertaintyCalculator,
    UncertaintyResult,
)

from greenlang.calculation.audit_trail import (
    AuditTrailGenerator,
    AuditTrail,
    CalculationStep,
)

from greenlang.calculation.batch_calculator import (
    BatchCalculator,
    BatchResult,
)

from greenlang.calculation.validator import (
    CalculationValidator,
    ValidationResult,
    ValidationError,
)

__all__ = [
    # Core
    "EmissionCalculator",
    "CalculationResult",
    "CalculationRequest",
    "FactorResolution",
    # Scope-specific
    "Scope1Calculator",
    "Scope1Result",
    "Scope2Calculator",
    "Scope2Result",
    "Scope3Calculator",
    "Scope3Result",
    # Multi-gas
    "MultiGasCalculator",
    "GasBreakdown",
    "GWP_AR6_100YR",
    # Utilities
    "UnitConverter",
    "UnitConversionError",
    "UncertaintyCalculator",
    "UncertaintyResult",
    "AuditTrailGenerator",
    "AuditTrail",
    "CalculationStep",
    "BatchCalculator",
    "BatchResult",
    "CalculationValidator",
    "ValidationResult",
    "ValidationError",
]

__version__ = "1.0.0"
