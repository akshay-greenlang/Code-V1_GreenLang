"""
GreenLang Framework - Shared Utilities

Common utilities for all GreenLang agents including:
- Provenance tracking (SHA-256 based)
- Unit conversion (SI, Imperial, CGS)
- Validation engines (Pydantic-compatible)
- Base calculator classes (deterministic)
- Physical constants (NIST, IAPWS)
- Emission factors (DEFRA, EPA, IPCC)
- Guardrails integration (OWASP LLM Top 10, NIST AI RMF)
- Circuit breaker (fault tolerance)
"""

from .provenance import ProvenanceTracker, ProvenanceRecord, TrackingContext
from .units import (
    UnitConverter,
    UnitSystem,
    TemperatureUnit,
    EnergyUnit,
    PowerUnit,
    MassUnit,
    PressureUnit,
    UNIT_CONVERTER,
)
from .validation import (
    ValidationEngine,
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
    GreenLangValidators,
)
from .calculator_base import (
    DeterministicCalculator,
    CalculationResult,
    BatchCalculator,
    CachedCalculator,
)
from .constants import (
    PhysicalConstants,
    EmissionFactors,
    EmissionFactor,
    ConstantSource,
    GWP,
)
from .guardrails_integration import (
    GuardrailsIntegration,
    GuardrailExecutionResult,
    ViolationRecord,
    ViolationLogger,
    GuardrailViolationError,
    GuardrailProfile,
    GuardrailMode,
    with_guardrails,
    validate_input,
    validate_output,
    get_integration,
)
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitMetrics,
    StateTransitionEvent,
    CircuitState,
    CircuitBreakerError,
    CircuitOpenError,
    CircuitHalfOpenError,
    circuit_breaker,
    CIRCUIT_BREAKER_REGISTRY,
)

__all__ = [
    # Provenance
    "ProvenanceTracker",
    "ProvenanceRecord",
    "TrackingContext",
    # Units
    "UnitConverter",
    "UnitSystem",
    "TemperatureUnit",
    "EnergyUnit",
    "PowerUnit",
    "MassUnit",
    "PressureUnit",
    "UNIT_CONVERTER",
    # Validation
    "ValidationEngine",
    "ValidationResult",
    "ValidationIssue",
    "ValidationSeverity",
    "GreenLangValidators",
    # Calculators
    "DeterministicCalculator",
    "CalculationResult",
    "BatchCalculator",
    "CachedCalculator",
    # Constants
    "PhysicalConstants",
    "EmissionFactors",
    "EmissionFactor",
    "ConstantSource",
    "GWP",
    # Guardrails Integration
    "GuardrailsIntegration",
    "GuardrailExecutionResult",
    "ViolationRecord",
    "ViolationLogger",
    "GuardrailViolationError",
    "GuardrailProfile",
    "GuardrailMode",
    "with_guardrails",
    "validate_input",
    "validate_output",
    "get_integration",
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "CircuitMetrics",
    "StateTransitionEvent",
    "CircuitState",
    "CircuitBreakerError",
    "CircuitOpenError",
    "CircuitHalfOpenError",
    "circuit_breaker",
    "CIRCUIT_BREAKER_REGISTRY",
]
