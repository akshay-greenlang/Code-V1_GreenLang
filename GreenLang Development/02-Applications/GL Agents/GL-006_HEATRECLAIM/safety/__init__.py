"""
GL-006 HEATRECLAIM - Safety Module

Safety constraint enforcement for heat exchanger network (HEN) designs.
Ensures compliance with ASME PTC 4.3/4.4, API 660, ISO 14414, and IEC 61511 standards.

This module implements fail-closed safety validation to prevent unsafe
HEN designs from being generated or deployed.

Safety Constraints Enforced (from pack.yaml):
1. DELTA_T_MIN: 5C minimum approach temperature
2. MAX_FILM_TEMPERATURE: 400C coking prevention
3. ACID_DEW_POINT: 120C minimum outlet for flue gas
4. MAX_PRESSURE_DROP: 50 kPa liquids, 5 kPa gases
5. THERMAL_STRESS_RATE: 5C/min maximum temperature change

IEC 61511 SIL-Rated Validators:
- SIL-1: Basic safety functions (flow rate constraints)
- SIL-2: Standard safety functions (temperature, pressure, pinch)
- SIL-3: High integrity functions (MAWP, emergency shutdown)

Zero-Hallucination: All constraint checks use deterministic arithmetic
with SHA-256 provenance tracking. No LLM inference for safety decisions.

Example:
    >>> from safety import SILRatedSafetySystem
    >>> safety_system = SILRatedSafetySystem(config, sil_level=2)
    >>> result = safety_system.validate_hen_design(design, hot_streams, cold_streams)
    >>> if not result.is_safe:
    ...     safety_system.trigger_emergency_action(result)
"""

from .safety_validator import SafetyValidator, SafetyValidationResult
from .constraint_validator import (
    ConstraintValidator,
    ConstraintType,
    ConstraintLimit,
    ConstraintCheckResult,
    ConstraintCheckSummary,
    PenaltyLevel,
)
from .safety_constraints import (
    # Enums
    SILLevel,
    SafetyAction,
    ConstraintCategory,
    ValidationStatus,
    # Data classes
    SafetyLimit,
    SafetyCheckResult,
    SafetyValidationSummary,
    # Validators
    BaseSafetyValidator,
    TemperatureLimitsValidator,
    PressureBoundsValidator,
    FlowRateConstraintValidator,
    PinchPointProtector,
    ThermalStressValidator,
    # Safety system
    SILRatedSafetySystem,
)
from .exceptions import (
    SafetyViolationError,
    ApproachTemperatureViolation,
    FilmTemperatureViolation,
    AcidDewPointViolation,
    PressureDropViolation,
    ThermalStressViolation,
    ViolationDetails,
    ViolationSeverity,
)
from .circuit_breaker import (
    # Enums
    CircuitBreakerState,
    CircuitBreakerEvent,
    HealthLevel,
    # Exceptions
    CircuitBreakerError,
    CircuitOpenError,
    CircuitHalfOpenError,
    LoadShedError,
    # Models
    DynamicCircuitBreakerConfig,
    CircuitBreakerMetrics,
    CircuitBreakerAuditRecord,
    # Core Classes
    DynamicCircuitBreaker,
    CircuitBreakerRegistry,
    HeatReclaimCircuitBreakers,
    # Functions
    get_circuit_breaker,
    get_or_create_circuit_breaker,
    circuit_protected,
)

__all__ = [
    # Original validator
    "SafetyValidator",
    "SafetyValidationResult",
    # Constraint validator
    "ConstraintValidator",
    "ConstraintType",
    "ConstraintLimit",
    "ConstraintCheckResult",
    "ConstraintCheckSummary",
    "PenaltyLevel",
    # IEC 61511 SIL-rated safety constraints
    "SILLevel",
    "SafetyAction",
    "ConstraintCategory",
    "ValidationStatus",
    "SafetyLimit",
    "SafetyCheckResult",
    "SafetyValidationSummary",
    "BaseSafetyValidator",
    "TemperatureLimitsValidator",
    "PressureBoundsValidator",
    "FlowRateConstraintValidator",
    "PinchPointProtector",
    "ThermalStressValidator",
    "SILRatedSafetySystem",
    # Exceptions
    "SafetyViolationError",
    "ApproachTemperatureViolation",
    "FilmTemperatureViolation",
    "AcidDewPointViolation",
    "PressureDropViolation",
    "ThermalStressViolation",
    "ViolationDetails",
    "ViolationSeverity",
    # Dynamic Circuit Breaker
    "CircuitBreakerState",
    "CircuitBreakerEvent",
    "HealthLevel",
    "CircuitBreakerError",
    "CircuitOpenError",
    "CircuitHalfOpenError",
    "LoadShedError",
    "DynamicCircuitBreakerConfig",
    "CircuitBreakerMetrics",
    "CircuitBreakerAuditRecord",
    "DynamicCircuitBreaker",
    "CircuitBreakerRegistry",
    "HeatReclaimCircuitBreakers",
    "get_circuit_breaker",
    "get_or_create_circuit_breaker",
    "circuit_protected",
]
