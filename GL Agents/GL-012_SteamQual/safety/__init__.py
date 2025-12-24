"""
GL-012 STEAMQUAL - Safety Module

Production-grade safety systems for Steam Quality Controller including:
- Circuit breaker pattern for fault tolerance
- Constraint validation for quality parameters
- Interlock management (read-only advisory monitoring)
- Physical bounds validation for sensor inputs

Safety Philosophy:
This module implements FAIL-SAFE behavior throughout. When data quality
is poor or systems are degraded, the module degrades gracefully rather
than failing catastrophically. All safety decisions default to the
conservative (safe) option.

Standards Compliance:
    - IEC 61508 (Functional Safety of E/E/PE Safety-Related Systems)
    - IEC 61511 (Functional Safety - Safety Instrumented Systems)
    - ASME PTC 19.11 (Steam Properties)
    - API 560/API 530 (Process Steam Systems)
    - NIST SP 800-160 (Systems Security Engineering - Resilience Patterns)

Zero-Hallucination Guarantee:
All safety decisions use deterministic rules from published standards.
No LLM or AI inference is used for any safety-critical calculations.
SHA-256 provenance hashing provides complete audit trail.

Module Structure:
    circuit_breaker.py - Fault tolerance with open/closed/half-open states
    constraint_validator.py - Steam quality constraint validation
    interlock_manager.py - Read-only SIS interlock monitoring
    bounds_validator.py - Physical bounds validation for sensor inputs

Example:
    >>> from safety import SteamQualCircuitBreaker, ConstraintValidator
    >>> from safety import InterlockManager, SteamQualBoundsValidator
    >>>
    >>> # Initialize safety components
    >>> circuit_breaker = SteamQualCircuitBreaker(name="scada_connector")
    >>> validator = ConstraintValidator()
    >>> interlock_mgr = InterlockManager(header_id="STEAM-HDR-001")
    >>> bounds = SteamQualBoundsValidator()
    >>>
    >>> # Use circuit breaker for external calls
    >>> async with circuit_breaker.protect():
    ...     data = await fetch_from_scada()
    >>>
    >>> # Validate steam quality constraints
    >>> result = validator.validate_quality_constraints(quality_reading)
    >>> if not result.is_valid:
    ...     logger.warning(f"Quality constraint violation: {result.violations}")

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations

# Circuit Breaker exports
from .circuit_breaker import (
    # Enums
    CircuitBreakerState,
    CircuitBreakerEvent,
    # Exceptions
    CircuitBreakerError,
    CircuitOpenError,
    CircuitHalfOpenError,
    # Models
    CircuitBreakerConfig,
    CircuitBreakerMetrics,
    CircuitBreakerAuditRecord,
    # Main class
    SteamQualCircuitBreaker,
    CircuitBreakerRegistry,
    # Functions
    get_circuit_breaker,
    get_or_create_circuit_breaker,
    circuit_protected,
    # Pre-configured breakers
    SteamQualCircuitBreakers,
)

# Constraint Validator exports
from .constraint_validator import (
    # Enums
    ConstraintType,
    ConstraintSeverity,
    ViolationSeverity,
    # Data classes
    QualityConstraint,
    ConstraintViolation,
    # Models
    ConstraintValidationResult,
    # Main class
    ConstraintValidator,
    # Constants
    STEAM_QUALITY_CONSTRAINTS,
)

# Interlock Manager exports
from .interlock_manager import (
    # Enums
    InterlockStatus,
    InterlockType,
    # Data classes
    SafetyInterlock,
    InterlockReading,
    # Models
    InterlockStatusSummary,
    # Main class
    InterlockManager,
)

# Bounds Validator exports
from .bounds_validator import (
    # Enums
    BoundsViolationSeverity,
    ValidationStatus,
    ParameterCategory,
    # Data classes
    PhysicalBounds,
    BoundsViolation,
    # Models
    BoundsValidationResult,
    SteamQualDiagnosticInput,
    # Main class
    SteamQualBoundsValidator,
    # Constants
    STEAM_QUALITY_BOUNDS,
)


# Module version
__version__ = "1.0.0"

# Safety module metadata
__safety_standards__ = [
    "IEC 61508",
    "IEC 61511",
    "ASME PTC 19.11",
    "API 560",
    "API 530",
]


__all__ = [
    # Version
    "__version__",
    "__safety_standards__",

    # Circuit Breaker
    "CircuitBreakerState",
    "CircuitBreakerEvent",
    "CircuitBreakerError",
    "CircuitOpenError",
    "CircuitHalfOpenError",
    "CircuitBreakerConfig",
    "CircuitBreakerMetrics",
    "CircuitBreakerAuditRecord",
    "SteamQualCircuitBreaker",
    "CircuitBreakerRegistry",
    "get_circuit_breaker",
    "get_or_create_circuit_breaker",
    "circuit_protected",
    "SteamQualCircuitBreakers",

    # Constraint Validator
    "ConstraintType",
    "ConstraintSeverity",
    "ViolationSeverity",
    "QualityConstraint",
    "ConstraintViolation",
    "ConstraintValidationResult",
    "ConstraintValidator",
    "STEAM_QUALITY_CONSTRAINTS",

    # Interlock Manager
    "InterlockStatus",
    "InterlockType",
    "SafetyInterlock",
    "InterlockReading",
    "InterlockStatusSummary",
    "InterlockManager",

    # Bounds Validator
    "BoundsViolationSeverity",
    "ValidationStatus",
    "ParameterCategory",
    "PhysicalBounds",
    "BoundsViolation",
    "BoundsValidationResult",
    "SteamQualDiagnosticInput",
    "SteamQualBoundsValidator",
    "STEAM_QUALITY_BOUNDS",
]
