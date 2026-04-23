# -*- coding: utf-8 -*-
"""
GL-014 Exchangerpro - Safety Module

Production-grade safety enforcement for heat exchanger optimization and
cleaning recommendation systems. This module implements comprehensive
safety guardrails following GreenLang's zero-hallucination principles.

Safety Modules:
1. exceptions.py - Custom exception hierarchy for safety violations
2. velocity_limiter.py - Recommendation rate limiting to prevent oscillation
3. constraint_validator.py - Physical bounds and energy balance validation
4. guardrails.py - Operational guardrails (SIS protection, OOD detection)
5. circuit_breaker.py - ML service failure handling with graceful degradation
6. emergency_shutdown.py - Critical condition detection and alert escalation

Safety Principles:
- Never present predictions as certainties
- Fail safe on poor data quality
- Request engineering review when outside training distribution
- No sensitive OT data export without authorization
- Never bypass Safety Instrumented Systems (SIS)
- No direct control-loop manipulation (recommendations only)

Standards Compliance:
- IEC 61511: Safety Instrumented Systems for Process Industries
- IEC 61508: Functional Safety of E/E/PE Systems
- ASME PTC 4.3: Air Heater Performance
- ASME PTC 4.4: HRSG Performance
- API 660: Shell and Tube Heat Exchangers
- ISO 14414: Pump System Energy Assessment

Zero-Hallucination Guarantee:
All safety-critical checks use deterministic arithmetic with SHA-256
provenance tracking. No LLM inference is used for safety decisions.

Example:
    >>> from safety import (
    ...     ConstraintValidator,
    ...     OperationalGuardrails,
    ...     ServiceCircuitBreaker,
    ...     EmergencyResponseHandler,
    ... )
    >>>
    >>> # Validate physical constraints
    >>> validator = ConstraintValidator()
    >>> summary = validator.validate(exchanger_data)
    >>> if not summary.is_valid:
    ...     raise SafetyError(summary.get_rejection_reasons())
    >>>
    >>> # Check operational guardrails
    >>> guardrails = OperationalGuardrails()
    >>> result = guardrails.check_action(ActionType.RECOMMENDATION, uncertainty)
    >>> if result.decision == GuardrailDecision.BLOCK:
    ...     raise SafetyError(result.reason)

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "GL-BackendDeveloper"

# =============================================================================
# EXCEPTIONS
# =============================================================================

from .exceptions import (
    # Enums
    ViolationSeverity,
    SafetyDomain,
    # Context and details
    ViolationContext,
    ViolationDetails,
    # Base exception
    ExchangerproSafetyError,
    # Physical bounds
    PhysicalBoundsViolation,
    EffectivenessOutOfBoundsError,
    NegativeFlowError,
    TemperatureOutOfRangeError,
    PressureDropExceededError,
    # Energy balance
    EnergyBalanceError,
    # Instrumentation
    InstrumentationFault,
    SensorStuckFault,
    SensorStaleDataFault,
    # Model service
    ModelUnavailableError,
    ModelPredictionError,
    # Operational safety
    OperationalSafetyViolation,
    SISBypassAttemptError,
    ControlLoopManipulationError,
    UnauthorizedDataExportError,
    # Data quality
    DataQualityError,
)

# =============================================================================
# VELOCITY LIMITER
# =============================================================================

from .velocity_limiter import (
    # Enums
    RecommendationPriority,
    VelocityViolationType,
    ConstraintAction,
    CooldownReason,
    # Config
    VelocityLimiterConfig,
    # Data models
    RecommendationState,
    VelocityCheckResult,
    CooldownEvent,
    # Main class
    RecommendationVelocityLimiter,
    # Convenience function
    check_recommendation_velocity,
)

# =============================================================================
# CONSTRAINT VALIDATOR
# =============================================================================

from .constraint_validator import (
    # Enums
    ConstraintType,
    ConstraintSeverity,
    FluidPhase,
    # Config
    ConstraintLimits,
    ConstraintValidatorConfig,
    # Data models
    ConstraintCheckResult,
    ConstraintValidationSummary,
    ExchangerData,
    # Main class
    ConstraintValidator,
    # Convenience functions
    validate_effectiveness,
    validate_energy_balance,
)

# =============================================================================
# OPERATIONAL GUARDRAILS
# =============================================================================

from .guardrails import (
    # Enums
    ActionType,
    GuardrailDecision,
    UncertaintyLevel,
    DistributionStatus,
    ConservativeMode,
    # Config
    UncertaintyThresholds,
    DistributionBounds,
    GuardrailsConfig,
    # Data models
    GuardrailCheckResult,
    InputFeatures,
    UncertaintyEstimate,
    RecommendationAdjustment,
    # Main class
    OperationalGuardrails,
    # Convenience functions
    check_is_recommendation_only,
    get_conservative_urgency,
)

# =============================================================================
# CIRCUIT BREAKER
# =============================================================================

from .circuit_breaker import (
    # Enums
    CircuitState,
    CircuitEvent,
    ServiceType,
    DegradationLevel,
    # Config
    CircuitBreakerConfig,
    # Data models
    CircuitBreakerMetrics,
    CircuitBreakerEvent,
    FallbackResult,
    # Main class
    ServiceCircuitBreaker,
    # Registry
    CircuitBreakerRegistry,
    get_circuit_breaker,
    get_or_create_circuit_breaker,
    # Decorator
    circuit_protected,
    # Pre-configured
    ExchangerproCircuitBreakers,
)

# =============================================================================
# EMERGENCY RESPONSE
# =============================================================================

from .emergency_shutdown import (
    # Enums
    AlertSeverity,
    AlertState,
    EscalationLevel,
    ConditionType,
    SafeStateType,
    # Config
    AlertThresholds,
    EscalationPolicy,
    EmergencyResponseConfig,
    # Data models
    CriticalCondition,
    Alert,
    SafeStateRecommendation,
    # Main class
    EmergencyResponseHandler,
)

# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Version
    "__version__",
    "__author__",

    # === EXCEPTIONS ===
    # Enums
    "ViolationSeverity",
    "SafetyDomain",
    # Context and details
    "ViolationContext",
    "ViolationDetails",
    # Base exception
    "ExchangerproSafetyError",
    # Physical bounds
    "PhysicalBoundsViolation",
    "EffectivenessOutOfBoundsError",
    "NegativeFlowError",
    "TemperatureOutOfRangeError",
    "PressureDropExceededError",
    # Energy balance
    "EnergyBalanceError",
    # Instrumentation
    "InstrumentationFault",
    "SensorStuckFault",
    "SensorStaleDataFault",
    # Model service
    "ModelUnavailableError",
    "ModelPredictionError",
    # Operational safety
    "OperationalSafetyViolation",
    "SISBypassAttemptError",
    "ControlLoopManipulationError",
    "UnauthorizedDataExportError",
    # Data quality
    "DataQualityError",

    # === VELOCITY LIMITER ===
    # Enums
    "RecommendationPriority",
    "VelocityViolationType",
    "ConstraintAction",
    "CooldownReason",
    # Config
    "VelocityLimiterConfig",
    # Data models
    "RecommendationState",
    "VelocityCheckResult",
    "CooldownEvent",
    # Main class
    "RecommendationVelocityLimiter",
    # Convenience function
    "check_recommendation_velocity",

    # === CONSTRAINT VALIDATOR ===
    # Enums
    "ConstraintType",
    "ConstraintSeverity",
    "FluidPhase",
    # Config
    "ConstraintLimits",
    "ConstraintValidatorConfig",
    # Data models
    "ConstraintCheckResult",
    "ConstraintValidationSummary",
    "ExchangerData",
    # Main class
    "ConstraintValidator",
    # Convenience functions
    "validate_effectiveness",
    "validate_energy_balance",

    # === OPERATIONAL GUARDRAILS ===
    # Enums
    "ActionType",
    "GuardrailDecision",
    "UncertaintyLevel",
    "DistributionStatus",
    "ConservativeMode",
    # Config
    "UncertaintyThresholds",
    "DistributionBounds",
    "GuardrailsConfig",
    # Data models
    "GuardrailCheckResult",
    "InputFeatures",
    "UncertaintyEstimate",
    "RecommendationAdjustment",
    # Main class
    "OperationalGuardrails",
    # Convenience functions
    "check_is_recommendation_only",
    "get_conservative_urgency",

    # === CIRCUIT BREAKER ===
    # Enums
    "CircuitState",
    "CircuitEvent",
    "ServiceType",
    "DegradationLevel",
    # Config
    "CircuitBreakerConfig",
    # Data models
    "CircuitBreakerMetrics",
    "CircuitBreakerEvent",
    "FallbackResult",
    # Main class
    "ServiceCircuitBreaker",
    # Registry
    "CircuitBreakerRegistry",
    "get_circuit_breaker",
    "get_or_create_circuit_breaker",
    # Decorator
    "circuit_protected",
    # Pre-configured
    "ExchangerproCircuitBreakers",

    # === EMERGENCY RESPONSE ===
    # Enums
    "AlertSeverity",
    "AlertState",
    "EscalationLevel",
    "ConditionType",
    "SafeStateType",
    # Config
    "AlertThresholds",
    "EscalationPolicy",
    "EmergencyResponseConfig",
    # Data models
    "CriticalCondition",
    "Alert",
    "SafeStateRecommendation",
    # Main class
    "EmergencyResponseHandler",
]
