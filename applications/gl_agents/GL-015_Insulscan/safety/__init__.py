# -*- coding: utf-8 -*-
"""
GL-015 Insulscan - Safety Module

Production-grade safety enforcement for insulation scanning and thermal
assessment systems. This module implements comprehensive safety guardrails
following GreenLang's zero-hallucination principles.

Safety Modules:
1. exceptions.py - Custom exception hierarchy for safety violations
2. velocity_limiter.py - Condition assessment rate limiting to prevent oscillation
3. circuit_breaker.py - ML service failure handling with graceful degradation
4. constraint_validator.py - Physical bounds and temperature validation
5. guardrails.py - Operational guardrails (personnel safety, investment limits)
6. emergency_shutdown.py - Emergency shutdown procedures for insulation monitoring

Safety Principles:
- Personnel safety is non-negotiable (burn prevention)
- Never present predictions as certainties
- Fail safe on poor data quality
- Recommendations only, no autonomous actions
- Request engineering review when outside training distribution
- Full provenance tracking for audit trails

Standards Compliance:
- ASTM C680: Heat Loss Calculations for Insulation
- ASTM C1055: Safe Surface Temperature Limit
- ISO 12241: Thermal Insulation for Building Equipment
- CINI Manual: Insulation Inspection Guidelines
- OSHA 1910.147: Control of Hazardous Energy
- OSHA 29 CFR 1910.132: Personal Protective Equipment

Zero-Hallucination Guarantee:
All safety-critical checks use deterministic arithmetic with SHA-256
provenance tracking. No LLM inference is used for safety decisions.

Example:
    >>> from safety import (
    ...     InsulationConstraintValidator,
    ...     InsulationGuardrails,
    ...     InsulationCircuitBreaker,
    ...     InsulationVelocityLimiter,
    ... )
    >>>
    >>> # Validate physical constraints
    >>> validator = InsulationConstraintValidator()
    >>> summary = validator.validate(insulation_data)
    >>> if not summary.is_valid:
    ...     raise SafetyError(summary.get_rejection_reasons())
    >>>
    >>> # Check operational guardrails
    >>> guardrails = InsulationGuardrails()
    >>> result = guardrails.evaluate_recommendation(recommendation)
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
    InsulscanSafetyError,
    # Validation errors
    InsulationValidationError,
    # Thermal measurement errors
    ThermalMeasurementError,
    SurfaceBelowAmbientError,
    # Constraint violations
    ConstraintViolationError,
    InsulationThicknessError,
    HeatLossImplausibleError,
    # Safety limit errors
    SafetyLimitExceededError,
    BurnRiskError,
    InvestmentLimitExceededError,
    # Model service errors
    ModelUnavailableError,
)

# =============================================================================
# VELOCITY LIMITER
# =============================================================================

from .velocity_limiter import (
    # Enums
    ConditionPriority,
    VelocityViolationType,
    ConstraintAction,
    CooldownReason,
    # Config
    InsulationVelocityLimiterConfig,
    # Data models
    ConditionState,
    VelocityCheckResult,
    CooldownEvent,
    # Main class
    InsulationVelocityLimiter,
    # Convenience function
    check_condition_velocity,
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
    InsulationCircuitBreakerConfig,
    # Data models
    CircuitBreakerMetrics,
    CircuitBreakerEvent,
    FallbackResult,
    # Main class
    InsulationCircuitBreaker,
    # Registry
    InsulationCircuitBreakerRegistry,
    get_insulation_circuit_breaker,
    get_or_create_insulation_circuit_breaker,
    # Decorator
    insulation_circuit_protected,
    # Pre-configured
    InsulscanCircuitBreakers,
)

# =============================================================================
# CONSTRAINT VALIDATOR
# =============================================================================

from .constraint_validator import (
    # Enums
    ConstraintType,
    ConstraintSeverity,
    InsulationType,
    AssetType,
    # Config
    InsulationConstraintLimits,
    InsulationConstraintValidatorConfig,
    # Data models
    ConstraintCheckResult,
    InsulationConstraintValidationSummary,
    InsulationData,
    # Main class
    InsulationConstraintValidator,
    # Convenience functions
    validate_temperature_relationship,
    validate_heat_loss,
)

# =============================================================================
# EMERGENCY SHUTDOWN
# =============================================================================

from .emergency_shutdown import (
    # Enums
    ShutdownLevel,
    ConditionType as EmergencyConditionType,
    ConditionState as EmergencyConditionState,
    EscalationLevel,
    ResponseAction,
    # Data models
    ThermalMeasurement,
    ShutdownCondition,
    ShutdownEvent,
    ShutdownResult,
    # Config
    EmergencyShutdownConfig,
    # Main class
    InsulationEmergencyShutdown,
    # Convenience functions
    check_burn_risk_quick,
    calculate_thermal_runaway_rate,
)

# =============================================================================
# OPERATIONAL GUARDRAILS
# =============================================================================

from .guardrails import (
    # Enums
    ActionType,
    GuardrailDecision,
    SafetyPriority,
    BurnRiskLevel,
    InvestmentCategory,
    # Config
    PersonnelSafetyConfig,
    InvestmentLimitsConfig,
    PriorityOverrideConfig,
    InsulationGuardrailsConfig,
    # Data models
    GuardrailCheckResult,
    RepairRecommendation,
    SafetyAlert,
    # Main class
    InsulationGuardrails,
    # Convenience functions
    check_burn_risk_simple,
    get_investment_category,
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
    "InsulscanSafetyError",
    # Validation errors
    "InsulationValidationError",
    # Thermal measurement errors
    "ThermalMeasurementError",
    "SurfaceBelowAmbientError",
    # Constraint violations
    "ConstraintViolationError",
    "InsulationThicknessError",
    "HeatLossImplausibleError",
    # Safety limit errors
    "SafetyLimitExceededError",
    "BurnRiskError",
    "InvestmentLimitExceededError",
    # Model service errors
    "ModelUnavailableError",

    # === VELOCITY LIMITER ===
    # Enums
    "ConditionPriority",
    "VelocityViolationType",
    "ConstraintAction",
    "CooldownReason",
    # Config
    "InsulationVelocityLimiterConfig",
    # Data models
    "ConditionState",
    "VelocityCheckResult",
    "CooldownEvent",
    # Main class
    "InsulationVelocityLimiter",
    # Convenience function
    "check_condition_velocity",

    # === CIRCUIT BREAKER ===
    # Enums
    "CircuitState",
    "CircuitEvent",
    "ServiceType",
    "DegradationLevel",
    # Config
    "InsulationCircuitBreakerConfig",
    # Data models
    "CircuitBreakerMetrics",
    "CircuitBreakerEvent",
    "FallbackResult",
    # Main class
    "InsulationCircuitBreaker",
    # Registry
    "InsulationCircuitBreakerRegistry",
    "get_insulation_circuit_breaker",
    "get_or_create_insulation_circuit_breaker",
    # Decorator
    "insulation_circuit_protected",
    # Pre-configured
    "InsulscanCircuitBreakers",

    # === CONSTRAINT VALIDATOR ===
    # Enums
    "ConstraintType",
    "ConstraintSeverity",
    "InsulationType",
    "AssetType",
    # Config
    "InsulationConstraintLimits",
    "InsulationConstraintValidatorConfig",
    # Data models
    "ConstraintCheckResult",
    "InsulationConstraintValidationSummary",
    "InsulationData",
    # Main class
    "InsulationConstraintValidator",
    # Convenience functions
    "validate_temperature_relationship",
    "validate_heat_loss",

    # === EMERGENCY SHUTDOWN ===
    # Enums
    "ShutdownLevel",
    "EmergencyConditionType",
    "EmergencyConditionState",
    "EscalationLevel",
    "ResponseAction",
    # Data models
    "ThermalMeasurement",
    "ShutdownCondition",
    "ShutdownEvent",
    "ShutdownResult",
    # Config
    "EmergencyShutdownConfig",
    # Main class
    "InsulationEmergencyShutdown",
    # Convenience functions
    "check_burn_risk_quick",
    "calculate_thermal_runaway_rate",

    # === OPERATIONAL GUARDRAILS ===
    # Enums
    "ActionType",
    "GuardrailDecision",
    "SafetyPriority",
    "BurnRiskLevel",
    "InvestmentCategory",
    # Config
    "PersonnelSafetyConfig",
    "InvestmentLimitsConfig",
    "PriorityOverrideConfig",
    "InsulationGuardrailsConfig",
    # Data models
    "GuardrailCheckResult",
    "RepairRecommendation",
    "SafetyAlert",
    # Main class
    "InsulationGuardrails",
    # Convenience functions
    "check_burn_risk_simple",
    "get_investment_category",
]
