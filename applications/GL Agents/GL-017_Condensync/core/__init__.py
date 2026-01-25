# -*- coding: utf-8 -*-
"""
GL-017 CONDENSYNC Core Module

Core components for condenser optimization and state classification.

This module provides:
- CondenserStateClassifier: Multimodal health state classification
- CondenserBoundsValidator: Physical bounds validation with energy balance
- CircuitBreaker: Resilience patterns for external service calls
- CondenserGuardrails: Safety bounds and action gating
- SeedManager: Deterministic RNG for reproducibility

Standards Compliance:
- HEI Standards for Steam Surface Condensers (12th Edition)
- ASME PTC 12.2: Steam Surface Condensers
- GreenLang Global AI Standards v2.0

Zero-Hallucination Guarantee:
All components use deterministic algorithms.
No LLM or AI inference in core calculation paths.
Same inputs always produce identical outputs.

Example:
    >>> from core import CondenserStateClassifier, CondenserBoundsValidator
    >>> classifier = CondenserStateClassifier()
    >>> validator = CondenserBoundsValidator()
    >>> # Validate inputs
    >>> validation = validator.validate_condenser_input(
    ...     vacuum_pressure_kpa=5.0,
    ...     cw_inlet_temp_c=25.0,
    ...     cleanliness_factor=0.85
    ... )
    >>> if validation.is_valid:
    ...     # Classify condenser state
    ...     result = classifier.classify(condenser_input)

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

# Condenser State Classifier
from .condenser_state_classifier import (
    CondenserStateClassifier,
    ClassificationConfig,
    ClassificationResult,
    CondenserInput,
    IndicatorScore,
    FeatureImportance,
    CondenserState,
    ConfidenceLevel,
    SeverityLevel,
    IndicatorCategory,
    ModalityWeight,
    CF_THRESHOLDS,
    TTD_THRESHOLDS,
    AIR_LEAK_THRESHOLDS,
)

# Bounds Validator
from .bounds_validator import (
    CondenserBoundsValidator,
    BoundsValidator,  # Legacy alias
    PhysicalBounds,
    BoundsViolation,
    BoundsValidationResult,
    EnergyBalanceResult,
    DataQualityScore,
    CondenserDiagnosticInput,
    BoundsViolationSeverity,
    ValidationStatus,
    ParameterCategory,
    OperatingRegime,
    CONDENSER_BOUNDS,
)

# Circuit Breaker
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitStats,
    FailureRecord,
    CircuitState,
    FailureType,
    CircuitBreakerError,
    CircuitOpenError,
    CallTimeoutError,
    get_circuit_breaker,
    get_all_circuit_breakers,
    reset_all_circuit_breakers,
    with_circuit_breaker,
)

# Guardrails Integration
from .guardrails_integration import (
    CondenserGuardrails,
    GuardrailViolation,
    GuardrailResult,
    SafetyEnvelope,
    ViolationSeverity,
    GuardrailProfile,
    GuardrailType,
    ActionType,
    DEFAULT_SAFETY_ENVELOPE,
    STRICT_SAFETY_ENVELOPE,
    STARTUP_SAFETY_ENVELOPE,
    get_guardrails,
    with_guardrails,
)

# Seed Manager
from .seed_manager import (
    SeedManager,
    SeedConfig,
    SeedRecord,
    SeedMetrics,
    SeedDerivationMethod,
    SeedScope,
    DEFAULT_SEED,
    MAX_SEED_VALUE,
)

# Module version
__version__ = "1.0.0"

# Module exports
__all__ = [
    # Version
    "__version__",

    # Condenser State Classifier
    "CondenserStateClassifier",
    "ClassificationConfig",
    "ClassificationResult",
    "CondenserInput",
    "IndicatorScore",
    "FeatureImportance",
    "CondenserState",
    "ConfidenceLevel",
    "SeverityLevel",
    "IndicatorCategory",
    "ModalityWeight",
    "CF_THRESHOLDS",
    "TTD_THRESHOLDS",
    "AIR_LEAK_THRESHOLDS",

    # Bounds Validator
    "CondenserBoundsValidator",
    "BoundsValidator",
    "PhysicalBounds",
    "BoundsViolation",
    "BoundsValidationResult",
    "EnergyBalanceResult",
    "DataQualityScore",
    "CondenserDiagnosticInput",
    "BoundsViolationSeverity",
    "ValidationStatus",
    "ParameterCategory",
    "OperatingRegime",
    "CONDENSER_BOUNDS",

    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitStats",
    "FailureRecord",
    "CircuitState",
    "FailureType",
    "CircuitBreakerError",
    "CircuitOpenError",
    "CallTimeoutError",
    "get_circuit_breaker",
    "get_all_circuit_breakers",
    "reset_all_circuit_breakers",
    "with_circuit_breaker",

    # Guardrails Integration
    "CondenserGuardrails",
    "GuardrailViolation",
    "GuardrailResult",
    "SafetyEnvelope",
    "ViolationSeverity",
    "GuardrailProfile",
    "GuardrailType",
    "ActionType",
    "DEFAULT_SAFETY_ENVELOPE",
    "STRICT_SAFETY_ENVELOPE",
    "STARTUP_SAFETY_ENVELOPE",
    "get_guardrails",
    "with_guardrails",

    # Seed Manager
    "SeedManager",
    "SeedConfig",
    "SeedRecord",
    "SeedMetrics",
    "SeedDerivationMethod",
    "SeedScope",
    "DEFAULT_SEED",
    "MAX_SEED_VALUE",
]
