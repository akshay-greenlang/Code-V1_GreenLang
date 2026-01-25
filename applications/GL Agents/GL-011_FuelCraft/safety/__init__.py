# -*- coding: utf-8 -*-
"""
GL-011 FuelCraft Safety Module.

This module provides comprehensive safety functionality for fuel blending and
storage operations including:

- Fuel compatibility matrix with co-mingling rules
- Quality constraints for sulfur, ash, water, metals
- Storage constraints for tanks (fill levels, flash point, vapor pressure)
- Transfer constraints with interlock management
- IEC 61511 compliant circuit breaker
- ISA-18.2 compliant alarm management

All safety modules implement:
- Fail-closed behavior for unknown/critical conditions
- Zero-hallucination (deterministic rule lookup)
- SHA-256 provenance tracking for audit trails

Reference Standards:
- IEC 61511: Functional Safety - Safety Instrumented Systems
- ISA-18.2: Management of Alarm Systems
- NFPA 30: Flammable and Combustible Liquids Code
- API 2610: Terminal and Tank Facilities
- ISO 8217: Marine Fuel Standards
- MARPOL Annex VI: Sulfur Limits

Example:
    >>> from safety import FuelCompatibilityMatrix, QualityConstraintValidator
    >>> from safety import CircuitBreaker, AlarmManager
    >>>
    >>> # Check fuel compatibility
    >>> matrix = FuelCompatibilityMatrix()
    >>> result = matrix.check_compatibility("HFO", "VLSFO")
    >>> if not result.can_co_mingle:
    ...     raise SafetyError(result.reason)
    >>>
    >>> # Validate fuel quality
    >>> validator = QualityConstraintValidator()
    >>> result = validator.validate_fuel_quality(sulfur_pct=0.45)
    >>> assert result.status == "pass"

Author: GL-BackendDeveloper
Date: 2025-01-01
Version: 1.0.0
"""

from .fuel_compatibility import (
    FuelCategory,
    CompatibilityLevel,
    ContaminationRisk,
    SegregationType,
    FuelType,
    CompatibilityRule,
    CompatibilityCheckResult,
    FuelCompatibilityMatrix,
    DEFAULT_FUEL_TYPES,
    DEFAULT_COMPATIBILITY_RULES,
)

from .quality_constraints import (
    ConstraintSeverity,
    ValidationStatus,
    ConstraintViolation,
    QualityValidationResult,
    QualityLimit,
    SulfurConstraint,
    AshConstraint,
    WaterContentConstraint,
    ViscosityBandConstraint,
    MetalsConstraint,
    EquipmentQualityLimits,
    QualityConstraintValidator,
)

from .storage_constraints import (
    SafetyAction,
    StorageViolation,
    StorageValidationResult,
    TankConfig,
    FlashPointValidator,
    VaporPressureValidator,
    MinHeelConstraint,
    MaxFillConstraint,
    OverfillProtectionPolicy,
    TemperatureRangeValidator,
    StorageConstraintValidator,
)

from .transfer_constraints import (
    TransferAction,
    InterlockState,
    ValveState,
    PumpState,
    TransferViolation,
    TransferValidationResult,
    InterlockStatus,
    TransferRate,
    RateLimitValidator,
    MinStableFlowConstraint,
    InterlockManager,
    TransferSequenceValidator,
    TransferConstraintValidator,
)

from .circuit_breaker import (
    CircuitState,
    SILLevel,
    FailureMode,
    RecoveryStrategy,
    CircuitEvent,
    CircuitMetrics,
    CircuitBreakerConfig,
    CircuitBreaker,
    CircuitOpenError,
    CircuitBreakerRegistry,
    create_safety_circuit_breaker,
)

from .alarm_manager import (
    AlarmSeverity,
    AlarmState,
    AlarmPriority,
    EscalationLevel,
    AlarmDefinition,
    AlarmInstance,
    EscalationRule,
    AlarmStatistics,
    OperatorReviewTrigger,
    RecommendationBlockPolicy,
    AlarmManager,
)

__all__ = [
    # Fuel Compatibility
    "FuelCategory",
    "CompatibilityLevel",
    "ContaminationRisk",
    "SegregationType",
    "FuelType",
    "CompatibilityRule",
    "CompatibilityCheckResult",
    "FuelCompatibilityMatrix",
    "DEFAULT_FUEL_TYPES",
    "DEFAULT_COMPATIBILITY_RULES",
    # Quality Constraints
    "ConstraintSeverity",
    "ValidationStatus",
    "ConstraintViolation",
    "QualityValidationResult",
    "QualityLimit",
    "SulfurConstraint",
    "AshConstraint",
    "WaterContentConstraint",
    "ViscosityBandConstraint",
    "MetalsConstraint",
    "EquipmentQualityLimits",
    "QualityConstraintValidator",
    # Storage Constraints
    "SafetyAction",
    "StorageViolation",
    "StorageValidationResult",
    "TankConfig",
    "FlashPointValidator",
    "VaporPressureValidator",
    "MinHeelConstraint",
    "MaxFillConstraint",
    "OverfillProtectionPolicy",
    "TemperatureRangeValidator",
    "StorageConstraintValidator",
    # Transfer Constraints
    "TransferAction",
    "InterlockState",
    "ValveState",
    "PumpState",
    "TransferViolation",
    "TransferValidationResult",
    "InterlockStatus",
    "TransferRate",
    "RateLimitValidator",
    "MinStableFlowConstraint",
    "InterlockManager",
    "TransferSequenceValidator",
    "TransferConstraintValidator",
    # Circuit Breaker
    "CircuitState",
    "SILLevel",
    "FailureMode",
    "RecoveryStrategy",
    "CircuitEvent",
    "CircuitMetrics",
    "CircuitBreakerConfig",
    "CircuitBreaker",
    "CircuitOpenError",
    "CircuitBreakerRegistry",
    "create_safety_circuit_breaker",
    # Alarm Manager
    "AlarmSeverity",
    "AlarmState",
    "AlarmPriority",
    "EscalationLevel",
    "AlarmDefinition",
    "AlarmInstance",
    "EscalationRule",
    "AlarmStatistics",
    "OperatorReviewTrigger",
    "RecommendationBlockPolicy",
    "AlarmManager",
]

__version__ = "1.0.0"
