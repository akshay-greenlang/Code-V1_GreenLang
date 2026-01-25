# -*- coding: utf-8 -*-
"""
GL-013 PredictiveMaintenance - Safety and Governance Module

Provides uncertainty gating, human-in-the-loop controls, and velocity limiting.
"""

__version__ = "1.0.0"

from .uncertainty_gating import (
    UncertaintyGate,
    UncertaintyThresholds,
    UncertaintyLevel,
    DecisionGate,
    GatingDecision,
    HumanInTheLoop,
    HumanDecision,
    AuditLogger,
    AuditLogEntry,
    AuditAction,
)

from .velocity_limiter import (
    VelocityLimiter,
    VelocityConfig,
    VelocityCheckResult,
    VelocityState,
    VelocityViolationType,
    ConstraintAction,
    check_velocity,
)

__all__ = [
    "__version__",
    # Uncertainty gating
    "UncertaintyGate",
    "UncertaintyThresholds",
    "UncertaintyLevel",
    "DecisionGate",
    "GatingDecision",
    "HumanInTheLoop",
    "HumanDecision",
    "AuditLogger",
    "AuditLogEntry",
    "AuditAction",
    # Velocity limiting
    "VelocityLimiter",
    "VelocityConfig",
    "VelocityCheckResult",
    "VelocityState",
    "VelocityViolationType",
    "ConstraintAction",
    "check_velocity",
]
