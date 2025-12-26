# -*- coding: utf-8 -*-
"""
Control Module for GL-005 CombustionControlAgent

This module provides advanced control system components including:
- Adaptive PID control with online learning
- Relay feedback auto-tuning (Astrom-Hagglund method)
- Model Reference Adaptive Control (MRAC)
- Recursive Least Squares (RLS) parameter estimation
- Gain scheduling based on operating point
- Performance metrics and tuning assistance
- Velocity limiting for combustion parameters (NEW)

All implementations follow zero-hallucination principles using deterministic
algorithms from classical control theory.

Reference Standards:
- ISA-5.1: Instrumentation Symbols and Identification
- ANSI/ISA-51.1: Process Instrumentation Terminology
- IEC 61508: Functional Safety
- NFPA 85: Boiler and Combustion Systems
- Astrom and Hagglund: PID Controllers - Theory, Design, and Tuning
"""

from control.adaptive_pid import (
    AdaptivePIDController,
    AdaptivePIDConfig,
    AdaptivePIDInput,
    AdaptivePIDOutput,
    AdaptiveTuningMethod,
    GainScheduleEntry,
    PerformanceMetrics,
    TuningReport,
    SafetyConstraints,
    RelayFeedbackResult,
    RLSEstimator,
    MRACController,
    TuningAssistant,
)

from control.velocity_limiter import (
    VelocityLimiter,
    VelocityLimiterConfig,
    VelocityLimiterInput,
    VelocityLimiterOutput,
    VelocityLimiterSummary,
    VelocityLimit,
    VelocityViolation,
    ParameterState,
    ParameterType,
    VelocityLimitStatus,
    RampDirection,
    SafetyBoundType,
    create_default_limiter,
    create_conservative_limiter,
    create_aggressive_limiter,
)

__all__ = [
    # Adaptive PID
    "AdaptivePIDController",
    "AdaptivePIDConfig",
    "AdaptivePIDInput",
    "AdaptivePIDOutput",
    "AdaptiveTuningMethod",
    "GainScheduleEntry",
    "PerformanceMetrics",
    "TuningReport",
    "SafetyConstraints",
    "RelayFeedbackResult",
    "RLSEstimator",
    "MRACController",
    "TuningAssistant",
    # Velocity Limiter
    "VelocityLimiter",
    "VelocityLimiterConfig",
    "VelocityLimiterInput",
    "VelocityLimiterOutput",
    "VelocityLimiterSummary",
    "VelocityLimit",
    "VelocityViolation",
    "ParameterState",
    "ParameterType",
    "VelocityLimitStatus",
    "RampDirection",
    "SafetyBoundType",
    "create_default_limiter",
    "create_conservative_limiter",
    "create_aggressive_limiter",
]
