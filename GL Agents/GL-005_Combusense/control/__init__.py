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

All implementations follow zero-hallucination principles using deterministic
algorithms from classical control theory. Adaptive tuning is ONLY for setpoint
optimization loops - safety-critical control loops use fixed, validated gains.

Reference Standards:
- ISA-5.1: Instrumentation Symbols and Identification
- ANSI/ISA-51.1: Process Instrumentation Terminology
- IEC 61508: Functional Safety
- Astrom & Hagglund: PID Controllers - Theory, Design, and Tuning
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

__all__ = [
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
]
