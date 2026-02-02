"""
GL-012 STEAMQUAL SteamQualityController - Control Module

This module provides advanced control capabilities for steam quality management
including supervisory quality control, drain valve control, setpoint advisories,
and pre-approved mitigation playbook execution.

Control Architecture:
    - Advisory mode (default): Recommendations require operator confirmation
    - Automation mode: Auto-implementation within safe bounds (requires site approval)
    - Safety envelope enforcement at all times
    - Zero-hallucination deterministic control logic

Control Components:
    - QualityController: Supervisory quality control layer with constraint computation
    - DrainController: Drain valve and separator control with flooding prevention
    - SetpointAdvisor: Setpoint recommendations for desuperheaters and headers
    - PlaybookExecutor: Pre-approved mitigation playbook execution

Reference Standards:
    - ISA-5.1 Instrumentation Symbols and Identification
    - ISA-18.2 Management of Alarm Systems
    - IEC 61511 Functional Safety
    - ASME B31.1 Power Piping
    - ASME PTC 19.11 Steam and Water Sampling

Author: GreenLang Control Systems Team
Version: 1.0.0
"""

from .quality_controller import (
    QualityController,
    ControlMode,
    QualityStatus,
    ActionPriority,
    ConstraintType,
    QualityMeasurement,
    QualityTargets,
    QualityConstraints,
    OptimizationConstraint,
    CoordinatedAction,
    ControlRecommendation,
    SupervisoryState,
)

from .drain_controller import (
    DrainController,
    DrainValveState,
    SeparatorState,
    DrainSequenceType,
    BackpressureStatus,
    DrainValveConfig,
    SeparatorConfig,
    DrainSetpoint,
    SequenceStep,
    DrainSequence,
    FloodingRisk,
    BackpressureAnalysis,
    DrainRecommendation,
)

from .setpoint_advisor import (
    SetpointAdvisor,
    SetpointType,
    RampRatePolicy,
    SuperheatPolicy,
    DesuperheaterBounds,
    HeaderPressureBounds,
    RampRateLimits,
    SuperheatMarginPolicy,
    SetpointRecommendation,
    LoadChangeGuidance,
    SetpointValidation,
)

from .playbook_executor import (
    PlaybookExecutor,
    PlaybookType,
    PlaybookStatus,
    ExecutionMode,
    ConfirmationStatus,
    PlaybookStep,
    PlaybookDefinition,
    OperatorConfirmation,
    StepResult,
    PlaybookExecution,
    PlaybookAuditRecord,
)

__all__ = [
    # Quality Controller
    "QualityController",
    "ControlMode",
    "QualityStatus",
    "ActionPriority",
    "ConstraintType",
    "QualityMeasurement",
    "QualityTargets",
    "QualityConstraints",
    "OptimizationConstraint",
    "CoordinatedAction",
    "ControlRecommendation",
    "SupervisoryState",
    # Drain Controller
    "DrainController",
    "DrainValveState",
    "SeparatorState",
    "DrainSequenceType",
    "BackpressureStatus",
    "DrainValveConfig",
    "SeparatorConfig",
    "DrainSetpoint",
    "SequenceStep",
    "DrainSequence",
    "FloodingRisk",
    "BackpressureAnalysis",
    "DrainRecommendation",
    # Setpoint Advisor
    "SetpointAdvisor",
    "SetpointType",
    "RampRatePolicy",
    "SuperheatPolicy",
    "DesuperheaterBounds",
    "HeaderPressureBounds",
    "RampRateLimits",
    "SuperheatMarginPolicy",
    "SetpointRecommendation",
    "LoadChangeGuidance",
    "SetpointValidation",
    # Playbook Executor
    "PlaybookExecutor",
    "PlaybookType",
    "PlaybookStatus",
    "ExecutionMode",
    "ConfirmationStatus",
    "PlaybookStep",
    "PlaybookDefinition",
    "OperatorConfirmation",
    "StepResult",
    "PlaybookExecution",
    "PlaybookAuditRecord",
]
