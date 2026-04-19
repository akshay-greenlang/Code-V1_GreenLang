"""
GL-010: SBTi Validation Agent

This module implements the Science Based Targets initiative (SBTi) Validation Agent
for validating corporate emission reduction targets against SBTi methodologies.

Exports:
    SBTiValidationAgent: Main agent class
    SBTiInput: Input data model
    SBTiOutput: Output data model
    TargetType: Enum for target types
    AmbitionLevel: Enum for ambition levels
    PathwayType: Enum for decarbonization pathways
    SectorPathway: Enum for sectoral pathways
    Scope3EngagementType: Enum for Scope 3 approaches
    NeutralizationType: Enum for net-zero neutralization
    ProgressStatus: Enum for progress tracking status
"""

from .agent import (
    # Main agent
    SBTiValidationAgent,
    # Input models
    SBTiInput,
    ScopeEmissions,
    TargetDefinition,
    IntensityMetric,
    Scope3EngagementTarget,
    NeutralizationPlan,
    FLAGTarget,
    CurrentProgress,
    # Output models
    SBTiOutput,
    ValidationResult,
    TargetValidation,
    ProgressTracking,
    Recommendation,
    PathwayCalculation,
    TargetTrajectory,
    TargetTrajectoryPoint,
    NetZeroValidation,
    FLAGValidation,
    # Enums
    TargetType,
    AmbitionLevel,
    PathwayType,
    SectorPathway,
    ValidationStatus,
    ScopeType,
    Scope3EngagementType,
    NeutralizationType,
    ProgressStatus,
    # Constants
    SBTiPathwayConstants,
)

__all__ = [
    # Main agent
    "SBTiValidationAgent",
    # Input models
    "SBTiInput",
    "ScopeEmissions",
    "TargetDefinition",
    "IntensityMetric",
    "Scope3EngagementTarget",
    "NeutralizationPlan",
    "FLAGTarget",
    "CurrentProgress",
    # Output models
    "SBTiOutput",
    "ValidationResult",
    "TargetValidation",
    "ProgressTracking",
    "Recommendation",
    "PathwayCalculation",
    "TargetTrajectory",
    "TargetTrajectoryPoint",
    "NetZeroValidation",
    "FLAGValidation",
    # Enums
    "TargetType",
    "AmbitionLevel",
    "PathwayType",
    "SectorPathway",
    "ValidationStatus",
    "ScopeType",
    "Scope3EngagementType",
    "NeutralizationType",
    "ProgressStatus",
    # Constants
    "SBTiPathwayConstants",
]
