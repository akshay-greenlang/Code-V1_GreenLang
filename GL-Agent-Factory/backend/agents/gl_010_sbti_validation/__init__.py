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
"""

from .agent import (
    SBTiValidationAgent,
    SBTiInput,
    SBTiOutput,
    TargetType,
    AmbitionLevel,
    PathwayType,
    SectorPathway,
    ScopeEmissions,
    TargetDefinition,
    ValidationResult,
    ProgressTracking,
    Recommendation,
)

__all__ = [
    "SBTiValidationAgent",
    "SBTiInput",
    "SBTiOutput",
    "TargetType",
    "AmbitionLevel",
    "PathwayType",
    "SectorPathway",
    "ScopeEmissions",
    "TargetDefinition",
    "ValidationResult",
    "ProgressTracking",
    "Recommendation",
]
