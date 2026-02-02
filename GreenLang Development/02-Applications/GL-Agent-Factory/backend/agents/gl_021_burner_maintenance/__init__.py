"""
GL-021 BURNERSENTRY - Burner Maintenance Predictor Agent

This module provides the BurnerMaintenancePredictorAgent for predictive
maintenance of industrial burners using reliability engineering methods.

The agent implements Weibull failure analysis, flame quality scoring,
and health-based maintenance scheduling following zero-hallucination
principles with deterministic calculations only.

Agent ID: GL-021
Agent Name: BURNERSENTRY
Version: 1.0.0

Example:
    >>> from gl_021_burner_maintenance import BurnerMaintenancePredictorAgent
    >>> agent = BurnerMaintenancePredictorAgent(config)
    >>> result = agent.run(input_data)
"""

from .agent import (
    BurnerMaintenancePredictorAgent,
    BurnerInput,
    BurnerOutput,
    BurnerComponentHealth,
    FlameQualityMetrics,
    MaintenanceRecommendation,
    WeibullParameters,
)

# Agent metadata
AGENT_ID = "GL-021"
AGENT_NAME = "BURNERSENTRY"
VERSION = "1.0.0"
DESCRIPTION = "Burner Maintenance Predictor Agent using Weibull reliability analysis"

__all__ = [
    # Agent class
    "BurnerMaintenancePredictorAgent",
    # Input/Output models
    "BurnerInput",
    "BurnerOutput",
    # Supporting models
    "BurnerComponentHealth",
    "FlameQualityMetrics",
    "MaintenanceRecommendation",
    "WeibullParameters",
    # Metadata
    "AGENT_ID",
    "AGENT_NAME",
    "VERSION",
    "DESCRIPTION",
]
