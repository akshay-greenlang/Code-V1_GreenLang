"""
GL-004 BURNMASTER Orchestration Module

Multi-burner orchestration and coordination for combustion optimization.
Provides lead/lag sequencing, load balancing, safety coordination, and
inter-burner communication for multi-burner installations.

Author: GreenLang AI Agent Workforce
Version: 1.0.0
"""

from .multi_burner import (
    MultiBurnerOrchestrator,
    BurnerState,
    BurnerRole,
    SequencePhase,
    CoordinationStrategy,
    LoadBalancingStrategy,
    BurnerStatus,
    OrchestrationConfig,
    BurnerCommand,
    SequenceEvent,
    LoadDistribution,
    SafetyCoordinationStatus,
)

__all__ = [
    "MultiBurnerOrchestrator",
    "BurnerState",
    "BurnerRole",
    "SequencePhase",
    "CoordinationStrategy",
    "LoadBalancingStrategy",
    "BurnerStatus",
    "OrchestrationConfig",
    "BurnerCommand",
    "SequenceEvent",
    "LoadDistribution",
    "SafetyCoordinationStatus",
]
