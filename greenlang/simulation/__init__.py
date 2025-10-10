"""
GreenLang Simulation Module

Provides runtime execution for scenario-based simulations with
deterministic random number generation and provenance tracking.

Author: GreenLang Framework Team
Date: October 2025
Spec: SIM-401 (Scenario Spec & Seeded RNG)
"""

from .runner import ScenarioRunner

__all__ = ["ScenarioRunner"]
