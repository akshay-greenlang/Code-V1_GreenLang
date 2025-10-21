"""
GreenLang Scenario Spec v1 - SIM-401 Compliant Location

This module provides the ScenarioSpec v1 implementation as specified in SIM-401.
The actual implementation is in greenlang.specs.scenariospec_v1 for architectural
consistency with AgentSpec v2, but this wrapper provides the spec-compliant import path.

Usage:
    from greenlang.simulation.spec import ScenarioSpecV1, from_yaml, to_yaml

    # Load scenario
    spec = from_yaml("scenarios/baseline_sweep.yaml")

    # Validate and save
    to_yaml(spec, "output/validated.yaml")

Author: GreenLang Framework Team
Date: October 2025
Spec: SIM-401 (Scenario Spec & Seeded RNG)
"""

# Import all public APIs from the actual implementation
from greenlang.specs.scenariospec_v1 import (
    # Models
    ScenarioSpecV1,
    ParameterSpec,
    DistributionSpec,
    MonteCarloSpec,

    # Functions
    from_yaml,
    from_json,
    to_yaml,
    validate_spec,
    to_json_schema,

    # Re-export for convenience
    GLValidationError,
    GLVErr,
)

__all__ = [
    # Models
    "ScenarioSpecV1",
    "ParameterSpec",
    "DistributionSpec",
    "MonteCarloSpec",

    # Functions
    "from_yaml",
    "from_json",
    "to_yaml",
    "validate_spec",
    "to_json_schema",

    # Errors
    "GLValidationError",
    "GLVErr",
]
