# -*- coding: utf-8 -*-
"""
GreenLang Specifications Module

This module contains schema definitions and validation logic for GreenLang specifications:
- AgentSpec v2: Schema for agent pack manifests (compute, AI, realtime, provenance)
- ScenarioSpec v1: Schema for simulation scenarios (parameters, distributions, Monte Carlo)
- Validation errors: Structured error codes for spec validation

Author: GreenLang Framework Team
Date: October 2025
"""

from .errors import GLVErr, GLValidationError, raise_validation_error
from .agentspec_v2 import (
    # Models
    AgentSpecV2,
    ComputeSpec,
    AISpec,
    RealtimeSpec,
    ProvenanceSpec,
    IOField,
    OutputField,
    FactorRef,
    AIBudget,
    AITool,
    ConnectorRef,
    # Helper functions
    from_yaml as agent_from_yaml,
    from_json as agent_from_json,
    validate_spec as agent_validate_spec,
    to_json_schema as agent_to_json_schema,
)
from .scenariospec_v1 import (
    # Models
    ScenarioSpecV1,
    ParameterSpec,
    DistributionSpec,
    MonteCarloSpec,
    # Helper functions
    from_yaml as scenario_from_yaml,
    from_json as scenario_from_json,
    validate_spec as scenario_validate_spec,
    to_yaml as scenario_to_yaml,
    to_json_schema as scenario_to_json_schema,
)

__all__ = [
    # Error handling
    "GLVErr",
    "GLValidationError",
    "raise_validation_error",
    # AgentSpec v2
    "AgentSpecV2",
    "ComputeSpec",
    "AISpec",
    "RealtimeSpec",
    "ProvenanceSpec",
    "IOField",
    "OutputField",
    "FactorRef",
    "AIBudget",
    "AITool",
    "ConnectorRef",
    "agent_from_yaml",
    "agent_from_json",
    "agent_validate_spec",
    "agent_to_json_schema",
    # ScenarioSpec v1
    "ScenarioSpecV1",
    "ParameterSpec",
    "DistributionSpec",
    "MonteCarloSpec",
    "scenario_from_yaml",
    "scenario_from_json",
    "scenario_validate_spec",
    "scenario_to_yaml",
    "scenario_to_json_schema",
]
