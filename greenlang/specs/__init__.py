"""
GreenLang Specifications Module

This module contains schema definitions and validation logic for GreenLang specifications:
- AgentSpec v2: Schema for agent pack manifests (compute, AI, realtime, provenance)
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
    from_yaml,
    from_json,
    validate_spec,
    to_json_schema,
)

__all__ = [
    # Error handling
    "GLVErr",
    "GLValidationError",
    "raise_validation_error",
    # Models
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
    # Helper functions
    "from_yaml",
    "from_json",
    "validate_spec",
    "to_json_schema",
]
