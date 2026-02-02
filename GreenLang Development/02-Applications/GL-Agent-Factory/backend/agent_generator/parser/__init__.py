"""
Parser Module for Agent Generator

This module provides YAML parsing and validation capabilities for
transforming pack.yaml AgentSpec files into structured Python objects.

Components:
    - YAMLParser: Loads and parses pack.yaml files
    - SpecValidator: Validates AgentSpec against GreenLang requirements
    - Schema Models: Pydantic models for AgentSpec structure

Example:
    >>> from backend.agent_generator.parser import YAMLParser, SpecValidator
    >>> from pathlib import Path
    >>>
    >>> # Parse pack.yaml
    >>> parser = YAMLParser()
    >>> spec = parser.parse(Path("08-regulatory-agents/eudr/pack.yaml"))
    >>>
    >>> # Validate specification
    >>> validator = SpecValidator()
    >>> result = validator.validate(spec)
    >>> if not result.is_valid:
    ...     for error in result.errors:
    ...         print(f"Error: {error}")
"""

from typing import List

from .schema import (
    AgentDef,
    AgentSpec,
    AgentType,
    GoldenTestCategory,
    GoldenTestSpec,
    InputDef,
    MetadataRegulation,
    OutputDef,
    PackMeta,
    ToolConfig,
    ToolDef,
    ToolType,
    ValidationRule,
)
from .spec_validator import SpecValidator, ValidationResult
from .yaml_parser import YAMLParser, ParseError

__all__: List[str] = [
    # Parser
    "YAMLParser",
    "ParseError",
    # Validator
    "SpecValidator",
    "ValidationResult",
    # Schema Models
    "AgentSpec",
    "PackMeta",
    "AgentDef",
    "AgentType",
    "InputDef",
    "OutputDef",
    "ToolDef",
    "ToolType",
    "ToolConfig",
    "ValidationRule",
    "GoldenTestSpec",
    "GoldenTestCategory",
    "MetadataRegulation",
]
