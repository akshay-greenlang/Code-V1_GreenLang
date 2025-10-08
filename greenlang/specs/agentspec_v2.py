"""
GreenLang AgentSpec v2 - Pydantic Models

This module defines the authoritative Pydantic models for AgentSpec v2.
These models provide type-safe agent pack manifests with comprehensive validation.

Schema Sections:
- compute: Entrypoint, inputs/outputs, emission factors, determinism
- ai: LLM configuration, tools, RAG, budget constraints
- realtime: Replay/live modes, connector configuration
- provenance: Factor pinning, reproducibility, audit trails

Design Principles:
- Pydantic v2 as source of truth (JSON Schema is generated from these models)
- Strict validation (extra='forbid' catches typos)
- Determinism by default (compute.deterministic=true)
- Unit-aware (climate units validated against whitelist)
- Security-conscious (safe tool enforcement, URI validation)

Author: GreenLang Framework Team
Date: October 2025
Spec: FRMW-201 (AgentSpec v2 Schema + Validators)
CTO Approval: Required for all changes
"""

import ast
import inspect
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import jsonschema
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
    ValidationError,
)

from .errors import GLVErr, GLValidationError, raise_validation_error
from .safety import validate_safe_tool


# ============================================================================
# REGEX PATTERNS (Production-Ready)
# ============================================================================

# Semantic Versioning 2.0.0: https://semver.org/
SEMVER_RE = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)

# Agent ID slug: lowercase alphanumeric with separators (/, -, _)
# Example: "buildings/boiler_ng_v1"
SLUG_RE = re.compile(
    r"^[a-z0-9]+(?:[._-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)+$"
)

# Python URI: python://module.path:function_name
# Example: "python://gl.agents.boiler.ng:compute"
PYTHON_URI_RE = re.compile(
    r"^python://([a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)*):([a-z_][a-z0-9_]*)$",
    re.IGNORECASE
)

# Emission Factor URI: ef://authority/path/to/factor
# Example: "ef://ipcc_ar6/combustion/ng/co2e_kg_per_mj"
EF_URI_RE = re.compile(r"^ef://[a-z0-9_/-]+$", re.IGNORECASE)

# Simple unit syntax (enhanced validation below)
UNIT_RE = re.compile(r"^[A-Za-z0-9*/^ \-()]+$")


# ============================================================================
# CLIMATE UNITS WHITELIST
# ============================================================================
# Production-ready climate units for emissions, energy, and climate calculations.
# Based on:
# - GHG Protocol Corporate Standard
# - IPCC AR6 Guidelines
# - ISO 14064-1:2018
# - Common industry practice (CDP, TCFD, SBTi)

CLIMATE_UNITS = {
    # Dimensionless
    "1",
    "",  # Some systems use empty string for dimensionless

    # GHG Emissions
    "kgCO2e",
    "tCO2e",
    "MtCO2e",
    "GtCO2e",
    "kgCO2",
    "tCO2",
    "kgCH4",
    "tCH4",
    "kgN2O",
    "tN2O",

    # Energy
    "J",
    "kJ",
    "MJ",
    "GJ",
    "TJ",
    "Wh",
    "kWh",
    "MWh",
    "GWh",
    "TWh",
    "BTU",
    "kBTU",
    "MMBTU",
    "therm",

    # Power
    "W",
    "kW",
    "MW",
    "GW",

    # Mass
    "g",
    "kg",
    "t",
    "Mt",
    "Gt",
    "lb",
    "ton",  # US ton
    "tonne",  # Metric ton

    # Volume
    "L",
    "kL",
    "ML",
    "m3",
    "m^3",
    "gal",
    "ft3",
    "ft^3",

    # Area
    "m2",
    "m^2",
    "km2",
    "km^2",
    "ha",
    "ft2",
    "ft^2",

    # Distance
    "m",
    "km",
    "mi",
    "ft",

    # Temperature
    "K",
    "degC",
    "degF",

    # Pressure
    "Pa",
    "kPa",
    "MPa",
    "bar",
    "psi",

    # Time
    "s",
    "min",
    "h",
    "hr",
    "d",
    "day",
    "yr",
    "year",

    # Monetary
    "USD",
    "EUR",
    "GBP",

    # Intensity units (ratio units)
    "kgCO2e/kWh",
    "tCO2e/MWh",
    "kgCO2e/km",
    "tCO2e/t",
    "kgCO2e/m2",
    "kgCO2e/m^2",
    "kgCO2e/USD",
    "gCO2e/MJ",
    "kgCO2e/MJ",      # kilograms CO2e per megajoule (common emission intensity)
    "kgCO2e/GJ",
    "tCO2e/TJ",

    # Fuel heating values
    "MJ/kg",
    "MJ/L",
    "MJ/m3",
    "MJ/m^3",
    "kJ/kg",
    "BTU/lb",

    # Grid intensity
    "gCO2e/kWh",
    "kgCO2e/MWh",
    "lbCO2e/MWh",
}


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def validate_unit(unit: str, field_path: List[str]) -> str:
    """
    Validate unit string against climate units whitelist.

    Climate calculations require precise unit handling. This validator ensures:
    1. Unit syntax is valid (parseable)
    2. Unit is in approved whitelist (prevents typos like "kgg" → "kg")
    3. Unit is appropriate for climate/energy/emissions calculations

    Args:
        unit: Unit string to validate
        field_path: Field path for error reporting

    Returns:
        Validated unit string (normalized)

    Raises:
        GLValidationError: If unit is invalid or not in whitelist
    """
    # Check basic syntax
    if not UNIT_RE.match(unit):
        raise GLValidationError(
            GLVErr.UNIT_SYNTAX,
            f"Unit '{unit}' has invalid syntax. Expected format: 'kg', 'kWh', 'kgCO2e/MWh', etc.",
            field_path
        )

    # Check whitelist (case-sensitive for scientific accuracy)
    if unit not in CLIMATE_UNITS:
        raise GLValidationError(
            GLVErr.UNIT_SYNTAX,
            f"Unit '{unit}' is not in approved climate units whitelist. "
            f"Common units: kgCO2e, tCO2e, kWh, MWh, GJ, m3. "
            f"If this is a valid unit, contact the GreenLang team to add it.",
            field_path
        )

    return unit


def validate_python_uri(uri: str, field_path: List[str]) -> str:
    """
    Validate Python URI format and prevent security issues.

    Security checks:
    1. Format: python://module.path:function_name
    2. No path traversal (../)
    3. No absolute paths (/etc/passwd)
    4. Valid Python identifiers

    Args:
        uri: Python URI to validate
        field_path: Field path for error reporting

    Returns:
        Validated URI string

    Raises:
        GLValidationError: If URI is malformed or insecure
    """
    if not PYTHON_URI_RE.match(uri):
        raise GLValidationError(
            GLVErr.INVALID_URI,
            f"Invalid python:// URI: '{uri}'. "
            f"Expected format: 'python://module.path:function_name'. "
            f"Example: 'python://gl.agents.boiler.ng:compute'",
            field_path
        )

    # Security: Check for path traversal
    if ".." in uri or uri.startswith("/"):
        raise GLValidationError(
            GLVErr.INVALID_URI,
            f"Security violation: URI contains path traversal or absolute path: '{uri}'",
            field_path
        )

    return uri


def validate_ef_uri(uri: str, field_path: List[str]) -> str:
    """
    Validate Emission Factor URI format.

    Args:
        uri: Emission factor URI to validate
        field_path: Field path for error reporting

    Returns:
        Validated URI string

    Raises:
        GLValidationError: If URI is malformed
    """
    if not EF_URI_RE.match(uri):
        raise GLValidationError(
            GLVErr.INVALID_URI,
            f"Invalid ef:// URI: '{uri}'. "
            f"Expected format: 'ef://authority/path/to/factor'. "
            f"Example: 'ef://ipcc_ar6/combustion/ng/co2e_kg_per_mj'",
            field_path
        )

    return uri


# AST safety validation is now in greenlang/specs/safety.py
# Imported at top of file: from .safety import validate_safe_tool


# ============================================================================
# PYDANTIC MODELS - Leaf Nodes
# ============================================================================

class IOField(BaseModel):
    """
    Input field specification for compute section.

    Defines a single input parameter with:
    - Data type (float32, float64, int32, int64, string, bool)
    - Physical unit (validated against climate units whitelist)
    - Required flag (default true)
    - Optional constraints (ge, gt, le, lt, enum)
    - Optional default value

    Example:
        >>> IOField(
        ...     dtype="float64",
        ...     unit="m^3",
        ...     required=True,
        ...     ge=0,
        ...     description="Natural gas volume consumed"
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    dtype: Literal["float32", "float64", "int32", "int64", "string", "bool"] = Field(
        ...,
        description="Data type of input parameter"
    )
    unit: str = Field(
        ...,
        description="Physical unit (validated against climate units whitelist). Use '1' for dimensionless."
    )
    required: bool = Field(
        default=True,
        description="Whether input is required (default: true)"
    )
    default: Optional[Any] = Field(
        default=None,
        description="Default value if input not provided (only valid if required=false)"
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description of input parameter"
    )

    # Numeric constraints
    ge: Optional[float] = Field(default=None, description="Greater than or equal (≥)")
    gt: Optional[float] = Field(default=None, description="Greater than (>)")
    le: Optional[float] = Field(default=None, description="Less than or equal (≤)")
    lt: Optional[float] = Field(default=None, description="Less than (<)")

    # Enum constraint
    enum: Optional[List[Any]] = Field(
        default=None,
        description="Allowed values (enum constraint)"
    )

    @field_validator("unit")
    @classmethod
    def validate_unit_field(cls, v: str, info) -> str:
        """Validate unit against climate units whitelist."""
        field_path = ["compute", "inputs", "(field)", "unit"]
        return validate_unit(v, field_path)

    @field_validator("dtype")
    @classmethod
    def validate_unit_for_string_bool(cls, v: str, info) -> str:
        """Ensure string/bool types use dimensionless unit."""
        # This will be checked in model_validator
        return v

    @model_validator(mode="after")
    def check_unit_for_dtype(self):
        """Enforce: string/bool inputs must have unit='1' (dimensionless)."""
        if self.dtype in ("string", "bool") and self.unit not in ("1", ""):
            raise GLValidationError(
                GLVErr.UNIT_FORBIDDEN,
                f"String and bool inputs must use dimensionless unit '1', got '{self.unit}'",
                ["compute", "inputs", "(field)"]
            )
        return self

    @model_validator(mode="after")
    def check_default_requires_optional(self):
        """Enforce: default value only valid if required=false."""
        if self.default is not None and self.required:
            raise GLValidationError(
                GLVErr.CONSTRAINT,
                "default value can only be set if required=false",
                ["compute", "inputs", "(field)"]
            )
        return self


class OutputField(BaseModel):
    """
    Output field specification for compute section.

    Simpler than IOField: only dtype and unit (outputs are always required).

    Example:
        >>> OutputField(
        ...     dtype="float64",
        ...     unit="kgCO2e",
        ...     description="Total CO2 equivalent emissions"
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    dtype: Literal["float32", "float64", "int32", "int64", "string", "bool"] = Field(
        ...,
        description="Data type of output parameter"
    )
    unit: str = Field(
        ...,
        description="Physical unit (validated against climate units whitelist)"
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description of output parameter"
    )

    @field_validator("unit")
    @classmethod
    def validate_unit_field(cls, v: str, info) -> str:
        """Validate unit against climate units whitelist."""
        field_path = ["compute", "outputs", "(field)", "unit"]
        return validate_unit(v, field_path)


class FactorRef(BaseModel):
    """
    Emission factor reference for compute section.

    Points to an emission factor in the factor registry using ef:// URI.
    Optionally specifies GWP set (AR6GWP100, AR5GWP100, etc.).

    Example:
        >>> FactorRef(
        ...     ref="ef://ipcc_ar6/combustion/ng/co2e_kg_per_mj",
        ...     gwp_set="AR6GWP100",
        ...     description="Natural gas combustion emission factor (IPCC AR6)"
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    ref: str = Field(
        ...,
        description="Emission factor URI (ef:// scheme)"
    )
    gwp_set: Optional[str] = Field(
        default=None,
        description="GWP set for CH4/N2O conversion (AR6GWP100, AR5GWP100, SAR, etc.)"
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description of emission factor"
    )

    @field_validator("ref")
    @classmethod
    def validate_ef_uri_field(cls, v: str, info) -> str:
        """Validate ef:// URI format."""
        field_path = ["compute", "factors", "(field)", "ref"]
        return validate_ef_uri(v, field_path)


class AIBudget(BaseModel):
    """
    AI cost and token budget constraints.

    Prevents runaway LLM costs in production agents.
    All constraints are optional (None = unlimited, use with caution!).

    Example:
        >>> AIBudget(
        ...     max_cost_usd=1.00,
        ...     max_input_tokens=15000,
        ...     max_output_tokens=2000,
        ...     max_retries=3
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    max_cost_usd: Optional[float] = Field(
        default=None,
        ge=0,
        description="Maximum USD cost for LLM calls (cumulative per agent run)"
    )
    max_input_tokens: Optional[int] = Field(
        default=None,
        ge=0,
        description="Maximum input tokens (cumulative)"
    )
    max_output_tokens: Optional[int] = Field(
        default=None,
        ge=0,
        description="Maximum output tokens (cumulative)"
    )
    max_retries: Optional[int] = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retries for failed LLM calls (default: 3)"
    )


class AITool(BaseModel):
    """
    AI tool specification for LLM function calling.

    Defines a tool that the LLM can call during agent execution.
    Tools have input/output schemas (JSON Schema draft-2020-12) and Python implementations.

    Security: Tools marked 'safe=true' undergo AST analysis to prevent unsafe operations.

    Example:
        >>> AITool(
        ...     name="select_emission_factor",
        ...     description="Select appropriate emission factor based on region and year",
        ...     schema_in={
        ...         "type": "object",
        ...         "properties": {
        ...             "region": {"type": "string"},
        ...             "year": {"type": "integer"}
        ...         },
        ...         "required": ["region", "year"]
        ...     },
        ...     schema_out={
        ...         "type": "object",
        ...         "properties": {
        ...             "ef_uri": {"type": "string", "pattern": "^ef://"}
        ...         },
        ...         "required": ["ef_uri"]
        ...     },
        ...     impl="python://gl.ai.tools.ef:select",
        ...     safe=True
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$",
        description="Tool name (valid Python identifier)"
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description of tool purpose"
    )
    schema_in: Dict[str, Any] = Field(
        ...,
        description="JSON Schema (draft-2020-12) for tool input parameters"
    )
    schema_out: Dict[str, Any] = Field(
        ...,
        description="JSON Schema (draft-2020-12) for tool output"
    )
    impl: str = Field(
        ...,
        description="Python URI for tool implementation (python://module:function)"
    )
    safe: bool = Field(
        default=True,
        description="Whether tool is safe (pure function, no side effects). Enforced via AST analysis."
    )

    @field_validator("impl")
    @classmethod
    def validate_impl_uri(cls, v: str, info) -> str:
        """Validate python:// URI format."""
        field_path = ["ai", "tools", "(tool)", "impl"]
        return validate_python_uri(v, field_path)

    @model_validator(mode="after")
    def validate_safe_tool_impl(self):
        """If safe=true, validate tool implementation is actually safe."""
        if self.safe:
            validate_safe_tool(self.impl, self.name)
        return self

    @model_validator(mode="after")
    def validate_json_schemas(self):
        """Validate schema_in and schema_out are valid JSON Schema draft-2020-12."""
        # Validate schema_in
        try:
            jsonschema.Draft202012Validator.check_schema(self.schema_in)
        except jsonschema.SchemaError as e:
            raise GLValidationError(
                GLVErr.AI_SCHEMA_INVALID,
                f"Tool '{self.name}' has invalid schema_in: {e.message}",
                ["ai", "tools", self.name, "schema_in"]
            )

        # Validate schema_out
        try:
            jsonschema.Draft202012Validator.check_schema(self.schema_out)
        except jsonschema.SchemaError as e:
            raise GLValidationError(
                GLVErr.AI_SCHEMA_INVALID,
                f"Tool '{self.name}' has invalid schema_out: {e.message}",
                ["ai", "tools", self.name, "schema_out"]
            )

        return self


class ConnectorRef(BaseModel):
    """
    Realtime connector reference for live data streams.

    Connectors fetch external data (grid intensity, weather, commodity prices).
    Only active in 'live' mode; 'replay' mode uses cached snapshots.

    Example:
        >>> ConnectorRef(
        ...     name="grid_intensity",
        ...     topic="region_hourly_ci",
        ...     window="1h",
        ...     ttl="6h",
        ...     required=False
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        description="Connector name (must match registered connector)"
    )
    topic: str = Field(
        ...,
        description="Data topic/stream identifier"
    )
    window: Optional[str] = Field(
        default=None,
        description="Time window for data aggregation (e.g., '1h', '15min')"
    )
    ttl: Optional[str] = Field(
        default=None,
        description="Time-to-live for cached data (e.g., '6h', '1d')"
    )
    required: bool = Field(
        default=False,
        description="Whether connector is required (agent fails if unavailable)"
    )


# ============================================================================
# PYDANTIC MODELS - Section Specs
# ============================================================================

class ComputeSpec(BaseModel):
    """
    Compute section: entrypoint, inputs/outputs, factors, determinism.

    Defines the core computational logic of the agent:
    - Python entrypoint (function to execute)
    - Input parameters with types, units, constraints
    - Output parameters with types, units
    - Emission factors (optional)
    - Determinism flag (default: true)

    Example:
        >>> ComputeSpec(
        ...     entrypoint="python://gl.agents.boiler.ng:compute",
        ...     deterministic=True,
        ...     inputs={
        ...         "fuel_volume": IOField(dtype="float64", unit="m^3", required=True, ge=0),
        ...         "efficiency": IOField(dtype="float64", unit="1", required=True, gt=0, le=1)
        ...     },
        ...     outputs={
        ...         "co2e_kg": OutputField(dtype="float64", unit="kgCO2e")
        ...     },
        ...     factors={
        ...         "co2e_factor": FactorRef(ref="ef://ipcc_ar6/combustion/ng/co2e_kg_per_mj")
        ...     }
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    entrypoint: str = Field(
        ...,
        description="Python URI for compute entrypoint (python://module:function)"
    )
    deterministic: bool = Field(
        default=True,
        description="Whether computation is deterministic (same inputs → same outputs). Default: true."
    )
    inputs: Dict[str, IOField] = Field(
        ...,
        description="Input parameters (name → IOField spec)"
    )
    outputs: Dict[str, OutputField] = Field(
        ...,
        description="Output parameters (name → OutputField spec)"
    )
    factors: Dict[str, FactorRef] = Field(
        default_factory=dict,
        description="Emission factor references (optional)"
    )

    # P0 enhancements (added Day 4)
    dependencies: Optional[List[str]] = Field(
        default=None,
        description="Python package dependencies with versions (e.g., ['pandas==2.1.4', 'numpy==1.26.0'])"
    )
    python_version: Optional[str] = Field(
        default=None,
        description="Required Python version (e.g., '3.11', '3.11.5')"
    )
    timeout_s: int = Field(
        default=30,
        ge=1,
        le=3600,
        description="Maximum execution time in seconds (default: 30s, max: 1 hour)"
    )
    memory_limit_mb: int = Field(
        default=512,
        ge=128,
        le=16384,
        description="Maximum memory usage in MB (default: 512MB, max: 16GB)"
    )

    @field_validator("entrypoint")
    @classmethod
    def validate_entrypoint_uri(cls, v: str, info) -> str:
        """Validate python:// URI format for entrypoint."""
        field_path = ["compute", "entrypoint"]
        return validate_python_uri(v, field_path)

    @model_validator(mode="after")
    def warn_non_deterministic(self):
        """Warn if deterministic=false (non-determinism should be rare)."""
        if not self.deterministic:
            # TODO: Add warning logging (non-blocking)
            # For now, we allow it but document that it's discouraged
            pass
        return self


class AISpec(BaseModel):
    """
    AI section: LLM configuration, tools, RAG, budget.

    Defines how the agent uses AI capabilities:
    - JSON mode (structured output)
    - System prompt (instructions for LLM)
    - Budget constraints (cost, tokens, retries)
    - RAG collections (document retrieval)
    - Tools (function calling)

    Example:
        >>> AISpec(
        ...     json_mode=True,
        ...     system_prompt="You are a climate advisor. Use tools; never guess numbers.",
        ...     budget=AIBudget(max_cost_usd=1.00, max_input_tokens=15000),
        ...     rag_collections=["ghg_protocol_corp", "ipcc_ar6"],
        ...     tools=[
        ...         AITool(
        ...             name="select_emission_factor",
        ...             schema_in={...},
        ...             schema_out={...},
        ...             impl="python://gl.ai.tools.ef:select"
        ...         )
        ...     ]
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    json_mode: bool = Field(
        default=True,
        description="Whether to use JSON mode for structured output (default: true)"
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="System prompt for LLM (instructions, context, constraints)"
    )
    budget: Optional[AIBudget] = Field(
        default=None,
        description="Cost and token budget constraints"
    )
    rag_collections: List[str] = Field(
        default_factory=list,
        description="RAG collection names for document retrieval"
    )
    tools: List[AITool] = Field(
        default_factory=list,
        description="AI tools for function calling"
    )

    @model_validator(mode="after")
    def validate_tool_names_unique(self):
        """Ensure tool names are unique within AI spec."""
        tool_names = [tool.name for tool in self.tools]
        if len(tool_names) != len(set(tool_names)):
            duplicates = [name for name in tool_names if tool_names.count(name) > 1]
            raise GLValidationError(
                GLVErr.DUPLICATE_NAME,
                f"Duplicate tool names: {duplicates}",
                ["ai", "tools"]
            )
        return self


class RealtimeSpec(BaseModel):
    """
    Realtime section: replay/live modes, connector configuration.

    Controls how agent handles external data:
    - replay: Use cached snapshots (deterministic, auditable)
    - live: Fetch fresh data from connectors (non-deterministic)

    Example:
        >>> RealtimeSpec(
        ...     default_mode="replay",
        ...     snapshot_path="snapshots/2024-10-06_grid_intensity.json",
        ...     connectors=[
        ...         ConnectorRef(
        ...             name="grid_intensity",
        ...             topic="region_hourly_ci",
        ...             window="1h",
        ...             required=False
        ...         )
        ...     ]
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    default_mode: Literal["replay", "live"] = Field(
        default="replay",
        description="Default realtime mode (replay=cached, live=fresh data). Default: replay."
    )
    snapshot_path: Optional[str] = Field(
        default=None,
        description="Path to cached data snapshot for replay mode (P1 enhancement)"
    )
    connectors: List[ConnectorRef] = Field(
        default_factory=list,
        description="Realtime connector configurations"
    )

    @model_validator(mode="after")
    def validate_live_mode_has_connectors(self):
        """If default_mode='live', require at least one connector."""
        if self.default_mode == "live" and not self.connectors:
            raise GLValidationError(
                GLVErr.MODE_INVALID,
                "Realtime mode 'live' requires at least one connector",
                ["realtime", "default_mode"]
            )
        return self

    @model_validator(mode="after")
    def validate_connector_names_unique(self):
        """Ensure connector names are unique within realtime spec."""
        connector_names = [conn.name for conn in self.connectors]
        if len(connector_names) != len(set(connector_names)):
            duplicates = [name for name in connector_names if connector_names.count(name) > 1]
            raise GLValidationError(
                GLVErr.DUPLICATE_NAME,
                f"Duplicate connector names: {duplicates}",
                ["realtime", "connectors"]
            )
        return self


class ProvenanceSpec(BaseModel):
    """
    Provenance section: factor pinning, reproducibility, audit trails.

    Ensures agent runs are reproducible and auditable:
    - pin_ef: Whether to pin emission factor versions (default: true)
    - gwp_set: GWP set for CH4/N2O conversion (AR6GWP100, etc.)
    - record: Fields to include in provenance hash

    Example:
        >>> ProvenanceSpec(
        ...     pin_ef=True,
        ...     gwp_set="AR6GWP100",
        ...     record=["inputs", "outputs", "factors", "ef_uri", "ef_cid", "code_sha", "seed"]
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    pin_ef: bool = Field(
        default=True,
        description="Whether to pin emission factor versions (default: true)"
    )
    gwp_set: Optional[str] = Field(
        default="AR6GWP100",
        description="GWP set for GHG conversions (AR6GWP100, AR5GWP100, SAR, etc.). Default: AR6GWP100 (P0 enhancement)"
    )
    record: List[str] = Field(
        ...,
        description="Fields to include in provenance record (e.g., inputs, outputs, factors, code_sha, seed)"
    )

    @field_validator("record")
    @classmethod
    def validate_record_unique(cls, v: List[str], info) -> List[str]:
        """Ensure record fields are unique."""
        if len(v) != len(set(v)):
            duplicates = [field for field in v if v.count(field) > 1]
            raise GLValidationError(
                GLVErr.DUPLICATE_NAME,
                f"Duplicate provenance record fields: {duplicates}",
                ["provenance", "record"]
            )
        return v


# ============================================================================
# PYDANTIC MODELS - Top-Level AgentSpec v2
# ============================================================================

class AgentSpecV2(BaseModel):
    """
    GreenLang AgentSpec v2 - Top-Level Schema

    This is the authoritative specification for GreenLang agent packs.
    Every agent pack MUST conform to this schema.

    Schema Sections:
    - Metadata: schema_version, id, name, version, summary, tags, owners, license
    - compute: Computational logic (entrypoint, inputs/outputs, factors)
    - ai: AI capabilities (LLM, tools, RAG, budget)
    - realtime: Data streaming (replay/live, connectors)
    - provenance: Reproducibility and audit (factor pinning, record fields)
    - tests: Golden tests and property-based tests (optional)

    Example:
        >>> spec = AgentSpecV2(
        ...     schema_version="2.0.0",
        ...     id="buildings/boiler_ng_v1",
        ...     name="Boiler – Natural Gas (LHV)",
        ...     version="2.1.3",
        ...     summary="Computes CO2e from NG boiler fuel using LHV.",
        ...     compute=ComputeSpec(...),
        ...     ai=AISpec(...),
        ...     realtime=RealtimeSpec(...),
        ...     provenance=ProvenanceSpec(...)
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    # Metadata
    schema_version: Literal["2.0.0"] = Field(
        ...,
        description="AgentSpec schema version (MUST be '2.0.0')"
    )
    id: str = Field(
        ...,
        description="Agent ID slug (e.g., 'buildings/boiler_ng_v1'). Format: segment/segment/..."
    )
    name: str = Field(
        ...,
        min_length=3,
        description="Human-readable agent name"
    )
    version: str = Field(
        ...,
        description="Agent version (semantic versioning 2.0.0)"
    )
    summary: Optional[str] = Field(
        default=None,
        description="Short description of agent purpose"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for categorization and search"
    )
    owners: Optional[List[str]] = Field(
        default=None,
        description="Agent owners (e.g., ['@gl/industry-buildings'])"
    )
    license: Optional[str] = Field(
        default=None,
        description="License identifier (e.g., 'Apache-2.0', 'MIT')"
    )

    # Core sections (required)
    compute: ComputeSpec = Field(
        ...,
        description="Compute specification (entrypoint, inputs/outputs, factors)"
    )
    ai: AISpec = Field(
        ...,
        description="AI specification (LLM, tools, RAG, budget)"
    )
    realtime: RealtimeSpec = Field(
        ...,
        description="Realtime specification (replay/live, connectors)"
    )
    provenance: ProvenanceSpec = Field(
        ...,
        description="Provenance specification (factor pinning, audit trails)"
    )

    # Optional sections
    security: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Security configuration (P1 enhancement: allowlist_hosts, etc.)"
    )
    tests: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Test configuration (golden tests, property-based tests)"
    )

    @field_validator("id")
    @classmethod
    def validate_id_slug(cls, v: str, info) -> str:
        """Validate agent ID slug format."""
        if not SLUG_RE.match(v):
            raise GLValidationError(
                GLVErr.INVALID_SLUG,
                f"Invalid agent ID slug: '{v}'. "
                f"Expected format: 'segment/segment/...' with lowercase alphanumeric and separators (-, _). "
                f"Example: 'buildings/boiler_ng_v1'",
                ["id"]
            )
        return v

    @field_validator("version")
    @classmethod
    def validate_version_semver(cls, v: str, info) -> str:
        """Validate version conforms to Semantic Versioning 2.0.0."""
        if not SEMVER_RE.match(v):
            raise GLValidationError(
                GLVErr.INVALID_SEMVER,
                f"Invalid semantic version: '{v}'. "
                f"Expected format: MAJOR.MINOR.PATCH (e.g., '2.1.3'). "
                f"See https://semver.org/ for specification.",
                ["version"]
            )
        return v

    @field_validator("tags")
    @classmethod
    def validate_tags_unique(cls, v: List[str], info) -> List[str]:
        """Ensure tags are unique."""
        if len(v) != len(set(v)):
            duplicates = [tag for tag in v if v.count(tag) > 1]
            raise GLValidationError(
                GLVErr.DUPLICATE_NAME,
                f"Duplicate tags: {duplicates}",
                ["tags"]
            )
        return v

    @model_validator(mode="after")
    def validate_pin_ef_requires_factors(self):
        """
        CRITICAL BLOCKER: If provenance.pin_ef=true, require at least one factor.

        This is the key compliance validator from the expert review.
        Climate audits demand: "pinned factors" means factors actually exist.
        """
        if self.provenance.pin_ef and not self.compute.factors:
            raise GLValidationError(
                GLVErr.PROVENANCE_INVALID,
                "provenance.pin_ef=true requires at least one emission factor in compute.factors",
                ["provenance", "pin_ef"]
            )
        return self

    @model_validator(mode="after")
    def validate_no_duplicate_names_across_namespaces(self):
        """
        Ensure no duplicate names across all namespaces:
        - compute.inputs
        - compute.outputs
        - compute.factors
        - ai.tools
        - realtime.connectors

        This prevents namespace collisions in generated code and runtime.
        """
        all_names = []
        all_names.extend(self.compute.inputs.keys())
        all_names.extend(self.compute.outputs.keys())
        all_names.extend(self.compute.factors.keys())
        all_names.extend([tool.name for tool in self.ai.tools])
        all_names.extend([conn.name for conn in self.realtime.connectors])

        if len(all_names) != len(set(all_names)):
            duplicates = [name for name in all_names if all_names.count(name) > 1]
            raise GLValidationError(
                GLVErr.DUPLICATE_NAME,
                f"Duplicate names across namespaces (inputs/outputs/factors/tools/connectors): {duplicates}",
                ["(global)"]
            )
        return self


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def from_yaml(path: Union[str, Path]) -> AgentSpecV2:
    """
    Load AgentSpec v2 from YAML file.

    Args:
        path: Path to pack.yaml file

    Returns:
        Validated AgentSpecV2 instance

    Raises:
        GLValidationError: If spec is invalid
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML is malformed
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"AgentSpec file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    try:
        return AgentSpecV2.model_validate(data)
    except ValidationError as e:
        gl_errors = GLValidationError.from_pydantic(e, context=str(path))
        # Raise first error (caller can catch and iterate through all if needed)
        raise gl_errors[0] if gl_errors else e


def from_json(path: Union[str, Path]) -> AgentSpecV2:
    """
    Load AgentSpec v2 from JSON file.

    Args:
        path: Path to pack.json file

    Returns:
        Validated AgentSpecV2 instance

    Raises:
        GLValidationError: If spec is invalid
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON is malformed
    """
    import json

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"AgentSpec file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        return AgentSpecV2.model_validate(data)
    except ValidationError as e:
        gl_errors = GLValidationError.from_pydantic(e, context=str(path))
        raise gl_errors[0] if gl_errors else e


def validate_spec(data: dict) -> AgentSpecV2:
    """
    Validate AgentSpec v2 from dictionary.

    Args:
        data: AgentSpec data as dictionary

    Returns:
        Validated AgentSpecV2 instance

    Raises:
        GLValidationError: If spec is invalid
    """
    try:
        return AgentSpecV2.model_validate(data)
    except ValidationError as e:
        gl_errors = GLValidationError.from_pydantic(e)
        raise gl_errors[0] if gl_errors else e


def to_json_schema() -> dict:
    """
    Export AgentSpec v2 as JSON Schema (draft-2020-12).

    This is generated from the Pydantic models (Pydantic is source of truth).
    Used by:
    - CLI validator (gl spec validate)
    - Documentation generation
    - External tooling (VS Code, CI checks)

    Returns:
        JSON Schema dictionary
    """
    schema = AgentSpecV2.model_json_schema(mode="serialization")

    # Add JSON Schema metadata
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://greenlang.io/specs/agentspec_v2.json"
    schema["title"] = "GreenLang AgentSpec v2"

    return schema
