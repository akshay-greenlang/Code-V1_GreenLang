"""
GreenLang ScenarioSpec v1 - Pydantic Models

This module defines the authoritative Pydantic models for ScenarioSpec v1.
Scenario specs enable parameter sweeps, Monte Carlo simulations, and
deterministic scenario analysis for climate modeling workflows.

Schema Sections:
- metadata: schema_version, name, description, seed
- parameters: Parameter value specifications (fixed, range, list)
- monte_carlo: Monte Carlo configuration
- metadata: Owner, tags, additional info

Design Principles:
- Pydantic v2 as source of truth (JSON Schema generated from models)
- Strict validation (extra='forbid' catches typos)
- Deterministic by default (fixed seed)
- Security-conscious (seed range validation, distribution param checks)
- Follows GreenLang patterns (GLValidationError, stable versioning)

Author: GreenLang Framework Team
Date: October 2025
Spec: SIM-401 (Scenario Spec & Seeded RNG)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
    ValidationError,
)

from .errors import GLVErr, GLValidationError


# ============================================================================
# PYDANTIC MODELS - Parameter Specifications
# ============================================================================


class ParameterSpec(BaseModel):
    """
    Parameter specification with type and optional constraints.

    Supports sweep types (list of discrete values) and stochastic types
    (sampled from distributions in Monte Carlo trials).
    """
    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        ...,
        pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$",
        description="Parameter identifier (valid Python identifier)"
    )
    type: Literal["sweep", "distribution"] = Field(
        ...,
        description="Parameter type: sweep (deterministic grid) or distribution (stochastic)"
    )

    # Sweep parameters
    values: Optional[List[Union[float, int, str]]] = Field(
        default=None,
        description="Discrete values for sweep parameters"
    )

    # Distribution parameters
    distribution: Optional[DistributionSpec] = Field(
        default=None,
        description="Distribution specification for stochastic parameters"
    )

    @model_validator(mode="after")
    def validate_type_consistency(self) -> ParameterSpec:
        """Ensure type matches provided fields."""
        if self.type == "sweep":
            if self.values is None:
                raise GLValidationError(
                    GLVErr.CONSTRAINT,
                    f"Parameter '{self.id}' with type='sweep' must have 'values' field",
                    ["parameters", self.id, "values"]
                )
            if self.distribution is not None:
                raise GLValidationError(
                    GLVErr.CONSTRAINT,
                    f"Parameter '{self.id}' with type='sweep' cannot have 'distribution' field",
                    ["parameters", self.id, "distribution"]
                )
            if len(self.values) == 0:
                raise GLValidationError(
                    GLVErr.CONSTRAINT,
                    f"Parameter '{self.id}' sweep values cannot be empty",
                    ["parameters", self.id, "values"]
                )
            if len(self.values) > 100000:
                raise GLValidationError(
                    GLVErr.CONSTRAINT,
                    f"Parameter '{self.id}' sweep has too many values ({len(self.values)}), max 100k",
                    ["parameters", self.id, "values"]
                )

        elif self.type == "distribution":
            if self.distribution is None:
                raise GLValidationError(
                    GLVErr.CONSTRAINT,
                    f"Parameter '{self.id}' with type='distribution' must have 'distribution' field",
                    ["parameters", self.id, "distribution"]
                )
            if self.values is not None:
                raise GLValidationError(
                    GLVErr.CONSTRAINT,
                    f"Parameter '{self.id}' with type='distribution' cannot have 'values' field",
                    ["parameters", self.id, "values"]
                )

        return self


class DistributionSpec(BaseModel):
    """
    Distribution specification for stochastic parameters.

    Supports common distributions: uniform, normal, lognormal, triangular.
    All parameters are validated for mathematical validity.
    """
    model_config = ConfigDict(extra="forbid")

    kind: Literal["uniform", "normal", "lognormal", "triangular"] = Field(
        ...,
        description="Distribution type"
    )

    # Uniform distribution
    low: Optional[float] = Field(default=None, description="Lower bound (uniform, triangular)")
    high: Optional[float] = Field(default=None, description="Upper bound (uniform, triangular)")

    # Normal distribution
    mean: Optional[float] = Field(default=None, description="Mean (normal, lognormal)")
    std: Optional[float] = Field(default=None, description="Standard deviation (normal)")
    sigma: Optional[float] = Field(default=None, description="Log-space std (lognormal)")

    # Triangular distribution
    mode: Optional[float] = Field(default=None, description="Mode (triangular)")

    @model_validator(mode="after")
    def validate_distribution_params(self) -> DistributionSpec:
        """Validate distribution parameters based on kind."""
        if self.kind == "uniform":
            if self.low is None or self.high is None:
                raise GLValidationError(
                    GLVErr.CONSTRAINT,
                    "Uniform distribution requires 'low' and 'high' parameters",
                    ["distribution"]
                )
            if self.low >= self.high:
                raise GLValidationError(
                    GLVErr.CONSTRAINT,
                    f"Uniform distribution: low ({self.low}) must be < high ({self.high})",
                    ["distribution", "low"]
                )

        elif self.kind == "normal":
            if self.mean is None or self.std is None:
                raise GLValidationError(
                    GLVErr.CONSTRAINT,
                    "Normal distribution requires 'mean' and 'std' parameters",
                    ["distribution"]
                )
            if self.std <= 0:
                raise GLValidationError(
                    GLVErr.CONSTRAINT,
                    f"Normal distribution: std must be positive, got {self.std}",
                    ["distribution", "std"]
                )

        elif self.kind == "lognormal":
            if self.mean is None or self.sigma is None:
                raise GLValidationError(
                    GLVErr.CONSTRAINT,
                    "Lognormal distribution requires 'mean' and 'sigma' parameters",
                    ["distribution"]
                )
            if self.sigma <= 0:
                raise GLValidationError(
                    GLVErr.CONSTRAINT,
                    f"Lognormal distribution: sigma must be positive, got {self.sigma}",
                    ["distribution", "sigma"]
                )

        elif self.kind == "triangular":
            if self.low is None or self.mode is None or self.high is None:
                raise GLValidationError(
                    GLVErr.CONSTRAINT,
                    "Triangular distribution requires 'low', 'mode', and 'high' parameters",
                    ["distribution"]
                )
            if not (self.low <= self.mode <= self.high):
                raise GLValidationError(
                    GLVErr.CONSTRAINT,
                    f"Triangular distribution: must have low ({self.low}) ≤ mode ({self.mode}) ≤ high ({self.high})",
                    ["distribution", "bounds"]
                )

        return self


class MonteCarloSpec(BaseModel):
    """
    Monte Carlo configuration for stochastic scenarios.

    Specifies number of trials and optional seed strategy for
    deterministic substream derivation.
    """
    model_config = ConfigDict(extra="forbid")

    trials: int = Field(
        ...,
        ge=1,
        le=10_000_000,
        description="Number of Monte Carlo trials (1 to 10 million)"
    )
    seed_strategy: Literal["derive-by-path", "sequence", "fixed"] = Field(
        default="derive-by-path",
        description="Seed derivation strategy for substreams"
    )


# ============================================================================
# PYDANTIC MODELS - Top-Level ScenarioSpec v1
# ============================================================================


class ScenarioSpecV1(BaseModel):
    """
    GreenLang ScenarioSpec v1 - Top-Level Schema

    Defines deterministic and stochastic scenario configurations for
    climate modeling workflows. Supports parameter sweeps, Monte Carlo
    simulations, and reproducible random number generation.

    Example:
        >>> spec = ScenarioSpecV1(
        ...     schema_version="1.0.0",
        ...     name="building_decarb_baseline",
        ...     description="Baseline + retrofit sweep with MC on price sensitivity",
        ...     seed=123456789,
        ...     mode="replay",
        ...     parameters=[
        ...         ParameterSpec(
        ...             id="retrofit_level",
        ...             type="sweep",
        ...             values=["none", "light", "deep"]
        ...         ),
        ...         ParameterSpec(
        ...             id="electricity_price_usd_per_kwh",
        ...             type="distribution",
        ...             distribution=DistributionSpec(
        ...                 kind="triangular",
        ...                 low=0.08,
        ...                 mode=0.12,
        ...                 high=0.22
        ...             )
        ...         )
        ...     ],
        ...     monte_carlo=MonteCarloSpec(trials=2000)
        ... )
    """
    model_config = ConfigDict(extra="forbid")

    # Metadata (following GreenLang pattern)
    schema_version: Literal["1.0.0"] = Field(
        default="1.0.0",
        description="ScenarioSpec schema version (MUST be '1.0.0')"
    )
    name: str = Field(
        ...,
        min_length=3,
        description="Scenario name identifier"
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description of scenario purpose"
    )

    # Determinism & reproducibility
    seed: int = Field(
        ...,
        ge=0,
        le=2**64 - 1,
        description="Master seed for deterministic RNG (0 to 2^64-1)"
    )
    mode: Literal["replay", "live"] = Field(
        default="replay",
        description="Execution mode: replay (deterministic) or live (non-deterministic)"
    )

    # Core configuration
    parameters: List[ParameterSpec] = Field(
        ...,
        min_length=1,
        description="Parameter specifications (sweep or distribution)"
    )
    monte_carlo: Optional[MonteCarloSpec] = Field(
        default=None,
        description="Monte Carlo configuration (required if any distribution parameters)"
    )

    # Additional metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (owner, tags, etc.)"
    )

    @field_validator("name")
    @classmethod
    def validate_name_format(cls, v: str) -> str:
        """Validate scenario name format."""
        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$", v):
            raise GLValidationError(
                GLVErr.INVALID_SLUG,
                f"Invalid scenario name: '{v}'. Must start with alphanumeric, contain only alphanumeric, underscore, or hyphen",
                ["name"]
            )
        return v

    @model_validator(mode="after")
    def validate_parameter_ids_unique(self) -> ScenarioSpecV1:
        """Ensure parameter IDs are unique."""
        param_ids = [p.id for p in self.parameters]
        if len(param_ids) != len(set(param_ids)):
            duplicates = [pid for pid in param_ids if param_ids.count(pid) > 1]
            raise GLValidationError(
                GLVErr.DUPLICATE_NAME,
                f"Duplicate parameter IDs: {duplicates}",
                ["parameters"]
            )
        return self

    @model_validator(mode="after")
    def validate_monte_carlo_required(self) -> ScenarioSpecV1:
        """Require monte_carlo config if any distribution parameters exist."""
        has_distributions = any(p.type == "distribution" for p in self.parameters)

        if has_distributions and self.monte_carlo is None:
            raise GLValidationError(
                GLVErr.CONSTRAINT,
                "Monte Carlo configuration required when using distribution parameters",
                ["monte_carlo"]
            )

        if not has_distributions and self.monte_carlo is not None:
            raise GLValidationError(
                GLVErr.CONSTRAINT,
                "Monte Carlo configuration provided but no distribution parameters defined",
                ["monte_carlo"]
            )

        return self


# ============================================================================
# HELPER FUNCTIONS (following GreenLang AgentSpec pattern)
# ============================================================================


def from_yaml(path: Union[str, Path]) -> ScenarioSpecV1:
    """
    Load ScenarioSpec v1 from YAML file.

    Args:
        path: Path to scenario.yaml file

    Returns:
        Validated ScenarioSpecV1 instance

    Raises:
        GLValidationError: If spec is invalid
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML is malformed

    Example:
        >>> spec = from_yaml("scenarios/baseline_sweep.yaml")
        >>> print(spec.name)
        'building_decarb_baseline'
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ScenarioSpec file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    try:
        return ScenarioSpecV1.model_validate(data)
    except ValidationError as e:
        gl_errors = GLValidationError.from_pydantic(e, context=str(path))
        raise gl_errors[0] if gl_errors else e


def from_json(path: Union[str, Path]) -> ScenarioSpecV1:
    """
    Load ScenarioSpec v1 from JSON file.

    Args:
        path: Path to scenario.json file

    Returns:
        Validated ScenarioSpecV1 instance

    Raises:
        GLValidationError: If spec is invalid
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON is malformed
    """
    import json

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ScenarioSpec file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        return ScenarioSpecV1.model_validate(data)
    except ValidationError as e:
        gl_errors = GLValidationError.from_pydantic(e, context=str(path))
        raise gl_errors[0] if gl_errors else e


def validate_spec(data: dict) -> ScenarioSpecV1:
    """
    Validate ScenarioSpec v1 from dictionary.

    Args:
        data: ScenarioSpec data as dictionary

    Returns:
        Validated ScenarioSpecV1 instance

    Raises:
        GLValidationError: If spec is invalid
    """
    try:
        return ScenarioSpecV1.model_validate(data)
    except ValidationError as e:
        gl_errors = GLValidationError.from_pydantic(e)
        raise gl_errors[0] if gl_errors else e


def to_yaml(spec: ScenarioSpecV1, path: Union[str, Path]) -> None:
    """
    Save ScenarioSpec v1 to YAML file.

    Args:
        spec: ScenarioSpecV1 instance
        path: Path to save scenario.yaml

    Example:
        >>> spec = ScenarioSpecV1(...)
        >>> to_yaml(spec, "scenarios/my_scenario.yaml")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = spec.model_dump(exclude_none=True, exclude_defaults=False)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def to_json_schema() -> dict:
    """
    Export ScenarioSpec v1 as JSON Schema (draft-2020-12).

    Generated from Pydantic models (Pydantic is source of truth).
    Used by:
    - CLI validator (gl scenario validate)
    - Documentation generation
    - External tooling (VS Code, CI checks)

    Returns:
        JSON Schema dictionary

    Example:
        >>> schema = to_json_schema()
        >>> print(schema["$id"])
        'https://greenlang.io/specs/scenariospec_v1.json'
    """
    schema = ScenarioSpecV1.model_json_schema(mode="serialization")

    # Add JSON Schema metadata
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://greenlang.io/specs/scenariospec_v1.json"
    schema["title"] = "GreenLang ScenarioSpec v1"
    schema["description"] = "Scenario specification for deterministic and stochastic climate modeling workflows"

    return schema
