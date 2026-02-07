# -*- coding: utf-8 -*-
"""
Pack Format - agent.pack.yaml specification and parser for GreenLang agents.

Defines the canonical pack manifest schema that every GreenLang agent must
provide. Handles parsing, validation, serialization, and template generation.

The agent.pack.yaml file is the single source of truth for an agent's
identity, dependencies, resource requirements, and metadata.

Example:
    >>> pack = PackFormat.load("agents/emissions_calc/agent.pack.yaml")
    >>> print(pack.name, pack.version)
    emissions-calc 1.2.0

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_AGENT_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9._-]{1,127}$")
_SEMVER_PATTERN = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)
SPEC_VERSION = "1.0"
DEFAULT_CPU_LIMIT = "500m"
DEFAULT_MEMORY_LIMIT = "512Mi"
DEFAULT_TIMEOUT_SECONDS = 300


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class AgentType(str, Enum):
    """Classification of GreenLang agent processing strategy."""

    DETERMINISTIC = "deterministic"
    """Agent uses only deterministic calculations (zero-hallucination path)."""

    REASONING = "reasoning"
    """Agent uses LLM reasoning for classification or entity resolution."""

    INSIGHT = "insight"
    """Agent generates narrative insights or summaries using LLM."""


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class AgentDependency(BaseModel):
    """Dependency on another GreenLang agent.

    Attributes:
        name: Agent key of the dependency.
        version_constraint: Semver range constraint (e.g. ^1.0.0, ~1.2.0, >=1.0.0,<2.0.0).
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(..., min_length=2, max_length=128, description="Agent key.")
    version_constraint: str = Field(
        default="*",
        max_length=64,
        description="Semver range constraint.",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate agent key format."""
        if not _AGENT_KEY_PATTERN.match(v):
            raise ValueError(f"Invalid agent dependency name: '{v}'")
        return v


class PythonDependency(BaseModel):
    """Dependency on a Python package from PyPI.

    Attributes:
        package: PyPI package name.
        version_constraint: PEP 440 version specifier.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    package: str = Field(..., min_length=1, max_length=128, description="PyPI package.")
    version_constraint: str = Field(
        default="*",
        max_length=64,
        description="PEP 440 version specifier.",
    )


class ResourceSpec(BaseModel):
    """Compute resource limits for agent execution.

    Attributes:
        cpu_limit: Kubernetes CPU limit (e.g. '500m', '1').
        memory_limit: Kubernetes memory limit (e.g. '512Mi', '1Gi').
        timeout_seconds: Maximum execution time before forced termination.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    cpu_limit: str = Field(default=DEFAULT_CPU_LIMIT, description="K8s CPU limit.")
    memory_limit: str = Field(default=DEFAULT_MEMORY_LIMIT, description="K8s memory limit.")
    timeout_seconds: int = Field(
        default=DEFAULT_TIMEOUT_SECONDS,
        ge=1,
        le=86400,
        description="Max execution time in seconds.",
    )


class InputOutputSchema(BaseModel):
    """Schema definition for agent inputs or outputs.

    Attributes:
        schema_type: Schema format identifier (e.g. 'json_schema', 'pydantic').
        fields: Mapping of field name to type/description specification.
    """

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    schema_type: str = Field(default="json_schema", description="Schema format.")
    fields: Dict[str, Any] = Field(default_factory=dict, description="Field definitions.")


class AgentMetadata(BaseModel):
    """Descriptive metadata for an agent package.

    Attributes:
        author: Author name or team.
        license: SPDX license identifier.
        tags: Searchable tags for the agent hub.
        regulatory: Regulatory frameworks this agent supports.
    """

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    author: str = Field(default="", max_length=256, description="Author name.")
    license: str = Field(default="Proprietary", max_length=64, description="SPDX license.")
    tags: List[str] = Field(default_factory=list, description="Searchable tags.")
    regulatory: List[str] = Field(default_factory=list, description="Regulatory frameworks.")


# ---------------------------------------------------------------------------
# Main Pack Model
# ---------------------------------------------------------------------------


class AgentPack(BaseModel):
    """Complete agent package manifest (agent.pack.yaml).

    Attributes:
        name: Unique agent key used across the platform.
        version: Semantic version of this agent release.
        description: Human-readable description of the agent's purpose.
        agent_type: Processing classification.
        spec_version: Pack format specification version.
        entry_point: Python module path to the agent class.
        base_class: Fully qualified base class name.
        dependencies: Agent and Python dependency declarations.
        inputs: Input schema definition.
        outputs: Output schema definition.
        resources: Compute resource limits.
        metadata: Descriptive metadata.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(..., min_length=2, max_length=128, description="Unique agent key.")
    version: str = Field(..., min_length=5, max_length=64, description="Semantic version.")
    description: str = Field(default="", max_length=2048, description="Agent description.")
    agent_type: AgentType = Field(default=AgentType.DETERMINISTIC, description="Agent type.")
    spec_version: str = Field(default=SPEC_VERSION, description="Pack spec version.")
    entry_point: str = Field(..., min_length=1, max_length=256, description="Python module path.")
    base_class: str = Field(
        default="greenlang.agents.base.BaseAgent",
        max_length=256,
        description="Base class.",
    )
    dependencies: Dict[str, List[Any]] = Field(
        default_factory=lambda: {"agents": [], "python": []},
        description="Dependency declarations.",
    )
    inputs: InputOutputSchema = Field(
        default_factory=InputOutputSchema, description="Input schema."
    )
    outputs: InputOutputSchema = Field(
        default_factory=InputOutputSchema, description="Output schema."
    )
    resources: ResourceSpec = Field(
        default_factory=ResourceSpec, description="Resource limits."
    )
    metadata: AgentMetadata = Field(
        default_factory=AgentMetadata, description="Agent metadata."
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate agent key pattern."""
        if not _AGENT_KEY_PATTERN.match(v):
            raise ValueError(
                f"Agent name '{v}' is invalid. Must match: [a-z][a-z0-9._-]{{1,127}}"
            )
        return v

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate semantic version format."""
        if not _SEMVER_PATTERN.match(v):
            raise ValueError(
                f"Version '{v}' is not a valid semantic version (MAJOR.MINOR.PATCH)."
            )
        return v

    @model_validator(mode="after")
    def parse_dependency_objects(self) -> AgentPack:
        """Convert raw dependency dicts into typed dependency objects."""
        raw = self.dependencies
        agents_raw = raw.get("agents", [])
        python_raw = raw.get("python", [])
        parsed_agents: List[Any] = []
        for dep in agents_raw:
            if isinstance(dep, dict):
                parsed_agents.append(AgentDependency(**dep))
            elif isinstance(dep, AgentDependency):
                parsed_agents.append(dep)
        parsed_python: List[Any] = []
        for dep in python_raw:
            if isinstance(dep, dict):
                parsed_python.append(PythonDependency(**dep))
            elif isinstance(dep, PythonDependency):
                parsed_python.append(dep)
        self.dependencies = {"agents": parsed_agents, "python": parsed_python}
        return self

    @property
    def agent_dependencies(self) -> List[AgentDependency]:
        """Return typed list of agent dependencies."""
        return self.dependencies.get("agents", [])

    @property
    def python_dependencies(self) -> List[PythonDependency]:
        """Return typed list of Python dependencies."""
        return self.dependencies.get("python", [])

    def to_yaml_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict suitable for YAML output."""
        data = self.model_dump(mode="json")
        # Flatten dependency objects
        agents = []
        for dep in self.agent_dependencies:
            agents.append({"name": dep.name, "version_constraint": dep.version_constraint})
        python_deps = []
        for dep in self.python_dependencies:
            python_deps.append({"package": dep.package, "version_constraint": dep.version_constraint})
        data["dependencies"] = {"agents": agents, "python": python_deps}
        return data


# ---------------------------------------------------------------------------
# PackFormat Facade
# ---------------------------------------------------------------------------


class PackFormat:
    """Facade for loading, saving, and generating agent.pack.yaml files.

    Example:
        >>> pack = PackFormat.load("agents/my_agent/agent.pack.yaml")
        >>> PackFormat.save(pack, "agents/my_agent/agent.pack.yaml")
        >>> template = PackFormat.generate_template("my-new-agent")
    """

    @staticmethod
    def load(path: str | Path) -> AgentPack:
        """Load and validate an agent.pack.yaml file.

        Args:
            path: Filesystem path to the YAML file.

        Returns:
            Validated AgentPack model.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file content is invalid.
        """
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Pack file not found: {filepath}")
        logger.info("Loading pack file: %s", filepath)
        with open(filepath, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if not isinstance(raw, dict):
            raise ValueError(f"Pack file must contain a YAML mapping, got {type(raw).__name__}")
        return AgentPack(**raw)

    @staticmethod
    def save(pack: AgentPack, path: str | Path) -> None:
        """Save an AgentPack to a YAML file.

        Args:
            pack: Validated AgentPack model.
            path: Destination file path.
        """
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = pack.to_yaml_dict()
        with open(filepath, "w", encoding="utf-8") as fh:
            yaml.dump(data, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)
        logger.info("Saved pack file: %s", filepath)

    @staticmethod
    def generate_template(
        name: str,
        agent_type: AgentType = AgentType.DETERMINISTIC,
        version: str = "0.1.0",
    ) -> AgentPack:
        """Generate a minimal pack template for a new agent.

        Args:
            name: Agent key (must match naming convention).
            agent_type: Classification of the agent.
            version: Initial version string.

        Returns:
            AgentPack with sensible defaults.
        """
        return AgentPack(
            name=name,
            version=version,
            description=f"GreenLang agent: {name}",
            agent_type=agent_type,
            entry_point=f"greenlang.agents.{name.replace('-', '_')}.agent",
            base_class="greenlang.agents.base.BaseAgent",
            dependencies={"agents": [], "python": []},
            resources=ResourceSpec(),
            metadata=AgentMetadata(
                author="GreenLang Platform Team",
                license="Proprietary",
                tags=[name, agent_type.value],
            ),
        )

    @staticmethod
    def validate(path: str | Path) -> List[str]:
        """Validate a pack file and return a list of error messages.

        Returns:
            Empty list if valid, list of error strings otherwise.
        """
        errors: List[str] = []
        try:
            PackFormat.load(path)
        except FileNotFoundError as exc:
            errors.append(str(exc))
        except (ValueError, Exception) as exc:
            errors.append(f"Validation failed: {exc}")
        return errors
