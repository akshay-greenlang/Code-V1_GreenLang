# -*- coding: utf-8 -*-
"""
GreenLang Pipeline Template Engine
==================================

FR-005: Reusable Pipeline Template Support for GreenLang Orchestrator.

This module provides pipeline template functionality enabling:

- Reusable template definitions with versioned parameters
- Template registry for storing and retrieving templates
- Template resolution with deterministic step expansion
- Nested template support (templates using templates)
- Parameter validation for template inputs
- Content-addressable hashing for audit trails

Template YAML Structure:
    name: data-quality-checks
    version: "1.0.0"
    description: Standard data quality validation suite
    parameters:
      dataset_uri:
        type: string
        required: true
      threshold:
        type: number
        default: 0.95
    steps:
      - id: validate_schema
        agent: OPS.DATA.SchemaValidator
        in:
          uri: "{{ params.dataset_uri }}"

Usage:
    >>> registry = TemplateRegistry()
    >>> registry.register_from_yaml(template_yaml)
    >>> resolver = TemplateResolver(registry)
    >>> result = resolver.resolve_imports(pipeline_def, imports)

Author: GreenLang Framework Team
Date: January 2026
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# ==============================================================================
# Constants and Patterns
# ==============================================================================

# Semver pattern for version validation
SEMVER_PATTERN = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?$"
)

# Template parameter reference pattern: {{ params.name }}
TEMPLATE_PARAM_PATTERN = re.compile(r"\{\{\s*params\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")

# Template reference pattern: alias.template_name or full name
TEMPLATE_REF_PATTERN = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_-]*)\.([a-zA-Z_][a-zA-Z0-9_-]*)$")

# Maximum nesting depth for template expansion
MAX_TEMPLATE_NESTING_DEPTH = 10


# ==============================================================================
# Enums
# ==============================================================================

class TemplateParameterType(str, Enum):
    """Supported template parameter types."""

    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


class TemplateStatus(str, Enum):
    """Template lifecycle status."""

    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


# ==============================================================================
# Template Parameter Definition
# ==============================================================================

class TemplateParameter(BaseModel):
    """
    Definition for a template parameter.

    Parameters define inputs that must be provided when using the template.
    They support type validation, default values, and constraints.

    Attributes:
        type: Data type of the parameter
        required: Whether the parameter must be provided
        default: Default value if not provided
        description: Human-readable description
        enum: List of allowed values
        minimum: Minimum value (for numeric types)
        maximum: Maximum value (for numeric types)
        pattern: Regex pattern (for string type)

    Example:
        >>> param = TemplateParameter(
        ...     type=TemplateParameterType.STRING,
        ...     required=True,
        ...     description="URI for input dataset"
        ... )
    """

    type: TemplateParameterType = Field(
        default=TemplateParameterType.STRING,
        description="Data type of the parameter"
    )
    required: bool = Field(
        default=False,
        description="Whether the parameter is required"
    )
    default: Optional[Any] = Field(
        default=None,
        description="Default value if not provided"
    )
    description: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Human-readable description"
    )
    enum: Optional[List[Any]] = Field(
        default=None,
        description="Allowed values for the parameter"
    )
    minimum: Optional[Union[int, float]] = Field(
        default=None,
        description="Minimum value (for numeric types)"
    )
    maximum: Optional[Union[int, float]] = Field(
        default=None,
        description="Maximum value (for numeric types)"
    )
    pattern: Optional[str] = Field(
        default=None,
        description="Regex pattern for validation (for string type)"
    )

    @field_validator("type", mode="before")
    @classmethod
    def normalize_type(cls, v: Any) -> TemplateParameterType:
        """Normalize type string to enum."""
        if isinstance(v, str):
            try:
                return TemplateParameterType(v.lower())
            except ValueError:
                raise ValueError(f"Invalid parameter type: {v}")
        return v

    @model_validator(mode="after")
    def validate_constraints(self) -> "TemplateParameter":
        """Validate parameter constraints are consistent."""
        # Required parameters cannot have default (they must be provided)
        if self.required and self.default is not None:
            logger.warning(
                "Parameter marked as required but has default value; "
                "default will be ignored"
            )

        # Validate minimum <= maximum
        if (self.minimum is not None and self.maximum is not None and
                self.minimum > self.maximum):
            raise ValueError(
                f"minimum ({self.minimum}) cannot exceed maximum ({self.maximum})"
            )

        # Validate pattern is valid regex
        if self.pattern is not None:
            try:
                re.compile(self.pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {e}")

        return self

    def validate_value(self, value: Any, param_name: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a value against this parameter definition.

        Args:
            value: The value to validate
            param_name: Parameter name for error messages

        Returns:
            Tuple of (is_valid, error_message)
        """
        if value is None:
            if self.required:
                return False, f"Required parameter '{param_name}' is missing"
            return True, None

        # Type validation
        type_checks = {
            TemplateParameterType.STRING: lambda v: isinstance(v, str),
            TemplateParameterType.INTEGER: lambda v: isinstance(v, int) and not isinstance(v, bool),
            TemplateParameterType.NUMBER: lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            TemplateParameterType.BOOLEAN: lambda v: isinstance(v, bool),
            TemplateParameterType.ARRAY: lambda v: isinstance(v, list),
            TemplateParameterType.OBJECT: lambda v: isinstance(v, dict),
        }

        if not type_checks.get(self.type, lambda v: True)(value):
            return False, f"Parameter '{param_name}' must be of type {self.type.value}, got {type(value).__name__}"

        # Enum validation
        if self.enum is not None and value not in self.enum:
            return False, f"Parameter '{param_name}' must be one of {self.enum}, got {value}"

        # Range validation for numeric types
        if self.type in (TemplateParameterType.INTEGER, TemplateParameterType.NUMBER):
            if self.minimum is not None and value < self.minimum:
                return False, f"Parameter '{param_name}' must be >= {self.minimum}, got {value}"
            if self.maximum is not None and value > self.maximum:
                return False, f"Parameter '{param_name}' must be <= {self.maximum}, got {value}"

        # Pattern validation for string type
        if self.type == TemplateParameterType.STRING and self.pattern is not None:
            if not re.match(self.pattern, value):
                return False, f"Parameter '{param_name}' does not match pattern {self.pattern}"

        return True, None


# ==============================================================================
# Template Step Definition
# ==============================================================================

class TemplateStep(BaseModel):
    """
    Step definition within a template.

    Template steps follow the same structure as pipeline steps but support
    parameter templating for dynamic configuration.

    Attributes:
        id: Unique step identifier within template
        agent: Agent ID to execute
        in_: Input parameters with template placeholders
        outputs: Output extraction mapping
        dependsOn: List of step IDs this step depends on
        timeoutSeconds: Step execution timeout
        retries: Number of retry attempts
        condition: Conditional execution expression
        template: Nested template reference (for template composition)
        templateParams: Parameters for nested template

    Example:
        >>> step = TemplateStep(
        ...     id="validate",
        ...     agent="OPS.DATA.SchemaValidator",
        ...     in_={"uri": "{{ params.dataset_uri }}"}
        ... )
    """

    id: str = Field(
        ...,
        min_length=1,
        max_length=63,
        description="Unique step identifier"
    )
    agent: Optional[str] = Field(
        default=None,
        description="Agent ID to execute"
    )
    in_: Dict[str, Any] = Field(
        default_factory=dict,
        alias="in",
        description="Input parameters for the step"
    )
    outputs: Dict[str, str] = Field(
        default_factory=dict,
        description="Output extraction mapping"
    )
    dependsOn: List[str] = Field(
        default_factory=list,
        description="List of step IDs this step depends on"
    )
    timeoutSeconds: int = Field(
        default=900,
        ge=1,
        le=86400,
        description="Step execution timeout"
    )
    retries: int = Field(
        default=0,
        ge=0,
        le=10,
        description="Number of retry attempts"
    )
    condition: Optional[str] = Field(
        default=None,
        description="Conditional execution expression"
    )
    # Nested template support
    template: Optional[str] = Field(
        default=None,
        description="Nested template reference (e.g., 'dq.standard_checks')"
    )
    templateParams: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for nested template"
    )

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def validate_step_type(self) -> "TemplateStep":
        """Ensure step has either agent or template reference."""
        if self.agent is None and self.template is None:
            raise ValueError(
                f"Step '{self.id}' must have either 'agent' or 'template' defined"
            )
        if self.agent is not None and self.template is not None:
            raise ValueError(
                f"Step '{self.id}' cannot have both 'agent' and 'template' defined"
            )
        return self


# ==============================================================================
# Pipeline Template
# ==============================================================================

class PipelineTemplate(BaseModel):
    """
    Reusable pipeline template definition.

    Templates encapsulate common patterns of pipeline steps that can be
    reused across multiple pipelines with different parameters.

    Attributes:
        name: Template name (unique within namespace)
        version: Semantic version of the template
        description: Human-readable description
        parameters: Parameter definitions for template inputs
        steps: List of step definitions
        status: Template lifecycle status
        author: Template author
        tags: Tags for categorization
        created_at: Creation timestamp
        updated_at: Last update timestamp

    Example:
        >>> template = PipelineTemplate(
        ...     name="data-quality-checks",
        ...     version="1.0.0",
        ...     description="Standard data quality validation suite",
        ...     parameters={
        ...         "dataset_uri": TemplateParameter(type=TemplateParameterType.STRING, required=True)
        ...     },
        ...     steps=[TemplateStep(id="validate", agent="OPS.DATA.Validator")]
        ... )
    """

    name: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Template name (unique within namespace)"
    )
    version: str = Field(
        ...,
        description="Semantic version of the template"
    )
    description: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Human-readable description"
    )
    parameters: Dict[str, TemplateParameter] = Field(
        default_factory=dict,
        description="Parameter definitions for template inputs"
    )
    steps: List[TemplateStep] = Field(
        ...,
        min_length=1,
        description="List of step definitions"
    )
    status: TemplateStatus = Field(
        default=TemplateStatus.ACTIVE,
        description="Template lifecycle status"
    )
    author: Optional[str] = Field(
        default=None,
        description="Template author"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for categorization"
    )
    created_at: Optional[datetime] = Field(
        default=None,
        description="Creation timestamp"
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        description="Last update timestamp"
    )

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate semantic version format."""
        if not re.match(r"^\d+\.\d+(\.\d+)?$", v) and not SEMVER_PATTERN.match(v):
            raise ValueError(
                f"Invalid version '{v}': must be semantic version (e.g., '1.0.0')"
            )
        return v

    @field_validator("parameters", mode="before")
    @classmethod
    def normalize_parameters(cls, v: Any) -> Dict[str, TemplateParameter]:
        """Normalize parameters to dict format."""
        if v is None:
            return {}
        if isinstance(v, dict):
            result = {}
            for name, param in v.items():
                if isinstance(param, dict):
                    result[name] = TemplateParameter(**param)
                elif isinstance(param, TemplateParameter):
                    result[name] = param
                else:
                    raise ValueError(f"Invalid parameter definition for '{name}'")
            return result
        return v

    @model_validator(mode="after")
    def validate_template(self) -> "PipelineTemplate":
        """Validate template structure."""
        step_ids: Set[str] = set()
        for step in self.steps:
            if step.id in step_ids:
                raise ValueError(f"Duplicate step ID in template: '{step.id}'")
            step_ids.add(step.id)

        for step in self.steps:
            for dep in step.dependsOn:
                if dep not in step_ids:
                    raise ValueError(
                        f"Step '{step.id}' depends on unknown step '{dep}'"
                    )

        self._validate_parameter_references()
        return self

    def _validate_parameter_references(self) -> None:
        """Validate that all parameter references exist."""
        param_names = set(self.parameters.keys())

        def check_value(value: Any, context: str) -> None:
            if isinstance(value, str):
                for match in TEMPLATE_PARAM_PATTERN.finditer(value):
                    param_name = match.group(1)
                    if param_name not in param_names:
                        raise ValueError(
                            f"In {context}: reference to undefined parameter '{param_name}'"
                        )
            elif isinstance(value, dict):
                for v in value.values():
                    check_value(v, context)
            elif isinstance(value, list):
                for item in value:
                    check_value(item, context)

        for step in self.steps:
            context = f"step '{step.id}'"
            check_value(step.in_, context)
            check_value(step.templateParams, context)

    def compute_content_hash(self) -> str:
        """
        Compute SHA-256 hash of template content.

        Returns:
            Hexadecimal SHA-256 hash string.
        """
        content = {
            "name": self.name,
            "version": self.version,
            "parameters": {
                name: param.model_dump()
                for name, param in sorted(self.parameters.items())
            },
            "steps": [step.model_dump(by_alias=True) for step in self.steps],
        }
        json_str = json.dumps(content, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


# ==============================================================================
# Template Import Definition
# ==============================================================================

class TemplateImport(BaseModel):
    """
    Template import declaration for a pipeline.

    Attributes:
        name: Full template name (may include namespace)
        version: Required template version (semver)
        as_: Local alias for the template

    Example:
        >>> import_def = TemplateImport(
        ...     name="data-quality-checks",
        ...     version="1.0.0",
        ...     as_="dq"
        ... )
    """

    name: str = Field(..., min_length=1, description="Full template name")
    version: str = Field(..., description="Required template version")
    as_: str = Field(
        ...,
        alias="as",
        min_length=1,
        max_length=32,
        description="Local alias for the template"
    )

    model_config = {"populate_by_name": True}

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate version format."""
        if not re.match(r"^\d+\.\d+(\.\d+)?$", v) and not SEMVER_PATTERN.match(v):
            raise ValueError(f"Invalid version format: {v}")
        return v

    @field_validator("as_")
    @classmethod
    def validate_alias(cls, v: str) -> str:
        """Validate alias format."""
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", v):
            raise ValueError(
                f"Invalid alias '{v}': must be alphanumeric starting with letter/underscore"
            )
        return v


# ==============================================================================
# Template Expansion Result
# ==============================================================================

class ExpandedStep(BaseModel):
    """A step after template expansion with all parameters substituted."""

    id: str = Field(..., description="Unique step ID (may include prefix)")
    agent: str = Field(..., description="Resolved agent ID")
    in_: Dict[str, Any] = Field(default_factory=dict, alias="in")
    outputs: Dict[str, str] = Field(default_factory=dict)
    dependsOn: List[str] = Field(default_factory=list)
    timeoutSeconds: int = Field(default=900)
    retries: int = Field(default=0)
    condition: Optional[str] = Field(default=None)
    source_template: Optional[str] = Field(default=None)
    source_template_version: Optional[str] = Field(default=None)

    model_config = {"populate_by_name": True}


class TemplateExpansionResult(BaseModel):
    """
    Result of template expansion with expanded steps and metadata.

    Attributes:
        steps: List of expanded step definitions
        content_hash: SHA-256 hash of expansion result
        templates_used: Map of template name -> version used
        expansion_depth: Maximum nesting depth encountered
        warnings: Any warnings generated during expansion
    """

    steps: List[ExpandedStep] = Field(default_factory=list)
    content_hash: str = Field(..., description="SHA-256 hash of expansion result")
    templates_used: Dict[str, str] = Field(default_factory=dict)
    expansion_depth: int = Field(default=0)
    warnings: List[str] = Field(default_factory=list)


# ==============================================================================
# Template Registry
# ==============================================================================

class TemplateRegistry:
    """
    Registry for storing and retrieving pipeline templates.

    Supports template registration with versioning, version-aware retrieval,
    and loading templates from YAML files or directories.

    Example:
        >>> registry = TemplateRegistry()
        >>> registry.register(template)
        >>> template = registry.get("data-quality-checks", "1.0.0")
    """

    def __init__(self) -> None:
        """Initialize empty template registry."""
        self._templates: Dict[str, Dict[str, PipelineTemplate]] = {}
        self._load_count = 0
        logger.info("Initialized TemplateRegistry")

    def register(self, template: PipelineTemplate) -> None:
        """
        Register a template in the registry.

        Args:
            template: Template to register

        Raises:
            ValueError: If template version already exists
        """
        name = template.name
        version = template.version

        if name not in self._templates:
            self._templates[name] = {}

        if version in self._templates[name]:
            raise ValueError(
                f"Template '{name}' version '{version}' already registered"
            )

        self._templates[name][version] = template
        self._load_count += 1
        logger.info(f"Registered template: {name}@{version}")

    def register_from_yaml(self, yaml_content: str) -> PipelineTemplate:
        """
        Register a template from YAML content.

        Args:
            yaml_content: YAML string containing template definition

        Returns:
            The registered PipelineTemplate
        """
        import yaml

        try:
            data = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML: {e}") from e

        if data is None:
            raise ValueError("Empty YAML content")

        template = PipelineTemplate.model_validate(data)
        self.register(template)
        return template

    def register_from_file(self, path: Union[str, Path]) -> PipelineTemplate:
        """Register a template from a YAML file."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Template file not found: {path}")

        with open(file_path, "r", encoding="utf-8") as f:
            return self.register_from_yaml(f.read())

    def load_directory(self, directory: Union[str, Path]) -> int:
        """Load all templates from a directory."""
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        loaded = 0
        for yaml_file in dir_path.glob("*.yaml"):
            try:
                self.register_from_file(yaml_file)
                loaded += 1
            except Exception as e:
                logger.warning(f"Failed to load template from {yaml_file}: {e}")

        for yml_file in dir_path.glob("*.yml"):
            try:
                self.register_from_file(yml_file)
                loaded += 1
            except Exception as e:
                logger.warning(f"Failed to load template from {yml_file}: {e}")

        logger.info(f"Loaded {loaded} templates from {directory}")
        return loaded

    def get(self, name: str, version: Optional[str] = None) -> Optional[PipelineTemplate]:
        """Get a template by name and optional version."""
        if name not in self._templates:
            return None

        versions = self._templates[name]

        if version is not None:
            return versions.get(version)

        if not versions:
            return None

        latest_version = self._get_latest_version(list(versions.keys()))
        return versions.get(latest_version)

    def get_versions(self, name: str) -> List[str]:
        """Get all versions of a template, sorted newest first."""
        if name not in self._templates:
            return []
        versions = list(self._templates[name].keys())
        return sorted(versions, key=self._version_key, reverse=True)

    def list_templates(self) -> List[str]:
        """List all registered template names."""
        return list(self._templates.keys())

    def exists(self, name: str, version: Optional[str] = None) -> bool:
        """Check if a template exists."""
        if name not in self._templates:
            return False
        if version is None:
            return len(self._templates[name]) > 0
        return version in self._templates[name]

    def unregister(self, name: str, version: Optional[str] = None) -> bool:
        """Unregister a template."""
        if name not in self._templates:
            return False

        if version is None:
            del self._templates[name]
            logger.info(f"Unregistered all versions of template: {name}")
            return True

        if version in self._templates[name]:
            del self._templates[name][version]
            if not self._templates[name]:
                del self._templates[name]
            logger.info(f"Unregistered template: {name}@{version}")
            return True

        return False

    def clear(self) -> None:
        """Clear all registered templates."""
        self._templates.clear()
        logger.info("Cleared template registry")

    @property
    def template_count(self) -> int:
        """Total number of template versions registered."""
        return sum(len(versions) for versions in self._templates.values())

    def _get_latest_version(self, versions: List[str]) -> str:
        """Get the latest version from a list of versions."""
        return max(versions, key=self._version_key)

    @staticmethod
    def _version_key(version: str) -> Tuple[int, ...]:
        """Convert version string to sortable tuple."""
        base_version = version.split("-")[0]
        parts = base_version.split(".")
        return tuple(int(p) for p in parts)


# ==============================================================================
# Template Resolver
# ==============================================================================

class TemplateResolver:
    """
    Resolves template references and expands pipelines.

    Handles resolving template imports, expanding template references,
    validating parameters, and computing deterministic expansion hashes.

    Example:
        >>> registry = TemplateRegistry()
        >>> registry.register(template)
        >>> resolver = TemplateResolver(registry)
        >>> result = resolver.expand_template(template, params)
    """

    def __init__(self, registry: TemplateRegistry) -> None:
        """Initialize resolver with a template registry."""
        self._registry = registry
        self._expansion_cache: Dict[str, TemplateExpansionResult] = {}
        logger.info("Initialized TemplateResolver")

    def resolve_imports(self, imports: List[TemplateImport]) -> Dict[str, PipelineTemplate]:
        """
        Resolve template imports to actual templates.

        Args:
            imports: List of template imports to resolve

        Returns:
            Dict mapping alias to resolved template
        """
        resolved: Dict[str, PipelineTemplate] = {}

        for imp in imports:
            template = self._registry.get(imp.name, imp.version)
            if template is None:
                raise ValueError(f"Template not found: {imp.name}@{imp.version}")

            if imp.as_ in resolved:
                raise ValueError(f"Duplicate template alias: '{imp.as_}'")

            resolved[imp.as_] = template
            logger.debug(f"Resolved import: {imp.as_} -> {imp.name}@{imp.version}")

        return resolved

    def expand_template(
        self,
        template: PipelineTemplate,
        params: Dict[str, Any],
        step_prefix: str = "",
        resolved_templates: Optional[Dict[str, PipelineTemplate]] = None,
        depth: int = 0,
        expansion_stack: Optional[Set[str]] = None
    ) -> TemplateExpansionResult:
        """
        Expand a template with given parameters.

        Args:
            template: Template to expand
            params: Parameter values to substitute
            step_prefix: Prefix for generated step IDs
            resolved_templates: Pre-resolved templates for nested expansion
            depth: Current expansion depth
            expansion_stack: Stack for cycle detection

        Returns:
            TemplateExpansionResult with expanded steps
        """
        if expansion_stack is None:
            expansion_stack = set()

        template_key = f"{template.name}@{template.version}"
        if template_key in expansion_stack:
            raise ValueError(
                f"Template cycle detected: {template_key} is already in expansion stack"
            )

        if depth > MAX_TEMPLATE_NESTING_DEPTH:
            raise ValueError(
                f"Template nesting depth exceeds limit ({MAX_TEMPLATE_NESTING_DEPTH})"
            )

        expansion_stack.add(template_key)

        try:
            effective_params = self._validate_and_apply_defaults(template, params)

            cache_key = self._compute_cache_key(template, effective_params, step_prefix)
            if cache_key in self._expansion_cache:
                logger.debug(f"Cache hit for template expansion: {template.name}")
                return self._expansion_cache[cache_key]

            expanded_steps: List[ExpandedStep] = []
            templates_used: Dict[str, str] = {template.name: template.version}
            max_depth = depth
            warnings: List[str] = []

            for step in template.steps:
                if step.template is not None:
                    nested_result = self._expand_nested_template(
                        step=step,
                        effective_params=effective_params,
                        step_prefix=step_prefix,
                        resolved_templates=resolved_templates or {},
                        depth=depth + 1,
                        expansion_stack=expansion_stack.copy()
                    )
                    expanded_steps.extend(nested_result.steps)
                    templates_used.update(nested_result.templates_used)
                    max_depth = max(max_depth, nested_result.expansion_depth)
                    warnings.extend(nested_result.warnings)
                else:
                    expanded_step = self._expand_step(
                        step=step,
                        params=effective_params,
                        step_prefix=step_prefix,
                        template=template
                    )
                    expanded_steps.append(expanded_step)

            expanded_steps = self._update_dependencies(expanded_steps, step_prefix, template)

            content_hash = self._compute_expansion_hash(expanded_steps, templates_used)

            result = TemplateExpansionResult(
                steps=expanded_steps,
                content_hash=content_hash,
                templates_used=templates_used,
                expansion_depth=max_depth,
                warnings=warnings
            )

            self._expansion_cache[cache_key] = result
            return result

        finally:
            expansion_stack.discard(template_key)

    def expand_step_reference(
        self,
        template_ref: str,
        template_params: Dict[str, Any],
        step_id: str,
        resolved_templates: Dict[str, PipelineTemplate],
        depth: int = 0
    ) -> TemplateExpansionResult:
        """Expand a template reference from a step."""
        match = TEMPLATE_REF_PATTERN.match(template_ref)
        if not match:
            raise ValueError(f"Invalid template reference format: '{template_ref}'")

        alias = match.group(1)

        if alias not in resolved_templates:
            raise ValueError(f"Template alias '{alias}' not found in imports")

        template = resolved_templates[alias]

        return self.expand_template(
            template=template,
            params=template_params,
            step_prefix=f"{step_id}_",
            resolved_templates=resolved_templates,
            depth=depth
        )

    def validate_parameters(
        self,
        template: PipelineTemplate,
        params: Dict[str, Any]
    ) -> List[str]:
        """Validate parameters against template definition."""
        errors: List[str] = []

        for name, param_def in template.parameters.items():
            value = params.get(name)
            if value is None and param_def.default is not None:
                continue
            is_valid, error = param_def.validate_value(value, name)
            if not is_valid:
                errors.append(error)

        known_params = set(template.parameters.keys())
        for name in params:
            if name not in known_params:
                errors.append(f"Unknown parameter: '{name}'")

        return errors

    def clear_cache(self) -> None:
        """Clear the expansion cache."""
        self._expansion_cache.clear()
        logger.debug("Cleared template expansion cache")

    def _validate_and_apply_defaults(
        self,
        template: PipelineTemplate,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate parameters and apply defaults."""
        errors = self.validate_parameters(template, params)
        if errors:
            raise ValueError(f"Template parameter validation failed: {'; '.join(errors)}")

        effective_params = dict(params)
        for name, param_def in template.parameters.items():
            if name not in effective_params and param_def.default is not None:
                effective_params[name] = param_def.default

        return effective_params

    def _expand_step(
        self,
        step: TemplateStep,
        params: Dict[str, Any],
        step_prefix: str,
        template: PipelineTemplate
    ) -> ExpandedStep:
        """Expand a single template step."""
        prefixed_id = f"{step_prefix}{step.id}" if step_prefix else step.id
        expanded_inputs = self._substitute_params(step.in_, params)

        return ExpandedStep(
            id=prefixed_id,
            agent=step.agent,
            in_=expanded_inputs,
            outputs=step.outputs,
            dependsOn=[],
            timeoutSeconds=step.timeoutSeconds,
            retries=step.retries,
            condition=self._substitute_params_str(step.condition, params) if step.condition else None,
            source_template=template.name,
            source_template_version=template.version
        )

    def _expand_nested_template(
        self,
        step: TemplateStep,
        effective_params: Dict[str, Any],
        step_prefix: str,
        resolved_templates: Dict[str, PipelineTemplate],
        depth: int,
        expansion_stack: Set[str]
    ) -> TemplateExpansionResult:
        """Expand a nested template reference."""
        expanded_template_params = self._substitute_params(
            step.templateParams, effective_params
        )

        template_ref = step.template
        match = TEMPLATE_REF_PATTERN.match(template_ref)

        if match:
            alias = match.group(1)
            if alias not in resolved_templates:
                raise ValueError(
                    f"Template alias '{alias}' in step '{step.id}' not found in imports"
                )
            nested_template = resolved_templates[alias]
        else:
            nested_template = self._registry.get(template_ref)
            if nested_template is None:
                raise ValueError(f"Template '{template_ref}' in step '{step.id}' not found")

        nested_prefix = f"{step_prefix}{step.id}_" if step_prefix else f"{step.id}_"

        return self.expand_template(
            template=nested_template,
            params=expanded_template_params,
            step_prefix=nested_prefix,
            resolved_templates=resolved_templates,
            depth=depth,
            expansion_stack=expansion_stack
        )

    def _substitute_params(self, value: Any, params: Dict[str, Any]) -> Any:
        """Recursively substitute parameters in a value."""
        if isinstance(value, str):
            return self._substitute_params_str(value, params)
        elif isinstance(value, dict):
            return {k: self._substitute_params(v, params) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._substitute_params(item, params) for item in value]
        else:
            return value

    def _substitute_params_str(self, value: str, params: Dict[str, Any]) -> Any:
        """Substitute parameters in a string value."""
        result = value

        for match in TEMPLATE_PARAM_PATTERN.finditer(value):
            param_name = match.group(1)
            if param_name in params:
                param_value = params[param_name]
                if isinstance(param_value, (str, int, float, bool)):
                    placeholder = match.group(0)
                    if value == placeholder:
                        return param_value
                    result = result.replace(placeholder, str(param_value))
                else:
                    placeholder = match.group(0)
                    result = result.replace(placeholder, json.dumps(param_value))

        return result

    def _update_dependencies(
        self,
        steps: List[ExpandedStep],
        prefix: str,
        template: PipelineTemplate
    ) -> List[ExpandedStep]:
        """Update step dependencies with prefixes."""
        step_ids = {step.id for step in steps}
        template_step_deps = {step.id: step.dependsOn for step in template.steps}
        updated_steps = []

        for step in steps:
            original_id = step.id[len(prefix):] if prefix and step.id.startswith(prefix) else step.id
            original_deps = template_step_deps.get(original_id, [])

            updated_deps = []
            for dep in original_deps:
                prefixed_dep = f"{prefix}{dep}" if prefix else dep
                if prefixed_dep in step_ids:
                    updated_deps.append(prefixed_dep)
                elif dep in step_ids:
                    updated_deps.append(dep)
                else:
                    updated_deps.append(dep)

            updated_step = ExpandedStep(
                id=step.id,
                agent=step.agent,
                in_=step.in_,
                outputs=step.outputs,
                dependsOn=updated_deps,
                timeoutSeconds=step.timeoutSeconds,
                retries=step.retries,
                condition=step.condition,
                source_template=step.source_template,
                source_template_version=step.source_template_version
            )
            updated_steps.append(updated_step)

        return updated_steps

    def _compute_cache_key(
        self,
        template: PipelineTemplate,
        params: Dict[str, Any],
        prefix: str
    ) -> str:
        """Compute cache key for template expansion."""
        content = {
            "template": template.name,
            "version": template.version,
            "params": params,
            "prefix": prefix
        }
        json_str = json.dumps(content, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _compute_expansion_hash(
        self,
        steps: List[ExpandedStep],
        templates_used: Dict[str, str]
    ) -> str:
        """Compute deterministic hash of expansion result."""
        content = {
            "steps": [step.model_dump(by_alias=True) for step in steps],
            "templates": dict(sorted(templates_used.items()))
        }
        json_str = json.dumps(content, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(json_str.encode()).hexdigest()


# ==============================================================================
# Utility Functions
# ==============================================================================

def load_template_yaml(yaml_content: str) -> PipelineTemplate:
    """Load a template from YAML content."""
    import yaml

    try:
        data = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML: {e}") from e

    if data is None:
        raise ValueError("Empty YAML content")

    return PipelineTemplate.model_validate(data)


def load_template_file(path: Union[str, Path]) -> PipelineTemplate:
    """Load a template from a YAML file."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Template file not found: {path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return load_template_yaml(f.read())


def create_template_registry(
    templates_dir: Optional[Union[str, Path]] = None
) -> TemplateRegistry:
    """Create a template registry, optionally loading from directory."""
    registry = TemplateRegistry()
    if templates_dir:
        registry.load_directory(templates_dir)
    return registry


# ==============================================================================
# Exports
# ==============================================================================

__all__ = [
    "PipelineTemplate",
    "TemplateParameter",
    "TemplateStep",
    "TemplateImport",
    "ExpandedStep",
    "TemplateExpansionResult",
    "TemplateParameterType",
    "TemplateStatus",
    "TemplateRegistry",
    "TemplateResolver",
    "MAX_TEMPLATE_NESTING_DEPTH",
    "load_template_yaml",
    "load_template_file",
    "create_template_registry",
]
