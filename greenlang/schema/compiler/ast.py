# -*- coding: utf-8 -*-
"""
Schema Abstract Syntax Tree (AST) for GL-FOUND-X-002
====================================================

This module defines the AST node types for GreenLang schemas, supporting
JSON Schema Draft 2020-12 with GreenLang extensions ($unit, $dimension,
$rules, $aliases, $deprecated, $renamed_from).

The AST provides a structured, immutable representation of parsed schemas
that can be analyzed, validated, and compiled to Intermediate Representation (IR).

Key Components:
    - SchemaNode: Base class for all AST nodes
    - SchemaDocument: Root document node
    - TypeNode: Base class for type definition nodes
    - ObjectTypeNode, ArrayTypeNode, StringTypeNode, NumericTypeNode: Type nodes
    - BooleanTypeNode, NullTypeNode, RefNode, CompositionNode, EnumTypeNode: Type nodes
    - GreenLangExtensions: GreenLang-specific extensions
    - UnitSpec, DeprecationInfo, RuleBinding: Extension metadata

Design Principles:
    - All nodes are immutable after creation (frozen=True)
    - Complete Pydantic v2 validation
    - JSON-serializable for debugging and caching
    - Clear error messages for validation failures
    - Forward references supported for recursive types

Example:
    >>> from greenlang.schema.compiler.ast import (
    ...     SchemaDocument, ObjectTypeNode, StringTypeNode, create_node_id
    ... )
    >>> name_field = StringTypeNode(
    ...     node_id=create_node_id("/properties/name"),
    ...     type="string",
    ...     min_length=1,
    ...     max_length=100
    ... )
    >>> root = ObjectTypeNode(
    ...     node_id=create_node_id("/"),
    ...     type="object",
    ...     properties={"name": name_field},
    ...     required=["name"]
    ... )
    >>> schema = SchemaDocument(
    ...     node_id="root",
    ...     schema_id="emissions/activity",
    ...     version="1.3.0",
    ...     root=root
    ... )

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator - Task 1.2
"""

from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, ForwardRef, List, Literal, Optional, Tuple, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# =============================================================================
# Constants
# =============================================================================

# JSON Schema Draft 2020-12 dialect URI
JSON_SCHEMA_DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"

# GreenLang extension prefixes
GL_EXTENSION_PREFIX = "$gl_"
GL_UNIT_KEY = "$unit"
GL_DIMENSION_KEY = "$dimension"
GL_RULES_KEY = "$rules"
GL_ALIASES_KEY = "$aliases"
GL_DEPRECATED_KEY = "$deprecated"
GL_RENAMED_FROM_KEY = "$renamed_from"

# JSON Schema type values
JSON_SCHEMA_TYPES = frozenset(
    ["string", "number", "integer", "boolean", "object", "array", "null"]
)

# Semantic versioning pattern
SEMVER_PATTERN = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(-((0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(\.(0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(\+([0-9a-zA-Z-]+(\.[0-9a-zA-Z-]+)*))?$"
)

# Schema ID pattern (lowercase alphanumeric with slashes)
SCHEMA_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(/[a-z][a-z0-9_]*)*$")

# JSON Pointer pattern (RFC 6901)
JSON_POINTER_PATTERN = re.compile(r"^(/[^/]*)*$")


# =============================================================================
# Severity Enum for Rules
# =============================================================================


class RuleSeverity(str, Enum):
    """
    Severity level for rule bindings.

    Defines how severe a rule violation is treated:
    - ERROR: Validation failure, payload is invalid
    - WARNING: Potential issue, but payload may still be valid
    - INFO: Informational note, no action required
    """

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

    def is_blocking(self) -> bool:
        """Check if this severity blocks validation."""
        return self == RuleSeverity.ERROR


# =============================================================================
# GreenLang Extensions Models
# =============================================================================


class UnitSpec(BaseModel):
    """
    Unit specification for a field ($unit extension).

    Defines the physical dimension and unit constraints for numeric fields
    that represent physical quantities.

    Attributes:
        dimension: Physical dimension category (e.g., "energy", "mass", "volume").
        canonical: The canonical (SI) unit for normalization (e.g., "kWh", "kg").
        allowed: List of allowed input units that can be converted to canonical.

    Example:
        >>> unit = UnitSpec(
        ...     dimension="energy",
        ...     canonical="kWh",
        ...     allowed=["kWh", "MWh", "GJ", "MMBTU"]
        ... )
    """

    dimension: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Physical dimension (e.g., 'energy', 'mass', 'volume')",
    )

    canonical: str = Field(
        ...,
        min_length=1,
        max_length=32,
        description="Canonical SI unit for normalization (e.g., 'kWh', 'kg')",
    )

    allowed: List[str] = Field(
        default_factory=list,
        max_length=100,
        description="Allowed input units for conversion",
    )

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"dimension": "energy", "canonical": "kWh", "allowed": ["kWh", "MWh", "GJ"]},
                {"dimension": "mass", "canonical": "kg", "allowed": ["kg", "t", "lb"]},
            ]
        },
    )

    @field_validator("dimension")
    @classmethod
    def validate_dimension(cls, v: str) -> str:
        """Validate dimension is lowercase alphanumeric."""
        if not re.match(r"^[a-z][a-z0-9_]*$", v):
            raise ValueError(
                f"Invalid dimension '{v}'. Must be lowercase alphanumeric with underscores."
            )
        return v

    @field_validator("canonical")
    @classmethod
    def validate_canonical(cls, v: str) -> str:
        """Validate canonical unit format."""
        if not v.strip():
            raise ValueError("Canonical unit cannot be empty or whitespace.")
        return v

    @field_validator("allowed")
    @classmethod
    def validate_allowed(cls, v: List[str]) -> List[str]:
        """Validate allowed units are non-empty strings."""
        for unit in v:
            if not unit or not unit.strip():
                raise ValueError("Allowed units cannot contain empty strings.")
        return v

    def contains_canonical(self) -> bool:
        """Check if canonical unit is in allowed list."""
        return self.canonical in self.allowed

    def is_unit_allowed(self, unit: str) -> bool:
        """Check if a unit is allowed (case-sensitive)."""
        if not self.allowed:
            return True  # No restrictions
        return unit in self.allowed


class DeprecationInfo(BaseModel):
    """
    Deprecation metadata for a field ($deprecated extension).

    Provides information about deprecated fields to help users migrate
    to newer schema versions.

    Attributes:
        since_version: Schema version when deprecation was introduced.
        message: Human-readable deprecation message.
        replacement: Suggested replacement field path (optional).
        removal_version: Schema version when field will be removed (optional).

    Example:
        >>> deprecation = DeprecationInfo(
        ...     since_version="2.0.0",
        ...     message="Use 'energy_consumption' instead",
        ...     replacement="/energy_consumption",
        ...     removal_version="3.0.0"
        ... )
    """

    since_version: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Schema version when deprecation was introduced",
    )

    message: str = Field(
        ...,
        min_length=1,
        max_length=1024,
        description="Human-readable deprecation message",
    )

    replacement: Optional[str] = Field(
        default=None,
        max_length=256,
        description="Suggested replacement field path",
    )

    removal_version: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Schema version when field will be removed",
    )

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "since_version": "2.0.0",
                    "message": "Use 'energy_consumption' instead",
                    "replacement": "/energy_consumption",
                    "removal_version": "3.0.0",
                }
            ]
        },
    )

    @field_validator("since_version", "removal_version")
    @classmethod
    def validate_version(cls, v: Optional[str]) -> Optional[str]:
        """Validate version follows semver format."""
        if v is not None and not SEMVER_PATTERN.match(v):
            raise ValueError(
                f"Invalid version '{v}'. Must follow semantic versioning "
                "(e.g., '1.0.0', '2.0.0-beta.1')."
            )
        return v

    def is_removal_imminent(self, current_version: str) -> bool:
        """Check if removal version is close to current version."""
        if not self.removal_version:
            return False
        # Simple major version comparison
        try:
            current_major = int(current_version.split(".")[0])
            removal_major = int(self.removal_version.split(".")[0])
            return removal_major <= current_major + 1
        except (ValueError, IndexError):
            return False


class RuleBinding(BaseModel):
    """
    Binding to a cross-field validation rule ($rules extension).

    Defines a rule that validates relationships between multiple fields,
    such as "if field A is present, field B must also be present" or
    "sum of components must equal total".

    Attributes:
        rule_id: Unique identifier for the rule.
        rule_pack: Optional rule pack identifier for grouping rules.
        severity: Severity level (error, warning, info).
        when: Optional condition expression for when the rule applies.
        check: Validation expression that must evaluate to true.
        message: Error message if rule fails.
        message_template: Template with {{ var }} placeholders for dynamic messages.

    Example:
        >>> rule = RuleBinding(
        ...     rule_id="scope1_scope2_sum",
        ...     severity=RuleSeverity.ERROR,
        ...     check={"$eq": ["$scope1 + $scope2", "$total"]},
        ...     message="Scope 1 + Scope 2 must equal total emissions"
        ... )
    """

    rule_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Unique identifier for the rule",
    )

    rule_pack: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Optional rule pack identifier",
    )

    severity: RuleSeverity = Field(
        default=RuleSeverity.ERROR,
        description="Severity level for rule violations",
    )

    when: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Condition expression for when the rule applies",
    )

    check: Dict[str, Any] = Field(
        default_factory=dict,
        description="Validation expression that must evaluate to true",
    )

    message: str = Field(
        default="Rule validation failed",
        min_length=1,
        max_length=1024,
        description="Error message if rule fails",
    )

    message_template: Optional[str] = Field(
        default=None,
        max_length=2048,
        description="Template with {{ var }} placeholders",
    )

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "rule_id": "scope_sum_check",
                    "severity": "error",
                    "check": {"$eq": ["$scope1 + $scope2", "$total"]},
                    "message": "Scope 1 + Scope 2 must equal total",
                }
            ]
        },
    )

    @field_validator("rule_id")
    @classmethod
    def validate_rule_id(cls, v: str) -> str:
        """Validate rule_id format."""
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_-]*$", v):
            raise ValueError(
                f"Invalid rule_id '{v}'. Must start with a letter and contain "
                "only letters, numbers, underscores, and hyphens."
            )
        return v

    def has_condition(self) -> bool:
        """Check if this rule has a when condition."""
        return self.when is not None

    def is_blocking(self) -> bool:
        """Check if this rule blocks validation on failure."""
        return self.severity == RuleSeverity.ERROR


class GreenLangExtensions(BaseModel):
    """
    GreenLang-specific schema extensions.

    Container for all GreenLang extensions that can be attached to
    schema nodes. These extensions provide domain-specific functionality
    for sustainability reporting.

    Attributes:
        unit: Unit specification for numeric fields with physical units.
        dimension: Standalone dimension specification (alternative to unit.dimension).
        rules: List of cross-field validation rules.
        aliases: Map of alternative field names to canonical names.
        deprecated: Deprecation information for the field.
        renamed_from: Previous field name (for migration support).

    Example:
        >>> extensions = GreenLangExtensions(
        ...     unit=UnitSpec(dimension="energy", canonical="kWh"),
        ...     rules=[RuleBinding(rule_id="positive_value", ...)],
        ...     aliases={"energyConsumption": "energy_consumption"}
        ... )
    """

    unit: Optional[UnitSpec] = Field(
        default=None,
        description="Unit specification for numeric fields",
    )

    dimension: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Physical dimension (alternative to unit.dimension)",
    )

    rules: List[RuleBinding] = Field(
        default_factory=list,
        max_length=100,
        description="Cross-field validation rules",
    )

    aliases: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of alternative names to canonical names",
    )

    deprecated: Optional[DeprecationInfo] = Field(
        default=None,
        description="Deprecation information",
    )

    renamed_from: Optional[str] = Field(
        default=None,
        max_length=256,
        description="Previous field name for migration",
    )

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    @field_validator("dimension")
    @classmethod
    def validate_dimension(cls, v: Optional[str]) -> Optional[str]:
        """Validate dimension format."""
        if v is not None and not re.match(r"^[a-z][a-z0-9_]*$", v):
            raise ValueError(
                f"Invalid dimension '{v}'. Must be lowercase alphanumeric."
            )
        return v

    @field_validator("aliases")
    @classmethod
    def validate_aliases(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Validate alias keys and values."""
        for alias, canonical in v.items():
            if not alias or not canonical:
                raise ValueError("Alias keys and values cannot be empty.")
            if alias == canonical:
                raise ValueError(f"Alias '{alias}' cannot map to itself.")
        return v

    def has_unit(self) -> bool:
        """Check if unit specification is present."""
        return self.unit is not None

    def has_rules(self) -> bool:
        """Check if any rules are defined."""
        return len(self.rules) > 0

    def has_aliases(self) -> bool:
        """Check if any aliases are defined."""
        return len(self.aliases) > 0

    def is_deprecated(self) -> bool:
        """Check if field is deprecated."""
        return self.deprecated is not None

    def get_effective_dimension(self) -> Optional[str]:
        """Get dimension from unit spec or standalone dimension."""
        if self.unit:
            return self.unit.dimension
        return self.dimension

    def merge_with(self, other: "GreenLangExtensions") -> "GreenLangExtensions":
        """
        Merge with another extensions object (for inheritance).

        The current object takes precedence over the other.
        """
        merged_rules = list(self.rules) + [r for r in other.rules if r not in self.rules]
        merged_aliases = {**other.aliases, **self.aliases}

        return GreenLangExtensions(
            unit=self.unit or other.unit,
            dimension=self.dimension or other.dimension,
            rules=merged_rules,
            aliases=merged_aliases,
            deprecated=self.deprecated or other.deprecated,
            renamed_from=self.renamed_from or other.renamed_from,
        )


# =============================================================================
# Schema AST Node Models
# =============================================================================


class SchemaNode(BaseModel):
    """
    Base class for all schema AST nodes.

    Every node in the AST has a unique identifier and optional source
    location for error reporting. Nodes are immutable after creation.

    Attributes:
        node_id: Unique identifier for this node (typically derived from JSON path).
        location: Source location string (e.g., "file.yaml:10:5").

    Example:
        >>> node = TypeNode(node_id="/properties/name", type="string")
    """

    node_id: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="Unique identifier for this node",
    )

    location: Optional[str] = Field(
        default=None,
        max_length=512,
        description="Source location (e.g., 'file.yaml:10:5')",
    )

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize node to dictionary for debugging.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return self.model_dump(mode="json", exclude_none=True)

    def get_path_segments(self) -> List[str]:
        """
        Get path segments from node_id.

        Returns:
            List of path segments (empty list for root).

        Example:
            >>> node = TypeNode(node_id="/properties/name")
            >>> node.get_path_segments()
            ['properties', 'name']
        """
        if self.node_id == "/" or self.node_id == "root":
            return []
        # Handle JSON Pointer format
        if self.node_id.startswith("/"):
            return [seg for seg in self.node_id.split("/") if seg]
        return self.node_id.split("/")


class TypeNode(SchemaNode):
    """
    Base type definition node.

    Represents any type definition in the schema. Specialized subclasses
    provide additional constraints for specific types.

    Attributes:
        type: JSON Schema type(s) - single string or list for union types.
        const: Constant value requirement.
        default: Default value when field is missing.
        examples: Example values for documentation.
        gl_extensions: GreenLang-specific extensions.

    Example:
        >>> node = TypeNode(
        ...     node_id="/properties/status",
        ...     type="string",
        ...     const="active"
        ... )
    """

    type: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="JSON Schema type(s)",
    )

    const: Optional[Any] = Field(
        default=None,
        description="Constant value requirement",
    )

    default: Optional[Any] = Field(
        default=None,
        description="Default value when field is missing",
    )

    examples: List[Any] = Field(
        default_factory=list,
        max_length=20,
        description="Example values for documentation",
    )

    gl_extensions: Optional[GreenLangExtensions] = Field(
        default=None,
        description="GreenLang-specific extensions",
    )

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: Optional[Union[str, List[str]]]) -> Optional[Union[str, List[str]]]:
        """Validate type values are valid JSON Schema types."""
        if v is None:
            return v

        types_to_check = [v] if isinstance(v, str) else v
        for t in types_to_check:
            if t not in JSON_SCHEMA_TYPES:
                raise ValueError(
                    f"Invalid type '{t}'. Must be one of: {', '.join(sorted(JSON_SCHEMA_TYPES))}"
                )
        return v

    def is_type(self, type_name: str) -> bool:
        """Check if this node has the specified type."""
        if self.type is None:
            return False
        if isinstance(self.type, str):
            return self.type == type_name
        return type_name in self.type

    def is_nullable(self) -> bool:
        """Check if this type allows null values."""
        return self.is_type("null")

    def has_default(self) -> bool:
        """Check if a default value is defined."""
        return self.default is not None

    def has_const(self) -> bool:
        """Check if a const value is defined."""
        return self.const is not None

    def get_extensions(self) -> GreenLangExtensions:
        """Get extensions, returning empty extensions if None."""
        return self.gl_extensions or GreenLangExtensions()


class ObjectTypeNode(TypeNode):
    """
    Object type with properties.

    Represents a JSON object with named properties and constraints.

    Attributes:
        properties: Map of property name to TypeNode.
        required: List of required property names.
        additional_properties: Whether additional properties are allowed,
            or a TypeNode defining their schema.
        property_names: Constraints on property names (StringTypeNode).
        min_properties: Minimum number of properties.
        max_properties: Maximum number of properties.
        dependencies: Legacy property dependencies (JSON Schema Draft 4-7).
        dependent_required: Property presence dependencies (Draft 2019+).
        dependent_schemas: Property schema dependencies (Draft 2019+).
        pattern_properties: Regex-based property definitions.

    Example:
        >>> obj = ObjectTypeNode(
        ...     node_id="/",
        ...     type="object",
        ...     properties={"name": StringTypeNode(node_id="/properties/name")},
        ...     required=["name"]
        ... )
    """

    properties: Dict[str, "TypeNode"] = Field(
        default_factory=dict,
        description="Map of property name to TypeNode",
    )

    required: List[str] = Field(
        default_factory=list,
        description="List of required property names",
    )

    additional_properties: Union[bool, "TypeNode"] = Field(
        default=True,
        description="Additional properties policy or schema",
    )

    property_names: Optional["StringTypeNode"] = Field(
        default=None,
        description="Constraints on property names",
    )

    min_properties: Optional[int] = Field(
        default=None,
        ge=0,
        description="Minimum number of properties",
    )

    max_properties: Optional[int] = Field(
        default=None,
        ge=0,
        description="Maximum number of properties",
    )

    dependencies: Dict[str, Union[List[str], "TypeNode"]] = Field(
        default_factory=dict,
        description="Legacy property dependencies",
    )

    dependent_required: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Property presence dependencies",
    )

    dependent_schemas: Dict[str, "TypeNode"] = Field(
        default_factory=dict,
        description="Property schema dependencies",
    )

    pattern_properties: Dict[str, "TypeNode"] = Field(
        default_factory=dict,
        description="Regex-based property definitions",
    )

    @model_validator(mode="after")
    def validate_object_constraints(self) -> "ObjectTypeNode":
        """Validate object-specific constraints."""
        # Validate min/max properties
        if (
            self.min_properties is not None
            and self.max_properties is not None
            and self.min_properties > self.max_properties
        ):
            raise ValueError(
                f"min_properties ({self.min_properties}) cannot exceed "
                f"max_properties ({self.max_properties})"
            )

        # Validate required properties exist in properties
        for req in self.required:
            if req not in self.properties:
                # This is a warning condition, not an error
                # The property might be defined via additionalProperties
                pass

        return self

    def has_property(self, name: str) -> bool:
        """Check if a property is defined."""
        return name in self.properties

    def is_required(self, name: str) -> bool:
        """Check if a property is required."""
        return name in self.required

    def allows_additional_properties(self) -> bool:
        """Check if additional properties are allowed."""
        if isinstance(self.additional_properties, bool):
            return self.additional_properties
        return True  # TypeNode means additional properties with schema

    def get_property_names(self) -> List[str]:
        """Get list of defined property names."""
        return list(self.properties.keys())


class ArrayTypeNode(TypeNode):
    """
    Array type with items.

    Represents a JSON array with item constraints.

    Attributes:
        items: TypeNode for array items (applies to all items).
        prefix_items: TypeNodes for positional items (tuple validation).
        contains: TypeNode that at least one item must match.
        min_items: Minimum number of items.
        max_items: Maximum number of items.
        unique_items: Whether items must be unique.
        min_contains: Minimum number of items matching contains.
        max_contains: Maximum number of items matching contains.

    Example:
        >>> arr = ArrayTypeNode(
        ...     node_id="/properties/values",
        ...     type="array",
        ...     items=NumericTypeNode(node_id="/properties/values/items"),
        ...     min_items=1,
        ...     unique_items=True
        ... )
    """

    items: Optional["TypeNode"] = Field(
        default=None,
        description="TypeNode for array items",
    )

    prefix_items: List["TypeNode"] = Field(
        default_factory=list,
        description="TypeNodes for positional items (tuple validation)",
    )

    contains: Optional["TypeNode"] = Field(
        default=None,
        description="TypeNode that at least one item must match",
    )

    min_items: Optional[int] = Field(
        default=None,
        ge=0,
        description="Minimum number of items",
    )

    max_items: Optional[int] = Field(
        default=None,
        ge=0,
        description="Maximum number of items",
    )

    unique_items: bool = Field(
        default=False,
        description="Whether items must be unique",
    )

    min_contains: Optional[int] = Field(
        default=None,
        ge=0,
        description="Minimum number of items matching contains",
    )

    max_contains: Optional[int] = Field(
        default=None,
        ge=0,
        description="Maximum number of items matching contains",
    )

    @model_validator(mode="after")
    def validate_array_constraints(self) -> "ArrayTypeNode":
        """Validate array-specific constraints."""
        # Validate min/max items
        if (
            self.min_items is not None
            and self.max_items is not None
            and self.min_items > self.max_items
        ):
            raise ValueError(
                f"min_items ({self.min_items}) cannot exceed max_items ({self.max_items})"
            )

        # Validate min/max contains
        if (
            self.min_contains is not None
            and self.max_contains is not None
            and self.min_contains > self.max_contains
        ):
            raise ValueError(
                f"min_contains ({self.min_contains}) cannot exceed "
                f"max_contains ({self.max_contains})"
            )

        # Validate contains requirements
        if (self.min_contains is not None or self.max_contains is not None) and self.contains is None:
            raise ValueError("min_contains/max_contains require 'contains' to be defined")

        return self

    def is_tuple(self) -> bool:
        """Check if this is a tuple validation (prefix_items defined)."""
        return len(self.prefix_items) > 0

    def has_item_schema(self) -> bool:
        """Check if items schema is defined."""
        return self.items is not None

    def has_contains(self) -> bool:
        """Check if contains constraint is defined."""
        return self.contains is not None


class StringTypeNode(TypeNode):
    """
    String type with constraints.

    Represents a JSON string with length, pattern, and format constraints.

    Attributes:
        min_length: Minimum string length.
        max_length: Maximum string length.
        pattern: Regex pattern the string must match.
        format: Semantic format (e.g., "date", "email", "uri").
        content_encoding: Content encoding (e.g., "base64").
        content_media_type: Media type of encoded content.

    Example:
        >>> string = StringTypeNode(
        ...     node_id="/properties/email",
        ...     type="string",
        ...     format="email",
        ...     max_length=254
        ... )
    """

    min_length: Optional[int] = Field(
        default=None,
        ge=0,
        description="Minimum string length",
    )

    max_length: Optional[int] = Field(
        default=None,
        ge=0,
        description="Maximum string length",
    )

    pattern: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Regex pattern the string must match",
    )

    format: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Semantic format (e.g., 'date', 'email', 'uri')",
    )

    content_encoding: Optional[str] = Field(
        default=None,
        max_length=32,
        description="Content encoding (e.g., 'base64')",
    )

    content_media_type: Optional[str] = Field(
        default=None,
        max_length=128,
        description="Media type of encoded content",
    )

    @model_validator(mode="after")
    def validate_string_constraints(self) -> "StringTypeNode":
        """Validate string-specific constraints."""
        if (
            self.min_length is not None
            and self.max_length is not None
            and self.min_length > self.max_length
        ):
            raise ValueError(
                f"min_length ({self.min_length}) cannot exceed max_length ({self.max_length})"
            )
        return self

    @field_validator("pattern")
    @classmethod
    def validate_pattern(cls, v: Optional[str]) -> Optional[str]:
        """Validate pattern is a valid regex."""
        if v is not None:
            try:
                re.compile(v)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {e}")
        return v

    def has_pattern(self) -> bool:
        """Check if pattern constraint is defined."""
        return self.pattern is not None

    def has_format(self) -> bool:
        """Check if format constraint is defined."""
        return self.format is not None


class NumericTypeNode(TypeNode):
    """
    Number/Integer type with constraints.

    Represents a JSON number or integer with range and divisibility constraints.

    Attributes:
        minimum: Minimum value (inclusive).
        maximum: Maximum value (inclusive).
        exclusive_minimum: Exclusive minimum value.
        exclusive_maximum: Exclusive maximum value.
        multiple_of: Value must be a multiple of this number.

    Example:
        >>> number = NumericTypeNode(
        ...     node_id="/properties/quantity",
        ...     type="integer",
        ...     minimum=0,
        ...     maximum=1000
        ... )
    """

    minimum: Optional[float] = Field(
        default=None,
        description="Minimum value (inclusive)",
    )

    maximum: Optional[float] = Field(
        default=None,
        description="Maximum value (inclusive)",
    )

    exclusive_minimum: Optional[float] = Field(
        default=None,
        description="Exclusive minimum value",
    )

    exclusive_maximum: Optional[float] = Field(
        default=None,
        description="Exclusive maximum value",
    )

    multiple_of: Optional[float] = Field(
        default=None,
        gt=0,
        description="Value must be a multiple of this number",
    )

    @model_validator(mode="after")
    def validate_numeric_constraints(self) -> "NumericTypeNode":
        """Validate numeric-specific constraints."""
        # Check minimum vs maximum
        if self.minimum is not None and self.maximum is not None:
            if self.minimum > self.maximum:
                raise ValueError(
                    f"minimum ({self.minimum}) cannot exceed maximum ({self.maximum})"
                )

        # Check exclusive bounds don't contradict inclusive bounds
        if self.minimum is not None and self.exclusive_minimum is not None:
            if self.exclusive_minimum >= self.minimum:
                pass  # Both can be specified, exclusive takes precedence

        if self.maximum is not None and self.exclusive_maximum is not None:
            if self.exclusive_maximum <= self.maximum:
                pass  # Both can be specified, exclusive takes precedence

        return self

    def is_integer(self) -> bool:
        """Check if this is an integer type."""
        return self.is_type("integer")

    def has_range(self) -> bool:
        """Check if any range constraint is defined."""
        return any([
            self.minimum is not None,
            self.maximum is not None,
            self.exclusive_minimum is not None,
            self.exclusive_maximum is not None,
        ])

    def get_effective_minimum(self) -> Optional[float]:
        """Get effective minimum value (exclusive takes precedence)."""
        if self.exclusive_minimum is not None:
            return self.exclusive_minimum
        return self.minimum

    def get_effective_maximum(self) -> Optional[float]:
        """Get effective maximum value (exclusive takes precedence)."""
        if self.exclusive_maximum is not None:
            return self.exclusive_maximum
        return self.maximum


class BooleanTypeNode(TypeNode):
    """
    Boolean type.

    Represents a JSON boolean value.

    Example:
        >>> boolean = BooleanTypeNode(
        ...     node_id="/properties/active",
        ...     type="boolean",
        ...     default=True
        ... )
    """

    pass


class NullTypeNode(TypeNode):
    """
    Null type.

    Represents a JSON null value.

    Example:
        >>> null = NullTypeNode(
        ...     node_id="/properties/deleted_at",
        ...     type="null"
        ... )
    """

    pass


class RefNode(TypeNode):
    """
    Reference to another schema ($ref).

    Represents a JSON Schema $ref that points to another schema definition.
    The reference is resolved during compilation.

    Attributes:
        ref: The $ref value (e.g., "#/$defs/Address", "gl://schemas/...").
        resolved: The resolved TypeNode (populated after resolution).

    Example:
        >>> ref = RefNode(
        ...     node_id="/properties/address",
        ...     ref="#/$defs/Address"
        ... )
    """

    ref: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        alias="$ref",
        description="Reference URI ($ref value)",
    )

    resolved: Optional["TypeNode"] = Field(
        default=None,
        description="Resolved TypeNode (populated after resolution)",
        exclude=True,
    )

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        populate_by_name=True,
    )

    @field_validator("ref")
    @classmethod
    def validate_ref(cls, v: str) -> str:
        """Validate ref format."""
        if not v.strip():
            raise ValueError("$ref cannot be empty or whitespace.")
        return v

    def is_local_ref(self) -> bool:
        """Check if this is a local reference (starts with #)."""
        return self.ref.startswith("#")

    def is_external_ref(self) -> bool:
        """Check if this is an external reference."""
        return not self.is_local_ref()

    def is_greenlang_ref(self) -> bool:
        """Check if this is a GreenLang registry reference."""
        return self.ref.startswith("gl://")

    def is_resolved(self) -> bool:
        """Check if this reference has been resolved."""
        return self.resolved is not None

    def get_local_path(self) -> Optional[str]:
        """
        Get local reference path.

        Returns:
            Path after # for local refs, None for external refs.

        Example:
            >>> ref = RefNode(node_id="ref1", ref="#/$defs/Address")
            >>> ref.get_local_path()
            '/$defs/Address'
        """
        if self.is_local_ref():
            return self.ref[1:]  # Remove leading #
        return None


class CompositionNode(TypeNode):
    """
    Composition schemas (allOf, anyOf, oneOf, not, if/then/else).

    Represents JSON Schema composition keywords for combining schemas.

    Attributes:
        all_of: All schemas must match (allOf).
        any_of: At least one schema must match (anyOf).
        one_of: Exactly one schema must match (oneOf).
        not_: Schema must not match (not).
        if_: Condition schema (if).
        then_: Schema to apply if condition matches (then).
        else_: Schema to apply if condition doesn't match (else).

    Example:
        >>> comp = CompositionNode(
        ...     node_id="/properties/value",
        ...     any_of=[
        ...         StringTypeNode(node_id="/properties/value/anyOf/0"),
        ...         NumericTypeNode(node_id="/properties/value/anyOf/1")
        ...     ]
        ... )
    """

    all_of: List["TypeNode"] = Field(
        default_factory=list,
        alias="allOf",
        description="All schemas must match",
    )

    any_of: List["TypeNode"] = Field(
        default_factory=list,
        alias="anyOf",
        description="At least one schema must match",
    )

    one_of: List["TypeNode"] = Field(
        default_factory=list,
        alias="oneOf",
        description="Exactly one schema must match",
    )

    not_: Optional["TypeNode"] = Field(
        default=None,
        alias="not",
        description="Schema must not match",
    )

    if_: Optional["TypeNode"] = Field(
        default=None,
        alias="if",
        description="Condition schema",
    )

    then_: Optional["TypeNode"] = Field(
        default=None,
        alias="then",
        description="Schema to apply if condition matches",
    )

    else_: Optional["TypeNode"] = Field(
        default=None,
        alias="else",
        description="Schema to apply if condition doesn't match",
    )

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        populate_by_name=True,
    )

    @model_validator(mode="after")
    def validate_composition(self) -> "CompositionNode":
        """Validate composition constraints."""
        # then/else without if is meaningless but allowed
        if (self.then_ is not None or self.else_ is not None) and self.if_ is None:
            pass  # JSON Schema allows this, just ignored

        return self

    def has_all_of(self) -> bool:
        """Check if allOf is defined."""
        return len(self.all_of) > 0

    def has_any_of(self) -> bool:
        """Check if anyOf is defined."""
        return len(self.any_of) > 0

    def has_one_of(self) -> bool:
        """Check if oneOf is defined."""
        return len(self.one_of) > 0

    def has_not(self) -> bool:
        """Check if not is defined."""
        return self.not_ is not None

    def has_conditional(self) -> bool:
        """Check if if/then/else is defined."""
        return self.if_ is not None

    def get_composition_type(self) -> Optional[str]:
        """
        Get the primary composition type.

        Returns:
            "allOf", "anyOf", "oneOf", "not", "conditional", or None.
        """
        if self.has_all_of():
            return "allOf"
        if self.has_any_of():
            return "anyOf"
        if self.has_one_of():
            return "oneOf"
        if self.has_not():
            return "not"
        if self.has_conditional():
            return "conditional"
        return None


class EnumTypeNode(TypeNode):
    """
    Enum constraint.

    Represents a JSON Schema enum constraint that restricts values
    to a specific set.

    Attributes:
        enum: List of allowed values.

    Example:
        >>> enum = EnumTypeNode(
        ...     node_id="/properties/status",
        ...     type="string",
        ...     enum=["pending", "active", "completed"]
        ... )
    """

    enum: List[Any] = Field(
        default_factory=list,
        min_length=1,
        max_length=10000,
        description="List of allowed values",
    )

    @field_validator("enum")
    @classmethod
    def validate_enum(cls, v: List[Any]) -> List[Any]:
        """Validate enum has at least one value."""
        if len(v) == 0:
            raise ValueError("Enum must have at least one value.")
        return v

    def contains_value(self, value: Any) -> bool:
        """Check if value is in enum."""
        return value in self.enum

    def get_value_count(self) -> int:
        """Get number of enum values."""
        return len(self.enum)


# =============================================================================
# Schema Document (Root Node)
# =============================================================================


class SchemaDocument(SchemaNode):
    """
    Root schema document.

    Represents a complete GreenLang schema document with metadata,
    root type definition, and local definitions.

    Attributes:
        schema_id: Unique schema identifier (e.g., "emissions/activity").
        version: Schema version using semver (e.g., "1.3.0").
        dialect: JSON Schema dialect URI.
        title: Human-readable schema title.
        description: Schema description.
        root: Root type definition.
        definitions: Local schema definitions ($defs).
        gl_extensions: Document-level GreenLang extensions.

    Example:
        >>> doc = SchemaDocument(
        ...     node_id="root",
        ...     schema_id="emissions/activity",
        ...     version="1.3.0",
        ...     title="Activity Data Schema",
        ...     root=ObjectTypeNode(node_id="/", type="object", ...)
        ... )
    """

    schema_id: str = Field(
        ...,
        min_length=1,
        max_length=253,
        description="Unique schema identifier",
    )

    version: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Schema version (semver)",
    )

    dialect: str = Field(
        default=JSON_SCHEMA_DRAFT_2020_12,
        max_length=256,
        description="JSON Schema dialect URI",
    )

    title: Optional[str] = Field(
        default=None,
        max_length=256,
        description="Human-readable schema title",
    )

    description: Optional[str] = Field(
        default=None,
        max_length=4096,
        description="Schema description",
    )

    root: "TypeNode" = Field(
        ...,
        description="Root type definition",
    )

    definitions: Dict[str, "TypeNode"] = Field(
        default_factory=dict,
        description="Local schema definitions ($defs)",
    )

    gl_extensions: GreenLangExtensions = Field(
        default_factory=GreenLangExtensions,
        description="Document-level GreenLang extensions",
    )

    @field_validator("schema_id")
    @classmethod
    def validate_schema_id(cls, v: str) -> str:
        """Validate schema_id format."""
        if not SCHEMA_ID_PATTERN.match(v):
            raise ValueError(
                f"Invalid schema_id '{v}'. Must be lowercase alphanumeric "
                "with underscores, using '/' for namespacing."
            )
        return v

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate version follows semver."""
        if not SEMVER_PATTERN.match(v):
            raise ValueError(
                f"Invalid version '{v}'. Must follow semantic versioning "
                "(e.g., '1.0.0', '2.0.0-beta.1')."
            )
        return v

    def get_definition(self, name: str) -> Optional["TypeNode"]:
        """Get a definition by name."""
        return self.definitions.get(name)

    def has_definition(self, name: str) -> bool:
        """Check if a definition exists."""
        return name in self.definitions

    def get_definition_names(self) -> List[str]:
        """Get list of definition names."""
        return list(self.definitions.keys())

    def to_uri(self) -> str:
        """
        Convert to GreenLang schema URI.

        Returns:
            Schema URI in format: gl://schemas/{schema_id}@{version}
        """
        return f"gl://schemas/{self.schema_id}@{self.version}"

    def compute_hash(self) -> str:
        """
        Compute SHA-256 hash of the schema for caching.

        Returns:
            Hex-encoded SHA-256 hash of canonical JSON representation.
        """
        import json

        # Sort keys for deterministic output
        json_str = json.dumps(
            self.model_dump(mode="json", exclude={"location"}),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


# =============================================================================
# Model Rebuilding for Forward References
# =============================================================================

# Rebuild models to resolve forward references
ObjectTypeNode.model_rebuild()
ArrayTypeNode.model_rebuild()
RefNode.model_rebuild()
CompositionNode.model_rebuild()
SchemaDocument.model_rebuild()


# =============================================================================
# Helper Functions
# =============================================================================


def create_node_id(path: str, index: int = 0) -> str:
    """
    Create a unique node ID from a JSON path.

    Args:
        path: JSON Pointer path (e.g., "/properties/name").
        index: Optional index for disambiguation.

    Returns:
        Unique node ID string.

    Example:
        >>> create_node_id("/properties/name")
        '/properties/name'
        >>> create_node_id("/items", 0)
        '/items[0]'
    """
    if index > 0:
        return f"{path}[{index}]"
    return path


def create_unique_node_id(prefix: str = "node") -> str:
    """
    Create a globally unique node ID using UUID.

    Args:
        prefix: Prefix for the node ID.

    Returns:
        Unique node ID string.

    Example:
        >>> id = create_unique_node_id("ref")
        >>> id.startswith("ref_")
        True
    """
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def parse_type_node(
    data: Dict[str, Any],
    path: str,
    parent_extensions: Optional[GreenLangExtensions] = None,
) -> TypeNode:
    """
    Parse a JSON Schema type definition into the appropriate TypeNode.

    This function analyzes the schema data and creates the correct
    TypeNode subclass based on the content.

    Args:
        data: JSON Schema definition as a dictionary.
        path: JSON Pointer path for this node.
        parent_extensions: Extensions inherited from parent node.

    Returns:
        Appropriate TypeNode subclass instance.

    Raises:
        ValueError: If the schema data is invalid.

    Example:
        >>> data = {"type": "string", "minLength": 1, "maxLength": 100}
        >>> node = parse_type_node(data, "/properties/name")
        >>> isinstance(node, StringTypeNode)
        True
    """
    # Parse GreenLang extensions
    gl_extensions = _parse_gl_extensions(data)
    if parent_extensions:
        gl_extensions = gl_extensions.merge_with(parent_extensions)

    # Common fields
    common_kwargs: Dict[str, Any] = {
        "node_id": path,
        "const": data.get("const"),
        "default": data.get("default"),
        "examples": data.get("examples", []),
        "gl_extensions": gl_extensions if _has_extensions(gl_extensions) else None,
    }

    # Check for $ref first (takes precedence)
    if "$ref" in data:
        return RefNode(
            **common_kwargs,
            ref=data["$ref"],
        )

    # Check for composition keywords
    if any(k in data for k in ("allOf", "anyOf", "oneOf", "not", "if")):
        return _parse_composition_node(data, path, common_kwargs)

    # Check for enum
    if "enum" in data:
        return EnumTypeNode(
            **common_kwargs,
            type=data.get("type"),
            enum=data["enum"],
        )

    # Determine type from explicit type or infer from keywords
    schema_type = data.get("type")

    # Handle union types (array of types)
    if isinstance(schema_type, list):
        # For union types, create a basic TypeNode
        return TypeNode(**common_kwargs, type=schema_type)

    # Handle single type
    if schema_type == "object" or (
        schema_type is None and any(k in data for k in ("properties", "required", "additionalProperties"))
    ):
        return _parse_object_node(data, path, common_kwargs)

    if schema_type == "array" or (
        schema_type is None and any(k in data for k in ("items", "prefixItems", "contains"))
    ):
        return _parse_array_node(data, path, common_kwargs)

    if schema_type == "string" or (
        schema_type is None and any(k in data for k in ("minLength", "maxLength", "pattern", "format"))
    ):
        return _parse_string_node(data, path, common_kwargs)

    if schema_type in ("number", "integer") or (
        schema_type is None and any(k in data for k in ("minimum", "maximum", "multipleOf"))
    ):
        return _parse_numeric_node(data, path, common_kwargs, schema_type or "number")

    if schema_type == "boolean":
        return BooleanTypeNode(**common_kwargs, type="boolean")

    if schema_type == "null":
        return NullTypeNode(**common_kwargs, type="null")

    # Default: generic TypeNode
    return TypeNode(**common_kwargs, type=schema_type)


def _parse_gl_extensions(data: Dict[str, Any]) -> GreenLangExtensions:
    """Parse GreenLang extensions from schema data."""
    unit = None
    if GL_UNIT_KEY in data:
        unit_data = data[GL_UNIT_KEY]
        if isinstance(unit_data, dict):
            unit = UnitSpec(
                dimension=unit_data.get("dimension", ""),
                canonical=unit_data.get("canonical", ""),
                allowed=unit_data.get("allowed", []),
            )

    rules = []
    if GL_RULES_KEY in data:
        rules_data = data[GL_RULES_KEY]
        if isinstance(rules_data, list):
            for rule_data in rules_data:
                if isinstance(rule_data, dict):
                    rules.append(
                        RuleBinding(
                            rule_id=rule_data.get("rule_id", ""),
                            rule_pack=rule_data.get("rule_pack"),
                            severity=RuleSeverity(rule_data.get("severity", "error")),
                            when=rule_data.get("when"),
                            check=rule_data.get("check", {}),
                            message=rule_data.get("message", ""),
                            message_template=rule_data.get("message_template"),
                        )
                    )

    deprecated = None
    if GL_DEPRECATED_KEY in data:
        dep_data = data[GL_DEPRECATED_KEY]
        if isinstance(dep_data, dict):
            deprecated = DeprecationInfo(
                since_version=dep_data.get("since_version", ""),
                message=dep_data.get("message", ""),
                replacement=dep_data.get("replacement"),
                removal_version=dep_data.get("removal_version"),
            )

    return GreenLangExtensions(
        unit=unit,
        dimension=data.get(GL_DIMENSION_KEY),
        rules=rules,
        aliases=data.get(GL_ALIASES_KEY, {}),
        deprecated=deprecated,
        renamed_from=data.get(GL_RENAMED_FROM_KEY),
    )


def _has_extensions(ext: GreenLangExtensions) -> bool:
    """Check if extensions object has any non-default values."""
    return (
        ext.unit is not None
        or ext.dimension is not None
        or len(ext.rules) > 0
        or len(ext.aliases) > 0
        or ext.deprecated is not None
        or ext.renamed_from is not None
    )


def _parse_composition_node(
    data: Dict[str, Any],
    path: str,
    common_kwargs: Dict[str, Any],
) -> CompositionNode:
    """Parse composition schema (allOf, anyOf, oneOf, not, if/then/else)."""
    all_of = []
    if "allOf" in data:
        for i, item in enumerate(data["allOf"]):
            all_of.append(parse_type_node(item, f"{path}/allOf/{i}"))

    any_of = []
    if "anyOf" in data:
        for i, item in enumerate(data["anyOf"]):
            any_of.append(parse_type_node(item, f"{path}/anyOf/{i}"))

    one_of = []
    if "oneOf" in data:
        for i, item in enumerate(data["oneOf"]):
            one_of.append(parse_type_node(item, f"{path}/oneOf/{i}"))

    not_schema = None
    if "not" in data:
        not_schema = parse_type_node(data["not"], f"{path}/not")

    if_schema = None
    if "if" in data:
        if_schema = parse_type_node(data["if"], f"{path}/if")

    then_schema = None
    if "then" in data:
        then_schema = parse_type_node(data["then"], f"{path}/then")

    else_schema = None
    if "else" in data:
        else_schema = parse_type_node(data["else"], f"{path}/else")

    return CompositionNode(
        **common_kwargs,
        type=data.get("type"),
        all_of=all_of,
        any_of=any_of,
        one_of=one_of,
        not_=not_schema,
        if_=if_schema,
        then_=then_schema,
        else_=else_schema,
    )


def _parse_object_node(
    data: Dict[str, Any],
    path: str,
    common_kwargs: Dict[str, Any],
) -> ObjectTypeNode:
    """Parse object type schema."""
    properties = {}
    if "properties" in data:
        for name, prop_data in data["properties"].items():
            properties[name] = parse_type_node(prop_data, f"{path}/properties/{name}")

    additional_properties: Union[bool, TypeNode] = data.get("additionalProperties", True)
    if isinstance(additional_properties, dict):
        additional_properties = parse_type_node(
            additional_properties, f"{path}/additionalProperties"
        )

    property_names = None
    if "propertyNames" in data:
        property_names = parse_type_node(data["propertyNames"], f"{path}/propertyNames")  # type: ignore

    pattern_properties = {}
    if "patternProperties" in data:
        for pattern, prop_data in data["patternProperties"].items():
            pattern_properties[pattern] = parse_type_node(
                prop_data, f"{path}/patternProperties/{pattern}"
            )

    dependencies = {}
    if "dependencies" in data:
        for name, dep_data in data["dependencies"].items():
            if isinstance(dep_data, list):
                dependencies[name] = dep_data
            else:
                dependencies[name] = parse_type_node(dep_data, f"{path}/dependencies/{name}")

    dependent_required = data.get("dependentRequired", {})

    dependent_schemas = {}
    if "dependentSchemas" in data:
        for name, schema_data in data["dependentSchemas"].items():
            dependent_schemas[name] = parse_type_node(
                schema_data, f"{path}/dependentSchemas/{name}"
            )

    return ObjectTypeNode(
        **common_kwargs,
        type="object",
        properties=properties,
        required=data.get("required", []),
        additional_properties=additional_properties,
        property_names=property_names,
        min_properties=data.get("minProperties"),
        max_properties=data.get("maxProperties"),
        dependencies=dependencies,
        dependent_required=dependent_required,
        dependent_schemas=dependent_schemas,
        pattern_properties=pattern_properties,
    )


def _parse_array_node(
    data: Dict[str, Any],
    path: str,
    common_kwargs: Dict[str, Any],
) -> ArrayTypeNode:
    """Parse array type schema."""
    items = None
    if "items" in data:
        items = parse_type_node(data["items"], f"{path}/items")

    prefix_items = []
    if "prefixItems" in data:
        for i, item_data in enumerate(data["prefixItems"]):
            prefix_items.append(parse_type_node(item_data, f"{path}/prefixItems/{i}"))

    contains = None
    if "contains" in data:
        contains = parse_type_node(data["contains"], f"{path}/contains")

    return ArrayTypeNode(
        **common_kwargs,
        type="array",
        items=items,
        prefix_items=prefix_items,
        contains=contains,
        min_items=data.get("minItems"),
        max_items=data.get("maxItems"),
        unique_items=data.get("uniqueItems", False),
        min_contains=data.get("minContains"),
        max_contains=data.get("maxContains"),
    )


def _parse_string_node(
    data: Dict[str, Any],
    path: str,
    common_kwargs: Dict[str, Any],
) -> StringTypeNode:
    """Parse string type schema."""
    return StringTypeNode(
        **common_kwargs,
        type="string",
        min_length=data.get("minLength"),
        max_length=data.get("maxLength"),
        pattern=data.get("pattern"),
        format=data.get("format"),
        content_encoding=data.get("contentEncoding"),
        content_media_type=data.get("contentMediaType"),
    )


def _parse_numeric_node(
    data: Dict[str, Any],
    path: str,
    common_kwargs: Dict[str, Any],
    numeric_type: str,
) -> NumericTypeNode:
    """Parse number/integer type schema."""
    return NumericTypeNode(
        **common_kwargs,
        type=numeric_type,
        minimum=data.get("minimum"),
        maximum=data.get("maximum"),
        exclusive_minimum=data.get("exclusiveMinimum"),
        exclusive_maximum=data.get("exclusiveMaximum"),
        multiple_of=data.get("multipleOf"),
    )


def build_ast(
    schema_dict: Dict[str, Any],
    schema_id: str,
    version: str,
) -> SchemaDocument:
    """
    Build a complete AST from a parsed schema dictionary.

    This is the main entry point for converting a raw schema dictionary
    into a structured AST for compilation and validation.

    Args:
        schema_dict: The parsed schema as a dictionary.
        schema_id: The schema identifier.
        version: The schema version.

    Returns:
        SchemaDocument AST root.

    Raises:
        ValueError: If the schema is invalid.

    Example:
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {"name": {"type": "string"}},
        ...     "required": ["name"]
        ... }
        >>> doc = build_ast(schema, "test/schema", "1.0.0")
        >>> doc.schema_id
        'test/schema'
    """
    # Parse root type
    root = parse_type_node(schema_dict, "/")

    # Parse definitions ($defs)
    definitions = {}
    defs_data = schema_dict.get("$defs") or schema_dict.get("definitions") or {}
    for name, def_data in defs_data.items():
        definitions[name] = parse_type_node(def_data, f"/$defs/{name}")

    # Parse document-level extensions
    gl_extensions = _parse_gl_extensions(schema_dict)

    return SchemaDocument(
        node_id="root",
        schema_id=schema_id,
        version=version,
        dialect=schema_dict.get("$schema", JSON_SCHEMA_DRAFT_2020_12),
        title=schema_dict.get("title"),
        description=schema_dict.get("description"),
        root=root,
        definitions=definitions,
        gl_extensions=gl_extensions,
    )


def validate_ast(document: SchemaDocument) -> List[str]:
    """
    Validate an AST for internal consistency.

    Performs validation checks on the AST to ensure:
    - All required fields are present
    - Constraint values are consistent (min <= max)
    - GreenLang extensions are well-formed
    - References are syntactically valid

    Args:
        document: The schema document to validate.

    Returns:
        List of validation error messages (empty if valid).

    Example:
        >>> doc = build_ast({"type": "object"}, "test", "1.0.0")
        >>> errors = validate_ast(doc)
        >>> len(errors)
        0
    """
    errors: List[str] = []

    def validate_node(node: TypeNode, path: str) -> None:
        """Recursively validate a type node."""
        # Validate based on node type
        if isinstance(node, ObjectTypeNode):
            # Check required properties are defined (warning, not error)
            for req in node.required:
                if req not in node.properties:
                    # This is actually valid - properties may come from additionalProperties
                    pass

            # Validate nested properties
            for name, prop in node.properties.items():
                validate_node(prop, f"{path}/properties/{name}")

            # Validate pattern properties
            for pattern, prop in node.pattern_properties.items():
                # Validate pattern is valid regex
                try:
                    re.compile(pattern)
                except re.error as e:
                    errors.append(f"{path}/patternProperties/{pattern}: Invalid regex pattern: {e}")
                validate_node(prop, f"{path}/patternProperties/{pattern}")

        elif isinstance(node, ArrayTypeNode):
            if node.items:
                validate_node(node.items, f"{path}/items")
            for i, item in enumerate(node.prefix_items):
                validate_node(item, f"{path}/prefixItems/{i}")
            if node.contains:
                validate_node(node.contains, f"{path}/contains")

        elif isinstance(node, CompositionNode):
            for i, item in enumerate(node.all_of):
                validate_node(item, f"{path}/allOf/{i}")
            for i, item in enumerate(node.any_of):
                validate_node(item, f"{path}/anyOf/{i}")
            for i, item in enumerate(node.one_of):
                validate_node(item, f"{path}/oneOf/{i}")
            if node.not_:
                validate_node(node.not_, f"{path}/not")
            if node.if_:
                validate_node(node.if_, f"{path}/if")
            if node.then_:
                validate_node(node.then_, f"{path}/then")
            if node.else_:
                validate_node(node.else_, f"{path}/else")

        elif isinstance(node, RefNode):
            # Validate ref format
            if not node.ref:
                errors.append(f"{path}: Empty $ref value")
            elif node.is_local_ref():
                local_path = node.get_local_path()
                if local_path and not local_path.startswith("/"):
                    errors.append(f"{path}: Invalid local $ref path (must start with /)")

        # Validate GreenLang extensions
        if node.gl_extensions:
            ext = node.gl_extensions
            if ext.unit:
                if not ext.unit.dimension:
                    errors.append(f"{path}/$unit: Missing dimension")
                if not ext.unit.canonical:
                    errors.append(f"{path}/$unit: Missing canonical unit")

    # Validate root
    validate_node(document.root, "/")

    # Validate definitions
    for name, def_node in document.definitions.items():
        validate_node(def_node, f"/$defs/{name}")

    return errors


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Constants
    "JSON_SCHEMA_DRAFT_2020_12",
    "JSON_SCHEMA_TYPES",
    # Enums
    "RuleSeverity",
    # Extension Models
    "UnitSpec",
    "DeprecationInfo",
    "RuleBinding",
    "GreenLangExtensions",
    # AST Nodes
    "SchemaNode",
    "TypeNode",
    "ObjectTypeNode",
    "ArrayTypeNode",
    "StringTypeNode",
    "NumericTypeNode",
    "BooleanTypeNode",
    "NullTypeNode",
    "RefNode",
    "CompositionNode",
    "EnumTypeNode",
    "SchemaDocument",
    # Helper Functions
    "create_node_id",
    "create_unique_node_id",
    "parse_type_node",
    "build_ast",
    "validate_ast",
]
