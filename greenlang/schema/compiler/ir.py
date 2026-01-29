# -*- coding: utf-8 -*-
"""
Intermediate Representation (IR) for GL-FOUND-X-002.

This module defines the compiled schema Intermediate Representation (IR)
that is optimized for fast validation. The IR flattens property maps for
O(1) lookup and precompiles regex patterns with safety metadata.

Key Features:
    - Flattened property maps for O(1) path lookup
    - Precompiled regex patterns with ReDoS safety analysis
    - Cached unit specifications
    - Indexed rule bindings
    - Deprecation tracking

Design Principles:
    - All IR models are immutable after creation for thread safety
    - Complete Pydantic v2 validation with type hints
    - JSON-serializable for caching and debugging
    - O(1) lookup for common validation operations

Example:
    >>> from greenlang.schema.compiler.ir import SchemaIR
    >>> ir = compiler.compile(schema_source).ir
    >>> print(ir.schema_hash)
    >>> print(ir.properties["/energy_consumption"])

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator - Task 1.4
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


# =============================================================================
# COMPILER VERSION
# =============================================================================

COMPILER_VERSION: str = "0.1.0"


# =============================================================================
# COMPILED PATTERN MODELS
# =============================================================================


class CompiledPattern(BaseModel):
    """
    Precompiled regex pattern with safety metadata.

    This model represents a regex pattern that has been analyzed for ReDoS
    vulnerabilities and optionally precompiled for fast matching.

    Attributes:
        pattern: The original pattern string
        complexity_score: ReDoS complexity score (0.0 = safe, 1.0 = dangerous)
        is_safe: Whether the pattern is safe from ReDoS attacks
        timeout_ms: Maximum allowed matching time in milliseconds
        is_re2_compatible: Whether the pattern can use RE2 (no backtracking)
        vulnerability_type: Type of vulnerability detected, if any
        recommendation: Suggested fix for unsafe patterns

    Example:
        >>> pattern = CompiledPattern(
        ...     pattern="^[a-zA-Z0-9]+$",
        ...     complexity_score=0.1,
        ...     is_safe=True,
        ...     timeout_ms=100
        ... )
    """

    pattern: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The original regex pattern string"
    )
    complexity_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="ReDoS complexity score (0.0=safe, 1.0=dangerous)"
    )
    is_safe: bool = Field(
        default=True,
        description="Whether the pattern is safe from ReDoS attacks"
    )
    timeout_ms: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum allowed matching time in milliseconds"
    )
    is_re2_compatible: bool = Field(
        default=True,
        description="Whether the pattern can use RE2 (no backtracking)"
    )
    vulnerability_type: Optional[str] = Field(
        default=None,
        description="Type of vulnerability detected (if any)"
    )
    recommendation: Optional[str] = Field(
        default=None,
        description="Suggested fix for unsafe patterns"
    )

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    def get_compiled(self) -> Optional[re.Pattern]:
        """
        Get compiled regex pattern for matching.

        Returns:
            Compiled re.Pattern object, or None if pattern is unsafe

        Note:
            Pattern is compiled on demand, not cached in the model
            due to serialization concerns.
        """
        if not self.is_safe:
            return None
        try:
            return re.compile(self.pattern)
        except re.error:
            return None


# =============================================================================
# CONSTRAINT IR MODELS
# =============================================================================


class NumericConstraintIR(BaseModel):
    """
    Precompiled numeric constraints for fast validation.

    This model represents all numeric constraints for a single field,
    indexed by JSON Pointer path for O(1) lookup during validation.

    Attributes:
        path: JSON Pointer path to the field
        minimum: Inclusive minimum value
        maximum: Inclusive maximum value
        exclusive_minimum: Exclusive minimum value
        exclusive_maximum: Exclusive maximum value
        multiple_of: Value must be a multiple of this number

    Example:
        >>> constraint = NumericConstraintIR(
        ...     path="/temperature",
        ...     minimum=0.0,
        ...     maximum=100.0,
        ...     multiple_of=0.5
        ... )
    """

    path: str = Field(
        ...,
        description="JSON Pointer path to the field"
    )
    minimum: Optional[float] = Field(
        default=None,
        description="Inclusive minimum value"
    )
    maximum: Optional[float] = Field(
        default=None,
        description="Inclusive maximum value"
    )
    exclusive_minimum: Optional[float] = Field(
        default=None,
        description="Exclusive minimum value"
    )
    exclusive_maximum: Optional[float] = Field(
        default=None,
        description="Exclusive maximum value"
    )
    multiple_of: Optional[float] = Field(
        default=None,
        gt=0,
        description="Value must be a multiple of this number"
    )

    model_config = ConfigDict(frozen=True, extra="forbid")

    def has_constraints(self) -> bool:
        """Check if any numeric constraints are defined."""
        return any([
            self.minimum is not None,
            self.maximum is not None,
            self.exclusive_minimum is not None,
            self.exclusive_maximum is not None,
            self.multiple_of is not None,
        ])

    def get_effective_minimum(self) -> Optional[Tuple[float, bool]]:
        """
        Get effective minimum with exclusivity flag.

        Returns:
            Tuple of (value, is_exclusive) or None if no minimum
        """
        if self.exclusive_minimum is not None:
            return (self.exclusive_minimum, True)
        if self.minimum is not None:
            return (self.minimum, False)
        return None

    def get_effective_maximum(self) -> Optional[Tuple[float, bool]]:
        """
        Get effective maximum with exclusivity flag.

        Returns:
            Tuple of (value, is_exclusive) or None if no maximum
        """
        if self.exclusive_maximum is not None:
            return (self.exclusive_maximum, True)
        if self.maximum is not None:
            return (self.maximum, False)
        return None


class StringConstraintIR(BaseModel):
    """
    Precompiled string constraints for fast validation.

    This model represents all string constraints for a single field,
    including length limits, regex patterns, and semantic formats.

    Attributes:
        path: JSON Pointer path to the field
        min_length: Minimum string length
        max_length: Maximum string length
        pattern: Regex pattern string (not compiled pattern object)
        pattern_compiled: Precompiled pattern with safety metadata
        format: Semantic format (e.g., "date", "email", "uri")

    Example:
        >>> constraint = StringConstraintIR(
        ...     path="/email",
        ...     format="email",
        ...     max_length=254
        ... )
    """

    path: str = Field(..., description="JSON Pointer path to the field")
    min_length: Optional[int] = Field(
        default=None,
        ge=0,
        description="Minimum string length"
    )
    max_length: Optional[int] = Field(
        default=None,
        ge=0,
        description="Maximum string length"
    )
    pattern: Optional[str] = Field(
        default=None,
        description="Regex pattern string"
    )
    pattern_compiled: Optional[CompiledPattern] = Field(
        default=None,
        description="Precompiled pattern with safety metadata"
    )
    format: Optional[str] = Field(
        default=None,
        description="Semantic format (e.g., 'date', 'email', 'uri')"
    )

    model_config = ConfigDict(frozen=True, extra="forbid")

    def has_constraints(self) -> bool:
        """Check if any string constraints are defined."""
        return any([
            self.min_length is not None,
            self.max_length is not None,
            self.pattern is not None,
            self.format is not None,
        ])

    def has_pattern(self) -> bool:
        """Check if a pattern constraint is defined."""
        return self.pattern is not None

    def has_format(self) -> bool:
        """Check if a format constraint is defined."""
        return self.format is not None


class ArrayConstraintIR(BaseModel):
    """
    Precompiled array constraints for fast validation.

    This model represents all array constraints for a single field,
    including length limits and uniqueness requirements.

    Attributes:
        path: JSON Pointer path to the field
        min_items: Minimum number of items
        max_items: Maximum number of items
        unique_items: Whether items must be unique

    Example:
        >>> constraint = ArrayConstraintIR(
        ...     path="/values",
        ...     min_items=1,
        ...     max_items=100,
        ...     unique_items=True
        ... )
    """

    path: str = Field(..., description="JSON Pointer path to the field")
    min_items: Optional[int] = Field(
        default=None,
        ge=0,
        description="Minimum number of items"
    )
    max_items: Optional[int] = Field(
        default=None,
        ge=0,
        description="Maximum number of items"
    )
    unique_items: bool = Field(
        default=False,
        description="Whether items must be unique"
    )

    model_config = ConfigDict(frozen=True, extra="forbid")

    def has_constraints(self) -> bool:
        """Check if any array constraints are defined."""
        return any([
            self.min_items is not None,
            self.max_items is not None,
            self.unique_items,
        ])


# =============================================================================
# UNIT SPECIFICATION IR
# =============================================================================


class UnitSpecIR(BaseModel):
    """
    Compiled unit specification for fast validation.

    This model represents unit requirements for numeric fields with
    physical units, enabling dimension checking and canonical conversion.

    Attributes:
        path: JSON Pointer path to the field
        dimension: Physical dimension (e.g., "energy", "mass", "volume")
        canonical: Canonical unit for normalization (e.g., "kWh", "kg")
        allowed: List of allowed input units

    Example:
        >>> unit_spec = UnitSpecIR(
        ...     path="/energy_consumption",
        ...     dimension="energy",
        ...     canonical="kWh",
        ...     allowed=["kWh", "MWh", "GJ", "MMBTU"]
        ... )
    """

    path: str = Field(..., description="JSON Pointer path to the field")
    dimension: str = Field(
        ...,
        min_length=1,
        description="Physical dimension (e.g., 'energy', 'mass')"
    )
    canonical: str = Field(
        ...,
        min_length=1,
        description="Canonical unit for normalization"
    )
    allowed: List[str] = Field(
        default_factory=list,
        description="List of allowed input units"
    )

    model_config = ConfigDict(frozen=True, extra="forbid")

    def is_unit_allowed(self, unit: str) -> bool:
        """Check if a unit is in the allowed list."""
        if not self.allowed:
            return True  # No restrictions
        return unit in self.allowed

    def get_allowed_set(self) -> FrozenSet[str]:
        """Get allowed units as a frozen set for fast lookup."""
        return frozenset(self.allowed)


# =============================================================================
# RULE BINDING IR
# =============================================================================


class RuleBindingIR(BaseModel):
    """
    Compiled rule binding for cross-field validation.

    This model represents a validation rule that checks relationships
    between multiple fields, such as consistency checks or conditional
    requirements.

    Attributes:
        rule_id: Unique identifier for the rule
        rule_pack: Optional rule pack identifier for grouping
        severity: Severity level ("error", "warning", "info")
        applies_to: List of JSON Pointer paths this rule applies to
        when: Optional condition expression (evaluated before check)
        check: Validation expression that must evaluate to true
        message: Error message if rule fails
        message_template: Template with {{ var }} placeholders

    Example:
        >>> rule = RuleBindingIR(
        ...     rule_id="scope_sum_check",
        ...     severity="error",
        ...     applies_to=["/scope1", "/scope2", "/total"],
        ...     check={"$eq": ["{{ scope1 }} + {{ scope2 }}", "{{ total }}"]},
        ...     message="Scope 1 + Scope 2 must equal total"
        ... )
    """

    rule_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Unique identifier for the rule"
    )
    rule_pack: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Optional rule pack identifier"
    )
    severity: str = Field(
        default="error",
        description="Severity level: error, warning, or info"
    )
    applies_to: List[str] = Field(
        default_factory=list,
        description="List of JSON Pointer paths this rule applies to"
    )
    when: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Condition expression for when rule applies"
    )
    check: Dict[str, Any] = Field(
        default_factory=dict,
        description="Validation expression that must evaluate to true"
    )
    message: str = Field(
        default="Rule validation failed",
        description="Error message if rule fails"
    )
    message_template: Optional[str] = Field(
        default=None,
        description="Template with {{ var }} placeholders"
    )

    model_config = ConfigDict(frozen=True, extra="forbid")

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        """Validate severity is a known value."""
        valid_severities = {"error", "warning", "info"}
        if v.lower() not in valid_severities:
            raise ValueError(
                f"Invalid severity '{v}'. Must be one of: {valid_severities}"
            )
        return v.lower()

    def is_blocking(self) -> bool:
        """Check if this rule blocks validation on failure."""
        return self.severity == "error"


# =============================================================================
# PROPERTY IR
# =============================================================================


class PropertyIR(BaseModel):
    """
    Compiled property information for fast validation.

    This model represents a single property in the flattened property map,
    containing all metadata needed for validation.

    Attributes:
        path: JSON Pointer path to this property
        type: Expected type(s) for the value
        required: Whether this property is required
        has_default: Whether a default value exists
        default_value: The default value if defined
        gl_extensions: GreenLang extension metadata

    Example:
        >>> prop = PropertyIR(
        ...     path="/energy_consumption",
        ...     type="number",
        ...     required=True,
        ...     has_default=False
        ... )
    """

    path: str = Field(..., description="JSON Pointer path")
    type: Optional[str] = Field(
        default=None,
        description="Expected type(s) for the value"
    )
    required: bool = Field(
        default=False,
        description="Whether this property is required"
    )
    has_default: bool = Field(
        default=False,
        description="Whether a default value exists"
    )
    default_value: Optional[Any] = Field(
        default=None,
        description="The default value if defined"
    )
    gl_extensions: Optional[Dict[str, Any]] = Field(
        default=None,
        description="GreenLang extension metadata"
    )

    model_config = ConfigDict(frozen=True, extra="forbid")


# =============================================================================
# DEPRECATION IR
# =============================================================================


class DeprecationInfoIR(BaseModel):
    """
    Compiled deprecation information for a field.

    This model tracks deprecated fields for migration warnings.

    Attributes:
        path: JSON Pointer path to the deprecated field
        since_version: Version when deprecation was introduced
        message: Human-readable deprecation message
        replacement: Suggested replacement field path
        removal_version: Version when field will be removed

    Example:
        >>> deprecation = DeprecationInfoIR(
        ...     path="/old_field",
        ...     since_version="2.0.0",
        ...     message="Use 'new_field' instead",
        ...     replacement="/new_field",
        ...     removal_version="3.0.0"
        ... )
    """

    path: str = Field(..., description="JSON Pointer path to deprecated field")
    since_version: str = Field(
        ...,
        description="Version when deprecation was introduced"
    )
    message: str = Field(
        ...,
        description="Human-readable deprecation message"
    )
    replacement: Optional[str] = Field(
        default=None,
        description="Suggested replacement field path"
    )
    removal_version: Optional[str] = Field(
        default=None,
        description="Version when field will be removed"
    )

    model_config = ConfigDict(frozen=True, extra="forbid")


# =============================================================================
# SCHEMA IR (Main Structure)
# =============================================================================


class SchemaIR(BaseModel):
    """
    Compiled schema Intermediate Representation.

    The IR is the compiled form of a JSON Schema optimized for fast
    validation. It provides O(1) property lookups, precompiled constraints,
    and indexed metadata.

    This is the main output of the schema compiler and the primary input
    to the validator.

    Attributes:
        schema_id: Unique schema identifier
        version: Schema version string
        schema_hash: SHA-256 hash of canonical schema for caching
        compiled_at: Timestamp when schema was compiled
        compiler_version: Version of the compiler used

        properties: Flattened property map (path -> PropertyIR)
        required_paths: Set of required field paths

        numeric_constraints: Numeric constraints by path
        string_constraints: String constraints by path
        array_constraints: Array constraints by path

        patterns: Precompiled regex patterns by path
        unit_specs: Unit specifications by path
        rule_bindings: Cross-field rule bindings

        deprecated_fields: Deprecated field information by path
        renamed_fields: Map of old field names to new names
        enums: Enum values by path for quick lookup

    Example:
        >>> ir = SchemaIR(
        ...     schema_id="emissions/activity",
        ...     version="1.3.0",
        ...     schema_hash="abc123...",
        ...     compiled_at=datetime.now(),
        ...     compiler_version="0.1.0"
        ... )
    """

    # Schema identification
    schema_id: str = Field(..., description="Unique schema identifier")
    version: str = Field(..., description="Schema version string")
    schema_hash: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 hash of canonical schema"
    )
    compiled_at: datetime = Field(
        ...,
        description="Timestamp when schema was compiled"
    )
    compiler_version: str = Field(
        default=COMPILER_VERSION,
        description="Version of the compiler used"
    )

    # Flattened property maps for O(1) lookup
    properties: Dict[str, PropertyIR] = Field(
        default_factory=dict,
        description="Flattened property map (path -> PropertyIR)"
    )
    required_paths: Set[str] = Field(
        default_factory=set,
        description="Set of required field paths"
    )

    # Precompiled constraints
    numeric_constraints: Dict[str, NumericConstraintIR] = Field(
        default_factory=dict,
        description="Numeric constraints by path"
    )
    string_constraints: Dict[str, StringConstraintIR] = Field(
        default_factory=dict,
        description="String constraints by path"
    )
    array_constraints: Dict[str, ArrayConstraintIR] = Field(
        default_factory=dict,
        description="Array constraints by path"
    )

    # Precompiled regexes with safety metadata
    patterns: Dict[str, CompiledPattern] = Field(
        default_factory=dict,
        description="Precompiled regex patterns by path"
    )

    # Unit metadata
    unit_specs: Dict[str, UnitSpecIR] = Field(
        default_factory=dict,
        description="Unit specifications by path"
    )

    # Rule bindings
    rule_bindings: List[RuleBindingIR] = Field(
        default_factory=list,
        description="Cross-field rule bindings"
    )

    # Deprecation index
    deprecated_fields: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Deprecated field information by path"
    )
    renamed_fields: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of old field names to new names"
    )

    # Enums for quick lookup
    enums: Dict[str, List[Any]] = Field(
        default_factory=dict,
        description="Enum values by path for quick lookup"
    )

    model_config = ConfigDict(
        frozen=False,  # Allow modification during compilation
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    # -------------------------------------------------------------------------
    # Lookup Methods
    # -------------------------------------------------------------------------

    def get_property(self, path: str) -> Optional[PropertyIR]:
        """
        Get property IR by JSON Pointer path.

        Args:
            path: JSON Pointer path (e.g., "/energy_consumption")

        Returns:
            PropertyIR if found, None otherwise
        """
        return self.properties.get(path)

    def is_required(self, path: str) -> bool:
        """
        Check if a path is required.

        Args:
            path: JSON Pointer path

        Returns:
            True if the path is required
        """
        return path in self.required_paths

    def get_numeric_constraint(self, path: str) -> Optional[NumericConstraintIR]:
        """
        Get numeric constraints for a path.

        Args:
            path: JSON Pointer path

        Returns:
            NumericConstraintIR if found, None otherwise
        """
        return self.numeric_constraints.get(path)

    def get_string_constraint(self, path: str) -> Optional[StringConstraintIR]:
        """
        Get string constraints for a path.

        Args:
            path: JSON Pointer path

        Returns:
            StringConstraintIR if found, None otherwise
        """
        return self.string_constraints.get(path)

    def get_array_constraint(self, path: str) -> Optional[ArrayConstraintIR]:
        """
        Get array constraints for a path.

        Args:
            path: JSON Pointer path

        Returns:
            ArrayConstraintIR if found, None otherwise
        """
        return self.array_constraints.get(path)

    def get_pattern(self, path: str) -> Optional[CompiledPattern]:
        """
        Get compiled pattern for a path.

        Args:
            path: JSON Pointer path

        Returns:
            CompiledPattern if found, None otherwise
        """
        return self.patterns.get(path)

    def get_unit_spec(self, path: str) -> Optional[UnitSpecIR]:
        """
        Get unit specification for a path.

        Args:
            path: JSON Pointer path

        Returns:
            UnitSpecIR if found, None otherwise
        """
        return self.unit_specs.get(path)

    def get_enum(self, path: str) -> Optional[List[Any]]:
        """
        Get enum values for a path.

        Args:
            path: JSON Pointer path

        Returns:
            List of allowed enum values if found, None otherwise
        """
        return self.enums.get(path)

    def get_deprecation_info(self, path: str) -> Optional[Dict[str, Any]]:
        """
        Get deprecation info for a path.

        Args:
            path: JSON Pointer path

        Returns:
            Deprecation info dict if found, None otherwise
        """
        return self.deprecated_fields.get(path)

    def is_deprecated(self, path: str) -> bool:
        """
        Check if a path is deprecated.

        Args:
            path: JSON Pointer path

        Returns:
            True if the path is deprecated
        """
        return path in self.deprecated_fields

    def get_renamed_to(self, old_path: str) -> Optional[str]:
        """
        Get the new name for a renamed field.

        Args:
            old_path: Old field path

        Returns:
            New field path if renamed, None otherwise
        """
        return self.renamed_fields.get(old_path)

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_stats(self) -> Dict[str, int]:
        """
        Get compilation statistics.

        Returns:
            Dictionary with counts of various IR elements
        """
        return {
            "properties": len(self.properties),
            "required_paths": len(self.required_paths),
            "numeric_constraints": len(self.numeric_constraints),
            "string_constraints": len(self.string_constraints),
            "array_constraints": len(self.array_constraints),
            "patterns": len(self.patterns),
            "unit_specs": len(self.unit_specs),
            "rule_bindings": len(self.rule_bindings),
            "deprecated_fields": len(self.deprecated_fields),
            "renamed_fields": len(self.renamed_fields),
            "enums": len(self.enums),
        }


# =============================================================================
# COMPILATION RESULT
# =============================================================================


class CompilationResult(BaseModel):
    """
    Result of schema compilation.

    This model encapsulates the compilation outcome, including the
    compiled IR (if successful), warnings, errors, and timing information.

    Attributes:
        ir: The compiled Intermediate Representation (None if failed)
        warnings: List of compilation warnings
        errors: List of compilation errors
        compile_time_ms: Time taken to compile in milliseconds

    Example:
        >>> result = compiler.compile(schema_source, schema_ref)
        >>> if result.success:
        ...     print(f"Compiled in {result.compile_time_ms}ms")
        ... else:
        ...     print(f"Errors: {result.errors}")
    """

    ir: Optional[SchemaIR] = Field(
        default=None,
        description="The compiled Intermediate Representation"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="List of compilation warnings"
    )
    errors: List[str] = Field(
        default_factory=list,
        description="List of compilation errors"
    )
    compile_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time taken to compile in milliseconds"
    )

    model_config = ConfigDict(frozen=False, extra="forbid")

    @property
    def success(self) -> bool:
        """Check if compilation was successful."""
        return self.ir is not None and len(self.errors) == 0

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)


# =============================================================================
# COMPILATION ERROR
# =============================================================================


class CompilationError(Exception):
    """
    Exception raised during schema compilation.

    Provides structured error information for compilation failures.

    Attributes:
        code: Error code (GLSCHEMA-E5xx)
        message: Human-readable error message
        path: JSON Pointer path where error occurred
    """

    def __init__(
        self,
        code: str,
        message: str,
        path: str = ""
    ):
        """
        Initialize CompilationError.

        Args:
            code: Error code string
            message: Human-readable error message
            path: JSON Pointer path where error occurred
        """
        self.code = code
        self.message = message
        self.path = path
        super().__init__(f"[{code}] {message}")

    def __str__(self) -> str:
        """Return formatted error string."""
        if self.path:
            return f"[{self.code}] {self.message} at path '{self.path}'"
        return f"[{self.code}] {self.message}"

    def __repr__(self) -> str:
        """Return detailed representation."""
        return f"CompilationError(code={self.code!r}, message={self.message!r}, path={self.path!r})"


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Constants
    "COMPILER_VERSION",
    # Pattern models
    "CompiledPattern",
    # Constraint models
    "NumericConstraintIR",
    "StringConstraintIR",
    "ArrayConstraintIR",
    # Specification models
    "UnitSpecIR",
    "RuleBindingIR",
    "PropertyIR",
    "DeprecationInfoIR",
    # Main IR
    "SchemaIR",
    # Result models
    "CompilationResult",
    "CompilationError",
]
