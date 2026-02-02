# -*- coding: utf-8 -*-
"""
GreenLang Schema SDK - Clean Python API for GL-FOUND-X-002.

This module provides a user-friendly, high-level API for schema validation,
compilation, and fix suggestion handling. It serves as the primary programmatic
interface for developers integrating GreenLang schema validation into their
Python applications.

Design Goals:
    - Simple, intuitive API with sensible defaults
    - Type-safe with comprehensive type hints
    - Zero-hallucination: All calculations deterministic
    - Provenance tracking via schema hashes
    - Pythonic conventions throughout

Quick Start:
    >>> from greenlang.schema import validate, compile_schema, SchemaRef
    >>>
    >>> # Simple validation
    >>> result = validate(payload, "gl://schemas/activity@1.0.0")
    >>>
    >>> # Check result
    >>> if result.valid:
    ...     print("Validation passed!")
    ...     normalized = result.normalized_payload
    ... else:
    ...     for finding in result.findings:
    ...         print(f"{finding.code}: {finding.message}")
    >>>
    >>> # Apply safe fixes
    >>> for fix in result.fix_suggestions or []:
    ...     if fix.safety == "safe":
    ...         print(f"Apply: {fix.rationale}")

API Overview:
    - validate(): Validate a payload against a schema
    - validate_batch(): Validate multiple payloads efficiently
    - compile_schema(): Pre-compile a schema for repeated validation
    - apply_fixes(): Apply safe fix suggestions to a payload
    - CompiledSchema: Pre-compiled schema for fast validation

Author: GreenLang Framework Team
Version: 1.0.0
GL-FOUND-X-002: Schema Compiler & Validator - Task 7.5
"""

from __future__ import annotations

import copy
import logging
import time
from functools import lru_cache
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)

from greenlang.schema.compiler.compiler import SchemaCompiler
from greenlang.schema.compiler.ir import SchemaIR, CompilationResult
from greenlang.schema.compiler.parser import parse_payload, ParseError
from greenlang.schema.constants import MAX_BATCH_ITEMS
from greenlang.schema.models.config import (
    CoercionPolicy,
    PatchLevel,
    UnknownFieldPolicy,
    ValidationOptions,
    ValidationProfile,
)
from greenlang.schema.models.finding import Finding, Severity
from greenlang.schema.models.patch import FixSuggestion, JSONPatchOp, PatchSafety
from greenlang.schema.models.report import (
    BatchValidationReport,
    ItemResult,
    ValidationReport,
    ValidationSummary,
)
from greenlang.schema.models.schema_ref import SchemaRef
from greenlang.schema.registry.resolver import SchemaRegistry
from greenlang.schema.validator.core import SchemaValidator
from greenlang.schema.validator.units import UnitCatalog


logger = logging.getLogger(__name__)


# =============================================================================
# TYPE ALIASES
# =============================================================================


SchemaInput = Union[SchemaRef, str, Dict[str, Any]]
"""
Type alias for schema input formats.

Accepted formats:
    - SchemaRef: Explicit schema reference object
    - str: Schema URI (e.g., "gl://schemas/activity@1.0.0")
    - Dict[str, Any]: Inline schema definition
"""

PayloadInput = Union[str, Dict[str, Any]]
"""
Type alias for payload input formats.

Accepted formats:
    - str: YAML or JSON string
    - Dict[str, Any]: Pre-parsed dictionary
"""

ProfileInput = Union[ValidationProfile, Literal["strict", "standard", "permissive"]]
"""
Type alias for validation profile input.

Accepted formats:
    - ValidationProfile: Enum value
    - str: Profile name ("strict", "standard", "permissive")
"""

PatchLevelInput = Union[PatchLevel, Literal["safe", "needs_review", "unsafe"]]
"""
Type alias for patch level input.

Accepted formats:
    - PatchLevel: Enum value
    - str: Level name ("safe", "needs_review", "unsafe")
"""


# =============================================================================
# COMPILED SCHEMA
# =============================================================================


class CompiledSchema:
    """
    Pre-compiled schema for efficient repeated validation.

    When validating many payloads against the same schema, pre-compiling
    the schema avoids repeated parsing and compilation overhead. This class
    wraps a compiled schema IR and provides a simple validation interface.

    Attributes:
        schema_ref: The schema reference
        schema_hash: SHA-256 hash of the compiled schema (for provenance)
        compile_time_ms: Time taken to compile the schema
        properties: Number of properties in the schema
        rules: Number of rules in the schema

    Example:
        >>> compiled = compile_schema("gl://schemas/activity@1.0.0")
        >>> print(f"Schema hash: {compiled.schema_hash[:16]}...")
        >>>
        >>> # Validate multiple payloads efficiently
        >>> for payload in payloads:
        ...     result = compiled.validate(payload)
        ...     if not result.valid:
        ...         handle_errors(result)

    Thread Safety:
        CompiledSchema instances are thread-safe for concurrent validation.
        The underlying IR is immutable after compilation.
    """

    def __init__(
        self,
        ir: SchemaIR,
        schema_ref: SchemaRef,
        compile_time_ms: float,
        registry: Optional[SchemaRegistry] = None,
        unit_catalog: Optional[UnitCatalog] = None,
    ):
        """
        Initialize a CompiledSchema.

        This constructor is typically not called directly. Use compile_schema()
        to create CompiledSchema instances.

        Args:
            ir: Compiled schema Intermediate Representation
            schema_ref: Schema reference
            compile_time_ms: Compilation time in milliseconds
            registry: Optional schema registry for nested refs
            unit_catalog: Optional unit catalog for unit validation
        """
        self._ir = ir
        self._schema_ref = schema_ref
        self._compile_time_ms = compile_time_ms
        self._registry = registry
        self._unit_catalog = unit_catalog or UnitCatalog()
        self._validator: Optional[SchemaValidator] = None

    @property
    def schema_ref(self) -> SchemaRef:
        """Get the schema reference."""
        return self._schema_ref

    @property
    def schema_hash(self) -> str:
        """Get the SHA-256 hash of the compiled schema."""
        return self._ir.schema_hash

    @property
    def compile_time_ms(self) -> float:
        """Get the compilation time in milliseconds."""
        return self._compile_time_ms

    @property
    def schema_id(self) -> str:
        """Get the schema identifier."""
        return self._ir.schema_id

    @property
    def version(self) -> str:
        """Get the schema version."""
        return self._ir.version

    @property
    def properties(self) -> int:
        """Get the number of properties in the schema."""
        return len(self._ir.properties)

    @property
    def rules(self) -> int:
        """Get the number of rules in the schema."""
        return len(self._ir.rule_bindings)

    def validate(
        self,
        payload: PayloadInput,
        *,
        profile: Optional[ProfileInput] = None,
        normalize: bool = True,
        emit_patches: bool = True,
        patch_level: Optional[PatchLevelInput] = None,
        max_errors: int = 100,
        fail_fast: bool = False,
    ) -> ValidationReport:
        """
        Validate a payload against this compiled schema.

        This method provides the same functionality as the top-level validate()
        function, but uses the pre-compiled schema for better performance.

        Args:
            payload: Payload as YAML/JSON string or dictionary
            profile: Validation strictness ("strict", "standard", "permissive")
            normalize: Whether to normalize the payload
            emit_patches: Whether to generate fix suggestions
            patch_level: Maximum safety level for patches ("safe", "needs_review", "unsafe")
            max_errors: Maximum errors to collect
            fail_fast: Stop at first error

        Returns:
            ValidationReport with validation results

        Example:
            >>> result = compiled.validate(payload, profile="strict")
            >>> if result.valid:
            ...     print(f"Normalized: {result.normalized_payload}")
        """
        options = _build_options(
            profile=profile,
            normalize=normalize,
            emit_patches=emit_patches,
            patch_level=patch_level,
            max_errors=max_errors,
            fail_fast=fail_fast,
        )

        # Get or create validator
        if self._validator is None:
            self._validator = SchemaValidator(
                schema_registry=self._registry,
                unit_catalog=self._unit_catalog,
                options=options,
            )
            # Pre-cache the IR
            cache_key = self._schema_ref.to_cache_key()
            self._validator._ir_cache[cache_key] = self._ir

        return self._validator.validate(payload, self._schema_ref, options)

    def validate_batch(
        self,
        payloads: Sequence[PayloadInput],
        *,
        profile: Optional[ProfileInput] = None,
        normalize: bool = True,
        max_errors: int = 100,
    ) -> BatchValidationReport:
        """
        Validate multiple payloads efficiently.

        Uses the pre-compiled schema for all payloads, maximizing performance
        for batch validation scenarios.

        Args:
            payloads: Sequence of payloads to validate
            profile: Validation strictness
            normalize: Whether to normalize payloads
            max_errors: Maximum errors per payload

        Returns:
            BatchValidationReport with results for all payloads

        Example:
            >>> batch_result = compiled.validate_batch(payloads)
            >>> print(f"Valid: {batch_result.summary.valid_count}/{batch_result.summary.total_items}")
        """
        options = _build_options(
            profile=profile,
            normalize=normalize,
            max_errors=max_errors,
        )

        if self._validator is None:
            self._validator = SchemaValidator(
                schema_registry=self._registry,
                unit_catalog=self._unit_catalog,
                options=options,
            )
            cache_key = self._schema_ref.to_cache_key()
            self._validator._ir_cache[cache_key] = self._ir

        return self._validator.validate_batch(list(payloads), self._schema_ref, options)

    def __repr__(self) -> str:
        """Return detailed representation."""
        return (
            f"CompiledSchema(schema_id='{self.schema_id}', version='{self.version}', "
            f"properties={self.properties}, hash='{self.schema_hash[:16]}...')"
        )


# =============================================================================
# MAIN VALIDATION FUNCTIONS
# =============================================================================


def validate(
    payload: PayloadInput,
    schema: SchemaInput,
    *,
    profile: Optional[ProfileInput] = None,
    normalize: bool = True,
    emit_patches: bool = True,
    patch_level: Optional[PatchLevelInput] = None,
    max_errors: int = 100,
    fail_fast: bool = False,
    unknown_field_policy: Optional[Literal["error", "warn", "ignore"]] = None,
    coercion_policy: Optional[Literal["off", "safe", "aggressive"]] = None,
    registry: Optional[SchemaRegistry] = None,
) -> ValidationReport:
    """
    Validate a payload against a schema.

    This is the primary entry point for schema validation. It provides a simple,
    intuitive interface with sensible defaults while allowing full customization
    when needed.

    Args:
        payload: Payload to validate. Can be:
            - YAML/JSON string
            - Pre-parsed dictionary
        schema: Schema to validate against. Can be:
            - Schema URI (e.g., "gl://schemas/activity@1.0.0")
            - SchemaRef object
            - Inline schema dictionary
        profile: Validation strictness level:
            - "strict": All warnings become errors, unknown fields rejected
            - "standard" (default): Standard validation
            - "permissive": Only critical errors
        normalize: If True (default), normalize payload to canonical form
        emit_patches: If True (default), generate fix suggestions
        patch_level: Maximum safety level for patches:
            - "safe" (default): Only safe auto-apply patches
            - "needs_review": Include patches needing review
            - "unsafe": Include all patches
        max_errors: Maximum errors to collect (default 100)
        fail_fast: If True, stop at first error (default False)
        unknown_field_policy: How to handle unknown fields:
            - "error": Fail validation
            - "warn" (default): Generate warning
            - "ignore": Silently ignore
        coercion_policy: Type coercion behavior:
            - "off": No coercion
            - "safe" (default): Safe coercions only
            - "aggressive": All possible coercions
        registry: Optional schema registry for resolving references

    Returns:
        ValidationReport containing:
            - valid: Whether payload passed validation
            - schema_ref: The schema reference used
            - schema_hash: SHA-256 hash of the schema (for provenance)
            - summary: Counts of errors, warnings, and info
            - findings: List of all validation findings
            - normalized_payload: Normalized payload (if normalize=True)
            - fix_suggestions: Suggested fixes (if emit_patches=True)
            - timings: Performance timing information

    Raises:
        No exceptions are raised; all errors are captured in the report.

    Examples:
        Basic validation:
            >>> result = validate(payload, "gl://schemas/activity@1.0.0")
            >>> if result.valid:
            ...     print("Valid!")

        Strict validation:
            >>> result = validate(payload, schema_ref, profile="strict")

        Check findings:
            >>> for finding in result.findings:
            ...     print(f"{finding.code}: {finding.message}")

        Get normalized payload:
            >>> if result.valid:
            ...     normalized = result.normalized_payload

        Review fix suggestions:
            >>> for fix in result.fix_suggestions or []:
            ...     if fix.safety == "safe":
            ...         print(f"Apply: {fix.rationale}")

        Inline schema:
            >>> schema = {
            ...     "type": "object",
            ...     "properties": {"name": {"type": "string"}},
            ...     "required": ["name"]
            ... }
            >>> result = validate({"name": "test"}, schema)
    """
    # Build validation options
    options = _build_options(
        profile=profile,
        normalize=normalize,
        emit_patches=emit_patches,
        patch_level=patch_level,
        max_errors=max_errors,
        fail_fast=fail_fast,
        unknown_field_policy=unknown_field_policy,
        coercion_policy=coercion_policy,
    )

    # Create validator
    validator = SchemaValidator(
        schema_registry=registry,
        options=options,
    )

    # Parse schema reference
    schema_ref = _parse_schema_input(schema)

    # Handle inline schemas
    if isinstance(schema, dict):
        # Compile inline schema and cache it
        result = validator._compiler.compile(
            schema_source=schema,
            schema_id=schema_ref.schema_id,
            version=schema_ref.version,
        )

        if not result.success:
            return _create_compile_error_report(schema_ref, result.errors)

        # Cache the IR
        validator._ir_cache[schema_ref.to_cache_key()] = result.ir

    return validator.validate(payload, schema_ref, options)


def validate_batch(
    payloads: Sequence[PayloadInput],
    schema: SchemaInput,
    *,
    profile: Optional[ProfileInput] = None,
    normalize: bool = True,
    max_errors: int = 100,
    registry: Optional[SchemaRegistry] = None,
) -> BatchValidationReport:
    """
    Validate multiple payloads against a schema efficiently.

    Shares the compiled schema IR across all payloads for better performance.
    This is the recommended method when validating many payloads against
    the same schema.

    Args:
        payloads: Sequence of payloads to validate
        schema: Schema to validate against
        profile: Validation strictness level
        normalize: If True, normalize payloads
        max_errors: Maximum errors per payload
        registry: Optional schema registry

    Returns:
        BatchValidationReport containing:
            - schema_ref: The schema reference
            - schema_hash: SHA-256 hash of the schema
            - summary: Aggregate summary of all items
            - results: Individual results for each payload

    Raises:
        ValueError: If batch exceeds MAX_BATCH_ITEMS limit

    Example:
        >>> payloads = [
        ...     {"energy": 100, "unit": "kWh"},
        ...     {"energy": 200, "unit": "MWh"},
        ... ]
        >>> result = validate_batch(payloads, "gl://schemas/activity@1.0.0")
        >>> print(f"Valid: {result.summary.valid_count}/{result.summary.total_items}")
        >>>
        >>> # Process failed items
        >>> for item in result.failed_items():
        ...     print(f"Item {item.index}: {len(item.findings)} errors")
    """
    if len(payloads) > MAX_BATCH_ITEMS:
        raise ValueError(
            f"Batch size {len(payloads)} exceeds maximum {MAX_BATCH_ITEMS}"
        )

    options = _build_options(
        profile=profile,
        normalize=normalize,
        max_errors=max_errors,
    )

    validator = SchemaValidator(
        schema_registry=registry,
        options=options,
    )

    schema_ref = _parse_schema_input(schema)

    # Handle inline schemas
    if isinstance(schema, dict):
        result = validator._compiler.compile(
            schema_source=schema,
            schema_id=schema_ref.schema_id,
            version=schema_ref.version,
        )

        if not result.success:
            # Return batch error report
            return _create_batch_compile_error_report(
                schema_ref, result.errors, len(payloads)
            )

        validator._ir_cache[schema_ref.to_cache_key()] = result.ir

    return validator.validate_batch(list(payloads), schema_ref, options)


def compile_schema(
    schema: SchemaInput,
    *,
    registry: Optional[SchemaRegistry] = None,
) -> CompiledSchema:
    """
    Pre-compile a schema for efficient repeated validation.

    When validating many payloads against the same schema, pre-compiling
    avoids repeated parsing and compilation overhead. The compiled schema
    can be reused across multiple validate() calls.

    Args:
        schema: Schema to compile. Can be:
            - Schema URI (e.g., "gl://schemas/activity@1.0.0")
            - SchemaRef object
            - Inline schema dictionary
        registry: Optional schema registry for resolving references

    Returns:
        CompiledSchema object for efficient validation

    Raises:
        ValueError: If schema compilation fails

    Example:
        >>> # Pre-compile schema
        >>> compiled = compile_schema("gl://schemas/activity@1.0.0")
        >>> print(f"Schema hash: {compiled.schema_hash[:16]}...")
        >>>
        >>> # Validate many payloads efficiently
        >>> for payload in payloads:
        ...     result = compiled.validate(payload)
        ...     process_result(result)

    Performance:
        Pre-compilation eliminates the ~2-5ms compilation overhead per
        validation call, which can be significant when validating
        thousands of payloads.
    """
    start_time = time.perf_counter()

    schema_ref = _parse_schema_input(schema)
    compiler = SchemaCompiler()

    # Get schema source
    if isinstance(schema, dict):
        schema_source = schema
    elif registry is not None:
        source = registry.resolve(schema_ref.schema_id, schema_ref.version)
        schema_source = source.content
    else:
        raise ValueError(
            f"Cannot resolve schema '{schema_ref}' without a registry. "
            "Provide an inline schema dict or configure a registry."
        )

    # Compile schema
    result = compiler.compile(
        schema_source=schema_source,
        schema_id=schema_ref.schema_id,
        version=schema_ref.version,
    )

    if not result.success:
        errors = "; ".join(result.errors)
        raise ValueError(f"Schema compilation failed: {errors}")

    compile_time_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        f"Compiled schema {schema_ref} in {compile_time_ms:.2f}ms "
        f"(hash={result.ir.schema_hash[:16]}...)"
    )

    return CompiledSchema(
        ir=result.ir,
        schema_ref=schema_ref,
        compile_time_ms=compile_time_ms,
        registry=registry,
    )


# =============================================================================
# FIX SUGGESTION HELPERS
# =============================================================================


def apply_fixes(
    payload: Dict[str, Any],
    fixes: Sequence[FixSuggestion],
    *,
    safety: Literal["safe", "needs_review", "unsafe"] = "safe",
) -> Tuple[Dict[str, Any], List[FixSuggestion]]:
    """
    Apply fix suggestions to a payload.

    Applies only suggestions at or above the specified safety level.
    Returns a new payload (original is not modified) and the list of
    applied suggestions.

    Args:
        payload: The original payload
        fixes: Sequence of fix suggestions
        safety: Maximum safety level to apply:
            - "safe" (default): Only safe patches
            - "needs_review": Safe + needs_review patches
            - "unsafe": All patches

    Returns:
        Tuple of (modified_payload, applied_fixes)

    Example:
        >>> # Apply only safe fixes
        >>> new_payload, applied = apply_fixes(payload, result.fix_suggestions)
        >>> print(f"Applied {len(applied)} fixes")
        >>>
        >>> # Apply safe and needs_review fixes
        >>> new_payload, applied = apply_fixes(
        ...     payload,
        ...     result.fix_suggestions,
        ...     safety="needs_review"
        ... )

    Note:
        This function creates a deep copy of the payload before applying
        fixes. The original payload is never modified.
    """
    # Import here to avoid circular import
    from greenlang.schema.suggestions.engine import apply_suggestions

    # Convert safety string to PatchSafety
    safety_level = PatchSafety(safety)

    return apply_suggestions(payload, list(fixes), safety_level)


def safe_fixes(fixes: Optional[Sequence[FixSuggestion]]) -> List[FixSuggestion]:
    """
    Filter fix suggestions to only safe ones.

    Convenience function to extract only safe-to-apply fixes from a list
    of suggestions.

    Args:
        fixes: Sequence of fix suggestions (or None)

    Returns:
        List of safe fix suggestions

    Example:
        >>> for fix in safe_fixes(result.fix_suggestions):
        ...     print(f"Can safely apply: {fix.rationale}")
    """
    if not fixes:
        return []

    return [f for f in fixes if f.safety == PatchSafety.SAFE]


def review_fixes(fixes: Optional[Sequence[FixSuggestion]]) -> List[FixSuggestion]:
    """
    Filter fix suggestions to ones needing review.

    Args:
        fixes: Sequence of fix suggestions (or None)

    Returns:
        List of fix suggestions that need human review

    Example:
        >>> for fix in review_fixes(result.fix_suggestions):
        ...     print(f"Review needed: {fix.rationale}")
    """
    if not fixes:
        return []

    return [f for f in fixes if f.safety == PatchSafety.NEEDS_REVIEW]


# =============================================================================
# FINDING HELPERS
# =============================================================================


def errors_only(findings: Sequence[Finding]) -> List[Finding]:
    """
    Filter findings to only errors.

    Args:
        findings: Sequence of validation findings

    Returns:
        List of error-level findings

    Example:
        >>> for error in errors_only(result.findings):
        ...     print(f"ERROR: {error.code} at {error.path}")
    """
    return [f for f in findings if f.severity == Severity.ERROR]


def warnings_only(findings: Sequence[Finding]) -> List[Finding]:
    """
    Filter findings to only warnings.

    Args:
        findings: Sequence of validation findings

    Returns:
        List of warning-level findings

    Example:
        >>> for warning in warnings_only(result.findings):
        ...     print(f"WARNING: {warning.code} at {warning.path}")
    """
    return [f for f in findings if f.severity == Severity.WARNING]


def findings_by_path(
    findings: Sequence[Finding],
    path: str,
) -> List[Finding]:
    """
    Get findings at a specific path.

    Args:
        findings: Sequence of validation findings
        path: JSON Pointer path (e.g., "/energy" or "/items/0/name")

    Returns:
        List of findings at that path

    Example:
        >>> for finding in findings_by_path(result.findings, "/energy"):
        ...     print(f"{finding.code}: {finding.message}")
    """
    return [f for f in findings if f.path == path]


def findings_by_code(
    findings: Sequence[Finding],
    code: str,
) -> List[Finding]:
    """
    Get findings with a specific error code.

    Args:
        findings: Sequence of validation findings
        code: Error code (e.g., "GLSCHEMA-E100")

    Returns:
        List of findings with that code

    Example:
        >>> missing_fields = findings_by_code(result.findings, "GLSCHEMA-E100")
        >>> print(f"Found {len(missing_fields)} missing required fields")
    """
    return [f for f in findings if f.code == code]


# =============================================================================
# SCHEMA REFERENCE HELPERS
# =============================================================================


def parse_schema_ref(uri: str) -> SchemaRef:
    """
    Parse a schema URI into a SchemaRef object.

    Args:
        uri: Schema URI (e.g., "gl://schemas/activity@1.0.0")

    Returns:
        SchemaRef object

    Raises:
        ValueError: If URI format is invalid

    Example:
        >>> ref = parse_schema_ref("gl://schemas/emissions/activity@1.3.0")
        >>> print(f"Schema: {ref.schema_id}, Version: {ref.version}")
        Schema: emissions/activity, Version: 1.3.0
    """
    return SchemaRef.from_uri(uri)


def schema_ref(
    schema_id: str,
    version: str,
    variant: Optional[str] = None,
) -> SchemaRef:
    """
    Create a SchemaRef from components.

    Args:
        schema_id: Schema identifier (e.g., "emissions/activity")
        version: Schema version (e.g., "1.3.0")
        variant: Optional variant identifier (e.g., "strict")

    Returns:
        SchemaRef object

    Example:
        >>> ref = schema_ref("emissions/activity", "1.3.0")
        >>> print(ref.to_uri())
        gl://schemas/emissions/activity@1.3.0
    """
    return SchemaRef(schema_id=schema_id, version=version, variant=variant)


# =============================================================================
# INTERNAL HELPERS
# =============================================================================


def _build_options(
    profile: Optional[ProfileInput] = None,
    normalize: bool = True,
    emit_patches: bool = True,
    patch_level: Optional[PatchLevelInput] = None,
    max_errors: int = 100,
    fail_fast: bool = False,
    unknown_field_policy: Optional[Literal["error", "warn", "ignore"]] = None,
    coercion_policy: Optional[Literal["off", "safe", "aggressive"]] = None,
) -> ValidationOptions:
    """
    Build ValidationOptions from SDK parameters.

    Internal helper that converts SDK-friendly parameters to
    ValidationOptions model.
    """
    # Convert profile
    if profile is None:
        profile_enum = ValidationProfile.STANDARD
    elif isinstance(profile, str):
        profile_enum = ValidationProfile(profile)
    else:
        profile_enum = profile

    # Convert patch_level
    if patch_level is None:
        patch_level_enum = PatchLevel.SAFE
    elif isinstance(patch_level, str):
        patch_level_enum = PatchLevel(patch_level)
    else:
        patch_level_enum = patch_level

    # Convert unknown_field_policy
    if unknown_field_policy is None:
        unknown_policy_enum = UnknownFieldPolicy.WARN
    else:
        unknown_policy_enum = UnknownFieldPolicy(unknown_field_policy)

    # Convert coercion_policy
    if coercion_policy is None:
        coercion_enum = CoercionPolicy.SAFE
    else:
        coercion_enum = CoercionPolicy(coercion_policy)

    return ValidationOptions(
        profile=profile_enum,
        normalize=normalize,
        emit_patches=emit_patches,
        patch_level=patch_level_enum,
        max_errors=max_errors,
        fail_fast=fail_fast,
        unknown_field_policy=unknown_policy_enum,
        coercion_policy=coercion_enum,
    )


def _parse_schema_input(schema: SchemaInput) -> SchemaRef:
    """
    Parse schema input to SchemaRef.

    Internal helper that converts various schema input formats
    to a SchemaRef object.
    """
    if isinstance(schema, SchemaRef):
        return schema
    elif isinstance(schema, str):
        return SchemaRef.from_uri(schema)
    elif isinstance(schema, dict):
        # Inline schema - create a synthetic reference with valid format
        return SchemaRef(schema_id="inline/schema", version="1.0.0")
    else:
        raise TypeError(
            f"Invalid schema type: {type(schema)}. "
            "Expected SchemaRef, str (URI), or dict (inline schema)."
        )


def _create_compile_error_report(
    schema_ref: SchemaRef,
    errors: List[str],
) -> ValidationReport:
    """
    Create a validation report for compilation errors.

    Internal helper for generating error reports when schema
    compilation fails.
    """
    from greenlang.schema.models.report import TimingInfo

    findings = [
        Finding(
            code="GLSCHEMA-E502",
            severity=Severity.ERROR,
            path="",
            message=f"Schema compilation failed: {error}",
        )
        for error in errors
    ]

    summary = ValidationSummary(
        valid=False,
        error_count=len(findings),
        warning_count=0,
        info_count=0,
    )

    return ValidationReport(
        valid=False,
        schema_ref=schema_ref,
        schema_hash="0" * 64,
        summary=summary,
        findings=findings,
        normalized_payload=None,
        fix_suggestions=None,
        timings=TimingInfo(total_ms=0.0),
    )


def _create_batch_compile_error_report(
    schema_ref: SchemaRef,
    errors: List[str],
    num_payloads: int,
) -> BatchValidationReport:
    """
    Create a batch validation report for compilation errors.

    Internal helper for generating batch error reports when schema
    compilation fails.
    """
    from greenlang.schema.models.report import BatchSummary

    compile_findings = [
        Finding(
            code="GLSCHEMA-E502",
            severity=Severity.ERROR,
            path="",
            message=f"Schema compilation failed: {error}",
        )
        for error in errors
    ]

    results = [
        ItemResult(
            index=i,
            valid=False,
            findings=compile_findings,
        )
        for i in range(num_payloads)
    ]

    summary = BatchSummary(
        total_items=num_payloads,
        valid_count=0,
        error_count=num_payloads,
        warning_count=0,
    )

    return BatchValidationReport(
        schema_ref=schema_ref,
        schema_hash="0" * 64,
        summary=summary,
        results=results,
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Type aliases
    "SchemaInput",
    "PayloadInput",
    "ProfileInput",
    "PatchLevelInput",
    # Main classes
    "CompiledSchema",
    # Main functions
    "validate",
    "validate_batch",
    "compile_schema",
    # Fix suggestion helpers
    "apply_fixes",
    "safe_fixes",
    "review_fixes",
    # Finding helpers
    "errors_only",
    "warnings_only",
    "findings_by_path",
    "findings_by_code",
    # Schema reference helpers
    "parse_schema_ref",
    "schema_ref",
    # Re-exports for convenience
    "SchemaRef",
    "ValidationReport",
    "BatchValidationReport",
    "Finding",
    "FixSuggestion",
    "ValidationProfile",
    "PatchLevel",
]
