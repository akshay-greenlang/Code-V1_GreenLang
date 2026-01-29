# -*- coding: utf-8 -*-
"""
Schema Self-Validation for GL-FOUND-X-002.

This module validates schema documents themselves (not payloads) for governance
compliance per PRD section 6.8. It ensures that schema definitions are well-formed,
internally consistent, and follow GreenLang conventions.

Validation Checks:
    1. Reference resolution (no missing refs)
    2. Cycle detection with clear trace
    3. No duplicate property keys after alias resolution
    4. Deprecated field metadata well-formed
    5. Unit metadata consistent (unit in catalog, dimension specified)
    6. Constraints internally consistent (min <= max)
    7. Rule expressions are valid
    8. Pattern regexes are safe (no ReDoS)

Example:
    >>> from greenlang.schema.compiler.schema_validator import validate_schema
    >>> result = validate_schema({
    ...     "$schema": "https://json-schema.org/draft/2020-12/schema",
    ...     "type": "object",
    ...     "properties": {
    ...         "energy": {"type": "number", "minimum": 0, "maximum": 100}
    ...     }
    ... })
    >>> print(result.valid)
    True

    >>> # Invalid schema with min > max
    >>> result = validate_schema({
    ...     "type": "number",
    ...     "minimum": 100,
    ...     "maximum": 50
    ... })
    >>> print(result.valid)
    False
    >>> print(result.errors[0].code)
    GLSCHEMA-E509

Author: GreenLang Team
Date: 2026-01-29
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
    TYPE_CHECKING,
)

from pydantic import BaseModel, Field, ConfigDict

from greenlang.schema.constants import (
    MAX_REGEX_LENGTH,
    MAX_REGEX_COMPLEXITY_SCORE,
    SUPPORTED_DIALECTS,
)
from greenlang.schema.errors import ErrorCode, Severity

if TYPE_CHECKING:
    from greenlang.schema.units.catalog import UnitCatalog
    from greenlang.schema.models.schema_ref import SchemaRef

logger = logging.getLogger(__name__)


# ============================================================================
# PYDANTIC MODELS
# ============================================================================


class SchemaValidationFinding(BaseModel):
    """
    Finding from schema validation.

    Represents a single issue discovered during schema self-validation.
    Findings can be errors (which cause validation failure), warnings,
    or informational messages.

    Attributes:
        code: The GLSCHEMA-* error code (e.g., "GLSCHEMA-E509")
        severity: Severity level ("error", "warning", "info")
        path: JSON Pointer (RFC 6901) to the problem location
        message: Human-readable description of the issue
        details: Additional context-specific details

    Example:
        >>> finding = SchemaValidationFinding(
        ...     code="GLSCHEMA-E509",
        ...     severity="error",
        ...     path="/properties/value",
        ...     message="Constraint inconsistent: minimum (100) > maximum (50)"
        ... )
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str = Field(
        ...,
        description="GLSCHEMA-* error code",
        pattern=r"^GLSCHEMA-[EWI][0-9]{3}$",
    )
    severity: str = Field(
        ...,
        description="Severity level: error, warning, or info",
    )
    path: str = Field(
        ...,
        description="JSON Pointer to problem location",
    )
    message: str = Field(
        ...,
        description="Human-readable error message",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context details",
    )


class SchemaValidationResult(BaseModel):
    """
    Result of schema self-validation.

    Contains the overall validity status and all findings categorized
    by severity level. Also includes timing information for performance
    monitoring.

    Attributes:
        valid: True if no errors were found (warnings don't affect validity)
        schema_id: Optional schema identifier if available
        version: Optional schema version if available
        errors: List of error-level findings (cause validation failure)
        warnings: List of warning-level findings (advisory)
        info: List of informational findings (FYI)
        validation_time_ms: Time spent validating in milliseconds

    Example:
        >>> result = SchemaValidationResult(
        ...     valid=False,
        ...     schema_id="emissions/activity",
        ...     version="1.3.0",
        ...     errors=[finding],
        ...     validation_time_ms=12.5
        ... )
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    valid: bool = Field(
        ...,
        description="True if schema is valid (no errors)",
    )
    schema_id: Optional[str] = Field(
        default=None,
        description="Schema identifier if available",
    )
    version: Optional[str] = Field(
        default=None,
        description="Schema version if available",
    )
    errors: List[SchemaValidationFinding] = Field(
        default_factory=list,
        description="Error-level findings (cause validation failure)",
    )
    warnings: List[SchemaValidationFinding] = Field(
        default_factory=list,
        description="Warning-level findings (advisory)",
    )
    info: List[SchemaValidationFinding] = Field(
        default_factory=list,
        description="Informational findings",
    )
    validation_time_ms: float = Field(
        default=0.0,
        description="Validation duration in milliseconds",
    )

    @property
    def finding_count(self) -> int:
        """Total number of findings across all severity levels."""
        return len(self.errors) + len(self.warnings) + len(self.info)

    @property
    def error_count(self) -> int:
        """Number of error-level findings."""
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        """Number of warning-level findings."""
        return len(self.warnings)

    def to_summary(self) -> str:
        """Generate a human-readable summary."""
        status = "VALID" if self.valid else "INVALID"
        parts = [f"{status}"]
        if self.schema_id:
            parts.append(f"schema={self.schema_id}")
        if self.version:
            parts.append(f"v{self.version}")
        parts.append(
            f"({self.error_count} errors, {self.warning_count} warnings)"
        )
        return " ".join(parts)


# ============================================================================
# REGEX ANALYZER PROTOCOL (for type checking without circular import)
# ============================================================================


class RegexAnalyzerProtocol(Protocol):
    """Protocol for regex safety analyzer."""

    def analyze(self, pattern: str) -> "RegexAnalysisResult":
        """Analyze a regex pattern for safety."""
        ...


class RegexAnalysisResult:
    """Simple regex analysis result for inline implementation."""

    def __init__(
        self,
        pattern: str,
        is_safe: bool,
        complexity_score: float,
        vulnerability_type: Optional[str] = None,
        recommendation: str = "",
    ):
        self.pattern = pattern
        self.is_safe = is_safe
        self.complexity_score = complexity_score
        self.vulnerability_type = vulnerability_type
        self.recommendation = recommendation


# ============================================================================
# INLINE REGEX ANALYZER (minimal implementation for schema validation)
# ============================================================================


class RegexAnalyzer:
    """
    Regex safety analyzer for ReDoS detection.

    This is a minimal implementation for schema validation that detects
    common ReDoS patterns. For full functionality, use the dedicated
    RegexAnalyzer from greenlang.schema.compiler.regex_analyzer.

    Detected Patterns:
        - Nested quantifiers: (a+)+, (a*)+, (a+)*
        - Overlapping alternations: (a|a)+, (ab|a)+
        - Exponential backtracking: .*.*.*

    Example:
        >>> analyzer = RegexAnalyzer()
        >>> result = analyzer.analyze("(a+)+")
        >>> print(result.is_safe)
        False
    """

    # Patterns that indicate nested quantifiers
    NESTED_QUANTIFIER_PATTERNS = [
        re.compile(r"\([^)]*[+*]\)[+*?]"),  # (X+)+ or (X*)+ etc.
        re.compile(r"\([^)]*[+*]\)\{"),     # (X+){n,m}
        re.compile(r"\(\?:[^)]*[+*]\)[+*]"), # (?:X+)+ non-capturing
    ]

    # Patterns that indicate overlapping alternations
    OVERLAPPING_ALT_PATTERNS = [
        re.compile(r"\(([^|)]+)\|\1\)"),     # (X|X) exact duplicate
    ]

    # Patterns indicating excessive backtracking
    BACKTRACKING_PATTERNS = [
        re.compile(r"(\.\*){3,}"),           # .*.*.*
        re.compile(r"(\.\+){3,}"),           # .+.+.+
    ]

    def __init__(self, max_length: int = MAX_REGEX_LENGTH):
        """
        Initialize the regex analyzer.

        Args:
            max_length: Maximum allowed pattern length
        """
        self.max_length = max_length

    def analyze(self, pattern: str) -> RegexAnalysisResult:
        """
        Analyze a regex pattern for ReDoS vulnerability.

        Args:
            pattern: The regex pattern to analyze

        Returns:
            RegexAnalysisResult with safety assessment
        """
        # Check length
        if len(pattern) > self.max_length:
            return RegexAnalysisResult(
                pattern=pattern,
                is_safe=False,
                complexity_score=1.0,
                vulnerability_type="pattern_too_long",
                recommendation=f"Pattern exceeds maximum length ({self.max_length})",
            )

        # Check for nested quantifiers
        for regex in self.NESTED_QUANTIFIER_PATTERNS:
            if regex.search(pattern):
                return RegexAnalysisResult(
                    pattern=pattern,
                    is_safe=False,
                    complexity_score=0.9,
                    vulnerability_type="nested_quantifier",
                    recommendation=(
                        "Avoid nested quantifiers like (a+)+. Consider using "
                        "possessive quantifiers or atomic groups if available."
                    ),
                )

        # Check for overlapping alternations
        for regex in self.OVERLAPPING_ALT_PATTERNS:
            if regex.search(pattern):
                return RegexAnalysisResult(
                    pattern=pattern,
                    is_safe=False,
                    complexity_score=0.85,
                    vulnerability_type="overlapping_alternation",
                    recommendation=(
                        "Remove duplicate alternatives in alternation groups."
                    ),
                )

        # Check for excessive backtracking patterns
        for regex in self.BACKTRACKING_PATTERNS:
            if regex.search(pattern):
                return RegexAnalysisResult(
                    pattern=pattern,
                    is_safe=False,
                    complexity_score=0.95,
                    vulnerability_type="excessive_backtracking",
                    recommendation=(
                        "Reduce or eliminate consecutive .* or .+ quantifiers."
                    ),
                )

        # Calculate complexity score based on pattern characteristics
        complexity = self._calculate_complexity(pattern)

        return RegexAnalysisResult(
            pattern=pattern,
            is_safe=complexity < MAX_REGEX_COMPLEXITY_SCORE,
            complexity_score=complexity,
            vulnerability_type=None if complexity < MAX_REGEX_COMPLEXITY_SCORE else "high_complexity",
            recommendation="" if complexity < MAX_REGEX_COMPLEXITY_SCORE else "Consider simplifying the pattern",
        )

    def _calculate_complexity(self, pattern: str) -> float:
        """
        Calculate complexity score based on pattern characteristics.

        Args:
            pattern: The regex pattern

        Returns:
            Complexity score between 0.0 and 1.0
        """
        score = 0.0

        # Count quantifiers
        quantifiers = len(re.findall(r"[+*?]", pattern))
        score += min(quantifiers * 0.05, 0.3)

        # Count alternations
        alternations = pattern.count("|")
        score += min(alternations * 0.05, 0.2)

        # Count groups
        groups = len(re.findall(r"\((?!\?)", pattern))
        score += min(groups * 0.03, 0.15)

        # Check for wildcards with quantifiers
        wild_quant = len(re.findall(r"\.[+*]", pattern))
        score += min(wild_quant * 0.1, 0.3)

        # Length factor
        score += min(len(pattern) / 1000, 0.1)

        return min(score, 1.0)

    def is_valid_regex(self, pattern: str) -> Tuple[bool, Optional[str]]:
        """
        Check if pattern is a valid regex.

        Args:
            pattern: The regex pattern to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            re.compile(pattern)
            return True, None
        except re.error as e:
            return False, str(e)


# ============================================================================
# SEMVER VALIDATION
# ============================================================================

# Semantic version pattern
SEMVER_PATTERN = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(-((0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(\.(0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(\+([0-9a-zA-Z-]+(\.[0-9a-zA-Z-]+)*))?$"
)


def is_valid_semver(version: str) -> bool:
    """Check if version string is valid semver."""
    return bool(SEMVER_PATTERN.match(version))


def compare_semver(v1: str, v2: str) -> int:
    """
    Compare two semver versions.

    Returns:
        -1 if v1 < v2, 0 if v1 == v2, 1 if v1 > v2
    """
    def parse_version(v: str) -> Tuple[int, int, int, str]:
        # Remove prerelease and build metadata for comparison
        base = v.split("-")[0].split("+")[0]
        parts = base.split(".")
        prerelease = v.split("-")[1].split("+")[0] if "-" in v else ""
        return (int(parts[0]), int(parts[1]), int(parts[2]), prerelease)

    p1 = parse_version(v1)
    p2 = parse_version(v2)

    # Compare major.minor.patch
    for i in range(3):
        if p1[i] < p2[i]:
            return -1
        if p1[i] > p2[i]:
            return 1

    # Compare prerelease (empty prerelease is greater than any prerelease)
    if p1[3] == "" and p2[3] != "":
        return 1
    if p1[3] != "" and p2[3] == "":
        return -1
    if p1[3] < p2[3]:
        return -1
    if p1[3] > p2[3]:
        return 1

    return 0


# ============================================================================
# SCHEMA VALIDATOR
# ============================================================================


class SchemaValidator:
    """
    Validates schema documents for governance compliance.

    This validator checks schema documents (not payloads) for internal
    consistency, correct references, safe patterns, and proper metadata.
    It implements the governance checks specified in PRD section 6.8.

    Validation Phases:
        1. Structure validation (basic schema structure)
        2. Reference validation (no missing refs, no cycles)
        3. Key uniqueness (no duplicates after alias resolution)
        4. Deprecation metadata (well-formed)
        5. Unit metadata (consistent with catalog)
        6. Constraint consistency (min <= max)
        7. Pattern safety (no ReDoS)
        8. Rule expression validation

    Thread Safety:
        SchemaValidator instances are NOT thread-safe. Create a new instance
        per thread or use proper synchronization.

    Example:
        >>> validator = SchemaValidator()
        >>> result = validator.validate({
        ...     "type": "object",
        ...     "properties": {
        ...         "value": {"type": "number", "minimum": 0, "maximum": 100}
        ...     }
        ... })
        >>> print(result.valid)
        True

        >>> # Invalid: min > max
        >>> result = validator.validate({
        ...     "type": "number",
        ...     "minimum": 100,
        ...     "maximum": 50
        ... })
        >>> print(result.valid)
        False
    """

    def __init__(
        self,
        unit_catalog: Optional["UnitCatalog"] = None,
        regex_analyzer: Optional[RegexAnalyzer] = None,
        strict: bool = True,
    ):
        """
        Initialize the SchemaValidator.

        Args:
            unit_catalog: Optional unit catalog for unit validation.
                If None, unit validation is skipped.
            regex_analyzer: Optional regex analyzer for pattern safety.
                If None, a default analyzer is created.
            strict: If True, apply strict validation rules.
                If False, some checks become warnings instead of errors.
        """
        self.unit_catalog = unit_catalog
        self.regex_analyzer = regex_analyzer or RegexAnalyzer()
        self.strict = strict

        # Validation state (reset per validation call)
        self._findings: List[SchemaValidationFinding] = []
        self._resolved_refs: Set[str] = set()
        self._resolution_stack: List[str] = []
        self._known_definitions: Set[str] = set()

        logger.debug(
            f"SchemaValidator initialized: strict={strict}, "
            f"unit_catalog={'present' if unit_catalog else 'None'}"
        )

    def validate(
        self,
        schema_source: Union[str, Dict[str, Any]],
        schema_ref: Optional["SchemaRef"] = None,
    ) -> SchemaValidationResult:
        """
        Validate a schema document.

        This is the main entry point for schema validation. It runs all
        validation checks and returns a comprehensive result.

        Args:
            schema_source: Schema as string (YAML/JSON) or dict
            schema_ref: Optional schema reference for context

        Returns:
            SchemaValidationResult with all findings

        Example:
            >>> validator = SchemaValidator()
            >>> result = validator.validate({
            ...     "type": "object",
            ...     "properties": {"name": {"type": "string"}}
            ... })
            >>> print(result.valid)
            True
        """
        start_time = time.perf_counter()

        # Reset state
        self._findings = []
        self._resolved_refs = set()
        self._resolution_stack = []
        self._known_definitions = set()

        # Parse schema if string
        schema: Dict[str, Any]
        if isinstance(schema_source, str):
            try:
                schema = self._parse_schema_string(schema_source)
            except Exception as e:
                self._add_finding(
                    code=ErrorCode.SCHEMA_PARSE_ERROR.value,
                    severity="error",
                    path="",
                    message=f"Failed to parse schema: {e}",
                    error=str(e),
                )
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                return self._build_result(None, None, elapsed_ms)
        else:
            schema = schema_source

        # Extract schema ID and version if available
        schema_id = schema.get("$id") or (schema_ref.schema_id if schema_ref else None)
        version = schema.get("x-gl-version") or (schema_ref.version if schema_ref else None)

        # Build index of definitions for reference checking
        self._index_definitions(schema)

        # Run all validation checks
        self._validate_structure(schema)
        self._validate_references(schema, "")
        self._validate_no_duplicate_keys(schema, "")
        self._validate_deprecation_metadata(schema, "")
        self._validate_unit_metadata(schema, "")
        self._validate_constraints_consistent(schema, "")
        self._validate_patterns_safe(schema, "")
        self._validate_rule_expressions(schema, "")

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            f"Schema validation complete: "
            f"schema_id={schema_id}, "
            f"valid={len([f for f in self._findings if f.severity == 'error']) == 0}, "
            f"findings={len(self._findings)}, "
            f"time_ms={elapsed_ms:.2f}"
        )

        return self._build_result(schema_id, version, elapsed_ms)

    def _parse_schema_string(self, content: str) -> Dict[str, Any]:
        """
        Parse schema from string (JSON or YAML).

        Args:
            content: Schema content as string

        Returns:
            Parsed schema dictionary

        Raises:
            ValueError: If parsing fails
        """
        # Try JSON first
        try:
            result = json.loads(content)
            if not isinstance(result, dict):
                raise ValueError(
                    f"Schema must be a JSON object, got {type(result).__name__}"
                )
            return result
        except json.JSONDecodeError:
            pass

        # Try YAML
        try:
            import yaml
            result = yaml.safe_load(content)
            if not isinstance(result, dict):
                raise ValueError(
                    f"Schema must be a YAML mapping, got {type(result).__name__}"
                )
            return result
        except ImportError:
            raise ValueError(
                "Content is not valid JSON and PyYAML is not installed for YAML parsing"
            )
        except Exception as e:
            raise ValueError(f"Failed to parse as JSON or YAML: {e}")

    def _index_definitions(self, schema: Dict[str, Any]) -> None:
        """
        Build index of all definitions in the schema.

        This allows reference validation to check if targets exist.

        Args:
            schema: The schema document
        """
        # Index $defs (JSON Schema Draft 2019-09+)
        if "$defs" in schema:
            for name in schema["$defs"]:
                self._known_definitions.add(f"#/$defs/{name}")

        # Index definitions (JSON Schema Draft-07 style)
        if "definitions" in schema:
            for name in schema["definitions"]:
                self._known_definitions.add(f"#/definitions/{name}")

        logger.debug(f"Indexed {len(self._known_definitions)} definitions")

    def _validate_structure(self, schema: Dict[str, Any]) -> None:
        """
        Validate basic schema structure.

        Checks:
        - Schema is a valid object
        - $schema dialect is supported (if specified)
        - type keyword has valid value

        Args:
            schema: The schema document
        """
        # Check $schema dialect
        if "$schema" in schema:
            dialect = schema["$schema"]
            if dialect not in SUPPORTED_DIALECTS:
                self._add_finding(
                    code=ErrorCode.SCHEMA_DIALECT_UNSUPPORTED.value,
                    severity="warning",
                    path="/$schema",
                    message=f"Schema dialect '{dialect}' is not in supported list",
                    dialect=dialect,
                    supported=list(SUPPORTED_DIALECTS),
                )

        # Validate type keyword if present
        if "type" in schema:
            valid_types = {"string", "number", "integer", "boolean", "object", "array", "null"}
            schema_type = schema["type"]

            if isinstance(schema_type, str):
                if schema_type not in valid_types:
                    self._add_finding(
                        code=ErrorCode.SCHEMA_KEYWORD_INVALID.value,
                        severity="error",
                        path="/type",
                        message=f"Invalid type '{schema_type}'",
                        valid_types=list(valid_types),
                    )
            elif isinstance(schema_type, list):
                for t in schema_type:
                    if t not in valid_types:
                        self._add_finding(
                            code=ErrorCode.SCHEMA_KEYWORD_INVALID.value,
                            severity="error",
                            path="/type",
                            message=f"Invalid type in array: '{t}'",
                            valid_types=list(valid_types),
                        )

    def _validate_references(
        self,
        schema: Dict[str, Any],
        path: str = "",
        root_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Validate all $ref references can be resolved.

        Checks:
        - All $ref targets exist
        - No circular references

        Args:
            schema: The schema document or subschema
            path: Current JSON Pointer path
            root_schema: The root schema document (for resolving local refs)
        """
        if not isinstance(schema, dict):
            return

        # Use root_schema for resolving refs, default to schema on first call
        if root_schema is None:
            root_schema = schema

        # Check for $ref
        if "$ref" in schema:
            ref = schema["$ref"]
            ref_path = f"{path}/$ref"

            # Check for cycles
            if ref in self._resolution_stack:
                cycle = self._resolution_stack[self._resolution_stack.index(ref):] + [ref]
                self._add_finding(
                    code=ErrorCode.CIRCULAR_REF.value,
                    severity="error",
                    path=ref_path,
                    message=f"Circular reference detected: {' -> '.join(cycle)}",
                    cycle=cycle,
                )
                return

            # Track resolution for cycle detection
            self._resolution_stack.append(ref)

            try:
                # Check if ref is resolvable (local refs only for now)
                if ref.startswith("#"):
                    # Try to resolve the ref target
                    target = self._resolve_local_ref(root_schema, ref)

                    if target is None:
                        self._add_finding(
                            code=ErrorCode.REF_RESOLUTION_FAILED.value,
                            severity="error",
                            path=ref_path,
                            message=f"Cannot resolve local $ref '{ref}'",
                            ref=ref,
                            known_definitions=list(self._known_definitions)[:10],
                        )
                    else:
                        # Follow the ref to check for cycles in the target
                        self._validate_references(target, ref, root_schema)

                # Mark as resolved
                self._resolved_refs.add(ref)

            finally:
                self._resolution_stack.pop()

        # Recurse into nested schemas
        for key, value in schema.items():
            if key in ("$ref",):
                continue

            new_path = f"{path}/{key}"

            if isinstance(value, dict):
                self._validate_references(value, new_path, root_schema)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        self._validate_references(item, f"{new_path}/{i}", root_schema)

    def _resolve_local_ref(
        self,
        root_schema: Dict[str, Any],
        ref: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Resolve a local $ref to its target schema.

        Args:
            root_schema: The root schema document
            ref: The $ref value (e.g., "#/$defs/Name")

        Returns:
            The target schema dict, or None if not found
        """
        if not ref.startswith("#"):
            return None

        pointer = ref[1:]  # Remove #
        if not pointer or pointer == "/":
            return root_schema

        # Navigate the path
        parts = pointer.split("/")[1:]  # Skip empty first element
        current: Any = root_schema

        for part in parts:
            # Unescape JSON Pointer tokens
            part = part.replace("~1", "/").replace("~0", "~")

            if isinstance(current, dict) and part in current:
                current = current[part]
            elif isinstance(current, list):
                try:
                    idx = int(part)
                    if 0 <= idx < len(current):
                        current = current[idx]
                    else:
                        return None
                except ValueError:
                    return None
            else:
                return None

        return current if isinstance(current, dict) else None

    def _path_exists_in_schema(self, root_schema: Dict[str, Any], ref: str) -> bool:
        """
        Check if a JSON Pointer path exists in the schema.

        Args:
            root_schema: The root schema document
            ref: JSON Pointer reference (starting with #)

        Returns:
            True if path exists
        """
        return self._resolve_local_ref(root_schema, ref) is not None

    def _validate_no_duplicate_keys(
        self,
        schema: Dict[str, Any],
        path: str = "",
    ) -> None:
        """
        Check for duplicate property keys after alias resolution.

        If a schema defines aliases (via x-gl-aliases), ensure no
        duplicate keys would result after alias expansion.

        Args:
            schema: The schema document or subschema
            path: Current JSON Pointer path
        """
        if not isinstance(schema, dict):
            return

        # Check properties for duplicates after alias resolution
        if "properties" in schema:
            properties = schema["properties"]
            aliases = schema.get("x-gl-aliases", {})

            # Build set of effective keys (including aliases)
            effective_keys: Dict[str, List[str]] = {}

            for prop_name in properties:
                # Canonical name
                canonical = prop_name
                if canonical not in effective_keys:
                    effective_keys[canonical] = []
                effective_keys[canonical].append(prop_name)

                # Check if this property has aliases that collide
                if prop_name in aliases:
                    alias_target = aliases[prop_name]
                    if alias_target in properties and alias_target != prop_name:
                        self._add_finding(
                            code=ErrorCode.DUPLICATE_KEY.value,
                            severity="error",
                            path=f"{path}/properties/{prop_name}",
                            message=(
                                f"Property '{prop_name}' has alias '{alias_target}' "
                                f"which is also defined as a property"
                            ),
                            property=prop_name,
                            alias=alias_target,
                        )

            # Check aliases pointing to same target
            alias_targets: Dict[str, List[str]] = {}
            for alias, target in aliases.items():
                if target not in alias_targets:
                    alias_targets[target] = []
                alias_targets[target].append(alias)

            for target, sources in alias_targets.items():
                if len(sources) > 1:
                    self._add_finding(
                        code=ErrorCode.DUPLICATE_KEY.value,
                        severity="warning",
                        path=f"{path}/x-gl-aliases",
                        message=(
                            f"Multiple aliases point to '{target}': {sources}"
                        ),
                        target=target,
                        aliases=sources,
                    )

        # Recurse
        for key, value in schema.items():
            new_path = f"{path}/{key}"
            if isinstance(value, dict):
                self._validate_no_duplicate_keys(value, new_path)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        self._validate_no_duplicate_keys(item, f"{new_path}/{i}")

    def _validate_deprecation_metadata(
        self,
        schema: Dict[str, Any],
        path: str = "",
    ) -> None:
        """
        Validate deprecated field metadata is well-formed.

        Checks:
        - since_version is valid semver
        - replacement field exists if specified
        - removal_version > since_version

        Args:
            schema: The schema document or subschema
            path: Current JSON Pointer path
        """
        if not isinstance(schema, dict):
            return

        # Check x-gl-deprecated metadata
        if "x-gl-deprecated" in schema:
            dep = schema["x-gl-deprecated"]
            dep_path = f"{path}/x-gl-deprecated"

            if not isinstance(dep, dict):
                self._add_finding(
                    code=ErrorCode.SCHEMA_KEYWORD_INVALID.value,
                    severity="error",
                    path=dep_path,
                    message="x-gl-deprecated must be an object",
                )
            else:
                # Validate since_version
                if "since_version" in dep:
                    since = dep["since_version"]
                    if not is_valid_semver(since):
                        self._add_finding(
                            code=ErrorCode.SCHEMA_KEYWORD_INVALID.value,
                            severity="error",
                            path=f"{dep_path}/since_version",
                            message=f"Invalid semver: '{since}'",
                            value=since,
                        )

                # Validate removal_version
                if "removal_version" in dep:
                    removal = dep["removal_version"]
                    if not is_valid_semver(removal):
                        self._add_finding(
                            code=ErrorCode.SCHEMA_KEYWORD_INVALID.value,
                            severity="error",
                            path=f"{dep_path}/removal_version",
                            message=f"Invalid semver: '{removal}'",
                            value=removal,
                        )
                    elif "since_version" in dep:
                        since = dep["since_version"]
                        if is_valid_semver(since) and compare_semver(removal, since) <= 0:
                            self._add_finding(
                                code=ErrorCode.SCHEMA_CONSTRAINT_INCONSISTENT.value,
                                severity="error",
                                path=dep_path,
                                message=(
                                    f"removal_version ({removal}) must be greater than "
                                    f"since_version ({since})"
                                ),
                                since_version=since,
                                removal_version=removal,
                            )

                # Check replacement exists (if we can verify it)
                if "replacement" in dep:
                    replacement = dep["replacement"]
                    # If replacement is a local path, try to verify it exists
                    if replacement.startswith("/"):
                        ref = f"#{replacement}"
                        if not self._path_exists_in_schema(schema, ref):
                            self._add_finding(
                                code=ErrorCode.SCHEMA_KEYWORD_INVALID.value,
                                severity="warning",
                                path=f"{dep_path}/replacement",
                                message=f"Replacement path '{replacement}' may not exist",
                                replacement=replacement,
                            )

        # Check deprecated keyword (JSON Schema)
        if schema.get("deprecated") is True:
            # Should have x-gl-deprecated for full metadata
            if "x-gl-deprecated" not in schema:
                self._add_finding(
                    code=ErrorCode.SCHEMA_KEYWORD_INVALID.value,
                    severity="info",
                    path=path,
                    message=(
                        "Field marked 'deprecated' but missing x-gl-deprecated metadata "
                        "(since_version, replacement, message)"
                    ),
                )

        # Recurse into properties
        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                self._validate_deprecation_metadata(
                    prop_schema, f"{path}/properties/{prop_name}"
                )

        # Recurse into other schema keywords
        for key in ("items", "additionalProperties", "allOf", "anyOf", "oneOf", "not"):
            if key in schema:
                value = schema[key]
                if isinstance(value, dict):
                    self._validate_deprecation_metadata(value, f"{path}/{key}")
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            self._validate_deprecation_metadata(item, f"{path}/{key}/{i}")

    def _validate_unit_metadata(
        self,
        schema: Dict[str, Any],
        path: str = "",
    ) -> None:
        """
        Validate unit metadata is consistent.

        Checks:
        - unit exists in catalog
        - dimension is specified
        - canonical unit is convertible from allowed units

        Args:
            schema: The schema document or subschema
            path: Current JSON Pointer path
        """
        if not isinstance(schema, dict):
            return

        # Check x-gl-unit metadata
        if "x-gl-unit" in schema:
            unit_spec = schema["x-gl-unit"]
            unit_path = f"{path}/x-gl-unit"

            if not isinstance(unit_spec, dict):
                self._add_finding(
                    code=ErrorCode.SCHEMA_KEYWORD_INVALID.value,
                    severity="error",
                    path=unit_path,
                    message="x-gl-unit must be an object",
                )
            else:
                # Check dimension is specified
                if "dimension" not in unit_spec:
                    self._add_finding(
                        code=ErrorCode.DIMENSION_INVALID.value,
                        severity="error",
                        path=unit_path,
                        message="x-gl-unit must specify 'dimension'",
                    )

                # Check canonical unit
                if "canonical" not in unit_spec:
                    self._add_finding(
                        code=ErrorCode.UNIT_MISSING.value,
                        severity="error",
                        path=unit_path,
                        message="x-gl-unit must specify 'canonical' unit",
                    )

                # If we have a unit catalog, validate units exist
                if self.unit_catalog is not None:
                    dimension = unit_spec.get("dimension")
                    canonical = unit_spec.get("canonical")
                    allowed = unit_spec.get("allowed", [])

                    # Check canonical unit exists
                    if canonical and not self.unit_catalog.is_known(canonical):
                        self._add_finding(
                            code=ErrorCode.UNIT_UNKNOWN.value,
                            severity="error",
                            path=f"{unit_path}/canonical",
                            message=f"Unknown unit '{canonical}' not in catalog",
                            unit=canonical,
                        )
                    elif canonical and dimension:
                        # Check canonical unit has correct dimension
                        unit_dim = self.unit_catalog.get_dimension(canonical)
                        if unit_dim and unit_dim != dimension:
                            self._add_finding(
                                code=ErrorCode.UNIT_INCOMPATIBLE.value,
                                severity="error",
                                path=f"{unit_path}/canonical",
                                message=(
                                    f"Canonical unit '{canonical}' has dimension '{unit_dim}' "
                                    f"but schema specifies dimension '{dimension}'"
                                ),
                                unit=canonical,
                                expected_dimension=dimension,
                                actual_dimension=unit_dim,
                            )

                    # Check allowed units
                    for i, unit in enumerate(allowed):
                        if not self.unit_catalog.is_known(unit):
                            self._add_finding(
                                code=ErrorCode.UNIT_UNKNOWN.value,
                                severity="error",
                                path=f"{unit_path}/allowed/{i}",
                                message=f"Unknown unit '{unit}' not in catalog",
                                unit=unit,
                            )
                        elif canonical and not self.unit_catalog.is_compatible(unit, canonical):
                            self._add_finding(
                                code=ErrorCode.UNIT_INCOMPATIBLE.value,
                                severity="error",
                                path=f"{unit_path}/allowed/{i}",
                                message=(
                                    f"Allowed unit '{unit}' is not compatible with "
                                    f"canonical unit '{canonical}'"
                                ),
                                unit=unit,
                                canonical=canonical,
                            )

        # Recurse
        for key, value in schema.items():
            if key == "x-gl-unit":
                continue
            new_path = f"{path}/{key}"
            if isinstance(value, dict):
                self._validate_unit_metadata(value, new_path)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        self._validate_unit_metadata(item, f"{new_path}/{i}")

    def _validate_constraints_consistent(
        self,
        schema: Dict[str, Any],
        path: str = "",
    ) -> None:
        """
        Validate constraints are internally consistent.

        Checks:
        - min <= max for numbers
        - minLength <= maxLength for strings
        - minItems <= maxItems for arrays
        - enum values match declared type
        - exclusiveMinimum < exclusiveMaximum

        Args:
            schema: The schema document or subschema
            path: Current JSON Pointer path
        """
        if not isinstance(schema, dict):
            return

        # Check numeric constraints
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        exc_min = schema.get("exclusiveMinimum")
        exc_max = schema.get("exclusiveMaximum")

        if minimum is not None and maximum is not None:
            if minimum > maximum:
                self._add_finding(
                    code=ErrorCode.SCHEMA_CONSTRAINT_INCONSISTENT.value,
                    severity="error",
                    path=path,
                    message=f"minimum ({minimum}) > maximum ({maximum})",
                    minimum=minimum,
                    maximum=maximum,
                )

        if exc_min is not None and exc_max is not None:
            if exc_min >= exc_max:
                self._add_finding(
                    code=ErrorCode.SCHEMA_CONSTRAINT_INCONSISTENT.value,
                    severity="error",
                    path=path,
                    message=f"exclusiveMinimum ({exc_min}) >= exclusiveMaximum ({exc_max})",
                    exclusiveMinimum=exc_min,
                    exclusiveMaximum=exc_max,
                )

        # Check min/excMin vs max/excMax
        if minimum is not None and exc_max is not None:
            if minimum >= exc_max:
                self._add_finding(
                    code=ErrorCode.SCHEMA_CONSTRAINT_INCONSISTENT.value,
                    severity="error",
                    path=path,
                    message=f"minimum ({minimum}) >= exclusiveMaximum ({exc_max})",
                    minimum=minimum,
                    exclusiveMaximum=exc_max,
                )

        if exc_min is not None and maximum is not None:
            if exc_min >= maximum:
                self._add_finding(
                    code=ErrorCode.SCHEMA_CONSTRAINT_INCONSISTENT.value,
                    severity="error",
                    path=path,
                    message=f"exclusiveMinimum ({exc_min}) >= maximum ({maximum})",
                    exclusiveMinimum=exc_min,
                    maximum=maximum,
                )

        # Check string constraints
        min_length = schema.get("minLength")
        max_length = schema.get("maxLength")

        if min_length is not None and max_length is not None:
            if min_length > max_length:
                self._add_finding(
                    code=ErrorCode.SCHEMA_CONSTRAINT_INCONSISTENT.value,
                    severity="error",
                    path=path,
                    message=f"minLength ({min_length}) > maxLength ({max_length})",
                    minLength=min_length,
                    maxLength=max_length,
                )

        # Check array constraints
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")

        if min_items is not None and max_items is not None:
            if min_items > max_items:
                self._add_finding(
                    code=ErrorCode.SCHEMA_CONSTRAINT_INCONSISTENT.value,
                    severity="error",
                    path=path,
                    message=f"minItems ({min_items}) > maxItems ({max_items})",
                    minItems=min_items,
                    maxItems=max_items,
                )

        # Check object property constraints
        min_props = schema.get("minProperties")
        max_props = schema.get("maxProperties")

        if min_props is not None and max_props is not None:
            if min_props > max_props:
                self._add_finding(
                    code=ErrorCode.SCHEMA_CONSTRAINT_INCONSISTENT.value,
                    severity="error",
                    path=path,
                    message=f"minProperties ({min_props}) > maxProperties ({max_props})",
                    minProperties=min_props,
                    maxProperties=max_props,
                )

        # Check enum values match type
        if "enum" in schema and "type" in schema:
            schema_type = schema["type"]
            enum_values = schema["enum"]

            if isinstance(schema_type, str):
                for i, val in enumerate(enum_values):
                    if not self._value_matches_type(val, schema_type):
                        self._add_finding(
                            code=ErrorCode.SCHEMA_CONSTRAINT_INCONSISTENT.value,
                            severity="error",
                            path=f"{path}/enum/{i}",
                            message=(
                                f"Enum value {val!r} does not match declared type '{schema_type}'"
                            ),
                            value=val,
                            expected_type=schema_type,
                        )

        # Check const matches type
        if "const" in schema and "type" in schema:
            schema_type = schema["type"]
            const_val = schema["const"]

            if isinstance(schema_type, str):
                if not self._value_matches_type(const_val, schema_type):
                    self._add_finding(
                        code=ErrorCode.SCHEMA_CONSTRAINT_INCONSISTENT.value,
                        severity="error",
                        path=f"{path}/const",
                        message=(
                            f"Const value {const_val!r} does not match declared type '{schema_type}'"
                        ),
                        value=const_val,
                        expected_type=schema_type,
                    )

        # Check minContains/maxContains
        min_contains = schema.get("minContains")
        max_contains = schema.get("maxContains")

        if min_contains is not None and max_contains is not None:
            if min_contains > max_contains:
                self._add_finding(
                    code=ErrorCode.SCHEMA_CONSTRAINT_INCONSISTENT.value,
                    severity="error",
                    path=path,
                    message=f"minContains ({min_contains}) > maxContains ({max_contains})",
                    minContains=min_contains,
                    maxContains=max_contains,
                )

        # Recurse
        for key, value in schema.items():
            new_path = f"{path}/{key}"
            if isinstance(value, dict):
                self._validate_constraints_consistent(value, new_path)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        self._validate_constraints_consistent(item, f"{new_path}/{i}")

    def _value_matches_type(self, value: Any, schema_type: str) -> bool:
        """
        Check if a value matches a JSON Schema type.

        Args:
            value: The value to check
            schema_type: The expected JSON Schema type

        Returns:
            True if value matches type
        """
        type_checks = {
            "string": lambda v: isinstance(v, str),
            "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "boolean": lambda v: isinstance(v, bool),
            "array": lambda v: isinstance(v, list),
            "object": lambda v: isinstance(v, dict),
            "null": lambda v: v is None,
        }

        check = type_checks.get(schema_type)
        if check is None:
            return True  # Unknown type, assume valid

        return check(value)

    def _validate_patterns_safe(
        self,
        schema: Dict[str, Any],
        path: str = "",
    ) -> None:
        """
        Validate regex patterns are safe (no ReDoS).

        Uses RegexAnalyzer to detect potentially dangerous patterns.

        Args:
            schema: The schema document or subschema
            path: Current JSON Pointer path
        """
        if not isinstance(schema, dict):
            return

        # Check pattern keyword
        if "pattern" in schema:
            pattern = schema["pattern"]
            pattern_path = f"{path}/pattern"

            # First, check if it's a valid regex
            is_valid, error = self.regex_analyzer.is_valid_regex(pattern)
            if not is_valid:
                self._add_finding(
                    code=ErrorCode.SCHEMA_KEYWORD_INVALID.value,
                    severity="error",
                    path=pattern_path,
                    message=f"Invalid regex pattern: {error}",
                    pattern=pattern,
                    error=error,
                )
            else:
                # Analyze for ReDoS
                result = self.regex_analyzer.analyze(pattern)
                if not result.is_safe:
                    severity = "error" if self.strict else "warning"
                    self._add_finding(
                        code=ErrorCode.REGEX_TOO_COMPLEX.value,
                        severity=severity,
                        path=pattern_path,
                        message=(
                            f"Potentially unsafe regex pattern: {result.vulnerability_type}. "
                            f"{result.recommendation}"
                        ),
                        pattern=pattern,
                        complexity_score=result.complexity_score,
                        vulnerability_type=result.vulnerability_type,
                    )

        # Check patternProperties keys
        if "patternProperties" in schema:
            for pattern, prop_schema in schema["patternProperties"].items():
                pattern_path = f"{path}/patternProperties/{pattern}"

                is_valid, error = self.regex_analyzer.is_valid_regex(pattern)
                if not is_valid:
                    self._add_finding(
                        code=ErrorCode.SCHEMA_KEYWORD_INVALID.value,
                        severity="error",
                        path=pattern_path,
                        message=f"Invalid regex in patternProperties key: {error}",
                        pattern=pattern,
                        error=error,
                    )
                else:
                    result = self.regex_analyzer.analyze(pattern)
                    if not result.is_safe:
                        severity = "error" if self.strict else "warning"
                        self._add_finding(
                            code=ErrorCode.REGEX_TOO_COMPLEX.value,
                            severity=severity,
                            path=pattern_path,
                            message=(
                                f"Potentially unsafe regex in patternProperties: "
                                f"{result.vulnerability_type}. {result.recommendation}"
                            ),
                            pattern=pattern,
                            complexity_score=result.complexity_score,
                        )

                # Recurse into the property schema
                self._validate_patterns_safe(prop_schema, pattern_path)

        # Recurse
        for key, value in schema.items():
            if key in ("pattern", "patternProperties"):
                continue
            new_path = f"{path}/{key}"
            if isinstance(value, dict):
                self._validate_patterns_safe(value, new_path)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        self._validate_patterns_safe(item, f"{new_path}/{i}")

    def _validate_rule_expressions(
        self,
        schema: Dict[str, Any],
        path: str = "",
    ) -> None:
        """
        Validate rule expressions.

        Checks:
        - Referenced paths exist in schema
        - Operators are valid
        - Types are compatible

        Args:
            schema: The schema document or subschema
            path: Current JSON Pointer path
        """
        if not isinstance(schema, dict):
            return

        # Check x-gl-rules
        if "x-gl-rules" in schema:
            rules = schema["x-gl-rules"]
            rules_path = f"{path}/x-gl-rules"

            if not isinstance(rules, list):
                self._add_finding(
                    code=ErrorCode.SCHEMA_KEYWORD_INVALID.value,
                    severity="error",
                    path=rules_path,
                    message="x-gl-rules must be an array",
                )
            else:
                for i, rule in enumerate(rules):
                    rule_path = f"{rules_path}/{i}"
                    self._validate_single_rule(rule, rule_path, schema)

        # Recurse
        for key, value in schema.items():
            if key == "x-gl-rules":
                continue
            new_path = f"{path}/{key}"
            if isinstance(value, dict):
                self._validate_rule_expressions(value, new_path)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        self._validate_rule_expressions(item, f"{new_path}/{i}")

    def _validate_single_rule(
        self,
        rule: Any,
        path: str,
        context_schema: Dict[str, Any],
    ) -> None:
        """
        Validate a single rule definition.

        Args:
            rule: The rule definition
            path: Path to the rule
            context_schema: The schema containing the rule (for path validation)
        """
        if not isinstance(rule, dict):
            self._add_finding(
                code=ErrorCode.SCHEMA_KEYWORD_INVALID.value,
                severity="error",
                path=path,
                message="Rule must be an object",
            )
            return

        # Check required fields
        if "id" not in rule:
            self._add_finding(
                code=ErrorCode.SCHEMA_KEYWORD_INVALID.value,
                severity="error",
                path=path,
                message="Rule must have 'id' field",
            )

        if "check" not in rule:
            self._add_finding(
                code=ErrorCode.SCHEMA_KEYWORD_INVALID.value,
                severity="error",
                path=path,
                message="Rule must have 'check' expression",
            )
        else:
            # Validate the check expression
            self._validate_rule_expression(rule["check"], f"{path}/check", context_schema)

        # Validate optional 'when' condition
        if "when" in rule:
            self._validate_rule_expression(rule["when"], f"{path}/when", context_schema)

        # Validate severity if specified
        if "severity" in rule:
            valid_severities = {"error", "warning", "info"}
            if rule["severity"] not in valid_severities:
                self._add_finding(
                    code=ErrorCode.SCHEMA_KEYWORD_INVALID.value,
                    severity="error",
                    path=f"{path}/severity",
                    message=f"Invalid severity '{rule['severity']}'. Must be one of: {valid_severities}",
                    value=rule["severity"],
                    valid_values=list(valid_severities),
                )

    def _validate_rule_expression(
        self,
        expr: Any,
        path: str,
        context_schema: Dict[str, Any],
    ) -> None:
        """
        Validate a rule expression.

        Args:
            expr: The expression to validate
            path: Path to the expression
            context_schema: The schema containing properties (for path validation)
        """
        if not isinstance(expr, dict):
            # Simple expressions (constants) are valid
            return

        # Known operators
        valid_operators = {
            "eq", "ne", "lt", "le", "gt", "ge",  # Comparison
            "and", "or", "not",                   # Logical
            "in", "contains",                      # Containment
            "sum", "min", "max", "avg", "count",  # Aggregation
            "path", "ref",                         # Path reference
            "if", "then", "else",                 # Conditional
        }

        for op, operand in expr.items():
            if op not in valid_operators:
                self._add_finding(
                    code=ErrorCode.SCHEMA_KEYWORD_INVALID.value,
                    severity="warning",
                    path=path,
                    message=f"Unknown rule operator '{op}'",
                    operator=op,
                    valid_operators=list(valid_operators),
                )

            # Validate path references
            if op == "path":
                if isinstance(operand, str):
                    # Check if path exists in schema properties
                    self._validate_rule_path(operand, f"{path}/{op}", context_schema)

            # Recurse into nested expressions
            if isinstance(operand, dict):
                self._validate_rule_expression(operand, f"{path}/{op}", context_schema)
            elif isinstance(operand, list):
                for i, item in enumerate(operand):
                    if isinstance(item, dict):
                        self._validate_rule_expression(item, f"{path}/{op}/{i}", context_schema)

    def _validate_rule_path(
        self,
        rule_path: str,
        location: str,
        context_schema: Dict[str, Any],
    ) -> None:
        """
        Validate that a path in a rule expression exists in the schema.

        Args:
            rule_path: The path referenced in the rule (e.g., "total", "items.*.value")
            location: Location of the path reference for error reporting
            context_schema: The schema containing properties
        """
        # Skip validation for complex path patterns (wildcards, etc.)
        if "*" in rule_path or ".." in rule_path:
            return

        # Get properties from context schema
        properties = context_schema.get("properties", {})

        # Simple single-level path check
        parts = rule_path.split(".")
        first_part = parts[0]

        if first_part and first_part not in properties:
            self._add_finding(
                code=ErrorCode.SCHEMA_KEYWORD_INVALID.value,
                severity="warning",
                path=location,
                message=(
                    f"Rule references path '{rule_path}' but property '{first_part}' "
                    f"is not defined in schema"
                ),
                rule_path=rule_path,
                available_properties=list(properties.keys())[:10],
            )

    def _add_finding(
        self,
        code: str,
        severity: str,
        path: str,
        message: str,
        **details: Any,
    ) -> None:
        """
        Add a validation finding.

        Args:
            code: GLSCHEMA-* error code
            severity: Severity level (error, warning, info)
            path: JSON Pointer to problem location
            message: Human-readable message
            **details: Additional context details
        """
        finding = SchemaValidationFinding(
            code=code,
            severity=severity,
            path=path,
            message=message,
            details=details,
        )
        self._findings.append(finding)

        log_level = {
            "error": logging.ERROR,
            "warning": logging.WARNING,
            "info": logging.INFO,
        }.get(severity, logging.INFO)

        logger.log(log_level, f"[{code}] {path}: {message}")

    def _build_result(
        self,
        schema_id: Optional[str],
        version: Optional[str],
        elapsed_ms: float,
    ) -> SchemaValidationResult:
        """
        Build the final validation result from collected findings.

        Args:
            schema_id: Schema identifier
            version: Schema version
            elapsed_ms: Validation time in milliseconds

        Returns:
            SchemaValidationResult with categorized findings
        """
        errors = [f for f in self._findings if f.severity == "error"]
        warnings = [f for f in self._findings if f.severity == "warning"]
        info = [f for f in self._findings if f.severity == "info"]

        return SchemaValidationResult(
            valid=len(errors) == 0,
            schema_id=schema_id,
            version=version,
            errors=errors,
            warnings=warnings,
            info=info,
            validation_time_ms=elapsed_ms,
        )


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================


def validate_schema(
    schema_source: Union[str, Dict[str, Any]],
    strict: bool = True,
    unit_catalog: Optional["UnitCatalog"] = None,
) -> SchemaValidationResult:
    """
    Convenience function to validate a schema.

    This creates a SchemaValidator and validates the provided schema
    in a single call. For repeated validations, create a SchemaValidator
    instance directly for better performance.

    Args:
        schema_source: Schema as string (YAML/JSON) or dict
        strict: If True, apply strict validation rules
        unit_catalog: Optional unit catalog for unit validation

    Returns:
        SchemaValidationResult with all findings

    Example:
        >>> result = validate_schema({
        ...     "type": "object",
        ...     "properties": {
        ...         "value": {"type": "number", "minimum": 0}
        ...     }
        ... })
        >>> print(result.valid)
        True

        >>> # Invalid schema
        >>> result = validate_schema({
        ...     "type": "number",
        ...     "minimum": 100,
        ...     "maximum": 50  # min > max!
        ... })
        >>> print(result.valid)
        False
        >>> print(result.errors[0].code)
        GLSCHEMA-E509
    """
    validator = SchemaValidator(
        unit_catalog=unit_catalog,
        strict=strict,
    )
    return validator.validate(schema_source)


# ============================================================================
# MODULE EXPORTS
# ============================================================================


__all__ = [
    # Models
    "SchemaValidationFinding",
    "SchemaValidationResult",
    # Classes
    "SchemaValidator",
    "RegexAnalyzer",
    "RegexAnalysisResult",
    # Functions
    "validate_schema",
    "is_valid_semver",
    "compare_semver",
]
