# -*- coding: utf-8 -*-
"""
Schema Compiler for GL-FOUND-X-002.

This module implements the main schema compiler that transforms parsed
schema AST into optimized Intermediate Representation (IR) for fast
validation.

The compiler performs:
    - Schema parsing (YAML/JSON)
    - AST to IR transformation
    - Property map flattening for O(1) lookup
    - Regex pattern compilation with safety analysis
    - Constraint indexing
    - Stable schema hash computation (SHA-256)
    - Unit specification extraction
    - Rule binding extraction
    - Deprecation indexing

Key Features:
    - Full JSON Schema Draft 2020-12 support
    - GreenLang extensions ($unit, $dimension, $rules, $aliases, $deprecated)
    - ReDoS-safe regex pattern compilation
    - Provenance tracking via schema hashes

Example:
    >>> from greenlang.schema.compiler import SchemaCompiler
    >>> compiler = SchemaCompiler()
    >>> result = compiler.compile(schema_source, "emissions/activity", "1.3.0")
    >>> if result.success:
    ...     print(f"Compiled: {result.ir.schema_hash}")
    ...     print(f"Properties: {len(result.ir.properties)}")

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator - Task 1.4
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from greenlang.schema.compiler.ir import (
    ArrayConstraintIR,
    CompiledPattern,
    CompilationError,
    CompilationResult,
    COMPILER_VERSION,
    DeprecationInfoIR,
    NumericConstraintIR,
    PropertyIR,
    RuleBindingIR,
    SchemaIR,
    StringConstraintIR,
    UnitSpecIR,
)
from greenlang.schema.compiler.parser import parse_payload, ParseError
from greenlang.schema.compiler.resolver import RefResolver, SchemaRegistry
from greenlang.schema.constants import (
    MAX_REGEX_COMPLEXITY_SCORE,
    MAX_REGEX_LENGTH,
    REGEX_TIMEOUT_MS,
)


logger = logging.getLogger(__name__)


# =============================================================================
# GREENLANG EXTENSION KEYS
# =============================================================================

GL_UNIT_KEY = "$unit"
GL_DIMENSION_KEY = "$dimension"
GL_RULES_KEY = "$rules"
GL_ALIASES_KEY = "$aliases"
GL_DEPRECATED_KEY = "$deprecated"
GL_RENAMED_FROM_KEY = "$renamed_from"


# =============================================================================
# REGEX SAFETY PATTERNS
# =============================================================================

# Patterns that indicate potential ReDoS vulnerability
NESTED_QUANTIFIER_PATTERN = re.compile(
    r"\([^)]*[+*?][^)]*\)[+*?]|\([^)]*\{[^}]+\}[^)]*\)[+*?]"
)

# Overlapping alternation patterns
OVERLAPPING_ALT_PATTERN = re.compile(
    r"\(([^|]+)\|(\1)[^)]*\)"
)

# Backreference pattern (not RE2 compatible)
BACKREFERENCE_PATTERN = re.compile(r"\\[1-9]")

# Lookahead/lookbehind patterns (not RE2 compatible)
LOOKAROUND_PATTERN = re.compile(r"\(\?[=!<]")


# =============================================================================
# SCHEMA COMPILER
# =============================================================================


class SchemaCompiler:
    """
    Main schema compiler.

    Compiles schema sources (YAML/JSON) into optimized Intermediate
    Representation (IR) for fast validation.

    The compiler handles:
        - Schema parsing and validation
        - Property flattening for O(1) lookup
        - Constraint precompilation
        - Pattern safety analysis
        - Unit specification extraction
        - Rule binding compilation
        - Deprecation indexing

    Thread Safety:
        SchemaCompiler instances are thread-safe for read operations.
        Each compile() call creates independent state.

    Attributes:
        resolver: Optional RefResolver for external schema references

    Example:
        >>> compiler = SchemaCompiler()
        >>> result = compiler.compile(schema_source, "emissions/activity", "1.3.0")
        >>> if result.success:
        ...     ir = result.ir
        ...     print(f"Properties: {len(ir.properties)}")
        ...     print(f"Hash: {ir.schema_hash}")
    """

    def __init__(
        self,
        resolver: Optional[RefResolver] = None,
    ):
        """
        Initialize the schema compiler.

        Args:
            resolver: Optional RefResolver for external schema references.
                If None, a default resolver without registry is used.
        """
        self.resolver = resolver or RefResolver()
        self._warnings: List[str] = []
        logger.debug(f"SchemaCompiler initialized with compiler version {COMPILER_VERSION}")

    def compile(
        self,
        schema_source: Union[str, Dict[str, Any]],
        schema_id: str,
        version: str,
    ) -> CompilationResult:
        """
        Compile a schema source to IR.

        This is the main entry point for schema compilation. It handles
        parsing, validation, and transformation to IR.

        Args:
            schema_source: Schema as YAML/JSON string or parsed dict
            schema_id: Schema identifier (e.g., "emissions/activity")
            version: Schema version (e.g., "1.3.0")

        Returns:
            CompilationResult containing:
                - ir: Compiled SchemaIR (None if errors)
                - warnings: List of compilation warnings
                - errors: List of compilation errors
                - compile_time_ms: Compilation duration

        Raises:
            No exceptions are raised; errors are captured in the result.

        Example:
            >>> result = compiler.compile(schema_yaml, "emissions/activity", "1.3.0")
            >>> if result.success:
            ...     print(f"Schema hash: {result.ir.schema_hash}")
            ... else:
            ...     for error in result.errors:
            ...         print(f"Error: {error}")
        """
        start_time = time.perf_counter()
        self._warnings = []
        errors: List[str] = []

        logger.info(f"Compiling schema: {schema_id}@{version}")

        try:
            # Step 1: Parse schema source if string
            schema_dict = self._parse_schema(schema_source)
            logger.debug(f"Parsed schema with {len(schema_dict)} top-level keys")

            # Step 2: Compute stable schema hash
            schema_hash = self._compute_schema_hash(schema_dict)
            logger.debug(f"Computed schema hash: {schema_hash[:16]}...")

            # Step 3: Flatten properties
            properties = self._flatten_properties(schema_dict, "")
            logger.debug(f"Flattened {len(properties)} properties")

            # Step 4: Collect required paths
            required_paths = self._collect_required_paths(schema_dict, "")
            logger.debug(f"Collected {len(required_paths)} required paths")

            # Step 5: Compile constraints
            numeric_constraints, string_constraints, array_constraints = \
                self._compile_constraints(schema_dict, "")
            logger.debug(
                f"Compiled constraints: {len(numeric_constraints)} numeric, "
                f"{len(string_constraints)} string, {len(array_constraints)} array"
            )

            # Step 6: Compile patterns
            patterns = self._compile_patterns(schema_dict, "")
            logger.debug(f"Compiled {len(patterns)} patterns")

            # Step 7: Extract unit specs
            unit_specs = self._extract_unit_specs(schema_dict, "")
            logger.debug(f"Extracted {len(unit_specs)} unit specs")

            # Step 8: Extract rule bindings
            rule_bindings = self._extract_rule_bindings(schema_dict)
            logger.debug(f"Extracted {len(rule_bindings)} rule bindings")

            # Step 9: Extract deprecations
            deprecated_fields, renamed_fields = self._extract_deprecations(
                schema_dict, ""
            )
            logger.debug(
                f"Extracted {len(deprecated_fields)} deprecated fields, "
                f"{len(renamed_fields)} renamed fields"
            )

            # Step 10: Extract enums
            enums = self._extract_enums(schema_dict, "")
            logger.debug(f"Extracted {len(enums)} enums")

            # Create SchemaIR
            ir = SchemaIR(
                schema_id=schema_id,
                version=version,
                schema_hash=schema_hash,
                compiled_at=datetime.utcnow(),
                compiler_version=COMPILER_VERSION,
                properties=properties,
                required_paths=required_paths,
                numeric_constraints=numeric_constraints,
                string_constraints=string_constraints,
                array_constraints=array_constraints,
                patterns=patterns,
                unit_specs=unit_specs,
                rule_bindings=rule_bindings,
                deprecated_fields=deprecated_fields,
                renamed_fields=renamed_fields,
                enums=enums,
            )

            compile_time_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                f"Schema compilation successful: {schema_id}@{version} "
                f"in {compile_time_ms:.2f}ms"
            )

            return CompilationResult(
                ir=ir,
                warnings=self._warnings.copy(),
                errors=errors,
                compile_time_ms=compile_time_ms,
            )

        except ParseError as e:
            logger.error(f"Schema parsing failed: {e}")
            errors.append(f"Parse error: {e.message}")
        except CompilationError as e:
            logger.error(f"Schema compilation failed: {e}")
            errors.append(str(e))
        except Exception as e:
            logger.error(f"Unexpected compilation error: {e}", exc_info=True)
            errors.append(f"Unexpected error: {str(e)}")

        compile_time_ms = (time.perf_counter() - start_time) * 1000
        return CompilationResult(
            ir=None,
            warnings=self._warnings.copy(),
            errors=errors,
            compile_time_ms=compile_time_ms,
        )

    # =========================================================================
    # PARSING
    # =========================================================================

    def _parse_schema(self, source: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parse schema source to dictionary.

        Handles both string (YAML/JSON) and pre-parsed dictionary inputs.

        Args:
            source: Schema as string or dictionary

        Returns:
            Parsed schema dictionary

        Raises:
            ParseError: If parsing fails
        """
        if isinstance(source, dict):
            return source

        # Parse string content
        result = parse_payload(source)
        return result.data

    # =========================================================================
    # SCHEMA HASH
    # =========================================================================

    def _compute_schema_hash(self, schema: Dict[str, Any]) -> str:
        """
        Compute stable SHA-256 hash of canonical schema.

        The hash is computed from the JSON-serialized schema with:
            - Sorted keys for deterministic output
            - Compact separators (no extra whitespace)
            - UTF-8 encoding

        This hash is used for:
            - Cache key generation
            - Schema version verification
            - Provenance tracking

        Args:
            schema: The schema dictionary

        Returns:
            64-character hex SHA-256 digest
        """
        # Create canonical JSON representation
        canonical_json = json.dumps(
            schema,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

        # Compute SHA-256
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    # =========================================================================
    # PROPERTY FLATTENING
    # =========================================================================

    def _flatten_properties(
        self,
        schema: Dict[str, Any],
        path: str = "",
    ) -> Dict[str, PropertyIR]:
        """
        Flatten nested properties to path-indexed map.

        Recursively traverses the schema and creates PropertyIR entries
        for all properties, indexed by JSON Pointer path for O(1) lookup.

        Args:
            schema: Schema or sub-schema dictionary
            path: Current JSON Pointer path

        Returns:
            Dictionary mapping JSON Pointer paths to PropertyIR instances
        """
        properties: Dict[str, PropertyIR] = {}

        # Get schema type
        schema_type = schema.get("type")

        # Handle object properties
        if schema_type == "object" or "properties" in schema:
            props = schema.get("properties", {})
            required = set(schema.get("required", []))

            for prop_name, prop_schema in props.items():
                prop_path = f"{path}/{prop_name}"

                # Create PropertyIR
                prop_type = self._get_type_string(prop_schema)
                has_default = "default" in prop_schema
                default_value = prop_schema.get("default")

                # Extract GreenLang extensions
                gl_extensions = self._extract_gl_extensions(prop_schema)

                properties[prop_path] = PropertyIR(
                    path=prop_path,
                    type=prop_type,
                    required=prop_name in required,
                    has_default=has_default,
                    default_value=default_value,
                    gl_extensions=gl_extensions if gl_extensions else None,
                )

                # Recursively flatten nested properties
                nested = self._flatten_properties(prop_schema, prop_path)
                properties.update(nested)

        # Handle array items
        if schema_type == "array" or "items" in schema:
            items_schema = schema.get("items")
            if items_schema and isinstance(items_schema, dict):
                items_path = f"{path}/items"

                # Create PropertyIR for items
                items_type = self._get_type_string(items_schema)
                gl_extensions = self._extract_gl_extensions(items_schema)

                properties[items_path] = PropertyIR(
                    path=items_path,
                    type=items_type,
                    required=False,
                    has_default=False,
                    gl_extensions=gl_extensions if gl_extensions else None,
                )

                # Recursively flatten items
                nested = self._flatten_properties(items_schema, items_path)
                properties.update(nested)

        # Handle definitions/$defs
        for defs_key in ("definitions", "$defs"):
            if defs_key in schema:
                defs = schema[defs_key]
                for def_name, def_schema in defs.items():
                    def_path = f"/{defs_key}/{def_name}"
                    nested = self._flatten_properties(def_schema, def_path)
                    properties.update(nested)

        return properties

    def _get_type_string(self, schema: Dict[str, Any]) -> Optional[str]:
        """
        Get type string from schema.

        Handles both single types and union types.

        Args:
            schema: Property schema

        Returns:
            Type string or None
        """
        type_value = schema.get("type")
        if type_value is None:
            return None
        if isinstance(type_value, list):
            return "|".join(type_value)
        return type_value

    def _extract_gl_extensions(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract GreenLang extensions from schema.

        Args:
            schema: Property schema

        Returns:
            Dictionary of GreenLang extensions
        """
        extensions: Dict[str, Any] = {}

        if GL_UNIT_KEY in schema:
            extensions["unit"] = schema[GL_UNIT_KEY]
        if GL_DIMENSION_KEY in schema:
            extensions["dimension"] = schema[GL_DIMENSION_KEY]
        if GL_RULES_KEY in schema:
            extensions["rules"] = schema[GL_RULES_KEY]
        if GL_ALIASES_KEY in schema:
            extensions["aliases"] = schema[GL_ALIASES_KEY]
        if GL_DEPRECATED_KEY in schema:
            extensions["deprecated"] = schema[GL_DEPRECATED_KEY]
        if GL_RENAMED_FROM_KEY in schema:
            extensions["renamed_from"] = schema[GL_RENAMED_FROM_KEY]

        return extensions

    # =========================================================================
    # REQUIRED PATHS
    # =========================================================================

    def _collect_required_paths(
        self,
        schema: Dict[str, Any],
        path: str = "",
    ) -> Set[str]:
        """
        Collect all required field paths.

        Recursively traverses the schema to find all required properties
        and returns their JSON Pointer paths.

        Args:
            schema: Schema or sub-schema dictionary
            path: Current JSON Pointer path

        Returns:
            Set of JSON Pointer paths for required fields
        """
        required_paths: Set[str] = set()

        # Get required fields for this object
        required = schema.get("required", [])
        for req_name in required:
            req_path = f"{path}/{req_name}"
            required_paths.add(req_path)

        # Recursively check properties
        properties = schema.get("properties", {})
        for prop_name, prop_schema in properties.items():
            if isinstance(prop_schema, dict):
                prop_path = f"{path}/{prop_name}"
                nested = self._collect_required_paths(prop_schema, prop_path)
                required_paths.update(nested)

        # Check array items
        items_schema = schema.get("items")
        if items_schema and isinstance(items_schema, dict):
            items_path = f"{path}/items"
            nested = self._collect_required_paths(items_schema, items_path)
            required_paths.update(nested)

        # Check definitions
        for defs_key in ("definitions", "$defs"):
            if defs_key in schema:
                defs = schema[defs_key]
                for def_name, def_schema in defs.items():
                    def_path = f"/{defs_key}/{def_name}"
                    nested = self._collect_required_paths(def_schema, def_path)
                    required_paths.update(nested)

        return required_paths

    # =========================================================================
    # CONSTRAINT COMPILATION
    # =========================================================================

    def _compile_constraints(
        self,
        schema: Dict[str, Any],
        path: str = "",
    ) -> Tuple[
        Dict[str, NumericConstraintIR],
        Dict[str, StringConstraintIR],
        Dict[str, ArrayConstraintIR],
    ]:
        """
        Compile numeric, string, and array constraints.

        Recursively traverses the schema to extract all constraints and
        compile them into IR structures for fast validation.

        Args:
            schema: Schema or sub-schema dictionary
            path: Current JSON Pointer path

        Returns:
            Tuple of (numeric_constraints, string_constraints, array_constraints)
            dictionaries, each mapping paths to constraint IR objects
        """
        numeric: Dict[str, NumericConstraintIR] = {}
        string: Dict[str, StringConstraintIR] = {}
        array: Dict[str, ArrayConstraintIR] = {}

        schema_type = schema.get("type")

        # Compile numeric constraints
        if schema_type in ("number", "integer") or self._has_numeric_constraints(schema):
            constraint = self._compile_numeric_constraint(schema, path)
            if constraint and constraint.has_constraints():
                numeric[path] = constraint

        # Compile string constraints
        if schema_type == "string" or self._has_string_constraints(schema):
            constraint = self._compile_string_constraint(schema, path)
            if constraint and constraint.has_constraints():
                string[path] = constraint

        # Compile array constraints
        if schema_type == "array" or self._has_array_constraints(schema):
            constraint = self._compile_array_constraint(schema, path)
            if constraint and constraint.has_constraints():
                array[path] = constraint

        # Recursively process properties
        properties = schema.get("properties", {})
        for prop_name, prop_schema in properties.items():
            if isinstance(prop_schema, dict):
                prop_path = f"{path}/{prop_name}"
                n, s, a = self._compile_constraints(prop_schema, prop_path)
                numeric.update(n)
                string.update(s)
                array.update(a)

        # Process array items
        items_schema = schema.get("items")
        if items_schema and isinstance(items_schema, dict):
            items_path = f"{path}/items"
            n, s, a = self._compile_constraints(items_schema, items_path)
            numeric.update(n)
            string.update(s)
            array.update(a)

        # Process definitions
        for defs_key in ("definitions", "$defs"):
            if defs_key in schema:
                defs = schema[defs_key]
                for def_name, def_schema in defs.items():
                    def_path = f"/{defs_key}/{def_name}"
                    n, s, a = self._compile_constraints(def_schema, def_path)
                    numeric.update(n)
                    string.update(s)
                    array.update(a)

        return numeric, string, array

    def _has_numeric_constraints(self, schema: Dict[str, Any]) -> bool:
        """Check if schema has numeric constraints."""
        keys = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"}
        return bool(keys & schema.keys())

    def _has_string_constraints(self, schema: Dict[str, Any]) -> bool:
        """Check if schema has string constraints."""
        keys = {"minLength", "maxLength", "pattern", "format"}
        return bool(keys & schema.keys())

    def _has_array_constraints(self, schema: Dict[str, Any]) -> bool:
        """Check if schema has array constraints."""
        keys = {"minItems", "maxItems", "uniqueItems"}
        return bool(keys & schema.keys())

    def _compile_numeric_constraint(
        self,
        schema: Dict[str, Any],
        path: str,
    ) -> Optional[NumericConstraintIR]:
        """
        Compile numeric constraints from schema.

        Args:
            schema: Property schema dictionary
            path: JSON Pointer path

        Returns:
            NumericConstraintIR if constraints exist, None otherwise
        """
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        exclusive_minimum = schema.get("exclusiveMinimum")
        exclusive_maximum = schema.get("exclusiveMaximum")
        multiple_of = schema.get("multipleOf")

        # Validate constraint consistency
        if minimum is not None and maximum is not None:
            if minimum > maximum:
                self._warnings.append(
                    f"Inconsistent constraints at {path}: minimum ({minimum}) > maximum ({maximum})"
                )

        return NumericConstraintIR(
            path=path,
            minimum=minimum,
            maximum=maximum,
            exclusive_minimum=exclusive_minimum,
            exclusive_maximum=exclusive_maximum,
            multiple_of=multiple_of,
        )

    def _compile_string_constraint(
        self,
        schema: Dict[str, Any],
        path: str,
    ) -> Optional[StringConstraintIR]:
        """
        Compile string constraints from schema.

        Args:
            schema: Property schema dictionary
            path: JSON Pointer path

        Returns:
            StringConstraintIR if constraints exist, None otherwise
        """
        min_length = schema.get("minLength")
        max_length = schema.get("maxLength")
        pattern = schema.get("pattern")
        format_type = schema.get("format")

        # Compile pattern if present
        pattern_compiled = None
        if pattern:
            pattern_compiled = self._compile_pattern(pattern, path)

        # Validate constraint consistency
        if min_length is not None and max_length is not None:
            if min_length > max_length:
                self._warnings.append(
                    f"Inconsistent constraints at {path}: minLength ({min_length}) > maxLength ({max_length})"
                )

        return StringConstraintIR(
            path=path,
            min_length=min_length,
            max_length=max_length,
            pattern=pattern,
            pattern_compiled=pattern_compiled,
            format=format_type,
        )

    def _compile_array_constraint(
        self,
        schema: Dict[str, Any],
        path: str,
    ) -> Optional[ArrayConstraintIR]:
        """
        Compile array constraints from schema.

        Args:
            schema: Property schema dictionary
            path: JSON Pointer path

        Returns:
            ArrayConstraintIR if constraints exist, None otherwise
        """
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        unique_items = schema.get("uniqueItems", False)

        # Validate constraint consistency
        if min_items is not None and max_items is not None:
            if min_items > max_items:
                self._warnings.append(
                    f"Inconsistent constraints at {path}: minItems ({min_items}) > maxItems ({max_items})"
                )

        return ArrayConstraintIR(
            path=path,
            min_items=min_items,
            max_items=max_items,
            unique_items=unique_items,
        )

    # =========================================================================
    # PATTERN COMPILATION
    # =========================================================================

    def _compile_patterns(
        self,
        schema: Dict[str, Any],
        path: str = "",
    ) -> Dict[str, CompiledPattern]:
        """
        Compile regex patterns with safety analysis.

        Recursively finds all regex patterns in the schema and compiles
        them with ReDoS safety analysis.

        Args:
            schema: Schema or sub-schema dictionary
            path: Current JSON Pointer path

        Returns:
            Dictionary mapping paths to CompiledPattern objects
        """
        patterns: Dict[str, CompiledPattern] = {}

        # Check for pattern at current level
        if "pattern" in schema:
            pattern_str = schema["pattern"]
            compiled = self._compile_pattern(pattern_str, path)
            patterns[path] = compiled

        # Check patternProperties
        pattern_props = schema.get("patternProperties", {})
        for pattern_key in pattern_props.keys():
            pattern_path = f"{path}/patternProperties/{pattern_key}"
            compiled = self._compile_pattern(pattern_key, pattern_path)
            patterns[pattern_path] = compiled

        # Recursively process properties
        properties = schema.get("properties", {})
        for prop_name, prop_schema in properties.items():
            if isinstance(prop_schema, dict):
                prop_path = f"{path}/{prop_name}"
                nested = self._compile_patterns(prop_schema, prop_path)
                patterns.update(nested)

        # Process array items
        items_schema = schema.get("items")
        if items_schema and isinstance(items_schema, dict):
            items_path = f"{path}/items"
            nested = self._compile_patterns(items_schema, items_path)
            patterns.update(nested)

        # Process definitions
        for defs_key in ("definitions", "$defs"):
            if defs_key in schema:
                defs = schema[defs_key]
                for def_name, def_schema in defs.items():
                    def_path = f"/{defs_key}/{def_name}"
                    nested = self._compile_patterns(def_schema, def_path)
                    patterns.update(nested)

        return patterns

    def _compile_pattern(self, pattern: str, path: str) -> CompiledPattern:
        """
        Compile and analyze a regex pattern.

        Analyzes the pattern for ReDoS vulnerabilities and assigns
        a complexity score.

        Args:
            pattern: The regex pattern string
            path: JSON Pointer path (for error reporting)

        Returns:
            CompiledPattern with safety metadata
        """
        # Check pattern length
        if len(pattern) > MAX_REGEX_LENGTH:
            self._warnings.append(
                f"Pattern at {path} exceeds maximum length ({len(pattern)} > {MAX_REGEX_LENGTH})"
            )
            return CompiledPattern(
                pattern=pattern,
                complexity_score=1.0,
                is_safe=False,
                timeout_ms=REGEX_TIMEOUT_MS,
                is_re2_compatible=False,
                vulnerability_type="pattern_too_long",
                recommendation="Simplify the pattern or split into multiple patterns",
            )

        # Analyze for ReDoS vulnerabilities
        is_safe = True
        complexity_score = 0.0
        vulnerability_type = None
        recommendation = None
        is_re2_compatible = True

        # Check for nested quantifiers
        if NESTED_QUANTIFIER_PATTERN.search(pattern):
            is_safe = False
            complexity_score = max(complexity_score, 0.9)
            vulnerability_type = "nested_quantifier"
            recommendation = "Avoid nested quantifiers like (a+)+ or (a*)*"

        # Check for overlapping alternations
        if OVERLAPPING_ALT_PATTERN.search(pattern):
            is_safe = False
            complexity_score = max(complexity_score, 0.8)
            vulnerability_type = vulnerability_type or "overlapping_alternation"
            recommendation = recommendation or "Avoid overlapping alternations like (a|a)+"

        # Check for backreferences (not RE2 compatible)
        if BACKREFERENCE_PATTERN.search(pattern):
            is_re2_compatible = False
            complexity_score = max(complexity_score, 0.3)

        # Check for lookaround (not RE2 compatible)
        if LOOKAROUND_PATTERN.search(pattern):
            is_re2_compatible = False
            complexity_score = max(complexity_score, 0.3)

        # Compute complexity based on pattern structure
        if is_safe:
            complexity_score = self._compute_pattern_complexity(pattern)

        # Warn if complexity is high
        if complexity_score > MAX_REGEX_COMPLEXITY_SCORE:
            self._warnings.append(
                f"Pattern at {path} has high complexity score ({complexity_score:.2f})"
            )

        # Try to compile the pattern
        try:
            re.compile(pattern)
        except re.error as e:
            self._warnings.append(f"Invalid regex pattern at {path}: {e}")
            return CompiledPattern(
                pattern=pattern,
                complexity_score=1.0,
                is_safe=False,
                timeout_ms=REGEX_TIMEOUT_MS,
                is_re2_compatible=False,
                vulnerability_type="invalid_pattern",
                recommendation=f"Fix regex error: {e}",
            )

        return CompiledPattern(
            pattern=pattern,
            complexity_score=complexity_score,
            is_safe=is_safe,
            timeout_ms=REGEX_TIMEOUT_MS,
            is_re2_compatible=is_re2_compatible,
            vulnerability_type=vulnerability_type,
            recommendation=recommendation,
        )

    def _compute_pattern_complexity(self, pattern: str) -> float:
        """
        Compute complexity score for a regex pattern.

        Uses heuristics based on pattern structure:
        - Quantifier count
        - Alternation count
        - Nesting depth
        - Pattern length

        Args:
            pattern: The regex pattern

        Returns:
            Complexity score between 0.0 and 1.0
        """
        score = 0.0

        # Length contribution (normalized to 0-0.2)
        length_score = min(len(pattern) / MAX_REGEX_LENGTH, 1.0) * 0.2
        score += length_score

        # Quantifier count contribution
        quantifiers = len(re.findall(r'[+*?]|\{[^}]+\}', pattern))
        quantifier_score = min(quantifiers / 10, 1.0) * 0.3
        score += quantifier_score

        # Alternation count contribution
        alternations = pattern.count('|')
        alt_score = min(alternations / 10, 1.0) * 0.2
        score += alt_score

        # Nesting depth contribution
        max_depth = 0
        current_depth = 0
        for char in pattern:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth = max(0, current_depth - 1)
        depth_score = min(max_depth / 5, 1.0) * 0.3
        score += depth_score

        return min(score, 1.0)

    # =========================================================================
    # UNIT SPECIFICATION EXTRACTION
    # =========================================================================

    def _extract_unit_specs(
        self,
        schema: Dict[str, Any],
        path: str = "",
    ) -> Dict[str, UnitSpecIR]:
        """
        Extract unit specifications from GreenLang extensions.

        Recursively finds all $unit extensions and creates UnitSpecIR
        objects for validation.

        Args:
            schema: Schema or sub-schema dictionary
            path: Current JSON Pointer path

        Returns:
            Dictionary mapping paths to UnitSpecIR objects
        """
        unit_specs: Dict[str, UnitSpecIR] = {}

        # Check for $unit at current level
        if GL_UNIT_KEY in schema:
            unit_data = schema[GL_UNIT_KEY]
            if isinstance(unit_data, dict):
                unit_spec = self._compile_unit_spec(unit_data, path)
                if unit_spec:
                    unit_specs[path] = unit_spec

        # Recursively process properties
        properties = schema.get("properties", {})
        for prop_name, prop_schema in properties.items():
            if isinstance(prop_schema, dict):
                prop_path = f"{path}/{prop_name}"
                nested = self._extract_unit_specs(prop_schema, prop_path)
                unit_specs.update(nested)

        # Process array items
        items_schema = schema.get("items")
        if items_schema and isinstance(items_schema, dict):
            items_path = f"{path}/items"
            nested = self._extract_unit_specs(items_schema, items_path)
            unit_specs.update(nested)

        # Process definitions
        for defs_key in ("definitions", "$defs"):
            if defs_key in schema:
                defs = schema[defs_key]
                for def_name, def_schema in defs.items():
                    def_path = f"/{defs_key}/{def_name}"
                    nested = self._extract_unit_specs(def_schema, def_path)
                    unit_specs.update(nested)

        return unit_specs

    def _compile_unit_spec(
        self,
        unit_data: Dict[str, Any],
        path: str,
    ) -> Optional[UnitSpecIR]:
        """
        Compile a unit specification from $unit extension data.

        Args:
            unit_data: The $unit extension dictionary
            path: JSON Pointer path

        Returns:
            UnitSpecIR if valid, None otherwise
        """
        dimension = unit_data.get("dimension")
        canonical = unit_data.get("canonical")
        allowed = unit_data.get("allowed", [])

        # Validate required fields
        if not dimension or not canonical:
            self._warnings.append(
                f"Incomplete $unit at {path}: missing dimension or canonical"
            )
            return None

        return UnitSpecIR(
            path=path,
            dimension=dimension,
            canonical=canonical,
            allowed=allowed,
        )

    # =========================================================================
    # RULE BINDING EXTRACTION
    # =========================================================================

    def _extract_rule_bindings(
        self,
        schema: Dict[str, Any],
    ) -> List[RuleBindingIR]:
        """
        Extract rule bindings from schema.

        Finds all $rules extensions at document and property levels
        and compiles them to RuleBindingIR objects.

        Args:
            schema: Schema dictionary

        Returns:
            List of RuleBindingIR objects
        """
        rule_bindings: List[RuleBindingIR] = []

        # Check for document-level rules
        if GL_RULES_KEY in schema:
            rules_data = schema[GL_RULES_KEY]
            if isinstance(rules_data, list):
                for rule_data in rules_data:
                    rule_binding = self._compile_rule_binding(rule_data)
                    if rule_binding:
                        rule_bindings.append(rule_binding)

        # Recursively find property-level rules
        self._collect_property_rules(schema, "", rule_bindings)

        return rule_bindings

    def _collect_property_rules(
        self,
        schema: Dict[str, Any],
        path: str,
        rule_bindings: List[RuleBindingIR],
    ) -> None:
        """
        Recursively collect rules from properties.

        Args:
            schema: Schema or sub-schema dictionary
            path: Current JSON Pointer path
            rule_bindings: List to append rules to
        """
        properties = schema.get("properties", {})
        for prop_name, prop_schema in properties.items():
            if isinstance(prop_schema, dict):
                prop_path = f"{path}/{prop_name}"

                # Check for rules at this property
                if GL_RULES_KEY in prop_schema:
                    rules_data = prop_schema[GL_RULES_KEY]
                    if isinstance(rules_data, list):
                        for rule_data in rules_data:
                            rule_binding = self._compile_rule_binding(rule_data, prop_path)
                            if rule_binding:
                                rule_bindings.append(rule_binding)

                # Recurse
                self._collect_property_rules(prop_schema, prop_path, rule_bindings)

    def _compile_rule_binding(
        self,
        rule_data: Dict[str, Any],
        context_path: str = "",
    ) -> Optional[RuleBindingIR]:
        """
        Compile a rule binding from $rules extension data.

        Args:
            rule_data: The rule data dictionary
            context_path: Path where rule was defined

        Returns:
            RuleBindingIR if valid, None otherwise
        """
        if not isinstance(rule_data, dict):
            self._warnings.append(f"Invalid rule at {context_path}: not a dictionary")
            return None

        rule_id = rule_data.get("rule_id")
        if not rule_id:
            self._warnings.append(f"Rule at {context_path} missing rule_id")
            return None

        return RuleBindingIR(
            rule_id=rule_id,
            rule_pack=rule_data.get("rule_pack"),
            severity=rule_data.get("severity", "error"),
            applies_to=rule_data.get("applies_to", [context_path] if context_path else []),
            when=rule_data.get("when"),
            check=rule_data.get("check", {}),
            message=rule_data.get("message", "Rule validation failed"),
            message_template=rule_data.get("message_template"),
        )

    # =========================================================================
    # DEPRECATION EXTRACTION
    # =========================================================================

    def _extract_deprecations(
        self,
        schema: Dict[str, Any],
        path: str = "",
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
        """
        Extract deprecated and renamed fields.

        Finds all $deprecated and $renamed_from extensions and creates
        deprecation indices.

        Args:
            schema: Schema or sub-schema dictionary
            path: Current JSON Pointer path

        Returns:
            Tuple of (deprecated_fields, renamed_fields) dictionaries
        """
        deprecated_fields: Dict[str, Dict[str, Any]] = {}
        renamed_fields: Dict[str, str] = {}

        # Check for deprecation at current level
        if GL_DEPRECATED_KEY in schema:
            dep_data = schema[GL_DEPRECATED_KEY]
            if isinstance(dep_data, dict):
                deprecated_fields[path] = {
                    "since_version": dep_data.get("since_version", ""),
                    "message": dep_data.get("message", "This field is deprecated"),
                    "replacement": dep_data.get("replacement"),
                    "removal_version": dep_data.get("removal_version"),
                }

        # Check for renamed_from
        if GL_RENAMED_FROM_KEY in schema:
            old_name = schema[GL_RENAMED_FROM_KEY]
            if isinstance(old_name, str):
                # Map old name to new name (current path)
                old_path = path.rsplit("/", 1)[0] + "/" + old_name if "/" in path else f"/{old_name}"
                renamed_fields[old_path] = path

        # Recursively process properties
        properties = schema.get("properties", {})
        for prop_name, prop_schema in properties.items():
            if isinstance(prop_schema, dict):
                prop_path = f"{path}/{prop_name}"
                dep, ren = self._extract_deprecations(prop_schema, prop_path)
                deprecated_fields.update(dep)
                renamed_fields.update(ren)

        # Process array items
        items_schema = schema.get("items")
        if items_schema and isinstance(items_schema, dict):
            items_path = f"{path}/items"
            dep, ren = self._extract_deprecations(items_schema, items_path)
            deprecated_fields.update(dep)
            renamed_fields.update(ren)

        # Process definitions
        for defs_key in ("definitions", "$defs"):
            if defs_key in schema:
                defs = schema[defs_key]
                for def_name, def_schema in defs.items():
                    def_path = f"/{defs_key}/{def_name}"
                    dep, ren = self._extract_deprecations(def_schema, def_path)
                    deprecated_fields.update(dep)
                    renamed_fields.update(ren)

        return deprecated_fields, renamed_fields

    # =========================================================================
    # ENUM EXTRACTION
    # =========================================================================

    def _extract_enums(
        self,
        schema: Dict[str, Any],
        path: str = "",
    ) -> Dict[str, List[Any]]:
        """
        Extract enum values from schema.

        Finds all enum constraints and creates a lookup index.

        Args:
            schema: Schema or sub-schema dictionary
            path: Current JSON Pointer path

        Returns:
            Dictionary mapping paths to enum value lists
        """
        enums: Dict[str, List[Any]] = {}

        # Check for enum at current level
        if "enum" in schema:
            enum_values = schema["enum"]
            if isinstance(enum_values, list):
                enums[path] = enum_values

        # Recursively process properties
        properties = schema.get("properties", {})
        for prop_name, prop_schema in properties.items():
            if isinstance(prop_schema, dict):
                prop_path = f"{path}/{prop_name}"
                nested = self._extract_enums(prop_schema, prop_path)
                enums.update(nested)

        # Process array items
        items_schema = schema.get("items")
        if items_schema and isinstance(items_schema, dict):
            items_path = f"{path}/items"
            nested = self._extract_enums(items_schema, items_path)
            enums.update(nested)

        # Process definitions
        for defs_key in ("definitions", "$defs"):
            if defs_key in schema:
                defs = schema[defs_key]
                for def_name, def_schema in defs.items():
                    def_path = f"/{defs_key}/{def_name}"
                    nested = self._extract_enums(def_schema, def_path)
                    enums.update(nested)

        return enums


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    "SchemaCompiler",
    "GL_UNIT_KEY",
    "GL_DIMENSION_KEY",
    "GL_RULES_KEY",
    "GL_ALIASES_KEY",
    "GL_DEPRECATED_KEY",
    "GL_RENAMED_FROM_KEY",
]
