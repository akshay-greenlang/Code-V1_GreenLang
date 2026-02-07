# -*- coding: utf-8 -*-
"""
Schema Compiler Module for GL-FOUND-X-002
=========================================

This module provides schema parsing, AST generation, and compilation to
Intermediate Representation (IR) for efficient validation.

Components:
    - parser: Safe YAML/JSON parsing with size limits
    - ast: Abstract Syntax Tree definitions for schemas
    - ir: Intermediate Representation for compiled schemas
    - compiler: Main compilation orchestration
    - resolver: $ref resolution with cycle detection
    - regex_analyzer: Regex safety analysis for ReDoS prevention

Example:
    >>> from greenlang.schema.compiler import (
    ...     SchemaDocument, ObjectTypeNode, build_ast, parse_type_node
    ... )
    >>> schema_dict = {"type": "object", "properties": {"name": {"type": "string"}}}
    >>> doc = build_ast(schema_dict, "test/schema", "1.0.0")
    >>> print(doc.schema_id)
    test/schema

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator
"""

from greenlang.schema.compiler.ast import (
    # Constants
    JSON_SCHEMA_DRAFT_2020_12,
    JSON_SCHEMA_TYPES,
    # Enums
    RuleSeverity,
    # Extension Models
    UnitSpec,
    DeprecationInfo,
    RuleBinding,
    GreenLangExtensions,
    # AST Nodes
    SchemaNode,
    TypeNode,
    ObjectTypeNode,
    ArrayTypeNode,
    StringTypeNode,
    NumericTypeNode,
    BooleanTypeNode,
    NullTypeNode,
    RefNode,
    CompositionNode,
    EnumTypeNode,
    SchemaDocument,
    # Helper Functions
    create_node_id,
    create_unique_node_id,
    parse_type_node,
    build_ast,
    validate_ast,
)

# Resolver components (Task 1.3 - Complete)
from greenlang.schema.compiler.resolver import (
    # Exceptions
    CircularRefError,
    RefResolutionError,
    MaxExpansionsExceededError,
    # Data models
    SchemaSource,
    ParsedRef,
    RefType,
    # Protocols
    SchemaRegistry,
    # Main classes
    RefResolver,
    LocalFileRegistry,
    # Functions
    parse_json_pointer,
    navigate_json_pointer,
    parse_ref,
    resolve_all_refs,
    validate_ref_format,
)

# Parser components (Task 1.1 - Complete)
from greenlang.schema.compiler.parser import (
    ParseResult,
    ParseError,
    SafeYAMLLoader,
    parse_payload,
    detect_format,
    validate_payload_size,
    is_valid_json,
    is_valid_yaml,
)

# Regex Analyzer components (Task 1.5 - Complete)
from greenlang.schema.compiler.regex_analyzer import (
    # Main class
    RegexAnalyzer,
    # Result model
    RegexAnalysisResult,
    # Enums
    VulnerabilityType,
    # Module-level functions
    analyze_regex_safety,
    is_safe_pattern,
    is_re2_compatible,
    sanitize_pattern,
    compile_with_timeout,
    # Constants
    DANGEROUS_PATTERNS,
    SAFE_PATTERNS,
)

# Schema Validator components (Task 1.6 - Complete)
from greenlang.schema.compiler.schema_validator import (
    # Result models
    SchemaValidationFinding,
    SchemaValidationResult,
    # Main class
    SchemaValidator,
    # Convenience function
    validate_schema,
    # Helper functions
    is_valid_semver,
    compare_semver,
)

# IR components (Task 1.4 - Complete)
from greenlang.schema.compiler.ir import (
    SchemaIR,
    PropertyIR,
    CompiledPattern,
)

# Compiler components (Task 1.4 - Complete)
from greenlang.schema.compiler.compiler import (
    SchemaCompiler,
    CompilationResult,
)

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
    # Resolver - Exceptions (Task 1.3)
    "CircularRefError",
    "RefResolutionError",
    "MaxExpansionsExceededError",
    # Resolver - Data models
    "SchemaSource",
    "ParsedRef",
    "RefType",
    # Resolver - Protocols
    "SchemaRegistry",
    # Resolver - Main classes
    "RefResolver",
    "LocalFileRegistry",
    # Resolver - Functions
    "parse_json_pointer",
    "navigate_json_pointer",
    "parse_ref",
    "resolve_all_refs",
    "validate_ref_format",
    # Parser - Classes (Task 1.1)
    "ParseResult",
    "ParseError",
    "SafeYAMLLoader",
    # Parser - Functions (Task 1.1)
    "parse_payload",
    "detect_format",
    "validate_payload_size",
    "is_valid_json",
    "is_valid_yaml",
    # Regex Analyzer - Classes (Task 1.5)
    "RegexAnalyzer",
    "RegexAnalysisResult",
    "VulnerabilityType",
    # Regex Analyzer - Functions (Task 1.5)
    "analyze_regex_safety",
    "is_safe_pattern",
    "is_re2_compatible",
    "sanitize_pattern",
    "compile_with_timeout",
    # Regex Analyzer - Constants (Task 1.5)
    "DANGEROUS_PATTERNS",
    "SAFE_PATTERNS",
    # Schema Validator - Classes (Task 1.6)
    "SchemaValidationFinding",
    "SchemaValidationResult",
    "SchemaValidator",
    # Schema Validator - Functions (Task 1.6)
    "validate_schema",
    "is_valid_semver",
    "compare_semver",
    # IR exports (Task 1.4)
    "SchemaIR",
    "PropertyIR",
    "CompiledPattern",
    # Compiler exports (Task 1.4)
    "SchemaCompiler",
    "CompilationResult",
]
