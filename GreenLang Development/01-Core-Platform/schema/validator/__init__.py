# -*- coding: utf-8 -*-
"""
Validator Module for GL-FOUND-X-002 (GreenLang Schema Compiler & Validator).

This module provides the core validation engine for validating payloads
against compiled schema IRs. It supports:

- Structural validation (types, required fields, additional properties)
- Constraint validation (ranges, patterns, enums, lengths)
- Unit validation (presence, compatibility, conversion)
- Cross-field rule validation
- Linting (non-blocking warnings)

Components:
    - core: Main validator orchestration
    - structural: Structural validation (types, required fields)
    - constraints: Constraint validation (ranges, patterns)
    - units: Unit validation and conversion
    - rules: Cross-field rule evaluation
    - linter: Non-blocking lint checks

Validation Pipeline:
    1. Parse payload (with safety limits)
    2. Resolve schema (with IR caching)
    3. Structural validation
    4. Constraint validation
    5. Unit validation
    6. Rule validation
    7. Linting (non-blocking)

Example:
    >>> from greenlang.schema.validator import SchemaValidator, validate
    >>> validator = SchemaValidator(registry, unit_catalog)
    >>> result = validator.validate(payload, schema_ref)
    >>> print(result.valid)

    # Or use the convenience function
    >>> result = validate(payload, "gl://schemas/activity@1.0.0")

Author: GreenLang Framework Team
Version: 1.0.0
GL-FOUND-X-002: Schema Compiler & Validator
"""

from greenlang.schema.validator.core import (
    SchemaValidator,
    validate,
)

# Structural validator - Task 2.1 COMPLETE
from greenlang.schema.validator.structural import (
    StructuralValidator,
    validate_structure,
    PYTHON_TO_JSON_TYPE,
    TYPE_COMPATIBILITY,
)

# Constraint validator - Task 2.2 COMPLETE
from greenlang.schema.validator.constraints import (
    ConstraintValidator,
    FORMAT_VALIDATORS,
)

# Linter - Task 2.6 COMPLETE
from greenlang.schema.validator.linter import (
    SchemaLinter,
    lint_payload,
    # Casing detection helpers
    is_snake_case,
    is_camel_case,
    is_pascal_case,
    is_kebab_case,
    is_screaming_snake_case,
    to_snake_case,
    to_camel_case,
    to_pascal_case,
)

# Rule validator - Task 2.4 COMPLETE
from greenlang.schema.validator.rules import (
    RuleValidator,
    ExpressionEvaluator,
    Rule,
    RuleExpression,
    # Error codes
    ERROR_CODE_RULE_VIOLATION,
    ERROR_CODE_CONDITIONAL_REQUIRED,
    ERROR_CODE_CONSISTENCY_ERROR,
    # Operator constants
    COMPARISON_OPERATORS,
    EXTENDED_OPERATORS,
    NULL_CHECK_OPERATORS,
    ARITHMETIC_OPERATORS,
    # Factory functions
    create_conditional_required_rule,
    create_sum_consistency_rule,
    create_range_rule,
)

# Unit validator - Task 2.3 COMPLETE
from greenlang.schema.validator.units import (
    UnitValidator,
    NormalizedUnit,
    create_unit_finding,
)

__all__ = [
    # Core validator
    "SchemaValidator",
    "validate",
    # Structural validator (Task 2.1)
    "StructuralValidator",
    "validate_structure",
    "PYTHON_TO_JSON_TYPE",
    "TYPE_COMPATIBILITY",
    # Constraint validator (Task 2.2)
    "ConstraintValidator",
    "FORMAT_VALIDATORS",
    # Linter (Task 2.6)
    "SchemaLinter",
    "lint_payload",
    "is_snake_case",
    "is_camel_case",
    "is_pascal_case",
    "is_kebab_case",
    "is_screaming_snake_case",
    "to_snake_case",
    "to_camel_case",
    "to_pascal_case",
    # Rule validator (Task 2.4)
    "RuleValidator",
    "ExpressionEvaluator",
    "Rule",
    "RuleExpression",
    "ERROR_CODE_RULE_VIOLATION",
    "ERROR_CODE_CONDITIONAL_REQUIRED",
    "ERROR_CODE_CONSISTENCY_ERROR",
    "COMPARISON_OPERATORS",
    "EXTENDED_OPERATORS",
    "NULL_CHECK_OPERATORS",
    "ARITHMETIC_OPERATORS",
    "create_conditional_required_rule",
    "create_sum_consistency_rule",
    "create_range_rule",
    # Unit validator (Task 2.3)
    "UnitValidator",
    "NormalizedUnit",
    "create_unit_finding",
]
