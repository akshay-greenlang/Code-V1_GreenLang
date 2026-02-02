# -*- coding: utf-8 -*-
"""
Suggestions Module for GL-FOUND-X-002.

This module provides fix suggestion capabilities including:
    - JSON Patch generation (RFC 6902)
    - JSON Pointer utilities (RFC 6901)
    - Patch safety classification
    - Fix heuristics for common errors

Components:
    - patches: JSON Patch operation generation and application
    - safety: Patch safety classification
    - heuristics: Fix heuristics for common errors
    - engine: Main fix suggestion orchestration

Example:
    >>> from greenlang.schema.suggestions import PatchGenerator, apply_patch
    >>> generator = PatchGenerator()
    >>> patch = generator.generate_add("/energy", {"value": 100, "unit": "kWh"})
    >>>
    >>> doc = {"name": "test"}
    >>> result = apply_patch(doc, [patch])
    >>> print(result)
    {'name': 'test', 'energy': {'value': 100, 'unit': 'kWh'}}

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator
"""

from greenlang.schema.suggestions.patches import (
    # Exceptions
    JSONPointerError,
    PatchApplicationError,
    # Enums
    PatchOp,
    # Models
    JSONPatchOperation,
    PatchSequence,
    # Generator
    PatchGenerator,
    # JSON Pointer utilities
    parse_json_pointer,
    build_json_pointer,
    get_value_at_pointer,
    set_value_at_pointer,
    insert_value_at_pointer,
    remove_value_at_pointer,
    pointer_parent,
    pointer_last_segment,
    escape_json_pointer_token,
    unescape_json_pointer_token,
    # Patch application
    apply_patch,
    apply_patch_sequence,
    # Patch validation
    validate_patch,
    validate_patch_syntax,
    # Convenience functions
    create_add_patch,
    create_remove_patch,
    create_replace_patch,
    create_move_patch,
    # Legacy compatibility
    JSONPatchOp,
)

# Safety classification
from greenlang.schema.suggestions.safety import (
    # Constants
    SAFE_OPERATIONS,
    NEEDS_REVIEW_OPERATIONS,
    UNSAFE_OPERATIONS,
    HIGH_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    LARGE_UNIT_FACTOR_THRESHOLD,
    # Enums
    PatchSafety,
    # Note: PatchOp is already imported from patches
    # Models
    PatchContext,
    SafetyClassification,
    # Main class
    PatchSafetyClassifier,
    # Convenience functions
    is_safe_patch,
    classify_patches,
    filter_safe_patches,
)

# Heuristics for fix suggestions
from greenlang.schema.suggestions.heuristics import (
    FixHeuristics,
    MAX_TYPO_DISTANCE,
    MAX_ENUM_DISTANCE,
    RANGE_TYPO_THRESHOLD,
    BOOLEAN_TRUE_VALUES,
    BOOLEAN_FALSE_VALUES,
)

# Engine for orchestrating fix suggestion generation
from greenlang.schema.suggestions.engine import (
    FIXABLE_CODES,
    SuggestionEngineResult,
    FixSuggestionEngine,
    generate_suggestions,
    apply_suggestions,
    get_fixable_codes,
)

__all__ = [
    # Exceptions
    "JSONPointerError",
    "PatchApplicationError",
    # Enums
    "PatchOp",
    # Models
    "JSONPatchOperation",
    "PatchSequence",
    # Generator
    "PatchGenerator",
    # JSON Pointer utilities
    "parse_json_pointer",
    "build_json_pointer",
    "get_value_at_pointer",
    "set_value_at_pointer",
    "insert_value_at_pointer",
    "remove_value_at_pointer",
    "pointer_parent",
    "pointer_last_segment",
    "escape_json_pointer_token",
    "unescape_json_pointer_token",
    # Patch application
    "apply_patch",
    "apply_patch_sequence",
    # Patch validation
    "validate_patch",
    "validate_patch_syntax",
    # Convenience functions
    "create_add_patch",
    "create_remove_patch",
    "create_replace_patch",
    "create_move_patch",
    # Legacy compatibility
    "JSONPatchOp",
    # Safety classification
    "SAFE_OPERATIONS",
    "NEEDS_REVIEW_OPERATIONS",
    "UNSAFE_OPERATIONS",
    "HIGH_CONFIDENCE_THRESHOLD",
    "MEDIUM_CONFIDENCE_THRESHOLD",
    "LOW_CONFIDENCE_THRESHOLD",
    "LARGE_UNIT_FACTOR_THRESHOLD",
    "PatchSafety",
    "PatchContext",
    "SafetyClassification",
    "PatchSafetyClassifier",
    "is_safe_patch",
    "classify_patches",
    "filter_safe_patches",
    # Heuristics
    "FixHeuristics",
    "MAX_TYPO_DISTANCE",
    "MAX_ENUM_DISTANCE",
    "RANGE_TYPO_THRESHOLD",
    "BOOLEAN_TRUE_VALUES",
    "BOOLEAN_FALSE_VALUES",
    # Engine
    "FIXABLE_CODES",
    "SuggestionEngineResult",
    "FixSuggestionEngine",
    "generate_suggestions",
    "apply_suggestions",
    "get_fixable_codes",
]
