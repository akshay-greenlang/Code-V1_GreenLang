# -*- coding: utf-8 -*-
"""
Normalizer Module for GL-FOUND-X-002.

This module provides payload normalization capabilities including:
    - Type coercion (string -> number, etc.)
    - Unit canonicalization (Wh -> kWh)
    - Key canonicalization (alias resolution)
    - Default value application

Components:
    - engine: Main normalization orchestration
    - coercions: Type coercion engine
    - canonicalizer: Key and unit canonicalization
    - keys: Advanced key canonicalization

Example:
    >>> from greenlang.schema.normalizer import NormalizationEngine, normalize
    >>> from greenlang.schema.compiler.ir import SchemaIR
    >>> from greenlang.schema.units.catalog import UnitCatalog
    >>> from greenlang.schema.models.config import ValidationOptions
    >>>
    >>> engine = NormalizationEngine(ir, UnitCatalog(), ValidationOptions())
    >>> result = engine.normalize({"Energy": "100 Wh"})
    >>> print(result.is_modified)
    True

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator
"""

from greenlang.schema.normalizer.canonicalizer import (
    CANONICAL_UNITS,
    ConversionRecord,
    CanonicalizedValue,
    KeyRename,
    UnitCanonicalizer,
    KeyCanonicalizer,
    get_canonical_unit,
    is_canonical_unit,
)

from greenlang.schema.normalizer.coercions import (
    CoercionEngine,
    CoercionRecord,
    CoercionResult,
    CoercionType,
    JSON_TYPE_ARRAY,
    JSON_TYPE_BOOLEAN,
    JSON_TYPE_INTEGER,
    JSON_TYPE_NULL,
    JSON_TYPE_NUMBER,
    JSON_TYPE_OBJECT,
    JSON_TYPE_STRING,
    can_coerce,
    get_python_type_name,
)

# Advanced key canonicalization (Task 3.3)
from greenlang.schema.normalizer.keys import (
    KeyCanonicalizer as AdvancedKeyCanonicalizer,
    KeyRename as AdvancedKeyRename,
    RenameReason,
    to_snake_case,
    to_camel_case,
    to_pascal_case,
    detect_casing,
    normalize_to_casing,
)

# Normalization Engine (Task 3.4)
from greenlang.schema.normalizer.engine import (
    NormalizationEngine,
    NormalizationMeta,
    NormalizationResult,
    normalize,
    is_normalization_idempotent,
)

__all__ = [
    # Canonicalizer constants
    "CANONICAL_UNITS",
    # Canonicalizer models
    "ConversionRecord",
    "CanonicalizedValue",
    "KeyRename",
    # Canonicalizers
    "UnitCanonicalizer",
    "KeyCanonicalizer",
    # Canonicalizer helper functions
    "get_canonical_unit",
    "is_canonical_unit",
    # Coercion type constants
    "JSON_TYPE_STRING",
    "JSON_TYPE_NUMBER",
    "JSON_TYPE_INTEGER",
    "JSON_TYPE_BOOLEAN",
    "JSON_TYPE_NULL",
    "JSON_TYPE_OBJECT",
    "JSON_TYPE_ARRAY",
    # Coercion enums
    "CoercionType",
    # Coercion models
    "CoercionRecord",
    "CoercionResult",
    # Coercion engine
    "CoercionEngine",
    # Coercion utility functions
    "can_coerce",
    "get_python_type_name",
    # Advanced key canonicalization (Task 3.3)
    "AdvancedKeyCanonicalizer",
    "AdvancedKeyRename",
    "RenameReason",
    # Casing utility functions
    "to_snake_case",
    "to_camel_case",
    "to_pascal_case",
    "detect_casing",
    "normalize_to_casing",
    # Normalization Engine (Task 3.4)
    "NormalizationEngine",
    "NormalizationMeta",
    "NormalizationResult",
    "normalize",
    "is_normalization_idempotent",
]
