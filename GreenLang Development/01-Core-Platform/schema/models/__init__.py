# -*- coding: utf-8 -*-
"""
GreenLang Schema Models
=======================

Pydantic v2 models for GL-FOUND-X-002: GreenLang Schema Compiler & Validator.

This module provides data models for:
- Schema references and identifiers
- Validation configuration and options
- Validation findings and error reporting
- Validation reports and summaries
- JSON Patch operations and fix suggestions

All models follow GreenLang conventions:
- Pydantic v2 with comprehensive validation
- JSON-serializable for API responses
- Type-safe with complete type hints
- Documented with docstrings

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator
"""

from greenlang.schema.models.schema_ref import SchemaRef
from greenlang.schema.models.config import (
    ValidationProfile,
    CoercionPolicy,
    UnknownFieldPolicy,
    PatchLevel,
    ValidationOptions,
)
from greenlang.schema.models.finding import (
    Severity,
    FindingHint,
    Finding,
)
from greenlang.schema.models.report import (
    ValidationSummary,
    TimingInfo,
    ValidationReport,
    BatchSummary,
    ItemResult,
    BatchValidationReport,
)
from greenlang.schema.models.patch import (
    JSONPatchOp,
    PatchSafety,
    FixSuggestion,
)

__all__ = [
    # Schema reference
    "SchemaRef",
    # Configuration
    "ValidationProfile",
    "CoercionPolicy",
    "UnknownFieldPolicy",
    "PatchLevel",
    "ValidationOptions",
    # Findings
    "Severity",
    "FindingHint",
    "Finding",
    # Reports
    "ValidationSummary",
    "TimingInfo",
    "ValidationReport",
    "BatchSummary",
    "ItemResult",
    "BatchValidationReport",
    # Patches
    "JSONPatchOp",
    "PatchSafety",
    "FixSuggestion",
]
