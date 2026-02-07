"""
GL-FOUND-X-002: GreenLang Schema Compiler & Validator

This package provides the schema compilation and validation infrastructure
for the GreenLang framework. It supports:

- JSON Schema Draft 2020-12 with GreenLang extensions
- Safe YAML/JSON parsing with size limits
- Type coercion and unit normalization
- Cross-field rule validation
- Fix suggestions with JSON Patch generation

Key Components:
    - compiler: Schema parsing, AST, and IR compilation
    - validator: Structural, constraint, unit, and rule validation
    - normalizer: Type coercion and unit canonicalization
    - suggestions: Fix suggestion engine with JSON Patch
    - registry: Git-backed schema registry
    - cli: Command-line interface
    - api: FastAPI HTTP service

Example:
    >>> from greenlang.schema import validate, SchemaRef
    >>> result = validate(
    ...     payload={"energy": 100, "unit": "kWh"},
    ...     schema=SchemaRef(schema_id="emissions/activity", version="1.3.0")
    ... )
    >>> print(result.valid)
    True

For CLI usage:
    $ greenlang schema validate data.yaml --schema emissions/activity@1.3.0

Agent ID: GL-FOUND-X-002
Agent Name: GreenLang Schema Compiler & Validator
"""

from greenlang.schema.version import (
    __version__,
    __agent_id__,
    __agent_name__,
    __compiler_version__,
    get_version_info,
)

from greenlang.schema.errors import ErrorCode, get_error_by_code

from greenlang.schema.constants import (
    MAX_PAYLOAD_BYTES,
    MAX_SCHEMA_BYTES,
    MAX_OBJECT_DEPTH,
    MAX_ARRAY_ITEMS,
    MAX_TOTAL_NODES,
    MAX_REF_EXPANSIONS,
    MAX_FINDINGS,
)

# Core models
from greenlang.schema.models import (
    SchemaRef,
    ValidationProfile,
    CoercionPolicy,
    UnknownFieldPolicy,
    PatchLevel,
    ValidationOptions,
    Severity,
    FindingHint,
    Finding,
    ValidationSummary,
    TimingInfo,
    ValidationReport,
    BatchSummary,
    ItemResult,
    BatchValidationReport,
    JSONPatchOp,
    PatchSafety,
    FixSuggestion,
)

# SDK functions - user-friendly API
from greenlang.schema.sdk import (
    validate,
    validate_batch,
    compile_schema,
    CompiledSchema,
    apply_fixes,
    safe_fixes,
    review_fixes,
    errors_only,
    warnings_only,
    findings_by_path,
    findings_by_code,
    parse_schema_ref,
    schema_ref,
)

# Prometheus metrics
from greenlang.schema.metrics import (
    PROMETHEUS_AVAILABLE,
    record_validation,
    record_compilation,
    record_error,
    record_warning,
    record_fix_applied,
    record_cache_hit,
    record_cache_miss,
    record_batch,
    record_payload_bytes,
    update_active_validations,
    update_registered_schemas,
)

# Service setup facade
from greenlang.schema.setup import (
    SchemaService,
    configure_schema_service,
    get_schema_service,
)

__all__ = [
    # Version info
    "__version__",
    "__agent_id__",
    "__agent_name__",
    "__compiler_version__",
    "get_version_info",
    # Error codes
    "ErrorCode",
    "get_error_by_code",
    # Constants
    "MAX_PAYLOAD_BYTES",
    "MAX_SCHEMA_BYTES",
    "MAX_OBJECT_DEPTH",
    "MAX_ARRAY_ITEMS",
    "MAX_TOTAL_NODES",
    "MAX_REF_EXPANSIONS",
    "MAX_FINDINGS",
    # Core models
    "SchemaRef",
    "ValidationProfile",
    "CoercionPolicy",
    "UnknownFieldPolicy",
    "PatchLevel",
    "ValidationOptions",
    "Severity",
    "FindingHint",
    "Finding",
    "ValidationSummary",
    "TimingInfo",
    "ValidationReport",
    "BatchSummary",
    "ItemResult",
    "BatchValidationReport",
    "JSONPatchOp",
    "PatchSafety",
    "FixSuggestion",
    # SDK functions - user-friendly API
    "validate",
    "validate_batch",
    "compile_schema",
    "CompiledSchema",
    "apply_fixes",
    "safe_fixes",
    "review_fixes",
    "errors_only",
    "warnings_only",
    "findings_by_path",
    "findings_by_code",
    "parse_schema_ref",
    "schema_ref",
    # Prometheus metrics
    "PROMETHEUS_AVAILABLE",
    "record_validation",
    "record_compilation",
    "record_error",
    "record_warning",
    "record_fix_applied",
    "record_cache_hit",
    "record_cache_miss",
    "record_batch",
    "record_payload_bytes",
    "update_active_validations",
    "update_registered_schemas",
    # Service setup facade
    "SchemaService",
    "configure_schema_service",
    "get_schema_service",
]
