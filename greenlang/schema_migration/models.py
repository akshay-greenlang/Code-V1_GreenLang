# -*- coding: utf-8 -*-
"""
Schema Migration Agent Service Data Models - AGENT-DATA-017

Pydantic v2 data models for the Schema Migration Agent SDK. Attempts to
re-export Layer 1 enumerations and models from the Data Quality Profiler
(QualityDimension, RuleType) and Schema Compiler (SchemaCompilerEngine),
and defines all SDK models for schema registry, versioning, change
detection, compatibility analysis, migration planning, execution,
rollback, field mapping, drift monitoring, audit trails, and pipeline
orchestration.

Re-exported Layer 1 sources (best-effort, with fallback stubs):
    - greenlang.data_quality_profiler.models: QualityDimension, RuleType
    - greenlang.schema_compiler: SchemaCompilerEngine

New enumerations (14):
    - SchemaType, SchemaStatus, ChangeType, ChangeSeverity,
      CompatibilityLevel, MigrationPlanStatus, ExecutionStatus,
      RollbackType, RollbackStatus, DriftType, DriftSeverity,
      MappingType, EffortLevel, TransformationType

New SDK models (16):
    - SchemaDefinition, SchemaVersion, SchemaChange,
      CompatibilityResult, MigrationStep, MigrationPlan,
      MigrationExecution, RollbackRecord, FieldMapping, DriftEvent,
      AuditEntry, SchemaGroup, MigrationReport, SchemaStatistics,
      VersionComparison, PipelineResult

Request models (8):
    - RegisterSchemaRequest, UpdateSchemaRequest, CreateVersionRequest,
      DetectChangesRequest, CheckCompatibilityRequest, CreatePlanRequest,
      ExecuteMigrationRequest, RunPipelineRequest

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-017 Schema Migration Agent (GL-DATA-X-020)
Status: Production Ready
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Layer 1 Re-exports (best-effort with stubs on ImportError)
# ---------------------------------------------------------------------------

try:
    from greenlang.data_quality_profiler.models import (  # type: ignore[import]
        QualityDimension as L1QualityDimension,
        RuleType as L1RuleType,
    )

    QualityDimension = L1QualityDimension
    RuleType = L1RuleType
    _DQ_AVAILABLE = True
except ImportError:  # pragma: no cover
    _DQ_AVAILABLE = False

    class QualityDimension(str, Enum):  # type: ignore[no-redef]
        """Stub re-export when data_quality_profiler is unavailable."""

        COMPLETENESS = "completeness"
        VALIDITY = "validity"
        CONSISTENCY = "consistency"
        TIMELINESS = "timeliness"
        UNIQUENESS = "uniqueness"
        ACCURACY = "accuracy"

    class RuleType(str, Enum):  # type: ignore[no-redef]
        """Stub re-export when data_quality_profiler is unavailable."""

        COMPLETENESS = "completeness"
        RANGE = "range"
        FORMAT = "format"
        UNIQUENESS = "uniqueness"
        CUSTOM = "custom"
        FRESHNESS = "freshness"


try:
    from greenlang.schema_compiler import (  # type: ignore[import]
        SchemaCompilerEngine as L1SchemaCompilerEngine,
    )

    SchemaCompilerEngine = L1SchemaCompilerEngine
    _SC_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SC_AVAILABLE = False
    SchemaCompilerEngine = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Service version string.
VERSION: str = "1.0.0"

#: Maximum number of schema definitions allowed per namespace.
MAX_SCHEMAS_PER_NAMESPACE: int = 10_000

#: Maximum number of fields allowed in a single schema definition.
MAX_FIELDS_PER_SCHEMA: int = 1_000

#: Maximum number of steps in a single migration plan.
MAX_MIGRATION_STEPS: int = 500

#: Schema serialization formats supported by the migration agent.
SUPPORTED_SCHEMA_TYPES: tuple = ("json_schema", "avro", "protobuf")

#: Default compatibility rules per schema type.
#: Each entry maps a schema type to its default CompatibilityLevel.
DEFAULT_COMPATIBILITY_RULES: Dict[str, str] = {
    "json_schema": "backward",
    "avro": "full",
    "protobuf": "backward",
}

#: Severity ordering from least to most severe (for comparisons).
CHANGE_SEVERITY_ORDER: tuple = ("cosmetic", "non_breaking", "breaking")

#: Default batch size for migration execution.
DEFAULT_MIGRATION_BATCH_SIZE: int = 1_000

#: Minimum confidence score for auto-accepted field mappings.
AUTO_ACCEPT_CONFIDENCE_THRESHOLD: float = 0.90

#: Maximum number of drift events stored per schema version.
MAX_DRIFT_EVENTS_PER_VERSION: int = 10_000

#: Semver pattern used for version string validation.
_SEMVER_RE: re.Pattern = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)

#: Pattern for valid namespace strings (alphanumeric, hyphens, dots).
_NAMESPACE_RE: re.Pattern = re.compile(r"^[a-zA-Z0-9]([a-zA-Z0-9\-\.]*[a-zA-Z0-9])?$")


# =============================================================================
# Enumerations (14)
# =============================================================================


class SchemaType(str, Enum):
    """Schema serialization format for a registered schema.

    Determines which parser and validator is used during schema
    compilation, change detection, and compatibility checking.
    """

    JSON_SCHEMA = "json_schema"
    AVRO = "avro"
    PROTOBUF = "protobuf"


class SchemaStatus(str, Enum):
    """Lifecycle status of a schema definition in the registry.

    Controls whether consumers can use a schema and whether
    new versions can be registered against it.

    DRAFT: Schema is under development and not production-ready.
    ACTIVE: Schema is in production use.
    DEPRECATED: Schema is superseded; existing consumers should migrate.
    ARCHIVED: Schema is retired and no longer usable.
    """

    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ChangeType(str, Enum):
    """Type of structural change detected between two schema versions.

    Used to classify each diff entry produced by the change detection
    engine, enabling compatibility analysis and migration planning.
    """

    ADDED = "added"
    REMOVED = "removed"
    RENAMED = "renamed"
    RETYPED = "retyped"
    REORDERED = "reordered"
    CONSTRAINT_CHANGED = "constraint_changed"
    ENUM_CHANGED = "enum_changed"
    DEFAULT_CHANGED = "default_changed"


class ChangeSeverity(str, Enum):
    """Severity classification for a schema change.

    BREAKING: Change is incompatible with existing consumers
        (e.g., removing a required field).
    NON_BREAKING: Change is backward-compatible
        (e.g., adding an optional field with a default).
    COSMETIC: Change has no impact on data or contract
        (e.g., updating a description).
    """

    BREAKING = "breaking"
    NON_BREAKING = "non_breaking"
    COSMETIC = "cosmetic"


class CompatibilityLevel(str, Enum):
    """Result of a compatibility check between two schema versions.

    BACKWARD: New schema can read data written with the old schema.
    FORWARD: Old schema can read data written with the new schema.
    FULL: Both backward and forward compatibility hold.
    BREAKING: Neither direction of compatibility is maintained.
    NONE: Compatibility has not been assessed.
    """

    BACKWARD = "backward"
    FORWARD = "forward"
    FULL = "full"
    BREAKING = "breaking"
    NONE = "none"


class MigrationPlanStatus(str, Enum):
    """Lifecycle status of a migration plan.

    DRAFT: Plan is being assembled; not yet validated.
    VALIDATED: Plan has passed dry-run validation.
    APPROVED: Plan has been approved for execution.
    EXECUTING: Plan is currently being executed.
    COMPLETED: Plan execution finished successfully.
    FAILED: Plan execution failed; see error_details.
    ROLLED_BACK: Plan was rolled back after a failure.
    """

    DRAFT = "draft"
    VALIDATED = "validated"
    APPROVED = "approved"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class ExecutionStatus(str, Enum):
    """Status of a single migration execution run.

    PENDING: Execution has been queued but not started.
    RUNNING: Execution is actively processing records.
    COMPLETED: Execution finished without errors.
    FAILED: Execution stopped due to an unrecoverable error.
    ROLLED_BACK: Execution was reversed via rollback procedure.
    TIMED_OUT: Execution exceeded the configured timeout.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    TIMED_OUT = "timed_out"


class RollbackType(str, Enum):
    """Scope of a rollback operation.

    FULL: Revert all steps executed in the migration run.
    PARTIAL: Revert only steps beyond a specified checkpoint.
    CHECKPOINT: Revert to the most recent successful checkpoint.
    """

    FULL = "full"
    PARTIAL = "partial"
    CHECKPOINT = "checkpoint"


class RollbackStatus(str, Enum):
    """Status of a rollback operation.

    PENDING: Rollback has been requested but not started.
    RUNNING: Rollback is actively reverting changes.
    COMPLETED: Rollback finished successfully.
    FAILED: Rollback encountered an unrecoverable error.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class DriftType(str, Enum):
    """Category of schema drift detected in a live dataset.

    MISSING_FIELD: A field declared in the schema is absent from the data.
    EXTRA_FIELD: The data contains a field not declared in the schema.
    TYPE_MISMATCH: A field's observed type differs from the declared type.
    CONSTRAINT_VIOLATION: A field value violates a schema constraint.
    ENUM_VIOLATION: A field value is not in the declared enum set.
    """

    MISSING_FIELD = "missing_field"
    EXTRA_FIELD = "extra_field"
    TYPE_MISMATCH = "type_mismatch"
    CONSTRAINT_VIOLATION = "constraint_violation"
    ENUM_VIOLATION = "enum_violation"


class DriftSeverity(str, Enum):
    """Severity classification for a schema drift event.

    LOW: Drift is cosmetic or affects only optional metadata.
    MEDIUM: Drift may cause downstream warnings or soft failures.
    HIGH: Drift is likely to cause data pipeline errors.
    CRITICAL: Drift causes immediate data loss or compliance risk.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MappingType(str, Enum):
    """Method by which a source field was mapped to a target field.

    EXACT: Field names and types match exactly; no transformation needed.
    ALIAS: Field was matched via a known alias or synonym.
    COMPUTED: Target field is derived from one or more source fields.
    MANUAL: Mapping was created or confirmed manually by a user.
    """

    EXACT = "exact"
    ALIAS = "alias"
    COMPUTED = "computed"
    MANUAL = "manual"


class EffortLevel(str, Enum):
    """Estimated implementation effort for a migration plan.

    LOW: Automated migration; minimal or no human intervention required.
    MEDIUM: Semi-automated; some human review or scripting needed.
    HIGH: Complex migration; significant engineering effort required.
    CRITICAL: High-risk migration; requires architecture review and sign-off.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TransformationType(str, Enum):
    """Type of data transformation applied in a migration step.

    RENAME_FIELD: Renames a field without changing its value or type.
    CAST_TYPE: Casts a field value from one type to another.
    SET_DEFAULT: Populates null or missing values with a default.
    COMPUTE_FIELD: Derives a new field value from an expression.
    SPLIT_FIELD: Splits one field into multiple target fields.
    MERGE_FIELDS: Combines multiple source fields into one target field.
    REMOVE_FIELD: Removes a field entirely from the target schema.
    ADD_FIELD: Adds a new field that was not present in the source schema.
    """

    RENAME_FIELD = "rename_field"
    CAST_TYPE = "cast_type"
    SET_DEFAULT = "set_default"
    COMPUTE_FIELD = "compute_field"
    SPLIT_FIELD = "split_field"
    MERGE_FIELDS = "merge_fields"
    REMOVE_FIELD = "remove_field"
    ADD_FIELD = "add_field"


# =============================================================================
# SDK Data Models (16)
# =============================================================================


class SchemaDefinition(BaseModel):
    """A registered schema definition in the Schema Migration Agent registry.

    Represents the top-level schema entity that groups all versions under
    a namespace and name key. Each schema has exactly one active version
    at a time, with historical versions retained for audit and rollback.

    Attributes:
        id: Unique schema identifier (UUID v4).
        namespace: Dot/hyphen-separated namespace owning this schema.
        name: Human-readable schema name (max 255 characters).
        schema_type: Serialization format (json_schema, avro, protobuf).
        owner: Team or service responsible for this schema.
        tags: Arbitrary key-value labels for discovery and filtering.
        status: Current lifecycle status of the schema.
        description: Human-readable description of the schema's purpose.
        metadata: Additional unstructured metadata key-value pairs.
        definition_json: The initial schema definition as a JSON-serializable dict.
        created_at: UTC timestamp when the schema was first registered.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique schema identifier (UUID v4)",
    )
    namespace: str = Field(
        ...,
        description="Dot/hyphen-separated namespace owning this schema",
    )
    name: str = Field(
        ...,
        max_length=255,
        description="Human-readable schema name (max 255 characters)",
    )
    schema_type: SchemaType = Field(
        default=SchemaType.JSON_SCHEMA,
        description="Serialization format (json_schema, avro, protobuf)",
    )
    owner: str = Field(
        default="",
        description="Team or service responsible for this schema",
    )
    tags: Dict[str, str] = Field(
        default_factory=dict,
        description="Arbitrary key-value labels for discovery and filtering",
    )
    status: SchemaStatus = Field(
        default=SchemaStatus.DRAFT,
        description="Current lifecycle status of the schema",
    )
    description: str = Field(
        default="",
        description="Human-readable description of the schema's purpose",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional unstructured metadata key-value pairs",
    )
    definition_json: Dict[str, Any] = Field(
        default_factory=dict,
        description="The initial schema definition as a JSON-serializable dict",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the schema was first registered",
    )

    model_config = {"extra": "forbid"}

    @field_validator("namespace")
    @classmethod
    def validate_namespace(cls, v: str) -> str:
        """Validate namespace is non-empty and uses only alphanumeric, hyphens, and dots."""
        if not v or not v.strip():
            raise ValueError("namespace must be non-empty")
        if not _NAMESPACE_RE.match(v):
            raise ValueError(
                "namespace must contain only alphanumeric characters, hyphens, "
                "and dots, and must start and end with an alphanumeric character"
            )
        return v

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty and within length limit."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        if len(v) > 255:
            raise ValueError("name must not exceed 255 characters")
        return v


class SchemaVersion(BaseModel):
    """A versioned snapshot of a schema definition.

    Each time a schema is updated, a new SchemaVersion is created.
    Versions are immutable once created. The version string follows
    semantic versioning (SemVer) conventions.

    Attributes:
        id: Unique version identifier (UUID v4).
        schema_id: Reference to the parent SchemaDefinition id.
        version: SemVer string (e.g., "1.2.3").
        definition_json: The schema definition at this version.
        changelog: Human-readable notes describing what changed.
        is_deprecated: Whether this version has been deprecated.
        deprecated_at: UTC timestamp when the version was deprecated.
        sunset_date: UTC date after which the version will be archived.
        created_by: Actor (user or service) that created this version.
        created_at: UTC timestamp when the version was registered.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique version identifier (UUID v4)",
    )
    schema_id: str = Field(
        ...,
        description="Reference to the parent SchemaDefinition id",
    )
    version: str = Field(
        ...,
        description="SemVer string (e.g., '1.2.3')",
    )
    definition_json: Dict[str, Any] = Field(
        default_factory=dict,
        description="The schema definition at this version",
    )
    changelog: str = Field(
        default="",
        description="Human-readable notes describing what changed",
    )
    is_deprecated: bool = Field(
        default=False,
        description="Whether this version has been deprecated",
    )
    deprecated_at: Optional[datetime] = Field(
        None,
        description="UTC timestamp when the version was deprecated",
    )
    sunset_date: Optional[datetime] = Field(
        None,
        description="UTC date after which the version will be archived",
    )
    created_by: str = Field(
        default="",
        description="Actor (user or service) that created this version",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the version was registered",
    )

    model_config = {"extra": "forbid"}

    @field_validator("schema_id")
    @classmethod
    def validate_schema_id(cls, v: str) -> str:
        """Validate schema_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("schema_id must be non-empty")
        return v

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate version follows SemVer format."""
        if not v or not v.strip():
            raise ValueError("version must be non-empty")
        if not _SEMVER_RE.match(v):
            raise ValueError(
                f"version '{v}' does not follow SemVer format (MAJOR.MINOR.PATCH)"
            )
        return v


class SchemaChange(BaseModel):
    """A single structural change detected between two schema versions.

    Produced by the change detection engine when comparing a source
    and target version. Each change describes one field-level or
    schema-level modification with its severity and impact.

    Attributes:
        id: Unique change identifier (UUID v4).
        source_version_id: ID of the version being changed from.
        target_version_id: ID of the version being changed to.
        change_type: Category of structural change.
        field_path: Dot-notation path to the affected field (e.g., "user.address.city").
        old_value: JSON-serializable representation of the previous value.
        new_value: JSON-serializable representation of the new value.
        severity: Breaking, non-breaking, or cosmetic classification.
        description: Human-readable explanation of the change.
        detected_at: UTC timestamp when the change was detected.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique change identifier (UUID v4)",
    )
    source_version_id: str = Field(
        ...,
        description="ID of the version being changed from",
    )
    target_version_id: str = Field(
        ...,
        description="ID of the version being changed to",
    )
    change_type: ChangeType = Field(
        ...,
        description="Category of structural change",
    )
    field_path: str = Field(
        default="",
        description="Dot-notation path to the affected field",
    )
    old_value: Optional[Any] = Field(
        None,
        description="JSON-serializable representation of the previous value",
    )
    new_value: Optional[Any] = Field(
        None,
        description="JSON-serializable representation of the new value",
    )
    severity: ChangeSeverity = Field(
        default=ChangeSeverity.NON_BREAKING,
        description="Breaking, non-breaking, or cosmetic classification",
    )
    description: str = Field(
        default="",
        description="Human-readable explanation of the change",
    )
    detected_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the change was detected",
    )

    model_config = {"extra": "forbid"}

    @field_validator("source_version_id")
    @classmethod
    def validate_source_version_id(cls, v: str) -> str:
        """Validate source_version_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("source_version_id must be non-empty")
        return v

    @field_validator("target_version_id")
    @classmethod
    def validate_target_version_id(cls, v: str) -> str:
        """Validate target_version_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("target_version_id must be non-empty")
        return v


class CompatibilityResult(BaseModel):
    """Result of a compatibility analysis between two schema versions.

    Produced by the compatibility checker to determine whether a schema
    change is safe to deploy without breaking existing producers or
    consumers. Includes a list of specific compatibility issues and
    recommended remediation steps.

    Attributes:
        id: Unique result identifier (UUID v4).
        source_version_id: ID of the source (older) schema version.
        target_version_id: ID of the target (newer) schema version.
        compatibility_level: Overall compatibility classification.
        issues: List of specific compatibility problem descriptions.
        recommendations: List of suggested remediation actions.
        checked_by: Actor (user or service) that requested the check.
        checked_at: UTC timestamp when the check was performed.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique result identifier (UUID v4)",
    )
    source_version_id: str = Field(
        ...,
        description="ID of the source (older) schema version",
    )
    target_version_id: str = Field(
        ...,
        description="ID of the target (newer) schema version",
    )
    compatibility_level: CompatibilityLevel = Field(
        default=CompatibilityLevel.NONE,
        description="Overall compatibility classification",
    )
    issues: List[str] = Field(
        default_factory=list,
        description="List of specific compatibility problem descriptions",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="List of suggested remediation actions",
    )
    checked_by: str = Field(
        default="",
        description="Actor (user or service) that requested the check",
    )
    checked_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the check was performed",
    )

    model_config = {"extra": "forbid"}

    @field_validator("source_version_id")
    @classmethod
    def validate_source_version_id(cls, v: str) -> str:
        """Validate source_version_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("source_version_id must be non-empty")
        return v

    @field_validator("target_version_id")
    @classmethod
    def validate_target_version_id(cls, v: str) -> str:
        """Validate target_version_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("target_version_id must be non-empty")
        return v


class MigrationStep(BaseModel):
    """A single atomic operation within a migration plan.

    Each step describes one data transformation that must be applied
    in sequence to migrate records from the source schema to the
    target schema. Steps are numbered from 1 for ordered execution.

    Attributes:
        step_number: 1-based ordinal position of this step in the plan.
        operation: Transformation type to apply.
        source_field: Dot-notation path to the source field (if applicable).
        target_field: Dot-notation path to the target field (if applicable).
        parameters: Operation-specific parameters (e.g., default value, cast type).
        reversible: Whether this step can be undone during rollback.
        description: Human-readable description of what this step does.
    """

    step_number: int = Field(
        ...,
        ge=1,
        description="1-based ordinal position of this step in the plan",
    )
    operation: TransformationType = Field(
        ...,
        description="Transformation type to apply",
    )
    source_field: Optional[str] = Field(
        None,
        description="Dot-notation path to the source field (if applicable)",
    )
    target_field: Optional[str] = Field(
        None,
        description="Dot-notation path to the target field (if applicable)",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Operation-specific parameters (e.g., default value, cast type)",
    )
    reversible: bool = Field(
        default=True,
        description="Whether this step can be undone during rollback",
    )
    description: str = Field(
        default="",
        description="Human-readable description of what this step does",
    )

    model_config = {"extra": "forbid"}


class MigrationPlan(BaseModel):
    """A complete migration plan for transforming data from one schema to another.

    A MigrationPlan is constructed by the migration planner after change
    detection and compatibility analysis. It contains an ordered list of
    MigrationStep objects that, when executed, will transform all records
    conforming to the source schema into records conforming to the target schema.

    Attributes:
        id: Unique plan identifier (UUID v4).
        source_schema_id: ID of the source schema definition.
        target_schema_id: ID of the target schema definition.
        source_version: SemVer string of the source schema version.
        target_version: SemVer string of the target schema version.
        steps: Ordered list of migration steps to execute.
        status: Current lifecycle status of the plan.
        estimated_effort: Estimated implementation effort level.
        estimated_records: Estimated number of records to migrate.
        dry_run_result: Results from the most recent dry-run validation.
        created_by: Actor (user or service) that created the plan.
        created_at: UTC timestamp when the plan was created.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique plan identifier (UUID v4)",
    )
    source_schema_id: str = Field(
        ...,
        description="ID of the source schema definition",
    )
    target_schema_id: str = Field(
        ...,
        description="ID of the target schema definition",
    )
    source_version: str = Field(
        ...,
        description="SemVer string of the source schema version",
    )
    target_version: str = Field(
        ...,
        description="SemVer string of the target schema version",
    )
    steps: List[MigrationStep] = Field(
        default_factory=list,
        description="Ordered list of migration steps to execute",
    )
    status: MigrationPlanStatus = Field(
        default=MigrationPlanStatus.DRAFT,
        description="Current lifecycle status of the plan",
    )
    estimated_effort: EffortLevel = Field(
        default=EffortLevel.MEDIUM,
        description="Estimated implementation effort level",
    )
    estimated_records: int = Field(
        default=0,
        ge=0,
        description="Estimated number of records to migrate",
    )
    dry_run_result: Dict[str, Any] = Field(
        default_factory=dict,
        description="Results from the most recent dry-run validation",
    )
    created_by: str = Field(
        default="",
        description="Actor (user or service) that created the plan",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the plan was created",
    )

    model_config = {"extra": "forbid"}

    @field_validator("source_schema_id")
    @classmethod
    def validate_source_schema_id(cls, v: str) -> str:
        """Validate source_schema_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("source_schema_id must be non-empty")
        return v

    @field_validator("target_schema_id")
    @classmethod
    def validate_target_schema_id(cls, v: str) -> str:
        """Validate target_schema_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("target_schema_id must be non-empty")
        return v

    @field_validator("source_version")
    @classmethod
    def validate_source_version(cls, v: str) -> str:
        """Validate source_version follows SemVer format."""
        if not v or not v.strip():
            raise ValueError("source_version must be non-empty")
        if not _SEMVER_RE.match(v):
            raise ValueError(
                f"source_version '{v}' does not follow SemVer format (MAJOR.MINOR.PATCH)"
            )
        return v

    @field_validator("target_version")
    @classmethod
    def validate_target_version(cls, v: str) -> str:
        """Validate target_version follows SemVer format."""
        if not v or not v.strip():
            raise ValueError("target_version must be non-empty")
        if not _SEMVER_RE.match(v):
            raise ValueError(
                f"target_version '{v}' does not follow SemVer format (MAJOR.MINOR.PATCH)"
            )
        return v


class MigrationExecution(BaseModel):
    """A runtime execution record for a migration plan.

    Tracks the real-time state of a migration run including current
    step progress, record processing counts, checkpoint data for
    safe resumption after failures, and error details.

    Attributes:
        id: Unique execution identifier (UUID v4).
        plan_id: Reference to the MigrationPlan being executed.
        started_at: UTC timestamp when execution began.
        completed_at: UTC timestamp when execution ended (or None).
        status: Current execution status.
        current_step: Step number currently being processed.
        total_steps: Total number of steps in the plan.
        records_processed: Count of records successfully transformed.
        records_failed: Count of records that failed transformation.
        records_skipped: Count of records skipped (e.g., already migrated).
        checkpoint_data: Serializable state for safe resumption.
        error_details: Error message and stack trace if status is FAILED.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique execution identifier (UUID v4)",
    )
    plan_id: str = Field(
        ...,
        description="Reference to the MigrationPlan being executed",
    )
    started_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when execution began",
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="UTC timestamp when execution ended (or None if still running)",
    )
    status: ExecutionStatus = Field(
        default=ExecutionStatus.PENDING,
        description="Current execution status",
    )
    current_step: int = Field(
        default=0,
        ge=0,
        description="Step number currently being processed",
    )
    total_steps: int = Field(
        default=0,
        ge=0,
        description="Total number of steps in the plan",
    )
    records_processed: int = Field(
        default=0,
        ge=0,
        description="Count of records successfully transformed",
    )
    records_failed: int = Field(
        default=0,
        ge=0,
        description="Count of records that failed transformation",
    )
    records_skipped: int = Field(
        default=0,
        ge=0,
        description="Count of records skipped (e.g., already migrated)",
    )
    checkpoint_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Serializable state for safe resumption after failure",
    )
    error_details: Optional[str] = Field(
        None,
        description="Error message and stack trace if status is FAILED",
    )

    model_config = {"extra": "forbid"}

    @field_validator("plan_id")
    @classmethod
    def validate_plan_id(cls, v: str) -> str:
        """Validate plan_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("plan_id must be non-empty")
        return v


class RollbackRecord(BaseModel):
    """An audit record for a rollback operation.

    Created whenever a migration execution is rolled back, either
    automatically after a failure or manually by an operator.
    Tracks how many records were reverted and the rollback outcome.

    Attributes:
        id: Unique rollback record identifier (UUID v4).
        execution_id: Reference to the MigrationExecution being rolled back.
        reason: Human-readable reason why the rollback was initiated.
        rollback_type: Scope of the rollback operation.
        rolled_back_to_step: Step number to which execution was reverted.
        records_reverted: Count of records successfully reverted.
        started_at: UTC timestamp when the rollback began.
        completed_at: UTC timestamp when the rollback ended (or None).
        status: Current rollback status.
        error_details: Error message if the rollback itself failed.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique rollback record identifier (UUID v4)",
    )
    execution_id: str = Field(
        ...,
        description="Reference to the MigrationExecution being rolled back",
    )
    reason: str = Field(
        default="",
        description="Human-readable reason why the rollback was initiated",
    )
    rollback_type: RollbackType = Field(
        default=RollbackType.FULL,
        description="Scope of the rollback operation",
    )
    rolled_back_to_step: int = Field(
        default=0,
        ge=0,
        description="Step number to which execution was reverted",
    )
    records_reverted: int = Field(
        default=0,
        ge=0,
        description="Count of records successfully reverted",
    )
    started_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the rollback began",
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="UTC timestamp when the rollback ended (or None if still running)",
    )
    status: RollbackStatus = Field(
        default=RollbackStatus.PENDING,
        description="Current rollback status",
    )
    error_details: Optional[str] = Field(
        None,
        description="Error message if the rollback itself failed",
    )

    model_config = {"extra": "forbid"}

    @field_validator("execution_id")
    @classmethod
    def validate_execution_id(cls, v: str) -> str:
        """Validate execution_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("execution_id must be non-empty")
        return v


class FieldMapping(BaseModel):
    """A mapping between a field in the source schema and the target schema.

    Field mappings are generated automatically by the migration planner
    using exact match, alias resolution, and fuzzy matching. Mappings
    with confidence below AUTO_ACCEPT_CONFIDENCE_THRESHOLD require manual
    confirmation before the plan can be approved.

    Attributes:
        id: Unique mapping identifier (UUID v4).
        source_schema_id: ID of the source schema definition.
        target_schema_id: ID of the target schema definition.
        source_field: Dot-notation path to the source field.
        target_field: Dot-notation path to the target field.
        transform_rule: Optional transformation expression (e.g., "UPPER(source)").
        confidence: Mapping confidence score (0.0 to 1.0).
        mapping_type: Method by which the mapping was established.
        created_by: Actor (user or service) that created this mapping.
        created_at: UTC timestamp when the mapping was created.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique mapping identifier (UUID v4)",
    )
    source_schema_id: str = Field(
        ...,
        description="ID of the source schema definition",
    )
    target_schema_id: str = Field(
        ...,
        description="ID of the target schema definition",
    )
    source_field: str = Field(
        ...,
        description="Dot-notation path to the source field",
    )
    target_field: str = Field(
        ...,
        description="Dot-notation path to the target field",
    )
    transform_rule: Optional[str] = Field(
        None,
        description="Optional transformation expression (e.g., 'UPPER(source)')",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Mapping confidence score (0.0 to 1.0)",
    )
    mapping_type: MappingType = Field(
        default=MappingType.EXACT,
        description="Method by which the mapping was established",
    )
    created_by: str = Field(
        default="",
        description="Actor (user or service) that created this mapping",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the mapping was created",
    )

    model_config = {"extra": "forbid"}

    @field_validator("source_schema_id")
    @classmethod
    def validate_source_schema_id(cls, v: str) -> str:
        """Validate source_schema_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("source_schema_id must be non-empty")
        return v

    @field_validator("target_schema_id")
    @classmethod
    def validate_target_schema_id(cls, v: str) -> str:
        """Validate target_schema_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("target_schema_id must be non-empty")
        return v

    @field_validator("source_field")
    @classmethod
    def validate_source_field(cls, v: str) -> str:
        """Validate source_field is non-empty."""
        if not v or not v.strip():
            raise ValueError("source_field must be non-empty")
        return v

    @field_validator("target_field")
    @classmethod
    def validate_target_field(cls, v: str) -> str:
        """Validate target_field is non-empty."""
        if not v or not v.strip():
            raise ValueError("target_field must be non-empty")
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Validate confidence is in range [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {v}")
        return v


class DriftEvent(BaseModel):
    """A single schema drift event detected in a live dataset.

    Drift events are produced by the drift monitor when actual data
    observed in a dataset does not conform to the declared schema
    version. Events are stored for alerting, trend analysis, and
    triggering automated schema evolution workflows.

    Attributes:
        id: Unique drift event identifier (UUID v4).
        schema_id: ID of the schema definition against which drift was detected.
        version_id: ID of the specific schema version used for comparison.
        dataset_id: Identifier of the dataset or data source being monitored.
        drift_type: Category of schema drift detected.
        field_path: Dot-notation path to the drifted field (if applicable).
        expected_value: Expected field type, constraint, or enum set.
        actual_value: Observed field type, constraint, or value.
        severity: Impact severity of this drift event.
        sample_count: Number of records exhibiting this drift.
        detected_at: UTC timestamp when the drift was first detected.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique drift event identifier (UUID v4)",
    )
    schema_id: str = Field(
        ...,
        description="ID of the schema definition against which drift was detected",
    )
    version_id: str = Field(
        ...,
        description="ID of the specific schema version used for comparison",
    )
    dataset_id: str = Field(
        ...,
        description="Identifier of the dataset or data source being monitored",
    )
    drift_type: DriftType = Field(
        ...,
        description="Category of schema drift detected",
    )
    field_path: str = Field(
        default="",
        description="Dot-notation path to the drifted field (if applicable)",
    )
    expected_value: Optional[str] = Field(
        None,
        description="Expected field type, constraint, or enum set",
    )
    actual_value: Optional[str] = Field(
        None,
        description="Observed field type, constraint, or value",
    )
    severity: DriftSeverity = Field(
        default=DriftSeverity.MEDIUM,
        description="Impact severity of this drift event",
    )
    sample_count: int = Field(
        default=1,
        ge=1,
        description="Number of records exhibiting this drift",
    )
    detected_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the drift was first detected",
    )

    model_config = {"extra": "forbid"}

    @field_validator("schema_id")
    @classmethod
    def validate_schema_id(cls, v: str) -> str:
        """Validate schema_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("schema_id must be non-empty")
        return v

    @field_validator("version_id")
    @classmethod
    def validate_version_id(cls, v: str) -> str:
        """Validate version_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("version_id must be non-empty")
        return v

    @field_validator("dataset_id")
    @classmethod
    def validate_dataset_id(cls, v: str) -> str:
        """Validate dataset_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("dataset_id must be non-empty")
        return v


class AuditEntry(BaseModel):
    """An immutable audit log entry for any schema migration action.

    All create, update, delete, and execution actions in the Schema
    Migration Agent produce an AuditEntry. Entries form a provenance
    chain using SHA-256 hashes linking each entry to its parent.

    Attributes:
        id: Unique audit entry identifier (UUID v4).
        action: Action verb (e.g., "register_schema", "execute_migration").
        entity_type: Type of entity acted upon (e.g., "SchemaDefinition").
        entity_id: ID of the entity that was acted upon.
        actor: User, service, or system that performed the action.
        details: Structured details about the action and its parameters.
        previous_state: Snapshot of the entity state before the action.
        new_state: Snapshot of the entity state after the action.
        provenance_hash: SHA-256 hash of this entry's content.
        parent_hash: SHA-256 hash of the immediately preceding audit entry.
        created_at: UTC timestamp when the audit entry was created.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique audit entry identifier (UUID v4)",
    )
    action: str = Field(
        ...,
        description="Action verb (e.g., 'register_schema', 'execute_migration')",
    )
    entity_type: str = Field(
        ...,
        description="Type of entity acted upon (e.g., 'SchemaDefinition')",
    )
    entity_id: str = Field(
        ...,
        description="ID of the entity that was acted upon",
    )
    actor: str = Field(
        default="system",
        description="User, service, or system that performed the action",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured details about the action and its parameters",
    )
    previous_state: Optional[Dict[str, Any]] = Field(
        None,
        description="Snapshot of the entity state before the action",
    )
    new_state: Optional[Dict[str, Any]] = Field(
        None,
        description="Snapshot of the entity state after the action",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of this entry's content for tamper detection",
    )
    parent_hash: str = Field(
        default="",
        description="SHA-256 hash of the immediately preceding audit entry",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the audit entry was created",
    )

    model_config = {"extra": "forbid"}

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        """Validate action is non-empty."""
        if not v or not v.strip():
            raise ValueError("action must be non-empty")
        return v

    @field_validator("entity_type")
    @classmethod
    def validate_entity_type(cls, v: str) -> str:
        """Validate entity_type is non-empty."""
        if not v or not v.strip():
            raise ValueError("entity_type must be non-empty")
        return v

    @field_validator("entity_id")
    @classmethod
    def validate_entity_id(cls, v: str) -> str:
        """Validate entity_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("entity_id must be non-empty")
        return v


class SchemaGroup(BaseModel):
    """A logical grouping of related schema definitions.

    SchemaGroups enable bulk operations such as cross-schema compatibility
    checks, grouped migration planning, and namespace-level governance
    policy enforcement.

    Attributes:
        name: Unique human-readable name for the group.
        description: Description of the group's purpose and membership criteria.
        schema_ids: List of SchemaDefinition IDs belonging to this group.
        created_at: UTC timestamp when the group was created.
    """

    name: str = Field(
        ...,
        description="Unique human-readable name for the group",
    )
    description: str = Field(
        default="",
        description="Description of the group's purpose and membership criteria",
    )
    schema_ids: List[str] = Field(
        default_factory=list,
        description="List of SchemaDefinition IDs belonging to this group",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the group was created",
    )

    model_config = {"extra": "forbid"}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v


class MigrationReport(BaseModel):
    """Aggregated summary report for a completed migration pipeline run.

    Provides a high-level overview of all work performed during a single
    pipeline invocation, including counts of schemas processed, changes
    detected, migrations executed, rollbacks triggered, and drift events
    observed.

    Attributes:
        pipeline_id: Unique identifier for the pipeline run (UUID v4).
        schemas_processed: Number of schema definitions processed.
        versions_created: Number of new schema versions registered.
        changes_detected: Total number of structural changes detected.
        migrations_executed: Number of migration plans executed.
        rollbacks: Number of rollback operations triggered.
        drift_events: Number of schema drift events detected.
        total_processing_time_ms: Total wall-clock time for the pipeline run.
        created_at: UTC timestamp when the report was generated.
    """

    pipeline_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the pipeline run (UUID v4)",
    )
    schemas_processed: int = Field(
        default=0,
        ge=0,
        description="Number of schema definitions processed",
    )
    versions_created: int = Field(
        default=0,
        ge=0,
        description="Number of new schema versions registered",
    )
    changes_detected: int = Field(
        default=0,
        ge=0,
        description="Total number of structural changes detected",
    )
    migrations_executed: int = Field(
        default=0,
        ge=0,
        description="Number of migration plans executed",
    )
    rollbacks: int = Field(
        default=0,
        ge=0,
        description="Number of rollback operations triggered",
    )
    drift_events: int = Field(
        default=0,
        ge=0,
        description="Number of schema drift events detected",
    )
    total_processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Total wall-clock time for the pipeline run in milliseconds",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the report was generated",
    )

    model_config = {"extra": "forbid"}


class SchemaStatistics(BaseModel):
    """Aggregated operational statistics for the Schema Migration Agent service.

    Provides high-level metrics for service health monitoring, capacity
    planning, and SLO tracking. All counts are cumulative since service
    inception unless otherwise noted.

    Attributes:
        total_schemas: Total number of registered schema definitions.
        total_versions: Total number of registered schema versions.
        total_changes: Total number of structural changes detected.
        total_migrations: Total number of migration plans executed.
        total_rollbacks: Total number of rollback operations performed.
        total_drift_events: Total number of drift events recorded.
        schemas_by_type: Schema count broken down by SchemaType.
        schemas_by_status: Schema count broken down by SchemaStatus.
    """

    total_schemas: int = Field(
        default=0,
        ge=0,
        description="Total number of registered schema definitions",
    )
    total_versions: int = Field(
        default=0,
        ge=0,
        description="Total number of registered schema versions",
    )
    total_changes: int = Field(
        default=0,
        ge=0,
        description="Total number of structural changes detected",
    )
    total_migrations: int = Field(
        default=0,
        ge=0,
        description="Total number of migration plans executed",
    )
    total_rollbacks: int = Field(
        default=0,
        ge=0,
        description="Total number of rollback operations performed",
    )
    total_drift_events: int = Field(
        default=0,
        ge=0,
        description="Total number of drift events recorded",
    )
    schemas_by_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Schema count broken down by SchemaType",
    )
    schemas_by_status: Dict[str, int] = Field(
        default_factory=dict,
        description="Schema count broken down by SchemaStatus",
    )

    model_config = {"extra": "forbid"}


class VersionComparison(BaseModel):
    """A side-by-side comparison result for two schema versions.

    Produced by the change detection engine when comparing a source
    and target version. Contains the full list of detected changes,
    the resolved compatibility level, and whether a migration plan
    is required to safely transition between versions.

    Attributes:
        source_version: SemVer string of the source (older) version.
        target_version: SemVer string of the target (newer) version.
        changes: List of all structural changes detected.
        compatibility_level: Overall compatibility classification.
        migration_needed: Whether a migration plan must be executed.
    """

    source_version: str = Field(
        ...,
        description="SemVer string of the source (older) version",
    )
    target_version: str = Field(
        ...,
        description="SemVer string of the target (newer) version",
    )
    changes: List[SchemaChange] = Field(
        default_factory=list,
        description="List of all structural changes detected",
    )
    compatibility_level: CompatibilityLevel = Field(
        default=CompatibilityLevel.NONE,
        description="Overall compatibility classification",
    )
    migration_needed: bool = Field(
        default=False,
        description="Whether a migration plan must be executed",
    )

    model_config = {"extra": "forbid"}

    @field_validator("source_version")
    @classmethod
    def validate_source_version(cls, v: str) -> str:
        """Validate source_version follows SemVer format."""
        if not v or not v.strip():
            raise ValueError("source_version must be non-empty")
        if not _SEMVER_RE.match(v):
            raise ValueError(
                f"source_version '{v}' does not follow SemVer format (MAJOR.MINOR.PATCH)"
            )
        return v

    @field_validator("target_version")
    @classmethod
    def validate_target_version(cls, v: str) -> str:
        """Validate target_version follows SemVer format."""
        if not v or not v.strip():
            raise ValueError("target_version must be non-empty")
        if not _SEMVER_RE.match(v):
            raise ValueError(
                f"target_version '{v}' does not follow SemVer format (MAJOR.MINOR.PATCH)"
            )
        return v


class PipelineResult(BaseModel):
    """The complete result of a full schema migration pipeline run.

    A pipeline run encompasses all stages: change detection, compatibility
    checking, migration plan creation, dry-run execution, and verification.
    The PipelineResult aggregates the outputs of each stage into a single
    return value from the RunPipeline endpoint.

    Attributes:
        pipeline_id: Unique identifier for this pipeline run (UUID v4).
        stages_completed: Names of stages that completed successfully.
        stages_failed: Names of stages that failed during the run.
        changes: VersionComparison result from the change detection stage.
        compatibility: CompatibilityResult from the compatibility stage.
        plan: MigrationPlan created or updated during the run.
        execution: MigrationExecution record if execution was performed.
        verification: Structured verification results after execution.
        total_time_ms: Total wall-clock time for the entire pipeline run.
    """

    pipeline_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this pipeline run (UUID v4)",
    )
    stages_completed: List[str] = Field(
        default_factory=list,
        description="Names of stages that completed successfully",
    )
    stages_failed: List[str] = Field(
        default_factory=list,
        description="Names of stages that failed during the run",
    )
    changes: Optional[VersionComparison] = Field(
        None,
        description="VersionComparison result from the change detection stage",
    )
    compatibility: Optional[CompatibilityResult] = Field(
        None,
        description="CompatibilityResult from the compatibility analysis stage",
    )
    plan: Optional[MigrationPlan] = Field(
        None,
        description="MigrationPlan created or updated during the run",
    )
    execution: Optional[MigrationExecution] = Field(
        None,
        description="MigrationExecution record if execution was performed",
    )
    verification: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured verification results after execution",
    )
    total_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Total wall-clock time for the entire pipeline run in milliseconds",
    )

    model_config = {"extra": "forbid"}


# =============================================================================
# Request Models (8)
# =============================================================================


class RegisterSchemaRequest(BaseModel):
    """Request body for registering a new schema definition.

    Attributes:
        namespace: Dot/hyphen-separated namespace for the schema.
        name: Human-readable schema name (max 255 characters).
        schema_type: Serialization format (json_schema, avro, protobuf).
        owner: Team or service responsible for this schema.
        tags: Arbitrary key-value labels for discovery and filtering.
        description: Human-readable description of the schema's purpose.
        definition_json: The schema definition as a JSON-serializable dict.
    """

    namespace: str = Field(
        ...,
        description="Dot/hyphen-separated namespace for the schema",
    )
    name: str = Field(
        ...,
        max_length=255,
        description="Human-readable schema name (max 255 characters)",
    )
    schema_type: SchemaType = Field(
        default=SchemaType.JSON_SCHEMA,
        description="Serialization format (json_schema, avro, protobuf)",
    )
    owner: str = Field(
        default="",
        description="Team or service responsible for this schema",
    )
    tags: Dict[str, str] = Field(
        default_factory=dict,
        description="Arbitrary key-value labels for discovery and filtering",
    )
    description: str = Field(
        default="",
        description="Human-readable description of the schema's purpose",
    )
    definition_json: Dict[str, Any] = Field(
        default_factory=dict,
        description="The schema definition as a JSON-serializable dict",
    )

    model_config = {"extra": "forbid"}

    @field_validator("namespace")
    @classmethod
    def validate_namespace(cls, v: str) -> str:
        """Validate namespace is non-empty and uses only alphanumeric, hyphens, and dots."""
        if not v or not v.strip():
            raise ValueError("namespace must be non-empty")
        if not _NAMESPACE_RE.match(v):
            raise ValueError(
                "namespace must contain only alphanumeric characters, hyphens, "
                "and dots, and must start and end with an alphanumeric character"
            )
        return v

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty and within length limit."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        if len(v) > 255:
            raise ValueError("name must not exceed 255 characters")
        return v


class UpdateSchemaRequest(BaseModel):
    """Request body for updating mutable fields of an existing schema definition.

    Only fields explicitly included in this model can be updated.
    The namespace, name, and schema_type are immutable once registered.

    Attributes:
        owner: Updated team or service owner of the schema.
        tags: Updated key-value labels for discovery and filtering.
        status: Updated lifecycle status (e.g., ACTIVE, DEPRECATED).
        description: Updated human-readable description.
    """

    owner: Optional[str] = Field(
        None,
        description="Updated team or service owner of the schema",
    )
    tags: Optional[Dict[str, str]] = Field(
        None,
        description="Updated key-value labels for discovery and filtering",
    )
    status: Optional[SchemaStatus] = Field(
        None,
        description="Updated lifecycle status (e.g., ACTIVE, DEPRECATED)",
    )
    description: Optional[str] = Field(
        None,
        description="Updated human-readable description",
    )

    model_config = {"extra": "forbid"}


class CreateVersionRequest(BaseModel):
    """Request body for registering a new version of an existing schema.

    The version string must be greater than the current latest version
    according to SemVer comparison rules.

    Attributes:
        schema_id: ID of the parent SchemaDefinition to version.
        definition_json: The new schema definition at this version.
        changelog_note: Human-readable description of what changed.
    """

    schema_id: str = Field(
        ...,
        description="ID of the parent SchemaDefinition to version",
    )
    definition_json: Dict[str, Any] = Field(
        ...,
        description="The new schema definition at this version",
    )
    changelog_note: str = Field(
        default="",
        description="Human-readable description of what changed in this version",
    )

    model_config = {"extra": "forbid"}

    @field_validator("schema_id")
    @classmethod
    def validate_schema_id(cls, v: str) -> str:
        """Validate schema_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("schema_id must be non-empty")
        return v


class DetectChangesRequest(BaseModel):
    """Request body for detecting structural changes between two schema versions.

    Triggers the change detection engine to produce a list of SchemaChange
    objects describing every field-level or schema-level difference.

    Attributes:
        source_version_id: ID of the source (older) schema version.
        target_version_id: ID of the target (newer) schema version.
    """

    source_version_id: str = Field(
        ...,
        description="ID of the source (older) schema version",
    )
    target_version_id: str = Field(
        ...,
        description="ID of the target (newer) schema version",
    )

    model_config = {"extra": "forbid"}

    @field_validator("source_version_id")
    @classmethod
    def validate_source_version_id(cls, v: str) -> str:
        """Validate source_version_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("source_version_id must be non-empty")
        return v

    @field_validator("target_version_id")
    @classmethod
    def validate_target_version_id(cls, v: str) -> str:
        """Validate target_version_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("target_version_id must be non-empty")
        return v


class CheckCompatibilityRequest(BaseModel):
    """Request body for checking compatibility between two schema versions.

    Triggers the compatibility checker to determine whether the change
    from source to target is backward-compatible, forward-compatible,
    fully compatible, or breaking.

    Attributes:
        source_version_id: ID of the source (older) schema version.
        target_version_id: ID of the target (newer) schema version.
    """

    source_version_id: str = Field(
        ...,
        description="ID of the source (older) schema version",
    )
    target_version_id: str = Field(
        ...,
        description="ID of the target (newer) schema version",
    )

    model_config = {"extra": "forbid"}

    @field_validator("source_version_id")
    @classmethod
    def validate_source_version_id(cls, v: str) -> str:
        """Validate source_version_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("source_version_id must be non-empty")
        return v

    @field_validator("target_version_id")
    @classmethod
    def validate_target_version_id(cls, v: str) -> str:
        """Validate target_version_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("target_version_id must be non-empty")
        return v


class CreatePlanRequest(BaseModel):
    """Request body for generating a migration plan between two schema versions.

    Triggers the migration planner to produce an ordered list of
    MigrationStep objects for transforming records from the source
    version to the target version.

    Attributes:
        source_schema_id: ID of the source schema definition.
        target_schema_id: ID of the target schema definition.
        source_version: SemVer string of the source schema version.
        target_version: SemVer string of the target schema version.
    """

    source_schema_id: str = Field(
        ...,
        description="ID of the source schema definition",
    )
    target_schema_id: str = Field(
        ...,
        description="ID of the target schema definition",
    )
    source_version: str = Field(
        ...,
        description="SemVer string of the source schema version",
    )
    target_version: str = Field(
        ...,
        description="SemVer string of the target schema version",
    )

    model_config = {"extra": "forbid"}

    @field_validator("source_schema_id")
    @classmethod
    def validate_source_schema_id(cls, v: str) -> str:
        """Validate source_schema_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("source_schema_id must be non-empty")
        return v

    @field_validator("target_schema_id")
    @classmethod
    def validate_target_schema_id(cls, v: str) -> str:
        """Validate target_schema_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("target_schema_id must be non-empty")
        return v

    @field_validator("source_version")
    @classmethod
    def validate_source_version(cls, v: str) -> str:
        """Validate source_version follows SemVer format."""
        if not v or not v.strip():
            raise ValueError("source_version must be non-empty")
        if not _SEMVER_RE.match(v):
            raise ValueError(
                f"source_version '{v}' does not follow SemVer format (MAJOR.MINOR.PATCH)"
            )
        return v

    @field_validator("target_version")
    @classmethod
    def validate_target_version(cls, v: str) -> str:
        """Validate target_version follows SemVer format."""
        if not v or not v.strip():
            raise ValueError("target_version must be non-empty")
        if not _SEMVER_RE.match(v):
            raise ValueError(
                f"target_version '{v}' does not follow SemVer format (MAJOR.MINOR.PATCH)"
            )
        return v


class ExecuteMigrationRequest(BaseModel):
    """Request body for executing an approved migration plan.

    Triggers the migration executor to apply the plan's steps to
    all records in the target dataset. Supports dry-run mode for
    pre-execution validation without committing changes.

    Attributes:
        plan_id: ID of the MigrationPlan to execute.
        dry_run: If True, validate without committing any changes.
        batch_size: Number of records to process per execution batch.
    """

    plan_id: str = Field(
        ...,
        description="ID of the MigrationPlan to execute",
    )
    dry_run: bool = Field(
        default=False,
        description="If True, validate without committing any changes",
    )
    batch_size: int = Field(
        default=DEFAULT_MIGRATION_BATCH_SIZE,
        ge=1,
        le=100_000,
        description="Number of records to process per execution batch",
    )

    model_config = {"extra": "forbid"}

    @field_validator("plan_id")
    @classmethod
    def validate_plan_id(cls, v: str) -> str:
        """Validate plan_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("plan_id must be non-empty")
        return v


class RunPipelineRequest(BaseModel):
    """Request body for running the full end-to-end schema migration pipeline.

    A single pipeline invocation encompasses all stages: change detection,
    compatibility checking, migration plan creation, optional dry-run
    execution, and post-migration verification.

    Attributes:
        schema_id: ID of the existing schema definition to migrate.
        target_definition_json: The new target schema definition to migrate towards.
        skip_compatibility: If True, skip the compatibility analysis stage.
        skip_dry_run: If True, execute the plan without a prior dry-run.
    """

    schema_id: str = Field(
        ...,
        description="ID of the existing schema definition to migrate",
    )
    target_definition_json: Dict[str, Any] = Field(
        ...,
        description="The new target schema definition to migrate towards",
    )
    skip_compatibility: bool = Field(
        default=False,
        description="If True, skip the compatibility analysis stage",
    )
    skip_dry_run: bool = Field(
        default=False,
        description="If True, execute the plan without a prior dry-run",
    )

    model_config = {"extra": "forbid"}

    @field_validator("schema_id")
    @classmethod
    def validate_schema_id(cls, v: str) -> str:
        """Validate schema_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("schema_id must be non-empty")
        return v


# =============================================================================
# __all__ export list
# =============================================================================

__all__ = [
    # -------------------------------------------------------------------------
    # Layer 1 re-exports (data_quality_profiler.models + schema_compiler)
    # -------------------------------------------------------------------------
    "QualityDimension",
    "RuleType",
    "SchemaCompilerEngine",
    # -------------------------------------------------------------------------
    # Availability flags (for downstream feature detection)
    # -------------------------------------------------------------------------
    "_DQ_AVAILABLE",
    "_SC_AVAILABLE",
    # -------------------------------------------------------------------------
    # Constants
    # -------------------------------------------------------------------------
    "VERSION",
    "MAX_SCHEMAS_PER_NAMESPACE",
    "MAX_FIELDS_PER_SCHEMA",
    "MAX_MIGRATION_STEPS",
    "SUPPORTED_SCHEMA_TYPES",
    "DEFAULT_COMPATIBILITY_RULES",
    "CHANGE_SEVERITY_ORDER",
    "DEFAULT_MIGRATION_BATCH_SIZE",
    "AUTO_ACCEPT_CONFIDENCE_THRESHOLD",
    "MAX_DRIFT_EVENTS_PER_VERSION",
    # -------------------------------------------------------------------------
    # Enumerations (14)
    # -------------------------------------------------------------------------
    "SchemaType",
    "SchemaStatus",
    "ChangeType",
    "ChangeSeverity",
    "CompatibilityLevel",
    "MigrationPlanStatus",
    "ExecutionStatus",
    "RollbackType",
    "RollbackStatus",
    "DriftType",
    "DriftSeverity",
    "MappingType",
    "EffortLevel",
    "TransformationType",
    # -------------------------------------------------------------------------
    # SDK data models (16)
    # -------------------------------------------------------------------------
    "SchemaDefinition",
    "SchemaVersion",
    "SchemaChange",
    "CompatibilityResult",
    "MigrationStep",
    "MigrationPlan",
    "MigrationExecution",
    "RollbackRecord",
    "FieldMapping",
    "DriftEvent",
    "AuditEntry",
    "SchemaGroup",
    "MigrationReport",
    "SchemaStatistics",
    "VersionComparison",
    "PipelineResult",
    # -------------------------------------------------------------------------
    # Request models (8)
    # -------------------------------------------------------------------------
    "RegisterSchemaRequest",
    "UpdateSchemaRequest",
    "CreateVersionRequest",
    "DetectChangesRequest",
    "CheckCompatibilityRequest",
    "CreatePlanRequest",
    "ExecuteMigrationRequest",
    "RunPipelineRequest",
]
