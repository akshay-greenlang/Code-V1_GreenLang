# -*- coding: utf-8 -*-
"""
GL-DATA-X-020: GreenLang Schema Migration Agent SDK
=====================================================

This package provides schema registry management, schema versioning,
change detection, compatibility checking, migration planning, migration
execution, and end-to-end pipeline orchestration SDK for the GreenLang
framework. It supports:

- Schema registry with multi-format support (JSON Schema, Avro, Protobuf,
  SQL DDL, custom) and namespace isolation
- Schema versioning with SemVer (major/minor/patch) bump semantics
  and full version history tracking
- Change detection between schema versions (field additions, removals,
  renames, type changes, constraint changes, index changes)
- Compatibility checking (backward, forward, full, none, transitive
  variants) following Apache Avro / Confluent Schema Registry semantics
- Migration planning with dependency-aware step generation, dry-run
  validation, and rollback strategy computation
- Migration execution with batch processing, checkpointing, automatic
  rollback on failure, and records-migrated tracking
- End-to-end pipeline orchestration chaining all 6 engines with
  configurable stages and short-circuit on failure
- SHA-256 provenance chain tracking for complete audit trails
- 12 Prometheus metrics with gl_sm_ prefix for observability
- FastAPI REST API with 20 endpoints at /api/v1/schema-migration
- Thread-safe configuration with GL_SM_ env prefix

Key Components:
    - config: SchemaMigrationConfig with GL_SM_ env prefix
    - schema_registry: Schema registration and lookup engine
    - schema_versioner: Schema versioning engine with SemVer semantics
    - change_detector: Schema change detection and diff engine
    - compatibility_checker: Schema compatibility validation engine
    - migration_planner: Migration plan generation engine
    - migration_executor: Migration execution engine with rollback
    - schema_migration_pipeline: End-to-end pipeline orchestration engine
    - provenance: SHA-256 chain-hashed audit trails
    - metrics: 12 Prometheus metrics with gl_sm_ prefix
    - api: FastAPI HTTP service with 20 endpoints
    - setup: SchemaMigrationService facade

Example:
    >>> from greenlang.schema_migration import SchemaMigrationService
    >>> service = SchemaMigrationService()
    >>> result = service.register_schema(
    ...     name="emissions_v2",
    ...     schema_type="json_schema",
    ...     definition={"type": "object", "properties": {"co2e": {"type": "number"}}},
    ...     namespace="emissions",
    ... )
    >>> print(result.schema_id, result.version)
    emissions_v2 1.0.0

Agent ID: GL-DATA-X-020
Agent Name: Schema Migration Agent
"""

__version__ = "1.0.0"
__agent_id__ = "GL-DATA-X-020"
__agent_name__ = "Schema Migration Agent"

# SDK availability flag
SCHEMA_MIGRATION_SDK_AVAILABLE = True

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from greenlang.schema_migration.config import (
    SchemaMigrationConfig,
    get_config,
    set_config,
    reset_config,
)

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
from greenlang.schema_migration.provenance import ProvenanceTracker

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
from greenlang.schema_migration.metrics import (
    PROMETHEUS_AVAILABLE,
    # Metric objects
    sm_schemas_registered_total,
    sm_versions_created_total,
    sm_changes_detected_total,
    sm_compatibility_checks_total,
    sm_migrations_planned_total,
    sm_migrations_executed_total,
    sm_rollbacks_total,
    sm_drift_detected_total,
    sm_migration_duration_seconds,
    sm_records_migrated,
    sm_processing_duration_seconds,
    sm_active_migrations,
    # Helper functions
    record_schema_registered,
    record_version_created,
    record_change_detected,
    record_compatibility_check,
    record_migration_planned,
    record_migration_executed,
    record_rollback,
    record_drift_detected,
    observe_migration_duration,
    observe_records_migrated,
    observe_processing_duration,
    set_active_migrations,
)

# ---------------------------------------------------------------------------
# Core engines (Layer 2 SDK) - optional imports with graceful fallback
# ---------------------------------------------------------------------------
try:
    from greenlang.schema_migration.schema_registry import SchemaRegistryEngine
except ImportError:
    SchemaRegistryEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.schema_migration.schema_versioner import SchemaVersionerEngine
except ImportError:
    SchemaVersionerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.schema_migration.change_detector import ChangeDetectorEngine
except ImportError:
    ChangeDetectorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.schema_migration.compatibility_checker import CompatibilityCheckerEngine
except ImportError:
    CompatibilityCheckerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.schema_migration.migration_planner import MigrationPlannerEngine
except ImportError:
    MigrationPlannerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.schema_migration.migration_executor import MigrationExecutorEngine
except ImportError:
    MigrationExecutorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.schema_migration.schema_migration_pipeline import SchemaMigrationPipelineEngine
except ImportError:
    SchemaMigrationPipelineEngine = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Service setup facade and models
# ---------------------------------------------------------------------------
from greenlang.schema_migration.setup import (
    SchemaMigrationService,
    configure_schema_migration,
    get_schema_migration,
    get_router,
    # Models
    SchemaResponse,
    SchemaVersionResponse,
    ChangeDetectionResponse,
    CompatibilityCheckResponse,
    MigrationPlanResponse,
    MigrationExecutionResponse,
    PipelineResultResponse,
    SchemaMigrationStatisticsResponse,
)

__all__ = [
    # Version
    "__version__",
    "__agent_id__",
    "__agent_name__",
    # SDK flag
    "SCHEMA_MIGRATION_SDK_AVAILABLE",
    # Configuration
    "SchemaMigrationConfig",
    "get_config",
    "set_config",
    "reset_config",
    # Provenance
    "ProvenanceTracker",
    # Metric objects
    "PROMETHEUS_AVAILABLE",
    "sm_schemas_registered_total",
    "sm_versions_created_total",
    "sm_changes_detected_total",
    "sm_compatibility_checks_total",
    "sm_migrations_planned_total",
    "sm_migrations_executed_total",
    "sm_rollbacks_total",
    "sm_drift_detected_total",
    "sm_migration_duration_seconds",
    "sm_records_migrated",
    "sm_processing_duration_seconds",
    "sm_active_migrations",
    # Metric helper functions
    "record_schema_registered",
    "record_version_created",
    "record_change_detected",
    "record_compatibility_check",
    "record_migration_planned",
    "record_migration_executed",
    "record_rollback",
    "record_drift_detected",
    "observe_migration_duration",
    "observe_records_migrated",
    "observe_processing_duration",
    "set_active_migrations",
    # Core engines (Layer 2)
    "SchemaRegistryEngine",
    "SchemaVersionerEngine",
    "ChangeDetectorEngine",
    "CompatibilityCheckerEngine",
    "MigrationPlannerEngine",
    "MigrationExecutorEngine",
    "SchemaMigrationPipelineEngine",
    # Service setup facade
    "SchemaMigrationService",
    "configure_schema_migration",
    "get_schema_migration",
    "get_router",
    # Response models
    "SchemaResponse",
    "SchemaVersionResponse",
    "ChangeDetectionResponse",
    "CompatibilityCheckResponse",
    "MigrationPlanResponse",
    "MigrationExecutionResponse",
    "PipelineResultResponse",
    "SchemaMigrationStatisticsResponse",
]
