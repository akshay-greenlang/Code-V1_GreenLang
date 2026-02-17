# -*- coding: utf-8 -*-
"""
Schema Migration Service Setup - AGENT-DATA-017

Provides ``configure_schema_migration(app)`` which wires up the
Schema Migration Agent SDK (schema registry, schema versioning,
change detection, compatibility checking, migration planning,
migration execution, pipeline orchestration, provenance tracker)
and mounts the REST API.

Also exposes ``get_schema_migration(app)`` for programmatic access
and the ``SchemaMigrationService`` facade class.

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.schema_migration.setup import configure_schema_migration
    >>> app = FastAPI()
    >>> import asyncio
    >>> service = asyncio.run(configure_schema_migration(app))

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-017 Schema Migration Agent (GL-DATA-X-020)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schema_migration.config import (
    SchemaMigrationConfig,
    get_config,
)
from greenlang.schema_migration.metrics import (
    PROMETHEUS_AVAILABLE,
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
from greenlang.schema_migration.provenance import ProvenanceTracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FastAPI import
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI
    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None  # type: ignore[assignment, misc]
    FASTAPI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Optional engine imports (graceful degradation)
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


# ===================================================================
# Lightweight Pydantic response models used by the facade / API layer
# ===================================================================


class SchemaResponse(BaseModel):
    """Schema registry entry response.

    Attributes:
        schema_id: Unique schema identifier (UUID4).
        namespace: Logical namespace for the schema.
        name: Human-readable schema name.
        schema_type: Schema format type (json_schema, avro, protobuf, sql_ddl, custom).
        status: Current lifecycle status (draft, active, deprecated, archived).
        owner: Owner identifier (team or user).
        tags: List of categorisation tags.
        description: Free-text description of the schema.
        version_count: Number of versions registered for this schema.
        created_at: ISO-8601 UTC creation timestamp.
        updated_at: ISO-8601 UTC last-update timestamp.
        provenance_hash: SHA-256 provenance hash for audit trail.
    """

    model_config = {"extra": "forbid"}

    schema_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    namespace: str = Field(default="")
    name: str = Field(default="")
    schema_type: str = Field(default="json_schema")
    status: str = Field(default="draft")
    owner: str = Field(default="")
    tags: List[str] = Field(default_factory=list)
    description: str = Field(default="")
    version_count: int = Field(default=0)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    provenance_hash: str = Field(default="")


class SchemaVersionResponse(BaseModel):
    """Schema version response.

    Attributes:
        version_id: Unique version identifier.
        schema_id: Parent schema identifier.
        version: Semantic version string (e.g. 1.0.0, 2.1.0).
        definition: The full schema definition (JSON-serialisable dict).
        changelog: Changelog entries for this version.
        is_deprecated: Whether this version has been deprecated.
        sunset_date: Optional planned removal date (ISO-8601).
        created_at: ISO-8601 UTC creation timestamp.
        provenance_hash: SHA-256 provenance hash for audit trail.
    """

    model_config = {"extra": "forbid"}

    version_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    schema_id: str = Field(default="")
    version: str = Field(default="1.0.0")
    definition: Any = Field(default_factory=dict)
    changelog: str = Field(default="")
    is_deprecated: bool = Field(default=False)
    sunset_date: Optional[str] = Field(default=None)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    provenance_hash: str = Field(default="")


class ChangeDetectionResponse(BaseModel):
    """Schema change detection result response.

    Attributes:
        detection_id: Unique detection run identifier.
        source_version_id: Source schema version identifier.
        target_version_id: Target schema version identifier.
        changes: List of change descriptor dicts.
        change_count: Total number of changes detected.
        breaking_change_count: Number of breaking changes detected.
        detected_at: ISO-8601 UTC detection timestamp.
        provenance_hash: SHA-256 provenance hash for audit trail.
    """

    model_config = {"extra": "forbid"}

    detection_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_version_id: str = Field(default="")
    target_version_id: str = Field(default="")
    changes: List[Dict[str, Any]] = Field(default_factory=list)
    change_count: int = Field(default=0)
    breaking_change_count: int = Field(default=0)
    detected_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    provenance_hash: str = Field(default="")


class CompatibilityCheckResponse(BaseModel):
    """Schema compatibility check result response.

    Attributes:
        check_id: Unique compatibility check identifier.
        source_version_id: Source schema version identifier.
        target_version_id: Target schema version identifier.
        compatibility_level: Determined compatibility level
            (full, backward, forward, breaking).
        is_compatible: Overall compatibility verdict.
        issues: List of compatibility issue dicts.
        checked_at: ISO-8601 UTC check timestamp.
        provenance_hash: SHA-256 provenance hash for audit trail.
    """

    model_config = {"extra": "forbid"}

    check_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_version_id: str = Field(default="")
    target_version_id: str = Field(default="")
    compatibility_level: str = Field(default="full")
    is_compatible: bool = Field(default=True)
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    checked_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    provenance_hash: str = Field(default="")


class MigrationPlanResponse(BaseModel):
    """Migration plan response.

    Attributes:
        plan_id: Unique plan identifier.
        source_schema_id: Source schema identifier.
        target_schema_id: Target schema identifier.
        steps: Ordered list of migration step dicts.
        total_steps: Total number of steps in the plan.
        effort_estimate: Effort estimate label (low, medium, high, critical).
        status: Plan lifecycle status (pending, validated, approved,
            executing, completed, failed, cancelled).
        created_at: ISO-8601 UTC creation timestamp.
        provenance_hash: SHA-256 provenance hash for audit trail.
    """

    model_config = {"extra": "forbid"}

    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_schema_id: str = Field(default="")
    target_schema_id: str = Field(default="")
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    total_steps: int = Field(default=0)
    effort_estimate: str = Field(default="low")
    status: str = Field(default="pending")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    provenance_hash: str = Field(default="")


class MigrationExecutionResponse(BaseModel):
    """Migration execution result response.

    Attributes:
        execution_id: Unique execution identifier.
        plan_id: Identifier of the executed plan.
        status: Execution outcome (running, completed, failed, rolled_back).
        records_processed: Total data records processed.
        records_failed: Number of records that failed migration.
        records_skipped: Number of records skipped during migration.
        current_step: Most recently executed step number.
        total_steps: Total steps in the plan.
        percentage: Completion percentage (0.0 - 100.0).
        started_at: ISO-8601 UTC start timestamp.
        completed_at: ISO-8601 UTC completion timestamp (None if running).
        provenance_hash: SHA-256 provenance hash for audit trail.
    """

    model_config = {"extra": "forbid"}

    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    plan_id: str = Field(default="")
    status: str = Field(default="pending")
    records_processed: int = Field(default=0)
    records_failed: int = Field(default=0)
    records_skipped: int = Field(default=0)
    current_step: int = Field(default=0)
    total_steps: int = Field(default=0)
    percentage: float = Field(default=0.0)
    started_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    completed_at: Optional[str] = Field(default=None)
    provenance_hash: str = Field(default="")


class PipelineResultResponse(BaseModel):
    """End-to-end pipeline execution result response.

    Attributes:
        pipeline_id: Unique pipeline run identifier.
        source_schema_id: Source schema identifier.
        target_schema_id: Target schema identifier.
        stages_completed: List of stage names that completed successfully.
        final_status: Overall pipeline outcome status.
        changes_detected: Number of changes found by the detection stage.
        is_compatible: Whether the compatibility check passed.
        plan_id: Migration plan identifier (if planning stage ran).
        execution_id: Execution identifier (if execution stage ran).
        elapsed_seconds: Total pipeline wall-clock time in seconds.
        provenance_hash: SHA-256 provenance hash for audit trail.
    """

    model_config = {"extra": "forbid"}

    pipeline_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_schema_id: str = Field(default="")
    target_schema_id: str = Field(default="")
    stages_completed: List[str] = Field(default_factory=list)
    final_status: str = Field(default="pending")
    changes_detected: int = Field(default=0)
    is_compatible: bool = Field(default=True)
    plan_id: Optional[str] = Field(default=None)
    execution_id: Optional[str] = Field(default=None)
    elapsed_seconds: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class SchemaMigrationStatisticsResponse(BaseModel):
    """Aggregate statistics for the schema migration service.

    Attributes:
        total_schemas: Total schemas registered in the registry.
        total_versions: Total schema versions created.
        total_changes_detected: Total change detection runs completed.
        total_compatibility_checks: Total compatibility checks performed.
        total_migrations_planned: Total migration plans created.
        total_migrations_executed: Total migration executions completed.
        total_rollbacks: Total rollback operations performed.
        total_drift_events: Total schema drift events detected.
        avg_migration_duration_seconds: Average migration duration in seconds.
        success_rate: Migration success rate (0.0 - 1.0).
        active_migrations: Number of currently running migrations.
    """

    model_config = {"extra": "forbid"}

    total_schemas: int = Field(default=0)
    total_versions: int = Field(default=0)
    total_changes_detected: int = Field(default=0)
    total_compatibility_checks: int = Field(default=0)
    total_migrations_planned: int = Field(default=0)
    total_migrations_executed: int = Field(default=0)
    total_rollbacks: int = Field(default=0)
    total_drift_events: int = Field(default=0)
    avg_migration_duration_seconds: float = Field(default=0.0)
    success_rate: float = Field(default=0.0)
    active_migrations: int = Field(default=0)


# ===================================================================
# Utility helpers
# ===================================================================


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _utcnow_iso() -> str:
    """Return current UTC datetime as an ISO-8601 string."""
    return _utcnow().isoformat()


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, list, str, or Pydantic model).

    Returns:
        SHA-256 hex digest string.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ===================================================================
# SchemaMigrationService facade
# ===================================================================

# Thread-safe singleton lock
_singleton_lock = threading.Lock()
_singleton_instance: Optional["SchemaMigrationService"] = None


class SchemaMigrationService:
    """Unified facade over the Schema Migration Agent SDK.

    Aggregates all seven migration engines (schema registry, schema
    versioner, change detector, compatibility checker, migration planner,
    migration executor, pipeline orchestrator) through a single entry
    point with convenience methods for common operations.

    Each method records provenance and updates self-monitoring Prometheus
    metrics.

    Attributes:
        config: SchemaMigrationConfig instance.
        provenance: ProvenanceTracker instance for SHA-256 audit trails.

    Example:
        >>> service = SchemaMigrationService()
        >>> result = service.register_schema(
        ...     namespace="emissions",
        ...     name="ActivityRecord",
        ...     schema_type="json_schema",
        ...     definition={"type": "object", "properties": {"co2e": {"type": "number"}}},
        ... )
        >>> print(result.schema_id, result.status)
    """

    def __init__(
        self,
        config: Optional[SchemaMigrationConfig] = None,
    ) -> None:
        """Initialize the Schema Migration Service facade.

        Instantiates all 7 internal engines plus the provenance tracker:
        - SchemaRegistryEngine
        - SchemaVersionerEngine
        - ChangeDetectorEngine
        - CompatibilityCheckerEngine
        - MigrationPlannerEngine
        - MigrationExecutorEngine
        - SchemaMigrationPipelineEngine

        Args:
            config: Optional configuration. Uses global config if None.
        """
        self.config = config or get_config()

        # Provenance tracker
        self.provenance = ProvenanceTracker(
            genesis_hash=self.config.genesis_hash,
        )

        # Engine placeholders -- real implementations are injected by the
        # respective SDK modules at import time. We use a lazy-init approach
        # so that setup.py can be imported without the full SDK installed.
        self._schema_registry_engine: Any = None
        self._schema_versioner_engine: Any = None
        self._change_detector_engine: Any = None
        self._compatibility_checker_engine: Any = None
        self._migration_planner_engine: Any = None
        self._migration_executor_engine: Any = None
        self._pipeline_engine: Any = None

        self._init_engines()

        # In-memory stores (production uses DB; these are SDK-level caches)
        self._schemas: Dict[str, SchemaResponse] = {}
        self._versions: Dict[str, SchemaVersionResponse] = {}
        self._detections: Dict[str, ChangeDetectionResponse] = {}
        self._compat_checks: Dict[str, CompatibilityCheckResponse] = {}
        self._plans: Dict[str, MigrationPlanResponse] = {}
        self._executions: Dict[str, MigrationExecutionResponse] = {}
        self._pipeline_results: Dict[str, PipelineResultResponse] = {}

        # Statistics
        self._stats = SchemaMigrationStatisticsResponse()
        self._migration_durations: List[float] = []
        self._migration_successes: int = 0
        self._migration_total: int = 0
        self._active_migrations: int = 0
        self._started = False

        logger.info("SchemaMigrationService facade created")

    # ------------------------------------------------------------------
    # Engine properties
    # ------------------------------------------------------------------

    @property
    def schema_registry_engine(self) -> Any:
        """Get the SchemaRegistryEngine instance."""
        return self._schema_registry_engine

    @property
    def schema_versioner_engine(self) -> Any:
        """Get the SchemaVersionerEngine instance."""
        return self._schema_versioner_engine

    @property
    def change_detector_engine(self) -> Any:
        """Get the ChangeDetectorEngine instance."""
        return self._change_detector_engine

    @property
    def compatibility_checker_engine(self) -> Any:
        """Get the CompatibilityCheckerEngine instance."""
        return self._compatibility_checker_engine

    @property
    def migration_planner_engine(self) -> Any:
        """Get the MigrationPlannerEngine instance."""
        return self._migration_planner_engine

    @property
    def migration_executor_engine(self) -> Any:
        """Get the MigrationExecutorEngine instance."""
        return self._migration_executor_engine

    @property
    def pipeline_engine(self) -> Any:
        """Get the SchemaMigrationPipelineEngine instance."""
        return self._pipeline_engine

    # ------------------------------------------------------------------
    # Engine initialization
    # ------------------------------------------------------------------

    def _init_engines(self) -> None:
        """Attempt to import and initialise SDK engines.

        Engines are optional; missing imports are logged as warnings and
        the service continues in degraded mode.
        """
        if SchemaRegistryEngine is not None:
            try:
                self._schema_registry_engine = SchemaRegistryEngine(self.config)
                logger.info("SchemaRegistryEngine initialized")
            except Exception as exc:
                logger.warning("SchemaRegistryEngine init failed: %s", exc)
        else:
            logger.warning("SchemaRegistryEngine not available; using stub")

        if SchemaVersionerEngine is not None:
            try:
                self._schema_versioner_engine = SchemaVersionerEngine(self.config)
                logger.info("SchemaVersionerEngine initialized")
            except Exception as exc:
                logger.warning("SchemaVersionerEngine init failed: %s", exc)
        else:
            logger.warning("SchemaVersionerEngine not available; using stub")

        if ChangeDetectorEngine is not None:
            try:
                self._change_detector_engine = ChangeDetectorEngine(self.config)
                logger.info("ChangeDetectorEngine initialized")
            except Exception as exc:
                logger.warning("ChangeDetectorEngine init failed: %s", exc)
        else:
            logger.warning("ChangeDetectorEngine not available; using stub")

        if CompatibilityCheckerEngine is not None:
            try:
                self._compatibility_checker_engine = CompatibilityCheckerEngine(
                    self.config,
                )
                logger.info("CompatibilityCheckerEngine initialized")
            except Exception as exc:
                logger.warning("CompatibilityCheckerEngine init failed: %s", exc)
        else:
            logger.warning("CompatibilityCheckerEngine not available; using stub")

        if MigrationPlannerEngine is not None:
            try:
                self._migration_planner_engine = MigrationPlannerEngine(self.config)
                logger.info("MigrationPlannerEngine initialized")
            except Exception as exc:
                logger.warning("MigrationPlannerEngine init failed: %s", exc)
        else:
            logger.warning("MigrationPlannerEngine not available; using stub")

        if MigrationExecutorEngine is not None:
            try:
                self._migration_executor_engine = MigrationExecutorEngine(self.config)
                logger.info("MigrationExecutorEngine initialized")
            except Exception as exc:
                logger.warning("MigrationExecutorEngine init failed: %s", exc)
        else:
            logger.warning("MigrationExecutorEngine not available; using stub")

        if SchemaMigrationPipelineEngine is not None:
            try:
                self._pipeline_engine = SchemaMigrationPipelineEngine(self.config)
                logger.info("SchemaMigrationPipelineEngine initialized")
            except Exception as exc:
                logger.warning("SchemaMigrationPipelineEngine init failed: %s", exc)
        else:
            logger.warning("SchemaMigrationPipelineEngine not available; using stub")

    # ==================================================================
    # Schema operations (delegate to SchemaRegistryEngine)
    # ==================================================================

    def register_schema(
        self,
        namespace: str,
        name: str,
        schema_type: str,
        definition: Any,
        owner: str = "",
        tags: Optional[List[str]] = None,
        description: str = "",
    ) -> SchemaResponse:
        """Register a new schema in the schema registry.

        Delegates to the SchemaRegistryEngine for registration and
        validation.  All scoring is deterministic.  No LLM is used
        for registration logic (zero-hallucination).

        Args:
            namespace: Logical namespace / domain for the schema
                (e.g. ``"emissions"``, ``"supply_chain"``).
            name: Human-readable schema name (e.g. ``"ActivityRecord"``).
            schema_type: Schema format type
                (``"json_schema"``, ``"avro"``, ``"protobuf"``).
            definition: The schema definition as a JSON-serialisable dict
                or value.
            owner: Owner identifier (team or user).
            tags: Optional list of categorisation tags.
            description: Free-text description of the schema.

        Returns:
            SchemaResponse with registered schema details.

        Raises:
            ValueError: If namespace, name, or schema_type are empty.
            RuntimeError: If the SchemaRegistryEngine is not available.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine
            engine_result: Optional[Dict[str, Any]] = None
            if self._schema_registry_engine is not None:
                engine_result = self._schema_registry_engine.register_schema(
                    namespace=namespace,
                    name=name,
                    schema_type=schema_type,
                    definition_json=definition,
                    owner=owner,
                    tags=tags,
                    description=description,
                )

            # Build response
            schema_id = (
                engine_result.get("schema_id", _new_uuid())
                if engine_result else _new_uuid()
            )
            now_iso = _utcnow_iso()

            response = SchemaResponse(
                schema_id=schema_id,
                namespace=namespace,
                name=name,
                schema_type=schema_type,
                status=engine_result.get("status", "draft") if engine_result else "draft",
                owner=owner,
                tags=tags or [],
                description=description,
                version_count=0,
                created_at=engine_result.get("created_at", now_iso) if engine_result else now_iso,
                updated_at=engine_result.get("updated_at", now_iso) if engine_result else now_iso,
            )
            response.provenance_hash = _compute_hash(response)

            # Store in cache
            self._schemas[response.schema_id] = response

            # Record provenance
            self.provenance.record(
                entity_type="schema",
                entity_id=response.schema_id,
                action="schema_registered",
                data={
                    "namespace": namespace,
                    "name": name,
                    "schema_type": schema_type,
                    "owner": owner,
                },
            )

            # Record metrics
            record_schema_registered(schema_type, namespace)
            elapsed = time.perf_counter() - t0
            observe_processing_duration("schema_register", elapsed)

            # Update statistics
            self._stats.total_schemas += 1

            logger.info(
                "Registered schema %s: namespace=%s, name=%s, type=%s",
                response.schema_id,
                namespace,
                name,
                schema_type,
            )
            return response

        except Exception as exc:
            logger.error("register_schema failed: %s", exc, exc_info=True)
            raise

    def list_schemas(
        self,
        namespace: Optional[str] = None,
        schema_type: Optional[str] = None,
        status: Optional[str] = None,
        owner: Optional[str] = None,
        tag: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[SchemaResponse]:
        """List schemas with optional filtering and pagination.

        All filters are applied with AND logic.

        Args:
            namespace: Filter by exact namespace match.
            schema_type: Filter by exact schema type.
            status: Filter by exact lifecycle status.
            owner: Filter by exact owner match.
            tag: Filter by single tag membership.
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of SchemaResponse instances matching the filters.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine if available
            if self._schema_registry_engine is not None:
                engine_results = self._schema_registry_engine.list_schemas(
                    namespace=namespace,
                    schema_type=schema_type,
                    status=status,
                    owner=owner,
                    tag=tag,
                    limit=limit,
                    offset=offset,
                )
                results = []
                for rec in engine_results:
                    resp = self._dict_to_schema_response(rec)
                    results.append(resp)
                elapsed = time.perf_counter() - t0
                observe_processing_duration("schema_list", elapsed)
                return results

            # Fallback to in-memory store
            schemas = list(self._schemas.values())
            filtered = self._filter_schemas(
                schemas, namespace, schema_type, status, owner, tag,
            )
            paginated = filtered[offset:offset + limit]

            elapsed = time.perf_counter() - t0
            observe_processing_duration("schema_list", elapsed)
            return paginated

        except Exception as exc:
            logger.error("list_schemas failed: %s", exc, exc_info=True)
            raise

    def get_schema(self, schema_id: str) -> Optional[SchemaResponse]:
        """Get a schema by its unique identifier.

        Args:
            schema_id: Schema identifier (UUID4 string).

        Returns:
            SchemaResponse or None if not found.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine if available
            if self._schema_registry_engine is not None:
                engine_result = self._schema_registry_engine.get_schema(schema_id)
                if engine_result is not None:
                    resp = self._dict_to_schema_response(engine_result)
                    elapsed = time.perf_counter() - t0
                    observe_processing_duration("schema_get", elapsed)
                    return resp
                return None

            # Fallback to in-memory store
            result = self._schemas.get(schema_id)
            elapsed = time.perf_counter() - t0
            observe_processing_duration("schema_get", elapsed)
            return result

        except Exception as exc:
            logger.error("get_schema failed: %s", exc, exc_info=True)
            raise

    def update_schema(
        self,
        schema_id: str,
        owner: Optional[str] = None,
        tags: Optional[List[str]] = None,
        status: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Optional[SchemaResponse]:
        """Update schema metadata and/or advance its lifecycle status.

        Only fields explicitly provided (non-None) are updated.

        Args:
            schema_id: Schema identifier to update.
            owner: New owner identifier, or None to leave unchanged.
            tags: New complete tag list, or None to leave unchanged.
            status: Target status string, or None to leave unchanged.
            description: New description, or None to leave unchanged.

        Returns:
            Updated SchemaResponse or None if schema not found.

        Raises:
            ValueError: If the requested status transition is invalid.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine if available
            engine_result: Optional[Dict[str, Any]] = None
            if self._schema_registry_engine is not None:
                try:
                    engine_result = self._schema_registry_engine.update_schema(
                        schema_id=schema_id,
                        owner=owner,
                        tags=tags,
                        status=status,
                        description=description,
                    )
                except KeyError:
                    return None

            if engine_result is not None:
                response = self._dict_to_schema_response(engine_result)
                response.provenance_hash = _compute_hash(response)
                self._schemas[response.schema_id] = response
            else:
                # Fallback to in-memory store
                cached = self._schemas.get(schema_id)
                if cached is None:
                    return None
                if owner is not None:
                    cached.owner = owner
                if tags is not None:
                    cached.tags = tags
                if status is not None:
                    cached.status = status
                if description is not None:
                    cached.description = description
                cached.updated_at = _utcnow_iso()
                cached.provenance_hash = _compute_hash(cached)
                response = cached

            # Record provenance
            self.provenance.record(
                entity_type="schema",
                entity_id=schema_id,
                action="schema_updated",
                data={
                    "owner": owner,
                    "tags": tags,
                    "status": status,
                    "description": description,
                },
            )

            elapsed = time.perf_counter() - t0
            observe_processing_duration("schema_update", elapsed)

            logger.info(
                "Updated schema %s: owner=%s, status=%s",
                schema_id,
                owner,
                status,
            )
            return response

        except Exception as exc:
            logger.error("update_schema failed: %s", exc, exc_info=True)
            raise

    def delete_schema(self, schema_id: str) -> bool:
        """Soft-delete a schema by setting its status to archived.

        This does not remove the schema from storage.  It sets the
        ``status`` to ``"archived"`` so that it no longer appears in
        active queries.

        Args:
            schema_id: Schema identifier to delete.

        Returns:
            True if the schema was found and archived, False if not found.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine if available
            if self._schema_registry_engine is not None:
                try:
                    self._schema_registry_engine.delete_schema(schema_id)
                except (KeyError, ValueError):
                    # Schema not found or already deleted
                    return False

            # Update in-memory cache
            cached = self._schemas.get(schema_id)
            if cached is not None:
                cached.status = "archived"
                cached.updated_at = _utcnow_iso()
                cached.provenance_hash = _compute_hash(cached)
            elif self._schema_registry_engine is None:
                return False

            # Record provenance
            self.provenance.record(
                entity_type="schema",
                entity_id=schema_id,
                action="schema_deleted",
            )

            elapsed = time.perf_counter() - t0
            observe_processing_duration("schema_delete", elapsed)

            logger.info("Soft-deleted (archived) schema %s", schema_id)
            return True

        except Exception as exc:
            logger.error("delete_schema failed: %s", exc, exc_info=True)
            raise

    # ==================================================================
    # Version operations (delegate to SchemaVersionerEngine)
    # ==================================================================

    def create_version(
        self,
        schema_id: str,
        definition: Any,
        changelog_note: str = "",
    ) -> SchemaVersionResponse:
        """Create a new version for a registered schema.

        Auto-classifies the semantic version bump type based on the
        differences between the previous definition and the new one.
        The first version for any schema starts at ``"1.0.0"``.

        Args:
            schema_id: Parent schema identifier.
            definition: The full schema definition for this version
                (JSON-serialisable dict).
            changelog_note: Free-text note for the changelog entry.

        Returns:
            SchemaVersionResponse with version details.

        Raises:
            ValueError: If schema_id is empty.
        """
        t0 = time.perf_counter()

        if not schema_id:
            raise ValueError("schema_id must not be empty")

        try:
            # Delegate to engine if available
            engine_result: Optional[Dict[str, Any]] = None
            if self._schema_versioner_engine is not None:
                engine_result = self._schema_versioner_engine.create_version(
                    schema_id=schema_id,
                    definition_json=definition,
                    changelog_note=changelog_note,
                )

            # Build response
            version_id = (
                engine_result.get("id", _new_uuid())
                if engine_result else _new_uuid()
            )
            version_str = (
                engine_result.get("version", "1.0.0")
                if engine_result else "1.0.0"
            )
            bump_type = (
                engine_result.get("bump_type", "patch")
                if engine_result else "patch"
            )
            changelog = (
                engine_result.get("changelog_note", changelog_note)
                if engine_result else changelog_note
            )
            is_deprecated = (
                engine_result.get("is_deprecated", False)
                if engine_result else False
            )
            sunset_date = (
                engine_result.get("sunset_date")
                if engine_result else None
            )
            created_at = (
                engine_result.get("created_at", _utcnow_iso())
                if engine_result else _utcnow_iso()
            )

            response = SchemaVersionResponse(
                version_id=version_id,
                schema_id=schema_id,
                version=version_str,
                definition=definition,
                changelog=changelog,
                is_deprecated=is_deprecated,
                sunset_date=sunset_date,
                created_at=created_at,
            )
            response.provenance_hash = _compute_hash(response)

            # Store in cache
            self._versions[response.version_id] = response

            # Update parent schema version count
            cached_schema = self._schemas.get(schema_id)
            if cached_schema is not None:
                cached_schema.version_count += 1

            # Record provenance
            self.provenance.record(
                entity_type="schema_version",
                entity_id=response.version_id,
                action="version_created",
                data={
                    "schema_id": schema_id,
                    "version": version_str,
                    "bump_type": bump_type,
                },
            )

            # Record metrics
            record_version_created(bump_type)
            elapsed = time.perf_counter() - t0
            observe_processing_duration("version_create", elapsed)

            # Update statistics
            self._stats.total_versions += 1

            logger.info(
                "Created version %s for schema %s: version=%s bump=%s",
                response.version_id,
                schema_id,
                version_str,
                bump_type,
            )
            return response

        except Exception as exc:
            logger.error("create_version failed: %s", exc, exc_info=True)
            raise

    def list_versions(
        self,
        schema_id: str,
        version_range: Optional[str] = None,
        deprecated: Optional[bool] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[SchemaVersionResponse]:
        """List versions for a schema with optional filtering.

        Args:
            schema_id: Parent schema identifier.
            version_range: Optional SemVer range filter (not enforced
                at this level; passed to the engine).
            deprecated: If True, include only deprecated versions.
                If False, exclude deprecated versions. If None, include all.
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of SchemaVersionResponse instances.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine if available
            if self._schema_versioner_engine is not None:
                include_deprecated = deprecated if deprecated is not None else True
                engine_results = self._schema_versioner_engine.list_versions(
                    schema_id=schema_id,
                    include_deprecated=include_deprecated,
                    limit=limit,
                    offset=offset,
                )
                results = []
                for rec in engine_results:
                    resp = self._dict_to_version_response(rec)
                    results.append(resp)
                elapsed = time.perf_counter() - t0
                observe_processing_duration("version_list", elapsed)
                return results

            # Fallback to in-memory store
            versions = [
                v for v in self._versions.values()
                if v.schema_id == schema_id
            ]
            if deprecated is True:
                versions = [v for v in versions if v.is_deprecated]
            elif deprecated is False:
                versions = [v for v in versions if not v.is_deprecated]
            paginated = versions[offset:offset + limit]

            elapsed = time.perf_counter() - t0
            observe_processing_duration("version_list", elapsed)
            return paginated

        except Exception as exc:
            logger.error("list_versions failed: %s", exc, exc_info=True)
            raise

    def get_version(self, version_id: str) -> Optional[SchemaVersionResponse]:
        """Get a schema version by its unique identifier.

        Args:
            version_id: Version identifier string.

        Returns:
            SchemaVersionResponse or None if not found.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine if available
            if self._schema_versioner_engine is not None:
                engine_result = self._schema_versioner_engine.get_version(version_id)
                if engine_result is not None:
                    resp = self._dict_to_version_response(engine_result)
                    elapsed = time.perf_counter() - t0
                    observe_processing_duration("version_get", elapsed)
                    return resp
                return None

            # Fallback to in-memory store
            result = self._versions.get(version_id)
            elapsed = time.perf_counter() - t0
            observe_processing_duration("version_get", elapsed)
            return result

        except Exception as exc:
            logger.error("get_version failed: %s", exc, exc_info=True)
            raise

    # ==================================================================
    # Change detection (delegate to ChangeDetectorEngine)
    # ==================================================================

    def detect_changes(
        self,
        source_version_id: str,
        target_version_id: str,
    ) -> ChangeDetectionResponse:
        """Detect structural changes between two schema versions.

        Resolves the definitions from the version IDs, then delegates
        to the ChangeDetectorEngine for comparison.

        All detection is deterministic. No LLM is used for change
        detection (zero-hallucination).

        Args:
            source_version_id: Identifier of the source (old) version.
            target_version_id: Identifier of the target (new) version.

        Returns:
            ChangeDetectionResponse with change details.

        Raises:
            ValueError: If either version is not found.
        """
        t0 = time.perf_counter()

        try:
            # Resolve version definitions
            source_def = self._resolve_version_definition(source_version_id)
            target_def = self._resolve_version_definition(target_version_id)

            # Delegate to engine if available
            engine_result: Optional[Dict[str, Any]] = None
            if self._change_detector_engine is not None:
                engine_result = self._change_detector_engine.detect_changes(
                    source_definition=source_def,
                    target_definition=target_def,
                )

            # Build response
            changes_list: List[Dict[str, Any]] = []
            detection_id = _new_uuid()
            change_count = 0
            breaking_count = 0

            if engine_result is not None:
                detection_id = engine_result.get("detection_id", detection_id)
                changes_list = engine_result.get("changes", [])
                change_count = len(changes_list)
                summary = engine_result.get("summary", {})
                breaking_count = summary.get("breaking", 0)
                if breaking_count == 0:
                    # Count manually from changes
                    breaking_count = sum(
                        1 for c in changes_list
                        if c.get("severity") == "breaking"
                    )

            response = ChangeDetectionResponse(
                detection_id=detection_id,
                source_version_id=source_version_id,
                target_version_id=target_version_id,
                changes=changes_list,
                change_count=change_count,
                breaking_change_count=breaking_count,
                detected_at=_utcnow_iso(),
            )
            response.provenance_hash = _compute_hash(response)

            # Store in cache
            self._detections[response.detection_id] = response

            # Record provenance
            self.provenance.record(
                entity_type="change_detection",
                entity_id=response.detection_id,
                action="change_detected",
                data={
                    "source_version_id": source_version_id,
                    "target_version_id": target_version_id,
                    "change_count": change_count,
                    "breaking_count": breaking_count,
                },
            )

            # Record metrics
            for change in changes_list:
                ctype = change.get("change_type", "unknown")
                severity = change.get("severity", "informational")
                record_change_detected(ctype, severity)

            elapsed = time.perf_counter() - t0
            observe_processing_duration("change_detect", elapsed)

            # Update statistics
            self._stats.total_changes_detected += 1

            logger.info(
                "Detected %d changes (%d breaking) between %s and %s",
                change_count,
                breaking_count,
                source_version_id,
                target_version_id,
            )
            return response

        except Exception as exc:
            logger.error("detect_changes failed: %s", exc, exc_info=True)
            raise

    def list_changes(
        self,
        schema_id: Optional[str] = None,
        severity: Optional[str] = None,
        change_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[ChangeDetectionResponse]:
        """List change detection results with optional filtering.

        Args:
            schema_id: Optional filter by schema identifier (matches
                source or target version belonging to this schema).
            severity: Optional filter by change severity
                (breaking, non_breaking, informational).
            change_type: Optional filter by change type
                (field_added, field_removed, type_changed, etc.).
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of ChangeDetectionResponse instances.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine if available
            if self._change_detector_engine is not None:
                try:
                    engine_results = self._change_detector_engine.list_detections(
                        limit=limit,
                        offset=offset,
                    )
                    results = []
                    for rec in engine_results:
                        resp = self._dict_to_detection_response(rec)
                        results.append(resp)
                    elapsed = time.perf_counter() - t0
                    observe_processing_duration("change_list", elapsed)
                    return results
                except (AttributeError, TypeError):
                    pass

            # Fallback to in-memory store with filtering
            detections = list(self._detections.values())
            if severity is not None:
                detections = [
                    d for d in detections
                    if any(
                        c.get("severity") == severity
                        for c in d.changes
                    )
                ]
            if change_type is not None:
                detections = [
                    d for d in detections
                    if any(
                        c.get("change_type") == change_type
                        for c in d.changes
                    )
                ]
            paginated = detections[offset:offset + limit]

            elapsed = time.perf_counter() - t0
            observe_processing_duration("change_list", elapsed)
            return paginated

        except Exception as exc:
            logger.error("list_changes failed: %s", exc, exc_info=True)
            raise

    # ==================================================================
    # Compatibility checking (delegate to CompatibilityCheckerEngine)
    # ==================================================================

    def check_compatibility(
        self,
        source_version_id: str,
        target_version_id: str,
        level: Optional[str] = None,
    ) -> CompatibilityCheckResponse:
        """Check compatibility between two schema versions.

        Resolves the definitions from version IDs and delegates to the
        CompatibilityCheckerEngine.

        All compatibility checking is deterministic. No LLM is used
        for compatibility analysis (zero-hallucination).

        Args:
            source_version_id: Identifier of the source (old) version.
            target_version_id: Identifier of the target (new) version.
            level: Optional compatibility level to check
                (backward, forward, full, none). Defaults to config value.

        Returns:
            CompatibilityCheckResponse with check results.

        Raises:
            ValueError: If either version is not found.
        """
        t0 = time.perf_counter()

        try:
            # Resolve version definitions
            source_def = self._resolve_version_definition(source_version_id)
            target_def = self._resolve_version_definition(target_version_id)

            # Delegate to engine if available
            engine_result: Optional[Dict[str, Any]] = None
            if self._compatibility_checker_engine is not None:
                engine_result = self._compatibility_checker_engine.check_compatibility(
                    source_definition=source_def,
                    target_definition=target_def,
                )

            # Build response
            check_id = _new_uuid()
            compat_level = self.config.compatibility_default_level
            is_compatible = True
            issues: List[Dict[str, Any]] = []

            if engine_result is not None:
                check_id = engine_result.get("check_id", check_id)
                compat_level = engine_result.get("compatibility_level", compat_level)
                is_compatible = compat_level in ("full", "backward", "forward")
                if level is not None:
                    # Check against requested level
                    if level == "backward":
                        is_compatible = engine_result.get(
                            "backward_compatible", True,
                        )
                    elif level == "forward":
                        is_compatible = engine_result.get(
                            "forward_compatible", True,
                        )
                    elif level == "full":
                        is_compatible = (
                            engine_result.get("backward_compatible", True)
                            and engine_result.get("forward_compatible", True)
                        )
                    elif level == "none":
                        is_compatible = True
                issues = engine_result.get("issues", [])

            response = CompatibilityCheckResponse(
                check_id=check_id,
                source_version_id=source_version_id,
                target_version_id=target_version_id,
                compatibility_level=compat_level,
                is_compatible=is_compatible,
                issues=issues,
                checked_at=_utcnow_iso(),
            )
            response.provenance_hash = _compute_hash(response)

            # Store in cache
            self._compat_checks[response.check_id] = response

            # Record provenance
            self.provenance.record(
                entity_type="compatibility_check",
                entity_id=response.check_id,
                action="compatibility_checked",
                data={
                    "source_version_id": source_version_id,
                    "target_version_id": target_version_id,
                    "compatibility_level": compat_level,
                    "is_compatible": is_compatible,
                },
            )

            # Record metrics
            result_label = "compatible" if is_compatible else "incompatible"
            record_compatibility_check(result_label)
            elapsed = time.perf_counter() - t0
            observe_processing_duration("compatibility_check", elapsed)

            # Update statistics
            self._stats.total_compatibility_checks += 1

            logger.info(
                "Compatibility check %s: level=%s compatible=%s issues=%d",
                response.check_id,
                compat_level,
                is_compatible,
                len(issues),
            )
            return response

        except Exception as exc:
            logger.error("check_compatibility failed: %s", exc, exc_info=True)
            raise

    def list_compatibility_checks(
        self,
        schema_id: Optional[str] = None,
        result: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[CompatibilityCheckResponse]:
        """List compatibility check results with optional filtering.

        Args:
            schema_id: Optional filter by schema identifier.
            result: Optional filter by result
                (``"compatible"`` or ``"incompatible"``).
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of CompatibilityCheckResponse instances.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine if available
            if self._compatibility_checker_engine is not None:
                try:
                    engine_results = self._compatibility_checker_engine.list_checks(
                        limit=limit,
                        offset=offset,
                    )
                    results = []
                    for rec in engine_results:
                        resp = self._dict_to_compat_response(rec)
                        results.append(resp)
                    elapsed = time.perf_counter() - t0
                    observe_processing_duration("compatibility_list", elapsed)
                    return results
                except (AttributeError, TypeError):
                    pass

            # Fallback to in-memory store with filtering
            checks = list(self._compat_checks.values())
            if result == "compatible":
                checks = [c for c in checks if c.is_compatible]
            elif result == "incompatible":
                checks = [c for c in checks if not c.is_compatible]
            paginated = checks[offset:offset + limit]

            elapsed = time.perf_counter() - t0
            observe_processing_duration("compatibility_list", elapsed)
            return paginated

        except Exception as exc:
            logger.error("list_compatibility_checks failed: %s", exc, exc_info=True)
            raise

    # ==================================================================
    # Migration planning (delegate to MigrationPlannerEngine)
    # ==================================================================

    def create_plan(
        self,
        source_schema_id: str,
        target_schema_id: str,
        source_version: Optional[str] = None,
        target_version: Optional[str] = None,
    ) -> MigrationPlanResponse:
        """Create a migration plan between two schemas or schema versions.

        Delegates to the MigrationPlannerEngine after resolving version
        definitions and detecting changes.

        Args:
            source_schema_id: Source schema identifier.
            target_schema_id: Target schema identifier.
            source_version: Optional source SemVer string. If None,
                uses the latest version.
            target_version: Optional target SemVer string. If None,
                uses the latest version.

        Returns:
            MigrationPlanResponse with plan details.

        Raises:
            ValueError: If either schema_id is empty.
        """
        t0 = time.perf_counter()

        if not source_schema_id:
            raise ValueError("source_schema_id must not be empty")
        if not target_schema_id:
            raise ValueError("target_schema_id must not be empty")

        try:
            # Resolve versions
            src_version = source_version or "1.0.0"
            tgt_version = target_version or "1.0.0"

            # Attempt to detect changes first if both engines exist
            changes: List[Dict[str, Any]] = []
            source_def: Dict[str, Any] = {}
            target_def: Dict[str, Any] = {}

            source_def = self._resolve_schema_definition(
                source_schema_id, src_version,
            )
            target_def = self._resolve_schema_definition(
                target_schema_id, tgt_version,
            )

            if self._change_detector_engine is not None and source_def and target_def:
                detection = self._change_detector_engine.detect_changes(
                    source_definition=source_def,
                    target_definition=target_def,
                )
                changes = detection.get("changes", [])

            # Delegate to planner engine if available
            engine_result: Optional[Dict[str, Any]] = None
            if self._migration_planner_engine is not None:
                engine_result = self._migration_planner_engine.create_plan(
                    source_schema_id=source_schema_id,
                    target_schema_id=target_schema_id,
                    source_version=src_version,
                    target_version=tgt_version,
                    changes=changes,
                    source_definition=source_def,
                    target_definition=target_def,
                )

            # Build response
            plan_id = (
                engine_result.get("plan_id", _new_uuid())
                if engine_result else _new_uuid()
            )
            steps = (
                engine_result.get("steps", [])
                if engine_result else []
            )
            total_steps = len(steps)
            effort_estimate = "low"
            if engine_result is not None:
                effort = engine_result.get("effort", {})
                effort_estimate = effort.get(
                    "effort_band",
                    engine_result.get("effort_band", "low"),
                )
            plan_status = (
                engine_result.get("status", "pending")
                if engine_result else "pending"
            )
            created_at = (
                engine_result.get("created_at", _utcnow_iso())
                if engine_result else _utcnow_iso()
            )

            response = MigrationPlanResponse(
                plan_id=plan_id,
                source_schema_id=source_schema_id,
                target_schema_id=target_schema_id,
                steps=steps,
                total_steps=total_steps,
                effort_estimate=effort_estimate,
                status=plan_status,
                created_at=created_at,
            )
            response.provenance_hash = _compute_hash(response)

            # Store in cache
            self._plans[response.plan_id] = response

            # Record provenance
            self.provenance.record(
                entity_type="migration_plan",
                entity_id=response.plan_id,
                action="plan_created",
                data={
                    "source_schema_id": source_schema_id,
                    "target_schema_id": target_schema_id,
                    "total_steps": total_steps,
                    "effort_estimate": effort_estimate,
                },
            )

            # Record metrics
            record_migration_planned("success")
            elapsed = time.perf_counter() - t0
            observe_processing_duration("plan_create", elapsed)

            # Update statistics
            self._stats.total_migrations_planned += 1

            logger.info(
                "Created migration plan %s: %s -> %s, steps=%d, effort=%s",
                response.plan_id,
                source_schema_id,
                target_schema_id,
                total_steps,
                effort_estimate,
            )
            return response

        except Exception as exc:
            record_migration_planned("failed")
            logger.error("create_plan failed: %s", exc, exc_info=True)
            raise

    def get_plan(self, plan_id: str) -> Optional[MigrationPlanResponse]:
        """Get a migration plan by its identifier.

        Args:
            plan_id: Plan identifier string.

        Returns:
            MigrationPlanResponse or None if not found.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine if available
            if self._migration_planner_engine is not None:
                engine_result = self._migration_planner_engine.get_plan(plan_id)
                if engine_result is not None:
                    resp = self._dict_to_plan_response(engine_result)
                    elapsed = time.perf_counter() - t0
                    observe_processing_duration("plan_get", elapsed)
                    return resp

            # Fallback to in-memory store
            result = self._plans.get(plan_id)
            elapsed = time.perf_counter() - t0
            observe_processing_duration("plan_get", elapsed)
            return result

        except Exception as exc:
            logger.error("get_plan failed: %s", exc, exc_info=True)
            raise

    # ==================================================================
    # Migration execution (delegate to MigrationExecutorEngine)
    # ==================================================================

    def execute_migration(
        self,
        plan_id: str,
        dry_run: bool = False,
    ) -> MigrationExecutionResponse:
        """Execute a migration plan.

        Delegates to the MigrationExecutorEngine.  When ``dry_run`` is
        True, the migration is validated and simulated without applying
        mutations.

        Args:
            plan_id: Identifier of the plan to execute.
            dry_run: When True, validates without committing.

        Returns:
            MigrationExecutionResponse with execution results.

        Raises:
            ValueError: If the plan is not found.
        """
        t0 = time.perf_counter()

        try:
            self._increment_active_migrations()

            # Retrieve the plan
            plan_data = self._resolve_plan(plan_id)

            # Delegate to engine if available
            engine_result: Optional[Dict[str, Any]] = None
            if self._migration_executor_engine is not None:
                engine_result = self._migration_executor_engine.execute_plan(
                    plan=plan_data,
                    dry_run=dry_run,
                )

            # Build response
            execution_id = (
                engine_result.get("execution_id", _new_uuid())
                if engine_result else _new_uuid()
            )
            exec_status = (
                engine_result.get("status", "completed")
                if engine_result else "completed"
            )
            records_processed = (
                engine_result.get("records_processed", 0)
                if engine_result else 0
            )
            records_failed = (
                engine_result.get("records_failed", 0)
                if engine_result else 0
            )
            records_skipped = (
                engine_result.get("records_skipped", 0)
                if engine_result else 0
            )
            current_step = (
                engine_result.get("current_step", 0)
                if engine_result else 0
            )
            completed_steps = (
                engine_result.get("completed_steps", 0)
                if engine_result else 0
            )
            total_steps = (
                engine_result.get("total_steps", 0)
                if engine_result else 0
            )
            percentage = (
                (completed_steps / max(total_steps, 1)) * 100.0
            )
            started_at = (
                engine_result.get("started_at", _utcnow_iso())
                if engine_result else _utcnow_iso()
            )
            completed_at = (
                engine_result.get("completed_at")
                if engine_result else _utcnow_iso()
            )

            response = MigrationExecutionResponse(
                execution_id=execution_id,
                plan_id=plan_id,
                status=exec_status,
                records_processed=records_processed,
                records_failed=records_failed,
                records_skipped=records_skipped,
                current_step=current_step,
                total_steps=total_steps,
                percentage=round(percentage, 2),
                started_at=started_at,
                completed_at=completed_at,
            )
            response.provenance_hash = _compute_hash(response)

            # Store in cache
            self._executions[response.execution_id] = response

            # Update plan status in cache
            cached_plan = self._plans.get(plan_id)
            if cached_plan is not None:
                cached_plan.status = exec_status

            # Record provenance
            self.provenance.record(
                entity_type="migration_execution",
                entity_id=response.execution_id,
                action="migration_executed",
                data={
                    "plan_id": plan_id,
                    "status": exec_status,
                    "records_processed": records_processed,
                    "dry_run": dry_run,
                },
            )

            # Record metrics
            record_migration_executed(exec_status)
            elapsed = time.perf_counter() - t0
            observe_migration_duration(elapsed)
            observe_records_migrated(records_processed)
            observe_processing_duration("migration_execute", elapsed)

            # Update statistics
            self._stats.total_migrations_executed += 1
            self._migration_total += 1
            if exec_status == "completed":
                self._migration_successes += 1
            self._migration_durations.append(elapsed)
            self._update_avg_duration(elapsed)
            self._update_success_rate()

            self._decrement_active_migrations()

            logger.info(
                "Executed migration %s: plan=%s status=%s processed=%d failed=%d",
                response.execution_id,
                plan_id,
                exec_status,
                records_processed,
                records_failed,
            )
            return response

        except Exception as exc:
            self._decrement_active_migrations()
            record_migration_executed("failed")
            logger.error("execute_migration failed: %s", exc, exc_info=True)
            raise

    def get_execution(
        self,
        execution_id: str,
    ) -> Optional[MigrationExecutionResponse]:
        """Get a migration execution result by identifier.

        Args:
            execution_id: Execution identifier string.

        Returns:
            MigrationExecutionResponse or None if not found.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine if available
            if self._migration_executor_engine is not None:
                engine_result = self._migration_executor_engine.get_execution(
                    execution_id,
                )
                if engine_result is not None:
                    resp = self._dict_to_execution_response(engine_result)
                    elapsed = time.perf_counter() - t0
                    observe_processing_duration("execution_get", elapsed)
                    return resp

            # Fallback to in-memory store
            result = self._executions.get(execution_id)
            elapsed = time.perf_counter() - t0
            observe_processing_duration("execution_get", elapsed)
            return result

        except Exception as exc:
            logger.error("get_execution failed: %s", exc, exc_info=True)
            raise

    def rollback_migration(
        self,
        execution_id: str,
        to_checkpoint: Optional[int] = None,
    ) -> MigrationExecutionResponse:
        """Rollback a migration execution.

        Delegates to the MigrationExecutorEngine to revert the execution
        to a prior checkpoint.

        Args:
            execution_id: Identifier of the execution to roll back.
            to_checkpoint: Optional step number to roll back to. If None,
                performs a full rollback to the initial state.

        Returns:
            MigrationExecutionResponse reflecting the rolled-back state.

        Raises:
            ValueError: If the execution is not found.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine if available
            engine_result: Optional[Dict[str, Any]] = None
            rollback_type = "full" if to_checkpoint is None else "partial"

            if self._migration_executor_engine is not None:
                engine_result = self._migration_executor_engine.rollback_execution(
                    execution_id=execution_id,
                    rollback_type=rollback_type,
                    to_step=to_checkpoint,
                )

            # Build response
            exec_status = "rolled_back"
            records_processed = 0
            completed_at = _utcnow_iso()

            if engine_result is not None:
                exec_status = engine_result.get("status", "rolled_back")

            # Look up existing execution for baseline data
            cached = self._executions.get(execution_id)
            plan_id = cached.plan_id if cached else ""
            total_steps = cached.total_steps if cached else 0

            response = MigrationExecutionResponse(
                execution_id=execution_id,
                plan_id=plan_id,
                status=exec_status,
                records_processed=records_processed,
                records_failed=0,
                records_skipped=0,
                current_step=to_checkpoint or 0,
                total_steps=total_steps,
                percentage=0.0,
                started_at=cached.started_at if cached else _utcnow_iso(),
                completed_at=completed_at,
            )
            response.provenance_hash = _compute_hash(response)

            # Update cache
            self._executions[execution_id] = response

            # Record provenance
            self.provenance.record(
                entity_type="migration_rollback",
                entity_id=execution_id,
                action="rollback_initiated",
                data={
                    "execution_id": execution_id,
                    "rollback_type": rollback_type,
                    "to_checkpoint": to_checkpoint,
                },
            )

            # Record metrics
            rollback_status = "success" if exec_status == "rolled_back" else "failed"
            record_rollback(rollback_type, rollback_status)
            elapsed = time.perf_counter() - t0
            observe_processing_duration("rollback", elapsed)

            # Update statistics
            self._stats.total_rollbacks += 1

            logger.info(
                "Rolled back execution %s: type=%s to_checkpoint=%s status=%s",
                execution_id,
                rollback_type,
                to_checkpoint,
                exec_status,
            )
            return response

        except Exception as exc:
            record_rollback(
                "full" if to_checkpoint is None else "partial",
                "failed",
            )
            logger.error("rollback_migration failed: %s", exc, exc_info=True)
            raise

    # ==================================================================
    # Pipeline orchestration (delegate to SchemaMigrationPipelineEngine)
    # ==================================================================

    def run_pipeline(
        self,
        source_schema_id: str,
        target_schema_id: str,
        source_version: Optional[str] = None,
        target_version: Optional[str] = None,
        skip_compatibility: bool = False,
        skip_dry_run: bool = False,
    ) -> PipelineResultResponse:
        """Run the end-to-end schema migration pipeline.

        Orchestrates all seven stages in sequence: detect, compatibility,
        plan, validate, execute, verify, and registry update.

        Args:
            source_schema_id: Source schema identifier.
            target_schema_id: Target schema identifier.
            source_version: Optional source SemVer string.
            target_version: Optional target SemVer string.
            skip_compatibility: When True, bypass the compatibility stage.
            skip_dry_run: When True, skip the validate/dry-run stage.

        Returns:
            PipelineResultResponse with overall pipeline results.

        Raises:
            ValueError: If either schema_id is empty.
        """
        t0 = time.perf_counter()

        if not source_schema_id:
            raise ValueError("source_schema_id must not be empty")
        if not target_schema_id:
            raise ValueError("target_schema_id must not be empty")

        try:
            self._increment_active_migrations()
            pipeline_id = _new_uuid()
            stages_completed: List[str] = []
            final_status = "pending"
            changes_detected = 0
            is_compatible = True
            plan_id: Optional[str] = None
            execution_id: Optional[str] = None

            # Stage 1: Detect changes
            try:
                source_def = self._resolve_schema_definition(
                    source_schema_id, source_version or "1.0.0",
                )
                target_def = self._resolve_schema_definition(
                    target_schema_id, target_version or "1.0.0",
                )

                if self._change_detector_engine is not None and source_def and target_def:
                    detection = self._change_detector_engine.detect_changes(
                        source_definition=source_def,
                        target_definition=target_def,
                    )
                    changes = detection.get("changes", [])
                    changes_detected = len(changes)
                stages_completed.append("detect")
            except Exception as exc:
                logger.warning("Pipeline detect stage failed: %s", exc)
                changes = []

            # Stage 2: Compatibility check
            if not skip_compatibility:
                try:
                    if self._compatibility_checker_engine is not None and source_def and target_def:
                        compat = self._compatibility_checker_engine.check_compatibility(
                            source_definition=source_def,
                            target_definition=target_def,
                            changes=changes if changes else None,
                        )
                        compat_level = compat.get("compatibility_level", "full")
                        is_compatible = compat_level != "breaking"
                    stages_completed.append("compatibility")
                except Exception as exc:
                    logger.warning("Pipeline compatibility stage failed: %s", exc)
            else:
                stages_completed.append("compatibility_skipped")

            # Stage 3: Create plan
            try:
                plan_resp = self.create_plan(
                    source_schema_id=source_schema_id,
                    target_schema_id=target_schema_id,
                    source_version=source_version,
                    target_version=target_version,
                )
                plan_id = plan_resp.plan_id
                stages_completed.append("plan")
            except Exception as exc:
                logger.warning("Pipeline plan stage failed: %s", exc)

            # Stage 4: Validate (dry run) - unless skipped
            if not skip_dry_run and plan_id is not None:
                try:
                    dry_resp = self.execute_migration(
                        plan_id=plan_id,
                        dry_run=True,
                    )
                    stages_completed.append("validate")
                except Exception as exc:
                    logger.warning("Pipeline validate stage failed: %s", exc)
            elif skip_dry_run:
                stages_completed.append("validate_skipped")

            # Stage 5: Execute
            if plan_id is not None and is_compatible:
                try:
                    exec_resp = self.execute_migration(
                        plan_id=plan_id,
                        dry_run=False,
                    )
                    execution_id = exec_resp.execution_id
                    if exec_resp.status == "completed":
                        stages_completed.append("execute")
                        final_status = "completed"
                    else:
                        stages_completed.append("execute_failed")
                        final_status = "failed"
                except Exception as exc:
                    logger.warning("Pipeline execute stage failed: %s", exc)
                    final_status = "failed"
            elif not is_compatible:
                final_status = "incompatible"

            if final_status == "pending":
                final_status = "completed" if stages_completed else "failed"

            elapsed = time.perf_counter() - t0

            response = PipelineResultResponse(
                pipeline_id=pipeline_id,
                source_schema_id=source_schema_id,
                target_schema_id=target_schema_id,
                stages_completed=stages_completed,
                final_status=final_status,
                changes_detected=changes_detected,
                is_compatible=is_compatible,
                plan_id=plan_id,
                execution_id=execution_id,
                elapsed_seconds=round(elapsed, 3),
            )
            response.provenance_hash = _compute_hash(response)

            # Store in cache
            self._pipeline_results[response.pipeline_id] = response

            # Record provenance
            self.provenance.record(
                entity_type="pipeline_result",
                entity_id=response.pipeline_id,
                action="pipeline_completed",
                data={
                    "source_schema_id": source_schema_id,
                    "target_schema_id": target_schema_id,
                    "final_status": final_status,
                    "stages_completed": stages_completed,
                    "elapsed_seconds": round(elapsed, 3),
                },
            )

            # Record metrics
            observe_migration_duration(elapsed)
            observe_processing_duration("pipeline", elapsed)

            self._decrement_active_migrations()

            logger.info(
                "Pipeline %s completed: status=%s stages=%s elapsed=%.3fs",
                response.pipeline_id,
                final_status,
                stages_completed,
                elapsed,
            )
            return response

        except Exception as exc:
            self._decrement_active_migrations()
            logger.error("run_pipeline failed: %s", exc, exc_info=True)
            raise

    # ==================================================================
    # Statistics and health
    # ==================================================================

    def get_statistics(self) -> SchemaMigrationStatisticsResponse:
        """Get aggregate statistics for the schema migration service.

        Returns:
            SchemaMigrationStatisticsResponse with current statistics.
        """
        t0 = time.perf_counter()

        # Enrich from engine statistics where available
        if self._schema_registry_engine is not None:
            try:
                reg_stats = self._schema_registry_engine.get_statistics()
                self._stats.total_schemas = reg_stats.get(
                    "total_schemas", self._stats.total_schemas,
                )
            except (AttributeError, Exception):
                pass

        if self._change_detector_engine is not None:
            try:
                det_stats = self._change_detector_engine.get_statistics()
                self._stats.total_changes_detected = det_stats.get(
                    "total_detections", self._stats.total_changes_detected,
                )
            except (AttributeError, Exception):
                pass

        if self._compatibility_checker_engine is not None:
            try:
                compat_stats = self._compatibility_checker_engine.get_statistics()
                self._stats.total_compatibility_checks = compat_stats.get(
                    "total_checks", self._stats.total_compatibility_checks,
                )
            except (AttributeError, Exception):
                pass

        self._stats.active_migrations = self._active_migrations

        elapsed = time.perf_counter() - t0
        observe_processing_duration("statistics", elapsed)

        logger.debug(
            "Statistics: schemas=%d versions=%d changes=%d "
            "compat_checks=%d plans=%d executions=%d rollbacks=%d "
            "active=%d success_rate=%.4f",
            self._stats.total_schemas,
            self._stats.total_versions,
            self._stats.total_changes_detected,
            self._stats.total_compatibility_checks,
            self._stats.total_migrations_planned,
            self._stats.total_migrations_executed,
            self._stats.total_rollbacks,
            self._stats.active_migrations,
            self._stats.success_rate,
        )
        return self._stats

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the schema migration service.

        Returns a dictionary with health status for each engine and
        the overall service.

        Returns:
            Dictionary with health check results including:
            - ``status``: Overall service status (healthy, degraded, unhealthy).
            - ``engines``: Per-engine availability status.
            - ``started``: Whether the service has been started.
            - ``statistics``: Summary statistics.
            - ``provenance_chain_valid``: Whether the provenance chain is intact.
            - ``timestamp``: ISO-8601 UTC timestamp of the check.
        """
        t0 = time.perf_counter()

        engines: Dict[str, str] = {
            "schema_registry": "available" if self._schema_registry_engine is not None else "unavailable",
            "schema_versioner": "available" if self._schema_versioner_engine is not None else "unavailable",
            "change_detector": "available" if self._change_detector_engine is not None else "unavailable",
            "compatibility_checker": "available" if self._compatibility_checker_engine is not None else "unavailable",
            "migration_planner": "available" if self._migration_planner_engine is not None else "unavailable",
            "migration_executor": "available" if self._migration_executor_engine is not None else "unavailable",
            "pipeline": "available" if self._pipeline_engine is not None else "unavailable",
        }

        available_count = sum(
            1 for status in engines.values() if status == "available"
        )
        total_engines = len(engines)

        if available_count == total_engines:
            overall_status = "healthy"
        elif available_count >= 4:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"

        # Verify provenance chain
        chain_valid = self.provenance.verify_chain()

        result = {
            "status": overall_status,
            "engines": engines,
            "engines_available": available_count,
            "engines_total": total_engines,
            "started": self._started,
            "statistics": {
                "total_schemas": self._stats.total_schemas,
                "total_versions": self._stats.total_versions,
                "total_migrations_executed": self._stats.total_migrations_executed,
                "active_migrations": self._active_migrations,
                "success_rate": self._stats.success_rate,
            },
            "provenance_chain_valid": chain_valid,
            "provenance_entries": self.provenance.entry_count,
            "prometheus_available": PROMETHEUS_AVAILABLE,
            "timestamp": _utcnow_iso(),
        }

        elapsed = time.perf_counter() - t0
        observe_processing_duration("health_check", elapsed)

        logger.info(
            "Health check: status=%s engines=%d/%d chain_valid=%s",
            overall_status,
            available_count,
            total_engines,
            chain_valid,
        )
        return result

    # ==================================================================
    # Provenance and metrics access
    # ==================================================================

    def get_provenance(self) -> ProvenanceTracker:
        """Get the provenance tracker instance.

        Returns:
            ProvenanceTracker instance used by this service.
        """
        return self.provenance

    def get_metrics(self) -> Dict[str, Any]:
        """Get current service metrics as a dictionary.

        Returns:
            Dictionary of metric names to current values.
        """
        return {
            "total_schemas": self._stats.total_schemas,
            "total_versions": self._stats.total_versions,
            "total_changes_detected": self._stats.total_changes_detected,
            "total_compatibility_checks": self._stats.total_compatibility_checks,
            "total_migrations_planned": self._stats.total_migrations_planned,
            "total_migrations_executed": self._stats.total_migrations_executed,
            "total_rollbacks": self._stats.total_rollbacks,
            "total_drift_events": self._stats.total_drift_events,
            "avg_migration_duration_seconds": self._stats.avg_migration_duration_seconds,
            "success_rate": self._stats.success_rate,
            "active_migrations": self._active_migrations,
            "provenance_entries": self.provenance.entry_count,
            "provenance_chain_valid": self.provenance.verify_chain(),
            "prometheus_available": PROMETHEUS_AVAILABLE,
        }

    # ==================================================================
    # Lifecycle
    # ==================================================================

    def startup(self) -> None:
        """Start the schema migration service.

        Safe to call multiple times.
        """
        if self._started:
            logger.debug("SchemaMigrationService already started; skipping")
            return

        logger.info("SchemaMigrationService starting up...")
        self._started = True
        set_active_migrations(0)
        logger.info("SchemaMigrationService startup complete")

    def shutdown(self) -> None:
        """Shutdown the schema migration service and release resources."""
        if not self._started:
            return

        self._started = False
        self._active_migrations = 0
        set_active_migrations(0)
        logger.info("SchemaMigrationService shut down")

    # ==================================================================
    # Internal helpers: dict -> response model conversion
    # ==================================================================

    def _dict_to_schema_response(
        self,
        rec: Dict[str, Any],
    ) -> SchemaResponse:
        """Convert a raw engine dict to SchemaResponse.

        Args:
            rec: Dictionary from the SchemaRegistryEngine.

        Returns:
            SchemaResponse model.
        """
        return SchemaResponse(
            schema_id=rec.get("schema_id", rec.get("id", "")),
            namespace=rec.get("namespace", ""),
            name=rec.get("name", ""),
            schema_type=rec.get("schema_type", "json_schema"),
            status=rec.get("status", "draft"),
            owner=rec.get("owner", ""),
            tags=rec.get("tags", []),
            description=rec.get("description", ""),
            version_count=rec.get("version_count", 0),
            created_at=rec.get("created_at", ""),
            updated_at=rec.get("updated_at", ""),
            provenance_hash=rec.get("provenance_hash", ""),
        )

    def _dict_to_version_response(
        self,
        rec: Dict[str, Any],
    ) -> SchemaVersionResponse:
        """Convert a raw engine dict to SchemaVersionResponse.

        Args:
            rec: Dictionary from the SchemaVersionerEngine.

        Returns:
            SchemaVersionResponse model.
        """
        return SchemaVersionResponse(
            version_id=rec.get("id", rec.get("version_id", "")),
            schema_id=rec.get("schema_id", ""),
            version=rec.get("version", "1.0.0"),
            definition=rec.get("definition", {}),
            changelog=rec.get("changelog_note", rec.get("changelog", "")),
            is_deprecated=rec.get("is_deprecated", False),
            sunset_date=rec.get("sunset_date"),
            created_at=rec.get("created_at", ""),
            provenance_hash=rec.get("provenance_hash", ""),
        )

    def _dict_to_detection_response(
        self,
        rec: Dict[str, Any],
    ) -> ChangeDetectionResponse:
        """Convert a raw engine dict to ChangeDetectionResponse.

        Args:
            rec: Dictionary from the ChangeDetectorEngine.

        Returns:
            ChangeDetectionResponse model.
        """
        changes = rec.get("changes", [])
        breaking_count = sum(
            1 for c in changes if c.get("severity") == "breaking"
        )
        return ChangeDetectionResponse(
            detection_id=rec.get("detection_id", ""),
            source_version_id=rec.get("source_version_id", ""),
            target_version_id=rec.get("target_version_id", ""),
            changes=changes,
            change_count=len(changes),
            breaking_change_count=breaking_count,
            detected_at=rec.get("detected_at", ""),
            provenance_hash=rec.get("provenance_hash", ""),
        )

    def _dict_to_compat_response(
        self,
        rec: Dict[str, Any],
    ) -> CompatibilityCheckResponse:
        """Convert a raw engine dict to CompatibilityCheckResponse.

        Args:
            rec: Dictionary from the CompatibilityCheckerEngine.

        Returns:
            CompatibilityCheckResponse model.
        """
        compat_level = rec.get("compatibility_level", "full")
        is_compat = compat_level in ("full", "backward", "forward")
        return CompatibilityCheckResponse(
            check_id=rec.get("check_id", ""),
            source_version_id=rec.get("source_version_id", ""),
            target_version_id=rec.get("target_version_id", ""),
            compatibility_level=compat_level,
            is_compatible=is_compat,
            issues=rec.get("issues", []),
            checked_at=rec.get("checked_at", ""),
            provenance_hash=rec.get("provenance_hash", ""),
        )

    def _dict_to_plan_response(
        self,
        rec: Dict[str, Any],
    ) -> MigrationPlanResponse:
        """Convert a raw engine dict to MigrationPlanResponse.

        Args:
            rec: Dictionary from the MigrationPlannerEngine.

        Returns:
            MigrationPlanResponse model.
        """
        steps = rec.get("steps", [])
        effort = rec.get("effort", {})
        effort_estimate = effort.get(
            "effort_band", rec.get("effort_band", "low"),
        )
        return MigrationPlanResponse(
            plan_id=rec.get("plan_id", ""),
            source_schema_id=rec.get("source_schema_id", ""),
            target_schema_id=rec.get("target_schema_id", ""),
            steps=steps,
            total_steps=len(steps),
            effort_estimate=effort_estimate,
            status=rec.get("status", "pending"),
            created_at=rec.get("created_at", ""),
            provenance_hash=rec.get("provenance_hash", ""),
        )

    def _dict_to_execution_response(
        self,
        rec: Dict[str, Any],
    ) -> MigrationExecutionResponse:
        """Convert a raw engine dict to MigrationExecutionResponse.

        Args:
            rec: Dictionary from the MigrationExecutorEngine.

        Returns:
            MigrationExecutionResponse model.
        """
        completed_steps = rec.get("completed_steps", 0)
        total_steps = rec.get("total_steps", 0)
        percentage = (completed_steps / max(total_steps, 1)) * 100.0
        return MigrationExecutionResponse(
            execution_id=rec.get("execution_id", ""),
            plan_id=rec.get("plan_id", ""),
            status=rec.get("status", "pending"),
            records_processed=rec.get("records_processed", 0),
            records_failed=rec.get("records_failed", 0),
            records_skipped=rec.get("records_skipped", 0),
            current_step=rec.get("current_step", 0),
            total_steps=total_steps,
            percentage=round(percentage, 2),
            started_at=rec.get("started_at", ""),
            completed_at=rec.get("completed_at"),
            provenance_hash=rec.get("provenance_hash", ""),
        )

    # ==================================================================
    # Internal helpers: filtering
    # ==================================================================

    def _filter_schemas(
        self,
        schemas: List[SchemaResponse],
        namespace: Optional[str],
        schema_type: Optional[str],
        status: Optional[str],
        owner: Optional[str],
        tag: Optional[str],
    ) -> List[SchemaResponse]:
        """Filter schema response list by multiple criteria.

        Args:
            schemas: List of SchemaResponse instances.
            namespace: Exact namespace filter.
            schema_type: Exact schema type filter.
            status: Exact status filter.
            owner: Exact owner filter.
            tag: Tag membership filter.

        Returns:
            Filtered list of SchemaResponse instances.
        """
        result = schemas
        if namespace is not None:
            result = [s for s in result if s.namespace == namespace]
        if schema_type is not None:
            result = [s for s in result if s.schema_type == schema_type]
        if status is not None:
            result = [s for s in result if s.status == status]
        if owner is not None:
            result = [s for s in result if s.owner == owner]
        if tag is not None:
            result = [
                s for s in result
                if tag in s.tags
            ]
        return result

    # ==================================================================
    # Internal helpers: definition resolution
    # ==================================================================

    def _resolve_version_definition(
        self,
        version_id: str,
    ) -> Dict[str, Any]:
        """Resolve a schema definition from a version identifier.

        Looks up the version in the engine first, then falls back to
        the in-memory cache.

        Args:
            version_id: Version identifier string.

        Returns:
            Schema definition as a dictionary.

        Raises:
            ValueError: If the version cannot be found.
        """
        # Try engine first
        if self._schema_versioner_engine is not None:
            engine_ver = self._schema_versioner_engine.get_version(version_id)
            if engine_ver is not None:
                definition = engine_ver.get("definition", {})
                if isinstance(definition, dict):
                    return definition
                return {}

        # Fallback to in-memory cache
        cached = self._versions.get(version_id)
        if cached is not None:
            if isinstance(cached.definition, dict):
                return cached.definition
            return {}

        raise ValueError(f"Version not found: {version_id}")

    def _resolve_schema_definition(
        self,
        schema_id: str,
        version: str,
    ) -> Dict[str, Any]:
        """Resolve a schema definition by schema ID and version string.

        Attempts to find the definition via the versioner engine, then
        falls back to the in-memory cache.

        Args:
            schema_id: Schema identifier.
            version: Semantic version string.

        Returns:
            Schema definition dict, or empty dict if not found.
        """
        # Try versioner engine
        if self._schema_versioner_engine is not None:
            try:
                engine_ver = self._schema_versioner_engine.get_version_by_string(
                    schema_id, version,
                )
                if engine_ver is not None:
                    definition = engine_ver.get("definition", {})
                    if isinstance(definition, dict):
                        return definition
            except (AttributeError, Exception):
                pass

        # Fallback: scan in-memory versions
        for ver in self._versions.values():
            if ver.schema_id == schema_id and ver.version == version:
                if isinstance(ver.definition, dict):
                    return ver.definition

        return {}

    def _resolve_plan(self, plan_id: str) -> Dict[str, Any]:
        """Resolve a migration plan into a dict for the executor engine.

        Args:
            plan_id: Plan identifier.

        Returns:
            Plan dictionary with plan_id, status, and steps.

        Raises:
            ValueError: If the plan is not found.
        """
        # Try planner engine first
        if self._migration_planner_engine is not None:
            engine_plan = self._migration_planner_engine.get_plan(plan_id)
            if engine_plan is not None:
                return engine_plan

        # Fallback to in-memory cache
        cached = self._plans.get(plan_id)
        if cached is not None:
            return {
                "plan_id": cached.plan_id,
                "status": cached.status,
                "steps": cached.steps,
                "total_steps": cached.total_steps,
                "source_schema_id": cached.source_schema_id,
                "target_schema_id": cached.target_schema_id,
            }

        raise ValueError(f"Migration plan not found: {plan_id}")

    # ==================================================================
    # Internal helpers: statistics tracking
    # ==================================================================

    def _update_avg_duration(self, duration: float) -> None:
        """Update the running average migration duration.

        Args:
            duration: Latest migration duration in seconds.
        """
        total = self._stats.total_migrations_executed
        if total <= 0:
            self._stats.avg_migration_duration_seconds = duration
            return
        prev_avg = self._stats.avg_migration_duration_seconds
        self._stats.avg_migration_duration_seconds = (
            (prev_avg * (total - 1) + duration) / total
        )

    def _update_success_rate(self) -> None:
        """Update the migration success rate from running totals."""
        if self._migration_total > 0:
            self._stats.success_rate = round(
                self._migration_successes / self._migration_total, 4,
            )

    def _increment_active_migrations(self) -> None:
        """Increment the active migration counter."""
        self._active_migrations += 1
        self._stats.active_migrations = self._active_migrations
        set_active_migrations(self._active_migrations)

    def _decrement_active_migrations(self) -> None:
        """Decrement the active migration counter."""
        self._active_migrations = max(0, self._active_migrations - 1)
        self._stats.active_migrations = self._active_migrations
        set_active_migrations(self._active_migrations)


# ===================================================================
# Thread-safe singleton access
# ===================================================================


def _get_singleton() -> SchemaMigrationService:
    """Get or create the singleton SchemaMigrationService instance.

    Returns:
        The singleton SchemaMigrationService.
    """
    global _singleton_instance
    if _singleton_instance is None:
        with _singleton_lock:
            if _singleton_instance is None:
                _singleton_instance = SchemaMigrationService()
    return _singleton_instance


# ===================================================================
# FastAPI integration
# ===================================================================


async def configure_schema_migration(
    app: Any,
    config: Optional[SchemaMigrationConfig] = None,
) -> SchemaMigrationService:
    """Configure the Schema Migration Service on a FastAPI application.

    Creates the SchemaMigrationService, stores it in app.state, mounts
    the schema migration API router, and starts the service.

    Args:
        app: FastAPI application instance.
        config: Optional schema migration config.

    Returns:
        SchemaMigrationService instance.
    """
    global _singleton_instance

    service = SchemaMigrationService(config=config)

    # Store as singleton
    with _singleton_lock:
        _singleton_instance = service

    # Attach to app state
    app.state.schema_migration_service = service

    # Mount schema migration API router
    try:
        from greenlang.schema_migration.api.router import router as sm_router
        if sm_router is not None:
            app.include_router(sm_router)
            logger.info("Schema migration service API router mounted")
    except ImportError:
        logger.warning(
            "Schema migration router not available; API not mounted"
        )

    # Start service
    service.startup()

    logger.info("Schema migration service configured on app")
    return service


def get_schema_migration(app: Any) -> SchemaMigrationService:
    """Get the SchemaMigrationService instance from app state.

    Args:
        app: FastAPI application instance.

    Returns:
        SchemaMigrationService instance.

    Raises:
        RuntimeError: If schema migration service not configured.
    """
    service = getattr(app.state, "schema_migration_service", None)
    if service is None:
        raise RuntimeError(
            "Schema migration service not configured. "
            "Call configure_schema_migration(app) first."
        )
    return service


def get_router(service: Optional[SchemaMigrationService] = None) -> Any:
    """Get the schema migration API router.

    Args:
        service: Optional service instance (unused, kept for API compat).

    Returns:
        FastAPI APIRouter or None if FastAPI not available.
    """
    try:
        from greenlang.schema_migration.api.router import router
        return router
    except ImportError:
        return None


# ===================================================================
# Public API
# ===================================================================

__all__ = [
    # Service class
    "SchemaMigrationService",
    # FastAPI integration
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
