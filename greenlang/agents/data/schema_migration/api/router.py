# -*- coding: utf-8 -*-
"""
Schema Migration Agent REST API Router - AGENT-DATA-017

FastAPI router providing 20 endpoints for schema registration,
versioning, change detection, compatibility checking, migration
planning, execution, rollback, pipeline orchestration, health,
and statistics.

All endpoints are mounted under ``/api/v1/schema-migration``.

Endpoints:
    1.  POST   /schemas                      - Register a new schema
    2.  GET    /schemas                      - List registered schemas
    3.  GET    /schemas/{schema_id}          - Get full schema details
    4.  PUT    /schemas/{schema_id}          - Update schema metadata
    5.  DELETE /schemas/{schema_id}          - Deregister schema (soft delete)
    6.  POST   /versions                     - Create a new schema version
    7.  GET    /versions                     - List versions
    8.  GET    /versions/{version_id}        - Get version details
    9.  POST   /changes/detect               - Detect changes between two versions
    10. GET    /changes                      - List detected changes
    11. POST   /compatibility/check          - Check compatibility between versions
    12. GET    /compatibility                - List compatibility check results
    13. POST   /plans                        - Generate a migration plan
    14. GET    /plans/{plan_id}              - Get migration plan details
    15. POST   /execute                      - Execute a migration plan
    16. GET    /executions/{execution_id}    - Get execution status
    17. POST   /rollback/{execution_id}      - Rollback a migration execution
    18. POST   /pipeline                     - Run the full migration pipeline
    19. GET    /health                       - Health check
    20. GET    /stats                        - Service statistics

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-017 Schema Migration Agent (GL-DATA-X-020)
Status: Production Ready
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FastAPI import (no `from __future__ import annotations` here)
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, HTTPException, Query, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None  # type: ignore[assignment, misc]
    logger.warning(
        "FastAPI not available; schema migration router is None"
    )


# ---------------------------------------------------------------------------
# Pydantic request/response models (only when FastAPI is available)
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    class RegisterSchemaBody(BaseModel):
        """Request body for registering a new schema definition."""
        namespace: str = Field(
            ..., description="Dot/hyphen-separated namespace for the schema",
        )
        name: str = Field(
            ..., description="Human-readable schema name (max 255 chars)",
        )
        schema_type: str = Field(
            default="json_schema",
            description="Schema serialization format (json_schema, avro, protobuf)",
        )
        definition: Dict[str, Any] = Field(
            ..., description="The schema definition as a JSON-serializable dict",
        )
        owner: Optional[str] = Field(
            None, description="Team or service responsible for this schema",
        )
        tags: Optional[List[str]] = Field(
            None, description="Tags for discovery and filtering",
        )
        description: Optional[str] = Field(
            None, description="Human-readable description of the schema",
        )

    class UpdateSchemaBody(BaseModel):
        """Request body for updating mutable fields of a schema."""
        owner: Optional[str] = Field(
            None, description="Updated team or service owner",
        )
        tags: Optional[List[str]] = Field(
            None, description="Updated tags for discovery and filtering",
        )
        status: Optional[str] = Field(
            None, description="Updated lifecycle status (active, deprecated, archived)",
        )
        description: Optional[str] = Field(
            None, description="Updated human-readable description",
        )

    class CreateVersionBody(BaseModel):
        """Request body for creating a new schema version."""
        schema_id: str = Field(
            ..., description="ID of the parent SchemaDefinition to version",
        )
        definition: Dict[str, Any] = Field(
            ..., description="The new schema definition at this version",
        )
        changelog_note: Optional[str] = Field(
            None, description="Human-readable description of what changed",
        )

    class DetectChangesBody(BaseModel):
        """Request body for detecting changes between two schema versions."""
        source_version_id: str = Field(
            ..., description="ID of the source (older) schema version",
        )
        target_version_id: str = Field(
            ..., description="ID of the target (newer) schema version",
        )

    class CheckCompatibilityBody(BaseModel):
        """Request body for compatibility check between two schema versions."""
        source_version_id: str = Field(
            ..., description="ID of the source (older) schema version",
        )
        target_version_id: str = Field(
            ..., description="ID of the target (newer) schema version",
        )
        level: Optional[str] = Field(
            "backward",
            description="Compatibility level to check (backward, forward, full)",
        )

    class CreatePlanBody(BaseModel):
        """Request body for generating a migration plan."""
        source_schema_id: str = Field(
            ..., description="ID of the source schema definition",
        )
        target_schema_id: str = Field(
            ..., description="ID of the target schema definition",
        )
        source_version: Optional[str] = Field(
            None, description="SemVer string of the source schema version",
        )
        target_version: Optional[str] = Field(
            None, description="SemVer string of the target schema version",
        )

    class ExecuteMigrationBody(BaseModel):
        """Request body for executing a migration plan."""
        plan_id: str = Field(
            ..., description="ID of the MigrationPlan to execute",
        )
        dry_run: Optional[bool] = Field(
            False, description="If True, validate without committing changes",
        )

    class RollbackBody(BaseModel):
        """Request body for rollback configuration."""
        to_checkpoint: Optional[int] = Field(
            None, description="Checkpoint step number to rollback to",
        )

    class RunPipelineBody(BaseModel):
        """Request body for running the full migration pipeline."""
        source_schema_id: str = Field(
            ..., description="ID of the source schema definition",
        )
        target_schema_id: str = Field(
            ..., description="ID of the target schema definition",
        )
        source_version: Optional[str] = Field(
            None, description="SemVer string of the source schema version",
        )
        target_version: Optional[str] = Field(
            None, description="SemVer string of the target schema version",
        )
        skip_compatibility: bool = Field(
            default=False,
            description="If True, skip the compatibility analysis stage",
        )
        skip_dry_run: bool = Field(
            default=False,
            description="If True, execute the plan without a prior dry-run",
        )


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def _get_service(request: "Request", service: Any = None) -> Any:
    """Extract SchemaMigrationService from app state or use provided service.

    Args:
        request: FastAPI request object.
        service: Optional pre-configured service instance.

    Returns:
        SchemaMigrationService instance.

    Raises:
        HTTPException: If service is not configured.
    """
    if service is not None:
        return service
    svc = getattr(
        request.app.state, "schema_migration_service", None,
    )
    if svc is None:
        raise HTTPException(
            status_code=503,
            detail="Schema migration service not configured",
        )
    return svc


def create_schema_migration_router(service: Any = None) -> "APIRouter":
    """Create and return the Schema Migration API router.

    This factory function creates a FastAPI APIRouter with all 20 endpoints
    for the Schema Migration Agent. The router is prefixed with
    ``/api/v1/schema-migration`` and tagged for OpenAPI documentation.

    Each endpoint resolves its service dependency either from the ``service``
    argument (for testing) or from ``request.app.state.schema_migration_service``
    (for production deployment).

    Args:
        service: Optional pre-configured SchemaMigrationService instance.
            When provided, all endpoints use this service directly instead
            of looking it up from the request's app state. This is primarily
            useful for unit testing.

    Returns:
        Configured APIRouter instance with all 20 endpoints registered.

    Raises:
        RuntimeError: If FastAPI is not installed.

    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> router = create_schema_migration_router()
        >>> app.include_router(router)
    """
    if not FASTAPI_AVAILABLE:
        raise RuntimeError(
            "FastAPI is required to create the schema migration router. "
            "Install it with: pip install fastapi"
        )

    router = APIRouter(
        prefix="/api/v1/schema-migration",
        tags=["Schema Migration"],
    )

    # ==================================================================
    # 1. POST /schemas - Register a new schema
    # ==================================================================
    @router.post("/schemas", status_code=201)
    async def register_schema(
        body: RegisterSchemaBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Register a new schema definition in the migration registry.

        Creates a new schema entry with the given namespace, name, type,
        and definition. Returns the complete schema record including the
        generated schema_id and initial status.
        """
        svc = _get_service(request, service)
        try:
            logger.info(
                "POST /schemas: namespace=%s name=%s type=%s",
                body.namespace,
                body.name,
                body.schema_type,
            )
            result = svc.register_schema(
                namespace=body.namespace,
                name=body.name,
                schema_type=body.schema_type,
                definition=body.definition,
                owner=body.owner,
                tags=body.tags,
                description=body.description,
            )
            if isinstance(result, BaseModel):
                return result.model_dump(mode="json")
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error("POST /schemas failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 2. GET /schemas - List registered schemas
    # ==================================================================
    @router.get("/schemas")
    async def list_schemas(
        namespace: Optional[str] = Query(None, description="Filter by namespace"),
        schema_type: Optional[str] = Query(None, description="Filter by schema type"),
        status: Optional[str] = Query(None, description="Filter by status"),
        owner: Optional[str] = Query(None, description="Filter by owner"),
        tag: Optional[str] = Query(None, description="Filter by tag"),
        limit: int = Query(50, ge=1, le=1000, description="Maximum results"),
        offset: int = Query(0, ge=0, description="Pagination offset"),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List registered schemas with optional filtering and pagination.

        Supports filtering by namespace, schema_type, status, owner, and
        tag. Results are paginated with configurable limit and offset.
        """
        svc = _get_service(request, service)
        try:
            logger.info(
                "GET /schemas: namespace=%s type=%s status=%s owner=%s "
                "tag=%s limit=%d offset=%d",
                namespace,
                schema_type,
                status,
                owner,
                tag,
                limit,
                offset,
            )
            schemas = svc.list_schemas(
                namespace=namespace,
                schema_type=schema_type,
                status=status,
                owner=owner,
                tag=tag,
                limit=limit,
                offset=offset,
            )
            items = [
                s.model_dump(mode="json") if isinstance(s, BaseModel) else s
                for s in schemas
            ]
            return {
                "schemas": items,
                "count": len(items),
                "limit": limit,
                "offset": offset,
            }
        except Exception as exc:
            logger.error("GET /schemas failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 3. GET /schemas/{schema_id} - Get full schema details
    # ==================================================================
    @router.get("/schemas/{schema_id}")
    async def get_schema(
        schema_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Retrieve a single schema definition by its unique identifier.

        Returns the complete schema record including definition, metadata,
        status, tags, and creation timestamp.
        """
        svc = _get_service(request, service)
        try:
            logger.info("GET /schemas/%s", schema_id)
            result = svc.get_schema(schema_id)
            if result is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Schema {schema_id} not found",
                )
            if isinstance(result, BaseModel):
                return result.model_dump(mode="json")
            return result
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "GET /schemas/%s failed: %s", schema_id, exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 4. PUT /schemas/{schema_id} - Update schema metadata
    # ==================================================================
    @router.put("/schemas/{schema_id}")
    async def update_schema(
        schema_id: str,
        body: UpdateSchemaBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Update mutable fields of an existing schema definition.

        Only owner, tags, status, and description can be updated.
        The namespace, name, and schema_type are immutable once registered.
        """
        svc = _get_service(request, service)
        try:
            logger.info("PUT /schemas/%s", schema_id)
            updates = body.model_dump(exclude_none=True)
            if not updates:
                raise HTTPException(
                    status_code=400,
                    detail="No update fields provided",
                )
            result = svc.update_schema(schema_id=schema_id, updates=updates)
            if result is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Schema {schema_id} not found",
                )
            if isinstance(result, BaseModel):
                return result.model_dump(mode="json")
            return result
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error(
                "PUT /schemas/%s failed: %s", schema_id, exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 5. DELETE /schemas/{schema_id} - Deregister schema (soft delete)
    # ==================================================================
    @router.delete("/schemas/{schema_id}", status_code=200)
    async def delete_schema(
        schema_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Soft-delete a schema definition by setting its status to archived.

        The schema record is retained for audit trail purposes but is no
        longer usable for new versions or migration operations.
        """
        svc = _get_service(request, service)
        try:
            logger.info("DELETE /schemas/%s", schema_id)
            deleted = svc.delete_schema(schema_id)
            if not deleted:
                raise HTTPException(
                    status_code=404,
                    detail=f"Schema {schema_id} not found",
                )
            return {
                "deleted": True,
                "schema_id": schema_id,
            }
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "DELETE /schemas/%s failed: %s", schema_id, exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 6. POST /versions - Create a new schema version
    # ==================================================================
    @router.post("/versions", status_code=201)
    async def create_version(
        body: CreateVersionBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Create a new version of an existing schema definition.

        The version string is automatically determined by the versioner
        engine based on the severity of detected changes. Returns the
        complete version record with the assigned SemVer string.
        """
        svc = _get_service(request, service)
        try:
            logger.info(
                "POST /versions: schema_id=%s", body.schema_id,
            )
            result = svc.create_version(
                schema_id=body.schema_id,
                definition=body.definition,
                changelog_note=body.changelog_note,
            )
            if isinstance(result, BaseModel):
                return result.model_dump(mode="json")
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error(
                "POST /versions failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 7. GET /versions - List versions
    # ==================================================================
    @router.get("/versions")
    async def list_versions(
        schema_id: Optional[str] = Query(None, description="Filter by schema ID"),
        version_range: Optional[str] = Query(
            None, description="SemVer range filter (e.g., '>=1.0.0 <2.0.0')",
        ),
        deprecated: Optional[bool] = Query(
            None, description="Filter by deprecation status",
        ),
        limit: int = Query(50, ge=1, le=1000, description="Maximum results"),
        offset: int = Query(0, ge=0, description="Pagination offset"),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List schema versions with optional filtering and pagination.

        Supports filtering by parent schema_id, SemVer version range,
        and deprecation status. Results are paginated.
        """
        svc = _get_service(request, service)
        try:
            logger.info(
                "GET /versions: schema_id=%s range=%s deprecated=%s "
                "limit=%d offset=%d",
                schema_id,
                version_range,
                deprecated,
                limit,
                offset,
            )
            versions = svc.list_versions(
                schema_id=schema_id,
                version_range=version_range,
                deprecated=deprecated,
                limit=limit,
                offset=offset,
            )
            items = [
                v.model_dump(mode="json") if isinstance(v, BaseModel) else v
                for v in versions
            ]
            return {
                "versions": items,
                "count": len(items),
                "limit": limit,
                "offset": offset,
            }
        except Exception as exc:
            logger.error("GET /versions failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 8. GET /versions/{version_id} - Get version details
    # ==================================================================
    @router.get("/versions/{version_id}")
    async def get_version(
        version_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Retrieve a single schema version by its unique identifier.

        Returns the complete version record including the definition
        snapshot, changelog, deprecation status, and provenance data.
        """
        svc = _get_service(request, service)
        try:
            logger.info("GET /versions/%s", version_id)
            result = svc.get_version(version_id)
            if result is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Version {version_id} not found",
                )
            if isinstance(result, BaseModel):
                return result.model_dump(mode="json")
            return result
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "GET /versions/%s failed: %s", version_id, exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 9. POST /changes/detect - Detect changes between two versions
    # ==================================================================
    @router.post("/changes/detect", status_code=201)
    async def detect_changes(
        body: DetectChangesBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Detect structural changes between two schema versions.

        Triggers the change detection engine to produce a list of
        SchemaChange objects describing every field-level or schema-level
        difference between the source and target versions.
        """
        svc = _get_service(request, service)
        try:
            logger.info(
                "POST /changes/detect: source=%s target=%s",
                body.source_version_id,
                body.target_version_id,
            )
            result = svc.detect_changes(
                source_version_id=body.source_version_id,
                target_version_id=body.target_version_id,
            )
            if isinstance(result, BaseModel):
                return result.model_dump(mode="json")
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error(
                "POST /changes/detect failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 10. GET /changes - List detected changes
    # ==================================================================
    @router.get("/changes")
    async def list_changes(
        schema_id: Optional[str] = Query(
            None, description="Filter by schema ID",
        ),
        severity: Optional[str] = Query(
            None, description="Filter by severity (breaking, non_breaking, cosmetic)",
        ),
        change_type: Optional[str] = Query(
            None,
            description="Filter by change type (added, removed, renamed, retyped, etc.)",
        ),
        limit: int = Query(50, ge=1, le=1000, description="Maximum results"),
        offset: int = Query(0, ge=0, description="Pagination offset"),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List detected schema changes with optional filtering.

        Supports filtering by parent schema_id, change severity, and
        change type. Results are paginated with configurable limit
        and offset.
        """
        svc = _get_service(request, service)
        try:
            logger.info(
                "GET /changes: schema_id=%s severity=%s type=%s "
                "limit=%d offset=%d",
                schema_id,
                severity,
                change_type,
                limit,
                offset,
            )
            changes = svc.list_changes(
                schema_id=schema_id,
                severity=severity,
                change_type=change_type,
                limit=limit,
                offset=offset,
            )
            items = [
                c.model_dump(mode="json") if isinstance(c, BaseModel) else c
                for c in changes
            ]
            return {
                "changes": items,
                "count": len(items),
                "limit": limit,
                "offset": offset,
            }
        except Exception as exc:
            logger.error("GET /changes failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 11. POST /compatibility/check - Check compatibility
    # ==================================================================
    @router.post("/compatibility/check", status_code=201)
    async def check_compatibility(
        body: CheckCompatibilityBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Check compatibility between two schema versions.

        Determines whether the change from source to target version is
        backward-compatible, forward-compatible, fully compatible, or
        breaking. Returns the compatibility result with issues and
        recommendations.
        """
        svc = _get_service(request, service)
        try:
            logger.info(
                "POST /compatibility/check: source=%s target=%s level=%s",
                body.source_version_id,
                body.target_version_id,
                body.level,
            )
            result = svc.check_compatibility(
                source_version_id=body.source_version_id,
                target_version_id=body.target_version_id,
                level=body.level,
            )
            if isinstance(result, BaseModel):
                return result.model_dump(mode="json")
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error(
                "POST /compatibility/check failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 12. GET /compatibility - List compatibility check results
    # ==================================================================
    @router.get("/compatibility")
    async def list_compatibility_results(
        schema_id: Optional[str] = Query(
            None, description="Filter by schema ID",
        ),
        result: Optional[str] = Query(
            None,
            description="Filter by result (backward, forward, full, breaking)",
        ),
        limit: int = Query(50, ge=1, le=1000, description="Maximum results"),
        offset: int = Query(0, ge=0, description="Pagination offset"),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List compatibility check results with optional filtering.

        Supports filtering by parent schema_id and compatibility result
        level. Results are paginated with configurable limit and offset.
        """
        svc = _get_service(request, service)
        try:
            logger.info(
                "GET /compatibility: schema_id=%s result=%s "
                "limit=%d offset=%d",
                schema_id,
                result,
                limit,
                offset,
            )
            results = svc.list_compatibility_results(
                schema_id=schema_id,
                result_filter=result,
                limit=limit,
                offset=offset,
            )
            items = [
                r.model_dump(mode="json") if isinstance(r, BaseModel) else r
                for r in results
            ]
            return {
                "compatibility_results": items,
                "count": len(items),
                "limit": limit,
                "offset": offset,
            }
        except Exception as exc:
            logger.error(
                "GET /compatibility failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 13. POST /plans - Generate a migration plan
    # ==================================================================
    @router.post("/plans", status_code=201)
    async def create_plan(
        body: CreatePlanBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Generate a migration plan between two schema versions.

        Triggers the migration planner to produce an ordered list of
        transformation steps for migrating records from the source
        schema to the target schema.
        """
        svc = _get_service(request, service)
        try:
            logger.info(
                "POST /plans: source_schema=%s target_schema=%s "
                "source_version=%s target_version=%s",
                body.source_schema_id,
                body.target_schema_id,
                body.source_version,
                body.target_version,
            )
            result = svc.create_plan(
                source_schema_id=body.source_schema_id,
                target_schema_id=body.target_schema_id,
                source_version=body.source_version,
                target_version=body.target_version,
            )
            if isinstance(result, BaseModel):
                return result.model_dump(mode="json")
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error(
                "POST /plans failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 14. GET /plans/{plan_id} - Get migration plan details
    # ==================================================================
    @router.get("/plans/{plan_id}")
    async def get_plan(
        plan_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Retrieve a migration plan by its unique identifier.

        Returns the complete plan record including all transformation
        steps, estimated effort, status, and provenance data.
        """
        svc = _get_service(request, service)
        try:
            logger.info("GET /plans/%s", plan_id)
            result = svc.get_plan(plan_id)
            if result is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Plan {plan_id} not found",
                )
            if isinstance(result, BaseModel):
                return result.model_dump(mode="json")
            return result
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "GET /plans/%s failed: %s", plan_id, exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 15. POST /execute - Execute a migration plan
    # ==================================================================
    @router.post("/execute", status_code=201)
    async def execute_migration(
        body: ExecuteMigrationBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Execute a validated migration plan.

        Triggers the migration executor to apply the plan's transformation
        steps to all records in the target dataset. Supports dry-run mode
        for pre-execution validation without committing changes.
        """
        svc = _get_service(request, service)
        try:
            logger.info(
                "POST /execute: plan_id=%s dry_run=%s",
                body.plan_id,
                body.dry_run,
            )
            result = svc.execute_migration(
                plan_id=body.plan_id,
                dry_run=body.dry_run,
            )
            if isinstance(result, BaseModel):
                return result.model_dump(mode="json")
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error(
                "POST /execute failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 16. GET /executions/{execution_id} - Get execution status
    # ==================================================================
    @router.get("/executions/{execution_id}")
    async def get_execution(
        execution_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Retrieve the status and details of a migration execution.

        Returns the complete execution record including current step
        progress, record counts, checkpoint data, and error details.
        """
        svc = _get_service(request, service)
        try:
            logger.info("GET /executions/%s", execution_id)
            result = svc.get_execution(execution_id)
            if result is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Execution {execution_id} not found",
                )
            if isinstance(result, BaseModel):
                return result.model_dump(mode="json")
            return result
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "GET /executions/%s failed: %s",
                execution_id,
                exc,
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 17. POST /rollback/{execution_id} - Rollback a migration execution
    # ==================================================================
    @router.post("/rollback/{execution_id}", status_code=201)
    async def rollback_execution(
        execution_id: str,
        body: RollbackBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Rollback a migration execution to a previous state.

        Reverts the effects of a migration execution either fully or
        partially to a specified checkpoint. Returns the rollback record
        with details on reverted records and status.
        """
        svc = _get_service(request, service)
        try:
            logger.info(
                "POST /rollback/%s: to_checkpoint=%s",
                execution_id,
                body.to_checkpoint,
            )
            result = svc.rollback_execution(
                execution_id=execution_id,
                to_checkpoint=body.to_checkpoint,
            )
            if isinstance(result, BaseModel):
                return result.model_dump(mode="json")
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error(
                "POST /rollback/%s failed: %s",
                execution_id,
                exc,
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 18. POST /pipeline - Run the full migration pipeline
    # ==================================================================
    @router.post("/pipeline", status_code=201)
    async def run_pipeline(
        body: RunPipelineBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Run the full end-to-end migration pipeline.

        Orchestrates all stages: change detection, compatibility checking,
        migration plan creation, optional dry-run execution, and
        post-migration verification. Returns the aggregated pipeline
        result with outputs from each completed stage.
        """
        svc = _get_service(request, service)
        try:
            logger.info(
                "POST /pipeline: source_schema=%s target_schema=%s "
                "source_version=%s target_version=%s "
                "skip_compat=%s skip_dry_run=%s",
                body.source_schema_id,
                body.target_schema_id,
                body.source_version,
                body.target_version,
                body.skip_compatibility,
                body.skip_dry_run,
            )
            result = svc.run_pipeline(
                source_schema_id=body.source_schema_id,
                target_schema_id=body.target_schema_id,
                source_version=body.source_version,
                target_version=body.target_version,
                skip_compatibility=body.skip_compatibility,
                skip_dry_run=body.skip_dry_run,
            )
            if isinstance(result, BaseModel):
                return result.model_dump(mode="json")
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error(
                "POST /pipeline failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 19. GET /health - Health check
    # ==================================================================
    @router.get("/health")
    async def health(
        request: Request,
    ) -> Dict[str, Any]:
        """Schema migration service health check endpoint.

        Returns the service health status including engine availability,
        uptime, and version information.
        """
        svc = _get_service(request, service)
        try:
            logger.info("GET /health")
            result = svc.health_check()
            return result
        except Exception as exc:
            logger.error(
                "GET /health failed: %s", exc, exc_info=True,
            )
            return {
                "status": "unhealthy",
                "error": str(exc),
            }

    # ==================================================================
    # 20. GET /stats - Service statistics
    # ==================================================================
    @router.get("/stats")
    async def stats(
        request: Request,
    ) -> Dict[str, Any]:
        """Retrieve aggregated operational statistics for the service.

        Returns high-level metrics including total schemas, versions,
        changes detected, migrations executed, rollbacks performed,
        and breakdowns by schema type and status.
        """
        svc = _get_service(request, service)
        try:
            logger.info("GET /stats")
            result = svc.get_stats()
            if isinstance(result, BaseModel):
                return result.model_dump(mode="json")
            return result
        except Exception as exc:
            logger.error(
                "GET /stats failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    return router


# ---------------------------------------------------------------------------
# Module-level router (for backwards compatibility with include_router usage)
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:
    router = create_schema_migration_router()
else:
    router = None  # type: ignore[assignment]


__all__ = [
    "create_schema_migration_router",
    "router",
    "FASTAPI_AVAILABLE",
]
