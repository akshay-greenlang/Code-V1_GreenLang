# -*- coding: utf-8 -*-
"""
GreenLang Orchestrator Control Plane API - Routes
==================================================

FastAPI routes for the GL-FOUND-X-001 GreenLang Orchestrator Control Plane.

This module implements REST endpoints for:
- Pipeline Management (register, list, get, delete)
- Run Operations (submit, list, get, cancel, logs, audit)
- Health and Metrics (health, metrics, ready)
- Agent Registry (list, get)

All endpoints include:
- OpenAPI documentation with examples
- Request validation via Pydantic
- Authentication via X-API-Key header
- Distributed tracing support
- GreenLang error codes

Author: GreenLang Team
Version: 1.0.0
GL-FOUND-X-001: Control Plane API Routes
"""

import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import JSONResponse, PlainTextResponse

from greenlang.orchestrator.api.deps import (
    APIConfig,
    AuthContext,
    RequestTrace,
    get_agent_registry,
    get_api_key,
    get_config,
    get_event_store,
    get_orchestrator,
    get_policy_engine,
    get_request_trace,
)
from greenlang.orchestrator.api.models import (
    AgentDetailResponse,
    AgentListParams,
    AgentListResponse,
    AgentResponse,
    AgentCapabilityResponse,
    AuditEvent,
    ComponentHealth,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    HealthStatus,
    LogEntry,
    LogLevel,
    LogsQueryParams,
    MetricsResponse,
    PipelineDetailResponse,
    PipelineListParams,
    PipelineListResponse,
    PipelineRegisterRequest,
    PipelineResponse,
    ReadinessResponse,
    RunAuditResponse,
    RunCancelRequest,
    RunCancelResponse,
    RunDetailResponse,
    RunListParams,
    RunListResponse,
    RunLogsResponse,
    RunResponse,
    RunStatus,
    RunSubmitRequest,
    StepStatus,
    StepStatusResponse,
    # FR-074: Checkpoint and retry models
    CheckpointStatusEnum,
    StepCheckpointResponse,
    RunCheckpointResponse,
    RunRetryRequest,
    RunRetryResponse,
    NonIdempotentStepWarning,
    CheckpointClearResponse,
)

# FR-043: Import approval routes
from greenlang.orchestrator.api.approval_routes import (
    approval_router,
    run_approval_router,
)

logger = logging.getLogger(__name__)

# =============================================================================
# ROUTER INITIALIZATION
# =============================================================================

# Main API router
router = APIRouter()

# Pipeline management router
pipeline_router = APIRouter(prefix="/pipelines", tags=["Pipeline Management"])

# Run operations router
run_router = APIRouter(prefix="/runs", tags=["Run Operations"])

# Agent registry router
agent_router = APIRouter(prefix="/agents", tags=["Agent Registry"])

# FR-024: Quota management router
quota_router = APIRouter(prefix="/quotas", tags=["Quota Management"])

# FR-024: Namespace-specific quota router
namespace_quota_router = APIRouter(prefix="/namespaces", tags=["Namespace Quotas"])


# =============================================================================
# ERROR CODES
# =============================================================================

class GreenLangErrorCodes:
    """GreenLang error code constants."""
    # Validation errors (GL-E-VAL-*)
    PARAM_MISSING = "GL-E-VAL-001"
    PARAM_INVALID = "GL-E-VAL-002"
    PAYLOAD_INVALID = "GL-E-VAL-003"

    # Resource errors (GL-E-RES-*)
    PIPELINE_NOT_FOUND = "GL-E-RES-001"
    RUN_NOT_FOUND = "GL-E-RES-002"
    AGENT_NOT_FOUND = "GL-E-RES-003"
    NAMESPACE_NOT_FOUND = "GL-E-RES-004"  # FR-024
    CHECKPOINT_NOT_FOUND = "GL-E-RES-005"  # FR-074

    # Operation errors (GL-E-OPS-*)
    PIPELINE_EXISTS = "GL-E-OPS-001"
    RUN_NOT_CANCELABLE = "GL-E-OPS-002"
    POLICY_VIOLATION = "GL-E-OPS-003"
    RUN_NOT_RETRYABLE = "GL-E-OPS-004"  # FR-074
    MAX_RETRIES_EXCEEDED = "GL-E-OPS-005"  # FR-074
    SCHEMA_INCOMPATIBLE = "GL-E-OPS-006"  # FR-074
    CHECKPOINT_EXPIRED = "GL-E-OPS-007"  # FR-074

    # Auth errors (GL-E-AUTH-*)
    UNAUTHORIZED = "GL-E-AUTH-001"
    FORBIDDEN = "GL-E-AUTH-002"

    # System errors (GL-E-SYS-*)
    INTERNAL_ERROR = "GL-E-SYS-001"
    SERVICE_UNAVAILABLE = "GL-E-SYS-002"


def create_error_response(
    error_type: str,
    message: str,
    details: Optional[List[ErrorDetail]] = None,
    trace_id: Optional[str] = None,
) -> ErrorResponse:
    """Create a standardized error response."""
    return ErrorResponse(
        error=error_type,
        message=message,
        details=details or [],
        trace_id=trace_id,
        timestamp=datetime.now(timezone.utc),
    )


# =============================================================================
# METRICS TRACKING
# =============================================================================

# Global metrics (would use Prometheus client in production)
_metrics = {
    "runs_total": 0,
    "runs_active": 0,
    "runs_succeeded": 0,
    "runs_failed": 0,
    "steps_total": 0,
    "steps_succeeded": 0,
    "steps_failed": 0,
    "policy_evaluations": 0,
    "policy_denials": 0,
    "start_time": time.time(),
}


# =============================================================================
# PIPELINE MANAGEMENT ENDPOINTS
# =============================================================================


@pipeline_router.post(
    "",
    response_model=PipelineResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a pipeline",
    description="Register a new pipeline definition. The pipeline spec is validated and stored.",
    responses={
        201: {"description": "Pipeline registered successfully"},
        400: {"model": ErrorResponse, "description": "Invalid pipeline definition"},
        409: {"model": ErrorResponse, "description": "Pipeline already exists"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
async def register_pipeline(
    request: PipelineRegisterRequest,
    auth: AuthContext = Depends(get_api_key),
    trace: RequestTrace = Depends(get_request_trace),
    orchestrator=Depends(get_orchestrator),
) -> PipelineResponse:
    """
    Register a new pipeline definition.

    The pipeline YAML/JSON is validated for:
    - Step dependency graph (no cycles)
    - Agent availability
    - Parameter schema

    Returns the registered pipeline with generated ID and content hash.
    """
    logger.info(f"Registering pipeline: {request.metadata.name} [{trace.trace_id}]")

    try:
        # Convert to internal format
        pipeline_data = {
            "api_version": request.api_version,
            "kind": request.kind,
            "metadata": request.metadata.model_dump(),
            "spec": request.spec.model_dump(by_alias=True),
        }

        # Compute content hash
        import json
        content_str = json.dumps(pipeline_data, sort_keys=True)
        content_hash = hashlib.sha256(content_str.encode()).hexdigest()

        # Register with orchestrator
        pipeline_id = await orchestrator.register_pipeline(pipeline_data)

        # Build response
        now = datetime.now(timezone.utc)
        return PipelineResponse(
            pipeline_id=pipeline_id,
            name=request.metadata.name,
            namespace=request.metadata.namespace,
            version=request.metadata.version,
            description=request.metadata.description,
            owner=request.metadata.owner,
            team=request.metadata.team,
            labels=request.metadata.labels,
            tags=request.metadata.tags,
            step_count=len(request.spec.steps),
            created_at=now,
            updated_at=now,
            content_hash=content_hash,
        )

    except ValueError as e:
        logger.warning(f"Pipeline validation failed: {e} [{trace.trace_id}]")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=create_error_response(
                error_type="validation_error",
                message=str(e),
                details=[
                    ErrorDetail(
                        code=GreenLangErrorCodes.PAYLOAD_INVALID,
                        message=str(e),
                    )
                ],
                trace_id=trace.trace_id,
            ).model_dump(),
        )
    except Exception as e:
        logger.error(f"Pipeline registration failed: {e} [{trace.trace_id}]", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_type="internal_error",
                message="Failed to register pipeline",
                details=[
                    ErrorDetail(
                        code=GreenLangErrorCodes.INTERNAL_ERROR,
                        message=str(e) if auth.has_scope("admin") else "Internal error",
                    )
                ],
                trace_id=trace.trace_id,
            ).model_dump(),
        )


@pipeline_router.get(
    "",
    response_model=PipelineListResponse,
    summary="List pipelines",
    description="List all registered pipelines with optional filtering.",
    responses={
        200: {"description": "List of pipelines"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
async def list_pipelines(
    namespace: Optional[str] = Query(None, description="Filter by namespace"),
    owner: Optional[str] = Query(None, description="Filter by owner"),
    team: Optional[str] = Query(None, description="Filter by team"),
    label: Optional[str] = Query(None, description="Filter by label (key=value)"),
    search: Optional[str] = Query(None, description="Search in name/description"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(100, ge=1, le=1000, description="Pagination limit"),
    auth: AuthContext = Depends(get_api_key),
    trace: RequestTrace = Depends(get_request_trace),
    orchestrator=Depends(get_orchestrator),
) -> PipelineListResponse:
    """
    List registered pipelines with filtering and pagination.

    Supports filtering by namespace, owner, team, labels, and text search.
    """
    logger.debug(f"Listing pipelines [{trace.trace_id}]")

    try:
        # Build filter params
        filters = {}
        if namespace:
            filters["namespace"] = namespace
        if owner:
            filters["owner"] = owner
        if team:
            filters["team"] = team
        if label:
            key, _, value = label.partition("=")
            filters["labels"] = {key: value}
        if search:
            filters["search"] = search

        # Get pipelines from orchestrator
        raw_pipelines = await orchestrator.list_pipelines(
            offset=offset,
            limit=limit,
            **filters,
        )

        # Convert to response format
        pipelines = []
        for p in raw_pipelines:
            pipelines.append(
                PipelineResponse(
                    pipeline_id=p.get("id", p.get("pipeline_id", "")),
                    name=p.get("definition", {}).get("metadata", {}).get("name", ""),
                    namespace=p.get("definition", {}).get("metadata", {}).get("namespace", "default"),
                    version=p.get("definition", {}).get("metadata", {}).get("version", "1.0.0"),
                    description=p.get("definition", {}).get("metadata", {}).get("description"),
                    owner=p.get("definition", {}).get("metadata", {}).get("owner"),
                    team=p.get("definition", {}).get("metadata", {}).get("team"),
                    labels=p.get("definition", {}).get("metadata", {}).get("labels", {}),
                    tags=p.get("definition", {}).get("metadata", {}).get("tags", []),
                    step_count=len(p.get("definition", {}).get("spec", {}).get("steps", [])),
                    created_at=p.get("created_at", datetime.now(timezone.utc)),
                    updated_at=p.get("updated_at", p.get("created_at", datetime.now(timezone.utc))),
                    content_hash=p.get("content_hash", ""),
                )
            )

        return PipelineListResponse(
            pipelines=pipelines,
            total=len(raw_pipelines),  # In production, this would be the total count
            offset=offset,
            limit=limit,
        )

    except Exception as e:
        logger.error(f"Failed to list pipelines: {e} [{trace.trace_id}]", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_type="internal_error",
                message="Failed to list pipelines",
                trace_id=trace.trace_id,
            ).model_dump(),
        )


@pipeline_router.get(
    "/{pipeline_id}",
    response_model=PipelineDetailResponse,
    summary="Get pipeline details",
    description="Get detailed information about a specific pipeline.",
    responses={
        200: {"description": "Pipeline details"},
        404: {"model": ErrorResponse, "description": "Pipeline not found"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
async def get_pipeline(
    pipeline_id: str,
    auth: AuthContext = Depends(get_api_key),
    trace: RequestTrace = Depends(get_request_trace),
    orchestrator=Depends(get_orchestrator),
) -> PipelineDetailResponse:
    """
    Get detailed information about a pipeline including its full specification.
    """
    logger.debug(f"Getting pipeline: {pipeline_id} [{trace.trace_id}]")

    try:
        pipeline = await orchestrator.get_pipeline(pipeline_id)

        if pipeline is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=create_error_response(
                    error_type="not_found",
                    message=f"Pipeline not found: {pipeline_id}",
                    details=[
                        ErrorDetail(
                            code=GreenLangErrorCodes.PIPELINE_NOT_FOUND,
                            message=f"Pipeline with ID '{pipeline_id}' does not exist",
                        )
                    ],
                    trace_id=trace.trace_id,
                ).model_dump(),
            )

        # Build execution order from steps
        spec = pipeline.get("definition", {}).get("spec", {})
        steps = spec.get("steps", [])

        # Simple topological sort for execution order
        execution_order = []
        dependency_graph = {}
        for step in steps:
            step_id = step.get("id", "")
            deps = step.get("depends_on", step.get("dependsOn", []))
            dependency_graph[step_id] = deps
            if not deps:
                execution_order.append(step_id)

        # Add remaining steps
        for step in steps:
            step_id = step.get("id", "")
            if step_id not in execution_order:
                execution_order.append(step_id)

        metadata = pipeline.get("definition", {}).get("metadata", {})

        return PipelineDetailResponse(
            pipeline_id=pipeline.get("id", pipeline_id),
            name=metadata.get("name", ""),
            namespace=metadata.get("namespace", "default"),
            version=metadata.get("version", "1.0.0"),
            description=metadata.get("description"),
            owner=metadata.get("owner"),
            team=metadata.get("team"),
            labels=metadata.get("labels", {}),
            tags=metadata.get("tags", []),
            step_count=len(steps),
            created_at=pipeline.get("created_at", datetime.now(timezone.utc)),
            updated_at=pipeline.get("updated_at", datetime.now(timezone.utc)),
            content_hash=pipeline.get("content_hash", ""),
            spec=spec,
            execution_order=execution_order,
            dependency_graph=dependency_graph,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get pipeline: {e} [{trace.trace_id}]", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_type="internal_error",
                message="Failed to get pipeline",
                trace_id=trace.trace_id,
            ).model_dump(),
        )


@pipeline_router.delete(
    "/{pipeline_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete pipeline",
    description="Unregister a pipeline definition.",
    responses={
        204: {"description": "Pipeline deleted"},
        404: {"model": ErrorResponse, "description": "Pipeline not found"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
async def delete_pipeline(
    pipeline_id: str,
    auth: AuthContext = Depends(get_api_key),
    trace: RequestTrace = Depends(get_request_trace),
    orchestrator=Depends(get_orchestrator),
) -> Response:
    """
    Delete a registered pipeline.

    Note: This does not affect runs that have already been submitted.
    """
    logger.info(f"Deleting pipeline: {pipeline_id} [{trace.trace_id}]")

    try:
        deleted = await orchestrator.delete_pipeline(pipeline_id)

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=create_error_response(
                    error_type="not_found",
                    message=f"Pipeline not found: {pipeline_id}",
                    details=[
                        ErrorDetail(
                            code=GreenLangErrorCodes.PIPELINE_NOT_FOUND,
                            message=f"Pipeline with ID '{pipeline_id}' does not exist",
                        )
                    ],
                    trace_id=trace.trace_id,
                ).model_dump(),
            )

        return Response(status_code=status.HTTP_204_NO_CONTENT)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete pipeline: {e} [{trace.trace_id}]", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_type="internal_error",
                message="Failed to delete pipeline",
                trace_id=trace.trace_id,
            ).model_dump(),
        )


# =============================================================================
# RUN OPERATIONS ENDPOINTS
# =============================================================================


@run_router.post(
    "",
    response_model=RunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a run",
    description="Submit a new pipeline run for execution.",
    responses={
        202: {"description": "Run submitted successfully"},
        400: {"model": ErrorResponse, "description": "Invalid run request"},
        404: {"model": ErrorResponse, "description": "Pipeline not found"},
        403: {"model": ErrorResponse, "description": "Policy violation"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
async def submit_run(
    request: RunSubmitRequest,
    auth: AuthContext = Depends(get_api_key),
    trace: RequestTrace = Depends(get_request_trace),
    orchestrator=Depends(get_orchestrator),
    policy_engine=Depends(get_policy_engine),
) -> RunResponse:
    """
    Submit a new pipeline run.

    The run is validated against policies, then queued for execution.
    Returns immediately with run ID for status tracking.
    """
    logger.info(
        f"Submitting run for pipeline {request.pipeline_id} "
        f"[tenant={request.tenant_id}] [{trace.trace_id}]"
    )

    try:
        # Check if pipeline exists
        pipeline = await orchestrator.get_pipeline(request.pipeline_id)
        if pipeline is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=create_error_response(
                    error_type="not_found",
                    message=f"Pipeline not found: {request.pipeline_id}",
                    details=[
                        ErrorDetail(
                            code=GreenLangErrorCodes.PIPELINE_NOT_FOUND,
                            message=f"Pipeline with ID '{request.pipeline_id}' does not exist",
                        )
                    ],
                    trace_id=trace.trace_id,
                ).model_dump(),
            )

        # Policy evaluation (unless skipped for development)
        if not request.skip_policy_check:
            policy_result = await policy_engine.evaluate({
                "pipeline_id": request.pipeline_id,
                "tenant_id": request.tenant_id,
                "user_id": request.user_id,
                "parameters": request.parameters,
            })
            _metrics["policy_evaluations"] += 1

            if not policy_result.allowed:
                _metrics["policy_denials"] += 1
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=create_error_response(
                        error_type="policy_violation",
                        message="Run blocked by policy",
                        details=[
                            ErrorDetail(
                                code=GreenLangErrorCodes.POLICY_VIOLATION,
                                message=str(v),
                            )
                            for v in policy_result.violations
                        ],
                        trace_id=trace.trace_id,
                    ).model_dump(),
                )

        # Submit to orchestrator
        run_id = await orchestrator.submit_run(
            pipeline_id=request.pipeline_id,
            parameters=request.parameters,
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            labels=request.labels,
            priority=request.priority,
            dry_run=request.dry_run,
            timeout_seconds=request.timeout_seconds,
        )

        _metrics["runs_total"] += 1
        _metrics["runs_active"] += 1

        # Get pipeline metadata
        pipeline_name = pipeline.get("definition", {}).get("metadata", {}).get("name", "")
        step_count = len(pipeline.get("definition", {}).get("spec", {}).get("steps", []))

        return RunResponse(
            run_id=run_id,
            pipeline_id=request.pipeline_id,
            pipeline_name=pipeline_name,
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            status=RunStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            progress_percent=0.0,
            steps_total=step_count,
            steps_completed=0,
            steps_failed=0,
            steps_running=0,
            labels=request.labels,
            trace_id=trace.trace_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit run: {e} [{trace.trace_id}]", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_type="internal_error",
                message="Failed to submit run",
                trace_id=trace.trace_id,
            ).model_dump(),
        )


@run_router.get(
    "",
    response_model=RunListResponse,
    summary="List runs",
    description="List pipeline runs with filtering.",
    responses={
        200: {"description": "List of runs"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
async def list_runs(
    status_filter: Optional[RunStatus] = Query(None, alias="status", description="Filter by status"),
    pipeline_id: Optional[str] = Query(None, description="Filter by pipeline ID"),
    tenant_id: Optional[str] = Query(None, description="Filter by tenant ID"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    created_after: Optional[datetime] = Query(None, description="Filter by creation date (after)"),
    created_before: Optional[datetime] = Query(None, description="Filter by creation date (before)"),
    label: Optional[str] = Query(None, description="Filter by label (key=value)"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(100, ge=1, le=1000, description="Pagination limit"),
    auth: AuthContext = Depends(get_api_key),
    trace: RequestTrace = Depends(get_request_trace),
    orchestrator=Depends(get_orchestrator),
) -> RunListResponse:
    """
    List pipeline runs with filtering and pagination.
    """
    logger.debug(f"Listing runs [{trace.trace_id}]")

    try:
        # Build filters
        filters = {}
        if status_filter:
            filters["status"] = status_filter.value
        if pipeline_id:
            filters["pipeline_id"] = pipeline_id
        if tenant_id:
            filters["tenant_id"] = tenant_id
        if user_id:
            filters["user_id"] = user_id
        if created_after:
            filters["created_after"] = created_after
        if created_before:
            filters["created_before"] = created_before
        if label:
            key, _, value = label.partition("=")
            filters["labels"] = {key: value}

        # Get runs from orchestrator
        raw_runs = await orchestrator.list_runs(
            offset=offset,
            limit=limit,
            **filters,
        )

        # Convert to response format
        runs = []
        for r in raw_runs:
            runs.append(
                RunResponse(
                    run_id=r.get("id", r.get("run_id", "")),
                    pipeline_id=r.get("pipeline_id", ""),
                    pipeline_name=r.get("pipeline_name", ""),
                    tenant_id=r.get("tenant_id", ""),
                    user_id=r.get("user_id"),
                    status=RunStatus(r.get("status", "pending")),
                    created_at=r.get("created_at", datetime.now(timezone.utc)),
                    started_at=r.get("started_at"),
                    completed_at=r.get("completed_at"),
                    duration_ms=r.get("duration_ms"),
                    progress_percent=r.get("progress_percent", 0.0),
                    steps_total=r.get("steps_total", 0),
                    steps_completed=r.get("steps_completed", 0),
                    steps_failed=r.get("steps_failed", 0),
                    steps_running=r.get("steps_running", 0),
                    labels=r.get("labels", {}),
                    error_message=r.get("error_message"),
                    trace_id=r.get("trace_id", trace.trace_id),
                )
            )

        return RunListResponse(
            runs=runs,
            total=len(raw_runs),
            offset=offset,
            limit=limit,
        )

    except Exception as e:
        logger.error(f"Failed to list runs: {e} [{trace.trace_id}]", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_type="internal_error",
                message="Failed to list runs",
                trace_id=trace.trace_id,
            ).model_dump(),
        )


@run_router.get(
    "/{run_id}",
    response_model=RunDetailResponse,
    summary="Get run details",
    description="Get detailed information about a specific run.",
    responses={
        200: {"description": "Run details"},
        404: {"model": ErrorResponse, "description": "Run not found"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
async def get_run(
    run_id: str,
    auth: AuthContext = Depends(get_api_key),
    trace: RequestTrace = Depends(get_request_trace),
    orchestrator=Depends(get_orchestrator),
) -> RunDetailResponse:
    """
    Get detailed information about a run including step statuses.
    """
    logger.debug(f"Getting run: {run_id} [{trace.trace_id}]")

    try:
        run = await orchestrator.get_run(run_id)

        if run is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=create_error_response(
                    error_type="not_found",
                    message=f"Run not found: {run_id}",
                    details=[
                        ErrorDetail(
                            code=GreenLangErrorCodes.RUN_NOT_FOUND,
                            message=f"Run with ID '{run_id}' does not exist",
                        )
                    ],
                    trace_id=trace.trace_id,
                ).model_dump(),
            )

        # Build step statuses
        steps = []
        for s in run.get("steps", []):
            steps.append(
                StepStatusResponse(
                    step_id=s.get("step_id", ""),
                    agent_id=s.get("agent_id", ""),
                    status=StepStatus(s.get("status", "pending")),
                    started_at=s.get("started_at"),
                    completed_at=s.get("completed_at"),
                    duration_ms=s.get("duration_ms"),
                    attempts=s.get("attempts", 0),
                    error_message=s.get("error_message"),
                    output_uri=s.get("output_uri"),
                    progress_percent=s.get("progress_percent"),
                )
            )

        return RunDetailResponse(
            run_id=run.get("id", run_id),
            pipeline_id=run.get("pipeline_id", ""),
            pipeline_name=run.get("pipeline_name", ""),
            tenant_id=run.get("tenant_id", ""),
            user_id=run.get("user_id"),
            status=RunStatus(run.get("status", "pending")),
            created_at=run.get("created_at", datetime.now(timezone.utc)),
            started_at=run.get("started_at"),
            completed_at=run.get("completed_at"),
            duration_ms=run.get("duration_ms"),
            progress_percent=run.get("progress_percent", 0.0),
            steps_total=run.get("steps_total", len(steps)),
            steps_completed=run.get("steps_completed", 0),
            steps_failed=run.get("steps_failed", 0),
            steps_running=run.get("steps_running", 0),
            labels=run.get("labels", {}),
            error_message=run.get("error_message"),
            trace_id=run.get("trace_id", trace.trace_id),
            parameters=run.get("parameters", {}),
            steps=steps,
            plan_id=run.get("plan_id"),
            dry_run=run.get("dry_run", False),
            outputs=run.get("outputs"),
            lineage=run.get("lineage"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get run: {e} [{trace.trace_id}]", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_type="internal_error",
                message="Failed to get run",
                trace_id=trace.trace_id,
            ).model_dump(),
        )


@run_router.post(
    "/{run_id}/cancel",
    response_model=RunCancelResponse,
    summary="Cancel a run",
    description="Cancel a running or pending run.",
    responses={
        200: {"description": "Run canceled"},
        404: {"model": ErrorResponse, "description": "Run not found"},
        409: {"model": ErrorResponse, "description": "Run cannot be canceled"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
async def cancel_run(
    run_id: str,
    request: RunCancelRequest,
    auth: AuthContext = Depends(get_api_key),
    trace: RequestTrace = Depends(get_request_trace),
    orchestrator=Depends(get_orchestrator),
) -> RunCancelResponse:
    """
    Cancel a running or pending run.

    Running steps will be terminated gracefully unless force=True.
    """
    logger.info(f"Canceling run: {run_id} [{trace.trace_id}]")

    try:
        # Check if run exists
        run = await orchestrator.get_run(run_id)
        if run is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=create_error_response(
                    error_type="not_found",
                    message=f"Run not found: {run_id}",
                    details=[
                        ErrorDetail(
                            code=GreenLangErrorCodes.RUN_NOT_FOUND,
                            message=f"Run with ID '{run_id}' does not exist",
                        )
                    ],
                    trace_id=trace.trace_id,
                ).model_dump(),
            )

        # Check if run can be canceled
        current_status = run.get("status", "pending")
        if current_status in ["completed", "failed", "canceled"]:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=create_error_response(
                    error_type="conflict",
                    message=f"Run cannot be canceled: status is {current_status}",
                    details=[
                        ErrorDetail(
                            code=GreenLangErrorCodes.RUN_NOT_CANCELABLE,
                            message=f"Run with status '{current_status}' cannot be canceled",
                        )
                    ],
                    trace_id=trace.trace_id,
                ).model_dump(),
            )

        # Cancel the run
        canceled = await orchestrator.cancel_run(run_id, reason=request.reason)

        if canceled:
            _metrics["runs_active"] = max(0, _metrics["runs_active"] - 1)

        return RunCancelResponse(
            run_id=run_id,
            status=RunStatus.CANCELED,
            canceled_at=datetime.now(timezone.utc),
            reason=request.reason,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel run: {e} [{trace.trace_id}]", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_type="internal_error",
                message="Failed to cancel run",
                trace_id=trace.trace_id,
            ).model_dump(),
        )


@run_router.get(
    "/{run_id}/logs",
    response_model=RunLogsResponse,
    summary="Get run logs",
    description="Get logs for a pipeline run.",
    responses={
        200: {"description": "Run logs"},
        404: {"model": ErrorResponse, "description": "Run not found"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
async def get_run_logs(
    run_id: str,
    level: Optional[LogLevel] = Query(None, description="Filter by log level"),
    step_id: Optional[str] = Query(None, description="Filter by step ID"),
    since: Optional[datetime] = Query(None, description="Logs after this timestamp"),
    until: Optional[datetime] = Query(None, description="Logs before this timestamp"),
    search: Optional[str] = Query(None, description="Search in log messages"),
    cursor: Optional[str] = Query(None, description="Pagination cursor"),
    limit: int = Query(100, ge=1, le=1000, description="Max entries to return"),
    auth: AuthContext = Depends(get_api_key),
    trace: RequestTrace = Depends(get_request_trace),
    orchestrator=Depends(get_orchestrator),
) -> RunLogsResponse:
    """
    Get logs for a pipeline run with filtering.
    """
    logger.debug(f"Getting logs for run: {run_id} [{trace.trace_id}]")

    try:
        # Check if run exists
        run = await orchestrator.get_run(run_id)
        if run is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=create_error_response(
                    error_type="not_found",
                    message=f"Run not found: {run_id}",
                    details=[
                        ErrorDetail(
                            code=GreenLangErrorCodes.RUN_NOT_FOUND,
                            message=f"Run with ID '{run_id}' does not exist",
                        )
                    ],
                    trace_id=trace.trace_id,
                ).model_dump(),
            )

        # Generate sample logs (in production, this would query a log store)
        logs = [
            LogEntry(
                timestamp=datetime.now(timezone.utc),
                level=LogLevel.INFO,
                step_id=None,
                agent_id=None,
                message=f"Run {run_id} created",
                context={"run_id": run_id},
            ),
        ]

        return RunLogsResponse(
            run_id=run_id,
            logs=logs,
            total=len(logs),
            has_more=False,
            next_cursor=None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get run logs: {e} [{trace.trace_id}]", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_type="internal_error",
                message="Failed to get run logs",
                trace_id=trace.trace_id,
            ).model_dump(),
        )


@run_router.get(
    "/{run_id}/audit",
    response_model=RunAuditResponse,
    summary="Get run audit trail",
    description="Get the hash-chained audit trail for a run.",
    responses={
        200: {"description": "Run audit trail"},
        404: {"model": ErrorResponse, "description": "Run not found"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
async def get_run_audit(
    run_id: str,
    auth: AuthContext = Depends(get_api_key),
    trace: RequestTrace = Depends(get_request_trace),
    orchestrator=Depends(get_orchestrator),
    event_store=Depends(get_event_store),
) -> RunAuditResponse:
    """
    Get the hash-chained audit trail for a run.

    The audit trail provides tamper-evident compliance logging.
    """
    logger.debug(f"Getting audit trail for run: {run_id} [{trace.trace_id}]")

    try:
        # Check if run exists
        run = await orchestrator.get_run(run_id)
        if run is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=create_error_response(
                    error_type="not_found",
                    message=f"Run not found: {run_id}",
                    details=[
                        ErrorDetail(
                            code=GreenLangErrorCodes.RUN_NOT_FOUND,
                            message=f"Run with ID '{run_id}' does not exist",
                        )
                    ],
                    trace_id=trace.trace_id,
                ).model_dump(),
            )

        # Export audit package
        package = await event_store.export_audit_package(run_id)

        # Convert events to response format
        events = []
        raw_events = package.get("events", []) if isinstance(package, dict) else package.events
        for e in raw_events:
            if isinstance(e, dict):
                events.append(
                    AuditEvent(
                        event_id=e.get("event_id", ""),
                        event_type=e.get("event_type", ""),
                        timestamp=e.get("timestamp", datetime.now(timezone.utc)),
                        step_id=e.get("step_id"),
                        payload=e.get("payload", {}),
                        prev_event_hash=e.get("prev_event_hash", ""),
                        event_hash=e.get("event_hash", ""),
                    )
                )
            else:
                events.append(
                    AuditEvent(
                        event_id=e.event_id,
                        event_type=e.event_type.value if hasattr(e.event_type, 'value') else str(e.event_type),
                        timestamp=e.timestamp,
                        step_id=e.step_id,
                        payload=e.payload,
                        prev_event_hash=e.prev_event_hash,
                        event_hash=e.event_hash,
                    )
                )

        chain_valid = package.get("chain_valid", True) if isinstance(package, dict) else package.chain_valid
        exported_at = package.get("exported_at", datetime.now(timezone.utc)) if isinstance(package, dict) else package.exported_at

        return RunAuditResponse(
            run_id=run_id,
            events=events,
            chain_valid=chain_valid,
            event_count=len(events),
            exported_at=exported_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get run audit: {e} [{trace.trace_id}]", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_type="internal_error",
                message="Failed to get run audit trail",
                trace_id=trace.trace_id,
            ).model_dump(),
        )


# =============================================================================
# FR-074: CHECKPOINT AND RETRY ENDPOINTS
# =============================================================================

# Global checkpoint manager instance (initialized on startup)
_checkpoint_manager_instance = None


async def get_checkpoint_manager():
    """
    Get the CheckpointManager instance.

    Returns:
        CheckpointManager instance

    Raises:
        HTTPException: If checkpoint manager is not initialized
    """
    global _checkpoint_manager_instance

    if _checkpoint_manager_instance is None:
        try:
            from greenlang.execution.core.checkpoint_manager import (
                CheckpointManager,
                InMemoryCheckpointStore,
            )

            # Initialize with in-memory store by default
            # Production would use PostgresCheckpointStore
            store = InMemoryCheckpointStore()
            event_store = await get_event_store()
            _checkpoint_manager_instance = CheckpointManager(
                store=store,
                event_store=event_store,
            )
            logger.info("CheckpointManager initialized via dependency injection")
        except ImportError as e:
            logger.warning(f"CheckpointManager not available: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Checkpoint management service unavailable",
            )

    return _checkpoint_manager_instance


def set_checkpoint_manager(manager) -> None:
    """Set the checkpoint manager instance (for testing or custom initialization)."""
    global _checkpoint_manager_instance
    _checkpoint_manager_instance = manager


@run_router.post(
    "/{run_id}:retry",
    response_model=RunRetryResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Retry a failed run from checkpoint",
    description="Retry a failed or canceled run, optionally resuming from checkpoint.",
    responses={
        202: {"description": "Retry submitted successfully"},
        400: {"model": ErrorResponse, "description": "Invalid retry request"},
        404: {"model": ErrorResponse, "description": "Run or checkpoint not found"},
        409: {"model": ErrorResponse, "description": "Run cannot be retried"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
async def retry_run(
    run_id: str,
    request: RunRetryRequest,
    auth: AuthContext = Depends(get_api_key),
    trace: RequestTrace = Depends(get_request_trace),
    orchestrator=Depends(get_orchestrator),
    policy_engine=Depends(get_policy_engine),
) -> RunRetryResponse:
    """
    Retry a failed or canceled run.

    When from_checkpoint=True, the retry will:
    - Skip steps that completed successfully (unless in force_rerun_steps)
    - Use outputs from completed steps as inputs to dependent steps
    - Track retry lineage via parent_run_id

    Edge cases handled:
    - Non-idempotent steps: Returns warnings in response
    - Schema changes: Validates plan hash compatibility
    - Expired checkpoints: Returns appropriate error
    - Max retries exceeded: Returns error when limit reached
    """
    logger.info(f"Retry requested for run: {run_id} [{trace.trace_id}]")

    try:
        # Step 1: Check if original run exists
        run = await orchestrator.get_run(run_id)
        if run is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=create_error_response(
                    error_type="not_found",
                    message=f"Run not found: {run_id}",
                    details=[
                        ErrorDetail(
                            code=GreenLangErrorCodes.RUN_NOT_FOUND,
                            message=f"Run with ID '{run_id}' does not exist",
                        )
                    ],
                    trace_id=trace.trace_id,
                ).model_dump(),
            )

        # Step 2: Check if run can be retried (must be failed or canceled)
        current_status = run.get("status", "pending")
        if current_status not in ["failed", "canceled"]:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=create_error_response(
                    error_type="conflict",
                    message=f"Run cannot be retried: status is {current_status}",
                    details=[
                        ErrorDetail(
                            code=GreenLangErrorCodes.RUN_NOT_RETRYABLE,
                            message=f"Only failed or canceled runs can be retried. Current status: {current_status}",
                        )
                    ],
                    trace_id=trace.trace_id,
                ).model_dump(),
            )

        # Step 3: Get checkpoint manager and load checkpoint if using checkpoint
        checkpoint_manager = await get_checkpoint_manager()
        checkpoint = None
        skipped_steps = []
        steps_to_execute = []
        non_idempotent_warnings = []
        schema_compatible = True

        if request.from_checkpoint:
            checkpoint = await checkpoint_manager.get_run_checkpoint(run_id)

            if checkpoint is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=create_error_response(
                        error_type="not_found",
                        message=f"Checkpoint not found for run: {run_id}",
                        details=[
                            ErrorDetail(
                                code=GreenLangErrorCodes.CHECKPOINT_NOT_FOUND,
                                message=f"No checkpoint exists for run '{run_id}'",
                            )
                        ],
                        trace_id=trace.trace_id,
                    ).model_dump(),
                )

            # Check if checkpoint is expired
            if checkpoint.is_expired():
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=create_error_response(
                        error_type="conflict",
                        message=f"Checkpoint expired for run: {run_id}",
                        details=[
                            ErrorDetail(
                                code=GreenLangErrorCodes.CHECKPOINT_EXPIRED,
                                message=f"Checkpoint expired at {checkpoint.expires_at.isoformat()}",
                            )
                        ],
                        trace_id=trace.trace_id,
                    ).model_dump(),
                )

            # Check max retry count
            from greenlang.execution.core.checkpoint_manager import DEFAULT_MAX_RETRY_COUNT
            if checkpoint.retry_count >= DEFAULT_MAX_RETRY_COUNT:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=create_error_response(
                        error_type="conflict",
                        message=f"Max retry count exceeded for run: {run_id}",
                        details=[
                            ErrorDetail(
                                code=GreenLangErrorCodes.MAX_RETRIES_EXCEEDED,
                                message=f"Maximum retry count ({DEFAULT_MAX_RETRY_COUNT}) exceeded. Current: {checkpoint.retry_count}",
                            )
                        ],
                        trace_id=trace.trace_id,
                    ).model_dump(),
                )

            # Check schema compatibility if pipeline has changed
            pipeline = await orchestrator.get_pipeline(run.get("pipeline_id", ""))
            if pipeline:
                import json
                pipeline_data = pipeline.get("definition", {})
                content_str = json.dumps(pipeline_data, sort_keys=True)
                current_plan_hash = hashlib.sha256(content_str.encode()).hexdigest()

                schema_compatible = await checkpoint_manager.validate_schema_compatibility(
                    run_id, current_plan_hash
                )
                if not schema_compatible:
                    logger.warning(f"Schema mismatch for retry of {run_id}")

            # Determine which steps to skip and execute
            all_step_ids = [s.get("id", "") for s in pipeline.get("definition", {}).get("spec", {}).get("steps", [])]
            force_rerun_set = set(request.force_rerun_steps or [])

            if request.skip_succeeded:
                completed_steps = checkpoint.get_completed_steps()
                skipped_steps = [s for s in completed_steps if s not in force_rerun_set]
                steps_to_execute = [s for s in all_step_ids if s not in skipped_steps]
            else:
                steps_to_execute = all_step_ids

            # Check for non-idempotent steps being re-run
            step_metadata = {}
            for step in pipeline.get("definition", {}).get("spec", {}).get("steps", []):
                step_id = step.get("id", "")
                step_metadata[step_id] = {
                    "idempotency_behavior": step.get("idempotency_behavior", "unknown")
                }

            non_idempotent_step_ids = await checkpoint_manager.get_non_idempotent_steps(
                run_id, step_metadata
            )
            for step_id in non_idempotent_step_ids:
                if step_id in steps_to_execute:
                    non_idempotent_warnings.append(
                        NonIdempotentStepWarning(
                            step_id=step_id,
                            warning_message=f"Step '{step_id}' may not be idempotent and could have side effects",
                            idempotency_behavior="non_idempotent",
                            recommendation="Consider manual verification after retry"
                        )
                    )

        # Step 4: Generate new run ID
        new_run_id = f"run-retry-{str(uuid4())[:8]}"

        # Step 5: Prepare retry checkpoint if using checkpoint
        retry_count = 1
        if request.from_checkpoint and checkpoint:
            new_checkpoint = await checkpoint_manager.prepare_retry(
                original_run_id=run_id,
                new_run_id=new_run_id,
                skip_succeeded=request.skip_succeeded,
                force_rerun_steps=request.force_rerun_steps,
            )
            if new_checkpoint:
                retry_count = new_checkpoint.retry_count

        # Step 6: Submit new run to orchestrator
        parameters = request.new_parameters or run.get("parameters", {})

        await orchestrator.submit_run(
            pipeline_id=run.get("pipeline_id", ""),
            parameters=parameters,
            tenant_id=run.get("tenant_id", ""),
            user_id=run.get("user_id"),
            labels={**run.get("labels", {}), "retry_from": run_id},
            priority=run.get("priority", 5),
            dry_run=False,
            timeout_seconds=run.get("timeout_seconds"),
        )

        _metrics["runs_total"] += 1
        _metrics["runs_active"] += 1

        # Build response
        resume_step = steps_to_execute[0] if steps_to_execute else None
        message = f"Retry submitted successfully."
        if request.from_checkpoint and resume_step:
            message += f" Resuming from step '{resume_step}'."
        if skipped_steps:
            message += f" Skipping {len(skipped_steps)} completed step(s)."

        return RunRetryResponse(
            new_run_id=new_run_id,
            original_run_id=run_id,
            skipped_steps=skipped_steps,
            steps_to_execute=steps_to_execute,
            retry_count=retry_count,
            from_checkpoint=request.from_checkpoint,
            non_idempotent_warnings=non_idempotent_warnings,
            schema_compatible=schema_compatible,
            created_at=datetime.now(timezone.utc),
            message=message,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retry run: {e} [{trace.trace_id}]", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_type="internal_error",
                message="Failed to retry run",
                trace_id=trace.trace_id,
            ).model_dump(),
        )


@run_router.get(
    "/{run_id}/checkpoint",
    response_model=RunCheckpointResponse,
    summary="Get run checkpoint state",
    description="Get the current checkpoint state for a run.",
    responses={
        200: {"description": "Checkpoint state"},
        404: {"model": ErrorResponse, "description": "Run or checkpoint not found"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
async def get_run_checkpoint(
    run_id: str,
    auth: AuthContext = Depends(get_api_key),
    trace: RequestTrace = Depends(get_request_trace),
    orchestrator=Depends(get_orchestrator),
) -> RunCheckpointResponse:
    """
    Get the current checkpoint state for a run.

    Returns the complete checkpoint including:
    - Step checkpoint states (status, outputs, artifacts)
    - Retry count and lineage (parent_run_id)
    - Expiration status
    - State hash for integrity verification
    """
    logger.debug(f"Getting checkpoint for run: {run_id} [{trace.trace_id}]")

    try:
        # Check if run exists
        run = await orchestrator.get_run(run_id)
        if run is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=create_error_response(
                    error_type="not_found",
                    message=f"Run not found: {run_id}",
                    details=[
                        ErrorDetail(
                            code=GreenLangErrorCodes.RUN_NOT_FOUND,
                            message=f"Run with ID '{run_id}' does not exist",
                        )
                    ],
                    trace_id=trace.trace_id,
                ).model_dump(),
            )

        # Get checkpoint
        checkpoint_manager = await get_checkpoint_manager()
        checkpoint = await checkpoint_manager.get_run_checkpoint(run_id)

        if checkpoint is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=create_error_response(
                    error_type="not_found",
                    message=f"Checkpoint not found for run: {run_id}",
                    details=[
                        ErrorDetail(
                            code=GreenLangErrorCodes.CHECKPOINT_NOT_FOUND,
                            message=f"No checkpoint exists for run '{run_id}'",
                        )
                    ],
                    trace_id=trace.trace_id,
                ).model_dump(),
            )

        # Convert step checkpoints to response format
        step_checkpoints = {}
        for step_id, state in checkpoint.step_checkpoints.items():
            step_checkpoints[step_id] = StepCheckpointResponse(
                step_id=state.step_id,
                status=CheckpointStatusEnum(state.status.value),
                outputs=state.outputs,
                artifacts=state.artifacts,
                idempotency_key=state.idempotency_key,
                attempt=state.attempt,
                error_message=state.error_message,
                started_at=state.started_at,
                completed_at=state.completed_at,
            )

        return RunCheckpointResponse(
            run_id=checkpoint.run_id,
            plan_id=checkpoint.plan_id,
            plan_hash=checkpoint.plan_hash,
            pipeline_id=checkpoint.pipeline_id,
            step_checkpoints=step_checkpoints,
            last_successful_step=checkpoint.last_successful_step,
            retry_count=checkpoint.retry_count,
            parent_run_id=checkpoint.parent_run_id,
            created_at=checkpoint.created_at,
            updated_at=checkpoint.updated_at,
            expires_at=checkpoint.expires_at,
            is_expired=checkpoint.is_expired(),
            completed_steps=checkpoint.get_completed_steps(),
            failed_steps=checkpoint.get_failed_steps(),
            state_hash=checkpoint.compute_state_hash(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get checkpoint: {e} [{trace.trace_id}]", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_type="internal_error",
                message="Failed to get checkpoint",
                trace_id=trace.trace_id,
            ).model_dump(),
        )


@run_router.delete(
    "/{run_id}/checkpoint",
    response_model=CheckpointClearResponse,
    summary="Clear run checkpoint",
    description="Delete the checkpoint for a run.",
    responses={
        200: {"description": "Checkpoint cleared"},
        404: {"model": ErrorResponse, "description": "Run or checkpoint not found"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
async def clear_run_checkpoint(
    run_id: str,
    auth: AuthContext = Depends(get_api_key),
    trace: RequestTrace = Depends(get_request_trace),
    orchestrator=Depends(get_orchestrator),
) -> CheckpointClearResponse:
    """
    Delete the checkpoint for a run.

    This operation is irreversible. Once cleared, the run cannot be
    retried from checkpoint.
    """
    logger.info(f"Clearing checkpoint for run: {run_id} [{trace.trace_id}]")

    try:
        # Check if run exists
        run = await orchestrator.get_run(run_id)
        if run is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=create_error_response(
                    error_type="not_found",
                    message=f"Run not found: {run_id}",
                    details=[
                        ErrorDetail(
                            code=GreenLangErrorCodes.RUN_NOT_FOUND,
                            message=f"Run with ID '{run_id}' does not exist",
                        )
                    ],
                    trace_id=trace.trace_id,
                ).model_dump(),
            )

        # Clear checkpoint
        checkpoint_manager = await get_checkpoint_manager()
        cleared = await checkpoint_manager.clear_checkpoint(run_id)

        if not cleared:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=create_error_response(
                    error_type="not_found",
                    message=f"Checkpoint not found for run: {run_id}",
                    details=[
                        ErrorDetail(
                            code=GreenLangErrorCodes.CHECKPOINT_NOT_FOUND,
                            message=f"No checkpoint exists for run '{run_id}'",
                        )
                    ],
                    trace_id=trace.trace_id,
                ).model_dump(),
            )

        return CheckpointClearResponse(
            run_id=run_id,
            cleared=True,
            message=f"Checkpoint for run '{run_id}' has been cleared successfully.",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear checkpoint: {e} [{trace.trace_id}]", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_type="internal_error",
                message="Failed to clear checkpoint",
                trace_id=trace.trace_id,
            ).model_dump(),
        )


# =============================================================================
# HEALTH AND METRICS ENDPOINTS
# =============================================================================


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health status of the API and its dependencies.",
    tags=["System"],
    responses={
        200: {"description": "Service is healthy"},
        503: {"description": "Service is unhealthy"},
    },
)
async def health_check(
    config: APIConfig = Depends(get_config),
) -> HealthResponse:
    """
    Health check endpoint for load balancers and monitoring.
    """
    components = []
    overall_healthy = True

    # Check orchestrator
    try:
        orchestrator = await get_orchestrator()
        components.append(
            ComponentHealth(
                name="orchestrator",
                status=HealthStatus.HEALTHY,
                latency_ms=1.0,
                last_checked=datetime.now(timezone.utc),
            )
        )
    except Exception as e:
        logger.error(f"Orchestrator health check failed: {e}")
        components.append(
            ComponentHealth(
                name="orchestrator",
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                last_checked=datetime.now(timezone.utc),
            )
        )
        overall_healthy = False

    # Check agent registry
    try:
        registry = await get_agent_registry()
        components.append(
            ComponentHealth(
                name="agent_registry",
                status=HealthStatus.HEALTHY,
                latency_ms=0.5,
                last_checked=datetime.now(timezone.utc),
            )
        )
    except Exception as e:
        logger.error(f"Agent registry health check failed: {e}")
        components.append(
            ComponentHealth(
                name="agent_registry",
                status=HealthStatus.DEGRADED,
                message=str(e),
                last_checked=datetime.now(timezone.utc),
            )
        )

    # Check event store
    try:
        store = await get_event_store()
        components.append(
            ComponentHealth(
                name="event_store",
                status=HealthStatus.HEALTHY,
                latency_ms=2.0,
                last_checked=datetime.now(timezone.utc),
            )
        )
    except Exception as e:
        logger.error(f"Event store health check failed: {e}")
        components.append(
            ComponentHealth(
                name="event_store",
                status=HealthStatus.DEGRADED,
                message=str(e),
                last_checked=datetime.now(timezone.utc),
            )
        )

    status_value = HealthStatus.HEALTHY if overall_healthy else HealthStatus.UNHEALTHY

    response = HealthResponse(
        status=status_value,
        version=config.api_version,
        timestamp=datetime.now(timezone.utc),
        components=components,
    )

    if not overall_healthy:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=response.model_dump(mode="json"),
        )

    return response


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    description="Check if the service is ready to accept traffic.",
    tags=["System"],
    responses={
        200: {"description": "Service is ready"},
        503: {"description": "Service is not ready"},
    },
)
async def readiness_probe() -> ReadinessResponse:
    """
    Readiness probe for Kubernetes.

    Returns ready only if all critical components are initialized.
    """
    try:
        # Check that critical dependencies are available
        await get_orchestrator()
        await get_agent_registry()

        return ReadinessResponse(ready=True, message="Service is ready")

    except Exception as e:
        logger.warning(f"Readiness check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ReadinessResponse(ready=False, message=str(e)).model_dump(),
        )


@router.get(
    "/metrics",
    response_class=PlainTextResponse,
    summary="Prometheus metrics",
    description="Get Prometheus-formatted metrics.",
    tags=["System"],
)
async def get_metrics(
    format: str = Query("prometheus", description="Output format: prometheus or json"),
) -> Response:
    """
    Get metrics in Prometheus format or JSON.
    """
    uptime = time.time() - _metrics["start_time"]

    if format == "json":
        return JSONResponse(
            content=MetricsResponse(
                runs_total=_metrics["runs_total"],
                runs_active=_metrics["runs_active"],
                runs_succeeded=_metrics["runs_succeeded"],
                runs_failed=_metrics["runs_failed"],
                steps_total=_metrics["steps_total"],
                steps_succeeded=_metrics["steps_succeeded"],
                steps_failed=_metrics["steps_failed"],
                avg_run_duration_ms=0.0,  # Would calculate from actual data
                avg_step_duration_ms=0.0,
                policy_evaluations=_metrics["policy_evaluations"],
                policy_denials=_metrics["policy_denials"],
                uptime_seconds=uptime,
                timestamp=datetime.now(timezone.utc),
            ).model_dump(mode="json")
        )

    # Prometheus format
    prometheus_output = f"""# HELP greenlang_runs_total Total number of runs
# TYPE greenlang_runs_total counter
greenlang_runs_total {_metrics["runs_total"]}

# HELP greenlang_runs_active Currently active runs
# TYPE greenlang_runs_active gauge
greenlang_runs_active {_metrics["runs_active"]}

# HELP greenlang_runs_succeeded Successful runs
# TYPE greenlang_runs_succeeded counter
greenlang_runs_succeeded {_metrics["runs_succeeded"]}

# HELP greenlang_runs_failed Failed runs
# TYPE greenlang_runs_failed counter
greenlang_runs_failed {_metrics["runs_failed"]}

# HELP greenlang_steps_total Total steps executed
# TYPE greenlang_steps_total counter
greenlang_steps_total {_metrics["steps_total"]}

# HELP greenlang_policy_evaluations Total policy evaluations
# TYPE greenlang_policy_evaluations counter
greenlang_policy_evaluations {_metrics["policy_evaluations"]}

# HELP greenlang_policy_denials Policy denials
# TYPE greenlang_policy_denials counter
greenlang_policy_denials {_metrics["policy_denials"]}

# HELP greenlang_uptime_seconds Service uptime in seconds
# TYPE greenlang_uptime_seconds gauge
greenlang_uptime_seconds {uptime}
"""

    # FR-024: Add quota metrics if quota manager is available
    try:
        quota_manager = await get_quota_manager()
        quota_prometheus = quota_manager.get_prometheus_metrics()
        prometheus_output += "\n" + quota_prometheus
    except Exception as e:
        logger.debug(f"Quota metrics not available: {e}")

    return PlainTextResponse(content=prometheus_output, media_type="text/plain")


# =============================================================================
# AGENT REGISTRY ENDPOINTS
# =============================================================================


@agent_router.get(
    "",
    response_model=AgentListResponse,
    summary="List agents",
    description="List registered agents with filtering.",
    responses={
        200: {"description": "List of agents"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
async def list_agents(
    layer: Optional[str] = Query(None, description="Filter by layer"),
    sector: Optional[str] = Query(None, description="Filter by sector"),
    capability: Optional[str] = Query(None, description="Filter by capability"),
    execution_mode: Optional[str] = Query(None, description="Filter by execution mode"),
    glip_compatible: Optional[bool] = Query(None, description="Filter GLIP v1 compatible agents"),
    health_status: Optional[str] = Query(None, description="Filter by health status"),
    search: Optional[str] = Query(None, description="Search in name/description"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(100, ge=1, le=1000, description="Pagination limit"),
    auth: AuthContext = Depends(get_api_key),
    trace: RequestTrace = Depends(get_request_trace),
    agent_registry=Depends(get_agent_registry),
) -> AgentListResponse:
    """
    List registered agents with filtering.
    """
    logger.debug(f"Listing agents [{trace.trace_id}]")

    try:
        # Build query
        query_params = {
            "limit": limit,
            "offset": offset,
        }
        if layer:
            query_params["layer"] = layer
        if sector:
            query_params["sector"] = sector
        if capability:
            query_params["capability"] = capability
        if execution_mode:
            query_params["execution_mode"] = execution_mode
        if glip_compatible is not None:
            query_params["glip_compatible"] = glip_compatible
        if health_status:
            query_params["health_status"] = health_status
        if search:
            query_params["search_text"] = search

        # Query registry
        # Note: This adapts to the VersionedAgentRegistry API
        try:
            from greenlang.agents.foundation.agent_registry import RegistryQueryInput
            query = RegistryQueryInput(**query_params)
            result = agent_registry.query_agents(query)
            raw_agents = result.agents
            total = result.total_count
        except ImportError:
            # Use mock registry
            result = agent_registry.query_agents(query_params)
            raw_agents = result.agents if hasattr(result, 'agents') else []
            total = result.total_count if hasattr(result, 'total_count') else len(raw_agents)

        # Convert to response format
        agents = []
        for a in raw_agents:
            if isinstance(a, dict):
                agents.append(
                    AgentResponse(
                        agent_id=a.get("agent_id", ""),
                        name=a.get("name", ""),
                        description=a.get("description", ""),
                        version=a.get("version", ""),
                        layer=a.get("layer", ""),
                        sectors=a.get("sectors", []),
                        capabilities=[],
                        execution_mode=a.get("execution_mode", "legacy_http"),
                        health_status=a.get("health_status", "unknown"),
                        is_glip_compatible=a.get("is_glip_compatible", False),
                        is_idempotent=a.get("is_idempotent", False),
                        deterministic=a.get("deterministic", True),
                        tags=a.get("tags", []),
                    )
                )
            else:
                # AgentMetadataEntry object
                capabilities = []
                for c in getattr(a, 'capabilities', []):
                    capabilities.append(
                        AgentCapabilityResponse(
                            name=c.name,
                            category=c.category.value if hasattr(c.category, 'value') else str(c.category),
                            description=c.description,
                            input_types=c.input_types,
                            output_types=c.output_types,
                        )
                    )

                agents.append(
                    AgentResponse(
                        agent_id=a.agent_id,
                        name=a.name,
                        description=a.description,
                        version=a.version,
                        layer=a.layer.value if hasattr(a.layer, 'value') else str(a.layer),
                        sectors=[s.value if hasattr(s, 'value') else str(s) for s in a.sectors],
                        capabilities=capabilities,
                        execution_mode=a.execution_mode.value if hasattr(a.execution_mode, 'value') else str(a.execution_mode),
                        health_status=a.health_status.value if hasattr(a.health_status, 'value') else str(a.health_status),
                        is_glip_compatible=a.is_glip_compatible,
                        is_idempotent=a.is_idempotent,
                        deterministic=a.deterministic,
                        tags=a.tags,
                    )
                )

        return AgentListResponse(
            agents=agents,
            total=total,
            offset=offset,
            limit=limit,
        )

    except Exception as e:
        logger.error(f"Failed to list agents: {e} [{trace.trace_id}]", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_type="internal_error",
                message="Failed to list agents",
                trace_id=trace.trace_id,
            ).model_dump(),
        )


@agent_router.get(
    "/{agent_id}",
    response_model=AgentDetailResponse,
    summary="Get agent details",
    description="Get detailed information about a specific agent.",
    responses={
        200: {"description": "Agent details"},
        404: {"model": ErrorResponse, "description": "Agent not found"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
async def get_agent(
    agent_id: str,
    version: Optional[str] = Query(None, description="Specific version (default: latest)"),
    auth: AuthContext = Depends(get_api_key),
    trace: RequestTrace = Depends(get_request_trace),
    agent_registry=Depends(get_agent_registry),
) -> AgentDetailResponse:
    """
    Get detailed information about an agent.
    """
    logger.debug(f"Getting agent: {agent_id} [{trace.trace_id}]")

    try:
        agent = agent_registry.get_agent(agent_id, version)

        if agent is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=create_error_response(
                    error_type="not_found",
                    message=f"Agent not found: {agent_id}",
                    details=[
                        ErrorDetail(
                            code=GreenLangErrorCodes.AGENT_NOT_FOUND,
                            message=f"Agent with ID '{agent_id}' does not exist",
                        )
                    ],
                    trace_id=trace.trace_id,
                ).model_dump(),
            )

        # Convert to response format
        if isinstance(agent, dict):
            return AgentDetailResponse(
                agent_id=agent.get("agent_id", agent_id),
                name=agent.get("name", ""),
                description=agent.get("description", ""),
                version=agent.get("version", ""),
                layer=agent.get("layer", ""),
                sectors=agent.get("sectors", []),
                capabilities=[],
                execution_mode=agent.get("execution_mode", "legacy_http"),
                health_status=agent.get("health_status", "unknown"),
                is_glip_compatible=agent.get("is_glip_compatible", False),
                is_idempotent=agent.get("is_idempotent", False),
                deterministic=agent.get("deterministic", True),
                tags=agent.get("tags", []),
                resource_profile=agent.get("resource_profile"),
                container_spec=agent.get("container_spec"),
                dependencies=agent.get("dependencies", []),
                variants=agent.get("variants", []),
                registered_at=agent.get("registered_at", datetime.now(timezone.utc)),
                updated_at=agent.get("updated_at", datetime.now(timezone.utc)),
            )
        else:
            # AgentMetadataEntry object
            capabilities = []
            for c in getattr(agent, 'capabilities', []):
                capabilities.append(
                    AgentCapabilityResponse(
                        name=c.name,
                        category=c.category.value if hasattr(c.category, 'value') else str(c.category),
                        description=c.description,
                        input_types=c.input_types,
                        output_types=c.output_types,
                    )
                )

            return AgentDetailResponse(
                agent_id=agent.agent_id,
                name=agent.name,
                description=agent.description,
                version=agent.version,
                layer=agent.layer.value if hasattr(agent.layer, 'value') else str(agent.layer),
                sectors=[s.value if hasattr(s, 'value') else str(s) for s in agent.sectors],
                capabilities=capabilities,
                execution_mode=agent.execution_mode.value if hasattr(agent.execution_mode, 'value') else str(agent.execution_mode),
                health_status=agent.health_status.value if hasattr(agent.health_status, 'value') else str(agent.health_status),
                is_glip_compatible=agent.is_glip_compatible,
                is_idempotent=agent.is_idempotent,
                deterministic=agent.deterministic,
                tags=agent.tags,
                resource_profile=agent.resource_profile.model_dump() if agent.resource_profile else None,
                container_spec=agent.container_spec.model_dump() if agent.container_spec else None,
                dependencies=[d.model_dump() for d in agent.dependencies],
                variants=[v.model_dump() for v in agent.variants],
                registered_at=agent.registered_at,
                updated_at=agent.updated_at,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent: {e} [{trace.trace_id}]", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_type="internal_error",
                message="Failed to get agent",
                trace_id=trace.trace_id,
            ).model_dump(),
        )


# =============================================================================
# FR-024: QUOTA MANAGEMENT ENDPOINTS
# =============================================================================

# Global quota manager instance (initialized on startup)
_quota_manager_instance = None


async def get_quota_manager():
    """
    Get the QuotaManager instance.

    Returns:
        QuotaManager instance

    Raises:
        HTTPException: If quota manager is not initialized
    """
    global _quota_manager_instance

    if _quota_manager_instance is None:
        try:
            from greenlang.orchestrator.quotas import QuotaManager

            _quota_manager_instance = QuotaManager()
            # Try to load from default config
            try:
                import os
                config_path = os.environ.get(
                    "QUOTA_CONFIG_PATH",
                    "config/namespace_quotas.yaml"
                )
                _quota_manager_instance.load_from_yaml(config_path)
            except Exception as e:
                logger.warning(f"Could not load quota config: {e}")

            logger.info("QuotaManager initialized via dependency injection")
        except ImportError as e:
            logger.warning(f"QuotaManager not available: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Quota management service unavailable",
            )

    return _quota_manager_instance


def set_quota_manager(manager) -> None:
    """Set the quota manager instance (for testing or custom initialization)."""
    global _quota_manager_instance
    _quota_manager_instance = manager


# Pydantic models for quota API
from pydantic import BaseModel, Field
from typing import Dict, List, Any


class QuotaConfigRequest(BaseModel):
    """Request model for updating namespace quotas."""
    max_concurrent_runs: int = Field(
        default=20, ge=1, le=1000, description="Maximum concurrent runs"
    )
    max_concurrent_steps: int = Field(
        default=100, ge=1, le=10000, description="Maximum concurrent steps"
    )
    max_queued_runs: int = Field(
        default=50, ge=0, le=10000, description="Maximum queued runs"
    )
    priority_weight: float = Field(
        default=1.0, ge=0.1, le=10.0, description="Priority weight"
    )
    queue_timeout_seconds: float = Field(
        default=300.0, ge=10.0, le=86400.0, description="Queue timeout in seconds"
    )


class QuotaConfigResponse(BaseModel):
    """Response model for quota configuration."""
    namespace: str = Field(..., description="Namespace identifier")
    max_concurrent_runs: int = Field(..., description="Maximum concurrent runs")
    max_concurrent_steps: int = Field(..., description="Maximum concurrent steps")
    max_queued_runs: int = Field(..., description="Maximum queued runs")
    priority_weight: float = Field(..., description="Priority weight")
    queue_timeout_seconds: float = Field(..., description="Queue timeout in seconds")


class QuotaUsageResponse(BaseModel):
    """Response model for quota usage."""
    namespace: str = Field(..., description="Namespace identifier")
    current_runs: int = Field(..., description="Current active runs")
    current_steps: int = Field(..., description="Current active steps")
    queued_runs: int = Field(..., description="Runs waiting in queue")
    active_run_ids: List[str] = Field(..., description="Active run IDs")
    last_updated: str = Field(..., description="Last update timestamp")
    total_runs_started: int = Field(..., description="Total runs started")
    total_runs_completed: int = Field(..., description="Total runs completed")
    total_queue_timeouts: int = Field(..., description="Total queue timeouts")


class QuotaMetricsResponse(BaseModel):
    """Response model for quota metrics."""
    namespace: str = Field(..., description="Namespace identifier")
    quota_usage_percent: float = Field(..., description="Overall quota utilization")
    queue_depth: int = Field(..., description="Queue depth")
    queue_wait_time_seconds: float = Field(..., description="Average queue wait time")
    runs_utilization_percent: float = Field(..., description="Run slots utilization")
    steps_utilization_percent: float = Field(..., description="Step slots utilization")


class AllQuotasResponse(BaseModel):
    """Response model for listing all quotas."""
    quotas: List[QuotaConfigResponse] = Field(..., description="List of quota configs")
    total: int = Field(..., description="Total count")


@quota_router.get(
    "",
    response_model=AllQuotasResponse,
    summary="List all namespace quotas",
    description="Get quota configuration for all namespaces.",
    responses={
        200: {"description": "List of quotas"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
async def list_quotas(
    auth: AuthContext = Depends(get_api_key),
    trace: RequestTrace = Depends(get_request_trace),
) -> AllQuotasResponse:
    """
    List quota configurations for all namespaces.

    Returns both explicitly configured namespaces and the default quota.
    """
    logger.debug(f"Listing all quotas [{trace.trace_id}]")

    try:
        quota_manager = await get_quota_manager()
        all_quotas = quota_manager.get_all_quotas()

        quotas_list = []
        for namespace, config in all_quotas.items():
            quotas_list.append(
                QuotaConfigResponse(
                    namespace=namespace,
                    max_concurrent_runs=config.max_concurrent_runs,
                    max_concurrent_steps=config.max_concurrent_steps,
                    max_queued_runs=config.max_queued_runs,
                    priority_weight=config.priority_weight,
                    queue_timeout_seconds=config.queue_timeout_seconds,
                )
            )

        return AllQuotasResponse(quotas=quotas_list, total=len(quotas_list))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list quotas: {e} [{trace.trace_id}]", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_type="internal_error",
                message="Failed to list quotas",
                trace_id=trace.trace_id,
            ).model_dump(),
        )


@namespace_quota_router.get(
    "/{namespace}/quota",
    response_model=QuotaConfigResponse,
    summary="Get namespace quota",
    description="Get quota configuration for a specific namespace.",
    responses={
        200: {"description": "Quota configuration"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
async def get_namespace_quota(
    namespace: str,
    auth: AuthContext = Depends(get_api_key),
    trace: RequestTrace = Depends(get_request_trace),
) -> QuotaConfigResponse:
    """
    Get quota configuration for a specific namespace.

    If no explicit quota is configured, returns the default quota.
    """
    logger.debug(f"Getting quota for namespace: {namespace} [{trace.trace_id}]")

    try:
        quota_manager = await get_quota_manager()
        config = quota_manager.get_quota(namespace)

        return QuotaConfigResponse(
            namespace=namespace,
            max_concurrent_runs=config.max_concurrent_runs,
            max_concurrent_steps=config.max_concurrent_steps,
            max_queued_runs=config.max_queued_runs,
            priority_weight=config.priority_weight,
            queue_timeout_seconds=config.queue_timeout_seconds,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get quota: {e} [{trace.trace_id}]", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_type="internal_error",
                message="Failed to get namespace quota",
                trace_id=trace.trace_id,
            ).model_dump(),
        )


@namespace_quota_router.put(
    "/{namespace}/quota",
    response_model=QuotaConfigResponse,
    summary="Update namespace quota",
    description="Update quota configuration for a namespace (admin only).",
    responses={
        200: {"description": "Updated quota configuration"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Forbidden - admin only"},
    },
)
async def update_namespace_quota(
    namespace: str,
    request: QuotaConfigRequest,
    auth: AuthContext = Depends(get_api_key),
    trace: RequestTrace = Depends(get_request_trace),
) -> QuotaConfigResponse:
    """
    Update quota configuration for a namespace.

    Requires admin scope.
    """
    logger.info(f"Updating quota for namespace: {namespace} [{trace.trace_id}]")

    # Check admin permission
    if not auth.has_scope("admin") and not auth.has_scope("quotas:write"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=create_error_response(
                error_type="forbidden",
                message="Admin scope required to update quotas",
                details=[
                    ErrorDetail(
                        code=GreenLangErrorCodes.FORBIDDEN,
                        message="Requires 'admin' or 'quotas:write' scope",
                    )
                ],
                trace_id=trace.trace_id,
            ).model_dump(),
        )

    try:
        quota_manager = await get_quota_manager()

        # Import QuotaConfig from the quotas module
        from greenlang.orchestrator.quotas import QuotaConfig

        config = QuotaConfig(
            max_concurrent_runs=request.max_concurrent_runs,
            max_concurrent_steps=request.max_concurrent_steps,
            max_queued_runs=request.max_queued_runs,
            priority_weight=request.priority_weight,
            queue_timeout_seconds=request.queue_timeout_seconds,
        )

        quota_manager.set_quota(namespace, config)

        return QuotaConfigResponse(
            namespace=namespace,
            max_concurrent_runs=config.max_concurrent_runs,
            max_concurrent_steps=config.max_concurrent_steps,
            max_queued_runs=config.max_queued_runs,
            priority_weight=config.priority_weight,
            queue_timeout_seconds=config.queue_timeout_seconds,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update quota: {e} [{trace.trace_id}]", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_type="internal_error",
                message="Failed to update namespace quota",
                trace_id=trace.trace_id,
            ).model_dump(),
        )


@namespace_quota_router.get(
    "/{namespace}/quota/usage",
    response_model=QuotaUsageResponse,
    summary="Get namespace quota usage",
    description="Get current usage metrics for a namespace.",
    responses={
        200: {"description": "Current usage metrics"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
async def get_namespace_quota_usage(
    namespace: str,
    auth: AuthContext = Depends(get_api_key),
    trace: RequestTrace = Depends(get_request_trace),
) -> QuotaUsageResponse:
    """
    Get current usage metrics for a namespace.

    Returns real-time information about active runs, steps, and queue status.
    """
    logger.debug(f"Getting quota usage for namespace: {namespace} [{trace.trace_id}]")

    try:
        quota_manager = await get_quota_manager()
        usage = quota_manager.get_usage(namespace)

        return QuotaUsageResponse(
            namespace=namespace,
            current_runs=usage.current_runs,
            current_steps=usage.current_steps,
            queued_runs=usage.queued_runs,
            active_run_ids=list(usage.active_run_ids),
            last_updated=usage.last_updated.isoformat(),
            total_runs_started=usage.total_runs_started,
            total_runs_completed=usage.total_runs_completed,
            total_queue_timeouts=usage.total_queue_timeouts,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get quota usage: {e} [{trace.trace_id}]", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                error_type="internal_error",
                message="Failed to get namespace quota usage",
                trace_id=trace.trace_id,
            ).model_dump(),
        )


# =============================================================================
# FR-063: WEBHOOK MANAGEMENT ENDPOINTS
# =============================================================================

webhook_router = APIRouter(prefix="/namespaces", tags=["Webhook Management"])

_alert_manager_instance = None


async def get_alert_manager():
    """Get or create the AlertManager instance."""
    global _alert_manager_instance
    if _alert_manager_instance is None:
        try:
            from greenlang.orchestrator.alerting import AlertManager
            _alert_manager_instance = AlertManager()
            try:
                import os
                config_path = os.environ.get("GREENLANG_ALERTING_CONFIG", "config/alerting.yaml")
                if os.path.exists(config_path):
                    await _alert_manager_instance.load_config(config_path)
            except Exception as e:
                logger.warning(f"Could not load alerting config: {e}")
            logger.info("AlertManager initialized via dependency injection")
        except ImportError as e:
            logger.warning(f"AlertManager not available: {e}")
            _alert_manager_instance = None
    return _alert_manager_instance


def set_alert_manager(manager) -> None:
    """Set the alert manager instance (for testing)."""
    global _alert_manager_instance
    _alert_manager_instance = manager


from pydantic import BaseModel as PydanticBaseModel


class WebhookRegisterRequest(PydanticBaseModel):
    """Request to register a new webhook."""
    name: str = Field(..., description="Webhook name")
    provider: str = Field(default="custom", description="Provider type")
    url: Optional[str] = Field(None, description="Webhook URL")
    secret: Optional[str] = Field(None, description="HMAC signing secret")
    routing_key: Optional[str] = Field(None, description="Routing key (PagerDuty)")
    events: List[str] = Field(default_factory=list, description="Event types")
    severity_threshold: str = Field(default="medium", description="Severity threshold")
    retries: int = Field(default=3, ge=0, le=10, description="Retry attempts")
    timeout_seconds: int = Field(default=30, ge=1, le=300, description="Timeout")
    enabled: bool = Field(default=True, description="Whether enabled")
    headers: Dict[str, str] = Field(default_factory=dict, description="HTTP headers")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata")


class WebhookResponse(PydanticBaseModel):
    """Response for webhook operations."""
    webhook_id: str = Field(..., description="Webhook ID")
    name: str = Field(..., description="Webhook name")
    provider: str = Field(..., description="Provider type")
    events: List[str] = Field(..., description="Subscribed events")
    severity_threshold: str = Field(..., description="Severity threshold")
    enabled: bool = Field(..., description="Whether enabled")
    retries: int = Field(..., description="Retry attempts")
    timeout_seconds: int = Field(..., description="Request timeout")


class WebhookListResponse(PydanticBaseModel):
    """Response for listing webhooks."""
    webhooks: List[WebhookResponse] = Field(..., description="List of webhooks")
    total: int = Field(..., description="Total count")
    namespace: str = Field(..., description="Namespace")


class TestAlertRequest(PydanticBaseModel):
    """Request to send a test alert."""
    namespace: str = Field(..., description="Namespace")
    webhook_id: Optional[str] = Field(None, description="Specific webhook")


class TestAlertResponse(PydanticBaseModel):
    """Response for test alert."""
    success: bool = Field(..., description="Whether successful")
    results: List[Dict[str, Any]] = Field(..., description="Delivery results")
    message: str = Field(..., description="Status message")


@webhook_router.get(
    "/{namespace}/webhooks",
    response_model=WebhookListResponse,
    summary="List webhooks for namespace",
    responses={200: {"description": "List of webhooks"}, 401: {"model": ErrorResponse}},
)
async def list_webhooks(
    namespace: str,
    auth: AuthContext = Depends(get_api_key),
    trace: RequestTrace = Depends(get_request_trace),
) -> WebhookListResponse:
    """List all webhooks configured for a namespace."""
    logger.debug(f"Listing webhooks for namespace '{namespace}' [{trace.trace_id}]")
    try:
        alert_manager = await get_alert_manager()
        if alert_manager is None:
            return WebhookListResponse(webhooks=[], total=0, namespace=namespace)
        webhooks = alert_manager.list_webhooks(namespace)
        webhook_responses = [
            WebhookResponse(
                webhook_id=wh.webhook_id, name=wh.name, provider=wh.provider,
                events=wh.events, severity_threshold=wh.severity_threshold.value,
                enabled=wh.enabled, retries=wh.retries, timeout_seconds=wh.timeout_seconds,
            )
            for wh in webhooks
        ]
        return WebhookListResponse(webhooks=webhook_responses, total=len(webhook_responses), namespace=namespace)
    except Exception as e:
        logger.error(f"Failed to list webhooks: {e} [{trace.trace_id}]", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(error_type="internal_error", message="Failed to list webhooks", trace_id=trace.trace_id).model_dump())


@webhook_router.post(
    "/{namespace}/webhooks",
    response_model=WebhookResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a webhook",
    responses={201: {"description": "Webhook registered"}, 401: {"model": ErrorResponse}},
)
async def register_webhook(
    namespace: str,
    request: WebhookRegisterRequest,
    auth: AuthContext = Depends(get_api_key),
    trace: RequestTrace = Depends(get_request_trace),
) -> WebhookResponse:
    """Register a new webhook for a namespace."""
    logger.info(f"Registering webhook '{request.name}' for '{namespace}' [{trace.trace_id}]")
    try:
        alert_manager = await get_alert_manager()
        if alert_manager is None:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=create_error_response(error_type="service_unavailable", message="Alert manager not available", trace_id=trace.trace_id).model_dump())
        from greenlang.orchestrator.alerting import AlertSeverity, WebhookConfig
        try:
            severity_threshold = AlertSeverity(request.severity_threshold.lower())
        except ValueError:
            severity_threshold = AlertSeverity.MEDIUM
        webhook_config = WebhookConfig(
            name=request.name, provider=request.provider, url=request.url, secret=request.secret,
            routing_key=request.routing_key, events=request.events, severity_threshold=severity_threshold,
            retries=request.retries, timeout_seconds=request.timeout_seconds, enabled=request.enabled,
            headers=request.headers, metadata=request.metadata,
        )
        webhook_id = alert_manager.register_webhook(namespace, webhook_config)
        logger.info(f"Webhook registered: {webhook_id} [{trace.trace_id}]")
        return WebhookResponse(
            webhook_id=webhook_id, name=webhook_config.name, provider=webhook_config.provider,
            events=webhook_config.events, severity_threshold=webhook_config.severity_threshold.value,
            enabled=webhook_config.enabled, retries=webhook_config.retries, timeout_seconds=webhook_config.timeout_seconds,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to register webhook: {e} [{trace.trace_id}]", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(error_type="internal_error", message="Failed to register webhook", trace_id=trace.trace_id).model_dump())


@webhook_router.delete(
    "/{namespace}/webhooks/{webhook_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a webhook",
    responses={204: {"description": "Webhook deleted"}, 404: {"model": ErrorResponse}},
)
async def delete_webhook(
    namespace: str,
    webhook_id: str,
    auth: AuthContext = Depends(get_api_key),
    trace: RequestTrace = Depends(get_request_trace),
) -> Response:
    """Unregister a webhook from a namespace."""
    logger.info(f"Deleting webhook '{webhook_id}' from '{namespace}' [{trace.trace_id}]")
    try:
        alert_manager = await get_alert_manager()
        if alert_manager is None:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=create_error_response(error_type="service_unavailable", message="Alert manager not available", trace_id=trace.trace_id).model_dump())
        deleted = alert_manager.unregister_webhook(namespace, webhook_id)
        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                detail=create_error_response(error_type="not_found", message=f"Webhook not found: {webhook_id}",
                    details=[ErrorDetail(code="GL-E-RES-004", message=f"Webhook '{webhook_id}' not in namespace '{namespace}'")],
                    trace_id=trace.trace_id).model_dump())
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete webhook: {e} [{trace.trace_id}]", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(error_type="internal_error", message="Failed to delete webhook", trace_id=trace.trace_id).model_dump())


@router.post(
    "/api/v1/webhooks/test",
    response_model=TestAlertResponse,
    summary="Send test alert",
    tags=["Webhook Management"],
    responses={200: {"description": "Test alert sent"}, 401: {"model": ErrorResponse}},
)
async def send_test_alert(
    request: TestAlertRequest,
    auth: AuthContext = Depends(get_api_key),
    trace: RequestTrace = Depends(get_request_trace),
) -> TestAlertResponse:
    """Send a test alert to verify webhook configuration."""
    logger.info(f"Sending test alert to '{request.namespace}' [{trace.trace_id}]")
    try:
        alert_manager = await get_alert_manager()
        if alert_manager is None:
            return TestAlertResponse(success=False, results=[], message="Alert manager not available")
        results = await alert_manager.send_test_alert(namespace=request.namespace, webhook_id=request.webhook_id)
        result_dicts = [{"webhook_id": r.webhook_id, "status": r.status.value, "status_code": r.status_code,
            "attempts": r.attempts, "error_message": r.error_message, "latency_ms": r.latency_ms} for r in results]
        success = all(r.status.value == "delivered" for r in results) if results else False
        delivered = sum(1 for r in results if r.status.value == "delivered")
        message = f"Test alert sent to {delivered}/{len(results)} webhooks" if results else "No webhooks configured"
        return TestAlertResponse(success=success, results=result_dicts, message=message)
    except Exception as e:
        logger.error(f"Failed to send test alert: {e} [{trace.trace_id}]", exc_info=True)
        return TestAlertResponse(success=False, results=[], message=f"Failed: {str(e)}")


# =============================================================================
# INCLUDE SUB-ROUTERS
# =============================================================================

router.include_router(pipeline_router, prefix="/api/v1")
router.include_router(run_router, prefix="/api/v1")
router.include_router(agent_router, prefix="/api/v1")

# FR-043: Include approval routers
router.include_router(approval_router, prefix="/api/v1")
router.include_router(run_approval_router, prefix="/api/v1")

# FR-024: Include quota management routers
router.include_router(quota_router, prefix="/api/v1")
router.include_router(namespace_quota_router, prefix="/api/v1")

# FR-063: Include webhook management router
router.include_router(webhook_router, prefix="/api/v1")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "router",
    "pipeline_router",
    "run_router",
    "agent_router",
    "approval_router",
    "run_approval_router",
    "quota_router",
    "namespace_quota_router",
    "webhook_router",
    "get_quota_manager",
    "set_quota_manager",
    # FR-074: Checkpoint management
    "get_checkpoint_manager",
    "set_checkpoint_manager",
    # FR-063: Alert management
    "get_alert_manager",
    "set_alert_manager",
    "GreenLangErrorCodes",
]
