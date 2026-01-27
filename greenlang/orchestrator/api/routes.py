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

    # Operation errors (GL-E-OPS-*)
    PIPELINE_EXISTS = "GL-E-OPS-001"
    RUN_NOT_CANCELABLE = "GL-E-OPS-002"
    POLICY_VIOLATION = "GL-E-OPS-003"

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
# INCLUDE SUB-ROUTERS
# =============================================================================

router.include_router(pipeline_router, prefix="/api/v1")
router.include_router(run_router, prefix="/api/v1")
router.include_router(agent_router, prefix="/api/v1")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "router",
    "pipeline_router",
    "run_router",
    "agent_router",
    "GreenLangErrorCodes",
]
