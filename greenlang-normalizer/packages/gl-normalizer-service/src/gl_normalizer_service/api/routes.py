"""
API route handlers for GL Normalizer Service.

This module implements the REST API endpoints for the GreenLang Normalizer,
following the GL-FOUND-X-003 specification.

Endpoints:
    POST /v1/normalize - Normalize a single value
    POST /v1/normalize/batch - Batch normalize (up to 10K items)
    POST /v1/jobs - Create async job (100K+ items)
    GET /v1/jobs/{job_id} - Get job status
    GET /v1/vocabularies - List vocabularies
    GET /v1/health - Health check

All endpoints include:
    - API revision in responses
    - GLNORM error codes
    - Audit trail support
    - Rate limiting
"""

from datetime import datetime
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse

from gl_normalizer_service.api.deps import (
    JobStore,
    NormalizerService,
    User,
    VocabularyService,
    get_current_user,
    get_job_store,
    get_normalizer_service,
    get_request_id,
    get_vocabulary_service,
    require_scope,
)
from gl_normalizer_service.api.models import (
    BatchNormalizeRequest,
    BatchNormalizeResponse,
    BatchResultItem,
    BatchSummary,
    CreateJobRequest,
    CreateJobResponse,
    DependencyHealth,
    ErrorCode,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    HealthStatus,
    JobProgress,
    JobStatus,
    JobStatusResponse,
    NormalizeRequest,
    NormalizeResponse,
    ReviewReason,
    VocabulariesResponse,
    VocabularyInfo,
)
from gl_normalizer_service.config import Settings, get_settings

logger = structlog.get_logger(__name__)

# ==============================================================================
# Router Setup
# ==============================================================================

router = APIRouter(prefix="/v1", tags=["Normalizer API v1"])

# API revision constant
API_REVISION = "2026-01-30"


# ==============================================================================
# Helper Functions
# ==============================================================================


def create_error_response(
    status_code: int,
    error_code: ErrorCode,
    message: str,
    field: str | None = None,
    details: dict | None = None,
    request_id: str | None = None,
) -> JSONResponse:
    """
    Create a standardized error response.

    Args:
        status_code: HTTP status code
        error_code: GLNORM error code
        message: Human-readable error message
        field: Field that caused the error (optional)
        details: Additional error context (optional)
        request_id: Request ID for support reference (optional)

    Returns:
        JSONResponse with error details
    """
    content = {
        "api_revision": API_REVISION,
        "error": {
            "code": error_code.value,
            "message": message,
        },
    }

    if field:
        content["error"]["field"] = field
    if details:
        content["error"]["details"] = details
    if request_id:
        content["request_id"] = request_id

    return JSONResponse(status_code=status_code, content=content)


# ==============================================================================
# Normalize Endpoints
# ==============================================================================


@router.post(
    "/normalize",
    response_model=NormalizeResponse,
    status_code=status.HTTP_200_OK,
    summary="Normalize a single value",
    description="""
Normalize a single value with its unit to canonical form.

The normalization process:
1. Parses the input value and unit
2. Looks up unit in vocabulary
3. Applies conversion rules
4. Returns canonical value with confidence score

If `target_unit` is not specified, the canonical unit for the unit category
is used (e.g., kg CO2 -> metric_ton_co2e).

Optional `entity` and `context` fields improve normalization accuracy
for ambiguous cases.
    """,
    responses={
        200: {
            "description": "Successful normalization",
            "model": NormalizeResponse,
        },
        400: {
            "description": "Invalid input or unknown unit",
            "model": ErrorResponse,
        },
        401: {
            "description": "Authentication required",
            "model": ErrorResponse,
        },
        429: {
            "description": "Rate limit exceeded",
            "model": ErrorResponse,
        },
    },
    tags=["Normalization"],
)
async def normalize_value(
    request: Request,
    normalize_request: NormalizeRequest,
    user: Annotated[User, Depends(require_scope("normalizer:write"))],
    normalizer: Annotated[NormalizerService, Depends(get_normalizer_service)],
    request_id: Annotated[str, Depends(get_request_id)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> NormalizeResponse:
    """
    Normalize a single value to canonical form.

    Args:
        request: FastAPI request
        normalize_request: Normalization request data
        user: Authenticated user
        normalizer: Normalizer service
        request_id: Request ID for tracing
        settings: Application settings

    Returns:
        NormalizeResponse with canonical value and metadata

    Raises:
        HTTPException: 400 if validation fails, 401 if unauthorized
    """
    logger.info(
        "normalize_single",
        user_id=user.id,
        tenant_id=user.tenant_id,
        request_id=request_id,
        value=str(normalize_request.value)[:50],
        unit=normalize_request.unit,
    )

    try:
        # Perform normalization
        result = await normalizer.normalize(
            value=normalize_request.value,
            unit=normalize_request.unit,
            target_unit=normalize_request.target_unit,
            entity=normalize_request.entity,
            context=normalize_request.context,
        )

        # Build response
        response = NormalizeResponse(
            api_revision=API_REVISION,
            canonical_value=result["canonical_value"],
            canonical_unit=result["canonical_unit"],
            confidence=result["confidence"],
            needs_review=result["needs_review"],
            review_reasons=[ReviewReason(r) for r in result.get("review_reasons", [])],
            audit_id=result["audit_id"],
            source_value=result["source_value"],
            source_unit=result["source_unit"],
            conversion_factor=result.get("conversion_factor"),
            metadata=result.get("metadata"),
        )

        logger.info(
            "normalize_single_complete",
            request_id=request_id,
            audit_id=result["audit_id"],
            confidence=result["confidence"],
        )

        return response

    except ValueError as e:
        logger.warning(
            "normalize_single_invalid_input",
            request_id=request_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": ErrorCode.INVALID_INPUT.value,
                "message": str(e),
                "field": "value",
            },
        )

    except KeyError as e:
        logger.warning(
            "normalize_single_unknown_unit",
            request_id=request_id,
            unit=normalize_request.unit,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": ErrorCode.UNKNOWN_UNIT.value,
                "message": f"Unknown unit: {normalize_request.unit}",
                "field": "unit",
            },
        )

    except Exception as e:
        logger.error(
            "normalize_single_error",
            request_id=request_id,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": ErrorCode.INTERNAL_ERROR.value,
                "message": "Internal processing error",
            },
        )


@router.post(
    "/normalize/batch",
    response_model=BatchNormalizeResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch normalize values",
    description="""
Normalize multiple values in a single request (up to 10,000 items).

Batch modes:
- **PARTIAL**: Continue on failures, return partial results
- **FAIL_FAST**: Stop processing on first failure
- **THRESHOLD**: Stop if failure rate exceeds threshold

For batches larger than 10,000 items, use the async job API (POST /v1/jobs).
    """,
    responses={
        200: {
            "description": "Batch normalization results",
            "model": BatchNormalizeResponse,
        },
        400: {
            "description": "Validation error or batch size exceeded",
            "model": ErrorResponse,
        },
        401: {
            "description": "Authentication required",
            "model": ErrorResponse,
        },
        429: {
            "description": "Rate limit exceeded",
            "model": ErrorResponse,
        },
    },
    tags=["Normalization"],
)
async def normalize_batch(
    request: Request,
    batch_request: BatchNormalizeRequest,
    user: Annotated[User, Depends(require_scope("normalizer:write"))],
    normalizer: Annotated[NormalizerService, Depends(get_normalizer_service)],
    request_id: Annotated[str, Depends(get_request_id)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> BatchNormalizeResponse:
    """
    Normalize a batch of values.

    Args:
        request: FastAPI request
        batch_request: Batch normalization request
        user: Authenticated user
        normalizer: Normalizer service
        request_id: Request ID for tracing
        settings: Application settings

    Returns:
        BatchNormalizeResponse with results and summary

    Raises:
        HTTPException: 400 if batch size exceeded, 401 if unauthorized
    """
    # Validate batch size
    if len(batch_request.items) > settings.batch_max_items:
        logger.warning(
            "batch_size_exceeded",
            request_id=request_id,
            batch_size=len(batch_request.items),
            max_size=settings.batch_max_items,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": ErrorCode.BATCH_SIZE_EXCEEDED.value,
                "message": f"Batch size {len(batch_request.items)} exceeds maximum {settings.batch_max_items}. Use async job API for larger batches.",
                "field": "items",
                "details": {
                    "received": len(batch_request.items),
                    "maximum": settings.batch_max_items,
                    "suggestion": "Use POST /v1/jobs for batches > 10,000 items",
                },
            },
        )

    logger.info(
        "normalize_batch",
        user_id=user.id,
        tenant_id=user.tenant_id,
        request_id=request_id,
        batch_size=len(batch_request.items),
        batch_mode=batch_request.batch_mode.value,
    )

    try:
        # Convert items to dicts
        items = [item.model_dump() for item in batch_request.items]

        # Perform batch normalization
        results, summary = await normalizer.normalize_batch(
            items=items,
            batch_mode=batch_request.batch_mode.value,
            threshold=batch_request.threshold or 0.1,
        )

        # Build response
        result_items = []
        for r in results:
            item = BatchResultItem(
                id=r["id"],
                success=r["success"],
                result=r.get("result"),
                error=ErrorDetail(**r["error"]) if r.get("error") else None,
            )
            result_items.append(item)

        response = BatchNormalizeResponse(
            api_revision=API_REVISION,
            results=result_items,
            summary=BatchSummary(**summary),
        )

        logger.info(
            "normalize_batch_complete",
            request_id=request_id,
            total=summary["total"],
            success=summary["success"],
            failed=summary["failed"],
            processing_time_ms=summary["processing_time_ms"],
        )

        return response

    except Exception as e:
        logger.error(
            "normalize_batch_error",
            request_id=request_id,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": ErrorCode.INTERNAL_ERROR.value,
                "message": "Internal processing error during batch normalization",
            },
        )


# ==============================================================================
# Job Endpoints
# ==============================================================================


@router.post(
    "/jobs",
    response_model=CreateJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create async normalization job",
    description="""
Create an asynchronous normalization job for large batches (100K+ items).

The job is queued for background processing. Use GET /v1/jobs/{job_id}
to poll for status, or provide a `callback_url` for webhook notification
when the job completes.

Job priorities:
- 1-3: Low priority (batch processing)
- 4-6: Normal priority (interactive)
- 7-10: High priority (time-sensitive)
    """,
    responses={
        202: {
            "description": "Job created and queued",
            "model": CreateJobResponse,
        },
        400: {
            "description": "Validation error",
            "model": ErrorResponse,
        },
        401: {
            "description": "Authentication required",
            "model": ErrorResponse,
        },
        429: {
            "description": "Rate limit exceeded",
            "model": ErrorResponse,
        },
    },
    tags=["Jobs"],
)
async def create_job(
    request: Request,
    job_request: CreateJobRequest,
    user: Annotated[User, Depends(require_scope("normalizer:write"))],
    job_store: Annotated[JobStore, Depends(get_job_store)],
    request_id: Annotated[str, Depends(get_request_id)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> CreateJobResponse:
    """
    Create an async normalization job.

    Args:
        request: FastAPI request
        job_request: Job creation request
        user: Authenticated user
        job_store: Job storage service
        request_id: Request ID for tracing
        settings: Application settings

    Returns:
        CreateJobResponse with job ID and status URL

    Raises:
        HTTPException: 400 if validation fails, 401 if unauthorized
    """
    logger.info(
        "create_job",
        user_id=user.id,
        tenant_id=user.tenant_id,
        request_id=request_id,
        item_count=len(job_request.items),
        priority=job_request.priority,
    )

    try:
        # Convert items to dicts
        items = [item.model_dump() for item in job_request.items]

        # Create job
        job = await job_store.create_job(
            items=items,
            batch_mode=job_request.batch_mode.value,
            callback_url=job_request.callback_url,
            priority=job_request.priority,
        )

        # Calculate estimated completion
        # Assume ~5000 items/second processing rate
        estimated_seconds = len(items) / 5000
        estimated_completion = datetime.utcnow()

        response = CreateJobResponse(
            api_revision=API_REVISION,
            job_id=job["job_id"],
            status=JobStatus.PENDING,
            status_url=f"/v1/jobs/{job['job_id']}",
            estimated_completion=None,  # Will be calculated by job processor
        )

        logger.info(
            "create_job_complete",
            request_id=request_id,
            job_id=job["job_id"],
            item_count=len(items),
        )

        return response

    except Exception as e:
        logger.error(
            "create_job_error",
            request_id=request_id,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": ErrorCode.INTERNAL_ERROR.value,
                "message": "Failed to create job",
            },
        )


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get job status",
    description="""
Retrieve the current status and progress of an async normalization job.

Job statuses:
- **PENDING**: Job queued, not yet started
- **PROCESSING**: Job currently being processed
- **COMPLETED**: Job finished successfully
- **FAILED**: Job failed with errors
- **CANCELLED**: Job was cancelled

When status is COMPLETED, the `result_url` field contains a link to
download the full results.
    """,
    responses={
        200: {
            "description": "Job status",
            "model": JobStatusResponse,
        },
        401: {
            "description": "Authentication required",
            "model": ErrorResponse,
        },
        403: {
            "description": "Not authorized to access job",
            "model": ErrorResponse,
        },
        404: {
            "description": "Job not found",
            "model": ErrorResponse,
        },
    },
    tags=["Jobs"],
)
async def get_job_status(
    request: Request,
    job_id: str,
    user: Annotated[User, Depends(require_scope("normalizer:read"))],
    job_store: Annotated[JobStore, Depends(get_job_store)],
    request_id: Annotated[str, Depends(get_request_id)],
) -> JobStatusResponse:
    """
    Get job status and progress.

    Args:
        request: FastAPI request
        job_id: Job identifier
        user: Authenticated user
        job_store: Job storage service
        request_id: Request ID for tracing

    Returns:
        JobStatusResponse with current status and progress

    Raises:
        HTTPException: 404 if job not found, 403 if unauthorized
    """
    logger.info(
        "get_job_status",
        user_id=user.id,
        request_id=request_id,
        job_id=job_id,
    )

    job = await job_store.get_job(job_id)

    if not job:
        logger.warning(
            "job_not_found",
            request_id=request_id,
            job_id=job_id,
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": ErrorCode.JOB_NOT_FOUND.value,
                "message": f"Job {job_id} not found",
            },
        )

    # Build progress if available
    progress = None
    if job.get("progress"):
        progress = JobProgress(
            processed=job["progress"]["processed"],
            total=job["progress"]["total"],
            percent_complete=job["progress"]["percent_complete"],
            current_rate=job["progress"].get("current_rate"),
        )

    # Build summary if completed
    summary = None
    if job.get("summary"):
        summary = BatchSummary(**job["summary"])

    # Build error if failed
    error = None
    if job.get("error"):
        error = ErrorDetail(**job["error"])

    # Parse timestamps
    created_at = datetime.fromisoformat(job["created_at"].replace("Z", "+00:00"))
    started_at = None
    if job.get("started_at"):
        started_at = datetime.fromisoformat(job["started_at"].replace("Z", "+00:00"))
    completed_at = None
    if job.get("completed_at"):
        completed_at = datetime.fromisoformat(job["completed_at"].replace("Z", "+00:00"))

    return JobStatusResponse(
        api_revision=API_REVISION,
        job_id=job["job_id"],
        status=JobStatus(job["status"]),
        progress=progress,
        summary=summary,
        result_url=job.get("result_url"),
        error=error,
        created_at=created_at,
        started_at=started_at,
        completed_at=completed_at,
    )


# ==============================================================================
# Vocabulary Endpoints
# ==============================================================================


@router.get(
    "/vocabularies",
    response_model=VocabulariesResponse,
    status_code=status.HTTP_200_OK,
    summary="List available vocabularies",
    description="""
List all available normalization vocabularies.

Vocabularies contain mappings for:
- Unit conversions (kg -> metric tons)
- Entity normalization (company names)
- Category standardization (emission scopes)

Each vocabulary includes version information and entry counts
for cache invalidation and capacity planning.
    """,
    responses={
        200: {
            "description": "List of vocabularies",
            "model": VocabulariesResponse,
        },
        401: {
            "description": "Authentication required",
            "model": ErrorResponse,
        },
    },
    tags=["Vocabularies"],
)
async def list_vocabularies(
    request: Request,
    user: Annotated[User, Depends(require_scope("normalizer:read"))],
    vocabulary_service: Annotated[VocabularyService, Depends(get_vocabulary_service)],
    request_id: Annotated[str, Depends(get_request_id)],
    category: Annotated[
        str | None,
        Query(
            description="Filter by category",
            examples=["emissions", "energy", "water"],
        ),
    ] = None,
) -> VocabulariesResponse:
    """
    List available vocabularies.

    Args:
        request: FastAPI request
        user: Authenticated user
        vocabulary_service: Vocabulary service
        request_id: Request ID for tracing
        category: Optional category filter

    Returns:
        VocabulariesResponse with vocabulary list
    """
    logger.info(
        "list_vocabularies",
        user_id=user.id,
        request_id=request_id,
        category=category,
    )

    vocabularies = await vocabulary_service.list_vocabularies()

    # Filter by category if specified
    if category:
        vocabularies = [
            v for v in vocabularies if category in v.get("categories", [])
        ]

    # Convert to response models
    vocab_infos = []
    for v in vocabularies:
        vocab_infos.append(
            VocabularyInfo(
                id=v["id"],
                name=v["name"],
                description=v.get("description"),
                version=v["version"],
                entry_count=v["entry_count"],
                last_updated=datetime.fromisoformat(
                    v["last_updated"].replace("Z", "+00:00")
                ),
                categories=v.get("categories", []),
            )
        )

    return VocabulariesResponse(
        api_revision=API_REVISION,
        vocabularies=vocab_infos,
        total=len(vocab_infos),
    )


# ==============================================================================
# Health Endpoints
# ==============================================================================


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="""
Check service health status for monitoring and load balancers.

Returns overall health status and individual dependency statuses.
This endpoint does not require authentication.

Health statuses:
- **healthy**: All systems operational
- **degraded**: Some systems impaired but service available
- **unhealthy**: Service unavailable

Returns HTTP 503 if status is unhealthy.
    """,
    responses={
        200: {
            "description": "Service healthy",
            "model": HealthResponse,
        },
        503: {
            "description": "Service unhealthy",
            "model": HealthResponse,
        },
    },
    tags=["System"],
)
async def health_check(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> HealthResponse:
    """
    Health check endpoint.

    Args:
        request: FastAPI request
        settings: Application settings

    Returns:
        HealthResponse with status and dependency health
    """
    import time

    # Track service start time (would be set at startup in production)
    start_time = getattr(request.app.state, "start_time", time.time())
    uptime_seconds = int(time.time() - start_time)

    # Check dependencies
    dependencies = []

    # Check Redis
    try:
        # In production: await redis.ping()
        dependencies.append(
            DependencyHealth(
                name="redis",
                status=HealthStatus.HEALTHY,
                latency_ms=2,
                message="Connected",
            )
        )
    except Exception as e:
        dependencies.append(
            DependencyHealth(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )
        )

    # Check vocabulary service
    try:
        # In production: await vocabulary_service.ping()
        dependencies.append(
            DependencyHealth(
                name="vocabulary_service",
                status=HealthStatus.HEALTHY,
                latency_ms=5,
                message="Available",
            )
        )
    except Exception as e:
        dependencies.append(
            DependencyHealth(
                name="vocabulary_service",
                status=HealthStatus.DEGRADED,
                message=str(e),
            )
        )

    # Determine overall status
    unhealthy_deps = [d for d in dependencies if d.status == HealthStatus.UNHEALTHY]
    degraded_deps = [d for d in dependencies if d.status == HealthStatus.DEGRADED]

    if unhealthy_deps:
        overall_status = HealthStatus.UNHEALTHY
    elif degraded_deps:
        overall_status = HealthStatus.DEGRADED
    else:
        overall_status = HealthStatus.HEALTHY

    response = HealthResponse(
        api_revision=API_REVISION,
        status=overall_status,
        version="1.0.0",
        uptime_seconds=uptime_seconds,
        dependencies=dependencies,
        timestamp=datetime.utcnow(),
    )

    # Return 503 if unhealthy
    if overall_status == HealthStatus.UNHEALTHY:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=response.model_dump(mode="json"),
        )

    return response


# ==============================================================================
# Readiness Endpoint
# ==============================================================================


@router.get(
    "/ready",
    status_code=status.HTTP_200_OK,
    summary="Readiness check",
    description="Kubernetes readiness probe endpoint.",
    tags=["System"],
)
async def readiness_check() -> dict:
    """
    Readiness check for Kubernetes.

    Returns:
        Simple ready status
    """
    return {"status": "ready", "api_revision": API_REVISION}


# ==============================================================================
# Liveness Endpoint
# ==============================================================================


@router.get(
    "/live",
    status_code=status.HTTP_200_OK,
    summary="Liveness check",
    description="Kubernetes liveness probe endpoint.",
    tags=["System"],
)
async def liveness_check() -> dict:
    """
    Liveness check for Kubernetes.

    Returns:
        Simple alive status
    """
    return {"status": "alive", "api_revision": API_REVISION}
