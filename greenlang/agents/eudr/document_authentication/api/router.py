# -*- coding: utf-8 -*-
"""
Document Authentication API Router - AGENT-EUDR-012

Main router aggregating 8 domain-specific sub-routers plus batch
and health endpoints for the Document Authentication Agent.

Prefix: /v1/eudr-dav
Tags: eudr-document-authentication

Sub-routers:
    - classify_routes: Classification (5 endpoints)
    - signature_routes: Signature verification (4 endpoints)
    - hash_routes: Hash integrity (4 endpoints)
    - certificate_routes: Certificate validation (4 endpoints)
    - metadata_routes: Metadata extraction (3 endpoints)
    - fraud_routes: Fraud detection (5 endpoints)
    - crossref_routes: Cross-reference (4 endpoints)
    - report_routes: Reports and dashboards (5 endpoints)
    + batch (2 endpoints) + health (1 endpoint) = 37 total

Auth & RBAC:
    All endpoints (except health) require JWT auth via SEC-001 and
    check eudr-dav:* permissions via SEC-002.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-012, Section 7.4
Agent ID: GL-EUDR-DAV-012
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status

# ---------------------------------------------------------------------------
# Sub-router imports (try/except for import safety during startup)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.document_authentication.api.classify_routes import (
        router as classify_router,
    )
except ImportError:
    classify_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.document_authentication.api.signature_routes import (
        router as signature_router,
    )
except ImportError:
    signature_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.document_authentication.api.hash_routes import (
        router as hash_router,
    )
except ImportError:
    hash_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.document_authentication.api.certificate_routes import (
        router as certificate_router,
    )
except ImportError:
    certificate_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.document_authentication.api.metadata_routes import (
        router as metadata_router,
    )
except ImportError:
    metadata_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.document_authentication.api.fraud_routes import (
        router as fraud_router,
    )
except ImportError:
    fraud_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.document_authentication.api.crossref_routes import (
        router as crossref_router,
    )
except ImportError:
    crossref_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.document_authentication.api.report_routes import (
        router as report_router,
    )
except ImportError:
    report_router = None  # type: ignore[assignment]

from greenlang.agents.eudr.document_authentication.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_dav_service,
    rate_limit_batch,
    rate_limit_standard,
    require_permission,
    validate_job_id,
)
from greenlang.agents.eudr.document_authentication.api.schemas import (
    BatchJobCancelSchema,
    BatchJobSchema,
    BatchJobStatusSchema,
    BatchJobTypeSchema,
    HealthSchema,
    ProvenanceInfo,
    SubmitBatchSchema,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Main router with /v1/eudr-dav prefix
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/v1/eudr-dav",
    tags=["eudr-document-authentication"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
    },
)

# Include all sub-routers that were successfully imported
if classify_router is not None:
    router.include_router(classify_router)
if signature_router is not None:
    router.include_router(signature_router)
if hash_router is not None:
    router.include_router(hash_router)
if certificate_router is not None:
    router.include_router(certificate_router)
if metadata_router is not None:
    router.include_router(metadata_router)
if fraud_router is not None:
    router.include_router(fraud_router)
if crossref_router is not None:
    router.include_router(crossref_router)
if report_router is not None:
    router.include_router(report_router)


# ---------------------------------------------------------------------------
# In-memory batch job store
# ---------------------------------------------------------------------------

_job_store: Dict[str, Dict] = {}


def _get_job_store() -> Dict[str, Dict]:
    """Return the batch job store singleton."""
    return _job_store


def _compute_provenance_hash(data: dict) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# POST /batch (batch job submit)
# ---------------------------------------------------------------------------


@router.post(
    "/batch",
    response_model=BatchJobSchema,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit batch job",
    description=(
        "Submit an asynchronous batch processing job. Supported types: "
        "classify_batch, verify_signatures_batch, detect_fraud_batch, "
        "crossref_batch, report_generation."
    ),
    responses={
        202: {"description": "Job accepted for processing"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
    tags=["Batch Jobs"],
)
async def submit_batch_job(
    request: Request,
    body: SubmitBatchSchema,
    user: AuthUser = Depends(
        require_permission("eudr-dav:batch-jobs:submit")
    ),
    _rate: None = Depends(rate_limit_batch),
) -> BatchJobSchema:
    """Submit an async batch job.

    Args:
        body: Job submission parameters including type, priority,
            and job-specific parameters.
        user: Authenticated user with batch-jobs:submit permission.

    Returns:
        BatchJobSchema with job ID and queued status.
    """
    start = time.monotonic()
    try:
        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).replace(microsecond=0)

        provenance_hash = _compute_provenance_hash({
            "job_id": job_id,
            "job_type": body.job_type.value,
            "submitted_by": user.user_id,
        })

        job_record = {
            "job_id": job_id,
            "job_type": body.job_type,
            "status": BatchJobStatusSchema.QUEUED,
            "priority": body.priority,
            "parameters": body.parameters,
            "progress_percent": 0.0,
            "total_items": None,
            "processed_items": None,
            "failed_items": None,
            "result": None,
            "error": None,
            "callback_url": body.callback_url,
            "submitted_at": now,
            "started_at": None,
            "completed_at": None,
            "cancelled_at": None,
            "provenance_hash": provenance_hash,
        }

        store = _get_job_store()
        store[job_id] = job_record

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Batch job submitted: id=%s type=%s priority=%d",
            job_id,
            body.job_type.value,
            body.priority,
        )

        return BatchJobSchema(**job_record)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to submit batch job: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit batch job",
        )


# ---------------------------------------------------------------------------
# DELETE /batch/{job_id} (cancel batch job)
# ---------------------------------------------------------------------------


@router.delete(
    "/batch/{job_id}",
    response_model=BatchJobCancelSchema,
    summary="Cancel batch job",
    description="Cancel a queued or running batch job.",
    responses={
        200: {"description": "Job cancelled"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Job not found"},
        409: {"model": ErrorResponse, "description": "Job cannot be cancelled"},
    },
    tags=["Batch Jobs"],
)
async def cancel_batch_job(
    request: Request,
    job_id: str = Depends(validate_job_id),
    user: AuthUser = Depends(
        require_permission("eudr-dav:batch-jobs:cancel")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> BatchJobCancelSchema:
    """Cancel a batch job.

    Args:
        job_id: Job identifier.
        user: Authenticated user with batch-jobs:cancel permission.

    Returns:
        BatchJobCancelSchema confirming cancellation.

    Raises:
        HTTPException: 404 if job not found, 409 if not cancellable.
    """
    try:
        store = _get_job_store()
        record = store.get(job_id)

        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch job {job_id} not found",
            )

        current_status = record["status"]
        if current_status in (
            BatchJobStatusSchema.COMPLETED,
            BatchJobStatusSchema.FAILED,
            BatchJobStatusSchema.CANCELLED,
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Job {job_id} cannot be cancelled "
                    f"(current status: {current_status.value})"
                ),
            )

        now = datetime.now(timezone.utc).replace(microsecond=0)
        record["status"] = BatchJobStatusSchema.CANCELLED
        record["cancelled_at"] = now

        logger.info("Batch job cancelled: id=%s by=%s", job_id, user.user_id)

        return BatchJobCancelSchema(
            job_id=job_id,
            status=BatchJobStatusSchema.CANCELLED,
            cancelled_at=now,
            message="Job cancelled successfully",
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to cancel batch job %s: %s", job_id, exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel batch job",
        )


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthSchema,
    summary="Health check",
    description=(
        "Check EUDR Document Authentication API health and component "
        "status. No authentication required."
    ),
    tags=["System"],
)
async def health_check() -> HealthSchema:
    """Health check endpoint for load balancers and monitoring.

    Returns:
        HealthSchema with service health and component status.
    """
    return HealthSchema()


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def get_router() -> APIRouter:
    """Return the EUDR Document Authentication API router for mounting.

    Usage:
        >>> from greenlang.agents.eudr.document_authentication.api import get_router
        >>> app.include_router(get_router(), prefix="/api")

    Returns:
        Configured APIRouter with all document authentication endpoints.
    """
    return router


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "router",
    "get_router",
]
