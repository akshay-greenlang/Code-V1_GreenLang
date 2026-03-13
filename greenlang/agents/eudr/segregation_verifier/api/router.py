# -*- coding: utf-8 -*-
"""
Main Router Registration - AGENT-EUDR-010 Segregation Verifier API

Aggregates all route modules under the ``/v1/eudr-sgv`` prefix and
provides ``get_router()`` for integration with the GreenLang platform.

Route Module Summary (37 endpoints):
    scp_routes:            6 endpoints (POST register, GET detail, PUT update, POST validate, POST batch-import, POST search)
    storage_routes:        5 endpoints (POST zones, GET zones, POST events, POST audit, GET score)
    transport_routes:      5 endpoints (POST vehicles, GET vehicle, POST verify, POST cleaning, GET history)
    processing_routes:     5 endpoints (POST lines, GET line, POST changeover, POST verify, GET score)
    contamination_routes:  5 endpoints (POST detect, POST events, GET event, POST impact, GET heatmap)
    assessment_routes:     3 endpoints (POST assessment, GET latest, GET history)
    report_routes:         5 endpoints (POST audit, POST contamination, POST evidence, GET report, GET download)
    + batch job:           2 endpoints (POST submit, DELETE cancel)
    + health:              1 endpoint (GET health)
    -------
    Total: 37 unique endpoints

Auth & RBAC:
    All endpoints (except health) require JWT auth via SEC-001 and
    check eudr-sgv:* permissions via SEC-002.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-010, Section 7.4
Agent ID: GL-EUDR-SGV-010
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

from greenlang.agents.eudr.segregation_verifier.api.assessment_routes import (
    router as assessment_router,
)
from greenlang.agents.eudr.segregation_verifier.api.contamination_routes import (
    router as contamination_router,
)
from greenlang.agents.eudr.segregation_verifier.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_sgv_service,
    rate_limit_batch,
    rate_limit_standard,
    require_permission,
    validate_job_id,
)
from greenlang.agents.eudr.segregation_verifier.api.processing_routes import (
    router as processing_router,
)
from greenlang.agents.eudr.segregation_verifier.api.report_routes import (
    router as report_router,
)
from greenlang.agents.eudr.segregation_verifier.api.schemas import (
    BatchJobCancelResponse,
    BatchJobResponse,
    BatchJobStatus,
    BatchJobSubmitRequest,
    BatchJobType,
    HealthResponse,
    ProvenanceInfo,
)
from greenlang.agents.eudr.segregation_verifier.api.scp_routes import (
    router as scp_router,
)
from greenlang.agents.eudr.segregation_verifier.api.storage_routes import (
    router as storage_router,
)
from greenlang.agents.eudr.segregation_verifier.api.transport_routes import (
    router as transport_router,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Main router with /v1/eudr-sgv prefix
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/v1/eudr-sgv",
    tags=["eudr-segregation-verifier"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
    },
)

# Include all sub-routers
router.include_router(scp_router)
router.include_router(storage_router)
router.include_router(transport_router)
router.include_router(processing_router)
router.include_router(contamination_router)
router.include_router(assessment_router)
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
    response_model=BatchJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit batch job",
    description=(
        "Submit an asynchronous batch processing job. Supported types: "
        "scp_import, storage_audit, transport_verification, "
        "processing_verification, contamination_scan, "
        "facility_assessment, report_generation."
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
    body: BatchJobSubmitRequest,
    user: AuthUser = Depends(
        require_permission("eudr-sgv:batch-jobs:submit")
    ),
    _rate: None = Depends(rate_limit_batch),
) -> BatchJobResponse:
    """Submit an async batch job.

    Args:
        body: Job submission parameters.
        user: Authenticated user with batch-jobs:submit permission.

    Returns:
        BatchJobResponse with job ID and status.
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
            "status": BatchJobStatus.QUEUED,
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

        return BatchJobResponse(**job_record)

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
    response_model=BatchJobCancelResponse,
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
        require_permission("eudr-sgv:batch-jobs:cancel")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> BatchJobCancelResponse:
    """Cancel a batch job.

    Args:
        job_id: Job identifier.
        user: Authenticated user with batch-jobs:cancel permission.

    Returns:
        BatchJobCancelResponse confirming cancellation.

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
            BatchJobStatus.COMPLETED,
            BatchJobStatus.FAILED,
            BatchJobStatus.CANCELLED,
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Job {job_id} cannot be cancelled "
                    f"(current status: {current_status.value})"
                ),
            )

        now = datetime.now(timezone.utc).replace(microsecond=0)
        record["status"] = BatchJobStatus.CANCELLED
        record["cancelled_at"] = now

        logger.info("Batch job cancelled: id=%s", job_id)

        return BatchJobCancelResponse(
            job_id=job_id,
            status=BatchJobStatus.CANCELLED,
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
    response_model=HealthResponse,
    summary="Health check",
    description="Check EUDR Segregation Verifier API health and component status.",
    tags=["System"],
)
async def health_check() -> HealthResponse:
    """Health check endpoint for load balancers and monitoring.

    Returns:
        HealthResponse with component health statuses.
    """
    return HealthResponse()


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def get_router() -> APIRouter:
    """Return the EUDR Segregation Verifier API router for mounting.

    Usage:
        >>> from greenlang.agents.eudr.segregation_verifier.api import get_router
        >>> app.include_router(get_router(), prefix="/api")

    Returns:
        Configured APIRouter with all segregation verifier endpoints.
    """
    return router


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "router",
    "get_router",
]
