# -*- coding: utf-8 -*-
"""
Main Router Registration - AGENT-EUDR-009 Chain of Custody API

Aggregates all route modules under the ``/v1/eudr-coc`` prefix and
provides ``get_router()`` for integration with the GreenLang platform.

Route Module Summary (35 endpoints):
    event_routes:          5 endpoints (POST create, POST batch, GET detail, GET chain, POST amend)
    batch_routes:          7 endpoints (POST create, GET detail, POST split, POST merge, POST blend, GET genealogy, POST search)
    model_routes:          4 endpoints (POST assign, GET facility, POST validate, GET compliance)
    balance_routes:       11 endpoints (balance: input, output, get, reconcile, history; transform: create, batch, get; documents: link, list, validate)
    verification_routes:   3 endpoints (POST chain, POST batch, GET result)
    report_routes:         4 endpoints (POST traceability, POST mass-balance, GET report, GET download)
    + batch job:           2 endpoints (POST submit, DELETE cancel)
    + health:              1 endpoint (GET health)
    -------
    Total: 37 unique endpoints

Auth & RBAC:
    All endpoints (except health) require JWT auth via SEC-001 and
    check eudr-coc:* permissions via SEC-002.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-009, Section 7.4
Agent ID: GL-EUDR-COC-009
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

from greenlang.agents.eudr.chain_of_custody.api.balance_routes import (
    router as balance_router,
)
from greenlang.agents.eudr.chain_of_custody.api.batch_routes import (
    router as batch_router,
)
from greenlang.agents.eudr.chain_of_custody.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_coc_service,
    rate_limit_batch,
    rate_limit_standard,
    require_permission,
    validate_job_id,
)
from greenlang.agents.eudr.chain_of_custody.api.event_routes import (
    router as event_router,
)
from greenlang.agents.eudr.chain_of_custody.api.model_routes import (
    router as model_router,
)
from greenlang.agents.eudr.chain_of_custody.api.report_routes import (
    router as report_router,
)
from greenlang.agents.eudr.chain_of_custody.api.schemas import (
    BatchJobCancelResponse,
    BatchJobResponse,
    BatchJobStatus,
    BatchJobSubmitRequest,
    BatchJobType,
    HealthResponse,
    ProvenanceInfo,
)
from greenlang.agents.eudr.chain_of_custody.api.verification_routes import (
    router as verification_router,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Main router with /v1/eudr-coc prefix
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/v1/eudr-coc",
    tags=["eudr-chain-of-custody"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
    },
)

# Include all sub-routers
router.include_router(event_router)
router.include_router(batch_router)
router.include_router(model_router)
router.include_router(balance_router)
router.include_router(verification_router)
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
        "event_import, batch_verification, balance_reconciliation, "
        "report_generation, document_validation."
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
        require_permission("eudr-coc:batch-jobs:submit")
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
# DELETE /batch/{batch_id} (cancel batch job)
# ---------------------------------------------------------------------------


@router.delete(
    "/batch/{batch_id}",
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
    batch_id: str = Depends(validate_job_id),
    user: AuthUser = Depends(
        require_permission("eudr-coc:batch-jobs:cancel")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> BatchJobCancelResponse:
    """Cancel a batch job.

    Args:
        batch_id: Job identifier.
        user: Authenticated user with batch-jobs:cancel permission.

    Returns:
        BatchJobCancelResponse confirming cancellation.

    Raises:
        HTTPException: 404 if job not found, 409 if not cancellable.
    """
    try:
        store = _get_job_store()
        record = store.get(batch_id)

        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch job {batch_id} not found",
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
                    f"Job {batch_id} cannot be cancelled "
                    f"(current status: {current_status.value})"
                ),
            )

        now = datetime.now(timezone.utc).replace(microsecond=0)
        record["status"] = BatchJobStatus.CANCELLED
        record["cancelled_at"] = now

        logger.info("Batch job cancelled: id=%s", batch_id)

        return BatchJobCancelResponse(
            job_id=batch_id,
            status=BatchJobStatus.CANCELLED,
            cancelled_at=now,
            message="Job cancelled successfully",
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to cancel batch job %s: %s", batch_id, exc, exc_info=True
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
    description="Check EUDR Chain of Custody API health and component status.",
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
    """Return the EUDR Chain of Custody API router for mounting.

    Usage:
        >>> from greenlang.agents.eudr.chain_of_custody.api import get_router
        >>> app.include_router(get_router(), prefix="/api")

    Returns:
        Configured APIRouter with all chain of custody endpoints.
    """
    return router


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "router",
    "get_router",
]
