# -*- coding: utf-8 -*-
"""
Batch Job Routes - AGENT-EUDR-008 Multi-Tier Supplier Tracker API

Endpoints for submitting and managing batch processing jobs covering
discovery, risk assessment, compliance checking, profile import,
and gap analysis operations.

Endpoints:
    POST   /batch           - Submit batch job
    DELETE /batch/{batch_id} - Cancel batch job

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-008 Multi-Tier Supplier Tracker (GL-EUDR-MST-008)
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Path, Request, status

from greenlang.agents.eudr.multi_tier_supplier.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_supplier_service,
    rate_limit_batch,
    require_permission,
)
from greenlang.agents.eudr.multi_tier_supplier.api.schemas import (
    BatchJobRequestSchema,
    BatchJobResponseSchema,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Batch Jobs"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_provenance(data: str) -> str:
    """Compute SHA-256 provenance hash for audit trail.

    Args:
        data: String to hash.

    Returns:
        Hex-encoded SHA-256 hash.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# Estimated durations per job type (seconds)
_ESTIMATED_DURATIONS: Dict[str, int] = {
    "discovery": 300,
    "risk_assessment": 180,
    "compliance_check": 120,
    "profile_import": 240,
    "gap_analysis": 150,
}


# ---------------------------------------------------------------------------
# POST /batch
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=BatchJobResponseSchema,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit batch processing job",
    description=(
        "Submit a batch processing job for asynchronous execution. "
        "Supported job types: discovery, risk_assessment, "
        "compliance_check, profile_import, gap_analysis. "
        "Returns a batch ID for status tracking. Optionally specify "
        "a callback URL for completion notifications."
    ),
    responses={
        202: {"description": "Batch job accepted"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def submit_batch_job(
    body: BatchJobRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-mst:batch:write")
    ),
    _rate: None = Depends(rate_limit_batch),
) -> BatchJobResponseSchema:
    """Submit a batch processing job.

    Creates an asynchronous batch job that will be processed by the
    background job system. Returns immediately with a batch ID for
    polling status.

    Args:
        body: Batch job request with type, parameters, and priority.
        request: FastAPI request object.
        user: Authenticated user with batch:write permission.

    Returns:
        BatchJobResponseSchema with batch ID and estimated duration.

    Raises:
        HTTPException: 400 on invalid input, 500 on internal error.
    """
    start = time.monotonic()
    logger.info(
        "Submit batch job: user=%s job_type=%s priority=%s",
        user.user_id,
        body.job_type,
        body.priority,
    )

    try:
        service = get_supplier_service()

        result = service.submit_batch_job(
            job_type=body.job_type,
            parameters=body.parameters,
            priority=body.priority,
            callback_url=body.callback_url,
            submitted_by=user.user_id,
            tenant_id=user.tenant_id,
        )

        elapsed = time.monotonic() - start
        batch_id = result.get("batch_id", str(uuid.uuid4()))
        provenance = _compute_provenance(
            f"batch_submit|{batch_id}|{body.job_type}|{body.priority}|{elapsed}"
        )

        estimated_duration = _ESTIMATED_DURATIONS.get(body.job_type, 300)
        total_items = result.get("total_items")

        logger.info(
            "Batch job submitted: user=%s batch_id=%s job_type=%s "
            "priority=%s estimated_duration=%ds elapsed_ms=%.1f",
            user.user_id,
            batch_id,
            body.job_type,
            body.priority,
            estimated_duration,
            elapsed * 1000,
        )

        return BatchJobResponseSchema(
            batch_id=batch_id,
            job_type=body.job_type,
            status="pending",
            priority=body.priority,
            estimated_duration_seconds=estimated_duration,
            total_items=total_items,
            processed_items=0,
            failed_items=0,
            progress_pct=0.0,
            callback_url=body.callback_url,
            provenance_hash=provenance,
        )

    except ValueError as exc:
        logger.warning(
            "Batch job validation error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch job submission validation failed: {exc}",
        )
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Batch job submission failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch job submission failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# GET /batch/{batch_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{batch_id}",
    response_model=BatchJobResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Get batch job status",
    description=(
        "Retrieve the current status and progress of a batch processing "
        "job. Returns processing progress, item counts, and result "
        "summary when completed."
    ),
    responses={
        200: {"description": "Batch job status retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Batch job not found"},
    },
)
async def get_batch_status(
    batch_id: str = Path(
        ..., min_length=1, max_length=100, description="Batch job identifier"
    ),
    request: Request = None,
    user: AuthUser = Depends(
        require_permission("eudr-mst:batch:read")
    ),
) -> BatchJobResponseSchema:
    """Get batch job status.

    Args:
        batch_id: Batch job identifier.
        request: FastAPI request object.
        user: Authenticated user with batch:read permission.

    Returns:
        BatchJobResponseSchema with current status and progress.

    Raises:
        HTTPException: 404 if batch job not found.
    """
    start = time.monotonic()
    logger.info(
        "Get batch status: user=%s batch_id=%s",
        user.user_id,
        batch_id,
    )

    try:
        service = get_supplier_service()
        result = service.get_batch_status(batch_id=batch_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch job not found: {batch_id}",
            )

        elapsed = time.monotonic() - start
        provenance = _compute_provenance(
            f"batch_status|{batch_id}|{result.get('status', 'unknown')}|{elapsed}"
        )

        logger.info(
            "Batch status retrieved: user=%s batch_id=%s status=%s "
            "progress=%.1f%% elapsed_ms=%.1f",
            user.user_id,
            batch_id,
            result.get("status", "unknown"),
            result.get("progress_pct", 0),
            elapsed * 1000,
        )

        result["provenance_hash"] = provenance
        return BatchJobResponseSchema(**result)

    except HTTPException:
        raise
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Get batch status failed: user=%s batch_id=%s error=%s",
            user.user_id,
            batch_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch job status retrieval failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# DELETE /batch/{batch_id}
# ---------------------------------------------------------------------------


@router.delete(
    "/{batch_id}",
    response_model=BatchJobResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Cancel batch job",
    description=(
        "Cancel a pending or running batch processing job. "
        "Jobs that have already completed cannot be cancelled. "
        "Partially completed work is preserved."
    ),
    responses={
        200: {"description": "Batch job cancelled"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Batch job not found"},
        409: {"model": ErrorResponse, "description": "Job already completed"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def cancel_batch_job(
    batch_id: str = Path(
        ..., min_length=1, max_length=100, description="Batch job identifier"
    ),
    request: Request = None,
    user: AuthUser = Depends(
        require_permission("eudr-mst:batch:delete")
    ),
    _rate: None = Depends(rate_limit_batch),
) -> BatchJobResponseSchema:
    """Cancel a batch processing job.

    Args:
        batch_id: Batch job identifier to cancel.
        request: FastAPI request object.
        user: Authenticated user with batch:delete permission.

    Returns:
        BatchJobResponseSchema with updated cancelled status.

    Raises:
        HTTPException: 404 if not found, 409 if already completed.
    """
    start = time.monotonic()
    logger.info(
        "Cancel batch job: user=%s batch_id=%s",
        user.user_id,
        batch_id,
    )

    try:
        service = get_supplier_service()

        # Check current status first
        current = service.get_batch_status(batch_id=batch_id)

        if current is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch job not found: {batch_id}",
            )

        current_status = current.get("status", "")
        if current_status in ("completed", "failed"):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Cannot cancel batch job in '{current_status}' state. "
                    "Only 'pending' and 'running' jobs can be cancelled."
                ),
            )

        result = service.cancel_batch_job(
            batch_id=batch_id,
            cancelled_by=user.user_id,
        )

        elapsed = time.monotonic() - start
        provenance = _compute_provenance(
            f"batch_cancel|{batch_id}|{elapsed}"
        )

        logger.info(
            "Batch job cancelled: user=%s batch_id=%s elapsed_ms=%.1f",
            user.user_id,
            batch_id,
            elapsed * 1000,
        )

        result["provenance_hash"] = provenance
        result["status"] = "cancelled"
        return BatchJobResponseSchema(**result)

    except HTTPException:
        raise
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Cancel batch job failed: user=%s batch_id=%s error=%s",
            user.user_id,
            batch_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch job cancellation failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = ["router"]
