# -*- coding: utf-8 -*-
"""
Blockchain Integration API Router - AGENT-EUDR-013

Main router aggregating 8 domain-specific sub-routers plus batch
and health endpoints for the Blockchain Integration Agent.

Prefix: /v1/eudr-bci
Tags: eudr-blockchain-integration

Sub-routers:
    - anchor_routes: Transaction anchoring (5 endpoints)
    - contract_routes: Smart contract management (5 endpoints)
    - chain_routes: Multi-chain connections (4 endpoints)
    - verification_routes: On-chain verification (4 endpoints)
    - event_routes: Event listening (5 endpoints)
    - merkle_routes: Merkle proof operations (4 endpoints)
    - sharing_routes: Cross-party data sharing (5 endpoints)
    - evidence_routes: Compliance evidence (4 endpoints)
    + batch (2 endpoints) + health (1 endpoint) = 39 total

Auth & RBAC:
    All endpoints (except health) require JWT auth via SEC-001 and
    check eudr-bci:* permissions via SEC-002.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-013, Section 7.4
Agent ID: GL-EUDR-BCI-013
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
    from greenlang.agents.eudr.blockchain_integration.api.anchor_routes import (
        router as anchor_router,
    )
except ImportError:
    anchor_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.blockchain_integration.api.contract_routes import (
        router as contract_router,
    )
except ImportError:
    contract_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.blockchain_integration.api.chain_routes import (
        router as chain_router,
    )
except ImportError:
    chain_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.blockchain_integration.api.verification_routes import (
        router as verification_router,
    )
except ImportError:
    verification_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.blockchain_integration.api.event_routes import (
        router as event_router,
    )
except ImportError:
    event_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.blockchain_integration.api.merkle_routes import (
        router as merkle_router,
    )
except ImportError:
    merkle_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.blockchain_integration.api.sharing_routes import (
        router as sharing_router,
    )
except ImportError:
    sharing_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.blockchain_integration.api.evidence_routes import (
        router as evidence_router,
    )
except ImportError:
    evidence_router = None  # type: ignore[assignment]

from greenlang.agents.eudr.blockchain_integration.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_blockchain_service,
    rate_limit_batch,
    rate_limit_standard,
    require_permission,
    validate_job_id,
)
from greenlang.agents.eudr.blockchain_integration.api.schemas import (
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
# Main router with /v1/eudr-bci prefix
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/v1/eudr-bci",
    tags=["eudr-blockchain-integration"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
    },
)

# Include all sub-routers that were successfully imported
if anchor_router is not None:
    router.include_router(anchor_router)
if contract_router is not None:
    router.include_router(contract_router)
if chain_router is not None:
    router.include_router(chain_router)
if verification_router is not None:
    router.include_router(verification_router)
if event_router is not None:
    router.include_router(event_router)
if merkle_router is not None:
    router.include_router(merkle_router)
if sharing_router is not None:
    router.include_router(sharing_router)
if evidence_router is not None:
    router.include_router(evidence_router)


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
        "anchor_batch, verify_batch, evidence_batch, merkle_build, "
        "event_replay."
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
        require_permission("eudr-bci:batch-jobs:submit")
    ),
    _rate: None = Depends(rate_limit_batch),
) -> BatchJobSchema:
    """Submit an async batch job.

    Args:
        request: FastAPI request object.
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
            "Batch job submitted: id=%s type=%s priority=%d elapsed_ms=%.1f",
            job_id,
            body.job_type.value,
            body.priority,
            elapsed_ms,
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
        require_permission("eudr-bci:batch-jobs:cancel")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> BatchJobCancelSchema:
    """Cancel a batch job.

    Args:
        request: FastAPI request object.
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
        "Check EUDR Blockchain Integration API health and component "
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
    """Return the EUDR Blockchain Integration API router for mounting.

    Usage:
        >>> from greenlang.agents.eudr.blockchain_integration.api import get_router
        >>> app.include_router(get_router(), prefix="/api")

    Returns:
        Configured APIRouter with all blockchain integration endpoints.
    """
    return router


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "router",
    "get_router",
]
