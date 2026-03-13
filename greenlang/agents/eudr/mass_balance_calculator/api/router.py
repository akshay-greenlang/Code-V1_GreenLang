# -*- coding: utf-8 -*-
"""
Mass Balance Calculator API Router - AGENT-EUDR-011

Main router aggregating 7 domain-specific sub-routers plus batch
and health endpoints for the Mass Balance Calculator Agent.

Prefix: /v1/eudr-mbc
Tags: eudr-mass-balance-calculator

Sub-routers:
    - ledger_routes: Ledger CRUD and transactions (7 endpoints)
    - period_routes: Credit period management (5 endpoints)
    - factor_routes: Conversion factor validation (4 endpoints)
    - overdraft_routes: Overdraft detection/alerts (5 endpoints)
    - loss_routes: Loss and waste tracking (4 endpoints)
    - reconciliation_routes: Reconciliation (4 endpoints)
    - consolidation_routes: Multi-facility consolidation (5 endpoints)
    + batch (2 endpoints) + health (1 endpoint) = 37 total

Auth & RBAC:
    All endpoints (except health) require JWT auth via SEC-001 and
    check eudr-mbc:* permissions via SEC-002.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-011, Section 7.4
Agent ID: GL-EUDR-MBC-011
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
    from greenlang.agents.eudr.mass_balance_calculator.api.ledger_routes import (
        router as ledger_router,
    )
except ImportError:
    ledger_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.mass_balance_calculator.api.period_routes import (
        router as period_router,
    )
except ImportError:
    period_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.mass_balance_calculator.api.factor_routes import (
        router as factor_router,
    )
except ImportError:
    factor_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.mass_balance_calculator.api.overdraft_routes import (
        router as overdraft_router,
    )
except ImportError:
    overdraft_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.mass_balance_calculator.api.loss_routes import (
        router as loss_router,
    )
except ImportError:
    loss_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.mass_balance_calculator.api.reconciliation_routes import (
        router as reconciliation_router,
    )
except ImportError:
    reconciliation_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.mass_balance_calculator.api.consolidation_routes import (
        router as consolidation_router,
    )
except ImportError:
    consolidation_router = None  # type: ignore[assignment]

from greenlang.agents.eudr.mass_balance_calculator.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_mbc_service,
    rate_limit_batch,
    rate_limit_standard,
    require_permission,
    validate_job_id,
)
from greenlang.agents.eudr.mass_balance_calculator.api.schemas import (
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
# Main router with /v1/eudr-mbc prefix
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/v1/eudr-mbc",
    tags=["eudr-mass-balance-calculator"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
    },
)

# Include all sub-routers that were successfully imported
if ledger_router is not None:
    router.include_router(ledger_router)
if period_router is not None:
    router.include_router(period_router)
if factor_router is not None:
    router.include_router(factor_router)
if overdraft_router is not None:
    router.include_router(overdraft_router)
if loss_router is not None:
    router.include_router(loss_router)
if reconciliation_router is not None:
    router.include_router(reconciliation_router)
if consolidation_router is not None:
    router.include_router(consolidation_router)


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
        "ledger_entry_import, reconciliation_batch, report_generation, "
        "loss_import, factor_validation."
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
        require_permission("eudr-mbc:batch-jobs:submit")
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
        require_permission("eudr-mbc:batch-jobs:cancel")
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
        "Check EUDR Mass Balance Calculator API health and component status. "
        "No authentication required."
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
    """Return the EUDR Mass Balance Calculator API router for mounting.

    Usage:
        >>> from greenlang.agents.eudr.mass_balance_calculator.api import get_router
        >>> app.include_router(get_router(), prefix="/api")

    Returns:
        Configured APIRouter with all mass balance calculator endpoints.
    """
    return router


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "router",
    "get_router",
]
