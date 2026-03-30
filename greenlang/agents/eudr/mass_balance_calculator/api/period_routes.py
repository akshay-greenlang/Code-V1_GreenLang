# -*- coding: utf-8 -*-
"""
Period Routes - AGENT-EUDR-011 Mass Balance Calculator API

Endpoints for credit period lifecycle management including creation,
detail retrieval, extension, rollover, and active period listing.
Credit periods govern the accounting window for mass balance
calculations per ISO 22095:2020 and EUDR Article 14.

Standard-specific durations:
    - RSPO: 90 days (RSPO SCC 2020)
    - FSC: 365 days (FSC-STD-40-004)
    - ISCC: 365 days (ISCC 203)
    - EUDR default: 365 days

Endpoints:
    POST  /periods                      - Create a new credit period
    GET   /periods/{period_id}          - Get period details
    PUT   /periods/{period_id}          - Extend period end date
    POST  /periods/rollover             - Trigger period rollover
    GET   /periods/active/{facility_id} - Get active periods for facility

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-011, Feature 2 (Credit Period Lifecycle)
Agent ID: GL-EUDR-MBC-011
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from greenlang.schemas import utcnow

from greenlang.agents.eudr.mass_balance_calculator.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_mbc_service,
    get_request_id,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
    validate_facility_id,
    validate_period_id,
)
from greenlang.agents.eudr.mass_balance_calculator.api.schemas import (
    ActivePeriodsSchema,
    CreatePeriodSchema,
    ExtendPeriodSchema,
    PeriodDetailSchema,
    PeriodStatusSchema,
    ProvenanceInfo,
    RolloverPeriodSchema,
    RolloverResultSchema,
    StandardTypeSchema,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Credit Periods"])

# ---------------------------------------------------------------------------
# In-memory store (replaced by database in production)
# ---------------------------------------------------------------------------

_period_store: Dict[str, Dict] = {}

def _get_period_store() -> Dict[str, Dict]:
    """Return the period store singleton."""
    return _period_store

def _compute_provenance_hash(data: dict) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

# Standard-specific credit period durations in days
_STANDARD_PERIOD_DAYS: Dict[str, int] = {
    "rspo": 90,
    "fsc": 365,
    "iscc": 365,
    "utz_ra": 365,
    "fairtrade": 365,
    "eudr_default": 365,
}

# ---------------------------------------------------------------------------
# POST /periods
# ---------------------------------------------------------------------------

@router.post(
    "/periods",
    response_model=PeriodDetailSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new credit period",
    description=(
        "Create a new credit period for a facility and commodity. "
        "If end_date is not specified, it is auto-calculated based on "
        "the certification standard's default period duration."
    ),
    responses={
        201: {"description": "Credit period created"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def create_period(
    request: Request,
    body: CreatePeriodSchema,
    user: AuthUser = Depends(
        require_permission("eudr-mbc:periods:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> PeriodDetailSchema:
    """Create a new credit period.

    Args:
        body: Period creation parameters including facility_id, commodity,
            standard, start_date, and optional end_date.
        user: Authenticated user with periods:create permission.

    Returns:
        PeriodDetailSchema with the newly created period details.
    """
    start = time.monotonic()
    try:
        period_id = str(uuid.uuid4())
        now = utcnow()

        # Calculate end date if not provided
        standard_key = body.standard.value if body.standard else "eudr_default"
        period_days = _STANDARD_PERIOD_DAYS.get(standard_key, 365)
        end_date = body.end_date or (body.start_date + timedelta(days=period_days))

        # Calculate grace period end (5 days after period end)
        grace_period_end = end_date + timedelta(days=5)

        provenance_data = body.model_dump(mode="json")
        provenance_data["period_id"] = period_id
        provenance_data["created_by"] = user.user_id
        provenance_hash = _compute_provenance_hash(provenance_data)

        provenance = ProvenanceInfo(
            provenance_hash=provenance_hash,
            created_by=user.user_id,
            created_at=now,
            source="api",
        )

        period_record = {
            "period_id": period_id,
            "facility_id": body.facility_id,
            "commodity": body.commodity,
            "standard": body.standard,
            "start_date": body.start_date,
            "end_date": end_date,
            "status": PeriodStatusSchema.ACTIVE,
            "grace_period_end": grace_period_end,
            "carry_forward_balance": Decimal("0"),
            "opening_balance": body.opening_balance,
            "closing_balance": None,
            "total_inputs": Decimal("0"),
            "total_outputs": Decimal("0"),
            "total_losses": Decimal("0"),
            "metadata": body.metadata,
            "provenance": provenance,
            "created_at": now,
            "updated_at": now,
        }

        store = _get_period_store()
        store[period_id] = period_record

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Credit period created: id=%s facility=%s commodity=%s "
            "standard=%s start=%s end=%s",
            period_id,
            body.facility_id,
            body.commodity,
            standard_key,
            body.start_date.isoformat(),
            end_date.isoformat(),
        )

        return PeriodDetailSchema(
            **period_record,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to create credit period: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create credit period",
        )

# ---------------------------------------------------------------------------
# GET /periods/{period_id}
# ---------------------------------------------------------------------------

@router.get(
    "/periods/{period_id}",
    response_model=PeriodDetailSchema,
    summary="Get period details",
    description=(
        "Retrieve full details of a credit period including status, "
        "balances, grace period, and provenance information."
    ),
    responses={
        200: {"description": "Period details retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Period not found"},
    },
)
async def get_period(
    request: Request,
    period_id: str = Depends(validate_period_id),
    user: AuthUser = Depends(
        require_permission("eudr-mbc:periods:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> PeriodDetailSchema:
    """Get details of a specific credit period.

    Args:
        period_id: Unique period identifier.
        user: Authenticated user with periods:read permission.

    Returns:
        PeriodDetailSchema with full period details.

    Raises:
        HTTPException: 404 if period not found.
    """
    start = time.monotonic()
    try:
        store = _get_period_store()
        record = store.get(period_id)

        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Credit period {period_id} not found",
            )

        elapsed_ms = (time.monotonic() - start) * 1000.0

        return PeriodDetailSchema(
            **record,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to get period %s: %s", period_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve credit period",
        )

# ---------------------------------------------------------------------------
# PUT /periods/{period_id}
# ---------------------------------------------------------------------------

@router.put(
    "/periods/{period_id}",
    response_model=PeriodDetailSchema,
    summary="Extend credit period",
    description=(
        "Extend the end date of an active credit period. The new end date "
        "must be after the current end date. Requires justification."
    ),
    responses={
        200: {"description": "Period extended successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Period not found"},
        409: {"model": ErrorResponse, "description": "Period cannot be extended"},
    },
)
async def extend_period(
    request: Request,
    body: ExtendPeriodSchema,
    period_id: str = Depends(validate_period_id),
    user: AuthUser = Depends(
        require_permission("eudr-mbc:periods:extend")
    ),
    _rate: None = Depends(rate_limit_write),
) -> PeriodDetailSchema:
    """Extend the end date of a credit period.

    Args:
        body: Extension parameters including new end date and reason.
        period_id: Period to extend.
        user: Authenticated user with periods:extend permission.

    Returns:
        PeriodDetailSchema with updated period details.

    Raises:
        HTTPException: 404 if not found, 409 if period cannot be extended.
    """
    start = time.monotonic()
    try:
        store = _get_period_store()
        record = store.get(period_id)

        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Credit period {period_id} not found",
            )

        # Only active periods can be extended
        if record["status"] != PeriodStatusSchema.ACTIVE:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Period {period_id} cannot be extended "
                    f"(current status: {record['status'].value})"
                ),
            )

        # New end date must be after current end date
        if body.new_end_date <= record["end_date"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="new_end_date must be after the current end date",
            )

        now = utcnow()
        record["end_date"] = body.new_end_date
        record["grace_period_end"] = body.new_end_date + timedelta(days=5)
        record["updated_at"] = now

        # Record extension in metadata
        extensions = record["metadata"].get("extensions", [])
        extensions.append({
            "previous_end_date": str(record["end_date"]),
            "new_end_date": str(body.new_end_date),
            "reason": body.reason,
            "extended_by": body.operator_id,
            "extended_at": str(now),
        })
        record["metadata"]["extensions"] = extensions

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Credit period extended: id=%s new_end=%s reason=%s",
            period_id,
            body.new_end_date.isoformat(),
            body.reason[:100],
        )

        return PeriodDetailSchema(
            **record,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to extend period %s: %s", period_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to extend credit period",
        )

# ---------------------------------------------------------------------------
# POST /periods/rollover
# ---------------------------------------------------------------------------

@router.post(
    "/periods/rollover",
    response_model=RolloverResultSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Trigger period rollover",
    description=(
        "Close the current credit period and create a new one with "
        "carry-forward balance. The carry-forward percentage determines "
        "how much of the closing balance transfers to the new period."
    ),
    responses={
        201: {"description": "Rollover completed successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Period not found"},
        409: {"model": ErrorResponse, "description": "Period cannot be rolled over"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def rollover_period(
    request: Request,
    body: RolloverPeriodSchema,
    user: AuthUser = Depends(
        require_permission("eudr-mbc:periods:rollover")
    ),
    _rate: None = Depends(rate_limit_write),
) -> RolloverResultSchema:
    """Trigger period rollover.

    Args:
        body: Rollover parameters including closing period ID and
            carry-forward percentage.
        user: Authenticated user with periods:rollover permission.

    Returns:
        RolloverResultSchema with closed and new period details.

    Raises:
        HTTPException: 404 if closing period not found, 409 if not rollable.
    """
    start = time.monotonic()
    try:
        store = _get_period_store()
        closing_period = store.get(body.closing_period_id)

        if closing_period is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Period {body.closing_period_id} not found",
            )

        # Only active or reconciling periods can be rolled over
        if closing_period["status"] not in (
            PeriodStatusSchema.ACTIVE,
            PeriodStatusSchema.RECONCILING,
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Period {body.closing_period_id} cannot be rolled over "
                    f"(current status: {closing_period['status'].value})"
                ),
            )

        now = utcnow()

        # Calculate carry-forward amount
        closing_balance = closing_period.get("closing_balance")
        if closing_balance is None:
            closing_balance = closing_period.get(
                "opening_balance", Decimal("0")
            ) + closing_period.get(
                "total_inputs", Decimal("0")
            ) - closing_period.get(
                "total_outputs", Decimal("0")
            ) - closing_period.get(
                "total_losses", Decimal("0")
            )

        carry_forward_amount = (
            closing_balance * Decimal(str(body.carry_forward_percent / 100.0))
        )

        # Close the current period
        closing_period["status"] = PeriodStatusSchema.CLOSED
        closing_period["closing_balance"] = closing_balance
        closing_period["updated_at"] = now

        # Create new period
        new_period_id = str(uuid.uuid4())
        standard_key = closing_period["standard"].value if hasattr(
            closing_period["standard"], "value"
        ) else str(closing_period["standard"])
        period_days = _STANDARD_PERIOD_DAYS.get(standard_key, 365)
        new_start = closing_period["end_date"]
        new_end = new_start + timedelta(days=period_days)

        provenance_hash = _compute_provenance_hash({
            "closing_period_id": body.closing_period_id,
            "new_period_id": new_period_id,
            "carry_forward_amount": str(carry_forward_amount),
            "operator": user.user_id,
        })
        provenance = ProvenanceInfo(
            provenance_hash=provenance_hash,
            created_by=user.user_id,
            created_at=now,
            source="api",
        )

        new_period_record = {
            "period_id": new_period_id,
            "facility_id": closing_period["facility_id"],
            "commodity": closing_period["commodity"],
            "standard": closing_period["standard"],
            "start_date": new_start,
            "end_date": new_end,
            "status": PeriodStatusSchema.ACTIVE,
            "grace_period_end": new_end + timedelta(days=5),
            "carry_forward_balance": carry_forward_amount,
            "opening_balance": carry_forward_amount,
            "closing_balance": None,
            "total_inputs": Decimal("0"),
            "total_outputs": Decimal("0"),
            "total_losses": Decimal("0"),
            "metadata": {"rollover_from": body.closing_period_id},
            "provenance": provenance,
            "created_at": now,
            "updated_at": now,
        }

        store[new_period_id] = new_period_record

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Period rollover: closed=%s new=%s carry_forward=%s (%.1f%%)",
            body.closing_period_id,
            new_period_id,
            carry_forward_amount,
            body.carry_forward_percent,
        )

        return RolloverResultSchema(
            closed_period=PeriodDetailSchema(**closing_period),
            new_period=PeriodDetailSchema(**new_period_record),
            carry_forward_amount=carry_forward_amount,
            carry_forward_percent=body.carry_forward_percent,
            provenance=provenance,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to rollover period: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to rollover credit period",
        )

# ---------------------------------------------------------------------------
# GET /periods/active/{facility_id}
# ---------------------------------------------------------------------------

@router.get(
    "/periods/active/{facility_id}",
    response_model=ActivePeriodsSchema,
    summary="Get active periods for facility",
    description=(
        "List all active credit periods for a specific facility. "
        "Returns periods with status 'active' or 'reconciling'."
    ),
    responses={
        200: {"description": "Active periods retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_active_periods(
    request: Request,
    facility_id: str = Depends(validate_facility_id),
    user: AuthUser = Depends(
        require_permission("eudr-mbc:periods:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ActivePeriodsSchema:
    """Get active credit periods for a facility.

    Args:
        facility_id: Facility identifier.
        user: Authenticated user with periods:read permission.

    Returns:
        ActivePeriodsSchema listing all active periods.
    """
    start = time.monotonic()
    try:
        store = _get_period_store()
        active_periods = []

        for period in store.values():
            if period["facility_id"] != facility_id:
                continue
            if period["status"] in (
                PeriodStatusSchema.ACTIVE,
                PeriodStatusSchema.RECONCILING,
            ):
                active_periods.append(PeriodDetailSchema(**period))

        # Sort by start_date descending
        active_periods.sort(key=lambda p: p.start_date, reverse=True)

        elapsed_ms = (time.monotonic() - start) * 1000.0

        return ActivePeriodsSchema(
            facility_id=facility_id,
            periods=active_periods,
            total_count=len(active_periods),
            processing_time_ms=elapsed_ms,
            timestamp=utcnow(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get active periods for %s: %s",
            facility_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve active periods",
        )

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "router",
]
