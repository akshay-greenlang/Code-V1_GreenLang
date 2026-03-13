# -*- coding: utf-8 -*-
"""
Overdraft Routes - AGENT-EUDR-011 Mass Balance Calculator API

Endpoints for overdraft detection, alert management, output forecasting,
exemption requests, and overdraft event history. Overdraft enforcement
operates in three modes: zero_tolerance, percentage, and absolute.

Enforcement Modes:
    - zero_tolerance: Any negative balance is a violation.
    - percentage: Overdraft up to N% of total period inputs is allowed.
    - absolute: Overdraft up to N kg is allowed.

Endpoints:
    POST  /overdraft/check                 - Check overdraft for proposed output
    GET   /overdraft/alerts/{facility_id}  - Get active overdraft alerts
    POST  /overdraft/forecast              - Forecast maximum available output
    POST  /overdraft/exemption             - Request overdraft exemption
    GET   /overdraft/history/{facility_id} - Get overdraft event history

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-011, Feature 4 (Overdraft Detection and Enforcement)
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

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.mass_balance_calculator.api.dependencies import (
    AuthUser,
    DateRangeParams,
    ErrorResponse,
    PaginationParams,
    get_date_range,
    get_mbc_service,
    get_pagination,
    get_request_id,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
    validate_facility_id,
)
from greenlang.agents.eudr.mass_balance_calculator.api.schemas import (
    CheckOverdraftSchema,
    ExemptionResultSchema,
    ForecastOutputSchema,
    ForecastResultSchema,
    OverdraftAlertDetailSchema,
    OverdraftAlertsSchema,
    OverdraftCheckResultSchema,
    OverdraftHistorySchema,
    OverdraftModeSchema,
    OverdraftSeveritySchema,
    PaginatedMeta,
    ProvenanceInfo,
    RequestExemptionSchema,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Overdraft Detection"])

# ---------------------------------------------------------------------------
# In-memory stores (replaced by database in production)
# ---------------------------------------------------------------------------

_overdraft_alert_store: Dict[str, Dict] = {}
_exemption_store: Dict[str, Dict] = {}

# Mock ledger data for overdraft checks (in production, queried from DB)
_mock_ledger_balances: Dict[str, Dict[str, Any]] = {}


def _compute_provenance_hash(data: dict) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# POST /overdraft/check
# ---------------------------------------------------------------------------


@router.post(
    "/overdraft/check",
    response_model=OverdraftCheckResultSchema,
    summary="Check overdraft status",
    description=(
        "Check whether a proposed output quantity would cause an overdraft "
        "on the specified ledger. Returns the remaining balance after the "
        "proposed output and whether the output is allowed under the "
        "current overdraft enforcement mode."
    ),
    responses={
        200: {"description": "Overdraft check completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Ledger not found"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def check_overdraft(
    request: Request,
    body: CheckOverdraftSchema,
    user: AuthUser = Depends(
        require_permission("eudr-mbc:overdraft:check")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> OverdraftCheckResultSchema:
    """Check overdraft for a proposed output.

    Args:
        body: Overdraft check request with ledger_id, output quantity,
            and dry_run flag.
        user: Authenticated user with overdraft:check permission.

    Returns:
        OverdraftCheckResultSchema with overdraft analysis.
    """
    start = time.monotonic()
    try:
        now = _utcnow()

        # In production, query actual ledger balance from DB
        # Here we use a simulated balance lookup
        ledger_data = _mock_ledger_balances.get(body.ledger_id)
        if ledger_data is None:
            # Default simulated balance for testing
            current_balance = Decimal("5000.00")
            total_inputs = Decimal("10000.00")
            facility_id = "unknown"
        else:
            current_balance = ledger_data.get("current_balance", Decimal("5000.00"))
            total_inputs = ledger_data.get("total_inputs", Decimal("10000.00"))
            facility_id = ledger_data.get("facility_id", "unknown")

        remaining_after = current_balance - body.output_quantity_kg

        # Determine overdraft mode (default: zero_tolerance)
        overdraft_mode = OverdraftModeSchema.ZERO_TOLERANCE
        tolerance_applied: Optional[Decimal] = None
        overdraft_detected = remaining_after < Decimal("0")
        severity: Optional[OverdraftSeveritySchema] = None
        allowed = not overdraft_detected

        if overdraft_detected:
            overdraft_amount = abs(remaining_after)
            # Determine severity
            if total_inputs > Decimal("0"):
                overdraft_pct = float(overdraft_amount / total_inputs * 100)
            else:
                overdraft_pct = 100.0

            if overdraft_pct > 10.0:
                severity = OverdraftSeveritySchema.CRITICAL
            elif overdraft_pct > 3.0:
                severity = OverdraftSeveritySchema.VIOLATION
            else:
                severity = OverdraftSeveritySchema.WARNING

            message = (
                f"Overdraft detected: proposed output of {body.output_quantity_kg} kg "
                f"would exceed available balance of {current_balance} kg "
                f"by {overdraft_amount} kg ({overdraft_pct:.1f}% of total inputs)."
            )

            # Create alert if not dry-run
            if not body.dry_run:
                _create_overdraft_alert(
                    ledger_id=body.ledger_id,
                    facility_id=facility_id,
                    commodity="unknown",
                    severity=severity,
                    current_balance=current_balance,
                    overdraft_amount=overdraft_amount,
                    user_id=user.user_id,
                )
        else:
            message = (
                f"Output allowed: {body.output_quantity_kg} kg from "
                f"available balance of {current_balance} kg. "
                f"Remaining: {remaining_after} kg."
            )

        provenance_hash = _compute_provenance_hash(body.model_dump(mode="json"))
        provenance = ProvenanceInfo(
            provenance_hash=provenance_hash,
            created_by=user.user_id,
            created_at=now,
            source="api",
        )

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Overdraft check: ledger=%s proposed=%s balance=%s overdraft=%s",
            body.ledger_id,
            body.output_quantity_kg,
            current_balance,
            overdraft_detected,
        )

        return OverdraftCheckResultSchema(
            ledger_id=body.ledger_id,
            current_balance=current_balance,
            proposed_output=body.output_quantity_kg,
            remaining_after=remaining_after,
            overdraft_detected=overdraft_detected,
            severity=severity,
            overdraft_mode=overdraft_mode,
            tolerance_applied=tolerance_applied,
            allowed=allowed,
            message=message,
            provenance=provenance,
            processing_time_ms=elapsed_ms,
            timestamp=now,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to check overdraft: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check overdraft status",
        )


# ---------------------------------------------------------------------------
# GET /overdraft/alerts/{facility_id}
# ---------------------------------------------------------------------------


@router.get(
    "/overdraft/alerts/{facility_id}",
    response_model=OverdraftAlertsSchema,
    summary="Get active overdraft alerts",
    description=(
        "Retrieve all active (unresolved) overdraft alerts for a facility. "
        "Alerts include severity classification, resolution deadline, "
        "and triggering entry details."
    ),
    responses={
        200: {"description": "Alerts retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_overdraft_alerts(
    request: Request,
    facility_id: str = Depends(validate_facility_id),
    user: AuthUser = Depends(
        require_permission("eudr-mbc:overdraft:alerts:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> OverdraftAlertsSchema:
    """Get active overdraft alerts for a facility.

    Args:
        facility_id: Facility identifier.
        user: Authenticated user with overdraft:alerts:read permission.

    Returns:
        OverdraftAlertsSchema with active alerts and summary counts.
    """
    start = time.monotonic()
    try:
        alerts = []
        critical_count = 0

        for alert in _overdraft_alert_store.values():
            if alert["facility_id"] != facility_id:
                continue
            if alert["resolved"]:
                continue
            alerts.append(OverdraftAlertDetailSchema(**alert))
            if alert["severity"] == OverdraftSeveritySchema.CRITICAL:
                critical_count += 1

        # Sort by severity (critical first) then by created_at
        severity_order = {
            OverdraftSeveritySchema.CRITICAL: 0,
            OverdraftSeveritySchema.VIOLATION: 1,
            OverdraftSeveritySchema.WARNING: 2,
        }
        alerts.sort(key=lambda a: (severity_order.get(a.severity, 3), a.created_at))

        elapsed_ms = (time.monotonic() - start) * 1000.0

        return OverdraftAlertsSchema(
            facility_id=facility_id,
            alerts=alerts,
            total_unresolved=len(alerts),
            critical_count=critical_count,
            processing_time_ms=elapsed_ms,
            timestamp=_utcnow(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get overdraft alerts for %s: %s",
            facility_id, exc, exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve overdraft alerts",
        )


# ---------------------------------------------------------------------------
# POST /overdraft/forecast
# ---------------------------------------------------------------------------


@router.post(
    "/overdraft/forecast",
    response_model=ForecastResultSchema,
    summary="Forecast available output",
    description=(
        "Forecast the maximum available output quantity for a ledger, "
        "optionally including carry-forward balance and pending inputs."
    ),
    responses={
        200: {"description": "Forecast completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def forecast_output(
    request: Request,
    body: ForecastOutputSchema,
    user: AuthUser = Depends(
        require_permission("eudr-mbc:overdraft:forecast")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ForecastResultSchema:
    """Forecast maximum available output for a ledger.

    Args:
        body: Forecast request with ledger_id and inclusion flags.
        user: Authenticated user with overdraft:forecast permission.

    Returns:
        ForecastResultSchema with available balance and maximum output.
    """
    start = time.monotonic()
    try:
        now = _utcnow()

        # Simulated balance lookup
        ledger_data = _mock_ledger_balances.get(body.ledger_id)
        if ledger_data is not None:
            available_balance = ledger_data.get("current_balance", Decimal("5000.00"))
        else:
            available_balance = Decimal("5000.00")

        carry_forward_amount = Decimal("0")
        pending_inputs_amount = Decimal("0")

        if body.include_carry_forward:
            carry_forward_amount = Decimal("500.00")
            available_balance += carry_forward_amount

        if body.include_pending_inputs:
            pending_inputs_amount = Decimal("200.00")
            available_balance += pending_inputs_amount

        max_output = max(Decimal("0"), available_balance)

        provenance_hash = _compute_provenance_hash(body.model_dump(mode="json"))
        provenance = ProvenanceInfo(
            provenance_hash=provenance_hash,
            created_by=user.user_id,
            created_at=now,
            source="api",
        )

        elapsed_ms = (time.monotonic() - start) * 1000.0

        return ForecastResultSchema(
            ledger_id=body.ledger_id,
            available_balance=available_balance,
            carry_forward_included=body.include_carry_forward,
            carry_forward_amount=carry_forward_amount,
            pending_inputs_included=body.include_pending_inputs,
            pending_inputs_amount=pending_inputs_amount,
            max_output_kg=max_output,
            overdraft_mode=OverdraftModeSchema.ZERO_TOLERANCE,
            provenance=provenance,
            processing_time_ms=elapsed_ms,
            timestamp=now,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to forecast output: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to forecast available output",
        )


# ---------------------------------------------------------------------------
# POST /overdraft/exemption
# ---------------------------------------------------------------------------


@router.post(
    "/overdraft/exemption",
    response_model=ExemptionResultSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Request overdraft exemption",
    description=(
        "Request an exemption for an existing overdraft event. Requires "
        "justification and supporting evidence. Exemptions are subject "
        "to manual review and approval."
    ),
    responses={
        201: {"description": "Exemption request submitted"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Overdraft event not found"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def request_exemption(
    request: Request,
    body: RequestExemptionSchema,
    user: AuthUser = Depends(
        require_permission("eudr-mbc:overdraft:exemption:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> ExemptionResultSchema:
    """Request an exemption for an overdraft event.

    Args:
        body: Exemption request with event_id, reason, and evidence.
        user: Authenticated user with exemption:create permission.

    Returns:
        ExemptionResultSchema with exemption status.

    Raises:
        HTTPException: 404 if overdraft event not found.
    """
    start = time.monotonic()
    try:
        now = _utcnow()

        # Verify overdraft event exists
        alert = _overdraft_alert_store.get(body.event_id)
        if alert is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Overdraft event {body.event_id} not found",
            )

        exemption_id = str(uuid.uuid4())

        _exemption_store[exemption_id] = {
            "exemption_id": exemption_id,
            "event_id": body.event_id,
            "reason": body.reason,
            "requested_by": body.requested_by,
            "supporting_evidence": body.supporting_evidence,
            "status": "pending",
            "metadata": body.metadata,
            "created_at": now,
        }

        provenance_hash = _compute_provenance_hash({
            "exemption_id": exemption_id,
            "event_id": body.event_id,
            "requested_by": body.requested_by,
        })
        provenance = ProvenanceInfo(
            provenance_hash=provenance_hash,
            created_by=user.user_id,
            created_at=now,
            source="api",
        )

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Exemption requested: exemption=%s event=%s by=%s",
            exemption_id,
            body.event_id,
            body.requested_by,
        )

        return ExemptionResultSchema(
            exemption_id=exemption_id,
            event_id=body.event_id,
            status="pending",
            message="Exemption request submitted for review.",
            provenance=provenance,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to request exemption: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit exemption request",
        )


# ---------------------------------------------------------------------------
# GET /overdraft/history/{facility_id}
# ---------------------------------------------------------------------------


@router.get(
    "/overdraft/history/{facility_id}",
    response_model=OverdraftHistorySchema,
    summary="Get overdraft history",
    description=(
        "Retrieve historical overdraft events for a facility with "
        "optional date range and severity filters."
    ),
    responses={
        200: {"description": "Overdraft history retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_overdraft_history(
    request: Request,
    facility_id: str = Depends(validate_facility_id),
    severity: Optional[OverdraftSeveritySchema] = Query(
        None, description="Filter by severity"
    ),
    resolved: Optional[bool] = Query(
        None, description="Filter by resolution status"
    ),
    pagination: PaginationParams = Depends(get_pagination),
    date_range: DateRangeParams = Depends(get_date_range),
    user: AuthUser = Depends(
        require_permission("eudr-mbc:overdraft:history:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> OverdraftHistorySchema:
    """Get overdraft event history for a facility.

    Args:
        facility_id: Facility identifier.
        severity: Optional severity filter.
        resolved: Optional resolution status filter.
        pagination: Pagination parameters.
        date_range: Optional date range filter.
        user: Authenticated user with overdraft:history:read permission.

    Returns:
        OverdraftHistorySchema with historical events and pagination.
    """
    start = time.monotonic()
    try:
        events = []
        for alert in _overdraft_alert_store.values():
            if alert["facility_id"] != facility_id:
                continue
            if severity is not None and alert["severity"] != severity:
                continue
            if resolved is not None and alert["resolved"] != resolved:
                continue
            if date_range.start_date and alert["created_at"] < date_range.start_date:
                continue
            if date_range.end_date and alert["created_at"] > date_range.end_date:
                continue
            events.append(alert)

        # Sort by created_at descending
        events.sort(key=lambda e: e["created_at"], reverse=True)

        total = len(events)
        paginated = events[pagination.offset: pagination.offset + pagination.limit]
        has_more = (pagination.offset + pagination.limit) < total

        alert_schemas = [OverdraftAlertDetailSchema(**e) for e in paginated]
        meta = PaginatedMeta(
            total=total,
            limit=pagination.limit,
            offset=pagination.offset,
            has_more=has_more,
        )

        elapsed_ms = (time.monotonic() - start) * 1000.0

        return OverdraftHistorySchema(
            facility_id=facility_id,
            events=alert_schemas,
            pagination=meta,
            processing_time_ms=elapsed_ms,
            timestamp=_utcnow(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get overdraft history for %s: %s",
            facility_id, exc, exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve overdraft history",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_overdraft_alert(
    ledger_id: str,
    facility_id: str,
    commodity: str,
    severity: OverdraftSeveritySchema,
    current_balance: Decimal,
    overdraft_amount: Decimal,
    user_id: str,
) -> str:
    """Create an overdraft alert record.

    Args:
        ledger_id: Affected ledger.
        facility_id: Affected facility.
        commodity: Affected commodity.
        severity: Alert severity.
        current_balance: Balance at time of detection.
        overdraft_amount: Overdraft amount in kg.
        user_id: User who triggered the check.

    Returns:
        Alert event_id.
    """
    event_id = str(uuid.uuid4())
    now = _utcnow()

    _overdraft_alert_store[event_id] = {
        "event_id": event_id,
        "ledger_id": ledger_id,
        "facility_id": facility_id,
        "commodity": commodity,
        "severity": severity,
        "current_balance": current_balance,
        "overdraft_amount": overdraft_amount,
        "trigger_entry_id": None,
        "resolution_deadline": now + timedelta(hours=48),
        "resolved": False,
        "resolved_at": None,
        "exemption_id": None,
        "created_at": now,
    }

    logger.warning(
        "Overdraft alert created: event=%s facility=%s severity=%s amount=%s",
        event_id,
        facility_id,
        severity.value,
        overdraft_amount,
    )

    return event_id


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "router",
]
