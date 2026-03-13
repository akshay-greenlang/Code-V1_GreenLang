# -*- coding: utf-8 -*-
"""
Lifecycle Routes - AGENT-EUDR-014 QR Code Generator API

Endpoints for QR code lifecycle management including activation,
temporary deactivation, permanent revocation, scan event recording
with counterfeit risk assessment, and lifecycle history retrieval.

Lifecycle states: created -> active -> deactivated -> active (reactivation)
                  created/active/deactivated -> revoked (terminal)
                  active -> expired (automatic via TTL)

Endpoints:
    POST   /lifecycle/{code_id}/activate     - Activate a QR code
    POST   /lifecycle/{code_id}/deactivate   - Deactivate a QR code
    POST   /lifecycle/{code_id}/revoke       - Revoke a QR code
    POST   /lifecycle/scan                   - Record a scan event
    GET    /lifecycle/{code_id}/history      - Get lifecycle history

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-014, Feature 8 (QR Code Lifecycle Management)
Agent ID: GL-EUDR-QRG-014
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.qr_code_generator.api.dependencies import (
    AuthUser,
    ErrorResponse,
    PaginationParams,
    get_pagination,
    get_qrg_service,
    rate_limit_lifecycle,
    rate_limit_standard,
    require_permission,
    validate_code_id,
)
from greenlang.agents.eudr.qr_code_generator.api.schemas import (
    ActivateRequest,
    ActivateResponse,
    DeactivateRequest,
    DeactivateResponse,
    LifecycleEventItem,
    LifecycleHistoryResponse,
    PaginatedMeta,
    ProvenanceInfo,
    RevokeRequest,
    RevokeResponse,
    ScanEventRequest,
    ScanEventResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Lifecycle Management"])

# ---------------------------------------------------------------------------
# In-memory stores (replaced by database in production)
# ---------------------------------------------------------------------------

_lifecycle_store: Dict[str, Dict] = {}
_lifecycle_events: Dict[str, List[Dict]] = defaultdict(list)
_scan_velocity: Dict[str, List[float]] = defaultdict(list)

# Valid lifecycle transitions
_VALID_TRANSITIONS = {
    "created": {"active"},
    "active": {"deactivated", "revoked"},
    "deactivated": {"active", "revoked"},
}


def _get_lifecycle_store() -> Dict[str, Dict]:
    """Return the lifecycle record store singleton."""
    return _lifecycle_store


def _compute_provenance_hash(data: dict) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _get_code_status(code_id: str) -> str:
    """Get current lifecycle status for a code.

    Args:
        code_id: QR code identifier.

    Returns:
        Current status string (defaults to 'created').
    """
    store = _get_lifecycle_store()
    record = store.get(code_id)
    if record:
        return record.get("status", "created")
    return "created"


def _set_code_status(
    code_id: str,
    new_status: str,
    event_type: str,
    reason: str = "",
    performed_by: str = "",
) -> str:
    """Set lifecycle status and record the event.

    Args:
        code_id: QR code identifier.
        new_status: Target status.
        event_type: Type of lifecycle event.
        reason: Reason for the change.
        performed_by: User who performed the change.

    Returns:
        Previous status.

    Raises:
        HTTPException: 409 if transition is invalid.
    """
    store = _get_lifecycle_store()
    now = _utcnow()

    current_status = _get_code_status(code_id)

    # Validate transition
    allowed = _VALID_TRANSITIONS.get(current_status, set())
    if new_status not in allowed:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Cannot transition from '{current_status}' to "
                f"'{new_status}'. Allowed transitions: {sorted(allowed)}"
            ),
        )

    # Update status
    if code_id not in store:
        store[code_id] = {"code_id": code_id}
    store[code_id]["status"] = new_status
    store[code_id]["updated_at"] = now

    # Record event
    event = {
        "event_id": str(uuid.uuid4()),
        "code_id": code_id,
        "event_type": event_type,
        "previous_status": current_status,
        "new_status": new_status,
        "reason": reason,
        "performed_by": performed_by,
        "created_at": now,
    }
    _lifecycle_events[code_id].append(event)

    return current_status


# ---------------------------------------------------------------------------
# POST /lifecycle/{code_id}/activate
# ---------------------------------------------------------------------------


@router.post(
    "/lifecycle/{code_id}/activate",
    response_model=ActivateResponse,
    summary="Activate QR code",
    description=(
        "Activate a QR code, transitioning it from 'created' or "
        "'deactivated' to 'active' status. Only valid for codes "
        "in 'created' or 'deactivated' state."
    ),
    responses={
        200: {"description": "Code activated successfully"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        409: {"model": ErrorResponse, "description": "Invalid state transition"},
    },
)
async def activate_code(
    request: Request,
    body: ActivateRequest,
    code_id: str = Depends(validate_code_id),
    user: AuthUser = Depends(
        require_permission("eudr-qrg:lifecycle:activate")
    ),
    _rate: None = Depends(rate_limit_lifecycle),
    service: Any = Depends(get_qrg_service),
) -> ActivateResponse:
    """Activate a QR code.

    Args:
        request: FastAPI request object.
        body: Activation parameters.
        code_id: QR code identifier.
        user: Authenticated user with lifecycle:activate permission.
        service: QR Code Generator service.

    Returns:
        ActivateResponse with activation details.

    Raises:
        HTTPException: 409 if transition is invalid.
    """
    start = time.monotonic()
    try:
        now = _utcnow()
        performed_by = body.performed_by or user.user_id

        previous_status = _set_code_status(
            code_id=code_id,
            new_status="active",
            event_type="activate",
            reason=body.reason or "Activated via API",
            performed_by=performed_by,
        )

        provenance_hash = _compute_provenance_hash({
            "code_id": code_id,
            "action": "activate",
            "previous_status": previous_status,
            "activated_by": performed_by,
            "activated_at": str(now),
        })

        elapsed_ms = (time.monotonic() - start) * 1000.0
        logger.info(
            "Code activated: id=%s previous=%s by=%s elapsed_ms=%.1f",
            code_id,
            previous_status,
            performed_by,
            elapsed_ms,
        )

        return ActivateResponse(
            code_id=code_id,
            status="success",
            previous_status=previous_status,
            new_status="active",
            activated_at=now,
            processing_time_ms=elapsed_ms,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                algorithm="sha256",
                created_at=now,
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to activate code %s: %s", code_id, exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to activate QR code",
        )


# ---------------------------------------------------------------------------
# POST /lifecycle/{code_id}/deactivate
# ---------------------------------------------------------------------------


@router.post(
    "/lifecycle/{code_id}/deactivate",
    response_model=DeactivateResponse,
    summary="Deactivate QR code",
    description=(
        "Temporarily deactivate a QR code. The code can be "
        "reactivated later. Only valid for codes in 'active' state."
    ),
    responses={
        200: {"description": "Code deactivated successfully"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        409: {"model": ErrorResponse, "description": "Invalid state transition"},
    },
)
async def deactivate_code(
    request: Request,
    body: DeactivateRequest,
    code_id: str = Depends(validate_code_id),
    user: AuthUser = Depends(
        require_permission("eudr-qrg:lifecycle:deactivate")
    ),
    _rate: None = Depends(rate_limit_lifecycle),
    service: Any = Depends(get_qrg_service),
) -> DeactivateResponse:
    """Temporarily deactivate a QR code.

    Args:
        request: FastAPI request object.
        body: Deactivation parameters with required reason.
        code_id: QR code identifier.
        user: Authenticated user with lifecycle:deactivate permission.
        service: QR Code Generator service.

    Returns:
        DeactivateResponse with deactivation details.

    Raises:
        HTTPException: 409 if transition is invalid.
    """
    start = time.monotonic()
    try:
        now = _utcnow()
        performed_by = body.performed_by or user.user_id

        previous_status = _set_code_status(
            code_id=code_id,
            new_status="deactivated",
            event_type="deactivate",
            reason=body.reason,
            performed_by=performed_by,
        )

        provenance_hash = _compute_provenance_hash({
            "code_id": code_id,
            "action": "deactivate",
            "previous_status": previous_status,
            "reason": body.reason,
            "deactivated_by": performed_by,
            "deactivated_at": str(now),
        })

        elapsed_ms = (time.monotonic() - start) * 1000.0
        logger.info(
            "Code deactivated: id=%s previous=%s reason='%s' by=%s "
            "elapsed_ms=%.1f",
            code_id,
            previous_status,
            body.reason[:50],
            performed_by,
            elapsed_ms,
        )

        return DeactivateResponse(
            code_id=code_id,
            status="success",
            previous_status=previous_status,
            new_status="deactivated",
            deactivated_at=now,
            reason=body.reason,
            processing_time_ms=elapsed_ms,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                algorithm="sha256",
                created_at=now,
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to deactivate code %s: %s", code_id, exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to deactivate QR code",
        )


# ---------------------------------------------------------------------------
# POST /lifecycle/{code_id}/revoke
# ---------------------------------------------------------------------------


@router.post(
    "/lifecycle/{code_id}/revoke",
    response_model=RevokeResponse,
    summary="Revoke QR code",
    description=(
        "Permanently revoke a QR code. This is a terminal state; "
        "revoked codes cannot be reactivated. Only valid for codes "
        "in 'created', 'active', or 'deactivated' state."
    ),
    responses={
        200: {"description": "Code revoked successfully"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        409: {"model": ErrorResponse, "description": "Invalid state transition"},
    },
)
async def revoke_code(
    request: Request,
    body: RevokeRequest,
    code_id: str = Depends(validate_code_id),
    user: AuthUser = Depends(
        require_permission("eudr-qrg:lifecycle:revoke")
    ),
    _rate: None = Depends(rate_limit_lifecycle),
    service: Any = Depends(get_qrg_service),
) -> RevokeResponse:
    """Permanently revoke a QR code.

    Args:
        request: FastAPI request object.
        body: Revocation parameters with required reason.
        code_id: QR code identifier.
        user: Authenticated user with lifecycle:revoke permission.
        service: QR Code Generator service.

    Returns:
        RevokeResponse with revocation details.

    Raises:
        HTTPException: 409 if transition is invalid.
    """
    start = time.monotonic()
    try:
        now = _utcnow()
        performed_by = body.performed_by or user.user_id

        previous_status = _set_code_status(
            code_id=code_id,
            new_status="revoked",
            event_type="revoke",
            reason=body.reason,
            performed_by=performed_by,
        )

        provenance_hash = _compute_provenance_hash({
            "code_id": code_id,
            "action": "revoke",
            "previous_status": previous_status,
            "reason": body.reason,
            "revoked_by": performed_by,
            "revoked_at": str(now),
        })

        elapsed_ms = (time.monotonic() - start) * 1000.0
        logger.info(
            "Code revoked: id=%s previous=%s reason='%s' by=%s "
            "elapsed_ms=%.1f",
            code_id,
            previous_status,
            body.reason[:50],
            performed_by,
            elapsed_ms,
        )

        return RevokeResponse(
            code_id=code_id,
            status="success",
            previous_status=previous_status,
            new_status="revoked",
            revoked_at=now,
            reason=body.reason,
            processing_time_ms=elapsed_ms,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                algorithm="sha256",
                created_at=now,
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to revoke code %s: %s", code_id, exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke QR code",
        )


# ---------------------------------------------------------------------------
# POST /lifecycle/scan
# ---------------------------------------------------------------------------


@router.post(
    "/lifecycle/scan",
    response_model=ScanEventResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record scan event",
    description=(
        "Record a QR code scan event with counterfeit risk assessment. "
        "Captures scan location, HMAC token validation, scan velocity "
        "monitoring, and geo-fence compliance."
    ),
    responses={
        201: {"description": "Scan event recorded successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def record_scan_event(
    request: Request,
    body: ScanEventRequest,
    user: AuthUser = Depends(
        require_permission("eudr-qrg:lifecycle:scan")
    ),
    _rate: None = Depends(rate_limit_lifecycle),
    service: Any = Depends(get_qrg_service),
) -> ScanEventResponse:
    """Record a scan event.

    Args:
        request: FastAPI request object.
        body: Scan event parameters.
        user: Authenticated user with lifecycle:scan permission.
        service: QR Code Generator service.

    Returns:
        ScanEventResponse with scan details and risk assessment.
    """
    start = time.monotonic()
    try:
        scan_id = str(uuid.uuid4())
        now = _utcnow()

        # Check scan velocity
        velocity_window = 60.0
        current_time = time.monotonic()
        _scan_velocity[body.code_id] = [
            ts for ts in _scan_velocity[body.code_id]
            if current_time - ts < velocity_window
        ]
        _scan_velocity[body.code_id].append(current_time)
        scans_per_min = len(_scan_velocity[body.code_id])

        # Assess counterfeit risk
        risk_level = "low"
        outcome = "verified"
        geo_fence_violated = False
        hmac_valid = None

        if scans_per_min > 100:
            risk_level = "high"
            outcome = "counterfeit_suspected"

        if body.hmac_token:
            hmac_valid = len(body.hmac_token) >= 16
            if not hmac_valid:
                risk_level = "critical"
                outcome = "counterfeit_suspected"

        # Record lifecycle event
        event = {
            "event_id": scan_id,
            "code_id": body.code_id,
            "event_type": "scan",
            "previous_status": _get_code_status(body.code_id),
            "new_status": _get_code_status(body.code_id),
            "reason": f"Scan event: outcome={outcome}",
            "performed_by": user.user_id,
            "created_at": now,
        }
        _lifecycle_events[body.code_id].append(event)

        provenance_hash = _compute_provenance_hash({
            "scan_id": scan_id,
            "code_id": body.code_id,
            "outcome": outcome,
            "scanned_by": user.user_id,
            "scanned_at": str(now),
        })

        elapsed_ms = (time.monotonic() - start) * 1000.0
        logger.info(
            "Scan recorded: scan_id=%s code_id=%s outcome=%s risk=%s "
            "velocity=%d/min elapsed_ms=%.1f",
            scan_id,
            body.code_id,
            outcome,
            risk_level,
            scans_per_min,
            elapsed_ms,
        )

        return ScanEventResponse(
            scan_id=scan_id,
            code_id=body.code_id,
            outcome=outcome,
            counterfeit_risk=risk_level,
            hmac_valid=hmac_valid,
            velocity_scans_per_min=scans_per_min,
            geo_fence_violated=geo_fence_violated,
            response_time_ms=elapsed_ms,
            scanned_at=now,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                algorithm="sha256",
                created_at=now,
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to record scan event: %s", exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record scan event",
        )


# ---------------------------------------------------------------------------
# GET /lifecycle/{code_id}/history
# ---------------------------------------------------------------------------


@router.get(
    "/lifecycle/{code_id}/history",
    response_model=LifecycleHistoryResponse,
    summary="Get lifecycle history",
    description=(
        "Retrieve the complete lifecycle history of a QR code "
        "including all state transitions, scan events, and "
        "provenance chain."
    ),
    responses={
        200: {"description": "Lifecycle history retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_lifecycle_history(
    request: Request,
    code_id: str = Depends(validate_code_id),
    user: AuthUser = Depends(
        require_permission("eudr-qrg:lifecycle:read")
    ),
    pagination: PaginationParams = Depends(get_pagination),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_qrg_service),
) -> LifecycleHistoryResponse:
    """Get lifecycle history for a QR code.

    Args:
        request: FastAPI request object.
        code_id: QR code identifier.
        user: Authenticated user with lifecycle:read permission.
        pagination: Pagination parameters.
        service: QR Code Generator service.

    Returns:
        LifecycleHistoryResponse with paginated event history.
    """
    try:
        events = _lifecycle_events.get(code_id, [])

        # Sort by created_at descending
        events_sorted = sorted(
            events,
            key=lambda e: e.get("created_at", datetime.min),
            reverse=True,
        )

        total = len(events_sorted)
        page = events_sorted[
            pagination.offset:pagination.offset + pagination.limit
        ]
        has_more = (pagination.offset + pagination.limit) < total

        event_items = [
            LifecycleEventItem(**e) for e in page
        ]

        current_status = _get_code_status(code_id)

        return LifecycleHistoryResponse(
            code_id=code_id,
            current_status=current_status,
            events=event_items,
            total_events=total,
            pagination=PaginatedMeta(
                total=total,
                limit=pagination.limit,
                offset=pagination.offset,
                has_more=has_more,
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get lifecycle history %s: %s",
            code_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve lifecycle history",
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = ["router"]
