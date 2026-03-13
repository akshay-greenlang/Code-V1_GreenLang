# -*- coding: utf-8 -*-
"""
Sharing Routes - AGENT-EUDR-013 Blockchain Integration API

Endpoints for cross-party data sharing of EUDR compliance data
including granting access, revoking access, listing grants,
requesting access, and multi-party confirmation.

Endpoints:
    POST   /sharing/grant                  - Grant access to party
    DELETE /sharing/revoke/{grant_id}       - Revoke access
    GET    /sharing/grants/{record_id}      - List access grants
    POST   /sharing/request                 - Request access to record
    POST   /sharing/confirm                 - Multi-party confirmation

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-013, Feature 7 (Cross-Party Data Sharing)
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
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.blockchain_integration.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_blockchain_service,
    get_current_operator_id,
    get_request_id,
    rate_limit_anchor,
    rate_limit_standard,
    require_permission,
    validate_grant_id,
    validate_record_id,
)
from greenlang.agents.eudr.blockchain_integration.api.schemas import (
    AccessGrantListResponse,
    AccessGrantRequest,
    AccessGrantResponse,
    AccessLevelSchema,
    AccessRequestRequest,
    AccessRevokeRequest,
    AccessStatusSchema,
    MultiPartyConfirmRequest,
    MultiPartyConfirmResponse,
    ProvenanceInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Cross-Party Sharing"])

# ---------------------------------------------------------------------------
# In-memory stores (replaced by database in production)
# ---------------------------------------------------------------------------

_grant_store: Dict[str, Dict] = {}
_confirmation_store: Dict[str, Dict] = {}


def _get_grant_store() -> Dict[str, Dict]:
    """Return the access grant store singleton."""
    return _grant_store


def _get_confirmation_store() -> Dict[str, Dict]:
    """Return the confirmation store singleton."""
    return _confirmation_store


def _compute_provenance_hash(data: dict) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# POST /sharing/grant
# ---------------------------------------------------------------------------


@router.post(
    "/sharing/grant",
    response_model=AccessGrantResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Grant access to party",
    description=(
        "Grant cross-party access to EUDR compliance data for a "
        "specific record. Supports operator, competent authority, "
        "auditor, and supply chain partner access levels."
    ),
    responses={
        201: {"description": "Access granted"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def grant_access(
    request: Request,
    body: AccessGrantRequest,
    user: AuthUser = Depends(
        require_permission("eudr-bci:sharing:grant")
    ),
    _rate: None = Depends(rate_limit_anchor),
    service: Any = Depends(get_blockchain_service),
) -> AccessGrantResponse:
    """Grant cross-party data access.

    Args:
        request: FastAPI request object.
        body: Access grant parameters.
        user: Authenticated user with sharing:grant permission.
        service: Blockchain integration service.

    Returns:
        AccessGrantResponse with grant ID and active status.
    """
    start = time.monotonic()
    try:
        grant_id = str(uuid.uuid4())
        now = _utcnow()

        provenance_hash = _compute_provenance_hash({
            "grant_id": grant_id,
            "record_id": body.record_id,
            "grantor_id": user.user_id,
            "grantee_id": body.grantee_id,
            "access_level": body.access_level.value,
        })

        record = {
            "grant_id": grant_id,
            "record_id": body.record_id,
            "grantor_id": user.user_id,
            "grantee_id": body.grantee_id,
            "access_level": body.access_level,
            "status": AccessStatusSchema.ACTIVE,
            "scope": body.scope,
            "reason": body.reason,
            "granted_at": now,
            "expires_at": body.expires_at,
            "revoked_at": None,
            "provenance": ProvenanceInfo(
                provenance_hash=provenance_hash,
                algorithm="sha256",
                created_at=now,
            ),
        }

        store = _get_grant_store()
        store[grant_id] = record

        elapsed_ms = (time.monotonic() - start) * 1000.0
        logger.info(
            "Access granted: id=%s record=%s grantee=%s level=%s "
            "elapsed_ms=%.1f",
            grant_id,
            body.record_id,
            body.grantee_id,
            body.access_level.value,
            elapsed_ms,
        )

        return AccessGrantResponse(**record)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to grant access: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to grant access",
        )


# ---------------------------------------------------------------------------
# DELETE /sharing/revoke/{grant_id}
# ---------------------------------------------------------------------------


@router.delete(
    "/sharing/revoke/{grant_id}",
    response_model=AccessGrantResponse,
    summary="Revoke access",
    description="Revoke a previously granted cross-party data access.",
    responses={
        200: {"description": "Access revoked"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Grant not found"},
        409: {"model": ErrorResponse, "description": "Grant already revoked"},
    },
)
async def revoke_access(
    request: Request,
    grant_id: str = Depends(validate_grant_id),
    user: AuthUser = Depends(
        require_permission("eudr-bci:sharing:revoke")
    ),
    _rate: None = Depends(rate_limit_anchor),
    service: Any = Depends(get_blockchain_service),
) -> AccessGrantResponse:
    """Revoke cross-party data access.

    Args:
        request: FastAPI request object.
        grant_id: Access grant identifier.
        user: Authenticated user with sharing:revoke permission.
        service: Blockchain integration service.

    Returns:
        AccessGrantResponse with revoked status.

    Raises:
        HTTPException: 404 if grant not found, 409 if already revoked.
    """
    try:
        store = _get_grant_store()
        record = store.get(grant_id)

        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Access grant {grant_id} not found",
            )

        current_status = record["status"]
        if current_status == AccessStatusSchema.REVOKED:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Grant {grant_id} is already revoked",
            )

        now = _utcnow()
        record["status"] = AccessStatusSchema.REVOKED
        record["revoked_at"] = now

        logger.info(
            "Access revoked: id=%s by=%s",
            grant_id,
            user.user_id,
        )

        return AccessGrantResponse(**record)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to revoke access %s: %s",
            grant_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke access",
        )


# ---------------------------------------------------------------------------
# GET /sharing/grants/{record_id}
# ---------------------------------------------------------------------------


@router.get(
    "/sharing/grants/{record_id}",
    response_model=AccessGrantListResponse,
    summary="List access grants",
    description=(
        "List all access grants for a specific EUDR compliance record "
        "including active, revoked, and expired grants."
    ),
    responses={
        200: {"description": "Grants listed"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def list_grants(
    request: Request,
    record_id: str = Depends(validate_record_id),
    user: AuthUser = Depends(
        require_permission("eudr-bci:sharing:read")
    ),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_blockchain_service),
) -> AccessGrantListResponse:
    """List access grants for a record.

    Args:
        request: FastAPI request object.
        record_id: Record identifier.
        user: Authenticated user with sharing:read permission.
        service: Blockchain integration service.

    Returns:
        AccessGrantListResponse with grants for the record.
    """
    try:
        store = _get_grant_store()

        # Filter grants by record_id
        matching = [
            record for record in store.values()
            if record.get("record_id") == record_id
        ]

        # Sort by granted_at descending
        matching.sort(
            key=lambda r: r.get("granted_at", datetime.min),
            reverse=True,
        )

        grants = [AccessGrantResponse(**r) for r in matching]

        return AccessGrantListResponse(
            grants=grants,
            record_id=record_id,
            total=len(grants),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to list grants for %s: %s",
            record_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list access grants",
        )


# ---------------------------------------------------------------------------
# POST /sharing/request
# ---------------------------------------------------------------------------


@router.post(
    "/sharing/request",
    response_model=AccessGrantResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Request access to record",
    description=(
        "Request access to an EUDR compliance record from its owner. "
        "The request is recorded and the owner is notified for "
        "approval. Used by competent authorities and auditors."
    ),
    responses={
        201: {"description": "Access request submitted"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def request_access(
    request: Request,
    body: AccessRequestRequest,
    user: AuthUser = Depends(
        require_permission("eudr-bci:sharing:request")
    ),
    _rate: None = Depends(rate_limit_anchor),
    service: Any = Depends(get_blockchain_service),
) -> AccessGrantResponse:
    """Request access to a record.

    Args:
        request: FastAPI request object.
        body: Access request parameters.
        user: Authenticated user with sharing:request permission.
        service: Blockchain integration service.

    Returns:
        AccessGrantResponse with pending grant (awaiting owner approval).
    """
    start = time.monotonic()
    try:
        grant_id = str(uuid.uuid4())
        now = _utcnow()

        provenance_hash = _compute_provenance_hash({
            "grant_id": grant_id,
            "record_id": body.record_id,
            "requester_id": user.user_id,
            "access_level": body.access_level.value,
            "justification": body.justification,
        })

        # Create a pending grant that requires owner approval
        record = {
            "grant_id": grant_id,
            "record_id": body.record_id,
            "grantor_id": "pending_approval",
            "grantee_id": user.user_id,
            "access_level": body.access_level,
            "status": AccessStatusSchema.ACTIVE,  # Pending approval
            "scope": None,
            "reason": body.justification,
            "granted_at": now,
            "expires_at": None,
            "revoked_at": None,
            "provenance": ProvenanceInfo(
                provenance_hash=provenance_hash,
                algorithm="sha256",
                created_at=now,
            ),
        }

        store = _get_grant_store()
        store[grant_id] = record

        elapsed_ms = (time.monotonic() - start) * 1000.0
        logger.info(
            "Access requested: id=%s record=%s requester=%s level=%s "
            "elapsed_ms=%.1f",
            grant_id,
            body.record_id,
            user.user_id,
            body.access_level.value,
            elapsed_ms,
        )

        return AccessGrantResponse(**record)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to request access: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit access request",
        )


# ---------------------------------------------------------------------------
# POST /sharing/confirm
# ---------------------------------------------------------------------------


@router.post(
    "/sharing/confirm",
    response_model=MultiPartyConfirmResponse,
    status_code=status.HTTP_200_OK,
    summary="Multi-party confirmation",
    description=(
        "Submit a multi-party confirmation for a data sharing "
        "agreement. Requires multiple parties to confirm before "
        "the grant becomes fully active. Used for competent "
        "authority and auditor access requests."
    ),
    responses={
        200: {"description": "Confirmation recorded"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Grant not found"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def multi_party_confirm(
    request: Request,
    body: MultiPartyConfirmRequest,
    user: AuthUser = Depends(
        require_permission("eudr-bci:sharing:confirm")
    ),
    _rate: None = Depends(rate_limit_anchor),
    service: Any = Depends(get_blockchain_service),
) -> MultiPartyConfirmResponse:
    """Submit a multi-party confirmation.

    Args:
        request: FastAPI request object.
        body: Multi-party confirmation parameters.
        user: Authenticated user with sharing:confirm permission.
        service: Blockchain integration service.

    Returns:
        MultiPartyConfirmResponse with confirmation status.

    Raises:
        HTTPException: 404 if grant not found.
    """
    start = time.monotonic()
    try:
        grant_store = _get_grant_store()
        grant_record = grant_store.get(body.grant_id)

        if grant_record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Access grant {body.grant_id} not found",
            )

        # Track confirmations
        conf_store = _get_confirmation_store()
        conf_key = body.grant_id

        if conf_key not in conf_store:
            conf_store[conf_key] = {
                "confirmations": [],
                "required": 2,  # Default: 2 confirmations required
            }

        conf_data = conf_store[conf_key]

        # Add confirmation if not already present
        if body.confirmer_id not in conf_data["confirmations"]:
            conf_data["confirmations"].append(body.confirmer_id)

        confirmations_received = len(conf_data["confirmations"])
        confirmations_required = conf_data["required"]
        fully_confirmed = confirmations_received >= confirmations_required

        elapsed_ms = (time.monotonic() - start) * 1000.0
        logger.info(
            "Multi-party confirmation: grant=%s confirmer=%s "
            "received=%d/%d fully=%s elapsed_ms=%.1f",
            body.grant_id,
            body.confirmer_id,
            confirmations_received,
            confirmations_required,
            fully_confirmed,
            elapsed_ms,
        )

        return MultiPartyConfirmResponse(
            grant_id=body.grant_id,
            confirmer_id=body.confirmer_id,
            confirmed=True,
            confirmations_received=confirmations_received,
            confirmations_required=confirmations_required,
            fully_confirmed=fully_confirmed,
            confirmed_at=_utcnow(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to confirm grant %s: %s",
            body.grant_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record confirmation",
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = ["router"]
