# -*- coding: utf-8 -*-
"""
Supplier Onboarding Routes - AGENT-EUDR-001 Supply Chain Mapper API

Endpoints for managing supplier onboarding invitations, enabling
EUDR-regulated operators to invite sub-tier suppliers to self-register
their supply chain data.

Endpoints:
    POST /onboarding/invite         - Create an onboarding invitation
    GET  /onboarding/{token}        - Get invitation status (public)
    POST /onboarding/{token}/submit - Submit onboarding data (public)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-001, Section 7.5 (Feature 8)
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.supply_chain_mapper.api.dependencies import (
    AuthUser,
    ErrorResponse,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.supply_chain_mapper.api.schemas import (
    OnboardingInviteRequest,
    OnboardingInviteResponse,
    OnboardingStatusResponse,
    OnboardingSubmitRequest,
    OnboardingSubmitResponse,
)
from greenlang.agents.eudr.supply_chain_mapper.models import (
    EUDRCommodity,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Supplier Onboarding"])

# ---------------------------------------------------------------------------
# In-memory invitation store (replaced by database in production)
# ---------------------------------------------------------------------------

_invitation_store: Dict[str, dict] = {}


def _get_invitation_store() -> Dict[str, dict]:
    """Return the invitation store singleton."""
    return _invitation_store


def _generate_token() -> str:
    """Generate a secure onboarding token."""
    raw = str(uuid.uuid4()) + str(uuid.uuid4())
    return hashlib.sha256(raw.encode()).hexdigest()[:48]


# ---------------------------------------------------------------------------
# POST /onboarding/invite
# ---------------------------------------------------------------------------


@router.post(
    "/onboarding/invite",
    response_model=OnboardingInviteResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a supplier onboarding invitation",
    description=(
        "Invite a supplier to complete their EUDR supply chain profile. "
        "Generates a secure, time-limited token that the supplier can use "
        "to access the onboarding portal without authentication."
    ),
    responses={
        201: {"description": "Invitation created"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def create_invitation(
    body: OnboardingInviteRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-supply-chain:onboarding:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> OnboardingInviteResponse:
    """Create a supplier onboarding invitation.

    Args:
        body: Invitation parameters (supplier details, commodity, expiry).
        user: Authenticated user with onboarding:write permission.

    Returns:
        OnboardingInviteResponse with token and onboarding URL.
    """
    invitation_id = str(uuid.uuid4())
    token = _generate_token()
    now = datetime.now(timezone.utc).replace(microsecond=0)
    expires = now + timedelta(days=body.expires_in_days)

    invitation = {
        "invitation_id": invitation_id,
        "token": token,
        "supplier_name": body.supplier_name,
        "supplier_email": body.supplier_email,
        "supplier_country": body.supplier_country,
        "commodity": body.commodity.value,
        "graph_id": body.graph_id,
        "message": body.message,
        "created_by": user.user_id,
        "operator_id": user.operator_id or user.user_id,
        "status": "pending",
        "created_at": now.isoformat(),
        "expires_at": expires.isoformat(),
        "submitted_at": None,
        "submission_data": None,
    }

    store = _get_invitation_store()
    store[token] = invitation

    # Build onboarding URL
    base_url = str(request.base_url).rstrip("/")
    onboarding_url = f"{base_url}/api/v1/eudr-scm/onboarding/{token}"

    logger.info(
        "Onboarding invitation created: id=%s supplier=%s commodity=%s",
        invitation_id,
        body.supplier_name,
        body.commodity.value,
    )

    return OnboardingInviteResponse(
        invitation_id=invitation_id,
        token=token,
        supplier_name=body.supplier_name,
        supplier_email=body.supplier_email,
        status="pending",
        expires_at=expires,
        onboarding_url=onboarding_url,
    )


# ---------------------------------------------------------------------------
# GET /onboarding/{token}
# ---------------------------------------------------------------------------


@router.get(
    "/onboarding/{token}",
    response_model=OnboardingStatusResponse,
    summary="Get onboarding invitation status",
    description=(
        "Retrieve the status and details of an onboarding invitation "
        "using its secure token. This endpoint is publicly accessible "
        "to allow suppliers to check their invitation status."
    ),
    responses={
        200: {"description": "Invitation details"},
        404: {"model": ErrorResponse, "description": "Invitation not found"},
        410: {"model": ErrorResponse, "description": "Invitation expired"},
    },
)
async def get_invitation_status(
    token: str,
    request: Request,
) -> OnboardingStatusResponse:
    """Get onboarding invitation status by token.

    This endpoint does NOT require authentication, as it is used by
    external suppliers who may not have GreenLang accounts.

    Args:
        token: Secure invitation token.

    Returns:
        OnboardingStatusResponse with invitation details.

    Raises:
        HTTPException: 404 if not found, 410 if expired.
    """
    store = _get_invitation_store()
    invitation = store.get(token)

    if invitation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invitation not found or invalid token",
        )

    # Check expiry
    expires_at = datetime.fromisoformat(invitation["expires_at"])
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)

    if now > expires_at:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Invitation has expired",
        )

    submitted_at = None
    if invitation.get("submitted_at"):
        submitted_at = datetime.fromisoformat(invitation["submitted_at"])

    return OnboardingStatusResponse(
        invitation_id=invitation["invitation_id"],
        supplier_name=invitation["supplier_name"],
        supplier_email=invitation["supplier_email"],
        status=invitation["status"],
        commodity=EUDRCommodity(invitation["commodity"]),
        supplier_country=invitation["supplier_country"],
        graph_id=invitation.get("graph_id"),
        expires_at=expires_at,
        submitted_at=submitted_at,
    )


# ---------------------------------------------------------------------------
# POST /onboarding/{token}/submit
# ---------------------------------------------------------------------------


@router.post(
    "/onboarding/{token}/submit",
    response_model=OnboardingSubmitResponse,
    summary="Submit supplier onboarding data",
    description=(
        "Submit supply chain data for a pending onboarding invitation. "
        "The supplier provides their operator details, coordinates, "
        "commodities, certifications, and optional sub-supplier information. "
        "This endpoint is publicly accessible via the secure token."
    ),
    responses={
        200: {"description": "Submission accepted"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "Invitation not found"},
        409: {"model": ErrorResponse, "description": "Already submitted"},
        410: {"model": ErrorResponse, "description": "Invitation expired"},
    },
)
async def submit_onboarding(
    token: str,
    body: OnboardingSubmitRequest,
    request: Request,
) -> OnboardingSubmitResponse:
    """Submit supplier onboarding data.

    This endpoint does NOT require authentication. Suppliers access
    it via the secure token generated during invitation.

    Args:
        token: Secure invitation token.
        body: Supplier onboarding data.

    Returns:
        OnboardingSubmitResponse confirming submission.

    Raises:
        HTTPException: 404 if not found, 409 if already submitted, 410 if expired.
    """
    store = _get_invitation_store()
    invitation = store.get(token)

    if invitation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invitation not found or invalid token",
        )

    # Check expiry
    expires_at = datetime.fromisoformat(invitation["expires_at"])
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)

    if now > expires_at:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Invitation has expired",
        )

    # Check if already submitted
    if invitation["status"] == "submitted":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Onboarding data has already been submitted for this invitation",
        )

    # Record submission
    invitation["status"] = "submitted"
    invitation["submitted_at"] = now.isoformat()
    invitation["submission_data"] = body.model_dump(mode="json")

    # In production, this would also add the node to the graph
    node_id = None
    if invitation.get("graph_id"):
        node_id = str(uuid.uuid4())

    logger.info(
        "Onboarding submission received: invitation=%s supplier=%s",
        invitation["invitation_id"],
        body.operator_name,
    )

    return OnboardingSubmitResponse(
        invitation_id=invitation["invitation_id"],
        node_id=node_id,
        status="submitted",
        submitted_at=now,
    )
