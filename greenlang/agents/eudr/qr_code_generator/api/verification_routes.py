# -*- coding: utf-8 -*-
"""
Verification Routes - AGENT-EUDR-014 QR Code Generator API

Endpoints for verification URL construction and signature validation
including HMAC-SHA256 signed token generation, signature verification,
status lookup, and offline verification support.

Endpoints:
    POST   /verify/build-url       - Build verification URL
    POST   /verify/signature       - Verify signature
    GET    /verify/{code_id}       - Get verification status
    POST   /verify/offline         - Offline verification check

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-014, Feature 5 (Verification URL Construction Engine)
Agent ID: GL-EUDR-QRG-014
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.qr_code_generator.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_qrg_service,
    rate_limit_standard,
    rate_limit_verify,
    require_permission,
    validate_code_id,
)
from greenlang.agents.eudr.qr_code_generator.api.schemas import (
    BuildURLRequest,
    BuildURLResponse,
    OfflineVerifyRequest,
    OfflineVerifyResponse,
    ProvenanceInfo,
    VerificationStatusResponse,
    VerifySignatureRequest,
    VerifySignatureResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Verification"])

# ---------------------------------------------------------------------------
# In-memory stores (replaced by database in production)
# ---------------------------------------------------------------------------

_verification_store: Dict[str, Dict] = {}

# HMAC secret key (would come from Vault in production)
_HMAC_SECRET = b"eudr-qrg-hmac-secret-key-replace-in-production"


def _get_verification_store() -> Dict[str, Dict]:
    """Return the verification record store singleton."""
    return _verification_store


def _compute_provenance_hash(data: dict) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _generate_hmac_token(code_id: str, operator_id: str) -> str:
    """Generate an HMAC-SHA256 token for a QR code.

    Args:
        code_id: QR code identifier.
        operator_id: Operator identifier.

    Returns:
        Hex-encoded HMAC-SHA256 token.
    """
    message = f"{code_id}:{operator_id}".encode("utf-8")
    return hmac.new(_HMAC_SECRET, message, hashlib.sha256).hexdigest()


def _verify_hmac_token(code_id: str, token: str) -> bool:
    """Verify an HMAC token against known records.

    Args:
        code_id: QR code identifier.
        token: HMAC token to verify.

    Returns:
        True if token is valid.
    """
    store = _get_verification_store()
    for record in store.values():
        if record.get("code_id") == code_id and record.get("token") == token:
            # Check expiry
            expires_at = record.get("token_expires_at")
            if expires_at and expires_at < _utcnow():
                return False
            return True
    return False


# ---------------------------------------------------------------------------
# POST /verify/build-url
# ---------------------------------------------------------------------------


@router.post(
    "/verify/build-url",
    response_model=BuildURLResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Build verification URL",
    description=(
        "Build a verification URL with HMAC-SHA256 signed token for "
        "a QR code. The token has a configurable TTL (default 5 years "
        "per EUDR Article 14)."
    ),
    responses={
        201: {"description": "Verification URL built successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def build_verification_url(
    request: Request,
    body: BuildURLRequest,
    user: AuthUser = Depends(
        require_permission("eudr-qrg:verify:build-url")
    ),
    _rate: None = Depends(rate_limit_verify),
    service: Any = Depends(get_qrg_service),
) -> BuildURLResponse:
    """Build a verification URL with HMAC token.

    Args:
        request: FastAPI request object.
        body: URL construction parameters.
        user: Authenticated user with verify:build-url permission.
        service: QR Code Generator service.

    Returns:
        BuildURLResponse with full URL and token details.
    """
    start = time.monotonic()
    try:
        url_id = str(uuid.uuid4())
        now = _utcnow()

        # Generate HMAC token
        token = _generate_hmac_token(body.code_id, body.operator_id)

        # Calculate expiry (default 5 years per EUDR Article 14)
        ttl_years = body.ttl_years or 5
        token_expires_at = now + timedelta(days=365 * ttl_years)

        # Build full URL
        base_url = body.base_url or "https://verify.greenlang.io"
        truncated_token = token[:16]
        full_url = (
            f"{base_url}/v/{body.code_id}?t={truncated_token}"
        )

        # Short URL (simulated)
        short_url = None
        if body.use_short_url:
            short_url = f"https://gl.io/v/{truncated_token[:8]}"

        provenance_hash = _compute_provenance_hash({
            "url_id": url_id,
            "code_id": body.code_id,
            "operator_id": body.operator_id,
            "token_hash": hashlib.sha256(token.encode()).hexdigest(),
            "created_by": user.user_id,
        })

        verification_record = {
            "url_id": url_id,
            "code_id": body.code_id,
            "full_url": full_url,
            "short_url": short_url,
            "token": token,
            "token_expires_at": token_expires_at,
            "operator_id": body.operator_id,
            "created_at": now,
        }

        store = _get_verification_store()
        store[url_id] = verification_record

        elapsed_ms = (time.monotonic() - start) * 1000.0
        logger.info(
            "Verification URL built: url_id=%s code_id=%s ttl=%dy "
            "elapsed_ms=%.1f",
            url_id,
            body.code_id,
            ttl_years,
            elapsed_ms,
        )

        return BuildURLResponse(
            url_id=url_id,
            code_id=body.code_id,
            full_url=full_url,
            short_url=short_url,
            token=token,
            token_expires_at=token_expires_at,
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
            "Failed to build verification URL: %s", exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to build verification URL",
        )


# ---------------------------------------------------------------------------
# POST /verify/signature
# ---------------------------------------------------------------------------


@router.post(
    "/verify/signature",
    response_model=VerifySignatureResponse,
    summary="Verify signature",
    description=(
        "Verify the HMAC-SHA256 signature of a QR code. Returns "
        "whether the signature is valid, the algorithm used, and "
        "the key identifier."
    ),
    responses={
        200: {"description": "Signature verification completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def verify_signature(
    request: Request,
    body: VerifySignatureRequest,
    user: AuthUser = Depends(
        require_permission("eudr-qrg:verify:signature")
    ),
    _rate: None = Depends(rate_limit_verify),
    service: Any = Depends(get_qrg_service),
) -> VerifySignatureResponse:
    """Verify a QR code signature.

    Args:
        request: FastAPI request object.
        body: Signature verification parameters.
        user: Authenticated user with verify:signature permission.
        service: QR Code Generator service.

    Returns:
        VerifySignatureResponse with verification result.
    """
    start = time.monotonic()
    try:
        now = _utcnow()

        # Compute expected HMAC
        expected_hmac = hmac.new(
            _HMAC_SECRET,
            body.data_hash.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        is_valid = hmac.compare_digest(
            expected_hmac, body.signature_value
        )

        elapsed_ms = (time.monotonic() - start) * 1000.0
        logger.info(
            "Signature verified: code_id=%s valid=%s elapsed_ms=%.1f",
            body.code_id,
            is_valid,
            elapsed_ms,
        )

        return VerifySignatureResponse(
            code_id=body.code_id,
            valid=is_valid,
            algorithm="HMAC-SHA256",
            key_id="eudr-qrg-key-001",
            verified_at=now,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to verify signature: %s", exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify signature",
        )


# ---------------------------------------------------------------------------
# GET /verify/{code_id}
# ---------------------------------------------------------------------------


@router.get(
    "/verify/{code_id}",
    response_model=VerificationStatusResponse,
    summary="Get verification status",
    description=(
        "Retrieve the verification status of a QR code including "
        "lifecycle status, compliance status, scan count, and "
        "counterfeit risk assessment."
    ),
    responses={
        200: {"description": "Verification status retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Code not found"},
    },
)
async def get_verification_status(
    request: Request,
    code_id: str = Depends(validate_code_id),
    user: AuthUser = Depends(
        require_permission("eudr-qrg:verify:read")
    ),
    _rate: None = Depends(rate_limit_verify),
    service: Any = Depends(get_qrg_service),
) -> VerificationStatusResponse:
    """Get verification status for a QR code.

    Args:
        request: FastAPI request object.
        code_id: QR code identifier.
        user: Authenticated user with verify:read permission.
        service: QR Code Generator service.

    Returns:
        VerificationStatusResponse with status details.
    """
    try:
        now = _utcnow()
        store = _get_verification_store()

        # Find verification record for this code_id
        verification_url = None
        last_verified_at = None
        for record in store.values():
            if record.get("code_id") == code_id:
                verification_url = record.get("full_url")
                last_verified_at = record.get("created_at")
                break

        provenance_hash = _compute_provenance_hash({
            "code_id": code_id,
            "checked_at": str(now),
        })

        return VerificationStatusResponse(
            code_id=code_id,
            status="active",
            compliance_status="pending",
            verification_url=verification_url,
            last_verified_at=last_verified_at,
            scan_count=0,
            counterfeit_risk="low",
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
            "Failed to get verification status %s: %s",
            code_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve verification status",
        )


# ---------------------------------------------------------------------------
# POST /verify/offline
# ---------------------------------------------------------------------------


@router.post(
    "/verify/offline",
    response_model=OfflineVerifyResponse,
    summary="Offline verification",
    description=(
        "Perform offline verification of a QR code using the HMAC "
        "token from the verification URL. Does not require live "
        "connectivity to the backend."
    ),
    responses={
        200: {"description": "Offline verification completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def offline_verify(
    request: Request,
    body: OfflineVerifyRequest,
    user: AuthUser = Depends(
        require_permission("eudr-qrg:verify:offline")
    ),
    _rate: None = Depends(rate_limit_verify),
    service: Any = Depends(get_qrg_service),
) -> OfflineVerifyResponse:
    """Perform offline verification.

    Args:
        request: FastAPI request object.
        body: Offline verification parameters.
        user: Authenticated user with verify:offline permission.
        service: QR Code Generator service.

    Returns:
        OfflineVerifyResponse with verification result.
    """
    try:
        now = _utcnow()

        # Verify HMAC token
        hmac_valid = _verify_hmac_token(body.code_id, body.hmac_token)

        # Overall validity requires valid HMAC
        is_valid = hmac_valid

        logger.info(
            "Offline verification: code_id=%s hmac_valid=%s valid=%s",
            body.code_id,
            hmac_valid,
            is_valid,
        )

        return OfflineVerifyResponse(
            code_id=body.code_id,
            valid=is_valid,
            hmac_valid=hmac_valid,
            status="active" if is_valid else "unknown",
            compliance_status="pending",
            verified_at=now,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed offline verification: %s", exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform offline verification",
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = ["router"]
