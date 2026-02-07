# -*- coding: utf-8 -*-
"""
User Self-Service Routes - JWT Authentication Service (SEC-001)

FastAPI router for user self-service operations:

  POST /auth/password/change  - Change own password (requires current password).
  POST /auth/password/reset   - Request a password-reset token (unauthenticated).
  POST /auth/mfa/setup        - Initiate MFA TOTP enrolment.
  POST /auth/mfa/verify       - Verify MFA TOTP code to complete enrolment.

All endpoints that modify security state emit audit events and, where
applicable, revoke existing tokens to force re-authentication.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, EmailStr, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/auth", tags=["user"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ChangePasswordRequest(BaseModel):
    """Payload for ``POST /auth/password/change``."""

    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8, max_length=128)


class ChangePasswordResponse(BaseModel):
    """Response for ``POST /auth/password/change``."""

    status: str = "password_changed"
    message: str = "Password updated successfully. All sessions have been revoked."


class PasswordResetRequest(BaseModel):
    """Payload for ``POST /auth/password/reset``."""

    email: str = Field(..., min_length=1, max_length=256)
    tenant_id: Optional[str] = None


class PasswordResetResponse(BaseModel):
    """Response for ``POST /auth/password/reset``.

    Always returns 200 regardless of whether the email exists, to
    prevent user-enumeration attacks.
    """

    status: str = "reset_requested"
    message: str = (
        "If an account with that email exists, a password reset "
        "link has been sent."
    )


class MFASetupRequest(BaseModel):
    """Payload for ``POST /auth/mfa/setup``."""

    method: str = Field("totp", pattern="^(totp|sms)$")
    device_name: Optional[str] = Field("Default", max_length=64)
    phone_number: Optional[str] = Field(None, max_length=20)


class MFASetupResponse(BaseModel):
    """Response for ``POST /auth/mfa/setup``."""

    device_id: str
    method: str
    provisioning_uri: Optional[str] = None
    qr_code_base64: Optional[str] = None
    message: str = "Scan the QR code with your authenticator app."


class MFAVerifyRequest(BaseModel):
    """Payload for ``POST /auth/mfa/verify``."""

    device_id: str = Field(..., min_length=1)
    code: str = Field(..., min_length=6, max_length=6)


class MFAVerifyResponse(BaseModel):
    """Response for ``POST /auth/mfa/verify``."""

    verified: bool
    backup_codes: List[str] = []
    message: str = ""


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------


def _get_correlation_id(
    x_correlation_id: Optional[str] = Header(None),
) -> str:
    """Extract or generate a correlation ID."""
    return x_correlation_id or str(uuid.uuid4())


def _get_client_ip(request: Request) -> str:
    """Extract real client IP from request."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


async def _get_authenticated_claims(request: Request) -> Any:
    """Validate the bearer token and return claims.

    Raises 401 if the token is missing or invalid.
    """
    token_service = getattr(request.app.state, "token_service", None)
    if token_service is None:
        raise HTTPException(status_code=503, detail="Auth service unavailable")

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")

    token_str = auth_header[7:]
    claims = await token_service.validate_token(token_str)
    if claims is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return claims


def _emit_audit(
    request: Request,
    event_type: str,
    *,
    user_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    success: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit an audit event if the audit logger is available."""
    audit_logger = getattr(request.app.state, "audit_logger", None)
    if audit_logger is None:
        return
    try:
        from greenlang.auth.audit import AuditEvent, AuditEventType, AuditSeverity

        severity = AuditSeverity.INFO if success else AuditSeverity.WARNING
        audit_logger.log(
            AuditEvent(
                event_type=AuditEventType.RESOURCE_UPDATED,
                severity=severity,
                user_id=user_id,
                tenant_id=tenant_id,
                ip_address=_get_client_ip(request),
                action=event_type,
                result="success" if success else "failure",
                metadata=metadata or {},
            )
        )
    except Exception as exc:
        logger.warning("Audit event emission failed: %s", exc)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/password/change", response_model=ChangePasswordResponse)
async def change_password(
    body: ChangePasswordRequest,
    request: Request,
    correlation_id: str = Depends(_get_correlation_id),
) -> ChangePasswordResponse:
    """Change the authenticated user's password.

    Requires the current password for verification.  On success, all
    existing sessions and tokens for the user are revoked to force
    re-authentication with the new password.
    """
    claims = await _get_authenticated_claims(request)
    auth_manager = getattr(request.app.state, "auth_manager", None)
    revocation_service = getattr(
        request.app.state, "revocation_service", None
    )
    refresh_manager = getattr(
        request.app.state, "refresh_token_manager", None
    )

    if auth_manager is None:
        raise HTTPException(status_code=503, detail="Auth service unavailable")

    # Verify current password
    user_data = _find_user(auth_manager, claims.sub)
    if user_data is None:
        raise HTTPException(status_code=404, detail="User not found")

    if not auth_manager._verify_password(
        body.current_password, user_data["password_hash"]
    ):
        _emit_audit(
            request,
            "password_change_failed",
            user_id=claims.sub,
            tenant_id=claims.tenant_id,
            success=False,
            metadata={"reason": "incorrect_current_password"},
        )
        raise HTTPException(
            status_code=401, detail="Current password is incorrect"
        )

    # Update password
    new_hash = auth_manager._hash_password(body.new_password)
    user_data["password_hash"] = new_hash

    # Revoke all tokens
    if revocation_service is not None:
        await revocation_service.revoke_all_for_user(
            claims.sub, reason="password_change"
        )
    if refresh_manager is not None:
        await refresh_manager.revoke_all_for_user(claims.sub)

    _emit_audit(
        request,
        "password_changed",
        user_id=claims.sub,
        tenant_id=claims.tenant_id,
        success=True,
    )

    logger.info("Password changed for user=%s", claims.sub)
    return ChangePasswordResponse()


@router.post("/password/reset", response_model=PasswordResetResponse)
async def request_password_reset(
    body: PasswordResetRequest,
    request: Request,
    correlation_id: str = Depends(_get_correlation_id),
) -> PasswordResetResponse:
    """Request a password-reset link.

    This endpoint is **unauthenticated**.  To prevent user enumeration,
    it always returns a 200 response regardless of whether the email
    address exists.

    In production the reset token would be sent via email.  This
    implementation logs the event for audit purposes.
    """
    client_ip = _get_client_ip(request)

    _emit_audit(
        request,
        "password_reset_requested",
        tenant_id=body.tenant_id,
        success=True,
        metadata={"email": body.email, "ip": client_ip},
    )

    logger.info(
        "Password reset requested  email=%s  tenant=%s  ip=%s",
        body.email,
        body.tenant_id,
        client_ip,
    )

    # In production: generate a time-limited reset token, store it,
    # and send via email.  Omitted here as it requires an email service.

    return PasswordResetResponse()


@router.post("/mfa/setup", response_model=MFASetupResponse)
async def setup_mfa(
    body: MFASetupRequest,
    request: Request,
    correlation_id: str = Depends(_get_correlation_id),
) -> MFASetupResponse:
    """Initiate MFA enrolment for the authenticated user.

    Currently supports TOTP (Time-based One-Time Password).  Returns a
    provisioning URI and optional base64-encoded QR code image for the
    user to scan with an authenticator app.
    """
    claims = await _get_authenticated_claims(request)

    try:
        from greenlang.auth.mfa import MFAManager, MFAConfig

        mfa_manager: Optional[MFAManager] = getattr(
            request.app.state, "mfa_manager", None
        )
        if mfa_manager is None:
            # Create a default manager for TOTP
            mfa_manager = MFAManager(MFAConfig())
    except ImportError:
        raise HTTPException(
            status_code=503, detail="MFA module not available"
        )

    if body.method == "totp":
        device_id, secret, qr_bytes = mfa_manager.enroll_totp(
            user_id=claims.sub,
            device_name=body.device_name or "Default",
        )

        # Build provisioning URI
        try:
            import pyotp

            totp = pyotp.TOTP(secret)
            provisioning_uri = totp.provisioning_uri(
                name=claims.email or claims.sub,
                issuer_name="GreenLang",
            )
        except ImportError:
            provisioning_uri = None

        # Base64-encode the QR code image
        import base64

        qr_b64 = base64.b64encode(qr_bytes).decode("ascii") if qr_bytes else None

        _emit_audit(
            request,
            "mfa_setup_initiated",
            user_id=claims.sub,
            tenant_id=claims.tenant_id,
            success=True,
            metadata={"method": "totp", "device_id": device_id},
        )

        return MFASetupResponse(
            device_id=device_id,
            method="totp",
            provisioning_uri=provisioning_uri,
            qr_code_base64=qr_b64,
            message="Scan the QR code with your authenticator app, then verify with a code.",
        )

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported MFA method: {body.method}",
    )


@router.post("/mfa/verify", response_model=MFAVerifyResponse)
async def verify_mfa(
    body: MFAVerifyRequest,
    request: Request,
    correlation_id: str = Depends(_get_correlation_id),
) -> MFAVerifyResponse:
    """Verify an MFA code to complete enrolment or step-up authentication.

    On successful first-time verification, backup codes are generated
    and returned.
    """
    claims = await _get_authenticated_claims(request)

    try:
        from greenlang.auth.mfa import MFAManager, MFAConfig, MFAMethod

        mfa_manager: Optional[MFAManager] = getattr(
            request.app.state, "mfa_manager", None
        )
        if mfa_manager is None:
            mfa_manager = MFAManager(MFAConfig())
    except ImportError:
        raise HTTPException(
            status_code=503, detail="MFA module not available"
        )

    # Try verifying as enrolment confirmation first
    try:
        verified = mfa_manager.verify_totp_enrollment(
            user_id=claims.sub,
            device_id=body.device_id,
            code=body.code,
        )
    except Exception as exc:
        # Fall back to standard verification
        try:
            verified = mfa_manager.verify_mfa(
                user_id=claims.sub,
                method=MFAMethod.TOTP,
                code=body.code,
                device_id=body.device_id,
            )
        except Exception as verify_exc:
            logger.warning("MFA verification failed: %s", verify_exc)
            verified = False

    backup_codes: List[str] = []
    message = "MFA verification failed."

    if verified:
        message = "MFA verification successful."
        # Generate backup codes on first successful enrolment
        enrollment = mfa_manager.get_enrollment(claims.sub)
        if enrollment is not None and not enrollment.backup_codes:
            backup_codes = mfa_manager.generate_backup_codes(claims.sub)
            message = (
                "MFA enabled successfully. Save your backup codes securely."
            )

        _emit_audit(
            request,
            "mfa_verified",
            user_id=claims.sub,
            tenant_id=claims.tenant_id,
            success=True,
            metadata={"device_id": body.device_id},
        )
    else:
        _emit_audit(
            request,
            "mfa_verify_failed",
            user_id=claims.sub,
            tenant_id=claims.tenant_id,
            success=False,
            metadata={"device_id": body.device_id},
        )

    return MFAVerifyResponse(
        verified=verified,
        backup_codes=backup_codes,
        message=message,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_user(auth_manager: Any, user_id: str) -> Optional[Dict[str, Any]]:
    """Look up a user by ID in the AuthManager's in-memory store.

    In production this would query the ``security.users`` table.
    """
    users = getattr(auth_manager, "users", {})
    return users.get(user_id)
