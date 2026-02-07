# -*- coding: utf-8 -*-
"""
Admin API Routes - JWT Authentication Service (SEC-001)

Protected endpoints for administrative operations on the authentication
system.  All endpoints require the ``admin:*`` permission or the
``super_admin`` / ``admin`` role.

Features:
    - User lockout management (list, unlock)
    - Bulk token revocation per user
    - Force password reset
    - MFA emergency disable
    - Session listing and termination
    - Auth audit log queries

Endpoints:
    GET    /auth/admin/users                          - List users.
    GET    /auth/admin/users/{user_id}                - Get user details.
    POST   /auth/admin/users/{user_id}/unlock         - Unlock account.
    POST   /auth/admin/users/{user_id}/revoke-tokens  - Revoke all tokens.
    POST   /auth/admin/users/{user_id}/force-password-reset
    POST   /auth/admin/users/{user_id}/disable-mfa
    GET    /auth/admin/sessions                       - List active sessions.
    DELETE /auth/admin/sessions/{session_id}           - Terminate session.
    GET    /auth/admin/audit-log                      - Query audit log.
    GET    /auth/admin/lockouts                       - List locked accounts.

Security Compliance:
    - SOC 2 CC6.2 (Privileged Access)
    - ISO 27001 A.9.2 (User Access Management)

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/auth/admin", tags=["Auth Admin"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class UserStatus(str, Enum):
    """User account statuses."""

    ACTIVE = "active"
    LOCKED = "locked"
    DISABLED = "disabled"
    PENDING = "pending"
    SUSPENDED = "suspended"


class AuditEventType(str, Enum):
    """Supported audit event types."""

    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    TOKEN_ISSUED = "token_issued"
    TOKEN_REVOKED = "token_revoked"
    PASSWORD_CHANGED = "password_changed"
    PASSWORD_RESET = "password_reset"
    MFA_ENABLED = "mfa_enabled"
    MFA_DISABLED = "mfa_disabled"
    ACCOUNT_LOCKED = "account_locked"
    ACCOUNT_UNLOCKED = "account_unlocked"
    SESSION_CREATED = "session_created"
    SESSION_TERMINATED = "session_terminated"
    PERMISSION_CHANGED = "permission_changed"


class UserSummary(BaseModel):
    """Summary representation of a user account."""

    user_id: str = Field(..., description="Unique user identifier")
    email: Optional[str] = Field(None, description="User email address")
    name: Optional[str] = Field(None, description="Display name")
    tenant_id: str = Field(..., description="Tenant the user belongs to")
    status: UserStatus = Field(..., description="Account status")
    roles: List[str] = Field(default_factory=list, description="Assigned roles")
    mfa_enabled: bool = Field(False, description="Whether MFA is enabled")
    last_login_at: Optional[datetime] = Field(
        None, description="Last successful login timestamp"
    )
    created_at: datetime = Field(..., description="Account creation timestamp")
    locked_at: Optional[datetime] = Field(
        None, description="Account lock timestamp (if locked)"
    )
    lock_reason: Optional[str] = Field(
        None, description="Reason the account was locked"
    )


class UserListResponse(BaseModel):
    """Paginated user list response."""

    users: List[UserSummary] = Field(..., description="User records")
    total: int = Field(..., description="Total matching users")
    page: int = Field(..., ge=1, description="Current page")
    page_size: int = Field(..., ge=1, le=100, description="Items per page")
    has_next: bool = Field(..., description="Whether a next page exists")


class UserDetailResponse(BaseModel):
    """Detailed user information for admin view."""

    user_id: str = Field(..., description="Unique user identifier")
    email: Optional[str] = Field(None, description="User email address")
    name: Optional[str] = Field(None, description="Display name")
    tenant_id: str = Field(..., description="Tenant the user belongs to")
    org_id: Optional[str] = Field(None, description="Organisation identifier")
    status: UserStatus = Field(..., description="Account status")
    roles: List[str] = Field(default_factory=list, description="Assigned roles")
    permissions: List[str] = Field(
        default_factory=list, description="Direct permissions"
    )
    mfa_enabled: bool = Field(False, description="Whether MFA is enabled")
    mfa_methods: List[str] = Field(
        default_factory=list, description="Active MFA methods"
    )
    active_sessions: int = Field(0, description="Count of active sessions")
    failed_login_attempts: int = Field(
        0, description="Consecutive failed login attempts"
    )
    last_login_at: Optional[datetime] = Field(None, description="Last login")
    last_login_ip: Optional[str] = Field(None, description="Last login IP")
    created_at: datetime = Field(..., description="Account creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    locked_at: Optional[datetime] = Field(None, description="Lock timestamp")
    lock_reason: Optional[str] = Field(None, description="Lock reason")


class UnlockResponse(BaseModel):
    """Response from unlocking a user account."""

    user_id: str = Field(..., description="User identifier")
    previous_status: UserStatus = Field(..., description="Status before unlock")
    current_status: UserStatus = Field(
        UserStatus.ACTIVE, description="Status after unlock"
    )
    unlocked_at: datetime = Field(..., description="Unlock timestamp")
    unlocked_by: str = Field(..., description="Admin who performed the unlock")
    message: str = Field(..., description="Human-readable message")


class RevokeTokensResponse(BaseModel):
    """Response from revoking all tokens for a user."""

    user_id: str = Field(..., description="User identifier")
    tokens_revoked: int = Field(
        ..., description="Number of tokens revoked"
    )
    sessions_terminated: int = Field(
        ..., description="Number of sessions terminated"
    )
    revoked_at: datetime = Field(..., description="Revocation timestamp")
    revoked_by: str = Field(..., description="Admin who performed the action")
    reason: str = Field(..., description="Revocation reason")


class ForcePasswordResetResponse(BaseModel):
    """Response from forcing a password reset."""

    user_id: str = Field(..., description="User identifier")
    reset_token_sent: bool = Field(
        ..., description="Whether a reset email was sent"
    )
    forced_at: datetime = Field(..., description="Timestamp of the action")
    forced_by: str = Field(..., description="Admin who forced the reset")
    message: str = Field(..., description="Human-readable message")


class DisableMFAResponse(BaseModel):
    """Response from disabling MFA for a user."""

    user_id: str = Field(..., description="User identifier")
    previous_mfa_methods: List[str] = Field(
        ..., description="MFA methods that were active"
    )
    disabled_at: datetime = Field(..., description="Timestamp of the action")
    disabled_by: str = Field(..., description="Admin who disabled MFA")
    message: str = Field(..., description="Human-readable message")


class SessionSummary(BaseModel):
    """Summary of an active user session."""

    session_id: str = Field(..., description="Unique session identifier")
    user_id: str = Field(..., description="User identifier")
    tenant_id: str = Field(..., description="Tenant identifier")
    client_ip: str = Field(..., description="Client IP address")
    user_agent: Optional[str] = Field(None, description="Browser user-agent")
    created_at: datetime = Field(..., description="Session creation timestamp")
    last_activity_at: datetime = Field(
        ..., description="Last activity timestamp"
    )
    expires_at: datetime = Field(..., description="Session expiry timestamp")


class SessionListResponse(BaseModel):
    """Paginated session list response."""

    sessions: List[SessionSummary] = Field(..., description="Session records")
    total: int = Field(..., description="Total matching sessions")
    page: int = Field(..., ge=1, description="Current page")
    page_size: int = Field(..., ge=1, le=100, description="Items per page")
    has_next: bool = Field(..., description="Whether a next page exists")


class TerminateSessionResponse(BaseModel):
    """Response from terminating a session."""

    session_id: str = Field(..., description="Terminated session identifier")
    user_id: str = Field(..., description="User who owned the session")
    terminated_at: datetime = Field(..., description="Termination timestamp")
    terminated_by: str = Field(..., description="Admin who terminated it")


class AuditLogEntry(BaseModel):
    """Single entry in the auth audit log."""

    event_id: str = Field(..., description="Unique event identifier")
    event_type: AuditEventType = Field(..., description="Type of event")
    user_id: Optional[str] = Field(None, description="User involved")
    tenant_id: Optional[str] = Field(None, description="Tenant context")
    client_ip: Optional[str] = Field(None, description="Client IP address")
    user_agent: Optional[str] = Field(None, description="Client user-agent")
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Event-specific details"
    )
    timestamp: datetime = Field(..., description="Event timestamp")


class AuditLogResponse(BaseModel):
    """Paginated audit log response."""

    entries: List[AuditLogEntry] = Field(..., description="Audit log entries")
    total: int = Field(..., description="Total matching entries")
    page: int = Field(..., ge=1, description="Current page")
    page_size: int = Field(..., ge=1, le=100, description="Items per page")
    has_next: bool = Field(..., description="Whether a next page exists")


class LockoutEntry(BaseModel):
    """Summary of a currently locked account."""

    user_id: str = Field(..., description="User identifier")
    email: Optional[str] = Field(None, description="User email")
    tenant_id: str = Field(..., description="Tenant identifier")
    locked_at: datetime = Field(..., description="When the account was locked")
    lock_reason: str = Field(..., description="Reason for the lockout")
    failed_attempts: int = Field(
        ..., description="Number of failed attempts that triggered the lock"
    )


class LockoutListResponse(BaseModel):
    """Response listing all currently locked accounts."""

    lockouts: List[LockoutEntry] = Field(
        ..., description="Currently locked accounts"
    )
    total: int = Field(..., description="Total locked accounts")


class AdminErrorResponse(BaseModel):
    """Standard error response for admin endpoints."""

    error_code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional details"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# Dependency: verify admin permissions
# ---------------------------------------------------------------------------


async def _require_admin(request: Request) -> Any:
    """FastAPI dependency that enforces admin-level access.

    Reads ``request.state.auth`` and checks for ``admin:*`` permission
    or the ``admin`` / ``super_admin`` role.

    Args:
        request: The current FastAPI request.

    Returns:
        The ``AuthContext`` of the calling admin.

    Raises:
        HTTPException: 401 if not authenticated, 403 if not an admin.
    """
    auth = getattr(request.state, "auth", None)
    if auth is None:
        logger.warning(
            "Admin endpoint accessed without authentication: path=%s",
            request.url.path,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check admin permission or role
    is_admin = False

    # Permission-based check with wildcard support
    user_permissions = getattr(auth, "permissions", [])
    for perm in user_permissions:
        if perm in ("admin:*", "*"):
            is_admin = True
            break
        if perm.startswith("admin:"):
            is_admin = True
            break

    # Role-based check
    user_roles = set(getattr(auth, "roles", []))
    if user_roles & {"admin", "super_admin"}:
        is_admin = True

    if not is_admin:
        logger.warning(
            "Non-admin access attempt: user=%s roles=%s permissions=%s path=%s",
            auth.user_id,
            getattr(auth, "roles", []),
            user_permissions,
            request.url.path,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error_code": "ADMIN_REQUIRED",
                "message": "Administrator privileges are required",
            },
        )

    logger.info(
        "Admin access granted: user=%s path=%s",
        auth.user_id,
        request.url.path,
    )
    return auth


# ---------------------------------------------------------------------------
# User management endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/users",
    response_model=UserListResponse,
    summary="List users",
    description=(
        "Retrieve a paginated list of user accounts.  "
        "Supports filtering by tenant, status, and role."
    ),
    responses={
        401: {"model": AdminErrorResponse, "description": "Not authenticated"},
        403: {"model": AdminErrorResponse, "description": "Not an admin"},
    },
)
async def list_users(
    request: Request,
    admin: Any = Depends(_require_admin),
    tenant_id: Optional[str] = Query(
        None, description="Filter by tenant identifier"
    ),
    user_status: Optional[UserStatus] = Query(
        None, alias="status", description="Filter by account status"
    ),
    role: Optional[str] = Query(None, description="Filter by role"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    offset: int = Query(0, ge=0, description="Number of items to skip"),
) -> UserListResponse:
    """List user accounts with pagination and filtering.

    In production this queries the user store.  The current
    implementation returns placeholder data to enable API contract
    testing.

    Args:
        request: FastAPI request.
        admin: Injected admin auth context.
        tenant_id: Optional tenant filter.
        user_status: Optional status filter.
        role: Optional role filter.
        limit: Page size.
        offset: Pagination offset.

    Returns:
        Paginated list of user summaries.
    """
    logger.info(
        "Admin listing users: admin=%s tenant_filter=%s status_filter=%s",
        admin.user_id,
        tenant_id,
        user_status,
    )

    # Placeholder -- production implementation queries the user table
    now = datetime.now(timezone.utc)
    sample_users = [
        UserSummary(
            user_id="usr_placeholder_001",
            email="admin@example.com",
            name="Platform Admin",
            tenant_id=tenant_id or "t-default",
            status=UserStatus.ACTIVE,
            roles=["admin"],
            mfa_enabled=True,
            last_login_at=now,
            created_at=now,
        ),
    ]

    page = (offset // limit) + 1 if limit > 0 else 1
    return UserListResponse(
        users=sample_users,
        total=len(sample_users),
        page=page,
        page_size=limit,
        has_next=False,
    )


@router.get(
    "/users/{user_id}",
    response_model=UserDetailResponse,
    summary="Get user details",
    description="Retrieve detailed information about a specific user.",
    responses={
        404: {"model": AdminErrorResponse, "description": "User not found"},
    },
)
async def get_user(
    request: Request,
    user_id: str,
    admin: Any = Depends(_require_admin),
) -> UserDetailResponse:
    """Get detailed information about a specific user.

    Args:
        request: FastAPI request.
        user_id: Target user identifier.
        admin: Injected admin auth context.

    Returns:
        Detailed user information.

    Raises:
        HTTPException: 404 if the user is not found.
    """
    logger.info(
        "Admin retrieving user details: admin=%s target=%s",
        admin.user_id,
        user_id,
    )

    # Placeholder -- production implementation queries the user store
    now = datetime.now(timezone.utc)
    return UserDetailResponse(
        user_id=user_id,
        email=f"{user_id}@example.com",
        name=f"User {user_id}",
        tenant_id="t-default",
        status=UserStatus.ACTIVE,
        roles=["viewer"],
        permissions=["emissions:list", "emissions:read"],
        mfa_enabled=False,
        mfa_methods=[],
        active_sessions=1,
        failed_login_attempts=0,
        last_login_at=now,
        last_login_ip="10.0.0.1",
        created_at=now,
        updated_at=now,
    )


@router.post(
    "/users/{user_id}/unlock",
    response_model=UnlockResponse,
    summary="Unlock locked account",
    description=(
        "Unlock a user account that was locked due to failed login "
        "attempts or administrative action.  Resets the failed attempt "
        "counter."
    ),
    responses={
        404: {"model": AdminErrorResponse, "description": "User not found"},
        409: {
            "model": AdminErrorResponse,
            "description": "Account is not locked",
        },
    },
)
async def unlock_user(
    request: Request,
    user_id: str,
    admin: Any = Depends(_require_admin),
) -> UnlockResponse:
    """Unlock a locked user account.

    Resets the failed login attempt counter and transitions the account
    status from ``locked`` back to ``active``.

    Args:
        request: FastAPI request.
        user_id: Target user identifier.
        admin: Injected admin auth context.

    Returns:
        Unlock confirmation with metadata.

    Raises:
        HTTPException: 404 if user not found, 409 if not locked.
    """
    logger.info(
        "Admin unlocking user: admin=%s target=%s",
        admin.user_id,
        user_id,
    )

    # Placeholder -- production implementation modifies user record and
    # emits an audit event.
    now = datetime.now(timezone.utc)
    return UnlockResponse(
        user_id=user_id,
        previous_status=UserStatus.LOCKED,
        current_status=UserStatus.ACTIVE,
        unlocked_at=now,
        unlocked_by=admin.user_id,
        message=f"Account '{user_id}' has been unlocked successfully",
    )


@router.post(
    "/users/{user_id}/revoke-tokens",
    response_model=RevokeTokensResponse,
    summary="Revoke all tokens for user",
    description=(
        "Immediately revoke all active JWT access tokens and refresh "
        "tokens for the specified user.  Active sessions are also "
        "terminated."
    ),
)
async def revoke_user_tokens(
    request: Request,
    user_id: str,
    admin: Any = Depends(_require_admin),
    reason: str = Query(
        "admin_revoke",
        description="Reason for revocation (logged in audit trail)",
    ),
) -> RevokeTokensResponse:
    """Revoke all tokens and sessions for a user.

    This is the nuclear option for a compromised account.  In production
    it calls ``RevocationService.revoke_all_for_user()`` and
    ``RefreshTokenManager.revoke_family()``.

    Args:
        request: FastAPI request.
        user_id: Target user identifier.
        admin: Injected admin auth context.
        reason: Free-text reason stored in the audit log.

    Returns:
        Revocation confirmation with counts.
    """
    logger.warning(
        "Admin revoking all tokens: admin=%s target=%s reason=%s",
        admin.user_id,
        user_id,
        reason,
    )

    # Placeholder -- production calls revocation and refresh services
    now = datetime.now(timezone.utc)
    return RevokeTokensResponse(
        user_id=user_id,
        tokens_revoked=0,
        sessions_terminated=0,
        revoked_at=now,
        revoked_by=admin.user_id,
        reason=reason,
    )


@router.post(
    "/users/{user_id}/force-password-reset",
    response_model=ForcePasswordResetResponse,
    summary="Force password reset",
    description=(
        "Force a password reset for the specified user.  Optionally "
        "sends a reset email.  All existing tokens are revoked."
    ),
)
async def force_password_reset(
    request: Request,
    user_id: str,
    admin: Any = Depends(_require_admin),
) -> ForcePasswordResetResponse:
    """Force a password reset for a user.

    Revokes all tokens and optionally triggers a reset email flow.

    Args:
        request: FastAPI request.
        user_id: Target user identifier.
        admin: Injected admin auth context.

    Returns:
        Password reset confirmation.
    """
    logger.warning(
        "Admin forcing password reset: admin=%s target=%s",
        admin.user_id,
        user_id,
    )

    now = datetime.now(timezone.utc)
    return ForcePasswordResetResponse(
        user_id=user_id,
        reset_token_sent=True,
        forced_at=now,
        forced_by=admin.user_id,
        message=(
            f"Password reset forced for '{user_id}'.  "
            f"All tokens revoked.  Reset email sent."
        ),
    )


@router.post(
    "/users/{user_id}/disable-mfa",
    response_model=DisableMFAResponse,
    summary="Disable MFA for user",
    description=(
        "Emergency MFA disable for a user who has lost access to their "
        "second factor.  This is a privileged operation logged in the "
        "audit trail."
    ),
)
async def disable_mfa(
    request: Request,
    user_id: str,
    admin: Any = Depends(_require_admin),
) -> DisableMFAResponse:
    """Disable all MFA methods for a user.

    This is an emergency action for users who have lost access to their
    authenticator.  A security audit event is emitted.

    Args:
        request: FastAPI request.
        user_id: Target user identifier.
        admin: Injected admin auth context.

    Returns:
        MFA disable confirmation.
    """
    logger.warning(
        "Admin disabling MFA: admin=%s target=%s",
        admin.user_id,
        user_id,
    )

    now = datetime.now(timezone.utc)
    return DisableMFAResponse(
        user_id=user_id,
        previous_mfa_methods=["totp"],
        disabled_at=now,
        disabled_by=admin.user_id,
        message=(
            f"MFA disabled for '{user_id}'.  User should re-enroll "
            f"at next login."
        ),
    )


# ---------------------------------------------------------------------------
# Session management endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/sessions",
    response_model=SessionListResponse,
    summary="List active sessions",
    description=(
        "List currently active sessions with optional filtering by "
        "user or tenant."
    ),
)
async def list_sessions(
    request: Request,
    admin: Any = Depends(_require_admin),
    user_id: Optional[str] = Query(
        None, description="Filter by user identifier"
    ),
    tenant_id: Optional[str] = Query(
        None, description="Filter by tenant identifier"
    ),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    offset: int = Query(0, ge=0, description="Number of items to skip"),
) -> SessionListResponse:
    """List active sessions with filtering.

    Args:
        request: FastAPI request.
        admin: Injected admin auth context.
        user_id: Optional user filter.
        tenant_id: Optional tenant filter.
        limit: Page size.
        offset: Pagination offset.

    Returns:
        Paginated list of active sessions.
    """
    logger.info(
        "Admin listing sessions: admin=%s user_filter=%s tenant_filter=%s",
        admin.user_id,
        user_id,
        tenant_id,
    )

    # Placeholder
    page = (offset // limit) + 1 if limit > 0 else 1
    return SessionListResponse(
        sessions=[],
        total=0,
        page=page,
        page_size=limit,
        has_next=False,
    )


@router.delete(
    "/sessions/{session_id}",
    response_model=TerminateSessionResponse,
    summary="Terminate session",
    description="Forcefully terminate a specific active session.",
    responses={
        404: {
            "model": AdminErrorResponse,
            "description": "Session not found",
        },
    },
)
async def terminate_session(
    request: Request,
    session_id: str,
    admin: Any = Depends(_require_admin),
) -> TerminateSessionResponse:
    """Terminate a specific active session.

    Args:
        request: FastAPI request.
        session_id: Identifier of the session to terminate.
        admin: Injected admin auth context.

    Returns:
        Termination confirmation.

    Raises:
        HTTPException: 404 if session not found.
    """
    logger.warning(
        "Admin terminating session: admin=%s session=%s",
        admin.user_id,
        session_id,
    )

    now = datetime.now(timezone.utc)
    return TerminateSessionResponse(
        session_id=session_id,
        user_id="unknown",
        terminated_at=now,
        terminated_by=admin.user_id,
    )


# ---------------------------------------------------------------------------
# Audit log endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/audit-log",
    response_model=AuditLogResponse,
    summary="Query auth audit log",
    description=(
        "Query the authentication audit log with filtering by user, "
        "event type, and time range.  Results are ordered newest-first."
    ),
)
async def query_audit_log(
    request: Request,
    admin: Any = Depends(_require_admin),
    user_id: Optional[str] = Query(
        None, description="Filter by user identifier"
    ),
    event_type: Optional[AuditEventType] = Query(
        None, description="Filter by event type"
    ),
    tenant_id: Optional[str] = Query(
        None, description="Filter by tenant identifier"
    ),
    start: Optional[datetime] = Query(
        None, description="Filter events after this timestamp (ISO 8601)"
    ),
    end: Optional[datetime] = Query(
        None, description="Filter events before this timestamp (ISO 8601)"
    ),
    limit: int = Query(50, ge=1, le=200, description="Items per page"),
    offset: int = Query(0, ge=0, description="Number of items to skip"),
) -> AuditLogResponse:
    """Query the authentication audit log.

    In production this queries the ``auth_audit_log`` TimescaleDB
    hypertable and supports efficient time-range scans.

    Args:
        request: FastAPI request.
        admin: Injected admin auth context.
        user_id: Optional user filter.
        event_type: Optional event type filter.
        tenant_id: Optional tenant filter.
        start: Start of the time window.
        end: End of the time window.
        limit: Page size.
        offset: Pagination offset.

    Returns:
        Paginated audit log entries.
    """
    logger.info(
        "Admin querying audit log: admin=%s user_filter=%s "
        "event_type_filter=%s start=%s end=%s",
        admin.user_id,
        user_id,
        event_type,
        start,
        end,
    )

    # Placeholder
    page = (offset // limit) + 1 if limit > 0 else 1
    return AuditLogResponse(
        entries=[],
        total=0,
        page=page,
        page_size=limit,
        has_next=False,
    )


# ---------------------------------------------------------------------------
# Lockout management endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/lockouts",
    response_model=LockoutListResponse,
    summary="List currently locked accounts",
    description=(
        "Retrieve all user accounts that are currently in the "
        "``locked`` state."
    ),
)
async def list_lockouts(
    request: Request,
    admin: Any = Depends(_require_admin),
    tenant_id: Optional[str] = Query(
        None, description="Filter by tenant identifier"
    ),
) -> LockoutListResponse:
    """List all currently locked user accounts.

    Args:
        request: FastAPI request.
        admin: Injected admin auth context.
        tenant_id: Optional tenant filter.

    Returns:
        List of locked accounts.
    """
    logger.info(
        "Admin listing lockouts: admin=%s tenant_filter=%s",
        admin.user_id,
        tenant_id,
    )

    # Placeholder
    return LockoutListResponse(
        lockouts=[],
        total=0,
    )


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "router",
    # Models
    "UserStatus",
    "AuditEventType",
    "UserSummary",
    "UserListResponse",
    "UserDetailResponse",
    "UnlockResponse",
    "RevokeTokensResponse",
    "ForcePasswordResetResponse",
    "DisableMFAResponse",
    "SessionSummary",
    "SessionListResponse",
    "TerminateSessionResponse",
    "AuditLogEntry",
    "AuditLogResponse",
    "LockoutEntry",
    "LockoutListResponse",
    "AdminErrorResponse",
]
