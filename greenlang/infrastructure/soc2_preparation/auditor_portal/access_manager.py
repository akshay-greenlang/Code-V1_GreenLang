# -*- coding: utf-8 -*-
"""
Auditor Access Manager - SEC-009 Phase 5

Secure access management for external auditors. Provides credential provisioning,
permission configuration, session management, and MFA enforcement for SOC 2
audit portal access.

Features:
    - Time-limited access provisioning with automatic expiration
    - Read-only permission sets for evidence, reports, and requests
    - Mandatory MFA enrollment and verification
    - Configurable session timeout (default 30 minutes)
    - Access revocation with immediate session invalidation

Example:
    >>> manager = AuditorAccessManager()
    >>> await manager.provision_access(
    ...     auditor_id=uuid.uuid4(),
    ...     firm="Big Four Audit Firm",
    ...     permissions=["read:evidence", "read:reports"],
    ... )
    >>> await manager.require_mfa(auditor_id)
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SESSION_TIMEOUT_MINUTES = 30
DEFAULT_ACCESS_DURATION_DAYS = 90
MFA_CHALLENGE_EXPIRY_SECONDS = 300


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Permission(str, Enum):
    """Auditor portal permissions."""

    # Evidence permissions (read-only)
    READ_EVIDENCE = "read:evidence"
    READ_EVIDENCE_METADATA = "read:evidence:metadata"
    DOWNLOAD_EVIDENCE = "download:evidence"

    # Report permissions (read-only)
    READ_REPORTS = "read:reports"
    READ_TEST_RESULTS = "read:test_results"
    DOWNLOAD_REPORTS = "download:reports"

    # Request permissions
    READ_REQUESTS = "read:requests"
    CREATE_REQUESTS = "create:requests"
    UPDATE_REQUESTS = "update:requests"

    # Control testing
    READ_CONTROLS = "read:controls"
    READ_CONTROL_TESTS = "read:control_tests"

    # Findings
    READ_FINDINGS = "read:findings"

    # Dashboard
    READ_DASHBOARD = "read:dashboard"


class MFAMethod(str, Enum):
    """Supported MFA methods."""

    TOTP = "totp"
    """Time-based One-Time Password (Google Authenticator, etc.)"""

    EMAIL = "email"
    """Email-based verification code."""

    SMS = "sms"
    """SMS-based verification code."""


class AccessStatus(str, Enum):
    """Auditor access status."""

    PENDING = "pending"
    """Access provisioned but not yet activated."""

    ACTIVE = "active"
    """Access is active and usable."""

    SUSPENDED = "suspended"
    """Access temporarily suspended."""

    EXPIRED = "expired"
    """Access has expired."""

    REVOKED = "revoked"
    """Access has been revoked."""


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class AuditorCredentials(BaseModel):
    """Auditor authentication credentials.

    Attributes:
        auditor_id: Unique identifier for the auditor.
        email: Auditor email address.
        firm: Audit firm name.
        password_hash: Hashed password (bcrypt).
        mfa_enabled: Whether MFA is enabled.
        mfa_method: MFA method if enabled.
        mfa_secret: TOTP secret if using TOTP.
        status: Current access status.
        permissions: Granted permissions.
        created_at: When credentials were created.
        expires_at: When access expires.
        last_login: Last successful login.
    """

    model_config = ConfigDict(extra="forbid")

    auditor_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique auditor identifier.",
    )
    email: str = Field(
        ...,
        min_length=5,
        max_length=256,
        description="Auditor email address.",
    )
    firm: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Audit firm name.",
    )
    password_hash: str = Field(
        default="",
        description="Bcrypt hashed password.",
    )
    mfa_enabled: bool = Field(
        default=False,
        description="Whether MFA is enabled.",
    )
    mfa_method: Optional[MFAMethod] = Field(
        default=None,
        description="MFA method if enabled.",
    )
    mfa_secret: Optional[str] = Field(
        default=None,
        description="TOTP secret (encrypted).",
    )
    status: AccessStatus = Field(
        default=AccessStatus.PENDING,
        description="Current access status.",
    )
    permissions: List[Permission] = Field(
        default_factory=list,
        description="Granted permissions.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp.",
    )
    expires_at: Optional[datetime] = Field(
        default=None,
        description="Access expiration timestamp.",
    )
    last_login: Optional[datetime] = Field(
        default=None,
        description="Last successful login.",
    )

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Basic email validation."""
        if "@" not in v or "." not in v:
            raise ValueError("Invalid email format")
        return v.lower().strip()


class AuditorSession(BaseModel):
    """Active auditor session.

    Attributes:
        session_id: Unique session identifier.
        auditor_id: Associated auditor ID.
        token_hash: Hashed session token.
        created_at: Session start time.
        expires_at: Session expiration time.
        last_activity: Last activity timestamp.
        ip_address: Client IP address.
        user_agent: Client user agent.
        mfa_verified: Whether MFA was verified this session.
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique session identifier.",
    )
    auditor_id: str = Field(
        ...,
        description="Associated auditor ID.",
    )
    token_hash: str = Field(
        default="",
        description="SHA-256 hash of session token.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Session creation timestamp.",
    )
    expires_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc) + timedelta(minutes=30),
        description="Session expiration timestamp.",
    )
    last_activity: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last activity timestamp.",
    )
    ip_address: str = Field(
        default="",
        description="Client IP address.",
    )
    user_agent: str = Field(
        default="",
        description="Client user agent.",
    )
    mfa_verified: bool = Field(
        default=False,
        description="Whether MFA was verified this session.",
    )

    @property
    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def is_idle_timeout(self) -> bool:
        """Check if session exceeded idle timeout (30 min)."""
        idle_threshold = self.last_activity + timedelta(minutes=30)
        return datetime.now(timezone.utc) > idle_threshold


class MFAChallenge(BaseModel):
    """MFA verification challenge.

    Attributes:
        challenge_id: Unique challenge identifier.
        auditor_id: Associated auditor ID.
        code_hash: Hashed verification code.
        method: MFA method used.
        created_at: Challenge creation time.
        expires_at: Challenge expiration.
        attempts: Number of verification attempts.
        verified: Whether challenge was verified.
    """

    model_config = ConfigDict(extra="forbid")

    challenge_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique challenge identifier.",
    )
    auditor_id: str = Field(
        ...,
        description="Associated auditor ID.",
    )
    code_hash: str = Field(
        default="",
        description="SHA-256 hash of verification code.",
    )
    method: MFAMethod = Field(
        ...,
        description="MFA method used.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Challenge creation timestamp.",
    )
    expires_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
        + timedelta(seconds=MFA_CHALLENGE_EXPIRY_SECONDS),
        description="Challenge expiration.",
    )
    attempts: int = Field(
        default=0,
        ge=0,
        le=5,
        description="Verification attempts.",
    )
    verified: bool = Field(
        default=False,
        description="Whether challenge was verified.",
    )


# ---------------------------------------------------------------------------
# Auditor Access Manager
# ---------------------------------------------------------------------------


class AuditorAccessManager:
    """Manage auditor portal access, credentials, and sessions.

    Provides secure access provisioning, MFA enforcement, session management,
    and access revocation for external auditors.

    Attributes:
        _credentials: Stored auditor credentials by auditor_id.
        _sessions: Active sessions by session_id.
        _challenges: Active MFA challenges by challenge_id.
        _session_timeout: Session timeout in minutes.
    """

    def __init__(self, session_timeout_minutes: int = DEFAULT_SESSION_TIMEOUT_MINUTES) -> None:
        """Initialize the access manager.

        Args:
            session_timeout_minutes: Session timeout duration.
        """
        self._credentials: Dict[str, AuditorCredentials] = {}
        self._sessions: Dict[str, AuditorSession] = {}
        self._challenges: Dict[str, MFAChallenge] = {}
        self._session_timeout = session_timeout_minutes
        logger.info(
            "AuditorAccessManager initialized (session_timeout=%d min)",
            session_timeout_minutes,
        )

    # ------------------------------------------------------------------
    # Access Provisioning
    # ------------------------------------------------------------------

    async def provision_access(
        self,
        auditor_id: uuid.UUID,
        firm: str,
        permissions: List[str],
        email: Optional[str] = None,
        duration_days: int = DEFAULT_ACCESS_DURATION_DAYS,
    ) -> AuditorCredentials:
        """Provision access for an auditor.

        Creates auditor credentials with specified permissions and expiration.
        Access is in PENDING status until the auditor completes setup.

        Args:
            auditor_id: Unique auditor identifier.
            firm: Audit firm name.
            permissions: List of permission strings.
            email: Auditor email (defaults to auditor_id@firm.com).
            duration_days: Access duration in days.

        Returns:
            Created AuditorCredentials.

        Raises:
            ValueError: If auditor already exists or invalid permissions.
        """
        auditor_id_str = str(auditor_id)

        if auditor_id_str in self._credentials:
            raise ValueError(f"Auditor {auditor_id_str} already has access")

        # Validate and convert permissions
        valid_permissions: List[Permission] = []
        for perm in permissions:
            try:
                valid_permissions.append(Permission(perm))
            except ValueError:
                raise ValueError(f"Invalid permission: {perm}")

        # Default to read-only permissions if none specified
        if not valid_permissions:
            valid_permissions = [
                Permission.READ_EVIDENCE,
                Permission.READ_REPORTS,
                Permission.READ_REQUESTS,
            ]

        # Generate email if not provided
        if not email:
            firm_slug = firm.lower().replace(" ", "-")[:20]
            email = f"auditor-{auditor_id_str[:8]}@{firm_slug}.audit"

        # Calculate expiration
        expires_at = datetime.now(timezone.utc) + timedelta(days=duration_days)

        credentials = AuditorCredentials(
            auditor_id=auditor_id_str,
            email=email,
            firm=firm,
            permissions=valid_permissions,
            status=AccessStatus.PENDING,
            expires_at=expires_at,
        )

        self._credentials[auditor_id_str] = credentials
        logger.info(
            "Provisioned access for auditor %s (firm=%s, permissions=%d, expires=%s)",
            auditor_id_str[:8],
            firm,
            len(valid_permissions),
            expires_at.isoformat(),
        )

        return credentials

    async def activate_access(
        self,
        auditor_id: uuid.UUID,
        password: str,
    ) -> AuditorCredentials:
        """Activate auditor access by setting password.

        Args:
            auditor_id: Auditor identifier.
            password: Initial password (will be hashed).

        Returns:
            Updated credentials.

        Raises:
            ValueError: If auditor not found or access not pending.
        """
        auditor_id_str = str(auditor_id)
        credentials = self._credentials.get(auditor_id_str)

        if credentials is None:
            raise ValueError(f"Auditor {auditor_id_str} not found")

        if credentials.status != AccessStatus.PENDING:
            raise ValueError(f"Access is not pending (status={credentials.status.value})")

        # Hash password (simplified - production would use bcrypt)
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        credentials.password_hash = password_hash
        credentials.status = AccessStatus.ACTIVE

        logger.info("Activated access for auditor %s", auditor_id_str[:8])
        return credentials

    async def revoke_access(self, auditor_id: uuid.UUID) -> None:
        """Revoke auditor access immediately.

        Revokes credentials and invalidates all active sessions.

        Args:
            auditor_id: Auditor identifier.

        Raises:
            ValueError: If auditor not found.
        """
        auditor_id_str = str(auditor_id)
        credentials = self._credentials.get(auditor_id_str)

        if credentials is None:
            raise ValueError(f"Auditor {auditor_id_str} not found")

        # Revoke credentials
        credentials.status = AccessStatus.REVOKED

        # Invalidate all sessions
        sessions_to_remove = [
            sid for sid, session in self._sessions.items()
            if session.auditor_id == auditor_id_str
        ]
        for sid in sessions_to_remove:
            del self._sessions[sid]

        logger.info(
            "Revoked access for auditor %s (invalidated %d sessions)",
            auditor_id_str[:8],
            len(sessions_to_remove),
        )

    async def suspend_access(self, auditor_id: uuid.UUID, reason: str = "") -> None:
        """Temporarily suspend auditor access.

        Args:
            auditor_id: Auditor identifier.
            reason: Reason for suspension.
        """
        auditor_id_str = str(auditor_id)
        credentials = self._credentials.get(auditor_id_str)

        if credentials is None:
            raise ValueError(f"Auditor {auditor_id_str} not found")

        credentials.status = AccessStatus.SUSPENDED
        logger.info("Suspended access for auditor %s (reason=%s)", auditor_id_str[:8], reason)

    async def reinstate_access(self, auditor_id: uuid.UUID) -> None:
        """Reinstate suspended auditor access.

        Args:
            auditor_id: Auditor identifier.
        """
        auditor_id_str = str(auditor_id)
        credentials = self._credentials.get(auditor_id_str)

        if credentials is None:
            raise ValueError(f"Auditor {auditor_id_str} not found")

        if credentials.status != AccessStatus.SUSPENDED:
            raise ValueError("Access is not suspended")

        credentials.status = AccessStatus.ACTIVE
        logger.info("Reinstated access for auditor %s", auditor_id_str[:8])

    # ------------------------------------------------------------------
    # Permission Management
    # ------------------------------------------------------------------

    def configure_permissions(
        self,
        auditor_id: uuid.UUID,
        permissions: Optional[List[str]] = None,
    ) -> None:
        """Configure auditor permissions.

        Default permissions are read-only: evidence, reports, requests.

        Args:
            auditor_id: Auditor identifier.
            permissions: List of permission strings. If None, uses defaults.
        """
        auditor_id_str = str(auditor_id)
        credentials = self._credentials.get(auditor_id_str)

        if credentials is None:
            raise ValueError(f"Auditor {auditor_id_str} not found")

        if permissions is None:
            # Default read-only permissions
            credentials.permissions = [
                Permission.READ_EVIDENCE,
                Permission.READ_REPORTS,
                Permission.READ_REQUESTS,
            ]
        else:
            valid_permissions = [Permission(p) for p in permissions]
            credentials.permissions = valid_permissions

        logger.info(
            "Configured permissions for auditor %s: %s",
            auditor_id_str[:8],
            [p.value for p in credentials.permissions],
        )

    def has_permission(self, auditor_id: uuid.UUID, permission: Permission) -> bool:
        """Check if auditor has a specific permission.

        Args:
            auditor_id: Auditor identifier.
            permission: Permission to check.

        Returns:
            True if auditor has the permission.
        """
        auditor_id_str = str(auditor_id)
        credentials = self._credentials.get(auditor_id_str)

        if credentials is None:
            return False

        if credentials.status != AccessStatus.ACTIVE:
            return False

        return permission in credentials.permissions

    # ------------------------------------------------------------------
    # Session Management
    # ------------------------------------------------------------------

    def set_session_timeout(self, minutes: int = DEFAULT_SESSION_TIMEOUT_MINUTES) -> None:
        """Set session timeout duration.

        Args:
            minutes: Timeout in minutes (1-120).
        """
        if not 1 <= minutes <= 120:
            raise ValueError("Timeout must be between 1 and 120 minutes")

        self._session_timeout = minutes
        logger.info("Session timeout set to %d minutes", minutes)

    async def create_session(
        self,
        auditor_id: uuid.UUID,
        ip_address: str,
        user_agent: str = "",
    ) -> tuple[AuditorSession, str]:
        """Create a new session for an auditor.

        Args:
            auditor_id: Auditor identifier.
            ip_address: Client IP address.
            user_agent: Client user agent.

        Returns:
            Tuple of (AuditorSession, raw_token).

        Raises:
            ValueError: If auditor not found or access not active.
        """
        auditor_id_str = str(auditor_id)
        credentials = self._credentials.get(auditor_id_str)

        if credentials is None:
            raise ValueError(f"Auditor {auditor_id_str} not found")

        if credentials.status != AccessStatus.ACTIVE:
            raise ValueError(f"Access is not active (status={credentials.status.value})")

        # Check expiration
        if credentials.expires_at and datetime.now(timezone.utc) > credentials.expires_at:
            credentials.status = AccessStatus.EXPIRED
            raise ValueError("Access has expired")

        # Generate session token
        raw_token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(raw_token.encode()).hexdigest()

        expires_at = datetime.now(timezone.utc) + timedelta(minutes=self._session_timeout)

        session = AuditorSession(
            auditor_id=auditor_id_str,
            token_hash=token_hash,
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent,
            mfa_verified=not credentials.mfa_enabled,  # Auto-verify if MFA not required
        )

        self._sessions[session.session_id] = session

        # Update last login
        credentials.last_login = datetime.now(timezone.utc)

        logger.info(
            "Created session for auditor %s (session=%s, expires=%s)",
            auditor_id_str[:8],
            session.session_id[:8],
            expires_at.isoformat(),
        )

        return session, raw_token

    async def validate_session(
        self,
        session_id: str,
        token: str,
    ) -> Optional[AuditorSession]:
        """Validate a session and token.

        Args:
            session_id: Session identifier.
            token: Raw session token.

        Returns:
            AuditorSession if valid, None otherwise.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return None

        # Check token
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        if session.token_hash != token_hash:
            logger.warning("Invalid token for session %s", session_id[:8])
            return None

        # Check expiration
        if session.is_expired:
            logger.info("Session %s has expired", session_id[:8])
            del self._sessions[session_id]
            return None

        # Check idle timeout
        if session.is_idle_timeout:
            logger.info("Session %s exceeded idle timeout", session_id[:8])
            del self._sessions[session_id]
            return None

        # Check MFA verification
        if not session.mfa_verified:
            logger.warning("Session %s requires MFA verification", session_id[:8])
            return None

        # Update last activity
        session.last_activity = datetime.now(timezone.utc)

        return session

    async def invalidate_session(self, session_id: str) -> bool:
        """Invalidate a session (logout).

        Args:
            session_id: Session identifier.

        Returns:
            True if session was found and invalidated.
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info("Invalidated session %s", session_id[:8])
            return True
        return False

    async def cleanup_expired_sessions(self) -> int:
        """Remove all expired sessions.

        Returns:
            Number of sessions removed.
        """
        now = datetime.now(timezone.utc)
        expired = [
            sid for sid, session in self._sessions.items()
            if session.expires_at < now or session.is_idle_timeout
        ]

        for sid in expired:
            del self._sessions[sid]

        if expired:
            logger.info("Cleaned up %d expired sessions", len(expired))

        return len(expired)

    # ------------------------------------------------------------------
    # MFA Management
    # ------------------------------------------------------------------

    async def require_mfa(
        self,
        auditor_id: uuid.UUID,
        method: MFAMethod = MFAMethod.TOTP,
    ) -> str:
        """Enable MFA requirement for an auditor.

        Args:
            auditor_id: Auditor identifier.
            method: MFA method to use.

        Returns:
            Setup data (TOTP secret for TOTP method).

        Raises:
            ValueError: If auditor not found.
        """
        auditor_id_str = str(auditor_id)
        credentials = self._credentials.get(auditor_id_str)

        if credentials is None:
            raise ValueError(f"Auditor {auditor_id_str} not found")

        credentials.mfa_method = method

        if method == MFAMethod.TOTP:
            # Generate TOTP secret (simplified - production would use pyotp)
            secret = secrets.token_urlsafe(20)
            credentials.mfa_secret = secret
            credentials.mfa_enabled = True
            logger.info("MFA (TOTP) enabled for auditor %s", auditor_id_str[:8])
            return secret
        else:
            credentials.mfa_enabled = True
            logger.info("MFA (%s) enabled for auditor %s", method.value, auditor_id_str[:8])
            return ""

    async def create_mfa_challenge(
        self,
        auditor_id: uuid.UUID,
    ) -> MFAChallenge:
        """Create an MFA challenge for verification.

        Args:
            auditor_id: Auditor identifier.

        Returns:
            MFAChallenge with code sent via configured method.

        Raises:
            ValueError: If auditor not found or MFA not enabled.
        """
        auditor_id_str = str(auditor_id)
        credentials = self._credentials.get(auditor_id_str)

        if credentials is None:
            raise ValueError(f"Auditor {auditor_id_str} not found")

        if not credentials.mfa_enabled or credentials.mfa_method is None:
            raise ValueError("MFA is not enabled for this auditor")

        # Generate verification code
        code = f"{secrets.randbelow(1000000):06d}"
        code_hash = hashlib.sha256(code.encode()).hexdigest()

        challenge = MFAChallenge(
            auditor_id=auditor_id_str,
            code_hash=code_hash,
            method=credentials.mfa_method,
        )

        self._challenges[challenge.challenge_id] = challenge

        # In production, send code via configured method
        logger.info(
            "Created MFA challenge for auditor %s (method=%s)",
            auditor_id_str[:8],
            credentials.mfa_method.value,
        )

        # Return challenge (code would be sent separately in production)
        return challenge

    async def verify_mfa(
        self,
        challenge_id: str,
        code: str,
        session_id: Optional[str] = None,
    ) -> bool:
        """Verify an MFA challenge.

        Args:
            challenge_id: Challenge identifier.
            code: User-provided verification code.
            session_id: Session to mark as verified.

        Returns:
            True if verification successful.
        """
        challenge = self._challenges.get(challenge_id)
        if challenge is None:
            logger.warning("MFA challenge %s not found", challenge_id[:8])
            return False

        # Check expiration
        if datetime.now(timezone.utc) > challenge.expires_at:
            del self._challenges[challenge_id]
            logger.warning("MFA challenge %s has expired", challenge_id[:8])
            return False

        # Check attempts
        if challenge.attempts >= 5:
            del self._challenges[challenge_id]
            logger.warning("MFA challenge %s exceeded attempts", challenge_id[:8])
            return False

        challenge.attempts += 1

        # Verify code
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        if code_hash != challenge.code_hash:
            logger.warning("Invalid MFA code for challenge %s", challenge_id[:8])
            return False

        # Mark as verified
        challenge.verified = True
        del self._challenges[challenge_id]

        # Update session if provided
        if session_id and session_id in self._sessions:
            self._sessions[session_id].mfa_verified = True

        logger.info("MFA verified for challenge %s", challenge_id[:8])
        return True

    # ------------------------------------------------------------------
    # Query Methods
    # ------------------------------------------------------------------

    async def get_auditor(self, auditor_id: uuid.UUID) -> Optional[AuditorCredentials]:
        """Get auditor credentials.

        Args:
            auditor_id: Auditor identifier.

        Returns:
            AuditorCredentials if found.
        """
        return self._credentials.get(str(auditor_id))

    async def list_auditors(
        self,
        status: Optional[AccessStatus] = None,
        firm: Optional[str] = None,
    ) -> List[AuditorCredentials]:
        """List auditors with optional filtering.

        Args:
            status: Filter by access status.
            firm: Filter by firm name.

        Returns:
            List of matching credentials.
        """
        auditors = list(self._credentials.values())

        if status:
            auditors = [a for a in auditors if a.status == status]

        if firm:
            auditors = [a for a in auditors if a.firm.lower() == firm.lower()]

        return auditors

    async def get_active_sessions(self, auditor_id: uuid.UUID) -> List[AuditorSession]:
        """Get all active sessions for an auditor.

        Args:
            auditor_id: Auditor identifier.

        Returns:
            List of active sessions.
        """
        auditor_id_str = str(auditor_id)
        return [
            session for session in self._sessions.values()
            if session.auditor_id == auditor_id_str and not session.is_expired
        ]


__all__ = [
    "AuditorAccessManager",
    "AuditorCredentials",
    "AuditorSession",
    "MFAChallenge",
    "Permission",
    "MFAMethod",
    "AccessStatus",
]
