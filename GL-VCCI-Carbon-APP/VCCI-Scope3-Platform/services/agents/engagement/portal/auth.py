# -*- coding: utf-8 -*-
"""
Authentication for supplier portal.

Supports OAuth 2.0 and magic link (passwordless) authentication.
"""
import logging
from typing import Optional, Dict
from datetime import datetime, timedelta
import secrets
import hashlib
import uuid

from ..models import SupplierPortalSession
from ..exceptions import AuthenticationError
from greenlang.determinism import DeterministicClock
from greenlang.determinism import deterministic_uuid, DeterministicClock


logger = logging.getLogger(__name__)


class PortalAuthenticator:
    """
    Handles supplier portal authentication with OAuth 2.0 and magic links.

    Features:
    - Magic link generation (passwordless)
    - OAuth 2.0 provider integration (stubs)
    - Session management
    - Token validation
    """

    def __init__(self, session_duration_hours: int = 24):
        """
        Initialize portal authenticator.

        Args:
            session_duration_hours: Session duration in hours
        """
        self.session_duration_hours = session_duration_hours
        self.sessions: Dict[str, SupplierPortalSession] = {}
        self.magic_links: Dict[str, Dict] = {}  # token -> {supplier_id, email, expires_at}
        logger.info("PortalAuthenticator initialized")

    def generate_magic_link(
        self,
        supplier_id: str,
        email: str,
        base_url: str = "https://portal.company.com"
    ) -> str:
        """
        Generate magic link for passwordless login.

        Args:
            supplier_id: Supplier identifier
            email: Supplier email
            base_url: Portal base URL

        Returns:
            Magic link URL
        """
        # Generate secure token
        token = secrets.token_urlsafe(32)

        # Store magic link with expiry (15 minutes)
        expires_at = DeterministicClock.utcnow() + timedelta(minutes=15)
        self.magic_links[token] = {
            "supplier_id": supplier_id,
            "email": email,
            "expires_at": expires_at
        }

        # Generate URL
        magic_link = f"{base_url}/auth/magic-link?token={token}"

        logger.info(f"Generated magic link for supplier {supplier_id}")

        return magic_link

    def validate_magic_link(self, token: str) -> Optional[Dict]:
        """
        Validate magic link token.

        Args:
            token: Magic link token

        Returns:
            Magic link data if valid, None otherwise
        """
        if token not in self.magic_links:
            logger.warning(f"Invalid magic link token: {token[:10]}...")
            return None

        magic_link_data = self.magic_links[token]

        # Check expiry
        if DeterministicClock.utcnow() > magic_link_data['expires_at']:
            logger.warning(f"Expired magic link token: {token[:10]}...")
            del self.magic_links[token]
            return None

        return magic_link_data

    def authenticate_with_magic_link(
        self,
        token: str,
        ip_address: Optional[str] = None
    ) -> SupplierPortalSession:
        """
        Authenticate supplier with magic link.

        Args:
            token: Magic link token
            ip_address: Client IP address

        Returns:
            Portal session

        Raises:
            AuthenticationError: If authentication fails
        """
        # Validate token
        magic_link_data = self.validate_magic_link(token)
        if not magic_link_data:
            raise AuthenticationError("", "Invalid or expired magic link")

        # Create session
        session = self._create_session(
            supplier_id=magic_link_data['supplier_id'],
            email=magic_link_data['email'],
            magic_link_token=token,
            ip_address=ip_address
        )

        # Remove used magic link
        del self.magic_links[token]

        logger.info(f"Authenticated supplier {magic_link_data['supplier_id']} via magic link")

        return session

    def authenticate_with_oauth(
        self,
        supplier_id: str,
        email: str,
        provider: str,
        oauth_token: str,
        ip_address: Optional[str] = None
    ) -> SupplierPortalSession:
        """
        Authenticate supplier with OAuth 2.0 provider.

        Args:
            supplier_id: Supplier identifier
            email: Supplier email
            provider: OAuth provider (google, microsoft)
            oauth_token: OAuth access token
            ip_address: Client IP address

        Returns:
            Portal session

        Raises:
            AuthenticationError: If authentication fails
        """
        # In production, validate OAuth token with provider
        # For now, stub implementation
        if not oauth_token:
            raise AuthenticationError(supplier_id, "Invalid OAuth token")

        # Create session
        session = self._create_session(
            supplier_id=supplier_id,
            email=email,
            oauth_provider=provider,
            ip_address=ip_address
        )

        logger.info(
            f"Authenticated supplier {supplier_id} via OAuth ({provider})"
        )

        return session

    def _create_session(
        self,
        supplier_id: str,
        email: str,
        magic_link_token: Optional[str] = None,
        oauth_provider: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> SupplierPortalSession:
        """
        Create portal session.

        Args:
            supplier_id: Supplier identifier
            email: Supplier email
            magic_link_token: Magic link token (if used)
            oauth_provider: OAuth provider (if used)
            ip_address: Client IP address

        Returns:
            Portal session
        """
        session_id = f"sess_{deterministic_uuid(__name__, str(DeterministicClock.now())).hex[:16]}"
        expires_at = DeterministicClock.utcnow() + timedelta(hours=self.session_duration_hours)

        session = SupplierPortalSession(
            session_id=session_id,
            supplier_id=supplier_id,
            email=email,
            magic_link_token=magic_link_token,
            oauth_provider=oauth_provider,
            expires_at=expires_at,
            ip_address=ip_address
        )

        self.sessions[session_id] = session

        return session

    def validate_session(self, session_id: str) -> bool:
        """
        Validate portal session.

        Args:
            session_id: Session identifier

        Returns:
            True if session is valid, False otherwise
        """
        if session_id not in self.sessions:
            return False

        session = self.sessions[session_id]

        # Check expiry
        if DeterministicClock.utcnow() > session.expires_at:
            logger.info(f"Session {session_id} expired")
            del self.sessions[session_id]
            return False

        # Update last activity
        session.last_activity = DeterministicClock.utcnow()

        return True

    def get_session(self, session_id: str) -> Optional[SupplierPortalSession]:
        """
        Get portal session.

        Args:
            session_id: Session identifier

        Returns:
            Portal session if valid, None otherwise
        """
        if not self.validate_session(session_id):
            return None

        return self.sessions.get(session_id)

    def logout(self, session_id: str):
        """
        Logout and invalidate session.

        Args:
            session_id: Session identifier
        """
        if session_id in self.sessions:
            supplier_id = self.sessions[session_id].supplier_id
            del self.sessions[session_id]
            logger.info(f"Logged out supplier {supplier_id}")

    def cleanup_expired_sessions(self) -> int:
        """
        Remove expired sessions.

        Returns:
            Number of sessions removed
        """
        now = DeterministicClock.utcnow()
        expired = [
            sid for sid, session in self.sessions.items()
            if now > session.expires_at
        ]

        for session_id in expired:
            del self.sessions[session_id]

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")

        return len(expired)
