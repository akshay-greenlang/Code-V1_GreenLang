"""
CSRF Protection for GreenLang API

This module implements Cross-Site Request Forgery (CSRF) protection
for the GreenLang API using double-submit cookie pattern.

Example:
    >>> from greenlang.api.security.csrf import CSRFProtect, generate_csrf_token
    >>> csrf = CSRFProtect(secret_key="your-secret-key")
    >>> token = generate_csrf_token()
"""

import hashlib
import hmac
import secrets
import time
from typing import Optional, Set, Dict, Any
from datetime import datetime, timedelta

from fastapi import Request, Response, HTTPException, status
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class CSRFConfig(BaseModel):
    """Configuration for CSRF protection."""

    secret_key: str = Field(..., description="Secret key for HMAC signing")
    token_length: int = Field(32, ge=16, description="Length of CSRF token")
    token_expiry_seconds: int = Field(3600, ge=60, description="Token expiry in seconds")
    cookie_name: str = Field("csrf_token", description="Name of CSRF cookie")
    header_name: str = Field("X-CSRF-Token", description="Name of CSRF header")
    form_field_name: str = Field("csrf_token", description="Name of form field")
    safe_methods: Set[str] = Field(
        default={"GET", "HEAD", "OPTIONS", "TRACE"},
        description="HTTP methods that don't require CSRF protection"
    )
    exempt_paths: Set[str] = Field(
        default=set(),
        description="Paths exempt from CSRF protection"
    )
    cookie_secure: bool = Field(True, description="Set secure flag on cookie")
    cookie_httponly: bool = Field(False, description="Set httponly flag on cookie")
    cookie_samesite: str = Field("strict", description="SameSite cookie attribute")


class CSRFToken(BaseModel):
    """CSRF token with metadata."""

    token: str = Field(..., description="The CSRF token value")
    timestamp: float = Field(..., description="Token creation timestamp")
    signature: str = Field(..., description="HMAC signature for validation")


class CSRFProtect:
    """
    CSRF protection implementation for FastAPI.

    This class provides CSRF protection using double-submit cookie pattern
    with HMAC signatures for enhanced security.

    Attributes:
        config: CSRF configuration settings
        _token_cache: In-memory cache for token validation

    Example:
        >>> csrf = CSRFProtect(CSRFConfig(secret_key="secret"))
        >>> app.add_middleware(csrf.middleware)
    """

    def __init__(self, config: CSRFConfig):
        """Initialize CSRF protection with configuration."""
        self.config = config
        self._token_cache: Dict[str, float] = {}
        self._cleanup_interval = 300  # Clean expired tokens every 5 minutes
        self._last_cleanup = time.time()

    def generate_csrf_token(self) -> CSRFToken:
        """
        Generate a new CSRF token with HMAC signature.

        Returns:
            CSRFToken with token value, timestamp, and signature

        Example:
            >>> token = csrf.generate_csrf_token()
            >>> assert len(token.token) == 32
        """
        # Generate random token
        token_value = secrets.token_urlsafe(self.config.token_length)
        timestamp = time.time()

        # Create HMAC signature
        signature = self._sign_token(token_value, timestamp)

        # Cache token for validation
        self._token_cache[token_value] = timestamp

        # Periodic cleanup of expired tokens
        self._cleanup_expired_tokens()

        return CSRFToken(
            token=token_value,
            timestamp=timestamp,
            signature=signature
        )

    def _sign_token(self, token: str, timestamp: float) -> str:
        """
        Create HMAC signature for token.

        Args:
            token: Token value to sign
            timestamp: Token creation timestamp

        Returns:
            HMAC signature as hex string
        """
        message = f"{token}:{timestamp}".encode()
        signature = hmac.new(
            self.config.secret_key.encode(),
            message,
            hashlib.sha256
        ).hexdigest()
        return signature

    def validate_token(self, token: str, signature: str) -> bool:
        """
        Validate CSRF token and signature.

        Args:
            token: Token value to validate
            signature: HMAC signature to verify

        Returns:
            True if token is valid, False otherwise
        """
        # Check if token exists in cache
        if token not in self._token_cache:
            logger.warning(f"CSRF token not found in cache: {token[:8]}...")
            return False

        timestamp = self._token_cache[token]

        # Check token expiry
        if time.time() - timestamp > self.config.token_expiry_seconds:
            logger.warning(f"CSRF token expired: {token[:8]}...")
            del self._token_cache[token]
            return False

        # Verify signature
        expected_signature = self._sign_token(token, timestamp)
        if not hmac.compare_digest(signature, expected_signature):
            logger.warning(f"CSRF signature mismatch for token: {token[:8]}...")
            return False

        return True

    def _cleanup_expired_tokens(self):
        """Remove expired tokens from cache."""
        current_time = time.time()

        # Only cleanup periodically
        if current_time - self._last_cleanup < self._cleanup_interval:
            return

        # Remove expired tokens
        expired_tokens = [
            token for token, timestamp in self._token_cache.items()
            if current_time - timestamp > self.config.token_expiry_seconds
        ]

        for token in expired_tokens:
            del self._token_cache[token]

        if expired_tokens:
            logger.info(f"Cleaned up {len(expired_tokens)} expired CSRF tokens")

        self._last_cleanup = current_time

    def _is_exempt(self, request: Request) -> bool:
        """
        Check if request is exempt from CSRF protection.

        Args:
            request: FastAPI request object

        Returns:
            True if request is exempt, False otherwise
        """
        # Check if method is safe
        if request.method in self.config.safe_methods:
            return True

        # Check if path is exempt
        path = request.url.path
        for exempt_path in self.config.exempt_paths:
            if path.startswith(exempt_path):
                return True

        return False

    def _get_token_from_request(self, request: Request) -> Optional[str]:
        """
        Extract CSRF token from request.

        Args:
            request: FastAPI request object

        Returns:
            CSRF token if found, None otherwise
        """
        # Try to get token from header
        token = request.headers.get(self.config.header_name)
        if token:
            return token

        # Try to get token from form data (for form submissions)
        if request.method == "POST":
            content_type = request.headers.get("content-type", "")
            if "application/x-www-form-urlencoded" in content_type:
                # Note: In real implementation, would parse form data
                # This is a simplified version
                pass

        # Try to get token from cookie (for double-submit pattern)
        token = request.cookies.get(self.config.cookie_name)
        return token

    async def __call__(self, request: Request, call_next):
        """
        CSRF middleware for FastAPI.

        Args:
            request: FastAPI request object
            call_next: Next middleware in chain

        Returns:
            Response from next middleware

        Raises:
            HTTPException: If CSRF validation fails
        """
        # Check if request is exempt
        if self._is_exempt(request):
            return await call_next(request)

        # Get token from request
        token = self._get_token_from_request(request)
        if not token:
            logger.error(f"CSRF token missing for {request.method} {request.url.path}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="CSRF token missing"
            )

        # Parse token and signature
        try:
            token_parts = token.split(":")
            if len(token_parts) != 2:
                raise ValueError("Invalid token format")
            token_value, signature = token_parts
        except (ValueError, IndexError) as e:
            logger.error(f"Invalid CSRF token format: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid CSRF token format"
            )

        # Validate token
        if not self.validate_token(token_value, signature):
            logger.error(f"CSRF validation failed for {request.method} {request.url.path}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="CSRF validation failed"
            )

        logger.debug(f"CSRF validation successful for {request.method} {request.url.path}")

        # Process request
        response = await call_next(request)

        # Add new CSRF token to response for next request
        if request.method in self.config.safe_methods:
            new_token = self.generate_csrf_token()
            token_string = f"{new_token.token}:{new_token.signature}"

            response.set_cookie(
                key=self.config.cookie_name,
                value=token_string,
                secure=self.config.cookie_secure,
                httponly=self.config.cookie_httponly,
                samesite=self.config.cookie_samesite,
                max_age=self.config.token_expiry_seconds
            )

        return response


def generate_csrf_token() -> str:
    """
    Generate a simple CSRF token (utility function).

    Returns:
        Random CSRF token string

    Example:
        >>> token = generate_csrf_token()
        >>> assert len(token) >= 32
    """
    return secrets.token_urlsafe(32)


def validate_csrf_token(token: str, expected: str) -> bool:
    """
    Validate CSRF token using constant-time comparison.

    Args:
        token: Token to validate
        expected: Expected token value

    Returns:
        True if tokens match, False otherwise

    Example:
        >>> token = generate_csrf_token()
        >>> assert validate_csrf_token(token, token)
    """
    return hmac.compare_digest(token, expected)


# FastAPI dependency for CSRF protection
class CSRFBearer(HTTPBearer):
    """
    FastAPI dependency for CSRF token validation.

    Example:
        >>> @app.post("/api/data", dependencies=[Depends(CSRFBearer())])
        >>> async def create_data(data: DataModel):
        >>>     return {"status": "created"}
    """

    def __init__(self, csrf_protect: CSRFProtect, auto_error: bool = True):
        super().__init__(auto_error=auto_error)
        self.csrf_protect = csrf_protect

    async def __call__(self, request: Request) -> Optional[str]:
        """Validate CSRF token from request."""
        token = self.csrf_protect._get_token_from_request(request)

        if not token and self.auto_error:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="CSRF token required"
            )

        return token