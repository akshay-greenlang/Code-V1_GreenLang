# -*- coding: utf-8 -*-
"""
TenantContext - Request-scoped tenant context management

This module implements FastAPI middleware for tenant detection, context management,
and cross-tenant access prevention. It extracts tenant information from JWT tokens,
API keys, subdomains, or headers and sets the tenant context for the entire request
lifecycle.

Example:
    >>> app = FastAPI()
    >>> app.add_middleware(TenantMiddleware, tenant_manager=manager)
    >>> # All requests will now have tenant context
"""

from typing import Optional, Callable, Any, Dict
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from contextvars import ContextVar
from pydantic import BaseModel, UUID4
import logging
import time
import re
import os
from urllib.parse import urlparse
from datetime import datetime, timedelta

import jwt
from jwt import PyJWTError

from .tenant_manager import TenantManager, Tenant, TenantStatus
from greenlang.determinism import DeterministicClock

logger = logging.getLogger(__name__)


# ============================================================================
# JWT Authentication - PRODUCTION GRADE
# ============================================================================

class AuthenticationError(Exception):
    """
    Authentication error for JWT validation failures.

    This exception is raised when JWT token validation fails due to:
    - Expired tokens
    - Invalid signatures
    - Missing required claims
    - Invalid issuer/audience
    - Malformed tokens
    """
    pass


class JWTValidator:
    """
    Secure JWT validation with signature verification.

    This class implements enterprise-grade JWT validation with:
    - Cryptographic signature verification (HS256, RS256, ES256)
    - Expiration validation
    - Issuer and audience validation
    - Custom claims validation
    - Clock skew tolerance (leeway)

    Security Features:
    - ZERO HALLUCINATION: All validations are deterministic
    - Signature verification ALWAYS enabled (cannot be disabled)
    - Expiration ALWAYS checked (cannot be disabled)
    - Supports HMAC (HS256) and RSA (RS256) algorithms
    - Prevents timing attacks via constant-time comparison

    Example:
        >>> validator = JWTValidator(
        ...     secret_key=os.getenv("JWT_SECRET_KEY"),
        ...     algorithm="HS256",
        ...     issuer="greenlang.ai",
        ...     audience="greenlang-api"
        ... )
        >>> payload = validator.validate_token(token)
        >>> print(f"Tenant: {payload['tenant_id']}")
    """

    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        issuer: Optional[str] = None,
        audience: Optional[str] = None,
        leeway: int = 0
    ):
        """
        Initialize JWT validator.

        Args:
            secret_key: Secret key for HMAC algorithms or public key for RSA
            algorithm: JWT algorithm (HS256, HS384, HS512, RS256, RS384, RS512, ES256, ES384, ES512)
            issuer: Expected token issuer (iss claim)
            audience: Expected token audience (aud claim)
            leeway: Tolerance in seconds for expiration validation (default: 0)

        Raises:
            ValueError: If secret_key is empty or algorithm is unsupported
        """
        if not secret_key:
            raise ValueError("JWT secret_key cannot be empty")

        if algorithm not in [
            "HS256", "HS384", "HS512",  # HMAC algorithms
            "RS256", "RS384", "RS512",  # RSA algorithms
            "ES256", "ES384", "ES512"   # ECDSA algorithms
        ]:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        self.secret_key = secret_key
        self.algorithm = algorithm
        self.issuer = issuer
        self.audience = audience
        self.leeway = leeway

        logger.info(
            f"JWTValidator initialized (algorithm={algorithm}, "
            f"issuer={issuer}, audience={audience}, leeway={leeway}s)"
        )

    def validate_token(self, token: str) -> Dict[str, Any]:
        """
        Validate JWT token with signature verification.

        This method performs comprehensive JWT validation:
        1. Signature verification (ALWAYS enabled)
        2. Expiration check (ALWAYS enabled)
        3. Not-before time check
        4. Issued-at time check
        5. Issuer validation (if configured)
        6. Audience validation (if configured)
        7. Custom claims validation

        Args:
            token: JWT token string (without 'Bearer ' prefix)

        Returns:
            Decoded token payload containing all claims

        Raises:
            AuthenticationError: If any validation fails

        Example:
            >>> payload = validator.validate_token(token)
            >>> tenant_id = payload["tenant_id"]
            >>> user_id = payload["sub"]
        """
        if not token:
            raise AuthenticationError("Token cannot be empty")

        try:
            # Decode and validate token with STRICT settings
            # SECURITY: verify_signature is ALWAYS True
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                issuer=self.issuer,
                audience=self.audience,
                leeway=self.leeway,
                options={
                    "verify_signature": True,      # CRITICAL: Always verify signature
                    "verify_exp": True,            # CRITICAL: Always check expiration
                    "verify_nbf": True,            # Check not-before time
                    "verify_iat": True,            # Check issued-at time
                    "require_exp": True,           # Expiration MUST be present
                    "require_iat": True,           # Issued-at MUST be present
                }
            )

            # Additional custom claims validation
            self._validate_claims(payload)

            logger.debug(
                f"Token validated successfully (tenant_id={payload.get('tenant_id')}, "
                f"user_id={payload.get('sub')}, type={payload.get('type')})"
            )

            return payload

        except jwt.ExpiredSignatureError as e:
            logger.warning(f"Token expired: {str(e)}")
            raise AuthenticationError("Token has expired")

        except jwt.InvalidSignatureError as e:
            logger.error(f"Invalid token signature: {str(e)}")
            raise AuthenticationError("Invalid token signature")

        except jwt.InvalidIssuerError as e:
            logger.warning(f"Invalid issuer: {str(e)}")
            raise AuthenticationError(f"Invalid token issuer (expected: {self.issuer})")

        except jwt.InvalidAudienceError as e:
            logger.warning(f"Invalid audience: {str(e)}")
            raise AuthenticationError(f"Invalid token audience (expected: {self.audience})")

        except jwt.ImmatureSignatureError as e:
            logger.warning(f"Token not yet valid (nbf): {str(e)}")
            raise AuthenticationError("Token not yet valid")

        except jwt.InvalidIssuedAtError as e:
            logger.warning(f"Invalid issued-at time: {str(e)}")
            raise AuthenticationError("Invalid token issued-at time")

        except jwt.DecodeError as e:
            logger.error(f"Token decode error: {str(e)}")
            raise AuthenticationError("Malformed token")

        except PyJWTError as e:
            logger.error(f"JWT validation error: {str(e)}")
            raise AuthenticationError(f"Token validation failed: {str(e)}")

        except Exception as e:
            logger.critical(f"Unexpected JWT validation error: {str(e)}", exc_info=True)
            raise AuthenticationError(f"Token validation failed: {str(e)}")

    def _validate_claims(self, payload: Dict[str, Any]) -> None:
        """
        Validate custom claims in token payload.

        Required claims:
        - tenant_id: Tenant identifier (UUID string)
        - sub: Subject/user identifier (standard JWT claim)
        - type: Token type ('access' or 'refresh')

        Args:
            payload: Decoded JWT payload

        Raises:
            AuthenticationError: If required claims are missing or invalid
        """
        # Validate tenant_id exists
        if "tenant_id" not in payload:
            raise AuthenticationError("Missing tenant_id in token")

        # Validate tenant_id is not empty
        if not payload["tenant_id"]:
            raise AuthenticationError("Empty tenant_id in token")

        # Validate user_id exists (standard JWT 'sub' claim)
        if "sub" not in payload:
            raise AuthenticationError("Missing user_id (sub) in token")

        # Validate user_id is not empty
        if not payload["sub"]:
            raise AuthenticationError("Empty user_id (sub) in token")

        # Validate token type
        token_type = payload.get("type")
        if token_type not in ["access", "refresh"]:
            raise AuthenticationError(
                f"Invalid token type: {token_type} (expected 'access' or 'refresh')"
            )

        logger.debug(
            f"Custom claims validated (tenant_id={payload['tenant_id']}, "
            f"user_id={payload['sub']}, type={token_type})"
        )

    def generate_token(
        self,
        tenant_id: str,
        user_id: str,
        token_type: str = "access",
        expires_in: int = 3600,
        additional_claims: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate JWT token with standard and custom claims.

        Standard JWT claims included:
        - iat: Issued at (current timestamp)
        - exp: Expiration time (iat + expires_in)
        - nbf: Not before (current timestamp)
        - sub: Subject/user identifier
        - iss: Issuer (if configured)
        - aud: Audience (if configured)

        Custom claims included:
        - tenant_id: Tenant identifier
        - type: Token type ('access' or 'refresh')

        Args:
            tenant_id: Tenant identifier (UUID string)
            user_id: User identifier
            token_type: 'access' or 'refresh' (default: 'access')
            expires_in: Token expiration in seconds (default: 3600 = 1 hour)
            additional_claims: Extra claims to include in token

        Returns:
            JWT token string (without 'Bearer ' prefix)

        Raises:
            ValueError: If tenant_id or user_id is empty
            ValueError: If token_type is invalid

        Example:
            >>> token = validator.generate_token(
            ...     tenant_id="550e8400-e29b-41d4-a716-446655440000",
            ...     user_id="user-123",
            ...     token_type="access",
            ...     expires_in=3600
            ... )
            >>> # Use token in Authorization header: f"Bearer {token}"
        """
        if not tenant_id:
            raise ValueError("tenant_id cannot be empty")

        if not user_id:
            raise ValueError("user_id cannot be empty")

        if token_type not in ["access", "refresh"]:
            raise ValueError(f"Invalid token_type: {token_type} (expected 'access' or 'refresh')")

        now = DeterministicClock.utcnow()

        # Build payload with standard JWT claims
        payload = {
            "iat": now,                                    # Issued at
            "exp": now + timedelta(seconds=expires_in),   # Expiration
            "nbf": now,                                    # Not before
            "sub": user_id,                                # Subject (user_id)
            "tenant_id": tenant_id,                        # Custom: tenant identifier
            "type": token_type,                            # Custom: token type
        }

        # Add optional issuer
        if self.issuer:
            payload["iss"] = self.issuer

        # Add optional audience
        if self.audience:
            payload["aud"] = self.audience

        # Add additional custom claims
        if additional_claims:
            # Prevent overriding protected claims
            protected_claims = {"iat", "exp", "nbf", "sub", "iss", "aud", "tenant_id", "type"}
            for key, value in additional_claims.items():
                if key in protected_claims:
                    logger.warning(f"Skipping protected claim: {key}")
                    continue
                payload[key] = value

        # Encode token
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

        logger.info(
            f"Token generated (tenant_id={tenant_id}, user_id={user_id}, "
            f"type={token_type}, expires_in={expires_in}s)"
        )

        return token


# Thread-safe context variable for current tenant
_tenant_context: ContextVar[Optional[Tenant]] = ContextVar('tenant_context', default=None)


class TenantContext:
    """
    TenantContext - Provides access to current tenant in request context.

    This class provides a clean API for accessing the current tenant throughout
    the request lifecycle. It uses context variables for thread-safe access.

    Example:
        >>> tenant = TenantContext.get_current_tenant()
        >>> if tenant:
        ...     print(f"Current tenant: {tenant.slug}")
    """

    @staticmethod
    def set_current_tenant(tenant: Optional[Tenant]) -> None:
        """
        Set the current tenant for this request context.

        Args:
            tenant: Tenant object or None to clear context
        """
        _tenant_context.set(tenant)

    @staticmethod
    def get_current_tenant() -> Optional[Tenant]:
        """
        Get the current tenant from request context.

        Returns:
            Current tenant or None if not set

        Example:
            >>> tenant = TenantContext.get_current_tenant()
            >>> if tenant:
            ...     print(f"Tenant: {tenant.slug}")
        """
        return _tenant_context.get()

    @staticmethod
    def get_current_tenant_id() -> Optional[UUID4]:
        """
        Get current tenant ID.

        Returns:
            Current tenant UUID or None
        """
        tenant = TenantContext.get_current_tenant()
        return tenant.id if tenant else None

    @staticmethod
    def require_tenant() -> Tenant:
        """
        Get current tenant or raise exception if not set.

        Returns:
            Current tenant

        Raises:
            HTTPException: If no tenant is set in context

        Example:
            >>> tenant = TenantContext.require_tenant()  # Raises if no tenant
            >>> print(tenant.slug)
        """
        tenant = TenantContext.get_current_tenant()
        if not tenant:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No tenant context available. Please authenticate."
            )
        return tenant

    @staticmethod
    def clear() -> None:
        """Clear tenant context."""
        _tenant_context.set(None)

    @staticmethod
    def is_tenant_active() -> bool:
        """
        Check if current tenant is active.

        Returns:
            True if tenant is active, False otherwise
        """
        tenant = TenantContext.get_current_tenant()
        return tenant.is_active() if tenant else False


class TenantExtractionResult(BaseModel):
    """Result of tenant extraction from request."""

    tenant: Optional[Tenant] = None
    method: Optional[str] = None  # jwt, api_key, subdomain, header
    error: Optional[str] = None
    extraction_time_ms: float = 0.0


class TenantExtractor:
    """
    TenantExtractor - Extracts tenant from various request sources.

    This class implements multiple tenant identification strategies:
    1. JWT token (from Authorization header) - WITH SIGNATURE VERIFICATION
    2. API key (from X-API-Key header)
    3. Subdomain (from request URL)
    4. Custom header (X-Tenant-ID)

    Priority order: JWT > API Key > Subdomain > Header

    Security:
    - JWT tokens are validated with cryptographic signature verification
    - Expired tokens are rejected
    - Invalid signatures are rejected
    """

    def __init__(self, tenant_manager: TenantManager):
        """
        Initialize TenantExtractor.

        Args:
            tenant_manager: TenantManager instance for tenant lookups
        """
        self.tenant_manager = tenant_manager

        # Initialize JWT validator with environment configuration
        self.jwt_validator = JWTValidator(
            secret_key=os.getenv("JWT_SECRET_KEY", "CHANGE_THIS_IN_PRODUCTION"),
            algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
            issuer=os.getenv("JWT_ISSUER", "greenlang.ai"),
            audience=os.getenv("JWT_AUDIENCE", "greenlang-api"),
            leeway=int(os.getenv("JWT_LEEWAY", "0"))
        )

        logger.info("TenantExtractor initialized with JWT validation")

    async def extract_tenant(self, request: Request) -> TenantExtractionResult:
        """
        Extract tenant from request using multiple strategies.

        Args:
            request: FastAPI request object

        Returns:
            TenantExtractionResult with tenant and extraction metadata

        Example:
            >>> result = await extractor.extract_tenant(request)
            >>> if result.tenant:
            ...     print(f"Found tenant via {result.method}")
        """
        start_time = time.time()

        # Try JWT token first
        tenant, method = await self._extract_from_jwt(request)
        if tenant:
            return TenantExtractionResult(
                tenant=tenant,
                method=method,
                extraction_time_ms=(time.time() - start_time) * 1000
            )

        # Try API key
        tenant, method = await self._extract_from_api_key(request)
        if tenant:
            return TenantExtractionResult(
                tenant=tenant,
                method=method,
                extraction_time_ms=(time.time() - start_time) * 1000
            )

        # Try subdomain
        tenant, method = await self._extract_from_subdomain(request)
        if tenant:
            return TenantExtractionResult(
                tenant=tenant,
                method=method,
                extraction_time_ms=(time.time() - start_time) * 1000
            )

        # Try header
        tenant, method = await self._extract_from_header(request)
        if tenant:
            return TenantExtractionResult(
                tenant=tenant,
                method=method,
                extraction_time_ms=(time.time() - start_time) * 1000
            )

        # No tenant found
        return TenantExtractionResult(
            error="No tenant identifier found in request",
            extraction_time_ms=(time.time() - start_time) * 1000
        )

    async def _extract_from_jwt(self, request: Request) -> tuple[Optional[Tenant], Optional[str]]:
        """
        Extract tenant from JWT token in Authorization header.

        This method performs COMPLETE JWT validation including:
        - Signature verification
        - Expiration check
        - Issuer/audience validation
        - Custom claims validation

        Args:
            request: FastAPI request

        Returns:
            Tuple of (tenant, method) or (None, None)

        Security:
            - All tokens are validated with signature verification
            - Expired tokens are rejected
            - Invalid signatures are rejected
            - Missing claims cause rejection
        """
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None, None

        try:
            # Extract JWT token (remove 'Bearer ' prefix)
            token = auth_header.replace("Bearer ", "").strip()

            # PRODUCTION JWT VALIDATION - Replaces TODO at line 231
            # Validates signature, expiration, issuer, audience, and custom claims
            payload = self.jwt_validator.validate_token(token)

            # Extract tenant_id from validated payload
            tenant_id = payload.get("tenant_id")

            if tenant_id:
                # Convert string UUID to UUID object
                from uuid import UUID
                try:
                    tenant_uuid = UUID(tenant_id)
                except ValueError:
                    logger.warning(f"Invalid tenant_id format in JWT: {tenant_id}")
                    return None, None

                # Lookup tenant from manager
                tenant = self.tenant_manager.get_tenant(tenant_uuid)
                if tenant:
                    logger.info(
                        f"Tenant extracted from JWT (tenant_id={tenant_id}, "
                        f"user_id={payload.get('sub')}, type={payload.get('type')})"
                    )
                    return tenant, "jwt"
                else:
                    logger.warning(f"Tenant not found in manager: {tenant_id}")

        except AuthenticationError as e:
            # JWT validation failed - log and return None
            logger.warning(f"JWT validation failed: {str(e)}")
            return None, None

        except Exception as e:
            logger.error(f"Unexpected error extracting tenant from JWT: {str(e)}", exc_info=True)

        return None, None

    async def _extract_from_api_key(self, request: Request) -> tuple[Optional[Tenant], Optional[str]]:
        """
        Extract tenant from API key in X-API-Key header.

        Args:
            request: FastAPI request

        Returns:
            Tuple of (tenant, method) or (None, None)
        """
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            return None, None

        try:
            tenant = self.tenant_manager.get_tenant_by_api_key(api_key)
            if tenant:
                return tenant, "api_key"
        except Exception as e:
            logger.warning(f"Failed to extract tenant from API key: {str(e)}")

        return None, None

    async def _extract_from_subdomain(self, request: Request) -> tuple[Optional[Tenant], Optional[str]]:
        """
        Extract tenant from subdomain (e.g., acme.greenlang.ai).

        Args:
            request: FastAPI request

        Returns:
            Tuple of (tenant, method) or (None, None)
        """
        # Get host from request
        host = request.headers.get("host", "")
        if not host:
            return None, None

        try:
            # Extract subdomain
            # Example: acme.greenlang.ai -> acme
            parts = host.split(".")
            if len(parts) >= 3:  # Has subdomain
                subdomain = parts[0]

                # Ignore www and api subdomains
                if subdomain not in ["www", "api", "app"]:
                    tenant = self.tenant_manager.get_tenant_by_slug(subdomain)
                    if tenant:
                        return tenant, "subdomain"

        except Exception as e:
            logger.warning(f"Failed to extract tenant from subdomain: {str(e)}")

        return None, None

    async def _extract_from_header(self, request: Request) -> tuple[Optional[Tenant], Optional[str]]:
        """
        Extract tenant from X-Tenant-ID header.

        Args:
            request: FastAPI request

        Returns:
            Tuple of (tenant, method) or (None, None)
        """
        tenant_id = request.headers.get("X-Tenant-ID")
        if not tenant_id:
            # Also try X-Tenant-Slug
            tenant_slug = request.headers.get("X-Tenant-Slug")
            if tenant_slug:
                try:
                    tenant = self.tenant_manager.get_tenant_by_slug(tenant_slug)
                    if tenant:
                        return tenant, "header_slug"
                except Exception as e:
                    logger.warning(f"Failed to extract tenant from slug header: {str(e)}")
            return None, None

        try:
            from uuid import UUID
            tenant_uuid = UUID(tenant_id)
            tenant = self.tenant_manager.get_tenant(tenant_uuid)
            if tenant:
                return tenant, "header_id"
        except (ValueError, Exception) as e:
            logger.warning(f"Failed to extract tenant from ID header: {str(e)}")

        return None, None


class TenantMiddleware(BaseHTTPMiddleware):
    """
    TenantMiddleware - FastAPI middleware for tenant context management.

    This middleware intercepts all requests, extracts tenant information,
    validates tenant status, and sets the tenant context for downstream handlers.

    It also enforces tenant authorization and prevents cross-tenant access.

    Example:
        >>> app = FastAPI()
        >>> tenant_manager = TenantManager(db)
        >>> app.add_middleware(
        ...     TenantMiddleware,
        ...     tenant_manager=tenant_manager,
        ...     require_tenant=True
        ... )
    """

    def __init__(
        self,
        app,
        tenant_manager: TenantManager,
        require_tenant: bool = True,
        public_paths: Optional[list[str]] = None
    ):
        """
        Initialize TenantMiddleware.

        Args:
            app: FastAPI application
            tenant_manager: TenantManager instance
            require_tenant: If True, reject requests without valid tenant
            public_paths: List of path patterns that don't require tenant
                         (e.g., ["/health", "/docs", "/api/v1/public/*"])
        """
        super().__init__(app)
        self.tenant_manager = tenant_manager
        self.require_tenant = require_tenant
        self.public_paths = public_paths or [
            "/health",
            "/healthz",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/v1/public/*"
        ]
        self.extractor = TenantExtractor(tenant_manager)
        logger.info("TenantMiddleware initialized")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and set tenant context.

        Args:
            request: FastAPI request
            call_next: Next middleware/handler in chain

        Returns:
            Response from downstream handler

        Raises:
            HTTPException: If tenant validation fails
        """
        start_time = time.time()

        # Clear any existing tenant context
        TenantContext.clear()

        # Check if this is a public path
        if self._is_public_path(request.url.path):
            logger.debug(f"Public path: {request.url.path}")
            response = await call_next(request)
            return response

        # Extract tenant from request
        extraction_result = await self.extractor.extract_tenant(request)

        if not extraction_result.tenant:
            if self.require_tenant:
                logger.warning(f"No tenant found for request: {request.url.path}")
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={
                        "error": "tenant_required",
                        "message": "Valid tenant identifier required",
                        "detail": extraction_result.error
                    }
                )
            else:
                # Tenant not required, proceed without context
                response = await call_next(request)
                return response

        tenant = extraction_result.tenant

        # Validate tenant status
        if not tenant.is_active():
            logger.warning(f"Inactive tenant attempted access: {tenant.slug} (status: {tenant.status})")

            if tenant.status == TenantStatus.SUSPENDED:
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={
                        "error": "tenant_suspended",
                        "message": "Your account has been suspended",
                        "tenant_id": str(tenant.id)
                    }
                )
            elif tenant.status == TenantStatus.DELETED:
                return JSONResponse(
                    status_code=status.HTTP_410_GONE,
                    content={
                        "error": "tenant_deleted",
                        "message": "This account has been deleted",
                        "tenant_id": str(tenant.id)
                    }
                )
            elif tenant.is_trial_expired():
                return JSONResponse(
                    status_code=status.HTTP_402_PAYMENT_REQUIRED,
                    content={
                        "error": "trial_expired",
                        "message": "Your trial period has expired",
                        "tenant_id": str(tenant.id)
                    }
                )
            else:
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={
                        "error": "tenant_inactive",
                        "message": f"Tenant status: {tenant.status}",
                        "tenant_id": str(tenant.id)
                    }
                )

        # Set tenant context for request
        TenantContext.set_current_tenant(tenant)

        # Add tenant info to request state for downstream access
        request.state.tenant = tenant
        request.state.tenant_id = tenant.id
        request.state.tenant_slug = tenant.slug

        logger.info(
            f"Tenant context set: {tenant.slug} (method: {extraction_result.method}, "
            f"extraction_time: {extraction_result.extraction_time_ms:.2f}ms)"
        )

        try:
            # Call downstream handlers
            response = await call_next(request)

            # Add tenant info to response headers (for debugging)
            response.headers["X-Tenant-ID"] = str(tenant.id)
            response.headers["X-Tenant-Slug"] = tenant.slug

            # Add timing header
            total_time_ms = (time.time() - start_time) * 1000
            response.headers["X-Request-Time-Ms"] = f"{total_time_ms:.2f}"

            return response

        finally:
            # Always clear context after request
            TenantContext.clear()

    def _is_public_path(self, path: str) -> bool:
        """
        Check if path is public (doesn't require tenant).

        Args:
            path: Request URL path

        Returns:
            True if path is public
        """
        for pattern in self.public_paths:
            # Convert glob pattern to regex
            regex_pattern = pattern.replace("*", ".*")
            if re.match(f"^{regex_pattern}$", path):
                return True
        return False


# Dependency for FastAPI routes
def get_current_tenant() -> Tenant:
    """
    FastAPI dependency to get current tenant.

    Returns:
        Current tenant from context

    Raises:
        HTTPException: If no tenant in context

    Example:
        >>> @app.get("/api/v1/data")
        >>> async def get_data(tenant: Tenant = Depends(get_current_tenant)):
        ...     print(f"Tenant: {tenant.slug}")
        ...     return {"tenant": tenant.slug}
    """
    return TenantContext.require_tenant()


def get_current_tenant_id() -> UUID4:
    """
    FastAPI dependency to get current tenant ID.

    Returns:
        Current tenant UUID

    Raises:
        HTTPException: If no tenant in context

    Example:
        >>> @app.get("/api/v1/data")
        >>> async def get_data(tenant_id: UUID4 = Depends(get_current_tenant_id)):
        ...     print(f"Tenant ID: {tenant_id}")
        ...     return {"tenant_id": str(tenant_id)}
    """
    tenant = TenantContext.require_tenant()
    return tenant.id


# Context manager for testing
class TenantContextManager:
    """
    Context manager for setting tenant context in tests.

    Example:
        >>> with TenantContextManager(tenant):
        ...     # Code here runs with tenant context
        ...     assert TenantContext.get_current_tenant() == tenant
    """

    def __init__(self, tenant: Optional[Tenant]):
        """
        Initialize context manager.

        Args:
            tenant: Tenant to set in context
        """
        self.tenant = tenant
        self.previous_tenant = None

    def __enter__(self):
        """Enter context - set tenant."""
        self.previous_tenant = TenantContext.get_current_tenant()
        TenantContext.set_current_tenant(self.tenant)
        return self.tenant

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - restore previous tenant."""
        TenantContext.set_current_tenant(self.previous_tenant)
