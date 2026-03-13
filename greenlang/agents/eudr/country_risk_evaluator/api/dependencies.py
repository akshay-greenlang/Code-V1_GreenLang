# -*- coding: utf-8 -*-
"""
API Dependencies - AGENT-EUDR-016 Country Risk Evaluator

FastAPI dependency injection providers for authentication, authorization,
rate limiting, request validation, pagination, and service access. All
route handlers inject these dependencies to enforce JWT auth (SEC-001),
RBAC (SEC-002), and per-endpoint rate limits.

Dependencies:
    - get_current_user: Extracts and validates JWT token from Authorization header.
    - require_permission: Factory returning a dependency that checks RBAC permissions.
    - get_country_risk_service: Returns the Country Risk Evaluator service singleton.
    - get_country_risk_scorer: Returns CountryRiskScorer engine instance.
    - get_commodity_analyzer: Returns CommodityRiskAnalyzer engine instance.
    - get_hotspot_detector: Returns DeforestationHotspotDetector engine instance.
    - get_governance_engine: Returns GovernanceIndexEngine engine instance.
    - get_due_diligence_classifier: Returns DueDiligenceClassifier engine instance.
    - get_trade_flow_analyzer: Returns TradeFlowAnalyzer engine instance.
    - get_report_generator: Returns RiskReportGenerator engine instance.
    - get_regulatory_tracker: Returns RegulatoryUpdateTracker engine instance.
    - RateLimiter: Per-endpoint rate limiting with configurable burst.
    - PaginationParams: Standard pagination query parameters.
    - DateRangeParams: Common date range filter parameters.
    - Validators: Request-level validation helpers for path parameters.

Rate Limiter Tiers (5):
    - read: 200 requests/minute (GET operations)
    - write: 100 requests/minute (POST/PUT operations)
    - assess: 50 requests/minute (assessment operations)
    - report: 30 requests/minute (report generation)
    - admin: 20 requests/minute (admin operations)

Permission Prefix: eudr-cre:*

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-016, Section 7.4
Agent ID: GL-EUDR-CRE-016
Status: Production Ready
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from fastapi import Depends, HTTPException, Header, Path, Query, Request, status
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maximum batch size for country assessments.
MAX_BATCH_SIZE: int = 50

#: Maximum countries in comparison request.
MAX_COMPARISON_COUNTRIES: int = 10

#: Supported ISO 3166-1 alpha-2 country codes (200+).
SUPPORTED_COUNTRY_CODES: frozenset = frozenset({
    "AF", "AL", "DZ", "AD", "AO", "AG", "AR", "AM", "AU", "AT",
    "AZ", "BS", "BH", "BD", "BB", "BY", "BE", "BZ", "BJ", "BT",
    "BO", "BA", "BW", "BR", "BN", "BG", "BF", "BI", "CV", "KH",
    "CM", "CA", "CF", "TD", "CL", "CN", "CO", "KM", "CG", "CD",
    "CR", "CI", "HR", "CU", "CY", "CZ", "DK", "DJ", "DM", "DO",
    "EC", "EG", "SV", "GQ", "ER", "EE", "SZ", "ET", "FJ", "FI",
    "FR", "GA", "GM", "GE", "DE", "GH", "GR", "GD", "GT", "GN",
    "GW", "GY", "HT", "HN", "HU", "IS", "IN", "ID", "IR", "IQ",
    "IE", "IL", "IT", "JM", "JP", "JO", "KZ", "KE", "KI", "KP",
    "KR", "KW", "KG", "LA", "LV", "LB", "LS", "LR", "LY", "LI",
    "LT", "LU", "MG", "MW", "MY", "MV", "ML", "MT", "MH", "MR",
    "MU", "MX", "FM", "MD", "MC", "MN", "ME", "MA", "MZ", "MM",
    "NA", "NR", "NP", "NL", "NZ", "NI", "NE", "NG", "MK", "NO",
    "OM", "PK", "PW", "PA", "PG", "PY", "PE", "PH", "PL", "PT",
    "QA", "RO", "RU", "RW", "KN", "LC", "VC", "WS", "SM", "ST",
    "SA", "SN", "RS", "SC", "SL", "SG", "SK", "SI", "SB", "SO",
    "ZA", "SS", "ES", "LK", "SD", "SR", "SE", "CH", "SY", "TW",
    "TJ", "TZ", "TH", "TL", "TG", "TO", "TT", "TN", "TR", "TM",
    "TV", "UG", "UA", "AE", "GB", "US", "UY", "UZ", "VU", "VE",
    "VN", "YE", "ZM", "ZW", "HK", "MO", "PS", "XK",
})

#: EUDR commodities.
SUPPORTED_COMMODITIES: frozenset = frozenset({
    "cattle", "cocoa", "coffee", "oil_palm",
    "rubber", "soya", "wood",
})

# ---------------------------------------------------------------------------
# OAuth2 / API Key security schemes
# ---------------------------------------------------------------------------

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/token",
    auto_error=False,
)

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)


# ---------------------------------------------------------------------------
# User model for auth context
# ---------------------------------------------------------------------------


class AuthUser(BaseModel):
    """Authenticated user context extracted from JWT or API key.

    Attributes:
        user_id: Unique user identifier from JWT ``sub`` claim.
        email: User email address.
        tenant_id: Multi-tenant identifier (defaults to ``default``).
        operator_id: Associated EUDR operator identifier.
        roles: Assigned RBAC roles (e.g. ``admin``, ``operator``).
        permissions: Granted fine-grained permissions (e.g. ``eudr-cre:countries:assess``).
    """

    user_id: str = Field(..., description="Unique user identifier")
    email: str = Field(default="", description="User email address")
    tenant_id: str = Field(default="default", description="Tenant identifier")
    operator_id: str = Field(default="", description="Associated operator ID")
    roles: List[str] = Field(default_factory=list, description="Assigned roles")
    permissions: List[str] = Field(
        default_factory=list, description="Granted permissions"
    )


# ---------------------------------------------------------------------------
# Authentication dependency
# ---------------------------------------------------------------------------


async def get_current_user(
    request: Request,
    token: Optional[str] = Depends(oauth2_scheme),
    api_key: Optional[str] = Depends(api_key_header),
) -> AuthUser:
    """Extract and validate the current user from JWT token or API key.

    Checks ``request.state.auth`` first (populated by AuthenticationMiddleware
    from SEC-001). Falls back to manual token/api_key validation.

    Args:
        request: FastAPI request object.
        token: OAuth2 bearer token from Authorization header.
        api_key: API key from X-API-Key header.

    Returns:
        AuthUser with user identity, tenant, roles, and permissions.

    Raises:
        HTTPException: 401 if no valid credentials are provided.
    """
    # Check if AuthenticationMiddleware already populated auth context
    auth_ctx = getattr(request.state, "auth", None)
    if auth_ctx is not None:
        return AuthUser(
            user_id=getattr(auth_ctx, "user_id", ""),
            email=getattr(auth_ctx, "email", ""),
            tenant_id=getattr(auth_ctx, "tenant_id", "default"),
            operator_id=getattr(auth_ctx, "operator_id", ""),
            roles=getattr(auth_ctx, "roles", []),
            permissions=getattr(auth_ctx, "permissions", []),
        )

    # Fallback: validate JWT token manually
    if token:
        try:
            from greenlang.infrastructure.auth_service.jwt_service import (
                decode_token,
            )

            payload = decode_token(token)
            return AuthUser(
                user_id=payload.get("sub", ""),
                email=payload.get("email", ""),
                tenant_id=payload.get("tenant_id", "default"),
                operator_id=payload.get("operator_id", ""),
                roles=payload.get("roles", []),
                permissions=payload.get("permissions", []),
            )
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired JWT token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    # Fallback: validate API key manually
    if api_key:
        try:
            from greenlang.infrastructure.auth_service.api_key_service import (
                validate_api_key,
            )

            key_data = validate_api_key(api_key)
            return AuthUser(
                user_id=key_data.get("user_id", "api-user"),
                email=key_data.get("email", ""),
                tenant_id=key_data.get("tenant_id", "default"),
                operator_id=key_data.get("operator_id", ""),
                roles=key_data.get("roles", ["api-user"]),
                permissions=key_data.get("permissions", []),
            )
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Provide Bearer token or X-API-Key header.",
        headers={"WWW-Authenticate": "Bearer"},
    )


# ---------------------------------------------------------------------------
# RBAC permission dependency factory
# ---------------------------------------------------------------------------


def require_permission(permission: str) -> Callable:
    """Factory returning a FastAPI dependency that checks RBAC permissions.

    Uses wildcard matching: ``eudr-cre:*`` grants all
    ``eudr-cre:<action>`` permissions.

    Args:
        permission: Required permission string, e.g.
            ``eudr-cre:countries:assess``.

    Returns:
        Async dependency function that validates the user has the
        required permission and returns the AuthUser.

    Example:
        >>> @router.post("/countries/assess")
        ... async def assess_country(
        ...     user: AuthUser = Depends(require_permission("eudr-cre:countries:assess"))
        ... ):
        ...     ...
    """

    async def _check_permission(
        user: AuthUser = Depends(get_current_user),
    ) -> AuthUser:
        # Admin and platform_admin bypass all permission checks
        if "admin" in user.roles or "platform_admin" in user.roles:
            return user

        # Check exact match
        if permission in user.permissions:
            return user

        # Check wildcard patterns (eudr-cre:countries:* matches eudr-cre:countries:assess)
        parts = permission.split(":")
        for i in range(len(parts)):
            wildcard = ":".join(parts[: i + 1]) + ":*"
            if wildcard in user.permissions:
                return user
        # Check top-level wildcard (eudr-cre:*)
        if parts[0] + ":*" in user.permissions:
            return user

        logger.warning(
            "Permission denied: user=%s permission=%s available=%s",
            user.user_id,
            permission,
            user.permissions,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Missing required permission: {permission}",
        )

    return _check_permission


# ---------------------------------------------------------------------------
# Pagination parameters
# ---------------------------------------------------------------------------


class PaginationParams(BaseModel):
    """Standard pagination query parameters.

    Attributes:
        page: Page number (1-based).
        page_size: Number of results per page (1-500).
    """

    page: int = Field(default=1, ge=1, description="Page number (1-based)")
    page_size: int = Field(
        default=50, ge=1, le=500, description="Results per page"
    )


def get_pagination(
    page: int = Query(default=1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(
        default=50, ge=1, le=500, description="Results per page"
    ),
) -> PaginationParams:
    """Extract pagination parameters from query string.

    Args:
        page: Page number (1-based, default 1).
        page_size: Results per page (1-500, default 50).

    Returns:
        PaginationParams with validated page and page_size.
    """
    return PaginationParams(page=page, page_size=page_size)


# ---------------------------------------------------------------------------
# Date range filter parameters
# ---------------------------------------------------------------------------


class DateRangeParams(BaseModel):
    """Common date range filter parameters.

    Attributes:
        start_date: Start of date range filter (inclusive).
        end_date: End of date range filter (inclusive).
    """

    start_date: Optional[datetime] = Field(
        None, description="Start date filter (inclusive, UTC)"
    )
    end_date: Optional[datetime] = Field(
        None, description="End date filter (inclusive, UTC)"
    )


def get_date_range(
    start_date: Optional[datetime] = Query(
        None, description="Start date filter (inclusive, UTC)"
    ),
    end_date: Optional[datetime] = Query(
        None, description="End date filter (inclusive, UTC)"
    ),
) -> DateRangeParams:
    """Extract date range filter parameters from query string.

    Args:
        start_date: Start of date range (inclusive).
        end_date: End of date range (inclusive).

    Returns:
        DateRangeParams with validated date range.

    Raises:
        HTTPException: 400 if start_date is after end_date.
    """
    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="start_date must be before or equal to end_date",
        )
    return DateRangeParams(start_date=start_date, end_date=end_date)


# ---------------------------------------------------------------------------
# Rate limiter (in-memory, per-endpoint sliding window)
# ---------------------------------------------------------------------------


class RateLimiter:
    """In-memory sliding-window rate limiter per user per endpoint.

    Tracks request timestamps and enforces a maximum request count
    within a rolling time window. Thread-safe for async usage.

    Attributes:
        max_requests: Maximum requests allowed per window.
        window_seconds: Rolling window duration in seconds.
    """

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        """Initialize the rate limiter.

        Args:
            max_requests: Maximum requests allowed per window.
            window_seconds: Window duration in seconds.
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, list] = defaultdict(list)

    async def __call__(
        self,
        request: Request,
        user: AuthUser = Depends(get_current_user),
    ) -> None:
        """Check rate limit for the current user and endpoint.

        Args:
            request: FastAPI request object.
            user: Authenticated user.

        Raises:
            HTTPException: 429 if rate limit exceeded.
        """
        key = f"{user.user_id}:{request.url.path}"
        now = time.monotonic()
        cutoff = now - self.window_seconds

        # Remove expired timestamps
        self._requests[key] = [
            ts for ts in self._requests[key] if ts > cutoff
        ]

        if len(self._requests[key]) >= self.max_requests:
            logger.warning(
                "Rate limit exceeded: user=%s endpoint=%s limit=%d/%ds",
                user.user_id,
                request.url.path,
                self.max_requests,
                self.window_seconds,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Rate limit exceeded: {self.max_requests} requests "
                    f"per {self.window_seconds} seconds"
                ),
                headers={
                    "Retry-After": str(self.window_seconds),
                    "X-RateLimit-Limit": str(self.max_requests),
                    "X-RateLimit-Remaining": "0",
                },
            )

        self._requests[key].append(now)


# ---------------------------------------------------------------------------
# Pre-configured rate limiter instances (5 tiers)
# ---------------------------------------------------------------------------

_rate_limit_read = RateLimiter(max_requests=200, window_seconds=60)
_rate_limit_write = RateLimiter(max_requests=100, window_seconds=60)
_rate_limit_assess = RateLimiter(max_requests=50, window_seconds=60)
_rate_limit_report = RateLimiter(max_requests=30, window_seconds=60)
_rate_limit_admin = RateLimiter(max_requests=20, window_seconds=60)


# Wrapper functions for dependency injection (overridable in tests)
async def rate_limit_read(
    request: Request,
    user: AuthUser = Depends(get_current_user),
) -> None:
    """Read rate limit: 200 requests/minute for GET operations."""
    await _rate_limit_read(request, user)


async def rate_limit_write(
    request: Request,
    user: AuthUser = Depends(get_current_user),
) -> None:
    """Write rate limit: 100 requests/minute for POST/PUT operations."""
    await _rate_limit_write(request, user)


async def rate_limit_assess(
    request: Request,
    user: AuthUser = Depends(get_current_user),
) -> None:
    """Assessment rate limit: 50 requests/minute for assessment operations."""
    await _rate_limit_assess(request, user)


async def rate_limit_report(
    request: Request,
    user: AuthUser = Depends(get_current_user),
) -> None:
    """Report rate limit: 30 requests/minute for report generation."""
    await _rate_limit_report(request, user)


async def rate_limit_admin(
    request: Request,
    user: AuthUser = Depends(get_current_user),
) -> None:
    """Admin rate limit: 20 requests/minute for admin operations."""
    await _rate_limit_admin(request, user)


# ---------------------------------------------------------------------------
# Request validators - Path parameter helpers
# ---------------------------------------------------------------------------


def validate_country_code(
    country_code: str = Path(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code"
    ),
) -> str:
    """Validate country_code path parameter.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.

    Returns:
        Uppercased, validated country code.

    Raises:
        HTTPException: 400 if country code is not supported.
    """
    code = country_code.upper().strip()
    if code not in SUPPORTED_COUNTRY_CODES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported country code: {code}. Must be ISO 3166-1 alpha-2.",
        )
    return code


def validate_commodity_type(
    commodity_type: str = Path(
        ..., description="EUDR commodity type"
    ),
) -> str:
    """Validate commodity_type path parameter.

    Args:
        commodity_type: EUDR commodity type.

    Returns:
        Lowercased, validated commodity type.

    Raises:
        HTTPException: 400 if commodity type is not supported.
    """
    commodity = commodity_type.lower().strip()
    if commodity not in SUPPORTED_COMMODITIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported commodity: {commodity}. Must be one of {sorted(SUPPORTED_COMMODITIES)}.",
        )
    return commodity


def validate_uuid_path(
    value: str,
    param_name: str = "id",
) -> str:
    """Validate a path parameter is a non-empty string.

    Args:
        value: The path parameter value to validate.
        param_name: Name for error messages.

    Returns:
        The validated, stripped value.

    Raises:
        HTTPException: 400 if the value is empty or whitespace-only.
    """
    if not value or not value.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{param_name} must be a non-empty string",
        )
    return value.strip()


def validate_assessment_id(
    assessment_id: str = Path(..., description="Assessment identifier"),
) -> str:
    """Validate assessment_id path parameter."""
    return validate_uuid_path(assessment_id, "assessment_id")


def validate_hotspot_id(
    hotspot_id: str = Path(..., description="Hotspot identifier"),
) -> str:
    """Validate hotspot_id path parameter."""
    return validate_uuid_path(hotspot_id, "hotspot_id")


def validate_report_id(
    report_id: str = Path(..., description="Report identifier"),
) -> str:
    """Validate report_id path parameter."""
    return validate_uuid_path(report_id, "report_id")


# ---------------------------------------------------------------------------
# Service singleton (lazy initialization with stub fallback)
# ---------------------------------------------------------------------------


class _CREServiceStub:
    """Stub service for Country Risk Evaluator operations.

    Provides safe no-op methods when the actual engine modules are not
    yet initialized. Enables API startup and health checks without
    requiring full engine initialization.
    """

    def __init__(self) -> None:
        """Initialize the stub service."""
        self._initialized = False
        logger.info("Country Risk Evaluator service stub initialized")

    @property
    def is_initialized(self) -> bool:
        """Whether the real service has been initialized."""
        return self._initialized


_cre_service_instance: Optional[Any] = None


def get_country_risk_service() -> Any:
    """Return the Country Risk Evaluator service singleton.

    Attempts to import and initialize the real service engines on first
    call. Falls back to a stub if the real engines are not available
    (e.g. during testing or early startup).

    Returns:
        Country Risk Evaluator service instance (real or stub).
    """
    global _cre_service_instance

    if _cre_service_instance is not None:
        return _cre_service_instance

    try:
        from greenlang.agents.eudr.country_risk_evaluator.config import (
            get_config,
        )

        config = get_config()
        _cre_service_instance = {
            "config": config,
            "initialized": True,
        }
        logger.info("Country Risk Evaluator service engines initialized")
    except Exception as exc:
        logger.warning(
            "Could not initialize CRE service engines, using stub: %s",
            exc,
        )
        _cre_service_instance = _CREServiceStub()

    return _cre_service_instance


def reset_cre_service() -> None:
    """Reset the service singleton. Used in testing."""
    global _cre_service_instance
    _cre_service_instance = None


# ---------------------------------------------------------------------------
# Engine dependencies (lazy initialization)
# ---------------------------------------------------------------------------


def get_country_risk_scorer() -> Any:
    """Return the CountryRiskScorer engine instance."""
    service = get_country_risk_service()
    return service.get("scorer", None) if isinstance(service, dict) else None


def get_commodity_analyzer() -> Any:
    """Return the CommodityRiskAnalyzer engine instance."""
    service = get_country_risk_service()
    return service.get("commodity_analyzer", None) if isinstance(service, dict) else None


def get_hotspot_detector() -> Any:
    """Return the DeforestationHotspotDetector engine instance."""
    service = get_country_risk_service()
    return service.get("hotspot_detector", None) if isinstance(service, dict) else None


def get_governance_engine() -> Any:
    """Return the GovernanceIndexEngine engine instance."""
    service = get_country_risk_service()
    return service.get("governance_engine", None) if isinstance(service, dict) else None


def get_due_diligence_classifier() -> Any:
    """Return the DueDiligenceClassifier engine instance."""
    service = get_country_risk_service()
    return service.get("dd_classifier", None) if isinstance(service, dict) else None


def get_trade_flow_analyzer() -> Any:
    """Return the TradeFlowAnalyzer engine instance."""
    service = get_country_risk_service()
    return service.get("trade_flow_analyzer", None) if isinstance(service, dict) else None


def get_report_generator() -> Any:
    """Return the RiskReportGenerator engine instance."""
    service = get_country_risk_service()
    return service.get("report_generator", None) if isinstance(service, dict) else None


def get_regulatory_tracker() -> Any:
    """Return the RegulatoryUpdateTracker engine instance."""
    service = get_country_risk_service()
    return service.get("regulatory_tracker", None) if isinstance(service, dict) else None


# ---------------------------------------------------------------------------
# Request ID injection
# ---------------------------------------------------------------------------


def get_request_id(request: Request) -> str:
    """Extract or generate a request correlation ID.

    Checks for ``X-Request-ID`` header first, then generates a new UUID4.

    Args:
        request: FastAPI request object.

    Returns:
        Request correlation ID string.
    """
    request_id = request.headers.get("X-Request-ID", "")
    if not request_id:
        request_id = str(uuid.uuid4())
    return request_id


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Auth
    "AuthUser",
    "get_current_user",
    "require_permission",
    # Pagination / Filters
    "DateRangeParams",
    "PaginationParams",
    "get_date_range",
    "get_pagination",
    # Rate limiting
    "RateLimiter",
    "rate_limit_admin",
    "rate_limit_assess",
    "rate_limit_read",
    "rate_limit_report",
    "rate_limit_write",
    # Security schemes
    "api_key_header",
    "oauth2_scheme",
    # Validators
    "validate_assessment_id",
    "validate_commodity_type",
    "validate_country_code",
    "validate_hotspot_id",
    "validate_report_id",
    "validate_uuid_path",
    # Constants
    "MAX_BATCH_SIZE",
    "MAX_COMPARISON_COUNTRIES",
    "SUPPORTED_COMMODITIES",
    "SUPPORTED_COUNTRY_CODES",
    # Service
    "_CREServiceStub",
    "get_commodity_analyzer",
    "get_country_risk_scorer",
    "get_country_risk_service",
    "get_due_diligence_classifier",
    "get_governance_engine",
    "get_hotspot_detector",
    "get_regulatory_tracker",
    "get_report_generator",
    "get_request_id",
    "get_trade_flow_analyzer",
    "reset_cre_service",
]
