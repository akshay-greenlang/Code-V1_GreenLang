# -*- coding: utf-8 -*-
"""
API Dependencies - AGENT-EUDR-018 Commodity Risk Analyzer

FastAPI dependency injection providers for authentication, authorization,
rate limiting, engine access, and common query parameter validation. All
route handlers inject these dependencies to enforce JWT auth (SEC-001),
RBAC (SEC-002), and per-endpoint rate limits.

Dependencies:
    - get_current_user: Extracts and validates JWT token from Authorization header.
    - require_permission: Factory returning a dependency that checks RBAC permissions.
    - get_commodity_risk_analyzer_setup: Returns the CommodityRiskAnalyzerConfig singleton.
    - get_commodity_profiler: Returns the CommodityProfiler engine singleton.
    - get_derived_product_analyzer: Returns the DerivedProductAnalyzer engine singleton.
    - get_price_volatility_engine: Returns the PriceVolatilityEngine singleton.
    - get_production_forecast_engine: Returns the ProductionForecastEngine singleton.
    - get_substitution_risk_analyzer: Returns the SubstitutionRiskAnalyzer singleton.
    - get_regulatory_compliance_engine: Returns the RegulatoryComplianceEngine singleton.
    - get_commodity_dd_engine: Returns the CommodityDueDiligenceEngine singleton.
    - get_portfolio_risk_aggregator: Returns the PortfolioRiskAggregator singleton.
    - RateLimiter: Per-endpoint rate limiting with configurable burst.
    - PaginationParams: Standard pagination query parameters.
    - validate_commodity_type: Validates commodity type query parameter.
    - validate_date_range: Validates start/end date query parameters.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-018 Commodity Risk Analyzer (GL-EUDR-CRA-018)
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional

from fastapi import Depends, HTTPException, Query, Request, status
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from pydantic import Field
from greenlang.schemas import GreenLangBase

logger = logging.getLogger(__name__)

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


class AuthUser(GreenLangBase):
    """Authenticated user context extracted from JWT or API key."""

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

    Checks request.state.auth first (populated by AuthenticationMiddleware
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

    # Fallback: validate token manually
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

    # Fallback: validate API key
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

    Uses wildcard matching: ``eudr-commodity-risk:*`` grants all
    ``eudr-commodity-risk:<action>`` permissions.

    Args:
        permission: Required permission string, e.g.
            ``eudr-commodity-risk:commodities:read``.

    Returns:
        Async dependency function that validates the user has the
        required permission and returns the AuthUser.

    Example:
        >>> @router.get("/commodities")
        ... async def list_commodities(
        ...     user: AuthUser = Depends(require_permission("eudr-commodity-risk:commodities:read"))
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

        # Check wildcard patterns
        parts = permission.split(":")
        for i in range(len(parts)):
            wildcard = ":".join(parts[: i + 1]) + ":*"
            if wildcard in user.permissions:
                return user
        # Check top-level wildcard
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


class PaginationParams(GreenLangBase):
    """Standard pagination query parameters."""

    limit: int = Field(default=50, ge=1, le=1000, description="Results per page")
    offset: int = Field(default=0, ge=0, description="Number of results to skip")


def get_pagination(
    limit: int = Query(default=50, ge=1, le=1000, description="Results per page"),
    offset: int = Query(default=0, ge=0, description="Number of results to skip"),
) -> PaginationParams:
    """Extract pagination parameters from query string.

    Args:
        limit: Maximum number of results to return (1-1000).
        offset: Number of results to skip for pagination.

    Returns:
        PaginationParams with validated limit and offset.
    """
    return PaginationParams(limit=limit, offset=offset)


# ---------------------------------------------------------------------------
# Rate limiter (in-memory, per-endpoint)
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
                "Rate limit exceeded: user=%s endpoint=%s",
                user.user_id,
                request.url.path,
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


# Pre-configured rate limiter instances
_rate_limit_standard = RateLimiter(max_requests=100, window_seconds=60)
_rate_limit_write = RateLimiter(max_requests=30, window_seconds=60)
_rate_limit_heavy = RateLimiter(max_requests=10, window_seconds=60)
_rate_limit_export = RateLimiter(max_requests=5, window_seconds=60)


# Wrapper functions for dependency injection (overridable in tests)
async def rate_limit_standard(
    request: Request,
    user: AuthUser = Depends(get_current_user),
) -> None:
    """Standard rate limit: 100 requests/minute."""
    await _rate_limit_standard(request, user)


async def rate_limit_write(
    request: Request,
    user: AuthUser = Depends(get_current_user),
) -> None:
    """Write rate limit: 30 requests/minute."""
    await _rate_limit_write(request, user)


async def rate_limit_heavy(
    request: Request,
    user: AuthUser = Depends(get_current_user),
) -> None:
    """Heavy operation rate limit: 10 requests/minute."""
    await _rate_limit_heavy(request, user)


async def rate_limit_export(
    request: Request,
    user: AuthUser = Depends(get_current_user),
) -> None:
    """Export rate limit: 5 requests/minute."""
    await _rate_limit_export(request, user)


# ---------------------------------------------------------------------------
# Error response model
# ---------------------------------------------------------------------------


class ErrorResponse(GreenLangBase):
    """Structured error response for all API endpoints."""

    error: str = Field(..., description="Error type identifier")
    message: str = Field(..., description="Human-readable error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request correlation ID")


# ---------------------------------------------------------------------------
# Success envelope response
# ---------------------------------------------------------------------------


class SuccessResponse(GreenLangBase):
    """Standard success response wrapper."""

    status: str = Field(default="success", description="Response status")
    message: str = Field(default="", description="Response message")
    data: Optional[Any] = Field(None, description="Response payload")


# ---------------------------------------------------------------------------
# Common query parameter validators
# ---------------------------------------------------------------------------

# Valid EUDR commodity types
_VALID_COMMODITY_TYPES = {
    "cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood"
}


def validate_commodity_type(
    commodity_type: Optional[str] = Query(
        None,
        description="EUDR commodity type filter (cattle, cocoa, coffee, oil_palm, rubber, soya, wood)",
    ),
) -> Optional[str]:
    """Validate and normalize a commodity type query parameter.

    Args:
        commodity_type: Optional commodity type string.

    Returns:
        Normalized commodity type or None.

    Raises:
        HTTPException: 400 if commodity type is invalid.
    """
    if commodity_type is None:
        return None
    normalized = commodity_type.strip().lower()
    if normalized not in _VALID_COMMODITY_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid commodity type: {commodity_type}. "
                f"Must be one of: {', '.join(sorted(_VALID_COMMODITY_TYPES))}"
            ),
        )
    return normalized


def validate_date_range(
    start_date: Optional[date] = Query(
        None,
        description="Start date for date range filter (YYYY-MM-DD)",
    ),
    end_date: Optional[date] = Query(
        None,
        description="End date for date range filter (YYYY-MM-DD)",
    ),
) -> Dict[str, Optional[date]]:
    """Validate that start_date <= end_date when both provided.

    Args:
        start_date: Optional start date.
        end_date: Optional end date.

    Returns:
        Dictionary with 'start_date' and 'end_date' keys.

    Raises:
        HTTPException: 400 if start_date > end_date.
    """
    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"start_date ({start_date}) must be <= end_date ({end_date})"
            ),
        )
    return {"start_date": start_date, "end_date": end_date}


# ---------------------------------------------------------------------------
# Engine singleton accessors
# ---------------------------------------------------------------------------

# Global singletons (lazily initialized)
_commodity_risk_analyzer_config: Optional[Any] = None
_commodity_profiler: Optional[Any] = None
_derived_product_analyzer: Optional[Any] = None
_price_volatility_engine: Optional[Any] = None
_production_forecast_engine: Optional[Any] = None
_substitution_risk_analyzer: Optional[Any] = None
_regulatory_compliance_engine: Optional[Any] = None
_commodity_dd_engine: Optional[Any] = None
_portfolio_risk_aggregator: Optional[Any] = None


def get_commodity_risk_analyzer_setup() -> Any:
    """Return the CommodityRiskAnalyzerConfig singleton.

    Lazily initializes the configuration on first access using
    the ``get_config()`` factory from the config module.

    Returns:
        CommodityRiskAnalyzerConfig instance.

    Raises:
        HTTPException: 503 if configuration cannot be loaded.
    """
    global _commodity_risk_analyzer_config
    if _commodity_risk_analyzer_config is None:
        try:
            from greenlang.agents.eudr.commodity_risk_analyzer.config import (
                get_config,
            )

            _commodity_risk_analyzer_config = get_config()
            logger.info("CommodityRiskAnalyzerConfig initialized")
        except Exception as exc:
            logger.error(
                "Failed to initialize CommodityRiskAnalyzerConfig: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Commodity Risk Analyzer configuration unavailable",
            )
    return _commodity_risk_analyzer_config


def get_commodity_profiler() -> Any:
    """Return the CommodityProfiler engine singleton.

    Lazily initializes the engine on first access.

    Returns:
        CommodityProfiler engine instance.

    Raises:
        HTTPException: 503 if engine cannot be initialized.
    """
    global _commodity_profiler
    if _commodity_profiler is None:
        try:
            from greenlang.agents.eudr.commodity_risk_analyzer.engines.commodity_profiler import (
                CommodityProfiler,
            )

            config = get_commodity_risk_analyzer_setup()
            _commodity_profiler = CommodityProfiler(config)
            logger.info("CommodityProfiler engine initialized")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to initialize CommodityProfiler: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="CommodityProfiler engine unavailable",
            )
    return _commodity_profiler


def get_derived_product_analyzer() -> Any:
    """Return the DerivedProductAnalyzer engine singleton.

    Lazily initializes the engine on first access.

    Returns:
        DerivedProductAnalyzer engine instance.

    Raises:
        HTTPException: 503 if engine cannot be initialized.
    """
    global _derived_product_analyzer
    if _derived_product_analyzer is None:
        try:
            from greenlang.agents.eudr.commodity_risk_analyzer.engines.derived_product_analyzer import (
                DerivedProductAnalyzer,
            )

            config = get_commodity_risk_analyzer_setup()
            _derived_product_analyzer = DerivedProductAnalyzer(config)
            logger.info("DerivedProductAnalyzer engine initialized")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to initialize DerivedProductAnalyzer: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="DerivedProductAnalyzer engine unavailable",
            )
    return _derived_product_analyzer


def get_price_volatility_engine() -> Any:
    """Return the PriceVolatilityEngine singleton.

    Lazily initializes the engine on first access.

    Returns:
        PriceVolatilityEngine instance.

    Raises:
        HTTPException: 503 if engine cannot be initialized.
    """
    global _price_volatility_engine
    if _price_volatility_engine is None:
        try:
            from greenlang.agents.eudr.commodity_risk_analyzer.engines.price_volatility_engine import (
                PriceVolatilityEngine,
            )

            config = get_commodity_risk_analyzer_setup()
            _price_volatility_engine = PriceVolatilityEngine(config)
            logger.info("PriceVolatilityEngine initialized")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to initialize PriceVolatilityEngine: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="PriceVolatilityEngine unavailable",
            )
    return _price_volatility_engine


def get_production_forecast_engine() -> Any:
    """Return the ProductionForecastEngine singleton.

    Lazily initializes the engine on first access.

    Returns:
        ProductionForecastEngine instance.

    Raises:
        HTTPException: 503 if engine cannot be initialized.
    """
    global _production_forecast_engine
    if _production_forecast_engine is None:
        try:
            from greenlang.agents.eudr.commodity_risk_analyzer.engines.production_forecast_engine import (
                ProductionForecastEngine,
            )

            config = get_commodity_risk_analyzer_setup()
            _production_forecast_engine = ProductionForecastEngine(config)
            logger.info("ProductionForecastEngine initialized")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to initialize ProductionForecastEngine: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ProductionForecastEngine unavailable",
            )
    return _production_forecast_engine


def get_substitution_risk_analyzer() -> Any:
    """Return the SubstitutionRiskAnalyzer engine singleton.

    Lazily initializes the engine on first access.

    Returns:
        SubstitutionRiskAnalyzer engine instance.

    Raises:
        HTTPException: 503 if engine cannot be initialized.
    """
    global _substitution_risk_analyzer
    if _substitution_risk_analyzer is None:
        try:
            from greenlang.agents.eudr.commodity_risk_analyzer.engines.substitution_risk_analyzer import (
                SubstitutionRiskAnalyzer,
            )

            config = get_commodity_risk_analyzer_setup()
            _substitution_risk_analyzer = SubstitutionRiskAnalyzer(config)
            logger.info("SubstitutionRiskAnalyzer engine initialized")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to initialize SubstitutionRiskAnalyzer: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="SubstitutionRiskAnalyzer engine unavailable",
            )
    return _substitution_risk_analyzer


def get_regulatory_compliance_engine() -> Any:
    """Return the RegulatoryComplianceEngine singleton.

    Lazily initializes the engine on first access.

    Returns:
        RegulatoryComplianceEngine instance.

    Raises:
        HTTPException: 503 if engine cannot be initialized.
    """
    global _regulatory_compliance_engine
    if _regulatory_compliance_engine is None:
        try:
            from greenlang.agents.eudr.commodity_risk_analyzer.engines.regulatory_compliance_engine import (
                RegulatoryComplianceEngine,
            )

            config = get_commodity_risk_analyzer_setup()
            _regulatory_compliance_engine = RegulatoryComplianceEngine(config)
            logger.info("RegulatoryComplianceEngine initialized")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to initialize RegulatoryComplianceEngine: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RegulatoryComplianceEngine unavailable",
            )
    return _regulatory_compliance_engine


def get_commodity_dd_engine() -> Any:
    """Return the CommodityDueDiligenceEngine singleton.

    Lazily initializes the engine on first access.

    Returns:
        CommodityDueDiligenceEngine instance.

    Raises:
        HTTPException: 503 if engine cannot be initialized.
    """
    global _commodity_dd_engine
    if _commodity_dd_engine is None:
        try:
            from greenlang.agents.eudr.commodity_risk_analyzer.engines.commodity_due_diligence_engine import (
                CommodityDueDiligenceEngine,
            )

            config = get_commodity_risk_analyzer_setup()
            _commodity_dd_engine = CommodityDueDiligenceEngine(config)
            logger.info("CommodityDueDiligenceEngine initialized")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to initialize CommodityDueDiligenceEngine: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="CommodityDueDiligenceEngine unavailable",
            )
    return _commodity_dd_engine


def get_portfolio_risk_aggregator() -> Any:
    """Return the PortfolioRiskAggregator engine singleton.

    Lazily initializes the engine on first access.

    Returns:
        PortfolioRiskAggregator engine instance.

    Raises:
        HTTPException: 503 if engine cannot be initialized.
    """
    global _portfolio_risk_aggregator
    if _portfolio_risk_aggregator is None:
        try:
            from greenlang.agents.eudr.commodity_risk_analyzer.engines.portfolio_risk_aggregator import (
                PortfolioRiskAggregator,
            )

            config = get_commodity_risk_analyzer_setup()
            _portfolio_risk_aggregator = PortfolioRiskAggregator(config)
            logger.info("PortfolioRiskAggregator engine initialized")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to initialize PortfolioRiskAggregator: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="PortfolioRiskAggregator engine unavailable",
            )
    return _portfolio_risk_aggregator


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "AuthUser",
    "ErrorResponse",
    "PaginationParams",
    "RateLimiter",
    "SuccessResponse",
    "api_key_header",
    "get_commodity_dd_engine",
    "get_commodity_profiler",
    "get_commodity_risk_analyzer_setup",
    "get_current_user",
    "get_derived_product_analyzer",
    "get_pagination",
    "get_portfolio_risk_aggregator",
    "get_price_volatility_engine",
    "get_production_forecast_engine",
    "get_regulatory_compliance_engine",
    "get_substitution_risk_analyzer",
    "oauth2_scheme",
    "rate_limit_export",
    "rate_limit_heavy",
    "rate_limit_standard",
    "rate_limit_write",
    "require_permission",
    "validate_commodity_type",
    "validate_date_range",
]
