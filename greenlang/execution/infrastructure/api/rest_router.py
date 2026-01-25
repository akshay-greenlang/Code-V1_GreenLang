"""
REST Router with Versioning for GreenLang

This module provides a FastAPI router with API versioning,
standard middleware, and GreenLang-specific extensions.

Features:
- API versioning (URL path, header, query param)
- Standard error responses
- Request/response logging
- Correlation ID tracking
- Pagination helpers
- Rate limiting integration

Example:
    >>> router = RESTRouter(version="v1")
    >>> @router.get("/emissions")
    >>> async def list_emissions(): ...
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field

try:
    from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
    from fastapi.responses import JSONResponse
    from starlette.middleware.base import BaseHTTPMiddleware
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = object
    Depends = None
    HTTPException = Exception
    Query = None
    Request = None
    Response = None
    JSONResponse = None

logger = logging.getLogger(__name__)

T = TypeVar("T")


class APIVersion(str, Enum):
    """API version enumeration."""
    V1 = "v1"
    V2 = "v2"
    V3 = "v3"


class VersioningStrategy(str, Enum):
    """API versioning strategies."""
    URL_PATH = "url_path"
    HEADER = "header"
    QUERY_PARAM = "query_param"


@dataclass
class RESTRouterConfig:
    """Configuration for REST router."""
    version: APIVersion = APIVersion.V1
    versioning_strategy: VersioningStrategy = VersioningStrategy.URL_PATH
    version_header: str = "X-API-Version"
    version_param: str = "api_version"
    prefix: str = ""
    tags: List[str] = field(default_factory=list)
    enable_correlation_id: bool = True
    correlation_id_header: str = "X-Correlation-ID"
    enable_request_logging: bool = True
    enable_response_time: bool = True


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error_code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Error details")
    correlation_id: Optional[str] = Field(default=None, description="Correlation ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    path: Optional[str] = Field(default=None, description="Request path")


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response model."""
    items: List[T] = Field(..., description="List of items")
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Has next page")
    has_prev: bool = Field(..., description="Has previous page")


class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")
    sort_by: Optional[str] = Field(default=None, description="Sort field")
    sort_order: str = Field(default="asc", description="Sort order (asc/desc)")


class HealthStatus(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Health status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    checks: Dict[str, bool] = Field(default_factory=dict, description="Individual checks")


class RESTRouter:
    """
    REST router with versioning and GreenLang extensions.

    Provides a wrapper around FastAPI's APIRouter with built-in
    versioning, error handling, and observability features.

    Attributes:
        config: Router configuration
        router: Underlying FastAPI router

    Example:
        >>> config = RESTRouterConfig(version=APIVersion.V1)
        >>> rest_router = RESTRouter(config)
        >>> @rest_router.get("/emissions")
        >>> async def list_emissions(
        ...     pagination: PaginationParams = Depends()
        ... ):
        ...     return {"items": [...]}
    """

    def __init__(self, config: Optional[RESTRouterConfig] = None):
        """
        Initialize REST router.

        Args:
            config: Router configuration
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError(
                "FastAPI is required for REST router. "
                "Install with: pip install fastapi"
            )

        self.config = config or RESTRouterConfig()
        self._setup_router()

        logger.info(
            f"RESTRouter initialized for version {self.config.version.value}"
        )

    def _setup_router(self) -> None:
        """Set up the underlying FastAPI router."""
        prefix = self._get_prefix()

        self.router = APIRouter(
            prefix=prefix,
            tags=self.config.tags or [self.config.version.value],
        )

        # Add standard endpoints
        self._add_health_endpoint()

    def _get_prefix(self) -> str:
        """Get the route prefix based on versioning strategy."""
        if self.config.versioning_strategy == VersioningStrategy.URL_PATH:
            base = self.config.prefix or ""
            return f"{base}/{self.config.version.value}"
        return self.config.prefix

    def _add_health_endpoint(self) -> None:
        """Add standard health check endpoint."""
        @self.router.get(
            "/health",
            response_model=HealthStatus,
            tags=["Health"],
            summary="Health check"
        )
        async def health_check():
            return HealthStatus(
                status="healthy",
                version=self.config.version.value,
                checks={"api": True}
            )

    def get(
        self,
        path: str,
        **kwargs
    ) -> Callable:
        """
        Register a GET endpoint.

        Args:
            path: Route path
            **kwargs: Additional router parameters

        Returns:
            Decorator function
        """
        return self.router.get(path, **kwargs)

    def post(
        self,
        path: str,
        **kwargs
    ) -> Callable:
        """
        Register a POST endpoint.

        Args:
            path: Route path
            **kwargs: Additional router parameters

        Returns:
            Decorator function
        """
        return self.router.post(path, **kwargs)

    def put(
        self,
        path: str,
        **kwargs
    ) -> Callable:
        """
        Register a PUT endpoint.

        Args:
            path: Route path
            **kwargs: Additional router parameters

        Returns:
            Decorator function
        """
        return self.router.put(path, **kwargs)

    def patch(
        self,
        path: str,
        **kwargs
    ) -> Callable:
        """
        Register a PATCH endpoint.

        Args:
            path: Route path
            **kwargs: Additional router parameters

        Returns:
            Decorator function
        """
        return self.router.patch(path, **kwargs)

    def delete(
        self,
        path: str,
        **kwargs
    ) -> Callable:
        """
        Register a DELETE endpoint.

        Args:
            path: Route path
            **kwargs: Additional router parameters

        Returns:
            Decorator function
        """
        return self.router.delete(path, **kwargs)

    @staticmethod
    def paginate(
        items: List[T],
        total: int,
        params: PaginationParams
    ) -> PaginatedResponse[T]:
        """
        Create a paginated response.

        Args:
            items: List of items for current page
            total: Total number of items
            params: Pagination parameters

        Returns:
            Paginated response
        """
        total_pages = (total + params.page_size - 1) // params.page_size

        return PaginatedResponse(
            items=items,
            total=total,
            page=params.page,
            page_size=params.page_size,
            total_pages=total_pages,
            has_next=params.page < total_pages,
            has_prev=params.page > 1,
        )

    @staticmethod
    def error_response(
        status_code: int,
        error_code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> JSONResponse:
        """
        Create a standard error response.

        Args:
            status_code: HTTP status code
            error_code: Application error code
            message: Error message
            details: Additional error details
            correlation_id: Request correlation ID

        Returns:
            JSON response with error
        """
        error = ErrorResponse(
            error_code=error_code,
            message=message,
            details=details,
            correlation_id=correlation_id,
        )

        return JSONResponse(
            status_code=status_code,
            content=error.dict()
        )

    def include_router(
        self,
        other_router: "RESTRouter",
        prefix: str = "",
        **kwargs
    ) -> None:
        """
        Include another router.

        Args:
            other_router: Router to include
            prefix: Additional prefix
            **kwargs: Additional parameters
        """
        self.router.include_router(
            other_router.router,
            prefix=prefix,
            **kwargs
        )


def get_pagination_params(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    sort_by: Optional[str] = Query(None, description="Sort field"),
    sort_order: str = Query("asc", description="Sort order")
) -> PaginationParams:
    """
    Dependency for pagination parameters.

    Returns:
        PaginationParams instance
    """
    return PaginationParams(
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order
    )


def get_correlation_id(
    request: Request,
    header_name: str = "X-Correlation-ID"
) -> str:
    """
    Get or generate correlation ID.

    Args:
        request: FastAPI request
        header_name: Header name for correlation ID

    Returns:
        Correlation ID
    """
    correlation_id = request.headers.get(header_name)
    if not correlation_id:
        correlation_id = str(uuid4())
    return correlation_id


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for request/response logging.
    """

    async def dispatch(self, request: Request, call_next):
        """Process request with logging."""
        start_time = time.time()
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid4()))

        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"correlation_id={correlation_id}"
        )

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000

        # Add headers
        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

        # Log response
        logger.info(
            f"Response: {request.method} {request.url.path} "
            f"status={response.status_code} "
            f"duration={duration_ms:.2f}ms "
            f"correlation_id={correlation_id}"
        )

        return response


class VersionCheckMiddleware(BaseHTTPMiddleware):
    """
    Middleware for API version validation.
    """

    def __init__(
        self,
        app,
        supported_versions: List[APIVersion],
        versioning_strategy: VersioningStrategy = VersioningStrategy.URL_PATH,
        version_header: str = "X-API-Version",
        version_param: str = "api_version"
    ):
        """Initialize version check middleware."""
        super().__init__(app)
        self.supported_versions = supported_versions
        self.versioning_strategy = versioning_strategy
        self.version_header = version_header
        self.version_param = version_param

    async def dispatch(self, request: Request, call_next):
        """Check API version."""
        version = self._extract_version(request)

        if version and version not in [v.value for v in self.supported_versions]:
            return JSONResponse(
                status_code=400,
                content={
                    "error_code": "UNSUPPORTED_VERSION",
                    "message": f"API version '{version}' is not supported",
                    "details": {
                        "supported_versions": [v.value for v in self.supported_versions]
                    }
                }
            )

        return await call_next(request)

    def _extract_version(self, request: Request) -> Optional[str]:
        """Extract version from request."""
        if self.versioning_strategy == VersioningStrategy.URL_PATH:
            # Extract from path like /v1/emissions
            parts = request.url.path.split("/")
            for part in parts:
                if part.startswith("v") and part[1:].isdigit():
                    return part
            return None

        elif self.versioning_strategy == VersioningStrategy.HEADER:
            return request.headers.get(self.version_header)

        elif self.versioning_strategy == VersioningStrategy.QUERY_PARAM:
            return request.query_params.get(self.version_param)

        return None


def create_crud_router(
    resource_name: str,
    model_class: Type[BaseModel],
    version: APIVersion = APIVersion.V1,
    tags: Optional[List[str]] = None
) -> RESTRouter:
    """
    Create a CRUD router for a resource.

    Args:
        resource_name: Name of the resource (plural)
        model_class: Pydantic model class
        version: API version
        tags: Router tags

    Returns:
        Configured REST router
    """
    config = RESTRouterConfig(
        version=version,
        tags=tags or [resource_name.title()]
    )
    router = RESTRouter(config)

    # This would be customized with actual CRUD handlers
    # For now, we return the base router

    return router
