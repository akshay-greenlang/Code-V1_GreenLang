# -*- coding: utf-8 -*-
"""
API Module for GL-FOUND-X-002 (GreenLang Schema Compiler & Validator).

This module provides the FastAPI HTTP service for remote validation.

Endpoints:
    POST /v1/schema/validate           - Validate single payload
    POST /v1/schema/validate/batch     - Validate multiple payloads
    POST /v1/schema/compile            - Compile schema to IR
    GET  /v1/schema/{schema_id}/versions - List schema versions
    GET  /v1/schema/{schema_id}/{version} - Get schema details
    GET  /health                       - Health check
    GET  /metrics                      - Prometheus metrics

Example:
    >>> from greenlang.schema.api import create_app
    >>> app = create_app()
    >>> # Run with: uvicorn app:app

    >>> # Or use the routers directly
    >>> from fastapi import FastAPI
    >>> from greenlang.schema.api import router, system_router
    >>> app = FastAPI()
    >>> app.include_router(router)
    >>> app.include_router(system_router)

Author: GreenLang Framework Team
Version: 1.0.0
GL-FOUND-X-002: Schema Compiler & Validator - Task 5.3
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from greenlang.schema.api.routes import router, system_router
from greenlang.schema.api.dependencies import (
    APIConfig,
    SERVICE_VERSION,
    MetricsCollector,
    RateLimiter,
    RequestContext,
    get_config,
    get_metrics,
    get_validator,
    get_compiler,
    get_registry,
)
from greenlang.schema.api.models import (
    ValidateRequest,
    ValidateResponse,
    BatchValidateRequest,
    BatchValidateResponse,
    CompileRequest,
    CompileResponse,
    HealthResponse,
    MetricsResponse,
    SchemaVersionsResponse,
    SchemaDetailResponse,
    ErrorResponse,
    ErrorDetail,
)


def create_app(
    title: str = "GreenLang Schema Validator API",
    description: Optional[str] = None,
    version: str = SERVICE_VERSION,
    cors_origins: Optional[list] = None,
    docs_url: str = "/api/docs",
    redoc_url: str = "/api/redoc",
    openapi_url: str = "/api/openapi.json",
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    This factory function creates a fully configured FastAPI application
    with all routes, middleware, and dependencies set up.

    Args:
        title: API title for OpenAPI documentation.
        description: API description for OpenAPI documentation.
        version: API version string.
        cors_origins: List of allowed CORS origins. If None, allows all.
        docs_url: URL path for Swagger UI docs.
        redoc_url: URL path for ReDoc documentation.
        openapi_url: URL path for OpenAPI JSON spec.

    Returns:
        FastAPI: Configured FastAPI application instance.

    Example:
        >>> app = create_app()
        >>> # Run with: uvicorn module:app --host 0.0.0.0 --port 8000

        >>> # With custom configuration
        >>> app = create_app(
        ...     title="My Schema API",
        ...     cors_origins=["https://myapp.com"],
        ... )
    """
    if description is None:
        description = """
GreenLang Schema Compiler & Validator REST API.

This API provides endpoints for validating payloads against GreenLang schemas,
compiling schemas to intermediate representation, and managing schema versions.

## Features

- **Single Validation**: Validate one payload against a schema
- **Batch Validation**: Validate multiple payloads efficiently
- **Schema Compilation**: Compile schemas to optimized IR
- **Version Management**: List and retrieve schema versions
- **Health Monitoring**: Health check and Prometheus metrics

## Authentication

Authentication is optional and can be enabled via environment variables:
- `GL_SCHEMA_API_REQUIRE_KEY=true` - Enable API key requirement
- `GL_SCHEMA_API_KEYS=key1,key2` - Comma-separated valid API keys

When enabled, include the API key in the `X-API-Key` header.

## Rate Limiting

Rate limiting is enabled by default (1000 requests/minute per IP).
Configure via environment variable:
- `GL_SCHEMA_API_RATE_LIMIT=1000` - Requests per minute

## Tracing

Include an `X-Trace-ID` header to track requests across systems.
If not provided, one will be generated.
        """

    # Create FastAPI app
    app = FastAPI(
        title=title,
        description=description,
        version=version,
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url=openapi_url,
        openapi_tags=[
            {
                "name": "Schema Validation",
                "description": "Validate payloads against GreenLang schemas",
            },
            {
                "name": "System",
                "description": "Health checks and metrics",
            },
        ],
    )

    # Add CORS middleware
    if cors_origins is None:
        # Default: allow all origins in development
        cors_origins = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Trace-ID", "X-RateLimit-Remaining", "Retry-After"],
    )

    # Include routers
    app.include_router(router)
    app.include_router(system_router)

    # Add startup event
    @app.on_event("startup")
    async def startup_event():
        """Initialize resources on startup."""
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"GreenLang Schema Validator API v{version} starting up")

    # Add shutdown event
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup resources on shutdown."""
        import logging
        logger = logging.getLogger(__name__)
        logger.info("GreenLang Schema Validator API shutting down")

    return app


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Application factory
    "create_app",

    # Routers
    "router",
    "system_router",

    # Configuration
    "APIConfig",
    "SERVICE_VERSION",

    # Dependencies
    "MetricsCollector",
    "RateLimiter",
    "RequestContext",
    "get_config",
    "get_metrics",
    "get_validator",
    "get_compiler",
    "get_registry",

    # Request/Response models
    "ValidateRequest",
    "ValidateResponse",
    "BatchValidateRequest",
    "BatchValidateResponse",
    "CompileRequest",
    "CompileResponse",
    "HealthResponse",
    "MetricsResponse",
    "SchemaVersionsResponse",
    "SchemaDetailResponse",
    "ErrorResponse",
    "ErrorDetail",
]
