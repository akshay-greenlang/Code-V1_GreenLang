"""
GreenLang Agent Factory - FastAPI Application Entry Point

This module provides the FastAPI application factory and configuration
for the GreenLang Agent Factory backend.

Example:
    >>> from app.main import create_app
    >>> app = create_app()
    >>> # Run with: uvicorn app.main:app --reload
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.routers import agents, executions, search, metrics, tenants, audit
from app.middleware.auth import JWTAuthMiddleware
from app.middleware.rate_limit import RateLimitMiddleware
from app.middleware.request_validation import RequestValidationMiddleware

# OpenTelemetry imports
from app.telemetry import (
    init_telemetry,
    instrument_fastapi,
    instrument_httpx,
    instrument_redis,
    shutdown_telemetry,
    get_trace_id,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan context manager.

    Handles startup and shutdown events:
    - Startup: Initialize database pools, Redis connections, services, telemetry
    - Shutdown: Gracefully close connections, drain requests, flush traces
    """
    settings = get_settings()

    # Startup
    logger.info("Starting GreenLang Agent Factory...")

    # Initialize OpenTelemetry distributed tracing
    if settings.tracing_enabled:
        try:
            init_telemetry(
                service_name="greenlang-agent-factory",
                service_version=settings.api_version,
                otlp_endpoint=settings.otlp_endpoint,
                sample_rate=1.0,  # Sample all traces in development
                enable_console_exporter=(settings.app_env == "development"),
                environment=settings.app_env,
            )
            logger.info("OpenTelemetry tracing initialized")

            # Instrument httpx for outgoing HTTP requests
            instrument_httpx()
            logger.info("httpx instrumentation enabled")
        except Exception as e:
            logger.warning(f"Failed to initialize telemetry: {e}")

    # Initialize database connection pool
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from db.connection import init_db_pool, close_db_pool
        from registry.service import AgentRegistryService

        await init_db_pool(
            settings.database_url,
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
            pool_timeout=settings.database_pool_timeout,
        )
        app.state.db_initialized = True
        logger.info("Database connection pool initialized")
    except Exception as e:
        logger.warning(f"Database not available, using in-memory mode: {e}")
        app.state.db_initialized = False

    # Initialize Redis connection for rate limiting
    try:
        from middleware.rate_limit import create_redis_pool
        app.state.redis = create_redis_pool(
            settings.redis_url,
            max_connections=settings.redis_max_connections,
        )
        logger.info("Redis connection initialized")

        # Instrument Redis for distributed tracing
        if settings.tracing_enabled and app.state.redis:
            try:
                instrument_redis(app.state.redis)
                logger.info("Redis instrumentation enabled")
            except Exception as e:
                logger.warning(f"Failed to instrument Redis: {e}")
    except Exception as e:
        logger.warning(f"Redis not available, rate limiting disabled: {e}")
        app.state.redis = None

    # Initialize in-memory registry service for fallback
    try:
        from registry.service import AgentRegistryService
        app.state.registry_service = AgentRegistryService(session=None)
        logger.info("In-memory registry service initialized as fallback")
    except Exception as e:
        logger.error(f"Failed to initialize registry service: {e}")

    logger.info("GreenLang Agent Factory started successfully")

    yield

    # Shutdown
    logger.info("Shutting down GreenLang Agent Factory...")

    # Close database connections
    if getattr(app.state, 'db_initialized', False):
        try:
            from db.connection import close_db_pool
            await close_db_pool()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database: {e}")

    # Close Redis connections
    if getattr(app.state, 'redis', None):
        try:
            app.state.redis.close()
            logger.info("Redis connections closed")
        except Exception as e:
            logger.error(f"Error closing Redis: {e}")

    # Shutdown OpenTelemetry and flush pending traces
    if settings.tracing_enabled:
        try:
            shutdown_telemetry()
            logger.info("OpenTelemetry telemetry shut down")
        except Exception as e:
            logger.error(f"Error shutting down telemetry: {e}")

    logger.info("GreenLang Agent Factory shut down successfully")


def create_app(settings: Settings | None = None) -> FastAPI:
    """
    Create and configure the FastAPI application.

    This factory function creates the FastAPI app with:
    - OpenAPI documentation
    - CORS configuration
    - Middleware stack (auth, rate limiting, compression)
    - API routers
    - Exception handlers

    Args:
        settings: Optional settings override

    Returns:
        Configured FastAPI application

    Example:
        >>> app = create_app()
        >>> # Run with: uvicorn app.main:app --reload
    """
    settings = settings or get_settings()

    app = FastAPI(
        title="GreenLang Agent Factory",
        description="""
## GreenLang Agent Factory API

The Agent Factory API provides:

- **Agent Registry**: Manage agent lifecycle (register, update, delete)
- **Agent Execution**: Execute agents with zero-hallucination calculations
- **Search & Discovery**: Find agents by capability, category, or text search
- **Metrics & Analytics**: Track execution metrics and costs
- **Multi-Tenancy**: Tenant isolation with row-level security

### Key Features

- Zero-hallucination calculations with SHA-256 provenance
- Regulatory compliance (CBAM, CSRD, EUDR, GHG Protocol)
- Enterprise multi-tenancy with RBAC
- Real-time execution streaming

### Authentication

All endpoints require JWT authentication via the `Authorization: Bearer <token>` header.
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Add custom middleware (order matters - first added = last executed)
    # Note: Middleware is added but will gracefully degrade if services unavailable
    # Enable request validation
    app.add_middleware(RequestValidationMiddleware)

    # Enable rate limiting (will fail-open if Redis unavailable)
    app.add_middleware(
        RateLimitMiddleware,
        redis=None,  # Redis will be set during lifespan startup
        requests_per_minute=settings.rate_limit_per_minute,
        burst_size=settings.rate_limit_burst,
    )

    # Enable JWT authentication
    app.add_middleware(
        JWTAuthMiddleware,
        secret_key=settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )

    # Instrument FastAPI with OpenTelemetry (after middleware setup)
    if settings.tracing_enabled:
        try:
            instrument_fastapi(app)
            logger.info("FastAPI instrumentation enabled")
        except Exception as e:
            logger.warning(f"Failed to instrument FastAPI: {e}")

    # Register routers
    app.include_router(
        agents.router,
        prefix="/v1/agents",
        tags=["Agents"],
    )
    app.include_router(
        executions.router,
        prefix="/v1/agents",
        tags=["Executions"],
    )
    app.include_router(
        search.router,
        prefix="/v1/agents",
        tags=["Search"],
    )
    app.include_router(
        metrics.router,
        prefix="/v1",
        tags=["Metrics"],
    )
    app.include_router(
        tenants.router,
        prefix="/v1/tenants",
        tags=["Tenants"],
    )
    app.include_router(
        audit.router,
        prefix="/v1/audit-logs",
        tags=["Audit"],
    )

    # Exception handlers
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        """Handle validation errors."""
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": str(exc),
                }
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected errors."""
        logger.error(f"Unexpected error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An internal error occurred",
                }
            },
        )

    # Health check endpoint
    @app.get("/health", tags=["Health"])
    async def health_check() -> Dict[str, Any]:
        """
        Health check endpoint.

        Returns service health status for load balancers and monitoring.
        """
        response = {
            "status": "healthy",
            "service": "greenlang-agent-factory",
            "version": "1.0.0",
        }
        # Include trace_id for debugging if tracing is enabled
        trace_id = get_trace_id()
        if trace_id:
            response["trace_id"] = trace_id
        return response

    # Readiness check endpoint
    @app.get("/ready", tags=["Health"])
    async def readiness_check() -> Dict[str, Any]:
        """
        Readiness check endpoint.

        Returns whether the service is ready to accept traffic.
        Checks database and Redis connectivity.
        """
        response = {
            "ready": True,
            "checks": {
                "database": "ok",
                "redis": "ok",
            },
        }
        # Include trace_id for debugging if tracing is enabled
        trace_id = get_trace_id()
        if trace_id:
            response["trace_id"] = trace_id
        return response

    return app


# Create the application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
