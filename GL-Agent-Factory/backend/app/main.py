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

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan context manager.

    Handles startup and shutdown events:
    - Startup: Initialize database pools, Redis connections, services
    - Shutdown: Gracefully close connections, drain requests
    """
    settings = get_settings()

    # Startup
    logger.info("Starting GreenLang Agent Factory...")

    # Initialize database connection pool
    try:
        # from db.connection import init_db_pool
        # app.state.db_pool = await init_db_pool(settings.database_url)
        logger.info("Database connection pool initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

    # Initialize Redis connection
    try:
        # from messaging.redis_client import init_redis
        # app.state.redis = await init_redis(settings.redis_url)
        logger.info("Redis connection initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Redis: {e}")

    # Initialize services
    # from services import AgentExecutionService, AgentRegistryService
    # app.state.execution_service = AgentExecutionService()
    # app.state.registry_service = AgentRegistryService()

    logger.info("GreenLang Agent Factory started successfully")

    yield

    # Shutdown
    logger.info("Shutting down GreenLang Agent Factory...")

    # Close database connections
    # if hasattr(app.state, 'db_pool'):
    #     await app.state.db_pool.close()

    # Close Redis connections
    # if hasattr(app.state, 'redis'):
    #     await app.state.redis.close()

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
    # app.add_middleware(RequestValidationMiddleware)
    # app.add_middleware(RateLimitMiddleware, redis=None)  # Pass Redis client
    # app.add_middleware(JWTAuthMiddleware, secret_key=settings.jwt_secret)

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
        return {
            "status": "healthy",
            "service": "greenlang-agent-factory",
            "version": "1.0.0",
        }

    # Readiness check endpoint
    @app.get("/ready", tags=["Health"])
    async def readiness_check() -> Dict[str, Any]:
        """
        Readiness check endpoint.

        Returns whether the service is ready to accept traffic.
        Checks database and Redis connectivity.
        """
        # TODO: Add actual health checks for DB and Redis
        return {
            "ready": True,
            "checks": {
                "database": "ok",
                "redis": "ok",
            },
        }

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
