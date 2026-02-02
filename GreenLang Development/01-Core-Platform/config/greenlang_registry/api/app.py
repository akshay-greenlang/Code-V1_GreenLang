"""
FastAPI Application for GreenLang Agent Registry

This module creates and configures the FastAPI application for the Agent Registry.
It provides RESTful API endpoints for agent lifecycle management.

Key Features:
- Agent publish, list, get, and promote operations
- Async database operations with connection pooling
- Request validation with Pydantic models
- Comprehensive error handling
- Health check endpoints
- OpenAPI documentation

Example:
    >>> from greenlang_registry.api import create_app
    >>> app = create_app()
    >>> # Run with: uvicorn greenlang_registry.api.app:app --reload
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional
import uuid

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from greenlang_registry import __version__
from greenlang_registry.db.client import init_database, close_database, get_database_client
from greenlang_registry.api.routes import router as registry_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler for startup/shutdown events.

    Manages database connection lifecycle:
    - On startup: Initialize database connection pool
    - On shutdown: Close all database connections
    """
    # Startup
    logger.info("Starting GreenLang Agent Registry API...")

    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@localhost:5432/greenlang_registry"
    )

    pool_size = int(os.getenv("DB_POOL_SIZE", "20"))
    max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "10"))
    echo_sql = os.getenv("DB_ECHO", "false").lower() == "true"

    try:
        await init_database(
            database_url=database_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            echo=echo_sql,
        )
        logger.info("Database connection established successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down GreenLang Agent Registry API...")
    await close_database()
    logger.info("Database connections closed")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured application instance

    Example:
        >>> app = create_app()
        >>> # Access the app in tests or for custom configuration
    """
    application = FastAPI(
        title="GreenLang Agent Registry",
        description="""
        The centralized metadata repository for GreenLang agents.

        ## Features

        - **Agent Publishing**: Register new agents and versions
        - **Agent Discovery**: Search and list available agents
        - **Lifecycle Management**: Promote agents through draft -> experimental -> certified -> deprecated
        - **Multi-tenant Support**: Tenant isolation for enterprise deployments

        ## Lifecycle States

        | State | Description |
        |-------|-------------|
        | draft | Initial state for new agents |
        | experimental | Ready for limited testing |
        | certified | Production-ready, fully tested |
        | deprecated | Scheduled for removal |

        ## Authentication

        All endpoints require authentication via API key or JWT token.
        Include the token in the `Authorization` header.
        """,
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Configure CORS
    cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
    application.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add request ID middleware
    @application.middleware("http")
    async def add_request_id(request: Request, call_next):
        """Add unique request ID to each request for tracing."""
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    # Global exception handler
    @application.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Handle unhandled exceptions."""
        request_id = getattr(request.state, "request_id", "unknown")
        logger.error(
            f"Unhandled exception [request_id={request_id}]: {exc}",
            exc_info=True
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred",
                "request_id": request_id,
            },
        )

    # Include API routes
    application.include_router(
        registry_router,
        prefix="/api/v1/registry",
        tags=["Registry"],
    )

    # Health check endpoints
    @application.get("/health", tags=["Health"])
    async def health_check():
        """
        Basic health check endpoint.

        Returns:
            dict: Health status
        """
        return {
            "status": "healthy",
            "version": __version__,
            "timestamp": datetime.utcnow().isoformat(),
        }

    @application.get("/health/ready", tags=["Health"])
    async def readiness_check():
        """
        Readiness check including database connectivity.

        Returns:
            dict: Readiness status with database health
        """
        try:
            client = get_database_client()
            db_health = await client.health_check()

            return {
                "status": "ready" if db_health["status"] == "healthy" else "not_ready",
                "database": db_health["database"],
                "pool": db_health.get("pool", {}),
                "version": __version__,
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"Readiness check failed: {e}")
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "database": "disconnected",
                    "error": str(e),
                    "version": __version__,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

    @application.get("/health/live", tags=["Health"])
    async def liveness_check():
        """
        Liveness check for Kubernetes probes.

        Returns:
            dict: Liveness status
        """
        return {"status": "alive"}

    return application


# Create default application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "greenlang_registry.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
