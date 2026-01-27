# -*- coding: utf-8 -*-
"""
GreenLang Orchestrator Control Plane API
=========================================

FastAPI-based REST API for the GL-FOUND-X-001 GreenLang Orchestrator.

This module provides the Control Plane API for:
- Pipeline Management (register, list, get, delete)
- Run Operations (submit, list, get, cancel, logs, audit)
- Health and Metrics (health, metrics, ready)
- Agent Registry (list, get)

Usage:
    # Create and run the app
    from greenlang.orchestrator.api import create_app

    app = create_app()

    # With uvicorn
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

    # Or use the router directly
    from greenlang.orchestrator.api import router
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)

Endpoints:
    Pipeline Management:
        - POST /api/v1/pipelines - Register a pipeline definition
        - GET /api/v1/pipelines - List pipelines
        - GET /api/v1/pipelines/{id} - Get pipeline details
        - DELETE /api/v1/pipelines/{id} - Unregister pipeline

    Run Operations:
        - POST /api/v1/runs - Submit a new run
        - GET /api/v1/runs - List runs (with filters)
        - GET /api/v1/runs/{id} - Get run details and status
        - POST /api/v1/runs/{id}/cancel - Cancel a running run
        - GET /api/v1/runs/{id}/logs - Get run logs
        - GET /api/v1/runs/{id}/audit - Get audit trail

    Health and Metrics:
        - GET /health - Health check
        - GET /metrics - Prometheus metrics
        - GET /ready - Readiness probe

    Agent Registry:
        - GET /api/v1/agents - List registered agents
        - GET /api/v1/agents/{id} - Get agent details

Author: GreenLang Team
Version: 1.0.0
GL-FOUND-X-001: Control Plane API
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from greenlang.orchestrator.api.deps import (
    APIConfig,
    get_config,
    shutdown_dependencies,
    startup_dependencies,
)
from greenlang.orchestrator.api.models import ErrorResponse
from greenlang.orchestrator.api.routes import router

logger = logging.getLogger(__name__)


def create_app(config: Optional[APIConfig] = None) -> FastAPI:
    """
    Create and configure the FastAPI application.

    This is the application factory function that creates a fully configured
    FastAPI application with all middleware, routes, and handlers.

    Args:
        config: Optional API configuration. If not provided, loads from environment.

    Returns:
        Configured FastAPI application

    Example:
        >>> app = create_app()
        >>> # Run with uvicorn
        >>> import uvicorn
        >>> uvicorn.run(app, host="0.0.0.0", port=8000)
    """
    # Load config
    if config is None:
        config = get_config()

    # Create lifespan context manager
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Manage application lifecycle."""
        logger.info("Starting GreenLang Orchestrator Control Plane API...")
        try:
            await startup_dependencies()
            logger.info("API startup complete")
            yield
        finally:
            logger.info("Shutting down API...")
            await shutdown_dependencies()
            logger.info("API shutdown complete")

    # Create FastAPI app
    app = FastAPI(
        title=config.api_title,
        description=config.api_description,
        version=config.api_version,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
        responses={
            400: {"model": ErrorResponse, "description": "Bad Request"},
            401: {"model": ErrorResponse, "description": "Unauthorized"},
            403: {"model": ErrorResponse, "description": "Forbidden"},
            404: {"model": ErrorResponse, "description": "Not Found"},
            500: {"model": ErrorResponse, "description": "Internal Server Error"},
            503: {"model": ErrorResponse, "description": "Service Unavailable"},
        },
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Trace-ID", "X-Request-ID"],
    )

    # Add GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Add request ID middleware
    @app.middleware("http")
    async def add_request_id(request, call_next):
        """Add request ID header for tracing."""
        import uuid

        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    # Add request logging middleware
    @app.middleware("http")
    async def log_requests(request, call_next):
        """Log all requests."""
        import time

        start_time = time.time()
        response = await call_next(request)
        duration = (time.time() - start_time) * 1000

        logger.info(
            f"{request.method} {request.url.path} "
            f"status={response.status_code} "
            f"duration={duration:.2f}ms"
        )

        return response

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        """Handle unhandled exceptions."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)

        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "message": "An unexpected error occurred",
                "details": [],
            },
        )

    # Include main router
    app.include_router(router)

    # Add root endpoint
    @app.get("/", include_in_schema=False)
    async def root():
        """Root endpoint with API info."""
        return {
            "name": config.api_title,
            "version": config.api_version,
            "docs": "/api/docs",
            "openapi": "/api/openapi.json",
        }

    logger.info(f"Created {config.api_title} v{config.api_version}")

    return app


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # App factory
    "create_app",
    # Router
    "router",
    # Models
    "ErrorResponse",
    # Config
    "APIConfig",
    "get_config",
]
