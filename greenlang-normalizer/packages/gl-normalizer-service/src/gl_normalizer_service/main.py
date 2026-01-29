"""
GL Normalizer Service - FastAPI Application Entry Point

This module provides the FastAPI application factory and startup configuration
for the GreenLang Normalizer REST API service.

The service implements the GL-FOUND-X-003 specification for climate data
normalization with the following features:

- Single value normalization (POST /v1/normalize)
- Batch normalization up to 10K items (POST /v1/normalize/batch)
- Async job processing for 100K+ items (POST /v1/jobs)
- Vocabulary management (GET /v1/vocabularies)
- Health checks and monitoring endpoints

Running the Service:
    Development:
        $ uvicorn gl_normalizer_service.main:app --reload

    Production:
        $ uvicorn gl_normalizer_service.main:app --host 0.0.0.0 --port 8000 --workers 4

    Or use the CLI:
        $ gl-normalizer-service

Environment Variables:
    GL_NORMALIZER_ENV: Environment (development, staging, production)
    GL_NORMALIZER_DEBUG: Enable debug mode
    GL_NORMALIZER_SECRET_KEY: JWT signing secret
    GL_NORMALIZER_REDIS_URL: Redis connection URL
    GL_NORMALIZER_RATE_LIMIT_REQUESTS: Rate limit per window
    GL_NORMALIZER_RATE_LIMIT_WINDOW: Rate limit window (seconds)

API Documentation:
    Swagger UI: http://localhost:8000/docs
    ReDoc: http://localhost:8000/redoc
    OpenAPI JSON: http://localhost:8000/openapi.json
"""

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi

from gl_normalizer_service import __version__, __api_revision__
from gl_normalizer_service.api.routes import router as api_router
from gl_normalizer_service.config import Settings, get_settings
from gl_normalizer_service.middleware.audit import AuditMiddleware
from gl_normalizer_service.middleware.rate_limit import RateLimitMiddleware

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


# ==============================================================================
# Application Lifespan
# ==============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Handles startup and shutdown events for the FastAPI application.
    Initializes connections to external services and cleans them up on shutdown.

    Args:
        app: FastAPI application instance

    Yields:
        None (application runs during yield)
    """
    # Startup
    logger.info(
        "service_starting",
        version=__version__,
        api_revision=__api_revision__,
        environment=app.state.settings.env,
    )

    # Store start time for uptime calculation
    app.state.start_time = time.time()

    # Initialize external connections
    # TODO: Initialize Redis connection for rate limiting
    # TODO: Initialize vocabulary service connection
    # TODO: Initialize job queue connection

    logger.info("service_started", message="GL Normalizer Service is ready")

    yield

    # Shutdown
    logger.info("service_stopping")

    # Cleanup connections
    # TODO: Close Redis connection
    # TODO: Close other connections

    logger.info("service_stopped")


# ==============================================================================
# Application Factory
# ==============================================================================


def create_app(settings: Settings | None = None) -> FastAPI:
    """
    Create and configure the FastAPI application.

    This factory function creates a fully configured FastAPI application
    with all middleware, routes, and exception handlers.

    Args:
        settings: Optional settings override (uses defaults if not provided)

    Returns:
        Configured FastAPI application

    Example:
        >>> from gl_normalizer_service import create_app
        >>> app = create_app()
        >>> # Run with: uvicorn mymodule:app
    """
    settings = settings or get_settings()

    # Create application
    app = FastAPI(
        title="GL Normalizer Service",
        description="""
## GreenLang Normalizer REST API

Production-grade REST API for climate data normalization (GL-FOUND-X-003).

### Features

- **Single Value Normalization**: Normalize individual values with confidence scoring
- **Batch Processing**: Process up to 10,000 items synchronously
- **Async Jobs**: Handle 100K+ items with background processing
- **Vocabulary Management**: Access normalization vocabularies and mappings

### Authentication

All endpoints except health checks require authentication via:
- **API Key**: `X-API-Key` header
- **JWT Bearer Token**: `Authorization: Bearer <token>` header

### Rate Limiting

Default limits:
- Single normalization: 100 requests/minute
- Batch normalization: 20 requests/minute
- Job creation: 50 requests/minute

### Error Codes

| Code | Description |
|------|-------------|
| GLNORM-001 | Invalid input value |
| GLNORM-002 | Unknown unit |
| GLNORM-003 | Incompatible unit conversion |
| GLNORM-004 | Batch size exceeded |
| GLNORM-005 | Job not found |
| GLNORM-006 | Vocabulary not found |
| GLNORM-007 | Authentication failed |
| GLNORM-008 | Rate limit exceeded |
| GLNORM-009 | Internal processing error |
| GLNORM-010 | Validation failed |
        """,
        version=__version__,
        openapi_url="/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
        openapi_tags=[
            {
                "name": "Normalization",
                "description": "Value normalization endpoints",
            },
            {
                "name": "Jobs",
                "description": "Async job management",
            },
            {
                "name": "Vocabularies",
                "description": "Vocabulary access and management",
            },
            {
                "name": "System",
                "description": "Health and status endpoints",
            },
        ],
    )

    # Store settings in app state
    app.state.settings = settings

    # ==============================================================================
    # Middleware Configuration (order matters - last added runs first)
    # ==============================================================================

    # GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Audit middleware (outermost - logs all requests)
    app.add_middleware(
        AuditMiddleware,
        capture_body=settings.debug,  # Only capture bodies in debug mode
        max_body_size=10000,
    )

    # Rate limiting middleware
    if settings.rate_limit_enabled:
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_window=settings.rate_limit_requests,
            window_seconds=settings.rate_limit_window,
        )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=[
            "X-Request-ID",
            "X-Response-Time",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
        ],
    )

    # Trusted host middleware (production only)
    if settings.is_production:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.trusted_hosts,
        )

    # ==============================================================================
    # Route Registration
    # ==============================================================================

    # API v1 routes
    app.include_router(api_router)

    # ==============================================================================
    # Exception Handlers
    # ==============================================================================

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """
        Global exception handler for unhandled errors.

        Logs the error and returns a generic error response to avoid
        leaking internal details.
        """
        request_id = getattr(request.state, "request_id", "unknown")

        logger.error(
            "unhandled_exception",
            request_id=request_id,
            path=request.url.path,
            method=request.method,
            error_type=type(exc).__name__,
            error_message=str(exc),
            exc_info=True,
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "api_revision": __api_revision__,
                "error": {
                    "code": "GLNORM-009",
                    "message": "An internal error occurred. Please try again later.",
                },
                "request_id": request_id,
            },
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        """Handle ValueError as bad request."""
        request_id = getattr(request.state, "request_id", "unknown")

        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "api_revision": __api_revision__,
                "error": {
                    "code": "GLNORM-001",
                    "message": str(exc),
                },
                "request_id": request_id,
            },
        )

    # ==============================================================================
    # Root Endpoint
    # ==============================================================================

    @app.get("/", include_in_schema=False)
    async def root() -> dict:
        """Root endpoint with service information."""
        return {
            "service": "GL Normalizer Service",
            "version": __version__,
            "api_revision": __api_revision__,
            "documentation": "/docs",
            "health": "/v1/health",
        }

    return app


# ==============================================================================
# Default Application Instance
# ==============================================================================

# Create default application instance for uvicorn
app = create_app()


# ==============================================================================
# CLI Entry Point
# ==============================================================================


def run() -> None:
    """
    CLI entry point for running the service.

    Usage:
        $ gl-normalizer-service
        $ gl-normalizer-service --host 0.0.0.0 --port 8080
    """
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(
        description="GL Normalizer Service - Climate Data Normalization API"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind (default: 8000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    args = parser.parse_args()

    settings = get_settings()

    logger.info(
        "starting_server",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        environment=settings.env,
    )

    uvicorn.run(
        "gl_normalizer_service.main:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level="debug" if settings.debug else "info",
    )


if __name__ == "__main__":
    run()
