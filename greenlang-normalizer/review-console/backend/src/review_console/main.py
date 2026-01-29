"""
Review Console Backend - FastAPI Application.

This module provides the main FastAPI application for the Review Console,
a human review queue for GreenLang entity resolution (GL-FOUND-X-003).

Features:
    - REST API for review queue management
    - JWT-based authentication
    - Rate limiting with slowapi
    - CORS support for frontend integration
    - Health check endpoints
    - Comprehensive request/response logging

Usage:
    # Development
    uvicorn review_console.main:app --reload --port 8000

    # Production
    uvicorn review_console.main:app --host 0.0.0.0 --port 8000 --workers 4

Example:
    >>> from review_console.main import app
    >>> # Use with ASGI server
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator
import uuid

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import structlog

from review_console import __version__
from review_console.config import get_settings
from review_console.db.session import init_db, close_db
from review_console.api.routes import router as api_router

# Configure structured logging
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        structlog.get_logger().level if hasattr(structlog.get_logger(), 'level') else 0
    ),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()
settings = get_settings()


# ============================================================================
# Rate Limiting
# ============================================================================


def get_rate_limit_key(request: Request) -> str:
    """
    Get rate limit key from request.

    Uses user ID from JWT token if available, otherwise falls back to IP address.
    """
    # Try to get user ID from token (if already validated)
    # For simplicity, we use IP address here
    return get_remote_address(request)


limiter = Limiter(
    key_func=get_rate_limit_key,
    default_limits=[settings.rate_limit_string],
    enabled=settings.rate_limit_enabled,
)


# ============================================================================
# Application Lifecycle
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan context manager.

    Handles startup and shutdown events:
    - Startup: Initialize database tables
    - Shutdown: Close database connections

    Args:
        app: FastAPI application instance
    """
    # Startup
    logger.info(
        "Starting Review Console Backend",
        version=__version__,
        env=settings.env,
    )

    try:
        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error("Failed to initialize database", error=str(e))
        # Don't raise - allow app to start for health checks

    yield

    # Shutdown
    logger.info("Shutting down Review Console Backend")
    await close_db()
    logger.info("Database connections closed")


# ============================================================================
# FastAPI Application
# ============================================================================


app = FastAPI(
    title="Review Console API",
    description="""
# Review Console Backend

Human Review Queue API for GreenLang Entity Resolution (GL-FOUND-X-003).

## Overview

The Review Console provides a REST API for managing entity resolution items
that require human review. This includes:

- Viewing and filtering review queue items
- Resolving items with canonical entity selections
- Rejecting or escalating items
- Suggesting new vocabulary entries
- Dashboard statistics

## Authentication

All endpoints require JWT Bearer token authentication.

Include the token in the Authorization header:
```
Authorization: Bearer <your-jwt-token>
```

## Rate Limiting

API requests are rate limited to {rate_limit} per window.
Rate limit information is included in response headers.
    """.format(rate_limit=settings.rate_limit_string),
    version=__version__,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
    contact={
        "name": "GreenLang Engineering",
        "email": "engineering@greenlang.io",
    },
    license_info={
        "name": "Proprietary",
    },
)

# Attach limiter to app state
app.state.limiter = limiter


# ============================================================================
# Middleware
# ============================================================================


# Rate limiting middleware
app.add_middleware(SlowAPIMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
)

# Trusted host middleware (only in production)
if settings.is_production:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.trusted_hosts,
    )


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """
    Middleware for request logging and request ID generation.

    Adds a unique request ID to each request and logs request/response details.
    """
    # Generate request ID
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    # Bind request ID to logger context
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)

    # Log request
    logger.info(
        "Request started",
        method=request.method,
        path=request.url.path,
        client_ip=request.client.host if request.client else None,
    )

    # Process request
    start_time = datetime.now(timezone.utc)
    response = await call_next(request)
    duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

    # Add request ID to response headers
    response.headers["X-Request-ID"] = request_id

    # Log response
    logger.info(
        "Request completed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round(duration_ms, 2),
    )

    return response


# ============================================================================
# Exception Handlers
# ============================================================================


@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """Handle rate limit exceeded errors."""
    logger.warning(
        "Rate limit exceeded",
        path=request.url.path,
        client_ip=request.client.host if request.client else None,
    )

    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "error": "RATE_LIMIT_EXCEEDED",
            "message": "Too many requests. Please try again later.",
            "retry_after": exc.detail if hasattr(exc, 'detail') else None,
        },
        headers={"Retry-After": str(settings.rate_limit_window)},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    logger.warning(
        "Validation error",
        path=request.url.path,
        errors=exc.errors(),
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "VALIDATION_ERROR",
            "message": "Request validation failed",
            "details": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    logger.error(
        "Unexpected error",
        path=request.url.path,
        error=str(exc),
        error_type=type(exc).__name__,
    )

    # Don't expose internal error details in production
    if settings.is_production:
        message = "An internal error occurred"
    else:
        message = str(exc)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "INTERNAL_ERROR",
            "message": message,
            "request_id": getattr(request.state, 'request_id', None),
        },
    )


# ============================================================================
# Routes
# ============================================================================


# Include API routes
app.include_router(api_router, prefix="/api")


# Root endpoint
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Review Console API",
        "version": __version__,
        "docs": "/api/docs",
        "health": "/api/health",
    }


# Ready endpoint (for Kubernetes readiness probes)
@app.get("/ready", include_in_schema=False)
async def ready():
    """Readiness probe endpoint."""
    from review_console.db.session import check_database_health

    db_healthy = await check_database_health()

    if not db_healthy:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"ready": False, "reason": "Database unavailable"},
        )

    return {"ready": True}


# Live endpoint (for Kubernetes liveness probes)
@app.get("/live", include_in_schema=False)
async def live():
    """Liveness probe endpoint."""
    return {"live": True}


# ============================================================================
# Entry Point
# ============================================================================


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "review_console.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info" if not settings.debug else "debug",
    )
