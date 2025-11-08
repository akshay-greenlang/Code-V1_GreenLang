"""
GL-VCCI Scope 3 Carbon Intelligence Platform - Backend API
Main FastAPI Application Entry Point

This module initializes the FastAPI application and registers all routes,
middleware, and dependencies for the GL-VCCI platform.

SECURITY UPDATE (2025-11-08):
- Added JWT authentication middleware (CRIT-003)
- Added rate limiting middleware
- Added security headers middleware

Version: 2.0.0
Security Update: 2025-11-08
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.redis import RedisIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
import uvicorn

# Add parent directory to path to import services
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration
from config.settings import get_settings
from config.database import get_db, init_db
from config.redis_client import get_redis, init_redis

# Import logging from services (structured logging)
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.logging_config import setup_logging, CorrelationContext, CarbonContext

# Import routers from agents
from services.agents.intake.routes import router as intake_router
from services.agents.calculator.routes import router as calculator_router
from services.agents.hotspot.routes import router as hotspot_router
from services.agents.engagement.routes import router as engagement_router
from services.agents.reporting.routes import router as reporting_router

# Import utility routers
from services.factor_broker.routes import router as factor_broker_router
from services.methodologies.routes import router as methodologies_router
from connectors.routes import router as connectors_router

# Import authentication
from backend.auth import validate_jwt_config, verify_token

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Security headers middleware.

    Adds security headers to all responses to prevent common web vulnerabilities.
    """

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=()"
        )

        # Cache control for API responses
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = (
                "no-store, no-cache, must-revalidate, private"
            )
            response.headers["Pragma"] = "no-cache"

        return response


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Lifespan context manager for FastAPI application.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("🚀 Starting GL-VCCI Backend API...")
    logger.info(f"Environment: {settings.APP_ENV}")
    logger.info(f"Version: {settings.API_VERSION}")

    try:
        # Initialize Sentry error tracking
        if hasattr(settings, 'SENTRY_DSN') and settings.SENTRY_DSN:
            sentry_sdk.init(
                dsn=settings.SENTRY_DSN,
                environment=settings.APP_ENV,
                release=f"vcci-scope3-platform@{settings.API_VERSION}",
                integrations=[
                    FastApiIntegration(),
                    RedisIntegration(),
                    SqlalchemyIntegration(),
                ],
                traces_sample_rate=0.1 if settings.APP_ENV == "production" else 1.0,
                profiles_sample_rate=0.1 if settings.APP_ENV == "production" else 1.0,
                # Carbon-specific context
                before_send=lambda event, hint: _sentry_before_send(event, hint),
            )
            logger.info("✅ Sentry error tracking configured")

        # Validate JWT configuration
        validate_jwt_config()
        logger.info("✅ JWT authentication configured")

        # Initialize database
        await init_db()
        logger.info("✅ Database connection established")

        # Initialize Redis
        await init_redis()
        logger.info("✅ Redis connection established")

        logger.info("🎉 GL-VCCI Backend API started successfully!")

    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise

    yield

    # Shutdown
    logger.info("🛑 Shutting down GL-VCCI Backend API...")
    # Cleanup code here if needed
    logger.info("👋 Shutdown complete")


# Initialize FastAPI application
app = FastAPI(
    title="GL-VCCI Scope 3 Carbon Intelligence API",
    description="Enterprise-grade Scope 3 emissions tracking platform with AI-powered intelligence",
    version=settings.API_VERSION,
    docs_url="/docs" if settings.APP_ENV != "production" else None,
    redoc_url="/redoc" if settings.APP_ENV != "production" else None,
    openapi_url="/openapi.json" if settings.APP_ENV != "production" else None,
    lifespan=lifespan,
)


# ==============================================================================
# Middleware Configuration
# ==============================================================================

# Security Headers Middleware (FIRST - applies to all responses)
app.add_middleware(SecurityHeadersMiddleware)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Total-Count", "X-Page", "X-Per-Page"],
)

# GZip Compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Trusted Host Middleware
if settings.APP_ENV == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.ALLOWED_HOSTS,
    )

# Rate Limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ==============================================================================
# Exception Handlers
# ==============================================================================

def _sentry_before_send(event, hint):
    """
    Sentry event processing - add carbon-specific context.
    """
    # Add carbon context if available
    carbon_ctx = CarbonContext.get_context()
    if any(carbon_ctx.values()):
        event.setdefault('contexts', {})['carbon'] = carbon_ctx

    # Add correlation ID
    correlation_id = CorrelationContext.get_correlation_id()
    if correlation_id:
        event.setdefault('tags', {})['correlation_id'] = correlation_id

    return event


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for all unhandled exceptions."""
    # Capture in Sentry with context
    if hasattr(settings, 'SENTRY_DSN') and settings.SENTRY_DSN:
        with sentry_sdk.push_scope() as scope:
            scope.set_context("request", {
                "url": str(request.url),
                "method": request.method,
                "headers": dict(request.headers),
            })
            scope.set_tag("endpoint", request.url.path)
            sentry_sdk.capture_exception(exc)

    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
            "request_id": request.headers.get("X-Request-ID"),
        },
    )


# ==============================================================================
# Health Check Endpoints
# ==============================================================================

@app.get("/health/live", tags=["Health"])
async def liveness_probe():
    """
    Kubernetes liveness probe endpoint.
    Returns 200 if the application is running.
    """
    return {"status": "alive", "service": "gl-vcci-api"}


@app.get("/health/ready", tags=["Health"])
async def readiness_probe():
    """
    Kubernetes readiness probe endpoint.
    Returns 200 if the application is ready to serve traffic.
    Checks database and Redis connections.
    """
    checks = {
        "database": False,
        "redis": False,
    }

    try:
        # Check database
        db = await get_db()
        await db.execute("SELECT 1")
        checks["database"] = True
    except Exception as e:
        logger.error(f"Database check failed: {str(e)}")

    try:
        # Check Redis
        redis = await get_redis()
        await redis.ping()
        checks["redis"] = True
    except Exception as e:
        logger.error(f"Redis check failed: {str(e)}")

    all_ready = all(checks.values())
    status_code = status.HTTP_200_OK if all_ready else status.HTTP_503_SERVICE_UNAVAILABLE

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if all_ready else "not_ready",
            "service": "gl-vcci-api",
            "checks": checks,
        },
    )


@app.get("/health/startup", tags=["Health"])
async def startup_probe():
    """
    Kubernetes startup probe endpoint.
    Returns 200 when the application has completed initialization.
    """
    return {"status": "started", "service": "gl-vcci-api"}


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "service": "GL-VCCI Scope 3 Carbon Intelligence API",
        "version": settings.API_VERSION,
        "environment": settings.APP_ENV,
        "docs": f"/docs" if settings.APP_ENV != "production" else "Disabled in production",
        "health": {
            "liveness": "/health/live",
            "readiness": "/health/ready",
            "startup": "/health/startup",
        },
    }


# ==============================================================================
# Register Routers
# ==============================================================================

# SECURITY NOTE: All API routes now require authentication via dependencies=[Depends(verify_token)]
# This implements the fix for CRIT-003: Missing API Authentication Middleware
#
# For development/testing, you can temporarily disable authentication by removing
# the dependencies parameter, but NEVER do this in production!

# Core Agent Routers (PROTECTED - Require Authentication)
app.include_router(
    intake_router,
    prefix="/api/v1/intake",
    tags=["Intake Agent"],
    dependencies=[Depends(verify_token)],  # SECURITY: Require authentication
)

app.include_router(
    calculator_router,
    prefix="/api/v1/calculator",
    tags=["Calculator Agent"],
    dependencies=[Depends(verify_token)],  # SECURITY: Require authentication
)

app.include_router(
    hotspot_router,
    prefix="/api/v1/hotspot",
    tags=["Hotspot Agent"],
    dependencies=[Depends(verify_token)],  # SECURITY: Require authentication
)

app.include_router(
    engagement_router,
    prefix="/api/v1/engagement",
    tags=["Engagement Agent"],
    dependencies=[Depends(verify_token)],  # SECURITY: Require authentication
)

app.include_router(
    reporting_router,
    prefix="/api/v1/reporting",
    tags=["Reporting Agent"],
    dependencies=[Depends(verify_token)],  # SECURITY: Require authentication
)

# Utility Routers (PROTECTED - Require Authentication)
app.include_router(
    factor_broker_router,
    prefix="/api/v1/factors",
    tags=["Factor Broker"],
    dependencies=[Depends(verify_token)],  # SECURITY: Require authentication
)

app.include_router(
    methodologies_router,
    prefix="/api/v1/methodologies",
    tags=["Methodologies"],
    dependencies=[Depends(verify_token)],  # SECURITY: Require authentication
)

app.include_router(
    connectors_router,
    prefix="/api/v1/connectors",
    tags=["ERP Connectors"],
    dependencies=[Depends(verify_token)],  # SECURITY: Require authentication
)


# ==============================================================================
# Prometheus Metrics
# ==============================================================================

# Setup Prometheus metrics instrumentation
# Note: Using custom metrics from services.metrics instead of default instrumentator
from services.metrics import get_metrics, create_metrics_endpoint

# Get global metrics instance
vcci_metrics = get_metrics()

# Add custom metrics endpoint
metrics_route = create_metrics_endpoint(vcci_metrics)
if metrics_route:
    app.add_api_route(**metrics_route)

# Also add default HTTP metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics/http")


# ==============================================================================
# Main Entry Point (for direct execution)
# ==============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.APP_ENV == "development",
        log_level="info",
        access_log=True,
    )
