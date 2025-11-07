"""
GL-VCCI Scope 3 Carbon Intelligence Platform - Backend API
Main FastAPI Application Entry Point

This module initializes the FastAPI application and registers all routes,
middleware, and dependencies for the GL-VCCI platform.

Version: 2.0.0
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
import uvicorn

# Add parent directory to path to import services
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration
from config.settings import get_settings
from config.database import get_db, init_db
from config.redis_client import get_redis, init_redis
from config.logging_config import setup_logging

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

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()


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


# ==============================================================================
# Exception Handlers
# ==============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for all unhandled exceptions."""
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

# Core Agent Routers
app.include_router(
    intake_router,
    prefix="/api/v1/intake",
    tags=["Intake Agent"],
)

app.include_router(
    calculator_router,
    prefix="/api/v1/calculator",
    tags=["Calculator Agent"],
)

app.include_router(
    hotspot_router,
    prefix="/api/v1/hotspot",
    tags=["Hotspot Agent"],
)

app.include_router(
    engagement_router,
    prefix="/api/v1/engagement",
    tags=["Engagement Agent"],
)

app.include_router(
    reporting_router,
    prefix="/api/v1/reporting",
    tags=["Reporting Agent"],
)

# Utility Routers
app.include_router(
    factor_broker_router,
    prefix="/api/v1/factors",
    tags=["Factor Broker"],
)

app.include_router(
    methodologies_router,
    prefix="/api/v1/methodologies",
    tags=["Methodologies"],
)

app.include_router(
    connectors_router,
    prefix="/api/v1/connectors",
    tags=["ERP Connectors"],
)


# ==============================================================================
# Prometheus Metrics
# ==============================================================================

# Setup Prometheus metrics instrumentation
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


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
