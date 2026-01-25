"""
GL-020 ECONOPULSE FastAPI Application

Main FastAPI application for Economizer Performance Monitoring.
Implements GreenLang standard patterns: JWT auth, rate limiting, audit trails.

Agent ID: GL-020
Codename: ECONOPULSE
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Callable

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from .routes import router
from .schemas import HealthStatus, ReadinessStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("gl-020-econopulse")

# Prometheus metrics
REQUEST_COUNT = Counter(
    "econopulse_requests_total",
    "Total number of requests",
    ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "econopulse_request_latency_seconds",
    "Request latency in seconds",
    ["method", "endpoint"]
)
ACTIVE_REQUESTS = Gauge(
    "econopulse_active_requests",
    "Number of active requests"
)
ECONOMIZER_COUNT = Gauge(
    "econopulse_monitored_economizers",
    "Number of monitored economizers"
)
ACTIVE_ALERTS = Gauge(
    "econopulse_active_alerts",
    "Number of active alerts"
)
FOULING_SCORE = Gauge(
    "econopulse_average_fouling_score",
    "Average fouling score across economizers"
)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    # Startup
    logger.info("GL-020 ECONOPULSE API starting up...")
    logger.info("Initializing database connections...")
    logger.info("Loading economizer configurations...")

    # Initialize metrics
    ECONOMIZER_COUNT.set(0)
    ACTIVE_ALERTS.set(0)
    FOULING_SCORE.set(0.0)

    yield

    # Shutdown
    logger.info("GL-020 ECONOPULSE API shutting down...")
    logger.info("Closing database connections...")


# Create FastAPI application
app = FastAPI(
    title="GL-020 ECONOPULSE API",
    description="""
## Economizer Performance Monitoring API

GL-020 ECONOPULSE provides comprehensive monitoring and analysis of economizer
performance in industrial boiler systems.

### Key Features

- **Performance Monitoring**: Real-time and historical performance metrics
- **Fouling Analysis**: Detect, predict, and track economizer fouling
- **Alert Management**: Configurable alerts with acknowledgment workflow
- **Efficiency Analysis**: Quantify efficiency losses and potential savings
- **Soot Blower Integration**: Trigger and optimize cleaning cycles
- **Reporting**: Daily, weekly, and custom reports with export capabilities

### Authentication

All endpoints require JWT Bearer token authentication. Obtain tokens from
the `/api/v1/auth/token` endpoint.

### Rate Limiting

- Standard endpoints: 1000 requests/minute
- Report endpoints: 100 requests/minute
- Export endpoints: 10 requests/minute

### Agent Information

- **Agent ID**: GL-020
- **Codename**: ECONOPULSE
- **Name**: EconomizerPerformanceAgent
""",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    openapi_tags=[
        {
            "name": "Health",
            "description": "Health check and readiness endpoints"
        },
        {
            "name": "Economizers",
            "description": "Economizer management and details"
        },
        {
            "name": "Performance",
            "description": "Performance monitoring and trends"
        },
        {
            "name": "Fouling",
            "description": "Fouling analysis and predictions"
        },
        {
            "name": "Alerts",
            "description": "Alert management and configuration"
        },
        {
            "name": "Efficiency",
            "description": "Efficiency metrics and savings analysis"
        },
        {
            "name": "Soot Blowers",
            "description": "Soot blower integration and cleaning"
        },
        {
            "name": "Reports",
            "description": "Report generation and export"
        },
        {
            "name": "Metrics",
            "description": "Prometheus metrics endpoint"
        }
    ],
    lifespan=lifespan,
    contact={
        "name": "GreenLang API Support",
        "email": "support@greenlang.io"
    },
    license_info={
        "name": "Proprietary",
        "url": "https://greenlang.io/license"
    }
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://*.greenlang.io",
        "https://localhost:3000",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-RateLimit-Remaining"]
)

# GZip compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Trusted host middleware (for production)
# app.add_middleware(
#     TrustedHostMiddleware,
#     allowed_hosts=["*.greenlang.io", "localhost"]
# )


@app.middleware("http")
async def request_middleware(request: Request, call_next: Callable) -> Response:
    """
    Middleware for request logging, tracing, and metrics.

    Adds:
    - Request ID for tracing
    - Request/response logging
    - Prometheus metrics collection
    - Request timing
    """
    # Generate request ID
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    # Start timing
    start_time = time.time()
    ACTIVE_REQUESTS.inc()

    # Log request
    logger.info(
        f"Request started",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "client_ip": get_remote_address(request)
        }
    )

    try:
        # Process request
        response = await call_next(request)

        # Calculate latency
        latency = time.time() - start_time

        # Update metrics
        endpoint = request.url.path
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status=response.status_code
        ).inc()
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=endpoint
        ).observe(latency)

        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{latency:.3f}s"

        # Log response
        logger.info(
            f"Request completed",
            extra={
                "request_id": request_id,
                "status_code": response.status_code,
                "latency_ms": round(latency * 1000, 2)
            }
        )

        return response

    except Exception as e:
        latency = time.time() - start_time
        logger.error(
            f"Request failed: {str(e)}",
            extra={
                "request_id": request_id,
                "error": str(e),
                "latency_ms": round(latency * 1000, 2)
            },
            exc_info=True
        )
        raise

    finally:
        ACTIVE_REQUESTS.dec()


# Health check endpoints
@app.get(
    "/health/liveness",
    response_model=HealthStatus,
    tags=["Health"],
    summary="Liveness probe",
    description="Kubernetes liveness probe endpoint. Returns 200 if the service is alive."
)
async def liveness_probe() -> HealthStatus:
    """
    Liveness probe for Kubernetes.

    Returns OK if the service is running. Does not check dependencies.
    """
    return HealthStatus(
        status="healthy",
        timestamp=datetime.utcnow(),
        service="gl-020-econopulse",
        version="1.0.0"
    )


@app.get(
    "/health/readiness",
    response_model=ReadinessStatus,
    tags=["Health"],
    summary="Readiness probe",
    description="Kubernetes readiness probe endpoint. Returns 200 if the service is ready to accept traffic."
)
async def readiness_probe() -> ReadinessStatus:
    """
    Readiness probe for Kubernetes.

    Checks all dependencies:
    - Database connectivity
    - Redis cache availability
    - External service health
    """
    # Check dependencies (mock implementation)
    checks = {
        "database": True,
        "redis": True,
        "historian": True,
        "message_queue": True
    }

    all_healthy = all(checks.values())

    if not all_healthy:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "checks": checks,
                "message": "One or more dependencies are unhealthy"
            }
        )

    return ReadinessStatus(
        status="ready",
        timestamp=datetime.utcnow(),
        checks=checks,
        message="All dependencies healthy"
    )


@app.get(
    "/metrics",
    tags=["Metrics"],
    summary="Prometheus metrics",
    description="Prometheus metrics endpoint for monitoring and alerting."
)
async def metrics():
    """
    Prometheus metrics endpoint.

    Exposes metrics for:
    - Request counts and latencies
    - Active requests
    - Economizer counts
    - Alert counts
    - Fouling scores
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# Include API routes
app.include_router(router, prefix="/api/v1")


# Global exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors."""
    logger.warning(f"Validation error: {exc}", extra={"request_id": getattr(request.state, 'request_id', 'unknown')})
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "validation_error",
            "message": str(exc),
            "request_id": getattr(request.state, 'request_id', 'unknown')
        }
    )


@app.exception_handler(PermissionError)
async def permission_error_handler(request: Request, exc: PermissionError):
    """Handle permission errors."""
    logger.warning(f"Permission denied: {exc}", extra={"request_id": getattr(request.state, 'request_id', 'unknown')})
    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN,
        content={
            "error": "forbidden",
            "message": str(exc),
            "request_id": getattr(request.state, 'request_id', 'unknown')
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    request_id = getattr(request.state, 'request_id', 'unknown')
    logger.error(f"Unexpected error: {exc}", extra={"request_id": request_id}, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_error",
            "message": "An unexpected error occurred",
            "request_id": request_id
        }
    )


# Root endpoint
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirect to docs."""
    return {
        "service": "GL-020 ECONOPULSE",
        "description": "Economizer Performance Monitoring API",
        "version": "1.0.0",
        "docs": "/api/docs",
        "health": "/health/liveness"
    }
