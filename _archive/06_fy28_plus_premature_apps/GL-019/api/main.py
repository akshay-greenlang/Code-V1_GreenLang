"""
GL-019 HEATSCHEDULER FastAPI Application

Main application module for ProcessHeatingScheduler REST API.
Implements GreenLang standard patterns: JWT auth, rate limiting, audit trails.

Author: GL-APIDeveloper
Version: 1.0.0
"""

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("gl019.api")

# Prometheus metrics
REQUEST_COUNT = Counter(
    "gl019_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "gl019_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"]
)
SCHEDULE_OPTIMIZATIONS = Counter(
    "gl019_schedule_optimizations_total",
    "Total schedule optimization requests",
    ["status"]
)
ENERGY_SAVINGS = Counter(
    "gl019_energy_savings_kwh_total",
    "Total energy savings in kWh"
)

# Application metadata
APP_METADATA = {
    "title": "GL-019 HEATSCHEDULER API",
    "description": """
## ProcessHeatingScheduler REST API

Schedules process heating operations to minimize energy costs while meeting production requirements.

### Key Features

- **Schedule Optimization**: AI-powered scheduling to minimize energy costs
- **Production Integration**: Sync with ERP systems for production batch data
- **Tariff Management**: Handle time-of-use and demand-based pricing
- **Equipment Control**: Monitor and control heating equipment
- **Cost Analytics**: Detailed savings reports and forecasting
- **Demand Response**: Participate in grid demand response programs

### Authentication

All endpoints require authentication via:
- **OAuth2 Bearer Token**: For user-based access
- **API Key**: For system-to-system integration

### Rate Limits

- Standard tier: 100 requests/minute
- Premium tier: 1000 requests/minute
- Enterprise tier: Custom limits

### Support

For API support, contact: api-support@greenlang.io
""",
    "version": "1.0.0",
    "terms_of_service": "https://greenlang.io/terms",
    "contact": {
        "name": "GreenLang API Support",
        "url": "https://greenlang.io/support",
        "email": "api-support@greenlang.io"
    },
    "license_info": {
        "name": "Proprietary",
        "url": "https://greenlang.io/license"
    }
}

# OpenAPI tags
TAGS_METADATA = [
    {
        "name": "Health",
        "description": "Health check and readiness endpoints for orchestration platforms"
    },
    {
        "name": "Schedules",
        "description": "Create and manage optimized heating schedules"
    },
    {
        "name": "Production",
        "description": "Production batch integration and ERP synchronization"
    },
    {
        "name": "Tariffs",
        "description": "Energy tariff management and forecasting"
    },
    {
        "name": "Equipment",
        "description": "Heating equipment monitoring and control"
    },
    {
        "name": "Analytics",
        "description": "Cost analytics, savings reports, and forecasting"
    },
    {
        "name": "Demand Response",
        "description": "Grid demand response event handling"
    },
    {
        "name": "Metrics",
        "description": "Prometheus metrics for monitoring"
    }
]


# Application state
class ApplicationState:
    """Application state container for health checks."""

    def __init__(self):
        self.is_ready: bool = False
        self.startup_time: datetime = None
        self.last_health_check: datetime = None
        self.database_connected: bool = False
        self.cache_connected: bool = False
        self.scheduler_running: bool = False


app_state = ApplicationState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting GL-019 HEATSCHEDULER API...")
    app_state.startup_time = datetime.utcnow()

    # Initialize connections (simulated)
    try:
        # Database connection
        logger.info("Connecting to database...")
        app_state.database_connected = True

        # Cache connection
        logger.info("Connecting to cache...")
        app_state.cache_connected = True

        # Start scheduler
        logger.info("Starting scheduler service...")
        app_state.scheduler_running = True

        app_state.is_ready = True
        logger.info("GL-019 HEATSCHEDULER API started successfully")

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down GL-019 HEATSCHEDULER API...")
    app_state.is_ready = False
    app_state.scheduler_running = False
    app_state.cache_connected = False
    app_state.database_connected = False
    logger.info("GL-019 HEATSCHEDULER API shutdown complete")


def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance
    """
    application = FastAPI(
        **APP_METADATA,
        openapi_tags=TAGS_METADATA,
        docs_url=None,  # Custom docs endpoint
        redoc_url=None,  # Custom redoc endpoint
        openapi_url="/api/v1/openapi.json",
        lifespan=lifespan
    )

    # Add middlewares
    configure_middleware(application)

    # Add exception handlers
    configure_exception_handlers(application)

    # Add core routes
    configure_core_routes(application)

    # Import and include API routes
    from api.routes import router
    application.include_router(router)

    return application


def configure_middleware(app: FastAPI) -> None:
    """Configure application middleware."""

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "https://*.greenlang.io",
            "https://app.greenlang.io",
            "http://localhost:3000",  # Development
            "http://localhost:8080",  # Development
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
        max_age=600,
    )

    # GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Request timing middleware
    @app.middleware("http")
    async def timing_middleware(request: Request, call_next):
        """Add request timing and tracking."""
        start_time = time.time()
        request_id = request.headers.get("X-Request-ID", f"req_{int(time.time() * 1000)}")

        # Log request
        logger.info(f"[{request_id}] {request.method} {request.url.path}")

        # Process request
        response: Response = await call_next(request)

        # Calculate latency
        latency = time.time() - start_time

        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{latency:.3f}s"

        # Record metrics
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

        # Log response
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} "
            f"-> {response.status_code} ({latency:.3f}s)"
        )

        return response


def configure_exception_handlers(app: FastAPI) -> None:
    """Configure application exception handlers."""

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Handle all unhandled exceptions."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred",
                "request_id": request.headers.get("X-Request-ID")
            }
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        """Handle validation errors."""
        logger.warning(f"Validation error: {exc}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "validation_error",
                "message": str(exc),
                "request_id": request.headers.get("X-Request-ID")
            }
        )


def configure_core_routes(app: FastAPI) -> None:
    """Configure core application routes."""

    @app.get(
        "/health/liveness",
        tags=["Health"],
        summary="Liveness probe",
        description="Check if the application is alive. Used by Kubernetes liveness probe.",
        response_model=Dict[str, Any]
    )
    async def liveness_check() -> Dict[str, Any]:
        """
        Liveness probe endpoint.

        Returns 200 if the application process is running.
        Does not check dependencies.
        """
        return {
            "status": "alive",
            "timestamp": datetime.utcnow().isoformat(),
            "agent_id": "GL-019",
            "codename": "HEATSCHEDULER"
        }

    @app.get(
        "/health/readiness",
        tags=["Health"],
        summary="Readiness probe",
        description="Check if the application is ready to serve traffic. Used by Kubernetes readiness probe.",
        response_model=Dict[str, Any]
    )
    async def readiness_check() -> Dict[str, Any]:
        """
        Readiness probe endpoint.

        Returns 200 if the application is ready to serve traffic.
        Checks all critical dependencies.
        """
        app_state.last_health_check = datetime.utcnow()

        checks = {
            "database": app_state.database_connected,
            "cache": app_state.cache_connected,
            "scheduler": app_state.scheduler_running
        }

        all_healthy = all(checks.values())

        if not all_healthy:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "not_ready",
                    "timestamp": datetime.utcnow().isoformat(),
                    "checks": checks
                }
            )

        return {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": (
                datetime.utcnow() - app_state.startup_time
            ).total_seconds() if app_state.startup_time else 0,
            "checks": checks
        }

    @app.get(
        "/metrics",
        tags=["Metrics"],
        summary="Prometheus metrics",
        description="Prometheus-compatible metrics endpoint for monitoring.",
        response_class=Response
    )
    async def prometheus_metrics() -> Response:
        """
        Prometheus metrics endpoint.

        Returns metrics in Prometheus text format.
        """
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )

    @app.get(
        "/api/docs",
        tags=["Documentation"],
        include_in_schema=False
    )
    async def swagger_ui():
        """Swagger UI documentation."""
        return get_swagger_ui_html(
            openapi_url="/api/v1/openapi.json",
            title="GL-019 HEATSCHEDULER API - Documentation",
            swagger_favicon_url="https://greenlang.io/favicon.ico"
        )

    @app.get(
        "/api/redoc",
        tags=["Documentation"],
        include_in_schema=False
    )
    async def redoc():
        """ReDoc documentation."""
        return get_redoc_html(
            openapi_url="/api/v1/openapi.json",
            title="GL-019 HEATSCHEDULER API - Documentation",
            redoc_favicon_url="https://greenlang.io/favicon.ico"
        )

    @app.get(
        "/",
        tags=["Health"],
        summary="Root endpoint",
        description="API information and links",
        response_model=Dict[str, Any]
    )
    async def root() -> Dict[str, Any]:
        """Root endpoint with API information."""
        return {
            "name": "GL-019 HEATSCHEDULER API",
            "version": "1.0.0",
            "description": "ProcessHeatingScheduler - Schedules process heating operations to minimize energy costs",
            "documentation": {
                "swagger": "/api/docs",
                "redoc": "/api/redoc",
                "openapi": "/api/v1/openapi.json"
            },
            "health": {
                "liveness": "/health/liveness",
                "readiness": "/health/readiness"
            },
            "metrics": "/metrics"
        }


# Create the application instance
app = create_application()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )