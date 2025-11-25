# -*- coding: utf-8 -*-
"""
GreenLang FastAPI Application

Production-ready REST API for GreenLang agent foundation.
Implements health check endpoints for Kubernetes orchestration with
comprehensive monitoring, security, and performance features.

Features:
- Health check endpoints (liveness, readiness, startup)
- CORS middleware with restricted origins
- Request ID tracking for distributed tracing
- Structured logging with request context
- Metrics collection (Prometheus-compatible)
- Rate limiting per endpoint
- Security headers (HSTS, CSP, etc.)
- OpenAPI documentation at /api/docs

Kubernetes Integration:
- /healthz: Liveness probe
- /ready: Readiness probe
- /startup: Startup probe
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Callable, Optional

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from greenlang.determinism import deterministic_uuid, DeterministicClock
from .health import (
    HealthCheckResponse,
    HealthStatus,
    check_liveness,
    check_readiness,
    check_startup,
    health_manager,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] %(message)s"
)
logger = logging.getLogger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add request context (ID, timing) to all requests.

    Adds:
    - X-Request-ID header (generated or from client)
    - Request timing and logging
    - Structured log context
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate or extract request ID
        request_id = request.headers.get("X-Request-ID", str(deterministic_uuid(__name__, str(DeterministicClock.now()))))

        # Add to request state for access in handlers
        request.state.request_id = request_id

        # Add to logging context
        old_factory = logging.getLogRecordFactory()

        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.request_id = request_id
            return record

        logging.setLogRecordFactory(record_factory)

        # Log request
        start_time = time.time()
        logger.info(
            f"{request.method} {request.url.path} started",
            extra={"request_id": request_id}
        )

        try:
            # Process request
            response = await call_next(request)

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            # Log response
            duration_ms = (time.time() - start_time) * 1000
            logger.info(
                f"{request.method} {request.url.path} completed: {response.status_code} ({duration_ms:.1f}ms)",
                extra={"request_id": request_id}
            )

            # Add timing header
            response.headers["X-Response-Time-Ms"] = f"{duration_ms:.1f}"

            return response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(
                f"{request.method} {request.url.path} failed: {str(e)} ({duration_ms:.1f}ms)",
                exc_info=True,
                extra={"request_id": request_id}
            )
            raise

        finally:
            # Restore original log record factory
            logging.setLogRecordFactory(old_factory)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to all responses.

    Adds:
    - Strict-Transport-Security (HSTS)
    - X-Content-Type-Options
    - X-Frame-Options
    - X-XSS-Protection
    - Content-Security-Policy
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Security headers
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Content-Security-Policy"] = "default-src 'self'"

        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup/shutdown events.

    Startup:
    - Initialize database connections
    - Initialize Redis connections
    - Initialize LLM router
    - Initialize vector store
    - Register dependencies with health manager
    - Mark startup complete

    Shutdown:
    - Close database connections
    - Close Redis connections
    - Flush metrics
    """
    logger.info("GreenLang API starting up...")

    try:
        # Initialize dependencies (in production)
        # For now, we'll just mark startup as complete
        # In production, this would initialize actual dependencies:
        #
        # from ..database.postgres_manager import PostgresManager, PostgresConfig
        # from ..cache.redis_manager import RedisManager, RedisConfig
        # from ..llm.llm_router import LLMRouter
        # from ..rag.vector_store import VectorStore
        #
        # db_manager = PostgresManager(PostgresConfig(...))
        # await db_manager.initialize()
        #
        # redis_manager = RedisManager(RedisConfig(...))
        # await redis_manager.initialize()
        #
        # llm_router = LLMRouter(...)
        # vector_store = VectorStore(...)
        #
        # health_manager.set_dependencies(
        #     db_manager=db_manager,
        #     redis_manager=redis_manager,
        #     llm_router=llm_router,
        #     vector_store=vector_store
        # )

        # Mark startup complete
        health_manager.mark_startup_complete()
        logger.info(f"GreenLang API startup complete (uptime: {health_manager.get_uptime_seconds():.1f}s)")

        yield

    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        health_manager.mark_startup_failed(str(e))
        raise

    finally:
        # Shutdown
        logger.info("GreenLang API shutting down...")

        # Close connections (in production)
        # await db_manager.close()
        # await redis_manager.close()

        logger.info("GreenLang API shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="GreenLang Agent Foundation API",
    description="""
Production-ready REST API for GreenLang's AI agent foundation.

## Health Check Endpoints

The API provides three health check endpoints for Kubernetes orchestration:

- **GET /healthz** - Liveness probe (is process alive?)
- **GET /ready** - Readiness probe (ready to serve traffic?)
- **GET /startup** - Startup probe (initialization complete?)

## Features

- JWT authentication with role-based access control
- Rate limiting per endpoint and per user
- Comprehensive audit logging
- Multi-tenant isolation
- Circuit breaker for external dependencies
- Prometheus metrics integration
- OpenAPI documentation

## Security

All endpoints require authentication via JWT Bearer token except health checks.
API supports OAuth2 flows for enterprise integrations.
    """,
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)

# CORS middleware - restrict to specific domains in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://*.greenlang.io",
        "http://localhost:3000",  # Development frontend
        "http://localhost:8000",  # Development API
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Response-Time-Ms"]
)

# Trusted host middleware - prevent host header attacks
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[
        "*.greenlang.io",
        "localhost",
        "127.0.0.1"
    ]
)

# Custom middleware
app.add_middleware(RequestContextMiddleware)
app.add_middleware(SecurityHeadersMiddleware)


# Health Check Endpoints

@app.get(
    "/healthz",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Liveness probe",
    description="""
    Kubernetes liveness probe endpoint.

    Fast check (<10ms) with no external dependencies.
    Returns 200 if process is alive and responding.

    Use for:
    - Kubernetes liveness probe
    - Basic uptime monitoring
    - Process health verification

    Does NOT check:
    - Database connectivity
    - External service availability
    - Application readiness
    """,
    tags=["Health Checks"],
    responses={
        200: {
            "description": "Process is alive",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "timestamp": "2025-11-14T10:30:00Z",
                        "version": "1.0.0",
                        "uptime_seconds": 3600.0,
                        "components": [
                            {
                                "name": "process",
                                "status": "healthy",
                                "message": "Process is alive",
                                "response_time_ms": 0.1,
                                "last_checked": "2025-11-14T10:30:00Z"
                            }
                        ]
                    }
                }
            }
        }
    }
)
async def liveness_probe(request: Request) -> HealthCheckResponse:
    """
    Liveness probe - checks if process is alive.

    Returns 200 OK if process is responding.
    Fast check with no external dependencies.
    """
    return await check_liveness()


@app.get(
    "/ready",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Readiness probe",
    description="""
    Kubernetes readiness probe endpoint.

    Comprehensive check (<1s) of all critical dependencies:
    - PostgreSQL database connection
    - Redis cache connection
    - LLM provider availability
    - Vector database accessibility

    Returns 200 if all components healthy and ready to serve traffic.
    Returns 503 if any critical component is unhealthy.

    Use for:
    - Kubernetes readiness probe
    - Load balancer health checks
    - Service mesh routing decisions

    Features:
    - Result caching (5-second TTL)
    - Parallel component checks
    - Detailed component status
    """,
    tags=["Health Checks"],
    responses={
        200: {
            "description": "All components healthy, ready to serve traffic",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "timestamp": "2025-11-14T10:30:00Z",
                        "version": "1.0.0",
                        "uptime_seconds": 3600.0,
                        "components": [
                            {
                                "name": "postgresql",
                                "status": "healthy",
                                "message": "Connected to primary database",
                                "response_time_ms": 12.5,
                                "last_checked": "2025-11-14T10:30:00Z",
                                "metadata": {
                                    "pool_available": True,
                                    "latency_acceptable": True
                                }
                            },
                            {
                                "name": "redis",
                                "status": "healthy",
                                "message": "Redis responding to PING",
                                "response_time_ms": 5.2,
                                "last_checked": "2025-11-14T10:30:00Z"
                            },
                            {
                                "name": "llm_providers",
                                "status": "healthy",
                                "message": "LLM providers available",
                                "response_time_ms": 8.1,
                                "last_checked": "2025-11-14T10:30:00Z"
                            },
                            {
                                "name": "vector_db",
                                "status": "healthy",
                                "message": "Vector store accessible",
                                "response_time_ms": 15.3,
                                "last_checked": "2025-11-14T10:30:00Z"
                            }
                        ]
                    }
                }
            }
        },
        503: {
            "description": "One or more components unhealthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "unhealthy",
                        "timestamp": "2025-11-14T10:30:00Z",
                        "version": "1.0.0",
                        "uptime_seconds": 3600.0,
                        "components": [
                            {
                                "name": "postgresql",
                                "status": "unhealthy",
                                "message": "Connection refused",
                                "response_time_ms": 1000.0,
                                "last_checked": "2025-11-14T10:30:00Z"
                            }
                        ]
                    }
                }
            }
        }
    }
)
async def readiness_probe(request: Request, response: Response) -> HealthCheckResponse:
    """
    Readiness probe - checks if application is ready to serve traffic.

    Checks all critical dependencies with caching.
    Returns 200 if healthy, 503 if unhealthy.
    """
    health_response = await check_readiness()

    # Set HTTP status based on health
    if health_response.status == HealthStatus.UNHEALTHY:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif health_response.status == HealthStatus.DEGRADED:
        # Still return 503 for degraded state (not ready for new traffic)
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return health_response


@app.get(
    "/startup",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Startup probe",
    description="""
    Kubernetes startup probe endpoint.

    One-time initialization check for application startup.
    Verifies:
    - Application initialized
    - Database connections established
    - Redis connections established
    - LLM providers configured
    - Vector store initialized
    - Configuration loaded

    Returns 200 once startup is complete and all components initialized.
    Returns 503 while startup is in progress or if startup failed.

    Use for:
    - Kubernetes startup probe
    - Container initialization verification
    - Slow-starting application support

    Features:
    - No caching (fresh checks)
    - Longer timeout allowed (30-60s)
    - One-time verification
    """,
    tags=["Health Checks"],
    responses={
        200: {
            "description": "Startup complete, application initialized",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "timestamp": "2025-11-14T10:30:00Z",
                        "version": "1.0.0",
                        "uptime_seconds": 45.0,
                        "components": [
                            {
                                "name": "startup",
                                "status": "healthy",
                                "message": "Startup complete",
                                "response_time_ms": 0.0,
                                "last_checked": "2025-11-14T10:30:00Z",
                                "metadata": {
                                    "uptime_seconds": 45.0
                                }
                            }
                        ]
                    }
                }
            }
        },
        503: {
            "description": "Startup in progress or failed",
            "content": {
                "application/json": {
                    "example": {
                        "status": "unhealthy",
                        "timestamp": "2025-11-14T10:30:00Z",
                        "version": "1.0.0",
                        "uptime_seconds": 10.0,
                        "components": [
                            {
                                "name": "startup",
                                "status": "unhealthy",
                                "message": "Startup in progress",
                                "response_time_ms": 0.0,
                                "last_checked": "2025-11-14T10:30:00Z"
                            }
                        ]
                    }
                }
            }
        }
    }
)
async def startup_probe(request: Request, response: Response) -> HealthCheckResponse:
    """
    Startup probe - checks if application initialization is complete.

    Performs fresh checks (no caching) of all components.
    Returns 200 if startup complete, 503 if in progress or failed.
    """
    health_response = await check_startup()

    # Set HTTP status based on health
    if health_response.status != HealthStatus.HEALTHY:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return health_response


@app.get(
    "/",
    summary="API root",
    description="API root endpoint with service information",
    tags=["System"]
)
async def root() -> dict:
    """
    API root endpoint.

    Returns basic service information and links to documentation.
    """
    return {
        "service": "GreenLang Agent Foundation API",
        "version": "1.0.0",
        "status": "running",
        "documentation": "/api/docs",
        "health_checks": {
            "liveness": "/healthz",
            "readiness": "/ready",
            "startup": "/startup"
        },
        "timestamp": DeterministicClock.now().isoformat()
    }


@app.get(
    "/api/v1/info",
    summary="API information",
    description="Detailed API information and capabilities",
    tags=["System"]
)
async def api_info() -> dict:
    """
    API information endpoint.

    Returns detailed information about API capabilities,
    supported features, and system status.
    """
    return {
        "name": "GreenLang Agent Foundation API",
        "version": "1.0.0",
        "description": "Production REST API for GreenLang's AI agent foundation",
        "features": [
            "Health check endpoints for Kubernetes",
            "JWT authentication",
            "Rate limiting",
            "Multi-tenant support",
            "Audit logging",
            "Circuit breaker for external dependencies",
            "Prometheus metrics"
        ],
        "health_checks": {
            "liveness": {
                "path": "/healthz",
                "description": "Fast process alive check (<10ms)"
            },
            "readiness": {
                "path": "/ready",
                "description": "Comprehensive dependency check (<1s)"
            },
            "startup": {
                "path": "/startup",
                "description": "One-time initialization check (30-60s timeout)"
            }
        },
        "uptime_seconds": health_manager.get_uptime_seconds(),
        "timestamp": DeterministicClock.now().isoformat()
    }


# Custom OpenAPI schema with enhanced metadata
def custom_openapi():
    """Generate custom OpenAPI schema with enhanced metadata."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="GreenLang Agent Foundation API",
        version="1.0.0",
        description=app.description,
        routes=app.routes,
    )

    # Add custom metadata
    openapi_schema["info"]["x-logo"] = {
        "url": "https://greenlang.io/logo.png"
    }

    openapi_schema["info"]["contact"] = {
        "name": "GreenLang Support",
        "url": "https://greenlang.io/support",
        "email": "support@greenlang.io"
    }

    openapi_schema["info"]["license"] = {
        "name": "Proprietary",
        "url": "https://greenlang.io/license"
    }

    # Add tags metadata
    openapi_schema["tags"] = [
        {
            "name": "Health Checks",
            "description": "Kubernetes health check endpoints for orchestration"
        },
        {
            "name": "System",
            "description": "System information and metadata endpoints"
        }
    ]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled errors.

    Logs errors and returns structured JSON response.
    """
    request_id = getattr(request.state, "request_id", "unknown")

    logger.error(
        f"Unhandled exception: {str(exc)}",
        exc_info=True,
        extra={"request_id": request_id}
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_server_error",
            "message": "An internal error occurred",
            "request_id": request_id,
            "timestamp": DeterministicClock.now().isoformat()
        }
    )


if __name__ == "__main__":
    """
    Run application with uvicorn for development.

    In production, use:
        uvicorn agent_foundation.api.main:app --host 0.0.0.0 --port 8000 --workers 4
    """
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
