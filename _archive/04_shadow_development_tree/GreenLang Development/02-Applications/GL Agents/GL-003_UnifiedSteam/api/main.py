"""
GL-003 UnifiedSteam API Main Application

FastAPI application entry point for SteamSystemOptimizer API.
Provides GraphQL, REST, and gRPC endpoints with comprehensive authentication and monitoring.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from .graphql_api import graphql_app, schema, data_store
from .rest_api import router as rest_router
from .api_auth import (
    get_current_user,
    SteamSystemUser,
    Role,
    Permission,
    require_permissions,
    log_security_event,
    TokenResponse,
    create_access_token,
    create_refresh_token,
    get_auth_config,
)
from .api_schemas import (
    SystemStatus,
    ServiceHealth,
)
from .grpc_services import serve_grpc, SteamPropertiesServicer
from .config import get_settings, Settings


# =============================================================================
# Logging Configuration
# =============================================================================

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format=settings.log_format,
)
logger = logging.getLogger("unifiedsteam.api")


# =============================================================================
# Prometheus Metrics
# =============================================================================

REQUEST_COUNT = Counter(
    "unifiedsteam_requests_total",
    "Total number of requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "unifiedsteam_request_latency_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
)

STEAM_PROPERTY_COUNT = Counter(
    "unifiedsteam_steam_property_requests_total",
    "Total number of steam property computations",
    ["result"],
)

OPTIMIZATION_COUNT = Counter(
    "unifiedsteam_optimization_requests_total",
    "Total number of optimization requests",
    ["type", "status"],
)

TRAP_DIAGNOSTICS_COUNT = Counter(
    "unifiedsteam_trap_diagnostics_total",
    "Total number of trap diagnostics requests",
    ["condition"],
)


# =============================================================================
# Middleware
# =============================================================================

class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting Prometheus metrics."""

    async def dispatch(self, request: Request, call_next):
        """Process request and collect metrics."""
        start_time = datetime.utcnow()

        response = await call_next(request)

        # Calculate latency
        latency = (datetime.utcnow() - start_time).total_seconds()

        # Record metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
        ).inc()

        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path,
        ).observe(latency)

        return response


class AuditMiddleware(BaseHTTPMiddleware):
    """Middleware for audit logging."""

    async def dispatch(self, request: Request, call_next):
        """Log request for audit purposes."""
        # Skip health check endpoints
        if request.url.path in ["/health", "/ready", "/metrics"]:
            return await call_next(request)

        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )

        response = await call_next(request)

        logger.info(
            f"Response: {request.method} {request.url.path} "
            f"status={response.status_code}"
        )

        return response


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Middleware to add request ID to all requests."""

    async def dispatch(self, request: Request, call_next):
        """Add request ID header."""
        import uuid
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


# =============================================================================
# Application Lifecycle
# =============================================================================

grpc_server = None
startup_time = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    global grpc_server, startup_time

    logger.info("Starting GL-003 UnifiedSteam SteamSystemOptimizer API...")
    startup_time = datetime.utcnow()

    # Start gRPC server if enabled
    if settings.grpc_enabled:
        try:
            grpc_server = await serve_grpc(
                host=settings.grpc_host,
                port=settings.grpc_port,
                max_workers=settings.grpc_max_workers,
                enable_reflection=settings.grpc_reflection_enabled,
            )
            logger.info(f"gRPC server started on {settings.grpc_host}:{settings.grpc_port}")
        except Exception as e:
            logger.error(f"Failed to start gRPC server: {e}")

    logger.info("UnifiedSteam API started successfully")

    yield

    # Shutdown
    logger.info("Shutting down UnifiedSteam API...")

    if grpc_server:
        await grpc_server.stop(grace=5)
        logger.info("gRPC server stopped")

    logger.info("UnifiedSteam API shutdown complete")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="GL-003 UnifiedSteam SteamSystemOptimizer API",
    description="""
## Steam System Optimization API

Production-grade GraphQL/gRPC/REST API for industrial steam system optimization.

### Features

- **Steam Properties**: IAPWS IF97 thermodynamic property calculations
- **Enthalpy Balance**: Equipment-level energy balance computations
- **Optimization**: Desuperheater, condensate recovery, and network optimization
- **Trap Diagnostics**: Steam trap condition monitoring and failure prediction
- **Root Cause Analysis**: Causal inference for event investigation
- **Explainability**: SHAP/LIME explanations for recommendations
- **KPI Dashboard**: Real-time performance metrics
- **Climate Impact**: Emissions tracking and sustainability reporting

### API Protocols

- **REST API**: Standard HTTP endpoints at `/api/v1/`
- **GraphQL API**: Full-featured GraphQL at `/graphql`
- **gRPC Services**: Low-latency RPC at port 50052

### Authentication

All endpoints require authentication via:
- Bearer token (JWT) in Authorization header
- API key in X-API-Key header
- mTLS client certificate (for service-to-service)

### Rate Limits

- 100 requests/minute for standard endpoints
- 500 requests/minute for steam properties
- 20 requests/minute for optimization requests
- 10 requests/minute for authentication
    """,
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)


# =============================================================================
# Rate Limiting
# =============================================================================

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# =============================================================================
# Middleware Configuration
# =============================================================================

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Trusted hosts
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.allowed_hosts_list,
)

# GZip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Custom middleware
app.add_middleware(MetricsMiddleware)
app.add_middleware(AuditMiddleware)
app.add_middleware(RequestIdMiddleware)


# =============================================================================
# Router Registration
# =============================================================================

# REST API
app.include_router(rest_router)

# GraphQL
app.include_router(graphql_app, prefix="/graphql", tags=["GraphQL"])


# =============================================================================
# Health Check Endpoints
# =============================================================================

@app.get(
    "/health",
    summary="Health check",
    description="Check if the API is healthy",
    tags=["System"],
)
async def health_check():
    """
    Health check endpoint for load balancers and monitoring.

    Returns:
        Status OK if all systems are healthy
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.app_version,
        "service": "GL-003 UnifiedSteam",
    }


@app.get(
    "/ready",
    summary="Readiness check",
    description="Check if the API is ready to accept requests",
    tags=["System"],
)
async def readiness_check():
    """
    Readiness check for Kubernetes-style deployments.

    Returns:
        Ready status and service health details
    """
    services = []

    # Check REST API
    services.append(ServiceHealth(
        service_name="rest_api",
        status="healthy",
        latency_ms=1.0,
        last_check=datetime.utcnow(),
    ))

    # Check GraphQL
    services.append(ServiceHealth(
        service_name="graphql",
        status="healthy",
        latency_ms=1.0,
        last_check=datetime.utcnow(),
    ))

    # Check gRPC
    grpc_status = "healthy" if grpc_server else "not_started"
    if not settings.grpc_enabled:
        grpc_status = "disabled"

    services.append(ServiceHealth(
        service_name="grpc",
        status=grpc_status,
        latency_ms=1.0,
        last_check=datetime.utcnow(),
    ))

    # Check data store / compute services
    try:
        # Try a simple property computation
        from .graphql_api import SteamPropertiesInput
        await data_store.get_steam_properties(SteamPropertiesInput(
            pressure_kpa=1000.0,
            temperature_c=200.0,
        ))
        compute_status = "healthy"
    except Exception as e:
        compute_status = "unhealthy"
        logger.error(f"Compute service health check failed: {e}")

    services.append(ServiceHealth(
        service_name="compute_engine",
        status=compute_status,
        latency_ms=5.0,
        last_check=datetime.utcnow(),
    ))

    overall_status = "healthy" if all(
        s.status in ["healthy", "disabled"] for s in services
    ) else "degraded"

    # Calculate uptime
    uptime_seconds = 0.0
    if startup_time:
        uptime_seconds = (datetime.utcnow() - startup_time).total_seconds()

    return SystemStatus(
        status=overall_status,
        version=settings.app_version,
        uptime_seconds=uptime_seconds,
        timestamp=datetime.utcnow(),
        services=services,
        active_connections=0,  # Would track actual connections in production
        requests_per_minute=0,  # Would track from metrics in production
    )


@app.get(
    "/metrics",
    summary="Prometheus metrics",
    description="Prometheus metrics endpoint",
    tags=["System"],
)
async def metrics():
    """
    Prometheus metrics endpoint.

    Returns:
        Prometheus-formatted metrics
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# =============================================================================
# Authentication Endpoints
# =============================================================================

@app.post(
    "/api/v1/auth/token",
    response_model=TokenResponse,
    summary="Get access token",
    description="Authenticate and receive JWT access token",
    tags=["Authentication"],
)
@limiter.limit(settings.rate_limit_auth)
async def login(request: Request):
    """
    Authenticate user and return JWT tokens.

    This is a simplified implementation. In production, this would:
    - Validate username/password against user database
    - Support OAuth2 flows (authorization code, client credentials)
    - Implement MFA if required

    Returns:
        JWT access and refresh tokens
    """
    config = get_auth_config()

    # Mock implementation - create demo user
    from uuid import uuid4
    mock_user = SteamSystemUser(
        user_id=uuid4(),
        username="demo_user",
        email="demo@greenlang.io",
        tenant_id=uuid4(),
        roles=[Role.OPERATOR],
        created_at=datetime.utcnow(),
    )

    access_token = create_access_token(mock_user, config)
    refresh_token = create_refresh_token(mock_user, config)

    await log_security_event(
        event_type="auth",
        action="login",
        resource_type="user",
        request=request,
        user=mock_user,
        success=True,
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=config.jwt_access_token_expire_minutes * 60,
    )


@app.post(
    "/api/v1/auth/refresh",
    response_model=TokenResponse,
    summary="Refresh access token",
    description="Get new access token using refresh token",
    tags=["Authentication"],
)
@limiter.limit(settings.rate_limit_auth)
async def refresh_token(request: Request, refresh_token: str):
    """
    Refresh access token using refresh token.

    Args:
        refresh_token: Valid refresh token

    Returns:
        New JWT access and refresh tokens
    """
    config = get_auth_config()

    # In production, validate refresh token and load user
    # This is a mock implementation
    from uuid import uuid4

    mock_user = SteamSystemUser(
        user_id=uuid4(),
        username="demo_user",
        email="demo@greenlang.io",
        tenant_id=uuid4(),
        roles=[Role.OPERATOR],
        created_at=datetime.utcnow(),
    )

    access_token = create_access_token(mock_user, config)
    new_refresh_token = create_refresh_token(mock_user, config)

    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=config.jwt_access_token_expire_minutes * 60,
    )


@app.get(
    "/api/v1/auth/me",
    summary="Get current user",
    description="Get information about the authenticated user",
    tags=["Authentication"],
)
async def get_me(
    current_user: SteamSystemUser = Depends(get_current_user),
):
    """
    Get current user information.

    Returns:
        User profile and permissions
    """
    return {
        "user_id": str(current_user.user_id),
        "username": current_user.username,
        "email": current_user.email,
        "tenant_id": str(current_user.tenant_id),
        "roles": [role.value for role in current_user.roles],
        "permissions": [perm.value for perm in current_user.get_all_permissions()],
        "auth_method": current_user.auth_method,
    }


# =============================================================================
# Exception Handlers
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured response."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat(),
            "path": request.url.path,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat(),
            "path": request.url.path,
        },
    )


@app.exception_handler(ValueError)
async def validation_exception_handler(request: Request, exc: ValueError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Validation error",
            "detail": str(exc),
            "status_code": 400,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


# =============================================================================
# Custom OpenAPI Schema
# =============================================================================

def custom_openapi():
    """Generate custom OpenAPI schema."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="GL-003 UnifiedSteam SteamSystemOptimizer API",
        version="1.0.0",
        description=app.description,
        routes=app.routes,
    )

    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT access token from /api/v1/auth/token",
        },
        "apiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for service authentication",
        },
    }

    # Add global security
    openapi_schema["security"] = [
        {"bearerAuth": []},
        {"apiKeyAuth": []},
    ]

    # Add server info
    openapi_schema["servers"] = [
        {
            "url": "https://api.greenlang.io",
            "description": "Production server",
        },
        {
            "url": "https://staging-api.greenlang.io",
            "description": "Staging server",
        },
        {
            "url": "http://localhost:8000",
            "description": "Local development",
        },
    ]

    # Add contact info
    openapi_schema["info"]["contact"] = {
        "name": "GreenLang API Support",
        "email": "api-support@greenlang.io",
        "url": "https://docs.greenlang.io/api",
    }

    # Add license
    openapi_schema["info"]["license"] = {
        "name": "Proprietary",
        "url": "https://greenlang.io/license",
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# =============================================================================
# Root Endpoint
# =============================================================================

@app.get(
    "/",
    summary="API Root",
    description="API information and available endpoints",
    tags=["System"],
)
async def root():
    """
    API root endpoint with service information.
    """
    return {
        "service": "GL-003 UnifiedSteam SteamSystemOptimizer",
        "version": settings.app_version,
        "description": "Steam system optimization API",
        "docs": {
            "openapi": "/api/openapi.json",
            "swagger": "/api/docs",
            "redoc": "/api/redoc",
            "graphql": "/graphql",
        },
        "endpoints": {
            "rest": "/api/v1/",
            "graphql": "/graphql",
            "grpc": f"{settings.grpc_host}:{settings.grpc_port}",
        },
        "health": "/health",
        "ready": "/ready",
        "metrics": "/metrics",
    }


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run the application."""
    logger.info(f"Starting UnifiedSteam API on {settings.host}:{settings.port}")

    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
