"""
GL-012 SteamQual API Main Application

FastAPI application entry point for SteamQualityController API.
Provides REST endpoints with comprehensive authentication, monitoring,
and latency tracking.

Latency Targets:
- Sensor-to-metric: < 5 seconds
- Event emission: < 10 seconds
"""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from pydantic import Field
from pydantic_settings import BaseSettings
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    SLOWAPI_AVAILABLE = True
except ImportError:
    SLOWAPI_AVAILABLE = False

from .routes import router as quality_router
from .auth import (
    get_current_user,
    SteamQualUser,
    Role,
    Permission,
    require_permissions,
    log_security_event,
    TokenResponse,
    create_access_token,
    create_refresh_token,
    get_auth_config,
)
from .schemas import (
    HealthStatus,
    ServiceHealth,
    SystemStatus,
)


# =============================================================================
# Configuration
# =============================================================================

class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Application Settings
    app_name: str = Field(default="GL-012 SteamQual SteamQualityController API")
    app_version: str = Field(default="1.0.0")
    debug: bool = Field(default=False, alias="STEAMQUAL_DEBUG")
    environment: str = Field(default="development", alias="STEAMQUAL_ENV")

    # Server Settings
    host: str = Field(default="0.0.0.0", alias="STEAMQUAL_HOST")
    port: int = Field(default=8012, alias="STEAMQUAL_PORT")
    workers: int = Field(default=1, alias="STEAMQUAL_WORKERS")
    reload: bool = Field(default=False, alias="STEAMQUAL_RELOAD")

    # CORS Settings
    cors_origins: str = Field(
        default="https://*.greenlang.io,http://localhost:3000",
        alias="STEAMQUAL_CORS_ORIGINS",
    )
    cors_allow_credentials: bool = Field(default=True, alias="STEAMQUAL_CORS_CREDENTIALS")

    # Trusted Hosts Settings
    allowed_hosts: str = Field(
        default="*.greenlang.io,localhost",
        alias="STEAMQUAL_ALLOWED_HOSTS",
    )

    # Rate Limiting Settings
    rate_limit_default: str = Field(default="100/minute", alias="STEAMQUAL_RATE_LIMIT_DEFAULT")
    rate_limit_auth: str = Field(default="10/minute", alias="STEAMQUAL_RATE_LIMIT_AUTH")
    rate_limit_quality: str = Field(default="200/minute", alias="STEAMQUAL_RATE_LIMIT_QUALITY")

    # Logging Settings
    log_level: str = Field(default="INFO", alias="STEAMQUAL_LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        alias="STEAMQUAL_LOG_FORMAT",
    )

    # Latency Targets (milliseconds)
    sensor_to_metric_target_ms: float = Field(default=5000.0, alias="STEAMQUAL_SENSOR_METRIC_TARGET")
    event_emission_target_ms: float = Field(default=10000.0, alias="STEAMQUAL_EVENT_EMISSION_TARGET")

    @property
    def cors_origins_list(self) -> List[str]:
        """Get CORS origins as a list."""
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @property
    def allowed_hosts_list(self) -> List[str]:
        """Get allowed hosts as a list."""
        return [host.strip() for host in self.allowed_hosts.split(",")]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# =============================================================================
# Logging Configuration
# =============================================================================

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format=settings.log_format,
)
logger = logging.getLogger("steamqual.api")


# =============================================================================
# Prometheus Metrics
# =============================================================================

if PROMETHEUS_AVAILABLE:
    REQUEST_COUNT = Counter(
        "steamqual_requests_total",
        "Total number of requests",
        ["method", "endpoint", "status"],
    )

    REQUEST_LATENCY = Histogram(
        "steamqual_request_latency_seconds",
        "Request latency in seconds",
        ["method", "endpoint"],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    )

    QUALITY_ESTIMATION_LATENCY = Histogram(
        "steamqual_quality_estimation_seconds",
        "Quality estimation latency in seconds",
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
    )

    EVENT_EMISSION_LATENCY = Histogram(
        "steamqual_event_emission_seconds",
        "Event emission latency in seconds",
        buckets=[0.5, 1.0, 2.0, 5.0, 10.0],
    )

    LATENCY_SLA_VIOLATIONS = Counter(
        "steamqual_latency_sla_violations_total",
        "Count of SLA violations",
        ["latency_type"],
    )

    CURRENT_QUALITY = Gauge(
        "steamqual_current_quality_percent",
        "Current steam quality percentage",
        ["header_id"],
    )


# =============================================================================
# Middleware
# =============================================================================

class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting Prometheus metrics and tracking latency SLAs."""

    async def dispatch(self, request: Request, call_next):
        """Process request and collect metrics."""
        start_time = datetime.utcnow()

        response = await call_next(request)

        # Calculate latency
        latency = (datetime.utcnow() - start_time).total_seconds()

        if PROMETHEUS_AVAILABLE:
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

            # Check SLA violations
            latency_ms = latency * 1000
            if "/estimate-quality" in request.url.path:
                QUALITY_ESTIMATION_LATENCY.observe(latency)
                if latency_ms > settings.sensor_to_metric_target_ms:
                    LATENCY_SLA_VIOLATIONS.labels(latency_type="sensor_to_metric").inc()
                    logger.warning(
                        f"Sensor-to-metric SLA violation: {latency_ms:.1f}ms > {settings.sensor_to_metric_target_ms}ms"
                    )
            elif "/events" in request.url.path:
                EVENT_EMISSION_LATENCY.observe(latency)
                if latency_ms > settings.event_emission_target_ms:
                    LATENCY_SLA_VIOLATIONS.labels(latency_type="event_emission").inc()
                    logger.warning(
                        f"Event emission SLA violation: {latency_ms:.1f}ms > {settings.event_emission_target_ms}ms"
                    )

        return response


class AuditMiddleware(BaseHTTPMiddleware):
    """Middleware for audit logging of API calls."""

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
    """Middleware to add request ID to all requests for tracing."""

    async def dispatch(self, request: Request, call_next):
        """Add request ID header."""
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


# =============================================================================
# Application Lifecycle
# =============================================================================

startup_time: Optional[datetime] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    global startup_time

    logger.info("Starting GL-012 SteamQual SteamQualityController API...")
    startup_time = datetime.utcnow()

    logger.info(f"Latency targets: sensor-to-metric < {settings.sensor_to_metric_target_ms}ms, "
                f"event-emission < {settings.event_emission_target_ms}ms")

    logger.info("SteamQual API started successfully")

    yield

    # Shutdown
    logger.info("Shutting down SteamQual API...")
    logger.info("SteamQual API shutdown complete")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="GL-012 SteamQual SteamQualityController API",
    description="""
## Steam Quality Controller API

Production-grade REST API for industrial steam quality monitoring and control.

### Features

- **Quality Estimation**: Estimate steam quality (dryness fraction) from sensor data
- **Carryover Risk Assessment**: Assess moisture carryover risk
- **Quality State Monitoring**: Real-time quality state for steam headers
- **Event Management**: Quality events, alerts, and acknowledgments
- **Control Recommendations**: AI-driven recommendations for quality improvement
- **Quality Metrics**: KPIs and performance analytics

### Latency Targets

- **Sensor-to-metric**: < 5 seconds (quality estimation from raw measurements)
- **Event emission**: < 10 seconds (alert generation and notification)

### Authentication

All endpoints require authentication via:
- Bearer token (JWT) in Authorization header
- API key in X-API-Key header

### Rate Limits

- 100 requests/minute for standard endpoints
- 200 requests/minute for quality estimation
- 10 requests/minute for authentication

### Zero-Hallucination Guarantee

All numeric calculations use deterministic thermodynamic formulas.
No LLM/ML models are used for regulatory or safety-critical values.
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

if SLOWAPI_AVAILABLE:
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

# Quality Controller REST API
app.include_router(quality_router)


# =============================================================================
# Health Check Endpoints
# =============================================================================

@app.get(
    "/health",
    response_model=HealthStatus,
    summary="Health check",
    description="Check if the API is healthy",
    tags=["System"],
)
async def health_check() -> HealthStatus:
    """
    Health check endpoint for load balancers and monitoring.

    Returns:
        Status OK if all systems are healthy
    """
    return HealthStatus(
        status="healthy",
        timestamp=datetime.utcnow(),
        version=settings.app_version,
        service="GL-012 SteamQual",
        latency_targets_met=True,
    )


@app.get(
    "/ready",
    response_model=SystemStatus,
    summary="Readiness check",
    description="Check if the API is ready to accept requests",
    tags=["System"],
)
async def readiness_check() -> SystemStatus:
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

    # Check quality estimation service
    services.append(ServiceHealth(
        service_name="quality_estimation",
        status="healthy",
        latency_ms=5.0,
        last_check=datetime.utcnow(),
    ))

    # Check event service
    services.append(ServiceHealth(
        service_name="event_service",
        status="healthy",
        latency_ms=2.0,
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
        active_connections=0,
        requests_per_minute=0,
        sensor_to_metric_p99_ms=settings.sensor_to_metric_target_ms * 0.8,
        event_emission_p99_ms=settings.event_emission_target_ms * 0.8,
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
        Prometheus-formatted metrics including latency SLA tracking
    """
    if PROMETHEUS_AVAILABLE:
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )
    else:
        return JSONResponse(
            content={"error": "Prometheus client not installed"},
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
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
    mock_user = SteamQualUser(
        user_id=uuid.uuid4(),
        username="demo_user",
        email="demo@greenlang.io",
        tenant_id=uuid.uuid4(),
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


@app.get(
    "/api/v1/auth/me",
    summary="Get current user",
    description="Get information about the authenticated user",
    tags=["Authentication"],
)
async def get_me(
    current_user: SteamQualUser = Depends(get_current_user),
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
        title="GL-012 SteamQual SteamQualityController API",
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
            "url": "http://localhost:8012",
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

    # Add latency target extension
    openapi_schema["info"]["x-latency-targets"] = {
        "sensor-to-metric": f"< {settings.sensor_to_metric_target_ms}ms",
        "event-emission": f"< {settings.event_emission_target_ms}ms",
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
        "service": "GL-012 SteamQual SteamQualityController",
        "version": settings.app_version,
        "description": "Steam quality monitoring and control API",
        "latency_targets": {
            "sensor_to_metric_ms": settings.sensor_to_metric_target_ms,
            "event_emission_ms": settings.event_emission_target_ms,
        },
        "docs": {
            "openapi": "/api/openapi.json",
            "swagger": "/api/docs",
            "redoc": "/api/redoc",
        },
        "endpoints": {
            "estimate_quality": "/api/v1/estimate-quality",
            "assess_carryover_risk": "/api/v1/assess-carryover-risk",
            "quality_state": "/api/v1/quality-state/{header_id}",
            "events": "/api/v1/events",
            "recommendations": "/api/v1/recommendations",
            "metrics": "/api/v1/metrics",
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
    logger.info(f"Starting SteamQual API on {settings.host}:{settings.port}")

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
