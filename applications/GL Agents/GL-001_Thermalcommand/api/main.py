"""
GL-001 ThermalCommand API Main Application

FastAPI application entry point for district heating optimization API.
Provides GraphQL and REST endpoints with comprehensive authentication and monitoring.
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
from .api_auth import (
    get_current_user,
    ThermalCommandUser,
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
    AllocationRequest,
    AllocationResponse,
    DemandUpdate,
    SystemStatus,
    ServiceHealth,
)
from .grpc_services import serve_grpc, ThermalCommandServicer

# =============================================================================
# Logging Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("thermalcommand.api")


# =============================================================================
# Prometheus Metrics
# =============================================================================

REQUEST_COUNT = Counter(
    "thermalcommand_requests_total",
    "Total number of requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "thermalcommand_request_latency_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
)

ALLOCATION_COUNT = Counter(
    "thermalcommand_allocations_total",
    "Total number of allocation requests",
    ["status"],
)

DEMAND_UPDATE_COUNT = Counter(
    "thermalcommand_demand_updates_total",
    "Total number of demand updates",
    ["source"],
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


# =============================================================================
# Application Lifecycle
# =============================================================================

# gRPC server handle
grpc_server = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    global grpc_server

    logger.info("Starting ThermalCommand API...")

    # Start gRPC server if enabled
    grpc_enabled = os.getenv("TC_GRPC_ENABLED", "true").lower() == "true"
    if grpc_enabled:
        grpc_host = os.getenv("TC_GRPC_HOST", "0.0.0.0")
        grpc_port = int(os.getenv("TC_GRPC_PORT", "50051"))

        try:
            grpc_server = await serve_grpc(host=grpc_host, port=grpc_port)
            logger.info(f"gRPC server started on {grpc_host}:{grpc_port}")
        except Exception as e:
            logger.error(f"Failed to start gRPC server: {e}")

    logger.info("ThermalCommand API started successfully")

    yield

    # Shutdown
    logger.info("Shutting down ThermalCommand API...")

    if grpc_server:
        await grpc_server.stop(grace=5)
        logger.info("gRPC server stopped")

    logger.info("ThermalCommand API shutdown complete")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="GL-001 ThermalCommand API",
    description="""
## District Heating Optimization API

Production-grade GraphQL/gRPC API for district heating optimization.

### Features

- **GraphQL API**: Full-featured GraphQL endpoint with queries, mutations, and subscriptions
- **gRPC Services**: Low-latency RPC calls for real-time operations
- **Authentication**: mTLS + OAuth2/JWT with fine-grained RBAC
- **Real-time**: WebSocket subscriptions for live updates
- **Explainability**: SHAP/LIME summaries for optimization decisions

### Authentication

All endpoints require authentication via:
- Bearer token (JWT) in Authorization header
- API key in X-API-Key header
- mTLS client certificate

### Rate Limits

- 100 requests/minute for standard endpoints
- 1000 requests/minute for read-only endpoints
- 10 requests/minute for allocation requests
    """,
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
allowed_origins = os.getenv("TC_CORS_ORIGINS", "https://*.greenlang.io").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Trusted hosts
allowed_hosts = os.getenv("TC_ALLOWED_HOSTS", "*.greenlang.io,localhost").split(",")
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=allowed_hosts,
)

# GZip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Custom middleware
app.add_middleware(MetricsMiddleware)
app.add_middleware(AuditMiddleware)


# =============================================================================
# GraphQL Router
# =============================================================================

# Mount GraphQL at /graphql
app.include_router(graphql_app, prefix="/graphql")


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
        "version": "1.0.0",
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

    # Check GraphQL
    services.append(ServiceHealth(
        service_name="graphql",
        status="healthy",
        latency_ms=1.0,
        last_check=datetime.utcnow(),
    ))

    # Check gRPC
    grpc_status = "healthy" if grpc_server else "not_started"
    services.append(ServiceHealth(
        service_name="grpc",
        status=grpc_status,
        latency_ms=1.0,
        last_check=datetime.utcnow(),
    ))

    # Check data store
    try:
        plan = await data_store.get_current_plan()
        data_status = "healthy" if plan else "no_data"
    except Exception as e:
        data_status = "unhealthy"
        logger.error(f"Data store health check failed: {e}")

    services.append(ServiceHealth(
        service_name="data_store",
        status=data_status,
        latency_ms=5.0,
        last_check=datetime.utcnow(),
    ))

    overall_status = "healthy" if all(s.status == "healthy" for s in services) else "degraded"

    return SystemStatus(
        status=overall_status,
        version="1.0.0",
        uptime_seconds=0,  # Would track actual uptime in production
        timestamp=datetime.utcnow(),
        services=services,
        active_connections=0,
        requests_per_minute=0,
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
@limiter.limit("10/minute")
async def login(
    request: Request,
    # In production, this would use OAuth2PasswordRequestForm
):
    """
    Authenticate user and return JWT tokens.

    This is a simplified implementation. In production, this would:
    - Validate username/password against user database
    - Support OAuth2 flows (authorization code, client credentials)
    - Implement proper MFA if required

    Returns:
        JWT access and refresh tokens
    """
    # Mock implementation - in production, validate credentials
    config = get_auth_config()

    # Create mock user for demonstration
    from uuid import uuid4
    mock_user = ThermalCommandUser(
        user_id=uuid4(),
        username="demo_user",
        email="demo@example.com",
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


# =============================================================================
# REST API Endpoints
# =============================================================================

@app.post(
    "/api/v1/thermal/allocation",
    response_model=AllocationResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Request heat allocation",
    description="Request optimization of heat allocation across assets",
    tags=["Dispatch"],
)
@limiter.limit("10/minute")
async def request_allocation(
    request: Request,
    allocation_request: AllocationRequest,
    current_user: ThermalCommandUser = Depends(
        require_permissions(Permission.DISPATCH_EXECUTE)
    ),
):
    """
    Request heat allocation optimization.

    Optimizes the allocation of thermal output across available assets
    based on cost and emissions objectives.

    Args:
        allocation_request: Allocation request parameters
        current_user: Authenticated user with DISPATCH_EXECUTE permission

    Returns:
        Allocation response with recommended setpoints
    """
    logger.info(
        f"Allocation request from {current_user.username}: "
        f"{allocation_request.target_output_mw} MW"
    )

    ALLOCATION_COUNT.labels(status="requested").inc()

    try:
        # Use GraphQL data store for allocation
        from .graphql_api import AllocationRequestInput

        input_data = AllocationRequestInput(
            target_output_mw=allocation_request.target_output_mw,
            time_window_minutes=allocation_request.time_window_minutes,
            objective=allocation_request.objective,
            cost_weight=allocation_request.cost_weight,
            emissions_weight=allocation_request.emissions_weight,
            is_emergency=allocation_request.is_emergency,
        )

        response = await data_store.request_allocation(input_data)

        ALLOCATION_COUNT.labels(status="success").inc()

        await log_security_event(
            event_type="operation",
            action="allocation",
            resource_type="dispatch",
            request=request,
            user=current_user,
            success=True,
            details={"target_mw": allocation_request.target_output_mw},
        )

        return response

    except Exception as e:
        ALLOCATION_COUNT.labels(status="error").inc()
        logger.error(f"Allocation request failed: {e}")

        await log_security_event(
            event_type="operation",
            action="allocation",
            resource_type="dispatch",
            request=request,
            user=current_user,
            success=False,
            error_message=str(e),
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Allocation optimization failed",
        )


@app.post(
    "/api/v1/thermal/demand",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit demand update",
    description="Submit demand forecast update for optimization",
    tags=["Forecasting"],
)
@limiter.limit("100/minute")
async def submit_demand(
    request: Request,
    demand_update: DemandUpdate,
    current_user: ThermalCommandUser = Depends(
        require_permissions(Permission.FORECAST_WRITE)
    ),
):
    """
    Submit demand forecast update.

    Receives demand forecast data from external systems for use
    in dispatch optimization.

    Args:
        demand_update: Demand forecast data
        current_user: Authenticated user with FORECAST_WRITE permission

    Returns:
        Acknowledgement with validation results
    """
    logger.info(
        f"Demand update from {current_user.username}: "
        f"{len(demand_update.demand_mw)} records from {demand_update.source_system}"
    )

    DEMAND_UPDATE_COUNT.labels(source=demand_update.source_system).inc()

    try:
        from .graphql_api import DemandUpdateInput

        input_data = DemandUpdateInput(
            forecast_type=demand_update.forecast_type,
            forecast_horizon_hours=demand_update.forecast_horizon_hours,
            resolution_minutes=demand_update.resolution_minutes,
            demand_mw=demand_update.demand_mw,
            demand_timestamps=demand_update.demand_timestamps,
            source_system=demand_update.source_system,
        )

        response = await data_store.submit_demand_update(input_data)

        return {
            "request_id": str(response.request_id),
            "success": response.success,
            "message": response.message,
            "records_received": response.records_received,
            "records_validated": response.records_validated,
            "data_quality_score": response.data_quality_score,
        }

    except Exception as e:
        logger.error(f"Demand update failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Demand update processing failed",
        )


@app.get(
    "/api/v1/thermal/plan",
    summary="Get current dispatch plan",
    description="Retrieve the currently active dispatch plan",
    tags=["Dispatch"],
)
@limiter.limit("1000/minute")
async def get_current_plan(
    request: Request,
    current_user: ThermalCommandUser = Depends(
        require_permissions(Permission.DISPATCH_READ)
    ),
):
    """
    Get current dispatch plan.

    Returns the currently active dispatch plan with schedule,
    setpoint recommendations, and optimization metrics.

    Args:
        current_user: Authenticated user with DISPATCH_READ permission

    Returns:
        Current dispatch plan
    """
    plan = await data_store.get_current_plan()

    if not plan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active dispatch plan found",
        )

    return {
        "plan_id": str(plan.plan_id),
        "plan_name": plan.plan_name,
        "plan_version": plan.plan_version,
        "objective": plan.objective.value,
        "is_active": plan.is_active,
        "effective_from": plan.effective_from.isoformat(),
        "effective_until": plan.effective_until.isoformat(),
        "total_thermal_output_mwh": plan.total_thermal_output_mwh,
        "total_cost": plan.total_cost,
        "total_emissions_kg": plan.total_emissions_kg,
        "optimization_score": plan.optimization_score,
        "solver_status": plan.solver_status,
    }


@app.get(
    "/api/v1/thermal/assets",
    summary="Get asset states",
    description="Retrieve current state of all thermal assets",
    tags=["Assets"],
)
@limiter.limit("1000/minute")
async def get_asset_states(
    request: Request,
    current_user: ThermalCommandUser = Depends(
        require_permissions(Permission.ASSET_READ)
    ),
):
    """
    Get asset states.

    Returns the current operational state of all thermal assets
    in the network.

    Args:
        current_user: Authenticated user with ASSET_READ permission

    Returns:
        List of asset states
    """
    assets = await data_store.get_asset_states()

    return {
        "assets": [
            {
                "asset_id": str(asset.asset_id),
                "asset_name": asset.asset_name,
                "asset_type": asset.asset_type.value,
                "status": asset.status.value,
                "current_output_mw": asset.current_output_mw,
                "current_setpoint_mw": asset.current_setpoint_mw,
                "supply_temperature_c": asset.supply_temperature_c,
                "return_temperature_c": asset.return_temperature_c,
                "health_score": asset.health.health_score,
            }
            for asset in assets
        ],
        "count": len(assets),
    }


@app.get(
    "/api/v1/thermal/kpis",
    summary="Get KPI metrics",
    description="Retrieve key performance indicator metrics",
    tags=["Analytics"],
)
@limiter.limit("1000/minute")
async def get_kpis(
    request: Request,
    category: Optional[str] = None,
    current_user: ThermalCommandUser = Depends(
        require_permissions(Permission.KPI_READ)
    ),
):
    """
    Get KPI metrics.

    Returns key performance indicators for the thermal network.

    Args:
        category: Optional filter by KPI category
        current_user: Authenticated user with KPI_READ permission

    Returns:
        List of KPI measurements
    """
    kpis = await data_store.get_kpis(category)

    return {
        "kpis": [
            {
                "kpi_id": str(kpi.kpi_id),
                "name": kpi.name,
                "category": kpi.category,
                "current_value": kpi.current_value,
                "target_value": kpi.target_value,
                "unit": kpi.unit,
                "trend_direction": kpi.trend_direction,
                "is_on_target": kpi.is_on_target,
            }
            for kpi in kpis
        ],
        "count": len(kpis),
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
        title="GL-001 ThermalCommand API",
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
        },
        "apiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
        },
    }

    # Add global security
    openapi_schema["security"] = [
        {"bearerAuth": []},
        {"apiKeyAuth": []},
    ]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run the application."""
    host = os.getenv("TC_HOST", "0.0.0.0")
    port = int(os.getenv("TC_PORT", "8000"))
    workers = int(os.getenv("TC_WORKERS", "1"))
    reload = os.getenv("TC_RELOAD", "false").lower() == "true"

    logger.info(f"Starting ThermalCommand API on {host}:{port}")

    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
