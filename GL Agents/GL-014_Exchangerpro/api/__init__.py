"""
GL-014 EXCHANGERPRO - REST API Module

FastAPI-based REST API for heat exchanger optimization.

This module provides a production-grade REST API with:
- Thermal KPI computation (Q, UA, LMTD, epsilon, NTU, delta-P)
- Fouling prediction and forecasting
- Cleaning schedule optimization
- What-if scenario analysis
- Explainability endpoints (LIME, SHAP, natural language)
- Full audit trail for compliance
- Prometheus metrics

All API responses include:
- computation_hash for traceability (SHA-256)
- timestamp (UTC)
- agent_version
- warnings array

Zero-Hallucination Principle:
    All thermal calculations are performed by the deterministic engine.
    The LLM is used only for natural language explanations and summaries.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, Optional
import logging
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi

from .routes import router
from .middleware import (
    RequestLoggingMiddleware,
    RateLimitMiddleware,
    AuthenticationMiddleware,
    ErrorHandlingMiddleware,
    ProvenanceMiddleware,
    AuditLoggingMiddleware,
    RequestValidationMiddleware,
)
from .dependencies import get_settings, Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Agent metadata
__version__ = "1.0.0"
__agent_id__ = "GL-014"
__agent_name__ = "EXCHANGERPRO"
__full_name__ = "Heat Exchanger Optimizer API"


# =============================================================================
# Application Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Handles startup and shutdown events:
    - Startup: Initialize services, load models, connect to databases
    - Shutdown: Clean up resources, close connections
    """
    # Startup
    logger.info(f"Starting {__agent_name__} API v{__version__}")
    logger.info(f"Agent ID: {__agent_id__}")

    # Initialize settings
    settings = get_settings()
    logger.info(f"Environment: {'production' if settings.require_authentication else 'development'}")

    # Load ML models (placeholder)
    logger.info("Loading fouling prediction model...")
    # await load_fouling_model(settings.fouling_model_path)

    # Initialize database connections (placeholder)
    logger.info("Initializing database connections...")
    # await init_database(settings.database_url)

    # Initialize cache (placeholder)
    logger.info("Initializing cache...")
    # await init_cache(settings.redis_url)

    logger.info(f"{__agent_name__} API startup complete")

    yield

    # Shutdown
    logger.info(f"Shutting down {__agent_name__} API...")

    # Close connections (placeholder)
    # await close_database()
    # await close_cache()

    logger.info(f"{__agent_name__} API shutdown complete")


# =============================================================================
# Custom OpenAPI Schema
# =============================================================================

def custom_openapi():
    """Generate custom OpenAPI schema with agent metadata."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=f"{__agent_name__} API",
        version=__version__,
        description=f"""
## {__full_name__}

Production REST API for industrial heat exchanger performance monitoring,
fouling prediction, and cleaning schedule optimization.

### Agent Information
- **Agent ID**: {__agent_id__}
- **Version**: {__version__}
- **Category**: Thermal/Optimization
- **Standards**: TEMA, ASME

### Zero-Hallucination Principle
All thermal calculations (Q, UA, LMTD, epsilon, NTU, delta-P) are performed
by deterministic calculation engines. The LLM is used only for natural
language explanations and formatting recommendations.

### Traceability
Every API response includes:
- `computation_hash`: SHA-256 hash of computation inputs
- `timestamp`: UTC timestamp of computation
- `agent_version`: Version of the API
- `warnings`: Array of warning messages

### Authentication
Supports API key (X-API-Key header) and JWT Bearer token authentication.

### Rate Limiting
Default: 60 requests/minute per client IP or API key.
        """,
        routes=app.routes,
    )

    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for authentication",
        },
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT token for authentication",
        },
    }

    # Add server information
    openapi_schema["servers"] = [
        {
            "url": "/api",
            "description": "API base path",
        },
    ]

    # Add tags metadata
    openapi_schema["tags"] = [
        {
            "name": "Thermal Calculations",
            "description": "Compute and retrieve thermal KPIs",
        },
        {
            "name": "Fouling Prediction",
            "description": "Predict and forecast fouling progression",
        },
        {
            "name": "Cleaning Optimization",
            "description": "Optimize cleaning schedules and get recommendations",
        },
        {
            "name": "Analysis",
            "description": "What-if scenario analysis",
        },
        {
            "name": "Explainability",
            "description": "Get explanations for computation results",
        },
        {
            "name": "Audit",
            "description": "Audit trail for compliance",
        },
        {
            "name": "System",
            "description": "Health checks and metrics",
        },
    ]

    # Add contact and license info
    openapi_schema["info"]["contact"] = {
        "name": "GreenLang Support",
        "email": "support@greenlang.io",
    }
    openapi_schema["info"]["license"] = {
        "name": "Proprietary",
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


# =============================================================================
# Create Application
# =============================================================================

def create_app(
    debug: bool = False,
    settings: Optional[Settings] = None,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        debug: Enable debug mode
        settings: Application settings (uses default if not provided)

    Returns:
        Configured FastAPI application
    """
    if settings is None:
        settings = get_settings()

    # Create FastAPI app
    application = FastAPI(
        title=f"{__agent_name__} API",
        description=f"REST API for {__full_name__}",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if debug or not settings.require_authentication else None,
        redoc_url="/redoc" if debug or not settings.require_authentication else None,
        openapi_url="/openapi.json" if debug or not settings.require_authentication else None,
        debug=debug,
    )

    # Add middleware (order matters - first added is outermost)

    # 1. Error handling (outermost to catch all errors)
    application.add_middleware(
        ErrorHandlingMiddleware,
        debug=debug,
        include_stack_trace=debug,
    )

    # 2. Request validation
    application.add_middleware(
        RequestValidationMiddleware,
        max_request_size_bytes=10 * 1024 * 1024,  # 10MB
    )

    # 3. Provenance tracking
    application.add_middleware(
        ProvenanceMiddleware,
        agent_id=__agent_id__,
        agent_name=__agent_name__,
        agent_version=__version__,
    )

    # 4. Audit logging
    application.add_middleware(
        AuditLoggingMiddleware,
        log_to_file=True,
        audit_file_path=settings.audit_log_path,
    )

    # 5. Authentication
    api_keys = _load_api_keys()
    application.add_middleware(
        AuthenticationMiddleware,
        api_keys=api_keys,
        jwt_secret=settings.jwt_secret if settings.jwt_secret else None,
        require_auth=settings.require_authentication,
    )

    # 6. Rate limiting
    application.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=settings.rate_limit_requests_per_minute,
        burst_size=settings.rate_limit_burst_size,
    )

    # 7. Request logging (innermost custom middleware)
    application.add_middleware(
        RequestLoggingMiddleware,
        log_request_body=settings.log_request_body,
    )

    # 8. CORS (should be after custom middleware)
    allowed_origins = os.getenv("CORS_ORIGINS", "*").split(",")
    application.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=[
            "X-Request-ID",
            "X-Correlation-ID",
            "X-Process-Time-Ms",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
            "X-GL-Agent-ID",
            "X-GL-Agent-Version",
        ],
    )

    # 9. Trusted hosts (optional, for production)
    trusted_hosts = os.getenv("TRUSTED_HOSTS", "").split(",")
    if trusted_hosts and trusted_hosts[0]:
        application.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=trusted_hosts,
        )

    # Include API router
    application.include_router(router, prefix="/api")

    # Set custom OpenAPI schema
    application.openapi = custom_openapi

    # Add exception handlers
    @application.exception_handler(404)
    async def not_found_handler(request: Request, exc):
        return JSONResponse(
            status_code=404,
            content={
                "error": "not_found",
                "message": f"Endpoint {request.url.path} not found",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    @application.exception_handler(405)
    async def method_not_allowed_handler(request: Request, exc):
        return JSONResponse(
            status_code=405,
            content={
                "error": "method_not_allowed",
                "message": f"Method {request.method} not allowed for {request.url.path}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    @application.exception_handler(422)
    async def validation_error_handler(request: Request, exc):
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "message": "Request validation failed",
                "details": exc.errors() if hasattr(exc, "errors") else str(exc),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    return application


def _load_api_keys() -> Dict[str, dict]:
    """
    Load API keys from environment or configuration.

    In production, this would load from a secure vault or database.
    """
    api_keys = {}

    # Load from environment (comma-separated KEY:USER_ID:TENANT_ID)
    keys_env = os.getenv("API_KEYS", "")
    if keys_env:
        for key_spec in keys_env.split(","):
            parts = key_spec.strip().split(":")
            if len(parts) >= 2:
                api_key = parts[0]
                user_id = parts[1]
                tenant_id = parts[2] if len(parts) > 2 else None
                roles = parts[3].split(";") if len(parts) > 3 else ["user"]

                api_keys[api_key] = {
                    "user_id": user_id,
                    "tenant_id": tenant_id,
                    "roles": roles,
                    "key_id": f"key_{user_id[:8]}",
                }

    # Add development key if in debug mode
    if os.getenv("DEBUG", "false").lower() == "true":
        api_keys["dev-api-key-12345"] = {
            "user_id": "dev_user",
            "tenant_id": "dev_tenant",
            "roles": ["admin", "user"],
            "key_id": "key_dev",
        }

    return api_keys


# =============================================================================
# Application Instance
# =============================================================================

# Create default application instance
app = create_app(
    debug=os.getenv("DEBUG", "false").lower() == "true",
)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Application
    "app",
    "create_app",
    # Metadata
    "__version__",
    "__agent_id__",
    "__agent_name__",
    "__full_name__",
    # Router
    "router",
    # Middleware
    "RequestLoggingMiddleware",
    "RateLimitMiddleware",
    "AuthenticationMiddleware",
    "ErrorHandlingMiddleware",
    "ProvenanceMiddleware",
    "AuditLoggingMiddleware",
    "RequestValidationMiddleware",
]


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Run the API server using uvicorn."""
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    workers = int(os.getenv("WORKERS", "1"))
    reload = os.getenv("DEBUG", "false").lower() == "true"

    logger.info(f"Starting {__agent_name__} API server on {host}:{port}")

    uvicorn.run(
        "gl_agents.gl_014_exchangerpro.api:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
