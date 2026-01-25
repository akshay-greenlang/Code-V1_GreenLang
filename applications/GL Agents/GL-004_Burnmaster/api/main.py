"""
GL-004 BURNMASTER API Main Application

FastAPI application setup with middleware configuration, OpenAPI documentation,
startup/shutdown events, and error handling.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from strawberry.fastapi import GraphQLRouter
import logging
import time
import uuid
from datetime import datetime

from .config import get_settings, is_production, is_development
from .rest_api import router as rest_router
from .graphql_api import schema as graphql_schema
from .websocket_handler import router as websocket_router, start_background_tasks
from .api_auth import audit_logger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

settings = get_settings()


# ============================================================================
# Rate Limiter
# ============================================================================

limiter = Limiter(key_func=get_remote_address)


# ============================================================================
# Lifespan Events
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events.
    """
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")

    # Start background tasks for WebSocket
    if settings.websocket.enabled:
        await start_background_tasks()
        logger.info("WebSocket background tasks started")

    # Initialize database connection (placeholder)
    # await init_database()

    # Initialize cache (placeholder)
    # await init_cache()

    logger.info("Application startup complete")

    yield

    # Shutdown
    logger.info("Shutting down application...")

    # Close database connections (placeholder)
    # await close_database()

    # Close cache connections (placeholder)
    # await close_cache()

    logger.info("Application shutdown complete")


# ============================================================================
# Application Factory
# ============================================================================

def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title=settings.app_name,
        description=settings.app_description,
        version=settings.app_version,
        docs_url="/api/docs" if is_development() else None,
        redoc_url="/api/redoc" if is_development() else None,
        openapi_url="/api/openapi.json" if is_development() else None,
        lifespan=lifespan,
        openapi_tags=[
            {
                "name": "Burner Optimization",
                "description": "Burner unit status, KPIs, and optimization operations"
            },
            {
                "name": "WebSocket",
                "description": "Real-time WebSocket endpoints for streaming data"
            },
            {
                "name": "System",
                "description": "Health checks and system information"
            }
        ]
    )

    # Configure rate limiter
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Add middleware
    configure_middleware(app)

    # Add exception handlers
    configure_exception_handlers(app)

    # Include routers
    app.include_router(rest_router)
    app.include_router(websocket_router)

    # Add GraphQL router
    if settings.graphql.enabled:
        graphql_router = GraphQLRouter(
            graphql_schema,
            path=settings.graphql.path
        )
        app.include_router(graphql_router, prefix="/api")

    logger.info(f"Application configured with {len(app.routes)} routes")

    return app


# ============================================================================
# Middleware Configuration
# ============================================================================

def configure_middleware(app: FastAPI):
    """Configure application middleware."""

    # CORS Middleware
    if settings.cors.enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors.allow_origins,
            allow_credentials=settings.cors.allow_credentials,
            allow_methods=settings.cors.allow_methods,
            allow_headers=settings.cors.allow_headers,
            expose_headers=settings.cors.expose_headers,
            max_age=settings.cors.max_age
        )
        logger.info("CORS middleware configured")

    # Trusted Host Middleware (production only)
    if is_production():
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*.greenlang.io", "localhost"]
        )
        logger.info("Trusted host middleware configured")

    # GZip Middleware
    app.add_middleware(
        GZipMiddleware,
        minimum_size=1000
    )

    # Request ID Middleware
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        """Add unique request ID to each request."""
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        return response

    # Timing Middleware
    @app.middleware("http")
    async def add_timing(request: Request, call_next):
        """Add request timing headers."""
        start_time = time.time()

        response = await call_next(request)

        process_time = (time.time() - start_time) * 1000
        response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"

        # Log slow requests
        if process_time > 1000:  # 1 second
            logger.warning(
                f"Slow request: {request.method} {request.url.path} "
                f"took {process_time:.2f}ms"
            )

        return response

    # Security Headers Middleware
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        """Add security headers to responses."""
        response = await call_next(request)

        if is_production():
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            response.headers["Content-Security-Policy"] = "default-src 'self'"

        return response

    logger.info("Middleware configuration complete")


# ============================================================================
# Exception Handlers
# ============================================================================

def configure_exception_handlers(app: FastAPI):
    """Configure custom exception handlers."""

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle HTTP exceptions with structured response."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.status_code,
                    "message": exc.detail,
                    "type": "http_error"
                },
                "request_id": getattr(request.state, 'request_id', None),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle validation errors with detailed response."""
        errors = []
        for error in exc.errors():
            errors.append({
                "field": ".".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"]
            })

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "code": 422,
                    "message": "Validation error",
                    "type": "validation_error",
                    "details": errors
                },
                "request_id": getattr(request.state, 'request_id', None),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        logger.exception(f"Unhandled exception: {exc}")

        # Don't expose internal errors in production
        message = str(exc) if is_development() else "Internal server error"

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": 500,
                    "message": message,
                    "type": "internal_error"
                },
                "request_id": getattr(request.state, 'request_id', None),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    logger.info("Exception handlers configured")


# ============================================================================
# Application Instance
# ============================================================================

app = create_app()


# ============================================================================
# Root Endpoints
# ============================================================================

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirect to docs."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/api/docs",
        "health": "/api/v1/health"
    }


@app.get("/api", include_in_schema=False)
async def api_root():
    """API root with available endpoints."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "endpoints": {
            "rest": "/api/v1",
            "graphql": settings.graphql.path if settings.graphql.enabled else None,
            "websocket": "/ws",
            "docs": "/api/docs" if is_development() else None,
            "health": "/api/v1/health"
        }
    }


# ============================================================================
# CLI Entry Point
# ============================================================================

def run_server():
    """Run the API server using uvicorn."""
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.server.host,
        port=settings.server.port,
        workers=settings.server.workers if is_production() else 1,
        reload=settings.server.reload or is_development(),
        log_level=settings.server.log_level,
        access_log=True
    )


if __name__ == "__main__":
    run_server()
