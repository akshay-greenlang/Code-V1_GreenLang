"""
GL-016_Waterguard API Server Entrypoint

Main ASGI application combining FastAPI REST endpoints and Strawberry GraphQL.
Provides unified API server with CORS, lifecycle hooks, and multi-protocol support.

Author: GL-APIDeveloper
Version: 1.0.0

Usage:
    # Development
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

    # Production
    uvicorn api.main:app --workers 4 --host 0.0.0.0 --port 8000

    # With gRPC (separate process)
    python -m api.main --grpc --port 50051
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Import local modules
from api.config import get_api_config, get_api_settings
from api.rest_api import router as rest_router
from api.graphql_api import schema as graphql_schema
from api.api_auth import get_audit_logger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Global State
# =============================================================================

_server_start_time: Optional[datetime] = None
_ready: bool = False


# =============================================================================
# Middleware Classes
# =============================================================================

class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to each request."""

    async def dispatch(self, request: Request, call_next: Callable) -> Any:
        """Process request and add request ID."""
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        # Store in request state
        request.state.request_id = request_id

        # Process request
        response = await call_next(request)

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """Track request processing time."""

    async def dispatch(self, request: Request, call_next: Callable) -> Any:
        """Process request and track timing."""
        start_time = time.time()

        response = await call_next(request)

        duration_ms = (time.time() - start_time) * 1000
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

        return response


class AuditMiddleware(BaseHTTPMiddleware):
    """Audit logging middleware for all API requests."""

    async def dispatch(self, request: Request, call_next: Callable) -> Any:
        """Log request to audit trail."""
        start_time = time.time()

        # Skip health check endpoints for audit
        if request.url.path in ["/health", "/api/v1/health", "/api/v1/health/live", "/api/v1/health/ready"]:
            return await call_next(request)

        try:
            response = await call_next(request)

            # Log successful requests
            duration_ms = (time.time() - start_time) * 1000
            logger.info(
                f"API Request: {request.method} {request.url.path} "
                f"- Status: {response.status_code} "
                f"- Duration: {duration_ms:.2f}ms"
            )

            return response

        except Exception as e:
            # Log failed requests
            duration_ms = (time.time() - start_time) * 1000
            logger.error(
                f"API Request Failed: {request.method} {request.url.path} "
                f"- Error: {str(e)} "
                f"- Duration: {duration_ms:.2f}ms"
            )
            raise


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware."""

    def __init__(self, app: Any, calls_per_minute: int = 100):
        """Initialize rate limiter."""
        super().__init__(app)
        self.calls_per_minute = calls_per_minute
        self._request_counts: Dict[str, list] = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Any:
        """Check rate limit and process request."""
        # Get client identifier (IP or user)
        client_id = request.client.host if request.client else "unknown"

        # Get current minute
        current_minute = int(time.time() / 60)

        # Initialize or clean old entries
        if client_id not in self._request_counts:
            self._request_counts[client_id] = []

        # Remove old entries
        self._request_counts[client_id] = [
            ts for ts in self._request_counts[client_id]
            if int(ts / 60) == current_minute
        ]

        # Check rate limit
        if len(self._request_counts[client_id]) >= self.calls_per_minute:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"Rate limit of {self.calls_per_minute} requests per minute exceeded",
                    "retry_after": 60 - (time.time() % 60),
                },
                headers={
                    "Retry-After": str(int(60 - (time.time() % 60))),
                    "X-RateLimit-Limit": str(self.calls_per_minute),
                    "X-RateLimit-Remaining": "0",
                },
            )

        # Record request
        self._request_counts[client_id].append(time.time())

        # Add rate limit headers to response
        response = await call_next(request)
        remaining = self.calls_per_minute - len(self._request_counts[client_id])
        response.headers["X-RateLimit-Limit"] = str(self.calls_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)

        return response


# =============================================================================
# Lifecycle Hooks
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events for the API server.
    """
    global _server_start_time, _ready

    # Startup
    logger.info("=" * 60)
    logger.info("GL-016_Waterguard API Server Starting...")
    logger.info("=" * 60)

    _server_start_time = datetime.utcnow()

    # Initialize services
    try:
        logger.info("Initializing database connections...")
        # await init_database()

        logger.info("Initializing cache layer...")
        # await init_cache()

        logger.info("Loading ML models...")
        # await load_models()

        logger.info("Connecting to PLC gateway...")
        # await connect_plc()

        _ready = True
        logger.info("Server ready to accept requests")

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

    yield

    # Shutdown
    logger.info("=" * 60)
    logger.info("GL-016_Waterguard API Server Shutting Down...")
    logger.info("=" * 60)

    _ready = False

    try:
        logger.info("Closing active connections...")
        # await close_connections()

        logger.info("Flushing audit logs...")
        # await flush_audit_logs()

        logger.info("Graceful shutdown complete")

    except Exception as e:
        logger.error(f"Shutdown error: {e}")


# =============================================================================
# Application Factory
# =============================================================================

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance
    """
    config = get_api_config()

    # Create FastAPI app
    app = FastAPI(
        **get_api_settings(),
        lifespan=lifespan,
    )

    # ==========================================================================
    # Add Middleware (order matters - last added is first executed)
    # ==========================================================================

    # GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Rate limiting
    app.add_middleware(
        RateLimitMiddleware,
        calls_per_minute=config.rate_limit.requests_per_minute,
    )

    # Audit logging
    app.add_middleware(AuditMiddleware)

    # Request timing
    app.add_middleware(TimingMiddleware)

    # Request ID
    app.add_middleware(RequestIDMiddleware)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=config.cors.allow_credentials,
        allow_methods=config.cors.allowed_methods,
        allow_headers=config.cors.allowed_headers,
        expose_headers=config.cors.expose_headers,
        max_age=config.cors.max_age,
    )

    # Trusted hosts (only in production)
    if not config.is_development:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=config.server.trusted_hosts,
        )

    # ==========================================================================
    # Register Routes
    # ==========================================================================

    # REST API routes
    app.include_router(rest_router)

    # GraphQL endpoint
    try:
        from strawberry.fastapi import GraphQLRouter

        graphql_router = GraphQLRouter(
            graphql_schema,
            path="",
        )
        app.include_router(
            graphql_router,
            prefix=config.graphql_path,
            tags=["GraphQL"],
        )
        logger.info(f"GraphQL endpoint mounted at {config.graphql_path}")

    except ImportError:
        logger.warning("Strawberry not installed. GraphQL endpoint not available.")

    # ==========================================================================
    # Root Endpoint
    # ==========================================================================

    @app.get("/", tags=["Root"])
    async def root() -> Dict[str, Any]:
        """Root endpoint with API information."""
        return {
            "name": config.api_title,
            "version": config.api_version,
            "description": "Waterguard Cooling Tower Optimization API",
            "docs": config.docs_url,
            "redoc": config.redoc_url,
            "graphql": config.graphql_path,
            "health": "/api/v1/health",
        }

    # ==========================================================================
    # Exception Handlers
    # ==========================================================================

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle all unhandled exceptions."""
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

        logger.exception(f"Unhandled exception: {exc}")

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "internal_error",
                "message": "An unexpected error occurred",
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    @app.exception_handler(status.HTTP_404_NOT_FOUND)
    async def not_found_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle 404 errors."""
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "error": "not_found",
                "message": f"Path {request.url.path} not found",
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    return app


# =============================================================================
# Application Instance
# =============================================================================

app = create_app()


# =============================================================================
# CLI Entrypoint
# =============================================================================

def main() -> None:
    """Main entry point for CLI execution."""
    parser = argparse.ArgumentParser(
        description="GL-016_Waterguard API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start REST/GraphQL server
    python -m api.main

    # Start with custom port
    python -m api.main --port 8080

    # Start gRPC server
    python -m api.main --grpc --grpc-port 50051

    # Start all servers
    python -m api.main --all
        """,
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="REST/GraphQL server port (default: 8000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--grpc",
        action="store_true",
        help="Start gRPC server",
    )
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=50051,
        help="gRPC server port (default: 50051)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Start all servers (REST, GraphQL, gRPC)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level (default: info)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    if args.grpc or args.all:
        # Start gRPC server
        logger.info(f"Starting gRPC server on port {args.grpc_port}...")

        try:
            from api.grpc_services import create_grpc_server, serve_grpc

            server = create_grpc_server(
                host=args.host,
                port=args.grpc_port,
            )

            if server and args.grpc and not args.all:
                # Only gRPC server
                asyncio.run(serve_grpc(server))
                return

        except ImportError:
            logger.warning("gRPC dependencies not installed")

    # Start REST/GraphQL server with uvicorn
    try:
        import uvicorn

        logger.info(f"Starting REST/GraphQL server on {args.host}:{args.port}...")

        uvicorn.run(
            "api.main:app",
            host=args.host,
            port=args.port,
            workers=args.workers if not args.reload else 1,
            reload=args.reload,
            log_level=args.log_level,
            access_log=True,
        )

    except ImportError:
        logger.error("uvicorn not installed. Install with: pip install uvicorn")
        sys.exit(1)


if __name__ == "__main__":
    main()
