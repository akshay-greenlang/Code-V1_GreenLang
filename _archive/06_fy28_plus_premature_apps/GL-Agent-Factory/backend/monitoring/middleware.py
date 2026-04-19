"""
GreenLang FastAPI Monitoring Middleware

Production-grade middleware for automatic request metrics collection,
error tracking, and metrics endpoint exposure.
"""

import time
import logging
from typing import Callable, Optional
from functools import wraps

from fastapi import FastAPI, Request, Response
from fastapi.routing import APIRoute
from fastapi.responses import PlainTextResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    REGISTRY,
)

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# HTTP Metrics
# =============================================================================

# Request counter
http_requests_total = Counter(
    'http_requests_total',
    'Total number of HTTP requests',
    ['method', 'handler', 'status']
)

# Request duration histogram
http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'handler'],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0]
)

# Request size histogram
http_request_size_bytes = Histogram(
    'http_request_size_bytes',
    'HTTP request size in bytes',
    ['handler'],
    buckets=[100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
)

# Response size histogram
http_response_size_bytes = Histogram(
    'http_response_size_bytes',
    'HTTP response size in bytes',
    ['handler'],
    buckets=[100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
)

# In-progress requests gauge
http_requests_in_progress = Gauge(
    'http_requests_in_progress',
    'Number of HTTP requests currently being processed',
    ['method', 'handler']
)

# Error counter
http_exceptions_total = Counter(
    'http_exceptions_total',
    'Total number of HTTP exceptions',
    ['method', 'handler', 'exception_type']
)


# =============================================================================
# Prometheus Middleware
# =============================================================================

class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for automatic Prometheus metrics collection.

    Collects:
    - Request count by method, path, and status code
    - Request duration (latency)
    - Request/response sizes
    - In-progress requests
    - Exception counts
    """

    def __init__(
        self,
        app: ASGIApp,
        app_name: str = "greenlang",
        prefix: str = "",
        skip_paths: Optional[list] = None,
        group_paths: bool = True,
    ):
        """
        Initialize the middleware.

        Args:
            app: The ASGI application
            app_name: Application name for metric labels
            prefix: Prefix for metrics endpoint
            skip_paths: List of paths to skip (e.g., ['/health', '/metrics'])
            group_paths: Whether to group paths with path parameters
        """
        super().__init__(app)
        self.app_name = app_name
        self.prefix = prefix
        self.skip_paths = skip_paths or ['/metrics', '/health', '/ready', '/live']
        self.group_paths = group_paths

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request and collect metrics."""
        # Skip metrics for excluded paths
        if request.url.path in self.skip_paths:
            return await call_next(request)

        # Extract handler name from route
        handler = self._get_handler_name(request)
        method = request.method

        # Track in-progress requests
        http_requests_in_progress.labels(method=method, handler=handler).inc()

        # Get request size
        request_size = int(request.headers.get('content-length', 0))
        if request_size > 0:
            http_request_size_bytes.labels(handler=handler).observe(request_size)

        # Start timing
        start_time = time.perf_counter()

        try:
            # Process the request
            response = await call_next(request)

            # Calculate duration
            duration = time.perf_counter() - start_time

            # Get response size
            response_size = int(response.headers.get('content-length', 0))
            if response_size > 0:
                http_response_size_bytes.labels(handler=handler).observe(response_size)

            # Record metrics
            status = str(response.status_code)
            http_requests_total.labels(
                method=method,
                handler=handler,
                status=status
            ).inc()

            http_request_duration_seconds.labels(
                method=method,
                handler=handler
            ).observe(duration)

            return response

        except Exception as exc:
            # Record exception
            duration = time.perf_counter() - start_time
            exception_type = type(exc).__name__

            http_exceptions_total.labels(
                method=method,
                handler=handler,
                exception_type=exception_type
            ).inc()

            http_requests_total.labels(
                method=method,
                handler=handler,
                status='500'
            ).inc()

            http_request_duration_seconds.labels(
                method=method,
                handler=handler
            ).observe(duration)

            # Re-raise the exception
            raise

        finally:
            # Always decrement in-progress counter
            http_requests_in_progress.labels(method=method, handler=handler).dec()

    def _get_handler_name(self, request: Request) -> str:
        """
        Get the handler name from the request.

        Groups paths with path parameters together (e.g., /users/{id} instead of /users/123).
        """
        if not self.group_paths:
            return request.url.path

        # Try to get the route path template
        try:
            route = request.scope.get('route')
            if route and hasattr(route, 'path'):
                return route.path
        except Exception:
            pass

        return request.url.path


# =============================================================================
# Metrics Endpoint Setup
# =============================================================================

def setup_metrics_endpoint(app: FastAPI, path: str = "/metrics") -> None:
    """
    Add a /metrics endpoint to the FastAPI application.

    Args:
        app: The FastAPI application instance
        path: The path for the metrics endpoint (default: /metrics)
    """
    @app.get(path, include_in_schema=False)
    async def metrics() -> Response:
        """Expose Prometheus metrics."""
        return Response(
            content=generate_latest(REGISTRY),
            media_type=CONTENT_TYPE_LATEST
        )


def setup_health_endpoints(app: FastAPI) -> None:
    """
    Add health check endpoints to the FastAPI application.

    Adds:
    - /health - General health check
    - /ready - Readiness probe for K8s
    - /live - Liveness probe for K8s
    """
    @app.get("/health", include_in_schema=False)
    async def health() -> dict:
        """General health check endpoint."""
        return {"status": "healthy"}

    @app.get("/api/v1/health", include_in_schema=False)
    async def api_health() -> dict:
        """API health check endpoint."""
        return {"status": "healthy", "version": "1.0.0"}

    @app.get("/ready", include_in_schema=False)
    async def ready() -> dict:
        """
        Kubernetes readiness probe.
        Returns 200 when the application is ready to receive traffic.
        """
        # Add custom readiness checks here
        # e.g., database connection, cache connection, etc.
        return {"status": "ready"}

    @app.get("/api/v1/ready", include_in_schema=False)
    async def api_ready() -> dict:
        """API readiness probe."""
        return {"status": "ready"}

    @app.get("/live", include_in_schema=False)
    async def live() -> dict:
        """
        Kubernetes liveness probe.
        Returns 200 when the application is alive.
        """
        return {"status": "alive"}


# =============================================================================
# Instrumentation Utilities
# =============================================================================

def instrument_app(
    app: FastAPI,
    app_name: str = "greenlang",
    metrics_path: str = "/metrics",
    skip_paths: Optional[list] = None,
) -> FastAPI:
    """
    Fully instrument a FastAPI application with Prometheus metrics.

    This is the main entry point for adding monitoring to your application.

    Args:
        app: The FastAPI application instance
        app_name: Application name for metric labels
        metrics_path: The path for the metrics endpoint
        skip_paths: List of paths to skip from metrics collection

    Returns:
        The instrumented FastAPI application

    Example:
        ```python
        from fastapi import FastAPI
        from backend.monitoring.middleware import instrument_app

        app = FastAPI()
        app = instrument_app(app, app_name="greenlang-api")
        ```
    """
    # Default skip paths
    default_skip = ['/metrics', '/health', '/ready', '/live', '/api/v1/health', '/api/v1/ready']
    all_skip_paths = list(set(default_skip + (skip_paths or [])))

    # Add Prometheus middleware
    app.add_middleware(
        PrometheusMiddleware,
        app_name=app_name,
        skip_paths=all_skip_paths,
    )

    # Add metrics endpoint
    setup_metrics_endpoint(app, path=metrics_path)

    # Add health endpoints
    setup_health_endpoints(app)

    logger.info(f"Instrumented {app_name} with Prometheus metrics at {metrics_path}")

    return app


# =============================================================================
# Request Timing Decorator
# =============================================================================

def timed_endpoint(metric_name: str, labels: Optional[dict] = None):
    """
    Decorator for timing specific endpoints with custom metrics.

    Args:
        metric_name: Name for the custom metric
        labels: Additional labels for the metric

    Example:
        ```python
        @app.get("/api/v1/calculations")
        @timed_endpoint("calculation_request", labels={"type": "list"})
        async def list_calculations():
            ...
        ```
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start_time
                # Log custom timing (can be expanded with custom metrics)
                logger.debug(f"{metric_name}: {duration:.4f}s")

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start_time
                logger.debug(f"{metric_name}: {duration:.4f}s")

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# =============================================================================
# Custom Route Class with Metrics
# =============================================================================

class MetricsRoute(APIRoute):
    """
    Custom APIRoute that automatically adds timing metrics.

    Use this as the route_class for routers that need detailed timing.

    Example:
        ```python
        from fastapi import APIRouter
        from backend.monitoring.middleware import MetricsRoute

        router = APIRouter(route_class=MetricsRoute)

        @router.get("/calculations")
        async def get_calculations():
            ...
        ```
    """

    def get_route_handler(self) -> Callable:
        original_handler = super().get_route_handler()

        async def custom_handler(request: Request) -> Response:
            handler_name = self.path
            method = request.method

            start_time = time.perf_counter()
            try:
                response = await original_handler(request)
                return response
            finally:
                duration = time.perf_counter() - start_time
                logger.debug(f"{method} {handler_name}: {duration:.4f}s")

        return custom_handler
