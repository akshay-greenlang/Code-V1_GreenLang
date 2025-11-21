# -*- coding: utf-8 -*-
"""
CBAM Importer Copilot - FastAPI Application (Optional)

Example FastAPI application with full monitoring integration:
- Health check endpoints (/health, /health/ready, /health/live)
- Prometheus metrics endpoint (/metrics)
- Structured JSON logging
- Request correlation IDs
- Performance tracking

This is an OPTIONAL component for web deployment. The core CBAM
functionality works as a CLI tool without this.

Usage:
    # Development
    uvicorn backend.app:app --reload

    # Production
    uvicorn backend.app:app --host 0.0.0.0 --port 8000 --workers 4

Version: 1.0.0
Author: GreenLang CBAM Team (Team A3: Monitoring & Observability)
"""

import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any
from greenlang.determinism import deterministic_uuid, DeterministicClock

try:
    from fastapi import FastAPI, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    FASTAPI_AVAILABLE = True
    SLOWAPI_AVAILABLE = True
except ImportError as e:
    FASTAPI_AVAILABLE = False
    SLOWAPI_AVAILABLE = False
    print("Warning: FastAPI or slowapi not installed. Web API will not be available.")
    print("Install with: pip install fastapi uvicorn[standard] slowapi")

from backend.health import CBAMHealthChecker, create_health_endpoints
from backend.metrics import CBAMMetrics, create_metrics_endpoint
from backend.logging_config import (
    configure_production_logging,
    StructuredLogger,
    CorrelationContext
)


# ============================================================================
# APPLICATION LIFECYCLE
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events.
    """
    # Startup
    logger.info("CBAM Importer Copilot starting up...")

    # Initialize metrics
    app.state.metrics = CBAMMetrics()
    app.state.health_checker = CBAMHealthChecker()

    logger.info("Monitoring initialized",
                metrics_enabled=True,
                health_checks_enabled=True)

    yield

    # Shutdown
    logger.info("CBAM Importer Copilot shutting down...")


# ============================================================================
# APPLICATION INITIALIZATION
# ============================================================================

if FASTAPI_AVAILABLE:
    # Configure logging
    configure_production_logging()
    logger = StructuredLogger("cbam.api")

    # SECURITY FIX (HIGH-SEC-002): Initialize rate limiter
    limiter = Limiter(key_func=get_remote_address) if SLOWAPI_AVAILABLE else None

    # Create FastAPI app
    app = FastAPI(
        title="CBAM Importer Copilot",
        description="EU CBAM Reporting API with comprehensive monitoring",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # SECURITY: Add rate limiter to app state
    if limiter:
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # CORS middleware
    # SECURITY FIX (HIGH-SEC-001): Restrict CORS origins to prevent unauthorized access
    allowed_origins = os.getenv("CORS_ORIGINS", "").split(",")

    # Fallback to localhost for development only
    if not allowed_origins or allowed_origins == [""]:
        import logging
        logger = logging.getLogger("cbam.security")
        logger.warning(
            "SECURITY WARNING: CORS_ORIGINS not configured. Using localhost only. "
            "Set CORS_ORIGINS environment variable for production."
        )
        allowed_origins = ["http://localhost:3000", "http://localhost:8000"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,  # SECURITY: Restricted origins from environment
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # SECURITY: Explicit methods only
        allow_headers=["Content-Type", "Authorization", "X-Correlation-ID"],  # SECURITY: Explicit headers
    )

    # ========================================================================
    # MIDDLEWARE
    # ========================================================================

    @app.middleware("http")
    async def correlation_id_middleware(request: Request, call_next):
        """
        Add correlation ID to all requests for distributed tracing.
        """
        # Get or generate correlation ID
        correlation_id = request.headers.get("X-Correlation-ID")
        if not correlation_id:
            correlation_id = str(deterministic_uuid(__name__, str(DeterministicClock.now())))

        CorrelationContext.set_correlation_id(correlation_id)

        # Process request
        response = await call_next(request)

        # Add correlation ID to response headers
        response.headers["X-Correlation-ID"] = correlation_id

        # Clear correlation ID
        CorrelationContext.clear_correlation_id()

        return response

    @app.middleware("http")
    async def logging_middleware(request: Request, call_next):
        """
        Log all requests with timing and status.
        """
        start_time = time.time()

        logger.info(
            "Request started",
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else "unknown"
        )

        response = await call_next(request)

        duration_ms = (time.time() - start_time) * 1000

        logger.info(
            "Request completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2)
        )

        return response

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        """
        Collect request metrics.
        """
        start_time = time.time()

        # Increment active requests
        if hasattr(app.state, 'metrics'):
            app.state.metrics.pipeline_active.inc()

        try:
            response = await call_next(request)
            duration = time.time() - start_time

            # Record metrics
            if hasattr(app.state, 'metrics'):
                status = "success" if response.status_code < 400 else "failed"
                app.state.metrics.record_pipeline_execution(
                    status=status,
                    duration_seconds=duration,
                    stage="request"
                )

            return response

        finally:
            # Decrement active requests
            if hasattr(app.state, 'metrics'):
                app.state.metrics.pipeline_active.dec()

    # ========================================================================
    # HEALTH CHECK ENDPOINTS
    # ========================================================================

    @app.get("/health", tags=["health"])
    async def health():
        """Basic health check - service is running."""
        result = app.state.health_checker.basic_health()
        return JSONResponse(content=result, status_code=200)

    @app.get("/health/ready", tags=["health"])
    async def readiness():
        """Readiness check - service is ready to accept requests."""
        result, status_code = app.state.health_checker.readiness_check()
        return JSONResponse(content=result, status_code=status_code)

    @app.get("/health/live", tags=["health"])
    async def liveness():
        """Liveness check - service is functioning correctly."""
        result, status_code = app.state.health_checker.liveness_check()
        return JSONResponse(content=result, status_code=status_code)

    # ========================================================================
    # METRICS ENDPOINT
    # ========================================================================

    @app.get("/metrics", tags=["metrics"])
    async def metrics():
        """
        Prometheus metrics endpoint.

        Returns metrics in Prometheus exposition format.
        """
        # Update system metrics before export
        app.state.metrics.update_system_metrics()

        from backend.metrics import MetricsExporter
        exporter = MetricsExporter(app.state.metrics)

        return Response(
            content=exporter.export_text(),
            media_type=exporter.get_content_type()
        )

    # ========================================================================
    # API ENDPOINTS (Examples)
    # ========================================================================

    @app.get("/", tags=["root"])
    @limiter.limit("100/minute") if limiter else lambda x: x
    async def root():
        """Root endpoint with API information."""
        return {
            "service": "CBAM Importer Copilot",
            "version": "1.0.0",
            "status": "operational",
            "documentation": "/docs",
            "health": "/health",
            "metrics": "/metrics"
        }

    @app.get("/api/v1/info", tags=["info"])
    async def info():
        """
        Get application information.

        Returns version, capabilities, and status.
        """
        return {
            "name": "CBAM Importer Copilot",
            "version": "1.0.0",
            "description": "EU CBAM Transitional Registry Reporting",
            "capabilities": [
                "Shipment intake and validation",
                "Emissions calculation (zero hallucination)",
                "CBAM report generation",
                "Provenance tracking",
                "Health monitoring",
                "Prometheus metrics"
            ],
            "monitoring": {
                "health_checks": True,
                "metrics": True,
                "structured_logging": True,
                "correlation_ids": True
            },
            "sla": {
                "availability": "99.9%",
                "success_rate": "99%",
                "latency_p95": "10 minutes"
            }
        }

    # ========================================================================
    # EXAMPLE: PIPELINE EXECUTION ENDPOINT
    # ========================================================================

    @app.post("/api/v1/pipeline/execute", tags=["pipeline"])
    @limiter.limit("10/minute") if limiter else lambda x: x  # SECURITY: Rate limit pipeline execution
    async def execute_pipeline(request: Request):
        """
        Execute CBAM pipeline (example endpoint).

        This is a placeholder showing how to integrate monitoring
        with actual pipeline execution.

        In production, you would:
        1. Accept shipment data as request body
        2. Execute the 3-agent pipeline
        3. Return CBAM report
        """
        logger.info("Pipeline execution requested")

        # Record pipeline start
        app.state.metrics.pipeline_active.inc()
        start_time = time.time()

        try:
            # Placeholder: actual pipeline execution would go here
            # from cbam_pipeline import CBAMPipeline
            # pipeline = CBAMPipeline(...)
            # result = pipeline.run(...)

            result = {
                "status": "success",
                "message": "Pipeline execution completed",
                "report_id": "CBAM-2024-Q4-001",
                "records_processed": 1000,
                "emissions_tco2": 12345.67
            }

            duration = time.time() - start_time

            # Record successful execution
            app.state.metrics.record_pipeline_execution(
                status="success",
                duration_seconds=duration,
                stage="total"
            )
            app.state.metrics.records_processed_total.labels(
                stage="total",
                status="success"
            ).inc(1000)

            logger.info(
                "Pipeline execution completed",
                duration_seconds=duration,
                records_processed=1000
            )

            return result

        except Exception as e:
            duration = time.time() - start_time

            # Record failed execution
            app.state.metrics.record_pipeline_execution(
                status="failed",
                duration_seconds=duration,
                stage="total"
            )
            app.state.metrics.record_exception(e)

            logger.error(
                "Pipeline execution failed",
                exception=e,
                duration_seconds=duration
            )

            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": str(e)
                }
            )

        finally:
            app.state.metrics.pipeline_active.dec()

    # ========================================================================
    # ERROR HANDLERS
    # ========================================================================

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """
        Global exception handler with logging and metrics.
        """
        logger.error(
            "Unhandled exception",
            exception=exc,
            path=request.url.path,
            method=request.method
        )

        if hasattr(app.state, 'metrics'):
            app.state.metrics.record_exception(exc)

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Internal server error",
                "correlation_id": CorrelationContext.get_correlation_id()
            }
        )

else:
    # FastAPI not available
    app = None
    print("FastAPI application not created. Install FastAPI to enable web API.")


# ============================================================================
# CLI RUNNER
# ============================================================================

if __name__ == "__main__":
    if FASTAPI_AVAILABLE:
        import uvicorn

        print("Starting CBAM Importer Copilot API...")
        print("Health: http://localhost:8000/health")
        print("Metrics: http://localhost:8000/metrics")
        print("Docs: http://localhost:8000/docs")

        uvicorn.run(
            "backend.app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    else:
        print("ERROR: FastAPI is required to run the web API")
        print("Install with: pip install fastapi uvicorn[standard]")
