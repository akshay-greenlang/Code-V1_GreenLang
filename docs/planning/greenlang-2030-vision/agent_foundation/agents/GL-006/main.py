# -*- coding: utf-8 -*-
"""
GL-006 HeatRecoveryMaximizer Main Application Entry Point.

This module provides the FastAPI application for the GL-006 HeatRecoveryMaximizer
agent, including API endpoints, health checks, and metrics exposure.
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Local imports
from agents.config import get_config, HeatRecoveryConfig
from monitoring.metrics import (
    metrics_collector,
    get_metrics_app,
    PROMETHEUS_AVAILABLE,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Get configuration
config = get_config()

# Application startup time
startup_time = datetime.utcnow()


# Pydantic models for API
class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = "healthy"
    agent_id: str = "GL-006"
    codename: str = "HEATRECLAIM"
    version: str = "1.0.0"
    uptime_seconds: float = 0
    timestamp: str = ""


class ReadyResponse(BaseModel):
    """Readiness check response model."""
    ready: bool = True
    checks: Dict[str, bool] = {}


class HeatStream(BaseModel):
    """Heat stream input model."""
    stream_id: str
    stream_type: str = Field(..., pattern="^(hot|cold)$")
    inlet_temperature_c: float = Field(..., ge=-273.15)
    outlet_temperature_c: float = Field(..., ge=-273.15)
    flow_rate_kg_s: float = Field(..., ge=0)
    specific_heat_kj_kg_k: float = Field(default=4.18, gt=0)
    fluid_type: str = "water"


class AnalysisRequest(BaseModel):
    """Heat recovery analysis request model."""
    hot_streams: List[HeatStream]
    cold_streams: List[HeatStream]
    min_temperature_approach_c: float = Field(default=10.0, ge=1.0)
    optimization_mode: str = "balanced"


class AnalysisResponse(BaseModel):
    """Heat recovery analysis response model."""
    analysis_id: str
    pinch_temperature_c: Optional[float] = None
    min_hot_utility_kw: Optional[float] = None
    min_cold_utility_kw: Optional[float] = None
    heat_recovery_potential_kw: Optional[float] = None
    opportunities: List[Dict[str, Any]] = []
    status: str = "completed"
    duration_ms: float = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting GL-006 HeatRecoveryMaximizer")
    logger.info(f"Environment: {config.ENVIRONMENT}")
    logger.info(f"API Port: {config.API_PORT}")

    # Set agent status
    metrics_collector.agent_status.set(1)
    metrics_collector.agent_uptime_seconds.set(0)

    yield

    # Shutdown
    logger.info("Shutting down GL-006 HeatRecoveryMaximizer")
    metrics_collector.agent_status.set(0)


# Create FastAPI application
app = FastAPI(
    title="GL-006 HeatRecoveryMaximizer",
    description="Heat Recovery Optimization Agent - Maximizes waste heat recovery across process streams",
    version="1.0.0",
    docs_url="/docs" if config.ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if config.ENVIRONMENT != "production" else None,
    lifespan=lifespan,
)

# Add CORS middleware
if config.ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint for liveness probes.

    Returns basic health status of the agent.
    """
    uptime = (datetime.utcnow() - startup_time).total_seconds()
    metrics_collector.agent_uptime_seconds.set(uptime)

    return HealthResponse(
        status="healthy",
        agent_id="GL-006",
        codename="HEATRECLAIM",
        version="1.0.0",
        uptime_seconds=uptime,
        timestamp=datetime.utcnow().isoformat(),
    )


# Readiness check endpoint
@app.get("/ready", response_model=ReadyResponse, tags=["Health"])
async def readiness_check():
    """
    Readiness check endpoint for readiness probes.

    Checks all dependencies and returns readiness status.
    """
    checks = {
        "config_loaded": True,
        "metrics_available": PROMETHEUS_AVAILABLE,
    }

    # Add more checks as needed (database, redis, etc.)

    all_ready = all(checks.values())

    if not all_ready:
        raise HTTPException(status_code=503, detail="Service not ready")

    return ReadyResponse(ready=all_ready, checks=checks)


# Metrics endpoint
@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Prometheus metrics endpoint.

    Returns all collected metrics in Prometheus format.
    """
    if PROMETHEUS_AVAILABLE:
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )
    return Response(content=b"# Metrics not available", media_type="text/plain")


# API Info endpoint
@app.get("/api/v1/info", tags=["Info"])
async def api_info():
    """
    Get agent information.

    Returns detailed information about the agent configuration and capabilities.
    """
    return {
        "agent_id": "GL-006",
        "codename": "HEATRECLAIM",
        "name": "HeatRecoveryMaximizer",
        "version": "1.0.0",
        "domain": "Heat Recovery",
        "mission": "Maximizes waste heat recovery across all process streams",
        "capabilities": [
            "pinch_analysis",
            "exergy_analysis",
            "network_synthesis",
            "roi_calculation",
        ],
        "environment": config.ENVIRONMENT.value,
        "api_version": "v1",
    }


# Heat Recovery Analysis endpoint
@app.post(
    "/api/v1/heat-recovery/analyze",
    response_model=AnalysisResponse,
    tags=["Heat Recovery"],
)
async def analyze_heat_recovery(request: AnalysisRequest):
    """
    Perform heat recovery analysis.

    Analyzes hot and cold streams to identify heat recovery opportunities,
    calculate pinch temperature, and determine minimum utility requirements.
    """
    import time
    import uuid

    start_time = time.time()
    analysis_id = str(uuid.uuid4())

    try:
        # Record metrics
        for stream in request.hot_streams:
            metrics_collector.record_stream_analysis("hot")
        for stream in request.cold_streams:
            metrics_collector.record_stream_analysis("cold")

        # Calculate total heat duties
        hot_duty = sum(
            s.flow_rate_kg_s * s.specific_heat_kj_kg_k * (s.inlet_temperature_c - s.outlet_temperature_c)
            for s in request.hot_streams
        )
        cold_duty = sum(
            s.flow_rate_kg_s * s.specific_heat_kj_kg_k * (s.outlet_temperature_c - s.inlet_temperature_c)
            for s in request.cold_streams
        )

        # Simple pinch calculation (placeholder for full implementation)
        # In production, this would use the full pinch analysis calculator
        hot_temps = [s.inlet_temperature_c for s in request.hot_streams]
        cold_temps = [s.outlet_temperature_c for s in request.cold_streams]

        if hot_temps and cold_temps:
            pinch_temp = (min(hot_temps) + max(cold_temps)) / 2
        else:
            pinch_temp = None

        # Calculate recovery potential
        recovery_potential = min(hot_duty, cold_duty) if hot_duty > 0 and cold_duty > 0 else 0

        # Calculate utilities
        min_hot_utility = max(0, cold_duty - hot_duty)
        min_cold_utility = max(0, hot_duty - cold_duty)

        duration_ms = (time.time() - start_time) * 1000

        # Record pinch analysis metrics
        if pinch_temp is not None:
            metrics_collector.record_pinch_analysis(
                pinch_temp=pinch_temp,
                hot_utility=min_hot_utility,
                cold_utility=min_cold_utility,
                duration=duration_ms / 1000,
            )

        return AnalysisResponse(
            analysis_id=analysis_id,
            pinch_temperature_c=pinch_temp,
            min_hot_utility_kw=min_hot_utility,
            min_cold_utility_kw=min_cold_utility,
            heat_recovery_potential_kw=recovery_potential,
            opportunities=[],  # Would be populated by full analysis
            status="completed",
            duration_ms=duration_ms,
        )

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        metrics_collector.record_error("analysis_error", "heat_recovery")
        raise HTTPException(status_code=500, detail=str(e))


# Configuration endpoint (for debugging, disable in production)
@app.get("/api/v1/config", tags=["Config"])
async def get_configuration():
    """
    Get current configuration (non-sensitive values only).

    Only available in non-production environments.
    """
    if config.ENVIRONMENT.value == "production":
        raise HTTPException(status_code=403, detail="Not available in production")

    # Return non-sensitive config values
    return {
        "environment": config.ENVIRONMENT.value,
        "log_level": config.LOG_LEVEL,
        "min_temperature_approach_c": config.MIN_TEMPERATURE_APPROACH_C,
        "min_recoverable_temperature_c": config.MIN_RECOVERABLE_TEMPERATURE_C,
        "optimization_mode": config.OPTIMIZATION_MODE.value,
        "enable_pinch_analysis": config.ENABLE_PINCH_ANALYSIS,
        "enable_exergy_analysis": config.ENABLE_EXERGY_ANALYSIS,
    }


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    metrics_collector.record_error("unhandled_exception", "global")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if config.DEBUG_MODE else "An error occurred",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.DEBUG_MODE,
        log_level=config.LOG_LEVEL.lower(),
    )
