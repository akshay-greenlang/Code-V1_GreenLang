#!/usr/bin/env python3
"""
Building Energy Performance Calculator Agent - FastAPI Entrypoint

This module provides the HTTP API for the Building Energy Performance Calculator Agent.
It exposes health checks, metrics, and the agent execution endpoint.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Any, Dict

# Add paths for imports
sys.path.insert(0, "/app")
sys.path.insert(0, "/app/greenlang_sdk")

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import agent
from agent.agent import BuildingEnergyPerformanceCalculatorAgent

# Configure logging
logging.basicConfig(
    level=os.getenv("GREENLANG_LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Building Energy Performance Calculator",
    description="Calculates EUI and BPS compliance for buildings",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware - SECURITY: Configure specific origins in production
import os
_cors_origins = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000")
_allowed_origins = [origin.strip() for origin in _cors_origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-Request-ID"],
)

# Global agent instance
agent = BuildingEnergyPerformanceCalculatorAgent()

# Track startup time
startup_time = datetime.utcnow()


# =============================================================================
# Request/Response Models
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Health status")
    agent_id: str = Field(..., description="Agent identifier")
    agent_version: str = Field(..., description="Agent version")
    uptime_seconds: float = Field(..., description="Uptime in seconds")


class ReadyResponse(BaseModel):
    """Readiness check response."""
    ready: bool = Field(..., description="Readiness status")
    checks: Dict[str, bool] = Field(..., description="Individual check results")


class ExecuteRequest(BaseModel):
    """Agent execution request."""
    building_type: str = Field(..., description="Building type (office, residential, retail, industrial, warehouse, hotel, hospital, school, restaurant)")
    gross_floor_area_sqft: float = Field(..., description="Gross floor area in square feet")
    annual_energy_kwh: float = Field(..., description="Annual energy consumption in kWh")
    climate_zone: str = Field("4A", description="ASHRAE climate zone")
    jurisdiction: str = Field("NYC_LL97", description="Regulatory jurisdiction (NYC_LL97, ENERGY_STAR, ASHRAE_90.1)")
    year: int = Field(2024, description="Compliance year")


class ExecuteResponse(BaseModel):
    """Agent execution response."""
    eui_kbtu_sqft: float = Field(..., description="Energy Use Intensity (kBTU/sqft)")
    threshold_eui: float = Field(..., description="Threshold EUI for compliance")
    compliant: bool = Field(..., description="BPS compliance status")
    penalty_estimate: float = Field(None, description="Estimated penalty if non-compliant (USD)")
    provenance_hash: str = Field(None, description="Provenance hash for audit")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


# =============================================================================
# Health Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Health check endpoint for liveness probes."""
    uptime = (datetime.utcnow() - startup_time).total_seconds()
    return HealthResponse(
        status="healthy",
        agent_id=agent.agent_id,
        agent_version=agent.agent_version,
        uptime_seconds=uptime
    )


@app.get("/health/live", tags=["Health"])
async def liveness():
    """Kubernetes liveness probe endpoint."""
    return {"status": "alive"}


@app.get("/health/ready", response_model=ReadyResponse, tags=["Health"])
async def readiness():
    """Readiness check endpoint for readiness probes."""
    checks = {
        "agent_initialized": agent is not None,
        "tools_registered": len(agent._tools) > 0,
    }

    all_ready = all(checks.values())

    if not all_ready:
        raise HTTPException(status_code=503, detail="Service not ready")

    return ReadyResponse(ready=all_ready, checks=checks)


# =============================================================================
# Metrics Endpoint
# =============================================================================

@app.get("/metrics", tags=["Observability"])
async def metrics():
    """Prometheus metrics endpoint."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# =============================================================================
# Agent Endpoints
# =============================================================================

@app.post("/api/v1/execute", response_model=ExecuteResponse, tags=["Agent"])
async def execute(request: ExecuteRequest):
    """
    Execute the Building Energy Performance Calculator.

    Calculates Energy Use Intensity (EUI) and checks compliance
    against Building Performance Standards (BPS).
    """
    try:
        logger.info(f"Executing agent with input: {request.model_dump()}")

        # Execute agent (simplified - adapt to actual agent interface)
        result = await agent.run({
            "building_type": request.building_type,
            "gross_floor_area_sqft": request.gross_floor_area_sqft,
            "annual_energy_kwh": request.annual_energy_kwh,
            "climate_zone": request.climate_zone,
            "jurisdiction": request.jurisdiction,
            "year": request.year
        })

        # Return response
        return ExecuteResponse(
            eui_kbtu_sqft=result.output.get("eui_kbtu_sqft", 0.0),
            threshold_eui=result.output.get("threshold_eui", 0.0),
            compliant=result.output.get("compliant", False),
            penalty_estimate=result.output.get("penalty_estimate"),
            provenance_hash=result.output.get("provenance_hash"),
            processing_time_ms=result.output.get("processing_time_ms", 0.0)
        )

    except Exception as e:
        logger.error(f"Agent execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/info", tags=["Agent"])
async def agent_info():
    """Get agent information and capabilities."""
    return {
        "agent_id": agent.agent_id,
        "agent_version": agent.agent_version,
        "description": "Calculates EUI and BPS compliance for buildings",
        "supported_building_types": [
            "office", "residential", "retail", "industrial",
            "warehouse", "hotel", "hospital", "school", "restaurant"
        ],
        "supported_jurisdictions": ["NYC_LL97", "ENERGY_STAR", "ASHRAE_90.1"],
        "tools": list(agent._tools.keys()),
        "capabilities": {
            "provenance_tracking": agent.enable_provenance,
            "citation_tracking": agent.enable_citations
        }
    }


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "agent_id": agent.agent_id
        }
    )


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", "8000"))

    logger.info(f"Starting Building Energy Performance Calculator on {host}:{port}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=os.getenv("GREENLANG_LOG_LEVEL", "info").lower(),
        access_log=True
    )
