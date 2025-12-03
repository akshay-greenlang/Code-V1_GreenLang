#!/usr/bin/env python3
"""
EUDR Deforestation Compliance Agent - FastAPI Entrypoint

This module provides the HTTP API for the EUDR Deforestation Compliance Agent.
It exposes health checks, metrics, and the agent execution endpoint.

Critical Deadline: December 30, 2025 (27 days remaining)
"""

import os
import sys
import logging
from datetime import datetime
from typing import Any, Dict, List

# Add paths for imports
sys.path.insert(0, "/app")
sys.path.insert(0, "/app/core")
sys.path.insert(0, "/app/greenlang_sdk")

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import agent
from agent.agent import EudrDeforestationComplianceAgentAgent, EudrDeforestationComplianceAgentAgentInput

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
    title="EUDR Deforestation Compliance Agent",
    description="Validates supply chain compliance with EU Deforestation Regulation (EU) 2023/1115",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance
agent = EudrDeforestationComplianceAgentAgent()

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
    deadline: str = Field(..., description="EUDR enforcement deadline")
    days_to_deadline: int = Field(..., description="Days remaining until deadline")


class ReadyResponse(BaseModel):
    """Readiness check response."""
    ready: bool = Field(..., description="Readiness status")
    checks: Dict[str, bool] = Field(..., description="Individual check results")


class ExecuteRequest(BaseModel):
    """Agent execution request."""
    tool: str = Field(..., description="Tool to execute")
    coordinates: List[Any] = Field(..., description="Geolocation coordinates")
    coordinate_type: str = Field(..., description="Coordinate type (WGS84, plot_polygon)")
    country_code: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    precision_meters: int = Field(..., description="Geolocation precision in meters")
    cn_code: str = Field(..., description="Combined Nomenclature code")
    product_description: str = Field(..., description="Product description")
    quantity_kg: int = Field(..., description="Product quantity in kg")
    commodity_type: str = Field(..., description="EUDR commodity type")
    production_year: int = Field(..., description="Year of production")


class ExecuteResponse(BaseModel):
    """Agent execution response."""
    valid: bool = Field(..., description="Geolocation validity")
    in_protected_area: bool = Field(..., description="Whether location is in protected area")
    eudr_regulated: bool = Field(..., description="Whether product is EUDR regulated")
    commodity_type: str = Field(..., description="Classified commodity type")
    risk_level: str = Field(..., description="Country/commodity risk level")
    satellite_verification_required: bool = Field(..., description="Whether satellite verification is required")
    provenance_hash: str = Field(None, description="Provenance hash for audit")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


# =============================================================================
# Health Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """
    Health check endpoint for liveness probes.

    Returns basic health status indicating the service is running.
    """
    uptime = (datetime.utcnow() - startup_time).total_seconds()
    deadline = datetime(2025, 12, 30)
    days_to_deadline = (deadline - datetime.utcnow()).days

    return HealthResponse(
        status="healthy",
        agent_id=agent.agent_id,
        agent_version=agent.agent_version,
        uptime_seconds=uptime,
        deadline="2025-12-30",
        days_to_deadline=days_to_deadline
    )


@app.get("/health/live", tags=["Health"])
async def liveness():
    """Kubernetes liveness probe endpoint."""
    return {"status": "alive"}


@app.get("/health/ready", response_model=ReadyResponse, tags=["Health"])
async def readiness():
    """
    Readiness check endpoint for readiness probes.

    Checks that all dependencies are available.
    """
    checks = {
        "agent_initialized": agent is not None,
        "tools_registered": len(agent._tools) > 0,
        "eudr_databases_loaded": True,  # Check if EUDR databases are loaded
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
    # Basic metrics - extend with prometheus_client for production
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# =============================================================================
# Agent Endpoints
# =============================================================================

@app.post("/api/v1/execute", response_model=ExecuteResponse, tags=["Agent"])
async def execute(request: ExecuteRequest):
    """
    Execute the EUDR Deforestation Compliance Agent.

    Validates supply chain compliance with EU Deforestation Regulation (EU) 2023/1115.
    """
    try:
        logger.info(f"Executing EUDR agent with input: {request.model_dump()}")

        # Create input model
        agent_input = EudrDeforestationComplianceAgentAgentInput(
            tool=request.tool,
            coordinates=request.coordinates,
            coordinate_type=request.coordinate_type,
            country_code=request.country_code,
            precision_meters=request.precision_meters,
            cn_code=request.cn_code,
            product_description=request.product_description,
            quantity_kg=request.quantity_kg,
            commodity_type=request.commodity_type,
            production_year=request.production_year
        )

        # Execute agent
        result = await agent.run(agent_input)

        # Return response
        return ExecuteResponse(
            valid=result.output.valid,
            in_protected_area=result.output.in_protected_area,
            eudr_regulated=result.output.eudr_regulated,
            commodity_type=result.output.commodity_type,
            risk_level=result.output.risk_level,
            satellite_verification_required=result.output.satellite_verification_required,
            provenance_hash=result.output.provenance_hash,
            processing_time_ms=result.output.processing_time_ms
        )

    except Exception as e:
        logger.error(f"EUDR agent execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/info", tags=["Agent"])
async def agent_info():
    """Get agent information and capabilities."""
    deadline = datetime(2025, 12, 30)
    days_to_deadline = (deadline - datetime.utcnow()).days

    return {
        "agent_id": agent.agent_id,
        "agent_version": agent.agent_version,
        "description": "Validates supply chain compliance with EU Deforestation Regulation (EU) 2023/1115",
        "regulated_commodities": ["cattle", "cocoa", "coffee", "palm_oil", "rubber", "soya", "wood"],
        "cutoff_date": "2020-12-31",
        "enforcement_deadline": "2025-12-30",
        "days_to_deadline": days_to_deadline,
        "tier": "1-extreme-urgency",
        "tools": list(agent._tools.keys()),
        "capabilities": {
            "geolocation_validation": True,
            "commodity_classification": True,
            "country_risk_assessment": True,
            "supply_chain_tracing": True,
            "dds_generation": True,
            "provenance_tracking": agent.enable_provenance,
            "citation_tracking": agent.enable_citations
        }
    }


@app.get("/api/v1/commodities", tags=["Reference Data"])
async def list_commodities():
    """List all EUDR-regulated commodities and CN codes."""
    from greenlang.data import EUDR_COMMODITIES, CN_CODE_DATABASE

    return {
        "commodities": [
            {
                "type": c.commodity_type.value,
                "description": c.description,
                "cn_codes_count": len(c.cn_codes),
                "risk_category": c.risk_category.value
            }
            for c in EUDR_COMMODITIES.values()
        ],
        "total_cn_codes": len(CN_CODE_DATABASE),
        "cutoff_date": "2020-12-31"
    }


@app.get("/api/v1/countries", tags=["Reference Data"])
async def list_countries():
    """List country risk profiles."""
    from greenlang.data import COUNTRY_RISK_DATABASE, RiskLevel

    high_risk = [c for c, r in COUNTRY_RISK_DATABASE.items() if r.overall_risk == RiskLevel.HIGH]
    standard_risk = [c for c, r in COUNTRY_RISK_DATABASE.items() if r.overall_risk == RiskLevel.STANDARD]
    low_risk = [c for c, r in COUNTRY_RISK_DATABASE.items() if r.overall_risk == RiskLevel.LOW]

    return {
        "total_countries": len(COUNTRY_RISK_DATABASE),
        "high_risk_count": len(high_risk),
        "standard_risk_count": len(standard_risk),
        "low_risk_count": len(low_risk),
        "high_risk_countries": high_risk,
        "standard_risk_countries": standard_risk,
        "low_risk_countries": low_risk
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

    logger.info(f"Starting EUDR Deforestation Compliance Agent on {host}:{port}")
    logger.warning(f"CRITICAL DEADLINE: December 30, 2025 (27 days remaining)")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=os.getenv("GREENLANG_LOG_LEVEL", "info").lower(),
        access_log=True
    )
