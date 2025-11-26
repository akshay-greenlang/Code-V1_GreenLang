# -*- coding: utf-8 -*-
"""
GL-009 THERMALIQ - FastAPI Application Entry Point.

This module provides the FastAPI application for the ThermalEfficiencyCalculator
agent with REST API endpoints for all operation modes, health checks, and
Prometheus metrics.

Standards Compliance:
- ASME PTC 4.1 - Steam Generating Units
- ISO 50001:2018 - Energy Management Systems
- EPA 40 CFR Part 60 - Emissions Standards

Author: GreenLang Foundation
Version: 1.0.0
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException, Request, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
except ImportError:
    # Fallback for environments without FastAPI
    FastAPI = None
    HTTPException = None
    BaseModel = object

from .config import ThermalEfficiencyConfig, create_config
from .thermal_efficiency_orchestrator import (
    ThermalEfficiencyOrchestrator,
    OperationMode,
    create_orchestrator
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class EnergyInputModel(BaseModel):
    """Energy input data model."""
    fuel_inputs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of fuel inputs with type, mass_flow_kg_hr, heating_value_mj_kg"
    )
    electrical_inputs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of electrical inputs with power_kw"
    )


class UsefulOutputModel(BaseModel):
    """Useful output data model."""
    process_heat_kw: float = Field(default=0.0, description="Direct process heat in kW")
    process_temperature_c: float = Field(default=200.0, description="Process temperature in Celsius")
    steam_output: List[Dict[str, Any]] = Field(default_factory=list, description="Steam outputs")
    hot_water_output: List[Dict[str, Any]] = Field(default_factory=list, description="Hot water outputs")


class HeatLossModel(BaseModel):
    """Heat loss data model."""
    flue_gas_losses: Dict[str, Any] = Field(default_factory=dict, description="Flue gas losses")
    radiation_losses: Dict[str, Any] = Field(default_factory=dict, description="Radiation losses")
    convection_losses: Dict[str, Any] = Field(default_factory=dict, description="Convection losses")
    blowdown_losses: Dict[str, Any] = Field(default_factory=dict, description="Blowdown losses")


class AmbientConditionsModel(BaseModel):
    """Ambient conditions model."""
    ambient_temperature_c: float = Field(default=25.0, description="Ambient temperature in Celsius")
    ambient_pressure_bar: float = Field(default=1.01325, description="Ambient pressure in bar")


class ProcessParametersModel(BaseModel):
    """Process parameters model."""
    process_type: str = Field(default="boiler", description="Type of thermal process")
    design_efficiency_percent: Optional[float] = Field(default=None, description="Design efficiency")
    rated_capacity_kw: Optional[float] = Field(default=None, description="Rated capacity")


class CalculationRequest(BaseModel):
    """Request model for efficiency calculation."""
    operation_mode: str = Field(
        default="calculate",
        description="Operation mode: calculate, analyze, benchmark, visualize, report"
    )
    energy_inputs: EnergyInputModel = Field(..., description="Energy input data")
    useful_outputs: UsefulOutputModel = Field(..., description="Useful output data")
    heat_losses: Optional[HeatLossModel] = Field(default=None, description="Heat loss data")
    ambient_conditions: Optional[AmbientConditionsModel] = Field(default=None, description="Ambient conditions")
    process_parameters: Optional[ProcessParametersModel] = Field(default=None, description="Process parameters")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    agent_id: str
    version: str
    timestamp: str
    checks: Dict[str, bool]


class MetricsResponse(BaseModel):
    """Metrics response model."""
    calculations_performed: int
    avg_calculation_time_ms: float
    cache_hit_rate_percent: float
    errors_encountered: int
    timestamp: str


# ============================================================================
# APPLICATION LIFECYCLE
# ============================================================================

# Global orchestrator instance
orchestrator: Optional[ThermalEfficiencyOrchestrator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown of the orchestrator.
    """
    global orchestrator

    # Startup
    logger.info("Starting GL-009 THERMALIQ ThermalEfficiencyCalculator...")

    try:
        config = create_config()
        orchestrator = create_orchestrator(config)
        logger.info(f"Orchestrator initialized: {orchestrator.config.agent_id}")
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down GL-009 THERMALIQ...")
    if orchestrator:
        await orchestrator.shutdown()
    logger.info("Shutdown complete")


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

if FastAPI is not None:
    app = FastAPI(
        title="GL-009 THERMALIQ - ThermalEfficiencyCalculator",
        description="Zero-hallucination thermal efficiency calculations for industrial processes",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ========================================================================
    # HEALTH ENDPOINTS
    # ========================================================================

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        """
        Health check endpoint for Kubernetes probes.

        Returns:
            Health status with component checks
        """
        if orchestrator is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Orchestrator not initialized"
            )

        health = orchestrator.get_health()

        return HealthResponse(
            status=health['status'],
            agent_id=orchestrator.config.agent_id,
            version=orchestrator.config.version,
            timestamp=datetime.now(timezone.utc).isoformat(),
            checks=health['checks']
        )

    @app.get("/ready", tags=["Health"])
    async def readiness_check():
        """
        Readiness check endpoint for Kubernetes.

        Returns:
            Ready status
        """
        if orchestrator is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not ready"
            )

        state = orchestrator.get_state()
        if state['state'] in ['ready', 'executing']:
            return {"status": "ready", "state": state['state']}

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service not ready: {state['state']}"
        )

    @app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
    async def get_metrics():
        """
        Prometheus-compatible metrics endpoint.

        Returns:
            Current performance metrics
        """
        if orchestrator is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Orchestrator not initialized"
            )

        metrics = orchestrator.metrics.get_metrics()

        return MetricsResponse(
            calculations_performed=metrics['calculations_performed'],
            avg_calculation_time_ms=round(metrics['avg_calculation_time_ms'], 2),
            cache_hit_rate_percent=round(metrics.get('cache_hit_rate_percent', 0), 2),
            errors_encountered=metrics['errors_encountered'],
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    # ========================================================================
    # CALCULATION ENDPOINTS
    # ========================================================================

    @app.post("/api/v1/calculate", tags=["Calculations"])
    async def calculate_efficiency(request: CalculationRequest):
        """
        Calculate thermal efficiency.

        Performs First Law and Second Law efficiency calculations
        based on the provided energy inputs and outputs.

        Args:
            request: Calculation request with energy data

        Returns:
            Efficiency calculation results
        """
        if orchestrator is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Orchestrator not initialized"
            )

        try:
            input_data = {
                'operation_mode': request.operation_mode,
                'energy_inputs': request.energy_inputs.model_dump(),
                'useful_outputs': request.useful_outputs.model_dump(),
                'heat_losses': request.heat_losses.model_dump() if request.heat_losses else {},
                'ambient_conditions': request.ambient_conditions.model_dump() if request.ambient_conditions else {},
                'process_parameters': request.process_parameters.model_dump() if request.process_parameters else {}
            }

            result = await orchestrator.execute(input_data)
            return JSONResponse(content=result)

        except ValueError as e:
            logger.error(f"Validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"Calculation failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Calculation failed: {str(e)}"
            )

    @app.post("/api/v1/analyze", tags=["Calculations"])
    async def analyze_efficiency(request: CalculationRequest):
        """
        Deep analysis with loss breakdown.

        Performs comprehensive efficiency analysis including
        detailed loss breakdown and improvement opportunities.
        """
        request.operation_mode = "analyze"
        return await calculate_efficiency(request)

    @app.post("/api/v1/benchmark", tags=["Calculations"])
    async def benchmark_efficiency(request: CalculationRequest):
        """
        Industry benchmark comparison.

        Compares calculated efficiency against industry benchmarks
        and identifies performance gaps.
        """
        request.operation_mode = "benchmark"
        return await calculate_efficiency(request)

    @app.post("/api/v1/visualize", tags=["Calculations"])
    async def visualize_efficiency(request: CalculationRequest):
        """
        Generate Sankey diagram.

        Creates Sankey diagram data for energy flow visualization.
        """
        request.operation_mode = "visualize"
        return await calculate_efficiency(request)

    @app.post("/api/v1/report", tags=["Calculations"])
    async def generate_report(request: CalculationRequest):
        """
        Generate comprehensive report.

        Creates full efficiency report with executive summary,
        recommendations, and compliance status.
        """
        request.operation_mode = "report"
        return await calculate_efficiency(request)

    # ========================================================================
    # INFO ENDPOINTS
    # ========================================================================

    @app.get("/api/v1/info", tags=["Info"])
    async def get_info():
        """
        Get agent information.

        Returns:
            Agent identification and configuration details
        """
        if orchestrator is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Orchestrator not initialized"
            )

        return {
            "agent_id": orchestrator.config.agent_id,
            "codename": orchestrator.config.codename,
            "full_name": orchestrator.config.full_name,
            "version": orchestrator.config.version,
            "deterministic": orchestrator.config.deterministic,
            "standards": [
                "ASME PTC 4.1",
                "ISO 50001:2018",
                "EPA 40 CFR Part 60"
            ],
            "operation_modes": [mode.value for mode in OperationMode],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    @app.get("/api/v1/state", tags=["Info"])
    async def get_state():
        """
        Get current orchestrator state.

        Returns:
            Current state and performance metrics
        """
        if orchestrator is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Orchestrator not initialized"
            )

        return orchestrator.get_state()

    @app.get("/api/v1/tools", tags=["Info"])
    async def get_tools():
        """
        Get available calculation tools.

        Returns:
            List of available tools with schemas
        """
        if orchestrator is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Orchestrator not initialized"
            )

        return orchestrator.tools.get_tool_schemas()

else:
    # Fallback for environments without FastAPI
    app = None


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    """
    Main entry point for CLI execution.

    Starts the FastAPI server with Uvicorn.
    """
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8009"))
    workers = int(os.environ.get("WORKERS", "1"))
    reload = os.environ.get("RELOAD", "false").lower() == "true"

    logger.info(f"Starting GL-009 THERMALIQ on {host}:{port}")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
