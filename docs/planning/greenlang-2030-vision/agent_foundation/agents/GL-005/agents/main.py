# -*- coding: utf-8 -*-
"""
GL-005 CombustionControlAgent - FastAPI Application

Main FastAPI application entry point for GL-005 CombustionControlAgent.
Provides HTTP API for real-time combustion control operations.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, status, Query
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from pydantic import BaseModel, Field

from combustion_control_orchestrator import CombustionControlOrchestrator
from config import settings
from monitoring.metrics import metrics_collector
from greenlang.determinism import DeterministicClock

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global agent instance
agent: Optional[CombustionControlOrchestrator] = None


# Request/Response Models
class ControlRequest(BaseModel):
    """Request model for manual control trigger"""
    heat_demand_kw: Optional[float] = Field(None, description="Target heat output (kW)")
    override_interlocks: bool = Field(False, description="Override safety interlocks (use with caution)")


class ControlResponse(BaseModel):
    """Response model for control operations"""
    success: bool
    action_id: Optional[str] = None
    message: str
    cycle_time_ms: Optional[float] = None
    timestamp: str = Field(default_factory=lambda: DeterministicClock.utcnow().isoformat())


class EnableControlRequest(BaseModel):
    """Request model for enabling/disabling control"""
    enabled: bool = Field(..., description="Enable or disable automatic control")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    Handles startup and shutdown events
    """
    global agent

    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.GREENLANG_ENV}")
    logger.info(f"Control loop interval: {settings.CONTROL_LOOP_INTERVAL_MS}ms")

    try:
        # Initialize agent
        agent = CombustionControlOrchestrator()
        await agent.initialize_integrations()

        # Start agent in background if auto-start enabled
        if settings.CONTROL_AUTO_START:
            asyncio.create_task(agent.start())
            logger.info("Agent started in automatic control mode")
        else:
            logger.info("Agent initialized in manual mode")

        logger.info("Agent startup complete")

    except Exception as e:
        logger.error(f"Failed to start agent: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down agent...")
    if agent:
        await agent.stop()
    logger.info("Agent shutdown complete")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Real-time combustion control agent for consistent heat output and stability",
    lifespan=lifespan
)


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint with agent information"""
    return {
        "agent": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "agent_id": "GL-005",
        "status": "running",
        "control_loop_interval_ms": settings.CONTROL_LOOP_INTERVAL_MS,
        "documentation": "/docs"
    }


@app.get("/health")
async def health() -> Dict[str, Any]:
    """
    Health check endpoint for Kubernetes liveness probe
    Returns 200 if agent is healthy, 503 if not
    """
    if not agent or not agent.is_running:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not running"
        )

    return {
        "status": "healthy",
        "timestamp": DeterministicClock.utcnow().isoformat(),
        "agent_id": agent.agent_id,
        "version": agent.version,
        "control_enabled": agent.control_enabled
    }


@app.get("/readiness")
async def readiness() -> Dict[str, Any]:
    """
    Readiness check endpoint for Kubernetes readiness probe
    Returns 200 if agent is ready to serve requests
    """
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )

    # Check critical integrations
    integrations_ok = all([
        agent.dcs is not None,
        agent.plc is not None,
        agent.combustion_analyzer is not None,
        agent.flow_meters is not None
    ])

    if not integrations_ok:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Integrations not ready"
        )

    return {
        "status": "ready",
        "timestamp": DeterministicClock.utcnow().isoformat(),
        "integrations": {
            "dcs": agent.dcs is not None,
            "plc": agent.plc is not None,
            "analyzer": agent.combustion_analyzer is not None,
            "flow_meters": agent.flow_meters is not None
        }
    }


@app.get("/status")
async def get_status() -> Dict[str, Any]:
    """Get detailed agent status including control performance"""
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )

    return agent.get_status()


@app.get("/combustion/state")
async def get_combustion_state() -> Dict[str, Any]:
    """Get current combustion state (latest sensor readings)"""
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )

    if not agent.current_state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No combustion state available yet"
        )

    return agent.current_state.dict()


@app.get("/combustion/stability")
async def get_stability_metrics() -> Dict[str, Any]:
    """Get latest combustion stability metrics"""
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )

    if not agent.stability_history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No stability metrics available yet"
        )

    return agent.stability_history[-1].dict()


@app.post("/control", response_model=ControlResponse)
async def trigger_control_cycle(request: ControlRequest = None) -> ControlResponse:
    """
    Trigger immediate control cycle (manual mode)
    Useful for testing or manual interventions
    """
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )

    try:
        # Extract heat demand if provided
        heat_demand = request.heat_demand_kw if request else None

        # Run control cycle
        result = await agent.run_control_cycle(heat_demand_kw=heat_demand)

        if result['success']:
            return ControlResponse(
                success=True,
                action_id=result.get('action_id'),
                message="Control cycle executed successfully",
                cycle_time_ms=result.get('cycle_time_ms')
            )
        else:
            return ControlResponse(
                success=False,
                message=f"Control cycle failed: {result.get('reason', 'Unknown error')}",
                cycle_time_ms=result.get('cycle_time_ms')
            )

    except Exception as e:
        logger.error(f"Control cycle failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Control cycle failed: {str(e)}"
        )


@app.post("/control/enable", response_model=ControlResponse)
async def enable_control(request: EnableControlRequest) -> ControlResponse:
    """Enable or disable automatic control"""
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )

    try:
        if request.enabled:
            agent.enable_control()
            message = "Automatic control enabled"
        else:
            agent.disable_control()
            message = "Automatic control disabled (manual mode)"

        return ControlResponse(
            success=True,
            message=message
        )

    except Exception as e:
        logger.error(f"Failed to change control mode: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to change control mode: {str(e)}"
        )


@app.get("/control/history")
async def get_control_history(limit: int = Query(10, ge=1, le=1000)) -> Dict[str, Any]:
    """Get recent control action history"""
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )

    history = list(agent.control_history)[-limit:]

    return {
        "count": len(history),
        "control_actions": [action.dict() for action in history]
    }


@app.get("/control/action/{action_id}")
async def get_control_action(action_id: str) -> Dict[str, Any]:
    """Get specific control action by ID"""
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )

    # Search control history
    for action in agent.control_history:
        if action.action_id == action_id:
            return action.dict()

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Control action {action_id} not found"
    )


@app.get("/state/history")
async def get_state_history(limit: int = Query(10, ge=1, le=1000)) -> Dict[str, Any]:
    """Get recent combustion state history"""
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )

    history = list(agent.state_history)[-limit:]

    return {
        "count": len(history),
        "states": [state.dict() for state in history]
    }


@app.get("/safety/interlocks")
async def get_safety_interlocks() -> Dict[str, Any]:
    """Get current safety interlock status"""
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )

    try:
        interlocks = await agent.check_safety_interlocks()
        return interlocks.dict()

    except Exception as e:
        logger.error(f"Failed to check safety interlocks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check safety interlocks: {str(e)}"
        )


@app.get("/performance/metrics")
async def get_performance_metrics() -> Dict[str, Any]:
    """Get control loop performance metrics"""
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )

    cycle_times = list(agent.cycle_times)
    avg_cycle_time = sum(cycle_times) / len(cycle_times) if cycle_times else 0
    max_cycle_time = max(cycle_times) if cycle_times else 0
    min_cycle_time = min(cycle_times) if cycle_times else 0

    return {
        "control_loop_interval_ms": settings.CONTROL_LOOP_INTERVAL_MS,
        "avg_cycle_time_ms": avg_cycle_time,
        "max_cycle_time_ms": max_cycle_time,
        "min_cycle_time_ms": min_cycle_time,
        "cycles_executed": len(agent.control_history),
        "control_errors": agent.control_errors,
        "cycles_exceeding_target": sum(1 for t in cycle_times if t > settings.CONTROL_LOOP_INTERVAL_MS),
        "performance_score": 100 - (avg_cycle_time / settings.CONTROL_LOOP_INTERVAL_MS * 100) if avg_cycle_time else 0
    }


@app.get("/metrics")
async def metrics() -> Response:
    """
    Prometheus metrics endpoint
    Returns metrics in Prometheus exposition format
    """
    return Response(
        content=generate_latest(metrics_collector.registry),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/config")
async def get_config() -> Dict[str, Any]:
    """Get current configuration (non-sensitive values)"""
    return {
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.GREENLANG_ENV,
        "control_loop_interval_ms": settings.CONTROL_LOOP_INTERVAL_MS,
        "fuel_type": settings.FUEL_TYPE,
        "heat_output_target_kw": settings.HEAT_OUTPUT_TARGET_KW,
        "min_fuel_flow": settings.MIN_FUEL_FLOW,
        "max_fuel_flow": settings.MAX_FUEL_FLOW,
        "min_air_flow": settings.MIN_AIR_FLOW,
        "max_air_flow": settings.MAX_AIR_FLOW,
        "target_o2_percent": settings.TARGET_O2_PERCENT,
        "optimal_excess_air_percent": settings.OPTIMAL_EXCESS_AIR_PERCENT,
        "control_auto_start": settings.CONTROL_AUTO_START,
        "o2_trim_enabled": settings.O2_TRIM_ENABLED
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "An error occurred",
            "timestamp": DeterministicClock.utcnow().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        workers=1  # Must be 1 for background control loop
    )
