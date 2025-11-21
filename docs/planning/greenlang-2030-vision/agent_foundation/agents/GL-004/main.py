# -*- coding: utf-8 -*-
"""
GL-004 BurnerOptimizationAgent - FastAPI Application

Main FastAPI application entry point for GL-004 BurnerOptimizationAgent.
Provides HTTP API for burner optimization operations.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from prometheus_client import CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from burner_optimization_orchestrator import BurnerOptimizationOrchestrator
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
agent: BurnerOptimizationOrchestrator = None


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

    try:
        # Initialize agent
        agent = BurnerOptimizationOrchestrator()
        await agent.initialize_integrations()

        # Start agent in background
        asyncio.create_task(agent.start())

        logger.info("Agent started successfully")

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
    description="Burner optimization agent for complete combustion and reduced emissions",
    lifespan=lifespan
)


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint"""
    return {
        "agent": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running"
    }


@app.get("/health")
async def health() -> Dict[str, Any]:
    """
    Health check endpoint
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
        "version": agent.version
    }


@app.get("/readiness")
async def readiness() -> Dict[str, Any]:
    """
    Readiness check endpoint for Kubernetes
    Returns 200 if agent is ready to serve requests
    """
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )

    # Check integrations
    integrations_ok = all([
        agent.burner_controller is not None,
        agent.o2_analyzer is not None,
        agent.emissions_monitor is not None
    ])

    if not integrations_ok:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Integrations not ready"
        )

    return {
        "status": "ready",
        "timestamp": DeterministicClock.utcnow().isoformat()
    }


@app.get("/status")
async def get_status() -> Dict[str, Any]:
    """Get detailed agent status"""
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )

    return agent.get_status()


@app.get("/burner/state")
async def get_burner_state() -> Dict[str, Any]:
    """Get current burner state"""
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )

    if not agent.current_state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No burner state available"
        )

    return agent.current_state.dict()


@app.post("/optimize")
async def trigger_optimization() -> Dict[str, Any]:
    """Trigger immediate optimization cycle"""
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )

    try:
        result = await agent.run_optimization_cycle()
        return result.dict()

    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization failed: {str(e)}"
        )


@app.get("/optimization/history")
async def get_optimization_history(limit: int = 10) -> Dict[str, Any]:
    """Get optimization history"""
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )

    history = agent.optimization_history[-limit:]

    return {
        "count": len(history),
        "optimizations": [opt.dict() for opt in history]
    }


@app.get("/optimization/{optimization_id}")
async def get_optimization_result(optimization_id: str) -> Dict[str, Any]:
    """Get specific optimization result by ID"""
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )

    # Find optimization in history
    for opt in agent.optimization_history:
        if opt.optimization_id == optimization_id:
            return opt.dict()

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Optimization {optimization_id} not found"
    )


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


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "An error occurred"
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
