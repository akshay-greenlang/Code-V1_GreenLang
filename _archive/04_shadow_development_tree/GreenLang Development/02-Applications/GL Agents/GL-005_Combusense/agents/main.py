# -*- coding: utf-8 -*-
"""
GL-005 CombustionControlAgent - FastAPI Application

Main FastAPI application entry point for GL-005 CombustionControlAgent.
Provides HTTP API for real-time combustion control operations.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, status, Query, Depends, Security, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from pydantic import BaseModel, Field, field_validator, ValidationInfo
import jwt

from .combustion_control_orchestrator import CombustionControlOrchestrator
from .config import settings
from monitoring.metrics import metrics_collector
from greenlang.determinism import DeterministicClock
from .security_validator import validate_startup_security
from .websocket_handler import ws_manager, StreamType

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global agent instance
agent: Optional[CombustionControlOrchestrator] = None

# Security setup
security = HTTPBearer()

# Rate limiting tracker (simple in-memory for MVP)
# In production, use Redis or similar distributed storage
request_tracker: Dict[str, list] = {}


def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict[str, Any]:
    """
    Verify JWT token and return decoded payload

    Security requirements per IEC 62443-4-2:
    - Token-based authentication for all control endpoints
    - Token expiration validation
    - Algorithm verification (prevent algorithm confusion attacks)
    """
    try:
        token = credentials.credentials

        # Decode and verify token
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM]
        )

        # Verify token hasn't expired
        exp = payload.get('exp')
        if not exp or datetime.fromtimestamp(exp) < datetime.utcnow():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"}
            )

        return payload

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"}
        )


def check_rate_limit(client_id: str, max_requests: int = None) -> None:
    """
    Simple rate limiting (in-memory implementation)
    In production, use Redis with sliding window
    """
    if max_requests is None:
        max_requests = settings.RATE_LIMIT_PER_MINUTE

    now = datetime.utcnow()
    minute_ago = now - timedelta(minutes=1)

    # Clean old entries
    if client_id in request_tracker:
        request_tracker[client_id] = [
            req_time for req_time in request_tracker[client_id]
            if req_time > minute_ago
        ]
    else:
        request_tracker[client_id] = []

    # Check limit
    if len(request_tracker[client_id]) >= max_requests:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Maximum {max_requests} requests per minute."
        )

    # Record request
    request_tracker[client_id].append(now)


# Request/Response Models
class ControlRequest(BaseModel):
    """Request model for manual control trigger"""
    heat_demand_kw: Optional[float] = Field(
        None,
        description="Target heat output (kW)",
        ge=0,
        le=50000
    )
    override_interlocks: bool = Field(False, description="Override safety interlocks (use with caution)")

    @field_validator('heat_demand_kw')
    @classmethod
    def validate_heat_demand(cls, v: Optional[float]) -> Optional[float]:
        """
        Validate heat demand is within safe operating limits
        Per IEC 62443-4-2: Input validation for all control parameters
        """
        if v is None:
            return v

        # Range validation
        if v < 0:
            raise ValueError("heat_demand_kw must be non-negative")

        if v > settings.HEAT_OUTPUT_MAX_KW:
            raise ValueError(
                f"heat_demand_kw exceeds maximum safe limit: "
                f"{v} kW > {settings.HEAT_OUTPUT_MAX_KW} kW"
            )

        if v > 0 and v < settings.HEAT_OUTPUT_MIN_KW:
            raise ValueError(
                f"heat_demand_kw below minimum operating limit: "
                f"{v} kW < {settings.HEAT_OUTPUT_MIN_KW} kW. "
                f"Use 0 to shut down, or set to minimum {settings.HEAT_OUTPUT_MIN_KW} kW"
            )

        return v


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

    # CRITICAL: Validate security configuration before proceeding
    # This will abort startup if security requirements are not met
    try:
        logger.info("Running security validation checks...")
        validate_startup_security(fail_fast=True)
    except Exception as e:
        logger.critical(f"Security validation failed: {e}")
        logger.critical("STARTUP ABORTED - Fix security issues before deployment")
        raise

    try:
        # Initialize agent
        agent = CombustionControlOrchestrator()
        await agent.initialize_integrations()

        # Initialize WebSocket manager with agent reference
        ws_manager.set_agent(agent)
        await ws_manager.start_streaming()
        logger.info("WebSocket streaming initialized")

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

    # Stop WebSocket streaming first
    await ws_manager.stop_streaming()
    logger.info("WebSocket streaming stopped")

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
async def trigger_control_cycle(
    request: ControlRequest = None,
    token: Dict[str, Any] = Depends(verify_token)
) -> ControlResponse:
    """
    Trigger immediate control cycle (manual mode)
    Useful for testing or manual interventions

    SECURITY: Requires valid JWT token
    Per IEC 62443-4-2: Authentication required for all control operations
    """
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )

    # Rate limiting based on user/client ID from token
    client_id = token.get('sub', 'unknown')
    check_rate_limit(client_id, max_requests=60)  # More restrictive for control endpoints

    try:
        # Extract heat demand if provided
        heat_demand = request.heat_demand_kw if request else None

        # Log authenticated control action
        logger.info(
            f"Control cycle triggered by {client_id}, heat_demand={heat_demand} kW"
        )

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
async def enable_control(
    request: EnableControlRequest,
    token: Dict[str, Any] = Depends(verify_token)
) -> ControlResponse:
    """
    Enable or disable automatic control

    SECURITY: Requires valid JWT token
    Critical operation - changes control mode
    """
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )

    # Rate limiting
    client_id = token.get('sub', 'unknown')
    check_rate_limit(client_id, max_requests=20)  # Very restrictive for mode changes

    try:
        if request.enabled:
            agent.enable_control()
            message = "Automatic control enabled"
            logger.warning(f"Automatic control ENABLED by {client_id}")
        else:
            agent.disable_control()
            message = "Automatic control disabled (manual mode)"
            logger.warning(f"Automatic control DISABLED by {client_id}")

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


# =============================================================================
# WebSocket Real-Time Streaming Endpoint
# =============================================================================

@app.websocket("/ws/stream")
async def websocket_stream(
    websocket: WebSocket,
    token: str = Query(..., description="JWT authentication token")
):
    """
    WebSocket endpoint for real-time combustion data streaming.

    Authentication:
        Pass JWT token as query parameter: /ws/stream?token=<jwt_token>

    Streams:
        - combustion_state: 10Hz (100ms interval) - fuel flow, air flow, temps, O2, emissions
        - stability_metrics: 1Hz (1s interval) - stability scores, oscillation detection
        - control_action: On-demand - notifications when control actions are taken

    Client Messages:
        - {"type": "ping"} - Health check, server responds with pong
        - {"type": "subscribe", "streams": ["combustion_state"]} - Subscribe to streams
        - {"type": "unsubscribe", "streams": ["stability_metrics"]} - Unsubscribe from streams

    Rate Limiting:
        Maximum 5 connections per client (identified by JWT 'sub' claim)

    Connection Lifecycle:
        1. Connect with JWT token in query params
        2. Receive 'connected' message with subscription info
        3. Receive streaming data based on subscriptions
        4. Send 'ping' messages to maintain connection
        5. Connection closed on timeout, error, or server shutdown
    """
    # Attempt to connect and authenticate
    conn_info = await ws_manager.connect(websocket, token)

    if not conn_info:
        # Connection was rejected (auth failed or rate limited)
        return

    try:
        # Listen for client messages
        while True:
            try:
                # Wait for message with timeout
                message = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=60.0  # 60 second timeout
                )
                await ws_manager.handle_client_message(websocket, message)

            except asyncio.TimeoutError:
                # No message received, connection still alive
                continue

    except WebSocketDisconnect:
        logger.debug(f"WebSocket client disconnected normally")
    except Exception as e:
        logger.debug(f"WebSocket error: {e}")
    finally:
        await ws_manager.disconnect(websocket)


@app.get("/ws/stats")
async def get_websocket_stats() -> Dict[str, Any]:
    """
    Get WebSocket connection statistics.

    Returns current connection counts, stream rates, and client breakdown.
    """
    return ws_manager.get_connection_stats()


@app.post("/ws/broadcast/control-action")
async def broadcast_control_action_notification(
    token: Dict[str, Any] = Depends(verify_token)
) -> Dict[str, Any]:
    """
    Manually trigger a control action broadcast to all connected WebSocket clients.

    This endpoint is useful for testing WebSocket notifications.
    In normal operation, control actions are broadcast automatically by the agent.

    SECURITY: Requires valid JWT token
    """
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )

    if not agent.control_history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No control actions available to broadcast"
        )

    # Broadcast the latest control action
    latest_action = agent.control_history[-1]
    sent_count = await ws_manager.broadcast_control_action(latest_action)

    return {
        "success": True,
        "action_id": latest_action.action_id,
        "clients_notified": sent_count,
        "timestamp": DeterministicClock.utcnow().isoformat()
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
