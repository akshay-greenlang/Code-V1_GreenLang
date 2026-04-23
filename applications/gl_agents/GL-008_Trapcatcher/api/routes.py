# -*- coding: utf-8 -*-
"""
FastAPI Routes for GL-008 TRAPCATCHER

REST API endpoints for steam trap monitoring and diagnostics.

Endpoints:
- POST /diagnose: Single trap diagnostic
- POST /analyze/fleet: Fleet-wide analysis
- GET /health: Health check
- GET /status: Agent status and statistics
- GET /explain/{trap_id}: Get explanation for last diagnosis

Author: GL-APIEngineer
Date: December 2025
Version: 1.0.0
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Import agent components
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import TrapcatcherAgent, TrapDiagnosticInput, AgentConfig

router = APIRouter(prefix="/api/v1", tags=["trapcatcher"])

# Global agent instance
_agent: Optional[TrapcatcherAgent] = None


def get_agent() -> TrapcatcherAgent:
    """Get or create the global agent instance."""
    global _agent
    if _agent is None:
        _agent = TrapcatcherAgent()
    return _agent


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class DiagnosticRequest(BaseModel):
    """Request model for single trap diagnostic."""
    trap_id: str = Field(..., description="Unique trap identifier")
    acoustic_amplitude_db: Optional[float] = Field(None, description="Peak acoustic amplitude (dB)")
    acoustic_frequency_khz: Optional[float] = Field(None, description="Dominant frequency (kHz)")
    inlet_temp_c: Optional[float] = Field(None, description="Inlet temperature (Celsius)")
    outlet_temp_c: Optional[float] = Field(None, description="Outlet temperature (Celsius)")
    pressure_bar_g: float = Field(10.0, description="Operating pressure (bar gauge)")
    trap_type: str = Field("thermodynamic", description="Type of steam trap")
    orifice_diameter_mm: float = Field(6.35, description="Orifice diameter (mm)")
    trap_age_years: float = Field(0.0, description="Age of trap in years")
    last_maintenance_days: int = Field(0, description="Days since last maintenance")
    location: str = Field("", description="Physical location")
    system: str = Field("", description="Steam system identifier")
    include_explanation: bool = Field(True, description="Include SHAP explanation")


class DiagnosticResponse(BaseModel):
    """Response model for diagnostic result."""
    trap_id: str
    timestamp: str
    condition: str
    severity: str
    confidence: float
    energy_loss_kw: float
    annual_cost_usd: float
    annual_co2_kg: float
    recommended_action: str
    alert_level: str
    provenance_hash: str
    explanation: Optional[Dict[str, Any]] = None


class FleetRequest(BaseModel):
    """Request model for fleet analysis."""
    traps: List[DiagnosticRequest]


class FleetSummaryResponse(BaseModel):
    """Response model for fleet summary."""
    total_traps: int
    healthy_count: int
    failed_count: int
    leaking_count: int
    unknown_count: int
    total_energy_loss_kw: float
    total_annual_cost_usd: float
    total_annual_co2_kg: float
    fleet_health_score: float
    critical_alerts: int


class FleetResponse(BaseModel):
    """Response model for fleet analysis."""
    summary: FleetSummaryResponse
    diagnostics: List[DiagnosticResponse]


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: str
    agent_id: str
    version: str


class StatusResponse(BaseModel):
    """Response model for agent status."""
    agent_id: str
    agent_name: str
    version: str
    mode: str
    status: str
    statistics: Dict[str, int]
    components: Dict[str, Any]


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns basic health status of the TRAPCATCHER agent.
    """
    agent = get_agent()
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        agent_id=agent.config.agent_id,
        version=agent.config.version
    )


@router.get("/status", response_model=StatusResponse)
async def get_status():
    """
    Get agent status and statistics.

    Returns detailed status including diagnostic counts and component status.
    """
    agent = get_agent()
    status = agent.get_status()
    return StatusResponse(**status)


@router.post("/diagnose", response_model=DiagnosticResponse)
async def diagnose_trap(request: DiagnosticRequest):
    """
    Perform diagnostic on a single steam trap.

    Analyzes acoustic and thermal data to classify trap condition,
    estimate energy loss, and provide recommendations.

    Zero-Hallucination: Uses deterministic classification algorithms.
    """
    agent = get_agent()

    # Convert request to input
    input_data = TrapDiagnosticInput(
        trap_id=request.trap_id,
        acoustic_amplitude_db=request.acoustic_amplitude_db,
        acoustic_frequency_khz=request.acoustic_frequency_khz,
        inlet_temp_c=request.inlet_temp_c,
        outlet_temp_c=request.outlet_temp_c,
        pressure_bar_g=request.pressure_bar_g,
        trap_type=request.trap_type,
        orifice_diameter_mm=request.orifice_diameter_mm,
        trap_age_years=request.trap_age_years,
        last_maintenance_days=request.last_maintenance_days,
        location=request.location,
        system=request.system
    )

    try:
        result = agent.diagnose_trap(input_data, include_explanation=request.include_explanation)

        response_data = {
            "trap_id": result.trap_id,
            "timestamp": result.timestamp.isoformat(),
            "condition": result.condition,
            "severity": result.severity,
            "confidence": result.confidence,
            "energy_loss_kw": result.energy_loss_kw,
            "annual_cost_usd": result.annual_cost_usd,
            "annual_co2_kg": result.annual_co2_kg,
            "recommended_action": result.recommended_action,
            "alert_level": result.alert_level.value,
            "provenance_hash": result.provenance_hash,
            "explanation": result.explanation.to_dict() if result.explanation else None
        }

        return DiagnosticResponse(**response_data)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Diagnostic error: {str(e)}"
        )


@router.post("/analyze/fleet", response_model=FleetResponse)
async def analyze_fleet(request: FleetRequest):
    """
    Analyze entire fleet of steam traps.

    Performs diagnostic on all provided traps and returns individual
    results plus fleet-wide summary statistics.

    Zero-Hallucination: Uses deterministic calculations for all metrics.
    """
    agent = get_agent()

    # Convert requests to inputs
    inputs = []
    for trap in request.traps:
        inputs.append(TrapDiagnosticInput(
            trap_id=trap.trap_id,
            acoustic_amplitude_db=trap.acoustic_amplitude_db,
            acoustic_frequency_khz=trap.acoustic_frequency_khz,
            inlet_temp_c=trap.inlet_temp_c,
            outlet_temp_c=trap.outlet_temp_c,
            pressure_bar_g=trap.pressure_bar_g,
            trap_type=trap.trap_type,
            orifice_diameter_mm=trap.orifice_diameter_mm,
            trap_age_years=trap.trap_age_years,
            last_maintenance_days=trap.last_maintenance_days,
            location=trap.location,
            system=trap.system
        ))

    try:
        results, summary = agent.analyze_fleet(inputs)

        # Convert results to response format
        diagnostics = []
        for result in results:
            diagnostics.append(DiagnosticResponse(
                trap_id=result.trap_id,
                timestamp=result.timestamp.isoformat(),
                condition=result.condition,
                severity=result.severity,
                confidence=result.confidence,
                energy_loss_kw=result.energy_loss_kw,
                annual_cost_usd=result.annual_cost_usd,
                annual_co2_kg=result.annual_co2_kg,
                recommended_action=result.recommended_action,
                alert_level=result.alert_level.value,
                provenance_hash=result.provenance_hash,
                explanation=None  # Fleet analysis excludes explanations for performance
            ))

        fleet_summary = FleetSummaryResponse(
            total_traps=summary.total_traps,
            healthy_count=summary.healthy_count,
            failed_count=summary.failed_count,
            leaking_count=summary.leaking_count,
            unknown_count=summary.unknown_count,
            total_energy_loss_kw=summary.total_energy_loss_kw,
            total_annual_cost_usd=summary.total_annual_cost_usd,
            total_annual_co2_kg=summary.total_annual_co2_kg,
            fleet_health_score=summary.fleet_health_score,
            critical_alerts=summary.critical_alerts
        )

        return FleetResponse(summary=fleet_summary, diagnostics=diagnostics)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fleet analysis error: {str(e)}"
        )


# ============================================================================
# APP FACTORY
# ============================================================================

def create_app() -> FastAPI:
    """
    Create FastAPI application for TRAPCATCHER.

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="GL-008 TRAPCATCHER API",
        description="Steam Trap Monitoring and Diagnostic Agent API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    app.include_router(router)

    @app.on_event("startup")
    async def startup():
        """Initialize agent on startup."""
        get_agent()

    return app


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8008)
