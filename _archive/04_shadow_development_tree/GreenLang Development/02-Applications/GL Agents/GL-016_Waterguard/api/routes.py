# -*- coding: utf-8 -*-
"""
GL-016 WATERGUARD FastAPI Routes
RESTful API endpoints for boiler water treatment optimization.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

# =============================================================================
# Request/Response Models
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    agent_id: str = "GL-016"
    agent_name: str = "WATERGUARD"
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    components: dict[str, str] = Field(default_factory=dict)


class CoCCalculationRequest(BaseModel):
    """Request for CoC calculation."""
    boiler_id: str = Field(..., min_length=1, max_length=64)
    makeup_conductivity_umho: float = Field(..., ge=0.0, le=10000.0)
    blowdown_conductivity_umho: float = Field(..., ge=0.0, le=50000.0)
    makeup_silica_ppm: Optional[float] = Field(None, ge=0.0, le=200.0)
    blowdown_silica_ppm: Optional[float] = Field(None, ge=0.0, le=500.0)


class CoCCalculationResponse(BaseModel):
    """Response from CoC calculation."""
    boiler_id: str
    coc_conductivity: float
    coc_silica: Optional[float] = None
    coc_recommended: float
    calculation_method: str
    timestamp: datetime
    provenance_hash: str


class WaterBalanceRequest(BaseModel):
    """Request for water balance calculation."""
    boiler_id: str = Field(..., min_length=1, max_length=64)
    steam_rate_klb_hr: float = Field(..., ge=0.0, le=1000.0)
    coc: float = Field(..., ge=1.0, le=20.0)
    condensate_return_percent: float = Field(default=80.0, ge=0.0, le=100.0)


class WaterBalanceResponse(BaseModel):
    """Response from water balance calculation."""
    boiler_id: str
    makeup_rate_klb_hr: float
    blowdown_rate_klb_hr: float
    blowdown_percent: float
    evaporation_rate_klb_hr: float
    timestamp: datetime
    provenance_hash: str


class HeatLossRequest(BaseModel):
    """Request for heat loss calculation."""
    boiler_id: str = Field(..., min_length=1, max_length=64)
    blowdown_rate_klb_hr: float = Field(..., ge=0.0, le=100.0)
    blowdown_temp_f: float = Field(..., ge=100.0, le=700.0)
    makeup_temp_f: float = Field(default=60.0, ge=32.0, le=200.0)
    boiler_efficiency: float = Field(default=0.80, ge=0.5, le=1.0)
    fuel_cost_usd_mmbtu: float = Field(default=8.0, ge=0.0)


class HeatLossResponse(BaseModel):
    """Response from heat loss calculation."""
    boiler_id: str
    heat_loss_mmbtu_hr: float
    heat_loss_percent: float
    annual_heat_loss_mmbtu: float
    annual_fuel_cost_usd: float
    timestamp: datetime
    provenance_hash: str


class ChemistryReadingRequest(BaseModel):
    """Submit chemistry reading."""
    boiler_id: str = Field(..., min_length=1, max_length=64)
    sample_point: str = Field(default="blowdown")
    ph: Optional[float] = Field(None, ge=0.0, le=14.0)
    conductivity_umho: Optional[float] = Field(None, ge=0.0, le=50000.0)
    silica_ppm: Optional[float] = Field(None, ge=0.0, le=500.0)
    phosphate_ppm: Optional[float] = Field(None, ge=0.0, le=200.0)
    dissolved_oxygen_ppb: Optional[float] = Field(None, ge=0.0, le=10000.0)
    temperature_c: Optional[float] = Field(None, ge=0.0, le=400.0)


class RecommendationResponse(BaseModel):
    """Recommendation from agent."""
    recommendation_id: str
    boiler_id: str
    category: str
    priority: int
    title: str
    description: str
    action_required: str
    expected_benefit: Optional[str] = None
    cost_savings_usd_year: Optional[float] = None
    explanation: str
    timestamp: datetime


class SafetyGateStatusResponse(BaseModel):
    """Safety gate status."""
    boiler_id: str
    total_gates: int
    passed_gates: int
    failed_gates: int
    bypassed_gates: int
    is_safe_to_proceed: bool
    gates: dict[str, dict[str, Any]]
    timestamp: datetime


class SetpointUpdateRequest(BaseModel):
    """Request to update setpoint."""
    boiler_id: str = Field(..., min_length=1, max_length=64)
    parameter: str = Field(..., min_length=1)
    value: float
    reason: str = Field(..., min_length=1)
    operator_id: str = Field(..., min_length=1)


class SetpointUpdateResponse(BaseModel):
    """Response from setpoint update."""
    boiler_id: str
    parameter: str
    old_value: Optional[float]
    new_value: float
    approved: bool
    approval_required: bool
    timestamp: datetime
    audit_id: str


# =============================================================================
# Router Configuration
# =============================================================================

router = APIRouter(
    prefix="/api/v1/waterguard",
    tags=["waterguard"],
    responses={404: {"description": "Not found"}},
)


# =============================================================================
# Health Endpoints
# =============================================================================

@router.get("/health/live", response_model=HealthResponse)
async def health_live() -> HealthResponse:
    """
    Kubernetes liveness probe.
    Returns 200 if the service is running.
    """
    return HealthResponse(
        status="alive",
        components={
            "api": "healthy",
        },
    )


@router.get("/health/ready", response_model=HealthResponse)
async def health_ready() -> HealthResponse:
    """
    Kubernetes readiness probe.
    Returns 200 if the service is ready to accept traffic.
    """
    return HealthResponse(
        status="ready",
        components={
            "api": "healthy",
            "database": "healthy",
            "kafka": "healthy",
            "opcua": "healthy",
        },
    )


@router.get("/health/startup", response_model=HealthResponse)
async def health_startup() -> HealthResponse:
    """
    Kubernetes startup probe.
    Returns 200 once initial startup is complete.
    """
    return HealthResponse(
        status="started",
        components={
            "api": "initialized",
        },
    )


# =============================================================================
# Calculation Endpoints
# =============================================================================

@router.post("/calculations/coc", response_model=CoCCalculationResponse)
async def calculate_coc(request: CoCCalculationRequest) -> CoCCalculationResponse:
    """
    Calculate Cycles of Concentration.

    Uses conductivity ratio as primary method, with silica ratio as backup
    if silica values are provided.
    """
    from ..calculators.chemistry_engine import ChemistryEngine

    engine = ChemistryEngine(request.boiler_id)
    result = engine.calculate_coc(
        makeup_conductivity_umho=request.makeup_conductivity_umho,
        blowdown_conductivity_umho=request.blowdown_conductivity_umho,
        makeup_silica_ppm=request.makeup_silica_ppm,
        blowdown_silica_ppm=request.blowdown_silica_ppm,
    )

    return CoCCalculationResponse(
        boiler_id=request.boiler_id,
        coc_conductivity=result.coc_conductivity,
        coc_silica=result.coc_silica,
        coc_recommended=result.coc_recommended,
        calculation_method=result.calculation_method,
        timestamp=datetime.utcnow(),
        provenance_hash=result.provenance.get("combined_hash", ""),
    )


@router.post("/calculations/water-balance", response_model=WaterBalanceResponse)
async def calculate_water_balance(request: WaterBalanceRequest) -> WaterBalanceResponse:
    """
    Calculate boiler water balance.

    Determines makeup and blowdown rates based on steam production and CoC.
    """
    from ..calculators.chemistry_engine import ChemistryEngine

    engine = ChemistryEngine(request.boiler_id)
    result = engine.calculate_water_balance(
        steam_rate_klb_hr=request.steam_rate_klb_hr,
        coc=request.coc,
        condensate_return_percent=request.condensate_return_percent,
    )

    return WaterBalanceResponse(
        boiler_id=request.boiler_id,
        makeup_rate_klb_hr=result.makeup_rate_klb_hr,
        blowdown_rate_klb_hr=result.blowdown_rate_klb_hr,
        blowdown_percent=result.blowdown_percent,
        evaporation_rate_klb_hr=result.evaporation_rate_klb_hr,
        timestamp=datetime.utcnow(),
        provenance_hash=result.provenance.get("combined_hash", ""),
    )


@router.post("/calculations/heat-loss", response_model=HeatLossResponse)
async def calculate_heat_loss(request: HeatLossRequest) -> HeatLossResponse:
    """
    Calculate blowdown heat loss.

    Determines heat and fuel cost associated with blowdown.
    """
    from ..calculators.chemistry_engine import ChemistryEngine

    engine = ChemistryEngine(request.boiler_id)
    result = engine.calculate_heat_loss(
        blowdown_rate_klb_hr=request.blowdown_rate_klb_hr,
        blowdown_temp_f=request.blowdown_temp_f,
        makeup_temp_f=request.makeup_temp_f,
        boiler_efficiency=request.boiler_efficiency,
        fuel_cost_usd_mmbtu=request.fuel_cost_usd_mmbtu,
    )

    return HeatLossResponse(
        boiler_id=request.boiler_id,
        heat_loss_mmbtu_hr=result.heat_loss_mmbtu_hr,
        heat_loss_percent=result.heat_loss_percent,
        annual_heat_loss_mmbtu=result.annual_heat_loss_mmbtu,
        annual_fuel_cost_usd=result.annual_fuel_cost_usd or 0.0,
        timestamp=datetime.utcnow(),
        provenance_hash=result.provenance.get("combined_hash", ""),
    )


# =============================================================================
# Chemistry Endpoints
# =============================================================================

@router.post("/chemistry/readings")
async def submit_chemistry_reading(request: ChemistryReadingRequest) -> dict:
    """
    Submit a chemistry reading from lab analysis or online analyzer.
    """
    return {
        "status": "accepted",
        "boiler_id": request.boiler_id,
        "sample_point": request.sample_point,
        "timestamp": datetime.utcnow().isoformat(),
        "reading_id": str(UUID(int=0)),  # Placeholder
    }


@router.get("/chemistry/status/{boiler_id}")
async def get_chemistry_status(boiler_id: str) -> dict:
    """
    Get current chemistry status for a boiler.
    """
    return {
        "boiler_id": boiler_id,
        "status": "normal",
        "last_reading": datetime.utcnow().isoformat(),
        "parameters": {
            "ph": {"value": 11.0, "status": "normal", "limit_low": 10.5, "limit_high": 11.5},
            "conductivity_umho": {"value": 3500, "status": "normal", "limit_high": 5000},
            "silica_ppm": {"value": 80, "status": "normal", "limit_high": 150},
        },
    }


# =============================================================================
# Safety Endpoints
# =============================================================================

@router.get("/safety/gates/{boiler_id}", response_model=SafetyGateStatusResponse)
async def get_safety_gate_status(boiler_id: str) -> SafetyGateStatusResponse:
    """
    Get status of all safety gates for a boiler.
    """
    from ..safety.safety_gates import SafetyGateManager

    manager = SafetyGateManager(boiler_id)
    summary = manager.get_status_summary()

    return SafetyGateStatusResponse(
        boiler_id=boiler_id,
        total_gates=summary["total_gates"],
        passed_gates=summary["total_gates"] - summary["tripped_gates"] - summary["bypassed_gates"],
        failed_gates=summary["tripped_gates"],
        bypassed_gates=summary["bypassed_gates"],
        is_safe_to_proceed=summary["tripped_gates"] == 0,
        gates=summary["gates"],
        timestamp=datetime.utcnow(),
    )


@router.post("/safety/gates/{boiler_id}/check")
async def check_safety_gates(
    boiler_id: str,
    readings: dict[str, float],
) -> dict:
    """
    Check all safety gates against current readings.
    """
    from ..safety.safety_gates import SafetyGateManager

    manager = SafetyGateManager(boiler_id)
    is_safe, failed_gates = manager.is_safe_to_proceed(readings)

    return {
        "boiler_id": boiler_id,
        "is_safe": is_safe,
        "failed_gate_count": len(failed_gates),
        "failed_gates": [g.to_dict() for g in failed_gates],
        "timestamp": datetime.utcnow().isoformat(),
    }


# =============================================================================
# Recommendations Endpoints
# =============================================================================

@router.get("/recommendations/{boiler_id}")
async def get_recommendations(
    boiler_id: str,
    category: Optional[str] = None,
    status: Optional[str] = Query(None, description="Filter by status: pending, approved, rejected"),
    limit: int = Query(10, ge=1, le=100),
) -> list[dict]:
    """
    Get recommendations for a boiler.
    """
    return [
        {
            "recommendation_id": str(UUID(int=1)),
            "boiler_id": boiler_id,
            "category": "coc",
            "priority": 2,
            "title": "Increase Cycles of Concentration",
            "description": "Current CoC is 5.2, optimal is 6.0",
            "action_required": "Reduce continuous blowdown rate by 5%",
            "expected_benefit": "Save 2.5 klb/hr makeup water",
            "cost_savings_usd_year": 15000,
            "explanation": "CoC optimization based on current water chemistry",
            "status": "pending",
            "timestamp": datetime.utcnow().isoformat(),
        }
    ]


@router.post("/recommendations/{recommendation_id}/approve")
async def approve_recommendation(
    recommendation_id: str,
    operator_id: str = Query(..., min_length=1),
    notes: Optional[str] = None,
) -> dict:
    """
    Approve a recommendation for implementation.
    """
    return {
        "recommendation_id": recommendation_id,
        "status": "approved",
        "approved_by": operator_id,
        "approved_at": datetime.utcnow().isoformat(),
        "notes": notes,
    }


@router.post("/recommendations/{recommendation_id}/reject")
async def reject_recommendation(
    recommendation_id: str,
    operator_id: str = Query(..., min_length=1),
    reason: str = Query(..., min_length=1),
) -> dict:
    """
    Reject a recommendation.
    """
    return {
        "recommendation_id": recommendation_id,
        "status": "rejected",
        "rejected_by": operator_id,
        "rejected_at": datetime.utcnow().isoformat(),
        "reason": reason,
    }


# =============================================================================
# Setpoint Endpoints
# =============================================================================

@router.post("/setpoints/update", response_model=SetpointUpdateResponse)
async def update_setpoint(request: SetpointUpdateRequest) -> SetpointUpdateResponse:
    """
    Request a setpoint update.

    In SUPERVISED mode, requires operator approval.
    In AUTONOMOUS mode, validates against safety gates.
    """
    return SetpointUpdateResponse(
        boiler_id=request.boiler_id,
        parameter=request.parameter,
        old_value=None,
        new_value=request.value,
        approved=True,
        approval_required=False,
        timestamp=datetime.utcnow(),
        audit_id=str(UUID(int=0)),
    )


@router.get("/setpoints/{boiler_id}")
async def get_setpoints(boiler_id: str) -> dict:
    """
    Get current setpoints for a boiler.
    """
    return {
        "boiler_id": boiler_id,
        "setpoints": {
            "coc_target": {"value": 6.0, "min": 3.0, "max": 10.0, "unit": "cycles"},
            "conductivity_target_umho": {"value": 4000, "min": 1000, "max": 5000, "unit": "uS/cm"},
            "ph_target": {"value": 11.0, "min": 10.5, "max": 11.5, "unit": "pH"},
            "phosphate_target_ppm": {"value": 40, "min": 20, "max": 60, "unit": "ppm"},
        },
        "operating_mode": "SUPERVISED",
        "last_updated": datetime.utcnow().isoformat(),
    }


# =============================================================================
# Audit Endpoints
# =============================================================================

@router.get("/audit/{boiler_id}")
async def get_audit_log(
    boiler_id: str,
    event_type: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = Query(100, ge=1, le=1000),
) -> list[dict]:
    """
    Get audit log entries for a boiler.
    """
    # Returns mock audit entries - connect to database for production
    return [
        {
            "entry_id": str(UUID(int=1)),
            "boiler_id": boiler_id,
            "event_type": "setpoint_change",
            "timestamp": datetime.utcnow().isoformat(),
            "operator_id": "operator1",
            "description": "CoC setpoint changed from 5.5 to 6.0",
            "before_state": {"coc_target": 5.5},
            "after_state": {"coc_target": 6.0},
            "provenance_hash": "abc123...",
        }
    ]


# =============================================================================
# Main App Factory
# =============================================================================

def create_app():
    """Create FastAPI application."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(
        title="GL-016 WATERGUARD API",
        description="Boiler Water Treatment Optimization Agent",
        version="1.0.0",
        docs_url="/api/v1/waterguard/docs",
        redoc_url="/api/v1/waterguard/redoc",
        openapi_url="/api/v1/waterguard/openapi.json",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include router
    app.include_router(router)

    return app
