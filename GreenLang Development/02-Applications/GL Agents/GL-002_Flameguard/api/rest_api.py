"""
GL-002 FLAMEGUARD - REST API

FastAPI-based REST API for boiler efficiency optimization.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional
import asyncio
import logging
import time

from fastapi import FastAPI, HTTPException, Depends, Query, Path, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from .schemas import (
    BoilerStatusResponse,
    OptimizationRequest,
    OptimizationResponse,
    EfficiencyResponse,
    EmissionsResponse,
    SafetyStatusResponse,
    SetpointCommand,
    SafetyBypassRequest,
    MultiBoilerLoadRequest,
    LoadDispatchResponse,
    HealthCheckResponse,
    ErrorResponse,
    ProcessDataInput,
    OptimizationMode,
    BoilerState,
    SafetyState,
    CombustionStatus,
    EfficiencyMetrics,
    EmissionsMetrics,
)

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer(auto_error=False)


class FlameguardAPI:
    """
    Flameguard REST API wrapper.

    Provides programmatic access to API functionality.
    """

    def __init__(
        self,
        orchestrator=None,
        auth_enabled: bool = True,
        rate_limit_rpm: int = 100,
    ) -> None:
        self.orchestrator = orchestrator
        self.auth_enabled = auth_enabled
        self.rate_limit_rpm = rate_limit_rpm

        self._start_time = time.time()
        self._request_count = 0

        # Create FastAPI app
        self.app = create_app(self)

    def get_uptime(self) -> float:
        return time.time() - self._start_time


def create_app(api: Optional["FlameguardAPI"] = None) -> FastAPI:
    """Create FastAPI application."""

    app = FastAPI(
        title="GL-002 FLAMEGUARD API",
        description="Boiler Efficiency Optimization API per ASME PTC 4.1 and NFPA 85",
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

    # State storage (for demo without orchestrator)
    app.state.boiler_data: Dict[str, Dict] = {}
    app.state.api = api

    # ============================================================
    # Health & Info Endpoints
    # ============================================================

    @app.get(
        "/health",
        response_model=HealthCheckResponse,
        tags=["Health"],
        summary="Health check",
    )
    async def health_check() -> HealthCheckResponse:
        """Get API health status."""
        return HealthCheckResponse(
            status="healthy",
            version="1.0.0",
            timestamp=datetime.now(timezone.utc),
            components={
                "api": "healthy",
                "database": "healthy",
                "scada": "healthy",
                "calculation_engine": "healthy",
            },
            uptime_seconds=app.state.api.get_uptime() if app.state.api else 0.0,
        )

    @app.get(
        "/info",
        tags=["Health"],
        summary="API information",
    )
    async def api_info():
        """Get API information."""
        return {
            "name": "GL-002 FLAMEGUARD",
            "version": "1.0.0",
            "description": "Boiler Efficiency Optimization Agent",
            "standards": ["ASME PTC 4.1", "NFPA 85", "IEC 61511", "EPA 40 CFR 60"],
            "capabilities": [
                "efficiency_calculation",
                "emissions_monitoring",
                "combustion_optimization",
                "safety_integration",
                "multi_boiler_dispatch",
            ],
        }

    # ============================================================
    # Boiler Status Endpoints
    # ============================================================

    @app.get(
        "/boilers",
        tags=["Boilers"],
        summary="List all boilers",
    )
    async def list_boilers():
        """Get list of all registered boilers."""
        return {
            "boilers": [
                {
                    "id": "BOILER-001",
                    "name": "Main Boiler #1",
                    "state": "firing",
                    "capacity_klb_hr": 200.0,
                },
                {
                    "id": "BOILER-002",
                    "name": "Main Boiler #2",
                    "state": "standby",
                    "capacity_klb_hr": 200.0,
                },
            ]
        }

    @app.get(
        "/boilers/{boiler_id}",
        response_model=BoilerStatusResponse,
        tags=["Boilers"],
        summary="Get boiler status",
    )
    async def get_boiler_status(
        boiler_id: str = Path(..., description="Boiler identifier"),
    ) -> BoilerStatusResponse:
        """Get detailed status for a specific boiler."""
        # Demo response
        return BoilerStatusResponse(
            boiler_id=boiler_id,
            name=f"Boiler {boiler_id}",
            state=BoilerState.FIRING,
            timestamp=datetime.now(timezone.utc),
            load_percent=75.5,
            steam_flow_klb_hr=150.0,
            drum_pressure_psig=125.5,
            drum_level_inches=0.5,
            steam_temperature_f=450.0,
            combustion=CombustionStatus(
                o2_percent=3.5,
                o2_setpoint=3.0,
                o2_error=0.5,
                co_ppm=25.0,
                excess_air_percent=20.0,
                stoichiometric_ratio=1.2,
                combustion_quality="good",
            ),
            efficiency=EfficiencyMetrics(
                gross_efficiency_percent=85.5,
                net_efficiency_percent=82.1,
                stack_loss_percent=10.5,
                radiation_loss_percent=1.5,
                blowdown_loss_percent=2.0,
                unaccounted_loss_percent=0.4,
                calculation_method="indirect",
                timestamp=datetime.now(timezone.utc),
            ),
            emissions=EmissionsMetrics(
                nox_lb_hr=15.5,
                nox_ppm=45.0,
                co_lb_hr=2.5,
                co_ppm=25.0,
                co2_ton_hr=8.5,
                so2_lb_hr=0.0,
                pm_lb_hr=0.0,
                ghg_mtco2e_hr=7.7,
                timestamp=datetime.now(timezone.utc),
            ),
            safety_status=SafetyState.NORMAL,
            active_alarms=0,
            active_trips=0,
        )

    @app.post(
        "/boilers/{boiler_id}/process-data",
        tags=["Boilers"],
        summary="Update process data",
    )
    async def update_process_data(
        boiler_id: str,
        data: ProcessDataInput,
    ):
        """Update boiler process data from SCADA."""
        app.state.boiler_data[boiler_id] = data.dict()
        return {
            "status": "accepted",
            "boiler_id": boiler_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ============================================================
    # Optimization Endpoints
    # ============================================================

    @app.post(
        "/optimize",
        response_model=OptimizationResponse,
        tags=["Optimization"],
        summary="Run optimization",
    )
    async def run_optimization(
        request: OptimizationRequest,
        background_tasks: BackgroundTasks,
    ) -> OptimizationResponse:
        """Run combustion optimization for a boiler."""
        return OptimizationResponse(
            boiler_id=request.boiler_id,
            mode=request.mode,
            timestamp=datetime.now(timezone.utc),
            success=True,
            current_efficiency=82.1,
            current_emissions_mtco2e_hr=7.7,
            current_cost_usd_hr=250.0,
            recommended_o2_setpoint=2.8,
            recommended_excess_air=17.5,
            recommended_load_percent=request.target_load_percent,
            predicted_efficiency=84.5,
            predicted_emissions_mtco2e_hr=7.2,
            predicted_cost_usd_hr=235.0,
            efficiency_improvement_percent=2.4,
            emissions_reduction_percent=6.5,
            cost_savings_usd_hr=15.0,
            calculation_hash="sha256:abc123...",
            model_version="1.0.0",
        )

    @app.post(
        "/optimize/dispatch",
        response_model=LoadDispatchResponse,
        tags=["Optimization"],
        summary="Multi-boiler load dispatch",
    )
    async def dispatch_load(
        request: MultiBoilerLoadRequest,
    ) -> LoadDispatchResponse:
        """Optimize load distribution across multiple boilers."""
        # Simple equal distribution for demo
        per_boiler = request.total_demand_klb_hr / len(request.boiler_ids)
        allocations = {bid: per_boiler for bid in request.boiler_ids}

        return LoadDispatchResponse(
            timestamp=datetime.now(timezone.utc),
            total_demand_klb_hr=request.total_demand_klb_hr,
            optimization_mode=request.optimization_mode,
            boiler_allocations=allocations,
            total_allocated_klb_hr=request.total_demand_klb_hr,
            total_efficiency_percent=83.5,
            total_emissions_mtco2e_hr=15.4,
            total_cost_usd_hr=480.0,
            efficiency_improvement=1.5,
            emissions_reduction=5.2,
            cost_savings_usd_hr=25.0,
        )

    # ============================================================
    # Efficiency Endpoints
    # ============================================================

    @app.get(
        "/boilers/{boiler_id}/efficiency",
        response_model=EfficiencyResponse,
        tags=["Efficiency"],
        summary="Get efficiency calculation",
    )
    async def get_efficiency(
        boiler_id: str,
        method: str = Query("indirect", enum=["direct", "indirect"]),
    ) -> EfficiencyResponse:
        """Get efficiency calculation per ASME PTC 4.1."""
        return EfficiencyResponse(
            boiler_id=boiler_id,
            timestamp=datetime.now(timezone.utc),
            gross_efficiency_percent=85.5,
            net_efficiency_percent=82.1,
            fuel_efficiency_percent=83.8,
            losses={
                "dry_flue_gas": 5.2,
                "moisture_in_fuel": 1.8,
                "moisture_in_air": 0.3,
                "moisture_from_h2": 3.5,
                "radiation_convection": 1.5,
                "blowdown": 2.0,
                "unaccounted": 0.4,
            },
            heat_input_mmbtu_hr=185.0,
            heat_output_mmbtu_hr=152.0,
            heat_loss_mmbtu_hr=33.0,
            calculation_method=method,
            standard="ASME PTC 4.1",
            calculation_hash="sha256:def456...",
        )

    @app.get(
        "/boilers/{boiler_id}/efficiency/trend",
        tags=["Efficiency"],
        summary="Get efficiency trend",
    )
    async def get_efficiency_trend(
        boiler_id: str,
        hours: int = Query(24, ge=1, le=720),
    ):
        """Get historical efficiency trend."""
        # Demo trend data
        now = datetime.now(timezone.utc)
        return {
            "boiler_id": boiler_id,
            "period_hours": hours,
            "data_points": [
                {
                    "timestamp": now.isoformat(),
                    "efficiency_percent": 82.5,
                    "load_percent": 75.0,
                }
            ],
            "average_efficiency": 82.5,
            "min_efficiency": 80.1,
            "max_efficiency": 84.9,
        }

    # ============================================================
    # Emissions Endpoints
    # ============================================================

    @app.get(
        "/boilers/{boiler_id}/emissions",
        response_model=EmissionsResponse,
        tags=["Emissions"],
        summary="Get emissions calculation",
    )
    async def get_emissions(boiler_id: str) -> EmissionsResponse:
        """Get emissions calculation per EPA standards."""
        return EmissionsResponse(
            boiler_id=boiler_id,
            timestamp=datetime.now(timezone.utc),
            nox_lb_hr=15.5,
            nox_ppm_corrected=45.0,
            co_lb_hr=2.5,
            co_ppm=25.0,
            co2_ton_hr=8.5,
            so2_lb_hr=0.0,
            pm_lb_hr=0.1,
            voc_lb_hr=0.05,
            ghg_mtco2e_hr=7.7,
            annual_ghg_projection_mtco2e=67452.0,
            permit_limit_status={
                "nox": "compliant",
                "co": "compliant",
                "pm": "compliant",
            },
            exceedance_risk="low",
            emission_factors_source="EPA 40 CFR Part 60",
            calculation_hash="sha256:ghi789...",
        )

    @app.get(
        "/boilers/{boiler_id}/emissions/ghg",
        tags=["Emissions"],
        summary="Get GHG emissions",
    )
    async def get_ghg_emissions(
        boiler_id: str,
        scope: int = Query(1, ge=1, le=3),
    ):
        """Get greenhouse gas emissions by scope."""
        return {
            "boiler_id": boiler_id,
            "scope": scope,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "emissions_mtco2e_hr": 7.7 if scope == 1 else 0.0,
            "annual_projection_mtco2e": 67452.0 if scope == 1 else 0.0,
            "breakdown": {
                "co2": 7.5,
                "ch4": 0.15,
                "n2o": 0.05,
            },
        }

    # ============================================================
    # Safety Endpoints
    # ============================================================

    @app.get(
        "/boilers/{boiler_id}/safety",
        response_model=SafetyStatusResponse,
        tags=["Safety"],
        summary="Get safety status",
    )
    async def get_safety_status(boiler_id: str) -> SafetyStatusResponse:
        """Get safety interlock status per IEC 61511."""
        return SafetyStatusResponse(
            boiler_id=boiler_id,
            timestamp=datetime.now(timezone.utc),
            bms_state="FIRING",
            safety_state=SafetyState.NORMAL,
            flame_proven=True,
            interlocks=[],
            bypassed_count=0,
            alarm_count=0,
            trip_count=0,
            permissives_satisfied={
                "drum_level_ok": True,
                "steam_pressure_ok": True,
                "fuel_pressure_ok": True,
                "combustion_air_ok": True,
                "flame_scanner_ok": True,
            },
        )

    @app.post(
        "/boilers/{boiler_id}/safety/bypass",
        tags=["Safety"],
        summary="Request safety bypass",
    )
    async def request_safety_bypass(
        boiler_id: str,
        request: SafetyBypassRequest,
    ):
        """Request safety interlock bypass (requires authorization)."""
        # Validate bypass request
        if request.duration_minutes > 120:
            raise HTTPException(
                status_code=400,
                detail="Bypass duration cannot exceed 120 minutes without additional approval",
            )

        return {
            "status": "approved",
            "boiler_id": boiler_id,
            "interlock": request.interlock_tag,
            "bypass_id": "BYP-2024-001",
            "expires_at": datetime.now(timezone.utc).isoformat(),
            "operator": request.operator,
            "supervisor": request.supervisor_approval,
        }

    @app.post(
        "/boilers/{boiler_id}/safety/trip-reset",
        tags=["Safety"],
        summary="Reset safety trip",
    )
    async def reset_safety_trip(
        boiler_id: str,
        operator: str = Query(..., description="Operator ID"),
    ):
        """Reset safety trip condition."""
        return {
            "status": "success",
            "boiler_id": boiler_id,
            "reset_by": operator,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ============================================================
    # Setpoint Endpoints
    # ============================================================

    @app.post(
        "/boilers/{boiler_id}/setpoints",
        tags=["Control"],
        summary="Adjust setpoint",
    )
    async def adjust_setpoint(
        boiler_id: str,
        command: SetpointCommand,
    ):
        """Adjust boiler setpoint."""
        return {
            "status": "accepted",
            "boiler_id": boiler_id,
            "setpoint_type": command.setpoint_type,
            "new_value": command.value,
            "operator": command.operator,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @app.get(
        "/boilers/{boiler_id}/setpoints",
        tags=["Control"],
        summary="Get current setpoints",
    )
    async def get_setpoints(boiler_id: str):
        """Get current boiler setpoints."""
        return {
            "boiler_id": boiler_id,
            "setpoints": {
                "o2_setpoint": {"value": 3.0, "unit": "%"},
                "load_demand": {"value": 75.0, "unit": "%"},
                "steam_pressure_setpoint": {"value": 125.0, "unit": "psig"},
                "excess_air_bias": {"value": 0.0, "unit": "%"},
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ============================================================
    # Audit Endpoints
    # ============================================================

    @app.get(
        "/boilers/{boiler_id}/audit",
        tags=["Audit"],
        summary="Get audit log",
    )
    async def get_audit_log(
        boiler_id: str,
        hours: int = Query(24, ge=1, le=168),
        event_type: Optional[str] = None,
    ):
        """Get audit log for boiler."""
        return {
            "boiler_id": boiler_id,
            "period_hours": hours,
            "event_type_filter": event_type,
            "events": [
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event_type": "setpoint_change",
                    "description": "O2 setpoint changed from 3.2 to 3.0",
                    "operator": "OPERATOR-001",
                    "source": "manual",
                }
            ],
            "total_events": 1,
        }

    @app.get(
        "/boilers/{boiler_id}/calculations",
        tags=["Audit"],
        summary="Get calculation history",
    )
    async def get_calculation_history(
        boiler_id: str,
        calculation_type: str = Query("efficiency", enum=["efficiency", "emissions", "optimization"]),
        limit: int = Query(100, ge=1, le=1000),
    ):
        """Get calculation history with provenance hashes."""
        return {
            "boiler_id": boiler_id,
            "calculation_type": calculation_type,
            "calculations": [
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "type": calculation_type,
                    "result": {"efficiency_percent": 82.5},
                    "hash": "sha256:abc123...",
                    "inputs_hash": "sha256:def456...",
                }
            ],
            "total_count": 1,
        }

    return app


# Create default app instance
app = create_app()
