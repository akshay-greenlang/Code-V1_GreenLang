# -*- coding: utf-8 -*-
"""
Flaring REST API Router - AGENT-MRV-006
==========================================

20 REST endpoints for the Flaring Agent (GL-MRV-SCOPE1-006).

Prefix: ``/api/v1/flaring``

Endpoints:
     1. POST   /calculate              - Calculate flaring emissions
     2. POST   /calculate/batch        - Batch calculate flaring emissions
     3. GET    /calculations           - List calculations (paginated)
     4. GET    /calculations/{calc_id} - Get calculation details
     5. POST   /flares                 - Register flare system
     6. GET    /flares                 - List flare systems
     7. GET    /flares/{flare_id}      - Get flare system details
     8. POST   /events                 - Log flaring event
     9. GET    /events                 - List flaring events
    10. GET    /events/{event_id}      - Get event details
    11. POST   /compositions           - Register gas composition
    12. GET    /compositions           - List gas compositions
    13. POST   /factors                - Register custom emission factor
    14. GET    /factors                - List emission factors
    15. POST   /efficiency             - Log combustion efficiency test
    16. GET    /efficiency             - List CE records
    17. POST   /uncertainty            - Run uncertainty analysis
    18. POST   /compliance/check       - Run compliance check
    19. GET    /health                 - Health check
    20. GET    /stats                  - Service statistics

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-006 Flaring Agent (GL-MRV-SCOPE1-006)
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from fastapi import APIRouter, HTTPException, Query, Path
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.debug("FastAPI not installed; router unavailable")


# ===================================================================
# Request body models (Pydantic)
# ===================================================================

if FASTAPI_AVAILABLE:

    class CalculateRequest(BaseModel):
        """Request body for single flaring emission calculation."""

        flare_type: Optional[str] = Field(
            default=None,
            description="Flare type classification "
            "(ELEVATED_STEAM_ASSISTED, ENCLOSED_GROUND, etc.)",
        )
        flare_id: Optional[str] = Field(
            default=None,
            description="Registered flare system identifier",
        )
        gas_volume_mscf: Optional[float] = Field(
            default=None, ge=0,
            description="Gas volume in thousand standard cubic feet",
        )
        gas_volume_nm3: Optional[float] = Field(
            default=None, ge=0,
            description="Gas volume in normal cubic meters",
        )
        method: Optional[str] = Field(
            default="DEFAULT_EF",
            description="Calculation method "
            "(GAS_COMPOSITION, DEFAULT_EF, ENGINEERING_ESTIMATE, "
            "DIRECT_MEASUREMENT)",
        )
        event_category: Optional[str] = Field(
            default="ROUTINE",
            description="Event category "
            "(ROUTINE, NON_ROUTINE, EMERGENCY, MAINTENANCE, "
            "PILOT_PURGE, WELL_COMPLETION)",
        )
        gas_composition: Optional[Dict[str, float]] = Field(
            default=None,
            description="Gas composition as component mole fractions",
        )
        combustion_efficiency: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Combustion efficiency override (0.0-1.0)",
        )
        gwp_source: Optional[str] = Field(
            default="AR6",
            description="GWP source (AR4, AR5, AR6, AR6_20yr)",
        )
        ef_source: Optional[str] = Field(
            default="EPA_SUBPART_W",
            description="Emission factor source "
            "(EPA_SUBPART_W, IPCC_2006, API_2009)",
        )
        wind_speed_ms: Optional[float] = Field(
            default=None, ge=0,
            description="Wind speed at flare tip in m/s",
        )
        tip_velocity_mach: Optional[float] = Field(
            default=None, ge=0,
            description="Flare tip velocity as Mach number",
        )
        steam_to_gas_ratio: Optional[float] = Field(
            default=None, ge=0,
            description="Steam-to-gas mass ratio (lb/lb)",
        )
        air_to_gas_ratio: Optional[float] = Field(
            default=None, ge=0,
            description="Air-to-gas ratio",
        )
        pilot_gas_mmbtu_hr: Optional[float] = Field(
            default=None, ge=0,
            description="Pilot gas flow rate (MMBTU/hr per tip)",
        )
        num_pilot_tips: Optional[int] = Field(
            default=1, ge=0,
            description="Number of pilot tips",
        )
        purge_gas_type: Optional[str] = Field(
            default="N2",
            description="Purge gas type (N2 or NATURAL_GAS)",
        )
        purge_gas_flow_scfh: Optional[float] = Field(
            default=0, ge=0,
            description="Purge gas flow rate in SCF/hr",
        )
        operating_hours: Optional[float] = Field(
            default=1.0, gt=0,
            description="Operating duration in hours",
        )
        facility_id: Optional[str] = Field(
            default=None,
            description="Facility/installation identifier",
        )
        ogmp_level: Optional[int] = Field(
            default=None, ge=1, le=5,
            description="OGMP 2.0 reporting level (1-5)",
        )

    class BatchCalculateRequest(BaseModel):
        """Request body for batch flaring emission calculation."""

        calculations: List[Dict[str, Any]] = Field(
            ..., min_length=1,
            description="List of calculation request dictionaries",
        )

    class FlareSystemRegisterRequest(BaseModel):
        """Request body for flare system registration."""

        flare_id: Optional[str] = Field(
            default=None,
            description="Unique flare ID (auto-generated if omitted)",
        )
        flare_type: str = Field(
            default="ELEVATED_STEAM_ASSISTED",
            description="Flare type classification",
        )
        name: Optional[str] = Field(
            default=None, description="Human-readable name",
        )
        description: str = Field(
            default="", description="Flare system description",
        )
        default_ce: float = Field(
            default=0.98, ge=0.0, le=1.0,
            description="Default combustion efficiency",
        )
        assist_type: str = Field(
            default="NONE",
            description="Assist medium (STEAM, AIR, NONE)",
        )
        min_hhv_btu_scf: float = Field(
            default=200.0, ge=0,
            description="Minimum HHV threshold (BTU/scf)",
        )

    class FlaringEventRequest(BaseModel):
        """Request body for logging a flaring event."""

        flare_id: str = Field(
            ..., description="Associated flare system identifier",
        )
        event_category: str = Field(
            default="ROUTINE",
            description="Event category classification",
        )
        gas_volume_mscf: float = Field(
            default=0, ge=0,
            description="Gas volume in MSCF",
        )
        duration_hours: float = Field(
            default=0, ge=0,
            description="Event duration in hours",
        )

    class GasCompositionRegisterRequest(BaseModel):
        """Request body for gas composition registration."""

        composition_id: Optional[str] = Field(
            default=None,
            description="Unique composition ID (auto-generated if omitted)",
        )
        name: str = Field(
            default="", description="Composition name/label",
        )
        components: Dict[str, float] = Field(
            ..., description="Component mole fractions (must sum to ~1.0)",
        )
        source: str = Field(
            default="CUSTOM",
            description="Data source (LAB_ANALYSIS, CUSTOM, DEFAULT)",
        )

    class EmissionFactorRegisterRequest(BaseModel):
        """Request body for emission factor registration."""

        factor_id: Optional[str] = Field(
            default=None,
            description="Unique factor ID (auto-generated if omitted)",
        )
        source: str = Field(
            default="CUSTOM", description="Factor source authority",
        )
        co2_kg_per_mscf: float = Field(
            default=0, ge=0,
            description="CO2 emission factor (kg/MSCF)",
        )
        ch4_kg_per_mscf: float = Field(
            default=0, ge=0,
            description="CH4 emission factor (kg/MSCF)",
        )
        n2o_kg_per_mscf: float = Field(
            default=0, ge=0,
            description="N2O emission factor (kg/MSCF)",
        )

    class EfficiencyTestRequest(BaseModel):
        """Request body for combustion efficiency test logging."""

        flare_id: str = Field(
            ..., description="Associated flare system identifier",
        )
        measured_ce: float = Field(
            ..., ge=0.0, le=1.0,
            description="Measured combustion efficiency (0-1)",
        )
        wind_speed_ms: Optional[float] = Field(
            default=None, ge=0,
            description="Wind speed during test (m/s)",
        )
        tip_velocity_mach: Optional[float] = Field(
            default=None, ge=0,
            description="Tip velocity during test (Mach)",
        )
        test_date: Optional[str] = Field(
            default=None,
            description="Test date (ISO-8601)",
        )
        notes: str = Field(
            default="",
            description="Test notes/observations",
        )

    class UncertaintyRequest(BaseModel):
        """Request body for uncertainty analysis."""

        calculation_id: str = Field(
            ..., description="ID of a previous calculation",
        )
        method: str = Field(
            default="monte_carlo",
            description="Uncertainty method (monte_carlo or analytical)",
        )
        iterations: int = Field(
            default=5000, gt=0, le=1_000_000,
            description="Monte Carlo iterations",
        )

    class ComplianceCheckRequest(BaseModel):
        """Request body for compliance check."""

        calculation_id: str = Field(
            default="",
            description="ID of a previous calculation (optional)",
        )
        frameworks: List[str] = Field(
            default_factory=list,
            description="Frameworks to check (empty = all 8)",
        )


# ===================================================================
# Router factory
# ===================================================================


def create_router() -> "APIRouter":
    """Create and return the Flaring FastAPI APIRouter.

    Returns:
        Configured APIRouter with 20 endpoints.

    Raises:
        RuntimeError: If FastAPI is not installed.
    """
    if not FASTAPI_AVAILABLE:
        raise RuntimeError(
            "FastAPI is required for the flaring router"
        )

    router = APIRouter(
        prefix="/api/v1/flaring",
        tags=["Flaring"],
    )

    # ------------------------------------------------------------------
    # Lazy service accessor
    # ------------------------------------------------------------------

    def _get_service():
        """Get the FlaringService singleton.

        Raises HTTPException 503 if the service has not been initialized.
        """
        from greenlang.flaring.setup import get_service
        svc = get_service()
        if svc is None:
            raise HTTPException(
                status_code=503,
                detail="Flaring service not initialized",
            )
        return svc

    # ==================================================================
    # 1. POST /calculate - Calculate flaring emissions
    # ==================================================================

    @router.post("/calculate", status_code=200)
    async def calculate_emissions(
        body: CalculateRequest,
    ) -> Dict[str, Any]:
        """Calculate GHG emissions from a single flaring event.

        Applies the specified methodology (gas composition, default EF,
        engineering estimate, or direct measurement) to compute Scope 1
        flaring emissions including pilot and purge gas.
        """
        svc = _get_service()

        request_data: Dict[str, Any] = {}
        if body.flare_type is not None:
            request_data["flare_type"] = body.flare_type
        if body.flare_id is not None:
            request_data["flare_id"] = body.flare_id
        if body.gas_volume_mscf is not None:
            request_data["gas_volume_mscf"] = body.gas_volume_mscf
        if body.gas_volume_nm3 is not None:
            request_data["gas_volume_nm3"] = body.gas_volume_nm3
        if body.method is not None:
            request_data["method"] = body.method
        if body.event_category is not None:
            request_data["event_category"] = body.event_category
        if body.gas_composition is not None:
            request_data["gas_composition"] = body.gas_composition
        if body.combustion_efficiency is not None:
            request_data["combustion_efficiency"] = body.combustion_efficiency
        if body.gwp_source is not None:
            request_data["gwp_source"] = body.gwp_source
        if body.ef_source is not None:
            request_data["ef_source"] = body.ef_source
        if body.wind_speed_ms is not None:
            request_data["wind_speed_ms"] = body.wind_speed_ms
        if body.tip_velocity_mach is not None:
            request_data["tip_velocity_mach"] = body.tip_velocity_mach
        if body.steam_to_gas_ratio is not None:
            request_data["steam_to_gas_ratio"] = body.steam_to_gas_ratio
        if body.air_to_gas_ratio is not None:
            request_data["air_to_gas_ratio"] = body.air_to_gas_ratio
        if body.pilot_gas_mmbtu_hr is not None:
            request_data["pilot_gas_mmbtu_hr"] = body.pilot_gas_mmbtu_hr
        if body.num_pilot_tips is not None:
            request_data["num_pilot_tips"] = body.num_pilot_tips
        if body.purge_gas_type is not None:
            request_data["purge_gas_type"] = body.purge_gas_type
        if body.purge_gas_flow_scfh is not None:
            request_data["purge_gas_flow_scfh"] = body.purge_gas_flow_scfh
        if body.operating_hours is not None:
            request_data["operating_hours"] = body.operating_hours
        if body.facility_id is not None:
            request_data["facility_id"] = body.facility_id
        if body.ogmp_level is not None:
            request_data["ogmp_level"] = body.ogmp_level

        try:
            result = svc.calculate(request_data)
            return result.model_dump()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error("calculate endpoint failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 2. POST /calculate/batch - Batch calculate flaring emissions
    # ==================================================================

    @router.post("/calculate/batch", status_code=200)
    async def calculate_batch(
        body: BatchCalculateRequest,
    ) -> Dict[str, Any]:
        """Batch calculate GHG emissions for multiple flaring records.

        Each item in the ``calculations`` list follows the same schema
        as the single calculate endpoint.
        """
        svc = _get_service()
        try:
            result = svc.calculate_batch(body.calculations)
            return result.model_dump()
        except Exception as exc:
            logger.error(
                "batch calculate endpoint failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 3. GET /calculations - List calculations (paginated)
    # ==================================================================

    @router.get("/calculations", status_code=200)
    async def list_calculations(
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(
            20, ge=1, le=100, description="Items per page",
        ),
        flare_type: Optional[str] = Query(
            None, description="Filter by flare type",
        ),
        method: Optional[str] = Query(
            None, description="Filter by calculation method",
        ),
        event_category: Optional[str] = Query(
            None, description="Filter by event category",
        ),
    ) -> Dict[str, Any]:
        """List stored calculation results with pagination and filters."""
        svc = _get_service()
        all_calcs = list(svc._calculations)

        if flare_type is not None:
            all_calcs = [
                c for c in all_calcs
                if c.get("flare_type") == flare_type
            ]
        if method is not None:
            all_calcs = [
                c for c in all_calcs
                if c.get("method") == method.upper()
            ]
        if event_category is not None:
            all_calcs = [
                c for c in all_calcs
                if c.get("event_category") == event_category
            ]

        total = len(all_calcs)
        start = (page - 1) * page_size
        end = start + page_size
        page_data = all_calcs[start:end]

        return {
            "calculations": page_data,
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    # ==================================================================
    # 4. GET /calculations/{calc_id} - Get calculation details
    # ==================================================================

    @router.get("/calculations/{calc_id}", status_code=200)
    async def get_calculation(
        calc_id: str = Path(
            ..., description="Calculation identifier",
        ),
    ) -> Dict[str, Any]:
        """Get calculation details and per-gas breakdown."""
        svc = _get_service()

        for calc in svc._calculations:
            if calc.get("calculation_id") == calc_id:
                return calc

        raise HTTPException(
            status_code=404,
            detail=f"Calculation {calc_id} not found",
        )

    # ==================================================================
    # 5. POST /flares - Register flare system
    # ==================================================================

    @router.post("/flares", status_code=201)
    async def register_flare_system(
        body: FlareSystemRegisterRequest,
    ) -> Dict[str, Any]:
        """Register a new flare system."""
        svc = _get_service()
        try:
            result = svc.register_flare_system(body.model_dump())
            return result.model_dump()
        except Exception as exc:
            logger.error(
                "register_flare_system failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 6. GET /flares - List flare systems
    # ==================================================================

    @router.get("/flares", status_code=200)
    async def list_flare_systems() -> Dict[str, Any]:
        """List all registered flare systems."""
        svc = _get_service()
        result = svc.get_flare_systems()
        return result.model_dump()

    # ==================================================================
    # 7. GET /flares/{flare_id} - Get flare system details
    # ==================================================================

    @router.get("/flares/{flare_id}", status_code=200)
    async def get_flare_system(
        flare_id: str = Path(
            ..., description="Flare system identifier",
        ),
    ) -> Dict[str, Any]:
        """Get detailed information about a specific flare system."""
        svc = _get_service()
        result = svc.get_flare_system(flare_id)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Flare system {flare_id} not found",
            )
        return result.model_dump()

    # ==================================================================
    # 8. POST /events - Log flaring event
    # ==================================================================

    @router.post("/events", status_code=201)
    async def log_flaring_event(
        body: FlaringEventRequest,
    ) -> Dict[str, Any]:
        """Log a flaring event (routine, emergency, maintenance, etc.)."""
        svc = _get_service()
        try:
            result = svc.record_event(body.model_dump())
            return result.model_dump()
        except Exception as exc:
            logger.error(
                "log_flaring_event failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 9. GET /events - List flaring events
    # ==================================================================

    @router.get("/events", status_code=200)
    async def list_flaring_events(
        flare_id: Optional[str] = Query(
            None, description="Filter by flare system",
        ),
        event_category: Optional[str] = Query(
            None, description="Filter by event category",
        ),
    ) -> Dict[str, Any]:
        """List recorded flaring events with optional filters."""
        svc = _get_service()
        result = svc.get_events(
            flare_id=flare_id,
            event_category=event_category,
        )
        return result.model_dump()

    # ==================================================================
    # 10. GET /events/{event_id} - Get event details
    # ==================================================================

    @router.get("/events/{event_id}", status_code=200)
    async def get_flaring_event(
        event_id: str = Path(
            ..., description="Flaring event identifier",
        ),
    ) -> Dict[str, Any]:
        """Get detailed information about a specific flaring event."""
        svc = _get_service()
        result = svc.get_event(event_id)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Flaring event {event_id} not found",
            )
        return result.model_dump()

    # ==================================================================
    # 11. POST /compositions - Register gas composition
    # ==================================================================

    @router.post("/compositions", status_code=201)
    async def register_composition(
        body: GasCompositionRegisterRequest,
    ) -> Dict[str, Any]:
        """Register a gas composition analysis record."""
        svc = _get_service()
        try:
            result = svc.register_composition(body.model_dump())
            return result.model_dump()
        except Exception as exc:
            logger.error(
                "register_composition failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 12. GET /compositions - List gas compositions
    # ==================================================================

    @router.get("/compositions", status_code=200)
    async def list_compositions() -> Dict[str, Any]:
        """List all registered gas compositions."""
        svc = _get_service()
        result = svc.get_compositions()
        return result.model_dump()

    # ==================================================================
    # 13. POST /factors - Register custom emission factor
    # ==================================================================

    @router.post("/factors", status_code=201)
    async def register_factor(
        body: EmissionFactorRegisterRequest,
    ) -> Dict[str, Any]:
        """Register a custom flaring emission factor."""
        svc = _get_service()
        try:
            result = svc.register_factor(body.model_dump())
            return result.model_dump()
        except Exception as exc:
            logger.error(
                "register_factor failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 14. GET /factors - List emission factors
    # ==================================================================

    @router.get("/factors", status_code=200)
    async def list_factors() -> Dict[str, Any]:
        """List all registered flaring emission factors."""
        svc = _get_service()
        result = svc.get_factors()
        return result.model_dump()

    # ==================================================================
    # 15. POST /efficiency - Log combustion efficiency test
    # ==================================================================

    @router.post("/efficiency", status_code=201)
    async def log_efficiency_test(
        body: EfficiencyTestRequest,
    ) -> Dict[str, Any]:
        """Log a combustion efficiency test result."""
        svc = _get_service()
        try:
            result = svc.log_efficiency_test(body.model_dump())
            return result.model_dump()
        except Exception as exc:
            logger.error(
                "log_efficiency_test failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 16. GET /efficiency - List CE records
    # ==================================================================

    @router.get("/efficiency", status_code=200)
    async def list_efficiency_records() -> Dict[str, Any]:
        """List all combustion efficiency test records."""
        svc = _get_service()
        result = svc.get_efficiency_records()
        return result.model_dump()

    # ==================================================================
    # 17. POST /uncertainty - Run uncertainty analysis
    # ==================================================================

    @router.post("/uncertainty", status_code=200)
    async def run_uncertainty(
        body: UncertaintyRequest,
    ) -> Dict[str, Any]:
        """Run Monte Carlo or analytical uncertainty analysis.

        Requires a previous calculation_id. Returns statistical
        characterization of emission estimate uncertainty.
        """
        svc = _get_service()
        try:
            result = svc.run_uncertainty(body.model_dump())
            return result.model_dump()
        except Exception as exc:
            logger.error(
                "run_uncertainty failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 18. POST /compliance/check - Run compliance check
    # ==================================================================

    @router.post("/compliance/check", status_code=200)
    async def check_compliance(
        body: ComplianceCheckRequest,
    ) -> Dict[str, Any]:
        """Run regulatory compliance check against multiple frameworks.

        Evaluates the calculation against GHG Protocol, ISO 14064,
        CSRD/ESRS E1, EPA Subpart W, EU ETS MRR, EU Methane Regulation,
        World Bank ZRF, and OGMP 2.0.
        """
        svc = _get_service()
        try:
            result = svc.check_compliance(body.model_dump())
            return result.model_dump()
        except Exception as exc:
            logger.error(
                "check_compliance failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 19. GET /health - Health check
    # ==================================================================

    @router.get("/health", status_code=200)
    async def health_check() -> Dict[str, Any]:
        """Health check for the flaring service.

        Returns engine availability and overall service status.
        No authentication required.
        """
        svc = _get_service()
        result = svc.health_check()
        return result.model_dump()

    # ==================================================================
    # 20. GET /stats - Service statistics
    # ==================================================================

    @router.get("/stats", status_code=200)
    async def get_stats() -> Dict[str, Any]:
        """Get aggregate statistics for the flaring service.

        Includes total calculations, registered entities, and uptime.
        """
        svc = _get_service()
        result = svc.get_stats()
        return result.model_dump()

    return router


# ===================================================================
# Public API
# ===================================================================

__all__ = ["create_router"]
