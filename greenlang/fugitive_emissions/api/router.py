# -*- coding: utf-8 -*-
"""
Fugitive Emissions REST API Router - AGENT-MRV-005
====================================================

20 REST endpoints for the Fugitive Emissions Agent (GL-MRV-SCOPE1-005).

Prefix: ``/api/v1/fugitive-emissions``

Endpoints:
     1. POST   /calculate              - Calculate fugitive emissions
     2. POST   /calculate/batch        - Batch calculate fugitive emissions
     3. GET    /calculations           - List calculations (paginated)
     4. GET    /calculations/{calc_id} - Get calculation details
     5. POST   /sources                - Register source type
     6. GET    /sources                - List source types
     7. GET    /sources/{source_id}    - Get source details
     8. POST   /components             - Register equipment component
     9. GET    /components             - List equipment components
    10. GET    /components/{component_id} - Get component details
    11. POST   /surveys                - Register LDAR survey
    12. GET    /surveys                - List LDAR surveys
    13. POST   /factors                - Register emission factor
    14. GET    /factors                - List emission factors
    15. POST   /repairs                - Register component repair
    16. GET    /repairs                - List component repairs
    17. POST   /uncertainty            - Run uncertainty analysis
    18. POST   /compliance/check       - Run compliance check
    19. GET    /health                 - Health check
    20. GET    /stats                  - Service statistics

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-005 Fugitive Emissions (GL-MRV-SCOPE1-005)
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
        """Request body for single fugitive emission calculation."""

        source_type: str = Field(
            default="EQUIPMENT_LEAK",
            description="Fugitive emission source type "
            "(EQUIPMENT_LEAK, TANK_STORAGE, WASTEWATER_TREATMENT, "
            "PNEUMATIC_DEVICE, COMPRESSOR_SEAL, DEHYDRATOR)",
        )
        facility_id: str = Field(
            default="", description="Facility identifier",
        )
        component_count: Optional[int] = Field(
            default=None, ge=0,
            description="Number of equipment components",
        )
        calculation_method: Optional[str] = Field(
            default=None,
            description="Calculation method override "
            "(AVERAGE_EMISSION_FACTOR, SCREENING_RANGE, "
            "UNIT_CORRELATION, DIRECT_MEASUREMENT, MASS_BALANCE)",
        )
        gwp_source: Optional[str] = Field(
            default=None,
            description="GWP source (AR4, AR5, AR6, AR6_20YR)",
        )
        gas_composition: Optional[Dict[str, float]] = Field(
            default=None,
            description="Gas composition by species {gas: fraction}",
        )
        service_type: Optional[str] = Field(
            default=None,
            description="Service type (GAS, LIGHT_LIQUID, HEAVY_LIQUID, "
            "HYDROGEN)",
        )
        tank_type: Optional[str] = Field(
            default=None,
            description="Tank type for storage loss calculations",
        )
        tank_parameters: Optional[Dict[str, Any]] = Field(
            default=None,
            description="Tank physical parameters for AP-42 calculations",
        )
        abatement_efficiency: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Emission abatement/recovery efficiency (0.0-1.0)",
        )
        operating_hours: Optional[float] = Field(
            default=None, gt=0,
            description="Annual operating hours",
        )

    class BatchCalculateRequest(BaseModel):
        """Request body for batch fugitive emission calculation."""

        calculations: List[Dict[str, Any]] = Field(
            ..., min_length=1,
            description="List of calculation request dictionaries",
        )

    class SourceRegisterRequest(BaseModel):
        """Request body for source type registration."""

        source_type: str = Field(
            ..., description="Source type identifier",
        )
        name: Optional[str] = Field(
            default=None, description="Human-readable name",
        )
        gases: List[str] = Field(
            default_factory=lambda: ["CH4"],
            description="Associated greenhouse gases",
        )
        methods: List[str] = Field(
            default_factory=lambda: ["AVERAGE_EMISSION_FACTOR"],
            description="Applicable calculation methods",
        )

    class ComponentRegisterRequest(BaseModel):
        """Request body for equipment component registration."""

        tag_number: str = Field(
            ..., description="Equipment tag number",
        )
        component_type: str = Field(
            default="other",
            description="Component type "
            "(valve, connector, pump_seal, compressor_seal, "
            "flange, open_ended_line, pressure_relief_valve, "
            "sampling_connection, instrument, other)",
        )
        service_type: str = Field(
            default="gas",
            description="Service type (gas, light_liquid, "
            "heavy_liquid, hydrogen)",
        )
        facility_id: str = Field(
            default="", description="Facility identifier",
        )
        unit_id: Optional[str] = Field(
            default=None, description="Process unit identifier",
        )
        location: Optional[str] = Field(
            default=None, description="Physical location description",
        )

    class SurveyRegisterRequest(BaseModel):
        """Request body for LDAR survey registration."""

        survey_type: str = Field(
            default="OGI",
            description="Survey method (OGI, METHOD21, "
            "ACOUSTIC, SATELLITE, DRONE)",
        )
        facility_id: str = Field(
            default="", description="Facility identifier",
        )
        survey_date: Optional[str] = Field(
            default=None,
            description="Survey date (ISO-8601)",
        )
        components_surveyed: Optional[int] = Field(
            default=None, ge=0,
            description="Number of components surveyed",
        )
        leaks_found: Optional[int] = Field(
            default=None, ge=0,
            description="Number of leaks detected",
        )
        threshold_ppm: Optional[float] = Field(
            default=None, ge=0,
            description="Leak detection threshold in ppm",
        )

    class FactorRegisterRequest(BaseModel):
        """Request body for emission factor registration."""

        source_type: str = Field(
            ..., description="Source type this factor applies to",
        )
        component_type: str = Field(
            default="",
            description="Component type (optional)",
        )
        gas: str = Field(
            default="CH4", description="Greenhouse gas species",
        )
        value: float = Field(
            ..., gt=0, description="Emission factor value (kg/hr/component)",
        )
        source: str = Field(
            default="CUSTOM",
            description="Factor source authority "
            "(EPA, IPCC, API, EU, CUSTOM)",
        )

    class RepairRegisterRequest(BaseModel):
        """Request body for component repair registration."""

        component_id: str = Field(
            ..., description="Component being repaired",
        )
        repair_type: str = Field(
            default="minor",
            description="Repair type (minor, major, replacement)",
        )
        repair_date: Optional[str] = Field(
            default=None, description="Repair date (ISO-8601)",
        )
        pre_repair_rate_ppm: Optional[float] = Field(
            default=None, ge=0,
            description="Pre-repair screening value (ppm)",
        )
        post_repair_rate_ppm: Optional[float] = Field(
            default=None, ge=0,
            description="Post-repair screening value (ppm)",
        )
        cost_usd: Optional[float] = Field(
            default=None, ge=0,
            description="Repair cost in USD",
        )
        notes: Optional[str] = Field(
            default=None, description="Repair notes",
        )

    class UncertaintyRequest(BaseModel):
        """Request body for uncertainty analysis."""

        calculation_id: str = Field(
            ..., description="ID of a previous calculation",
        )
        method: str = Field(
            default="monte_carlo",
            description="Uncertainty method "
            "(monte_carlo or analytical)",
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
            description="Frameworks to check "
            "(empty = all 7 frameworks)",
        )


# ===================================================================
# Router factory
# ===================================================================


def create_router() -> "APIRouter":
    """Create and return the Fugitive Emissions FastAPI APIRouter.

    Returns:
        Configured APIRouter with 20 endpoints.

    Raises:
        RuntimeError: If FastAPI is not installed.
    """
    if not FASTAPI_AVAILABLE:
        raise RuntimeError(
            "FastAPI is required for the fugitive emissions router"
        )

    router = APIRouter(
        prefix="/api/v1/fugitive-emissions",
        tags=["Fugitive Emissions"],
    )

    # ------------------------------------------------------------------
    # Lazy service accessor
    # ------------------------------------------------------------------

    def _get_service():
        """Get the FugitiveEmissionsService singleton.

        Raises HTTPException 503 if the service has not been initialized.
        """
        from greenlang.fugitive_emissions.setup import get_service
        svc = get_service()
        if svc is None:
            raise HTTPException(
                status_code=503,
                detail="Fugitive Emissions service not initialized",
            )
        return svc

    # ==================================================================
    # 1. POST /calculate - Calculate fugitive emissions
    # ==================================================================

    @router.post("/calculate", status_code=200)
    async def calculate_emissions(
        body: CalculateRequest,
    ) -> Dict[str, Any]:
        """Calculate GHG emissions for a single fugitive emission source.

        Applies the specified methodology (average emission factor,
        screening range, unit correlation, direct measurement, or
        mass balance) to compute Scope 1 fugitive emissions from
        equipment leaks, tank storage, pneumatic devices, etc.
        """
        svc = _get_service()

        request_data: Dict[str, Any] = {
            "source_type": body.source_type,
            "facility_id": body.facility_id,
        }
        if body.component_count is not None:
            request_data["component_count"] = body.component_count
        if body.calculation_method is not None:
            request_data["calculation_method"] = body.calculation_method
        if body.gwp_source is not None:
            request_data["gwp_source"] = body.gwp_source
        if body.gas_composition is not None:
            request_data["gas_composition"] = body.gas_composition
        if body.service_type is not None:
            request_data["service_type"] = body.service_type
        if body.tank_type is not None:
            request_data["tank_type"] = body.tank_type
        if body.tank_parameters is not None:
            request_data["tank_parameters"] = body.tank_parameters
        if body.abatement_efficiency is not None:
            request_data["abatement_efficiency"] = (
                body.abatement_efficiency
            )
        if body.operating_hours is not None:
            request_data["operating_hours"] = body.operating_hours

        try:
            result = svc.calculate(request_data)
            return result.model_dump()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "calculate endpoint failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 2. POST /calculate/batch - Batch calculate fugitive emissions
    # ==================================================================

    @router.post("/calculate/batch", status_code=200)
    async def calculate_batch(
        body: BatchCalculateRequest,
    ) -> Dict[str, Any]:
        """Batch calculate GHG emissions for multiple fugitive sources.

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
        source_type: Optional[str] = Query(
            None, description="Filter by source type",
        ),
        method: Optional[str] = Query(
            None, description="Filter by calculation method",
        ),
    ) -> Dict[str, Any]:
        """List stored calculation results with pagination and filters."""
        svc = _get_service()
        all_calcs = list(svc._calculations)

        # Apply filters
        if source_type is not None:
            all_calcs = [
                c for c in all_calcs
                if c.get("source_type") == source_type
            ]
        if method is not None:
            all_calcs = [
                c for c in all_calcs
                if c.get("method") == method.upper()
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
    # 5. POST /sources - Register source type
    # ==================================================================

    @router.post("/sources", status_code=201)
    async def register_source(
        body: SourceRegisterRequest,
    ) -> Dict[str, Any]:
        """Register a new fugitive emission source type."""
        svc = _get_service()
        try:
            result = svc.register_source(body.model_dump())
            return result.model_dump()
        except Exception as exc:
            logger.error(
                "register_source failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 6. GET /sources - List source types
    # ==================================================================

    @router.get("/sources", status_code=200)
    async def list_sources(
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(
            20, ge=1, le=100, description="Items per page",
        ),
    ) -> Dict[str, Any]:
        """List registered fugitive emission source types."""
        svc = _get_service()
        result = svc.list_sources(page=page, page_size=page_size)
        return result.model_dump()

    # ==================================================================
    # 7. GET /sources/{source_id} - Get source details
    # ==================================================================

    @router.get("/sources/{source_id}", status_code=200)
    async def get_source(
        source_id: str = Path(
            ..., description="Source type identifier",
        ),
    ) -> Dict[str, Any]:
        """Get detailed information about a specific source type."""
        svc = _get_service()
        result = svc.get_source(source_id)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Source type {source_id} not found",
            )
        return result.model_dump()

    # ==================================================================
    # 8. POST /components - Register equipment component
    # ==================================================================

    @router.post("/components", status_code=201)
    async def register_component(
        body: ComponentRegisterRequest,
    ) -> Dict[str, Any]:
        """Register an equipment component for LDAR tracking."""
        svc = _get_service()
        try:
            result = svc.register_component(body.model_dump())
            return result.model_dump()
        except Exception as exc:
            logger.error(
                "register_component failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 9. GET /components - List equipment components
    # ==================================================================

    @router.get("/components", status_code=200)
    async def list_components(
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(
            20, ge=1, le=100, description="Items per page",
        ),
    ) -> Dict[str, Any]:
        """List registered equipment components with pagination."""
        svc = _get_service()
        result = svc.list_components(page=page, page_size=page_size)
        return result.model_dump()

    # ==================================================================
    # 10. GET /components/{component_id} - Get component details
    # ==================================================================

    @router.get("/components/{component_id}", status_code=200)
    async def get_component(
        component_id: str = Path(
            ..., description="Component identifier",
        ),
    ) -> Dict[str, Any]:
        """Get detailed information about a specific component."""
        svc = _get_service()
        result = svc.get_component(component_id)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Component {component_id} not found",
            )
        return result.model_dump()

    # ==================================================================
    # 11. POST /surveys - Register LDAR survey
    # ==================================================================

    @router.post("/surveys", status_code=201)
    async def register_survey(
        body: SurveyRegisterRequest,
    ) -> Dict[str, Any]:
        """Register an LDAR survey event.

        Records a leak detection and repair survey with method type,
        date, components surveyed, and leaks found.
        """
        svc = _get_service()
        try:
            result = svc.register_survey(body.model_dump())
            return result
        except Exception as exc:
            logger.error(
                "register_survey failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 12. GET /surveys - List LDAR surveys
    # ==================================================================

    @router.get("/surveys", status_code=200)
    async def list_surveys() -> Dict[str, Any]:
        """List all registered LDAR surveys."""
        svc = _get_service()
        result = svc.list_surveys()
        return result.model_dump()

    # ==================================================================
    # 13. POST /factors - Register emission factor
    # ==================================================================

    @router.post("/factors", status_code=201)
    async def register_factor(
        body: FactorRegisterRequest,
    ) -> Dict[str, Any]:
        """Register a custom fugitive emission factor.

        Allows registration of site-specific or custom emission
        factors (kg/hr/component) for different source types,
        component types, and gases.
        """
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
        """List all registered emission factors."""
        svc = _get_service()
        result = svc.list_factors()
        return result.model_dump()

    # ==================================================================
    # 15. POST /repairs - Register component repair
    # ==================================================================

    @router.post("/repairs", status_code=201)
    async def register_repair(
        body: RepairRegisterRequest,
    ) -> Dict[str, Any]:
        """Register a component repair event.

        Records pre/post repair screening values, repair type,
        cost, and notes for LDAR compliance tracking.
        """
        svc = _get_service()
        try:
            result = svc.register_repair(body.model_dump())
            return result
        except Exception as exc:
            logger.error(
                "register_repair failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 16. GET /repairs - List component repairs
    # ==================================================================

    @router.get("/repairs", status_code=200)
    async def list_repairs() -> Dict[str, Any]:
        """List all registered component repairs."""
        svc = _get_service()
        result = svc.list_repairs()
        return result.model_dump()

    # ==================================================================
    # 17. POST /uncertainty - Run uncertainty analysis
    # ==================================================================

    @router.post("/uncertainty", status_code=200)
    async def run_uncertainty(
        body: UncertaintyRequest,
    ) -> Dict[str, Any]:
        """Run Monte Carlo or analytical uncertainty analysis.

        Requires a previous ``calculation_id``. Returns statistical
        characterization of emission estimate uncertainty including
        confidence intervals, DQI score, and sensitivity analysis.
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

        Evaluates the calculation against up to 7 frameworks:
        GHG Protocol, ISO 14064-1, CSRD/ESRS E1, EPA Subpart W,
        EPA LDAR, EU Methane Regulation, and UK SECR.
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
        """Health check for the fugitive emissions service.

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
        """Get aggregate statistics for the fugitive emissions service.

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
