# -*- coding: utf-8 -*-
"""
Process Emissions REST API Router - AGENT-MRV-004
====================================================

20 REST endpoints for the Process Emissions Agent (GL-MRV-SCOPE1-004).

Prefix: ``/api/v1/process-emissions``

Endpoints:
     1. POST   /calculate              - Calculate process emissions
     2. POST   /calculate/batch        - Batch calculate process emissions
     3. GET    /calculations           - List calculations (paginated)
     4. GET    /calculations/{calc_id} - Get calculation details
     5. POST   /processes              - Register process type
     6. GET    /processes              - List process types
     7. GET    /processes/{process_id} - Get process details
     8. POST   /materials              - Register raw material
     9. GET    /materials              - List raw materials
    10. GET    /materials/{material_id}- Get material details
    11. POST   /units                  - Register process unit
    12. GET    /units                  - List process units
    13. POST   /factors                - Register emission factor
    14. GET    /factors                - List emission factors
    15. POST   /abatement              - Register abatement technology
    16. GET    /abatement              - List abatement records
    17. POST   /uncertainty            - Run uncertainty analysis
    18. POST   /compliance/check       - Run compliance check
    19. GET    /health                 - Health check
    20. GET    /stats                  - Service statistics

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-004 Process Emissions (GL-MRV-SCOPE1-004)
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
        """Request body for single process emission calculation."""

        process_type: str = Field(
            ..., description="Industrial process type identifier",
        )
        activity_data: float = Field(
            ..., gt=0, description="Production quantity",
        )
        activity_unit: str = Field(
            default="tonne", description="Unit of activity data",
        )
        calculation_method: Optional[str] = Field(
            default=None,
            description="Calculation method override "
            "(EMISSION_FACTOR, MASS_BALANCE, STOICHIOMETRIC, "
            "DIRECT_MEASUREMENT)",
        )
        calculation_tier: Optional[str] = Field(
            default=None,
            description="Calculation tier (TIER_1, TIER_2, TIER_3)",
        )
        gwp_source: Optional[str] = Field(
            default=None,
            description="GWP source (AR4, AR5, AR6, AR6_20YR)",
        )
        ef_source: Optional[str] = Field(
            default=None,
            description="Emission factor source "
            "(EPA, IPCC, DEFRA, EU_ETS, CUSTOM)",
        )
        production_route: Optional[str] = Field(
            default=None,
            description="Production route (for iron/steel, aluminum)",
        )
        abatement_type: Optional[str] = Field(
            default=None,
            description="Abatement technology type",
        )
        abatement_efficiency: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Abatement efficiency (0.0-1.0)",
        )
        materials: Optional[List[Dict[str, Any]]] = Field(
            default=None,
            description="Material inputs for mass balance method",
        )

    class BatchCalculateRequest(BaseModel):
        """Request body for batch process emission calculation."""

        calculations: List[Dict[str, Any]] = Field(
            ..., min_length=1,
            description="List of calculation request dictionaries",
        )

    class ProcessRegisterRequest(BaseModel):
        """Request body for process type registration."""

        process_type: str = Field(
            ..., description="Process type identifier",
        )
        category: str = Field(
            default="other", description="Process category",
        )
        name: Optional[str] = Field(
            default=None, description="Human-readable name",
        )
        description: str = Field(
            default="", description="Process description",
        )
        primary_gases: List[str] = Field(
            default_factory=lambda: ["CO2"],
            description="Primary greenhouse gases",
        )
        applicable_tiers: List[str] = Field(
            default_factory=lambda: ["TIER_1", "TIER_2", "TIER_3"],
            description="Applicable calculation tiers",
        )
        default_emission_factor: Optional[float] = Field(
            default=None, description="Default emission factor value",
        )
        production_routes: List[str] = Field(
            default_factory=list,
            description="Available production routes",
        )

    class MaterialRegisterRequest(BaseModel):
        """Request body for raw material registration."""

        material_type: str = Field(
            ..., description="Material type identifier",
        )
        name: Optional[str] = Field(
            default=None, description="Human-readable name",
        )
        carbon_content: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Carbon content fraction",
        )
        carbonate_content: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Carbonate content fraction",
        )

    class UnitRegisterRequest(BaseModel):
        """Request body for process unit registration."""

        unit_id: Optional[str] = Field(
            default=None,
            description="Unique unit ID (auto-generated if omitted)",
        )
        unit_name: str = Field(
            ..., description="Human-readable unit name",
        )
        unit_type: str = Field(
            default="other", description="Equipment classification",
        )
        process_type: str = Field(
            ..., description="Industrial process type",
        )

    class FactorRegisterRequest(BaseModel):
        """Request body for emission factor registration."""

        factor_id: Optional[str] = Field(
            default=None,
            description="Unique factor ID (auto-generated if omitted)",
        )
        process_type: str = Field(
            ..., description="Process type this factor applies to",
        )
        gas: str = Field(
            default="CO2", description="Greenhouse gas species",
        )
        value: float = Field(
            ..., gt=0, description="Emission factor value",
        )
        source: str = Field(
            default="CUSTOM", description="Factor source authority",
        )

    class AbatementRegisterRequest(BaseModel):
        """Request body for abatement technology registration."""

        abatement_id: Optional[str] = Field(
            default=None,
            description="Unique abatement ID (auto-generated if omitted)",
        )
        unit_id: str = Field(
            ..., description="Process unit where abatement is applied",
        )
        abatement_type: str = Field(
            ..., description="Abatement technology type",
        )
        efficiency: float = Field(
            ..., ge=0.0, le=1.0,
            description="Abatement efficiency (0.0-1.0)",
        )
        target_gas: str = Field(
            default="CO2", description="Target greenhouse gas",
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
            description="Frameworks to check (empty = all)",
        )


# ===================================================================
# Router factory
# ===================================================================


def create_router() -> "APIRouter":
    """Create and return the Process Emissions FastAPI APIRouter.

    Returns:
        Configured APIRouter with 20 endpoints.

    Raises:
        RuntimeError: If FastAPI is not installed.
    """
    if not FASTAPI_AVAILABLE:
        raise RuntimeError(
            "FastAPI is required for the process emissions router"
        )

    router = APIRouter(
        prefix="/api/v1/process-emissions",
        tags=["Process Emissions"],
    )

    # ------------------------------------------------------------------
    # Lazy service accessor
    # ------------------------------------------------------------------

    def _get_service():
        """Get the ProcessEmissionsService singleton.

        Raises HTTPException 503 if the service has not been initialized.
        """
        from greenlang.process_emissions.setup import get_service
        svc = get_service()
        if svc is None:
            raise HTTPException(
                status_code=503,
                detail="Process Emissions service not initialized",
            )
        return svc

    # ==================================================================
    # 1. POST /calculate - Calculate process emissions
    # ==================================================================

    @router.post("/calculate", status_code=200)
    async def calculate_emissions(
        body: CalculateRequest,
    ) -> Dict[str, Any]:
        """Calculate GHG emissions for a single industrial process record.

        Applies the specified methodology (emission factor, mass balance,
        stoichiometric, or direct measurement) to compute Scope 1
        non-combustion process emissions.
        """
        svc = _get_service()

        request_data: Dict[str, Any] = {
            "process_type": body.process_type,
            "activity_data": body.activity_data,
            "activity_unit": body.activity_unit,
        }
        if body.calculation_method is not None:
            request_data["calculation_method"] = body.calculation_method
        if body.calculation_tier is not None:
            request_data["calculation_tier"] = body.calculation_tier
        if body.gwp_source is not None:
            request_data["gwp_source"] = body.gwp_source
        if body.ef_source is not None:
            request_data["ef_source"] = body.ef_source
        if body.production_route is not None:
            request_data["production_route"] = body.production_route
        if body.abatement_type is not None:
            request_data["abatement_type"] = body.abatement_type
        if body.abatement_efficiency is not None:
            request_data["abatement_efficiency"] = (
                body.abatement_efficiency
            )
        if body.materials is not None:
            request_data["materials"] = body.materials

        try:
            result = svc.calculate(request_data)
            return result.model_dump()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error("calculate endpoint failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 2. POST /calculate/batch - Batch calculate process emissions
    # ==================================================================

    @router.post("/calculate/batch", status_code=200)
    async def calculate_batch(
        body: BatchCalculateRequest,
    ) -> Dict[str, Any]:
        """Batch calculate GHG emissions for multiple process records.

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
        process_type: Optional[str] = Query(
            None, description="Filter by process type",
        ),
        method: Optional[str] = Query(
            None, description="Filter by calculation method",
        ),
    ) -> Dict[str, Any]:
        """List stored calculation results with pagination and filters."""
        svc = _get_service()
        all_calcs = list(svc._calculations)

        # Apply filters
        if process_type is not None:
            all_calcs = [
                c for c in all_calcs
                if c.get("process_type") == process_type
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
    # 5. POST /processes - Register process type
    # ==================================================================

    @router.post("/processes", status_code=201)
    async def register_process(
        body: ProcessRegisterRequest,
    ) -> Dict[str, Any]:
        """Register a new industrial process type."""
        svc = _get_service()
        try:
            result = svc.register_process(body.model_dump())
            return result.model_dump()
        except Exception as exc:
            logger.error(
                "register_process failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 6. GET /processes - List process types
    # ==================================================================

    @router.get("/processes", status_code=200)
    async def list_processes(
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(
            20, ge=1, le=100, description="Items per page",
        ),
    ) -> Dict[str, Any]:
        """List registered process types with pagination."""
        svc = _get_service()
        result = svc.list_processes(page=page, page_size=page_size)
        return result.model_dump()

    # ==================================================================
    # 7. GET /processes/{process_id} - Get process details
    # ==================================================================

    @router.get("/processes/{process_id}", status_code=200)
    async def get_process(
        process_id: str = Path(
            ..., description="Process type identifier",
        ),
    ) -> Dict[str, Any]:
        """Get detailed information about a specific process type."""
        svc = _get_service()
        result = svc.get_process(process_id)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Process type {process_id} not found",
            )
        return result.model_dump()

    # ==================================================================
    # 8. POST /materials - Register raw material
    # ==================================================================

    @router.post("/materials", status_code=201)
    async def register_material(
        body: MaterialRegisterRequest,
    ) -> Dict[str, Any]:
        """Register a raw material with its physical properties."""
        svc = _get_service()
        try:
            result = svc.register_material(body.model_dump())
            return result.model_dump()
        except Exception as exc:
            logger.error(
                "register_material failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 9. GET /materials - List raw materials
    # ==================================================================

    @router.get("/materials", status_code=200)
    async def list_materials() -> Dict[str, Any]:
        """List all registered raw materials."""
        svc = _get_service()
        result = svc.list_materials()
        return result.model_dump()

    # ==================================================================
    # 10. GET /materials/{material_id} - Get material details
    # ==================================================================

    @router.get("/materials/{material_id}", status_code=200)
    async def get_material(
        material_id: str = Path(
            ..., description="Material type identifier",
        ),
    ) -> Dict[str, Any]:
        """Get detailed information about a specific raw material."""
        svc = _get_service()
        result = svc.get_material(material_id)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Material {material_id} not found",
            )
        return result.model_dump()

    # ==================================================================
    # 11. POST /units - Register process unit
    # ==================================================================

    @router.post("/units", status_code=201)
    async def register_unit(
        body: UnitRegisterRequest,
    ) -> Dict[str, Any]:
        """Register an industrial process unit (equipment)."""
        svc = _get_service()
        try:
            result = svc.register_unit(body.model_dump())
            return result.model_dump()
        except Exception as exc:
            logger.error(
                "register_unit failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 12. GET /units - List process units
    # ==================================================================

    @router.get("/units", status_code=200)
    async def list_units() -> Dict[str, Any]:
        """List all registered process units."""
        svc = _get_service()
        result = svc.list_units()
        return result.model_dump()

    # ==================================================================
    # 13. POST /factors - Register emission factor
    # ==================================================================

    @router.post("/factors", status_code=201)
    async def register_factor(
        body: FactorRegisterRequest,
    ) -> Dict[str, Any]:
        """Register a custom emission factor."""
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
    # 15. POST /abatement - Register abatement technology
    # ==================================================================

    @router.post("/abatement", status_code=201)
    async def register_abatement(
        body: AbatementRegisterRequest,
    ) -> Dict[str, Any]:
        """Register an abatement technology record."""
        svc = _get_service()
        try:
            result = svc.register_abatement(body.model_dump())
            return result
        except Exception as exc:
            logger.error(
                "register_abatement failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 16. GET /abatement - List abatement records
    # ==================================================================

    @router.get("/abatement", status_code=200)
    async def list_abatement() -> Dict[str, Any]:
        """List all registered abatement records."""
        svc = _get_service()
        result = svc.list_abatement()
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
        CSRD/ESRS E1, EPA 40 CFR Part 98, UK SECR, and EU ETS.
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
        """Health check for the process emissions service.

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
        """Get aggregate statistics for the process emissions service.

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
