# -*- coding: utf-8 -*-
"""
Land Use Emissions REST API Router - AGENT-MRV-006
====================================================

20 REST endpoints for the Land Use Emissions Agent (GL-MRV-SCOPE1-006).

Prefix: ``/api/v1/land-use-emissions``

Endpoints:
     1. POST   /calculations                - Execute single calculation
     2. POST   /calculations/batch          - Execute batch calculations
     3. GET    /calculations/{id}           - Get calculation by ID
     4. GET    /calculations                - List calculations with filters
     5. DELETE /calculations/{id}           - Delete calculation
     6. POST   /carbon-stocks               - Record carbon stock snapshot
     7. GET    /carbon-stocks/{parcel_id}   - Carbon stock history for parcel
     8. GET    /carbon-stocks/{parcel_id}/summary - Summary across all pools
     9. POST   /land-parcels                - Register a land parcel
    10. GET    /land-parcels                - List parcels with filters
    11. PUT    /land-parcels/{id}           - Update a land parcel
    12. POST   /transitions                 - Record a transition event
    13. GET    /transitions                 - List transitions with filters
    14. GET    /transitions/matrix          - 6x6 transition matrix summary
    15. POST   /soc-assessments             - Run SOC assessment
    16. GET    /soc-assessments/{parcel_id} - SOC assessment history
    17. POST   /compliance/check            - Run compliance check
    18. GET    /compliance/{id}             - Get compliance result
    19. POST   /uncertainty                 - Run Monte Carlo uncertainty
    20. GET    /aggregations                - Get aggregated emissions

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-006 Land Use Emissions (GL-MRV-SCOPE1-006)
Status: Production Ready
"""

from __future__ import annotations

import logging
from decimal import Decimal
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

    class SingleCalculationRequest(BaseModel):
        """Request body for a single land use emission calculation."""

        parcel_id: str = Field(
            ...,
            description="Reference to the land parcel",
        )
        from_category: str = Field(
            ...,
            description="IPCC land category before transition "
            "(forest_land, cropland, grassland, wetland, "
            "settlement, other_land)",
        )
        to_category: str = Field(
            ...,
            description="IPCC land category after transition",
        )
        area_ha: float = Field(
            ..., gt=0,
            description="Area in hectares",
        )
        climate_zone: str = Field(
            ...,
            description="IPCC climate zone "
            "(tropical_wet, tropical_moist, tropical_dry, "
            "tropical_montane, warm_temperate_moist, "
            "warm_temperate_dry, cool_temperate_moist, "
            "cool_temperate_dry, boreal_moist, boreal_dry, "
            "polar_moist, polar_dry)",
        )
        soil_type: str = Field(
            ...,
            description="IPCC soil type "
            "(high_activity_clay, low_activity_clay, sandy, "
            "spodic, volcanic, wetland_organic, other)",
        )
        tier: Optional[str] = Field(
            default=None,
            description="IPCC calculation tier (tier_1, tier_2, tier_3)",
        )
        method: Optional[str] = Field(
            default=None,
            description="Calculation method "
            "(stock_difference or gain_loss)",
        )
        gwp_source: Optional[str] = Field(
            default=None,
            description="GWP source (AR4, AR5, AR6, AR6_GTP)",
        )
        pools: Optional[List[str]] = Field(
            default=None,
            description="Carbon pools to include "
            "(above_ground_biomass, below_ground_biomass, "
            "dead_wood, litter, soil_organic_carbon)",
        )
        management_practice: Optional[str] = Field(
            default=None,
            description="Soil management practice "
            "(full_tillage, reduced_tillage, no_till, "
            "improved, degraded, nominally_managed)",
        )
        input_level: Optional[str] = Field(
            default=None,
            description="Carbon input level "
            "(low, medium, high, high_with_manure)",
        )
        include_fire: bool = Field(
            default=False,
            description="Whether to include fire emissions",
        )
        include_n2o: bool = Field(
            default=False,
            description="Whether to include soil N2O emissions",
        )
        include_peatland: bool = Field(
            default=False,
            description="Whether to include peatland emissions",
        )
        disturbance_type: Optional[str] = Field(
            default=None,
            description="Disturbance type "
            "(fire, harvest, storm, insects, drought, "
            "flood, land_clearing, none)",
        )
        peatland_status: Optional[str] = Field(
            default=None,
            description="Peatland status "
            "(natural, drained, rewetted, extracted)",
        )
        transition_years: Optional[int] = Field(
            default=None, gt=0, le=100,
            description="Transition period in years",
        )
        reference_year: Optional[int] = Field(
            default=None, ge=1900, le=2100,
            description="Reference year for the calculation",
        )
        tenant_id: Optional[str] = Field(
            default=None,
            description="Owning tenant identifier",
        )

    class BatchCalculationBody(BaseModel):
        """Request body for batch land use emission calculations."""

        calculations: List[Dict[str, Any]] = Field(
            ..., min_length=1,
            description="List of calculation request dictionaries",
        )
        gwp_source: Optional[str] = Field(
            default=None,
            description="GWP source applied to all calculations (AR4, AR5, AR6)",
        )
        tenant_id: Optional[str] = Field(
            default=None,
            description="Owning tenant identifier",
        )

    class CarbonStockBody(BaseModel):
        """Request body for recording a carbon stock snapshot."""

        parcel_id: str = Field(
            ..., description="Reference to the land parcel",
        )
        pool: str = Field(
            ...,
            description="Carbon pool "
            "(above_ground_biomass, below_ground_biomass, "
            "dead_wood, litter, soil_organic_carbon)",
        )
        stock_tc_ha: float = Field(
            ..., ge=0,
            description="Carbon stock in tonnes C per hectare",
        )
        measurement_date: str = Field(
            ...,
            description="Date of the measurement (ISO-8601)",
        )
        tier: Optional[str] = Field(
            default=None,
            description="IPCC calculation tier (tier_1, tier_2, tier_3)",
        )
        source: Optional[str] = Field(
            default=None,
            description="Source authority for the stock value",
        )
        uncertainty_pct: Optional[float] = Field(
            default=None, ge=0, le=100,
            description="Measurement uncertainty as a percentage",
        )
        notes: Optional[str] = Field(
            default=None, max_length=2000,
            description="Optional notes about the measurement",
        )

    class LandParcelBody(BaseModel):
        """Request body for registering a land parcel."""

        name: str = Field(
            ..., min_length=1, max_length=500,
            description="Human-readable parcel name",
        )
        area_ha: float = Field(
            ..., gt=0,
            description="Parcel area in hectares",
        )
        land_category: str = Field(
            ...,
            description="Current IPCC land-use category",
        )
        climate_zone: str = Field(
            ...,
            description="IPCC climate zone",
        )
        soil_type: str = Field(
            ...,
            description="IPCC soil type",
        )
        latitude: float = Field(
            ..., ge=-90, le=90,
            description="WGS84 latitude in decimal degrees",
        )
        longitude: float = Field(
            ..., ge=-180, le=180,
            description="WGS84 longitude in decimal degrees",
        )
        tenant_id: str = Field(
            ..., min_length=1,
            description="Owning tenant identifier",
        )
        country_code: Optional[str] = Field(
            default=None, max_length=2,
            description="ISO 3166-1 alpha-2 country code",
        )
        management_practice: Optional[str] = Field(
            default=None,
            description="Current soil management practice",
        )
        input_level: Optional[str] = Field(
            default=None,
            description="Carbon input level",
        )
        peatland_status: Optional[str] = Field(
            default=None,
            description="Peatland management status if applicable",
        )

    class LandParcelUpdateBody(BaseModel):
        """Request body for updating a land parcel."""

        name: Optional[str] = Field(
            default=None, min_length=1, max_length=500,
            description="Human-readable parcel name",
        )
        area_ha: Optional[float] = Field(
            default=None, gt=0,
            description="Parcel area in hectares",
        )
        land_category: Optional[str] = Field(
            default=None,
            description="Current IPCC land-use category",
        )
        management_practice: Optional[str] = Field(
            default=None,
            description="Current soil management practice",
        )
        input_level: Optional[str] = Field(
            default=None,
            description="Carbon input level",
        )
        peatland_status: Optional[str] = Field(
            default=None,
            description="Peatland management status",
        )
        country_code: Optional[str] = Field(
            default=None, max_length=2,
            description="ISO 3166-1 alpha-2 country code",
        )

    class TransitionBody(BaseModel):
        """Request body for recording a land-use transition."""

        parcel_id: str = Field(
            ..., min_length=1,
            description="Reference to the land parcel",
        )
        from_category: str = Field(
            ...,
            description="IPCC land category before transition",
        )
        to_category: str = Field(
            ...,
            description="IPCC land category after transition",
        )
        transition_date: str = Field(
            ...,
            description="Date of the transition event (ISO-8601)",
        )
        area_ha: float = Field(
            ..., gt=0,
            description="Area affected in hectares",
        )
        transition_type: str = Field(
            ...,
            description="Transition type (remaining or conversion)",
        )
        disturbance_type: Optional[str] = Field(
            default=None,
            description="Type of disturbance causing transition",
        )
        notes: Optional[str] = Field(
            default=None, max_length=2000,
            description="Optional notes about the transition",
        )

    class SOCAssessmentBody(BaseModel):
        """Request body for a soil organic carbon assessment."""

        parcel_id: str = Field(
            ..., min_length=1,
            description="Reference to the land parcel",
        )
        climate_zone: str = Field(
            ...,
            description="IPCC climate zone",
        )
        soil_type: str = Field(
            ...,
            description="IPCC soil type",
        )
        land_category: str = Field(
            ...,
            description="Current land-use category",
        )
        management_practice: Optional[str] = Field(
            default=None,
            description="Soil management practice",
        )
        input_level: Optional[str] = Field(
            default=None,
            description="Carbon input level",
        )
        depth_cm: Optional[int] = Field(
            default=None, gt=0, le=300,
            description="Soil assessment depth in centimeters",
        )
        transition_years: Optional[int] = Field(
            default=None, gt=0, le=100,
            description="Transition period for annualisation",
        )
        previous_land_category: Optional[str] = Field(
            default=None,
            description="Previous land category for change calc",
        )
        previous_management: Optional[str] = Field(
            default=None,
            description="Previous management practice",
        )
        previous_input_level: Optional[str] = Field(
            default=None,
            description="Previous input level",
        )

    class ComplianceCheckBody(BaseModel):
        """Request body for a compliance check."""

        calculation_id: str = Field(
            default="",
            description="ID of a previous calculation (optional)",
        )
        frameworks: List[str] = Field(
            default_factory=list,
            description="Frameworks to check "
            "(empty = all frameworks). "
            "Options: GHG_PROTOCOL, IPCC, CSRD, "
            "EU_LULUCF, UK_SECR, UNFCCC",
        )
        tenant_id: Optional[str] = Field(
            default=None,
            description="Tenant identifier for scoping",
        )

    class UncertaintyBody(BaseModel):
        """Request body for uncertainty analysis."""

        calculation_id: str = Field(
            ..., description="ID of a previous calculation",
        )
        iterations: int = Field(
            default=5000, gt=0, le=1_000_000,
            description="Monte Carlo iterations",
        )
        seed: int = Field(
            default=42, ge=0,
            description="Random seed for reproducibility",
        )
        confidence_level: float = Field(
            default=95.0, gt=0, lt=100,
            description="Confidence level percentage",
        )

    class AggregationQuery(BaseModel):
        """Request body for aggregation queries."""

        tenant_id: str = Field(
            ..., min_length=1,
            description="Tenant identifier for scoping",
        )
        period: Optional[str] = Field(
            default=None,
            description="Reporting period "
            "(monthly, quarterly, annual, custom)",
        )
        group_by: Optional[List[str]] = Field(
            default=None,
            description="Fields to group results by",
        )
        date_from: Optional[str] = Field(
            default=None,
            description="Start date (ISO-8601)",
        )
        date_to: Optional[str] = Field(
            default=None,
            description="End date (ISO-8601)",
        )
        land_categories: Optional[List[str]] = Field(
            default=None,
            description="Filter by land categories",
        )
        climate_zones: Optional[List[str]] = Field(
            default=None,
            description="Filter by climate zones",
        )


# ===================================================================
# Router factory
# ===================================================================


def create_router() -> "APIRouter":
    """Create and return the Land Use Emissions FastAPI APIRouter.

    Returns:
        Configured APIRouter with 20 endpoints.

    Raises:
        RuntimeError: If FastAPI is not installed.
    """
    if not FASTAPI_AVAILABLE:
        raise RuntimeError(
            "FastAPI is required for the land use emissions router"
        )

    router = APIRouter(
        prefix="/api/v1/land-use-emissions",
        tags=["Land Use Emissions"],
    )

    # ------------------------------------------------------------------
    # Lazy service accessor
    # ------------------------------------------------------------------

    def _get_service():
        """Get the LandUseEmissionsService singleton.

        Raises HTTPException 503 if the service has not been initialized.
        """
        from greenlang.land_use_emissions.setup import get_service
        svc = get_service()
        if svc is None:
            raise HTTPException(
                status_code=503,
                detail="Land Use Emissions service not initialized",
            )
        return svc

    # ==================================================================
    # 1. POST /calculations - Execute single calculation
    # ==================================================================

    @router.post("/calculations", status_code=201)
    async def create_calculation(
        body: SingleCalculationRequest,
    ) -> Dict[str, Any]:
        """Execute a single land use emission calculation.

        Computes carbon stock changes and resulting GHG emissions for a
        land parcel or transition event using the specified IPCC tier,
        method, and carbon pools. Supports stock-difference and
        gain-loss methods at Tier 1/2/3.
        """
        svc = _get_service()

        request_data: Dict[str, Any] = {
            "parcel_id": body.parcel_id,
            "from_category": body.from_category,
            "to_category": body.to_category,
            "area_ha": body.area_ha,
            "climate_zone": body.climate_zone,
            "soil_type": body.soil_type,
            "include_fire": body.include_fire,
            "include_n2o": body.include_n2o,
            "include_peatland": body.include_peatland,
        }
        if body.tier is not None:
            request_data["tier"] = body.tier
        if body.method is not None:
            request_data["method"] = body.method
        if body.gwp_source is not None:
            request_data["gwp_source"] = body.gwp_source
        if body.pools is not None:
            request_data["pools"] = body.pools
        if body.management_practice is not None:
            request_data["management_practice"] = body.management_practice
        if body.input_level is not None:
            request_data["input_level"] = body.input_level
        if body.disturbance_type is not None:
            request_data["disturbance_type"] = body.disturbance_type
        if body.peatland_status is not None:
            request_data["peatland_status"] = body.peatland_status
        if body.transition_years is not None:
            request_data["transition_years"] = body.transition_years
        if body.reference_year is not None:
            request_data["reference_year"] = body.reference_year
        if body.tenant_id is not None:
            request_data["tenant_id"] = body.tenant_id

        try:
            result = svc.calculate(request_data)
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "create_calculation failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 2. POST /calculations/batch - Execute batch calculations
    # ==================================================================

    @router.post("/calculations/batch", status_code=201)
    async def create_batch_calculation(
        body: BatchCalculationBody,
    ) -> Dict[str, Any]:
        """Execute batch land use emission calculations.

        Processes multiple calculation requests in a single batch. Each
        item in the ``calculations`` list follows the same schema as
        the single calculate endpoint.
        """
        svc = _get_service()
        try:
            result = svc.calculate_batch(
                body.calculations,
                gwp_source=body.gwp_source,
                tenant_id=body.tenant_id,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "create_batch_calculation failed: %s", exc,
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 3. GET /calculations/{id} - Get calculation by ID
    # ==================================================================

    @router.get("/calculations/{calc_id}", status_code=200)
    async def get_calculation(
        calc_id: str = Path(
            ..., description="Calculation identifier",
        ),
    ) -> Dict[str, Any]:
        """Get a calculation result by its unique identifier."""
        svc = _get_service()

        for calc in svc._calculations:
            if calc.get("calculation_id") == calc_id:
                return calc

        raise HTTPException(
            status_code=404,
            detail=f"Calculation {calc_id} not found",
        )

    # ==================================================================
    # 4. GET /calculations - List calculations with filters
    # ==================================================================

    @router.get("/calculations", status_code=200)
    async def list_calculations(
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(
            20, ge=1, le=100, description="Items per page",
        ),
        land_category: Optional[str] = Query(
            None, description="Filter by land category",
        ),
        method: Optional[str] = Query(
            None, description="Filter by calculation method",
        ),
        tenant_id: Optional[str] = Query(
            None, description="Filter by tenant identifier",
        ),
    ) -> Dict[str, Any]:
        """List calculation results with pagination and filters."""
        svc = _get_service()
        all_calcs = list(svc._calculations)

        if land_category is not None:
            all_calcs = [
                c for c in all_calcs
                if c.get("to_category") == land_category
                or c.get("from_category") == land_category
            ]
        if method is not None:
            all_calcs = [
                c for c in all_calcs
                if c.get("method") == method
            ]
        if tenant_id is not None:
            all_calcs = [
                c for c in all_calcs
                if c.get("tenant_id") == tenant_id
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
    # 5. DELETE /calculations/{id} - Delete calculation
    # ==================================================================

    @router.delete("/calculations/{calc_id}", status_code=200)
    async def delete_calculation(
        calc_id: str = Path(
            ..., description="Calculation identifier",
        ),
    ) -> Dict[str, Any]:
        """Delete a calculation result by its unique identifier."""
        svc = _get_service()

        for i, calc in enumerate(svc._calculations):
            if calc.get("calculation_id") == calc_id:
                svc._calculations.pop(i)
                svc._total_calculations = max(
                    0, svc._total_calculations - 1,
                )
                return {
                    "deleted": True,
                    "calculation_id": calc_id,
                }

        raise HTTPException(
            status_code=404,
            detail=f"Calculation {calc_id} not found",
        )

    # ==================================================================
    # 6. POST /carbon-stocks - Record carbon stock snapshot
    # ==================================================================

    @router.post("/carbon-stocks", status_code=201)
    async def create_carbon_stock(
        body: CarbonStockBody,
    ) -> Dict[str, Any]:
        """Record a point-in-time carbon stock measurement.

        Creates a carbon stock snapshot for a specific parcel and pool.
        Snapshots are used in the stock-difference method to compute
        carbon stock changes over time.
        """
        svc = _get_service()
        try:
            result = svc.record_carbon_stock(body.model_dump())
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "create_carbon_stock failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 7. GET /carbon-stocks/{parcel_id} - History for parcel
    # ==================================================================

    @router.get("/carbon-stocks/{parcel_id}", status_code=200)
    async def get_carbon_stocks(
        parcel_id: str = Path(
            ..., description="Land parcel identifier",
        ),
        pool: Optional[str] = Query(
            None, description="Filter by carbon pool",
        ),
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(
            20, ge=1, le=100, description="Items per page",
        ),
    ) -> Dict[str, Any]:
        """Get carbon stock measurement history for a parcel.

        Returns all carbon stock snapshots ordered by measurement date,
        optionally filtered by pool.
        """
        svc = _get_service()
        try:
            result = svc.get_carbon_stocks(
                parcel_id=parcel_id,
                pool=pool,
                page=page,
                page_size=page_size,
            )
            return result
        except Exception as exc:
            logger.error(
                "get_carbon_stocks failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 8. GET /carbon-stocks/{parcel_id}/summary - Summary all pools
    # ==================================================================

    @router.get(
        "/carbon-stocks/{parcel_id}/summary", status_code=200,
    )
    async def get_carbon_stocks_summary(
        parcel_id: str = Path(
            ..., description="Land parcel identifier",
        ),
    ) -> Dict[str, Any]:
        """Get a summary of latest carbon stocks across all pools.

        Returns the most recent snapshot for each carbon pool for the
        specified parcel, along with total carbon stock.
        """
        svc = _get_service()
        try:
            stocks = svc.get_carbon_stocks(
                parcel_id=parcel_id, page=1, page_size=10000,
            )
            snapshots = stocks.get("snapshots", [])

            # Get latest per pool
            latest_by_pool: Dict[str, Any] = {}
            for snap in snapshots:
                pool_name = snap.get("pool", "")
                existing = latest_by_pool.get(pool_name)
                if existing is None:
                    latest_by_pool[pool_name] = snap
                else:
                    if snap.get("measurement_date", "") > existing.get(
                        "measurement_date", ""
                    ):
                        latest_by_pool[pool_name] = snap

            total_stock = sum(
                float(s.get("stock_tc_ha", 0))
                for s in latest_by_pool.values()
            )

            return {
                "parcel_id": parcel_id,
                "pools": latest_by_pool,
                "total_stock_tc_ha": total_stock,
                "pool_count": len(latest_by_pool),
            }
        except Exception as exc:
            logger.error(
                "get_carbon_stocks_summary failed: %s", exc,
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 9. POST /land-parcels - Register a land parcel
    # ==================================================================

    @router.post("/land-parcels", status_code=201)
    async def create_land_parcel(
        body: LandParcelBody,
    ) -> Dict[str, Any]:
        """Register a new land parcel for LULUCF tracking.

        Creates a parcel record with geographic, climatic, and soil
        characteristics. Every parcel is scoped to a tenant.
        """
        svc = _get_service()
        try:
            result = svc.register_parcel(body.model_dump())
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "create_land_parcel failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 10. GET /land-parcels - List parcels with filters
    # ==================================================================

    @router.get("/land-parcels", status_code=200)
    async def list_land_parcels(
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(
            20, ge=1, le=100, description="Items per page",
        ),
        tenant_id: Optional[str] = Query(
            None, description="Filter by tenant identifier",
        ),
        land_category: Optional[str] = Query(
            None, description="Filter by land category",
        ),
        climate_zone: Optional[str] = Query(
            None, description="Filter by climate zone",
        ),
    ) -> Dict[str, Any]:
        """List registered land parcels with pagination and filters."""
        svc = _get_service()
        try:
            result = svc.list_parcels(
                page=page,
                page_size=page_size,
                tenant_id=tenant_id,
                land_category=land_category,
                climate_zone=climate_zone,
            )
            return result
        except Exception as exc:
            logger.error(
                "list_land_parcels failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 11. PUT /land-parcels/{id} - Update a land parcel
    # ==================================================================

    @router.put("/land-parcels/{parcel_id}", status_code=200)
    async def update_land_parcel(
        body: LandParcelUpdateBody,
        parcel_id: str = Path(
            ..., description="Land parcel identifier",
        ),
    ) -> Dict[str, Any]:
        """Update an existing land parcel's attributes.

        Only non-null fields in the request body will be updated.
        """
        svc = _get_service()
        update_data = {
            k: v for k, v in body.model_dump().items()
            if v is not None
        }
        try:
            result = svc.update_parcel(parcel_id, update_data)
            if result is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Parcel {parcel_id} not found",
                )
            return result
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "update_land_parcel failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 12. POST /transitions - Record a transition event
    # ==================================================================

    @router.post("/transitions", status_code=201)
    async def create_transition(
        body: TransitionBody,
    ) -> Dict[str, Any]:
        """Record a land-use transition event.

        Tracks a parcel's transition from one IPCC land category to
        another. Transitions can be of type ``remaining`` (same category)
        or ``conversion`` (different category).
        """
        svc = _get_service()
        try:
            result = svc.record_transition(body.model_dump())
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "create_transition failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 13. GET /transitions - List transitions with filters
    # ==================================================================

    @router.get("/transitions", status_code=200)
    async def list_transitions(
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(
            20, ge=1, le=100, description="Items per page",
        ),
        parcel_id: Optional[str] = Query(
            None, description="Filter by parcel identifier",
        ),
        from_category: Optional[str] = Query(
            None, description="Filter by source land category",
        ),
        to_category: Optional[str] = Query(
            None, description="Filter by target land category",
        ),
        transition_type: Optional[str] = Query(
            None, description="Filter by transition type",
        ),
    ) -> Dict[str, Any]:
        """List land-use transition records with pagination and filters."""
        svc = _get_service()
        try:
            result = svc.get_transitions(
                page=page,
                page_size=page_size,
                parcel_id=parcel_id,
                from_category=from_category,
                to_category=to_category,
                transition_type=transition_type,
            )
            return result
        except Exception as exc:
            logger.error(
                "list_transitions failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 14. GET /transitions/matrix - 6x6 transition matrix summary
    # ==================================================================

    @router.get("/transitions/matrix", status_code=200)
    async def get_transition_matrix(
        tenant_id: Optional[str] = Query(
            None, description="Filter by tenant identifier",
        ),
    ) -> Dict[str, Any]:
        """Get 6x6 IPCC land category transition matrix.

        Returns a summary matrix showing total area transitioned
        between each pair of IPCC land categories. The matrix has
        rows as source categories and columns as target categories.
        """
        svc = _get_service()
        try:
            result = svc.get_transition_matrix(tenant_id=tenant_id)
            return result
        except Exception as exc:
            logger.error(
                "get_transition_matrix failed: %s", exc,
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 15. POST /soc-assessments - Run SOC assessment
    # ==================================================================

    @router.post("/soc-assessments", status_code=201)
    async def create_soc_assessment(
        body: SOCAssessmentBody,
    ) -> Dict[str, Any]:
        """Run a soil organic carbon assessment.

        Uses the IPCC Tier 1 approach to estimate SOC stocks:
        ``SOC = SOC_ref * F_LU * F_MG * F_I``
        where SOC_ref is the reference stock, F_LU is the land use
        factor, F_MG is the management factor, and F_I is the
        input factor.
        """
        svc = _get_service()
        try:
            result = svc.assess_soc(body.model_dump())
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "create_soc_assessment failed: %s", exc,
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 16. GET /soc-assessments/{parcel_id} - SOC assessment history
    # ==================================================================

    @router.get("/soc-assessments/{parcel_id}", status_code=200)
    async def get_soc_assessments(
        parcel_id: str = Path(
            ..., description="Land parcel identifier",
        ),
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(
            20, ge=1, le=100, description="Items per page",
        ),
    ) -> Dict[str, Any]:
        """Get SOC assessment history for a land parcel.

        Returns all SOC assessment results ordered by assessment date.
        """
        svc = _get_service()
        try:
            assessments = [
                a for a in svc._soc_assessments
                if a.get("parcel_id") == parcel_id
            ]
            total = len(assessments)
            start = (page - 1) * page_size
            end = start + page_size
            page_data = assessments[start:end]

            return {
                "parcel_id": parcel_id,
                "assessments": page_data,
                "total": total,
                "page": page,
                "page_size": page_size,
            }
        except Exception as exc:
            logger.error(
                "get_soc_assessments failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 17. POST /compliance/check - Run compliance check
    # ==================================================================

    @router.post("/compliance/check", status_code=200)
    async def check_compliance(
        body: ComplianceCheckBody,
    ) -> Dict[str, Any]:
        """Run regulatory compliance check against multiple frameworks.

        Evaluates the calculation against applicable LULUCF frameworks:
        GHG Protocol, IPCC 2006 Guidelines, CSRD/ESRS E1, EU LULUCF
        Regulation, UK SECR, and UNFCCC reporting requirements.
        """
        svc = _get_service()
        try:
            result = svc.check_compliance(body.model_dump())
            return result
        except Exception as exc:
            logger.error(
                "check_compliance failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 18. GET /compliance/{id} - Get compliance result
    # ==================================================================

    @router.get("/compliance/{compliance_id}", status_code=200)
    async def get_compliance_result(
        compliance_id: str = Path(
            ..., description="Compliance check identifier",
        ),
    ) -> Dict[str, Any]:
        """Get a compliance check result by its unique identifier."""
        svc = _get_service()

        for result in svc._compliance_results:
            if result.get("id") == compliance_id:
                return result

        raise HTTPException(
            status_code=404,
            detail=f"Compliance result {compliance_id} not found",
        )

    # ==================================================================
    # 19. POST /uncertainty - Run Monte Carlo uncertainty
    # ==================================================================

    @router.post("/uncertainty", status_code=200)
    async def run_uncertainty(
        body: UncertaintyBody,
    ) -> Dict[str, Any]:
        """Run Monte Carlo uncertainty analysis on a calculation.

        Requires a previous ``calculation_id``. Returns statistical
        characterization of emission estimate uncertainty including
        mean, standard deviation, confidence intervals, and
        coefficient of variation.
        """
        svc = _get_service()
        try:
            result = svc.run_uncertainty(body.model_dump())
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "run_uncertainty failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 20. GET /aggregations - Get aggregated emissions
    # ==================================================================

    @router.get("/aggregations", status_code=200)
    async def get_aggregations(
        tenant_id: str = Query(
            ..., min_length=1,
            description="Tenant identifier for scoping",
        ),
        period: Optional[str] = Query(
            None,
            description="Reporting period "
            "(monthly, quarterly, annual, custom)",
        ),
        group_by: Optional[str] = Query(
            None,
            description="Comma-separated fields to group by "
            "(land_category, climate_zone, parcel_id)",
        ),
        date_from: Optional[str] = Query(
            None, description="Start date (ISO-8601)",
        ),
        date_to: Optional[str] = Query(
            None, description="End date (ISO-8601)",
        ),
        land_categories: Optional[str] = Query(
            None,
            description="Comma-separated land category filter",
        ),
    ) -> Dict[str, Any]:
        """Get aggregated land use emissions.

        Aggregates calculation results by specified grouping fields
        and reporting period. Supports filtering by tenant, land
        categories, date range, and climate zones.
        """
        svc = _get_service()
        try:
            agg_data: Dict[str, Any] = {
                "tenant_id": tenant_id,
            }
            if period is not None:
                agg_data["period"] = period
            if group_by is not None:
                agg_data["group_by"] = [
                    g.strip() for g in group_by.split(",")
                    if g.strip()
                ]
            if date_from is not None:
                agg_data["date_from"] = date_from
            if date_to is not None:
                agg_data["date_to"] = date_to
            if land_categories is not None:
                agg_data["land_categories"] = [
                    c.strip() for c in land_categories.split(",")
                    if c.strip()
                ]

            result = svc.aggregate(agg_data)
            return result
        except Exception as exc:
            logger.error(
                "get_aggregations failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    return router


# ===================================================================
# Public API
# ===================================================================

__all__ = ["create_router"]
