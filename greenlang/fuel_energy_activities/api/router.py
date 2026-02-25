# -*- coding: utf-8 -*-
"""
Fuel & Energy Activities REST API Router - AGENT-MRV-016
========================================================

20 REST endpoints for the Fuel & Energy Activities Agent (GL-MRV-S3-003 / Scope 3 Category 3).

Prefix: ``/api/v1/fuel-energy-activities``

Endpoints:
     1. POST   /calculate                        - Calculate emissions (Activity 3a/3b/3c or combined)
     2. POST   /calculate/batch                  - Batch calculation
     3. GET    /calculations                     - List calculations with pagination
     4. GET    /calculations/{calc_id}           - Get specific calculation
     5. DELETE /calculations/{calc_id}           - Delete calculation
     6. POST   /fuel-consumption                 - Record fuel consumption
     7. GET    /fuel-consumption                 - List fuel consumption records
     8. PUT    /fuel-consumption/{record_id}     - Update fuel consumption record
     9. POST   /electricity-consumption          - Record electricity consumption
    10. GET    /electricity-consumption          - List electricity consumption records
    11. GET    /emission-factors                 - List WTT emission factors
    12. GET    /emission-factors/{factor_id}     - Get specific emission factor
    13. POST   /emission-factors/custom          - Register custom WTT factor
    14. GET    /td-loss-factors                  - List T&D loss factors
    15. GET    /td-loss-factors/{country_code}   - Get T&D loss factor by country
    16. POST   /compliance/check                 - Run compliance check
    17. GET    /compliance/{check_id}            - Get compliance check result
    18. POST   /uncertainty                      - Run uncertainty analysis
    19. GET    /aggregations                     - Get aggregated results
    20. GET    /health                           - Health check endpoint

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-016 Fuel & Energy Activities (GL-MRV-SCOPE3-003)
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
        """Request body for fuel & energy activities emission calculation."""

        activity_type: str = Field(
            ..., min_length=1,
            description="Activity type (UPSTREAM_FUEL_WTT / Activity 3a, "
            "UPSTREAM_ELECTRICITY_WTT / Activity 3b, "
            "UPSTREAM_ELECTRICITY_TD_LOSS / Activity 3c, ALL)",
        )
        fuel_consumption_records: Optional[List[Dict[str, Any]]] = Field(
            default=None,
            description="Fuel consumption records for Activity 3a (WTT for purchased fuels). "
            "Each record includes fuel_type, quantity, unit, region, etc.",
        )
        electricity_consumption_records: Optional[List[Dict[str, Any]]] = Field(
            default=None,
            description="Electricity consumption records for Activity 3b (WTT for electricity) "
            "and 3c (T&D losses). Each record includes quantity, unit, country, grid_type, etc.",
        )
        reporting_period_start: str = Field(
            ...,
            description="Reporting period start date (ISO-8601)",
        )
        reporting_period_end: str = Field(
            ...,
            description="Reporting period end date (ISO-8601)",
        )
        calculation_method: str = Field(
            default="AVERAGE_DATA",
            description="Calculation method (AVERAGE_DATA, SUPPLIER_SPECIFIC, HYBRID)",
        )
        gwp_source: Optional[str] = Field(
            default=None,
            description="GWP source (AR4, AR5, AR6, AR6_GTP)",
        )
        include_uncertainty: bool = Field(
            default=True,
            description="Include uncertainty analysis",
        )
        tenant_id: Optional[str] = Field(
            default=None,
            description="Owning tenant identifier",
        )

    class BatchCalculateRequest(BaseModel):
        """Request body for batch calculations."""

        requests: List[Dict[str, Any]] = Field(
            ..., min_length=1,
            description="List of calculation requests to process",
        )
        parallel: bool = Field(
            default=True,
            description="Execute calculations in parallel",
        )
        tenant_id: Optional[str] = Field(
            default=None,
            description="Owning tenant identifier",
        )

    class FuelConsumptionRequest(BaseModel):
        """Request body for fuel consumption record."""

        record_id: str = Field(
            ..., min_length=1, max_length=200,
            description="Unique fuel consumption record identifier",
        )
        fuel_type: str = Field(
            ..., min_length=1, max_length=100,
            description="Fuel type (e.g., DIESEL, GASOLINE, NATURAL_GAS, COAL, etc.)",
        )
        quantity: float = Field(
            ..., gt=0,
            description="Fuel quantity consumed",
        )
        unit: str = Field(
            ..., min_length=1, max_length=50,
            description="Unit of measurement (e.g., liters, kg, m3, GJ, kWh)",
        )
        consumption_date: str = Field(
            ...,
            description="Date of fuel consumption (ISO-8601)",
        )
        facility_id: Optional[str] = Field(
            default=None, max_length=200,
            description="Facility identifier where fuel was consumed",
        )
        region: Optional[str] = Field(
            default=None, max_length=100,
            description="Geographic region (ISO 3166 code or regional identifier)",
        )
        supplier_id: Optional[str] = Field(
            default=None, max_length=200,
            description="Fuel supplier identifier",
        )
        supplier_emission_factor: Optional[float] = Field(
            default=None, ge=0,
            description="Supplier-specific WTT emission factor (kgCO2e/unit) if available",
        )
        purpose: Optional[str] = Field(
            default=None, max_length=200,
            description="Purpose of fuel consumption (HEATING, PROCESS, TRANSPORT, etc.)",
        )
        metadata: Optional[Dict[str, Any]] = Field(
            default=None,
            description="Additional consumption metadata",
        )
        tenant_id: str = Field(
            ..., min_length=1,
            description="Owning tenant identifier",
        )

    class FuelConsumptionUpdateRequest(BaseModel):
        """Request body for fuel consumption record update."""

        quantity: Optional[float] = Field(
            default=None, gt=0,
            description="Updated fuel quantity",
        )
        unit: Optional[str] = Field(
            default=None, max_length=50,
            description="Updated unit",
        )
        consumption_date: Optional[str] = Field(
            default=None,
            description="Updated consumption date",
        )
        facility_id: Optional[str] = Field(
            default=None, max_length=200,
            description="Updated facility identifier",
        )
        region: Optional[str] = Field(
            default=None, max_length=100,
            description="Updated region",
        )
        supplier_emission_factor: Optional[float] = Field(
            default=None, ge=0,
            description="Updated supplier-specific WTT emission factor",
        )
        metadata: Optional[Dict[str, Any]] = Field(
            default=None,
            description="Additional metadata updates",
        )

    class ElectricityConsumptionRequest(BaseModel):
        """Request body for electricity consumption record."""

        record_id: str = Field(
            ..., min_length=1, max_length=200,
            description="Unique electricity consumption record identifier",
        )
        quantity: float = Field(
            ..., gt=0,
            description="Electricity quantity consumed (kWh or MWh)",
        )
        unit: str = Field(
            default="kWh", max_length=50,
            description="Unit of measurement (kWh, MWh, GJ)",
        )
        consumption_date: str = Field(
            ...,
            description="Date of electricity consumption (ISO-8601)",
        )
        facility_id: Optional[str] = Field(
            default=None, max_length=200,
            description="Facility identifier where electricity was consumed",
        )
        country: str = Field(
            ..., min_length=2, max_length=3,
            description="Country code (ISO 3166-1 alpha-2 or alpha-3)",
        )
        grid_type: str = Field(
            default="NATIONAL_GRID",
            description="Grid type (NATIONAL_GRID, REGIONAL_GRID, RENEWABLE_ONLY)",
        )
        supplier_id: Optional[str] = Field(
            default=None, max_length=200,
            description="Electricity supplier identifier",
        )
        supplier_wtt_factor: Optional[float] = Field(
            default=None, ge=0,
            description="Supplier-specific WTT emission factor (kgCO2e/kWh) if available",
        )
        supplier_td_loss_factor: Optional[float] = Field(
            default=None, ge=0, le=1,
            description="Supplier-specific T&D loss factor (0-1) if available",
        )
        renewable_percentage: Optional[float] = Field(
            default=None, ge=0, le=100,
            description="Percentage of renewable energy in supply (0-100)",
        )
        metadata: Optional[Dict[str, Any]] = Field(
            default=None,
            description="Additional consumption metadata",
        )
        tenant_id: str = Field(
            ..., min_length=1,
            description="Owning tenant identifier",
        )

    class CustomFactorRequest(BaseModel):
        """Request body for custom WTT emission factor registration."""

        factor_id: str = Field(
            ..., min_length=1, max_length=200,
            description="Unique emission factor identifier",
        )
        factor_type: str = Field(
            ..., min_length=1,
            description="Factor type (FUEL_WTT, ELECTRICITY_WTT, TD_LOSS)",
        )
        fuel_type: Optional[str] = Field(
            default=None, max_length=100,
            description="Fuel type (required for FUEL_WTT factors)",
        )
        country: Optional[str] = Field(
            default=None, max_length=3,
            description="Country code (required for ELECTRICITY_WTT and TD_LOSS factors)",
        )
        emission_factor: float = Field(
            ..., ge=0,
            description="Emission factor value (kgCO2e/unit for WTT, 0-1 for TD loss)",
        )
        ef_unit: str = Field(
            ..., min_length=1, max_length=100,
            description="Emission factor unit (e.g., kgCO2e/liter, kgCO2e/kWh, ratio)",
        )
        data_source: str = Field(
            ..., min_length=1, max_length=500,
            description="Source of emission factor data",
        )
        geographic_scope: Optional[str] = Field(
            default=None, max_length=100,
            description="Geographic applicability (ISO 3166 code)",
        )
        temporal_scope: Optional[int] = Field(
            default=None, ge=1990, le=2030,
            description="Reference year",
        )
        uncertainty: Optional[float] = Field(
            default=None, ge=0, le=100,
            description="Uncertainty percentage",
        )
        metadata: Optional[Dict[str, Any]] = Field(
            default=None,
            description="Additional EF metadata",
        )
        tenant_id: str = Field(
            ..., min_length=1,
            description="Owning tenant identifier",
        )

    class ComplianceCheckRequest(BaseModel):
        """Request body for compliance check."""

        result_id: str = Field(
            ..., min_length=1,
            description="Calculation result identifier",
        )
        frameworks: List[str] = Field(
            default_factory=list,
            description="Frameworks to check (empty = all frameworks). "
            "Options: GHG_PROTOCOL, CSRD, CDP, SBTi, ISO14064, TCFD, GLEC",
        )
        tenant_id: Optional[str] = Field(
            default=None,
            description="Tenant identifier for scoping",
        )

    class UncertaintyRequest(BaseModel):
        """Request body for uncertainty analysis."""

        result_id: str = Field(
            ..., min_length=1,
            description="Calculation result identifier",
        )
        method: str = Field(
            default="MONTE_CARLO",
            description="Uncertainty method (MONTE_CARLO, PEDIGREE_MATRIX, "
            "IPCC_TIER2_PROPAGATION, BOOTSTRAP)",
        )
        iterations: int = Field(
            default=10000, ge=1000, le=100000,
            description="Number of iterations for simulation-based methods",
        )
        confidence_level: float = Field(
            default=0.95, ge=0.5, le=0.99,
            description="Confidence level for uncertainty bounds",
        )

    class AggregationRequest(BaseModel):
        """Request body for aggregation query."""

        filters: Optional[Dict[str, Any]] = Field(
            default=None,
            description="Filters to apply (tenant_id, date_range, activity_type, "
            "fuel_type, country)",
        )
        group_by: List[str] = Field(
            default_factory=lambda: ["activity_type"],
            description="Grouping dimensions (activity_type, fuel_type, country, "
            "facility, year, month)",
        )
        metrics: List[str] = Field(
            default_factory=lambda: ["total_emissions", "consumption_count"],
            description="Metrics to aggregate (total_emissions, avg_emissions, "
            "consumption_count, total_fuel_quantity, total_electricity_quantity)",
        )


# ===================================================================
# Router factory
# ===================================================================


def create_router() -> "APIRouter":
    """Create and return the Fuel & Energy Activities FastAPI APIRouter.

    Returns:
        Configured APIRouter with 20 endpoints.

    Raises:
        RuntimeError: If FastAPI is not installed.
    """
    if not FASTAPI_AVAILABLE:
        raise RuntimeError(
            "FastAPI is required for the fuel & energy activities router"
        )

    router = APIRouter(
        prefix="/api/v1/fuel-energy-activities",
        tags=["Fuel & Energy Activities"],
    )

    # ------------------------------------------------------------------
    # Lazy service accessor
    # ------------------------------------------------------------------

    def _get_service():
        """Get the FuelEnergyActivitiesService singleton.

        Raises HTTPException 503 if the service has not been initialized.
        """
        from greenlang.fuel_energy_activities.setup import get_service
        svc = get_service()
        if svc is None:
            raise HTTPException(
                status_code=503,
                detail="Fuel & Energy Activities service not initialized",
            )
        return svc

    # ==================================================================
    # 1. POST /calculate - Calculate emissions (Activity 3a/3b/3c or combined)
    # ==================================================================

    @router.post("/calculate", status_code=201)
    async def calculate(
        body: CalculateRequest,
    ) -> Dict[str, Any]:
        """Calculate emissions for fuel & energy activities.

        Calculates upstream emissions from:
        - Activity 3a: WTT emissions from purchased fuels
        - Activity 3b: WTT emissions from purchased electricity generation
        - Activity 3c: T&D losses from purchased electricity

        Supports average-data, supplier-specific, and hybrid calculation
        methods per GHG Protocol Scope 3 Category 3 guidance.
        """
        svc = _get_service()

        request_data: Dict[str, Any] = {
            "activity_type": body.activity_type,
            "reporting_period_start": body.reporting_period_start,
            "reporting_period_end": body.reporting_period_end,
            "calculation_method": body.calculation_method,
            "include_uncertainty": body.include_uncertainty,
        }
        if body.fuel_consumption_records is not None:
            request_data["fuel_consumption_records"] = body.fuel_consumption_records
        if body.electricity_consumption_records is not None:
            request_data["electricity_consumption_records"] = body.electricity_consumption_records
        if body.gwp_source is not None:
            request_data["gwp_source"] = body.gwp_source
        if body.tenant_id is not None:
            request_data["tenant_id"] = body.tenant_id

        try:
            result = svc.calculate(request_data)
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "calculate failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 2. POST /calculate/batch - Batch calculation
    # ==================================================================

    @router.post("/calculate/batch", status_code=201)
    async def calculate_batch(
        body: BatchCalculateRequest,
    ) -> Dict[str, Any]:
        """Execute batch calculations for multiple activity periods.

        Processes multiple calculation requests in a single API call.
        Supports parallel execution for improved performance.
        """
        svc = _get_service()
        try:
            result = svc.calculate_batch(
                requests=body.requests,
                parallel=body.parallel,
                tenant_id=body.tenant_id,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "calculate_batch failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 3. GET /calculations - List calculations with pagination
    # ==================================================================

    @router.get("/calculations", status_code=200)
    async def list_calculations(
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(
            50, ge=1, le=500, description="Items per page",
        ),
        tenant_id: Optional[str] = Query(
            None, description="Filter by tenant",
        ),
        activity_type: Optional[str] = Query(
            None, description="Filter by activity type (UPSTREAM_FUEL_WTT, "
            "UPSTREAM_ELECTRICITY_WTT, UPSTREAM_ELECTRICITY_TD_LOSS, ALL)",
        ),
        start_date: Optional[str] = Query(
            None, description="Filter by start date (ISO-8601)",
        ),
        end_date: Optional[str] = Query(
            None, description="Filter by end date (ISO-8601)",
        ),
    ) -> Dict[str, Any]:
        """List calculation results with pagination and filtering.

        Returns paginated list of calculation results with summary statistics
        including total emissions by activity type, consumption counts, and
        calculation method.
        """
        svc = _get_service()
        try:
            result = svc.list_calculations(
                page=page,
                page_size=page_size,
                tenant_id=tenant_id,
                activity_type=activity_type,
                start_date=start_date,
                end_date=end_date,
            )
            return result
        except Exception as exc:
            logger.error(
                "list_calculations failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 4. GET /calculations/{calc_id} - Get specific calculation
    # ==================================================================

    @router.get("/calculations/{calc_id}", status_code=200)
    async def get_calculation(
        calc_id: str = Path(
            ..., description="Calculation identifier",
        ),
    ) -> Dict[str, Any]:
        """Get a calculation result by its unique identifier.

        Returns complete calculation details including total emissions by
        activity (3a/3b/3c), consumption record breakdown, emission factors
        used, method applied, and uncertainty metrics.
        """
        svc = _get_service()
        try:
            result = svc.get_calculation(calc_id)
            if result is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Calculation {calc_id} not found",
                )
            return result
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "get_calculation failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 5. DELETE /calculations/{calc_id} - Delete calculation
    # ==================================================================

    @router.delete("/calculations/{calc_id}", status_code=204)
    async def delete_calculation(
        calc_id: str = Path(
            ..., description="Calculation identifier",
        ),
    ) -> None:
        """Delete a calculation result.

        Permanently removes the calculation and associated data.
        This operation cannot be undone.
        """
        svc = _get_service()
        try:
            success = svc.delete_calculation(calc_id)
            if not success:
                raise HTTPException(
                    status_code=404,
                    detail=f"Calculation {calc_id} not found",
                )
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "delete_calculation failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 6. POST /fuel-consumption - Record fuel consumption
    # ==================================================================

    @router.post("/fuel-consumption", status_code=201)
    async def record_fuel_consumption(
        body: FuelConsumptionRequest,
    ) -> Dict[str, Any]:
        """Record fuel consumption data.

        Stores fuel consumption records for Activity 3a (WTT emissions from
        purchased fuels). Includes fuel type, quantity, region, supplier,
        and optional supplier-specific emission factors.
        """
        svc = _get_service()
        try:
            result = svc.record_fuel_consumption(body.model_dump())
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "record_fuel_consumption failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 7. GET /fuel-consumption - List fuel consumption records
    # ==================================================================

    @router.get("/fuel-consumption", status_code=200)
    async def list_fuel_consumption(
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(
            50, ge=1, le=500, description="Items per page",
        ),
        tenant_id: str = Query(
            ..., min_length=1,
            description="Tenant identifier",
        ),
        fuel_type: Optional[str] = Query(
            None, description="Filter by fuel type",
        ),
        facility_id: Optional[str] = Query(
            None, description="Filter by facility",
        ),
        region: Optional[str] = Query(
            None, description="Filter by region",
        ),
        start_date: Optional[str] = Query(
            None, description="Filter by start date (ISO-8601)",
        ),
        end_date: Optional[str] = Query(
            None, description="Filter by end date (ISO-8601)",
        ),
        search: Optional[str] = Query(
            None, description="Search in record ID or facility ID",
        ),
    ) -> Dict[str, Any]:
        """List fuel consumption records with pagination and filtering.

        Returns paginated list of fuel consumption records with summary
        information and filtering capabilities.
        """
        svc = _get_service()
        try:
            result = svc.list_fuel_consumption(
                page=page,
                page_size=page_size,
                tenant_id=tenant_id,
                fuel_type=fuel_type,
                facility_id=facility_id,
                region=region,
                start_date=start_date,
                end_date=end_date,
                search=search,
            )
            return result
        except Exception as exc:
            logger.error(
                "list_fuel_consumption failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 8. PUT /fuel-consumption/{record_id} - Update fuel consumption record
    # ==================================================================

    @router.put("/fuel-consumption/{record_id}", status_code=200)
    async def update_fuel_consumption(
        record_id: str = Path(
            ..., description="Fuel consumption record identifier",
        ),
        body: FuelConsumptionUpdateRequest = ...,
    ) -> Dict[str, Any]:
        """Update a fuel consumption record.

        Updates fuel consumption details including quantity, unit,
        consumption date, facility, region, and supplier emission factor.
        Only provided fields are updated.
        """
        svc = _get_service()
        try:
            result = svc.update_fuel_consumption(
                record_id=record_id,
                updates=body.model_dump(exclude_unset=True),
            )
            if result is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Fuel consumption record {record_id} not found",
                )
            return result
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "update_fuel_consumption failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 9. POST /electricity-consumption - Record electricity consumption
    # ==================================================================

    @router.post("/electricity-consumption", status_code=201)
    async def record_electricity_consumption(
        body: ElectricityConsumptionRequest,
    ) -> Dict[str, Any]:
        """Record electricity consumption data.

        Stores electricity consumption records for Activity 3b (WTT emissions)
        and Activity 3c (T&D losses). Includes quantity, country, grid type,
        supplier, and optional supplier-specific factors.
        """
        svc = _get_service()
        try:
            result = svc.record_electricity_consumption(body.model_dump())
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "record_electricity_consumption failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 10. GET /electricity-consumption - List electricity consumption records
    # ==================================================================

    @router.get("/electricity-consumption", status_code=200)
    async def list_electricity_consumption(
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(
            50, ge=1, le=500, description="Items per page",
        ),
        tenant_id: str = Query(
            ..., min_length=1,
            description="Tenant identifier",
        ),
        country: Optional[str] = Query(
            None, description="Filter by country",
        ),
        facility_id: Optional[str] = Query(
            None, description="Filter by facility",
        ),
        grid_type: Optional[str] = Query(
            None, description="Filter by grid type",
        ),
        start_date: Optional[str] = Query(
            None, description="Filter by start date (ISO-8601)",
        ),
        end_date: Optional[str] = Query(
            None, description="Filter by end date (ISO-8601)",
        ),
        search: Optional[str] = Query(
            None, description="Search in record ID or facility ID",
        ),
    ) -> Dict[str, Any]:
        """List electricity consumption records with pagination and filtering.

        Returns paginated list of electricity consumption records with
        summary information and filtering capabilities.
        """
        svc = _get_service()
        try:
            result = svc.list_electricity_consumption(
                page=page,
                page_size=page_size,
                tenant_id=tenant_id,
                country=country,
                facility_id=facility_id,
                grid_type=grid_type,
                start_date=start_date,
                end_date=end_date,
                search=search,
            )
            return result
        except Exception as exc:
            logger.error(
                "list_electricity_consumption failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 11. GET /emission-factors - List WTT emission factors
    # ==================================================================

    @router.get("/emission-factors", status_code=200)
    async def list_emission_factors(
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(
            50, ge=1, le=500, description="Items per page",
        ),
        factor_type: Optional[str] = Query(
            None,
            description="Filter by factor type (FUEL_WTT, ELECTRICITY_WTT, TD_LOSS)",
        ),
        fuel_type: Optional[str] = Query(
            None, description="Filter by fuel type (for FUEL_WTT factors)",
        ),
        country: Optional[str] = Query(
            None, description="Filter by country (for ELECTRICITY_WTT/TD_LOSS factors)",
        ),
        source: Optional[str] = Query(
            None,
            description="Filter by EF source (IPCC, GHG_PROTOCOL, DEFRA, "
            "IEA, ECOINVENT, CUSTOM)",
        ),
        search: Optional[str] = Query(
            None, description="Search in fuel type or country",
        ),
    ) -> Dict[str, Any]:
        """List WTT emission factors and T&D loss factors with filtering.

        Returns available emission factors for fuel WTT, electricity WTT,
        and T&D losses including custom factors.
        """
        svc = _get_service()
        try:
            result = svc.list_emission_factors(
                page=page,
                page_size=page_size,
                factor_type=factor_type,
                fuel_type=fuel_type,
                country=country,
                source=source,
                search=search,
            )
            return result
        except Exception as exc:
            logger.error(
                "list_emission_factors failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 12. GET /emission-factors/{factor_id} - Get specific emission factor
    # ==================================================================

    @router.get("/emission-factors/{factor_id}", status_code=200)
    async def get_emission_factor(
        factor_id: str = Path(
            ..., description="Emission factor identifier",
        ),
    ) -> Dict[str, Any]:
        """Get emission factor details.

        Returns complete emission factor information including value,
        unit, data source, geographic/temporal scope, and uncertainty.
        """
        svc = _get_service()
        try:
            result = svc.get_emission_factor(factor_id)
            if result is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Emission factor {factor_id} not found",
                )
            return result
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "get_emission_factor failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 13. POST /emission-factors/custom - Register custom WTT factor
    # ==================================================================

    @router.post("/emission-factors/custom", status_code=201)
    async def register_custom_factor(
        body: CustomFactorRequest,
    ) -> Dict[str, Any]:
        """Register a custom WTT or T&D loss factor.

        Stores tenant-specific emission factors for fuel WTT, electricity WTT,
        or T&D losses. Custom factors take precedence over standard factors.
        """
        svc = _get_service()
        try:
            result = svc.register_custom_factor(body.model_dump())
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "register_custom_factor failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 14. GET /td-loss-factors - List T&D loss factors
    # ==================================================================

    @router.get("/td-loss-factors", status_code=200)
    async def list_td_loss_factors(
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(
            50, ge=1, le=500, description="Items per page",
        ),
        country: Optional[str] = Query(
            None, description="Filter by country code",
        ),
        region: Optional[str] = Query(
            None, description="Filter by region (e.g., OECD, EU, ASIA)",
        ),
        source: Optional[str] = Query(
            None,
            description="Filter by source (IEA, DEFRA, EPA, CUSTOM)",
        ),
        search: Optional[str] = Query(
            None, description="Search in country or region",
        ),
    ) -> Dict[str, Any]:
        """List T&D loss factors with filtering.

        Returns available transmission and distribution loss factors by
        country or region for Activity 3c calculations.
        """
        svc = _get_service()
        try:
            result = svc.list_td_loss_factors(
                page=page,
                page_size=page_size,
                country=country,
                region=region,
                source=source,
                search=search,
            )
            return result
        except Exception as exc:
            logger.error(
                "list_td_loss_factors failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 15. GET /td-loss-factors/{country_code} - Get T&D loss factor by country
    # ==================================================================

    @router.get("/td-loss-factors/{country_code}", status_code=200)
    async def get_td_loss_factor(
        country_code: str = Path(
            ..., min_length=2, max_length=3,
            description="Country code (ISO 3166-1 alpha-2 or alpha-3)",
        ),
    ) -> Dict[str, Any]:
        """Get T&D loss factor for a specific country.

        Returns transmission and distribution loss factor for the specified
        country including value, unit, data source, and uncertainty.
        """
        svc = _get_service()
        try:
            result = svc.get_td_loss_factor(country_code)
            if result is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"T&D loss factor for country {country_code} not found",
                )
            return result
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "get_td_loss_factor failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 16. POST /compliance/check - Run compliance check
    # ==================================================================

    @router.post("/compliance/check", status_code=200)
    async def check_compliance(
        body: ComplianceCheckRequest,
    ) -> Dict[str, Any]:
        """Run regulatory compliance check against multiple frameworks.

        Evaluates the calculation against applicable Scope 3 Category 3
        frameworks: GHG Protocol Scope 3, CSRD/ESRS E1, CDP Climate Change,
        SBTi, ISO 14064-1, TCFD, and GLEC Framework.
        """
        svc = _get_service()
        try:
            result = svc.check_compliance(body.model_dump())
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "check_compliance failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 17. GET /compliance/{check_id} - Get compliance check result
    # ==================================================================

    @router.get("/compliance/{check_id}", status_code=200)
    async def get_compliance_result(
        check_id: str = Path(
            ..., description="Compliance check identifier",
        ),
    ) -> Dict[str, Any]:
        """Get compliance check results.

        Returns detailed compliance check results including framework-specific
        assessments, requirement coverage, and recommendations.
        """
        svc = _get_service()
        try:
            result = svc.get_compliance_result(check_id)
            if result is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Compliance check {check_id} not found",
                )
            return result
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "get_compliance_result failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 18. POST /uncertainty - Run uncertainty analysis
    # ==================================================================

    @router.post("/uncertainty", status_code=200)
    async def analyze_uncertainty(
        body: UncertaintyRequest,
    ) -> Dict[str, Any]:
        """Run uncertainty analysis on calculation results.

        Applies Monte Carlo simulation, pedigree matrix, IPCC Tier 2
        propagation, or bootstrap methods to quantify uncertainty in
        emission estimates for all three activities (3a, 3b, 3c).
        """
        svc = _get_service()
        try:
            result = svc.analyze_uncertainty(body.model_dump())
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "analyze_uncertainty failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 19. GET /aggregations - Get aggregated results
    # ==================================================================

    @router.get("/aggregations", status_code=200)
    async def get_aggregations(
        tenant_id: str = Query(
            ..., min_length=1,
            description="Tenant identifier",
        ),
        group_by: str = Query(
            default="activity_type",
            description="Grouping dimension (activity_type, fuel_type, country, "
            "facility, year, month)",
        ),
        start_date: Optional[str] = Query(
            None, description="Start date filter (ISO-8601)",
        ),
        end_date: Optional[str] = Query(
            None, description="End date filter (ISO-8601)",
        ),
        activity_type: Optional[str] = Query(
            None, description="Filter by activity type",
        ),
        fuel_type: Optional[str] = Query(
            None, description="Filter by fuel type",
        ),
        country: Optional[str] = Query(
            None, description="Filter by country",
        ),
    ) -> Dict[str, Any]:
        """Get aggregated emission results.

        Returns aggregated emissions by activity type, fuel type, country,
        facility, year, or month with summary statistics.
        """
        svc = _get_service()
        try:
            result = svc.get_aggregations(
                tenant_id=tenant_id,
                group_by=group_by,
                start_date=start_date,
                end_date=end_date,
                activity_type=activity_type,
                fuel_type=fuel_type,
                country=country,
            )
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "get_aggregations failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 20. GET /health - Health check endpoint
    # ==================================================================

    @router.get("/health", status_code=200)
    async def health_check() -> Dict[str, Any]:
        """Health check endpoint.

        Returns service health status including database connectivity,
        emission factor data availability, and recent calculation metrics.
        """
        svc = _get_service()
        try:
            result = svc.health_check()
            return result
        except Exception as exc:
            logger.error(
                "health_check failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=503, detail=str(exc))

    return router


# ===================================================================
# Public API
# ===================================================================

__all__ = ["create_router"]
