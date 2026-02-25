# -*- coding: utf-8 -*-
"""
Purchased Goods & Services REST API Router - AGENT-MRV-014
===========================================================

20 REST endpoints for the Purchased Goods & Services Agent (GL-MRV-SCOPE3-001).

Prefix: ``/api/v1/purchased-goods``

Endpoints:
     1. POST   /calculate/spend-based            - Run spend-based calculation
     2. POST   /calculate/average-data           - Run average-data calculation
     3. POST   /calculate/supplier-specific      - Run supplier-specific calculation
     4. POST   /calculate/hybrid                 - Run hybrid (multi-method) calculation
     5. POST   /calculate/batch                  - Batch multi-period calculation
     6. GET    /calculations/{calculation_id}    - Get calculation result
     7. GET    /calculations/{calculation_id}/details - Get line-item detail
     8. POST   /procurement/upload               - Upload procurement records
     9. GET    /procurement/summary              - Get procurement spend summary
    10. POST   /suppliers                        - Register or update supplier profile
    11. GET    /suppliers/{supplier_id}          - Get supplier profile and EF data
    12. GET    /suppliers/{supplier_id}/emissions - Get supplier emission allocation
    13. GET    /emission-factors/eeio            - List EEIO factors (with filtering)
    14. GET    /emission-factors/physical        - List physical EFs (with filtering)
    15. POST   /dqi/score                        - Score data quality for a record set
    16. GET    /dqi/{calculation_id}             - Get DQI results for a calculation
    17. POST   /compliance/check                 - Run compliance check (all 7 frameworks)
    18. GET    /compliance/frameworks            - List available compliance frameworks
    19. POST   /export                           - Export results (JSON/CSV/Excel)
    20. GET    /health                           - Health check

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-014 Purchased Goods & Services (GL-MRV-SCOPE3-001)
Status: Production Ready
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from fastapi import APIRouter, HTTPException, Query, Path, UploadFile, File
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.debug("FastAPI not installed; router unavailable")


# ===================================================================
# Request body models (Pydantic)
# ===================================================================

if FASTAPI_AVAILABLE:

    class SpendBasedCalculationRequest(BaseModel):
        """Request body for spend-based calculation."""

        procurement_records: List[Dict[str, Any]] = Field(
            ..., min_length=1,
            description="List of procurement line items with spend data",
        )
        eeio_database: Optional[str] = Field(
            default=None,
            description="EEIO database to use "
            "(USEEIO_v2_0, EXIOBASE_v3_8, GTAP_v11, EORA_v26, "
            "OPENIO_EUROSTAT_v2, WIOD_2016, DEFRA_2024)",
        )
        currency: str = Field(
            default="USD",
            description="Currency code for spend data (ISO 4217)",
        )
        base_year: Optional[int] = Field(
            default=None, ge=1990, le=2030,
            description="Base year for EEIO factors",
        )
        include_upstream: bool = Field(
            default=True,
            description="Include upstream supply chain emissions",
        )
        gwp_source: Optional[str] = Field(
            default=None,
            description="GWP source (AR4, AR5, AR6, AR6_GTP)",
        )
        tenant_id: Optional[str] = Field(
            default=None,
            description="Owning tenant identifier",
        )

    class AverageDataCalculationRequest(BaseModel):
        """Request body for average-data calculation."""

        procurement_records: List[Dict[str, Any]] = Field(
            ..., min_length=1,
            description="List of procurement line items with physical quantities",
        )
        physical_ef_source: Optional[str] = Field(
            default=None,
            description="Physical EF source "
            "(GHG_PROTOCOL, IPCC_2006, DEFRA_2024, EPA_2024, "
            "ADEME_v23, ECOINVENT_v3_10)",
        )
        product_category: Optional[str] = Field(
            default=None,
            description="Product category for EF lookup",
        )
        gwp_source: Optional[str] = Field(
            default=None,
            description="GWP source (AR4, AR5, AR6, AR6_GTP)",
        )
        tenant_id: Optional[str] = Field(
            default=None,
            description="Owning tenant identifier",
        )

    class SupplierSpecificCalculationRequest(BaseModel):
        """Request body for supplier-specific calculation."""

        procurement_records: List[Dict[str, Any]] = Field(
            ..., min_length=1,
            description="List of procurement line items with supplier data",
        )
        supplier_emissions: Dict[str, Any] = Field(
            ...,
            description="Supplier-reported emission intensities or total emissions",
        )
        allocation_method: str = Field(
            default="revenue",
            description="Allocation method "
            "(revenue, mass, economic_value, energy, custom)",
        )
        gwp_source: Optional[str] = Field(
            default=None,
            description="GWP source (AR4, AR5, AR6, AR6_GTP)",
        )
        tenant_id: Optional[str] = Field(
            default=None,
            description="Owning tenant identifier",
        )

    class HybridCalculationRequest(BaseModel):
        """Request body for hybrid (multi-method) calculation."""

        procurement_records: List[Dict[str, Any]] = Field(
            ..., min_length=1,
            description="List of procurement line items with mixed data availability",
        )
        method_priority: List[str] = Field(
            default=["supplier_specific", "average_data", "spend_based"],
            description="Method priority cascade "
            "(supplier_specific, average_data, spend_based)",
        )
        supplier_emissions: Optional[Dict[str, Any]] = Field(
            default=None,
            description="Supplier-reported emission data (if available)",
        )
        eeio_database: Optional[str] = Field(
            default=None,
            description="EEIO database for spend-based fallback",
        )
        physical_ef_source: Optional[str] = Field(
            default=None,
            description="Physical EF source for average-data fallback",
        )
        currency: str = Field(
            default="USD",
            description="Currency code for spend data (ISO 4217)",
        )
        gwp_source: Optional[str] = Field(
            default=None,
            description="GWP source (AR4, AR5, AR6, AR6_GTP)",
        )
        tenant_id: Optional[str] = Field(
            default=None,
            description="Owning tenant identifier",
        )

    class BatchCalculationBody(BaseModel):
        """Request body for batch multi-period calculations."""

        periods: List[Dict[str, Any]] = Field(
            ..., min_length=1,
            description="List of reporting periods with procurement data",
        )
        method: str = Field(
            default="hybrid",
            description="Calculation method (spend_based, average_data, "
            "supplier_specific, hybrid)",
        )
        eeio_database: Optional[str] = Field(
            default=None,
            description="EEIO database",
        )
        physical_ef_source: Optional[str] = Field(
            default=None,
            description="Physical EF source",
        )
        currency: str = Field(
            default="USD",
            description="Currency code (ISO 4217)",
        )
        gwp_source: Optional[str] = Field(
            default=None,
            description="GWP source (AR4, AR5, AR6, AR6_GTP)",
        )
        tenant_id: Optional[str] = Field(
            default=None,
            description="Owning tenant identifier",
        )

    class ProcurementUploadBody(BaseModel):
        """Request body for procurement record upload."""

        records: List[Dict[str, Any]] = Field(
            ..., min_length=1,
            description="List of procurement records",
        )
        period_start: str = Field(
            ...,
            description="Reporting period start date (ISO-8601)",
        )
        period_end: str = Field(
            ...,
            description="Reporting period end date (ISO-8601)",
        )
        tenant_id: str = Field(
            ..., min_length=1,
            description="Owning tenant identifier",
        )
        source_system: Optional[str] = Field(
            default=None,
            description="Source ERP/procurement system",
        )
        currency: str = Field(
            default="USD",
            description="Currency code (ISO 4217)",
        )

    class SupplierBody(BaseModel):
        """Request body for supplier registration/update."""

        supplier_id: str = Field(
            ..., min_length=1,
            description="Unique supplier identifier",
        )
        name: str = Field(
            ..., min_length=1, max_length=500,
            description="Supplier legal name",
        )
        emission_intensity: Optional[float] = Field(
            default=None, ge=0,
            description="Supplier-reported emission intensity (tCO2e per unit)",
        )
        emission_intensity_unit: Optional[str] = Field(
            default=None,
            description="Unit for emission intensity (revenue, mass, energy)",
        )
        total_emissions: Optional[float] = Field(
            default=None, ge=0,
            description="Supplier total annual emissions (tCO2e)",
        )
        total_revenue: Optional[float] = Field(
            default=None, ge=0,
            description="Supplier total annual revenue (currency)",
        )
        revenue_currency: Optional[str] = Field(
            default=None,
            description="Currency for revenue (ISO 4217)",
        )
        product_categories: Optional[List[str]] = Field(
            default=None,
            description="Primary product categories supplied",
        )
        country_code: Optional[str] = Field(
            default=None, max_length=2,
            description="ISO 3166-1 alpha-2 country code",
        )
        tenant_id: str = Field(
            ..., min_length=1,
            description="Owning tenant identifier",
        )
        data_source: Optional[str] = Field(
            default=None,
            description="Data source (CDP, SBTi, annual_report, direct_report)",
        )
        reporting_year: Optional[int] = Field(
            default=None, ge=2000, le=2030,
            description="Year of reported data",
        )

    class DQIScoreBody(BaseModel):
        """Request body for data quality scoring."""

        records: List[Dict[str, Any]] = Field(
            ..., min_length=1,
            description="Records to score for data quality",
        )
        method: str = Field(
            default="hybrid",
            description="Calculation method used",
        )
        coverage_level: Optional[str] = Field(
            default=None,
            description="Coverage level (FULL, PARTIAL, MINIMAL)",
        )

    class ComplianceCheckBody(BaseModel):
        """Request body for compliance check."""

        calculation_id: str = Field(
            ..., min_length=1,
            description="ID of a previous calculation",
        )
        frameworks: List[str] = Field(
            default_factory=list,
            description="Frameworks to check (empty = all frameworks). "
            "Options: GHG_PROTOCOL, CSRD, CDP, SBTi, ISO14064, "
            "TCFD, GLEC",
        )
        tenant_id: Optional[str] = Field(
            default=None,
            description="Tenant identifier for scoping",
        )

    class ExportBody(BaseModel):
        """Request body for export."""

        calculation_id: str = Field(
            ..., min_length=1,
            description="Calculation ID to export",
        )
        export_format: str = Field(
            default="JSON",
            description="Export format (JSON, CSV, EXCEL, GHG_PROTOCOL_TEMPLATE)",
        )
        include_line_items: bool = Field(
            default=True,
            description="Include line-item details",
        )
        include_dqi: bool = Field(
            default=False,
            description="Include DQI scores",
        )
        include_compliance: bool = Field(
            default=False,
            description="Include compliance check results",
        )


# ===================================================================
# Router factory
# ===================================================================


def create_router() -> "APIRouter":
    """Create and return the Purchased Goods & Services FastAPI APIRouter.

    Returns:
        Configured APIRouter with 20 endpoints.

    Raises:
        RuntimeError: If FastAPI is not installed.
    """
    if not FASTAPI_AVAILABLE:
        raise RuntimeError(
            "FastAPI is required for the purchased goods & services router"
        )

    router = APIRouter(
        prefix="/api/v1/purchased-goods",
        tags=["Purchased Goods & Services"],
    )

    # ------------------------------------------------------------------
    # Lazy service accessor
    # ------------------------------------------------------------------

    def _get_service():
        """Get the PurchasedGoodsServicesService singleton.

        Raises HTTPException 503 if the service has not been initialized.
        """
        from greenlang.purchased_goods_services.setup import get_service
        svc = get_service()
        if svc is None:
            raise HTTPException(
                status_code=503,
                detail="Purchased Goods & Services service not initialized",
            )
        return svc

    # ==================================================================
    # 1. POST /calculate/spend-based - Run spend-based calculation
    # ==================================================================

    @router.post("/calculate/spend-based", status_code=201)
    async def calculate_spend_based(
        body: SpendBasedCalculationRequest,
    ) -> Dict[str, Any]:
        """Execute spend-based calculation using EEIO factors.

        Applies EEIO (Environmentally Extended Input-Output) emission factors
        to procurement spend data. This is the GHG Protocol recommended
        approach when supplier-specific or physical activity data is
        not available.
        """
        svc = _get_service()

        request_data: Dict[str, Any] = {
            "procurement_records": body.procurement_records,
            "currency": body.currency,
            "include_upstream": body.include_upstream,
        }
        if body.eeio_database is not None:
            request_data["eeio_database"] = body.eeio_database
        if body.base_year is not None:
            request_data["base_year"] = body.base_year
        if body.gwp_source is not None:
            request_data["gwp_source"] = body.gwp_source
        if body.tenant_id is not None:
            request_data["tenant_id"] = body.tenant_id

        try:
            result = svc.calculate_spend_based(request_data)
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "calculate_spend_based failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 2. POST /calculate/average-data - Run average-data calculation
    # ==================================================================

    @router.post("/calculate/average-data", status_code=201)
    async def calculate_average_data(
        body: AverageDataCalculationRequest,
    ) -> Dict[str, Any]:
        """Execute average-data calculation using physical emission factors.

        Applies physical emission factors (e.g., kgCO2e per kg product) to
        procurement quantity data. This is the GHG Protocol Scope 3
        Category 1 approach when physical activity data is available.
        """
        svc = _get_service()

        request_data: Dict[str, Any] = {
            "procurement_records": body.procurement_records,
        }
        if body.physical_ef_source is not None:
            request_data["physical_ef_source"] = body.physical_ef_source
        if body.product_category is not None:
            request_data["product_category"] = body.product_category
        if body.gwp_source is not None:
            request_data["gwp_source"] = body.gwp_source
        if body.tenant_id is not None:
            request_data["tenant_id"] = body.tenant_id

        try:
            result = svc.calculate_average_data(request_data)
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "calculate_average_data failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 3. POST /calculate/supplier-specific - Run supplier-specific
    # ==================================================================

    @router.post("/calculate/supplier-specific", status_code=201)
    async def calculate_supplier_specific(
        body: SupplierSpecificCalculationRequest,
    ) -> Dict[str, Any]:
        """Execute supplier-specific calculation using primary data.

        Allocates supplier-reported emissions to purchased goods/services
        using revenue, mass, economic value, or energy allocation methods.
        This is the highest-quality approach per GHG Protocol hierarchy.
        """
        svc = _get_service()

        request_data: Dict[str, Any] = {
            "procurement_records": body.procurement_records,
            "supplier_emissions": body.supplier_emissions,
            "allocation_method": body.allocation_method,
        }
        if body.gwp_source is not None:
            request_data["gwp_source"] = body.gwp_source
        if body.tenant_id is not None:
            request_data["tenant_id"] = body.tenant_id

        try:
            result = svc.calculate_supplier_specific(request_data)
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "calculate_supplier_specific failed: %s", exc,
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 4. POST /calculate/hybrid - Run hybrid (multi-method)
    # ==================================================================

    @router.post("/calculate/hybrid", status_code=201)
    async def calculate_hybrid(
        body: HybridCalculationRequest,
    ) -> Dict[str, Any]:
        """Execute hybrid calculation using multiple methods.

        Applies the data quality hierarchy: supplier-specific data (best),
        average-data (secondary), spend-based (tertiary). Each line item
        uses the best available method based on data availability.
        """
        svc = _get_service()

        request_data: Dict[str, Any] = {
            "procurement_records": body.procurement_records,
            "method_priority": body.method_priority,
            "currency": body.currency,
        }
        if body.supplier_emissions is not None:
            request_data["supplier_emissions"] = body.supplier_emissions
        if body.eeio_database is not None:
            request_data["eeio_database"] = body.eeio_database
        if body.physical_ef_source is not None:
            request_data["physical_ef_source"] = body.physical_ef_source
        if body.gwp_source is not None:
            request_data["gwp_source"] = body.gwp_source
        if body.tenant_id is not None:
            request_data["tenant_id"] = body.tenant_id

        try:
            result = svc.calculate_hybrid(request_data)
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "calculate_hybrid failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 5. POST /calculate/batch - Batch multi-period calculation
    # ==================================================================

    @router.post("/calculate/batch", status_code=201)
    async def calculate_batch(
        body: BatchCalculationBody,
    ) -> Dict[str, Any]:
        """Execute batch multi-period calculations.

        Processes multiple reporting periods in a single request.
        Each period can have different procurement records and
        calculation parameters.
        """
        svc = _get_service()
        try:
            result = svc.calculate_batch(
                periods=body.periods,
                method=body.method,
                eeio_database=body.eeio_database,
                physical_ef_source=body.physical_ef_source,
                currency=body.currency,
                gwp_source=body.gwp_source,
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
    # 6. GET /calculations/{calculation_id} - Get calculation result
    # ==================================================================

    @router.get("/calculations/{calculation_id}", status_code=200)
    async def get_calculation(
        calculation_id: str = Path(
            ..., description="Calculation identifier",
        ),
    ) -> Dict[str, Any]:
        """Get a calculation result by its unique identifier.

        Returns the full calculation result including total emissions,
        method breakdown, coverage statistics, and DQI score.
        """
        svc = _get_service()

        for calc in svc._calculations:
            if calc.get("calculation_id") == calculation_id:
                return calc

        raise HTTPException(
            status_code=404,
            detail=f"Calculation {calculation_id} not found",
        )

    # ==================================================================
    # 7. GET /calculations/{calculation_id}/details - Line-item detail
    # ==================================================================

    @router.get(
        "/calculations/{calculation_id}/details", status_code=200,
    )
    async def get_calculation_details(
        calculation_id: str = Path(
            ..., description="Calculation identifier",
        ),
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(
            50, ge=1, le=500, description="Items per page",
        ),
        method: Optional[str] = Query(
            None,
            description="Filter by calculation method",
        ),
        product_category: Optional[str] = Query(
            None,
            description="Filter by product category",
        ),
    ) -> Dict[str, Any]:
        """Get line-item details for a calculation.

        Returns detailed emission results for each procurement line item,
        including method used, emission factors applied, and
        uncertainty metrics.
        """
        svc = _get_service()
        try:
            result = svc.get_calculation_details(
                calculation_id=calculation_id,
                page=page,
                page_size=page_size,
                method=method,
                product_category=product_category,
            )
            return result
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            logger.error(
                "get_calculation_details failed: %s", exc,
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 8. POST /procurement/upload - Upload procurement records
    # ==================================================================

    @router.post("/procurement/upload", status_code=201)
    async def upload_procurement(
        body: ProcurementUploadBody,
    ) -> Dict[str, Any]:
        """Upload procurement records for a reporting period.

        Stores procurement data for future calculations. Records are
        validated against the procurement schema and stored by tenant
        and reporting period.
        """
        svc = _get_service()
        try:
            result = svc.upload_procurement(body.model_dump())
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "upload_procurement failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 9. GET /procurement/summary - Get procurement spend summary
    # ==================================================================

    @router.get("/procurement/summary", status_code=200)
    async def get_procurement_summary(
        tenant_id: str = Query(
            ..., min_length=1,
            description="Tenant identifier",
        ),
        period_start: Optional[str] = Query(
            None, description="Period start date (ISO-8601)",
        ),
        period_end: Optional[str] = Query(
            None, description="Period end date (ISO-8601)",
        ),
        group_by: Optional[str] = Query(
            None,
            description="Grouping dimension "
            "(supplier, category, country, month)",
        ),
    ) -> Dict[str, Any]:
        """Get procurement spend summary.

        Returns aggregated procurement statistics including total spend,
        supplier count, category distribution, and geographic breakdown.
        """
        svc = _get_service()
        try:
            result = svc.get_procurement_summary(
                tenant_id=tenant_id,
                period_start=period_start,
                period_end=period_end,
                group_by=group_by,
            )
            return result
        except Exception as exc:
            logger.error(
                "get_procurement_summary failed: %s", exc,
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 10. POST /suppliers - Register or update supplier profile
    # ==================================================================

    @router.post("/suppliers", status_code=201)
    async def create_supplier(
        body: SupplierBody,
    ) -> Dict[str, Any]:
        """Register or update a supplier profile.

        Stores supplier-specific emission data including emission
        intensity, total emissions, revenue, and product categories.
        Used for supplier-specific calculations.
        """
        svc = _get_service()
        try:
            result = svc.register_supplier(body.model_dump())
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "create_supplier failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 11. GET /suppliers/{supplier_id} - Get supplier profile
    # ==================================================================

    @router.get("/suppliers/{supplier_id}", status_code=200)
    async def get_supplier(
        supplier_id: str = Path(
            ..., description="Supplier identifier",
        ),
        tenant_id: Optional[str] = Query(
            None, description="Tenant identifier for scoping",
        ),
    ) -> Dict[str, Any]:
        """Get supplier profile and emission factor data.

        Returns supplier details including name, emission intensity,
        total emissions, revenue, and reporting metadata.
        """
        svc = _get_service()
        try:
            result = svc.get_supplier(
                supplier_id=supplier_id,
                tenant_id=tenant_id,
            )
            if result is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Supplier {supplier_id} not found",
                )
            return result
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "get_supplier failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 12. GET /suppliers/{supplier_id}/emissions - Emission allocation
    # ==================================================================

    @router.get(
        "/suppliers/{supplier_id}/emissions", status_code=200,
    )
    async def get_supplier_emissions(
        supplier_id: str = Path(
            ..., description="Supplier identifier",
        ),
        tenant_id: str = Query(
            ..., min_length=1,
            description="Tenant identifier",
        ),
        allocation_method: str = Query(
            default="revenue",
            description="Allocation method (revenue, mass, economic_value)",
        ),
        purchased_amount: Optional[float] = Query(
            default=None, ge=0,
            description="Purchased amount for allocation",
        ),
    ) -> Dict[str, Any]:
        """Get supplier emission allocation for purchased goods/services.

        Calculates allocated emissions based on supplier's total emissions
        and the specified allocation method (revenue, mass, economic value).
        """
        svc = _get_service()
        try:
            result = svc.get_supplier_emissions(
                supplier_id=supplier_id,
                tenant_id=tenant_id,
                allocation_method=allocation_method,
                purchased_amount=purchased_amount,
            )
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "get_supplier_emissions failed: %s", exc,
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 13. GET /emission-factors/eeio - List EEIO factors
    # ==================================================================

    @router.get("/emission-factors/eeio", status_code=200)
    async def list_eeio_factors(
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(
            50, ge=1, le=500, description="Items per page",
        ),
        database: Optional[str] = Query(
            None,
            description="Filter by EEIO database",
        ),
        sector: Optional[str] = Query(
            None,
            description="Filter by sector code or name",
        ),
        base_year: Optional[int] = Query(
            None, ge=1990, le=2030,
            description="Filter by base year",
        ),
        search: Optional[str] = Query(
            None,
            description="Search in sector name or description",
        ),
    ) -> Dict[str, Any]:
        """List EEIO emission factors with filtering.

        Returns available EEIO factors from databases including USEEIO,
        EXIOBASE, GTAP, EORA, OpenIO Eurostat, WIOD, and DEFRA.
        """
        svc = _get_service()
        try:
            result = svc.list_eeio_factors(
                page=page,
                page_size=page_size,
                database=database,
                sector=sector,
                base_year=base_year,
                search=search,
            )
            return result
        except Exception as exc:
            logger.error(
                "list_eeio_factors failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 14. GET /emission-factors/physical - List physical EFs
    # ==================================================================

    @router.get("/emission-factors/physical", status_code=200)
    async def list_physical_factors(
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(
            50, ge=1, le=500, description="Items per page",
        ),
        source: Optional[str] = Query(
            None,
            description="Filter by EF source",
        ),
        category: Optional[str] = Query(
            None,
            description="Filter by product category",
        ),
        region: Optional[str] = Query(
            None,
            description="Filter by region/country",
        ),
        search: Optional[str] = Query(
            None,
            description="Search in product name or description",
        ),
    ) -> Dict[str, Any]:
        """List physical emission factors with filtering.

        Returns available physical emission factors from sources including
        GHG Protocol, IPCC 2006, DEFRA, EPA, ADEME, and ecoinvent.
        """
        svc = _get_service()
        try:
            result = svc.list_physical_factors(
                page=page,
                page_size=page_size,
                source=source,
                category=category,
                region=region,
                search=search,
            )
            return result
        except Exception as exc:
            logger.error(
                "list_physical_factors failed: %s", exc,
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 15. POST /dqi/score - Score data quality
    # ==================================================================

    @router.post("/dqi/score", status_code=200)
    async def score_dqi(
        body: DQIScoreBody,
    ) -> Dict[str, Any]:
        """Score data quality for a record set.

        Calculates GHG Protocol Data Quality Indicators (DQI) including
        Technological Representativeness, Temporal Representativeness,
        Geographic Representativeness, Completeness, and Reliability.
        """
        svc = _get_service()
        try:
            result = svc.score_dqi(
                records=body.records,
                method=body.method,
                coverage_level=body.coverage_level,
            )
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "score_dqi failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 16. GET /dqi/{calculation_id} - Get DQI results
    # ==================================================================

    @router.get("/dqi/{calculation_id}", status_code=200)
    async def get_dqi(
        calculation_id: str = Path(
            ..., description="Calculation identifier",
        ),
    ) -> Dict[str, Any]:
        """Get DQI results for a calculation.

        Returns Data Quality Indicator scores for the specified
        calculation including overall DQI, dimension scores, and
        line-item level detail.
        """
        svc = _get_service()
        try:
            result = svc.get_dqi(calculation_id=calculation_id)
            if result is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"DQI for calculation {calculation_id} not found",
                )
            return result
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "get_dqi failed: %s", exc, exc_info=True,
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

        Evaluates the calculation against applicable Scope 3 Category 1
        frameworks: GHG Protocol Scope 3, CSRD/ESRS E1, CDP Supply Chain,
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
    # 18. GET /compliance/frameworks - List compliance frameworks
    # ==================================================================

    @router.get("/compliance/frameworks", status_code=200)
    async def list_compliance_frameworks() -> Dict[str, Any]:
        """List available compliance frameworks.

        Returns metadata for all supported Scope 3 Category 1 compliance
        frameworks including requirements, guidance references, and
        validation rules.
        """
        svc = _get_service()
        try:
            result = svc.list_compliance_frameworks()
            return result
        except Exception as exc:
            logger.error(
                "list_compliance_frameworks failed: %s", exc,
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 19. POST /export - Export results
    # ==================================================================

    @router.post("/export", status_code=200)
    async def export_results(
        body: ExportBody,
    ) -> Dict[str, Any]:
        """Export calculation results to various formats.

        Exports results as JSON, CSV, Excel, or GHG Protocol Corporate
        Value Chain (Scope 3) Accounting and Reporting Standard template.
        """
        svc = _get_service()
        try:
            result = svc.export_results(body.model_dump())
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "export_results failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 20. GET /health - Health check
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
