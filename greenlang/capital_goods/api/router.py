# -*- coding: utf-8 -*-
"""
Capital Goods REST API Router - AGENT-MRV-015
==============================================

20 REST endpoints for the Capital Goods Agent (GL-MRV-S3-002 / Scope 3 Category 2).

Prefix: ``/api/v1/capital-goods``

Endpoints:
     1. POST   /calculate                        - Calculate emissions for capital goods
     2. POST   /calculate/batch                  - Batch calculation
     3. GET    /calculations                     - List calculations with pagination
     4. GET    /calculations/{calc_id}           - Get specific calculation
     5. DELETE /calculations/{calc_id}           - Delete calculation
     6. POST   /assets                           - Register capital asset
     7. GET    /assets                           - List capital assets
     8. PUT    /assets/{asset_id}                - Update capital asset
     9. GET    /emission-factors                 - List emission factors
    10. GET    /emission-factors/{factor_id}     - Get specific factor
    11. POST   /emission-factors/custom          - Register custom EF
    12. POST   /classify                         - Classify assets
    13. POST   /compliance/check                 - Run compliance checks
    14. GET    /compliance/{check_id}            - Get compliance result
    15. POST   /uncertainty                      - Run uncertainty analysis
    16. GET    /aggregations                     - Get aggregated results
    17. GET    /hot-spots                        - Get hot-spot analysis
    18. POST   /export                           - Export results
    19. GET    /health                           - Health check
    20. GET    /stats                            - Engine statistics

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-015 Capital Goods (GL-MRV-SCOPE3-002)
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
        """Request body for capital goods emission calculation."""

        asset_records: List[Dict[str, Any]] = Field(
            ..., min_length=1,
            description="List of capital asset records with purchase/construction data",
        )
        calculation_method: str = Field(
            default="LIFECYCLE_HYBRID",
            description="Calculation method (LIFECYCLE_AVERAGE_DATA, "
            "LIFECYCLE_SUPPLIER_SPECIFIC, LIFECYCLE_HYBRID, "
            "ANNUALIZED_AMORTIZATION, ANNUALIZED_STRAIGHT_LINE)",
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

    class AssetRegisterRequest(BaseModel):
        """Request body for asset registration."""

        asset_id: str = Field(
            ..., min_length=1, max_length=200,
            description="Unique asset identifier",
        )
        asset_name: str = Field(
            ..., min_length=1, max_length=500,
            description="Asset name or description",
        )
        asset_type: str = Field(
            ..., min_length=1, max_length=100,
            description="Asset type (BUILDING, MACHINERY, VEHICLE, "
            "EQUIPMENT, INFRASTRUCTURE, FURNITURE, IT_HARDWARE, OTHER)",
        )
        asset_category: Optional[str] = Field(
            default=None, max_length=200,
            description="Detailed asset category",
        )
        purchase_date: str = Field(
            ...,
            description="Purchase/construction completion date (ISO-8601)",
        )
        purchase_cost: float = Field(
            ..., ge=0,
            description="Total purchase/construction cost (monetary)",
        )
        currency: str = Field(
            default="USD", max_length=3,
            description="Currency code (ISO 4217)",
        )
        useful_life_years: Optional[float] = Field(
            default=None, ge=1, le=100,
            description="Expected useful life in years",
        )
        supplier_id: Optional[str] = Field(
            default=None, max_length=200,
            description="Supplier/manufacturer identifier",
        )
        location: Optional[str] = Field(
            default=None, max_length=200,
            description="Asset location (country/region code)",
        )
        physical_quantity: Optional[float] = Field(
            default=None, ge=0,
            description="Physical quantity (e.g., mass in kg, area in m²)",
        )
        physical_unit: Optional[str] = Field(
            default=None, max_length=50,
            description="Unit for physical quantity",
        )
        metadata: Optional[Dict[str, Any]] = Field(
            default=None,
            description="Additional asset metadata",
        )
        tenant_id: str = Field(
            ..., min_length=1,
            description="Owning tenant identifier",
        )

    class AssetUpdateRequest(BaseModel):
        """Request body for asset update."""

        asset_name: Optional[str] = Field(
            default=None, max_length=500,
            description="Updated asset name",
        )
        asset_type: Optional[str] = Field(
            default=None, max_length=100,
            description="Updated asset type",
        )
        asset_category: Optional[str] = Field(
            default=None, max_length=200,
            description="Updated asset category",
        )
        useful_life_years: Optional[float] = Field(
            default=None, ge=1, le=100,
            description="Updated useful life",
        )
        location: Optional[str] = Field(
            default=None, max_length=200,
            description="Updated location",
        )
        metadata: Optional[Dict[str, Any]] = Field(
            default=None,
            description="Additional metadata updates",
        )

    class CustomEFRequest(BaseModel):
        """Request body for custom emission factor registration."""

        ef_id: str = Field(
            ..., min_length=1, max_length=200,
            description="Unique emission factor identifier",
        )
        asset_category: str = Field(
            ..., min_length=1, max_length=200,
            description="Applicable asset category",
        )
        emission_factor: float = Field(
            ..., ge=0,
            description="Emission factor value",
        )
        ef_unit: str = Field(
            ..., min_length=1, max_length=100,
            description="Emission factor unit (e.g., kgCO2e/USD, tCO2e/kg)",
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

    class ClassifyRequest(BaseModel):
        """Request body for asset classification."""

        records: List[Dict[str, Any]] = Field(
            ..., min_length=1,
            description="Asset records to classify",
        )
        classification_level: str = Field(
            default="DETAILED",
            description="Classification level (BROAD, DETAILED, GRANULAR)",
        )

    class ComplianceRequest(BaseModel):
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

    class ExportRequest(BaseModel):
        """Request body for result export."""

        result_id: str = Field(
            ..., min_length=1,
            description="Calculation result identifier",
        )
        format: str = Field(
            default="JSON",
            description="Export format (JSON, CSV, EXCEL, GHG_PROTOCOL_TEMPLATE)",
        )
        include_assets: bool = Field(
            default=True,
            description="Include asset-level details",
        )
        include_uncertainty: bool = Field(
            default=False,
            description="Include uncertainty analysis",
        )
        include_compliance: bool = Field(
            default=False,
            description="Include compliance check results",
        )

    class AggregationRequest(BaseModel):
        """Request body for aggregation query."""

        filters: Optional[Dict[str, Any]] = Field(
            default=None,
            description="Filters to apply (tenant_id, date_range, asset_type)",
        )
        group_by: List[str] = Field(
            default_factory=lambda: ["asset_type"],
            description="Grouping dimensions (asset_type, category, year, "
            "supplier, location)",
        )
        metrics: List[str] = Field(
            default_factory=lambda: ["total_emissions", "asset_count"],
            description="Metrics to aggregate (total_emissions, avg_emissions, "
            "asset_count, total_cost)",
        )


# ===================================================================
# Router factory
# ===================================================================


def create_router() -> "APIRouter":
    """Create and return the Capital Goods FastAPI APIRouter.

    Returns:
        Configured APIRouter with 20 endpoints.

    Raises:
        RuntimeError: If FastAPI is not installed.
    """
    if not FASTAPI_AVAILABLE:
        raise RuntimeError(
            "FastAPI is required for the capital goods router"
        )

    router = APIRouter(
        prefix="/api/v1/capital-goods",
        tags=["Capital Goods"],
    )

    # ------------------------------------------------------------------
    # Lazy service accessor
    # ------------------------------------------------------------------

    def _get_service():
        """Get the CapitalGoodsService singleton.

        Raises HTTPException 503 if the service has not been initialized.
        """
        from greenlang.capital_goods.setup import get_service
        svc = get_service()
        if svc is None:
            raise HTTPException(
                status_code=503,
                detail="Capital Goods service not initialized",
            )
        return svc

    # ==================================================================
    # 1. POST /calculate - Calculate emissions for capital goods
    # ==================================================================

    @router.post("/calculate", status_code=201)
    async def calculate(
        body: CalculateRequest,
    ) -> Dict[str, Any]:
        """Calculate emissions for capital goods.

        Applies lifecycle-based or annualized amortization methods to capital
        asset data. Supports average-data, supplier-specific, and hybrid
        calculation approaches per GHG Protocol Scope 3 Category 2 guidance.
        """
        svc = _get_service()

        request_data: Dict[str, Any] = {
            "asset_records": body.asset_records,
            "calculation_method": body.calculation_method,
            "include_uncertainty": body.include_uncertainty,
        }
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
        """Execute batch calculations for multiple asset sets.

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
        method: Optional[str] = Query(
            None, description="Filter by calculation method",
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
        including total emissions, asset count, and calculation method.
        """
        svc = _get_service()
        try:
            result = svc.list_calculations(
                page=page,
                page_size=page_size,
                tenant_id=tenant_id,
                method=method,
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

        Returns complete calculation details including total emissions,
        asset-level breakdown, method used, and uncertainty metrics.
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
    # 6. POST /assets - Register capital asset
    # ==================================================================

    @router.post("/assets", status_code=201)
    async def register_asset(
        body: AssetRegisterRequest,
    ) -> Dict[str, Any]:
        """Register a capital asset.

        Stores capital asset details including purchase data, type,
        useful life, and physical characteristics for future calculations.
        """
        svc = _get_service()
        try:
            result = svc.register_asset(body.model_dump())
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "register_asset failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 7. GET /assets - List capital assets
    # ==================================================================

    @router.get("/assets", status_code=200)
    async def list_assets(
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(
            50, ge=1, le=500, description="Items per page",
        ),
        tenant_id: str = Query(
            ..., min_length=1,
            description="Tenant identifier",
        ),
        asset_type: Optional[str] = Query(
            None, description="Filter by asset type",
        ),
        category: Optional[str] = Query(
            None, description="Filter by asset category",
        ),
        location: Optional[str] = Query(
            None, description="Filter by location",
        ),
        search: Optional[str] = Query(
            None, description="Search in asset name or ID",
        ),
    ) -> Dict[str, Any]:
        """List capital assets with pagination and filtering.

        Returns paginated list of registered capital assets with
        summary information and filtering capabilities.
        """
        svc = _get_service()
        try:
            result = svc.list_assets(
                page=page,
                page_size=page_size,
                tenant_id=tenant_id,
                asset_type=asset_type,
                category=category,
                location=location,
                search=search,
            )
            return result
        except Exception as exc:
            logger.error(
                "list_assets failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 8. PUT /assets/{asset_id} - Update capital asset
    # ==================================================================

    @router.put("/assets/{asset_id}", status_code=200)
    async def update_asset(
        asset_id: str = Path(
            ..., description="Asset identifier",
        ),
        body: AssetUpdateRequest = ...,
    ) -> Dict[str, Any]:
        """Update a capital asset.

        Updates asset details including name, type, category, useful life,
        and location. Only provided fields are updated.
        """
        svc = _get_service()
        try:
            result = svc.update_asset(
                asset_id=asset_id,
                updates=body.model_dump(exclude_unset=True),
            )
            if result is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Asset {asset_id} not found",
                )
            return result
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "update_asset failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 9. GET /emission-factors - List emission factors
    # ==================================================================

    @router.get("/emission-factors", status_code=200)
    async def list_emission_factors(
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(
            50, ge=1, le=500, description="Items per page",
        ),
        source: Optional[str] = Query(
            None,
            description="Filter by EF source (IPCC, GHG_PROTOCOL, ECOINVENT, "
            "DEFRA, EPA, ADEME, CUSTOM)",
        ),
        category: Optional[str] = Query(
            None, description="Filter by asset category",
        ),
        region: Optional[str] = Query(
            None, description="Filter by geographic scope",
        ),
        search: Optional[str] = Query(
            None, description="Search in category or description",
        ),
    ) -> Dict[str, Any]:
        """List emission factors with filtering.

        Returns available emission factors for capital goods including
        lifecycle EFs and custom factors.
        """
        svc = _get_service()
        try:
            result = svc.list_emission_factors(
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
                "list_emission_factors failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 10. GET /emission-factors/{factor_id} - Get specific factor
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
    # 11. POST /emission-factors/custom - Register custom EF
    # ==================================================================

    @router.post("/emission-factors/custom", status_code=201)
    async def register_custom_ef(
        body: CustomEFRequest,
    ) -> Dict[str, Any]:
        """Register a custom emission factor.

        Stores tenant-specific emission factors for capital goods
        categories. Custom EFs take precedence over standard factors.
        """
        svc = _get_service()
        try:
            result = svc.register_custom_ef(body.model_dump())
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "register_custom_ef failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 12. POST /classify - Classify assets
    # ==================================================================

    @router.post("/classify", status_code=200)
    async def classify_assets(
        body: ClassifyRequest,
    ) -> Dict[str, Any]:
        """Classify capital assets into categories and subcategories.

        Uses ML-based classification to assign assets to standardized
        categories (BUILDING, MACHINERY, VEHICLE, etc.) and detailed
        subcategories based on asset descriptions.
        """
        svc = _get_service()
        try:
            result = svc.classify_assets(
                records=body.records,
                classification_level=body.classification_level,
            )
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "classify_assets failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 13. POST /compliance/check - Run compliance checks
    # ==================================================================

    @router.post("/compliance/check", status_code=200)
    async def check_compliance(
        body: ComplianceRequest,
    ) -> Dict[str, Any]:
        """Run regulatory compliance check against multiple frameworks.

        Evaluates the calculation against applicable Scope 3 Category 2
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
    # 14. GET /compliance/{check_id} - Get compliance result
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
    # 15. POST /uncertainty - Run uncertainty analysis
    # ==================================================================

    @router.post("/uncertainty", status_code=200)
    async def analyze_uncertainty(
        body: UncertaintyRequest,
    ) -> Dict[str, Any]:
        """Run uncertainty analysis on calculation results.

        Applies Monte Carlo simulation, pedigree matrix, IPCC Tier 2
        propagation, or bootstrap methods to quantify uncertainty in
        emission estimates.
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
    # 16. GET /aggregations - Get aggregated results
    # ==================================================================

    @router.get("/aggregations", status_code=200)
    async def get_aggregations(
        tenant_id: str = Query(
            ..., min_length=1,
            description="Tenant identifier",
        ),
        group_by: str = Query(
            default="asset_type",
            description="Grouping dimension (asset_type, category, year, "
            "supplier, location)",
        ),
        start_date: Optional[str] = Query(
            None, description="Start date filter (ISO-8601)",
        ),
        end_date: Optional[str] = Query(
            None, description="End date filter (ISO-8601)",
        ),
        asset_type: Optional[str] = Query(
            None, description="Filter by asset type",
        ),
    ) -> Dict[str, Any]:
        """Get aggregated emission results.

        Returns aggregated emissions by asset type, category, year,
        supplier, or location with summary statistics.
        """
        svc = _get_service()
        try:
            result = svc.get_aggregations(
                tenant_id=tenant_id,
                group_by=group_by,
                start_date=start_date,
                end_date=end_date,
                asset_type=asset_type,
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
    # 17. GET /hot-spots - Get hot-spot analysis
    # ==================================================================

    @router.get("/hot-spots", status_code=200)
    async def get_hot_spots(
        tenant_id: str = Query(
            ..., min_length=1,
            description="Tenant identifier",
        ),
        limit: int = Query(
            10, ge=1, le=100,
            description="Number of hot spots to return",
        ),
        dimension: str = Query(
            default="asset",
            description="Hot-spot dimension (asset, category, supplier, location)",
        ),
        start_date: Optional[str] = Query(
            None, description="Start date filter (ISO-8601)",
        ),
        end_date: Optional[str] = Query(
            None, description="End date filter (ISO-8601)",
        ),
    ) -> Dict[str, Any]:
        """Get hot-spot analysis.

        Identifies highest-emission assets, categories, suppliers, or
        locations with contribution percentages and reduction opportunities.
        """
        svc = _get_service()
        try:
            result = svc.get_hot_spots(
                tenant_id=tenant_id,
                limit=limit,
                dimension=dimension,
                start_date=start_date,
                end_date=end_date,
            )
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "get_hot_spots failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 18. POST /export - Export results
    # ==================================================================

    @router.post("/export", status_code=200)
    async def export_results(
        body: ExportRequest,
    ) -> Dict[str, Any]:
        """Export calculation results to various formats.

        Exports results as JSON, CSV, Excel, or GHG Protocol Corporate
        Value Chain (Scope 3) Accounting and Reporting Standard template
        for Category 2 (Capital Goods).
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
    # 19. GET /health - Health check
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

    # ==================================================================
    # 20. GET /stats - Engine statistics
    # ==================================================================

    @router.get("/stats", status_code=200)
    async def get_stats(
        tenant_id: Optional[str] = Query(
            None, description="Filter by tenant (admin only)",
        ),
        time_range: str = Query(
            default="24h",
            description="Time range (1h, 24h, 7d, 30d, 90d, 1y)",
        ),
    ) -> Dict[str, Any]:
        """Get engine statistics.

        Returns operational metrics including calculation count, total
        emissions processed, method distribution, average processing time,
        and error rate.
        """
        svc = _get_service()
        try:
            result = svc.get_stats(
                tenant_id=tenant_id,
                time_range=time_range,
            )
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "get_stats failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    return router


# ===================================================================
# Public API
# ===================================================================

__all__ = ["create_router"]
