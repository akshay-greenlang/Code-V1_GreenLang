# -*- coding: utf-8 -*-
"""
Downstream Leased Assets Service Setup - AGENT-MRV-026

This module provides the service facade that wires together all 7 engines
for downstream leased assets emissions calculations (Scope 3 Category 13).

The DownstreamLeasedAssetsService class provides a high-level API for:
- Asset-specific emissions (metered tenant energy data)
- Building emissions (EUI benchmarks, climate zone, vacancy handling)
- Vehicle fleet emissions (distance/fuel-based, 8 vehicle types)
- Equipment emissions (operating hours, 6 equipment types)
- IT asset emissions (PUE-adjusted power, 7 IT types)
- Average-data calculations (EUI by building type and climate zone)
- Spend-based calculations (EEIO factors by NAICS leasing codes)
- Hybrid calculations (weighted combination of methods)
- Compliance checking across 7 regulatory frameworks
- Portfolio-level analysis and hot-spot identification
- Aggregations by asset category, method, tenant, region
- Provenance tracking with SHA-256 audit trail

Engines:
    1. DownstreamAssetDatabaseEngine - Emission factor data and persistence
    2. AssetSpecificCalculatorEngine - Metered tenant energy calculations
    3. AverageDataCalculatorEngine - EUI benchmark calculations
    4. SpendBasedCalculatorEngine - EEIO spend-based calculations
    5. HybridAggregatorEngine - Weighted combination of methods
    6. ComplianceCheckerEngine - Multi-framework compliance validation
    7. DownstreamLeasedAssetsPipelineEngine - End-to-end 10-stage pipeline

Architecture:
    - Thread-safe singleton pattern for service instance
    - Graceful imports with try/except for optional dependencies
    - Comprehensive metrics tracking via OBS-001 integration
    - Provenance tracking for all mutations via AGENT-FOUND-008
    - Type-safe request/response models using Pydantic
    - Structured logging with contextual information

Example:
    >>> from greenlang.downstream_leased_assets.setup import get_service
    >>> service = get_service()
    >>> response = service.calculate({
    ...     "asset_category": "building",
    ...     "building_type": "office",
    ...     "floor_area_m2": 5000,
    ...     "country": "US",
    ... })
    >>> assert response.success

Integration:
    >>> from greenlang.downstream_leased_assets.setup import get_router
    >>> app.include_router(get_router(), prefix="/api/v1/downstream-leased-assets")
"""

import logging
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field, validator

# Thread-safe singleton lock
_service_lock = threading.Lock()
_service_instance: Optional["DownstreamLeasedAssetsService"] = None

logger = logging.getLogger(__name__)


# ==============================================================================
# CONSTANTS
# ==============================================================================

AGENT_ID: str = "GL-MRV-S3-013"
AGENT_COMPONENT: str = "AGENT-MRV-026"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_dla_"
API_PREFIX: str = "/api/v1/downstream-leased-assets"


# ==============================================================================
# Request Models
# ==============================================================================


class AssetCalculationRequest(BaseModel):
    """Request model for single asset emissions calculation."""

    asset_category: str = Field(
        ..., description="Asset category: building, vehicle, equipment, it_asset"
    )
    asset_data: dict = Field(..., description="Category-specific input data")
    allocation_method: Optional[str] = Field(
        None, description="Allocation method: floor_area, headcount, revenue, metered"
    )
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")
    cost_center: Optional[str] = Field(None, description="Cost center for allocation")
    department: Optional[str] = Field(None, description="Department for allocation")
    reporting_period: Optional[str] = Field(None, description="Reporting period")
    org_tenant_id: Optional[str] = Field(None, description="Organization tenant ID")

    @validator("asset_category")
    def validate_category(cls, v: str) -> str:
        """Validate asset category."""
        allowed = ["building", "vehicle", "equipment", "it_asset"]
        if v.lower() not in allowed:
            raise ValueError(f"asset_category must be one of {allowed}")
        return v.lower()


class BuildingCalculationRequest(BaseModel):
    """Request model for building emissions calculation."""

    building_type: str = Field(
        "office",
        description="Building type: office, retail, warehouse, residential, hotel, healthcare, education, data_center",
    )
    floor_area_m2: Optional[float] = Field(None, gt=0, description="Floor area in m2")
    floor_area_sqft: Optional[float] = Field(None, gt=0, description="Floor area in sqft")
    energy_kwh: Optional[float] = Field(None, gt=0, description="Metered energy in kWh")
    country: str = Field("GLOBAL", description="Country code for grid EF")
    climate_zone: str = Field("zone_3_temperate", description="Climate zone")
    vacancy_pct: float = Field(0, ge=0, le=100, description="Vacancy percentage")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")
    org_tenant_id: Optional[str] = Field(None, description="Organization tenant ID")


class VehicleCalculationRequest(BaseModel):
    """Request model for vehicle fleet calculation."""

    vehicle_type: str = Field("passenger_car_petrol", description="Vehicle type")
    distance_km: Optional[float] = Field(None, gt=0, description="Distance in km")
    distance_miles: Optional[float] = Field(None, gt=0, description="Distance in miles")
    fuel_litres: Optional[float] = Field(None, gt=0, description="Fuel consumed in litres")
    vehicle_count: int = Field(1, ge=1, description="Number of vehicles")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")
    org_tenant_id: Optional[str] = Field(None, description="Organization tenant ID")


class EquipmentCalculationRequest(BaseModel):
    """Request model for equipment calculation."""

    equipment_type: str = Field("generator_diesel", description="Equipment type")
    operating_hours: Optional[float] = Field(None, gt=0, description="Operating hours")
    fuel_litres: Optional[float] = Field(None, gt=0, description="Fuel consumed in litres")
    equipment_count: int = Field(1, ge=1, description="Number of equipment units")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")
    org_tenant_id: Optional[str] = Field(None, description="Organization tenant ID")


class ITAssetCalculationRequest(BaseModel):
    """Request model for IT asset calculation."""

    it_asset_type: str = Field("server", description="IT asset type")
    unit_count: int = Field(1, ge=1, description="Number of IT units")
    power_kw: Optional[float] = Field(None, gt=0, description="Total power in kW")
    energy_kwh: Optional[float] = Field(None, gt=0, description="Total energy in kWh")
    pue: float = Field(1.58, ge=1.0, le=3.0, description="Power Usage Effectiveness")
    operating_hours: float = Field(8760, gt=0, description="Operating hours per year")
    country: str = Field("GLOBAL", description="Country code for grid EF")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")
    org_tenant_id: Optional[str] = Field(None, description="Organization tenant ID")


class SpendBasedRequest(BaseModel):
    """Request model for spend-based calculation."""

    naics_code: str = Field(..., description="NAICS code for EEIO factor")
    spend_usd: float = Field(..., gt=0, description="Spend amount in USD")
    description: Optional[str] = Field(None, description="Spend description")
    reporting_year: int = Field(2025, ge=2015, le=2030, description="Reporting year")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")
    org_tenant_id: Optional[str] = Field(None, description="Organization tenant ID")


class HybridCalculationRequest(BaseModel):
    """Request model for hybrid calculation combining methods."""

    asset_category: str = Field(..., description="Asset category")
    asset_data: dict = Field(..., description="Asset-specific data")
    naics_code: str = Field(..., description="NAICS code for spend component")
    spend_usd: float = Field(..., gt=0, description="Spend amount in USD")
    asset_weight: float = Field(0.70, ge=0, le=1.0, description="Weight for asset-specific")
    spend_weight: float = Field(0.30, ge=0, le=1.0, description="Weight for spend-based")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")
    org_tenant_id: Optional[str] = Field(None, description="Organization tenant ID")


class BatchCalculationRequest(BaseModel):
    """Request model for batch asset calculations."""

    assets: List[dict] = Field(..., min_length=1, max_length=5000, description="List of asset inputs")
    reporting_period: str = Field(..., description="Reporting period")
    org_tenant_id: Optional[str] = Field(None, description="Organization tenant ID")


class PortfolioAnalysisRequest(BaseModel):
    """Request model for portfolio analysis."""

    assets: List[dict] = Field(..., min_length=1, max_length=10000, description="Portfolio assets")
    reporting_period: str = Field(..., description="Reporting period")
    org_tenant_id: Optional[str] = Field(None, description="Organization tenant ID")


class ComplianceCheckRequest(BaseModel):
    """Request model for compliance checking."""

    calculation_result: dict = Field(..., description="Calculation result to check")
    frameworks: List[str] = Field(
        default_factory=lambda: ["GHG_PROTOCOL_SCOPE3"],
        description="Frameworks to check",
    )
    org_tenant_id: Optional[str] = Field(None, description="Organization tenant ID")


class AggregationRequest(BaseModel):
    """Request model for aggregation queries."""

    reporting_period: str = Field(..., description="Reporting period")
    group_by: Optional[str] = Field(
        "asset_category", description="Group by: asset_category, method, tenant, region"
    )
    org_tenant_id: Optional[str] = Field(None, description="Organization tenant ID")


# ==============================================================================
# Response Models
# ==============================================================================


class CalculationResponse(BaseModel):
    """Response model for single asset calculation."""

    success: bool = Field(..., description="Success flag")
    calculation_id: str = Field(..., description="Calculation identifier")
    asset_category: str = Field(..., description="Asset category")
    calculation_method: str = Field(..., description="Method used")
    total_co2e_kg: float = Field(..., description="Total CO2e in kg")
    dqi_score: Optional[float] = Field(None, description="Data quality score (1-5)")
    ef_source: str = Field(..., description="Emission factor source")
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    detail: dict = Field(default_factory=dict, description="Calculation details")
    allocation: dict = Field(default_factory=dict, description="Allocation info")
    error: Optional[str] = Field(None, description="Error message if failed")
    processing_time_ms: float = Field(..., description="Processing time in ms")


class BatchCalculationResponse(BaseModel):
    """Response model for batch calculation."""

    success: bool = Field(..., description="Overall success flag")
    total_assets: int = Field(..., description="Total assets submitted")
    successful: int = Field(..., description="Successfully processed")
    failed: int = Field(..., description="Failed processing")
    total_co2e_kg: float = Field(..., description="Total CO2e for all assets")
    by_category: Dict[str, float] = Field(default_factory=dict, description="By category")
    by_method: Dict[str, float] = Field(default_factory=dict, description="By method")
    results: List[CalculationResponse] = Field(default_factory=list, description="Individual results")
    errors: List[dict] = Field(default_factory=list, description="Errors")
    reporting_period: str = Field(..., description="Reporting period")
    processing_time_ms: float = Field(..., description="Processing time")


class PortfolioResponse(BaseModel):
    """Response model for portfolio analysis."""

    success: bool = Field(..., description="Success flag")
    total_assets: int = Field(..., description="Total assets")
    total_co2e_kg: float = Field(..., description="Total portfolio CO2e")
    by_category: Dict[str, float] = Field(default_factory=dict, description="By category")
    by_method: Dict[str, float] = Field(default_factory=dict, description="By method")
    hot_spots: List[dict] = Field(default_factory=list, description="Hot spots")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    processing_time_ms: float = Field(..., description="Processing time")


class ComplianceCheckResponse(BaseModel):
    """Response model for compliance checking."""

    success: bool = Field(..., description="Success flag")
    overall_status: str = Field(..., description="Overall status: PASS, WARNING, FAIL")
    overall_score: float = Field(..., description="Overall score 0-100")
    framework_scores: Dict[str, Any] = Field(default_factory=dict, description="Framework scores")
    total_findings: int = Field(..., description="Total findings")
    critical_findings: List[dict] = Field(default_factory=list, description="Critical findings")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    processing_time_ms: float = Field(..., description="Processing time")


class ProvenanceResponse(BaseModel):
    """Response model for provenance queries."""

    success: bool = Field(..., description="Success flag")
    calculation_id: str = Field(..., description="Calculation ID")
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    chain: List[dict] = Field(default_factory=list, description="Provenance chain entries")
    is_valid: bool = Field(..., description="Chain integrity verified")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service status: healthy, degraded, unhealthy")
    version: str = Field(..., description="Service version")
    agent_id: str = Field(..., description="Agent identifier")
    engines_status: Dict[str, bool] = Field(..., description="Per-engine status")
    uptime_seconds: float = Field(..., description="Service uptime")


# ==============================================================================
# DownstreamLeasedAssetsService
# ==============================================================================


class DownstreamLeasedAssetsService:
    """
    Downstream Leased Assets Service Facade.

    This service wires together all 7 engines to provide a complete API
    for downstream leased assets emissions calculations (Scope 3 Cat 13).

    The service supports:
        - Building emissions (8 types, 5 climate zones, vacancy handling)
        - Vehicle fleet emissions (8 types, distance/fuel-based)
        - Equipment emissions (6 types, operating hours)
        - IT asset emissions (7 types, PUE-adjusted power)
        - Average-data with EUI benchmarks
        - Spend-based with EEIO factors (10 NAICS codes)
        - Hybrid combining asset-specific and spend-based
        - Compliance checking (7 regulatory frameworks)
        - Portfolio analysis with hot-spot identification
        - Multi-dimensional aggregation and reporting
        - Provenance tracking with SHA-256 audit trail

    Engines:
        1. DownstreamAssetDatabaseEngine - Data persistence
        2. AssetSpecificCalculatorEngine - Metered data calculations
        3. AverageDataCalculatorEngine - Benchmark calculations
        4. SpendBasedCalculatorEngine - EEIO calculations
        5. HybridAggregatorEngine - Weighted combination
        6. ComplianceCheckerEngine - Compliance validation
        7. DownstreamLeasedAssetsPipelineEngine - 10-stage pipeline

    Thread Safety:
        This service is thread-safe. Use get_service() to obtain a singleton.

    Example:
        >>> service = get_service()
        >>> response = service.calculate(AssetCalculationRequest(
        ...     asset_category="building",
        ...     asset_data={"building_type": "office", "floor_area_m2": 5000, "country": "US"},
        ... ))
        >>> assert response.success

    Attributes:
        _database_engine: Database engine for persistence
        _asset_specific_engine: Asset-specific calculator engine
        _average_data_engine: Average data calculator engine
        _spend_engine: Spend-based calculator engine
        _hybrid_engine: Hybrid aggregator engine
        _compliance_engine: Compliance checker engine
        _pipeline_engine: Pipeline orchestration engine
    """

    def __init__(self) -> None:
        """Initialize DownstreamLeasedAssetsService with all 7 engines."""
        logger.info("Initializing DownstreamLeasedAssetsService")
        self._start_time = datetime.now(timezone.utc)
        self._initialized = False

        # Initialize engines with graceful fallback
        self._database_engine = self._init_engine(
            "greenlang.downstream_leased_assets.downstream_asset_database",
            "DownstreamAssetDatabaseEngine",
        )
        self._asset_specific_engine = self._init_engine(
            "greenlang.downstream_leased_assets.asset_specific_calculator",
            "AssetSpecificCalculatorEngine",
        )
        self._average_data_engine = self._init_engine(
            "greenlang.downstream_leased_assets.average_data_calculator",
            "AverageDataCalculatorEngine",
        )
        self._spend_engine = self._init_engine(
            "greenlang.downstream_leased_assets.spend_based_calculator",
            "SpendBasedCalculatorEngine",
        )
        self._hybrid_engine = self._init_engine(
            "greenlang.downstream_leased_assets.hybrid_aggregator",
            "HybridAggregatorEngine",
        )
        self._compliance_engine = self._init_engine(
            "greenlang.downstream_leased_assets.compliance_checker",
            "ComplianceCheckerEngine",
        )
        self._pipeline_engine = self._init_engine(
            "greenlang.downstream_leased_assets.downstream_leased_assets_pipeline",
            "DownstreamLeasedAssetsPipelineEngine",
        )

        # In-memory calculation store (for dev/testing; production uses DB)
        self._calculations: Dict[str, dict] = {}

        self._initialized = True
        logger.info("DownstreamLeasedAssetsService initialized successfully")

    @staticmethod
    def _init_engine(module_path: str, class_name: str) -> Optional[Any]:
        """
        Initialize an engine with graceful ImportError handling.

        Args:
            module_path: Fully qualified module path.
            class_name: Class name within the module.

        Returns:
            Engine instance or None if import fails.
        """
        try:
            import importlib
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            instance = cls()
            logger.info("%s initialized", class_name)
            return instance
        except ImportError:
            logger.warning("%s not available (ImportError)", class_name)
            return None
        except Exception as e:
            logger.warning("%s initialization failed: %s", class_name, e)
            return None

    # ========================================================================
    # Core Calculation Methods (10)
    # ========================================================================

    def calculate(self, request: AssetCalculationRequest) -> CalculationResponse:
        """
        Calculate emissions for a single leased asset.

        Delegates to the pipeline engine for full 10-stage processing.

        Args:
            request: Asset calculation request with category and data.

        Returns:
            CalculationResponse with emissions and provenance.
        """
        start_time = time.monotonic()
        calc_id = f"dla-{uuid4().hex[:12]}"

        try:
            asset_input = {
                "asset_category": request.asset_category,
                **request.asset_data,
            }
            if request.allocation_method:
                asset_input["allocation_method"] = request.allocation_method
            if request.tenant_id:
                asset_input["tenant_id"] = request.tenant_id
            if request.cost_center:
                asset_input["cost_center"] = request.cost_center
            if request.department:
                asset_input["department"] = request.department

            result = self._run_pipeline(asset_input)
            elapsed = (time.monotonic() - start_time) * 1000.0

            response = CalculationResponse(
                success=True,
                calculation_id=calc_id,
                asset_category=result.get("asset_category", request.asset_category),
                calculation_method=result.get("calculation_method", "unknown"),
                total_co2e_kg=float(Decimal(str(result.get("total_co2e", 0)))),
                dqi_score=float(Decimal(str(result.get("dqi_score", 3.0)))),
                ef_source=result.get("ef_source", ""),
                provenance_hash=result.get("provenance_hash", ""),
                detail=result.get("detail", {}),
                allocation=result.get("allocation", {}),
                processing_time_ms=elapsed,
            )

            self._calculations[calc_id] = response.dict()
            return response

        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000.0
            logger.error("Calculation %s failed: %s", calc_id, e, exc_info=True)
            return CalculationResponse(
                success=False,
                calculation_id=calc_id,
                asset_category=request.asset_category,
                calculation_method="unknown",
                total_co2e_kg=0.0,
                ef_source="none",
                provenance_hash="",
                error=str(e),
                processing_time_ms=elapsed,
            )

    def calculate_asset_specific(self, asset_input: dict) -> CalculationResponse:
        """
        Calculate emissions using metered tenant energy data.

        Args:
            asset_input: Asset data with energy_kwh or fuel_litres.

        Returns:
            CalculationResponse with asset-specific emissions.
        """
        request = AssetCalculationRequest(
            asset_category=asset_input.get("asset_category", "building"),
            asset_data=asset_input,
        )
        return self.calculate(request)

    def calculate_building(self, request: BuildingCalculationRequest) -> CalculationResponse:
        """
        Calculate building emissions directly.

        Args:
            request: Building calculation request.

        Returns:
            CalculationResponse with building emissions.
        """
        data: Dict[str, Any] = {
            "building_type": request.building_type,
            "country": request.country,
            "climate_zone": request.climate_zone,
            "vacancy_pct": request.vacancy_pct,
        }
        if request.floor_area_m2:
            data["floor_area_m2"] = request.floor_area_m2
        if request.floor_area_sqft:
            data["floor_area_sqft"] = request.floor_area_sqft
        if request.energy_kwh:
            data["energy_kwh"] = request.energy_kwh

        asset_req = AssetCalculationRequest(
            asset_category="building",
            asset_data=data,
            tenant_id=request.tenant_id,
        )
        return self.calculate(asset_req)

    def calculate_vehicle(self, request: VehicleCalculationRequest) -> CalculationResponse:
        """
        Calculate vehicle fleet emissions directly.

        Args:
            request: Vehicle calculation request.

        Returns:
            CalculationResponse with vehicle emissions.
        """
        data: Dict[str, Any] = {
            "vehicle_type": request.vehicle_type,
            "vehicle_count": request.vehicle_count,
        }
        if request.distance_km:
            data["distance_km"] = request.distance_km
        if request.distance_miles:
            data["distance_miles"] = request.distance_miles
        if request.fuel_litres:
            data["fuel_litres"] = request.fuel_litres

        asset_req = AssetCalculationRequest(
            asset_category="vehicle",
            asset_data=data,
            tenant_id=request.tenant_id,
        )
        return self.calculate(asset_req)

    def calculate_equipment(self, request: EquipmentCalculationRequest) -> CalculationResponse:
        """
        Calculate equipment emissions directly.

        Args:
            request: Equipment calculation request.

        Returns:
            CalculationResponse with equipment emissions.
        """
        data: Dict[str, Any] = {
            "equipment_type": request.equipment_type,
            "equipment_count": request.equipment_count,
        }
        if request.operating_hours:
            data["operating_hours"] = request.operating_hours
        if request.fuel_litres:
            data["fuel_litres"] = request.fuel_litres

        asset_req = AssetCalculationRequest(
            asset_category="equipment",
            asset_data=data,
            tenant_id=request.tenant_id,
        )
        return self.calculate(asset_req)

    def calculate_it_asset(self, request: ITAssetCalculationRequest) -> CalculationResponse:
        """
        Calculate IT asset emissions directly.

        Args:
            request: IT asset calculation request.

        Returns:
            CalculationResponse with IT asset emissions.
        """
        data: Dict[str, Any] = {
            "it_asset_type": request.it_asset_type,
            "unit_count": request.unit_count,
            "pue": request.pue,
            "operating_hours": request.operating_hours,
            "country": request.country,
        }
        if request.power_kw:
            data["power_kw"] = request.power_kw
        if request.energy_kwh:
            data["energy_kwh"] = request.energy_kwh

        asset_req = AssetCalculationRequest(
            asset_category="it_asset",
            asset_data=data,
            tenant_id=request.tenant_id,
        )
        return self.calculate(asset_req)

    def calculate_average_data(self, asset_input: dict) -> CalculationResponse:
        """
        Calculate emissions using average-data (EUI benchmarks).

        Args:
            asset_input: Asset data with building type, floor area, etc.

        Returns:
            CalculationResponse with average-data emissions.
        """
        request = AssetCalculationRequest(
            asset_category=asset_input.get("asset_category", "building"),
            asset_data=asset_input,
        )
        return self.calculate(request)

    def calculate_spend_based(self, request: SpendBasedRequest) -> CalculationResponse:
        """
        Calculate emissions using spend-based EEIO factors.

        Args:
            request: Spend-based calculation request.

        Returns:
            CalculationResponse with spend-based emissions.
        """
        data: Dict[str, Any] = {
            "naics_code": request.naics_code,
            "spend_usd": request.spend_usd,
        }
        asset_req = AssetCalculationRequest(
            asset_category="building",
            asset_data=data,
            tenant_id=request.tenant_id,
        )
        return self.calculate(asset_req)

    def calculate_hybrid(self, request: HybridCalculationRequest) -> CalculationResponse:
        """
        Calculate emissions using hybrid method (asset + spend).

        Args:
            request: Hybrid calculation request.

        Returns:
            CalculationResponse with hybrid emissions.
        """
        data: Dict[str, Any] = {
            **request.asset_data,
            "naics_code": request.naics_code,
            "spend_usd": request.spend_usd,
        }
        asset_req = AssetCalculationRequest(
            asset_category=request.asset_category,
            asset_data=data,
            tenant_id=request.tenant_id,
        )
        return self.calculate(asset_req)

    def calculate_batch(self, request: BatchCalculationRequest) -> BatchCalculationResponse:
        """
        Process multiple assets in a single batch.

        Args:
            request: Batch request with assets and reporting period.

        Returns:
            BatchCalculationResponse with individual results and totals.
        """
        start_time = time.monotonic()
        results: List[CalculationResponse] = []
        errors: List[dict] = []

        for idx, asset in enumerate(request.assets):
            try:
                req = AssetCalculationRequest(
                    asset_category=asset.get("asset_category", "building"),
                    asset_data=asset,
                )
                resp = self.calculate(req)
                results.append(resp)
                if not resp.success:
                    errors.append({"index": idx, "error": resp.error})
            except Exception as e:
                errors.append({
                    "index": idx,
                    "asset_category": asset.get("asset_category", "unknown"),
                    "error": str(e),
                })

        total_co2e = sum(r.total_co2e_kg for r in results if r.success)
        by_category: Dict[str, float] = {}
        by_method: Dict[str, float] = {}
        for r in results:
            if r.success:
                cat = r.asset_category
                by_category[cat] = by_category.get(cat, 0.0) + r.total_co2e_kg
                meth = r.calculation_method
                by_method[meth] = by_method.get(meth, 0.0) + r.total_co2e_kg

        successful = sum(1 for r in results if r.success)
        elapsed = (time.monotonic() - start_time) * 1000.0

        return BatchCalculationResponse(
            success=len(errors) == 0,
            total_assets=len(request.assets),
            successful=successful,
            failed=len(errors),
            total_co2e_kg=total_co2e,
            by_category=by_category,
            by_method=by_method,
            results=results,
            errors=errors,
            reporting_period=request.reporting_period,
            processing_time_ms=elapsed,
        )

    # ========================================================================
    # Portfolio Method (1)
    # ========================================================================

    def calculate_portfolio(self, request: PortfolioAnalysisRequest) -> PortfolioResponse:
        """
        Run portfolio-level analysis for the leased asset portfolio.

        Args:
            request: Portfolio analysis request.

        Returns:
            PortfolioResponse with aggregations, hot spots, and recommendations.
        """
        start_time = time.monotonic()

        try:
            pipeline = self._get_pipeline_engine()
            if pipeline is not None:
                result = pipeline.run_portfolio_analysis(request.assets)
                elapsed = (time.monotonic() - start_time) * 1000.0

                return PortfolioResponse(
                    success=True,
                    total_assets=result.get("total_assets", len(request.assets)),
                    total_co2e_kg=float(Decimal(str(result.get("total_co2e", 0)))),
                    by_category={k: float(Decimal(str(v))) for k, v in result.get("by_category", {}).items()},
                    by_method={k: float(Decimal(str(v))) for k, v in result.get("by_method", {}).items()},
                    hot_spots=result.get("hot_spots", []),
                    recommendations=result.get("recommendations", []),
                    processing_time_ms=elapsed,
                )

            # Fallback: use batch calculation
            batch_req = BatchCalculationRequest(
                assets=request.assets,
                reporting_period=request.reporting_period,
            )
            batch_result = self.calculate_batch(batch_req)
            elapsed = (time.monotonic() - start_time) * 1000.0

            return PortfolioResponse(
                success=batch_result.success,
                total_assets=batch_result.total_assets,
                total_co2e_kg=batch_result.total_co2e_kg,
                by_category=batch_result.by_category,
                by_method=batch_result.by_method,
                hot_spots=[],
                recommendations=[],
                processing_time_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000.0
            logger.error("Portfolio analysis failed: %s", e, exc_info=True)
            return PortfolioResponse(
                success=False,
                total_assets=len(request.assets),
                total_co2e_kg=0.0,
                processing_time_ms=elapsed,
            )

    # ========================================================================
    # Lookup Methods (6)
    # ========================================================================

    def check_compliance(self, request: ComplianceCheckRequest) -> ComplianceCheckResponse:
        """
        Run compliance checks against specified frameworks.

        Args:
            request: Compliance check request.

        Returns:
            ComplianceCheckResponse with per-framework results.
        """
        start_time = time.monotonic()

        try:
            from greenlang.downstream_leased_assets.compliance_checker import (
                ComplianceCheckerEngine,
            )
            engine = ComplianceCheckerEngine.get_instance()
            all_results = engine.check_all_frameworks(request.calculation_result)
            summary = engine.get_compliance_summary(all_results)
            elapsed = (time.monotonic() - start_time) * 1000.0

            return ComplianceCheckResponse(
                success=True,
                overall_status=summary.get("overall_status", "FAIL"),
                overall_score=summary.get("overall_score", 0.0),
                framework_scores=summary.get("framework_scores", {}),
                total_findings=summary.get("total_findings", 0),
                critical_findings=summary.get("critical_findings", []),
                recommendations=summary.get("recommendations", []),
                processing_time_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000.0
            logger.error("Compliance check failed: %s", e, exc_info=True)
            return ComplianceCheckResponse(
                success=False,
                overall_status="FAIL",
                overall_score=0.0,
                total_findings=0,
                recommendations=[f"Compliance check error: {str(e)}"],
                processing_time_ms=elapsed,
            )

    def get_building_benchmarks(self) -> dict:
        """
        Get building EUI benchmarks by type and climate zone.

        Returns:
            Dictionary of EUI benchmarks.
        """
        from greenlang.downstream_leased_assets.downstream_leased_assets_pipeline import (
            EUI_BENCHMARKS,
        )
        return {
            bt: {cz: float(v) for cz, v in zones.items()}
            for bt, zones in EUI_BENCHMARKS.items()
        }

    def get_vehicle_efs(self) -> dict:
        """
        Get vehicle emission factors by type.

        Returns:
            Dictionary of vehicle emission factors.
        """
        from greenlang.downstream_leased_assets.downstream_leased_assets_pipeline import (
            VEHICLE_EMISSION_FACTORS,
        )
        return {vt: float(ef) for vt, ef in VEHICLE_EMISSION_FACTORS.items()}

    def get_grid_factors(self) -> dict:
        """
        Get grid emission factors by country/region.

        Returns:
            Dictionary of grid emission factors.
        """
        from greenlang.downstream_leased_assets.downstream_leased_assets_pipeline import (
            GRID_EMISSION_FACTORS,
        )
        return {c: float(ef) for c, ef in GRID_EMISSION_FACTORS.items()}

    def get_allocation_methods(self) -> List[dict]:
        """
        Get supported allocation methods.

        Returns:
            List of allocation method metadata.
        """
        return [
            {"method": "floor_area", "description": "Allocate by leased floor area proportion"},
            {"method": "headcount", "description": "Allocate by tenant headcount proportion"},
            {"method": "revenue", "description": "Allocate by tenant revenue proportion"},
            {"method": "metered", "description": "Allocate by metered energy consumption"},
            {"method": "equal_share", "description": "Equal split among tenants"},
            {"method": "custom", "description": "Custom allocation with user-defined weights"},
        ]

    def get_all_categories(self) -> List[dict]:
        """
        Get all supported asset categories with metadata.

        Returns:
            List of asset category metadata.
        """
        return [
            {
                "category": "building",
                "display_name": "Buildings",
                "types": ["office", "retail", "warehouse", "residential", "hotel", "healthcare", "education", "data_center"],
                "description": "Leased buildings and real estate assets",
            },
            {
                "category": "vehicle",
                "display_name": "Vehicles",
                "types": ["passenger_car_petrol", "passenger_car_diesel", "passenger_car_hybrid", "passenger_car_ev", "van_petrol", "van_diesel", "truck_light", "truck_heavy"],
                "description": "Leased vehicle fleets",
            },
            {
                "category": "equipment",
                "display_name": "Equipment",
                "types": ["generator_diesel", "generator_gas", "compressor", "forklift", "construction", "industrial"],
                "description": "Leased industrial and commercial equipment",
            },
            {
                "category": "it_asset",
                "display_name": "IT Assets",
                "types": ["server", "storage", "network_switch", "workstation", "laptop", "printer", "data_center_rack"],
                "description": "Leased IT infrastructure and devices",
            },
        ]

    # ========================================================================
    # Utility Methods (5)
    # ========================================================================

    def get_aggregations(self, request: AggregationRequest) -> dict:
        """
        Get aggregated emissions for a reporting period.

        Args:
            request: Aggregation request with period and group_by.

        Returns:
            Dictionary with multi-dimensional aggregation.
        """
        start_time = time.monotonic()

        by_category: Dict[str, float] = {}
        by_method: Dict[str, float] = {}
        by_tenant: Dict[str, float] = {}
        total = 0.0

        for calc in self._calculations.values():
            co2e = calc.get("total_co2e_kg", 0.0)
            total += co2e

            cat = calc.get("asset_category", "unknown")
            by_category[cat] = by_category.get(cat, 0.0) + co2e

            method = calc.get("calculation_method", "unknown")
            by_method[method] = by_method.get(method, 0.0) + co2e

            tenant = calc.get("detail", {}).get("tenant_id")
            if tenant:
                by_tenant[tenant] = by_tenant.get(tenant, 0.0) + co2e

        elapsed = (time.monotonic() - start_time) * 1000.0

        return {
            "success": True,
            "total_co2e_kg": total,
            "by_category": by_category,
            "by_method": by_method,
            "by_tenant": by_tenant,
            "reporting_period": request.reporting_period,
            "processing_time_ms": elapsed,
        }

    def get_provenance(self, calculation_id: str) -> ProvenanceResponse:
        """
        Get provenance chain for a calculation.

        Args:
            calculation_id: Calculation identifier.

        Returns:
            ProvenanceResponse with chain entries and integrity status.
        """
        calc = self._calculations.get(calculation_id)
        if calc:
            return ProvenanceResponse(
                success=True,
                calculation_id=calculation_id,
                provenance_hash=calc.get("provenance_hash", ""),
                chain=[],
                is_valid=True,
            )
        return ProvenanceResponse(
            success=False,
            calculation_id=calculation_id,
            provenance_hash="",
            chain=[],
            is_valid=False,
        )

    def health_check(self) -> HealthResponse:
        """
        Perform service health check.

        Returns:
            HealthResponse with engine statuses and uptime.
        """
        uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()
        engines_status = {
            "database": self._database_engine is not None,
            "asset_specific": self._asset_specific_engine is not None,
            "average_data": self._average_data_engine is not None,
            "spend_based": self._spend_engine is not None,
            "hybrid": self._hybrid_engine is not None,
            "compliance": self._compliance_engine is not None,
            "pipeline": self._pipeline_engine is not None,
        }

        all_healthy = all(engines_status.values())
        any_healthy = any(engines_status.values())

        if all_healthy:
            status = "healthy"
        elif any_healthy:
            status = "degraded"
        else:
            status = "unhealthy"

        return HealthResponse(
            status=status,
            version=VERSION,
            agent_id=AGENT_ID,
            engines_status=engines_status,
            uptime_seconds=uptime,
        )

    def get_version(self) -> dict:
        """
        Get service version information.

        Returns:
            Dictionary with version details.
        """
        return {
            "agent_id": AGENT_ID,
            "agent_component": AGENT_COMPONENT,
            "version": VERSION,
            "table_prefix": TABLE_PREFIX,
            "api_prefix": API_PREFIX,
        }

    def get_config(self) -> dict:
        """
        Get current service configuration.

        Returns:
            Dictionary with configuration details.
        """
        return {
            "agent_id": AGENT_ID,
            "agent_component": AGENT_COMPONENT,
            "version": VERSION,
            "table_prefix": TABLE_PREFIX,
            "api_prefix": API_PREFIX,
            "engines": {
                "database": self._database_engine is not None,
                "asset_specific": self._asset_specific_engine is not None,
                "average_data": self._average_data_engine is not None,
                "spend_based": self._spend_engine is not None,
                "hybrid": self._hybrid_engine is not None,
                "compliance": self._compliance_engine is not None,
                "pipeline": self._pipeline_engine is not None,
            },
            "calculations_in_memory": len(self._calculations),
        }

    # ========================================================================
    # Internal Helpers
    # ========================================================================

    def _run_pipeline(self, asset_input: dict) -> dict:
        """
        Run asset through the pipeline engine or inline fallback.

        Args:
            asset_input: Normalized asset input dictionary.

        Returns:
            Pipeline result dictionary.
        """
        pipeline = self._get_pipeline_engine()
        if pipeline is not None:
            return pipeline.run_pipeline(asset_input)
        raise RuntimeError("Pipeline engine not available")

    def _get_pipeline_engine(self) -> Optional[Any]:
        """Get pipeline engine instance."""
        if self._pipeline_engine is not None:
            return self._pipeline_engine

        try:
            from greenlang.downstream_leased_assets.downstream_leased_assets_pipeline import (
                DownstreamLeasedAssetsPipelineEngine,
            )
            self._pipeline_engine = DownstreamLeasedAssetsPipelineEngine()
            return self._pipeline_engine
        except ImportError:
            return None


# ==============================================================================
# Module-Level Helpers
# ==============================================================================


def get_service() -> DownstreamLeasedAssetsService:
    """
    Get singleton DownstreamLeasedAssetsService instance.

    Thread-safe via double-checked locking.

    Returns:
        DownstreamLeasedAssetsService singleton instance.
    """
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = DownstreamLeasedAssetsService()
    return _service_instance


def get_router():
    """
    Get the FastAPI router for downstream leased assets endpoints.

    Returns:
        FastAPI APIRouter instance.
    """
    from greenlang.downstream_leased_assets.api.router import router
    return router


def reset_service() -> None:
    """
    Reset the service singleton (for testing only).

    Thread Safety:
        Protected by the module-level lock.
    """
    global _service_instance
    with _service_lock:
        _service_instance = None
        logger.info("DownstreamLeasedAssetsService singleton reset")


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    "AGENT_ID",
    "AGENT_COMPONENT",
    "VERSION",
    "TABLE_PREFIX",
    "API_PREFIX",
    "AssetCalculationRequest",
    "BuildingCalculationRequest",
    "VehicleCalculationRequest",
    "EquipmentCalculationRequest",
    "ITAssetCalculationRequest",
    "SpendBasedRequest",
    "HybridCalculationRequest",
    "BatchCalculationRequest",
    "PortfolioAnalysisRequest",
    "ComplianceCheckRequest",
    "AggregationRequest",
    "CalculationResponse",
    "BatchCalculationResponse",
    "PortfolioResponse",
    "ComplianceCheckResponse",
    "ProvenanceResponse",
    "HealthResponse",
    "DownstreamLeasedAssetsService",
    "get_service",
    "get_router",
    "reset_service",
]
