"""
Upstream Leased Assets Agent API Router - AGENT-MRV-021

This module implements the FastAPI router for upstream leased assets emissions
calculations following GHG Protocol Scope 3 Category 8 requirements.

Provides 22 REST endpoints for:
- Emissions calculations (full pipeline, batch, building, vehicle, equipment,
  IT asset, lessor-specific, spend-based, portfolio-level)
- Calculation CRUD (get, list, delete)
- Emission factor lookup by asset type
- Building energy benchmarks by type and climate zone
- Grid emission factors by country
- Lease classification (operating vs finance)
- Compliance checking across 7 regulatory frameworks
- Uncertainty analysis (Monte Carlo, analytical, IPCC Tier 2)
- Aggregations by period with asset type and department breakdowns
- Portfolio analysis for hot-spot identification
- Provenance tracking with SHA-256 chain verification
- Health monitoring

Follows GreenLang's zero-hallucination principle with deterministic calculations.
All numeric outputs use deterministic formulas; no LLM calls in the calculation path.

Agent ID: GL-MRV-S3-008
Package: greenlang.upstream_leased_assets
API Prefix: /api/v1/upstream-leased-assets
DB Migration: V072
Metrics Prefix: gl_ula_
Table Prefix: gl_ula_

Asset categories (4):
    Buildings, Vehicles, Equipment, IT Assets

Calculation methods (4):
    Asset-specific, Lessor-specific, Average-data, Spend-based (EEIO)

Allocation methods (4):
    Floor area, Headcount, Lease term, Equal

Regulatory frameworks (7):
    GHG Protocol, ISO 14064, CSRD ESRS E1, CDP, SBTi, SB 253, GRI 305

Example:
    >>> from fastapi import FastAPI
    >>> from greenlang.upstream_leased_assets.api.router import router
    >>> app = FastAPI()
    >>> app.include_router(router)
"""

from fastapi import APIRouter, HTTPException, Query, Path, Depends, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from decimal import Decimal
import logging
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

# Router configuration
router = APIRouter(
    prefix="/api/v1/upstream-leased-assets",
    tags=["upstream-leased-assets"],
    responses={404: {"description": "Not found"}},
)


# ============================================================================
# SERVICE DEPENDENCY
# ============================================================================


_service_instance = None


def get_service():
    """
    Get or create UpstreamLeasedAssetsService singleton instance.

    Uses lazy initialization to avoid circular imports and ensure the
    service is only created when first needed. The service wires together
    all 7 engines (database, building calculator, vehicle fleet calculator,
    equipment calculator, IT assets calculator, compliance checker, pipeline).

    Returns:
        UpstreamLeasedAssetsService instance

    Raises:
        HTTPException: If service initialization fails (503)
    """
    global _service_instance

    if _service_instance is None:
        try:
            from greenlang.upstream_leased_assets.setup import UpstreamLeasedAssetsService
            _service_instance = UpstreamLeasedAssetsService()
            logger.info("UpstreamLeasedAssetsService initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize UpstreamLeasedAssetsService: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service initialization failed",
            )

    return _service_instance


# ============================================================================
# REQUEST MODELS
# ============================================================================


class BuildingCalculateRequest(BaseModel):
    """
    Request model for building emissions calculation.

    Calculates emissions from leased building space using energy consumption
    data (electricity, gas, heating, cooling) with country-specific grid
    emission factors and allocation by leased floor area fraction.

    Attributes:
        building_type: Type of building (office, retail, warehouse, data_center, etc.)
        floor_area_sqm: Total leased floor area in square metres
        electricity_kwh: Annual electricity consumption in kWh
        gas_kwh: Annual natural gas consumption in kWh
        heating_kwh: Annual heating energy consumption in kWh
        cooling_kwh: Annual cooling energy consumption in kWh
        country_code: ISO 3166-1 alpha-2 country code for grid EF
        allocation_factor: Fraction of building allocated to lessee (0.0-1.0)
        lease_months: Number of months in the reporting period lease is active
        climate_zone: Climate zone for benchmark comparison
    """

    building_type: str = Field(
        ...,
        description=(
            "Building type (office, retail, warehouse, data_center, "
            "industrial, mixed_use, laboratory, hospital)"
        ),
    )
    floor_area_sqm: float = Field(
        ...,
        gt=0,
        le=10000000,
        description="Total leased floor area in square metres",
    )
    electricity_kwh: float = Field(
        0.0,
        ge=0,
        description="Annual electricity consumption in kWh",
    )
    gas_kwh: float = Field(
        0.0,
        ge=0,
        description="Annual natural gas consumption in kWh",
    )
    heating_kwh: float = Field(
        0.0,
        ge=0,
        description="Annual heating energy consumption in kWh",
    )
    cooling_kwh: float = Field(
        0.0,
        ge=0,
        description="Annual cooling energy consumption in kWh",
    )
    country_code: str = Field(
        "US",
        min_length=2,
        max_length=6,
        description="ISO 3166-1 alpha-2 country code or 'GLOBAL'",
    )
    allocation_factor: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Fraction of building allocated to lessee (0.0-1.0)",
    )
    lease_months: int = Field(
        12,
        ge=1,
        le=12,
        description="Number of months lease is active in reporting period",
    )
    climate_zone: Optional[str] = Field(
        None,
        description=(
            "Climate zone (hot_humid, hot_dry, warm_humid, warm_dry, "
            "mixed_humid, mixed_dry, cool_humid, cool_dry, cold, subarctic)"
        ),
    )


class VehicleCalculateRequest(BaseModel):
    """
    Request model for vehicle fleet emissions calculation.

    Calculates emissions from leased vehicle fleets using fuel-based or
    distance-based methods with DEFRA 2024 emission factors.

    Attributes:
        vehicle_type: Type of vehicle (car_small, car_medium, car_large, etc.)
        fuel_type: Fuel type (petrol, diesel, lpg, cng, bev, hybrid, phev)
        annual_km: Annual distance driven per vehicle in kilometres
        fuel_litres: Annual fuel consumed per vehicle in litres
        count: Number of leased vehicles of this type
        country_code: ISO 3166-1 alpha-2 country code
        vehicle_age: Average vehicle age in years (for age-adjusted EF)
    """

    vehicle_type: str = Field(
        ...,
        description=(
            "Vehicle type (car_small, car_medium, car_large, suv, "
            "van_small, van_medium, van_large, truck_rigid, "
            "truck_articulated, motorcycle, bus)"
        ),
    )
    fuel_type: str = Field(
        "diesel",
        description="Fuel type (petrol, diesel, lpg, cng, bev, hybrid, phev)",
    )
    annual_km: Optional[float] = Field(
        None,
        gt=0,
        description="Annual distance per vehicle in km (distance-based method)",
    )
    fuel_litres: Optional[float] = Field(
        None,
        gt=0,
        description="Annual fuel per vehicle in litres (fuel-based method)",
    )
    count: int = Field(
        1,
        ge=1,
        le=100000,
        description="Number of leased vehicles of this type",
    )
    country_code: str = Field(
        "US",
        min_length=2,
        max_length=6,
        description="ISO 3166-1 alpha-2 country code",
    )
    vehicle_age: Optional[int] = Field(
        None,
        ge=0,
        le=30,
        description="Average vehicle age in years for age-adjusted EF",
    )


class EquipmentCalculateRequest(BaseModel):
    """
    Request model for equipment emissions calculation.

    Calculates emissions from leased industrial, construction, or
    material handling equipment based on power rating, operating hours,
    and load factor with fuel-specific emission factors.

    Attributes:
        equipment_type: Type of equipment (generator, compressor, etc.)
        power_kw: Equipment rated power in kilowatts
        operating_hours: Annual operating hours
        load_factor: Average load factor (0.0-1.0)
        fuel_type: Fuel type (diesel, petrol, natural_gas, electric, propane)
        fuel_litres: Annual fuel consumed in litres (optional override)
        count: Number of leased equipment units
        country_code: ISO 3166-1 alpha-2 country code
    """

    equipment_type: str = Field(
        ...,
        description=(
            "Equipment type (generator, compressor, chiller, boiler, "
            "forklift, crane, excavator, pump, hvac_unit, conveyor)"
        ),
    )
    power_kw: float = Field(
        ...,
        gt=0,
        le=100000,
        description="Equipment rated power in kilowatts",
    )
    operating_hours: float = Field(
        ...,
        gt=0,
        le=8760,
        description="Annual operating hours",
    )
    load_factor: float = Field(
        0.6,
        gt=0.0,
        le=1.0,
        description="Average load factor (fraction of rated power)",
    )
    fuel_type: str = Field(
        "diesel",
        description="Fuel type (diesel, petrol, natural_gas, electric, propane)",
    )
    fuel_litres: Optional[float] = Field(
        None,
        gt=0,
        description="Annual fuel in litres (overrides calculated from kW*hours*load)",
    )
    count: int = Field(
        1,
        ge=1,
        le=10000,
        description="Number of leased equipment units",
    )
    country_code: str = Field(
        "US",
        min_length=2,
        max_length=6,
        description="ISO 3166-1 alpha-2 country code",
    )


class ITAssetCalculateRequest(BaseModel):
    """
    Request model for IT asset emissions calculation.

    Calculates emissions from leased IT assets (servers, storage, networking,
    desktops) using power draw, PUE, utilization, and grid emission factors.

    Attributes:
        it_type: IT asset type (server, storage, network_switch, desktop, etc.)
        power_kw: IT asset rated power in kilowatts
        pue: Power Usage Effectiveness for data centre overhead
        utilization: Average utilization factor (0.0-1.0)
        operating_hours: Annual operating hours
        count: Number of leased IT assets
        country_code: ISO 3166-1 alpha-2 country code
    """

    it_type: str = Field(
        ...,
        description=(
            "IT asset type (server_rack, server_blade, server_tower, "
            "storage_san, storage_nas, network_switch, network_router, "
            "desktop, laptop, monitor, printer, ups)"
        ),
    )
    power_kw: float = Field(
        ...,
        gt=0,
        le=5000,
        description="IT asset rated power in kilowatts",
    )
    pue: float = Field(
        1.58,
        ge=1.0,
        le=3.0,
        description="Power Usage Effectiveness (typical 1.1-2.5)",
    )
    utilization: float = Field(
        0.5,
        gt=0.0,
        le=1.0,
        description="Average utilization factor (fraction of rated power)",
    )
    operating_hours: float = Field(
        8760,
        gt=0,
        le=8760,
        description="Annual operating hours (default 24x365 for always-on IT)",
    )
    count: int = Field(
        1,
        ge=1,
        le=100000,
        description="Number of leased IT assets",
    )
    country_code: str = Field(
        "US",
        min_length=2,
        max_length=6,
        description="ISO 3166-1 alpha-2 country code",
    )


class LessorCalculateRequest(BaseModel):
    """
    Request model for lessor-specific emissions calculation.

    Uses emissions data reported directly by the lessor, allocated to
    the lessee based on lease terms and area/headcount fraction.

    Attributes:
        lessor_name: Name of the lessor organization
        reported_co2e_kg: Total CO2e reported by the lessor in kg
        methodology: Methodology used by lessor for their reporting
        allocation_factor: Fraction allocated to the lessee (0.0-1.0)
        lease_months: Months of active lease in reporting period
    """

    lessor_name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Name of the lessor organization",
    )
    reported_co2e_kg: float = Field(
        ...,
        gt=0,
        description="Total CO2e reported by the lessor in kg",
    )
    methodology: str = Field(
        "ghg_protocol",
        description=(
            "Lessor's reporting methodology (ghg_protocol, iso_14064, "
            "csrd_esrs, proprietary, estimated)"
        ),
    )
    allocation_factor: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Fraction allocated to the lessee (0.0-1.0)",
    )
    lease_months: int = Field(
        12,
        ge=1,
        le=12,
        description="Months of active lease in reporting period",
    )


class SpendCalculateRequest(BaseModel):
    """
    Request model for spend-based emissions calculation.

    Uses EEIO (Environmentally Extended Input-Output) factors with
    CPI deflation and currency conversion for lease expenditure
    categories including equipment, real estate, and vehicle leases.

    Attributes:
        naics_code: NAICS industry code for EEIO factor selection
        amount: Spend amount in the specified currency
        currency: ISO 4217 currency code
        reporting_year: Year for CPI deflation adjustment
    """

    naics_code: str = Field(
        ...,
        description=(
            "NAICS code for EEIO factor selection (531120 real estate, "
            "532100 auto rental, 532400 equipment rental, 518210 data centers, "
            "531210 offices, 493110 warehousing)"
        ),
    )
    amount: float = Field(
        ...,
        gt=0,
        description="Spend amount in the specified currency",
    )
    currency: str = Field(
        "USD",
        min_length=3,
        max_length=3,
        description="ISO 4217 currency code (USD, EUR, GBP, CAD, AUD, etc.)",
    )
    reporting_year: int = Field(
        2024,
        ge=2000,
        le=2100,
        description="Reporting year for CPI deflation to base year (2021 USD)",
    )


class PortfolioCalculateRequest(BaseModel):
    """
    Request model for full portfolio emissions calculation.

    Processes all leased assets across buildings, vehicles, equipment,
    and IT assets in a single request with cross-category aggregation,
    double-counting prevention, and portfolio-level compliance checking.

    Attributes:
        buildings: List of building asset data dictionaries
        vehicles: List of vehicle fleet data dictionaries
        equipment: List of equipment data dictionaries
        it_assets: List of IT asset data dictionaries
        reporting_year: Reporting year
        organization_id: Organization identifier for multi-tenant
    """

    buildings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of building asset data dictionaries",
    )
    vehicles: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of vehicle fleet data dictionaries",
    )
    equipment: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of equipment data dictionaries",
    )
    it_assets: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of IT asset data dictionaries",
    )
    reporting_year: int = Field(
        2024,
        ge=2000,
        le=2100,
        description="Reporting year",
    )
    organization_id: Optional[str] = Field(
        None,
        description="Organization identifier for multi-tenant isolation",
    )


class BatchCalculateRequest(BaseModel):
    """
    Request model for batch emissions calculations.

    Processes multiple heterogeneous asset calculation requests in a
    single batch with per-item error isolation.

    Attributes:
        items: List of calculation request dictionaries (each must include 'asset_type')
        max_items: Maximum items to process (safety limit)
    """

    items: List[Dict[str, Any]] = Field(
        ...,
        min_items=1,
        max_items=1000,
        description=(
            "List of calculation request dicts. Each must include 'asset_type' "
            "(building, vehicle, equipment, it_asset, lessor, spend)"
        ),
    )
    max_items: int = Field(
        1000,
        ge=1,
        le=10000,
        description="Maximum items to process in this batch",
    )


class ComplianceCheckRequest(BaseModel):
    """
    Request model for multi-framework compliance checking.

    Checks upstream leased assets calculation results against selected
    regulatory frameworks for completeness, boundary correctness,
    allocation method disclosure, lease classification, and data quality.

    Attributes:
        frameworks: List of framework identifiers to check against
        total_co2e: Total CO2e to validate in kg
        method_used: Calculation method used
        reporting_period: Reporting period identifier
    """

    frameworks: List[str] = Field(
        ...,
        min_items=1,
        description=(
            "Frameworks to check (ghg_protocol, iso_14064, csrd_esrs, "
            "cdp, sbti, sb_253, gri)"
        ),
    )
    total_co2e: float = Field(
        ...,
        ge=0,
        description="Total CO2e emissions to validate in kg",
    )
    method_used: str = Field(
        "asset_specific",
        description=(
            "Calculation method used (asset_specific, lessor_specific, "
            "average_data, spend_based)"
        ),
    )
    reporting_period: str = Field(
        ...,
        description="Reporting period identifier (e.g. '2024', '2024-Q1')",
    )


class UncertaintyRequest(BaseModel):
    """
    Request model for uncertainty analysis.

    Supports Monte Carlo simulation, analytical error propagation,
    and IPCC Tier 2 default uncertainty ranges. Upstream leased assets
    uncertainty is typically moderate to high depending on data source.

    Attributes:
        method: Uncertainty analysis method
        iterations: Monte Carlo iterations (if applicable)
        confidence_level: Confidence interval level (0.90, 0.95, 0.99)
        total_co2e: Total CO2e to analyze in kg
    """

    method: str = Field(
        "monte_carlo",
        description="Uncertainty method (monte_carlo, analytical, ipcc_tier_2)",
    )
    iterations: int = Field(
        10000,
        ge=1000,
        le=100000,
        description="Monte Carlo iterations (ignored for analytical/ipcc_tier_2)",
    )
    confidence_level: float = Field(
        0.95,
        ge=0.80,
        le=0.99,
        description="Confidence interval level",
    )
    total_co2e: float = Field(
        ...,
        ge=0,
        description="Total CO2e to analyze for uncertainty in kg",
    )


class PortfolioAnalyzeRequest(BaseModel):
    """
    Request model for portfolio hot-spot analysis.

    Identifies top emission sources across the leased asset portfolio
    with Pareto-based reduction opportunity ranking.

    Attributes:
        buildings: List of building calculation result dicts
        vehicles: List of vehicle calculation result dicts
        equipment: List of equipment calculation result dicts
        it_assets: List of IT asset calculation result dicts
    """

    buildings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Building calculation results to analyze",
    )
    vehicles: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Vehicle calculation results to analyze",
    )
    equipment: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Equipment calculation results to analyze",
    )
    it_assets: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="IT asset calculation results to analyze",
    )


class AllocationRequest(BaseModel):
    """
    Request model for emissions allocation calculation.

    Allocates total building or facility emissions to the lessee
    based on floor area, headcount, lease term, or equal allocation.

    Attributes:
        total_co2e: Total facility CO2e in kg before allocation
        method: Allocation method
        leased_area: Leased floor area in sqm (for area-based)
        total_area: Total building floor area in sqm (for area-based)
        headcount_lessee: Lessee headcount (for headcount-based)
        headcount_total: Total building headcount (for headcount-based)
    """

    total_co2e: float = Field(
        ...,
        gt=0,
        description="Total facility CO2e in kg before allocation",
    )
    method: str = Field(
        "floor_area",
        description="Allocation method (floor_area, headcount, lease_term, equal)",
    )
    leased_area: Optional[float] = Field(
        None,
        gt=0,
        description="Leased floor area in sqm (required for floor_area method)",
    )
    total_area: Optional[float] = Field(
        None,
        gt=0,
        description="Total building floor area in sqm (required for floor_area method)",
    )
    headcount_lessee: Optional[int] = Field(
        None,
        ge=1,
        description="Lessee headcount (required for headcount method)",
    )
    headcount_total: Optional[int] = Field(
        None,
        ge=1,
        description="Total building headcount (required for headcount method)",
    )


# ============================================================================
# RESPONSE MODELS
# ============================================================================


class CalculationResponse(BaseModel):
    """Response model for single calculation result."""

    success: bool = Field(
        ..., description="Whether calculation completed successfully"
    )
    calculation_id: str = Field(..., description="Unique calculation UUID")
    asset_type: str = Field(
        ..., description="Asset type (building, vehicle, equipment, it_asset, lessor, spend)"
    )
    method: str = Field(
        ..., description="Calculation method applied"
    )
    total_co2e_kg: float = Field(
        ..., description="Total CO2e emissions in kg"
    )
    co2_kg: Optional[float] = Field(
        None, description="CO2 component in kg"
    )
    ch4_co2e_kg: Optional[float] = Field(
        None, description="CH4 component as CO2e in kg"
    )
    n2o_co2e_kg: Optional[float] = Field(
        None, description="N2O component as CO2e in kg"
    )
    allocation_factor: Optional[float] = Field(
        None, description="Allocation factor applied (0.0-1.0)"
    )
    dqi_score: Optional[float] = Field(
        None, description="Data quality indicator composite score (1.0-5.0)"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance chain hash"
    )
    detail: Optional[Dict[str, Any]] = Field(
        None, description="Full calculation detail payload"
    )
    calculated_at: str = Field(
        ..., description="ISO 8601 calculation timestamp"
    )


class BatchResponse(BaseModel):
    """Response model for batch calculation."""

    success: bool = Field(
        ..., description="Whether batch completed successfully"
    )
    batch_id: str = Field(..., description="Unique batch UUID")
    total_items: int = Field(
        ..., description="Total items in batch"
    )
    successful: int = Field(
        ..., description="Number of successful calculations"
    )
    failed: int = Field(
        ..., description="Number of failed calculations"
    )
    total_co2e_kg: float = Field(
        ..., description="Total CO2e for all items in batch (kg)"
    )
    results: List[Dict[str, Any]] = Field(
        ..., description="Individual calculation results"
    )
    errors: List[Dict[str, Any]] = Field(
        default_factory=list, description="Per-item error details"
    )


class ComplianceResponse(BaseModel):
    """Response model for compliance check."""

    success: bool = Field(
        ..., description="Whether compliance check completed"
    )
    overall_status: str = Field(
        ..., description="Overall compliance status (pass, fail, warning)"
    )
    overall_score: float = Field(
        ..., description="Overall compliance score (0.0-1.0)"
    )
    framework_results: List[Dict[str, Any]] = Field(
        ..., description="Per-framework compliance results with findings"
    )
    double_counting_flags: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Double-counting prevention rule violations",
    )


class UncertaintyResponse(BaseModel):
    """Response model for uncertainty analysis."""

    success: bool = Field(
        ..., description="Whether analysis completed successfully"
    )
    mean_co2e_kg: float = Field(..., description="Mean CO2e (kg)")
    std_dev_kg: float = Field(..., description="Standard deviation (kg)")
    ci_lower_kg: float = Field(
        ..., description="Confidence interval lower bound (kg)"
    )
    ci_upper_kg: float = Field(
        ..., description="Confidence interval upper bound (kg)"
    )
    uncertainty_pct: float = Field(
        ..., description="Uncertainty as percentage (+/-%)"
    )
    method: str = Field(..., description="Uncertainty method used")
    iterations: int = Field(
        ..., description="Iterations performed (0 for non-Monte Carlo)"
    )
    confidence_level: float = Field(
        ..., description="Confidence level used"
    )


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service health status")
    agent_id: str = Field(..., description="Agent identifier")
    version: str = Field(..., description="Agent version")
    uptime_seconds: float = Field(
        ..., description="Seconds since service start"
    )
    engines_status: Optional[Dict[str, str]] = Field(
        None,
        description="Per-engine health status",
    )


class EmissionFactorListResponse(BaseModel):
    """Response model for emission factor listing."""

    success: bool = Field(
        ..., description="Whether retrieval succeeded"
    )
    ef_type: str = Field(
        ..., description="Emission factor type (building, vehicle, equipment, it_asset)"
    )
    factors: List[Dict[str, Any]] = Field(
        ..., description="List of emission factors"
    )
    count: int = Field(..., description="Total factor count returned")


class BuildingBenchmarkResponse(BaseModel):
    """Response model for building energy benchmarks."""

    success: bool = Field(
        ..., description="Whether retrieval succeeded"
    )
    benchmarks: List[Dict[str, Any]] = Field(
        ..., description="Building energy intensity benchmarks"
    )
    count: int = Field(..., description="Number of benchmarks returned")


class GridFactorResponse(BaseModel):
    """Response model for grid emission factors."""

    success: bool = Field(
        ..., description="Whether retrieval succeeded"
    )
    country_code: str = Field(
        ..., description="Country or region code"
    )
    grid_ef_kgco2e_per_kwh: float = Field(
        ..., description="Grid emission factor (kgCO2e/kWh)"
    )
    source: str = Field(
        ..., description="Data source (IEA, eGRID, AIB)"
    )
    source_year: int = Field(
        ..., description="Source data year"
    )
    subregions: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Subregional factors (US eGRID only)",
    )


class LeaseClassificationResponse(BaseModel):
    """Response model for lease classification guidance."""

    success: bool = Field(
        ..., description="Whether retrieval succeeded"
    )
    classifications: List[Dict[str, Any]] = Field(
        ..., description="Lease classification rules and guidance"
    )
    count: int = Field(
        ..., description="Number of classification entries"
    )


class AggregationResponse(BaseModel):
    """Response model for aggregated emissions."""

    success: bool = Field(
        ..., description="Whether aggregation succeeded"
    )
    period: str = Field(..., description="Aggregation period identifier")
    total_co2e_kg: float = Field(
        ..., description="Total CO2e for the period (kg)"
    )
    by_asset_type: Dict[str, float] = Field(
        ..., description="CO2e breakdown by asset type"
    )
    by_country: Dict[str, float] = Field(
        ..., description="CO2e breakdown by country"
    )
    by_method: Dict[str, float] = Field(
        ..., description="CO2e breakdown by calculation method"
    )
    asset_count: int = Field(
        ..., description="Total assets in aggregation"
    )


class ProvenanceResponse(BaseModel):
    """Response model for provenance chain verification."""

    success: bool = Field(
        ..., description="Whether provenance retrieval succeeded"
    )
    calculation_id: str = Field(..., description="Calculation UUID")
    chain: List[Dict[str, Any]] = Field(
        ..., description="Ordered list of provenance stage records"
    )
    is_valid: bool = Field(
        ..., description="Whether the provenance chain is intact"
    )
    root_hash: str = Field(
        ..., description="Root SHA-256 hash of the chain"
    )
    stages_count: int = Field(
        ..., description="Number of stages in chain"
    )


class CalculationListResponse(BaseModel):
    """Response model for paginated calculation listing."""

    calculations: List[Dict[str, Any]] = Field(
        ..., description="Calculation summaries"
    )
    count: int = Field(..., description="Total matching calculations")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Page size")


class CalculationDetailResponse(BaseModel):
    """Response model for single calculation detail."""

    success: bool = Field(
        ..., description="Whether retrieval succeeded"
    )
    calculation_id: str = Field(..., description="Unique calculation UUID")
    asset_type: str = Field(..., description="Asset type")
    method: str = Field(..., description="Calculation method")
    total_co2e_kg: float = Field(..., description="Total CO2e (kg)")
    details: Dict[str, Any] = Field(
        ..., description="Full calculation detail payload"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance chain hash"
    )
    calculated_at: str = Field(
        ..., description="ISO 8601 calculation timestamp"
    )


class DeleteResponse(BaseModel):
    """Response model for soft deletion."""

    calculation_id: str = Field(..., description="Deleted calculation UUID")
    deleted: bool = Field(..., description="Whether deletion succeeded")
    message: str = Field(..., description="Human-readable status message")


# ============================================================================
# MODULE-LEVEL TRACKING
# ============================================================================

_start_time: datetime = datetime.utcnow()


# ============================================================================
# ENDPOINTS - CALCULATIONS (9)
# ============================================================================


@router.post(
    "/calculate",
    response_model=CalculationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate upstream leased asset emissions",
    description=(
        "Calculate GHG emissions for a single upstream leased asset through the "
        "full pipeline. Accepts a generic request dict with 'asset_type' field to "
        "route to the appropriate engine. Returns deterministic results with "
        "SHA-256 provenance hash."
    ),
)
async def calculate_emissions(
    request: Dict[str, Any],
    service=Depends(get_service),
) -> CalculationResponse:
    """
    Calculate upstream leased asset emissions through the full pipeline.

    Args:
        request: Calculation request dict with asset_type and asset-specific data
        service: UpstreamLeasedAssetsService instance

    Returns:
        CalculationResponse with emissions and provenance hash

    Raises:
        HTTPException: 400 for validation errors, 500 for processing failures
    """
    try:
        asset_type = request.get("asset_type", "unknown")
        logger.info(
            f"Calculating upstream leased asset emissions for asset_type={asset_type}"
        )

        result = service.calculate(request)
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return CalculationResponse(
            success=True,
            calculation_id=calculation_id,
            asset_type=result.get("asset_type", asset_type),
            method=result.get("method", "asset_specific"),
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            co2_kg=result.get("co2_kg"),
            ch4_co2e_kg=result.get("ch4_co2e_kg"),
            n2o_co2e_kg=result.get("n2o_co2e_kg"),
            allocation_factor=result.get("allocation_factor"),
            dqi_score=result.get("dqi_score"),
            provenance_hash=result.get("provenance_hash", ""),
            detail=result.get("detail"),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error(f"Validation error in calculate_emissions: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error in calculate_emissions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Calculation failed",
        )


@router.post(
    "/calculate/building",
    response_model=CalculationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate building emissions",
    description=(
        "Calculate GHG emissions for a leased building using energy consumption "
        "data (electricity, gas, heating, cooling). Applies country-specific grid "
        "emission factors, fuel emission factors, and floor area allocation. "
        "Supports 8 building types with climate zone benchmarks."
    ),
)
async def calculate_building_emissions(
    request: BuildingCalculateRequest,
    service=Depends(get_service),
) -> CalculationResponse:
    """
    Calculate building-level leased asset emissions.

    Args:
        request: Building calculation request with energy consumption data
        service: UpstreamLeasedAssetsService instance

    Returns:
        CalculationResponse with building emissions

    Raises:
        HTTPException: 400 for validation errors, 500 for processing failures
    """
    try:
        logger.info(
            f"Calculating building emissions: type={request.building_type}, "
            f"area={request.floor_area_sqm}sqm, country={request.country_code}"
        )

        result = service.calculate_building(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return CalculationResponse(
            success=True,
            calculation_id=calculation_id,
            asset_type="building",
            method=result.get("method", "asset_specific"),
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            co2_kg=result.get("co2_kg"),
            ch4_co2e_kg=result.get("ch4_co2e_kg"),
            n2o_co2e_kg=result.get("n2o_co2e_kg"),
            allocation_factor=result.get("allocation_factor", request.allocation_factor),
            dqi_score=result.get("dqi_score"),
            provenance_hash=result.get("provenance_hash", ""),
            detail=result.get("detail"),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error(f"Validation error in calculate_building_emissions: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in calculate_building_emissions: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Building calculation failed",
        )


@router.post(
    "/calculate/vehicle",
    response_model=CalculationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate vehicle fleet emissions",
    description=(
        "Calculate GHG emissions for leased vehicle fleets using distance-based "
        "or fuel-based methods. Supports 11 vehicle types with DEFRA 2024 "
        "emission factors. Includes well-to-tank (WTT) emissions and optional "
        "vehicle age adjustment."
    ),
)
async def calculate_vehicle_emissions(
    request: VehicleCalculateRequest,
    service=Depends(get_service),
) -> CalculationResponse:
    """
    Calculate leased vehicle fleet emissions.

    Args:
        request: Vehicle calculation request with type, fuel, and distance/litres
        service: UpstreamLeasedAssetsService instance

    Returns:
        CalculationResponse with vehicle fleet emissions

    Raises:
        HTTPException: 400 for missing distance/fuel data, 500 for failures
    """
    try:
        logger.info(
            f"Calculating vehicle emissions: type={request.vehicle_type}, "
            f"fuel={request.fuel_type}, count={request.count}"
        )

        result = service.calculate_vehicle(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return CalculationResponse(
            success=True,
            calculation_id=calculation_id,
            asset_type="vehicle",
            method=result.get("method", "distance_based"),
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            co2_kg=result.get("co2_kg"),
            ch4_co2e_kg=result.get("ch4_co2e_kg"),
            n2o_co2e_kg=result.get("n2o_co2e_kg"),
            allocation_factor=None,
            dqi_score=result.get("dqi_score"),
            provenance_hash=result.get("provenance_hash", ""),
            detail=result.get("detail"),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error(f"Validation error in calculate_vehicle_emissions: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in calculate_vehicle_emissions: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Vehicle calculation failed",
        )


@router.post(
    "/calculate/equipment",
    response_model=CalculationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate equipment emissions",
    description=(
        "Calculate GHG emissions for leased equipment using power rating, "
        "operating hours, load factor, and fuel-specific emission factors. "
        "Supports 10 equipment types including generators, compressors, "
        "forklifts, and HVAC units."
    ),
)
async def calculate_equipment_emissions(
    request: EquipmentCalculateRequest,
    service=Depends(get_service),
) -> CalculationResponse:
    """
    Calculate leased equipment emissions.

    Args:
        request: Equipment calculation request with power, hours, and load factor
        service: UpstreamLeasedAssetsService instance

    Returns:
        CalculationResponse with equipment emissions

    Raises:
        HTTPException: 400 for invalid parameters, 500 for failures
    """
    try:
        logger.info(
            f"Calculating equipment emissions: type={request.equipment_type}, "
            f"power={request.power_kw}kW, hours={request.operating_hours}"
        )

        result = service.calculate_equipment(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return CalculationResponse(
            success=True,
            calculation_id=calculation_id,
            asset_type="equipment",
            method=result.get("method", "engineering"),
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            co2_kg=result.get("co2_kg"),
            ch4_co2e_kg=result.get("ch4_co2e_kg"),
            n2o_co2e_kg=result.get("n2o_co2e_kg"),
            allocation_factor=None,
            dqi_score=result.get("dqi_score"),
            provenance_hash=result.get("provenance_hash", ""),
            detail=result.get("detail"),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error(f"Validation error in calculate_equipment_emissions: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in calculate_equipment_emissions: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Equipment calculation failed",
        )


@router.post(
    "/calculate/it-asset",
    response_model=CalculationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate IT asset emissions",
    description=(
        "Calculate GHG emissions for leased IT assets using power draw, PUE, "
        "utilization, and grid emission factors. Supports 12 IT asset types "
        "including servers, storage, network equipment, and end-user devices. "
        "PUE accounts for data centre cooling and infrastructure overhead."
    ),
)
async def calculate_it_asset_emissions(
    request: ITAssetCalculateRequest,
    service=Depends(get_service),
) -> CalculationResponse:
    """
    Calculate leased IT asset emissions.

    Args:
        request: IT asset calculation request with power, PUE, utilization
        service: UpstreamLeasedAssetsService instance

    Returns:
        CalculationResponse with IT asset emissions

    Raises:
        HTTPException: 400 for invalid parameters, 500 for failures
    """
    try:
        logger.info(
            f"Calculating IT asset emissions: type={request.it_type}, "
            f"power={request.power_kw}kW, pue={request.pue}, count={request.count}"
        )

        result = service.calculate_it_asset(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return CalculationResponse(
            success=True,
            calculation_id=calculation_id,
            asset_type="it_asset",
            method=result.get("method", "engineering"),
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            co2_kg=result.get("co2_kg"),
            ch4_co2e_kg=result.get("ch4_co2e_kg"),
            n2o_co2e_kg=result.get("n2o_co2e_kg"),
            allocation_factor=None,
            dqi_score=result.get("dqi_score"),
            provenance_hash=result.get("provenance_hash", ""),
            detail=result.get("detail"),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error(f"Validation error in calculate_it_asset_emissions: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in calculate_it_asset_emissions: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="IT asset calculation failed",
        )


@router.post(
    "/calculate/lessor",
    response_model=CalculationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate lessor-specific emissions",
    description=(
        "Calculate emissions using data reported directly by the lessor. "
        "Applies allocation factor and lease term proration. Requires "
        "lessor-reported CO2e, methodology, and allocation parameters."
    ),
)
async def calculate_lessor_emissions(
    request: LessorCalculateRequest,
    service=Depends(get_service),
) -> CalculationResponse:
    """
    Calculate emissions from lessor-reported data.

    Args:
        request: Lessor calculation request with reported CO2e and allocation
        service: UpstreamLeasedAssetsService instance

    Returns:
        CalculationResponse with allocated lessor emissions

    Raises:
        HTTPException: 400 for invalid data, 500 for failures
    """
    try:
        logger.info(
            f"Calculating lessor emissions: lessor={request.lessor_name}, "
            f"reported_co2e={request.reported_co2e_kg}kg"
        )

        result = service.calculate_lessor(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return CalculationResponse(
            success=True,
            calculation_id=calculation_id,
            asset_type="lessor",
            method=result.get("method", "lessor_specific"),
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            co2_kg=result.get("co2_kg"),
            ch4_co2e_kg=result.get("ch4_co2e_kg"),
            n2o_co2e_kg=result.get("n2o_co2e_kg"),
            allocation_factor=result.get(
                "allocation_factor", request.allocation_factor
            ),
            dqi_score=result.get("dqi_score"),
            provenance_hash=result.get("provenance_hash", ""),
            detail=result.get("detail"),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error(f"Validation error in calculate_lessor_emissions: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in calculate_lessor_emissions: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Lessor calculation failed",
        )


@router.post(
    "/calculate/spend",
    response_model=CalculationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate spend-based emissions",
    description=(
        "Calculate GHG emissions using spend-based EEIO factors. Applies CPI "
        "deflation to base year (2021 USD), currency conversion, and margin "
        "removal. Supports NAICS codes for real estate, equipment rental, "
        "vehicle rental, and data centre leasing."
    ),
)
async def calculate_spend_emissions(
    request: SpendCalculateRequest,
    service=Depends(get_service),
) -> CalculationResponse:
    """
    Calculate spend-based emissions using EEIO factors.

    Args:
        request: Spend calculation request with NAICS code and amount
        service: UpstreamLeasedAssetsService instance

    Returns:
        CalculationResponse with spend-based emissions

    Raises:
        HTTPException: 400 for invalid NAICS code, 500 for failures
    """
    try:
        logger.info(
            f"Calculating spend emissions: naics={request.naics_code}, "
            f"amount={request.amount} {request.currency}"
        )

        result = service.calculate_spend(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return CalculationResponse(
            success=True,
            calculation_id=calculation_id,
            asset_type="spend",
            method=result.get("method", "spend_based"),
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            co2_kg=result.get("co2_kg"),
            ch4_co2e_kg=result.get("ch4_co2e_kg"),
            n2o_co2e_kg=result.get("n2o_co2e_kg"),
            allocation_factor=None,
            dqi_score=result.get("dqi_score"),
            provenance_hash=result.get("provenance_hash", ""),
            detail=result.get("detail"),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error(f"Validation error in calculate_spend_emissions: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in calculate_spend_emissions: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Spend calculation failed",
        )


@router.post(
    "/calculate/batch",
    response_model=BatchResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Batch calculate upstream leased asset emissions",
    description=(
        "Calculate GHG emissions for multiple leased assets in a single request. "
        "Processes up to 1,000 items with per-item error isolation. Each item "
        "must include 'asset_type' to route to the appropriate calculation engine. "
        "Returns aggregated totals with individual results and any per-item errors."
    ),
)
async def calculate_batch_emissions(
    request: BatchCalculateRequest,
    service=Depends(get_service),
) -> BatchResponse:
    """
    Calculate batch upstream leased asset emissions.

    Args:
        request: Batch calculation request with item list
        service: UpstreamLeasedAssetsService instance

    Returns:
        BatchResponse with aggregated and per-item results

    Raises:
        HTTPException: 400 for validation errors, 500 for batch failures
    """
    try:
        logger.info(
            f"Calculating batch emissions for {len(request.items)} items"
        )

        result = service.calculate_batch(request.dict())
        batch_id = result.get("batch_id", str(uuid.uuid4()))

        return BatchResponse(
            success=True,
            batch_id=batch_id,
            total_items=result.get("total_items", len(request.items)),
            successful=result.get("successful", 0),
            failed=result.get("failed", 0),
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            results=result.get("results", []),
            errors=result.get("errors", []),
        )

    except ValueError as e:
        logger.error(f"Validation error in calculate_batch_emissions: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in calculate_batch_emissions: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch calculation failed",
        )


@router.post(
    "/calculate/portfolio",
    response_model=CalculationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate portfolio-level emissions",
    description=(
        "Calculate GHG emissions across the entire leased asset portfolio "
        "including buildings, vehicles, equipment, and IT assets. Performs "
        "cross-category aggregation, double-counting prevention, and "
        "portfolio-level compliance checking."
    ),
)
async def calculate_portfolio_emissions(
    request: PortfolioCalculateRequest,
    service=Depends(get_service),
) -> CalculationResponse:
    """
    Calculate portfolio-level upstream leased asset emissions.

    Args:
        request: Portfolio calculation request with all asset categories
        service: UpstreamLeasedAssetsService instance

    Returns:
        CalculationResponse with portfolio-level aggregated emissions

    Raises:
        HTTPException: 400 for validation errors, 500 for failures
    """
    try:
        total_assets = (
            len(request.buildings)
            + len(request.vehicles)
            + len(request.equipment)
            + len(request.it_assets)
        )
        logger.info(
            f"Calculating portfolio emissions: {total_assets} total assets, "
            f"year={request.reporting_year}"
        )

        result = service.calculate_portfolio(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return CalculationResponse(
            success=True,
            calculation_id=calculation_id,
            asset_type="portfolio",
            method=result.get("method", "portfolio_aggregation"),
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            co2_kg=result.get("co2_kg"),
            ch4_co2e_kg=result.get("ch4_co2e_kg"),
            n2o_co2e_kg=result.get("n2o_co2e_kg"),
            allocation_factor=None,
            dqi_score=result.get("dqi_score"),
            provenance_hash=result.get("provenance_hash", ""),
            detail=result.get("detail"),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error(f"Validation error in calculate_portfolio_emissions: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in calculate_portfolio_emissions: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Portfolio calculation failed",
        )


# ============================================================================
# ENDPOINTS - COMPLIANCE & UNCERTAINTY (2)
# ============================================================================


@router.post(
    "/compliance/check",
    response_model=ComplianceResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Check multi-framework compliance",
    description=(
        "Check upstream leased assets calculation results against one or more "
        "regulatory frameworks. Validates completeness, boundary correctness, "
        "allocation method disclosure, lease classification, data quality, and "
        "double-counting prevention. Supports GHG Protocol, ISO 14064, CSRD "
        "ESRS E1, CDP, SBTi, SB 253, and GRI 305."
    ),
)
async def check_compliance(
    request: ComplianceCheckRequest,
    service=Depends(get_service),
) -> ComplianceResponse:
    """
    Check calculation compliance against regulatory frameworks.

    Args:
        request: Compliance check request with frameworks and total CO2e
        service: UpstreamLeasedAssetsService instance

    Returns:
        ComplianceResponse with per-framework findings

    Raises:
        HTTPException: 400 for invalid frameworks, 500 for check failures
    """
    try:
        logger.info(
            f"Checking compliance for {len(request.frameworks)} frameworks, "
            f"total_co2e={request.total_co2e}kg"
        )

        result = service.check_compliance(request.dict())

        return ComplianceResponse(
            success=True,
            overall_status=result.get("overall_status", "unknown"),
            overall_score=result.get("overall_score", 0.0),
            framework_results=result.get("framework_results", []),
            double_counting_flags=result.get("double_counting_flags", []),
        )

    except ValueError as e:
        logger.error(f"Validation error in check_compliance: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error in check_compliance: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Compliance check failed",
        )


@router.post(
    "/uncertainty/analyze",
    response_model=UncertaintyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Analyze calculation uncertainty",
    description=(
        "Perform uncertainty analysis on upstream leased assets emissions "
        "calculations. Supports Monte Carlo simulation, analytical error "
        "propagation, and IPCC Tier 2 default ranges. Returns mean, "
        "standard deviation, and confidence intervals."
    ),
)
async def analyze_uncertainty(
    request: UncertaintyRequest,
    service=Depends(get_service),
) -> UncertaintyResponse:
    """
    Perform uncertainty analysis on calculation results.

    Args:
        request: Uncertainty analysis request
        service: UpstreamLeasedAssetsService instance

    Returns:
        UncertaintyResponse with statistical uncertainty metrics

    Raises:
        HTTPException: 400 for invalid method, 500 for analysis failures
    """
    try:
        logger.info(
            f"Analyzing uncertainty: method={request.method}, "
            f"iterations={request.iterations}, "
            f"confidence={request.confidence_level}"
        )

        result = service.analyze_uncertainty(request.dict())

        return UncertaintyResponse(
            success=True,
            mean_co2e_kg=result.get("mean_co2e_kg", 0.0),
            std_dev_kg=result.get("std_dev_kg", 0.0),
            ci_lower_kg=result.get("ci_lower_kg", 0.0),
            ci_upper_kg=result.get("ci_upper_kg", 0.0),
            uncertainty_pct=result.get("uncertainty_pct", 0.0),
            method=result.get("method", request.method),
            iterations=result.get("iterations", request.iterations),
            confidence_level=result.get(
                "confidence_level", request.confidence_level
            ),
        )

    except ValueError as e:
        logger.error(f"Validation error in analyze_uncertainty: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error in analyze_uncertainty: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Uncertainty analysis failed",
        )


# ============================================================================
# ENDPOINTS - CALCULATION CRUD (3)
# ============================================================================


@router.get(
    "/calculations/{calc_id}",
    response_model=CalculationDetailResponse,
    summary="Get calculation detail",
    description=(
        "Retrieve detailed information for a specific upstream leased assets "
        "calculation including full input/output payload, emission breakdown "
        "by gas species, allocation details, provenance hash, and calculation "
        "metadata."
    ),
)
async def get_calculation_detail(
    calc_id: str = Path(..., description="Calculation UUID"),
    service=Depends(get_service),
) -> CalculationDetailResponse:
    """
    Get detailed information for a specific calculation.

    Args:
        calc_id: Calculation UUID
        service: UpstreamLeasedAssetsService instance

    Returns:
        CalculationDetailResponse with full calculation data

    Raises:
        HTTPException: 404 if calculation not found, 500 for failures
    """
    try:
        logger.info(f"Getting calculation detail: {calc_id}")

        result = service.get_calculation(calc_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Calculation {calc_id} not found",
            )

        return CalculationDetailResponse(
            success=True,
            calculation_id=result.get("calculation_id", calc_id),
            asset_type=result.get("asset_type", ""),
            method=result.get("method", ""),
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            details=result.get("details", result.get("detail", {})),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_calculation_detail: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve calculation",
        )


@router.get(
    "/calculations",
    response_model=CalculationListResponse,
    summary="List calculations",
    description=(
        "Retrieve a paginated list of upstream leased assets calculations. "
        "Supports filtering by asset type, calculation method, country, "
        "and date range. Returns summary information for each calculation."
    ),
)
async def list_calculations(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=500, description="Results per page"),
    asset_type: Optional[str] = Query(
        None, description="Filter by asset type"
    ),
    method: Optional[str] = Query(
        None,
        description="Filter by calculation method",
    ),
    country_code: Optional[str] = Query(
        None, description="Filter by country code"
    ),
    from_date: Optional[str] = Query(
        None, description="Filter from date (ISO 8601)"
    ),
    to_date: Optional[str] = Query(
        None, description="Filter to date (ISO 8601)"
    ),
    service=Depends(get_service),
) -> CalculationListResponse:
    """
    List upstream leased assets calculations with filtering and pagination.

    Args:
        page: Page number (1-indexed)
        page_size: Number of results per page
        asset_type: Optional asset type filter
        method: Optional calculation method filter
        country_code: Optional country code filter
        from_date: Optional start date filter
        to_date: Optional end date filter
        service: UpstreamLeasedAssetsService instance

    Returns:
        CalculationListResponse with paginated results

    Raises:
        HTTPException: 500 for listing failures
    """
    try:
        logger.info(
            f"Listing calculations: page={page}, size={page_size}, "
            f"asset_type={asset_type}, method={method}"
        )

        filters = {
            "page": page,
            "page_size": page_size,
            "asset_type": asset_type,
            "method": method,
            "country_code": country_code,
            "from_date": from_date,
            "to_date": to_date,
        }

        result = service.list_calculations(filters)

        return CalculationListResponse(
            calculations=result.get("calculations", []),
            count=result.get("count", 0),
            page=page,
            page_size=page_size,
        )

    except Exception as e:
        logger.error(f"Error in list_calculations: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list calculations",
        )


@router.delete(
    "/calculations/{calc_id}",
    response_model=DeleteResponse,
    summary="Delete calculation",
    description=(
        "Soft-delete a specific upstream leased assets calculation. "
        "Marks the calculation as deleted with audit trail; "
        "data is retained for regulatory compliance per GHG Protocol "
        "and CSRD data retention requirements."
    ),
)
async def delete_calculation(
    calc_id: str = Path(..., description="Calculation UUID"),
    service=Depends(get_service),
) -> DeleteResponse:
    """
    Soft-delete a specific calculation.

    Args:
        calc_id: Calculation UUID
        service: UpstreamLeasedAssetsService instance

    Returns:
        DeleteResponse with deletion confirmation

    Raises:
        HTTPException: 404 if not found, 500 for deletion failures
    """
    try:
        logger.info(f"Deleting calculation: {calc_id}")

        deleted = service.delete_calculation(calc_id)

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Calculation {calc_id} not found",
            )

        return DeleteResponse(
            calculation_id=calc_id,
            deleted=True,
            message=f"Calculation {calc_id} soft-deleted successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in delete_calculation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete calculation",
        )


# ============================================================================
# ENDPOINTS - EMISSION FACTORS & REFERENCE DATA (5)
# ============================================================================


@router.get(
    "/emission-factors/{ef_type}",
    response_model=EmissionFactorListResponse,
    summary="Get emission factors by asset type",
    description=(
        "Retrieve emission factors for a specific asset type. Returns "
        "factors for buildings (grid, gas, heating), vehicles (per-km, "
        "per-litre), equipment (per-kWh, per-litre), or IT assets "
        "(per-kWh with PUE)."
    ),
)
async def get_emission_factors(
    ef_type: str = Path(
        ...,
        description="Emission factor type (building, vehicle, equipment, it_asset)",
    ),
    country_code: Optional[str] = Query(
        None, description="Filter by country code"
    ),
    service=Depends(get_service),
) -> EmissionFactorListResponse:
    """
    Get emission factors for a specific asset type.

    Args:
        ef_type: Emission factor type identifier
        country_code: Optional country code filter
        service: UpstreamLeasedAssetsService instance

    Returns:
        EmissionFactorListResponse with factors for the asset type

    Raises:
        HTTPException: 400 for invalid type, 500 for retrieval failures
    """
    try:
        logger.info(
            f"Getting emission factors: type={ef_type}, country={country_code}"
        )

        result = service.get_emission_factors(ef_type, country_code)

        return EmissionFactorListResponse(
            success=True,
            ef_type=ef_type,
            factors=result.get("factors", []),
            count=result.get("count", 0),
        )

    except ValueError as e:
        logger.error(
            f"Validation error in get_emission_factors: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in get_emission_factors: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve emission factors",
        )


@router.get(
    "/building-benchmarks",
    response_model=BuildingBenchmarkResponse,
    summary="Get building energy benchmarks",
    description=(
        "Retrieve building energy intensity benchmarks by building type "
        "and climate zone. Returns kWh/sqm/year for electricity, gas, "
        "heating, and cooling from ASHRAE, CIBSE, and Energy Star data. "
        "Used for gap analysis and data quality assessment."
    ),
)
async def get_building_benchmarks(
    building_type: Optional[str] = Query(
        None, description="Filter by building type"
    ),
    climate_zone: Optional[str] = Query(
        None, description="Filter by climate zone"
    ),
    service=Depends(get_service),
) -> BuildingBenchmarkResponse:
    """
    Get building energy intensity benchmarks.

    Args:
        building_type: Optional building type filter
        climate_zone: Optional climate zone filter
        service: UpstreamLeasedAssetsService instance

    Returns:
        BuildingBenchmarkResponse with energy benchmarks

    Raises:
        HTTPException: 500 for retrieval failures
    """
    try:
        logger.info(
            f"Getting building benchmarks: type={building_type}, "
            f"zone={climate_zone}"
        )

        result = service.get_building_benchmarks(building_type, climate_zone)

        return BuildingBenchmarkResponse(
            success=True,
            benchmarks=result.get("benchmarks", []),
            count=result.get("count", 0),
        )

    except Exception as e:
        logger.error(
            f"Error in get_building_benchmarks: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve building benchmarks",
        )


@router.get(
    "/grid-factors/{country}",
    response_model=GridFactorResponse,
    summary="Get grid emission factors",
    description=(
        "Retrieve grid emission factors for a country. Returns kgCO2e/kWh "
        "from IEA 2024 data for 19+ countries. For the US, also returns "
        "eGRID subregional factors (26 subregions)."
    ),
)
async def get_grid_factors(
    country: str = Path(
        ...,
        description="ISO country code (US, GB, DE, FR, JP, CA, AU, etc.)",
    ),
    service=Depends(get_service),
) -> GridFactorResponse:
    """
    Get grid emission factors for a country.

    Args:
        country: ISO country code
        service: UpstreamLeasedAssetsService instance

    Returns:
        GridFactorResponse with grid emission factor data

    Raises:
        HTTPException: 400 for invalid country, 500 for retrieval failures
    """
    try:
        logger.info(f"Getting grid factors for country: {country}")

        result = service.get_grid_factors(country)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No grid factors found for country '{country}'",
            )

        return GridFactorResponse(
            success=True,
            country_code=result.get("country_code", country.upper()),
            grid_ef_kgco2e_per_kwh=result.get("grid_ef_kgco2e_per_kwh", 0.0),
            source=result.get("source", "IEA 2024"),
            source_year=result.get("source_year", 2024),
            subregions=result.get("subregions"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_grid_factors: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve grid factors",
        )


@router.get(
    "/lease-classification",
    response_model=LeaseClassificationResponse,
    summary="Get lease classification guidance",
    description=(
        "Retrieve operating vs finance lease classification guidance per "
        "GHG Protocol, IFRS 16, and ASC 842. Determines whether leased "
        "assets should be reported under Scope 3 Category 8 (upstream "
        "leased assets) or Scope 1/2 (if lessee has operational control)."
    ),
)
async def get_lease_classification(
    service=Depends(get_service),
) -> LeaseClassificationResponse:
    """
    Get lease classification guidance.

    Args:
        service: UpstreamLeasedAssetsService instance

    Returns:
        LeaseClassificationResponse with classification rules

    Raises:
        HTTPException: 500 for retrieval failures
    """
    try:
        logger.info("Getting lease classification guidance")

        result = service.get_lease_classification()

        return LeaseClassificationResponse(
            success=True,
            classifications=result.get("classifications", []),
            count=result.get("count", 0),
        )

    except Exception as e:
        logger.error(
            f"Error in get_lease_classification: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve lease classification",
        )


@router.get(
    "/aggregations",
    response_model=AggregationResponse,
    summary="Get aggregated emissions",
    description=(
        "Retrieve aggregated upstream leased assets emissions. Returns "
        "totals with breakdowns by asset type, country, and calculation "
        "method. Supports date range filtering."
    ),
)
async def get_aggregations(
    period: str = Query(
        "annual",
        description="Aggregation period (monthly, quarterly, annual)",
    ),
    from_date: Optional[str] = Query(
        None, description="Start date (ISO 8601)"
    ),
    to_date: Optional[str] = Query(
        None, description="End date (ISO 8601)"
    ),
    service=Depends(get_service),
) -> AggregationResponse:
    """
    Get aggregated emissions for a specified period.

    Args:
        period: Aggregation period identifier
        from_date: Optional start date filter
        to_date: Optional end date filter
        service: UpstreamLeasedAssetsService instance

    Returns:
        AggregationResponse with aggregated emissions data

    Raises:
        HTTPException: 400 for invalid period, 500 for aggregation failures
    """
    try:
        logger.info(
            f"Getting aggregations: period={period}, "
            f"from={from_date}, to={to_date}"
        )

        valid_periods = {"monthly", "quarterly", "annual"}
        if period not in valid_periods:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Invalid period '{period}'. "
                    f"Must be one of: {', '.join(sorted(valid_periods))}"
                ),
            )

        filters = {
            "period": period,
            "from_date": from_date,
            "to_date": to_date,
        }

        result = service.get_aggregations(filters)

        return AggregationResponse(
            success=True,
            period=period,
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            by_asset_type=result.get("by_asset_type", {}),
            by_country=result.get("by_country", {}),
            by_method=result.get("by_method", {}),
            asset_count=result.get("asset_count", 0),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_aggregations: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Aggregation failed",
        )


@router.get(
    "/provenance/{calc_id}",
    response_model=ProvenanceResponse,
    summary="Get provenance chain",
    description=(
        "Retrieve the complete SHA-256 provenance chain for a calculation. "
        "Includes all pipeline stages with per-stage hashes and verification."
    ),
)
async def get_provenance(
    calc_id: str = Path(..., description="Calculation UUID"),
    service=Depends(get_service),
) -> ProvenanceResponse:
    """
    Get provenance chain for a specific calculation.

    Args:
        calc_id: Calculation UUID
        service: UpstreamLeasedAssetsService instance

    Returns:
        ProvenanceResponse with chain stages and verification status

    Raises:
        HTTPException: 404 if not found, 500 for retrieval failures
    """
    try:
        logger.info(f"Getting provenance for calculation: {calc_id}")

        result = service.get_provenance(calc_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Provenance for calculation {calc_id} not found",
            )

        return ProvenanceResponse(
            success=True,
            calculation_id=result.get("calculation_id", calc_id),
            chain=result.get("chain", []),
            is_valid=result.get("is_valid", False),
            root_hash=result.get("root_hash", ""),
            stages_count=result.get("stages_count", 0),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_provenance: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve provenance",
        )


# ============================================================================
# ENDPOINTS - ANALYSIS (1)
# ============================================================================


@router.post(
    "/portfolio/analyze",
    response_model=CalculationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Analyze portfolio hot-spots",
    description=(
        "Identify top emission sources across the leased asset portfolio "
        "with Pareto-based reduction opportunity ranking. Returns "
        "category-level breakdowns and optimization recommendations."
    ),
)
async def analyze_portfolio(
    request: PortfolioAnalyzeRequest,
    service=Depends(get_service),
) -> CalculationResponse:
    """
    Analyze portfolio for emission hot-spots.

    Args:
        request: Portfolio analysis request with asset results
        service: UpstreamLeasedAssetsService instance

    Returns:
        CalculationResponse with portfolio analysis results

    Raises:
        HTTPException: 400 for invalid input, 500 for analysis failures
    """
    try:
        total_assets = (
            len(request.buildings)
            + len(request.vehicles)
            + len(request.equipment)
            + len(request.it_assets)
        )
        logger.info(
            f"Analyzing portfolio: {total_assets} total assets"
        )

        result = service.analyze_portfolio(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return CalculationResponse(
            success=True,
            calculation_id=calculation_id,
            asset_type="portfolio_analysis",
            method=result.get("method", "hot_spot_analysis"),
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            co2_kg=result.get("co2_kg"),
            ch4_co2e_kg=result.get("ch4_co2e_kg"),
            n2o_co2e_kg=result.get("n2o_co2e_kg"),
            allocation_factor=None,
            dqi_score=result.get("dqi_score"),
            provenance_hash=result.get("provenance_hash", ""),
            detail=result.get("detail"),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error(f"Validation error in analyze_portfolio: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error in analyze_portfolio: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Portfolio analysis failed",
        )


# ============================================================================
# ENDPOINTS - HEALTH (1)
# ============================================================================


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description=(
        "Health check endpoint for the Upstream Leased Assets Agent. "
        "Returns service status, agent identifier, version, uptime, and "
        "per-engine health status. No authentication required."
    ),
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint (no auth required).

    Returns:
        HealthResponse with service status and uptime
    """
    try:
        uptime = (datetime.utcnow() - _start_time).total_seconds()

        # Attempt to get engine statuses from service
        engines_status = None
        try:
            svc = get_service()
            result = svc.health_check()
            engines_status = result.get("engines_status")
        except Exception:
            pass

        return HealthResponse(
            status="healthy",
            agent_id="GL-MRV-S3-008",
            version="1.0.0",
            uptime_seconds=round(uptime, 2),
            engines_status=engines_status,
        )

    except Exception as e:
        logger.error(f"Error in health_check: {e}", exc_info=True)
        return HealthResponse(
            status="unhealthy",
            agent_id="GL-MRV-S3-008",
            version="1.0.0",
            uptime_seconds=0.0,
        )
