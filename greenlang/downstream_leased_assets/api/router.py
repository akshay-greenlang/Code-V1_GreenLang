"""
Downstream Leased Assets Agent API Router - AGENT-MRV-026

This module implements the FastAPI router for downstream leased assets emissions
calculations following GHG Protocol Scope 3 Category 13 requirements.

Category 13 covers assets OWNED by the reporting company and LEASED TO other
entities. The reporter is the LESSOR. This is the mirror of Category 8
(Upstream Leased Assets) where the reporter is the lessee.

Provides 22 REST endpoints for:
- Emissions calculations (full pipeline, asset-specific building/vehicle/equipment/IT,
  average-data, spend-based, hybrid, batch, portfolio analysis)
- Allocation methods and tenant energy data handling
- Compliance checking across 7 regulatory frameworks
- Calculation CRUD (get, list, delete)
- Emission factor, building benchmark, and grid factor lookup
- Aggregations by period with asset category breakdowns
- Provenance tracking with SHA-256 chain verification
- Health check

Follows GreenLang's zero-hallucination principle with deterministic calculations.
All numeric outputs use deterministic formulas; no LLM calls in the calculation path.

Agent ID: GL-MRV-S3-013
Package: greenlang.downstream_leased_assets
API Prefix: /api/v1/downstream-leased-assets
DB Migration: V077
Metrics Prefix: gl_dla_
Table Prefix: gl_dla_

Supported asset categories (4):
    Buildings (8 types, 5 climate zones, EUI benchmarks, vacancy handling)
    Vehicles (8 types, 7 fuel types, fleet management)
    Equipment (6 types, fuel-based and load-factor calculations)
    IT Assets (7 types, PUE-adjusted power, data center focus)

Calculation methods (4):
    Asset-specific (metered energy data from tenants)
    Average-data (EUI benchmarks by building type and climate zone)
    Spend-based (EEIO factors by NAICS leasing codes)
    Hybrid (combines multiple methods with weighted aggregation)

Allocation approaches:
    Floor area, headcount, revenue, time-based, equal split, full

Regulatory frameworks (7):
    GHG Protocol, ISO 14064, CSRD ESRS E1, CDP, SBTi, SB 253, GRI 305

Example:
    >>> from fastapi import FastAPI
    >>> from greenlang.downstream_leased_assets.api.router import router
    >>> app = FastAPI()
    >>> app.include_router(router)
"""

from fastapi import APIRouter, HTTPException, Query, Path, Depends, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from decimal import Decimal
import json
import logging
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

# Router configuration
router = APIRouter(
    prefix="/api/v1/downstream-leased-assets",
    tags=["Downstream Leased Assets"],
    responses={404: {"description": "Not found"}},
)


# ============================================================================
# DECIMAL ENCODER
# ============================================================================


class DecimalEncoder(json.JSONEncoder):
    """
    JSON encoder that converts Decimal values to float.

    Required for serializing database results that contain Decimal types
    from PostgreSQL NUMERIC columns into JSON-compatible float values.

    Example:
        >>> json.dumps({"value": Decimal("3.14")}, cls=DecimalEncoder)
        '{"value": 3.14}'
    """

    def default(self, obj: Any) -> Any:
        """Encode Decimal objects as float."""
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)


# ============================================================================
# SERVICE DEPENDENCY
# ============================================================================


_service_instance = None


def get_service():
    """
    Get or create DownstreamLeasedAssetsService singleton instance.

    Uses lazy initialization to avoid circular imports and ensure the
    service is only created when first needed. The service wires together
    all 7 engines (database, asset-specific calculator, average-data
    calculator, spend-based calculator, hybrid aggregator, compliance
    checker, pipeline).

    Returns:
        DownstreamLeasedAssetsService instance

    Raises:
        HTTPException: If service initialization fails (503)
    """
    global _service_instance

    if _service_instance is None:
        try:
            from greenlang.downstream_leased_assets.setup import (
                DownstreamLeasedAssetsService,
            )
            _service_instance = DownstreamLeasedAssetsService()
            logger.info("DownstreamLeasedAssetsService initialized successfully")
        except Exception as e:
            logger.error(
                f"Failed to initialize DownstreamLeasedAssetsService: {e}"
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service initialization failed",
            )

    return _service_instance


# ============================================================================
# REQUEST MODELS
# ============================================================================


class AssetItem(BaseModel):
    """Individual asset within a full pipeline request."""

    asset_category: str = Field(
        ...,
        description=(
            "Asset category: building, vehicle, equipment, it_asset"
        ),
    )
    asset_type: str = Field(
        ...,
        description="Specific asset type within category (e.g. office, small_car, server)",
    )
    method: Optional[str] = Field(
        None,
        description="Calculation method override: asset_specific, average_data, spend_based",
    )
    floor_area_sqm: Optional[float] = Field(
        None, gt=0, description="Floor area in square meters (buildings)",
    )
    electricity_kwh: Optional[float] = Field(
        None, ge=0, description="Annual electricity consumption in kWh",
    )
    natural_gas_kwh: Optional[float] = Field(
        None, ge=0, description="Annual natural gas consumption in kWh",
    )
    steam_kwh: Optional[float] = Field(
        None, ge=0, description="Annual steam/district heating in kWh",
    )
    cooling_kwh: Optional[float] = Field(
        None, ge=0, description="Annual district cooling in kWh",
    )
    climate_zone: Optional[str] = Field(
        None, description="ASHRAE climate zone (e.g. 1A_very_hot_humid)",
    )
    country_code: Optional[str] = Field(
        None, description="ISO 3166-1 alpha-2 country code or eGRID subregion",
    )
    fuel_type: Optional[str] = Field(
        None,
        description="Fuel type for vehicles/equipment (petrol, diesel, etc.)",
    )
    annual_distance_km: Optional[float] = Field(
        None, ge=0, description="Annual distance driven in km (vehicles)",
    )
    fuel_consumed_litres: Optional[float] = Field(
        None, ge=0, description="Annual fuel consumed in litres (vehicles/equipment)",
    )
    fleet_count: Optional[int] = Field(
        None, ge=1, description="Number of vehicles in fleet",
    )
    operating_hours: Optional[float] = Field(
        None, ge=0, description="Annual operating hours (equipment)",
    )
    load_factor: Optional[float] = Field(
        None, ge=0, le=1.0, description="Equipment load factor (0.0-1.0)",
    )
    power_kw: Optional[float] = Field(
        None, ge=0, description="Rated power in kW (IT assets/equipment)",
    )
    pue: Optional[float] = Field(
        None, ge=1.0, le=5.0, description="Power Usage Effectiveness (IT assets)",
    )
    hours_per_year: Optional[float] = Field(
        None, ge=0, le=8784, description="Hours of operation per year (IT assets)",
    )
    quantity: Optional[int] = Field(
        None, ge=1, description="Number of IT asset units",
    )
    occupancy_rate: Optional[float] = Field(
        None, ge=0, le=1.0, description="Building occupancy rate (0.0-1.0)",
    )
    vacancy_adjusted: Optional[bool] = Field(
        False, description="Whether to apply vacancy adjustment to base load",
    )
    lease_revenue_usd: Optional[float] = Field(
        None, ge=0, description="Annual lease revenue in USD (spend-based)",
    )
    naics_code: Optional[str] = Field(
        None, description="NAICS code for spend-based EEIO factor lookup",
    )
    allocation_method: Optional[str] = Field(
        None,
        description="Allocation method: floor_area, headcount, revenue, time_based, equal_split, full",
    )
    allocation_factor: Optional[float] = Field(
        None, ge=0, le=1.0, description="Allocation factor (0.0-1.0)",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional asset-specific metadata",
    )


class FullPipelineRequest(BaseModel):
    """
    Request model for full pipeline downstream leased assets emissions calculation.

    Accepts a portfolio of owned-and-leased-out assets across all four asset
    categories (buildings, vehicles, equipment, IT assets). Each asset can
    specify its own calculation method, or a default method is applied
    based on data availability.

    Attributes:
        assets: List of asset items to calculate emissions for
        org_id: Organization identifier
        reporting_year: Reporting year (e.g. 2024)
        region: Default region/country for grid factor lookup
        consolidation_approach: GHG Protocol consolidation approach
    """

    assets: List[AssetItem] = Field(
        ...,
        min_items=1,
        max_items=10000,
        description="List of leased-out assets to calculate emissions for",
    )
    org_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Organization identifier",
    )
    reporting_year: int = Field(
        ...,
        ge=2000,
        le=2100,
        description="Reporting year",
    )
    region: Optional[str] = Field(
        None,
        description="Default region or country code for grid factor lookup",
    )
    consolidation_approach: str = Field(
        "operational_control",
        description="GHG Protocol consolidation: operational_control, financial_control, equity_share",
    )


class AssetSpecificBuildingRequest(BaseModel):
    """
    Request model for asset-specific building calculation.

    Uses metered energy consumption data collected from tenants for
    leased buildings. Applies grid emission factors for electricity
    and fuel emission factors for natural gas, steam, and cooling.

    Attributes:
        building_type: Type of building (office, retail, warehouse, etc.)
        floor_area_sqm: Total floor area in square meters
        electricity_kwh: Metered electricity consumption in kWh
        natural_gas_kwh: Metered natural gas consumption in kWh
        steam_kwh: Metered steam/district heating consumption in kWh
        cooling_kwh: Metered district cooling consumption in kWh
        climate_zone: ASHRAE climate zone
        country_code: ISO 3166-1 alpha-2 country code
        occupancy_rate: Building occupancy rate (0.0-1.0)
        vacancy_adjusted: Whether to apply vacancy base-load adjustment
        allocation_method: Emissions allocation method
        allocation_factor: Fraction allocated to leased portion
    """

    building_type: str = Field(
        ...,
        description=(
            "Building type: office, retail, warehouse, industrial, "
            "data_center, hospital, hotel, mixed_use"
        ),
    )
    floor_area_sqm: float = Field(
        ..., gt=0, description="Total building floor area in square meters",
    )
    electricity_kwh: Optional[float] = Field(
        None, ge=0, description="Annual metered electricity consumption in kWh",
    )
    natural_gas_kwh: Optional[float] = Field(
        None, ge=0, description="Annual metered natural gas consumption in kWh",
    )
    steam_kwh: Optional[float] = Field(
        None, ge=0, description="Annual metered steam/district heating in kWh",
    )
    cooling_kwh: Optional[float] = Field(
        None, ge=0, description="Annual metered district cooling in kWh",
    )
    climate_zone: Optional[str] = Field(
        None,
        description="ASHRAE climate zone (1A_very_hot_humid through 8_subarctic)",
    )
    country_code: str = Field(
        "US",
        min_length=2,
        max_length=10,
        description="ISO 3166-1 alpha-2 country code or eGRID subregion",
    )
    occupancy_rate: Optional[float] = Field(
        None, ge=0, le=1.0, description="Building occupancy rate (0.0=vacant, 1.0=fully occupied)",
    )
    vacancy_adjusted: bool = Field(
        False,
        description="Apply vacancy base-load adjustment using building type default",
    )
    allocation_method: str = Field(
        "floor_area",
        description="Allocation method: floor_area, headcount, revenue, time_based, equal_split, full",
    )
    allocation_factor: float = Field(
        1.0, ge=0, le=1.0, description="Fraction of emissions allocated to leased portion",
    )


class AssetSpecificVehicleRequest(BaseModel):
    """
    Request model for asset-specific vehicle fleet calculation.

    Calculates emissions from vehicles owned by reporter and leased
    to others using distance-based or fuel-based methods.

    Attributes:
        vehicle_type: Type of vehicle (small_car, medium_car, etc.)
        fuel_type: Fuel type (petrol, diesel, hybrid, etc.)
        annual_distance_km: Annual distance driven in km (distance-based)
        fuel_consumed_litres: Annual fuel consumed in litres (fuel-based)
        fleet_count: Number of vehicles in fleet
        country_code: Country for grid factor (EVs) or regional factor
    """

    vehicle_type: str = Field(
        ...,
        description=(
            "Vehicle type: small_car, medium_car, large_car, suv, "
            "light_van, heavy_van, light_truck, heavy_truck"
        ),
    )
    fuel_type: str = Field(
        ...,
        description=(
            "Fuel type: petrol, diesel, hybrid, phev, ev, lpg, cng"
        ),
    )
    annual_distance_km: Optional[float] = Field(
        None, ge=0, description="Annual distance driven per vehicle in km",
    )
    fuel_consumed_litres: Optional[float] = Field(
        None, ge=0, description="Annual fuel consumed per vehicle in litres",
    )
    fleet_count: int = Field(
        1, ge=1, le=100000, description="Number of vehicles in the leased fleet",
    )
    country_code: str = Field(
        "US",
        min_length=2,
        max_length=10,
        description="Country code for regional emission factors",
    )


class AssetSpecificEquipmentRequest(BaseModel):
    """
    Request model for asset-specific equipment calculation.

    Calculates emissions from leased equipment (generators, compressors,
    forklifts, etc.) using rated power, operating hours, and load factor.

    Attributes:
        equipment_type: Type of equipment (generator, compressor, etc.)
        fuel_type: Fuel type (diesel, natural_gas, etc.)
        operating_hours: Annual operating hours
        load_factor: Equipment load factor (0.0-1.0)
        rated_power_kw: Rated power output in kW
        fuel_consumed_litres: Direct fuel consumption in litres
        count: Number of equipment units
    """

    equipment_type: str = Field(
        ...,
        description=(
            "Equipment type: generator, compressor, pump, "
            "forklift, crane, hvac_unit"
        ),
    )
    fuel_type: str = Field(
        ...,
        description="Fuel type: diesel, natural_gas, petrol, lpg, electric",
    )
    operating_hours: float = Field(
        ..., ge=0, le=8784, description="Annual operating hours",
    )
    load_factor: float = Field(
        0.5, ge=0, le=1.0, description="Equipment load factor (0.0-1.0)",
    )
    rated_power_kw: Optional[float] = Field(
        None, ge=0, description="Rated power output in kW",
    )
    fuel_consumed_litres: Optional[float] = Field(
        None, ge=0, description="Direct fuel consumption in litres (overrides power calc)",
    )
    count: int = Field(
        1, ge=1, le=10000, description="Number of equipment units",
    )


class AssetSpecificITRequest(BaseModel):
    """
    Request model for asset-specific IT infrastructure calculation.

    Calculates emissions from leased IT assets (servers, storage,
    networking, etc.) using power draw, PUE, and utilization hours.

    Attributes:
        it_asset_type: Type of IT asset (server, storage, network_switch, etc.)
        power_kw: Rated or measured power draw in kW per unit
        pue: Power Usage Effectiveness for the hosting facility
        hours_per_year: Hours of operation per year
        quantity: Number of units
        country_code: Country for grid emission factor lookup
    """

    it_asset_type: str = Field(
        ...,
        description=(
            "IT asset type: server, storage, network_switch, "
            "router, ups, cooling_unit, workstation"
        ),
    )
    power_kw: float = Field(
        ..., ge=0, description="Power draw per unit in kW",
    )
    pue: float = Field(
        1.58, ge=1.0, le=5.0,
        description="Power Usage Effectiveness (1.0=ideal, industry avg ~1.58)",
    )
    hours_per_year: float = Field(
        8760.0, ge=0, le=8784,
        description="Hours of operation per year",
    )
    quantity: int = Field(
        1, ge=1, le=100000, description="Number of IT asset units",
    )
    country_code: str = Field(
        "US",
        min_length=2,
        max_length=10,
        description="Country code for grid emission factor lookup",
    )


class AverageDataRequest(BaseModel):
    """
    Request model for average-data (benchmark-based) calculation.

    Uses published EUI benchmarks by building type and climate zone
    or average emission factors by vehicle/equipment type when
    metered data is not available from tenants.

    Attributes:
        asset_category: Asset category (building, vehicle, equipment, it_asset)
        asset_type: Specific asset type within category
        floor_area_sqm: Floor area for buildings (required for buildings)
        annual_distance_km: Annual distance for vehicles
        operating_hours: Operating hours for equipment
        quantity: Number of units for IT assets
        region: Region or country code for benchmark selection
        climate_zone: Climate zone for building EUI benchmarks
    """

    asset_category: str = Field(
        ...,
        description="Asset category: building, vehicle, equipment, it_asset",
    )
    asset_type: str = Field(
        ...,
        description="Specific asset type within category",
    )
    floor_area_sqm: Optional[float] = Field(
        None, gt=0, description="Floor area in sqm (required for buildings)",
    )
    annual_distance_km: Optional[float] = Field(
        None, ge=0, description="Annual distance in km (vehicles)",
    )
    operating_hours: Optional[float] = Field(
        None, ge=0, description="Annual operating hours (equipment)",
    )
    quantity: Optional[int] = Field(
        None, ge=1, description="Number of units (IT assets)",
    )
    region: str = Field(
        "US",
        description="Region or country code for benchmark selection",
    )
    climate_zone: Optional[str] = Field(
        None,
        description="ASHRAE climate zone for building EUI benchmark",
    )


class SpendBasedRequest(BaseModel):
    """
    Request model for spend-based (EEIO) calculation.

    Uses Environmentally Extended Input-Output factors by NAICS code
    to convert lease revenue into estimated emissions. Applies CPI
    deflation to base year before factor application.

    Attributes:
        lease_revenue: Annual lease revenue
        naics_code: NAICS industry code for EEIO factor
        currency: ISO 4217 currency code
        reporting_year: Year for CPI deflation
    """

    lease_revenue: float = Field(
        ..., gt=0, description="Annual lease revenue in specified currency",
    )
    naics_code: str = Field(
        ...,
        min_length=4,
        max_length=10,
        description="NAICS code for EEIO factor selection (e.g. 531110)",
    )
    currency: str = Field(
        "USD",
        min_length=3,
        max_length=3,
        description="ISO 4217 currency code",
    )
    reporting_year: int = Field(
        2024,
        ge=2000,
        le=2100,
        description="Reporting year for CPI deflation",
    )


class HybridAssetItem(BaseModel):
    """Individual asset item within a hybrid calculation request."""

    asset_category: str = Field(
        ...,
        description="Asset category: building, vehicle, equipment, it_asset",
    )
    asset_type: str = Field(
        ...,
        description="Specific asset type within category",
    )
    preferred_method: Optional[str] = Field(
        None,
        description="Preferred method: asset_specific, average_data, spend_based",
    )
    data: Dict[str, Any] = Field(
        ...,
        description="Asset-specific data fields (varies by category and method)",
    )


class HybridRequest(BaseModel):
    """
    Request model for hybrid (multi-method) calculation.

    Combines asset-specific, average-data, and spend-based methods
    across a mixed portfolio. Each asset can specify a preferred method;
    the engine uses a waterfall approach: asset-specific if metered data
    is available, then average-data, then spend-based as fallback.

    Attributes:
        assets: List of assets with method preferences
        org_id: Organization identifier
        reporting_year: Reporting year
        default_region: Default region for factor lookup
    """

    assets: List[HybridAssetItem] = Field(
        ...,
        min_items=1,
        max_items=10000,
        description="List of assets with method preferences",
    )
    org_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Organization identifier",
    )
    reporting_year: int = Field(
        ..., ge=2000, le=2100, description="Reporting year",
    )
    default_region: str = Field(
        "US",
        description="Default region for emission factor lookup",
    )


class BatchCalculateRequest(BaseModel):
    """
    Request model for batch downstream leased asset calculations.

    Processes up to 10,000 assets in a single request with parallel
    execution and per-asset error isolation.

    Attributes:
        assets: List of asset data dictionaries
        org_id: Organization identifier
        reporting_year: Reporting year
        default_method: Default calculation method for all assets
    """

    assets: List[Dict[str, Any]] = Field(
        ...,
        min_items=1,
        max_items=10000,
        description="List of asset data dictionaries (each must include asset_category and asset_type)",
    )
    org_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Organization identifier",
    )
    reporting_year: int = Field(
        ..., ge=2000, le=2100, description="Reporting year",
    )
    default_method: str = Field(
        "average_data",
        description="Default calculation method: asset_specific, average_data, spend_based",
    )


class PortfolioAnalysisRequest(BaseModel):
    """
    Request model for portfolio-level analysis.

    Analyzes the full portfolio of downstream leased assets including
    category breakdown, top emitters, allocation summary, and
    year-over-year comparison.

    Attributes:
        assets: Full asset portfolio list
        org_id: Organization identifier
        reporting_year: Reporting year
        comparison_year: Optional previous year for YoY analysis
        allocation_method: Default allocation method across portfolio
    """

    assets: List[AssetItem] = Field(
        ...,
        min_items=1,
        max_items=10000,
        description="Full portfolio of leased-out assets",
    )
    org_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Organization identifier",
    )
    reporting_year: int = Field(
        ..., ge=2000, le=2100, description="Reporting year",
    )
    comparison_year: Optional[int] = Field(
        None, ge=2000, le=2100, description="Previous year for YoY comparison",
    )
    allocation_method: str = Field(
        "floor_area",
        description="Default portfolio allocation method",
    )


class ComplianceCheckRequest(BaseModel):
    """
    Request model for multi-framework compliance checking.

    Checks calculation results against selected regulatory frameworks
    for completeness, boundary correctness, method appropriateness,
    allocation transparency, and disclosure requirements.

    Attributes:
        frameworks: List of framework identifiers to check against
        calculation_results: List of calculation result dicts to validate
        consolidation_approach: GHG Protocol consolidation approach used
        allocation_disclosed: Whether allocation method is disclosed
        vacancy_handling_disclosed: Whether vacancy handling is documented
    """

    frameworks: List[str] = Field(
        ...,
        min_items=1,
        description=(
            "Frameworks to check: ghg_protocol, iso_14064, csrd_esrs, "
            "cdp, sbti, sb_253, gri"
        ),
    )
    calculation_results: List[Dict[str, Any]] = Field(
        ...,
        min_items=1,
        description="Calculation results to check for compliance",
    )
    consolidation_approach: str = Field(
        "operational_control",
        description="GHG Protocol consolidation approach used",
    )
    allocation_disclosed: bool = Field(
        False,
        description="Whether allocation methodology has been disclosed",
    )
    vacancy_handling_disclosed: bool = Field(
        False,
        description="Whether vacancy handling approach is documented",
    )


class AllocationRequest(BaseModel):
    """
    Request model for emissions allocation calculation.

    Calculates how total building/asset emissions should be allocated
    between the lessor and tenants, handling common areas, vacant
    space, and shared services.

    Attributes:
        total_co2e_kg: Total emissions before allocation
        allocation_method: Allocation method to apply
        tenant_shares: List of tenant shares (fraction or absolute)
        common_area_pct: Percentage of common area
        vacancy_pct: Current vacancy percentage
        building_type: Building type for vacancy base-load lookup
    """

    total_co2e_kg: float = Field(
        ..., ge=0, description="Total emissions before allocation in kgCO2e",
    )
    allocation_method: str = Field(
        "floor_area",
        description="Allocation method: floor_area, headcount, revenue, time_based, equal_split, full",
    )
    tenant_shares: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="List of tenant dicts with name and share (fraction 0-1)",
    )
    common_area_pct: float = Field(
        0.15, ge=0, le=1.0, description="Common area as fraction of total (0.0-1.0)",
    )
    vacancy_pct: float = Field(
        0.0, ge=0, le=1.0, description="Current vacancy rate as fraction (0.0-1.0)",
    )
    building_type: Optional[str] = Field(
        None,
        description="Building type for vacancy base-load lookup",
    )


# ============================================================================
# RESPONSE MODELS
# ============================================================================


class CalculateResponse(BaseModel):
    """Response model for full pipeline or general calculation."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    method: str = Field(..., description="Calculation method applied")
    total_co2e_kg: float = Field(..., description="Total CO2e emissions in kg")
    scope3_category: int = Field(
        default=13, description="Scope 3 category number"
    )
    asset_count: int = Field(..., description="Number of assets processed")
    building_co2e_kg: Optional[float] = Field(
        None, description="Building subtotal CO2e (kg)"
    )
    vehicle_co2e_kg: Optional[float] = Field(
        None, description="Vehicle subtotal CO2e (kg)"
    )
    equipment_co2e_kg: Optional[float] = Field(
        None, description="Equipment subtotal CO2e (kg)"
    )
    it_asset_co2e_kg: Optional[float] = Field(
        None, description="IT asset subtotal CO2e (kg)"
    )
    dqi_score: Optional[float] = Field(
        None, description="Data quality indicator score (1.0-5.0)"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance chain hash"
    )
    calculated_at: str = Field(
        ..., description="ISO 8601 calculation timestamp"
    )


class BuildingResponse(BaseModel):
    """Response model for asset-specific building calculation."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    building_type: str = Field(..., description="Building type")
    floor_area_sqm: float = Field(..., description="Floor area in sqm")
    method: str = Field(
        default="asset_specific", description="Calculation method"
    )
    electricity_co2e_kg: float = Field(
        ..., description="Emissions from electricity (kgCO2e)"
    )
    gas_co2e_kg: float = Field(
        ..., description="Emissions from natural gas (kgCO2e)"
    )
    steam_co2e_kg: float = Field(
        ..., description="Emissions from steam/heating (kgCO2e)"
    )
    cooling_co2e_kg: float = Field(
        ..., description="Emissions from cooling (kgCO2e)"
    )
    total_co2e_kg: float = Field(
        ..., description="Total building CO2e (kgCO2e)"
    )
    allocated_co2e_kg: float = Field(
        ..., description="CO2e after allocation factor (kgCO2e)"
    )
    allocation_method: str = Field(
        ..., description="Allocation method applied"
    )
    allocation_factor: float = Field(
        ..., description="Allocation factor (0.0-1.0)"
    )
    eui_kwh_per_sqm: Optional[float] = Field(
        None, description="Energy Use Intensity if benchmark used"
    )
    grid_factor_used: Optional[float] = Field(
        None, description="Grid emission factor applied (kgCO2e/kWh)"
    )
    vacancy_adjustment: Optional[float] = Field(
        None, description="Vacancy base-load adjustment applied"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance chain hash"
    )
    calculated_at: str = Field(
        ..., description="ISO 8601 calculation timestamp"
    )


class VehicleResponse(BaseModel):
    """Response model for asset-specific vehicle fleet calculation."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    vehicle_type: str = Field(..., description="Vehicle type")
    fuel_type: str = Field(..., description="Fuel type")
    method: str = Field(
        default="asset_specific", description="Calculation method"
    )
    fleet_count: int = Field(..., description="Number of vehicles")
    per_vehicle_co2e_kg: float = Field(
        ..., description="CO2e per vehicle (kgCO2e)"
    )
    ttw_co2e_kg: float = Field(
        ..., description="Tank-to-wheel emissions (kgCO2e)"
    )
    wtt_co2e_kg: float = Field(
        ..., description="Well-to-tank emissions (kgCO2e)"
    )
    total_co2e_kg: float = Field(
        ..., description="Total fleet CO2e (kgCO2e)"
    )
    ef_per_km: Optional[float] = Field(
        None, description="Emission factor used (kgCO2e/km)"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance chain hash"
    )
    calculated_at: str = Field(
        ..., description="ISO 8601 calculation timestamp"
    )


class EquipmentResponse(BaseModel):
    """Response model for asset-specific equipment calculation."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    equipment_type: str = Field(..., description="Equipment type")
    fuel_type: str = Field(..., description="Fuel type")
    method: str = Field(
        default="asset_specific", description="Calculation method"
    )
    count: int = Field(..., description="Number of equipment units")
    operating_hours: float = Field(..., description="Annual operating hours")
    load_factor: float = Field(..., description="Load factor applied")
    per_unit_co2e_kg: float = Field(
        ..., description="CO2e per unit (kgCO2e)"
    )
    total_co2e_kg: float = Field(
        ..., description="Total equipment CO2e (kgCO2e)"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance chain hash"
    )
    calculated_at: str = Field(
        ..., description="ISO 8601 calculation timestamp"
    )


class ITAssetResponse(BaseModel):
    """Response model for asset-specific IT infrastructure calculation."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    it_asset_type: str = Field(..., description="IT asset type")
    method: str = Field(
        default="asset_specific", description="Calculation method"
    )
    quantity: int = Field(..., description="Number of IT asset units")
    power_kw: float = Field(..., description="Power draw per unit (kW)")
    pue: float = Field(..., description="PUE applied")
    annual_energy_kwh: float = Field(
        ..., description="Total annual energy (kWh) including PUE"
    )
    per_unit_co2e_kg: float = Field(
        ..., description="CO2e per unit (kgCO2e)"
    )
    total_co2e_kg: float = Field(
        ..., description="Total IT asset CO2e (kgCO2e)"
    )
    grid_factor_used: float = Field(
        ..., description="Grid emission factor applied (kgCO2e/kWh)"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance chain hash"
    )
    calculated_at: str = Field(
        ..., description="ISO 8601 calculation timestamp"
    )


class AverageDataResponse(BaseModel):
    """Response model for average-data (benchmark) calculation."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    asset_category: str = Field(..., description="Asset category")
    asset_type: str = Field(..., description="Asset type")
    method: str = Field(
        default="average_data", description="Calculation method"
    )
    benchmark_used: str = Field(
        ..., description="Benchmark or EUI value used"
    )
    total_co2e_kg: float = Field(
        ..., description="Total CO2e (kgCO2e)"
    )
    dqi_score: float = Field(
        ..., description="Data quality score (1.0-5.0, higher = lower quality)"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance chain hash"
    )
    calculated_at: str = Field(
        ..., description="ISO 8601 calculation timestamp"
    )


class SpendBasedResponse(BaseModel):
    """Response model for spend-based (EEIO) calculation."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    method: str = Field(
        default="spend_based", description="Calculation method"
    )
    naics_code: str = Field(..., description="NAICS code used")
    lease_revenue_original: float = Field(
        ..., description="Original lease revenue amount"
    )
    currency: str = Field(..., description="Currency code")
    lease_revenue_usd: float = Field(
        ..., description="Revenue converted to USD"
    )
    cpi_deflator: Optional[float] = Field(
        None, description="CPI deflator applied"
    )
    eeio_factor: float = Field(
        ..., description="EEIO factor used (kgCO2e/USD)"
    )
    total_co2e_kg: float = Field(
        ..., description="Total CO2e (kgCO2e)"
    )
    dqi_score: float = Field(
        ..., description="Data quality score (spend-based = 4.0-5.0)"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance chain hash"
    )
    calculated_at: str = Field(
        ..., description="ISO 8601 calculation timestamp"
    )


class BatchCalculateResponse(BaseModel):
    """Response model for batch calculation."""

    batch_id: str = Field(..., description="Unique batch UUID")
    results: List[Dict[str, Any]] = Field(
        ..., description="Individual asset calculation results"
    )
    total_co2e_kg: float = Field(
        ..., description="Total CO2e across all assets in batch"
    )
    count: int = Field(..., description="Number of successful calculations")
    errors: List[Dict[str, Any]] = Field(
        default_factory=list, description="Per-asset error details"
    )
    reporting_year: int = Field(..., description="Reporting year")


class PortfolioAnalysisResponse(BaseModel):
    """Response model for portfolio analysis."""

    analysis_id: str = Field(..., description="Unique analysis UUID")
    total_co2e_kg: float = Field(
        ..., description="Total portfolio CO2e (kgCO2e)"
    )
    by_category: Dict[str, float] = Field(
        ..., description="CO2e breakdown by asset category"
    )
    by_method: Dict[str, float] = Field(
        ..., description="CO2e breakdown by calculation method"
    )
    top_emitters: List[Dict[str, Any]] = Field(
        ..., description="Top 10 emitting assets"
    )
    allocation_summary: Dict[str, Any] = Field(
        ..., description="Allocation method distribution and totals"
    )
    yoy_change: Optional[Dict[str, Any]] = Field(
        None, description="Year-over-year change if comparison_year provided"
    )
    asset_count: int = Field(..., description="Total assets analyzed")
    dqi_avg: float = Field(
        ..., description="Average DQI score across portfolio"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )


class ComplianceCheckResponse(BaseModel):
    """Response model for compliance check."""

    results: List[Dict[str, Any]] = Field(
        ..., description="Per-framework compliance results"
    )
    overall_status: str = Field(
        ..., description="Overall compliance status (pass, fail, warning)"
    )
    overall_score: float = Field(
        ..., description="Overall compliance score (0.0-1.0)"
    )


class CalculationDetailResponse(BaseModel):
    """Response model for single calculation detail."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    method: str = Field(..., description="Calculation method")
    total_co2e_kg: float = Field(..., description="Total CO2e (kg)")
    asset_count: int = Field(..., description="Number of assets")
    details: Dict[str, Any] = Field(
        ..., description="Full calculation detail payload"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance chain hash"
    )
    calculated_at: str = Field(
        ..., description="ISO 8601 calculation timestamp"
    )


class CalculationListResponse(BaseModel):
    """Response model for paginated calculation listing."""

    calculations: List[Dict[str, Any]] = Field(
        ..., description="Calculation summaries"
    )
    count: int = Field(..., description="Total matching calculations")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Page size")


class DeleteResponse(BaseModel):
    """Response model for soft deletion."""

    calculation_id: str = Field(..., description="Deleted calculation UUID")
    deleted: bool = Field(..., description="Whether deletion succeeded")
    message: str = Field(..., description="Human-readable status message")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service health status")
    agent_id: str = Field(..., description="Agent identifier")
    version: str = Field(..., description="Agent version")
    uptime_seconds: float = Field(
        ..., description="Seconds since service start"
    )


# ============================================================================
# MODULE-LEVEL TRACKING
# ============================================================================

_start_time: datetime = datetime.utcnow()


# ============================================================================
# ENDPOINTS - CALCULATIONS (10)
# ============================================================================


@router.post(
    "/calculate",
    response_model=CalculateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate downstream leased assets emissions (full pipeline)",
    description=(
        "Calculate Scope 3 Category 13 emissions for a portfolio of assets "
        "owned by the reporter and leased to others through the full 10-stage "
        "pipeline. Accepts buildings, vehicles, equipment, and IT assets with "
        "method selection per asset. Returns deterministic results with "
        "SHA-256 provenance hash."
    ),
)
async def calculate_emissions(
    request: FullPipelineRequest,
    service=Depends(get_service),
) -> CalculateResponse:
    """
    Calculate downstream leased assets emissions through the full pipeline.

    Args:
        request: Full pipeline request with asset portfolio
        service: DownstreamLeasedAssetsService instance

    Returns:
        CalculateResponse with total emissions and category breakdown

    Raises:
        HTTPException: 400 for validation errors, 500 for processing failures
    """
    try:
        logger.info(
            f"Calculating DLA emissions: org={request.org_id}, "
            f"year={request.reporting_year}, assets={len(request.assets)}"
        )

        result = await service.calculate(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return CalculateResponse(
            calculation_id=calculation_id,
            method=result.get("method", "hybrid"),
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            scope3_category=13,
            asset_count=result.get("asset_count", len(request.assets)),
            building_co2e_kg=result.get("building_co2e_kg"),
            vehicle_co2e_kg=result.get("vehicle_co2e_kg"),
            equipment_co2e_kg=result.get("equipment_co2e_kg"),
            it_asset_co2e_kg=result.get("it_asset_co2e_kg"),
            dqi_score=result.get("dqi_score"),
            provenance_hash=result.get("provenance_hash", ""),
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
    "/calculate/asset-specific",
    response_model=CalculateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate asset-specific emissions (general)",
    description=(
        "Calculate emissions for a single downstream leased asset using "
        "the asset-specific method with metered energy or fuel data. "
        "Automatically routes to the correct calculator based on asset_category."
    ),
)
async def calculate_asset_specific(
    request: AssetItem,
    service=Depends(get_service),
) -> CalculateResponse:
    """
    Calculate asset-specific emissions for a single asset.

    Args:
        request: Asset item with metered data
        service: DownstreamLeasedAssetsService instance

    Returns:
        CalculateResponse with asset-specific emissions

    Raises:
        HTTPException: 400 for validation errors, 500 for processing failures
    """
    try:
        logger.info(
            f"Calculating asset-specific DLA emissions: "
            f"category={request.asset_category}, type={request.asset_type}"
        )

        result = await service.calculate_asset_specific(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return CalculateResponse(
            calculation_id=calculation_id,
            method=result.get("method", "asset_specific"),
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            scope3_category=13,
            asset_count=1,
            building_co2e_kg=result.get("building_co2e_kg"),
            vehicle_co2e_kg=result.get("vehicle_co2e_kg"),
            equipment_co2e_kg=result.get("equipment_co2e_kg"),
            it_asset_co2e_kg=result.get("it_asset_co2e_kg"),
            dqi_score=result.get("dqi_score"),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error(f"Validation error in calculate_asset_specific: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in calculate_asset_specific: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Asset-specific calculation failed",
        )


@router.post(
    "/calculate/asset-specific/building",
    response_model=BuildingResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate building emissions (asset-specific)",
    description=(
        "Calculate emissions for a specific leased-out building using "
        "metered energy data. Applies grid emission factors for electricity, "
        "fuel emission factors for gas, and steam/cooling factors. Supports "
        "occupancy adjustment, vacancy base-load handling, and allocation "
        "between lessor and tenants."
    ),
)
async def calculate_building_emissions(
    request: AssetSpecificBuildingRequest,
    service=Depends(get_service),
) -> BuildingResponse:
    """
    Calculate building emissions using metered tenant energy data.

    Args:
        request: Building calculation request with energy consumption
        service: DownstreamLeasedAssetsService instance

    Returns:
        BuildingResponse with energy source breakdown and allocation

    Raises:
        HTTPException: 400 for validation errors, 500 for processing failures
    """
    try:
        logger.info(
            f"Calculating building DLA emissions: type={request.building_type}, "
            f"area={request.floor_area_sqm}sqm, "
            f"country={request.country_code}"
        )

        result = await service.calculate_building(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return BuildingResponse(
            calculation_id=calculation_id,
            building_type=result.get("building_type", request.building_type),
            floor_area_sqm=result.get("floor_area_sqm", request.floor_area_sqm),
            method=result.get("method", "asset_specific"),
            electricity_co2e_kg=result.get("electricity_co2e_kg", 0.0),
            gas_co2e_kg=result.get("gas_co2e_kg", 0.0),
            steam_co2e_kg=result.get("steam_co2e_kg", 0.0),
            cooling_co2e_kg=result.get("cooling_co2e_kg", 0.0),
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            allocated_co2e_kg=result.get("allocated_co2e_kg", 0.0),
            allocation_method=result.get(
                "allocation_method", request.allocation_method
            ),
            allocation_factor=result.get(
                "allocation_factor", request.allocation_factor
            ),
            eui_kwh_per_sqm=result.get("eui_kwh_per_sqm"),
            grid_factor_used=result.get("grid_factor_used"),
            vacancy_adjustment=result.get("vacancy_adjustment"),
            provenance_hash=result.get("provenance_hash", ""),
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
    "/calculate/asset-specific/vehicle",
    response_model=VehicleResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate vehicle fleet emissions (asset-specific)",
    description=(
        "Calculate emissions from leased-out vehicle fleets using "
        "distance-based (per-km) or fuel-based (per-litre) methods. "
        "Supports 8 vehicle types and 7 fuel types with DEFRA 2024 "
        "emission factors. Returns TTW and WTT breakdowns."
    ),
)
async def calculate_vehicle_emissions(
    request: AssetSpecificVehicleRequest,
    service=Depends(get_service),
) -> VehicleResponse:
    """
    Calculate vehicle fleet emissions for leased-out vehicles.

    Args:
        request: Vehicle calculation request with distance or fuel data
        service: DownstreamLeasedAssetsService instance

    Returns:
        VehicleResponse with per-vehicle and fleet totals

    Raises:
        HTTPException: 400 for missing distance/fuel, 500 for failures
    """
    try:
        logger.info(
            f"Calculating vehicle DLA emissions: type={request.vehicle_type}, "
            f"fuel={request.fuel_type}, fleet={request.fleet_count}"
        )

        result = await service.calculate_vehicle(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return VehicleResponse(
            calculation_id=calculation_id,
            vehicle_type=result.get("vehicle_type", request.vehicle_type),
            fuel_type=result.get("fuel_type", request.fuel_type),
            method=result.get("method", "asset_specific"),
            fleet_count=result.get("fleet_count", request.fleet_count),
            per_vehicle_co2e_kg=result.get("per_vehicle_co2e_kg", 0.0),
            ttw_co2e_kg=result.get("ttw_co2e_kg", 0.0),
            wtt_co2e_kg=result.get("wtt_co2e_kg", 0.0),
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            ef_per_km=result.get("ef_per_km"),
            provenance_hash=result.get("provenance_hash", ""),
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
    "/calculate/asset-specific/equipment",
    response_model=EquipmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate equipment emissions (asset-specific)",
    description=(
        "Calculate emissions from leased-out equipment (generators, "
        "compressors, forklifts, etc.) using rated power, operating hours, "
        "and load factor. Supports 6 equipment types with fuel-based "
        "emission factors."
    ),
)
async def calculate_equipment_emissions(
    request: AssetSpecificEquipmentRequest,
    service=Depends(get_service),
) -> EquipmentResponse:
    """
    Calculate equipment emissions for leased-out equipment.

    Args:
        request: Equipment calculation request with operating parameters
        service: DownstreamLeasedAssetsService instance

    Returns:
        EquipmentResponse with per-unit and total emissions

    Raises:
        HTTPException: 400 for validation errors, 500 for processing failures
    """
    try:
        logger.info(
            f"Calculating equipment DLA emissions: type={request.equipment_type}, "
            f"fuel={request.fuel_type}, hours={request.operating_hours}, "
            f"count={request.count}"
        )

        result = await service.calculate_equipment(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return EquipmentResponse(
            calculation_id=calculation_id,
            equipment_type=result.get("equipment_type", request.equipment_type),
            fuel_type=result.get("fuel_type", request.fuel_type),
            method=result.get("method", "asset_specific"),
            count=result.get("count", request.count),
            operating_hours=result.get("operating_hours", request.operating_hours),
            load_factor=result.get("load_factor", request.load_factor),
            per_unit_co2e_kg=result.get("per_unit_co2e_kg", 0.0),
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error(
            f"Validation error in calculate_equipment_emissions: {e}"
        )
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
    "/calculate/asset-specific/it-asset",
    response_model=ITAssetResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate IT infrastructure emissions (asset-specific)",
    description=(
        "Calculate emissions from leased-out IT infrastructure (servers, "
        "storage, networking) using power draw, PUE, and utilization hours. "
        "Supports 7 IT asset types. PUE adjusts power for facility overhead "
        "(cooling, lighting, UPS losses)."
    ),
)
async def calculate_it_asset_emissions(
    request: AssetSpecificITRequest,
    service=Depends(get_service),
) -> ITAssetResponse:
    """
    Calculate IT infrastructure emissions for leased-out IT assets.

    Args:
        request: IT asset calculation request with power and PUE data
        service: DownstreamLeasedAssetsService instance

    Returns:
        ITAssetResponse with energy and emissions per unit

    Raises:
        HTTPException: 400 for validation errors, 500 for processing failures
    """
    try:
        logger.info(
            f"Calculating IT asset DLA emissions: type={request.it_asset_type}, "
            f"power={request.power_kw}kW, pue={request.pue}, "
            f"quantity={request.quantity}"
        )

        result = await service.calculate_it_asset(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return ITAssetResponse(
            calculation_id=calculation_id,
            it_asset_type=result.get("it_asset_type", request.it_asset_type),
            method=result.get("method", "asset_specific"),
            quantity=result.get("quantity", request.quantity),
            power_kw=result.get("power_kw", request.power_kw),
            pue=result.get("pue", request.pue),
            annual_energy_kwh=result.get("annual_energy_kwh", 0.0),
            per_unit_co2e_kg=result.get("per_unit_co2e_kg", 0.0),
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            grid_factor_used=result.get("grid_factor_used", 0.0),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error(
            f"Validation error in calculate_it_asset_emissions: {e}"
        )
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
    "/calculate/average-data",
    response_model=AverageDataResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate emissions using average data (benchmarks)",
    description=(
        "Calculate emissions using published benchmarks and average data "
        "when metered tenant energy data is not available. Uses EUI "
        "benchmarks (kWh/sqm/yr) for buildings by type and climate zone, "
        "average per-km factors for vehicles, and rated-power defaults "
        "for equipment and IT assets."
    ),
)
async def calculate_average_data(
    request: AverageDataRequest,
    service=Depends(get_service),
) -> AverageDataResponse:
    """
    Calculate emissions using average data benchmarks.

    Args:
        request: Average data request with asset category and size metrics
        service: DownstreamLeasedAssetsService instance

    Returns:
        AverageDataResponse with benchmark-based emissions

    Raises:
        HTTPException: 400 for invalid category, 500 for failures
    """
    try:
        logger.info(
            f"Calculating average-data DLA emissions: "
            f"category={request.asset_category}, type={request.asset_type}, "
            f"region={request.region}"
        )

        result = await service.calculate_average_data(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return AverageDataResponse(
            calculation_id=calculation_id,
            asset_category=result.get(
                "asset_category", request.asset_category
            ),
            asset_type=result.get("asset_type", request.asset_type),
            method=result.get("method", "average_data"),
            benchmark_used=result.get("benchmark_used", ""),
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            dqi_score=result.get("dqi_score", 3.0),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error(f"Validation error in calculate_average_data: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in calculate_average_data: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Average-data calculation failed",
        )


@router.post(
    "/calculate/spend-based",
    response_model=SpendBasedResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate spend-based emissions (EEIO)",
    description=(
        "Calculate emissions using spend-based EEIO factors by NAICS code. "
        "Converts lease revenue to estimated emissions with CPI deflation "
        "to base year and currency conversion. Supports 10 NAICS leasing/ "
        "rental industry codes. Lowest data quality tier."
    ),
)
async def calculate_spend_based(
    request: SpendBasedRequest,
    service=Depends(get_service),
) -> SpendBasedResponse:
    """
    Calculate spend-based emissions using EEIO factors.

    Args:
        request: Spend-based request with lease revenue and NAICS code
        service: DownstreamLeasedAssetsService instance

    Returns:
        SpendBasedResponse with EEIO-based emissions

    Raises:
        HTTPException: 400 for invalid NAICS, 500 for failures
    """
    try:
        logger.info(
            f"Calculating spend-based DLA emissions: "
            f"naics={request.naics_code}, revenue={request.lease_revenue} "
            f"{request.currency}"
        )

        result = await service.calculate_spend_based(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return SpendBasedResponse(
            calculation_id=calculation_id,
            method=result.get("method", "spend_based"),
            naics_code=result.get("naics_code", request.naics_code),
            lease_revenue_original=result.get(
                "lease_revenue_original", request.lease_revenue
            ),
            currency=result.get("currency", request.currency),
            lease_revenue_usd=result.get("lease_revenue_usd", 0.0),
            cpi_deflator=result.get("cpi_deflator"),
            eeio_factor=result.get("eeio_factor", 0.0),
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            dqi_score=result.get("dqi_score", 4.0),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error(f"Validation error in calculate_spend_based: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in calculate_spend_based: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Spend-based calculation failed",
        )


@router.post(
    "/calculate/hybrid",
    response_model=CalculateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate hybrid emissions (multi-method waterfall)",
    description=(
        "Calculate emissions using a hybrid approach that combines "
        "asset-specific, average-data, and spend-based methods. Each "
        "asset uses the highest-quality method for which data is available: "
        "asset-specific (if metered data exists), average-data (if size "
        "metrics available), or spend-based (fallback). The waterfall "
        "ensures maximum data quality across a mixed portfolio."
    ),
)
async def calculate_hybrid(
    request: HybridRequest,
    service=Depends(get_service),
) -> CalculateResponse:
    """
    Calculate hybrid emissions using waterfall method selection.

    Args:
        request: Hybrid request with assets and method preferences
        service: DownstreamLeasedAssetsService instance

    Returns:
        CalculateResponse with aggregated hybrid emissions

    Raises:
        HTTPException: 400 for validation errors, 500 for processing failures
    """
    try:
        logger.info(
            f"Calculating hybrid DLA emissions: org={request.org_id}, "
            f"year={request.reporting_year}, assets={len(request.assets)}"
        )

        result = await service.calculate_hybrid(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return CalculateResponse(
            calculation_id=calculation_id,
            method=result.get("method", "hybrid"),
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            scope3_category=13,
            asset_count=result.get("asset_count", len(request.assets)),
            building_co2e_kg=result.get("building_co2e_kg"),
            vehicle_co2e_kg=result.get("vehicle_co2e_kg"),
            equipment_co2e_kg=result.get("equipment_co2e_kg"),
            it_asset_co2e_kg=result.get("it_asset_co2e_kg"),
            dqi_score=result.get("dqi_score"),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error(f"Validation error in calculate_hybrid: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error in calculate_hybrid: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Hybrid calculation failed",
        )


@router.post(
    "/calculate/batch",
    response_model=BatchCalculateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Batch calculate downstream leased assets emissions",
    description=(
        "Calculate emissions for up to 10,000 downstream leased assets "
        "in a single request with parallel execution and per-asset error "
        "isolation. Returns aggregated totals plus individual results "
        "and any per-asset errors."
    ),
)
async def calculate_batch_emissions(
    request: BatchCalculateRequest,
    service=Depends(get_service),
) -> BatchCalculateResponse:
    """
    Calculate batch downstream leased assets emissions.

    Args:
        request: Batch calculation request with asset list
        service: DownstreamLeasedAssetsService instance

    Returns:
        BatchCalculateResponse with aggregated and per-asset results

    Raises:
        HTTPException: 400 for validation errors, 500 for batch failures
    """
    try:
        logger.info(
            f"Calculating batch DLA emissions: {len(request.assets)} assets, "
            f"org={request.org_id}, year={request.reporting_year}"
        )

        result = await service.calculate_batch(request.dict())
        batch_id = result.get("batch_id", str(uuid.uuid4()))

        return BatchCalculateResponse(
            batch_id=batch_id,
            results=result.get("results", []),
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            count=result.get("count", 0),
            errors=result.get("errors", []),
            reporting_year=result.get(
                "reporting_year", request.reporting_year
            ),
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
    response_model=PortfolioAnalysisResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Portfolio analysis of downstream leased assets",
    description=(
        "Analyze the full portfolio of downstream leased assets with "
        "category breakdown, top emitter identification, allocation "
        "summary, and optional year-over-year comparison. Provides "
        "Pareto ranking of emission hotspots across the lessor portfolio."
    ),
)
async def analyze_portfolio(
    request: PortfolioAnalysisRequest,
    service=Depends(get_service),
) -> PortfolioAnalysisResponse:
    """
    Analyze portfolio of downstream leased assets.

    Args:
        request: Portfolio analysis request with full asset list
        service: DownstreamLeasedAssetsService instance

    Returns:
        PortfolioAnalysisResponse with category breakdown and insights

    Raises:
        HTTPException: 400 for validation errors, 500 for analysis failures
    """
    try:
        logger.info(
            f"Analyzing DLA portfolio: org={request.org_id}, "
            f"year={request.reporting_year}, assets={len(request.assets)}"
        )

        result = await service.analyze_portfolio(request.dict())
        analysis_id = result.get("analysis_id", str(uuid.uuid4()))

        return PortfolioAnalysisResponse(
            analysis_id=analysis_id,
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            by_category=result.get("by_category", {}),
            by_method=result.get("by_method", {}),
            top_emitters=result.get("top_emitters", []),
            allocation_summary=result.get("allocation_summary", {}),
            yoy_change=result.get("yoy_change"),
            asset_count=result.get("asset_count", len(request.assets)),
            dqi_avg=result.get("dqi_avg", 3.0),
            provenance_hash=result.get("provenance_hash", ""),
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
# ENDPOINTS - COMPLIANCE (1)
# ============================================================================


@router.post(
    "/compliance/check",
    response_model=ComplianceCheckResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Check multi-framework compliance",
    description=(
        "Check downstream leased assets calculation results against one or "
        "more regulatory frameworks. Validates completeness, consolidation "
        "approach, allocation transparency, vacancy handling documentation, "
        "and category-specific disclosure requirements. Supports GHG "
        "Protocol, ISO 14064, CSRD ESRS E1, CDP, SBTi, SB 253, and GRI 305."
    ),
)
async def check_compliance(
    request: ComplianceCheckRequest,
    service=Depends(get_service),
) -> ComplianceCheckResponse:
    """
    Check calculation compliance against regulatory frameworks.

    Args:
        request: Compliance check request with frameworks and results
        service: DownstreamLeasedAssetsService instance

    Returns:
        ComplianceCheckResponse with per-framework findings

    Raises:
        HTTPException: 400 for invalid frameworks, 500 for check failures
    """
    try:
        logger.info(
            f"Checking DLA compliance: {len(request.frameworks)} frameworks, "
            f"{len(request.calculation_results)} results"
        )

        result = await service.check_compliance(request.dict())

        return ComplianceCheckResponse(
            results=result.get("results", []),
            overall_status=result.get("overall_status", "unknown"),
            overall_score=result.get("overall_score", 0.0),
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


# ============================================================================
# ENDPOINTS - CALCULATION CRUD (3)
# ============================================================================


@router.get(
    "/calculations/{calculation_id}",
    response_model=CalculationDetailResponse,
    summary="Get calculation detail",
    description=(
        "Retrieve detailed information for a specific downstream leased "
        "assets calculation including full input/output payload, per-asset "
        "breakdown, allocation details, and provenance hash."
    ),
)
async def get_calculation_detail(
    calculation_id: str = Path(..., description="Calculation UUID"),
    service=Depends(get_service),
) -> CalculationDetailResponse:
    """
    Get detailed information for a specific calculation.

    Args:
        calculation_id: Calculation UUID
        service: DownstreamLeasedAssetsService instance

    Returns:
        CalculationDetailResponse with full calculation data

    Raises:
        HTTPException: 404 if not found, 500 for retrieval failures
    """
    try:
        logger.info(f"Getting DLA calculation detail: {calculation_id}")

        result = await service.get_calculation(calculation_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Calculation {calculation_id} not found",
            )

        return CalculationDetailResponse(
            calculation_id=result.get("calculation_id", calculation_id),
            method=result.get("method", ""),
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            asset_count=result.get("asset_count", 0),
            details=result.get("details", {}),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error in get_calculation_detail: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve calculation",
        )


@router.get(
    "/calculations",
    response_model=CalculationListResponse,
    summary="List calculations",
    description=(
        "Retrieve a paginated list of downstream leased assets calculations. "
        "Supports filtering by method, asset category, organization, and "
        "date range. Returns summary information for each calculation."
    ),
)
async def list_calculations(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=500, description="Results per page"),
    method: Optional[str] = Query(
        None, description="Filter by calculation method"
    ),
    asset_category: Optional[str] = Query(
        None, description="Filter by asset category"
    ),
    org_id: Optional[str] = Query(
        None, description="Filter by organization ID"
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
    List downstream leased assets calculations with filtering.

    Args:
        page: Page number (1-indexed)
        page_size: Results per page
        method: Optional calculation method filter
        asset_category: Optional asset category filter
        org_id: Optional organization ID filter
        from_date: Optional start date filter
        to_date: Optional end date filter
        service: DownstreamLeasedAssetsService instance

    Returns:
        CalculationListResponse with paginated results

    Raises:
        HTTPException: 500 for listing failures
    """
    try:
        logger.info(
            f"Listing DLA calculations: page={page}, size={page_size}, "
            f"method={method}, category={asset_category}"
        )

        filters = {
            "page": page,
            "page_size": page_size,
            "method": method,
            "asset_category": asset_category,
            "org_id": org_id,
            "from_date": from_date,
            "to_date": to_date,
        }

        result = await service.list_calculations(filters)

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
    "/calculations/{calculation_id}",
    response_model=DeleteResponse,
    summary="Delete calculation",
    description=(
        "Soft-delete a specific downstream leased assets calculation. "
        "Marks the calculation as deleted with audit trail; data is "
        "retained for regulatory compliance and provenance integrity."
    ),
)
async def delete_calculation(
    calculation_id: str = Path(..., description="Calculation UUID"),
    service=Depends(get_service),
) -> DeleteResponse:
    """
    Soft-delete a specific calculation.

    Args:
        calculation_id: Calculation UUID
        service: DownstreamLeasedAssetsService instance

    Returns:
        DeleteResponse with deletion confirmation

    Raises:
        HTTPException: 404 if not found, 500 for deletion failures
    """
    try:
        logger.info(f"Deleting DLA calculation: {calculation_id}")

        deleted = await service.delete_calculation(calculation_id)

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Calculation {calculation_id} not found",
            )

        return DeleteResponse(
            calculation_id=calculation_id,
            deleted=True,
            message=f"Calculation {calculation_id} soft-deleted successfully",
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
# ENDPOINTS - REFERENCE DATA & FACTORS (4)
# ============================================================================


@router.get(
    "/emission-factors/{asset_type}",
    summary="Get emission factors by asset type",
    description=(
        "Retrieve emission factors for a specific asset type. Returns "
        "vehicle EFs (kgCO2e/km by fuel), equipment EFs (fuel consumption "
        "rates and load factors), IT asset power defaults, or building "
        "fuel EFs depending on the asset_type requested."
    ),
)
async def get_emission_factors(
    asset_type: str = Path(
        ...,
        description="Asset type (e.g. small_car, generator, server, office)",
    ),
    service=Depends(get_service),
) -> Dict[str, Any]:
    """
    Get emission factors for a specific asset type.

    Args:
        asset_type: Asset type identifier
        service: DownstreamLeasedAssetsService instance

    Returns:
        Dictionary with matching emission factors

    Raises:
        HTTPException: 400 for invalid asset type, 500 for retrieval failures
    """
    try:
        logger.info(f"Getting DLA emission factors: asset_type={asset_type}")

        result = await service.get_emission_factors(asset_type)

        if result is None or len(result.get("factors", [])) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No emission factors found for asset type: {asset_type}",
            )

        return json.loads(json.dumps(result, cls=DecimalEncoder))

    except HTTPException:
        raise
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
    summary="Get building EUI benchmarks",
    description=(
        "Retrieve Energy Use Intensity (EUI) benchmarks for buildings "
        "by type and climate zone. Returns kWh/sqm/year values from "
        "ASHRAE 90.1, CIBSE TM46, and regional standards. Covers "
        "8 building types across 5 ASHRAE climate zones."
    ),
)
async def get_building_benchmarks(
    building_type: Optional[str] = Query(
        None, description="Filter by building type"
    ),
    climate_zone: Optional[str] = Query(
        None, description="Filter by ASHRAE climate zone"
    ),
    service=Depends(get_service),
) -> Dict[str, Any]:
    """
    Get building EUI benchmarks by type and climate zone.

    Args:
        building_type: Optional building type filter
        climate_zone: Optional climate zone filter
        service: DownstreamLeasedAssetsService instance

    Returns:
        Dictionary with matching building benchmarks

    Raises:
        HTTPException: 500 for retrieval failures
    """
    try:
        logger.info(
            f"Getting building benchmarks: type={building_type}, "
            f"zone={climate_zone}"
        )

        filters = {
            "building_type": building_type,
            "climate_zone": climate_zone,
        }
        result = await service.get_building_benchmarks(filters)

        return json.loads(json.dumps(result, cls=DecimalEncoder))

    except Exception as e:
        logger.error(
            f"Error in get_building_benchmarks: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve building benchmarks",
        )


@router.get(
    "/grid-factors",
    summary="Get grid emission factors",
    description=(
        "Retrieve electricity grid emission factors by country or "
        "eGRID subregion. Returns kgCO2e/kWh values from IEA 2024 "
        "for international grids and EPA eGRID 2024 for US subregions. "
        "Covers 12 countries and 26 eGRID subregions."
    ),
)
async def get_grid_factors(
    country_code: Optional[str] = Query(
        None, description="Filter by ISO country code"
    ),
    region: Optional[str] = Query(
        None, description="Filter by eGRID subregion code"
    ),
    service=Depends(get_service),
) -> Dict[str, Any]:
    """
    Get grid emission factors by country or region.

    Args:
        country_code: Optional country code filter
        region: Optional eGRID subregion filter
        service: DownstreamLeasedAssetsService instance

    Returns:
        Dictionary with matching grid emission factors

    Raises:
        HTTPException: 500 for retrieval failures
    """
    try:
        logger.info(
            f"Getting grid factors: country={country_code}, region={region}"
        )

        filters = {"country_code": country_code, "region": region}
        result = await service.get_grid_factors(filters)

        return json.loads(json.dumps(result, cls=DecimalEncoder))

    except Exception as e:
        logger.error(f"Error in get_grid_factors: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve grid factors",
        )


@router.get(
    "/allocation-methods",
    summary="Get available allocation methods",
    description=(
        "Retrieve all available emissions allocation methods with "
        "descriptions and default parameters. Methods include floor "
        "area, headcount, revenue, time-based, equal split, and full "
        "allocation. Also returns default common area percentages "
        "and vacancy base-load factors by building type."
    ),
)
async def get_allocation_methods(
    service=Depends(get_service),
) -> Dict[str, Any]:
    """
    Get available allocation methods with defaults.

    Args:
        service: DownstreamLeasedAssetsService instance

    Returns:
        Dictionary with allocation method details and defaults

    Raises:
        HTTPException: 500 for retrieval failures
    """
    try:
        logger.info("Getting DLA allocation methods")

        result = await service.get_allocation_methods()

        return json.loads(json.dumps(result, cls=DecimalEncoder))

    except Exception as e:
        logger.error(
            f"Error in get_allocation_methods: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve allocation methods",
        )


# ============================================================================
# ENDPOINTS - AGGREGATION (1)
# ============================================================================


@router.get(
    "/aggregations",
    summary="Get aggregated emissions",
    description=(
        "Retrieve aggregated downstream leased assets emissions for "
        "time-series analysis. Returns totals with breakdowns by asset "
        "category, calculation method, and region. Supports daily, "
        "weekly, monthly, quarterly, and annual aggregation periods."
    ),
)
async def get_aggregations(
    period: str = Query(
        "monthly",
        description="Aggregation period: daily, weekly, monthly, quarterly, annual",
    ),
    from_date: Optional[str] = Query(
        None, description="Start date (ISO 8601)"
    ),
    to_date: Optional[str] = Query(
        None, description="End date (ISO 8601)"
    ),
    asset_category: Optional[str] = Query(
        None, description="Filter by asset category"
    ),
    org_id: Optional[str] = Query(
        None, description="Filter by organization ID"
    ),
    service=Depends(get_service),
) -> Dict[str, Any]:
    """
    Get aggregated emissions for time-series analysis.

    Args:
        period: Aggregation period
        from_date: Optional start date filter
        to_date: Optional end date filter
        asset_category: Optional asset category filter
        org_id: Optional organization ID filter
        service: DownstreamLeasedAssetsService instance

    Returns:
        Dictionary with aggregated emissions data

    Raises:
        HTTPException: 400 for invalid period, 500 for aggregation failures
    """
    try:
        logger.info(
            f"Getting DLA aggregations: period={period}, "
            f"from={from_date}, to={to_date}"
        )

        valid_periods = {"daily", "weekly", "monthly", "quarterly", "annual"}
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
            "asset_category": asset_category,
            "org_id": org_id,
        }

        result = await service.get_aggregations(filters)

        return json.loads(json.dumps(result, cls=DecimalEncoder))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_aggregations: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Aggregation failed",
        )


# ============================================================================
# ENDPOINTS - PROVENANCE (1)
# ============================================================================


@router.get(
    "/provenance/{calculation_id}",
    summary="Get provenance chain",
    description=(
        "Retrieve the complete SHA-256 provenance chain for a calculation. "
        "Includes all 10 pipeline stages (validate, classify, normalize, "
        "resolve_efs, calculate, allocate, aggregate, compliance, "
        "provenance, seal) with per-stage hashes and verification."
    ),
)
async def get_provenance(
    calculation_id: str = Path(..., description="Calculation UUID"),
    service=Depends(get_service),
) -> Dict[str, Any]:
    """
    Get provenance chain for a specific calculation.

    Args:
        calculation_id: Calculation UUID
        service: DownstreamLeasedAssetsService instance

    Returns:
        Dictionary with chain stages and verification status

    Raises:
        HTTPException: 404 if not found, 500 for retrieval failures
    """
    try:
        logger.info(
            f"Getting DLA provenance: calculation_id={calculation_id}"
        )

        result = await service.get_provenance(calculation_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"Provenance for calculation {calculation_id} not found"
                ),
            )

        return json.loads(json.dumps(result, cls=DecimalEncoder))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_provenance: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve provenance",
        )


# ============================================================================
# ENDPOINTS - HEALTH (1)
# ============================================================================


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description=(
        "Health check endpoint for the Downstream Leased Assets Agent. "
        "Returns service status, agent identifier, version, and uptime. "
        "No authentication required."
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

        return HealthResponse(
            status="healthy",
            agent_id="GL-MRV-S3-013",
            version="1.0.0",
            uptime_seconds=round(uptime, 2),
        )

    except Exception as e:
        logger.error(f"Error in health_check: {e}", exc_info=True)
        return HealthResponse(
            status="unhealthy",
            agent_id="GL-MRV-S3-013",
            version="1.0.0",
            uptime_seconds=0.0,
        )
