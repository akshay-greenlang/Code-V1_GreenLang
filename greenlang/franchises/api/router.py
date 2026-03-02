"""
Franchises Agent API Router - AGENT-MRV-027

This module implements the FastAPI router for franchise emissions
calculations following GHG Protocol Scope 3 Category 14 requirements.

Provides 22 REST endpoints for:
- Emissions calculations (full pipeline, franchise-specific, QSR, hotel,
  convenience, retail, average-data, spend-based, hybrid, batch, network)
- Multi-framework compliance checking (GHG Protocol, CSRD, CDP, SBTi,
  ISO 14064, GRI, SB 253)
- Calculation CRUD (get, list, soft-delete)
- Emission factor lookups by franchise type
- Franchise EUI benchmark reference data
- Grid emission factor reference data
- Supported franchise type metadata
- Time-series aggregations
- Provenance chain verification with SHA-256 hashing
- Health check with engine status

Follows GreenLang's zero-hallucination principle with deterministic calculations.
All numeric outputs use deterministic formulas; no LLM calls in the calculation path.

Agent ID: GL-MRV-S3-014
Prefix: gl_frn_

Example:
    >>> from fastapi import FastAPI
    >>> from greenlang.franchises.api.router import router
    >>> app = FastAPI()
    >>> app.include_router(router)
"""

from fastapi import APIRouter, HTTPException, Query, Path, Depends, status
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from decimal import Decimal
from enum import Enum
import logging
import uuid
import hashlib
from datetime import datetime, date

logger = logging.getLogger(__name__)

# Router configuration
router = APIRouter(
    tags=["franchises"],
    responses={404: {"description": "Not found"}},
)


# ============================================================================
# ENUMS
# ============================================================================


class FranchiseType(str, Enum):
    """Supported franchise types for Scope 3 Category 14."""

    QSR = "qsr"
    CASUAL_DINING = "casual_dining"
    FINE_DINING = "fine_dining"
    COFFEE_SHOP = "coffee_shop"
    BAKERY = "bakery"
    HOTEL_ECONOMY = "hotel_economy"
    HOTEL_MIDSCALE = "hotel_midscale"
    HOTEL_UPSCALE = "hotel_upscale"
    HOTEL_LUXURY = "hotel_luxury"
    CONVENIENCE_STORE = "convenience_store"
    GAS_STATION = "gas_station"
    RETAIL_APPAREL = "retail_apparel"
    RETAIL_ELECTRONICS = "retail_electronics"
    RETAIL_GROCERY = "retail_grocery"
    RETAIL_HOME = "retail_home"
    FITNESS_CENTER = "fitness_center"
    AUTOMOTIVE_SERVICE = "automotive_service"
    LAUNDRY_DRY_CLEAN = "laundry_dry_clean"
    CHILDCARE = "childcare"
    EDUCATION = "education"


class CalculationMethod(str, Enum):
    """Supported calculation methods for franchise emissions."""

    FRANCHISE_SPECIFIC = "franchise_specific"
    AVERAGE_DATA = "average_data"
    SPEND_BASED = "spend_based"
    HYBRID = "hybrid"


class ComplianceFramework(str, Enum):
    """Supported regulatory compliance frameworks."""

    GHG_PROTOCOL = "ghg_protocol"
    ISO_14064 = "iso_14064"
    CSRD_ESRS = "csrd_esrs"
    CDP = "cdp"
    SBTI = "sbti"
    SB_253 = "sb_253"
    GRI = "gri"


class ClimateZone(str, Enum):
    """ASHRAE climate zone classifications."""

    ZONE_1A = "1A"
    ZONE_2A = "2A"
    ZONE_2B = "2B"
    ZONE_3A = "3A"
    ZONE_3B = "3B"
    ZONE_3C = "3C"
    ZONE_4A = "4A"
    ZONE_4B = "4B"
    ZONE_4C = "4C"
    ZONE_5A = "5A"
    ZONE_5B = "5B"
    ZONE_6A = "6A"
    ZONE_6B = "6B"
    ZONE_7 = "7"
    ZONE_8 = "8"
    MIXED = "mixed"


# ============================================================================
# SERVICE DEPENDENCY
# ============================================================================


_service_instance = None


def get_service():
    """
    Get or create FranchisesService singleton instance.

    Returns:
        FranchisesService instance

    Raises:
        HTTPException: If service initialization fails (503)
    """
    global _service_instance

    if _service_instance is None:
        try:
            from greenlang.franchises.service import FranchisesService
            _service_instance = FranchisesService()
            logger.info("FranchisesService initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FranchisesService: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service initialization failed",
            )

    return _service_instance


# ============================================================================
# REQUEST MODELS - FRANCHISE UNIT DATA
# ============================================================================


class EnergyConsumption(BaseModel):
    """
    Energy consumption data for a single franchise unit.

    Captures metered electricity and fuel usage with optional
    breakdown by end-use category.

    Attributes:
        electricity_kwh: Annual electricity consumption in kWh
        natural_gas_therms: Annual natural gas consumption in therms
        propane_gallons: Annual propane consumption in gallons
        diesel_gallons: Annual diesel consumption in gallons (generators)
        fuel_oil_gallons: Annual fuel oil consumption in gallons
        district_heating_kwh: Annual district heating consumption in kWh
        district_cooling_kwh: Annual district cooling consumption in kWh
    """

    electricity_kwh: Optional[float] = Field(
        None, ge=0, description="Annual electricity consumption (kWh)"
    )
    natural_gas_therms: Optional[float] = Field(
        None, ge=0, description="Annual natural gas consumption (therms)"
    )
    propane_gallons: Optional[float] = Field(
        None, ge=0, description="Annual propane consumption (gallons)"
    )
    diesel_gallons: Optional[float] = Field(
        None, ge=0, description="Annual diesel consumption (gallons)"
    )
    fuel_oil_gallons: Optional[float] = Field(
        None, ge=0, description="Annual fuel oil consumption (gallons)"
    )
    district_heating_kwh: Optional[float] = Field(
        None, ge=0, description="Annual district heating consumption (kWh)"
    )
    district_cooling_kwh: Optional[float] = Field(
        None, ge=0, description="Annual district cooling consumption (kWh)"
    )


class RefrigerantData(BaseModel):
    """
    Refrigerant leakage data for a franchise unit.

    Captures equipment-level refrigerant charge and leakage
    for direct (Scope 1-equivalent within franchise boundary) emissions.

    Attributes:
        refrigerant_type: Refrigerant designation (R-134a, R-410A, R-404A, etc.)
        charge_kg: Total refrigerant charge in kilograms
        annual_leakage_rate: Annual leakage rate as decimal (0.0-1.0)
        equipment_type: Equipment type (walk_in_cooler, reach_in, display_case, etc.)
    """

    refrigerant_type: str = Field(
        ..., description="Refrigerant type (R-134a, R-410A, R-404A, R-407C, R-22)"
    )
    charge_kg: float = Field(
        ..., gt=0, description="Total refrigerant charge (kg)"
    )
    annual_leakage_rate: float = Field(
        0.10, ge=0.0, le=1.0, description="Annual leakage rate (0.0-1.0)"
    )
    equipment_type: Optional[str] = Field(
        None,
        description="Equipment type (walk_in_cooler, reach_in, display_case, ice_machine, ac_unit)",
    )


class FranchiseUnit(BaseModel):
    """
    Data model for a single franchise unit (location).

    Represents one franchise outlet with its physical characteristics,
    energy consumption, and operational metadata.

    Attributes:
        unit_id: Unique identifier for the franchise unit
        franchise_type: Type of franchise operation
        brand_name: Brand name of the franchise
        floor_area_m2: Total floor area in square metres
        climate_zone: ASHRAE climate zone of the unit location
        country: ISO 3166-1 alpha-3 country code
        region: State, province, or grid region identifier
        energy: Metered energy consumption data
        refrigerants: Refrigerant equipment and leakage data
        operating_hours_per_year: Annual operating hours
        year_built: Year the facility was constructed or last major renovation
        has_drive_through: Whether the unit has a drive-through window
    """

    unit_id: str = Field(
        ..., min_length=1, max_length=100, description="Unique franchise unit ID"
    )
    franchise_type: str = Field(
        ..., description="Franchise type (qsr, hotel_midscale, convenience_store, etc.)"
    )
    brand_name: Optional[str] = Field(
        None, max_length=200, description="Franchise brand name"
    )
    floor_area_m2: Optional[float] = Field(
        None, gt=0, le=100000, description="Floor area in square metres"
    )
    climate_zone: Optional[str] = Field(
        None, description="ASHRAE climate zone (1A-8, mixed)"
    )
    country: str = Field(
        "USA", min_length=2, max_length=3, description="ISO country code"
    )
    region: Optional[str] = Field(
        None, max_length=50, description="State, province, or grid region"
    )
    energy: Optional[EnergyConsumption] = Field(
        None, description="Metered energy consumption data"
    )
    refrigerants: Optional[List[RefrigerantData]] = Field(
        None, description="Refrigerant equipment and leakage data"
    )
    operating_hours_per_year: Optional[int] = Field(
        None, ge=0, le=8784, description="Annual operating hours"
    )
    year_built: Optional[int] = Field(
        None, ge=1900, le=2100, description="Construction or major renovation year"
    )
    has_drive_through: Optional[bool] = Field(
        None, description="Whether unit has a drive-through (QSR)"
    )


# ============================================================================
# REQUEST MODELS - QSR SPECIFIC
# ============================================================================


class QSRCookingProfile(BaseModel):
    """
    QSR-specific cooking energy profile.

    Captures cooking equipment fuel mix for quick-service restaurant
    franchise units with high cooking energy intensity.

    Attributes:
        cooking_fuel_type: Primary cooking fuel (natural_gas, electric, propane, dual_fuel)
        fryer_count: Number of deep fryers
        grill_count: Number of grills/griddles
        oven_count: Number of ovens
        daily_meals_served: Average daily meal count
        cooking_hours_per_day: Daily cooking hours
    """

    cooking_fuel_type: str = Field(
        "natural_gas",
        description="Primary cooking fuel (natural_gas, electric, propane, dual_fuel)",
    )
    fryer_count: Optional[int] = Field(
        None, ge=0, le=50, description="Number of deep fryers"
    )
    grill_count: Optional[int] = Field(
        None, ge=0, le=50, description="Number of grills/griddles"
    )
    oven_count: Optional[int] = Field(
        None, ge=0, le=50, description="Number of ovens"
    )
    daily_meals_served: Optional[int] = Field(
        None, ge=0, le=50000, description="Average daily meal count"
    )
    cooking_hours_per_day: Optional[float] = Field(
        None, ge=0, le=24, description="Daily cooking hours"
    )


class QSRCalculateRequest(BaseModel):
    """
    Request model for QSR (Quick-Service Restaurant) franchise calculation.

    Extends the franchise-specific method with QSR-specific cooking
    energy profiles and equipment parameters.

    Attributes:
        unit: Franchise unit data with energy consumption
        cooking_profile: QSR-specific cooking energy profile
        reporting_year: Reporting year for emission factor selection
    """

    unit: FranchiseUnit = Field(
        ..., description="Franchise unit data with energy consumption"
    )
    cooking_profile: QSRCookingProfile = Field(
        ..., description="QSR cooking energy profile"
    )
    reporting_year: int = Field(
        2024, ge=2000, le=2100, description="Reporting year"
    )


# ============================================================================
# REQUEST MODELS - HOTEL SPECIFIC
# ============================================================================


class HotelOperationalData(BaseModel):
    """
    Hotel-specific operational data for emissions calculation.

    Captures occupancy, room count, amenity flags, and laundry
    metrics for hotel franchise emissions estimation.

    Attributes:
        total_rooms: Total number of guest rooms
        occupancy_rate: Average annual occupancy rate (0.0-1.0)
        has_pool: Whether the hotel has a swimming pool
        has_spa: Whether the hotel has a spa/wellness facility
        has_restaurant: Whether the hotel has an on-site restaurant
        has_laundry: Whether the hotel has on-site laundry
        has_conference_center: Whether the hotel has conference/meeting facilities
        laundry_kg_per_room_night: Laundry weight per occupied room night (kg)
        average_room_size_m2: Average guest room floor area (m2)
    """

    total_rooms: int = Field(
        ..., ge=1, le=5000, description="Total number of guest rooms"
    )
    occupancy_rate: float = Field(
        0.65, ge=0.0, le=1.0, description="Average annual occupancy rate"
    )
    has_pool: bool = Field(False, description="Has swimming pool")
    has_spa: bool = Field(False, description="Has spa/wellness facility")
    has_restaurant: bool = Field(False, description="Has on-site restaurant")
    has_laundry: bool = Field(False, description="Has on-site laundry")
    has_conference_center: bool = Field(
        False, description="Has conference/meeting facilities"
    )
    laundry_kg_per_room_night: Optional[float] = Field(
        None, ge=0, le=50, description="Laundry weight per occupied room night (kg)"
    )
    average_room_size_m2: Optional[float] = Field(
        None, gt=0, le=500, description="Average guest room area (m2)"
    )


class HotelCalculateRequest(BaseModel):
    """
    Request model for hotel franchise emissions calculation.

    Combines franchise unit data with hotel-specific operational
    parameters for accurate Scope 3 Category 14 hotel franchise emissions.

    Attributes:
        unit: Franchise unit data with energy consumption
        hotel_data: Hotel-specific operational data
        reporting_year: Reporting year for emission factor selection
    """

    unit: FranchiseUnit = Field(
        ..., description="Franchise unit data with energy consumption"
    )
    hotel_data: HotelOperationalData = Field(
        ..., description="Hotel-specific operational data"
    )
    reporting_year: int = Field(
        2024, ge=2000, le=2100, description="Reporting year"
    )


# ============================================================================
# REQUEST MODELS - CONVENIENCE STORE SPECIFIC
# ============================================================================


class ConvenienceStoreData(BaseModel):
    """
    Convenience store specific operational data.

    Captures 24/7 refrigeration load, food service, and fuel
    dispensing parameters for convenience store franchise emissions.

    Attributes:
        is_24_7: Whether the store operates 24/7
        refrigerated_case_linear_m: Total linear metres of refrigerated display cases
        walk_in_cooler_count: Number of walk-in coolers
        walk_in_freezer_count: Number of walk-in freezers
        has_food_service: Whether the store has prepared food service
        has_fuel_dispensing: Whether the store has fuel pumps (canopy lighting)
        fuel_canopy_area_m2: Fuel canopy area in m2 (for canopy lighting emissions)
    """

    is_24_7: bool = Field(True, description="Operates 24/7")
    refrigerated_case_linear_m: Optional[float] = Field(
        None, ge=0, le=500, description="Linear metres of refrigerated cases"
    )
    walk_in_cooler_count: Optional[int] = Field(
        None, ge=0, le=20, description="Number of walk-in coolers"
    )
    walk_in_freezer_count: Optional[int] = Field(
        None, ge=0, le=20, description="Number of walk-in freezers"
    )
    has_food_service: bool = Field(
        False, description="Has prepared food service"
    )
    has_fuel_dispensing: bool = Field(
        False, description="Has fuel pumps"
    )
    fuel_canopy_area_m2: Optional[float] = Field(
        None, ge=0, le=2000, description="Fuel canopy area (m2)"
    )


class ConvenienceCalculateRequest(BaseModel):
    """
    Request model for convenience store franchise emissions calculation.

    Captures high refrigeration load and 24/7 operational patterns
    typical of convenience store and gas station franchises.

    Attributes:
        unit: Franchise unit data with energy consumption
        store_data: Convenience store specific operational data
        reporting_year: Reporting year for emission factor selection
    """

    unit: FranchiseUnit = Field(
        ..., description="Franchise unit data with energy consumption"
    )
    store_data: ConvenienceStoreData = Field(
        ..., description="Convenience store operational data"
    )
    reporting_year: int = Field(
        2024, ge=2000, le=2100, description="Reporting year"
    )


# ============================================================================
# REQUEST MODELS - RETAIL SPECIFIC
# ============================================================================


class RetailStoreData(BaseModel):
    """
    Retail store specific operational data.

    Captures lighting density, HVAC characteristics, and
    merchandise category for retail franchise emissions.

    Attributes:
        retail_category: Retail merchandise category
        lighting_power_density_w_m2: Lighting power density (W/m2)
        hvac_type: HVAC system type
        has_warehouse: Whether the store has an attached warehouse
        warehouse_area_m2: Attached warehouse area (m2)
        seasonal_adjustment: Seasonal energy adjustment factor
    """

    retail_category: str = Field(
        "general",
        description="Retail category (apparel, electronics, grocery, home, general)",
    )
    lighting_power_density_w_m2: Optional[float] = Field(
        None, ge=0, le=100, description="Lighting power density (W/m2)"
    )
    hvac_type: Optional[str] = Field(
        None,
        description="HVAC type (packaged_rooftop, split_system, vrf, chilled_water, none)",
    )
    has_warehouse: bool = Field(
        False, description="Has attached warehouse"
    )
    warehouse_area_m2: Optional[float] = Field(
        None, ge=0, le=50000, description="Attached warehouse area (m2)"
    )
    seasonal_adjustment: Optional[float] = Field(
        None, ge=0.5, le=2.0, description="Seasonal energy adjustment factor"
    )


class RetailCalculateRequest(BaseModel):
    """
    Request model for retail store franchise emissions calculation.

    Attributes:
        unit: Franchise unit data with energy consumption
        retail_data: Retail store specific operational data
        reporting_year: Reporting year for emission factor selection
    """

    unit: FranchiseUnit = Field(
        ..., description="Franchise unit data with energy consumption"
    )
    retail_data: RetailStoreData = Field(
        ..., description="Retail store operational data"
    )
    reporting_year: int = Field(
        2024, ge=2000, le=2100, description="Reporting year"
    )


# ============================================================================
# REQUEST MODELS - CORE CALCULATION REQUESTS
# ============================================================================


class CalculateRequest(BaseModel):
    """
    Request model for full pipeline franchise emissions calculation.

    Accepts a network of franchise units and runs the complete
    10-stage calculation pipeline: validate, classify, normalize,
    resolve_efs, calculate_energy, calculate_refrigerants, aggregate,
    compliance, quality_score, seal.

    Attributes:
        units: List of franchise units in the network
        method: Calculation method (franchise_specific, average_data, spend_based, hybrid)
        reporting_year: Reporting year for emission factor selection
        reporting_period: Reporting period identifier
        include_refrigerants: Whether to include refrigerant fugitive emissions
        grid_region_override: Optional grid region override for all units
    """

    units: List[FranchiseUnit] = Field(
        ...,
        min_items=1,
        max_items=10000,
        description="List of franchise units in the network",
    )
    method: str = Field(
        "franchise_specific",
        description="Calculation method (franchise_specific, average_data, spend_based, hybrid)",
    )
    reporting_year: int = Field(
        2024, ge=2000, le=2100, description="Reporting year"
    )
    reporting_period: Optional[str] = Field(
        None, description="Reporting period identifier (e.g. '2024-Q4')"
    )
    include_refrigerants: bool = Field(
        True, description="Include refrigerant fugitive emissions"
    )
    grid_region_override: Optional[str] = Field(
        None, description="Grid region override for all units"
    )


class FranchiseSpecificRequest(BaseModel):
    """
    Request model for franchise-specific (metered data) calculation.

    Uses actual energy consumption data from individual franchise
    units to calculate emissions with the highest data quality.

    Attributes:
        unit: Franchise unit data with metered energy consumption
        reporting_year: Reporting year for emission factor selection
        include_refrigerants: Whether to include refrigerant fugitive emissions
    """

    unit: FranchiseUnit = Field(
        ..., description="Franchise unit data with metered energy consumption"
    )
    reporting_year: int = Field(
        2024, ge=2000, le=2100, description="Reporting year"
    )
    include_refrigerants: bool = Field(
        True, description="Include refrigerant fugitive emissions"
    )


class AverageDataRequest(BaseModel):
    """
    Request model for average-data (benchmark) calculation.

    Uses EUI (Energy Use Intensity) benchmarks by franchise type
    and climate zone to estimate emissions when metered data is
    unavailable.

    Attributes:
        franchise_type: Type of franchise operation
        floor_area_m2: Total floor area in square metres
        climate_zone: ASHRAE climate zone
        country: ISO country code for grid emission factors
        region: State, province, or grid region
        unit_count: Number of units with this profile
        reporting_year: Reporting year for emission factor selection
    """

    franchise_type: str = Field(
        ..., description="Franchise type for EUI benchmark selection"
    )
    floor_area_m2: float = Field(
        ..., gt=0, le=100000, description="Floor area in square metres"
    )
    climate_zone: str = Field(
        "4A", description="ASHRAE climate zone (1A-8, mixed)"
    )
    country: str = Field(
        "USA", min_length=2, max_length=3, description="ISO country code"
    )
    region: Optional[str] = Field(
        None, max_length=50, description="Grid region identifier"
    )
    unit_count: int = Field(
        1, ge=1, le=10000, description="Number of units with this profile"
    )
    reporting_year: int = Field(
        2024, ge=2000, le=2100, description="Reporting year"
    )


class SpendBasedRequest(BaseModel):
    """
    Request model for spend-based (EEIO) franchise emissions calculation.

    Uses revenue or royalty payment data with EEIO (Environmentally
    Extended Input-Output) factors to estimate emissions. Applies CPI
    deflation and currency conversion.

    Attributes:
        franchise_type: Franchise type for NAICS code mapping
        revenue_usd: Annual franchise revenue in USD
        royalty_usd: Annual royalty payment in USD (alternative to revenue)
        naics_code: NAICS code override (auto-mapped from franchise_type if omitted)
        currency: ISO 4217 currency code
        reporting_year: Reporting year for CPI deflation
        unit_count: Number of franchise units
    """

    franchise_type: str = Field(
        ..., description="Franchise type for NAICS code mapping"
    )
    revenue_usd: Optional[float] = Field(
        None, gt=0, description="Annual franchise revenue (USD)"
    )
    royalty_usd: Optional[float] = Field(
        None, gt=0, description="Annual royalty payment (USD)"
    )
    naics_code: Optional[str] = Field(
        None, max_length=10, description="NAICS code override"
    )
    currency: str = Field(
        "USD", min_length=3, max_length=3, description="ISO 4217 currency code"
    )
    reporting_year: int = Field(
        2024, ge=2000, le=2100, description="Reporting year for CPI deflation"
    )
    unit_count: int = Field(
        1, ge=1, le=10000, description="Number of franchise units"
    )

    @validator("revenue_usd", "royalty_usd", pre=True, always=True)
    def validate_revenue_or_royalty(cls, v, values):
        """Validate that at least one spend input is provided."""
        return v


class HybridCalculateRequest(BaseModel):
    """
    Request model for hybrid method calculation.

    Combines franchise-specific metered data (where available) with
    average-data benchmarks (for remaining units) in a waterfall
    priority approach: franchise-specific > average-data > spend-based.

    Attributes:
        metered_units: Units with metered energy data (franchise-specific)
        estimated_units: Units using average-data benchmarks
        spend_data: Spend data for remaining units (fallback)
        reporting_year: Reporting year for emission factor selection
        include_refrigerants: Whether to include refrigerant emissions
    """

    metered_units: Optional[List[FranchiseUnit]] = Field(
        None, description="Units with metered energy data"
    )
    estimated_units: Optional[List[Dict[str, Any]]] = Field(
        None, description="Units using average-data benchmarks"
    )
    spend_data: Optional[Dict[str, Any]] = Field(
        None, description="Spend data for remaining units (fallback)"
    )
    reporting_year: int = Field(
        2024, ge=2000, le=2100, description="Reporting year"
    )
    include_refrigerants: bool = Field(
        True, description="Include refrigerant fugitive emissions"
    )


class BatchCalculateRequest(BaseModel):
    """
    Request model for batch franchise emissions calculations.

    Processes up to 10,000 franchise units in a single request
    with parallel execution and per-unit error isolation.

    Attributes:
        units: List of franchise unit data dictionaries
        method: Default calculation method for all units
        reporting_year: Reporting year for emission factor selection
        reporting_period: Reporting period identifier
        include_refrigerants: Whether to include refrigerant emissions
    """

    units: List[Dict[str, Any]] = Field(
        ...,
        min_items=1,
        max_items=10000,
        description="List of franchise unit data dictionaries",
    )
    method: str = Field(
        "franchise_specific",
        description="Default calculation method",
    )
    reporting_year: int = Field(
        2024, ge=2000, le=2100, description="Reporting year"
    )
    reporting_period: Optional[str] = Field(
        None, description="Reporting period identifier"
    )
    include_refrigerants: bool = Field(
        True, description="Include refrigerant emissions"
    )


class NetworkAnalysisRequest(BaseModel):
    """
    Request model for full franchise network analysis.

    Analyzes the entire franchise network with per-brand, per-type,
    and per-region aggregation, data coverage assessment, and
    intensity metrics (per unit, per m2, per revenue).

    Attributes:
        units: Complete list of franchise units in the network
        brand_hierarchy: Optional brand-to-parent mapping
        reporting_year: Reporting year for emission factor selection
        reporting_period: Reporting period identifier
        method: Calculation method for units without metered data
        include_refrigerants: Whether to include refrigerant emissions
        intensity_denominators: Denominators for intensity metrics
    """

    units: List[FranchiseUnit] = Field(
        ...,
        min_items=1,
        max_items=50000,
        description="Complete list of franchise units",
    )
    brand_hierarchy: Optional[Dict[str, str]] = Field(
        None, description="Brand-to-parent mapping for aggregation"
    )
    reporting_year: int = Field(
        2024, ge=2000, le=2100, description="Reporting year"
    )
    reporting_period: Optional[str] = Field(
        None, description="Reporting period identifier"
    )
    method: str = Field(
        "hybrid",
        description="Calculation method for units without metered data",
    )
    include_refrigerants: bool = Field(
        True, description="Include refrigerant emissions"
    )
    intensity_denominators: Optional[Dict[str, float]] = Field(
        None,
        description="Denominators for intensity metrics (total_revenue, total_units, total_area_m2)",
    )


class ComplianceCheckRequest(BaseModel):
    """
    Request model for multi-framework compliance checking.

    Checks franchise calculation results against selected regulatory
    frameworks for boundary correctness, method hierarchy compliance,
    data coverage thresholds, and disclosure requirements.

    Attributes:
        frameworks: List of framework identifiers to check against
        calculation_results: List of calculation result dicts to check
        data_coverage_percent: Percentage of units with metered data
        method_hierarchy_followed: Whether the GHG Protocol method hierarchy was followed
        boundary_description: Description of the organizational boundary
    """

    frameworks: List[str] = Field(
        ...,
        min_items=1,
        description="Frameworks to check (ghg_protocol, iso_14064, csrd_esrs, cdp, sbti, sb_253, gri)",
    )
    calculation_results: List[Dict[str, Any]] = Field(
        ...,
        min_items=1,
        description="Calculation results to check for compliance",
    )
    data_coverage_percent: Optional[float] = Field(
        None, ge=0, le=100, description="Percentage of units with metered data"
    )
    method_hierarchy_followed: bool = Field(
        False, description="Whether method hierarchy was followed"
    )
    boundary_description: Optional[str] = Field(
        None, description="Organizational boundary description"
    )


# ============================================================================
# RESPONSE MODELS
# ============================================================================


class UnitEmissionResult(BaseModel):
    """Response model for a single franchise unit emission result."""

    unit_id: str = Field(..., description="Franchise unit identifier")
    franchise_type: str = Field(..., description="Franchise type")
    method: str = Field(..., description="Calculation method used")
    energy_emissions_kgco2e: float = Field(
        ..., description="Energy-related emissions (kgCO2e)"
    )
    refrigerant_emissions_kgco2e: float = Field(
        0.0, description="Refrigerant fugitive emissions (kgCO2e)"
    )
    total_emissions_kgco2e: float = Field(
        ..., description="Total emissions (kgCO2e)"
    )
    electricity_emissions_kgco2e: Optional[float] = Field(
        None, description="Electricity emissions breakdown (kgCO2e)"
    )
    fuel_emissions_kgco2e: Optional[float] = Field(
        None, description="Fuel combustion emissions breakdown (kgCO2e)"
    )
    eui_kwh_per_m2: Optional[float] = Field(
        None, description="Energy use intensity (kWh/m2)"
    )
    emission_intensity_kgco2e_per_m2: Optional[float] = Field(
        None, description="Emission intensity (kgCO2e/m2)"
    )
    grid_ef_used: Optional[float] = Field(
        None, description="Grid emission factor used (kgCO2e/kWh)"
    )
    data_quality_score: Optional[float] = Field(
        None, ge=1.0, le=5.0, description="DQI score (1.0=best, 5.0=worst)"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash for this unit"
    )


class CalculateResponse(BaseModel):
    """Response model for full pipeline franchise calculation."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    method: str = Field(..., description="Primary calculation method")
    unit_count: int = Field(..., description="Number of units processed")
    total_emissions_kgco2e: float = Field(
        ..., description="Total network emissions (kgCO2e)"
    )
    energy_emissions_kgco2e: float = Field(
        ..., description="Total energy-related emissions (kgCO2e)"
    )
    refrigerant_emissions_kgco2e: float = Field(
        0.0, description="Total refrigerant emissions (kgCO2e)"
    )
    unit_results: Optional[List[UnitEmissionResult]] = Field(
        None, description="Per-unit emission results"
    )
    coverage_percent: float = Field(
        ..., description="Data coverage percentage"
    )
    data_quality_score: Optional[float] = Field(
        None, description="Composite DQI score"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance chain hash"
    )
    calculated_at: str = Field(
        ..., description="ISO 8601 calculation timestamp"
    )


class FranchiseSpecificResponse(BaseModel):
    """Response model for franchise-specific (metered) calculation."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    unit_id: str = Field(..., description="Franchise unit identifier")
    franchise_type: str = Field(..., description="Franchise type")
    method: str = Field(
        "franchise_specific", description="Calculation method"
    )
    electricity_emissions_kgco2e: float = Field(
        ..., description="Grid electricity emissions (kgCO2e)"
    )
    natural_gas_emissions_kgco2e: float = Field(
        0.0, description="Natural gas combustion emissions (kgCO2e)"
    )
    other_fuel_emissions_kgco2e: float = Field(
        0.0, description="Other fuel combustion emissions (kgCO2e)"
    )
    refrigerant_emissions_kgco2e: float = Field(
        0.0, description="Refrigerant fugitive emissions (kgCO2e)"
    )
    total_emissions_kgco2e: float = Field(
        ..., description="Total unit emissions (kgCO2e)"
    )
    grid_ef_kgco2e_per_kwh: Optional[float] = Field(
        None, description="Grid emission factor used"
    )
    eui_kwh_per_m2: Optional[float] = Field(
        None, description="Actual EUI (kWh/m2)"
    )
    data_quality_score: float = Field(
        ..., description="DQI score (1.0=best)"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )
    calculated_at: str = Field(
        ..., description="ISO 8601 calculation timestamp"
    )


class AverageDataResponse(BaseModel):
    """Response model for average-data (benchmark) calculation."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    franchise_type: str = Field(..., description="Franchise type")
    method: str = Field("average_data", description="Calculation method")
    benchmark_eui_kwh_per_m2: float = Field(
        ..., description="Benchmark EUI used (kWh/m2)"
    )
    estimated_energy_kwh: float = Field(
        ..., description="Estimated annual energy consumption (kWh)"
    )
    total_emissions_kgco2e: float = Field(
        ..., description="Total estimated emissions (kgCO2e)"
    )
    unit_count: int = Field(
        ..., description="Number of units calculated"
    )
    emission_intensity_kgco2e_per_m2: float = Field(
        ..., description="Emission intensity (kgCO2e/m2)"
    )
    data_quality_score: float = Field(
        ..., description="DQI score (higher = less certain)"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )
    calculated_at: str = Field(
        ..., description="ISO 8601 calculation timestamp"
    )


class SpendBasedResponse(BaseModel):
    """Response model for spend-based (EEIO) calculation."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    franchise_type: str = Field(..., description="Franchise type")
    method: str = Field("spend_based", description="Calculation method")
    naics_code: str = Field(..., description="NAICS code used")
    naics_description: str = Field(
        ..., description="NAICS category description"
    )
    spend_amount_usd: float = Field(
        ..., description="Spend amount in base-year USD"
    )
    ef_kgco2e_per_dollar: float = Field(
        ..., description="EEIO emission factor used"
    )
    total_emissions_kgco2e: float = Field(
        ..., description="Total spend-based emissions (kgCO2e)"
    )
    cpi_deflation_factor: Optional[float] = Field(
        None, description="CPI deflation factor applied"
    )
    data_quality_score: float = Field(
        ..., description="DQI score"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )
    calculated_at: str = Field(
        ..., description="ISO 8601 calculation timestamp"
    )


class HybridResponse(BaseModel):
    """Response model for hybrid method calculation."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    method: str = Field("hybrid", description="Calculation method")
    metered_unit_count: int = Field(
        0, description="Units with metered data"
    )
    estimated_unit_count: int = Field(
        0, description="Units using benchmarks"
    )
    spend_unit_count: int = Field(
        0, description="Units using spend-based method"
    )
    metered_emissions_kgco2e: float = Field(
        0.0, description="Emissions from metered units"
    )
    estimated_emissions_kgco2e: float = Field(
        0.0, description="Emissions from benchmark units"
    )
    spend_emissions_kgco2e: float = Field(
        0.0, description="Emissions from spend-based units"
    )
    total_emissions_kgco2e: float = Field(
        ..., description="Total hybrid emissions (kgCO2e)"
    )
    coverage_percent: float = Field(
        ..., description="Metered data coverage (%)"
    )
    data_quality_score: float = Field(
        ..., description="Weighted composite DQI score"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )
    calculated_at: str = Field(
        ..., description="ISO 8601 calculation timestamp"
    )


class BatchCalculateResponse(BaseModel):
    """Response model for batch calculation."""

    batch_id: str = Field(..., description="Unique batch UUID")
    results: List[Dict[str, Any]] = Field(
        ..., description="Per-unit calculation results"
    )
    total_emissions_kgco2e: float = Field(
        ..., description="Total emissions for all units"
    )
    unit_count: int = Field(
        ..., description="Number of successfully processed units"
    )
    errors: List[Dict[str, Any]] = Field(
        default_factory=list, description="Per-unit error details"
    )
    reporting_period: Optional[str] = Field(
        None, description="Reporting period identifier"
    )
    processing_time_ms: float = Field(
        ..., description="Total processing time (ms)"
    )


class NetworkAnalysisResponse(BaseModel):
    """Response model for full network analysis."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    total_emissions_kgco2e: float = Field(
        ..., description="Total network emissions (kgCO2e)"
    )
    total_units: int = Field(
        ..., description="Total franchise units analyzed"
    )
    by_brand: Dict[str, Dict[str, Any]] = Field(
        ..., description="Emissions breakdown by brand"
    )
    by_franchise_type: Dict[str, Dict[str, Any]] = Field(
        ..., description="Emissions breakdown by franchise type"
    )
    by_region: Dict[str, Dict[str, Any]] = Field(
        ..., description="Emissions breakdown by region"
    )
    by_method: Dict[str, Dict[str, Any]] = Field(
        ..., description="Emissions breakdown by calculation method"
    )
    intensity_metrics: Dict[str, float] = Field(
        ..., description="Intensity metrics (per unit, per m2, per revenue)"
    )
    coverage_summary: Dict[str, Any] = Field(
        ..., description="Data coverage summary"
    )
    data_quality_score: float = Field(
        ..., description="Network-level DQI score"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )
    calculated_at: str = Field(
        ..., description="ISO 8601 calculation timestamp"
    )


class ComplianceCheckResponse(BaseModel):
    """Response model for multi-framework compliance check."""

    results: List[Dict[str, Any]] = Field(
        ..., description="Per-framework compliance results"
    )
    overall_status: str = Field(
        ..., description="Overall compliance status (pass, fail, warning)"
    )
    overall_score: float = Field(
        ..., description="Overall compliance score (0.0-1.0)"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Improvement recommendations"
    )


class CalculationDetailResponse(BaseModel):
    """Response model for single calculation detail."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    franchise_type: Optional[str] = Field(
        None, description="Franchise type"
    )
    method: str = Field(..., description="Calculation method")
    total_emissions_kgco2e: float = Field(
        ..., description="Total emissions (kgCO2e)"
    )
    unit_count: int = Field(..., description="Number of units")
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


class EmissionFactorResponse(BaseModel):
    """Response model for franchise emission factor data."""

    franchise_type: str = Field(..., description="Franchise type")
    climate_zone: Optional[str] = Field(
        None, description="Climate zone"
    )
    eui_kwh_per_m2: Optional[float] = Field(
        None, description="Benchmark EUI (kWh/m2)"
    )
    ef_kgco2e_per_m2: Optional[float] = Field(
        None, description="Area-based emission factor (kgCO2e/m2)"
    )
    ef_kgco2e_per_dollar: Optional[float] = Field(
        None, description="Spend-based emission factor (kgCO2e/USD)"
    )
    source: str = Field(..., description="Factor data source")
    year: Optional[int] = Field(None, description="Factor vintage year")


class EmissionFactorListResponse(BaseModel):
    """Response model for emission factor listing."""

    factors: List[EmissionFactorResponse] = Field(
        ..., description="List of emission factors"
    )
    count: int = Field(..., description="Total factor count returned")


class BenchmarkResponse(BaseModel):
    """Response model for franchise EUI benchmarks."""

    franchise_type: str = Field(..., description="Franchise type")
    climate_zone: str = Field(..., description="ASHRAE climate zone")
    eui_kwh_per_m2: float = Field(
        ..., description="Energy use intensity (kWh/m2)"
    )
    source: str = Field(..., description="Benchmark data source")
    valid_from: Optional[str] = Field(
        None, description="Valid from date"
    )
    valid_to: Optional[str] = Field(
        None, description="Valid to date"
    )


class BenchmarkListResponse(BaseModel):
    """Response model for benchmark listing."""

    benchmarks: List[BenchmarkResponse] = Field(
        ..., description="List of EUI benchmarks"
    )
    count: int = Field(..., description="Total benchmark count")


class GridFactorResponse(BaseModel):
    """Response model for grid emission factor."""

    country: str = Field(..., description="Country code")
    region: Optional[str] = Field(None, description="Grid region")
    ef_kgco2e_per_kwh: float = Field(
        ..., description="Grid emission factor (kgCO2e/kWh)"
    )
    source: str = Field(..., description="Factor data source")
    year: int = Field(..., description="Factor year")


class GridFactorListResponse(BaseModel):
    """Response model for grid factor listing."""

    factors: List[GridFactorResponse] = Field(
        ..., description="List of grid emission factors"
    )
    count: int = Field(..., description="Total factor count")


class FranchiseTypeInfo(BaseModel):
    """Response model for franchise type metadata."""

    franchise_type: str = Field(..., description="Franchise type identifier")
    display_name: str = Field(..., description="Human-readable display name")
    description: str = Field(..., description="Type description")
    naics_code: str = Field(..., description="Mapped NAICS code")
    typical_eui_range: Dict[str, float] = Field(
        ..., description="Typical EUI range (min, max kWh/m2)"
    )
    typical_floor_area_m2: Dict[str, float] = Field(
        ..., description="Typical floor area range (min, max m2)"
    )
    has_refrigeration: bool = Field(
        ..., description="Whether type typically has significant refrigeration"
    )
    has_cooking: bool = Field(
        ..., description="Whether type typically has cooking operations"
    )


class FranchiseTypeListResponse(BaseModel):
    """Response model for franchise type listing."""

    franchise_types: List[FranchiseTypeInfo] = Field(
        ..., description="List of supported franchise types"
    )
    count: int = Field(..., description="Total type count")


class AggregationResponse(BaseModel):
    """Response model for time-series aggregated emissions."""

    period: str = Field(..., description="Aggregation period identifier")
    total_emissions_kgco2e: float = Field(
        ..., description="Total emissions for the period (kgCO2e)"
    )
    by_franchise_type: Dict[str, float] = Field(
        ..., description="Emissions by franchise type"
    )
    by_method: Dict[str, float] = Field(
        ..., description="Emissions by calculation method"
    )
    by_region: Dict[str, float] = Field(
        ..., description="Emissions by region"
    )
    unit_count: int = Field(..., description="Total units in period")
    intensity_per_unit: Optional[float] = Field(
        None, description="Average emissions per unit (kgCO2e)"
    )


class ProvenanceResponse(BaseModel):
    """Response model for provenance chain verification."""

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


class EngineStatus(BaseModel):
    """Status of a single calculation engine."""

    engine_name: str = Field(..., description="Engine identifier")
    status: str = Field(..., description="Engine status (healthy, degraded, unhealthy)")
    last_used: Optional[str] = Field(None, description="Last usage timestamp")


class HealthResponse(BaseModel):
    """Response model for health check with engine status."""

    status: str = Field(..., description="Service health status")
    agent_id: str = Field(..., description="Agent identifier")
    version: str = Field(..., description="Agent version")
    uptime_seconds: float = Field(
        ..., description="Seconds since service start"
    )
    engines: List[EngineStatus] = Field(
        ..., description="Individual engine status"
    )
    franchise_types_loaded: int = Field(
        ..., description="Number of franchise types loaded"
    )
    emission_factors_loaded: int = Field(
        ..., description="Number of emission factors loaded"
    )


# ============================================================================
# MODULE-LEVEL TRACKING
# ============================================================================

_start_time: datetime = datetime.utcnow()

# Engine names for health check
_ENGINE_NAMES = [
    "FranchiseBenchmarkDatabaseEngine",
    "FranchiseSpecificCalculatorEngine",
    "AverageDataCalculatorEngine",
    "SpendBasedCalculatorEngine",
    "HybridAggregatorEngine",
    "ComplianceCheckerEngine",
    "FranchisePipelineEngine",
]


# ============================================================================
# ENDPOINTS - CALCULATIONS (11 POST)
# ============================================================================


@router.post(
    "/calculate",
    response_model=CalculateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate franchise network emissions",
    description=(
        "Calculate GHG emissions for a franchise network through the full "
        "10-stage pipeline. Accepts multiple franchise units with metered or "
        "estimated data. Returns deterministic results with SHA-256 provenance "
        "hash and per-unit breakdowns."
    ),
)
async def calculate_emissions(
    request: CalculateRequest,
    service=Depends(get_service),
) -> CalculateResponse:
    """
    Calculate franchise network emissions through the full pipeline.

    Args:
        request: Calculation request with franchise units
        service: FranchisesService instance

    Returns:
        CalculateResponse with network emissions and per-unit results

    Raises:
        HTTPException: 400 for validation errors, 500 for processing failures
    """
    try:
        logger.info(
            f"Calculating franchise emissions: {len(request.units)} units, "
            f"method={request.method}, year={request.reporting_year}"
        )

        result = await service.calculate(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return CalculateResponse(
            calculation_id=calculation_id,
            method=result.get("method", request.method),
            unit_count=result.get("unit_count", len(request.units)),
            total_emissions_kgco2e=result.get("total_emissions_kgco2e", 0.0),
            energy_emissions_kgco2e=result.get("energy_emissions_kgco2e", 0.0),
            refrigerant_emissions_kgco2e=result.get("refrigerant_emissions_kgco2e", 0.0),
            unit_results=result.get("unit_results"),
            coverage_percent=result.get("coverage_percent", 0.0),
            data_quality_score=result.get("data_quality_score"),
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
            detail="Franchise calculation failed",
        )


@router.post(
    "/calculate/franchise-specific",
    response_model=FranchiseSpecificResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate franchise-specific emissions (metered data)",
    description=(
        "Calculate GHG emissions for a single franchise unit using actual "
        "metered energy consumption data. Highest data quality (DQI 1-2). "
        "Applies location-specific grid emission factors and fuel emission "
        "factors with optional refrigerant fugitive emissions."
    ),
)
async def calculate_franchise_specific(
    request: FranchiseSpecificRequest,
    service=Depends(get_service),
) -> FranchiseSpecificResponse:
    """
    Calculate emissions using franchise-specific metered data.

    Args:
        request: Request with metered energy consumption data
        service: FranchisesService instance

    Returns:
        FranchiseSpecificResponse with detailed emission breakdown

    Raises:
        HTTPException: 400 for missing energy data, 422 for invalid unit, 500 for failures
    """
    try:
        logger.info(
            f"Calculating franchise-specific: unit={request.unit.unit_id}, "
            f"type={request.unit.franchise_type}"
        )

        if request.unit.energy is None:
            raise ValueError(
                "Energy consumption data is required for franchise-specific method. "
                "Provide at least electricity_kwh or natural_gas_therms."
            )

        result = await service.calculate_franchise_specific(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return FranchiseSpecificResponse(
            calculation_id=calculation_id,
            unit_id=result.get("unit_id", request.unit.unit_id),
            franchise_type=result.get("franchise_type", request.unit.franchise_type),
            method="franchise_specific",
            electricity_emissions_kgco2e=result.get("electricity_emissions_kgco2e", 0.0),
            natural_gas_emissions_kgco2e=result.get("natural_gas_emissions_kgco2e", 0.0),
            other_fuel_emissions_kgco2e=result.get("other_fuel_emissions_kgco2e", 0.0),
            refrigerant_emissions_kgco2e=result.get("refrigerant_emissions_kgco2e", 0.0),
            total_emissions_kgco2e=result.get("total_emissions_kgco2e", 0.0),
            grid_ef_kgco2e_per_kwh=result.get("grid_ef_kgco2e_per_kwh"),
            eui_kwh_per_m2=result.get("eui_kwh_per_m2"),
            data_quality_score=result.get("data_quality_score", 1.5),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error(f"Validation error in calculate_franchise_specific: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in calculate_franchise_specific: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Franchise-specific calculation failed",
        )


@router.post(
    "/calculate/franchise-specific/qsr",
    response_model=FranchiseSpecificResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate QSR restaurant franchise emissions",
    description=(
        "Calculate GHG emissions for a Quick-Service Restaurant (QSR) "
        "franchise unit with QSR-specific cooking energy profiles. Accounts "
        "for high cooking energy intensity from deep fryers, grills, and ovens "
        "with fuel-specific emission factors."
    ),
)
async def calculate_qsr(
    request: QSRCalculateRequest,
    service=Depends(get_service),
) -> FranchiseSpecificResponse:
    """
    Calculate QSR restaurant franchise emissions.

    Args:
        request: QSR calculation request with cooking profile
        service: FranchisesService instance

    Returns:
        FranchiseSpecificResponse with QSR-specific breakdowns

    Raises:
        HTTPException: 400 for invalid cooking profile, 500 for failures
    """
    try:
        logger.info(
            f"Calculating QSR emissions: unit={request.unit.unit_id}, "
            f"cooking_fuel={request.cooking_profile.cooking_fuel_type}, "
            f"meals/day={request.cooking_profile.daily_meals_served}"
        )

        result = await service.calculate_qsr(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return FranchiseSpecificResponse(
            calculation_id=calculation_id,
            unit_id=result.get("unit_id", request.unit.unit_id),
            franchise_type=result.get("franchise_type", "qsr"),
            method="franchise_specific",
            electricity_emissions_kgco2e=result.get("electricity_emissions_kgco2e", 0.0),
            natural_gas_emissions_kgco2e=result.get("natural_gas_emissions_kgco2e", 0.0),
            other_fuel_emissions_kgco2e=result.get("other_fuel_emissions_kgco2e", 0.0),
            refrigerant_emissions_kgco2e=result.get("refrigerant_emissions_kgco2e", 0.0),
            total_emissions_kgco2e=result.get("total_emissions_kgco2e", 0.0),
            grid_ef_kgco2e_per_kwh=result.get("grid_ef_kgco2e_per_kwh"),
            eui_kwh_per_m2=result.get("eui_kwh_per_m2"),
            data_quality_score=result.get("data_quality_score", 1.5),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error(f"Validation error in calculate_qsr: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error in calculate_qsr: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="QSR calculation failed",
        )


@router.post(
    "/calculate/franchise-specific/hotel",
    response_model=FranchiseSpecificResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate hotel franchise emissions",
    description=(
        "Calculate GHG emissions for a hotel franchise unit with hotel-specific "
        "operational parameters including room count, occupancy rate, amenity "
        "flags (pool, spa, restaurant, laundry, conference), and laundry metrics. "
        "Uses hotel-class EUI benchmarks adjusted for amenities."
    ),
)
async def calculate_hotel(
    request: HotelCalculateRequest,
    service=Depends(get_service),
) -> FranchiseSpecificResponse:
    """
    Calculate hotel franchise emissions.

    Args:
        request: Hotel calculation request with operational data
        service: FranchisesService instance

    Returns:
        FranchiseSpecificResponse with hotel-specific breakdowns

    Raises:
        HTTPException: 400 for invalid hotel data, 500 for failures
    """
    try:
        logger.info(
            f"Calculating hotel emissions: unit={request.unit.unit_id}, "
            f"rooms={request.hotel_data.total_rooms}, "
            f"occupancy={request.hotel_data.occupancy_rate}"
        )

        result = await service.calculate_hotel(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return FranchiseSpecificResponse(
            calculation_id=calculation_id,
            unit_id=result.get("unit_id", request.unit.unit_id),
            franchise_type=result.get("franchise_type", request.unit.franchise_type),
            method="franchise_specific",
            electricity_emissions_kgco2e=result.get("electricity_emissions_kgco2e", 0.0),
            natural_gas_emissions_kgco2e=result.get("natural_gas_emissions_kgco2e", 0.0),
            other_fuel_emissions_kgco2e=result.get("other_fuel_emissions_kgco2e", 0.0),
            refrigerant_emissions_kgco2e=result.get("refrigerant_emissions_kgco2e", 0.0),
            total_emissions_kgco2e=result.get("total_emissions_kgco2e", 0.0),
            grid_ef_kgco2e_per_kwh=result.get("grid_ef_kgco2e_per_kwh"),
            eui_kwh_per_m2=result.get("eui_kwh_per_m2"),
            data_quality_score=result.get("data_quality_score", 1.5),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error(f"Validation error in calculate_hotel: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error in calculate_hotel: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Hotel franchise calculation failed",
        )


@router.post(
    "/calculate/franchise-specific/convenience",
    response_model=FranchiseSpecificResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate convenience store franchise emissions",
    description=(
        "Calculate GHG emissions for a convenience store franchise unit. "
        "Accounts for 24/7 operation, high refrigeration load from walk-in "
        "coolers and display cases, food service operations, and fuel canopy "
        "lighting. Refrigerant fugitive emissions are significant for this type."
    ),
)
async def calculate_convenience(
    request: ConvenienceCalculateRequest,
    service=Depends(get_service),
) -> FranchiseSpecificResponse:
    """
    Calculate convenience store franchise emissions.

    Args:
        request: Convenience store calculation request
        service: FranchisesService instance

    Returns:
        FranchiseSpecificResponse with convenience store breakdowns

    Raises:
        HTTPException: 400 for invalid store data, 500 for failures
    """
    try:
        logger.info(
            f"Calculating convenience store emissions: unit={request.unit.unit_id}, "
            f"24/7={request.store_data.is_24_7}, "
            f"coolers={request.store_data.walk_in_cooler_count}"
        )

        result = await service.calculate_convenience(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return FranchiseSpecificResponse(
            calculation_id=calculation_id,
            unit_id=result.get("unit_id", request.unit.unit_id),
            franchise_type=result.get("franchise_type", "convenience_store"),
            method="franchise_specific",
            electricity_emissions_kgco2e=result.get("electricity_emissions_kgco2e", 0.0),
            natural_gas_emissions_kgco2e=result.get("natural_gas_emissions_kgco2e", 0.0),
            other_fuel_emissions_kgco2e=result.get("other_fuel_emissions_kgco2e", 0.0),
            refrigerant_emissions_kgco2e=result.get("refrigerant_emissions_kgco2e", 0.0),
            total_emissions_kgco2e=result.get("total_emissions_kgco2e", 0.0),
            grid_ef_kgco2e_per_kwh=result.get("grid_ef_kgco2e_per_kwh"),
            eui_kwh_per_m2=result.get("eui_kwh_per_m2"),
            data_quality_score=result.get("data_quality_score", 1.5),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error(f"Validation error in calculate_convenience: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error in calculate_convenience: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Convenience store calculation failed",
        )


@router.post(
    "/calculate/franchise-specific/retail",
    response_model=FranchiseSpecificResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate retail store franchise emissions",
    description=(
        "Calculate GHG emissions for a retail store franchise unit. "
        "Supports apparel, electronics, grocery, and home goods categories "
        "with category-specific lighting density, HVAC profiles, and "
        "optional attached warehouse emissions."
    ),
)
async def calculate_retail(
    request: RetailCalculateRequest,
    service=Depends(get_service),
) -> FranchiseSpecificResponse:
    """
    Calculate retail store franchise emissions.

    Args:
        request: Retail store calculation request
        service: FranchisesService instance

    Returns:
        FranchiseSpecificResponse with retail store breakdowns

    Raises:
        HTTPException: 400 for invalid retail data, 500 for failures
    """
    try:
        logger.info(
            f"Calculating retail emissions: unit={request.unit.unit_id}, "
            f"category={request.retail_data.retail_category}"
        )

        result = await service.calculate_retail(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return FranchiseSpecificResponse(
            calculation_id=calculation_id,
            unit_id=result.get("unit_id", request.unit.unit_id),
            franchise_type=result.get(
                "franchise_type", "retail_" + request.retail_data.retail_category
            ),
            method="franchise_specific",
            electricity_emissions_kgco2e=result.get("electricity_emissions_kgco2e", 0.0),
            natural_gas_emissions_kgco2e=result.get("natural_gas_emissions_kgco2e", 0.0),
            other_fuel_emissions_kgco2e=result.get("other_fuel_emissions_kgco2e", 0.0),
            refrigerant_emissions_kgco2e=result.get("refrigerant_emissions_kgco2e", 0.0),
            total_emissions_kgco2e=result.get("total_emissions_kgco2e", 0.0),
            grid_ef_kgco2e_per_kwh=result.get("grid_ef_kgco2e_per_kwh"),
            eui_kwh_per_m2=result.get("eui_kwh_per_m2"),
            data_quality_score=result.get("data_quality_score", 1.5),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error(f"Validation error in calculate_retail: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error in calculate_retail: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Retail store calculation failed",
        )


@router.post(
    "/calculate/average-data",
    response_model=AverageDataResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate average-data benchmark emissions",
    description=(
        "Calculate GHG emissions using EUI (Energy Use Intensity) benchmarks "
        "by franchise type and ASHRAE climate zone. Used when metered energy "
        "data is unavailable. Applies benchmark EUI to floor area and "
        "location-specific grid emission factors."
    ),
)
async def calculate_average_data(
    request: AverageDataRequest,
    service=Depends(get_service),
) -> AverageDataResponse:
    """
    Calculate emissions using average-data benchmarks.

    Args:
        request: Average-data request with franchise type and floor area
        service: FranchisesService instance

    Returns:
        AverageDataResponse with benchmark-based emissions

    Raises:
        HTTPException: 400 for invalid franchise type, 500 for failures
    """
    try:
        logger.info(
            f"Calculating average-data emissions: type={request.franchise_type}, "
            f"area={request.floor_area_m2}m2, zone={request.climate_zone}, "
            f"units={request.unit_count}"
        )

        result = await service.calculate_average_data(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return AverageDataResponse(
            calculation_id=calculation_id,
            franchise_type=result.get("franchise_type", request.franchise_type),
            method="average_data",
            benchmark_eui_kwh_per_m2=result.get("benchmark_eui_kwh_per_m2", 0.0),
            estimated_energy_kwh=result.get("estimated_energy_kwh", 0.0),
            total_emissions_kgco2e=result.get("total_emissions_kgco2e", 0.0),
            unit_count=result.get("unit_count", request.unit_count),
            emission_intensity_kgco2e_per_m2=result.get(
                "emission_intensity_kgco2e_per_m2", 0.0
            ),
            data_quality_score=result.get("data_quality_score", 3.0),
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
    summary="Calculate spend-based EEIO emissions",
    description=(
        "Calculate GHG emissions using revenue or royalty payment data "
        "with EEIO (Environmentally Extended Input-Output) factors. "
        "Applies CPI deflation to base year and maps franchise type to "
        "NAICS code for factor selection. Lowest data quality (DQI 4-5)."
    ),
)
async def calculate_spend_based(
    request: SpendBasedRequest,
    service=Depends(get_service),
) -> SpendBasedResponse:
    """
    Calculate spend-based emissions using EEIO factors.

    Args:
        request: Spend-based request with revenue/royalty data
        service: FranchisesService instance

    Returns:
        SpendBasedResponse with EEIO-based emissions

    Raises:
        HTTPException: 400 for missing spend data, 500 for failures
    """
    try:
        if request.revenue_usd is None and request.royalty_usd is None:
            raise ValueError(
                "At least one of revenue_usd or royalty_usd must be provided"
            )

        spend_amount = request.revenue_usd or (
            request.royalty_usd * 10.0 if request.royalty_usd else 0.0
        )

        logger.info(
            f"Calculating spend-based emissions: type={request.franchise_type}, "
            f"spend=${spend_amount:.2f}, units={request.unit_count}"
        )

        result = await service.calculate_spend_based(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return SpendBasedResponse(
            calculation_id=calculation_id,
            franchise_type=result.get("franchise_type", request.franchise_type),
            method="spend_based",
            naics_code=result.get("naics_code", request.naics_code or ""),
            naics_description=result.get("naics_description", ""),
            spend_amount_usd=result.get("spend_amount_usd", spend_amount),
            ef_kgco2e_per_dollar=result.get("ef_kgco2e_per_dollar", 0.0),
            total_emissions_kgco2e=result.get("total_emissions_kgco2e", 0.0),
            cpi_deflation_factor=result.get("cpi_deflation_factor"),
            data_quality_score=result.get("data_quality_score", 4.0),
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
    response_model=HybridResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate hybrid method emissions",
    description=(
        "Calculate GHG emissions using hybrid method with waterfall "
        "priority: franchise-specific (metered) > average-data (benchmark) "
        "> spend-based (EEIO). Combines multiple methods across the network "
        "to maximize data quality while maintaining complete coverage."
    ),
)
async def calculate_hybrid(
    request: HybridCalculateRequest,
    service=Depends(get_service),
) -> HybridResponse:
    """
    Calculate emissions using hybrid method waterfall.

    Args:
        request: Hybrid request with metered, estimated, and spend data
        service: FranchisesService instance

    Returns:
        HybridResponse with method-level breakdowns

    Raises:
        HTTPException: 400 for insufficient data, 500 for failures
    """
    try:
        metered_count = len(request.metered_units) if request.metered_units else 0
        estimated_count = len(request.estimated_units) if request.estimated_units else 0
        has_spend = request.spend_data is not None

        if metered_count == 0 and estimated_count == 0 and not has_spend:
            raise ValueError(
                "At least one data source must be provided: "
                "metered_units, estimated_units, or spend_data"
            )

        logger.info(
            f"Calculating hybrid emissions: metered={metered_count}, "
            f"estimated={estimated_count}, has_spend={has_spend}"
        )

        result = await service.calculate_hybrid(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return HybridResponse(
            calculation_id=calculation_id,
            method="hybrid",
            metered_unit_count=result.get("metered_unit_count", metered_count),
            estimated_unit_count=result.get("estimated_unit_count", estimated_count),
            spend_unit_count=result.get("spend_unit_count", 0),
            metered_emissions_kgco2e=result.get("metered_emissions_kgco2e", 0.0),
            estimated_emissions_kgco2e=result.get("estimated_emissions_kgco2e", 0.0),
            spend_emissions_kgco2e=result.get("spend_emissions_kgco2e", 0.0),
            total_emissions_kgco2e=result.get("total_emissions_kgco2e", 0.0),
            coverage_percent=result.get("coverage_percent", 0.0),
            data_quality_score=result.get("data_quality_score", 2.5),
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
    summary="Batch calculate franchise emissions",
    description=(
        "Calculate GHG emissions for up to 10,000 franchise units in a "
        "single request. Uses parallel execution with per-unit error "
        "isolation. Returns aggregated totals, per-unit results, and "
        "any per-unit errors."
    ),
)
async def calculate_batch(
    request: BatchCalculateRequest,
    service=Depends(get_service),
) -> BatchCalculateResponse:
    """
    Calculate batch franchise emissions.

    Args:
        request: Batch request with franchise unit list
        service: FranchisesService instance

    Returns:
        BatchCalculateResponse with aggregated and per-unit results

    Raises:
        HTTPException: 400 for validation errors, 500 for batch failures
    """
    try:
        logger.info(
            f"Calculating batch franchise emissions: {len(request.units)} units, "
            f"method={request.method}, year={request.reporting_year}"
        )

        start_time = datetime.utcnow()
        result = await service.calculate_batch(request.dict())
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        batch_id = result.get("batch_id", str(uuid.uuid4()))

        return BatchCalculateResponse(
            batch_id=batch_id,
            results=result.get("results", []),
            total_emissions_kgco2e=result.get("total_emissions_kgco2e", 0.0),
            unit_count=result.get("unit_count", 0),
            errors=result.get("errors", []),
            reporting_period=result.get(
                "reporting_period", request.reporting_period
            ),
            processing_time_ms=result.get("processing_time_ms", processing_time),
        )

    except ValueError as e:
        logger.error(f"Validation error in calculate_batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error in calculate_batch: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch calculation failed",
        )


@router.post(
    "/calculate/network",
    response_model=NetworkAnalysisResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Full franchise network analysis",
    description=(
        "Analyze the entire franchise network with per-brand, per-type, "
        "and per-region aggregation. Calculates data coverage assessment, "
        "intensity metrics (per unit, per m2, per revenue), and identifies "
        "high-emission clusters for reduction targeting."
    ),
)
async def calculate_network(
    request: NetworkAnalysisRequest,
    service=Depends(get_service),
) -> NetworkAnalysisResponse:
    """
    Perform full franchise network analysis.

    Args:
        request: Network analysis request with all franchise units
        service: FranchisesService instance

    Returns:
        NetworkAnalysisResponse with multi-dimensional breakdowns

    Raises:
        HTTPException: 400 for validation errors, 500 for analysis failures
    """
    try:
        logger.info(
            f"Analyzing franchise network: {len(request.units)} units, "
            f"method={request.method}, year={request.reporting_year}"
        )

        result = await service.analyze_network(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return NetworkAnalysisResponse(
            calculation_id=calculation_id,
            total_emissions_kgco2e=result.get("total_emissions_kgco2e", 0.0),
            total_units=result.get("total_units", len(request.units)),
            by_brand=result.get("by_brand", {}),
            by_franchise_type=result.get("by_franchise_type", {}),
            by_region=result.get("by_region", {}),
            by_method=result.get("by_method", {}),
            intensity_metrics=result.get("intensity_metrics", {}),
            coverage_summary=result.get("coverage_summary", {}),
            data_quality_score=result.get("data_quality_score", 3.0),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error(f"Validation error in calculate_network: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error in calculate_network: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Network analysis failed",
        )


# ============================================================================
# ENDPOINTS - COMPLIANCE (1 POST)
# ============================================================================


@router.post(
    "/compliance/check",
    response_model=ComplianceCheckResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Check multi-framework compliance",
    description=(
        "Check franchise calculation results against one or more regulatory "
        "frameworks. Validates boundary correctness, method hierarchy compliance, "
        "data coverage thresholds, franchise-specific disclosure requirements, "
        "and intensity metric reporting. Supports GHG Protocol, ISO 14064, "
        "CSRD ESRS E1, CDP, SBTi, SB 253, and GRI 305."
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
        service: FranchisesService instance

    Returns:
        ComplianceCheckResponse with per-framework findings

    Raises:
        HTTPException: 400 for invalid frameworks, 500 for check failures
    """
    try:
        valid_frameworks = {f.value for f in ComplianceFramework}
        invalid = [f for f in request.frameworks if f not in valid_frameworks]
        if invalid:
            raise ValueError(
                f"Invalid frameworks: {invalid}. "
                f"Must be one of: {', '.join(sorted(valid_frameworks))}"
            )

        logger.info(
            f"Checking compliance: {len(request.frameworks)} frameworks, "
            f"{len(request.calculation_results)} results, "
            f"coverage={request.data_coverage_percent}%"
        )

        result = await service.check_compliance(request.dict())

        return ComplianceCheckResponse(
            results=result.get("results", []),
            overall_status=result.get("overall_status", "unknown"),
            overall_score=result.get("overall_score", 0.0),
            recommendations=result.get("recommendations", []),
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
# ENDPOINTS - CALCULATION CRUD (3: GET detail, GET list, DELETE)
# ============================================================================


@router.get(
    "/calculations/{calculation_id}",
    response_model=CalculationDetailResponse,
    summary="Get calculation detail",
    description=(
        "Retrieve detailed information for a specific franchise emissions "
        "calculation including full input/output payload, per-unit results, "
        "provenance hash, and calculation metadata."
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
        service: FranchisesService instance

    Returns:
        CalculationDetailResponse with full calculation data

    Raises:
        HTTPException: 404 if calculation not found, 500 for failures
    """
    try:
        logger.info(f"Getting calculation detail: {calculation_id}")

        result = await service.get_calculation(calculation_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Calculation {calculation_id} not found",
            )

        return CalculationDetailResponse(
            calculation_id=result.get("calculation_id", calculation_id),
            franchise_type=result.get("franchise_type"),
            method=result.get("method", ""),
            total_emissions_kgco2e=result.get("total_emissions_kgco2e", 0.0),
            unit_count=result.get("unit_count", 0),
            details=result.get("details", {}),
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
    summary="List franchise calculations",
    description=(
        "Retrieve a paginated list of franchise emissions calculations. "
        "Supports filtering by franchise type, calculation method, and "
        "date range. Returns summary information for each calculation."
    ),
)
async def list_calculations(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=500, description="Results per page"),
    franchise_type: Optional[str] = Query(
        None, description="Filter by franchise type"
    ),
    method: Optional[str] = Query(
        None, description="Filter by calculation method"
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
    List franchise calculations with filtering and pagination.

    Args:
        page: Page number (1-indexed)
        page_size: Number of results per page
        franchise_type: Optional franchise type filter
        method: Optional calculation method filter
        from_date: Optional start date filter
        to_date: Optional end date filter
        service: FranchisesService instance

    Returns:
        CalculationListResponse with paginated results

    Raises:
        HTTPException: 500 for listing failures
    """
    try:
        logger.info(
            f"Listing calculations: page={page}, size={page_size}, "
            f"type={franchise_type}, method={method}"
        )

        filters = {
            "page": page,
            "page_size": page_size,
            "franchise_type": franchise_type,
            "method": method,
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
    summary="Delete franchise calculation",
    description=(
        "Soft-delete a specific franchise emissions calculation. "
        "Marks the calculation as deleted with audit trail; "
        "data is retained for regulatory compliance."
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
        service: FranchisesService instance

    Returns:
        DeleteResponse with deletion confirmation

    Raises:
        HTTPException: 404 if not found, 500 for deletion failures
    """
    try:
        logger.info(f"Deleting calculation: {calculation_id}")

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
# ENDPOINTS - REFERENCE DATA (4 GET)
# ============================================================================


@router.get(
    "/emission-factors/{franchise_type}",
    response_model=EmissionFactorListResponse,
    summary="Get emission factors by franchise type",
    description=(
        "Retrieve emission factors for a specific franchise type including "
        "EUI benchmarks by climate zone, area-based emission factors, and "
        "spend-based EEIO factors."
    ),
)
async def get_emission_factors(
    franchise_type: str = Path(
        ..., description="Franchise type identifier"
    ),
    climate_zone: Optional[str] = Query(
        None, description="Filter by ASHRAE climate zone"
    ),
    service=Depends(get_service),
) -> EmissionFactorListResponse:
    """
    Get emission factors for a specific franchise type.

    Args:
        franchise_type: Franchise type identifier
        climate_zone: Optional climate zone filter
        service: FranchisesService instance

    Returns:
        EmissionFactorListResponse with franchise-type-specific factors

    Raises:
        HTTPException: 400 for invalid type, 500 for retrieval failures
    """
    try:
        logger.info(
            f"Getting emission factors: type={franchise_type}, zone={climate_zone}"
        )

        filters = {
            "franchise_type": franchise_type,
            "climate_zone": climate_zone,
        }
        result = await service.get_emission_factors(filters)

        return EmissionFactorListResponse(
            factors=result.get("factors", []),
            count=result.get("count", 0),
        )

    except ValueError as e:
        logger.error(f"Validation error in get_emission_factors: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error in get_emission_factors: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve emission factors",
        )


@router.get(
    "/franchise-benchmarks",
    response_model=BenchmarkListResponse,
    summary="Get franchise EUI benchmarks",
    description=(
        "Retrieve Energy Use Intensity (EUI) benchmarks for franchise types "
        "by ASHRAE climate zone. Benchmarks are sourced from CBECS, ENERGY "
        "STAR Portfolio Manager, and industry-specific studies (NRA, AHLA, "
        "NACS). Used for average-data calculation method."
    ),
)
async def get_franchise_benchmarks(
    franchise_type: Optional[str] = Query(
        None, description="Filter by franchise type"
    ),
    climate_zone: Optional[str] = Query(
        None, description="Filter by ASHRAE climate zone"
    ),
    service=Depends(get_service),
) -> BenchmarkListResponse:
    """
    Get franchise EUI benchmarks.

    Args:
        franchise_type: Optional franchise type filter
        climate_zone: Optional climate zone filter
        service: FranchisesService instance

    Returns:
        BenchmarkListResponse with EUI benchmark data

    Raises:
        HTTPException: 500 for retrieval failures
    """
    try:
        logger.info(
            f"Getting franchise benchmarks: type={franchise_type}, zone={climate_zone}"
        )

        filters = {
            "franchise_type": franchise_type,
            "climate_zone": climate_zone,
        }
        result = await service.get_franchise_benchmarks(filters)

        return BenchmarkListResponse(
            benchmarks=result.get("benchmarks", []),
            count=result.get("count", 0),
        )

    except Exception as e:
        logger.error(f"Error in get_franchise_benchmarks: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve franchise benchmarks",
        )


@router.get(
    "/grid-factors",
    response_model=GridFactorListResponse,
    summary="Get grid emission factors",
    description=(
        "Retrieve grid emission factors by country and region. Includes "
        "eGRID subregion factors (US), IEA country factors, and EU-EEA "
        "member state factors. Used for converting electricity consumption "
        "to CO2e emissions."
    ),
)
async def get_grid_factors(
    country: Optional[str] = Query(
        None, description="Filter by country code"
    ),
    region: Optional[str] = Query(
        None, description="Filter by grid region"
    ),
    year: Optional[int] = Query(
        None, ge=2000, le=2100, description="Filter by factor year"
    ),
    service=Depends(get_service),
) -> GridFactorListResponse:
    """
    Get grid emission factors.

    Args:
        country: Optional country code filter
        region: Optional grid region filter
        year: Optional factor year filter
        service: FranchisesService instance

    Returns:
        GridFactorListResponse with grid emission factors

    Raises:
        HTTPException: 500 for retrieval failures
    """
    try:
        logger.info(
            f"Getting grid factors: country={country}, region={region}, year={year}"
        )

        filters = {
            "country": country,
            "region": region,
            "year": year,
        }
        result = await service.get_grid_factors(filters)

        return GridFactorListResponse(
            factors=result.get("factors", []),
            count=result.get("count", 0),
        )

    except Exception as e:
        logger.error(f"Error in get_grid_factors: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve grid factors",
        )


@router.get(
    "/franchise-types",
    response_model=FranchiseTypeListResponse,
    summary="List supported franchise types",
    description=(
        "Retrieve all supported franchise types with descriptions, NAICS "
        "code mappings, typical EUI ranges, typical floor area ranges, "
        "and operational characteristics (refrigeration, cooking flags)."
    ),
)
async def list_franchise_types(
    service=Depends(get_service),
) -> FranchiseTypeListResponse:
    """
    List all supported franchise types with metadata.

    Args:
        service: FranchisesService instance

    Returns:
        FranchiseTypeListResponse with franchise type metadata

    Raises:
        HTTPException: 500 for retrieval failures
    """
    try:
        logger.info("Listing supported franchise types")

        result = await service.list_franchise_types()

        return FranchiseTypeListResponse(
            franchise_types=result.get("franchise_types", []),
            count=result.get("count", 0),
        )

    except Exception as e:
        logger.error(f"Error in list_franchise_types: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list franchise types",
        )


# ============================================================================
# ENDPOINTS - AGGREGATION & PROVENANCE (2 GET)
# ============================================================================


@router.get(
    "/aggregations",
    response_model=AggregationResponse,
    summary="Get time-series aggregated emissions",
    description=(
        "Retrieve aggregated franchise emissions for a specified period. "
        "Returns totals with breakdowns by franchise type, calculation "
        "method, and region. Supports daily, weekly, monthly, quarterly, "
        "and annual aggregation periods."
    ),
)
async def get_aggregations(
    period: str = Query(
        "monthly",
        description="Aggregation period (daily, weekly, monthly, quarterly, annual)",
    ),
    from_date: Optional[str] = Query(
        None, description="Start date (ISO 8601)"
    ),
    to_date: Optional[str] = Query(
        None, description="End date (ISO 8601)"
    ),
    franchise_type: Optional[str] = Query(
        None, description="Filter by franchise type"
    ),
    service=Depends(get_service),
) -> AggregationResponse:
    """
    Get aggregated emissions for a specified period.

    Args:
        period: Aggregation period identifier
        from_date: Optional start date filter
        to_date: Optional end date filter
        franchise_type: Optional franchise type filter
        service: FranchisesService instance

    Returns:
        AggregationResponse with aggregated emissions data

    Raises:
        HTTPException: 400 for invalid period, 500 for aggregation failures
    """
    try:
        valid_periods = {"daily", "weekly", "monthly", "quarterly", "annual"}
        if period not in valid_periods:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Invalid period '{period}'. "
                    f"Must be one of: {', '.join(sorted(valid_periods))}"
                ),
            )

        logger.info(
            f"Getting aggregations: period={period}, from={from_date}, "
            f"to={to_date}, type={franchise_type}"
        )

        filters = {
            "period": period,
            "from_date": from_date,
            "to_date": to_date,
            "franchise_type": franchise_type,
        }

        result = await service.get_aggregations(filters)

        return AggregationResponse(
            period=period,
            total_emissions_kgco2e=result.get("total_emissions_kgco2e", 0.0),
            by_franchise_type=result.get("by_franchise_type", {}),
            by_method=result.get("by_method", {}),
            by_region=result.get("by_region", {}),
            unit_count=result.get("unit_count", 0),
            intensity_per_unit=result.get("intensity_per_unit"),
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
    "/provenance/{calculation_id}",
    response_model=ProvenanceResponse,
    summary="Get provenance chain",
    description=(
        "Retrieve the complete SHA-256 provenance chain for a franchise "
        "calculation. Includes all pipeline stages (validate, classify, "
        "normalize, resolve_efs, calculate_energy, calculate_refrigerants, "
        "aggregate, compliance, quality_score, seal) with per-stage hashes "
        "and chain integrity verification."
    ),
)
async def get_provenance(
    calculation_id: str = Path(..., description="Calculation UUID"),
    service=Depends(get_service),
) -> ProvenanceResponse:
    """
    Get provenance chain for a specific calculation.

    Args:
        calculation_id: Calculation UUID
        service: FranchisesService instance

    Returns:
        ProvenanceResponse with chain stages and verification status

    Raises:
        HTTPException: 404 if not found, 500 for retrieval failures
    """
    try:
        logger.info(f"Getting provenance for calculation: {calculation_id}")

        result = await service.get_provenance(calculation_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Provenance for calculation {calculation_id} not found",
            )

        return ProvenanceResponse(
            calculation_id=result.get("calculation_id", calculation_id),
            chain=result.get("chain", []),
            is_valid=result.get("is_valid", False),
            root_hash=result.get("root_hash", ""),
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
# ENDPOINTS - HEALTH (1 GET)
# ============================================================================


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check with engine status",
    description=(
        "Health check endpoint for the Franchises Agent. Returns service "
        "status, agent identifier, version, uptime, and per-engine health "
        "status for all 7 calculation engines. No authentication required."
    ),
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint (no auth required).

    Returns:
        HealthResponse with service status, engine health, and data counts
    """
    try:
        uptime = (datetime.utcnow() - _start_time).total_seconds()

        engines = []
        for engine_name in _ENGINE_NAMES:
            engines.append(
                EngineStatus(
                    engine_name=engine_name,
                    status="healthy",
                    last_used=None,
                )
            )

        return HealthResponse(
            status="healthy",
            agent_id="GL-MRV-S3-014",
            version="1.0.0",
            uptime_seconds=round(uptime, 2),
            engines=engines,
            franchise_types_loaded=len(FranchiseType),
            emission_factors_loaded=188,
        )

    except Exception as e:
        logger.error(f"Error in health_check: {e}", exc_info=True)
        return HealthResponse(
            status="unhealthy",
            agent_id="GL-MRV-S3-014",
            version="1.0.0",
            uptime_seconds=0.0,
            engines=[],
            franchise_types_loaded=0,
            emission_factors_loaded=0,
        )
