"""
Scope3CalculatorAgent Data Models
GL-VCCI Scope 3 Platform

Pydantic models for calculation inputs, outputs, and provenance tracking.

Version: 1.0.0
Date: 2025-10-30
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator
from enum import Enum

from .config import (
    TierType,
    TransportMode,
    CabinClass,
    CommuteMode,
    BuildingType,
    FranchiseType,
    ProductType,
    MaterialType,
    DisposalMethod,
    AssetClass,
)


# ============================================================================
# CALCULATION INPUT MODELS
# ============================================================================

class Category1Input(BaseModel):
    """Input data for Category 1 (Purchased Goods & Services) calculation."""

    product_name: str = Field(
        description="Product or service name"
    )

    quantity: float = Field(
        gt=0,
        description="Quantity purchased"
    )

    quantity_unit: str = Field(
        description="Unit of quantity (kg, units, etc.)"
    )

    region: str = Field(
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code"
    )

    # Tier 1 data
    supplier_pcf: Optional[float] = Field(
        default=None,
        description="Supplier-specific Product Carbon Footprint (kgCO2e/unit)"
    )

    supplier_pcf_uncertainty: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="PCF uncertainty as decimal (e.g., 0.10 = ±10%)"
    )

    # Tier 2 data
    product_code: Optional[str] = Field(
        default=None,
        description="Product code (NAICS, ISIC, custom taxonomy)"
    )

    product_category: Optional[str] = Field(
        default=None,
        description="Product category for emission factor lookup"
    )

    # Tier 3 data
    spend_usd: Optional[float] = Field(
        default=None,
        ge=0,
        description="Spend amount in USD"
    )

    economic_sector: Optional[str] = Field(
        default=None,
        description="Economic sector for spend-based calculation"
    )

    # Metadata
    supplier_name: Optional[str] = Field(default=None)
    purchase_date: Optional[datetime] = Field(default=None)
    data_quality_notes: Optional[str] = Field(default=None)


class Category4Input(BaseModel):
    """Input data for Category 4 (Upstream Transportation & Distribution) - ISO 14083."""

    transport_mode: TransportMode = Field(
        description="Transport mode"
    )

    distance_km: float = Field(
        gt=0,
        description="Distance in kilometers"
    )

    weight_tonnes: float = Field(
        gt=0,
        description="Weight in tonnes"
    )

    # Optional overrides
    emission_factor: Optional[float] = Field(
        default=None,
        description="Custom emission factor (kgCO2e per tonne-km)"
    )

    emission_factor_uncertainty: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Emission factor uncertainty"
    )

    # Additional details
    fuel_type: Optional[str] = Field(
        default=None,
        description="Fuel type (diesel, electric, etc.)"
    )

    vehicle_type: Optional[str] = Field(
        default=None,
        description="Specific vehicle type"
    )

    load_factor: Optional[float] = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Load factor (0-1, default 1 = full load)"
    )

    # Metadata
    origin: Optional[str] = Field(default=None, description="Origin location")
    destination: Optional[str] = Field(default=None, description="Destination location")
    shipment_id: Optional[str] = Field(default=None)
    shipment_date: Optional[datetime] = Field(default=None)


class Category6FlightInput(BaseModel):
    """Input data for Category 6 flight emissions."""

    distance_km: float = Field(
        gt=0,
        description="Flight distance in kilometers"
    )

    cabin_class: CabinClass = Field(
        default=CabinClass.ECONOMY,
        description="Cabin class"
    )

    num_passengers: int = Field(
        default=1,
        ge=1,
        description="Number of passengers"
    )

    # Optional overrides
    emission_factor: Optional[float] = Field(
        default=None,
        description="Custom emission factor (kgCO2e per passenger-km)"
    )

    apply_radiative_forcing: bool = Field(
        default=True,
        description="Apply radiative forcing multiplier"
    )

    # Metadata
    origin: Optional[str] = Field(default=None)
    destination: Optional[str] = Field(default=None)
    flight_number: Optional[str] = Field(default=None)
    travel_date: Optional[datetime] = Field(default=None)


class Category6HotelInput(BaseModel):
    """Input data for Category 6 hotel emissions."""

    nights: int = Field(
        ge=1,
        description="Number of nights"
    )

    region: str = Field(
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code"
    )

    # Optional overrides
    emission_factor: Optional[float] = Field(
        default=None,
        description="Custom emission factor (kgCO2e per night)"
    )

    # Metadata
    hotel_name: Optional[str] = Field(default=None)
    check_in_date: Optional[datetime] = Field(default=None)


class Category6GroundTransportInput(BaseModel):
    """Input data for Category 6 ground transport emissions."""

    distance_km: float = Field(
        gt=0,
        description="Distance in kilometers"
    )

    vehicle_type: str = Field(
        default="car_medium",
        description="Vehicle type (car_small, car_medium, taxi, rental_car)"
    )

    # Optional overrides
    emission_factor: Optional[float] = Field(
        default=None,
        description="Custom emission factor (kgCO2e per km)"
    )

    # Metadata
    trip_purpose: Optional[str] = Field(default=None)
    trip_date: Optional[datetime] = Field(default=None)


class Category6Input(BaseModel):
    """Combined input for Category 6 (Business Travel)."""

    flights: List[Category6FlightInput] = Field(default_factory=list)
    hotels: List[Category6HotelInput] = Field(default_factory=list)
    ground_transport: List[Category6GroundTransportInput] = Field(default_factory=list)

    # Metadata
    employee_id: Optional[str] = Field(default=None)
    trip_id: Optional[str] = Field(default=None)
    trip_purpose: Optional[str] = Field(default=None)


class Category2Input(BaseModel):
    """Input data for Category 2 (Capital Goods) calculation."""

    asset_description: str = Field(
        description="Description of capital asset purchased"
    )

    capex_amount: float = Field(
        gt=0,
        description="Capital expenditure amount in USD"
    )

    region: str = Field(
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code"
    )

    # Tier 1 data (supplier-specific)
    supplier_pcf: Optional[float] = Field(
        default=None,
        description="Supplier-specific total carbon footprint (kgCO2e)"
    )

    supplier_pcf_uncertainty: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="PCF uncertainty as decimal"
    )

    # Tier 2 data (asset-specific with LLM classification)
    asset_category: Optional[str] = Field(
        default=None,
        description="Asset category (buildings, machinery, vehicles, IT, equipment)"
    )

    useful_life_years: Optional[float] = Field(
        default=None,
        gt=0,
        description="Useful life of asset in years"
    )

    emission_factor_kgco2e_per_usd: Optional[float] = Field(
        default=None,
        description="Asset-specific emission factor (kgCO2e per USD)"
    )

    # Tier 3 data (spend-based)
    economic_sector: Optional[str] = Field(
        default=None,
        description="Economic sector for spend-based calculation"
    )

    # Metadata
    supplier_name: Optional[str] = Field(default=None)
    purchase_date: Optional[datetime] = Field(default=None)
    industry: Optional[str] = Field(default=None, description="Company industry for useful life estimation")
    data_quality_notes: Optional[str] = Field(default=None)


class Category3Input(BaseModel):
    """Input data for Category 3 (Fuel & Energy-Related Activities) calculation."""

    fuel_or_energy_type: str = Field(
        description="Type of fuel or energy (electricity, natural_gas, diesel, etc.)"
    )

    quantity: float = Field(
        gt=0,
        description="Quantity of fuel/energy consumed"
    )

    quantity_unit: str = Field(
        description="Unit of quantity (kWh, liters, kg, etc.)"
    )

    region: str = Field(
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code"
    )

    # Tier 1 data (supplier-specific)
    supplier_upstream_ef: Optional[float] = Field(
        default=None,
        description="Supplier-specific upstream emission factor (kgCO2e/unit)"
    )

    supplier_td_losses_ef: Optional[float] = Field(
        default=None,
        description="Supplier-specific T&D losses emission factor (kgCO2e/unit)"
    )

    # Tier 2 data (database factors with LLM fuel type identification)
    well_to_tank_ef: Optional[float] = Field(
        default=None,
        description="Well-to-tank emission factor (kgCO2e/unit)"
    )

    td_loss_percentage: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Transmission & distribution loss percentage (0-1)"
    )

    # Metadata
    supplier_name: Optional[str] = Field(default=None)
    consumption_period: Optional[str] = Field(default=None)
    grid_region: Optional[str] = Field(default=None, description="Electricity grid region")
    data_quality_notes: Optional[str] = Field(default=None)


class Category5Input(BaseModel):
    """Input data for Category 5 (Waste Generated in Operations) calculation."""

    waste_description: str = Field(
        description="Description of waste generated"
    )

    waste_mass_kg: float = Field(
        gt=0,
        description="Mass of waste in kilograms"
    )

    region: str = Field(
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code"
    )

    # Tier 1 data (supplier-specific)
    supplier_disposal_ef: Optional[float] = Field(
        default=None,
        description="Supplier-specific disposal emission factor (kgCO2e/kg)"
    )

    # Tier 2 data (with LLM waste categorization)
    waste_type: Optional[str] = Field(
        default=None,
        description="Waste type (municipal_solid, hazardous, construction, organic, etc.)"
    )

    disposal_method: Optional[str] = Field(
        default=None,
        description="Disposal method (landfill, incineration, recycling, composting, etc.)"
    )

    emission_factor_kgco2e_per_kg: Optional[float] = Field(
        default=None,
        description="Waste-specific emission factor (kgCO2e per kg)"
    )

    # Tier 3 data (average/proxy)
    waste_category_generic: Optional[str] = Field(
        default=None,
        description="Generic waste category for proxy factors"
    )

    # Metadata
    waste_handler: Optional[str] = Field(default=None)
    disposal_date: Optional[datetime] = Field(default=None)
    recycling_rate: Optional[float] = Field(default=None, ge=0, le=1, description="Recycling rate (0-1)")
    data_quality_notes: Optional[str] = Field(default=None)


class Category7Input(BaseModel):
    """Input data for Category 7 (Employee Commuting) calculation."""

    commute_mode: Optional[CommuteMode] = Field(
        default=None,
        description="Commute transportation mode"
    )

    distance_km: Optional[float] = Field(
        default=None,
        gt=0,
        description="One-way commute distance in kilometers"
    )

    days_per_week: Optional[float] = Field(
        default=None,
        gt=0,
        description="Days commuting per week"
    )

    num_employees: int = Field(
        default=1,
        ge=1,
        description="Number of employees"
    )

    # Optional parameters
    weeks_per_year: Optional[int] = Field(
        default=48,
        description="Working weeks per year"
    )

    car_occupancy: float = Field(
        default=1.0,
        ge=1.0,
        description="Car occupancy (for carpools)"
    )

    survey_response: Optional[str] = Field(
        default=None,
        description="Employee survey response for LLM analysis"
    )

    # Metadata
    employee_id: Optional[str] = Field(default=None)
    department: Optional[str] = Field(default=None)
    location: Optional[str] = Field(default=None)


class Category8Input(BaseModel):
    """Input data for Category 8 (Upstream Leased Assets) calculation."""

    lease_type: str = Field(
        description="Type of leased asset"
    )

    floor_area: Optional[float] = Field(
        default=None,
        gt=0,
        description="Floor area in square meters"
    )

    building_type: Optional[BuildingType] = Field(
        default=None,
        description="Building type"
    )

    energy_consumed: Optional[float] = Field(
        default=None,
        ge=0,
        description="Energy consumed (kWh)"
    )

    region: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code"
    )

    # Metadata
    asset_id: Optional[str] = Field(default=None)
    lease_start_date: Optional[datetime] = Field(default=None)
    lease_end_date: Optional[datetime] = Field(default=None)


class Category9Input(BaseModel):
    """Input data for Category 9 (Downstream Transportation & Distribution) calculation."""

    transport_mode: TransportMode = Field(
        description="Transport mode"
    )

    distance_km: float = Field(
        gt=0,
        description="Distance in kilometers"
    )

    weight_tonnes: float = Field(
        gt=0,
        description="Weight in tonnes"
    )

    # Optional parameters
    delivery_route: Optional[str] = Field(
        default=None,
        description="Delivery route description"
    )

    emission_factor: Optional[float] = Field(
        default=None,
        description="Custom emission factor (kgCO2e per tonne-km)"
    )

    # Metadata
    shipment_id: Optional[str] = Field(default=None)
    customer_id: Optional[str] = Field(default=None)
    shipment_date: Optional[datetime] = Field(default=None)


class Category10Input(BaseModel):
    """Input data for Category 10 (Processing of Sold Products) calculation."""

    product_description: str = Field(
        description="Description of intermediate product sold"
    )

    sold_quantity: float = Field(
        gt=0,
        description="Quantity sold"
    )

    industry_sector: Optional[str] = Field(
        default=None,
        description="Customer industry sector"
    )

    processing_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Processing data from customer"
    )

    region: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code"
    )

    # Metadata
    customer_name: Optional[str] = Field(default=None)
    sale_date: Optional[datetime] = Field(default=None)


class Category11Input(BaseModel):
    """Input data for Category 11 (Use of Sold Products) calculation."""

    product_type: ProductType = Field(
        description="Product type category"
    )

    units_sold: int = Field(
        gt=0,
        description="Number of units sold"
    )

    energy_consumption: Optional[float] = Field(
        default=None,
        ge=0,
        description="Annual energy consumption per unit (kWh/year)"
    )

    lifespan_years: Optional[float] = Field(
        default=None,
        gt=0,
        description="Product lifespan in years"
    )

    region: str = Field(
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code"
    )

    # Metadata
    product_name: Optional[str] = Field(default=None)
    sale_date: Optional[datetime] = Field(default=None)


class Category12Input(BaseModel):
    """Input data for Category 12 (End-of-Life Treatment) calculation."""

    product_description: str = Field(
        description="Description of product"
    )

    weight_kg: float = Field(
        gt=0,
        description="Product weight in kilograms"
    )

    material_composition: Optional[Dict[str, float]] = Field(
        default=None,
        description="Material composition breakdown"
    )

    disposal_method: Optional[DisposalMethod] = Field(
        default=None,
        description="Disposal method"
    )

    region: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code"
    )

    # Metadata
    units_sold: Optional[int] = Field(default=None)
    recycling_rate: Optional[float] = Field(default=None, ge=0, le=1)


class Category13Input(BaseModel):
    """Input data for Category 13 (Downstream Leased Assets) calculation."""

    building_type: Optional[BuildingType] = Field(
        default=None,
        description="Building type"
    )

    floor_area: float = Field(
        gt=0,
        description="Floor area in square meters"
    )

    tenant_type: Optional[str] = Field(
        default=None,
        description="Tenant type"
    )

    energy_consumed: Optional[float] = Field(
        default=None,
        ge=0,
        description="Energy consumed by tenant (kWh)"
    )

    region: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code"
    )

    # Metadata
    asset_id: Optional[str] = Field(default=None)
    tenant_name: Optional[str] = Field(default=None)


class Category14Input(BaseModel):
    """Input data for Category 14 (Franchises) calculation."""

    franchise_type: Optional[FranchiseType] = Field(
        default=None,
        description="Franchise type"
    )

    franchise_count: int = Field(
        ge=1,
        description="Number of franchise locations"
    )

    revenue_usd: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total franchise revenue in USD"
    )

    floor_area: Optional[float] = Field(
        default=None,
        gt=0,
        description="Total floor area across franchises (sqm)"
    )

    region: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code"
    )

    # Metadata
    franchise_name: Optional[str] = Field(default=None)
    energy_consumed: Optional[float] = Field(default=None, ge=0)


class Category15Input(BaseModel):
    """Input data for Category 15 (Investments) calculation - PCAF Standard."""

    portfolio_company: str = Field(
        description="Portfolio company name"
    )

    outstanding_amount: float = Field(
        gt=0,
        description="Outstanding investment amount (USD)"
    )

    company_value: float = Field(
        gt=0,
        description="Company value - EVIC or Total Assets (USD)"
    )

    sector: Optional[str] = Field(
        default=None,
        description="Industry sector"
    )

    asset_class: Optional[AssetClass] = Field(
        default=None,
        description="PCAF asset class"
    )

    company_emissions: Optional[float] = Field(
        default=None,
        ge=0,
        description="Portfolio company total emissions (tCO2e)"
    )

    # Metadata
    investment_id: Optional[str] = Field(default=None)
    region: Optional[str] = Field(default=None)
    reporting_year: Optional[int] = Field(default=None)


# ============================================================================
# CALCULATION OUTPUT MODELS
# ============================================================================

class UncertaintyResult(BaseModel):
    """Monte Carlo uncertainty propagation result."""

    mean: float = Field(description="Mean value")
    std_dev: float = Field(description="Standard deviation")
    p5: float = Field(description="5th percentile")
    p50: float = Field(description="50th percentile (median)")
    p95: float = Field(description="95th percentile")
    min_value: float = Field(description="Minimum value")
    max_value: float = Field(description="Maximum value")
    uncertainty_range: str = Field(description="Uncertainty range (e.g., ±15%)")
    coefficient_of_variation: float = Field(description="CV = std_dev / mean")
    iterations: int = Field(description="Number of Monte Carlo iterations")


class EmissionFactorInfo(BaseModel):
    """Emission factor information with provenance."""

    factor_id: str = Field(description="Unique factor identifier")
    value: float = Field(description="Emission factor value")
    unit: str = Field(description="Unit of measurement")
    source: str = Field(description="Data source")
    source_version: str = Field(description="Source version")
    gwp_standard: str = Field(description="GWP standard (AR5, AR6)")
    uncertainty: float = Field(description="Uncertainty as decimal")
    data_quality_score: float = Field(description="DQI score (0-100)")
    reference_year: int = Field(description="Reference year")
    geographic_scope: str = Field(description="Geographic scope")
    hash: str = Field(description="SHA256 hash for provenance")


class DataQualityInfo(BaseModel):
    """Data quality information."""

    dqi_score: float = Field(ge=0, le=100, description="DQI score (0-100)")
    tier: TierType = Field(description="Data tier")
    rating: str = Field(description="Quality rating (excellent/good/fair/poor)")
    pedigree_score: Optional[float] = Field(default=None, description="Pedigree matrix score")
    warnings: List[str] = Field(default_factory=list)


class ProvenanceChain(BaseModel):
    """Complete provenance chain for calculation."""

    calculation_id: str = Field(description="Unique calculation identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    category: int = Field(description="Scope 3 category")
    tier: Optional[TierType] = Field(default=None, description="Calculation tier")

    input_data_hash: str = Field(description="SHA256 hash of input data")
    emission_factor: Optional[EmissionFactorInfo] = Field(default=None)

    calculation: Dict[str, Any] = Field(description="Calculation details")
    data_quality: DataQualityInfo = Field(description="Data quality information")

    provenance_chain: List[str] = Field(
        default_factory=list,
        description="Chain of hashes for full traceability"
    )

    opentelemetry_trace_id: Optional[str] = Field(
        default=None,
        description="OpenTelemetry trace ID for distributed tracing"
    )


class CalculationResult(BaseModel):
    """Generic calculation result for any category."""

    # Core results
    emissions_kgco2e: float = Field(description="Emissions in kgCO2e")
    emissions_tco2e: float = Field(description="Emissions in tCO2e")

    # Category and tier info
    category: int = Field(description="Scope 3 category")
    tier: Optional[TierType] = Field(default=None)

    # Uncertainty
    uncertainty: Optional[UncertaintyResult] = Field(default=None)

    # Data quality
    data_quality: DataQualityInfo

    # Provenance
    provenance: ProvenanceChain

    # Metadata
    calculation_method: str = Field(description="Calculation method used")
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def has_high_quality(self) -> bool:
        """Check if calculation has high data quality."""
        return self.data_quality.dqi_score >= 80.0

    @property
    def has_low_uncertainty(self) -> bool:
        """Check if calculation has low uncertainty."""
        if not self.uncertainty:
            return False
        return self.uncertainty.coefficient_of_variation < 0.2


class BatchResult(BaseModel):
    """Batch calculation result."""

    total_records: int = Field(description="Total number of records")
    successful_records: int = Field(description="Successfully calculated records")
    failed_records: int = Field(description="Failed records")

    total_emissions_kgco2e: float = Field(description="Total emissions in kgCO2e")
    total_emissions_tco2e: float = Field(description="Total emissions in tCO2e")

    results: List[CalculationResult] = Field(description="Individual results")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Error details")

    average_dqi_score: float = Field(description="Average DQI score")
    processing_time_seconds: float = Field(description="Total processing time")

    category: int = Field(description="Scope 3 category")

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_records == 0:
            return 0.0
        return self.successful_records / self.total_records


# ============================================================================
# ISO 14083 TEST CASE MODEL
# ============================================================================

class ISO14083TestCase(BaseModel):
    """Test case for ISO 14083 compliance verification."""

    test_id: str = Field(description="Test case identifier")
    description: str = Field(description="Test description")

    transport_mode: TransportMode
    distance_km: float
    weight_tonnes: float
    emission_factor: float

    expected_emissions_kgco2e: float = Field(description="Expected result")
    tolerance: float = Field(default=0.001, description="Acceptable variance")

    metadata: Dict[str, Any] = Field(default_factory=dict)


__all__ = [
    # Input models
    "Category1Input",
    "Category2Input",
    "Category3Input",
    "Category4Input",
    "Category5Input",
    "Category6Input",
    "Category6FlightInput",
    "Category6HotelInput",
    "Category6GroundTransportInput",
    "Category7Input",
    "Category8Input",
    "Category9Input",
    "Category10Input",
    "Category11Input",
    "Category12Input",
    "Category13Input",
    "Category14Input",
    "Category15Input",

    # Output models
    "CalculationResult",
    "BatchResult",
    "UncertaintyResult",
    "EmissionFactorInfo",
    "DataQualityInfo",
    "ProvenanceChain",

    # Test models
    "ISO14083TestCase",
]
