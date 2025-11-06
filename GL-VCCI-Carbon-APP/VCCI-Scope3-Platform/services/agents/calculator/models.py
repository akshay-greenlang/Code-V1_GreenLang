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

from .config import TierType, TransportMode, CabinClass


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
    "Category4Input",
    "Category6Input",
    "Category6FlightInput",
    "Category6HotelInput",
    "Category6GroundTransportInput",

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
