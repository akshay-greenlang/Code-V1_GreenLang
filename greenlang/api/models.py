"""
greenlang/api/models.py

Pydantic models for REST API requests and responses.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
from datetime import date, datetime
from enum import Enum


# ==================== ENUMS ====================

class ScopeEnum(str, Enum):
    """GHG Protocol scopes"""
    SCOPE_1 = "1"
    SCOPE_2 = "2"
    SCOPE_3 = "3"


class BoundaryEnum(str, Enum):
    """Emission boundaries"""
    COMBUSTION = "combustion"
    WTT = "WTT"
    WTW = "WTW"
    CRADLE_TO_GATE = "cradle_to_gate"
    CRADLE_TO_GRAVE = "cradle_to_grave"


class GWPSetEnum(str, Enum):
    """GWP reference sets"""
    IPCC_AR6_100 = "IPCC_AR6_100"
    IPCC_AR6_20 = "IPCC_AR6_20"
    IPCC_AR5_100 = "IPCC_AR5_100"


# ==================== REQUEST MODELS ====================

class CalculationRequest(BaseModel):
    """Single calculation request"""
    factor_id: Optional[str] = Field(
        None,
        description="Emission factor ID (if known)"
    )
    fuel_type: str = Field(
        ...,
        description="Fuel type: diesel, gasoline, natural_gas, electricity, etc.",
        example="diesel"
    )
    activity_amount: float = Field(
        ...,
        gt=0,
        description="Activity amount (must be positive)",
        example=100.0
    )
    activity_unit: str = Field(
        ...,
        description="Activity unit: gallons, kWh, therms, liters, etc.",
        example="gallons"
    )
    geography: Optional[str] = Field(
        "US",
        description="Geography: ISO country code (US, UK, EU, etc.)",
        example="US"
    )
    scope: Optional[ScopeEnum] = Field(
        ScopeEnum.SCOPE_1,
        description="GHG scope (1, 2, or 3)"
    )
    boundary: Optional[BoundaryEnum] = Field(
        BoundaryEnum.COMBUSTION,
        description="Emission boundary"
    )
    gwp_set: Optional[GWPSetEnum] = Field(
        GWPSetEnum.IPCC_AR6_100,
        description="GWP reference set"
    )
    calculation_date: Optional[date] = Field(
        None,
        description="Calculation date for historical queries"
    )

    class Config:
        schema_extra = {
            "example": {
                "fuel_type": "diesel",
                "activity_amount": 100,
                "activity_unit": "gallons",
                "geography": "US",
                "scope": "1",
                "boundary": "combustion"
            }
        }


class Scope1Request(BaseModel):
    """Scope 1 calculation request (direct emissions)"""
    fuel_type: str = Field(..., description="Fuel type")
    consumption: float = Field(..., gt=0, description="Fuel consumption")
    unit: str = Field(..., description="Consumption unit")
    geography: str = Field("US", description="Geography")

    class Config:
        schema_extra = {
            "example": {
                "fuel_type": "natural_gas",
                "consumption": 500,
                "unit": "therms",
                "geography": "US"
            }
        }


class Scope2Request(BaseModel):
    """Scope 2 calculation request (purchased electricity)"""
    electricity_kwh: float = Field(..., gt=0, description="Electricity consumption in kWh")
    geography: str = Field("US", description="Grid geography")
    market_based_factor: Optional[float] = Field(
        None,
        description="Optional market-based factor (for renewable energy purchases)"
    )

    class Config:
        schema_extra = {
            "example": {
                "electricity_kwh": 10000,
                "geography": "US"
            }
        }


class Scope3Request(BaseModel):
    """Scope 3 calculation request (indirect emissions)"""
    category: str = Field(
        ...,
        description="Scope 3 category (business_travel, employee_commuting, etc.)"
    )
    activity_data: Dict[str, float] = Field(
        ...,
        description="Activity data specific to category"
    )
    geography: str = Field("US", description="Geography")

    class Config:
        schema_extra = {
            "example": {
                "category": "business_travel",
                "activity_data": {
                    "miles_driven": 1000,
                    "vehicle_type": "sedan"
                },
                "geography": "US"
            }
        }


class BatchCalculationRequest(BaseModel):
    """Batch calculation request"""
    calculations: List[CalculationRequest] = Field(
        ...,
        max_items=100,
        description="List of calculations (max 100)"
    )

    @validator('calculations')
    def validate_batch_size(cls, v):
        if len(v) > 100:
            raise ValueError("Maximum 100 calculations per batch request")
        return v


# ==================== RESPONSE MODELS ====================

class GHGBreakdown(BaseModel):
    """Greenhouse gas breakdown"""
    CO2: float = Field(..., description="CO2 emissions in kg")
    CH4: float = Field(..., description="CH4 emissions in kg")
    N2O: float = Field(..., description="N2O emissions in kg")
    HFCs: Optional[float] = Field(None, description="HFC emissions in kg")
    PFCs: Optional[float] = Field(None, description="PFC emissions in kg")
    SF6: Optional[float] = Field(None, description="SF6 emissions in kg")


class EmissionResult(BaseModel):
    """Emission calculation result"""
    emissions_kg_co2e: float = Field(..., description="Total CO2e emissions in kg")
    emissions_tonnes_co2e: float = Field(..., description="Total CO2e emissions in tonnes")
    gas_breakdown: GHGBreakdown = Field(..., description="Individual gas breakdown")


class EmissionFactorSummary(BaseModel):
    """Summary of emission factor used"""
    factor_id: str
    fuel_type: str
    unit: str
    geography: str
    scope: str
    boundary: str
    co2e_per_unit: float = Field(..., description="kg CO2e per unit")
    source: str = Field(..., description="Source organization")
    source_year: int
    data_quality_score: float = Field(..., description="Data quality score (1-5)")
    uncertainty_percent: float = Field(..., description="Uncertainty as percentage")


class CalculationResponse(BaseModel):
    """Single calculation response"""
    calculation_id: str = Field(..., description="Unique calculation ID")
    emissions_kg_co2e: float = Field(..., description="Total CO2e emissions in kg")
    emissions_tonnes_co2e: float = Field(..., description="Total CO2e emissions in tonnes")
    emissions_by_gas: Dict[str, float] = Field(..., description="Emissions by gas type")
    factor_used: EmissionFactorSummary = Field(..., description="Factor metadata")
    timestamp: datetime = Field(..., description="Calculation timestamp")

    class Config:
        schema_extra = {
            "example": {
                "calculation_id": "calc_abc123xyz",
                "emissions_kg_co2e": 1021.0,
                "emissions_tonnes_co2e": 1.021,
                "emissions_by_gas": {
                    "CO2": 1018.0,
                    "CH4": 2.3,
                    "N2O": 0.7
                },
                "factor_used": {
                    "factor_id": "EF:US:diesel:2024:v1",
                    "fuel_type": "diesel",
                    "unit": "gallons",
                    "geography": "US",
                    "scope": "1",
                    "boundary": "combustion",
                    "co2e_per_unit": 10.21,
                    "source": "EPA",
                    "source_year": 2024,
                    "data_quality_score": 4.6,
                    "uncertainty_percent": 5.0
                },
                "timestamp": "2025-11-19T10:30:00Z"
            }
        }


class BatchCalculationResponse(BaseModel):
    """Batch calculation response"""
    batch_id: str = Field(..., description="Unique batch ID")
    total_emissions_kg_co2e: float = Field(..., description="Total emissions across all calculations")
    total_emissions_tonnes_co2e: float = Field(..., description="Total emissions in tonnes")
    calculations: List[CalculationResponse] = Field(..., description="Individual calculation results")
    count: int = Field(..., description="Number of calculations")
    timestamp: datetime


class DataQuality(BaseModel):
    """Data quality indicators"""
    temporal: int = Field(..., ge=1, le=5)
    geographical: int = Field(..., ge=1, le=5)
    technological: int = Field(..., ge=1, le=5)
    representativeness: int = Field(..., ge=1, le=5)
    methodological: int = Field(..., ge=1, le=5)
    overall_score: float
    rating: str


class SourceInfo(BaseModel):
    """Source provenance information"""
    organization: str
    publication: str
    year: int
    url: Optional[str] = None
    methodology: str
    version: str


class EmissionFactorResponse(BaseModel):
    """Detailed emission factor response"""
    factor_id: str
    fuel_type: str
    unit: str
    geography: str
    geography_level: str
    scope: str
    boundary: str

    # Emission vectors
    co2_per_unit: float
    ch4_per_unit: float
    n2o_per_unit: float
    co2e_per_unit: float

    # GWP info
    gwp_set: str
    ch4_gwp: float
    n2o_gwp: float

    # Quality and provenance
    data_quality: DataQuality
    source: SourceInfo
    uncertainty_95ci: float

    # Validity
    valid_from: date
    valid_to: Optional[date] = None

    # Metadata
    license: str
    compliance_frameworks: List[str]
    tags: List[str]
    notes: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "factor_id": "EF:US:diesel:2024:v1",
                "fuel_type": "diesel",
                "unit": "gallons",
                "geography": "US",
                "geography_level": "country",
                "scope": "1",
                "boundary": "combustion",
                "co2_per_unit": 10.18,
                "ch4_per_unit": 0.00082,
                "n2o_per_unit": 0.000164,
                "co2e_per_unit": 10.21,
                "gwp_set": "IPCC_AR6_100",
                "ch4_gwp": 28,
                "n2o_gwp": 273,
                "data_quality": {
                    "temporal": 5,
                    "geographical": 4,
                    "technological": 4,
                    "representativeness": 4,
                    "methodological": 5,
                    "overall_score": 4.6,
                    "rating": "excellent"
                },
                "source": {
                    "organization": "EPA",
                    "publication": "Emission Factors for GHG Inventories 2024",
                    "year": 2024,
                    "methodology": "IPCC_Tier_1",
                    "version": "v1"
                },
                "uncertainty_95ci": 0.05,
                "valid_from": "2024-01-01",
                "valid_to": "2024-12-31",
                "license": "CC0-1.0",
                "compliance_frameworks": ["GHG_Protocol", "IPCC_2006", "EPA_MRR"],
                "tags": ["fossil", "transport", "stationary"]
            }
        }


class FactorListResponse(BaseModel):
    """Paginated list of emission factors"""
    factors: List[EmissionFactorSummary]
    total_count: int = Field(..., description="Total number of factors matching query")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")


class FactorSearchResponse(BaseModel):
    """Search results for factors"""
    query: str = Field(..., description="Search query")
    factors: List[EmissionFactorSummary]
    count: int = Field(..., description="Number of results")
    search_time_ms: float = Field(..., description="Search execution time in ms")


class CoverageStats(BaseModel):
    """Coverage statistics"""
    total_factors: int
    geographies: int
    fuel_types: int
    scopes: Dict[str, int]
    boundaries: Dict[str, int]
    by_geography: Dict[str, int]
    by_fuel_type: Dict[str, int]


class CacheStats(BaseModel):
    """Cache statistics"""
    enabled: bool
    hits: int
    misses: int
    hit_rate_pct: float
    size: int
    max_size: int


class StatsResponse(BaseModel):
    """API statistics"""
    version: str
    total_factors: int
    calculations_today: int
    cache_stats: CacheStats
    uptime_seconds: float
    timestamp: datetime


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="healthy or unhealthy")
    version: str
    timestamp: datetime
    database: str = Field(..., description="Database status")
    cache: str = Field(..., description="Cache status")
    uptime_seconds: float

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2025-11-19T10:30:00Z",
                "database": "connected",
                "cache": "available",
                "uptime_seconds": 86400.0
            }
        }


class ErrorResponse(BaseModel):
    """Error response"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict] = Field(None, description="Additional error details")
    timestamp: datetime

    class Config:
        schema_extra = {
            "example": {
                "error": "validation_error",
                "message": "Invalid fuel type: xyz",
                "details": {
                    "field": "fuel_type",
                    "allowed_values": ["diesel", "gasoline", "natural_gas", "electricity"]
                },
                "timestamp": "2025-11-19T10:30:00Z"
            }
        }
