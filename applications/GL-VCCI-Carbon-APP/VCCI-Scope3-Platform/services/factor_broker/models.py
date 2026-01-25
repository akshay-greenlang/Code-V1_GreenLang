# -*- coding: utf-8 -*-
"""
Factor Broker Data Models
GL-VCCI Scope 3 Platform

Pydantic models for Factor Broker service including request/response models,
metadata, and data quality indicators.

Version: 1.0.0
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator, field_validator
from enum import Enum


class GWPStandard(str, Enum):
    """GWP (Global Warming Potential) standards."""
    AR5 = "AR5"  # IPCC Fifth Assessment Report (2014)
    AR6 = "AR6"  # IPCC Sixth Assessment Report (2021)


class SourceType(str, Enum):
    """Data source types."""
    ECOINVENT = "ecoinvent"
    DESNZ_UK = "desnz_uk"
    EPA_US = "epa_us"
    PROXY = "proxy"


class UnitType(str, Enum):
    """Common emission factor units."""
    KG_CO2E_PER_KG = "kgCO2e/kg"
    KG_CO2E_PER_KWH = "kgCO2e/kWh"
    KG_CO2E_PER_TONNE = "kgCO2e/tonne"
    KG_CO2E_PER_KM = "kgCO2e/km"
    KG_CO2E_PER_LITRE = "kgCO2e/litre"
    KG_CO2E_PER_M3 = "kgCO2e/m3"
    KG_CO2E_PER_UNIT = "kgCO2e/unit"


class DataQualityIndicator(BaseModel):
    """
    Data Quality Indicator (DQI) for emission factors.

    Based on the Pedigree Matrix approach used by ecoinvent and LCA databases.
    Scores indicate reliability, completeness, and temporal/geographical relevance.

    Attributes:
        reliability: Reliability of the source data (0-5, higher is better)
        completeness: Completeness of the dataset (0-5, higher is better)
        temporal_correlation: How recent the data is (0-5, higher is better)
        geographical_correlation: Geographic relevance (0-5, higher is better)
        technological_correlation: Technology relevance (0-5, higher is better)
        overall_score: Combined quality score (0-100)
    """
    reliability: int = Field(
        default=3,
        ge=0,
        le=5,
        description="Data source reliability (0=lowest, 5=highest)"
    )

    completeness: int = Field(
        default=3,
        ge=0,
        le=5,
        description="Dataset completeness (0=incomplete, 5=complete)"
    )

    temporal_correlation: int = Field(
        default=3,
        ge=0,
        le=5,
        description="Temporal relevance (0=old, 5=current)"
    )

    geographical_correlation: int = Field(
        default=3,
        ge=0,
        le=5,
        description="Geographic relevance (0=far, 5=exact match)"
    )

    technological_correlation: int = Field(
        default=3,
        ge=0,
        le=5,
        description="Technology relevance (0=different, 5=exact match)"
    )

    overall_score: int = Field(
        default=60,
        ge=0,
        le=100,
        description="Overall quality score (0-100)"
    )

    def calculate_overall_score(self) -> int:
        """
        Calculate overall quality score from individual indicators.

        Uses weighted average:
        - Reliability: 30%
        - Completeness: 25%
        - Temporal: 20%
        - Geographical: 15%
        - Technological: 10%

        Returns:
            Overall score (0-100)
        """
        score = (
            self.reliability * 0.30 +
            self.completeness * 0.25 +
            self.temporal_correlation * 0.20 +
            self.geographical_correlation * 0.15 +
            self.technological_correlation * 0.10
        ) / 5.0 * 100

        return int(score)

    class Config:
        json_schema_extra = {
            "example": {
                "reliability": 5,
                "completeness": 5,
                "temporal_correlation": 5,
                "geographical_correlation": 5,
                "technological_correlation": 5,
                "overall_score": 100
            }
        }


class ProvenanceInfo(BaseModel):
    """
    Provenance information for emission factor lookup.

    Tracks the lookup process for full reproducibility and audit trails.

    Attributes:
        lookup_timestamp: When the factor was looked up
        cache_hit: Whether the factor was served from cache
        is_proxy: Whether the factor is a calculated proxy
        fallback_chain: Sources tried in order
        proxy_method: Method used if proxy calculation was performed
        calculation_hash: SHA256 hash of the calculation
    """
    lookup_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp of factor lookup"
    )

    cache_hit: bool = Field(
        default=False,
        description="Whether factor was served from cache"
    )

    is_proxy: bool = Field(
        default=False,
        description="Whether factor is a calculated proxy"
    )

    fallback_chain: List[str] = Field(
        default_factory=list,
        description="Sources tried in cascade order"
    )

    proxy_method: Optional[str] = Field(
        default=None,
        description="Proxy calculation method if applicable"
    )

    calculation_hash: Optional[str] = Field(
        default=None,
        description="SHA256 hash for provenance chain"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "lookup_timestamp": "2025-01-25T10:30:00Z",
                "cache_hit": False,
                "is_proxy": False,
                "fallback_chain": ["ecoinvent"],
                "proxy_method": None,
                "calculation_hash": "a1b2c3d4e5f6..."
            }
        }


class FactorMetadata(BaseModel):
    """
    Metadata for an emission factor.

    Provides additional context and information about the factor,
    including source details, version, and data quality.

    Attributes:
        source: Data source
        source_version: Version of the source database
        source_dataset_id: Original dataset ID in source system
        gwp_standard: GWP standard used
        reference_year: Reference year for the factor
        last_updated: When the factor was last updated
        geographic_scope: Geographic scope of the factor
        technology_scope: Technology scope of the factor
        data_quality: Data quality indicator
        citation: Citation for the data source
        license_info: License information
    """
    source: SourceType = Field(
        description="Data source"
    )

    source_version: str = Field(
        description="Version of source database"
    )

    source_dataset_id: Optional[str] = Field(
        default=None,
        description="Original dataset ID in source system"
    )

    gwp_standard: GWPStandard = Field(
        default=GWPStandard.AR6,
        description="GWP standard used"
    )

    reference_year: int = Field(
        default=2024,
        ge=2000,
        le=2100,
        description="Reference year for the factor"
    )

    last_updated: Optional[datetime] = Field(
        default=None,
        description="When factor was last updated"
    )

    geographic_scope: str = Field(
        default="Global",
        description="Geographic scope (e.g., 'US', 'EU', 'Global')"
    )

    technology_scope: Optional[str] = Field(
        default=None,
        description="Technology scope (e.g., 'Average', 'Best Available')"
    )

    data_quality: DataQualityIndicator = Field(
        default_factory=DataQualityIndicator,
        description="Data quality indicator"
    )

    citation: Optional[str] = Field(
        default=None,
        description="Citation for data source"
    )

    license_info: Optional[str] = Field(
        default=None,
        description="License information"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "source": "ecoinvent",
                "source_version": "3.10",
                "source_dataset_id": "steel_production_us_001",
                "gwp_standard": "AR6",
                "reference_year": 2024,
                "geographic_scope": "US",
                "technology_scope": "Average mix",
                "data_quality": {
                    "reliability": 5,
                    "completeness": 5,
                    "temporal_correlation": 5,
                    "geographical_correlation": 5,
                    "technological_correlation": 5,
                    "overall_score": 100
                }
            }
        }


class FactorRequest(BaseModel):
    """
    Request model for emission factor lookup.

    Attributes:
        product: Product or service name
        region: ISO 3166-1 alpha-2 country code
        gwp_standard: GWP standard to use
        unit: Desired unit of measurement
        year: Reference year (defaults to latest)
        category: Product category for proxy calculation
    """
    product: str = Field(
        min_length=1,
        max_length=200,
        description="Product or service name"
    )

    region: str = Field(
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code (e.g., 'US', 'GB')"
    )

    gwp_standard: GWPStandard = Field(
        default=GWPStandard.AR6,
        description="GWP standard (AR5 or AR6)"
    )

    unit: Optional[str] = Field(
        default=None,
        description="Desired unit (e.g., 'kg', 'kWh')"
    )

    year: Optional[int] = Field(
        default=None,
        ge=2000,
        le=2100,
        description="Reference year (defaults to latest)"
    )

    category: Optional[str] = Field(
        default=None,
        description="Product category for proxy calculation"
    )

    @field_validator('region')
    @classmethod
    def validate_region(cls, v: str) -> str:
        """Validate region code format."""
        if not v.isupper():
            raise ValueError("Region code must be uppercase (e.g., 'US', not 'us')")
        if len(v) != 2:
            raise ValueError("Region code must be 2 characters (ISO 3166-1 alpha-2)")
        return v

    @field_validator('product')
    @classmethod
    def validate_product(cls, v: str) -> str:
        """Validate product name."""
        if not v.strip():
            raise ValueError("Product name cannot be empty")
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "product": "Steel",
                "region": "US",
                "gwp_standard": "AR6",
                "unit": "kg",
                "year": 2024,
                "category": "Metals"
            }
        }


class FactorResponse(BaseModel):
    """
    Response model for emission factor lookup.

    Attributes:
        factor_id: Unique factor identifier
        value: Emission factor value
        unit: Unit of measurement
        uncertainty: Uncertainty percentage (±%)
        metadata: Factor metadata
        provenance: Provenance information
        warning: Optional warning message (e.g., for low quality factors)
    """
    factor_id: str = Field(
        description="Unique factor identifier"
    )

    value: float = Field(
        gt=0,
        description="Emission factor value (must be positive)"
    )

    unit: str = Field(
        description="Unit of measurement (e.g., 'kgCO2e/kg')"
    )

    uncertainty: float = Field(
        ge=0,
        le=1,
        description="Uncertainty as decimal (e.g., 0.10 = ±10%)"
    )

    metadata: FactorMetadata = Field(
        description="Factor metadata"
    )

    provenance: ProvenanceInfo = Field(
        description="Provenance information"
    )

    warning: Optional[str] = Field(
        default=None,
        description="Warning message for low quality or proxy factors"
    )

    @property
    def data_quality_score(self) -> int:
        """Get overall data quality score."""
        return self.metadata.data_quality.overall_score

    @property
    def is_proxy(self) -> bool:
        """Check if factor is a proxy."""
        return self.provenance.is_proxy

    @property
    def source(self) -> str:
        """Get source name."""
        return self.metadata.source.value

    def to_calculation_input(self) -> Dict[str, Any]:
        """
        Convert to calculation input format.

        Returns:
            Dictionary suitable for Scope3Calculator
        """
        return {
            "factor_id": self.factor_id,
            "factor_value": self.value,
            "factor_unit": self.unit,
            "gwp_standard": self.metadata.gwp_standard.value,
            "uncertainty": self.uncertainty,
            "data_quality_score": self.data_quality_score
        }

    class Config:
        json_schema_extra = {
            "example": {
                "factor_id": "ecoinvent_3.10_steel_us_ar6_kg",
                "value": 1.85,
                "unit": "kgCO2e/kg",
                "uncertainty": 0.10,
                "metadata": {
                    "source": "ecoinvent",
                    "source_version": "3.10",
                    "gwp_standard": "AR6",
                    "reference_year": 2024,
                    "geographic_scope": "US",
                    "data_quality": {
                        "reliability": 5,
                        "completeness": 5,
                        "temporal_correlation": 5,
                        "geographical_correlation": 5,
                        "technological_correlation": 5,
                        "overall_score": 100
                    }
                },
                "provenance": {
                    "lookup_timestamp": "2025-01-25T10:30:00Z",
                    "cache_hit": False,
                    "is_proxy": False,
                    "fallback_chain": ["ecoinvent"]
                },
                "warning": None
            }
        }


class GWPComparisonRequest(BaseModel):
    """
    Request model for GWP standard comparison.

    Attributes:
        product: Product or service name
        region: ISO 3166-1 alpha-2 country code
        unit: Desired unit of measurement
    """
    product: str = Field(
        min_length=1,
        max_length=200,
        description="Product or service name"
    )

    region: str = Field(
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code"
    )

    unit: Optional[str] = Field(
        default=None,
        description="Desired unit"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "product": "Steel",
                "region": "US",
                "unit": "kg"
            }
        }


class GWPComparisonResponse(BaseModel):
    """
    Response model for GWP standard comparison.

    Attributes:
        product: Product name
        region: Region code
        ar5: Factor using AR5 standard
        ar6: Factor using AR6 standard
        difference_percent: Percentage difference (AR6 vs AR5)
        difference_absolute: Absolute difference
    """
    product: str = Field(description="Product name")
    region: str = Field(description="Region code")

    ar5: FactorResponse = Field(
        description="Factor using AR5 GWP standard"
    )

    ar6: FactorResponse = Field(
        description="Factor using AR6 GWP standard"
    )

    difference_percent: float = Field(
        description="Percentage difference ((AR6 - AR5) / AR5 * 100)"
    )

    difference_absolute: float = Field(
        description="Absolute difference (AR6 - AR5)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "product": "Steel",
                "region": "US",
                "ar5": {
                    "factor_id": "ecoinvent_3.10_steel_us_ar5_kg",
                    "value": 1.82,
                    "unit": "kgCO2e/kg"
                },
                "ar6": {
                    "factor_id": "ecoinvent_3.10_steel_us_ar6_kg",
                    "value": 1.85,
                    "unit": "kgCO2e/kg"
                },
                "difference_percent": 1.65,
                "difference_absolute": 0.03
            }
        }


class HealthCheckResponse(BaseModel):
    """
    Response model for health check endpoint.

    Attributes:
        status: Overall service status
        cache_hit_rate: Current cache hit rate
        average_latency_ms: Average latency in milliseconds
        data_sources: Status of each data source
        timestamp: Timestamp of health check
    """
    status: str = Field(
        description="Service status (healthy, degraded, unhealthy)"
    )

    cache_hit_rate: float = Field(
        ge=0,
        le=1,
        description="Cache hit rate (0-1)"
    )

    average_latency_ms: float = Field(
        ge=0,
        description="Average latency in milliseconds"
    )

    data_sources: Dict[str, Dict[str, Any]] = Field(
        description="Status of each data source"
    )

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of health check"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "cache_hit_rate": 0.87,
                "average_latency_ms": 42.5,
                "data_sources": {
                    "ecoinvent": {
                        "status": "healthy",
                        "last_check": "2025-01-25T10:30:00Z",
                        "success_rate": 0.99
                    },
                    "desnz_uk": {
                        "status": "healthy",
                        "last_check": "2025-01-25T10:30:00Z",
                        "success_rate": 1.0
                    }
                },
                "timestamp": "2025-01-25T10:30:00Z"
            }
        }
