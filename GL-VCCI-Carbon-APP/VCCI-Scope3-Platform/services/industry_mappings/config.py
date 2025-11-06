"""
Configuration Management for Industry Mappings

Manages configuration settings, match thresholds, and regional defaults.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, validator
from pathlib import Path
import os


class MatchThresholds(BaseModel):
    """Matching threshold configuration"""
    exact_match: float = Field(default=1.0, ge=0.0, le=1.0, description="Exact match threshold")
    high_confidence: float = Field(default=0.9, ge=0.0, le=1.0, description="High confidence threshold")
    medium_confidence: float = Field(default=0.7, ge=0.0, le=1.0, description="Medium confidence threshold")
    low_confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Low confidence threshold")
    minimum_match: float = Field(default=0.4, ge=0.0, le=1.0, description="Minimum acceptable match")
    fuzzy_match: float = Field(default=0.8, ge=0.0, le=1.0, description="Fuzzy match threshold")
    keyword_match: float = Field(default=0.75, ge=0.0, le=1.0, description="Keyword match threshold")
    partial_match: float = Field(default=0.6, ge=0.0, le=1.0, description="Partial match threshold")

    @validator("*", pre=True)
    def validate_threshold(cls, v):
        """Ensure all thresholds are between 0 and 1"""
        if isinstance(v, (int, float)):
            if not 0.0 <= v <= 1.0:
                raise ValueError("Threshold must be between 0.0 and 1.0")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "exact_match": 1.0,
                "high_confidence": 0.9,
                "medium_confidence": 0.7,
                "low_confidence": 0.5,
                "minimum_match": 0.4
            }
        }


class RegionalConfig(BaseModel):
    """Regional configuration settings"""
    region_code: str = Field(..., description="ISO region code (US, EU, CN, etc.)")
    region_name: str = Field(..., description="Region name")
    primary_taxonomy: str = Field(..., description="Primary taxonomy (NAICS, ISIC, etc.)")
    language: str = Field(default="en", description="Primary language code")
    unit_system: str = Field(default="metric", description="Unit system (metric/imperial)")
    currency: str = Field(default="USD", description="Default currency")
    timezone: str = Field(default="UTC", description="Default timezone")
    regional_adjustments: Dict[str, float] = Field(
        default_factory=dict,
        description="Regional adjustment factors"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "region_code": "US",
                "region_name": "United States",
                "primary_taxonomy": "NAICS",
                "language": "en",
                "unit_system": "imperial",
                "currency": "USD"
            }
        }


class SearchConfig(BaseModel):
    """Search configuration settings"""
    max_results: int = Field(default=10, ge=1, le=100, description="Maximum search results")
    enable_fuzzy: bool = Field(default=True, description="Enable fuzzy matching")
    enable_stemming: bool = Field(default=True, description="Enable word stemming")
    enable_synonyms: bool = Field(default=True, description="Enable synonym matching")
    case_sensitive: bool = Field(default=False, description="Case sensitive search")
    whole_word_only: bool = Field(default=False, description="Match whole words only")
    include_inactive: bool = Field(default=False, description="Include inactive codes")
    keyword_boost: float = Field(default=1.5, ge=1.0, le=3.0, description="Keyword match boost")
    exact_boost: float = Field(default=2.0, ge=1.0, le=5.0, description="Exact match boost")
    min_keyword_length: int = Field(default=3, ge=2, le=10, description="Min keyword length")

    class Config:
        json_schema_extra = {
            "example": {
                "max_results": 10,
                "enable_fuzzy": True,
                "enable_stemming": True,
                "keyword_boost": 1.5,
                "exact_boost": 2.0
            }
        }


class CacheConfig(BaseModel):
    """Cache configuration settings"""
    enable_cache: bool = Field(default=True, description="Enable result caching")
    cache_ttl: int = Field(default=3600, ge=0, description="Cache TTL in seconds")
    max_cache_size: int = Field(default=10000, ge=100, description="Max cache entries")
    cache_strategy: str = Field(default="LRU", description="Cache eviction strategy")
    warm_cache_on_startup: bool = Field(default=True, description="Warm cache on startup")
    cache_hit_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Cache hit rate threshold")

    class Config:
        json_schema_extra = {
            "example": {
                "enable_cache": True,
                "cache_ttl": 3600,
                "max_cache_size": 10000,
                "cache_strategy": "LRU"
            }
        }


class PerformanceConfig(BaseModel):
    """Performance configuration settings"""
    max_processing_time_ms: int = Field(default=10, ge=1, description="Max processing time (ms)")
    enable_parallel_search: bool = Field(default=True, description="Enable parallel search")
    max_workers: int = Field(default=4, ge=1, le=16, description="Max parallel workers")
    batch_size: int = Field(default=100, ge=1, description="Batch processing size")
    enable_profiling: bool = Field(default=False, description="Enable performance profiling")
    log_slow_queries: bool = Field(default=True, description="Log slow queries")
    slow_query_threshold_ms: int = Field(default=50, ge=1, description="Slow query threshold (ms)")

    class Config:
        json_schema_extra = {
            "example": {
                "max_processing_time_ms": 10,
                "enable_parallel_search": True,
                "max_workers": 4,
                "batch_size": 100
            }
        }


class DataConfig(BaseModel):
    """Data file configuration"""
    data_dir: Path = Field(..., description="Data directory path")
    naics_file: str = Field(default="naics_2022.csv", description="NAICS data file")
    isic_file: str = Field(default="isic_rev4.csv", description="ISIC data file")
    taxonomy_file: str = Field(default="custom_taxonomy.csv", description="Custom taxonomy file")
    crosswalk_file: str = Field(default="naics_isic_crosswalk.csv", description="Crosswalk file")
    auto_reload: bool = Field(default=False, description="Auto-reload data on change")
    validate_on_load: bool = Field(default=True, description="Validate data on load")
    backup_enabled: bool = Field(default=True, description="Enable data backups")

    @validator("data_dir")
    def validate_data_dir(cls, v):
        """Ensure data directory exists"""
        path = Path(v)
        if not path.exists():
            # Try to create it
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ValueError(f"Cannot create data directory: {e}")
        return path

    class Config:
        json_schema_extra = {
            "example": {
                "data_dir": "./data/industry_mappings",
                "naics_file": "naics_2022.csv",
                "isic_file": "isic_rev4.csv",
                "auto_reload": False
            }
        }


class IndustryMappingConfig(BaseModel):
    """Main Industry Mapping Configuration"""
    environment: str = Field(default="production", description="Environment (dev/staging/production)")
    version: str = Field(default="1.0.0", description="Config version")
    match_thresholds: MatchThresholds = Field(default_factory=MatchThresholds)
    search: SearchConfig = Field(default_factory=SearchConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    regional_configs: Dict[str, RegionalConfig] = Field(default_factory=dict)
    default_region: str = Field(default="US", description="Default region code")

    # Data configuration
    data: Optional[DataConfig] = None

    # Feature flags
    enable_ml_classification: bool = Field(default=False, description="Enable ML classification")
    enable_crosswalk: bool = Field(default=True, description="Enable NAICS-ISIC crosswalk")
    enable_validation: bool = Field(default=True, description="Enable mapping validation")
    enable_logging: bool = Field(default=True, description="Enable detailed logging")
    log_level: str = Field(default="INFO", description="Logging level")

    # Quality settings
    require_validation: bool = Field(default=True, description="Require validation before use")
    min_coverage_threshold: float = Field(default=0.9, ge=0.0, le=1.0, description="Min coverage threshold")
    quality_check_interval: int = Field(default=86400, ge=0, description="Quality check interval (seconds)")

    class Config:
        json_schema_extra = {
            "example": {
                "environment": "production",
                "version": "1.0.0",
                "default_region": "US",
                "enable_ml_classification": False,
                "min_coverage_threshold": 0.9
            }
        }


# Default regional configurations
DEFAULT_REGIONAL_CONFIGS: Dict[str, RegionalConfig] = {
    "US": RegionalConfig(
        region_code="US",
        region_name="United States",
        primary_taxonomy="NAICS",
        language="en",
        unit_system="imperial",
        currency="USD",
        timezone="America/New_York"
    ),
    "EU": RegionalConfig(
        region_code="EU",
        region_name="European Union",
        primary_taxonomy="ISIC",
        language="en",
        unit_system="metric",
        currency="EUR",
        timezone="Europe/Brussels"
    ),
    "UK": RegionalConfig(
        region_code="UK",
        region_name="United Kingdom",
        primary_taxonomy="ISIC",
        language="en",
        unit_system="metric",
        currency="GBP",
        timezone="Europe/London"
    ),
    "CN": RegionalConfig(
        region_code="CN",
        region_name="China",
        primary_taxonomy="ISIC",
        language="zh",
        unit_system="metric",
        currency="CNY",
        timezone="Asia/Shanghai"
    ),
    "IN": RegionalConfig(
        region_code="IN",
        region_name="India",
        primary_taxonomy="ISIC",
        language="en",
        unit_system="metric",
        currency="INR",
        timezone="Asia/Kolkata"
    ),
    "CA": RegionalConfig(
        region_code="CA",
        region_name="Canada",
        primary_taxonomy="NAICS",
        language="en",
        unit_system="metric",
        currency="CAD",
        timezone="America/Toronto"
    ),
    "AU": RegionalConfig(
        region_code="AU",
        region_name="Australia",
        primary_taxonomy="ISIC",
        language="en",
        unit_system="metric",
        currency="AUD",
        timezone="Australia/Sydney"
    )
}


def get_default_config() -> IndustryMappingConfig:
    """Get default configuration with all regional configs"""
    # Get the data directory relative to this file
    current_dir = Path(__file__).parent
    data_dir = current_dir / "data" / "industry_mappings"

    data_config = DataConfig(
        data_dir=data_dir,
        naics_file="naics_2022.csv",
        isic_file="isic_rev4.csv",
        taxonomy_file="custom_taxonomy.csv",
        crosswalk_file="naics_isic_crosswalk.csv"
    )

    config = IndustryMappingConfig(
        environment=os.getenv("ENVIRONMENT", "production"),
        data=data_config,
        regional_configs=DEFAULT_REGIONAL_CONFIGS,
        default_region="US"
    )

    return config


def get_regional_config(region_code: str) -> Optional[RegionalConfig]:
    """Get regional configuration by code"""
    return DEFAULT_REGIONAL_CONFIGS.get(region_code.upper())


def create_custom_config(
    environment: str = "production",
    match_thresholds: Optional[MatchThresholds] = None,
    region: str = "US"
) -> IndustryMappingConfig:
    """Create custom configuration"""
    config = get_default_config()
    config.environment = environment
    config.default_region = region

    if match_thresholds:
        config.match_thresholds = match_thresholds

    return config


# Export default instance
default_config = get_default_config()
