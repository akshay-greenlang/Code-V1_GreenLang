"""
ValueChain Intake Agent Configuration

Configuration management for intake agent operations.

Version: 1.0.0
Phase: 3 (Weeks 7-10)
Date: 2025-10-30
"""

from typing import Dict, Optional, List
from pydantic import BaseModel, Field
from pathlib import Path


# ============================================================================
# CONFIGURATION MODELS
# ============================================================================

class ParserConfig(BaseModel):
    """Configuration for file parsers."""

    # CSV Parser
    csv_delimiter: str = Field(default=",", description="CSV delimiter")
    csv_encoding_fallbacks: List[str] = Field(
        default=["utf-8", "latin-1", "cp1252", "iso-8859-1"],
        description="CSV encoding detection fallbacks"
    )
    csv_skip_rows: int = Field(default=0, ge=0, description="Rows to skip")

    # Excel Parser
    excel_sheet_name: Optional[str] = Field(None, description="Excel sheet name (None = first sheet)")
    excel_header_row: int = Field(default=0, ge=0, description="Header row index")

    # XML Parser
    xml_namespace_aware: bool = Field(default=True, description="XML namespace awareness")

    # PDF OCR Parser
    pdf_ocr_enabled: bool = Field(default=True, description="Enable PDF OCR")
    pdf_ocr_language: str = Field(default="eng", description="Tesseract OCR language")
    pdf_azure_form_recognizer_enabled: bool = Field(
        default=False,
        description="Enable Azure Form Recognizer"
    )


class EntityResolutionConfig(BaseModel):
    """Configuration for entity resolution."""

    # Confidence thresholds
    auto_match_threshold: float = Field(default=85.0, ge=0.0, le=100.0, description="Auto-match threshold")
    review_threshold: float = Field(default=85.0, ge=0.0, le=100.0, description="Review threshold")

    # Fuzzy matching
    fuzzy_algorithm: str = Field(default="token_sort_ratio", description="Fuzzy match algorithm")
    fuzzy_min_score: int = Field(default=70, ge=0, le=100, description="Min fuzzy score")

    # MDM Integration
    mdm_enabled: bool = Field(default=True, description="Enable MDM lookups")
    lei_lookup_enabled: bool = Field(default=True, description="Enable LEI lookups")
    duns_lookup_enabled: bool = Field(default=True, description="Enable DUNS lookups")
    opencorporates_enabled: bool = Field(default=False, description="Enable OpenCorporates lookups")

    # Caching
    cache_enabled: bool = Field(default=True, description="Enable resolution caching")
    cache_ttl_seconds: int = Field(default=3600, ge=0, description="Cache TTL in seconds")


class ReviewQueueConfig(BaseModel):
    """Configuration for review queue."""

    # Storage
    storage_type: str = Field(default="json", description="Storage type (json, sqlite)")
    storage_path: Path = Field(default=Path("data/review_queue"), description="Storage path")

    # Queue management
    max_queue_size: int = Field(default=10000, ge=1, description="Max queue size")
    auto_cleanup_days: int = Field(default=90, ge=1, description="Auto-cleanup after N days")

    # Priority rules
    high_priority_spend_threshold: float = Field(
        default=1000000.0,
        ge=0.0,
        description="Spend threshold for high priority"
    )


class DataQualityConfig(BaseModel):
    """Configuration for data quality assessment."""

    # DQI Settings
    dqi_enabled: bool = Field(default=True, description="Enable DQI calculation")
    dqi_pedigree_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="Pedigree weight")
    dqi_source_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Source weight")
    dqi_tier_weight: float = Field(default=0.2, ge=0.0, le=1.0, description="Tier weight")

    # Completeness
    completeness_critical_fields: List[str] = Field(
        default=["entity_name", "entity_identifier"],
        description="Critical fields for completeness"
    )

    # Validation
    schema_validation_enabled: bool = Field(default=True, description="Enable schema validation")
    strict_validation: bool = Field(default=False, description="Strict validation mode")


class PerformanceConfig(BaseModel):
    """Configuration for performance tuning."""

    # Batch processing
    batch_size: int = Field(default=1000, ge=1, le=100000, description="Batch processing size")
    max_workers: int = Field(default=4, ge=1, le=32, description="Max parallel workers")

    # Timeouts
    parser_timeout_seconds: int = Field(default=300, ge=1, description="Parser timeout")
    connector_timeout_seconds: int = Field(default=60, ge=1, description="Connector timeout")
    resolution_timeout_seconds: int = Field(default=30, ge=1, description="Resolution timeout")

    # Performance targets
    target_records_per_hour: int = Field(default=100000, ge=1, description="Target throughput")


class IntakeAgentConfig(BaseModel):
    """Main configuration for ValueChain Intake Agent."""

    # Sub-configurations
    parser: ParserConfig = Field(default_factory=ParserConfig, description="Parser configuration")
    resolution: EntityResolutionConfig = Field(
        default_factory=EntityResolutionConfig,
        description="Entity resolution configuration"
    )
    review_queue: ReviewQueueConfig = Field(
        default_factory=ReviewQueueConfig,
        description="Review queue configuration"
    )
    data_quality: DataQualityConfig = Field(
        default_factory=DataQualityConfig,
        description="Data quality configuration"
    )
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig,
        description="Performance configuration"
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )

    # Feature flags
    enable_gap_analysis: bool = Field(default=True, description="Enable gap analysis")
    enable_quality_scoring: bool = Field(default=True, description="Enable quality scoring")
    enable_entity_resolution: bool = Field(default=True, description="Enable entity resolution")


# ============================================================================
# CONFIGURATION FACTORY
# ============================================================================

_config_instance: Optional[IntakeAgentConfig] = None


def get_config() -> IntakeAgentConfig:
    """
    Get or create configuration instance (singleton pattern).

    Returns:
        IntakeAgentConfig: Configuration instance
    """
    global _config_instance

    if _config_instance is None:
        _config_instance = IntakeAgentConfig()

    return _config_instance


def set_config(config: IntakeAgentConfig) -> None:
    """
    Set global configuration instance.

    Args:
        config: Configuration instance to set
    """
    global _config_instance
    _config_instance = config


def reset_config() -> None:
    """Reset configuration to defaults."""
    global _config_instance
    _config_instance = None


def load_config_from_dict(config_dict: Dict) -> IntakeAgentConfig:
    """
    Load configuration from dictionary.

    Args:
        config_dict: Configuration dictionary

    Returns:
        IntakeAgentConfig: Configuration instance
    """
    config = IntakeAgentConfig(**config_dict)
    set_config(config)
    return config


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "ParserConfig",
    "EntityResolutionConfig",
    "ReviewQueueConfig",
    "DataQualityConfig",
    "PerformanceConfig",
    "IntakeAgentConfig",
    "get_config",
    "set_config",
    "reset_config",
    "load_config_from_dict",
]
