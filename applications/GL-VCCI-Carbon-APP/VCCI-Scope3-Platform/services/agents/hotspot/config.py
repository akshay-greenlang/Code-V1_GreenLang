# -*- coding: utf-8 -*-
"""
HotspotAnalysisAgent Configuration
GL-VCCI Scope 3 Platform

Configuration constants and settings for hotspot analysis.

Version: 1.0.0
Phase: 3 (Weeks 14-16)
Date: 2025-10-30
"""

from typing import Dict, List
from pydantic import BaseModel, Field
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class AnalysisDimension(str, Enum):
    """Available dimensions for segmentation analysis."""
    SUPPLIER = "supplier"
    CATEGORY = "category"
    PRODUCT = "product"
    REGION = "region"
    FACILITY = "facility"
    TIME = "time"


class ScenarioType(str, Enum):
    """Scenario types for emission reduction modeling."""
    SUPPLIER_SWITCH = "supplier_switch"
    MODAL_SHIFT = "modal_shift"
    PRODUCT_SUBSTITUTION = "product_substitution"
    EFFICIENCY_IMPROVEMENT = "efficiency_improvement"
    RENEWABLE_ENERGY = "renewable_energy"


class InsightPriority(str, Enum):
    """Priority levels for insights."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class InsightType(str, Enum):
    """Types of insights generated."""
    HIGH_EMISSIONS_SUPPLIER = "high_emissions_supplier"
    HIGH_EMISSIONS_CATEGORY = "high_emissions_category"
    LOW_DATA_QUALITY = "low_data_quality"
    CONCENTRATION_RISK = "concentration_risk"
    QUICK_WIN = "quick_win"
    COST_EFFECTIVE_REDUCTION = "cost_effective_reduction"
    TIER_UPGRADE_OPPORTUNITY = "tier_upgrade_opportunity"


# ============================================================================
# CONFIGURATION MODELS
# ============================================================================

class HotspotCriteria(BaseModel):
    """Criteria for identifying emissions hotspots."""

    emission_threshold_tco2e: float = Field(
        default=1000.0,
        gt=0,
        description="Flag if emissions > threshold (tCO2e)"
    )

    percent_threshold: float = Field(
        default=5.0,
        gt=0,
        le=100,
        description="Flag if > X% of total emissions"
    )

    dqi_threshold: float = Field(
        default=50.0,
        ge=0,
        le=100,
        description="Flag if DQI < threshold (poor quality)"
    )

    tier_threshold: int = Field(
        default=3,
        ge=1,
        le=3,
        description="Flag if data tier >= threshold"
    )

    concentration_threshold: float = Field(
        default=30.0,
        gt=0,
        le=100,
        description="Flag if single entity > X% (concentration risk)"
    )


class ParetoConfig(BaseModel):
    """Configuration for Pareto analysis."""

    pareto_threshold: float = Field(
        default=0.80,
        gt=0,
        le=1,
        description="Pareto threshold (e.g., 0.80 for 80/20 rule)"
    )

    top_n_percent: float = Field(
        default=0.20,
        gt=0,
        le=1,
        description="Top N percent to analyze (e.g., 0.20 for top 20%)"
    )

    min_records: int = Field(
        default=5,
        ge=1,
        description="Minimum records required for Pareto analysis"
    )


class ROIConfig(BaseModel):
    """Configuration for ROI calculations."""

    discount_rate: float = Field(
        default=0.08,
        ge=0,
        le=1,
        description="Discount rate for NPV calculations (e.g., 0.08 = 8%)"
    )

    analysis_period_years: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Analysis period for NPV/IRR (years)"
    )

    carbon_price_usd_per_tco2e: float = Field(
        default=50.0,
        ge=0,
        description="Carbon price for valuation (USD/tCO2e)"
    )


class SegmentationConfig(BaseModel):
    """Configuration for segmentation analysis."""

    max_segments_per_dimension: int = Field(
        default=50,
        ge=1,
        description="Maximum segments to return per dimension"
    )

    min_emission_threshold_tco2e: float = Field(
        default=0.1,
        ge=0,
        description="Minimum emissions to include in segments (tCO2e)"
    )

    aggregate_small_segments: bool = Field(
        default=True,
        description="Aggregate small segments into 'Other'"
    )


class HotspotAnalysisConfig(BaseModel):
    """Master configuration for HotspotAnalysisAgent."""

    hotspot_criteria: HotspotCriteria = Field(
        default_factory=HotspotCriteria,
        description="Hotspot identification criteria"
    )

    pareto_config: ParetoConfig = Field(
        default_factory=ParetoConfig,
        description="Pareto analysis configuration"
    )

    roi_config: ROIConfig = Field(
        default_factory=ROIConfig,
        description="ROI calculation configuration"
    )

    segmentation_config: SegmentationConfig = Field(
        default_factory=SegmentationConfig,
        description="Segmentation analysis configuration"
    )

    # Performance settings
    max_records_in_memory: int = Field(
        default=100000,
        ge=1000,
        description="Maximum records to process in memory"
    )

    enable_parallel_processing: bool = Field(
        default=True,
        description="Enable parallel processing for large datasets"
    )


# ============================================================================
# DEFAULT CONFIGURATIONS
# ============================================================================

DEFAULT_CONFIG = HotspotAnalysisConfig()


# ============================================================================
# CONSTANTS
# ============================================================================

# Dimension field mappings
DIMENSION_FIELD_MAP: Dict[AnalysisDimension, str] = {
    AnalysisDimension.SUPPLIER: "supplier_name",
    AnalysisDimension.CATEGORY: "scope3_category",
    AnalysisDimension.PRODUCT: "product_name",
    AnalysisDimension.REGION: "region",
    AnalysisDimension.FACILITY: "facility_name",
    AnalysisDimension.TIME: "time_period"
}

# Required fields for emissions data
REQUIRED_EMISSION_FIELDS: List[str] = [
    "emissions_tco2e",
    "emissions_kgco2e"
]

# Optional fields for enhanced analysis
OPTIONAL_EMISSION_FIELDS: List[str] = [
    "supplier_name",
    "scope3_category",
    "product_name",
    "region",
    "facility_name",
    "dqi_score",
    "tier",
    "uncertainty_pct",
    "spend_usd",
    "calculation_date"
]


__all__ = [
    # Enums
    "AnalysisDimension",
    "ScenarioType",
    "InsightPriority",
    "InsightType",

    # Config Models
    "HotspotCriteria",
    "ParetoConfig",
    "ROIConfig",
    "SegmentationConfig",
    "HotspotAnalysisConfig",

    # Defaults
    "DEFAULT_CONFIG",

    # Constants
    "DIMENSION_FIELD_MAP",
    "REQUIRED_EMISSION_FIELDS",
    "OPTIONAL_EMISSION_FIELDS",
]
