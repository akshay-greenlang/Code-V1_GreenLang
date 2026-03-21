"""
PACK-029 Interim Targets Pack - Configuration Manager

This module implements the InterimTargetsConfig and PackConfig classes that load,
merge, and validate all configuration for the Interim Targets Pack. It provides
comprehensive Pydantic v2 models for interim target setting, progress monitoring,
variance analysis, corrective action planning, carbon budget allocation, trend
extrapolation, and multi-framework reporting aligned with SBTi near-term and
long-term target requirements.

Interim Target Architecture:
    - 5-Year Target (2030): SBTi near-term, minimum 42% (1.5C) or 30% (WB2C)
    - 10-Year Target (2035): Interpolated milestone between near-term and net-zero
    - Custom Milestones: User-defined interim checkpoints (e.g., 2028, 2032, 2038)
    - Annual Carbon Budgets: Year-by-year emission allowances along the pathway

Pathway Types (4 models):
    - LINEAR: Constant absolute reduction per year (simplest, SBTi default)
    - MILESTONE_BASED: Stepped reductions at defined milestone years
    - ACCELERATING: Increasing rate of reduction (back-loaded effort)
    - S_CURVE: Sigmoidal curve (slow start, rapid mid-phase, plateau)

Variance Analysis Methods:
    - LMDI: Logarithmic Mean Divisia Index decomposition
    - KAYA: Kaya Identity decomposition (GDP, energy intensity, carbon intensity)
    - BOTH: Combined LMDI + Kaya analysis for comprehensive attribution

Trend Extrapolation Methods:
    - LINEAR: Simple linear regression extrapolation
    - EXPONENTIAL_SMOOTHING: Holt-Winters exponential smoothing
    - ARIMA: Auto-Regressive Integrated Moving Average time series forecasting

Performance Scoring:
    - GREEN: On track or ahead of target (variance <= tolerance)
    - AMBER: Behind target but recoverable (variance within 2x tolerance)
    - RED: Significantly behind target (variance > 2x tolerance, corrective action needed)

Corrective Action Framework:
    - Gap Quantification: tCO2e gap to target and cumulative budget impact
    - Initiative Optimization: Re-sequence and accelerate existing initiatives
    - MACC Integration: Pull forward cost-effective levers from PACK-028
    - Scenario Remodeling: Recalculate pathway with updated assumptions

SBTi Validation (21 criteria per SBTi Corporate Net-Zero Standard v1.0):
    - C1: Boundary completeness (>= 95% Scope 1+2, >= 67% Scope 3)
    - C2: Timeframe (5-10 year near-term, no later than 2050 long-term)
    - C3: Ambition level (>= 4.2%/yr 1.5C or >= 2.5%/yr WB2C for Scope 1+2)
    - C4: Scope 3 minimum ambition (>= 2.5%/yr linear for Scope 3)
    - C5: Base year recalculation policy (significant threshold trigger)
    - C6-C21: Additional criteria covering methods, reporting, governance

Carbon Budget Allocation Methods:
    - LINEAR: Equal annual budget allocation
    - FRONT_LOADED: Larger budgets early, tightening over time
    - BACK_LOADED: Tighter budgets early, allowing catch-up later

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest
    2. Preset YAML (7 presets)
    3. Environment overrides (INTERIM_TARGETS_* environment variables)
    4. Explicit runtime overrides

Regulatory Context:
    - SBTi Corporate Net-Zero Standard v1.0 (2021, updated 2024)
    - SBTi Corporate Manual (near-term target setting criteria)
    - SBTi Monitoring, Reporting and Verification Guidance
    - GHG Protocol Corporate Standard (revised)
    - ISO 14064-1:2018 (Quantification and reporting)
    - CDP Climate Change Questionnaire (C4 Targets and Performance)
    - TCFD Recommendations (Metrics and Targets pillar)
    - Paris Agreement Article 4 (progressive ambition)
    - EU Climate Law (55% reduction by 2030, net-zero by 2050)

Cross-Pack Integration:
    - PACK-021: Long-term net-zero target and baseline year
    - PACK-022: Acceleration levers and reduction initiatives
    - PACK-027: Enterprise-level net-zero orchestration
    - PACK-028: Sector pathway alignment and MACC curves

Example:
    >>> config = PackConfig.from_preset("sbti_1_5c_pathway")
    >>> print(config.pack.sbti_pathway)
    1.5C
    >>> print(config.pack.interim_target_5yr_min_reduction)
    42.0
    >>> config = PackConfig.from_preset("quarterly_monitoring", overrides={"variance_tolerance_pct": 5.0})
"""

import hashlib
import logging
import os
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# Base directory for all pack configuration files
PACK_BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = Path(__file__).resolve().parent


# =============================================================================
# Constants
# =============================================================================

DEFAULT_BASELINE_YEAR: int = 2019
DEFAULT_CURRENT_YEAR: int = 2024
DEFAULT_REPORTING_YEAR: int = 2024
DEFAULT_LONG_TERM_TARGET_YEAR: int = 2050
DEFAULT_LONG_TERM_REDUCTION_PCT: float = 90.0
DEFAULT_NET_ZERO_YEAR: int = 2050
DEFAULT_5YR_TARGET_YEAR: int = 2030
DEFAULT_10YR_TARGET_YEAR: int = 2035
DEFAULT_VARIANCE_TOLERANCE_PCT: float = 10.0
DEFAULT_FORECAST_HORIZON_YEARS: int = 3
DEFAULT_CONFIDENCE_INTERVAL: float = 0.95
DEFAULT_SCOPE_3_LAG_YEARS: int = 5
DEFAULT_SBTI_CRITERIA_COUNT: int = 21
DEFAULT_RETENTION_YEARS: int = 7

# SBTi minimum near-term reduction requirements (% from base year by 2030)
SBTI_NEAR_TERM_MINIMUMS: Dict[str, Dict[str, float]] = {
    "1.5C": {
        "scope_1_2_reduction_pct": 42.0,
        "scope_1_2_annual_linear_pct": 4.2,
        "scope_3_reduction_pct": 25.0,
        "scope_3_annual_linear_pct": 2.5,
        "long_term_reduction_pct": 90.0,
    },
    "WB2C": {
        "scope_1_2_reduction_pct": 30.0,
        "scope_1_2_annual_linear_pct": 2.5,
        "scope_3_reduction_pct": 20.0,
        "scope_3_annual_linear_pct": 1.8,
        "long_term_reduction_pct": 90.0,
    },
}

# SBTi coverage thresholds
SBTI_COVERAGE_THRESHOLDS: Dict[str, float] = {
    "near_term_scope_1_pct": 95.0,
    "near_term_scope_2_pct": 95.0,
    "near_term_scope_3_pct": 67.0,
    "long_term_scope_1_pct": 95.0,
    "long_term_scope_2_pct": 95.0,
    "long_term_scope_3_pct": 90.0,
}

# SBTi 21 validation criteria categories
SBTI_VALIDATION_CRITERIA: Dict[str, str] = {
    "C01": "Boundary Completeness - Scope 1 coverage >= 95%",
    "C02": "Boundary Completeness - Scope 2 coverage >= 95%",
    "C03": "Boundary Completeness - Scope 3 screening completed",
    "C04": "Scope 3 Significance - >= 40% of total to require Scope 3 target",
    "C05": "Near-Term Timeframe - 5 to 10 years from submission date",
    "C06": "Near-Term Ambition - >= 4.2%/yr (1.5C) or >= 2.5%/yr (WB2C) Scope 1+2",
    "C07": "Near-Term Scope 3 - >= 2.5%/yr linear reduction if Scope 3 significant",
    "C08": "Long-Term Timeframe - No later than 2050",
    "C09": "Long-Term Ambition - >= 90% reduction from base year",
    "C10": "Long-Term Scope 3 Coverage - >= 90% of total Scope 3",
    "C11": "Base Year Selection - Within 2 years of most recent inventory",
    "C12": "Base Year Recalculation - Policy for significant changes (>=5%)",
    "C13": "Methods - GHG Protocol or ISO 14064 compliant methodology",
    "C14": "Bioenergy Accounting - Separate reporting of biogenic CO2",
    "C15": "Offsets Exclusion - Offsets not counted toward target achievement",
    "C16": "Renewable Energy - Scope 2 market-based accounting for RE claims",
    "C17": "Progress Reporting - Annual disclosure via CDP or equivalent",
    "C18": "Target Recalculation - Recalculate if structural change > threshold",
    "C19": "Sector Guidance - Comply with applicable SBTi sector guidance",
    "C20": "Governance - Board or C-suite oversight of target",
    "C21": "Public Commitment - Public disclosure of SBTi commitment",
}

# Performance scoring thresholds (traffic light system)
PERFORMANCE_SCORES: Dict[str, Dict[str, Any]] = {
    "GREEN": {
        "label": "On Track",
        "description": "Performance is on track or ahead of target",
        "variance_max_pct": None,  # Within tolerance
        "action_required": False,
        "color_hex": "#22C55E",
    },
    "AMBER": {
        "label": "At Risk",
        "description": "Performance is behind target but recoverable",
        "variance_max_pct": None,  # Within 2x tolerance
        "action_required": True,
        "color_hex": "#F59E0B",
    },
    "RED": {
        "label": "Off Track",
        "description": "Significantly behind target, corrective action urgent",
        "variance_max_pct": None,  # Beyond 2x tolerance
        "action_required": True,
        "color_hex": "#EF4444",
    },
}

# Variance analysis decomposition categories
VARIANCE_DECOMPOSITION_LEVELS: Dict[str, str] = {
    "scope": "Decompose variance by emission scope (1, 2, 3)",
    "category": "Decompose by GHG Protocol category within each scope",
    "initiative": "Decompose by reduction initiative contribution",
    "facility": "Decompose by site/facility within organization",
    "business_unit": "Decompose by business unit or division",
    "geography": "Decompose by country or region",
}

# Root cause classification taxonomy
ROOT_CAUSE_CATEGORIES: Dict[str, List[str]] = {
    "activity_change": [
        "production_increase", "production_decrease",
        "acquisition", "divestiture",
        "new_facility", "facility_closure",
        "market_expansion", "market_contraction",
    ],
    "intensity_change": [
        "efficiency_improvement", "efficiency_degradation",
        "fuel_switching", "fuel_regression",
        "technology_upgrade", "technology_failure",
        "grid_decarbonization", "grid_carbonization",
    ],
    "structural_change": [
        "product_mix_shift", "geographic_shift",
        "outsourcing", "insourcing",
        "supply_chain_change", "customer_base_change",
    ],
    "external_factors": [
        "weather_anomaly", "regulatory_change",
        "energy_price_shock", "supply_disruption",
        "pandemic_impact", "conflict_impact",
    ],
    "measurement_change": [
        "methodology_update", "emission_factor_update",
        "scope_boundary_change", "data_quality_improvement",
        "base_year_recalculation", "reporting_correction",
    ],
}

# Carbon budget allocation profiles
BUDGET_ALLOCATION_PROFILES: Dict[str, Dict[str, Any]] = {
    "linear": {
        "name": "Linear Allocation",
        "description": "Equal annual reduction from baseline to target year",
        "front_loading_factor": 1.0,
        "risk_profile": "balanced",
    },
    "front_loaded": {
        "name": "Front-Loaded Allocation",
        "description": "Larger reductions in early years, easing over time",
        "front_loading_factor": 1.5,
        "risk_profile": "conservative",
    },
    "back_loaded": {
        "name": "Back-Loaded Allocation",
        "description": "Smaller reductions early, accelerating over time",
        "front_loading_factor": 0.6,
        "risk_profile": "aggressive",
    },
}

# Reporting framework mapping for interim targets
REPORTING_FRAMEWORK_MAPPING: Dict[str, Dict[str, str]] = {
    "sbti": {
        "name": "Science Based Targets initiative",
        "version": "Corporate Net-Zero Standard v1.0 (2024 update)",
        "relevant_sections": "Near-term targets, Long-term targets, Progress reporting",
        "disclosure_frequency": "Annual",
    },
    "cdp": {
        "name": "CDP Climate Change",
        "version": "2024 Questionnaire",
        "relevant_sections": "C4.1-C4.3 Targets, C4.4-C4.5 Progress",
        "disclosure_frequency": "Annual",
    },
    "tcfd": {
        "name": "Task Force on Climate-related Financial Disclosures",
        "version": "2023 Recommendations",
        "relevant_sections": "Metrics and Targets (c), Strategy (b)",
        "disclosure_frequency": "Annual (integrated report)",
    },
    "gri": {
        "name": "Global Reporting Initiative",
        "version": "GRI 305 (2016, updated 2022)",
        "relevant_sections": "305-5 Reduction of GHG emissions",
        "disclosure_frequency": "Annual",
    },
    "esrs": {
        "name": "European Sustainability Reporting Standards",
        "version": "ESRS E1 Climate Change (2024)",
        "relevant_sections": "E1-4 Targets, E1-5 Energy, E1-6 GHG emissions",
        "disclosure_frequency": "Annual (CSRD)",
    },
    "iso14064": {
        "name": "ISO 14064-1",
        "version": "2018",
        "relevant_sections": "Clause 9: Quantification of GHG reductions",
        "disclosure_frequency": "Annual",
    },
}

# Trend extrapolation model parameters
EXTRAPOLATION_MODEL_PARAMS: Dict[str, Dict[str, Any]] = {
    "linear": {
        "name": "Linear Regression",
        "description": "Simple OLS linear trend extrapolation",
        "min_data_points": 3,
        "max_forecast_years": 10,
        "uncertainty_method": "prediction_interval",
    },
    "exponential_smoothing": {
        "name": "Holt-Winters Exponential Smoothing",
        "description": "Double exponential smoothing with trend component",
        "min_data_points": 4,
        "max_forecast_years": 5,
        "uncertainty_method": "simulation",
        "alpha_range": [0.1, 0.9],
        "beta_range": [0.01, 0.5],
    },
    "arima": {
        "name": "ARIMA Time Series",
        "description": "Auto-Regressive Integrated Moving Average with auto-order selection",
        "min_data_points": 8,
        "max_forecast_years": 10,
        "uncertainty_method": "confidence_interval",
        "max_p": 5,
        "max_d": 2,
        "max_q": 5,
    },
}

# Corrective action priority matrix
CORRECTIVE_ACTION_PRIORITIES: Dict[str, Dict[str, Any]] = {
    "critical": {
        "gap_pct_min": 20.0,
        "response_days": 30,
        "escalation_level": "board",
        "review_frequency": "monthly",
    },
    "high": {
        "gap_pct_min": 10.0,
        "response_days": 60,
        "escalation_level": "c_suite",
        "review_frequency": "monthly",
    },
    "medium": {
        "gap_pct_min": 5.0,
        "response_days": 90,
        "escalation_level": "sustainability_director",
        "review_frequency": "quarterly",
    },
    "low": {
        "gap_pct_min": 0.0,
        "response_days": 180,
        "escalation_level": "sustainability_manager",
        "review_frequency": "quarterly",
    },
}

# Supported preset configurations
SUPPORTED_PRESETS: Dict[str, str] = {
    "sbti_1_5c_pathway": "SBTi 1.5C-aligned interim targets with 42% near-term reduction",
    "sbti_wb2c_pathway": "SBTi Well-Below 2C interim targets with 30% near-term reduction",
    "quarterly_monitoring": "Frequent monitoring with real-time alerting and escalation",
    "annual_review": "Comprehensive annual review with dual variance analysis and assurance",
    "corrective_action": "Proactive corrective action focus with MACC integration and forecasting",
    "sector_specific": "Sector pathway integration with PACK-028 milestones and technology alignment",
    "scope_3_extended": "Extended Scope 3 timeline per SBTi guidance with category-level tracking",
}


# =============================================================================
# Helper Functions
# =============================================================================


def _utcnow() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: str) -> str:
    """Compute SHA-256 hash of a string.

    Args:
        data: Input string to hash.

    Returns:
        Hex-encoded SHA-256 hash.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# =============================================================================
# Enums (10 enums)
# =============================================================================


class SBTiPathwayLevel(str, Enum):
    """SBTi temperature alignment pathway level.

    Determines minimum annual reduction rates and near-term ambition
    thresholds per SBTi Corporate Net-Zero Standard.
    """

    CELSIUS_1_5 = "1.5C"
    WELL_BELOW_2 = "WB2C"


class PathwayType(str, Enum):
    """Interim target pathway mathematical model.

    Determines the shape of the emission reduction curve from
    base year to long-term target year, with interim milestones
    placed along the modeled curve.
    """

    LINEAR = "linear"
    MILESTONE_BASED = "milestone_based"
    ACCELERATING = "accelerating"
    S_CURVE = "s_curve"


class MonitoringFrequency(str, Enum):
    """Progress monitoring and reporting frequency."""

    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


class VarianceMethod(str, Enum):
    """Variance analysis decomposition method.

    LMDI is preferred for additive decomposition without residuals.
    Kaya Identity is widely used for national-level analysis.
    BOTH runs both methods for comprehensive attribution.
    """

    LMDI = "lmdi"
    KAYA = "kaya"
    BOTH = "both"


class ExtrapolationMethod(str, Enum):
    """Trend extrapolation forecasting method.

    LINEAR is simplest with fewest data requirements.
    EXPONENTIAL_SMOOTHING captures trends with adaptive weighting.
    ARIMA provides sophisticated time series modeling.
    """

    LINEAR = "linear"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    ARIMA = "arima"


class CorrectiveActionTrigger(str, Enum):
    """Performance level that triggers corrective action process.

    AMBER triggers proactive intervention when variance exceeds tolerance.
    RED triggers only when variance exceeds 2x tolerance (reactive).
    """

    AMBER = "amber"
    RED = "red"


class BudgetAllocationMethod(str, Enum):
    """Carbon budget year-by-year allocation strategy.

    LINEAR: Equal annual reduction (simplest, SBTi default).
    FRONT_LOADED: Aggressive early reductions, easing later.
    BACK_LOADED: Modest early reductions, accelerating later.
    """

    LINEAR = "linear"
    FRONT_LOADED = "front_loaded"
    BACK_LOADED = "back_loaded"


class AssuranceLevel(str, Enum):
    """External assurance engagement level for reported interim progress."""

    NONE = "none"
    LIMITED = "limited"
    REASONABLE = "reasonable"


class ReportingFrequency(str, Enum):
    """External reporting and disclosure frequency."""

    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUAL = "semi_annual"
    ANNUAL = "annual"


class PerformanceScore(str, Enum):
    """Traffic light performance scoring classification."""

    GREEN = "green"
    AMBER = "amber"
    RED = "red"


# =============================================================================
# Pydantic Sub-Config Models (9 models)
# =============================================================================


class InterimTargetConfig(BaseModel):
    """Configuration for a single interim target milestone.

    Defines the target year, minimum reduction percentage, scope coverage,
    and validation requirements for one interim checkpoint along the
    pathway from base year to net-zero.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    enabled: bool = Field(
        True,
        description="Whether this interim target is active",
    )
    target_year: int = Field(
        DEFAULT_5YR_TARGET_YEAR,
        ge=2025,
        le=2055,
        description="Year by which the interim reduction target must be achieved",
    )
    min_reduction_pct: float = Field(
        42.0,
        ge=0.0,
        le=100.0,
        description="Minimum percentage reduction from base year emissions",
    )
    scope_1_2_reduction_pct: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Scope 1+2 specific reduction target (overrides min_reduction_pct if set)",
    )
    scope_3_reduction_pct: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Scope 3 specific reduction target (may lag behind Scope 1+2)",
    )
    label: str = Field(
        "5-year target",
        description="Human-readable label for this interim target milestone",
    )

    @field_validator("target_year")
    @classmethod
    def validate_target_year_range(cls, v: int) -> int:
        """Validate target year is in a reasonable range."""
        if v < 2025:
            raise ValueError(
                f"Interim target year ({v}) cannot be in the past. "
                f"Minimum is 2025."
            )
        return v


class ScopeConfig(BaseModel):
    """Configuration for emission scope coverage in interim targets.

    Defines which scopes are included, Scope 3 lag allowance per SBTi
    guidance, and category-level tracking granularity.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    scope_1_included: bool = Field(
        True,
        description="Include Scope 1 (direct) emissions in interim target tracking",
    )
    scope_2_included: bool = Field(
        True,
        description="Include Scope 2 (indirect energy) emissions in interim target tracking",
    )
    scope_2_method: str = Field(
        "market_based",
        description="Scope 2 accounting method: 'market_based' or 'location_based'",
    )
    scope_3_included: bool = Field(
        True,
        description="Include Scope 3 (value chain) emissions in interim target tracking",
    )
    scope_3_lag_years: int = Field(
        DEFAULT_SCOPE_3_LAG_YEARS,
        ge=0,
        le=10,
        description="Years Scope 3 interim targets can lag behind Scope 1+2 per SBTi guidance",
    )
    scope_3_categories: List[int] = Field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 11, 12],
        description="Scope 3 categories included in target (1-15 per GHG Protocol)",
    )
    scope_3_significant_threshold_pct: float = Field(
        40.0,
        ge=0.0,
        le=100.0,
        description="Threshold (% of total) above which Scope 3 target is mandatory per SBTi",
    )
    scope_3_coverage_pct: float = Field(
        67.0,
        ge=0.0,
        le=100.0,
        description="Minimum percentage of Scope 3 emissions covered by near-term target",
    )
    scope_1_coverage_pct: float = Field(
        95.0,
        ge=0.0,
        le=100.0,
        description="Percentage of Scope 1 emissions covered by target",
    )
    scope_2_coverage_pct: float = Field(
        95.0,
        ge=0.0,
        le=100.0,
        description="Percentage of Scope 2 emissions covered by target",
    )

    @field_validator("scope_2_method")
    @classmethod
    def validate_scope_2_method(cls, v: str) -> str:
        """Validate Scope 2 accounting method."""
        valid_methods = {"market_based", "location_based", "dual_reporting"}
        if v not in valid_methods:
            raise ValueError(
                f"Invalid scope_2_method: {v}. Must be one of: {sorted(valid_methods)}"
            )
        return v

    @field_validator("scope_3_categories")
    @classmethod
    def validate_scope_3_categories(cls, v: List[int]) -> List[int]:
        """Validate Scope 3 categories are in 1-15 range."""
        invalid = [c for c in v if c < 1 or c > 15]
        if invalid:
            raise ValueError(
                f"Invalid Scope 3 categories: {invalid}. Must be 1-15."
            )
        return sorted(set(v))


class MonitoringConfig(BaseModel):
    """Configuration for progress monitoring and performance scoring.

    Defines monitoring frequency, variance tolerance thresholds,
    traffic light scoring parameters, and trend detection settings.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    monitoring_frequency: MonitoringFrequency = Field(
        MonitoringFrequency.QUARTERLY,
        description="How frequently progress against interim targets is assessed",
    )
    variance_tolerance_pct: float = Field(
        DEFAULT_VARIANCE_TOLERANCE_PCT,
        ge=1.0,
        le=50.0,
        description="Percentage tolerance (+-) before triggering amber alert",
    )
    performance_scoring_enabled: bool = Field(
        True,
        description="Enable RED/AMBER/GREEN traffic light performance scoring",
    )
    amber_threshold_multiplier: float = Field(
        1.0,
        ge=0.5,
        le=3.0,
        description="Multiplier of variance_tolerance_pct for AMBER threshold",
    )
    red_threshold_multiplier: float = Field(
        2.0,
        ge=1.0,
        le=5.0,
        description="Multiplier of variance_tolerance_pct for RED threshold",
    )
    trend_detection_enabled: bool = Field(
        True,
        description="Enable automatic trend detection in progress time series",
    )
    trend_window_periods: int = Field(
        4,
        ge=2,
        le=12,
        description="Number of monitoring periods for rolling trend calculation",
    )
    cumulative_tracking_enabled: bool = Field(
        True,
        description="Track cumulative performance against carbon budget in addition to point-in-time",
    )
    year_on_year_comparison: bool = Field(
        True,
        description="Include year-over-year change analysis in monitoring reports",
    )
    normalized_tracking: bool = Field(
        True,
        description="Track intensity-based (normalized) targets alongside absolute targets",
    )

    @model_validator(mode="after")
    def validate_threshold_ordering(self) -> "MonitoringConfig":
        """Ensure RED threshold is >= AMBER threshold."""
        if self.red_threshold_multiplier < self.amber_threshold_multiplier:
            raise ValueError(
                f"red_threshold_multiplier ({self.red_threshold_multiplier}) must be >= "
                f"amber_threshold_multiplier ({self.amber_threshold_multiplier})"
            )
        return self


class VarianceAnalysisConfig(BaseModel):
    """Configuration for variance analysis and root cause attribution.

    Defines the decomposition method (LMDI, Kaya, or both), analysis
    granularity levels, and root cause classification settings.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    variance_method: VarianceMethod = Field(
        VarianceMethod.LMDI,
        description="Primary variance decomposition method",
    )
    decomposition_levels: List[str] = Field(
        default_factory=lambda: ["scope", "category", "initiative"],
        description="Levels of decomposition for variance attribution",
    )
    root_cause_classification: bool = Field(
        True,
        description="Enable structured root cause classification for each variance driver",
    )
    root_cause_taxonomy: str = Field(
        "standard",
        description="Root cause taxonomy: 'standard' (5 categories) or 'extended' (with sub-causes)",
    )
    materiality_threshold_pct: float = Field(
        2.0,
        ge=0.1,
        le=20.0,
        description="Minimum variance contribution (%) to include in decomposition output",
    )
    historical_comparison_years: int = Field(
        3,
        ge=1,
        le=10,
        description="Number of historical years to include in variance trend analysis",
    )
    waterfall_visualization: bool = Field(
        True,
        description="Generate waterfall chart data for variance decomposition visualization",
    )
    attribution_confidence_enabled: bool = Field(
        True,
        description="Include confidence scores for each root cause attribution",
    )

    @field_validator("decomposition_levels")
    @classmethod
    def validate_decomposition_levels(cls, v: List[str]) -> List[str]:
        """Validate decomposition levels are recognized."""
        valid_levels = set(VARIANCE_DECOMPOSITION_LEVELS.keys())
        invalid = [level for level in v if level not in valid_levels]
        if invalid:
            logger.warning(
                "Unrecognized decomposition levels: %s. "
                "Valid levels: %s", invalid, sorted(valid_levels)
            )
        return v


class ExtrapolationConfig(BaseModel):
    """Configuration for trend extrapolation and forecasting.

    Defines the forecasting method, projection horizon, confidence
    intervals, and model validation parameters.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    extrapolation_method: ExtrapolationMethod = Field(
        ExtrapolationMethod.EXPONENTIAL_SMOOTHING,
        description="Primary trend extrapolation forecasting method",
    )
    forecast_horizon_years: int = Field(
        DEFAULT_FORECAST_HORIZON_YEARS,
        ge=1,
        le=10,
        description="Number of years to forecast forward",
    )
    confidence_interval: float = Field(
        DEFAULT_CONFIDENCE_INTERVAL,
        ge=0.80,
        le=0.99,
        description="Confidence interval for forecast uncertainty bands (e.g., 0.95 = 95%)",
    )
    backtesting_enabled: bool = Field(
        True,
        description="Enable backtesting to validate forecast model accuracy",
    )
    backtesting_holdout_years: int = Field(
        2,
        ge=1,
        le=5,
        description="Number of years to hold out for backtesting validation",
    )
    ensemble_enabled: bool = Field(
        False,
        description="Use ensemble of all three methods and weight by backtesting accuracy",
    )
    scenario_adjustment_enabled: bool = Field(
        True,
        description="Allow scenario-based adjustments to base forecast (e.g., planned initiatives)",
    )
    min_data_points: int = Field(
        3,
        ge=2,
        le=20,
        description="Minimum number of historical data points required for forecasting",
    )
    outlier_detection: bool = Field(
        True,
        description="Detect and flag outlier data points before forecasting",
    )
    outlier_z_score_threshold: float = Field(
        2.5,
        ge=1.5,
        le=4.0,
        description="Z-score threshold for outlier detection in historical data",
    )

    @model_validator(mode="after")
    def validate_min_data_for_method(self) -> "ExtrapolationConfig":
        """Ensure min_data_points is sufficient for the chosen method."""
        method_params = EXTRAPOLATION_MODEL_PARAMS.get(self.extrapolation_method.value, {})
        method_min = method_params.get("min_data_points", 3)
        if self.min_data_points < method_min:
            logger.warning(
                "min_data_points (%d) is below recommended minimum (%d) for %s method. "
                "Adjusting to %d.",
                self.min_data_points, method_min,
                self.extrapolation_method.value, method_min,
            )
            object.__setattr__(self, "min_data_points", method_min)
        return self


class CorrectiveActionConfig(BaseModel):
    """Configuration for corrective action planning and execution.

    Defines when corrective actions are triggered, gap quantification
    settings, initiative optimization parameters, and MACC curve
    integration with PACK-028.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    corrective_action_trigger: CorrectiveActionTrigger = Field(
        CorrectiveActionTrigger.AMBER,
        description="Performance score level that triggers corrective action process",
    )
    gap_quantification_enabled: bool = Field(
        True,
        description="Quantify the tCO2e gap to target and cumulative budget impact",
    )
    initiative_optimization: bool = Field(
        True,
        description="Re-sequence and accelerate existing reduction initiatives to close gap",
    )
    macc_integration: bool = Field(
        True,
        description="Integrate MACC curve from PACK-028 to identify cost-effective gap-closing levers",
    )
    macc_carbon_price_usd: float = Field(
        80.0,
        ge=0.0,
        le=500.0,
        description="Shadow carbon price (USD/tCO2e) for MACC lever evaluation",
    )
    scenario_remodeling: bool = Field(
        True,
        description="Recalculate pathway scenarios with updated assumptions when gap detected",
    )
    action_plan_template: bool = Field(
        True,
        description="Generate structured corrective action plan template with responsibilities",
    )
    escalation_enabled: bool = Field(
        True,
        description="Enable escalation workflow for persistent off-track performance",
    )
    escalation_threshold_periods: int = Field(
        2,
        ge=1,
        le=6,
        description="Number of consecutive off-track periods before escalation triggers",
    )
    max_corrective_actions: int = Field(
        20,
        ge=5,
        le=100,
        description="Maximum number of corrective actions to propose per review cycle",
    )
    cost_benefit_analysis: bool = Field(
        True,
        description="Include cost-benefit analysis for each proposed corrective action",
    )
    implementation_timeline: bool = Field(
        True,
        description="Generate implementation timeline with milestones for corrective actions",
    )


class CarbonBudgetConfig(BaseModel):
    """Configuration for carbon budget allocation and tracking.

    Defines the annual emission budget calculation method, overshoot
    policies, and cumulative budget tracking parameters.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    carbon_budget_enabled: bool = Field(
        True,
        description="Enable annual carbon budget allocation and tracking",
    )
    budget_allocation_method: BudgetAllocationMethod = Field(
        BudgetAllocationMethod.LINEAR,
        description="Method for allocating total carbon budget to individual years",
    )
    budget_overshoot_allowed: bool = Field(
        False,
        description="Whether annual budget overshoot is permitted (must be compensated later)",
    )
    overshoot_compensation_years: int = Field(
        2,
        ge=1,
        le=5,
        description="Maximum years allowed to compensate for a budget overshoot",
    )
    cumulative_budget_tracking: bool = Field(
        True,
        description="Track cumulative emissions against cumulative budget (area under curve)",
    )
    budget_reserve_pct: float = Field(
        5.0,
        ge=0.0,
        le=20.0,
        description="Reserve percentage of annual budget held back for unexpected emissions",
    )
    quarterly_budget_splits: bool = Field(
        True,
        description="Sub-allocate annual budgets to quarterly periods for granular tracking",
    )
    scope_level_budgets: bool = Field(
        True,
        description="Allocate separate budgets per emission scope (1, 2, 3)",
    )
    facility_level_budgets: bool = Field(
        False,
        description="Allocate budgets at individual facility level",
    )
    budget_unit: str = Field(
        "tCO2e",
        description="Unit for carbon budget reporting (tCO2e, ktCO2e, MtCO2e)",
    )

    @field_validator("budget_unit")
    @classmethod
    def validate_budget_unit(cls, v: str) -> str:
        """Validate carbon budget unit."""
        valid_units = {"tCO2e", "ktCO2e", "MtCO2e"}
        if v not in valid_units:
            raise ValueError(
                f"Invalid budget_unit: {v}. Must be one of: {sorted(valid_units)}"
            )
        return v


class SBTiValidationConfig(BaseModel):
    """Configuration for SBTi target validation checks.

    Runs the 21 SBTi validation criteria against the configured
    interim targets to verify compliance before formal submission.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    sbti_validation_enabled: bool = Field(
        True,
        description="Enable SBTi validation checks against 21 criteria",
    )
    sbti_criteria_count: int = Field(
        DEFAULT_SBTI_CRITERIA_COUNT,
        ge=1,
        le=50,
        description="Number of SBTi validation criteria to check",
    )
    sbti_minimum_ambition_check: bool = Field(
        True,
        description="Check minimum annual reduction rate against SBTi requirements",
    )
    sbti_boundary_check: bool = Field(
        True,
        description="Check scope coverage thresholds against SBTi requirements",
    )
    sbti_timeframe_check: bool = Field(
        True,
        description="Check target timeframes against SBTi 5-10 year near-term and 2050 long-term",
    )
    sbti_recalculation_trigger_pct: float = Field(
        5.0,
        ge=1.0,
        le=20.0,
        description="Percentage change in base year emissions triggering recalculation per SBTi C12",
    )
    sbti_submission_date: Optional[str] = Field(
        None,
        description="Planned SBTi submission date (ISO format, e.g., '2025-06-15')",
    )
    sbti_commitment_type: str = Field(
        "near_term_and_net_zero",
        description="SBTi commitment type: 'near_term_only', 'near_term_and_net_zero', 'net_zero'",
    )
    auto_fix_suggestions: bool = Field(
        True,
        description="Generate automatic fix suggestions when validation criteria fail",
    )

    @field_validator("sbti_commitment_type")
    @classmethod
    def validate_commitment_type(cls, v: str) -> str:
        """Validate SBTi commitment type."""
        valid_types = {"near_term_only", "near_term_and_net_zero", "net_zero"}
        if v not in valid_types:
            raise ValueError(
                f"Invalid sbti_commitment_type: {v}. Must be one of: {sorted(valid_types)}"
            )
        return v


class ReportingConfig(BaseModel):
    """Configuration for multi-framework reporting and disclosure.

    Defines reporting frameworks, frequency, assurance level,
    and provenance tracking for interim target progress reports.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    reporting_frameworks: List[str] = Field(
        default_factory=lambda: ["sbti", "cdp", "tcfd"],
        description="Reporting frameworks for interim target progress disclosure",
    )
    reporting_frequency: ReportingFrequency = Field(
        ReportingFrequency.ANNUAL,
        description="External reporting and disclosure frequency",
    )
    public_disclosure_enabled: bool = Field(
        True,
        description="Enable public disclosure of interim target progress",
    )
    assurance_level: AssuranceLevel = Field(
        AssuranceLevel.LIMITED,
        description="External assurance engagement level for reported progress",
    )
    sha256_provenance: bool = Field(
        True,
        description="Generate SHA-256 provenance hashes for all report outputs",
    )
    calculation_trace: bool = Field(
        True,
        description="Generate step-by-step calculation trace for auditability",
    )
    assumption_register: bool = Field(
        True,
        description="Maintain assumption register for all target calculations",
    )
    data_lineage_enabled: bool = Field(
        True,
        description="Track full data lineage from emission source to target progress",
    )
    retention_years: int = Field(
        DEFAULT_RETENTION_YEARS,
        ge=3,
        le=15,
        description="Report and audit trail retention period in years",
    )
    executive_summary_enabled: bool = Field(
        True,
        description="Generate executive summary for board and C-suite reporting",
    )
    benchmark_comparison: bool = Field(
        True,
        description="Include peer benchmark comparison in progress reports",
    )

    @field_validator("reporting_frameworks")
    @classmethod
    def validate_frameworks(cls, v: List[str]) -> List[str]:
        """Validate reporting framework identifiers."""
        valid = set(REPORTING_FRAMEWORK_MAPPING.keys())
        invalid = [f for f in v if f.lower() not in valid]
        if invalid:
            logger.warning(
                "Unrecognized reporting frameworks: %s. "
                "Recognized: %s", invalid, sorted(valid)
            )
        return [f.lower() for f in v]


class AlertingConfig(BaseModel):
    """Configuration for alerting and notification channels.

    Defines email, Slack, and webhook alerting for interim target
    performance monitoring with escalation workflows.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    email_alerts_enabled: bool = Field(
        True,
        description="Enable email notifications for performance alerts",
    )
    slack_alerts_enabled: bool = Field(
        False,
        description="Enable Slack channel notifications for performance alerts",
    )
    webhook_alerts_enabled: bool = Field(
        False,
        description="Enable webhook notifications for integration with external systems",
    )
    alert_recipients: List[str] = Field(
        default_factory=list,
        description="Email addresses for alert recipients",
    )
    slack_channel: Optional[str] = Field(
        None,
        description="Slack channel ID or name for notifications (e.g., '#sustainability-alerts')",
    )
    webhook_url: Optional[str] = Field(
        None,
        description="Webhook URL for external system notifications",
    )
    alert_escalation_enabled: bool = Field(
        True,
        description="Enable escalation for persistent off-track performance",
    )
    escalation_recipients: List[str] = Field(
        default_factory=list,
        description="Additional recipients for escalated alerts (C-suite, board)",
    )
    alert_on_green: bool = Field(
        False,
        description="Send notifications even when performance is on track (GREEN)",
    )
    alert_on_amber: bool = Field(
        True,
        description="Send notifications when performance is at risk (AMBER)",
    )
    alert_on_red: bool = Field(
        True,
        description="Send notifications when performance is off track (RED)",
    )
    digest_frequency: str = Field(
        "weekly",
        description="Frequency of alert digest summaries: 'daily', 'weekly', 'monthly'",
    )

    @field_validator("digest_frequency")
    @classmethod
    def validate_digest_frequency(cls, v: str) -> str:
        """Validate digest frequency."""
        valid = {"daily", "weekly", "monthly"}
        if v not in valid:
            raise ValueError(
                f"Invalid digest_frequency: {v}. Must be one of: {sorted(valid)}"
            )
        return v


class PerformanceConfig(BaseModel):
    """Configuration for runtime performance tuning.

    Defines caching, concurrency, and timeout settings for the
    interim targets calculation pipeline.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    cache_enabled: bool = Field(
        True,
        description="Enable Redis-based caching for target calculations",
    )
    cache_ttl_seconds: int = Field(
        3600,
        ge=60,
        le=86400,
        description="Cache time-to-live in seconds",
    )
    max_concurrent_calcs: int = Field(
        4,
        ge=1,
        le=32,
        description="Maximum concurrent target calculation threads",
    )
    timeout_seconds: int = Field(
        300,
        ge=30,
        le=3600,
        description="Maximum timeout for a single engine calculation",
    )
    batch_size: int = Field(
        1000,
        ge=100,
        le=10000,
        description="Batch size for bulk data processing",
    )
    memory_limit_mb: int = Field(
        4096,
        ge=512,
        le=32768,
        description="Memory limit in MB for the calculation pipeline",
    )


# =============================================================================
# Main Configuration Model
# =============================================================================


class InterimTargetsConfig(BaseModel):
    """Main configuration model for PACK-029 Interim Targets Pack.

    This is the root Pydantic v2 configuration model containing all parameters
    for interim target setting, progress monitoring, variance analysis,
    corrective action planning, carbon budget allocation, trend extrapolation,
    SBTi validation, and multi-framework reporting.

    The model supports configurable 5-year and 10-year interim targets with
    custom milestones, four pathway types (linear, milestone_based, accelerating,
    s_curve), three variance analysis methods (LMDI, Kaya, both), three
    extrapolation methods (linear, exponential_smoothing, ARIMA), and
    comprehensive SBTi 21-criteria validation.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        json_schema_extra={
            "title": "PACK-029 Interim Targets Configuration",
            "description": "Configuration for interim target setting, monitoring, and corrective action",
        },
    )

    # --- Organization Identity ---
    organization_name: str = Field(
        "",
        description="Legal entity name of the organization",
    )
    organization_id: Optional[str] = Field(
        None,
        description="Unique organization identifier (UUID or internal code)",
    )

    # --- Temporal Settings ---
    baseline_year: int = Field(
        DEFAULT_BASELINE_YEAR,
        ge=2010,
        le=2030,
        description="Base year for emission baseline measurement (GHG Protocol compliant)",
    )
    current_year: int = Field(
        DEFAULT_CURRENT_YEAR,
        ge=2020,
        le=2035,
        description="Current calendar year for progress tracking",
    )
    reporting_year: int = Field(
        DEFAULT_REPORTING_YEAR,
        ge=2020,
        le=2035,
        description="Reporting year for latest available emission inventory",
    )

    # --- Long-Term Target (from PACK-021) ---
    long_term_target_year: int = Field(
        DEFAULT_LONG_TERM_TARGET_YEAR,
        ge=2040,
        le=2060,
        description="Long-term net-zero target year (no later than 2050 per SBTi)",
    )
    long_term_reduction_pct: float = Field(
        DEFAULT_LONG_TERM_REDUCTION_PCT,
        ge=50.0,
        le=100.0,
        description="Long-term reduction percentage from base year (>= 90% per SBTi)",
    )
    net_zero_year: int = Field(
        DEFAULT_NET_ZERO_YEAR,
        ge=2040,
        le=2060,
        description="Planned net-zero achievement year",
    )
    sbti_pathway: SBTiPathwayLevel = Field(
        SBTiPathwayLevel.CELSIUS_1_5,
        description="SBTi temperature alignment pathway level (1.5C or WB2C)",
    )

    # --- Interim Target Settings ---
    interim_target_5yr: InterimTargetConfig = Field(
        default_factory=lambda: InterimTargetConfig(
            enabled=True,
            target_year=DEFAULT_5YR_TARGET_YEAR,
            min_reduction_pct=42.0,
            label="5-year near-term target (SBTi)",
        ),
        description="5-year interim target configuration (SBTi near-term)",
    )
    interim_target_10yr: InterimTargetConfig = Field(
        default_factory=lambda: InterimTargetConfig(
            enabled=True,
            target_year=DEFAULT_10YR_TARGET_YEAR,
            min_reduction_pct=65.0,
            label="10-year mid-term target",
        ),
        description="10-year interim target configuration (interpolated milestone)",
    )
    custom_milestones: List[InterimTargetConfig] = Field(
        default_factory=list,
        description="Additional custom interim milestones beyond 5yr and 10yr",
    )

    # --- Pathway Type ---
    pathway_type: PathwayType = Field(
        PathwayType.LINEAR,
        description="Mathematical model for emission reduction pathway shape",
    )
    annual_reduction_rate: Optional[float] = Field(
        None,
        ge=0.0,
        le=20.0,
        description="Annual linear reduction rate (% per year) for LINEAR pathway",
    )
    s_curve_inflection_year: Optional[int] = Field(
        None,
        ge=2025,
        le=2050,
        description="Inflection year for S_CURVE pathway (steepest reduction point)",
    )
    milestone_years: Optional[List[int]] = Field(
        None,
        description="Milestone years for MILESTONE_BASED pathway (e.g., [2028, 2030, 2035, 2040, 2045])",
    )
    milestone_reductions_pct: Optional[List[float]] = Field(
        None,
        description="Reduction percentages at each milestone year (aligned with milestone_years)",
    )

    # --- Sub-Configurations ---
    scope: ScopeConfig = Field(
        default_factory=ScopeConfig,
        description="Emission scope coverage configuration",
    )
    monitoring: MonitoringConfig = Field(
        default_factory=MonitoringConfig,
        description="Progress monitoring and scoring configuration",
    )
    variance: VarianceAnalysisConfig = Field(
        default_factory=VarianceAnalysisConfig,
        description="Variance analysis and root cause attribution configuration",
    )
    extrapolation: ExtrapolationConfig = Field(
        default_factory=ExtrapolationConfig,
        description="Trend extrapolation and forecasting configuration",
    )
    corrective_action: CorrectiveActionConfig = Field(
        default_factory=CorrectiveActionConfig,
        description="Corrective action planning configuration",
    )
    carbon_budget: CarbonBudgetConfig = Field(
        default_factory=CarbonBudgetConfig,
        description="Carbon budget allocation and tracking configuration",
    )
    sbti_validation: SBTiValidationConfig = Field(
        default_factory=SBTiValidationConfig,
        description="SBTi target validation configuration",
    )
    reporting: ReportingConfig = Field(
        default_factory=ReportingConfig,
        description="Multi-framework reporting and disclosure configuration",
    )
    alerting: AlertingConfig = Field(
        default_factory=AlertingConfig,
        description="Alerting and notification configuration",
    )
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig,
        description="Runtime performance tuning configuration",
    )

    # --- Pack Metadata ---
    pack_version: str = Field(
        "1.0.0",
        description="Pack configuration version",
    )

    # --- Cross-Cutting Validators ---

    @model_validator(mode="after")
    def validate_baseline_before_current(self) -> "InterimTargetsConfig":
        """Ensure baseline year is not after current year."""
        if self.baseline_year > self.current_year:
            raise ValueError(
                f"baseline_year ({self.baseline_year}) must not be after "
                f"current_year ({self.current_year})"
            )
        return self

    @model_validator(mode="after")
    def validate_baseline_before_reporting(self) -> "InterimTargetsConfig":
        """Ensure baseline year is not after reporting year."""
        if self.baseline_year > self.reporting_year:
            raise ValueError(
                f"baseline_year ({self.baseline_year}) must not be after "
                f"reporting_year ({self.reporting_year})"
            )
        return self

    @model_validator(mode="after")
    def validate_interim_target_years_ordering(self) -> "InterimTargetsConfig":
        """Ensure interim target years are properly ordered."""
        if (self.interim_target_5yr.enabled and self.interim_target_10yr.enabled
                and self.interim_target_5yr.target_year >= self.interim_target_10yr.target_year):
            raise ValueError(
                f"5-year target year ({self.interim_target_5yr.target_year}) must be before "
                f"10-year target year ({self.interim_target_10yr.target_year})"
            )
        return self

    @model_validator(mode="after")
    def validate_interim_before_long_term(self) -> "InterimTargetsConfig":
        """Ensure interim targets are before long-term target."""
        if self.interim_target_5yr.enabled:
            if self.interim_target_5yr.target_year >= self.long_term_target_year:
                raise ValueError(
                    f"5-year target year ({self.interim_target_5yr.target_year}) must be before "
                    f"long_term_target_year ({self.long_term_target_year})"
                )
        if self.interim_target_10yr.enabled:
            if self.interim_target_10yr.target_year >= self.long_term_target_year:
                raise ValueError(
                    f"10-year target year ({self.interim_target_10yr.target_year}) must be before "
                    f"long_term_target_year ({self.long_term_target_year})"
                )
        return self

    @model_validator(mode="after")
    def validate_sbti_minimum_ambition(self) -> "InterimTargetsConfig":
        """Warn if interim target reduction is below SBTi minimum for pathway."""
        minimums = SBTI_NEAR_TERM_MINIMUMS.get(self.sbti_pathway.value, {})
        min_scope_1_2 = minimums.get("scope_1_2_reduction_pct", 42.0)

        if self.interim_target_5yr.enabled:
            if self.interim_target_5yr.min_reduction_pct < min_scope_1_2:
                logger.warning(
                    "5-year interim target reduction (%.1f%%) is below SBTi minimum "
                    "(%.1f%%) for %s pathway. Target may not be validated.",
                    self.interim_target_5yr.min_reduction_pct,
                    min_scope_1_2,
                    self.sbti_pathway.value,
                )
        return self

    @model_validator(mode="after")
    def validate_net_zero_year_sbti(self) -> "InterimTargetsConfig":
        """Warn if net-zero year exceeds SBTi 2050 deadline."""
        if self.net_zero_year > 2050:
            logger.warning(
                "Net-zero year (%d) exceeds SBTi Net-Zero Standard maximum of 2050. "
                "Target may not be eligible for SBTi validation.",
                self.net_zero_year,
            )
        return self

    @model_validator(mode="after")
    def validate_s_curve_inflection(self) -> "InterimTargetsConfig":
        """Ensure S-curve pathway has inflection year set."""
        if self.pathway_type == PathwayType.S_CURVE and self.s_curve_inflection_year is None:
            logger.warning(
                "S_CURVE pathway selected but no s_curve_inflection_year set. "
                "Defaulting to 2035."
            )
            object.__setattr__(self, "s_curve_inflection_year", 2035)
        return self

    @model_validator(mode="after")
    def validate_milestone_pathway_data(self) -> "InterimTargetsConfig":
        """Ensure milestone pathway has years and reductions defined."""
        if self.pathway_type == PathwayType.MILESTONE_BASED:
            if not self.milestone_years or not self.milestone_reductions_pct:
                logger.warning(
                    "MILESTONE_BASED pathway selected but milestone_years or "
                    "milestone_reductions_pct not set. Defaulting to 5-year intervals."
                )
                if not self.milestone_years:
                    object.__setattr__(
                        self, "milestone_years", [2028, 2030, 2035, 2040, 2045, 2050]
                    )
                if not self.milestone_reductions_pct:
                    object.__setattr__(
                        self, "milestone_reductions_pct",
                        [25.0, 42.0, 65.0, 80.0, 90.0, 95.0],
                    )
            elif len(self.milestone_years) != len(self.milestone_reductions_pct):
                raise ValueError(
                    f"milestone_years ({len(self.milestone_years)}) and "
                    f"milestone_reductions_pct ({len(self.milestone_reductions_pct)}) "
                    f"must have the same length"
                )
        return self

    @model_validator(mode="after")
    def validate_long_term_reduction(self) -> "InterimTargetsConfig":
        """Warn if long-term reduction is below SBTi 90% minimum."""
        if self.long_term_reduction_pct < 90.0:
            logger.warning(
                "Long-term reduction (%.1f%%) is below SBTi minimum of 90%%. "
                "Net-zero target may not be validated.",
                self.long_term_reduction_pct,
            )
        return self

    def get_enabled_engines(self) -> List[str]:
        """Return list of engine identifiers that should be enabled.

        Returns:
            Sorted list of enabled engine identifier strings.
        """
        engines = [
            "interim_target_calculator",
            "pathway_generator",
            "progress_tracker",
        ]

        if self.monitoring.performance_scoring_enabled:
            engines.append("performance_scorer")

        if self.variance.variance_method != VarianceMethod.LMDI or self.variance.root_cause_classification:
            engines.append("variance_analyzer")

        if self.extrapolation.extrapolation_method is not None:
            engines.append("trend_extrapolator")

        if (self.corrective_action.gap_quantification_enabled
                or self.corrective_action.initiative_optimization):
            engines.append("corrective_action_planner")

        if self.carbon_budget.carbon_budget_enabled:
            engines.append("carbon_budget_allocator")

        if self.sbti_validation.sbti_validation_enabled:
            engines.append("sbti_validator")

        if self.reporting.reporting_frameworks:
            engines.append("report_generator")

        return sorted(set(engines))

    def get_all_interim_targets(self) -> List[InterimTargetConfig]:
        """Return all enabled interim targets sorted by year.

        Returns:
            List of InterimTargetConfig instances, sorted by target_year.
        """
        targets = []
        if self.interim_target_5yr.enabled:
            targets.append(self.interim_target_5yr)
        if self.interim_target_10yr.enabled:
            targets.append(self.interim_target_10yr)
        for milestone in self.custom_milestones:
            if milestone.enabled:
                targets.append(milestone)
        return sorted(targets, key=lambda t: t.target_year)

    def get_sbti_minimums(self) -> Dict[str, float]:
        """Return SBTi minimum reduction requirements for the configured pathway.

        Returns:
            Dictionary with scope_1_2 and scope_3 annual and total minimums.
        """
        return SBTI_NEAR_TERM_MINIMUMS.get(
            self.sbti_pathway.value,
            SBTI_NEAR_TERM_MINIMUMS["1.5C"],
        )

    def get_target_year_range(self) -> List[int]:
        """Return the full range of years from baseline to long-term target.

        Returns:
            List of years from baseline_year to long_term_target_year inclusive.
        """
        return list(range(self.baseline_year, self.long_term_target_year + 1))

    def get_performance_score(self, variance_pct: float) -> PerformanceScore:
        """Classify a variance percentage into a performance score.

        Args:
            variance_pct: Actual variance from target as percentage (positive = behind).

        Returns:
            PerformanceScore enum value (GREEN, AMBER, or RED).
        """
        tolerance = self.monitoring.variance_tolerance_pct
        amber_threshold = tolerance * self.monitoring.amber_threshold_multiplier
        red_threshold = tolerance * self.monitoring.red_threshold_multiplier

        if variance_pct <= amber_threshold:
            return PerformanceScore.GREEN
        elif variance_pct <= red_threshold:
            return PerformanceScore.AMBER
        else:
            return PerformanceScore.RED

    def get_scope_3_target_year(self) -> int:
        """Return the adjusted Scope 3 interim target year accounting for lag.

        Returns:
            Scope 3 target year (5yr target year + scope_3_lag_years).
        """
        if self.interim_target_5yr.enabled:
            return self.interim_target_5yr.target_year + self.scope.scope_3_lag_years
        return DEFAULT_5YR_TARGET_YEAR + self.scope.scope_3_lag_years


# =============================================================================
# Pack Configuration Wrapper
# =============================================================================


class PackConfig(BaseModel):
    """Top-level pack configuration wrapper for PACK-029.

    Handles preset loading, environment variable overrides, and
    configuration merging. Provides SHA-256 config hashing for
    provenance tracking and JSON Schema export for API documentation.

    Example:
        >>> config = PackConfig.from_preset("sbti_1_5c_pathway")
        >>> print(config.pack.sbti_pathway)
        SBTiPathwayLevel.CELSIUS_1_5
        >>> config = PackConfig.from_preset("quarterly_monitoring", overrides={"monitoring": {"variance_tolerance_pct": 5.0}})
        >>> print(config.pack.monitoring.variance_tolerance_pct)
        5.0
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    pack: InterimTargetsConfig = Field(
        default_factory=InterimTargetsConfig,
        description="Main Interim Targets configuration",
    )
    preset_name: Optional[str] = Field(
        None,
        description="Name of the loaded preset",
    )
    config_version: str = Field(
        "1.0.0",
        description="Configuration schema version",
    )
    pack_id: str = Field(
        "PACK-029-interim-targets",
        description="Pack identifier",
    )

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "PackConfig":
        """Load configuration from a named preset.

        Loads the preset YAML file, applies environment variable overrides
        (INTERIM_TARGETS_* prefix), then applies any explicit runtime overrides.

        Args:
            preset_name: Name of the preset (sbti_1_5c_pathway, sbti_wb2c_pathway,
                quarterly_monitoring, annual_review, corrective_action,
                sector_specific, scope_3_extended).
            overrides: Optional dictionary of configuration overrides.

        Returns:
            PackConfig instance with preset values applied.

        Raises:
            FileNotFoundError: If preset YAML file does not exist.
            ValueError: If preset_name is not in SUPPORTED_PRESETS.
        """
        if preset_name not in SUPPORTED_PRESETS:
            raise ValueError(
                f"Unknown preset: {preset_name}. "
                f"Available presets: {sorted(SUPPORTED_PRESETS.keys())}"
            )

        preset_path = CONFIG_DIR / "presets" / f"{preset_name}.yaml"
        if not preset_path.exists():
            raise FileNotFoundError(
                f"Preset file not found: {preset_path}. "
                f"Run setup wizard to generate presets."
            )

        with open(preset_path, "r", encoding="utf-8") as f:
            preset_data = yaml.safe_load(f) or {}

        # Apply environment variable overrides
        env_overrides = _get_env_overrides("INTERIM_TARGETS_")
        if env_overrides:
            preset_data = _merge_config(preset_data, env_overrides)

        # Apply explicit overrides
        if overrides:
            preset_data = _merge_config(preset_data, overrides)

        pack_config = InterimTargetsConfig(**preset_data)
        return cls(pack=pack_config, preset_name=preset_name)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "PackConfig":
        """Load configuration from a YAML file.

        Args:
            yaml_path: Path to YAML configuration file.

        Returns:
            PackConfig instance with YAML values applied.

        Raises:
            FileNotFoundError: If YAML file does not exist.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

        pack_config = InterimTargetsConfig(**config_data)
        return cls(pack=pack_config)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PackConfig":
        """Load configuration from a dictionary.

        Args:
            config_dict: Configuration dictionary.

        Returns:
            PackConfig instance.
        """
        pack_config = InterimTargetsConfig(**config_dict)
        return cls(pack=pack_config)

    def get_config_hash(self) -> str:
        """Generate SHA-256 hash of the current configuration for provenance.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        config_json = self.model_dump_json(indent=None)
        return _compute_hash(config_json)

    def validate_config(self) -> List[str]:
        """Cross-field validation returning warnings.

        Performs advisory validation beyond Pydantic's built-in validation.
        Returns warnings, not hard errors.

        Returns:
            List of warning messages (empty if fully valid).
        """
        return validate_config(self.pack)

    def export_json_schema(self) -> Dict[str, Any]:
        """Export the configuration JSON Schema for API documentation.

        Returns:
            JSON Schema dictionary for the InterimTargetsConfig model.
        """
        return InterimTargetsConfig.model_json_schema()


# =============================================================================
# Utility Functions
# =============================================================================


def load_config(yaml_path: Union[str, Path]) -> PackConfig:
    """Load configuration from a YAML file.

    Convenience wrapper around PackConfig.from_yaml().

    Args:
        yaml_path: Path to YAML configuration file.

    Returns:
        PackConfig instance.
    """
    return PackConfig.from_yaml(yaml_path)


def load_preset(
    preset_name: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> PackConfig:
    """Load a named preset configuration.

    Convenience wrapper around PackConfig.from_preset().

    Args:
        preset_name: Name of the preset to load.
        overrides: Optional configuration overrides.

    Returns:
        PackConfig instance with preset applied.
    """
    return PackConfig.from_preset(preset_name, overrides)


def get_pathway_defaults(
    pathway: Union[str, SBTiPathwayLevel],
) -> InterimTargetsConfig:
    """Get default configuration for a given SBTi pathway level.

    Creates an InterimTargetsConfig with pathway-appropriate defaults
    for near-term reduction targets, annual rates, and monitoring
    settings.

    Args:
        pathway: SBTi pathway enum or string value ("1.5C" or "WB2C").

    Returns:
        InterimTargetsConfig with pathway defaults applied.
    """
    if isinstance(pathway, str):
        pathway = SBTiPathwayLevel(pathway)

    minimums = SBTI_NEAR_TERM_MINIMUMS.get(pathway.value, SBTI_NEAR_TERM_MINIMUMS["1.5C"])

    near_term_reduction = minimums["scope_1_2_reduction_pct"]
    annual_rate = minimums["scope_1_2_annual_linear_pct"]

    # Interpolate 10-year target (mid-point between near-term and long-term)
    long_term_pct = minimums["long_term_reduction_pct"]
    ten_year_pct = near_term_reduction + (long_term_pct - near_term_reduction) * 0.5

    return InterimTargetsConfig(
        sbti_pathway=pathway,
        interim_target_5yr=InterimTargetConfig(
            enabled=True,
            target_year=DEFAULT_5YR_TARGET_YEAR,
            min_reduction_pct=near_term_reduction,
            label=f"SBTi {pathway.value} near-term target",
        ),
        interim_target_10yr=InterimTargetConfig(
            enabled=True,
            target_year=DEFAULT_10YR_TARGET_YEAR,
            min_reduction_pct=round(ten_year_pct, 1),
            label=f"SBTi {pathway.value} mid-term target",
        ),
        annual_reduction_rate=annual_rate,
        long_term_reduction_pct=long_term_pct,
    )


def _merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence.

    Args:
        base: Base configuration dictionary.
        override: Override dictionary (values take precedence).

    Returns:
        Merged dictionary.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_config(result[key], value)
        else:
            result[key] = value
    return result


def merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Public deep merge two dictionaries, with override taking precedence.

    Args:
        base: Base configuration dictionary.
        override: Override dictionary (values take precedence).

    Returns:
        Merged dictionary.
    """
    return _merge_config(base, override)


def _get_env_overrides(prefix: str) -> Dict[str, Any]:
    """Load configuration overrides from environment variables.

    Environment variables prefixed with the given prefix are loaded and
    mapped to configuration keys. Nested keys use double underscore.

    Example:
        INTERIM_TARGETS_REPORTING_YEAR=2026
        INTERIM_TARGETS_MONITORING__VARIANCE_TOLERANCE_PCT=5.0
        INTERIM_TARGETS_SCOPE__SCOPE_3_LAG_YEARS=3

    Args:
        prefix: Environment variable prefix to search for.

    Returns:
        Dictionary of parsed overrides.
    """
    overrides: Dict[str, Any] = {}
    for key, value in os.environ.items():
        if key.startswith(prefix):
            config_key = key[len(prefix):].lower()
            parts = config_key.split("__")
            current = overrides
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            # Parse value types
            if value.lower() in ("true", "yes", "1"):
                current[parts[-1]] = True
            elif value.lower() in ("false", "no", "0"):
                current[parts[-1]] = False
            else:
                try:
                    current[parts[-1]] = int(value)
                except ValueError:
                    try:
                        current[parts[-1]] = float(value)
                    except ValueError:
                        current[parts[-1]] = value
    return overrides


def get_env_overrides(prefix: str) -> Dict[str, Any]:
    """Public wrapper for loading environment variable overrides.

    Args:
        prefix: Environment variable prefix to search for.

    Returns:
        Dictionary of parsed overrides.
    """
    return _get_env_overrides(prefix)


def validate_config(config: InterimTargetsConfig) -> List[str]:
    """Validate an interim targets configuration and return any warnings.

    Performs cross-field validation beyond what Pydantic validators cover.
    Returns advisory warnings, not hard errors.

    Args:
        config: InterimTargetsConfig instance to validate.

    Returns:
        List of warning messages (empty if fully valid).
    """
    warnings: List[str] = []

    # Check organization name is set
    if not config.organization_name:
        warnings.append(
            "Organization name is empty. Set organization_name for meaningful reports."
        )

    # Check baseline year is reasonable
    if config.baseline_year > config.reporting_year:
        warnings.append(
            f"Base year ({config.baseline_year}) is after reporting year "
            f"({config.reporting_year}). Base year should precede reporting year."
        )

    # Check SBTi minimum ambition
    minimums = SBTI_NEAR_TERM_MINIMUMS.get(config.sbti_pathway.value, {})
    min_scope_1_2 = minimums.get("scope_1_2_reduction_pct", 42.0)

    if config.interim_target_5yr.enabled:
        if config.interim_target_5yr.min_reduction_pct < min_scope_1_2:
            warnings.append(
                f"5-year interim target reduction ({config.interim_target_5yr.min_reduction_pct}%) "
                f"is below SBTi minimum ({min_scope_1_2}%) for {config.sbti_pathway.value} pathway."
            )

    # Check long-term reduction meets SBTi minimum
    if config.long_term_reduction_pct < 90.0:
        warnings.append(
            f"Long-term reduction ({config.long_term_reduction_pct}%) is below "
            f"SBTi minimum of 90%. Net-zero target may not be validated."
        )

    # Check net-zero year <= 2050
    if config.net_zero_year > 2050:
        warnings.append(
            f"Net-zero year ({config.net_zero_year}) exceeds SBTi maximum of 2050."
        )

    # Check scope coverage thresholds
    if config.scope.scope_1_coverage_pct < SBTI_COVERAGE_THRESHOLDS["near_term_scope_1_pct"]:
        warnings.append(
            f"Scope 1 coverage ({config.scope.scope_1_coverage_pct}%) is below "
            f"SBTi minimum of {SBTI_COVERAGE_THRESHOLDS['near_term_scope_1_pct']}%."
        )
    if config.scope.scope_2_coverage_pct < SBTI_COVERAGE_THRESHOLDS["near_term_scope_2_pct"]:
        warnings.append(
            f"Scope 2 coverage ({config.scope.scope_2_coverage_pct}%) is below "
            f"SBTi minimum of {SBTI_COVERAGE_THRESHOLDS['near_term_scope_2_pct']}%."
        )
    if config.scope.scope_3_coverage_pct < SBTI_COVERAGE_THRESHOLDS["near_term_scope_3_pct"]:
        warnings.append(
            f"Scope 3 coverage ({config.scope.scope_3_coverage_pct}%) is below "
            f"SBTi minimum of {SBTI_COVERAGE_THRESHOLDS['near_term_scope_3_pct']}%."
        )

    # Check pathway consistency
    if config.pathway_type == PathwayType.LINEAR and config.annual_reduction_rate is None:
        warnings.append(
            "LINEAR pathway selected but annual_reduction_rate not set. "
            "Consider setting annual_reduction_rate for explicit linear path."
        )

    if config.pathway_type == PathwayType.S_CURVE and config.s_curve_inflection_year is None:
        warnings.append(
            "S_CURVE pathway selected but s_curve_inflection_year not set. "
            "Default inflection year of 2035 will be used."
        )

    if config.pathway_type == PathwayType.MILESTONE_BASED:
        if not config.milestone_years:
            warnings.append(
                "MILESTONE_BASED pathway selected but milestone_years not set."
            )
        if not config.milestone_reductions_pct:
            warnings.append(
                "MILESTONE_BASED pathway selected but milestone_reductions_pct not set."
            )

    # Check monitoring frequency vs variance tolerance
    if (config.monitoring.monitoring_frequency == MonitoringFrequency.ANNUAL
            and config.monitoring.variance_tolerance_pct < 5.0):
        warnings.append(
            "Very tight variance tolerance with annual monitoring may generate "
            "excessive false alarms. Consider quarterly monitoring or wider tolerance."
        )

    # Check carbon budget consistency
    if config.carbon_budget.carbon_budget_enabled:
        if config.carbon_budget.budget_overshoot_allowed and not config.carbon_budget.cumulative_budget_tracking:
            warnings.append(
                "Budget overshoot is allowed but cumulative tracking is disabled. "
                "Enable cumulative_budget_tracking to track compensation."
            )

    # Check corrective action MACC integration
    if config.corrective_action.macc_integration:
        if not config.corrective_action.gap_quantification_enabled:
            warnings.append(
                "MACC integration enabled but gap quantification disabled. "
                "Enable gap_quantification_enabled for MACC to identify gap-closing levers."
            )

    # Check reporting frameworks
    if not config.reporting.reporting_frameworks:
        warnings.append(
            "No reporting frameworks configured. Add at least one framework "
            "(sbti, cdp, tcfd, gri) for disclosure readiness."
        )

    # Check SBTi validation consistency
    if config.sbti_validation.sbti_validation_enabled:
        if not config.scope.scope_3_included and config.scope.scope_3_coverage_pct >= 40.0:
            warnings.append(
                "Scope 3 is excluded from target tracking but Scope 3 coverage >= 40%. "
                "SBTi requires Scope 3 target when Scope 3 >= 40% of total."
            )

    # Check alerting configuration
    if config.alerting.email_alerts_enabled and not config.alerting.alert_recipients:
        warnings.append(
            "Email alerts enabled but no alert_recipients configured."
        )

    if config.alerting.slack_alerts_enabled and not config.alerting.slack_channel:
        warnings.append(
            "Slack alerts enabled but no slack_channel configured."
        )

    if config.alerting.webhook_alerts_enabled and not config.alerting.webhook_url:
        warnings.append(
            "Webhook alerts enabled but no webhook_url configured."
        )

    # Check interim target 10yr is interpolated correctly
    if (config.interim_target_5yr.enabled and config.interim_target_10yr.enabled):
        if config.interim_target_10yr.min_reduction_pct <= config.interim_target_5yr.min_reduction_pct:
            warnings.append(
                f"10-year target reduction ({config.interim_target_10yr.min_reduction_pct}%) "
                f"is not greater than 5-year target ({config.interim_target_5yr.min_reduction_pct}%). "
                f"10-year target should show additional progress toward net-zero."
            )

    # Check custom milestones ordering
    if config.custom_milestones:
        years = [m.target_year for m in config.custom_milestones if m.enabled]
        if years != sorted(years):
            warnings.append(
                "Custom milestones are not in chronological order. "
                "Consider re-ordering for clarity."
            )

    return warnings


def get_sbti_minimums(pathway: Union[str, SBTiPathwayLevel]) -> Dict[str, float]:
    """Get SBTi minimum reduction requirements for a pathway level.

    Args:
        pathway: SBTi pathway enum or string value.

    Returns:
        Dictionary with scope_1_2 and scope_3 reduction requirements.
    """
    key = pathway.value if isinstance(pathway, SBTiPathwayLevel) else pathway
    return SBTI_NEAR_TERM_MINIMUMS.get(key, SBTI_NEAR_TERM_MINIMUMS["1.5C"])


def get_sbti_criteria() -> Dict[str, str]:
    """Get all 21 SBTi validation criteria descriptions.

    Returns:
        Dictionary mapping criteria codes (C01-C21) to descriptions.
    """
    return SBTI_VALIDATION_CRITERIA.copy()


def get_performance_score_info(score: Union[str, PerformanceScore]) -> Dict[str, Any]:
    """Get performance score display information.

    Args:
        score: Performance score enum or string value.

    Returns:
        Dictionary with label, description, color_hex, and action_required.
    """
    key = score.value.upper() if isinstance(score, PerformanceScore) else score.upper()
    return PERFORMANCE_SCORES.get(key, PERFORMANCE_SCORES["RED"])


def get_root_cause_categories() -> Dict[str, List[str]]:
    """Get the root cause classification taxonomy.

    Returns:
        Dictionary mapping category names to lists of specific causes.
    """
    return ROOT_CAUSE_CATEGORIES.copy()


def get_budget_allocation_info(
    method: Union[str, BudgetAllocationMethod],
) -> Dict[str, Any]:
    """Get carbon budget allocation profile information.

    Args:
        method: Budget allocation method enum or string value.

    Returns:
        Dictionary with name, description, front_loading_factor, and risk_profile.
    """
    key = method.value if isinstance(method, BudgetAllocationMethod) else method
    return BUDGET_ALLOCATION_PROFILES.get(key, BUDGET_ALLOCATION_PROFILES["linear"])


def get_reporting_framework_info(framework: str) -> Dict[str, str]:
    """Get reporting framework details.

    Args:
        framework: Framework identifier (sbti, cdp, tcfd, gri, esrs, iso14064).

    Returns:
        Dictionary with name, version, relevant_sections, and disclosure_frequency.
    """
    return REPORTING_FRAMEWORK_MAPPING.get(
        framework.lower(),
        {"name": framework, "version": "Unknown", "relevant_sections": "N/A", "disclosure_frequency": "Annual"},
    )


def get_extrapolation_model_info(
    method: Union[str, ExtrapolationMethod],
) -> Dict[str, Any]:
    """Get trend extrapolation model parameters.

    Args:
        method: Extrapolation method enum or string value.

    Returns:
        Dictionary with name, description, min_data_points, and max_forecast_years.
    """
    key = method.value if isinstance(method, ExtrapolationMethod) else method
    return EXTRAPOLATION_MODEL_PARAMS.get(key, EXTRAPOLATION_MODEL_PARAMS["linear"])


def get_corrective_action_priority(gap_pct: float) -> Dict[str, Any]:
    """Get corrective action priority level based on gap percentage.

    Args:
        gap_pct: Percentage gap between actual and target emissions.

    Returns:
        Dictionary with priority level, response_days, escalation_level, and review_frequency.
    """
    for priority, params in sorted(
        CORRECTIVE_ACTION_PRIORITIES.items(),
        key=lambda x: x[1]["gap_pct_min"],
        reverse=True,
    ):
        if gap_pct >= params["gap_pct_min"]:
            return {"priority": priority, **params}
    return {"priority": "low", **CORRECTIVE_ACTION_PRIORITIES["low"]}


def list_available_presets() -> Dict[str, str]:
    """List all available configuration presets.

    Returns:
        Dictionary mapping preset names to descriptions.
    """
    return SUPPORTED_PRESETS.copy()


def list_sbti_pathways() -> Dict[str, Dict[str, float]]:
    """List all supported SBTi pathway levels with their minimums.

    Returns:
        Dictionary mapping pathway names to reduction requirements.
    """
    return SBTI_NEAR_TERM_MINIMUMS.copy()


def list_reporting_frameworks() -> Dict[str, str]:
    """List all supported reporting frameworks.

    Returns:
        Dictionary mapping framework codes to names.
    """
    return {k: v["name"] for k, v in REPORTING_FRAMEWORK_MAPPING.items()}


def list_variance_methods() -> Dict[str, str]:
    """List all supported variance analysis methods.

    Returns:
        Dictionary mapping method codes to descriptions.
    """
    return {
        "lmdi": "Logarithmic Mean Divisia Index decomposition",
        "kaya": "Kaya Identity decomposition",
        "both": "Combined LMDI + Kaya analysis",
    }


def list_extrapolation_methods() -> Dict[str, str]:
    """List all supported trend extrapolation methods.

    Returns:
        Dictionary mapping method codes to names.
    """
    return {k: v["name"] for k, v in EXTRAPOLATION_MODEL_PARAMS.items()}


def list_pathway_types() -> Dict[str, str]:
    """List all supported pathway types.

    Returns:
        Dictionary mapping pathway type codes to descriptions.
    """
    return {
        "linear": "Constant absolute reduction per year",
        "milestone_based": "Stepped reductions at defined milestone years",
        "accelerating": "Increasing rate of reduction over time",
        "s_curve": "Sigmoidal curve with slow start, rapid mid-phase, and plateau",
    }
