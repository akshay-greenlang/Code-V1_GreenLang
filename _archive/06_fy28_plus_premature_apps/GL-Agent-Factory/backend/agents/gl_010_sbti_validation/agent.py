"""
GL-010: SBTi Validation Agent

This module implements the Science Based Targets initiative (SBTi) Validation Agent
for validating corporate emission reduction targets against SBTi methodologies.

The agent supports:
- Near-term targets (5-10 years) with 4.2% annual linear reduction for 1.5C
- Long-term targets (2050) with 90% absolute reduction
- Net-zero targets (90-95% reduction + neutralization + BVCM)
- Scope 3 targets with 2.5% annual reduction OR 67% supplier engagement
- Absolute Contraction Approach (ACA) pathway validation
- Sectoral Decarbonization Approach (SDA) for sector-specific targets
- FLAG (Forest, Land, Agriculture) sector-specific pathways
- Complete SHA-256 provenance tracking

Aligned with:
- SBTi Corporate Net-Zero Standard (Version 1.2, 2024)
- SBTi Corporate Manual (2024)
- SBTi FLAG Guidance (2022)
- GHG Protocol Corporate Standard

Example:
    >>> agent = SBTiValidationAgent()
    >>> result = agent.run(SBTiInput(
    ...     company_id="COMPANY-001",
    ...     base_year=2019,
    ...     base_year_emissions=ScopeEmissions(scope1=1000, scope2=500, scope3=3000),
    ...     targets=[TargetDefinition(
    ...         target_year=2030,
    ...         target_type=TargetType.NEAR_TERM,
    ...         reduction_pct=46.2
    ...     )]
    ... ))
    >>> print(f"Target valid: {result.validation_result.is_valid}")
"""

import hashlib
import json
import logging
import math
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================


class TargetType(str, Enum):
    """SBTi target types with required timeframes."""

    NEAR_TERM = "near_term"  # 5-10 years from submission
    LONG_TERM = "long_term"  # 2050 or sooner
    NET_ZERO = "net_zero"  # 90-95% reduction + neutralization
    FLAG = "flag"  # Forest, Land, and Agriculture sector


class AmbitionLevel(str, Enum):
    """Target ambition levels aligned with climate scenarios."""

    CELSIUS_1_5 = "1.5C"  # 1.5C pathway - highest ambition
    WELL_BELOW_2C = "WB2C"  # Well-below 2C pathway
    CELSIUS_2 = "2C"  # 2C pathway - minimum for SBTi
    BELOW_THRESHOLD = "below_threshold"  # Does not meet SBTi minimum


class PathwayType(str, Enum):
    """Decarbonization pathway approaches."""

    ACA = "absolute_contraction"  # Absolute Contraction Approach
    SDA = "sectoral_decarbonization"  # Sectoral Decarbonization Approach
    FLAG = "flag_pathway"  # FLAG sector pathway


class SectorPathway(str, Enum):
    """Sectors with SDA pathway support."""

    POWER_GENERATION = "power_generation"
    STEEL = "steel"
    CEMENT = "cement"
    ALUMINUM = "aluminum"
    TRANSPORT_ROAD = "transport_road"
    TRANSPORT_AVIATION = "transport_aviation"
    TRANSPORT_SHIPPING = "transport_shipping"
    BUILDINGS = "buildings"
    PULP_PAPER = "pulp_paper"
    CHEMICALS = "chemicals"
    OIL_GAS = "oil_gas"
    GENERAL = "general"  # Cross-sector default
    # FLAG sectors
    FLAG_AGRICULTURE = "flag_agriculture"
    FLAG_FORESTRY = "flag_forestry"
    FLAG_LAND_USE = "flag_land_use"


class ValidationStatus(str, Enum):
    """Target validation status."""

    VALID = "valid"
    INVALID = "invalid"
    NEEDS_REVIEW = "needs_review"


class ScopeType(str, Enum):
    """GHG Protocol emission scopes."""

    SCOPE_1 = "scope1"
    SCOPE_2 = "scope2"
    SCOPE_3 = "scope3"


class Scope3EngagementType(str, Enum):
    """Scope 3 reduction approaches."""

    ABSOLUTE_REDUCTION = "absolute_reduction"  # 2.5% annual reduction
    SUPPLIER_ENGAGEMENT = "supplier_engagement"  # 67% of suppliers with SBTs
    HYBRID = "hybrid"  # Combined approach


class NeutralizationType(str, Enum):
    """Net-zero neutralization approaches."""

    CARBON_REMOVAL = "carbon_removal"
    NATURE_BASED = "nature_based"
    TECHNOLOGY_BASED = "technology_based"
    COMBINED = "combined"


class ProgressStatus(str, Enum):
    """Progress tracking status."""

    ON_TRACK = "on_track"
    SLIGHTLY_BEHIND = "slightly_behind"
    SIGNIFICANTLY_BEHIND = "significantly_behind"
    AHEAD = "ahead"
    AT_RISK = "at_risk"


# =============================================================================
# INPUT MODELS
# =============================================================================


class ScopeEmissions(BaseModel):
    """
    GHG emissions by scope.

    All values in tonnes CO2e (tCO2e).
    """

    scope1: float = Field(..., ge=0, description="Scope 1 direct emissions (tCO2e)")
    scope2: float = Field(..., ge=0, description="Scope 2 indirect emissions (tCO2e)")
    scope3: float = Field(0.0, ge=0, description="Scope 3 value chain emissions (tCO2e)")
    scope2_method: str = Field(
        "market",
        description="Scope 2 accounting method (location/market)"
    )
    flag_emissions: float = Field(
        0.0,
        ge=0,
        description="FLAG sector emissions (tCO2e) - separate from Scopes"
    )

    @property
    def total(self) -> float:
        """Calculate total emissions across all scopes."""
        return self.scope1 + self.scope2 + self.scope3

    @property
    def total_with_flag(self) -> float:
        """Calculate total emissions including FLAG."""
        return self.total + self.flag_emissions

    @property
    def scope3_percentage(self) -> float:
        """Calculate Scope 3 as percentage of total emissions."""
        total = self.total
        if total == 0:
            return 0.0
        return (self.scope3 / total) * 100

    @property
    def scope12_total(self) -> float:
        """Calculate combined Scope 1 and 2 emissions."""
        return self.scope1 + self.scope2

    @property
    def flag_percentage(self) -> float:
        """Calculate FLAG as percentage of total emissions."""
        total_with_flag = self.total_with_flag
        if total_with_flag == 0:
            return 0.0
        return (self.flag_emissions / total_with_flag) * 100


class IntensityMetric(BaseModel):
    """
    Intensity metric for SDA pathway.

    Example: tCO2e per MWh, tCO2e per tonne of steel
    """

    value: float = Field(..., ge=0, description="Intensity value")
    numerator_unit: str = Field("tCO2e", description="Emissions unit")
    denominator_unit: str = Field(..., description="Activity unit (e.g., MWh, tonne)")
    activity_type: str = Field(..., description="Activity type for intensity")


class Scope3EngagementTarget(BaseModel):
    """
    Scope 3 supplier engagement target details.

    For companies using the engagement approach instead of absolute reduction.
    """

    engagement_type: Scope3EngagementType = Field(
        Scope3EngagementType.ABSOLUTE_REDUCTION,
        description="Type of Scope 3 approach"
    )
    supplier_coverage_pct: float = Field(
        0.0,
        ge=0,
        le=100,
        description="Percentage of suppliers with SBTs"
    )
    supplier_target_year: Optional[int] = Field(
        None,
        description="Year by which suppliers should have SBTs"
    )
    scope3_reduction_pct: float = Field(
        0.0,
        ge=0,
        le=100,
        description="Scope 3 absolute reduction percentage"
    )


class NeutralizationPlan(BaseModel):
    """
    Net-zero neutralization plan for residual emissions.

    Required for net-zero targets to address the 5-10% residual emissions.
    """

    neutralization_type: NeutralizationType = Field(
        ...,
        description="Type of neutralization approach"
    )
    residual_emissions_pct: float = Field(
        ...,
        ge=0,
        le=15,
        description="Percentage of residual emissions (max 10% for most sectors)"
    )
    removal_capacity_tco2e: float = Field(
        ...,
        ge=0,
        description="Annual carbon removal capacity"
    )
    removal_sources: List[str] = Field(
        default_factory=list,
        description="Sources of carbon removal"
    )
    bvcm_commitment: bool = Field(
        False,
        description="Beyond Value Chain Mitigation commitment"
    )
    bvcm_investment_pct: float = Field(
        0.0,
        ge=0,
        description="Percentage of revenue committed to BVCM"
    )


class FLAGTarget(BaseModel):
    """
    Forest, Land, and Agriculture (FLAG) sector target.

    Separate from non-FLAG targets per SBTi FLAG guidance.
    """

    base_year_flag_emissions: float = Field(
        ...,
        ge=0,
        description="Base year FLAG emissions (tCO2e)"
    )
    target_year_flag_emissions: float = Field(
        ...,
        ge=0,
        description="Target year FLAG emissions (tCO2e)"
    )
    no_deforestation_commitment: bool = Field(
        True,
        description="Commitment to zero deforestation by 2025"
    )
    no_deforestation_date: Optional[int] = Field(
        2025,
        description="Year for no-deforestation commitment"
    )
    land_conversion_commitment: bool = Field(
        True,
        description="Commitment to no land conversion"
    )
    commodity_focus: List[str] = Field(
        default_factory=list,
        description="FLAG commodities covered (e.g., soy, palm, beef)"
    )
    sequestration_target: float = Field(
        0.0,
        ge=0,
        description="Carbon sequestration target (tCO2e)"
    )


class TargetDefinition(BaseModel):
    """
    Definition of a single emission reduction target.

    Supports both absolute and intensity-based targets.
    """

    target_id: Optional[str] = Field(None, description="Unique target identifier")
    target_year: int = Field(..., ge=2020, le=2100, description="Target year")
    target_type: TargetType = Field(..., description="Type of target")
    reduction_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Percentage reduction from base year"
    )
    scopes_covered: List[ScopeType] = Field(
        default_factory=lambda: [ScopeType.SCOPE_1, ScopeType.SCOPE_2],
        description="Emission scopes included in target"
    )
    pathway_type: PathwayType = Field(
        PathwayType.ACA,
        description="Decarbonization pathway approach"
    )
    sector: Optional[SectorPathway] = Field(
        None,
        description="Sector for SDA pathway (required if pathway=SDA)"
    )
    base_intensity: Optional[IntensityMetric] = Field(
        None,
        description="Base year intensity (for intensity targets)"
    )
    target_intensity: Optional[IntensityMetric] = Field(
        None,
        description="Target year intensity (for intensity targets)"
    )
    is_absolute_target: bool = Field(
        True,
        description="True for absolute targets, False for intensity"
    )
    scope3_categories_covered: List[int] = Field(
        default_factory=list,
        description="Scope 3 categories included (1-15)"
    )
    scope3_engagement: Optional[Scope3EngagementTarget] = Field(
        None,
        description="Scope 3 supplier engagement details"
    )
    neutralization_plan: Optional[NeutralizationPlan] = Field(
        None,
        description="Neutralization plan for net-zero targets"
    )
    flag_target: Optional[FLAGTarget] = Field(
        None,
        description="FLAG sector target details"
    )

    @model_validator(mode="after")
    def validate_sda_sector(self):
        """Validate SDA pathway has sector specified."""
        if self.pathway_type == PathwayType.SDA and not self.sector:
            logger.warning("SDA pathway requires sector specification, defaulting to GENERAL")
            self.sector = SectorPathway.GENERAL
        return self

    @model_validator(mode="after")
    def validate_net_zero_requirements(self):
        """Validate net-zero targets have neutralization plan."""
        if self.target_type == TargetType.NET_ZERO and not self.neutralization_plan:
            logger.warning(
                "Net-zero target should include neutralization plan for residual emissions"
            )
        return self

    @model_validator(mode="after")
    def validate_flag_requirements(self):
        """Validate FLAG targets have FLAG details."""
        if self.target_type == TargetType.FLAG and not self.flag_target:
            logger.warning("FLAG target type requires flag_target details")
        return self


class CurrentProgress(BaseModel):
    """
    Current progress towards target.

    Tracks actual emissions for progress calculation.
    """

    reporting_year: int = Field(..., ge=2015, description="Current reporting year")
    current_emissions: ScopeEmissions = Field(..., description="Current year emissions")
    current_intensity: Optional[IntensityMetric] = Field(
        None,
        description="Current intensity (for intensity targets)"
    )
    verified: bool = Field(
        False,
        description="Whether emissions have been third-party verified"
    )
    verification_standard: Optional[str] = Field(
        None,
        description="Verification standard used (e.g., ISO 14064-3)"
    )


class SBTiInput(BaseModel):
    """
    Complete input model for SBTi Validation Agent.

    Attributes:
        company_id: Unique company identifier
        company_name: Human-readable company name
        base_year: Emissions base year for targets
        base_year_emissions: Base year emissions by scope
        targets: List of emission reduction targets
        current_progress: Optional current progress data
        sector: Company's primary sector
        submission_date: Date of target submission to SBTi
        has_flag_emissions: Whether company has FLAG sector emissions
        metadata: Additional company metadata
    """

    # Company identification
    company_id: str = Field(..., description="Unique company identifier")
    company_name: Optional[str] = Field(None, description="Company name")

    # Base year data
    base_year: int = Field(..., ge=2015, le=2025, description="Emissions base year")
    base_year_emissions: ScopeEmissions = Field(..., description="Base year emissions")

    # Targets
    targets: List[TargetDefinition] = Field(
        ...,
        min_length=1,
        description="Emission reduction targets"
    )

    # Current progress (optional)
    current_progress: Optional[CurrentProgress] = Field(
        None,
        description="Current progress towards targets"
    )

    # Company context
    sector: SectorPathway = Field(
        SectorPathway.GENERAL,
        description="Company's primary sector"
    )
    submission_date: Optional[datetime] = Field(
        None,
        description="Target submission date to SBTi"
    )

    # FLAG sector
    has_flag_emissions: bool = Field(
        False,
        description="Whether company has FLAG sector emissions"
    )
    flag_threshold_pct: float = Field(
        20.0,
        description="FLAG emissions threshold requiring separate target"
    )

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("base_year")
    @classmethod
    def validate_base_year(cls, v: int) -> int:
        """Validate base year is not more than 2 years old for submission."""
        current_year = datetime.now().year
        if v < current_year - 7:
            logger.warning(
                f"Base year {v} is more than 7 years old. "
                "SBTi recommends recalculating for older base years."
            )
        return v

    @model_validator(mode="after")
    def validate_scope3_threshold(self):
        """Warn if Scope 3 coverage required but not provided."""
        scope3_pct = self.base_year_emissions.scope3_percentage
        if scope3_pct > 40:
            has_scope3_target = any(
                ScopeType.SCOPE_3 in t.scopes_covered
                for t in self.targets
            )
            if not has_scope3_target:
                logger.warning(
                    f"Scope 3 is {scope3_pct:.1f}% of total emissions (>40%). "
                    "SBTi requires Scope 3 target for near-term targets."
                )
        return self

    @model_validator(mode="after")
    def validate_flag_threshold(self):
        """Warn if FLAG target required but not provided."""
        if self.has_flag_emissions:
            flag_pct = self.base_year_emissions.flag_percentage
            if flag_pct > self.flag_threshold_pct:
                has_flag_target = any(
                    t.target_type == TargetType.FLAG
                    for t in self.targets
                )
                if not has_flag_target:
                    logger.warning(
                        f"FLAG emissions are {flag_pct:.1f}% of total (>{self.flag_threshold_pct}%). "
                        "SBTi requires separate FLAG target."
                    )
        return self


# =============================================================================
# OUTPUT MODELS
# =============================================================================


class TargetValidation(BaseModel):
    """Validation result for a single target."""

    target_id: Optional[str] = Field(None, description="Target identifier")
    target_year: int = Field(..., description="Target year")
    target_type: str = Field(..., description="Target type")
    is_valid: bool = Field(..., description="Whether target is valid")
    status: ValidationStatus = Field(..., description="Validation status")
    ambition_level: str = Field(..., description="Target ambition classification")
    required_reduction_pct: float = Field(
        ...,
        description="Minimum required reduction for pathway"
    )
    actual_reduction_pct: float = Field(..., description="Submitted reduction percentage")
    reduction_gap_pct: float = Field(
        ...,
        description="Gap between actual and required reduction"
    )
    annual_reduction_rate: float = Field(
        0.0,
        description="Required annual reduction rate"
    )
    timeframe_valid: bool = Field(..., description="Whether timeframe meets requirements")
    scope_coverage_valid: bool = Field(..., description="Whether scope coverage is valid")
    net_zero_compliant: bool = Field(
        False,
        description="Whether target meets net-zero requirements"
    )
    flag_compliant: bool = Field(
        False,
        description="Whether FLAG requirements are met"
    )
    messages: List[str] = Field(default_factory=list, description="Validation messages")


class ValidationResult(BaseModel):
    """Complete validation result for all targets."""

    is_valid: bool = Field(..., description="Overall validation status")
    status: ValidationStatus = Field(..., description="Overall status")
    highest_ambition: str = Field(..., description="Highest ambition level achieved")
    target_validations: List[TargetValidation] = Field(
        default_factory=list,
        description="Individual target validations"
    )
    overall_messages: List[str] = Field(
        default_factory=list,
        description="Overall validation messages"
    )
    sbti_aligned: bool = Field(..., description="Whether targets are SBTi-aligned")
    net_zero_aligned: bool = Field(
        False,
        description="Whether targets meet net-zero standard"
    )
    flag_compliant: bool = Field(
        False,
        description="Whether FLAG requirements are met"
    )
    scope3_compliant: bool = Field(
        False,
        description="Whether Scope 3 requirements are met"
    )


class TargetTrajectoryPoint(BaseModel):
    """Single point on a target trajectory."""

    year: int = Field(..., description="Calendar year")
    target_emissions: float = Field(..., description="Target emissions for this year")
    cumulative_reduction_pct: float = Field(
        ...,
        description="Cumulative reduction from base year"
    )
    annual_reduction_rate: float = Field(
        ...,
        description="Annual reduction rate to next year"
    )
    actual_emissions: Optional[float] = Field(
        None,
        description="Actual emissions if available"
    )
    variance_pct: Optional[float] = Field(
        None,
        description="Variance from target if actual available"
    )


class TargetTrajectory(BaseModel):
    """Complete target trajectory from base year to target year."""

    base_year: int = Field(..., description="Base year")
    target_year: int = Field(..., description="Target year")
    base_emissions: float = Field(..., description="Base year emissions")
    target_emissions: float = Field(..., description="Target year emissions")
    total_reduction_pct: float = Field(..., description="Total reduction percentage")
    annual_reduction_rate: float = Field(..., description="Required annual reduction rate")
    pathway: str = Field(..., description="Pathway used (1.5C, WB2C, 2C)")
    trajectory_points: List[TargetTrajectoryPoint] = Field(
        default_factory=list,
        description="Year-by-year trajectory points"
    )


class ProgressTracking(BaseModel):
    """Progress tracking towards targets."""

    base_year: int = Field(..., description="Base year")
    current_year: int = Field(..., description="Current reporting year")
    target_year: int = Field(..., description="Target year")
    years_elapsed: int = Field(..., description="Years since base year")
    years_remaining: int = Field(..., description="Years to target")
    base_emissions_tco2e: float = Field(..., description="Base year emissions")
    current_emissions_tco2e: float = Field(..., description="Current emissions")
    target_emissions_tco2e: float = Field(..., description="Target year emissions")
    reduction_achieved_pct: float = Field(
        ...,
        description="Reduction achieved from base year"
    )
    reduction_required_pct: float = Field(
        ...,
        description="Total reduction required"
    )
    expected_reduction_by_now_pct: float = Field(
        ...,
        description="Expected reduction at this point"
    )
    on_track: bool = Field(..., description="Whether on track to meet target")
    progress_status: ProgressStatus = Field(
        ...,
        description="Detailed progress status"
    )
    gap_to_trajectory_pct: float = Field(
        ...,
        description="Gap between actual and expected progress"
    )
    expected_reduction_pct: float = Field(
        ...,
        description="Expected reduction at current pace"
    )
    annual_reduction_needed_pct: float = Field(
        ...,
        description="Annual reduction needed to meet target"
    )
    trajectory_chart_data: List[Dict[str, float]] = Field(
        default_factory=list,
        description="Data points for trajectory visualization"
    )
    gap_analysis: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed gap analysis"
    )


class Recommendation(BaseModel):
    """Improvement recommendation."""

    priority: str = Field(..., description="Priority: high, medium, low")
    category: str = Field(..., description="Category of recommendation")
    title: str = Field(..., description="Recommendation title")
    description: str = Field(..., description="Detailed description")
    impact: str = Field(..., description="Expected impact")
    implementation_steps: List[str] = Field(
        default_factory=list,
        description="Steps to implement"
    )


class PathwayCalculation(BaseModel):
    """Detailed pathway calculation results."""

    pathway_type: str = Field(..., description="Pathway type used")
    annual_reduction_rate: float = Field(
        ...,
        description="Required annual reduction rate"
    )
    base_year_value: float = Field(..., description="Base year value")
    target_year_value: float = Field(..., description="Calculated target value")
    formula_used: str = Field(..., description="Formula applied")
    sector_benchmark: Optional[float] = Field(
        None,
        description="Sector benchmark value (for SDA)"
    )
    calculation_steps: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Calculation audit trail"
    )


class NetZeroValidation(BaseModel):
    """Net-zero specific validation results."""

    is_net_zero_compliant: bool = Field(..., description="Overall net-zero compliance")
    gross_reduction_pct: float = Field(..., description="Gross emissions reduction")
    residual_emissions_pct: float = Field(..., description="Residual emissions percentage")
    neutralization_valid: bool = Field(..., description="Neutralization plan validity")
    bvcm_commitment: bool = Field(..., description="BVCM commitment present")
    removal_capacity_sufficient: bool = Field(
        ...,
        description="Whether removal capacity covers residuals"
    )
    near_term_target_present: bool = Field(
        ...,
        description="Whether near-term target exists"
    )
    long_term_target_present: bool = Field(
        ...,
        description="Whether long-term target exists"
    )
    messages: List[str] = Field(default_factory=list, description="Validation messages")


class FLAGValidation(BaseModel):
    """FLAG sector validation results."""

    is_flag_compliant: bool = Field(..., description="Overall FLAG compliance")
    flag_emissions_pct: float = Field(..., description="FLAG as percentage of total")
    separate_target_required: bool = Field(..., description="Whether separate target needed")
    separate_target_present: bool = Field(..., description="Whether separate target exists")
    no_deforestation_commitment: bool = Field(..., description="No-deforestation commitment")
    no_deforestation_by_2025: bool = Field(..., description="Meeting 2025 deadline")
    land_conversion_commitment: bool = Field(..., description="Land conversion commitment")
    reduction_pct: float = Field(0.0, description="FLAG emissions reduction percentage")
    sequestration_included: bool = Field(False, description="Sequestration in targets")
    messages: List[str] = Field(default_factory=list, description="Validation messages")


class SBTiOutput(BaseModel):
    """
    Complete output model for SBTi Validation Agent.

    Comprehensive target validation, progress tracking, and recommendations.
    """

    # Company identification
    company_id: str = Field(..., description="Company identifier")
    validation_id: str = Field(..., description="Unique validation ID")

    # Validation results
    validation_result: ValidationResult = Field(
        ...,
        description="Complete validation result"
    )

    # Target classifications
    target_classification: str = Field(
        ...,
        description="Overall target classification (1.5C, WB2C, 2C)"
    )

    # Pathway calculations
    pathway_calculations: List[PathwayCalculation] = Field(
        default_factory=list,
        description="Detailed pathway calculations"
    )

    # Target trajectories
    target_trajectories: List[TargetTrajectory] = Field(
        default_factory=list,
        description="Year-by-year target trajectories"
    )

    # Progress tracking
    progress_tracking: Optional[ProgressTracking] = Field(
        None,
        description="Progress towards targets"
    )

    # Net-zero validation
    net_zero_validation: Optional[NetZeroValidation] = Field(
        None,
        description="Net-zero specific validation"
    )

    # FLAG validation
    flag_validation: Optional[FLAGValidation] = Field(
        None,
        description="FLAG sector validation"
    )

    # Recommendations
    recommendations: List[Recommendation] = Field(
        default_factory=list,
        description="Improvement recommendations"
    )

    # Scope 3 analysis
    scope3_required: bool = Field(..., description="Whether Scope 3 target required")
    scope3_coverage_pct: float = Field(
        ...,
        description="Percentage of Scope 3 covered by target"
    )

    # Audit trail
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    calculation_methodology: str = Field(
        "SBTi Corporate Net-Zero Standard v1.2",
        description="Methodology used"
    )
    processing_time_ms: float = Field(..., description="Processing duration")
    calculated_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# SBTI PATHWAY CONSTANTS
# =============================================================================


class SBTiPathwayConstants:
    """
    SBTi pathway constants and benchmarks.

    Source: SBTi Corporate Net-Zero Standard v1.2 (2024)
    """

    # ACA Annual Reduction Rates (ZERO-HALLUCINATION: Fixed values from SBTi)
    # Required linear annual reduction for 1.5C pathway
    ACA_1_5C_ANNUAL_REDUCTION = 0.042  # 4.2% per year

    # Required linear annual reduction for well-below 2C pathway
    ACA_WB2C_ANNUAL_REDUCTION = 0.025  # 2.5% per year

    # Required linear annual reduction for 2C pathway
    ACA_2C_ANNUAL_REDUCTION = 0.016  # 1.6% per year

    # Scope 3 annual reduction rate for 1.5C
    SCOPE3_1_5C_ANNUAL_REDUCTION = 0.025  # 2.5% per year

    # Near-term target requirements (by 2030 from base year)
    NEAR_TERM_MIN_YEARS = 5
    NEAR_TERM_MAX_YEARS = 10
    NEAR_TERM_1_5C_MIN_REDUCTION = 42.0  # 42% by 2030
    NEAR_TERM_WB2C_MIN_REDUCTION = 25.0  # 25% by 2030

    # Long-term target requirements (2050)
    LONG_TERM_TARGET_YEAR = 2050
    LONG_TERM_1_5C_REDUCTION = 90.0  # 90% reduction for 1.5C
    LONG_TERM_2C_REDUCTION = 80.0  # 80% reduction for 2C

    # Net-zero requirements
    NET_ZERO_MIN_REDUCTION = 90.0  # Minimum 90% before neutralization
    NET_ZERO_MAX_RESIDUAL = 10.0  # Maximum 10% residual emissions
    NET_ZERO_FLAG_MAX_RESIDUAL = 20.0  # Maximum 20% for FLAG-heavy companies

    # Scope 3 thresholds
    SCOPE3_THRESHOLD_PCT = 40.0  # Scope 3 target required if >40%
    SCOPE3_MIN_COVERAGE_PCT = 67.0  # At least 67% of Scope 3 must be covered
    SCOPE3_SUPPLIER_ENGAGEMENT_PCT = 67.0  # 67% of suppliers with SBTs
    SCOPE3_TIMELINE_YEARS = 5  # Scope 3 targets within 5-10 years

    # FLAG requirements
    FLAG_THRESHOLD_PCT = 20.0  # Separate FLAG target if >20% of emissions
    FLAG_NO_DEFORESTATION_YEAR = 2025  # No deforestation by 2025
    FLAG_1_5C_REDUCTION = 72.0  # 72% reduction for FLAG by 2050 (1.5C)
    FLAG_WB2C_REDUCTION = 50.0  # 50% reduction for FLAG by 2050 (WB2C)

    # SDA Sector intensity targets (2050 benchmarks in tCO2e per unit)
    SDA_TARGETS_2050: Dict[SectorPathway, Dict[str, float]] = {
        SectorPathway.POWER_GENERATION: {
            "intensity_2050": 0.014,  # tCO2e/MWh
            "unit": "MWh",
            "decay_rate": 0.065,  # k factor for exponential decay
        },
        SectorPathway.STEEL: {
            "intensity_2050": 0.38,  # tCO2e/tonne steel
            "unit": "tonne",
            "decay_rate": 0.045,
        },
        SectorPathway.CEMENT: {
            "intensity_2050": 0.25,  # tCO2e/tonne cement
            "unit": "tonne",
            "decay_rate": 0.040,
        },
        SectorPathway.ALUMINUM: {
            "intensity_2050": 0.50,  # tCO2e/tonne aluminum
            "unit": "tonne",
            "decay_rate": 0.055,
        },
        SectorPathway.TRANSPORT_ROAD: {
            "intensity_2050": 0.020,  # tCO2e/km
            "unit": "vehicle-km",
            "decay_rate": 0.058,
        },
        SectorPathway.TRANSPORT_AVIATION: {
            "intensity_2050": 0.35,  # tCO2e/tonne-km
            "unit": "tonne-km",
            "decay_rate": 0.035,
        },
        SectorPathway.TRANSPORT_SHIPPING: {
            "intensity_2050": 0.008,  # tCO2e/tonne-km
            "unit": "tonne-km",
            "decay_rate": 0.042,
        },
        SectorPathway.BUILDINGS: {
            "intensity_2050": 0.015,  # tCO2e/m2
            "unit": "m2",
            "decay_rate": 0.048,
        },
        SectorPathway.PULP_PAPER: {
            "intensity_2050": 0.28,  # tCO2e/tonne
            "unit": "tonne",
            "decay_rate": 0.038,
        },
        SectorPathway.CHEMICALS: {
            "intensity_2050": 0.60,  # tCO2e/tonne
            "unit": "tonne",
            "decay_rate": 0.035,
        },
        SectorPathway.OIL_GAS: {
            "intensity_2050": 0.015,  # tCO2e/GJ
            "unit": "GJ",
            "decay_rate": 0.055,
        },
        SectorPathway.GENERAL: {
            "intensity_2050": 0.50,  # Default cross-sector
            "unit": "unit",
            "decay_rate": 0.042,
        },
    }

    # FLAG sector pathway rates
    FLAG_PATHWAY_RATES: Dict[str, float] = {
        "agriculture": 0.035,  # 3.5% annual reduction
        "forestry": 0.040,  # 4.0% annual reduction
        "land_use": 0.030,  # 3.0% annual reduction
    }


# =============================================================================
# SBTI VALIDATION AGENT IMPLEMENTATION
# =============================================================================


class SBTiValidationAgent:
    """
    GL-010: SBTi Validation Agent.

    This agent validates corporate emission reduction targets against
    Science Based Targets initiative (SBTi) methodologies using
    zero-hallucination deterministic calculations.

    Validation Approaches:
    1. Absolute Contraction Approach (ACA):
       target_emissions = base_emissions * (1 - annual_rate) ^ years

    2. Sectoral Decarbonization Approach (SDA):
       target_intensity = I_2050 + (I_base - I_2050) * exp(-k * years)

    3. FLAG Pathway:
       Separate pathway for Forest, Land, and Agriculture emissions

    Target Types:
    - Near-term: 5-10 years, minimum 42% for 1.5C
    - Long-term: 2050, minimum 90% for 1.5C
    - Net-zero: 90%+ reduction before neutralization
    - FLAG: Separate targets for land sector emissions

    Attributes:
        constants: SBTi pathway constants
        config: Agent configuration

    Example:
        >>> agent = SBTiValidationAgent()
        >>> result = agent.run(SBTiInput(
        ...     company_id="COMPANY-001",
        ...     base_year=2019,
        ...     base_year_emissions=ScopeEmissions(scope1=1000, scope2=500),
        ...     targets=[TargetDefinition(
        ...         target_year=2030,
        ...         target_type=TargetType.NEAR_TERM,
        ...         reduction_pct=46.2
        ...     )]
        ... ))
        >>> assert result.validation_result.sbti_aligned
    """

    AGENT_ID = "targets/sbti_validation_v1"
    VERSION = "2.0.0"
    DESCRIPTION = "SBTi target validation with pathway analysis, FLAG support, and net-zero"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the SBTi Validation Agent.

        Args:
            config: Optional configuration overrides
        """
        self.config = config or {}
        self.constants = SBTiPathwayConstants()
        self._provenance_steps: List[Dict] = []

        logger.info(f"SBTiValidationAgent initialized (version {self.VERSION})")

    def run(self, input_data: SBTiInput) -> SBTiOutput:
        """
        Execute SBTi target validation.

        ZERO-HALLUCINATION validation using:
        - Fixed SBTi pathway constants
        - Deterministic formulas from SBTi methodology
        - Explicit threshold checks

        Args:
            input_data: Validated SBTi input data

        Returns:
            Comprehensive validation output with recommendations

        Raises:
            ValueError: If input validation fails
        """
        start_time = datetime.utcnow()
        self._provenance_steps = []

        # Generate validation ID
        validation_id = f"SBTI-VAL-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        logger.info(
            f"Validating SBTi targets: company={input_data.company_id}, "
            f"base_year={input_data.base_year}, "
            f"targets={len(input_data.targets)}"
        )

        try:
            # Step 1: Check Scope 3 requirements
            scope3_required, scope3_coverage = self._check_scope3_requirements(
                input_data
            )
            self._track_step("scope3_check", {
                "scope3_pct_of_total": input_data.base_year_emissions.scope3_percentage,
                "scope3_required": scope3_required,
                "scope3_coverage_pct": scope3_coverage,
            })

            # Step 2: Validate each target
            target_validations: List[TargetValidation] = []
            pathway_calculations: List[PathwayCalculation] = []
            target_trajectories: List[TargetTrajectory] = []

            for target in input_data.targets:
                validation, pathway_calc = self._validate_target(
                    target,
                    input_data.base_year,
                    input_data.base_year_emissions,
                    input_data.submission_date,
                )
                target_validations.append(validation)
                if pathway_calc:
                    pathway_calculations.append(pathway_calc)

                # Generate trajectory for each target
                trajectory = self.calculate_target_trajectory(
                    base_year=input_data.base_year,
                    base_year_emissions=self._get_emissions_for_scopes(
                        input_data.base_year_emissions,
                        target.scopes_covered
                    ),
                    target_year=target.target_year,
                    pathway=self._determine_pathway_for_ambition(
                        target.reduction_pct,
                        target.target_year - input_data.base_year
                    ),
                )
                target_trajectories.append(trajectory)

                self._track_step(f"target_validation_{target.target_year}", {
                    "target_type": target.target_type.value,
                    "is_valid": validation.is_valid,
                    "ambition_level": validation.ambition_level,
                    "reduction_gap": validation.reduction_gap_pct,
                })

            # Step 3: Validate FLAG requirements if applicable
            flag_validation = None
            if input_data.has_flag_emissions:
                flag_validation = self._validate_flag_requirements(
                    input_data.base_year_emissions,
                    input_data.targets,
                )
                self._track_step("flag_validation", {
                    "flag_compliant": flag_validation.is_flag_compliant,
                    "flag_emissions_pct": flag_validation.flag_emissions_pct,
                })

            # Step 4: Validate net-zero requirements for net-zero targets
            net_zero_validation = None
            net_zero_targets = [t for t in input_data.targets if t.target_type == TargetType.NET_ZERO]
            if net_zero_targets:
                net_zero_validation = self._validate_net_zero_requirements(
                    net_zero_targets[0],
                    input_data.base_year_emissions,
                    target_validations,
                )
                self._track_step("net_zero_validation", {
                    "net_zero_compliant": net_zero_validation.is_net_zero_compliant,
                    "gross_reduction_pct": net_zero_validation.gross_reduction_pct,
                })

            # Step 5: Determine overall validation status
            validation_result = self._determine_overall_validation(
                target_validations,
                scope3_required,
                scope3_coverage,
                input_data,
                flag_validation,
                net_zero_validation,
            )

            self._track_step("overall_validation", {
                "is_valid": validation_result.is_valid,
                "status": validation_result.status.value,
                "sbti_aligned": validation_result.sbti_aligned,
                "net_zero_aligned": validation_result.net_zero_aligned,
            })

            # Step 6: Calculate progress if current data provided
            progress_tracking = None
            if input_data.current_progress:
                progress_tracking = self._calculate_progress(
                    input_data.base_year,
                    input_data.base_year_emissions,
                    input_data.current_progress,
                    input_data.targets[0],  # Use first target for progress
                )
                self._track_step("progress_calculation", {
                    "on_track": progress_tracking.on_track,
                    "reduction_achieved_pct": progress_tracking.reduction_achieved_pct,
                    "progress_status": progress_tracking.progress_status.value,
                })

            # Step 7: Generate recommendations
            recommendations = self._generate_recommendations(
                validation_result,
                target_validations,
                scope3_required,
                scope3_coverage,
                progress_tracking,
                flag_validation,
                net_zero_validation,
            )

            # Step 8: Determine target classification
            target_classification = self._determine_classification(
                validation_result.highest_ambition
            )

            # Step 9: Calculate provenance hash
            provenance_hash = self._calculate_provenance_hash()

            # Step 10: Create output
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            output = SBTiOutput(
                company_id=input_data.company_id,
                validation_id=validation_id,
                validation_result=validation_result,
                target_classification=target_classification,
                pathway_calculations=pathway_calculations,
                target_trajectories=target_trajectories,
                progress_tracking=progress_tracking,
                net_zero_validation=net_zero_validation,
                flag_validation=flag_validation,
                recommendations=recommendations,
                scope3_required=scope3_required,
                scope3_coverage_pct=scope3_coverage,
                provenance_hash=provenance_hash,
                processing_time_ms=round(processing_time, 2),
            )

            logger.info(
                f"SBTi validation complete: valid={validation_result.is_valid}, "
                f"ambition={validation_result.highest_ambition}, "
                f"sbti_aligned={validation_result.sbti_aligned} "
                f"(duration: {processing_time:.2f}ms)"
            )

            return output

        except Exception as e:
            logger.error(f"SBTi validation failed: {str(e)}", exc_info=True)
            raise

    def _check_scope3_requirements(
        self,
        input_data: SBTiInput
    ) -> Tuple[bool, float]:
        """
        Check if Scope 3 target is required.

        ZERO-HALLUCINATION: SBTi requires Scope 3 if >40% of total emissions.

        Args:
            input_data: The SBTi input data

        Returns:
            Tuple of (is_required, coverage_percentage)
        """
        scope3_pct = input_data.base_year_emissions.scope3_percentage

        # SBTi threshold: Scope 3 target required if >40%
        is_required = scope3_pct > self.constants.SCOPE3_THRESHOLD_PCT

        # Calculate coverage from targets
        coverage = 0.0
        for target in input_data.targets:
            if ScopeType.SCOPE_3 in target.scopes_covered:
                # Check if using supplier engagement approach
                if target.scope3_engagement:
                    if target.scope3_engagement.engagement_type == Scope3EngagementType.SUPPLIER_ENGAGEMENT:
                        coverage = target.scope3_engagement.supplier_coverage_pct
                    elif target.scope3_engagement.engagement_type == Scope3EngagementType.ABSOLUTE_REDUCTION:
                        # For absolute reduction, check category coverage
                        if target.scope3_categories_covered:
                            coverage = len(target.scope3_categories_covered) * 6.67
                        else:
                            coverage = 100.0
                else:
                    # Estimate coverage based on categories
                    if target.scope3_categories_covered:
                        coverage = len(target.scope3_categories_covered) * 6.67
                    else:
                        coverage = 100.0
                break

        return is_required, min(coverage, 100.0)

    def _validate_target(
        self,
        target: TargetDefinition,
        base_year: int,
        base_emissions: ScopeEmissions,
        submission_date: Optional[datetime],
    ) -> Tuple[TargetValidation, Optional[PathwayCalculation]]:
        """
        Validate a single target against SBTi requirements.

        ZERO-HALLUCINATION calculations using:
        - ACA: target = base * (1 - rate)^years
        - SDA: target_intensity = I_2050 + (I_base - I_2050) * exp(-k*years)
        - FLAG: sector-specific reduction rates

        Args:
            target: Target definition to validate
            base_year: Emissions base year
            base_emissions: Base year emissions
            submission_date: Target submission date

        Returns:
            Tuple of (TargetValidation, PathwayCalculation)
        """
        messages: List[str] = []

        # Calculate years to target
        years = target.target_year - base_year
        submission_year = submission_date.year if submission_date else datetime.now().year
        years_from_submission = target.target_year - submission_year

        # Validate timeframe
        timeframe_valid = self._validate_timeframe(
            target.target_type,
            years_from_submission,
            target.target_year,
            messages,
        )

        # Validate scope coverage
        scope_coverage_valid = self._validate_scope_coverage(
            target.scopes_covered,
            target.target_type,
            messages,
        )

        # Validate Scope 3 engagement if applicable
        if target.scope3_engagement and ScopeType.SCOPE_3 in target.scopes_covered:
            self._validate_scope3_engagement(target.scope3_engagement, messages)

        # Calculate required reduction based on pathway
        if target.pathway_type == PathwayType.FLAG:
            required_reduction, ambition, pathway_calc = self._calculate_flag_target(
                target,
                years,
                base_emissions,
                messages,
            )
        elif target.pathway_type == PathwayType.ACA:
            required_reduction, ambition, pathway_calc = self._calculate_aca_target(
                target,
                years,
                base_emissions,
                messages,
            )
        else:
            required_reduction, ambition, pathway_calc = self._calculate_sda_target(
                target,
                years,
                base_emissions,
                messages,
            )

        # Special validation for net-zero targets
        net_zero_compliant = False
        if target.target_type == TargetType.NET_ZERO:
            net_zero_compliant = self._check_net_zero_target(target, messages)

        # Special validation for FLAG targets
        flag_compliant = False
        if target.target_type == TargetType.FLAG:
            flag_compliant = self._check_flag_target(target, messages)

        # Determine if target meets requirements
        reduction_gap = target.reduction_pct - required_reduction
        is_valid = (
            target.reduction_pct >= required_reduction
            and timeframe_valid
            and scope_coverage_valid
        )

        # Calculate annual reduction rate
        if years > 0:
            annual_rate = 1 - ((1 - target.reduction_pct / 100) ** (1 / years))
        else:
            annual_rate = 0.0

        # Determine validation status
        if is_valid:
            status = ValidationStatus.VALID
            messages.append(f"Target meets {ambition} pathway requirements")
        elif reduction_gap >= -5.0:  # Within 5% of threshold
            status = ValidationStatus.NEEDS_REVIEW
            messages.append(
                f"Target is {abs(reduction_gap):.1f}% below {ambition} threshold, "
                "may qualify with updated methodology"
            )
        else:
            status = ValidationStatus.INVALID
            messages.append(
                f"Target reduction of {target.reduction_pct:.1f}% is below "
                f"minimum {required_reduction:.1f}% for {ambition} pathway"
            )

        return (
            TargetValidation(
                target_id=target.target_id,
                target_year=target.target_year,
                target_type=target.target_type.value,
                is_valid=is_valid,
                status=status,
                ambition_level=ambition,
                required_reduction_pct=round(required_reduction, 2),
                actual_reduction_pct=round(target.reduction_pct, 2),
                reduction_gap_pct=round(reduction_gap, 2),
                annual_reduction_rate=round(annual_rate * 100, 2),
                timeframe_valid=timeframe_valid,
                scope_coverage_valid=scope_coverage_valid,
                net_zero_compliant=net_zero_compliant,
                flag_compliant=flag_compliant,
                messages=messages,
            ),
            pathway_calc,
        )

    def _validate_timeframe(
        self,
        target_type: TargetType,
        years_from_submission: int,
        target_year: int,
        messages: List[str],
    ) -> bool:
        """
        Validate target timeframe against SBTi requirements.

        Near-term: 5-10 years from submission
        Long-term: 2050 or sooner
        Net-zero: 2050 or sooner (with near-term target)

        Args:
            target_type: Type of target
            years_from_submission: Years from submission to target
            target_year: Target year
            messages: List to append validation messages

        Returns:
            True if timeframe is valid
        """
        if target_type == TargetType.NEAR_TERM:
            if years_from_submission < self.constants.NEAR_TERM_MIN_YEARS:
                messages.append(
                    f"Near-term target must be at least {self.constants.NEAR_TERM_MIN_YEARS} years "
                    f"from submission (got {years_from_submission} years)"
                )
                return False
            if years_from_submission > self.constants.NEAR_TERM_MAX_YEARS:
                messages.append(
                    f"Near-term target must be no more than {self.constants.NEAR_TERM_MAX_YEARS} years "
                    f"from submission (got {years_from_submission} years)"
                )
                return False
            return True

        elif target_type in [TargetType.LONG_TERM, TargetType.NET_ZERO]:
            if target_year > self.constants.LONG_TERM_TARGET_YEAR:
                messages.append(
                    f"Long-term/net-zero target year must be 2050 or sooner "
                    f"(got {target_year})"
                )
                return False
            return True

        elif target_type == TargetType.FLAG:
            # FLAG targets follow same timeframe as regular targets
            if target_year > self.constants.LONG_TERM_TARGET_YEAR:
                messages.append(
                    f"FLAG target year must be 2050 or sooner (got {target_year})"
                )
                return False
            return True

        return True

    def _validate_scope_coverage(
        self,
        scopes_covered: List[ScopeType],
        target_type: TargetType,
        messages: List[str],
    ) -> bool:
        """
        Validate emission scope coverage.

        SBTi requires Scope 1 and Scope 2 for all targets (except FLAG).

        Args:
            scopes_covered: Scopes included in target
            target_type: Type of target
            messages: List to append validation messages

        Returns:
            True if scope coverage is valid
        """
        # FLAG targets don't require Scope 1/2 (they cover land sector)
        if target_type == TargetType.FLAG:
            return True

        # Scope 1 and 2 are required for all other SBTi targets
        if ScopeType.SCOPE_1 not in scopes_covered:
            messages.append("Scope 1 emissions must be included in target")
            return False

        if ScopeType.SCOPE_2 not in scopes_covered:
            messages.append("Scope 2 emissions must be included in target")
            return False

        return True

    def _validate_scope3_engagement(
        self,
        engagement: Scope3EngagementTarget,
        messages: List[str],
    ) -> bool:
        """
        Validate Scope 3 engagement target.

        Options:
        1. 2.5% annual reduction (absolute)
        2. 67% supplier engagement with SBTs

        Args:
            engagement: Scope 3 engagement details
            messages: List to append validation messages

        Returns:
            True if engagement is valid
        """
        if engagement.engagement_type == Scope3EngagementType.SUPPLIER_ENGAGEMENT:
            if engagement.supplier_coverage_pct < self.constants.SCOPE3_SUPPLIER_ENGAGEMENT_PCT:
                messages.append(
                    f"Supplier engagement must cover at least "
                    f"{self.constants.SCOPE3_SUPPLIER_ENGAGEMENT_PCT}% of suppliers "
                    f"(got {engagement.supplier_coverage_pct}%)"
                )
                return False
            messages.append(
                f"Scope 3 supplier engagement target: {engagement.supplier_coverage_pct}% "
                "of suppliers with science-based targets"
            )
            return True

        elif engagement.engagement_type == Scope3EngagementType.ABSOLUTE_REDUCTION:
            # Check minimum annual reduction rate
            if engagement.scope3_reduction_pct < 25.0:  # 2.5% * 10 years
                messages.append(
                    "Scope 3 absolute reduction should achieve at least 25% over 10 years "
                    f"(2.5% annual, got {engagement.scope3_reduction_pct}%)"
                )
                return False
            return True

        return True

    def _calculate_aca_target(
        self,
        target: TargetDefinition,
        years: int,
        base_emissions: ScopeEmissions,
        messages: List[str],
    ) -> Tuple[float, str, PathwayCalculation]:
        """
        Calculate required reduction using Absolute Contraction Approach.

        ZERO-HALLUCINATION CALCULATION:
        target_emissions = base_emissions * (1 - annual_rate) ^ years

        1.5C pathway: 4.2% annual reduction
        WB2C pathway: 2.5% annual reduction
        2C pathway: 1.6% annual reduction

        Scope 3 1.5C pathway: 2.5% annual reduction

        Args:
            target: Target definition
            years: Years from base year to target year
            base_emissions: Base year emissions
            messages: List to append messages

        Returns:
            Tuple of (required_reduction_pct, ambition_level, pathway_calculation)
        """
        calculation_steps: List[Dict[str, Any]] = []

        # Get base emissions for covered scopes
        if ScopeType.SCOPE_3 in target.scopes_covered:
            if len(target.scopes_covered) == 1:
                # Scope 3 only target
                base_value = base_emissions.scope3
                scope_label = "Scope 3"
            else:
                base_value = base_emissions.total
                scope_label = "Scope 1+2+3"
        else:
            base_value = base_emissions.scope12_total
            scope_label = "Scope 1+2"

        calculation_steps.append({
            "step": "base_emissions",
            "description": f"Base year emissions ({scope_label})",
            "value": base_value,
            "unit": "tCO2e",
        })

        # Calculate required reduction for each ambition level
        # ZERO-HALLUCINATION: Fixed formula from SBTi
        # reduction_pct = 1 - (1 - annual_rate)^years

        # Determine if this is Scope 3 only (different rate applies)
        is_scope3_only = (
            len(target.scopes_covered) == 1 and
            ScopeType.SCOPE_3 in target.scopes_covered
        )

        if is_scope3_only:
            # Scope 3 uses 2.5% annual reduction for 1.5C
            rate_1_5c = self.constants.SCOPE3_1_5C_ANNUAL_REDUCTION
        else:
            rate_1_5c = self.constants.ACA_1_5C_ANNUAL_REDUCTION

        reduction_1_5c = (1 - (1 - rate_1_5c) ** years) * 100

        calculation_steps.append({
            "step": "1.5C_calculation",
            "formula": f"reduction = 1 - (1 - {rate_1_5c})^years",
            "annual_rate": rate_1_5c,
            "years": years,
            "result_pct": round(reduction_1_5c, 2),
        })

        # WB2C pathway
        rate_wb2c = self.constants.ACA_WB2C_ANNUAL_REDUCTION
        reduction_wb2c = (1 - (1 - rate_wb2c) ** years) * 100

        calculation_steps.append({
            "step": "WB2C_calculation",
            "formula": "reduction = 1 - (1 - 0.025)^years",
            "annual_rate": rate_wb2c,
            "years": years,
            "result_pct": round(reduction_wb2c, 2),
        })

        # 2C pathway
        rate_2c = self.constants.ACA_2C_ANNUAL_REDUCTION
        reduction_2c = (1 - (1 - rate_2c) ** years) * 100

        calculation_steps.append({
            "step": "2C_calculation",
            "formula": "reduction = 1 - (1 - 0.016)^years",
            "annual_rate": rate_2c,
            "years": years,
            "result_pct": round(reduction_2c, 2),
        })

        # Determine ambition level based on actual reduction
        actual_reduction = target.reduction_pct

        if actual_reduction >= reduction_1_5c:
            ambition = AmbitionLevel.CELSIUS_1_5.value
            required = reduction_1_5c
            annual_rate = rate_1_5c
        elif actual_reduction >= reduction_wb2c:
            ambition = AmbitionLevel.WELL_BELOW_2C.value
            required = reduction_wb2c
            annual_rate = rate_wb2c
        elif actual_reduction >= reduction_2c:
            ambition = AmbitionLevel.CELSIUS_2.value
            required = reduction_2c
            annual_rate = rate_2c
        else:
            ambition = AmbitionLevel.BELOW_THRESHOLD.value
            required = reduction_2c  # Use 2C as minimum threshold
            annual_rate = rate_2c
            messages.append(
                f"Target does not meet minimum SBTi threshold of {reduction_2c:.1f}% "
                f"for {years}-year timeframe"
            )

        # Calculate target year emissions
        target_emissions = base_value * (1 - actual_reduction / 100)

        calculation_steps.append({
            "step": "target_emissions",
            "formula": f"target = {base_value:.2f} * (1 - {actual_reduction/100:.4f})",
            "result": round(target_emissions, 2),
            "unit": "tCO2e",
        })

        pathway_calc = PathwayCalculation(
            pathway_type=PathwayType.ACA.value,
            annual_reduction_rate=annual_rate,
            base_year_value=base_value,
            target_year_value=target_emissions,
            formula_used="target = base * (1 - annual_rate)^years",
            calculation_steps=calculation_steps,
        )

        return required, ambition, pathway_calc

    def _calculate_sda_target(
        self,
        target: TargetDefinition,
        years: int,
        base_emissions: ScopeEmissions,
        messages: List[str],
    ) -> Tuple[float, str, PathwayCalculation]:
        """
        Calculate required reduction using Sectoral Decarbonization Approach.

        ZERO-HALLUCINATION CALCULATION:
        target_intensity = I_2050 + (I_base - I_2050) * exp(-k * years)

        Where:
            I_2050 = Sector 2050 intensity benchmark
            I_base = Base year intensity
            k = Sector-specific decay rate

        Args:
            target: Target definition with intensity data
            years: Years from base year to target year
            base_emissions: Base year emissions
            messages: List to append messages

        Returns:
            Tuple of (required_reduction_pct, ambition_level, pathway_calculation)
        """
        calculation_steps: List[Dict[str, Any]] = []

        # Get sector pathway data
        sector = target.sector or SectorPathway.GENERAL
        sector_data = self.constants.SDA_TARGETS_2050.get(
            sector,
            self.constants.SDA_TARGETS_2050[SectorPathway.GENERAL]
        )

        i_2050 = sector_data["intensity_2050"]
        k = sector_data["decay_rate"]

        calculation_steps.append({
            "step": "sector_parameters",
            "sector": sector.value,
            "intensity_2050": i_2050,
            "decay_rate_k": k,
            "unit": sector_data["unit"],
        })

        # Get base intensity
        if target.base_intensity:
            i_base = target.base_intensity.value
        else:
            # Estimate from emissions/activity (simplified)
            i_base = 1.0  # Default if not provided
            messages.append(
                "Base year intensity not provided, using default value. "
                "Provide actual intensity for accurate SDA validation."
            )

        calculation_steps.append({
            "step": "base_intensity",
            "value": i_base,
            "source": "provided" if target.base_intensity else "default",
        })

        # ZERO-HALLUCINATION CALCULATION
        # SDA formula: I_target = I_2050 + (I_base - I_2050) * exp(-k * years)
        i_target = i_2050 + (i_base - i_2050) * math.exp(-k * years)

        calculation_steps.append({
            "step": "sda_calculation",
            "formula": "I_target = I_2050 + (I_base - I_2050) * exp(-k * years)",
            "I_2050": i_2050,
            "I_base": i_base,
            "k": k,
            "years": years,
            "I_target": round(i_target, 6),
        })

        # Calculate required reduction percentage
        if i_base > 0:
            required_reduction = ((i_base - i_target) / i_base) * 100
        else:
            required_reduction = 0.0
            messages.append("Cannot calculate reduction with zero base intensity")

        calculation_steps.append({
            "step": "reduction_calculation",
            "formula": "reduction = (I_base - I_target) / I_base * 100",
            "result_pct": round(required_reduction, 2),
        })

        # Determine ambition level for SDA
        # SDA is inherently 1.5C or WB2C aligned depending on sector
        if required_reduction >= 42.0:  # Near-term 1.5C threshold
            ambition = AmbitionLevel.CELSIUS_1_5.value
        elif required_reduction >= 25.0:
            ambition = AmbitionLevel.WELL_BELOW_2C.value
        else:
            ambition = AmbitionLevel.CELSIUS_2.value

        # Check if actual target meets requirement
        if target.target_intensity:
            actual_intensity = target.target_intensity.value
            actual_reduction = ((i_base - actual_intensity) / i_base) * 100 if i_base > 0 else 0
        else:
            actual_reduction = target.reduction_pct

        if actual_reduction < required_reduction:
            messages.append(
                f"SDA pathway requires {required_reduction:.1f}% intensity reduction, "
                f"target provides {actual_reduction:.1f}%"
            )

        pathway_calc = PathwayCalculation(
            pathway_type=PathwayType.SDA.value,
            annual_reduction_rate=k,
            base_year_value=i_base,
            target_year_value=i_target,
            formula_used="I_target = I_2050 + (I_base - I_2050) * exp(-k * years)",
            sector_benchmark=i_2050,
            calculation_steps=calculation_steps,
        )

        return required_reduction, ambition, pathway_calc

    def _calculate_flag_target(
        self,
        target: TargetDefinition,
        years: int,
        base_emissions: ScopeEmissions,
        messages: List[str],
    ) -> Tuple[float, str, PathwayCalculation]:
        """
        Calculate required reduction for FLAG (Forest, Land, Agriculture) sector.

        ZERO-HALLUCINATION: Uses SBTi FLAG guidance rates.

        FLAG targets require:
        - 72% reduction by 2050 for 1.5C
        - No deforestation by 2025
        - Separate from non-FLAG targets

        Args:
            target: Target definition with FLAG data
            years: Years from base year to target year
            base_emissions: Base year emissions
            messages: List to append messages

        Returns:
            Tuple of (required_reduction_pct, ambition_level, pathway_calculation)
        """
        calculation_steps: List[Dict[str, Any]] = []

        # Get FLAG emissions
        if target.flag_target:
            base_value = target.flag_target.base_year_flag_emissions
        else:
            base_value = base_emissions.flag_emissions

        calculation_steps.append({
            "step": "flag_base_emissions",
            "value": base_value,
            "unit": "tCO2e",
        })

        # Calculate FLAG pathway reduction
        # FLAG uses different rates than regular emissions
        years_to_2050 = 2050 - (2050 - years)  # Normalize to 2050 baseline

        # ZERO-HALLUCINATION: FLAG reduction requirements
        # 72% reduction by 2050 for 1.5C
        # 50% reduction by 2050 for WB2C
        flag_1_5c_rate = self.constants.FLAG_1_5C_REDUCTION / (2050 - 2020)  # Annual rate
        flag_wb2c_rate = self.constants.FLAG_WB2C_REDUCTION / (2050 - 2020)

        reduction_1_5c = min(flag_1_5c_rate * years, self.constants.FLAG_1_5C_REDUCTION)
        reduction_wb2c = min(flag_wb2c_rate * years, self.constants.FLAG_WB2C_REDUCTION)

        calculation_steps.append({
            "step": "flag_pathway",
            "1_5c_annual_rate": round(flag_1_5c_rate, 2),
            "1_5c_reduction": round(reduction_1_5c, 2),
            "wb2c_reduction": round(reduction_wb2c, 2),
            "years": years,
        })

        # Determine ambition level
        actual_reduction = target.reduction_pct

        if actual_reduction >= reduction_1_5c:
            ambition = AmbitionLevel.CELSIUS_1_5.value
            required = reduction_1_5c
        elif actual_reduction >= reduction_wb2c:
            ambition = AmbitionLevel.WELL_BELOW_2C.value
            required = reduction_wb2c
        else:
            ambition = AmbitionLevel.BELOW_THRESHOLD.value
            required = reduction_wb2c
            messages.append(
                f"FLAG target does not meet minimum {reduction_wb2c:.1f}% reduction"
            )

        # Check no-deforestation commitment
        if target.flag_target:
            if not target.flag_target.no_deforestation_commitment:
                messages.append(
                    "FLAG target requires no-deforestation commitment by 2025"
                )
            if target.flag_target.no_deforestation_date and target.flag_target.no_deforestation_date > 2025:
                messages.append(
                    f"No-deforestation commitment date ({target.flag_target.no_deforestation_date}) "
                    "should be 2025 or earlier"
                )

        target_emissions = base_value * (1 - actual_reduction / 100)

        pathway_calc = PathwayCalculation(
            pathway_type=PathwayType.FLAG.value,
            annual_reduction_rate=flag_1_5c_rate / 100,
            base_year_value=base_value,
            target_year_value=target_emissions,
            formula_used="FLAG linear reduction to 72% by 2050",
            calculation_steps=calculation_steps,
        )

        return required, ambition, pathway_calc

    def _check_net_zero_target(
        self,
        target: TargetDefinition,
        messages: List[str],
    ) -> bool:
        """
        Check if target meets net-zero requirements.

        Net-zero requires:
        - 90% minimum reduction before neutralization
        - Neutralization plan for residual emissions
        - BVCM commitment (recommended)

        Args:
            target: Target definition
            messages: List to append messages

        Returns:
            True if net-zero compliant
        """
        is_compliant = True

        # Check minimum reduction
        if target.reduction_pct < self.constants.NET_ZERO_MIN_REDUCTION:
            messages.append(
                f"Net-zero requires minimum {self.constants.NET_ZERO_MIN_REDUCTION}% "
                f"reduction (got {target.reduction_pct}%)"
            )
            is_compliant = False

        # Check neutralization plan
        if target.neutralization_plan:
            plan = target.neutralization_plan

            # Check residual emissions
            if plan.residual_emissions_pct > self.constants.NET_ZERO_MAX_RESIDUAL:
                messages.append(
                    f"Residual emissions ({plan.residual_emissions_pct}%) exceed "
                    f"maximum {self.constants.NET_ZERO_MAX_RESIDUAL}%"
                )
                is_compliant = False

            # Check removal capacity
            if plan.removal_capacity_tco2e <= 0:
                messages.append("Net-zero requires carbon removal capacity for residual emissions")
                is_compliant = False

            # BVCM is recommended but not required
            if plan.bvcm_commitment:
                messages.append("BVCM commitment present - exceeds minimum requirements")
        else:
            messages.append("Net-zero target requires neutralization plan")
            is_compliant = False

        return is_compliant

    def _check_flag_target(
        self,
        target: TargetDefinition,
        messages: List[str],
    ) -> bool:
        """
        Check if FLAG target meets requirements.

        FLAG requires:
        - No deforestation by 2025
        - No land conversion
        - Separate from non-FLAG targets

        Args:
            target: Target definition
            messages: List to append messages

        Returns:
            True if FLAG compliant
        """
        is_compliant = True

        if not target.flag_target:
            messages.append("FLAG target type requires flag_target details")
            return False

        flag = target.flag_target

        # Check no-deforestation
        if not flag.no_deforestation_commitment:
            messages.append("FLAG requires no-deforestation commitment")
            is_compliant = False
        elif flag.no_deforestation_date and flag.no_deforestation_date > self.constants.FLAG_NO_DEFORESTATION_YEAR:
            messages.append(
                f"No-deforestation date ({flag.no_deforestation_date}) must be "
                f"{self.constants.FLAG_NO_DEFORESTATION_YEAR} or earlier"
            )
            is_compliant = False

        # Check land conversion
        if not flag.land_conversion_commitment:
            messages.append("FLAG requires commitment to no land conversion")
            is_compliant = False

        return is_compliant

    def _validate_flag_requirements(
        self,
        base_emissions: ScopeEmissions,
        targets: List[TargetDefinition],
    ) -> FLAGValidation:
        """
        Validate FLAG sector requirements.

        If FLAG emissions are >20% of total, a separate FLAG target is required.

        Args:
            base_emissions: Base year emissions
            targets: List of targets

        Returns:
            FLAGValidation result
        """
        messages: List[str] = []

        flag_pct = base_emissions.flag_percentage
        separate_required = flag_pct > self.constants.FLAG_THRESHOLD_PCT

        # Check if FLAG target exists
        flag_targets = [t for t in targets if t.target_type == TargetType.FLAG]
        separate_present = len(flag_targets) > 0

        no_deforestation = False
        no_deforestation_by_2025 = False
        land_conversion = False
        reduction_pct = 0.0
        sequestration = False

        if flag_targets:
            ft = flag_targets[0]
            if ft.flag_target:
                no_deforestation = ft.flag_target.no_deforestation_commitment
                no_deforestation_by_2025 = (
                    ft.flag_target.no_deforestation_date is not None and
                    ft.flag_target.no_deforestation_date <= 2025
                )
                land_conversion = ft.flag_target.land_conversion_commitment
                if ft.flag_target.base_year_flag_emissions > 0:
                    reduction_pct = (
                        (ft.flag_target.base_year_flag_emissions - ft.flag_target.target_year_flag_emissions)
                        / ft.flag_target.base_year_flag_emissions * 100
                    )
                sequestration = ft.flag_target.sequestration_target > 0

        # Determine compliance
        is_compliant = True

        if separate_required and not separate_present:
            messages.append(
                f"FLAG emissions are {flag_pct:.1f}% of total. "
                "A separate FLAG target is required."
            )
            is_compliant = False

        if separate_present and not no_deforestation:
            messages.append("FLAG target must include no-deforestation commitment")
            is_compliant = False

        if separate_present and not no_deforestation_by_2025:
            messages.append("No-deforestation commitment must be achieved by 2025")
            is_compliant = False

        return FLAGValidation(
            is_flag_compliant=is_compliant,
            flag_emissions_pct=round(flag_pct, 2),
            separate_target_required=separate_required,
            separate_target_present=separate_present,
            no_deforestation_commitment=no_deforestation,
            no_deforestation_by_2025=no_deforestation_by_2025,
            land_conversion_commitment=land_conversion,
            reduction_pct=round(reduction_pct, 2),
            sequestration_included=sequestration,
            messages=messages,
        )

    def _validate_net_zero_requirements(
        self,
        net_zero_target: TargetDefinition,
        base_emissions: ScopeEmissions,
        target_validations: List[TargetValidation],
    ) -> NetZeroValidation:
        """
        Validate complete net-zero requirements.

        Net-zero standard requires:
        - Near-term target (5-10 years)
        - Long-term target (2050)
        - 90% minimum reduction
        - Neutralization for residuals
        - All scopes covered

        Args:
            net_zero_target: The net-zero target
            base_emissions: Base year emissions
            target_validations: All target validations

        Returns:
            NetZeroValidation result
        """
        messages: List[str] = []

        gross_reduction = net_zero_target.reduction_pct

        # Calculate residual
        residual_pct = 100 - gross_reduction

        # Check neutralization
        neutralization_valid = False
        bvcm = False
        removal_sufficient = False

        if net_zero_target.neutralization_plan:
            plan = net_zero_target.neutralization_plan
            neutralization_valid = plan.residual_emissions_pct <= self.constants.NET_ZERO_MAX_RESIDUAL
            bvcm = plan.bvcm_commitment

            # Check if removal covers residual emissions
            residual_emissions = base_emissions.total * (residual_pct / 100)
            removal_sufficient = plan.removal_capacity_tco2e >= residual_emissions

        # Check for near-term target
        near_term_present = any(
            tv.target_type == TargetType.NEAR_TERM.value and tv.is_valid
            for tv in target_validations
        )

        # Check for long-term target
        long_term_present = any(
            tv.target_type in [TargetType.LONG_TERM.value, TargetType.NET_ZERO.value] and tv.is_valid
            for tv in target_validations
        )

        # Determine overall compliance
        is_compliant = (
            gross_reduction >= self.constants.NET_ZERO_MIN_REDUCTION and
            neutralization_valid and
            near_term_present and
            long_term_present
        )

        if not near_term_present:
            messages.append("Net-zero standard requires a valid near-term target")
        if not neutralization_valid:
            messages.append("Neutralization plan does not meet requirements")
        if not removal_sufficient:
            messages.append("Carbon removal capacity is insufficient for residual emissions")
        if bvcm:
            messages.append("BVCM commitment exceeds minimum net-zero requirements")

        return NetZeroValidation(
            is_net_zero_compliant=is_compliant,
            gross_reduction_pct=round(gross_reduction, 2),
            residual_emissions_pct=round(residual_pct, 2),
            neutralization_valid=neutralization_valid,
            bvcm_commitment=bvcm,
            removal_capacity_sufficient=removal_sufficient,
            near_term_target_present=near_term_present,
            long_term_target_present=long_term_present,
            messages=messages,
        )

    def _determine_overall_validation(
        self,
        target_validations: List[TargetValidation],
        scope3_required: bool,
        scope3_coverage: float,
        input_data: SBTiInput,
        flag_validation: Optional[FLAGValidation],
        net_zero_validation: Optional[NetZeroValidation],
    ) -> ValidationResult:
        """
        Determine overall validation result from individual target validations.

        Args:
            target_validations: List of individual target validations
            scope3_required: Whether Scope 3 target is required
            scope3_coverage: Percentage of Scope 3 covered
            input_data: Original input data
            flag_validation: FLAG validation result
            net_zero_validation: Net-zero validation result

        Returns:
            Overall validation result
        """
        messages: List[str] = []

        # Check all targets valid
        all_valid = all(tv.is_valid for tv in target_validations)

        # Check for near-term and long-term/net-zero targets
        has_near_term = any(
            tv.target_type == TargetType.NEAR_TERM.value
            for tv in target_validations
        )
        has_long_term = any(
            tv.target_type in [TargetType.LONG_TERM.value, TargetType.NET_ZERO.value]
            for tv in target_validations
        )

        # Check Scope 3 compliance
        scope3_compliant = True
        if scope3_required:
            has_scope3_target = any(
                ScopeType.SCOPE_3 in t.scopes_covered
                for t in input_data.targets
            )
            if not has_scope3_target:
                scope3_compliant = False
                messages.append(
                    f"Scope 3 emissions are {input_data.base_year_emissions.scope3_percentage:.1f}% "
                    "of total (>40%). A Scope 3 target is required for SBTi alignment."
                )
            elif scope3_coverage < self.constants.SCOPE3_MIN_COVERAGE_PCT:
                scope3_compliant = False
                messages.append(
                    f"Scope 3 coverage is {scope3_coverage:.1f}% "
                    f"(minimum {self.constants.SCOPE3_MIN_COVERAGE_PCT}% required)"
                )

        # Check FLAG compliance
        flag_compliant = True
        if flag_validation:
            flag_compliant = flag_validation.is_flag_compliant

        # Determine highest ambition
        ambition_order = [
            AmbitionLevel.CELSIUS_1_5.value,
            AmbitionLevel.WELL_BELOW_2C.value,
            AmbitionLevel.CELSIUS_2.value,
            AmbitionLevel.BELOW_THRESHOLD.value,
        ]

        highest_ambition = AmbitionLevel.BELOW_THRESHOLD.value
        for tv in target_validations:
            if tv.is_valid:
                for amb in ambition_order:
                    if tv.ambition_level == amb:
                        if ambition_order.index(amb) < ambition_order.index(highest_ambition):
                            highest_ambition = amb
                        break

        # Determine SBTi alignment
        sbti_aligned = (
            all_valid
            and has_near_term
            and scope3_compliant
            and flag_compliant
            and highest_ambition != AmbitionLevel.BELOW_THRESHOLD.value
        )

        if not has_near_term:
            messages.append("SBTi alignment requires at least one near-term target (5-10 years)")

        # Determine net-zero alignment
        net_zero_aligned = False
        if net_zero_validation:
            net_zero_aligned = net_zero_validation.is_net_zero_compliant
            if net_zero_aligned:
                messages.append(
                    f"Net-zero target of {net_zero_validation.gross_reduction_pct:.1f}% reduction "
                    "meets SBTi Net-Zero Standard requirements"
                )

        # Determine overall status
        if sbti_aligned and all_valid:
            status = ValidationStatus.VALID
        elif any(tv.status == ValidationStatus.NEEDS_REVIEW for tv in target_validations):
            status = ValidationStatus.NEEDS_REVIEW
        else:
            status = ValidationStatus.INVALID

        return ValidationResult(
            is_valid=all_valid and sbti_aligned,
            status=status,
            highest_ambition=highest_ambition,
            target_validations=target_validations,
            overall_messages=messages,
            sbti_aligned=sbti_aligned,
            net_zero_aligned=net_zero_aligned,
            flag_compliant=flag_compliant,
            scope3_compliant=scope3_compliant,
        )

    def _calculate_progress(
        self,
        base_year: int,
        base_emissions: ScopeEmissions,
        current_progress: CurrentProgress,
        target: TargetDefinition,
    ) -> ProgressTracking:
        """
        Calculate progress towards target with detailed gap analysis.

        ZERO-HALLUCINATION: Linear interpolation of expected progress.

        Args:
            base_year: Emissions base year
            base_emissions: Base year emissions
            current_progress: Current year data
            target: Target being tracked

        Returns:
            Progress tracking data with gap analysis
        """
        current_year = current_progress.reporting_year
        years_elapsed = current_year - base_year
        years_to_target = target.target_year - base_year
        years_remaining = target.target_year - current_year

        # Get relevant emissions
        base_value = self._get_emissions_for_scopes(base_emissions, target.scopes_covered)
        current_value = self._get_emissions_for_scopes(
            current_progress.current_emissions,
            target.scopes_covered
        )

        # Calculate target emissions
        target_value = base_value * (1 - target.reduction_pct / 100)

        # Calculate achieved reduction
        # ZERO-HALLUCINATION: reduction = (base - current) / base * 100
        if base_value > 0:
            reduction_achieved = ((base_value - current_value) / base_value) * 100
        else:
            reduction_achieved = 0.0

        # Calculate expected reduction at this point (linear trajectory)
        if years_to_target > 0:
            expected_reduction_now = (target.reduction_pct / years_to_target) * years_elapsed
        else:
            expected_reduction_now = target.reduction_pct

        # Calculate gap to trajectory
        gap_to_trajectory = reduction_achieved - expected_reduction_now

        # Determine progress status
        if gap_to_trajectory >= 5.0:
            progress_status = ProgressStatus.AHEAD
            on_track = True
        elif gap_to_trajectory >= -2.0:
            progress_status = ProgressStatus.ON_TRACK
            on_track = True
        elif gap_to_trajectory >= -10.0:
            progress_status = ProgressStatus.SLIGHTLY_BEHIND
            on_track = False
        elif gap_to_trajectory >= -20.0:
            progress_status = ProgressStatus.SIGNIFICANTLY_BEHIND
            on_track = False
        else:
            progress_status = ProgressStatus.AT_RISK
            on_track = False

        # Calculate expected reduction at current pace
        if years_elapsed > 0:
            annual_reduction_rate = reduction_achieved / years_elapsed
            expected_at_target = annual_reduction_rate * years_to_target
        else:
            annual_reduction_rate = 0.0
            expected_at_target = 0.0

        # Calculate annual reduction needed to meet target
        remaining_reduction = target.reduction_pct - reduction_achieved
        if years_remaining > 0 and current_value > 0:
            annual_needed = 1 - ((1 - remaining_reduction / 100) ** (1 / years_remaining))
            annual_needed_pct = annual_needed * 100
        else:
            annual_needed_pct = 0.0

        # Generate trajectory data points
        trajectory_data: List[Dict[str, float]] = []
        for year in range(base_year, target.target_year + 1):
            years_from_base = year - base_year

            # Expected pathway (linear)
            expected = base_value * (1 - (target.reduction_pct / 100) * (years_from_base / years_to_target))

            data_point: Dict[str, float] = {
                "year": float(year),
                "expected_emissions": round(expected, 2),
            }

            # Actual data points
            if year == base_year:
                data_point["actual_emissions"] = round(base_value, 2)
            elif year == current_year:
                data_point["actual_emissions"] = round(current_value, 2)

            trajectory_data.append(data_point)

        # Gap analysis
        gap_analysis = {
            "current_vs_expected_gap_pct": round(gap_to_trajectory, 2),
            "emissions_gap_tco2e": round(current_value - (base_value * (1 - expected_reduction_now / 100)), 2),
            "acceleration_required": not on_track,
            "catch_up_annual_rate_pct": round(annual_needed_pct, 2) if not on_track else 0.0,
            "risk_level": progress_status.value,
        }

        return ProgressTracking(
            base_year=base_year,
            current_year=current_year,
            target_year=target.target_year,
            years_elapsed=years_elapsed,
            years_remaining=years_remaining,
            base_emissions_tco2e=round(base_value, 2),
            current_emissions_tco2e=round(current_value, 2),
            target_emissions_tco2e=round(target_value, 2),
            reduction_achieved_pct=round(reduction_achieved, 2),
            reduction_required_pct=round(target.reduction_pct, 2),
            expected_reduction_by_now_pct=round(expected_reduction_now, 2),
            on_track=on_track,
            progress_status=progress_status,
            gap_to_trajectory_pct=round(gap_to_trajectory, 2),
            expected_reduction_pct=round(expected_at_target, 2),
            annual_reduction_needed_pct=round(annual_needed_pct, 2),
            trajectory_chart_data=trajectory_data,
            gap_analysis=gap_analysis,
        )

    def _generate_recommendations(
        self,
        validation_result: ValidationResult,
        target_validations: List[TargetValidation],
        scope3_required: bool,
        scope3_coverage: float,
        progress: Optional[ProgressTracking],
        flag_validation: Optional[FLAGValidation],
        net_zero_validation: Optional[NetZeroValidation],
    ) -> List[Recommendation]:
        """
        Generate improvement recommendations based on validation results.

        Args:
            validation_result: Overall validation result
            target_validations: Individual target validations
            scope3_required: Whether Scope 3 target required
            scope3_coverage: Current Scope 3 coverage
            progress: Progress tracking data
            flag_validation: FLAG validation result
            net_zero_validation: Net-zero validation result

        Returns:
            List of prioritized recommendations
        """
        recommendations: List[Recommendation] = []

        # Recommendation 1: Increase ambition if below 1.5C
        if validation_result.highest_ambition != AmbitionLevel.CELSIUS_1_5.value:
            for tv in target_validations:
                if tv.ambition_level != AmbitionLevel.CELSIUS_1_5.value:
                    gap = tv.required_reduction_pct - tv.actual_reduction_pct
                    if gap > 0:
                        recommendations.append(Recommendation(
                            priority="high",
                            category="ambition",
                            title=f"Increase {tv.target_year} target ambition to 1.5C pathway",
                            description=(
                                f"Current target of {tv.actual_reduction_pct:.1f}% reduction "
                                f"is {abs(tv.reduction_gap_pct):.1f}% below 1.5C pathway. "
                                f"Consider increasing to {tv.required_reduction_pct:.1f}% "
                                "to align with Paris Agreement goals."
                            ),
                            impact="Achieve 1.5C alignment and leadership positioning",
                            implementation_steps=[
                                "Review decarbonization levers across value chain",
                                "Assess renewable energy procurement options",
                                "Evaluate energy efficiency investments",
                                "Consider science-based supplier engagement",
                                "Update target and resubmit to SBTi",
                            ],
                        ))
                    break

        # Recommendation 2: Add Scope 3 target if required
        if scope3_required and scope3_coverage < 67.0:
            recommendations.append(Recommendation(
                priority="high",
                category="scope_coverage",
                title="Add or expand Scope 3 emissions target",
                description=(
                    f"Scope 3 emissions represent a significant portion of total footprint. "
                    f"Current coverage is {scope3_coverage:.1f}% (minimum 67% required). "
                    "Options: 2.5% annual reduction OR 67% supplier engagement with SBTs."
                ),
                impact="Complete SBTi alignment and address full value chain",
                implementation_steps=[
                    "Complete Scope 3 screening across all 15 categories",
                    "Identify top 3-5 material categories",
                    "Choose approach: absolute reduction or supplier engagement",
                    "Set reduction targets for material categories",
                    "Engage suppliers on emission reduction or SBT adoption",
                ],
            ))

        # Recommendation 3: Add near-term target if missing
        has_near_term = any(
            tv.target_type == TargetType.NEAR_TERM.value
            for tv in target_validations
        )
        if not has_near_term:
            recommendations.append(Recommendation(
                priority="high",
                category="target_structure",
                title="Add near-term target (5-10 years)",
                description=(
                    "SBTi requires at least one near-term target. "
                    "Near-term targets ensure immediate action and accountability. "
                    "For 1.5C: 4.2% annual reduction (42% by 2030 from 2020 base)."
                ),
                impact="Meet SBTi minimum requirements for target structure",
                implementation_steps=[
                    "Calculate 2030 target using 4.2% annual reduction",
                    "Define scope coverage (minimum Scope 1+2)",
                    "Document reduction pathway and key initiatives",
                    "Submit near-term target to SBTi for validation",
                ],
            ))

        # Recommendation 4: Add net-zero target if only short-term
        has_long_term = any(
            tv.target_type in [TargetType.LONG_TERM.value, TargetType.NET_ZERO.value]
            for tv in target_validations
        )
        if not has_long_term and has_near_term:
            recommendations.append(Recommendation(
                priority="medium",
                category="target_structure",
                title="Add long-term net-zero target (2050)",
                description=(
                    "Consider adding a net-zero target to demonstrate "
                    "long-term commitment to full decarbonization. "
                    "Net-zero requires 90%+ reduction plus neutralization of residuals."
                ),
                impact="Achieve net-zero alignment and stakeholder confidence",
                implementation_steps=[
                    "Develop 2050 decarbonization roadmap",
                    "Identify residual emissions and neutralization strategy",
                    "Set 90%+ reduction target for 2050",
                    "Develop carbon removal portfolio for residuals",
                    "Consider BVCM commitment for additional impact",
                    "Submit to SBTi Net-Zero Standard",
                ],
            ))

        # Recommendation 5: Accelerate if behind on progress
        if progress and not progress.on_track:
            recommendations.append(Recommendation(
                priority="high",
                category="progress",
                title="Accelerate emission reduction to meet target",
                description=(
                    f"Current progress ({progress.reduction_achieved_pct:.1f}%) is behind "
                    f"the expected trajectory ({progress.expected_reduction_by_now_pct:.1f}%). "
                    f"Need {progress.annual_reduction_needed_pct:.1f}% annual reduction to meet target."
                ),
                impact=f"Get back on track for {progress.reduction_required_pct:.1f}% target",
                implementation_steps=[
                    "Review and update emission reduction roadmap",
                    "Accelerate renewable energy transition",
                    "Implement additional efficiency measures",
                    "Consider power purchase agreements (PPAs)",
                    "Evaluate electrification opportunities",
                    "Increase supplier engagement intensity",
                ],
            ))

        # Recommendation 6: FLAG target if required
        if flag_validation and not flag_validation.is_flag_compliant:
            if flag_validation.separate_target_required and not flag_validation.separate_target_present:
                recommendations.append(Recommendation(
                    priority="high",
                    category="flag",
                    title="Add separate FLAG sector target",
                    description=(
                        f"FLAG emissions represent {flag_validation.flag_emissions_pct:.1f}% "
                        "of total emissions. SBTi requires a separate FLAG target "
                        "including no-deforestation commitment by 2025."
                    ),
                    impact="Meet SBTi FLAG requirements and address land sector emissions",
                    implementation_steps=[
                        "Quantify FLAG emissions by category",
                        "Set FLAG reduction target (72% by 2050 for 1.5C)",
                        "Commit to no-deforestation by 2025",
                        "Commit to no land conversion",
                        "Develop forest conservation and restoration strategy",
                    ],
                ))

        # Recommendation 7: Net-zero neutralization if missing
        if net_zero_validation and not net_zero_validation.neutralization_valid:
            recommendations.append(Recommendation(
                priority="medium",
                category="net_zero",
                title="Develop carbon removal portfolio for net-zero",
                description=(
                    "Net-zero target requires neutralization of residual emissions "
                    f"({net_zero_validation.residual_emissions_pct:.1f}%). "
                    "Develop a portfolio of high-quality carbon removals."
                ),
                impact="Complete net-zero compliance with credible neutralization",
                implementation_steps=[
                    "Quantify projected residual emissions at 2050",
                    "Assess nature-based removal options (reforestation, soil carbon)",
                    "Evaluate technology-based options (DAC, BECCS)",
                    "Develop phased carbon removal procurement strategy",
                    "Consider BVCM investment for near-term impact",
                ],
            ))

        return recommendations

    def _determine_classification(self, highest_ambition: str) -> str:
        """
        Determine target classification label.

        Args:
            highest_ambition: Highest ambition level achieved

        Returns:
            Classification string
        """
        classification_map = {
            AmbitionLevel.CELSIUS_1_5.value: "1.5C-aligned",
            AmbitionLevel.WELL_BELOW_2C.value: "Well-Below-2C",
            AmbitionLevel.CELSIUS_2.value: "2C-aligned",
            AmbitionLevel.BELOW_THRESHOLD.value: "Below SBTi Threshold",
        }
        return classification_map.get(highest_ambition, "Unclassified")

    def _get_emissions_for_scopes(
        self,
        emissions: ScopeEmissions,
        scopes: List[ScopeType],
    ) -> float:
        """Get total emissions for specified scopes."""
        total = 0.0
        if ScopeType.SCOPE_1 in scopes:
            total += emissions.scope1
        if ScopeType.SCOPE_2 in scopes:
            total += emissions.scope2
        if ScopeType.SCOPE_3 in scopes:
            total += emissions.scope3
        return total

    def _determine_pathway_for_ambition(
        self,
        reduction_pct: float,
        years: int,
    ) -> str:
        """Determine pathway based on reduction and timeframe."""
        rate_1_5c = self.constants.ACA_1_5C_ANNUAL_REDUCTION
        required_1_5c = (1 - (1 - rate_1_5c) ** years) * 100

        rate_wb2c = self.constants.ACA_WB2C_ANNUAL_REDUCTION
        required_wb2c = (1 - (1 - rate_wb2c) ** years) * 100

        if reduction_pct >= required_1_5c:
            return "1.5C"
        elif reduction_pct >= required_wb2c:
            return "WB2C"
        else:
            return "2C"

    def _track_step(self, step_type: str, data: Dict[str, Any]) -> None:
        """Track a processing step for provenance."""
        self._provenance_steps.append({
            "step_type": step_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
        })

    def _calculate_provenance_hash(self) -> str:
        """
        Calculate SHA-256 hash of complete provenance chain.

        This hash enables:
        - Verification that validation was deterministic
        - Audit trail for regulatory compliance
        - Reproducibility checking
        """
        provenance_data = {
            "agent_id": self.AGENT_ID,
            "version": self.VERSION,
            "steps": self._provenance_steps,
            "timestamp": datetime.utcnow().isoformat(),
        }

        json_str = json.dumps(provenance_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    # =========================================================================
    # PUBLIC API METHODS
    # =========================================================================

    def calculate_target_trajectory(
        self,
        base_year: int,
        base_year_emissions: float,
        target_year: int,
        pathway: str = "1.5C",
    ) -> TargetTrajectory:
        """
        Calculate year-by-year target trajectory.

        ZERO-HALLUCINATION: Uses SBTi-defined annual reduction rates.

        Args:
            base_year: Emissions base year
            base_year_emissions: Base year emissions in tCO2e
            target_year: Target year
            pathway: Climate pathway ("1.5C", "WB2C", "2C")

        Returns:
            TargetTrajectory with year-by-year projections
        """
        years = target_year - base_year

        # Get annual reduction rate for pathway
        rate_map = {
            "1.5C": self.constants.ACA_1_5C_ANNUAL_REDUCTION,
            "WB2C": self.constants.ACA_WB2C_ANNUAL_REDUCTION,
            "2C": self.constants.ACA_2C_ANNUAL_REDUCTION,
        }
        annual_rate = rate_map.get(pathway, self.constants.ACA_1_5C_ANNUAL_REDUCTION)

        # Calculate total reduction
        total_reduction = (1 - (1 - annual_rate) ** years) * 100
        target_emissions = base_year_emissions * (1 - total_reduction / 100)

        # Generate year-by-year trajectory
        trajectory_points: List[TargetTrajectoryPoint] = []

        for year in range(base_year, target_year + 1):
            years_from_base = year - base_year

            # Calculate cumulative reduction
            cumulative_reduction = (1 - (1 - annual_rate) ** years_from_base) * 100
            year_emissions = base_year_emissions * (1 - cumulative_reduction / 100)

            # Calculate annual rate to next year
            if year < target_year:
                next_year_emissions = base_year_emissions * (1 - (1 - annual_rate) ** (years_from_base + 1))
                annual_reduction_rate = ((year_emissions - next_year_emissions) / year_emissions) * 100 if year_emissions > 0 else 0
            else:
                annual_reduction_rate = 0.0

            trajectory_points.append(TargetTrajectoryPoint(
                year=year,
                target_emissions=round(year_emissions, 2),
                cumulative_reduction_pct=round(cumulative_reduction, 2),
                annual_reduction_rate=round(annual_reduction_rate, 2),
            ))

        return TargetTrajectory(
            base_year=base_year,
            target_year=target_year,
            base_emissions=round(base_year_emissions, 2),
            target_emissions=round(target_emissions, 2),
            total_reduction_pct=round(total_reduction, 2),
            annual_reduction_rate=round(annual_rate * 100, 2),
            pathway=pathway,
            trajectory_points=trajectory_points,
        )

    def validate_target(
        self,
        target_year: int,
        reduction_pct: float,
        base_year: int = 2019,
        target_type: TargetType = TargetType.NEAR_TERM,
    ) -> Dict[str, Any]:
        """
        Quick validation of a single target.

        Public API for simple target validation without full input.

        Args:
            target_year: Target year
            reduction_pct: Reduction percentage
            base_year: Emissions base year
            target_type: Type of target

        Returns:
            Dictionary with validation result
        """
        years = target_year - base_year

        # Calculate required reduction for 1.5C
        rate = self.constants.ACA_1_5C_ANNUAL_REDUCTION
        required = (1 - (1 - rate) ** years) * 100

        is_valid = reduction_pct >= required
        gap = reduction_pct - required

        return {
            "is_valid": is_valid,
            "required_reduction_pct": round(required, 2),
            "actual_reduction_pct": round(reduction_pct, 2),
            "gap_pct": round(gap, 2),
            "ambition_level": AmbitionLevel.CELSIUS_1_5.value if is_valid else AmbitionLevel.BELOW_THRESHOLD.value,
            "pathway": "ACA",
            "years": years,
            "annual_rate_required": round(rate * 100, 2),
        }

    def calculate_aca_target(
        self,
        base_year: int,
        target_year: int,
        ambition: AmbitionLevel = AmbitionLevel.CELSIUS_1_5,
    ) -> Dict[str, float]:
        """
        Calculate required ACA target for given timeframe.

        ZERO-HALLUCINATION: Uses SBTi-defined annual reduction rates.

        Args:
            base_year: Emissions base year
            target_year: Target year
            ambition: Ambition level

        Returns:
            Dictionary with target details
        """
        years = target_year - base_year

        rate_map = {
            AmbitionLevel.CELSIUS_1_5: self.constants.ACA_1_5C_ANNUAL_REDUCTION,
            AmbitionLevel.WELL_BELOW_2C: self.constants.ACA_WB2C_ANNUAL_REDUCTION,
            AmbitionLevel.CELSIUS_2: self.constants.ACA_2C_ANNUAL_REDUCTION,
        }

        rate = rate_map.get(ambition, self.constants.ACA_1_5C_ANNUAL_REDUCTION)
        reduction = (1 - (1 - rate) ** years) * 100

        return {
            "base_year": base_year,
            "target_year": target_year,
            "years": years,
            "annual_reduction_rate": rate,
            "annual_reduction_rate_pct": round(rate * 100, 2),
            "total_reduction_pct": round(reduction, 2),
            "ambition_level": ambition.value,
        }

    def calculate_sda_intensity(
        self,
        base_intensity: float,
        sector: SectorPathway,
        target_year: int,
        base_year: int = 2019,
    ) -> Dict[str, float]:
        """
        Calculate SDA target intensity for a sector.

        ZERO-HALLUCINATION: Uses SBTi sector pathway data.

        Args:
            base_intensity: Base year intensity
            sector: Sector pathway
            target_year: Target year
            base_year: Base year

        Returns:
            Dictionary with intensity calculation
        """
        years = target_year - base_year
        sector_data = self.constants.SDA_TARGETS_2050.get(
            sector,
            self.constants.SDA_TARGETS_2050[SectorPathway.GENERAL]
        )

        i_2050 = sector_data["intensity_2050"]
        k = sector_data["decay_rate"]

        # SDA formula
        target_intensity = i_2050 + (base_intensity - i_2050) * math.exp(-k * years)

        reduction_pct = ((base_intensity - target_intensity) / base_intensity * 100) if base_intensity > 0 else 0

        return {
            "base_year": base_year,
            "target_year": target_year,
            "base_intensity": base_intensity,
            "target_intensity": round(target_intensity, 6),
            "sector_2050_benchmark": i_2050,
            "reduction_pct": round(reduction_pct, 2),
            "sector": sector.value,
            "unit": sector_data["unit"],
        }

    def calculate_scope3_requirements(
        self,
        total_emissions: float,
        scope3_emissions: float,
    ) -> Dict[str, Any]:
        """
        Calculate Scope 3 target requirements.

        Args:
            total_emissions: Total emissions across all scopes
            scope3_emissions: Scope 3 emissions

        Returns:
            Dictionary with Scope 3 requirements
        """
        scope3_pct = (scope3_emissions / total_emissions * 100) if total_emissions > 0 else 0

        return {
            "scope3_percentage": round(scope3_pct, 2),
            "scope3_target_required": scope3_pct > self.constants.SCOPE3_THRESHOLD_PCT,
            "threshold_pct": self.constants.SCOPE3_THRESHOLD_PCT,
            "options": {
                "absolute_reduction": {
                    "annual_rate_pct": self.constants.SCOPE3_1_5C_ANNUAL_REDUCTION * 100,
                    "description": "2.5% annual reduction in Scope 3 emissions",
                },
                "supplier_engagement": {
                    "coverage_required_pct": self.constants.SCOPE3_SUPPLIER_ENGAGEMENT_PCT,
                    "description": "67% of suppliers (by emissions) have SBTs",
                },
            },
            "minimum_coverage_pct": self.constants.SCOPE3_MIN_COVERAGE_PCT,
        }

    def get_pathway_constants(self) -> Dict[str, Any]:
        """Get SBTi pathway constants for reference."""
        return {
            "aca_1_5c_annual_rate": self.constants.ACA_1_5C_ANNUAL_REDUCTION,
            "aca_wb2c_annual_rate": self.constants.ACA_WB2C_ANNUAL_REDUCTION,
            "aca_2c_annual_rate": self.constants.ACA_2C_ANNUAL_REDUCTION,
            "scope3_1_5c_annual_rate": self.constants.SCOPE3_1_5C_ANNUAL_REDUCTION,
            "near_term_min_years": self.constants.NEAR_TERM_MIN_YEARS,
            "near_term_max_years": self.constants.NEAR_TERM_MAX_YEARS,
            "scope3_threshold_pct": self.constants.SCOPE3_THRESHOLD_PCT,
            "scope3_min_coverage_pct": self.constants.SCOPE3_MIN_COVERAGE_PCT,
            "scope3_supplier_engagement_pct": self.constants.SCOPE3_SUPPLIER_ENGAGEMENT_PCT,
            "net_zero_min_reduction": self.constants.NET_ZERO_MIN_REDUCTION,
            "net_zero_max_residual": self.constants.NET_ZERO_MAX_RESIDUAL,
            "long_term_target_year": self.constants.LONG_TERM_TARGET_YEAR,
            "flag_threshold_pct": self.constants.FLAG_THRESHOLD_PCT,
            "flag_1_5c_reduction": self.constants.FLAG_1_5C_REDUCTION,
            "flag_no_deforestation_year": self.constants.FLAG_NO_DEFORESTATION_YEAR,
        }

    def get_supported_sectors(self) -> List[str]:
        """Get list of sectors with SDA pathway support."""
        return [sector.value for sector in SectorPathway]


# =============================================================================
# PACK SPECIFICATION
# =============================================================================


PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "targets/sbti_validation_v1",
    "name": "SBTi Validation Agent",
    "version": "2.0.0",
    "summary": "Validate corporate targets against SBTi Corporate Net-Zero Standard",
    "tags": [
        "sbti",
        "science-based-targets",
        "net-zero",
        "decarbonization",
        "climate-targets",
        "ghg-protocol",
        "flag",
        "1.5c-pathway",
    ],
    "owners": ["targets-team"],
    "compute": {
        "entrypoint": "python://agents.gl_010_sbti_validation.agent:SBTiValidationAgent",
        "deterministic": True,
    },
    "factors": [
        {"ref": "std://sbti/corporate-net-zero/v1.2"},
        {"ref": "std://sbti/corporate-manual/2024"},
        {"ref": "std://sbti/flag-guidance/2022"},
        {"ref": "std://ghg-protocol/corporate/2015"},
    ],
    "provenance": {
        "methodology": "SBTi Corporate Net-Zero Standard v1.2",
        "aca_source": "SBTi Corporate Manual 2024",
        "sda_source": "SBTi Sector-Specific Guidance",
        "flag_source": "SBTi FLAG Guidance 2022",
        "enable_audit": True,
    },
    "pathways": {
        "ACA": "Absolute Contraction Approach",
        "SDA": "Sectoral Decarbonization Approach",
        "FLAG": "Forest, Land, and Agriculture Pathway",
    },
    "supported_sectors": [
        "power_generation",
        "steel",
        "cement",
        "aluminum",
        "transport_road",
        "transport_aviation",
        "transport_shipping",
        "buildings",
        "chemicals",
        "oil_gas",
        "flag_agriculture",
        "flag_forestry",
        "flag_land_use",
    ],
}
