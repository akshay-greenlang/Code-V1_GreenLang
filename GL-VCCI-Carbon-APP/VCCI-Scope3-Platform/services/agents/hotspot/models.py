"""
HotspotAnalysisAgent Data Models
GL-VCCI Scope 3 Platform

Pydantic models for hotspot analysis inputs, outputs, and scenarios.

Version: 1.0.0
Phase: 3 (Weeks 14-16)
Date: 2025-10-30
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

from .config import ScenarioType, InsightPriority, InsightType, AnalysisDimension


# ============================================================================
# INPUT MODELS
# ============================================================================

class EmissionRecord(BaseModel):
    """Standard emission record for analysis."""

    record_id: Optional[str] = Field(None, description="Unique record identifier")

    # Emissions
    emissions_tco2e: float = Field(..., ge=0, description="Emissions in tCO2e")
    emissions_kgco2e: float = Field(..., ge=0, description="Emissions in kgCO2e")

    # Dimensions
    supplier_name: Optional[str] = Field(None, description="Supplier name")
    scope3_category: Optional[int] = Field(None, ge=1, le=15, description="Scope 3 category")
    product_name: Optional[str] = Field(None, description="Product name")
    region: Optional[str] = Field(None, description="Region/country")
    facility_name: Optional[str] = Field(None, description="Facility name")

    # Quality
    dqi_score: Optional[float] = Field(None, ge=0, le=100, description="DQI score")
    tier: Optional[int] = Field(None, ge=1, le=3, description="Data tier")
    uncertainty_pct: Optional[float] = Field(None, ge=0, description="Uncertainty %")

    # Financial
    spend_usd: Optional[float] = Field(None, ge=0, description="Spend in USD")

    # Temporal
    calculation_date: Optional[datetime] = Field(None, description="Calculation date")
    time_period: Optional[str] = Field(None, description="Time period (YYYY-MM)")

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# ============================================================================
# PARETO ANALYSIS MODELS
# ============================================================================

class ParetoItem(BaseModel):
    """Individual item in Pareto analysis."""

    rank: int = Field(..., ge=1, description="Rank by emissions")
    entity_name: str = Field(..., description="Entity name (supplier, category, etc)")
    emissions_tco2e: float = Field(..., ge=0, description="Emissions in tCO2e")
    percent_of_total: float = Field(..., ge=0, le=100, description="Percentage of total")
    cumulative_percent: float = Field(..., ge=0, le=100, description="Cumulative percentage")


class ParetoAnalysis(BaseModel):
    """Pareto analysis result (80/20 rule)."""

    dimension: str = Field(..., description="Analysis dimension")
    total_emissions_tco2e: float = Field(..., ge=0, description="Total emissions")
    total_entities: int = Field(..., ge=0, description="Total number of entities")

    # Top contributors
    top_20_percent: List[ParetoItem] = Field(..., description="Top 20% contributors")
    n_entities_in_top_20: int = Field(..., ge=0, description="Number in top 20%")

    # Pareto metrics
    pareto_threshold: float = Field(..., description="Pareto threshold (e.g., 0.80)")
    pareto_efficiency: float = Field(..., description="Actual cumulative at 20% mark")
    pareto_achieved: bool = Field(..., description="Whether 80/20 rule is achieved")

    # Visualization data
    chart_data: Dict[str, Any] = Field(..., description="Data for Pareto chart")

    # Metadata
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# SEGMENTATION MODELS
# ============================================================================

class Segment(BaseModel):
    """Individual segment in multi-dimensional analysis."""

    segment_name: str = Field(..., description="Segment name")
    emissions_tco2e: float = Field(..., ge=0, description="Segment emissions")
    percent_of_total: float = Field(..., ge=0, le=100, description="Percentage of total")
    record_count: int = Field(..., ge=0, description="Number of records")

    # Quality metrics
    avg_dqi_score: Optional[float] = Field(None, description="Average DQI score")
    avg_uncertainty_pct: Optional[float] = Field(None, description="Average uncertainty")

    # Financial
    total_spend_usd: Optional[float] = Field(None, description="Total spend")

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SegmentationAnalysis(BaseModel):
    """Multi-dimensional segmentation analysis result."""

    dimension: AnalysisDimension = Field(..., description="Analysis dimension")
    total_emissions_tco2e: float = Field(..., ge=0, description="Total emissions")
    total_records: int = Field(..., ge=0, description="Total records")

    # Segments
    segments: List[Segment] = Field(..., description="Segments")
    top_10_segments: List[Segment] = Field(..., description="Top 10 segments by emissions")

    # Summary statistics
    n_segments: int = Field(..., ge=0, description="Number of segments")
    top_3_concentration: float = Field(..., ge=0, le=100, description="Top 3 concentration %")

    # Visualization data
    chart_data: Dict[str, Any] = Field(..., description="Data for charts")

    # Metadata
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# SCENARIO MODELS
# ============================================================================

class BaseScenario(BaseModel):
    """Base scenario model."""

    scenario_id: Optional[str] = Field(None, description="Scenario identifier")
    scenario_type: ScenarioType = Field(..., description="Scenario type")
    name: str = Field(..., description="Scenario name")
    description: Optional[str] = Field(None, description="Scenario description")

    # Estimated impact
    estimated_reduction_tco2e: float = Field(..., ge=0, description="Estimated reduction")
    estimated_cost_usd: float = Field(..., description="Estimated cost (negative = savings)")

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SupplierSwitchScenario(BaseScenario):
    """Supplier switching scenario."""

    scenario_type: ScenarioType = Field(default=ScenarioType.SUPPLIER_SWITCH, frozen=True)

    from_supplier: str = Field(..., description="Current supplier")
    to_supplier: str = Field(..., description="New supplier")
    products: List[str] = Field(..., description="Products to switch")

    # Impact details
    current_emissions_tco2e: float = Field(..., ge=0)
    new_emissions_tco2e: float = Field(..., ge=0)


class ModalShiftScenario(BaseScenario):
    """Transport modal shift scenario."""

    scenario_type: ScenarioType = Field(default=ScenarioType.MODAL_SHIFT, frozen=True)

    from_mode: str = Field(..., description="Current transport mode")
    to_mode: str = Field(..., description="New transport mode")
    routes: List[str] = Field(..., description="Routes affected")
    volume_pct: float = Field(..., ge=0, le=100, description="% of volume to shift")


class ProductSubstitutionScenario(BaseScenario):
    """Product substitution scenario."""

    scenario_type: ScenarioType = Field(default=ScenarioType.PRODUCT_SUBSTITUTION, frozen=True)

    from_product: str = Field(..., description="Current product")
    to_product: str = Field(..., description="Substitute product")
    volume_tonnes: float = Field(..., ge=0, description="Volume to substitute")

    # Emission factors
    current_ef_kgco2e_per_tonne: float = Field(..., ge=0)
    new_ef_kgco2e_per_tonne: float = Field(..., ge=0)


class ScenarioResult(BaseModel):
    """Result of scenario modeling."""

    scenario: BaseScenario = Field(..., description="Scenario configuration")

    # Impact
    baseline_emissions_tco2e: float = Field(..., ge=0, description="Baseline emissions")
    projected_emissions_tco2e: float = Field(..., ge=0, description="Projected emissions")
    reduction_tco2e: float = Field(..., ge=0, description="Absolute reduction")
    reduction_percent: float = Field(..., ge=0, le=100, description="Reduction percentage")

    # Economics
    implementation_cost_usd: float = Field(..., description="Implementation cost")
    annual_savings_usd: float = Field(default=0.0, description="Annual operational savings")

    # ROI metrics
    roi_usd_per_tco2e: float = Field(..., description="ROI (USD per tCO2e)")
    payback_period_years: Optional[float] = Field(None, description="Payback period")

    # Feasibility
    feasibility_score: Optional[float] = Field(None, ge=0, le=100, description="Feasibility score")
    risks: List[str] = Field(default_factory=list, description="Identified risks")
    assumptions: List[str] = Field(default_factory=list, description="Key assumptions")

    # Metadata
    modeled_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# ROI MODELS
# ============================================================================

class Initiative(BaseModel):
    """Emission reduction initiative for ROI analysis."""

    initiative_id: Optional[str] = Field(None, description="Initiative identifier")
    name: str = Field(..., description="Initiative name")
    description: Optional[str] = Field(None, description="Description")

    # Impact
    reduction_potential_tco2e: float = Field(..., ge=0, description="Reduction potential")

    # Costs
    implementation_cost_usd: float = Field(..., ge=0, description="Implementation cost")
    annual_operating_cost_usd: float = Field(default=0.0, description="Annual operating cost")
    annual_savings_usd: float = Field(default=0.0, ge=0, description="Annual savings")

    # Timeline
    implementation_period_months: int = Field(default=12, ge=1, description="Implementation period")

    # Metadata
    category: Optional[str] = Field(None, description="Initiative category")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ROIAnalysis(BaseModel):
    """ROI analysis result for initiative."""

    initiative: Initiative = Field(..., description="Initiative details")

    # ROI metrics
    roi_usd_per_tco2e: float = Field(..., description="Cost per tCO2e reduced")
    payback_period_years: Optional[float] = Field(None, description="Payback period")
    annual_savings_usd: float = Field(..., description="Annual savings")

    # NPV/IRR
    npv_10y_usd: Optional[float] = Field(None, description="10-year NPV")
    irr: Optional[float] = Field(None, description="Internal rate of return")

    # Carbon value
    carbon_value_usd: float = Field(..., description="Value of carbon reduced")

    # Analysis parameters
    discount_rate: float = Field(..., description="Discount rate used")
    carbon_price_usd_per_tco2e: float = Field(..., description="Carbon price used")
    analysis_period_years: int = Field(..., description="Analysis period")

    # Metadata
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)


class AbatementCurvePoint(BaseModel):
    """Point on marginal abatement cost curve."""

    initiative_name: str = Field(..., description="Initiative name")
    reduction_tco2e: float = Field(..., ge=0, description="Emissions reduction")
    cost_per_tco2e: float = Field(..., description="Cost per tCO2e (negative = savings)")
    cumulative_reduction: float = Field(..., ge=0, description="Cumulative reduction")
    cumulative_cost: float = Field(..., description="Cumulative cost")


class AbatementCurve(BaseModel):
    """Marginal abatement cost curve (MACC)."""

    # Initiatives sorted by cost-effectiveness
    initiatives: List[AbatementCurvePoint] = Field(..., description="Sorted initiatives")

    # Summary
    total_reduction_potential_tco2e: float = Field(..., ge=0, description="Total reduction potential")
    total_cost_usd: float = Field(..., description="Total cost")
    weighted_average_cost_per_tco2e: float = Field(..., description="Weighted average cost")

    # Breakdown
    n_negative_cost: int = Field(..., ge=0, description="Number with negative cost (savings)")
    n_positive_cost: int = Field(..., ge=0, description="Number with positive cost")

    # Visualization data
    chart_data: Dict[str, Any] = Field(..., description="Data for MACC chart")

    # Metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# HOTSPOT DETECTION MODELS
# ============================================================================

class Hotspot(BaseModel):
    """Identified emissions hotspot."""

    hotspot_id: str = Field(..., description="Hotspot identifier")
    hotspot_type: str = Field(..., description="Type (supplier, category, product, etc)")

    # Identification
    entity_name: str = Field(..., description="Entity name")
    emissions_tco2e: float = Field(..., ge=0, description="Emissions")
    percent_of_total: float = Field(..., ge=0, le=100, description="Percentage of total")

    # Triggers
    triggered_rules: List[str] = Field(..., description="Rules that flagged this hotspot")

    # Data quality
    dqi_score: Optional[float] = Field(None, description="DQI score")
    tier: Optional[int] = Field(None, description="Data tier")
    data_quality_flag: bool = Field(default=False, description="Low quality flag")

    # Priority
    priority: InsightPriority = Field(..., description="Priority level")

    # Metadata
    record_count: int = Field(..., ge=0, description="Number of records")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HotspotReport(BaseModel):
    """Comprehensive hotspot detection report."""

    total_emissions_tco2e: float = Field(..., ge=0, description="Total emissions analyzed")
    total_records: int = Field(..., ge=0, description="Total records analyzed")

    # Hotspots
    hotspots: List[Hotspot] = Field(..., description="Identified hotspots")
    n_hotspots: int = Field(..., ge=0, description="Number of hotspots")

    # Breakdown
    critical_hotspots: List[Hotspot] = Field(..., description="Critical priority")
    high_hotspots: List[Hotspot] = Field(..., description="High priority")

    # Summary
    hotspot_emissions_tco2e: float = Field(..., ge=0, description="Emissions from hotspots")
    hotspot_coverage_pct: float = Field(..., ge=0, le=100, description="% of emissions in hotspots")

    # Metadata
    criteria_used: Dict[str, Any] = Field(..., description="Criteria used")
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# INSIGHT MODELS
# ============================================================================

class Insight(BaseModel):
    """Actionable insight from analysis."""

    insight_id: str = Field(..., description="Insight identifier")
    insight_type: InsightType = Field(..., description="Insight type")
    priority: InsightPriority = Field(..., description="Priority level")

    # Content
    title: str = Field(..., description="Insight title")
    description: str = Field(..., description="Detailed description")
    recommendation: str = Field(..., description="Actionable recommendation")

    # Context
    affected_entity: Optional[str] = Field(None, description="Affected entity")
    emissions_tco2e: Optional[float] = Field(None, description="Associated emissions")
    percent_of_total: Optional[float] = Field(None, description="Percentage of total")

    # Impact
    estimated_impact: Optional[str] = Field(None, description="Estimated impact description")
    potential_reduction_tco2e: Optional[float] = Field(None, description="Potential reduction")

    # Supporting data
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Supporting metrics")

    # Metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class InsightReport(BaseModel):
    """Collection of insights from analysis."""

    total_insights: int = Field(..., ge=0, description="Total insights generated")

    # Insights by priority
    critical_insights: List[Insight] = Field(..., description="Critical insights")
    high_insights: List[Insight] = Field(..., description="High priority insights")
    medium_insights: List[Insight] = Field(..., description="Medium priority insights")
    low_insights: List[Insight] = Field(..., description="Low priority insights")

    # All insights
    all_insights: List[Insight] = Field(..., description="All insights")

    # Summary
    summary: str = Field(..., description="Executive summary")
    top_recommendations: List[str] = Field(..., description="Top 5 recommendations")

    # Metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)


__all__ = [
    # Input
    "EmissionRecord",

    # Pareto
    "ParetoItem",
    "ParetoAnalysis",

    # Segmentation
    "Segment",
    "SegmentationAnalysis",

    # Scenarios
    "BaseScenario",
    "SupplierSwitchScenario",
    "ModalShiftScenario",
    "ProductSubstitutionScenario",
    "ScenarioResult",

    # ROI
    "Initiative",
    "ROIAnalysis",
    "AbatementCurvePoint",
    "AbatementCurve",

    # Hotspots
    "Hotspot",
    "HotspotReport",

    # Insights
    "Insight",
    "InsightReport",
]
