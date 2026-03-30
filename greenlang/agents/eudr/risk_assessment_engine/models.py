# -*- coding: utf-8 -*-
"""
Risk Assessment Engine Models - AGENT-EUDR-028

Pydantic v2 models for risk assessment operations, composite risk scoring,
Article 10 criteria evaluation, country benchmarking, simplified DD eligibility,
trend analysis, manual overrides, and risk assessment reports.

All models use Decimal for numeric scores to ensure deterministic,
bit-perfect reproducibility in compliance calculations.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-028 Risk Assessment Engine (GL-EUDR-RAE-028)
Regulation: EU 2023/1115 (EUDR) Articles 10, 13, 29, 31
Status: Production Ready
"""
from __future__ import annotations

import enum
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import Field
from greenlang.schemas import GreenLangBase


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RiskDimension(str, enum.Enum):
    """Risk dimensions contributing to the composite risk score."""

    COUNTRY = "country"
    COMMODITY = "commodity"
    SUPPLIER = "supplier"
    DEFORESTATION = "deforestation"
    CORRUPTION = "corruption"
    SUPPLY_CHAIN_COMPLEXITY = "supply_chain_complexity"
    MIXING_RISK = "mixing_risk"
    CIRCUMVENTION_RISK = "circumvention_risk"


class RiskLevel(str, enum.Enum):
    """Risk classification levels per EUDR Article 10(2)."""

    NEGLIGIBLE = "negligible"
    LOW = "low"
    STANDARD = "standard"
    HIGH = "high"
    CRITICAL = "critical"


class RiskAssessmentStatus(str, enum.Enum):
    """Risk assessment operation lifecycle status."""

    INITIATED = "initiated"
    AGGREGATING = "aggregating"
    EVALUATING = "evaluating"
    CLASSIFYING = "classifying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CountryBenchmarkLevel(str, enum.Enum):
    """Country benchmarking classification per Article 29(2)."""

    LOW = "low"
    STANDARD = "standard"
    HIGH = "high"


class Article10Criterion(str, enum.Enum):
    """Article 10(2) risk assessment criteria."""

    PREVALENCE_OF_DEFORESTATION = "prevalence_of_deforestation"
    SUPPLY_CHAIN_COMPLEXITY = "supply_chain_complexity"
    MIXING_RISK = "mixing_risk"
    CIRCUMVENTION_RISK = "circumvention_risk"
    COUNTRY_GOVERNANCE = "country_governance"
    SUPPLIER_COMPLIANCE = "supplier_compliance"
    COMMODITY_RISK_PROFILE = "commodity_risk_profile"
    CERTIFICATION_COVERAGE = "certification_coverage"
    DEFORESTATION_ALERTS = "deforestation_alerts"
    LEGAL_FRAMEWORK = "legal_framework"


class CriterionResult(str, enum.Enum):
    """Result of evaluating a single Article 10 criterion."""

    PASS = "pass"
    CONCERN = "concern"
    FAIL = "fail"
    NOT_EVALUATED = "not_evaluated"


class TrendDirection(str, enum.Enum):
    """Direction of risk score trend over time."""

    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    INSUFFICIENT_DATA = "insufficient_data"


class OverrideReason(str, enum.Enum):
    """Justification categories for manual risk overrides."""

    EXPERT_JUDGMENT = "expert_judgment"
    NEW_EVIDENCE = "new_evidence"
    REGULATORY_CHANGE = "regulatory_change"
    DATA_CORRECTION = "data_correction"
    MITIGATING_FACTORS = "mitigating_factors"


class EUDRCommodity(str, enum.Enum):
    """EUDR regulated commodities (Article 1)."""

    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    OIL_PALM = "oil_palm"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"


class SourceAgent(str, enum.Enum):
    """Upstream EUDR agents providing risk factor inputs."""

    EUDR_016_COUNTRY = "eudr_016_country_risk"
    EUDR_017_SUPPLIER = "eudr_017_supplier_risk"
    EUDR_018_COMMODITY = "eudr_018_commodity_risk"
    EUDR_019_CORRUPTION = "eudr_019_corruption_index"
    EUDR_020_DEFORESTATION = "eudr_020_deforestation_alert"
    EUDR_028_DERIVED = "eudr_028_derived"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: Dict[RiskDimension, Decimal] = {
    RiskDimension.COUNTRY: Decimal("0.20"),
    RiskDimension.COMMODITY: Decimal("0.15"),
    RiskDimension.SUPPLIER: Decimal("0.20"),
    RiskDimension.DEFORESTATION: Decimal("0.20"),
    RiskDimension.CORRUPTION: Decimal("0.10"),
    RiskDimension.SUPPLY_CHAIN_COMPLEXITY: Decimal("0.05"),
    RiskDimension.MIXING_RISK: Decimal("0.05"),
    RiskDimension.CIRCUMVENTION_RISK: Decimal("0.05"),
}

RISK_THRESHOLDS: Dict[RiskLevel, Decimal] = {
    RiskLevel.NEGLIGIBLE: Decimal("15"),
    RiskLevel.LOW: Decimal("30"),
    RiskLevel.STANDARD: Decimal("60"),
    RiskLevel.HIGH: Decimal("80"),
    RiskLevel.CRITICAL: Decimal("100"),
}

COUNTRY_BENCHMARK_MULTIPLIERS: Dict[CountryBenchmarkLevel, Decimal] = {
    CountryBenchmarkLevel.LOW: Decimal("0.70"),
    CountryBenchmarkLevel.STANDARD: Decimal("1.00"),
    CountryBenchmarkLevel.HIGH: Decimal("1.50"),
}

SUPPORTED_COMMODITIES: List[str] = [c.value for c in EUDRCommodity]

VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class RiskFactorInput(GreenLangBase):
    """A single risk factor input from an upstream EUDR agent."""

    source_agent: SourceAgent
    dimension: RiskDimension
    raw_score: Decimal = Field(..., ge=0, le=100, description="Raw risk score 0-100")
    confidence: Decimal = Field(..., ge=0, le=1, description="Confidence level 0-1")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class DimensionScore(GreenLangBase):
    """Weighted score for a single risk dimension."""

    dimension: RiskDimension
    weighted_score: Decimal = Field(..., description="Weight-adjusted score")
    raw_score: Decimal = Field(..., description="Original raw score 0-100")
    weight: Decimal = Field(..., description="Dimension weight applied")
    confidence: Decimal = Field(..., ge=0, le=1, description="Confidence level")
    source_agent: SourceAgent
    explanation: str = ""


class CompositeRiskScore(GreenLangBase):
    """Aggregated composite risk score across all dimensions."""

    overall_score: Decimal = Field(
        ..., ge=0, le=100, description="Composite risk score 0-100"
    )
    risk_level: RiskLevel
    dimension_scores: List[DimensionScore] = Field(default_factory=list)
    total_weight: Decimal = Field(
        default=Decimal("1.00"), description="Sum of applied weights"
    )
    effective_confidence: Decimal = Field(
        default=Decimal("0"), description="Weighted average confidence"
    )
    country_benchmark_applied: bool = False
    benchmark_multiplier: Optional[Decimal] = None
    provenance_hash: str = ""


class Article10CriterionEvaluation(GreenLangBase):
    """Evaluation result for a single Article 10(2) criterion."""

    criterion: Article10Criterion
    result: CriterionResult = CriterionResult.NOT_EVALUATED
    score: Decimal = Field(default=Decimal("0"), description="Criterion score 0-100")
    threshold: Optional[Decimal] = None
    explanation: str = ""
    evidence: List[str] = Field(default_factory=list)
    evidence_summary: str = ""
    data_sources: List[str] = Field(default_factory=list)


class Article10CriteriaResult(GreenLangBase):
    """Aggregated Article 10(2) criteria evaluation results."""

    evaluations: List[Article10CriterionEvaluation] = Field(default_factory=list)
    overall_concern_count: int = 0
    criteria_evaluated: int = 0
    criteria_passed: int = 0
    criteria_with_concerns: int = 0
    # Engine-produced fields
    total_evaluated: int = 0
    pass_count: int = 0
    concern_count: int = 0
    fail_count: int = 0
    not_evaluated_count: int = 0
    evaluated_at: Optional[datetime] = None
    provenance_hash: str = ""


class CountryBenchmark(GreenLangBase):
    """Country benchmarking data per Article 29(2)."""

    country_code: str = Field(..., min_length=2, max_length=3)
    benchmark_level: CountryBenchmarkLevel = CountryBenchmarkLevel.STANDARD
    level: Optional[CountryBenchmarkLevel] = None
    multiplier: Optional[Decimal] = None
    effective_date: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    source: str = ""
    governance_score: Decimal = Field(
        default=Decimal("0"), description="Governance quality score"
    )
    deforestation_rate: Decimal = Field(
        default=Decimal("0"), description="Annual deforestation rate"
    )
    confidence: Decimal = Field(default=Decimal("0"), ge=0, le=1)

    def model_post_init(self, __context: Any) -> None:
        """Synchronize level/benchmark_level fields."""
        if self.level is not None and self.benchmark_level == CountryBenchmarkLevel.STANDARD:
            object.__setattr__(self, "benchmark_level", self.level)
        elif self.level is None:
            object.__setattr__(self, "level", self.benchmark_level)


class SimplifiedDDEligibility(GreenLangBase):
    """Article 13 simplified due diligence eligibility assessment."""

    is_eligible: bool = False
    reasons: List[str] = Field(default_factory=list)
    country_benchmarks: List[CountryBenchmark] = Field(default_factory=list)
    composite_score: Optional[Decimal] = None
    all_countries_low: bool = False
    evaluated_at: Optional[datetime] = None
    provenance_hash: str = ""

    model_config = {"populate_by_name": True}

    def __init__(self, **data: Any) -> None:
        """Support 'eligible' as an alias for 'is_eligible'."""
        if "eligible" in data and "is_eligible" not in data:
            data["is_eligible"] = data.pop("eligible")
        super().__init__(**data)

    @property
    def eligible(self) -> bool:
        """Alias for is_eligible."""
        return self.is_eligible


class RiskTrendPoint(GreenLangBase):
    """A single data point in a risk trend time series."""

    # Engine-produced fields
    point_id: str = ""
    operator_id: str = ""
    commodity: str = ""
    score: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    level: Optional[RiskLevel] = None
    timestamp: Optional[datetime] = None

    # Legacy/report fields
    assessment_date: Optional[datetime] = None
    composite_score: Optional[Decimal] = Field(default=None, ge=0, le=100)
    risk_level: Optional[RiskLevel] = None
    key_changes: Any = Field(default_factory=list)

    model_config = {"extra": "ignore"}


class RiskTrendAnalysis(GreenLangBase):
    """Risk trend analysis over a configurable time window."""

    operator_id: str
    commodity: str = ""
    trend_points: List[RiskTrendPoint] = Field(default_factory=list)
    direction: TrendDirection = TrendDirection.INSUFFICIENT_DATA
    average_score: Decimal = Field(default=Decimal("0"))
    score_change_30d: Optional[Decimal] = None
    score_change_90d: Optional[Decimal] = None
    score_change_365d: Optional[Decimal] = None

    # Engine-produced fields
    data_point_count: int = 0
    latest_score: Decimal = Field(default=Decimal("0"))
    latest_level: Optional[RiskLevel] = None
    change_30d: Optional[Decimal] = None
    change_90d: Optional[Decimal] = None
    change_365d: Optional[Decimal] = None
    analyzed_at: Optional[datetime] = None
    provenance_hash: str = ""

    model_config = {"extra": "ignore"}


class RiskOverride(GreenLangBase):
    """Manual risk score override with audit trail."""

    override_id: str
    assessment_id: str
    original_score: Decimal = Field(..., ge=0, le=100)
    overridden_score: Decimal = Field(..., ge=0, le=100)
    original_level: RiskLevel
    overridden_level: RiskLevel
    reason: OverrideReason
    justification: str = ""
    overridden_by: str = ""
    approved_by: Optional[str] = None
    valid_until: Optional[datetime] = None


class RiskAssessmentReport(GreenLangBase):
    """Complete risk assessment report for DDS submission."""

    report_id: str
    assessment_id: str = ""
    operator_id: str = ""
    commodity: Optional[str] = None
    composite_score: Optional[CompositeRiskScore] = None
    risk_level: Optional[RiskLevel] = None
    dimension_scores: List[DimensionScore] = Field(default_factory=list)
    article10_criteria: Optional[Article10CriteriaResult] = None
    article10_result: Optional[Article10CriteriaResult] = None
    country_benchmarks: List[CountryBenchmark] = Field(default_factory=list)
    simplified_dd_eligibility: Optional[SimplifiedDDEligibility] = None
    trend_analysis: Optional[RiskTrendAnalysis] = None
    overrides: List[RiskOverride] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    dds_ready: bool = False
    provenance_hash: str = ""
    generated_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    operation: Optional[RiskAssessmentOperation] = None

    model_config = {"extra": "ignore"}

    def model_post_init(self, __context: Any) -> None:
        """Extract fields from operation if provided."""
        if self.operation is not None:
            if not self.assessment_id:
                object.__setattr__(self, "assessment_id", self.operation.operation_id)
            if not self.operator_id:
                object.__setattr__(self, "operator_id", self.operation.operator_id)
            if self.commodity is None:
                object.__setattr__(self, "commodity", self.operation.commodity.value)


class RiskAssessmentOperation(GreenLangBase):
    """Top-level risk assessment operation tracking."""

    operation_id: str
    operator_id: str
    commodity: EUDRCommodity
    workflow_id: Optional[str] = None
    status: RiskAssessmentStatus = RiskAssessmentStatus.INITIATED
    risk_factor_inputs: List[RiskFactorInput] = Field(default_factory=list)
    composite_score: Optional[CompositeRiskScore] = None
    risk_level: Optional[RiskLevel] = None
    article10_result: Optional[Article10CriteriaResult] = None
    report_id: Optional[str] = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    provenance_hash: str = ""


class RiskAssessmentRequest(GreenLangBase):
    """Request to perform a risk assessment for an operator-commodity pair."""

    operator_id: str
    commodity: EUDRCommodity
    country_codes: List[str] = Field(default_factory=list)
    supplier_ids: List[str] = Field(default_factory=list)
    override_score: Optional[Decimal] = Field(
        default=None, ge=0, le=100, description="Optional manual override score"
    )
    override_reason: Optional[OverrideReason] = None
    override_justification: Optional[str] = None


class BatchRiskAssessmentRequest(GreenLangBase):
    """Batch request for multiple risk assessments."""

    assessments: List[RiskAssessmentRequest] = Field(default_factory=list)


class HealthResponse(GreenLangBase):
    """Health check response for the Risk Assessment Engine."""

    status: str = "healthy"
    version: str = VERSION
    agent_id: str = "AGENT-EUDR-028"
    active_assessments: int = 0
    upstream_agents_status: Dict[str, str] = Field(default_factory=dict)
    database_connected: bool = False
    cache_connected: bool = False
