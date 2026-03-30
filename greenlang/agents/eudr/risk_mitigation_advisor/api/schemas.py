# -*- coding: utf-8 -*-
"""
API Schemas - AGENT-EUDR-025 Risk Mitigation Advisor

Pydantic v2 request/response schemas specifically tailored for the REST API
layer. These schemas wrap the core domain models from ``models.py`` with
API-specific concerns: pagination metadata, error envelopes, query filter
models, and OpenAPI-friendly response wrappers.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-025 Risk Mitigation Advisor (GL-EUDR-RMA-025)
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import Field
from greenlang.schemas import GreenLangBase


# ---------------------------------------------------------------------------
# Common Response Wrappers
# ---------------------------------------------------------------------------


class PaginatedMeta(GreenLangBase):
    """Pagination metadata included in list responses."""

    total: int = Field(..., ge=0, description="Total number of records matching filters")
    limit: int = Field(..., ge=1, description="Page size")
    offset: int = Field(..., ge=0, description="Current offset")
    has_more: bool = Field(..., description="Whether more records exist beyond this page")


class ProvenanceInfo(GreenLangBase):
    """Provenance information for audit trail."""

    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    model_version: str = Field(default="", description="Model or engine version used")
    computation_mode: str = Field(default="deterministic", description="ml or deterministic")
    timestamp: datetime = Field(default_factory=lambda: datetime.utcnow(), description="Computation timestamp")


class ErrorResponse(GreenLangBase):
    """Standard error response envelope."""

    error: str = Field(..., description="Error type identifier")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Any] = Field(default=None, description="Additional error context")
    request_id: Optional[str] = Field(default=None, description="Request correlation ID")


# ---------------------------------------------------------------------------
# Strategy Schemas
# ---------------------------------------------------------------------------


class StrategyRecommendRequest(GreenLangBase):
    """Request body for POST /strategies/recommend."""

    operator_id: str = Field(..., min_length=1, description="Operator identifier")
    supplier_id: str = Field(..., min_length=1, description="Supplier identifier")
    country_code: str = Field(..., min_length=2, max_length=2, description="ISO 3166-1 alpha-2 country code")
    commodity: str = Field(..., description="EUDR commodity: cattle, cocoa, coffee, palm_oil, rubber, soya, wood")
    country_risk_score: Decimal = Field(default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"), description="EUDR-016 country risk score")
    supplier_risk_score: Decimal = Field(default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"), description="EUDR-017 supplier risk score")
    commodity_risk_score: Decimal = Field(default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"), description="EUDR-018 commodity risk score")
    corruption_risk_score: Decimal = Field(default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"), description="EUDR-019 corruption risk score")
    deforestation_risk_score: Decimal = Field(default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"), description="EUDR-020 deforestation risk score")
    indigenous_rights_score: Decimal = Field(default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"), description="EUDR-021 indigenous rights score")
    protected_areas_score: Decimal = Field(default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"), description="EUDR-022 protected areas score")
    legal_compliance_score: Decimal = Field(default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"), description="EUDR-023 legal compliance score")
    audit_risk_score: Decimal = Field(default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"), description="EUDR-024 audit risk score")
    due_diligence_level: str = Field(default="standard", description="Required DDS level: simplified, standard, enhanced")
    risk_factors: Dict[str, Any] = Field(default_factory=dict, description="Additional risk factor details from upstream agents")
    mode: str = Field(default="ml", description="Recommendation mode: ml or deterministic")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of strategies to recommend")

    class Config:
        json_schema_extra = {
            "example": {
                "operator_id": "op-001",
                "supplier_id": "sup-001",
                "country_code": "ID",
                "commodity": "palm_oil",
                "country_risk_score": "75.00",
                "supplier_risk_score": "65.00",
                "commodity_risk_score": "70.00",
                "corruption_risk_score": "60.00",
                "deforestation_risk_score": "80.00",
                "mode": "ml",
                "top_k": 5,
            }
        }


class StrategyEntry(GreenLangBase):
    """A single strategy in a recommendation response."""

    strategy_id: str = Field(..., description="Strategy unique identifier")
    name: str = Field(..., description="Strategy name")
    description: str = Field(default="", description="Strategy description")
    risk_categories: List[str] = Field(default_factory=list, description="Addressed risk categories")
    iso_31000_type: str = Field(default="reduce", description="ISO 31000 treatment type")
    predicted_effectiveness: Decimal = Field(default=Decimal("0"), description="Predicted risk reduction 0-100")
    confidence_score: Decimal = Field(default=Decimal("0"), description="ML confidence 0-1")
    cost_estimate: Dict[str, Any] = Field(default_factory=dict, description="Structured cost estimate")
    implementation_complexity: str = Field(default="medium", description="low, medium, high, very_high")
    time_to_effect_weeks: int = Field(default=8, description="Weeks to measurable impact")
    eudr_articles: List[str] = Field(default_factory=list, description="Linked EUDR articles")
    shap_explanation: Dict[str, float] = Field(default_factory=dict, description="SHAP feature importance values")
    measure_ids: List[str] = Field(default_factory=list, description="Linked mitigation measure IDs")
    model_version: str = Field(default="", description="Model version used")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    status: str = Field(default="recommended", description="Strategy status")
    created_at: Optional[datetime] = Field(default=None, description="Creation timestamp")


class StrategyRecommendResponse(GreenLangBase):
    """Response for POST /strategies/recommend."""

    strategies: List[StrategyEntry] = Field(default_factory=list, description="Ranked strategy recommendations")
    risk_profile_summary: Dict[str, Any] = Field(default_factory=dict, description="Input risk profile summary")
    computation_mode: str = Field(default="ml", description="Mode used: ml or deterministic")
    provenance: ProvenanceInfo = Field(default_factory=ProvenanceInfo, description="Provenance metadata")


class StrategyListResponse(GreenLangBase):
    """Response for GET /strategies."""

    strategies: List[StrategyEntry] = Field(default_factory=list, description="Strategy list")
    meta: PaginatedMeta = Field(..., description="Pagination metadata")


class StrategySelectRequest(GreenLangBase):
    """Request body for POST /strategies/{strategy_id}/select."""

    selected_by: str = Field(..., min_length=1, description="User selecting the strategy")
    notes: str = Field(default="", description="Selection rationale")


class StrategyExplainResponse(GreenLangBase):
    """Response for GET /strategies/{strategy_id}/explain."""

    strategy_id: str = Field(..., description="Strategy identifier")
    strategy_name: str = Field(default="", description="Strategy name")
    shap_values: Dict[str, float] = Field(default_factory=dict, description="SHAP values by feature")
    feature_importance_ranked: List[Dict[str, Any]] = Field(default_factory=list, description="Features ranked by absolute SHAP value")
    model_version: str = Field(default="", description="ML model version")
    computation_mode: str = Field(default="ml", description="ml or deterministic")
    provenance: ProvenanceInfo = Field(default_factory=ProvenanceInfo, description="Provenance metadata")


# ---------------------------------------------------------------------------
# Remediation Plan Schemas
# ---------------------------------------------------------------------------


class CreatePlanRequest(GreenLangBase):
    """Request body for POST /plans."""

    operator_id: str = Field(..., min_length=1, description="Operator identifier")
    supplier_id: Optional[str] = Field(default=None, description="Supplier identifier")
    plan_name: str = Field(..., min_length=1, max_length=500, description="Plan name")
    strategy_ids: List[str] = Field(default_factory=list, description="Selected strategy IDs to implement")
    risk_finding_ids: List[str] = Field(default_factory=list, description="Linked risk finding IDs")
    template: Optional[str] = Field(default=None, description="Plan template name")
    budget_allocated: Decimal = Field(default=Decimal("0"), ge=Decimal("0"), description="Total budget (EUR)")
    start_date: Optional[date] = Field(default=None, description="Plan start date")
    responsible_parties: List[Dict[str, Any]] = Field(default_factory=list, description="Responsible parties")
    commodity: Optional[str] = Field(default=None, description="EUDR commodity context")

    class Config:
        json_schema_extra = {
            "example": {
                "operator_id": "op-001",
                "supplier_id": "sup-001",
                "plan_name": "Supplier Capacity Building - PT Agro Indonesia",
                "strategy_ids": ["strat-001", "strat-002"],
                "template": "supplier_capacity_building",
                "budget_allocated": "50000.00",
                "start_date": "2026-04-01",
            }
        }


class PlanEntry(GreenLangBase):
    """A remediation plan summary entry."""

    plan_id: str = Field(..., description="Plan identifier")
    operator_id: str = Field(default="", description="Operator identifier")
    supplier_id: Optional[str] = Field(default=None, description="Supplier identifier")
    plan_name: str = Field(default="", description="Plan name")
    status: str = Field(default="draft", description="Plan status")
    plan_template: Optional[str] = Field(default=None, description="Template used")
    budget_allocated: Decimal = Field(default=Decimal("0"), description="Budget allocated (EUR)")
    budget_spent: Decimal = Field(default=Decimal("0"), description="Budget spent (EUR)")
    start_date: Optional[date] = Field(default=None, description="Start date")
    target_end_date: Optional[date] = Field(default=None, description="Target end date")
    milestone_count: int = Field(default=0, description="Total milestones")
    milestones_completed: int = Field(default=0, description="Completed milestones")
    progress_pct: Decimal = Field(default=Decimal("0"), description="Progress percentage")
    version: int = Field(default=1, description="Plan version")
    created_at: Optional[datetime] = Field(default=None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Last update timestamp")


class PlanDetailResponse(GreenLangBase):
    """Full plan detail response for GET /plans/{plan_id}."""

    plan: PlanEntry = Field(..., description="Plan summary")
    phases: List[Dict[str, Any]] = Field(default_factory=list, description="Plan phases with milestones")
    milestones: List[Dict[str, Any]] = Field(default_factory=list, description="All milestones")
    kpis: List[Dict[str, Any]] = Field(default_factory=list, description="Key performance indicators")
    responsible_parties: List[Dict[str, Any]] = Field(default_factory=list, description="Responsible parties")
    escalation_triggers: List[Dict[str, Any]] = Field(default_factory=list, description="Escalation triggers")
    strategy_ids: List[str] = Field(default_factory=list, description="Linked strategy IDs")
    risk_finding_ids: List[str] = Field(default_factory=list, description="Linked risk finding IDs")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class PlanListResponse(GreenLangBase):
    """Response for GET /plans."""

    plans: List[PlanEntry] = Field(default_factory=list, description="Plan list")
    meta: PaginatedMeta = Field(..., description="Pagination metadata")


class PlanStatusUpdateRequest(GreenLangBase):
    """Request body for PUT /plans/{plan_id}/status."""

    status: str = Field(..., description="New status: active, on_track, at_risk, delayed, completed, suspended, abandoned")
    reason: str = Field(default="", description="Reason for status change")
    approved_by: Optional[str] = Field(default=None, description="Approver for activation/completion")


class PlanCloneRequest(GreenLangBase):
    """Request body for POST /plans/{plan_id}/clone."""

    target_supplier_id: str = Field(..., min_length=1, description="Target supplier for cloned plan")
    new_plan_name: Optional[str] = Field(default=None, description="Override plan name for clone")
    start_date: Optional[date] = Field(default=None, description="New start date for clone")
    budget_allocated: Optional[Decimal] = Field(default=None, ge=Decimal("0"), description="Override budget for clone")


class GanttChartResponse(GreenLangBase):
    """Response for GET /plans/{plan_id}/gantt."""

    plan_id: str = Field(..., description="Plan identifier")
    plan_name: str = Field(default="", description="Plan name")
    phases: List[Dict[str, Any]] = Field(default_factory=list, description="Phase timeline data")
    milestones: List[Dict[str, Any]] = Field(default_factory=list, description="Milestone timeline data")
    dependencies: List[Dict[str, str]] = Field(default_factory=list, description="Milestone dependency pairs")
    critical_path: List[str] = Field(default_factory=list, description="Milestone IDs on critical path")
    start_date: Optional[date] = Field(default=None, description="Plan start date")
    end_date: Optional[date] = Field(default=None, description="Plan target end date")


class MilestoneCreateRequest(GreenLangBase):
    """Request body for POST /plans/{plan_id}/milestones."""

    name: str = Field(..., min_length=1, max_length=500, description="Milestone name")
    description: str = Field(default="", description="Milestone description")
    phase: str = Field(default="implementation", description="Phase: preparation, implementation, verification, monitoring")
    due_date: date = Field(..., description="Target completion date")
    kpi_target: Optional[str] = Field(default=None, description="Target KPI value")
    evidence_required: List[str] = Field(default_factory=list, description="Required evidence types")
    eudr_article: Optional[str] = Field(default=None, description="Linked EUDR article")
    dependencies: List[str] = Field(default_factory=list, description="Prerequisite milestone IDs")


class MilestoneUpdateRequest(GreenLangBase):
    """Request body for PUT /plans/{plan_id}/milestones/{milestone_id}."""

    status: Optional[str] = Field(default=None, description="New status: pending, in_progress, completed, overdue, skipped")
    completed_date: Optional[date] = Field(default=None, description="Completion date")
    kpi_actual: Optional[str] = Field(default=None, description="Actual KPI value achieved")
    notes: str = Field(default="", description="Update notes")


class EvidenceUploadRequest(GreenLangBase):
    """Request body for POST /plans/{plan_id}/milestones/{milestone_id}/evidence."""

    file_name: str = Field(..., min_length=1, description="Evidence file name")
    document_type: str = Field(..., description="Evidence type: report, photo, certificate, audit, satellite, gps")
    description: str = Field(default="", description="Evidence description")
    file_size_bytes: int = Field(default=0, ge=0, description="File size in bytes")
    content_hash: str = Field(default="", description="SHA-256 hash of file content")
    storage_url: str = Field(default="", description="S3 storage URL after upload")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# ---------------------------------------------------------------------------
# Capacity Building Schemas
# ---------------------------------------------------------------------------


class EnrollSupplierRequest(GreenLangBase):
    """Request body for POST /capacity-building/enroll."""

    supplier_id: str = Field(..., min_length=1, description="Supplier identifier")
    commodity: str = Field(..., description="EUDR commodity for training: cattle, cocoa, coffee, palm_oil, rubber, soya, wood")
    program_id: Optional[str] = Field(default=None, description="Specific program ID")
    initial_tier: int = Field(default=1, ge=1, le=4, description="Starting capacity tier (1-4)")
    target_completion_date: Optional[date] = Field(default=None, description="Target completion date")
    risk_score_at_enrollment: Optional[Decimal] = Field(default=None, ge=Decimal("0"), le=Decimal("100"), description="Current supplier risk score")

    class Config:
        json_schema_extra = {
            "example": {
                "supplier_id": "sup-001",
                "commodity": "palm_oil",
                "initial_tier": 1,
                "target_completion_date": "2026-12-31",
                "risk_score_at_enrollment": "65.00",
            }
        }


class EnrollmentEntry(GreenLangBase):
    """A capacity building enrollment entry."""

    enrollment_id: str = Field(..., description="Enrollment identifier")
    supplier_id: str = Field(default="", description="Supplier identifier")
    program_id: str = Field(default="", description="Program identifier")
    commodity: str = Field(default="", description="EUDR commodity")
    current_tier: int = Field(default=1, description="Current capacity tier (1-4)")
    modules_completed: int = Field(default=0, description="Modules completed")
    modules_total: int = Field(default=22, description="Total modules in program")
    completion_pct: Decimal = Field(default=Decimal("0"), description="Completion percentage")
    competency_scores: Dict[str, Any] = Field(default_factory=dict, description="Competency scores by module")
    enrolled_date: Optional[date] = Field(default=None, description="Enrollment date")
    target_completion_date: Optional[date] = Field(default=None, description="Target completion")
    status: str = Field(default="active", description="Enrollment status")
    risk_score_at_enrollment: Optional[Decimal] = Field(default=None, description="Risk score at enrollment")
    current_risk_score: Optional[Decimal] = Field(default=None, description="Current risk score")
    risk_reduction_pct: Optional[Decimal] = Field(default=None, description="Risk reduction since enrollment")


class EnrollmentListResponse(GreenLangBase):
    """Response for GET /capacity-building/enrollments."""

    enrollments: List[EnrollmentEntry] = Field(default_factory=list, description="Enrollment list")
    meta: PaginatedMeta = Field(..., description="Pagination metadata")


class ProgressUpdateRequest(GreenLangBase):
    """Request body for PUT /capacity-building/enrollments/{enrollment_id}/progress."""

    modules_completed: Optional[int] = Field(default=None, ge=0, description="Updated modules completed count")
    competency_scores: Optional[Dict[str, Any]] = Field(default=None, description="Updated competency scores")
    current_risk_score: Optional[Decimal] = Field(default=None, ge=Decimal("0"), le=Decimal("100"), description="Updated risk score")
    notes: str = Field(default="", description="Progress notes")


class ScorecardResponse(GreenLangBase):
    """Response for GET /capacity-building/scorecard/{supplier_id}."""

    supplier_id: str = Field(..., description="Supplier identifier")
    enrollments: List[EnrollmentEntry] = Field(default_factory=list, description="All enrollments for supplier")
    overall_tier: int = Field(default=1, description="Highest tier achieved across programs")
    overall_completion_pct: Decimal = Field(default=Decimal("0"), description="Average completion across programs")
    risk_score_improvement: Optional[Decimal] = Field(default=None, description="Risk score change since first enrollment")
    tier_advancement_eligible: bool = Field(default=False, description="Whether supplier can advance tier")
    recommendations: List[str] = Field(default_factory=list, description="Next-step recommendations")


# ---------------------------------------------------------------------------
# Measure Library Schemas
# ---------------------------------------------------------------------------


class MeasureEntry(GreenLangBase):
    """A mitigation measure summary."""

    measure_id: str = Field(..., description="Measure identifier")
    name: str = Field(default="", description="Measure name")
    description: str = Field(default="", description="Measure description")
    risk_category: str = Field(default="", description="Primary risk category")
    sub_category: str = Field(default="", description="Risk sub-category")
    effectiveness_rating: Decimal = Field(default=Decimal("0"), description="Effectiveness rating 0-100")
    cost_estimate_min: Optional[Decimal] = Field(default=None, description="Minimum cost (EUR)")
    cost_estimate_max: Optional[Decimal] = Field(default=None, description="Maximum cost (EUR)")
    implementation_complexity: str = Field(default="medium", description="Complexity level")
    time_to_effect_weeks: int = Field(default=8, description="Weeks to impact")
    expected_risk_reduction_min: Optional[Decimal] = Field(default=None, description="Min risk reduction %")
    expected_risk_reduction_max: Optional[Decimal] = Field(default=None, description="Max risk reduction %")
    iso_31000_type: str = Field(default="reduce", description="ISO 31000 treatment type")
    eudr_articles: List[str] = Field(default_factory=list, description="Linked EUDR articles")
    certification_schemes: List[str] = Field(default_factory=list, description="Applicable certification schemes")
    tags: List[str] = Field(default_factory=list, description="Searchable tags")
    applicability: Dict[str, Any] = Field(default_factory=dict, description="Applicability criteria")


class MeasureDetailResponse(GreenLangBase):
    """Full measure detail response for GET /measures/{measure_id}."""

    measure: MeasureEntry = Field(..., description="Measure data")
    effectiveness_evidence: List[Dict[str, Any]] = Field(default_factory=list, description="Supporting evidence")
    prerequisite_conditions: List[str] = Field(default_factory=list, description="Prerequisites")
    target_risk_factors: List[str] = Field(default_factory=list, description="Risk factors addressed")


class MeasureSearchResponse(GreenLangBase):
    """Response for GET /measures."""

    measures: List[MeasureEntry] = Field(default_factory=list, description="Matching measures")
    meta: PaginatedMeta = Field(..., description="Pagination metadata")
    facets: Dict[str, Any] = Field(default_factory=dict, description="Available facet counts for refinement")


class MeasureCompareResponse(GreenLangBase):
    """Response for GET /measures/compare."""

    measures: List[MeasureEntry] = Field(default_factory=list, description="Measures being compared")
    comparison_matrix: Dict[str, Any] = Field(default_factory=dict, description="Side-by-side comparison data")
    recommendation: Optional[str] = Field(default=None, description="Recommended measure based on context")


class MeasurePackageResponse(GreenLangBase):
    """Response for GET /measures/packages/{risk_scenario}."""

    risk_scenario: str = Field(..., description="Risk scenario identifier")
    description: str = Field(default="", description="Scenario description")
    recommended_measures: List[MeasureEntry] = Field(default_factory=list, description="Recommended measures")
    total_estimated_cost: Dict[str, Any] = Field(default_factory=dict, description="Total cost estimate range")
    expected_risk_reduction: Decimal = Field(default=Decimal("0"), description="Expected aggregate risk reduction %")


# ---------------------------------------------------------------------------
# Effectiveness Tracking Schemas
# ---------------------------------------------------------------------------


class EffectivenessEntry(GreenLangBase):
    """An effectiveness measurement record."""

    record_id: str = Field(default="", description="Record identifier")
    plan_id: str = Field(default="", description="Plan identifier")
    supplier_id: str = Field(default="", description="Supplier identifier")
    baseline_risk_scores: Dict[str, Any] = Field(default_factory=dict, description="T0 risk scores")
    current_risk_scores: Dict[str, Any] = Field(default_factory=dict, description="Current risk scores")
    dimension_reductions: Dict[str, Any] = Field(default_factory=dict, description="Per-dimension reduction %")
    composite_reduction_pct: Decimal = Field(default=Decimal("0"), description="Weighted composite reduction %")
    predicted_reduction_pct: Optional[Decimal] = Field(default=None, description="Predicted reduction %")
    deviation_pct: Optional[Decimal] = Field(default=None, description="Predicted vs actual deviation %")
    roi: Optional[Decimal] = Field(default=None, description="Return on investment %")
    cost_to_date: Decimal = Field(default=Decimal("0"), description="Mitigation cost to date (EUR)")
    statistical_significance: bool = Field(default=False, description="Whether reduction is statistically significant")
    p_value: Optional[Decimal] = Field(default=None, description="Statistical p-value")
    measured_at: Optional[datetime] = Field(default=None, description="Measurement timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class EffectivenessPlanResponse(GreenLangBase):
    """Response for GET /effectiveness/{plan_id}."""

    plan_id: str = Field(..., description="Plan identifier")
    records: List[EffectivenessEntry] = Field(default_factory=list, description="Effectiveness records over time")
    latest: Optional[EffectivenessEntry] = Field(default=None, description="Most recent measurement")
    trend: str = Field(default="stable", description="Trend direction: improving, stable, degrading")


class EffectivenessSupplierResponse(GreenLangBase):
    """Response for GET /effectiveness/supplier/{supplier_id}."""

    supplier_id: str = Field(..., description="Supplier identifier")
    records: List[EffectivenessEntry] = Field(default_factory=list, description="All effectiveness records")
    aggregate_reduction_pct: Decimal = Field(default=Decimal("0"), description="Aggregate risk reduction across plans")
    aggregate_roi: Optional[Decimal] = Field(default=None, description="Aggregate ROI across plans")


class PortfolioEffectivenessResponse(GreenLangBase):
    """Response for GET /effectiveness/portfolio."""

    total_plans_active: int = Field(default=0, description="Active plans count")
    total_suppliers_under_mitigation: int = Field(default=0, description="Suppliers under mitigation")
    average_composite_reduction_pct: Decimal = Field(default=Decimal("0"), description="Average risk reduction")
    average_roi: Optional[Decimal] = Field(default=None, description="Average ROI across portfolio")
    top_performing_plans: List[Dict[str, Any]] = Field(default_factory=list, description="Top performing plans by reduction")
    underperforming_plans: List[Dict[str, Any]] = Field(default_factory=list, description="Plans needing attention")
    trend_data: List[Dict[str, Any]] = Field(default_factory=list, description="Monthly trend data points")


class ROIAnalysisResponse(GreenLangBase):
    """Response for GET /effectiveness/roi."""

    total_investment_eur: Decimal = Field(default=Decimal("0"), description="Total mitigation investment")
    total_risk_reduction_value_eur: Decimal = Field(default=Decimal("0"), description="Total risk reduction value")
    portfolio_roi_pct: Decimal = Field(default=Decimal("0"), description="Portfolio-level ROI %")
    by_risk_category: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="ROI by risk category")
    by_commodity: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="ROI by commodity")
    cost_effectiveness_ranking: List[Dict[str, Any]] = Field(default_factory=list, description="Measures ranked by cost-effectiveness")


# ---------------------------------------------------------------------------
# Monitoring / Adaptive Management Schemas
# ---------------------------------------------------------------------------


class TriggerEventEntry(GreenLangBase):
    """A monitoring trigger event."""

    event_id: str = Field(default="", description="Event identifier")
    plan_id: Optional[str] = Field(default=None, description="Affected plan identifier")
    trigger_type: str = Field(default="", description="Trigger type identifier")
    source_agent: str = Field(default="", description="Source upstream agent")
    severity: str = Field(default="medium", description="Event severity: critical, high, medium, low")
    description: str = Field(default="", description="Event description")
    risk_data: Dict[str, Any] = Field(default_factory=dict, description="Associated risk data")
    recommended_adjustment: Dict[str, Any] = Field(default_factory=dict, description="Recommended adjustment")
    adjustment_type: Optional[str] = Field(default=None, description="Adjustment type")
    acknowledged: bool = Field(default=False, description="Whether event has been acknowledged")
    acknowledged_by: Optional[str] = Field(default=None, description="User who acknowledged")
    acknowledged_at: Optional[datetime] = Field(default=None, description="Acknowledgement timestamp")
    resolved: bool = Field(default=False, description="Whether event has been resolved")
    resolved_at: Optional[datetime] = Field(default=None, description="Resolution timestamp")
    detected_at: Optional[datetime] = Field(default=None, description="Detection timestamp")


class TriggerListResponse(GreenLangBase):
    """Response for GET /monitoring/triggers."""

    triggers: List[TriggerEventEntry] = Field(default_factory=list, description="Trigger events")
    meta: PaginatedMeta = Field(..., description="Pagination metadata")


class AcknowledgeRequest(GreenLangBase):
    """Request body for PUT /monitoring/triggers/{event_id}/acknowledge."""

    notes: str = Field(default="", description="Acknowledgement notes")
    action_taken: Optional[str] = Field(default=None, description="Immediate action taken")


class MonitoringDashboardResponse(GreenLangBase):
    """Response for GET /monitoring/dashboard."""

    active_plans_count: int = Field(default=0, description="Active plans")
    unresolved_triggers_count: int = Field(default=0, description="Unresolved triggers")
    critical_triggers_count: int = Field(default=0, description="Critical unresolved triggers")
    triggers_by_severity: Dict[str, int] = Field(default_factory=dict, description="Trigger counts by severity")
    triggers_by_source: Dict[str, int] = Field(default_factory=dict, description="Trigger counts by source agent")
    plans_at_risk: List[Dict[str, Any]] = Field(default_factory=list, description="Plans flagged at risk")
    recent_triggers: List[TriggerEventEntry] = Field(default_factory=list, description="Most recent trigger events")
    next_annual_review: Optional[date] = Field(default=None, description="Next Article 8(3) annual review date")


class DriftAnalysisResponse(GreenLangBase):
    """Response for GET /monitoring/drift/{plan_id}."""

    plan_id: str = Field(..., description="Plan identifier")
    plan_name: str = Field(default="", description="Plan name")
    planned_trajectory: List[Dict[str, Any]] = Field(default_factory=list, description="Planned risk reduction trajectory")
    actual_trajectory: List[Dict[str, Any]] = Field(default_factory=list, description="Actual risk reduction trajectory")
    drift_pct: Decimal = Field(default=Decimal("0"), description="Current drift percentage")
    drift_direction: str = Field(default="on_track", description="on_track, positive_drift, negative_drift")
    recommendation: str = Field(default="", description="Drift-based recommendation")


# ---------------------------------------------------------------------------
# Cost-Benefit Optimization Schemas
# ---------------------------------------------------------------------------


class OptimizeBudgetRequest(GreenLangBase):
    """Request body for POST /optimization/run."""

    operator_id: str = Field(..., min_length=1, description="Operator identifier")
    total_budget: Decimal = Field(..., gt=Decimal("0"), description="Total budget constraint (EUR)")
    per_supplier_cap: Optional[Decimal] = Field(default=None, gt=Decimal("0"), description="Per-supplier budget cap (EUR)")
    category_budgets: Optional[Dict[str, Decimal]] = Field(default=None, description="Per-risk-category budget constraints")
    supplier_ids: Optional[List[str]] = Field(default=None, description="Specific supplier IDs to optimize for")
    scenario_name: str = Field(default="default", description="Scenario name for tracking")

    class Config:
        json_schema_extra = {
            "example": {
                "operator_id": "op-001",
                "total_budget": "500000.00",
                "per_supplier_cap": "50000.00",
                "scenario_name": "Q2_2026_allocation",
            }
        }


class OptimizationResultResponse(GreenLangBase):
    """Response for POST /optimization/run and GET /optimization/{id}."""

    optimization_id: str = Field(..., description="Optimization result identifier")
    operator_id: str = Field(default="", description="Operator identifier")
    scenario_name: str = Field(default="", description="Scenario name")
    total_budget: Decimal = Field(default=Decimal("0"), description="Total budget (EUR)")
    total_cost: Decimal = Field(default=Decimal("0"), description="Allocated cost (EUR)")
    budget_utilization_pct: Decimal = Field(default=Decimal("0"), description="Budget utilization %")
    total_predicted_risk_reduction: Decimal = Field(default=Decimal("0"), description="Predicted risk reduction")
    allocations: List[Dict[str, Any]] = Field(default_factory=list, description="Budget allocations per supplier/measure")
    solver_status: str = Field(default="", description="LP solver status")
    computation_time_ms: int = Field(default=0, description="Solver computation time in ms")
    created_at: Optional[datetime] = Field(default=None, description="Creation timestamp")


class ParetoFrontierResponse(GreenLangBase):
    """Response for GET /optimization/{id}/pareto."""

    optimization_id: str = Field(..., description="Optimization identifier")
    points: List[Dict[str, Any]] = Field(default_factory=list, description="Pareto-optimal points (budget, risk_reduction, allocation_count)")
    budget_range: Dict[str, Decimal] = Field(default_factory=dict, description="Budget range (min, max)")
    recommended_point: Optional[Dict[str, Any]] = Field(default=None, description="Recommended operating point")


class SensitivityAnalysisResponse(GreenLangBase):
    """Response for GET /optimization/{id}/sensitivity."""

    optimization_id: str = Field(..., description="Optimization identifier")
    budget_sensitivity: List[Dict[str, Any]] = Field(default_factory=list, description="Risk reduction vs budget sensitivity")
    measure_sensitivity: List[Dict[str, Any]] = Field(default_factory=list, description="Impact of adding/removing measures")
    supplier_priority_ranking: List[Dict[str, Any]] = Field(default_factory=list, description="Supplier priority by cost-effectiveness")
    constraint_binding: Dict[str, bool] = Field(default_factory=dict, description="Which constraints are binding")


# ---------------------------------------------------------------------------
# Collaboration Schemas
# ---------------------------------------------------------------------------


class PostMessageRequest(GreenLangBase):
    """Request body for POST /collaboration/{plan_id}/messages."""

    content: str = Field(..., min_length=1, description="Message content")
    message_type: str = Field(default="text", description="Message type: text, task_update, evidence_upload, system_notification")
    attachments: List[Dict[str, Any]] = Field(default_factory=list, description="Attached file references")
    mentions: List[str] = Field(default_factory=list, description="Mentioned user IDs")


class MessageEntry(GreenLangBase):
    """A collaboration message entry."""

    message_id: str = Field(default="", description="Message identifier")
    plan_id: str = Field(default="", description="Plan identifier")
    sender_id: str = Field(default="", description="Sender user identifier")
    sender_role: str = Field(default="", description="Sender stakeholder role")
    message_type: str = Field(default="text", description="Message type")
    content: str = Field(default="", description="Message content")
    attachments: List[Dict[str, Any]] = Field(default_factory=list, description="Attachments")
    mentions: List[str] = Field(default_factory=list, description="Mentioned users")
    read_by: List[str] = Field(default_factory=list, description="Users who have read this message")
    sent_at: Optional[datetime] = Field(default=None, description="Send timestamp")


class MessageListResponse(GreenLangBase):
    """Response for GET /collaboration/{plan_id}/messages."""

    messages: List[MessageEntry] = Field(default_factory=list, description="Messages in thread")
    meta: PaginatedMeta = Field(..., description="Pagination metadata")


class CreateTaskRequest(GreenLangBase):
    """Request body for POST /collaboration/{plan_id}/tasks."""

    title: str = Field(..., min_length=1, max_length=500, description="Task title")
    description: str = Field(default="", description="Task description")
    assigned_to: str = Field(..., min_length=1, description="Assignee user ID")
    assigned_role: str = Field(default="internal_compliance", description="Assignee stakeholder role")
    due_date: Optional[date] = Field(default=None, description="Task due date")
    priority: str = Field(default="medium", description="Priority: low, medium, high, urgent")


class TaskEntry(GreenLangBase):
    """A collaboration task entry."""

    task_id: str = Field(default="", description="Task identifier")
    plan_id: str = Field(default="", description="Plan identifier")
    title: str = Field(default="", description="Task title")
    description: str = Field(default="", description="Task description")
    assigned_to: str = Field(default="", description="Assignee user ID")
    assigned_role: str = Field(default="", description="Assignee role")
    due_date: Optional[date] = Field(default=None, description="Due date")
    priority: str = Field(default="medium", description="Priority")
    status: str = Field(default="pending", description="Status: pending, in_progress, completed, cancelled")
    created_by: str = Field(default="", description="Creator user ID")
    created_at: Optional[datetime] = Field(default=None, description="Creation timestamp")


class SupplierPortalResponse(GreenLangBase):
    """Response for GET /collaboration/supplier-portal/{supplier_id}."""

    supplier_id: str = Field(..., description="Supplier identifier")
    active_plans: List[PlanEntry] = Field(default_factory=list, description="Active plans for supplier")
    capacity_building: List[EnrollmentEntry] = Field(default_factory=list, description="Capacity building enrollments")
    pending_tasks: List[TaskEntry] = Field(default_factory=list, description="Tasks assigned to supplier")
    risk_score_history: List[Dict[str, Any]] = Field(default_factory=list, description="Risk score over time")
    recent_messages: List[MessageEntry] = Field(default_factory=list, description="Recent messages")
    evidence_upload_url: str = Field(default="", description="Pre-signed URL for evidence upload")


# ---------------------------------------------------------------------------
# Reporting Schemas
# ---------------------------------------------------------------------------


class GenerateReportRequest(GreenLangBase):
    """Request body for POST /reports/generate."""

    operator_id: str = Field(..., min_length=1, description="Operator identifier")
    report_type: str = Field(
        ...,
        description=(
            "Report type: dds_mitigation, authority_package, annual_review, "
            "supplier_scorecard, portfolio_summary, risk_mapping, effectiveness_analysis"
        ),
    )
    format: str = Field(default="pdf", description="Output format: pdf, json, html, xlsx, xml")
    language: str = Field(default="en", description="Report language: en, fr, de, es, pt")
    scope: Dict[str, Any] = Field(
        default_factory=dict,
        description="Report scope: supplier_ids, plan_ids, date_range, commodities",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "operator_id": "op-001",
                "report_type": "dds_mitigation",
                "format": "pdf",
                "language": "en",
                "scope": {
                    "supplier_ids": ["sup-001", "sup-002"],
                    "date_range": {"start": "2026-01-01", "end": "2026-03-31"},
                },
            }
        }


class ReportEntry(GreenLangBase):
    """A generated report entry."""

    report_id: str = Field(default="", description="Report identifier")
    operator_id: str = Field(default="", description="Operator identifier")
    report_type: str = Field(default="", description="Report type")
    format: str = Field(default="pdf", description="Output format")
    language: str = Field(default="en", description="Report language")
    report_scope: Dict[str, Any] = Field(default_factory=dict, description="Report scope")
    s3_key: Optional[str] = Field(default=None, description="S3 storage key")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    generated_at: Optional[datetime] = Field(default=None, description="Generation timestamp")


class ReportListResponse(GreenLangBase):
    """Response for GET /reports."""

    reports: List[ReportEntry] = Field(default_factory=list, description="Report list")
    meta: PaginatedMeta = Field(..., description="Pagination metadata")


class ReportDownloadResponse(GreenLangBase):
    """Response for GET /reports/{report_id}/download."""

    report_id: str = Field(..., description="Report identifier")
    download_url: str = Field(default="", description="Pre-signed download URL")
    format: str = Field(default="pdf", description="Report format")
    file_size_bytes: int = Field(default=0, description="File size in bytes")
    expires_at: Optional[datetime] = Field(default=None, description="URL expiration time")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class DDSSectionResponse(GreenLangBase):
    """Response for GET /reports/dds-section/{operator_id}."""

    operator_id: str = Field(..., description="Operator identifier")
    article_12_2_d_data: Dict[str, Any] = Field(default_factory=dict, description="Article 12(2)(d) mitigation section data")
    risk_findings_count: int = Field(default=0, description="Number of risk findings with mitigation")
    active_plans_count: int = Field(default=0, description="Active remediation plans")
    mitigation_measures_deployed: int = Field(default=0, description="Deployed mitigation measures count")
    average_risk_reduction_pct: Decimal = Field(default=Decimal("0"), description="Average risk reduction achieved")
    compliance_status: str = Field(default="", description="Overall mitigation compliance status")
    evidence_completeness_pct: Decimal = Field(default=Decimal("0"), description="Evidence completeness %")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    generated_at: Optional[datetime] = Field(default=None, description="Generation timestamp")


# ---------------------------------------------------------------------------
# Health / Stats Schemas
# ---------------------------------------------------------------------------


class HealthResponse(GreenLangBase):
    """Health check response."""

    status: str = Field(default="healthy", description="Service status")
    version: str = Field(default="1.0.0", description="Service version")
    agent_id: str = Field(default="GL-EUDR-RMA-025", description="Agent identifier")
    engines_available: int = Field(default=0, description="Number of engines initialized")
    db_connected: bool = Field(default=False, description="Database connectivity")
    redis_connected: bool = Field(default=False, description="Redis connectivity")
    timestamp: Optional[datetime] = Field(default=None, description="Check timestamp")


class StatsResponse(GreenLangBase):
    """Service statistics response."""

    total_strategies_recommended: int = Field(default=0, description="Total strategies recommended")
    total_plans_created: int = Field(default=0, description="Total plans created")
    active_plans: int = Field(default=0, description="Currently active plans")
    total_enrollments: int = Field(default=0, description="Total capacity building enrollments")
    active_enrollments: int = Field(default=0, description="Active enrollments")
    library_measures_count: int = Field(default=0, description="Measures in library")
    unresolved_triggers: int = Field(default=0, description="Unresolved trigger events")
    total_optimizations_run: int = Field(default=0, description="Optimizations executed")
    total_reports_generated: int = Field(default=0, description="Reports generated")
    average_risk_reduction_pct: Decimal = Field(default=Decimal("0"), description="Portfolio avg risk reduction")
    ml_model_version: str = Field(default="", description="Current ML model version")
    uptime_seconds: int = Field(default=0, description="Service uptime in seconds")
