# -*- coding: utf-8 -*-
"""
PACK-021 Net-Zero Starter Pack - Workflow Orchestration
==========================================================

Net-zero-specific workflow orchestrators for establishing GHG baselines,
setting SBTi-aligned targets, planning emission reductions, designing
offset portfolios, tracking progress, and conducting full net-zero
maturity assessments.

Workflows:
    - NetZeroOnboardingWorkflow: 4-phase baseline establishment
      with multi-source data collection, Scope 1+2+3 calculation,
      data quality scoring, and baseline report generation.

    - TargetSettingWorkflow: 4-phase SBTi target definition
      with sector analysis, pathway selection (ACA/SDA/FLAG),
      near-term and long-term target definition, and validation
      against SBTi Net-Zero Standard v1.2.

    - ReductionPlanningWorkflow: 5-phase abatement planning
      with emissions profiling, technology matching, cost analysis
      (NPV/IRR/payback), prioritisation with MACC, and phased
      roadmap generation (short/medium/long-term).

    - OffsetStrategyWorkflow: 4-phase carbon credit strategy
      with residual emissions calculation, credit screening against
      quality criteria, portfolio design with diversification, and
      compliance validation against SBTi/VCMI/Oxford Principles.

    - ProgressReviewWorkflow: 4-phase annual progress review
      with data ingestion, YoY and cumulative progress calculation,
      gap analysis with RAG status, and report generation with
      corrective action recommendations.

    - FullNetZeroAssessmentWorkflow: 6-phase master workflow
      that chains all sub-workflows, adds a maturity scorecard,
      and compiles a unified net-zero strategy document.

Author: GreenLang Team
Version: 21.0.0
"""

# ---------------------------------------------------------------------------
# Net-Zero Onboarding Workflow
# ---------------------------------------------------------------------------
from .net_zero_onboarding_workflow import (
    NetZeroOnboardingWorkflow,
    OnboardingConfig,
    OnboardingInput,
    OnboardingResult,
    ScopeBreakdown,
    DataQualityReport,
    DataQualityItem,
    BaselineReportSummary,
    FacilityRecord,
    EnergyDataRecord,
    FuelRecord,
    FleetRecord,
    ProcurementRecord,
    PhaseResult as OnboardingPhaseResult,
    PhaseStatus as OnboardingPhaseStatus,
    WorkflowStatus as OnboardingWorkflowStatus,
)

# ---------------------------------------------------------------------------
# Target Setting Workflow
# ---------------------------------------------------------------------------
from .target_setting_workflow import (
    TargetSettingWorkflow,
    TargetSettingConfig,
    TargetSettingResult,
    BaselineEmissions,
    SectorAnalysisResult,
    PathwayDetail,
    TargetDefinition,
    Milestone,
    ValidationFinding,
    ValidationReport,
    PhaseResult as TargetPhaseResult,
    PhaseStatus as TargetPhaseStatus,
    WorkflowStatus as TargetWorkflowStatus,
)

# ---------------------------------------------------------------------------
# Reduction Planning Workflow
# ---------------------------------------------------------------------------
from .reduction_planning_workflow import (
    ReductionPlanningWorkflow,
    ReductionPlanningConfig,
    ReductionPlanningResult,
    EmissionsProfile,
    EmissionSource,
    AbatementAction,
    MACCDataPoint,
    RoadmapPhase,
    PhaseResult as ReductionPhaseResult,
    PhaseStatus as ReductionPhaseStatus,
    WorkflowStatus as ReductionWorkflowStatus,
)

# ---------------------------------------------------------------------------
# Offset Strategy Workflow
# ---------------------------------------------------------------------------
from .offset_strategy_workflow import (
    OffsetStrategyWorkflow,
    OffsetStrategyConfig,
    OffsetStrategyResult,
    ResidualBudget,
    CreditScreeningResult,
    PortfolioAllocation,
    PortfolioDesign,
    ComplianceFinding,
    ComplianceReport,
    PhaseResult as OffsetPhaseResult,
    PhaseStatus as OffsetPhaseStatus,
    WorkflowStatus as OffsetWorkflowStatus,
)

# ---------------------------------------------------------------------------
# Progress Review Workflow
# ---------------------------------------------------------------------------
from .progress_review_workflow import (
    ProgressReviewWorkflow,
    ProgressReviewConfig,
    ProgressReviewResult,
    AnnualEmissions,
    TargetPathwayPoint,
    YearOverYearChange,
    CumulativeProgress,
    IntensityMetric,
    GapAnalysisResult,
    TrendDataPoint,
    CorrectiveAction,
    ProgressSummary,
    RAGStatus,
    TrendDirection,
    PhaseResult as ProgressPhaseResult,
    PhaseStatus as ProgressPhaseStatus,
    WorkflowStatus as ProgressWorkflowStatus,
)

# ---------------------------------------------------------------------------
# Full Net-Zero Assessment Workflow (Master)
# ---------------------------------------------------------------------------
from .full_net_zero_assessment_workflow import (
    FullNetZeroAssessmentWorkflow,
    FullAssessmentConfig,
    FullAssessmentResult,
    NetZeroScorecard,
    ScorecardDimension,
    StrategySummary,
    MaturityLevel,
    PhaseResult as AssessmentPhaseResult,
    PhaseStatus as AssessmentPhaseStatus,
    WorkflowStatus as AssessmentWorkflowStatus,
)

__all__ = [
    # --- Net-Zero Onboarding Workflow ---
    "NetZeroOnboardingWorkflow",
    "OnboardingConfig",
    "OnboardingInput",
    "OnboardingResult",
    "ScopeBreakdown",
    "DataQualityReport",
    "DataQualityItem",
    "BaselineReportSummary",
    "FacilityRecord",
    "EnergyDataRecord",
    "FuelRecord",
    "FleetRecord",
    "ProcurementRecord",
    # --- Target Setting Workflow ---
    "TargetSettingWorkflow",
    "TargetSettingConfig",
    "TargetSettingResult",
    "BaselineEmissions",
    "SectorAnalysisResult",
    "PathwayDetail",
    "TargetDefinition",
    "Milestone",
    "ValidationFinding",
    "ValidationReport",
    # --- Reduction Planning Workflow ---
    "ReductionPlanningWorkflow",
    "ReductionPlanningConfig",
    "ReductionPlanningResult",
    "EmissionsProfile",
    "EmissionSource",
    "AbatementAction",
    "MACCDataPoint",
    "RoadmapPhase",
    # --- Offset Strategy Workflow ---
    "OffsetStrategyWorkflow",
    "OffsetStrategyConfig",
    "OffsetStrategyResult",
    "ResidualBudget",
    "CreditScreeningResult",
    "PortfolioAllocation",
    "PortfolioDesign",
    "ComplianceFinding",
    "ComplianceReport",
    # --- Progress Review Workflow ---
    "ProgressReviewWorkflow",
    "ProgressReviewConfig",
    "ProgressReviewResult",
    "AnnualEmissions",
    "TargetPathwayPoint",
    "YearOverYearChange",
    "CumulativeProgress",
    "IntensityMetric",
    "GapAnalysisResult",
    "TrendDataPoint",
    "CorrectiveAction",
    "ProgressSummary",
    "RAGStatus",
    "TrendDirection",
    # --- Full Net-Zero Assessment Workflow ---
    "FullNetZeroAssessmentWorkflow",
    "FullAssessmentConfig",
    "FullAssessmentResult",
    "NetZeroScorecard",
    "ScorecardDimension",
    "StrategySummary",
    "MaturityLevel",
]
