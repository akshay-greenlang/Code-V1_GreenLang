# -*- coding: utf-8 -*-
"""
PACK-019 CSDDD Readiness Pack - Engine modules.

This package contains the assessment engines for the EU Corporate
Sustainability Due Diligence Directive (CSDDD / Directive 2024/1760).
"""

from .due_diligence_policy_engine import (
    ArticleReference,
    CompanyScope,
    ComplianceStatus,
    CompanyProfile,
    PolicyArea,
    PolicyAssessment,
    ArticleAssessment,
    ScopeAssessment,
    DueDiligencePolicyResult,
    DueDiligencePolicyEngine,
)

from .adverse_impact_engine import (
    AdverseImpactType,
    ImpactSeverity,
    ImpactLikelihood,
    ImpactStatus,
    ValueChainPosition,
    HumanRightsCategory,
    EnvironmentalCategory,
    RiskLevel,
    AdverseImpact,
    RiskMatrix,
    SummaryStatistics,
    ImpactAssessmentResult,
    AdverseImpactEngine,
)

from .prevention_mitigation_engine import (
    MeasureType,
    MeasureStatus,
    EffectivenessRating,
    MeasureCategory,
    PreventionMeasure,
    MeasureEffectiveness,
    BudgetSummary,
    CoverageAnalysis,
    GapAnalysis,
    EffectivenessSummary,
    PreventionResult,
    PreventionMitigationEngine,
)

from .remediation_tracking_engine import (
    RemediationStatus,
    RemediationType,
    CompanyContribution,
    VictimEngagementLevel,
    RemediationAction,
    TimelineAnalysis,
    FinancialAnalysis,
    VictimEngagementAnalysis,
    CompletenessAssessment,
    RemediationResult,
    RemediationTrackingEngine,
)

from .grievance_mechanism_engine import (
    GrievanceMechanismEngine,
    GrievanceCase,
    GrievanceResult,
    GrievanceStatus,
    GrievanceChannel,
    MechanismCriteria,
    MechanismConfig,
    MechanismAssessment,
    StakeholderGroup as GrievanceStakeholderGroup,
)

from .stakeholder_engagement_engine import (
    StakeholderEngagementEngine,
    StakeholderEngagement,
    EngagementResult,
    EngagementMethod,
    EngagementQuality,
    DueDiligenceStage,
    StakeholderGroup as EngagementStakeholderGroup,
)

from .climate_transition_engine import (
    ClimateTransitionEngine,
    ClimateTarget,
    ClimateTransitionResult,
    TransitionPlanStatus,
    EmissionScope,
    AlignmentLevel,
    TransitionElement,
    TransitionPlanDetails,
    InterimMilestone,
)

from .civil_liability_engine import (
    CivilLiabilityEngine,
    LiabilityScenario,
    CivilLiabilityResult,
    LiabilityTrigger,
    DefencePosition,
    ExposureLevel,
    ImpactSeverity as LiabilityImpactSeverity,
    ImpactDomain,
)

__all__ = [
    # Engine 1: Due Diligence Policy
    "ArticleReference",
    "CompanyScope",
    "ComplianceStatus",
    "CompanyProfile",
    "PolicyArea",
    "PolicyAssessment",
    "ArticleAssessment",
    "ScopeAssessment",
    "DueDiligencePolicyResult",
    "DueDiligencePolicyEngine",
    # Engine 2: Adverse Impact
    "AdverseImpactType",
    "ImpactSeverity",
    "ImpactLikelihood",
    "ImpactStatus",
    "ValueChainPosition",
    "HumanRightsCategory",
    "EnvironmentalCategory",
    "RiskLevel",
    "AdverseImpact",
    "RiskMatrix",
    "SummaryStatistics",
    "ImpactAssessmentResult",
    "AdverseImpactEngine",
    # Engine 3: Prevention & Mitigation
    "MeasureType",
    "MeasureStatus",
    "EffectivenessRating",
    "MeasureCategory",
    "PreventionMeasure",
    "MeasureEffectiveness",
    "BudgetSummary",
    "CoverageAnalysis",
    "GapAnalysis",
    "EffectivenessSummary",
    "PreventionResult",
    "PreventionMitigationEngine",
    # Engine 4: Remediation Tracking
    "RemediationStatus",
    "RemediationType",
    "CompanyContribution",
    "VictimEngagementLevel",
    "RemediationAction",
    "TimelineAnalysis",
    "FinancialAnalysis",
    "VictimEngagementAnalysis",
    "CompletenessAssessment",
    "RemediationResult",
    "RemediationTrackingEngine",
    # Engine 5: Grievance Mechanism
    "GrievanceMechanismEngine",
    "GrievanceCase",
    "GrievanceResult",
    "GrievanceStatus",
    "GrievanceChannel",
    "MechanismCriteria",
    "MechanismConfig",
    "MechanismAssessment",
    "GrievanceStakeholderGroup",
    # Engine 6: Stakeholder Engagement
    "StakeholderEngagementEngine",
    "StakeholderEngagement",
    "EngagementResult",
    "EngagementMethod",
    "EngagementQuality",
    "DueDiligenceStage",
    "EngagementStakeholderGroup",
    # Engine 7: Climate Transition
    "ClimateTransitionEngine",
    "ClimateTarget",
    "ClimateTransitionResult",
    "TransitionPlanStatus",
    "EmissionScope",
    "AlignmentLevel",
    "TransitionElement",
    "TransitionPlanDetails",
    "InterimMilestone",
    # Engine 8: Civil Liability
    "CivilLiabilityEngine",
    "LiabilityScenario",
    "CivilLiabilityResult",
    "LiabilityTrigger",
    "DefencePosition",
    "ExposureLevel",
    "LiabilityImpactSeverity",
    "ImpactDomain",
]
