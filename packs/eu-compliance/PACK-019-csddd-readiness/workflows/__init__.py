# -*- coding: utf-8 -*-
"""
PACK-019 CSDDD Readiness Pack - Workflows
==========================================

Eight production workflows implementing the full CSDDD due diligence lifecycle:

1. DueDiligenceAssessmentWorkflow  - 5-phase scope, policy, gap, risk, readiness
2. ValueChainMappingWorkflow       - 4-phase supplier, tier, activity, risk
3. ImpactAssessmentWorkflow        - 4-phase scan, score, prioritize, validate
4. PreventionPlanningWorkflow      - 4-phase design, resource, timeline, KPIs
5. GrievanceManagementWorkflow     - 4-phase mechanism, channel, case, resolution
6. MonitoringReviewWorkflow        - 4-phase KPI, data, analysis, review
7. ClimateTransitionPlanningWorkflow - 4-phase baseline, target, pathway, progress
8. RegulatorySubmissionWorkflow    - 4-phase docs, readiness, package, tracking
"""

from .due_diligence_assessment_workflow import (
    DueDiligenceAssessmentWorkflow,
    DueDiligenceAssessmentInput,
    DueDiligenceAssessmentResult,
    CompanyProfile,
    PolicyRecord,
    RiskIndicator,
    ArticleAssessment,
)
from .value_chain_mapping_workflow import (
    ValueChainMappingWorkflow,
    ValueChainMappingInput,
    ValueChainMappingResult,
    SupplierRecord,
    CountryRiskData,
    SupplierRiskProfile,
)
from .impact_assessment_workflow import (
    ImpactAssessmentWorkflow,
    ImpactAssessmentInput,
    ImpactAssessmentResult,
    AdverseImpact,
    StakeholderEngagement,
    ScoredImpact,
)
from .prevention_planning_workflow import (
    PreventionPlanningWorkflow,
    PreventionPlanningInput,
    PreventionPlanningResult,
    PrioritizedImpact,
    ExistingMeasure,
    PreventionMeasure,
)
from .grievance_management_workflow import (
    GrievanceManagementWorkflow,
    GrievanceManagementInput,
    GrievanceManagementResult,
    MechanismConfig,
    GrievanceCase,
    StakeholderGroup,
)
from .monitoring_review_workflow import (
    MonitoringReviewWorkflow,
    MonitoringReviewInput,
    MonitoringReviewResult,
    KPIDefinition,
    MonitoringDataPoint,
    PreviousReview,
    KPIResult,
)
from .climate_transition_planning_workflow import (
    ClimateTransitionPlanningWorkflow,
    ClimateTransitionPlanningInput,
    ClimateTransitionPlanningResult,
    EmissionData,
    ClimateTarget,
    TransitionAction,
)
from .regulatory_submission_workflow import (
    RegulatorySubmissionWorkflow,
    RegulatorySubmissionInput,
    RegulatorySubmissionResult,
    DDWorkflowResult,
    CompanySubmissionProfile,
    ComplianceItem,
)

__all__ = [
    # Workflow 1: Due Diligence Assessment
    "DueDiligenceAssessmentWorkflow",
    "DueDiligenceAssessmentInput",
    "DueDiligenceAssessmentResult",
    "CompanyProfile",
    "PolicyRecord",
    "RiskIndicator",
    "ArticleAssessment",
    # Workflow 2: Value Chain Mapping
    "ValueChainMappingWorkflow",
    "ValueChainMappingInput",
    "ValueChainMappingResult",
    "SupplierRecord",
    "CountryRiskData",
    "SupplierRiskProfile",
    # Workflow 3: Impact Assessment
    "ImpactAssessmentWorkflow",
    "ImpactAssessmentInput",
    "ImpactAssessmentResult",
    "AdverseImpact",
    "StakeholderEngagement",
    "ScoredImpact",
    # Workflow 4: Prevention Planning
    "PreventionPlanningWorkflow",
    "PreventionPlanningInput",
    "PreventionPlanningResult",
    "PrioritizedImpact",
    "ExistingMeasure",
    "PreventionMeasure",
    # Workflow 5: Grievance Management
    "GrievanceManagementWorkflow",
    "GrievanceManagementInput",
    "GrievanceManagementResult",
    "MechanismConfig",
    "GrievanceCase",
    "StakeholderGroup",
    # Workflow 6: Monitoring Review
    "MonitoringReviewWorkflow",
    "MonitoringReviewInput",
    "MonitoringReviewResult",
    "KPIDefinition",
    "MonitoringDataPoint",
    "PreviousReview",
    "KPIResult",
    # Workflow 7: Climate Transition Planning
    "ClimateTransitionPlanningWorkflow",
    "ClimateTransitionPlanningInput",
    "ClimateTransitionPlanningResult",
    "EmissionData",
    "ClimateTarget",
    "TransitionAction",
    # Workflow 8: Regulatory Submission
    "RegulatorySubmissionWorkflow",
    "RegulatorySubmissionInput",
    "RegulatorySubmissionResult",
    "DDWorkflowResult",
    "CompanySubmissionProfile",
    "ComplianceItem",
]
