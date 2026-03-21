# -*- coding: utf-8 -*-
"""
PACK-019 CSDDD Readiness Pack - Workflows Module
=====================================================

Eight production workflows implementing the full CSDDD due diligence lifecycle:

1. DueDiligenceAssessmentWorkflow   - 5-phase scope, policy, gap, risk, readiness
2. ValueChainMappingWorkflow        - 4-phase supplier, tier, activity, risk
3. ImpactAssessmentWorkflow         - 4-phase scan, score, prioritize, validate
4. PreventionPlanningWorkflow       - 4-phase design, resource, timeline, KPIs
5. GrievanceManagementWorkflow      - 4-phase mechanism, channel, case, resolution
6. MonitoringReviewWorkflow         - 4-phase KPI, data, analysis, review
7. ClimateTransitionPlanningWorkflow - 4-phase baseline, target, pathway, progress
8. RegulatorySubmissionWorkflow     - 4-phase docs, readiness, package, tracking

Author: GreenLang Team
Version: 19.0.0
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

__version__: str = "19.0.0"
__pack__: str = "PACK-019"

_loaded_workflows: List[str] = []

# ---------------------------------------------------------------------------
# Workflow 1: Due Diligence Assessment
# ---------------------------------------------------------------------------
try:
    from .due_diligence_assessment_workflow import (
        DueDiligenceAssessmentWorkflow,
        DueDiligenceAssessmentInput,
        DueDiligenceAssessmentResult,
        CompanyProfile,
        PolicyRecord,
        RiskIndicator,
        ArticleAssessment,
    )
    _loaded_workflows.append("DueDiligenceAssessmentWorkflow")
except ImportError as e:
    DueDiligenceAssessmentWorkflow = None  # type: ignore[assignment,misc]
    logger.debug("DueDiligenceAssessmentWorkflow not available: %s", e)

# ---------------------------------------------------------------------------
# Workflow 2: Value Chain Mapping
# ---------------------------------------------------------------------------
try:
    from .value_chain_mapping_workflow import (
        ValueChainMappingWorkflow,
        ValueChainMappingInput,
        ValueChainMappingResult,
        SupplierRecord,
        CountryRiskData,
        SupplierRiskProfile,
    )
    _loaded_workflows.append("ValueChainMappingWorkflow")
except ImportError as e:
    ValueChainMappingWorkflow = None  # type: ignore[assignment,misc]
    logger.debug("ValueChainMappingWorkflow not available: %s", e)

# ---------------------------------------------------------------------------
# Workflow 3: Impact Assessment
# ---------------------------------------------------------------------------
try:
    from .impact_assessment_workflow import (
        ImpactAssessmentWorkflow,
        ImpactAssessmentInput,
        ImpactAssessmentResult,
        AdverseImpact,
        StakeholderEngagement,
        ScoredImpact,
    )
    _loaded_workflows.append("ImpactAssessmentWorkflow")
except ImportError as e:
    ImpactAssessmentWorkflow = None  # type: ignore[assignment,misc]
    logger.debug("ImpactAssessmentWorkflow not available: %s", e)

# ---------------------------------------------------------------------------
# Workflow 4: Prevention Planning
# ---------------------------------------------------------------------------
try:
    from .prevention_planning_workflow import (
        PreventionPlanningWorkflow,
        PreventionPlanningInput,
        PreventionPlanningResult,
        PrioritizedImpact,
        ExistingMeasure,
        PreventionMeasure,
    )
    _loaded_workflows.append("PreventionPlanningWorkflow")
except ImportError as e:
    PreventionPlanningWorkflow = None  # type: ignore[assignment,misc]
    logger.debug("PreventionPlanningWorkflow not available: %s", e)

# ---------------------------------------------------------------------------
# Workflow 5: Grievance Management
# ---------------------------------------------------------------------------
try:
    from .grievance_management_workflow import (
        GrievanceManagementWorkflow,
        GrievanceManagementInput,
        GrievanceManagementResult,
        MechanismConfig,
        GrievanceCase,
        StakeholderGroup,
    )
    _loaded_workflows.append("GrievanceManagementWorkflow")
except ImportError as e:
    GrievanceManagementWorkflow = None  # type: ignore[assignment,misc]
    logger.debug("GrievanceManagementWorkflow not available: %s", e)

# ---------------------------------------------------------------------------
# Workflow 6: Monitoring Review
# ---------------------------------------------------------------------------
try:
    from .monitoring_review_workflow import (
        MonitoringReviewWorkflow,
        MonitoringReviewInput,
        MonitoringReviewResult,
        KPIDefinition,
        MonitoringDataPoint,
        PreviousReview,
        KPIResult,
    )
    _loaded_workflows.append("MonitoringReviewWorkflow")
except ImportError as e:
    MonitoringReviewWorkflow = None  # type: ignore[assignment,misc]
    logger.debug("MonitoringReviewWorkflow not available: %s", e)

# ---------------------------------------------------------------------------
# Workflow 7: Climate Transition Planning
# ---------------------------------------------------------------------------
try:
    from .climate_transition_planning_workflow import (
        ClimateTransitionPlanningWorkflow,
        ClimateTransitionPlanningInput,
        ClimateTransitionPlanningResult,
        EmissionData,
        ClimateTarget,
        TransitionAction,
    )
    _loaded_workflows.append("ClimateTransitionPlanningWorkflow")
except ImportError as e:
    ClimateTransitionPlanningWorkflow = None  # type: ignore[assignment,misc]
    logger.debug("ClimateTransitionPlanningWorkflow not available: %s", e)

# ---------------------------------------------------------------------------
# Workflow 8: Regulatory Submission
# ---------------------------------------------------------------------------
try:
    from .regulatory_submission_workflow import (
        RegulatorySubmissionWorkflow,
        RegulatorySubmissionInput,
        RegulatorySubmissionResult,
        DDWorkflowResult,
        CompanySubmissionProfile,
        ComplianceItem,
    )
    _loaded_workflows.append("RegulatorySubmissionWorkflow")
except ImportError as e:
    RegulatorySubmissionWorkflow = None  # type: ignore[assignment,misc]
    logger.debug("RegulatorySubmissionWorkflow not available: %s", e)


# ---------------------------------------------------------------------------
# Dynamic __all__
# ---------------------------------------------------------------------------

__all__: List[str] = [
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
    # Utility
    "get_loaded_workflows",
    "get_workflow_count",
    "get_workflow_mapping",
]


def get_loaded_workflows() -> List[str]:
    """Return list of successfully loaded workflow class names."""
    return list(_loaded_workflows)


def get_workflow_count() -> int:
    """Return count of loaded workflows."""
    return len(_loaded_workflows)


def get_workflow_mapping() -> Dict[str, str]:
    """Return mapping of workflow purpose to workflow class name."""
    return {
        "due_diligence_assessment": "DueDiligenceAssessmentWorkflow",
        "value_chain_mapping": "ValueChainMappingWorkflow",
        "impact_assessment": "ImpactAssessmentWorkflow",
        "prevention_planning": "PreventionPlanningWorkflow",
        "grievance_management": "GrievanceManagementWorkflow",
        "monitoring_review": "MonitoringReviewWorkflow",
        "climate_transition_planning": "ClimateTransitionPlanningWorkflow",
        "regulatory_submission": "RegulatorySubmissionWorkflow",
    }
