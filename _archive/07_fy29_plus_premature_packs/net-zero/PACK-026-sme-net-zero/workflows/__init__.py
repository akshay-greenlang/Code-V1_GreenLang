# -*- coding: utf-8 -*-
"""
PACK-026 SME Net Zero Pack - Workflow Layer
================================================

6 SME-friendly workflows optimised for time-constrained small and
medium enterprises.  All workflows use short phases with clear
progress indicators and mobile-friendly summaries.

Workflows:
    1. ExpressOnboardingWorkflow (4 phases, 15-20 min)
       Profile -> Quick Baseline -> Auto-Target -> Quick Wins

    2. StandardSetupWorkflow (6 phases, 1-2 hours)
       Profile -> Data Collection -> Silver Baseline -> Target Validation
       -> Action Prioritisation -> Grant Matching

    3. GrantApplicationWorkflow (5 phases)
       Grant Search -> Eligibility Check -> Data Preparation
       -> Application Support -> Submission Export

    4. QuarterlyReviewWorkflow (3 phases, 15-30 min)
       Data Update -> Progress Calculation -> Reporting

    5. QuickWinsImplementationWorkflow (5 phases)
       Action Selection -> Vendor Research -> Cost-Benefit
       -> Implementation -> Verification

    6. CertificationPathwayWorkflow (6 phases)
       Pathway Selection -> Readiness Assessment -> Gap Closure
       -> Documentation -> Submission -> Verification Tracking

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-026 SME Net Zero Pack
Status: Production Ready
"""

__version__ = "1.0.0"
__pack_id__ = "PACK-026"
__pack_name__ = "SME Net Zero Pack"

# ---------------------------------------------------------------------------
# Express Onboarding Workflow
# ---------------------------------------------------------------------------
from .express_onboarding_workflow import (
    ExpressOnboardingWorkflow,
    ExpressOnboardingConfig,
    ExpressOnboardingInput,
    ExpressOnboardingResult,
    SMEOrganizationProfile,
    BronzeBaselineInput,
    BronzeBaseline,
    AutoTarget,
    QuickWinAction,
    BaselineTier,
    IndustrySector,
    CompanySizeBand,
    QuickWinCategory,
    PhaseResult as ExpressPhaseResult,
    PhaseStatus as ExpressPhaseStatus,
    WorkflowStatus as ExpressWorkflowStatus,
)

# ---------------------------------------------------------------------------
# Standard Setup Workflow
# ---------------------------------------------------------------------------
from .standard_setup_workflow import (
    StandardSetupWorkflow,
    StandardSetupConfig,
    StandardSetupInput,
    StandardSetupResult,
    SilverBaseline,
    ValidatedTarget,
    MACCAction,
    GrantMatch,
    EnergyBillRecord,
    FuelRecord as StandardFuelRecord,
    TravelRecord,
    ProcurementCategory,
    DataSourceType,
    DataQualityLevel,
    MACCActionType,
    GrantStatus,
    PhaseResult as StandardPhaseResult,
    PhaseStatus as StandardPhaseStatus,
    WorkflowStatus as StandardWorkflowStatus,
)

# ---------------------------------------------------------------------------
# Grant Application Workflow
# ---------------------------------------------------------------------------
from .grant_application_workflow import (
    GrantApplicationWorkflow,
    GrantApplicationConfig,
    GrantApplicationInput,
    GrantApplicationResult,
    GrantSearchCriteria,
    GrantSearchResult,
    EligibilityCheckResult,
    ProjectDescription,
    ApplicationTemplate,
    SubmissionPackage,
    GrantType,
    EligibilityStatus,
    ApplicationStatus,
    DocumentType,
    PhaseResult as GrantPhaseResult,
    PhaseStatus as GrantPhaseStatus,
    WorkflowStatus as GrantWorkflowStatus,
)

# ---------------------------------------------------------------------------
# Quarterly Review Workflow
# ---------------------------------------------------------------------------
from .quarterly_review_workflow import (
    QuarterlyReviewWorkflow,
    QuarterlyReviewConfig,
    QuarterlyReviewInput,
    QuarterlyReviewResult,
    QuarterlySpendUpdate,
    QuarterlyEmissions,
    ProgressMetrics,
    QuickWinProgress,
    BoardBrief,
    TargetPathwayPoint,
    RAGStatus,
    TrendDirection,
    Quarter,
    PhaseResult as QuarterlyPhaseResult,
    PhaseStatus as QuarterlyPhaseStatus,
    WorkflowStatus as QuarterlyWorkflowStatus,
)

# ---------------------------------------------------------------------------
# Quick Wins Implementation Workflow
# ---------------------------------------------------------------------------
from .quick_wins_implementation_workflow import (
    QuickWinsImplementationWorkflow,
    QuickWinsImplementationConfig,
    QuickWinsImplementationInput,
    QuickWinsImplementationResult,
    QuickWinCandidate,
    VendorOption,
    CostBenefitDetail,
    ImplementationMilestone,
    ImplementationPlan,
    VerificationReport,
    ActionStatus,
    VendorStatus,
    VerificationResult,
    PhaseResult as QuickWinsPhaseResult,
    PhaseStatus as QuickWinsPhaseStatus,
    WorkflowStatus as QuickWinsWorkflowStatus,
)

# ---------------------------------------------------------------------------
# Certification Pathway Workflow
# ---------------------------------------------------------------------------
from .certification_pathway_workflow import (
    CertificationPathwayWorkflow,
    CertificationPathwayConfig,
    CertificationPathwayInput,
    CertificationPathwayResult,
    ReadinessScorecard,
    CriterionAssessment,
    GapClosureAction,
    CertificationDocument,
    SubmissionRecord,
    VerificationTracking,
    CertificationType,
    ReadinessLevel,
    GapSeverity,
    DocumentStatus,
    CertificationStatus,
    PhaseResult as CertPhaseResult,
    PhaseStatus as CertPhaseStatus,
    WorkflowStatus as CertWorkflowStatus,
)

__all__ = [
    "__version__",
    "__pack_id__",
    "__pack_name__",
    # --- Express Onboarding Workflow ---
    "ExpressOnboardingWorkflow",
    "ExpressOnboardingConfig",
    "ExpressOnboardingInput",
    "ExpressOnboardingResult",
    "SMEOrganizationProfile",
    "BronzeBaselineInput",
    "BronzeBaseline",
    "AutoTarget",
    "QuickWinAction",
    "BaselineTier",
    "IndustrySector",
    "CompanySizeBand",
    "QuickWinCategory",
    # --- Standard Setup Workflow ---
    "StandardSetupWorkflow",
    "StandardSetupConfig",
    "StandardSetupInput",
    "StandardSetupResult",
    "SilverBaseline",
    "ValidatedTarget",
    "MACCAction",
    "GrantMatch",
    "EnergyBillRecord",
    "StandardFuelRecord",
    "TravelRecord",
    "ProcurementCategory",
    "DataSourceType",
    "DataQualityLevel",
    "MACCActionType",
    "GrantStatus",
    # --- Grant Application Workflow ---
    "GrantApplicationWorkflow",
    "GrantApplicationConfig",
    "GrantApplicationInput",
    "GrantApplicationResult",
    "GrantSearchCriteria",
    "GrantSearchResult",
    "EligibilityCheckResult",
    "ProjectDescription",
    "ApplicationTemplate",
    "SubmissionPackage",
    "GrantType",
    "EligibilityStatus",
    "ApplicationStatus",
    "DocumentType",
    # --- Quarterly Review Workflow ---
    "QuarterlyReviewWorkflow",
    "QuarterlyReviewConfig",
    "QuarterlyReviewInput",
    "QuarterlyReviewResult",
    "QuarterlySpendUpdate",
    "QuarterlyEmissions",
    "ProgressMetrics",
    "QuickWinProgress",
    "BoardBrief",
    "TargetPathwayPoint",
    "RAGStatus",
    "TrendDirection",
    "Quarter",
    # --- Quick Wins Implementation Workflow ---
    "QuickWinsImplementationWorkflow",
    "QuickWinsImplementationConfig",
    "QuickWinsImplementationInput",
    "QuickWinsImplementationResult",
    "QuickWinCandidate",
    "VendorOption",
    "CostBenefitDetail",
    "ImplementationMilestone",
    "ImplementationPlan",
    "VerificationReport",
    "ActionStatus",
    "VendorStatus",
    "VerificationResult",
    # --- Certification Pathway Workflow ---
    "CertificationPathwayWorkflow",
    "CertificationPathwayConfig",
    "CertificationPathwayInput",
    "CertificationPathwayResult",
    "ReadinessScorecard",
    "CriterionAssessment",
    "GapClosureAction",
    "CertificationDocument",
    "SubmissionRecord",
    "VerificationTracking",
    "CertificationType",
    "ReadinessLevel",
    "GapSeverity",
    "DocumentStatus",
    "CertificationStatus",
]
