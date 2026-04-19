# -*- coding: utf-8 -*-
"""
PACK-024 Carbon Neutral Pack - Workflow Layer
=================================================

8 workflows implementing the complete carbon neutrality lifecycle from
footprint assessment through credit procurement, retirement, neutralization
balance, claims validation, third-party verification, and full annual cycle.

Workflows:
    1. FootprintAssessmentWorkflow       -- 4 phases
    2. CarbonMgmtPlanWorkflow            -- 5 phases
    3. CreditProcurementWorkflow         -- 4 phases
    4. RetirementWorkflow                -- 3 phases
    5. NeutralizationWorkflow            -- 5 phases
    6. ClaimsValidationWorkflow          -- 4 phases
    7. VerificationWorkflow              -- 4 phases
    8. FullAnnualCycleWorkflow           -- 10 phases

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-024 Carbon Neutral Pack
Status: Production Ready
"""

__version__ = "1.0.0"
__pack_id__ = "PACK-024"
__pack_name__ = "Carbon Neutral Pack"

from .footprint_assessment_workflow import (
    FootprintAssessmentWorkflow,
    FootprintAssessmentConfig,
    FootprintAssessmentResult,
    FootprintPhase,
    BoundaryApproach,
    EmissionScope,
    DataQualityTier,
    ScopeBreakdown,
    CategoryBreakdown,
    DataQualityAssessment,
)

from .carbon_mgmt_plan_workflow import (
    CarbonMgmtPlanWorkflow,
    CarbonMgmtPlanConfig,
    CarbonMgmtPlanResult,
    MgmtPlanPhase,
    ReductionStrategy,
    ReductionTarget,
    AbatementAction,
    ResidualProfile,
    MgmtPlanTimeline,
)

from .credit_procurement_workflow import (
    CreditProcurementWorkflow,
    CreditProcurementConfig,
    CreditProcurementResult,
    ProcurementPhase,
    CreditStandard,
    CreditType,
    QualityTier,
    ProcurementStrategy,
    SupplierEvaluation,
    ContractTerms,
)

from .retirement_workflow import (
    RetirementWorkflow,
    RetirementConfig,
    RetirementResult,
    RetirementPhase,
    RegistryType,
    RetirementStatus,
    RetirementRecord,
    RegistryConfirmation,
    RetirementBatch,
)

from .neutralization_workflow import (
    NeutralizationWorkflow,
    NeutralizationConfig,
    NeutralizationResult,
    NeutralizationPhase,
    NeutralizationStatus,
    BalanceSheet,
    EmissionsCoverage,
    GapAnalysis,
    NeutralizationEvidence,
)

from .claims_validation_workflow import (
    ClaimsValidationWorkflow,
    ClaimsValidationConfig,
    ClaimsValidationResult,
    ClaimsPhase,
    ClaimType,
    ClaimStatus,
    ClaimAssessment,
    ComplianceCheck,
    SubstantiationEvidence,
)

from .verification_workflow import (
    VerificationWorkflow,
    VerificationConfig,
    VerificationResult,
    VerificationPhase,
    AssuranceLevel,
    VerificationStandard,
    FindingCategory,
    VerificationFinding,
    VerificationOpinion,
)

from .full_annual_cycle_workflow import (
    FullAnnualCycleWorkflow,
    AnnualCycleConfig,
    AnnualCycleResult,
    AnnualCyclePhase,
    CycleStatus,
    PhaseGate,
    CycleMetrics,
    AnnualSummary,
)

__all__ = [
    "__version__",
    "__pack_id__",
    "__pack_name__",
    # Footprint Assessment
    "FootprintAssessmentWorkflow",
    "FootprintAssessmentConfig",
    "FootprintAssessmentResult",
    "FootprintPhase",
    "BoundaryApproach",
    "EmissionScope",
    "DataQualityTier",
    "ScopeBreakdown",
    "CategoryBreakdown",
    "DataQualityAssessment",
    # Carbon Management Plan
    "CarbonMgmtPlanWorkflow",
    "CarbonMgmtPlanConfig",
    "CarbonMgmtPlanResult",
    "MgmtPlanPhase",
    "ReductionStrategy",
    "ReductionTarget",
    "AbatementAction",
    "ResidualProfile",
    "MgmtPlanTimeline",
    # Credit Procurement
    "CreditProcurementWorkflow",
    "CreditProcurementConfig",
    "CreditProcurementResult",
    "ProcurementPhase",
    "CreditStandard",
    "CreditType",
    "QualityTier",
    "ProcurementStrategy",
    "SupplierEvaluation",
    "ContractTerms",
    # Retirement
    "RetirementWorkflow",
    "RetirementConfig",
    "RetirementResult",
    "RetirementPhase",
    "RegistryType",
    "RetirementStatus",
    "RetirementRecord",
    "RegistryConfirmation",
    "RetirementBatch",
    # Neutralization
    "NeutralizationWorkflow",
    "NeutralizationConfig",
    "NeutralizationResult",
    "NeutralizationPhase",
    "NeutralizationStatus",
    "BalanceSheet",
    "EmissionsCoverage",
    "GapAnalysis",
    "NeutralizationEvidence",
    # Claims Validation
    "ClaimsValidationWorkflow",
    "ClaimsValidationConfig",
    "ClaimsValidationResult",
    "ClaimsPhase",
    "ClaimType",
    "ClaimStatus",
    "ClaimAssessment",
    "ComplianceCheck",
    "SubstantiationEvidence",
    # Verification
    "VerificationWorkflow",
    "VerificationConfig",
    "VerificationResult",
    "VerificationPhase",
    "AssuranceLevel",
    "VerificationStandard",
    "FindingCategory",
    "VerificationFinding",
    "VerificationOpinion",
    # Full Annual Cycle
    "FullAnnualCycleWorkflow",
    "AnnualCycleConfig",
    "AnnualCycleResult",
    "AnnualCyclePhase",
    "CycleStatus",
    "PhaseGate",
    "CycleMetrics",
    "AnnualSummary",
]
