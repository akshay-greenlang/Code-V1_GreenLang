# -*- coding: utf-8 -*-
"""
PACK-045 Base Year Management Pack - Workflow Orchestration
===============================================================

Complete base year management workflow orchestrators for the
GHG Protocol Corporate Standard Chapter 5, ISO 14064-1:2018 Clause 9,
and SBTi target recalculation requirements. Each workflow coordinates
base year management processes covering initial establishment, trigger
assessment, recalculation execution, target rebasing, audit verification,
annual review, M&A adjustments, and full end-to-end pipeline orchestration.

Workflows:
    - BaseYearEstablishmentWorkflow: 5-phase workflow for initial base year
      selection with candidate assessment, data quality scoring, multi-criteria
      selection, inventory snapshot, and documentation generation.

    - RecalculationAssessmentWorkflow: 4-phase workflow for trigger
      identification with trigger detection, significance testing, policy
      compliance verification, and recommendation generation.

    - RecalculationExecutionWorkflow: 5-phase workflow for approved
      recalculation with adjustment calculation, impact validation, approval
      collection, adjustment application, and audit recording.

    - TargetRebasingWorkflow: 4-phase workflow for target adjustment with
      impact assessment, target recalculation preserving ambition, stakeholder
      notification, and official target update.

    - AuditVerificationWorkflow: 4-phase workflow for third-party verification
      with evidence collection, completeness checking against ISO 14064-3,
      package generation, and verifier support material preparation.

    - AnnualReviewWorkflow: 4-phase annual review with policy review,
      trigger scanning, time series consistency checking, and report
      generation.

    - MergerAcquisitionWorkflow: 5-phase M&A-specific workflow with entity
      identification, emission quantification, pro-rata temporal allocation,
      significance testing, and adjustment execution.

    - FullBaseYearPipelineWorkflow: 10-phase end-to-end orchestrator
      invoking all sub-workflows with milestone tracking, time series
      validation, audit preparation, and comprehensive reporting.

Author: GreenLang Team
Version: 45.0.0
"""

from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Base Year Establishment Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_045_base_year.workflows.base_year_establishment_workflow import (
        BaseYearEstablishmentWorkflow,
        BaseYearEstablishmentInput,
        BaseYearEstablishmentResult,
        CandidateYear,
        ScopeEmissions,
        CandidateAssessmentResult,
        InventorySnapshot,
        GeneratedDocument,
        SelectionWeights,
        QualityScore,
        EstablishmentPhase,
        QualityDimension,
        CandidateStatus,
        DocumentType,
    )
except ImportError:
    BaseYearEstablishmentWorkflow = None  # type: ignore[assignment,misc]
    BaseYearEstablishmentInput = None  # type: ignore[assignment,misc]
    BaseYearEstablishmentResult = None  # type: ignore[assignment,misc]
    CandidateYear = None  # type: ignore[assignment,misc]
    ScopeEmissions = None  # type: ignore[assignment,misc]
    CandidateAssessmentResult = None  # type: ignore[assignment,misc]
    InventorySnapshot = None  # type: ignore[assignment,misc]
    GeneratedDocument = None  # type: ignore[assignment,misc]
    SelectionWeights = None  # type: ignore[assignment,misc]
    QualityScore = None  # type: ignore[assignment,misc]
    EstablishmentPhase = None  # type: ignore[assignment,misc]
    QualityDimension = None  # type: ignore[assignment,misc]
    CandidateStatus = None  # type: ignore[assignment,misc]
    DocumentType = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Recalculation Assessment Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_045_base_year.workflows.recalculation_assessment_workflow import (
        RecalculationAssessmentWorkflow,
        RecalculationAssessmentInput,
        RecalculationAssessmentResult,
        ExternalEvent,
        EmissionsInventory,
        RecalculationPolicy,
        DetectedTrigger,
        SignificanceResult,
        PolicyComplianceResult,
        Recommendation,
        AssessmentPhase,
        TriggerType,
        TriggerCategory,
        SignificanceLevel,
        ComplianceStatus,
        RecommendationPriority,
    )
except ImportError:
    RecalculationAssessmentWorkflow = None  # type: ignore[assignment,misc]
    RecalculationAssessmentInput = None  # type: ignore[assignment,misc]
    RecalculationAssessmentResult = None  # type: ignore[assignment,misc]
    ExternalEvent = None  # type: ignore[assignment,misc]
    EmissionsInventory = None  # type: ignore[assignment,misc]
    RecalculationPolicy = None  # type: ignore[assignment,misc]
    DetectedTrigger = None  # type: ignore[assignment,misc]
    SignificanceResult = None  # type: ignore[assignment,misc]
    PolicyComplianceResult = None  # type: ignore[assignment,misc]
    Recommendation = None  # type: ignore[assignment,misc]
    AssessmentPhase = None  # type: ignore[assignment,misc]
    TriggerType = None  # type: ignore[assignment,misc]
    TriggerCategory = None  # type: ignore[assignment,misc]
    SignificanceLevel = None  # type: ignore[assignment,misc]
    ComplianceStatus = None  # type: ignore[assignment,misc]
    RecommendationPriority = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Recalculation Execution Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_045_base_year.workflows.recalculation_execution_workflow import (
        RecalculationExecutionWorkflow,
        RecalculationExecutionInput,
        RecalculationExecutionResult,
        ApprovedTrigger,
        BaseYearInventory,
        AdjustmentLine,
        AdjustmentPackage,
        ValidationResult,
        ApprovalRecord,
        AuditEntry,
        ExecutionPhase,
        AdjustmentType,
        ValidationStatus,
        ApprovalDecision,
        AuditEventType,
    )
except ImportError:
    RecalculationExecutionWorkflow = None  # type: ignore[assignment,misc]
    RecalculationExecutionInput = None  # type: ignore[assignment,misc]
    RecalculationExecutionResult = None  # type: ignore[assignment,misc]
    ApprovedTrigger = None  # type: ignore[assignment,misc]
    BaseYearInventory = None  # type: ignore[assignment,misc]
    AdjustmentLine = None  # type: ignore[assignment,misc]
    AdjustmentPackage = None  # type: ignore[assignment,misc]
    ValidationResult = None  # type: ignore[assignment,misc]
    ApprovalRecord = None  # type: ignore[assignment,misc]
    AuditEntry = None  # type: ignore[assignment,misc]
    ExecutionPhase = None  # type: ignore[assignment,misc]
    AdjustmentType = None  # type: ignore[assignment,misc]
    ValidationStatus = None  # type: ignore[assignment,misc]
    ApprovalDecision = None  # type: ignore[assignment,misc]
    AuditEventType = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Target Rebasing Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_045_base_year.workflows.target_rebasing_workflow import (
        TargetRebasingWorkflow,
        TargetRebasingInput,
        TargetRebasingResult,
        EmissionTarget,
        TargetImpact,
        RebasedTarget,
        StakeholderNotification,
        ImpactSummary,
        RebasingPhase,
        TargetType,
        TargetScope,
        TargetFramework,
        NotificationChannel,
        NotificationStatus,
    )
except ImportError:
    TargetRebasingWorkflow = None  # type: ignore[assignment,misc]
    TargetRebasingInput = None  # type: ignore[assignment,misc]
    TargetRebasingResult = None  # type: ignore[assignment,misc]
    EmissionTarget = None  # type: ignore[assignment,misc]
    TargetImpact = None  # type: ignore[assignment,misc]
    RebasedTarget = None  # type: ignore[assignment,misc]
    StakeholderNotification = None  # type: ignore[assignment,misc]
    ImpactSummary = None  # type: ignore[assignment,misc]
    RebasingPhase = None  # type: ignore[assignment,misc]
    TargetType = None  # type: ignore[assignment,misc]
    TargetScope = None  # type: ignore[assignment,misc]
    TargetFramework = None  # type: ignore[assignment,misc]
    NotificationChannel = None  # type: ignore[assignment,misc]
    NotificationStatus = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Audit Verification Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_045_base_year.workflows.audit_verification_workflow import (
        AuditVerificationWorkflow,
        AuditVerificationInput,
        AuditVerificationResult,
        AuditTrailEntry,
        EvidenceArtifact,
        CompletenessGap,
        VerificationPackageSection,
        SupportMaterial,
        VerificationPhase,
        VerificationLevel,
        EvidenceCategory,
        EvidenceStatus,
        GapSeverity,
        PackageSection,
    )
except ImportError:
    AuditVerificationWorkflow = None  # type: ignore[assignment,misc]
    AuditVerificationInput = None  # type: ignore[assignment,misc]
    AuditVerificationResult = None  # type: ignore[assignment,misc]
    AuditTrailEntry = None  # type: ignore[assignment,misc]
    EvidenceArtifact = None  # type: ignore[assignment,misc]
    CompletenessGap = None  # type: ignore[assignment,misc]
    VerificationPackageSection = None  # type: ignore[assignment,misc]
    SupportMaterial = None  # type: ignore[assignment,misc]
    VerificationPhase = None  # type: ignore[assignment,misc]
    VerificationLevel = None  # type: ignore[assignment,misc]
    EvidenceCategory = None  # type: ignore[assignment,misc]
    EvidenceStatus = None  # type: ignore[assignment,misc]
    GapSeverity = None  # type: ignore[assignment,misc]
    PackageSection = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Annual Review Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_045_base_year.workflows.annual_review_workflow import (
        AnnualReviewWorkflow,
        AnnualReviewInput,
        AnnualReviewResult,
        BaseYearPolicy,
        YearData,
        PolicyFinding,
        ScanResult,
        ConsistencyCheckResult,
        ReviewReport,
        ReviewPhase,
        PolicyStatus,
        ConsistencyStatus,
        FindingSeverity,
        TriggerScanStatus,
    )
except ImportError:
    AnnualReviewWorkflow = None  # type: ignore[assignment,misc]
    AnnualReviewInput = None  # type: ignore[assignment,misc]
    AnnualReviewResult = None  # type: ignore[assignment,misc]
    BaseYearPolicy = None  # type: ignore[assignment,misc]
    YearData = None  # type: ignore[assignment,misc]
    PolicyFinding = None  # type: ignore[assignment,misc]
    ScanResult = None  # type: ignore[assignment,misc]
    ConsistencyCheckResult = None  # type: ignore[assignment,misc]
    ReviewReport = None  # type: ignore[assignment,misc]
    ReviewPhase = None  # type: ignore[assignment,misc]
    PolicyStatus = None  # type: ignore[assignment,misc]
    ConsistencyStatus = None  # type: ignore[assignment,misc]
    FindingSeverity = None  # type: ignore[assignment,misc]
    TriggerScanStatus = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Merger & Acquisition Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_045_base_year.workflows.merger_acquisition_workflow import (
        MergerAcquisitionWorkflow,
        MergerAcquisitionInput,
        MergerAcquisitionResult,
        MAEntity,
        EntityEmissions,
        ProRataAdjustment,
        SignificanceTestResult,
        MAPhase,
        TransactionType,
        EntityType,
        DataQualityTier,
        SignificanceOutcome,
        AdjustmentDirection,
    )
except ImportError:
    MergerAcquisitionWorkflow = None  # type: ignore[assignment,misc]
    MergerAcquisitionInput = None  # type: ignore[assignment,misc]
    MergerAcquisitionResult = None  # type: ignore[assignment,misc]
    MAEntity = None  # type: ignore[assignment,misc]
    EntityEmissions = None  # type: ignore[assignment,misc]
    ProRataAdjustment = None  # type: ignore[assignment,misc]
    SignificanceTestResult = None  # type: ignore[assignment,misc]
    MAPhase = None  # type: ignore[assignment,misc]
    TransactionType = None  # type: ignore[assignment,misc]
    EntityType = None  # type: ignore[assignment,misc]
    DataQualityTier = None  # type: ignore[assignment,misc]
    SignificanceOutcome = None  # type: ignore[assignment,misc]
    AdjustmentDirection = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Full Base Year Pipeline Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_045_base_year.workflows.full_base_year_pipeline_workflow import (
        FullBaseYearPipelineWorkflow,
        FullBaseYearPipelineInput,
        FullBaseYearPipelineResult,
        BaseYearManagementConfig,
        MilestoneRecord,
        TimeSeriesEntry,
        PipelineReport,
        TargetProgress,
        RecalculationRecord,
        PipelinePhase,
        PipelineMilestone,
        ReportType,
    )
except ImportError:
    FullBaseYearPipelineWorkflow = None  # type: ignore[assignment,misc]
    FullBaseYearPipelineInput = None  # type: ignore[assignment,misc]
    FullBaseYearPipelineResult = None  # type: ignore[assignment,misc]
    BaseYearManagementConfig = None  # type: ignore[assignment,misc]
    MilestoneRecord = None  # type: ignore[assignment,misc]
    TimeSeriesEntry = None  # type: ignore[assignment,misc]
    PipelineReport = None  # type: ignore[assignment,misc]
    TargetProgress = None  # type: ignore[assignment,misc]
    RecalculationRecord = None  # type: ignore[assignment,misc]
    PipelinePhase = None  # type: ignore[assignment,misc]
    PipelineMilestone = None  # type: ignore[assignment,misc]
    ReportType = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Type Aliases
# ---------------------------------------------------------------------------

# Base Year Establishment types
EstablishInput = BaseYearEstablishmentInput
EstablishOutput = BaseYearEstablishmentResult

# Recalculation Assessment types
AssessInput = RecalculationAssessmentInput
AssessOutput = RecalculationAssessmentResult

# Recalculation Execution types
ExecInput = RecalculationExecutionInput
ExecOutput = RecalculationExecutionResult

# Target Rebasing types
RebaseInput = TargetRebasingInput
RebaseOutput = TargetRebasingResult

# Audit Verification types
AuditInput = AuditVerificationInput
AuditOutput = AuditVerificationResult

# Annual Review types
ReviewInput = AnnualReviewInput
ReviewOutput = AnnualReviewResult

# Merger & Acquisition types
MAInput = MergerAcquisitionInput
MAOutput = MergerAcquisitionResult

# Full Pipeline types
PipelineInput = FullBaseYearPipelineInput
PipelineOutput = FullBaseYearPipelineResult


# ---------------------------------------------------------------------------
# Loaded Workflow Helper
# ---------------------------------------------------------------------------

def get_loaded_workflows() -> Dict[str, Optional[type]]:
    """
    Return a dictionary of workflow names to their classes.

    Workflows that failed to import will have None values.

    Returns:
        Dict mapping workflow name to workflow class or None.
    """
    return {
        "base_year_establishment": BaseYearEstablishmentWorkflow,
        "recalculation_assessment": RecalculationAssessmentWorkflow,
        "recalculation_execution": RecalculationExecutionWorkflow,
        "target_rebasing": TargetRebasingWorkflow,
        "audit_verification": AuditVerificationWorkflow,
        "annual_review": AnnualReviewWorkflow,
        "merger_acquisition": MergerAcquisitionWorkflow,
        "full_base_year_pipeline": FullBaseYearPipelineWorkflow,
    }


__all__ = [
    # --- Base Year Establishment Workflow ---
    "BaseYearEstablishmentWorkflow",
    "BaseYearEstablishmentInput",
    "BaseYearEstablishmentResult",
    "CandidateYear",
    "ScopeEmissions",
    "CandidateAssessmentResult",
    "InventorySnapshot",
    "GeneratedDocument",
    "SelectionWeights",
    "QualityScore",
    "EstablishmentPhase",
    "QualityDimension",
    "CandidateStatus",
    "DocumentType",
    # --- Recalculation Assessment Workflow ---
    "RecalculationAssessmentWorkflow",
    "RecalculationAssessmentInput",
    "RecalculationAssessmentResult",
    "ExternalEvent",
    "EmissionsInventory",
    "RecalculationPolicy",
    "DetectedTrigger",
    "SignificanceResult",
    "PolicyComplianceResult",
    "Recommendation",
    "AssessmentPhase",
    "TriggerType",
    "TriggerCategory",
    "SignificanceLevel",
    "ComplianceStatus",
    "RecommendationPriority",
    # --- Recalculation Execution Workflow ---
    "RecalculationExecutionWorkflow",
    "RecalculationExecutionInput",
    "RecalculationExecutionResult",
    "ApprovedTrigger",
    "BaseYearInventory",
    "AdjustmentLine",
    "AdjustmentPackage",
    "ValidationResult",
    "ApprovalRecord",
    "AuditEntry",
    "ExecutionPhase",
    "AdjustmentType",
    "ValidationStatus",
    "ApprovalDecision",
    "AuditEventType",
    # --- Target Rebasing Workflow ---
    "TargetRebasingWorkflow",
    "TargetRebasingInput",
    "TargetRebasingResult",
    "EmissionTarget",
    "TargetImpact",
    "RebasedTarget",
    "StakeholderNotification",
    "ImpactSummary",
    "RebasingPhase",
    "TargetType",
    "TargetScope",
    "TargetFramework",
    "NotificationChannel",
    "NotificationStatus",
    # --- Audit Verification Workflow ---
    "AuditVerificationWorkflow",
    "AuditVerificationInput",
    "AuditVerificationResult",
    "AuditTrailEntry",
    "EvidenceArtifact",
    "CompletenessGap",
    "VerificationPackageSection",
    "SupportMaterial",
    "VerificationPhase",
    "VerificationLevel",
    "EvidenceCategory",
    "EvidenceStatus",
    "GapSeverity",
    "PackageSection",
    # --- Annual Review Workflow ---
    "AnnualReviewWorkflow",
    "AnnualReviewInput",
    "AnnualReviewResult",
    "BaseYearPolicy",
    "YearData",
    "PolicyFinding",
    "ScanResult",
    "ConsistencyCheckResult",
    "ReviewReport",
    "ReviewPhase",
    "PolicyStatus",
    "ConsistencyStatus",
    "FindingSeverity",
    "TriggerScanStatus",
    # --- Merger & Acquisition Workflow ---
    "MergerAcquisitionWorkflow",
    "MergerAcquisitionInput",
    "MergerAcquisitionResult",
    "MAEntity",
    "EntityEmissions",
    "ProRataAdjustment",
    "SignificanceTestResult",
    "MAPhase",
    "TransactionType",
    "EntityType",
    "DataQualityTier",
    "SignificanceOutcome",
    "AdjustmentDirection",
    # --- Full Base Year Pipeline Workflow ---
    "FullBaseYearPipelineWorkflow",
    "FullBaseYearPipelineInput",
    "FullBaseYearPipelineResult",
    "BaseYearManagementConfig",
    "MilestoneRecord",
    "TimeSeriesEntry",
    "PipelineReport",
    "TargetProgress",
    "RecalculationRecord",
    "PipelinePhase",
    "PipelineMilestone",
    "ReportType",
    # --- Type Aliases ---
    "EstablishInput",
    "EstablishOutput",
    "AssessInput",
    "AssessOutput",
    "ExecInput",
    "ExecOutput",
    "RebaseInput",
    "RebaseOutput",
    "AuditInput",
    "AuditOutput",
    "ReviewInput",
    "ReviewOutput",
    "MAInput",
    "MAOutput",
    "PipelineInput",
    "PipelineOutput",
    # --- Helpers ---
    "get_loaded_workflows",
]
