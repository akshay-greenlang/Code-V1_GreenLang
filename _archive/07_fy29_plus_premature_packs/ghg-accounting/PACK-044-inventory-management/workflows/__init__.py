# -*- coding: utf-8 -*-
"""
PACK-044 GHG Inventory Management Pack - Workflow Orchestration
===================================================================

Complete GHG inventory management workflow orchestrators for the
GHG Protocol Corporate Standard, ISO 14064-1:2018, and ESRS E1
compliance. Each workflow coordinates inventory management processes
covering annual cycle management, data collection campaigns, quality
review and certification, change assessment with base year recalculation,
inventory finalization with digital approval, multi-entity consolidation,
improvement planning, and full end-to-end pipeline orchestration.

Workflows:
    - AnnualInventoryCycleWorkflow: 8-phase annual inventory cycle with
      period setup, data collection, calculation, quality review, internal
      review, finalization, reporting, and improvement planning.

    - DataCollectionCampaignWorkflow: 5-phase data collection campaign
      with planning, distribution, monitoring, validation, and completion
      tracking across all facilities.

    - QualityReviewWorkflow: 4-phase quality review with automated QA/QC
      checks, issue resolution, manual expert review, and quality
      certification with deterministic scoring.

    - ChangeAssessmentWorkflow: 4-phase change assessment with change
      identification, impact quantification, governance approval, and
      implementation with base year recalculation triggers.

    - InventoryFinalizationWorkflow: 5-phase finalization with pre-checks,
      version creation, digital approval with signature chain, lock and
      archive with integrity hashing, and stakeholder distribution.

    - ConsolidationWorkflow: 4-phase multi-entity consolidation with
      hierarchy mapping, subsidiary data collection, consolidation
      approach execution (equity/financial/operational), and audit review.

    - ImprovementPlanningWorkflow: 4-phase improvement planning with gap
      identification against benchmarks, cost-benefit option evaluation,
      prioritized quarterly roadmap, and detailed action planning.

    - FullManagementPipelineWorkflow: 12-phase end-to-end orchestrator
      invoking all sub-workflows in sequence with milestone tracking,
      scope summaries, entity consolidation, and comprehensive metrics.

Author: GreenLang Team
Version: 44.0.0
"""

from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Annual Inventory Cycle Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_044_inventory_management.workflows.annual_inventory_cycle_workflow import (
        AnnualInventoryCycleWorkflow,
        AnnualInventoryCycleInput,
        AnnualInventoryCycleResult,
        PeriodConfig,
        FacilityCollectionStatus,
        CalculationSummary,
        QualityScorecard,
        ReviewRecord,
        ReportOutput,
        ImprovementAction,
        CyclePhase,
        ReviewStatus,
        DataCompleteness,
        ImprovementPriority,
    )
except ImportError:
    AnnualInventoryCycleWorkflow = None  # type: ignore[assignment,misc]
    AnnualInventoryCycleInput = None  # type: ignore[assignment,misc]
    AnnualInventoryCycleResult = None  # type: ignore[assignment,misc]
    PeriodConfig = None  # type: ignore[assignment,misc]
    FacilityCollectionStatus = None  # type: ignore[assignment,misc]
    CalculationSummary = None  # type: ignore[assignment,misc]
    QualityScorecard = None  # type: ignore[assignment,misc]
    ReviewRecord = None  # type: ignore[assignment,misc]
    ReportOutput = None  # type: ignore[assignment,misc]
    ImprovementAction = None  # type: ignore[assignment,misc]
    CyclePhase = None  # type: ignore[assignment,misc]
    ReviewStatus = None  # type: ignore[assignment,misc]
    DataCompleteness = None  # type: ignore[assignment,misc]
    ImprovementPriority = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Data Collection Campaign Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_044_inventory_management.workflows.data_collection_campaign_workflow import (
        DataCollectionCampaignWorkflow,
        DataCollectionCampaignInput,
        DataCollectionCampaignResult,
        DataRequirement,
        DataRequest,
        SubmissionProgress,
        ValidationIssue,
        CampaignSummary,
        CampaignPhase,
        RequestStatus,
        DataRequestType,
        ValidationSeverity,
        EscalationLevel,
    )
except ImportError:
    DataCollectionCampaignWorkflow = None  # type: ignore[assignment,misc]
    DataCollectionCampaignInput = None  # type: ignore[assignment,misc]
    DataCollectionCampaignResult = None  # type: ignore[assignment,misc]
    DataRequirement = None  # type: ignore[assignment,misc]
    DataRequest = None  # type: ignore[assignment,misc]
    SubmissionProgress = None  # type: ignore[assignment,misc]
    ValidationIssue = None  # type: ignore[assignment,misc]
    CampaignSummary = None  # type: ignore[assignment,misc]
    CampaignPhase = None  # type: ignore[assignment,misc]
    RequestStatus = None  # type: ignore[assignment,misc]
    DataRequestType = None  # type: ignore[assignment,misc]
    ValidationSeverity = None  # type: ignore[assignment,misc]
    EscalationLevel = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Quality Review Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_044_inventory_management.workflows.quality_review_workflow import (
        QualityReviewWorkflow,
        QualityReviewInput,
        QualityReviewResult,
        QACheckResult,
        QualityIssue,
        ManualReviewItem,
        QualityCertificate,
        QAQCPhase,
        CheckCategory,
        IssueSeverity,
        IssueStatus,
        CertificationLevel,
    )
except ImportError:
    QualityReviewWorkflow = None  # type: ignore[assignment,misc]
    QualityReviewInput = None  # type: ignore[assignment,misc]
    QualityReviewResult = None  # type: ignore[assignment,misc]
    QACheckResult = None  # type: ignore[assignment,misc]
    QualityIssue = None  # type: ignore[assignment,misc]
    ManualReviewItem = None  # type: ignore[assignment,misc]
    QualityCertificate = None  # type: ignore[assignment,misc]
    QAQCPhase = None  # type: ignore[assignment,misc]
    CheckCategory = None  # type: ignore[assignment,misc]
    IssueSeverity = None  # type: ignore[assignment,misc]
    IssueStatus = None  # type: ignore[assignment,misc]
    CertificationLevel = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Change Assessment Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_044_inventory_management.workflows.change_assessment_workflow import (
        ChangeAssessmentWorkflow,
        ChangeAssessmentInput,
        ChangeAssessmentResult,
        ChangeRecord,
        ImpactAssessment,
        ApprovalVote,
        ChangeProposal,
        ImplementationRecord,
        ChangeAssessmentPhase,
        ChangeType,
        ImpactLevel,
        ApprovalStatus,
        RecalculationTrigger,
    )
except ImportError:
    ChangeAssessmentWorkflow = None  # type: ignore[assignment,misc]
    ChangeAssessmentInput = None  # type: ignore[assignment,misc]
    ChangeAssessmentResult = None  # type: ignore[assignment,misc]
    ChangeRecord = None  # type: ignore[assignment,misc]
    ImpactAssessment = None  # type: ignore[assignment,misc]
    ApprovalVote = None  # type: ignore[assignment,misc]
    ChangeProposal = None  # type: ignore[assignment,misc]
    ImplementationRecord = None  # type: ignore[assignment,misc]
    ChangeAssessmentPhase = None  # type: ignore[assignment,misc]
    ChangeType = None  # type: ignore[assignment,misc]
    ImpactLevel = None  # type: ignore[assignment,misc]
    ApprovalStatus = None  # type: ignore[assignment,misc]
    RecalculationTrigger = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Inventory Finalization Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_044_inventory_management.workflows.inventory_finalization_workflow import (
        InventoryFinalizationWorkflow,
        InventoryFinalizationInput,
        InventoryFinalizationResult,
        PreCheckResult,
        InventoryVersion,
        DigitalSignature,
        ArchivePackage,
        DistributionRecord,
        FinalizationPhase,
        PreCheckCategory,
        SignatureStatus,
        ArchiveStatus,
        DistributionChannel,
    )
except ImportError:
    InventoryFinalizationWorkflow = None  # type: ignore[assignment,misc]
    InventoryFinalizationInput = None  # type: ignore[assignment,misc]
    InventoryFinalizationResult = None  # type: ignore[assignment,misc]
    PreCheckResult = None  # type: ignore[assignment,misc]
    InventoryVersion = None  # type: ignore[assignment,misc]
    DigitalSignature = None  # type: ignore[assignment,misc]
    ArchivePackage = None  # type: ignore[assignment,misc]
    DistributionRecord = None  # type: ignore[assignment,misc]
    FinalizationPhase = None  # type: ignore[assignment,misc]
    PreCheckCategory = None  # type: ignore[assignment,misc]
    SignatureStatus = None  # type: ignore[assignment,misc]
    ArchiveStatus = None  # type: ignore[assignment,misc]
    DistributionChannel = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Consolidation Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_044_inventory_management.workflows.consolidation_workflow import (
        ConsolidationWorkflow,
        ConsolidationInput,
        ConsolidationResult,
        EntityNode,
        SubsidiaryInventory,
        ConsolidatedEntityResult,
        ConsolidationCheck,
        ConsolidationSummary,
        ConsolidationPhase,
        ConsolidationApproach,
        EntityRelationType,
        SubsidiaryDataStatus,
        ConsolidationCheckType,
    )
except ImportError:
    ConsolidationWorkflow = None  # type: ignore[assignment,misc]
    ConsolidationInput = None  # type: ignore[assignment,misc]
    ConsolidationResult = None  # type: ignore[assignment,misc]
    EntityNode = None  # type: ignore[assignment,misc]
    SubsidiaryInventory = None  # type: ignore[assignment,misc]
    ConsolidatedEntityResult = None  # type: ignore[assignment,misc]
    ConsolidationCheck = None  # type: ignore[assignment,misc]
    ConsolidationSummary = None  # type: ignore[assignment,misc]
    ConsolidationPhase = None  # type: ignore[assignment,misc]
    ConsolidationApproach = None  # type: ignore[assignment,misc]
    EntityRelationType = None  # type: ignore[assignment,misc]
    SubsidiaryDataStatus = None  # type: ignore[assignment,misc]
    ConsolidationCheckType = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Improvement Planning Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_044_inventory_management.workflows.improvement_planning_workflow import (
        ImprovementPlanningWorkflow,
        ImprovementPlanningInput,
        ImprovementPlanningResult,
        InventoryGap,
        ImprovementOption,
        RoadmapItem,
        ActionItem,
        ImprovementPlanSummary,
        ImprovementPhase,
        GapCategory,
        GapSeverity,
        FeasibilityLevel,
        ActionStatus,
        Quarter,
    )
except ImportError:
    ImprovementPlanningWorkflow = None  # type: ignore[assignment,misc]
    ImprovementPlanningInput = None  # type: ignore[assignment,misc]
    ImprovementPlanningResult = None  # type: ignore[assignment,misc]
    InventoryGap = None  # type: ignore[assignment,misc]
    ImprovementOption = None  # type: ignore[assignment,misc]
    RoadmapItem = None  # type: ignore[assignment,misc]
    ActionItem = None  # type: ignore[assignment,misc]
    ImprovementPlanSummary = None  # type: ignore[assignment,misc]
    ImprovementPhase = None  # type: ignore[assignment,misc]
    GapCategory = None  # type: ignore[assignment,misc]
    GapSeverity = None  # type: ignore[assignment,misc]
    FeasibilityLevel = None  # type: ignore[assignment,misc]
    ActionStatus = None  # type: ignore[assignment,misc]
    Quarter = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Full Management Pipeline Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_044_inventory_management.workflows.full_management_pipeline_workflow import (
        FullManagementPipelineWorkflow,
        FullManagementPipelineInput,
        FullManagementPipelineResult,
        MilestoneRecord,
        ScopeSummary,
        EntitySummary,
        ReportRecord,
        ImprovementRecord,
        PipelineMetrics,
        PipelinePhase,
        PipelineMilestone,
    )
except ImportError:
    FullManagementPipelineWorkflow = None  # type: ignore[assignment,misc]
    FullManagementPipelineInput = None  # type: ignore[assignment,misc]
    FullManagementPipelineResult = None  # type: ignore[assignment,misc]
    MilestoneRecord = None  # type: ignore[assignment,misc]
    ScopeSummary = None  # type: ignore[assignment,misc]
    EntitySummary = None  # type: ignore[assignment,misc]
    ReportRecord = None  # type: ignore[assignment,misc]
    ImprovementRecord = None  # type: ignore[assignment,misc]
    PipelineMetrics = None  # type: ignore[assignment,misc]
    PipelinePhase = None  # type: ignore[assignment,misc]
    PipelineMilestone = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Type Aliases
# ---------------------------------------------------------------------------

# Annual Inventory Cycle types
CycleInput = AnnualInventoryCycleInput
CycleOutput = AnnualInventoryCycleResult

# Data Collection Campaign types
CampaignInput = DataCollectionCampaignInput
CampaignOutput = DataCollectionCampaignResult

# Quality Review types
QRInput = QualityReviewInput
QROutput = QualityReviewResult

# Change Assessment types
ChangeInput = ChangeAssessmentInput
ChangeOutput = ChangeAssessmentResult

# Inventory Finalization types
FinalInput = InventoryFinalizationInput
FinalOutput = InventoryFinalizationResult

# Consolidation types
ConsInput = ConsolidationInput
ConsOutput = ConsolidationResult

# Improvement Planning types
ImpInput = ImprovementPlanningInput
ImpOutput = ImprovementPlanningResult

# Full Pipeline types
PipelineInput = FullManagementPipelineInput
PipelineOutput = FullManagementPipelineResult


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
        "annual_inventory_cycle": AnnualInventoryCycleWorkflow,
        "data_collection_campaign": DataCollectionCampaignWorkflow,
        "quality_review": QualityReviewWorkflow,
        "change_assessment": ChangeAssessmentWorkflow,
        "inventory_finalization": InventoryFinalizationWorkflow,
        "consolidation": ConsolidationWorkflow,
        "improvement_planning": ImprovementPlanningWorkflow,
        "full_management_pipeline": FullManagementPipelineWorkflow,
    }


__all__ = [
    # --- Annual Inventory Cycle Workflow ---
    "AnnualInventoryCycleWorkflow",
    "AnnualInventoryCycleInput",
    "AnnualInventoryCycleResult",
    "PeriodConfig",
    "FacilityCollectionStatus",
    "CalculationSummary",
    "QualityScorecard",
    "ReviewRecord",
    "ReportOutput",
    "ImprovementAction",
    "CyclePhase",
    "ReviewStatus",
    "DataCompleteness",
    "ImprovementPriority",
    # --- Data Collection Campaign Workflow ---
    "DataCollectionCampaignWorkflow",
    "DataCollectionCampaignInput",
    "DataCollectionCampaignResult",
    "DataRequirement",
    "DataRequest",
    "SubmissionProgress",
    "ValidationIssue",
    "CampaignSummary",
    "CampaignPhase",
    "RequestStatus",
    "DataRequestType",
    "ValidationSeverity",
    "EscalationLevel",
    # --- Quality Review Workflow ---
    "QualityReviewWorkflow",
    "QualityReviewInput",
    "QualityReviewResult",
    "QACheckResult",
    "QualityIssue",
    "ManualReviewItem",
    "QualityCertificate",
    "QAQCPhase",
    "CheckCategory",
    "IssueSeverity",
    "IssueStatus",
    "CertificationLevel",
    # --- Change Assessment Workflow ---
    "ChangeAssessmentWorkflow",
    "ChangeAssessmentInput",
    "ChangeAssessmentResult",
    "ChangeRecord",
    "ImpactAssessment",
    "ApprovalVote",
    "ChangeProposal",
    "ImplementationRecord",
    "ChangeAssessmentPhase",
    "ChangeType",
    "ImpactLevel",
    "ApprovalStatus",
    "RecalculationTrigger",
    # --- Inventory Finalization Workflow ---
    "InventoryFinalizationWorkflow",
    "InventoryFinalizationInput",
    "InventoryFinalizationResult",
    "PreCheckResult",
    "InventoryVersion",
    "DigitalSignature",
    "ArchivePackage",
    "DistributionRecord",
    "FinalizationPhase",
    "PreCheckCategory",
    "SignatureStatus",
    "ArchiveStatus",
    "DistributionChannel",
    # --- Consolidation Workflow ---
    "ConsolidationWorkflow",
    "ConsolidationInput",
    "ConsolidationResult",
    "EntityNode",
    "SubsidiaryInventory",
    "ConsolidatedEntityResult",
    "ConsolidationCheck",
    "ConsolidationSummary",
    "ConsolidationPhase",
    "ConsolidationApproach",
    "EntityRelationType",
    "SubsidiaryDataStatus",
    "ConsolidationCheckType",
    # --- Improvement Planning Workflow ---
    "ImprovementPlanningWorkflow",
    "ImprovementPlanningInput",
    "ImprovementPlanningResult",
    "InventoryGap",
    "ImprovementOption",
    "RoadmapItem",
    "ActionItem",
    "ImprovementPlanSummary",
    "ImprovementPhase",
    "GapCategory",
    "GapSeverity",
    "FeasibilityLevel",
    "ActionStatus",
    "Quarter",
    # --- Full Management Pipeline Workflow ---
    "FullManagementPipelineWorkflow",
    "FullManagementPipelineInput",
    "FullManagementPipelineResult",
    "MilestoneRecord",
    "ScopeSummary",
    "EntitySummary",
    "ReportRecord",
    "ImprovementRecord",
    "PipelineMetrics",
    "PipelinePhase",
    "PipelineMilestone",
    # --- Type Aliases ---
    "CycleInput",
    "CycleOutput",
    "CampaignInput",
    "CampaignOutput",
    "QRInput",
    "QROutput",
    "ChangeInput",
    "ChangeOutput",
    "FinalInput",
    "FinalOutput",
    "ConsInput",
    "ConsOutput",
    "ImpInput",
    "ImpOutput",
    "PipelineInput",
    "PipelineOutput",
    # --- Helpers ---
    "get_loaded_workflows",
]
