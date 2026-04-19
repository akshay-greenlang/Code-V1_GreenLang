# -*- coding: utf-8 -*-
"""
PACK-049 GHG Multi-Site Management Pack - Workflow Orchestration
==================================================================

Complete multi-site GHG management workflow orchestrators covering site
registration, data collection, boundary definition, consolidation,
allocation, site comparison, quality improvement, and full end-to-end
pipeline orchestration.

Workflows:
    - SiteRegistrationWorkflow: 5-phase site discovery, classification,
      characteristics, boundary assignment, and activation.
    - DataCollectionWorkflow: 5-phase period setup, template distribution,
      site submission, validation review, and approval.
    - BoundaryDefinitionWorkflow: 5-phase entity mapping, ownership chain,
      consolidation approach, materiality check, and boundary lock.
    - ConsolidationWorkflow: 5-phase site data gather, elimination check,
      equity adjustment, reconciliation, and consolidated total.
    - AllocationWorkflow: 4-phase shared service identification, method
      selection, calculation, and verification.
    - SiteComparisonWorkflow: 5-phase peer group build, KPI calculation,
      ranking, gap analysis, and best practice reporting.
    - QualityImprovementWorkflow: 5-phase quality assessment, gap
      identification, remediation planning, implementation, verification.
    - FullMultiSitePipelineWorkflow: 8-phase end-to-end orchestrator
      chaining all sub-workflows with checkpointing and provenance.

Author: GreenLang Team
Version: 49.0.0
"""

from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Site Registration Workflow
# ---------------------------------------------------------------------------
try:
    from .site_registration_workflow import (
        SiteRegistrationWorkflow,
        SiteRegistrationInput,
        SiteRegistrationResult,
        RegistrationPhase,
        FacilityType,
        SectorClassification,
        ConsolidationApproach as RegConsolidationApproach,
        SiteStatus,
        GeographicRegion,
        CandidateSite,
        ClassifiedSite,
        SiteCharacteristics,
        BoundaryAssignment,
        RegisteredSite,
    )
except ImportError:
    SiteRegistrationWorkflow = None  # type: ignore[assignment,misc]
    SiteRegistrationInput = None  # type: ignore[assignment,misc]
    SiteRegistrationResult = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Data Collection Workflow
# ---------------------------------------------------------------------------
try:
    from .data_collection_workflow import (
        DataCollectionWorkflow,
        DataCollectionInput,
        DataCollectionResult,
        CollectionPhase,
        EmissionScope as CollEmissionScope,
        SubmissionStatus,
        ValidationSeverity,
        ApprovalDecision,
        TemplateType,
        CollectionRoundConfig,
        SiteTemplate,
        DataEntry,
        ValidationFinding,
        SiteSubmissionRecord,
    )
except ImportError:
    DataCollectionWorkflow = None  # type: ignore[assignment,misc]
    DataCollectionInput = None  # type: ignore[assignment,misc]
    DataCollectionResult = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Boundary Definition Workflow
# ---------------------------------------------------------------------------
try:
    from .boundary_definition_workflow import (
        BoundaryDefinitionWorkflow,
        BoundaryDefinitionInput,
        BoundaryDefinitionResult,
        BoundaryPhase,
        ConsolidationApproach as BndConsolidationApproach,
        EntityType,
        ControlType,
        MaterialityClassification,
        BoundaryLockStatus,
        LegalEntity,
        EntityFacilityMapping,
        OwnershipLink,
        ConsolidationResult as BndConsolidationResult,
        MaterialityAssessment,
        BoundaryDocument,
    )
except ImportError:
    BoundaryDefinitionWorkflow = None  # type: ignore[assignment,misc]
    BoundaryDefinitionInput = None  # type: ignore[assignment,misc]
    BoundaryDefinitionResult = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Consolidation Workflow
# ---------------------------------------------------------------------------
try:
    from .consolidation_workflow import (
        ConsolidationWorkflow,
        ConsolidationInput,
        ConsolidationResult,
        ConsolidationPhase,
        EmissionScope as ConsEmissionScope,
        ConsolidationApproach as ConsConsolidationApproach,
        EliminationType,
        ReconciliationStatus,
        SiteEmissionTotal,
        EliminationEntry,
        EquityAdjustment,
        ReconciliationRecord,
        ConsolidatedTotals,
    )
except ImportError:
    ConsolidationWorkflow = None  # type: ignore[assignment,misc]
    ConsolidationInput = None  # type: ignore[assignment,misc]
    ConsolidationResult = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Allocation Workflow
# ---------------------------------------------------------------------------
try:
    from .allocation_workflow import (
        AllocationWorkflow,
        AllocationInput,
        AllocationResult,
        AllocationPhase,
        SharedServiceType,
        AllocationMethod,
        VerificationStatus as AllocVerificationStatus,
        SharedService,
        SiteAllocationDriver,
        AllocationLineItem,
        VerificationCheck,
    )
except ImportError:
    AllocationWorkflow = None  # type: ignore[assignment,misc]
    AllocationInput = None  # type: ignore[assignment,misc]
    AllocationResult = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Site Comparison Workflow
# ---------------------------------------------------------------------------
try:
    from .site_comparison_workflow import (
        SiteComparisonWorkflow,
        SiteComparisonInput,
        SiteComparisonResult,
        ComparisonPhase,
        PeerGroupCriteria,
        KPIType,
        PerformanceBand,
        SiteMetrics,
        PeerGroup,
        SiteKPI,
        SiteRanking,
        GapAnalysisItem,
        BestPracticeEntry,
    )
except ImportError:
    SiteComparisonWorkflow = None  # type: ignore[assignment,misc]
    SiteComparisonInput = None  # type: ignore[assignment,misc]
    SiteComparisonResult = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Quality Improvement Workflow
# ---------------------------------------------------------------------------
try:
    from .quality_improvement_workflow import (
        QualityImprovementWorkflow,
        QualityImprovementInput,
        QualityImprovementResult,
        QualityPhase,
        QualityDimension,
        QualityTier,
        RemediationPriority,
        RemediationStatus,
        ImprovementVerdict,
        SiteDimensionScore,
        SiteQualityAssessment,
        QualityGap,
        RemediationAction,
        ImplementationProgress,
        VerificationResult as QualVerificationResult,
    )
except ImportError:
    QualityImprovementWorkflow = None  # type: ignore[assignment,misc]
    QualityImprovementInput = None  # type: ignore[assignment,misc]
    QualityImprovementResult = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Full Multi-Site Pipeline Workflow
# ---------------------------------------------------------------------------
try:
    from .full_multi_site_pipeline_workflow import (
        FullMultiSitePipelineWorkflow,
        FullPipelineInput,
        FullPipelineResult,
        PipelinePhase,
        ReportFormat,
        CheckpointStatus,
        PipelineCheckpoint,
        PipelineMilestone,
        PipelineReport,
    )
except ImportError:
    FullMultiSitePipelineWorkflow = None  # type: ignore[assignment,misc]
    FullPipelineInput = None  # type: ignore[assignment,misc]
    FullPipelineResult = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def get_loaded_workflows() -> Dict[str, Optional[type]]:
    """Return a dictionary of workflow names to their classes."""
    return {
        "site_registration": SiteRegistrationWorkflow,
        "data_collection": DataCollectionWorkflow,
        "boundary_definition": BoundaryDefinitionWorkflow,
        "consolidation": ConsolidationWorkflow,
        "allocation": AllocationWorkflow,
        "site_comparison": SiteComparisonWorkflow,
        "quality_improvement": QualityImprovementWorkflow,
        "full_multi_site_pipeline": FullMultiSitePipelineWorkflow,
    }


__all__ = [
    # Site Registration
    "SiteRegistrationWorkflow", "SiteRegistrationInput", "SiteRegistrationResult",
    # Data Collection
    "DataCollectionWorkflow", "DataCollectionInput", "DataCollectionResult",
    # Boundary Definition
    "BoundaryDefinitionWorkflow", "BoundaryDefinitionInput", "BoundaryDefinitionResult",
    # Consolidation
    "ConsolidationWorkflow", "ConsolidationInput", "ConsolidationResult",
    # Allocation
    "AllocationWorkflow", "AllocationInput", "AllocationResult",
    # Site Comparison
    "SiteComparisonWorkflow", "SiteComparisonInput", "SiteComparisonResult",
    # Quality Improvement
    "QualityImprovementWorkflow", "QualityImprovementInput", "QualityImprovementResult",
    # Full Pipeline
    "FullMultiSitePipelineWorkflow", "FullPipelineInput", "FullPipelineResult",
    # Helper
    "get_loaded_workflows",
]
