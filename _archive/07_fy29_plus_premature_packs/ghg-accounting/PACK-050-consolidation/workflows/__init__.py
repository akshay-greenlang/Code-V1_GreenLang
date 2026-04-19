# -*- coding: utf-8 -*-
"""
PACK-050 GHG Consolidation Pack - Workflow Orchestration
==================================================================

Complete multi-entity GHG consolidation workflow orchestrators covering
entity mapping, boundary selection, entity data collection, consolidation
execution, intercompany elimination, M&A adjustments, group reporting,
and full end-to-end pipeline orchestration per GHG Protocol Corporate
Standard Chapter 3.

Workflows:
    - EntityMappingWorkflow: 5-phase entity discovery, ownership chain
      resolution, control assessment, materiality screening, and
      registry lock.
    - BoundarySelectionWorkflow: 4-phase approach evaluation, impact
      analysis, stakeholder approval, and boundary lock.
    - EntityDataCollectionWorkflow: 5-phase entity assignment, data
      request distribution, submission collection, validation review,
      and gap resolution.
    - ConsolidationExecutionWorkflow: 6-phase data gathering, equity
      adjustment, control adjustment, intercompany elimination, manual
      adjustment application, and consolidated total.
    - EliminationWorkflow: 4-phase transfer identification, matching
      verification, elimination calculation, and reconciliation check.
    - MnAAdjustmentWorkflow: 5-phase event capture, boundary impact
      assessment, pro-rata calculation, base year restatement, and
      disclosure generation.
    - GroupReportingWorkflow: 4-phase data aggregation, framework
      mapping, report generation, and quality assurance.
    - FullConsolidationPipelineWorkflow: 8-phase end-to-end orchestrator
      chaining all sub-workflows with checkpointing and provenance.

Author: GreenLang Team
Version: 50.0.0
"""

from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Entity Mapping Workflow
# ---------------------------------------------------------------------------
try:
    from .entity_mapping_workflow import (
        EntityMappingWorkflow,
        EntityMappingInput,
        EntityMappingResult,
        EntityMappingPhase,
        EntityType,
        ControlType,
        MaterialityClassification,
        RegistryLockStatus,
        CandidateEntity,
        OwnershipChain,
        ControlAssessmentResult,
        MaterialityAssessment,
        LockedEntity,
    )
except ImportError:
    EntityMappingWorkflow = None  # type: ignore[assignment,misc]
    EntityMappingInput = None  # type: ignore[assignment,misc]
    EntityMappingResult = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Boundary Selection Workflow
# ---------------------------------------------------------------------------
try:
    from .boundary_selection_workflow import (
        BoundarySelectionWorkflow,
        BoundarySelectionInput,
        BoundarySelectionResult,
        BoundarySelectionPhase,
        ConsolidationApproach as BndConsolidationApproach,
        ApproachSuitability,
        ApprovalDecision,
        BoundaryLockStatus,
        EntitySummary,
        ApproachEvaluation,
        ImpactAnalysisResult,
        StakeholderVote,
        BoundaryLockRecord,
    )
except ImportError:
    BoundarySelectionWorkflow = None  # type: ignore[assignment,misc]
    BoundarySelectionInput = None  # type: ignore[assignment,misc]
    BoundarySelectionResult = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Entity Data Collection Workflow
# ---------------------------------------------------------------------------
try:
    from .entity_data_collection_workflow import (
        EntityDataCollectionWorkflow,
        EntityDataCollectionInput,
        EntityDataCollectionResult,
        DataCollectionPhase,
        SubmissionStatus,
        ValidationSeverity,
        GapResolutionMethod,
        EmissionScope as CollEmissionScope,
        StewardAssignment,
        DataRequest,
        EntitySubmission,
        ValidationFinding,
        DataGap,
    )
except ImportError:
    EntityDataCollectionWorkflow = None  # type: ignore[assignment,misc]
    EntityDataCollectionInput = None  # type: ignore[assignment,misc]
    EntityDataCollectionResult = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Consolidation Execution Workflow
# ---------------------------------------------------------------------------
try:
    from .consolidation_execution_workflow import (
        ConsolidationExecutionWorkflow,
        ConsolidationExecInput,
        ConsolidationExecResult,
        ConsolidationExecPhase,
        ConsolidationApproach as ExecConsolidationApproach,
        AdjustmentType,
        EliminationType as ExecEliminationType,
        ReconciliationStatus as ExecReconciliationStatus,
        EntityEmissionRecord,
        EquityAdjustmentRecord,
        ControlAdjustmentRecord,
        EliminationEntry as ExecEliminationEntry,
        ManualAdjustment,
        ConsolidatedTotal,
    )
except ImportError:
    ConsolidationExecutionWorkflow = None  # type: ignore[assignment,misc]
    ConsolidationExecInput = None  # type: ignore[assignment,misc]
    ConsolidationExecResult = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Elimination Workflow
# ---------------------------------------------------------------------------
try:
    from .elimination_workflow import (
        EliminationWorkflow,
        EliminationInput,
        EliminationResult,
        EliminationPhase,
        TransferType,
        MatchStatus,
        EmissionScope as ElimEmissionScope,
        ReconciliationVerdict,
        TransferRecord,
        MatchedPair,
        EliminationRecord,
        ReconciliationSummary,
    )
except ImportError:
    EliminationWorkflow = None  # type: ignore[assignment,misc]
    EliminationInput = None  # type: ignore[assignment,misc]
    EliminationResult = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# M&A Adjustment Workflow
# ---------------------------------------------------------------------------
try:
    from .mna_adjustment_workflow import (
        MnAAdjustmentWorkflow,
        MnAInput,
        MnAResult,
        MnAPhase,
        MnAEventType,
        BoundaryImpact,
        RestatementTrigger,
        MnAEvent,
        BoundaryImpactAssessment,
        ProRataResult,
        BaseYearRestatementResult,
        DisclosureNote,
    )
except ImportError:
    MnAAdjustmentWorkflow = None  # type: ignore[assignment,misc]
    MnAInput = None  # type: ignore[assignment,misc]
    MnAResult = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Group Reporting Workflow
# ---------------------------------------------------------------------------
try:
    from .group_reporting_workflow import (
        GroupReportingWorkflow,
        GroupReportingInput,
        GroupReportingResult,
        GroupReportingPhase,
        ReportingFramework,
        ReportFormat as ReportingReportFormat,
        QACheckStatus,
        SignOffStatus,
        AggregatedData,
        FrameworkOutput,
        GeneratedReport,
        QACheck,
        SignOffRecord,
    )
except ImportError:
    GroupReportingWorkflow = None  # type: ignore[assignment,misc]
    GroupReportingInput = None  # type: ignore[assignment,misc]
    GroupReportingResult = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Full Consolidation Pipeline Workflow
# ---------------------------------------------------------------------------
try:
    from .full_consolidation_pipeline_workflow import (
        FullConsolidationPipelineWorkflow,
        FullConsolidationInput,
        FullConsolidationResult,
        PipelinePhase,
        ReportFormat,
        CheckpointStatus,
        AuditVerdict,
        PipelineCheckpoint,
        PipelineMilestone,
        AuditTrailEntry,
        PipelineSummaryReport,
    )
except ImportError:
    FullConsolidationPipelineWorkflow = None  # type: ignore[assignment,misc]
    FullConsolidationInput = None  # type: ignore[assignment,misc]
    FullConsolidationResult = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def get_loaded_workflows() -> Dict[str, Optional[type]]:
    """Return a dictionary of workflow names to their classes."""
    return {
        "entity_mapping": EntityMappingWorkflow,
        "boundary_selection": BoundarySelectionWorkflow,
        "entity_data_collection": EntityDataCollectionWorkflow,
        "consolidation_execution": ConsolidationExecutionWorkflow,
        "elimination": EliminationWorkflow,
        "mna_adjustment": MnAAdjustmentWorkflow,
        "group_reporting": GroupReportingWorkflow,
        "full_consolidation_pipeline": FullConsolidationPipelineWorkflow,
    }


__all__ = [
    # Entity Mapping
    "EntityMappingWorkflow", "EntityMappingInput", "EntityMappingResult",
    # Boundary Selection
    "BoundarySelectionWorkflow", "BoundarySelectionInput", "BoundarySelectionResult",
    # Entity Data Collection
    "EntityDataCollectionWorkflow", "EntityDataCollectionInput", "EntityDataCollectionResult",
    # Consolidation Execution
    "ConsolidationExecutionWorkflow", "ConsolidationExecInput", "ConsolidationExecResult",
    # Elimination
    "EliminationWorkflow", "EliminationInput", "EliminationResult",
    # M&A Adjustment
    "MnAAdjustmentWorkflow", "MnAInput", "MnAResult",
    # Group Reporting
    "GroupReportingWorkflow", "GroupReportingInput", "GroupReportingResult",
    # Full Pipeline
    "FullConsolidationPipelineWorkflow", "FullConsolidationInput", "FullConsolidationResult",
    # Helper
    "get_loaded_workflows",
]
