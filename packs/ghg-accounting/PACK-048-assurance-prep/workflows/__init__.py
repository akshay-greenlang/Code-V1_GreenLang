# -*- coding: utf-8 -*-
"""
PACK-048 GHG Assurance Prep Pack - Workflow Orchestration
==================================================================

Complete assurance preparation workflow orchestrators for the ISAE 3410,
ISO 14064-3, AA1000AS v3, ISAE 3000, SSAE 18, EU CSRD, US SEC Climate
Disclosure, California SB 253, and GHG Protocol verification requirements.
Each workflow coordinates assurance processes covering readiness assessment,
evidence gathering, control testing, verifier engagement, materiality and
sampling, regulatory mapping, cost and timeline estimation, and full
end-to-end pipeline orchestration.

Workflows:
    - ReadinessAssessmentWorkflow: 5-phase workflow for assurance readiness
      assessment with standard selection, checklist generation, evidence
      checking, score calculation, and gap reporting across ISAE 3410,
      ISO 14064-3, and AA1000AS v3 assurance standards.

    - EvidenceCollectionWorkflow: 5-phase workflow for evidence gathering
      with scope inventory, source identification, document collection,
      quality grading, and evidence package build with SHA-256 hash chains.

    - ControlTestingWorkflow: 5-phase workflow for internal control testing
      with control identification from a 25-control register, design
      assessment, statistical sample selection, test execution, and
      deficiency reporting.

    - VerifierEngagementWorkflow: 5-phase workflow for verifier lifecycle
      management with engagement scoping, verifier onboarding, query
      management, finding tracking, and engagement closeout.

    - MaterialitySamplingWorkflow: 5-phase workflow for materiality and
      sampling with materiality calculation, population identification,
      stratification, statistical sample sizing, and selection plan.

    - RegulatoryMappingWorkflow: 4-phase workflow for multi-jurisdiction
      regulatory mapping with jurisdiction identification, requirement
      mapping, gap analysis, and compliance planning across 9 jurisdictions.

    - CostTimelineWorkflow: 5-phase workflow for assurance cost estimation
      and timeline planning with engagement scoping, cost estimation,
      timeline planning, resource allocation, and budget approval.

    - FullAssurancePrepPipelineWorkflow: 8-phase end-to-end orchestrator
      invoking all sub-workflows with conditional phase execution,
      phase-level caching, checkpoint support, and full provenance chain.

Author: GreenLang Team
Version: 48.0.0
"""

from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Readiness Assessment Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_048_assurance_prep.workflows.readiness_assessment_workflow import (
        ReadinessAssessmentWorkflow,
        ReadinessAssessmentInput,
        ReadinessAssessmentResult,
        ReadinessPhase,
        AssuranceStandard,
        AssuranceLevel,
        EvidenceStatus,
        ReadinessBand,
        GapPriority,
        ChecklistCategory,
        ChecklistItem,
        CategoryScore,
        GapItem,
        StandardConfig,
    )
except ImportError:
    ReadinessAssessmentWorkflow = None  # type: ignore[assignment,misc]
    ReadinessAssessmentInput = None  # type: ignore[assignment,misc]
    ReadinessAssessmentResult = None  # type: ignore[assignment,misc]
    ReadinessPhase = None  # type: ignore[assignment,misc]
    AssuranceStandard = None  # type: ignore[assignment,misc]
    AssuranceLevel = None  # type: ignore[assignment,misc]
    EvidenceStatus = None  # type: ignore[assignment,misc]
    ReadinessBand = None  # type: ignore[assignment,misc]
    GapPriority = None  # type: ignore[assignment,misc]
    ChecklistCategory = None  # type: ignore[assignment,misc]
    ChecklistItem = None  # type: ignore[assignment,misc]
    CategoryScore = None  # type: ignore[assignment,misc]
    GapItem = None  # type: ignore[assignment,misc]
    StandardConfig = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Evidence Collection Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_048_assurance_prep.workflows.evidence_collection_workflow import (
        EvidenceCollectionWorkflow,
        EvidenceCollectionInput,
        EvidenceCollectionResult,
        EvidenceCollectionPhase,
        EmissionScope,
        SourceType,
        EvidenceType,
        EvidenceQualityGrade,
        DocumentStatus,
        EmissionSourceRecord,
        EvidenceItem,
        PackageIndex,
        PackageSummary,
    )
except ImportError:
    EvidenceCollectionWorkflow = None  # type: ignore[assignment,misc]
    EvidenceCollectionInput = None  # type: ignore[assignment,misc]
    EvidenceCollectionResult = None  # type: ignore[assignment,misc]
    EvidenceCollectionPhase = None  # type: ignore[assignment,misc]
    EmissionScope = None  # type: ignore[assignment,misc]
    SourceType = None  # type: ignore[assignment,misc]
    EvidenceType = None  # type: ignore[assignment,misc]
    EvidenceQualityGrade = None  # type: ignore[assignment,misc]
    DocumentStatus = None  # type: ignore[assignment,misc]
    EmissionSourceRecord = None  # type: ignore[assignment,misc]
    EvidenceItem = None  # type: ignore[assignment,misc]
    PackageIndex = None  # type: ignore[assignment,misc]
    PackageSummary = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Control Testing Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_048_assurance_prep.workflows.control_testing_workflow import (
        ControlTestingWorkflow,
        ControlTestingInput,
        ControlTestingResult,
        ControlTestingPhase,
        ControlCategory,
        ControlType,
        ControlFrequency,
        DesignEffectiveness,
        TestResult,
        DeficiencyClassification,
        RiskLevel,
        ControlRecord,
        DeficiencyRecord,
        ControlTestSummary,
    )
except ImportError:
    ControlTestingWorkflow = None  # type: ignore[assignment,misc]
    ControlTestingInput = None  # type: ignore[assignment,misc]
    ControlTestingResult = None  # type: ignore[assignment,misc]
    ControlTestingPhase = None  # type: ignore[assignment,misc]
    ControlCategory = None  # type: ignore[assignment,misc]
    ControlType = None  # type: ignore[assignment,misc]
    ControlFrequency = None  # type: ignore[assignment,misc]
    DesignEffectiveness = None  # type: ignore[assignment,misc]
    TestResult = None  # type: ignore[assignment,misc]
    DeficiencyClassification = None  # type: ignore[assignment,misc]
    RiskLevel = None  # type: ignore[assignment,misc]
    ControlRecord = None  # type: ignore[assignment,misc]
    DeficiencyRecord = None  # type: ignore[assignment,misc]
    ControlTestSummary = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Verifier Engagement Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_048_assurance_prep.workflows.verifier_engagement_workflow import (
        VerifierEngagementWorkflow,
        VerifierEngagementInput,
        VerifierEngagementResult,
        EngagementPhase,
        EngagementStatus,
        QueryStatus,
        QueryPriority,
        FindingSeverity,
        FindingStatus,
        OpinionType,
        AccessLevel,
        EngagementScope,
        VerifierAccess,
        QueryRecord,
        FindingRecord,
        EngagementCloseoutRecord,
    )
except ImportError:
    VerifierEngagementWorkflow = None  # type: ignore[assignment,misc]
    VerifierEngagementInput = None  # type: ignore[assignment,misc]
    VerifierEngagementResult = None  # type: ignore[assignment,misc]
    EngagementPhase = None  # type: ignore[assignment,misc]
    EngagementStatus = None  # type: ignore[assignment,misc]
    QueryStatus = None  # type: ignore[assignment,misc]
    QueryPriority = None  # type: ignore[assignment,misc]
    FindingSeverity = None  # type: ignore[assignment,misc]
    FindingStatus = None  # type: ignore[assignment,misc]
    OpinionType = None  # type: ignore[assignment,misc]
    AccessLevel = None  # type: ignore[assignment,misc]
    EngagementScope = None  # type: ignore[assignment,misc]
    VerifierAccess = None  # type: ignore[assignment,misc]
    QueryRecord = None  # type: ignore[assignment,misc]
    FindingRecord = None  # type: ignore[assignment,misc]
    EngagementCloseoutRecord = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Materiality & Sampling Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_048_assurance_prep.workflows.materiality_sampling_workflow import (
        MaterialitySamplingWorkflow,
        MaterialitySamplingInput,
        MaterialitySamplingResult,
        MaterialitySamplingPhase,
        MaterialityBasis,
        StratumScope,
        SelectionMethod,
        MaterialityResult,
        PopulationItem,
        Stratum,
        SamplingPlan,
    )
except ImportError:
    MaterialitySamplingWorkflow = None  # type: ignore[assignment,misc]
    MaterialitySamplingInput = None  # type: ignore[assignment,misc]
    MaterialitySamplingResult = None  # type: ignore[assignment,misc]
    MaterialitySamplingPhase = None  # type: ignore[assignment,misc]
    MaterialityBasis = None  # type: ignore[assignment,misc]
    StratumScope = None  # type: ignore[assignment,misc]
    SelectionMethod = None  # type: ignore[assignment,misc]
    MaterialityResult = None  # type: ignore[assignment,misc]
    PopulationItem = None  # type: ignore[assignment,misc]
    Stratum = None  # type: ignore[assignment,misc]
    SamplingPlan = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Regulatory Mapping Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_048_assurance_prep.workflows.regulatory_mapping_workflow import (
        RegulatoryMappingWorkflow,
        RegulatoryMappingInput,
        RegulatoryMappingResult,
        RegulatoryMappingPhase,
        Jurisdiction,
        AssuranceRequirementLevel,
        ComplianceStatus,
        ActionPriority,
        JurisdictionRecord,
        RequirementGap,
        ComplianceAction,
    )
except ImportError:
    RegulatoryMappingWorkflow = None  # type: ignore[assignment,misc]
    RegulatoryMappingInput = None  # type: ignore[assignment,misc]
    RegulatoryMappingResult = None  # type: ignore[assignment,misc]
    RegulatoryMappingPhase = None  # type: ignore[assignment,misc]
    Jurisdiction = None  # type: ignore[assignment,misc]
    AssuranceRequirementLevel = None  # type: ignore[assignment,misc]
    ComplianceStatus = None  # type: ignore[assignment,misc]
    ActionPriority = None  # type: ignore[assignment,misc]
    JurisdictionRecord = None  # type: ignore[assignment,misc]
    RequirementGap = None  # type: ignore[assignment,misc]
    ComplianceAction = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Cost & Timeline Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_048_assurance_prep.workflows.cost_timeline_workflow import (
        CostTimelineWorkflow,
        CostTimelineInput,
        CostTimelineResult,
        CostTimelinePhase,
        ComplexityLevel,
        CostCategory,
        EngagementMilestone,
        InternalRole,
        ScopeParameters,
        CostLineItem,
        MilestoneRecord,
        ResourceAllocationRecord,
        BudgetPackage,
    )
except ImportError:
    CostTimelineWorkflow = None  # type: ignore[assignment,misc]
    CostTimelineInput = None  # type: ignore[assignment,misc]
    CostTimelineResult = None  # type: ignore[assignment,misc]
    CostTimelinePhase = None  # type: ignore[assignment,misc]
    ComplexityLevel = None  # type: ignore[assignment,misc]
    CostCategory = None  # type: ignore[assignment,misc]
    EngagementMilestone = None  # type: ignore[assignment,misc]
    InternalRole = None  # type: ignore[assignment,misc]
    ScopeParameters = None  # type: ignore[assignment,misc]
    CostLineItem = None  # type: ignore[assignment,misc]
    MilestoneRecord = None  # type: ignore[assignment,misc]
    ResourceAllocationRecord = None  # type: ignore[assignment,misc]
    BudgetPackage = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Full Assurance Prep Pipeline Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_048_assurance_prep.workflows.full_assurance_prep_pipeline_workflow import (
        FullAssurancePrepPipelineWorkflow,
        FullPipelineInput,
        FullPipelineResult,
        PipelinePhase,
        PipelineMilestoneType,
        ReportType,
        PipelinePhaseStatus,
        PipelineCheckpoint,
        PipelineMilestone,
        PipelineReport,
    )
except ImportError:
    FullAssurancePrepPipelineWorkflow = None  # type: ignore[assignment,misc]
    FullPipelineInput = None  # type: ignore[assignment,misc]
    FullPipelineResult = None  # type: ignore[assignment,misc]
    PipelinePhase = None  # type: ignore[assignment,misc]
    PipelineMilestoneType = None  # type: ignore[assignment,misc]
    ReportType = None  # type: ignore[assignment,misc]
    PipelinePhaseStatus = None  # type: ignore[assignment,misc]
    PipelineCheckpoint = None  # type: ignore[assignment,misc]
    PipelineMilestone = None  # type: ignore[assignment,misc]
    PipelineReport = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Type Aliases
# ---------------------------------------------------------------------------

# Readiness Assessment types
ReadinessIn = ReadinessAssessmentInput
ReadinessOut = ReadinessAssessmentResult

# Evidence Collection types
EvidenceIn = EvidenceCollectionInput
EvidenceOut = EvidenceCollectionResult

# Control Testing types
ControlIn = ControlTestingInput
ControlOut = ControlTestingResult

# Verifier Engagement types
VerifierIn = VerifierEngagementInput
VerifierOut = VerifierEngagementResult

# Materiality & Sampling types
MaterialityIn = MaterialitySamplingInput
MaterialityOut = MaterialitySamplingResult

# Regulatory Mapping types
RegulatoryIn = RegulatoryMappingInput
RegulatoryOut = RegulatoryMappingResult

# Cost & Timeline types
CostIn = CostTimelineInput
CostOut = CostTimelineResult

# Full Pipeline types
FullPipelineIn = FullPipelineInput
FullPipelineOut = FullPipelineResult


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
        "readiness_assessment": ReadinessAssessmentWorkflow,
        "evidence_collection": EvidenceCollectionWorkflow,
        "control_testing": ControlTestingWorkflow,
        "verifier_engagement": VerifierEngagementWorkflow,
        "materiality_sampling": MaterialitySamplingWorkflow,
        "regulatory_mapping": RegulatoryMappingWorkflow,
        "cost_timeline": CostTimelineWorkflow,
        "full_assurance_prep_pipeline": FullAssurancePrepPipelineWorkflow,
    }


__all__ = [
    # --- Readiness Assessment Workflow ---
    "ReadinessAssessmentWorkflow",
    "ReadinessAssessmentInput",
    "ReadinessAssessmentResult",
    "ReadinessPhase",
    "AssuranceStandard",
    "AssuranceLevel",
    "EvidenceStatus",
    "ReadinessBand",
    "GapPriority",
    "ChecklistCategory",
    "ChecklistItem",
    "CategoryScore",
    "GapItem",
    "StandardConfig",
    # --- Evidence Collection Workflow ---
    "EvidenceCollectionWorkflow",
    "EvidenceCollectionInput",
    "EvidenceCollectionResult",
    "EvidenceCollectionPhase",
    "EmissionScope",
    "SourceType",
    "EvidenceType",
    "EvidenceQualityGrade",
    "DocumentStatus",
    "EmissionSourceRecord",
    "EvidenceItem",
    "PackageIndex",
    "PackageSummary",
    # --- Control Testing Workflow ---
    "ControlTestingWorkflow",
    "ControlTestingInput",
    "ControlTestingResult",
    "ControlTestingPhase",
    "ControlCategory",
    "ControlType",
    "ControlFrequency",
    "DesignEffectiveness",
    "TestResult",
    "DeficiencyClassification",
    "RiskLevel",
    "ControlRecord",
    "DeficiencyRecord",
    "ControlTestSummary",
    # --- Verifier Engagement Workflow ---
    "VerifierEngagementWorkflow",
    "VerifierEngagementInput",
    "VerifierEngagementResult",
    "EngagementPhase",
    "EngagementStatus",
    "QueryStatus",
    "QueryPriority",
    "FindingSeverity",
    "FindingStatus",
    "OpinionType",
    "AccessLevel",
    "EngagementScope",
    "VerifierAccess",
    "QueryRecord",
    "FindingRecord",
    "EngagementCloseoutRecord",
    # --- Materiality & Sampling Workflow ---
    "MaterialitySamplingWorkflow",
    "MaterialitySamplingInput",
    "MaterialitySamplingResult",
    "MaterialitySamplingPhase",
    "MaterialityBasis",
    "StratumScope",
    "SelectionMethod",
    "MaterialityResult",
    "PopulationItem",
    "Stratum",
    "SamplingPlan",
    # --- Regulatory Mapping Workflow ---
    "RegulatoryMappingWorkflow",
    "RegulatoryMappingInput",
    "RegulatoryMappingResult",
    "RegulatoryMappingPhase",
    "Jurisdiction",
    "AssuranceRequirementLevel",
    "ComplianceStatus",
    "ActionPriority",
    "JurisdictionRecord",
    "RequirementGap",
    "ComplianceAction",
    # --- Cost & Timeline Workflow ---
    "CostTimelineWorkflow",
    "CostTimelineInput",
    "CostTimelineResult",
    "CostTimelinePhase",
    "ComplexityLevel",
    "CostCategory",
    "EngagementMilestone",
    "InternalRole",
    "ScopeParameters",
    "CostLineItem",
    "MilestoneRecord",
    "ResourceAllocationRecord",
    "BudgetPackage",
    # --- Full Assurance Prep Pipeline Workflow ---
    "FullAssurancePrepPipelineWorkflow",
    "FullPipelineInput",
    "FullPipelineResult",
    "PipelinePhase",
    "PipelineMilestoneType",
    "ReportType",
    "PipelinePhaseStatus",
    "PipelineCheckpoint",
    "PipelineMilestone",
    "PipelineReport",
    # --- Type Aliases ---
    "ReadinessIn",
    "ReadinessOut",
    "EvidenceIn",
    "EvidenceOut",
    "ControlIn",
    "ControlOut",
    "VerifierIn",
    "VerifierOut",
    "MaterialityIn",
    "MaterialityOut",
    "RegulatoryIn",
    "RegulatoryOut",
    "CostIn",
    "CostOut",
    "FullPipelineIn",
    "FullPipelineOut",
    # --- Helpers ---
    "get_loaded_workflows",
]
