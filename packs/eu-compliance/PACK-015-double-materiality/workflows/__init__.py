# -*- coding: utf-8 -*-
"""
PACK-015 Double Materiality Pack - Workflow Orchestration
=============================================================

Double materiality workflow orchestrators for CSRD/ESRS compliance
operations. Each workflow coordinates GreenLang agents, data pipelines,
AI engines, and validation systems into end-to-end double materiality
assessment processes covering impact assessment, financial assessment,
stakeholder engagement, IRO identification, materiality matrix
construction, ESRS mapping, full DMA orchestration, and annual updates.

Workflows:
    - ImpactAssessmentWorkflow: 4-phase inside-out materiality assessment
      with sustainability matter collection, ESRS topic identification,
      severity scoring (scale/scope/irremediability/likelihood), and
      impact ranking with configurable thresholds.

    - FinancialAssessmentWorkflow: 4-phase outside-in materiality assessment
      with financial exposure collection, risk/opportunity KPI mapping,
      magnitude/likelihood/time-horizon scoring, and financial ranking.

    - StakeholderEngagementWorkflow: 5-phase stakeholder engagement with
      identification, influence-impact matrix mapping, consultation
      recording, cross-group synthesis, and ESRS 1 s22-23 validation.

    - IROIdentificationWorkflow: 4-phase IRO lifecycle with value chain
      mapping, IRO discovery per ESRS topic, classification (impact/risk/
      opportunity), and composite-score prioritization.

    - MaterialityMatrixWorkflow: 3-phase matrix construction with score
      aggregation, 2x2 quadrant generation, and sector-specific threshold
      application with year-over-year comparison.

    - ESRSMappingWorkflow: 3-phase disclosure mapping with material topic
      selection, ESRS DR catalog mapping, and gap analysis with effort
      estimates and prioritization.

    - FullDMAWorkflow: 6-phase end-to-end orchestration coordinating all
      sub-workflows into a complete Double Materiality Assessment with
      aggregated topic results and completeness tracking.

    - DMAUpdateWorkflow: 4-phase annual refresh with change detection,
      re-assessment, delta analysis against prior year, and publication
      with full audit trail.

Author: GreenLang Team
Version: 15.0.0
"""

# ---------------------------------------------------------------------------
# Impact Assessment Workflow
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_015_double_materiality.workflows.impact_assessment_workflow import (
    ImpactAssessmentWorkflow,
    ImpactAssessmentInput,
    ImpactAssessmentResult,
    SustainabilityMatter,
    SectorData,
    SeverityScore,
    RankedImpact,
)

# ---------------------------------------------------------------------------
# Financial Assessment Workflow
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_015_double_materiality.workflows.financial_assessment_workflow import (
    FinancialAssessmentWorkflow,
    FinancialAssessmentInput,
    FinancialAssessmentResult,
    FinancialExposure,
    FinancialScore,
    RankedFinancialItem,
)

# ---------------------------------------------------------------------------
# Stakeholder Engagement Workflow
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_015_double_materiality.workflows.stakeholder_engagement_workflow import (
    StakeholderEngagementWorkflow,
    StakeholderEngagementInput,
    StakeholderEngagementResult,
    Stakeholder,
    ConsultationRecord,
    StakeholderFinding,
    ValidationCheck,
)

# ---------------------------------------------------------------------------
# IRO Identification Workflow
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_015_double_materiality.workflows.iro_identification_workflow import (
    IROIdentificationWorkflow,
    IROIdentificationInput,
    IROIdentificationResult,
    ValueChainActivity,
    IRORecord,
)

# ---------------------------------------------------------------------------
# Materiality Matrix Workflow
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_015_double_materiality.workflows.materiality_matrix_workflow import (
    MaterialityMatrixWorkflow,
    MaterialityMatrixInput,
    MaterialityMatrixResult,
    TopicScore,
    MatrixEntry,
    ThresholdResult,
)

# ---------------------------------------------------------------------------
# ESRS Mapping Workflow
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_015_double_materiality.workflows.esrs_mapping_workflow import (
    ESRSMappingWorkflow,
    ESRSMappingInput,
    ESRSMappingResult,
    MaterialTopic,
    DisclosureRequirement,
    DisclosureGap,
)

# ---------------------------------------------------------------------------
# Full DMA Workflow
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_015_double_materiality.workflows.full_dma_workflow import (
    FullDMAWorkflow,
    FullDMAInput,
    FullDMAResult,
    DMATopicResult,
)

# ---------------------------------------------------------------------------
# DMA Update Workflow
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_015_double_materiality.workflows.dma_update_workflow import (
    DMAUpdateWorkflow,
    DMAUpdateInput,
    DMAUpdateResult,
    DetectedChange,
    PriorDMARecord,
    ReAssessedTopic,
    DeltaEntry,
)

__all__ = [
    # --- Impact Assessment Workflow ---
    "ImpactAssessmentWorkflow",
    "ImpactAssessmentInput",
    "ImpactAssessmentResult",
    "SustainabilityMatter",
    "SectorData",
    "SeverityScore",
    "RankedImpact",
    # --- Financial Assessment Workflow ---
    "FinancialAssessmentWorkflow",
    "FinancialAssessmentInput",
    "FinancialAssessmentResult",
    "FinancialExposure",
    "FinancialScore",
    "RankedFinancialItem",
    # --- Stakeholder Engagement Workflow ---
    "StakeholderEngagementWorkflow",
    "StakeholderEngagementInput",
    "StakeholderEngagementResult",
    "Stakeholder",
    "ConsultationRecord",
    "StakeholderFinding",
    "ValidationCheck",
    # --- IRO Identification Workflow ---
    "IROIdentificationWorkflow",
    "IROIdentificationInput",
    "IROIdentificationResult",
    "ValueChainActivity",
    "IRORecord",
    # --- Materiality Matrix Workflow ---
    "MaterialityMatrixWorkflow",
    "MaterialityMatrixInput",
    "MaterialityMatrixResult",
    "TopicScore",
    "MatrixEntry",
    "ThresholdResult",
    # --- ESRS Mapping Workflow ---
    "ESRSMappingWorkflow",
    "ESRSMappingInput",
    "ESRSMappingResult",
    "MaterialTopic",
    "DisclosureRequirement",
    "DisclosureGap",
    # --- Full DMA Workflow ---
    "FullDMAWorkflow",
    "FullDMAInput",
    "FullDMAResult",
    "DMATopicResult",
    # --- DMA Update Workflow ---
    "DMAUpdateWorkflow",
    "DMAUpdateInput",
    "DMAUpdateResult",
    "DetectedChange",
    "PriorDMARecord",
    "ReAssessedTopic",
    "DeltaEntry",
]
