# -*- coding: utf-8 -*-
"""
PACK-042 Scope 3 Starter Pack - Workflow Orchestration
=============================================================

Complete Scope 3 GHG value chain inventory workflow orchestrators for the
GHG Protocol Corporate Value Chain (Scope 3) Standard, ISO 14064-1:2018,
and multi-framework disclosure compliance across all 15 Scope 3 categories.

Each workflow coordinates GreenLang MRV agents (MRV-014 through MRV-030),
DATA agents, calculation engines, and validation systems into end-to-end
Scope 3 inventory processes covering screening, data collection, category
calculation, consolidation with double-counting resolution, hotspot
identification, supplier engagement, and multi-framework disclosure.

Workflows:
    - Scope3ScreeningWorkflow: 4-phase rapid screening of all 15 categories
      using EEIO spend-based factors, relevance ranking, and methodology
      tier recommendations.

    - CategoryDataCollectionWorkflow: 4-phase data collection with per-category
      requirement templates (15 categories x 3 tiers), multi-source ingestion,
      deterministic validation (completeness, units, dates, plausibility), and
      outlier detection via IQR method.

    - CategoryCalculationWorkflow: 4-phase calculation with data sufficiency
      validation, MRV agent routing (MRV-014 through MRV-028 via MRV-029),
      parallel execution support, and sector benchmark cross-checks.

    - ConsolidationWorkflow: 4-phase consolidation with upstream/downstream
      aggregation, 12-rule double-counting detection engine, Scope 1+2
      integration from PACK-041, and reconciled totals with audit trail.

    - HotspotWorkflow: 4-phase hotspot identification with Pareto analysis,
      composite materiality scoring (magnitude + DQR + reduction potential),
      sector benchmarking, and prioritized reduction roadmap with ROI.

    - DisclosureWorkflow: 4-phase multi-framework disclosure with requirement
      mapping (GHG Protocol, ESRS E1, CDP, SBTi, SEC, SB 253, ISO 14064,
      TCFD), compliance gap analysis, and framework-native output generation.

    - SupplierEngagementWorkflow: 4-phase supplier engagement with spend-based
      prioritization, standardized questionnaire generation, response tracking,
      and 5-level data quality scoring with tier upgrade recommendations.

    - FullScope3PipelineWorkflow: 8-phase end-to-end orchestrator invoking all
      sub-workflows in sequence with full data handoff, plus cross-cutting
      data quality assessment and Monte Carlo uncertainty analysis.

Author: GreenLang Platform Team
Version: 42.0.0
"""

_MODULE_VERSION: str = "42.0.0"

from typing import Dict, List, Optional, Type

# ---------------------------------------------------------------------------
# Scope 3 Screening Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_042_scope_3_starter.workflows.scope3_screening_workflow import (
        Scope3ScreeningWorkflow,
        Scope3ScreeningInput,
        ScreeningOutput,
        OrganizationProfile,
        SpendRecord,
        CategoryScreeningResult,
        WorkflowState,
        Scope3Category,
        MethodologyTier,
        RelevanceLevel,
        SectorClassification,
    )
except ImportError:
    Scope3ScreeningWorkflow = None  # type: ignore[assignment,misc]
    Scope3ScreeningInput = None  # type: ignore[assignment,misc]
    ScreeningOutput = None  # type: ignore[assignment,misc]
    OrganizationProfile = None  # type: ignore[assignment,misc]
    SpendRecord = None  # type: ignore[assignment,misc]
    CategoryScreeningResult = None  # type: ignore[assignment,misc]
    WorkflowState = None  # type: ignore[assignment,misc]
    Scope3Category = None  # type: ignore[assignment,misc]
    MethodologyTier = None  # type: ignore[assignment,misc]
    RelevanceLevel = None  # type: ignore[assignment,misc]
    SectorClassification = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Category Data Collection Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_042_scope_3_starter.workflows.category_data_collection_workflow import (
        CategoryDataCollectionWorkflow,
        CategoryDataCollectionInput,
        CategoryDataCollectionResult,
        CategoryDataRequirements,
        DataRequirementField,
        IngestedDataRecord,
        ValidationIssue,
        CategoryCollectionProgress,
        DataSourceType,
        ValidationSeverity,
        CompletionStatus,
    )
except ImportError:
    CategoryDataCollectionWorkflow = None  # type: ignore[assignment,misc]
    CategoryDataCollectionInput = None  # type: ignore[assignment,misc]
    CategoryDataCollectionResult = None  # type: ignore[assignment,misc]
    CategoryDataRequirements = None  # type: ignore[assignment,misc]
    DataRequirementField = None  # type: ignore[assignment,misc]
    IngestedDataRecord = None  # type: ignore[assignment,misc]
    ValidationIssue = None  # type: ignore[assignment,misc]
    CategoryCollectionProgress = None  # type: ignore[assignment,misc]
    DataSourceType = None  # type: ignore[assignment,misc]
    ValidationSeverity = None  # type: ignore[assignment,misc]
    CompletionStatus = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Category Calculation Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_042_scope_3_starter.workflows.category_calculation_workflow import (
        CategoryCalculationWorkflow,
        CategoryCalculationInput,
        CategoryCalculationOutput,
        CategoryActivityData,
        DataSufficiencyCheck,
        AgentRoutingEntry,
        CategoryCalculationResult,
        BenchmarkComparison,
        DataSufficiencyLevel,
        CalculationStatus,
        BenchmarkResult,
    )
except ImportError:
    CategoryCalculationWorkflow = None  # type: ignore[assignment,misc]
    CategoryCalculationInput = None  # type: ignore[assignment,misc]
    CategoryCalculationOutput = None  # type: ignore[assignment,misc]
    CategoryActivityData = None  # type: ignore[assignment,misc]
    DataSufficiencyCheck = None  # type: ignore[assignment,misc]
    AgentRoutingEntry = None  # type: ignore[assignment,misc]
    CategoryCalculationResult = None  # type: ignore[assignment,misc]
    BenchmarkComparison = None  # type: ignore[assignment,misc]
    DataSufficiencyLevel = None  # type: ignore[assignment,misc]
    CalculationStatus = None  # type: ignore[assignment,misc]
    BenchmarkResult = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Consolidation Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_042_scope_3_starter.workflows.consolidation_workflow import (
        ConsolidationWorkflow,
        ConsolidationInput,
        ConsolidationOutput,
        CategoryEmission,
        DoubleCountFlag,
        Scope12Data,
        AuditTrailEntry,
        DoubleCountType,
        ResolutionAction,
    )
except ImportError:
    ConsolidationWorkflow = None  # type: ignore[assignment,misc]
    ConsolidationInput = None  # type: ignore[assignment,misc]
    ConsolidationOutput = None  # type: ignore[assignment,misc]
    CategoryEmission = None  # type: ignore[assignment,misc]
    DoubleCountFlag = None  # type: ignore[assignment,misc]
    Scope12Data = None  # type: ignore[assignment,misc]
    AuditTrailEntry = None  # type: ignore[assignment,misc]
    DoubleCountType = None  # type: ignore[assignment,misc]
    ResolutionAction = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Hotspot Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_042_scope_3_starter.workflows.hotspot_workflow import (
        HotspotWorkflow,
        HotspotInput,
        HotspotOutput,
        CategoryResult,
        ParetoEntry,
        MaterialityScore,
        BenchmarkEntry,
        ReductionAction,
        HotspotPriority,
        ActionType,
        BenchmarkPosition,
    )
except ImportError:
    HotspotWorkflow = None  # type: ignore[assignment,misc]
    HotspotInput = None  # type: ignore[assignment,misc]
    HotspotOutput = None  # type: ignore[assignment,misc]
    CategoryResult = None  # type: ignore[assignment,misc]
    ParetoEntry = None  # type: ignore[assignment,misc]
    MaterialityScore = None  # type: ignore[assignment,misc]
    BenchmarkEntry = None  # type: ignore[assignment,misc]
    ReductionAction = None  # type: ignore[assignment,misc]
    HotspotPriority = None  # type: ignore[assignment,misc]
    ActionType = None  # type: ignore[assignment,misc]
    BenchmarkPosition = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Disclosure Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_042_scope_3_starter.workflows.disclosure_workflow import (
        DisclosureWorkflow,
        DisclosureInput,
        DisclosureOutput,
        InventoryData,
        FrameworkRequirement,
        ComplianceScore,
        ComplianceGap,
        DisclosureDocument,
        DisclosureFramework,
        ComplianceStatus,
        OutputFormat,
        GapSeverity,
    )
except ImportError:
    DisclosureWorkflow = None  # type: ignore[assignment,misc]
    DisclosureInput = None  # type: ignore[assignment,misc]
    DisclosureOutput = None  # type: ignore[assignment,misc]
    InventoryData = None  # type: ignore[assignment,misc]
    FrameworkRequirement = None  # type: ignore[assignment,misc]
    ComplianceScore = None  # type: ignore[assignment,misc]
    ComplianceGap = None  # type: ignore[assignment,misc]
    DisclosureDocument = None  # type: ignore[assignment,misc]
    DisclosureFramework = None  # type: ignore[assignment,misc]
    ComplianceStatus = None  # type: ignore[assignment,misc]
    OutputFormat = None  # type: ignore[assignment,misc]
    GapSeverity = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Supplier Engagement Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_042_scope_3_starter.workflows.supplier_engagement_workflow import (
        SupplierEngagementWorkflow,
        SupplierEngagementInput,
        SupplierEngagementOutput,
        SupplierRecord,
        SupplierPrioritization,
        DataRequest,
        QualityAssessment,
        SupplierTier,
        ResponseStatus,
        DataQualityLevel,
        QuestionnaireType,
    )
except ImportError:
    SupplierEngagementWorkflow = None  # type: ignore[assignment,misc]
    SupplierEngagementInput = None  # type: ignore[assignment,misc]
    SupplierEngagementOutput = None  # type: ignore[assignment,misc]
    SupplierRecord = None  # type: ignore[assignment,misc]
    SupplierPrioritization = None  # type: ignore[assignment,misc]
    DataRequest = None  # type: ignore[assignment,misc]
    QualityAssessment = None  # type: ignore[assignment,misc]
    SupplierTier = None  # type: ignore[assignment,misc]
    ResponseStatus = None  # type: ignore[assignment,misc]
    DataQualityLevel = None  # type: ignore[assignment,misc]
    QuestionnaireType = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Full Scope 3 Pipeline Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_042_scope_3_starter.workflows.full_scope3_pipeline_workflow import (
        FullScope3PipelineWorkflow,
        FullScope3PipelineInput,
        FullScope3PipelineOutput,
        Scope12IntegrationData,
        DataQualityResult,
        UncertaintyResult,
        DataQualityRating,
    )
except ImportError:
    FullScope3PipelineWorkflow = None  # type: ignore[assignment,misc]
    FullScope3PipelineInput = None  # type: ignore[assignment,misc]
    FullScope3PipelineOutput = None  # type: ignore[assignment,misc]
    Scope12IntegrationData = None  # type: ignore[assignment,misc]
    DataQualityResult = None  # type: ignore[assignment,misc]
    UncertaintyResult = None  # type: ignore[assignment,misc]
    DataQualityRating = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Type Aliases
# ---------------------------------------------------------------------------

# Screening
ScreeningInput = Scope3ScreeningInput
ScreeningResult = ScreeningOutput

# Data Collection
DataCollectionInput = CategoryDataCollectionInput
DataCollectionResult = CategoryDataCollectionResult

# Calculation
CalcInput = CategoryCalculationInput
CalcOutput = CategoryCalculationOutput

# Consolidation
ConsInput = ConsolidationInput
ConsOutput = ConsolidationOutput

# Hotspot
HotInput = HotspotInput
HotOutput = HotspotOutput

# Disclosure
DiscInput = DisclosureInput
DiscOutput = DisclosureOutput

# Supplier Engagement
SupplierInput = SupplierEngagementInput
SupplierOutput = SupplierEngagementOutput

# Full Pipeline
PipelineInput = FullScope3PipelineInput
PipelineOutput = FullScope3PipelineOutput


# ---------------------------------------------------------------------------
# Workflow Catalog
# ---------------------------------------------------------------------------

WORKFLOW_CATALOG: Dict[str, Dict[str, object]] = {
    "scope3_screening": {
        "name": "Scope 3 Screening",
        "description": (
            "Rapid screening-level assessment of all 15 Scope 3 categories "
            "using EEIO spend-based emission factors"
        ),
        "phases": 4,
        "duration": "2-4 hours",
        "class": Scope3ScreeningWorkflow,
    },
    "category_data_collection": {
        "name": "Category Data Collection",
        "description": (
            "Structured data collection for selected Scope 3 categories "
            "with per-category requirement templates and validation"
        ),
        "phases": 4,
        "duration": "1-3 weeks per category",
        "class": CategoryDataCollectionWorkflow,
    },
    "category_calculation": {
        "name": "Category Calculation",
        "description": (
            "Emission calculation for selected categories using MRV agents "
            "MRV-014 through MRV-028 with data sufficiency checks"
        ),
        "phases": 4,
        "duration": "1-4 hours per category",
        "class": CategoryCalculationWorkflow,
    },
    "consolidation": {
        "name": "Consolidation",
        "description": (
            "Consolidate category results with 12-rule double-counting "
            "detection, Scope 1+2 integration, and reconciliation"
        ),
        "phases": 4,
        "duration": "1-2 hours",
        "class": ConsolidationWorkflow,
    },
    "hotspot_analysis": {
        "name": "Hotspot Analysis",
        "description": (
            "Pareto analysis, materiality scoring, sector benchmarking, "
            "and prioritized reduction roadmap with ROI"
        ),
        "phases": 4,
        "duration": "2-4 hours",
        "class": HotspotWorkflow,
    },
    "disclosure": {
        "name": "Disclosure",
        "description": (
            "Multi-framework Scope 3 disclosure generation for GHG Protocol, "
            "ESRS E1, CDP, SBTi, SEC, SB 253, ISO 14064, and TCFD"
        ),
        "phases": 4,
        "duration": "2-4 hours",
        "class": DisclosureWorkflow,
    },
    "supplier_engagement": {
        "name": "Supplier Engagement",
        "description": (
            "Supplier prioritization, questionnaire generation, response "
            "tracking, and data quality assessment with tier upgrades"
        ),
        "phases": 4,
        "duration": "Ongoing (quarterly cycle)",
        "class": SupplierEngagementWorkflow,
    },
    "full_scope3_pipeline": {
        "name": "Full Scope 3 Pipeline",
        "description": (
            "End-to-end Scope 3 inventory from screening through disclosure "
            "with data quality and Monte Carlo uncertainty analysis"
        ),
        "phases": 8,
        "duration": "2-6 weeks (first-time)",
        "class": FullScope3PipelineWorkflow,
    },
}


# ---------------------------------------------------------------------------
# Workflow Registry
# ---------------------------------------------------------------------------


class WorkflowRegistry:
    """
    Registry for PACK-042 Scope 3 Starter Pack workflows.

    Provides lookup, listing, and filtering of available workflows.

    Example:
        >>> registry = WorkflowRegistry()
        >>> wf_class = registry.get_workflow("scope3_screening")
        >>> wf = wf_class()
    """

    def __init__(self) -> None:
        """Initialize WorkflowRegistry with the workflow catalog."""
        self._catalog = WORKFLOW_CATALOG

    def get_workflow(self, name: str) -> Optional[type]:
        """
        Get a workflow class by name.

        Args:
            name: Workflow name (e.g. 'scope3_screening').

        Returns:
            Workflow class or None if not found/not loaded.
        """
        entry = self._catalog.get(name)
        if entry is None:
            return None
        return entry.get("class")  # type: ignore[return-value]

    def list_workflows(self) -> List[Dict[str, object]]:
        """
        List all available workflows with metadata.

        Returns:
            List of workflow metadata dicts.
        """
        return [
            {
                "key": key,
                "name": entry["name"],
                "description": entry["description"],
                "phases": entry["phases"],
                "duration": entry["duration"],
                "available": entry["class"] is not None,
            }
            for key, entry in self._catalog.items()
        ]

    def get_by_category(self, category: str) -> List[Dict[str, object]]:
        """
        Get workflows by category keyword.

        Args:
            category: Keyword to search (e.g. 'screening', 'calculation').

        Returns:
            List of matching workflow metadata dicts.
        """
        keyword = category.lower()
        return [
            wf for wf in self.list_workflows()
            if keyword in str(wf.get("key", "")).lower()
            or keyword in str(wf.get("description", "")).lower()
        ]

    def get_loaded_workflows(self) -> Dict[str, Optional[type]]:
        """
        Return a dictionary of workflow names to their classes.

        Workflows that failed to import will have None values.

        Returns:
            Dict mapping workflow name to workflow class or None.
        """
        return {
            key: entry.get("class")  # type: ignore[misc]
            for key, entry in self._catalog.items()
        }


def get_loaded_workflows() -> Dict[str, Optional[type]]:
    """
    Return a dictionary of workflow names to their classes.

    Workflows that failed to import will have None values.

    Returns:
        Dict mapping workflow name to workflow class or None.
    """
    return {
        "scope3_screening": Scope3ScreeningWorkflow,
        "category_data_collection": CategoryDataCollectionWorkflow,
        "category_calculation": CategoryCalculationWorkflow,
        "consolidation": ConsolidationWorkflow,
        "hotspot_analysis": HotspotWorkflow,
        "disclosure": DisclosureWorkflow,
        "supplier_engagement": SupplierEngagementWorkflow,
        "full_scope3_pipeline": FullScope3PipelineWorkflow,
    }


__all__ = [
    # --- Module Version ---
    "_MODULE_VERSION",
    # --- Scope 3 Screening Workflow ---
    "Scope3ScreeningWorkflow",
    "Scope3ScreeningInput",
    "ScreeningOutput",
    "OrganizationProfile",
    "SpendRecord",
    "CategoryScreeningResult",
    "WorkflowState",
    "Scope3Category",
    "MethodologyTier",
    "RelevanceLevel",
    "SectorClassification",
    # --- Category Data Collection Workflow ---
    "CategoryDataCollectionWorkflow",
    "CategoryDataCollectionInput",
    "CategoryDataCollectionResult",
    "CategoryDataRequirements",
    "DataRequirementField",
    "IngestedDataRecord",
    "ValidationIssue",
    "CategoryCollectionProgress",
    "DataSourceType",
    "ValidationSeverity",
    "CompletionStatus",
    # --- Category Calculation Workflow ---
    "CategoryCalculationWorkflow",
    "CategoryCalculationInput",
    "CategoryCalculationOutput",
    "CategoryActivityData",
    "DataSufficiencyCheck",
    "AgentRoutingEntry",
    "CategoryCalculationResult",
    "BenchmarkComparison",
    "DataSufficiencyLevel",
    "CalculationStatus",
    "BenchmarkResult",
    # --- Consolidation Workflow ---
    "ConsolidationWorkflow",
    "ConsolidationInput",
    "ConsolidationOutput",
    "CategoryEmission",
    "DoubleCountFlag",
    "Scope12Data",
    "AuditTrailEntry",
    "DoubleCountType",
    "ResolutionAction",
    # --- Hotspot Workflow ---
    "HotspotWorkflow",
    "HotspotInput",
    "HotspotOutput",
    "CategoryResult",
    "ParetoEntry",
    "MaterialityScore",
    "BenchmarkEntry",
    "ReductionAction",
    "HotspotPriority",
    "ActionType",
    "BenchmarkPosition",
    # --- Disclosure Workflow ---
    "DisclosureWorkflow",
    "DisclosureInput",
    "DisclosureOutput",
    "InventoryData",
    "FrameworkRequirement",
    "ComplianceScore",
    "ComplianceGap",
    "DisclosureDocument",
    "DisclosureFramework",
    "ComplianceStatus",
    "OutputFormat",
    "GapSeverity",
    # --- Supplier Engagement Workflow ---
    "SupplierEngagementWorkflow",
    "SupplierEngagementInput",
    "SupplierEngagementOutput",
    "SupplierRecord",
    "SupplierPrioritization",
    "DataRequest",
    "QualityAssessment",
    "SupplierTier",
    "ResponseStatus",
    "DataQualityLevel",
    "QuestionnaireType",
    # --- Full Scope 3 Pipeline Workflow ---
    "FullScope3PipelineWorkflow",
    "FullScope3PipelineInput",
    "FullScope3PipelineOutput",
    "Scope12IntegrationData",
    "DataQualityResult",
    "UncertaintyResult",
    "DataQualityRating",
    # --- Type Aliases ---
    "ScreeningInput",
    "ScreeningResult",
    "DataCollectionInput",
    "DataCollectionResult",
    "CalcInput",
    "CalcOutput",
    "ConsInput",
    "ConsOutput",
    "HotInput",
    "HotOutput",
    "DiscInput",
    "DiscOutput",
    "SupplierInput",
    "SupplierOutput",
    "PipelineInput",
    "PipelineOutput",
    # --- Catalog & Registry ---
    "WORKFLOW_CATALOG",
    "WorkflowRegistry",
    "get_loaded_workflows",
]
