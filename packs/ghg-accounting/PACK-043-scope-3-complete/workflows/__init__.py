# -*- coding: utf-8 -*-
"""
PACK-043 Scope 3 Complete Pack - Workflow Orchestration
=============================================================

Advanced Scope 3 GHG enterprise workflows extending PACK-042 Scope 3 Starter
with maturity assessment, LCA integration, multi-entity consolidation,
scenario planning, SBTi target management, climate risk assessment, supplier
programme management, and full enterprise pipeline orchestration.

Each workflow coordinates GreenLang agents, calculation engines, and validation
systems into end-to-end Scope 3 enterprise processes covering the full
lifecycle from maturity assessment through ISAE 3410 assurance readiness.

Workflows:
    - MaturityAssessmentWorkflow: 4-phase maturity assessment with current-state
      scan (tier, DQR, uncertainty), gap analysis, upgrade roadmap with
      dependencies, and ROI-prioritized budget allocation.

    - LCAIntegrationWorkflow: 4-phase product lifecycle assessment integration
      with revenue/volume product selection, BOM-to-factor mapping, lifecycle
      emission factor assignment, and cradle-to-gate/grave footprint with
      sensitivity analysis.

    - MultiEntityWorkflow: 4-phase multi-entity consolidation with entity
      hierarchy mapping, boundary definition (equity/operational/financial
      control), proportional consolidation, and inter-company double-counting
      elimination.

    - ScenarioPlanningWorkflow: 4-phase scenario planning with baseline
      establishment, intervention definition, MACC analysis and what-if
      modelling, and Paris-aligned portfolio optimization.

    - SBTiTargetWorkflow: 4-phase SBTi target management with Scope 3
      materiality check (40% threshold), pathway calculation (1.5C/WB2C),
      target validation (67% coverage, FLAG), and submission package
      generation.

    - ClimateRiskWorkflow: 4-phase climate risk assessment with transition/
      physical/opportunity identification, carbon pricing and supply chain
      exposure quantification, NPV financial impact over 10/20/30-year
      horizons, and IEA NZE / NGFS scenario analysis.

    - SupplierProgrammeWorkflow: 4-phase supplier programme with science-
      aligned target setting, SBTi/RE100/CDP commitment tracking, YoY
      progress measurement, and programme impact on Scope 3 trajectory.

    - FullEnterprisePipelineWorkflow: 8-phase orchestrator invoking all
      sub-workflows in sequence with PACK-042 bridge for inventory
      calculation and ISAE 3410 assurance package generation.

Author: GreenLang Platform Team
Version: 43.0.0
"""

_MODULE_VERSION: str = "43.0.0"

from typing import Dict, List, Optional, Type

# ---------------------------------------------------------------------------
# Maturity Assessment Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_043_scope_3_complete.workflows.maturity_assessment_workflow import (
        MaturityAssessmentWorkflow,
        MaturityAssessmentInput,
        MaturityAssessmentOutput,
        CategoryMaturityState,
        MaturityGap,
        UpgradeStep,
        UpgradeROI,
        MaturityLevel,
        MethodologyTier as MaturityMethodologyTier,
        GapSeverity,
        UpgradeEffort,
    )
except ImportError:
    MaturityAssessmentWorkflow = None  # type: ignore[assignment,misc]
    MaturityAssessmentInput = None  # type: ignore[assignment,misc]
    MaturityAssessmentOutput = None  # type: ignore[assignment,misc]
    CategoryMaturityState = None  # type: ignore[assignment,misc]
    MaturityGap = None  # type: ignore[assignment,misc]
    UpgradeStep = None  # type: ignore[assignment,misc]
    UpgradeROI = None  # type: ignore[assignment,misc]
    MaturityLevel = None  # type: ignore[assignment,misc]
    MaturityMethodologyTier = None  # type: ignore[assignment,misc]
    GapSeverity = None  # type: ignore[assignment,misc]
    UpgradeEffort = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# LCA Integration Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_043_scope_3_complete.workflows.lca_integration_workflow import (
        LCAIntegrationWorkflow,
        LCAIntegrationInput,
        LCAIntegrationOutput,
        ProductRecord,
        ProductBOM,
        BOMComponent,
        FactorAssignment,
        StageFootprint,
        SensitivityResult,
        ProductFootprint,
        LifecycleStage,
        LCABoundary,
        MaterialCategory,
        SensitivityParameter,
    )
except ImportError:
    LCAIntegrationWorkflow = None  # type: ignore[assignment,misc]
    LCAIntegrationInput = None  # type: ignore[assignment,misc]
    LCAIntegrationOutput = None  # type: ignore[assignment,misc]
    ProductRecord = None  # type: ignore[assignment,misc]
    ProductBOM = None  # type: ignore[assignment,misc]
    BOMComponent = None  # type: ignore[assignment,misc]
    FactorAssignment = None  # type: ignore[assignment,misc]
    StageFootprint = None  # type: ignore[assignment,misc]
    SensitivityResult = None  # type: ignore[assignment,misc]
    ProductFootprint = None  # type: ignore[assignment,misc]
    LifecycleStage = None  # type: ignore[assignment,misc]
    LCABoundary = None  # type: ignore[assignment,misc]
    MaterialCategory = None  # type: ignore[assignment,misc]
    SensitivityParameter = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Multi-Entity Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_043_scope_3_complete.workflows.multi_entity_workflow import (
        MultiEntityWorkflow,
        MultiEntityInput,
        MultiEntityOutput,
        EntityRecord,
        EntityScope3Data,
        EntityBoundary,
        ConsolidatedCategory,
        DoubleCountFlag,
        IntercompanyTransaction,
        EntityType,
        ConsolidationApproach,
        DoubleCountType,
    )
except ImportError:
    MultiEntityWorkflow = None  # type: ignore[assignment,misc]
    MultiEntityInput = None  # type: ignore[assignment,misc]
    MultiEntityOutput = None  # type: ignore[assignment,misc]
    EntityRecord = None  # type: ignore[assignment,misc]
    EntityScope3Data = None  # type: ignore[assignment,misc]
    EntityBoundary = None  # type: ignore[assignment,misc]
    ConsolidatedCategory = None  # type: ignore[assignment,misc]
    DoubleCountFlag = None  # type: ignore[assignment,misc]
    IntercompanyTransaction = None  # type: ignore[assignment,misc]
    EntityType = None  # type: ignore[assignment,misc]
    ConsolidationApproach = None  # type: ignore[assignment,misc]
    DoubleCountType = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Scenario Planning Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_043_scope_3_complete.workflows.scenario_planning_workflow import (
        ScenarioPlanningWorkflow,
        ScenarioPlanningInput,
        ScenarioPlanningOutput,
        BaselineCategory,
        Intervention,
        MACCEntry,
        ScenarioResult,
        OptimalPortfolio,
        InterventionType,
        ScenarioType,
        AlignmentStatus,
    )
except ImportError:
    ScenarioPlanningWorkflow = None  # type: ignore[assignment,misc]
    ScenarioPlanningInput = None  # type: ignore[assignment,misc]
    ScenarioPlanningOutput = None  # type: ignore[assignment,misc]
    BaselineCategory = None  # type: ignore[assignment,misc]
    Intervention = None  # type: ignore[assignment,misc]
    MACCEntry = None  # type: ignore[assignment,misc]
    ScenarioResult = None  # type: ignore[assignment,misc]
    OptimalPortfolio = None  # type: ignore[assignment,misc]
    InterventionType = None  # type: ignore[assignment,misc]
    ScenarioType = None  # type: ignore[assignment,misc]
    AlignmentStatus = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# SBTi Target Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_043_scope_3_complete.workflows.sbti_target_workflow import (
        SBTiTargetWorkflow,
        SBTiTargetInput,
        SBTiTargetOutput,
        EmissionsSummary,
        MaterialityCheck,
        PathwayResult,
        TargetValidation,
        SubmissionField,
        TargetType,
        TargetTimeframe,
        PathwayAmbition,
        FLAGApplicability,
        ValidationResult as SBTiValidationResult,
    )
except ImportError:
    SBTiTargetWorkflow = None  # type: ignore[assignment,misc]
    SBTiTargetInput = None  # type: ignore[assignment,misc]
    SBTiTargetOutput = None  # type: ignore[assignment,misc]
    EmissionsSummary = None  # type: ignore[assignment,misc]
    MaterialityCheck = None  # type: ignore[assignment,misc]
    PathwayResult = None  # type: ignore[assignment,misc]
    TargetValidation = None  # type: ignore[assignment,misc]
    SubmissionField = None  # type: ignore[assignment,misc]
    TargetType = None  # type: ignore[assignment,misc]
    TargetTimeframe = None  # type: ignore[assignment,misc]
    PathwayAmbition = None  # type: ignore[assignment,misc]
    FLAGApplicability = None  # type: ignore[assignment,misc]
    SBTiValidationResult = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Climate Risk Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_043_scope_3_complete.workflows.climate_risk_workflow import (
        ClimateRiskWorkflow,
        ClimateRiskInput,
        ClimateRiskOutput,
        IdentifiedRisk,
        ExposureResult,
        FinancialImpact,
        ScenarioOutput,
        RiskCategory,
        TransitionRiskType,
        PhysicalRiskType,
        ClimateScenario,
        RiskSeverity,
        TimeHorizon,
    )
except ImportError:
    ClimateRiskWorkflow = None  # type: ignore[assignment,misc]
    ClimateRiskInput = None  # type: ignore[assignment,misc]
    ClimateRiskOutput = None  # type: ignore[assignment,misc]
    IdentifiedRisk = None  # type: ignore[assignment,misc]
    ExposureResult = None  # type: ignore[assignment,misc]
    FinancialImpact = None  # type: ignore[assignment,misc]
    ScenarioOutput = None  # type: ignore[assignment,misc]
    RiskCategory = None  # type: ignore[assignment,misc]
    TransitionRiskType = None  # type: ignore[assignment,misc]
    PhysicalRiskType = None  # type: ignore[assignment,misc]
    ClimateScenario = None  # type: ignore[assignment,misc]
    RiskSeverity = None  # type: ignore[assignment,misc]
    TimeHorizon = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Supplier Programme Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_043_scope_3_complete.workflows.supplier_programme_workflow import (
        SupplierProgrammeWorkflow,
        SupplierProgrammeInput,
        SupplierProgrammeOutput,
        SupplierProfile,
        SupplierTarget,
        CommitmentRecord,
        ProgressRecord,
        ProgrammeImpact,
        SupplierTier,
        CommitmentType,
        CommitmentStatus,
        ProgressRating,
    )
except ImportError:
    SupplierProgrammeWorkflow = None  # type: ignore[assignment,misc]
    SupplierProgrammeInput = None  # type: ignore[assignment,misc]
    SupplierProgrammeOutput = None  # type: ignore[assignment,misc]
    SupplierProfile = None  # type: ignore[assignment,misc]
    SupplierTarget = None  # type: ignore[assignment,misc]
    CommitmentRecord = None  # type: ignore[assignment,misc]
    ProgressRecord = None  # type: ignore[assignment,misc]
    ProgrammeImpact = None  # type: ignore[assignment,misc]
    SupplierTier = None  # type: ignore[assignment,misc]
    CommitmentType = None  # type: ignore[assignment,misc]
    CommitmentStatus = None  # type: ignore[assignment,misc]
    ProgressRating = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Full Enterprise Pipeline Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_043_scope_3_complete.workflows.full_enterprise_pipeline_workflow import (
        FullEnterprisePipelineWorkflow,
        FullEnterprisePipelineInput,
        FullEnterprisePipelineOutput,
        InventoryBridgeResult,
        AssuranceEvidence,
        AssurancePackage,
        AssuranceLevel,
        AssuranceStandard,
    )
except ImportError:
    FullEnterprisePipelineWorkflow = None  # type: ignore[assignment,misc]
    FullEnterprisePipelineInput = None  # type: ignore[assignment,misc]
    FullEnterprisePipelineOutput = None  # type: ignore[assignment,misc]
    InventoryBridgeResult = None  # type: ignore[assignment,misc]
    AssuranceEvidence = None  # type: ignore[assignment,misc]
    AssurancePackage = None  # type: ignore[assignment,misc]
    AssuranceLevel = None  # type: ignore[assignment,misc]
    AssuranceStandard = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Type Aliases
# ---------------------------------------------------------------------------

MaturityInput = MaturityAssessmentInput
MaturityOutput = MaturityAssessmentOutput
LCAInput = LCAIntegrationInput
LCAOutput = LCAIntegrationOutput
MultiInput = MultiEntityInput
MultiOutput = MultiEntityOutput
ScenarioInput = ScenarioPlanningInput
ScenarioOutput_Alias = ScenarioPlanningOutput
SBTiInput = SBTiTargetInput
SBTiOutput = SBTiTargetOutput
RiskInput = ClimateRiskInput
RiskOutput = ClimateRiskOutput
SupplierInput = SupplierProgrammeInput
SupplierOutput = SupplierProgrammeOutput
PipelineInput = FullEnterprisePipelineInput
PipelineOutput = FullEnterprisePipelineOutput


# ---------------------------------------------------------------------------
# Workflow Catalog
# ---------------------------------------------------------------------------

WORKFLOW_CATALOG: Dict[str, Dict[str, object]] = {
    "maturity_assessment": {
        "name": "Maturity Assessment",
        "description": (
            "Assess current Scope 3 data maturity (tier, DQR, uncertainty), "
            "identify gaps, generate upgrade roadmap, and prioritize by ROI"
        ),
        "phases": 4,
        "duration": "2-4 hours",
        "class": MaturityAssessmentWorkflow,
    },
    "lca_integration": {
        "name": "LCA Integration",
        "description": (
            "Integrate product-level LCA data into Scope 3 inventory with "
            "BOM mapping, lifecycle factors, and sensitivity analysis"
        ),
        "phases": 4,
        "duration": "4-8 hours per product",
        "class": LCAIntegrationWorkflow,
    },
    "multi_entity": {
        "name": "Multi-Entity Consolidation",
        "description": (
            "Consolidate Scope 3 across entity hierarchy with proportional "
            "ownership and inter-company double-counting elimination"
        ),
        "phases": 4,
        "duration": "4-8 hours",
        "class": MultiEntityWorkflow,
    },
    "scenario_planning": {
        "name": "Scenario Planning",
        "description": (
            "Model Scope 3 reduction scenarios with MACC analysis, "
            "what-if modelling, and Paris-aligned portfolio optimization"
        ),
        "phases": 4,
        "duration": "4-8 hours",
        "class": ScenarioPlanningWorkflow,
    },
    "sbti_target": {
        "name": "SBTi Target Management",
        "description": (
            "Manage SBTi Scope 3 targets with materiality check, "
            "pathway calculation, validation, and submission package"
        ),
        "phases": 4,
        "duration": "2-4 hours",
        "class": SBTiTargetWorkflow,
    },
    "climate_risk": {
        "name": "Climate Risk Assessment",
        "description": (
            "Identify and quantify climate-related financial risks from "
            "Scope 3 data with multi-scenario TCFD analysis"
        ),
        "phases": 4,
        "duration": "4-8 hours",
        "class": ClimateRiskWorkflow,
    },
    "supplier_programme": {
        "name": "Supplier Programme",
        "description": (
            "Manage supplier decarbonization programme with target setting, "
            "commitment tracking, progress measurement, and impact assessment"
        ),
        "phases": 4,
        "duration": "2-4 hours per cycle",
        "class": SupplierProgrammeWorkflow,
    },
    "full_enterprise_pipeline": {
        "name": "Full Enterprise Pipeline",
        "description": (
            "End-to-end Scope 3 enterprise pipeline from maturity assessment "
            "through ISAE 3410 assurance package"
        ),
        "phases": 8,
        "duration": "2-8 weeks",
        "class": FullEnterprisePipelineWorkflow,
    },
}


# ---------------------------------------------------------------------------
# Workflow Registry
# ---------------------------------------------------------------------------


class WorkflowRegistry:
    """
    Registry for PACK-043 Scope 3 Complete Pack workflows.

    Provides lookup, listing, and filtering of available workflows.

    Example:
        >>> registry = WorkflowRegistry()
        >>> wf_class = registry.get_workflow("maturity_assessment")
        >>> wf = wf_class()
    """

    def __init__(self) -> None:
        """Initialize WorkflowRegistry with the workflow catalog."""
        self._catalog = WORKFLOW_CATALOG

    def get_workflow(self, name: str) -> Optional[type]:
        """
        Get a workflow class by name.

        Args:
            name: Workflow name (e.g. 'maturity_assessment').

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
            category: Keyword to search (e.g. 'maturity', 'risk').

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
        "maturity_assessment": MaturityAssessmentWorkflow,
        "lca_integration": LCAIntegrationWorkflow,
        "multi_entity": MultiEntityWorkflow,
        "scenario_planning": ScenarioPlanningWorkflow,
        "sbti_target": SBTiTargetWorkflow,
        "climate_risk": ClimateRiskWorkflow,
        "supplier_programme": SupplierProgrammeWorkflow,
        "full_enterprise_pipeline": FullEnterprisePipelineWorkflow,
    }


__all__ = [
    # --- Module Version ---
    "_MODULE_VERSION",
    # --- Maturity Assessment Workflow ---
    "MaturityAssessmentWorkflow",
    "MaturityAssessmentInput",
    "MaturityAssessmentOutput",
    "CategoryMaturityState",
    "MaturityGap",
    "UpgradeStep",
    "UpgradeROI",
    "MaturityLevel",
    "MaturityMethodologyTier",
    "GapSeverity",
    "UpgradeEffort",
    # --- LCA Integration Workflow ---
    "LCAIntegrationWorkflow",
    "LCAIntegrationInput",
    "LCAIntegrationOutput",
    "ProductRecord",
    "ProductBOM",
    "BOMComponent",
    "FactorAssignment",
    "StageFootprint",
    "SensitivityResult",
    "ProductFootprint",
    "LifecycleStage",
    "LCABoundary",
    "MaterialCategory",
    "SensitivityParameter",
    # --- Multi-Entity Workflow ---
    "MultiEntityWorkflow",
    "MultiEntityInput",
    "MultiEntityOutput",
    "EntityRecord",
    "EntityScope3Data",
    "EntityBoundary",
    "ConsolidatedCategory",
    "DoubleCountFlag",
    "IntercompanyTransaction",
    "EntityType",
    "ConsolidationApproach",
    "DoubleCountType",
    # --- Scenario Planning Workflow ---
    "ScenarioPlanningWorkflow",
    "ScenarioPlanningInput",
    "ScenarioPlanningOutput",
    "BaselineCategory",
    "Intervention",
    "MACCEntry",
    "ScenarioResult",
    "OptimalPortfolio",
    "InterventionType",
    "ScenarioType",
    "AlignmentStatus",
    # --- SBTi Target Workflow ---
    "SBTiTargetWorkflow",
    "SBTiTargetInput",
    "SBTiTargetOutput",
    "EmissionsSummary",
    "MaterialityCheck",
    "PathwayResult",
    "TargetValidation",
    "SubmissionField",
    "TargetType",
    "TargetTimeframe",
    "PathwayAmbition",
    "FLAGApplicability",
    "SBTiValidationResult",
    # --- Climate Risk Workflow ---
    "ClimateRiskWorkflow",
    "ClimateRiskInput",
    "ClimateRiskOutput",
    "IdentifiedRisk",
    "ExposureResult",
    "FinancialImpact",
    "ScenarioOutput",
    "RiskCategory",
    "TransitionRiskType",
    "PhysicalRiskType",
    "ClimateScenario",
    "RiskSeverity",
    "TimeHorizon",
    # --- Supplier Programme Workflow ---
    "SupplierProgrammeWorkflow",
    "SupplierProgrammeInput",
    "SupplierProgrammeOutput",
    "SupplierProfile",
    "SupplierTarget",
    "CommitmentRecord",
    "ProgressRecord",
    "ProgrammeImpact",
    "SupplierTier",
    "CommitmentType",
    "CommitmentStatus",
    "ProgressRating",
    # --- Full Enterprise Pipeline Workflow ---
    "FullEnterprisePipelineWorkflow",
    "FullEnterprisePipelineInput",
    "FullEnterprisePipelineOutput",
    "InventoryBridgeResult",
    "AssuranceEvidence",
    "AssurancePackage",
    "AssuranceLevel",
    "AssuranceStandard",
    # --- Type Aliases ---
    "MaturityInput",
    "MaturityOutput",
    "LCAInput",
    "LCAOutput",
    "MultiInput",
    "MultiOutput",
    "ScenarioInput",
    "ScenarioOutput_Alias",
    "SBTiInput",
    "SBTiOutput",
    "RiskInput",
    "RiskOutput",
    "SupplierInput",
    "SupplierOutput",
    "PipelineInput",
    "PipelineOutput",
    # --- Catalog & Registry ---
    "WORKFLOW_CATALOG",
    "WorkflowRegistry",
    "get_loaded_workflows",
]
