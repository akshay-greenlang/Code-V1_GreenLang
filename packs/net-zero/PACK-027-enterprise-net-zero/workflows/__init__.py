# -*- coding: utf-8 -*-
"""
PACK-027 Enterprise Net Zero Pack - Workflow Layer
====================================================

10 enterprise-grade workflows for large organizations with 100+ entities,
50,000+ suppliers, and multi-framework regulatory compliance.  All workflows
use DAG orchestration, error handling, progress tracking, and SHA-256
provenance hashing.

Workflows:
    1. ComprehensiveBaselineWorkflow (6 phases, 6-12 weeks)
       EntityMapping -> DataCollection -> QualityAssurance
       -> Calculation -> Consolidation -> Reporting

    2. SBTiSubmissionWorkflow (5 phases)
       BaselineValidation -> PathwaySelection -> TargetDefinition
       -> CriteriaValidation -> SubmissionPackage

    3. AnnualInventoryWorkflow (5 phases)
       DataRefresh -> Calculation -> BaseYearCheck
       -> Consolidation -> AnnualReport

    4. ScenarioAnalysisWorkflow (5 phases)
       ParameterSetup -> Simulation -> Sensitivity
       -> Comparison -> StrategyReport

    5. SupplyChainEngagementWorkflow (5 phases)
       SupplierMapping -> Tiering -> ProgramDesign
       -> Execution -> ImpactMeasurement

    6. InternalCarbonPricingWorkflow (4 phases)
       PriceDesign -> AllocationSetup -> ImpactAnalysis -> Reporting

    7. MultiEntityRollupWorkflow (5 phases)
       EntityRefresh -> DataValidation -> EntityCalculation
       -> Elimination -> ConsolidatedReport

    8. ExternalAssuranceWorkflow (5 phases)
       ScopeDefinition -> EvidenceCollection -> WorkpaperGeneration
       -> ControlTesting -> AssurancePackage

    9. BoardReportingWorkflow (6 phases)
       DataRefresh -> PerformanceAnalysis -> InitiativeStatus
       -> RiskAssessment -> ComplianceUpdate -> ReportGeneration

    10. RegulatoryFilingWorkflow (8 phases)
        FrameworkSelection -> DatapointMapping -> DataValidation
        -> CrosswalkReconciliation -> DocumentGeneration
        -> QualityReview -> FilingSubmission -> ComplianceTracker

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-027 Enterprise Net Zero Pack
Status: Production Ready
"""

__version__ = "1.0.0"
__pack_id__ = "PACK-027"
__pack_name__ = "Enterprise Net Zero Pack"

# ---------------------------------------------------------------------------
# 1. Comprehensive Baseline Workflow
# ---------------------------------------------------------------------------
from .comprehensive_baseline_workflow import (
    ComprehensiveBaselineWorkflow,
    ComprehensiveBaselineConfig,
    ComprehensiveBaselineInput,
    ComprehensiveBaselineResult,
    EntityDefinition,
    EntityHierarchy,
    EntityDataPackage,
    DataQualityReport,
    EntityEmissions,
    IntercompanyElimination,
    ConsolidatedBaseline,
    ConsolidationApproach,
    EntityType,
    DataSourceType as BaselineDataSourceType,
    DataQualityLevel,
    Scope3Category,
    MaterialityLevel,
    PhaseResult as BaselinePhaseResult,
    PhaseStatus as BaselinePhaseStatus,
    WorkflowStatus as BaselineWorkflowStatus,
)

# ---------------------------------------------------------------------------
# 2. SBTi Submission Workflow
# ---------------------------------------------------------------------------
from .sbti_submission_workflow import (
    SBTiSubmissionWorkflow,
    SBTiSubmissionConfig,
    SBTiSubmissionInput,
    SBTiSubmissionResult,
    BaselineSnapshot,
    CriterionValidation,
    TargetDefinition,
    SubmissionDocument,
    SBTiPathway,
    CriterionStatus,
    TargetType,
    SDAIndustrySector,
    PhaseResult as SBTiPhaseResult,
    PhaseStatus as SBTiPhaseStatus,
    WorkflowStatus as SBTiWorkflowStatus,
)

# ---------------------------------------------------------------------------
# 3. Annual Inventory Workflow
# ---------------------------------------------------------------------------
from .annual_inventory_workflow import (
    AnnualInventoryWorkflow,
    AnnualInventoryConfig,
    AnnualInventoryInput,
    AnnualInventoryResult,
    DataRefreshStatus,
    BaseYearRecalculation,
    AnnualComparison,
    RecalculationTrigger,
    PhaseResult as AnnualPhaseResult,
    PhaseStatus as AnnualPhaseStatus,
    WorkflowStatus as AnnualWorkflowStatus,
)

# ---------------------------------------------------------------------------
# 4. Scenario Analysis Workflow
# ---------------------------------------------------------------------------
from .scenario_analysis_workflow import (
    ScenarioAnalysisWorkflow,
    ScenarioAnalysisConfig,
    ScenarioAnalysisInput,
    ScenarioAnalysisResult,
    ScenarioDefinition,
    ScenarioParameter,
    ScenarioResult,
    SensitivityDriver,
    SimulationRun,
    ScenarioType,
    PhaseResult as ScenarioPhaseResult,
    PhaseStatus as ScenarioPhaseStatus,
    WorkflowStatus as ScenarioWorkflowStatus,
)

# ---------------------------------------------------------------------------
# 5. Supply Chain Engagement Workflow
# ---------------------------------------------------------------------------
from .supply_chain_engagement_workflow import (
    SupplyChainEngagementWorkflow,
    SupplyChainEngagementConfig,
    SupplyChainEngagementInput,
    SupplyChainEngagementResult,
    SupplierRecord,
    SupplierScorecard,
    EngagementProgram,
    EngagementProgress,
    SupplierTier,
    EngagementLevel,
    CDPScore,
    PhaseResult as SupplyChainPhaseResult,
    PhaseStatus as SupplyChainPhaseStatus,
    WorkflowStatus as SupplyChainWorkflowStatus,
)

# ---------------------------------------------------------------------------
# 6. Internal Carbon Pricing Workflow
# ---------------------------------------------------------------------------
from .internal_carbon_pricing_workflow import (
    InternalCarbonPricingWorkflow,
    InternalCarbonPricingConfig,
    InternalCarbonPricingInput,
    InternalCarbonPricingResult,
    BusinessUnit,
    InvestmentProposal,
    CarbonAdjustedInvestment,
    BUCarbonAllocation,
    CBAMExposure,
    CarbonPricingApproach,
    PhaseResult as CarbonPricingPhaseResult,
    PhaseStatus as CarbonPricingPhaseStatus,
    WorkflowStatus as CarbonPricingWorkflowStatus,
)

# ---------------------------------------------------------------------------
# 7. Multi-Entity Rollup Workflow
# ---------------------------------------------------------------------------
from .multi_entity_rollup_workflow import (
    MultiEntityRollupWorkflow,
    MultiEntityRollupConfig,
    MultiEntityRollupInput,
    MultiEntityRollupResult,
    EntityChange,
    EntityValidation,
    EntityResult,
    EliminationEntry,
    Reconciliation,
    EntityChangeType,
    PhaseResult as RollupPhaseResult,
    PhaseStatus as RollupPhaseStatus,
    WorkflowStatus as RollupWorkflowStatus,
)

# ---------------------------------------------------------------------------
# 8. External Assurance Workflow
# ---------------------------------------------------------------------------
from .external_assurance_workflow import (
    ExternalAssuranceWorkflow,
    ExternalAssuranceConfig,
    ExternalAssuranceInput,
    ExternalAssuranceResult,
    AssuranceScope,
    EvidenceItem,
    Workpaper,
    ControlTest,
    AssuranceLevel,
    AssuranceStandard,
    EvidenceType,
    ControlTestResult,
    PhaseResult as AssurancePhaseResult,
    PhaseStatus as AssurancePhaseStatus,
    WorkflowStatus as AssuranceWorkflowStatus,
)

# ---------------------------------------------------------------------------
# 9. Board Reporting Workflow
# ---------------------------------------------------------------------------
from .board_reporting_workflow import (
    BoardReportingWorkflow,
    BoardReportingConfig,
    BoardReportingInput,
    BoardReportingResult,
    KPI,
    Initiative,
    ClimateRisk,
    RegulatoryStatus,
    RAGStatus,
    TrendDirection,
    PhaseResult as BoardPhaseResult,
    PhaseStatus as BoardPhaseStatus,
    WorkflowStatus as BoardWorkflowStatus,
)

# ---------------------------------------------------------------------------
# 10. Regulatory Filing Workflow
# ---------------------------------------------------------------------------
from .regulatory_filing_workflow import (
    RegulatoryFilingWorkflow,
    RegulatoryFilingConfig,
    RegulatoryFilingInput,
    RegulatoryFilingResult,
    FrameworkFiling,
    CrosswalkMapping,
    RegulatoryFramework,
    FilingStatus,
    PhaseResult as FilingPhaseResult,
    PhaseStatus as FilingPhaseStatus,
    WorkflowStatus as FilingWorkflowStatus,
)

__all__ = [
    "__version__",
    "__pack_id__",
    "__pack_name__",
    # --- 1. Comprehensive Baseline ---
    "ComprehensiveBaselineWorkflow",
    "ComprehensiveBaselineConfig",
    "ComprehensiveBaselineInput",
    "ComprehensiveBaselineResult",
    "EntityDefinition",
    "EntityHierarchy",
    "EntityDataPackage",
    "DataQualityReport",
    "EntityEmissions",
    "IntercompanyElimination",
    "ConsolidatedBaseline",
    "ConsolidationApproach",
    "EntityType",
    "BaselineDataSourceType",
    "DataQualityLevel",
    "Scope3Category",
    "MaterialityLevel",
    # --- 2. SBTi Submission ---
    "SBTiSubmissionWorkflow",
    "SBTiSubmissionConfig",
    "SBTiSubmissionInput",
    "SBTiSubmissionResult",
    "BaselineSnapshot",
    "CriterionValidation",
    "TargetDefinition",
    "SubmissionDocument",
    "SBTiPathway",
    "CriterionStatus",
    "TargetType",
    "SDAIndustrySector",
    # --- 3. Annual Inventory ---
    "AnnualInventoryWorkflow",
    "AnnualInventoryConfig",
    "AnnualInventoryInput",
    "AnnualInventoryResult",
    "DataRefreshStatus",
    "BaseYearRecalculation",
    "AnnualComparison",
    "RecalculationTrigger",
    # --- 4. Scenario Analysis ---
    "ScenarioAnalysisWorkflow",
    "ScenarioAnalysisConfig",
    "ScenarioAnalysisInput",
    "ScenarioAnalysisResult",
    "ScenarioDefinition",
    "ScenarioParameter",
    "ScenarioResult",
    "SensitivityDriver",
    "SimulationRun",
    "ScenarioType",
    # --- 5. Supply Chain Engagement ---
    "SupplyChainEngagementWorkflow",
    "SupplyChainEngagementConfig",
    "SupplyChainEngagementInput",
    "SupplyChainEngagementResult",
    "SupplierRecord",
    "SupplierScorecard",
    "EngagementProgram",
    "EngagementProgress",
    "SupplierTier",
    "EngagementLevel",
    "CDPScore",
    # --- 6. Internal Carbon Pricing ---
    "InternalCarbonPricingWorkflow",
    "InternalCarbonPricingConfig",
    "InternalCarbonPricingInput",
    "InternalCarbonPricingResult",
    "BusinessUnit",
    "InvestmentProposal",
    "CarbonAdjustedInvestment",
    "BUCarbonAllocation",
    "CBAMExposure",
    "CarbonPricingApproach",
    # --- 7. Multi-Entity Rollup ---
    "MultiEntityRollupWorkflow",
    "MultiEntityRollupConfig",
    "MultiEntityRollupInput",
    "MultiEntityRollupResult",
    "EntityChange",
    "EntityValidation",
    "EntityResult",
    "EliminationEntry",
    "Reconciliation",
    "EntityChangeType",
    # --- 8. External Assurance ---
    "ExternalAssuranceWorkflow",
    "ExternalAssuranceConfig",
    "ExternalAssuranceInput",
    "ExternalAssuranceResult",
    "AssuranceScope",
    "EvidenceItem",
    "Workpaper",
    "ControlTest",
    "AssuranceLevel",
    "AssuranceStandard",
    "EvidenceType",
    "ControlTestResult",
    # --- 9. Board Reporting ---
    "BoardReportingWorkflow",
    "BoardReportingConfig",
    "BoardReportingInput",
    "BoardReportingResult",
    "KPI",
    "Initiative",
    "ClimateRisk",
    "RegulatoryStatus",
    "RAGStatus",
    "TrendDirection",
    # --- 10. Regulatory Filing ---
    "RegulatoryFilingWorkflow",
    "RegulatoryFilingConfig",
    "RegulatoryFilingInput",
    "RegulatoryFilingResult",
    "FrameworkFiling",
    "CrosswalkMapping",
    "RegulatoryFramework",
    "FilingStatus",
]
