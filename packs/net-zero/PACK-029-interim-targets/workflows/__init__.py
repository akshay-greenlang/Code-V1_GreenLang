# -*- coding: utf-8 -*-
"""
PACK-029 Interim Targets Pack - Workflow Layer
==================================================

7 interim-target-specific workflows for SBTi-aligned interim target setting,
annual progress review, quarterly monitoring, variance investigation, corrective
action planning, annual regulatory reporting, and target recalibration.  All
workflows use DAG orchestration, async execution, error handling, progress
tracking, and SHA-256 provenance hashing.

Workflows:
    1. InterimTargetSettingWorkflow (6 phases)
       LoadBaseline -> CalcInterimTargets -> ValidateTargets
       -> GeneratePathway -> AllocateBudget -> SummaryReport

    2. AnnualProgressReviewWorkflow (5 phases)
       CollectActuals -> CompareTarget -> VarianceAnalysis
       -> TrendExtrapolation -> AnnualReport

    3. QuarterlyMonitoringWorkflow (4 phases)
       CollectQuarterly -> CompareMilestone -> GenerateAlerts
       -> TriggerCorrective

    4. VarianceInvestigationWorkflow (5 phases)
       DecomposeVariance -> AttributeRootCauses -> ClassifyFactors
       -> QuantifyInitiatives -> VarianceReport

    5. CorrectiveActionPlanningWorkflow (6 phases)
       QuantifyGap -> IdentifyInitiatives -> OptimizePortfolio
       -> ScheduleDeployment -> UpdateBudget -> GenerateReport

    6. AnnualReportingWorkflow (4 phases)
       SBTiDisclosure -> CDPResponse -> TCFDDisclosure
       -> AssurancePackage

    7. TargetRecalibrationWorkflow (5 phases)
       LoadUpdatedBaseline -> RecalcInterimTargets -> ValidateNewTargets
       -> UpdatePathway -> RecalibrationReport

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-029 Interim Targets Pack
Status: Production Ready
"""

__version__ = "1.0.0"
__pack_id__ = "PACK-029"
__pack_name__ = "Interim Targets Pack"

# ---------------------------------------------------------------------------
# 1. Interim Target Setting Workflow
# ---------------------------------------------------------------------------
from .interim_target_setting_workflow import (
    InterimTargetSettingWorkflow,
    InterimTargetSettingConfig,
    InterimTargetSettingInput,
    InterimTargetSettingResult,
    BaselineData,
    InterimTarget,
    InterimTargetSet,
    ValidationFinding,
    ValidationSummary,
    AnnualPathwayPoint,
    AnnualPathway,
    BUBudgetAllocation,
    CarbonBudgetAllocation,
    InterimTargetReport,
    TargetType,
    TargetScope,
    TargetTimeframe,
    BudgetMethod,
    SBTiAmbition,
    SBTI_NEAR_TERM_THRESHOLDS,
    SBTI_LONG_TERM_THRESHOLDS,
    SBTI_CROSS_SECTOR_RATES,
    SBTI_INTERIM_MILESTONES,
    SECTOR_CARBON_BUDGETS,
    PhaseResult as SettingPhaseResult,
    PhaseStatus as SettingPhaseStatus,
    WorkflowStatus as SettingWorkflowStatus,
    ValidationResult as SettingValidationResult,
    RAGStatus as SettingRAGStatus,
)

# ---------------------------------------------------------------------------
# 2. Annual Progress Review Workflow
# ---------------------------------------------------------------------------
from .annual_progress_review_workflow import (
    AnnualProgressReviewWorkflow,
    AnnualProgressReviewConfig,
    AnnualProgressReviewInput,
    AnnualProgressReviewResult,
    EmissionsActual,
    TargetComparison,
    ProgressSummary,
    VarianceComponent,
    VarianceDecomposition,
    TrendProjection,
    TrendAnalysis,
    AnnualProgressReport,
    VarianceType,
    ProjectionMethod,
    TrendDirection,
    MRV_SCOPE_COVERAGE,
    PROGRESS_KPIS,
    ProgressAlert as ReviewProgressAlert,
    ProgressKPI as ReviewProgressKPI,
    PhaseResult as ReviewPhaseResult,
    PhaseStatus as ReviewPhaseStatus,
    WorkflowStatus as ReviewWorkflowStatus,
    RAGStatus as ReviewRAGStatus,
    AlertSeverity as ReviewAlertSeverity,
)

# ---------------------------------------------------------------------------
# 3. Quarterly Monitoring Workflow
# ---------------------------------------------------------------------------
from .quarterly_monitoring_workflow import (
    QuarterlyMonitoringWorkflow,
    QuarterlyMonitoringConfig,
    QuarterlyMonitoringInput,
    QuarterlyMonitoringResult,
    QuarterlyEmissions,
    QuarterlyMilestone,
    QuarterlyComparison,
    QuarterlyAlert,
    CorrectiveActionTrigger,
    QuarterlyMonitoringReport,
    Quarter,
    AlertCategory,
    CorrectiveActionPriority,
    QUARTERLY_ALERT_RULES,
    SEASONALITY_FACTORS,
    PhaseResult as MonitoringPhaseResult,
    PhaseStatus as MonitoringPhaseStatus,
    WorkflowStatus as MonitoringWorkflowStatus,
    RAGStatus as MonitoringRAGStatus,
    AlertSeverity as MonitoringAlertSeverity,
)

# ---------------------------------------------------------------------------
# 4. Variance Investigation Workflow
# ---------------------------------------------------------------------------
from .variance_investigation_workflow import (
    VarianceInvestigationWorkflow,
    VarianceInvestigationConfig,
    VarianceInvestigationInput,
    VarianceInvestigationResult,
    LMDIComponent,
    KayaDecomposition,
    RootCause,
    FactorClassificationResult,
    InitiativeEffectiveness,
    InitiativePortfolio,
    VarianceInvestigationReport,
    DecompositionMethod,
    VarianceDriverType,
    FactorClassification,
    ControlLevel,
    InitiativeStatus,
    KAYA_COMPONENTS,
    ROOT_CAUSE_CATEGORIES,
    PhaseResult as VariancePhaseResult,
    PhaseStatus as VariancePhaseStatus,
    WorkflowStatus as VarianceWorkflowStatus,
    RAGStatus as VarianceRAGStatus,
)

# ---------------------------------------------------------------------------
# 5. Corrective Action Planning Workflow
# ---------------------------------------------------------------------------
from .corrective_action_planning_workflow import (
    CorrectiveActionPlanningWorkflow,
    CorrectiveActionConfig,
    CorrectiveActionInput,
    CorrectiveActionResult,
    GapQuantification,
    CandidateInitiative,
    OptimizedPortfolio,
    DeploymentSchedule,
    BudgetUpdate,
    CorrectiveActionReport,
    GapSeverity,
    InitiativePriority,
    InitiativeCategory,
    RiskLevel,
    DeploymentPhase,
    MACC_INITIATIVE_LIBRARY,
    PhaseResult as CorrectivePhaseResult,
    PhaseStatus as CorrectivePhaseStatus,
    WorkflowStatus as CorrectiveWorkflowStatus,
    RAGStatus as CorrectiveRAGStatus,
)

# ---------------------------------------------------------------------------
# 6. Annual Reporting Workflow
# ---------------------------------------------------------------------------
from .annual_reporting_workflow import (
    AnnualReportingWorkflow,
    AnnualReportingConfig,
    AnnualReportingInput,
    AnnualReportingResult,
    SBTiDisclosureReport,
    CDPResponse,
    TCFDDisclosure,
    AssurancePackage,
    DisclosureStatus,
    AssuranceLevel,
    CDP_TARGET_QUESTIONS,
    TCFD_METRICS_REQUIREMENTS,
    SBTI_DISCLOSURE_FIELDS,
    PhaseResult as ReportingPhaseResult,
    PhaseStatus as ReportingPhaseStatus,
    WorkflowStatus as ReportingWorkflowStatus,
    RAGStatus as ReportingRAGStatus,
)

# ---------------------------------------------------------------------------
# 7. Target Recalibration Workflow
# ---------------------------------------------------------------------------
from .target_recalibration_workflow import (
    TargetRecalibrationWorkflow,
    TargetRecalibrationConfig,
    TargetRecalibrationInput,
    TargetRecalibrationResult,
    BaselineChange,
    UpdatedBaseline,
    RecalibratedTarget,
    RecalibrationValidation,
    UpdatedPathway,
    RecalibrationReport,
    RecalibrationTrigger,
    RecalibrationScope,
    RECALCULATION_RULES,
    MA_IMPACT_FACTORS,
    PhaseResult as RecalibrationPhaseResult,
    PhaseStatus as RecalibrationPhaseStatus,
    WorkflowStatus as RecalibrationWorkflowStatus,
    ValidationResult as RecalibrationValidationResult,
    RAGStatus as RecalibrationRAGStatus,
)


# =============================================================================
# WORKFLOW REGISTRY
# =============================================================================


WORKFLOW_REGISTRY: dict = {
    "interim_target_setting": {
        "class": InterimTargetSettingWorkflow,
        "config_class": InterimTargetSettingConfig,
        "input_class": InterimTargetSettingInput,
        "result_class": InterimTargetSettingResult,
        "phases": 6,
        "description": "Set SBTi-aligned interim targets with 5-year and 10-year milestones.",
        "dag": "LoadBaseline -> CalcInterimTargets -> ValidateTargets -> GeneratePathway -> AllocateBudget -> SummaryReport",
    },
    "annual_progress_review": {
        "class": AnnualProgressReviewWorkflow,
        "config_class": AnnualProgressReviewConfig,
        "input_class": AnnualProgressReviewInput,
        "result_class": AnnualProgressReviewResult,
        "phases": 5,
        "description": "Annual review of progress against interim targets.",
        "dag": "CollectActuals -> CompareTarget -> VarianceAnalysis -> TrendExtrapolation -> AnnualReport",
    },
    "quarterly_monitoring": {
        "class": QuarterlyMonitoringWorkflow,
        "config_class": QuarterlyMonitoringConfig,
        "input_class": QuarterlyMonitoringInput,
        "result_class": QuarterlyMonitoringResult,
        "phases": 4,
        "description": "Quarterly emissions monitoring with RAG alerting.",
        "dag": "CollectQuarterly -> CompareMilestone -> GenerateAlerts -> TriggerCorrective",
    },
    "variance_investigation": {
        "class": VarianceInvestigationWorkflow,
        "config_class": VarianceInvestigationConfig,
        "input_class": VarianceInvestigationInput,
        "result_class": VarianceInvestigationResult,
        "phases": 5,
        "description": "Deep-dive variance investigation using LMDI and Kaya decomposition.",
        "dag": "DecomposeVariance -> AttributeRootCauses -> ClassifyFactors -> QuantifyInitiatives -> VarianceReport",
    },
    "corrective_action_planning": {
        "class": CorrectiveActionPlanningWorkflow,
        "config_class": CorrectiveActionConfig,
        "input_class": CorrectiveActionInput,
        "result_class": CorrectiveActionResult,
        "phases": 6,
        "description": "Plan corrective actions to close emission gaps with MACC optimization.",
        "dag": "QuantifyGap -> IdentifyInitiatives -> OptimizePortfolio -> ScheduleDeployment -> UpdateBudget -> GenerateReport",
    },
    "annual_reporting": {
        "class": AnnualReportingWorkflow,
        "config_class": AnnualReportingConfig,
        "input_class": AnnualReportingInput,
        "result_class": AnnualReportingResult,
        "phases": 4,
        "description": "Generate SBTi, CDP, TCFD disclosure reports and assurance packages.",
        "dag": "SBTiDisclosure -> CDPResponse -> TCFDDisclosure -> AssurancePackage",
    },
    "target_recalibration": {
        "class": TargetRecalibrationWorkflow,
        "config_class": TargetRecalibrationConfig,
        "input_class": TargetRecalibrationInput,
        "result_class": TargetRecalibrationResult,
        "phases": 5,
        "description": "Recalibrate interim targets after structural changes (M&A, boundary).",
        "dag": "LoadUpdatedBaseline -> RecalcInterimTargets -> ValidateNewTargets -> UpdatePathway -> RecalibrationReport",
    },
}


# =============================================================================
# WORKFLOW ORCHESTRATION UTILITIES
# =============================================================================


def get_workflow(name: str):
    """Get a workflow class by name.

    Args:
        name: Workflow name (e.g., 'interim_target_setting').

    Returns:
        Workflow class instance.

    Raises:
        KeyError: If workflow name not found.
    """
    entry = WORKFLOW_REGISTRY.get(name)
    if not entry:
        available = ", ".join(WORKFLOW_REGISTRY.keys())
        raise KeyError(f"Unknown workflow '{name}'. Available: {available}")
    return entry["class"]()


def get_workflow_config(name: str):
    """Get the config class for a workflow.

    Args:
        name: Workflow name.

    Returns:
        Config class (not instantiated).
    """
    entry = WORKFLOW_REGISTRY.get(name)
    if not entry:
        raise KeyError(f"Unknown workflow '{name}'.")
    return entry["config_class"]


def get_workflow_input(name: str):
    """Get the input class for a workflow.

    Args:
        name: Workflow name.

    Returns:
        Input class (not instantiated).
    """
    entry = WORKFLOW_REGISTRY.get(name)
    if not entry:
        raise KeyError(f"Unknown workflow '{name}'.")
    return entry["input_class"]


def list_workflows() -> list:
    """List all available workflows with descriptions.

    Returns:
        List of dicts with workflow name, phases, description, and DAG.
    """
    return [
        {
            "name": name,
            "phases": entry["phases"],
            "description": entry["description"],
            "dag": entry["dag"],
        }
        for name, entry in WORKFLOW_REGISTRY.items()
    ]


async def run_workflow(name: str, input_data=None, config=None):
    """Run a workflow by name.

    Args:
        name: Workflow name.
        input_data: Input data (Pydantic model or dict).
        config: Config (Pydantic model or dict).

    Returns:
        Workflow result.
    """
    entry = WORKFLOW_REGISTRY.get(name)
    if not entry:
        available = ", ".join(WORKFLOW_REGISTRY.keys())
        raise KeyError(f"Unknown workflow '{name}'. Available: {available}")

    wf = entry["class"]()

    if input_data is None:
        input_class = entry["input_class"]
        if config is not None:
            config_class = entry["config_class"]
            if isinstance(config, dict):
                config = config_class(**config)
            input_data = input_class(config=config)
        else:
            input_data = input_class()

    return await wf.execute(input_data)


__all__ = [
    "__version__",
    "__pack_id__",
    "__pack_name__",
    # --- 1. Interim Target Setting ---
    "InterimTargetSettingWorkflow",
    "InterimTargetSettingConfig",
    "InterimTargetSettingInput",
    "InterimTargetSettingResult",
    "BaselineData",
    "InterimTarget",
    "InterimTargetSet",
    "ValidationFinding",
    "ValidationSummary",
    "AnnualPathwayPoint",
    "AnnualPathway",
    "BUBudgetAllocation",
    "CarbonBudgetAllocation",
    "InterimTargetReport",
    "TargetType",
    "TargetScope",
    "TargetTimeframe",
    "BudgetMethod",
    "SBTiAmbition",
    "SBTI_NEAR_TERM_THRESHOLDS",
    "SBTI_LONG_TERM_THRESHOLDS",
    "SBTI_CROSS_SECTOR_RATES",
    "SBTI_INTERIM_MILESTONES",
    "SECTOR_CARBON_BUDGETS",
    # --- 2. Annual Progress Review ---
    "AnnualProgressReviewWorkflow",
    "AnnualProgressReviewConfig",
    "AnnualProgressReviewInput",
    "AnnualProgressReviewResult",
    "EmissionsActual",
    "TargetComparison",
    "ProgressSummary",
    "VarianceComponent",
    "VarianceDecomposition",
    "TrendProjection",
    "TrendAnalysis",
    "AnnualProgressReport",
    "VarianceType",
    "ProjectionMethod",
    "TrendDirection",
    "MRV_SCOPE_COVERAGE",
    "PROGRESS_KPIS",
    # --- 3. Quarterly Monitoring ---
    "QuarterlyMonitoringWorkflow",
    "QuarterlyMonitoringConfig",
    "QuarterlyMonitoringInput",
    "QuarterlyMonitoringResult",
    "QuarterlyEmissions",
    "QuarterlyMilestone",
    "QuarterlyComparison",
    "QuarterlyAlert",
    "CorrectiveActionTrigger",
    "QuarterlyMonitoringReport",
    "Quarter",
    "AlertCategory",
    "CorrectiveActionPriority",
    "QUARTERLY_ALERT_RULES",
    "SEASONALITY_FACTORS",
    # --- 4. Variance Investigation ---
    "VarianceInvestigationWorkflow",
    "VarianceInvestigationConfig",
    "VarianceInvestigationInput",
    "VarianceInvestigationResult",
    "LMDIComponent",
    "KayaDecomposition",
    "RootCause",
    "FactorClassificationResult",
    "InitiativeEffectiveness",
    "InitiativePortfolio",
    "VarianceInvestigationReport",
    "DecompositionMethod",
    "VarianceDriverType",
    "FactorClassification",
    "ControlLevel",
    "InitiativeStatus",
    "KAYA_COMPONENTS",
    "ROOT_CAUSE_CATEGORIES",
    # --- 5. Corrective Action Planning ---
    "CorrectiveActionPlanningWorkflow",
    "CorrectiveActionConfig",
    "CorrectiveActionInput",
    "CorrectiveActionResult",
    "GapQuantification",
    "CandidateInitiative",
    "OptimizedPortfolio",
    "DeploymentSchedule",
    "BudgetUpdate",
    "CorrectiveActionReport",
    "GapSeverity",
    "InitiativePriority",
    "InitiativeCategory",
    "RiskLevel",
    "DeploymentPhase",
    "MACC_INITIATIVE_LIBRARY",
    # --- 6. Annual Reporting ---
    "AnnualReportingWorkflow",
    "AnnualReportingConfig",
    "AnnualReportingInput",
    "AnnualReportingResult",
    "SBTiDisclosureReport",
    "CDPResponse",
    "TCFDDisclosure",
    "AssurancePackage",
    "DisclosureStatus",
    "AssuranceLevel",
    "CDP_TARGET_QUESTIONS",
    "TCFD_METRICS_REQUIREMENTS",
    "SBTI_DISCLOSURE_FIELDS",
    # --- 7. Target Recalibration ---
    "TargetRecalibrationWorkflow",
    "TargetRecalibrationConfig",
    "TargetRecalibrationInput",
    "TargetRecalibrationResult",
    "BaselineChange",
    "UpdatedBaseline",
    "RecalibratedTarget",
    "RecalibrationValidation",
    "UpdatedPathway",
    "RecalibrationReport",
    "RecalibrationTrigger",
    "RecalibrationScope",
    "RECALCULATION_RULES",
    "MA_IMPACT_FACTORS",
    # --- Workflow Registry & Utilities ---
    "WORKFLOW_REGISTRY",
    "get_workflow",
    "get_workflow_config",
    "get_workflow_input",
    "list_workflows",
    "run_workflow",
]
