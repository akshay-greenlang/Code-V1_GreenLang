# -*- coding: utf-8 -*-
"""
PACK-030 Net Zero Reporting Pack - Workflow Layer
==================================================

8 net-zero-reporting-specific workflows for SBTi progress reporting,
CDP questionnaire generation, TCFD 4-pillar disclosure, GRI 305 emissions
disclosures, ISSB IFRS S2 climate disclosure, SEC 10-K climate disclosure,
CSRD ESRS E1 climate change disclosure, and multi-framework full reporting.
All workflows use DAG orchestration, async execution, error handling, progress
tracking, and SHA-256 provenance hashing.

Workflows:
    1. SBTiProgressWorkflow (8 phases)
       AggregateTargetData -> AggregateEmissions -> CalculateProgress
       -> GenerateVariance -> CompileReport -> ValidateSchema
       -> RenderOutputs -> PackageSubmission

    2. CDPQuestionnaireWorkflow (8 phases)
       AggregateEmissions -> PullTargetData -> PullGovernanceData
       -> PullRiskData -> PullOpportunityData -> GenerateNarratives
       -> ValidateCompleteness -> ExportExcelTemplate

    3. TCFDDisclosureWorkflow (8 phases)
       GovernancePillar -> StrategyPillar -> RiskManagementPillar
       -> MetricsTargetsPillar -> CompileExecutiveReport
       -> AddScenarioAnalysis -> RenderPDFWithCharts
       -> GenerateAssuranceEvidence

    4. GRI305Workflow (8 phases)
       GRI305_1 -> GRI305_2 -> GRI305_3 -> GRI305_4
       -> GRI305_5 -> GRI305_6 -> GRI305_7 -> GenerateContentIndex

    5. IFRSS2Workflow (7 phases)
       GovernanceDisclosure -> StrategyDisclosure -> RiskManagement
       -> MetricsTargets -> AddXBRLTagging -> ValidateIFRSS2
       -> RenderOutputs

    6. SECClimateWorkflow (8 phases)
       Item1_BusinessDescription -> Item1A_RiskFactors -> Item7_MDA
       -> RegSK_Emissions -> ApplyXBRLTagging -> ValidateSECSchema
       -> GenerateAttestationTemplate -> PackageFor10K

    7. CSRDESRSE1Workflow (12 phases)
       E1_1 -> E1_2 -> E1_3 -> E1_4 -> E1_5 -> E1_6
       -> E1_7 -> E1_8 -> E1_9 -> ApplyDigitalTaxonomy
       -> ValidateESRSE1 -> RenderDigitalReport

    8. MultiFrameworkWorkflow (7 phases)
       AggregateAllData -> GenerateSharedNarratives
       -> ExecuteFrameworkWorkflows -> ValidateCrossConsistency
       -> GenerateExecutiveDashboard -> CreateMasterEvidenceBundle
       -> PackageAllReports

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-030 Net Zero Reporting Pack
Status: Production Ready
"""

__version__ = "1.0.0"
__pack_id__ = "PACK-030"
__pack_name__ = "Net Zero Reporting Pack"

# ---------------------------------------------------------------------------
# 1. SBTi Progress Report Workflow
# ---------------------------------------------------------------------------
from .sbti_progress_workflow import (
    SBTiProgressWorkflow,
    SBTiProgressConfig,
    SBTiProgressInput,
    SBTiProgressResult,
    SBTiTargetData,
    EmissionsData,
    ProgressCalculation,
    VarianceExplanation,
    SBTiReportSection,
    SchemaValidationResult,
    RenderedOutput,
    SubmissionPackage,
    TargetType,
    TargetScope,
    ValidationSeverity,
    SubmissionStatus,
    VarianceDirection,
    OutputFormat,
    SBTI_AMBITION_THRESHOLDS,
    SBTI_DISCLOSURE_REQUIRED_FIELDS,
    SBTI_SECTOR_PATHWAYS,
    SBTI_PROGRESS_RAG_RULES,
    PhaseResult as SBTiPhaseResult,
    PhaseStatus as SBTiPhaseStatus,
    WorkflowStatus as SBTiWorkflowStatus,
    RAGStatus as SBTiRAGStatus,
)

# ---------------------------------------------------------------------------
# 2. CDP Questionnaire Workflow
# ---------------------------------------------------------------------------
from .cdp_questionnaire_workflow import (
    CDPQuestionnaireWorkflow,
    CDPQuestionnaireConfig,
    CDPQuestionnaireInput,
    CDPQuestionnaireResult,
    CDPQuestionResponse,
    CDPModuleCompletion,
    CDPEmissionsResponse,
    CDPTargetResponse,
    CDPGovernanceResponse,
    CDPRiskResponse,
    CDPOpportunityResponse,
    CDPNarrativeSet,
    CDPCompletenessScore,
    CDPExcelExport,
    CDPModule,
    CDPResponseType,
    CDPScoringBand,
    CompletionStatus,
    CDP_MODULES_SPEC,
    CDP_SCORING_CRITERIA,
    PhaseResult as CDPPhaseResult,
    PhaseStatus as CDPPhaseStatus,
    WorkflowStatus as CDPWorkflowStatus,
    RAGStatus as CDPRAGStatus,
)

# ---------------------------------------------------------------------------
# 3. TCFD Disclosure Workflow
# ---------------------------------------------------------------------------
from .tcfd_disclosure_workflow import (
    TCFDDisclosureWorkflow,
    TCFDDisclosureConfig,
    TCFDDisclosureInput,
    TCFDDisclosureResult,
    TCFDPillarContent,
    ScenarioAnalysis,
    TCFDExecutiveReport,
    TCFDRenderedPDF,
    TCFDAssuranceEvidence,
    TCFDPillar,
    ScenarioType,
    RiskType,
    TimeHorizon,
    ComplianceLevel,
    TCFD_RECOMMENDATIONS,
    TCFD_COMPLIANCE_SCORING,
    PhaseResult as TCFDPhaseResult,
    PhaseStatus as TCFDPhaseStatus,
    WorkflowStatus as TCFDWorkflowStatus,
    RAGStatus as TCFDRAGStatus,
)

# ---------------------------------------------------------------------------
# 4. GRI 305 Disclosure Workflow
# ---------------------------------------------------------------------------
from .gri_305_workflow import (
    GRI305Workflow,
    GRI305Config,
    GRI305Input,
    GRI305Result,
    GRIDisclosure,
    GRIContentIndex,
    GRIDisclosureType,
    ConsolidationApproach,
    GRI_305_REQUIREMENTS,
    PhaseResult as GRIPhaseResult,
    PhaseStatus as GRIPhaseStatus,
    WorkflowStatus as GRIWorkflowStatus,
    RAGStatus as GRIRAGStatus,
)

# ---------------------------------------------------------------------------
# 5. ISSB IFRS S2 Workflow
# ---------------------------------------------------------------------------
from .issb_ifrs_s2_workflow import (
    IFRSS2Workflow,
    IFRSS2Config,
    IFRSS2Input,
    IFRSS2Result,
    IFRSS2PillarContent,
    XBRLTagSet,
    IFRSS2ValidationResult,
    IFRSS2RenderedOutput,
    IFRSS2Pillar,
    IFRS_S2_REQUIREMENTS,
    PhaseResult as IFRSS2PhaseResult,
    PhaseStatus as IFRSS2PhaseStatus,
    WorkflowStatus as IFRSS2WorkflowStatus,
    RAGStatus as IFRSS2RAGStatus,
)

# ---------------------------------------------------------------------------
# 6. SEC Climate Disclosure Workflow
# ---------------------------------------------------------------------------
from .sec_climate_workflow import (
    SECClimateWorkflow,
    SECClimateConfig,
    SECClimateInput,
    SECClimateResult,
    SECSectionContent,
    SECXBRLOutput,
    SECAttestationTemplate,
    SEC10KPackage,
    SECSection,
    AttestationLevel,
    SEC_REG_SK_REQUIREMENTS,
    PhaseResult as SECPhaseResult,
    PhaseStatus as SECPhaseStatus,
    WorkflowStatus as SECWorkflowStatus,
    RAGStatus as SECRAGStatus,
)

# ---------------------------------------------------------------------------
# 7. CSRD ESRS E1 Workflow
# ---------------------------------------------------------------------------
from .csrd_esrs_e1_workflow import (
    CSRDESRSE1Workflow,
    CSRDE1Config,
    CSRDE1Input,
    CSRDE1Result,
    ESRSE1Section,
    CSRDTaxonomyOutput,
    ESRSE1ValidationResult,
    ESRSE1Disclosure,
    ESRS_E1_REQUIREMENTS,
    PhaseResult as CSRDE1PhaseResult,
    PhaseStatus as CSRDE1PhaseStatus,
    WorkflowStatus as CSRDE1WorkflowStatus,
    RAGStatus as CSRDE1RAGStatus,
)

# ---------------------------------------------------------------------------
# 8. Multi-Framework Full Report Workflow
# ---------------------------------------------------------------------------
from .multi_framework_workflow import (
    MultiFrameworkWorkflow,
    MultiFrameworkConfig,
    MultiFrameworkInput,
    MultiFrameworkResult,
    AggregatedData,
    SharedNarrativeSet,
    FrameworkReport,
    ConsistencyCheckResult,
    ConsistencyReport,
    ExecutiveDashboard,
    MasterEvidenceBundle,
    ReportPackage,
    ConsistencyLevel,
    DashboardViewType,
    CONSISTENCY_RULES,
    SUPPORTED_FRAMEWORKS,
    PhaseResult as MultiPhaseResult,
    PhaseStatus as MultiPhaseStatus,
    WorkflowStatus as MultiWorkflowStatus,
    RAGStatus as MultiRAGStatus,
)


# =============================================================================
# WORKFLOW REGISTRY
# =============================================================================


WORKFLOW_REGISTRY: dict = {
    "sbti_progress_report": {
        "class": SBTiProgressWorkflow,
        "config_class": SBTiProgressConfig,
        "input_class": SBTiProgressInput,
        "result_class": SBTiProgressResult,
        "phases": 8,
        "description": "Generate SBTi annual progress disclosure report.",
        "dag": "AggregateTargetData -> AggregateEmissions -> CalculateProgress -> GenerateVariance -> CompileReport -> ValidateSchema -> RenderOutputs -> PackageSubmission",
    },
    "cdp_questionnaire": {
        "class": CDPQuestionnaireWorkflow,
        "config_class": CDPQuestionnaireConfig,
        "input_class": CDPQuestionnaireInput,
        "result_class": CDPQuestionnaireResult,
        "phases": 8,
        "description": "Generate CDP Climate Change questionnaire responses (C0-C12).",
        "dag": "AggregateEmissions -> PullTargetData -> PullGovernanceData -> PullRiskData -> PullOpportunityData -> GenerateNarratives -> ValidateCompleteness -> ExportExcelTemplate",
    },
    "tcfd_disclosure": {
        "class": TCFDDisclosureWorkflow,
        "config_class": TCFDDisclosureConfig,
        "input_class": TCFDDisclosureInput,
        "result_class": TCFDDisclosureResult,
        "phases": 8,
        "description": "Generate TCFD 4-pillar disclosure report with scenario analysis.",
        "dag": "GovernancePillar -> StrategyPillar -> RiskManagementPillar -> MetricsTargetsPillar -> CompileExecutiveReport -> AddScenarioAnalysis -> RenderPDFWithCharts -> GenerateAssuranceEvidence",
    },
    "gri_305_disclosure": {
        "class": GRI305Workflow,
        "config_class": GRI305Config,
        "input_class": GRI305Input,
        "result_class": GRI305Result,
        "phases": 8,
        "description": "Generate GRI 305 emissions disclosures (305-1 through 305-7) with Content Index.",
        "dag": "GRI305_1 -> GRI305_2 -> GRI305_3 -> GRI305_4 -> GRI305_5 -> GRI305_6 -> GRI305_7 -> GenerateContentIndex",
    },
    "issb_ifrs_s2": {
        "class": IFRSS2Workflow,
        "config_class": IFRSS2Config,
        "input_class": IFRSS2Input,
        "result_class": IFRSS2Result,
        "phases": 7,
        "description": "Generate ISSB IFRS S2 climate disclosure with XBRL tagging.",
        "dag": "GovernanceDisclosure -> StrategyDisclosure -> RiskManagement -> MetricsTargets -> AddXBRLTagging -> ValidateIFRSS2 -> RenderOutputs",
    },
    "sec_climate_disclosure": {
        "class": SECClimateWorkflow,
        "config_class": SECClimateConfig,
        "input_class": SECClimateInput,
        "result_class": SECClimateResult,
        "phases": 8,
        "description": "Generate SEC 10-K climate disclosure sections with XBRL/iXBRL tagging.",
        "dag": "Item1_BusinessDescription -> Item1A_RiskFactors -> Item7_MDA -> RegSK_Emissions -> ApplyXBRLTagging -> ValidateSECSchema -> GenerateAttestationTemplate -> PackageFor10K",
    },
    "csrd_esrs_e1": {
        "class": CSRDESRSE1Workflow,
        "config_class": CSRDE1Config,
        "input_class": CSRDE1Input,
        "result_class": CSRDE1Result,
        "phases": 12,
        "description": "Generate CSRD ESRS E1 Climate Change disclosure with digital taxonomy.",
        "dag": "E1_1 -> E1_2 -> E1_3 -> E1_4 -> E1_5 -> E1_6 -> E1_7 -> E1_8 -> E1_9 -> ApplyDigitalTaxonomy -> ValidateESRSE1 -> RenderDigitalReport",
    },
    "multi_framework_report": {
        "class": MultiFrameworkWorkflow,
        "config_class": MultiFrameworkConfig,
        "input_class": MultiFrameworkInput,
        "result_class": MultiFrameworkResult,
        "phases": 7,
        "description": "Generate all 7 framework reports in parallel with cross-framework consistency validation.",
        "dag": "AggregateAllData -> GenerateSharedNarratives -> ExecuteFrameworkWorkflows -> ValidateCrossConsistency -> GenerateExecutiveDashboard -> CreateMasterEvidenceBundle -> PackageAllReports",
    },
}


# =============================================================================
# WORKFLOW ORCHESTRATION UTILITIES
# =============================================================================


def get_workflow(name: str):
    """Get a workflow class by name.

    Args:
        name: Workflow name (e.g., 'sbti_progress_report').

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
    # --- 1. SBTi Progress Report ---
    "SBTiProgressWorkflow",
    "SBTiProgressConfig",
    "SBTiProgressInput",
    "SBTiProgressResult",
    "SBTiTargetData",
    "EmissionsData",
    "ProgressCalculation",
    "VarianceExplanation",
    "SBTiReportSection",
    "SchemaValidationResult",
    "RenderedOutput",
    "SubmissionPackage",
    "TargetType",
    "TargetScope",
    "ValidationSeverity",
    "SubmissionStatus",
    "VarianceDirection",
    "OutputFormat",
    "SBTI_AMBITION_THRESHOLDS",
    "SBTI_DISCLOSURE_REQUIRED_FIELDS",
    "SBTI_SECTOR_PATHWAYS",
    "SBTI_PROGRESS_RAG_RULES",
    # --- 2. CDP Questionnaire ---
    "CDPQuestionnaireWorkflow",
    "CDPQuestionnaireConfig",
    "CDPQuestionnaireInput",
    "CDPQuestionnaireResult",
    "CDPQuestionResponse",
    "CDPModuleCompletion",
    "CDPEmissionsResponse",
    "CDPTargetResponse",
    "CDPGovernanceResponse",
    "CDPRiskResponse",
    "CDPOpportunityResponse",
    "CDPNarrativeSet",
    "CDPCompletenessScore",
    "CDPExcelExport",
    "CDPModule",
    "CDPResponseType",
    "CDPScoringBand",
    "CompletionStatus",
    "CDP_MODULES_SPEC",
    "CDP_SCORING_CRITERIA",
    # --- 3. TCFD Disclosure ---
    "TCFDDisclosureWorkflow",
    "TCFDDisclosureConfig",
    "TCFDDisclosureInput",
    "TCFDDisclosureResult",
    "TCFDPillarContent",
    "ScenarioAnalysis",
    "TCFDExecutiveReport",
    "TCFDRenderedPDF",
    "TCFDAssuranceEvidence",
    "TCFDPillar",
    "ScenarioType",
    "RiskType",
    "TimeHorizon",
    "ComplianceLevel",
    "TCFD_RECOMMENDATIONS",
    "TCFD_COMPLIANCE_SCORING",
    # --- 4. GRI 305 Disclosure ---
    "GRI305Workflow",
    "GRI305Config",
    "GRI305Input",
    "GRI305Result",
    "GRIDisclosure",
    "GRIContentIndex",
    "GRIDisclosureType",
    "ConsolidationApproach",
    "GRI_305_REQUIREMENTS",
    # --- 5. ISSB IFRS S2 ---
    "IFRSS2Workflow",
    "IFRSS2Config",
    "IFRSS2Input",
    "IFRSS2Result",
    "IFRSS2PillarContent",
    "XBRLTagSet",
    "IFRSS2ValidationResult",
    "IFRSS2RenderedOutput",
    "IFRSS2Pillar",
    "IFRS_S2_REQUIREMENTS",
    # --- 6. SEC Climate Disclosure ---
    "SECClimateWorkflow",
    "SECClimateConfig",
    "SECClimateInput",
    "SECClimateResult",
    "SECSectionContent",
    "SECXBRLOutput",
    "SECAttestationTemplate",
    "SEC10KPackage",
    "SECSection",
    "AttestationLevel",
    "SEC_REG_SK_REQUIREMENTS",
    # --- 7. CSRD ESRS E1 ---
    "CSRDESRSE1Workflow",
    "CSRDE1Config",
    "CSRDE1Input",
    "CSRDE1Result",
    "ESRSE1Section",
    "CSRDTaxonomyOutput",
    "ESRSE1ValidationResult",
    "ESRSE1Disclosure",
    "ESRS_E1_REQUIREMENTS",
    # --- 8. Multi-Framework ---
    "MultiFrameworkWorkflow",
    "MultiFrameworkConfig",
    "MultiFrameworkInput",
    "MultiFrameworkResult",
    "AggregatedData",
    "SharedNarrativeSet",
    "FrameworkReport",
    "ConsistencyCheckResult",
    "ConsistencyReport",
    "ExecutiveDashboard",
    "MasterEvidenceBundle",
    "ReportPackage",
    "ConsistencyLevel",
    "DashboardViewType",
    "CONSISTENCY_RULES",
    "SUPPORTED_FRAMEWORKS",
    # --- Workflow Registry & Utilities ---
    "WORKFLOW_REGISTRY",
    "get_workflow",
    "get_workflow_config",
    "get_workflow_input",
    "list_workflows",
    "run_workflow",
]
