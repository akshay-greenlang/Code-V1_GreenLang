# -*- coding: utf-8 -*-
"""
PACK-046 Intensity Metrics Pack - Workflow Orchestration
============================================================

Complete intensity metrics workflow orchestrators for the GHG Protocol
Corporate Standard, ESRS E1-6, SBTi SDA v2.0, CDP C6.10, SEC Climate
Disclosure, ISO 14064-1:2018, TCFD Metrics & Targets, GRI 305-4, and
IFRS S2 requirements. Each workflow coordinates intensity metrics processes
covering denominator setup, intensity calculation, LMDI decomposition
analysis, sector benchmarking, SBTi SDA target setting, scenario analysis,
multi-framework disclosure preparation, and full end-to-end pipeline
orchestration.

Workflows:
    - DenominatorSetupWorkflow: 4-phase workflow for denominator management
      with sector identification, denominator selection, data collection,
      and validation across all applicable reporting frameworks.

    - IntensityCalculationWorkflow: 4-phase workflow for multi-scope
      intensity computation with data ingestion, scope configuration,
      intensity calculation across all scope/denominator/period combinations,
      and quality assurance with outlier detection.

    - DecompositionAnalysisWorkflow: 3-phase workflow for LMDI decomposition
      with period selection, additive/multiplicative decomposition execution,
      and effect interpretation classifying organic vs structural changes.

    - BenchmarkingWorkflow: 4-phase workflow for sector benchmarking with
      peer group definition, data normalisation, benchmark comparison with
      percentile ranking, and ranking report generation.

    - TargetSettingWorkflow: 4-phase workflow for SBTi SDA target setting
      with baseline calculation, pathway selection (1.5C/WB2C), target
      calculation with annual reduction rates, and validation reporting.

    - ScenarioAnalysisWorkflow: 3-phase workflow for scenario modelling
      with scenario definition, Monte Carlo simulation execution, and
      probability assessment of target achievement.

    - DisclosurePreparationWorkflow: 4-phase workflow for multi-framework
      disclosure with metric aggregation, framework mapping (ESRS E1-6,
      CDP, SEC, SBTi, ISO, TCFD, GRI, IFRS S2), completeness checking,
      and disclosure package generation.

    - FullIntensityPipelineWorkflow: 8-phase end-to-end orchestrator
      invoking all sub-workflows with conditional phase execution,
      phase-level caching, checkpoint support, and full provenance chain.

Author: GreenLang Team
Version: 46.0.0
"""

from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Denominator Setup Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_046_intensity_metrics.workflows.denominator_setup_workflow import (
        DenominatorSetupWorkflow,
        DenominatorSetupInput,
        DenominatorSetupResult,
        SectorProfile,
        DenominatorCandidate,
        DenominatorRecord,
        ValidationFinding,
        SetupPhase,
        SectorClassification,
        DenominatorType,
        DenominatorUnit,
        DataQualityGrade,
        ValidationSeverity,
    )
except ImportError:
    DenominatorSetupWorkflow = None  # type: ignore[assignment,misc]
    DenominatorSetupInput = None  # type: ignore[assignment,misc]
    DenominatorSetupResult = None  # type: ignore[assignment,misc]
    SectorProfile = None  # type: ignore[assignment,misc]
    DenominatorCandidate = None  # type: ignore[assignment,misc]
    DenominatorRecord = None  # type: ignore[assignment,misc]
    ValidationFinding = None  # type: ignore[assignment,misc]
    SetupPhase = None  # type: ignore[assignment,misc]
    SectorClassification = None  # type: ignore[assignment,misc]
    DenominatorType = None  # type: ignore[assignment,misc]
    DenominatorUnit = None  # type: ignore[assignment,misc]
    DataQualityGrade = None  # type: ignore[assignment,misc]
    ValidationSeverity = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Intensity Calculation Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_046_intensity_metrics.workflows.intensity_calculation_workflow import (
        IntensityCalculationWorkflow,
        IntensityCalcInput,
        IntensityCalcResult,
        EmissionsDataSet,
        DenominatorDataSet,
        ScopeRule,
        IntensityMetric,
        QualityCheckResult,
        CalcPhase,
        ScopeInclusion,
        IntensityUnit,
        QualityCheckType,
        QualityOutcome,
    )
except ImportError:
    IntensityCalculationWorkflow = None  # type: ignore[assignment,misc]
    IntensityCalcInput = None  # type: ignore[assignment,misc]
    IntensityCalcResult = None  # type: ignore[assignment,misc]
    EmissionsDataSet = None  # type: ignore[assignment,misc]
    DenominatorDataSet = None  # type: ignore[assignment,misc]
    ScopeRule = None  # type: ignore[assignment,misc]
    IntensityMetric = None  # type: ignore[assignment,misc]
    QualityCheckResult = None  # type: ignore[assignment,misc]
    CalcPhase = None  # type: ignore[assignment,misc]
    ScopeInclusion = None  # type: ignore[assignment,misc]
    IntensityUnit = None  # type: ignore[assignment,misc]
    QualityCheckType = None  # type: ignore[assignment,misc]
    QualityOutcome = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Decomposition Analysis Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_046_intensity_metrics.workflows.decomposition_analysis_workflow import (
        DecompositionAnalysisWorkflow,
        DecompositionWorkflowInput,
        DecompositionWorkflowResult,
        PeriodData,
        DecompositionEffect,
        EffectInterpretation,
        DecompPhase,
        DecompositionMethod,
        EffectType,
        ChangeClassification,
    )
except ImportError:
    DecompositionAnalysisWorkflow = None  # type: ignore[assignment,misc]
    DecompositionWorkflowInput = None  # type: ignore[assignment,misc]
    DecompositionWorkflowResult = None  # type: ignore[assignment,misc]
    PeriodData = None  # type: ignore[assignment,misc]
    DecompositionEffect = None  # type: ignore[assignment,misc]
    EffectInterpretation = None  # type: ignore[assignment,misc]
    DecompPhase = None  # type: ignore[assignment,misc]
    DecompositionMethod = None  # type: ignore[assignment,misc]
    EffectType = None  # type: ignore[assignment,misc]
    ChangeClassification = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Benchmarking Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_046_intensity_metrics.workflows.benchmarking_workflow import (
        BenchmarkingWorkflow,
        BenchmarkWorkflowInput,
        BenchmarkWorkflowResult,
        PeerEntity,
        NormalisedPeer,
        BenchmarkComparison,
        RankingEntry,
        BenchmarkPhase,
        PeerSelectionCriteria,
        NormalisationMethod,
        BenchmarkSource,
        PerformanceBand,
    )
except ImportError:
    BenchmarkingWorkflow = None  # type: ignore[assignment,misc]
    BenchmarkWorkflowInput = None  # type: ignore[assignment,misc]
    BenchmarkWorkflowResult = None  # type: ignore[assignment,misc]
    PeerEntity = None  # type: ignore[assignment,misc]
    NormalisedPeer = None  # type: ignore[assignment,misc]
    BenchmarkComparison = None  # type: ignore[assignment,misc]
    RankingEntry = None  # type: ignore[assignment,misc]
    BenchmarkPhase = None  # type: ignore[assignment,misc]
    PeerSelectionCriteria = None  # type: ignore[assignment,misc]
    NormalisationMethod = None  # type: ignore[assignment,misc]
    BenchmarkSource = None  # type: ignore[assignment,misc]
    PerformanceBand = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Target Setting Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_046_intensity_metrics.workflows.target_setting_workflow import (
        TargetSettingWorkflow,
        TargetSettingInput,
        TargetSettingResult,
        BaselineData,
        PathwayConfig,
        AnnualTarget,
        TargetValidation,
        TargetPhase,
        SBTiPathway,
        TemperatureAlignment,
        AmbitionLevel,
        ValidationOutcome,
    )
except ImportError:
    TargetSettingWorkflow = None  # type: ignore[assignment,misc]
    TargetSettingInput = None  # type: ignore[assignment,misc]
    TargetSettingResult = None  # type: ignore[assignment,misc]
    BaselineData = None  # type: ignore[assignment,misc]
    PathwayConfig = None  # type: ignore[assignment,misc]
    AnnualTarget = None  # type: ignore[assignment,misc]
    TargetValidation = None  # type: ignore[assignment,misc]
    TargetPhase = None  # type: ignore[assignment,misc]
    SBTiPathway = None  # type: ignore[assignment,misc]
    TemperatureAlignment = None  # type: ignore[assignment,misc]
    AmbitionLevel = None  # type: ignore[assignment,misc]
    ValidationOutcome = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Scenario Analysis Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_046_intensity_metrics.workflows.scenario_analysis_workflow import (
        ScenarioAnalysisWorkflow,
        ScenarioWorkflowInput,
        ScenarioWorkflowResult,
        ScenarioDefinition,
        SimulationResult,
        ProbabilityAssessment,
        ScenarioPhase,
        ScenarioType,
        SimulationMethod,
        ProbabilityBand,
    )
except ImportError:
    ScenarioAnalysisWorkflow = None  # type: ignore[assignment,misc]
    ScenarioWorkflowInput = None  # type: ignore[assignment,misc]
    ScenarioWorkflowResult = None  # type: ignore[assignment,misc]
    ScenarioDefinition = None  # type: ignore[assignment,misc]
    SimulationResult = None  # type: ignore[assignment,misc]
    ProbabilityAssessment = None  # type: ignore[assignment,misc]
    ScenarioPhase = None  # type: ignore[assignment,misc]
    ScenarioType = None  # type: ignore[assignment,misc]
    SimulationMethod = None  # type: ignore[assignment,misc]
    ProbabilityBand = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Disclosure Preparation Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_046_intensity_metrics.workflows.disclosure_preparation_workflow import (
        DisclosurePreparationWorkflow,
        DisclosureInput,
        DisclosureResult,
        AggregatedMetric,
        FrameworkField,
        CompletenessGap,
        DisclosurePackageItem,
        DisclosurePhase,
        DisclosureFramework,
        FieldStatus,
        GapSeverity,
        PackageFormat,
    )
except ImportError:
    DisclosurePreparationWorkflow = None  # type: ignore[assignment,misc]
    DisclosureInput = None  # type: ignore[assignment,misc]
    DisclosureResult = None  # type: ignore[assignment,misc]
    AggregatedMetric = None  # type: ignore[assignment,misc]
    FrameworkField = None  # type: ignore[assignment,misc]
    CompletenessGap = None  # type: ignore[assignment,misc]
    DisclosurePackageItem = None  # type: ignore[assignment,misc]
    DisclosurePhase = None  # type: ignore[assignment,misc]
    DisclosureFramework = None  # type: ignore[assignment,misc]
    FieldStatus = None  # type: ignore[assignment,misc]
    GapSeverity = None  # type: ignore[assignment,misc]
    PackageFormat = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Full Intensity Pipeline Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_046_intensity_metrics.workflows.full_intensity_pipeline_workflow import (
        FullIntensityPipelineWorkflow,
        PipelineInput,
        PipelineResult,
        PipelineConfig,
        PipelinePhaseStatus,
        PipelineCheckpoint,
        PipelineMilestone,
        PipelineReport,
        PipelinePhase,
        PipelineMilestoneType,
        ReportType,
    )
except ImportError:
    FullIntensityPipelineWorkflow = None  # type: ignore[assignment,misc]
    PipelineInput = None  # type: ignore[assignment,misc]
    PipelineResult = None  # type: ignore[assignment,misc]
    PipelineConfig = None  # type: ignore[assignment,misc]
    PipelinePhaseStatus = None  # type: ignore[assignment,misc]
    PipelineCheckpoint = None  # type: ignore[assignment,misc]
    PipelineMilestone = None  # type: ignore[assignment,misc]
    PipelineReport = None  # type: ignore[assignment,misc]
    PipelinePhase = None  # type: ignore[assignment,misc]
    PipelineMilestoneType = None  # type: ignore[assignment,misc]
    ReportType = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Type Aliases
# ---------------------------------------------------------------------------

# Denominator Setup types
DenomSetupIn = DenominatorSetupInput
DenomSetupOut = DenominatorSetupResult

# Intensity Calculation types
IntensityCalcIn = IntensityCalcInput
IntensityCalcOut = IntensityCalcResult

# Decomposition Analysis types
DecompIn = DecompositionWorkflowInput
DecompOut = DecompositionWorkflowResult

# Benchmarking types
BenchmarkIn = BenchmarkWorkflowInput
BenchmarkOut = BenchmarkWorkflowResult

# Target Setting types
TargetIn = TargetSettingInput
TargetOut = TargetSettingResult

# Scenario Analysis types
ScenarioIn = ScenarioWorkflowInput
ScenarioOut = ScenarioWorkflowResult

# Disclosure Preparation types
DisclosureIn = DisclosureInput
DisclosureOut = DisclosureResult

# Full Pipeline types
FullPipelineIn = PipelineInput
FullPipelineOut = PipelineResult


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
        "denominator_setup": DenominatorSetupWorkflow,
        "intensity_calculation": IntensityCalculationWorkflow,
        "decomposition_analysis": DecompositionAnalysisWorkflow,
        "benchmarking": BenchmarkingWorkflow,
        "target_setting": TargetSettingWorkflow,
        "scenario_analysis": ScenarioAnalysisWorkflow,
        "disclosure_preparation": DisclosurePreparationWorkflow,
        "full_intensity_pipeline": FullIntensityPipelineWorkflow,
    }


__all__ = [
    # --- Denominator Setup Workflow ---
    "DenominatorSetupWorkflow",
    "DenominatorSetupInput",
    "DenominatorSetupResult",
    "SectorProfile",
    "DenominatorCandidate",
    "DenominatorRecord",
    "ValidationFinding",
    "SetupPhase",
    "SectorClassification",
    "DenominatorType",
    "DenominatorUnit",
    "DataQualityGrade",
    "ValidationSeverity",
    # --- Intensity Calculation Workflow ---
    "IntensityCalculationWorkflow",
    "IntensityCalcInput",
    "IntensityCalcResult",
    "EmissionsDataSet",
    "DenominatorDataSet",
    "ScopeRule",
    "IntensityMetric",
    "QualityCheckResult",
    "CalcPhase",
    "ScopeInclusion",
    "IntensityUnit",
    "QualityCheckType",
    "QualityOutcome",
    # --- Decomposition Analysis Workflow ---
    "DecompositionAnalysisWorkflow",
    "DecompositionWorkflowInput",
    "DecompositionWorkflowResult",
    "PeriodData",
    "DecompositionEffect",
    "EffectInterpretation",
    "DecompPhase",
    "DecompositionMethod",
    "EffectType",
    "ChangeClassification",
    # --- Benchmarking Workflow ---
    "BenchmarkingWorkflow",
    "BenchmarkWorkflowInput",
    "BenchmarkWorkflowResult",
    "PeerEntity",
    "NormalisedPeer",
    "BenchmarkComparison",
    "RankingEntry",
    "BenchmarkPhase",
    "PeerSelectionCriteria",
    "NormalisationMethod",
    "BenchmarkSource",
    "PerformanceBand",
    # --- Target Setting Workflow ---
    "TargetSettingWorkflow",
    "TargetSettingInput",
    "TargetSettingResult",
    "BaselineData",
    "PathwayConfig",
    "AnnualTarget",
    "TargetValidation",
    "TargetPhase",
    "SBTiPathway",
    "TemperatureAlignment",
    "AmbitionLevel",
    "ValidationOutcome",
    # --- Scenario Analysis Workflow ---
    "ScenarioAnalysisWorkflow",
    "ScenarioWorkflowInput",
    "ScenarioWorkflowResult",
    "ScenarioDefinition",
    "SimulationResult",
    "ProbabilityAssessment",
    "ScenarioPhase",
    "ScenarioType",
    "SimulationMethod",
    "ProbabilityBand",
    # --- Disclosure Preparation Workflow ---
    "DisclosurePreparationWorkflow",
    "DisclosureInput",
    "DisclosureResult",
    "AggregatedMetric",
    "FrameworkField",
    "CompletenessGap",
    "DisclosurePackageItem",
    "DisclosurePhase",
    "DisclosureFramework",
    "FieldStatus",
    "GapSeverity",
    "PackageFormat",
    # --- Full Intensity Pipeline Workflow ---
    "FullIntensityPipelineWorkflow",
    "PipelineInput",
    "PipelineResult",
    "PipelineConfig",
    "PipelinePhaseStatus",
    "PipelineCheckpoint",
    "PipelineMilestone",
    "PipelineReport",
    "PipelinePhase",
    "PipelineMilestoneType",
    "ReportType",
    # --- Type Aliases ---
    "DenomSetupIn",
    "DenomSetupOut",
    "IntensityCalcIn",
    "IntensityCalcOut",
    "DecompIn",
    "DecompOut",
    "BenchmarkIn",
    "BenchmarkOut",
    "TargetIn",
    "TargetOut",
    "ScenarioIn",
    "ScenarioOut",
    "DisclosureIn",
    "DisclosureOut",
    "FullPipelineIn",
    "FullPipelineOut",
    # --- Helpers ---
    "get_loaded_workflows",
]
