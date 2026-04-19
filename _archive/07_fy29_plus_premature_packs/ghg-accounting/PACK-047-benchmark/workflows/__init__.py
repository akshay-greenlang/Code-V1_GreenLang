# -*- coding: utf-8 -*-
"""
PACK-047 GHG Emissions Benchmark Pack - Workflow Orchestration
==================================================================

Complete emissions benchmarking workflow orchestrators for the GHG Protocol
Corporate Standard, ESRS E1-6, SBTi SDA v2.1, CDP C4.1/C4.2, SEC Climate
Disclosure, ISO 14064-1:2018, TCFD Metrics & Targets, PCAF v3, IFRS S2,
IEA NZE, IPCC AR6, and TPI requirements. Each workflow coordinates
benchmarking processes covering peer group construction, scope normalisation,
pathway alignment, implied temperature rise calculation, trajectory
benchmarking, portfolio-level carbon benchmarking, transition risk scoring,
multi-framework disclosure preparation, and full end-to-end pipeline
orchestration.

Workflows:
    - PeerGroupSetupWorkflow: 5-phase workflow for peer group construction
      with sector mapping, size banding, geographic weighting, peer scoring,
      and validation across GICS/NACE/ISIC classification systems.

    - BenchmarkAssessmentWorkflow: 5-phase workflow for emissions benchmark
      assessment with data ingestion, scope normalisation, peer comparison,
      percentile ranking, and report generation.

    - PathwayAlignmentWorkflow: 4-phase workflow for science-based pathway
      alignment with pathway loading, waypoint interpolation, gap analysis,
      and alignment scoring across IEA/IPCC/SBTi/OECM/TPI/CRREM pathways.

    - TrajectoryAnalysisWorkflow: 4-phase workflow for multi-year trajectory
      analysis with time series loading, CARR computation, convergence
      analysis (beta/sigma), and trajectory ranking.

    - PortfolioBenchmarkWorkflow: 5-phase workflow for financed emissions
      portfolio benchmarking with holdings loading, PCAF quality scoring,
      weighted aggregation, WACI calculation, and index comparison.

    - TransitionRiskWorkflow: 5-phase workflow for transition risk assessment
      with carbon budget allocation, stranding calculation, regulatory risk,
      competitive risk, and composite scoring.

    - DisclosurePreparationWorkflow: 4-phase workflow for multi-framework
      benchmark disclosure with metric aggregation, framework mapping
      (ESRS E1, CDP, SFDR, TCFD, SEC, GRI), QA checks, and package assembly.

    - FullBenchmarkPipelineWorkflow: 8-phase end-to-end orchestrator
      invoking all sub-workflows with conditional phase execution,
      phase-level caching, checkpoint support, and full provenance chain.

Author: GreenLang Team
Version: 47.0.0
"""

from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Peer Group Setup Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_047_benchmark.workflows.peer_group_setup_workflow import (
        PeerGroupSetupWorkflow,
        PeerGroupSetupInput,
        PeerGroupSetupResult,
        PeerSetupPhase,
        SectorSystem,
        GICSsector,
        SizeBand,
        GeographicRegion,
        DataQualityLevel,
        PeerStatus,
        ValidationSeverity,
        SectorMapping,
        PeerCandidate,
        ValidationFinding,
        PeerGroupStats,
    )
except ImportError:
    PeerGroupSetupWorkflow = None  # type: ignore[assignment,misc]
    PeerGroupSetupInput = None  # type: ignore[assignment,misc]
    PeerGroupSetupResult = None  # type: ignore[assignment,misc]
    PeerSetupPhase = None  # type: ignore[assignment,misc]
    SectorSystem = None  # type: ignore[assignment,misc]
    GICSsector = None  # type: ignore[assignment,misc]
    SizeBand = None  # type: ignore[assignment,misc]
    GeographicRegion = None  # type: ignore[assignment,misc]
    DataQualityLevel = None  # type: ignore[assignment,misc]
    PeerStatus = None  # type: ignore[assignment,misc]
    ValidationSeverity = None  # type: ignore[assignment,misc]
    SectorMapping = None  # type: ignore[assignment,misc]
    PeerCandidate = None  # type: ignore[assignment,misc]
    ValidationFinding = None  # type: ignore[assignment,misc]
    PeerGroupStats = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Benchmark Assessment Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_047_benchmark.workflows.benchmark_assessment_workflow import (
        BenchmarkAssessmentWorkflow,
        BenchmarkAssessmentInput,
        BenchmarkAssessmentResult,
        AssessmentPhase,
        GWPVintage,
        NormalisationMethod,
        PerformanceBand,
        RankingMetric,
        ReportSection,
        EntityEmissions,
        NormalisationAdjustment,
        PeerDistribution,
        GapAnalysisResult,
        PercentileRank,
        ReportItem,
    )
except ImportError:
    BenchmarkAssessmentWorkflow = None  # type: ignore[assignment,misc]
    BenchmarkAssessmentInput = None  # type: ignore[assignment,misc]
    BenchmarkAssessmentResult = None  # type: ignore[assignment,misc]
    AssessmentPhase = None  # type: ignore[assignment,misc]
    GWPVintage = None  # type: ignore[assignment,misc]
    NormalisationMethod = None  # type: ignore[assignment,misc]
    PerformanceBand = None  # type: ignore[assignment,misc]
    RankingMetric = None  # type: ignore[assignment,misc]
    ReportSection = None  # type: ignore[assignment,misc]
    EntityEmissions = None  # type: ignore[assignment,misc]
    NormalisationAdjustment = None  # type: ignore[assignment,misc]
    PeerDistribution = None  # type: ignore[assignment,misc]
    GapAnalysisResult = None  # type: ignore[assignment,misc]
    PercentileRank = None  # type: ignore[assignment,misc]
    ReportItem = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Pathway Alignment Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_047_benchmark.workflows.pathway_alignment_workflow import (
        PathwayAlignmentWorkflow,
        PathwayAlignmentInput,
        PathwayAlignmentResult,
        AlignmentPhase,
        PathwaySource,
        InterpolationMethod,
        TemperatureAlignment,
        AlignmentStatus,
        PathwayDefinition,
        AnnualWaypoint,
        PathwayGap,
        PathwayAlignmentScore,
    )
except ImportError:
    PathwayAlignmentWorkflow = None  # type: ignore[assignment,misc]
    PathwayAlignmentInput = None  # type: ignore[assignment,misc]
    PathwayAlignmentResult = None  # type: ignore[assignment,misc]
    AlignmentPhase = None  # type: ignore[assignment,misc]
    PathwaySource = None  # type: ignore[assignment,misc]
    InterpolationMethod = None  # type: ignore[assignment,misc]
    TemperatureAlignment = None  # type: ignore[assignment,misc]
    AlignmentStatus = None  # type: ignore[assignment,misc]
    PathwayDefinition = None  # type: ignore[assignment,misc]
    AnnualWaypoint = None  # type: ignore[assignment,misc]
    PathwayGap = None  # type: ignore[assignment,misc]
    PathwayAlignmentScore = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Trajectory Analysis Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_047_benchmark.workflows.trajectory_analysis_workflow import (
        TrajectoryAnalysisWorkflow,
        TrajectoryAnalysisInput,
        TrajectoryAnalysisResult,
        TrajectoryPhase,
        TrajectoryBand,
        ConvergenceType,
        ConvergenceStatus,
        EntityTimeSeries,
        CARRResult,
        ConvergenceResult,
        TrajectoryRank,
    )
except ImportError:
    TrajectoryAnalysisWorkflow = None  # type: ignore[assignment,misc]
    TrajectoryAnalysisInput = None  # type: ignore[assignment,misc]
    TrajectoryAnalysisResult = None  # type: ignore[assignment,misc]
    TrajectoryPhase = None  # type: ignore[assignment,misc]
    TrajectoryBand = None  # type: ignore[assignment,misc]
    ConvergenceType = None  # type: ignore[assignment,misc]
    ConvergenceStatus = None  # type: ignore[assignment,misc]
    EntityTimeSeries = None  # type: ignore[assignment,misc]
    CARRResult = None  # type: ignore[assignment,misc]
    ConvergenceResult = None  # type: ignore[assignment,misc]
    TrajectoryRank = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Portfolio Benchmark Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_047_benchmark.workflows.portfolio_benchmark_workflow import (
        PortfolioBenchmarkWorkflow,
        PortfolioBenchmarkInput,
        PortfolioBenchmarkResult,
        PortfolioPhase,
        AssetClass,
        PCAFLevel,
        BenchmarkIndex,
        PortfolioHolding,
        PCAFQualityResult,
        AssetClassAggregation,
        PortfolioMetric,
        IndexComparisonResult,
    )
except ImportError:
    PortfolioBenchmarkWorkflow = None  # type: ignore[assignment,misc]
    PortfolioBenchmarkInput = None  # type: ignore[assignment,misc]
    PortfolioBenchmarkResult = None  # type: ignore[assignment,misc]
    PortfolioPhase = None  # type: ignore[assignment,misc]
    AssetClass = None  # type: ignore[assignment,misc]
    PCAFLevel = None  # type: ignore[assignment,misc]
    BenchmarkIndex = None  # type: ignore[assignment,misc]
    PortfolioHolding = None  # type: ignore[assignment,misc]
    PCAFQualityResult = None  # type: ignore[assignment,misc]
    AssetClassAggregation = None  # type: ignore[assignment,misc]
    PortfolioMetric = None  # type: ignore[assignment,misc]
    IndexComparisonResult = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Transition Risk Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_047_benchmark.workflows.transition_risk_workflow import (
        TransitionRiskWorkflow,
        TransitionRiskInput,
        TransitionRiskResult,
        TransitionPhase,
        BudgetMethod,
        RiskLevel,
        Quartile,
        BudgetAllocation,
        StrandingResult,
        RegulatoryExposure,
        CompetitivePosition,
        CompositeRiskScore,
    )
except ImportError:
    TransitionRiskWorkflow = None  # type: ignore[assignment,misc]
    TransitionRiskInput = None  # type: ignore[assignment,misc]
    TransitionRiskResult = None  # type: ignore[assignment,misc]
    TransitionPhase = None  # type: ignore[assignment,misc]
    BudgetMethod = None  # type: ignore[assignment,misc]
    RiskLevel = None  # type: ignore[assignment,misc]
    Quartile = None  # type: ignore[assignment,misc]
    BudgetAllocation = None  # type: ignore[assignment,misc]
    StrandingResult = None  # type: ignore[assignment,misc]
    RegulatoryExposure = None  # type: ignore[assignment,misc]
    CompetitivePosition = None  # type: ignore[assignment,misc]
    CompositeRiskScore = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Disclosure Preparation Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_047_benchmark.workflows.disclosure_preparation_workflow import (
        DisclosurePreparationWorkflow,
        DisclosurePreparationInput,
        DisclosurePreparationResult,
        DisclosurePhase,
        DisclosureFramework,
        FieldStatus,
        GapSeverity,
        QACheckType,
        QAOutcome,
        PackageFormat,
        BenchmarkMetricInput,
        AggregatedBenchmarkMetric,
        FrameworkField,
        QACheckResult,
        CompletenessGap,
        DisclosurePackageItem,
    )
except ImportError:
    DisclosurePreparationWorkflow = None  # type: ignore[assignment,misc]
    DisclosurePreparationInput = None  # type: ignore[assignment,misc]
    DisclosurePreparationResult = None  # type: ignore[assignment,misc]
    DisclosurePhase = None  # type: ignore[assignment,misc]
    DisclosureFramework = None  # type: ignore[assignment,misc]
    FieldStatus = None  # type: ignore[assignment,misc]
    GapSeverity = None  # type: ignore[assignment,misc]
    QACheckType = None  # type: ignore[assignment,misc]
    QAOutcome = None  # type: ignore[assignment,misc]
    PackageFormat = None  # type: ignore[assignment,misc]
    BenchmarkMetricInput = None  # type: ignore[assignment,misc]
    AggregatedBenchmarkMetric = None  # type: ignore[assignment,misc]
    FrameworkField = None  # type: ignore[assignment,misc]
    QACheckResult = None  # type: ignore[assignment,misc]
    CompletenessGap = None  # type: ignore[assignment,misc]
    DisclosurePackageItem = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Full Benchmark Pipeline Workflow
# ---------------------------------------------------------------------------
try:
    from packs.ghg_accounting.PACK_047_benchmark.workflows.full_benchmark_pipeline_workflow import (
        FullBenchmarkPipelineWorkflow,
        FullPipelineInput,
        FullPipelineResult,
        PipelinePhase,
        PipelineMilestoneType,
        ReportType,
        PipelinePhaseStatus,
        PipelineCheckpoint,
        PipelineMilestone,
        PipelineReport,
        EmissionsInput,
        PeerInput,
    )
except ImportError:
    FullBenchmarkPipelineWorkflow = None  # type: ignore[assignment,misc]
    FullPipelineInput = None  # type: ignore[assignment,misc]
    FullPipelineResult = None  # type: ignore[assignment,misc]
    PipelinePhase = None  # type: ignore[assignment,misc]
    PipelineMilestoneType = None  # type: ignore[assignment,misc]
    ReportType = None  # type: ignore[assignment,misc]
    PipelinePhaseStatus = None  # type: ignore[assignment,misc]
    PipelineCheckpoint = None  # type: ignore[assignment,misc]
    PipelineMilestone = None  # type: ignore[assignment,misc]
    PipelineReport = None  # type: ignore[assignment,misc]
    EmissionsInput = None  # type: ignore[assignment,misc]
    PeerInput = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Type Aliases
# ---------------------------------------------------------------------------

# Peer Group Setup types
PeerSetupIn = PeerGroupSetupInput
PeerSetupOut = PeerGroupSetupResult

# Benchmark Assessment types
AssessmentIn = BenchmarkAssessmentInput
AssessmentOut = BenchmarkAssessmentResult

# Pathway Alignment types
PathwayIn = PathwayAlignmentInput
PathwayOut = PathwayAlignmentResult

# Trajectory Analysis types
TrajectoryIn = TrajectoryAnalysisInput
TrajectoryOut = TrajectoryAnalysisResult

# Portfolio Benchmark types
PortfolioIn = PortfolioBenchmarkInput
PortfolioOut = PortfolioBenchmarkResult

# Transition Risk types
RiskIn = TransitionRiskInput
RiskOut = TransitionRiskResult

# Disclosure Preparation types
DisclosureIn = DisclosurePreparationInput
DisclosureOut = DisclosurePreparationResult

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
        "peer_group_setup": PeerGroupSetupWorkflow,
        "benchmark_assessment": BenchmarkAssessmentWorkflow,
        "pathway_alignment": PathwayAlignmentWorkflow,
        "trajectory_analysis": TrajectoryAnalysisWorkflow,
        "portfolio_benchmark": PortfolioBenchmarkWorkflow,
        "transition_risk": TransitionRiskWorkflow,
        "disclosure_preparation": DisclosurePreparationWorkflow,
        "full_benchmark_pipeline": FullBenchmarkPipelineWorkflow,
    }


__all__ = [
    # --- Peer Group Setup Workflow ---
    "PeerGroupSetupWorkflow",
    "PeerGroupSetupInput",
    "PeerGroupSetupResult",
    "PeerSetupPhase",
    "SectorSystem",
    "GICSsector",
    "SizeBand",
    "GeographicRegion",
    "DataQualityLevel",
    "PeerStatus",
    "ValidationSeverity",
    "SectorMapping",
    "PeerCandidate",
    "ValidationFinding",
    "PeerGroupStats",
    # --- Benchmark Assessment Workflow ---
    "BenchmarkAssessmentWorkflow",
    "BenchmarkAssessmentInput",
    "BenchmarkAssessmentResult",
    "AssessmentPhase",
    "GWPVintage",
    "NormalisationMethod",
    "PerformanceBand",
    "RankingMetric",
    "ReportSection",
    "EntityEmissions",
    "NormalisationAdjustment",
    "PeerDistribution",
    "GapAnalysisResult",
    "PercentileRank",
    "ReportItem",
    # --- Pathway Alignment Workflow ---
    "PathwayAlignmentWorkflow",
    "PathwayAlignmentInput",
    "PathwayAlignmentResult",
    "AlignmentPhase",
    "PathwaySource",
    "InterpolationMethod",
    "TemperatureAlignment",
    "AlignmentStatus",
    "PathwayDefinition",
    "AnnualWaypoint",
    "PathwayGap",
    "PathwayAlignmentScore",
    # --- Trajectory Analysis Workflow ---
    "TrajectoryAnalysisWorkflow",
    "TrajectoryAnalysisInput",
    "TrajectoryAnalysisResult",
    "TrajectoryPhase",
    "TrajectoryBand",
    "ConvergenceType",
    "ConvergenceStatus",
    "EntityTimeSeries",
    "CARRResult",
    "ConvergenceResult",
    "TrajectoryRank",
    # --- Portfolio Benchmark Workflow ---
    "PortfolioBenchmarkWorkflow",
    "PortfolioBenchmarkInput",
    "PortfolioBenchmarkResult",
    "PortfolioPhase",
    "AssetClass",
    "PCAFLevel",
    "BenchmarkIndex",
    "PortfolioHolding",
    "PCAFQualityResult",
    "AssetClassAggregation",
    "PortfolioMetric",
    "IndexComparisonResult",
    # --- Transition Risk Workflow ---
    "TransitionRiskWorkflow",
    "TransitionRiskInput",
    "TransitionRiskResult",
    "TransitionPhase",
    "BudgetMethod",
    "RiskLevel",
    "Quartile",
    "BudgetAllocation",
    "StrandingResult",
    "RegulatoryExposure",
    "CompetitivePosition",
    "CompositeRiskScore",
    # --- Disclosure Preparation Workflow ---
    "DisclosurePreparationWorkflow",
    "DisclosurePreparationInput",
    "DisclosurePreparationResult",
    "DisclosurePhase",
    "DisclosureFramework",
    "FieldStatus",
    "GapSeverity",
    "QACheckType",
    "QAOutcome",
    "PackageFormat",
    "BenchmarkMetricInput",
    "AggregatedBenchmarkMetric",
    "FrameworkField",
    "QACheckResult",
    "CompletenessGap",
    "DisclosurePackageItem",
    # --- Full Benchmark Pipeline Workflow ---
    "FullBenchmarkPipelineWorkflow",
    "FullPipelineInput",
    "FullPipelineResult",
    "PipelinePhase",
    "PipelineMilestoneType",
    "ReportType",
    "PipelinePhaseStatus",
    "PipelineCheckpoint",
    "PipelineMilestone",
    "PipelineReport",
    "EmissionsInput",
    "PeerInput",
    # --- Type Aliases ---
    "PeerSetupIn",
    "PeerSetupOut",
    "AssessmentIn",
    "AssessmentOut",
    "PathwayIn",
    "PathwayOut",
    "TrajectoryIn",
    "TrajectoryOut",
    "PortfolioIn",
    "PortfolioOut",
    "RiskIn",
    "RiskOut",
    "DisclosureIn",
    "DisclosureOut",
    "FullPipelineIn",
    "FullPipelineOut",
    # --- Helpers ---
    "get_loaded_workflows",
]
