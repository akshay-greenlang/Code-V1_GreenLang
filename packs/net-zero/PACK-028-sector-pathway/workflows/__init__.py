# -*- coding: utf-8 -*-
"""
PACK-028 Sector Pathway Pack - Workflow Layer
==================================================

6 sector-specific decarbonization workflows for SBTi SDA-aligned pathway
design, validation, technology planning, progress monitoring, multi-scenario
analysis, and full sector assessment.  All workflows use DAG orchestration,
async execution, error handling, progress tracking, and SHA-256 provenance
hashing.

Workflows:
    1. SectorPathwayDesignWorkflow (5 phases)
       SectorClassify -> IntensityCalc -> PathwayGen
       -> GapAnalysis -> ValidationReport

    2. PathwayValidationWorkflow (4 phases)
       DataValidation -> PathwayValidation -> SBTiCriteriaCheck
       -> ComplianceReport

    3. TechnologyPlanningWorkflow (5 phases)
       TechInventory -> RoadmapGen -> CapExMapping
       -> DependencyAnalysis -> ImplementationPlan

    4. ProgressMonitoringWorkflow (4 phases)
       IntensityUpdate -> ConvergenceCheck -> BenchmarkUpdate
       -> ProgressReport

    5. MultiScenarioAnalysisWorkflow (5 phases)
       ScenarioSetup -> PathwayModeling -> RiskAnalysis
       -> ScenarioComparison -> StrategyRecommend

    6. FullSectorAssessmentWorkflow (7 phases)
       Classify -> Pathway -> Technology -> Abatement
       -> Benchmark -> Scenarios -> Strategy

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-028 Sector Pathway Pack
Status: Production Ready
"""

__version__ = "1.0.0"
__pack_id__ = "PACK-028"
__pack_name__ = "Sector Pathway Pack"

# ---------------------------------------------------------------------------
# 1. Sector Pathway Design Workflow
# ---------------------------------------------------------------------------
from .sector_pathway_design_workflow import (
    SectorPathwayDesignWorkflow,
    SectorPathwayDesignConfig,
    SectorPathwayDesignInput,
    SectorPathwayDesignResult,
    SectorClassification,
    IntensityMetric,
    PathwayPoint,
    SectorPathway,
    GapAnalysisResult,
    ValidationCriterion,
    ValidationReport,
    SDAEligibility,
    ConvergenceModel,
    ClimateScenario,
    GapSeverity,
    ValidationStatus,
    SDA_SECTORS,
    IEA_SCENARIO_MULTIPLIERS,
    PhaseResult as DesignPhaseResult,
    PhaseStatus as DesignPhaseStatus,
    WorkflowStatus as DesignWorkflowStatus,
)

# ---------------------------------------------------------------------------
# 2. Pathway Validation Workflow
# ---------------------------------------------------------------------------
from .pathway_validation_workflow import (
    PathwayValidationWorkflow,
    PathwayValidationConfig,
    PathwayValidationInput,
    PathwayValidationResult,
    DataValidationFinding,
    DataValidationSummary,
    PathwayValidationCheck,
    PathwayValidationSummary,
    SBTiCriterionCheck,
    SBTiComplianceSummary,
    ComplianceReportSection,
    ComplianceReport,
    ValidationSeverity,
    ComplianceStatus,
    DataQualityTier,
    PathwayProperty,
    SBTI_NEAR_TERM_REQUIREMENTS,
    SBTI_LONG_TERM_REQUIREMENTS,
    SDA_SECTOR_REQUIREMENTS,
    PhaseResult as ValidationPhaseResult,
    PhaseStatus as ValidationPhaseStatus,
    WorkflowStatus as ValidationWorkflowStatus,
)

# ---------------------------------------------------------------------------
# 3. Technology Planning Workflow
# ---------------------------------------------------------------------------
from .technology_planning_workflow import (
    TechnologyPlanningWorkflow,
    TechnologyPlanningConfig,
    TechnologyPlanningInput,
    TechnologyPlanningResult,
    TechnologyItem,
    RoadmapMilestone,
    TechRoadmap,
    CapExAllocation,
    CapExPlan,
    TechDependency,
    SupplyChainRisk,
    DependencyAnalysis,
    ImplementationAction,
    ImplementationPlan,
    TRL,
    TechCategory,
    RiskLevel as TechRiskLevel,
    MilestoneStatus,
    ImplementationPriority,
    IEA_SECTOR_TECHNOLOGIES,
    PhaseResult as TechPhaseResult,
    PhaseStatus as TechPhaseStatus,
    WorkflowStatus as TechWorkflowStatus,
)

# ---------------------------------------------------------------------------
# 4. Progress Monitoring Workflow
# ---------------------------------------------------------------------------
from .progress_monitoring_workflow import (
    ProgressMonitoringWorkflow,
    ProgressMonitoringConfig,
    ProgressMonitoringInput,
    ProgressMonitoringResult,
    IntensityDataPoint,
    IntensityUpdate,
    ConvergenceResult,
    BenchmarkComparison,
    BenchmarkSummary,
    ProgressAlert,
    ProgressKPI,
    ProgressReport,
    RAGStatus,
    TrendDirection,
    AlertSeverity,
    ConvergenceStatus,
    BenchmarkSource,
    SECTOR_BENCHMARKS,
    PhaseResult as ProgressPhaseResult,
    PhaseStatus as ProgressPhaseStatus,
    WorkflowStatus as ProgressWorkflowStatus,
)

# ---------------------------------------------------------------------------
# 5. Multi-Scenario Analysis Workflow
# ---------------------------------------------------------------------------
from .multi_scenario_analysis_workflow import (
    MultiScenarioAnalysisWorkflow,
    MultiScenarioConfig,
    MultiScenarioInput,
    MultiScenarioResult,
    ScenarioSetup,
    ScenarioPathwayResult,
    RiskAssessment,
    ScenarioRiskProfile,
    ScenarioComparisonMetric,
    ScenarioComparisonMatrix,
    StrategyRecommendation,
    ClimateScenario as ScenarioClimateScenario,
    RiskCategory,
    RiskLevel as ScenarioRiskLevel,
    RecommendationConfidence,
    SCENARIO_DEFINITIONS,
    PhaseResult as ScenarioPhaseResult,
    PhaseStatus as ScenarioPhaseStatus,
    WorkflowStatus as ScenarioWorkflowStatus,
)

# ---------------------------------------------------------------------------
# 6. Full Sector Assessment Workflow (Master)
# ---------------------------------------------------------------------------
from .full_sector_assessment_workflow import (
    FullSectorAssessmentWorkflow,
    FullSectorAssessmentConfig,
    FullSectorAssessmentInput,
    FullSectorAssessmentResult,
    SectorScorecard,
    AbatementLever,
    AbatementWaterfall,
    SectorStrategySummary,
    MaturityLevel,
    SECTOR_LEVERS,
    PhaseResult as AssessmentPhaseResult,
    PhaseStatus as AssessmentPhaseStatus,
    WorkflowStatus as AssessmentWorkflowStatus,
)

__all__ = [
    "__version__",
    "__pack_id__",
    "__pack_name__",
    # --- 1. Sector Pathway Design ---
    "SectorPathwayDesignWorkflow",
    "SectorPathwayDesignConfig",
    "SectorPathwayDesignInput",
    "SectorPathwayDesignResult",
    "SectorClassification",
    "IntensityMetric",
    "PathwayPoint",
    "SectorPathway",
    "GapAnalysisResult",
    "ValidationCriterion",
    "ValidationReport",
    "SDAEligibility",
    "ConvergenceModel",
    "ClimateScenario",
    "GapSeverity",
    "ValidationStatus",
    "SDA_SECTORS",
    "IEA_SCENARIO_MULTIPLIERS",
    # --- 2. Pathway Validation ---
    "PathwayValidationWorkflow",
    "PathwayValidationConfig",
    "PathwayValidationInput",
    "PathwayValidationResult",
    "DataValidationFinding",
    "DataValidationSummary",
    "PathwayValidationCheck",
    "PathwayValidationSummary",
    "SBTiCriterionCheck",
    "SBTiComplianceSummary",
    "ComplianceReportSection",
    "ComplianceReport",
    "ValidationSeverity",
    "ComplianceStatus",
    "DataQualityTier",
    "PathwayProperty",
    "SBTI_NEAR_TERM_REQUIREMENTS",
    "SBTI_LONG_TERM_REQUIREMENTS",
    "SDA_SECTOR_REQUIREMENTS",
    # --- 3. Technology Planning ---
    "TechnologyPlanningWorkflow",
    "TechnologyPlanningConfig",
    "TechnologyPlanningInput",
    "TechnologyPlanningResult",
    "TechnologyItem",
    "RoadmapMilestone",
    "TechRoadmap",
    "CapExAllocation",
    "CapExPlan",
    "TechDependency",
    "SupplyChainRisk",
    "DependencyAnalysis",
    "ImplementationAction",
    "ImplementationPlan",
    "TRL",
    "TechCategory",
    "TechRiskLevel",
    "MilestoneStatus",
    "ImplementationPriority",
    "IEA_SECTOR_TECHNOLOGIES",
    # --- 4. Progress Monitoring ---
    "ProgressMonitoringWorkflow",
    "ProgressMonitoringConfig",
    "ProgressMonitoringInput",
    "ProgressMonitoringResult",
    "IntensityDataPoint",
    "IntensityUpdate",
    "ConvergenceResult",
    "BenchmarkComparison",
    "BenchmarkSummary",
    "ProgressAlert",
    "ProgressKPI",
    "ProgressReport",
    "RAGStatus",
    "TrendDirection",
    "AlertSeverity",
    "ConvergenceStatus",
    "BenchmarkSource",
    "SECTOR_BENCHMARKS",
    # --- 5. Multi-Scenario Analysis ---
    "MultiScenarioAnalysisWorkflow",
    "MultiScenarioConfig",
    "MultiScenarioInput",
    "MultiScenarioResult",
    "ScenarioSetup",
    "ScenarioPathwayResult",
    "RiskAssessment",
    "ScenarioRiskProfile",
    "ScenarioComparisonMetric",
    "ScenarioComparisonMatrix",
    "StrategyRecommendation",
    "ScenarioClimateScenario",
    "RiskCategory",
    "ScenarioRiskLevel",
    "RecommendationConfidence",
    "SCENARIO_DEFINITIONS",
    # --- 6. Full Sector Assessment ---
    "FullSectorAssessmentWorkflow",
    "FullSectorAssessmentConfig",
    "FullSectorAssessmentInput",
    "FullSectorAssessmentResult",
    "SectorScorecard",
    "AbatementLever",
    "AbatementWaterfall",
    "SectorStrategySummary",
    "MaturityLevel",
    "SECTOR_LEVERS",
]
