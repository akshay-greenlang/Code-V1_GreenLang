# -*- coding: utf-8 -*-
"""
PACK-022 Net-Zero Acceleration Pack - Workflow Orchestration
================================================================

Advanced net-zero workflow orchestrators for scenario analysis, SDA
target setting, supplier engagement programs, transition finance
evaluation, advanced progress tracking, temperature alignment scoring,
VCMI certification assessment, and full acceleration strategy compilation.

Workflows:
    - ScenarioAnalysisWorkflow: 5-phase Monte Carlo scenario comparison
      with setup, model run, tornado sensitivity, multi-dimension
      comparison, and decision matrix recommendation.

    - SDATargetWorkflow: 4-phase Sectoral Decarbonisation Approach
      with sector classification, IEA NZE benchmark pathway calculation,
      company-specific intensity target convergence, and SBTi SDA
      validation with ACA comparison.

    - SupplierProgramWorkflow: 5-phase Scope 3 supplier engagement
      with assessment, Pareto-based tiering, program design per tier,
      execution tracking with RAG status, and impact reporting.

    - TransitionFinanceWorkflow: 4-phase climate investment evaluation
      with CapEx mapping, EU Taxonomy alignment (SC + DNSH), ICMA
      Green Bond Principles screening, and NPV/IRR investment case.

    - AdvancedProgressWorkflow: 5-phase emission progress analysis
      with data ingestion, LMDI-I decomposition (activity/intensity/
      structural), driver attribution, linear forecast, and deviation
      alert generation.

    - TemperatureAlignmentWorkflow: 4-phase SBTi temperature scoring
      with target collection, per-entity score calculation, portfolio
      aggregation (WATS/TETS/MOTS/EOTS), and contribution reporting.

    - VCMICertificationWorkflow: 4-phase VCMI Claims Code assessment
      with evidence collection, foundational criteria scoring, claim
      tier determination (Silver/Gold/Platinum), and certification
      report with greenwashing risk analysis.

    - FullAccelerationWorkflow: 8-phase master workflow chaining all
      sub-workflows with conditional SDA (sector-dependent) and VCMI
      (credit-dependent) phases, acceleration scorecard, and unified
      strategy compilation.

Author: GreenLang Team
Version: 22.0.0
"""

# ---------------------------------------------------------------------------
# Scenario Analysis Workflow
# ---------------------------------------------------------------------------
from .scenario_analysis_workflow import (
    ScenarioAnalysisWorkflow,
    ScenarioAnalysisConfig,
    ScenarioAnalysisResult,
    ScenarioDefinition,
    MonteCarloDistribution,
    SensitivityParameter,
    ScenarioComparison,
    DecisionMatrix,
    CarbonPriceScenario,
    ComparisonDimension,
    PhaseResult as ScenarioPhaseResult,
    PhaseStatus as ScenarioPhaseStatus,
    WorkflowStatus as ScenarioWorkflowStatus,
)

# ---------------------------------------------------------------------------
# SDA Target Workflow
# ---------------------------------------------------------------------------
from .sda_target_workflow import (
    SDATargetWorkflow,
    SDATargetConfig,
    SDATargetResult,
    SectorClassificationResult,
    BenchmarkPathwayPoint,
    BenchmarkPathway,
    CompanyIntensityTarget,
    ACAComparison,
    SDAValidationFinding,
    SDAValidationReport,
    SDASector,
    PhaseResult as SDAPhaseResult,
    PhaseStatus as SDAPhaseStatus,
    WorkflowStatus as SDAWorkflowStatus,
)

# ---------------------------------------------------------------------------
# Supplier Program Workflow
# ---------------------------------------------------------------------------
from .supplier_program_workflow import (
    SupplierProgramWorkflow,
    SupplierProgramConfig,
    SupplierProgramResult,
    SupplierRecord,
    SupplierAssessment,
    TierSummary,
    EngagementMilestone,
    TierProgram,
    SupplierProgress,
    ImpactReport,
    SupplierTier,
    SupplierMaturity,
    RAGStatus,
    PhaseResult as SupplierPhaseResult,
    PhaseStatus as SupplierPhaseStatus,
    WorkflowStatus as SupplierWorkflowStatus,
)

# ---------------------------------------------------------------------------
# Transition Finance Workflow
# ---------------------------------------------------------------------------
from .transition_finance_workflow import (
    TransitionFinanceWorkflow,
    TransitionFinanceConfig,
    TransitionFinanceResult,
    CapExItem,
    CapExClassification,
    TaxonomyAlignment,
    BondEligibility,
    InvestmentCaseResult,
    CapExCategory,
    TaxonomyObjective,
    PhaseResult as FinancePhaseResult,
    PhaseStatus as FinancePhaseStatus,
    WorkflowStatus as FinanceWorkflowStatus,
)

# ---------------------------------------------------------------------------
# Advanced Progress Workflow
# ---------------------------------------------------------------------------
from .advanced_progress_workflow import (
    AdvancedProgressWorkflow,
    AdvancedProgressConfig,
    AdvancedProgressResult,
    AnnualEmissionRecord,
    BusinessUnitEmission,
    TargetPathwayPoint,
    LMDIDecomposition,
    DriverAttribution,
    ForecastPoint,
    ProgressAlert,
    AlertSeverity,
    TrendDirection,
    PhaseResult as ProgressPhaseResult,
    PhaseStatus as ProgressPhaseStatus,
    WorkflowStatus as ProgressWorkflowStatus,
)

# ---------------------------------------------------------------------------
# Temperature Alignment Workflow
# ---------------------------------------------------------------------------
from .temperature_alignment_workflow import (
    TemperatureAlignmentWorkflow,
    TemperatureAlignmentConfig,
    TemperatureAlignmentResult,
    EntityTarget,
    EntityWeight,
    TemperatureScore,
    PortfolioTemperature,
    ContributionAnalysis,
    TargetScope,
    TargetTimeframe,
    AggregationMethod,
    PhaseResult as TempPhaseResult,
    PhaseStatus as TempPhaseStatus,
    WorkflowStatus as TempWorkflowStatus,
)

# ---------------------------------------------------------------------------
# VCMI Certification Workflow
# ---------------------------------------------------------------------------
from .vcmi_certification_workflow import (
    VCMICertificationWorkflow,
    VCMICertificationConfig,
    VCMICertificationResult,
    EvidenceItem,
    CreditPortfolioItem,
    CriterionAssessment,
    ClaimDetermination,
    ClaimTier,
    CriterionStatus,
    GreenwashingRisk,
    PhaseResult as VCMIPhaseResult,
    PhaseStatus as VCMIPhaseStatus,
    WorkflowStatus as VCMIWorkflowStatus,
)

# ---------------------------------------------------------------------------
# Full Acceleration Workflow (Master)
# ---------------------------------------------------------------------------
from .full_acceleration_workflow import (
    FullAccelerationWorkflow,
    FullAccelerationConfig,
    FullAccelerationResult,
    AccelerationScorecard,
    AccelerationStrategySummary,
    AccelerationMaturity,
    PhaseResult as AccelerationPhaseResult,
    PhaseStatus as AccelerationPhaseStatus,
    WorkflowStatus as AccelerationWorkflowStatus,
)

__all__ = [
    # --- Scenario Analysis Workflow ---
    "ScenarioAnalysisWorkflow",
    "ScenarioAnalysisConfig",
    "ScenarioAnalysisResult",
    "ScenarioDefinition",
    "MonteCarloDistribution",
    "SensitivityParameter",
    "ScenarioComparison",
    "DecisionMatrix",
    "CarbonPriceScenario",
    "ComparisonDimension",
    # --- SDA Target Workflow ---
    "SDATargetWorkflow",
    "SDATargetConfig",
    "SDATargetResult",
    "SectorClassificationResult",
    "BenchmarkPathwayPoint",
    "BenchmarkPathway",
    "CompanyIntensityTarget",
    "ACAComparison",
    "SDAValidationFinding",
    "SDAValidationReport",
    "SDASector",
    # --- Supplier Program Workflow ---
    "SupplierProgramWorkflow",
    "SupplierProgramConfig",
    "SupplierProgramResult",
    "SupplierRecord",
    "SupplierAssessment",
    "TierSummary",
    "EngagementMilestone",
    "TierProgram",
    "SupplierProgress",
    "ImpactReport",
    "SupplierTier",
    "SupplierMaturity",
    "RAGStatus",
    # --- Transition Finance Workflow ---
    "TransitionFinanceWorkflow",
    "TransitionFinanceConfig",
    "TransitionFinanceResult",
    "CapExItem",
    "CapExClassification",
    "TaxonomyAlignment",
    "BondEligibility",
    "InvestmentCaseResult",
    "CapExCategory",
    "TaxonomyObjective",
    # --- Advanced Progress Workflow ---
    "AdvancedProgressWorkflow",
    "AdvancedProgressConfig",
    "AdvancedProgressResult",
    "AnnualEmissionRecord",
    "BusinessUnitEmission",
    "TargetPathwayPoint",
    "LMDIDecomposition",
    "DriverAttribution",
    "ForecastPoint",
    "ProgressAlert",
    "AlertSeverity",
    "TrendDirection",
    # --- Temperature Alignment Workflow ---
    "TemperatureAlignmentWorkflow",
    "TemperatureAlignmentConfig",
    "TemperatureAlignmentResult",
    "EntityTarget",
    "EntityWeight",
    "TemperatureScore",
    "PortfolioTemperature",
    "ContributionAnalysis",
    "TargetScope",
    "TargetTimeframe",
    "AggregationMethod",
    # --- VCMI Certification Workflow ---
    "VCMICertificationWorkflow",
    "VCMICertificationConfig",
    "VCMICertificationResult",
    "EvidenceItem",
    "CreditPortfolioItem",
    "CriterionAssessment",
    "ClaimDetermination",
    "ClaimTier",
    "CriterionStatus",
    "GreenwashingRisk",
    # --- Full Acceleration Workflow ---
    "FullAccelerationWorkflow",
    "FullAccelerationConfig",
    "FullAccelerationResult",
    "AccelerationScorecard",
    "AccelerationStrategySummary",
    "AccelerationMaturity",
]
