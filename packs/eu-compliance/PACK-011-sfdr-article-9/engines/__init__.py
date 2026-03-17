# -*- coding: utf-8 -*-
"""
PACK-011 SFDR Article 9 Pack - Engines Module
================================================

Calculation engines for SFDR Article 9 ("dark green") product compliance
and disclosure.  Article 9 products have sustainable investment as their
objective -- every holding must qualify as a sustainable investment per
Article 2(17) of the SFDR.

Engines:
    1. SustainableObjectiveEngine       - Verify ALL investments qualify as sustainable
    2. EnhancedDNSHEngine               - Stricter DNSH for 100 % portfolio coverage
    3. FullTaxonomyAlignmentEngine      - Full EU Taxonomy alignment (Articles 5/6)
    4. ImpactMeasurementEngine          - Sustainability impact KPIs and SDG mapping
    5. BenchmarkAlignmentEngine         - EU Climate Benchmark (CTB/PAB) alignment
    6. PAIMandatoryEngine               - Mandatory PAI indicators (18 + additional)
    7. CarbonTrajectoryEngine           - Carbon trajectory, ITR, SBT, Net Zero
    8. InvestmentUniverseEngine         - Investment universe screening & exclusions
"""

from .sustainable_objective_engine import (
    SustainableObjectiveEngine,
    SustainableObjectiveConfig,
    HoldingData,
    SustainableObjectiveResult,
    HoldingClassification,
    CommitmentStatus,
    NonSustainableBreakdown,
    ObjectiveBreakdownEntry,
    ComplianceReport,
    ObjectiveType,
    EnvironmentalObjective,
    SocialObjective,
    ComplianceStatus,
    HoldingClassificationType,
)

from .enhanced_dnsh_engine import (
    EnhancedDNSHEngine,
    EnhancedDNSHConfig,
    HoldingPAIData,
    HoldingDNSHResult,
    PortfolioDNSHResult,
    RemediationPlan,
    RemediationStep,
    AutoExclusionResult,
    PAIThreshold,
    PAICheckResult,
    PAICategory,
    DNSHStatus,
    ThresholdDirection,
    SeverityLevel,
    ExclusionReason,
)

from .full_taxonomy_alignment import (
    FullTaxonomyAlignmentEngine,
    FullTaxonomyConfig,
    TaxonomyHoldingData,
    FullTaxonomyResult,
    MinimumSafeguardsResult,
    Article5Disclosure,
    Article6Disclosure,
    BarChartData,
    BarChartSeries,
    ObjectiveAlignmentEntry,
    TaxonomyEnvironmentalObjective,
    ArticleReference,
    SafeguardArea,
)

from .impact_measurement_engine import (
    ImpactMeasurementEngine,
    ImpactConfig,
    ImpactKPI,
    ImpactResult,
    SDGContribution,
    TheoryOfChange,
    AdditionalityResult,
    PeriodComparison,
    KPIUpdate,
    KPIDefinition,
    ImpactCategory,
    SDGGoal,
    ToCStage,
)

from .benchmark_alignment_engine import (
    BenchmarkAlignmentEngine,
    BenchmarkConfig,
    HoldingBenchmarkData,
    BenchmarkResult,
    CTBComplianceResult,
    PABComplianceResult,
    ExclusionViolation,
    TrajectoryDataPoint,
    TrackingErrorResult,
    MethodologyDisclosure,
    BenchmarkType,
    ComplianceStatus as BenchmarkComplianceStatus,
    ExclusionCategory,
)

from .pai_mandatory_engine import (
    PAIMandatoryEngine,
    PAIMandatoryConfig,
    InvesteeFullData,
    PAIMandatoryResult,
    PAISingleResult,
    IntegrationAssessment,
    ActionPlan,
    ActionPlanItem,
    DataQualityReport,
    AdditionalPAIResult,
    PAIMandatoryStatus,
    DataQualityLevel,
    PAIIndicatorId,
    PAICategory as PAIMandatoryCategory,
)

from .carbon_trajectory_engine import (
    CarbonTrajectoryEngine,
    TrajectoryConfig,
    HoldingTrajectoryData,
    TrajectoryResult,
    ITRResult,
    CarbonBudgetResult,
    SBTCoverageResult,
    NetZeroProgress,
    CarbonPathway,
    TransitionPlanQuality,
)

from .investment_universe_engine import (
    InvestmentUniverseEngine,
    UniverseConfig,
    SecurityData,
    ScreeningResult,
    ExclusionDetail,
    WatchListEntry,
    PreApprovalResult,
    UniverseCoverage,
    ScreeningLayer,
    ExclusionType,
)

__all__ = [
    # Engine 1: Sustainable Objective
    "SustainableObjectiveEngine",
    "SustainableObjectiveConfig",
    "HoldingData",
    "SustainableObjectiveResult",
    "HoldingClassification",
    "CommitmentStatus",
    "NonSustainableBreakdown",
    "ObjectiveBreakdownEntry",
    "ComplianceReport",
    "ObjectiveType",
    "EnvironmentalObjective",
    "SocialObjective",
    "ComplianceStatus",
    "HoldingClassificationType",
    # Engine 2: Enhanced DNSH
    "EnhancedDNSHEngine",
    "EnhancedDNSHConfig",
    "HoldingPAIData",
    "HoldingDNSHResult",
    "PortfolioDNSHResult",
    "RemediationPlan",
    "RemediationStep",
    "AutoExclusionResult",
    "PAIThreshold",
    "PAICheckResult",
    "PAICategory",
    "DNSHStatus",
    "ThresholdDirection",
    "SeverityLevel",
    "ExclusionReason",
    # Engine 3: Full Taxonomy Alignment
    "FullTaxonomyAlignmentEngine",
    "FullTaxonomyConfig",
    "TaxonomyHoldingData",
    "FullTaxonomyResult",
    "MinimumSafeguardsResult",
    "Article5Disclosure",
    "Article6Disclosure",
    "BarChartData",
    "BarChartSeries",
    "ObjectiveAlignmentEntry",
    "TaxonomyEnvironmentalObjective",
    "ArticleReference",
    "SafeguardArea",
    # Engine 4: Impact Measurement
    "ImpactMeasurementEngine",
    "ImpactConfig",
    "ImpactKPI",
    "ImpactResult",
    "SDGContribution",
    "TheoryOfChange",
    "AdditionalityResult",
    "PeriodComparison",
    "KPIUpdate",
    "KPIDefinition",
    "ImpactCategory",
    "SDGGoal",
    "ToCStage",
    # Engine 5: Benchmark Alignment
    "BenchmarkAlignmentEngine",
    "BenchmarkConfig",
    "HoldingBenchmarkData",
    "BenchmarkResult",
    "CTBComplianceResult",
    "PABComplianceResult",
    "ExclusionViolation",
    "TrajectoryDataPoint",
    "TrackingErrorResult",
    "MethodologyDisclosure",
    "BenchmarkType",
    "BenchmarkComplianceStatus",
    "ExclusionCategory",
    # Engine 6: PAI Mandatory
    "PAIMandatoryEngine",
    "PAIMandatoryConfig",
    "InvesteeFullData",
    "PAIMandatoryResult",
    "PAISingleResult",
    "IntegrationAssessment",
    "ActionPlan",
    "ActionPlanItem",
    "DataQualityReport",
    "AdditionalPAIResult",
    "PAIMandatoryStatus",
    "DataQualityLevel",
    "PAIIndicatorId",
    "PAIMandatoryCategory",
    # Engine 7: Carbon Trajectory
    "CarbonTrajectoryEngine",
    "TrajectoryConfig",
    "HoldingTrajectoryData",
    "TrajectoryResult",
    "ITRResult",
    "CarbonBudgetResult",
    "SBTCoverageResult",
    "NetZeroProgress",
    "CarbonPathway",
    "TransitionPlanQuality",
    # Engine 8: Investment Universe
    "InvestmentUniverseEngine",
    "UniverseConfig",
    "SecurityData",
    "ScreeningResult",
    "ExclusionDetail",
    "WatchListEntry",
    "PreApprovalResult",
    "UniverseCoverage",
    "ScreeningLayer",
    "ExclusionType",
]
