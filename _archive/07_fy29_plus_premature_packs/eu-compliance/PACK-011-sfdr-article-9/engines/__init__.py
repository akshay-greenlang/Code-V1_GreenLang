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

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack__: str = "PACK-011"
__pack_name__: str = "SFDR Article 9 Pack"
__engines_count__: int = 8

# Track which engines loaded successfully
_loaded_engines: list[str] = []


# ===================================================================
# Engine 1: Sustainable Objective
# ===================================================================
_ENGINE_1_SYMBOLS: list[str] = [
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
]

try:
    from .sustainable_objective_engine import (
        CommitmentStatus,
        ComplianceReport,
        ComplianceStatus,
        EnvironmentalObjective,
        HoldingClassification,
        HoldingClassificationType,
        HoldingData,
        NonSustainableBreakdown,
        ObjectiveBreakdownEntry,
        ObjectiveType,
        SocialObjective,
        SustainableObjectiveConfig,
        SustainableObjectiveEngine,
        SustainableObjectiveResult,
    )
    _loaded_engines.append("SustainableObjectiveEngine")
except ImportError as e:
    logger.debug("Engine 1 (SustainableObjectiveEngine) not available: %s", e)
    _ENGINE_1_SYMBOLS = []


# ===================================================================
# Engine 2: Enhanced DNSH
# ===================================================================
_ENGINE_2_SYMBOLS: list[str] = [
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
]

try:
    from .enhanced_dnsh_engine import (
        AutoExclusionResult,
        DNSHStatus,
        EnhancedDNSHConfig,
        EnhancedDNSHEngine,
        ExclusionReason,
        HoldingDNSHResult,
        HoldingPAIData,
        PAICategory,
        PAICheckResult,
        PAIThreshold,
        PortfolioDNSHResult,
        RemediationPlan,
        RemediationStep,
        SeverityLevel,
        ThresholdDirection,
    )
    _loaded_engines.append("EnhancedDNSHEngine")
except ImportError as e:
    logger.debug("Engine 2 (EnhancedDNSHEngine) not available: %s", e)
    _ENGINE_2_SYMBOLS = []


# ===================================================================
# Engine 3: Full Taxonomy Alignment
# ===================================================================
_ENGINE_3_SYMBOLS: list[str] = [
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
]

try:
    from .full_taxonomy_alignment import (
        Article5Disclosure,
        Article6Disclosure,
        ArticleReference,
        BarChartData,
        BarChartSeries,
        FullTaxonomyAlignmentEngine,
        FullTaxonomyConfig,
        FullTaxonomyResult,
        MinimumSafeguardsResult,
        ObjectiveAlignmentEntry,
        SafeguardArea,
        TaxonomyEnvironmentalObjective,
        TaxonomyHoldingData,
    )
    _loaded_engines.append("FullTaxonomyAlignmentEngine")
except ImportError as e:
    logger.debug("Engine 3 (FullTaxonomyAlignmentEngine) not available: %s", e)
    _ENGINE_3_SYMBOLS = []


# ===================================================================
# Engine 4: Impact Measurement
# ===================================================================
_ENGINE_4_SYMBOLS: list[str] = [
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
]

try:
    from .impact_measurement_engine import (
        AdditionalityResult,
        ImpactCategory,
        ImpactConfig,
        ImpactKPI,
        ImpactMeasurementEngine,
        ImpactResult,
        KPIDefinition,
        KPIUpdate,
        PeriodComparison,
        SDGContribution,
        SDGGoal,
        TheoryOfChange,
        ToCStage,
    )
    _loaded_engines.append("ImpactMeasurementEngine")
except ImportError as e:
    logger.debug("Engine 4 (ImpactMeasurementEngine) not available: %s", e)
    _ENGINE_4_SYMBOLS = []


# ===================================================================
# Engine 5: Benchmark Alignment
# ===================================================================
_ENGINE_5_SYMBOLS: list[str] = [
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
]

try:
    from .benchmark_alignment_engine import (
        BenchmarkAlignmentEngine,
        BenchmarkConfig,
        BenchmarkResult,
        BenchmarkType,
        ComplianceStatus as BenchmarkComplianceStatus,
        CTBComplianceResult,
        ExclusionCategory,
        ExclusionViolation,
        HoldingBenchmarkData,
        MethodologyDisclosure,
        PABComplianceResult,
        TrackingErrorResult,
        TrajectoryDataPoint,
    )
    _loaded_engines.append("BenchmarkAlignmentEngine")
except ImportError as e:
    logger.debug("Engine 5 (BenchmarkAlignmentEngine) not available: %s", e)
    _ENGINE_5_SYMBOLS = []


# ===================================================================
# Engine 6: PAI Mandatory
# ===================================================================
_ENGINE_6_SYMBOLS: list[str] = [
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
]

try:
    from .pai_mandatory_engine import (
        ActionPlan,
        ActionPlanItem,
        AdditionalPAIResult,
        DataQualityLevel,
        DataQualityReport,
        IntegrationAssessment,
        InvesteeFullData,
        PAICategory as PAIMandatoryCategory,
        PAIIndicatorId,
        PAIMandatoryConfig,
        PAIMandatoryEngine,
        PAIMandatoryResult,
        PAIMandatoryStatus,
        PAISingleResult,
    )
    _loaded_engines.append("PAIMandatoryEngine")
except ImportError as e:
    logger.debug("Engine 6 (PAIMandatoryEngine) not available: %s", e)
    _ENGINE_6_SYMBOLS = []


# ===================================================================
# Engine 7: Carbon Trajectory
# ===================================================================
_ENGINE_7_SYMBOLS: list[str] = [
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
]

try:
    from .carbon_trajectory_engine import (
        CarbonBudgetResult,
        CarbonPathway,
        CarbonTrajectoryEngine,
        HoldingTrajectoryData,
        ITRResult,
        NetZeroProgress,
        SBTCoverageResult,
        TrajectoryConfig,
        TrajectoryResult,
        TransitionPlanQuality,
    )
    _loaded_engines.append("CarbonTrajectoryEngine")
except ImportError as e:
    logger.debug("Engine 7 (CarbonTrajectoryEngine) not available: %s", e)
    _ENGINE_7_SYMBOLS = []


# ===================================================================
# Engine 8: Investment Universe
# ===================================================================
_ENGINE_8_SYMBOLS: list[str] = [
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

try:
    from .investment_universe_engine import (
        ExclusionDetail,
        ExclusionType,
        InvestmentUniverseEngine,
        PreApprovalResult,
        ScreeningLayer,
        ScreeningResult,
        SecurityData,
        UniverseConfig,
        UniverseCoverage,
        WatchListEntry,
    )
    _loaded_engines.append("InvestmentUniverseEngine")
except ImportError as e:
    logger.debug("Engine 8 (InvestmentUniverseEngine) not available: %s", e)
    _ENGINE_8_SYMBOLS = []


# ===================================================================
# Public API - dynamically collected from successfully loaded engines
# ===================================================================

_METADATA_SYMBOLS: list[str] = [
    "__version__",
    "__pack__",
    "__pack_name__",
    "__engines_count__",
]

__all__: list[str] = [
    *_METADATA_SYMBOLS,
    # Engine 1: Sustainable Objective
    *_ENGINE_1_SYMBOLS,
    # Engine 2: Enhanced DNSH
    *_ENGINE_2_SYMBOLS,
    # Engine 3: Full Taxonomy Alignment
    *_ENGINE_3_SYMBOLS,
    # Engine 4: Impact Measurement
    *_ENGINE_4_SYMBOLS,
    # Engine 5: Benchmark Alignment
    *_ENGINE_5_SYMBOLS,
    # Engine 6: PAI Mandatory
    *_ENGINE_6_SYMBOLS,
    # Engine 7: Carbon Trajectory
    *_ENGINE_7_SYMBOLS,
    # Engine 8: Investment Universe
    *_ENGINE_8_SYMBOLS,
]


def get_loaded_engines() -> list[str]:
    """Return list of engine class names that loaded successfully."""
    return list(_loaded_engines)


def get_engine_count() -> int:
    """Return count of engines that loaded successfully."""
    return len(_loaded_engines)


logger.info(
    "PACK-011 SFDR Article 9 engines: %d/%d loaded",
    len(_loaded_engines),
    __engines_count__,
)
