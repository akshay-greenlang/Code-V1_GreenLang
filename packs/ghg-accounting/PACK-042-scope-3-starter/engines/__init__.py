# -*- coding: utf-8 -*-
"""
PACK-042 Scope 3 Starter Pack - Engines Module
=================================================

Calculation engines for screening-level Scope 3 GHG inventory
management including category screening, spend classification,
consolidation, double-counting prevention, hotspot analysis,
data quality scoring, supplier engagement, reporting, uncertainty,
and SBTi target alignment.

Engines:
    1. Scope3ScreeningEngine              - GHG Protocol Ch 6 rapid screening
    2. SpendClassificationEngine          - Deterministic spend classification
    3. CategoryConsolidationEngine        - Consolidate all 15 Scope 3 categories
    4. DoubleCountingPreventionEngine     - Detect/resolve double-counting
    5. HotspotAnalysisEngine              - Identify hotspots, prioritise reductions
    6. DataQualityScoringEngine           - Score and improve data quality (TBD)
    7. SupplierEngagementEngine           - Supplier engagement for Scope 3 (TBD)
    8. Scope3ReportingEngine              - Generate Scope 3 reports (TBD)
    9. UncertaintyQuantificationEngine    - Scope 3 uncertainty analysis (TBD)
    10. SBTiScope3AlignmentEngine         - SBTi Scope 3 target alignment (TBD)

Regulatory Basis:
    GHG Protocol Corporate Value Chain (Scope 3) Standard (2011)
    GHG Protocol Technical Guidance for Calculating Scope 3 Emissions (2013)
    GHG Protocol Scope 3 Evaluator Tool methodology
    Exiobase 3 MRIO database (emission intensity factors)
    ISO 14064-1:2018 (Specification for GHG inventories)
    ESRS E1 (Delegated Act 2023/2772 - Climate Change)
    SBTi Corporate Net-Zero Standard (2021)
    SBTi Supplier Engagement Guidance (2023)
    CDP Climate Change Questionnaire (2024)
    PCAF Global GHG Accounting Standard (Category 15)

Pack Tier: Professional (PACK-042)
Author: GreenLang Platform Team
Date: March 2026
Version: 1.0.0
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack__: str = "PACK-042"
__pack_name__: str = "Scope 3 Starter Pack"
__engines_count__: int = 10

# Track which engines loaded successfully
_loaded_engines: list[str] = []


# ===================================================================
# Engine 1: Scope 3 Screening
# ===================================================================
_ENGINE_1_SYMBOLS: list[str] = [
    "Scope3ScreeningEngine",
    "OrgProfile",
    "ScreeningResult",
    "CategoryScreening",
    "RelevanceScore",
    "Scope3Category",
    "RelevanceTier",
    "DataAvailabilityLevel",
    "ScreeningStatus",
    "EEIO_INTENSITIES",
    "NAICS_SECTOR_PROFILES",
    "DOWNSTREAM_INTENSITY_BY_SECTOR",
    "RELEVANCE_WEIGHTS",
    "DEFAULT_SIGNIFICANCE_THRESHOLD_PCT",
    "CATEGORY_DESCRIPTIONS",
]

try:
    from .scope3_screening_engine import (
        CATEGORY_DESCRIPTIONS,
        DEFAULT_SIGNIFICANCE_THRESHOLD_PCT,
        DOWNSTREAM_INTENSITY_BY_SECTOR,
        DataAvailabilityLevel,
        EEIO_INTENSITIES,
        NAICS_SECTOR_PROFILES,
        OrgProfile,
        RELEVANCE_WEIGHTS,
        RelevanceScore,
        RelevanceTier,
        Scope3Category as Scope3Category_Screening,
        Scope3ScreeningEngine,
        CategoryScreening,
        ScreeningResult,
        ScreeningStatus,
    )
    _loaded_engines.append("Scope3ScreeningEngine")
except ImportError as e:
    logger.debug("Engine 1 (Scope3ScreeningEngine) not available: %s", e)
    _ENGINE_1_SYMBOLS = []


# ===================================================================
# Engine 2: Spend Classification
# ===================================================================
_ENGINE_2_SYMBOLS: list[str] = [
    "SpendClassificationEngine",
    "SpendTransaction",
    "ClassifiedTransaction",
    "ClassificationResult",
    "CategorySpendSummary",
    "SplitAllocation",
    "ClassificationConfidence",
    "ClassificationMethod",
    "ClassificationStatus",
    "NAICS_TO_SCOPE3",
    "ISIC_TO_SCOPE3",
    "UNSPSC_TO_SCOPE3",
    "GL_ACCOUNT_RANGES",
    "KEYWORD_TO_SCOPE3",
    "CURRENCY_TO_EUR",
    "CPI_INDICES",
]

try:
    from .spend_classification_engine import (
        CategorySpendSummary,
        ClassificationConfidence,
        ClassificationMethod,
        ClassificationResult,
        ClassificationStatus,
        ClassifiedTransaction,
        CPI_INDICES,
        CURRENCY_TO_EUR,
        GL_ACCOUNT_RANGES,
        ISIC_TO_SCOPE3,
        KEYWORD_TO_SCOPE3,
        NAICS_TO_SCOPE3,
        SpendClassificationEngine,
        SpendTransaction,
        SplitAllocation,
        UNSPSC_TO_SCOPE3,
    )
    _loaded_engines.append("SpendClassificationEngine")
except ImportError as e:
    logger.debug("Engine 2 (SpendClassificationEngine) not available: %s", e)
    _ENGINE_2_SYMBOLS = []


# ===================================================================
# Engine 3: Category Consolidation
# ===================================================================
_ENGINE_3_SYMBOLS: list[str] = [
    "CategoryConsolidationEngine",
    "CategoryResult",
    "ConsolidatedInventory",
    "ScopeSummary",
    "GasTotal",
    "CategoryConsolidated",
    "MethodologyTierSummary",
    "YoYComparison",
    "GasBreakdown",
    "BoundaryConfig",
    "BaseYearData",
    "GasType",
    "MethodologyTier",
    "DataQualityRating",
    "ConsolidationStatus",
    "BoundaryApproach",
    "UPSTREAM_CATEGORIES",
    "DOWNSTREAM_CATEGORIES",
    "CATEGORY_NAMES",
]

try:
    from .category_consolidation_engine import (
        BaseYearData,
        BoundaryApproach,
        BoundaryConfig,
        CATEGORY_NAMES,
        CategoryConsolidated,
        CategoryConsolidationEngine,
        CategoryResult,
        ConsolidatedInventory,
        ConsolidationStatus as ConsolidationStatus_Cat,
        DataQualityRating,
        DOWNSTREAM_CATEGORIES,
        GasBreakdown,
        GasTotal,
        GasType,
        MethodologyTier,
        MethodologyTierSummary,
        ScopeSummary,
        UPSTREAM_CATEGORIES,
        YoYComparison,
    )
    _loaded_engines.append("CategoryConsolidationEngine")
except ImportError as e:
    logger.debug("Engine 3 (CategoryConsolidationEngine) not available: %s", e)
    _ENGINE_3_SYMBOLS = []


# ===================================================================
# Engine 4: Double-Counting Prevention
# ===================================================================
_ENGINE_4_SYMBOLS: list[str] = [
    "DoubleCountingPreventionEngine",
    "CategoryEmissionInput",
    "OverlapDetection",
    "OverlapResolution",
    "AdjustedResult",
    "DoubleCountingResult",
    "OverlapRuleId",
    "AllocationMethod",
    "OverlapSeverity",
    "ResolutionStatus",
    "OVERLAP_RULES",
]

try:
    from .double_counting_engine import (
        AdjustedResult,
        AllocationMethod,
        CategoryEmissionInput,
        DoubleCountingPreventionEngine,
        DoubleCountingResult,
        OVERLAP_RULES,
        OverlapDetection,
        OverlapResolution,
        OverlapRuleId,
        OverlapSeverity,
        ResolutionStatus,
    )
    _loaded_engines.append("DoubleCountingPreventionEngine")
except ImportError as e:
    logger.debug("Engine 4 (DoubleCountingPreventionEngine) not available: %s", e)
    _ENGINE_4_SYMBOLS = []


# ===================================================================
# Engine 5: Hotspot Analysis
# ===================================================================
_ENGINE_5_SYMBOLS: list[str] = [
    "HotspotAnalysisEngine",
    "HotspotAnalysisInput",
    "HotspotResult",
    "ParetoItem",
    "SupplierConcentration",
    "MaterialityScore",
    "SectorBenchmarkResult",
    "GeographicDistribution",
    "ReductionOpportunity",
    "TierUpgradeImpact",
    "CategoryInput",
    "SupplierData",
    "ProductData",
    "HotspotType",
    "ReductionLever",
    "ImprovementDifficulty",
    "AnalysisStatus",
    "SECTOR_BENCHMARKS",
    "REDUCTION_POTENTIALS",
    "TIER_UPGRADE_RATIOS",
]

try:
    from .hotspot_analysis_engine import (
        AnalysisStatus,
        CategoryInput,
        GeographicDistribution,
        HotspotAnalysisEngine,
        HotspotAnalysisInput,
        HotspotResult,
        HotspotType,
        ImprovementDifficulty,
        MaterialityScore,
        ParetoItem,
        ProductData,
        REDUCTION_POTENTIALS,
        ReductionLever,
        ReductionOpportunity,
        SECTOR_BENCHMARKS,
        SectorBenchmarkResult,
        SupplierConcentration,
        SupplierData,
        TIER_UPGRADE_RATIOS,
        TierUpgradeImpact,
    )
    _loaded_engines.append("HotspotAnalysisEngine")
except ImportError as e:
    logger.debug("Engine 5 (HotspotAnalysisEngine) not available: %s", e)
    _ENGINE_5_SYMBOLS = []


# ===================================================================
# Engine 6: Supplier Engagement
# ===================================================================
_ENGINE_6_SYMBOLS: list[str] = [
    "SupplierEngagementEngine",
    "Supplier",
    "ProcurementItem",
    "SupplierResponseData",
    "SupplierPriority",
    "DataRequest",
    "QualityScore",
    "EngagementPlan",
    "EngagementROI",
    "ReminderSchedule",
    "EngagementMetrics",
    "SupplierEngagementResult",
    "DataQualityLevel",
    "EngagementStatus",
    "SupplierTier",
    "IndustryType",
    "ReminderType",
    "QUESTIONNAIRE_TEMPLATES",
    "CDP_SUPPLY_CHAIN_FIELDS",
    "DQI_UNCERTAINTY_RANGES",
]

try:
    from .supplier_engagement_engine import (
        CDP_SUPPLY_CHAIN_FIELDS,
        DQI_UNCERTAINTY_RANGES,
        DataQualityLevel,
        DataRequest,
        EngagementMetrics,
        EngagementPlan,
        EngagementROI,
        EngagementStatus,
        IndustryType,
        ProcurementItem,
        QUESTIONNAIRE_TEMPLATES,
        QualityScore,
        ReminderSchedule,
        ReminderType,
        Supplier,
        SupplierEngagementEngine,
        SupplierEngagementResult,
        SupplierPriority,
        SupplierResponseData,
        SupplierTier,
    )
    _loaded_engines.append("SupplierEngagementEngine")
except ImportError as e:
    logger.debug("Engine 6 (SupplierEngagementEngine) not available: %s", e)
    _ENGINE_6_SYMBOLS = []


# ===================================================================
# Engine 7: Data Quality Assessment
# ===================================================================
_ENGINE_7_SYMBOLS: list[str] = [
    "DataQualityAssessmentEngine",
    "DataSourceInfo",
    "CategoryDataInput",
    "HistoricalQualityRecord",
    "QualityIndicator",
    "QualityAssessment",
    "DQRScore",
    "ImprovementAction",
    "QualityTrend",
    "DataQualityResult",
    "ImprovementPriority",
    "DEFAULT_DQI_WEIGHTS",
    "FRAMEWORK_THRESHOLDS",
]

try:
    from .data_quality_engine import (
        CategoryDataInput,
        DQRScore,
        DEFAULT_DQI_WEIGHTS,
        DataQualityAssessmentEngine,
        DataQualityResult,
        DataSourceInfo,
        FRAMEWORK_THRESHOLDS,
        HistoricalQualityRecord,
        ImprovementAction,
        ImprovementPriority,
        QualityAssessment,
        QualityIndicator,
        QualityTrend,
    )
    _loaded_engines.append("DataQualityAssessmentEngine")
except ImportError as e:
    logger.debug("Engine 7 (DataQualityAssessmentEngine) not available: %s", e)
    _ENGINE_7_SYMBOLS = []


# ===================================================================
# Engine 8: Scope 3 Uncertainty
# ===================================================================
_ENGINE_8_SYMBOLS: list[str] = [
    "Scope3UncertaintyEngine",
    "CategoryUncertaintyInput",
    "CorrelationInput",
    "MonteCarloConfig",
    "AnalyticalResult",
    "MonteCarloResult",
    "SensitivityItem",
    "TierUpgradeImpact",
    "UncertaintyResult",
    "DistributionType",
    "DEFAULT_UNCERTAINTY_BY_TIER",
    "DEFAULT_CORRELATIONS",
]

try:
    from .scope3_uncertainty_engine import (
        AnalyticalResult,
        CategoryUncertaintyInput,
        CorrelationInput,
        DEFAULT_CORRELATIONS,
        DEFAULT_UNCERTAINTY_BY_TIER,
        DistributionType,
        MonteCarloConfig,
        MonteCarloResult,
        Scope3UncertaintyEngine,
        SensitivityItem,
        TierUpgradeImpact,
        UncertaintyResult,
    )
    _loaded_engines.append("Scope3UncertaintyEngine")
except ImportError as e:
    logger.debug("Engine 8 (Scope3UncertaintyEngine) not available: %s", e)
    _ENGINE_8_SYMBOLS = []


# ===================================================================
# Engine 9: Scope 3 Compliance
# ===================================================================
_ENGINE_9_SYMBOLS: list[str] = [
    "Scope3ComplianceEngine",
    "Scope3InventoryData",
    "RequirementCheck",
    "FrameworkResult",
    "ComplianceGap",
    "ActionItem",
    "ComplianceAssessment",
    "FrameworkType",
    "RequirementStatus",
    "ComplianceClassification",
    "GapPriority",
    "REQUIREMENT_DATABASE",
    "FRAMEWORK_WEIGHTS",
]

try:
    from .scope3_compliance_engine import (
        ActionItem,
        ComplianceAssessment,
        ComplianceClassification,
        ComplianceGap,
        FRAMEWORK_WEIGHTS,
        FrameworkResult,
        FrameworkType,
        GapPriority,
        REQUIREMENT_DATABASE,
        RequirementCheck,
        RequirementStatus,
        Scope3ComplianceEngine,
        Scope3InventoryData,
    )
    _loaded_engines.append("Scope3ComplianceEngine")
except ImportError as e:
    logger.debug("Engine 9 (Scope3ComplianceEngine) not available: %s", e)
    _ENGINE_9_SYMBOLS = []


# ===================================================================
# Engine 10: Scope 3 Reporting
# ===================================================================
_ENGINE_10_SYMBOLS: list[str] = [
    "Scope3ReportingEngine",
    "OrganizationInfo",
    "ReportConfig",
    "Scope3ReportData",
    "ReportSection",
    "AppendixItem",
    "ReportOutput",
    "Scope3ReportType",
    "OutputFormat",
]

try:
    from .scope3_reporting_engine import (
        AppendixItem,
        OrganizationInfo,
        OutputFormat,
        ReportConfig,
        ReportOutput,
        ReportSection,
        Scope3ReportData,
        Scope3ReportType,
        Scope3ReportingEngine,
    )
    _loaded_engines.append("Scope3ReportingEngine")
except ImportError as e:
    logger.debug("Engine 10 (Scope3ReportingEngine) not available: %s", e)
    _ENGINE_10_SYMBOLS = []


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
    *_ENGINE_1_SYMBOLS,
    *_ENGINE_2_SYMBOLS,
    *_ENGINE_3_SYMBOLS,
    *_ENGINE_4_SYMBOLS,
    *_ENGINE_5_SYMBOLS,
    *_ENGINE_6_SYMBOLS,
    *_ENGINE_7_SYMBOLS,
    *_ENGINE_8_SYMBOLS,
    *_ENGINE_9_SYMBOLS,
    *_ENGINE_10_SYMBOLS,
    "get_loaded_engines",
    "get_engine_count",
]


def get_loaded_engines() -> list[str]:
    """Return list of engine class names that loaded successfully.

    Returns:
        List of engine class name strings.

    Example:
        >>> engines = get_loaded_engines()
        >>> print(engines)
        ['Scope3ScreeningEngine', 'SpendClassificationEngine', ...]
    """
    return list(_loaded_engines)


def get_engine_count() -> int:
    """Return count of engines that loaded successfully.

    Returns:
        Integer count of loaded engines.

    Example:
        >>> count = get_engine_count()
        >>> print(count)
        5
    """
    return len(_loaded_engines)


logger.info(
    "PACK-042 Scope 3 Starter engines: %d/%d loaded",
    len(_loaded_engines),
    __engines_count__,
)
