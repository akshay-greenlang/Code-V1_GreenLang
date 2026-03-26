# -*- coding: utf-8 -*-
"""
PACK-041 Scope 1-2 Complete Pack - Engines Module
=====================================================

Calculation engines for comprehensive Scope 1 and Scope 2 GHG inventory
management including consolidation, emission factor management, gas-level
accounting, Scope 2 dual reporting, reconciliation, uncertainty aggregation,
base year recalculation, trend analysis, compliance mapping, and reporting.

Engines:
    1. OrganizationalBoundaryEngine     - GHG Protocol Ch 3 consolidation approaches
    2. SourceCompletenessEngine         - Source completeness and materiality assessment
    3. EmissionFactorManagerEngine      - Centralised emission factor management
    4. Scope1ConsolidationEngine        - Aggregate 8 MRV agents into Scope 1 total
    5. Scope2ConsolidationEngine        - Scope 2 dual reporting (location + market)
    6. UncertaintyAggregationEngine     - IPCC Approach 1/2 uncertainty aggregation
    7. BaseYearRecalculationEngine      - GHG Protocol Chapter 5 base year recalculation
    8. TrendAnalysisEngine              - YoY trends, Kaya decomposition, SBTi alignment
    9. ComplianceMappingEngine          - Map to GHG Protocol, ESRS E1, CDP, ISO, SBTi, SEC, SB 253
    10. InventoryReportingEngine        - Generate reports and verification packages

Regulatory Basis:
    GHG Protocol Corporate Standard (2004, revised 2015)
    GHG Protocol Scope 2 Guidance (2015)
    IPCC 2006 Guidelines for National GHG Inventories
    ISO 14064-1:2018 (Specification for GHG inventories)
    ESRS E1 (Delegated Act 2023/2772 - Climate Change)
    SBTi Corporate Manual (2023) and Criteria v5.1
    SEC Climate Disclosure Rule (Final Rule 33-11275)
    California SB 253 (Climate Corporate Data Accountability Act)

Pack Tier: Enterprise (PACK-041)
Author: GreenLang Platform Team
Date: March 2026
Version: 1.0.0
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack__: str = "PACK-041"
__pack_name__: str = "Scope 1-2 Complete Pack"
__engines_count__: int = 10

# Track which engines loaded successfully
_loaded_engines: list[str] = []


# ===================================================================
# Engine 1: Organizational Boundary
# ===================================================================
_ENGINE_1_SYMBOLS: list[str] = [
    "OrganizationalBoundaryEngine",
]

try:
    from .organizational_boundary_engine import (
        OrganizationalBoundaryEngine,
    )
    _loaded_engines.append("OrganizationalBoundaryEngine")
except ImportError as e:
    logger.debug("Engine 1 (OrganizationalBoundaryEngine) not available: %s", e)
    _ENGINE_1_SYMBOLS = []


# ===================================================================
# Engine 2: Source Completeness
# ===================================================================
_ENGINE_2_SYMBOLS: list[str] = [
    "SourceCompletenessEngine",
]

try:
    from .source_completeness_engine import (
        SourceCompletenessEngine,
    )
    _loaded_engines.append("SourceCompletenessEngine")
except ImportError as e:
    logger.debug("Engine 2 (SourceCompletenessEngine) not available: %s", e)
    _ENGINE_2_SYMBOLS = []


# ===================================================================
# Engine 3: Emission Factor Manager
# ===================================================================
_ENGINE_3_SYMBOLS: list[str] = [
    "EmissionFactorManagerEngine",
]

try:
    from .emission_factor_manager_engine import (
        EmissionFactorManagerEngine,
    )
    _loaded_engines.append("EmissionFactorManagerEngine")
except ImportError as e:
    logger.debug("Engine 3 (EmissionFactorManagerEngine) not available: %s", e)
    _ENGINE_3_SYMBOLS = []


# ===================================================================
# Engine 4: Scope 1 Consolidation
# ===================================================================
_ENGINE_4_SYMBOLS: list[str] = [
    "Scope1ConsolidationEngine",
]

try:
    from .scope1_consolidation_engine import (
        Scope1ConsolidationEngine,
    )
    _loaded_engines.append("Scope1ConsolidationEngine")
except ImportError as e:
    logger.debug("Engine 4 (Scope1ConsolidationEngine) not available: %s", e)
    _ENGINE_4_SYMBOLS = []


# ===================================================================
# Engine 5: Scope 2 Consolidation
# ===================================================================
_ENGINE_5_SYMBOLS: list[str] = [
    "Scope2ConsolidationEngine",
]

try:
    from .scope2_consolidation_engine import (
        Scope2ConsolidationEngine,
    )
    _loaded_engines.append("Scope2ConsolidationEngine")
except ImportError as e:
    logger.debug("Engine 5 (Scope2ConsolidationEngine) not available: %s", e)
    _ENGINE_5_SYMBOLS = []


# ===================================================================
# Engine 6: Uncertainty Aggregation
# ===================================================================
_ENGINE_6_SYMBOLS: list[str] = [
    "UncertaintyAggregationEngine",
    "SourceUncertainty",
    "MonteCarloConfig",
    "AnalyticalResult",
    "MonteCarloResult",
    "UncertaintyContributor",
    "DataQualityAssessment",
    "UncertaintyAggregationResult",
    "UncertaintyDistribution",
    "SourceCategoryType",
    "DataQualityTier",
    "AggregationMethod",
    "DEFAULT_UNCERTAINTY_RANGES",
    "DATA_QUALITY_MULTIPLIERS",
    "IMPROVEMENT_RECOMMENDATIONS",
]

try:
    from .uncertainty_aggregation_engine import (
        AggregationMethod,
        AnalyticalResult,
        DATA_QUALITY_MULTIPLIERS,
        DEFAULT_UNCERTAINTY_RANGES,
        DataQualityAssessment,
        DataQualityTier,
        IMPROVEMENT_RECOMMENDATIONS,
        MonteCarloConfig,
        MonteCarloResult,
        SourceCategoryType,
        SourceUncertainty,
        UncertaintyAggregationEngine,
        UncertaintyAggregationResult,
        UncertaintyContributor,
        UncertaintyDistribution,
    )
    _loaded_engines.append("UncertaintyAggregationEngine")
except ImportError as e:
    logger.debug("Engine 6 (UncertaintyAggregationEngine) not available: %s", e)
    _ENGINE_6_SYMBOLS = []


# ===================================================================
# Engine 7: Base Year Recalculation
# ===================================================================
_ENGINE_7_SYMBOLS: list[str] = [
    "BaseYearRecalculationEngine",
    "BaseYearData",
    "RecalculationTrigger",
    "RecalculationTriggerType",
    "RecalculationStatus",
    "AdjustmentType",
    "SignificanceAssessment",
    "RecalculationAdjustment",
    "AuditEntry",
    "BaseYearRecalculationResult",
    "DEFAULT_SIGNIFICANCE_THRESHOLD_PCT",
    "SBTI_SIGNIFICANCE_THRESHOLD_PCT",
]

try:
    from .base_year_recalculation_engine import (
        AdjustmentType,
        AuditEntry,
        BaseYearData,
        BaseYearRecalculationEngine,
        BaseYearRecalculationResult,
        DEFAULT_SIGNIFICANCE_THRESHOLD_PCT,
        RecalculationAdjustment,
        RecalculationStatus,
        RecalculationTrigger,
        RecalculationTriggerType,
        SBTI_SIGNIFICANCE_THRESHOLD_PCT,
        SignificanceAssessment,
    )
    _loaded_engines.append("BaseYearRecalculationEngine")
except ImportError as e:
    logger.debug("Engine 7 (BaseYearRecalculationEngine) not available: %s", e)
    _ENGINE_7_SYMBOLS = []


# ===================================================================
# Engine 8: Trend Analysis
# ===================================================================
_ENGINE_8_SYMBOLS: list[str] = [
    "TrendAnalysisEngine",
    "YearlyEmissions",
    "SBTiTarget",
    "YearOverYearChange",
    "IntensityMetric",
    "DecompositionResult",
    "KayaResult",
    "SBTiAlignment",
    "BaseYearComparison",
    "TrendAnalysisResult",
    "IntensityMetricType",
    "DecompositionFactor",
    "TrendDirection",
    "SBTiAmbitionLevel",
    "SBTI_ABSOLUTE_RATES",
    "INTENSITY_UNITS",
]

try:
    from .trend_analysis_engine import (
        BaseYearComparison,
        DecompositionFactor,
        DecompositionResult,
        INTENSITY_UNITS,
        IntensityMetric,
        IntensityMetricType,
        KayaResult,
        SBTI_ABSOLUTE_RATES,
        SBTiAlignment,
        SBTiAmbitionLevel,
        SBTiTarget,
        TrendAnalysisEngine,
        TrendAnalysisResult,
        TrendDirection,
        YearOverYearChange,
        YearlyEmissions,
    )
    _loaded_engines.append("TrendAnalysisEngine")
except ImportError as e:
    logger.debug("Engine 8 (TrendAnalysisEngine) not available: %s", e)
    _ENGINE_8_SYMBOLS = []


# ===================================================================
# Engine 9: Compliance Mapping
# ===================================================================
_ENGINE_9_SYMBOLS: list[str] = [
    "ComplianceMappingEngine",
    "InventoryData",
    "ComplianceRequirement",
    "RequirementResult",
    "FrameworkComplianceResult",
    "CriticalGap",
    "ComplianceMappingResult",
    "FrameworkType",
    "RequirementStatus",
    "ComplianceClassification",
    "GapPriority",
    "REQUIREMENT_DATABASE",
    "FRAMEWORK_WEIGHTS",
]

try:
    from .compliance_mapping_engine import (
        ComplianceClassification,
        ComplianceMappingEngine,
        ComplianceMappingResult,
        ComplianceRequirement,
        CriticalGap,
        FRAMEWORK_WEIGHTS,
        FrameworkComplianceResult,
        FrameworkType,
        GapPriority,
        InventoryData,
        REQUIREMENT_DATABASE,
        RequirementResult,
        RequirementStatus,
    )
    _loaded_engines.append("ComplianceMappingEngine")
except ImportError as e:
    logger.debug("Engine 9 (ComplianceMappingEngine) not available: %s", e)
    _ENGINE_9_SYMBOLS = []


# ===================================================================
# Engine 10: Inventory Reporting
# ===================================================================
_ENGINE_10_SYMBOLS: list[str] = [
    "InventoryReportingEngine",
    "OrganizationInfo",
    "ReportSection",
    "InventoryReportInput",
    "ReportMetadata",
    "InventoryReportOutput",
    "ReportType",
    "OutputFormat",
]

try:
    from .inventory_reporting_engine import (
        InventoryReportInput,
        InventoryReportOutput,
        InventoryReportingEngine,
        OrganizationInfo,
        OutputFormat,
        ReportMetadata,
        ReportSection,
        ReportType,
    )
    _loaded_engines.append("InventoryReportingEngine")
except ImportError as e:
    logger.debug("Engine 10 (InventoryReportingEngine) not available: %s", e)
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
]


def get_loaded_engines() -> list[str]:
    """Return list of engine class names that loaded successfully."""
    return list(_loaded_engines)


def get_engine_count() -> int:
    """Return count of engines that loaded successfully."""
    return len(_loaded_engines)


logger.info(
    "PACK-041 Scope 1-2 Complete engines: %d/%d loaded",
    len(_loaded_engines),
    __engines_count__,
)
