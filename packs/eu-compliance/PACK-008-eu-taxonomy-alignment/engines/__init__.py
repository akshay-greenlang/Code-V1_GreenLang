# -*- coding: utf-8 -*-
"""
PACK-008 EU Taxonomy Alignment Pack - Engines
================================================

10 calculation engines for EU Taxonomy Alignment Pack,
covering activity eligibility screening through taxonomy
reporting and disclosure generation per EU Taxonomy
Regulation (EU) 2020/852.

Engines:
    TaxonomyEligibilityEngine        -- Activity-level eligibility screening per 6 objectives
    SubstantialContributionEngine    -- Substantial contribution threshold assessment
    DNSHAssessmentEngine             -- Do No Significant Harm 6-objective assessment
    MinimumSafeguardsEngine          -- OECD/UNGP/ILO minimum safeguards evaluation
    KPICalculationEngine             -- Turnover/CapEx/OpEx KPI calculation
    GreenAssetRatioEngine            -- EBA Pillar 3 GAR/BTAR calculation
    TechnicalScreeningCriteriaEngine -- Delegated Act TSC evaluation
    TransitionActivityEngine         -- Article 10(2) transition activity assessment
    EnablingActivityEngine           -- Article 16 enabling activity assessment
    TaxonomyReportingEngine          -- Article 8 disclosure and XBRL generation

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-008 EU Taxonomy Alignment Pack
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack__: str = "PACK-008"
__pack_name__: str = "EU Taxonomy Alignment Pack"
__engines_count__: int = 10

_loaded_engines: list[str] = []

# --- Engine 1: Taxonomy Eligibility ------------------------------------------
_engine_1_symbols: list[str] = []
try:
    from .eligibility_engine import (  # noqa: F401
        TaxonomyEligibilityEngine,
        EligibilityResult,
        PortfolioEligibility,
        TaxonomyActivity,
        EnvironmentalObjective as EligibilityObjective,
        TaxonomySector,
        EligibilityStatus,
    )
    _engine_1_symbols = [
        "TaxonomyEligibilityEngine",
        "EligibilityResult",
        "PortfolioEligibility",
        "TaxonomyActivity",
        "EligibilityObjective",
        "TaxonomySector",
        "EligibilityStatus",
    ]
    logger.debug("Engine 1 (TaxonomyEligibilityEngine) loaded successfully")
except ImportError as exc:
    logger.debug("Engine 1 (TaxonomyEligibilityEngine) not available: %s", exc)

# --- Engine 2: Substantial Contribution --------------------------------------
_engine_2_symbols: list[str] = []
try:
    from .substantial_contribution_engine import (  # noqa: F401
        SubstantialContributionEngine,
        SCResult,
        ThresholdResult,
        TSCThreshold,
        SCStatus,
        ActivityClassification,
    )
    _engine_2_symbols = [
        "SubstantialContributionEngine",
        "SCResult",
        "ThresholdResult",
        "TSCThreshold",
        "SCStatus",
        "ActivityClassification",
    ]
    logger.debug("Engine 2 (SubstantialContributionEngine) loaded successfully")
except ImportError as exc:
    logger.debug("Engine 2 (SubstantialContributionEngine) not available: %s", exc)

# --- Engine 3: DNSH Assessment -----------------------------------------------
_engine_3_symbols: list[str] = []
try:
    from .dnsh_engine import (  # noqa: F401
        DNSHAssessmentEngine,
        DNSHResult,
        ObjectiveDNSHResult,
        DNSHCriterion,
        DNSHStatus,
    )
    _engine_3_symbols = [
        "DNSHAssessmentEngine",
        "DNSHResult",
        "ObjectiveDNSHResult",
        "DNSHCriterion",
        "DNSHStatus",
    ]
    logger.debug("Engine 3 (DNSHAssessmentEngine) loaded successfully")
except ImportError as exc:
    logger.debug("Engine 3 (DNSHAssessmentEngine) not available: %s", exc)

# --- Engine 4: Minimum Safeguards --------------------------------------------
_engine_4_symbols: list[str] = []
try:
    from .minimum_safeguards_engine import (  # noqa: F401
        MinimumSafeguardsEngine,
        MSResult,
        TopicResult,
        SafeguardTopic,
        TopicStatus,
    )
    _engine_4_symbols = [
        "MinimumSafeguardsEngine",
        "MSResult",
        "TopicResult",
        "SafeguardTopic",
        "TopicStatus",
    ]
    logger.debug("Engine 4 (MinimumSafeguardsEngine) loaded successfully")
except ImportError as exc:
    logger.debug("Engine 4 (MinimumSafeguardsEngine) not available: %s", exc)

# --- Engine 5: KPI Calculation ------------------------------------------------
_engine_5_symbols: list[str] = []
try:
    from .kpi_calculation_engine import (  # noqa: F401
        KPICalculationEngine,
        KPIResult,
        KPIType,
        CapExPlanStatus,
        YoYComparison,
        ObjectiveBreakdown,
    )
    _engine_5_symbols = [
        "KPICalculationEngine",
        "KPIResult",
        "KPIType",
        "CapExPlanStatus",
        "YoYComparison",
        "ObjectiveBreakdown",
    ]
    logger.debug("Engine 5 (KPICalculationEngine) loaded successfully")
except ImportError as exc:
    logger.debug("Engine 5 (KPICalculationEngine) not available: %s", exc)

# --- Engine 6: Green Asset Ratio (GAR) ----------------------------------------
_engine_6_symbols: list[str] = []
try:
    from .gar_engine import (  # noqa: F401
        GreenAssetRatioEngine,
        GARResult,
        BTARResult,
        ExposureClassification,
        Exposure,
        ExposureType,
        EPCRating,
        CounterpartyType,
        GARConfig,
    )
    _engine_6_symbols = [
        "GreenAssetRatioEngine",
        "GARResult",
        "BTARResult",
        "ExposureClassification",
        "Exposure",
        "ExposureType",
        "EPCRating",
        "CounterpartyType",
        "GARConfig",
    ]
    logger.debug("Engine 6 (GreenAssetRatioEngine) loaded successfully")
except ImportError as exc:
    logger.debug("Engine 6 (GreenAssetRatioEngine) not available: %s", exc)

# --- Engine 7: Technical Screening Criteria -----------------------------------
_engine_7_symbols: list[str] = []
try:
    from .tsc_engine import (  # noqa: F401
        TechnicalScreeningCriteriaEngine,
        TSCCriteria,
        TSCEvaluation,
        CriterionEvaluation,
        DelegatedActVersion,
        DelegatedActId,
        EnvironmentalObjective as TSCObjective,
        CriterionType,
        ComparisonOperator,
    )
    _engine_7_symbols = [
        "TechnicalScreeningCriteriaEngine",
        "TSCCriteria",
        "TSCEvaluation",
        "CriterionEvaluation",
        "DelegatedActVersion",
        "DelegatedActId",
        "TSCObjective",
        "CriterionType",
        "ComparisonOperator",
    ]
    logger.debug("Engine 7 (TechnicalScreeningCriteriaEngine) loaded successfully")
except ImportError as exc:
    logger.debug("Engine 7 (TechnicalScreeningCriteriaEngine) not available: %s", exc)

# --- Engine 8: Transition Activity --------------------------------------------
_engine_8_symbols: list[str] = []
try:
    from .transition_activity_engine import (  # noqa: F401
        TransitionActivityEngine,
        BATResult,
        LockInResult,
        TransitionPathway,
        TransitionStatus,
        LockInRisk,
        BATBenchmark,
        TransitionActivityInfo,
    )
    _engine_8_symbols = [
        "TransitionActivityEngine",
        "BATResult",
        "LockInResult",
        "TransitionPathway",
        "TransitionStatus",
        "LockInRisk",
        "BATBenchmark",
        "TransitionActivityInfo",
    ]
    logger.debug("Engine 8 (TransitionActivityEngine) loaded successfully")
except ImportError as exc:
    logger.debug("Engine 8 (TransitionActivityEngine) not available: %s", exc)

# --- Engine 9: Enabling Activity ----------------------------------------------
_engine_9_symbols: list[str] = []
try:
    from .enabling_activity_engine import (  # noqa: F401
        EnablingActivityEngine,
        EnablementResult,
        LifecycleResult,
        EnablementType,
        LifecycleImpact,
        MarketDistortionRisk,
        EnablingActivityInfo,
        EnabledActivity,
    )
    _engine_9_symbols = [
        "EnablingActivityEngine",
        "EnablementResult",
        "LifecycleResult",
        "EnablementType",
        "LifecycleImpact",
        "MarketDistortionRisk",
        "EnablingActivityInfo",
        "EnabledActivity",
    ]
    logger.debug("Engine 9 (EnablingActivityEngine) loaded successfully")
except ImportError as exc:
    logger.debug("Engine 9 (EnablingActivityEngine) not available: %s", exc)

# --- Engine 10: Taxonomy Reporting --------------------------------------------
_engine_10_symbols: list[str] = []
try:
    from .taxonomy_reporting_engine import (  # noqa: F401
        TaxonomyReportingEngine,
        Article8Report,
        EBATemplates,
        EBATemplate,
        XBRLOutput,
        XBRLTag,
        DisclosureTable,
        TableRow,
        ActivityKPIData,
        AlignmentData,
        GARInputData,
        PriorPeriodData,
        DisclosureType,
        EBATemplateId,
        KPIType as ReportingKPIType,
    )
    _engine_10_symbols = [
        "TaxonomyReportingEngine",
        "Article8Report",
        "EBATemplates",
        "EBATemplate",
        "XBRLOutput",
        "XBRLTag",
        "DisclosureTable",
        "TableRow",
        "ActivityKPIData",
        "AlignmentData",
        "GARInputData",
        "PriorPeriodData",
        "DisclosureType",
        "EBATemplateId",
        "ReportingKPIType",
    ]
    logger.debug("Engine 10 (TaxonomyReportingEngine) loaded successfully")
except ImportError as exc:
    logger.debug("Engine 10 (TaxonomyReportingEngine) not available: %s", exc)

# --- Dynamic __all__ ---------------------------------------------------------

if _engine_1_symbols:
    _loaded_engines.append("TaxonomyEligibilityEngine")
if _engine_2_symbols:
    _loaded_engines.append("SubstantialContributionEngine")
if _engine_3_symbols:
    _loaded_engines.append("DNSHAssessmentEngine")
if _engine_4_symbols:
    _loaded_engines.append("MinimumSafeguardsEngine")
if _engine_5_symbols:
    _loaded_engines.append("KPICalculationEngine")
if _engine_6_symbols:
    _loaded_engines.append("GreenAssetRatioEngine")
if _engine_7_symbols:
    _loaded_engines.append("TechnicalScreeningCriteriaEngine")
if _engine_8_symbols:
    _loaded_engines.append("TransitionActivityEngine")
if _engine_9_symbols:
    _loaded_engines.append("EnablingActivityEngine")
if _engine_10_symbols:
    _loaded_engines.append("TaxonomyReportingEngine")

__all__: list[str] = (
    _engine_1_symbols
    + _engine_2_symbols
    + _engine_3_symbols
    + _engine_4_symbols
    + _engine_5_symbols
    + _engine_6_symbols
    + _engine_7_symbols
    + _engine_8_symbols
    + _engine_9_symbols
    + _engine_10_symbols
)


def get_loaded_engines() -> list[str]:
    """Return names of successfully loaded engines."""
    return list(_loaded_engines)


def get_engine_count() -> int:
    """Return total number of expected engines."""
    return __engines_count__


def get_loaded_engine_count() -> int:
    """Return number of successfully loaded engines."""
    return len(_loaded_engines)


logger.info(
    "PACK-008 engines: %d / %d loaded",
    get_loaded_engine_count(),
    get_engine_count(),
)
