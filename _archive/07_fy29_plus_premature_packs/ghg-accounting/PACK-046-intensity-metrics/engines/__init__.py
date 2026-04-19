# -*- coding: utf-8 -*-
"""
PACK-046 Intensity Metrics Pack - Engines Module
=====================================================

Calculation engines for comprehensive GHG intensity metrics including
denominator management, intensity calculation, LMDI decomposition,
peer benchmarking, SBTi SDA target pathways, trend analysis, scenario
modelling, uncertainty quantification, disclosure mapping, and
multi-format reporting.

Engines:
    1. DenominatorRegistryEngine       - Denominator registry, validation, and recommendation
    2. IntensityCalculationEngine      - Core intensity = emissions / denominator
    3. DecompositionEngine             - LMDI-I activity/structure/intensity decomposition
    4. BenchmarkingEngine              - Peer benchmarking, percentile ranking, gap analysis
    5. TargetPathwayEngine             - SBTi SDA convergence pathways and target tracking
    6. TrendAnalysisEngine             - YoY, CARR, Mann-Kendall, Sen's slope, projections
    7. ScenarioEngine                  - Scenario modelling with Monte Carlo simulation
    8. UncertaintyEngine               - IPCC Tier 1/2 error propagation, data quality
    9. DisclosureMappingEngine         - Multi-framework disclosure completeness mapping
    10. IntensityReportingEngine       - Report aggregation and multi-format export

Regulatory Basis:
    GHG Protocol Corporate Standard (2004, revised 2015), Chapter 6
    GRI 305-4: GHG emissions intensity
    ESRS E1-6: Gross Scopes 1, 2, 3 and Total GHG emissions
    CDP Climate Change C6.10: Emissions intensities
    SEC Climate Disclosure Rule (2024), Item 1504(c)(1)
    SBTi Corporate Manual (2023) and SDA Tool v1.2
    ISO 14064-1:2018 Clause 5.3.4 (GHG intensity metrics)
    TCFD Recommended Disclosures: Metrics and Targets (b)
    IFRS S2: Climate-related Disclosures
    IPCC 2006/2019 Guidelines Vol 1 Ch 3 (Uncertainties)

Pack Tier: Enterprise (PACK-046)
Author: GreenLang Platform Team
Date: March 2026
Version: 1.0.0
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack__: str = "PACK-046"
__pack_name__: str = "Intensity Metrics Pack"
__engines_count__: int = 10

# Track which engines loaded successfully
_loaded_engines: list[str] = []


# ===================================================================
# Engine 1: Denominator Registry
# ===================================================================
_ENGINE_1_SYMBOLS: list[str] = [
    "DenominatorRegistryEngine",
]

try:
    from .denominator_registry_engine import (
        DenominatorRegistryEngine,
    )
    _loaded_engines.append("DenominatorRegistryEngine")
except ImportError as e:
    logger.debug("Engine 1 (DenominatorRegistryEngine) not available: %s", e)
    _ENGINE_1_SYMBOLS = []


# ===================================================================
# Engine 2: Intensity Calculation
# ===================================================================
_ENGINE_2_SYMBOLS: list[str] = [
    "IntensityCalculationEngine",
]

try:
    from .intensity_calculation_engine import (
        IntensityCalculationEngine,
    )
    _loaded_engines.append("IntensityCalculationEngine")
except ImportError as e:
    logger.debug("Engine 2 (IntensityCalculationEngine) not available: %s", e)
    _ENGINE_2_SYMBOLS = []


# ===================================================================
# Engine 3: Decomposition
# ===================================================================
_ENGINE_3_SYMBOLS: list[str] = [
    "DecompositionEngine",
]

try:
    from .decomposition_engine import (
        DecompositionEngine,
    )
    _loaded_engines.append("DecompositionEngine")
except ImportError as e:
    logger.debug("Engine 3 (DecompositionEngine) not available: %s", e)
    _ENGINE_3_SYMBOLS = []


# ===================================================================
# Engine 4: Benchmarking
# ===================================================================
_ENGINE_4_SYMBOLS: list[str] = [
    "BenchmarkingEngine",
]

try:
    from .benchmarking_engine import (
        BenchmarkingEngine,
    )
    _loaded_engines.append("BenchmarkingEngine")
except ImportError as e:
    logger.debug("Engine 4 (BenchmarkingEngine) not available: %s", e)
    _ENGINE_4_SYMBOLS = []


# ===================================================================
# Engine 5: Target Pathway
# ===================================================================
_ENGINE_5_SYMBOLS: list[str] = [
    "TargetPathwayEngine",
]

try:
    from .target_pathway_engine import (
        TargetPathwayEngine,
    )
    _loaded_engines.append("TargetPathwayEngine")
except ImportError as e:
    logger.debug("Engine 5 (TargetPathwayEngine) not available: %s", e)
    _ENGINE_5_SYMBOLS = []


# ===================================================================
# Engine 6: Trend Analysis
# ===================================================================
_ENGINE_6_SYMBOLS: list[str] = [
    "TrendAnalysisEngine",
]

try:
    from .trend_analysis_engine import (
        TrendAnalysisEngine,
    )
    _loaded_engines.append("TrendAnalysisEngine")
except ImportError as e:
    logger.debug("Engine 6 (TrendAnalysisEngine) not available: %s", e)
    _ENGINE_6_SYMBOLS = []


# ===================================================================
# Engine 7: Scenario
# ===================================================================
_ENGINE_7_SYMBOLS: list[str] = [
    "ScenarioEngine",
]

try:
    from .scenario_engine import (
        ScenarioEngine,
    )
    _loaded_engines.append("ScenarioEngine")
except ImportError as e:
    logger.debug("Engine 7 (ScenarioEngine) not available: %s", e)
    _ENGINE_7_SYMBOLS = []


# ===================================================================
# Engine 8: Uncertainty
# ===================================================================
_ENGINE_8_SYMBOLS: list[str] = [
    "UncertaintyEngine",
]

try:
    from .uncertainty_engine import (
        UncertaintyEngine,
    )
    _loaded_engines.append("UncertaintyEngine")
except ImportError as e:
    logger.debug("Engine 8 (UncertaintyEngine) not available: %s", e)
    _ENGINE_8_SYMBOLS = []


# ===================================================================
# Engine 9: Disclosure Mapping
# ===================================================================
_ENGINE_9_SYMBOLS: list[str] = [
    "DisclosureMappingEngine",
]

try:
    from .disclosure_mapping_engine import (
        DisclosureMappingEngine,
    )
    _loaded_engines.append("DisclosureMappingEngine")
except ImportError as e:
    logger.debug("Engine 9 (DisclosureMappingEngine) not available: %s", e)
    _ENGINE_9_SYMBOLS = []


# ===================================================================
# Engine 10: Intensity Reporting
# ===================================================================
_ENGINE_10_SYMBOLS: list[str] = [
    "IntensityReportingEngine",
]

try:
    from .intensity_reporting_engine import (
        IntensityReportingEngine,
    )
    _loaded_engines.append("IntensityReportingEngine")
except ImportError as e:
    logger.debug("Engine 10 (IntensityReportingEngine) not available: %s", e)
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
    "PACK-046 Intensity Metrics engines: %d/%d loaded",
    len(_loaded_engines),
    __engines_count__,
)
