# -*- coding: utf-8 -*-
"""
PACK-035 Energy Benchmark Pack - Engines Module
===================================================

Calculation engines for energy benchmarking, peer comparison, weather
normalisation, portfolio analysis, and regulatory performance rating
per EED, EPBD, ISO 50001/50006, ASHRAE 100, and ENERGY STAR.

Engines:
    1. EUICalculatorEngine                  - Site/source/primary EUI, accounting boundaries
    2. PeerComparisonEngine                 - ENERGY STAR, CIBSE TM46, DIN V 18599, BPIE
    3. SectorBenchmarkEngine                - Sector-specific benchmarks (office, retail, etc.)
    4. WeatherNormalisationEngine           - Degree-day regression 2P-5P, TMY normalisation
    5. EnergyPerformanceGapEngine           - End-use disaggregated gap analysis
    6. PortfolioBenchmarkEngine             - Multi-facility portfolio ranking and analysis
    7. RegressionAnalysisEngine             - TOWT, change-point, multivariate regression
    8. PerformanceRatingEngine              - EPC A-G, ENERGY STAR 1-100, NABERS, CRREM
    9. TrendAnalysisEngine                  - CUSUM, SPC, time-series trend decomposition
    10. BenchmarkReportEngine               - Report generation with charts and tables

Regulatory Basis:
    EU Directive 2023/1791 (EED - Energy Efficiency Directive)
    EU Directive 2024/1275 (EPBD - Energy Performance of Buildings)
    ISO 50001:2018 (Energy Management Systems)
    ISO 50006:2014 (Energy Performance Indicators)
    ASHRAE Standard 100-2018 (Building Energy Benchmarking)
    ENERGY STAR Portfolio Manager Technical Reference
    CIBSE TM46:2008 (UK Building Energy Benchmarks)
    GHG Protocol Corporate Standard (Carbon Intensity Benchmarking)

Pack Tier: Professional (PACK-035)
Author: GreenLang Platform Team
Date: March 2026
Version: 1.0.0
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack__: str = "PACK-035"
__pack_name__: str = "Energy Benchmark Pack"
__engines_count__: int = 10

# Track which engines loaded successfully
_loaded_engines: list[str] = []


# ===================================================================
# Engine 1: EUI Calculator
# ===================================================================
_ENGINE_1_SYMBOLS: list[str] = [
    "EUICalculatorEngine",
    "FacilityProfile",
    "EnergyMeterData",
    "EUIResult",
    "NormalisedEUI",
    "RollingEUIPoint",
    "EUICalculationResult",
    "EnergyCarrier",
    "FloorAreaType",
    "EUIAccountingBoundary",
    "CalculationPeriod",
    "SOURCE_ENERGY_FACTORS",
    "PRIMARY_ENERGY_FACTORS",
]

try:
    from .eui_calculator_engine import (
        CalculationPeriod,
        EUIAccountingBoundary,
        EUICalculationResult,
        EUICalculatorEngine,
        EUIResult,
        EnergyCarrier,
        EnergyMeterData,
        FacilityProfile,
        FloorAreaType,
        NormalisedEUI,
        PRIMARY_ENERGY_FACTORS,
        RollingEUIPoint,
        SOURCE_ENERGY_FACTORS,
    )
    _loaded_engines.append("EUICalculatorEngine")
except ImportError as e:
    logger.debug("Engine 1 (EUICalculatorEngine) not available: %s", e)
    _ENGINE_1_SYMBOLS = []


# ===================================================================
# Engine 2: Peer Comparison
# ===================================================================
_ENGINE_2_SYMBOLS: list[str] = [
    "PeerComparisonEngine",
    "PeerGroup",
    "PeerDistribution",
    "ComparisonResult",
    "PercentileRanking",
    "PeerGroupType",
    "QuartileBand",
    "ComparisonMethod",
]

try:
    from .peer_comparison_engine import (
        ComparisonMethod,
        ComparisonResult,
        PeerComparisonEngine,
        PeerDistribution,
        PeerGroup,
        PeerGroupType,
        PercentileRanking,
        QuartileBand,
    )
    _loaded_engines.append("PeerComparisonEngine")
except ImportError as e:
    logger.debug("Engine 2 (PeerComparisonEngine) not available: %s", e)
    _ENGINE_2_SYMBOLS = []


# ===================================================================
# Engine 3: Sector Benchmark
# ===================================================================
_ENGINE_3_SYMBOLS: list[str] = [
    "SectorBenchmarkEngine",
    "BenchmarkRecord",
    "SectorBenchmarkResult",
    "BenchmarkComparison",
    "BenchmarkSource",
    "BuildingType",
    "BenchmarkLevel",
    "CIBSE_TM46_BENCHMARKS",
    "ENERGY_STAR_MEDIANS",
    "DIN_V_18599_REFERENCE",
]

try:
    from .sector_benchmark_engine import (
        BenchmarkComparison,
        BenchmarkLevel,
        BenchmarkRecord,
        BenchmarkSource,
        BuildingType,
        CIBSE_TM46_BENCHMARKS,
        DIN_V_18599_REFERENCE,
        ENERGY_STAR_MEDIANS,
        SectorBenchmarkEngine,
        SectorBenchmarkResult,
    )
    _loaded_engines.append("SectorBenchmarkEngine")
except ImportError as e:
    logger.debug("Engine 3 (SectorBenchmarkEngine) not available: %s", e)
    _ENGINE_3_SYMBOLS = []


# ===================================================================
# Engine 4: Weather Normalisation
# ===================================================================
_ENGINE_4_SYMBOLS: list[str] = [
    "WeatherNormalisationEngine",
    "WeatherStation",
    "DegreeDayData",
    "RegressionModel",
    "NormalisationResult",
    "ModelValidation",
    "RegressionModelType",
    "NormalisationMethod",
    "ValidationStatus",
    "DEGREE_DAY_BASE_TEMPS",
    "ASHRAE_14_THRESHOLDS",
]

try:
    from .weather_normalisation_engine import (
        ASHRAE_14_THRESHOLDS,
        DEGREE_DAY_BASE_TEMPS,
        DegreeDayData,
        ModelValidation,
        NormalisationMethod,
        NormalisationResult,
        RegressionModel,
        RegressionModelType,
        ValidationStatus,
        WeatherNormalisationEngine,
        WeatherStation,
    )
    _loaded_engines.append("WeatherNormalisationEngine")
except ImportError as e:
    logger.debug("Engine 4 (WeatherNormalisationEngine) not available: %s", e)
    _ENGINE_4_SYMBOLS = []


# ===================================================================
# Engine 5: Energy Performance Gap
# ===================================================================
_ENGINE_5_SYMBOLS: list[str] = [
    "EnergyPerformanceGapEngine",
    "EndUseBreakdown",
    "GapAnalysisResult",
    "ImprovementPriority",
    "EndUseGap",
    "EndUseCategory",
    "DisaggregationMethod",
    "GapSeverity",
    "TYPICAL_END_USE_SPLITS",
    "END_USE_IMPROVEMENT_POTENTIAL",
]

try:
    from .energy_performance_gap_engine import (
        DisaggregationMethod,
        END_USE_IMPROVEMENT_POTENTIAL,
        EndUseBreakdown,
        EndUseCategory,
        EndUseGap,
        EnergyPerformanceGapEngine,
        GapAnalysisResult,
        GapSeverity,
        ImprovementPriority,
        TYPICAL_END_USE_SPLITS,
    )
    _loaded_engines.append("EnergyPerformanceGapEngine")
except ImportError as e:
    logger.debug("Engine 5 (EnergyPerformanceGapEngine) not available: %s", e)
    _ENGINE_5_SYMBOLS = []


# ===================================================================
# Engine 6: Portfolio Benchmark
# ===================================================================
_ENGINE_6_SYMBOLS: list[str] = [
    "PortfolioBenchmarkEngine",
    "PortfolioFacility",
    "PortfolioSummary",
    "FacilityRanking",
    "PortfolioTrend",
    "PortfolioBenchmarkResult",
    "RankingMetric",
    "AggregationMethod",
    "PortfolioTier",
    "OutlierDetectionMethod",
]

try:
    from .portfolio_benchmark_engine import (
        AggregationMethod,
        FacilityRanking,
        OutlierDetectionMethod,
        PortfolioBenchmarkEngine,
        PortfolioBenchmarkResult,
        PortfolioFacility,
        PortfolioSummary,
        PortfolioTier,
        PortfolioTrend,
        RankingMetric,
    )
    _loaded_engines.append("PortfolioBenchmarkEngine")
except ImportError as e:
    logger.debug("Engine 6 (PortfolioBenchmarkEngine) not available: %s", e)
    _ENGINE_6_SYMBOLS = []


# ===================================================================
# Engine 7: Regression Analysis
# ===================================================================
_ENGINE_7_SYMBOLS: list[str] = [
    "RegressionAnalysisEngine",
    "RegressionInput",
    "RegressionOutput",
    "ModelDiagnostics",
    "VariableImportance",
    "RegressionAnalysisResult",
    "ModelType",
    "VariableType",
    "GoodnessOfFit",
    "ResidualAnalysis",
]

try:
    from .regression_analysis_engine import (
        GoodnessOfFit,
        ModelDiagnostics,
        ModelType,
        RegressionAnalysisEngine,
        RegressionAnalysisResult,
        RegressionInput,
        RegressionOutput,
        ResidualAnalysis,
        VariableImportance,
        VariableType,
    )
    _loaded_engines.append("RegressionAnalysisEngine")
except ImportError as e:
    logger.debug("Engine 7 (RegressionAnalysisEngine) not available: %s", e)
    _ENGINE_7_SYMBOLS = []


# ===================================================================
# Engine 8: Performance Rating
# ===================================================================
_ENGINE_8_SYMBOLS: list[str] = [
    "PerformanceRatingEngine",
    "EPCRating",
    "EnergyStarScore",
    "NABERSRating",
    "CRREMPathway",
    "PerformanceRatingResult",
    "EPCClass",
    "RatingScheme",
    "CRREMScenario",
    "MEPSCompliance",
    "EPC_THRESHOLDS",
]

try:
    from .performance_rating_engine import (
        CRREMPathway,
        CRREMScenario,
        EPC_THRESHOLDS,
        EPCClass,
        EPCRating,
        EnergyStarScore,
        MEPSCompliance,
        NABERSRating,
        PerformanceRatingEngine,
        PerformanceRatingResult,
        RatingScheme,
    )
    _loaded_engines.append("PerformanceRatingEngine")
except ImportError as e:
    logger.debug("Engine 8 (PerformanceRatingEngine) not available: %s", e)
    _ENGINE_8_SYMBOLS = []


# ===================================================================
# Engine 9: Trend Analysis
# ===================================================================
_ENGINE_9_SYMBOLS: list[str] = [
    "TrendAnalysisEngine",
    "CUSUMResult",
    "SPCChart",
    "TrendDecomposition",
    "SeasonalPattern",
    "TrendAnalysisResult",
    "TrendDirection",
    "SPCRuleViolation",
    "DecompositionMethod",
    "AlertThreshold",
]

try:
    from .trend_analysis_engine import (
        AlertThreshold,
        CUSUMResult,
        DecompositionMethod,
        SPCChart,
        SPCRuleViolation,
        SeasonalPattern,
        TrendAnalysisEngine,
        TrendAnalysisResult,
        TrendDecomposition,
        TrendDirection,
    )
    _loaded_engines.append("TrendAnalysisEngine")
except ImportError as e:
    logger.debug("Engine 9 (TrendAnalysisEngine) not available: %s", e)
    _ENGINE_9_SYMBOLS = []


# ===================================================================
# Engine 10: Benchmark Report
# ===================================================================
_ENGINE_10_SYMBOLS: list[str] = [
    "BenchmarkReportEngine",
    "ReportSection",
    "ChartData",
    "TableData",
    "ReportMetadata",
    "BenchmarkReportResult",
    "ReportFormat",
    "ChartType",
    "ExportTarget",
    "ReportTemplate",
]

try:
    from .benchmark_report_engine import (
        BenchmarkReportEngine,
        BenchmarkReportResult,
        ChartData,
        ChartType,
        ExportTarget,
        ReportFormat,
        ReportMetadata,
        ReportSection,
        ReportTemplate,
        TableData,
    )
    _loaded_engines.append("BenchmarkReportEngine")
except ImportError as e:
    logger.debug("Engine 10 (BenchmarkReportEngine) not available: %s", e)
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
    "PACK-035 Energy Benchmark engines: %d/%d loaded",
    len(_loaded_engines),
    __engines_count__,
)
