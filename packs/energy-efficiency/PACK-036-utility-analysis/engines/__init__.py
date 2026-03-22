# -*- coding: utf-8 -*-
"""
PACK-036 Utility Analysis Pack - Engines Module
========================================================

Calculation engines for comprehensive utility bill analysis, rate
optimization, demand management, cost allocation, budget forecasting,
procurement intelligence, benchmarking, regulatory charge optimization,
weather normalization, and utility reporting.

Engines:
    1. UtilityBillParserEngine           - Multi-format bill parsing, error detection
    2. RateStructureAnalyzerEngine       - 500+ tariffs, rate comparison, TOU optimization
    3. DemandAnalysisEngine              - 15-min intervals, load factor, peak shaving, PF
    4. CostAllocationEngine              - Multi-level allocation, tenant invoicing
    5. BudgetForecastingEngine           - Monte Carlo, ARIMA, regression, scenarios
    6. ProcurementIntelligenceEngine     - Market analysis, contract comparison, hedging
    7. UtilityBenchmarkEngine            - EUI, Energy Star, CIBSE TM46, peer comparison
    8. RegulatoryChargeOptimizerEngine   - Grid charges, levies, exemptions, capacity
    9. WeatherNormalizationEngine        - HDD/CDD, change-point models, ASHRAE 14
    10. UtilityReportingEngine           - Multi-format reports, dashboards, KPIs

Regulatory Basis:
    EU Directive 2023/1791 (EED)
    ISO 50001:2018 / ISO 50006:2014
    ASHRAE Guideline 14-2014
    Energy Star Portfolio Manager
    CIBSE TM46
    GHG Protocol Corporate Standard
    EU Taxonomy Regulation 2020/852

Pack Tier: Professional (PACK-036)
Author: GreenLang Platform Team
Date: March 2026
Version: 1.0.0
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack__: str = "PACK-036"
__pack_name__: str = "Utility Analysis Pack"
__engines_count__: int = 10

_loaded_engines: list[str] = []

# ===================================================================
# Engine 1: Utility Bill Parser Engine
# ===================================================================
_ENGINE_1_SYMBOLS: list[str] = [
    "UtilityBillParserEngine",
    "CommodityType",
    "BillStatus",
    "BillErrorType",
    "ReadType",
    "ChargeCategory",
    "ErrorSeverity",
    "BillFormat",
    "MeterReading",
    "BillLineItem",
    "BillError",
    "UtilityBill",
    "ParsedBillResult",
    "BillValidation",
    "ConsumptionProfile",
    "BillSummary",
]

try:
    from .utility_bill_parser_engine import (
        BillError,
        BillErrorType,
        BillFormat,
        BillLineItem,
        BillStatus,
        BillSummary,
        BillValidation,
        ChargeCategory,
        CommodityType,
        ConsumptionProfile,
        ErrorSeverity,
        MeterReading,
        ParsedBillResult,
        ReadType,
        UtilityBill,
        UtilityBillParserEngine,
    )
    _loaded_engines.append("UtilityBillParserEngine")
except ImportError as e:
    logger.debug("Engine 1 (UtilityBillParserEngine) not available: %s", e)
    _ENGINE_1_SYMBOLS = []

# ===================================================================
# Engine 2: Rate Structure Analyzer Engine
# ===================================================================
_ENGINE_2_SYMBOLS: list[str] = [
    "RateStructureAnalyzerEngine",
    "RateType",
    "SeasonType",
    "TOUPeriod",
    "DemandType",
    "RateChangeImpact",
    "OptimizationStatus",
    "RateTier",
    "TOUSchedule",
    "DemandCharge",
    "RateStructure",
    "MonthlyConsumption",
    "RateComparison",
    "RateOptimizationResult",
    "AnnualCostProjection",
    "TariffChangeImpact",
]

try:
    from .rate_structure_analyzer_engine import (
        AnnualCostProjection,
        DemandCharge,
        DemandType,
        MonthlyConsumption,
        OptimizationStatus,
        RateChangeImpact,
        RateComparison,
        RateOptimizationResult,
        RateStructure,
        RateStructureAnalyzerEngine,
        RateTier,
        RateType,
        SeasonType,
        TariffChangeImpact,
        TOUPeriod,
        TOUSchedule,
    )
    _loaded_engines.append("RateStructureAnalyzerEngine")
except ImportError as e:
    logger.debug("Engine 2 (RateStructureAnalyzerEngine) not available: %s", e)
    _ENGINE_2_SYMBOLS = []

# ===================================================================
# Engine 3: Demand Analysis Engine
# ===================================================================
_ENGINE_3_SYMBOLS: list[str] = [
    "DemandAnalysisEngine",
    "LoadCategory",
    "DemandPeriod",
    "DemandResponseType",
    "PeakType",
    "IntervalResolution",
    "LoadShiftStrategy",
    "IntervalData",
    "DemandProfile",
    "LoadFactor",
    "LoadDurationCurve",
    "PeakEvent",
    "DemandResponseOpportunity",
    "PeakShavingAnalysis",
    "PowerFactorAnalysis",
    "DemandForecast",
]

try:
    from .demand_analysis_engine import (
        DemandAnalysisEngine,
        DemandForecast,
        DemandPeriod,
        DemandProfile,
        DemandResponseOpportunity,
        DemandResponseType,
        IntervalData,
        IntervalResolution,
        LoadCategory,
        LoadDurationCurve,
        LoadFactor,
        LoadShiftStrategy,
        PeakEvent,
        PeakShavingAnalysis,
        PeakType,
        PowerFactorAnalysis,
    )
    _loaded_engines.append("DemandAnalysisEngine")
except ImportError as e:
    logger.debug("Engine 3 (DemandAnalysisEngine) not available: %s", e)
    _ENGINE_3_SYMBOLS = []

# ===================================================================
# Engine 4: Cost Allocation Engine
# ===================================================================
_ENGINE_4_SYMBOLS: list[str] = [
    "CostAllocationEngine",
    "AllocationMethod",
    "CostComponent",
    "TenantType",
    "ReconciliationStatus",
    "AllocationFrequency",
    "DemandAllocationMethod",
    "AllocationEntity",
    "AllocationRule",
    "CostPool",
    "SubMeterData",
    "AllocationLineItem",
    "AllocationResult",
    "TenantInvoice",
    "ReconciliationReport",
    "FairnessMetrics",
]

try:
    from .cost_allocation_engine import (
        AllocationEntity,
        AllocationFrequency,
        AllocationLineItem,
        AllocationMethod,
        AllocationResult,
        AllocationRule,
        CostAllocationEngine,
        CostComponent,
        CostPool,
        DemandAllocationMethod,
        FairnessMetrics,
        ReconciliationReport,
        ReconciliationStatus,
        SubMeterData,
        TenantInvoice,
        TenantType,
    )
    _loaded_engines.append("CostAllocationEngine")
except ImportError as e:
    logger.debug("Engine 4 (CostAllocationEngine) not available: %s", e)
    _ENGINE_4_SYMBOLS = []

# ===================================================================
# Engine 5: Budget Forecasting Engine
# ===================================================================
_ENGINE_5_SYMBOLS: list[str] = [
    "BudgetForecastingEngine",
    "ForecastMethod",
    "ForecastHorizon",
    "ConfidenceLevel",
    "VarianceCategory",
    "ScenarioType",
    "TrendDirection",
    "HistoricalDataPoint",
    "ForecastInput",
    "MonthlyForecast",
    "ConfidenceBand",
    "BudgetVariance",
    "VarianceDecomposition",
    "ScenarioResult",
    "RollingForecast",
    "ForecastResult",
]

try:
    from .budget_forecasting_engine import (
        BudgetForecastingEngine,
        BudgetVariance,
        ConfidenceBand,
        ConfidenceLevel,
        ForecastHorizon,
        ForecastInput,
        ForecastMethod,
        ForecastResult,
        HistoricalDataPoint,
        MonthlyForecast,
        RollingForecast,
        ScenarioResult,
        ScenarioType,
        TrendDirection,
        VarianceCategory,
        VarianceDecomposition,
    )
    _loaded_engines.append("BudgetForecastingEngine")
except ImportError as e:
    logger.debug("Engine 5 (BudgetForecastingEngine) not available: %s", e)
    _ENGINE_5_SYMBOLS = []

# ===================================================================
# Engine 6: Procurement Intelligence Engine
# ===================================================================
_ENGINE_6_SYMBOLS: list[str] = [
    "ProcurementIntelligenceEngine",
    "ContractType",
    "MarketIndex",
    "ProcurementStrategy",
    "RiskMetric",
    "GreenProduct",
    "SupplierRating",
    "MarketCondition",
    "MarketPrice",
    "ForwardCurve",
    "ContractTerms",
    "ContractComparison",
    "ProcurementPlan",
    "PriceRiskAssessment",
    "LoadWeightedPrice",
    "GreenProcurement",
    "SupplierEvaluation",
    "ProcurementResult",
]

try:
    from .procurement_intelligence_engine import (
        ContractComparison,
        ContractTerms,
        ContractType,
        ForwardCurve,
        GreenProcurement,
        GreenProduct,
        LoadWeightedPrice,
        MarketCondition,
        MarketIndex,
        MarketPrice,
        PriceRiskAssessment,
        ProcurementIntelligenceEngine,
        ProcurementPlan,
        ProcurementResult,
        ProcurementStrategy,
        RiskMetric,
        SupplierEvaluation,
        SupplierRating,
    )
    _loaded_engines.append("ProcurementIntelligenceEngine")
except ImportError as e:
    logger.debug("Engine 6 (ProcurementIntelligenceEngine) not available: %s", e)
    _ENGINE_6_SYMBOLS = []

# ===================================================================
# Engine 7: Utility Benchmark Engine
# ===================================================================
_ENGINE_7_SYMBOLS: list[str] = [
    "UtilityBenchmarkEngine",
    "BenchmarkStandard",
    "EUIUnit",
    "BuildingType",
    "PerformanceQuartile",
    "BenchmarkScope",
    "EnergyConsumption",
    "FacilityMetrics",
    "EUICalculation",
    "BenchmarkTarget",
    "EnergyStarScore",
    "PeerComparison",
    "PortfolioRanking",
    "TrendAnalysis",
    "NormalizationFactor",
    "BenchmarkResult",
]

try:
    from .utility_benchmark_engine import (
        BenchmarkResult,
        BenchmarkScope,
        BenchmarkStandard,
        BenchmarkTarget,
        BuildingType,
        EnergyConsumption,
        EnergyStarScore,
        EUICalculation,
        EUIUnit,
        FacilityMetrics,
        NormalizationFactor,
        PeerComparison,
        PerformanceQuartile,
        PortfolioRanking,
        TrendAnalysis,
        UtilityBenchmarkEngine,
    )
    _loaded_engines.append("UtilityBenchmarkEngine")
except ImportError as e:
    logger.debug("Engine 7 (UtilityBenchmarkEngine) not available: %s", e)
    _ENGINE_7_SYMBOLS = []

# ===================================================================
# Engine 8: Regulatory Charge Optimizer Engine
# ===================================================================
_ENGINE_8_SYMBOLS: list[str] = [
    "RegulatoryChargeOptimizerEngine",
    "ChargeType",
    "Jurisdiction",
    "ExemptionType",
    "VoltageLevel",
    "OptimizationAction",
    "ChargeMethodology",
    "RegulatoryCharge",
    "ChargeBreakdown",
    "ExemptionAssessment",
    "CapacityOptimization",
    "PowerFactorOptimization",
    "VoltageLevelAnalysis",
    "GridChargeProjection",
    "SelfGenerationImpact",
    "ChargeOptimizationResult",
]

try:
    from .regulatory_charge_optimizer_engine import (
        CapacityOptimization,
        ChargeBreakdown,
        ChargeMethodology,
        ChargeOptimizationResult,
        ChargeType,
        ExemptionAssessment,
        ExemptionType,
        GridChargeProjection,
        Jurisdiction,
        OptimizationAction,
        PowerFactorOptimization,
        RegulatoryCharge,
        RegulatoryChargeOptimizerEngine,
        SelfGenerationImpact,
        VoltageLevelAnalysis,
        VoltageLevel,
    )
    _loaded_engines.append("RegulatoryChargeOptimizerEngine")
except ImportError as e:
    logger.debug("Engine 8 (RegulatoryChargeOptimizerEngine) not available: %s", e)
    _ENGINE_8_SYMBOLS = []

# ===================================================================
# Engine 9: Weather Normalization Engine
# ===================================================================
_ENGINE_9_SYMBOLS: list[str] = [
    "WeatherNormalizationEngine",
    "RegressionModel",
    "WeatherSource",
    "NormalizationType",
    "ClimateScenario",
    "ValidationStatus",
    "TemperatureUnit",
    "ModelFit",
    "WeatherStation",
    "DailyWeather",
    "DegreeDays",
    "MonthlyConsumptionWeather",
    "RegressionCoefficients",
    "ChangePointModel",
    "ModelValidation",
    "NormalizationResult",
    "WeatherImpact",
    "ClimateProjection",
    "WeatherAnalysisResult",
]

try:
    from .weather_normalization_engine import (
        ChangePointModel,
        ClimateProjection,
        ClimateScenario,
        DailyWeather,
        DegreeDays,
        ModelFit,
        ModelValidation,
        MonthlyConsumptionWeather,
        NormalizationResult,
        NormalizationType,
        RegressionCoefficients,
        RegressionModel,
        TemperatureUnit,
        ValidationStatus,
        WeatherAnalysisResult,
        WeatherImpact,
        WeatherNormalizationEngine,
        WeatherSource,
        WeatherStation,
    )
    _loaded_engines.append("WeatherNormalizationEngine")
except ImportError as e:
    logger.debug("Engine 9 (WeatherNormalizationEngine) not available: %s", e)
    _ENGINE_9_SYMBOLS = []

# ===================================================================
# Engine 10: Utility Reporting Engine
# ===================================================================
_ENGINE_10_SYMBOLS: list[str] = [
    "UtilityReportingEngine",
    "ReportType",
    "ReportFormat",
    "RAGStatus",
    "KPICategory",
    "AnomalyType",
    "WidgetType",
    "UtilityKPI",
    "MonthlyUtilitySummary",
    "PortfolioSummary",
    "VarianceExplanation",
    "TrendData",
    "AnomalyFlag",
    "DashboardWidget",
    "ExecutiveInsight",
    "ReportConfig",
    "ReportOutput",
]

try:
    from .utility_reporting_engine import (
        AnomalyFlag,
        AnomalyType,
        DashboardWidget,
        ExecutiveInsight,
        KPICategory,
        MonthlyUtilitySummary,
        PortfolioSummary,
        RAGStatus,
        ReportConfig,
        ReportFormat,
        ReportOutput,
        ReportType,
        TrendData,
        UtilityKPI,
        UtilityReportingEngine,
        VarianceExplanation,
        WidgetType,
    )
    _loaded_engines.append("UtilityReportingEngine")
except ImportError as e:
    logger.debug("Engine 10 (UtilityReportingEngine) not available: %s", e)
    _ENGINE_10_SYMBOLS = []

# ===================================================================
# Module Exports
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
    """
    Get list of successfully loaded engine classes.

    Returns:
        List of loaded engine class names
    """
    return list(_loaded_engines)


def get_engine_count() -> int:
    """
    Get count of successfully loaded engines.

    Returns:
        Number of loaded engines
    """
    return len(_loaded_engines)


logger.info(
    "PACK-036 Utility Analysis engines: %d/%d loaded",
    len(_loaded_engines),
    __engines_count__,
)
