# -*- coding: utf-8 -*-
"""
PACK-034 ISO 50001 Energy Management System Pack - Engines Module
====================================================================

Calculation engines for ISO 50001:2018 Energy Management System
implementation covering Significant Energy Use (SEU) analysis,
Energy Baseline (EnB) establishment, Energy Performance Indicator
(EnPI) calculation, CUSUM monitoring, degree-day normalization,
energy balance, action planning, compliance checking, performance
trending, and management review.

Engines:
    1. SEUAnalyzerEngine              - Clause 6.3 SEU identification & Pareto analysis
    2. EnergyBaselineEngine           - ISO 50006 EnB regression & normalization
    3. EnPICalculatorEngine           - ISO 50006 EnPI types & statistical validation
    4. CUSUMMonitorEngine             - Cumulative Sum control charts & alerts
    5. DegreeDayEngine                - HDD/CDD calculation & change-point models
    6. EnergyBalanceEngine            - Facility balance, Sankey, sub-metering
    7. ActionPlanEngine               - Clause 6.2 objectives, targets, SMART validation
    8. ComplianceCheckerEngine        - Clauses 4-10 gap analysis & certification readiness
    9. PerformanceTrendEngine         - YoY comparison, regression validation, ISO 50015
    10. ManagementReviewEngine        - Clause 9.3 review inputs/outputs & KPI dashboard

Regulatory Basis:
    ISO 50001:2018 (Energy Management Systems)
    ISO 50006:2014 (EnPI and EnB methodology)
    ISO 50015:2014 (M&V of energy performance)
    EU Directive 2023/1791 (EED - Energy Efficiency Directive)
    IPMVP (International Performance Measurement & Verification Protocol)
    ASHRAE Guideline 14-2014 (Measurement of Energy, Demand, and Water Savings)

Pack Tier: Professional (PACK-034)
Author: GreenLang Platform Team
Date: March 2026
Version: 1.0.0
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack__: str = "PACK-034"
__pack_name__: str = "ISO 50001 Energy Management System Pack"
__engines_count__: int = 10

# Track which engines loaded successfully
_loaded_engines: list[str] = []


# ===================================================================
# Engine 1: SEU Analyzer
# ===================================================================
_ENGINE_1_SYMBOLS: list[str] = [
    "SEUAnalyzerEngine",
    "EnergyConsumer",
    "SEUThresholds",
    "EnergyDriver",
    "FacilityEnergyProfile",
    "SEUResult",
    "SEUAnalysisResult",
    "SEUCategory",
    "OperatingPattern",
    "SEUStatus",
    "DeterminationMethod",
    "LoadType",
]

try:
    from .seu_analyzer_engine import (
        DeterminationMethod,
        EnergyConsumer,
        EnergyDriver,
        FacilityEnergyProfile,
        LoadType,
        OperatingPattern,
        SEUAnalysisResult,
        SEUAnalyzerEngine,
        SEUCategory,
        SEUResult,
        SEUStatus,
        SEUThresholds,
    )
    _loaded_engines.append("SEUAnalyzerEngine")
except ImportError as e:
    logger.debug("Engine 1 (SEUAnalyzerEngine) not available: %s", e)
    _ENGINE_1_SYMBOLS = []


# ===================================================================
# Engine 2: Energy Baseline
# ===================================================================
_ENGINE_2_SYMBOLS: list[str] = [
    "EnergyBaselineEngine",
    "BaselineDataPoint",
    "RegressionCoefficient",
    "RegressionModel",
    "StaticFactor",
    "RelevantVariable",
    "BaselineConfig",
    "BaselineResult",
    "BaselineAdjustment",
    "BaselineModelType",
    "BaselineStatus",
    "VariableType",
    "AdjustmentTrigger",
    "DataGranularity",
]

try:
    from .energy_baseline_engine import (
        AdjustmentTrigger,
        BaselineAdjustment,
        BaselineConfig,
        BaselineDataPoint,
        BaselineModelType,
        BaselineResult,
        BaselineStatus,
        DataGranularity,
        EnergyBaselineEngine,
        RegressionCoefficient,
        RegressionModel,
        RelevantVariable,
        StaticFactor,
        VariableType,
    )
    _loaded_engines.append("EnergyBaselineEngine")
except ImportError as e:
    logger.debug("Engine 2 (EnergyBaselineEngine) not available: %s", e)
    _ENGINE_2_SYMBOLS = []


# ===================================================================
# Engine 3: EnPI Calculator
# ===================================================================
_ENGINE_3_SYMBOLS: list[str] = [
    "EnPICalculatorEngine",
    "EnPIMeasurement",
    "EnPIDefinition",
    "EnPIValue",
    "StatisticalValidation",
    "EnPIResult",
    "PortfolioEnPIResult",
    "EnPIType",
    "NormalizationMethod",
    "ImprovementDirection",
    "AggregationLevel",
    "StatisticalTest",
]

try:
    from .enpi_calculator_engine import (
        AggregationLevel,
        EnPICalculatorEngine,
        EnPIDefinition,
        EnPIMeasurement,
        EnPIResult,
        EnPIType,
        EnPIValue,
        ImprovementDirection,
        NormalizationMethod,
        PortfolioEnPIResult,
        StatisticalTest,
        StatisticalValidation,
    )
    _loaded_engines.append("EnPICalculatorEngine")
except ImportError as e:
    logger.debug("Engine 3 (EnPICalculatorEngine) not available: %s", e)
    _ENGINE_3_SYMBOLS = []


# ===================================================================
# Engine 4: CUSUM Monitor
# ===================================================================
_ENGINE_4_SYMBOLS: list[str] = [
    "CUSUMMonitorEngine",
    "CUSUMConfig",
    "CUSUMDataPoint",
    "CUSUMAlert",
    "VMaskParameters",
    "CUSUMMonitorResult",
    "CUSUMSummary",
    "MonitoringInterval",
    "AlertType",
    "CUSUMMethod",
    "MonitorStatus",
    "SeasonalAdjustment",
]

try:
    from .cusum_monitor_engine import (
        AlertType,
        CUSUMAlert,
        CUSUMConfig,
        CUSUMDataPoint,
        CUSUMMethod,
        CUSUMMonitorEngine,
        CUSUMMonitorResult,
        CUSUMSummary,
        MonitoringInterval,
        MonitorStatus,
        SeasonalAdjustment,
        VMaskParameters,
    )
    _loaded_engines.append("CUSUMMonitorEngine")
except ImportError as e:
    logger.debug("Engine 4 (CUSUMMonitorEngine) not available: %s", e)
    _ENGINE_4_SYMBOLS = []


# ===================================================================
# Engine 5: Degree Day
# ===================================================================
_ENGINE_5_SYMBOLS: list[str] = [
    "DegreeDayEngine",
    "DailyTemperature",
    "DegreeDayResult",
    "MonthlyDegreeDays",
    "ChangePointModelResult",
    "WeatherNormalization",
    "BaseTemperatureOptimization",
    "DegreeDayAnalysisResult",
    "DegreeDayType",
    "ChangePointModel",
    "TemperatureUnit",
    "BaseTemperatureMethod",
    "NormalizationBasis",
]

try:
    from .degree_day_engine import (
        BaseTemperatureMethod,
        BaseTemperatureOptimization,
        ChangePointModel,
        ChangePointModelResult,
        DailyTemperature,
        DegreeDayAnalysisResult,
        DegreeDayEngine,
        DegreeDayResult,
        DegreeDayType,
        MonthlyDegreeDays,
        NormalizationBasis,
        TemperatureUnit,
        WeatherNormalization,
    )
    _loaded_engines.append("DegreeDayEngine")
except ImportError as e:
    logger.debug("Engine 5 (DegreeDayEngine) not available: %s", e)
    _ENGINE_5_SYMBOLS = []


# ===================================================================
# Engine 6: Energy Balance
# ===================================================================
_ENGINE_6_SYMBOLS: list[str] = [
    "EnergyBalanceEngine",
    "EnergyFlow",
    "MeterNode",
    "SankeyNode",
    "SankeyLink",
    "SankeyDiagram",
    "MeterReconciliation",
    "LossEstimate",
    "EnergyBalanceResult",
    "EnergyFlowType",
    "EnergySource",
    "EndUseCategory",
    "MeterType",
    "ReconciliationStatus",
]

try:
    from .energy_balance_engine import (
        EndUseCategory,
        EnergyBalanceEngine,
        EnergyBalanceResult,
        EnergyFlow,
        EnergyFlowType,
        EnergySource,
        LossEstimate,
        MeterNode,
        MeterReconciliation,
        MeterType,
        ReconciliationStatus,
        SankeyDiagram,
        SankeyLink,
        SankeyNode,
    )
    _loaded_engines.append("EnergyBalanceEngine")
except ImportError as e:
    logger.debug("Engine 6 (EnergyBalanceEngine) not available: %s", e)
    _ENGINE_6_SYMBOLS = []


# ===================================================================
# Engine 7: Action Plan
# ===================================================================
_ENGINE_7_SYMBOLS: list[str] = [
    "ActionPlanEngine",
    "SMARTObjective",
    "EnergyObjective",
    "EnergyTarget",
    "ResourceRequirement",
    "ActionItem",
    "ActionPlan",
    "ActionPlanPortfolio",
    "ObjectiveType",
    "TargetScope",
    "ActionStatus",
    "ActionPriority",
    "ResourceType",
    "VerificationMethod",
]

try:
    from .action_plan_engine import (
        ActionItem,
        ActionPlan,
        ActionPlanEngine,
        ActionPlanPortfolio,
        ActionPriority,
        ActionStatus,
        EnergyObjective,
        EnergyTarget,
        ObjectiveType,
        ResourceRequirement,
        ResourceType,
        SMARTObjective,
        TargetScope,
        VerificationMethod,
    )
    _loaded_engines.append("ActionPlanEngine")
except ImportError as e:
    logger.debug("Engine 7 (ActionPlanEngine) not available: %s", e)
    _ENGINE_7_SYMBOLS = []


# ===================================================================
# Engine 8: Compliance Checker
# ===================================================================
_ENGINE_8_SYMBOLS: list[str] = [
    "ComplianceCheckerEngine",
    "ISO50001Clause",
    "ClauseAssessment",
    "DocumentChecklist",
    "Nonconformity",
    "AuditFinding",
    "ComplianceScore",
    "ComplianceResult",
    "ClauseStatus",
    "AssessmentType",
    "NonconformitySeverity",
    "CorrectionStatus",
    "CertificationReadiness",
    "DocumentType",
]

try:
    from .compliance_checker_engine import (
        AssessmentType,
        AuditFinding,
        CertificationReadiness,
        ClauseAssessment,
        ClauseStatus,
        ComplianceCheckerEngine,
        ComplianceResult,
        ComplianceScore,
        CorrectionStatus,
        DocumentChecklist,
        DocumentType,
        ISO50001Clause,
        Nonconformity,
        NonconformitySeverity,
    )
    _loaded_engines.append("ComplianceCheckerEngine")
except ImportError as e:
    logger.debug("Engine 8 (ComplianceCheckerEngine) not available: %s", e)
    _ENGINE_8_SYMBOLS = []


# ===================================================================
# Engine 9: Performance Trend
# ===================================================================
_ENGINE_9_SYMBOLS: list[str] = [
    "PerformanceTrendEngine",
    "PerformanceDataPoint",
    "TrendAnalysis",
    "YearOverYearComparison",
    "RollingAnalysis",
    "RegressionValidation",
    "ForecastResult",
    "SavingsVerification",
    "PerformanceTrendResult",
    "TrendDirection",
    "AnalysisType",
    "RegressionMetric",
    "ForecastMethod",
    "VerificationStandard",
]

try:
    from .performance_trend_engine import (
        AnalysisType,
        ForecastMethod,
        ForecastResult,
        PerformanceDataPoint,
        PerformanceTrendEngine,
        PerformanceTrendResult,
        RegressionMetric,
        RegressionValidation,
        RollingAnalysis,
        SavingsVerification,
        TrendAnalysis,
        TrendDirection,
        VerificationStandard,
        YearOverYearComparison,
    )
    _loaded_engines.append("PerformanceTrendEngine")
except ImportError as e:
    logger.debug("Engine 9 (PerformanceTrendEngine) not available: %s", e)
    _ENGINE_9_SYMBOLS = []


# ===================================================================
# Engine 10: Management Review
# ===================================================================
_ENGINE_10_SYMBOLS: list[str] = [
    "ManagementReviewEngine",
    "ReviewInput",
    "PolicyReview",
    "ObjectivesReview",
    "EnPISummary",
    "ResourceReview",
    "AuditSummary",
    "ContinualImprovement",
    "ReviewDecision",
    "ManagementReviewMinutes",
    "ManagementReviewResult",
    "ReviewFrequency",
    "ReviewStatus",
    "DecisionType",
    "KPIStatus",
    "ActionItemPriority",
    "ResourceAdequacy",
]

try:
    from .management_review_engine import (
        ActionItemPriority,
        AuditSummary,
        ContinualImprovement,
        DecisionType,
        EnPISummary,
        KPIStatus,
        ManagementReviewEngine,
        ManagementReviewMinutes,
        ManagementReviewResult,
        ObjectivesReview,
        PolicyReview,
        ResourceAdequacy,
        ResourceReview,
        ReviewDecision,
        ReviewFrequency,
        ReviewInput,
        ReviewStatus,
    )
    _loaded_engines.append("ManagementReviewEngine")
except ImportError as e:
    logger.debug("Engine 10 (ManagementReviewEngine) not available: %s", e)
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
    "PACK-034 ISO 50001 EnMS engines: %d/%d loaded",
    len(_loaded_engines),
    __engines_count__,
)
