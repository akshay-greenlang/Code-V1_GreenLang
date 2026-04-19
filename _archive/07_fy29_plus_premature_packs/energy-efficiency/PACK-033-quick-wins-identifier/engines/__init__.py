# -*- coding: utf-8 -*-
"""
PACK-033 Quick Wins Identifier Pack - Engines Module
========================================================

Calculation engines for cross-sector quick wins identification,
financial analysis, energy savings estimation, carbon reduction
assessment, implementation prioritization, behavioral change
modeling, utility rebate matching, and progress reporting.

Engines:
    1. QuickWinsScannerEngine          - 80+ actions, 15 categories, building profiles
    2. PaybackCalculatorEngine         - NPV, IRR, ROI, LCOE, discounted payback
    3. EnergySavingsEstimatorEngine    - ASHRAE 14 uncertainty, interactive effects
    4. CarbonReductionEngine           - GHG Protocol scopes, SBTi alignment
    5. ImplementationPrioritizerEngine  - MCDA, Pareto frontier, dependency graphs
    6. BehavioralChangeEngine          - Rogers diffusion, persistence, gamification
    7. UtilityRebateEngine             - 100+ utility programs, rebate matching
    8. QuickWinsReportingEngine        - IPMVP verification, dashboards, export

Regulatory Basis:
    EU Directive 2023/1791 (EED - Energy Efficiency Directive)
    ISO 50001:2018 (Energy Management Systems)
    ASHRAE Guideline 14-2014 (Measurement of Energy, Demand, and Water Savings)
    IPMVP (International Performance Measurement & Verification Protocol)
    GHG Protocol Corporate Standard (Scope 1/2/3)
    SBTi Science-Based Targets framework

Pack Tier: Professional (PACK-033)
Author: GreenLang Platform Team
Date: March 2026
Version: 1.0.0
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack__: str = "PACK-033"
__pack_name__: str = "Quick Wins Identifier Pack"
__engines_count__: int = 8

# Track which engines loaded successfully
_loaded_engines: list[str] = []


# ===================================================================
# Engine 1: Quick Wins Scanner
# ===================================================================
_ENGINE_1_SYMBOLS: list[str] = [
    "QuickWinsScannerEngine",
    "FacilityProfile",
    "EquipmentSurvey",
    "QuickWinAction",
    "ScanResult",
    "QuickWinsScanResult",
    "BuildingType",
    "ActionCategory",
    "ActionComplexity",
    "ActionPriority",
    "ScanStatus",
    "DisruptionLevel",
    "QUICK_WINS_LIBRARY",
]

try:
    from .quick_wins_scanner_engine import (
        ActionCategory,
        ActionComplexity,
        ActionPriority,
        BuildingType,
        DisruptionLevel,
        EquipmentSurvey,
        FacilityProfile,
        QUICK_WINS_LIBRARY,
        QuickWinAction,
        QuickWinsScanResult,
        QuickWinsScannerEngine,
        ScanResult,
        ScanStatus,
    )
    _loaded_engines.append("QuickWinsScannerEngine")
except ImportError as e:
    logger.debug("Engine 1 (QuickWinsScannerEngine) not available: %s", e)
    _ENGINE_1_SYMBOLS = []


# ===================================================================
# Engine 2: Payback Calculator
# ===================================================================
_ENGINE_2_SYMBOLS: list[str] = [
    "PaybackCalculatorEngine",
    "MeasureFinancials",
    "FinancialParameters",
    "Incentive",
    "CashFlow",
    "PaybackResult",
    "BatchPaybackResult",
    "SensitivityResult",
    "AnalysisPeriod",
    "FinancialMetric",
    "TaxTreatment",
    "IncentiveType",
    "SensitivityParameter",
]

try:
    from .payback_calculator_engine import (
        AnalysisPeriod,
        BatchPaybackResult,
        CashFlow,
        FinancialMetric,
        FinancialParameters,
        Incentive,
        IncentiveType,
        MeasureFinancials,
        PaybackCalculatorEngine,
        PaybackResult,
        SensitivityParameter,
        SensitivityResult,
        TaxTreatment,
    )
    _loaded_engines.append("PaybackCalculatorEngine")
except ImportError as e:
    logger.debug("Engine 2 (PaybackCalculatorEngine) not available: %s", e)
    _ENGINE_2_SYMBOLS = []


# ===================================================================
# Engine 3: Energy Savings Estimator
# ===================================================================
_ENGINE_3_SYMBOLS: list[str] = [
    "EnergySavingsEstimatorEngine",
    "FacilityBaseline",
    "MeasureSavingsInput",
    "SavingsBand",
    "SavingsEstimate",
    "InteractiveEffect",
    "BundleSavingsResult",
    "EnergyType",
    "SavingsUnit",
    "ClimateZone",
    "EstimationMethod",
    "ConfidenceLevel",
    "InteractionType",
]

try:
    from .energy_savings_estimator_engine import (
        BundleSavingsResult,
        ClimateZone,
        ConfidenceLevel,
        EnergySavingsEstimatorEngine,
        EnergyType,
        EstimationMethod,
        FacilityBaseline,
        InteractionType,
        InteractiveEffect,
        MeasureSavingsInput,
        SavingsBand,
        SavingsEstimate,
        SavingsUnit,
    )
    _loaded_engines.append("EnergySavingsEstimatorEngine")
except ImportError as e:
    logger.debug("Engine 3 (EnergySavingsEstimatorEngine) not available: %s", e)
    _ENGINE_3_SYMBOLS = []


# ===================================================================
# Engine 4: Carbon Reduction
# ===================================================================
_ENGINE_4_SYMBOLS: list[str] = [
    "CarbonReductionEngine",
    "EmissionFactor",
    "EnergyReduction",
    "CarbonReductionResult",
    "PortfolioReduction",
    "SBTiAssessment",
    "AnnualProjection",
    "EmissionScope",
    "CalculationMethod",
    "EmissionFactorType",
    "FuelType",
    "GridRegion",
    "SBTiAmbition",
    "ProjectionMethod",
]

try:
    from .carbon_reduction_engine import (
        AnnualProjection,
        CalculationMethod,
        CarbonReductionEngine,
        CarbonReductionResult,
        EmissionFactor,
        EmissionFactorType,
        EmissionScope,
        EnergyReduction,
        FuelType,
        GridRegion,
        PortfolioReduction,
        ProjectionMethod,
        SBTiAmbition,
        SBTiAssessment,
    )
    _loaded_engines.append("CarbonReductionEngine")
except ImportError as e:
    logger.debug("Engine 4 (CarbonReductionEngine) not available: %s", e)
    _ENGINE_4_SYMBOLS = []


# ===================================================================
# Engine 5: Implementation Prioritizer
# ===================================================================
_ENGINE_5_SYMBOLS: list[str] = [
    "ImplementationPrioritizerEngine",
    "MeasureForPrioritization",
    "CriterionWeight",
    "WeightSet",
    "NormalizedScore",
    "PriorityResult",
    "DependencyEdge",
    "ImplementationSequence",
    "PrioritizationResult",
    "CriterionName",
    "WeightProfile",
    "DependencyType",
    "ImplementationPhase",
    "ParetoStatus",
    "ScoreNormalization",
]

try:
    from .implementation_prioritizer_engine import (
        CriterionName,
        CriterionWeight,
        DependencyEdge,
        DependencyType,
        ImplementationPhase,
        ImplementationPrioritizerEngine,
        ImplementationSequence,
        MeasureForPrioritization,
        NormalizedScore,
        ParetoStatus,
        PrioritizationResult,
        PriorityResult,
        ScoreNormalization,
        WeightProfile,
        WeightSet,
    )
    _loaded_engines.append("ImplementationPrioritizerEngine")
except ImportError as e:
    logger.debug("Engine 5 (ImplementationPrioritizerEngine) not available: %s", e)
    _ENGINE_5_SYMBOLS = []


# ===================================================================
# Engine 6: Behavioral Change
# ===================================================================
_ENGINE_6_SYMBOLS: list[str] = [
    "BehavioralChangeEngine",
    "BehavioralAction",
    "OrganizationProfile",
    "AdoptionCurvePoint",
    "PersistenceModel",
    "EngagementProgram",
    "GamificationScore",
    "BehavioralProgramResult",
    "BehavioralCategory",
    "AdoptionStage",
    "AdopterType",
    "EngagementChannel",
    "PersistenceLevel",
    "ProgramStatus",
]

try:
    from .behavioral_change_engine import (
        AdopterType,
        AdoptionCurvePoint,
        AdoptionStage,
        BehavioralAction,
        BehavioralCategory,
        BehavioralChangeEngine,
        BehavioralProgramResult,
        EngagementChannel,
        EngagementProgram,
        GamificationScore,
        OrganizationProfile,
        PersistenceLevel,
        PersistenceModel,
        ProgramStatus,
    )
    _loaded_engines.append("BehavioralChangeEngine")
except ImportError as e:
    logger.debug("Engine 6 (BehavioralChangeEngine) not available: %s", e)
    _ENGINE_6_SYMBOLS = []


# ===================================================================
# Engine 7: Utility Rebate
# ===================================================================
_ENGINE_7_SYMBOLS: list[str] = [
    "UtilityRebateEngine",
    "UtilityProgram",
    "MeasureForRebate",
    "RebateMatch",
    "RebateApplication",
    "RebatePortfolio",
    "ProgramType",
    "MeasureCategory",
    "ApplicationStatus",
    "CustomerSegment",
    "RebateUnit",
    "UtilityRegion",
]

try:
    from .utility_rebate_engine import (
        ApplicationStatus,
        CustomerSegment,
        MeasureCategory,
        MeasureForRebate,
        ProgramType,
        RebateApplication,
        RebateMatch,
        RebatePortfolio,
        RebateUnit,
        UtilityProgram,
        UtilityRebateEngine,
        UtilityRegion,
    )
    _loaded_engines.append("UtilityRebateEngine")
except ImportError as e:
    logger.debug("Engine 7 (UtilityRebateEngine) not available: %s", e)
    _ENGINE_7_SYMBOLS = []


# ===================================================================
# Engine 8: Quick Wins Reporting
# ===================================================================
_ENGINE_8_SYMBOLS: list[str] = [
    "QuickWinsReportingEngine",
    "KPIMetric",
    "ProgressEntry",
    "SavingsVerification",
    "DashboardData",
    "ExecutiveSummary",
    "ReportOutput",
    "ReportType",
    "ReportFormat",
    "VerificationMethod",
    "DashboardWidget",
    "ProgressStatus",
    "TrendDirection",
]

try:
    from .quick_wins_reporting_engine import (
        DashboardData,
        DashboardWidget,
        ExecutiveSummary,
        KPIMetric,
        ProgressEntry,
        ProgressStatus,
        QuickWinsReportingEngine,
        ReportFormat,
        ReportOutput,
        ReportType,
        SavingsVerification,
        TrendDirection,
        VerificationMethod,
    )
    _loaded_engines.append("QuickWinsReportingEngine")
except ImportError as e:
    logger.debug("Engine 8 (QuickWinsReportingEngine) not available: %s", e)
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
    *_ENGINE_1_SYMBOLS,
    *_ENGINE_2_SYMBOLS,
    *_ENGINE_3_SYMBOLS,
    *_ENGINE_4_SYMBOLS,
    *_ENGINE_5_SYMBOLS,
    *_ENGINE_6_SYMBOLS,
    *_ENGINE_7_SYMBOLS,
    *_ENGINE_8_SYMBOLS,
]


def get_loaded_engines() -> list[str]:
    """Return list of engine class names that loaded successfully."""
    return list(_loaded_engines)


def get_engine_count() -> int:
    """Return count of engines that loaded successfully."""
    return len(_loaded_engines)


logger.info(
    "PACK-033 Quick Wins Identifier engines: %d/%d loaded",
    len(_loaded_engines),
    __engines_count__,
)
