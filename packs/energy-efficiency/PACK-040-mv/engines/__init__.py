# -*- coding: utf-8 -*-
"""
PACK-040 M&V Pack - Engines Package
=======================================

Production-grade calculation engines for Measurement & Verification
(M&V) per IPMVP Core Concepts 2022, ASHRAE Guideline 14-2014,
ISO 50015:2014, FEMP M&V Guidelines 4.0, and EU EED Article 7.

Engines:
    1. BaselineEngine        - Multivariate regression baselines (3P/4P/5P/TOWT)
    2. AdjustmentEngine      - Routine and non-routine adjustments per IPMVP
    3. SavingsEngine         - Avoided energy, normalised savings, cost savings
    4. UncertaintyEngine     - ASHRAE 14 fractional savings uncertainty
    5. IPMVPOptionEngine     - IPMVP Options A/B/C/D with automated selection
    6. RegressionEngine      - OLS, change-point, TOWT regression with diagnostics
    7. WeatherEngine         - HDD/CDD, balance point, TMY normalisation
    8. MeteringEngine        - Metering plan, calibration, sampling protocols
    9. PersistenceEngine     - Multi-year savings persistence and degradation
   10. MVReportingEngine     - Automated M&V report generation and compliance

Architecture:
    All engines use deterministic calculations (no LLM in calc path),
    Pydantic v2 models, Python Decimal for financial precision, and
    SHA-256 provenance hashing for audit trail integrity.

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-040 M&V
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack_id__: str = "PACK-040"
__pack_name__: str = "M&V Pack"
__engines_count__: int = 10

_loaded_engines: list[str] = []

# ---------------------------------------------------------------------------
# Engine 1: Baseline Engine
# ---------------------------------------------------------------------------
try:
    from .baseline_engine import (
        BaselineConfig,
        BaselineEngine,
        BaselineModel,
        BaselineResult,
        BaselinePeriod,
        ModelComparison,
        ModelType,
        ModelValidation,
        ValidationCriteria,
        VariableDefinition,
        VariableType,
    )
    _loaded_engines.append("BaselineEngine")
except ImportError as e:
    logger.debug("Engine 1 (BaselineEngine) not available: %s", e)

# ---------------------------------------------------------------------------
# Engine 2: Adjustment Engine
# ---------------------------------------------------------------------------
try:
    from .adjustment_engine import (
        AdjustmentConfig,
        AdjustmentEngine,
        AdjustmentRecord,
        AdjustmentResult,
        AdjustmentType,
        NonRoutineAdjustment,
        NonRoutineType,
        RoutineAdjustment,
        RoutineType,
        StaticFactor,
        StaticFactorType,
    )
    _loaded_engines.append("AdjustmentEngine")
except ImportError as e:
    logger.debug("Engine 2 (AdjustmentEngine) not available: %s", e)

# ---------------------------------------------------------------------------
# Engine 3: Savings Engine
# ---------------------------------------------------------------------------
try:
    from .savings_engine import (
        CostSavingsResult,
        CumulativeSavings,
        EnergySavingsResult,
        NormalisedSavings,
        SavingsCalculationType,
        SavingsConfig,
        SavingsEngine,
        SavingsMethod,
        SavingsPeriod,
        SavingsResult,
        SavingsSummary,
    )
    _loaded_engines.append("SavingsEngine")
except ImportError as e:
    logger.debug("Engine 3 (SavingsEngine) not available: %s", e)

# ---------------------------------------------------------------------------
# Engine 4: Uncertainty Engine
# ---------------------------------------------------------------------------
try:
    from .uncertainty_engine import (
        CombinedUncertainty,
        ConfidenceLevel,
        MeasurementUncertainty,
        ModelUncertainty,
        SamplingUncertainty,
        UncertaintyBudget,
        UncertaintyComponent,
        UncertaintyConfig,
        UncertaintyEngine,
        UncertaintyResult,
        UncertaintyType,
    )
    _loaded_engines.append("UncertaintyEngine")
except ImportError as e:
    logger.debug("Engine 4 (UncertaintyEngine) not available: %s", e)

# ---------------------------------------------------------------------------
# Engine 5: IPMVP Option Engine
# ---------------------------------------------------------------------------
try:
    from .ipmvp_option_engine import (
        BoundaryDefinition,
        IPMVPOptionEngine,
        OptionComparison,
        OptionConfig,
        OptionEvaluation,
        OptionRecommendation,
        OptionResult,
        OptionScoring,
        OptionType as IPMVPOptionType,
        VerificationProtocol,
        VerificationType,
    )
    _loaded_engines.append("IPMVPOptionEngine")
except ImportError as e:
    logger.debug("Engine 5 (IPMVPOptionEngine) not available: %s", e)

# ---------------------------------------------------------------------------
# Engine 6: Regression Engine
# ---------------------------------------------------------------------------
try:
    from .regression_engine import (
        ChangePointResult,
        CoefficientDetail,
        DataFrequency,
        DiagnosticResult,
        DiagnosticStatus,
        IndependentVariable,
        ModelComparisonEntry,
        ModelComparisonResult,
        RegressionConfig,
        RegressionEngine,
        RegressionFitResult,
        RegressionModelType,
        RegressionStatistics,
        ResidualPattern,
        ValidationGrade,
        VariableRole,
    )
    _loaded_engines.append("RegressionEngine")
except ImportError as e:
    logger.debug("Engine 6 (RegressionEngine) not available: %s", e)

# ---------------------------------------------------------------------------
# Engine 7: Weather Engine
# ---------------------------------------------------------------------------
try:
    from .weather_engine import (
        BalancePointResult,
        DailyTemperature,
        DegreeDayRecord,
        DegreeDayRegressionResult,
        DegreeDayType,
        GapFillMethod as WeatherGapFillMethod,
        NormalisationMethod,
        QualityFlag,
        TMYNormalisationResult,
        TemperatureUnit,
        WeatherDataSource,
        WeatherEngine,
        WeatherQualityReport,
        WeatherReconciliationResult,
    )
    _loaded_engines.append("WeatherEngine")
except ImportError as e:
    logger.debug("Engine 7 (WeatherEngine) not available: %s", e)

# ---------------------------------------------------------------------------
# Engine 8: Metering Engine
# ---------------------------------------------------------------------------
try:
    from .metering_engine import (
        AccuracyClass,
        CalibrationRecord as MeterCalibrationRecord,
        CalibrationStatus,
        DataGap,
        DataQualityResult,
        GapFillMethod as MeteringGapFillMethod,
        GapSeverity,
        IPMVPOption,
        MeasurementPoint,
        MeterSelectionResult,
        MeterSpec,
        MeterType,
        MeterUncertaintyResult,
        MeteringEngine,
        MeteringPlan,
        SamplingProtocol,
    )
    _loaded_engines.append("MeteringEngine")
except ImportError as e:
    logger.debug("Engine 8 (MeteringEngine) not available: %s", e)

# ---------------------------------------------------------------------------
# Engine 9: Persistence Engine
# ---------------------------------------------------------------------------
try:
    from .persistence_engine import (
        AlertLevel,
        AnnualSavingsRecord,
        ContractType,
        DegradationModel,
        DegradationResult,
        GuaranteeTrackingResult,
        PersistenceAlert,
        PersistenceEngine,
        PersistenceResult,
        PersistenceStatus,
        SeasonType,
        SeasonalAnalysisResult,
        SeasonalPattern,
        TrendDirection,
        YearOverYearComparison,
    )
    _loaded_engines.append("PersistenceEngine")
except ImportError as e:
    logger.debug("Engine 9 (PersistenceEngine) not available: %s", e)

# ---------------------------------------------------------------------------
# Engine 10: MV Reporting Engine
# ---------------------------------------------------------------------------
try:
    from .mv_reporting_engine import (
        CheckStatus,
        ComplianceCheck,
        ComplianceFramework,
        ComplianceResult,
        DistributionChannel,
        MVReportType,
        MVReportingEngine,
        ReportConfig,
        ReportFormat,
        ReportOutput,
        ReportSchedule,
        ReportSection,
        ReportingResult,
        ScheduleFrequency,
    )
    _loaded_engines.append("MVReportingEngine")
except ImportError as e:
    logger.debug("Engine 10 (MVReportingEngine) not available: %s", e)


__all__: list[str] = [
    # Module metadata
    "__version__",
    "__pack_id__",
    "__pack_name__",
    "__engines_count__",
    # --- Engine 1: Baseline ---
    "BaselineEngine",
    "BaselineConfig",
    "BaselineModel",
    "BaselineResult",
    "BaselinePeriod",
    "ModelComparison",
    "ModelType",
    "ModelValidation",
    "ValidationCriteria",
    "VariableDefinition",
    "VariableType",
    # --- Engine 2: Adjustment ---
    "AdjustmentEngine",
    "AdjustmentConfig",
    "AdjustmentRecord",
    "AdjustmentResult",
    "AdjustmentType",
    "NonRoutineAdjustment",
    "NonRoutineType",
    "RoutineAdjustment",
    "RoutineType",
    "StaticFactor",
    "StaticFactorType",
    # --- Engine 3: Savings ---
    "SavingsEngine",
    "SavingsConfig",
    "SavingsResult",
    "SavingsSummary",
    "SavingsMethod",
    "SavingsPeriod",
    "SavingsCalculationType",
    "EnergySavingsResult",
    "CostSavingsResult",
    "CumulativeSavings",
    "NormalisedSavings",
    # --- Engine 4: Uncertainty ---
    "UncertaintyEngine",
    "UncertaintyConfig",
    "UncertaintyResult",
    "UncertaintyBudget",
    "UncertaintyComponent",
    "UncertaintyType",
    "ConfidenceLevel",
    "MeasurementUncertainty",
    "ModelUncertainty",
    "SamplingUncertainty",
    "CombinedUncertainty",
    # --- Engine 5: IPMVP Option ---
    "IPMVPOptionEngine",
    "IPMVPOptionType",
    "OptionConfig",
    "OptionResult",
    "OptionComparison",
    "OptionEvaluation",
    "OptionRecommendation",
    "OptionScoring",
    "BoundaryDefinition",
    "VerificationProtocol",
    "VerificationType",
    # --- Engine 6: Regression ---
    "RegressionEngine",
    "RegressionConfig",
    "RegressionFitResult",
    "RegressionStatistics",
    "RegressionModelType",
    "CoefficientDetail",
    "ChangePointResult",
    "DiagnosticResult",
    "DiagnosticStatus",
    "DataFrequency",
    "ValidationGrade",
    "ResidualPattern",
    "VariableRole",
    "IndependentVariable",
    "ModelComparisonEntry",
    "ModelComparisonResult",
    # --- Engine 7: Weather ---
    "WeatherEngine",
    "DegreeDayType",
    "TemperatureUnit",
    "WeatherDataSource",
    "QualityFlag",
    "WeatherGapFillMethod",
    "NormalisationMethod",
    "DailyTemperature",
    "DegreeDayRecord",
    "BalancePointResult",
    "DegreeDayRegressionResult",
    "TMYNormalisationResult",
    "WeatherQualityReport",
    "WeatherReconciliationResult",
    # --- Engine 8: Metering ---
    "MeteringEngine",
    "MeterType",
    "AccuracyClass",
    "CalibrationStatus",
    "GapSeverity",
    "MeteringGapFillMethod",
    "IPMVPOption",
    "MeasurementPoint",
    "MeterSpec",
    "MeterCalibrationRecord",
    "SamplingProtocol",
    "DataGap",
    "MeteringPlan",
    "DataQualityResult",
    "MeterSelectionResult",
    "MeterUncertaintyResult",
    # --- Engine 9: Persistence ---
    "PersistenceEngine",
    "DegradationModel",
    "PersistenceStatus",
    "AlertLevel",
    "ContractType",
    "SeasonType",
    "TrendDirection",
    "AnnualSavingsRecord",
    "DegradationResult",
    "PersistenceAlert",
    "GuaranteeTrackingResult",
    "SeasonalPattern",
    "SeasonalAnalysisResult",
    "PersistenceResult",
    "YearOverYearComparison",
    # --- Engine 10: MV Reporting ---
    "MVReportingEngine",
    "MVReportType",
    "ReportFormat",
    "ComplianceFramework",
    "CheckStatus",
    "ScheduleFrequency",
    "DistributionChannel",
    "ReportSection",
    "ComplianceCheck",
    "ComplianceResult",
    "ReportSchedule",
    "ReportConfig",
    "ReportOutput",
    "ReportingResult",
]


def get_loaded_engines() -> list[str]:
    """Return list of engine class names that loaded successfully."""
    return list(_loaded_engines)


logger.info(
    "PACK-040 M&V engines: %d/%d loaded",
    len(_loaded_engines),
    __engines_count__,
)
