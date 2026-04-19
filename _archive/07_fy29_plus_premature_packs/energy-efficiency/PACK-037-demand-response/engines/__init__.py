# -*- coding: utf-8 -*-
"""
PACK-037 Demand Response Pack - Engines Package
=================================================

Production-grade calculation engines for demand response program
management, DER coordination, performance tracking, revenue
optimization, carbon impact assessment, and reporting.

Engines:
    1. LoadFlexibilityEngine         - Load flexibility profiling & scoring
    2. DRProgramEngine               - DR program matching & revenue projection
    3. BaselineEngine                 - Customer baseline load calculation (8 methods)
    4. DispatchOptimizerEngine        - Multi-objective dispatch optimization
    5. EventManagerEngine             - 5-phase event lifecycle management
    6. DERCoordinatorEngine           - DER asset coordination & dispatch
    7. PerformanceTrackerEngine       - Event compliance & trend detection
    8. RevenueOptimizerEngine         - Multi-stream revenue optimization
    9. CarbonImpactEngine             - Marginal emission factor analysis
   10. DRReportingEngine              - Dashboard panels & report generation

Architecture:
    All engines use deterministic calculations (no LLM in calc path),
    Pydantic v2 models, Python Decimal for financial precision, and
    SHA-256 provenance hashing for audit trail integrity.

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-037 Demand Response
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack_id__: str = "PACK-037"
__pack_name__: str = "Demand Response Pack"
__engines_count__: int = 10

_loaded_engines: list[str] = []

# ---------------------------------------------------------------------------
# Engine 1: Load Flexibility
# ---------------------------------------------------------------------------
try:
    from .load_flexibility_engine import (
        AutomationLevel,
        CurtailmentCapacity,
        FlexibilityAssessment,
        FlexibilityGrade,
        FlexibilityRegister,
        LoadCategory,
        LoadFlexibilityEngine,
        LoadProfile,
        LoadType,
        NotificationTime,
    )
    _loaded_engines.append("LoadFlexibilityEngine")
except ImportError as e:
    logger.debug("Engine 1 (LoadFlexibilityEngine) not available: %s", e)

# ---------------------------------------------------------------------------
# Engine 2: DR Program
# ---------------------------------------------------------------------------
try:
    from .dr_program_engine import (
        DRProgram,
        DRProgramEngine,
        DRProgramType,
        EligibilityStatus,
        ISORegion,
        ProgramEligibility,
        ProgramPortfolio,
        RevenueConfidence,
        RevenueProjection,
        SeasonalAvailability,
    )
    _loaded_engines.append("DRProgramEngine")
except ImportError as e:
    logger.debug("Engine 2 (DRProgramEngine) not available: %s", e)

# ---------------------------------------------------------------------------
# Engine 3: Baseline
# ---------------------------------------------------------------------------
try:
    from .baseline_engine import (
        AdjustmentType,
        BaselineComparison,
        BaselineEngine,
        BaselineInput,
        BaselineMethodology,
        BaselineQuality,
        BaselineResult,
        DayData,
        DayType,
        IntervalData,
    )
    _loaded_engines.append("BaselineEngine")
except ImportError as e:
    logger.debug("Engine 3 (BaselineEngine) not available: %s", e)

# ---------------------------------------------------------------------------
# Engine 4: Dispatch Optimizer
# ---------------------------------------------------------------------------
try:
    from .dispatch_optimizer_engine import (
        CurtailmentCommand,
        CurtailmentStrategy,
        CommandType,
        DispatchInput,
        DispatchLoad,
        DispatchOptimizerEngine,
        DispatchPlan,
        LoadAllocation,
        ObjectiveWeight,
        PlanStatus,
        ReboundForecast,
        ReboundSeverity,
    )
    _loaded_engines.append("DispatchOptimizerEngine")
except ImportError as e:
    logger.debug("Engine 4 (DispatchOptimizerEngine) not available: %s", e)

# ---------------------------------------------------------------------------
# Engine 5: Event Manager
# ---------------------------------------------------------------------------
try:
    from .event_manager_engine import (
        CommandStatus,
        DREvent,
        EventAssessment,
        EventExecution,
        EventManagerEngine,
        EventPhase,
        EventStatus,
        EventType,
        LoadControlCommand,
        PerformanceGrade,
        PerformanceInterval,
    )
    _loaded_engines.append("EventManagerEngine")
except ImportError as e:
    logger.debug("Engine 5 (EventManagerEngine) not available: %s", e)

# ---------------------------------------------------------------------------
# Engine 6: DER Coordinator
# ---------------------------------------------------------------------------
try:
    from .der_coordinator_engine import (
        DERAsset,
        DERAssetType,
        DERCoordinatorEngine,
        DERDispatch,
        DERPerformance,
        DERPortfolio,
        DERStatus,
        DegradationModel,
        DispatchPriority,
        DispatchRequest,
        DispatchStrategy,
    )
    _loaded_engines.append("DERCoordinatorEngine")
except ImportError as e:
    logger.debug("Engine 6 (DERCoordinatorEngine) not available: %s", e)

# ---------------------------------------------------------------------------
# Engine 7: Performance Tracker
# ---------------------------------------------------------------------------
try:
    from .performance_tracker_engine import (
        ComplianceReport,
        ComplianceStatus,
        EventPerformance,
        EventRecord,
        PenaltyType,
        PerformanceTrackerEngine,
        PerformanceTrend,
        SeasonSummary,
        SeasonType,
        TrendDirection,
    )
    _loaded_engines.append("PerformanceTrackerEngine")
except ImportError as e:
    logger.debug("Engine 7 (PerformanceTrackerEngine) not available: %s", e)

# ---------------------------------------------------------------------------
# Engine 8: Revenue Optimizer
# ---------------------------------------------------------------------------
try:
    from .revenue_optimizer_engine import (
        AnnualCashFlow,
        CostCategory,
        CostItem,
        OptimisationObjective,
        ProgrammeFinancials,
        RevenueForecast,
        RevenueOptimization,
        RevenueOptimizerEngine,
        RevenueStream,
        RevenueStreamType,
        ScenarioParameter,
        WhatIfScenario,
    )
    _loaded_engines.append("RevenueOptimizerEngine")
except ImportError as e:
    logger.debug("Engine 8 (RevenueOptimizerEngine) not available: %s", e)

# ---------------------------------------------------------------------------
# Engine 9: Carbon Impact
# ---------------------------------------------------------------------------
try:
    from .carbon_impact_engine import (
        AnnualCarbonSummary,
        CarbonImpactEngine,
        CarbonReport,
        CarbonReportType,
        DREventCarbon,
        EventCarbonImpact,
        GridRegion,
        MarginalEmissionFactor,
        SBTiAmbition,
        Scope2Method,
        TimeOfDay,
    )
    _loaded_engines.append("CarbonImpactEngine")
except ImportError as e:
    logger.debug("Engine 9 (CarbonImpactEngine) not available: %s", e)

# ---------------------------------------------------------------------------
# Engine 10: DR Reporting
# ---------------------------------------------------------------------------
try:
    from .dr_reporting_engine import (
        DRReportingEngine,
        DashboardData,
        DashboardPanel,
        DashboardWidget,
        EventSummary,
        ExecutiveSummary,
        ExportFormat,
        KPIMetric,
        ProgrammeMetrics,
        ReportOutput,
        ReportType,
        SettlementPackage,
        WidgetType,
    )
    _loaded_engines.append("DRReportingEngine")
except ImportError as e:
    logger.debug("Engine 10 (DRReportingEngine) not available: %s", e)


__all__: list[str] = [
    # Module metadata
    "__version__",
    "__pack_id__",
    "__pack_name__",
    "__engines_count__",
    # --- Engine 1: Load Flexibility ---
    "LoadFlexibilityEngine",
    "LoadProfile",
    "FlexibilityAssessment",
    "CurtailmentCapacity",
    "FlexibilityRegister",
    "LoadCategory",
    "LoadType",
    "NotificationTime",
    "AutomationLevel",
    "FlexibilityGrade",
    # --- Engine 2: DR Program ---
    "DRProgramEngine",
    "DRProgram",
    "ProgramEligibility",
    "RevenueProjection",
    "ProgramPortfolio",
    "DRProgramType",
    "ISORegion",
    "EligibilityStatus",
    "RevenueConfidence",
    "SeasonalAvailability",
    # --- Engine 3: Baseline ---
    "BaselineEngine",
    "BaselineInput",
    "BaselineResult",
    "BaselineComparison",
    "IntervalData",
    "DayData",
    "BaselineMethodology",
    "AdjustmentType",
    "BaselineQuality",
    "DayType",
    # --- Engine 4: Dispatch Optimizer ---
    "DispatchOptimizerEngine",
    "DispatchInput",
    "DispatchLoad",
    "DispatchPlan",
    "LoadAllocation",
    "CurtailmentCommand",
    "ReboundForecast",
    "ObjectiveWeight",
    "CurtailmentStrategy",
    "CommandType",
    "PlanStatus",
    "ReboundSeverity",
    # --- Engine 5: Event Manager ---
    "EventManagerEngine",
    "DREvent",
    "EventExecution",
    "EventAssessment",
    "LoadControlCommand",
    "PerformanceInterval",
    "EventType",
    "EventStatus",
    "EventPhase",
    "PerformanceGrade",
    "CommandStatus",
    # --- Engine 6: DER Coordinator ---
    "DERCoordinatorEngine",
    "DERAsset",
    "DERDispatch",
    "DERPerformance",
    "DERPortfolio",
    "DispatchRequest",
    "DERAssetType",
    "DERStatus",
    "DispatchPriority",
    "DegradationModel",
    "DispatchStrategy",
    # --- Engine 7: Performance Tracker ---
    "PerformanceTrackerEngine",
    "EventRecord",
    "EventPerformance",
    "SeasonSummary",
    "PerformanceTrend",
    "ComplianceReport",
    "ComplianceStatus",
    "TrendDirection",
    "SeasonType",
    "PenaltyType",
    # --- Engine 8: Revenue Optimizer ---
    "RevenueOptimizerEngine",
    "RevenueStream",
    "CostItem",
    "ProgrammeFinancials",
    "AnnualCashFlow",
    "RevenueForecast",
    "WhatIfScenario",
    "RevenueOptimization",
    "RevenueStreamType",
    "CostCategory",
    "ScenarioParameter",
    "OptimisationObjective",
    # --- Engine 9: Carbon Impact ---
    "CarbonImpactEngine",
    "MarginalEmissionFactor",
    "DREventCarbon",
    "EventCarbonImpact",
    "AnnualCarbonSummary",
    "CarbonReport",
    "GridRegion",
    "TimeOfDay",
    "Scope2Method",
    "SBTiAmbition",
    "CarbonReportType",
    # --- Engine 10: DR Reporting ---
    "DRReportingEngine",
    "KPIMetric",
    "DashboardWidget",
    "ProgrammeMetrics",
    "EventSummary",
    "DashboardData",
    "ReportOutput",
    "ExecutiveSummary",
    "SettlementPackage",
    "DashboardPanel",
    "ReportType",
    "ExportFormat",
    "WidgetType",
]


def get_loaded_engines() -> list[str]:
    """Return list of engine class names that loaded successfully."""
    return list(_loaded_engines)


logger.info(
    "PACK-037 Demand Response engines: %d/%d loaded",
    len(_loaded_engines),
    __engines_count__,
)
