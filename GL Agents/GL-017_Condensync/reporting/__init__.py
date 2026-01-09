# -*- coding: utf-8 -*-
"""
GL-017 CONDENSYNC Reporting Module

Comprehensive reporting capabilities for condenser optimization including:
- Performance reports (daily, weekly, monthly)
- Maintenance reports (cleaning verification, effectiveness analysis)
- Climate impact reports (emissions, seasonal analysis, compliance)
- Executive dashboards (KPI summaries, ROI tracking, reliability)

Zero-Hallucination Guarantee:
- All metrics derived from actual operational data
- Deterministic calculations with full provenance
- No AI inference in any calculation path

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

# Performance Reporter
from .performance_reporter import (
    # Main class
    PerformanceReporter,
    # Configuration
    ReporterConfig,
    DesignConditions,
    # Enumerations
    ReportPeriod,
    PerformanceRating,
    TrendDirection,
    AlertSeverity,
    BenchmarkType,
    # KPI data classes
    HeatTransferKPIs,
    VacuumKPIs,
    EconomicKPIs,
    OperationalKPIs,
    # Event and recommendation models
    PerformanceEvent,
    Recommendation,
    BenchmarkComparison,
    # Report models
    DailyShiftReport,
    WeeklyPerformanceReport,
    MonthlyROIReport,
    # Input model
    CondenserDataPoint,
)

# Maintenance Reporter
from .maintenance_reporter import (
    # Main class
    MaintenanceReporter,
    # Configuration
    MaintenanceReporterConfig,
    # Enumerations
    CleaningMethod,
    CleaningEffectiveness,
    MaintenanceEventType,
    CMWSWorkOrderStatus,
    CMWSPriority,
    # Data models
    CondenserMeasurement,
    MeasurementPeriodStats,
    CleaningEvent,
    CMWSWorkOrder,
    CMWSEventHistory,
    # Report models
    CleaningVerificationReport,
    CleaningEffectivenessAnalysis,
    LessonsLearnedDocument,
)

# Climate Reporter
from .climate_reporter import (
    # Main class
    ClimateReporter,
    # Configuration
    ClimateReporterConfig,
    # Enumerations
    EmissionScope,
    ComplianceFramework,
    FuelType,
    Season,
    PerformanceRating as ClimatePerformanceRating,
    # Constants
    EMISSION_FACTORS_KG_CO2E_PER_MWH,
    REGIONAL_GRID_FACTORS,
    # Data models
    CondenserEmissionDataPoint,
    EmissionsBreakdown,
    SeasonalPerformance,
    ComplianceStatus,
    # Report models
    ClimateImpactReport,
    EnvironmentalComplianceReport,
)

# Executive Reporter
from .executive_reporter import (
    # Main class
    ExecutiveReporter,
    # Configuration
    ExecutiveReporterConfig,
    # Enumerations
    ExecutiveKPIStatus,
    TrendIndicator,
    ReportFrequency,
    ComplianceLevel,
    # Data models
    KPICard,
    FinancialSummary,
    ReliabilityMetrics,
    ComplianceSummary,
    PeriodPerformanceData,
    # Report models
    ExecutiveDashboard,
    ROITrackingReport,
)

__all__ = [
    # ========================================================================
    # Performance Reporter
    # ========================================================================
    "PerformanceReporter",
    "ReporterConfig",
    "DesignConditions",
    # Enums
    "ReportPeriod",
    "PerformanceRating",
    "TrendDirection",
    "AlertSeverity",
    "BenchmarkType",
    # KPIs
    "HeatTransferKPIs",
    "VacuumKPIs",
    "EconomicKPIs",
    "OperationalKPIs",
    # Events
    "PerformanceEvent",
    "Recommendation",
    "BenchmarkComparison",
    # Reports
    "DailyShiftReport",
    "WeeklyPerformanceReport",
    "MonthlyROIReport",
    # Input
    "CondenserDataPoint",

    # ========================================================================
    # Maintenance Reporter
    # ========================================================================
    "MaintenanceReporter",
    "MaintenanceReporterConfig",
    # Enums
    "CleaningMethod",
    "CleaningEffectiveness",
    "MaintenanceEventType",
    "CMWSWorkOrderStatus",
    "CMWSPriority",
    # Data models
    "CondenserMeasurement",
    "MeasurementPeriodStats",
    "CleaningEvent",
    "CMWSWorkOrder",
    "CMWSEventHistory",
    # Reports
    "CleaningVerificationReport",
    "CleaningEffectivenessAnalysis",
    "LessonsLearnedDocument",

    # ========================================================================
    # Climate Reporter
    # ========================================================================
    "ClimateReporter",
    "ClimateReporterConfig",
    # Enums
    "EmissionScope",
    "ComplianceFramework",
    "FuelType",
    "Season",
    "ClimatePerformanceRating",
    # Constants
    "EMISSION_FACTORS_KG_CO2E_PER_MWH",
    "REGIONAL_GRID_FACTORS",
    # Data models
    "CondenserEmissionDataPoint",
    "EmissionsBreakdown",
    "SeasonalPerformance",
    "ComplianceStatus",
    # Reports
    "ClimateImpactReport",
    "EnvironmentalComplianceReport",

    # ========================================================================
    # Executive Reporter
    # ========================================================================
    "ExecutiveReporter",
    "ExecutiveReporterConfig",
    # Enums
    "ExecutiveKPIStatus",
    "TrendIndicator",
    "ReportFrequency",
    "ComplianceLevel",
    # Data models
    "KPICard",
    "FinancialSummary",
    "ReliabilityMetrics",
    "ComplianceSummary",
    "PeriodPerformanceData",
    # Reports
    "ExecutiveDashboard",
    "ROITrackingReport",
]

__version__ = "1.0.0"
__author__ = "GL-BackendDeveloper"
