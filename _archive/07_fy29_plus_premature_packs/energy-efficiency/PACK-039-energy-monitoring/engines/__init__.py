# -*- coding: utf-8 -*-
"""
PACK-039 Energy Monitoring Pack - Engines Package
====================================================

Production-grade calculation engines for real-time energy monitoring,
meter management, data acquisition, validation, anomaly detection,
ISO 50001 EnPI tracking, cost allocation, budgeting, alarm management,
dashboards, and automated reporting.

Engines:
    1. MeterRegistryEngine       - Meter asset management and hierarchy
    2. DataAcquisitionEngine     - Multi-source data collection
    3. DataValidationEngine      - 12-check automated data quality (ASHRAE 14)
    4. AnomalyDetectionEngine    - Statistical anomaly detection
    5. EnPIEngine                - ISO 50001 Energy Performance Indicators
    6. CostAllocationEngine      - Tariff-aware tenant cost allocation
    7. BudgetEngine              - Budget tracking and variance analysis
    8. AlarmEngine               - ISA 18.2 alarm lifecycle management
    9. DashboardEngine           - Real-time dashboard with 8 panel types
   10. MonitoringReportingEngine - Automated multi-format report generation

Architecture:
    All engines use deterministic calculations (no LLM in calc path),
    Pydantic v2 models, Python Decimal for financial precision, and
    SHA-256 provenance hashing for audit trail integrity.

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-039 Energy Monitoring
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack_id__: str = "PACK-039"
__pack_name__: str = "Energy Monitoring Pack"
__engines_count__: int = 10

_loaded_engines: list[str] = []

# ---------------------------------------------------------------------------
# Engine 1: Meter Registry
# ---------------------------------------------------------------------------
try:
    from .meter_registry_engine import (
        CalibrationRecord,
        ChannelType,
        EnergyType,
        HierarchyLevel,
        MeterChannel,
        MeterConfig,
        MeterHierarchy,
        MeterProtocol,
        MeterRegistryEngine,
        MeterRegistryResult,
        MeterStatus,
        MeterType,
        VirtualMeterDefinition,
    )
    _loaded_engines.append("MeterRegistryEngine")
except ImportError as e:
    logger.debug("Engine 1 (MeterRegistryEngine) not available: %s", e)

# ---------------------------------------------------------------------------
# Engine 2: Data Acquisition
# ---------------------------------------------------------------------------
try:
    from .data_acquisition_engine import (
        AcquisitionConfig,
        AcquisitionMode,
        AcquisitionResult,
        AcquisitionSchedule,
        BufferStatus,
        DataAcquisitionEngine,
        DataStatus,
        IntervalLength,
        NormalizationMethod,
        NormalizedReading,
        RawReading,
        UnitCategory,
    )
    _loaded_engines.append("DataAcquisitionEngine")
except ImportError as e:
    logger.debug("Engine 2 (DataAcquisitionEngine) not available: %s", e)

# ---------------------------------------------------------------------------
# Engine 3: Data Validation
# ---------------------------------------------------------------------------
try:
    from .data_validation_engine import (
        CorrectionMethod,
        DataCorrection,
        DataSource,
        DataValidationEngine,
        QualityGrade,
        QualityScore,
        ValidationCheck,
        ValidationFinding,
        ValidationReport,
        ValidationRule,
        ValidationSeverity,
    )
    _loaded_engines.append("DataValidationEngine")
except ImportError as e:
    logger.debug("Engine 3 (DataValidationEngine) not available: %s", e)

# ---------------------------------------------------------------------------
# Engine 4: Anomaly Detection
# ---------------------------------------------------------------------------
try:
    from .anomaly_detection_engine import (
        AnomalyBaseline,
        AnomalyConfig,
        AnomalyDetectionEngine,
        AnomalyMethod,
        AnomalyReport,
        AnomalySeverity,
        AnomalyType,
        DetectedAnomaly,
        InvestigationRecord,
        InvestigationStatus,
        TimeContext,
    )
    _loaded_engines.append("AnomalyDetectionEngine")
except ImportError as e:
    logger.debug("Engine 4 (AnomalyDetectionEngine) not available: %s", e)

# ---------------------------------------------------------------------------
# Engine 5: EnPI
# ---------------------------------------------------------------------------
try:
    from .enpi_engine import (
        BaselineStatus,
        CUSUMTracker,
        EnPIDefinition,
        EnPIEngine,
        EnPIResult,
        EnPIType,
        EnPIValue,
        EnergyBaseline,
        PerformanceRating,
        RegressionModel,
        RegressionQuality,
        RelevantVariable,
        SignificanceLevel,
    )
    _loaded_engines.append("EnPIEngine")
except ImportError as e:
    logger.debug("Engine 5 (EnPIEngine) not available: %s", e)

# ---------------------------------------------------------------------------
# Engine 6: Cost Allocation
# ---------------------------------------------------------------------------
try:
    from .cost_allocation_engine import (
        AllocationMethod,
        AllocationResult,
        AllocationRule,
        BillingFrequency,
        BillingRecord,
        CostAllocationEngine,
        CostBreakdown,
        CostComponent,
        ReconciliationStatus,
        TenantAccount,
        TenantType,
    )
    _loaded_engines.append("CostAllocationEngine")
except ImportError as e:
    logger.debug("Engine 6 (CostAllocationEngine) not available: %s", e)

# ---------------------------------------------------------------------------
# Engine 7: Budget
# ---------------------------------------------------------------------------
try:
    from .budget_engine import (
        AlertThreshold,
        BudgetDefinition,
        BudgetEngine,
        BudgetMethod,
        BudgetPeriod,
        BudgetResult,
        BudgetStatus,
        ForecastMethod,
        RollingForecast,
        VarianceAnalysis,
        VarianceComponent,
    )
    _loaded_engines.append("BudgetEngine")
except ImportError as e:
    logger.debug("Engine 7 (BudgetEngine) not available: %s", e)

# ---------------------------------------------------------------------------
# Engine 8: Alarm
# ---------------------------------------------------------------------------
try:
    from .alarm_engine import (
        AlarmCategory,
        AlarmDefinition,
        AlarmEngine,
        AlarmEvent,
        AlarmPriority,
        AlarmReport,
        AlarmState,
        EscalationConfig,
        EscalationLevel,
        SuppressionRule,
        SuppressionType,
    )
    _loaded_engines.append("AlarmEngine")
except ImportError as e:
    logger.debug("Engine 8 (AlarmEngine) not available: %s", e)

# ---------------------------------------------------------------------------
# Engine 9: Dashboard
# ---------------------------------------------------------------------------
try:
    from .dashboard_engine import (
        DashboardConfig,
        DashboardEngine,
        DashboardResult,
        KPIMetric as DashboardKPIMetric,
        PanelConfig,
        PanelType,
        RefreshRate,
        TimeRange,
        TrendDirection,
        WidgetData,
        WidgetType,
    )
    _loaded_engines.append("DashboardEngine")
except ImportError as e:
    logger.debug("Engine 9 (DashboardEngine) not available: %s", e)

# ---------------------------------------------------------------------------
# Engine 10: Monitoring Reporting
# ---------------------------------------------------------------------------
try:
    from .monitoring_reporting_engine import (
        DistributionChannel,
        MonitoringReportingEngine,
        ReportConfig,
        ReportFormat,
        ReportOutput,
        ReportSchedule,
        ReportSection,
        ReportStatus,
        ReportType,
        ReportingResult,
        ScheduleFrequency,
    )
    _loaded_engines.append("MonitoringReportingEngine")
except ImportError as e:
    logger.debug("Engine 10 (MonitoringReportingEngine) not available: %s", e)


__all__: list[str] = [
    # Module metadata
    "__version__",
    "__pack_id__",
    "__pack_name__",
    "__engines_count__",
    # --- Engine 1: Meter Registry ---
    "MeterRegistryEngine",
    "MeterType",
    "MeterProtocol",
    "EnergyType",
    "MeterStatus",
    "ChannelType",
    "HierarchyLevel",
    "MeterChannel",
    "MeterConfig",
    "CalibrationRecord",
    "MeterHierarchy",
    "VirtualMeterDefinition",
    "MeterRegistryResult",
    # --- Engine 2: Data Acquisition ---
    "DataAcquisitionEngine",
    "AcquisitionMode",
    "DataStatus",
    "IntervalLength",
    "NormalizationMethod",
    "BufferStatus",
    "UnitCategory",
    "RawReading",
    "NormalizedReading",
    "AcquisitionSchedule",
    "AcquisitionConfig",
    "AcquisitionResult",
    # --- Engine 3: Data Validation ---
    "DataValidationEngine",
    "ValidationCheck",
    "ValidationSeverity",
    "QualityGrade",
    "CorrectionMethod",
    "DataSource",
    "ValidationRule",
    "ValidationFinding",
    "QualityScore",
    "DataCorrection",
    "ValidationReport",
    # --- Engine 4: Anomaly Detection ---
    "AnomalyDetectionEngine",
    "AnomalyMethod",
    "AnomalyType",
    "AnomalySeverity",
    "InvestigationStatus",
    "TimeContext",
    "AnomalyConfig",
    "AnomalyBaseline",
    "DetectedAnomaly",
    "InvestigationRecord",
    "AnomalyReport",
    # --- Engine 5: EnPI ---
    "EnPIEngine",
    "EnPIType",
    "RelevantVariable",
    "BaselineStatus",
    "SignificanceLevel",
    "PerformanceRating",
    "RegressionQuality",
    "EnPIDefinition",
    "RegressionModel",
    "EnergyBaseline",
    "EnPIValue",
    "CUSUMTracker",
    "EnPIResult",
    # --- Engine 6: Cost Allocation ---
    "CostAllocationEngine",
    "TenantAccount",
    "AllocationRule",
    "CostBreakdown",
    "BillingRecord",
    "AllocationResult",
    "AllocationMethod",
    "CostComponent",
    "TenantType",
    "BillingFrequency",
    "ReconciliationStatus",
    # --- Engine 7: Budget ---
    "BudgetEngine",
    "BudgetDefinition",
    "BudgetPeriod",
    "VarianceAnalysis",
    "RollingForecast",
    "BudgetResult",
    "BudgetMethod",
    "VarianceComponent",
    "BudgetStatus",
    "AlertThreshold",
    "ForecastMethod",
    # --- Engine 8: Alarm ---
    "AlarmEngine",
    "AlarmDefinition",
    "AlarmEvent",
    "SuppressionRule",
    "EscalationConfig",
    "AlarmReport",
    "AlarmPriority",
    "AlarmState",
    "SuppressionType",
    "EscalationLevel",
    "AlarmCategory",
    # --- Engine 9: Dashboard ---
    "DashboardEngine",
    "DashboardConfig",
    "PanelConfig",
    "WidgetData",
    "DashboardKPIMetric",
    "DashboardResult",
    "PanelType",
    "WidgetType",
    "TimeRange",
    "RefreshRate",
    "TrendDirection",
    # --- Engine 10: Monitoring Reporting ---
    "MonitoringReportingEngine",
    "ReportConfig",
    "ReportSchedule",
    "ReportSection",
    "ReportOutput",
    "ReportingResult",
    "ReportType",
    "ReportFormat",
    "ScheduleFrequency",
    "DistributionChannel",
    "ReportStatus",
]


def get_loaded_engines() -> list[str]:
    """Return list of engine class names that loaded successfully."""
    return list(_loaded_engines)


logger.info(
    "PACK-039 Energy Monitoring engines: %d/%d loaded",
    len(_loaded_engines),
    __engines_count__,
)
