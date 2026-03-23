# -*- coding: utf-8 -*-
"""
PACK-040 Measurement & Verification Pack - Integration Layer
===============================================================

Phase 4 integration layer for the M&V Pack that provides pipeline
orchestration, MRV emissions bridging, DATA agent routing, pack data
import bridges (PACK-031/032/033/039), weather service integration,
utility billing data import, 20-category health verification, 9-step
setup wizard, and multi-channel alert management.

Components:
    - MVOrchestrator: 12-phase pipeline with DAG dependency resolution,
      parallel execution, retry with exponential backoff, and SHA-256
      provenance tracking
    - MRVBridge: Routes verified savings to MRV agents (MRV-001/009/010)
      for Scope 1 and Scope 2 emissions reduction verification
    - DataBridge: Routes data intake to DATA agents for meter readings,
      utility bills, quality profiling, gap filling, and freshness
    - Pack031Bridge: Imports industrial audit baselines, ECM specs,
      equipment data from PACK-031
    - Pack032Bridge: Imports building assessment data, retrofit specs,
      envelope and HVAC profiles from PACK-032
    - Pack033Bridge: Imports quick win measures and estimated savings
      for verification from PACK-033
    - Pack039Bridge: Imports monitoring data, EnPI baselines, meter
      registry from PACK-039 Energy Monitoring
    - WeatherServiceBridge: Weather data for HDD/CDD, TMY data,
      station selection, balance point optimization
    - UtilityDataBridge: Utility billing data import, rate schedules,
      demand data, Green Button format support
    - HealthCheck: 20-category system health verification
    - SetupWizard: 9-step M&V project configuration with 8 facility
      presets
    - AlertBridge: Multi-channel alerting for savings degradation,
      compliance deadlines, baseline drift, meter calibration

Architecture:
    External Systems --> Integration Bridges --> Platform Components
                              |
                              v
    PACK-040 M&V <-- Composition <-- Zero Duplication
                              |
                              v
    MRV Agents <-- DATA Agents <-- FOUND Agents

Platform Integrations:
    - greenlang/agents/mrv/* (MRV-001, 009, 010)
    - greenlang/agents/data/* (DATA-001, 002, 003, 010, 014, 016)
    - greenlang/agents/foundation/* (10 FOUND agents)
    - packs/energy-efficiency/PACK-031 (Industrial Energy Audit)
    - packs/energy-efficiency/PACK-032 (Building Energy Assessment)
    - packs/energy-efficiency/PACK-033 (Quick Wins Identifier)
    - packs/energy-efficiency/PACK-039 (Energy Monitoring)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-040 Measurement & Verification
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack_id__: str = "PACK-040"
__pack_name__: str = "M&V Pack"
__integrations_count__: int = 12

_loaded_integrations: list[str] = []

# ---------------------------------------------------------------------------
# Pack Orchestrator
# ---------------------------------------------------------------------------
try:
    from .pack_orchestrator import (
        MVOrchestrator,
        ExecutionStatus,
        FacilityType,
        OrchestratorPhase,
        PARALLEL_PHASE_GROUPS,
        PHASE_DEPENDENCIES,
        PHASE_EXECUTION_ORDER,
        PhaseProvenance,
        PhaseResult,
        PipelineConfig,
        PipelineResult,
        RetryConfig,
        IPMVPOption,
        BaselineModelType,
        ComplianceFramework,
    )
    _loaded_integrations.append("MVOrchestrator")
except ImportError as e:
    logger.debug("Integration 1 (MVOrchestrator) not available: %s", e)

# ---------------------------------------------------------------------------
# MRV Bridge
# ---------------------------------------------------------------------------
try:
    from .mrv_bridge import (
        MRVBridge,
        MRVRequest,
        MRVResponse,
        MRVRouteConfig,
        MRVScope,
        MVEmissionCategory,
        EmissionFactorSet,
        EmissionFactorSource,
        MeterType,
        AccountingMethod,
        SavingsType,
        GRID_EMISSION_FACTORS,
        GAS_EMISSION_FACTORS,
    )
    _loaded_integrations.append("MRVBridge")
except ImportError as e:
    logger.debug("Integration 2 (MRVBridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Data Bridge
# ---------------------------------------------------------------------------
try:
    from .data_bridge import (
        DataBridge,
        DataQualityReport,
        DataRequest,
        DataResponse,
        DataRouteConfig,
        MVDataSource,
        DataFormat,
        DataAgentTarget,
        QualityLevel,
        GapFillMethod,
    )
    _loaded_integrations.append("DataBridge")
except ImportError as e:
    logger.debug("Integration 3 (DataBridge) not available: %s", e)

# ---------------------------------------------------------------------------
# PACK-031 Bridge
# ---------------------------------------------------------------------------
try:
    from .pack031_bridge import (
        Pack031Bridge,
        Pack031ImportResult,
        AuditBaseline,
        AuditLevel,
        ECMSpec,
        ECMCategory,
        EquipmentData,
        EquipmentStatus,
        BaselineSource,
        ImportStatus,
    )
    _loaded_integrations.append("Pack031Bridge")
except ImportError as e:
    logger.debug("Integration 4 (Pack031Bridge) not available: %s", e)

# ---------------------------------------------------------------------------
# PACK-032 Bridge
# ---------------------------------------------------------------------------
try:
    from .pack032_bridge import (
        Pack032Bridge,
        Pack032ImportResult,
        BuildingAssessment,
        BuildingType,
        RetrofitSpec,
        RetrofitCategory,
        AssessmentLevel,
        EnvelopeData,
        EnvelopeComponent,
        HVACProfile,
        HVACType,
    )
    _loaded_integrations.append("Pack032Bridge")
except ImportError as e:
    logger.debug("Integration 5 (Pack032Bridge) not available: %s", e)

# ---------------------------------------------------------------------------
# PACK-033 Bridge
# ---------------------------------------------------------------------------
try:
    from .pack033_bridge import (
        Pack033Bridge,
        Pack033ImportResult,
        QuickWinMeasure,
        QuickWinCategory,
        QuickWinSavingsSnapshot,
        ImplementationStatus,
        PersistenceRisk,
        VerificationMethod,
        MeasureComplexity,
    )
    _loaded_integrations.append("Pack033Bridge")
except ImportError as e:
    logger.debug("Integration 6 (Pack033Bridge) not available: %s", e)

# ---------------------------------------------------------------------------
# PACK-039 Bridge
# ---------------------------------------------------------------------------
try:
    from .pack039_bridge import (
        Pack039Bridge,
        Pack039ImportResult,
        MeterRegistryEntry,
        MeterReading as MonitoringMeterReading,
        EnPIBaseline,
        MonitoringMeterType,
        DataInterval,
        DataQualityFlag,
        EnPICategory,
        MeterProtocol,
    )
    _loaded_integrations.append("Pack039Bridge")
except ImportError as e:
    logger.debug("Integration 7 (Pack039Bridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Weather Service Bridge
# ---------------------------------------------------------------------------
try:
    from .weather_service_bridge import (
        WeatherServiceBridge,
        WeatherDataResult,
        WeatherStation,
        DegreeDayRecord,
        TMYData,
        WeatherSource,
        TemperatureUnit,
        DegreeDayType,
        WeatherQuality,
        NormalizationMethod,
    )
    _loaded_integrations.append("WeatherServiceBridge")
except ImportError as e:
    logger.debug("Integration 8 (WeatherServiceBridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Utility Data Bridge
# ---------------------------------------------------------------------------
try:
    from .utility_data_bridge import (
        UtilityDataBridge,
        UtilityImportResult,
        UtilityBill,
        RateSchedule,
        DemandData,
        UtilityType,
        RateType,
        BillFormat,
        DemandType,
        BillStatus,
    )
    _loaded_integrations.append("UtilityDataBridge")
except ImportError as e:
    logger.debug("Integration 9 (UtilityDataBridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------
try:
    from .health_check import (
        HealthCheck,
        HealthCheckConfig,
        SystemHealth,
        HealthResult,
        HealthSeverity,
        HealthStatus,
        HealthCategory,
        CheckType,
    )
    _loaded_integrations.append("HealthCheck")
except ImportError as e:
    logger.debug("Integration 10 (HealthCheck) not available: %s", e)

# ---------------------------------------------------------------------------
# Setup Wizard
# ---------------------------------------------------------------------------
try:
    from .setup_wizard import (
        SetupWizard,
        WizardStep,
        StepStatus,
        FacilityMVProfile,
        PresetConfig,
        WizardStepState,
        WizardConfig,
        SetupResult,
        FacilityMVType,
        MVTier,
        FACILITY_PRESETS,
    )
    _loaded_integrations.append("SetupWizard")
except ImportError as e:
    logger.debug("Integration 11 (SetupWizard) not available: %s", e)

# ---------------------------------------------------------------------------
# Alert Bridge
# ---------------------------------------------------------------------------
try:
    from .alert_bridge import (
        AlertBridge,
        AlertChannel,
        AlertConfig,
        AlertMessage,
        AlertSeverity,
        AlertType,
        EscalationLevel,
        EscalationRule,
        NotificationResult,
        AlertStatus,
        DEFAULT_ESCALATION_MAP,
        ESCALATION_CHANNELS,
    )
    _loaded_integrations.append("AlertBridge")
except ImportError as e:
    logger.debug("Integration 12 (AlertBridge) not available: %s", e)


__all__: list[str] = [
    # Module metadata
    "__version__",
    "__pack_id__",
    "__pack_name__",
    "__integrations_count__",
    # --- Pack Orchestrator ---
    "MVOrchestrator",
    "PipelineConfig",
    "RetryConfig",
    "OrchestratorPhase",
    "FacilityType",
    "ExecutionStatus",
    "PhaseProvenance",
    "PhaseResult",
    "PipelineResult",
    "IPMVPOption",
    "BaselineModelType",
    "ComplianceFramework",
    "PHASE_DEPENDENCIES",
    "PHASE_EXECUTION_ORDER",
    "PARALLEL_PHASE_GROUPS",
    # --- MRV Bridge ---
    "MRVBridge",
    "MRVRouteConfig",
    "MRVRequest",
    "MRVResponse",
    "EmissionFactorSet",
    "EmissionFactorSource",
    "MVEmissionCategory",
    "MRVScope",
    "MeterType",
    "AccountingMethod",
    "SavingsType",
    "GRID_EMISSION_FACTORS",
    "GAS_EMISSION_FACTORS",
    # --- Data Bridge ---
    "DataBridge",
    "DataRouteConfig",
    "DataRequest",
    "DataResponse",
    "DataQualityReport",
    "MVDataSource",
    "DataFormat",
    "DataAgentTarget",
    "QualityLevel",
    "GapFillMethod",
    # --- PACK-031 Bridge ---
    "Pack031Bridge",
    "Pack031ImportResult",
    "AuditBaseline",
    "AuditLevel",
    "ECMSpec",
    "ECMCategory",
    "EquipmentData",
    "EquipmentStatus",
    "BaselineSource",
    "ImportStatus",
    # --- PACK-032 Bridge ---
    "Pack032Bridge",
    "Pack032ImportResult",
    "BuildingAssessment",
    "BuildingType",
    "RetrofitSpec",
    "RetrofitCategory",
    "AssessmentLevel",
    "EnvelopeData",
    "EnvelopeComponent",
    "HVACProfile",
    "HVACType",
    # --- PACK-033 Bridge ---
    "Pack033Bridge",
    "Pack033ImportResult",
    "QuickWinMeasure",
    "QuickWinCategory",
    "QuickWinSavingsSnapshot",
    "ImplementationStatus",
    "PersistenceRisk",
    "VerificationMethod",
    "MeasureComplexity",
    # --- PACK-039 Bridge ---
    "Pack039Bridge",
    "Pack039ImportResult",
    "MeterRegistryEntry",
    "MonitoringMeterReading",
    "EnPIBaseline",
    "MonitoringMeterType",
    "DataInterval",
    "DataQualityFlag",
    "EnPICategory",
    "MeterProtocol",
    # --- Weather Service Bridge ---
    "WeatherServiceBridge",
    "WeatherDataResult",
    "WeatherStation",
    "DegreeDayRecord",
    "TMYData",
    "WeatherSource",
    "TemperatureUnit",
    "DegreeDayType",
    "WeatherQuality",
    "NormalizationMethod",
    # --- Utility Data Bridge ---
    "UtilityDataBridge",
    "UtilityImportResult",
    "UtilityBill",
    "RateSchedule",
    "DemandData",
    "UtilityType",
    "RateType",
    "BillFormat",
    "DemandType",
    "BillStatus",
    # --- Health Check ---
    "HealthCheck",
    "HealthCheckConfig",
    "SystemHealth",
    "HealthResult",
    "HealthSeverity",
    "HealthStatus",
    "HealthCategory",
    "CheckType",
    # --- Setup Wizard ---
    "SetupWizard",
    "WizardStep",
    "StepStatus",
    "FacilityMVProfile",
    "PresetConfig",
    "WizardStepState",
    "WizardConfig",
    "SetupResult",
    "FacilityMVType",
    "MVTier",
    "FACILITY_PRESETS",
    # --- Alert Bridge ---
    "AlertBridge",
    "AlertConfig",
    "AlertMessage",
    "EscalationRule",
    "AlertSeverity",
    "AlertChannel",
    "AlertType",
    "EscalationLevel",
    "NotificationResult",
    "AlertStatus",
    "DEFAULT_ESCALATION_MAP",
    "ESCALATION_CHANNELS",
]


def get_loaded_integrations() -> list[str]:
    """Return list of integration class names that loaded successfully."""
    return list(_loaded_integrations)


logger.info(
    "PACK-040 M&V integrations: %d/%d loaded",
    len(_loaded_integrations),
    __integrations_count__,
)
