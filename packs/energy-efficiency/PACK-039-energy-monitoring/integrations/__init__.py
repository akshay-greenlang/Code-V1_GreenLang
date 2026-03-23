# -*- coding: utf-8 -*-
"""
PACK-039 Energy Monitoring Pack - Integration Layer
====================================================

Phase 4 integration layer for the Energy Monitoring Pack that provides
pipeline orchestration, MRV emissions bridging, DATA agent routing,
meter protocol abstraction, AMI smart meter data import, BMS trend
extraction, IoT sensor integration, PACK-036/038 data import, 20-category
health verification, 9-step setup wizard, and multi-channel alert management.

Components:
    - MonitoringOrchestrator: 12-phase pipeline with DAG dependency
      resolution, parallel execution, retry with exponential backoff,
      and SHA-256 provenance tracking
    - MRVBridge: Routes metered energy to MRV agents (MRV-001/009/010)
      for Scope 1 and Scope 2 emissions from metered consumption
    - DataBridge: Routes data intake to DATA agents for meter readings,
      utility bills, quality profiling, gap filling, and freshness monitoring
    - MeterProtocolBridge: Protocol abstraction for Modbus RTU/TCP,
      BACnet IP/MSTP, MQTT, OPC-UA with connection pooling and retry
    - AMIBridge: Smart meter AMI data import, Green Button XML/CSV
      parsing, demand register and interval data normalization
    - BMSBridge: BMS trend data extraction for HVAC, lighting,
      equipment schedules, setpoints, and run hours
    - IoTSensorBridge: IoT platform integration (MQTT, HTTP, CoAP) for
      temperature, humidity, occupancy, and light level sensors
    - Pack036Bridge: Imports rate structures, TOU periods, and tariff
      data from PACK-036 for cost allocation
    - Pack038Bridge: Imports peak events, BESS dispatch data, and
      demand charge analysis from PACK-038
    - HealthCheck: 20-category system health verification
    - SetupWizard: 9-step guided energy monitoring configuration with
      8 facility presets
    - AlertBridge: Multi-channel alerting with escalation rules for
      anomaly alerts, budget warnings, EnPI deviations, and data quality

Architecture:
    External Systems --> Integration Bridges --> Platform Components
                              |
                              v
    PACK-039 Energy Monitoring <-- Composition <-- Zero Duplication
                              |
                              v
    MRV Agents <-- DATA Agents <-- FOUND Agents

Platform Integrations:
    - greenlang/agents/mrv/* (MRV-001, 009, 010)
    - greenlang/agents/data/* (DATA-001, 002, 003, 010, 014, 016)
    - greenlang/agents/foundation/* (10 FOUND agents)
    - packs/energy-efficiency/PACK-036 (Utility Analysis)
    - packs/energy-efficiency/PACK-038 (Peak Shaving)

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
__integrations_count__: int = 12

_loaded_integrations: list[str] = []

# ---------------------------------------------------------------------------
# Pack Orchestrator
# ---------------------------------------------------------------------------
try:
    from .pack_orchestrator import (
        MonitoringOrchestrator,
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
    )
    _loaded_integrations.append("MonitoringOrchestrator")
except ImportError as e:
    logger.debug("Integration 1 (MonitoringOrchestrator) not available: %s", e)

# ---------------------------------------------------------------------------
# MRV Bridge
# ---------------------------------------------------------------------------
try:
    from .mrv_bridge import (
        EmissionFactorSet,
        EmissionFactorSource,
        GRID_EMISSION_FACTORS,
        MRVBridge,
        MRVRequest,
        MRVResponse,
        MRVRouteConfig,
        MRVScope,
        EMEmissionCategory,
        MeterType,
        AccountingMethod,
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
        EMDataSource,
    )
    _loaded_integrations.append("DataBridge")
except ImportError as e:
    logger.debug("Integration 3 (DataBridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Meter Protocol Bridge
# ---------------------------------------------------------------------------
try:
    from .meter_protocol_bridge import (
        ConnectionPool,
        ConnectionState,
        DataType,
        MeterProtocol,
        MeterProtocolBridge,
        MeterReading,
        ProtocolConfig,
        ReadQuality,
        RegisterMapping,
        RegisterType,
    )
    _loaded_integrations.append("MeterProtocolBridge")
except ImportError as e:
    logger.debug("Integration 4 (MeterProtocolBridge) not available: %s", e)

# ---------------------------------------------------------------------------
# AMI Bridge
# ---------------------------------------------------------------------------
try:
    from .ami_bridge import (
        AMIBridge,
        AMIConfig,
        AMIDataFormat,
        AMIImportResult,
        DataQuality as AMIDataQuality,
        DemandRegister,
        DemandRegisterType,
        IntervalData,
        IntervalLength,
        MeterChannel,
        MeterConfig,
    )
    _loaded_integrations.append("AMIBridge")
except ImportError as e:
    logger.debug("Integration 5 (AMIBridge) not available: %s", e)

# ---------------------------------------------------------------------------
# BMS Bridge
# ---------------------------------------------------------------------------
try:
    from .bms_bridge import (
        BMSBridge,
        BMSConfig,
        BMSProtocol,
        ConnectionStatus,
        EquipmentSchedule,
        EquipmentType,
        RunHoursReport,
        TrendCategory,
        TrendExtractResult,
        TrendPoint,
        TrendQuality,
    )
    _loaded_integrations.append("BMSBridge")
except ImportError as e:
    logger.debug("Integration 6 (BMSBridge) not available: %s", e)

# ---------------------------------------------------------------------------
# IoT Sensor Bridge
# ---------------------------------------------------------------------------
try:
    from .iot_sensor_bridge import (
        IoTConfig,
        IoTProtocol,
        IoTSensorBridge,
        SensorBatchResult,
        SensorConfig,
        SensorLocation,
        SensorReading,
        SensorStatus,
        SensorType,
    )
    _loaded_integrations.append("IoTSensorBridge")
except ImportError as e:
    logger.debug("Integration 7 (IoTSensorBridge) not available: %s", e)

# ---------------------------------------------------------------------------
# PACK-036 Bridge
# ---------------------------------------------------------------------------
try:
    from .pack036_bridge import (
        DemandChargeProfile,
        DemandChargeType,
        Pack036Bridge,
        Pack036Config,
        RateStructure,
        RateType,
        TariffData,
        TOUPeriod,
    )
    _loaded_integrations.append("Pack036Bridge")
except ImportError as e:
    logger.debug("Integration 8 (Pack036Bridge) not available: %s", e)

# ---------------------------------------------------------------------------
# PACK-038 Bridge
# ---------------------------------------------------------------------------
try:
    from .pack038_bridge import (
        BESSDispatchData,
        BESSStatus,
        CPEventData,
        CPEventStatus,
        DemandChargeAnalysis,
        DemandChargeStatus,
        Pack038Bridge,
        Pack038Config,
        PeakEvent,
        PeakEventType,
    )
    _loaded_integrations.append("Pack038Bridge")
except ImportError as e:
    logger.debug("Integration 9 (Pack038Bridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------
try:
    from .health_check import (
        HealthCategory,
        HealthCheck,
        HealthCheckConfig,
        HealthResult,
        HealthSeverity,
        HealthStatus,
        SystemHealth,
    )
    _loaded_integrations.append("HealthCheck")
except ImportError as e:
    logger.debug("Integration 10 (HealthCheck) not available: %s", e)

# ---------------------------------------------------------------------------
# Setup Wizard
# ---------------------------------------------------------------------------
try:
    from .setup_wizard import (
        FacilityEMProfile,
        PresetConfig,
        SetupResult,
        SetupWizard,
        StepStatus,
        WizardConfig,
        WizardStep,
        WizardStepState,
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
    "MonitoringOrchestrator",
    "PipelineConfig",
    "RetryConfig",
    "OrchestratorPhase",
    "FacilityType",
    "ExecutionStatus",
    "PhaseProvenance",
    "PhaseResult",
    "PipelineResult",
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
    "EMEmissionCategory",
    "MRVScope",
    "MeterType",
    "AccountingMethod",
    "GRID_EMISSION_FACTORS",
    # --- Data Bridge ---
    "DataBridge",
    "DataRouteConfig",
    "DataRequest",
    "DataResponse",
    "DataQualityReport",
    "EMDataSource",
    # --- Meter Protocol Bridge ---
    "MeterProtocolBridge",
    "ProtocolConfig",
    "RegisterMapping",
    "MeterReading",
    "ConnectionPool",
    "MeterProtocol",
    "ConnectionState",
    "RegisterType",
    "DataType",
    "ReadQuality",
    # --- AMI Bridge ---
    "AMIBridge",
    "AMIConfig",
    "AMIImportResult",
    "IntervalData",
    "DemandRegister",
    "DemandRegisterType",
    "AMIDataFormat",
    "IntervalLength",
    "MeterChannel",
    "MeterConfig",
    # --- BMS Bridge ---
    "BMSBridge",
    "BMSConfig",
    "BMSProtocol",
    "TrendPoint",
    "TrendExtractResult",
    "EquipmentSchedule",
    "RunHoursReport",
    "TrendCategory",
    "EquipmentType",
    "TrendQuality",
    "ConnectionStatus",
    # --- IoT Sensor Bridge ---
    "IoTSensorBridge",
    "IoTConfig",
    "SensorReading",
    "SensorConfig",
    "SensorBatchResult",
    "IoTProtocol",
    "SensorType",
    "SensorLocation",
    "SensorStatus",
    # --- PACK-036 Bridge ---
    "Pack036Bridge",
    "Pack036Config",
    "RateStructure",
    "TariffData",
    "DemandChargeProfile",
    "RateType",
    "TOUPeriod",
    "DemandChargeType",
    # --- PACK-038 Bridge ---
    "Pack038Bridge",
    "Pack038Config",
    "PeakEvent",
    "BESSDispatchData",
    "DemandChargeAnalysis",
    "CPEventData",
    "PeakEventType",
    "BESSStatus",
    "DemandChargeStatus",
    "CPEventStatus",
    # --- Health Check ---
    "HealthCheck",
    "HealthCheckConfig",
    "SystemHealth",
    "HealthResult",
    "HealthSeverity",
    "HealthStatus",
    "HealthCategory",
    # --- Setup Wizard ---
    "SetupWizard",
    "WizardStep",
    "StepStatus",
    "FacilityEMProfile",
    "PresetConfig",
    "WizardStepState",
    "WizardConfig",
    "SetupResult",
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
]


def get_loaded_integrations() -> list[str]:
    """Return list of integration class names that loaded successfully."""
    return list(_loaded_integrations)


logger.info(
    "PACK-039 Energy Monitoring integrations: %d/%d loaded",
    len(_loaded_integrations),
    __integrations_count__,
)
