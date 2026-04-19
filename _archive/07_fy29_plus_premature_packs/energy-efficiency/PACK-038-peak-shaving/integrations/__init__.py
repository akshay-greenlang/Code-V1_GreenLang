# -*- coding: utf-8 -*-
"""
PACK-038 Peak Shaving Pack - Integration Layer
====================================================

Phase 4 integration layer for the Peak Shaving Pack that provides
pipeline orchestration, MRV emissions bridging, DATA agent routing,
AMI meter data integration, utility rate/tariff data, BESS control,
BMS load control, PACK-036/037 data import, 20-category health
verification, 9-step setup wizard, and multi-channel alert management.

Components:
    - PeakShavingOrchestrator: 12-phase pipeline with DAG dependency
      resolution, parallel execution, retry with exponential backoff,
      and SHA-256 provenance tracking
    - MRVBridge: Routes peak shaving data to MRV agents (MRV-009/010/013)
      using marginal emission factors for accurate peak-period accounting
    - DataBridge: Routes data intake to DATA agents for meter data,
      utility bills, quality profiling, gap filling, and reconciliation
    - MeterDataBridge: AMI integration for 15-minute interval data,
      Green Button parsing, and demand register tracking
    - UtilityRateBridge: Tariff data integration for demand charge
      structures, TOU periods, ratchet clauses, and CP schedules
    - BESSControlBridge: Battery management system integration for
      dispatch commands, SOC monitoring, and degradation tracking
    - BMSBridge: Building Management System integration for HVAC
      setpoint adjustment, lighting dimming, equipment curtailment
    - Pack036Bridge: Imports utility rate structures, TOU periods, and
      demand charge profiles from PACK-036
    - Pack037Bridge: Imports DR event schedules, revenue data, and
      coordinates BESS dispatch with DR events from PACK-037
    - HealthCheck: 20-category system health verification
    - SetupWizard: 9-step guided peak shaving configuration with
      8 facility presets
    - AlertBridge: Multi-channel alerting with escalation rules for
      peak warnings, CP alerts, BESS dispatch, and ratchet prevention

Architecture:
    External Systems --> Integration Bridges --> Platform Components
                              |
                              v
    PACK-038 Peak Shaving <-- Composition <-- Zero Duplication
                              |
                              v
    MRV Agents <-- DATA Agents <-- FOUND Agents

Platform Integrations:
    - greenlang/agents/mrv/* (MRV-009, 010, 013)
    - greenlang/agents/data/* (DATA-001, 002, 003, 010, 014, 015)
    - greenlang/agents/foundation/* (10 FOUND agents)
    - packs/energy-efficiency/PACK-036 (Utility Analysis)
    - packs/energy-efficiency/PACK-037 (Demand Response)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-038 Peak Shaving
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack_id__: str = "PACK-038"
__pack_name__: str = "Peak Shaving Pack"
__integrations_count__: int = 12

_loaded_integrations: list[str] = []

# ---------------------------------------------------------------------------
# Pack Orchestrator
# ---------------------------------------------------------------------------
try:
    from .pack_orchestrator import (
        PeakShavingOrchestrator,
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
    _loaded_integrations.append("PeakShavingOrchestrator")
except ImportError as e:
    logger.debug("Integration 1 (PeakShavingOrchestrator) not available: %s", e)

# ---------------------------------------------------------------------------
# MRV Bridge
# ---------------------------------------------------------------------------
try:
    from .mrv_bridge import (
        EmissionFactorSet,
        EmissionFactorSource,
        MARGINAL_EMISSION_FACTORS,
        MRVBridge,
        MRVRequest,
        MRVResponse,
        MRVRouteConfig,
        MRVScope,
        PSEmissionCategory,
        PeakPeriodType,
        ReductionMethod,
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
        PSDataSource,
    )
    _loaded_integrations.append("DataBridge")
except ImportError as e:
    logger.debug("Integration 3 (DataBridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Meter Data Bridge
# ---------------------------------------------------------------------------
try:
    from .meter_data_bridge import (
        DataQuality,
        DemandRegister,
        DemandRegisterType,
        IntervalData,
        IntervalLength,
        MeterChannel,
        MeterConfig,
        MeterDataBridge,
        MeterProtocol,
        MeterReading,
    )
    _loaded_integrations.append("MeterDataBridge")
except ImportError as e:
    logger.debug("Integration 4 (MeterDataBridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Utility Rate Bridge
# ---------------------------------------------------------------------------
try:
    from .utility_rate_bridge import (
        DemandChargeCategory,
        DemandChargeRate,
        RatchetClause,
        RatchetType,
        RateStructureType,
        TOUPeriod as URBTOUPeriod,
        TOUPeriodType,
        TariffSchedule,
        TariffSource,
        UtilityRateBridge,
        UtilityRateConfig,
    )
    _loaded_integrations.append("UtilityRateBridge")
except ImportError as e:
    logger.debug("Integration 5 (UtilityRateBridge) not available: %s", e)

# ---------------------------------------------------------------------------
# BESS Control Bridge
# ---------------------------------------------------------------------------
try:
    from .bess_control_bridge import (
        BESSConfig,
        BESSConnectionStatus,
        BESSControlBridge,
        BESSOperatingMode,
        BESSProtocol,
        BESSStatus,
        BatteryChemistry,
        DegradationReport,
        DispatchCommand,
        DispatchResult,
    )
    _loaded_integrations.append("BESSControlBridge")
except ImportError as e:
    logger.debug("Integration 6 (BESSControlBridge) not available: %s", e)

# ---------------------------------------------------------------------------
# BMS Bridge
# ---------------------------------------------------------------------------
try:
    from .bms_bridge import (
        BMSBridge,
        BMSConfig,
        BMSEndpoint,
        BMSProtocol,
        ConnectionStatus,
        ControlAction,
        ControlCommand,
        ControlResponse,
        ControlStatus,
        LoadCategory,
    )
    _loaded_integrations.append("BMSBridge")
except ImportError as e:
    logger.debug("Integration 7 (BMSBridge) not available: %s", e)

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
# PACK-037 Bridge
# ---------------------------------------------------------------------------
try:
    from .pack037_bridge import (
        BESSReservation,
        CoordinationPriority,
        CoordinationResult,
        DREventSchedule,
        DREventStatus,
        DRProgramType,
        DRRevenueData,
        Pack037Bridge,
        Pack037Config,
        StackingAnalysis,
        StackingStrategy,
    )
    _loaded_integrations.append("Pack037Bridge")
except ImportError as e:
    logger.debug("Integration 9 (Pack037Bridge) not available: %s", e)

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
        FacilityPSProfile,
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
    "PeakShavingOrchestrator",
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
    "PSEmissionCategory",
    "MRVScope",
    "PeakPeriodType",
    "ReductionMethod",
    "MARGINAL_EMISSION_FACTORS",
    # --- Data Bridge ---
    "DataBridge",
    "DataRouteConfig",
    "DataRequest",
    "DataResponse",
    "DataQualityReport",
    "PSDataSource",
    # --- Meter Data Bridge ---
    "MeterDataBridge",
    "MeterConfig",
    "MeterReading",
    "IntervalData",
    "DemandRegister",
    "DemandRegisterType",
    "MeterProtocol",
    "IntervalLength",
    "MeterChannel",
    "DataQuality",
    # --- Utility Rate Bridge ---
    "UtilityRateBridge",
    "UtilityRateConfig",
    "TariffSchedule",
    "TariffSource",
    "DemandChargeRate",
    "DemandChargeCategory",
    "RatchetClause",
    "RatchetType",
    "RateStructureType",
    "TOUPeriodType",
    # --- BESS Control Bridge ---
    "BESSControlBridge",
    "BESSConfig",
    "BESSStatus",
    "DispatchCommand",
    "DispatchResult",
    "DegradationReport",
    "BESSProtocol",
    "BatteryChemistry",
    "BESSOperatingMode",
    "BESSConnectionStatus",
    # --- BMS Bridge ---
    "BMSBridge",
    "BMSConfig",
    "BMSEndpoint",
    "BMSProtocol",
    "ControlCommand",
    "ControlResponse",
    "ControlAction",
    "ControlStatus",
    "ConnectionStatus",
    "LoadCategory",
    # --- PACK-036 Bridge ---
    "Pack036Bridge",
    "Pack036Config",
    "RateStructure",
    "TariffData",
    "DemandChargeProfile",
    "RateType",
    "TOUPeriod",
    "DemandChargeType",
    # --- PACK-037 Bridge ---
    "Pack037Bridge",
    "Pack037Config",
    "DREventSchedule",
    "DRRevenueData",
    "StackingAnalysis",
    "CoordinationResult",
    "DRProgramType",
    "DREventStatus",
    "CoordinationPriority",
    "BESSReservation",
    "StackingStrategy",
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
    "FacilityPSProfile",
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
    "PACK-038 Peak Shaving integrations: %d/%d loaded",
    len(_loaded_integrations),
    __integrations_count__,
)
