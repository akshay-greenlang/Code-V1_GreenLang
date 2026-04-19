# -*- coding: utf-8 -*-
"""
PACK-033 Quick Wins Identifier Pack - Integration Layer
============================================================

Phase 4 integration layer for the Quick Wins Identifier Pack that provides
quick win pipeline orchestration, MRV emissions bridging, DATA agent
routing, PACK-031/032 data import, utility rebate matching, BMS/SCADA
data ingestion, weather normalization, 15-category health verification,
8-step facility setup wizard, and alert management.

Components:
    - QuickWinsOrchestrator: 10-phase pipeline with DAG dependency
      resolution, parallel execution, retry with exponential backoff,
      and SHA-256 provenance tracking
    - MRVQuickWinsBridge: Routes quick win savings data to MRV agents
      (Stationary Combustion, Refrigerants, Scope 2, Category 3) and
      converts energy savings to avoided emissions (tCO2e)
    - DataQuickWinsBridge: Routes data intake to DATA agents for
      equipment data, utility bills, quality profiling, and
      validation rule enforcement
    - Pack031Bridge: Imports energy audit results, equipment efficiency
      data, and energy baselines from PACK-031
    - Pack032Bridge: Imports building assessment results, zone data,
      and HVAC profiles from PACK-032
    - UtilityRebateBridge: Searches utility rebate programs, submits
      applications, and tracks rebate status
    - BMSDataBridge: BACnet/Modbus/OPC-UA data model adapters,
      real-time meter reading ingestion, and alarm integration
    - WeatherBridge: HDD/CDD calculation, TMY data, climate zone
      determination, and weather-normalized consumption
    - HealthCheck: 15-category system health verification
    - SetupWizard: 8-step guided facility configuration with 8
      facility presets
    - AlertBridge: Multi-channel notification and alert management

Architecture:
    External Systems --> Integration Bridges --> Platform Components
                              |
                              v
    PACK-033 Quick Wins <-- Composition <-- Zero Duplication
                              |
                              v
    MRV Agents <-- DATA Agents <-- FOUND Agents

Platform Integrations:
    - greenlang/agents/mrv/* (MRV-001, 002, 009-012, 016)
    - greenlang/agents/data/* (DATA-002, 003, 010, 019)
    - greenlang/agents/foundation/* (10 FOUND agents)
    - packs/energy-efficiency/PACK-031 (Industrial Energy Audit)
    - packs/energy-efficiency/PACK-032 (Building Energy Assessment)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-033 Quick Wins Identifier
Status: Production Ready
"""

__version__ = "1.0.0"
__pack_id__ = "PACK-033"
__pack_name__ = "Quick Wins Identifier Pack"

# ---------------------------------------------------------------------------
# Pack Orchestrator
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_033_quick_wins_identifier.integrations.pack_orchestrator import (
    ExecutionStatus,
    FacilityType,
    OrchestratorConfig,
    OrchestratorPhase,
    PARALLEL_PHASE_GROUPS,
    PHASE_DEPENDENCIES,
    PHASE_EXECUTION_ORDER,
    PhaseProvenance,
    PhaseResult,
    PipelineResult,
    QuickWinsOrchestrator,
    RetryConfig,
)

# ---------------------------------------------------------------------------
# MRV Quick Wins Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_033_quick_wins_identifier.integrations.mrv_quickwins_bridge import (
    DEFAULT_EMISSION_FACTORS,
    MeasureCategory,
    MRVQuickWinsBridge,
    MRVRouteConfig,
    MRVScope,
    RoutingResult,
    SavingsToEmissionsMapping,
)

# ---------------------------------------------------------------------------
# Data Quick Wins Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_033_quick_wins_identifier.integrations.data_quickwins_bridge import (
    DataQualityCheck,
    DataQuickWinsBridge,
    DataRouteConfig,
    DataRoutingResult,
    QuickWinDataSource,
)

# ---------------------------------------------------------------------------
# PACK-031 Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_033_quick_wins_identifier.integrations.pack031_bridge import (
    AuditDataImport,
    AuditImportConfig,
    Pack031Bridge,
)

# ---------------------------------------------------------------------------
# PACK-032 Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_033_quick_wins_identifier.integrations.pack032_bridge import (
    AssessmentImportConfig,
    BuildingDataImport,
    Pack032Bridge,
)

# ---------------------------------------------------------------------------
# Utility Rebate Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_033_quick_wins_identifier.integrations.utility_rebate_bridge import (
    ApplicationStatus,
    ApplicationTracker,
    ProgramSearchResult,
    ProgramType,
    UtilityAPIConfig,
    UtilityRebateBridge,
)

# ---------------------------------------------------------------------------
# BMS Data Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_033_quick_wins_identifier.integrations.bms_data_bridge import (
    AlarmEvent,
    BMSConfig,
    BMSDataBridge,
    ConnectionStatus,
    DataPoint,
    DataPointType,
    MeterReading,
    ProtocolType,
)

# ---------------------------------------------------------------------------
# Weather Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_033_quick_wins_identifier.integrations.weather_bridge import (
    ClimateNormalization,
    DegreeDayData,
    WeatherBridge,
    WeatherConfig,
)

# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_033_quick_wins_identifier.integrations.health_check import (
    CheckCategory,
    ComponentHealth,
    HealthCheck,
    HealthCheckConfig,
    HealthCheckResult,
    HealthSeverity,
    HealthStatus,
)

# ---------------------------------------------------------------------------
# Setup Wizard
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_033_quick_wins_identifier.integrations.setup_wizard import (
    FacilitySetup,
    PresetConfig,
    SetupResult,
    SetupWizard,
    SetupWizardStep,
    StepStatus,
    WizardState,
    WizardStepState,
)

# ---------------------------------------------------------------------------
# Alert Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_033_quick_wins_identifier.integrations.alert_bridge import (
    Alert,
    AlertBridge,
    AlertChannel,
    AlertConfig,
    AlertRule,
    AlertSeverity,
    AlertType,
    NotificationResult,
)

__all__ = [
    # Module metadata
    "__version__",
    "__pack_id__",
    "__pack_name__",
    # --- Pack Orchestrator ---
    "QuickWinsOrchestrator",
    "OrchestratorConfig",
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
    # --- MRV Quick Wins Bridge ---
    "MRVQuickWinsBridge",
    "MRVRouteConfig",
    "MeasureCategory",
    "MRVScope",
    "RoutingResult",
    "SavingsToEmissionsMapping",
    "DEFAULT_EMISSION_FACTORS",
    # --- Data Quick Wins Bridge ---
    "DataQuickWinsBridge",
    "DataRouteConfig",
    "DataRoutingResult",
    "DataQualityCheck",
    "QuickWinDataSource",
    # --- PACK-031 Bridge ---
    "Pack031Bridge",
    "AuditImportConfig",
    "AuditDataImport",
    # --- PACK-032 Bridge ---
    "Pack032Bridge",
    "AssessmentImportConfig",
    "BuildingDataImport",
    # --- Utility Rebate Bridge ---
    "UtilityRebateBridge",
    "UtilityAPIConfig",
    "ProgramSearchResult",
    "ApplicationTracker",
    "ApplicationStatus",
    "ProgramType",
    # --- BMS Data Bridge ---
    "BMSDataBridge",
    "BMSConfig",
    "DataPoint",
    "DataPointType",
    "ProtocolType",
    "ConnectionStatus",
    "MeterReading",
    "AlarmEvent",
    # --- Weather Bridge ---
    "WeatherBridge",
    "WeatherConfig",
    "DegreeDayData",
    "ClimateNormalization",
    # --- Health Check ---
    "HealthCheck",
    "HealthCheckConfig",
    "HealthCheckResult",
    "ComponentHealth",
    "HealthSeverity",
    "HealthStatus",
    "CheckCategory",
    # --- Setup Wizard ---
    "SetupWizard",
    "SetupWizardStep",
    "StepStatus",
    "FacilitySetup",
    "PresetConfig",
    "WizardStepState",
    "WizardState",
    "SetupResult",
    # --- Alert Bridge ---
    "AlertBridge",
    "AlertConfig",
    "Alert",
    "AlertRule",
    "AlertSeverity",
    "AlertChannel",
    "AlertType",
    "NotificationResult",
]
