# -*- coding: utf-8 -*-
"""
PACK-031 Industrial Energy Audit Pack - Integration Layer
============================================================

Phase 4 integration layer for the Industrial Energy Audit Pack that provides
energy audit pipeline orchestration, MRV emissions bridging, DATA agent
routing, EED compliance management, ISO 50001 EnMS integration, BMS/SCADA
data ingestion, utility metering, equipment registry, weather normalization,
EU ETS carbon market integration, 22-category health verification, and
8-step facility setup wizard.

Components:
    - IndustrialEnergyAuditOrchestrator: 12-phase pipeline with DAG
      dependency resolution, parallel execution, retry with exponential
      backoff, and SHA-256 provenance tracking
    - MRVEnergyBridge: Routes industrial energy data to MRV agents
      (Stationary Combustion, Mobile Combustion, Scope 2, Category 3)
      and converts energy savings to avoided emissions (tCO2e)
    - DataEnergyBridge: Routes data intake to DATA agents for meter
      data, ERP procurement, quality profiling, gap filling, and
      validation rule enforcement
    - EEDComplianceBridge: EU Energy Efficiency Directive Article 8
      obligation assessment, 4-year audit cycle scheduling, ISO 50001/
      EMAS exemption tracking, and national authority reporting
    - ISO50001Bridge: Energy Management System documentation, internal
      audit scheduling, nonconformity tracking, corrective actions,
      management review aggregation, and certification tracking
    - BMSSCADABridge: BACnet/Modbus/OPC-UA data model adapters,
      real-time meter reading ingestion, SCADA data point mapping,
      alarm integration, and unit conversion
    - UtilityMeteringBridge: AMI data ingestion, sub-meter hierarchy,
      interval data processing, virtual meters, bill reconciliation,
      and demand profile analysis
    - EquipmentRegistryBridge: Asset management/CMMS integration,
      nameplate data, maintenance schedules, run-hour tracking,
      equipment lifecycle, and replacement assessment
    - WeatherNormalizationBridge: HDD/CDD calculation, TMY data,
      climate zone determination, weather-normalized baselines,
      and seasonal adjustment factors
    - HealthCheck: 22-category system health verification
    - SetupWizard: 8-step guided facility configuration with 6
      industry presets
    - EUETSBridge: Installation permit tracking, free allocation
      analysis, carbon price impact on energy savings ROI, EU ETS
      benchmark comparison, and compliance cycle tracking

Architecture:
    External Systems --> Integration Bridges --> Platform Components
                              |
                              v
    PACK-031 Energy Audit <-- Composition <-- Zero Duplication
                              |
                              v
    MRV Agents <-- DATA Agents <-- FOUND Agents

Platform Integrations:
    - greenlang/agents/mrv/* (MRV-001, 003, 009-012, 016)
    - greenlang/agents/data/* (DATA-002, 003, 010, 014, 019)
    - greenlang/agents/foundation/* (10 FOUND agents)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-031 Industrial Energy Audit
Status: Production Ready
"""

__version__ = "1.0.0"
__pack_id__ = "PACK-031"
__pack_name__ = "Industrial Energy Audit Pack"

# ---------------------------------------------------------------------------
# Pack Orchestrator
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_031_industrial_energy_audit.integrations.pack_orchestrator import (
    AuditPipelinePhase,
    ExecutionStatus,
    IndustrialEnergyAuditOrchestrator,
    IndustrySector,
    OrchestratorConfig,
    PARALLEL_PHASE_GROUPS,
    PHASE_DEPENDENCIES,
    PHASE_EXECUTION_ORDER,
    PhaseProvenance,
    PhaseResult,
    PipelineResult,
    RetryConfig,
)

# ---------------------------------------------------------------------------
# MRV Energy Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_031_industrial_energy_audit.integrations.mrv_energy_bridge import (
    BatchRoutingResult,
    EnergySource,
    MRVAgentRoute,
    MRVEnergyBridge,
    MRVEnergyBridgeConfig,
    MRVScope,
    RoutingResult,
    SavingsConversionResult,
)

# ---------------------------------------------------------------------------
# Data Energy Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_031_industrial_energy_audit.integrations.data_energy_bridge import (
    DataAgentRoute,
    DataEnergyBridge,
    DataEnergyBridgeConfig,
    DataRoutingResult,
    EnergyDataSource,
    EnergyERP,
    ERPExtractionResult,
    ERPFieldMapping,
)

# ---------------------------------------------------------------------------
# EED Compliance Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_031_industrial_energy_audit.integrations.eed_compliance_bridge import (
    AuditCycleRecord,
    AuditStandard,
    ComplianceStatus,
    EEDComplianceBridge,
    EEDComplianceBridgeConfig,
    EEDComplianceReport,
    EEDExemptionType,
    EEDObligationAssessment,
    EUMemberState,
    NationalTransposition,
)

# ---------------------------------------------------------------------------
# ISO 50001 Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_031_industrial_energy_audit.integrations.iso_50001_bridge import (
    CertificationRecord,
    CertificationStatus,
    CorrectiveAction,
    CorrectiveActionStatus,
    DocumentStatus,
    EnMSDocument,
    InternalAuditRecord,
    ISO50001Bridge,
    ISO50001BridgeConfig,
    ISO50001Clause,
    ManagementReviewData,
    Nonconformity,
    NonconformitySeverity,
)

# ---------------------------------------------------------------------------
# BMS/SCADA Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_031_industrial_energy_audit.integrations.bms_scada_bridge import (
    AlarmEvent,
    AlarmSeverity,
    BMSSCADABridge,
    BMSSCADABridgeConfig,
    ConnectionStatus,
    DataPointMapping,
    DataPointType,
    HistoricalDataRequest,
    HistoricalDataResult,
    MeterReading,
    ProtocolConfig,
    ProtocolType,
)

# ---------------------------------------------------------------------------
# Utility Metering Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_031_industrial_energy_audit.integrations.utility_metering_bridge import (
    BillReconciliationResult,
    DemandProfile,
    EnergyCarrier,
    HierarchyLevel,
    IntervalReading,
    IntervalResolution,
    MeterRegistration,
    MeterType,
    UtilityMeteringBridge,
    UtilityMeteringBridgeConfig,
    VirtualMeterDefinition,
)

# ---------------------------------------------------------------------------
# Equipment Registry Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_031_industrial_energy_audit.integrations.equipment_registry_bridge import (
    EfficiencyRating,
    EquipmentCategory,
    EquipmentCondition,
    EquipmentRecord,
    EquipmentRegistryBridge,
    EquipmentRegistryBridgeConfig,
    MaintenanceRecord,
    MaintenanceType,
    NameplateData,
    ReplacementAssessment,
)

# ---------------------------------------------------------------------------
# Weather Normalization Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_031_industrial_energy_audit.integrations.weather_normalization_bridge import (
    ClimateZone,
    DailyWeatherRecord,
    DegreeDayMethod,
    DegreeDayResult,
    SeasonalAdjustmentFactor,
    WeatherNormalizationBridge,
    WeatherNormalizationBridgeConfig,
    WeatherNormalizationResult,
    WeatherSource,
    WeatherStationConfig,
)

# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_031_industrial_energy_audit.integrations.health_check import (
    CheckCategory,
    ComponentHealth,
    HealthCheck,
    HealthCheckConfig,
    HealthCheckResult,
    HealthSeverity,
    HealthStatus,
    RemediationSuggestion,
)

# ---------------------------------------------------------------------------
# Setup Wizard
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_031_industrial_energy_audit.integrations.setup_wizard import (
    BaselineConfig,
    EnergyCarrierConfig,
    EquipmentSetupConfig,
    FacilityProfile,
    IndustrySectorConfig,
    MeterSetupConfig,
    PresetConfig,
    RegulatoryConfig,
    SetupResult,
    SetupWizard,
    SetupWizardStep,
    StepStatus,
    WizardState,
    WizardStepState,
)

# ---------------------------------------------------------------------------
# EU ETS Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_031_industrial_energy_audit.integrations.eu_ets_bridge import (
    AllocationMethod,
    CarbonLeakageStatus,
    CarbonPriceImpact,
    ComplianceCycle,
    ComplianceCycleStatus,
    ETSBenchmarkComparison,
    ETSPhase,
    EmissionsRecord,
    EUETSBridge,
    EUETSBridgeConfig,
    FreeAllocationRecord,
    InstallationPermit,
)

__all__ = [
    # Module metadata
    "__version__",
    "__pack_id__",
    "__pack_name__",
    # --- Pack Orchestrator ---
    "IndustrialEnergyAuditOrchestrator",
    "OrchestratorConfig",
    "RetryConfig",
    "AuditPipelinePhase",
    "IndustrySector",
    "ExecutionStatus",
    "PhaseProvenance",
    "PhaseResult",
    "PipelineResult",
    "PHASE_DEPENDENCIES",
    "PHASE_EXECUTION_ORDER",
    "PARALLEL_PHASE_GROUPS",
    # --- MRV Energy Bridge ---
    "MRVEnergyBridge",
    "MRVEnergyBridgeConfig",
    "MRVAgentRoute",
    "EnergySource",
    "MRVScope",
    "RoutingResult",
    "BatchRoutingResult",
    "SavingsConversionResult",
    # --- Data Energy Bridge ---
    "DataEnergyBridge",
    "DataEnergyBridgeConfig",
    "DataAgentRoute",
    "EnergyDataSource",
    "EnergyERP",
    "ERPFieldMapping",
    "DataRoutingResult",
    "ERPExtractionResult",
    # --- EED Compliance Bridge ---
    "EEDComplianceBridge",
    "EEDComplianceBridgeConfig",
    "EEDObligationAssessment",
    "AuditCycleRecord",
    "NationalTransposition",
    "EEDComplianceReport",
    "EEDExemptionType",
    "ComplianceStatus",
    "AuditStandard",
    "EUMemberState",
    # --- ISO 50001 Bridge ---
    "ISO50001Bridge",
    "ISO50001BridgeConfig",
    "ISO50001Clause",
    "DocumentStatus",
    "NonconformitySeverity",
    "CorrectiveActionStatus",
    "CertificationStatus",
    "EnMSDocument",
    "InternalAuditRecord",
    "Nonconformity",
    "CorrectiveAction",
    "ManagementReviewData",
    "CertificationRecord",
    # --- BMS/SCADA Bridge ---
    "BMSSCADABridge",
    "BMSSCADABridgeConfig",
    "ProtocolType",
    "DataPointType",
    "AlarmSeverity",
    "ConnectionStatus",
    "ProtocolConfig",
    "DataPointMapping",
    "MeterReading",
    "AlarmEvent",
    "HistoricalDataRequest",
    "HistoricalDataResult",
    # --- Utility Metering Bridge ---
    "UtilityMeteringBridge",
    "UtilityMeteringBridgeConfig",
    "MeterType",
    "EnergyCarrier",
    "IntervalResolution",
    "HierarchyLevel",
    "MeterRegistration",
    "IntervalReading",
    "VirtualMeterDefinition",
    "BillReconciliationResult",
    "DemandProfile",
    # --- Equipment Registry Bridge ---
    "EquipmentRegistryBridge",
    "EquipmentRegistryBridgeConfig",
    "EquipmentCategory",
    "EquipmentCondition",
    "EfficiencyRating",
    "MaintenanceType",
    "NameplateData",
    "EquipmentRecord",
    "MaintenanceRecord",
    "ReplacementAssessment",
    # --- Weather Normalization Bridge ---
    "WeatherNormalizationBridge",
    "WeatherNormalizationBridgeConfig",
    "ClimateZone",
    "DegreeDayMethod",
    "WeatherSource",
    "WeatherStationConfig",
    "DailyWeatherRecord",
    "DegreeDayResult",
    "WeatherNormalizationResult",
    "SeasonalAdjustmentFactor",
    # --- Health Check ---
    "HealthCheck",
    "HealthCheckConfig",
    "HealthCheckResult",
    "ComponentHealth",
    "HealthSeverity",
    "HealthStatus",
    "CheckCategory",
    "RemediationSuggestion",
    # --- Setup Wizard ---
    "SetupWizard",
    "SetupWizardStep",
    "StepStatus",
    "FacilityProfile",
    "IndustrySectorConfig",
    "EnergyCarrierConfig",
    "MeterSetupConfig",
    "EquipmentSetupConfig",
    "RegulatoryConfig",
    "PresetConfig",
    "BaselineConfig",
    "WizardStepState",
    "WizardState",
    "SetupResult",
    # --- EU ETS Bridge ---
    "EUETSBridge",
    "EUETSBridgeConfig",
    "CarbonLeakageStatus",
    "AllocationMethod",
    "ComplianceCycleStatus",
    "ETSPhase",
    "InstallationPermit",
    "FreeAllocationRecord",
    "EmissionsRecord",
    "CarbonPriceImpact",
    "ETSBenchmarkComparison",
    "ComplianceCycle",
]
