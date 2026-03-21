# -*- coding: utf-8 -*-
"""
PACK-034 ISO 50001 Energy Management System Pack - Integration Layer
======================================================================

Phase 4 integration layer for the ISO 50001 EnMS Pack that provides
EnMS pipeline orchestration, MRV emissions bridging, DATA agent routing,
PACK-031/032/033 data import, EED compliance tracking, BMS/SCADA data
ingestion, sub-metering hierarchy management, 15-category health
verification, 8-step EnMS setup wizard, and certification body interface.

Components:
    - EnMSOrchestrator: 10-phase pipeline with DAG dependency
      resolution, parallel execution, retry with exponential backoff,
      and SHA-256 provenance tracking
    - MRVEnMSBridge: Routes energy consumption and savings data to
      MRV agents (Stationary Combustion, Refrigerants, Scope 2
      Location/Market-Based, Steam, Cooling, Data Freshness) and
      converts energy savings to avoided emissions (tCO2e)
    - DataEnMSBridge: Routes data intake to DATA agents for energy
      data, meter exports, quality profiling, and validation rule
      enforcement
    - Pack031Bridge: Imports energy audit results, equipment inventory,
      energy baselines, and ISO 50001 gap analysis from PACK-031
    - Pack032Bridge: Imports building assessment results, zone data,
      HVAC profiles, and envelope data from PACK-032
    - Pack033Bridge: Imports quick win measures, payback analyses, and
      implementation plans from PACK-033
    - EEDComplianceBridge: EU Energy Efficiency Directive 2023/1791
      compliance tracking, Article 8 audit exemption for ISO 50001
      certified organizations, clause-to-EED mapping
    - BMSSCADABridge: BACnet/Modbus/OPC-UA/MQTT data model adapters,
      real-time meter reading ingestion, and alarm integration
    - MeteringBridge: Sub-metering hierarchy management, reconciliation,
      coverage validation, and virtual meter calculation
    - HealthCheck: 15-category system health verification
    - SetupWizard: 8-step guided EnMS configuration with 8 facility
      presets
    - CertificationBodyBridge: Certification body registration, audit
      scheduling, documentation submission, and certificate lifecycle

Architecture:
    External Systems --> Integration Bridges --> Platform Components
                              |
                              v
    PACK-034 EnMS <-- Composition <-- Zero Duplication
                              |
                              v
    MRV Agents <-- DATA Agents <-- FOUND Agents

Platform Integrations:
    - greenlang/agents/mrv/* (MRV-001, 002, 009-012, 016)
    - greenlang/agents/data/* (DATA-002, 003, 010, 019)
    - greenlang/agents/foundation/* (10 FOUND agents)
    - packs/energy-efficiency/PACK-031 (Industrial Energy Audit)
    - packs/energy-efficiency/PACK-032 (Building Energy Assessment)
    - packs/energy-efficiency/PACK-033 (Quick Wins Identifier)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-034 ISO 50001 Energy Management System
Status: Production Ready
"""

__version__ = "1.0.0"
__pack_id__ = "PACK-034"
__pack_name__ = "ISO 50001 Energy Management System Pack"

# ---------------------------------------------------------------------------
# Pack Orchestrator
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_034_iso50001_enms.integrations.pack_orchestrator import (
    EnMSOrchestrator,
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
    RetryConfig,
)

# ---------------------------------------------------------------------------
# MRV EnMS Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_034_iso50001_enms.integrations.mrv_enms_bridge import (
    DEFAULT_EMISSION_FACTORS,
    MeasureCategory,
    MRVEnMSBridge,
    MRVRouteConfig,
    MRVScope,
    RoutingResult,
    SavingsToEmissionsMapping,
)

# ---------------------------------------------------------------------------
# Data EnMS Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_034_iso50001_enms.integrations.data_enms_bridge import (
    DataEnMSBridge,
    DataQualityCheck,
    DataRouteConfig,
    DataRoutingResult,
    EnMSDataSource,
)

# ---------------------------------------------------------------------------
# PACK-031 Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_034_iso50001_enms.integrations.pack031_bridge import (
    AuditDataImport,
    AuditImportConfig,
    Pack031Bridge,
)

# ---------------------------------------------------------------------------
# PACK-032 Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_034_iso50001_enms.integrations.pack032_bridge import (
    AssessmentImportConfig,
    BuildingDataImport,
    Pack032Bridge,
)

# ---------------------------------------------------------------------------
# PACK-033 Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_034_iso50001_enms.integrations.pack033_bridge import (
    Pack033Bridge,
    QuickWinsDataImport,
    QuickWinsImportConfig,
)

# ---------------------------------------------------------------------------
# EED Compliance Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_034_iso50001_enms.integrations.eed_compliance_bridge import (
    ComplianceLevel,
    EEDArticle,
    EEDComplianceBridge,
    EEDComplianceConfig,
    EEDComplianceResult,
    ExemptionStatus,
)

# ---------------------------------------------------------------------------
# BMS/SCADA Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_034_iso50001_enms.integrations.bms_scada_bridge import (
    AlarmEvent,
    BMSConfig,
    BMSSCADABridge,
    ConnectionStatus,
    DataPoint,
    DataPointType,
    MeterReading as BMSMeterReading,
    ProtocolType,
)

# ---------------------------------------------------------------------------
# Metering Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_034_iso50001_enms.integrations.metering_bridge import (
    MeterConfig,
    MeterHierarchy,
    MeterReading,
    MeterType,
    MeteringBridge,
    ReadingQuality,
    ReconciliationResult,
    ReconciliationStatus,
)

# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_034_iso50001_enms.integrations.health_check import (
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
from packs.energy_efficiency.PACK_034_iso50001_enms.integrations.setup_wizard import (
    FacilitySetup,
    PresetConfig,
    SetupResult,
    SetupStep,
    SetupWizard,
    SetupWizardStep,
    StepStatus,
    WizardState,
    WizardStepState,
)

# ---------------------------------------------------------------------------
# Certification Body Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_034_iso50001_enms.integrations.certification_body_bridge import (
    AuditReport,
    AuditSchedule,
    AuditType,
    CertificateInfo,
    CertificateStatus,
    CertificationBody,
    CertificationBodyBridge,
    FindingType,
)

__all__ = [
    # Module metadata
    "__version__",
    "__pack_id__",
    "__pack_name__",
    # --- Pack Orchestrator ---
    "EnMSOrchestrator",
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
    # --- MRV EnMS Bridge ---
    "MRVEnMSBridge",
    "MRVRouteConfig",
    "MeasureCategory",
    "MRVScope",
    "RoutingResult",
    "SavingsToEmissionsMapping",
    "DEFAULT_EMISSION_FACTORS",
    # --- Data EnMS Bridge ---
    "DataEnMSBridge",
    "DataRouteConfig",
    "DataRoutingResult",
    "DataQualityCheck",
    "EnMSDataSource",
    # --- PACK-031 Bridge ---
    "Pack031Bridge",
    "AuditImportConfig",
    "AuditDataImport",
    # --- PACK-032 Bridge ---
    "Pack032Bridge",
    "AssessmentImportConfig",
    "BuildingDataImport",
    # --- PACK-033 Bridge ---
    "Pack033Bridge",
    "QuickWinsImportConfig",
    "QuickWinsDataImport",
    # --- EED Compliance Bridge ---
    "EEDComplianceBridge",
    "EEDComplianceConfig",
    "EEDComplianceResult",
    "EEDArticle",
    "ExemptionStatus",
    "ComplianceLevel",
    # --- BMS/SCADA Bridge ---
    "BMSSCADABridge",
    "BMSConfig",
    "DataPoint",
    "DataPointType",
    "ProtocolType",
    "ConnectionStatus",
    "BMSMeterReading",
    "AlarmEvent",
    # --- Metering Bridge ---
    "MeteringBridge",
    "MeterConfig",
    "MeterHierarchy",
    "MeterReading",
    "MeterType",
    "ReadingQuality",
    "ReconciliationResult",
    "ReconciliationStatus",
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
    "SetupStep",
    "StepStatus",
    "FacilitySetup",
    "PresetConfig",
    "WizardStepState",
    "WizardState",
    "SetupResult",
    # --- Certification Body Bridge ---
    "CertificationBodyBridge",
    "CertificationBody",
    "AuditSchedule",
    "AuditReport",
    "CertificateInfo",
    "AuditType",
    "CertificateStatus",
    "FindingType",
]
