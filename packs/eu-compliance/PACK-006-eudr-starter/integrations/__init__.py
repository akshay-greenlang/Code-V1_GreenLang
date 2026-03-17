# -*- coding: utf-8 -*-
"""
PACK-006 EUDR Starter Pack - Integration Layer
================================================

Phase 4 integration layer that connects EUDR engines, external bridges,
and application services into a cohesive EUDR compliance pipeline. This
module provides orchestration, bridging to 5 external systems, guided
setup, and 14-category health verification for production deployments.

Components:
    - EUDRStarterOrchestrator: 8-phase master pipeline orchestrator
    - EUDRAppBridge: Bridge to GL-EUDR-APP v1.0 (8 proxy services)
    - TraceabilityBridge: Bridge to EUDR Traceability Connector (7 proxies)
    - SatelliteBridge: Bridge to Deforestation Satellite Connector (7 methods)
    - GISBridge: Bridge to GIS/Mapping Connector (8 methods)
    - EUInformationSystemBridge: Bridge to EU IS for DDS submission (9 methods)
    - EUDRStarterSetupWizard: 8-step guided EUDR setup wizard
    - EUDRStarterHealthCheck: 14-category health verification system

Architecture:
    Suppliers/Plots --> DataIntake --> GeolocationValidation --> RiskAssessment
                                                                       |
                                                                       v
    EU IS <-- DDSSubmission <-- ComplianceCheck <-- DDSAssembly <-- Scored Data
                                                                       |
                                                                       v
    Dashboards <-- Reports <-- Templates <-- Compliance Engine

External Bridges:
    GL-EUDR-APP -----> EUDRAppBridge (8 proxy services)
    Traceability ----> TraceabilityBridge (7 proxy services)
    Satellite -------> SatelliteBridge (stub in Starter tier)
    GIS/Mapping -----> GISBridge (8 methods)
    EU IS -----------> EUInformationSystemBridge (9 methods)

Author: GreenLang Team
Version: 1.0.0
"""

from packs.eu_compliance.PACK_006_eudr_starter.integrations.pack_orchestrator import (
    EUDRStarterOrchestrator,
    OrchestratorConfig,
    OrchestrationResult,
    PhaseResult,
    CheckpointData,
    ExecutionPhase,
    PhaseStatusCode,
    RiskLevel,
    ProgressCallback,
)
from packs.eu_compliance.PACK_006_eudr_starter.integrations.eudr_app_bridge import (
    EUDRAppBridge,
    EUDRAppBridgeConfig,
    SupplierProxy,
    PlotProxy,
    DDSProxy,
    PipelineProxy,
    RiskProxy as AppRiskProxy,
    DashboardProxy,
    DocumentProxy,
    SettingsProxy,
    SupplierRecord,
    PlotRecord,
    DDSRecord,
    DDSStatus,
    PipelineStage,
    PipelineStatus,
    DashboardMetrics,
    DocumentRecord,
    SupplierComplianceStatus,
)
from packs.eu_compliance.PACK_006_eudr_starter.integrations.traceability_bridge import (
    TraceabilityBridge,
    TraceabilityBridgeConfig,
    PlotRegistryProxy,
    ChainOfCustodyProxy,
    CommodityProxy,
    ComplianceProxy,
    DueDiligenceProxy,
    RiskProxy as TraceRiskProxy,
    SupplyChainProxy,
    PlotData,
    CustodyEvent,
    BatchRecord,
    CommodityClassification,
    ComplianceCheckResult,
    SupplyChainNode,
)
from packs.eu_compliance.PACK_006_eudr_starter.integrations.satellite_bridge import (
    SatelliteBridge,
    SatelliteBridgeConfig,
    SatelliteDataResult,
    ForestChangeResult,
    DeforestationAlert,
    BaselineAssessment,
    MonitoringResult,
    AlertAggregation,
    SatelliteComplianceReport,
)
from packs.eu_compliance.PACK_006_eudr_starter.integrations.gis_bridge import (
    GISBridge,
    GISBridgeConfig,
    TransformedCoordinates,
    BoundaryResult,
    SpatialAnalysis,
    LandCoverResult,
    GeocodedLocation,
    ReverseGeocodeResult,
    ParsedGeoData,
    TopologyValidation,
)
from packs.eu_compliance.PACK_006_eudr_starter.integrations.eu_information_system_bridge import (
    EUInformationSystemBridge,
    EUISConfig,
    EUISEnvironment,
    SubmissionResult,
    SubmissionStatus,
    SubmissionStatusCode,
    AmendmentResult,
    DDSDocument,
    FormatValidation,
    RegistrationResult,
    OperatorStatus,
    OperatorStatusCode,
    ReferenceNumber,
    CountryBenchmark,
    CountryBenchmarkLevel,
    AuditEntry,
)
from packs.eu_compliance.PACK_006_eudr_starter.integrations.setup_wizard import (
    EUDRStarterSetupWizard,
    WizardState,
    WizardStep,
    WizardStepName,
    StepStatus,
    CommoditySelection,
    CompanySizeSelection,
    CompanySizeCategory,
    GeolocationConfig,
    RiskThresholdConfig,
    EUISConfiguration,
    InitialDataImport,
    SetupReport,
)
from packs.eu_compliance.PACK_006_eudr_starter.integrations.health_check import (
    EUDRStarterHealthCheck,
    HealthCheckConfig,
    HealthCheckResult,
    CategoryHealthResult,
    HealthStatus,
    HealthCategory,
    RemediationSuggestion,
    RemediationSeverity,
)

__version__ = "1.0.0"
__pack_id__ = "PACK-006"
__pack_name__ = "EUDR Starter Pack"

__all__ = [
    # --- Pack Orchestrator ---
    "EUDRStarterOrchestrator",
    "OrchestratorConfig",
    "OrchestrationResult",
    "PhaseResult",
    "CheckpointData",
    "ExecutionPhase",
    "PhaseStatusCode",
    "RiskLevel",
    "ProgressCallback",
    # --- EUDR App Bridge ---
    "EUDRAppBridge",
    "EUDRAppBridgeConfig",
    "SupplierProxy",
    "PlotProxy",
    "DDSProxy",
    "PipelineProxy",
    "AppRiskProxy",
    "DashboardProxy",
    "DocumentProxy",
    "SettingsProxy",
    "SupplierRecord",
    "PlotRecord",
    "DDSRecord",
    "DDSStatus",
    "PipelineStage",
    "PipelineStatus",
    "DashboardMetrics",
    "DocumentRecord",
    "SupplierComplianceStatus",
    # --- Traceability Bridge ---
    "TraceabilityBridge",
    "TraceabilityBridgeConfig",
    "PlotRegistryProxy",
    "ChainOfCustodyProxy",
    "CommodityProxy",
    "ComplianceProxy",
    "DueDiligenceProxy",
    "TraceRiskProxy",
    "SupplyChainProxy",
    "PlotData",
    "CustodyEvent",
    "BatchRecord",
    "CommodityClassification",
    "ComplianceCheckResult",
    "SupplyChainNode",
    # --- Satellite Bridge ---
    "SatelliteBridge",
    "SatelliteBridgeConfig",
    "SatelliteDataResult",
    "ForestChangeResult",
    "DeforestationAlert",
    "BaselineAssessment",
    "MonitoringResult",
    "AlertAggregation",
    "SatelliteComplianceReport",
    # --- GIS Bridge ---
    "GISBridge",
    "GISBridgeConfig",
    "TransformedCoordinates",
    "BoundaryResult",
    "SpatialAnalysis",
    "LandCoverResult",
    "GeocodedLocation",
    "ReverseGeocodeResult",
    "ParsedGeoData",
    "TopologyValidation",
    # --- EU Information System Bridge ---
    "EUInformationSystemBridge",
    "EUISConfig",
    "EUISEnvironment",
    "SubmissionResult",
    "SubmissionStatus",
    "SubmissionStatusCode",
    "AmendmentResult",
    "DDSDocument",
    "FormatValidation",
    "RegistrationResult",
    "OperatorStatus",
    "OperatorStatusCode",
    "ReferenceNumber",
    "CountryBenchmark",
    "CountryBenchmarkLevel",
    "AuditEntry",
    # --- Setup Wizard ---
    "EUDRStarterSetupWizard",
    "WizardState",
    "WizardStep",
    "WizardStepName",
    "StepStatus",
    "CommoditySelection",
    "CompanySizeSelection",
    "CompanySizeCategory",
    "GeolocationConfig",
    "RiskThresholdConfig",
    "EUISConfiguration",
    "InitialDataImport",
    "SetupReport",
    # --- Health Check ---
    "EUDRStarterHealthCheck",
    "HealthCheckConfig",
    "HealthCheckResult",
    "CategoryHealthResult",
    "HealthStatus",
    "HealthCategory",
    "RemediationSuggestion",
    "RemediationSeverity",
]
