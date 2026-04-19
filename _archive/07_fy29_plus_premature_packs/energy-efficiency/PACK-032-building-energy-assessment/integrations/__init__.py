# -*- coding: utf-8 -*-
"""
PACK-032 Building Energy Assessment Pack - Integration Layer
================================================================

Phase 4 integration layer for the Building Energy Assessment Pack that provides
building assessment pipeline orchestration, MRV emissions bridging, DATA agent
routing, EPBD compliance management, BMS data ingestion, weather normalization,
green building certification evaluation, grid carbon intensity tracking,
property registry management, CRREM pathway analysis, 22-category health
verification, and 8-step building setup wizard.

Components:
    - BuildingAssessmentOrchestrator: 12-phase pipeline with DAG dependency
      resolution, parallel execution, retry with exponential backoff, and
      SHA-256 provenance tracking for building energy assessments
    - MRVBuildingBridge: Routes building energy data to MRV agents
      (GL-MRV-BLD-001 through 008, plus fallback MRV-001, 002, 009-012)
      and converts energy savings to avoided emissions (tCO2e)
    - DataBuildingBridge: Routes data intake to DATA agents for meter data,
      ERP procurement (SAP RE-FX, Oracle PMS, Dynamics, Yardi, MRI),
      quality profiling, gap filling, and validation rule enforcement
    - EPBDComplianceBridge: EU Energy Performance of Buildings Directive
      (2024/1275) obligation assessment, MEES analysis, solar obligation
      checks, U-value compliance, and national transposition tracking
      for 10 EU/EEA member states
    - BMSIntegrationBridge: BACnet/Modbus/OPC-UA/MQTT data model adapters,
      real-time meter reading ingestion, Project Haystack tagging, alarm
      integration, and historical data retrieval
    - WeatherDataBridge: TMY data, HDD/CDD degree-day calculation, ASHRAE
      climate zone assignment, solar irradiance estimation, and weather-
      normalized energy baselines
    - CertificationBridge: LEED v4.1, BREEAM 2024, Energy Star, and NABERS
      green building certification evaluation with multi-scheme comparison
    - GridCarbonBridge: Location-based and market-based grid emission factors,
      residual mix factors, hourly carbon profiles, grid decarbonization
      projections (2025-2050), and REC/GO certificate tracking
    - PropertyRegistryBridge: UPRN/cadastral building registration, tenant
      management, energy apportionment, portfolio aggregation, and building
      lifecycle stage tracking
    - CRREMPathwayBridge: Carbon Risk Real Estate Monitor 1.5C and 2C
      decarbonization pathway alignment, stranding year assessment,
      retrofit pathway planning, and portfolio stranding analysis
    - HealthCheck: 22-category system health verification covering engines,
      workflows, templates, integrations, presets, and MRV agents
    - SetupWizard: 8-step guided building configuration with 8 building
      type presets (commercial office, retail, hotel, healthcare,
      education, residential multifamily, mixed-use, public sector)

Architecture:
    External Systems --> Integration Bridges --> Platform Components
                              |
                              v
    PACK-032 Building  <-- Composition <-- Zero Duplication
    Assessment                |
                              v
    MRV Agents <-- DATA Agents <-- FOUND Agents

Platform Integrations:
    - greenlang/agents/mrv/* (MRV-BLD-001..008, MRV-001, 002, 009-012)
    - greenlang/agents/data/* (DATA-002, 003, 010, 014, 019)
    - greenlang/agents/foundation/* (10 FOUND agents)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-032 Building Energy Assessment
Status: Production Ready
"""

__version__ = "1.0.0"
__pack_id__ = "PACK-032"
__pack_name__ = "Building Energy Assessment Pack"

# ---------------------------------------------------------------------------
# Pack Orchestrator
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_032_building_energy_assessment.integrations.pack_orchestrator import (
    AssessmentType,
    BuildingAssessmentOrchestrator,
    BuildingPipelinePhase,
    BuildingType,
    ExecutionStatus,
    OrchestratorConfig,
    PARALLEL_PHASE_GROUPS,
    PHASE_ASSESSMENT_TYPE_APPLICABILITY,
    PHASE_DEPENDENCIES,
    PHASE_EXECUTION_ORDER,
    PhaseProvenance,
    PhaseResult,
    PipelineResult,
    RetryConfig,
)

# ---------------------------------------------------------------------------
# MRV Building Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_032_building_energy_assessment.integrations.mrv_building_bridge import (
    BatchRoutingResult,
    BuildingEnergySource,
    BuildingMRVAgentType,
    BUILDING_TYPE_AGENT_MAP,
    GRID_EMISSION_FACTORS,
    MRVAgentRoute,
    MRVBuildingBridge,
    MRVBuildingBridgeConfig,
    MRV_ROUTING_TABLE,
    MRVScope,
    RoutingResult,
    SavingsConversionResult,
)

# ---------------------------------------------------------------------------
# Data Building Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_032_building_energy_assessment.integrations.data_building_bridge import (
    BuildingDataQualityReport,
    BuildingDataSource,
    BuildingERP,
    DataAgentRoute,
    DataBuildingBridge,
    DataBuildingBridgeConfig,
    DataRoutingResult,
    ERP_FIELD_MAPPINGS,
    ERPExtractionResult,
    ERPFieldMapping,
    UNIT_CONVERSION_TO_KWH,
    UtilityBillSchema,
)

# ---------------------------------------------------------------------------
# EPBD Compliance Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_032_building_energy_assessment.integrations.epbd_compliance_bridge import (
    BuildingCategory,
    ComplianceStatus,
    EPBDComplianceBridge,
    EPBDComplianceBridgeConfig,
    EPBDObligationAssessment,
    EPCRating,
    EUMemberState,
    MEESAssessment,
    NATIONAL_TRANSPOSITIONS,
    NationalTransposition,
    SolarObligationAssessment,
    SolarObligationType,
)

# ---------------------------------------------------------------------------
# BMS Integration Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_032_building_energy_assessment.integrations.bms_integration_bridge import (
    AlarmEvent,
    AlarmSeverity,
    BMSConnectionResult,
    BMSIntegrationBridge,
    BMSIntegrationBridgeConfig,
    ConnectionStatus,
    DataPointMapping,
    DataPointType,
    DEFAULT_POINT_TEMPLATES,
    HaystackMarker,
    HistoricalDataRequest,
    HistoricalDataResult,
    MeterReading,
    ProtocolConfig,
    ProtocolType,
)

# ---------------------------------------------------------------------------
# Weather Data Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_032_building_energy_assessment.integrations.weather_data_bridge import (
    CITY_CLIMATE_ZONES,
    CLIMATE_ZONE_SOLAR,
    ClimateZone,
    DailyWeatherRecord,
    DegreeDayMethod,
    DegreeDayResult,
    SolarIrradianceResult,
    WeatherDataBridge,
    WeatherDataBridgeConfig,
    WeatherNormalizationResult,
    WeatherSource,
    WeatherStationConfig,
    WindExposure,
)

# ---------------------------------------------------------------------------
# Certification Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_032_building_energy_assessment.integrations.certification_bridge import (
    BREEAM_ENERGY_CREDITS,
    BREEAM_LEVELS,
    CertificationBridge,
    CertificationBridgeConfig,
    CertificationEvaluation,
    CertificationLevel,
    CertificationScheme,
    CreditEvaluation,
    CreditStatus,
    LEED_ENERGY_CREDITS,
    LEED_LEVELS,
    MultiSchemeComparison,
)

# ---------------------------------------------------------------------------
# Grid Carbon Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_032_building_energy_assessment.integrations.grid_carbon_bridge import (
    CertificateType,
    EmissionFactorType,
    GridCarbonBridge,
    GridCarbonBridgeConfig,
    GridEmissionFactor,
    GRID_PROJECTIONS,
    GridProjection,
    GridRegion,
    HOURLY_PROFILES,
    HourlyEmissionProfile,
    LOCATION_BASED_FACTORS,
    RECertificate,
    RESIDUAL_MIX_FACTORS,
)

# ---------------------------------------------------------------------------
# Property Registry Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_032_building_energy_assessment.integrations.property_registry_bridge import (
    BuildingLifecycleStage,
    BuildingRecord,
    PortfolioCategory,
    PortfolioSummary,
    PropertyRegistryBridge,
    PropertyRegistryBridgeConfig,
    SpaceType,
    TenantRecord,
    TenureType,
)

# ---------------------------------------------------------------------------
# CRREM Pathway Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_032_building_energy_assessment.integrations.crrem_pathway_bridge import (
    CRREMBuildingType,
    CRREMPathway,
    CRREMPathwayBridge,
    CRREMPathwayBridgeConfig,
    CRREM_PATHWAYS_1_5C,
    CRREM_PATHWAYS_2C,
    CRREMScenario,
    PortfolioStrandingAnalysis,
    RetrofitAlignment,
    StrandingAssessment,
    TransitionRisk,
)

# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_032_building_energy_assessment.integrations.health_check import (
    BUILDING_ASSESSMENT_ENGINES,
    BUILDING_ASSESSMENT_INTEGRATIONS,
    BUILDING_ASSESSMENT_PRESETS,
    BUILDING_ASSESSMENT_TEMPLATES,
    BUILDING_ASSESSMENT_WORKFLOWS,
    CheckCategory,
    ComponentHealth,
    HealthCheck,
    HealthCheckConfig,
    HealthCheckResult,
    HealthSeverity,
    HealthStatus,
    MRV_BUILDING_AGENTS,
    RemediationSuggestion,
)

# ---------------------------------------------------------------------------
# Setup Wizard
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_032_building_energy_assessment.integrations.setup_wizard import (
    BUILDING_PRESETS,
    BuildingGeometryConfig,
    BuildingTypeConfig,
    EnvelopeConfig,
    HVACSystemsConfig,
    LightingDHWConfig,
    LocationClimateConfig,
    RegulatoryConfig,
    RenewableSystemsConfig,
    SetupResult,
    SetupWizard,
    SetupWizardStep,
    STEP_DISPLAY_NAMES,
    STEP_ORDER,
    StepStatus,
    WizardState,
    WizardStepState,
)

__all__ = [
    # Module metadata
    "__version__",
    "__pack_id__",
    "__pack_name__",
    # --- Pack Orchestrator ---
    "BuildingAssessmentOrchestrator",
    "OrchestratorConfig",
    "RetryConfig",
    "BuildingPipelinePhase",
    "BuildingType",
    "AssessmentType",
    "ExecutionStatus",
    "PhaseProvenance",
    "PhaseResult",
    "PipelineResult",
    "PHASE_DEPENDENCIES",
    "PHASE_EXECUTION_ORDER",
    "PARALLEL_PHASE_GROUPS",
    "PHASE_ASSESSMENT_TYPE_APPLICABILITY",
    # --- MRV Building Bridge ---
    "MRVBuildingBridge",
    "MRVBuildingBridgeConfig",
    "MRVAgentRoute",
    "BuildingEnergySource",
    "BuildingMRVAgentType",
    "MRVScope",
    "RoutingResult",
    "BatchRoutingResult",
    "SavingsConversionResult",
    "MRV_ROUTING_TABLE",
    "GRID_EMISSION_FACTORS",
    "BUILDING_TYPE_AGENT_MAP",
    # --- Data Building Bridge ---
    "DataBuildingBridge",
    "DataBuildingBridgeConfig",
    "DataAgentRoute",
    "BuildingDataSource",
    "BuildingERP",
    "ERPFieldMapping",
    "DataRoutingResult",
    "ERPExtractionResult",
    "UtilityBillSchema",
    "BuildingDataQualityReport",
    "ERP_FIELD_MAPPINGS",
    "UNIT_CONVERSION_TO_KWH",
    # --- EPBD Compliance Bridge ---
    "EPBDComplianceBridge",
    "EPBDComplianceBridgeConfig",
    "EPBDObligationAssessment",
    "MEESAssessment",
    "SolarObligationAssessment",
    "NationalTransposition",
    "EUMemberState",
    "ComplianceStatus",
    "BuildingCategory",
    "EPCRating",
    "SolarObligationType",
    "NATIONAL_TRANSPOSITIONS",
    # --- BMS Integration Bridge ---
    "BMSIntegrationBridge",
    "BMSIntegrationBridgeConfig",
    "ProtocolType",
    "DataPointType",
    "AlarmSeverity",
    "ConnectionStatus",
    "HaystackMarker",
    "ProtocolConfig",
    "DataPointMapping",
    "MeterReading",
    "AlarmEvent",
    "HistoricalDataRequest",
    "HistoricalDataResult",
    "BMSConnectionResult",
    "DEFAULT_POINT_TEMPLATES",
    # --- Weather Data Bridge ---
    "WeatherDataBridge",
    "WeatherDataBridgeConfig",
    "ClimateZone",
    "DegreeDayMethod",
    "WeatherSource",
    "WindExposure",
    "WeatherStationConfig",
    "DailyWeatherRecord",
    "DegreeDayResult",
    "SolarIrradianceResult",
    "WeatherNormalizationResult",
    "CITY_CLIMATE_ZONES",
    "CLIMATE_ZONE_SOLAR",
    # --- Certification Bridge ---
    "CertificationBridge",
    "CertificationBridgeConfig",
    "CertificationScheme",
    "CertificationLevel",
    "CreditStatus",
    "CreditEvaluation",
    "CertificationEvaluation",
    "MultiSchemeComparison",
    "LEED_ENERGY_CREDITS",
    "BREEAM_ENERGY_CREDITS",
    "BREEAM_LEVELS",
    "LEED_LEVELS",
    # --- Grid Carbon Bridge ---
    "GridCarbonBridge",
    "GridCarbonBridgeConfig",
    "EmissionFactorType",
    "CertificateType",
    "GridRegion",
    "GridEmissionFactor",
    "HourlyEmissionProfile",
    "GridProjection",
    "RECertificate",
    "LOCATION_BASED_FACTORS",
    "RESIDUAL_MIX_FACTORS",
    "GRID_PROJECTIONS",
    "HOURLY_PROFILES",
    # --- Property Registry Bridge ---
    "PropertyRegistryBridge",
    "PropertyRegistryBridgeConfig",
    "BuildingLifecycleStage",
    "TenureType",
    "PortfolioCategory",
    "SpaceType",
    "BuildingRecord",
    "TenantRecord",
    "PortfolioSummary",
    # --- CRREM Pathway Bridge ---
    "CRREMPathwayBridge",
    "CRREMPathwayBridgeConfig",
    "CRREMScenario",
    "CRREMBuildingType",
    "TransitionRisk",
    "CRREMPathway",
    "StrandingAssessment",
    "RetrofitAlignment",
    "PortfolioStrandingAnalysis",
    "CRREM_PATHWAYS_1_5C",
    "CRREM_PATHWAYS_2C",
    # --- Health Check ---
    "HealthCheck",
    "HealthCheckConfig",
    "HealthCheckResult",
    "ComponentHealth",
    "HealthSeverity",
    "HealthStatus",
    "CheckCategory",
    "RemediationSuggestion",
    "BUILDING_ASSESSMENT_ENGINES",
    "BUILDING_ASSESSMENT_WORKFLOWS",
    "BUILDING_ASSESSMENT_TEMPLATES",
    "BUILDING_ASSESSMENT_INTEGRATIONS",
    "BUILDING_ASSESSMENT_PRESETS",
    "MRV_BUILDING_AGENTS",
    # --- Setup Wizard ---
    "SetupWizard",
    "SetupWizardStep",
    "StepStatus",
    "BuildingTypeConfig",
    "LocationClimateConfig",
    "BuildingGeometryConfig",
    "EnvelopeConfig",
    "HVACSystemsConfig",
    "LightingDHWConfig",
    "RenewableSystemsConfig",
    "RegulatoryConfig",
    "WizardStepState",
    "WizardState",
    "SetupResult",
    "STEP_ORDER",
    "STEP_DISPLAY_NAMES",
    "BUILDING_PRESETS",
]
