# -*- coding: utf-8 -*-
"""
PACK-028 Sector Pathway Pack - Integration Layer
==========================================================

Phase 4 integration layer for the Sector Pathway Pack providing
10-phase DAG pipeline orchestration with sector-specific conditional
routing, SBTi SDA convergence pathway data for 12 sectors, IEA NZE
2050 scenario integration with 400+ technology milestones, IPCC AR6
GWP-100 values and emission factors, PACK-021 baseline/target import,
30-agent MRV routing with per-sector priority, decarbonization lever
registry with abatement waterfall analysis, 20-agent DATA routing with
sector activity data, 20-category system health monitoring, and 7-step
sector pathway configuration wizard.

Components:
    - SectorPathwayPipelineOrchestrator: 10-phase DAG pipeline
      (Sector Classification -> Activity Data Intake -> Intensity
      Calculation -> Pathway Generation -> [Convergence Analysis |
      Technology Roadmap | Abatement Waterfall | Sector Benchmarking]
      -> Scenario Comparison -> Strategy Synthesis)
    - SBTiSDABridge: SBTi Sectoral Decarbonization Approach integration
      with 12 sector convergence pathway lookup tables, 42-criteria
      validation, and submission package generation
    - IEANZEBridge: IEA NZE 2050 scenario data with 5 scenarios,
      400+ technology milestones, S-curve adoption modeling, and
      cross-sector technology interdependency graph
    - IPCCAR6Bridge: IPCC AR6 GWP-100 values (26 GHG species),
      emission factors (CO2/CH4/N2O for 20 fuel types), carbon budgets,
      process emission factors, and SSP pathway alignment
    - PACK021Bridge: PACK-021 Net Zero Starter Pack baseline/target
      import with SDA intensity overlay and gap enhancement
    - SectorMRVBridge: 30-agent MRV routing with per-sector priority
      matrix (critical/high/medium/low) and intensity calculation
    - SectorDecarbBridge: Decarbonization lever registry with per-sector
      abatement waterfall analysis and implementation roadmap
    - SectorDataBridge: 20-agent DATA routing with sector-specific
      activity data requirements and intake profiling
    - SectorPathwayHealthCheck: 20-category system health monitoring
      with SBTi data freshness, IEA milestone currency, IPCC integrity,
      convergence calculator validation, and sector coverage analysis
    - SectorPathwaySetupWizard: 7-step configuration wizard with sector
      selection, activity data setup, baseline, pathway preferences,
      technology inventory, abatement planning, and review/deploy

Architecture:
    External Data Sources --> Integration Bridges --> Platform Components
                                  |
                                  v
    PACK-028 Sector <-- Composition <-- Zero Duplication
                                  |
                                  v
    MRV Agents (30) <-- DATA Agents (20) <-- PACK-021 Baseline
                                  |
                                  v
    SBTi SDA <-- IEA NZE <-- IPCC AR6 <-- Decarb Levers
                                  |
                                  v
    Health Monitoring <-- Setup Wizard <-- Config/Presets

Sector Routing Groups:
    - heavy_industry: steel, cement, aluminum, chemicals, oil_gas_upstream
    - light_industry: pulp_paper, food_beverage
    - transport: aviation, shipping, road_transport, rail
    - power: power_generation
    - buildings: buildings_residential, buildings_commercial
    - agriculture: agriculture
    - cross_sector: cross_sector

Platform Integrations:
    - greenlang/agents/mrv/* (all 30 MRV agents)
    - greenlang/agents/data/* (all 20 DATA agents)
    - packs/net-zero/PACK-021-net-zero-starter/* (baseline/target)
    - SBTi Corporate Standard V5.3 (SDA convergence pathways)
    - IEA World Energy Outlook 2024 (NZE 2050 scenarios)
    - IPCC AR6 WG1 (GWP-100, emission factors, carbon budgets)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-028 Sector Pathway Pack
Status: Production Ready
"""

__version__ = "1.0.0"
__pack_id__ = "PACK-028"
__pack_name__ = "Sector Pathway Pack"

# ---------------------------------------------------------------------------
# Sector Pathway Pipeline Orchestrator
# ---------------------------------------------------------------------------
from .pack_orchestrator import (
    SectorPathwayPhase,
    ExecutionStatus,
    SectorPathType,
    SDAEligibleSector,
    ExtendedSector,
    ConvergenceModel,
    ClimateScenario,
    RetryConfig,
    SectorPathwayOrchestratorConfig,
    PhaseProvenance,
    PhaseResult,
    SectorClassificationOutput,
    IntensityMetricOutput,
    PathwayPoint,
    PipelineResult,
    PhaseProgress,
    SectorPathwayPipelineOrchestrator,
    SECTOR_NACE_MAPPING,
    SECTOR_ROUTING_GROUPS,
    SECTOR_MRV_PRIORITY,
    PHASE_DEPENDENCIES,
    PHASE_EXECUTION_ORDER,
    PARALLEL_PHASE_GROUP,
)

# ---------------------------------------------------------------------------
# SBTi SDA Bridge
# ---------------------------------------------------------------------------
from .sbti_sda_bridge import (
    SBTiSDAPathway,
    SBTiTargetType,
    CriteriaStatus,
    SBTiSubmissionStatus,
    TemperatureRating,
    ConvergenceMethod,
    SDASector,
    SBTiSDABridgeConfig,
    SectorClassification,
    IntensityConvergencePoint,
    IntensityConvergenceResult,
    CriteriaValidation,
    SBTiTargetDefinition,
    SBTiSDAValidationResult,
    SBTiSubmissionPackage,
    SBTiProgressReport,
    SBTiSDABridge,
    SDA_CONVERGENCE_PATHWAYS,
    SDA_INTENSITY_METRICS,
    NACE_TO_SDA_SECTOR,
    GICS_TO_SDA_SECTOR,
    SBTI_NEAR_TERM_CRITERIA,
    SBTI_NET_ZERO_CRITERIA,
)

# ---------------------------------------------------------------------------
# IEA NZE Bridge
# ---------------------------------------------------------------------------
from .iea_nze_bridge import (
    IEAScenario,
    IEASector,
    IEARegion,
    MilestoneStatus,
    TechnologyReadinessLevel,
    IEANZEBridgeConfig,
    SectorPathwayData,
    MilestoneTrackingResult,
    ScenarioComparisonResult,
    TechnologyAdoptionCurve,
    IEANZEBridge,
    IEA_SECTOR_PATHWAYS,
    REGIONAL_ADJUSTMENT_FACTORS,
    IEA_TECHNOLOGY_MILESTONES,
    TECHNOLOGY_INTERDEPENDENCIES,
)

# ---------------------------------------------------------------------------
# IPCC AR6 Bridge
# ---------------------------------------------------------------------------
from .ipcc_ar6_bridge import (
    GHGSpecies,
    SSPScenario,
    IPCCSector,
    FuelType,
    IPCCAR6BridgeConfig,
    GWPLookupResult,
    EmissionFactorResult,
    CarbonBudgetResult,
    GHGConversionResult,
    SSPAlignmentResult,
    IPCCAR6Bridge,
    GWP_100_AR6,
    CARBON_BUDGETS_GTCO2,
    EMISSION_FACTORS_CO2_KG_PER_TJ,
    EMISSION_FACTORS_CH4_KG_PER_TJ,
    EMISSION_FACTORS_N2O_KG_PER_TJ,
    PROCESS_EMISSION_FACTORS,
    AGRICULTURAL_EMISSION_FACTORS,
    SSP_EMISSION_PATHWAYS,
)

# ---------------------------------------------------------------------------
# PACK-021 Bridge
# ---------------------------------------------------------------------------
from .pack021_bridge import (
    BaselineStatus,
    TargetPathway,
    GapSeverity,
    PACK021BridgeConfig,
    BaselineImport,
    TargetImport,
    GapImport,
    SectorEnhancement,
    PACK021IntegrationResult,
    PACK021Bridge,
    PACK021_COMPONENTS,
)

# ---------------------------------------------------------------------------
# Sector MRV Bridge
# ---------------------------------------------------------------------------
from .mrv_bridge import (
    MRVScope,
    SectorPriority,
    SectorMRVAgentRoute,
    SectorMRVBridgeConfig,
    RoutingResult,
    IntensityResult,
    BatchRoutingResult,
    SectorMRVBridge,
    SECTOR_AGENT_PRIORITIES,
    SECTOR_MRV_ROUTING_TABLE,
    SECTOR_TO_ROUTING_GROUP,
    SECTOR_INTENSITY_METRICS as MRV_SECTOR_INTENSITY_METRICS,
)

# ---------------------------------------------------------------------------
# Sector Decarb Bridge
# ---------------------------------------------------------------------------
from .decarb_bridge import (
    LeverCategory,
    ImplementationPhase,
    LeverStatus,
    SectorDecarbBridgeConfig,
    LeverAnalysis,
    AbatementWaterfall,
    ImplementationRoadmap,
    SectorDecarbBridge,
    SECTOR_DECARB_LEVERS,
)

# ---------------------------------------------------------------------------
# Sector Data Bridge
# ---------------------------------------------------------------------------
from .data_bridge import (
    SectorDataBridgeConfig,
    IntakeResult,
    ReconciliationResult,
    ActivityDataProfile,
    SectorDataBridge,
    SECTOR_DATA_AGENT_ROUTING,
    SECTOR_ACTIVITY_REQUIREMENTS,
)

# ---------------------------------------------------------------------------
# Sector Pathway Setup Wizard
# ---------------------------------------------------------------------------
from .setup_wizard import (
    SectorWizardStep,
    StepStatus,
    SDAEligibility,
    ConvergencePreference,
    ScenarioPreference,
    TechnologyMaturity,
    SectorRoutingGroup,
    SectorSelectionData,
    ActivityDataSetup,
    BaselineConfig,
    PathwayPreferences,
    TechnologyInventoryData,
    AbatementPlanData,
    WizardStepState,
    WizardState,
    SectorPathwaySetupResult,
    SectorPathwaySetupWizard,
    STEP_ORDER,
    STEP_DISPLAY_NAMES,
    STEP_DESCRIPTIONS,
    SDA_SECTOR_OPTIONS,
    EXTENDED_SECTOR_OPTIONS,
    ALL_SECTOR_OPTIONS,
    SECTOR_TECHNOLOGY_PROFILES,
    SECTOR_LEVER_PRIORITIES,
    ROUTING_GROUP_MRV_PRIORITIES,
    ROUTING_GROUP_DATA_PRIORITIES,
)

# ---------------------------------------------------------------------------
# Sector Pathway Health Check
# ---------------------------------------------------------------------------
from .health_check import (
    HealthStatus,
    HealthSeverity,
    CheckCategory,
    DataFreshnessStatus,
    RemediationSuggestion,
    ComponentHealth,
    DataFreshnessCheck,
    SectorCoverageCheck,
    HealthCheckConfig,
    HealthCheckResult,
    SectorPathwayHealthCheck,
    SECTOR_PATHWAY_ENGINES,
    SECTOR_PATHWAY_WORKFLOWS,
    SECTOR_PATHWAY_TEMPLATES,
    SDA_SECTORS,
    EXTENDED_SECTORS,
    ALL_SECTORS,
    IEA_SCENARIOS,
    INTEGRATION_BRIDGES,
    MIGRATION_FILES,
    SECTOR_INTENSITY_METRICS,
    QUICK_CHECK_CATEGORIES,
    CRITICAL_CATEGORIES,
)

__all__ = [
    # Module metadata
    "__version__",
    "__pack_id__",
    "__pack_name__",
    # --- Sector Pathway Pipeline Orchestrator ---
    "SectorPathwayPipelineOrchestrator",
    "SectorPathwayOrchestratorConfig",
    "RetryConfig",
    "SectorPathwayPhase",
    "ExecutionStatus",
    "SectorPathType",
    "SDAEligibleSector",
    "ExtendedSector",
    "ConvergenceModel",
    "ClimateScenario",
    "PhaseProvenance",
    "PhaseResult",
    "SectorClassificationOutput",
    "IntensityMetricOutput",
    "PathwayPoint",
    "PipelineResult",
    "PhaseProgress",
    "SECTOR_NACE_MAPPING",
    "SECTOR_ROUTING_GROUPS",
    "SECTOR_MRV_PRIORITY",
    "PHASE_DEPENDENCIES",
    "PHASE_EXECUTION_ORDER",
    "PARALLEL_PHASE_GROUP",
    # --- SBTi SDA Bridge ---
    "SBTiSDABridge",
    "SBTiSDABridgeConfig",
    "SBTiSDAPathway",
    "SBTiTargetType",
    "CriteriaStatus",
    "SBTiSubmissionStatus",
    "TemperatureRating",
    "ConvergenceMethod",
    "SDASector",
    "SectorClassification",
    "IntensityConvergencePoint",
    "IntensityConvergenceResult",
    "CriteriaValidation",
    "SBTiTargetDefinition",
    "SBTiSDAValidationResult",
    "SBTiSubmissionPackage",
    "SBTiProgressReport",
    "SDA_CONVERGENCE_PATHWAYS",
    "SDA_INTENSITY_METRICS",
    "NACE_TO_SDA_SECTOR",
    "GICS_TO_SDA_SECTOR",
    "SBTI_NEAR_TERM_CRITERIA",
    "SBTI_NET_ZERO_CRITERIA",
    # --- IEA NZE Bridge ---
    "IEANZEBridge",
    "IEANZEBridgeConfig",
    "IEAScenario",
    "IEASector",
    "IEARegion",
    "MilestoneStatus",
    "TechnologyReadinessLevel",
    "SectorPathwayData",
    "MilestoneTrackingResult",
    "ScenarioComparisonResult",
    "TechnologyAdoptionCurve",
    "IEA_SECTOR_PATHWAYS",
    "REGIONAL_ADJUSTMENT_FACTORS",
    "IEA_TECHNOLOGY_MILESTONES",
    "TECHNOLOGY_INTERDEPENDENCIES",
    # --- IPCC AR6 Bridge ---
    "IPCCAR6Bridge",
    "IPCCAR6BridgeConfig",
    "GHGSpecies",
    "SSPScenario",
    "IPCCSector",
    "FuelType",
    "GWPLookupResult",
    "EmissionFactorResult",
    "CarbonBudgetResult",
    "GHGConversionResult",
    "SSPAlignmentResult",
    "GWP_100_AR6",
    "CARBON_BUDGETS_GTCO2",
    "EMISSION_FACTORS_CO2_KG_PER_TJ",
    "EMISSION_FACTORS_CH4_KG_PER_TJ",
    "EMISSION_FACTORS_N2O_KG_PER_TJ",
    "PROCESS_EMISSION_FACTORS",
    "AGRICULTURAL_EMISSION_FACTORS",
    "SSP_EMISSION_PATHWAYS",
    # --- PACK-021 Bridge ---
    "PACK021Bridge",
    "PACK021BridgeConfig",
    "BaselineStatus",
    "TargetPathway",
    "GapSeverity",
    "BaselineImport",
    "TargetImport",
    "GapImport",
    "SectorEnhancement",
    "PACK021IntegrationResult",
    "PACK021_COMPONENTS",
    # --- Sector MRV Bridge ---
    "SectorMRVBridge",
    "SectorMRVBridgeConfig",
    "SectorMRVAgentRoute",
    "MRVScope",
    "SectorPriority",
    "RoutingResult",
    "IntensityResult",
    "BatchRoutingResult",
    "SECTOR_AGENT_PRIORITIES",
    "SECTOR_MRV_ROUTING_TABLE",
    "SECTOR_TO_ROUTING_GROUP",
    "MRV_SECTOR_INTENSITY_METRICS",
    # --- Sector Decarb Bridge ---
    "SectorDecarbBridge",
    "SectorDecarbBridgeConfig",
    "LeverCategory",
    "ImplementationPhase",
    "LeverStatus",
    "LeverAnalysis",
    "AbatementWaterfall",
    "ImplementationRoadmap",
    "SECTOR_DECARB_LEVERS",
    # --- Sector Data Bridge ---
    "SectorDataBridge",
    "SectorDataBridgeConfig",
    "IntakeResult",
    "ReconciliationResult",
    "ActivityDataProfile",
    "SECTOR_DATA_AGENT_ROUTING",
    "SECTOR_ACTIVITY_REQUIREMENTS",
    # --- Sector Pathway Setup Wizard ---
    "SectorPathwaySetupWizard",
    "SectorWizardStep",
    "StepStatus",
    "SDAEligibility",
    "ConvergencePreference",
    "ScenarioPreference",
    "TechnologyMaturity",
    "SectorRoutingGroup",
    "SectorSelectionData",
    "ActivityDataSetup",
    "BaselineConfig",
    "PathwayPreferences",
    "TechnologyInventoryData",
    "AbatementPlanData",
    "WizardStepState",
    "WizardState",
    "SectorPathwaySetupResult",
    "STEP_ORDER",
    "STEP_DISPLAY_NAMES",
    "STEP_DESCRIPTIONS",
    "SDA_SECTOR_OPTIONS",
    "EXTENDED_SECTOR_OPTIONS",
    "ALL_SECTOR_OPTIONS",
    "SECTOR_TECHNOLOGY_PROFILES",
    "SECTOR_LEVER_PRIORITIES",
    "ROUTING_GROUP_MRV_PRIORITIES",
    "ROUTING_GROUP_DATA_PRIORITIES",
    # --- Sector Pathway Health Check ---
    "SectorPathwayHealthCheck",
    "HealthCheckConfig",
    "HealthCheckResult",
    "ComponentHealth",
    "HealthSeverity",
    "HealthStatus",
    "CheckCategory",
    "DataFreshnessStatus",
    "RemediationSuggestion",
    "DataFreshnessCheck",
    "SectorCoverageCheck",
    "SECTOR_PATHWAY_ENGINES",
    "SECTOR_PATHWAY_WORKFLOWS",
    "SECTOR_PATHWAY_TEMPLATES",
    "SDA_SECTORS",
    "EXTENDED_SECTORS",
    "ALL_SECTORS",
    "IEA_SCENARIOS",
    "INTEGRATION_BRIDGES",
    "MIGRATION_FILES",
    "SECTOR_INTENSITY_METRICS",
    "QUICK_CHECK_CATEGORIES",
    "CRITICAL_CATEGORIES",
]
