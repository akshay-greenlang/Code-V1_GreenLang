# -*- coding: utf-8 -*-
"""
PACK-024 Carbon Neutral Pack - Integration Layer
====================================================

Phase 4 integration layer for the Carbon Neutral Pack that provides
10-phase DAG pipeline orchestration, GL-GHG-APP bridging, MRV agent
routing with PAS 2060 neutralization mapping, DECARB agent routing,
DATA agent routing, 5-registry credit retirement bridging, credit
marketplace integration, verification body management, PACK-021/023
optional bridging, 20-category health verification, and 6-step
PAS 2060-specific setup wizard.

Components:
    - CarbonNeutralOrchestrator: 10-phase carbon neutrality pipeline with
      DAG dependency resolution, PAS 2060 phase requirements, retry with
      exponential backoff, and SHA-256 provenance tracking
    - CarbonNeutralMRVBridge: Routes emission sources to 30 MRV agents
      for PAS 2060 footprint quantification
    - CarbonNeutralGHGAppBridge: Bridge to GL-GHG-APP for inventory,
      base year, scope aggregation, multi-year trends, and data quality
    - CarbonNeutralDecarbBridge: Bridge to 21 DECARB-X agents for
      reduction planning with MACC, roadmap, and PAS 2060 alignment
    - CarbonNeutralDataBridge: Routes data intake to 20 DATA agents
      with carbon neutrality data requirements
    - CarbonNeutralRegistryBridge: Bridge to 5 carbon credit registries
      (Verra, Gold Standard, ACR, CAR, CDM) for retirement and validation
    - CarbonNeutralCreditMarketplaceBridge: Credit marketplace integration
      with ICVCM CCP quality screening and procurement optimization
    - CarbonNeutralVerificationBodyBridge: Verification body management
      with PAS 2060 evidence package assembly and opinion tracking
    - Pack021Bridge: Optional bridge to PACK-021 for baseline, targets,
      gap analysis, roadmap, residual budget, offsets, and scorecard
    - Pack023Bridge: Optional bridge to PACK-023 for SBTi targets,
      pathway data, temperature scoring, and progress tracking
    - CarbonNeutralHealthCheck: 20-category system health verification
    - CarbonNeutralSetupWizard: 6-step guided PAS 2060 configuration

Architecture:
    PAS 2060 Commitment --> 10-Phase DAG Pipeline
                                  |
                                  v
    footprint -> mgmt_plan -> credit_sourcing -> quality_check
    -> retirement -> neutralization -> claims -> verification
    -> annual_cycle -> compliance_check
                                  |
                                  v
    MRV Agents <-- DATA Agents <-- DECARB Agents
                                  |
                                  v
    GL-GHG-APP <-- Registries <-- Marketplace <-- Verification Bodies
                                  |
                                  v
    PACK-021 (optional) <-- PACK-023 (optional)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-024 Carbon Neutral Pack
Status: Production Ready
"""

__version__ = "1.0.0"
__pack_id__ = "PACK-024"
__pack_name__ = "Carbon Neutral Pack"

# ---------------------------------------------------------------------------
# Carbon Neutral Pipeline Orchestrator
# ---------------------------------------------------------------------------
from .pack_orchestrator import (
    CarbonNeutralOrchestrator,
    CarbonNeutralOrchestratorConfig,
    CarbonNeutralPhase,
    ExecutionStatus,
    PAS2060_PHASE_REQUIREMENTS,
    PHASE_DEPENDENCIES,
    PHASE_EXECUTION_ORDER,
    PhaseProvenance,
    PhaseResult,
    PipelineResult,
    RetryConfig,
)

# ---------------------------------------------------------------------------
# MRV Bridge
# ---------------------------------------------------------------------------
from .mrv_bridge import (
    BatchRoutingResult,
    CarbonNeutralMRVBridge,
    EmissionSource,
    FLAG_RELEVANT_SOURCES,
    FLAGEmissionsResult,
    MRVAgentRoute,
    MRVBridgeConfig,
    MRVScope,
    MRV_ROUTING_TABLE,
    RoutingResult,
    SBTiInventoryResult,
    SBTiTargetBoundary,
    Scope3ScreeningResult,
)

# ---------------------------------------------------------------------------
# GHG App Bridge
# ---------------------------------------------------------------------------
from .ghg_app_bridge import (
    AggregationResult,
    BaseYearResult,
    CarbonNeutralGHGAppBridge,
    DataQualityResult,
    GHGAppBridgeConfig,
    GHGScope,
    InventoryResult,
    InventoryStatus,
    MultiYearResult,
    RecalculationResult,
    RecalculationTrigger,
    ReportFormat,
    ReportResult,
)

# ---------------------------------------------------------------------------
# DECARB Bridge
# ---------------------------------------------------------------------------
from .decarb_bridge import (
    AbatementOption,
    AbatementResult,
    BusinessCaseResult,
    CarbonNeutralDecarbBridge,
    DecarbBridgeConfig,
    DecarbLever,
    DECARB_ROUTING_TABLE,
    MACCResult,
    PAS2060ReductionAlignment,
    ProgressMonitorResult,
    RoadmapResult as DecarbRoadmapResult,
    TechnologyReadiness,
    TechnologyResult,
)

# ---------------------------------------------------------------------------
# DATA Bridge
# ---------------------------------------------------------------------------
from .data_bridge import (
    CarbonNeutralDataBridge,
    CN_DATA_ROUTING,
    DataBridgeConfig,
    DataCategory,
    DataSourceType,
    DATA_ROUTING_TABLE,
    ERPSystem,
    FreshnessResult,
    IntakeResult,
    LineageResult,
    QualityDimension,
    QualityResult,
    ReconciliationResult,
)

# ---------------------------------------------------------------------------
# Registry Bridge
# ---------------------------------------------------------------------------
from .registry_bridge import (
    BatchRetirementResult,
    CarbonNeutralRegistryBridge,
    CreditRecord,
    CreditStatus,
    PortfolioValidationResult,
    REGISTRY_ENDPOINTS,
    RegistryBridgeConfig,
    RegistryName,
    RegistryVerificationResult,
    RetirementPurpose,
    RetirementRequest,
    RetirementResult,
    VerificationStatus,
)

# ---------------------------------------------------------------------------
# Credit Marketplace Bridge
# ---------------------------------------------------------------------------
from .credit_marketplace_bridge import (
    CarbonNeutralCreditMarketplaceBridge,
    CreditListing,
    CreditType,
    CREDIT_PRICE_RANGES,
    ICVCM_CCP_CRITERIA,
    Marketplace,
    MarketplaceBridgeConfig,
    MarketSearchResult,
    PricingAnalysisResult,
    ProcurementRecommendation,
    ProjectCategory,
    QualityTier,
)

# ---------------------------------------------------------------------------
# Verification Body Bridge
# ---------------------------------------------------------------------------
from .verification_body_bridge import (
    AssuranceLevel,
    CarbonNeutralVerificationBodyBridge,
    EngagementStatus,
    EvidencePackage,
    FindingSeverity,
    OpinionType,
    PAS_2060_VERIFICATION_CHECKLIST,
    VERIFICATION_BODIES,
    VerificationBodyBridgeConfig,
    VerificationEngagement,
    VerificationFinding,
    VerificationOpinion,
    VerificationScope,
)

# ---------------------------------------------------------------------------
# PACK-021 Bridge
# ---------------------------------------------------------------------------
from .pack021_bridge import (
    BaselineResult as Pack021BaselineResult,
    BenchmarkResult,
    GapAnalysisResult,
    OffsetPortfolioResult,
    Pack021Bridge,
    Pack021BridgeConfig,
    ResidualBudgetResult,
    RoadmapResult as Pack021RoadmapResult,
    ScorecardResult,
    TargetsResult,
)

# ---------------------------------------------------------------------------
# PACK-023 Bridge
# ---------------------------------------------------------------------------
from .pack023_bridge import (
    FLAGResult,
    Pack023Bridge,
    Pack023BridgeConfig,
    PathwayResult,
    ProgressResult,
    SBTiTargetResult,
    Scope3ScreeningResult as Pack023Scope3ScreeningResult,
    TemperatureScoreResult,
    ValidationResult,
)

# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------
from .health_check import (
    CarbonNeutralHealthCheck,
    CheckCategory,
    ComponentHealth,
    HealthCheckConfig,
    HealthCheckResult,
    HealthSeverity,
    HealthStatus,
    RemediationSuggestion,
)

# ---------------------------------------------------------------------------
# Setup Wizard
# ---------------------------------------------------------------------------
from .setup_wizard import (
    BoundarySelection,
    BoundaryType,
    CarbonNeutralSetupWizard,
    CarbonNeutralWizardStep,
    ConsolidationApproach,
    CreditPreferences,
    CreditQualityPreference,
    DataSourceSetup,
    EmissionFactorSource,
    OrganizationProfile,
    OrganizationSize,
    PresetSelection,
    Scope2Method,
    ScopeConfiguration,
    SECTOR_PRESETS,
    SetupResult,
    StepStatus,
    WizardState,
    WizardStepState,
)

__all__ = [
    # Module metadata
    "__version__",
    "__pack_id__",
    "__pack_name__",
    # --- Carbon Neutral Pipeline Orchestrator ---
    "CarbonNeutralOrchestrator",
    "CarbonNeutralOrchestratorConfig",
    "RetryConfig",
    "CarbonNeutralPhase",
    "ExecutionStatus",
    "PhaseProvenance",
    "PhaseResult",
    "PipelineResult",
    "PHASE_DEPENDENCIES",
    "PHASE_EXECUTION_ORDER",
    "PAS2060_PHASE_REQUIREMENTS",
    # --- MRV Bridge ---
    "CarbonNeutralMRVBridge",
    "MRVBridgeConfig",
    "MRVAgentRoute",
    "EmissionSource",
    "MRVScope",
    "SBTiTargetBoundary",
    "RoutingResult",
    "BatchRoutingResult",
    "SBTiInventoryResult",
    "Scope3ScreeningResult",
    "FLAGEmissionsResult",
    "MRV_ROUTING_TABLE",
    "FLAG_RELEVANT_SOURCES",
    # --- GHG App Bridge ---
    "CarbonNeutralGHGAppBridge",
    "GHGAppBridgeConfig",
    "GHGScope",
    "ReportFormat",
    "RecalculationTrigger",
    "InventoryStatus",
    "InventoryResult",
    "BaseYearResult",
    "AggregationResult",
    "MultiYearResult",
    "RecalculationResult",
    "DataQualityResult",
    "ReportResult",
    # --- DECARB Bridge ---
    "CarbonNeutralDecarbBridge",
    "DecarbBridgeConfig",
    "DecarbLever",
    "TechnologyReadiness",
    "PAS2060ReductionAlignment",
    "DECARB_ROUTING_TABLE",
    "AbatementOption",
    "AbatementResult",
    "MACCResult",
    "DecarbRoadmapResult",
    "TechnologyResult",
    "ProgressMonitorResult",
    "BusinessCaseResult",
    # --- DATA Bridge ---
    "CarbonNeutralDataBridge",
    "DataBridgeConfig",
    "DataSourceType",
    "ERPSystem",
    "DataCategory",
    "QualityDimension",
    "DATA_ROUTING_TABLE",
    "CN_DATA_ROUTING",
    "IntakeResult",
    "QualityResult",
    "LineageResult",
    "FreshnessResult",
    "ReconciliationResult",
    # --- Registry Bridge ---
    "CarbonNeutralRegistryBridge",
    "RegistryBridgeConfig",
    "RegistryName",
    "CreditStatus",
    "RetirementPurpose",
    "VerificationStatus",
    "REGISTRY_ENDPOINTS",
    "CreditRecord",
    "RegistryVerificationResult",
    "RetirementRequest",
    "RetirementResult",
    "BatchRetirementResult",
    "PortfolioValidationResult",
    # --- Credit Marketplace Bridge ---
    "CarbonNeutralCreditMarketplaceBridge",
    "MarketplaceBridgeConfig",
    "Marketplace",
    "CreditType",
    "ProjectCategory",
    "QualityTier",
    "ICVCM_CCP_CRITERIA",
    "CREDIT_PRICE_RANGES",
    "CreditListing",
    "MarketSearchResult",
    "ProcurementRecommendation",
    "PricingAnalysisResult",
    # --- Verification Body Bridge ---
    "CarbonNeutralVerificationBodyBridge",
    "VerificationBodyBridgeConfig",
    "AssuranceLevel",
    "OpinionType",
    "FindingSeverity",
    "VerificationScope",
    "EngagementStatus",
    "VERIFICATION_BODIES",
    "PAS_2060_VERIFICATION_CHECKLIST",
    "VerificationFinding",
    "EvidencePackage",
    "VerificationEngagement",
    "VerificationOpinion",
    # --- PACK-021 Bridge ---
    "Pack021Bridge",
    "Pack021BridgeConfig",
    "Pack021BaselineResult",
    "TargetsResult",
    "GapAnalysisResult",
    "Pack021RoadmapResult",
    "ResidualBudgetResult",
    "OffsetPortfolioResult",
    "ScorecardResult",
    "BenchmarkResult",
    # --- PACK-023 Bridge ---
    "Pack023Bridge",
    "Pack023BridgeConfig",
    "SBTiTargetResult",
    "PathwayResult",
    "TemperatureScoreResult",
    "ProgressResult",
    "Pack023Scope3ScreeningResult",
    "FLAGResult",
    "ValidationResult",
    # --- Health Check ---
    "CarbonNeutralHealthCheck",
    "HealthCheckConfig",
    "HealthCheckResult",
    "ComponentHealth",
    "HealthSeverity",
    "HealthStatus",
    "CheckCategory",
    "RemediationSuggestion",
    # --- Setup Wizard ---
    "CarbonNeutralSetupWizard",
    "CarbonNeutralWizardStep",
    "StepStatus",
    "ConsolidationApproach",
    "OrganizationSize",
    "BoundaryType",
    "Scope2Method",
    "CreditQualityPreference",
    "EmissionFactorSource",
    "SECTOR_PRESETS",
    "OrganizationProfile",
    "BoundarySelection",
    "ScopeConfiguration",
    "DataSourceSetup",
    "CreditPreferences",
    "PresetSelection",
    "WizardStepState",
    "WizardState",
    "SetupResult",
]
