# -*- coding: utf-8 -*-
"""
PACK-022 Net Zero Acceleration Pack - Integration Layer
==========================================================

Phase 4 integration layer for the Net Zero Acceleration Pack that provides
10-phase DAG pipeline orchestration, PACK-021 bridging, MRV agent routing
with activity-based preference, multi-entity GHG inventory management,
SDA pathway calculation, supplier engagement planning, ICVCM/VCMI carbon
market integrity, EU Taxonomy alignment, ISAE 3410 assurance preparation,
cross-framework reporting, bulk supplier data collection, 22-category
health verification, and 8-step guided setup wizard.

Components:
    - NetZeroAccelerationOrchestrator: 10-phase net-zero pipeline with DAG
      dependency resolution, conditional SDA pathway phase, retry with
      exponential backoff, and SHA-256 provenance tracking
    - Pack021Bridge: Bridge to PACK-021 for baseline, targets, gap analysis,
      roadmap, residual budget, offset portfolio, scorecard, and benchmarks
    - MRVBridge: Routes emission sources to 30 MRV agents with
      activity-based routing preferred over spend-based fallback
    - GHGAppBridge: Bridge to GL-GHG-APP for multi-entity inventory,
      base year, scope aggregation, multi-year trends, and reporting
    - SBTiAppBridge: Bridge to GL-SBTi-APP with SDA pathway calculation,
      temperature scoring, and sector benchmark comparison
    - DecarbBridge: Bridge to 21 DECARB-X agents with scenario filtering,
      Monte Carlo simulation, and supplier engagement planning
    - TaxonomyBridge: Bridge to GL-Taxonomy-APP for EU Taxonomy alignment,
      TSC criteria, DNSH evaluation, and taxonomy KPI calculation
    - DataBridge: Routes data intake to 20 DATA agents with bulk supplier
      data collection, multi-entity consolidation, and reconciliation
    - ReportingBridge: Cross-framework reporting to CDP, TCFD, ESRS E1,
      GHG Protocol, and ISAE 3410 assurance
    - OffsetBridge: Carbon credit management with ICVCM Core Carbon
      Principles verification and VCMI Claims Code eligibility
    - NetZeroAccelerationHealthCheck: 22-category system health verification
    - NetZeroAccelerationSetupWizard: 8-step guided configuration with
      SDA sector selection and 8 sector presets

Architecture:
    PACK-021 Outputs --> PACK-022 Acceleration Pipeline
                              |
                              v
    10-Phase DAG: init -> data -> quality -> scenario -> SDA (conditional)
                  -> supplier -> finance -> analytics -> temp_score -> report
                              |
                              v
    MRV Agents <-- DATA Agents <-- DECARB Agents <-- GL Apps (GHG/SBTi/Taxonomy)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-022 Net Zero Acceleration Pack
Status: Production Ready
"""

__version__ = "1.0.0"
__pack_id__ = "PACK-022"
__pack_name__ = "Net Zero Acceleration Pack"

# ---------------------------------------------------------------------------
# Net Zero Acceleration Pipeline Orchestrator
# ---------------------------------------------------------------------------
from .pack_orchestrator import (
    AccelerationOrchestratorConfig,
    AccelerationPipelinePhase,
    ExecutionStatus,
    NetZeroAccelerationOrchestrator,
    PHASE_DEPENDENCIES,
    PHASE_EXECUTION_ORDER,
    PhaseProvenance,
    PhaseResult,
    PipelineResult,
    RetryConfig,
    SDA_SECTORS,
)

# ---------------------------------------------------------------------------
# PACK-021 Bridge
# ---------------------------------------------------------------------------
from .pack021_bridge import (
    BaselineResult,
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
# MRV Bridge
# ---------------------------------------------------------------------------
from .mrv_bridge import (
    BatchRoutingResult,
    CalculationMethod,
    EmissionSource,
    MRVAgentRoute,
    MRVBridge,
    MRVBridgeConfig,
    MRVScope,
    RoutingResult,
)

# ---------------------------------------------------------------------------
# GHG App Bridge
# ---------------------------------------------------------------------------
from .ghg_app_bridge import (
    AggregationResult,
    BaseYearResult,
    GHGAppBridge,
    GHGAppBridgeConfig,
    GHGScope,
    InventoryResult,
    MultiYearResult,
    ReportFormat,
    ReportResult,
)

# ---------------------------------------------------------------------------
# SBTi App Bridge
# ---------------------------------------------------------------------------
from .sbti_app_bridge import (
    PathwayType,
    ProgressResult,
    SBTiAppBridge,
    SBTiAppBridgeConfig,
    SDAPathwayResult,
    SectorBenchmarkResult,
    TargetResult,
    TargetScope,
    TargetType,
    TemperatureScoreResult,
    ValidationResult,
    ValidationStatus,
)

# ---------------------------------------------------------------------------
# Decarb Bridge
# ---------------------------------------------------------------------------
from .decarb_bridge import (
    AbatementOption,
    AbatementResult,
    DecarbBridge,
    DecarbBridgeConfig,
    DecarbLever,
    MACCResult,
    MonteCarloResult,
    RoadmapResult as DecarbRoadmapResult,
    ScenarioFilter,
    SupplierEngagementResult,
    TechnologyReadiness,
    TechnologyResult,
)

# ---------------------------------------------------------------------------
# Taxonomy Bridge
# ---------------------------------------------------------------------------
from .taxonomy_bridge import (
    AlignmentResult,
    AlignmentStatus,
    DNSHResult,
    DNSHStatus,
    SubstantialContributionLevel,
    SubstantialContributionResult,
    TaxonomyBridge,
    TaxonomyBridgeConfig,
    TaxonomyKPIResult,
    TaxonomyObjective,
    TSCCriteria,
)

# ---------------------------------------------------------------------------
# Data Bridge
# ---------------------------------------------------------------------------
from .data_bridge import (
    DataBridge,
    DataBridgeConfig,
    DataCategory,
    DataSourceType,
    ERPFieldMapping,
    ERPSystem,
    IntakeResult,
    QualityResult,
    ReconciliationResult,
    SupplierDataResult,
)

# ---------------------------------------------------------------------------
# Reporting Bridge
# ---------------------------------------------------------------------------
from .reporting_bridge import (
    AssuranceLevel,
    AssuranceReportResult,
    FrameworkMappingResult,
    MappingStatus,
    MultiFrameworkReportResult,
    ReportingBridge,
    ReportingBridgeConfig,
    ReportingFramework,
)

# ---------------------------------------------------------------------------
# Offset Bridge
# ---------------------------------------------------------------------------
from .offset_bridge import (
    CreditStandard,
    CreditTrackingResult,
    CreditType,
    CreditValuationResult,
    OffsetBridge,
    OffsetBridgeConfig,
    OffsetStrategyResult,
    QualityTier,
    QualityVerificationResult,
    SBTiComplianceResult,
    SBTiOffsetRole,
    VCMIClaimTier,
    VCMIEligibilityResult,
)

# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------
from .health_check import (
    CheckCategory,
    ComponentHealth,
    HealthCheckConfig,
    HealthCheckResult,
    HealthSeverity,
    HealthStatus,
    NetZeroAccelerationHealthCheck,
    RemediationSuggestion,
)

# ---------------------------------------------------------------------------
# Setup Wizard
# ---------------------------------------------------------------------------
from .setup_wizard import (
    AccelerationWizardStep,
    AssuranceLevelChoice,
    AssuranceLevelConfig,
    FinanceIntegration,
    NetZeroAccelerationSetupWizard,
    OrganizationProfile,
    OrganizationSize,
    Pack021Status,
    PresetSelection,
    Scope3Strategy,
    SDASectorSelection,
    SetupResult,
    StepStatus,
    SupplierEngagementScope,
    SupplierProgramme,
    WizardState,
    WizardStepState,
)

__all__ = [
    # Module metadata
    "__version__",
    "__pack_id__",
    "__pack_name__",
    # --- Net Zero Acceleration Pipeline Orchestrator ---
    "NetZeroAccelerationOrchestrator",
    "AccelerationOrchestratorConfig",
    "RetryConfig",
    "AccelerationPipelinePhase",
    "ExecutionStatus",
    "PhaseProvenance",
    "PhaseResult",
    "PipelineResult",
    "PHASE_DEPENDENCIES",
    "PHASE_EXECUTION_ORDER",
    "SDA_SECTORS",
    # --- PACK-021 Bridge ---
    "Pack021Bridge",
    "Pack021BridgeConfig",
    "BaselineResult",
    "TargetsResult",
    "GapAnalysisResult",
    "Pack021RoadmapResult",
    "ResidualBudgetResult",
    "OffsetPortfolioResult",
    "ScorecardResult",
    "BenchmarkResult",
    # --- MRV Bridge ---
    "MRVBridge",
    "MRVBridgeConfig",
    "MRVAgentRoute",
    "EmissionSource",
    "MRVScope",
    "CalculationMethod",
    "RoutingResult",
    "BatchRoutingResult",
    # --- GHG App Bridge ---
    "GHGAppBridge",
    "GHGAppBridgeConfig",
    "GHGScope",
    "ReportFormat",
    "InventoryResult",
    "BaseYearResult",
    "AggregationResult",
    "MultiYearResult",
    "ReportResult",
    # --- SBTi App Bridge ---
    "SBTiAppBridge",
    "SBTiAppBridgeConfig",
    "PathwayType",
    "TargetScope",
    "TargetType",
    "ValidationStatus",
    "TargetResult",
    "ValidationResult",
    "SDAPathwayResult",
    "ProgressResult",
    "TemperatureScoreResult",
    "SectorBenchmarkResult",
    # --- Decarb Bridge ---
    "DecarbBridge",
    "DecarbBridgeConfig",
    "DecarbLever",
    "TechnologyReadiness",
    "ScenarioFilter",
    "AbatementOption",
    "AbatementResult",
    "MACCResult",
    "DecarbRoadmapResult",
    "MonteCarloResult",
    "SupplierEngagementResult",
    "TechnologyResult",
    # --- Taxonomy Bridge ---
    "TaxonomyBridge",
    "TaxonomyBridgeConfig",
    "TaxonomyObjective",
    "AlignmentStatus",
    "DNSHStatus",
    "SubstantialContributionLevel",
    "TSCCriteria",
    "AlignmentResult",
    "SubstantialContributionResult",
    "DNSHResult",
    "TaxonomyKPIResult",
    # --- Data Bridge ---
    "DataBridge",
    "DataBridgeConfig",
    "DataSourceType",
    "ERPSystem",
    "DataCategory",
    "ERPFieldMapping",
    "IntakeResult",
    "QualityResult",
    "SupplierDataResult",
    "ReconciliationResult",
    # --- Reporting Bridge ---
    "ReportingBridge",
    "ReportingBridgeConfig",
    "ReportingFramework",
    "MappingStatus",
    "AssuranceLevel",
    "FrameworkMappingResult",
    "AssuranceReportResult",
    "MultiFrameworkReportResult",
    # --- Offset Bridge ---
    "OffsetBridge",
    "OffsetBridgeConfig",
    "CreditType",
    "CreditStandard",
    "QualityTier",
    "SBTiOffsetRole",
    "VCMIClaimTier",
    "OffsetStrategyResult",
    "CreditValuationResult",
    "CreditTrackingResult",
    "QualityVerificationResult",
    "SBTiComplianceResult",
    "VCMIEligibilityResult",
    # --- Health Check ---
    "NetZeroAccelerationHealthCheck",
    "HealthCheckConfig",
    "HealthCheckResult",
    "ComponentHealth",
    "HealthSeverity",
    "HealthStatus",
    "CheckCategory",
    "RemediationSuggestion",
    # --- Setup Wizard ---
    "NetZeroAccelerationSetupWizard",
    "AccelerationWizardStep",
    "StepStatus",
    "AssuranceLevelChoice",
    "OrganizationSize",
    "SupplierEngagementScope",
    "OrganizationProfile",
    "Pack021Status",
    "Scope3Strategy",
    "SDASectorSelection",
    "SupplierProgramme",
    "FinanceIntegration",
    "AssuranceLevelConfig",
    "PresetSelection",
    "WizardStepState",
    "WizardState",
    "SetupResult",
]
