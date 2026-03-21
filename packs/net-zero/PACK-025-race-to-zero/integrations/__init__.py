# -*- coding: utf-8 -*-
"""
PACK-025 Race to Zero Pack - Integration Layer
==================================================

Phase 4 integration layer for the Race to Zero Pack that provides
10-phase DAG pipeline orchestration, MRV agent routing, GHG inventory
management, SBTi target validation with R2Z-specific criteria,
decarbonisation pathway planning, EU Taxonomy alignment, data quality
management, UNFCCC Race to Zero portal integration, CDP disclosure
mapping, GFANZ financial institution pathways, 22-category health
verification, and 8-step guided setup wizard.

Components:
    - RaceToZeroOrchestrator: 10-phase pipeline with DAG dependency
      resolution (onboarding, starting line, action planning,
      implementation, reporting, credibility, partnership, sector
      pathway, verification, continuous improvement), retry with
      exponential backoff, SHA-256 provenance, credibility scoring,
      and starting line criteria tracking
    - MRVBridge: Routes emission sources to 30 MRV agents across
      Scope 1 (8), Scope 2 (5), Scope 3 (15), Cross-Cutting (2)
      with activity-based routing and scope aggregation
    - GHGAppBridge: Bridge to GL-GHG-APP for GHG inventory, base year
      management, scope aggregation, completeness validation, multi-year
      trend analysis, and report generation
    - SBTiAppBridge: Bridge to GL-SBTi-APP with R2Z-specific target
      criteria (50% by 2030, net-zero by 2050), temperature alignment,
      SDA pathway calculation, and sector benchmarking
    - DecarbBridge: Bridge to 21 DECARB-X agents with R2Z-aligned
      filtering (no fossil expansion), MACC generation, roadmap building,
      budget optimization, and reduction prioritization
    - TaxonomyBridge: Bridge to GL-Taxonomy-APP for EU Taxonomy alignment,
      climate transition plan validation, DNSH evaluation, and green
      investment alignment
    - DataBridge: Routes data intake to 20 DATA agents with ERP field
      mapping, supplier data collection, quality profiling, and
      verification readiness assessment
    - UNFCCCBridge: Integration with UNFCCC Race to Zero verification
      portal for commitment submission, verification status tracking,
      annual reporting, badge retrieval, and compliance checking
    - CDPBridge: Integration with CDP disclosure platform for
      questionnaire mapping, automated response generation, score
      estimation, and R2Z/CDP alignment checking
    - GFANZBridge: Integration with GFANZ for financial institution
      pathways, portfolio alignment, financed emissions, transition
      plan evaluation, and sector pathway tracking
    - RaceToZeroSetupWizard: 8-step guided configuration with partner
      initiative selection, credibility preferences, and 8 sector presets
    - RaceToZeroHealthCheck: 22-category system health verification
      including UNFCCC, CDP, GFANZ connectivity and credibility
      criteria database currency

Architecture:
    External Systems --> Integration Bridges --> Platform Components
                              |
                              v
    PACK-025 R2Z <-- Composition <-- Zero Duplication
                              |
                              v
    MRV Agents <-- DATA Agents <-- DECARB Agents <-- GL Apps
                              |
                              v
    UNFCCC R2Z Portal <-- CDP Platform <-- GFANZ Framework

Platform Integrations:
    - greenlang/agents/mrv/* (30 MRV agents)
    - greenlang/agents/data/* (20 DATA agents)
    - greenlang/agents/decarb/* (21 DECARB-X agents)
    - greenlang/agents/foundation/* (10 FOUND agents)
    - greenlang/apps/ghg (GL-GHG-APP)
    - greenlang/apps/sbti (GL-SBTi-APP)
    - greenlang/apps/cdp (GL-CDP-APP)
    - greenlang/apps/taxonomy (GL-Taxonomy-APP)

External Integrations:
    - UNFCCC Race to Zero verification portal
    - CDP Climate Change disclosure platform
    - GFANZ (Glasgow Financial Alliance for Net Zero)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-025 Race to Zero Pack
Status: Production Ready
"""

__version__ = "1.0.0"
__pack_id__ = "PACK-025"
__pack_name__ = "Race to Zero Pack"

# ---------------------------------------------------------------------------
# Race to Zero Pipeline Orchestrator
# ---------------------------------------------------------------------------
from .pack_orchestrator import (
    CredibilityCriteria,
    CredibilityResult,
    ExecutionStatus,
    PartnerType,
    PHASE_DEPENDENCIES,
    PHASE_EXECUTION_ORDER,
    PhaseProvenance,
    PhaseResult,
    PipelineResult,
    RaceToZeroOrchestratorConfig,
    RaceToZeroOrchestrator,
    RaceToZeroPipelinePhase,
    RetryConfig,
    SectorPathwayStatus,
    StartingLineCriteria,
    StartingLineResult,
)

# ---------------------------------------------------------------------------
# MRV Bridge
# ---------------------------------------------------------------------------
from .mrv_bridge import (
    AggregationResult,
    BatchRoutingResult,
    CalculationMethod,
    DataQualityTier,
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
    AggregationResult as GHGAggregationResult,
    BaseYearResult,
    BaseYearValidity,
    CompletenessLevel,
    CompletenessResult,
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
    R2ZTargetCriteria,
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
    BudgetOptimizationResult,
    DecarbBridge,
    DecarbBridgeConfig,
    DecarbLever,
    MACCResult,
    PriorityTier,
    RoadmapResult,
    ScenarioFilter,
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
    GreenInvestmentResult,
    SubstantialContributionLevel,
    SubstantialContributionResult,
    TaxonomyBridge,
    TaxonomyBridgeConfig,
    TaxonomyKPIResult,
    TaxonomyObjective,
    TransitionPlanResult as TaxonomyTransitionPlanResult,
    TransitionPlanStatus,
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
    QualityLevel,
    QualityResult,
    ReconciliationResult,
    SupplierDataResult,
)

# ---------------------------------------------------------------------------
# UNFCCC Bridge
# ---------------------------------------------------------------------------
from .unfccc_bridge import (
    AnnualReportResult,
    AnnualReportSubmission,
    BadgeLevel,
    CommitmentResult,
    CommitmentStatus,
    CommitmentSubmission,
    ComplianceCheckResult,
    OAuthToken,
    RateLimitStatus,
    ReportType,
    UNFCCCBridge,
    UNFCCCBridgeConfig,
    VerificationBadge,
    VerificationOutcome,
    VerificationStatusResult,
)

# ---------------------------------------------------------------------------
# CDP Bridge
# ---------------------------------------------------------------------------
from .cdp_bridge import (
    AlignmentCheckResult,
    CDPBridge,
    CDPBridgeConfig,
    CDPMappingResult,
    CDPResponseResult,
    CDPScore,
    CDPScoreEstimate,
    CDPSection,
    MappingConfidence,
    QuestionnaireMapping,
    ResponseStatus,
    SectionResponse,
)

# ---------------------------------------------------------------------------
# GFANZ Bridge
# ---------------------------------------------------------------------------
from .gfanz_bridge import (
    AlignmentTier,
    FinancedEmissionsResult,
    FinancedEmissionsScope,
    GFANZAlliance,
    GFANZBridge,
    GFANZBridgeConfig,
    PortfolioAlignmentMethod,
    PortfolioAlignmentResult,
    PortfolioProgressResult,
    SectorPathwayResult,
    SectorPriority,
    TransitionPlanElement,
    TransitionPlanResult as GFANZTransitionPlanResult,
)

# ---------------------------------------------------------------------------
# Setup Wizard
# ---------------------------------------------------------------------------
from .setup_wizard import (
    AmbitionLevel,
    BaselineConfiguration,
    ConsolidationApproach,
    CredibilityPreferences,
    DataSourceSetup,
    OrganizationProfile,
    OrganizationSize,
    OrganizationType,
    PartnerInitiativeSelection,
    PresetSelection,
    RaceToZeroSetupWizard,
    RaceToZeroWizardStep,
    ScopeConfiguration,
    SetupResult,
    StepStatus,
    TargetSetting,
    WizardState,
    WizardStepState,
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
    RaceToZeroHealthCheck,
    RemediationSuggestion,
)

__all__ = [
    # Module metadata
    "__version__",
    "__pack_id__",
    "__pack_name__",
    # --- Race to Zero Pipeline Orchestrator ---
    "RaceToZeroOrchestrator",
    "RaceToZeroOrchestratorConfig",
    "RetryConfig",
    "RaceToZeroPipelinePhase",
    "ExecutionStatus",
    "CredibilityCriteria",
    "StartingLineCriteria",
    "PartnerType",
    "SectorPathwayStatus",
    "PhaseProvenance",
    "PhaseResult",
    "PipelineResult",
    "CredibilityResult",
    "StartingLineResult",
    "PHASE_DEPENDENCIES",
    "PHASE_EXECUTION_ORDER",
    # --- MRV Bridge ---
    "MRVBridge",
    "MRVBridgeConfig",
    "MRVAgentRoute",
    "EmissionSource",
    "MRVScope",
    "CalculationMethod",
    "DataQualityTier",
    "RoutingResult",
    "BatchRoutingResult",
    "AggregationResult",
    # --- GHG App Bridge ---
    "GHGAppBridge",
    "GHGAppBridgeConfig",
    "GHGScope",
    "ReportFormat",
    "CompletenessLevel",
    "BaseYearValidity",
    "InventoryResult",
    "BaseYearResult",
    "GHGAggregationResult",
    "CompletenessResult",
    "MultiYearResult",
    "ReportResult",
    # --- SBTi App Bridge ---
    "SBTiAppBridge",
    "SBTiAppBridgeConfig",
    "PathwayType",
    "TargetScope",
    "TargetType",
    "ValidationStatus",
    "R2ZTargetCriteria",
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
    "PriorityTier",
    "AbatementOption",
    "AbatementResult",
    "MACCResult",
    "RoadmapResult",
    "TechnologyResult",
    "BudgetOptimizationResult",
    # --- Taxonomy Bridge ---
    "TaxonomyBridge",
    "TaxonomyBridgeConfig",
    "TaxonomyObjective",
    "AlignmentStatus",
    "DNSHStatus",
    "SubstantialContributionLevel",
    "TransitionPlanStatus",
    "TSCCriteria",
    "AlignmentResult",
    "SubstantialContributionResult",
    "DNSHResult",
    "TaxonomyKPIResult",
    "GreenInvestmentResult",
    "TaxonomyTransitionPlanResult",
    # --- Data Bridge ---
    "DataBridge",
    "DataBridgeConfig",
    "DataSourceType",
    "ERPSystem",
    "DataCategory",
    "QualityLevel",
    "ERPFieldMapping",
    "IntakeResult",
    "QualityResult",
    "ReconciliationResult",
    "SupplierDataResult",
    # --- UNFCCC Bridge ---
    "UNFCCCBridge",
    "UNFCCCBridgeConfig",
    "CommitmentStatus",
    "VerificationOutcome",
    "BadgeLevel",
    "ReportType",
    "RateLimitStatus",
    "OAuthToken",
    "CommitmentSubmission",
    "CommitmentResult",
    "VerificationStatusResult",
    "AnnualReportSubmission",
    "AnnualReportResult",
    "VerificationBadge",
    "ComplianceCheckResult",
    # --- CDP Bridge ---
    "CDPBridge",
    "CDPBridgeConfig",
    "CDPScore",
    "CDPSection",
    "ResponseStatus",
    "MappingConfidence",
    "QuestionnaireMapping",
    "SectionResponse",
    "CDPMappingResult",
    "CDPResponseResult",
    "CDPScoreEstimate",
    "AlignmentCheckResult",
    # --- GFANZ Bridge ---
    "GFANZBridge",
    "GFANZBridgeConfig",
    "GFANZAlliance",
    "PortfolioAlignmentMethod",
    "FinancedEmissionsScope",
    "TransitionPlanElement",
    "AlignmentTier",
    "SectorPriority",
    "PortfolioAlignmentResult",
    "FinancedEmissionsResult",
    "GFANZTransitionPlanResult",
    "SectorPathwayResult",
    "PortfolioProgressResult",
    # --- Setup Wizard ---
    "RaceToZeroSetupWizard",
    "RaceToZeroWizardStep",
    "StepStatus",
    "OrganizationType",
    "OrganizationSize",
    "ConsolidationApproach",
    "AmbitionLevel",
    "OrganizationProfile",
    "PartnerInitiativeSelection",
    "BaselineConfiguration",
    "TargetSetting",
    "ScopeConfiguration",
    "DataSourceSetup",
    "CredibilityPreferences",
    "PresetSelection",
    "WizardStepState",
    "WizardState",
    "SetupResult",
    # --- Health Check ---
    "RaceToZeroHealthCheck",
    "HealthCheckConfig",
    "HealthCheckResult",
    "ComponentHealth",
    "HealthSeverity",
    "HealthStatus",
    "CheckCategory",
    "RemediationSuggestion",
]
