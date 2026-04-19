# -*- coding: utf-8 -*-
"""
PACK-021 Net Zero Starter Pack - Integration Layer
=====================================================

Phase 4 integration layer for the Net Zero Starter Pack that provides
pipeline orchestration, MRV agent routing, GHG inventory management,
SBTi target setting, decarbonisation planning, carbon offset strategy,
cross-framework reporting, data intake, 18-category health verification,
and 6-step guided setup wizard.

Components:
    - NetZeroPipelineOrchestrator: 8-phase net-zero pipeline with DAG
      dependency resolution, conditional offset phase, retry with
      exponential backoff, and SHA-256 provenance tracking
    - MRVBridge: Routes emission sources to 30 MRV agents across
      Scope 1 (8 agents), Scope 2 (5 agents), Scope 3 (15 agents),
      and cross-cutting (2 agents)
    - GHGAppBridge: Bridge to GL-GHG-APP for inventory management,
      base year calculation, scope aggregation, completeness validation,
      and report generation
    - SBTiAppBridge: Bridge to GL-SBTi-APP for science-based target
      setting, pathway calculation, progress tracking, temperature
      scoring, and target validation
    - DecarbBridge: Bridge to 21 DECARB-X agents for abatement options,
      MACC generation, roadmap building, technology assessment, and
      lever-specific planning
    - OffsetBridge: Bridge to carbon credit/offset agents for strategy
      planning, credit valuation, tracking, quality verification, and
      SBTi compliance checking
    - ReportingBridge: Cross-framework reporting to CDP Climate Change,
      TCFD, ESRS E1, and GHG Protocol
    - DataBridge: Routes data intake to 20 DATA agents with ERP field
      mapping for SAP/Oracle/Workday/Dynamics 365
    - NetZeroHealthCheck: 18-category system health verification
    - NetZeroSetupWizard: 6-step guided configuration with 6 sector
      presets

Architecture:
    External Systems --> Integration Bridges --> Platform Components
                              |
                              v
    PACK-021 Net Zero <-- Composition <-- Zero Duplication
                              |
                              v
    MRV Agents <-- DATA Agents <-- DECARB Agents <-- GL-GHG/SBTi APPs

Platform Integrations:
    - greenlang/agents/mrv/* (30 MRV agents)
    - greenlang/agents/data/* (20 DATA agents)
    - greenlang/agents/decarb/* (21 DECARB-X agents)
    - greenlang/agents/foundation/* (10 FOUND agents)
    - greenlang/apps/ghg (GL-GHG-APP)
    - greenlang/apps/sbti (GL-SBTi-APP)
    - greenlang/apps/cdp (GL-CDP-APP)
    - greenlang/apps/tcfd (GL-TCFD-APP)
    - greenlang/apps/csrd (GL-CSRD-APP)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-021 Net Zero Starter Pack
Status: Production Ready
"""

__version__ = "1.0.0"
__pack_id__ = "PACK-021"
__pack_name__ = "Net Zero Starter Pack"

# ---------------------------------------------------------------------------
# Net Zero Pipeline Orchestrator
# ---------------------------------------------------------------------------
from .pack_orchestrator import (
    ExecutionStatus,
    NetZeroPipelineOrchestrator,
    NetZeroPipelinePhase,
    OrchestratorConfig,
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
    CompletenessLevel,
    CompletenessResult,
    GHGAppBridge,
    GHGAppBridgeConfig,
    GHGScope,
    InventoryResult,
    ReportFormat,
    ReportResult,
)

# ---------------------------------------------------------------------------
# SBTi App Bridge
# ---------------------------------------------------------------------------
from .sbti_app_bridge import (
    PathwayResult,
    PathwayType,
    ProgressResult,
    SBTiAppBridge,
    SBTiAppBridgeConfig,
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
    LeverPlanResult,
    MACCResult,
    ProgressMonitorResult,
    RoadmapResult,
    TechnologyReadiness,
    TechnologyResult,
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
)

# ---------------------------------------------------------------------------
# Reporting Bridge
# ---------------------------------------------------------------------------
from .reporting_bridge import (
    FrameworkMappingResult,
    MappingStatus,
    MultiFrameworkReportResult,
    ReportingBridge,
    ReportingBridgeConfig,
    ReportingFramework,
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
    NetZeroHealthCheck,
    RemediationSuggestion,
)

# ---------------------------------------------------------------------------
# Setup Wizard
# ---------------------------------------------------------------------------
from .setup_wizard import (
    AmbitionLevel,
    BoundarySelection,
    ConsolidationApproach,
    DataSourceSetup,
    NetZeroSetupWizard,
    NetZeroWizardStep,
    OrganizationProfile,
    OrganizationSize,
    PresetSelection,
    ScopeConfiguration,
    SetupResult,
    StepStatus,
    TargetPreferences,
    WizardState,
    WizardStepState,
)

__all__ = [
    # Module metadata
    "__version__",
    "__pack_id__",
    "__pack_name__",
    # --- Net Zero Pipeline Orchestrator ---
    "NetZeroPipelineOrchestrator",
    "OrchestratorConfig",
    "RetryConfig",
    "NetZeroPipelinePhase",
    "ExecutionStatus",
    "PhaseProvenance",
    "PhaseResult",
    "PipelineResult",
    "PHASE_DEPENDENCIES",
    "PHASE_EXECUTION_ORDER",
    # --- MRV Bridge ---
    "MRVBridge",
    "MRVBridgeConfig",
    "MRVAgentRoute",
    "EmissionSource",
    "MRVScope",
    "RoutingResult",
    "BatchRoutingResult",
    # --- GHG App Bridge ---
    "GHGAppBridge",
    "GHGAppBridgeConfig",
    "GHGScope",
    "ReportFormat",
    "CompletenessLevel",
    "InventoryResult",
    "BaseYearResult",
    "AggregationResult",
    "CompletenessResult",
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
    "PathwayResult",
    "ProgressResult",
    "TemperatureScoreResult",
    # --- Decarb Bridge ---
    "DecarbBridge",
    "DecarbBridgeConfig",
    "DecarbLever",
    "TechnologyReadiness",
    "AbatementOption",
    "AbatementResult",
    "MACCResult",
    "RoadmapResult",
    "TechnologyResult",
    "LeverPlanResult",
    "ProgressMonitorResult",
    # --- Offset Bridge ---
    "OffsetBridge",
    "OffsetBridgeConfig",
    "CreditType",
    "CreditStandard",
    "QualityTier",
    "SBTiOffsetRole",
    "OffsetStrategyResult",
    "CreditValuationResult",
    "CreditTrackingResult",
    "QualityVerificationResult",
    "SBTiComplianceResult",
    # --- Reporting Bridge ---
    "ReportingBridge",
    "ReportingBridgeConfig",
    "ReportingFramework",
    "MappingStatus",
    "FrameworkMappingResult",
    "MultiFrameworkReportResult",
    # --- Data Bridge ---
    "DataBridge",
    "DataBridgeConfig",
    "DataSourceType",
    "ERPSystem",
    "DataCategory",
    "ERPFieldMapping",
    "IntakeResult",
    "QualityResult",
    # --- Health Check ---
    "NetZeroHealthCheck",
    "HealthCheckConfig",
    "HealthCheckResult",
    "ComponentHealth",
    "HealthSeverity",
    "HealthStatus",
    "CheckCategory",
    "RemediationSuggestion",
    # --- Setup Wizard ---
    "NetZeroSetupWizard",
    "NetZeroWizardStep",
    "StepStatus",
    "ConsolidationApproach",
    "AmbitionLevel",
    "OrganizationSize",
    "OrganizationProfile",
    "BoundarySelection",
    "ScopeConfiguration",
    "DataSourceSetup",
    "TargetPreferences",
    "PresetSelection",
    "WizardStepState",
    "WizardState",
    "SetupResult",
]
