# -*- coding: utf-8 -*-
"""
PACK-023 SBTi Alignment Pack - Integration Layer
====================================================

Phase 4 integration layer for the SBTi Alignment Pack that provides
10-phase DAG pipeline orchestration, GL-SBTi-APP bridging, MRV agent
routing with SBTi target boundary mapping, GL-GHG-APP inventory
management, PACK-021/022 optional bridging, DECARB agent routing,
DATA agent routing, cross-framework reporting (CDP/TCFD/ESRS/ISO),
carbon credit management with SBTi Net-Zero Standard compliance,
20-category health verification, and 6-step SBTi-specific setup wizard.

Components:
    - SBTiAlignmentOrchestrator: 10-phase SBTi pipeline with DAG
      dependency resolution, conditional FLAG and FI phases, retry with
      exponential backoff, and SHA-256 provenance tracking
    - SBTiAppBridge: Bridge to GL-SBTi-APP 14 engines for target
      configuration, validation, pathway calculation, temperature scoring,
      progress tracking, SDA/FLAG/FI assessment, and crosswalk
    - SBTiMRVBridge: Routes emission sources to 30 MRV agents with
      SBTi target boundary mapping, FLAG relevance tagging, and
      Scope 3 materiality screening data collection
    - SBTiGHGAppBridge: Bridge to GL-GHG-APP for inventory, base year,
      scope aggregation, multi-year trends, recalculation triggers, and
      data quality assessment
    - Pack021Bridge: Optional bridge to PACK-021 for baseline, targets,
      gap analysis, roadmap, residual budget, offsets, and scorecard
    - Pack022Bridge: Optional bridge to PACK-022 for scenarios, SDA
      pathway, temperature scoring, supplier engagement, multi-entity,
      MACC, finance, and analytics
    - SBTiDecarbBridge: Bridge to 21 DECARB-X agents for SBTi-aligned
      reduction planning with MACC, roadmap, technology assessment, and
      progress monitoring
    - SBTiDataBridge: Routes data intake to 20 DATA agents with SBTi
      activity data requirements and quality assessment
    - SBTiReportingBridge: Cross-framework reporting to CDP C4, TCFD,
      ESRS E1, GHG Protocol, and ISO 14064
    - SBTiOffsetBridge: Carbon credit management enforcing SBTi Net-Zero
      Standard neutralization rules, ICVCM CCP, and VCMI Claims Code
    - SBTiHealthCheck: 20-category system health verification
    - SBTiAlignmentSetupWizard: 6-step guided SBTi configuration with
      ACA/SDA/FLAG/FINZ pathway selection and 10 sector presets

Architecture:
    SBTi Commitment --> 10-Phase DAG Pipeline
                              |
                              v
    commitment -> inventory -> screening -> pathway -> flag (conditional)
    -> target_def -> validation -> fi (conditional) -> readiness -> report
                              |
                              v
    MRV Agents <-- DATA Agents <-- DECARB Agents
                              |
                              v
    GL Apps (GHG/SBTi) <-- PACK-021 (optional) <-- PACK-022 (optional)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-023 SBTi Alignment Pack
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack__: str = "PACK-023"
__pack_name__: str = "SBTi Alignment Pack"
__integrations_count__: int = 12

_loaded_integrations: list[str] = []

# ---------------------------------------------------------------------------
# SBTi Alignment Pipeline Orchestrator
# ---------------------------------------------------------------------------
_ORCHESTRATOR_SYMBOLS: list[str] = [
    "SBTiAlignmentOrchestrator",
    "SBTiOrchestratorConfig",
    "RetryConfig",
    "SBTiPipelinePhase",
    "ExecutionStatus",
    "PhaseProvenance",
    "PhaseResult",
    "PipelineResult",
    "PHASE_DEPENDENCIES",
    "PHASE_EXECUTION_ORDER",
    "FLAG_TRIGGER_THRESHOLD_PCT",
    "ACA_REQUIREMENTS",
]
try:
    from .pack_orchestrator import (
        SBTiAlignmentOrchestrator,
        SBTiOrchestratorConfig,
        RetryConfig,
        SBTiPipelinePhase,
        ExecutionStatus,
        PhaseProvenance,
        PhaseResult,
        PipelineResult,
        PHASE_DEPENDENCIES,
        PHASE_EXECUTION_ORDER,
        FLAG_TRIGGER_THRESHOLD_PCT,
        ACA_REQUIREMENTS,
        SDA_SECTORS as ORCHESTRATOR_SDA_SECTORS,
    )
    _loaded_integrations.append("SBTiAlignmentOrchestrator")
except ImportError as e:
    logger.debug("Integration (SBTiAlignmentOrchestrator) not available: %s", e)
    _ORCHESTRATOR_SYMBOLS = []

# ---------------------------------------------------------------------------
# SBTi App Bridge
# ---------------------------------------------------------------------------
_SBTI_APP_BRIDGE_SYMBOLS: list[str] = [
    "SBTiAppBridge",
    "SBTiAppBridgeConfig",
    "PathwayType",
    "PathwayMethod",
    "TargetScope",
    "TargetType",
    "TargetResult",
    "ValidationResult",
    "PathwayResult",
    "TemperatureResult",
    "ProgressResult",
    "SectorResult",
    "FLAGResult",
    "FIResult",
    "CrosswalkResult",
]
try:
    from .sbti_app_bridge import (
        SBTiAppBridge,
        SBTiAppBridgeConfig,
        PathwayType,
        PathwayMethod,
        TargetScope,
        TargetType,
        TargetResult,
        ValidationResult,
        PathwayResult,
        TemperatureResult,
        ProgressResult,
        SectorResult,
        FLAGResult,
        FIResult,
        CrosswalkResult,
        SDA_2050_BENCHMARKS as SBTI_SDA_2050_BENCHMARKS,
        SDA_ACTIVITY_METRICS as SBTI_SDA_ACTIVITY_METRICS,
    )
    _loaded_integrations.append("SBTiAppBridge")
except ImportError as e:
    logger.debug("Integration (SBTiAppBridge) not available: %s", e)
    _SBTI_APP_BRIDGE_SYMBOLS = []

# ---------------------------------------------------------------------------
# MRV Bridge
# ---------------------------------------------------------------------------
_MRV_BRIDGE_SYMBOLS: list[str] = [
    "SBTiMRVBridge",
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
]
try:
    from .mrv_bridge import (
        SBTiMRVBridge,
        MRVBridgeConfig,
        MRVAgentRoute,
        EmissionSource,
        MRVScope,
        MRV_ROUTING_TABLE,
        RoutingResult,
        BatchRoutingResult,
        SBTiInventoryResult,
        SBTiTargetBoundary,
        Scope3ScreeningResult,
        FLAGEmissionsResult,
        FLAG_RELEVANT_SOURCES,
    )
    _loaded_integrations.append("SBTiMRVBridge")
except ImportError as e:
    logger.debug("Integration (SBTiMRVBridge) not available: %s", e)
    _MRV_BRIDGE_SYMBOLS = []

# ---------------------------------------------------------------------------
# GHG App Bridge
# ---------------------------------------------------------------------------
_GHG_APP_BRIDGE_SYMBOLS: list[str] = [
    "SBTiGHGAppBridge",
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
]
try:
    from .ghg_app_bridge import (
        SBTiGHGAppBridge,
        GHGAppBridgeConfig,
        GHGScope,
        InventoryResult,
        BaseYearResult,
        AggregationResult,
        MultiYearResult,
        RecalculationResult,
        RecalculationTrigger,
        ReportFormat,
        InventoryStatus,
        DataQualityResult,
        ReportResult,
    )
    _loaded_integrations.append("SBTiGHGAppBridge")
except ImportError as e:
    logger.debug("Integration (SBTiGHGAppBridge) not available: %s", e)
    _GHG_APP_BRIDGE_SYMBOLS = []

# ---------------------------------------------------------------------------
# PACK-021 Bridge
# ---------------------------------------------------------------------------
_PACK021_BRIDGE_SYMBOLS: list[str] = [
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
]
try:
    from .pack021_bridge import (
        Pack021Bridge,
        Pack021BridgeConfig,
        BaselineResult,
        TargetsResult,
        GapAnalysisResult,
        ResidualBudgetResult,
        OffsetPortfolioResult,
        ScorecardResult,
        BenchmarkResult,
        RoadmapResult as Pack021RoadmapResult,
    )
    _loaded_integrations.append("Pack021Bridge")
except ImportError as e:
    logger.debug("Integration (Pack021Bridge) not available: %s", e)
    _PACK021_BRIDGE_SYMBOLS = []

# ---------------------------------------------------------------------------
# PACK-022 Bridge
# ---------------------------------------------------------------------------
_PACK022_BRIDGE_SYMBOLS: list[str] = [
    "Pack022Bridge",
    "Pack022BridgeConfig",
    "ScenarioType",
    "TemperatureHorizon",
    "AggregationMethod",
    "ScenarioResult",
    "Pack022SDAPathwayResult",
    "Pack022TemperatureScoreResult",
    "Pack022SupplierEngagementResult",
    "MultiEntityResult",
    "Pack022MACCResult",
    "FinanceMetricsResult",
    "AnalyticsResult",
]
try:
    from .pack022_bridge import (
        Pack022Bridge,
        Pack022BridgeConfig,
        AggregationMethod,
        AnalyticsResult,
        FinanceMetricsResult,
        MultiEntityResult,
        ScenarioType,
        TemperatureHorizon,
        ScenarioResult,
        MACCResult as Pack022MACCResult,
        SDAPathwayResult as Pack022SDAPathwayResult,
        SupplierEngagementResult as Pack022SupplierEngagementResult,
        TemperatureScoreResult as Pack022TemperatureScoreResult,
    )
    _loaded_integrations.append("Pack022Bridge")
except ImportError as e:
    logger.debug("Integration (Pack022Bridge) not available: %s", e)
    _PACK022_BRIDGE_SYMBOLS = []

# ---------------------------------------------------------------------------
# Decarb Bridge
# ---------------------------------------------------------------------------
_DECARB_BRIDGE_SYMBOLS: list[str] = [
    "SBTiDecarbBridge",
    "DecarbBridgeConfig",
    "DecarbLever",
    "TechnologyReadiness",
    "SBTiPathwayAlignment",
    "AbatementOption",
    "AbatementResult",
    "DecarbMACCResult",
    "DecarbRoadmapResult",
    "TechnologyResult",
    "ProgressMonitorResult",
    "BusinessCaseResult",
]
try:
    from .decarb_bridge import (
        SBTiDecarbBridge,
        DecarbBridgeConfig,
        DecarbLever,
        TechnologyReadiness,
        SBTiPathwayAlignment,
        AbatementOption,
        AbatementResult,
        TechnologyResult,
        ProgressMonitorResult,
        BusinessCaseResult,
        MACCResult as DecarbMACCResult,
        RoadmapResult as DecarbRoadmapResult,
    )
    _loaded_integrations.append("SBTiDecarbBridge")
except ImportError as e:
    logger.debug("Integration (SBTiDecarbBridge) not available: %s", e)
    _DECARB_BRIDGE_SYMBOLS = []

# ---------------------------------------------------------------------------
# Data Bridge
# ---------------------------------------------------------------------------
_DATA_BRIDGE_SYMBOLS: list[str] = [
    "SBTiDataBridge",
    "DataBridgeConfig",
    "DataCategory",
    "DataSourceType",
    "ERPFieldMapping",
    "ERPSystem",
    "FreshnessResult",
    "IntakeResult",
    "LineageResult",
    "QualityDimension",
    "QualityResult",
    "ReconciliationResult",
]
try:
    from .data_bridge import (
        SBTiDataBridge,
        DataBridgeConfig,
        DataCategory,
        DataSourceType,
        ERPFieldMapping,
        ERPSystem,
        FreshnessResult,
        IntakeResult,
        LineageResult,
        QualityDimension,
        QualityResult,
        ReconciliationResult,
    )
    _loaded_integrations.append("SBTiDataBridge")
except ImportError as e:
    logger.debug("Integration (SBTiDataBridge) not available: %s", e)
    _DATA_BRIDGE_SYMBOLS = []

# ---------------------------------------------------------------------------
# Reporting Bridge
# ---------------------------------------------------------------------------
_REPORTING_BRIDGE_SYMBOLS: list[str] = [
    "SBTiReportingBridge",
    "ReportingBridgeConfig",
    "ReportingFramework",
    "MappingStatus",
    "FrameworkMappingResult",
    "MultiFrameworkReportResult",
]
try:
    from .reporting_bridge import (
        SBTiReportingBridge,
        ReportingBridgeConfig,
        ReportingFramework,
        MappingStatus,
        FrameworkMappingResult,
        MultiFrameworkReportResult,
    )
    _loaded_integrations.append("SBTiReportingBridge")
except ImportError as e:
    logger.debug("Integration (SBTiReportingBridge) not available: %s", e)
    _REPORTING_BRIDGE_SYMBOLS = []

# ---------------------------------------------------------------------------
# Offset Bridge
# ---------------------------------------------------------------------------
_OFFSET_BRIDGE_SYMBOLS: list[str] = [
    "SBTiOffsetBridge",
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
    "BVCMResult",
]
try:
    from .offset_bridge import (
        SBTiOffsetBridge,
        OffsetBridgeConfig,
        CreditType,
        CreditStandard,
        QualityTier,
        SBTiOffsetRole,
        VCMIClaimTier,
        OffsetStrategyResult,
        CreditValuationResult,
        CreditTrackingResult,
        QualityVerificationResult,
        SBTiComplianceResult,
        BVCMResult,
    )
    _loaded_integrations.append("SBTiOffsetBridge")
except ImportError as e:
    logger.debug("Integration (SBTiOffsetBridge) not available: %s", e)
    _OFFSET_BRIDGE_SYMBOLS = []

# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------
_HEALTH_CHECK_SYMBOLS: list[str] = [
    "SBTiHealthCheck",
    "HealthCheckConfig",
    "HealthCheckResult",
    "ComponentHealth",
    "HealthSeverity",
    "HealthStatus",
    "CheckCategory",
    "RemediationSuggestion",
]
try:
    from .health_check import (
        SBTiHealthCheck,
        HealthCheckConfig,
        HealthCheckResult,
        ComponentHealth,
        HealthSeverity,
        HealthStatus,
        CheckCategory,
        RemediationSuggestion,
    )
    _loaded_integrations.append("SBTiHealthCheck")
except ImportError as e:
    logger.debug("Integration (SBTiHealthCheck) not available: %s", e)
    _HEALTH_CHECK_SYMBOLS = []

# ---------------------------------------------------------------------------
# Setup Wizard
# ---------------------------------------------------------------------------
_SETUP_WIZARD_SYMBOLS: list[str] = [
    "SBTiAlignmentSetupWizard",
    "SBTiWizardStep",
    "StepStatus",
    "ConsolidationApproach",
    "OrganizationSize",
    "SBTiPathwayType",
    "SBTiAmbitionLevel",
    "OrganizationType",
    "FLAGExposure",
    "EmissionFactorSource",
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
try:
    from .setup_wizard import (
        SBTiAlignmentSetupWizard,
        SBTiWizardStep,
        StepStatus,
        ConsolidationApproach,
        OrganizationSize,
        SBTiPathwayType,
        SBTiAmbitionLevel,
        OrganizationType,
        FLAGExposure,
        EmissionFactorSource,
        OrganizationProfile,
        BoundarySelection,
        ScopeConfiguration,
        DataSourceSetup,
        TargetPreferences,
        PresetSelection,
        WizardStepState,
        WizardState,
        SetupResult,
    )
    _loaded_integrations.append("SBTiAlignmentSetupWizard")
except ImportError as e:
    logger.debug("Integration (SBTiAlignmentSetupWizard) not available: %s", e)
    _SETUP_WIZARD_SYMBOLS = []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_METADATA_SYMBOLS: list[str] = [
    "__version__",
    "__pack__",
    "__pack_name__",
    "__integrations_count__",
]

__all__: list[str] = [
    *_METADATA_SYMBOLS,
    *_ORCHESTRATOR_SYMBOLS,
    "ORCHESTRATOR_SDA_SECTORS",
    *_SBTI_APP_BRIDGE_SYMBOLS,
    "SBTI_SDA_2050_BENCHMARKS",
    "SBTI_SDA_ACTIVITY_METRICS",
    *_MRV_BRIDGE_SYMBOLS,
    *_GHG_APP_BRIDGE_SYMBOLS,
    *_PACK021_BRIDGE_SYMBOLS,
    *_PACK022_BRIDGE_SYMBOLS,
    *_DECARB_BRIDGE_SYMBOLS,
    *_DATA_BRIDGE_SYMBOLS,
    *_REPORTING_BRIDGE_SYMBOLS,
    *_OFFSET_BRIDGE_SYMBOLS,
    *_HEALTH_CHECK_SYMBOLS,
    *_SETUP_WIZARD_SYMBOLS,
]


def get_loaded_integrations() -> list[str]:
    """Return names of successfully loaded integration classes."""
    return list(_loaded_integrations)


def get_integration_count() -> int:
    """Return number of successfully loaded integrations."""
    return len(_loaded_integrations)


logger.info(
    "PACK-023 SBTi Alignment integrations: %d/%d loaded",
    len(_loaded_integrations),
    __integrations_count__,
)
