# -*- coding: utf-8 -*-
"""
PACK-002 CSRD Professional Pack - Integration Layer
=====================================================

Phase 4 integration layer for the CSRD Professional Pack that extends
PACK-001 Starter with enterprise-grade features: multi-entity consolidation,
cross-framework alignment, webhook notifications, approval chain integration,
quality gate enforcement, and enhanced MRV routing.

Components:
    - ProfessionalPackOrchestrator: Enhanced orchestrator with retry, checkpoint,
      multi-entity dispatch, quality gates, and approval chain
    - ProfessionalMRVBridge: Extended MRV bridge with intensity metrics, biogenic
      carbon, base year recalculation, and Scope 3 screening
    - CrossFrameworkBridge: Routes ESRS data to CDP/TCFD/SBTi/EU Taxonomy/GRI/SASB
    - WebhookManager: Multi-channel event notification with HMAC signing
    - ProfessionalSetupWizard: 7-step guided setup for professional deployments
    - ProfessionalHealthCheck: 10-category health verification system

Architecture:
    Data Sources --> ProfessionalPackOrchestrator --> Quality Gates
                                  |                        |
                                  v                        v
    ESRS Metrics <-- ProfessionalMRVBridge <-- MRV Agents
                                  |
                                  v
    CDP/TCFD/SBTi <-- CrossFrameworkBridge --> Coverage Matrix
                                  |
                                  v
    Stakeholders <-- WebhookManager <-- Event Bus

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-002 CSRD Professional
Status: Production Ready
"""

from packs.eu_compliance.PACK_002_csrd_professional.integrations.pack_orchestrator import (
    AgentStatus,
    AgentStatusCode,
    ApprovalConfig,
    OrchestratorConfig,
    PackStatus,
    PhaseDataStore,
    ProfessionalPackOrchestrator,
    ProgressCallback,
    QualityGateConfig,
    QualityGateId,
    QualityGateResult,
    QualityGateStatus,
    RetryConfig,
    WebhookConfig,
    WebhookEvent,
    WorkflowCheckpoint,
    WorkflowExecution,
    WorkflowPhase,
    WorkflowType,
)
from packs.eu_compliance.PACK_002_csrd_professional.integrations.mrv_bridge import (
    AggregatedEmissions,
    BaseYearConfig,
    BaseYearResult,
    BiogenicCarbonResult,
    CalculationResult,
    CalculationStatus,
    CategoryScreening,
    EntityCalculationResult,
    IntensityMetricType,
    IntensityMetrics,
    MRVRoutingEntry,
    ProfessionalMRVBridge,
    ProfessionalMRVBridgeConfig,
    ProvenanceChainEntry,
    ScreeningSignificance,
    ScopeType,
)
from packs.eu_compliance.PACK_002_csrd_professional.integrations.cross_framework_bridge import (
    CDPScore,
    CDPScoringResult,
    CrossFrameworkBridge,
    CrossFrameworkBridgeConfig,
    CrossFrameworkResult,
    FrameworkId,
    FrameworkMapping,
    FrameworkMappingResult,
    Gap,
    GapSeverity,
    MappingStatus,
    SBTiTemperatureResult,
    ScenarioResult,
    TaxonomyAlignmentResult,
)
from packs.eu_compliance.PACK_002_csrd_professional.integrations.webhook_manager import (
    DeadLetterEntry,
    DeliveryResult,
    DeliveryStats,
    DeliveryStatus,
    WebhookChannel,
    WebhookEventType,
    WebhookManager,
    WebhookManagerConfig,
    WebhookSubscription,
)
from packs.eu_compliance.PACK_002_csrd_professional.integrations.setup_wizard import (
    AssuranceLevel,
    CompanyProfile,
    ConsolidationApproach,
    DataSourceConfig,
    PresetRecommendation,
    ProfessionalFeaturesConfig,
    ProfessionalSetupWizard,
    ReportingScope,
    SectorCode,
    SetupReport,
    StepStatus,
    SubsidiaryEntity,
    WizardState,
    WizardStep,
    WizardStepName,
)
from packs.eu_compliance.PACK_002_csrd_professional.integrations.health_check import (
    CheckCategory,
    ComponentHealth,
    HealthCheckResult,
    HealthSeverity,
    HealthStatus,
    ProfessionalHealthCheck,
    ProfessionalHealthCheckConfig,
    RemediationSuggestion,
)

__all__ = [
    # Pack Orchestrator
    "ProfessionalPackOrchestrator",
    "PackStatus",
    "WorkflowExecution",
    "WorkflowPhase",
    "WorkflowType",
    "AgentStatus",
    "AgentStatusCode",
    "OrchestratorConfig",
    "ProgressCallback",
    "RetryConfig",
    "WebhookConfig",
    "QualityGateConfig",
    "QualityGateId",
    "QualityGateStatus",
    "QualityGateResult",
    "ApprovalConfig",
    "PhaseDataStore",
    "WebhookEvent",
    "WorkflowCheckpoint",
    # MRV Bridge
    "ProfessionalMRVBridge",
    "ProfessionalMRVBridgeConfig",
    "ScopeType",
    "CalculationStatus",
    "CalculationResult",
    "MRVRoutingEntry",
    "ProvenanceChainEntry",
    "AggregatedEmissions",
    "IntensityMetricType",
    "IntensityMetrics",
    "BiogenicCarbonResult",
    "BaseYearConfig",
    "BaseYearResult",
    "CategoryScreening",
    "ScreeningSignificance",
    "EntityCalculationResult",
    # Cross-Framework Bridge
    "CrossFrameworkBridge",
    "CrossFrameworkBridgeConfig",
    "CrossFrameworkResult",
    "FrameworkId",
    "FrameworkMapping",
    "FrameworkMappingResult",
    "CDPScore",
    "CDPScoringResult",
    "SBTiTemperatureResult",
    "TaxonomyAlignmentResult",
    "ScenarioResult",
    "Gap",
    "GapSeverity",
    "MappingStatus",
    # Webhook Manager
    "WebhookManager",
    "WebhookManagerConfig",
    "WebhookEventType",
    "WebhookChannel",
    "DeliveryStatus",
    "WebhookSubscription",
    "DeliveryResult",
    "DeadLetterEntry",
    "DeliveryStats",
    # Setup Wizard
    "ProfessionalSetupWizard",
    "WizardStepName",
    "StepStatus",
    "WizardStep",
    "WizardState",
    "CompanyProfile",
    "ReportingScope",
    "DataSourceConfig",
    "ProfessionalFeaturesConfig",
    "PresetRecommendation",
    "SetupReport",
    "SectorCode",
    "ConsolidationApproach",
    "AssuranceLevel",
    "SubsidiaryEntity",
    # Health Check
    "ProfessionalHealthCheck",
    "ProfessionalHealthCheckConfig",
    "HealthCheckResult",
    "ComponentHealth",
    "HealthSeverity",
    "HealthStatus",
    "CheckCategory",
    "RemediationSuggestion",
]
