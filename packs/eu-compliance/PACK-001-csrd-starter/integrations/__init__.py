# -*- coding: utf-8 -*-
"""
PACK-001 CSRD Starter Pack - Integration Layer
================================================

Phase 4 integration layer that connects 66+ GreenLang agents into a cohesive
CSRD compliance pipeline. This module provides orchestration, bridging,
setup wizardry, and health verification for production deployments.

Components:
    - CSRDPackOrchestrator: Master pipeline orchestrator for all workflows
    - MRVBridge: Routes ESRS E1 metrics to MRV calculation engines (30 agents)
    - DataPipelineBridge: Routes data sources through intake + quality agents
    - CSRDSetupWizard: Guided interactive setup for new deployments
    - PackHealthCheck: Comprehensive health verification system

Architecture:
    Data Sources --> DataPipelineBridge --> Quality Pipeline --> CSRDPackOrchestrator
                                                                      |
                                                                      v
    ESRS Metrics <-- MRVBridge <-- Calculation Engines <-- Validated Data
                                                                      |
                                                                      v
    Reports <-- Templates <-- Disclosure Engine <-- Compliance Check

Author: GreenLang Team
Version: 1.0.0
"""

from packs.eu_compliance.PACK_001_csrd_starter.integrations.pack_orchestrator import (
    CSRDPackOrchestrator,
    PackStatus,
    WorkflowExecution,
    WorkflowPhase,
    WorkflowType,
    AgentStatus,
    OrchestratorConfig,
    ProgressCallback,
)
from packs.eu_compliance.PACK_001_csrd_starter.integrations.mrv_bridge import (
    MRVBridge,
    MRVBridgeConfig,
    CalculationResult,
    Scope1Result,
    Scope2Result,
    Scope3Result,
    MRVRoutingEntry,
    ProvenanceChainEntry,
    AggregatedEmissions,
)
from packs.eu_compliance.PACK_001_csrd_starter.integrations.data_pipeline_bridge import (
    DataPipelineBridge,
    DataPipelineConfig,
    DataSourceType,
    DataIngestionResult,
    QualityPipelineResult,
    UnifiedESRSDataset,
    DataQualityReport,
    SourceDetectionResult,
)
from packs.eu_compliance.PACK_001_csrd_starter.integrations.setup_wizard import (
    CSRDSetupWizard,
    WizardState,
    WizardStep,
    CompanyProfile,
    ReportingScope,
    DataSourceConfig,
    IntegrationConfig,
    SetupReport,
    PresetRecommendation,
)
from packs.eu_compliance.PACK_001_csrd_starter.integrations.health_check import (
    PackHealthCheck,
    HealthCheckResult,
    ComponentHealth,
    HealthSeverity,
    RemediationSuggestion,
    HealthCheckConfig,
)

__all__ = [
    # Pack Orchestrator
    "CSRDPackOrchestrator",
    "PackStatus",
    "WorkflowExecution",
    "WorkflowPhase",
    "WorkflowType",
    "AgentStatus",
    "OrchestratorConfig",
    "ProgressCallback",
    # MRV Bridge
    "MRVBridge",
    "MRVBridgeConfig",
    "CalculationResult",
    "Scope1Result",
    "Scope2Result",
    "Scope3Result",
    "MRVRoutingEntry",
    "ProvenanceChainEntry",
    "AggregatedEmissions",
    # Data Pipeline Bridge
    "DataPipelineBridge",
    "DataPipelineConfig",
    "DataSourceType",
    "DataIngestionResult",
    "QualityPipelineResult",
    "UnifiedESRSDataset",
    "DataQualityReport",
    "SourceDetectionResult",
    # Setup Wizard
    "CSRDSetupWizard",
    "WizardState",
    "WizardStep",
    "CompanyProfile",
    "ReportingScope",
    "DataSourceConfig",
    "IntegrationConfig",
    "SetupReport",
    "PresetRecommendation",
    # Health Check
    "PackHealthCheck",
    "HealthCheckResult",
    "ComponentHealth",
    "HealthSeverity",
    "RemediationSuggestion",
    "HealthCheckConfig",
]
