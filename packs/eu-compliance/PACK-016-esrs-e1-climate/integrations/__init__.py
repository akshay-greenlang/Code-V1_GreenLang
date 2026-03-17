# -*- coding: utf-8 -*-
"""
PACK-016 ESRS E1 Climate Change Pack - Integration Layer
==========================================================

Phase 4 integration layer for the ESRS E1 Climate Pack that provides
10-phase E1 pipeline orchestration, GL-GHG-APP bridging with bidirectional
data flow, MRV agent emissions routing (30 agents), PACK-015 DMA
materiality bridging, decarbonization planning import, climate adaptation
risk assessment, 12-category health verification, and 6-step guided
E1 setup wizard.

Components:
    - E1PackOrchestrator: 10-phase E1 pipeline with DAG dependency
      resolution, parallel carbon assessment, retry with exponential
      backoff, and SHA-256 provenance tracking
    - GHGAppBridge: Bridge to GL-GHG-APP for inventory import, base
      year sync, target import, and E1 results export
    - MRVAgentBridge: Routes emissions from 30 MRV agents to provide
      Scope 1/2/3 data for E1-6 disclosure
    - DMAPackBridge: Bridge to PACK-015 for E1 materiality import,
      IRO register, and climate disclosure export
    - DecarbonizationBridge: Import transition plan, abatement options,
      and pathway scenarios from decarb agents
    - AdaptationBridge: Import physical risks, climate scenarios, and
      resilience scores from adaptation agents
    - E1HealthCheck: 12-category system health verification
    - E1SetupWizard: 6-step E1 configuration wizard

Architecture:
    External Systems --> Integration Bridges --> Platform Components
                              |
                              v
    PACK-016 E1 Pack <-- Composition <-- Zero Duplication
                              |
                              v
    MRV Agents <-- GHG-APP <-- DMA Pack <-- Decarb/Adapt Agents

Platform Integrations:
    - greenlang/agents/mrv/* (30 MRV agents)
    - greenlang/apps/GL-GHG-APP (GHG inventory app)
    - packs/eu-compliance/PACK-015 (Double Materiality)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-016 ESRS E1 Climate Pack
Status: Production Ready
"""

__version__ = "1.0.0"
__pack_id__ = "PACK-016"
__pack_name__ = "ESRS E1 Climate Change Pack"

# ---------------------------------------------------------------------------
# E1 Pipeline Orchestrator
# ---------------------------------------------------------------------------
try:
    from packs.eu_compliance.PACK_016_esrs_e1_climate.integrations.pack_orchestrator import (
        E1PackOrchestrator,
        E1PipelinePhase,
        ExecutionStatus,
        OrchestratorConfig,
        PARALLEL_PHASE_GROUPS,
        PHASE_DEPENDENCIES,
        PHASE_EXECUTION_ORDER,
        PhaseProvenance,
        PhaseResult,
        PipelineResult,
        RetryConfig,
    )
except ImportError:
    E1PackOrchestrator = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# GHG App Bridge
# ---------------------------------------------------------------------------
try:
    from packs.eu_compliance.PACK_016_esrs_e1_climate.integrations.ghg_app_bridge import (
        BaseYearData,
        GHGAppBridge,
        GHGBridgeConfig,
        InventoryImport,
        SyncDirection,
        TargetImport,
    )
except ImportError:
    GHGAppBridge = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# MRV Agent Bridge
# ---------------------------------------------------------------------------
try:
    from packs.eu_compliance.PACK_016_esrs_e1_climate.integrations.mrv_agent_bridge import (
        AggregationResult,
        AgentStatus,
        MRVAgentBridge,
        MRVAgentMapping,
        MRVBridgeConfig,
        MRVScope,
        SCOPE1_AGENTS,
        SCOPE2_AGENTS,
        SCOPE3_AGENTS,
        ScopeImportResult,
    )
except ImportError:
    MRVAgentBridge = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# DMA Pack Bridge
# ---------------------------------------------------------------------------
try:
    from packs.eu_compliance.PACK_016_esrs_e1_climate.integrations.dma_pack_bridge import (
        DMABridgeConfig,
        DMAPackBridge,
        E1MaterialityResult,
        IROEntry,
        IROType,
        MaterialityStatus,
    )
except ImportError:
    DMAPackBridge = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Decarbonization Bridge
# ---------------------------------------------------------------------------
try:
    from packs.eu_compliance.PACK_016_esrs_e1_climate.integrations.decarbonization_bridge import (
        AbatementOption,
        DecarbBridgeConfig,
        DecarbonizationBridge,
        PathwayScenario,
    )
except ImportError:
    DecarbonizationBridge = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Adaptation Bridge
# ---------------------------------------------------------------------------
try:
    from packs.eu_compliance.PACK_016_esrs_e1_climate.integrations.adaptation_bridge import (
        AdaptationBridge,
        AdaptBridgeConfig,
        ClimateScenario,
        HazardCategory,
        PhysicalRisk,
        ResilienceScore,
        ScenarioFramework,
    )
except ImportError:
    AdaptationBridge = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# E1 Health Check
# ---------------------------------------------------------------------------
try:
    from packs.eu_compliance.PACK_016_esrs_e1_climate.integrations.health_check import (
        CheckCategory,
        ComponentHealth,
        E1HealthCheck,
        HealthCheckConfig,
        HealthCheckResult,
        HealthSeverity,
        HealthStatus,
        RemediationSuggestion,
    )
except ImportError:
    E1HealthCheck = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# E1 Setup Wizard
# ---------------------------------------------------------------------------
try:
    from packs.eu_compliance.PACK_016_esrs_e1_climate.integrations.setup_wizard import (
        CarbonPricingConfig,
        CompanyProfile,
        ConsolidationApproach,
        E1SetupWizard,
        E1WizardStep,
        EnergyScope,
        GHGScope,
        ReportingConfig,
        SetupResult,
        StepStatus,
        TargetConfig,
        WizardState,
        WizardStepState,
    )
except ImportError:
    E1SetupWizard = None  # type: ignore[assignment,misc]

__all__ = [
    # Module metadata
    "__version__",
    "__pack_id__",
    "__pack_name__",
    # --- E1 Pipeline Orchestrator ---
    "E1PackOrchestrator",
    "OrchestratorConfig",
    "RetryConfig",
    "E1PipelinePhase",
    "ExecutionStatus",
    "PhaseProvenance",
    "PhaseResult",
    "PipelineResult",
    "PHASE_DEPENDENCIES",
    "PHASE_EXECUTION_ORDER",
    "PARALLEL_PHASE_GROUPS",
    # --- GHG App Bridge ---
    "GHGAppBridge",
    "GHGBridgeConfig",
    "InventoryImport",
    "BaseYearData",
    "TargetImport",
    "SyncDirection",
    # --- MRV Agent Bridge ---
    "MRVAgentBridge",
    "MRVBridgeConfig",
    "MRVAgentMapping",
    "MRVScope",
    "AgentStatus",
    "ScopeImportResult",
    "AggregationResult",
    "SCOPE1_AGENTS",
    "SCOPE2_AGENTS",
    "SCOPE3_AGENTS",
    # --- DMA Pack Bridge ---
    "DMAPackBridge",
    "DMABridgeConfig",
    "E1MaterialityResult",
    "IROEntry",
    "IROType",
    "MaterialityStatus",
    # --- Decarbonization Bridge ---
    "DecarbonizationBridge",
    "DecarbBridgeConfig",
    "AbatementOption",
    "PathwayScenario",
    # --- Adaptation Bridge ---
    "AdaptationBridge",
    "AdaptBridgeConfig",
    "PhysicalRisk",
    "ClimateScenario",
    "ResilienceScore",
    "HazardCategory",
    "ScenarioFramework",
    # --- E1 Health Check ---
    "E1HealthCheck",
    "HealthCheckConfig",
    "HealthCheckResult",
    "ComponentHealth",
    "HealthSeverity",
    "HealthStatus",
    "CheckCategory",
    "RemediationSuggestion",
    # --- E1 Setup Wizard ---
    "E1SetupWizard",
    "E1WizardStep",
    "StepStatus",
    "ConsolidationApproach",
    "CompanyProfile",
    "GHGScope",
    "EnergyScope",
    "TargetConfig",
    "CarbonPricingConfig",
    "ReportingConfig",
    "WizardStepState",
    "WizardState",
    "SetupResult",
]
