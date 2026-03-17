# -*- coding: utf-8 -*-
"""
PACK-015 Double Materiality Assessment Pack - Integration Layer
================================================================

Phase 4 integration layer for the Double Materiality Assessment Pack that
provides 9-phase DMA pipeline orchestration, cross-pack CSRD bridging with
bidirectional data flow, MRV agent emissions context routing, DATA agent
intake and quality management, NACE sector classification with conglomerate
support, regulatory change monitoring, 12-category health verification,
and 6-step guided DMA setup wizard.

Components:
    - DMAPackOrchestrator: 9-phase DMA pipeline with DAG dependency
      resolution, parallel impact/financial assessment, retry with
      exponential backoff, and SHA-256 provenance tracking
    - CSRDPackBridge: Bridge to PACK-001/002/003 CSRD base packs for
      governance data import (GOV-1 through GOV-5), general disclosure
      import, and DMA result feed into CSRD reporting pipeline
    - MRVMaterialityBridge: Routes emissions data from 30 MRV agents
      to provide E1 climate materiality context and hotspot mapping
    - DataMaterialityBridge: Routes external data to DATA agents for
      stakeholder surveys, financial impact data, and quality profiling
    - SectorClassificationBridge: NACE sector mapping with 12 sector
      profiles, conglomerate support, and benchmark data
    - RegulatoryBridge: Monitors ESRS updates, CSRD Omnibus changes,
      EFRAG guidance, and generates alerts and threshold updates
    - DMAHealthCheck: 12-category system health verification
    - DMASetupWizard: 6-step DMA configuration with industry defaults

Architecture:
    External Systems --> Integration Bridges --> Platform Components
                              |
                              v
    PACK-015 DMA Pack <-- Composition <-- Zero Duplication
                              |
                              v
    MRV Agents <-- DATA Agents <-- FOUND Agents <-- CSRD Packs

Platform Integrations:
    - greenlang/agents/mrv/* (30 MRV agents)
    - greenlang/agents/data/* (20 DATA agents)
    - greenlang/agents/foundation/* (10 FOUND agents)
    - packs/eu-compliance/PACK-001 through PACK-003 (CSRD packs)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-015 Double Materiality Assessment
Status: Production Ready
"""

__version__ = "1.0.0"
__pack_id__ = "PACK-015"
__pack_name__ = "Double Materiality Assessment Pack"

# ---------------------------------------------------------------------------
# DMA Pipeline Orchestrator
# ---------------------------------------------------------------------------
try:
    from packs.eu_compliance.PACK_015_double_materiality.integrations.pack_orchestrator import (
        DMAPackOrchestrator,
        DMAPipelinePhase,
        ExecutionStatus,
        OrchestratorConfig,
        PARALLEL_PHASE_GROUPS,
        PHASE_DEPENDENCIES,
        PHASE_EXECUTION_ORDER,
        PhaseProvenance,
        PhaseResult,
        PipelineResult,
        RetryConfig,
        ScoringMethodology,
    )
except ImportError:
    DMAPackOrchestrator = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# CSRD Pack Bridge
# ---------------------------------------------------------------------------
try:
    from packs.eu_compliance.PACK_015_double_materiality.integrations.csrd_pack_bridge import (
        BasePack,
        BridgeResult,
        CSRDPackBridge,
        CSRDPackBridgeConfig,
        CompanyProfileImport,
        DataFlowDirection,
        GovernanceData,
        GovernanceDisclosure,
        MaterialTopicFeed,
    )
except ImportError:
    CSRDPackBridge = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# MRV Materiality Bridge
# ---------------------------------------------------------------------------
try:
    from packs.eu_compliance.PACK_015_double_materiality.integrations.mrv_materiality_bridge import (
        ESRSEnvironmentalTopic,
        EmissionsContext,
        HotspotMapping,
        MRVAgentMapping,
        MRVBridgeConfig,
        MRVMaterialityBridge,
        MRVQueryResult,
        MRVScope,
    )
except ImportError:
    MRVMaterialityBridge = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Data Materiality Bridge
# ---------------------------------------------------------------------------
try:
    from packs.eu_compliance.PACK_015_double_materiality.integrations.data_materiality_bridge import (
        DMADataSource,
        DataAgentRoute,
        DataBridgeConfig,
        DataMaterialityBridge,
        DataQualityLevel,
        DataQualityReport,
        DataRoutingResult,
        StakeholderSurveyResult,
    )
except ImportError:
    DataMaterialityBridge = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Sector Classification Bridge
# ---------------------------------------------------------------------------
try:
    from packs.eu_compliance.PACK_015_double_materiality.integrations.sector_classification_bridge import (
        NACECode,
        SectorBenchmark,
        SectorBridgeConfig,
        SectorClassificationBridge,
        SectorClassificationResult,
        SectorProfile,
        TopicPriority,
    )
except ImportError:
    SectorClassificationBridge = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Regulatory Bridge
# ---------------------------------------------------------------------------
try:
    from packs.eu_compliance.PACK_015_double_materiality.integrations.regulatory_bridge import (
        ChangeSeverity,
        ChangeStatus,
        ChangeType,
        RegulatoryAlert,
        RegulatoryBridge,
        RegulatoryBridgeConfig,
        RegulatoryChange,
        RegulatoryCheckResult,
        RegulatorySource,
        ThresholdUpdate,
    )
except ImportError:
    RegulatoryBridge = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# DMA Health Check
# ---------------------------------------------------------------------------
try:
    from packs.eu_compliance.PACK_015_double_materiality.integrations.health_check import (
        CheckCategory,
        ComponentHealth,
        DMAHealthCheck,
        HealthCheckConfig,
        HealthCheckResult,
        HealthSeverity,
        HealthStatus,
        RemediationSuggestion,
    )
except ImportError:
    DMAHealthCheck = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# DMA Setup Wizard
# ---------------------------------------------------------------------------
try:
    from packs.eu_compliance.PACK_015_double_materiality.integrations.setup_wizard import (
        CompanyProfile,
        DMASetupWizard,
        DMAWizardStep,
        ReportingPreferences,
        ScopeDefinition,
        ScoringMethod,
        ScoringMethodologyConfig,
        SetupResult,
        StakeholderConfig,
        StepStatus,
        ThresholdConfig,
        ValueChainBoundary,
        WizardState,
        WizardStepState,
    )
except ImportError:
    DMASetupWizard = None  # type: ignore[assignment,misc]

__all__ = [
    # Module metadata
    "__version__",
    "__pack_id__",
    "__pack_name__",
    # --- DMA Pipeline Orchestrator ---
    "DMAPackOrchestrator",
    "OrchestratorConfig",
    "RetryConfig",
    "DMAPipelinePhase",
    "ScoringMethodology",
    "ExecutionStatus",
    "PhaseProvenance",
    "PhaseResult",
    "PipelineResult",
    "PHASE_DEPENDENCIES",
    "PHASE_EXECUTION_ORDER",
    "PARALLEL_PHASE_GROUPS",
    # --- CSRD Pack Bridge ---
    "CSRDPackBridge",
    "CSRDPackBridgeConfig",
    "BasePack",
    "GovernanceDisclosure",
    "DataFlowDirection",
    "GovernanceData",
    "MaterialTopicFeed",
    "BridgeResult",
    "CompanyProfileImport",
    # --- MRV Materiality Bridge ---
    "MRVMaterialityBridge",
    "MRVBridgeConfig",
    "MRVAgentMapping",
    "MRVScope",
    "ESRSEnvironmentalTopic",
    "EmissionsContext",
    "HotspotMapping",
    "MRVQueryResult",
    # --- Data Materiality Bridge ---
    "DataMaterialityBridge",
    "DataBridgeConfig",
    "DMADataSource",
    "DataQualityLevel",
    "DataAgentRoute",
    "DataRoutingResult",
    "StakeholderSurveyResult",
    "DataQualityReport",
    # --- Sector Classification Bridge ---
    "SectorClassificationBridge",
    "SectorBridgeConfig",
    "NACECode",
    "TopicPriority",
    "SectorProfile",
    "SectorClassificationResult",
    "SectorBenchmark",
    # --- Regulatory Bridge ---
    "RegulatoryBridge",
    "RegulatoryBridgeConfig",
    "RegulatorySource",
    "ChangeType",
    "ChangeSeverity",
    "ChangeStatus",
    "RegulatoryChange",
    "RegulatoryAlert",
    "ThresholdUpdate",
    "RegulatoryCheckResult",
    # --- DMA Health Check ---
    "DMAHealthCheck",
    "HealthCheckConfig",
    "HealthCheckResult",
    "ComponentHealth",
    "HealthSeverity",
    "HealthStatus",
    "CheckCategory",
    "RemediationSuggestion",
    # --- DMA Setup Wizard ---
    "DMASetupWizard",
    "DMAWizardStep",
    "StepStatus",
    "ScoringMethod",
    "ValueChainBoundary",
    "CompanyProfile",
    "ScopeDefinition",
    "StakeholderConfig",
    "ScoringMethodologyConfig",
    "ThresholdConfig",
    "ReportingPreferences",
    "WizardStepState",
    "WizardState",
    "SetupResult",
]
