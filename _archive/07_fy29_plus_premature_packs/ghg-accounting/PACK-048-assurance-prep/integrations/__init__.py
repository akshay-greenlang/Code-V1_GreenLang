# -*- coding: utf-8 -*-
"""
PACK-048 GHG Assurance Prep Pack - Integration Layer
=============================================================

Phase 4 integration layer for the GHG Assurance Prep Pack that provides
10-phase DAG pipeline orchestration, PACK-041/042/043/044/045/046/047
import bridges, all 30 MRV agent routing (MRV-001 through MRV-030)
for calculation provenance extraction, DATA agent evidence retrieval,
AGENT-FOUND (Assumptions, Citations, Reproducibility) evidence bridges,
multi-channel alerting, 20-category health verification, and 8-step
assurance configuration setup wizard.

Components:
    - PackOrchestrator: 10-phase DAG pipeline with Kahn's topological
      sort, SHA-256 provenance chain, retry with exponential backoff,
      and progress tracking for assurance preparation
    - MRVBridge: Routes to all 30 MRV agents (MRV-001 through MRV-030)
      for per-calculation provenance chain extraction
    - DataBridge: Routes to DATA agents for source data evidence
      retrieval including PDF extraction and quality profiling
    - Pack041Bridge: Imports Scope 1 and Scope 2 emissions data with
      gas breakdown and calculation details for provenance chains
    - Pack042043Bridge: Combined bridge to PACK-042/043 for Scope 3
      evidence packages, leveraging PACK-043 AssuranceEngine directly
    - Pack044Bridge: Bridges to PACK-044 Inventory Management for
      review/approval, documentation, QA/QC, and change management
    - Pack045Bridge: Bridges to PACK-045 Base Year Management for
      base year data, recalculation docs, and significance tests
    - Pack046047Bridge: Combined bridge to PACK-046 Intensity Metrics
      and PACK-047 Benchmark for materiality context data
    - FoundationBridge: Bridges to FOUND-004 Assumptions, FOUND-005
      Citations, and FOUND-008 Reproducibility for core evidence
    - HealthCheck: 20-category system health verification
    - SetupWizard: 8-step assurance configuration wizard
    - AlertBridge: Multi-channel alerting for assurance engagement
      milestones, gaps, verifier queries, findings, and deadlines

Architecture:
    External Systems --> Integration Bridges --> Platform Components
                              |
                              v
    PACK-048 Assurance <-- Composition <-- Zero Duplication
                              |
                              v
    PACK-041..047 <-- MRV Agents (001-030) <-- DATA Agents (001-020)
                              |
                              v
    FOUND Agents (004, 005, 008) <-- Evidence & Provenance

Reference:
    ISAE 3410 Assurance Engagements on GHG Statements
    ISO 14064-3 Specification for Validation and Verification of GHG
    PCAF Global GHG Accounting and Reporting Standard
    GHG Protocol Corporate Standard, Chapter 10: Verification

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-048 GHG Assurance Prep
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack_id__: str = "PACK-048"
__pack_name__: str = "GHG Assurance Prep Pack"
__integrations_count__: int = 12

_loaded_integrations: list[str] = []

# ---------------------------------------------------------------------------
# Integration 1: Pack Orchestrator
# ---------------------------------------------------------------------------
try:
    from .pack_orchestrator import (
        PackOrchestrator,
        PipelinePhase,
        ExecutionStatus,
        PipelineConfig,
        PhaseResult,
        PipelineResult,
        PipelineStatus,
        PHASE_DEPENDENCIES,
        PARALLEL_PHASE_GROUPS,
    )
    _loaded_integrations.append("PackOrchestrator")
except ImportError as e:
    logger.debug("Integration 1 (PackOrchestrator) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 2: MRV Bridge (All 30 Agents)
# ---------------------------------------------------------------------------
try:
    from .mrv_bridge import (
        MRVBridge,
        MRVBridgeConfig,
        MRVScope,
        ProvenanceRequest,
        ProvenanceResponse,
        ScopedProvenance,
        MRVAgentProvenance,
        AGENT_SCOPE_MAP,
        AGENT_DESCRIPTIONS,
    )
    _loaded_integrations.append("MRVBridge")
except ImportError as e:
    logger.debug("Integration 2 (MRVBridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 3: Data Bridge
# ---------------------------------------------------------------------------
try:
    from .data_bridge import (
        DataBridge,
        DataBridgeConfig,
        EvidenceDataType,
        DataAgentTarget,
        EvidenceRequest,
        EvidenceResponse,
        EvidenceDataset,
    )
    _loaded_integrations.append("DataBridge")
except ImportError as e:
    logger.debug("Integration 3 (DataBridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 4: PACK-041 Bridge (Scope 1-2)
# ---------------------------------------------------------------------------
try:
    from .pack041_bridge import (
        Pack041Bridge,
        Pack041Config,
        Pack041Request,
        Pack041Response,
        Scope1Detail,
        Scope2Detail,
    )
    _loaded_integrations.append("Pack041Bridge")
except ImportError as e:
    logger.debug("Integration 4 (Pack041Bridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 5: PACK-042/043 Bridge (Scope 3)
# ---------------------------------------------------------------------------
try:
    from .pack042_043_bridge import (
        Pack042043Bridge,
        Pack042043Config,
        Scope3EvidenceRequest,
        Scope3EvidenceResponse,
        CategoryEvidence,
    )
    _loaded_integrations.append("Pack042043Bridge")
except ImportError as e:
    logger.debug("Integration 5 (Pack042043Bridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 6: PACK-044 Bridge (Inventory Management)
# ---------------------------------------------------------------------------
try:
    from .pack044_bridge import (
        Pack044Bridge,
        Pack044Config,
        InventoryEvidenceRequest,
        InventoryEvidenceResponse,
        ReviewRecord,
        DocumentationRecord,
    )
    _loaded_integrations.append("Pack044Bridge")
except ImportError as e:
    logger.debug("Integration 6 (Pack044Bridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 7: PACK-045 Bridge (Base Year Management)
# ---------------------------------------------------------------------------
try:
    from .pack045_bridge import (
        Pack045Bridge,
        Pack045Config,
        BaseYearEvidenceRequest,
        BaseYearEvidenceResponse,
        RecalculationDocumentation,
        SignificanceTestResult,
    )
    _loaded_integrations.append("Pack045Bridge")
except ImportError as e:
    logger.debug("Integration 7 (Pack045Bridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 8: PACK-046/047 Bridge (Intensity + Benchmark)
# ---------------------------------------------------------------------------
try:
    from .pack046_047_bridge import (
        Pack046047Bridge,
        Pack046047Config,
        IntensityContextRequest,
        IntensityContextResponse,
        BenchmarkContextRequest,
        BenchmarkContextResponse,
    )
    _loaded_integrations.append("Pack046047Bridge")
except ImportError as e:
    logger.debug("Integration 8 (Pack046047Bridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 9: Foundation Bridge (Assumptions, Citations, Reproducibility)
# ---------------------------------------------------------------------------
try:
    from .foundation_bridge import (
        FoundationBridge,
        FoundationBridgeConfig,
        AssumptionRecord,
        CitationRecord,
        ReproducibilityResult,
        FoundationEvidenceResponse,
    )
    _loaded_integrations.append("FoundationBridge")
except ImportError as e:
    logger.debug("Integration 9 (FoundationBridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 10: Health Check
# ---------------------------------------------------------------------------
try:
    from .health_check import (
        HealthCheck,
        HealthCheckConfig,
        HealthCheckCategory,
        HealthStatus,
        HealthSeverity,
        CheckType,
        ComponentHealth,
        SystemHealth,
    )
    _loaded_integrations.append("HealthCheck")
except ImportError as e:
    logger.debug("Integration 10 (HealthCheck) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 11: Setup Wizard
# ---------------------------------------------------------------------------
try:
    from .setup_wizard import (
        SetupWizard,
        SetupStep,
        StepStatus,
        WizardState,
        WizardInput,
        WizardResult,
        PackConfigOutput,
    )
    _loaded_integrations.append("SetupWizard")
except ImportError as e:
    logger.debug("Integration 11 (SetupWizard) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 12: Alert Bridge
# ---------------------------------------------------------------------------
try:
    from .alert_bridge import (
        AlertBridge,
        AlertConfig,
        AlertType,
        AlertSeverity,
        AlertChannel,
        Alert,
        AlertResult,
    )
    _loaded_integrations.append("AlertBridge")
except ImportError as e:
    logger.debug("Integration 12 (AlertBridge) not available: %s", e)


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__: list[str] = [
    # Module metadata
    "__version__",
    "__pack_id__",
    "__pack_name__",
    "__integrations_count__",
    # --- Integration 1: Pack Orchestrator ---
    "PackOrchestrator",
    "PipelinePhase",
    "ExecutionStatus",
    "PipelineConfig",
    "PhaseResult",
    "PipelineResult",
    "PipelineStatus",
    "PHASE_DEPENDENCIES",
    "PARALLEL_PHASE_GROUPS",
    # --- Integration 2: MRV Bridge ---
    "MRVBridge",
    "MRVBridgeConfig",
    "MRVScope",
    "ProvenanceRequest",
    "ProvenanceResponse",
    "ScopedProvenance",
    "MRVAgentProvenance",
    "AGENT_SCOPE_MAP",
    "AGENT_DESCRIPTIONS",
    # --- Integration 3: Data Bridge ---
    "DataBridge",
    "DataBridgeConfig",
    "EvidenceDataType",
    "DataAgentTarget",
    "EvidenceRequest",
    "EvidenceResponse",
    "EvidenceDataset",
    # --- Integration 4: PACK-041 Bridge ---
    "Pack041Bridge",
    "Pack041Config",
    "Pack041Request",
    "Pack041Response",
    "Scope1Detail",
    "Scope2Detail",
    # --- Integration 5: PACK-042/043 Bridge ---
    "Pack042043Bridge",
    "Pack042043Config",
    "Scope3EvidenceRequest",
    "Scope3EvidenceResponse",
    "CategoryEvidence",
    # --- Integration 6: PACK-044 Bridge ---
    "Pack044Bridge",
    "Pack044Config",
    "InventoryEvidenceRequest",
    "InventoryEvidenceResponse",
    "ReviewRecord",
    "DocumentationRecord",
    # --- Integration 7: PACK-045 Bridge ---
    "Pack045Bridge",
    "Pack045Config",
    "BaseYearEvidenceRequest",
    "BaseYearEvidenceResponse",
    "RecalculationDocumentation",
    "SignificanceTestResult",
    # --- Integration 8: PACK-046/047 Bridge ---
    "Pack046047Bridge",
    "Pack046047Config",
    "IntensityContextRequest",
    "IntensityContextResponse",
    "BenchmarkContextRequest",
    "BenchmarkContextResponse",
    # --- Integration 9: Foundation Bridge ---
    "FoundationBridge",
    "FoundationBridgeConfig",
    "AssumptionRecord",
    "CitationRecord",
    "ReproducibilityResult",
    "FoundationEvidenceResponse",
    # --- Integration 10: Health Check ---
    "HealthCheck",
    "HealthCheckConfig",
    "HealthCheckCategory",
    "HealthStatus",
    "HealthSeverity",
    "CheckType",
    "ComponentHealth",
    "SystemHealth",
    # --- Integration 11: Setup Wizard ---
    "SetupWizard",
    "SetupStep",
    "StepStatus",
    "WizardState",
    "WizardInput",
    "WizardResult",
    "PackConfigOutput",
    # --- Integration 12: Alert Bridge ---
    "AlertBridge",
    "AlertConfig",
    "AlertType",
    "AlertSeverity",
    "AlertChannel",
    "Alert",
    "AlertResult",
]


def get_loaded_integrations() -> list[str]:
    """Return list of integration class names that loaded successfully."""
    return list(_loaded_integrations)


logger.info(
    "PACK-048 GHG Assurance Prep integrations: %d/%d loaded",
    len(_loaded_integrations),
    __integrations_count__,
)
