# -*- coding: utf-8 -*-
"""
PACK-049 GHG Multi-Site Management Pack - Integration Layer
=============================================================

Phase 4 integration layer for the GHG Multi-Site Management Pack that
provides 10-phase DAG pipeline orchestration, all 30 MRV agent routing
per site (MRV-001 through MRV-030) for site-level emission calculations,
DATA agent collection and validation, PACK-041 through PACK-047 import
bridges, AGENT-FOUND (Units, Assumptions, Citations) normalisation bridges,
multi-channel alerting, 20-category health verification, and 8-step
multi-site configuration setup wizard.

Components:
    - PackOrchestrator: 10-phase DAG pipeline with Kahn's topological
      sort, SHA-256 provenance chain, retry with exponential backoff,
      and progress tracking for multi-site consolidation
    - MRVBridge: Routes to all 30 MRV agents (MRV-001 through MRV-030)
      for per-site emission calculation across all scopes
    - DataBridge: Routes to DATA agents for site data ingestion,
      validation, quality profiling, and cross-source reconciliation
    - Pack041Bridge: Imports Scope 1 and Scope 2 facility data including
      organisational boundary and emission factor assignments
    - Pack042043Bridge: Combined bridge to PACK-042/043 for Scope 3
      multi-entity boundary and intercompany elimination data
    - Pack044Bridge: Bridges to PACK-044 Inventory Management for
      consolidation runs, submission tracking, and review status
    - Pack045Bridge: Bridges to PACK-045 Base Year Management for
      per-site base year data, recalculation triggers, and adjustments
    - Pack046047Bridge: Combined bridge to PACK-046 Intensity Metrics
      and PACK-047 Benchmark for site-level KPIs and peer rankings
    - FoundationBridge: Bridges to FOUND-003 Unit Normaliser, FOUND-004
      Assumptions Registry, and FOUND-005 Citations for data integrity
    - HealthCheck: 20-category system health verification
    - SetupWizard: 8-step multi-site configuration wizard
    - AlertBridge: Multi-channel alerting for submission deadlines,
      overdue sites, quality drops, boundary changes, allocation
      variance, and completeness thresholds

Architecture:
    External Systems --> Integration Bridges --> Platform Components
                              |
                              v
    PACK-049 Multi-Site <-- Composition <-- Zero Duplication
                              |
                              v
    PACK-041..047 <-- MRV Agents (001-030) <-- DATA Agents (001-020)
                              |
                              v
    FOUND Agents (003, 004, 005) <-- Units, Assumptions, Citations

Reference:
    GHG Protocol Corporate Standard, Chapter 3: Setting Organisational
      Boundaries
    GHG Protocol Corporate Standard, Chapter 8: Reporting
    ISO 14064-1:2018 Clause 5.2: Organisational boundaries
    WRI/WBCSD GHG Protocol FAQ: Multi-site consolidation

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-049 GHG Multi-Site Management
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack_id__: str = "PACK-049"
__pack_name__: str = "GHG Multi-Site Management Pack"
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
# Integration 2: MRV Bridge (All 30 Agents per Site)
# ---------------------------------------------------------------------------
try:
    from .mrv_bridge import (
        MRVBridge,
        MRVBridgeConfig,
        MRVScope,
        MRVSiteEmissions,
        MRVScopeBreakdown,
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
        DataIngestionResult,
        DataQualityProfile,
    )
    _loaded_integrations.append("DataBridge")
except ImportError as e:
    logger.debug("Integration 3 (DataBridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 4: PACK-041 Bridge (Scope 1-2 Facilities)
# ---------------------------------------------------------------------------
try:
    from .pack041_bridge import (
        Pack041Bridge,
        Pack041Config,
        Pack041Facility,
        Pack041Boundary,
        Pack041EmissionFactors,
    )
    _loaded_integrations.append("Pack041Bridge")
except ImportError as e:
    logger.debug("Integration 4 (Pack041Bridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 5: PACK-042/043 Bridge (Scope 3 Multi-Entity)
# ---------------------------------------------------------------------------
try:
    from .pack042_043_bridge import (
        Pack042043Bridge,
        Pack042043Config,
        Pack043EntityBoundary,
        Pack043Scope3Totals,
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
        Pack044ConsolidationRun,
        Pack044SubmissionStatus,
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
        Pack045BaseYear,
        Pack045Adjustment,
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
        Pack046IntensityMetrics,
        Pack047BenchmarkPosition,
    )
    _loaded_integrations.append("Pack046047Bridge")
except ImportError as e:
    logger.debug("Integration 8 (Pack046047Bridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 9: Foundation Bridge (Units, Assumptions, Citations)
# ---------------------------------------------------------------------------
try:
    from .foundation_bridge import (
        FoundationBridge,
        FoundationBridgeConfig,
        NormalisationResult,
        AssumptionRecord,
        CitationRecord,
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
    "MRVSiteEmissions",
    "MRVScopeBreakdown",
    "AGENT_SCOPE_MAP",
    "AGENT_DESCRIPTIONS",
    # --- Integration 3: Data Bridge ---
    "DataBridge",
    "DataBridgeConfig",
    "DataIngestionResult",
    "DataQualityProfile",
    # --- Integration 4: PACK-041 Bridge ---
    "Pack041Bridge",
    "Pack041Config",
    "Pack041Facility",
    "Pack041Boundary",
    "Pack041EmissionFactors",
    # --- Integration 5: PACK-042/043 Bridge ---
    "Pack042043Bridge",
    "Pack042043Config",
    "Pack043EntityBoundary",
    "Pack043Scope3Totals",
    # --- Integration 6: PACK-044 Bridge ---
    "Pack044Bridge",
    "Pack044Config",
    "Pack044ConsolidationRun",
    "Pack044SubmissionStatus",
    # --- Integration 7: PACK-045 Bridge ---
    "Pack045Bridge",
    "Pack045Config",
    "Pack045BaseYear",
    "Pack045Adjustment",
    # --- Integration 8: PACK-046/047 Bridge ---
    "Pack046047Bridge",
    "Pack046047Config",
    "Pack046IntensityMetrics",
    "Pack047BenchmarkPosition",
    # --- Integration 9: Foundation Bridge ---
    "FoundationBridge",
    "FoundationBridgeConfig",
    "NormalisationResult",
    "AssumptionRecord",
    "CitationRecord",
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
    "PACK-049 GHG Multi-Site Management integrations: %d/%d loaded",
    len(_loaded_integrations),
    __integrations_count__,
)
