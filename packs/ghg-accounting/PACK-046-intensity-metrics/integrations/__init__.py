# -*- coding: utf-8 -*-
"""
PACK-046 Intensity Metrics Pack - Integration Layer
=======================================================

Phase 4 integration layer for the Intensity Metrics Pack that provides
10-phase DAG pipeline orchestration, PACK-041/042/043/044/045 import
bridges, all 30 MRV agent routing (MRV-001 through MRV-030), DATA agent
denominator data ingestion bridges, external benchmark data retrieval,
SBTi intensity pathway tracking, multi-channel alerting, 20-category
health verification, and 8-step setup wizard.

Components:
    - PackOrchestrator: 10-phase DAG pipeline with Kahn's topological
      sort, SHA-256 provenance chain, retry with exponential backoff,
      and progress tracking for intensity metrics calculation
    - MRVBridge: Routes to all 30 MRV agents (MRV-001 through MRV-030)
      for emissions numerator data retrieval by scope
    - DataBridge: Routes denominator data intake to DATA agents for
      extraction, normalization, quality profiling, and reconciliation
    - Pack041Bridge: Imports Scope 1 and Scope 2 emission totals from
      PACK-041 (Scope 1-2 Complete Pack) with dual Scope 2 reporting
    - Pack042043Bridge: Combined bridge to PACK-042 (Scope 3 Starter)
      or PACK-043 (Scope 3 Complete) for Scope 3 category totals
    - Pack044Bridge: Bridges to PACK-044 Inventory Management for
      inventory period definitions and data collection status
    - Pack045Bridge: Bridges to PACK-045 Base Year Management for
      base year emissions, denominators, and adjusted time series
    - BenchmarkDataBridge: External benchmark data integration from
      CDP, TPI, GRESB, and CRREM for peer comparison
    - SBTiPathwayBridge: SBTi SDA intensity pathway data for sector
      convergence targets and ambition-level tracking
    - HealthCheck: 20-category system health verification
    - SetupWizard: 8-step intensity metrics configuration wizard
    - AlertBridge: Multi-channel alerting for threshold breaches,
      target tracking, and disclosure deadlines

Architecture:
    External Systems --> Integration Bridges --> Platform Components
                              |
                              v
    PACK-046 Intensity Metrics <-- Composition <-- Zero Duplication
                              |
                              v
    MRV Agents (001-030) <-- DATA Agents (001-020) <-- FOUND Agents (001-010)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-046 Intensity Metrics
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack_id__: str = "PACK-046"
__pack_name__: str = "Intensity Metrics Pack"
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
        EmissionsRequest,
        EmissionsResponse,
        ScopedEmissions,
        MRVAgentResult,
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
        DenominatorType,
        DataAgentTarget,
        DataRequest,
        DataResponse,
        DenominatorDataset,
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
        Scope1Summary,
        Scope2Summary,
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
        Scope3Request,
        Scope3Response,
        CategoryCoverage,
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
        InventoryRequest,
        InventoryResponse,
        InventoryPeriod,
        CollectionStatus,
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
        BaseYearRequest,
        BaseYearResponse,
        BaseYearEmissions,
        BaseYearDenominators,
        RecalculationFlag,
    )
    _loaded_integrations.append("Pack045Bridge")
except ImportError as e:
    logger.debug("Integration 7 (Pack045Bridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 8: Benchmark Data Bridge
# ---------------------------------------------------------------------------
try:
    from .benchmark_data_bridge import (
        BenchmarkDataBridge,
        BenchmarkConfig,
        BenchmarkRequest,
        BenchmarkDataResponse,
        SectorBenchmark,
        BenchmarkSource,
    )
    _loaded_integrations.append("BenchmarkDataBridge")
except ImportError as e:
    logger.debug("Integration 8 (BenchmarkDataBridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 9: SBTi Pathway Bridge
# ---------------------------------------------------------------------------
try:
    from .sbti_pathway_bridge import (
        SBTiPathwayBridge,
        SBTiConfig,
        PathwayRequest,
        PathwayResponse,
        SectorPathway,
        AmbitionLevel,
        PATHWAY_DATA,
    )
    _loaded_integrations.append("SBTiPathwayBridge")
except ImportError as e:
    logger.debug("Integration 9 (SBTiPathwayBridge) not available: %s", e)

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
    "EmissionsRequest",
    "EmissionsResponse",
    "ScopedEmissions",
    "MRVAgentResult",
    "AGENT_SCOPE_MAP",
    "AGENT_DESCRIPTIONS",
    # --- Integration 3: Data Bridge ---
    "DataBridge",
    "DataBridgeConfig",
    "DenominatorType",
    "DataAgentTarget",
    "DataRequest",
    "DataResponse",
    "DenominatorDataset",
    # --- Integration 4: PACK-041 Bridge ---
    "Pack041Bridge",
    "Pack041Config",
    "Pack041Request",
    "Pack041Response",
    "Scope1Summary",
    "Scope2Summary",
    # --- Integration 5: PACK-042/043 Bridge ---
    "Pack042043Bridge",
    "Pack042043Config",
    "Scope3Request",
    "Scope3Response",
    "CategoryCoverage",
    # --- Integration 6: PACK-044 Bridge ---
    "Pack044Bridge",
    "Pack044Config",
    "InventoryRequest",
    "InventoryResponse",
    "InventoryPeriod",
    "CollectionStatus",
    # --- Integration 7: PACK-045 Bridge ---
    "Pack045Bridge",
    "Pack045Config",
    "BaseYearRequest",
    "BaseYearResponse",
    "BaseYearEmissions",
    "BaseYearDenominators",
    "RecalculationFlag",
    # --- Integration 8: Benchmark Data Bridge ---
    "BenchmarkDataBridge",
    "BenchmarkConfig",
    "BenchmarkRequest",
    "BenchmarkDataResponse",
    "SectorBenchmark",
    "BenchmarkSource",
    # --- Integration 9: SBTi Pathway Bridge ---
    "SBTiPathwayBridge",
    "SBTiConfig",
    "PathwayRequest",
    "PathwayResponse",
    "SectorPathway",
    "AmbitionLevel",
    "PATHWAY_DATA",
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
    "PACK-046 Intensity Metrics integrations: %d/%d loaded",
    len(_loaded_integrations),
    __integrations_count__,
)
