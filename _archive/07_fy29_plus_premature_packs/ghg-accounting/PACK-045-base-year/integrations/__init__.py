# -*- coding: utf-8 -*-
"""
PACK-045 Base Year Management Pack - Integration Layer
=========================================================

Phase 4 integration layer for the Base Year Management Pack that
provides 10-phase DAG pipeline orchestration, PACK-041/042/043/044
import bridges, all 30 MRV agent routing (MRV-001 through MRV-030),
DATA agent data ingestion bridges, Foundation agent routing, ERP
system connectivity, multi-channel notification, 20-category health
verification, and 8-step setup wizard.

Components:
    - BaseYearManagementOrchestrator: 10-phase DAG pipeline with
      Kahn's topological sort, SHA-256 provenance chain, retry with
      exponential backoff, and progress tracking
    - Pack041Bridge: Imports Scope 1-2 data from PACK-041 (Scope 1-2
      Complete Pack) including all Scope 1 categories and dual-reporting
      Scope 2
    - Pack042Bridge: Imports Scope 3 starter data from PACK-042 (Scope 3
      Starter Pack) covering 8 screening categories
    - Pack043Bridge: Imports full Scope 3 data from PACK-043 (Scope 3
      Complete Pack) covering all 15 categories with SBTi alignment
    - Pack044Bridge: Bridges to PACK-044 Inventory Management for change
      management triggers and version tracking
    - MRVBridge: Routes to all 30 MRV agents (MRV-001 through MRV-030)
      with scope-based grouping and emission factor lookups
    - DataBridge: Routes data intake to DATA agents for extraction,
      normalization, quality profiling, and lineage tracking
    - FoundationBridge: Routes to Foundation agents for schema validation,
      unit normalization, assumption registration, and citations
    - ERPConnector: Connects to SAP/Oracle/Dynamics for historical
      activity data extraction
    - NotificationBridge: Multi-channel notification delivery (email,
      Slack, Teams, webhook) for triggers, approvals, and reviews
    - HealthCheck: 20-category system health verification
    - SetupWizard: 8-step base year configuration wizard

Architecture:
    External Systems --> Integration Bridges --> Platform Components
                              |
                              v
    PACK-045 Base Year Mgmt <-- Composition <-- Zero Duplication
                              |
                              v
    MRV Agents (001-030) <-- DATA Agents (001-020) <-- FOUND Agents (001-010)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-045 Base Year Management
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack_id__: str = "PACK-045"
__pack_name__: str = "Base Year Management Pack"
__integrations_count__: int = 12

_loaded_integrations: list[str] = []

# ---------------------------------------------------------------------------
# Integration 1: Pack Orchestrator
# ---------------------------------------------------------------------------
try:
    from .pack_orchestrator import (
        BaseYearManagementOrchestrator,
        PipelinePhase,
        ExecutionStatus,
        PipelineConfig,
        PhaseResult,
        PipelineResult,
        PipelineStatus,
        PHASE_DEPENDENCIES,
        PARALLEL_PHASE_GROUPS,
    )
    _loaded_integrations.append("BaseYearManagementOrchestrator")
except ImportError as e:
    logger.debug("Integration 1 (BaseYearManagementOrchestrator) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 2: PACK-041 Bridge (Scope 1-2)
# ---------------------------------------------------------------------------
try:
    from .pack041_bridge import (
        Pack041Bridge,
        Pack041Config,
        Scope1Summary,
        Scope2Summary,
        ImportResult,
    )
    _loaded_integrations.append("Pack041Bridge")
except ImportError as e:
    logger.debug("Integration 2 (Pack041Bridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 3: PACK-042 Bridge (Scope 3 Starter)
# ---------------------------------------------------------------------------
try:
    from .pack042_bridge import (
        Pack042Bridge,
        Pack042Config,
        Scope3CategoryResult,
        Scope3ImportResult,
    )
    _loaded_integrations.append("Pack042Bridge")
except ImportError as e:
    logger.debug("Integration 3 (Pack042Bridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 4: PACK-043 Bridge (Scope 3 Complete)
# ---------------------------------------------------------------------------
try:
    from .pack043_bridge import (
        Pack043Bridge,
        Pack043Config,
        Scope3FullResult,
        Scope3CompleteImportResult,
    )
    _loaded_integrations.append("Pack043Bridge")
except ImportError as e:
    logger.debug("Integration 4 (Pack043Bridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 5: PACK-044 Bridge (Inventory Management)
# ---------------------------------------------------------------------------
try:
    from .pack044_bridge import (
        Pack044Bridge,
        Pack044Config,
        ChangeType,
        ChangeEvent,
        VersionInfo,
        Pack044ImportResult,
    )
    _loaded_integrations.append("Pack044Bridge")
except ImportError as e:
    logger.debug("Integration 5 (Pack044Bridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 6: MRV Bridge (All 30 Agents)
# ---------------------------------------------------------------------------
try:
    from .mrv_bridge import (
        MRVBridge,
        MRVBridgeConfig,
        MRVScope,
        MRVAgentResult,
        MRVScopeSummary,
        AGENT_SCOPE_MAP,
        AGENT_DESCRIPTIONS,
    )
    _loaded_integrations.append("MRVBridge")
except ImportError as e:
    logger.debug("Integration 6 (MRVBridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 7: Data Bridge
# ---------------------------------------------------------------------------
try:
    from .data_bridge import (
        DataBridge,
        GHGDataSource,
        DataFormat,
        DataAgentTarget,
        QualityLevel,
        DataRouteConfig,
        DataRequest,
        DataResponse,
        QualityReport,
        LineageRecord,
    )
    _loaded_integrations.append("DataBridge")
except ImportError as e:
    logger.debug("Integration 7 (DataBridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 8: Foundation Bridge
# ---------------------------------------------------------------------------
try:
    from .foundation_bridge import (
        FoundationBridge,
        FoundationConfig,
        ValidationResult as FoundationValidationResult,
        NormalizationResult,
        AssumptionRecord,
        CitationRecord,
        TelemetryEvent,
        ENERGY_CONVERSIONS,
    )
    _loaded_integrations.append("FoundationBridge")
except ImportError as e:
    logger.debug("Integration 8 (FoundationBridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 9: ERP Connector
# ---------------------------------------------------------------------------
try:
    from .erp_connector import (
        ERPConnector,
        ERPConnectorConfig,
        ERPSystemType,
        ConnectionStatus,
        ExtractionStatus,
        DateRange,
        ActivityRecord,
        ExtractionResult,
    )
    _loaded_integrations.append("ERPConnector")
except ImportError as e:
    logger.debug("Integration 9 (ERPConnector) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 10: Notification Bridge
# ---------------------------------------------------------------------------
try:
    from .notification_bridge import (
        NotificationBridge,
        NotificationConfig,
        NotificationChannel,
        NotificationType,
        NotificationPriority,
        NotificationStatus,
        Notification,
        SendResult as NotificationSendResult,
    )
    _loaded_integrations.append("NotificationBridge")
except ImportError as e:
    logger.debug("Integration 10 (NotificationBridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 11: Health Check
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
    logger.debug("Integration 11 (HealthCheck) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 12: Setup Wizard
# ---------------------------------------------------------------------------
try:
    from .setup_wizard import (
        SetupWizard,
        SetupStep,
        StepStatus,
        WizardState,
        PackConfig,
    )
    _loaded_integrations.append("SetupWizard")
except ImportError as e:
    logger.debug("Integration 12 (SetupWizard) not available: %s", e)


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
    "BaseYearManagementOrchestrator",
    "PipelinePhase",
    "ExecutionStatus",
    "PipelineConfig",
    "PhaseResult",
    "PipelineResult",
    "PipelineStatus",
    "PHASE_DEPENDENCIES",
    "PARALLEL_PHASE_GROUPS",
    # --- Integration 2: PACK-041 Bridge ---
    "Pack041Bridge",
    "Pack041Config",
    "Scope1Summary",
    "Scope2Summary",
    "ImportResult",
    # --- Integration 3: PACK-042 Bridge ---
    "Pack042Bridge",
    "Pack042Config",
    "Scope3CategoryResult",
    "Scope3ImportResult",
    # --- Integration 4: PACK-043 Bridge ---
    "Pack043Bridge",
    "Pack043Config",
    "Scope3FullResult",
    "Scope3CompleteImportResult",
    # --- Integration 5: PACK-044 Bridge ---
    "Pack044Bridge",
    "Pack044Config",
    "ChangeType",
    "ChangeEvent",
    "VersionInfo",
    "Pack044ImportResult",
    # --- Integration 6: MRV Bridge ---
    "MRVBridge",
    "MRVBridgeConfig",
    "MRVScope",
    "MRVAgentResult",
    "MRVScopeSummary",
    "AGENT_SCOPE_MAP",
    "AGENT_DESCRIPTIONS",
    # --- Integration 7: Data Bridge ---
    "DataBridge",
    "GHGDataSource",
    "DataFormat",
    "DataAgentTarget",
    "QualityLevel",
    "DataRouteConfig",
    "DataRequest",
    "DataResponse",
    "QualityReport",
    "LineageRecord",
    # --- Integration 8: Foundation Bridge ---
    "FoundationBridge",
    "FoundationConfig",
    "FoundationValidationResult",
    "NormalizationResult",
    "AssumptionRecord",
    "CitationRecord",
    "TelemetryEvent",
    "ENERGY_CONVERSIONS",
    # --- Integration 9: ERP Connector ---
    "ERPConnector",
    "ERPConnectorConfig",
    "ERPSystemType",
    "ConnectionStatus",
    "ExtractionStatus",
    "DateRange",
    "ActivityRecord",
    "ExtractionResult",
    # --- Integration 10: Notification Bridge ---
    "NotificationBridge",
    "NotificationConfig",
    "NotificationChannel",
    "NotificationType",
    "NotificationPriority",
    "NotificationStatus",
    "Notification",
    "NotificationSendResult",
    # --- Integration 11: Health Check ---
    "HealthCheck",
    "HealthCheckConfig",
    "HealthCheckCategory",
    "HealthStatus",
    "HealthSeverity",
    "CheckType",
    "ComponentHealth",
    "SystemHealth",
    # --- Integration 12: Setup Wizard ---
    "SetupWizard",
    "SetupStep",
    "StepStatus",
    "WizardState",
    "PackConfig",
]


def get_loaded_integrations() -> list[str]:
    """Return list of integration class names that loaded successfully."""
    return list(_loaded_integrations)


logger.info(
    "PACK-045 Base Year Management integrations: %d/%d loaded",
    len(_loaded_integrations),
    __integrations_count__,
)
