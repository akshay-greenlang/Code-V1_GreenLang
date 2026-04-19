# -*- coding: utf-8 -*-
"""
PACK-044 GHG Inventory Management Pack - Integration Layer
=============================================================

Phase 4 integration layer for the GHG Inventory Management Pack that
provides 12-phase DAG pipeline orchestration, PACK-041/042/043 import
bridges, all 30 MRV agent routing (MRV-001 through MRV-030), DATA agent
data ingestion bridges, Foundation agent routing, ERP system connectivity,
multi-channel notification, 20-category health verification, 10-step
setup wizard, and multi-channel alert management.

Components:
    - PackOrchestrator: 12-phase DAG pipeline with parallel execution,
      SHA-256 provenance chain, retry with exponential backoff, and
      progress tracking
    - Pack041Bridge: Imports Scope 1-2 data from PACK-041 (Scope 1-2
      Complete Pack) including stationary, mobile, process, fugitive
      emissions and dual-reporting Scope 2
    - Pack042Bridge: Imports Scope 3 starter data from PACK-042 (Scope 3
      Starter Pack) covering 8 screening categories
    - Pack043Bridge: Imports full Scope 3 data from PACK-043 (Scope 3
      Complete Pack) covering all 15 categories with SBTi alignment
    - MRVBridge: Routes to all 30 MRV agents (MRV-001 through MRV-030)
      with scope-based grouping and verification support
    - DataBridge: Routes data intake to DATA agents for PDF extraction,
      CSV/Excel normalization, ERP/API ingestion, quality profiling,
      outlier detection, gap filling, and lineage tracking
    - FoundationBridge: Routes to Foundation agents for DAG orchestration,
      schema validation, unit normalization, assumption registration,
      citation management, access control, and telemetry
    - ERPConnector: Connects to SAP/Oracle/Dynamics/Generic ERP for
      activity data extraction across all organizational entities
    - NotificationBridge: Multi-channel notification delivery (email,
      Slack, Teams, webhook) for review requests, data reminders,
      quality alerts, and status updates
    - HealthCheck: 20-category system health verification across engines,
      workflows, templates, bridges, and infrastructure
    - SetupWizard: 10-step GHG inventory configuration wizard
    - AlertBridge: Multi-channel alerting for deadlines, anomalies,
      quality degradation, version conflicts, and consolidation errors

Architecture:
    External Systems --> Integration Bridges --> Platform Components
                              |
                              v
    PACK-044 Inventory Mgmt <-- Composition <-- Zero Duplication
                              |
                              v
    MRV Agents (001-030) <-- DATA Agents (001-020) <-- FOUND Agents (001-010)

Platform Integrations:
    - greenlang/agents/mrv/* (MRV-001 through MRV-030)
    - greenlang/agents/data/* (DATA-001, 002, 003, 004, 010, 013, 014, 018)
    - greenlang/agents/foundation/* (FOUND-001 through FOUND-010)
    - packs/ghg-accounting/PACK-041 (Scope 1-2 Complete)
    - packs/ghg-accounting/PACK-042 (Scope 3 Starter)
    - packs/ghg-accounting/PACK-043 (Scope 3 Complete)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-044 GHG Inventory Management
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack_id__: str = "PACK-044"
__pack_name__: str = "GHG Inventory Management Pack"
__integrations_count__: int = 12

_loaded_integrations: list[str] = []

# ---------------------------------------------------------------------------
# Integration 1: Pack Orchestrator
# ---------------------------------------------------------------------------
try:
    from .pack_orchestrator import (
        InventoryManagementOrchestrator,
        PipelinePhase,
        ExecutionStatus,
        PipelineConfig,
        PhaseResult,
        PipelineResult,
        PipelineStatus,
        PHASE_DEPENDENCIES,
        PARALLEL_PHASE_GROUPS,
    )
    _loaded_integrations.append("InventoryManagementOrchestrator")
except ImportError as e:
    logger.debug("Integration 1 (InventoryManagementOrchestrator) not available: %s", e)

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
# Integration 5: MRV Bridge (All 30 Agents)
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
    logger.debug("Integration 5 (MRVBridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 6: Data Bridge
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
    logger.debug("Integration 6 (DataBridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 7: Foundation Bridge
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
    logger.debug("Integration 7 (FoundationBridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 8: ERP Connector
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
    logger.debug("Integration 8 (ERPConnector) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 9: Notification Bridge
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
    logger.debug("Integration 9 (NotificationBridge) not available: %s", e)

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
        PackConfig,
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
        AlertType,
        AlertSeverity,
        AlertChannel,
        AlertStatus,
        AlertConfig,
        Alert,
        ScheduledAlert,
        SendResult,
        FRAMEWORK_DEADLINES,
        REVIEW_CYCLE_DEFAULTS,
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
    "InventoryManagementOrchestrator",
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
    # --- Integration 5: MRV Bridge ---
    "MRVBridge",
    "MRVBridgeConfig",
    "MRVScope",
    "MRVAgentResult",
    "MRVScopeSummary",
    "AGENT_SCOPE_MAP",
    "AGENT_DESCRIPTIONS",
    # --- Integration 6: Data Bridge ---
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
    # --- Integration 7: Foundation Bridge ---
    "FoundationBridge",
    "FoundationConfig",
    "FoundationValidationResult",
    "NormalizationResult",
    "AssumptionRecord",
    "CitationRecord",
    "TelemetryEvent",
    "ENERGY_CONVERSIONS",
    # --- Integration 8: ERP Connector ---
    "ERPConnector",
    "ERPConnectorConfig",
    "ERPSystemType",
    "ConnectionStatus",
    "ExtractionStatus",
    "DateRange",
    "ActivityRecord",
    "ExtractionResult",
    # --- Integration 9: Notification Bridge ---
    "NotificationBridge",
    "NotificationConfig",
    "NotificationChannel",
    "NotificationType",
    "NotificationPriority",
    "NotificationStatus",
    "Notification",
    "NotificationSendResult",
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
    "PackConfig",
    # --- Integration 12: Alert Bridge ---
    "AlertBridge",
    "AlertType",
    "AlertSeverity",
    "AlertChannel",
    "AlertStatus",
    "AlertConfig",
    "Alert",
    "ScheduledAlert",
    "SendResult",
    "FRAMEWORK_DEADLINES",
    "REVIEW_CYCLE_DEFAULTS",
]


def get_loaded_integrations() -> list[str]:
    """Return list of integration class names that loaded successfully."""
    return list(_loaded_integrations)


logger.info(
    "PACK-044 GHG Inventory Management integrations: %d/%d loaded",
    len(_loaded_integrations),
    __integrations_count__,
)
