# -*- coding: utf-8 -*-
"""
PACK-042 Scope 3 Starter Pack - Integration Layer
========================================================

Phase 4 integration layer for the Scope 3 Starter Pack that provides
12-phase DAG pipeline orchestration, Scope 3 MRV agent routing (MRV-014
through MRV-028), cross-cutting agent routing (MRV-029 Category Mapper,
MRV-030 Audit Trail), DATA agent data ingestion bridges, Foundation
agent routing, PACK-041 Scope 1-2 integration, EEIO emission factor
database, ERP system connectivity, 22-category health verification,
8-step setup wizard, and multi-channel alert management.

Components:
    - Scope3PackOrchestrator: 12-phase DAG pipeline with parallel
      execution for independent phases, SHA-256 provenance chain,
      retry with exponential backoff, checkpoint/resume capability
    - MRVScope3Bridge: Routes to all 15 Scope 3 MRV agents (MRV-014
      through MRV-028) with spend-based, average-data, and supplier-
      specific calculation methodologies
    - CategoryMapperBridge: Routes to MRV-029 for NAICS/ISIC/UNSPSC/HS
      code classification into Scope 3 categories
    - AuditTrailBridge: Routes to MRV-030 for audit event logging,
      lineage DAG retrieval, evidence packaging, and hash chain
      verification
    - DataBridge: Routes data intake to DATA agents for PDF extraction,
      CSV/Excel normalization, ERP/API ingestion, spend categorization,
      quality profiling, and lineage tracking
    - FoundationBridge: Routes to Foundation agents for DAG orchestration,
      schema validation, unit normalization, assumption registration,
      citation management, access control, and telemetry
    - Scope12Bridge: Integrates with PACK-041 for full footprint
      calculation, Scope 3 share analysis, boundary alignment, and
      Category 3 fuel & energy reconciliation
    - EEIOFactorBridge: USEEIO 2.0 / Exiobase 3 emission factor
      database with currency conversion and CPI inflation adjustment
    - ERPConnector: Connects to SAP/Oracle/Dynamics/NetSuite for
      procurement, travel, vendor, and GL account data
    - HealthCheck: 22-category system health verification
    - SetupWizard: 8-step Scope 3 inventory configuration wizard
    - AlertBridge: Multi-channel alerting for data collection, supplier
      response, compliance deadlines, DQR warnings, and hotspot alerts

Architecture:
    External Systems --> Integration Bridges --> Platform Components
                              |
                              v
    PACK-042 Scope 3 <-- Composition <-- Zero Duplication
                              |
                              v
    MRV Agents (014-030) <-- DATA Agents (001-018) <-- FOUND Agents (001-010)
                              |
                              v
    PACK-041 (Scope 1-2) <-- EEIO Factors <-- ERP Systems

Platform Integrations:
    - greenlang/agents/mrv/* (MRV-014 through MRV-030)
    - greenlang/agents/data/* (DATA-001, 002, 003, 004, 009, 010, 018)
    - greenlang/agents/foundation/* (FOUND-001 through FOUND-010)
    - packs/ghg-accounting/PACK-041-scope-1-2-complete

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-042 Scope 3 Starter
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack_id__: str = "PACK-042"
__pack_name__: str = "Scope 3 Starter Pack"
__integrations_count__: int = 12

_loaded_integrations: list[str] = []

# ---------------------------------------------------------------------------
# Integration 1: Pack Orchestrator
# ---------------------------------------------------------------------------
try:
    from .pack_orchestrator import (
        Scope3PackOrchestrator,
        PipelinePhase,
        ExecutionStatus,
        Scope3Methodology,
        ComplianceFramework,
        ReportFormat,
        PhaseConfig,
        RetryConfig,
        PipelineConfig,
        PhaseResult,
        PipelineResult,
        PipelineStatus,
        CheckpointData,
        PHASE_DEPENDENCIES,
        PHASE_EXECUTION_ORDER,
        PARALLEL_PHASE_GROUPS,
    )
    _loaded_integrations.append("Scope3PackOrchestrator")
except ImportError as e:
    logger.debug("Integration 1 (Scope3PackOrchestrator) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 2: MRV Scope 3 Bridge
# ---------------------------------------------------------------------------
try:
    from .mrv_scope3_bridge import (
        MRVScope3Bridge,
        Scope3Category,
        AgentStatus,
        CalculationMethodology,
        Scope3AgentConfig,
        CategoryResult,
        ConsolidatedScope3Result,
        AGENT_CATEGORY_MAP,
        PREFERRED_METHODOLOGIES,
    )
    _loaded_integrations.append("MRVScope3Bridge")
except ImportError as e:
    logger.debug("Integration 2 (MRVScope3Bridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 3: Category Mapper Bridge
# ---------------------------------------------------------------------------
try:
    from .category_mapper_bridge import (
        CategoryMapperBridge,
        ClassificationSource,
        ConfidenceLevel,
        ClassificationInput,
        ClassificationResult,
        BatchClassificationResult,
        NAICS_TO_SCOPE3,
    )
    _loaded_integrations.append("CategoryMapperBridge")
except ImportError as e:
    logger.debug("Integration 3 (CategoryMapperBridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 4: Audit Trail Bridge
# ---------------------------------------------------------------------------
try:
    from .audit_trail_bridge import (
        AuditTrailBridge,
        AuditEventType,
        IntegrityStatus,
        AuditRecord,
        LineageNode,
        LineageEdge,
        LineageDAG,
        EvidenceBundle,
        HashChainVerification,
    )
    _loaded_integrations.append("AuditTrailBridge")
except ImportError as e:
    logger.debug("Integration 4 (AuditTrailBridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 5: Data Bridge
# ---------------------------------------------------------------------------
try:
    from .data_bridge import (
        DataBridge,
        Scope3DataSource,
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
    logger.debug("Integration 5 (DataBridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 6: Foundation Bridge
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
    logger.debug("Integration 6 (FoundationBridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 7: Scope 1-2 Bridge
# ---------------------------------------------------------------------------
try:
    from .scope12_bridge import (
        Scope12Bridge,
        BoundaryApproach,
        AlignmentStatus,
        Scope12Totals,
        FullFootprint,
        BoundaryAlignment,
        Cat3Alignment,
    )
    _loaded_integrations.append("Scope12Bridge")
except ImportError as e:
    logger.debug("Integration 7 (Scope12Bridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 8: EEIO Factor Bridge
# ---------------------------------------------------------------------------
try:
    from .eeio_factor_bridge import (
        EEIOFactorBridge,
        EEIOModel,
        CurrencyCode,
        EEIOFactorResult,
        USEEIO_FACTORS,
        EXCHANGE_RATES_TO_USD,
        CPI_DEFLATORS,
    )
    _loaded_integrations.append("EEIOFactorBridge")
except ImportError as e:
    logger.debug("Integration 8 (EEIOFactorBridge) not available: %s", e)

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
        ProcurementTransaction,
        TravelExpense,
        VendorRecord,
        ExtractionResult,
        GL_TO_SCOPE3_MAP,
    )
    _loaded_integrations.append("ERPConnector")
except ImportError as e:
    logger.debug("Integration 9 (ERPConnector) not available: %s", e)

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
        MethodologyTier,
        EngagementLevel,
        WizardState,
        Scope3PackConfig,
        SCOPE3_CATEGORIES,
        SECTOR_CATEGORY_DEFAULTS,
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
        DQR_THRESHOLDS,
    )
    _loaded_integrations.append("AlertBridge")
except ImportError as e:
    logger.debug("Integration 12 (AlertBridge) not available: %s", e)


# ---------------------------------------------------------------------------
# Integration Registry
# ---------------------------------------------------------------------------


INTEGRATION_CATALOG: dict[str, dict[str, str]] = {
    "Scope3PackOrchestrator": {
        "module": "pack_orchestrator",
        "description": "12-phase DAG pipeline orchestrator for Scope 3 inventory",
        "pack": "PACK-042",
    },
    "MRVScope3Bridge": {
        "module": "mrv_scope3_bridge",
        "description": "Routes to 15 Scope 3 MRV agents (MRV-014 to MRV-028)",
        "pack": "PACK-042",
    },
    "CategoryMapperBridge": {
        "module": "category_mapper_bridge",
        "description": "NAICS/ISIC/UNSPSC/HS code classification via MRV-029",
        "pack": "PACK-042",
    },
    "AuditTrailBridge": {
        "module": "audit_trail_bridge",
        "description": "Audit event logging and lineage tracking via MRV-030",
        "pack": "PACK-042",
    },
    "DataBridge": {
        "module": "data_bridge",
        "description": "Data intake routing to DATA agents for Scope 3 data",
        "pack": "PACK-042",
    },
    "FoundationBridge": {
        "module": "foundation_bridge",
        "description": "Foundation agent routing for platform services",
        "pack": "PACK-042",
    },
    "Scope12Bridge": {
        "module": "scope12_bridge",
        "description": "Integration with PACK-041 for full footprint calculation",
        "pack": "PACK-042",
    },
    "EEIOFactorBridge": {
        "module": "eeio_factor_bridge",
        "description": "USEEIO 2.0 / Exiobase 3 emission factor database",
        "pack": "PACK-042",
    },
    "ERPConnector": {
        "module": "erp_connector",
        "description": "ERP system connectivity for procurement and travel data",
        "pack": "PACK-042",
    },
    "HealthCheck": {
        "module": "health_check",
        "description": "22-category system health verification",
        "pack": "PACK-042",
    },
    "SetupWizard": {
        "module": "setup_wizard",
        "description": "8-step Scope 3 inventory configuration wizard",
        "pack": "PACK-042",
    },
    "AlertBridge": {
        "module": "alert_bridge",
        "description": "Multi-channel alerting for Scope 3 inventory management",
        "pack": "PACK-042",
    },
}


class IntegrationRegistry:
    """Registry for querying available PACK-042 integrations.

    Provides methods to list available integrations, check specific
    integration availability, and get integration metadata.

    Example:
        >>> registry = IntegrationRegistry()
        >>> registry.is_available("MRVScope3Bridge")
        True
        >>> registry.get_loaded()
        ['Scope3PackOrchestrator', 'MRVScope3Bridge', ...]
    """

    def get_loaded(self) -> list[str]:
        """Return list of integration class names that loaded successfully."""
        return list(_loaded_integrations)

    def is_available(self, integration_name: str) -> bool:
        """Check if a specific integration is available.

        Args:
            integration_name: Class name of the integration.

        Returns:
            True if the integration loaded successfully.
        """
        return integration_name in _loaded_integrations

    def get_catalog(self) -> dict[str, dict[str, str]]:
        """Get the full integration catalog.

        Returns:
            Dict mapping integration names to their metadata.
        """
        return dict(INTEGRATION_CATALOG)

    def get_info(self, integration_name: str) -> dict[str, str]:
        """Get info for a specific integration.

        Args:
            integration_name: Class name of the integration.

        Returns:
            Dict with module, description, and pack fields.
        """
        return INTEGRATION_CATALOG.get(integration_name, {})

    def get_load_summary(self) -> dict[str, Any]:
        """Get summary of integration loading status.

        Returns:
            Dict with loaded count, total, and missing list.
        """
        all_names = list(INTEGRATION_CATALOG.keys())
        missing = [n for n in all_names if n not in _loaded_integrations]
        return {
            "loaded": len(_loaded_integrations),
            "total": len(INTEGRATION_CATALOG),
            "missing": missing,
            "all_loaded": len(missing) == 0,
        }


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__: list[str] = [
    # Module metadata
    "__version__",
    "__pack_id__",
    "__pack_name__",
    "__integrations_count__",
    "INTEGRATION_CATALOG",
    "IntegrationRegistry",
    # --- Integration 1: Pack Orchestrator ---
    "Scope3PackOrchestrator",
    "PipelinePhase",
    "ExecutionStatus",
    "Scope3Methodology",
    "ComplianceFramework",
    "ReportFormat",
    "PhaseConfig",
    "RetryConfig",
    "PipelineConfig",
    "PhaseResult",
    "PipelineResult",
    "PipelineStatus",
    "CheckpointData",
    "PHASE_DEPENDENCIES",
    "PHASE_EXECUTION_ORDER",
    "PARALLEL_PHASE_GROUPS",
    # --- Integration 2: MRV Scope 3 Bridge ---
    "MRVScope3Bridge",
    "Scope3Category",
    "AgentStatus",
    "CalculationMethodology",
    "Scope3AgentConfig",
    "CategoryResult",
    "ConsolidatedScope3Result",
    "AGENT_CATEGORY_MAP",
    "PREFERRED_METHODOLOGIES",
    # --- Integration 3: Category Mapper Bridge ---
    "CategoryMapperBridge",
    "ClassificationSource",
    "ConfidenceLevel",
    "ClassificationInput",
    "ClassificationResult",
    "BatchClassificationResult",
    "NAICS_TO_SCOPE3",
    # --- Integration 4: Audit Trail Bridge ---
    "AuditTrailBridge",
    "AuditEventType",
    "IntegrityStatus",
    "AuditRecord",
    "LineageNode",
    "LineageEdge",
    "LineageDAG",
    "EvidenceBundle",
    "HashChainVerification",
    # --- Integration 5: Data Bridge ---
    "DataBridge",
    "Scope3DataSource",
    "DataFormat",
    "DataAgentTarget",
    "QualityLevel",
    "DataRouteConfig",
    "DataRequest",
    "DataResponse",
    "QualityReport",
    "LineageRecord",
    # --- Integration 6: Foundation Bridge ---
    "FoundationBridge",
    "FoundationConfig",
    "FoundationValidationResult",
    "NormalizationResult",
    "AssumptionRecord",
    "CitationRecord",
    "TelemetryEvent",
    "ENERGY_CONVERSIONS",
    # --- Integration 7: Scope 1-2 Bridge ---
    "Scope12Bridge",
    "BoundaryApproach",
    "AlignmentStatus",
    "Scope12Totals",
    "FullFootprint",
    "BoundaryAlignment",
    "Cat3Alignment",
    # --- Integration 8: EEIO Factor Bridge ---
    "EEIOFactorBridge",
    "EEIOModel",
    "CurrencyCode",
    "EEIOFactorResult",
    "USEEIO_FACTORS",
    "EXCHANGE_RATES_TO_USD",
    "CPI_DEFLATORS",
    # --- Integration 9: ERP Connector ---
    "ERPConnector",
    "ERPConnectorConfig",
    "ERPSystemType",
    "ConnectionStatus",
    "ExtractionStatus",
    "DateRange",
    "ProcurementTransaction",
    "TravelExpense",
    "VendorRecord",
    "ExtractionResult",
    "GL_TO_SCOPE3_MAP",
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
    "MethodologyTier",
    "EngagementLevel",
    "WizardState",
    "Scope3PackConfig",
    "SCOPE3_CATEGORIES",
    "SECTOR_CATEGORY_DEFAULTS",
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
    "DQR_THRESHOLDS",
]


def get_loaded_integrations() -> list[str]:
    """Return list of integration class names that loaded successfully."""
    return list(_loaded_integrations)


logger.info(
    "PACK-042 Scope 3 Starter integrations: %d/%d loaded",
    len(_loaded_integrations),
    __integrations_count__,
)
