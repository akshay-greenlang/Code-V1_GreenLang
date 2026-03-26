# -*- coding: utf-8 -*-
"""
PACK-043 Scope 3 Complete Pack - Integration Layer
========================================================

Phase 4 integration layer for the Scope 3 Complete Pack that provides
12-phase enterprise DAG pipeline orchestration, PACK-042 (Scope 3 Starter)
bridge dependency, PACK-041 (Scope 1-2 Complete) optional bridge, LCA
database connectors (ecoinvent/GaBi), SBTi target validation and sector
pathway data, TCFD scenario and climate risk reference data, supplier
portal programme management, cloud provider carbon data integration,
advanced ERP deep integration (SAP/Oracle/Dynamics), 24-category health
verification, 10-step enterprise setup wizard, and multi-channel
enterprise alert management.

Components:
    - Scope3CompleteOrchestrator: 12-phase enterprise DAG pipeline with
      parallel execution for independent phases, SHA-256 provenance chain,
      retry with exponential backoff, checkpoint/resume capability
    - Pack042Bridge: REQUIRED dependency bridge to PACK-042 Scope 3
      Starter for screening, category results, inventory, hotspots,
      supplier engagement, data quality, uncertainty, and compliance
    - Pack041Bridge: Optional bridge to PACK-041 Scope 1-2 Complete for
      full footprint calculation and boundary alignment
    - LCADatabaseBridge: LCA database connectors for ecoinvent 3.10 and
      GaBi process/material emission factor lookups
    - SBTiBridge: SBTi validation, SDA sector pathways, FLAG guidance,
      and target submission data generation
    - TCFDBridge: TCFD scenario data including carbon price forecasts,
      IEA NZE pathways, and NGFS scenario data
    - SupplierPortalBridge: Supplier data request distribution, response
      collection, quality gates, and commitment synchronization
    - CloudCarbonBridge: AWS/Azure/GCP carbon data integration and
      on-premise datacenter estimation
    - ERPDeepBridge: Advanced ERP integration for SAP MM/SRM, Oracle
      Procurement Cloud, and Dynamics 365 Supply Chain with GL mapping
    - HealthCheck: 24-category enterprise system health verification
    - SetupWizard: 10-step enterprise configuration wizard
    - AlertBridge: Enterprise multi-channel alerting for SBTi milestones,
      supplier programme deadlines, base year recalculation triggers,
      climate risk thresholds, and assurance readiness

Architecture:
    External Systems --> Integration Bridges --> Platform Components
                              |
                              v
    PACK-043 Scope 3 <-- Composition <-- Zero Duplication
                              |
                              v
    PACK-042 (Scope 3 Starter) <-- PACK-041 (Scope 1-2) <-- LCA DBs
                              |
                              v
    SBTi / TCFD / Suppliers <-- Cloud Carbon <-- ERP Deep

Platform Integrations:
    - packs/ghg-accounting/PACK-042-scope-3-starter (REQUIRED)
    - packs/ghg-accounting/PACK-041-scope-1-2-complete (optional)
    - greenlang/agents/mrv/* (MRV-014 through MRV-030)
    - greenlang/agents/data/* (DATA agents)
    - greenlang/agents/foundation/* (FOUND-001 through FOUND-010)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-043 Scope 3 Complete
Status: Production Ready
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack_id__: str = "PACK-043"
__pack_name__: str = "Scope 3 Complete Pack"
__integrations_count__: int = 12

_MODULE_VERSION: str = "43.0.0"

_loaded_integrations: list[str] = []

# ---------------------------------------------------------------------------
# Integration 1: Pack Orchestrator
# ---------------------------------------------------------------------------
try:
    from .pack_orchestrator import (
        Scope3CompleteOrchestrator,
        PipelinePhase,
        ExecutionStatus,
        MaturityLevel,
        AssuranceLevel,
        BoundaryApproach,
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
    _loaded_integrations.append("Scope3CompleteOrchestrator")
except ImportError as e:
    logger.debug("Integration 1 (Scope3CompleteOrchestrator) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 2: PACK-042 Bridge (REQUIRED)
# ---------------------------------------------------------------------------
try:
    from .pack042_bridge import (
        Pack042Bridge,
        Pack042Status,
        ScreeningData,
        CategoryResults,
        ConsolidatedInventory,
        HotspotAnalysis,
        SupplierEngagement,
        DataQualityResult,
        UncertaintyResult,
        ComplianceAssessment,
    )
    _loaded_integrations.append("Pack042Bridge")
except ImportError as e:
    logger.debug("Integration 2 (Pack042Bridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 3: PACK-041 Bridge (Optional)
# ---------------------------------------------------------------------------
try:
    from .pack041_bridge import (
        Pack041Bridge,
        Pack041Status,
        Scope12Totals,
        FullFootprintResult,
        BoundaryAlignmentResult,
    )
    _loaded_integrations.append("Pack041Bridge")
except ImportError as e:
    logger.debug("Integration 3 (Pack041Bridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 4: LCA Database Bridge
# ---------------------------------------------------------------------------
try:
    from .lca_database_bridge import (
        LCADatabaseBridge,
        LCADatabase,
        LCAProcess,
        MaterialFactor,
        ProcessSearchResult,
        BOMLookupResult,
        ECOINVENT_PROCESSES,
        MATERIAL_EMISSION_FACTORS,
    )
    _loaded_integrations.append("LCADatabaseBridge")
except ImportError as e:
    logger.debug("Integration 4 (LCADatabaseBridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 5: SBTi Bridge
# ---------------------------------------------------------------------------
try:
    from .sbti_bridge import (
        SBTiBridge,
        SBTiScenario,
        SBTiStatus,
        SectorPathway,
        TargetValidation,
        FLAGPathway,
        SubmissionData,
        SDA_SECTOR_PATHWAYS,
        FLAG_SECTOR_TARGETS,
    )
    _loaded_integrations.append("SBTiBridge")
except ImportError as e:
    logger.debug("Integration 5 (SBTiBridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 6: TCFD Bridge
# ---------------------------------------------------------------------------
try:
    from .tcfd_bridge import (
        TCFDBridge,
        TCFDScenario,
        CarbonPriceForecast,
        IEANZEPathway,
        NGFSScenarioData,
        PhysicalHazardData,
        CARBON_PRICE_PROJECTIONS,
        IEA_NZE_PATHWAYS,
    )
    _loaded_integrations.append("TCFDBridge")
except ImportError as e:
    logger.debug("Integration 6 (TCFDBridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 7: Supplier Portal Bridge
# ---------------------------------------------------------------------------
try:
    from .supplier_portal_bridge import (
        SupplierPortalBridge,
        RequestStatus,
        QualityGateResult,
        DataRequest,
        SupplierResponse,
        SupplierCommitment,
        ProgrammeMetrics,
    )
    _loaded_integrations.append("SupplierPortalBridge")
except ImportError as e:
    logger.debug("Integration 7 (SupplierPortalBridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 8: Cloud Carbon Bridge
# ---------------------------------------------------------------------------
try:
    from .cloud_carbon_bridge import (
        CloudCarbonBridge,
        CloudProvider,
        CloudCarbonResult,
        DataCenterProfile,
        CLOUD_PROVIDER_DEFAULTS,
    )
    _loaded_integrations.append("CloudCarbonBridge")
except ImportError as e:
    logger.debug("Integration 8 (CloudCarbonBridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 9: ERP Deep Bridge
# ---------------------------------------------------------------------------
try:
    from .erp_deep_bridge import (
        ERPDeepBridge,
        ERPSystemType,
        ExtractionStatus,
        ProcurementRecord,
        TravelExpenseRecord,
        VendorMapping,
        ExtractionResult,
        GL_SCOPE3_MAP_SAP,
        GL_SCOPE3_MAP_ORACLE,
        GL_SCOPE3_MAP_DYNAMICS,
    )
    _loaded_integrations.append("ERPDeepBridge")
except ImportError as e:
    logger.debug("Integration 9 (ERPDeepBridge) not available: %s", e)

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
        MaturityLevel as WizardMaturityLevel,
        AssuranceTarget,
        WizardState,
        Scope3CompleteConfig,
        SCOPE3_CATEGORIES,
        BOUNDARY_APPROACH_DEFAULTS,
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
        SBTI_MILESTONE_THRESHOLDS,
        ASSURANCE_MILESTONES,
    )
    _loaded_integrations.append("AlertBridge")
except ImportError as e:
    logger.debug("Integration 12 (AlertBridge) not available: %s", e)


# ---------------------------------------------------------------------------
# Integration Registry
# ---------------------------------------------------------------------------


INTEGRATION_CATALOG: dict[str, dict[str, str]] = {
    "Scope3CompleteOrchestrator": {
        "module": "pack_orchestrator",
        "description": "12-phase enterprise DAG pipeline orchestrator for Scope 3 Complete",
        "pack": "PACK-043",
    },
    "Pack042Bridge": {
        "module": "pack042_bridge",
        "description": "REQUIRED bridge to PACK-042 Scope 3 Starter for inventory data",
        "pack": "PACK-043",
    },
    "Pack041Bridge": {
        "module": "pack041_bridge",
        "description": "Optional bridge to PACK-041 for Scope 1-2 full footprint",
        "pack": "PACK-043",
    },
    "LCADatabaseBridge": {
        "module": "lca_database_bridge",
        "description": "LCA database connectors for ecoinvent/GaBi emission factors",
        "pack": "PACK-043",
    },
    "SBTiBridge": {
        "module": "sbti_bridge",
        "description": "SBTi validation, SDA pathways, FLAG guidance, submission data",
        "pack": "PACK-043",
    },
    "TCFDBridge": {
        "module": "tcfd_bridge",
        "description": "TCFD scenario data, carbon prices, IEA NZE, NGFS scenarios",
        "pack": "PACK-043",
    },
    "SupplierPortalBridge": {
        "module": "supplier_portal_bridge",
        "description": "Supplier data requests, response collection, programme management",
        "pack": "PACK-043",
    },
    "CloudCarbonBridge": {
        "module": "cloud_carbon_bridge",
        "description": "AWS/Azure/GCP carbon data and on-premise estimation",
        "pack": "PACK-043",
    },
    "ERPDeepBridge": {
        "module": "erp_deep_bridge",
        "description": "Advanced ERP integration for SAP/Oracle/Dynamics procurement",
        "pack": "PACK-043",
    },
    "HealthCheck": {
        "module": "health_check",
        "description": "24-category enterprise system health verification",
        "pack": "PACK-043",
    },
    "SetupWizard": {
        "module": "setup_wizard",
        "description": "10-step enterprise Scope 3 configuration wizard",
        "pack": "PACK-043",
    },
    "AlertBridge": {
        "module": "alert_bridge",
        "description": "Enterprise multi-channel alerting for SBTi, suppliers, assurance",
        "pack": "PACK-043",
    },
}


class IntegrationRegistry:
    """Registry for querying available PACK-043 integrations.

    Provides methods to list available integrations, check specific
    integration availability, and get integration metadata.

    Example:
        >>> registry = IntegrationRegistry()
        >>> registry.is_available("Pack042Bridge")
        True
        >>> registry.get_loaded()
        ['Scope3CompleteOrchestrator', 'Pack042Bridge', ...]
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
    "_MODULE_VERSION",
    "INTEGRATION_CATALOG",
    "IntegrationRegistry",
    # --- Integration 1: Pack Orchestrator ---
    "Scope3CompleteOrchestrator",
    "PipelinePhase",
    "ExecutionStatus",
    "MaturityLevel",
    "AssuranceLevel",
    "BoundaryApproach",
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
    # --- Integration 2: PACK-042 Bridge ---
    "Pack042Bridge",
    "Pack042Status",
    "ScreeningData",
    "CategoryResults",
    "ConsolidatedInventory",
    "HotspotAnalysis",
    "SupplierEngagement",
    "DataQualityResult",
    "UncertaintyResult",
    "ComplianceAssessment",
    # --- Integration 3: PACK-041 Bridge ---
    "Pack041Bridge",
    "Pack041Status",
    "Scope12Totals",
    "FullFootprintResult",
    "BoundaryAlignmentResult",
    # --- Integration 4: LCA Database Bridge ---
    "LCADatabaseBridge",
    "LCADatabase",
    "LCAProcess",
    "MaterialFactor",
    "ProcessSearchResult",
    "BOMLookupResult",
    "ECOINVENT_PROCESSES",
    "MATERIAL_EMISSION_FACTORS",
    # --- Integration 5: SBTi Bridge ---
    "SBTiBridge",
    "SBTiScenario",
    "SBTiStatus",
    "SectorPathway",
    "TargetValidation",
    "FLAGPathway",
    "SubmissionData",
    "SDA_SECTOR_PATHWAYS",
    "FLAG_SECTOR_TARGETS",
    # --- Integration 6: TCFD Bridge ---
    "TCFDBridge",
    "TCFDScenario",
    "CarbonPriceForecast",
    "IEANZEPathway",
    "NGFSScenarioData",
    "PhysicalHazardData",
    "CARBON_PRICE_PROJECTIONS",
    "IEA_NZE_PATHWAYS",
    # --- Integration 7: Supplier Portal Bridge ---
    "SupplierPortalBridge",
    "RequestStatus",
    "QualityGateResult",
    "DataRequest",
    "SupplierResponse",
    "SupplierCommitment",
    "ProgrammeMetrics",
    # --- Integration 8: Cloud Carbon Bridge ---
    "CloudCarbonBridge",
    "CloudProvider",
    "CloudCarbonResult",
    "DataCenterProfile",
    "CLOUD_PROVIDER_DEFAULTS",
    # --- Integration 9: ERP Deep Bridge ---
    "ERPDeepBridge",
    "ERPSystemType",
    "ExtractionStatus",
    "ProcurementRecord",
    "TravelExpenseRecord",
    "VendorMapping",
    "ExtractionResult",
    "GL_SCOPE3_MAP_SAP",
    "GL_SCOPE3_MAP_ORACLE",
    "GL_SCOPE3_MAP_DYNAMICS",
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
    "WizardMaturityLevel",
    "AssuranceTarget",
    "WizardState",
    "Scope3CompleteConfig",
    "SCOPE3_CATEGORIES",
    "BOUNDARY_APPROACH_DEFAULTS",
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
    "SBTI_MILESTONE_THRESHOLDS",
    "ASSURANCE_MILESTONES",
]


def get_loaded_integrations() -> list[str]:
    """Return list of integration class names that loaded successfully."""
    return list(_loaded_integrations)


logger.info(
    "PACK-043 Scope 3 Complete integrations: %d/%d loaded",
    len(_loaded_integrations),
    __integrations_count__,
)
