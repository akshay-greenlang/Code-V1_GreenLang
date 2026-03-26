# -*- coding: utf-8 -*-
"""
PACK-041 Scope 1-2 Complete Pack - Integration Layer
========================================================

Phase 4 integration layer for the Scope 1-2 Complete Pack that provides
12-phase DAG pipeline orchestration, Scope 1 MRV agent routing (MRV-001
through MRV-008), Scope 2 dual-reporting agent routing (MRV-009 through
MRV-013), DATA agent data ingestion bridges, Foundation agent routing,
Energy Efficiency pack imports (PACK-031 to PACK-040), Net Zero pack
integration (PACK-021 to PACK-030), ERP system connectivity, utility
bill/meter data import, 22-category health verification, 8-step setup
wizard, and multi-channel alert management.

Components:
    - PackOrchestrator: 12-phase DAG pipeline with parallel Scope 1
      execution (phases 3-6), SHA-256 provenance chain, retry with
      exponential backoff, and progress tracking
    - MRVScope1Bridge: Routes to all 8 Scope 1 MRV agents with
      deterministic emission calculations and GWP-weighted totals
    - MRVScope2Bridge: Routes to all 5 Scope 2 MRV agents with
      dual-reporting (location + market-based) and reconciliation
    - DataBridge: Routes data intake to DATA agents for PDF extraction,
      CSV/Excel normalization, ERP/API ingestion, quality profiling,
      outlier detection, gap filling, and lineage tracking
    - FoundationBridge: Routes to Foundation agents for DAG orchestration,
      schema validation, unit normalization, assumption registration,
      citation management, access control, and telemetry
    - EnergyEfficiencyBridge: Imports findings/assessments/quick wins
      from PACK-031 to PACK-035, links measures to emission reductions
    - NetZeroBridge: Provides baseline to PACK-021 to PACK-030, imports
      targets, tracks progress, assesses SBTi alignment
    - ERPConnector: Connects to SAP/Oracle/Dynamics/Generic ERP for
      fuel, fleet, electricity, refrigerant, production data
    - UtilityDataBridge: Imports utility bills and meter data, handles
      estimated bills, normalizes units, aggregates by facility
    - HealthCheck: 22-category system health verification
    - SetupWizard: 8-step GHG inventory configuration wizard
    - AlertBridge: Multi-channel alerting for deadlines, anomalies,
      EF updates, base year triggers, compliance, verification

Architecture:
    External Systems --> Integration Bridges --> Platform Components
                              |
                              v
    PACK-041 Scope 1-2 <-- Composition <-- Zero Duplication
                              |
                              v
    MRV Agents (001-013) <-- DATA Agents (001-018) <-- FOUND Agents (001-010)

Platform Integrations:
    - greenlang/agents/mrv/* (MRV-001 through MRV-013)
    - greenlang/agents/data/* (DATA-001, 002, 003, 004, 010, 013, 014, 018)
    - greenlang/agents/foundation/* (FOUND-001 through FOUND-010)
    - packs/energy-efficiency/PACK-031 through PACK-040
    - packs/net-zero/PACK-021 through PACK-030

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-041 Scope 1-2 Complete
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack_id__: str = "PACK-041"
__pack_name__: str = "Scope 1-2 Complete Pack"
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
        ConsolidationApproach,
        ComplianceFramework,
        ReportFormat,
        PhaseConfig,
        RetryConfig,
        PipelineConfig,
        PhaseResult,
        PipelineResult,
        PipelineStatus,
        PHASE_DEPENDENCIES,
        PHASE_EXECUTION_ORDER,
        PARALLEL_PHASE_GROUPS,
    )
    _loaded_integrations.append("PackOrchestrator")
except ImportError as e:
    logger.debug("Integration 1 (PackOrchestrator) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 2: MRV Scope 1 Bridge
# ---------------------------------------------------------------------------
try:
    from .mrv_scope1_bridge import (
        MRVScope1Bridge,
        Scope1Category,
        Scope1AgentConfig,
        AgentResult,
        AgentStatus,
        FuelType,
        RefrigerantType,
        AGENT_CATEGORY_MAP,
        FUEL_EMISSION_FACTORS,
        GWP_AR5,
    )
    _loaded_integrations.append("MRVScope1Bridge")
except ImportError as e:
    logger.debug("Integration 2 (MRVScope1Bridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 3: MRV Scope 2 Bridge
# ---------------------------------------------------------------------------
try:
    from .mrv_scope2_bridge import (
        MRVScope2Bridge,
        Scope2Method,
        EnergyType,
        MarketInstrument,
        ReconciliationStatus,
        Scope2AgentConfig,
        Scope2Result,
        Scope2DualResult,
        GRID_EMISSION_FACTORS,
        RESIDUAL_MIX_FACTORS,
        STEAM_EMISSION_FACTORS,
        COOLING_EMISSION_FACTORS,
    )
    _loaded_integrations.append("MRVScope2Bridge")
except ImportError as e:
    logger.debug("Integration 3 (MRVScope2Bridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 4: Data Bridge
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
    logger.debug("Integration 4 (DataBridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 5: Foundation Bridge
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
    logger.debug("Integration 5 (FoundationBridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 6: Energy Efficiency Bridge
# ---------------------------------------------------------------------------
try:
    from .energy_efficiency_bridge import (
        EnergyEfficiencyBridge,
        EfficiencyPackSource,
        MeasureCategory,
        EmissionScope,
        AuditFinding,
        BuildingAssessmentData,
        QuickWinMeasure,
        ISO50001Data,
        BenchmarkResult,
        EmissionReductionLink,
    )
    _loaded_integrations.append("EnergyEfficiencyBridge")
except ImportError as e:
    logger.debug("Integration 6 (EnergyEfficiencyBridge) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 7: Net Zero Bridge
# ---------------------------------------------------------------------------
try:
    from .net_zero_bridge import (
        NetZeroBridge,
        TargetType,
        TargetScope,
        SBTiPathway,
        ProgressStatus,
        BaselineData,
        EmissionTarget,
        ProgressReport,
        SBTiAlignment,
        SBTI_ANNUAL_REDUCTION_RATES,
    )
    _loaded_integrations.append("NetZeroBridge")
except ImportError as e:
    logger.debug("Integration 7 (NetZeroBridge) not available: %s", e)

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
        FuelPurchase,
        FleetRecord,
        ElectricityRecord,
        RefrigerantPurchase,
        ProductionRecord,
        ExtractionResult,
    )
    _loaded_integrations.append("ERPConnector")
except ImportError as e:
    logger.debug("Integration 8 (ERPConnector) not available: %s", e)

# ---------------------------------------------------------------------------
# Integration 9: Utility Data Bridge
# ---------------------------------------------------------------------------
try:
    from .utility_data_bridge import (
        UtilityDataBridge,
        UtilityType,
        BillStatus,
        MeterInterval,
        BillFormat,
        UtilityBill,
        MeterReading,
        ConsumptionSummary,
    )
    _loaded_integrations.append("UtilityDataBridge")
except ImportError as e:
    logger.debug("Integration 9 (UtilityDataBridge) not available: %s", e)

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
    "ConsolidationApproach",
    "ComplianceFramework",
    "ReportFormat",
    "PhaseConfig",
    "RetryConfig",
    "PipelineConfig",
    "PhaseResult",
    "PipelineResult",
    "PipelineStatus",
    "PHASE_DEPENDENCIES",
    "PHASE_EXECUTION_ORDER",
    "PARALLEL_PHASE_GROUPS",
    # --- Integration 2: MRV Scope 1 Bridge ---
    "MRVScope1Bridge",
    "Scope1Category",
    "Scope1AgentConfig",
    "AgentResult",
    "AgentStatus",
    "FuelType",
    "RefrigerantType",
    "AGENT_CATEGORY_MAP",
    "FUEL_EMISSION_FACTORS",
    "GWP_AR5",
    # --- Integration 3: MRV Scope 2 Bridge ---
    "MRVScope2Bridge",
    "Scope2Method",
    "EnergyType",
    "MarketInstrument",
    "ReconciliationStatus",
    "Scope2AgentConfig",
    "Scope2Result",
    "Scope2DualResult",
    "GRID_EMISSION_FACTORS",
    "RESIDUAL_MIX_FACTORS",
    "STEAM_EMISSION_FACTORS",
    "COOLING_EMISSION_FACTORS",
    # --- Integration 4: Data Bridge ---
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
    # --- Integration 5: Foundation Bridge ---
    "FoundationBridge",
    "FoundationConfig",
    "FoundationValidationResult",
    "NormalizationResult",
    "AssumptionRecord",
    "CitationRecord",
    "TelemetryEvent",
    "ENERGY_CONVERSIONS",
    # --- Integration 6: Energy Efficiency Bridge ---
    "EnergyEfficiencyBridge",
    "EfficiencyPackSource",
    "MeasureCategory",
    "EmissionScope",
    "AuditFinding",
    "BuildingAssessmentData",
    "QuickWinMeasure",
    "ISO50001Data",
    "BenchmarkResult",
    "EmissionReductionLink",
    # --- Integration 7: Net Zero Bridge ---
    "NetZeroBridge",
    "TargetType",
    "TargetScope",
    "SBTiPathway",
    "ProgressStatus",
    "BaselineData",
    "EmissionTarget",
    "ProgressReport",
    "SBTiAlignment",
    "SBTI_ANNUAL_REDUCTION_RATES",
    # --- Integration 8: ERP Connector ---
    "ERPConnector",
    "ERPConnectorConfig",
    "ERPSystemType",
    "ConnectionStatus",
    "ExtractionStatus",
    "DateRange",
    "FuelPurchase",
    "FleetRecord",
    "ElectricityRecord",
    "RefrigerantPurchase",
    "ProductionRecord",
    "ExtractionResult",
    # --- Integration 9: Utility Data Bridge ---
    "UtilityDataBridge",
    "UtilityType",
    "BillStatus",
    "MeterInterval",
    "BillFormat",
    "UtilityBill",
    "MeterReading",
    "ConsumptionSummary",
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
]


def get_loaded_integrations() -> list[str]:
    """Return list of integration class names that loaded successfully."""
    return list(_loaded_integrations)


logger.info(
    "PACK-041 Scope 1-2 Complete integrations: %d/%d loaded",
    len(_loaded_integrations),
    __integrations_count__,
)
