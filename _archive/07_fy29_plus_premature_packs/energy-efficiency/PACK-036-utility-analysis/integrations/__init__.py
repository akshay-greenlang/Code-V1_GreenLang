# -*- coding: utf-8 -*-
"""
PACK-036 Utility Analysis Pack - Integration Layer
============================================================

Phase 4 integration layer for the Utility Analysis Pack that provides
utility analysis pipeline orchestration, MRV emissions bridging, DATA agent
routing, PACK-031/032/033 data import, utility provider API connectivity,
weather data ingestion, market data feeds, multi-category health
verification, guided setup wizard, and alert management.

Components:
    - PackOrchestrator: 10-phase pipeline with DAG dependency
      resolution, parallel execution, retry with exponential backoff,
      and SHA-256 provenance tracking
    - MRVBridge: Routes utility consumption data to MRV agents
      (Stationary Combustion, Scope 2 Location/Market-Based) and
      converts energy consumption to emissions (tCO2e)
    - DataBridge: Routes data intake to DATA agents for
      utility bill parsing, meter data, quality profiling, and
      validation rule enforcement
    - Pack031Bridge: Imports energy audit results, equipment efficiency
      data, and energy baselines from PACK-031
    - Pack032Bridge: Imports building assessment results, zone data,
      and envelope performance from PACK-032
    - Pack033Bridge: Imports quick-win measures, savings estimates,
      and implementation status from PACK-033
    - UtilityProviderBridge: Connects to utility provider APIs
      (Green Button, ESPI, EDI) for automated bill retrieval,
      interval data download, and account management
    - WeatherBridge: HDD/CDD calculation, TMY data, climate zone
      determination, and weather-normalized consumption
    - MarketDataBridge: Energy market price feeds, wholesale rate
      tracking, futures pricing, and procurement intelligence data
    - HealthCheck: 15-category system health verification
    - SetupWizard: 8-step guided facility and utility account
      configuration with 8 facility presets
    - AlertBridge: Multi-channel notification and alert management
      for billing anomalies, budget variances, and demand events

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-036 Utility Analysis
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack_id__: str = "PACK-036"
__pack_name__: str = "Utility Analysis Pack"
__integrations_count__: int = 12

_loaded_integrations: list[str] = []

# ---------------------------------------------------------------------------
# Pack Orchestrator
# ---------------------------------------------------------------------------
try:
    from .pack_orchestrator import (
        ExecutionStatus,
        FacilityType,
        OrchestratorConfig,
        OrchestratorPhase,
        PARALLEL_PHASE_GROUPS,
        PHASE_DEPENDENCIES,
        PHASE_EXECUTION_ORDER,
        PhaseProvenance,
        PhaseResult,
        PipelineResult,
        PackOrchestrator,
        RetryConfig,
    )
    _loaded_integrations.append("PackOrchestrator")
except ImportError as e:
    logger.debug("PackOrchestrator not available: %s", e)

# ---------------------------------------------------------------------------
# MRV Bridge
# ---------------------------------------------------------------------------
try:
    from .mrv_bridge import (
        DEFAULT_EMISSION_FACTORS,
        ConsumptionCategory,
        MRVBridge,
        MRVRouteConfig,
        MRVScope,
        RoutingResult,
        ConsumptionToEmissionsMapping,
    )
    _loaded_integrations.append("MRVBridge")
except ImportError as e:
    logger.debug("MRVBridge not available: %s", e)

# ---------------------------------------------------------------------------
# Data Bridge
# ---------------------------------------------------------------------------
try:
    from .data_bridge import (
        DataQualityCheck,
        DataBridge,
        DataRouteConfig,
        DataRoutingResult,
        UtilityDataSource,
    )
    _loaded_integrations.append("DataBridge")
except ImportError as e:
    logger.debug("DataBridge not available: %s", e)

# ---------------------------------------------------------------------------
# PACK-031 Bridge
# ---------------------------------------------------------------------------
try:
    from .pack031_bridge import (
        AuditDataImport,
        AuditImportConfig,
        Pack031Bridge,
    )
    _loaded_integrations.append("Pack031Bridge")
except ImportError as e:
    logger.debug("Pack031Bridge not available: %s", e)

# ---------------------------------------------------------------------------
# PACK-032 Bridge
# ---------------------------------------------------------------------------
try:
    from .pack032_bridge import (
        AssessmentImportConfig,
        BuildingDataImport,
        Pack032Bridge,
    )
    _loaded_integrations.append("Pack032Bridge")
except ImportError as e:
    logger.debug("Pack032Bridge not available: %s", e)

# ---------------------------------------------------------------------------
# PACK-033 Bridge
# ---------------------------------------------------------------------------
try:
    from .pack033_bridge import (
        QuickWinsDataImport,
        QuickWinsImportConfig,
        Pack033Bridge,
    )
    _loaded_integrations.append("Pack033Bridge")
except ImportError as e:
    logger.debug("Pack033Bridge not available: %s", e)

# ---------------------------------------------------------------------------
# Utility Provider Bridge
# ---------------------------------------------------------------------------
try:
    from .utility_provider_bridge import (
        AccountInfo,
        BillRetrievalResult,
        IntervalDataResult,
        ProviderAPIConfig,
        ProviderProtocol,
        UtilityProviderBridge,
    )
    _loaded_integrations.append("UtilityProviderBridge")
except ImportError as e:
    logger.debug("UtilityProviderBridge not available: %s", e)

# ---------------------------------------------------------------------------
# Weather Bridge
# ---------------------------------------------------------------------------
try:
    from .weather_bridge import (
        ClimateNormalization,
        DegreeDayData,
        WeatherBridge,
        WeatherConfig,
    )
    _loaded_integrations.append("WeatherBridge")
except ImportError as e:
    logger.debug("WeatherBridge not available: %s", e)

# ---------------------------------------------------------------------------
# Market Data Bridge
# ---------------------------------------------------------------------------
try:
    from .market_data_bridge import (
        MarketDataBridge,
        MarketDataConfig,
        MarketPriceRecord,
        PriceForecast,
        WholesaleMarket,
    )
    _loaded_integrations.append("MarketDataBridge")
except ImportError as e:
    logger.debug("MarketDataBridge not available: %s", e)

# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------
try:
    from .health_check import (
        CheckCategory,
        ComponentHealth,
        HealthCheck,
        HealthCheckConfig,
        HealthCheckResult,
        HealthSeverity,
        HealthStatus,
    )
    _loaded_integrations.append("HealthCheck")
except ImportError as e:
    logger.debug("HealthCheck not available: %s", e)

# ---------------------------------------------------------------------------
# Setup Wizard
# ---------------------------------------------------------------------------
try:
    from .setup_wizard import (
        FacilitySetup,
        PresetConfig,
        SetupResult,
        SetupWizard,
        SetupWizardStep,
        StepStatus,
        WizardState,
        WizardStepState,
    )
    _loaded_integrations.append("SetupWizard")
except ImportError as e:
    logger.debug("SetupWizard not available: %s", e)

# ---------------------------------------------------------------------------
# Alert Bridge
# ---------------------------------------------------------------------------
try:
    from .alert_bridge import (
        Alert,
        AlertBridge,
        AlertChannel,
        AlertConfig,
        AlertRule,
        AlertSeverity,
        AlertType,
        NotificationResult,
    )
    _loaded_integrations.append("AlertBridge")
except ImportError as e:
    logger.debug("AlertBridge not available: %s", e)


# ===================================================================
# Public API
# ===================================================================

__all__ = [
    # Module metadata
    "__version__",
    "__pack_id__",
    "__pack_name__",
    "__integrations_count__",
    # --- Pack Orchestrator ---
    "PackOrchestrator",
    "OrchestratorConfig",
    "RetryConfig",
    "OrchestratorPhase",
    "FacilityType",
    "ExecutionStatus",
    "PhaseProvenance",
    "PhaseResult",
    "PipelineResult",
    "PHASE_DEPENDENCIES",
    "PHASE_EXECUTION_ORDER",
    "PARALLEL_PHASE_GROUPS",
    # --- MRV Bridge ---
    "MRVBridge",
    "MRVRouteConfig",
    "ConsumptionCategory",
    "MRVScope",
    "RoutingResult",
    "ConsumptionToEmissionsMapping",
    "DEFAULT_EMISSION_FACTORS",
    # --- Data Bridge ---
    "DataBridge",
    "DataRouteConfig",
    "DataRoutingResult",
    "DataQualityCheck",
    "UtilityDataSource",
    # --- PACK-031 Bridge ---
    "Pack031Bridge",
    "AuditImportConfig",
    "AuditDataImport",
    # --- PACK-032 Bridge ---
    "Pack032Bridge",
    "AssessmentImportConfig",
    "BuildingDataImport",
    # --- PACK-033 Bridge ---
    "Pack033Bridge",
    "QuickWinsImportConfig",
    "QuickWinsDataImport",
    # --- Utility Provider Bridge ---
    "UtilityProviderBridge",
    "ProviderAPIConfig",
    "ProviderProtocol",
    "AccountInfo",
    "BillRetrievalResult",
    "IntervalDataResult",
    # --- Weather Bridge ---
    "WeatherBridge",
    "WeatherConfig",
    "DegreeDayData",
    "ClimateNormalization",
    # --- Market Data Bridge ---
    "MarketDataBridge",
    "MarketDataConfig",
    "MarketPriceRecord",
    "PriceForecast",
    "WholesaleMarket",
    # --- Health Check ---
    "HealthCheck",
    "HealthCheckConfig",
    "HealthCheckResult",
    "ComponentHealth",
    "HealthSeverity",
    "HealthStatus",
    "CheckCategory",
    # --- Setup Wizard ---
    "SetupWizard",
    "SetupWizardStep",
    "StepStatus",
    "FacilitySetup",
    "PresetConfig",
    "WizardStepState",
    "WizardState",
    "SetupResult",
    # --- Alert Bridge ---
    "AlertBridge",
    "AlertConfig",
    "Alert",
    "AlertRule",
    "AlertSeverity",
    "AlertChannel",
    "AlertType",
    "NotificationResult",
]


def get_loaded_integrations() -> list[str]:
    """Return list of integration class names that loaded successfully."""
    return list(_loaded_integrations)


def get_integration_count() -> int:
    """Return count of integrations that loaded successfully."""
    return len(_loaded_integrations)


logger.info(
    "PACK-036 Utility Analysis integrations: %d/%d loaded",
    len(_loaded_integrations),
    __integrations_count__,
)
